"""Focused regression tests for incremental company-paper runs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from paper_pipeline.cli.e2e import _absorb, _arxiv_month_prefixes, app
from paper_pipeline.cli.sync import enrich_from_arxiv
from paper_pipeline.core.date_window import DateWindow, validate_openalex_save_mode
from paper_pipeline.core.organization_scope import (
    BROAD_ACADEMIC_IMPORT_LABELS,
    PURE_ACADEMIC_HF_NAMESPACES,
    PURE_ACADEMIC_ORG_LABELS,
    is_pure_academic_hf_namespace,
    is_pure_academic_org,
)
from paper_pipeline.ingest.hf_daily_papers import ArxivMeta


def _metadata(*, title: str, published: str) -> ArxivMeta:
    return ArxivMeta(
        title=title,
        abstract="",
        authors=[],
        affiliations={},
        categories=[],
        primary_category=None,
        comment=None,
        published=published,
        journal_ref=None,
        doi=None,
    )


class _Cursor:
    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, _query: str, _params: object = None) -> None:
        return None

    def fetchall(self) -> list[tuple[str]]:
        return []


class _Connection:
    def __enter__(self) -> _Connection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self) -> _Cursor:
        return _Cursor()


class _Batch:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, object]]] = []

    def __enter__(self) -> _Batch:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def save_paper(self, arxiv_id: str, **kwargs: object) -> None:
        self.saved.append((arxiv_id, kwargs))


class _Database:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict[str, object]]] = []

    def get_conn(self) -> _Connection:
        return _Connection()

    def batch(self) -> _Batch:
        batch = _Batch()
        batch.saved = self.saved
        return batch


class DeltaWindowTests(TestCase):
    def test_four_elapsed_days_means_thursday_through_monday(self) -> None:
        window = DateWindow.from_inputs(days=4, today=date(2026, 7, 27))

        self.assertEqual(window.since, date(2026, 7, 23))
        self.assertEqual(window.through, date(2026, 7, 27))

    def test_arxiv_month_prefilter_spans_exact_window(self) -> None:
        prefixes = _arxiv_month_prefixes(
            DateWindow(
                since=date(2025, 12, 28),
                through=date(2026, 2, 3),
            )
        )

        self.assertEqual(prefixes, ("2512.", "2601.", "2602."))

    def test_paper_e2e_dry_run_surfaces_exact_window(self) -> None:
        result = CliRunner().invoke(
            app,
            [
                "--days",
                "4",
                "--through",
                "2026-07-27",
                "--dry-run",
                "--no-summarize",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("2026-07-23..2026-07-27", result.output)
        self.assertIn("--no-save-neon", result.output)
        self.assertIn("authoritative arXiv date at absorb", result.output)
        self.assertIn("arxiv metadata for this delta only", result.output)

    def test_paper_e2e_exposes_explicit_force_web_flag(self) -> None:
        result = CliRunner().invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("--force-web", result.output)

    def test_absorb_rejects_browser_ids_outside_exact_window(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "browser.jsonl"
            rows = [
                {"arxiv_id": "2607.00001", "org_label": "Current Lab"},
                {"arxiv_id": "2401.00001", "org_label": "Historical Lab"},
            ]
            path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            db = _Database()
            metadata = {
                "2607.00001": _metadata(
                    title="Current",
                    published="2026-07-25T00:00:00Z",
                ),
            }

            with patch(
                "paper_pipeline.cli.e2e.fetch_arxiv_metadata",
                return_value=metadata,
            ):
                result = _absorb(
                    db,
                    path,
                    source="test",
                    publication_window=DateWindow(
                        since=date(2026, 7, 23),
                        through=date(2026, 7, 27),
                    ),
                )

        self.assertEqual(result.ids, {"2607.00001"})
        self.assertEqual(result.added, 1)
        self.assertEqual(result.rejected_outside_window, 1)
        self.assertEqual([row[0] for row in db.saved], ["2607.00001"])

    def test_absorb_rejects_pure_academic_sources(self) -> None:
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "browser.jsonl"
            rows = [
                {"arxiv_id": "2607.00001", "org_label": "Berkeley"},
                {"arxiv_id": "2607.00002", "org_label": "DeepMind"},
            ]
            path.write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
                encoding="utf-8",
            )
            db = _Database()
            metadata = {
                "2607.00002": _metadata(
                    title="Industry paper",
                    published="2026-07-25T00:00:00Z",
                ),
            }

            with patch(
                "paper_pipeline.cli.e2e.fetch_arxiv_metadata",
                return_value=metadata,
            ):
                result = _absorb(
                    db,
                    path,
                    source="test",
                    publication_window=DateWindow(
                        since=date(2026, 7, 23),
                        through=date(2026, 7, 27),
                    ),
                )

        self.assertEqual(result.ids, {"2607.00002"})
        self.assertEqual(result.added, 1)
        self.assertEqual(result.rejected_academic_source, 1)
        self.assertEqual([row[0] for row in db.saved], ["2607.00002"])

    def test_company_scope_excludes_every_broad_university_feed(self) -> None:
        self.assertTrue(BROAD_ACADEMIC_IMPORT_LABELS <= PURE_ACADEMIC_ORG_LABELS)
        self.assertTrue(all(is_pure_academic_org(label) for label in BROAD_ACADEMIC_IMPORT_LABELS))
        self.assertFalse(is_pure_academic_org("DeepMind"))
        self.assertTrue(all(is_pure_academic_hf_namespace(slug) for slug in PURE_ACADEMIC_HF_NAMESPACES))
        self.assertFalse(is_pure_academic_hf_namespace("zai-org"))

    def test_openalex_exact_window_cannot_bypass_authoritative_arxiv_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot save directly"):
            validate_openalex_save_mode(save_neon=True, exact_window=True)

        validate_openalex_save_mode(save_neon=False, exact_window=True)
        validate_openalex_save_mode(save_neon=True, exact_window=False)

    def test_scoped_enrichment_parameterizes_percent_pattern(self) -> None:
        db = MagicMock()
        connection = db.get_conn.return_value.__enter__.return_value
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        result = enrich_from_arxiv(db, ids={"2607.00001"})

        self.assertEqual(result, 0)
        cursor.execute.assert_called_once()
        query, params = cursor.execute.call_args.args
        self.assertIn("id NOT LIKE %s", query)
        self.assertIn("id = ANY(%s)", query)
        self.assertEqual(params, ("ext:%", ["2607.00001"]))
