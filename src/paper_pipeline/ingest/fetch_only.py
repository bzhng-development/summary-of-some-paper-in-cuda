#!/usr/bin/env python3
"""fetch_only.py — Fetch HF Daily Papers for a date range and merge them into
``all_scored.json`` WITHOUT running the LLM scorer.

Companion to ``hf_daily_papers.py`` for when scoring is unavailable (e.g. the
scoring backend is down) but you still want fresh papers to show up in the
viewer. New papers are written with ``score=0``, ``reason=""`` so that:
  * the SvelteKit viewer renders them (default ``minScore`` is 0), and
  * a later ``hf_daily_papers.py`` run still re-scores them (its cache-skip
    check requires a non-empty ``reason``).

Existing entries are preserved untouched; only papers whose ``arxiv_id`` is not
already present are appended.

Usage:
    uv run python daily_papers/fetch_only.py --out-dir ./src/paper_pipeline/ingest/papers_out
    uv run python daily_papers/fetch_only.py --from 2026-05-28 --to 2026-06-05
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from loguru import logger

from paper_pipeline.ingest.hf_daily_papers import Paper, fetch_papers_range


def _unscored_to_dict(p: Paper) -> dict:
    """Mirror ``hf_daily_papers._scored_to_dict`` but with no LLM score."""
    return {
        "arxiv_id": p.arxiv_id,
        "title": p.title,
        "score": 0,
        "similar_paper": "NONE",
        "reason": "",
        "upvotes": p.upvotes,
        "github": p.github_repo,
        "github_stars": p.github_stars,
        "keywords": p.ai_keywords,
        "authors": p.authors,
        "affiliations": p.affiliations,
        "organization": p.organization,
        "org_fullname": p.org_fullname,
        "categories": p.categories,
        "primary_category": p.primary_category,
        "arxiv_comment": p.arxiv_comment,
        "published": p.published,
        "journal_ref": p.journal_ref,
        "doi": p.doi,
        "summary": p.summary,
    }


def _parse_day(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC).date()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch HF Daily Papers (no scoring) and merge into all_scored.json")
    parser.add_argument("--out-dir", type=Path, default=Path("./src/paper_pipeline/ingest/papers_out"))
    today = datetime.now(tz=UTC).date()
    parser.add_argument(
        "--from",
        dest="start",
        type=_parse_day,
        default=today - timedelta(days=10),
        help="Start date YYYY-MM-DD (default: 10 days ago).",
    )
    parser.add_argument(
        "--to",
        dest="end",
        type=_parse_day,
        default=today,
        help="End date YYYY-MM-DD (default: today).",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    save_path = out_dir / "all_scored.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    existing_ids: set[str] = set()
    if save_path.exists():
        existing = json.loads(save_path.read_text())
        existing_ids = {e["arxiv_id"] for e in existing if e.get("arxiv_id")}
    logger.info("Loaded {} existing papers from {}", len(existing), save_path)

    logger.info("Fetching HF daily papers {} -> {} (no scoring)", args.start, args.end)
    fetched = fetch_papers_range(args.start, args.end)

    new_records = [_unscored_to_dict(p) for p in fetched if p.arxiv_id not in existing_ids]
    logger.info("Fetched {} papers; {} are new (not already in all_scored.json)", len(fetched), len(new_records))

    if not new_records:
        logger.info("Nothing new to add; leaving {} unchanged", save_path)
        return

    merged = existing + new_records
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
    try:
        with open(fd, "w") as f:
            json.dump(merged, f, indent=2)
        Path(tmp).replace(save_path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

    logger.success("Added {} new papers -> {} ({} total)", len(new_records), save_path, len(merged))


if __name__ == "__main__":
    main()
