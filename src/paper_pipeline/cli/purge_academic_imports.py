"""Back up and remove broad pure-university OpenAlex imports from Neon."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from psycopg.rows import dict_row
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from paper_pipeline.core.neon_db import TABLE, NeonDB
from paper_pipeline.core.organization_scope import BROAD_ACADEMIC_IMPORT_LABELS

console = Console()
app = typer.Typer(add_completion=False)


class PurgeSummary(BaseModel):
    """Auditable outcome of one academic-import cleanup."""

    matched: int
    deleted: int
    backup_path: Path | None = None
    organizations: dict[str, int]


def _default_backup_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("local_data") / "backups" / f"pure_academic_openalex_{timestamp}.ndjson"


def _fetch_candidates(db: NeonDB) -> list[dict[str, Any]]:
    labels = sorted(BROAD_ACADEMIC_IMPORT_LABELS)
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT *
            FROM {TABLE}
            WHERE score_source = %s
              AND organization = ANY(%s)
            ORDER BY id
            """,
            ("openalex_audit", labels),
        )
        return list(cur.fetchall())


def _write_backup(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _delete_candidates(db: NeonDB) -> int:
    labels = sorted(BROAD_ACADEMIC_IMPORT_LABELS)
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            DELETE FROM {TABLE}
            WHERE score_source = %s
              AND organization = ANY(%s)
            RETURNING id
            """,
            ("openalex_audit", labels),
        )
        return len(cur.fetchall())


def _organization_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("organization") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _render(summary: PurgeSummary, *, apply: bool) -> None:
    table = Table(title="Pure-academic OpenAlex cleanup")
    table.add_column("organization")
    table.add_column("rows", justify="right")
    for label, count in summary.organizations.items():
        table.add_row(label, str(count))
    table.add_section()
    table.add_row("total", str(summary.matched))
    console.print(table)
    if apply:
        console.print(f"[green]Deleted {summary.deleted} rows.[/]")
        console.print(f"Recovery backup: [bold]{summary.backup_path}[/]")
    else:
        console.print("[yellow]Dry run only; pass --apply to back up and delete this exact cohort.[/]")


@app.command()
def main(
    apply: Annotated[
        bool,
        typer.Option("--apply", help="Write a full NDJSON backup, then delete the matched rows."),
    ] = False,
    backup_path: Annotated[
        Path | None,
        typer.Option(help="Recovery backup path; defaults under local_data/backups/."),
    ] = None,
) -> None:
    """Remove only broad university rows whose provenance is OpenAlex."""
    db = NeonDB()
    rows = _fetch_candidates(db)
    organizations = _organization_counts(rows)
    resolved_backup = backup_path or _default_backup_path()
    deleted = 0
    if apply and rows:
        _write_backup(rows, resolved_backup)
        deleted = _delete_candidates(db)
        if deleted != len(rows):
            raise RuntimeError(f"delete count changed after backup: backed up {len(rows)}, deleted {deleted}")
    summary = PurgeSummary(
        matched=len(rows),
        deleted=deleted,
        backup_path=resolved_backup if apply and rows else None,
        organizations=organizations,
    )
    _render(summary, apply=apply)


if __name__ == "__main__":
    app()
