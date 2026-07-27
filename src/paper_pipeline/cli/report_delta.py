"""Export an auditable report of newly created company-paper rows."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from psycopg.rows import dict_row
from pydantic import BaseModel
from rich.console import Console

from paper_pipeline.core.date_window import DateWindow
from paper_pipeline.core.neon_db import TABLE, NeonDB

console = Console()
app = typer.Typer(add_completion=False)

COMPANY_DELTA_SOURCES = ("openalex_audit", "e2e_web", "e2e_meta")


class DeltaPaper(BaseModel):
    """One paper created by a company-paper delta run."""

    id: str
    title: str
    organization: str | None
    published: str
    score_source: str
    created_at: datetime
    url: str


def _load_rows(
    db: NeonDB,
    *,
    created_after: datetime,
    publication_window: DateWindow,
) -> list[DeltaPaper]:
    with db.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT id, title, organization, published, score_source, created_at, url
            FROM {TABLE}
            WHERE created_at > %s
              AND score_source = ANY(%s)
            ORDER BY published, organization, title, id
            """,
            (created_after, list(COMPANY_DELTA_SOURCES)),
        )
        rows = cur.fetchall()
    return [DeltaPaper.model_validate(row) for row in rows if publication_window.contains(row.get("published"))]


def _write_ndjson(rows: list[DeltaPaper], path: Path) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(rows: list[DeltaPaper], path: Path) -> None:
    fieldnames = list(DeltaPaper.model_fields)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _write_markdown(rows: list[DeltaPaper], path: Path, publication_window: DateWindow) -> None:
    lines = [
        "# New company papers",
        "",
        f"- Publication window: {publication_window.since} through {publication_window.through}, inclusive",
        f"- Net-new papers: {len(rows)}",
        "- Scope: company and industrial-research sources; pure university feeds excluded",
        "",
    ]
    current_date = ""
    for row in rows:
        published_date = row.published[:10]
        if published_date != current_date:
            current_date = published_date
            lines.extend((f"## {current_date}", ""))
        org = row.organization or "Unknown organization"
        lines.append(f"- **{row.title}** — {org} — [{row.id}]({row.url})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.command()
def main(
    created_after: Annotated[
        str,
        typer.Option(help="Exclusive database creation timestamp for the pre-run snapshot."),
    ],
    since: Annotated[str, typer.Option(help="Inclusive publication start date (YYYY-MM-DD).")],
    through: Annotated[str, typer.Option(help="Inclusive publication end date (YYYY-MM-DD).")],
    output_dir: Annotated[Path, typer.Option(help="Directory for CSV, NDJSON, and Markdown reports.")],
) -> None:
    """Export exact net-new rows for a completed company-paper catch-up."""
    publication_window = DateWindow.from_inputs(since=since, through=through)
    try:
        creation_boundary = datetime.fromisoformat(created_after)
    except ValueError as error:
        raise typer.BadParameter("created-after must be an ISO 8601 timestamp") from error
    rows = _load_rows(
        NeonDB(),
        created_after=creation_boundary,
        publication_window=publication_window,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = output_dir / "new_company_papers.ndjson"
    csv_path = output_dir / "new_company_papers.csv"
    markdown_path = output_dir / "new_company_papers.md"
    _write_ndjson(rows, ndjson_path)
    _write_csv(rows, csv_path)
    _write_markdown(rows, markdown_path, publication_window)
    console.print(f"[green]Exported {len(rows)} papers to {output_dir}[/]")


if __name__ == "__main__":
    app()
