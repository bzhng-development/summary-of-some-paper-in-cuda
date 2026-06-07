"""Stage 3: ingest OCR'd markdown bodies back into Neon.

Reads the append-only ``ocr_out.jsonl`` produced by Stage 2 (``chandra_ocr.py``) and
upserts each paper's markdown into the ``markdown`` column on ``nextjs-ui_paper``
(added idempotently via ``NeonDB.init_schema``). The jsonl shard stays on disk as the
git-trackable source of truth, per the repo's NDJSON-over-SQLite convention.

The reader is source-agnostic: any jsonl whose rows carry ``arxiv_id`` + ``markdown``
works, so a future arXiv-HTML-scrape stage (for the ``html_only`` bucket) can feed the
same ingest. ``status`` and ``markdown`` length are used only to skip empties.

Usage::

    uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl ocr_out.jsonl
    uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl ocr_out.jsonl --dry-run
    uv run python -m paper_pipeline.ocr.ingest_ocr add-column   # just ensure the schema
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from paper_pipeline.core.neon_db import NeonDB

console = Console()

_DEFAULT_STATUSES = ("success", "partial_success")


class IncomingRecord(BaseModel):
    """Lenient view of one OCR output row — extra fields are ignored."""

    model_config = {"extra": "ignore"}

    arxiv_id: str
    markdown: str = ""
    status: str = "success"


def _read_records(jsonl: Path) -> list[IncomingRecord]:
    records: list[IncomingRecord] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(IncomingRecord.model_validate_json(line))
    return records


app = typer.Typer(add_completion=False, help="Stage 3: upsert OCR markdown into Neon.")


@app.command()
def add_column() -> None:
    """Ensure the ``markdown`` column (and the rest of the schema) exists on Neon."""
    NeonDB().init_schema()
    console.print("[green]schema ensured (markdown column present)[/]")


@app.command()
def ingest(
    jsonl: Annotated[Path, typer.Option("--jsonl", help="OCR output jsonl to ingest.")],
    min_chars: Annotated[int, typer.Option(help="Skip rows whose markdown is shorter than this.")] = 200,
    statuses: Annotated[list[str], typer.Option("--status", help="Only ingest these statuses.")] = list(
        _DEFAULT_STATUSES
    ),
    dry_run: Annotated[bool, typer.Option(help="Report what would change without writing.")] = False,
    ids_file: Annotated[
        Path | None, typer.Option("--ids-file", help="Only ingest these arxiv_ids (gate to a subset).")
    ] = None,
    markdown_source: Annotated[
        str, typer.Option("--markdown-source", help="Provenance stamp for the markdown column.")
    ] = "glm-ocr-pdf",
) -> None:
    """Upsert markdown bodies from a Stage-2 jsonl into ``nextjs-ui_paper.markdown``."""
    records = _read_records(jsonl)
    keep = None
    if ids_file is not None:
        keep = {ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    accepted = [
        r
        for r in records
        if r.status in set(statuses) and len(r.markdown) >= min_chars and (keep is None or r.arxiv_id in keep)
    ]
    skipped = len(records) - len(accepted)

    table = Table(title=f"{jsonl.name}: {len(records):,} rows")
    table.add_column("metric", style="cyan")
    table.add_column("count", justify="right", style="magenta")
    table.add_row("accepted (will upsert)", f"{len(accepted):,}")
    table.add_row("skipped (status/empty)", f"{skipped:,}")
    if accepted:
        total_chars = sum(len(r.markdown) for r in accepted)
        table.add_row("avg markdown chars", f"{total_chars // len(accepted):,}")
    console.print(table)

    if dry_run:
        console.print("[yellow]dry-run: no writes[/]")
        return
    if not accepted:
        console.print("[yellow]nothing to ingest[/]")
        return

    db = NeonDB()
    db.init_schema()
    written = 0
    with db.batch() as batch:
        for record in accepted:
            batch.save_paper(record.arxiv_id, markdown=record.markdown, markdown_source=markdown_source)
            written += 1
    logger.success("upserted markdown for {} papers from {}", written, jsonl)
    console.print(f"[bold green]ingested {written:,} markdown bodies into Neon[/]")


if __name__ == "__main__":
    app()
