"""Backfill cited_by_count / fwci / abstract onto EXISTING Neon rows from the OpenAlex
audit JSONL. The save run only wrote net-new rows; this enriches the ones already in Neon
(in_neon=True, ~4.3k) — and is idempotent for the net-new ones too. Citation fields are
always set (authoritative from OpenAlex); abstract only fills where currently empty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from paper_pipeline.core.neon_db import NeonDB, TABLE

console = Console()
JSONL = Path("local_data/openalex_2000-2024_audit.jsonl")


def main(dry_run: Annotated[bool, typer.Option(help="Preview counts, write nothing.")] = False) -> None:
    # Dedupe by arxiv_id, keeping the record that carries citation data.
    best: dict[str, dict] = {}
    for line in JSONL.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        aid = r.get("arxiv_id")
        if not aid:
            continue
        if aid not in best or (r.get("cited_by_count") is not None and best[aid].get("cited_by_count") is None):
            best[aid] = r
    rows = [
        (r["arxiv_id"], r.get("cited_by_count"), r.get("fwci"), (r.get("abstract") or None))
        for r in best.values()
        if r.get("cited_by_count") is not None or r.get("fwci") is not None or r.get("abstract")
    ]
    console.print(f"[bold]{len(best):,}[/] unique arxiv_ids in JSONL; [green]{len(rows):,}[/] carry enrichable data")

    db = NeonDB()
    with db.get_conn() as c, c.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE score_source='openalex_audit' AND cited_by_count IS NULL")
        console.print(f"openalex rows currently missing cited_by_count: [yellow]{cur.fetchone()[0]:,}[/]")
        if dry_run:
            console.print("[yellow]dry-run: no writes[/]")
            return
        # citation fields authoritative -> always set; abstract only where currently empty.
        cur.executemany(
            f"""UPDATE {TABLE}
                SET cited_by_count = %s,
                    fwci = %s,
                    abstract = COALESCE(NULLIF(abstract, ''), %s)
                WHERE id = %s""",
            [(cc, fw, ab, aid) for (aid, cc, fw, ab) in rows],
        )
        c.commit()
        cur.execute(f"SELECT count(*) FROM {TABLE} WHERE cited_by_count IS NOT NULL")
        console.print(f"[bold green]done[/] — rows with cited_by_count now: {cur.fetchone()[0]:,}")


if __name__ == "__main__":
    typer.run(main)
