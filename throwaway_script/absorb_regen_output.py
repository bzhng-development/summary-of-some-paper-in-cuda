"""Absorb the offline_regen.py output JSONL back into Neon.

Reads {arxiv_id, title, category, pitch, summary, url} rows and updates the
matching row in the Neon `"nextjs-ui_paper"` table.

Usage:
    DATABASE_URL=... uv run python throwaway_script/absorb_regen_output.py \
        --input /tmp/regen_output_FULL.jsonl [--limit 5] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from neon_db import NeonDB


def iter_records(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    db = NeonDB()
    n_updated = 0
    n_seen = 0
    with db.batch() as batch:
        for rec in iter_records(inp):
            n_seen += 1
            if args.limit and n_seen > args.limit:
                break
            aid = rec.get("arxiv_id")
            if not aid:
                logger.warning(f"skipping row without arxiv_id: {rec.keys()}")
                continue
            kwargs = {}
            for k in ("title", "category", "pitch", "summary", "url"):
                v = rec.get(k)
                if v is not None:
                    kwargs[k] = v
            if not kwargs:
                logger.warning(f"{aid}: no fields to update")
                continue
            if args.dry_run:
                logger.info(
                    f"[dry-run] would update {aid} fields={list(kwargs)} summary_chars={len(kwargs.get('summary', ''))}"
                )
            else:
                batch.save_paper(aid, **kwargs)
                n_updated += 1
                if n_updated % 50 == 0:
                    logger.info(f"  updated {n_updated} so far...")
    logger.success(f"done. seen={n_seen} updated={n_updated}")


if __name__ == "__main__":
    main()
