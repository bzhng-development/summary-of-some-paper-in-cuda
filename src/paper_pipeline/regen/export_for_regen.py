"""Export all interested=1 papers with re-extracted paper text to JSONL.

Used as input for offline regeneration on the B300.

Usage:
    DATABASE_URL=... uv run python src/paper_pipeline/regen/export_for_regen.py \
        --out /tmp/regen_input.jsonl [--limit 5]

Resumes from last line if --out already exists (skips ids already present).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from loguru import logger
from psycopg.rows import dict_row

from paper_pipeline.core.neon_db import TABLE, NeonDB
from paper_pipeline.summarize.pdf import (
    arxiv_url_to_pdf_url,
    download_and_extract_text,
)


def already_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row.get("arxiv_id"), str):
            done.add(row["arxiv_id"])
    return done


def resolve_pdf_url(arxiv_id: str, url: str | None) -> str:
    if arxiv_id.startswith("ext:"):
        # external paper: URL is the direct PDF link
        if not url:
            raise ValueError(f"ext paper {arxiv_id} has no url")
        return url
    return arxiv_url_to_pdf_url(f"https://arxiv.org/abs/{arxiv_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--sleep", type=float, default=0.5, help="polite delay between PDF fetches (seconds)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = already_done(out_path)
    logger.info(f"Already in JSONL: {len(done)}")

    db = NeonDB()
    with db.get_conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT id, title, url, abstract, primary_category, organization FROM {TABLE} WHERE interested = 1 ORDER BY id"
        )
        rows = cur.fetchall()
    logger.info(f"Found {len(rows)} interested=1 papers in Neon")

    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"Will fetch {len(todo)} papers")

    ok = 0
    failed: list[tuple[str, str]] = []
    with out_path.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            aid = r["id"]
            try:
                pdf_url = resolve_pdf_url(aid, r.get("url"))
                text = download_and_extract_text(pdf_url, timeout=60.0)
            except Exception as exc:
                logger.warning(f"[{i}/{len(todo)}] {aid}: FAIL {exc!r}")
                failed.append((aid, repr(exc)))
                continue

            record = {
                "arxiv_id": aid,
                "title": r["title"],
                "url": r["url"],
                "abstract": r["abstract"],
                "primary_category": r["primary_category"],
                "organization": r["organization"],
                "paper_text": text,
                "paper_text_chars": len(text),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
            if i % 25 == 0:
                logger.info(f"  progress: {i}/{len(todo)} ok={ok} failed={len(failed)}")
            time.sleep(args.sleep)

    logger.success(f"Done. ok={ok} failed={len(failed)} total_in_jsonl={len(done) + ok}")
    if failed:
        fail_path = out_path.with_suffix(out_path.suffix + ".failed.json")
        fail_path.write_text(json.dumps(failed, indent=2))
        logger.warning(f"Failed ids written to {fail_path}")


if __name__ == "__main__":
    main()
