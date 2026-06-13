"""Build a regen_input.jsonl from HF Daily Papers (no Neon, no LLM scoring).

Output schema matches src/paper_pipeline/regen/export_for_regen.py — the input format
offline_regen.py consumes:

    {arxiv_id, title, url, abstract, primary_category, organization,
     paper_text, paper_text_chars}

Usage:
    # Today's papers
    uv run python src/paper_pipeline/regen/build_regen_input_from_hf.py \
        --out /tmp/regen_input_FULL.jsonl

    # Date range
    uv run python src/paper_pipeline/regen/build_regen_input_from_hf.py \
        --out /tmp/regen_input_FULL.jsonl --from 2026-05-01 --to 2026-05-25

    # Specific day
    uv run python src/paper_pipeline/regen/build_regen_input_from_hf.py \
        --out /tmp/regen_input_FULL.jsonl --date 2026-05-20

Resumes from --out if it exists (skips arxiv_ids already present).
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from paper_pipeline.ingest.hf_daily_papers import (
    Paper,
    fetch_arxiv_metadata,
    fetch_papers,
    fetch_papers_range,
)
from paper_pipeline.summarize.pdf import arxiv_url_to_pdf_url, download_and_extract_text


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


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def enrich_missing(papers: list[Paper]) -> None:
    """Fill in primary_category + abstract via arxiv API for papers missing it."""
    need = [p for p in papers if not p.primary_category or not p.summary]
    if not need:
        return
    logger.info(f"Enriching {len(need)}/{len(papers)} via arxiv API...")
    meta = fetch_arxiv_metadata([p.arxiv_id for p in need])
    for p in need:
        m = meta.get(p.arxiv_id)
        if not m:
            continue
        if not p.summary and m.abstract:
            p.summary = m.abstract
        if not p.primary_category:
            p.primary_category = m.primary_category
        if not p.categories:
            p.categories = m.categories
        if not p.affiliations:
            p.affiliations = m.affiliations


def fetch_one_pdf(p: Paper, timeout: float) -> tuple[Paper, str | None, str | None]:
    """Returns (paper, paper_text, error_repr)."""
    url = f"https://arxiv.org/abs/{p.arxiv_id}"
    try:
        text = download_and_extract_text(arxiv_url_to_pdf_url(url), timeout=timeout)
        return p, text, None
    except Exception as exc:
        return p, None, repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--date", help="Single date YYYY-MM-DD")
    ap.add_argument("--from", dest="date_from", help="Range start YYYY-MM-DD")
    ap.add_argument("--to", dest="date_to", help="Range end YYYY-MM-DD (default: today)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8, help="parallel PDF fetches")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--skip-arxiv-enrich",
        action="store_true",
        help="don't backfill missing primary_category/abstract via arxiv API",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.date_from:
        start = parse_date(args.date_from)
        end = parse_date(args.date_to) if args.date_to else date.today()
        papers = fetch_papers_range(start, end)
    elif args.date:
        papers = fetch_papers(parse_date(args.date))
    else:
        papers = fetch_papers()

    logger.info(f"Got {len(papers)} unique papers from HF")

    if not args.skip_arxiv_enrich:
        enrich_missing(papers)

    done = already_done(out_path)
    logger.info(f"Already in {out_path.name}: {len(done)}")

    todo = [p for p in papers if p.arxiv_id not in done]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"Will fetch {len(todo)} PDFs (workers={args.workers})")

    ok = 0
    failed: list[tuple[str, str]] = []
    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one_pdf, p, args.timeout): p for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            p, text, err = fut.result()
            if err is not None or not text:
                logger.warning(f"[{i}/{len(todo)}] {p.arxiv_id}: FAIL {err}")
                failed.append((p.arxiv_id, err or "empty text"))
                continue
            record = {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "url": f"https://arxiv.org/abs/{p.arxiv_id}",
                "abstract": p.summary or None,
                "primary_category": p.primary_category,
                "organization": p.org_fullname or p.organization,
                "paper_text": text,
                "paper_text_chars": len(text),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            ok += 1
            if i % 25 == 0:
                logger.info(f"  progress: {i}/{len(todo)} ok={ok} failed={len(failed)}")

    logger.success(f"Done. ok={ok} failed={len(failed)} total_in_jsonl={len(done) + ok}")
    if failed:
        fail_path = out_path.with_suffix(out_path.suffix + ".failed.json")
        fail_path.write_text(json.dumps(failed, indent=2))
        logger.warning(f"Failed ids -> {fail_path}")


if __name__ == "__main__":
    main()
