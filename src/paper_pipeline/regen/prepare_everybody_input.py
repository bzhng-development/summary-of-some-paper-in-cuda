"""Build the V4-Pro inference dataset from EVERY paper in Neon missing a summary.

Sibling of prepare_interested_input.py — drops the interested=1 filter so the
backfill covers the whole DB (including the recent company/lab scrape adds).
Same cache + resume + threaded PDF fetch as the interested variant.

  - Skips ext: papers (no arxiv PDF available)
  - Skips arxiv_ids whose summary is already populated in Neon
  - Skips arxiv_ids missing title OR abstract (can't summarize without metadata)
  - Reuses cached paper_text from local_data/regen_input_FULL.jsonl AND from
    local_data/regen_input_interested_v3.jsonl when present
  - Fetches the rest via multi_prompt_pkg.pdf.download_and_extract_text
  - Resume-safe: re-runs append, skipping arxiv_ids already in output

Usage:
    DATABASE_URL=... uv run python src/paper_pipeline/regen/prepare_everybody_input.py \\
        --output local_data/regen_input_everybody.jsonl \\
        --workers 24
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


from loguru import logger
from psycopg.rows import dict_row


from paper_pipeline.summarize.pdf import arxiv_url_to_pdf_url, download_and_extract_text
from paper_pipeline.core.neon_db import NeonDB, TABLE


def load_cached_input(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "arxiv_id" in r and r.get("paper_text"):
            out[r["arxiv_id"]] = r
    return out


def fetch(arxiv_id: str, url: str | None) -> tuple[str, str | None, str | None]:
    pdf_url = arxiv_url_to_pdf_url(url) if url else f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        text = download_and_extract_text(pdf_url, timeout=90.0)
        return arxiv_id, text, None
    except Exception as e:
        return arxiv_id, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/regen_input_everybody.jsonl"))
    ap.add_argument(
        "--cached",
        type=Path,
        nargs="*",
        default=[
            Path("local_data/regen_input_FULL.jsonl"),
            Path("local_data/regen_input_interested_v3.jsonl"),
            Path("local_data/regen_input_interested.jsonl"),
            Path("local_data/regen_input_interested_v2.jsonl"),
            Path("local_data/regen_input_company.jsonl"),
        ],
    )
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    T = TABLE if TABLE.startswith('"') else f'"{TABLE}"'
    db = NeonDB()
    with db.get_conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            SELECT id, title, url, abstract, primary_category, organization, summary
            FROM {T}
            WHERE (summary IS NULL OR summary = '')
              AND title IS NOT NULL AND title <> ''
              AND abstract IS NOT NULL AND abstract <> ''
              AND id NOT LIKE 'ext%'
            ORDER BY id
        """)
        rows = cur.fetchall()
    logger.info("candidates: {}", len(rows))

    cached: dict[str, dict] = {}
    for p in args.cached:
        c2 = load_cached_input(p)
        before = len(cached)
        cached.update(c2)
        logger.info("loaded {} from {} (+{} new)", len(c2), p, len(cached) - before)

    in_cache = [r for r in rows if r["id"] in cached]
    need_fetch = [r for r in rows if r["id"] not in cached]
    logger.info("cached: {}; need fresh fetch: {}", len(in_cache), len(need_fetch))

    if args.limit:
        need_fetch = need_fetch[: args.limit]
        logger.info("truncated need_fetch to {} (--limit)", len(need_fetch))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["arxiv_id"])
            except json.JSONDecodeError, KeyError:
                continue
        logger.info("resume: {} already in output", len(done))

    with args.output.open("a", encoding="utf-8") as out_fh:
        n_cached_written = 0
        for r in in_cache:
            if r["id"] in done:
                continue
            out_fh.write(json.dumps(cached[r["id"]], ensure_ascii=False) + "\n")
            done.add(r["id"])
            n_cached_written += 1
        out_fh.flush()
        logger.info("wrote {} cached rows", n_cached_written)

        to_fetch = [r for r in need_fetch if r["id"] not in done]
        logger.info("fetching {} PDFs (concurrency={})", len(to_fetch), args.workers)

        n_ok = n_fail = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch, r["id"], r.get("url")): r for r in to_fetch}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                aid, text, err = fut.result()
                if err is not None or text is None:
                    n_fail += 1
                    if n_fail <= 30 or n_fail % 50 == 0:
                        logger.warning("FAIL {}: {}", aid, err)
                    continue
                rec = {
                    "arxiv_id": aid,
                    "title": (r.get("title") or "").strip() or None,
                    "url": r.get("url") or f"https://arxiv.org/abs/{aid}",
                    "abstract": r.get("abstract") or None,
                    "primary_category": r.get("primary_category"),
                    "organization": r.get("organization"),
                    "paper_text": text,
                    "paper_text_chars": len(text),
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_ok += 1
                if i % 50 == 0 or i == len(to_fetch):
                    rate = i / max(time.perf_counter() - t0, 1e-3)
                    eta = (len(to_fetch) - i) / max(rate, 1e-3) / 60
                    logger.info(
                        "progress: {}/{}  ok={}  fail={}  rate={:.1f}/s  eta={:.1f}min",
                        i,
                        len(to_fetch),
                        n_ok,
                        n_fail,
                        rate,
                        eta,
                    )

    logger.info("done. output: {}", args.output)
    logger.info("final tally: cached={}, fetched_ok={}, fetch_failures={}", n_cached_written, n_ok, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
