"""Build inference input for the company-flagged papers.

Reads all rows where ``is_only_important_because_of_company = TRUE``, skips
the ones that already have a summary in Neon, then assembles a single jsonl
ready for ``offline_regen.py`` to run on (one row per paper).

For each paper that needs summary generation:
  - If the paper is already in local_data/regen_input_FULL.jsonl (the
    cached April-May scrape), reuse that row verbatim.
  - Else: fetch + extract text via multi_prompt_pkg.pdf.download_and_extract_text
    (tries `hf papers read <id>` first via the patched fast path, falls back
    to arxiv PDF + PyMuPDF).

Output: local_data/regen_input_company.jsonl with schema matching what
offline_regen.py expects: arxiv_id, title, url, abstract, paper_text.

Usage:
    DATABASE_URL=... uv run python src/paper_pipeline/regen/prepare_company_input.py \\
        --output local_data/regen_input_company.jsonl \\
        --workers 16
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from psycopg.rows import dict_row

from paper_pipeline.core.neon_db import TABLE, NeonDB
from paper_pipeline.summarize.pdf import arxiv_url_to_pdf_url, download_and_extract_text


def load_cached_input(path: Path) -> dict[str, dict]:
    """Return {arxiv_id: row} from regen_input_FULL.jsonl, skipping bad lines."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if "arxiv_id" in r and r.get("paper_text"):
                out[r["arxiv_id"]] = r
        except json.JSONDecodeError:
            continue
    return out


def fetch_paper_text(arxiv_id: str, url: str | None) -> tuple[str, str | None, str | None]:
    """Return (arxiv_id, paper_text, error_message). Used inside the thread pool."""
    pdf_url = arxiv_url_to_pdf_url(url) if url else f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        text = download_and_extract_text(pdf_url, timeout=90.0)
        return arxiv_id, text, None
    except Exception as e:
        return arxiv_id, None, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/regen_input_company.jsonl"))
    ap.add_argument("--cached", type=Path, default=Path("local_data/regen_input_FULL.jsonl"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="cap papers (debug)")
    args = ap.parse_args()

    db = NeonDB()
    with db.get_conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            SELECT id, title, url, abstract, primary_category, organization, summary
            FROM {TABLE}
            WHERE is_only_important_because_of_company = TRUE
            ORDER BY id
        """)
        flagged = cur.fetchall()
    logger.info("flagged: {} total", len(flagged))

    need = [r for r in flagged if not (r.get("summary") or "").strip()]
    skip_ext = [r for r in need if r["id"].startswith("ext")]
    arxiv_need = [r for r in need if not r["id"].startswith("ext")]
    logger.info(
        "of those: {} already-summarised; {} ext: (skipping, no arxiv PDF); {} arxiv to fetch",
        len(flagged) - len(need),
        len(skip_ext),
        len(arxiv_need),
    )

    cached = load_cached_input(args.cached)
    in_cache = [r for r in arxiv_need if r["id"] in cached]
    need_fetch = [r for r in arxiv_need if r["id"] not in cached]
    logger.info(
        "split: {} cached (paper_text already extracted), {} need fresh PDF fetch",
        len(in_cache),
        len(need_fetch),
    )

    if args.limit:
        need_fetch = need_fetch[: args.limit]
        logger.info("truncated need_fetch to {}", len(need_fetch))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip arxiv_ids already in output
    done: set[str] = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        logger.info("resume: {} already in output", len(done))

    # Open in append mode so re-runs are idempotent across crashes.
    with args.output.open("a", encoding="utf-8") as out_fh:
        # First: write all cached rows that aren't already in output.
        n_cached_written = 0
        for r in in_cache:
            if r["id"] in done:
                continue
            cached_row = cached[r["id"]]
            out_fh.write(json.dumps(cached_row, ensure_ascii=False) + "\n")
            done.add(r["id"])
            n_cached_written += 1
        out_fh.flush()
        logger.info("wrote {} cached rows", n_cached_written)

        # Then: fan out fetches. Filter against the resume set.
        to_fetch = [r for r in need_fetch if r["id"] not in done]
        logger.info("fetching {} PDFs (concurrency={})", len(to_fetch), args.workers)

        n_ok = 0
        n_fail = 0
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fetch_paper_text, r["id"], r.get("url")): r for r in to_fetch}
            for i, fut in enumerate(as_completed(futures), 1):
                r = futures[fut]
                aid, text, err = fut.result()
                if err is not None or text is None:
                    n_fail += 1
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
                if i % 25 == 0 or i == len(to_fetch):
                    rate = i / max(time.perf_counter() - t0, 1e-3)
                    logger.info("progress: {}/{}  ok={}  fail={}  rate={:.1f}/s", i, len(to_fetch), n_ok, n_fail, rate)

    logger.info("done. output: {}", args.output)
    logger.info("final: cached={}, newly-fetched-ok={}, fetch-failures={}", n_cached_written, n_ok, n_fail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
