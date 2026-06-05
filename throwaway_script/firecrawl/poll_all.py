"""Poll all Firecrawl crawl IDs from a submission JSONL and flatten results.

Reads firecrawl_jobs.jsonl (output of submit_throttled.py), pulls every
successful crawl_id, polls /v2/crawl/{id} until status=completed (paginates
via response.next), and writes one flat row per crawled page to a results
jsonl:

  {crawl_id, source_url (origin URL of the crawl), page_url, title, markdown, links}

Stays under the free-tier rate cap by sleeping between polls.

Usage:
    FIRECRAWL_API_KEY=... uv run python throwaway_script/firecrawl/poll_all.py \\
        --jobs local_data/firecrawl_jobs.jsonl \\
        --out local_data/firecrawl_results.jsonl \\
        --poll-interval 8 --max-polls-per-id 30
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from loguru import logger

API_BASE = "https://api.firecrawl.dev/v2/crawl"
TERMINAL = {"completed", "failed"}


def read_jobs(path: Path) -> list[tuple[str, str]]:
    """Return [(source_url, crawl_id), ...] for jobs that submitted ok."""
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status_code") != 200:
            continue
        cid = (r.get("response") or {}).get("id")
        if cid:
            out.append((r["url"], cid))
    # Deduplicate (the file may have stale 429 rows + later success rows).
    seen: dict[str, str] = {}
    for src, cid in out:
        seen[cid] = src
    return [(src, cid) for cid, src in seen.items()]


def read_done(path: Path) -> set[str]:
    """Return the set of crawl_ids that already have any completed/failed
    record in the output file — used to resume without redoing finished crawls."""
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("_status") in TERMINAL:
            done.add(r["crawl_id"])
    return done


def poll_one(
    session: requests.Session,
    crawl_id: str,
    source_url: str,
    out_path: Path,
    poll_interval: float,
    max_polls: int,
) -> str:
    """Poll one crawl id to completion. Streams new pages to out_path as they arrive."""
    seen_pages: set[str] = set()
    next_url: str | None = None
    polls = 0
    status = "unknown"
    while polls < max_polls:
        url = next_url or f"{API_BASE}/{crawl_id}"
        try:
            resp = session.get(url, timeout=60)
        except requests.RequestException as e:
            logger.warning("[{}] request failed: {}", crawl_id[:8], e)
            time.sleep(poll_interval)
            polls += 1
            continue

        if resp.status_code == 429:
            logger.warning("[{}] 429; sleeping 65s", crawl_id[:8])
            time.sleep(65)
            continue
        if resp.status_code != 200:
            logger.warning("[{}] http {} body={}", crawl_id[:8], resp.status_code, resp.text[:120])
            time.sleep(poll_interval)
            polls += 1
            continue

        body = resp.json()
        status = body.get("status", "unknown")
        for page in body.get("data") or []:
            meta = page.get("metadata") or {}
            page_url = meta.get("sourceURL") or meta.get("url") or ""
            if page_url in seen_pages:
                continue
            seen_pages.add(page_url)
            rec = {
                "_status": "page",
                "crawl_id": crawl_id,
                "source_url": source_url,
                "page_url": page_url,
                "title": meta.get("title"),
                "markdown": page.get("markdown"),
                "links": page.get("links"),
            }
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        next_url = body.get("next")
        completed = body.get("completed")
        total = body.get("total")
        logger.info(
            "[{}] {} | status={} completed={}/{} pages_streamed={} next={}",
            crawl_id[:8],
            source_url[:40],
            status,
            completed,
            total,
            len(seen_pages),
            "yes" if next_url else "no",
        )

        # Once we get a terminal status AND there's no next-page cursor, write
        # a marker row so the resume check picks this up and we move on.
        if status in TERMINAL and not next_url:
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "_status": status,
                            "crawl_id": crawl_id,
                            "source_url": source_url,
                            "pages_total": len(seen_pages),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return status

        time.sleep(poll_interval)
        polls += 1
    logger.warning("[{}] max polls reached, last status={}", crawl_id[:8], status)
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poll-interval", type=float, default=8.0)
    ap.add_argument("--max-polls-per-id", type=int, default=30)
    args = ap.parse_args()

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("FIRECRAWL_API_KEY not set")
        return 2

    jobs = read_jobs(args.jobs)
    done = read_done(args.out)
    todo = [(src, cid) for src, cid in jobs if cid not in done]
    logger.info("jobs total={}  done={}  todo={}", len(jobs), len(done), len(todo))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})

    for i, (src, cid) in enumerate(todo, 1):
        logger.info("--- [{}/{}] cid={} {} ---", i, len(todo), cid, src)
        poll_one(session, cid, src, args.out, args.poll_interval, args.max_polls_per_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
