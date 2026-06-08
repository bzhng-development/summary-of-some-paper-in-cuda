"""Scrape arxiv search-result pages via Firecrawl's /v2/scrape (synchronous).

The /v2/crawl endpoint is for multi-page batch crawls and queues each job for
minutes. For one-page-each arxiv search URLs, /v2/scrape returns the markdown
synchronously, much faster.

Reads a URL list, scrapes each, writes one row per URL to a results jsonl
with the page's markdown. Then a downstream pass can regex out arxiv IDs.

Rate-limit aware: Firecrawl free tier is ~3 req/min on /v2/scrape too, so we
sleep ~21s between requests.

Usage:
    FIRECRAWL_API_KEY=... uv run python src/paper_pipeline/discovery/firecrawl/scrape_arxiv_pages.py \\
        --input src/paper_pipeline/discovery/firecrawl/arxiv_search_urls.txt \\
        --out local_data/firecrawl_arxiv_scrapes.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from loguru import logger

API_URL = "https://api.firecrawl.dev/v2/scrape"
RETRY_AFTER_RE = re.compile(r"retry after (\d+)s", re.IGNORECASE)


def read_urls(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def read_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if r.get("ok"):
                done.add(r["url"])
        except json.JSONDecodeError:
            continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=21.0)
    args = ap.parse_args()

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("FIRECRAWL_API_KEY not set")
        return 2

    urls = read_urls(args.input)
    done = read_done(args.out)
    todo = [u for u in urls if u not in done]
    logger.info("urls total={}  done={}  todo={}", len(urls), len(done), len(todo))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})

    last = 0.0
    for i, url in enumerate(todo, 1):
        wait = max(0.0, args.interval - (time.perf_counter() - last))
        if wait > 0:
            time.sleep(wait)
        last = time.perf_counter()

        payload = {"url": url, "formats": ["markdown", "links"], "onlyMainContent": True}
        try:
            resp = session.post(API_URL, json=payload, timeout=120)
        except requests.RequestException as e:
            _write(args.out, {"url": url, "ok": False, "error": str(e)})
            logger.warning("[{}/{}] {} → exception: {}", i, len(todo), url[:60], e)
            continue

        body: dict | str
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text[:500]}

        if resp.status_code == 429:
            ra = 60
            err = body.get("error", "") if isinstance(body, dict) else ""
            m = RETRY_AFTER_RE.search(err)
            if m:
                ra = int(m.group(1)) + 2
            logger.warning("[{}/{}] 429 → sleeping {}s and retrying", i, len(todo), ra)
            time.sleep(ra)
            try:
                resp = session.post(API_URL, json=payload, timeout=120)
                body = resp.json()
            except Exception as e:
                _write(args.out, {"url": url, "ok": False, "error": f"retry {e}"})
                last = time.perf_counter()
                continue
            last = time.perf_counter()

        ok = resp.status_code == 200 and isinstance(body, dict) and body.get("success")
        data = (body or {}).get("data") or {}
        rec = {
            "url": url,
            "ok": ok,
            "status_code": resp.status_code,
            "markdown": data.get("markdown"),
            "links": data.get("links"),
            "metadata": data.get("metadata"),
        }
        _write(args.out, rec)
        md_len = len(rec.get("markdown") or "")
        logger.info("[{}/{}] {} → {}  md={}b", i, len(todo), url[:60], resp.status_code, md_len)
    return 0


def _write(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
