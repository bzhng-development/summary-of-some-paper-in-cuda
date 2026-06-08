"""Submit Firecrawl v2 crawl jobs with rate-limit-aware pacing.

Free tier is 3 req/min — burst-submitting 83 URLs gets all but the first 3
rate-limited. This wrapper:

  - Skips URLs already submitted ok (resume by reading the output jsonl).
  - Throttles new submissions to one every ~21 s so we stay under 3/min.
  - On 429, parses the "retry after Ns" hint and sleeps that long.
  - Writes each result to the same jsonl format the existing
    firecrawl_crawl_urls.py uses.

Usage:
    FIRECRAWL_API_KEY=... uv run python src/paper_pipeline/discovery/firecrawl/submit_throttled.py \\
        --input src/paper_pipeline/discovery/firecrawl/company_urls.txt \\
        --out local_data/firecrawl_jobs.jsonl \\
        --limit 200 --max-discovery-depth 2
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

API_URL = "https://api.firecrawl.dev/v2/crawl"
DEFAULT_INTERVAL_S = 21.0  # ~3 req/min cap with a tiny buffer
RETRY_AFTER_RE = re.compile(r"retry after (\d+)s", re.IGNORECASE)


def read_urls(path: Path) -> list[str]:
    urls: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def read_done(out_path: Path) -> set[str]:
    if not out_path.is_file():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status_code") == 200 and (r.get("response") or {}).get("id"):
            done.add(r["url"])
    return done


def parse_retry_after(text: str) -> float | None:
    m = RETRY_AFTER_RE.search(text or "")
    return float(m.group(1)) if m else None


def build_payload(url: str, args: argparse.Namespace) -> dict:
    payload: dict = {"url": url}
    if args.max_discovery_depth is not None:
        payload["maxDiscoveryDepth"] = args.max_discovery_depth
    if args.limit is not None:
        payload["limit"] = args.limit
    if args.allow_subdomains:
        payload["allowSubdomains"] = True
    if args.ignore_query_parameters:
        payload["ignoreQueryParameters"] = True
    if args.max_concurrency is not None:
        payload["maxConcurrency"] = args.max_concurrency
    if args.crawl_entire_domain:
        payload["crawlEntireDomain"] = True
    if args.include_paths:
        payload["includePaths"] = [p.strip() for p in args.include_paths.split(",") if p.strip()]
    if args.exclude_paths:
        payload["excludePaths"] = [p.strip() for p in args.exclude_paths.split(",") if p.strip()]
    scrape_opts: dict = {}
    if args.formats:
        scrape_opts["formats"] = [s.strip() for s in args.formats.split(",") if s.strip()]
    if args.only_main_content:
        scrape_opts["onlyMainContent"] = True
    if scrape_opts:
        payload["scrapeOptions"] = scrape_opts
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-discovery-depth", type=int, default=2)
    ap.add_argument("--allow-subdomains", action="store_true")
    ap.add_argument("--ignore-query-parameters", action="store_true", default=True)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--crawl-entire-domain", action="store_true")
    ap.add_argument("--include-paths", default="", help="comma-separated regex patterns; matches restrict scraped URLs")
    ap.add_argument("--exclude-paths", default="")
    ap.add_argument("--formats", default="markdown,links")
    ap.add_argument("--only-main-content", action="store_true", default=True)
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    args = ap.parse_args()

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("FIRECRAWL_API_KEY not set")
        return 2

    urls = read_urls(args.input)
    done = read_done(args.out)
    todo = [u for u in urls if u not in done]
    logger.info("urls total={}  already_submitted={}  todo={}", len(urls), len(done), len(todo))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})

    last_submit = 0.0
    for i, url in enumerate(todo, 1):
        elapsed = time.perf_counter() - last_submit
        wait = max(0.0, args.interval - elapsed)
        if wait > 0:
            time.sleep(wait)
        last_submit = time.perf_counter()

        payload = build_payload(url, args)
        try:
            resp = session.post(API_URL, json=payload, timeout=60)
        except requests.RequestException as e:
            logger.warning("request failed for {}: {}", url, e)
            _append(args.out, {"url": url, "error": str(e)})
            continue

        body = None
        try:
            body = resp.json()
        except ValueError:
            body = {"raw": resp.text}

        if resp.status_code == 429:
            ra = parse_retry_after(body.get("error", ""))
            wait_more = ra + 2.0 if ra else 65.0
            logger.warning("429 on {}  → sleeping {:.0f}s and retrying", url, wait_more)
            time.sleep(wait_more)
            try:
                resp = session.post(API_URL, json=payload, timeout=60)
                body = (
                    resp.json()
                    if resp.headers.get("content-type", "").startswith("application/json")
                    else {"raw": resp.text}
                )
            except requests.RequestException as e:
                logger.warning("retry failed for {}: {}", url, e)
                _append(args.out, {"url": url, "error": str(e), "retry": True})
                last_submit = time.perf_counter()
                continue
            last_submit = time.perf_counter()

        rec = {
            "url": url,
            "status_code": resp.status_code,
            "request": payload,
            "response": body,
        }
        _append(args.out, rec)
        ok = resp.status_code == 200 and isinstance(body, dict) and body.get("id")
        logger.info("[{}/{}] {} → {}  {}", i, len(todo), url, resp.status_code, body.get("id") if ok else "")
    return 0


def _append(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
