#!/usr/bin/env python3
"""Poll a Firecrawl v2 crawl job until completion and append results to JSONL."""

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional, Set, Tuple

import requests
from loguru import logger

from paper_pipeline.discovery.firecrawl.firecrawl_common import (
    API_BASE,
    append_jsonl,
    build_headers,
    require_api_key,
    setup_logger,
)

TERMINAL_STATUSES = {"completed", "failed"}


def _build_url(crawl_id: str, next_url: Optional[str]) -> str:
    if next_url:
        return next_url
    return f"{API_BASE}/{crawl_id}"


def _record_key(record: Dict[str, Any]) -> Tuple[Any, ...]:
    crawl_id = record.get("crawl_id")
    status_code = record.get("status_code")
    response = record.get("response") or {}
    if "error" in record:
        return (crawl_id, status_code, record.get("error"))
    return (
        crawl_id,
        status_code,
        response.get("status"),
        response.get("next"),
        response.get("completed"),
        response.get("total"),
    )


def _load_existing_keys(out_path: str) -> Set[Tuple[Any, ...]]:
    existing: Set[Tuple[Any, ...]] = set()
    try:
        with open(out_path, "r", encoding="utf-8") as out_f:
            for line in out_f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                existing.add(_record_key(record))
    except FileNotFoundError:
        return existing
    return existing


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll Firecrawl v2 crawl status and append results to JSONL.")
    parser.add_argument("crawl_id", help="Crawl job id")
    parser.add_argument(
        "--out",
        default="firecrawl_crawl_results.jsonl",
        help="JSONL output path (default: firecrawl_crawl_results.jsonl)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between polls (default: 5)",
    )
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="Maximum polls before exit (0 = no limit)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logger(args.log_level.upper())

    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    session = requests.Session()
    session.headers.update(build_headers(api_key))

    existing_keys = _load_existing_keys(args.out)
    if existing_keys:
        logger.info("Loaded {} existing records from {}", len(existing_keys), args.out)

    polls = 0
    next_url: Optional[str] = None
    last_url: Optional[str] = None
    stalled = 0
    logger.info("Starting poll for crawl_id={} output={}", args.crawl_id, args.out)
    try:
        while True:
            if args.max_polls and polls >= args.max_polls:
                logger.warning("Reached max polls={}; exiting", args.max_polls)
                return 3

            url = _build_url(args.crawl_id, next_url)
            logger.info("Poll {} url={}", polls + 1, url)
            try:
                resp = session.get(url, timeout=60)
            except requests.RequestException as exc:
                logger.warning("Request failed: {}", exc)
                error_record = {"crawl_id": args.crawl_id, "error": str(exc)}
                key = _record_key(error_record)
                if key not in existing_keys:
                    append_jsonl(args.out, error_record)
                    existing_keys.add(key)
                time.sleep(args.poll_interval)
                polls += 1
                continue

            record: Dict[str, Any] = {
                "crawl_id": args.crawl_id,
                "status_code": resp.status_code,
                "response": None,
            }
            try:
                record["response"] = resp.json()
            except ValueError:
                record["response"] = {"raw": resp.text}

            key = _record_key(record)
            if key not in existing_keys:
                append_jsonl(args.out, record)
                existing_keys.add(key)

            data = record.get("response") or {}
            status = data.get("status")
            next_url = data.get("next")
            logger.info("Status={} next_url={}", status, bool(next_url))

            if next_url:
                if next_url == last_url and status not in TERMINAL_STATUSES:
                    stalled += 1
                    time.sleep(args.poll_interval)
                    polls += 1
                    if stalled >= 3:
                        logger.info("Next URL stalled; polling base endpoint next")
                        next_url = None
                        last_url = None
                        stalled = 0
                    continue
                stalled = 0
                last_url = next_url
                continue

            if status in TERMINAL_STATUSES:
                logger.info("Terminal status reached: {}", status)
                return 0

            time.sleep(args.poll_interval)
            polls += 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user; exiting")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
