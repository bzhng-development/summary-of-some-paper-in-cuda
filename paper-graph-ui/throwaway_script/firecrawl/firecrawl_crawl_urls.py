#!/usr/bin/env python3
"""Start Firecrawl v2 crawls for a list of URLs and dump responses to JSONL."""

import argparse
import os
from typing import Any

import requests
from firecrawl_common import (
    API_BASE,
    append_jsonl,
    build_headers,
    require_api_key,
    setup_logger,
    split_csv,
)
from loguru import logger

API_URL = API_BASE


def _read_urls(path: str) -> list[str]:
    urls: list[str] = []
    if os.path.isfile(path):
        with open(path, encoding="ascii") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                urls.append(line)
        return urls

    if path.startswith(("http://", "https://")):
        return [path]

    raise ValueError(
        "Input must be a URL or a path to a text file of URLs. "
        "If you meant to poll a crawl id, use firecrawl_poll_crawl.py."
    )


def _build_payload(args: argparse.Namespace, url: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": url,
    }

    if args.prompt:
        payload["prompt"] = args.prompt

    if args.crawl_entire_domain:
        payload["crawlEntireDomain"] = True
    if args.max_discovery_depth is not None:
        payload["maxDiscoveryDepth"] = args.max_discovery_depth
    if args.sitemap:
        payload["sitemap"] = args.sitemap
    if args.limit is not None:
        payload["limit"] = args.limit
    if args.allow_external_links:
        payload["allowExternalLinks"] = True
    if args.allow_subdomains:
        payload["allowSubdomains"] = True
    if args.ignore_query_parameters:
        payload["ignoreQueryParameters"] = True
    if args.delay is not None:
        payload["delay"] = args.delay
    if args.max_concurrency is not None:
        payload["maxConcurrency"] = args.max_concurrency
    if args.zero_data_retention:
        payload["zeroDataRetention"] = True
    if args.include_paths:
        payload["includePaths"] = args.include_paths
    if args.exclude_paths:
        payload["excludePaths"] = args.exclude_paths

    scrape_options: dict[str, Any] = {}
    if args.formats:
        scrape_options["formats"] = args.formats
    if args.only_main_content:
        scrape_options["onlyMainContent"] = True
    if args.include_tags:
        scrape_options["includeTags"] = args.include_tags
    if args.exclude_tags:
        scrape_options["excludeTags"] = args.exclude_tags
    if args.max_age is not None:
        scrape_options["maxAge"] = args.max_age

    if scrape_options:
        payload["scrapeOptions"] = scrape_options

    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Firecrawl v2 crawls for URLs listed in a text file.")
    parser.add_argument("input", help="Path to a text file of URLs")
    parser.add_argument(
        "--out",
        default="firecrawl_crawl.jsonl",
        help="JSONL output path (default: firecrawl_crawl.jsonl)",
    )
    parser.add_argument(
        "--prompt",
        help="Natural-language crawl prompt (maps to crawler settings)",
    )
    parser.add_argument(
        "--crawl-entire-domain",
        action="store_true",
        help="Crawl the whole domain, not just child paths",
    )
    parser.add_argument(
        "--max-discovery-depth",
        type=int,
        default=5,
        help="Maximum discovery depth",
    )
    parser.add_argument(
        "--sitemap",
        choices=["include", "skip"],
        default="skip",
        help="Sitemap mode (include|skip)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum pages to crawl",
    )
    parser.add_argument(
        "--allow-external-links",
        action="store_true",
        help="Allow crawling external links",
    )
    parser.add_argument(
        "--allow-subdomains",
        action="store_true",
        help="Allow crawling subdomains",
    )
    parser.add_argument(
        "--ignore-query-parameters",
        action="store_true",
        default=True,
        help="Ignore query parameters when deduplicating URLs",
    )
    parser.add_argument(
        "--no-ignore-query-parameters",
        action="store_false",
        dest="ignore_query_parameters",
        help="Do not ignore query parameters when deduplicating URLs",
    )
    parser.add_argument(
        "--include-paths",
        help="Comma-separated regex patterns to include",
    )
    parser.add_argument(
        "--exclude-paths",
        help="Comma-separated regex patterns to exclude",
    )
    parser.add_argument(
        "--delay",
        type=float,
        help="Delay in seconds between scrapes",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum concurrent scrapes",
    )
    parser.add_argument(
        "--zero-data-retention",
        action="store_true",
        help="Enable zero data retention",
    )
    parser.add_argument(
        "--formats",
        default="markdown,links",
        help="Comma-separated formats (e.g., markdown,html,links)",
    )
    parser.add_argument(
        "--only-main-content",
        action="store_true",
        default=True,
        help="Only return main content",
    )
    parser.add_argument(
        "--no-only-main-content",
        action="store_false",
        dest="only_main_content",
        help="Return full content including nav/footers",
    )
    parser.add_argument(
        "--include-tags",
        help="Comma-separated HTML tags to include",
    )
    parser.add_argument(
        "--exclude-tags",
        help="Comma-separated HTML tags to exclude",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=600000,
        help="Cache max age in milliseconds (default: 600000)",
    )

    args = parser.parse_args()

    setup_logger("INFO")

    try:
        api_key = require_api_key()
    except RuntimeError as exc:
        logger.error("{}", exc)
        return 2

    args.formats = split_csv(args.formats)
    args.include_tags = split_csv(args.include_tags)
    args.exclude_tags = split_csv(args.exclude_tags)
    args.include_paths = split_csv(args.include_paths)
    args.exclude_paths = split_csv(args.exclude_paths)

    try:
        urls = _read_urls(args.input)
    except ValueError as exc:
        logger.error("{}", exc)
        return 2
    if not urls:
        logger.error("No URLs found in input file.")
        return 2

    session = requests.Session()
    session.headers.update(build_headers(api_key, content_type=True))

    for url in urls:
        payload = _build_payload(args, url)
        try:
            resp = session.post(API_URL, json=payload, timeout=60)
        except requests.RequestException as exc:
            logger.warning("Request failed for {}: {}", url, exc)
            record = {"url": url, "error": str(exc)}
            append_jsonl(args.out, record)
            continue

        record = {
            "url": url,
            "status_code": resp.status_code,
            "request": payload,
            "response": None,
        }
        try:
            record["response"] = resp.json()
        except ValueError:
            record["response"] = {"raw": resp.text}

        append_jsonl(args.out, record)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
