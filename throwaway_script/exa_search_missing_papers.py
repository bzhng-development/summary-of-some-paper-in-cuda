#!/usr/bin/env python3
"""Search missing papers with Exa and write one result per line to JSONL."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


YEAR_HEADER_RE = re.compile(r"^(?:\*\*(.+?)\*\*|##\s+(.+))$")
YEAR_VALUE_RE = re.compile(r"^(\d{4})$")
BULLET_RE = re.compile(r"^\s*-\s+")
INLINE_PAPER_RE = re.compile(r"^\*\*(.+?)\*\*")

DEFAULT_INPUT = "missing-classic-papers-2012-2024.md"
DEFAULT_OUTPUT = "missing-classic-papers-2012-2024.exa.jsonl"
DEFAULT_API_BASE = "https://api.exa.ai"
DEFAULT_API_KEY = "a4b79ae3-c968-43b8-b657-3283ce1c9950"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read missing-classic-papers markdown entries, send one Exa request "
            "per paper, and write results to a JSONL file."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Markdown file to read. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"JSONL file to write. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"Base URL for Exa API. Default: {DEFAULT_API_BASE}",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.75,
        help="Sleep between API calls. Default: 0.75",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=7,
        help="Maximum number of concurrent Exa requests. Default: 7.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of papers to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite the output file instead of resuming from existing JSONL entries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse entries and print requests without calling Exa.",
    )
    return parser.parse_args()


def normalize_year(header_text: str) -> str:
    header_text = header_text.strip()
    header_text = re.sub(r"\s*\(.+\)$", "", header_text).strip()
    year_match = YEAR_VALUE_RE.match(header_text)
    if year_match:
        return year_match.group(1)
    return header_text


def clean_title(raw_title: str) -> str:
    title = raw_title.strip()
    title = re.sub(r"^\d{4}\s+", "", title)
    return title.strip()


def parse_markdown_entries(markdown_path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current_year = "Unknown Year"

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = YEAR_HEADER_RE.match(line)
        if header_match:
            header_text = header_match.group(1) or header_match.group(2) or ""
            current_year = normalize_year(header_text)
            continue

        if BULLET_RE.match(line):
            title = clean_title(BULLET_RE.sub("", line, count=1))
            if title:
                entries.append({"year": current_year, "title": title})
            continue

        inline_match = INLINE_PAPER_RE.match(line)
        if inline_match:
            title = clean_title(inline_match.group(1))
            if title:
                entries.append({"year": current_year, "title": title})

    return entries


def build_prompt(entry: dict[str, str]) -> str:
    year = entry["year"]
    title = entry["title"]
    return (
        "Find the matching AI/ML paper for this reference and return only the arXiv ID.\n"
        f"Year: {year}\n"
        f"Title: {title}\n"
    )


def load_completed_titles(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = record.get("title")
            if isinstance(title, str):
                completed.add(title)
    return completed


def call_exa_answer(api_base: str, api_key: str, prompt: str) -> dict[str, Any]:
    url = f"{api_base.rstrip('/')}/answer"
    payload = {
        "query": prompt,
        "text": True,
        "outputSchema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": ["string", "null"],
                    "description": (
                        "The arXiv identifier only, for example '1706.03762'. "
                        "Return null if no arXiv version exists or the match is unclear."
                    ),
                }
            },
            "required": ["arxiv_id"],
            "additionalProperties": False,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def process_entry(
    entry: dict[str, str],
    api_base: str,
    delay_seconds: float,
) -> dict[str, Any]:
    prompt = build_prompt(entry)
    record: dict[str, Any] = {
        "year": entry["year"],
        "title": entry["title"],
        "prompt": prompt,
        "endpoint": "/answer",
    }

    try:
        response_json = call_exa_answer(
            api_base=api_base,
            api_key=DEFAULT_API_KEY,
            prompt=prompt,
        )
        record["response"] = response_json.get("answer")
        record["arxiv_id"] = (
            response_json.get("answer", {}) or {}
        ).get("arxiv_id")
        record["citations"] = response_json.get("citations")
        record["raw_response"] = response_json
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        record["error"] = {
            "type": "HTTPError",
            "status": exc.code,
            "reason": exc.reason,
            "body": error_body,
        }
    except urllib.error.URLError as exc:
        record["error"] = {
            "type": "URLError",
            "reason": str(exc.reason),
        }
    except Exception as exc:  # noqa: BLE001
        record["error"] = {
            "type": type(exc).__name__,
            "reason": str(exc),
        }

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    return record


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    entries = parse_markdown_entries(input_path)
    if args.limit is not None:
        entries = entries[: args.limit]

    completed_titles = set()
    if not args.overwrite and not args.dry_run:
        completed_titles = load_completed_titles(output_path)
        entries = [entry for entry in entries if entry["title"] not in completed_titles]

    if args.dry_run:
        for index, entry in enumerate(entries, start=1):
            print(f"[{index}/{len(entries)}] {entry['year']} | {entry['title']}")
            print(build_prompt(entry))
            print()
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    with output_path.open(mode, encoding="utf-8") as handle:
        max_workers = max(1, min(args.max_concurrent, 7))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {}
            for index, entry in enumerate(entries, start=1):
                print(f"[{index}/{len(entries)}] Queueing: {entry['title']}", file=sys.stderr)
                future = executor.submit(
                    process_entry,
                    entry,
                    args.api_base,
                    args.delay_seconds,
                )
                future_to_meta[future] = (index, entry["title"])

            for future in concurrent.futures.as_completed(future_to_meta):
                index, title = future_to_meta[future]
                print(f"[{index}/{len(entries)}] Finished: {title}", file=sys.stderr)
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
