"""Enumerate every Peterman Post URL via the substack-api.

Just URLs — no per-post metadata fetches (they rate-limit). The summarizer
script fetches metadata when it processes each post.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from substack_api import Newsletter

PUB_URL = "https://www.developing.dev"
OUT = Path(__file__).resolve().parent / "all_posts.json"


def main() -> int:
    nl = Newsletter(PUB_URL)
    posts = nl.get_posts(limit=2000)
    urls = [p.url for p in posts]
    print(f"got {len(urls)} URLs")
    OUT.write_text(json.dumps(urls, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
