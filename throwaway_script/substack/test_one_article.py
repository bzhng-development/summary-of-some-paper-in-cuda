"""Probe the substack-api lib against one Ryan Peterman podcast post.

Goal: confirm we can pull the *full* article body (no truncation) for free posts.
James Cowling (ex-Dropbox most senior eng) episode chosen as test target — it's
the most recent podcast-interview transcript on developing.dev.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from substack_api import Newsletter, Post

PUB_URL = "https://www.developing.dev"
TEST_POST_URL = "https://www.developing.dev/p/dropboxs-former-most-senior-eng-building"
OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Newsletter probe: {PUB_URL}")
    print("=" * 60)
    nl = Newsletter(PUB_URL)

    # How many posts total?
    recent = nl.get_posts(limit=5)
    print(f"recent posts (limit=5): {len(recent)}")
    for p in recent[:5]:
        print(f"  - {p.url}")

    print()
    print("=" * 60)
    print(f"Post probe: {TEST_POST_URL}")
    print("=" * 60)
    post = Post(TEST_POST_URL)

    meta = post.get_metadata()
    keys_sample = list(meta.keys())[:30]
    print(f"meta keys ({len(meta)}): {keys_sample}")
    print(f"title: {meta.get('title')}")
    print(f"subtitle: {meta.get('subtitle')}")
    print(f"is_paywalled: {meta.get('audience')}")
    print(f"published: {meta.get('post_date')}")
    print(f"type: {meta.get('type')}")
    print(f"wordcount: {meta.get('wordcount')}")
    print(f"body_html chars: {len(meta.get('body_html') or '')}")

    try:
        is_pw = post.is_paywalled()
        print(f"is_paywalled() -> {is_pw}")
    except Exception as e:
        print(f"is_paywalled() raised: {type(e).__name__}: {e}")

    content = post.get_content()
    print(f"get_content() len: {len(content) if isinstance(content, str) else type(content)}")
    if isinstance(content, str):
        print(f"content[:400]: {content[:400]}")
        print(f"content[-400:]: {content[-400:]}")

    # Save everything for inspection.
    (OUT_DIR / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if isinstance(content, str):
        (OUT_DIR / "content.html").write_text(content, encoding="utf-8")
    print(f"\nWrote: {OUT_DIR / 'meta.json'} and {OUT_DIR / 'content.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
