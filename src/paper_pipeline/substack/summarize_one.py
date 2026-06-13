"""Run the 2-prompt DeepSeek V4-Pro summary on ONE Substack post.

Usage:
    uv run python src/paper_pipeline/substack/summarize_one.py <post_url>

Requires an SSH tunnel to brayden@95.133.253.79:8000 → localhost:8000 (the
shared cluster vllm — same endpoint T45 is using).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from substack_api import Post

from paper_pipeline.substack.html_to_text import html_to_text
from paper_pipeline.substack.llm import call as llm_call
from paper_pipeline.substack.llm import make_client
from paper_pipeline.substack.prompts import P1_SYSTEM, P2_SYSTEM, p1_user, p2_user


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/p/")[-1] or "post"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Full Substack post URL")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "out")
    ap.add_argument("--iter", type=str, default=None, help="If set, outputs go under <out-dir>/iter<X>/<slug>/")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32000)
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    slug = slug_from_url(args.url)
    base = args.out_dir / f"iter{args.iter}" if args.iter else args.out_dir
    base.mkdir(parents=True, exist_ok=True)
    art_dir = base / slug
    art_dir.mkdir(exist_ok=True)

    print(f"[1/4] fetch post: {args.url}")
    post = Post(args.url)
    meta = post.get_metadata()
    title = meta.get("title") or ""
    wc = meta.get("wordcount")
    html = post.get_content()
    print(f"      title: {title}")
    print(f"      wordcount(meta): {wc}  html_chars: {len(html) if isinstance(html, str) else 'n/a'}")

    (art_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (art_dir / "content.html").write_text(html or "", encoding="utf-8")

    print("[2/4] HTML → text")
    text = html_to_text(html or "")
    (art_dir / "transcript.txt").write_text(text, encoding="utf-8")
    print(f"      transcript chars: {len(text)}")

    client = make_client()

    print("[3/4] P1 — chapter-narrative draft")
    t0 = time.perf_counter()
    p1_text, p1_usage = llm_call(
        client,
        system=P1_SYSTEM,
        user=p1_user(title, text),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    dt1 = time.perf_counter() - t0
    print(f"      p1: {len(p1_text)} chars  usage={p1_usage}  ({dt1:.1f}s)")
    (art_dir / "p1_draft.md").write_text(p1_text, encoding="utf-8")
    (art_dir / "p1_usage.json").write_text(json.dumps(p1_usage, indent=2), encoding="utf-8")

    print("[4/4] P2 — polish + capstone (subsumes P1)")
    t0 = time.perf_counter()
    p2_text, p2_usage = llm_call(
        client,
        system=P2_SYSTEM,
        user=p2_user(title, text, p1_text),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    dt2 = time.perf_counter() - t0
    print(f"      p2: {len(p2_text)} chars  usage={p2_usage}  ({dt2:.1f}s)")
    (art_dir / "p2_final.md").write_text(p2_text, encoding="utf-8")
    (art_dir / "p2_usage.json").write_text(json.dumps(p2_usage, indent=2), encoding="utf-8")

    # The final summary is P2's output (subsumes P1). We just prepend the source link.
    final = f"<sup>Source: <{args.url}></sup>\n\n{p2_text}\n"
    (art_dir / "summary.md").write_text(final, encoding="utf-8")
    print(f"\nDone. Outputs in {art_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
