"""Retry Substack URLs that failed in bulk_summarize.

Walks bulk_log.jsonl, finds URLs without a summary.md on disk, retries
each with workers=1 + a backoff sleep between attempts, and patches the
get_metadata None case with a retry loop.
"""

from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

from substack_api import Post

from paper_pipeline.substack.html_to_text import html_to_text
from paper_pipeline.substack.llm import call as llm_call
from paper_pipeline.substack.llm import make_client
from paper_pipeline.substack.prompts import P1_SYSTEM, P2_SYSTEM, p1_user, p2_user

URLS_PATH = Path(__file__).resolve().parent / "all_posts.json"
BULK_DIR = Path(__file__).resolve().parent / "out" / "bulk"
LOG_PATH = Path(__file__).resolve().parent / "out" / "retry_log.jsonl"


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/p/")[-1] or "post"


def append_log(rec: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def already_done(slug: str) -> bool:
    return (BULK_DIR / slug / "summary.md").is_file()


def fetch_with_retry(url: str, attempts: int = 5):
    """Hit Substack with backoff. Returns (meta, html) or raises."""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            post = Post(url)
            meta = post.get_metadata()
            if meta is None:
                raise RuntimeError("get_metadata returned None (rate limited?)")
            html = post.get_content() or ""
            return meta, html
        except Exception as e:
            last_err = e
            wait = 8 * (2**i) + random.uniform(0, 4)  # 8s, 16s, 32s, 64s, 128s
            print(f"      attempt {i + 1}/{attempts} failed: {type(e).__name__}: {str(e)[:80]} — sleep {wait:.1f}s")
            time.sleep(wait)
    raise last_err or RuntimeError("exhausted retries")


def summarize_one(client, url: str, sleep_between: float) -> dict:
    slug = slug_from_url(url)
    art_dir = BULK_DIR / slug
    art_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    rec: dict = {"url": url, "slug": slug}

    try:
        meta, html = fetch_with_retry(url)
        text = html_to_text(html)
        title = meta.get("title") or ""
        rec["title"] = title
        rec["wordcount_meta"] = meta.get("wordcount")
        rec["transcript_chars"] = len(text)
        rec["post_type"] = meta.get("type")

        (art_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        (art_dir / "transcript.txt").write_text(text, encoding="utf-8")

        if not text.strip():
            rec["status"] = "skip_empty"
            return rec

        t0 = time.perf_counter()
        p1_text, p1_usage = llm_call(
            client, system=P1_SYSTEM, user=p1_user(title, text), temperature=1.0, top_p=1.0, max_tokens=32000
        )
        rec["p1_seconds"] = round(time.perf_counter() - t0, 1)
        rec["p1_usage"] = p1_usage
        rec["p1_chars"] = len(p1_text)
        (art_dir / "p1_draft.md").write_text(p1_text, encoding="utf-8")

        t0 = time.perf_counter()
        p2_text, p2_usage = llm_call(
            client, system=P2_SYSTEM, user=p2_user(title, text, p1_text), temperature=1.0, top_p=1.0, max_tokens=32000
        )
        rec["p2_seconds"] = round(time.perf_counter() - t0, 1)
        rec["p2_usage"] = p2_usage
        rec["p2_chars"] = len(p2_text)
        (art_dir / "p2_final.md").write_text(p2_text, encoding="utf-8")

        final = f"<sup>Source: <{url}></sup>\n\n{p2_text}\n"
        (art_dir / "summary.md").write_text(final, encoding="utf-8")

        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)
        rec["traceback"] = traceback.format_exc()[-1500:]
    finally:
        rec["total_seconds"] = round(time.perf_counter() - t_start, 1)
        if sleep_between > 0:
            time.sleep(sleep_between)
    return rec


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=10.0, help="Seconds to sleep after each URL (rate-limit politeness)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    urls: list[str] = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    todo = [u for u in urls if not already_done(slug_from_url(u))]
    print(f"total={len(urls)}  done={len(urls) - len(todo)}  todo={len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"  truncated to {len(todo)} (--limit)")

    client = make_client()
    n_ok = n_err = 0
    t0 = time.perf_counter()
    for i, url in enumerate(todo, 1):
        short = url.rsplit("/", 1)[-1][:60]
        print(f"\n[{i:>3}/{len(todo)}] {short}")
        rec = summarize_one(client, url, args.sleep)
        append_log(rec)
        if rec["status"] == "ok":
            n_ok += 1
            print(f"      OK p1={rec.get('p1_chars')} p2={rec.get('p2_chars')} t={rec.get('total_seconds')}s")
        else:
            n_err += 1
            print(f"      {rec['status']}: {rec.get('error_type', '?')}: {str(rec.get('error', ''))[:100]}")
        rate = i / max(time.perf_counter() - t0, 1e-3)
        eta = (len(todo) - i) / max(rate, 1e-3) / 60
        print(f"      tally ok={n_ok} err={n_err} | eta={eta:.1f}min")

    print(f"\ndone. ok={n_ok}  err={n_err}  elapsed={(time.perf_counter() - t0) / 60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
