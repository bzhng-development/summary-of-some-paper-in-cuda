"""Run the 2-prompt DeepSeek V4-Pro summary over every Peterman Post.

Reads all_posts.json, summarizes anything that doesn't already have a final
output, in parallel with a small worker pool. Robust to errors and resumable.

Requires SSH tunnel: localhost:8000 → brayden@95.133.253.79:8000.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from substack_api import Post

from paper_pipeline.substack.html_to_text import html_to_text
from paper_pipeline.substack.llm import call as llm_call
from paper_pipeline.substack.llm import make_client
from paper_pipeline.substack.prompts import P1_SYSTEM, P2_SYSTEM, p1_user, p2_user

URLS_PATH = Path(__file__).resolve().parent / "all_posts.json"
BULK_DIR = Path(__file__).resolve().parent / "out" / "bulk"
LOG_PATH = Path(__file__).resolve().parent / "out" / "bulk_log.jsonl"
INDEX_PATH = Path(__file__).resolve().parent / "out" / "bulk_index.json"

LOCK = threading.Lock()


def slug_from_url(url: str) -> str:
    return url.rstrip("/").split("/p/")[-1] or "post"


def append_log(rec: dict) -> None:
    with LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def already_done(slug: str) -> bool:
    return (BULK_DIR / slug / "summary.md").is_file()


def summarize(client, url: str, max_tokens: int, temperature: float, top_p: float) -> dict:
    """Run the full 2-prompt pipeline on one URL. Returns a result record."""
    slug = slug_from_url(url)
    art_dir = BULK_DIR / slug
    art_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    rec: dict = {"url": url, "slug": slug, "start_ts": t_start}

    try:
        post = Post(url)
        meta = post.get_metadata()
        html = post.get_content() or ""
        text = html_to_text(html)
        title = meta.get("title") or ""
        rec["title"] = title
        rec["wordcount_meta"] = meta.get("wordcount")
        rec["transcript_chars"] = len(text)
        rec["post_type"] = meta.get("type")
        rec["audience"] = meta.get("audience")
        rec["post_date"] = meta.get("post_date")
        rec["podcast_duration"] = meta.get("podcast_duration")

        (art_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (art_dir / "transcript.txt").write_text(text, encoding="utf-8")

        if not text.strip():
            rec["status"] = "skip_empty"
            return rec

        # P1
        t0 = time.perf_counter()
        p1_text, p1_usage = llm_call(
            client,
            system=P1_SYSTEM,
            user=p1_user(title, text),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        rec["p1_seconds"] = round(time.perf_counter() - t0, 1)
        rec["p1_usage"] = p1_usage
        rec["p1_chars"] = len(p1_text)
        (art_dir / "p1_draft.md").write_text(p1_text, encoding="utf-8")

        # P2
        t0 = time.perf_counter()
        p2_text, p2_usage = llm_call(
            client,
            system=P2_SYSTEM,
            user=p2_user(title, text, p1_text),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        rec["p2_seconds"] = round(time.perf_counter() - t0, 1)
        rec["p2_usage"] = p2_usage
        rec["p2_chars"] = len(p2_text)
        (art_dir / "p2_final.md").write_text(p2_text, encoding="utf-8")

        # Final summary file (P2 prefixed with source link).
        final = f"<sup>Source: <{url}></sup>\n\n{p2_text}\n"
        (art_dir / "summary.md").write_text(final, encoding="utf-8")

        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "error"
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)
        rec["traceback"] = traceback.format_exc()[-2000:]
    finally:
        rec["total_seconds"] = round(time.perf_counter() - t_start, 1)

    return rec


def build_index() -> None:
    """Walk BULK_DIR and produce a JSON index of finished summaries."""
    items = []
    for art_dir in sorted(BULK_DIR.iterdir()):
        if not art_dir.is_dir():
            continue
        summary = art_dir / "summary.md"
        meta = art_dir / "meta.json"
        if not summary.is_file() or not meta.is_file():
            continue
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append(
            {
                "slug": art_dir.name,
                "title": m.get("title"),
                "subtitle": m.get("subtitle"),
                "type": m.get("type"),
                "audience": m.get("audience"),
                "post_date": m.get("post_date"),
                "wordcount": m.get("wordcount"),
                "podcast_duration": m.get("podcast_duration"),
                "summary_chars": summary.stat().st_size,
                "url": f"https://www.developing.dev/p/{art_dir.name}",
                "canonical_url": m.get("canonical_url"),
            }
        )
    items.sort(key=lambda r: r.get("post_date") or "", reverse=True)
    INDEX_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"index: {len(items)} items → {INDEX_PATH}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument(
        "--rebuild-index", action="store_true", help="Only rebuild the index from existing outputs and exit."
    )
    args = ap.parse_args()

    BULK_DIR.mkdir(parents=True, exist_ok=True)

    if args.rebuild_index:
        build_index()
        return 0

    urls: list[str] = json.loads(URLS_PATH.read_text(encoding="utf-8"))
    todo = [u for u in urls if not already_done(slug_from_url(u))]
    print(f"total={len(urls)}  done={len(urls) - len(todo)}  todo={len(todo)}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"  truncated to {len(todo)} (--limit)")

    client = make_client()

    n_ok = n_err = n_skip = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(summarize, client, u, args.max_tokens, args.temperature, args.top_p): u for u in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            url = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                rec = {"url": url, "status": "fatal", "error": str(e)}
            append_log(rec)
            status = rec.get("status", "?")
            if status == "ok":
                n_ok += 1
            elif status == "skip_empty":
                n_skip += 1
            else:
                n_err += 1
            t_elap = time.perf_counter() - t0
            rate = i / max(t_elap, 1e-3)
            eta = (len(todo) - i) / max(rate, 1e-3) / 60
            short = url.rsplit("/", 1)[-1][:60]
            extra = ""
            if status == "ok":
                extra = f"p1={rec.get('p1_chars')} p2={rec.get('p2_chars')} t={rec.get('total_seconds')}s"
            elif status == "error":
                extra = f"{rec.get('error_type')}: {str(rec.get('error', ''))[:80]}"
            print(f"[{i:>3}/{len(todo)}] {status:5s} {short:60s} {extra} | eta={eta:.1f}min")

    build_index()
    print(f"\ntotal {time.perf_counter() - t0:.1f}s  ok={n_ok}  err={n_err}  skip={n_skip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
