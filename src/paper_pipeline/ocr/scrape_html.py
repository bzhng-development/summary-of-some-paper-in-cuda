"""Stage 1.5: scrape arXiv HTML -> markdown for the ``html_only`` bucket (no GPU).

Most papers HF can't serve still have an arXiv HTML twin (LaTeXML output). For those we
skip OCR entirely and convert the HTML to markdown locally. Math is preserved without
pandoc: LaTeXML stores the original LaTeX in each ``<math alttext="...">``, so we splice
that back as ``$...$`` / ``$$...$$`` before running markdownify on the article body.

Source is Stage 1's ``out/html_only.txt``; the sink is ``out/html_out.jsonl`` with the
same ``arxiv_id`` + ``markdown`` + ``status`` shape Stage 3 (``ingest_ocr.py``) already
ingests — so the two fill-in paths (HTML scrape here, PDF OCR on the GPU box) converge on
one ingest.

Usage::

    uv run python -m paper_pipeline.ocr.scrape_html run
    uv run python -m paper_pipeline.ocr.scrape_html run --limit 50   # smoke test
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Final

import httpx
import typer
from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger
from markdownify import markdownify
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

console = Console()

_ARXIV_HTML_URL: Final = "https://arxiv.org/html/{id}"
_ARXIV_ABS_URL: Final = "https://arxiv.org/abs/{id}"
_USER_AGENT: Final = "paper-pipeline-ocr/1.0 (+scrape_html)"

_OUT_DIR: Final = Path(__file__).resolve().parent / "out"
_HTML_IDS_PATH: Final = _OUT_DIR / "html_only.txt"
_OUT_PATH: Final = _OUT_DIR / "html_out.jsonl"

_BLANKS = re.compile(r"\n{3,}")
_STRIP_TAGS = ("script", "style", "noscript", "form")


class HtmlRecord(BaseModel):
    arxiv_id: str
    paper_url: str
    source: str = "arxiv_html"
    status: str  # "success" | "error"
    fetched_at: str
    num_chars: int
    markdown: str
    error_message: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def html_to_markdown(html: str) -> str:
    """Extract the LaTeXML article body and convert it to markdown, math intact."""
    soup = BeautifulSoup(html, "lxml")

    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Splice LaTeX back in from each <math alttext="..."> before it becomes tag soup.
    for math in soup.find_all("math"):
        if not isinstance(math, Tag):
            continue
        latex = (math.get("alttext") or "").strip()
        if not latex:
            math.decompose()
            continue
        display = math.get("display") == "block"
        math.replace_with(NavigableString(f"\n$$\n{latex}\n$$\n" if display else f"${latex}$"))

    article = soup.select_one("article.ltx_document") or soup.find("article") or soup.find("main") or soup.body
    fragment = str(article) if article is not None else html
    md = markdownify(fragment, heading_style="ATX", escape_asterisks=False, escape_underscores=False)
    return _BLANKS.sub("\n\n", md).strip()


class _Throttle:
    """N concurrent requests with a minimum spacing between starts."""

    def __init__(self, concurrency: int, min_interval: float) -> None:
        self._sem = asyncio.Semaphore(concurrency)
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def __aenter__(self) -> None:
        await self._sem.acquire()
        async with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

    async def __aexit__(self, *_: object) -> None:
        self._sem.release()


async def _scrape_one(client: httpx.AsyncClient, arxiv_id: str, throttle: _Throttle, min_chars: int) -> HtmlRecord:
    base = {"arxiv_id": arxiv_id, "paper_url": _ARXIV_ABS_URL.format(id=arxiv_id), "fetched_at": _utc_now_iso()}
    try:
        async with throttle:
            resp = await client.get(_ARXIV_HTML_URL.format(id=arxiv_id))
        if resp.status_code != 200:
            return HtmlRecord(
                **base, status="error", num_chars=0, markdown="", error_message=f"HTTP {resp.status_code}"
            )
        markdown = await asyncio.to_thread(html_to_markdown, resp.text)
    except Exception as exc:
        return HtmlRecord(**base, status="error", num_chars=0, markdown="", error_message=str(exc))

    if len(markdown) < min_chars:
        return HtmlRecord(
            **base, status="error", num_chars=len(markdown), markdown="", error_message="markdown too short"
        )
    return HtmlRecord(**base, status="success", num_chars=len(markdown), markdown=markdown)


def _load_ids(ids_file: Path) -> list[str]:
    return [ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["arxiv_id"])
            except json.JSONDecodeError, KeyError:
                continue
    return done


async def _run_async(ids: list[str], concurrency: int, min_interval: float, min_chars: int) -> dict[str, int]:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    throttle = _Throttle(concurrency, min_interval)
    write_lock = asyncio.Lock()
    counts = {"success": 0, "error": 0}

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(60.0, connect=15.0)
    async with httpx.AsyncClient(
        follow_redirects=True, headers={"User-Agent": _USER_AGENT}, limits=limits, timeout=timeout
    ) as client:
        with (
            _OUT_PATH.open("a", encoding="utf-8") as sink,
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress,
        ):
            task = progress.add_task("scraping", total=len(ids))

            async def worker(arxiv_id: str) -> None:
                record = await _scrape_one(client, arxiv_id, throttle, min_chars)
                async with write_lock:
                    sink.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
                    sink.flush()
                    counts[record.status] += 1
                    progress.update(task, advance=1, description=f"ok={counts['success']} err={counts['error']}")

            await asyncio.gather(*(worker(pid) for pid in ids))
    return counts


app = typer.Typer(add_completion=False, help="Stage 1.5: scrape arXiv HTML -> markdown (html_only bucket).")


@app.command()
def run(
    ids_file: Annotated[Path, typer.Option("--ids-file", help="html_only id list from Stage 1.")] = _HTML_IDS_PATH,
    limit: Annotated[int | None, typer.Option(help="Only scrape the first N (smoke test).")] = None,
    concurrency: Annotated[int, typer.Option(help="Concurrent arXiv HTML fetches.")] = 8,
    min_interval: Annotated[float, typer.Option(help="Min seconds between fetch starts.")] = 0.3,
    min_chars: Annotated[int, typer.Option(help="Reject markdown shorter than this as failed.")] = 500,
) -> None:
    """Convert every html_only paper's arXiv HTML to markdown, appending to html_out.jsonl."""
    if not ids_file.exists():
        console.print(f"[red]{ids_file} not found — run Stage 1 (find_missing) first.[/]")
        raise typer.Exit(1)
    ids = _load_ids(ids_file)
    if limit is not None:
        ids = ids[:limit]
    done = _load_done(_OUT_PATH)
    todo = [pid for pid in ids if pid not in done]
    console.print(
        f"{len(ids):,} html_only ids, {len(done & set(ids)):,} done, [yellow]{len(todo):,} to scrape[/] -> {_OUT_PATH}"
    )
    if not todo:
        console.print("[green]nothing to do[/]")
        return
    counts = asyncio.run(_run_async(todo, concurrency, min_interval, min_chars))
    logger.success("scraped {} ok, {} failed -> {}", counts["success"], counts["error"], _OUT_PATH)
    console.print(f"[bold green]done[/] ok={counts['success']:,} err={counts['error']:,}")
    console.print(f"ingest with: uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl {_OUT_PATH}")


if __name__ == "__main__":
    app()
