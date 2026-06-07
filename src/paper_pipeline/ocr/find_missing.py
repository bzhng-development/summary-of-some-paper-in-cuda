"""Stage 1: find papers in Neon that lack HF-rendered ("hf cli") markdown.

For every arxiv-pattern paper id in ``nextjs-ui_paper`` we probe the Hugging Face
paper-markdown endpoint, and — only for the ones HF can't serve — the arXiv HTML
endpoint. Each paper is bucketed:

- ``hf_ok``     : ``huggingface.co/papers/{id}.md`` returned 200. We already have a
                  good markdown body; nothing to do.
- ``html_only`` : HF 404/4xx but ``arxiv.org/html/{id}`` is 200. The body can be
                  scraped from arXiv HTML — no GPU/OCR needed.
- ``ocr_needed``: both endpoints non-200 (scanned PDF / no HTML twin). These are the
                  papers Stage 2 OCRs with chandra-ocr-2 on vLLM.

Detection keys on HTTP status, not body length: HF's 404 page is a fixed ~49.8k-char
SPA shell, while a real rendered body can be smaller, so length is not separable.

The classification log is append-only NDJSON (one complete record per id), so a run
is resume-safe: ids already present are skipped. Only the hf-missing subset ever
touches arXiv, and that subset is probed under a hard concurrency+interval throttle.

Usage::

    uv run python -m paper_pipeline.ocr.find_missing run
    uv run python -m paper_pipeline.ocr.find_missing run --limit 200   # smoke test
    uv run python -m paper_pipeline.ocr.find_missing report            # counts only
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Annotated, Final, Literal

import httpx
import typer
from loguru import logger
from pydantic import BaseModel
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from paper_pipeline.core.neon_db import TABLE, NeonDB

console = Console()

# arxiv ids: modern "2602.08025" / "2602.08025v3", or old "hep-th/0401001".
_MODERN_ARXIV: Final = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_ARXIV: Final = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}$")

_HF_MD_URL: Final = "https://huggingface.co/papers/{id}.md"
_ARXIV_HTML_URL: Final = "https://arxiv.org/html/{id}"
_USER_AGENT: Final = "paper-pipeline-ocr/1.0 (+find_missing)"

# Anonymous HF .md probing gets 429'd within seconds; an authenticated token lifts
# the limit. Without a token the prober still works but must crawl at low concurrency.
_HF_TOKEN: Final = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

# Statuses that mean "ask again later" rather than a real answer about the paper.
_TRANSIENT = frozenset({0, 429, 500, 502, 503, 504})

_OUT_DIR: Final = Path(__file__).resolve().parent / "out"
_LOG_PATH: Final = _OUT_DIR / "classification.jsonl"
_OCR_IDS_PATH: Final = _OUT_DIR / "ocr_needed.txt"
_HTML_IDS_PATH: Final = _OUT_DIR / "html_only.txt"
# arxiv-only scan log: {"id", "html_status"}. This is the AUTHORITATIVE OCR signal —
# a paper needs OCR iff it has no arXiv HTML twin (HF .md renders FROM arXiv HTML, so
# "no HTML" ⟹ "no full-text markdown anywhere"). Unthrottled, unlike the HF probe.
_ARXIV_SCAN_LOG: Final = _OUT_DIR / "arxiv_scan.jsonl"

type Bucket = Literal["hf_ok", "html_only", "ocr_needed"]


class Classification(BaseModel):
    """One fully-probed paper. ``html_status`` is ``None`` when HF already had it."""

    id: str
    hf_status: int
    html_status: int | None
    bucket: Bucket


def is_arxiv_id(paper_id: str) -> bool:
    return bool(_MODERN_ARXIV.match(paper_id) or _OLD_ARXIV.match(paper_id))


def fetch_candidate_ids() -> list[str]:
    """All arxiv-pattern ids from Neon, sorted newest-first."""
    db = NeonDB()
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {TABLE} ORDER BY id DESC")
        all_ids = [row[0] for row in cur.fetchall()]
    arxiv_ids = [pid for pid in all_ids if is_arxiv_id(pid)]
    logger.info(
        "Neon rows: {} total, {} arxiv-pattern, {} excluded",
        len(all_ids),
        len(arxiv_ids),
        len(all_ids) - len(arxiv_ids),
    )
    return arxiv_ids


def load_done_ids() -> set[str]:
    if not _LOG_PATH.exists():
        return set()
    done: set[str] = set()
    for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            done.add(json.loads(line)["id"])
    return done


class _Transient(Exception):
    """A probe got no definitive answer (429/5xx/network). Re-probe on a later run."""


async def _probe_status(
    client: httpx.AsyncClient,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    retries: int = 4,
) -> int:
    """HTTP status of ``url`` without downloading the body. Retries 429/5xx with backoff.

    HEAD is used where the server supports it (arXiv); GET is streamed and closed
    before the body transfers. Returns 0 only when every attempt raised.
    """
    last = 0
    for attempt in range(retries):
        try:
            if method == "HEAD":
                resp = await client.head(url, headers=headers)
                last = resp.status_code
            else:
                async with client.stream("GET", url, headers=headers) as resp:
                    last = resp.status_code
            if last in _TRANSIENT and last != 0:
                await asyncio.sleep(min(30, 3 * (attempt + 1)))
                continue
            return last
        except httpx.HTTPError as exc:
            last = 0
            logger.debug("probe error {} (attempt {}): {}", url, attempt, exc)
            await asyncio.sleep(min(20, 2 ** (attempt + 1)))
    return last


class _ArxivThrottle:
    """Bound arXiv to N concurrent requests with a minimum spacing between them."""

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


async def _classify_one(
    client: httpx.AsyncClient,
    paper_id: str,
    hf_sem: asyncio.Semaphore,
    arxiv_throttle: _ArxivThrottle,
) -> Classification:
    hf_headers = {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else None
    async with hf_sem:
        hf_status = await _probe_status(client, _HF_MD_URL.format(id=paper_id), headers=hf_headers)
    if hf_status == 200:
        return Classification(id=paper_id, hf_status=hf_status, html_status=None, bucket="hf_ok")
    if hf_status in _TRANSIENT:
        raise _Transient(f"hf {hf_status} for {paper_id}")

    async with arxiv_throttle:
        html_status = await _probe_status(client, _ARXIV_HTML_URL.format(id=paper_id), method="HEAD")
    if html_status in _TRANSIENT:
        raise _Transient(f"arxiv {html_status} for {paper_id}")
    bucket: Bucket = "html_only" if html_status == 200 else "ocr_needed"
    return Classification(id=paper_id, hf_status=hf_status, html_status=html_status, bucket=bucket)


async def _run_async(
    ids: list[str],
    hf_concurrency: int,
    arxiv_concurrency: int,
    arxiv_min_interval: float,
) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    hf_sem = asyncio.Semaphore(hf_concurrency)
    arxiv_throttle = _ArxivThrottle(arxiv_concurrency, arxiv_min_interval)
    write_lock = asyncio.Lock()
    counts: Counter[str] = Counter()

    limits = httpx.Limits(max_connections=hf_concurrency + arxiv_concurrency, max_keepalive_connections=32)
    timeout = httpx.Timeout(30.0, connect=15.0)
    async with httpx.AsyncClient(
        follow_redirects=True, headers={"User-Agent": _USER_AGENT}, limits=limits, timeout=timeout
    ) as client:
        with (
            _LOG_PATH.open("a", encoding="utf-8") as log_f,
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress,
        ):
            task = progress.add_task("probing", total=len(ids))

            def _beat() -> None:
                # Plain stdout heartbeat so a background runner sees output and
                # doesn't reap the job for idleness (rich's live display is silent on a pipe).
                done = sum(counts.values())
                if done % 200 == 0:
                    print(
                        f"[heartbeat] {done}/{len(ids)} hf_ok={counts['hf_ok']} "
                        f"html={counts['html_only']} ocr={counts['ocr_needed']} throttled={counts['throttled']}",
                        flush=True,
                    )

            async def worker(paper_id: str) -> None:
                try:
                    result = await _classify_one(client, paper_id, hf_sem, arxiv_throttle)
                except _Transient:
                    # No definitive answer (rate-limited / 5xx). Don't write a record —
                    # the id stays absent from the log and is re-probed on the next run.
                    async with write_lock:
                        counts["throttled"] += 1
                        progress.update(task, advance=1)
                        _beat()
                    return
                async with write_lock:
                    log_f.write(json.dumps(result.model_dump(), sort_keys=True) + "\n")
                    log_f.flush()
                    counts[result.bucket] += 1
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"hf_ok={counts['hf_ok']} html={counts['html_only']} "
                            f"ocr={counts['ocr_needed']} throttled={counts['throttled']}"
                        ),
                    )
                    _beat()

            await asyncio.gather(*(worker(pid) for pid in ids))
    if counts["throttled"]:
        console.print(f"[yellow]{counts['throttled']:,} ids were throttled/transient — re-run to pick them up.[/]")


def _emit_id_lists_and_report() -> None:
    """Re-read the full log, write the ocr/html id lists, and print a summary table."""
    if not _LOG_PATH.exists():
        console.print("[yellow]No classification log yet — run the probe first.[/]")
        raise typer.Exit(1)

    counts: Counter[str] = Counter()
    ocr_ids: list[str] = []
    html_ids: list[str] = []
    for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = Classification.model_validate_json(line)
        counts[rec.bucket] += 1
        if rec.bucket == "ocr_needed":
            ocr_ids.append(rec.id)
        elif rec.bucket == "html_only":
            html_ids.append(rec.id)

    _OCR_IDS_PATH.write_text("\n".join(sorted(ocr_ids)) + ("\n" if ocr_ids else ""), encoding="utf-8")
    _HTML_IDS_PATH.write_text("\n".join(sorted(html_ids)) + ("\n" if html_ids else ""), encoding="utf-8")

    total = sum(counts.values())
    table = Table(title=f"Classification of {total:,} arxiv papers")
    table.add_column("bucket", style="cyan")
    table.add_column("count", justify="right", style="magenta")
    table.add_column("share", justify="right")
    table.add_column("meaning")
    meanings = {
        "hf_ok": "HF already serves markdown — skip",
        "html_only": "no HF md, but arXiv HTML exists — scrape (no GPU)",
        "ocr_needed": "no HF md, no arXiv HTML — OCR the PDF (Stage 2)",
    }
    for bucket in ("hf_ok", "html_only", "ocr_needed"):
        n = counts.get(bucket, 0)
        share = f"{100 * n / total:.1f}%" if total else "—"
        table.add_row(bucket, f"{n:,}", share, meanings[bucket])
    console.print(table)
    console.print(f"OCR-needed id list  -> {_OCR_IDS_PATH}  ({len(ocr_ids):,} ids)")
    console.print(f"HTML-only id list   -> {_HTML_IDS_PATH}  ({len(html_ids):,} ids)")


app = typer.Typer(add_completion=False, help="Stage 1: classify papers by markdown availability.")


@app.command()
def run(
    limit: Annotated[int | None, typer.Option(help="Only probe the first N candidates (smoke test).")] = None,
    hf_concurrency: Annotated[int, typer.Option(help="Concurrent HF .md probes (needs HF_TOKEN to go high).")] = 24,
    arxiv_concurrency: Annotated[int, typer.Option(help="Concurrent arXiv HEAD probes (fast/cheap).")] = 24,
    arxiv_min_interval: Annotated[float, typer.Option(help="Min seconds between arXiv probes.")] = 0.0,
) -> None:
    """Probe HF + arXiv for every arxiv-pattern paper and append to the classification log."""
    if not _HF_TOKEN:
        console.print(
            "[yellow]No HF_TOKEN in env — anonymous HF .md probing gets 429'd fast; forcing hf_concurrency=4.[/]"
        )
        hf_concurrency = min(hf_concurrency, 4)
    candidates = fetch_candidate_ids()
    if limit is not None:
        candidates = candidates[:limit]
    done = load_done_ids()
    todo = [pid for pid in candidates if pid not in done]
    console.print(
        f"[bold]{len(candidates):,}[/] candidates, [green]{len(done):,}[/] already classified, [yellow]{len(todo):,}[/] to probe"
    )
    if todo:
        asyncio.run(_run_async(todo, hf_concurrency, arxiv_concurrency, arxiv_min_interval))
    _emit_id_lists_and_report()


@app.command()
def report() -> None:
    """Re-emit the id lists and summary table from the existing log (no probing)."""
    _emit_id_lists_and_report()


def _load_arxiv_scan_done() -> set[str]:
    if not _ARXIV_SCAN_LOG.exists():
        return set()
    done: set[str] = set()
    for line in _ARXIV_SCAN_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            done.add(json.loads(line)["id"])
    return done


async def _arxiv_scan_async(ids: list[str], concurrency: int) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    done_count = 0
    ocr_count = 0
    timeout = httpx.Timeout(20.0, connect=10.0)
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=32)
    client_cm = httpx.AsyncClient(
        follow_redirects=True, headers={"User-Agent": _USER_AGENT}, limits=limits, timeout=timeout
    )
    async with client_cm as client:
        with _ARXIV_SCAN_LOG.open("a", encoding="utf-8") as log_f:

            async def worker(paper_id: str) -> None:
                nonlocal done_count, ocr_count
                async with sem:
                    status = await _probe_status(client, _ARXIV_HTML_URL.format(id=paper_id), method="HEAD")
                if status in _TRANSIENT:
                    return  # transient — leave absent, re-probe next run
                async with write_lock:
                    log_f.write(json.dumps({"id": paper_id, "html_status": status}, sort_keys=True) + "\n")
                    log_f.flush()
                    done_count += 1
                    if status != 200:
                        ocr_count += 1
                    if done_count % 500 == 0:
                        print(f"[arxiv-scan] {done_count}/{len(ids)} ocr_needed={ocr_count}", flush=True)

            await asyncio.gather(*(worker(pid) for pid in ids))


def _emit_ocr_list_from_arxiv_scan() -> None:
    """ocr_needed = ids whose arXiv HTML probe was not 200. Authoritative OCR set."""
    if not _ARXIV_SCAN_LOG.exists():
        console.print("[yellow]No arxiv-scan log yet — run `arxiv-scan` first.[/]")
        raise typer.Exit(1)
    latest: dict[str, int] = {}
    for line in _ARXIV_SCAN_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rec = json.loads(line)
            latest[rec["id"]] = rec["html_status"]
    ocr_ids = sorted(pid for pid, st in latest.items() if st != 200)
    html_ids = sorted(pid for pid, st in latest.items() if st == 200)
    _OCR_IDS_PATH.write_text("\n".join(ocr_ids) + ("\n" if ocr_ids else ""), encoding="utf-8")
    total = len(latest)
    table = Table(title=f"arXiv-HTML scan of {total:,} papers")
    table.add_column("bucket", style="cyan")
    table.add_column("count", justify="right", style="magenta")
    table.add_column("share", justify="right")
    table.add_row("has_html (no OCR)", f"{len(html_ids):,}", f"{100 * len(html_ids) / total:.1f}%" if total else "—")
    table.add_row("ocr_needed", f"{len(ocr_ids):,}", f"{100 * len(ocr_ids) / total:.1f}%" if total else "—")
    console.print(table)
    console.print(f"OCR-needed id list -> {_OCR_IDS_PATH}  ({len(ocr_ids):,} ids)")


@app.command(name="arxiv-scan")
def arxiv_scan(
    limit: Annotated[int | None, typer.Option(help="Only scan the first N candidates.")] = None,
    concurrency: Annotated[int, typer.Option(help="Concurrent arXiv HEAD probes (fast, unthrottled).")] = 32,
) -> None:
    """Determine the OCR set directly: ocr_needed = papers with no arXiv HTML twin.

    Fast and unthrottled (HEAD probes), unlike the HF-dependent ``run``. This is the
    authoritative way to build ``ocr_needed.txt`` for Stage 2.
    """
    candidates = fetch_candidate_ids()
    if limit is not None:
        candidates = candidates[:limit]
    done = _load_arxiv_scan_done()
    todo = [pid for pid in candidates if pid not in done]
    console.print(
        f"[bold]{len(candidates):,}[/] candidates, [green]{len(done):,}[/] scanned, [yellow]{len(todo):,}[/] to probe"
    )
    if todo:
        asyncio.run(_arxiv_scan_async(todo, concurrency))
    _emit_ocr_list_from_arxiv_scan()


if __name__ == "__main__":
    app()
