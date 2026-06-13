# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chandra-ocr",
#     "pypdfium2",
#     "openai",
#     "httpx",
#     "typer",
#     "rich",
#     "pydantic",
# ]
# ///
"""Stage 2: OCR arXiv PDFs to markdown with chandra-ocr-2 on vLLM — two decoupled phases.

The GPU must never wait on arXiv, and the vLLM server (which holds ~350 concurrent
sequences on one B300) must be saturated — so this is split:

  fetch : download every PDF in the id list to a local dir. Parallel, retry/backoff,
          resume-safe (skip existing). NO GPU — run it AHEAD of the OCR phase so all
          arXiv I/O is done before the GPU starts (or overlaps, never blocks it).

  ocr   : async pipeline. Submits EVERY page of EVERY downloaded paper to the vLLM
          OpenAI endpoint under ONE global semaphore (default 256 in-flight), renders
          pages in a CPU threadpool, parses each page with chandra's HTML->markdown,
          reassembles per paper, appends resume-safe records. Pushes the server to its
          concurrency ceiling instead of the package's serial one-paper-at-a-time loop.

Reuses chandra's exact request recipe (``OCR_LAYOUT_PROMPT``, ``scale_to_fit``,
PNG-base64 ``image_url`` + text, temp=0/top_p=0.1, repeat-token retry) and its output
parser, so quality matches the package — only the concurrency model differs.

Usage (inside the GPU box's container, vLLM already serving on :8000)::

    python3 chandra_ocr.py fetch --ids-file ocr_needed.txt --pdf-dir pdfs --concurrency 16
    python3 chandra_ocr.py ocr   --pdf-dir pdfs --out ocr_out.jsonl \
        --vllm-base-url http://localhost:8000/v1 --concurrency 256
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import httpx
import pypdfium2 as pdfium
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

console = Console()

MODEL_NAME = "chandra"  # --served-model-name on the vLLM side
PROMPT_TYPE = "ocr_layout"
USER_AGENT = "paper-pipeline-ocr/2.0 (chandra)"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{id}.pdf"
ARXIV_ABS_URL = "https://arxiv.org/abs/{id}"

# Exception groups referenced by NAME in `except` clauses below. This script runs on
# the GPU box's Python 3.12 container, but the repo's ruff target is py314 — and on
# py314 `ruff format` rewrites `except (A, B):` to the bare-tuple `except A, B:`, which
# is a SyntaxError on 3.12 (and uncatchable by a local 3.14 `ast.parse`). Binding the
# tuple to a name keeps the source valid on 3.12 AND immune to ruff's paren-stripping.
DOWNLOAD_RETRY_EXC = (httpx.HTTPError, OSError)
JSON_RECORD_EXC = (json.JSONDecodeError, KeyError)

type Status = Literal["success", "partial_success", "error"]


class OcrRecord(BaseModel):
    arxiv_id: str
    paper_url: str
    status: Status
    num_pages: int
    num_pages_processed: int
    failed_pages: list[int]
    model: str
    elapsed_seconds: float
    processed_at: str
    markdown: str
    error_message: str | None = None


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _safe_name(arxiv_id: str) -> str:
    """arXiv ids can contain '/' (old style hep-th/0401001) — make a flat filename."""
    return arxiv_id.replace("/", "__")


def _read_ids(ids_file: Path, num_shards: int, shard_index: int) -> list[str]:
    ids = [ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if num_shards > 1:
        ids = [pid for i, pid in enumerate(ids) if i % num_shards == shard_index]
    return ids


# ---------------------------------------------------------------------------
# Phase 1: fetch — download all PDFs (no GPU)
# ---------------------------------------------------------------------------


async def _download_one(
    client: httpx.AsyncClient, arxiv_id: str, dest: Path, sem: asyncio.Semaphore, retries: int
) -> str:
    """Return 'ok' | 'exists' | 'missing' (404/410) | 'error'. Resume-safe."""
    if dest.exists() and dest.stat().st_size > 0:
        return "exists"
    url = ARXIV_PDF_URL.format(id=arxiv_id)
    tmp = dest.with_suffix(".part")
    async with sem:
        for attempt in range(retries):
            try:
                async with client.stream("GET", url) as resp:
                    if resp.status_code in (404, 410):
                        return "missing"  # invalid / withdrawn id — don't retry
                    if resp.status_code == 429 or resp.status_code >= 500:
                        await asyncio.sleep(min(30, 3 * (attempt + 1)))
                        continue
                    if resp.status_code != 200:
                        return "error"
                    with tmp.open("wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
                if tmp.stat().st_size == 0:
                    raise OSError("empty PDF")
                tmp.rename(dest)
                return "ok"
            except DOWNLOAD_RETRY_EXC:
                await asyncio.sleep(min(20, 2 ** (attempt + 1)))
        return "error"


async def _fetch_async(ids: list[str], pdf_dir: Path, concurrency: int, retries: int) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    missing_log = pdf_dir / "_missing_ids.txt"
    sem = asyncio.Semaphore(concurrency)
    counts = {"ok": 0, "exists": 0, "missing": 0, "error": 0}
    lock = asyncio.Lock()
    limits = httpx.Limits(max_connections=concurrency + 4, max_keepalive_connections=concurrency)
    # Only the AsyncClient is an async context manager; Progress and the missing-ids
    # file are sync, so nest them in a plain `with` inside the `async with`.
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(120.0, connect=15.0),
        limits=limits,
    ) as client:
        with (
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress,
            missing_log.open("a", encoding="utf-8") as missing_f,
        ):
            task = progress.add_task("fetch", total=len(ids))

            async def worker(arxiv_id: str) -> None:
                result = await _download_one(client, arxiv_id, pdf_dir / f"{_safe_name(arxiv_id)}.pdf", sem, retries)
                async with lock:
                    counts[result] += 1
                    if result == "missing":
                        missing_f.write(arxiv_id + "\n")
                        missing_f.flush()
                    done = sum(counts.values())
                    progress.update(
                        task,
                        advance=1,
                        description=f"ok={counts['ok']} have={counts['exists']} 404={counts['missing']} err={counts['error']}",
                    )
                    if done % 500 == 0:
                        print(f"[fetch] {done}/{len(ids)} {dict(counts)}", flush=True)

            await asyncio.gather(*(worker(pid) for pid in ids))
    console.print(f"[bold]fetch done[/] {counts}")


# ---------------------------------------------------------------------------
# Phase 2: ocr — saturate the vLLM server (global page-level concurrency)
# ---------------------------------------------------------------------------


def _render_pdf(pdf_path: Path, max_pages: int, image_dpi: int, min_image_dim: int) -> tuple[int, list]:
    """Blocking pypdfium2 render -> (total_pages, [PIL.Image scaled-to-fit]). Runs in a threadpool."""
    from chandra.input import flatten
    from chandra.model.util import scale_to_fit

    document = pdfium.PdfDocument(str(pdf_path))
    document.init_forms()
    total_pages = len(document)
    n = min(total_pages, max_pages) if max_pages else total_pages
    images = []
    try:
        for i in range(n):
            page = document[i]
            min_dim = min(page.get_width(), page.get_height())
            scale_dpi = max((min_image_dim / min_dim) * 72, image_dpi)
            flatten(page)
            page = document[i]
            img = page.render(scale=scale_dpi / 72).to_pil().convert("RGB")
            images.append(scale_to_fit(img))
    finally:
        document.close()
    return total_pages, images


def _img_to_content(img) -> list[dict]:
    from chandra.prompts import PROMPT_MAPPING

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": PROMPT_MAPPING[PROMPT_TYPE]},
    ]


async def _ocr_page(
    client: AsyncOpenAI, content: list[dict], page_sem: asyncio.Semaphore, max_tokens: int, max_retries: int
) -> tuple[str, bool]:
    """One page -> (raw_html, error). Retries repeat-token / errors with rising temperature."""
    from chandra.model.util import detect_repeat_token

    temperature, top_p = 0.0, 0.1
    raw, error = "", True
    for attempt in range(max_retries + 1):
        try:
            async with page_sem:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            raw = resp.choices[0].message.content or ""
            error = False
        except Exception as exc:
            raw, error = "", True
            if attempt == max_retries:
                console.print(f"[red]page failed after {max_retries} retries: {exc}[/]")
                break
            await asyncio.sleep(min(20, 2 * (attempt + 1)))
            continue
        repeated = detect_repeat_token(raw) or (len(raw) > 50 and detect_repeat_token(raw, cut_from_end=50))
        if not repeated:
            return raw, False
        if attempt == max_retries:
            break
        temperature = min(temperature + 0.2 * (attempt + 1), 0.8)
        top_p = 0.95
    return raw, error


async def _ocr_paper(
    client: AsyncOpenAI,
    arxiv_id: str,
    pdf_path: Path,
    render_pool: ThreadPoolExecutor,
    page_sem: asyncio.Semaphore,
    max_pages: int,
    image_dpi: int,
    min_image_dim: int,
    max_tokens: int,
    max_retries: int,
) -> OcrRecord:
    from chandra.output import parse_markdown

    started = time.time()
    loop = asyncio.get_running_loop()
    total_pages, images = await loop.run_in_executor(
        render_pool, _render_pdf, pdf_path, max_pages, image_dpi, min_image_dim
    )

    page_results = await asyncio.gather(
        *(_ocr_page(client, _img_to_content(img), page_sem, max_tokens, max_retries) for img in images)
    )

    md_parts: list[str] = []
    failed_pages: list[int] = []
    for idx, (raw, error) in enumerate(page_results, start=1):
        if error or not raw.strip():
            failed_pages.append(idx)
            continue
        md_parts.append(parse_markdown(raw, include_headers_footers=False))

    has_content = any(p.strip() for p in md_parts)
    status: Status = "success" if not failed_pages else ("partial_success" if has_content else "error")
    return OcrRecord(
        arxiv_id=arxiv_id,
        paper_url=ARXIV_ABS_URL.format(id=arxiv_id),
        status=status,
        num_pages=total_pages,
        num_pages_processed=len(images),
        failed_pages=failed_pages,
        model=MODEL_NAME,
        elapsed_seconds=round(time.time() - started, 2),
        processed_at=_utc_now_iso(),
        markdown="\n\n".join(md_parts).strip(),
    )


def _load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                done.add(json.loads(line)["arxiv_id"])
            except JSON_RECORD_EXC:
                continue
    return done


async def _ocr_async(
    pdfs: list[tuple[str, Path]],
    out_path: Path,
    base_url: str,
    concurrency: int,
    paper_workers: int,
    max_pages: int,
    image_dpi: int,
    min_image_dim: int,
    max_tokens: int,
    max_retries: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_sem = asyncio.Semaphore(concurrency)
    paper_sem = asyncio.Semaphore(paper_workers)
    write_lock = asyncio.Lock()
    counts = {"success": 0, "partial_success": 0, "error": 0}
    # pypdfium2 is NOT thread-safe: rendering >1 PDF concurrently corrupts the native
    # heap (`double free or corruption` -> SIGABRT, killing the whole run before any
    # page reaches the GPU). A single dedicated render thread serializes all pdfium
    # calls. This does NOT throttle the GPU: saturation comes from page-level OCR
    # concurrency (page_sem), and rendering (~ms/page) far outpaces OCR (~s/page).
    render_pool = ThreadPoolExecutor(max_workers=1)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,
        timeout=600.0,
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 16), timeout=600.0),
    )
    with (
        out_path.open("a", encoding="utf-8") as sink,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("ocr", total=len(pdfs))

        async def worker(arxiv_id: str, pdf_path: Path) -> None:
            async with paper_sem:
                try:
                    record = await _ocr_paper(
                        client,
                        arxiv_id,
                        pdf_path,
                        render_pool,
                        page_sem,
                        max_pages,
                        image_dpi,
                        min_image_dim,
                        max_tokens,
                        max_retries,
                    )
                except Exception as exc:
                    record = OcrRecord(
                        arxiv_id=arxiv_id,
                        paper_url=ARXIV_ABS_URL.format(id=arxiv_id),
                        status="error",
                        num_pages=0,
                        num_pages_processed=0,
                        failed_pages=[],
                        model=MODEL_NAME,
                        elapsed_seconds=0.0,
                        processed_at=_utc_now_iso(),
                        markdown="",
                        error_message=str(exc),
                    )
            async with write_lock:
                sink.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
                sink.flush()
                counts[record.status] += 1
                done = sum(counts.values())
                progress.update(
                    task,
                    advance=1,
                    description=f"ok={counts['success']} partial={counts['partial_success']} err={counts['error']}",
                )
                if done % 50 == 0:
                    print(f"[ocr] {done}/{len(pdfs)} {dict(counts)}", flush=True)

        await asyncio.gather(*(worker(pid, path) for pid, path in pdfs))
    await client.close()
    render_pool.shutdown(wait=False)
    console.print(f"[bold green]ocr done[/] {counts}")


app = typer.Typer(add_completion=False, help="Stage 2: fetch arXiv PDFs, then OCR them on vLLM (chandra-ocr-2).")


@app.command()
def fetch(
    ids_file: Annotated[Path, typer.Option("--ids-file", help="One arXiv id per line (ocr_needed.txt).")],
    pdf_dir: Annotated[Path, typer.Option("--pdf-dir", help="Where to write {id}.pdf.")] = Path("pdfs"),
    concurrency: Annotated[int, typer.Option(help="Parallel downloads.")] = 16,
    retries: Annotated[int, typer.Option(help="Per-PDF retries on 429/5xx/network.")] = 4,
    num_shards: Annotated[int, typer.Option()] = 1,
    shard_index: Annotated[int, typer.Option()] = 0,
) -> None:
    """Download every PDF first (no GPU), so the OCR phase never waits on arXiv."""
    ids = _read_ids(ids_file, num_shards, shard_index)
    console.print(f"[bold]{len(ids):,}[/] ids -> {pdf_dir}")
    asyncio.run(_fetch_async(ids, pdf_dir, concurrency, retries))


@app.command()
def ocr(
    pdf_dir: Annotated[Path, typer.Option("--pdf-dir", help="Dir of pre-fetched {id}.pdf files.")] = Path("pdfs"),
    out: Annotated[Path, typer.Option("--out", help="Append-only jsonl sink (resume source).")] = Path("ocr_out.jsonl"),
    vllm_base_url: Annotated[str, typer.Option(help="vLLM OpenAI endpoint.")] = "http://localhost:8000/v1",
    concurrency: Annotated[int, typer.Option(help="Global in-flight page requests (saturate the server).")] = 256,
    paper_workers: Annotated[int, typer.Option(help="Papers rendered/in-flight at once.")] = 48,
    max_pages: Annotated[int, typer.Option(help="Hard cap on pages OCR'd per paper.")] = 30,
    image_dpi: Annotated[int, typer.Option()] = 192,
    min_image_dim: Annotated[int, typer.Option()] = 1024,
    max_tokens: Annotated[int, typer.Option()] = 12384,
    max_retries: Annotated[int, typer.Option()] = 6,
) -> None:
    """OCR all pre-fetched PDFs concurrently against vLLM, append resume-safe records."""
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    done = _load_done(out)
    pending = [(p.stem.replace("__", "/"), p) for p in all_pdfs if p.stem.replace("__", "/") not in done]
    console.print(
        f"[bold]{len(all_pdfs):,}[/] PDFs, [green]{len(done):,}[/] done, [yellow]{len(pending):,}[/] to OCR -> {out}"
    )
    if not pending:
        console.print("[green]nothing to do[/]")
        return
    asyncio.run(
        _ocr_async(
            pending,
            out,
            vllm_base_url,
            concurrency,
            paper_workers,
            max_pages,
            image_dpi,
            min_image_dim,
            max_tokens,
            max_retries,
        )
    )


if __name__ == "__main__":
    app()
