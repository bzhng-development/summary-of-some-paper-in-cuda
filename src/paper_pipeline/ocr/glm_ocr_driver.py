# /// script
# requires-python = ">=3.12"
# dependencies = ["pypdfium2", "openai", "httpx", "typer", "rich", "pydantic"]
# ///
"""Stage 2 (GLM-OCR): OCR pre-fetched arXiv PDFs -> markdown via self-hosted GLM-OCR on vLLM.

Direct full-page recognition (no SDK, no layout model): render each page to an image, send
it to the GLM-OCR OpenAI endpoint with the `"Text Recognition:"` prompt, and the model
returns markdown directly. GPU saturation comes from page-level async concurrency (one
global semaphore -> hundreds of in-flight page requests); rendering runs in a PROCESS pool
(pypdfium2 is not thread-safe, but is fine across separate processes -> parallel + a bad
PDF crashes only one worker).

Resume-safe: skip arxiv_ids already in the output jsonl. Rows match `ingest_ocr.py`.

    python glm_ocr_driver.py --pdf-dir pdfs --out glm_out.jsonl --first 500 \
        --base-url http://localhost:8080/v1 --concurrency 512 --render-workers 16
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import httpx
import typer
from openai import AsyncOpenAI
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

console = Console()

MODEL_NAME = "GLM-OCR"  # --served-model-name on the vLLM side
PROMPT = "Text Recognition:"
ARXIV_ABS_URL = "https://arxiv.org/abs/{id}"
# Named so `ruff format` (repo target py314) can't rewrite `except (A, B):` into the bare
# tuple `except A, B:`, which is a SyntaxError on the box's Python 3.12.
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


def _render_pdf_to_b64(pdf_path_str: str, max_pages: int, image_dpi: int) -> tuple[int, list[str]]:
    """Runs in a SEPARATE PROCESS (pdfium is process-safe). -> (total_pages, [base64 PNG per page])."""
    import base64 as _b64
    import io as _io

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(pdf_path_str)
    total_pages = len(document)
    n = min(total_pages, max_pages) if max_pages else total_pages
    out: list[str] = []
    try:
        for i in range(n):
            img = document[i].render(scale=image_dpi / 72).to_pil().convert("RGB")
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            out.append(_b64.b64encode(buf.getvalue()).decode())
    finally:
        document.close()
    return total_pages, out


def _content(b64: str) -> list[dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": PROMPT},
    ]


async def _ocr_page(
    client: AsyncOpenAI, b64: str, sem: asyncio.Semaphore, max_tokens: int, max_retries: int
) -> tuple[str, bool]:
    """One page -> (markdown, error). GLM-OCR returns markdown directly; no post-parse."""
    for attempt in range(max_retries + 1):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": _content(b64)}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
            return (resp.choices[0].message.content or ""), False
        except Exception as exc:
            if attempt == max_retries:
                console.print(f"[red]page failed after {max_retries} retries: {exc}[/]")
                return "", True
            await asyncio.sleep(min(20, 2 * (attempt + 1)))
    return "", True


async def _ocr_paper(
    client: AsyncOpenAI,
    arxiv_id: str,
    pdf_path: Path,
    render_pool: ProcessPoolExecutor,
    sem: asyncio.Semaphore,
    max_pages: int,
    image_dpi: int,
    max_tokens: int,
    max_retries: int,
) -> OcrRecord:
    started = time.time()
    loop = asyncio.get_running_loop()
    total_pages, b64s = await loop.run_in_executor(render_pool, _render_pdf_to_b64, str(pdf_path), max_pages, image_dpi)
    results = await asyncio.gather(*(_ocr_page(client, b, sem, max_tokens, max_retries) for b in b64s))

    parts: list[str] = []
    failed_pages: list[int] = []
    for idx, (raw, error) in enumerate(results, start=1):
        if error or not raw.strip():
            failed_pages.append(idx)
        else:
            parts.append(raw.strip())
    has_content = any(parts)
    status: Status = "success" if not failed_pages else ("partial_success" if has_content else "error")
    return OcrRecord(
        arxiv_id=arxiv_id,
        paper_url=ARXIV_ABS_URL.format(id=arxiv_id),
        status=status,
        num_pages=total_pages,
        num_pages_processed=len(b64s),
        failed_pages=failed_pages,
        model=MODEL_NAME,
        elapsed_seconds=round(time.time() - started, 2),
        processed_at=datetime.now(tz=UTC).isoformat(),
        markdown="\n\n".join(parts).strip(),
    )


def _load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rec = json.loads(line)
                # Only count real results as done; error rows (e.g. from a server crash) are
                # left to retry on the next run instead of being permanently skipped.
                if rec.get("status") in ("success", "partial_success"):
                    done.add(rec["arxiv_id"])
            except JSON_RECORD_EXC:
                continue
    return done


async def _run(
    pdfs: list[tuple[str, Path]],
    out_path: Path,
    base_url: str,
    concurrency: int,
    paper_workers: int,
    render_workers: int,
    max_pages: int,
    image_dpi: int,
    max_tokens: int,
    max_retries: int,
) -> None:
    sem = asyncio.Semaphore(concurrency)
    paper_sem = asyncio.Semaphore(paper_workers)
    write_lock = asyncio.Lock()
    counts = {"success": 0, "partial_success": 0, "error": 0}
    pages_total = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    render_pool = ProcessPoolExecutor(max_workers=render_workers, mp_context=ctx)
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="EMPTY",
        max_retries=0,
        timeout=600.0,
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency + 16), timeout=600.0),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
            nonlocal pages_total
            async with paper_sem:
                try:
                    record = await _ocr_paper(
                        client, arxiv_id, pdf_path, render_pool, sem, max_pages, image_dpi, max_tokens, max_retries
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
                        processed_at=datetime.now(tz=UTC).isoformat(),
                        markdown="",
                        error_message=str(exc),
                    )
            async with write_lock:
                sink.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
                sink.flush()
                counts[record.status] += 1
                pages_total += record.num_pages_processed
                done_n = sum(counts.values())
                progress.update(
                    task,
                    advance=1,
                    description=f"ok={counts['success']} partial={counts['partial_success']} err={counts['error']}",
                )
                if done_n % 25 == 0:
                    rate = done_n / max(1e-9, time.time() - t0)
                    print(f"[ocr] {done_n}/{len(pdfs)} {dict(counts)} {rate:.2f} papers/s", flush=True)

        await asyncio.gather(*(worker(a, p) for a, p in pdfs))
    await client.close()
    render_pool.shutdown(wait=False)
    elapsed = time.time() - t0
    papers_per_s = len(pdfs) / max(1e-9, elapsed)
    console.print(
        f"[bold green]ocr done[/] {counts} in {elapsed:.1f}s | "
        f"{papers_per_s:.2f} papers/s, {pages_total / max(1e-9, elapsed):.1f} pages/s | "
        f"ETA for 6,940: {6940 / max(1e-9, papers_per_s) / 60:.1f} min"
    )


def main(
    pdf_dir: Annotated[Path, typer.Option("--pdf-dir")] = Path("pdfs"),
    out: Annotated[Path, typer.Option("--out")] = Path("glm_out.jsonl"),
    base_url: Annotated[str, typer.Option("--base-url")] = "http://localhost:8080/v1",
    concurrency: Annotated[int, typer.Option("--concurrency", help="Global in-flight page requests (~512/GPU).")] = 512,
    paper_workers: Annotated[int, typer.Option("--paper-workers", help="Papers in flight at once.")] = 64,
    render_workers: Annotated[int, typer.Option("--render-workers", help="PDF render process pool size.")] = 16,
    max_pages: Annotated[int, typer.Option("--max-pages", help="Hard cap on pages per paper (0 = all).")] = 40,
    image_dpi: Annotated[int, typer.Option("--image-dpi")] = 150,
    max_tokens: Annotated[int, typer.Option("--max-tokens")] = 8192,
    max_retries: Annotated[int, typer.Option("--max-retries")] = 5,
    first: Annotated[int, typer.Option("--first", help="Only process the first N PDFs (0 = all).")] = 0,
    ids_file: Annotated[
        Path | None, typer.Option("--ids-file", help="Restrict OCR to these arxiv_ids (one per line).")
    ] = None,
) -> None:
    """OCR pre-fetched PDFs against a self-hosted GLM-OCR vLLM endpoint, append resume-safe jsonl."""
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if ids_file is not None:
        keep = {ln.strip() for ln in ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
        all_pdfs = [p for p in all_pdfs if p.stem.replace("__", "/") in keep]
    if first > 0:
        all_pdfs = all_pdfs[:first]
    done = _load_done(out)
    pending = [(p.stem.replace("__", "/"), p) for p in all_pdfs if p.stem.replace("__", "/") not in done]
    console.print(
        f"[bold]{len(all_pdfs):,}[/] PDFs, [green]{len(done):,}[/] done, [yellow]{len(pending):,}[/] to OCR -> {out}\n"
        f"concurrency={concurrency} paper_workers={paper_workers} render_workers={render_workers}"
    )
    if not pending:
        console.print("[green]nothing to do[/]")
        return
    asyncio.run(
        _run(
            pending,
            out,
            base_url,
            concurrency,
            paper_workers,
            render_workers,
            max_pages,
            image_dpi,
            max_tokens,
            max_retries,
        )
    )


if __name__ == "__main__":
    typer.run(main)
