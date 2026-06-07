"""Stage 2 (PaddleOCR-VL): OCR pre-fetched arXiv PDFs -> markdown via the PaddleOCRVL pipeline.

Runs in the box's `paddle-venv` (paddlepaddle + paddleocr[doc-parser]). The heavy VLM
recognition is served by vLLM (cu130) on the GPU; layout/cropping/reading-order run here
on CPU (PaddlePaddle SIGABRTs on the B300's sm_103a, so we force CPU for paddle).

Concurrency model: a process pool (each worker its own PaddleOCRVL — paddle isn't
thread-safe, and separate processes also crash-isolate a bad PDF and parallelize the CPU
layout). Each worker's `vl_rec_max_concurrency` is `rec_concurrency // workers` so the
TOTAL in-flight recognition requests against the server ~= `rec_concurrency` (512/GPU).

Output: append-only `ocr_out.jsonl` with rows `ingest_ocr.py` accepts (arxiv_id, markdown,
status, + diagnostics). Resume = skip arxiv_ids already present.

    python paddleocr_vl_ocr.py --pdf-dir pdfs --out ocr_out.jsonl --first 500 \
        --server-url http://localhost:8080/v1 --rec-concurrency 512 --workers 8
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

# Must be set before any paddle import in workers; paddle aborts on the B300 GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

console = Console()
ARXIV_ABS_URL = "https://arxiv.org/abs/{id}"
# Named so `ruff format` (repo target py314) can't rewrite a literal `except (A, B):` into
# the bare tuple `except A, B:`, which is a SyntaxError on the box's Python 3.12.
JSON_RECORD_EXC = (json.JSONDecodeError, KeyError)

# Per-process pipeline (one PaddleOCRVL per worker, built in the initializer).
_PIPE = None


class OcrRecord(BaseModel):
    arxiv_id: str
    paper_url: str
    status: str
    num_pages: int
    elapsed_seconds: float
    processed_at: str
    markdown: str
    error_message: str | None = None


def _init_worker(server_url: str, model_name: str, rec_concurrency: int) -> None:
    global _PIPE
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from paddleocr import PaddleOCRVL

    _PIPE = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="vllm-server",
        vl_rec_server_url=server_url,
        vl_rec_api_model_name=model_name,
        vl_rec_max_concurrency=rec_concurrency,
    )


def _ocr_one(pdf_path_str: str) -> dict:
    arxiv_id = Path(pdf_path_str).stem.replace("__", "/")
    started = time.time()
    try:
        results = list(_PIPE.predict(pdf_path_str))
        results.sort(key=lambda r: r.markdown.get("page_index", 0))
        parts = [r.markdown.get("markdown_texts", "") for r in results]
        markdown = "\n\n".join(p for p in parts if p and p.strip()).strip()
        status = "success" if markdown else "error"
        return {
            "arxiv_id": arxiv_id,
            "paper_url": ARXIV_ABS_URL.format(id=arxiv_id),
            "status": status,
            "num_pages": len(results),
            "elapsed_seconds": round(time.time() - started, 2),
            "processed_at": datetime.now(tz=UTC).isoformat(),
            "markdown": markdown,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "arxiv_id": arxiv_id,
            "paper_url": ARXIV_ABS_URL.format(id=arxiv_id),
            "status": "error",
            "num_pages": 0,
            "elapsed_seconds": round(time.time() - started, 2),
            "processed_at": datetime.now(tz=UTC).isoformat(),
            "markdown": "",
            "error_message": repr(exc)[:500],
        }


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


def main(
    pdf_dir: Annotated[Path, typer.Option("--pdf-dir", help="Dir of pre-fetched {id}.pdf files.")] = Path("pdfs"),
    out: Annotated[Path, typer.Option("--out", help="Append-only jsonl sink (resume source).")] = Path("ocr_out.jsonl"),
    server_url: Annotated[str, typer.Option("--server-url", help="vLLM OpenAI endpoint.")] = "http://localhost:8080/v1",
    model_name: Annotated[str, typer.Option("--model-name")] = "PaddleOCR-VL-1.6-0.9B",
    rec_concurrency: Annotated[
        int, typer.Option("--rec-concurrency", help="TOTAL in-flight requests (~512/GPU).")
    ] = 512,
    workers: Annotated[int, typer.Option("--workers", help="Process-pool size (parallel CPU layout).")] = 8,
    first: Annotated[int, typer.Option("--first", help="Only process the first N PDFs (0 = all).")] = 0,
) -> None:
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if first > 0:
        all_pdfs = all_pdfs[:first]
    done = _load_done(out)
    pending = [p for p in all_pdfs if p.stem.replace("__", "/") not in done]
    per_worker = max(1, rec_concurrency // workers)
    console.print(
        f"[bold]{len(all_pdfs):,}[/] PDFs, [green]{len(done):,}[/] done, [yellow]{len(pending):,}[/] to OCR -> {out}\n"
        f"workers={workers} rec_concurrency={rec_concurrency} (per-worker={per_worker})"
    )
    if not pending:
        console.print("[green]nothing to do[/]")
        return

    counts = {"success": 0, "error": 0}
    pages_total = 0
    t0 = time.time()
    ctx = mp.get_context("spawn")
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        out.open("a", encoding="utf-8") as sink,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
        ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(server_url, model_name, per_worker),
        ) as pool,
    ):
        task = progress.add_task("ocr", total=len(pending))
        futures = {pool.submit(_ocr_one, str(p)): p for p in pending}
        for fut in as_completed(futures):
            rec = fut.result()
            record = OcrRecord.model_validate(rec)
            sink.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
            sink.flush()
            counts[record.status] = counts.get(record.status, 0) + 1
            pages_total += record.num_pages
            done_n = sum(counts.values())
            progress.update(task, advance=1, description=f"ok={counts['success']} err={counts['error']}")
            if done_n % 25 == 0:
                rate = done_n / max(1e-9, time.time() - t0)
                print(f"[ocr] {done_n}/{len(pending)} {dict(counts)} {rate:.2f} papers/s", flush=True)

    elapsed = time.time() - t0
    papers_per_s = len(pending) / max(1e-9, elapsed)
    pages_per_s = pages_total / max(1e-9, elapsed)
    console.print(
        f"[bold green]ocr done[/] {counts} in {elapsed:.1f}s | "
        f"{papers_per_s:.2f} papers/s, {pages_per_s:.1f} pages/s | "
        f"ETA for 6,940: {6940 / max(1e-9, papers_per_s) / 60:.1f} min"
    )


if __name__ == "__main__":
    typer.run(main)
