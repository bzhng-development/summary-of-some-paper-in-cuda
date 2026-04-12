"""High-level orchestration: single-paper and batch runs, plus the dyn poller."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from .llm import LLMClient
from .pdf import (
    arxiv_id_from_url,
    arxiv_url_to_pdf_url,
    download_and_extract_text,
    download_pdf_bytes,
)
from .prompts import SectionSpec
from .schemas import PitchOutput
from .storage import (
    append_jsonl,
    get_db,
    get_interested_stubs,
    get_stub_ids,
    load_done_ids_from_jsonl,
    paper_has_real_summary,
    save_summary_markdown,
)
from .summarizer import MultiPromptSummarizer


# ---------------------------------------------------------------------------
# Single-paper orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PaperJob:
    """A single unit of work for the pipeline.

    ``paper_id`` is the caller-supplied ID (possibly an ``ext:`` sentinel).
    When absent, the ID is derived from the URL at execution time.
    """

    url: str
    paper_id: str | None = None

    def is_external(self) -> bool:
        return self.paper_id is not None and self.paper_id.startswith("ext:")

    def resolve_ids(self) -> tuple[str, str]:
        """Return ``(arxiv_id, pdf_url)`` for this job."""
        if self.is_external():
            # ext: papers store the direct PDF URL in the url slot.
            assert self.paper_id is not None  # narrowed by is_external()
            return self.paper_id, self.url
        return arxiv_id_from_url(self.url), arxiv_url_to_pdf_url(self.url)


@dataclass(frozen=True, slots=True)
class PaperResult:
    arxiv_id: str
    title: str
    category: str
    pitch: str
    summary: str
    url: str
    output_file: Path | None = None

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "category": self.category,
            "pitch": self.pitch,
            "summary": self.summary,
            "url": self.url,
        }
        if self.output_file is not None:
            record["output_file"] = str(self.output_file)
        return record


async def _fetch_paper_input(pdf_url: str, *, native_pdf: bool) -> str:
    """Fetch the paper and return the string payload for the summarizer."""
    if native_pdf:
        pdf_bytes = await asyncio.to_thread(download_pdf_bytes, pdf_url)
        return f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
    return await asyncio.to_thread(download_and_extract_text, pdf_url)


async def _persist_result(
    *,
    arxiv_id: str,
    arxiv_url: str,
    paper_id: str | None,
    pitch_output: PitchOutput,
    full_summary: str,
    category: str,
    jsonl_path: str | None,
) -> PaperResult:
    """Write the result to JSONL xor markdown+Neon, and return it."""
    if jsonl_path is not None:
        result = PaperResult(
            arxiv_id=arxiv_id,
            title=pitch_output.title,
            category=category,
            pitch=pitch_output.pitch,
            summary=full_summary,
            url=arxiv_url,
        )
        await asyncio.to_thread(append_jsonl, result.to_record(), jsonl_path)
        tqdm.write(f"[{arxiv_id}] Appended to {jsonl_path}")
        return result

    output_file = await asyncio.to_thread(
        save_summary_markdown,
        pitch_output,
        full_summary,
        category,
        arxiv_url=arxiv_url,
        paper_id=paper_id,
    )
    await get_db().asave_paper(
        arxiv_id,
        title=pitch_output.title,
        category=category,
        pitch=pitch_output.pitch,
        summary=full_summary,
        url=arxiv_url,
    )
    tqdm.write(f"[{arxiv_id}] Saved to: {output_file}")
    return PaperResult(
        arxiv_id=arxiv_id,
        title=pitch_output.title,
        category=category,
        pitch=pitch_output.pitch,
        summary=full_summary,
        url=arxiv_url,
        output_file=output_file,
    )


async def process_paper(
    summarizer: MultiPromptSummarizer,
    job: PaperJob,
    *,
    papers_pbar: tqdm | None = None,
    jsonl_path: str | None = None,
) -> PaperResult:
    """Run the full summarize-and-persist pipeline for a single paper."""
    arxiv_id, pdf_url = job.resolve_ids()
    paper_input = await _fetch_paper_input(pdf_url, native_pdf=summarizer.llm.native_pdf)

    total_steps = len(summarizer.sections) + 2  # sections + pitch + categorize
    pbar = tqdm(total=total_steps, desc=f"[{arxiv_id}]", leave=False, position=1)
    try:
        full_summary, _ = await summarizer.generate_full_summary(
            paper_input, arxiv_id=arxiv_id, pbar=pbar
        )
        pbar.update(len(summarizer.sections))

        pbar.set_postfix_str("pitch")
        pitch_output = await summarizer.generate_pitch(full_summary, paper_input)
        tqdm.write(f"\n[{arxiv_id}] Title: {pitch_output.title}")
        tqdm.write(f"[{arxiv_id}] Pitch: {pitch_output.pitch}")
        pbar.update(1)

        pbar.set_postfix_str("categorize")
        category = await summarizer.categorize_paper(
            pitch_output.title, pitch_output.pitch, full_summary
        )
        tqdm.write(f"[{arxiv_id}] Category: {category}")
        pbar.update(1)
    finally:
        pbar.close()

    result = await _persist_result(
        arxiv_id=arxiv_id,
        arxiv_url=job.url,
        paper_id=job.paper_id,
        pitch_output=pitch_output,
        full_summary=full_summary,
        category=category,
        jsonl_path=jsonl_path,
    )

    if papers_pbar is not None:
        papers_pbar.update(1)
    return result


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def _filter_already_done(
    jobs: list[PaperJob], *, jsonl_path: str | None
) -> list[PaperJob]:
    """Drop jobs that already have a finished summary in JSONL or Neon."""
    done_ids: set[str] = set()
    if jsonl_path is not None:
        done_ids = load_done_ids_from_jsonl(jsonl_path)
        if done_ids:
            logger.info("Found {} papers already in {}", len(done_ids), jsonl_path)

    filtered: list[PaperJob] = []
    for job in jobs:
        pid = job.paper_id or arxiv_id_from_url(job.url)
        if pid in done_ids:
            logger.info("Skipping {} (already in JSONL)", pid)
            continue
        if paper_has_real_summary(pid):
            logger.info("Skipping {} (already in DB)", pid)
            continue
        filtered.append(job)

    dropped = len(jobs) - len(filtered)
    if dropped:
        logger.info("Skipped {} papers already processed", dropped)
    return filtered


async def process_multiple(
    llm: LLMClient,
    jobs: list[PaperJob],
    *,
    model: str,
    sections: tuple[SectionSpec, ...] | list[SectionSpec] | None,
    concurrency: int,
    jsonl_path: str | None = None,
) -> None:
    """Process a finite batch of papers concurrently, bounded by ``concurrency``."""
    filtered = _filter_already_done(jobs, jsonl_path=jsonl_path)
    if not filtered:
        logger.success("All papers already processed!")
        return

    logger.info(
        "Processing {} papers (concurrency={}, native_pdf={})",
        len(filtered),
        concurrency,
        llm.native_pdf,
    )

    sem = asyncio.Semaphore(concurrency)
    papers_pbar = tqdm(total=len(filtered), desc="Papers", position=0)

    async def _run_one(job: PaperJob) -> None:
        async with sem:
            summarizer = MultiPromptSummarizer(llm, model=model, sections=sections)
            await process_paper(
                summarizer, job, papers_pbar=papers_pbar, jsonl_path=jsonl_path
            )

    # NOTE: we deliberately avoid asyncio.TaskGroup here. A single failing
    # paper must not cancel the rest of the batch — the daily backfill run
    # needs best-effort completion semantics.
    tasks = [asyncio.create_task(_run_one(job)) for job in filtered]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors: list[tuple[PaperJob, BaseException]] = []
    for job, outcome in zip(filtered, results, strict=True):
        if isinstance(outcome, BaseException):
            logger.error("Error processing {}: {!r}", job.url, outcome)
            errors.append((job, outcome))
            papers_pbar.update(1)

    papers_pbar.close()

    if errors:
        logger.warning("{} papers failed", len(errors))
    logger.success("Finished processing {} papers!", len(filtered))


# ---------------------------------------------------------------------------
# Dynamic poller
# ---------------------------------------------------------------------------


async def run_dynamic(
    llm: LLMClient,
    *,
    model: str,
    sections: tuple[SectionSpec, ...] | list[SectionSpec] | None,
    concurrency: int,
    interval: int,
) -> None:
    """Continuously poll the DB for stubs and dispatch them under a semaphore."""
    logger.info(
        "Dynamic mode: polling every {}s, concurrency={} (Ctrl-C to stop)",
        interval,
        concurrency,
    )
    sem = asyncio.Semaphore(concurrency)
    processed_total = 0
    in_flight: set[str] = set()
    tasks: set[asyncio.Task[None]] = set()

    async def _run_one(aid: str) -> None:
        nonlocal processed_total
        async with sem:
            summarizer = MultiPromptSummarizer(llm, model=model, sections=sections)
            job = PaperJob(url=f"https://arxiv.org/abs/{aid}")
            try:
                await process_paper(summarizer, job, papers_pbar=None)
                processed_total += 1
                logger.success("Completed {} ({} total)", aid, processed_total)
            except Exception as exc:
                logger.error("Failed {}: {!r}", aid, exc)
            finally:
                in_flight.discard(aid)

    try:
        while True:
            stubs = await asyncio.to_thread(get_stub_ids)
            new_stubs = [aid for aid in stubs if aid not in in_flight]

            # Reap finished tasks.
            tasks = {t for t in tasks if not t.done()}

            if not new_stubs:
                if not in_flight:
                    logger.debug("No stubs found, sleeping {}s...", interval)
                await asyncio.sleep(interval)
                continue

            logger.info(
                "Submitting {} new stubs ({} already in flight)",
                len(new_stubs),
                len(in_flight),
            )
            in_flight.update(new_stubs)
            for aid in new_stubs:
                tasks.add(asyncio.create_task(_run_one(aid)))
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("\nStopping... waiting for {} in-flight tasks", len(tasks))
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Stopped. Processed {} papers total.", processed_total)


# ---------------------------------------------------------------------------
# --backfill helper
# ---------------------------------------------------------------------------


def build_backfill_jobs() -> list[PaperJob]:
    """Translate the ``interested`` stub rows into :class:`PaperJob` instances."""
    stubs = get_interested_stubs()
    jobs: list[PaperJob] = []
    for aid, url in stubs:
        if aid.startswith("ext:"):
            jobs.append(PaperJob(url=url or "", paper_id=aid))
        else:
            jobs.append(PaperJob(url=f"https://arxiv.org/abs/{aid}", paper_id=aid))
    return jobs
