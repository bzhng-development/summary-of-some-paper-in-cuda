"""CLI dispatch for the long-form paper summarization pipeline."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from loguru import logger

from .config import DEFAULT_MODEL
from .llm import LLMClient, build_gemini_client, build_local_client
from .pipeline import (
    PaperJob,
    build_backfill_jobs,
    process_multiple,
    process_paper,
    run_dynamic,
)
from .prompts import SECTION_SPECS, SECTION_SPECS_2PASS, SectionSpec
from .summarizer import MultiPromptSummarizer

_GEMINI_DEFAULT_MODEL = "google/gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Parsed CLI args
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CLIArgs:
    model: str
    url: str | None
    urls: str | None
    concurrency: int
    dyn: bool
    interval: int
    gemini: bool
    backfill: bool
    jsonl: str | None
    many_pass: bool

    @property
    def sections(self) -> tuple[SectionSpec, ...]:
        # ``--many-pass`` enables the original sequential 7-call pipeline.
        # The default 2-pass layout is gentler on context windows.
        return SECTION_SPECS if self.many_pass else SECTION_SPECS_2PASS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-prompt paper summarizer")
    parser.add_argument("--model", default=None, help="Model name (auto-set for --gemini)")
    parser.add_argument("--url", default=None, help="ArXiv URL to summarize")
    parser.add_argument(
        "--urls", default=None, help="Comma-separated list of ArXiv URLs"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent papers (default: 5)",
    )
    parser.add_argument(
        "--dyn",
        action="store_true",
        help="Continuously poll Neon for stubs and process them",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds for --dyn (default: 30)",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Gemini 3 Flash via OpenRouter/Google AI Studio (free tier)",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Re-summarize all interested papers missing a summary",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Write results to JSONL instead of DB (for remote runs)",
    )
    parser.add_argument(
        "--many-pass",
        dest="many_pass",
        action="store_true",
        help="Use the 7-section sequential pipeline (old behavior, smaller models)",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> CLIArgs:
    parser = _build_parser()
    ns = parser.parse_args(argv)
    model = ns.model
    if model is None:
        model = _GEMINI_DEFAULT_MODEL if ns.gemini else DEFAULT_MODEL
    args = CLIArgs(
        model=model,
        url=ns.url,
        urls=ns.urls,
        concurrency=ns.concurrency,
        dyn=ns.dyn,
        interval=ns.interval,
        gemini=ns.gemini,
        backfill=ns.backfill,
        jsonl=ns.jsonl,
        many_pass=ns.many_pass,
    )
    if not (args.url or args.urls or args.dyn or args.backfill):
        parser.error("Must provide either --url, --urls, --dyn, or --backfill")
    return args


# ---------------------------------------------------------------------------
# Client selection
# ---------------------------------------------------------------------------


def _build_llm(args: CLIArgs) -> LLMClient:
    return build_gemini_client() if args.gemini else build_local_client()


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


async def _run_backfill(args: CLIArgs, llm: LLMClient) -> None:
    jobs = build_backfill_jobs()
    if not jobs:
        logger.success("No interested papers need summarization!")
        return
    logger.info("Backfill: {} interested papers to summarize", len(jobs))
    await process_multiple(
        llm,
        jobs,
        model=args.model,
        sections=args.sections,
        concurrency=args.concurrency,
        jsonl_path=args.jsonl,
    )


async def _run_batch(args: CLIArgs, llm: LLMClient) -> None:
    assert args.urls is not None
    raw_urls = [u.strip() for u in args.urls.split(",") if u.strip()]
    jobs = [PaperJob(url=u) for u in raw_urls]
    await process_multiple(
        llm,
        jobs,
        model=args.model,
        sections=args.sections,
        concurrency=args.concurrency,
        jsonl_path=args.jsonl,
    )


async def _run_single(args: CLIArgs, llm: LLMClient) -> None:
    assert args.url is not None
    summarizer = MultiPromptSummarizer(
        llm, model=args.model, sections=args.sections
    )
    await process_paper(
        summarizer, PaperJob(url=args.url), jsonl_path=args.jsonl
    )


async def _run_dyn(args: CLIArgs, llm: LLMClient) -> None:
    await run_dynamic(
        llm,
        model=args.model,
        sections=args.sections,
        concurrency=args.concurrency,
        interval=args.interval,
    )


async def _dispatch(args: CLIArgs) -> None:
    llm = _build_llm(args)
    try:
        if args.backfill:
            await _run_backfill(args, llm)
        elif args.dyn:
            await _run_dyn(args, llm)
        elif args.urls:
            await _run_batch(args, llm)
        else:
            await _run_single(args, llm)
    finally:
        await llm.aclose()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logger.info("multi_prompt CLI: mode={}", _mode_label(args))
    asyncio.run(_dispatch(args))


def _mode_label(args: CLIArgs) -> str:
    if args.backfill:
        return "backfill"
    if args.dyn:
        return "dyn"
    if args.urls:
        return "batch"
    return "single"
