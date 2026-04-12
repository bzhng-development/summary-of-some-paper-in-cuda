"""Long-form paper summarization pipeline — top-level entry point.

This module used to be a 1041-line god-module. It is now a thin shim that
re-exports the public API from :mod:`multi_prompt_pkg` and dispatches the CLI
via :func:`multi_prompt_pkg.cli.main`.

The entry point is preserved so that the user's daily runbook still works:

    uv run python multi_prompt.py --url https://arxiv.org/abs/2312.07104
    uv run python multi_prompt.py --urls "u1,u2,u3" --concurrency 5
    uv run python multi_prompt.py --dyn --concurrency 4096 --interval 30
    uv run python multi_prompt.py --backfill --concurrency 8
    uv run python multi_prompt.py --dyn --jsonl ./runs/remote.jsonl
    uv run python multi_prompt.py --gemini --url ...
    uv run python multi_prompt.py --many-pass --url ...

See :mod:`multi_prompt_pkg` for the refactored modules.
"""

from __future__ import annotations

from multi_prompt_pkg import (
    CATEGORIES,
    DEFAULT_MODEL,
    FALLBACK_CATEGORY,
    LOCAL_BASE_URL,
    MIN_REAL_SUMMARY_LEN,
    CategoryOutput,
    LLMClient,
    MultiPromptSummarizer,
    PaperJob,
    PaperResult,
    PitchOutput,
    SECTION_SPECS,
    SECTION_SPECS_2PASS,
    SYSTEM_PREAMBLE,
    SectionSpec,
    _SYSTEM_PREAMBLE,
    append_jsonl,
    arxiv_id_from_url,
    arxiv_url_to_pdf_url,
    build_backfill_jobs,
    build_gemini_client,
    build_local_client,
    download_and_extract_text,
    download_pdf_bytes,
    get_db,
    get_interested_stubs,
    get_stub_ids,
    normalize_title_for_filename,
    paper_has_real_summary,
    parse_retry_delay,
    process_multiple,
    process_paper,
    raise_fd_limit,
    retry_async,
    run_dynamic,
    save_summary_markdown,
)
from multi_prompt_pkg.cli import main

__all__ = [
    "CATEGORIES",
    "CategoryOutput",
    "DEFAULT_MODEL",
    "FALLBACK_CATEGORY",
    "LLMClient",
    "LOCAL_BASE_URL",
    "MIN_REAL_SUMMARY_LEN",
    "MultiPromptSummarizer",
    "PaperJob",
    "PaperResult",
    "PitchOutput",
    "SECTION_SPECS",
    "SECTION_SPECS_2PASS",
    "SYSTEM_PREAMBLE",
    "SectionSpec",
    "_SYSTEM_PREAMBLE",
    "append_jsonl",
    "arxiv_id_from_url",
    "arxiv_url_to_pdf_url",
    "build_backfill_jobs",
    "build_gemini_client",
    "build_local_client",
    "download_and_extract_text",
    "download_pdf_bytes",
    "get_db",
    "get_interested_stubs",
    "get_stub_ids",
    "main",
    "normalize_title_for_filename",
    "paper_has_real_summary",
    "parse_retry_delay",
    "process_multiple",
    "process_paper",
    "raise_fd_limit",
    "retry_async",
    "run_dynamic",
    "save_summary_markdown",
]


if __name__ == "__main__":
    main()
