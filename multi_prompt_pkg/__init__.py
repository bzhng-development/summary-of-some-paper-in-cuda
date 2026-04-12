"""Long-form paper summarization pipeline.

This package is the refactored home of what used to live in the 1041-LOC
``multi_prompt.py`` module. The top-level ``multi_prompt.py`` file is now a
thin re-export shim so that both the CLI entry point
``uv run python multi_prompt.py ...`` and ``from multi_prompt import ...``
import paths keep working.
"""

from __future__ import annotations

from .config import (
    CATEGORIES,
    DEFAULT_MODEL,
    FALLBACK_CATEGORY,
    LOCAL_BASE_URL,
    MIN_REAL_SUMMARY_LEN,
)
from .llm import (
    LLMClient,
    build_gemini_client,
    build_local_client,
    parse_retry_delay,
    raise_fd_limit,
    retry_async,
)
from .pdf import (
    arxiv_id_from_url,
    arxiv_url_to_pdf_url,
    download_and_extract_text,
    download_pdf_bytes,
)
from .pipeline import (
    PaperJob,
    PaperResult,
    build_backfill_jobs,
    process_multiple,
    process_paper,
    run_dynamic,
)
from .prompts import (
    SECTION_SPECS,
    SECTION_SPECS_2PASS,
    SYSTEM_PREAMBLE,
    SectionSpec,
)
from .schemas import CategoryOutput, PitchOutput
from .storage import (
    append_jsonl,
    get_db,
    get_interested_stubs,
    get_stub_ids,
    normalize_title_for_filename,
    paper_has_real_summary,
    save_summary_markdown,
)
from .summarizer import MultiPromptSummarizer

# Legacy alias retained so ``from multi_prompt import _SYSTEM_PREAMBLE`` in
# throaway_script/ keeps working during the migration.
_SYSTEM_PREAMBLE = SYSTEM_PREAMBLE

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
