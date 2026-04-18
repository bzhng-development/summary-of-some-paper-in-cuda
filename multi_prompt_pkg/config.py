"""Shared configuration for the long-form paper summarization pipeline."""

from __future__ import annotations

import os
from typing import Final

# Port of the local OpenAI-compatible LLM server (SGLang/vLLM/llama.cpp).
# Override via ``LOCAL_LLM_PORT`` env var. The scorer in
# ``daily_papers/hf_daily_papers.py`` reads the same variable, so setting it
# once in your shell configures both the summarizer and the scorer.
LOCAL_LLM_PORT: Final[int] = int(os.environ.get("LOCAL_LLM_PORT", "30000"))
LOCAL_BASE_URL: Final[str] = f"http://localhost:{LOCAL_LLM_PORT}/v1"
DEFAULT_MODEL: Final[str] = "default"

# Minimum character length for a ``summary`` column value to count as a real,
# pipeline-generated summary (rather than a stub / abstract).
MIN_REAL_SUMMARY_LEN: Final[int] = 5000

CATEGORIES: Final[tuple[str, ...]] = (
    "agents",
    "alignment",
    "architecture",
    "code",
    "context-optimization",
    "data",
    "diffusion",
    "distributed-training",
    "evaluation",
    "inference-optimization",
    "llm-systems",
    "low-precision",
    "moe",
    "multimodal",
    "pretraining",
    "prompting",
    "reasoning",
    "retrieval",
    "rl-training",
    "safety",
    "scaling-laws",
    "serving",
    "training-methods",
    "vision",
)
FALLBACK_CATEGORY: Final[str] = "uncategorized"
