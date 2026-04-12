"""Shared configuration for the long-form paper summarization pipeline."""

from __future__ import annotations

from typing import Final

LOCAL_BASE_URL: Final[str] = "http://localhost:30000/v1"
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
