"""Canonical organization scope for company-paper discovery.

The company-paper corpus intentionally excludes pure university feeds. Broad
institution queries pull every department represented in OpenAlex or Semantic
Scholar, which contaminates the company-focused corpus with unrelated physics,
medicine, and other academic work.

Papers from university authors can still enter through a tracked company or
industrial research lab. Only discovery *whose source organization is purely
academic* is rejected.
"""

from __future__ import annotations

from typing import Final

PURE_ACADEMIC_ORG_LABELS: Final[frozenset[str]] = frozenset(
    {
        # Broad OpenAlex institution feeds.
        "Berkeley",
        "Stanford",
        "CMU",
        "UW",
        "MIT",
        "Princeton",
        "NYU",
        "Tsinghua",
        # Semantic Scholar and browser university-lab feeds.
        "Berkeley BAIR",
        "Stanford NLP",
        "CMU LTI",
        "UW NLP",
        "Tsinghua University",
        "Berkeley-BAIR",
        "Stanford-NLP",
        "CMU-LTI",
        "UW-NLP",
        "THUDM",
        "THUNLP",
        "Tsinghua-THUDM",
        "KAIST AI",
    }
)

PURE_ACADEMIC_HF_NAMESPACES: Final[frozenset[str]] = frozenset(
    {
        "THUDM",
        "thunlp",
        "berkeley-nest",
        "stanfordnlp",
        "stanford-crfm",
        "cmu-lti",
        "uw-nlp",
    }
)

# These labels identify the broad institution-wide OpenAlex cohort already in
# Neon. Cleanup is intentionally narrower than the discovery denylist so it
# does not delete older hand-curated AI-lab papers.
BROAD_ACADEMIC_IMPORT_LABELS: Final[frozenset[str]] = frozenset(
    {
        "Berkeley",
        "Stanford",
        "CMU",
        "UW",
        "MIT",
        "Princeton",
        "NYU",
        "Tsinghua",
    }
)


def is_pure_academic_org(label: str | None) -> bool:
    """Return whether ``label`` represents a pure university discovery feed."""
    return bool(label and label.strip() in PURE_ACADEMIC_ORG_LABELS)


def is_pure_academic_hf_namespace(namespace: str | None) -> bool:
    """Return whether a Hugging Face namespace belongs to a university lab."""
    return bool(namespace and namespace.strip() in PURE_ACADEMIC_HF_NAMESPACES)
