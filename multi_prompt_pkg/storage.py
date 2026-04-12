"""Persistence helpers: JSONL append, markdown export, and Neon DB access."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from neon_db import NeonDB

from .config import MIN_REAL_SUMMARY_LEN
from .pdf import arxiv_id_from_url
from .schemas import PitchOutput

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
#
# Every call site inside this package reuses one NeonDB instance. Construction
# is cheap (it just captures DATABASE_URL) and short-lived connections are
# opened per-query inside NeonDB itself.

_db: NeonDB | None = None


def get_db() -> NeonDB:
    """Return a lazily constructed module-level :class:`NeonDB`."""
    global _db
    if _db is None:
        _db = NeonDB()
    return _db


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def normalize_title_for_filename(title: str, *, max_length: int = 80) -> str:
    """Produce a filesystem-safe slug for a paper title."""
    cleaned = "".join(ch for ch in title if ord(ch) >= 32)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned).strip("-")[:max_length]


def _unique_output_path(candidate: Path) -> Path:
    """Return ``candidate`` or a suffixed variant if the target already exists."""
    try:
        # EAFP — create exclusively; retry with a numeric suffix on collision.
        candidate.touch(exist_ok=False)
        return candidate
    except FileExistsError:
        pass

    stem = candidate.stem
    suffix = candidate.suffix
    parent = candidate.parent
    for k in range(2, 1000):
        alt = parent / f"{stem}-{k}{suffix}"
        try:
            alt.touch(exist_ok=False)
            return alt
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not allocate a unique output path near {candidate}")


def save_summary_markdown(
    pitch_output: PitchOutput,
    full_summary: str,
    category: str,
    *,
    arxiv_url: str | None = None,
    paper_id: str | None = None,
) -> Path:
    """Write a paper summary to ``docs/<category>/<slug>.md`` and return the path."""
    category_dir = Path("docs") / category
    category_dir.mkdir(parents=True, exist_ok=True)

    if paper_id is not None and paper_id.startswith("ext:"):
        arxiv_id: str | None = None
        file_id = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id.removeprefix("ext:")).strip("-")
        arxiv_link = f"\n**URL:** [{arxiv_url}]({arxiv_url})\n" if arxiv_url else ""
    else:
        arxiv_id = arxiv_id_from_url(arxiv_url) if arxiv_url else None
        file_id = arxiv_id
        arxiv_link = (
            f"\n**ArXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n"
            if arxiv_id
            else ""
        )

    normalized_title = normalize_title_for_filename(pitch_output.title)
    if file_id:
        base_name = (
            f"{file_id}-{normalized_title}.md" if normalized_title else f"{file_id}.md"
        )
    else:
        base_name = f"{normalized_title}.md" if normalized_title else "paper.md"

    output_file = _unique_output_path(category_dir / base_name)
    formatted_output = (
        f"# {pitch_output.title}\n"
        f"{arxiv_link}\n"
        "## Pitch\n\n"
        f"{pitch_output.pitch}\n\n"
        "---\n\n"
        f"{full_summary}\n"
    )
    output_file.write_text(formatted_output)
    return output_file


# ---------------------------------------------------------------------------
# JSONL sidecar
# ---------------------------------------------------------------------------


def append_jsonl(record: dict[str, Any], output_path: str | Path) -> None:
    """Append a paper record as one JSON line."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Paper {} appended to {}", record.get("arxiv_id", "?"), path)


def load_done_ids_from_jsonl(path: str | Path) -> set[str]:
    """Return the set of ``arxiv_id`` values already present in a JSONL file."""
    jsonl_path = Path(path)
    done: set[str] = set()
    try:
        text = jsonl_path.read_text()
    except FileNotFoundError:
        return done
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        paper_id = record.get("arxiv_id")
        if isinstance(paper_id, str):
            done.add(paper_id)
    return done


# ---------------------------------------------------------------------------
# DB queries used by the CLI
# ---------------------------------------------------------------------------


def paper_has_real_summary(
    arxiv_id: str, *, min_summary_len: int = MIN_REAL_SUMMARY_LEN
) -> bool:
    """Return True iff ``arxiv_id`` already has a pipeline-grade summary."""
    row = get_db().get_paper(arxiv_id)
    if row is None:
        return False
    summary = row.get("summary")
    return isinstance(summary, str) and len(summary) >= min_summary_len


def get_stub_ids() -> list[str]:
    """IDs of non-ext papers whose summary is missing or blank (``--dyn``)."""
    rows = get_db().get_stubs_without_summary()
    return [row["id"] for row in rows]


def get_interested_stubs() -> list[tuple[str, str | None]]:
    """``(id, url)`` tuples for ``interested=1`` papers missing a summary."""
    rows = get_db().get_stubs_without_summary(interested_only=True)
    return [(row["id"], row.get("url")) for row in rows]
