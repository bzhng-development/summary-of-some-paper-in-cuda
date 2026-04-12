"""Shared reading-history loader for the scoring and tagging scripts.

Both ``hf_daily_papers.py`` (LLM scorer) and ``papers_by_score.py`` (per-score
markdown splitter) need the exact same notion of "the user's reading history":
titles grouped by category, loaded from three sources and deduplicated.

Sources, in priority order:

1. The Neon Postgres ``nextjs-ui_paper`` table, filtered to ``interested = 1``.
   This is the canonical, mutable source (marked from the paper_server UI).
2. ``external_papers.db`` — still a SQLite file committed at the repo root. All
   external/non-arxiv papers live here and count as known-relevant history.
3. ``docs/**/*.md`` — every paper the user has manually written about. The
   directory name of each markdown file is its category.

Factoring this into one module kills ~60 lines of duplicated SQLite/IO logic
and gives both consumers a single place to change when the history sources
move again.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from neon_db import NeonDB


__all__ = [
    "ExamplePaper",
    "load_examples",
    "build_examples_block",
]


@dataclass(frozen=True, slots=True)
class ExamplePaper:
    """A single reading-history entry: a title and its category bucket."""

    title: str
    category: str


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_from_neon(db: NeonDB, out: dict[str, ExamplePaper]) -> None:
    """Populate ``out`` with rows marked interested in Neon."""
    try:
        rows = db.get_interested()
    except Exception as exc:
        logger.warning("Could not read interested rows from Neon: {}", exc)
        return

    for row in rows:
        title = row.get("title")
        if not title:
            continue
        out.setdefault(
            title,
            ExamplePaper(title=title, category=row.get("category") or "uncategorized"),
        )

    logger.debug("Loaded {} interested papers from Neon", len(out))


def _load_from_external_sqlite(repo_path: Path, out: dict[str, ExamplePaper]) -> None:
    """Pull (title, category) out of the committed ``external_papers.db`` file."""
    ext_db = repo_path / "external_papers.db"
    try:
        con = sqlite3.connect(str(ext_db))
    except sqlite3.OperationalError:
        return

    try:
        cursor = con.execute(
            "SELECT title, category FROM papers WHERE title IS NOT NULL"
        )
        for title, category in cursor:
            if title and title not in out:
                out[title] = ExamplePaper(
                    title=title,
                    category=category or "uncategorized",
                )
    finally:
        con.close()


_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _load_from_docs(repo_path: Path, out: dict[str, ExamplePaper]) -> None:
    """Walk ``docs/`` and treat each markdown file as a categorized entry."""
    docs_dir = repo_path / "docs"
    if not docs_dir.exists():
        return

    for md_file in docs_dir.rglob("*.md"):
        if md_file.name == "index.md":
            continue
        category = md_file.parent.name

        try:
            first_lines = md_file.read_text(errors="replace")[:500]
        except OSError:
            title = md_file.stem.replace("-", " ")
        else:
            match = _H1_RE.search(first_lines)
            title = match.group(1).strip() if match else md_file.stem.replace("-", " ")

        if title not in out:
            out[title] = ExamplePaper(title=title, category=category)


def load_examples(repo_path: Path, db: NeonDB | None = None) -> list[ExamplePaper]:
    """Load known-relevant papers from Neon, external SQLite, and docs/.

    ``db`` is optional only so callers can substitute a shared instance. If it
    is ``None`` a fresh :class:`NeonDB` is constructed — the connection is
    short-lived per Neon best practices.
    """
    client = db if db is not None else NeonDB()
    examples: dict[str, ExamplePaper] = {}

    _load_from_neon(client, examples)
    _load_from_external_sqlite(repo_path, examples)
    _load_from_docs(repo_path, examples)

    logger.info("Loaded {} example papers total", len(examples))
    return list(examples.values())


# ---------------------------------------------------------------------------
# Prompt block formatting
# ---------------------------------------------------------------------------


def build_examples_block(
    examples: list[ExamplePaper],
    *,
    header: str | None = None,
) -> str:
    """Render a reading-history block for an LLM system prompt.

    Both callers render the same shape: a header line followed by
    ``[category]`` groups with bulleted titles. The optional ``header`` argument
    lets ``papers_by_score.py`` inject the paper count while ``hf_daily_papers``
    uses the default.
    """
    if not examples:
        return ""

    by_cat: dict[str, list[str]] = {}
    for ex in examples:
        by_cat.setdefault(ex.category, []).append(ex.title)

    top = header if header is not None else "The user's reading history:"
    lines: list[str] = [f"{top}\n"]
    for cat in sorted(by_cat):
        lines.append(f"[{cat}]")
        for title in by_cat[cat]:
            lines.append(f"  - {title}")
    return "\n".join(lines)
