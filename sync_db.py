#!/usr/bin/env python3
"""Unify all paper sources into the Neon ``nextjs-ui_paper`` table.

Default mode (no flags): full sync pipeline.

    1. Migrate schema (``NeonDB.init_schema``)
    2. Backfill stubs from ``docs/``
    3. Absorb ``arxiv_index.txt``
    4. Merge ``external_papers.db`` (still SQLite)
    5. Absorb ``all_scored.json`` (daily papers)
    6. Enrich from arxiv API
    7. Import tags from JSONL (if ``--import-tags`` given or
       ``tagged_papers.jsonl`` exists)

Export mode: dump papers to JSONL for remote tagging.
Import mode: import tagged results from JSONL.

Usage:
    uv run python sync_db.py                                    # full sync
    uv run python sync_db.py --skip-arxiv                       # skip slow arxiv API
    uv run python sync_db.py --export papers_to_tag.jsonl       # export for remote tagging
    uv run python sync_db.py --import-tags tagged_papers.jsonl  # import tags
    uv run python sync_db.py --import-tags tagged.jsonl --dry-run
    uv run python sync_db.py --compact                          # compact all_scored.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from daily_papers.hf_daily_papers import fetch_arxiv_metadata
from neon_db import TABLE, NeonDB


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent
SCORED_PATH = REPO_ROOT / "daily_papers" / "papers_out" / "all_scored.json"
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_TAGS_PATH = REPO_ROOT / "tagged_papers.jsonl"
ARXIV_INDEX_PATH = REPO_ROOT / "arxiv_index.txt"
EXTERNAL_DB_PATH = REPO_ROOT / "external_papers.db"

_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_ID_SEARCH_RE = re.compile(r"(\d{4}\.\d{4,5})")
_ARXIV_LINE_RE = re.compile(r"\*\*ArXiv:\*\*\s*\[(\d{4}\.\d{4,5})\]")
_PITCH_RE = re.compile(r"## 🎯 Pitch\s*\n(.*?)(?=\n---)", re.DOTALL)


# ---------------------------------------------------------------------------
# Step 2: backfill from docs/
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParsedDoc:
    arxiv_id: str
    title: str | None
    category: str
    pitch: str | None
    summary: str


def _parse_doc_markdown(md_file: Path) -> _ParsedDoc | None:
    """Extract (id, title, category, pitch, summary) from one markdown file.

    EAFP: we read the file first and fail fast on any decode issue rather than
    pre-checking with ``md_file.exists()``.
    """
    try:
        text = md_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    arxiv_id: str | None = None
    match_name = _ARXIV_ID_SEARCH_RE.match(md_file.name)
    if match_name:
        arxiv_id = match_name.group(1)
    else:
        match_line = _ARXIV_LINE_RE.search(text)
        if match_line:
            arxiv_id = match_line.group(1)
    if arxiv_id is None:
        return None

    title: str | None = None
    for line in text.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    pitch: str | None = None
    pitch_match = _PITCH_RE.search(text)
    if pitch_match:
        pitch = pitch_match.group(1).strip()

    parts = text.split("---", 2)
    if len(parts) >= 3:
        summary = parts[2].strip()
    elif len(parts) == 2:
        summary = parts[1].strip()
    else:
        summary = ""
    if not summary:
        return None

    return _ParsedDoc(
        arxiv_id=arxiv_id,
        title=title,
        category=md_file.parent.name,
        pitch=pitch,
        summary=summary,
    )


def backfill_from_docs(db: NeonDB, docs_dir: Path = DOCS_DIR) -> int:
    """Parse ``docs/**/*.md`` and fill in missing summaries in Neon."""
    if not docs_dir.exists():
        logger.warning("{} not found, skipping", docs_dir)
        return 0

    stubs = {row["id"] for row in db.get_stubs_without_summary()}
    if not stubs:
        logger.info("No stubs to backfill")
        return 0

    filled = 0
    for md_file in docs_dir.rglob("*.md"):
        if md_file.name == "index.md":
            continue
        parsed = _parse_doc_markdown(md_file)
        if parsed is None or parsed.arxiv_id not in stubs:
            continue

        db.save_paper(
            parsed.arxiv_id,
            title=parsed.title,
            category=parsed.category,
            pitch=parsed.pitch,
            summary=parsed.summary,
            url=f"https://arxiv.org/abs/{parsed.arxiv_id}",
        )
        filled += 1

    logger.info("Backfilled {}/{} stubs from docs/", filled, len(stubs))
    return filled


# ---------------------------------------------------------------------------
# Step 3: absorb arxiv_index.txt
# ---------------------------------------------------------------------------


def absorb_arxiv_index(db: NeonDB, index_path: Path = ARXIV_INDEX_PATH) -> int:
    """Insert stub rows for every ID in ``arxiv_index.txt`` not already in Neon."""
    try:
        raw = index_path.read_text()
    except FileNotFoundError:
        logger.warning("{} not found, skipping", index_path)
        return 0

    existing = db.get_all_ids()
    new_ids = [
        line
        for line in (raw_line.strip() for raw_line in raw.splitlines())
        if line
        and not line.startswith("#")
        and _ARXIV_ID_RE.match(line)
        and line not in existing
    ]

    for arxiv_id in new_ids:
        db.save_paper(arxiv_id, url=f"https://arxiv.org/abs/{arxiv_id}")

    logger.info(
        "Absorbed {} new IDs from {} ({} already in DB)",
        len(new_ids),
        index_path,
        len(existing),
    )
    return len(new_ids)


# ---------------------------------------------------------------------------
# Step 4: merge external_papers.db
# ---------------------------------------------------------------------------


def _derive_external_id(row: dict[str, Any]) -> str:
    """Recover a usable paper id from an ``external_papers.db`` row."""
    paper_id = row.get("paper_id") or ""
    source_url = row.get("source_url") or ""

    if row.get("source") == "arxiv" and paper_id:
        return paper_id
    match_url = _ARXIV_ID_SEARCH_RE.search(source_url)
    if match_url:
        return match_url.group(1)
    if _ARXIV_ID_RE.match(paper_id):
        return paper_id
    return f"ext:{paper_id}"


def merge_external_db(db: NeonDB, external_path: Path = EXTERNAL_DB_PATH) -> int:
    """Copy rows from the (still-SQLite) external DB into Neon.

    ``external_papers.db`` is intentionally not migrated to Neon — it's a
    small, committed, read-only corpus. We read it with stdlib ``sqlite3`` and
    upsert each row into Neon.
    """
    try:
        ext_con = sqlite3.connect(str(external_path))
    except sqlite3.OperationalError:
        logger.warning("{} not found, skipping", external_path)
        return 0

    try:
        ext_con.row_factory = sqlite3.Row
        rows = [dict(r) for r in ext_con.execute("SELECT * FROM papers")]
    finally:
        ext_con.close()

    existing = db.get_all_ids()
    merged = 0
    for row in rows:
        arxiv_id = _derive_external_id(row)
        if arxiv_id in existing:
            continue

        first_author = row.get("first_author") or ""
        authors = [first_author] if first_author else None
        published = str(row["year"]) if row.get("year") else None

        db.save_paper(
            arxiv_id,
            title=row.get("title"),
            category=row.get("category"),
            summary=row.get("summary"),
            url=row.get("source_url") or None,
            authors=authors,
            published=published,
            journal_ref=row.get("venue"),
            doi=row.get("doi"),
        )
        existing.add(arxiv_id)
        merged += 1

    logger.info("Merged {} papers from {}", merged, external_path)
    return merged


# ---------------------------------------------------------------------------
# Step 5: absorb all_scored.json
# ---------------------------------------------------------------------------


def absorb_scored_json(db: NeonDB, scored_path: Path = SCORED_PATH) -> int:
    """Absorb ``all_scored.json`` entries (title, upvotes, github, etc.)."""
    try:
        data = json.loads(scored_path.read_text())
    except FileNotFoundError:
        logger.warning("{} not found, skipping", scored_path)
        return 0

    existing = db.get_all_ids()
    absorbed = 0

    for entry in data:
        aid = entry.get("arxiv_id")
        if not aid or not entry.get("title"):
            continue

        db.save_paper(
            aid,
            title=entry["title"],
            # Scored JSON "summary" is actually the arxiv abstract.
            abstract=entry.get("summary") or None,
            url=f"https://arxiv.org/abs/{aid}",
            upvotes=entry.get("upvotes"),
            github=entry.get("github"),
            github_stars=entry.get("github_stars"),
            authors=entry.get("authors") or None,
            affiliations=entry.get("affiliations") or None,
            organization=entry.get("organization"),
            org_fullname=entry.get("org_fullname"),
            categories=entry.get("categories") or None,
            primary_category=entry.get("primary_category"),
            arxiv_comment=entry.get("arxiv_comment"),
            published=entry.get("published"),
            journal_ref=entry.get("journal_ref"),
            doi=entry.get("doi"),
        )
        if aid not in existing:
            absorbed += 1
            existing.add(aid)

    logger.info(
        "Absorbed {} new papers from {} ({} total entries)",
        absorbed,
        scored_path.name,
        len(data),
    )
    return absorbed


# ---------------------------------------------------------------------------
# Step 6: enrich from arxiv API
# ---------------------------------------------------------------------------


def enrich_from_arxiv(db: NeonDB) -> int:
    """Fetch arxiv metadata for every paper that's still missing it."""
    sql = f"""
        SELECT id FROM {TABLE}
        WHERE id NOT LIKE 'ext:%'
          AND (
            categories IS NULL
            OR abstract IS NULL OR abstract = ''
            OR authors IS NULL OR authors = ''
            OR published IS NULL OR published = ''
          )
    """
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        ids = [row[0] for row in cur.fetchall()]

    if not ids:
        logger.info("All papers already enriched, nothing to fetch")
        return 0

    logger.info("Enriching {} papers from arxiv API...", len(ids))
    meta_map = fetch_arxiv_metadata(ids)

    enriched = 0
    for arxiv_id, meta in meta_map.items():
        db.save_paper(
            arxiv_id,
            title=meta.title or None,
            abstract=meta.abstract or None,
            authors=meta.authors or None,
            affiliations=meta.affiliations or None,
            categories=meta.categories or None,
            primary_category=meta.primary_category,
            arxiv_comment=meta.comment,
            published=meta.published,
            journal_ref=meta.journal_ref,
            doi=meta.doi,
        )
        enriched += 1

    logger.info("Enriched {}/{} papers with arxiv metadata", enriched, len(ids))
    return enriched


# ---------------------------------------------------------------------------
# Tag export / import
# ---------------------------------------------------------------------------


def export_for_tagging(db: NeonDB, output: Path) -> None:
    """Export papers from Neon to a JSONL for remote tagging."""
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT id, title, abstract FROM {TABLE} WHERE title IS NOT NULL"
        )
        rows = cur.fetchall()

    already: set[str] = set()
    try:
        existing_lines = output.read_text().splitlines()
    except FileNotFoundError:
        existing_lines = []
    for line in existing_lines:
        line = line.strip()
        if not line:
            continue
        try:
            already.add(json.loads(line)["arxiv_id"])
        except (json.JSONDecodeError, KeyError):
            continue

    written = 0
    with output.open("a") as f:
        for row_id, title, abstract in rows:
            if row_id in already:
                continue
            record = {
                "arxiv_id": row_id,
                "title": title,
                "abstract": (abstract or "")[:2000],
            }
            f.write(json.dumps(record) + "\n")
            written += 1

    logger.success(
        "Exported {} papers to {} ({} already present)",
        written,
        output,
        len(already),
    )


def load_tags(path: Path, *, min_confidence: float = 0.0) -> dict[str, dict]:
    """Load tagged results from JSONL, keyed by arxiv_id."""
    tags: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("confidence", 1.0) < min_confidence:
            continue
        aid = entry.get("arxiv_id")
        if aid:
            tags[aid] = entry
    return tags


def import_tags(
    db: NeonDB,
    tags_path: Path,
    *,
    min_confidence: float = 0.0,
    dry_run: bool = False,
    compact_scored: bool = False,
) -> None:
    """Import tagged results: update Neon categories, scored JSON, and docs/."""
    tags = load_tags(tags_path, min_confidence=min_confidence)
    logger.info("Loaded {} tags from {}", len(tags), tags_path)

    if dry_run:
        logger.info("--- DRY RUN ---")

    db_updated = _update_db_categories(db, tags, dry_run)
    logger.info(
        "DB: {} papers {}updated",
        db_updated,
        "would be " if dry_run else "",
    )

    scored_updated = _update_scored_json(tags, dry_run, compact=compact_scored)
    logger.info(
        "Scored JSON: {} papers {}updated",
        scored_updated,
        "would be " if dry_run else "",
    )

    docs_moved = _move_doc_files(tags, dry_run)
    logger.info(
        "Docs: {} files {}moved",
        docs_moved,
        "would be " if dry_run else "",
    )


def _update_db_categories(
    db: NeonDB,
    tags: dict[str, dict],
    dry_run: bool,
) -> int:
    """Write ``category`` for tagged papers that disagree with Neon."""
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT id, category FROM {TABLE}")
        existing = {row[0]: row[1] for row in cur.fetchall()}

    updated = 0
    for aid, tag in tags.items():
        if aid not in existing:
            continue
        old_cat = existing[aid]
        new_cat = tag["category"]
        if old_cat == new_cat:
            continue
        if dry_run:
            logger.debug("  [DB] {}: {} -> {}", aid, old_cat, new_cat)
        else:
            db.update_category(aid, new_cat)
        updated += 1
    return updated


def _update_scored_json(
    tags: dict[str, dict],
    dry_run: bool,
    *,
    compact: bool = False,
) -> int:
    """Merge tag categories into ``all_scored.json``.

    By default this rewrites the file with ``json.dumps(data, indent=2)``,
    which is noisy but preserves the historical diff format. Pass
    ``compact=True`` to use ``separators=(",", ":")`` and skip the 40 MB
    re-indent churn.
    """
    try:
        data = json.loads(SCORED_PATH.read_text())
    except FileNotFoundError:
        return 0

    updated = 0
    for entry in data:
        aid = entry.get("arxiv_id")
        if not aid or aid not in tags:
            continue
        new_cat = tags[aid]["category"]
        if entry.get("tag_category") == new_cat:
            continue
        if not dry_run:
            entry["tag_category"] = new_cat
            entry["tag_confidence"] = tags[aid].get("confidence")
            entry["tag_reason"] = tags[aid].get("reason")
        updated += 1

    if not dry_run and updated:
        if compact:
            SCORED_PATH.write_text(json.dumps(data, separators=(",", ":")))
        else:
            SCORED_PATH.write_text(json.dumps(data, indent=2))
    return updated


def _move_doc_files(tags: dict[str, dict], dry_run: bool) -> int:
    """Move ``docs/**/*.md`` files when the tagger has reassigned their category."""
    if not DOCS_DIR.exists():
        return 0

    moved = 0
    for aid, tag in tags.items():
        new_cat = tag["category"]
        for md_file in DOCS_DIR.rglob(f"{aid}*.md"):
            if md_file.parent.name == new_cat:
                continue
            new_dir = DOCS_DIR / new_cat
            new_path = new_dir / md_file.name
            if dry_run:
                logger.debug(
                    "  [docs] {} -> {}",
                    md_file.relative_to(REPO_ROOT),
                    new_path.relative_to(REPO_ROOT),
                )
            else:
                new_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(md_file), str(new_path))
            moved += 1
    return moved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _summarize(db: NeonDB) -> None:
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
        total = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE title IS NOT NULL")
        has_title = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE categories IS NOT NULL")
        has_cats = cur.fetchone()[0]
        cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE category IS NOT NULL")
        has_category = cur.fetchone()[0]

    logger.success(
        "Done! {} papers in DB ({} with titles, {} with arxiv metadata, {} categorized)",
        total,
        has_title,
        has_cats,
        has_category,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync the Neon papers table — full pipeline, export, or import tags",
    )
    parser.add_argument(
        "--skip-arxiv", action="store_true", help="Skip arxiv API enrichment"
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional DATABASE_URL override (otherwise read from env)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        metavar="FILE",
        help="Export papers to JSONL for remote tagging",
    )
    parser.add_argument(
        "--import-tags",
        type=Path,
        metavar="FILE",
        dest="import_tags",
        help="Import tagged JSONL results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview import changes without applying",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Skip tags below this confidence",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Write all_scored.json with compact separators instead of indent=2. "
            "Avoids the noisy 40 MB diff on every import."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    db = NeonDB(database_url=args.db) if args.db else NeonDB()

    # --- Export mode ---
    if args.export:
        db.init_schema()
        export_for_tagging(db, args.export)
        return

    # --- Import-only mode ---
    if args.import_tags:
        import_tags(
            db,
            args.import_tags,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            compact_scored=args.compact,
        )
        return

    # --- Full sync pipeline ---
    logger.info("Step 1/7: Migrating schema...")
    db.init_schema()

    logger.info("Step 2/7: Backfilling stubs from docs/...")
    backfill_from_docs(db)

    logger.info("Step 3/7: Absorbing arxiv_index.txt...")
    absorb_arxiv_index(db)

    logger.info("Step 4/7: Merging external_papers.db...")
    merge_external_db(db)

    logger.info("Step 5/7: Absorbing all_scored.json...")
    absorb_scored_json(db)

    if not args.skip_arxiv:
        logger.info("Step 6/7: Enriching from arxiv API...")
        enrich_from_arxiv(db)
    else:
        logger.info("Step 6/7: Skipped (--skip-arxiv)")

    if DEFAULT_TAGS_PATH.exists():
        logger.info("Step 7/7: Importing tags from tagged_papers.jsonl...")
        import_tags(
            db,
            DEFAULT_TAGS_PATH,
            min_confidence=args.min_confidence,
            dry_run=args.dry_run,
            compact_scored=args.compact,
        )
    else:
        logger.info("Step 7/7: No tagged_papers.jsonl found, skipping tag import")

    _summarize(db)


if __name__ == "__main__":
    main()
