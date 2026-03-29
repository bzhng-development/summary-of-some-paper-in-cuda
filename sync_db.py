#!/usr/bin/env python3
"""Unify all paper sources into papers.db with full arxiv metadata.

Default mode (no flags): full sync pipeline.
  1. Migrate papers.db schema
  2. Backfill stubs from docs/
  3. Absorb arxiv_index.txt
  4. Merge external_papers.db
  5. Absorb all_scored.json (daily papers)
  6. Enrich from arxiv API
  7. Import tags from JSONL (if --import-tags given or tagged_papers.jsonl exists)

Export mode: dump papers to JSONL for remote tagging.
Import mode: import tagged results from JSONL.

Usage:
    uv run python sync_db.py                                    # full sync
    uv run python sync_db.py --skip-arxiv                       # skip slow arxiv API
    uv run python sync_db.py --export papers_to_tag.jsonl       # export for remote tagging
    uv run python sync_db.py --import-tags tagged_papers.jsonl  # import tags
    uv run python sync_db.py --import-tags tagged.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
from pathlib import Path

from loguru import logger

from database import DB_PATH, _migrate_sync, save_paper_sync, get_all_ids_sync
from daily_papers.hf_daily_papers import fetch_arxiv_metadata

REPO_ROOT = Path(__file__).resolve().parent
SCORED_PATH = REPO_ROOT / "daily_papers" / "papers_out" / "all_scored.json"
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_TAGS_PATH = REPO_ROOT / "tagged_papers.jsonl"


# =============================================================================
# Sync steps
# =============================================================================


def backfill_from_docs(docs_dir: Path, db_path: str = DB_PATH) -> int:
    """Parse markdown files in docs/ and backfill DB rows that are missing summaries."""
    if not docs_dir.exists():
        logger.warning(f"{docs_dir} not found, skipping")
        return 0

    # Find stubs: papers with no summary
    con = sqlite3.connect(db_path)
    stubs = {
        row[0]
        for row in con.execute(
            "SELECT id FROM papers WHERE summary IS NULL OR summary = ''"
        ).fetchall()
    }
    con.close()

    if not stubs:
        logger.info("No stubs to backfill")
        return 0

    filled = 0
    for md_file in docs_dir.rglob("*.md"):
        if md_file.name == "index.md":
            continue
        text = md_file.read_text(encoding="utf-8")

        # Extract arxiv ID: try filename first, then **ArXiv:** line in content
        arxiv_id = None
        m = re.match(r"(\d{4}\.\d{4,5})", md_file.name)
        if m:
            arxiv_id = m.group(1)
        else:
            m = re.search(r"\*\*ArXiv:\*\*\s*\[(\d{4}\.\d{4,5})\]", text)
            if m:
                arxiv_id = m.group(1)
        if not arxiv_id:
            continue
        if arxiv_id not in stubs:
            continue

        category = md_file.parent.name  # directory name = category

        # Parse title: first "# " line
        title = None
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        # Parse pitch: text between "## 🎯 Pitch" and the next "---"
        pitch = None
        pitch_match = re.search(r"## 🎯 Pitch\s*\n(.*?)(?=\n---)", text, re.DOTALL)
        if pitch_match:
            pitch = pitch_match.group(1).strip()

        # Summary: everything after the first "---" following the pitch
        summary = None
        parts = text.split("---", 2)
        if len(parts) >= 3:
            summary = parts[2].strip()
        elif len(parts) == 2:
            summary = parts[1].strip()

        if not summary:
            continue

        con = sqlite3.connect(db_path)
        con.execute(
            """INSERT INTO papers (id, title, category, pitch, summary, url)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   title = COALESCE(excluded.title, title),
                   category = COALESCE(excluded.category, category),
                   pitch = COALESCE(excluded.pitch, pitch),
                   summary = COALESCE(excluded.summary, summary),
                   url = COALESCE(excluded.url, url)""",
            (arxiv_id, title, category, pitch, summary, f"https://arxiv.org/abs/{arxiv_id}"),
        )
        con.commit()
        con.close()
        filled += 1

    logger.info(f"Backfilled {filled}/{len(stubs)} stubs from docs/")
    return filled


def absorb_arxiv_index(index_path: Path, db_path: str = DB_PATH) -> int:
    """Read arxiv_index.txt and insert stub rows for IDs not already in DB."""
    if not index_path.exists():
        logger.warning(f"{index_path} not found, skipping")
        return 0

    existing = get_all_ids_sync(db_path)
    new_ids = []
    for line in index_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept bare IDs like 1301.3781 or 2512.23343
        if re.match(r"^\d{4}\.\d{4,5}$", line):
            if line not in existing:
                new_ids.append(line)

    con = sqlite3.connect(db_path)
    for arxiv_id in new_ids:
        con.execute(
            "INSERT OR IGNORE INTO papers (id, url) VALUES (?, ?)",
            (arxiv_id, f"https://arxiv.org/abs/{arxiv_id}"),
        )
    con.commit()
    con.close()
    logger.info(f"Absorbed {len(new_ids)} new IDs from {index_path} ({len(existing)} already in DB)")
    return len(new_ids)


def merge_external_db(external_path: Path, db_path: str = DB_PATH) -> int:
    """Merge external_papers.db into papers.db."""
    if not external_path.exists():
        logger.warning(f"{external_path} not found, skipping")
        return 0

    ext_con = sqlite3.connect(str(external_path))
    ext_con.row_factory = sqlite3.Row
    rows = ext_con.execute("SELECT * FROM papers").fetchall()
    ext_con.close()

    existing = get_all_ids_sync(db_path)
    merged = 0
    for row in rows:
        row = dict(row)
        # Try to derive an arxiv_id from the paper_id or source_url
        paper_id = row.get("paper_id", "")
        source_url = row.get("source_url") or ""

        arxiv_id = None
        # Check if source is arxiv
        if row.get("source") == "arxiv":
            arxiv_id = paper_id
        # Try to extract from source_url
        if not arxiv_id:
            m = re.search(r"(\d{4}\.\d{4,5})", source_url)
            if m:
                arxiv_id = m.group(1)
        # Fall back to paper_id if it looks like an ID
        if not arxiv_id and re.match(r"^\d{4}\.\d{4,5}$", paper_id):
            arxiv_id = paper_id
        # Use the hash/doi-based ID as fallback (prefix with ext: to avoid collisions)
        if not arxiv_id:
            arxiv_id = f"ext:{paper_id}"

        if arxiv_id in existing:
            continue

        first_author = row.get("first_author") or ""
        authors = [first_author] if first_author else []
        published = str(row["year"]) if row.get("year") else None

        save_paper_sync(
            arxiv_id,
            title=row.get("title"),
            category=row.get("category"),
            summary=row.get("summary"),
            url=source_url or None,
            authors=authors,
            published=published,
            journal_ref=row.get("venue"),
            doi=row.get("doi"),
            db_path=db_path,
        )
        existing.add(arxiv_id)
        merged += 1

    logger.info(f"Merged {merged} papers from {external_path}")
    return merged


def absorb_scored_json(scored_path: Path = SCORED_PATH, db_path: str = DB_PATH) -> int:
    """Absorb all_scored.json entries into papers.db (title, upvotes, github, etc.)."""
    if not scored_path.exists():
        logger.warning(f"{scored_path} not found, skipping")
        return 0

    data = json.loads(scored_path.read_text())
    existing = get_all_ids_sync(db_path)
    absorbed = 0

    for entry in data:
        aid = entry.get("arxiv_id")
        if not aid or not entry.get("title"):
            continue

        # For papers already in DB, only fill in missing metadata.
        # For new papers, create a row with what we have.
        # all_scored.json "summary" field is the arxiv abstract — store in abstract column
        arxiv_abstract = entry.get("summary") or None
        save_paper_sync(
            aid,
            title=entry["title"],
            abstract=arxiv_abstract,
            url=f"https://arxiv.org/abs/{aid}",
            upvotes=entry.get("upvotes"),
            github=entry.get("github"),
            github_stars=entry.get("github_stars"),
            authors=entry.get("authors") if entry.get("authors") else None,
            affiliations=entry.get("affiliations") if entry.get("affiliations") else None,
            organization=entry.get("organization"),
            org_fullname=entry.get("org_fullname"),
            categories=entry.get("categories") if entry.get("categories") else None,
            primary_category=entry.get("primary_category"),
            arxiv_comment=entry.get("arxiv_comment"),
            published=entry.get("published"),
            journal_ref=entry.get("journal_ref"),
            doi=entry.get("doi"),
            db_path=db_path,
        )
        if aid not in existing:
            absorbed += 1
            existing.add(aid)

    logger.info(f"Absorbed {absorbed} new papers from {scored_path.name} ({len(data)} total entries)")
    return absorbed


def enrich_from_arxiv(db_path: str = DB_PATH) -> int:
    """Fetch arxiv metadata for all papers missing it and update DB."""
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT id FROM papers
        WHERE id NOT LIKE 'ext:%%'
          AND (
            categories IS NULL
            OR abstract IS NULL OR abstract = ''
            OR authors IS NULL OR authors = ''
            OR published IS NULL OR published = ''
          )
    """).fetchall()
    con.close()

    ids = [row[0] for row in rows]
    if not ids:
        logger.info("All papers already enriched, nothing to fetch")
        return 0

    logger.info(f"Enriching {len(ids)} papers from arxiv API...")
    meta_map = fetch_arxiv_metadata(ids)

    enriched = 0
    for arxiv_id, meta in meta_map.items():
        save_paper_sync(
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
            db_path=db_path,
        )
        enriched += 1

    logger.info(f"Enriched {enriched}/{len(ids)} papers with arxiv metadata")
    return enriched


# =============================================================================
# Tag export / import
# =============================================================================


def export_for_tagging(output: Path, db_path: str = DB_PATH) -> None:
    """Export papers from DB to JSONL for remote tagging."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, title, abstract FROM papers WHERE title IS NOT NULL"
    ).fetchall()
    con.close()

    # Skip papers already in the output JSONL
    already: set[str] = set()
    if output.exists():
        for line in output.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    already.add(json.loads(line)["arxiv_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    written = 0
    with open(output, "a") as f:
        for r in rows:
            if r["id"] in already:
                continue
            row = {
                "arxiv_id": r["id"],
                "title": r["title"],
                "abstract": (r["abstract"] or "")[:2000],
            }
            f.write(json.dumps(row) + "\n")
            written += 1

    logger.success(f"Exported {written} papers to {output} ({len(already)} already present)")


def load_tags(path: Path, min_confidence: float = 0.0) -> dict[str, dict]:
    """Load tagged results from JSONL."""
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
    tags_path: Path,
    db_path: str = DB_PATH,
    min_confidence: float = 0.0,
    dry_run: bool = False,
) -> None:
    """Import tagged results: update DB categories, scored JSON, and move doc files."""
    tags = load_tags(tags_path, min_confidence=min_confidence)
    logger.info(f"Loaded {len(tags)} tags from {tags_path}")

    if dry_run:
        logger.info("--- DRY RUN ---")

    # 1) Update papers.db
    db_updated = _update_db_categories(tags, db_path, dry_run)
    logger.info(f"DB: {db_updated} papers {'would be ' if dry_run else ''}updated")

    # 2) Update all_scored.json
    scored_updated = _update_scored_json(tags, dry_run)
    logger.info(f"Scored JSON: {scored_updated} papers {'would be ' if dry_run else ''}updated")

    # 3) Move doc files
    docs_moved = _move_doc_files(tags, dry_run)
    logger.info(f"Docs: {docs_moved} files {'would be ' if dry_run else ''}moved")


def _update_db_categories(tags: dict[str, dict], db_path: str, dry_run: bool) -> int:
    """Update category column in papers.db for tagged papers."""
    con = sqlite3.connect(db_path)
    existing = {}
    for row in con.execute("SELECT id, category FROM papers"):
        existing[row[0]] = row[1]

    updated = 0
    for aid, tag in tags.items():
        if aid not in existing:
            continue
        old_cat = existing[aid]
        new_cat = tag["category"]
        if old_cat == new_cat:
            continue
        if dry_run:
            logger.debug(f"  [DB] {aid}: {old_cat} -> {new_cat}")
        else:
            con.execute("UPDATE papers SET category = ? WHERE id = ?", (new_cat, aid))
        updated += 1

    if not dry_run:
        con.commit()
    con.close()
    return updated


def _update_scored_json(tags: dict[str, dict], dry_run: bool) -> int:
    """Add/update tag_category field in all_scored.json."""
    if not SCORED_PATH.exists():
        return 0

    data = json.loads(SCORED_PATH.read_text())
    updated = 0
    for entry in data:
        aid = entry.get("arxiv_id")
        if aid and aid in tags:
            new_cat = tags[aid]["category"]
            if entry.get("tag_category") != new_cat:
                if not dry_run:
                    entry["tag_category"] = new_cat
                    entry["tag_confidence"] = tags[aid].get("confidence")
                    entry["tag_reason"] = tags[aid].get("reason")
                updated += 1

    if not dry_run and updated:
        SCORED_PATH.write_text(json.dumps(data, indent=2))
    return updated


def _move_doc_files(tags: dict[str, dict], dry_run: bool) -> int:
    """Move docs/ markdown files to new category directories."""
    if not DOCS_DIR.exists():
        return 0

    moved = 0
    for aid, tag in tags.items():
        new_cat = tag["category"]
        for md_file in DOCS_DIR.rglob(f"{aid}*.md"):
            current_cat = md_file.parent.name
            if current_cat == new_cat:
                continue
            new_dir = DOCS_DIR / new_cat
            new_path = new_dir / md_file.name
            if dry_run:
                logger.debug(f"  [docs] {md_file.relative_to(REPO_ROOT)} -> {new_path.relative_to(REPO_ROOT)}")
            else:
                new_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(md_file), str(new_path))
            moved += 1
    return moved


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Sync papers.db — full pipeline, export, or import tags")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arxiv API enrichment")
    parser.add_argument("--db", default=DB_PATH, help="Path to papers.db")

    # Export / import modes
    parser.add_argument("--export", type=Path, metavar="FILE", help="Export papers to JSONL for remote tagging")
    parser.add_argument("--import-tags", type=Path, metavar="FILE", help="Import tagged JSONL results")
    parser.add_argument("--dry-run", action="store_true", help="Preview import changes without applying")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Skip tags below this confidence")

    args = parser.parse_args()
    repo = REPO_ROOT

    # --- Export mode ---
    if args.export:
        _migrate_sync(args.db)
        export_for_tagging(args.export, args.db)
        return

    # --- Import-only mode ---
    if args.import_tags:
        import_tags(args.import_tags, args.db, args.min_confidence, args.dry_run)
        return

    # --- Full sync pipeline ---
    logger.info("Step 1/7: Migrating schema...")
    _migrate_sync(args.db)

    logger.info("Step 2/7: Backfilling stubs from docs/...")
    backfill_from_docs(repo / "docs", args.db)

    logger.info("Step 3/7: Absorbing arxiv_index.txt...")
    absorb_arxiv_index(repo / "arxiv_index.txt", args.db)

    logger.info("Step 4/7: Merging external_papers.db...")
    merge_external_db(repo / "external_papers.db", args.db)

    logger.info("Step 5/7: Absorbing all_scored.json...")
    absorb_scored_json(db_path=args.db)

    if not args.skip_arxiv:
        logger.info("Step 6/7: Enriching from arxiv API...")
        enrich_from_arxiv(args.db)
    else:
        logger.info("Step 6/7: Skipped (--skip-arxiv)")

    # Auto-import tags if file exists
    if DEFAULT_TAGS_PATH.exists():
        logger.info("Step 7/7: Importing tags from tagged_papers.jsonl...")
        import_tags(DEFAULT_TAGS_PATH, args.db, args.min_confidence, args.dry_run)
    else:
        logger.info("Step 7/7: No tagged_papers.jsonl found, skipping tag import")

    # Summary
    con = sqlite3.connect(args.db)
    total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    has_title = con.execute("SELECT COUNT(*) FROM papers WHERE title IS NOT NULL").fetchone()[0]
    has_cats = con.execute("SELECT COUNT(*) FROM papers WHERE categories IS NOT NULL").fetchone()[0]
    has_category = con.execute("SELECT COUNT(*) FROM papers WHERE category IS NOT NULL").fetchone()[0]
    con.close()
    logger.success(
        f"Done! {total} papers in DB ({has_title} with titles, {has_cats} with arxiv metadata, {has_category} categorized)"
    )


if __name__ == "__main__":
    main()
