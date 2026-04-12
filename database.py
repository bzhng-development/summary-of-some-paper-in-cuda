"""SQLite database module for storing paper summaries and API responses."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
from loguru import logger

if TYPE_CHECKING:
    from main import PaperRecord

# DB lives in local_data/ (gitignored — too large for github, rebuild via sync_db.py).
# Resolved relative to this file so scripts work from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = os.environ.get("PAPERS_DB_PATH", str(_REPO_ROOT / "local_data" / "papers.db"))

# All columns that should exist on the papers table.
# Migration adds any missing ones via ALTER TABLE.
_EXTRA_COLUMNS = {
    "authors": "TEXT",  # JSON array of author names
    "affiliations": "TEXT",  # JSON dict {name: [affs]}
    "categories": "TEXT",  # JSON array ["cs.CL", "cs.AI"]
    "primary_category": "TEXT",
    "arxiv_comment": "TEXT",
    "published": "TEXT",  # ISO date
    "journal_ref": "TEXT",
    "doi": "TEXT",
    "upvotes": "INTEGER",
    "github": "TEXT",
    "github_stars": "INTEGER",
    "organization": "TEXT",
    "org_fullname": "TEXT",
    "abstract": "TEXT",  # arxiv abstract (distinct from pipeline-generated summary)
    "interested": "INTEGER DEFAULT 0",  # 1 = user marked as interested
}


def _migrate_sync(db_path: str = DB_PATH) -> None:
    """Add any missing columns to an existing papers table (idempotent)."""
    con = sqlite3.connect(db_path)
    # Ensure base table exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            pitch TEXT,
            summary TEXT,
            url TEXT,
            full_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    existing = {row[1] for row in con.execute("PRAGMA table_info(papers)").fetchall()}
    added = []
    for col, typ in _EXTRA_COLUMNS.items():
        if col not in existing:
            con.execute(f"ALTER TABLE papers ADD COLUMN {col} {typ}")
            added.append(col)
    con.commit()
    con.close()
    if added:
        logger.info(f"Migrated papers table: added {added}")


async def init_db() -> None:
    """Initialize the database and create/migrate the papers table."""
    logger.info(f"Initializing database at {DB_PATH}")
    _migrate_sync(DB_PATH)
    logger.debug("Database initialized successfully")


async def save_to_db(record: PaperRecord) -> None:
    """Save paper data to the database.

    Uses ON CONFLICT to preserve any enriched metadata (arxiv API fields)
    that may already exist on the row.
    """
    logger.debug(f"Saving paper {record.arxiv_id} to database")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO papers (id, title, category, pitch, summary, url, full_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                category = excluded.category,
                pitch = excluded.pitch,
                summary = excluded.summary,
                url = excluded.url,
                full_response = excluded.full_response
            """,
            (
                record.arxiv_id,
                record.title,
                record.category,
                record.pitch,
                record.summary,
                record.url,
                record.full_response,
            ),
        )
        await db.commit()
    logger.info(f"Paper {record.arxiv_id} saved to database")


async def get_paper(arxiv_id: str) -> dict | None:
    """Retrieve a paper from the database by ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM papers WHERE id = ?", (arxiv_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
    return None


async def paper_exists(arxiv_id: str) -> bool:
    """Check if a paper exists in the database."""
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM papers WHERE id = ?", (arxiv_id,)) as cursor:
            return await cursor.fetchone() is not None


# ============================================================================
# Sync helpers (for scripts that don't need async)
# ============================================================================


def save_paper_sync(
    arxiv_id: str,
    *,
    title: str | None = None,
    category: str | None = None,
    pitch: str | None = None,
    summary: str | None = None,
    url: str | None = None,
    authors: list[str] | None = None,
    affiliations: dict | None = None,
    categories: list[str] | None = None,
    primary_category: str | None = None,
    arxiv_comment: str | None = None,
    published: str | None = None,
    journal_ref: str | None = None,
    doi: str | None = None,
    upvotes: int | None = None,
    github: str | None = None,
    github_stars: int | None = None,
    organization: str | None = None,
    org_fullname: str | None = None,
    abstract: str | None = None,
    db_path: str = DB_PATH,
) -> None:
    """Insert or update a paper with any subset of fields (sync)."""
    con = sqlite3.connect(db_path)
    # Build SET clause for only provided fields
    fields = {"id": arxiv_id}
    if title is not None:
        fields["title"] = title
    if category is not None:
        fields["category"] = category
    if pitch is not None:
        fields["pitch"] = pitch
    if summary is not None:
        fields["summary"] = summary
    if url is not None:
        fields["url"] = url
    if authors is not None:
        fields["authors"] = json.dumps(authors)
    if affiliations is not None:
        fields["affiliations"] = json.dumps(affiliations)
    if categories is not None:
        fields["categories"] = json.dumps(categories)
    if primary_category is not None:
        fields["primary_category"] = primary_category
    if arxiv_comment is not None:
        fields["arxiv_comment"] = arxiv_comment
    if published is not None:
        fields["published"] = published
    if journal_ref is not None:
        fields["journal_ref"] = journal_ref
    if doi is not None:
        fields["doi"] = doi
    if upvotes is not None:
        fields["upvotes"] = upvotes
    if github is not None:
        fields["github"] = github
    if github_stars is not None:
        fields["github_stars"] = github_stars
    if organization is not None:
        fields["organization"] = organization
    if org_fullname is not None:
        fields["org_fullname"] = org_fullname
    if abstract is not None:
        fields["abstract"] = abstract

    cols = ", ".join(fields.keys())
    placeholders = ", ".join("?" for _ in fields)
    # ON CONFLICT: update all non-id fields, but only if the new value is not NULL
    updates = ", ".join(
        f"{k} = COALESCE(excluded.{k}, {k})" for k in fields if k != "id"
    )
    sql = f"INSERT INTO papers ({cols}) VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}"
    con.execute(sql, list(fields.values()))
    con.commit()
    con.close()


def get_all_ids_sync(db_path: str = DB_PATH) -> set[str]:
    """Get all paper IDs in the database."""
    con = sqlite3.connect(db_path)
    ids = {row[0] for row in con.execute("SELECT id FROM papers").fetchall()}
    con.close()
    return ids
