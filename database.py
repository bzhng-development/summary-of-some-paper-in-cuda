"""SQLite database module for storing paper summaries and API responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite
from loguru import logger

if TYPE_CHECKING:
    from main import PaperRecord

DB_PATH = "papers.db"


async def init_db() -> None:
    """Initialize the database and create the papers table if it doesn't exist."""
    logger.info(f"Initializing database at {DB_PATH}")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
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
        await db.commit()
    logger.debug("Database initialized successfully")


async def save_to_db(record: PaperRecord) -> None:
    """Save paper data to the database.

    Args:
        record: PaperRecord containing all paper data

    """
    logger.debug(f"Saving paper {record.arxiv_id} to database")
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO papers (id, title, category, pitch, summary, url, full_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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
    """Retrieve a paper from the database by ID.

    Args:
        arxiv_id: ArXiv paper ID

    Returns:
        Paper data as a dictionary, or None if not found

    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM papers WHERE id = ?", (arxiv_id,)) as cursor:
            row = await cursor.fetchone()
            if row:
                return dict(row)
    return None


async def paper_exists(arxiv_id: str) -> bool:
    """Check if a paper exists in the database.

    Args:
        arxiv_id: ArXiv paper ID

    Returns:
        True if paper exists, False otherwise

    """
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT 1 FROM papers WHERE id = ?", (arxiv_id,)) as cursor:
            return await cursor.fetchone() is not None
