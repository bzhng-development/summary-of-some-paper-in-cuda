"""Neon Postgres layer for paper storage.

This module is the replacement for the former SQLite ``database.py`` layer. It
provides a single :class:`NeonDB` façade over a Neon (managed Postgres)
instance, plus a migration helper for pulling data out of the legacy
``local_data/papers.db`` SQLite file.

Design notes
------------
- Each method opens a short-lived autocommit ``psycopg.Connection`` via
  :meth:`NeonDB.get_conn`. Neon serverless connections are cheap and this
  avoids connection-pool lifecycle bugs in long-running scripts.
- ``save_paper`` is the hot path across the repo. It performs a partial upsert:
  only provided, non-``None`` fields update the row, via
  ``COALESCE(EXCLUDED.col, {table}.col)``. This matches the semantics of the
  previous ``save_paper_sync`` helper.
- The public ``database.py`` module is now a thin backwards-compat shim around
  this module so existing call sites keep working during the rolling refactor.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import psycopg
from dotenv import load_dotenv
from loguru import logger
from psycopg.rows import dict_row

__all__ = [
    "NeonDB",
    "PaperRow",
    "TABLE",
    "SCHEMA_COLUMNS",
    "migrate_sqlite_to_neon",
]


# ---------------------------------------------------------------------------
# .env discovery
# ---------------------------------------------------------------------------
#
# Follow the NeonDBResource pattern: walk up the repo to find the shared
# nextjs-ui/.env that holds DATABASE_URL. Also honour a repo-local .env as an
# override so you can point at a branch DB without touching the shared file.


def _load_env() -> None:
    here = Path(__file__).resolve().parent

    local_env = here / ".env"
    if local_env.exists():
        load_dotenv(local_env)

    # `<...>/open_source/summary-of-some-paper-in-cuda/neon_db.py` →
    # `<...>/open_source/company-scraper/nextjs-ui/.env`
    for parent in here.parents:
        candidate = parent / "company-scraper" / "nextjs-ui" / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_load_env()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


TABLE: Final[str] = '"nextjs-ui_paper"'

# Ordered, authoritative column list. Derived from the live
# ``local_data/papers.db`` SQLite schema (13 448 rows as of the refactor).
# Agents 2-4: if you add a column, add it here AND in ``init_schema`` below.
SCHEMA_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "title",
    "category",
    "pitch",
    "summary",
    "url",
    "full_response",
    "created_at",
    "authors",            # JSON array of author names
    "affiliations",       # JSON dict {name: [affs]}
    "categories",         # JSON array ["cs.CL", "cs.AI"]
    "primary_category",
    "arxiv_comment",
    "published",          # ISO date string
    "journal_ref",
    "doi",
    "upvotes",
    "github",
    "github_stars",
    "organization",
    "org_fullname",
    "abstract",
    "interested",
    "legacy_gpt_summary",
    "score",
    "similar_paper",
    "score_reason",
    "tag_category_v2",
    "tag_confidence",
    "tag_reason",
    "score_source",
)

# Columns the caller is allowed to pass to ``save_paper`` as kwargs. ``id`` is
# the positional ``arxiv_id`` and ``created_at`` is DB-managed.
_WRITABLE_COLUMNS: Final[frozenset[str]] = frozenset(
    c for c in SCHEMA_COLUMNS if c not in ("id", "created_at")
)

# Fields that should be JSON-encoded on the way in if the caller passed a list
# or dict. Everything else is passed through verbatim.
_JSON_COLUMNS: Final[frozenset[str]] = frozenset({"authors", "affiliations", "categories"})


@dataclass(frozen=True, kw_only=True, slots=True)
class PaperRow:
    """Typed view of a papers row. Returned by :meth:`NeonDB.get_paper`.

    This is intentionally *not* the write schema — writes go through
    ``save_paper(**fields)`` so partial updates stay ergonomic.
    """

    id: str
    title: str | None
    category: str | None
    pitch: str | None
    summary: str | None
    url: str | None
    full_response: str | None
    abstract: str | None
    interested: int
    score: int | None
    score_source: str | None


# ---------------------------------------------------------------------------
# NeonDB
# ---------------------------------------------------------------------------


class NeonDB:
    """Plain-Python Neon Postgres client for the papers table.

    Construct once per process (or use the module-level shim singleton in
    ``database.py``). Each method opens and closes its own connection.
    """

    def __init__(self, database_url: str | None = None) -> None:
        url = database_url if database_url is not None else os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError(
                "DATABASE_URL is not set. Put it in the repo's .env, or in "
                "../company-scraper/nextjs-ui/.env, or export it in the shell."
            )
        self._database_url = url

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def get_conn(self) -> psycopg.Connection:
        """Open an autocommit connection. Caller must use ``with``."""
        return psycopg.connect(self._database_url, autocommit=True)

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create the papers table and indexes if they do not already exist."""
        logger.info("Ensuring Neon schema for {}", TABLE)
        ddl = f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id                TEXT PRIMARY KEY,
                title             TEXT,
                category          TEXT,
                pitch             TEXT,
                summary           TEXT,
                url               TEXT,
                full_response     TEXT,
                created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                authors           TEXT,
                affiliations      TEXT,
                categories        TEXT,
                primary_category  TEXT,
                arxiv_comment     TEXT,
                published         TEXT,
                journal_ref       TEXT,
                doi               TEXT,
                upvotes           INTEGER,
                github            TEXT,
                github_stars      INTEGER,
                organization      TEXT,
                org_fullname      TEXT,
                abstract          TEXT,
                interested        INTEGER NOT NULL DEFAULT 0,
                legacy_gpt_summary TEXT,
                score             INTEGER,
                similar_paper     TEXT,
                score_reason      TEXT,
                tag_category_v2   TEXT,
                tag_confidence    DOUBLE PRECISION,
                tag_reason        TEXT,
                score_source      TEXT
            )
        """
        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            cur.execute(f'CREATE INDEX IF NOT EXISTS "nextjs-ui_paper_interested_idx" ON {TABLE} (interested)')
            cur.execute(f'CREATE INDEX IF NOT EXISTS "nextjs-ui_paper_score_idx" ON {TABLE} (score)')
        logger.debug("Neon schema ready")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fields(fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in fields.items():
            if value is None:
                continue
            if key in _JSON_COLUMNS and not isinstance(value, str):
                out[key] = json.dumps(value, ensure_ascii=False)
            else:
                out[key] = value
        return out

    def save_paper(self, arxiv_id: str, /, **fields: Any) -> None:
        """Partial upsert for a single paper row.

        Only non-``None`` kwargs are applied. Existing column values are
        preserved via ``COALESCE(EXCLUDED.col, {TABLE}.col)``. Unknown kwargs
        raise ``TypeError`` so typos fail loudly.
        """
        unknown = set(fields) - _WRITABLE_COLUMNS
        if unknown:
            raise TypeError(
                f"save_paper() got unknown columns: {sorted(unknown)}. "
                f"Allowed: {sorted(_WRITABLE_COLUMNS)}"
            )

        clean = self._normalize_fields(fields)
        payload: dict[str, Any] = {"id": arxiv_id, **clean}

        columns = list(payload.keys())
        placeholders = ", ".join(f"%({col})s" for col in columns)
        column_list = ", ".join(f'"{col}"' for col in columns)

        update_parts = [
            f'"{col}" = COALESCE(EXCLUDED."{col}", {TABLE}."{col}")'
            for col in columns
            if col != "id"
        ]

        if update_parts:
            sql = (
                f"INSERT INTO {TABLE} ({column_list}) VALUES ({placeholders}) "
                f"ON CONFLICT (id) DO UPDATE SET {', '.join(update_parts)}"
            )
        else:
            # Only the id was provided — insert-if-absent, else no-op.
            sql = (
                f"INSERT INTO {TABLE} ({column_list}) VALUES ({placeholders}) "
                f"ON CONFLICT (id) DO NOTHING"
            )

        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, payload)

    async def asave_paper(self, arxiv_id: str, /, **fields: Any) -> None:
        """Async mirror of :meth:`save_paper`.

        Implemented via ``asyncio.to_thread`` rather than psycopg's async
        client. The write volume across the codebase is modest (single-row
        upserts inside per-paper async pipelines) and this keeps the surface
        area small while staying non-blocking on the event loop.
        """
        await asyncio.to_thread(self.save_paper, arxiv_id, **fields)

    def mark_interested(self, arxiv_id: str) -> None:
        """Flip ``interested = 1`` on a single paper."""
        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE} SET interested = 1 WHERE id = %s",
                (arxiv_id,),
            )

    def update_category(self, arxiv_id: str, category: str) -> None:
        """Direct category overwrite (used by sync_db tag imports)."""
        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TABLE} SET category = %s WHERE id = %s",
                (category, arxiv_id),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """Fetch a single paper row as a dict, or ``None`` if missing."""
        with self.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT * FROM {TABLE} WHERE id = %s", (arxiv_id,))
            return cur.fetchone()

    def paper_exists(self, arxiv_id: str) -> bool:
        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {TABLE} WHERE id = %s", (arxiv_id,))
            return cur.fetchone() is not None

    def get_all_ids(self) -> set[str]:
        with self.get_conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT id FROM {TABLE}")
            return {row[0] for row in cur.fetchall()}

    def get_interested(self) -> list[dict[str, Any]]:
        """Rows marked ``interested = 1`` (used by load_examples + paper_server)."""
        with self.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT * FROM {TABLE} WHERE interested = 1")
            return list(cur.fetchall())

    def get_stubs_without_summary(self, *, interested_only: bool = False) -> list[dict[str, Any]]:
        """Rows with no real summary. Used by ``multi_prompt --backfill``/``--dyn``.

        Returns ``id``, ``title``, ``url``. When ``interested_only`` is True,
        mirrors the legacy ``get_interested_stubs_sync`` filter.
        """
        where = "(summary IS NULL OR summary = '')"
        if interested_only:
            where += " AND interested = 1"
        else:
            where += " AND id NOT LIKE 'ext:%'"
        with self.get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT id, title, url FROM {TABLE} WHERE {where}")
            return list(cur.fetchall())


# ---------------------------------------------------------------------------
# SQLite → Neon migration helper
# ---------------------------------------------------------------------------


def migrate_sqlite_to_neon(sqlite_path: str | Path, *, batch_size: int = 500) -> int:
    """Stream every row from the legacy SQLite DB into Neon.

    Uses :meth:`NeonDB.save_paper` so the column set is validated and JSON
    fields are preserved verbatim (they were already stored as TEXT in SQLite).
    Returns the number of rows copied.
    """
    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite source not found: {path}")

    db = NeonDB()
    db.init_schema()

    logger.info("Migrating {} → Neon {}", path, TABLE)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        total = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        logger.info("Source has {} rows; batch_size={}", total, batch_size)

        cursor = con.execute("SELECT * FROM papers")
        copied = 0
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            for row in batch:
                data = dict(row)
                arxiv_id = data.pop("id")
                data.pop("created_at", None)
                # Strip Nones so COALESCE-upsert preserves any existing values.
                data = {k: v for k, v in data.items() if v is not None}
                db.save_paper(arxiv_id, **data)
                copied += 1
            logger.info("Migrated {}/{}", copied, total)

        logger.info("Migration complete: {} rows", copied)
        return copied
    finally:
        con.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="neon_db", description="Neon papers layer")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-schema", help="Create the papers table if absent")

    p_mig = sub.add_parser("migrate", help="Copy rows from a local SQLite DB")
    p_mig.add_argument("--sqlite", required=True, help="Path to papers.db")
    p_mig.add_argument("--batch-size", type=int, default=500)

    args = parser.parse_args(argv)

    if args.command == "init-schema":
        NeonDB().init_schema()
        logger.info("Schema ensured")
        return 0

    if args.command == "migrate":
        migrate_sqlite_to_neon(args.sqlite, batch_size=args.batch_size)
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
