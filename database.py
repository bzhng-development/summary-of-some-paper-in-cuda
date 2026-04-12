"""Backwards-compatibility shim around :mod:`neon_db`.

The SQLite layer that used to live in this module has been replaced by a Neon
Postgres layer in :mod:`neon_db`. Everything in this file exists only so that
existing call sites — ``from database import save_paper_sync, DB_PATH, ...`` —
continue to work while Agents 2-4 rewrite the downstream scripts to import
from ``neon_db`` directly.

New code should import from :mod:`neon_db`. This shim will be deleted once the
rolling refactor is complete.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from loguru import logger

from neon_db import NeonDB

if TYPE_CHECKING:
    from main import PaperRecord


# ---------------------------------------------------------------------------
# Deprecated ``DB_PATH`` constant
# ---------------------------------------------------------------------------
#
# Several scripts still read ``database.DB_PATH`` for log lines or to pass as
# a CLI default. It no longer means anything — Neon is the source of truth —
# but we keep a sentinel so imports don't explode. Any code that actually
# opens a sqlite3 connection against it will fail loudly at that point, which
# is the correct signal for Agents 2-4 to rewrite that call site.

DB_PATH: str = "<deprecated: use neon_db.NeonDB; Neon is the source of truth>"


# ---------------------------------------------------------------------------
# Lazy singleton
# ---------------------------------------------------------------------------

_db: NeonDB | None = None
_warned = False


def _get_db() -> NeonDB:
    global _db, _warned
    if _db is None:
        _db = NeonDB()
    if not _warned:
        warnings.warn(
            "database.py is a compatibility shim; import from neon_db instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        _warned = True
    return _db


# ---------------------------------------------------------------------------
# Legacy API surface
# ---------------------------------------------------------------------------


def _migrate_sync(db_path: str | None = None) -> None:
    """Legacy name for ``NeonDB.init_schema``. ``db_path`` is ignored."""
    _ = db_path
    _get_db().init_schema()


def save_paper_sync(arxiv_id: str, /, *, db_path: str | None = None, **fields: Any) -> None:
    """Legacy partial-upsert entrypoint. ``db_path`` is ignored."""
    _ = db_path
    _get_db().save_paper(arxiv_id, **fields)


def get_all_ids_sync(db_path: str | None = None) -> set[str]:
    """Legacy name for ``NeonDB.get_all_ids``. ``db_path`` is ignored."""
    _ = db_path
    return _get_db().get_all_ids()


async def init_db() -> None:
    """Legacy async initializer — ensures the Neon schema exists."""
    logger.info("Initializing Neon schema via compatibility shim")
    _get_db().init_schema()


async def save_to_db(record: PaperRecord) -> None:
    """Legacy async save for ``main.PaperRecord``."""
    await _get_db().asave_paper(
        record.arxiv_id,
        title=record.title,
        category=record.category,
        pitch=record.pitch,
        summary=record.summary,
        url=record.url,
        full_response=record.full_response,
    )


async def get_paper(arxiv_id: str) -> dict[str, Any] | None:
    """Legacy async read."""
    import asyncio

    return await asyncio.to_thread(_get_db().get_paper, arxiv_id)


async def paper_exists(arxiv_id: str) -> bool:
    """Legacy async existence check."""
    import asyncio

    return await asyncio.to_thread(_get_db().paper_exists, arxiv_id)


__all__ = [
    "DB_PATH",
    "_migrate_sync",
    "save_paper_sync",
    "get_all_ids_sync",
    "init_db",
    "save_to_db",
    "get_paper",
    "paper_exists",
]
