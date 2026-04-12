#!/usr/bin/env python3
"""Tiny FastAPI server: serves the paper viewer and handles interested-marking.

Routes:
    GET  /              — render the ``paper_viewer`` HTML for ``all_scored.json``
    POST /interested    — mark one or more papers as interested in Neon
    POST /add-paper     — add a paper by arxiv/HF URL or bare arxiv id
    GET  /interested-ids — list ids currently flagged ``interested = 1``

Usage:
    uv run python daily_papers/paper_server.py
    uv run python daily_papers/paper_server.py --port 8787
    uv run python daily_papers/paper_server.py --json ./daily_papers/papers_out/all_scored.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel

# Make the repo root importable when this module is run directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from daily_papers.hf_daily_papers import fetch_arxiv_metadata  # noqa: E402
from daily_papers.paper_viewer import generate_html  # noqa: E402
from neon_db import NeonDB  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------


_db: NeonDB | None = None
_scored_json_path: Path | None = None
_arxiv_index_path: Path | None = None


def _get_db() -> NeonDB:
    """Lazy singleton — constructed on first request to keep import cheap."""
    global _db
    if _db is None:
        _db = NeonDB()
    return _db


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Ensure the Neon schema exists before accepting requests."""
    logger.info("Initializing Neon schema")
    _get_db().init_schema()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MarkInterestedRequest(BaseModel):
    papers: list[dict[str, Any]]


class AddPaperRequest(BaseModel):
    url: str


_ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")
_ARXIV_ID_SEARCH_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _parse_arxiv_id(url: str) -> str | None:
    """Extract an arxiv id from an arxiv URL, HF papers URL, or bare id."""
    url = url.strip()
    if not url:
        return None
    if "huggingface.co/papers/" in url:
        return url.split("/papers/")[-1].split("/")[0].split("?")[0]
    if "arxiv.org" in url:
        match = _ARXIV_ID_SEARCH_RE.search(url)
        return match.group(1) if match else None
    if _ARXIV_ID_RE.match(url):
        return url
    return None


def _interested_ids() -> set[str]:
    return {row["id"] for row in _get_db().get_interested()}


def _append_to_index(arxiv_ids: list[str]) -> None:
    """Best-effort append to ``arxiv_index.txt``; silently skipped if unset."""
    if not arxiv_ids or _arxiv_index_path is None:
        return
    with _arxiv_index_path.open("a") as f:
        for aid in arxiv_ids:
            f.write(f"{aid}\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def serve_viewer() -> str:
    if _scored_json_path is None:
        return "<h1>No scored papers found</h1>"
    try:
        papers = json.loads(_scored_json_path.read_text())
    except FileNotFoundError:
        return "<h1>No scored papers found</h1>"

    interested = _interested_ids()
    for paper in papers:
        paper["_interested"] = paper["arxiv_id"] in interested
    return generate_html(papers)


@app.post("/interested")
def mark_interested(req: MarkInterestedRequest) -> dict[str, Any]:
    db = _get_db()
    marked: list[str] = []
    skipped: list[str] = []

    for p in req.papers:
        arxiv_id = p.get("arxiv_id")
        if not arxiv_id:
            continue

        db.save_paper(
            arxiv_id,
            title=p.get("title"),
            abstract=p.get("summary"),
            url=f"https://arxiv.org/abs/{arxiv_id}",
            authors=p.get("authors"),
            affiliations=p.get("affiliations"),
            categories=p.get("categories"),
            primary_category=p.get("primary_category"),
            arxiv_comment=p.get("arxiv_comment"),
            published=p.get("published"),
            journal_ref=p.get("journal_ref"),
            doi=p.get("doi"),
            upvotes=p.get("upvotes"),
            github=p.get("github"),
            github_stars=p.get("github_stars"),
            organization=p.get("organization"),
            org_fullname=p.get("org_fullname"),
        )
        db.mark_interested(arxiv_id)
        marked.append(arxiv_id)

    _append_to_index(marked)
    logger.info("Marked {} papers as interested", len(marked))
    return {"marked": marked, "skipped": skipped}


@app.post("/add-paper")
def add_paper(req: AddPaperRequest) -> dict[str, Any]:
    arxiv_id = _parse_arxiv_id(req.url)
    if not arxiv_id:
        return {"error": f"Could not parse arxiv ID from: {req.url}"}

    db = _get_db()
    row = db.get_paper(arxiv_id)
    if row is not None and row.get("interested") == 1:
        return {"status": "already_exists", "arxiv_id": arxiv_id}

    meta = fetch_arxiv_metadata([arxiv_id]).get(arxiv_id)
    db.save_paper(
        arxiv_id,
        title=meta.title if meta else None,
        abstract=meta.abstract if meta else None,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        authors=meta.authors if meta else None,
        affiliations=meta.affiliations if meta else None,
        categories=meta.categories if meta else None,
        primary_category=meta.primary_category if meta else None,
        arxiv_comment=meta.comment if meta else None,
        published=meta.published if meta else None,
        journal_ref=meta.journal_ref if meta else None,
        doi=meta.doi if meta else None,
    )
    db.mark_interested(arxiv_id)
    _append_to_index([arxiv_id])

    logger.info("Added paper {} to DB (from URL: {})", arxiv_id, req.url)
    return {"status": "added", "arxiv_id": arxiv_id, "has_meta": meta is not None}


@app.get("/interested-ids")
def get_interested_ids() -> dict[str, list[str]]:
    return {"ids": list(_interested_ids())}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Paper viewer server")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("daily_papers/papers_out/all_scored.json"),
    )
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--index", type=Path, default=Path("arxiv_index.txt"))
    args = parser.parse_args(argv)

    global _scored_json_path, _arxiv_index_path
    _scored_json_path = args.json.resolve()
    _arxiv_index_path = args.index.resolve()

    logger.info("Serving viewer at http://localhost:{}", args.port)
    logger.info("JSON: {}", _scored_json_path)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
