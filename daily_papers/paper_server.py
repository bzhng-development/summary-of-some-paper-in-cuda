#!/usr/bin/env python3
"""Tiny server: serves the paper viewer and handles POST /interested to mark papers.

Usage:
    uv run python daily_papers/paper_server.py
    uv run python daily_papers/paper_server.py --port 8787
    uv run python daily_papers/paper_server.py --json ./daily_papers/papers_out/all_scored.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger
from pydantic import BaseModel

# Add repo root to path so we can import database
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database import DB_PATH, _migrate_sync, save_paper_sync
from daily_papers.paper_viewer import generate_html
from daily_papers.hf_daily_papers import fetch_arxiv_metadata

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Will be set by CLI args
_scored_json_path: Path | None = None
_arxiv_index_path: Path | None = None


def _get_interested_ids() -> set[str]:
    """Get IDs of papers marked as interested."""
    con = sqlite3.connect(DB_PATH)
    ids = {row[0] for row in con.execute("SELECT id FROM papers WHERE interested = 1").fetchall()}
    con.close()
    return ids


class MarkInterestedRequest(BaseModel):
    papers: list[dict]


@app.get("/", response_class=HTMLResponse)
def serve_viewer():
    if _scored_json_path and _scored_json_path.exists():
        papers = json.loads(_scored_json_path.read_text())
        interested_ids = _get_interested_ids()
        for p in papers:
            p["_interested"] = p["arxiv_id"] in interested_ids
        return generate_html(papers)
    return "<h1>No scored papers found</h1>"


@app.post("/interested")
def mark_interested(req: MarkInterestedRequest):
    _migrate_sync()
    marked = []
    skipped = []

    for p in req.papers:
        arxiv_id = p.get("arxiv_id")
        if not arxiv_id:
            continue

        # Ensure paper exists in DB with metadata, then mark interested
        save_paper_sync(
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

        # Set interested flag
        con = sqlite3.connect(DB_PATH)
        con.execute("UPDATE papers SET interested = 1 WHERE id = ?", (arxiv_id,))
        con.commit()
        con.close()
        marked.append(arxiv_id)

    # Also append to arxiv_index.txt
    if marked and _arxiv_index_path:
        with open(_arxiv_index_path, "a") as f:
            for aid in marked:
                f.write(f"{aid}\n")

    logger.info(f"Marked {len(marked)} papers as interested")
    return {"marked": marked, "skipped": skipped}


class AddPaperRequest(BaseModel):
    url: str


def _parse_arxiv_id(url: str) -> str | None:
    """Extract arxiv ID from an arxiv URL, HF daily papers URL, or bare ID."""
    url = url.strip()
    if not url:
        return None
    if "huggingface.co/papers/" in url:
        return url.split("/papers/")[-1].split("/")[0].split("?")[0]
    if "arxiv.org" in url:
        m = re.search(r"(\d{4}\.\d{4,5})", url)
        return m.group(1) if m else None
    if re.match(r"^\d{4}\.\d{4,5}$", url):
        return url
    return None


@app.post("/add-paper")
def add_paper(req: AddPaperRequest):
    arxiv_id = _parse_arxiv_id(req.url)
    if not arxiv_id:
        return {"error": f"Could not parse arxiv ID from: {req.url}"}

    # Check if already in DB with interested flag
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT interested FROM papers WHERE id = ?", (arxiv_id,)).fetchone()
    con.close()
    if row and row[0] == 1:
        return {"status": "already_exists", "arxiv_id": arxiv_id}

    # Fetch full metadata from arxiv API
    meta_map = fetch_arxiv_metadata([arxiv_id])
    meta = meta_map.get(arxiv_id)

    _migrate_sync()
    save_paper_sync(
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
    con = sqlite3.connect(DB_PATH)
    con.execute("UPDATE papers SET interested = 1 WHERE id = ?", (arxiv_id,))
    con.commit()
    con.close()

    if _arxiv_index_path:
        with open(_arxiv_index_path, "a") as f:
            f.write(f"{arxiv_id}\n")

    logger.info(f"Added paper {arxiv_id} to DB (from URL: {req.url})")
    return {"status": "added", "arxiv_id": arxiv_id, "has_meta": meta is not None}


@app.get("/interested-ids")
def get_interested_ids():
    return {"ids": list(_get_interested_ids())}


def main():
    parser = argparse.ArgumentParser(description="Paper viewer server")
    parser.add_argument("--json", type=Path, default=Path("daily_papers/papers_out/all_scored.json"))
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--index", type=Path, default=Path("arxiv_index.txt"))
    args = parser.parse_args()

    global _scored_json_path, _arxiv_index_path
    _scored_json_path = args.json.resolve()
    _arxiv_index_path = args.index.resolve()

    _migrate_sync()
    logger.info(f"Serving viewer at http://localhost:{args.port}")
    logger.info(f"JSON: {_scored_json_path}")
    logger.info(f"DB: {DB_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
