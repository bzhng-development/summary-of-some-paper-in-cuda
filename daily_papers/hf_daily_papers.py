#!/usr/bin/env python3
"""
hf_daily_papers.py — Fetch Hugging Face Daily Papers and filter for relevance
using a local LLM.

Usage:
    python hf_daily_papers.py --out-dir ./papers_out --workers 4
    python hf_daily_papers.py --out-dir ./papers_out --threshold 7

    # Fetch a date range (Jan 1 to today):
    python hf_daily_papers.py --out-dir ./papers_out --from 2026-01-01 --to 2026-03-03
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Ensure the repo root is importable when running this file directly from
# ``daily_papers/``. This mirrors paper_server.py's sys.path tweak.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from daily_papers.examples import (  # noqa: E402  (after sys.path tweak)
    build_examples_block,
    load_examples,
)
from neon_db import NeonDB  # noqa: E402


# ============================================================================
# Config
# ============================================================================

# Override any of these via environment variables.
#
# Priority for BASE_URL:
#   1. SGLANG_BASE_URL    — full URL override (back-compat)
#   2. LOCAL_LLM_PORT     — just the port; also consumed by multi_prompt_pkg
#   3. default localhost:30000
#
# Setting LOCAL_LLM_PORT once configures both the scorer and multi_prompt's
# summarizer, which is the common case. Use SGLANG_BASE_URL when you need to
# point at a non-localhost endpoint (Modal, remote SSH, etc.).
_LOCAL_LLM_PORT = int(os.environ.get("LOCAL_LLM_PORT", "30000"))
BASE_URL: str = os.environ.get("SGLANG_BASE_URL", f"http://localhost:{_LOCAL_LLM_PORT}/v1")
MODEL: str = os.environ.get("SGLANG_MODEL", "Qwen/Qwen3.5-122B-A10B")
HF_PAPERS_API: str = "https://huggingface.co/api/daily_papers"

# Path to the summary-of-some-paper-in-cuda repo for loading examples.
PAPERS_REPO = _REPO_ROOT


# ============================================================================
# Data
# ============================================================================


@dataclass
class Paper:
    arxiv_id: str
    title: str
    summary: str
    upvotes: int
    authors: list[str]
    affiliations: dict[str, list[str]]  # author name -> list of affiliations from arxiv
    github_repo: str | None
    github_stars: int | None
    ai_keywords: list[str]
    organization: str | None  # HF org short name
    org_fullname: str | None  # HF org display name
    # arxiv metadata (enriched via arxiv API)
    categories: list[str]  # e.g. ["cs.CL", "cs.AI"]
    primary_category: str | None  # e.g. "cs.CL"
    arxiv_comment: str | None  # e.g. "Accepted at NeurIPS 2025. 23 pages, 8 figures"
    published: str | None  # ISO date of first arxiv submission
    journal_ref: str | None  # e.g. "NeurIPS 2025"
    doi: str | None  # resolved DOI URL


@dataclass
class ScoredPaper:
    paper: Paper
    score: int  # 1-10
    similar_paper: str  # title from reading history, or "NONE"
    reason: str


# ``ExamplePaper`` / ``load_examples`` / ``build_examples_block`` now live in
# ``daily_papers.examples`` and are shared with ``papers_by_score.py``.


# ============================================================================
# Fetch
# ============================================================================


def _parse_entries(raw: list[dict]) -> list[Paper]:
    """Parse API response entries into Paper objects."""
    papers = []
    for entry in raw:
        p = entry["paper"]
        org = p.get("organization") or entry.get("organization")
        papers.append(
            Paper(
                arxiv_id=p["id"],
                title=p["title"],
                summary=p.get("summary", ""),
                upvotes=p.get("upvotes", 0),
                authors=[a.get("name", "") for a in p.get("authors", [])],
                affiliations={},  # populated later by fetch_arxiv_metadata
                github_repo=p.get("githubRepo"),
                github_stars=p.get("githubStars"),
                ai_keywords=p.get("ai_keywords", []),
                organization=org.get("name") if isinstance(org, dict) else None,
                org_fullname=org.get("fullname") if isinstance(org, dict) else None,
                categories=[],
                primary_category=None,
                arxiv_comment=None,
                published=None,
                journal_ref=None,
                doi=None,
            )
        )
    return papers


ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


@dataclass
class ArxivMeta:
    """All metadata scraped from the arxiv API for a single paper."""
    title: str
    abstract: str
    authors: list[str]
    affiliations: dict[str, list[str]]  # author name -> affiliations
    categories: list[str]
    primary_category: str | None
    comment: str | None
    published: str | None
    journal_ref: str | None
    doi: str | None


def _parse_arxiv_entry(entry: ET.Element) -> tuple[str, ArxivMeta] | None:
    """Parse a single <entry> from the arxiv API response."""
    id_el = entry.find("atom:id", ARXIV_NS)
    if id_el is None or id_el.text is None:
        return None
    raw_id = id_el.text.rsplit("/", 1)[-1]
    arxiv_id = re.sub(r"v\d+$", "", raw_id)

    title_el = entry.find("atom:title", ARXIV_NS)
    if title_el is not None and title_el.text and title_el.text.strip() == "Error":
        return None
    title = " ".join(title_el.text.split()) if title_el is not None and title_el.text else ""

    summary_el = entry.find("atom:summary", ARXIV_NS)
    abstract = " ".join(summary_el.text.split()) if summary_el is not None and summary_el.text else ""

    authors: list[str] = []
    affiliations: dict[str, list[str]] = {}
    for author_el in entry.findall("atom:author", ARXIV_NS):
        name_el = author_el.find("atom:name", ARXIV_NS)
        if name_el is None or not name_el.text:
            continue
        name = name_el.text.strip()
        authors.append(name)
        affs = [a.text.strip() for a in author_el.findall("arxiv:affiliation", ARXIV_NS) if a.text]
        if affs:
            affiliations[name] = affs

    categories = [cat.get("term", "") for cat in entry.findall("atom:category", ARXIV_NS) if cat.get("term")]
    primary_el = entry.find("arxiv:primary_category", ARXIV_NS)
    primary_category = primary_el.get("term") if primary_el is not None else None
    comment_el = entry.find("arxiv:comment", ARXIV_NS)
    comment = comment_el.text.strip() if comment_el is not None and comment_el.text else None
    pub_el = entry.find("atom:published", ARXIV_NS)
    published = pub_el.text.strip() if pub_el is not None and pub_el.text else None
    jref_el = entry.find("arxiv:journal_ref", ARXIV_NS)
    journal_ref = jref_el.text.strip() if jref_el is not None and jref_el.text else None
    doi_el = entry.find("arxiv:doi", ARXIV_NS)
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

    return arxiv_id, ArxivMeta(
        title=title, abstract=abstract, authors=authors, affiliations=affiliations,
        categories=categories, primary_category=primary_category, comment=comment,
        published=published, journal_ref=journal_ref, doi=doi,
    )


def fetch_arxiv_metadata(arxiv_ids: list[str]) -> dict[str, ArxivMeta]:
    """Batch-fetch full metadata from the arxiv API.

    Returns {arxiv_id: ArxivMeta}.
    Chunks into batches of 100 with a 1s delay between batches.
    """
    result: dict[str, ArxivMeta] = {}
    batch_size = 100

    with httpx.Client(limits=httpx.Limits(max_connections=5, max_keepalive_connections=2)) as client:
        for i in range(0, len(arxiv_ids), batch_size):
            batch = arxiv_ids[i : i + batch_size]
            if i > 0:
                time.sleep(1)

            id_list = ",".join(batch)
            resp = None
            for attempt in range(7):
                try:
                    resp = client.get(
                        ARXIV_API,
                        params={"id_list": id_list, "max_results": len(batch)},
                        timeout=30,
                        follow_redirects=True,
                    )
                    resp.raise_for_status()
                    break
                except Exception as e:
                    is_429 = "429" in str(e)
                    wait = min(3 ** attempt * (3 if is_429 else 1), 120)
                    logger.warning(f"arxiv API batch {i // batch_size} attempt {attempt + 1} failed: {e}, retrying in {wait}s")
                    time.sleep(wait)
            if resp is None or resp.status_code != 200:
                logger.error(f"arxiv API batch {i // batch_size} failed after 7 attempts, skipping")
                continue

            root = ET.fromstring(resp.text)
            for entry in root.findall("atom:entry", ARXIV_NS):
                parsed = _parse_arxiv_entry(entry)
                if parsed:
                    result[parsed[0]] = parsed[1]

            logger.debug(f"arxiv metadata batch {i // batch_size}: got {len(result)} papers so far")

    logger.info(f"Fetched arxiv metadata for {len(result)}/{len(arxiv_ids)} papers")
    return result


def _enrich_with_arxiv(papers: list[Paper]) -> None:
    """Fetch and attach arxiv metadata to papers in-place."""
    ids = [p.arxiv_id for p in papers]
    if not ids:
        return
    meta_map = fetch_arxiv_metadata(ids)
    for p in papers:
        meta = meta_map.get(p.arxiv_id)
        if not meta:
            continue
        p.affiliations = meta.affiliations
        p.categories = meta.categories
        p.primary_category = meta.primary_category
        p.arxiv_comment = meta.comment
        p.published = meta.published
        p.journal_ref = meta.journal_ref
        p.doi = meta.doi


def _fetch_papers_raw(day: date | None = None, client: httpx.Client | None = None) -> list[Paper]:
    """Fetch papers for a specific date (no arxiv enrichment)."""
    params: dict[str, str] = {}
    if day:
        params["date"] = day.isoformat()
    get = client.get if client else httpx.get
    resp = get(HF_PAPERS_API, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_entries(resp.json())


def fetch_papers(day: date | None = None) -> list[Paper]:
    """Fetch papers for a specific date (or today if None)."""
    papers = _fetch_papers_raw(day)
    _enrich_with_arxiv(papers)
    logger.info(f"Fetched {len(papers)} papers for {day or 'today'}")
    return papers


def fetch_papers_range(start: date, end: date, enrich_batch: int = 1000) -> list[Paper]:
    """Fetch papers for every day in [start, end], deduped by arxiv_id.

    Fetches all days concurrently from HF API, then kicks off arxiv
    enrichment in background batches of `enrich_batch` while returning
    papers immediately for scoring.
    """
    days = []
    day = start
    while day <= end:
        days.append(day)
        day += timedelta(days=1)

    logger.info(f"Fetching {len(days)} days from HF API concurrently...")

    # Concurrent HF API fetches with shared client (connection pooling)
    day_results: dict[date, list[Paper]] = {}
    with httpx.Client(limits=httpx.Limits(max_connections=32, max_keepalive_connections=10)) as hf_client:
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = {pool.submit(_fetch_papers_raw, d, hf_client): d for d in days}
            for fut in as_completed(futures):
                d = futures[fut]
                try:
                    day_results[d] = fut.result()
                except Exception as e:
                    logger.warning(f"Failed to fetch {d}: {e}")

    # Dedupe across days
    seen: set[str] = set()
    all_papers: list[Paper] = []
    for d in sorted(day_results.keys()):
        for p in day_results[d]:
            if p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                all_papers.append(p)

    logger.info(f"Fetched {len(all_papers)} unique papers from {len(day_results)} days, enriching arxiv in background...")

    # XXX: arxiv enrichment disabled for now. categories, affiliations,
    # arxiv_comment, journal_ref, doi, published will be missing.
    # we shall get it later. --vz

    return all_papers


# ============================================================================
# LLM scoring
# ============================================================================


class ScoreOutput(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Relevance score from 1-10")
    similar_paper: str = Field(
        ...,
        description="Title of the most similar paper from the reading history. "
        "Must be an exact title from the list. If no paper is genuinely similar, write 'NONE'.",
    )
    reason: str = Field(..., description="1-2 sentence justification for the score and similarity")


SYSTEM_PROMPT = """\
You are a strict research paper relevance filter.

You will receive a candidate paper and the user's reading history.
Determine if the candidate paper is **solving the same problem** as at least
one paper the user has already read.

"Same problem" means:
- Same core problem (e.g. both try to make attention faster, both try to compress KV cache)
- Direct follow-up or competing approach to a specific paper
- A new solution to a problem an existing paper already addresses

"Same problem" does NOT mean:
- Both use transformers (too broad)
- Both involve "reasoning" or "agents" (too vague)
- Both are about ML/AI (everything is)
- Shared minor detail (e.g. both happen to use RL, both happen to mention quantization)

SCORING:
- 8-10: The candidate is solving the same specific problem as a paper in the history.
        You can explain in one sentence exactly what shared problem they address.
- 5-7: Related subfield but not the same specific problem.
- 1-4: Different problem. No real connection beyond "both are ML papers."

AUTHOR QUALITY PENALTY:
The user only cares about papers from established researchers and labs.
Apply a -2 penalty (cap at 1) if ALL of the following are true:
- No recognizable institutional affiliation (top university, major lab like Google, Meta, DeepMind, OpenAI, NVIDIA, etc.)
- No HF organization listed
- Low upvotes (< 5)
- No GitHub repo or very few stars (< 20)
This is a strong signal that the paper is from unknown authors and unlikely to be impactful.
Do NOT penalize papers that have at least one author from a known institution.

You MUST cite the single most similar paper title from the reading history.
Copy the title EXACTLY as it appears in the list. If nothing is similar, write "NONE".

The user cares about papers that solve a related problem to something already in their
reading history. If a candidate paper doesn't clearly address a problem that an existing
paper also tackles, it's not interesting — no matter how impressive it sounds.

Be harsh. Most papers should score 1-4. A paper about viral capsid engineering
has NOTHING to do with policy optimization — do not fabricate connections.

{examples_block}
"""


def _format_authors_with_affiliations(paper: Paper) -> str:
    """Format authors with their affiliations if available."""
    if not paper.affiliations:
        return ", ".join(paper.authors)
    parts = []
    for name in paper.authors:
        affs = paper.affiliations.get(name)
        if affs:
            parts.append(f"{name} ({', '.join(affs)})")
        else:
            parts.append(name)
    return ", ".join(parts)


def _raise_fd_limit() -> None:
    """Try to raise the process soft fd limit to the hard limit.

    macOS defaults to 256, which collapses at any serious concurrency.
    """
    import resource  # POSIX-only, keep local so the module imports on Windows.

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError) as exc:
        logger.warning("Could not raise fd limit from {}: {}", soft, exc)
    else:
        logger.info("Raised fd limit: {} -> {}", soft, target)


class AsyncClientPool:
    """Single AsyncOpenAI client with a bounded httpx connection pool.

    Straightforward replacement for the round-robin pool: one client,
    one httpx.AsyncClient, max 64 concurrent TCP connections total.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        pool_size: int | None = None,
        concurrency: int = 4096,
        timeout: float = 1500.0,
    ):
        _raise_fd_limit()
        max_total_conns = 64
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=timeout,
            max_retries=0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max_total_conns,
                    max_keepalive_connections=max_total_conns,
                ),
                timeout=timeout,
            ),
        )
        self.pool_size = 1
        logger.info(
            f"Initialized 1 async client at {base_url} ({max_total_conns} total conns max)"
        )

    def get(self) -> AsyncOpenAI:
        return self.client

    async def close(self) -> None:
        await self.client.close()


async def score_paper(pool: AsyncClientPool, paper: Paper, examples_block: str) -> ScoredPaper:
    authors_str = _format_authors_with_affiliations(paper)
    org_str = f"\nOrganization: {paper.org_fullname or paper.organization}" if paper.organization else ""
    categories_str = ", ".join(paper.categories) if paper.categories else "N/A"
    comment_str = f"\nComment: {paper.arxiv_comment}" if paper.arxiv_comment else ""
    journal_str = f"\nJournal: {paper.journal_ref}" if paper.journal_ref else ""
    user_msg = f"""\
Title: {paper.title}
arXiv: {paper.arxiv_id}
Categories: {categories_str}
Upvotes: {paper.upvotes}
Authors: {authors_str}{org_str}{comment_str}{journal_str}
GitHub: {paper.github_repo or "N/A"} ({paper.github_stars or 0} stars)
Keywords: {", ".join(paper.ai_keywords) if paper.ai_keywords else "N/A"}

Abstract:
{paper.summary}
"""

    system = SYSTEM_PROMPT.format(examples_block=examples_block)
    # Pre-bind ``result`` so static analyzers don't flag the "possibly
    # unbound" path — the final ``attempt == 4`` branch always re-raises, so
    # ``result`` is only read after a successful ``break``, but linters can't
    # prove that without the explicit initial value.
    result: ScoreOutput | None = None

    for attempt in range(5):
        try:
            client = pool.get()
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "score_output",
                        "schema": ScoreOutput.model_json_schema(),
                    },
                },
            )
            raw = resp.choices[0].message.content or ""
            result = ScoreOutput.model_validate_json(raw)
            break
        except Exception as exc:
            if attempt == 4:
                raise
            wait = 2**attempt
            logger.warning(
                "score_paper {} attempt {} failed: {}, retrying in {}s",
                paper.arxiv_id,
                attempt + 1,
                exc,
                wait,
            )
            await asyncio.sleep(wait)

    assert result is not None  # narrowed by the loop above
    return ScoredPaper(
        paper=paper,
        score=result.score,
        similar_paper=result.similar_paper,
        reason=result.reason,
    )


# ============================================================================
# Main
# ============================================================================


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


async def async_main():
    parser = argparse.ArgumentParser(description="HF Daily Papers filter")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--concurrency", type=int, default=4096, help="Max concurrent scoring requests (default: 4096)")
    parser.add_argument("--threshold", type=int, default=6, help="Min score to include (1-10)")
    parser.add_argument(
        "--from",
        dest="date_from",
        type=_parse_date,
        default=None,
        help="Start date (YYYY-MM-DD). If set, fetches a date range.",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        type=_parse_date,
        default=None,
        help="End date (YYYY-MM-DD, default: today). Used with --from.",
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=None,
        help="Fetch a single date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Tasks launched per burst before sleeping (default: 32)")
    parser.add_argument("--batch-delay", type=float, default=0.3, help="Seconds to sleep between task-launch bursts (default: 0.3)")
    # Keep --workers as deprecated alias for --concurrency
    parser.add_argument("--workers", type=int, default=None, help="(deprecated, use --concurrency)")
    args = parser.parse_args()

    concurrency = args.workers if args.workers is not None else args.concurrency

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which papers to fetch (sync — HF/arxiv APIs are not the bottleneck)
    if args.date_from:
        end = args.date_to or date.today()
        papers = fetch_papers_range(args.date_from, end)
    else:
        papers = fetch_papers(args.date)

    logger.info(f"Using BASE_URL={BASE_URL} MODEL={MODEL}")

    pool = AsyncClientPool(base_url=BASE_URL, concurrency=concurrency)

    # Preflight: fail fast if SGLang/vLLM isn't actually running. Otherwise
    # every scoring call hits retry_async and we waste 10 backoff attempts per
    # paper × thousands of papers before anything visible happens.
    try:
        client = pool.get()
        models = await client.models.list()
        logger.info(f"Server models: {[m.id for m in models.data]}")
    except Exception as e:
        logger.error(
            f"Server health check failed at {BASE_URL}: {e!r}. "
            f"Start the LLM server and retry."
        )
        raise SystemExit(2) from e

    # Load examples from the papers repo (Neon + external SQLite + docs/)
    db = NeonDB()
    examples = load_examples(PAPERS_REPO, db=db)
    examples_block = build_examples_block(examples)

    # Skip papers already in examples (they'd trivially match themselves)
    example_titles = {e.title.lower() for e in examples}
    before = len(papers)
    papers = [p for p in papers if p.title.lower() not in example_titles]
    if before - len(papers) > 0:
        logger.info(f"Skipped {before - len(papers)} papers already in examples DB")

    # Load cached results to skip already-scored papers
    cached_results_path = out_dir / "all_scored.json"
    cached: dict[str, dict] = {}
    if cached_results_path.exists():
        try:
            for entry in json.loads(cached_results_path.read_text()):
                if entry.get("arxiv_id") and entry.get("score") is not None and entry.get("reason"):
                    cached[entry["arxiv_id"]] = entry
            logger.info(f"Loaded {len(cached)} cached scores from {cached_results_path}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load cache: {e}")

    scored: list[ScoredPaper] = []

    # Restore cached scores
    papers_to_score = []
    for p in papers:
        if p.arxiv_id in cached:
            c = cached[p.arxiv_id]
            scored.append(
                ScoredPaper(
                    paper=p,
                    score=c["score"],
                    similar_paper=c.get("similar_paper", "NONE"),
                    reason=c["reason"],
                )
            )
        else:
            papers_to_score.append(p)

    if papers_to_score:
        logger.info(f"Scoring {len(papers_to_score)} new papers ({len(scored)} cached, concurrency={concurrency})")
    else:
        logger.info(f"All {len(scored)} papers already scored, nothing to do")

    save_path = out_dir / "all_scored.json"

    def _save_progress_sync():
        snapshot = list(scored)
        all_results = [
            {
                "arxiv_id": s.paper.arxiv_id,
                "title": s.paper.title,
                "score": s.score,
                "similar_paper": s.similar_paper,
                "reason": s.reason,
                "upvotes": s.paper.upvotes,
                "github": s.paper.github_repo,
                "github_stars": s.paper.github_stars,
                "keywords": s.paper.ai_keywords,
                "authors": s.paper.authors,
                "affiliations": s.paper.affiliations,
                "organization": s.paper.organization,
                "org_fullname": s.paper.org_fullname,
                "categories": s.paper.categories,
                "primary_category": s.paper.primary_category,
                "arxiv_comment": s.paper.arxiv_comment,
                "published": s.paper.published,
                "journal_ref": s.paper.journal_ref,
                "doi": s.paper.doi,
                "summary": s.paper.summary,
            }
            for s in snapshot
        ]
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".json.tmp")
        try:
            with open(fd, "w") as f:
                json.dump(all_results, f, indent=2)
            Path(tmp).replace(save_path)
        except Exception:
            Path(tmp).unlink(missing_ok=True)
            raise

    save_every = 25
    completed = 0
    sem = asyncio.Semaphore(concurrency)

    async def _score_one(paper: Paper) -> ScoredPaper | None:
        nonlocal completed
        async with sem:
            try:
                result = await score_paper(pool, paper, examples_block)
                scored.append(result)
                completed += 1
                logger.info(f"[{result.score}/10] {paper.title[:80]} — {result.reason[:60]}")
                if completed % save_every == 0:
                    await asyncio.to_thread(_save_progress_sync)
                    logger.debug(f"Saved progress: {len(scored)} total scored")
                return result
            except Exception as e:
                import traceback
                logger.error(f"Failed {paper.arxiv_id}: {type(e).__name__}: {e}\n{traceback.format_exc()}")
                return None

    # Launch scoring tasks in bursts: post batch_size tasks, sleep, repeat.
    # Avoids hammering the port-forward with a single giant burst.
    tasks: list[asyncio.Task] = []
    batch_size = max(1, args.batch_size)
    batch_delay = max(0.0, args.batch_delay)
    for i in range(0, len(papers_to_score), batch_size):
        batch = papers_to_score[i : i + batch_size]
        tasks.extend(asyncio.create_task(_score_one(p)) for p in batch)
        if i + batch_size < len(papers_to_score) and batch_delay > 0:
            logger.debug(f"Launched {i + len(batch)}/{len(papers_to_score)} tasks, sleeping {batch_delay}s")
            await asyncio.sleep(batch_delay)
    await asyncio.gather(*tasks)

    # Final save
    await asyncio.to_thread(_save_progress_sync)

    await pool.close()

    scored.sort(key=lambda s: s.score, reverse=True)
    relevant = [s for s in scored if s.score >= args.threshold]

    # Write markdown summary of relevant papers
    lines = [f"# Relevant Papers ({len(relevant)}/{len(scored)} scored)\n"]
    for s in relevant:
        gh = f" | [code]({s.paper.github_repo})" if s.paper.github_repo else ""
        cats = f" | {', '.join(s.paper.categories)}" if s.paper.categories else ""
        lines.append(f"## [{s.score}/10] {s.paper.title}")
        lines.append(f"[arXiv](https://arxiv.org/abs/{s.paper.arxiv_id}){gh} | {s.paper.upvotes} upvotes{cats}")
        authors_str = _format_authors_with_affiliations(s.paper)
        lines.append(f"Authors: {authors_str}")
        if s.paper.org_fullname or s.paper.organization:
            lines.append(f"Org: {s.paper.org_fullname or s.paper.organization}")
        if s.paper.arxiv_comment:
            lines.append(f"Comment: {s.paper.arxiv_comment}")
        if s.paper.journal_ref:
            lines.append(f"Journal: {s.paper.journal_ref}")
        if s.paper.published:
            lines.append(f"Published: {s.paper.published}")
        if s.similar_paper and s.similar_paper != "NONE":
            lines.append(f"\n**Similar to:** {s.similar_paper}")
        lines.append(f"\n**Why:** {s.reason}\n")
        if s.paper.ai_keywords:
            lines.append(f"Keywords: {', '.join(s.paper.ai_keywords)}\n")
        lines.append("---\n")

    (out_dir / "relevant.md").write_text("\n".join(lines))

    logger.success(f"Done: {len(relevant)} relevant / {len(scored)} total (threshold={args.threshold})")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
