#!/usr/bin/env python3
"""External paper processing: Google Scholar URLs and arbitrary (non-arxiv) PDFs.

Consolidates paper types, metadata extraction, scholar scraping, the local
``external_papers.db`` SQLite layer, and the end-to-end processing flow.
Reuses :class:`PaperSummarizer` from :mod:`main` for the actual summarization.

Note: this module intentionally keeps its storage in SQLite
(``external_papers.db``). Only the main ``papers.db`` moved to Neon.
"""

from __future__ import annotations

import base64
import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import httpx
import orjson
from agents import Agent, ModelSettings, Runner
from anyio import Path as AsyncPath
from bs4 import BeautifulSoup
from loguru import logger
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from openai import AsyncOpenAI

# =============================================================================
# Types
# =============================================================================


class PaperSource(StrEnum):
    ARXIV = "arxiv"
    SCHOLAR = "scholar"
    LOCAL = "local"


class PaperMetadata(BaseModel):
    """LLM-extracted metadata for deduplication."""

    title: str = Field(description="Exact paper title")
    first_author: str = Field(description="First author's name")
    year: int | None = Field(default=None, description="Publication year")
    doi: str | None = Field(default=None, description="DOI if present")
    venue: str | None = Field(default=None, description="Publication venue")

    def generate_id(self) -> str:
        """Generate stable ID: DOI if available, else hash(title+author)."""
        if self.doi:
            return f"doi:{self.doi.lower().strip()}"
        normalized = f"{self.title.lower().strip()}:{self.first_author.lower().strip()}"
        return f"hash:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"


class ScholarData(BaseModel):
    """Scraped Google Scholar metadata."""

    title: str | None = None
    authors: str | None = None
    first_author: str | None = None
    venue: str | None = None
    year: str | None = None
    citation_count: int | None = None
    description: str | None = None
    pdf_link: str | None = None
    article_link: str | None = None
    scholar_url: str | None = None


class ExternalPaperRecord(BaseModel):
    """Record for external papers (Scholar/local PDFs)."""

    paper_id: str
    source: PaperSource
    title: str
    first_author: str
    year: int | None = None
    doi: str | None = None
    venue: str | None = None
    category: str
    pitch: str
    summary: str
    source_url: str | None = None
    pdf_path: str | None = None
    full_response: str


# =============================================================================
# Database (external_papers.db)
# =============================================================================

EXTERNAL_DB: str = "external_papers.db"

_CREATE_PAPERS_TABLE = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    title         TEXT,
    first_author  TEXT,
    year          INTEGER,
    doi           TEXT,
    venue         TEXT,
    category      TEXT,
    pitch         TEXT,
    summary       TEXT,
    source_url    TEXT,
    pdf_path      TEXT,
    full_response TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INSERT_PAPER = """
INSERT OR REPLACE INTO papers
    (paper_id, source, title, first_author, year, doi, venue,
     category, pitch, summary, source_url, pdf_path, full_response)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


async def init_external_db() -> None:
    """Create the ``external_papers.db`` schema if absent."""
    async with aiosqlite.connect(EXTERNAL_DB) as db:
        await db.execute(_CREATE_PAPERS_TABLE)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)")
        await db.commit()
    logger.debug("External DB initialized: {}", EXTERNAL_DB)


async def save_external_paper(record: ExternalPaperRecord) -> None:
    """Insert-or-replace a fully-processed external paper row."""
    async with aiosqlite.connect(EXTERNAL_DB) as db:
        await db.execute(
            _INSERT_PAPER,
            (
                record.paper_id,
                record.source,
                record.title,
                record.first_author,
                record.year,
                record.doi,
                record.venue,
                record.category,
                record.pitch,
                record.summary,
                record.source_url,
                record.pdf_path,
                record.full_response,
            ),
        )
        await db.commit()
    logger.info("Saved external paper: {}", record.paper_id)


async def external_paper_exists(paper_id: str) -> bool:
    """Return True if ``paper_id`` already has a row in the external DB."""
    async with (
        aiosqlite.connect(EXTERNAL_DB) as db,
        db.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)) as cur,
    ):
        return await cur.fetchone() is not None


# =============================================================================
# Scholar Scraper
# =============================================================================


_SCHOLAR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.5",
}
_SCHOLAR_TIMEOUT_SECONDS: float = 30.0


def _extract_scholar_title(soup: BeautifulSoup, result: ScholarData) -> None:
    title_el = soup.select_one("#gsc_oci_title a, #gsc_oci_title")
    if not title_el:
        return
    result.title = title_el.get_text(strip=True)
    href = title_el.get("href")
    if href:
        result.article_link = href


def _extract_scholar_fields(soup: BeautifulSoup, result: ScholarData) -> None:
    for row in soup.select(".gs_scl"):
        field = row.select_one(".gsc_oci_field")
        value = row.select_one(".gsc_oci_value")
        if not field or not value:
            continue
        fname = field.get_text(strip=True).lower()
        fval = value.get_text(strip=True)

        if "author" in fname:
            result.authors = fval
            result.first_author = fval.split(",")[0].strip() if fval else None
        elif "date" in fname:
            if m := re.search(r"\b(19|20)\d{2}\b", fval):
                result.year = m.group()
        elif any(x in fname for x in ("journal", "conference", "book")):
            result.venue = fval
        elif "description" in fname:
            result.description = fval


def _extract_scholar_citations(soup: BeautifulSoup, result: ScholarData) -> None:
    cited = soup.select_one('a[href*="cites="]')
    if not cited:
        return
    if m := re.search(r"Cited by (\d[\d,]*)", cited.get_text()):
        result.citation_count = int(m.group(1).replace(",", ""))


def _extract_scholar_pdf_link(soup: BeautifulSoup, result: ScholarData) -> None:
    for link in soup.select('a[href*=".pdf"]'):
        if href := link.get("href"):
            result.pdf_link = href
            return
    for link in soup.select(".gsc_oci_title_ggi a"):
        href = link.get("href", "")
        text = link.get_text(strip=True).lower()
        if "pdf" in text or ".pdf" in href:
            result.pdf_link = href
            return


def scrape_scholar(url: str) -> ScholarData:
    """Scrape a Google Scholar citation page into a :class:`ScholarData`."""
    resp = httpx.get(
        url,
        headers=_SCHOLAR_HEADERS,
        follow_redirects=True,
        timeout=_SCHOLAR_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    result = ScholarData(scholar_url=url)
    _extract_scholar_title(soup, result)
    _extract_scholar_fields(soup, result)
    _extract_scholar_citations(soup, result)
    _extract_scholar_pdf_link(soup, result)

    logger.info("Scraped: '{}' by {}", result.title, result.first_author)
    return result


# =============================================================================
# Metadata Extraction (LLM)
# =============================================================================

_EXTRACTION_PROMPT = """Extract metadata from this paper PDF. Return JSON with:
- title: exact paper title
- first_author: first author's full name
- year: publication year (integer or null)
- doi: DOI if present (format "10.xxxx/...") or null
- venue: publication venue or null

Return ONLY valid JSON."""


async def extract_metadata(client: AsyncOpenAI, pdf_input: dict[str, Any]) -> PaperMetadata:
    """Extract structured metadata from a PDF using an LLM call."""
    agent = Agent(
        name="Metadata Extractor",
        instructions=_EXTRACTION_PROMPT,
        model="gpt-4o-mini",
        output_type=PaperMetadata,
    )
    result = await Runner.run(agent, [pdf_input])
    metadata: PaperMetadata = result.final_output
    logger.info(
        "Extracted: '{}...' by {}, doi={}",
        metadata.title[:40],
        metadata.first_author,
        metadata.doi,
    )
    return metadata


# =============================================================================
# PDF Input Builder (reusable)
# =============================================================================


async def build_pdf_input(
    pdf_url: str | None = None,
    pdf_path: Path | None = None,
) -> dict[str, Any]:
    """Build the LLM ``input_file`` item from either a URL or a local path."""
    if pdf_url:
        return {
            "role": "user",
            "content": [{"type": "input_file", "file_url": pdf_url}],
        }
    if pdf_path and pdf_path.exists():
        raw = await AsyncPath(pdf_path).read_bytes()
        b64 = base64.b64encode(raw).decode()
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_data": f"data:application/pdf;base64,{b64}",
                    "filename": pdf_path.name,
                }
            ],
        }
    raise ValueError("No PDF source provided: pass pdf_url or an existing pdf_path")


# =============================================================================
# File Saving
# =============================================================================


_FILENAME_TITLE_LIMIT: int = 80
_MAX_FILENAME_SUFFIX_ATTEMPTS: int = 1000


def _unique_output_path(category_dir: Path, base_name: str) -> Path:
    """Return a non-colliding path inside ``category_dir`` for ``base_name``."""
    candidate = category_dir / base_name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for k in range(2, _MAX_FILENAME_SUFFIX_ATTEMPTS):
        alt = category_dir / f"{stem}-{k}{suffix}"
        if not alt.exists():
            return alt
    raise RuntimeError(f"Could not find a free filename under {category_dir} for {base_name}")


async def save_external_summary(record: ExternalPaperRecord) -> Path:
    """Write the markdown summary for an external paper under ``docs/{category}/``."""
    category_dir = Path("docs") / record.category
    category_dir.mkdir(parents=True, exist_ok=True)

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", record.title).strip("-")[:_FILENAME_TITLE_LIMIT]
    safe_id = record.paper_id.replace(":", "-").replace("/", "-")
    base_name = f"{safe_id}-{normalized}.md" if normalized else f"{safe_id}.md"
    out = _unique_output_path(category_dir, base_name)

    scholar_link = (
        f"\n**Google Scholar:** [{record.title}]({record.source_url})\n"
        if record.source == PaperSource.SCHOLAR and record.source_url
        else ""
    )
    doi_link = f"**DOI:** [{record.doi}](https://doi.org/{record.doi})\n" if record.doi else ""
    content = f"# {record.title}\n{scholar_link}{doi_link}\n## Pitch\n\n{record.pitch}\n\n---\n\n{record.summary}\n"

    await AsyncPath(out).write_text(content)
    logger.info("Saved: {}", out)
    return out


# =============================================================================
# Main Processing (uses PaperSummarizer from main.py)
# =============================================================================


async def _generate_full_summary_from_pdf_input(
    summarizer: Any,
    pdf_input: dict[str, Any],
    instructions: str,
) -> tuple[str, dict[str, Any]]:
    """Run the full analyzer agent directly on a raw ``pdf_input`` item.

    We don't use :meth:`PaperSummarizer.generate_full_summary` here because it
    assumes an ArXiv URL or local path; for external papers we already built
    the PDF input item up-front so it is reused verbatim for summary + pitch.
    """
    from main import ModelName, build_service_tier_args, serialize_response

    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high", summary="detailed"),
        verbosity="high",
        extra_args=build_service_tier_args(summarizer.service_tier),
    )
    agent = Agent(
        name="Paper Analyzer",
        instructions=instructions,
        model=ModelName.ANALYZER,
        model_settings=model_settings,
    )
    result = await Runner.run(agent, [pdf_input])
    return result.final_output, serialize_response(result)


async def _generate_pitch_from_pdf_input(
    summarizer: Any,
    pdf_input: dict[str, Any],
    full_summary: str,
) -> tuple[Any, dict[str, Any]]:
    """Run the pitch agent directly on an existing ``pdf_input`` item."""
    from main import ModelName, build_service_tier_args, serialize_response
    from multi_prompt import PitchOutput

    pitch_settings = ModelSettings(
        reasoning=Reasoning(effort="low", summary="auto"),
        verbosity="low",
        extra_args=build_service_tier_args(summarizer.service_tier),
    )
    pitch_agent = Agent(
        name="Pitch Generator",
        instructions="Extract the exact paper title and generate a compelling pitch.",
        model=ModelName.PITCH,
        output_type=PitchOutput,
        model_settings=pitch_settings,
    )
    prompt = (
        "Extract the exact title and generate a 2-3 sentence pitch. "
        "The pitch should capture the core contribution and why it matters.\n\n"
        f"Paper Analysis:\n{full_summary[:2000]}..."
    )
    result = await Runner.run(
        pitch_agent,
        [pdf_input, {"role": "user", "content": prompt}],
    )
    return result.final_output, serialize_response(result)


async def process_external(
    summarizer: Any,
    source: PaperSource,
    *,
    pdf_url: str | None = None,
    pdf_path: Path | None = None,
    scholar_url: str | None = None,
    instructions: str = "",
) -> ExternalPaperRecord | None:
    """Process a single external paper end-to-end.

    Args:
        summarizer: :class:`main.PaperSummarizer` instance (owns the LLM client).
        source: Provenance of the paper.
        pdf_url: Direct PDF URL (mutually exclusive with ``pdf_path``).
        pdf_path: Local PDF path (mutually exclusive with ``pdf_url``).
        scholar_url: Original Scholar page, kept for reference.
        instructions: System prompt for the analyzer agent.

    Returns:
        The persisted :class:`ExternalPaperRecord`, or ``None`` if the paper
        was already in ``external_papers.db`` (deduplicated by DOI or
        title+author hash).
    """
    await init_external_db()

    pdf_input = await build_pdf_input(pdf_url=pdf_url, pdf_path=pdf_path)
    metadata = await extract_metadata(summarizer.client, pdf_input)
    paper_id = metadata.generate_id()

    if await external_paper_exists(paper_id):
        logger.warning("Duplicate: {}", paper_id)
        return None

    logger.info("Step 1/3: Generating full analysis...")
    full_summary, summary_response = await _generate_full_summary_from_pdf_input(summarizer, pdf_input, instructions)

    logger.info("Step 2/3: Generating pitch...")
    pitch_output, pitch_response = await _generate_pitch_from_pdf_input(summarizer, pdf_input, full_summary)

    logger.info("Step 3/3: Categorizing...")
    category, cat_response = await summarizer.categorize_paper(pitch_output.title, pitch_output.pitch, full_summary)

    full_response = orjson.dumps(
        {
            "summary": summary_response,
            "pitch": pitch_response,
            "category": cat_response,
        }
    ).decode()

    record = ExternalPaperRecord(
        paper_id=paper_id,
        source=source,
        title=metadata.title,
        first_author=metadata.first_author,
        year=metadata.year,
        doi=metadata.doi,
        venue=metadata.venue,
        category=category,
        pitch=pitch_output.pitch,
        summary=full_summary,
        source_url=scholar_url or pdf_url,
        pdf_path=str(pdf_path) if pdf_path else None,
        full_response=full_response,
    )

    await save_external_paper(record)
    await save_external_summary(record)

    logger.success("Processed: {}... [{}]", record.title[:50], record.paper_id)
    return record


async def process_scholar_url(summarizer: Any, url: str, instructions: str) -> ExternalPaperRecord | None:
    """Scrape a Google Scholar page and run the external pipeline on its PDF."""
    data = scrape_scholar(url)
    if not data.pdf_link:
        raise ValueError(f"No PDF link found for: {url}")
    return await process_external(
        summarizer,
        PaperSource.SCHOLAR,
        pdf_url=data.pdf_link,
        scholar_url=url,
        instructions=instructions,
    )


async def process_local_pdf(summarizer: Any, pdf_path: Path, instructions: str) -> ExternalPaperRecord | None:
    """Run the external pipeline on a local non-arxiv PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return await process_external(
        summarizer,
        PaperSource.LOCAL,
        pdf_path=pdf_path,
        instructions=instructions,
    )
