#!/usr/bin/env python3
"""External paper processing: Google Scholar URLs and arbitrary PDFs.

Consolidates: paper types, metadata extraction, scholar scraping, DB, and processing.
Reuses PaperSummarizer from main.py for the actual summarization.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import httpx
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

EXTERNAL_DB = "external_papers.db"


async def init_external_db() -> None:
    """Initialize external papers database."""
    async with aiosqlite.connect(EXTERNAL_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                title TEXT,
                first_author TEXT,
                year INTEGER,
                doi TEXT,
                venue TEXT,
                category TEXT,
                pitch TEXT,
                summary TEXT,
                source_url TEXT,
                pdf_path TEXT,
                full_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)")
        await db.commit()
    logger.debug(f"External DB initialized: {EXTERNAL_DB}")


async def save_external_paper(record: ExternalPaperRecord) -> None:
    """Save external paper to database."""
    async with aiosqlite.connect(EXTERNAL_DB) as db:
        await db.execute(
            """INSERT OR REPLACE INTO papers
               (paper_id, source, title, first_author, year, doi, venue, category, pitch, summary, source_url, pdf_path, full_response)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.paper_id, record.source, record.title, record.first_author,
                record.year, record.doi, record.venue, record.category, record.pitch,
                record.summary, record.source_url, record.pdf_path, record.full_response,
            ),
        )
        await db.commit()
    logger.info(f"Saved external paper: {record.paper_id}")


async def external_paper_exists(paper_id: str) -> bool:
    """Check if paper exists by ID."""
    async with aiosqlite.connect(EXTERNAL_DB) as db:
        async with db.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)) as cur:
            return await cur.fetchone() is not None


# =============================================================================
# Scholar Scraper
# =============================================================================


def scrape_scholar(url: str) -> ScholarData:
    """Scrape Google Scholar citation page."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    result = ScholarData(scholar_url=url)

    # Title
    title_el = soup.select_one("#gsc_oci_title a, #gsc_oci_title")
    if title_el:
        result.title = title_el.get_text(strip=True)
        if title_el.get("href"):
            result.article_link = title_el["href"]

    # Field rows
    for row in soup.select(".gs_scl"):
        field = row.select_one(".gsc_oci_field")
        value = row.select_one(".gsc_oci_value")
        if not field or not value:
            continue
        fname, fval = field.get_text(strip=True).lower(), value.get_text(strip=True)

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

    # Citation count
    if cited := soup.select_one('a[href*="cites="]'):
        if m := re.search(r"Cited by (\d[\d,]*)", cited.get_text()):
            result.citation_count = int(m.group(1).replace(",", ""))

    # PDF link
    for link in soup.select('a[href*=".pdf"]'):
        if href := link.get("href"):
            result.pdf_link = href
            break
    if not result.pdf_link:
        for link in soup.select(".gsc_oci_title_ggi a"):
            href, text = link.get("href", ""), link.get_text(strip=True).lower()
            if "pdf" in text or ".pdf" in href:
                result.pdf_link = href
                break

    logger.info(f"Scraped: '{result.title}' by {result.first_author}")
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


async def extract_metadata(client: AsyncOpenAI, pdf_input: dict) -> PaperMetadata:
    """Extract metadata from PDF using LLM."""
    agent = Agent(
        name="Metadata Extractor",
        instructions=_EXTRACTION_PROMPT,
        model="gpt-4o-mini",
        output_type=PaperMetadata,  # Handles structured JSON output automatically
    )
    result = await Runner.run(agent, [pdf_input])
    metadata: PaperMetadata = result.final_output
    logger.info(f"Extracted: '{metadata.title[:40]}...' by {metadata.first_author}, doi={metadata.doi}")
    return metadata


# =============================================================================
# PDF Input Builder (reusable)
# =============================================================================


async def build_pdf_input(pdf_url: str | None = None, pdf_path: Path | None = None) -> dict[str, Any]:
    """Build PDF input item for LLM from URL or local path."""
    if pdf_url:
        return {"role": "user", "content": [{"type": "input_file", "file_url": pdf_url}]}
    if pdf_path and pdf_path.exists():
        import base64
        b64 = base64.b64encode(await AsyncPath(pdf_path).read_bytes()).decode()
        return {
            "role": "user",
            "content": [{"type": "input_file", "file_data": f"data:application/pdf;base64,{b64}", "filename": pdf_path.name}],
        }
    raise ValueError("No PDF source provided")


# =============================================================================
# File Saving
# =============================================================================


async def save_external_summary(record: ExternalPaperRecord) -> Path:
    """Save summary to docs/{category}/."""
    category_dir = Path("docs") / record.category
    category_dir.mkdir(parents=True, exist_ok=True)

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", record.title).strip("-")[:80]
    safe_id = record.paper_id.replace(":", "-").replace("/", "-")
    base = f"{safe_id}-{normalized}.md" if normalized else f"{safe_id}.md"
    out = category_dir / base

    if out.exists():
        for k in range(2, 1000):
            candidate = category_dir / f"{out.stem}-{k}.md"
            if not candidate.exists():
                out = candidate
                break

    scholar_link = f"\n**Google Scholar:** [{record.title}]({record.source_url})\n" if record.source == PaperSource.SCHOLAR and record.source_url else ""
    doi_link = f"**DOI:** [{record.doi}](https://doi.org/{record.doi})\n" if record.doi else ""
    content = f"""# {record.title}
{scholar_link}{doi_link}
## Pitch

{record.pitch}

---

{record.summary}
"""
    await AsyncPath(out).write_text(content)
    logger.info(f"Saved: {out}")
    return out


# =============================================================================
# Main Processing (uses PaperSummarizer from main.py)
# =============================================================================


async def process_external(
    summarizer: Any,  # PaperSummarizer from main.py
    source: PaperSource,
    pdf_url: str | None = None,
    pdf_path: Path | None = None,
    scholar_url: str | None = None,
    instructions: str = "",
) -> ExternalPaperRecord | None:
    """Process external paper using shared summarizer.

    Args:
        summarizer: PaperSummarizer instance from main.py
        source: PaperSource.SCHOLAR or PaperSource.LOCAL
        pdf_url: Direct PDF URL
        pdf_path: Local PDF path
        scholar_url: Original Scholar URL (for reference)
        instructions: System prompt for summarization

    Returns:
        ExternalPaperRecord or None if duplicate
    """
    await init_external_db()

    # Build PDF input
    pdf_input = await build_pdf_input(pdf_url=pdf_url, pdf_path=pdf_path)

    # Extract metadata for dedup
    metadata = await extract_metadata(summarizer.client, pdf_input)
    paper_id = metadata.generate_id()

    if await external_paper_exists(paper_id):
        logger.warning(f"Duplicate: {paper_id}")
        return None

    # Run summarization pipeline (reuse from main.py)
    # We need to import here to avoid circular imports
    from main import SummarizationRequest, PitchOutput
    import orjson

    # Create a request - we pass the pdf_url or pdf_path
    # For external papers we don't have arxiv_url
    request = SummarizationRequest(
        arxiv_url=None,
        pdf_path=pdf_path,
        instructions=instructions,
    )

    # Generate summary - but we need to handle the case where we have a URL not arxiv
    # Let's call the individual methods instead
    logger.info("Step 1/3: Generating full analysis...")

    # Build proper model settings
    from agents import ModelSettings
    from openai.types.shared import Reasoning
    from main import ModelName, build_service_tier_args, serialize_response

    model_settings = ModelSettings(
        reasoning=Reasoning(effort="high", summary="detailed"),
        verbosity="high",
        extra_args=build_service_tier_args(summarizer.service_tier),
    )

    from agents import Agent, Runner
    agent = Agent(
        name="Paper Analyzer",
        instructions=instructions,
        model=ModelName.ANALYZER,
        model_settings=model_settings,
    )

    result = await Runner.run(agent, [pdf_input])
    full_summary = result.final_output
    summary_response = serialize_response(result)

    logger.info("Step 2/3: Generating pitch...")
    pitch_output, pitch_response = await summarizer.generate_pitch(full_summary, pdf_path=pdf_path)
    # For URL-based, we need to handle differently - pass the pdf_input
    if pdf_url and not pdf_path:
        # Re-run pitch with URL
        from main import PitchOutput as PO
        pitch_settings = ModelSettings(
            reasoning=Reasoning(effort="low", summary="auto"),
            verbosity="low",
            extra_args=build_service_tier_args(summarizer.service_tier),
        )
        pitch_agent = Agent(
            name="Pitch Generator",
            instructions="Extract the exact paper title and generate a compelling pitch.",
            model=ModelName.PITCH,
            output_type=PO,
            model_settings=pitch_settings,
        )
        pitch_prompt = f"""Extract the exact title and generate a 2-3 sentence pitch.
The pitch should capture the core contribution and why it matters.

Paper Analysis:
{full_summary[:2000]}..."""
        pitch_result = await Runner.run(pitch_agent, [pdf_input, {"role": "user", "content": pitch_prompt}])
        pitch_output = pitch_result.final_output
        pitch_response = serialize_response(pitch_result)

    logger.info("Step 3/3: Categorizing...")
    category, cat_response = await summarizer.categorize_paper(pitch_output.title, pitch_output.pitch, full_summary)

    full_response = orjson.dumps({
        "summary": summary_response,
        "pitch": pitch_response,
        "category": cat_response,
    }).decode()

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

    logger.success(f"Processed: {record.title[:50]}... [{record.paper_id}]")
    return record


async def process_scholar_url(summarizer: Any, url: str, instructions: str) -> ExternalPaperRecord | None:
    """Process Google Scholar citation URL."""
    data = scrape_scholar(url)
    if not data.pdf_link:
        raise ValueError(f"No PDF link found for: {url}")
    return await process_external(
        summarizer, PaperSource.SCHOLAR,
        pdf_url=data.pdf_link, scholar_url=url, instructions=instructions,
    )


async def process_local_pdf(summarizer: Any, pdf_path: Path, instructions: str) -> ExternalPaperRecord | None:
    """Process local PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return await process_external(
        summarizer, PaperSource.LOCAL,
        pdf_path=pdf_path, instructions=instructions,
    )
