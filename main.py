import asyncio
import base64
import os
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import click
import orjson
from agents import Agent, ModelSettings, Runner, set_default_openai_client
from agents.tracing import set_tracing_disabled
from anyio import Path as AsyncPath
from dotenv import load_dotenv
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field
from tqdm import tqdm

from database import init_db, save_to_db

# NOTE: On newer Python versions, relying on dotenv's automatic discovery can be fragile in
# some execution modes. Prefer loading from the repo root deterministically.
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.exists():
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

# The openai-agents SDK enables a default tracing exporter that talks to OpenAI's backend.
# When using a non-OpenAI proxy key (e.g., Shopify proxy), this can generate noisy 401s.
# Disable tracing by default for this CLI tool.
set_tracing_disabled(True)


# =============================================================================
# Configuration Constants
# =============================================================================


class ModelName(StrEnum):
    """Available model names for different tasks."""

    ANALYZER = "gpt-5.2"
    PITCH = "gpt-5-mini-2025-08-07"
    CATEGORIZER = "gpt-5-mini-2025-08-07"


class Timeout:
    """Timeout configurations in seconds."""

    DEFAULT = 600.0
    FLEX = 900.0

    @classmethod
    def for_tier(cls, service_tier: str | None) -> float:
        """Get appropriate timeout for service tier."""
        return cls.FLEX if service_tier == "flex" else cls.DEFAULT


class DefaultFiles:
    """Default file paths."""

    PROMPT = "main_prompt.txt"
    FALLBACK_FILENAME = "paper.md"


# Fallback category when categorization fails or returns invalid result
FALLBACK_CATEGORY = "uncategorized"

CATEGORIES = [
    "alignment",
    "architecture",
    "context-optimization",
    "evaluation",
    "inference-optimization",
    "llm-systems",
    "low-precision",
    "multimodal",
    "pretraining",
    "prompting",
    "retrieval",
    "rl-training",
    "serving",
    "training-methods",
]


class ReasoningConfig(BaseModel):
    effort: Literal["minimal", "low", "medium", "high"] = Field(default="high")
    summary: Literal["auto", "concise", "detailed"] = Field(default="detailed")


class TextConfig(BaseModel):
    verbosity: Literal["low", "medium", "high"] = Field(default="high")


class SummarizationRequest(BaseModel):
    """Request configuration for paper summarization.

    Either arxiv_url or pdf_path must be provided (or both).
    """

    model: str = Field(default=ModelName.ANALYZER)
    arxiv_url: str | None = Field(default=None, description="ArXiv URL (preferred for URL mode)")
    pdf_path: Path | None = Field(default=None, description="Local PDF path (fallback)")
    question: str | None = Field(
        default=None,
        description="Optional user question (if omitted, just uses system prompt)",
    )
    instructions: str
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    service_tier: str | None = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def arxiv_id(self) -> str | None:
        """Extract ArXiv ID from URL if available."""
        if self.arxiv_url:
            return arxiv_id_from_url(self.arxiv_url)
        if self.pdf_path:
            return self.pdf_path.stem
        return None


class PitchOutput(BaseModel):
    title: str = Field(description="The exact title of the paper as it appears in the PDF")
    pitch: str = Field(
        description="A compelling 2-3 sentence pitch that captures the paper's core contribution and why it matters",
    )


class CategoryOutput(BaseModel):
    category: str = Field(description="Best matching category from the available list")
    reasoning: str = Field(description="Brief explanation for the categorization")


class PaperRecord(BaseModel):
    """Data model for paper storage in database.

    Consolidates all paper data into a single object for cleaner function signatures.
    """

    arxiv_id: str = Field(description="ArXiv paper ID (primary key)")
    title: str = Field(description="Paper title")
    category: str = Field(description="Assigned category")
    pitch: str = Field(description="Generated pitch")
    summary: str = Field(description="Full summary text")
    url: str | None = Field(default=None, description="Original ArXiv URL")
    full_response: str = Field(description="JSON-serialized full API response")


async def encode_pdf(file_path: Path) -> str:
    """Encode PDF file to base64 asynchronously."""
    content = await AsyncPath(file_path).read_bytes()
    return base64.b64encode(content).decode("utf-8")


def get_openai_config() -> tuple[str, str]:
    api_key = os.getenv(
        "OPENAI_API_KEY",""    )
    base_url = "https://api.openai.com/v1"

    if not api_key:
        raise ValueError("LLM API api key environment variable is required")

    return api_key, base_url


class PaperSummarizer:
    """Manages paper summarization with proper client lifecycle.

    This class holds the OpenAI client as an instance attribute, enabling:
    - Connection pooling across multiple API calls
    - Proper timeout configuration based on service tier
    - Easy testing via dependency injection
    """

    def __init__(self, service_tier: str | None = None):
        """Initialize the summarizer with appropriate client configuration.

        Args:
            service_tier: Optional service tier (e.g., "flex" for longer timeouts)

        """
        self.service_tier = service_tier
        timeout = Timeout.for_tier(service_tier)

        api_key, base_url = get_openai_config()
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        set_default_openai_client(client=self.client, use_for_tracing=False)
        logger.debug(f"Initialized OpenAI client (timeout={timeout}s, tier={service_tier})")

    async def generate_full_summary(self, request: SummarizationRequest) -> tuple[str, dict[str, Any]]:
        """Generate full summary and return both output and raw response."""
        model_settings = ModelSettings(
            reasoning=Reasoning(
                effort=request.reasoning.effort,
                summary=request.reasoning.summary,
            ),
            verbosity=request.text.verbosity,
            extra_args=build_service_tier_args(request.service_tier),
        )

        agent = Agent(
            name="Paper Analyzer",
            instructions=request.instructions,
            model=request.model,
            model_settings=model_settings,
        )

        pdf_input = await build_pdf_input_item(arxiv_url=request.arxiv_url, pdf_path=request.pdf_path)
        input_items = [pdf_input]
        if request.question:
            input_items.append({"role": "user", "content": request.question})

        result = await Runner.run(agent, input_items)  # type: ignore
        return result.final_output, serialize_response(result)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    async def generate_pitch(
        self,
        full_summary: str,
        arxiv_url: str | None = None,
        pdf_path: Path | None = None,
    ) -> tuple[PitchOutput, dict[str, Any]]:
        """Generate pitch and return both output and raw response."""
        pitch_model_settings = ModelSettings(
            reasoning=Reasoning(effort="low", summary="auto"),
            verbosity="low",
            extra_args=build_service_tier_args(self.service_tier),
        )

        pitch_agent = Agent(
            name="Pitch Generator",
            instructions="Extract the exact paper title and generate a compelling pitch.",
            model=ModelName.PITCH,
            output_type=PitchOutput,
            model_settings=pitch_model_settings,
        )

        pitch_prompt = f"""Extract the exact title from the PDF and generate a compelling 2-3 sentence pitch.

The pitch should capture:
1. The core contribution/innovation
2. Why it matters (impact/significance)

Paper Analysis (for context):
{full_summary[:2000]}..."""

        # Build input items - use PDF if available, otherwise just the prompt
        if arxiv_url or (pdf_path and pdf_path.exists()):
            pdf_input = await build_pdf_input_item(arxiv_url=arxiv_url, pdf_path=pdf_path)
            input_items: list | str = [pdf_input, {"role": "user", "content": pitch_prompt}]
        else:
            input_items = f"""Based on this analysis, extract a title and generate a compelling 2-3 sentence pitch:

{full_summary[:2000]}..."""

        result = await Runner.run(pitch_agent, input_items)  # type: ignore
        return result.final_output, serialize_response(result)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
    async def categorize_paper(self, title: str, pitch: str, full_summary: str) -> tuple[str, dict[str, Any]]:
        """Categorize paper using GPT-5-mini based on title, pitch, and full summary.

        Returns:
            Tuple of (category string, raw response dict)

        """
        response = await self.client.responses.create(
            model=ModelName.CATEGORIZER,
            reasoning=Reasoning(effort="low"),
            input=[
                {
                    "role": "system",
                    "content": f"Categorize the paper into one of these categories: {', '.join(CATEGORIES)}. Respond with ONLY the category name, nothing else.",
                },
                {
                    "role": "user",
                    "content": f"Title: {title}\n\nPitch: {pitch}\n\nFull Summary:\n{full_summary}",
                },
            ],
        )

        raw_response = serialize_response(response)

        # Parse category from response
        content = response.output_text
        if not content:
            logger.warning(f"Categorization returned empty response for '{title[:50]}...', using fallback")
            return FALLBACK_CATEGORY, raw_response

        category = content.strip().lower()

        # Validate and return
        if category in CATEGORIES:
            return category, raw_response

        # Fallback: find closest match
        for cat in CATEGORIES:
            if cat in category:
                logger.debug(f"Partial category match: '{category}' -> '{cat}'")
                return cat, raw_response

        logger.warning(f"Invalid category '{category}' for '{title[:50]}...', using fallback")
        return FALLBACK_CATEGORY, raw_response

    async def process_paper(self, request: SummarizationRequest) -> tuple[PitchOutput, str, str, dict[str, Any]]:
        """Process a paper through the full pipeline.

        Returns:
            Tuple of (pitch_output, full_summary, category, aggregated_responses)

        """
        logger.info("Step 1/3: Generating full analysis with GPT-5.2 (high reasoning)...")
        full_summary, summary_response = await self.generate_full_summary(request)

        logger.info("Step 2/3: Extracting title and generating pitch with GPT-5-mini...")
        pitch_output, pitch_response = await self.generate_pitch(
            full_summary,
            arxiv_url=request.arxiv_url,
            pdf_path=request.pdf_path,
        )

        logger.info("Step 3/3: Categorizing paper with GPT-5-mini...")
        category, category_response = await self.categorize_paper(pitch_output.title, pitch_output.pitch, full_summary)
        logger.info(f"Category: {category}")

        full_response = {
            "summary_response": summary_response,
            "pitch_response": pitch_response,
            "category_response": category_response,
        }

        return pitch_output, full_summary, category, full_response


def arxiv_id_from_url(url: str) -> str:
    """Extract ArXiv ID from any ArXiv URL format.

    Safely handles URLs with query parameters, fragments, etc.

    Supports URLs like:
    - https://arxiv.org/abs/2312.07104
    - https://arxiv.org/abs/2312.07104v2
    - https://arxiv.org/pdf/2312.07104.pdf
    - https://arxiv.org/abs/2312.07104?context=cs.AI
    """
    parsed = urlparse(url)
    # Get just the path, ignoring query params and fragments
    path = parsed.path.rstrip("/")
    last_segment = path.split("/")[-1]
    return last_segment.replace(".pdf", "")


def arxiv_url_to_pdf_url(url: str) -> str:
    """Convert any ArXiv URL to a direct PDF URL (redirect-free)."""
    return url.replace("/abs/", "/pdf/").removesuffix(".pdf")


def build_service_tier_args(service_tier: str | None) -> dict[str, str] | None:
    """Build extra_args dict for service tier configuration."""
    if service_tier:
        return {"service_tier": service_tier}
    return None


async def build_pdf_input_item(arxiv_url: str | None = None, pdf_path: Path | None = None) -> dict[str, Any]:
    """Build the input item for PDF content (either URL or base64-encoded).

    Args:
        arxiv_url: ArXiv URL (preferred, avoids request size limits)
        pdf_path: Local PDF path (fallback)

    Returns:
        Input item dict with file content

    Raises:
        ValueError: If neither arxiv_url nor valid pdf_path provided

    """
    if arxiv_url:
        pdf_url = arxiv_url_to_pdf_url(arxiv_url)
        return {
            "role": "user",
            "content": [{"type": "input_file", "file_url": pdf_url}],
        }
    if pdf_path and pdf_path.exists():
        b64_file = await encode_pdf(pdf_path)
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "file_data": f"data:application/pdf;base64,{b64_file}",
                    "filename": pdf_path.name,
                },
            ],
        }
    raise ValueError("No PDF available: provide a local PDF path or an arXiv URL.")


class PaperProcessingError(RuntimeError):
    def __init__(self, url: str, original: Exception):
        super().__init__(f"{url}: {original}")
        self.url = url
        self.original = original


def serialize_response(obj: Any) -> dict[str, Any]:
    """Serialize an API response object to a JSON-compatible dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return {k: str(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"raw": str(obj)}


def normalize_title_for_filename(title: str, max_length: int = 80) -> str:
    """Normalize a title string for use in filenames.

    Strips control chars, collapses whitespace, and removes invalid path characters.
    """
    title = "".join(ch for ch in title if ord(ch) >= 32)  # drop control chars
    title = re.sub(r"\s+", " ", title).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-")[:max_length]


async def save_summary(
    pitch_output: PitchOutput,
    full_summary: str,
    category: str,
    arxiv_url: str | None = None,
) -> Path:
    """Save paper summary to categorized directory with proper metadata header."""
    category_dir = Path("docs") / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Extract ArXiv ID if available
    arxiv_id = arxiv_id_from_url(arxiv_url) if arxiv_url else None
    arxiv_link = f"\n**ArXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n" if arxiv_id else ""

    # Build filename
    normalized_title = normalize_title_for_filename(pitch_output.title)
    if arxiv_id:
        base_name = f"{arxiv_id}-{normalized_title}.md" if normalized_title else f"{arxiv_id}.md"
    else:
        base_name = f"{normalized_title}.md" if normalized_title else DefaultFiles.FALLBACK_FILENAME

    output_file = category_dir / base_name
    if output_file.exists():
        stem = output_file.stem
        suffix = output_file.suffix
        for k in range(2, 1000):
            candidate = category_dir / f"{stem}-{k}{suffix}"
            if not candidate.exists():
                output_file = candidate
                break

    # Format content with metadata header
    formatted_output = f"""# {pitch_output.title}
{arxiv_link}
## 🎯 Pitch

{pitch_output.pitch}

---

{full_summary}
"""

    await AsyncPath(output_file).write_text(formatted_output)
    return output_file


def load_prompt(prompt_path: str | None) -> str:
    """Load prompt from file, defaulting to main_prompt.txt."""
    path = Path(prompt_path) if prompt_path else Path(DefaultFiles.PROMPT)
    return path.read_text(encoding="utf-8").strip()


async def process_multiple_urls(
    model: str,
    urls: list[str],
    question: str | None,
    instructions: str | None,
    service_tier: str | None = None,
    concurrency: int = 1,
):
    """Process multiple ArXiv URLs (sequentially by default; optionally concurrently)."""
    concurrency = max(1, int(concurrency))
    if concurrency == 1:
        logger.info(f"Processing {len(urls)} papers...")
    else:
        logger.info(f"Processing {len(urls)} papers (concurrency={concurrency})...")
    if service_tier:
        logger.info(f"Using {service_tier} processing (lower cost, slower responses)")

    if concurrency == 1:
        for i, url in tqdm(list(enumerate(urls, 1)), total=len(urls), desc="Papers"):
            logger.info(f"Processing paper {i}/{len(urls)}: {url}")

            try:
                await async_main(model, url, None, question, instructions, service_tier)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                logger.info("Continuing to next paper...")
                continue
    else:
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(i: int, url: str) -> None:
            async with sem:
                try:
                    await async_main(model, url, None, question, instructions, service_tier)
                except Exception as e:
                    raise PaperProcessingError(url, e) from e

        tasks: list[asyncio.Task[None]] = []
        for i, url in enumerate(urls, 1):
            t = asyncio.create_task(_run_one(i, url))
            tasks.append(t)

        errors: list[tuple[str, Exception]] = []
        with tqdm(total=len(tasks), desc="Papers") as pbar:
            for done in asyncio.as_completed(tasks):
                try:
                    await done
                except PaperProcessingError as e:
                    errors.append((e.url, e.original))
                except Exception as e:
                    errors.append(("<unknown>", e))
                finally:
                    pbar.update(1)

        for url, e in errors:
            logger.error(f"Error processing {url}: {e}")

    logger.success(f"Finished processing {len(urls)} papers!")


@click.command()
@click.option("--model", default=ModelName.ANALYZER, help="Model to use for summarization")
@click.option("--url", help="ArXiv URL to download and summarize")
@click.option("--urls", help="Comma-separated list of ArXiv URLs to process")
@click.option("--pdf", help="Local PDF path to summarize (use with --external for non-arxiv)")
@click.option("--scholar", help="Google Scholar citation URL to process")
@click.option("--external", is_flag=True, help="Treat --pdf as external (non-arxiv) paper")
@click.option("--question", help="Optional user question prompt file (text). If omitted, uses a short default.")
@click.option("--instructions", help=f"System prompt file (text). Defaults to {DefaultFiles.PROMPT}.")
@click.option(
    "--concurrency",
    default=1,
    show_default=True,
    type=int,
    help="When using --urls, how many papers to process concurrently (1 = sequential).",
)
@click.option(
    "--flex",
    is_flag=True,
    help="Use flex processing for lower costs (slower, may have resource unavailability)",
)
def main(
    model: str,
    url: str | None,
    urls: str | None,
    pdf: str | None,
    scholar: str | None,
    external: bool,
    question: str | None,
    instructions: str | None,
    concurrency: int,
    flex: bool,
):
    service_tier = "flex" if flex else None

    async def _run():
        # Dispatch to external module for scholar/external PDFs
        if scholar or external:
            from external import process_scholar_url, process_local_pdf

            instructions_text = load_prompt(instructions)
            summarizer = PaperSummarizer(service_tier=service_tier)

            if scholar:
                result = await process_scholar_url(summarizer, scholar, instructions_text)
            elif pdf:
                result = await process_local_pdf(summarizer, Path(pdf), instructions_text)
            else:
                raise click.UsageError("--external requires --pdf")

            if result:
                logger.success(f"Title: {result.title}")
                logger.success(f"Category: {result.category}")
                logger.success(f"Paper ID: {result.paper_id}")
            return

        # ArXiv path (existing)
        await init_db()

        if urls:
            url_list = [u.strip() for u in urls.split(",") if u.strip()]
            await process_multiple_urls(model, url_list, question, instructions, service_tier, concurrency)
        else:
            await async_main(model, url, pdf, question, instructions, service_tier)

    asyncio.run(_run())


async def async_main(
    model: str,
    url: str | None,
    pdf: str | None,
    question: str | None,
    instructions: str | None,
    service_tier: str | None = None,
):
    if not url and not pdf:
        raise click.UsageError("Must provide either --url or --pdf")

    # Build request with clean data model (no fake paths needed)
    arxiv_url = url if url else None
    pdf_path = Path(pdf) if pdf else None

    question_text = Path(question).read_text(encoding="utf-8").strip() if question else None
    instructions_text = load_prompt(instructions)

    request = SummarizationRequest(
        model=model,
        arxiv_url=arxiv_url,
        pdf_path=pdf_path,
        question=question_text,
        instructions=instructions_text,
        service_tier=service_tier,
    )

    # Use PaperSummarizer with proper client lifecycle
    summarizer = PaperSummarizer(service_tier=service_tier)
    pitch_output, full_summary, category, full_response = await summarizer.process_paper(request)

    # Save to file (async to avoid blocking)
    output_file = await save_summary(pitch_output, full_summary, category, arxiv_url)

    # Save to database using PaperRecord model
    arxiv_id = request.arxiv_id or "unknown"
    full_response_json = orjson.dumps(full_response).decode("utf-8")

    record = PaperRecord(
        arxiv_id=arxiv_id,
        title=pitch_output.title,
        category=category,
        pitch=pitch_output.pitch,
        summary=full_summary,
        url=arxiv_url,
        full_response=full_response_json,
    )
    await save_to_db(record)

    logger.success(f"Summary saved to: {output_file}")
    logger.info(f"Title: {pitch_output.title}")
    logger.info(f"Category: {category}")
    logger.info(f"Pitch: {pitch_output.pitch}")
    logger.debug(f"Full analysis preview: {full_summary[:500]}...")


if __name__ == "__main__":
    main()
