import asyncio
import base64
import os
from pathlib import Path
from typing import Literal

import click
import requests
from tqdm import tqdm
from agents import Agent, ModelSettings, Runner, set_default_openai_client
from agents.tracing import set_tracing_disabled
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel, Field, field_validator

# NOTE: On newer Python versions, relying on dotenv's automatic discovery can be fragile in
# some execution modes. Prefer loading from the repo root deterministically.
_dotenv_path = Path(__file__).resolve().parent / ".env"
if _dotenv_path.exists():
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

# The openai-agents SDK enables a default tracing exporter that talks to OpenAI’s backend.
# When using a non-OpenAI proxy key (e.g., Shopify proxy), this can generate noisy 401s.
# Disable tracing by default for this CLI tool.
set_tracing_disabled(True)

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


class ProxyConfig(BaseModel):
    base_url: str
    api_key: str

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("API key is required")
        return v


class ReasoningConfig(BaseModel):
    effort: Literal["minimal", "low", "medium", "high"] = Field(default="high")
    summary: Literal["auto", "concise", "detailed"] = Field(default="detailed")


class TextConfig(BaseModel):
    verbosity: Literal["low", "medium", "high"] = Field(default="high")


class FileInput(BaseModel):
    role: str = Field(default="user")
    content: list


class SummarizationRequest(BaseModel):
    model: str = Field(default="gpt-5.2")
    pdf_path: str
    question: str
    instructions: str
    reasoning: ReasoningConfig = Field(default_factory=ReasoningConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    service_tier: str | None = Field(default=None)


class PitchOutput(BaseModel):
    title: str = Field(description="The exact title of the paper as it appears in the PDF")
    pitch: str = Field(
        description="A compelling 2-3 sentence pitch that captures the paper's core contribution and why it matters",
    )


class CategoryOutput(BaseModel):
    category: str = Field(description="Best matching category from the available list")
    reasoning: str = Field(description="Brief explanation for the categorization")


def encode_pdf(file_path: Path) -> str:
    return base64.b64encode(file_path.read_bytes()).decode("utf-8")


def get_openai_config() -> tuple[str, str | None]:
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("LLM API api key environment variable is required")

    return api_key, base_url


def download_arxiv_pdf(url: str, output_path: Path) -> Path:
    pdf_url = url.replace("/abs/", "/pdf/") if "/abs/" in url else url
    # Prefer redirect-free arXiv PDF URLs: https://arxiv.org/pdf/<id>
    # (The .pdf suffix often redirects to the non-suffixed URL.)
    pdf_url = pdf_url.removesuffix(".pdf")

    response = requests.get(pdf_url, stream=True, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    return output_path


def arxiv_id_from_url(url: str) -> str:
    # Supports URLs like:
    # - https://arxiv.org/abs/2312.07104
    # - https://arxiv.org/abs/2312.07104v2
    # - https://arxiv.org/pdf/2312.07104.pdf
    last = url.rstrip("/").split("/")[-1]
    return last.replace(".pdf", "")


class PaperProcessingError(RuntimeError):
    def __init__(self, url: str, original: Exception):
        super().__init__(f"{url}: {original}")
        self.url = url
        self.original = original


async def generate_full_summary(request: SummarizationRequest, arxiv_url: str | None = None) -> str:
    api_key, base_url = get_openai_config()

    # Increase timeout to 15 minutes for flex processing
    timeout = 900.0 if request.service_tier == "flex" else 600.0

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

    set_default_openai_client(client=client, use_for_tracing=False)

    # Build model_settings with service_tier in extra_args for flex processing
    extra_args = {}
    if request.service_tier:
        extra_args["service_tier"] = request.service_tier

    model_settings = ModelSettings(
        reasoning=Reasoning(
            effort=request.reasoning.effort,
            summary=request.reasoning.summary,
        ),
        verbosity=request.text.verbosity,
        extra_args=extra_args if extra_args else None,
    )

    agent = Agent(
        name="Paper Analyzer",
        instructions=request.instructions,
        model=request.model,
        model_settings=model_settings,
    )

    pdf_path = Path(request.pdf_path)

    # Use URL mode by default (avoids request size limits from embedding PDF bytes).
    if arxiv_url:
        # Use redirect-free arXiv PDF URL: https://arxiv.org/pdf/<id>
        pdf_url = arxiv_url.replace("/abs/", "/pdf/").removesuffix(".pdf")
        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": pdf_url,
                    },
                ],
            },
            {
                "role": "user",
                "content": request.question,
            },
        ]
    elif pdf_path.exists():
        # Local PDF mode: embed bytes directly.
        b64_file = encode_pdf(pdf_path)
        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_data": f"data:application/pdf;base64,{b64_file}",
                        "filename": pdf_path.name,
                    },
                ],
            },
            {
                "role": "user",
                "content": request.question,
            },
        ]
    else:
        raise ValueError("No PDF available: provide a local PDF path or an arXiv URL.")

    result = await Runner.run(agent, input_items)  # type: ignore
    return result.final_output


async def generate_pitch(
    full_summary: str, arxiv_url: str | None = None, pdf_path: Path | None = None, service_tier: str | None = None
) -> PitchOutput:
    api_key, base_url = get_openai_config()

    # Increase timeout for flex processing
    timeout = 900.0 if service_tier == "flex" else 600.0

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

    set_default_openai_client(client=client, use_for_tracing=False)

    # Build model_settings with service_tier in extra_args for flex processing
    extra_args = {}
    if service_tier:
        extra_args["service_tier"] = service_tier

    pitch_model_settings = ModelSettings(
        reasoning=Reasoning(
            effort="low",
            summary="auto",
        ),
        verbosity="low",
        extra_args=extra_args if extra_args else None,
    )

    pitch_agent = Agent(
        name="Pitch Generator",
        instructions="Extract the exact paper title and generate a compelling pitch.",
        model="gpt-5-mini-2025-08-07",
        output_type=PitchOutput,
        model_settings=pitch_model_settings,
    )

    if arxiv_url:
        pdf_url = arxiv_url.replace("/abs/", "/pdf/").removesuffix(".pdf")
        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": pdf_url,
                    },
                ],
            },
            {
                "role": "user",
                "content": f"""Extract the exact title from the PDF and generate a compelling 2-3 sentence pitch.

The pitch should capture:
1. The core contribution/innovation
2. Why it matters (impact/significance)

Paper Analysis (for context):
{full_summary[:2000]}...""",
            },
        ]
    elif pdf_path and pdf_path.exists():
        b64_file = encode_pdf(pdf_path)
        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_data": f"data:application/pdf;base64,{b64_file}",
                        "filename": pdf_path.name,
                    },
                ],
            },
            {
                "role": "user",
                "content": f"""Extract the exact title from the PDF and generate a compelling 2-3 sentence pitch.

The pitch should capture:
1. The core contribution/innovation
2. Why it matters (impact/significance)

Paper Analysis (for context):
{full_summary[:2000]}...""",
            },
        ]
    else:
        input_items = f"""Based on this analysis, extract a title and generate a compelling 2-3 sentence pitch:

{full_summary[:2000]}..."""

    result = await Runner.run(pitch_agent, input_items)  # type: ignore
    return result.final_output


async def categorize_paper(title: str, pitch: str, full_summary: str) -> str:
    """Categorize paper using GPT-5-mini based on title, pitch, and full summary."""
    api_key, base_url = get_openai_config()

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    response = await client.responses.create(
        model="gpt-5-mini-2025-08-07",
        reasoning=Reasoning(
            effort="low",
        ),
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

    # Parse category from response
    content = response.output_text
    if not content:
        return "stg"  # fallback if no response

    category = content.strip().lower()

    # Validate and return
    if category in CATEGORIES:
        return category

    # Fallback: find closest match
    for cat in CATEGORIES:
        if cat in category:
            return cat

    return "stg"  # fallback to staging


def save_summary(pitch_output: PitchOutput, full_summary: str, category: str, arxiv_url: str | None = None) -> Path:
    """Save paper summary to categorized directory with proper metadata header."""
    # Determine output directory
    category_dir = Path("docs") / category
    category_dir.mkdir(parents=True, exist_ok=True)

    # Extract ArXiv ID if available
    arxiv_link = ""
    if arxiv_url:
        arxiv_id = arxiv_url.split("/")[-1].replace(".pdf", "")
        arxiv_link = f"\n**ArXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n"

    # Normalize title for filename
    normalized_title = pitch_output.title.replace(" ", "").replace("/", "-").replace(":", "-")[:60]
    if arxiv_url:
        arxiv_id = arxiv_url.split("/")[-1].replace(".pdf", "")
        base_name = f"{arxiv_id}-{normalized_title}.md"
    else:
        base_name = f"{normalized_title}.md"

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

    output_file.write_text(formatted_output)
    return output_file


DEFAULT_QUESTION = "Analyze the attached paper and follow the instructions in the system prompt."


def load_prompt(prompt_path: str | None, default_file: str) -> str:
    path = Path(prompt_path) if prompt_path else Path(default_file)
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
        print(f"\n📚 Processing {len(urls)} papers...")
    else:
        print(f"\n📚 Processing {len(urls)} papers (concurrency={concurrency})...")
    if service_tier:
        print(f"⚡ Using {service_tier} processing (lower cost, slower responses)\n")
    else:
        print()

    if concurrency == 1:
        for i, url in tqdm(list(enumerate(urls, 1)), total=len(urls), desc="Papers"):
            print(f"\n{'=' * 80}")
            print(f"Processing paper {i}/{len(urls)}: {url}")
            print(f"{'=' * 80}\n")

            try:
                await async_main(model, url, None, question, instructions, service_tier)
            except Exception as e:
                print(f"\n❌ Error processing {url}: {e}")
                print("Continuing to next paper...\n")
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
            print(f"\n❌ Error processing {url}: {e}")

    print(f"\n🎉 Finished processing {len(urls)} papers!")


@click.command()
@click.option("--model", default="gpt-5.2", help="Model to use for summarization")
@click.option("--url", help="ArXiv URL to download and summarize")
@click.option("--urls", help="Comma-separated list of ArXiv URLs to process")
@click.option("--pdf", help="Local PDF path to summarize")
@click.option("--question", help="Optional user question prompt file (text). If omitted, uses a short default.")
@click.option("--instructions", help="System prompt file (text). Defaults to main_prompt.txt.")
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
    question: str | None,
    instructions: str | None,
    concurrency: int,
    flex: bool,
):
    service_tier = "flex" if flex else None

    if urls:
        # Process multiple URLs from comma-separated string
        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        asyncio.run(process_multiple_urls(model, url_list, question, instructions, service_tier, concurrency))
    else:
        # Process single URL or PDF
        asyncio.run(async_main(model, url, pdf, question, instructions, service_tier))


async def async_main(
    model: str,
    url: str | None,
    pdf: str | None,
    question: str | None,
    instructions: str | None,
    service_tier: str | None = None,
):
    if url:
        arxiv_url = url
        arxiv_id = arxiv_id_from_url(url)
        pdf_path = Path(f"unused-{arxiv_id}.pdf")
    elif pdf:
        pdf_path = Path(pdf)
        arxiv_url = None
    else:
        raise click.UsageError("Must provide either --url or --pdf")

    question_text = Path(question).read_text(encoding="utf-8").strip() if question else DEFAULT_QUESTION
    instructions_text = load_prompt(instructions, "main_prompt.txt")

    request = SummarizationRequest(
        model=model,
        pdf_path=str(pdf_path),
        question=question_text,
        instructions=instructions_text,
        service_tier=service_tier,
    )

    print("\n📊 Step 1/3: Generating full analysis with GPT-5.2 (high reasoning)...")
    full_summary = await generate_full_summary(request, arxiv_url)

    print("📝 Step 2/3: Extracting title and generating pitch with GPT-5-mini...")
    pitch_output = await generate_pitch(full_summary, arxiv_url=arxiv_url, pdf_path=pdf_path if pdf else None, service_tier=service_tier)

    print("🗂️  Step 3/3: Categorizing paper with GPT-5-mini...")
    category = await categorize_paper(pitch_output.title, pitch_output.pitch, full_summary)
    print(f"   → Category: {category}")

    output_file = save_summary(pitch_output, full_summary, category, arxiv_url)

    print(f"\n✅ Summary saved to: {output_file}")
    print("\n" + "=" * 80)
    print(f"\n📄 TITLE: {pitch_output.title}")
    print(f"📁 CATEGORY: {category}\n")
    print("🎯 PITCH\n")
    print(pitch_output.pitch)
    print("\n" + "=" * 80)
    print("\n📖 FULL ANALYSIS\n")
    print(full_summary[:500] + "...")


if __name__ == "__main__":
    main()
