"""Multi-prompt paper summarization pipeline.

Instead of a single monolithic prompt, this module sends sequential per-section
prompts to the LLM.  Each follow-up prompt references what was already covered,
encouraging more verbose and detailed output per section.

Uses async OpenAI chat completions with a single client. Always local at :30000.

Usage (CLI):
    python multi_prompt.py --url https://arxiv.org/abs/2312.07104
    python multi_prompt.py --urls "url1,url2" --concurrency 3
    python multi_prompt.py --dyn --concurrency 4096
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pymupdf
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

from database import DB_PATH, _migrate_sync, get_all_ids_sync, save_paper_sync

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOCAL_BASE_URL = "http://localhost:30000/v1"
MODEL = "default"  # local server serves whatever is loaded

CATEGORIES = [
    "agents",
    "alignment",
    "architecture",
    "code",
    "context-optimization",
    "data",
    "diffusion",
    "distributed-training",
    "evaluation",
    "inference-optimization",
    "llm-systems",
    "low-precision",
    "moe",
    "multimodal",
    "pretraining",
    "prompting",
    "reasoning",
    "retrieval",
    "rl-training",
    "safety",
    "scaling-laws",
    "serving",
    "training-methods",
    "vision",
]
FALLBACK_CATEGORY = "uncategorized"

# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------


class PitchOutput(BaseModel):
    title: str = Field(description="The exact title of the paper as it appears in the PDF")
    pitch: str = Field(
        description="A compelling 2-3 sentence pitch that captures the paper's core contribution and why it matters",
    )


class CategoryOutput(BaseModel):
    category: str = Field(description="Best matching category from the available list")


# ---------------------------------------------------------------------------
# System preamble — shared across all section prompts, mirrors main_prompt.txt
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = """\
You are a technical educator who teaches research papers. Your goal is to make the reader \
fully understand this paper — as if they attended a detailed lecture on it. \
The reader should be able to explain the paper's contributions, methods, and results to \
someone else after reading your analysis.

Teach the material. Walk through the reasoning, not just the results. Explain WHY \
things work, not just WHAT the paper claims. Flag common misconceptions, subtle details, \
and non-obvious design choices. When a technique builds on prior work, briefly explain \
the prior approach so the reader understands what changed and why.

Analyze ONLY the provided paper content. Do not rely on external facts unless the user \
provides them in-context.

<constraints>
- Follow the requested structure exactly (section order, headings). Do not add extra sections.
- Anchor key claims to where they appear (e.g., "Figure 3", "Table 2", "Section 4.1").
- When details matter (numbers, thresholds, datasets, hyperparameters), quote or paraphrase \
  precisely rather than guessing.
- Never fabricate exact figures, hyperparameters, ablation results, or citations.
- If something is missing or unclear in the paper, say so explicitly.
</constraints>

# Hard Requirements
- Assume zero prior knowledge: the reader has NOT read the paper.
- Define technical terms selectively: if a term is uncommon, novel, or paper-specific, \
  define it on first use. Skip definitions for standard field terminology.
- Explain mechanisms and approaches — show HOW things work, not just what is claimed.
- Maintain logical flow: each section should build on previous context.
- Always include units and magnitudes when discussing scale.

# Output Format (GitHub-Flavored Markdown)
- Use markdown headers (##) for each section.
- Use bulleted lists for multi-point explanations.
- Use inline code formatting (`term`) for technical terms, variable names, or model names.
- Use block quotes when citing specific claims or results from the paper.
- Include specific figures, tables, or section references when discussing results.

## Math Formatting
- Use GitHub-compatible LaTeX for ALL mathematical expressions.
- Inline math: `$x^2$` (single dollar signs). Block math: ```$$...$$``` on their own lines.
- Do NOT use `\\(` `\\)` or `\\[` `\\]` delimiters — GitHub does not render them.

# Tone and Style
- Teach, don't summarize. Walk through concepts so the reader builds understanding.
- Use connected prose and logical sections — not massive bullet lists.
- Be direct and precise. Prioritize comprehension over brevity.
- Be critical but fair: highlight both strengths and weaknesses with evidence.
- Use present tense for describing the paper's content.
- No preamble or greeting — jump straight in.
"""

# Append one-shot example if available
_EXAMPLE_PATH = Path(__file__).parent / "examples" / "2408.03314_example.md"
if _EXAMPLE_PATH.exists():
    _SYSTEM_PREAMBLE += (
        "\n\n# Reference Example\n"
        "Below is a complete example of a high-quality paper summary. "
        "Match this level of depth, structure, and style.\n\n"
        f"<example>\n{_EXAMPLE_PATH.read_text()}\n</example>"
    )


# ---------------------------------------------------------------------------
# Section definitions — each prompt mirrors the original main_prompt.txt
# ---------------------------------------------------------------------------

@dataclass
class SectionSpec:
    """Specification for a single section prompt."""

    number: int
    title: str
    prompt: str
    depends_on: list[int] = field(default_factory=list)


SECTION_SPECS: list[SectionSpec] = [
    SectionSpec(
        number=1,
        title="Executive Summary",
        prompt=(
            "Produce **## 1. Executive Summary** only.\n\n"
            "State the paper's core contribution and primary significance in 2-3 sentences. "
            "Answer: What problem does this solve, and why does it matter?\n\n"
            "Be precise — include specific numbers, model names, or dataset names where relevant. "
            "Do NOT produce any other sections."
        ),
    ),
    SectionSpec(
        number=2,
        title="Context and Motivation",
        depends_on=[1],
        prompt=(
            "Produce **## 2. Context and Motivation** only.\n\n"
            "Cover ALL of the following:\n"
            "- What specific problem or gap does this paper address?\n"
            "- Why is this problem important (real-world impact, theoretical significance, or both)?\n"
            "- What prior approaches existed, and where do they fall short?\n"
            "- How does this paper position itself relative to existing work?\n\n"
            "Be thorough and detailed. The reader has NOT read the paper. "
            "Do NOT repeat the executive summary — build on it."
        ),
    ),
    SectionSpec(
        number=3,
        title="Technical Approach",
        depends_on=[1, 2],
        prompt=(
            "Produce **## 3. Technical Approach** only.\n\n"
            "NOTE: This should be the LONGEST and most detailed section. "
            "The reader has NOT read this paper and needs a complete standalone explanation.\n\n"
            "At the start, include these sub-sections with ### headings:\n\n"
            "### 3.1 Reader orientation (approachable technical breakdown)\n"
            "- One sentence on what the *system* is (or what is being built), in plain language.\n"
            "- One sentence on what problem it solves and the \"shape\" of the solution.\n\n"
            "### 3.2 Big-picture architecture (diagram in words)\n"
            "- A high-level \"box-and-arrows in words\" view of the major components.\n"
            "- Name each component and its responsibility; keep this overview short (you will expand below).\n\n"
            "### 3.3 Roadmap for the deep dive\n"
            "- 3–6 bullets that state the order you'll explain components and why that order helps understanding.\n\n"
            "### 3.4 Detailed, sentence-based technical breakdown\n"
            "- Treat this as a detailed technical breakdown of the system/mechanism in full sentences "
            "(not telegraphic fragments).\n"
            "- Even when using bullets, each bullet should be a complete sentence that explains a concrete "
            "mechanism, interface, or cause→effect relation.\n\n"
            "REQUIRED ELEMENTS:\n"
            "- Start with a one-sentence framing: what type of paper is this and the core idea.\n"
            "- Provide a \"system/data pipeline diagram in words\": describe major components, their "
            "inputs/outputs, and how information flows through them. Use an explicit \"what happens first, "
            "second, third\" narrative — no vague descriptions.\n"
            "- Include all key configurations, hyperparameters, and numbers mentioned in the paper.\n"
            "- If mathematical: present core equations with plain-language paraphrases BEFORE notation. "
            "Define all symbols.\n"
            "- Explain design choices: why this approach over alternatives?\n"
            "- Paraphrase technical terms in plain language before using them.\n\n"
            "Use GitHub-compatible LaTeX math ($...$ inline, $$...$$ block — no \\( \\) or \\[ \\] delimiters). Be exhaustive."
        ),
    ),
    SectionSpec(
        number=4,
        title="Key Insights and Innovations",
        depends_on=[1, 2, 3],
        prompt=(
            "Produce **## 4. Key Insights and Innovations** only.\n\n"
            "- Identify the 2-5 most novel contributions.\n"
            "- For each: explain what makes it different from prior work and why it's significant "
            "(performance gain, theoretical advance, new capability, etc.).\n"
            "- Distinguish between incremental improvements and fundamental innovations.\n\n"
            "Do NOT repeat technical details already covered in previous sections; "
            "reference them briefly and add new insight."
        ),
    ),
    SectionSpec(
        number=5,
        title="Experimental Analysis",
        depends_on=[1, 2, 3, 4],
        prompt=(
            "Produce **## 5. Experimental Analysis** only.\n\n"
            "- Describe evaluation methodology: datasets, metrics, baselines, experimental setup.\n"
            "- Summarize main quantitative results with SPECIFIC NUMBERS and comparisons.\n"
            "- Assess whether the experiments convincingly support the paper's claims.\n"
            "- Note any ablation studies, failure cases, or robustness checks.\n"
            "- If results are mixed or conditional, explain the conditions and trade-offs.\n\n"
            "Cite specific tables and figures. Be thorough with numbers."
        ),
    ),
    SectionSpec(
        number=6,
        title="Limitations and Trade-offs",
        depends_on=[1, 2, 3, 4, 5],
        prompt=(
            "Produce **## 6. Limitations and Trade-offs** only.\n\n"
            "- What assumptions does the approach rely on?\n"
            "- What scenarios, edge cases, or problem settings are NOT addressed?\n"
            "- Are there computational, data, or scalability constraints?\n"
            "- What weaknesses or open questions remain?\n\n"
            "Be critical but fair. Ground your points in evidence from the paper."
        ),
    ),
    SectionSpec(
        number=7,
        title="Implications and Future Directions",
        depends_on=[1, 2, 3, 4, 5, 6],
        prompt=(
            "Produce **## 7. Implications and Future Directions** only.\n\n"
            "- How does this work change the landscape of the field?\n"
            "- What follow-up research does it enable or suggest?\n"
            "- What are the practical applications or downstream use cases?\n"
            "- Repro/Integration Guidance: When applicable, briefly explain practical context—e.g., "
            "when to prefer this method over alternatives.\n\n"
            "Be concrete and forward-looking."
        ),
    ),
]

# ---------------------------------------------------------------------------
# 2-pass section specs (default) — for capable models that can handle
# sections 1-5 in a single prompt and 6-7 in a second pass.
# ---------------------------------------------------------------------------

SECTION_SPECS_2PASS: list[SectionSpec] = [
    SectionSpec(
        number=1,
        title="Core Analysis (Sections 1-5)",
        prompt=(
            "Produce sections 1 through 5 of a comprehensive paper analysis.\n\n"
            "## 1. Executive Summary\n"
            "State the paper's core contribution and primary significance in 2-3 sentences. "
            "Answer: What problem does this solve, and why does it matter? "
            "Be precise — include specific numbers, model names, or dataset names where relevant.\n\n"
            "## 2. Context and Motivation\n"
            "Cover ALL of the following:\n"
            "- What specific problem or gap does this paper address?\n"
            "- Why is this problem important (real-world impact, theoretical significance, or both)?\n"
            "- What prior approaches existed, and where do they fall short?\n"
            "- How does this paper position itself relative to existing work?\n\n"
            "Be thorough and detailed. The reader has NOT read the paper.\n\n"
            "## 3. Technical Approach\n"
            "NOTE: This should be the LONGEST and most detailed section. "
            "The reader has NOT read this paper and needs a complete standalone explanation.\n\n"
            "Include these sub-sections with ### headings:\n\n"
            "### 3.1 Reader orientation\n"
            "- One sentence on what the *system* is (or what is being built), in plain language.\n"
            "- One sentence on what problem it solves and the \"shape\" of the solution.\n\n"
            "### 3.2 Big-picture architecture (diagram in words)\n"
            "- A high-level \"box-and-arrows in words\" view of the major components.\n"
            "- Name each component and its responsibility; keep this overview short (you will expand below).\n\n"
            "### 3.3 Roadmap for the deep dive\n"
            "- 3-6 bullets that state the order you'll explain components and why that order helps understanding.\n\n"
            "### 3.4 Detailed, sentence-based technical breakdown\n"
            "- Treat this as a detailed technical breakdown of the system/mechanism in full sentences.\n"
            "- Even when using bullets, each bullet should be a complete sentence that explains a concrete "
            "mechanism, interface, or cause->effect relation.\n\n"
            "REQUIRED ELEMENTS:\n"
            "- Start with a one-sentence framing: what type of paper is this and the core idea.\n"
            "- Provide a \"system/data pipeline diagram in words\".\n"
            "- Include all key configurations, hyperparameters, and numbers mentioned in the paper.\n"
            "- If mathematical: present core equations with plain-language paraphrases BEFORE notation. "
            "Define all symbols.\n"
            "- Explain design choices: why this approach over alternatives?\n"
            "- Paraphrase technical terms in plain language before using them.\n\n"
            "Use GitHub-compatible LaTeX math ($...$ inline, $$...$$ block — no \\( \\) or \\[ \\] delimiters). Be exhaustive.\n\n"
            "## 4. Key Insights and Innovations\n"
            "- Identify the 2-5 most novel contributions.\n"
            "- For each: explain what makes it different from prior work and why it's significant.\n"
            "- Distinguish between incremental improvements and fundamental innovations.\n\n"
            "## 5. Experimental Analysis\n"
            "- Describe evaluation methodology: datasets, metrics, baselines, experimental setup.\n"
            "- Summarize main quantitative results with SPECIFIC NUMBERS and comparisons.\n"
            "- Assess whether the experiments convincingly support the paper's claims.\n"
            "- Note any ablation studies, failure cases, or robustness checks.\n"
            "- If results are mixed or conditional, explain the conditions and trade-offs.\n"
            "- Cite specific tables and figures. Be thorough with numbers."
        ),
    ),
    SectionSpec(
        number=2,
        title="Critical Assessment (Sections 6-7)",
        depends_on=[1],
        prompt=(
            "Produce sections 6 and 7 of the paper analysis.\n\n"
            "## 6. Limitations and Trade-offs\n"
            "- What assumptions does the approach rely on?\n"
            "- What scenarios, edge cases, or problem settings are NOT addressed?\n"
            "- Are there computational, data, or scalability constraints?\n"
            "- What weaknesses or open questions remain?\n\n"
            "Be critical but fair. Ground your points in evidence from the paper.\n\n"
            "## 7. Implications and Future Directions\n"
            "- How does this work change the landscape of the field?\n"
            "- What follow-up research does it enable or suggest?\n"
            "- What are the practical applications or downstream use cases?\n"
            "- Repro/Integration Guidance: When applicable, briefly explain practical context—e.g., "
            "when to prefer this method over alternatives.\n\n"
            "Be concrete and forward-looking."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def arxiv_id_from_url(url: str) -> str:
    """Extract ArXiv ID from any ArXiv URL format."""
    path = urlparse(url).path.rstrip("/")
    last_segment = path.split("/")[-1]
    return last_segment.replace(".pdf", "")


def arxiv_url_to_pdf_url(url: str) -> str:
    return url.replace("/abs/", "/pdf/").removesuffix(".pdf")


def _download_pdf_bytes(pdf_url: str) -> bytes:
    """Download a PDF and return raw bytes (for native PDF mode)."""
    logger.info(f"Downloading PDF: {pdf_url}")
    resp = httpx.get(pdf_url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    logger.info(f"Downloaded {len(resp.content) / 1024:.0f} KB")
    return resp.content


def download_and_extract_text(pdf_url: str) -> str:
    """Download a PDF from a URL and extract its full text."""
    logger.info(f"Downloading PDF: {pdf_url}")
    resp = httpx.get(pdf_url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    pdf_bytes = resp.content
    logger.info(f"Downloaded {len(pdf_bytes) / 1024:.0f} KB, extracting text...")

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()

    full_text = "\n\n".join(pages)
    logger.info(f"Extracted {len(full_text)} chars from {len(pages)} pages")
    return full_text


def _parse_retry_delay(exc: Exception) -> float | None:
    """Try to extract a retry delay (seconds) from a 429 error message."""
    msg = str(exc)
    m = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 1  # add 1s buffer
    if "429" in msg:
        return 30.0  # default backoff for rate limits
    return None


async def retry_async(fn, *, max_attempts: int = 10, label: str = ""):
    """Retry an async callable with exponential backoff and 429-aware delays."""
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait = _parse_retry_delay(e) or min(2 ** attempt, 60)
            logger.warning(f"{label}attempt {attempt + 1} failed: {e!r}, retrying in {wait:.0f}s")
            await asyncio.sleep(wait)


def normalize_title_for_filename(title: str, max_length: int = 80) -> str:
    title = "".join(ch for ch in title if ord(ch) >= 32)
    title = re.sub(r"\s+", " ", title).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-")[:max_length]


# ---------------------------------------------------------------------------
# Multi-prompt summarizer — plain chat completions, no SDK
# ---------------------------------------------------------------------------


def _raise_fd_limit() -> None:
    """Try to raise the process soft fd limit to the hard limit (macOS defaults to 256)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
    if soft < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info(f"Raised fd limit: {soft} → {target}")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not raise fd limit from {soft}: {e}")


def make_client(
    base_url: str = LOCAL_BASE_URL,
    api_key: str = "not-needed",
    timeout: float = 1500.0,
    extra_body: dict | None = None,
) -> tuple[AsyncOpenAI, dict | None]:
    """Create a single AsyncOpenAI client. Returns (client, extra_body)."""
    _raise_fd_limit()
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )
    if extra_body is None and base_url == LOCAL_BASE_URL:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    logger.info(f"Initialized async client at {base_url}")
    return client, extra_body


class MultiPromptSummarizer:
    """Sends one chat completion per section, accumulating context.

    Uses async OpenAI clients with a shared pool for high concurrency.
    When native_pdf=True, sends the raw PDF as a file attachment instead of
    extracting text (used for OpenRouter/Gemini which handle PDFs natively).
    """

    def __init__(
        self,
        model: str = MODEL,
        sections: list[SectionSpec] | None = None,
        client: AsyncOpenAI | None = None,
        extra_body: dict | None = None,
        native_pdf: bool = False,
    ):
        self.model = model
        self.sections = sections or SECTION_SPECS
        if client is None:
            client, extra_body = make_client()
            self._owns_client = True
        else:
            self._owns_client = False
        self.client = client
        self.extra_body = extra_body
        self.native_pdf = native_pdf

    def _build_context_block(
        self,
        completed: dict[int, str],
        depends_on: list[int],
    ) -> str:
        if not depends_on:
            return ""
        parts = [completed[n] for n in depends_on if n in completed]
        if not parts:
            return ""
        joined = "\n\n---\n\n".join(parts)
        return (
            "<prior_sections>\n"
            "The following sections have already been written for this paper. "
            "Do NOT repeat their content — reference it where needed and expand with new detail.\n\n"
            f"{joined}\n"
            "</prior_sections>\n\n"
        )

    async def _chat(self, system: str, user_content: str | list) -> str:
        """Async chat completion, returns content string."""
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def _chat_json(self, system: str, user_text: str, schema_name: str, schema: dict) -> str:
        """Async chat completion with json_schema response format."""
        kwargs: dict = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                },
            },
        )
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = await self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def generate_full_summary(
        self, paper_text_or_data_url: str, arxiv_id: str = "", pbar: tqdm | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Generate verbose summary by prompting each section sequentially.

        paper_text_or_data_url is either extracted text (local mode) or a
        base64 data URL for native PDF (gemini mode).
        """
        completed: dict[int, str] = {}
        all_outputs: dict[str, str] = {}
        label = f"[{arxiv_id}] " if arxiv_id else ""

        for spec in self.sections:
            if pbar is not None:
                pbar.set_postfix_str(f"s{spec.number}/{len(self.sections)} {spec.title}")
            context_block = self._build_context_block(completed, spec.depends_on)

            if self.native_pdf:
                user_content: str | list = [
                    {
                        "type": "file",
                        "file": {
                            "filename": "paper.pdf",
                            "file_data": paper_text_or_data_url,
                        },
                    },
                    {"type": "text", "text": f"{context_block}{spec.prompt}"},
                ]
            else:
                user_content = (
                    f"<paper>\n{paper_text_or_data_url}\n</paper>\n\n"
                    f"{context_block}{spec.prompt}"
                )

            output = await retry_async(
                lambda: self._chat(_SYSTEM_PREAMBLE, user_content),
                label=f"{label}Section {spec.number} ",
            )

            completed[spec.number] = output
            all_outputs[f"section_{spec.number}"] = output
            tqdm.write(f"\n{'='*60}\n{label}Section {spec.number}: {spec.title}\n{'='*60}\n{output}\n")

        combined = "\n\n".join(
            completed[s.number] for s in self.sections if s.number in completed
        )
        return combined, all_outputs

    async def generate_pitch(self, full_summary: str, paper_text: str) -> PitchOutput:
        """Extract title and generate pitch using structured output."""
        system = (
            "Extract the exact paper title and generate a compelling 2-3 sentence pitch. "
            "The pitch should capture the core contribution and why it matters."
        )
        user_msg = (
            f"<paper>\n{paper_text[:5000]}\n</paper>\n\n"
            f"Paper Analysis (for context):\n{full_summary[:3000]}..."
        )

        async def _do():
            raw = await self._chat_json(system, user_msg, "pitch_output", PitchOutput.model_json_schema())
            return PitchOutput.model_validate_json(raw)
        return await retry_async(_do, label="Pitch ")

    async def categorize_paper(self, title: str, pitch: str, full_summary: str) -> str:
        """Categorize paper using structured output."""
        system = (
            f"Categorize the paper into one of these categories: {', '.join(CATEGORIES)}. "
            "Respond with ONLY the category name in the JSON."
        )
        user_msg = f"Title: {title}\n\nPitch: {pitch}\n\nFull Summary:\n{full_summary}"

        async def _do():
            raw = await self._chat_json(system, user_msg, "category_output", CategoryOutput.model_json_schema())
            result = CategoryOutput.model_validate_json(raw)
            category = result.category.strip().lower()
            if category in CATEGORIES:
                return category
            for cat in CATEGORIES:
                if cat in category:
                    logger.debug(f"Partial category match: '{category}' -> '{cat}'")
                    return cat
            logger.warning(f"Invalid category '{category}', using fallback")
            return FALLBACK_CATEGORY
        return await retry_async(_do, label="Categorize ")

    async def process_paper(self, arxiv_url: str, papers_pbar: tqdm | None = None, jsonl_path: str | None = None, paper_id: str | None = None) -> dict:
        """Full pipeline: download PDF → extract text → multi-prompt summary → pitch → categorize → save.

        If jsonl_path is set, appends result to JSONL file instead of saving to DB/markdown.
        paper_id overrides the ID extracted from the URL (used for ext: papers).
        Returns dict with title, category, pitch, summary, output_file.
        """
        is_ext = paper_id is not None and paper_id.startswith("ext:")
        if is_ext:
            pdf_url = arxiv_url  # ext: papers store the direct PDF URL
            arxiv_id = paper_id
        else:
            pdf_url = arxiv_url_to_pdf_url(arxiv_url)
            arxiv_id = arxiv_id_from_url(arxiv_url)

        if self.native_pdf:
            pdf_bytes = await asyncio.to_thread(_download_pdf_bytes, pdf_url)
            paper_input = f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"
        else:
            paper_input = await asyncio.to_thread(download_and_extract_text, pdf_url)

        n = len(self.sections)
        total_steps = n + 2  # sections + pitch + categorize

        pbar = tqdm(total=total_steps, desc=f"[{arxiv_id}]", leave=False, position=1)

        full_summary, section_outputs = await self.generate_full_summary(paper_input, arxiv_id=arxiv_id, pbar=pbar)
        pbar.update(n)

        pbar.set_postfix_str("pitch")
        pitch_output = await self.generate_pitch(full_summary, paper_input)
        tqdm.write(f"\n[{arxiv_id}] Title: {pitch_output.title}")
        tqdm.write(f"[{arxiv_id}] Pitch: {pitch_output.pitch}")
        pbar.update(1)

        pbar.set_postfix_str("categorize")
        category = await self.categorize_paper(pitch_output.title, pitch_output.pitch, full_summary)
        tqdm.write(f"[{arxiv_id}] Category: {category}")
        pbar.update(1)
        pbar.close()

        result = {
            "arxiv_id": arxiv_id,
            "title": pitch_output.title,
            "category": category,
            "pitch": pitch_output.pitch,
            "summary": full_summary,
            "url": arxiv_url,
        }

        if jsonl_path:
            await asyncio.to_thread(append_jsonl_sync, result, jsonl_path)
            tqdm.write(f"[{arxiv_id}] Appended to {jsonl_path}")
        else:
            # File/DB I/O in threads to avoid blocking
            output_file = await asyncio.to_thread(
                save_summary_sync, pitch_output, full_summary, category, arxiv_url, paper_id,
            )
            await asyncio.to_thread(
                save_paper_sync,
                arxiv_id,
                title=pitch_output.title,
                category=category,
                pitch=pitch_output.pitch,
                summary=full_summary,
                url=arxiv_url,
            )
            result["output_file"] = str(output_file)
            tqdm.write(f"[{arxiv_id}] Saved to: {output_file}")

        if papers_pbar is not None:
            papers_pbar.update(1)

        return result


# ---------------------------------------------------------------------------
# Sync file/DB helpers (no async, no dependency on main.py)
# ---------------------------------------------------------------------------


def save_summary_sync(
    pitch_output: PitchOutput,
    full_summary: str,
    category: str,
    arxiv_url: str | None = None,
    paper_id: str | None = None,
) -> Path:
    """Save paper summary to categorized directory."""
    category_dir = Path("docs") / category
    category_dir.mkdir(parents=True, exist_ok=True)

    if paper_id and paper_id.startswith("ext:"):
        arxiv_id = None
        file_id = re.sub(r"[^A-Za-z0-9._-]+", "-", paper_id.removeprefix("ext:")).strip("-")
        arxiv_link = f"\n**URL:** [{arxiv_url}]({arxiv_url})\n" if arxiv_url else ""
    else:
        arxiv_id = arxiv_id_from_url(arxiv_url) if arxiv_url else None
        file_id = arxiv_id
        arxiv_link = f"\n**ArXiv:** [{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n" if arxiv_id else ""

    normalized_title = normalize_title_for_filename(pitch_output.title)
    if file_id:
        base_name = f"{file_id}-{normalized_title}.md" if normalized_title else f"{file_id}.md"
    else:
        base_name = f"{normalized_title}.md" if normalized_title else "paper.md"

    output_file = category_dir / base_name
    if output_file.exists():
        stem, suffix = output_file.stem, output_file.suffix
        for k in range(2, 1000):
            candidate = category_dir / f"{stem}-{k}{suffix}"
            if not candidate.exists():
                output_file = candidate
                break

    formatted_output = f"""# {pitch_output.title}
{arxiv_link}
## Pitch

{pitch_output.pitch}

---

{full_summary}
"""
    output_file.write_text(formatted_output)
    return output_file




def append_jsonl_sync(record: dict, output_path: str) -> None:
    """Append a paper record as one JSON line to a JSONL file."""
    with open(output_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"Paper {record.get('arxiv_id', '?')} appended to {output_path}")


def paper_exists_sync(arxiv_id: str, min_summary_len: int = 5000) -> bool:
    """Check if a paper has a real (pipeline-generated) summary, not just an abstract stub."""
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT length(summary) FROM papers WHERE id = ?", (arxiv_id,)).fetchone()
    con.close()
    return row is not None and row[0] is not None and row[0] >= min_summary_len


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_stubs_sync() -> list[str]:
    """Get paper IDs in DB that have no summary (stubs awaiting processing)."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id FROM papers WHERE (summary IS NULL OR summary = '') AND id NOT LIKE 'ext:%'"
    ).fetchall()
    con.close()
    return [row[0] for row in rows]


def get_interested_stubs_sync() -> list[tuple[str, str | None]]:
    """Get (id, url) of interested papers that need (re-)summarization."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, url FROM papers WHERE interested = 1 AND (summary IS NULL OR summary = '')"
    ).fetchall()
    con.close()
    return [(row[0], row[1]) for row in rows]


async def run_dynamic(
    model: str,
    interval: int = 30,
    concurrency: int = 4096,
    sections: list[SectionSpec] | None = None,
):
    """Continuously poll DB for stubs and process them concurrently.

    Ctrl-C to stop.
    """
    client, extra_body = make_client()
    logger.info(
        f"Dynamic mode: polling DB every {interval}s, concurrency={concurrency} (Ctrl-C to stop)"
    )
    sem = asyncio.Semaphore(concurrency)
    processed_total = 0
    in_flight: set[str] = set()
    tasks: set[asyncio.Task] = set()

    async def _run_one(aid: str) -> None:
        nonlocal processed_total
        async with sem:
            summarizer = MultiPromptSummarizer(model=model, client=client, extra_body=extra_body, sections=sections)
            arxiv_url = f"https://arxiv.org/abs/{aid}"
            try:
                await summarizer.process_paper(arxiv_url, papers_pbar=None)
                processed_total += 1
                logger.success(f"Completed {aid} ({processed_total} total)")
            except Exception as e:
                logger.error(f"Failed {aid}: {e}")
            finally:
                in_flight.discard(aid)

    try:
        while True:
            stubs = await asyncio.to_thread(get_stubs_sync)
            new_stubs = [aid for aid in stubs if aid not in in_flight]

            # Clean up finished tasks
            done = {t for t in tasks if t.done()}
            tasks -= done

            if not new_stubs:
                if not in_flight:
                    logger.debug(f"No stubs found, sleeping {interval}s...")
                await asyncio.sleep(interval)
                continue

            logger.info(f"Submitting {len(new_stubs)} new stubs ({len(in_flight)} already in flight)")
            in_flight.update(new_stubs)

            for aid in new_stubs:
                task = asyncio.create_task(_run_one(aid))
                tasks.add(task)

    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info(f"\nStopping... waiting for {len(tasks)} in-flight tasks")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Stopped. Processed {processed_total} papers total.")
    finally:
        await client.close()


async def process_multiple(
    model: str,
    urls: list[str],
    concurrency: int = 4096,
    gemini: bool = False,
    jsonl_path: str | None = None,
    sections: list[SectionSpec] | None = None,
    paper_ids: list[str] | None = None,
):
    """Process multiple papers concurrently, skipping those already done."""
    # Build set of already-done IDs from DB + JSONL
    done_ids: set[str] = set()
    if jsonl_path and Path(jsonl_path).exists():
        for line in Path(jsonl_path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    done_ids.add(json.loads(line)["arxiv_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            logger.info(f"Found {len(done_ids)} papers already in {jsonl_path}")

    filtered = []
    filtered_ids = []
    for i, u in enumerate(urls):
        pid = paper_ids[i] if paper_ids else arxiv_id_from_url(u)
        if pid in done_ids:
            logger.info(f"Skipping {pid} (already in JSONL)")
        elif paper_exists_sync(pid):
            logger.info(f"Skipping {pid} (already in DB)")
        else:
            filtered.append(u)
            filtered_ids.append(pid)

    if len(filtered) < len(urls):
        logger.info(f"Skipped {len(urls) - len(filtered)} papers already processed")

    if not filtered:
        logger.success("All papers already processed!")
        return

    if gemini:
        from dotenv import load_dotenv
        load_dotenv()
        client, extra_body = make_client(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            extra_body={"provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}},
        )
        native_pdf = True
    else:
        client, extra_body = make_client()
        native_pdf = False

    logger.info(f"Processing {len(filtered)} papers (concurrency={concurrency}, native_pdf={native_pdf})...")
    sem = asyncio.Semaphore(concurrency)
    errors: list[tuple[str, Exception]] = []
    papers_pbar = tqdm(total=len(filtered), desc="Papers", position=0)

    async def _run_one(url: str, pid: str | None = None) -> None:
        async with sem:
            summarizer = MultiPromptSummarizer(model=model, client=client, extra_body=extra_body, native_pdf=native_pdf, sections=sections)
            await summarizer.process_paper(url, papers_pbar=papers_pbar, jsonl_path=jsonl_path, paper_id=pid if pid and pid.startswith("ext:") else None)

    tasks = []
    for i, url in enumerate(filtered):
        pid = filtered_ids[i] if filtered_ids else None
        tasks.append(asyncio.create_task(_run_one(url, pid)))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for url, result in zip(filtered, results):
        if isinstance(result, Exception):
            logger.error(f"Error processing {url}: {result}")
            errors.append((url, result))
            papers_pbar.update(1)

    papers_pbar.close()
    await client.close()

    if errors:
        logger.warning(f"{len(errors)} papers failed")
    logger.success(f"Finished processing {len(filtered)} papers!")


def _make_client_from_args(args) -> tuple[AsyncOpenAI, dict | None, bool]:
    """Create client from CLI args. Returns (client, extra_body, native_pdf)."""
    if args.gemini:
        from dotenv import load_dotenv
        load_dotenv()
        client, extra_body = make_client(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            extra_body={"provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}},
        )
        return client, extra_body, True
    client, extra_body = make_client()
    return client, extra_body, False


def main():
    parser = argparse.ArgumentParser(description="Multi-prompt paper summarizer")
    parser.add_argument("--model", default=None, help="Model name (auto-set for --gemini)")
    parser.add_argument("--url", default=None, help="ArXiv URL to summarize")
    parser.add_argument("--urls", default=None, help="Comma-separated list of ArXiv URLs")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent papers (default: 5)")
    parser.add_argument("--dyn", action="store_true", help="Continuously poll DB for stubs and process them")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds for --dyn (default: 30)")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini 3 Flash via OpenRouter/Google AI Studio (free tier)")
    parser.add_argument("--backfill", action="store_true", help="Re-summarize all interested papers missing a summary")
    parser.add_argument("--jsonl", default=None, help="Write results to JSONL file instead of DB (for remote runs)")
    parser.add_argument("--many-pass", action="store_true", help="Use 7-section sequential pipeline (old behavior, for smaller models)")
    args = parser.parse_args()

    if args.model is None:
        args.model = "google/gemini-3-flash-preview" if args.gemini else MODEL

    sections = SECTION_SPECS if args.many_pass else SECTION_SPECS_2PASS

    _migrate_sync()

    if args.backfill:
        stubs = get_interested_stubs_sync()
        if not stubs:
            logger.success("No interested papers need summarization!")
            return
        urls = []
        for aid, url in stubs:
            if aid.startswith("ext:"):
                urls.append((aid, url or ""))
            else:
                urls.append((aid, f"https://arxiv.org/abs/{aid}"))
        logger.info(f"Backfill: {len(urls)} interested papers to summarize")
        asyncio.run(process_multiple(args.model, [u for _, u in urls], args.concurrency, gemini=args.gemini, jsonl_path=args.jsonl, sections=sections, paper_ids=[aid for aid, _ in urls]))
    elif args.dyn:
        asyncio.run(run_dynamic(args.model, args.interval, args.concurrency, sections=sections))
    elif args.urls:
        url_list = [u.strip() for u in args.urls.split(",") if u.strip()]
        asyncio.run(process_multiple(args.model, url_list, args.concurrency, gemini=args.gemini, jsonl_path=args.jsonl, sections=sections))
    elif args.url:
        async def _single():
            client, extra_body, native_pdf = _make_client_from_args(args)
            summarizer = MultiPromptSummarizer(model=args.model, client=client, extra_body=extra_body, native_pdf=native_pdf, sections=sections)
            await summarizer.process_paper(args.url, jsonl_path=args.jsonl)
            await client.close()
        asyncio.run(_single())
    else:
        parser.error("Must provide either --url, --urls, --dyn, or --backfill")


if __name__ == "__main__":
    main()
