"""The ``MultiPromptSummarizer`` — pure LLM orchestration, no I/O side effects."""

from __future__ import annotations

from typing import Any

from loguru import logger
from tqdm import tqdm

from .config import CATEGORIES, DEFAULT_MODEL, FALLBACK_CATEGORY
from .llm import LLMClient, retry_async
from .prompts import SECTION_SPECS, SYSTEM_PREAMBLE, SectionSpec
from .schemas import CategoryOutput, PitchOutput


class MultiPromptSummarizer:
    """Drive per-section chat completions against an :class:`LLMClient`.

    The summarizer is strictly concerned with *LLM orchestration*:
    building prompts, sending chat completions, parsing structured outputs.
    Downloading PDFs, writing markdown, and persisting to the DB all live in
    :mod:`multi_prompt_pkg.pipeline`.
    """

    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str = DEFAULT_MODEL,
        sections: tuple[SectionSpec, ...] | list[SectionSpec] | None = None,
    ) -> None:
        self.llm = llm
        self.model = model
        self.sections: tuple[SectionSpec, ...] = (
            tuple(sections) if sections is not None else SECTION_SPECS
        )

    # ------------------------------------------------------------------
    # Low-level chat helpers
    # ------------------------------------------------------------------

    async def _chat(self, system: str, user_content: str | list[Any]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        if self.llm.extra_body:
            kwargs["extra_body"] = self.llm.extra_body
        resp = await self.llm.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    async def _chat_json(
        self,
        system: str,
        user_text: str,
        *,
        schema_name: str,
        schema: dict[str, Any],
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema},
            },
        }
        if self.llm.extra_body:
            kwargs["extra_body"] = self.llm.extra_body
        resp = await self.llm.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_block(
        completed: dict[int, str],
        depends_on: tuple[int, ...],
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

    def _build_user_content(
        self,
        paper_input: str,
        context_block: str,
        section_prompt: str,
    ) -> str | list[dict[str, Any]]:
        if self.llm.native_pdf:
            return [
                {
                    "type": "file",
                    "file": {
                        "filename": "paper.pdf",
                        "file_data": paper_input,
                    },
                },
                {"type": "text", "text": f"{context_block}{section_prompt}"},
            ]
        return (
            f"<paper>\n{paper_input}\n</paper>\n\n"
            f"{context_block}{section_prompt}"
        )

    # ------------------------------------------------------------------
    # Section-by-section generation
    # ------------------------------------------------------------------

    async def generate_full_summary(
        self,
        paper_input: str,
        *,
        arxiv_id: str = "",
        pbar: tqdm | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Walk every section spec in order, accumulating completions.

        ``paper_input`` is either extracted text (local mode) or a
        base64-encoded ``data:application/pdf;base64,...`` URL (native PDF).
        """
        completed: dict[int, str] = {}
        all_outputs: dict[str, str] = {}
        label = f"[{arxiv_id}] " if arxiv_id else ""

        for spec in self.sections:
            if pbar is not None:
                pbar.set_postfix_str(f"s{spec.number}/{len(self.sections)} {spec.title}")
            context_block = self._build_context_block(completed, spec.depends_on)
            user_content = self._build_user_content(
                paper_input, context_block, spec.prompt
            )

            async def _call() -> str:
                return await self._chat(SYSTEM_PREAMBLE, user_content)

            output = await retry_async(
                _call, label=f"{label}Section {spec.number} "
            )
            completed[spec.number] = output
            all_outputs[f"section_{spec.number}"] = output
            tqdm.write(
                f"\n{'=' * 60}\n{label}Section {spec.number}: {spec.title}\n"
                f"{'=' * 60}\n{output}\n"
            )

        combined = "\n\n".join(
            completed[s.number] for s in self.sections if s.number in completed
        )
        return combined, all_outputs

    # ------------------------------------------------------------------
    # Structured post-processing: title/pitch and categorization
    # ------------------------------------------------------------------

    async def generate_pitch(
        self, full_summary: str, paper_text: str
    ) -> PitchOutput:
        """Extract an exact title plus a short pitch via structured output."""
        system = (
            "Extract the exact paper title and generate a compelling 2-3 sentence pitch. "
            "The pitch should capture the core contribution and why it matters."
        )
        user_msg = (
            f"<paper>\n{paper_text[:5000]}\n</paper>\n\n"
            f"Paper Analysis (for context):\n{full_summary[:3000]}..."
        )

        async def _call() -> PitchOutput:
            raw = await self._chat_json(
                system,
                user_msg,
                schema_name="pitch_output",
                schema=PitchOutput.model_json_schema(),
            )
            return PitchOutput.model_validate_json(raw)

        return await retry_async(_call, label="Pitch ")

    async def categorize_paper(
        self, title: str, pitch: str, full_summary: str
    ) -> str:
        """Assign the paper one of :data:`CATEGORIES` (or ``FALLBACK_CATEGORY``)."""
        system = (
            f"Categorize the paper into one of these categories: {', '.join(CATEGORIES)}. "
            "Respond with ONLY the category name in the JSON."
        )
        user_msg = f"Title: {title}\n\nPitch: {pitch}\n\nFull Summary:\n{full_summary}"

        async def _call() -> str:
            raw = await self._chat_json(
                system,
                user_msg,
                schema_name="category_output",
                schema=CategoryOutput.model_json_schema(),
            )
            result = CategoryOutput.model_validate_json(raw)
            category = result.category.strip().lower()
            if category in CATEGORIES:
                return category
            for cat in CATEGORIES:
                if cat in category:
                    logger.debug("Partial category match: {!r} -> {!r}", category, cat)
                    return cat
            logger.warning("Invalid category {!r}, using fallback", category)
            return FALLBACK_CATEGORY

        return await retry_async(_call, label="Categorize ")
