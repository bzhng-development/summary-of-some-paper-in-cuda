"""System preamble and per-section prompt specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_SYSTEM_PREAMBLE_BASE = """\
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


def _build_system_preamble() -> str:
    """Assemble the system preamble with the required one-shot example.

    The example file is *not* optional: it is the one-shot golden sample that
    anchors output quality. If it's missing we'd rather crash at import time
    with a clear error than silently produce much weaker summaries.
    """
    example_path = Path(__file__).resolve().parent.parent / "examples" / "2408.03314_example.md"
    try:
        example_text = example_path.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Golden-sample example not found at {example_path}. "
            f"This file is the one-shot reference anchor for every summary; "
            f"without it output quality collapses silently. Restore it from "
            f"git (examples/2408.03314_example.md) before running."
        ) from exc
    if not example_text.strip():
        raise ValueError(
            f"Golden-sample example at {example_path} is empty. "
            f"Restore it from git before running."
        )
    return (
        _SYSTEM_PREAMBLE_BASE
        + "\n\n# Reference Example\n"
        + "Below is a complete example of a high-quality paper summary. "
        + "Match this level of depth, structure, and style.\n\n"
        + f"<example>\n{example_text}\n</example>"
    )


SYSTEM_PREAMBLE: str = _build_system_preamble()


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """Specification for a single section prompt."""

    number: int
    title: str
    prompt: str
    depends_on: tuple[int, ...] = field(default_factory=tuple)


SECTION_SPECS: tuple[SectionSpec, ...] = (
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
        depends_on=(1,),
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
        depends_on=(1, 2),
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
        depends_on=(1, 2, 3),
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
        depends_on=(1, 2, 3, 4),
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
        depends_on=(1, 2, 3, 4, 5),
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
        depends_on=(1, 2, 3, 4, 5, 6),
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
)


SECTION_SPECS_2PASS: tuple[SectionSpec, ...] = (
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
        depends_on=(1,),
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
)
