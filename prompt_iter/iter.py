"""Single-shot section-prompt iteration driver against vLLM DeepSeek-V4-Pro.

Usage:
    uv run python prompt_iter/iter.py --section 1 --version v1 \
        --prompt-file prompt_iter/prompts/s1_v1.txt \
        [--prior s1=v3,s2=v2]
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
DEFAULT_PAPER_TXT = ROOT / "paper_2408.03314.txt"
REFERENCE_MD = ROOT / "reference_2408.03314.md"
DEFAULT_GEN_DIR = ROOT / "generations"
PROMPTS_DIR = ROOT / "prompts"

# Pulled verbatim from multi_prompt_pkg/prompts.py:_SYSTEM_PREAMBLE_BASE
SYSTEM_PREAMBLE_BASE = """\
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


def build_system_preamble() -> str:
    example_text = REFERENCE_MD.read_text()
    return (
        SYSTEM_PREAMBLE_BASE
        + "\n\n# Reference Example\n"
        + "Below is a complete example of a high-quality paper summary. "
        + "Match this level of depth, structure, and style.\n\n"
        + f"<example>\n{example_text}\n</example>"
    )


def load_prior_section(section_n: int, version: str, gen_dir: Path) -> str:
    """Load a previously generated section to feed as context for a later section."""
    path = gen_dir / f"s{section_n}_{version}.md"
    if not path.exists():
        raise SystemExit(f"prior section not found: {path}")
    return path.read_text()


def build_context_block(prior_specs: list[str], gen_dir: Path) -> str:
    if not prior_specs:
        return ""
    parts = []
    for spec in prior_specs:
        if "=" not in spec:
            raise SystemExit(f"bad --prior spec: {spec!r} (want sN=vK)")
        sn, ver = spec.split("=", 1)
        n = int(sn.lstrip("s"))
        parts.append(load_prior_section(n, ver, gen_dir))
    return (
        "<prior_sections>\n"
        "The following sections have already been written for this paper. "
        "Do NOT repeat their content — reference it where needed and expand with new detail.\n\n"
        + "\n\n---\n\n".join(parts)
        + "\n</prior_sections>\n\n"
    )


def generate(section: int, version: str, prompt_text: str, prior: list[str], thinking: str, paper_txt: Path, gen_dir: Path) -> str:
    paper_text = paper_txt.read_text()
    system = build_system_preamble()
    user = f"<paper>\n{paper_text}\n</paper>\n\n{build_context_block(prior, gen_dir)}{prompt_text}"

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    if thinking == "none":
        extra_body = {"chat_template_kwargs": {"thinking": False}}
    else:
        extra_body = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": thinking}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Pro",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=1.0,
        top_p=1.0,
        extra_body=extra_body,
    )
    dt = time.time() - t0
    out = resp.choices[0].message.content or ""

    # Defensive: if the deepseek_v4 reasoning parser leaks the thinking block
    # into message.content, strip everything up to and including the closing
    # </think> tag and keep only the real answer.
    if "</think>" in out:
        out = out.split("</think>", 1)[1].lstrip()

    meta = {
        "section": section,
        "version": version,
        "thinking_effort": thinking,
        "duration_sec": round(dt, 1),
        "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "output_tokens": resp.usage.completion_tokens if resp.usage else None,
        "reasoning_tokens": getattr(resp.usage, "reasoning_tokens", None) if resp.usage else None,
        "prompt_chars": len(prompt_text),
        "output_chars": len(out),
        "prior": prior,
    }

    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / f"s{section}_{version}.md"
    out_path.write_text(out)
    meta_path = gen_dir / f"s{section}_{version}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n=== s{section}_{version} ===")
    print(f"  duration: {dt:.1f}s  prompt: {meta['input_tokens']}tok  out: {meta['output_tokens']}tok  reasoning: {meta['reasoning_tokens']}tok")
    print(f"  saved: {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", type=int, required=True)
    ap.add_argument("--version", required=True, help="e.g. v1, v2a, etc.")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prior", default="", help="comma list sN=vK, e.g. s1=v3,s2=v2")
    ap.add_argument("--thinking", default="high", choices=["none", "high", "max"])
    ap.add_argument("--paper-file", default=str(DEFAULT_PAPER_TXT), help="path to extracted paper text")
    ap.add_argument("--gen-dir", default=str(DEFAULT_GEN_DIR), help="output dir for generations and prior lookups")
    args = ap.parse_args()

    prompt_text = Path(args.prompt_file).read_text()
    prior = [p for p in args.prior.split(",") if p]
    generate(args.section, args.version, prompt_text, prior, args.thinking, Path(args.paper_file), Path(args.gen_dir))


if __name__ == "__main__":
    main()
