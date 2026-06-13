"""Pitch-prompt iteration driver against vLLM DeepSeek-V4-Pro.

Usage:
    uv run python prompt_iter/iter_pitch.py --version v1 --system-file prompt_iter/prompts/pitch_v1.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DEFAULT_PAPER_TXT = ROOT / "paper_2408.03314.txt"
DEFAULT_GEN_DIR = ROOT / "generations"
DEFAULT_S1 = DEFAULT_GEN_DIR / "s1_LOCKED.md"


class PitchOutput(BaseModel):
    title: str = Field(description="The exact title of the paper as it appears in the PDF")
    pitch: str = Field(
        description="A compelling 2-3 sentence pitch that captures the paper's core contribution and why it matters"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--system-file", required=True)
    ap.add_argument("--thinking", default="high", choices=["none", "high", "max"])
    ap.add_argument("--paper-file", default=str(DEFAULT_PAPER_TXT))
    ap.add_argument("--s1-file", default=str(DEFAULT_S1))
    ap.add_argument("--gen-dir", default=str(DEFAULT_GEN_DIR))
    args = ap.parse_args()

    system = Path(args.system_file).read_text()
    paper_text = Path(args.paper_file).read_text()
    s1_path = Path(args.s1_file)
    s1_text = s1_path.read_text() if s1_path.exists() else ""
    gen_dir = Path(args.gen_dir)

    # Match the original generate_pitch() context shape: first 5000 chars of paper,
    # first 3000 chars of summary (here we use the s1 locked output as the "summary context").
    user_msg = f"<paper>\n{paper_text[:5000]}\n</paper>\n\nPaper Analysis (for context):\n{s1_text[:3000]}..."

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
    if args.thinking == "none":
        extra_body = {"chat_template_kwargs": {"thinking": False}}
    else:
        extra_body = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": args.thinking}}

    t0 = time.time()
    resp = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Pro",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        temperature=1.0,
        top_p=1.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pitch_output", "schema": PitchOutput.model_json_schema()},
        },
        extra_body=extra_body,
    )
    dt = time.time() - t0
    raw = resp.choices[0].message.content or ""
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].lstrip()

    try:
        parsed = PitchOutput.model_validate_json(raw)
        title, pitch = parsed.title, parsed.pitch
    except Exception as exc:
        print(f"PARSE_ERROR: {exc!r}")
        print(f"RAW:\n{raw}")
        return

    out = f"# {title}\n\n{pitch}\n"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / f"pitch_{args.version}.md"
    out_path.write_text(out)

    meta = {
        "version": args.version,
        "thinking_effort": args.thinking,
        "duration_sec": round(dt, 1),
        "input_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "output_tokens": resp.usage.completion_tokens if resp.usage else None,
        "system_chars": len(system),
        "title": title,
        "pitch": pitch,
    }
    (gen_dir / f"pitch_{args.version}.meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n=== pitch {args.version} ({dt:.1f}s) ===")
    print(f"Title: {title}")
    print(f"Pitch: {pitch}")


if __name__ == "__main__":
    main()
