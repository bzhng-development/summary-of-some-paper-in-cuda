#!/usr/bin/env python3
"""Generate paper summaries via OpenRouter.

Modes:
  Single paper : downloads one arxiv PDF and saves to examples/
  Backfill     : re-summarizes ALL interested papers in the DB that lack a summary

Usage:
    export OPENROUTER_API_KEY="..."
    uv run python generate_example.py --arxiv-id 2408.03314
    uv run python generate_example.py --gemini --backfill
"""
from __future__ import annotations

import argparse
import base64
import os
import sqlite3
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

load_dotenv()

from database import DB_PATH, save_paper_sync
from multi_prompt import SECTION_SPECS, _SYSTEM_PREAMBLE

EXAMPLES_DIR = Path(__file__).parent / "examples"


def download_pdf(arxiv_id: str) -> bytes:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    logger.info(f"Downloading {pdf_url}")
    resp = httpx.get(pdf_url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    logger.info(f"Downloaded {len(resp.content)} bytes")
    return resp.content


def generate_section(
    client: OpenAI,
    model: str,
    pdf_data_url: str,
    section_prompt: str,
    prior_sections: str,
    extra_body: dict | None = None,
) -> str:
    text_content = ""
    if prior_sections:
        text_content += f"<prior_sections>\n{prior_sections}\n</prior_sections>\n\n"
    text_content += section_prompt

    kwargs: dict = dict(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PREAMBLE},
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "paper.pdf",
                            "file_data": pdf_data_url,
                        },
                    },
                    {"type": "text", "text": text_content},
                ],
            },
        ],
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def summarize_paper(client: OpenAI, model: str, arxiv_id: str, extra_body: dict | None = None) -> str:
    """Download a paper and generate a full multi-section summary. Returns combined markdown."""
    pdf_bytes = download_pdf(arxiv_id)
    pdf_data_url = f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"

    sections: list[str] = []
    for spec in SECTION_SPECS:
        logger.info(f"  Section {spec.number}/{len(SECTION_SPECS)}: {spec.title}")
        prior = "\n\n".join(sections[i - 1] for i in spec.depends_on if i - 1 < len(sections))
        output = generate_section(client, model, pdf_data_url, spec.prompt, prior, extra_body)
        sections.append(output)
        logger.info(f"    -> {len(output)} chars")

    return "\n\n".join(sections)


def get_backfill_papers() -> list[tuple[str, str]]:
    """Return (arxiv_id, title) for interested papers missing a summary."""
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, title FROM papers WHERE interested = 1 AND (summary IS NULL OR summary = '')"
    ).fetchall()
    con.close()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate paper summaries via OpenRouter")
    parser.add_argument("--arxiv-id", default="2408.03314", help="ArXiv ID (single-paper mode)")
    parser.add_argument("--model", default="anthropic/claude-opus-4.6")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini 3 Flash via Google AI Studio (free tier)")
    parser.add_argument("--backfill", action="store_true", help="Re-summarize all interested papers missing a summary")
    args = parser.parse_args()

    if args.gemini:
        args.model = "google/gemini-3-flash-preview"

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
    )
    extra_body = None
    if args.gemini:
        extra_body = {"provider": {"only": ["google-ai-studio"], "allow_fallbacks": False}}
        logger.info(f"Gemini mode: {args.model} via Google AI Studio")

    if args.backfill:
        papers = get_backfill_papers()
        logger.info(f"Backfill: {len(papers)} interested papers need summaries")
        for i, (arxiv_id, title) in enumerate(papers, 1):
            logger.info(f"[{i}/{len(papers)}] {arxiv_id} — {title}")
            try:
                full_summary = summarize_paper(client, args.model, arxiv_id, extra_body)
                save_paper_sync(arxiv_id, summary=full_summary)
                logger.success(f"  Saved {len(full_summary)} chars to DB")
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue
        logger.success(f"Backfill complete: processed {len(papers)} papers")
    else:
        full_summary = summarize_paper(client, args.model, args.arxiv_id, extra_body)
        EXAMPLES_DIR.mkdir(exist_ok=True)
        out_path = EXAMPLES_DIR / f"{args.arxiv_id}_example.md"
        out_path.write_text(full_summary)
        logger.success(f"Saved {len(full_summary)} chars to {out_path}")


if __name__ == "__main__":
    main()
