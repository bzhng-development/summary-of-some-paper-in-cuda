#!/usr/bin/env python3
"""
papers_by_score.py — Split scored papers into per-score markdown files for LLM verification.

Pipeline: hf_daily_papers.py (score) → papers_by_score.py (split) → feed to LLM level-by-level

Usage:
    python papers_by_score.py --json ./papers_out/all_scored.json --out-dir ./papers_out/by_score
    python papers_by_score.py --json ./papers_out/all_scored.json --out-dir ./papers_out/by_score --min-score 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tiktoken

# Make the repo root importable when run directly from ``daily_papers/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from daily_papers.examples import build_examples_block, load_examples  # noqa: E402

# Same repo path as hf_daily_papers.py
PAPERS_REPO = _REPO_ROOT


def load_examples_block(repo_path: Path) -> str:
    """Return the reading-history block rendered for the LLM.

    Thin wrapper around :func:`daily_papers.examples.load_examples` so the
    header matches the original format (paper count on the first line).
    """
    examples = load_examples(repo_path)
    header = f"The user's reading history ({len(examples)} papers):"
    return build_examples_block(examples, header=header)


def _format_authors_with_affiliations(p: dict) -> str:
    """Format authors with affiliations from the scored JSON."""
    authors = p.get("authors", [])
    affiliations = p.get("affiliations") or {}
    if not affiliations:
        # Fallback to truncated author list
        result = ", ".join(authors[:3])
        if len(authors) > 3:
            result += f" +{len(authors) - 3}"
        return result

    parts = []
    for name in authors[:5]:
        affs = affiliations.get(name)
        if affs:
            parts.append(f"{name} ({', '.join(affs)})")
        else:
            parts.append(name)
    if len(authors) > 5:
        parts.append(f"+{len(authors) - 5}")
    return ", ".join(parts)


def format_paper(p: dict) -> str:
    arxiv_id = p["arxiv_id"]
    title = p["title"]
    upvotes = p.get("upvotes", 0)
    gh = p.get("github") or ""
    gh_stars = p.get("github_stars") or 0
    reason = p.get("reason", "")
    similar = p.get("similar_paper", "NONE")
    categories = ", ".join(p.get("categories", []))
    authors_str = _format_authors_with_affiliations(p)
    org = p.get("org_fullname") or p.get("organization") or ""

    comment = p.get("arxiv_comment") or ""
    journal = p.get("journal_ref") or ""
    published = p.get("published") or ""

    gh_part = f" | [code]({gh}) ({gh_stars}★)" if gh else ""
    lines = [f"## {title}"]
    lines.append(f"[arXiv](https://arxiv.org/abs/{arxiv_id}){gh_part} | {upvotes} upvotes | {categories}")
    lines.append(f"Authors: {authors_str}")
    if org:
        lines.append(f"Org: {org}")
    if comment:
        lines.append(f"Comment: {comment}")
    if journal:
        lines.append(f"Journal: {journal}")
    if published:
        lines.append(f"Published: {published}")
    if similar and similar != "NONE":
        lines.append(f"\n**Similar to:** {similar}")
    lines.append(f"\n**Why:** {reason}\n")
    lines.append("---\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Split scored papers into per-score .md files")
    parser.add_argument("--json", required=True, type=Path, help="Path to all_scored.json")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for score_N.md files")
    parser.add_argument("--min-score", type=int, default=1, help="Only generate files for scores >= this")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("o200k_base")

    papers = json.loads(args.json.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples_block = load_examples_block(PAPERS_REPO)

    by_score: dict[int, list[dict]] = {}
    for p in papers:
        s = p.get("score", 0)
        by_score.setdefault(s, []).append(p)

    for score in sorted(by_score, reverse=True):
        if score < args.min_score:
            continue
        group = sorted(by_score[score], key=lambda p: -p.get("upvotes", 0))

        preamble = (
            "# Verification Pass\n\n"
            f"Below are {len(group)} papers that were auto-scored **{score}/10** by an LLM.\n"
            "This is a noisy process — many of these may not actually be relevant.\n\n"
            "**Your task:** Review each paper and list ONLY the ones that are TRULY relevant "
            "to the user's reading history below. Be strict. Cut through the noise.\n\n"
            "For each paper you keep, output:\n"
            "- arxiv_id\n"
            "- title\n"
            "- 1-sentence reason why it's genuinely relevant\n\n"
            "Papers you don't mention are assumed irrelevant. It's fine to keep very few or none.\n\n"
            "## Reading History (what the user actually cares about)\n\n"
            f"{examples_block}\n\n"
            "---\n\n"
            f"# Candidates (score {score})\n\n"
        )

        body = "\n".join(format_paper(p) for p in group)
        md = preamble + body
        tok = len(enc.encode(md))

        out_path = args.out_dir / f"score_{score}.md"
        out_path.write_text(md)
        print(f"  score_{score}.md — {len(group)} papers, {tok:,} tokens")

    total = sum(len(g) for s, g in by_score.items() if s >= args.min_score)
    print(f"Done: {total} papers across {sum(1 for s in by_score if s >= args.min_score)} score levels")


if __name__ == "__main__":
    main()
