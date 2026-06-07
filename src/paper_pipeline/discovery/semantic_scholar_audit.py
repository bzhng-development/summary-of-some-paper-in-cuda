"""Semantic Scholar audit for the orgs OpenAlex misses.

OpenAlex landed 523 papers but returned ~0 for OpenAI / Anthropic / DeepSeek /
Mistral / Moonshot / etc. because their arxiv preprints don't have OpenAlex DOI
cross-links yet. Semantic Scholar stores author-level affiliations and links
arxiv via externalIds.ARXIV, so it's the right tool here.

Approach (avoiding the name-collision bug from arxiv author-search):
  1. For each problem org, list ~5-10 known authors.
  2. /author/search?query=<name> -> get author candidates with affiliations.
  3. Pick the one whose affiliations contain the org name (disambiguation).
  4. /author/{id}/papers?publicationDateOrYear=2026&fields=externalIds,title
  5. Filter papers with externalIds.ARXIV set + arxiv_id starts with "26".
  6. Cross-ref Neon and dump missing.

Rate limit: 1 req/sec without key, 100/sec with key. Sleep 1.1s between calls.

Usage:
    DATABASE_URL=... uv run python throwaway_script/discovery/semantic_scholar_audit.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
from loguru import logger


from paper_pipeline.core.neon_db import NeonDB, TABLE


S2_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"x-api-key": os.environ.get("S2_API_KEY")} if os.environ.get("S2_API_KEY") else {}
# S2 caps unauthenticated traffic at ~1 req/sec; bursts inside one second
# trigger 429 with multi-minute cooldowns. Enforce a strict GLOBAL gap so
# no two requests fire within the same second regardless of which function
# issues them.
_MIN_GAP = 1.05 if not HEADERS else 0.05  # tiny buffer over the spec limit
_last_request_at: float = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _MIN_GAP:
        time.sleep(_MIN_GAP - elapsed)
    _last_request_at = time.monotonic()


# Org -> (affiliation match patterns, list of known author names).
# Affiliation pattern matches the strings S2 stores on author records
# (e.g. "OpenAI", "Meta AI", "FAIR"). Used to disambiguate name collisions:
# a "Mark Chen" with affiliation "Stanford" is NOT the OpenAI Mark Chen.
ORG_CONFIG: dict[str, tuple[list[re.Pattern], list[str]]] = {
    "OpenAI": (
        [re.compile(r"\bopenai\b", re.IGNORECASE)],
        [
            "Mark Chen",
            "John Schulman",
            "Sam McCandlish",
            "Wojciech Zaremba",
            "Aleksander Madry",
            "Lukasz Kaiser",
            "Jakub Pachocki",
            "Ilya Sutskever",
            "Jacob Hilton",
            "Karl Cobbe",
            "Nat McAleese",
            "Hyung Won Chung",
        ],
    ),
    "Anthropic": (
        [re.compile(r"\banthropic\b", re.IGNORECASE)],
        [
            "Tom Henighan",
            "Sam Bowman",
            "Dario Amodei",
            "Tom Brown",
            "Karina Nguyen",
            "Christopher Olah",
            "Catherine Olsson",
            "Jared Kaplan",
            "Daniela Amodei",
            "Trenton Bricken",
            "Andy Jones",
        ],
    ),
    "DeepSeek": (
        [re.compile(r"\bdeepseek\b", re.IGNORECASE)],
        [
            "Daya Guo",
            "Qihao Zhu",
            "Wenfeng Liang",
            "Junxiao Song",
            "Bochao Wu",
            "Chong Ruan",
            "Damai Dai",
            "Y. Wu",
            "Haowei Zhang",
        ],
    ),
    "Mistral": (
        [re.compile(r"\bmistral\b", re.IGNORECASE)],
        [
            "Guillaume Lample",
            "Alexandre Sablayrolles",
            "Devendra Singh Chaplot",
            "Diego de Las Casas",
            "Lucile Saulnier",
            "Arthur Mensch",
            "Pierre Stock",
            "Marie-Anne Lachaux",
        ],
    ),
    "Moonshot AI": (
        [re.compile(r"\b(moonshot|kimi)\b", re.IGNORECASE)],
        ["Yuxin Wu", "Yixuan Bai", "Junjie Wang", "Jiawei Liu"],
    ),
    "Zhipu / GLM": (
        [re.compile(r"\b(zhipu|z\.ai|glm)\b", re.IGNORECASE)],
        ["Aohan Zeng", "Xiao Liu", "Wenyi Hong", "Jiale Cheng", "Hanlin Zhao", "Zihan Wang", "Yuxiao Dong", "Jie Tang"],
    ),
    "MiniMax": (
        [re.compile(r"\bminimax\b", re.IGNORECASE)],
        ["Junjie Yan", "Jiyao Wang", "Aonian Li", "Bangwei Gong"],
    ),
    "01.AI": (
        [re.compile(r"\b(01[\.-]?ai|01\.AI)\b", re.IGNORECASE)],
        ["Kai-Fu Lee", "Alex Young", "Bei Chen", "Chao Li"],
    ),
    "StepFun": (
        [re.compile(r"\bstepfun\b", re.IGNORECASE)],
        ["Daxin Jiang", "Xinran Zhao"],
    ),
    "ByteDance Seed": (
        [re.compile(r"\b(bytedance|tiktok|doubao|seed)\b", re.IGNORECASE)],
        ["Lei Li", "Xueqi Cheng", "Yu Zhang", "Hongyu Chen"],
    ),
    "Cohere": (
        [re.compile(r"\bcohere\b", re.IGNORECASE)],
        ["Aidan Gomez", "Nick Frosst", "Sara Hooker", "Marzieh Fadaee"],
    ),
    "Hugging Face": (
        [re.compile(r"\bhugging\s*face\b", re.IGNORECASE)],
        [
            "Loubna Ben Allal",
            "Anton Lozhkov",
            "Lewis Tunstall",
            "Edward Beeching",
            "Leandro von Werra",
            "Quentin Lhoest",
            "Thomas Wolf",
            "Patrick von Platen",
        ],
    ),
    "Snap Research": (
        [re.compile(r"\bsnap\s+(inc|research)\b", re.IGNORECASE)],
        ["Sergey Tulyakov", "Aliaksandr Siarohin", "Jian Ren"],
    ),
}


def s2_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    url = f"{S2_BASE}/{endpoint.lstrip('/')}"
    for attempt in range(6):
        _throttle()
        try:
            r = httpx.get(url, params=params, headers=HEADERS, timeout=30)
        except Exception as e:
            logger.warning(f"  request failed: {e}")
            time.sleep(2)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            # S2 puts a per-IP cooldown after even a small burst. Back off
            # AGGRESSIVELY — start at 30s and grow.
            wait = 30 * (attempt + 1)
            logger.warning(f"  429 rate-limit, sleeping {wait}s (attempt {attempt + 1}/6)")
            time.sleep(wait)
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(5)
            continue
        logger.warning(f"  HTTP {r.status_code} for {endpoint} params={params}: {r.text[:120]}")
        return None
    return None


def find_author(name: str, affil_patterns: list[re.Pattern]) -> dict | None:
    """Search S2 for author by name, return the one whose affiliations match."""
    data = s2_get(
        "author/search",
        params={"query": name, "fields": "name,affiliations,paperCount", "limit": 20},
    )
    if not data or "data" not in data:
        return None
    candidates = data["data"]
    # Match: any of the author's affiliation strings matches any org pattern
    for c in candidates:
        affs = c.get("affiliations") or []
        for aff in affs:
            for pat in affil_patterns:
                if pat.search(aff or ""):
                    return c
    return None


def fetch_2026_arxiv_papers(author_id: str) -> list[dict]:
    """Get all 2026 arxiv papers for an author."""
    out = []
    offset = 0
    while True:
        data = s2_get(
            f"author/{author_id}/papers",
            params={
                "publicationDateOrYear": "2026",
                "fields": "externalIds,title,year",
                "limit": 100,
                "offset": offset,
            },
        )
        if not data or "data" not in data:
            break
        for p in data["data"]:
            ext = p.get("externalIds") or {}
            arxiv = ext.get("ArXiv") or ext.get("ARXIV")
            if arxiv and re.match(r"^\d{4}\.\d{4,5}$", arxiv) and arxiv.startswith("26"):
                out.append(
                    {
                        "arxiv_id": arxiv,
                        "title": p.get("title") or "",
                        "year": p.get("year"),
                    }
                )
        if not data.get("next"):
            break
        offset = data["next"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/semantic_scholar_audit.jsonl"))
    args = ap.parse_args()

    db = NeonDB()
    with db.get_conn() as c, c.cursor() as cur:
        cur.execute(f"SELECT id FROM {TABLE} WHERE id LIKE '26%'")
        in_neon = {r[0] for r in cur.fetchall()}
    logger.info(f"Neon already has {len(in_neon)} arxiv 2026 ids\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grand: dict[str, set[str]] = {}
    with args.output.open("w", encoding="utf-8") as fh:
        for org, (patterns, authors) in ORG_CONFIG.items():
            logger.info(f"=== {org} ({len(authors)} candidate authors) ===")
            unique_ids: dict[str, dict] = {}
            for name in authors:
                candidate = find_author(name, patterns)
                if not candidate:
                    logger.info(f"  {name}: no affil-matching author found")
                    continue
                aid = candidate.get("authorId")
                affs = candidate.get("affiliations") or []
                logger.info(f"  {name} -> {candidate.get('name')} ({aid}) affils={affs}")
                for p in fetch_2026_arxiv_papers(aid):
                    if p["arxiv_id"] not in unique_ids:
                        unique_ids[p["arxiv_id"]] = p
            for aid, p in unique_ids.items():
                rec = {
                    "arxiv_id": aid,
                    "title": p["title"],
                    "org_label": org,
                    "in_neon": aid in in_neon,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            grand[org] = set(unique_ids)
            new_count = sum(1 for aid in unique_ids if aid not in in_neon)
            logger.info(f"  [{org}] total={len(unique_ids)}  NEW={new_count}\n")

    print("\n=== SUMMARY ===")
    print(f"{'org':18s} {'total':>6} {'NEW':>5}")
    print("-" * 32)
    total_new = 0
    for org, ids in sorted(grand.items(), key=lambda x: -len(x[1] - in_neon)):
        new = len(ids - in_neon)
        total_new += new
        print(f"{org:18s} {len(ids):>6} {new:>5}")
    print(f"\n  GRAND NEW: {total_new}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
