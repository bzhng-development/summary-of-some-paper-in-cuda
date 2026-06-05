"""OpenAlex-based 2026 affiliation audit — replaces the broken regex probe.

For each tracked org:
  1. Resolve OpenAlex institution ID via name search (with manual overrides
     for cases where the auto-pick would be wrong).
  2. Query Works().filter(
       authorships.institutions.id=<inst>,
       publication_year=2026,
       locations.source.id=S4306400194,  # arxiv
     ) — paginate all pages.
  3. Extract arxiv_id from `locations[*].landing_page_url`.
  4. Cross-reference against Neon; flag in_neon.
  5. Write JSONL with org_label + arxiv_id + title + authors + DOI.

Why this is right (vs the abstract-regex probe):
  - OpenAlex stores actual author affiliations from publisher metadata.
  - Server-side filter — no false positives from "GPT-4 by OpenAI" mentions.
  - Free, no key, 10 req/sec polite pool.

Usage:
    DATABASE_URL=... uv run --with pyalex python \\
        throwaway_script/openalex_2026_audit.py
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neon_db import NeonDB, TABLE
from pyalex import Institutions, Works
import pyalex


ARXIV_RE = re.compile(r"(?:arxiv\.org/abs/|arxiv\.org/pdf/)(\d{4}\.\d{4,5})")
ARXIV_DOI_RE = re.compile(r"10\.48550/arXiv\.(\d{4}\.\d{4,5})", re.IGNORECASE)
ARXIV_SOURCE_ID = "S4306400194"  # arxiv.org in OpenAlex

# Override map for orgs where auto-search would pick the wrong institution.
# Key = label to use in output; value = explicit OpenAlex institution ID.
# All IDs hand-verified via Institutions().search() — see openalex_audit
# follow-up. Many earlier guesses were wrong (returned 0 works).
INSTITUTION_OVERRIDES: dict[str, str | None] = {
    "DeepMind": "I4210090411",  # Google DeepMind UK, 8929 works
    "Google Research": "I1291425158",  # Google parent
    "OpenAI": "I4210161460",  # 1703 works
    "Anthropic": "I4387930290",  # 0 works currently — they don't push to OpenAlex
    "Meta / FAIR": "I4210114444",  # Meta (US), 4153 works (= Facebook + Meta AI)
    "Microsoft Research": "I1290206253",  # Microsoft parent
    "Apple": "I4210153776",  # Apple (US), 3041 works
    "NVIDIA": "I4210127875",  # Nvidia US, 5081 works
    "IBM Research": "I1283280774",
    "Salesforce": "I4210155268",  # Salesforce US, 1353 works
    "Mistral": "I4390039361",  # Mistral AI France, 34 works
    "Cohere": "I4401726847",  # Cohere Canada, 0 works
    "Snowflake": "I4400600931",
    "Tencent": "I2250653659",  # Tencent China, 9807 works (big)
    "ByteDance": "I4405256866",
    "Baidu": None,  # search fallback
    "Huawei": None,
    "Samsung": None,
    "Sony": None,
    "LinkedIn": "I1316064682",  # 1324 works
    "Snap": "I4210142583",  # 674 works
    "Pinterest": "I4401726932",
    "Netflix": "I869089601",
    "Spotify": "I4401726855",
    "Adobe": "I1306409833",  # Adobe Systems US, 5948 works
    "Amazon": None,  # search fallback
    "AllenAI": "I4210156221",  # Allen Inst for AI, 1291 works
    "HuggingFace": "I4387154989",  # 278 works
    "DeepSeek": "I4405257960",
    "Moonshot AI": "I4405260227",
    "Zhipu / GLM": "I4401726915",
    "MiniMax": "I4405258935",
    "01.AI": "I4405259341",
    "StepFun": "I4405260255",
    "Tsinghua": "I29280464",
    "Berkeley": "I95457486",
    "Stanford": "I97018004",
    "CMU": "I74973139",
    "UW": "I201448701",
    "MIT": "I63966007",
    "Princeton": "I20231570",
    "NYU": "I57206974",
    # No OpenAlex institution exists for these — skipped by resolver:
    "Together AI": None,
    "AI21": None,
    "EleutherAI": None,
    "LAION": None,
    "Baichuan": None,
    "Alibaba": None,
}


# What we want to track. Each entry: (display_label, [name_aliases_for_fallback_search])
ORGS: list[tuple[str, list[str]]] = [
    # Western LLM labs
    ("Meta / FAIR", ["Meta AI Research", "FAIR Labs"]),
    ("DeepMind", ["Google DeepMind"]),
    ("Anthropic", ["Anthropic"]),
    ("OpenAI", ["OpenAI"]),
    ("Microsoft Research", ["Microsoft Research"]),
    ("Google Research", ["Google Research"]),
    ("NVIDIA", ["Nvidia Research", "Nvidia"]),
    ("Apple", ["Apple Inc"]),
    ("IBM Research", ["IBM Research"]),
    ("Salesforce", ["Salesforce Research"]),
    ("Together AI", ["Together AI"]),
    ("Snowflake", ["Snowflake"]),
    ("AI21", ["AI21 Labs"]),
    ("Mistral", ["Mistral AI"]),
    ("Cohere", ["Cohere"]),
    ("LinkedIn", ["LinkedIn Corporation"]),
    ("HuggingFace", ["Hugging Face"]),
    ("Adobe", ["Adobe Research"]),
    ("Amazon", ["Amazon"]),
    ("Snap", ["Snap Inc"]),
    ("Pinterest", ["Pinterest"]),
    ("Netflix", ["Netflix"]),
    ("Spotify", ["Spotify"]),
    # Chinese / Asian labs
    ("DeepSeek", ["DeepSeek AI", "DeepSeek"]),
    ("Moonshot AI", ["Moonshot AI"]),
    ("Zhipu / GLM", ["Zhipu AI"]),
    ("MiniMax", ["MiniMax"]),
    ("01.AI", ["01.AI", "01 AI"]),
    ("StepFun", ["StepFun"]),
    ("Baichuan", ["Baichuan"]),
    ("Tencent", ["Tencent"]),
    ("ByteDance", ["ByteDance"]),
    ("Alibaba", ["Alibaba"]),
    ("Baidu", ["Baidu"]),
    ("Huawei", ["Huawei Noah's Ark"]),
    ("Samsung", ["Samsung Research"]),
    ("Sony", ["Sony"]),
    # Universities
    ("Berkeley", ["University of California Berkeley"]),
    ("Stanford", ["Stanford University"]),
    ("CMU", ["Carnegie Mellon University"]),
    ("UW", ["University of Washington"]),
    ("MIT", ["Massachusetts Institute of Technology"]),
    ("Princeton", ["Princeton University"]),
    ("NYU", ["New York University"]),
    ("Tsinghua", ["Tsinghua University"]),
    # Research nonprofits
    ("AllenAI", ["Allen Institute for AI"]),
    ("EleutherAI", ["EleutherAI"]),
    ("LAION", ["LAION"]),
]


def normalize_inst_id(s: str) -> str:
    return s.replace("https://openalex.org/", "")


def resolve_institution(label: str, aliases: list[str]) -> tuple[str, str] | None:
    """Return (inst_id_short, display_name) or None."""
    override = INSTITUTION_OVERRIDES.get(label)
    if override:
        return override, f"(override) {label}"
    # Try aliases in order
    for q in aliases:
        try:
            hits = Institutions().search(q).get(per_page=5)
        except Exception:
            continue
        if hits:
            # Pick the one with most works (most likely the lab itself)
            best = max(hits, key=lambda h: h.get("works_count") or 0)
            inst_id = normalize_inst_id(best["id"])
            return inst_id, best.get("display_name", q)
    return None


def extract_arxiv_id(work: dict) -> str | None:
    """Find an arxiv ID in DOI, ids field, or any location URL.

    DOI form: `10.48550/arXiv.XXXX.XXXXX` (arxiv's official DOI prefix).
    Locations: `landing_page_url` / `pdf_url` containing arxiv.org/abs|pdf/.
    """
    # 1. DOI — most reliable
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    if doi:
        m = ARXIV_DOI_RE.search(doi)
        if m:
            return m.group(1)
    # 2. ids field
    ids = work.get("ids") or {}
    for v in ids.values():
        if isinstance(v, str):
            m = ARXIV_RE.search(v) or ARXIV_DOI_RE.search(v)
            if m:
                return m.group(1)
    # 3. locations
    locations = list(work.get("locations") or [])
    if work.get("primary_location"):
        locations.append(work["primary_location"])
    if work.get("best_oa_location"):
        locations.append(work["best_oa_location"])
    for loc in locations:
        if not loc:
            continue
        for key in ("landing_page_url", "pdf_url"):
            url = loc.get(key) or ""
            m = ARXIV_RE.search(url)
            if m:
                return m.group(1)
    return None


def fetch_org_works(inst_id: str, max_pages: int = 10) -> list[dict]:
    """Get all 2026 works for an institution via pyalex paginate.

    NOTE: we DO NOT filter by locations.source=arxiv here — OpenAlex marks
    arxiv as a location for only a tiny fraction of papers. Instead we
    fetch all 2026 papers and apply our own arxiv-ID extractor downstream.
    """
    out: list[dict] = []
    try:
        query = Works().filter(
            authorships={"institutions": {"id": inst_id}},
            publication_year=2026,
        )
        for page in query.paginate(per_page=200, n_max=max_pages * 200):
            out.extend(page)
            if len(out) >= max_pages * 200:
                break
    except Exception as e:
        print(f"  ! query error: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/openalex_2026_audit.jsonl"))
    ap.add_argument(
        "--email", type=str, default=None, help="Optional contact email for OpenAlex polite pool (10/s vs default)."
    )
    args = ap.parse_args()

    if args.email:
        pyalex.config.email = args.email

    db = NeonDB()
    with db.get_conn() as c, c.cursor() as cur:
        cur.execute(f"SELECT id FROM {TABLE} WHERE id LIKE '26%'")
        in_neon_2026 = {row[0] for row in cur.fetchall()}
    print(f"Neon already has {len(in_neon_2026)} arxiv_ids in 2026\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    per_org: list[tuple[str, int, int, int]] = []  # (org, all26, in_neon, missing)
    grand_new = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for label, aliases in ORGS:
            resolved = resolve_institution(label, aliases)
            if not resolved:
                print(f"  [{label}] couldn't resolve institution — skipping")
                per_org.append((label, 0, 0, 0))
                continue
            inst_id, display = resolved
            works = fetch_org_works(inst_id)
            org_in_neon = 0
            org_missing = 0
            for w in works:
                aid = extract_arxiv_id(w)
                if not aid or not aid.startswith("26"):
                    continue
                already = aid in in_neon_2026
                if already:
                    org_in_neon += 1
                else:
                    org_missing += 1
                authors = [a.get("author", {}).get("display_name") for a in (w.get("authorships") or [])[:5]]
                rec = {
                    "arxiv_id": aid,
                    "title": (w.get("title") or "").strip(),
                    "org_label": label,
                    "openalex_inst": inst_id,
                    "openalex_inst_display": display,
                    "doi": w.get("doi"),
                    "authors": [a for a in authors if a],
                    "publication_date": w.get("publication_date"),
                    "in_neon": already,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            total = org_in_neon + org_missing
            per_org.append((label, total, org_in_neon, org_missing))
            grand_new += org_missing
            print(
                f"  [{label:25s}] total={total:>4}  in_neon={org_in_neon:>4}  MISSING={org_missing:>4}  ({display[:50]})"
            )
            time.sleep(0.15)

    print("\n=== SUMMARY (sorted by NEW missing from Neon) ===")
    print(f"{'org':25s} {'total':>6} {'in_neon':>8} {'MISSING':>8}")
    print("-" * 55)
    for label, t, i, m in sorted(per_org, key=lambda x: -x[3]):
        print(f"{label:25s} {t:>6} {i:>8} {m:>8}")
    print(f"\n  GRAND TOTAL new arxiv IDs missing from Neon: {grand_new}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
