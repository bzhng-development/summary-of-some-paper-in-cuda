"""Find arxiv papers from orgs that Firecrawl can't crawl (Meta, LinkedIn).

The arxiv search API doesn't expose author-affiliation as a queryable field,
but the atom feed DOES include affiliations on each author. So:
  - Query `abs:"<org marker>"` to net a superset
  - For each result, post-filter by `author.affiliation` regex
  - Output passing arxiv_ids + titles for downstream bulk-marking

Usage:
    uv run --with arxiv python throwaway_script/discovery/arxiv_org_search.py \\
        --output local_data/arxiv_org_papers.jsonl

The arxiv API rate-limits to ~1 req/3s; arxiv.py's Client enforces this
internally with default delay_seconds=3.0.
"""

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import arxiv
from loguru import logger


# (label, abstract-query OR'd terms, affiliation regex)
# The query nets a superset; the affiliation regex (case-insensitive) filters
# to actual org papers. Without the affiliation filter, abs: matches catch
# papers that just mention Meta in passing (e.g. "we compared with Meta's
# Llama-3"), which is the wrong signal here.
ORG_SEARCHES = [
    (
        "Meta / FAIR",
        '(abs:"Meta AI" OR abs:"FAIR" OR abs:"Facebook AI Research" OR abs:"Meta Research" OR abs:"Meta Platforms")',
        re.compile(
            r"\b(meta\s+ai|fair|facebook\s+ai|meta\s+platforms|meta\s+research|meta\s+gen\s+ai)\b", re.IGNORECASE
        ),
    ),
    (
        "LinkedIn",
        '(abs:"LinkedIn" OR abs:"LinkedIn AI" OR abs:"LinkedIn Engineering")',
        re.compile(r"\blinkedin\b", re.IGNORECASE),
    ),
    (
        "Instacart",
        '(abs:"Instacart")',
        re.compile(r"\binstacart\b", re.IGNORECASE),
    ),
]

DEFAULT_SINCE = date(2024, 1, 1)


def matches_affiliation(authors, pattern: re.Pattern) -> tuple[bool, list[str]]:
    """Check whether any author's affiliation matches the org regex.

    Returns (matched, list_of_matching_affiliations) for debugging.
    """
    hits: list[str] = []
    for author in authors:
        aff = getattr(author, "affiliation", None)
        if aff and pattern.search(aff):
            hits.append(aff)
    return bool(hits), hits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/arxiv_org_papers.jsonl"))
    ap.add_argument(
        "--max-results-per-org",
        type=int,
        default=2000,
        help="Cap papers fetched per org (raw, pre-filter). Default 2000.",
    )
    ap.add_argument(
        "--since",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=DEFAULT_SINCE,
        help="Only include papers published on/after this date (YYYY-MM-DD). Default 2024-01-01.",
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Slow client: arxiv API rate-limits aggressively (especially on a hot IP).
    # 100/page × 15s delay = ~7 pages/min = comfortable under any reasonable cap.
    # num_retries with exponential backoff handles transient 429s.
    client = arxiv.Client(page_size=100, delay_seconds=15.0, num_retries=8)

    already_seen: set[str] = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                already_seen.add(json.loads(line)["arxiv_id"])
            except json.JSONDecodeError, KeyError:
                pass
        logger.info("resume: {} arxiv_ids already in output", len(already_seen))

    totals: dict[str, dict[str, int]] = {}

    with args.output.open("a", encoding="utf-8") as fh:
        for label, query, aff_re in ORG_SEARCHES:
            logger.info("[{}] query: {}", label, query)
            search = arxiv.Search(
                query=query,
                max_results=args.max_results_per_org,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            org_stats = {"raw": 0, "kept": 0, "skip_too_old": 0, "skip_aff": 0, "skip_dup": 0}

            for result in client.results(search):
                org_stats["raw"] += 1
                aid = result.entry_id.split("/abs/")[-1].split("v")[0]

                if aid in already_seen:
                    org_stats["skip_dup"] += 1
                    continue

                # Sort is descending; once we see one before the cutoff we can stop.
                pub = result.published.date() if hasattr(result.published, "date") else None
                if pub and pub < args.since:
                    org_stats["skip_too_old"] += 1
                    # Continue iterating in case the sort isn't strict, but bail
                    # after we've seen a streak of old ones.
                    if org_stats["skip_too_old"] >= 25:
                        logger.info("[{}] 25 consecutive too-old results, stopping early", label)
                        break
                    continue

                matched, hits = matches_affiliation(result.authors, aff_re)
                if not matched:
                    org_stats["skip_aff"] += 1
                    continue

                rec = {
                    "arxiv_id": aid,
                    "title": result.title,
                    "org_label": label,
                    "published": pub.isoformat() if pub else None,
                    "authors": [a.name for a in result.authors],
                    "affiliations_matched": hits,
                    "primary_category": getattr(result, "primary_category", None),
                    "abstract": (result.summary or "")[:1200],
                    "pdf_url": result.pdf_url,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                already_seen.add(aid)
                org_stats["kept"] += 1

                if org_stats["kept"] % 25 == 0:
                    logger.info(
                        "[{}] kept={} raw={} skip(too-old={}, aff={}, dup={})",
                        label,
                        org_stats["kept"],
                        org_stats["raw"],
                        org_stats["skip_too_old"],
                        org_stats["skip_aff"],
                        org_stats["skip_dup"],
                    )

            totals[label] = org_stats
            logger.info("[{}] DONE — {}", label, org_stats)

    print("\n=== summary ===")
    for label, stats in totals.items():
        print(f"  {label:20s}  {stats}")
    print(f"\noutput: {args.output}  ({sum(s['kept'] for s in totals.values())} new papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
