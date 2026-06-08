"""Use Playwright to scrape arxiv for org-affiliated papers.

Why this exists:
  - Firecrawl free tier blocks engineering.linkedin.com / ai.meta.com /
    tech.instacart.com and we'd have to pay for premium IPs.
  - arxiv API blocks our IP for hours after the daily scoring run hits it.
  - Playwright hits arxiv.org directly through a headless Chromium from our
    own IP, no rate-limit issues, no payment.

Two-step flow per org keyword:
  1. List page — https://arxiv.org/search/?searchtype=all&query=<kw>&size=200
     parse arxiv IDs + titles from each result row.
  2. Abs page — https://arxiv.org/abs/<id> for each result. arxiv renders
     author affiliations inline (the .authors div has them). Verify whether
     at least one author's affiliation matches the org regex.

Output: local_data/playwright_org_papers.jsonl with one row per VERIFIED
paper (passes affiliation match), so the dataset is high-precision.

Usage:
    uv run --with playwright python src/paper_pipeline/discovery/playwright_org_scrape.py \\
        --output local_data/playwright_org_papers.jsonl \\
        --pages-per-query 4 \\
        [--orgs LinkedIn,Meta]

Resume-safe: re-runs skip arxiv_ids already in the output.
"""

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from playwright.async_api import async_playwright


# (org label, search-keyword, affiliation regex). Multiple keywords per org
# are searched in sequence; the affiliation regex is the truth filter.
@dataclass(frozen=True)
class OrgSearch:
    label: str
    keywords: tuple[str, ...]
    affiliation_re: re.Pattern


ORG_SEARCHES: tuple[OrgSearch, ...] = (
    OrgSearch(
        label="LinkedIn",
        keywords=("LinkedIn Corporation", "LinkedIn AI", "LinkedIn"),
        affiliation_re=re.compile(r"\blinkedin\b", re.IGNORECASE),
    ),
    OrgSearch(
        label="Meta / FAIR",
        keywords=("Meta AI", "FAIR", "Facebook AI Research", "Meta Platforms", "Meta GenAI"),
        affiliation_re=re.compile(
            r"\b(meta\s+ai|meta\s+platforms|facebook\s+ai|fair(?:\s+labs?)?|meta\s+research|meta\s+gen.?ai|facebook\s+research)\b",
            re.IGNORECASE,
        ),
    ),
    OrgSearch(
        label="Instacart",
        keywords=("Instacart",),
        affiliation_re=re.compile(r"\binstacart\b", re.IGNORECASE),
    ),
)


ARXIV_ID_RE = re.compile(r"/abs/(\d{4}\.\d{4,5})")


async def fetch_search_page(page, keyword: str, page_idx: int, size: int = 200) -> list[dict]:
    """Return [{arxiv_id, title, abstract, authors_text}] from one search page.

    arxiv abs/search HTML doesn't carry affiliations, so we rely on the
    abstract being visible inline (it is) and post-filter for first-person
    org mentions.
    """
    url = (
        f"https://arxiv.org/search/?searchtype=all&query={keyword.replace(' ', '+')}"
        f"&start={page_idx * size}&size={size}"
    )
    logger.info("[search] {} (page {})", keyword, page_idx)
    await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_selector("li.arxiv-result", timeout=30_000)
    # Expand all "more" abstracts so the full text is in the DOM.
    await page.evaluate("""() => document.querySelectorAll('a.abstract-full').forEach(a => a.click())""")
    results = await page.eval_on_selector_all(
        "li.arxiv-result",
        """nodes => nodes.map(n => {
            const idAnchor = n.querySelector('p.list-title a');
            const idMatch = idAnchor ? idAnchor.href.match(/abs\\/(\\d{4}\\.\\d{4,5})/) : null;
            const titleEl = n.querySelector('p.title');
            const absEl = n.querySelector('span.abstract-full') || n.querySelector('p.abstract');
            const authEl = n.querySelector('p.authors');
            return {
                arxiv_id: idMatch ? idMatch[1] : null,
                title: titleEl ? titleEl.innerText.trim() : '',
                abstract: absEl ? absEl.innerText.replace(/\\u2026 Less/g, '').replace(/△ Less/g, '').trim() : '',
                authors_text: authEl ? authEl.innerText.trim() : '',
            };
        }).filter(p => p.arxiv_id)""",
    )
    return results


# Regexes that detect first-person org affiliation in the abstract.
# We require either "at <Org>", "<Org> researchers", "we present <X> at <Org>",
# or the org name + first-person verbs. False-positive prone but useful at the
# abstract level.
FIRST_PERSON_AFFIL_PATTERNS = {
    "LinkedIn": [
        re.compile(r"\b(at|from|by)\s+linkedin\b", re.IGNORECASE),
        re.compile(
            r"\blinkedin('s)?\s+(team|researchers|recommendation|search|feed|production|system|platform|engineering|deploy)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwe\s+(deploy|present|introduce|propose|describe|launch|build).{0,80}\blinkedin\b", re.IGNORECASE
        ),
        re.compile(r"\bdeployed\s+(at|on|in)\s+linkedin\b", re.IGNORECASE),
    ],
    "Meta / FAIR": [
        re.compile(
            r"\b(at|from|by)\s+(meta\s+ai|fair|facebook\s+ai\s+research|meta\s+platforms|meta\s+gen.?ai)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(meta\s+ai|fair|facebook\s+ai\s+research)('s)?\s+(team|researchers|model|llama)\b", re.IGNORECASE
        ),
        re.compile(
            r"\bwe\s+(deploy|present|introduce|propose|describe|launch|build|train).{0,80}\b(meta\s+ai|fair|facebook)\b",
            re.IGNORECASE,
        ),
    ],
    "Instacart": [
        re.compile(r"\b(at|from|by)\s+instacart\b", re.IGNORECASE),
        re.compile(r"\binstacart('s)?\s+(team|researchers|recommendation|search|platform|production)\b", re.IGNORECASE),
        re.compile(r"\bwe\s+(deploy|present|introduce|propose|describe).{0,80}\binstacart\b", re.IGNORECASE),
        re.compile(r"\bdeployed\s+(at|on|in)\s+instacart\b", re.IGNORECASE),
    ],
}


def first_person_org_match(abstract: str, label: str) -> list[str]:
    """Return list of matching pattern fragments — non-empty if affiliation is strong."""
    hits: list[str] = []
    for pat in FIRST_PERSON_AFFIL_PATTERNS.get(label, []):
        for m in pat.finditer(abstract):
            hits.append(m.group(0))
    return hits


async def fetch_abs_authors(page, arxiv_id: str) -> list[dict]:
    """Return [{name, affiliation}, ...] from the /abs/ page."""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_selector("div.authors", timeout=15_000)
    except Exception:
        return []
    # Authors block is `<div class="authors">Authors:<a>Name</a>, ...</div>`
    # — affiliations aren't always shown inline on the abs page. The fuller
    # source is the META[citation_author_institution] tags; we read both.
    insts = await page.eval_on_selector_all(
        'meta[name="citation_author_institution"]',
        "nodes => nodes.map(n => n.content)",
    )
    names = await page.eval_on_selector_all(
        'meta[name="citation_author"]',
        "nodes => nodes.map(n => n.content)",
    )
    # citation_author + citation_author_institution are emitted in parallel
    # order but only WHEN affiliations exist for that author. If counts differ
    # we treat insts as a free-floating list and only emit the affiliation
    # values (downstream just regex-greps on the affiliation strings).
    if not insts and not names:
        return []
    out: list[dict] = []
    for i, name in enumerate(names):
        aff = insts[i] if i < len(insts) else None
        out.append({"name": name, "affiliation": aff})
    # If there are MORE insts than names (uncommon), tack the extras on as
    # affiliation-only rows so the regex check still sees them.
    for j in range(len(names), len(insts)):
        out.append({"name": None, "affiliation": insts[j]})
    return out


def author_matches(authors: list[dict], aff_re: re.Pattern) -> list[str]:
    hits: list[str] = []
    for a in authors:
        aff = a.get("affiliation") or ""
        if aff and aff_re.search(aff):
            hits.append(aff)
    return hits


async def go(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    already_seen: set[str] = set()
    if args.output.is_file():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                already_seen.add(json.loads(line)["arxiv_id"])
            except json.JSONDecodeError, KeyError:
                continue
        logger.info("resume: {} arxiv_ids already in output", len(already_seen))

    requested_orgs = set(args.orgs) if args.orgs else None
    selected = [o for o in ORG_SEARCHES if requested_orgs is None or o.label in requested_orgs]

    n_verified = 0
    n_rejected = 0
    out_fh = args.output.open("a", encoding="utf-8")
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 900},
        )
        page = await context.new_page()

        for org in selected:
            logger.info("=== org {} (keywords: {}) ===", org.label, list(org.keywords))
            candidates: dict[str, dict] = {}  # arxiv_id → search-row dict
            for kw in org.keywords:
                for p_idx in range(args.pages_per_query):
                    try:
                        results = await fetch_search_page(page, kw, p_idx, size=200)
                    except Exception as e:
                        logger.warning("[{}] page {} failed: {}", kw, p_idx, e)
                        break
                    if not results:
                        logger.info("[{}] empty page {} → stop pagination", kw, p_idx)
                        break
                    for r in results:
                        if r["arxiv_id"] not in candidates:
                            candidates[r["arxiv_id"]] = r
                    logger.info("[{}] page {}: +{} results (cumulative {})", kw, p_idx, len(results), len(candidates))

            todo = {aid: row for aid, row in candidates.items() if aid not in already_seen}
            logger.info("[{}] {} candidates, {} new to triage", org.label, len(candidates), len(todo))

            for aid, row in todo.items():
                abstract = row.get("abstract") or ""
                title = row.get("title") or ""
                # Strong-signal verification: first-person org match in
                # abstract OR explicit author org email/affiliation in
                # authors_text (some search rows include "(LinkedIn)" inline)
                hits = first_person_org_match(abstract, org.label)
                if not hits and org.label.lower() in (row.get("authors_text") or "").lower():
                    hits.append(f"authors_text contains '{org.label}'")
                if not hits:
                    n_rejected += 1
                    already_seen.add(aid)
                    continue
                rec = {
                    "arxiv_id": aid,
                    "title": title,
                    "abstract": abstract[:600],
                    "org_label": org.label,
                    "authors_text": row.get("authors_text"),
                    "affiliation_hits": hits,
                    "abs_url": f"https://arxiv.org/abs/{aid}",
                }
                out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_fh.flush()
                already_seen.add(aid)
                n_verified += 1

            logger.info("[{}] done — verified {} / rejected {} of {} new", org.label, n_verified, n_rejected, len(todo))

        await browser.close()

    out_fh.close()
    print("\n=== done ===")
    print(f"  output:    {args.output}")
    print(f"  verified:  {n_verified}")
    print(f"  rejected:  {n_rejected}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/playwright_org_papers.jsonl"))
    ap.add_argument(
        "--pages-per-query",
        type=int,
        default=4,
        help="Pages of 200 results to walk per keyword. Default 4 = up to 800 results per keyword.",
    )
    ap.add_argument(
        "--orgs",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        help="Subset of ORG labels to run (comma-separated). Defaults to all.",
    )
    args = ap.parse_args()
    asyncio.run(go(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
