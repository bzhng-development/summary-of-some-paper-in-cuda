"""Find every LinkedIn-affiliated arxiv paper submitted in 2026.

arxiv IDs from 2026 are 2601.xxxxx through 2612.xxxxx. We search arxiv's
listing page with LinkedIn keywords, then for each paper:
  - extract arxiv_id, title, abstract from the search result row
  - filter by first-person affiliation pattern in the abstract

Uses our existing playwright_org_scrape patterns; output is for review,
not direct DB insert.
"""

import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright

QUERIES = [
    "LinkedIn",
    "LinkedIn AI",
    "LinkedIn Corporation",
    "LinkedIn Engineering",
]

FIRST_PERSON_LI = [
    re.compile(r"\b(at|from|by)\s+linkedin\b", re.IGNORECASE),
    re.compile(
        r"\blinkedin('s)?\s+(team|researchers|recommendation|search|feed|production|system|platform|engineering|deploy|home|jobs|app|members|users|API|data|graph|infrastructure|content)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwe\s+(deploy|present|introduce|propose|describe|launch|build|train|develop|share).{0,80}\blinkedin\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bdeployed\s+(at|on|in)\s+linkedin\b", re.IGNORECASE),
    re.compile(r"\blinkedin\.com\b", re.IGNORECASE),
]


def linkedin_match(abstract: str) -> list[str]:
    return [m.group(0) for pat in FIRST_PERSON_LI for m in pat.finditer(abstract)]


async def fetch_search_page(page, keyword: str, page_idx: int, size: int = 200):
    url = (
        f"https://arxiv.org/search/?searchtype=all&query={keyword.replace(' ', '+')}"
        f"&start={page_idx * size}&size={size}"
    )
    print(f"  [search] {keyword} page {page_idx}")
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_selector("li.arxiv-result", timeout=15_000)
        await page.evaluate("() => document.querySelectorAll('a.abstract-full').forEach(a => a.click())")
    except Exception:
        return []
    return await page.eval_on_selector_all(
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


async def main():
    out = Path("local_data/linkedin_2026_probe.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, dict] = {}
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        context = await b.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        for kw in QUERIES:
            for p_idx in range(5):
                rows = await fetch_search_page(page, kw, p_idx, size=200)
                if not rows:
                    break
                for r in rows:
                    aid = r["arxiv_id"]
                    if aid and aid.startswith("26") and aid not in candidates:
                        candidates[aid] = r
        await b.close()
    print(f"\ncandidates with arxiv_id starting '26' (2026): {len(candidates)}")

    verified = []
    for row in candidates.values():
        hits = linkedin_match(row.get("abstract") or "")
        if hits or "linkedin" in (row.get("authors_text") or "").lower():
            verified.append({**row, "affiliation_hits": hits})
    print(f"verified LinkedIn-affiliated 2026: {len(verified)}")

    with out.open("w", encoding="utf-8") as fh:
        for rec in verified:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n=== verified LinkedIn 2026 papers ===")
    for rec in sorted(verified, key=lambda x: x["arxiv_id"]):
        print(f"  {rec['arxiv_id']}  {rec['title'][:90]}")


if __name__ == "__main__":
    asyncio.run(main())
