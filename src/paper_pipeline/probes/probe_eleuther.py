"""Probe EleutherAI + Google Research pages."""

import asyncio
import re

from playwright.async_api import async_playwright

ARXIV_RE = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\w.])")


async def probe(page, url, label):
    print(f"\n=== {label}: {url} ===")
    await page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(3000)

    # Aggressive scroll loop
    prev = 0
    for i in range(80):
        h = await page.evaluate("() => document.body.scrollHeight")
        if h == prev and i > 5:
            print(f"  stable at iter {i}, height {h}")
            break
        await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(700)
        prev = h

    # Find all arxiv anchors and arxiv-id-like strings
    hrefs = await page.eval_on_selector_all("a[href]", "nodes => nodes.map(n => n.href)")
    arxiv_hrefs = sorted({h for h in hrefs if "arxiv.org" in h})
    print(f"  arxiv hrefs: {len(arxiv_hrefs)}")
    for h in arxiv_hrefs[:5]:
        print(f"    {h}")

    full = await page.evaluate("() => document.body.innerText || ''")
    text_ids = sorted({m.group(1) for m in ARXIV_RE.finditer(full)})
    # filter to realistic years
    text_ids = [a for a in text_ids if 7 <= int(a[:2]) <= 26 and 1 <= int(a[2:4]) <= 12]
    print(f"  arxiv ids in body text: {len(text_ids)}")
    print(f"    sample: {text_ids[:10]}")

    # Total link count
    print(f"  total <a> links: {len(hrefs)}")

    # Look for pagination
    pag = [h for h in hrefs if any(kw in h.lower() for kw in ["page/", "?page=", "?p=", "/papers/page"])]
    print(f"  pagination-like hrefs: {len(set(pag))}")
    for p in list(set(pag))[:5]:
        print(f"    {p}")

    # Look for "Load more" buttons
    btns = await page.eval_on_selector_all(
        "button, a",
        "nodes => nodes.map(n => (n.innerText || '').trim()).filter(s => s.length > 0 && s.length < 30)",
    )
    interesting = [b for b in btns if any(k in b.lower() for k in ["more", "load", "next", "show"])]
    print(f"  load-more-like buttons: {interesting[:10]}")


async def main():
    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        ctx = await b.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 2200},
        )
        page = await ctx.new_page()
        await probe(page, "https://www.eleuther.ai/papers", "EleutherAI")
        await probe(page, "https://research.google/pubs/", "Google Research")
        await b.close()


asyncio.run(main())
