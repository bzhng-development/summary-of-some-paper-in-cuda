"""Find DeepMind pagination depth."""

import asyncio

from playwright.async_api import async_playwright


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

        last_ok_page = 1
        for p in range(1, 30):
            url = (
                f"https://deepmind.google/research/publications/page/{p}/"
                if p > 1
                else "https://deepmind.google/research/publications/"
            )
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await page.wait_for_timeout(1500)
            except Exception as e:
                print(f"  page {p}: goto failed {e}")
                break
            status = resp.status if resp else None
            hrefs = await page.eval_on_selector_all(
                "a[href*='/research/publications/']",
                "nodes => nodes.map(n => n.href)",
            )
            # Numeric per-paper slugs only
            ids = {h.rstrip("/").rsplit("/", 1)[-1] for h in hrefs if h.rstrip("/").rsplit("/", 1)[-1].isdigit()}
            print(f"  page {p}: status={status} ids={len(ids)}")
            if not ids and p > 1:
                print(f"  no more papers at page {p}")
                break
            if status and status >= 400:
                print(f"  http {status} at page {p}")
                break
            last_ok_page = p
        print(f"\nlast page with results: {last_ok_page}")
        await b.close()


asyncio.run(main())
