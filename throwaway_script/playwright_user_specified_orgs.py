"""Scrape arxiv IDs from 10 USER-SPECIFIED orgs whose primary URLs were wrong
in earlier sweeps.

Why this script exists
----------------------
The original `playwright_extra_companies.py` configured these 10 orgs with
wrong URLs (Roblox used `/newsroom/` which is press releases; eBay used the
wrong domain root; Lyft / Uber / PayPal / etc. used listing roots that don't
paginate the way the previous scraper expected). Empirically every one of
them (except Snap) emitted 0 net-new arxiv IDs in the prior run. This script
re-attempts each with the CORRECT URL + the correct pagination/extraction
mode, and writes a NEW JSONL output that dedups against the existing three
playwright outputs:
  - local_data/playwright_company_pubs.jsonl
  - local_data/playwright_crack_failed.jsonl
  - local_data/playwright_extra_companies.jsonl

Per-org modes
-------------
- "click_pagination" (Roblox): the listing is JS-driven with numbered
  buttons 1-11. We open the listing once, harvest the 9 paper-detail links
  per page, click the "next page number" button, repeat until page 11.
  Then we follow each paper detail page to extract its arxiv link.
- "paginated" (Uber): URL pattern `/page/N/` actually loads new content.
  Walk pages 1..max_page, collect post URLs, then follow each for arxiv.
- "sitemap_xml" (Bloomberg, Lyft, Instacart): the public sitemap gives
  several hundred post URLs; we fetch each via httpx and grep arxiv.
- "contentful_api" (Spotify): research.atspotify.com is a Next.js site
  backed by Contentful; the `/api/entries?content_type=publication` endpoint
  returns all 171 publications. The `fields.link` is rarely arxiv, but a few
  publications cite arxiv inside the rich-text `fields.content`; mine that.
- "js_bundle" (Snap): re-runs the existing publications-lib.js bundle
  harvest as a verifier; net-new IDs only.
- "medium_rss" (PayPal): `medium.com/feed/<publication>` is a working RSS
  feed up to ~50 posts; follow the linked Medium posts and harvest arxiv.
- "github_org" (Bloomberg fallback): enumerate the org's public repos via
  GitHub REST, fetch each README, grep arxiv.
- "scroll_simple" (Stitch Fix, Lyft eng tagged): a single page with
  infinite scroll; harvest detail-page links and follow.

Output
------
JSONL rows of the form
  {"arxiv_id": "...", "title": null|str, "org_label": "Roblox",
   "source_url": "https://about.roblox.com/publications/...", "via": "primary"}

Plus a `_completed: true` sentinel per org for resume.

Usage
-----
    uv run --with playwright --with loguru --with httpx \\
        python throwaway_script/playwright_user_specified_orgs.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from playwright.async_api import (
    BrowserContext,
    Page,
    TimeoutError as PWTimeoutError,
    async_playwright,
)


# arxiv IDs: YYMM.NNNNN (4-5 digit suffix). yy 07-26, mm 01-12.
ARXIV_ID_RE = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\w.])")
ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})")


def _valid_arxiv_id(aid: str) -> bool:
    try:
        yy = int(aid[:2]); mm = int(aid[2:4])
    except (ValueError, IndexError):
        return False
    if not (1 <= mm <= 12):
        return False
    return 7 <= yy <= 26


def harvest_arxiv_from_text(text: str) -> set[str]:
    out: set[str] = set()
    for m in ARXIV_URL_RE.finditer(text):
        aid = m.group(1)
        if _valid_arxiv_id(aid):
            out.add(aid)
    # Also any bare YYMM.NNNNN — but only when arxiv-context-y to reduce
    # false positives from things like phone numbers or revision dates.
    if "arxiv" in text.lower():
        for m in ARXIV_ID_RE.finditer(text):
            aid = m.group(1)
            if _valid_arxiv_id(aid):
                out.add(aid)
    return out


# ---------------------------------------------------- shared helpers


async def goto_safe(page: Page, url: str, *, timeout: int = 45_000, retries: int = 1) -> bool:
    for attempt in range(retries + 1):
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            if resp is None:
                return False
            if resp.status == 429:
                wait_s = 4 + attempt * 4
                logger.warning("  {} → HTTP 429, backing off {}s", url, wait_s)
                await asyncio.sleep(wait_s)
                continue
            if resp.status >= 400:
                logger.warning("  {} → HTTP {}", url, resp.status)
                return False
            return True
        except PWTimeoutError:
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return False
        except Exception as e:
            logger.warning("  goto failed {}: {!s:.160} (attempt {})", url, e, attempt + 1)
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return False
    return False


async def autoscroll(page: Page, n: int = 30, dwell_ms: int = 700) -> None:
    prev = -1
    stable = 0
    for _ in range(n):
        try:
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        except Exception:
            break
        await page.wait_for_timeout(dwell_ms)
        # Click load-more variants
        for sel in (
            "button:has-text('Load More')",
            "button:has-text('Load more')",
            "button:has-text('Show more')",
            "button:has-text('See more')",
            "a:has-text('Load more')",
        ):
            try:
                loc = page.locator(sel).first
                if await loc.is_visible(timeout=200):
                    await loc.click(timeout=1500, force=True)
                    await page.wait_for_timeout(900)
            except Exception:
                pass
        try:
            h = await page.evaluate("() => document.body.scrollHeight")
        except Exception:
            break
        if h == prev:
            stable += 1
            if stable >= 3:
                return
        else:
            stable = 0
            prev = h


async def follow_paper_links(
    page: Page,
    hrefs: list[str],
    *,
    label_for_log: str = "",
    sleep_between_ms: int = 600,
) -> dict[str, str]:
    """For each href, navigate + harvest arxiv. Returns {arxiv_id: title or placeholder}."""
    out: dict[str, str] = {}
    for i, href in enumerate(hrefs):
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        ok = await goto_safe(page, href, timeout=25_000, retries=1)
        if not ok:
            continue
        await page.wait_for_timeout(sleep_between_ms)
        # Mine arxiv from page content (HTML; covers <a href> AND raw text)
        try:
            content = await page.content()
        except Exception:
            content = ""
        ids = harvest_arxiv_from_text(content)
        title: Optional[str] = None
        if ids:
            try:
                title = await page.evaluate(
                    "() => { const h = document.querySelector('h1'); return h ? h.innerText.trim() : null; }"
                )
            except Exception:
                title = None
            for aid in ids:
                if aid not in out and title:
                    out[aid] = title[:280]
                else:
                    out.setdefault(aid, f"arxiv:{aid}")
        if (i + 1) % 25 == 0:
            logger.info(
                "    {} per-paper progress: {}/{} ({} total arxiv ids)",
                label_for_log, i + 1, len(hrefs), len(out),
            )
    return out


# ---------------------------------------------------- per-org scrapers


async def scrape_roblox(page: Page) -> dict[str, str]:
    """Roblox: click numbered pagination 1..11 on about.roblox.com/publications."""
    base = "https://about.roblox.com/publications"
    if not await goto_safe(page, base, timeout=60_000):
        return {}
    await page.wait_for_timeout(3000)

    detail_urls: set[str] = set()

    for p_num in range(1, 12):
        await page.wait_for_timeout(1000)
        try:
            hrefs = await page.eval_on_selector_all(
                "a[href*='/publications/']",
                "ns => ns.map(n => n.href)",
            )
        except Exception:
            hrefs = []
        # Keep only the EN /publications/<slug> form, drop locale paths
        for h in hrefs:
            if "/publications/" not in h:
                continue
            tail = h.split("/publications/", 1)[1]
            if not tail or "/" in tail.strip("/"):
                continue
            # Drop /<lang>/publications/ variants (ar, ja, zh-hans, etc.)
            if re.search(r"/[a-z]{2}(?:-[a-z]+)?/publications", h):
                continue
            detail_urls.add(h.split("#")[0].split("?")[0])
        logger.info("  roblox page {} → cumulative {} detail URLs", p_num, len(detail_urls))

        if p_num < 11:
            target = str(p_num + 1)
            try:
                # Click the pagination button whose text is exactly the next page number.
                # The Roblox SPA uses <button>N</button> in the bottom pagination row.
                locs = page.locator(f"button:has-text('{target}')")
                count = await locs.count()
                clicked = False
                for k in range(count):
                    btn = locs.nth(k)
                    txt = (await btn.text_content() or "").strip()
                    if txt == target:
                        try:
                            await btn.scroll_into_view_if_needed(timeout=2000)
                            await btn.click(timeout=3000, force=True)
                            await page.wait_for_timeout(2500)
                            clicked = True
                            break
                        except Exception:
                            continue
                if not clicked:
                    logger.warning("  roblox: couldn't click page {} button", target)
                    break
            except Exception as e:
                logger.warning("  roblox click {} failed: {}", target, e)
                break

    logger.info("  roblox: {} detail pages to follow", len(detail_urls))
    detail_list = sorted(detail_urls)
    return await follow_paper_links(page, detail_list, label_for_log="roblox", sleep_between_ms=900)


async def scrape_spotify_contentful(http_client: httpx.AsyncClient) -> dict[str, str]:
    """Spotify: walk /api/entries?content_type=publication, ~171 items."""
    out: dict[str, str] = {}
    base = "https://research.atspotify.com/api/entries"
    offset = 0
    limit = 100
    while True:
        params = {"content_type": "publication", "limit": limit, "skip": offset}
        try:
            r = await http_client.get(base, params=params, timeout=20.0)
        except Exception as e:
            logger.warning("  spotify api fail: {}", e)
            break
        if r.status_code != 200:
            logger.warning("  spotify api HTTP {}", r.status_code)
            break
        try:
            j = r.json()
        except Exception:
            break
        items = j.get("items", [])
        if not items:
            break
        for it in items:
            f = it.get("fields", {})
            title = f.get("title", "")
            text_blob = json.dumps(f, ensure_ascii=False)
            for aid in harvest_arxiv_from_text(text_blob):
                if aid not in out and title:
                    out[aid] = title[:280]
                else:
                    out.setdefault(aid, f"arxiv:{aid}")
        if len(items) < limit:
            break
        offset += limit
    logger.info("  spotify: {} arxiv IDs from Contentful API walk", len(out))
    return out


async def scrape_snap_bundle(http_client: httpx.AsyncClient) -> dict[str, str]:
    """Snap: fetch the publications-lib.js bundle and grep arxiv."""
    out: dict[str, str] = {}
    try:
        r = await http_client.get(
            "https://research.snap.com/assets/js/publications-lib.js",
            timeout=30.0,
        )
    except Exception as e:
        logger.warning("  snap bundle fetch failed: {}", e)
        return out
    if r.status_code != 200:
        logger.warning("  snap bundle HTTP {}", r.status_code)
        return out
    for aid in harvest_arxiv_from_text(r.text):
        out[aid] = f"arxiv:{aid}"
    logger.info("  snap js bundle → {} unique arxiv IDs", len(out))
    return out


async def scrape_uber(page: Page, max_pages: int = 30) -> dict[str, str]:
    """Uber: walk /us/en/blog/engineering/page/N/ for up to max_pages pages,
    collect post URLs, then follow each to mine arxiv from the post body."""
    all_posts: set[str] = set()
    CATEGORY_SLUGS = (
        "engineering", "research", "advertising", "business",
        "community-support", "ai-prototyping", "ai", "product", "safety",
        "rider", "driver", "eats", "transit", "delivery",
    )
    for p in range(1, max_pages + 1):
        url = (
            "https://www.uber.com/us/en/blog/engineering/"
            if p == 1
            else f"https://www.uber.com/us/en/blog/engineering/page/{p}/"
        )
        ok = await goto_safe(page, url)
        if not ok:
            logger.info("  uber: page {} non-200, stopping", p)
            break
        await page.wait_for_timeout(1500)
        try:
            hrefs = await page.eval_on_selector_all("a[href]", "ns => ns.map(n => n.href)")
        except Exception:
            hrefs = []
        new_posts = 0
        for h in hrefs:
            base = h.split("?")[0].split("#")[0]
            if "uber.com/" not in base or "/blog/" not in base:
                continue
            if "/page/" in base or base.endswith("/blog/"):
                continue
            # Reject category root URLs (.../blog/engineering/)
            if any(base.endswith(f"/blog/{c}/") for c in CATEGORY_SLUGS):
                continue
            if base not in all_posts:
                all_posts.add(base)
                new_posts += 1
        logger.info("  uber page {}: {} new, cumulative {}", p, new_posts, len(all_posts))
        if p > 2 and new_posts == 0:
            logger.info("  uber: page {} had no new posts, stopping", p)
            break
    logger.info("  uber: {} post URLs collected", len(all_posts))
    return await follow_paper_links(page, sorted(all_posts), label_for_log="uber", sleep_between_ms=500)


async def scrape_via_sitemap(
    page: Page,
    http_client: httpx.AsyncClient,
    sitemap_url: str,
    must_contain: Optional[str] = None,
    must_not_contain: tuple[str, ...] = (),
    cap: int = 400,
    label: str = "",
    use_playwright_for_posts: bool = True,
) -> dict[str, str]:
    """Bloomberg, Lyft, Instacart sitemaps. Fetch sitemap, extract <loc>,
    filter, then follow each URL (via playwright for JS sites; httpx for
    plain HTML) and mine arxiv.

    use_playwright_for_posts=False uses httpx for the per-post fetch (faster,
    but 403s on cloudflared sites).
    """
    out: dict[str, str] = {}
    # Fetch via httpx (sitemaps are typically not cloudflare-protected)
    try:
        r = await http_client.get(sitemap_url, timeout=30.0)
    except Exception as e:
        logger.warning("  {} sitemap fail: {}", label, e)
        return out
    if r.status_code != 200:
        logger.warning("  {} sitemap HTTP {}", label, r.status_code)
        return out
    locs = re.findall(r"<loc>([^<]+)</loc>", r.text)
    # Follow sub-sitemap if this is an index
    flat: set[str] = set()
    for u in locs:
        u = u.strip()
        if u.endswith(".xml"):
            try:
                rr = await http_client.get(u, timeout=20.0)
                if rr.status_code == 200:
                    for u2 in re.findall(r"<loc>([^<]+)</loc>", rr.text):
                        flat.add(u2.strip())
            except Exception:
                continue
            continue
        flat.add(u)

    # Apply filters
    filtered: list[str] = []
    for u in flat:
        if must_contain and must_contain not in u:
            continue
        if any(bad in u for bad in must_not_contain):
            continue
        if u.endswith(".xml"):
            continue
        filtered.append(u)
    filtered = sorted(set(filtered))[:cap]
    logger.info("  {} sitemap: {} candidate URLs (after filter)", label, len(filtered))

    if use_playwright_for_posts:
        return await follow_paper_links(page, filtered, label_for_log=label, sleep_between_ms=400)
    else:
        # Faster httpx-based fetch (works when site isn't behind cloudflare for /blog/<slug>/)
        for i, u in enumerate(filtered):
            try:
                rr = await http_client.get(u, timeout=20.0)
            except Exception:
                continue
            if rr.status_code != 200:
                continue
            for aid in harvest_arxiv_from_text(rr.text):
                out.setdefault(aid, f"arxiv:{aid}")
            if (i + 1) % 50 == 0:
                logger.info("    {} httpx progress: {}/{} ({} ids)",
                            label, i + 1, len(filtered), len(out))
        return out


async def scrape_paypal_medium_rss(http_client: httpx.AsyncClient, page: Page) -> dict[str, str]:
    """PayPal: medium.com/paypal-tech.

    Medium blocks both Playwright and httpx for the post URLs (HTTP 403),
    so we can't fetch individual posts. BUT the RSS feed embeds full post
    bodies in <content:encoded><![CDATA[...]]></content:encoded>, which
    DOES include arxiv links when present. So we mine the RSS body directly.

    The RSS limit is 10 most-recent posts. To extend, we also try the
    archive endpoint /paypal-tech/archive/<year>/<month> which is publicly
    crawlable (returns HTML with post body excerpts but no full bodies).
    """
    out: dict[str, str] = {}
    # RSS bodies — usually 10 most recent posts with full content
    try:
        r = await http_client.get("https://medium.com/feed/paypal-tech", timeout=30.0)
        if r.status_code == 200:
            for aid in harvest_arxiv_from_text(r.text):
                out.setdefault(aid, f"arxiv:{aid}")
            logger.info("  paypal RSS body: {} arxiv ids", len(out))
    except Exception as e:
        logger.warning("  paypal RSS fail: {}", e)
    # Archive endpoint walk — broader but contains excerpts only
    archive_arxiv = 0
    for year in range(2015, 2027):
        for month in range(1, 13):
            url = f"https://medium.com/paypal-tech/archive/{year}/{month:02d}"
            try:
                r = await http_client.get(url, timeout=15.0)
                if r.status_code == 200:
                    for aid in harvest_arxiv_from_text(r.text):
                        if aid not in out:
                            out[aid] = f"arxiv:{aid}"
                            archive_arxiv += 1
            except Exception:
                continue
    logger.info("  paypal archive: +{} arxiv ids (total {})", archive_arxiv, len(out))
    return out


async def scrape_stitchfix(page: Page, http_client: httpx.AsyncClient) -> dict[str, str]:
    """Stitch Fix: walk /blog/page/N/ via httpx (server-rendered HTML, no JS
    needed), collect /blog/<YYYY>/<MM>/<DD>/<slug>/ posts, then mine arxiv
    via httpx too — multithreaded.stitchfix.com isn't bot-blocked.

    Yields ~251 posts at the date of writing. After mining, expect ~3-10
    arxiv IDs (many posts cite arxiv at least once).
    """
    base = "https://multithreaded.stitchfix.com"
    all_posts: set[str] = set()
    for p_num in range(1, 40):
        url = f"{base}/blog/" if p_num == 1 else f"{base}/blog/page/{p_num}/"
        try:
            r = await http_client.get(url, timeout=15.0)
        except Exception:
            break
        if r.status_code != 200:
            logger.info("  stitchfix: page {} stop ({})", p_num, r.status_code)
            break
        posts = re.findall(r'href="(/blog/\d{4}/\d{2}/\d{2}/[^"]+)"', r.text)
        new_count = 0
        for path in posts:
            full = f"{base}{path}"
            if full not in all_posts:
                all_posts.add(full)
                new_count += 1
        logger.info("  stitchfix page {}: {} new (cumulative {})", p_num, new_count, len(all_posts))
        if p_num > 2 and new_count == 0:
            break
    logger.info("  stitchfix: {} posts to mine", len(all_posts))
    # Mine via httpx (much faster than Playwright; the site is plain HTML)
    out: dict[str, str] = {}
    for i, post in enumerate(sorted(all_posts)):
        try:
            rr = await http_client.get(post, timeout=15.0)
        except Exception:
            continue
        if rr.status_code != 200:
            continue
        for aid in harvest_arxiv_from_text(rr.text):
            out.setdefault(aid, f"arxiv:{aid}")
        if (i + 1) % 50 == 0:
            logger.info("    stitchfix httpx progress: {}/{} ({} ids)",
                        i + 1, len(all_posts), len(out))
    return out


async def scrape_lyft(page: Page, http_client: httpx.AsyncClient) -> dict[str, str]:
    """Lyft engineering blog (medium). Use both:
    - eng.lyft.com sitemap (508 entries)
    - eng.lyft.com listing (newest)
    Follow each post and mine arxiv.
    """
    # Sitemap gives a lot of post URLs but also lots of tag pages — filter.
    post_urls: set[str] = set()
    try:
        r = await http_client.get("https://eng.lyft.com/sitemap/sitemap.xml", timeout=30.0)
        if r.status_code == 200:
            for u in re.findall(r"<loc>([^<]+)</loc>", r.text):
                if u.endswith(".xml"):
                    try:
                        rr = await http_client.get(u.strip(), timeout=20.0)
                        if rr.status_code == 200:
                            for u2 in re.findall(r"<loc>([^<]+)</loc>", rr.text):
                                if "/tagged/" not in u2 and "eng.lyft.com" in u2 and u2 != "https://eng.lyft.com/":
                                    post_urls.add(u2.split("?")[0])
                    except Exception:
                        continue
                else:
                    if "/tagged/" not in u and "eng.lyft.com" in u and u != "https://eng.lyft.com/":
                        post_urls.add(u.split("?")[0])
    except Exception as e:
        logger.warning("  lyft sitemap fail: {}", e)
    logger.info("  lyft sitemap → {} candidate post URLs", len(post_urls))
    # cap to 300 most recent (medium sitemaps tend to list newest first; we
    # sort lexicographically here which is roughly chronological by Medium
    # URL hashes — accept some noise)
    post_urls_l = sorted(post_urls)[:300]
    return await follow_paper_links(page, post_urls_l, label_for_log="lyft", sleep_between_ms=500)


async def scrape_instacart(page: Page, http_client: httpx.AsyncClient) -> dict[str, str]:
    """Instacart tech blog via sitemap."""
    post_urls: set[str] = set()
    try:
        r = await http_client.get("https://tech.instacart.com/sitemap/sitemap.xml", timeout=30.0)
        if r.status_code == 200:
            for u in re.findall(r"<loc>([^<]+)</loc>", r.text):
                if u.endswith(".xml"):
                    try:
                        rr = await http_client.get(u.strip(), timeout=20.0)
                        if rr.status_code == 200:
                            for u2 in re.findall(r"<loc>([^<]+)</loc>", rr.text):
                                if "/tagged/" not in u2 and "tech.instacart.com" in u2 and u2 != "https://tech.instacart.com/":
                                    post_urls.add(u2.split("?")[0])
                    except Exception:
                        continue
                else:
                    if "/tagged/" not in u and "tech.instacart.com" in u and u != "https://tech.instacart.com/":
                        post_urls.add(u.split("?")[0])
    except Exception as e:
        logger.warning("  instacart sitemap fail: {}", e)
    post_urls_l = sorted(post_urls)[:300]
    logger.info("  instacart: {} post URLs", len(post_urls_l))
    return await follow_paper_links(page, post_urls_l, label_for_log="instacart", sleep_between_ms=500)


async def scrape_ebay(page: Page) -> dict[str, str]:
    """eBay's innovation.ebayinc.com — the sitemap is cloudflare-blocked
    but the listing is JS-rendered and works through Playwright.
    Walk innovation.ebayinc.com/stories/ via infinite scroll, also try
    /tech/research/, then follow each story for arxiv.
    """
    all_posts: set[str] = set()
    for url in [
        "https://innovation.ebayinc.com/stories/",
        "https://innovation.ebayinc.com/tech/",
        "https://innovation.ebayinc.com/tech/ai-machine-learning/",
        "https://innovation.ebayinc.com/tech/research/",
    ]:
        ok = await goto_safe(page, url)
        if not ok:
            continue
        await page.wait_for_timeout(2500)
        await autoscroll(page, n=40, dwell_ms=900)
        try:
            hrefs = await page.eval_on_selector_all("a[href]", "ns => ns.map(n => n.href)")
        except Exception:
            hrefs = []
        for h in hrefs:
            base = h.split("?")[0].split("#")[0]
            if "innovation.ebayinc.com/" not in base:
                continue
            # Story posts: stories/<slug>, tech/<topic>/<slug>, or post/<slug>
            if base.endswith("/stories/") or base.endswith("/tech/"):
                continue
            if not (
                "/stories/" in base
                or "/tech/" in base
                or "/post/" in base
            ):
                continue
            # Filter out category landing pages — they end with /<topic>/
            # Heuristic: a real post slug contains 3+ words separated by '-'
            tail = base.rstrip("/").rsplit("/", 1)[-1]
            if tail.count("-") < 2:
                continue
            all_posts.add(base)
        logger.info("  ebay {}: cumulative {} posts", url, len(all_posts))
    posts_l = sorted(all_posts)[:300]
    return await follow_paper_links(page, posts_l, label_for_log="ebay", sleep_between_ms=500)


def _gh_token() -> Optional[str]:
    """Use `gh auth token` if available, else GITHUB_TOKEN env."""
    import os, subprocess
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None


async def scrape_bloomberg(http_client: httpx.AsyncClient) -> dict[str, str]:
    """Bloomberg techatbloomberg.com is hard-403 even for blog posts via
    Playwright (Cloudflare bot challenge). Fall back to:
      1. github.com/bloomberg public repos — mine arxiv from each README.
      2. github.com/bloomberg PR/issue trackers occasionally cite papers.

    Empirically the GitHub route yields 2 IDs (koan 2012.15332, scatteract
    1704.06687) plus whatever is in the long tail of 216 repos.

    Uses `gh auth token` to bypass the 60/hr anonymous rate limit
    (authenticated is 5000/hr).
    """
    out: dict[str, str] = {}
    token = _gh_token()
    gh_headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Mozilla/5.0",
    }
    raw_headers = {
        "Accept": "application/vnd.github.raw",
        "User-Agent": "Mozilla/5.0",
    }
    if token:
        gh_headers["Authorization"] = f"Bearer {token}"
        raw_headers["Authorization"] = f"Bearer {token}"
        logger.info("  bloomberg: using gh auth token")
    else:
        logger.warning("  bloomberg: NO gh auth token — will rate-limit fast")

    # Walk all public repos
    repos: list[dict] = []
    page = 1
    while True:
        try:
            r = await http_client.get(
                f"https://api.github.com/orgs/bloomberg/repos?per_page=100&page={page}",
                headers=gh_headers,
                timeout=20.0,
            )
        except Exception as e:
            logger.warning("  bloomberg github api fail: {}", e)
            break
        if r.status_code != 200:
            logger.warning("  bloomberg github HTTP {}", r.status_code)
            break
        try:
            data = r.json()
        except Exception:
            break
        if not isinstance(data, list) or not data:
            break
        repos.extend(data)
        if len(data) < 100:
            break
        page += 1
    logger.info("  bloomberg github: {} repos to scan", len(repos))

    sem = asyncio.Semaphore(8)

    async def scan_repo(repo: dict) -> set[str]:
        name = repo.get("name")
        if not name:
            return set()
        desc = repo.get("description") or ""
        async with sem:
            try:
                rr = await http_client.get(
                    f"https://api.github.com/repos/bloomberg/{name}/readme",
                    headers=raw_headers,
                    timeout=15.0,
                )
            except Exception:
                return set()
            text = rr.text if rr.status_code == 200 else ""
            return harvest_arxiv_from_text(text + " " + desc)

    results = await asyncio.gather(*(scan_repo(r) for r in repos), return_exceptions=True)
    for r in results:
        if isinstance(r, set):
            for aid in r:
                out[aid] = f"arxiv:{aid}"
    logger.info("  bloomberg github → {} arxiv ids", len(out))
    return out


# ---------------------------------------------------- IO helpers


def load_existing_ids(*paths: Path) -> set[str]:
    seen: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            aid = r.get("arxiv_id")
            if aid:
                seen.add(aid)
    return seen


def load_completed_orgs(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("_completed"):
            done.add(r.get("org_label", ""))
    return done


# ---------------------------------------------------- driver


ORG_ORDER = [
    "Roblox",
    "Snap",
    "Spotify",
    "Bloomberg",
    "Lyft",
    "Instacart",
    "Uber",
    "StitchFix",
    "eBay",
    "PayPal",
]


async def run_org(
    label: str,
    page: Page,
    http_client: httpx.AsyncClient,
) -> tuple[dict[str, str], str]:
    """Returns (arxiv_id -> title, source_url for this org)."""
    if label == "Roblox":
        return await scrape_roblox(page), "https://about.roblox.com/publications"
    if label == "Spotify":
        return await scrape_spotify_contentful(http_client), "https://research.atspotify.com/api/entries"
    if label == "Snap":
        return await scrape_snap_bundle(http_client), "https://research.snap.com/assets/js/publications-lib.js"
    if label == "Uber":
        return await scrape_uber(page), "https://www.uber.com/blog/engineering/"
    if label == "PayPal":
        return await scrape_paypal_medium_rss(http_client, page), "https://medium.com/paypal-tech"
    if label == "StitchFix":
        return await scrape_stitchfix(page, http_client), "https://multithreaded.stitchfix.com/algorithms/"
    if label == "Lyft":
        return await scrape_lyft(page, http_client), "https://eng.lyft.com/"
    if label == "Instacart":
        return await scrape_instacart(page, http_client), "https://tech.instacart.com/"
    if label == "eBay":
        return await scrape_ebay(page), "https://innovation.ebayinc.com/"
    if label == "Bloomberg":
        return await scrape_bloomberg(http_client), "https://github.com/bloomberg"
    raise ValueError(f"unknown org: {label}")


async def run(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Dedup baseline: prior 3 outputs + own prior run
    baseline_ids = load_existing_ids(
        args.main_output,
        args.crack_output,
        args.extra_output,
        args.output,
    )
    logger.info("dedup baseline: {} arxiv_ids", len(baseline_ids))
    done = load_completed_orgs(args.output)
    logger.info("already-completed orgs: {}", sorted(done))

    if args.only:
        selected = [o for o in ORG_ORDER if o in set(args.only)]
    else:
        selected = [o for o in ORG_ORDER if o not in done] if not args.force else list(ORG_ORDER)
    if not selected:
        logger.warning("nothing to do — pass --force or --only")
        return 0
    logger.info("running {} orgs: {}", len(selected), selected)

    out_fh = args.output.open("a", encoding="utf-8")
    per_org_stats: list[dict] = []

    http_client = httpx.AsyncClient(
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
        verify=False,  # the company-scraper monorepo has internal CA issues on a few hosts
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context: BrowserContext = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 2000},
            locale="en-US",
            ignore_https_errors=True,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
            },
        )

        async def block_route(route):
            try:
                if route.request.resource_type in {"image", "media", "font"}:
                    await route.abort()
                else:
                    await route.continue_()
            except Exception:
                pass
        await context.route("**/*", block_route)
        page = await context.new_page()
        page.set_default_timeout(30_000)

        try:
            for label in selected:
                logger.info("=== {} ===", label)
                err_notes: list[str] = []
                ids_titles: dict[str, str] = {}
                source_url = ""
                try:
                    ids_titles, source_url = await run_org(label, page, http_client)
                except Exception as e:
                    err_notes.append(f"{type(e).__name__}: {str(e)[:200]}")
                    logger.exception("run_org crashed for {}", label)

                new_emitted = 0
                for aid in sorted(ids_titles):
                    if aid in baseline_ids:
                        continue
                    title = ids_titles[aid]
                    if isinstance(title, str) and title.startswith("arxiv:"):
                        title_out = None
                    else:
                        title_out = title
                    rec = {
                        "arxiv_id": aid,
                        "title": title_out,
                        "org_label": label,
                        "source_url": source_url,
                        "via": "primary",
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    baseline_ids.add(aid)
                    new_emitted += 1
                out_fh.flush()

                stat = {
                    "org": label,
                    "total_unique": len(ids_titles),
                    "new_emitted": new_emitted,
                    "errors": err_notes,
                }
                per_org_stats.append(stat)
                logger.info(
                    "[{}] DONE found={} NEW={}",
                    label, stat["total_unique"], stat["new_emitted"],
                )

                out_fh.write(json.dumps({
                    "_completed": True,
                    "org_label": label,
                    "total_unique": stat["total_unique"],
                    "new_emitted": stat["new_emitted"],
                    "errors": err_notes,
                }, ensure_ascii=False) + "\n")
                out_fh.flush()
                await asyncio.sleep(args.sleep_between_orgs)
        finally:
            try:
                await browser.close()
            except Exception:
                pass

    await http_client.aclose()
    out_fh.close()

    print("\n=== per-org stats (sorted by NEW desc) ===")
    print(f"{'org':<14s} {'unique':>8s} {'new':>6s}  errors")
    per_org_stats.sort(key=lambda s: -s["new_emitted"])
    for s in per_org_stats:
        errs = "; ".join(s["errors"])[:110]
        print(f"{s['org']:<14s} {s['total_unique']:8d} {s['new_emitted']:6d}  {errs}")
    total_new = sum(s["new_emitted"] for s in per_org_stats)
    print(f"\nTOTAL NET-NEW arxiv_ids: {total_new}")
    print(f"output: {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path,
                    default=Path("local_data/playwright_user_specified_orgs.jsonl"))
    ap.add_argument("--main-output", type=Path,
                    default=Path("local_data/playwright_company_pubs.jsonl"))
    ap.add_argument("--crack-output", type=Path,
                    default=Path("local_data/playwright_crack_failed.jsonl"))
    ap.add_argument("--extra-output", type=Path,
                    default=Path("local_data/playwright_extra_companies.jsonl"))
    ap.add_argument("--only",
                    type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-run orgs already marked _completed")
    ap.add_argument("--sleep-between-orgs", type=float, default=1.5)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
