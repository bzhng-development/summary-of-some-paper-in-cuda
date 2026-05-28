"""Exhaustively scrape every tracked AI company's publication pages + HF org pages.

Why this v3 exists:
  - Firecrawl floor: 520 unique arxiv IDs across 32 sources.
  - The HF `/papers?org=<slug>` URL filter is **broken** — it ignores `org=`
    and just returns the latest 54 trending papers across all of HF, so we
    cannot use that. Confirmed empirically against 6 different org slugs.
  - The CORRECT HF approach: enumerate the org's models via the public
    `https://huggingface.co/api/models?author=<slug>` API, then mine each
    model's README for arxiv IDs (model cards almost universally cite
    arxiv).
  - Many primary URLs don't link to arxiv directly — they link to in-house
    paper detail pages (Apple has 1000+ such links). Follow them.

Two scraping surfaces per org:
  1. PRIMARY URL — the org's own publication/research/news page.
     a. ``simple`` — single page, scroll-to-exhaustion, harvest arxiv.
     b. ``paginated`` — walk page-N URLs.
     c. ``sitemap`` — fetch sitemap, follow each leaf.
     d. After harvesting from list page, optionally follow internal paper
        detail links (``follow_paper_links=True``) and harvest from each.
  2. HF MODELS — enumerate the org's HF models via the public API, fetch
     each model README, and harvest arxiv IDs from the readme markdown.

Resume-safe via per-org ``_completed`` sentinel rows in the JSONL.

Usage:
    uv run --with playwright --with loguru --with httpx \\
        python throwaway_script/playwright_pub_scrape.py
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
    Page,
    TimeoutError as PWTimeoutError,
    async_playwright,
)


# arxiv IDs: YYMM.NNNNN (4-5 digit suffix).
ARXIV_ID_RE = re.compile(r"(?<![\w.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\w.])")


def _valid_arxiv_id(aid: str) -> bool:
    """arxiv YYMM.NNNNN sanity check — yy 07..26, mm 01..12."""
    try:
        yy = int(aid[:2])
        mm = int(aid[2:4])
    except (ValueError, IndexError):
        return False
    if not (1 <= mm <= 12):
        return False
    return 7 <= yy <= 26


@dataclass
class OrgPub:
    label: str
    primary_url: Optional[str] = None
    hf_orgs: tuple[str, ...] = ()
    # primary-page mode: "simple" | "paginated" | "sitemap" | "none"
    mode: str = "simple"
    wait_selector: Optional[str] = "body"
    max_scrolls: int = 60
    scroll_dwell_ms: int = 800
    load_more_selectors: tuple[str, ...] = field(default_factory=tuple)
    # paginated:
    page_url_template: Optional[str] = None
    max_page: int = 1
    # paginated/sitemap/simple: follow per-paper detail links to mine arxiv.
    follow_paper_links: bool = False
    # CSS selector for per-paper links on the list page.
    paper_link_selector: Optional[str] = None
    # href substring filter for per-paper links.
    paper_link_must_contain: Optional[str] = None
    # href substring filter to exclude (e.g. exclude "/research/team/").
    paper_link_must_not_contain: tuple[str, ...] = field(default_factory=tuple)
    max_paper_links: int = 400
    # sitemap:
    sitemap_urls: tuple[str, ...] = field(default_factory=tuple)
    # HF org model lookup tuning:
    hf_max_models: int = 500  # cap on models to enumerate per org
    skip_hf_models: bool = False  # set True to skip HF entirely
    hf_concurrency: int = 6  # concurrent readme fetches


# Buttons (only buttons; never anchors — paper titles like "Less is More"
# or "Show More: A Story Of..." would cause us to navigate away).
DEFAULT_LOAD_MORE = (
    "button:has-text('Load more')",
    "button:has-text('Show more')",
    "button:has-text('Load More')",
    "button:has-text('Show More')",
    "button:has-text('LOAD MORE')",
    "button:has-text('View more')",
    "button:has-text('See more')",
    "button[aria-label*='load more' i]",
    "button[aria-label*='show more' i]",
    "[class*='LoadMore' i] button",
    "[class*='loadMore' i] button",
    "[class*='load-more' i] button",
    "button[data-test*='load-more' i]",
    "button[data-testid*='load-more' i]",
)


# --------------------------------------------------------------------- ORG LIST
ORGS: tuple[OrgPub, ...] = (
    # --- Western LLM labs ---
    OrgPub(
        label="DeepMind",
        primary_url="https://deepmind.google/research/publications/",
        hf_orgs=("deepmind",),  # NOTE: the actual HF slug is "deepmind"; "google-deepmind" does NOT exist
        mode="paginated",
        page_url_template="https://deepmind.google/research/publications/page/{p}/",
        max_page=15,
        wait_selector="main",
        follow_paper_links=True,
        paper_link_selector="a[href*='/research/publications/']",
        paper_link_must_contain="/research/publications/",
        max_paper_links=600,
    ),
    OrgPub(
        label="OpenAI",
        primary_url="https://openai.com/research/",
        hf_orgs=("openai",),
        mode="sitemap",
        sitemap_urls=("https://openai.com/sitemap.xml/publication/",),
        paper_link_must_contain="openai.com/index/",
        max_paper_links=400,
    ),
    OrgPub(
        label="Anthropic",
        primary_url="https://www.anthropic.com/research",
        hf_orgs=("Anthropic",),
        mode="simple",
        max_scrolls=80,
        follow_paper_links=True,
        paper_link_selector="a[href*='/research/']",
        paper_link_must_contain="anthropic.com/research/",
        paper_link_must_not_contain=("/research/team",),
        max_paper_links=200,
    ),
    OrgPub(
        label="Meta-FAIR",
        primary_url="https://ai.meta.com/research/publications/",
        hf_orgs=("facebook", "meta-llama"),
        mode="simple",  # ai.meta.com sitemap returns a Facebook login HTML, can't use it
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        # No follow_paper_links — the listing page has 0 detail links on render.
    ),
    OrgPub(
        label="Microsoft-Research",
        primary_url="https://www.microsoft.com/en-us/research/publications/",
        hf_orgs=("microsoft",),
        mode="sitemap",
        sitemap_urls=(
            "https://www.microsoft.com/en-us/research/msr-research-item-sitemap.xml",
        ),
        paper_link_must_contain="/research/publication/",
        max_paper_links=300,  # cap; sitemap has 1000+ which would be too slow
    ),
    OrgPub(
        label="NVIDIA",
        primary_url="https://research.nvidia.com/publications",
        hf_orgs=("nvidia",),
        mode="simple",
        max_scrolls=100,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='publication']",
        paper_link_must_contain="research.nvidia.com",
        paper_link_must_not_contain=("research.nvidia.com/people/", "research.nvidia.com/labs/"),
        max_paper_links=400,
    ),
    OrgPub(
        label="AllenAI",
        primary_url="https://allenai.org/papers",
        hf_orgs=("allenai",),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        # semanticscholar.org/paper/ returns HTTP 405 to our UA; skip follow.
    ),
    OrgPub(
        label="Cohere",
        primary_url="https://cohere.com/research",
        hf_orgs=("CohereLabs", "CohereForAI"),
        mode="simple",
        max_scrolls=60,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='/research/']",
        paper_link_must_contain="cohere.com/research/",
        paper_link_must_not_contain=("/research/team",),
        max_paper_links=150,
    ),
    OrgPub(
        label="Mistral",
        primary_url="https://mistral.ai/news/",
        hf_orgs=("mistralai",),
        mode="simple",
        max_scrolls=60,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='/news/']",
        paper_link_must_contain="mistral.ai/news/",
        max_paper_links=200,
    ),
    OrgPub(
        label="xAI",
        primary_url=None,  # x.ai/about has nothing useful
        hf_orgs=("xai-org",),
        mode="none",
    ),
    OrgPub(
        label="Apple-ML",
        primary_url="https://machinelearning.apple.com/research",
        hf_orgs=("apple",),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='/research/']",
        paper_link_must_contain="machinelearning.apple.com/research/",
        max_paper_links=600,
    ),
    OrgPub(
        label="IBM-Research",
        primary_url="https://research.ibm.com/publications",
        hf_orgs=("ibm-granite", "ibm"),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='/publications/']",
        paper_link_must_contain="research.ibm.com/publications/",
        max_paper_links=200,
    ),
    OrgPub(
        label="Salesforce",
        primary_url="https://www.salesforceairesearch.com/publications",
        hf_orgs=("Salesforce", "SalesforceFoundationModels"),
        mode="simple",
        max_scrolls=120,
        load_more_selectors=DEFAULT_LOAD_MORE,
    ),
    OrgPub(
        label="Together-AI",
        primary_url="https://www.together.ai/research",
        hf_orgs=("togethercomputer",),
        mode="simple",
        max_scrolls=60,
    ),
    OrgPub(
        label="Snowflake",
        primary_url="https://www.snowflake.com/en/product/ai/ai-research/publications/",
        hf_orgs=("Snowflake",),
        mode="simple",
        max_scrolls=60,
        load_more_selectors=DEFAULT_LOAD_MORE,
    ),
    OrgPub(
        label="Stability-AI",
        hf_orgs=("stabilityai",),
        mode="none",
    ),
    OrgPub(
        label="EleutherAI",
        primary_url="https://www.eleuther.ai/papers",
        hf_orgs=("EleutherAI",),
        mode="sitemap",
        sitemap_urls=("https://www.eleuther.ai/sitemap.xml",),
        paper_link_must_contain="/papers-blog/",
        max_paper_links=300,
        hf_max_models=500,
    ),
    OrgPub(
        label="LAION",
        primary_url="https://laion.ai/",
        hf_orgs=("laion",),
        mode="simple",
        max_scrolls=80,
        follow_paper_links=True,
        paper_link_selector="a[href*='/blog/']",
        paper_link_must_contain="laion.ai/blog/",
        max_paper_links=200,
    ),
    OrgPub(
        label="Inflection-AI",
        primary_url="https://inflection.ai/",
        hf_orgs=("inflection",),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="Character-AI",
        hf_orgs=("CharacterAI",),
        mode="none",
    ),
    OrgPub(
        label="AI21",
        hf_orgs=("ai21labs",),
        mode="none",
    ),
    # --- Chinese LLM labs ---
    OrgPub(
        label="Qwen-Alibaba",
        primary_url="https://qwenlm.github.io/",
        hf_orgs=("Qwen",),
        mode="simple",
        max_scrolls=80,
        follow_paper_links=True,
        paper_link_selector="a[href*='qwenlm.github.io/']",
        paper_link_must_contain="qwenlm.github.io/",
        paper_link_must_not_contain=("qwenlm.github.io/about", "qwenlm.github.io/categories"),
        max_paper_links=300,
    ),
    OrgPub(
        label="DeepSeek",
        primary_url="https://www.deepseek.com/",
        hf_orgs=("deepseek-ai",),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="ByteDance-Seed",
        primary_url="https://team.doubao.com/en/research",
        hf_orgs=("ByteDance-Seed", "ByteDance"),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
    ),
    OrgPub(
        label="Moonshot-Kimi",
        primary_url="https://www.moonshot.cn/",
        hf_orgs=("moonshotai",),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="Zhipu-GLM",
        primary_url="https://www.zhipuai.cn/en",
        hf_orgs=("zai-org", "THUDM"),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="MiniMax",
        primary_url="https://www.minimaxi.com/en/news",
        hf_orgs=("MiniMaxAI",),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
    ),
    OrgPub(
        label="Hunyuan-Tencent",
        primary_url="https://hunyuan.tencent.com/",
        hf_orgs=("tencent",),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="StepFun",
        hf_orgs=("stepfun-ai",),
        mode="none",
    ),
    OrgPub(
        label="01-AI",
        hf_orgs=("01-ai",),
        mode="none",
    ),
    OrgPub(
        label="Baichuan",
        hf_orgs=("baichuan-inc",),
        mode="none",
    ),
    OrgPub(
        label="LongCat-Meituan",
        hf_orgs=("meituan-longcat",),
        mode="none",
    ),
    OrgPub(
        label="Alibaba-DAMO",
        primary_url="https://damo.alibaba.com/about?language=en",
        hf_orgs=(),
        mode="simple",
        max_scrolls=20,
    ),
    OrgPub(
        label="Baidu-Research",
        primary_url="https://research.baidu.com/",
        hf_orgs=("PaddlePaddle",),
        mode="simple",
        max_scrolls=20,
    ),
    OrgPub(
        label="Tsinghua-THUDM",
        primary_url="https://nlp.csai.tsinghua.edu.cn/",
        hf_orgs=("THUDM",),
        mode="simple",
        max_scrolls=20,
    ),
    OrgPub(
        label="KAIST-Kakao",
        hf_orgs=("kakaobrain",),
        mode="none",
    ),
    OrgPub(
        label="LG-EXAONE",
        primary_url="https://www.lgresearch.ai/exaone",
        hf_orgs=("LGAI-EXAONE",),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="OpenBMB",
        primary_url="https://www.openbmb.cn/",
        hf_orgs=("openbmb",),
        mode="simple",
        max_scrolls=60,
    ),
)


# ------------------------------------------------------------- scroll utilities


async def autoscroll(page: Page, max_scrolls: int = 60, dwell_ms: int = 800,
                     load_more_selectors: tuple[str, ...] = ()) -> int:
    """Scroll-to-exhaustion. Returns # iterations executed.

    Guard: if a click navigates the page to a different URL, navigate
    back and continue with scrolling only.
    """
    prev_height = -1
    stable_iters = 0
    iters_done = 0
    start_url = page.url
    selectors = load_more_selectors or DEFAULT_LOAD_MORE
    skip_clicks = False
    for i in range(max_scrolls):
        iters_done = i + 1
        clicks = 0
        if not skip_clicks:
            for sel in selectors:
                try:
                    locs = page.locator(sel)
                    count = await locs.count()
                    for k in range(min(count, 3)):
                        try:
                            el = locs.nth(k)
                            if await el.is_visible(timeout=400):
                                await el.scroll_into_view_if_needed(timeout=1500)
                                await el.click(timeout=2000, force=True)
                                clicks += 1
                                await page.wait_for_timeout(900)
                                if page.url != start_url:
                                    logger.warning("  load-more clicked us off-page, reverting")
                                    try:
                                        await page.goto(start_url, wait_until="domcontentloaded", timeout=20_000)
                                        await page.wait_for_timeout(1000)
                                    except Exception:
                                        pass
                                    skip_clicks = True
                                    break
                        except Exception:
                            continue
                    if skip_clicks:
                        break
                except Exception:
                    continue
                if skip_clicks:
                    break
        try:
            await page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        except Exception:
            break
        await page.wait_for_timeout(dwell_ms)
        try:
            height = await page.evaluate("() => document.body.scrollHeight")
        except Exception:
            break
        if height == prev_height and clicks == 0:
            stable_iters += 1
            if stable_iters >= 5:
                return iters_done
        else:
            stable_iters = 0
            prev_height = height
    return iters_done


# ------------------------------------------------------- arxiv id harvesting


async def harvest_arxiv_from_page(page: Page) -> dict[str, str]:
    """Mine arxiv IDs from current DOM. Returns {arxiv_id: title_or_placeholder}."""
    out: dict[str, str] = {}
    try:
        rows = await page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href]'))
                  .map(a => ({href: a.href, text: (a.innerText || a.textContent || '').trim()}))
                  .filter(r => r.href && (
                      r.href.includes('arxiv.org/abs/') ||
                      r.href.includes('arxiv.org/pdf/') ||
                      r.href.includes('huggingface.co/papers/')
                  ))"""
        )
    except Exception:
        rows = []
    for r in rows:
        m = ARXIV_ID_RE.search(r["href"])
        if not m:
            continue
        aid = m.group(1)
        if not _valid_arxiv_id(aid):
            continue
        title = (r.get("text") or "").strip()
        if title and len(title) > 5:
            out[aid] = title[:280]
        else:
            out.setdefault(aid, f"arxiv:{aid}")
    try:
        full_text = await page.evaluate("() => (document.body && document.body.innerText) || ''")
    except Exception:
        full_text = ""
    for m in ARXIV_ID_RE.finditer(full_text):
        aid = m.group(1)
        if _valid_arxiv_id(aid):
            out.setdefault(aid, f"arxiv:{aid}")
    return out


async def page_title_h1(page: Page) -> Optional[str]:
    try:
        return await page.evaluate(
            "() => { const h = document.querySelector('h1'); return h ? h.innerText.trim() : null; }"
        )
    except Exception:
        return None


# ----------------------------------------------------------------- HTTP / fetch


async def goto_safe(page: Page, url: str, *, timeout: int = 45_000, retries: int = 2) -> bool:
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
            logger.warning("  goto timeout: {} (attempt {})", url, attempt + 1)
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return False
        except Exception as e:
            logger.warning("  goto failed {}: {!s:.200} (attempt {})", url, e, attempt + 1)
            if attempt < retries:
                await asyncio.sleep(2)
                continue
            return False
    return False


async def dismiss_cookies(page: Page) -> None:
    for sel in (
        "button:has-text('Accept')",
        "button:has-text('I agree')",
        "button:has-text('Got it')",
        "button:has-text('OK')",
        "button:has-text('Allow all')",
        "[aria-label*='accept' i]",
    ):
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=300):
                await btn.click(timeout=2000, force=True)
                await page.wait_for_timeout(400)
                return
        except Exception:
            continue


# ------------------------------------------------------- per-paper hop helpers


async def collect_paper_links(page: Page, selector: str, must_contain: Optional[str],
                              must_not_contain: tuple[str, ...] = ()) -> list[str]:
    try:
        hrefs = await page.eval_on_selector_all(selector, "nodes => nodes.map(n => n.href)")
    except Exception:
        return []
    cleaned, seen = [], set()
    for h in hrefs:
        if not h:
            continue
        if must_contain and must_contain not in h:
            continue
        if any(banned in h for banned in must_not_contain):
            continue
        if h.startswith("mailto:") or h.startswith("javascript:"):
            continue
        # Trim fragments + queries
        h_clean = h.split("#")[0].split("?")[0]
        if not h_clean:
            continue
        if h_clean in seen:
            continue
        seen.add(h_clean)
        cleaned.append(h_clean)
    return cleaned


async def harvest_per_paper_loop(page: Page, hrefs: list[str]) -> dict[str, str]:
    """For each href, navigate + harvest arxiv. Skip mailto / javascript."""
    all_ids: dict[str, str] = {}
    for i, href in enumerate(hrefs):
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        if not await goto_safe(page, href, timeout=25_000, retries=1):
            continue
        await page.wait_for_timeout(200)
        ids = await harvest_arxiv_from_page(page)
        if ids:
            if len(ids) == 1:
                h1 = await page_title_h1(page)
                if h1:
                    only_aid = next(iter(ids))
                    all_ids[only_aid] = h1
                else:
                    all_ids.update(ids)
            else:
                all_ids.update(ids)
        if (i + 1) % 25 == 0:
            logger.info("    per-paper progress: {}/{} ({} total ids)",
                        i + 1, len(hrefs), len(all_ids))
    return all_ids


# ----------------------------------------------------------------- mode handlers


async def scrape_simple(page: Page, org: OrgPub) -> dict[str, str]:
    assert org.primary_url
    if not await goto_safe(page, org.primary_url, timeout=60_000):
        return {}
    if org.wait_selector:
        try:
            await page.wait_for_selector(org.wait_selector, timeout=10_000)
        except PWTimeoutError:
            pass
    await dismiss_cookies(page)
    iters = await autoscroll(
        page, max_scrolls=org.max_scrolls,
        dwell_ms=org.scroll_dwell_ms,
        load_more_selectors=org.load_more_selectors,
    )
    logger.info("  autoscroll: {} iters", iters)
    ids = await harvest_arxiv_from_page(page)
    logger.info("  list-page arxiv: {} ids", len(ids))
    # If configured, follow per-paper links and union.
    if org.follow_paper_links and org.paper_link_selector:
        hrefs = await collect_paper_links(
            page, org.paper_link_selector,
            org.paper_link_must_contain, org.paper_link_must_not_contain,
        )
        hrefs = hrefs[: org.max_paper_links]
        logger.info("  following {} per-paper links", len(hrefs))
        more = await harvest_per_paper_loop(page, hrefs)
        ids.update(more)
    return ids


async def scrape_paginated(page: Page, org: OrgPub) -> dict[str, str]:
    all_ids: dict[str, str] = {}
    paper_hrefs: list[str] = []

    for p in range(1, org.max_page + 1):
        url = org.primary_url if p == 1 else org.page_url_template.format(p=p)
        ok = await goto_safe(page, url)
        if not ok:
            if p > 1:
                logger.info("  page {} failed → stop pagination", p)
                break
            continue
        if org.wait_selector:
            try:
                await page.wait_for_selector(org.wait_selector, timeout=10_000)
            except PWTimeoutError:
                pass
        await page.wait_for_timeout(1000)
        await autoscroll(page, max_scrolls=20, dwell_ms=600,
                         load_more_selectors=org.load_more_selectors)
        ids = await harvest_arxiv_from_page(page)
        all_ids.update(ids)
        if org.follow_paper_links and org.paper_link_selector:
            hrefs = await collect_paper_links(
                page, org.paper_link_selector,
                org.paper_link_must_contain, org.paper_link_must_not_contain,
            )
            paper_hrefs.extend(hrefs)
        logger.info("  page {}: +{} ids, total {} ids / {} hrefs",
                    p, len(ids), len(all_ids), len(paper_hrefs))

    if org.follow_paper_links and paper_hrefs:
        paper_hrefs = list(dict.fromkeys(paper_hrefs))[: org.max_paper_links]
        logger.info("  following {} per-paper pages", len(paper_hrefs))
        more = await harvest_per_paper_loop(page, paper_hrefs)
        all_ids.update(more)
    return all_ids


async def fetch_sitemap_urls(client: httpx.AsyncClient, sitemap_urls: list[str],
                              must_contain: Optional[str]) -> list[str]:
    """Fetch each sitemap.xml via httpx and extract <loc>...</loc> URLs.

    Playwright's page.content() returns empty for some XML responses
    (Microsoft blocks the Chromium UA). httpx with a curl-like UA works.
    """
    found: set[str] = set()
    for sm in sitemap_urls:
        try:
            r = await client.get(sm, timeout=30.0)
        except Exception as e:
            logger.warning("  sitemap {} failed: {}", sm, e)
            continue
        if r.status_code != 200:
            logger.warning("  sitemap {} → HTTP {}", sm, r.status_code)
            continue
        text = r.text
        loc_urls = re.findall(r"<loc>([^<]+)</loc>", text)
        # If this is a sitemap-index, follow sub-sitemaps one level deep.
        for u in loc_urls:
            u = u.strip()
            if u.endswith(".xml") and must_contain not in u:
                # Sub-sitemap; fetch and accumulate
                try:
                    rr = await client.get(u, timeout=30.0)
                    if rr.status_code == 200:
                        for u2 in re.findall(r"<loc>([^<]+)</loc>", rr.text):
                            u2 = u2.strip()
                            if must_contain and must_contain not in u2:
                                continue
                            if u2.endswith(".xml"):
                                continue
                            found.add(u2.rstrip("/"))
                except Exception:
                    continue
                continue
            if must_contain and must_contain not in u:
                continue
            if u.endswith(".xml"):
                continue
            found.add(u.rstrip("/"))
    return sorted(found)


async def scrape_sitemap(page: Page, org: OrgPub, http_client: httpx.AsyncClient) -> dict[str, str]:
    paper_urls = await fetch_sitemap_urls(http_client, list(org.sitemap_urls), org.paper_link_must_contain)
    logger.info("  sitemap: {} candidate paper URLs", len(paper_urls))
    paper_urls = paper_urls[: org.max_paper_links]
    if not paper_urls:
        return {}
    return await harvest_per_paper_loop(page, paper_urls)


async def scrape_primary(page: Page, org: OrgPub, http_client: httpx.AsyncClient) -> dict[str, str]:
    if org.mode == "none" or not org.primary_url:
        return {}
    try:
        if org.mode == "simple":
            return await scrape_simple(page, org)
        if org.mode == "paginated":
            return await scrape_paginated(page, org)
        if org.mode == "sitemap":
            return await scrape_sitemap(page, org, http_client)
    except Exception as e:
        logger.error("  primary scrape crashed: {!r}", e)
        return {}
    return {}


# ----------------------------------------- HF model-card scraper (the real one)


async def hf_list_models(client: httpx.AsyncClient, author: str, limit: int = 500,
                          max_429_retries: int = 4) -> list[str]:
    """Return model IDs for an HF org via the public /api/models endpoint."""
    out: list[str] = []
    params = {"author": author, "limit": limit, "full": "false"}
    for attempt in range(max_429_retries + 1):
        try:
            r = await client.get("https://huggingface.co/api/models", params=params, timeout=30.0)
        except Exception as e:
            logger.warning("  hf models api failed for {}: {}", author, e)
            return out
        if r.status_code == 429:
            wait_s = 8 + attempt * 8
            logger.warning("  hf models api 429 for {}, backing off {}s (attempt {}/{})",
                           author, wait_s, attempt + 1, max_429_retries + 1)
            await asyncio.sleep(wait_s)
            continue
        if r.status_code >= 400:
            logger.warning("  hf models api {} → HTTP {}", author, r.status_code)
            return out
        try:
            data = r.json()
        except Exception:
            return out
        if not isinstance(data, list):
            return out
        for m in data:
            if isinstance(m, dict) and "id" in m:
                out.append(m["id"])
        return out
    return out


async def hf_fetch_readme_arxiv(client: httpx.AsyncClient, model_id: str) -> set[str]:
    """Fetch a model's README.md and harvest arxiv IDs from it.

    Backs off on 429 with up to 2 retries; never blocks indefinitely.
    """
    ids: set[str] = set()
    for branch in ("main", "master"):
        url = f"https://huggingface.co/{model_id}/raw/{branch}/README.md"
        for attempt in range(3):
            try:
                r = await client.get(url, timeout=20.0)
            except Exception:
                break
            if r.status_code == 200:
                for m in ARXIV_ID_RE.finditer(r.text):
                    aid = m.group(1)
                    if _valid_arxiv_id(aid):
                        ids.add(aid)
                for m in re.finditer(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", r.text):
                    aid = m.group(1)
                    if _valid_arxiv_id(aid):
                        ids.add(aid)
                return ids
            if r.status_code == 429:
                await asyncio.sleep(2 + attempt * 2)
                continue
            break
    return ids


async def hf_fetch_model_card_metadata(client: httpx.AsyncClient, model_id: str) -> set[str]:
    """Read the model API record — often has arxiv: tags in YAML frontmatter."""
    ids: set[str] = set()
    url = f"https://huggingface.co/api/models/{model_id}?full=true"
    for attempt in range(3):
        try:
            r = await client.get(url, timeout=20.0)
        except Exception:
            return ids
        if r.status_code == 429:
            await asyncio.sleep(2 + attempt * 2)
            continue
        if r.status_code != 200:
            return ids
        try:
            data = r.json()
        except Exception:
            return ids
        for tag in data.get("tags", []) or []:
            if isinstance(tag, str) and tag.startswith("arxiv:"):
                aid = tag.split(":", 1)[1].strip()
                if _valid_arxiv_id(aid):
                    ids.add(aid)
        card = data.get("cardData") or {}
        if isinstance(card, dict):
            for key in ("arxiv",):
                v = card.get(key)
                if isinstance(v, str) and _valid_arxiv_id(v):
                    ids.add(v)
                elif isinstance(v, list):
                    for x in v:
                        if isinstance(x, str) and _valid_arxiv_id(x):
                            ids.add(x)
        return ids
    return ids


async def scrape_hf_org(client: httpx.AsyncClient, slug: str, max_models: int = 200,
                        concurrency: int = 8) -> dict[str, str]:
    """Enumerate HF org's models + datasets and harvest arxiv from each."""
    out: dict[str, str] = {}
    model_ids = await hf_list_models(client, slug, limit=max_models)
    logger.info("  hf:{} → {} models", slug, len(model_ids))
    if not model_ids:
        return out

    sem = asyncio.Semaphore(concurrency)

    async def worker(mid: str) -> set[str]:
        async with sem:
            try:
                # Tags API call first (fast and usually has the arxiv)
                tag_ids = await hf_fetch_model_card_metadata(client, mid)
                # README parse for IDs in body
                readme_ids = await hf_fetch_readme_arxiv(client, mid)
                return tag_ids | readme_ids
            except Exception as e:
                logger.debug("  hf:{}/{} failed: {}", slug, mid, e)
                return set()

    tasks = [worker(m) for m in model_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    union: set[str] = set()
    for r in results:
        if isinstance(r, set):
            union |= r
    for aid in union:
        out[aid] = f"arxiv:{aid}"  # title unknown from this path
    logger.info("  hf:{} → {} unique arxiv IDs from {} models", slug, len(union), len(model_ids))
    return out


# ---------------------------------------------------------------- IO + driver


async def load_existing(path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    seen: set[tuple[str, str]] = set()
    done: set[str] = set()
    if not path.is_file():
        return seen, done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("_completed"):
            done.add(r.get("org_label", ""))
            continue
        if "arxiv_id" in r and "org_label" in r:
            seen.add((r["org_label"], r["arxiv_id"]))
    return seen, done


async def run(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen, done_orgs = await load_existing(args.output)
    logger.info("resume: {} (org,arxiv) tuples / {} orgs completed", len(seen), len(done_orgs))

    if args.smoke_test:
        selected = [o for o in ORGS if o.label == args.smoke_test]
    elif args.only:
        selected = [o for o in ORGS if o.label in set(args.only)]
    else:
        selected = list(ORGS)
        if not args.force:
            selected = [o for o in selected if o.label not in done_orgs]

    if not selected:
        logger.warning("nothing to do — all orgs already completed; pass --force or --only")
        return 0

    logger.info("running {} orgs", len(selected))

    out_fh = args.output.open("a", encoding="utf-8")
    per_org_stats: list[dict] = []

    http_client = httpx.AsyncClient(
        headers={"User-Agent": "playwright-pub-scrape/1.0 (research)"},
        follow_redirects=True,
    )

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1440, "height": 2000},
            locale="en-US",
            ignore_https_errors=True,
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
            for org in selected:
                logger.info("=== {} (mode={}, primary={}) ===",
                            org.label, org.mode, org.primary_url or "—")
                primary_ids: dict[str, str] = {}
                hf_ids_per_slug: dict[str, dict[str, str]] = {}
                err_notes: list[str] = []

                try:
                    primary_ids = await scrape_primary(page, org, http_client)
                except Exception as e:
                    err_notes.append(f"primary: {type(e).__name__}: {str(e)[:160]}")
                    logger.exception("primary scrape outer-exception for {}", org.label)

                if not org.skip_hf_models:
                    for slug in org.hf_orgs:
                        try:
                            ids = await scrape_hf_org(http_client, slug,
                                                       max_models=org.hf_max_models,
                                                       concurrency=org.hf_concurrency)
                            hf_ids_per_slug[slug] = ids
                        except Exception as e:
                            err_notes.append(f"hf:{slug}: {type(e).__name__}: {str(e)[:160]}")
                            logger.exception("hf scrape outer-exception for {}/{}", org.label, slug)

                all_titles: dict[str, str] = {}
                all_ids: set[str] = set()
                sources: dict[str, str] = {}
                via_map: dict[str, str] = {}
                for aid, title in primary_ids.items():
                    all_ids.add(aid)
                    if title and not title.startswith("arxiv:"):
                        all_titles[aid] = title
                    sources[aid] = org.primary_url or "—"
                    via_map[aid] = "primary"
                for slug, ids in hf_ids_per_slug.items():
                    for aid, title in ids.items():
                        all_ids.add(aid)
                        if aid not in all_titles and title and not title.startswith("arxiv:"):
                            all_titles[aid] = title
                        sources.setdefault(aid, f"https://huggingface.co/{slug}")
                        via_map.setdefault(aid, "hf_models")

                new_emitted = 0
                for aid in sorted(all_ids):
                    key = (org.label, aid)
                    if key in seen:
                        continue
                    rec = {
                        "arxiv_id": aid,
                        "title": all_titles.get(aid),
                        "org_label": org.label,
                        "source_url": sources.get(aid, org.primary_url),
                        "via": via_map.get(aid, "unknown"),
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen.add(key)
                    new_emitted += 1
                out_fh.flush()

                stat = {
                    "org": org.label,
                    "primary_count": len(primary_ids),
                    "hf_count": sum(len(v) for v in hf_ids_per_slug.values()),
                    "total_unique": len(all_ids),
                    "new_emitted": new_emitted,
                    "errors": err_notes,
                }
                per_org_stats.append(stat)
                logger.info("[{}] DONE primary={} hf={} unique={} new={}",
                            org.label, stat["primary_count"], stat["hf_count"],
                            stat["total_unique"], stat["new_emitted"])

                out_fh.write(json.dumps({
                    "_completed": True,
                    "org_label": org.label,
                    "primary_count": stat["primary_count"],
                    "hf_count": stat["hf_count"],
                    "total_unique": stat["total_unique"],
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

    print("\n=== per-org stats ===")
    print(f"{'org':<22s} {'primary':>9s} {'hf':>6s} {'unique':>8s} {'new':>6s}  errors")
    grand_total: set[str] = set()
    for line in args.output.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("_completed"):
            continue
        if r.get("arxiv_id"):
            grand_total.add(r["arxiv_id"])
    for s in per_org_stats:
        errs = "; ".join(s["errors"])[:100]
        print(f"{s['org']:<22s} {s['primary_count']:9d} {s['hf_count']:6d} {s['total_unique']:8d} {s['new_emitted']:6d}  {errs}")
    print(f"\nGRAND TOTAL UNIQUE arxiv_ids: {len(grand_total)}")
    print(f"output: {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/playwright_company_pubs.jsonl"))
    ap.add_argument("--only", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    default=None)
    ap.add_argument("--smoke-test", type=str, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sleep-between-orgs", type=float, default=1.5)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
