"""Scrape arxiv IDs from the EXTRA 34 orgs from the original ToC list.

Coverage of this script:
  LLM gaps:    InternLM, HuggingFace-Smol (HF only), Amazon Science, Google
               Research, THUNLP, Rakuten RIT  (6 orgs)
  Tencent:     Tencent-AI-Lab broader (covers AI Lab + WeChat AI + Research,
               separate from existing Hunyuan-Tencent)  (1 org)
  US Univs:    Berkeley BAIR, Stanford NLP/SAIL, CMU LTI, UW NLP  (4 orgs)
  All-ML:      GitHub, Netflix, Spotify, TikTok, Airbnb, Uber, Lyft,
               DoorDash, Zillow, eBay, Zoom, PayPal, Pinterest, Slack,
               Snap, Stripe, Stitch Fix, X-Twitter, Dropbox, Mozilla,
               Roblox, Naver, Yandex  (23 orgs)
  TOTAL: 34 orgs. (YouTube folds into Google Research; Wayfair excluded
  per task brief.)

Strategy — copy-paste of throwaway_script/playwright_pub_scrape.py with:
  * fresh OrgPub list scoped to the gap orgs above
  * separate output JSONL: local_data/playwright_extra_companies.jsonl
  * dedup against BOTH the main 1812-ID dump AND the 58-ID crack file so
    we only emit NET-NEW arxiv IDs
  * skip-on-error per-org so a single 404 doesn't kill the run

We REUSE all utilities (autoscroll, harvest_arxiv_from_page, sitemap
fetch, hf_list_models, hf_fetch_readme_arxiv, hf_fetch_model_card_metadata,
scrape_hf_org) by importing the playwright_pub_scrape module — do NOT
mutate it.

Usage:
    uv run --with playwright --with loguru --with httpx \\
        python throwaway_script/playwright_extra_companies.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from playwright.async_api import async_playwright

# Reuse the workhorse utilities from the original scraper. NEVER mutate
# anything imported from it — read-only.
from playwright_pub_scrape import (  # type: ignore
    ARXIV_ID_RE,
    DEFAULT_LOAD_MORE,
    OrgPub,
    _valid_arxiv_id,
    scrape_primary,
    scrape_hf_org,
)


# ---------------------------------------------------------------- ORG LIST (30)
EXTRA_ORGS: tuple[OrgPub, ...] = (
    # ============ LLMs section (high-yield publishers) ============
    OrgPub(
        label="InternLM-Shanghai-AI-Lab",
        # internlm.github.io is 404; shlab.org.cn is the actual lab page.
        primary_url="https://www.shlab.org.cn/",
        hf_orgs=("internlm",),
        mode="simple",
        max_scrolls=40,
    ),
    OrgPub(
        label="HuggingFace-Smol",
        # HF-only — they don't run a separate publications page.
        hf_orgs=("HuggingFaceTB",),
        mode="none",
    ),
    OrgPub(
        label="Amazon-Science",
        # Paginated via ?p=N — 15 leaves per page; cap at 30 pages = 450 papers.
        primary_url="https://www.amazon.science/publications",
        hf_orgs=("amazon",),
        mode="paginated",
        page_url_template="https://www.amazon.science/publications?p={p}",
        max_page=30,
        wait_selector="main",
        follow_paper_links=True,
        paper_link_selector="a[href*='amazon.science/publications/']",
        paper_link_must_contain="amazon.science/publications/",
        max_paper_links=400,
    ),
    OrgPub(
        label="Google-Research",
        # Paginated via ?page=N — 15 leaves per page; cap at 30 pages = 450 recent papers.
        primary_url="https://research.google/pubs/",
        hf_orgs=("google",),
        mode="paginated",
        page_url_template="https://research.google/pubs/?page={p}",
        max_page=30,
        wait_selector="main",
        follow_paper_links=True,
        paper_link_selector="a[href*='research.google/pubs/']",
        paper_link_must_contain="research.google/pubs/",
        paper_link_must_not_contain=("research.google/pubs/?",),
        max_paper_links=500,
    ),
    OrgPub(
        label="THUNLP",
        primary_url="https://nlp.csai.tsinghua.edu.cn/",
        hf_orgs=("thunlp",),
        mode="simple",
        max_scrolls=40,
        follow_paper_links=True,
        paper_link_selector="a[href*='nlp.csai.tsinghua.edu.cn']",
        paper_link_must_contain="nlp.csai.tsinghua.edu.cn",
        paper_link_must_not_contain=("/people", "/members"),
        max_paper_links=100,
    ),
    OrgPub(
        label="Rakuten-RIT",
        primary_url="https://rit.rakuten.com/",
        hf_orgs=("Rakuten",),
        mode="simple",
        max_scrolls=40,
        follow_paper_links=True,
        paper_link_selector="a[href*='rit.rakuten.com']",
        paper_link_must_contain="rit.rakuten.com",
        max_paper_links=80,
    ),

    # ============ Tencent broader (separate from Hunyuan-Tencent) ============
    # Existing Hunyuan-Tencent covers hf:tencent + hunyuan.tencent.com. This
    # entry covers Tencent AI Lab + WeChat AI + research arms via:
    #   - ai.tencent.com/ailab/en/index (Tencent AI Lab landing)
    #   - ai.tencent.com/aitm/en/publication.html (Tencent's AI-in-Translation pubs)
    #   - hf:tencent-AILab (smaller AI Lab-specific HF namespace)
    # We do TWO primary URLs by handling the second URL inline below.
    # NOTE: hf:tencent overlaps with Hunyuan-Tencent; we skip it here. The
    # tencent-AILab slug is the non-Hunyuan AI Lab portfolio.
    OrgPub(
        label="Tencent-AI-Lab",
        primary_url="https://ai.tencent.com/ailab/en/index",
        hf_orgs=("tencent-AILab",),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='ai.tencent.com']",
        paper_link_must_contain="ai.tencent.com",
        paper_link_must_not_contain=("/people", "/team", "/about", "/contact"),
        max_paper_links=200,
    ),

    # ============ US universities ============
    # Cap detail-page hops at 300 to stay polite. Most pub pages have arxiv
    # links inline in the listing (Stanford NLP) but BAIR's blog requires
    # following each post; CMU LTI and UW NLP are heavyweight listings that
    # we filter to arxiv-link-only outbound hops.
    OrgPub(
        label="Berkeley-BAIR",
        primary_url="https://bair.berkeley.edu/blog/",
        hf_orgs=("berkeley-nest",),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='bair.berkeley.edu/blog/']",
        paper_link_must_contain="bair.berkeley.edu/blog/",
        paper_link_must_not_contain=("/page/", "/author/", "/tags/", "/category/"),
        max_paper_links=300,
    ),
    OrgPub(
        label="Stanford-NLP",
        # nlp.stanford.edu/pubs/ is a static HTML page with hundreds of
        # paper entries — arxiv links sit in the bibliography rows. No detail
        # hops needed. We DO follow_paper_links=True to chase arxiv-direct
        # links anywhere on the page just in case.
        primary_url="https://nlp.stanford.edu/pubs/",
        hf_orgs=("stanfordnlp", "stanford-crfm"),
        mode="simple",
        max_scrolls=20,
    ),
    OrgPub(
        label="CMU-LTI",
        # The original /research/projects/publications path 404s.
        # Use the LTI main page + /research entry — the JS-rendered nav has
        # links to faculty pages that DO cite arxiv. Also use the cmu-lti
        # HF slug.
        primary_url="https://www.lti.cs.cmu.edu/",
        hf_orgs=("cmu-lti",),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href]",
        paper_link_must_contain="arxiv.org/abs/",
        max_paper_links=200,
    ),
    OrgPub(
        label="UW-NLP",
        # The /area/nlp path 404s. Use the actual NLP group pages.
        primary_url="https://nlp.washington.edu/",
        hf_orgs=("uw-nlp",),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href]",
        paper_link_must_contain="arxiv.org/abs/",
        max_paper_links=200,
    ),

    # ============ All-other-ML section (engineering blogs) ============
    OrgPub(
        label="GitHub",
        primary_url="https://github.blog/news-insights/research/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='github.blog/']",
        paper_link_must_contain="github.blog/",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Netflix",
        primary_url="https://netflixtechblog.com/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=60,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='netflixtechblog.com/']",
        paper_link_must_contain="netflixtechblog.com/",
        paper_link_must_not_contain=("/about", "/tagged/", "?source="),
        max_paper_links=80,
    ),
    OrgPub(
        label="Spotify-Research",
        primary_url="https://research.atspotify.com/publications",
        hf_orgs=(),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='research.atspotify.com']",
        paper_link_must_contain="research.atspotify.com",
        paper_link_must_not_contain=("/about", "/people", "/team"),
        max_paper_links=150,
    ),
    OrgPub(
        label="TikTok",
        # /blog returns 404; the actual developer landing page links to
        # /blog/<slug> posts though, so scrape the root and follow those.
        primary_url="https://developers.tiktok.com",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='/blog/']",
        paper_link_must_contain="/blog/",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/", "/blogs"),
        max_paper_links=60,
    ),
    OrgPub(
        label="Airbnb",
        primary_url="https://airbnb.tech/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=60,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='airbnb.tech/']",
        paper_link_must_contain="airbnb.tech/",
        paper_link_must_not_contain=("?source=", "/tagged/"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Uber",
        primary_url="https://www.uber.com/blog/research/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=60,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='uber.com/blog/']",
        paper_link_must_contain="uber.com/blog/",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Lyft",
        primary_url="https://eng.lyft.com/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='eng.lyft.com/']",
        paper_link_must_contain="eng.lyft.com/",
        paper_link_must_not_contain=("?source=", "/tagged/"),
        max_paper_links=60,
    ),
    OrgPub(
        label="DoorDash",
        primary_url="https://careersatdoordash.com/engineering-blog/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='careersatdoordash.com']",
        paper_link_must_contain="careersatdoordash.com",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/"),
        max_paper_links=60,
    ),
    OrgPub(
        label="Zillow",
        primary_url="https://www.zillow.com/tech/ai-ml/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=30,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='zillow.com/tech/']",
        paper_link_must_contain="zillow.com/tech/",
        max_paper_links=40,
    ),
    OrgPub(
        label="eBay",
        primary_url="https://innovation.ebayinc.com/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='innovation.ebayinc.com/']",
        paper_link_must_contain="innovation.ebayinc.com/",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Zoom",
        primary_url="https://www.zoom.com/en/blog/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=30,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='zoom.com/en/blog/']",
        paper_link_must_contain="zoom.com/en/blog/",
        max_paper_links=50,
    ),
    OrgPub(
        label="PayPal",
        primary_url="https://developer.paypal.com/community/blog/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=30,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='developer.paypal.com/community/blog/']",
        paper_link_must_contain="developer.paypal.com/community/blog/",
        max_paper_links=50,
    ),
    OrgPub(
        label="Pinterest",
        primary_url="https://medium.com/pinterest-engineering",
        hf_orgs=(),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='medium.com/pinterest-engineering/']",
        paper_link_must_contain="medium.com/pinterest-engineering/",
        paper_link_must_not_contain=("?source=", "/tagged/", "/followers"),
        max_paper_links=100,
    ),
    OrgPub(
        label="Slack",
        primary_url="https://slack.engineering/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=50,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='slack.engineering/']",
        paper_link_must_contain="slack.engineering/",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Snap-Research",
        # The publications.html shell is JS-rendered + empty until the JS bundle
        # `publications-lib.js` loads + populates it. That bundle is literally
        # the dataset — has 200+ arxiv URLs hardcoded. We fetch it directly
        # (handled below in `harvest_snap_js_bundle`). primary_url=None → skips
        # the playwright primary scrape, then we do the bundle harvest.
        primary_url=None,
        hf_orgs=("Snapchat",),
        mode="none",
    ),
    OrgPub(
        label="Stripe",
        primary_url="https://stripe.com/blog/engineering",
        hf_orgs=(),
        mode="simple",
        max_scrolls=30,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='stripe.com/blog/']",
        paper_link_must_contain="stripe.com/blog/",
        max_paper_links=50,
    ),
    OrgPub(
        label="StitchFix",
        primary_url="https://multithreaded.stitchfix.com/algorithms/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='multithreaded.stitchfix.com/']",
        paper_link_must_contain="multithreaded.stitchfix.com/",
        max_paper_links=60,
    ),
    OrgPub(
        label="X-Twitter",
        primary_url="https://blog.x.com/engineering/en_us",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='blog.x.com/engineering/']",
        paper_link_must_contain="blog.x.com/engineering/",
        max_paper_links=60,
    ),
    OrgPub(
        label="Dropbox",
        primary_url="https://dropbox.tech/machine-learning",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='dropbox.tech/']",
        paper_link_must_contain="dropbox.tech/",
        paper_link_must_not_contain=("/category/", "/tag/", "/author/"),
        max_paper_links=60,
    ),
    OrgPub(
        label="Mozilla-Research",
        primary_url="https://research.mozilla.org/",
        hf_orgs=("Mozilla",),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='research.mozilla.org/']",
        paper_link_must_contain="research.mozilla.org/",
        paper_link_must_not_contain=("/people", "/team"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Roblox",
        primary_url="https://corp.roblox.com/newsroom/",
        hf_orgs=(),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='corp.roblox.com/newsroom/']",
        paper_link_must_contain="corp.roblox.com/newsroom/",
        max_paper_links=60,
    ),
    OrgPub(
        label="Naver-Clova",
        primary_url="https://clova.ai/en/ai-research",
        hf_orgs=("naver-clova-ix", "naver"),
        mode="simple",
        max_scrolls=40,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='clova.ai/']",
        paper_link_must_contain="clova.ai/",
        paper_link_must_not_contain=("/people", "/team", "/about"),
        max_paper_links=80,
    ),
    OrgPub(
        label="Yandex-Research",
        primary_url="https://research.yandex.com/publications",
        hf_orgs=("yandex",),
        mode="simple",
        max_scrolls=80,
        load_more_selectors=DEFAULT_LOAD_MORE,
        follow_paper_links=True,
        paper_link_selector="a[href*='research.yandex.com']",
        paper_link_must_contain="research.yandex.com",
        paper_link_must_not_contain=("/people", "/team", "/about"),
        max_paper_links=200,
    ),
)


# ------------------------------------------------ Snap publications JS bundle
async def harvest_snap_js_bundle(client: httpx.AsyncClient) -> dict[str, str]:
    """Snap's publications.html is an empty shell — the real data is the JS
    bundle at /assets/js/publications-lib.js. Fetch it and extract arxiv IDs
    from the inlined URLs (200+ arxiv.org/pdf/<id> entries hardcoded).
    """
    out: dict[str, str] = {}
    try:
        r = await client.get(
            "https://research.snap.com/assets/js/publications-lib.js",
            timeout=30.0,
        )
    except Exception as e:
        logger.warning("  snap js bundle fetch failed: {}", e)
        return out
    if r.status_code != 200:
        logger.warning("  snap js bundle → HTTP {}", r.status_code)
        return out
    text = r.text
    # arxiv.org/pdf/YYMM.NNNNN OR arxiv.org/abs/YYMM.NNNNN (with optional v\d+)
    for m in re.finditer(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", text):
        aid = m.group(1)
        if _valid_arxiv_id(aid):
            out[aid] = f"arxiv:{aid}"
    # Loose YYMM.NNNNN fallback in case the URL has unusual formatting
    for m in ARXIV_ID_RE.finditer(text):
        aid = m.group(1)
        if _valid_arxiv_id(aid):
            out.setdefault(aid, f"arxiv:{aid}")
    logger.info("  snap js-bundle harvest: {} unique arxiv IDs", len(out))
    return out


# --------------------------------------------------- IO helpers (dedup baseline)
def load_existing_ids(*paths: Path) -> set[str]:
    """Load all arxiv_ids from prior playwright outputs to seed dedup."""
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


# ----------------------------------------------------------------- main driver
async def run(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Dedup baseline: BOTH prior playwright files + our own output (if resuming)
    baseline_ids = load_existing_ids(
        args.main_output,
        args.crack_output,
        args.output,
    )
    logger.info("dedup baseline: {} arxiv_ids (main + crack + own prior runs)",
                len(baseline_ids))

    done_orgs = load_completed_orgs(args.output)
    logger.info("already-completed orgs in own output: {}", len(done_orgs))

    if args.smoke_test:
        names = set(args.smoke_test)
        selected = [o for o in EXTRA_ORGS if o.label in names]
    elif args.only:
        names = set(args.only)
        selected = [o for o in EXTRA_ORGS if o.label in names]
    else:
        selected = list(EXTRA_ORGS)
        if not args.force:
            selected = [o for o in selected if o.label not in done_orgs]

    if not selected:
        logger.warning("nothing to do — pass --force or --only")
        return 0

    logger.info("running {} orgs", len(selected))

    out_fh = args.output.open("a", encoding="utf-8")
    per_org_stats: list[dict] = []

    http_client = httpx.AsyncClient(
        headers={"User-Agent": "playwright-extra-companies/1.0 (research)"},
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

                # Org-specific custom data-source supplements.
                if org.label == "Snap-Research":
                    try:
                        snap_ids = await harvest_snap_js_bundle(http_client)
                        primary_ids.update(snap_ids)
                    except Exception as e:
                        err_notes.append(f"snap-bundle: {type(e).__name__}: {str(e)[:160]}")
                        logger.exception("snap js-bundle harvest failed")

                for slug in org.hf_orgs:
                    try:
                        ids = await scrape_hf_org(
                            http_client, slug,
                            max_models=org.hf_max_models,
                            concurrency=org.hf_concurrency,
                        )
                        hf_ids_per_slug[slug] = ids
                    except Exception as e:
                        err_notes.append(f"hf:{slug}: {type(e).__name__}: {str(e)[:160]}")
                        logger.exception("hf scrape outer-exception for {}/{}",
                                         org.label, slug)

                # Merge primary + hf with via attribution
                # Per-org override of primary source URL for "none"-mode orgs
                # that have a custom data path (Snap bundle).
                primary_source = org.primary_url
                if org.label == "Snap-Research":
                    primary_source = "https://research.snap.com/publications.html"

                all_titles: dict[str, str] = {}
                all_ids: set[str] = set()
                sources: dict[str, str] = {}
                via_map: dict[str, str] = {}
                for aid, title in primary_ids.items():
                    all_ids.add(aid)
                    if title and not title.startswith("arxiv:"):
                        all_titles[aid] = title
                    sources[aid] = primary_source or "—"
                    via_map[aid] = "primary"
                for slug, ids in hf_ids_per_slug.items():
                    for aid, title in ids.items():
                        all_ids.add(aid)
                        if aid not in all_titles and title and not title.startswith("arxiv:"):
                            all_titles[aid] = title
                        sources.setdefault(aid, f"https://huggingface.co/{slug}")
                        via_map.setdefault(aid, "hf_models")

                # Emit only NET-NEW (not already in baseline)
                new_emitted = 0
                for aid in sorted(all_ids):
                    if aid in baseline_ids:
                        continue
                    rec = {
                        "arxiv_id": aid,
                        "title": all_titles.get(aid),
                        "org_label": org.label,
                        "source_url": sources.get(aid, org.primary_url),
                        "via": via_map.get(aid, "unknown"),
                    }
                    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    baseline_ids.add(aid)
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
                logger.info(
                    "[{}] DONE primary={} hf={} unique={} NEW={}",
                    org.label, stat["primary_count"], stat["hf_count"],
                    stat["total_unique"], stat["new_emitted"],
                )

                out_fh.write(json.dumps({
                    "_completed": True,
                    "org_label": org.label,
                    "primary_count": stat["primary_count"],
                    "hf_count": stat["hf_count"],
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
    print(f"{'org':<26s} {'primary':>9s} {'hf':>6s} {'unique':>8s} {'new':>6s}  errors")
    per_org_stats.sort(key=lambda s: -s["new_emitted"])
    for s in per_org_stats:
        errs = "; ".join(s["errors"])[:120]
        print(f"{s['org']:<26s} {s['primary_count']:9d} {s['hf_count']:6d} "
              f"{s['total_unique']:8d} {s['new_emitted']:6d}  {errs}")

    total_new = sum(s["new_emitted"] for s in per_org_stats)
    zero_orgs = [s["org"] for s in per_org_stats if s["new_emitted"] == 0]
    print(f"\nTOTAL NET-NEW arxiv_ids: {total_new}")
    print(f"Orgs with 0 new: {len(zero_orgs)} → {', '.join(zero_orgs) if zero_orgs else '—'}")
    print(f"output: {args.output}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path,
                    default=Path("local_data/playwright_extra_companies.jsonl"))
    ap.add_argument("--main-output", type=Path,
                    default=Path("local_data/playwright_company_pubs.jsonl"))
    ap.add_argument("--crack-output", type=Path,
                    default=Path("local_data/playwright_crack_failed.jsonl"))
    ap.add_argument("--only",
                    type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    default=None)
    ap.add_argument("--smoke-test",
                    type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    default=None,
                    help="Comma-separated list of org labels to run only (smoke test).")
    ap.add_argument("--force", action="store_true",
                    help="Re-run orgs already marked _completed in --output.")
    ap.add_argument("--sleep-between-orgs", type=float, default=1.5)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
