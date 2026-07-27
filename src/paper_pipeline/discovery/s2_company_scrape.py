"""S2 company scrape — discover NEW arxiv papers per tracked org.

For each org we track:
  1. /paper/search/bulk?query=<org name>&year=<range>&fields=externalIds,authors.affiliations,title,year
  2. Paginate via continuation token.
  3. Verify each hit by checking authors[].affiliations against an org-specific
     regex (server-side text search alone is too loose — "Together" matches
     "bringing together").
  4. Extract arxiv_id from externalIds.ArXiv.
  5. Cross-ref Neon, write only the NET-NEW ones.

Designed to run multiple times: use ``--year-range`` for a historical sweep or
``--since``/``--through`` for an exact inclusive publication-date delta.

Rate-limit: API key, but S2 caps at 1 req/sec cumulative across all endpoints
— enforced by a global throttle.

Usage:
    DATABASE_URL=... S2_API_KEY=s2k-... uv run python \\
        src/paper_pipeline/discovery/s2_company_scrape.py --year-range 2025-2026
"""

import argparse
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import httpx
from loguru import logger

from paper_pipeline.core.date_window import DateWindow
from paper_pipeline.core.neon_db import TABLE, NeonBatch, NeonDB
from paper_pipeline.core.organization_scope import is_pure_academic_org

S2_BASE = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.environ.get("S2_API_KEY")
if not API_KEY:
    raise SystemExit("S2_API_KEY env var required")
HEADERS = {"x-api-key": API_KEY}

MIN_GAP_SEC = 1.3  # 1 rps stated, leave ~30% headroom — bursts trigger long 429 cooldowns
_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_GAP_SEC:
        time.sleep(MIN_GAP_SEC - elapsed)
    _last_request_at = time.monotonic()


# Each tuple: (org_label, S2 search query, regex pattern that any author's
# affiliation must match). The query is broad (anything fulltext); the regex
# is strict (filters server-side noise).
ORGS: list[tuple[str, str, re.Pattern]] = [
    # ---- Western LLM labs ----
    ("OpenAI", "OpenAI", re.compile(r"\bopenai\b", re.IGNORECASE)),
    ("Anthropic", "Anthropic", re.compile(r"\banthropic\b", re.IGNORECASE)),
    (
        "Meta / FAIR",
        "Meta AI OR FAIR",
        re.compile(r"\b(meta\s+ai|meta\s+platforms|fair\s+labs?|fair$|facebook\s+ai|meta\s+research)\b", re.IGNORECASE),
    ),
    ("DeepMind", "DeepMind", re.compile(r"\b(deepmind|google\s+deepmind)\b", re.IGNORECASE)),
    ("Microsoft Research", "Microsoft Research", re.compile(r"\b(microsoft\s+(research|ai)|msr\b)\b", re.IGNORECASE)),
    ("Google Research", "Google Research", re.compile(r"\bgoogle\s+(research|ai|brain)\b", re.IGNORECASE)),
    ("NVIDIA", "NVIDIA", re.compile(r"\bnvidia\b", re.IGNORECASE)),
    ("Apple", "Apple", re.compile(r"\bapple(\s+inc|\s+ai|\s+ml)?\b", re.IGNORECASE)),
    ("IBM Research", "IBM Research", re.compile(r"\bibm\s+(research|ai)\b", re.IGNORECASE)),
    ("Salesforce", "Salesforce", re.compile(r"\bsalesforce\b", re.IGNORECASE)),
    ("Mistral", "Mistral AI", re.compile(r"\bmistral\s*ai\b", re.IGNORECASE)),
    ("Cohere", "Cohere", re.compile(r"\bcohere\b", re.IGNORECASE)),
    ("AI21", "AI21 Labs", re.compile(r"\bai21\b", re.IGNORECASE)),
    ("Snowflake", "Snowflake AI", re.compile(r"\bsnowflake(\s+(ai|inc|research))?\b", re.IGNORECASE)),
    ("Together AI", "Together AI", re.compile(r"\btogether\s+ai\b", re.IGNORECASE)),
    ("Stability AI", "Stability AI", re.compile(r"\bstability\s+ai\b", re.IGNORECASE)),
    ("Adobe", "Adobe Research", re.compile(r"\badobe\b", re.IGNORECASE)),
    ("Amazon", "Amazon Science", re.compile(r"\bamazon\b", re.IGNORECASE)),
    ("Inflection", "Inflection AI", re.compile(r"\binflection\s+ai\b", re.IGNORECASE)),
    # ---- Chinese / Asian LLM labs ----
    (
        "Qwen / Alibaba",
        "Qwen OR Tongyi OR Alibaba",
        re.compile(r"\b(qwen|tongyi|alibaba\s+(damo|cloud|group|inc))\b", re.IGNORECASE),
    ),
    ("DeepSeek", "DeepSeek", re.compile(r"\bdeepseek\b", re.IGNORECASE)),
    ("ByteDance", "ByteDance OR Doubao", re.compile(r"\b(bytedance|seed\s+team|doubao)\b", re.IGNORECASE)),
    ("Tencent / Hunyuan", "Tencent OR Hunyuan", re.compile(r"\b(tencent|hunyuan|wechat\s+ai)\b", re.IGNORECASE)),
    ("Moonshot / Kimi", "Moonshot AI OR Kimi", re.compile(r"\b(moonshot|kimi)\b", re.IGNORECASE)),
    ("Zhipu / GLM", "Zhipu OR Z.ai OR ChatGLM", re.compile(r"\b(zhipu|z\.ai|chatglm)\b", re.IGNORECASE)),
    ("MiniMax", "MiniMax AI", re.compile(r"\bminimax\s+(ai|inc)\b", re.IGNORECASE)),
    ("01.AI", "01.AI OR Yi model", re.compile(r"\b(01[\.\-]?ai|01\.AI|lingyi)\b", re.IGNORECASE)),
    ("StepFun", "StepFun", re.compile(r"\bstepfun\b", re.IGNORECASE)),
    ("Baichuan", "Baichuan", re.compile(r"\bbaichuan(\s+inc)?\b", re.IGNORECASE)),
    ("LongCat / Meituan", "LongCat OR Meituan", re.compile(r"\b(longcat|meituan)\b", re.IGNORECASE)),
    ("Baidu Research", "Baidu Research", re.compile(r"\bbaidu\b", re.IGNORECASE)),
    ("Alibaba DAMO", "DAMO Academy", re.compile(r"\b(damo\s+academy|alibaba\s+damo)\b", re.IGNORECASE)),
    ("Huawei Noah's Ark", "Huawei Noah", re.compile(r"\bhuawei\s+(noah|technologies)\b", re.IGNORECASE)),
    (
        "InternLM / Shanghai AI Lab",
        "Shanghai AI Laboratory OR InternLM",
        re.compile(r"\b(shanghai\s+ai\s+lab(oratory)?|internlm)\b", re.IGNORECASE),
    ),
    ("THUDM", "THUDM", re.compile(r"\b(thudm|tsinghua\s+keg)\b", re.IGNORECASE)),
    ("THUNLP", "THUNLP", re.compile(r"\b(thunlp|tsinghua\s+(nlp|natural))\b", re.IGNORECASE)),
    ("OpenBMB", "OpenBMB", re.compile(r"\bopenbmb\b", re.IGNORECASE)),
    ("KAIST AI", "KAIST AI", re.compile(r"\bkaist(\s+(ai|kim\s+jaechul))?\b", re.IGNORECASE)),
    ("LG / EXAONE", "LG AI Research OR EXAONE", re.compile(r"\blg\s+ai\s+research\b|\bexaone\b", re.IGNORECASE)),
    ("Kakao Brain", "Kakao Brain", re.compile(r"\bkakao\s+brain\b", re.IGNORECASE)),
    ("NAVER", "NAVER Clova", re.compile(r"\bnaver(\s+(ai|clova|labs?))\b", re.IGNORECASE)),
    ("Rakuten RIT", "Rakuten Institute", re.compile(r"\brakuten(\s+institute\s+of\s+technology)?\b", re.IGNORECASE)),
    ("Sony", "Sony AI OR Sony Research", re.compile(r"\bsony\s+(ai|research|computer)\b", re.IGNORECASE)),
    # ---- Research nonprofits ----
    (
        "AllenAI / Ai2",
        "Allen Institute for AI OR Ai2",
        re.compile(r"\b(allen\s+institute\s+for\s+ai|ai2|allenai)\b", re.IGNORECASE),
    ),
    ("EleutherAI", "EleutherAI", re.compile(r"\beleutherai\b", re.IGNORECASE)),
    ("LAION", "LAION", re.compile(r"\blaion\b", re.IGNORECASE)),
    ("Hugging Face", "Hugging Face", re.compile(r"\bhugging\s*face\b", re.IGNORECASE)),
    # ---- Big Tech eng/research blogs ----
    ("LinkedIn", "LinkedIn", re.compile(r"\blinkedin(\s+(corp|corporation|ai|engineering|inc))?\b", re.IGNORECASE)),
    ("Snap", "Snap Research OR Snap Inc", re.compile(r"\bsnap(\s+(inc|research))?\b", re.IGNORECASE)),
    ("Pinterest", "Pinterest", re.compile(r"\bpinterest(\s+(inc|engineering))?\b", re.IGNORECASE)),
    ("Netflix", "Netflix", re.compile(r"\bnetflix(\s+(inc|research))?\b", re.IGNORECASE)),
    ("Spotify", "Spotify", re.compile(r"\bspotify(\s+(research|inc))?\b", re.IGNORECASE)),
    ("TikTok", "TikTok Research", re.compile(r"\btiktok(\s+(research|inc))?\b", re.IGNORECASE)),
    ("Airbnb", "Airbnb", re.compile(r"\bairbnb(\s+(inc|engineering))?\b", re.IGNORECASE)),
    (
        "Uber",
        "Uber AI OR Uber Research",
        re.compile(r"\buber\s+(ai|research|engineering|technologies)\b", re.IGNORECASE),
    ),
    ("Lyft", "Lyft", re.compile(r"\blyft(\s+(inc|engineering))?\b", re.IGNORECASE)),
    ("DoorDash", "DoorDash", re.compile(r"\bdoordash(\s+(inc|engineering))?\b", re.IGNORECASE)),
    ("Instacart", "Instacart", re.compile(r"\binstacart\b", re.IGNORECASE)),
    ("Zillow", "Zillow", re.compile(r"\bzillow\b", re.IGNORECASE)),
    ("eBay", "eBay Research", re.compile(r"\bebay\b", re.IGNORECASE)),
    ("Zoom", "Zoom Video Communications", re.compile(r"\bzoom\s+(video|communications)\b", re.IGNORECASE)),
    ("PayPal", "PayPal", re.compile(r"\bpaypal\b", re.IGNORECASE)),
    ("Stripe", "Stripe", re.compile(r"\bstripe(\s+(inc|engineering))?\b", re.IGNORECASE)),
    ("Stitch Fix", "Stitch Fix", re.compile(r"\bstitch\s+fix\b", re.IGNORECASE)),
    ("X / Twitter", "Twitter OR X Corp", re.compile(r"\b(twitter|x\s+corp(oration)?)\b", re.IGNORECASE)),
    ("Dropbox", "Dropbox", re.compile(r"\bdropbox\b", re.IGNORECASE)),
    ("Mozilla", "Mozilla Research", re.compile(r"\bmozilla(\s+(research|foundation))?\b", re.IGNORECASE)),
    ("Roblox", "Roblox", re.compile(r"\broblox\b", re.IGNORECASE)),
    ("Yandex", "Yandex Research", re.compile(r"\byandex\b", re.IGNORECASE)),
    ("Bloomberg AI", "Bloomberg AI", re.compile(r"\bbloomberg(\s+ai|\s+lp)\b", re.IGNORECASE)),
    ("GitHub", "GitHub Research", re.compile(r"\bgithub(\s+(inc|research))\b", re.IGNORECASE)),
    ("Slack", "Slack engineering", re.compile(r"\bslack(\s+(inc|technologies))\b", re.IGNORECASE)),
    # ---- Universities (sanity benchmarks — we expect lots) ----
    ("Stanford NLP", "Stanford NLP", re.compile(r"\bstanford(\s+(university|nlp|ai|sail))\b", re.IGNORECASE)),
    ("Berkeley BAIR", "Berkeley AI Research", re.compile(r"\b(uc\s+berkeley|bair|berkeley\s+ai)\b", re.IGNORECASE)),
    ("CMU LTI", "Carnegie Mellon LTI", re.compile(r"\b(carnegie\s+mellon|cmu)\b", re.IGNORECASE)),
    (
        "UW NLP",
        "University of Washington NLP",
        re.compile(r"\b(university\s+of\s+washington|uw\s+nlp)\b", re.IGNORECASE),
    ),
    (
        "MIT",
        "MIT CSAIL OR MIT EECS",
        re.compile(r"\b(massachusetts\s+institute\s+of\s+technology|mit\s+(csail|eecs))\b", re.IGNORECASE),
    ),
    ("Princeton", "Princeton University", re.compile(r"\bprinceton\s+university\b", re.IGNORECASE)),
    ("NYU", "New York University", re.compile(r"\b(new\s+york\s+university|nyu)\b", re.IGNORECASE)),
    ("Tsinghua University", "Tsinghua University", re.compile(r"\btsinghua\s+university\b", re.IGNORECASE)),
]


def _normalize_query(q: str) -> str:
    """Convert human-written 'A OR B' queries into S2 bulk boolean syntax.

    S2's bulk `query` treats the literal word 'OR' as a required token, so
    'Qwen OR Tongyi OR Alibaba' matches ~0 papers (verified). The real OR
    operator is '|'. This was the dominant cause of zero-row orgs.

    NOTE: we deliberately do NOT quote multi-word alternatives into phrases —
    an unquoted 'TikTok Research' (implicit AND) keeps 34 rows, but the phrase
    '"TikTok Research"' collapses to 4. The keep-filter regex is the precision
    layer; the query only needs to surface candidates. Common-word OR
    alternatives (e.g. 'Seed', 'GLM') are pruned from the ORG queries directly
    rather than here, since they flood the page cap with noise.
    """
    return " | ".join(p.strip() for p in q.split(" OR "))


def search_bulk(
    query: str,
    year: str,
    token: str | None,
    publication_window: DateWindow | None = None,
) -> dict | None:
    """One /paper/search/bulk call."""
    _throttle()
    params = {
        "query": _normalize_query(query),
        # S2's affiliation coverage is sparse (<5% of authors populated even
        # for well-known papers like GPT-5 System Card) and often shows historic
        # university not current company. We trust the text query match instead.
        "fields": "externalIds,title,year,authors,publicationDate,abstract",
    }
    if publication_window is None:
        params["year"] = year
    else:
        params["publicationDateOrYear"] = (
            f"{publication_window.since.isoformat()}:{publication_window.through.isoformat()}"
        )
    if token:
        params["token"] = token
    for attempt in range(5):
        try:
            r = httpx.get(
                f"{S2_BASE}/paper/search/bulk",
                params=params,
                headers=HEADERS,
                timeout=60,
            )
        except Exception as e:
            logger.warning(f"  request err: {e}")
            time.sleep(3)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning(f"  429, sleeping {wait}s")
            time.sleep(wait)
            _throttle()
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(5)
            continue
        logger.warning(f"  HTTP {r.status_code}: {r.text[:200]}")
        return None
    return None


def extract_arxiv_id(externalIds: dict | None) -> str | None:
    if not externalIds:
        return None
    arxiv = externalIds.get("ArXiv") or externalIds.get("ARXIV")
    if arxiv and re.match(r"^\d{4}\.\d{4,5}$", arxiv):
        return arxiv
    return None


def affil_matches(authors: list, pat: re.Pattern) -> list[str]:
    """Legacy — kept in case S2 ever populates affiliations. Currently always [].

    We instead use trust_text_match() below to filter S2 results by checking
    title/abstract for the org regex.
    """
    return [aff for a in authors or [] for aff in a.get("affiliations") or [] if pat.search(aff or "")]


def trust_text_match(title: str, abstract: str, authors: list, pat: re.Pattern) -> list[str]:
    """Verify org match via title/abstract/author-name text. S2 returned this
    paper because the org name appears SOMEWHERE — figure out where to flag
    confidence level. For distinctive names this is almost always genuine.
    """
    hits: list[str] = []
    for src_name, txt in [("title", title), ("abstract", abstract[:600] if abstract else "")]:
        m = pat.search(txt or "")
        if m:
            hits.append(f"{src_name}:{m.group(0)}")
    # Even if title/abstract don't mention, an author named after the org
    # (rare — happens for "DeepSeek-AI" listed as author) is a strong signal.
    hits.extend(f"author:{a.get('name')}" for a in authors or [] if pat.search(a.get("name") or ""))
    return hits


def scrape_org(
    label: str,
    query: str,
    pat: re.Pattern,
    year: str,
    year_min: int,
    year_max: int,
    in_neon: set[str],
    out_fh,
    max_pages: int = 10,
    neon_batch: NeonBatch | None = None,
    publication_window: DateWindow | None = None,
) -> tuple[int, int, int]:
    """Returns (kept, total_seen, saved_to_neon). Streams matching records to
    out_fh, and — when ``neon_batch`` is given — upserts each NET-NEW paper
    (not already in Neon) as a stub row so the scoring/summary pipeline picks
    it up. The upsert is partial, so existing rows are never clobbered."""
    token = None
    seen_arxiv: set[str] = set()
    kept = 0
    total = 0
    saved = 0
    page = 0
    while True:
        data = search_bulk(query, year, token, publication_window)
        page += 1
        if not data or "data" not in data:
            break
        items = data.get("data") or []
        total += len(items)
        for p in items:
            aid = extract_arxiv_id(p.get("externalIds"))
            if not aid or aid in seen_arxiv:
                continue
            # Filter by year prefix (arxiv id starts with YY of submission year)
            yy = aid[:2]
            try:
                yyi = int(yy) + 2000
            except ValueError:
                continue
            if not (year_min <= yyi <= year_max):
                continue
            if publication_window is not None and not publication_window.contains(p.get("publicationDate")):
                continue
            # Verify match via title/abstract text (S2 affiliations are too
            # sparse to be useful; the text query already filtered to papers
            # mentioning the org).
            hits = trust_text_match(
                p.get("title") or "",
                p.get("abstract") or "",
                p.get("authors") or [],
                pat,
            )
            if not hits:
                continue
            seen_arxiv.add(aid)
            rec = {
                "arxiv_id": aid,
                "title": p.get("title") or "",
                "year": p.get("year"),
                "org_label": label,
                "affiliation_hits": hits[:3],
                "publicationDate": p.get("publicationDate"),
                "in_neon": aid in in_neon,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1
            if neon_batch is not None and aid not in in_neon:
                neon_batch.save_paper(
                    aid,
                    title=p.get("title") or None,
                    url=f"https://arxiv.org/abs/{aid}",
                    organization=label,
                    published=p.get("publicationDate") or None,
                    score_source="s2_company_scrape",
                )
                saved += 1
        token = data.get("token")
        if not token or page >= max_pages:
            break
        out_fh.flush()
    return kept, total, saved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--year-range",
        type=str,
        default=None,
        help="S2 year filter (e.g. '2025-2026' or '2020-2024'; default: 2025-2026).",
    )
    ap.add_argument("--since", type=str, default=None, help="Inclusive publication date (YYYY-MM-DD).")
    ap.add_argument("--through", type=str, default=None, help="Inclusive publication date (YYYY-MM-DD).")
    ap.add_argument("--max-pages", type=int, default=10, help="Pages of 1000 per org per call.")
    ap.add_argument("--orgs", type=str, default=None, help="Comma-separated subset of org labels to scrape.")
    ap.add_argument(
        "--save-neon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Upsert each net-new paper into Neon as a stub (default on). Use --no-save-neon for a dry run.",
    )
    args = ap.parse_args()

    exact_window = args.since is not None or args.through is not None
    try:
        publication_window = DateWindow.from_inputs(
            years=args.year_range,
            since=args.since,
            through=args.through,
        )
    except ValueError as error:
        ap.error(str(error))
    year_min = publication_window.since.year
    year_max = publication_window.through.year
    year_range = publication_window.years
    output = (
        args.output
        or Path("local_data") / f"s2_company_{publication_window.label if exact_window else year_range}.jsonl"
    )

    db = NeonDB()
    with db.get_conn() as c, c.cursor() as cur:
        # Pull arxiv ids whose YY prefix is in scope
        yy_clauses = " OR ".join([f"id LIKE '{y - 2000:02d}%'" for y in range(year_min, year_max + 1)])
        cur.execute(f"SELECT id FROM {TABLE} WHERE ({yy_clauses}) AND id NOT LIKE 'ext%'")
        in_neon = {r[0] for r in cur.fetchall()}
    logger.info(
        "Neon already has {} arxiv ids in {}..{}",
        len(in_neon),
        publication_window.since,
        publication_window.through,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    out_fh = output.open("a", encoding="utf-8")
    logger.info("Neon auto-save: {}", "ON" if args.save_neon else "OFF (dry run)")

    org_filter = set(args.orgs.split(",")) if args.orgs else None
    active_orgs = [org for org in ORGS if not is_pure_academic_org(org[0])]
    summary: list[tuple[str, int, int]] = []
    grand_new = 0
    grand_saved = 0
    batch_cm = db.batch() if args.save_neon else nullcontext(None)
    try:
        with batch_cm as nb:
            for i, (label, query, pat) in enumerate(active_orgs):
                if org_filter and label not in org_filter:
                    continue
                logger.info(f"[{i + 1}/{len(active_orgs)}] {label} (query={query!r})")
                kept, total, saved = scrape_org(
                    label,
                    query,
                    pat,
                    year_range,
                    year_min,
                    year_max,
                    in_neon,
                    out_fh,
                    max_pages=args.max_pages,
                    neon_batch=nb,
                    publication_window=publication_window if exact_window else None,
                )
                summary.append((label, kept, total))
                grand_new += kept
                grand_saved += saved
                logger.info(f"  -> verified={kept} of {total} total seen, saved_to_neon={saved}")
    finally:
        out_fh.close()

    print("\n=== SUMMARY (sorted by # verified) ===")
    print(f"{'org':30s} {'verified':>8} {'total_seen':>11}")
    print("-" * 53)
    for label, kept, total in sorted(summary, key=lambda x: -x[1]):
        print(f"{label:30s} {kept:>8} {total:>11}")
    print(f"\n  GRAND verified IDs: {grand_new}")
    print(
        f"  saved to Neon (net-new stubs): {grand_saved if args.save_neon else 0}{'' if args.save_neon else ' (dry run)'}"
    )
    print(f"  output: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
