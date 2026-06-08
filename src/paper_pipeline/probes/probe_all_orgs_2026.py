"""Audit every tracked org for 2026 arxiv coverage gaps.

Same pattern that worked for LinkedIn (9 of 10 papers were missing in Neon):
  1. For each org, run arxiv search across its top keywords.
  2. Collect every arxiv_id starting with "26" (i.e. 2026 submissions).
  3. Apply org-specific first-person affiliation regex to the abstract.
  4. Cross-reference against Neon — keep only NEW ones (not already flagged).
  5. Output: local_data/probe_all_orgs_2026.jsonl

Designed to be re-runnable. Adds ~7-10 min runtime for ~40 orgs.

Usage:
    DATABASE_URL=... uv run --with playwright python \\
        src/paper_pipeline/probes/probe_all_orgs_2026.py
"""

import asyncio
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


from paper_pipeline.core.neon_db import NeonDB, TABLE
from playwright.async_api import async_playwright


@dataclass(frozen=True)
class OrgProbe:
    label: str
    queries: tuple[str, ...]
    # Regexes that signal first-person affiliation in the abstract.
    # Combined with OR — any one match → keep.
    patterns: tuple[re.Pattern, ...]

    @classmethod
    def make(cls, label: str, queries: list[str], patterns: list[str]) -> "OrgProbe":
        return cls(
            label=label,
            queries=tuple(queries),
            patterns=tuple(re.compile(p, re.IGNORECASE) for p in patterns),
        )


# Affiliation pattern templates — reusable across many orgs:
def name_patterns(name: str, aliases: list[str] = None) -> list[str]:
    """Build first-person affiliation regexes for a single company name."""
    names = [re.escape(name)] + [re.escape(a) for a in (aliases or [])]
    name_re = "(?:" + "|".join(names) + ")"
    return [
        rf"\b(?:at|from|by)\s+{name_re}\b",
        rf"\b{name_re}('s)?\s+(?:team|researchers?|engineering|research|labs?|production|system|platform|model|API|deploy|members|users|recommendation|search|feed|services?|infrastructure)",
        rf"\bwe\s+(?:deploy|present|introduce|propose|describe|launch|build|train|develop|share|release).{{0,80}}\b{name_re}\b",
        rf"\bdeployed\s+(?:at|on|in)\s+{name_re}\b",
    ]


ORGS: tuple[OrgProbe, ...] = (
    # === Western LLM labs ===
    OrgProbe.make(
        "Meta / FAIR",
        ["Meta AI", "FAIR", "Facebook AI Research"],
        name_patterns("Meta AI", ["FAIR", "Facebook AI Research", "Meta Platforms"]),
    ),
    OrgProbe.make("DeepMind", ["DeepMind", "Google DeepMind"], name_patterns("DeepMind", ["Google DeepMind"])),
    OrgProbe.make("Anthropic", ["Anthropic"], name_patterns("Anthropic")),
    OrgProbe.make("OpenAI", ["OpenAI"], name_patterns("OpenAI")),
    OrgProbe.make(
        "Microsoft Research", ["Microsoft Research"], name_patterns("Microsoft Research", ["MSR", "Microsoft AI"])
    ),
    OrgProbe.make("Cohere", ["Cohere"], name_patterns("Cohere", ["Cohere For AI", "Cohere Labs"])),
    OrgProbe.make("Mistral", ["Mistral AI", "Mistral"], name_patterns("Mistral AI", ["Mistral"])),
    OrgProbe.make("NVIDIA", ["NVIDIA Research", "NVIDIA"], name_patterns("NVIDIA")),
    OrgProbe.make("Apple", ["Apple"], name_patterns("Apple", ["Apple ML", "Apple Intelligence"])),
    OrgProbe.make(
        "Salesforce",
        ["Salesforce AI Research", "Salesforce"],
        name_patterns("Salesforce", ["Salesforce AI Research", "Salesforce Research"]),
    ),
    OrgProbe.make("Together AI", ["Together AI"], name_patterns("Together AI", ["Together"])),
    OrgProbe.make("Snowflake", ["Snowflake AI"], name_patterns("Snowflake")),
    OrgProbe.make("AI21", ["AI21 Labs", "AI21"], name_patterns("AI21")),
    OrgProbe.make("IBM Research", ["IBM Research"], name_patterns("IBM Research", ["IBM AI"])),
    OrgProbe.make(
        "LinkedIn", ["LinkedIn Corporation", "LinkedIn AI", "LinkedIn"], name_patterns("LinkedIn", ["LinkedIn Corp"])
    ),
    # === Chinese labs ===
    OrgProbe.make(
        "Qwen / Alibaba",
        ["Qwen", "Alibaba", "Tongyi"],
        name_patterns("Qwen", ["Tongyi", "Alibaba Cloud", "Alibaba Group"]),
    ),
    OrgProbe.make("DeepSeek", ["DeepSeek"], name_patterns("DeepSeek")),
    OrgProbe.make(
        "ByteDance Seed", ["ByteDance", "ByteDance Seed"], name_patterns("ByteDance", ["ByteDance Seed", "Doubao"])
    ),
    OrgProbe.make(
        "Tencent (broader)",
        ["Tencent AI Lab", "Tencent"],
        name_patterns("Tencent", ["Tencent AI Lab", "WeChat AI", "Tencent ARC"]),
    ),
    OrgProbe.make("Hunyuan", ["Hunyuan"], name_patterns("Hunyuan", ["Tencent Hunyuan"])),
    OrgProbe.make("Moonshot / Kimi", ["Moonshot AI", "Kimi"], name_patterns("Moonshot", ["Kimi"])),
    OrgProbe.make("Zhipu / GLM", ["Zhipu AI", "Zhipu", "GLM"], name_patterns("Zhipu", ["Z.ai"])),
    OrgProbe.make("MiniMax", ["MiniMax"], name_patterns("MiniMax")),
    OrgProbe.make("01.AI / Yi", ["01.AI", "01.ai"], name_patterns("01.AI", ["Yi"])),
    OrgProbe.make("StepFun", ["StepFun", "Step-1"], name_patterns("StepFun", ["Step AI"])),
    OrgProbe.make("Baichuan", ["Baichuan"], name_patterns("Baichuan")),
    OrgProbe.make(
        "LongCat / Meituan", ["Meituan LongCat", "LongCat", "Meituan"], name_patterns("LongCat", ["Meituan"])
    ),
    OrgProbe.make("Alibaba DAMO", ["Alibaba DAMO", "DAMO Academy"], name_patterns("DAMO Academy", ["Alibaba DAMO"])),
    OrgProbe.make("Baidu Research", ["Baidu Research", "ERNIE"], name_patterns("Baidu", ["ERNIE"])),
    OrgProbe.make(
        "InternLM / Shanghai AI",
        ["InternLM", "Shanghai AI Laboratory"],
        name_patterns("InternLM", ["Shanghai AI Lab", "Shanghai AI Laboratory"]),
    ),
    OrgProbe.make("THUDM", ["THUDM"], name_patterns("THUDM", ["Tsinghua KEG"])),
    OrgProbe.make("THUNLP", ["THUNLP", "Tsinghua NLP"], name_patterns("THUNLP", ["Tsinghua NLP"])),
    OrgProbe.make("OpenBMB", ["OpenBMB"], name_patterns("OpenBMB")),
    # === Asian (non-Chinese) ===
    OrgProbe.make("KAIST AI", ["KAIST AI"], name_patterns("KAIST AI", ["KAIST"])),
    OrgProbe.make("LG / EXAONE", ["LG AI Research", "EXAONE"], name_patterns("LG AI Research", ["EXAONE"])),
    OrgProbe.make("Kakao Brain", ["Kakao Brain"], name_patterns("Kakao Brain", ["Kakao"])),
    OrgProbe.make("NAVER Clova", ["NAVER", "Clova"], name_patterns("NAVER", ["Clova"])),
    OrgProbe.make("Rakuten", ["Rakuten Institute", "Rakuten"], name_patterns("Rakuten")),
    OrgProbe.make("Yandex", ["Yandex Research"], name_patterns("Yandex")),
    # === Research nonprofits / consortiums ===
    OrgProbe.make(
        "AllenAI / Ai2", ["Allen Institute for AI", "AI2"], name_patterns("Allen Institute for AI", ["Ai2", "AllenAI"])
    ),
    OrgProbe.make("EleutherAI", ["EleutherAI"], name_patterns("EleutherAI")),
    OrgProbe.make("LAION", ["LAION"], name_patterns("LAION")),
    OrgProbe.make(
        "HuggingFace", ["HuggingFace research", "Hugging Face research"], name_patterns("Hugging Face", ["HuggingFace"])
    ),
    # === Big Tech ML (eng blogs heavy — low yield expected) ===
    OrgProbe.make("Amazon Science", ["Amazon Science"], name_patterns("Amazon", ["Amazon Science", "Amazon AWS"])),
    OrgProbe.make(
        "Google Research", ["Google Research"], name_patterns("Google Research", ["Google AI", "Google Brain"])
    ),
    OrgProbe.make("Snap Research", ["Snap Research", "Snap Inc"], name_patterns("Snap", ["Snap Research", "Snap Inc"])),
    OrgProbe.make("Pinterest", ["Pinterest"], name_patterns("Pinterest")),
    OrgProbe.make("Netflix", ["Netflix"], name_patterns("Netflix")),
    OrgProbe.make("Spotify", ["Spotify"], name_patterns("Spotify")),
    OrgProbe.make("TikTok", ["TikTok"], name_patterns("TikTok")),
    OrgProbe.make("Airbnb", ["Airbnb"], name_patterns("Airbnb")),
    OrgProbe.make("Uber", ["Uber AI"], name_patterns("Uber AI", ["Uber"])),
    OrgProbe.make("Lyft", ["Lyft"], name_patterns("Lyft")),
    OrgProbe.make("DoorDash", ["DoorDash"], name_patterns("DoorDash")),
    OrgProbe.make("Instacart", ["Instacart"], name_patterns("Instacart")),
    OrgProbe.make("Stripe", ["Stripe"], name_patterns("Stripe")),
    OrgProbe.make("PayPal", ["PayPal"], name_patterns("PayPal")),
    OrgProbe.make("eBay", ["eBay"], name_patterns("eBay")),
    OrgProbe.make("Zillow", ["Zillow"], name_patterns("Zillow")),
    OrgProbe.make("Yelp", ["Yelp"], name_patterns("Yelp")),
    OrgProbe.make("Roblox", ["Roblox"], name_patterns("Roblox")),
    OrgProbe.make("Zoom", ["Zoom"], name_patterns("Zoom")),
    OrgProbe.make("Stitch Fix", ["Stitch Fix"], name_patterns("Stitch Fix")),
    OrgProbe.make("Bloomberg", ["Bloomberg AI"], name_patterns("Bloomberg", ["Bloomberg AI"])),
    OrgProbe.make("Dropbox", ["Dropbox"], name_patterns("Dropbox")),
    OrgProbe.make("Mozilla", ["Mozilla"], name_patterns("Mozilla")),
    OrgProbe.make("Slack", ["Slack engineering"], name_patterns("Slack")),
    OrgProbe.make("GitHub", ["GitHub"], name_patterns("GitHub")),
    OrgProbe.make("X / Twitter", ["X Corp", "Twitter"], name_patterns("Twitter", ["X Corp"])),
)


async def fetch_search_page(page, keyword: str, page_idx: int, size: int = 200):
    url = (
        f"https://arxiv.org/search/?searchtype=all&query={keyword.replace(' ', '+')}"
        f"&start={page_idx * size}&size={size}"
    )
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_selector("li.arxiv-result", timeout=12_000)
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
        }).filter(p => p.arxiv_id && p.arxiv_id.startsWith('26'))""",
    )


def first_match(text: str, patterns) -> list[str]:
    hits = []
    for p in patterns:
        m = p.search(text)
        if m:
            hits.append(m.group(0))
    return hits


async def go():
    out_path = Path("local_data/probe_all_orgs_2026.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-fetch what's already in Neon for 26xx ids — so we only log NEW
    db = NeonDB()
    with db.get_conn() as c, c.cursor() as cur:
        cur.execute(f"SELECT id FROM {TABLE} WHERE id LIKE '26%'")
        in_neon_2026 = {row[0] for row in cur.fetchall()}
    print(f"Neon already has {len(in_neon_2026)} arxiv_ids in 2026")

    per_org_new: dict[str, list[dict]] = defaultdict(list)
    per_org_total: dict[str, int] = {}

    async with async_playwright() as pw:
        b = await pw.chromium.launch(headless=True)
        ctx = await b.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        page = await ctx.new_page()
        for org in ORGS:
            print(f"\n=== {org.label} ({len(org.queries)} queries) ===")
            seen_for_org: dict[str, dict] = {}
            for q in org.queries:
                for p_idx in range(3):
                    rows = await fetch_search_page(page, q, p_idx)
                    if not rows:
                        break
                    for r in rows:
                        aid = r["arxiv_id"]
                        if aid not in seen_for_org:
                            seen_for_org[aid] = r
            total = len(seen_for_org)
            per_org_total[org.label] = total

            kept_new = 0
            with out_path.open("a", encoding="utf-8") as fh:
                for aid, r in seen_for_org.items():
                    abstract = r.get("abstract") or ""
                    title = r.get("title") or ""
                    authors_text = r.get("authors_text") or ""
                    hits = first_match(abstract, org.patterns) or first_match(title, org.patterns)
                    if not hits and any(p.search(authors_text) for p in org.patterns):
                        hits = ["authors_text"]
                    if not hits:
                        continue
                    rec = {
                        "arxiv_id": aid,
                        "title": title,
                        "abstract": abstract[:500],
                        "org_label": org.label,
                        "affiliation_hits": hits,
                        "in_neon": aid in in_neon_2026,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    per_org_new[org.label].append(rec)
                    kept_new += 1
            print(
                f"  candidates 26xx={total}, verified-affil={kept_new}, NOT-in-Neon={sum(1 for r in per_org_new[org.label] if not r['in_neon'])}"
            )
        await b.close()

    print("\n=== SUMMARY (sorted by NEW missing from Neon) ===")
    print(f"{'org':30s} {'all-26':>7} {'in-Neon':>8} {'MISSING':>8}")
    print("-" * 60)
    grand_missing = 0
    for org_label, recs in sorted(per_org_new.items(), key=lambda x: -sum(1 for r in x[1] if not r["in_neon"])):
        in_n = sum(1 for r in recs if r["in_neon"])
        miss = sum(1 for r in recs if not r["in_neon"])
        grand_missing += miss
        print(f"{org_label:30s} {len(recs):>7} {in_n:>8} {miss:>8}")
    print(f"\n  GRAND TOTAL missing from Neon: {grand_missing}")
    print(f"  output: {out_path}")


if __name__ == "__main__":
    asyncio.run(go())
