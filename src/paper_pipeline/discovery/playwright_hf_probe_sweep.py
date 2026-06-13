"""HF probe sweep for orgs that yielded 0 in T32 + Tencent slugs.

Many orgs the main sweep couldn't crack via website may still have small HF
presences under non-obvious slugs. This pass exhaustively probes slug variants
for each org and harvests any HF org with >=1 model.

Slug strategy per org: try the lowercase company name, hyphenated, corporate,
and a few known variants.

Resume-safe: skips arxiv_ids already in the existing JSONL outputs.

Usage:
    uv run --with httpx --with loguru python \\
        src/paper_pipeline/discovery/playwright_hf_probe_sweep.py
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

ARXIV_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")
ARXIV_TAG_RE = re.compile(r"arxiv:(\d{4}\.\d{4,5})")


# Org → list of HF slug candidates to probe. Each will be tested via
# /api/models?author=<slug>&limit=1; if it returns >=1 model, we harvest.
PROBE_ORGS: dict[str, list[str]] = {
    # T32 orgs that returned 0 — try slug variants
    "Netflix": ["netflix", "Netflix", "netflix-research"],
    "Spotify": ["spotify", "Spotify", "spotify-research"],
    "Airbnb": ["airbnb", "Airbnb", "airbnb-engineering"],
    "Pinterest": ["pinterest", "Pinterest", "pinterest-engineering"],
    "StitchFix": ["stitchfix", "stitch-fix", "StitchFix"],
    "TikTok": ["tiktok", "TikTok", "TikTokResearch", "tiktok-research"],
    "Uber": ["uber", "Uber", "uber-research"],
    "Zillow": ["zillow", "Zillow", "zillow-research"],
    "eBay": ["ebay", "eBay", "ebayinc"],
    "X-Twitter": ["twitter", "Twitter", "x-twitter", "x-corp"],
    "Lyft": ["lyft", "Lyft", "lyft-research"],
    "Roblox": ["roblox", "Roblox", "RobloxResearch", "roblox-research"],
    "PayPal": ["paypal", "PayPal", "paypal-research"],
    "Zoom": ["zoom", "Zoom", "zoom-research"],
    "Yandex": ["yandex", "Yandex", "yandex-research"],
    "THUNLP": ["thunlp", "THUNLP", "tsinghua-nlp"],
    "UW-NLP": ["uw-nlp", "UW-NLP", "uwnlp", "washingtonNLP", "uwiml"],
    # Tencent broader — the T32 slug tencent-AILab 404'd; try the variants
    # known to work and the ones the user explicitly asked for
    "Tencent": [
        "tencent",
        "Tencent",
        "TencentAILab",
        "tencent-AI-Lab",
        "TencentARC",
        "tencent-arc",
        "TencentBAC",
        "wechat-ai",
        "WeChatAI",
    ],
    # Slack / GitHub / Stripe / Mozilla / Dropbox were partially hit; try
    # extra slug variants in case more exist
    "Slack": ["slack", "Slack", "slack-engineering"],
    "GitHub": ["github", "GitHub", "github-research"],
    "Stripe": ["stripe", "Stripe", "stripe-research"],
    "Dropbox": ["dropbox", "Dropbox", "dropbox-research"],
}


def load_seen(*paths: Path) -> set:
    seen: set = set()
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                aid = rec.get("arxiv_id")
                if aid:
                    seen.add(aid)
            except json.JSONDecodeError, KeyError:
                pass
    return seen


def probe_slug(client: httpx.Client, slug: str) -> int:
    """Return # models if slug is real, 0 otherwise."""
    try:
        r = client.get(
            "https://huggingface.co/api/models",
            params={"author": slug, "limit": 1},
            timeout=15,
        )
        if r.status_code != 200:
            return 0
        data = r.json()
        if not isinstance(data, list):
            return 0
        return len(data)
    except Exception:
        return 0


def harvest_slug(client: httpx.Client, slug: str, seen: set, throttle: float = 0.12) -> list[dict]:
    """Harvest arxiv IDs from an HF org's full model+dataset README + tags."""
    out: list[dict] = []
    # Models
    try:
        r = client.get(
            "https://huggingface.co/api/models",
            params={"author": slug, "limit": 1000, "full": "true"},
            timeout=30,
        )
        models = r.json() if r.status_code == 200 else []
    except Exception:
        models = []
    if not isinstance(models, list):
        models = []
    logger.info("    {} models on {}", len(models), slug)
    for m in models:
        mid = m.get("id") or m.get("modelId") or ""
        if not mid:
            continue
        # arxiv tag
        for tag in m.get("tags") or []:
            tm = ARXIV_TAG_RE.search(tag)
            if tm:
                aid = tm.group(1)
                if aid not in seen:
                    seen.add(aid)
                    out.append({"arxiv_id": aid, "via": "hf_tag", "hf_model": mid})
        # README
        try:
            r2 = client.get(f"https://huggingface.co/{mid}/raw/main/README.md", timeout=20)
            if r2.status_code == 200:
                for am in ARXIV_RE.finditer(r2.text):
                    aid = am.group(1)
                    if aid not in seen:
                        seen.add(aid)
                        out.append({"arxiv_id": aid, "via": "hf_readme", "hf_model": mid})
        except Exception:
            pass
        time.sleep(throttle)

    # Datasets — many AI labs cite arxiv only in dataset cards
    try:
        r = client.get(
            "https://huggingface.co/api/datasets",
            params={"author": slug, "limit": 500, "full": "true"},
            timeout=30,
        )
        ds = r.json() if r.status_code == 200 else []
    except Exception:
        ds = []
    if not isinstance(ds, list):
        ds = []
    if ds:
        logger.info("    {} datasets on {}", len(ds), slug)
    for d in ds:
        did = d.get("id") or ""
        if not did:
            continue
        for tag in d.get("tags") or []:
            tm = ARXIV_TAG_RE.search(tag)
            if tm:
                aid = tm.group(1)
                if aid not in seen:
                    seen.add(aid)
                    out.append({"arxiv_id": aid, "via": "hf_ds_tag", "hf_dataset": did})
        try:
            r2 = client.get(f"https://huggingface.co/datasets/{did}/raw/main/README.md", timeout=20)
            if r2.status_code == 200:
                for am in ARXIV_RE.finditer(r2.text):
                    aid = am.group(1)
                    if aid not in seen:
                        seen.add(aid)
                        out.append({"arxiv_id": aid, "via": "hf_ds_readme", "hf_dataset": did})
        except Exception:
            pass
        time.sleep(throttle)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/playwright_hf_probe_sweep.jsonl"))
    args = ap.parse_args()

    seen = load_seen(
        Path("local_data/playwright_company_pubs.jsonl"),
        Path("local_data/playwright_crack_failed.jsonl"),
        Path("local_data/playwright_extra_companies.jsonl"),
        args.output,
    )
    logger.info("seeded {} arxiv_ids (dedup baseline)", len(seen))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(headers={"User-Agent": "playwright-hf-probe/1.0"})
    per_org_new: dict[str, int] = {}
    found_slugs: dict[str, list[str]] = {}

    with args.output.open("a", encoding="utf-8") as fh:
        for org, slugs in PROBE_ORGS.items():
            logger.info("=== {} (slug candidates: {}) ===", org, slugs)
            valid_slugs = []
            for slug in slugs:
                n = probe_slug(client, slug)
                if n > 0:
                    valid_slugs.append(slug)
                    logger.info("  [probe] {}: HIT ({}+ models)", slug, n)
                time.sleep(0.2)
            if not valid_slugs:
                logger.info("  [{}] no HF slug found", org)
                per_org_new[org] = 0
                continue
            found_slugs[org] = valid_slugs
            total_new = 0
            for slug in valid_slugs:
                recs = harvest_slug(client, slug, seen)
                for rec in recs:
                    rec["org_label"] = org
                    rec["hf_slug"] = slug
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                total_new += len(recs)
            per_org_new[org] = total_new
            logger.info("[{}] net-new arxiv IDs: {}", org, total_new)

    print("\n=== HF probe sweep summary ===")
    total = 0
    for org, n in sorted(per_org_new.items(), key=lambda x: -x[1]):
        slugs = found_slugs.get(org, [])
        slug_str = f" via {slugs}" if slugs else " (no HF)"
        print(f"  {org:18s} +{n:4d}{slug_str}")
        total += n
    print(f"\n  TOTAL net-new: {total}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
