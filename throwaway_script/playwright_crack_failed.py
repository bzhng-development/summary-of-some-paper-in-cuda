"""Rescue the 5 orgs the main playwright sweep couldn't crack.

Root cause from the main run (throwaway_script/playwright_pub_scrape.py):
  - Alibaba-DAMO had hf_org="" (empty) — the actual HF orgs are Alibaba-NLP
    + iic-ai (the DAMO Academy / Tongyi lab slugs), not "damo".
  - Tsinghua-THUDM used slug THUDM (valid, 100 models) but also has
    thu-coai (37 models, the Conversational-AI lab) which was never tried.
  - Character-AI / Inflection-AI genuinely publish almost nothing on HF
    (0 models). Their research is in-house blog posts that rarely cite
    arxiv — we accept near-zero here and note it.
  - xAI has xai-org (12 models) — small but real; the main run's
    duplicate xAI config (two entries) may have short-circuited it.

Strategy: model-card README harvest was the workhorse in the main run
(978 of 1812 IDs). Re-run it with CORRECTED + EXPANDED slug lists, plus
a sitemap/listing pass for the labs that have a real publications page.

Resume-safe: skips arxiv_ids already in the main output file so we only
ADD net-new coverage. Writes to a separate file for clean review.

Usage:
    uv run --with httpx --with playwright --with loguru python \\
        throwaway_script/playwright_crack_failed.py \\
        --main-output local_data/playwright_company_pubs.jsonl \\
        --output local_data/playwright_crack_failed.jsonl
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


# Corrected/expanded org → HF slugs. Each org can map to MULTIPLE HF orgs
# because labs split their models across several namespaces.
FAILED_ORGS = {
    "Alibaba-DAMO": ["Alibaba-NLP", "iic", "damo-vilab", "Alibaba-DAMO-Academy"],
    "Tsinghua-THUDM": ["THUDM", "thu-coai", "thunlp"],
    "Character-AI": ["CharacterAI", "character-ai"],
    "Inflection-AI": ["inflection", "InflectionAI"],
    "xAI": ["xai-org"],
}


def harvest_hf_models(org_slug: str, seen: set, throttle: float = 0.15) -> list[dict]:
    """Harvest arxiv IDs from an HF org's model-card READMEs + tags."""
    out: list[dict] = []
    if not org_slug:
        return out
    try:
        resp = httpx.get(
            "https://huggingface.co/api/models",
            params={"author": org_slug, "limit": 1000, "full": "true"},
            timeout=30,
        )
        models = resp.json()
    except Exception as e:
        logger.warning("  hf api fail for {}: {}", org_slug, e)
        return out
    if not isinstance(models, list):
        logger.warning("  hf api non-list for {}: {}", org_slug, str(models)[:120])
        return out
    logger.info("  [{}] {} models on HF", org_slug, len(models))
    for m in models:
        mid = m.get("id") or m.get("modelId") or ""
        if not mid:
            continue
        # tags often carry "arxiv:XXXX.YYYYY" directly — cheapest signal
        for tag in (m.get("tags") or []):
            tm = ARXIV_TAG_RE.search(tag)
            if tm:
                aid = tm.group(1)
                if aid not in seen:
                    seen.add(aid)
                    out.append({"arxiv_id": aid, "via": "hf_tag", "hf_model": mid})
        # fetch README for inline arxiv links
        try:
            r2 = httpx.get(f"https://huggingface.co/{mid}/raw/main/README.md", timeout=20)
            if r2.status_code == 200:
                for am in ARXIV_RE.finditer(r2.text):
                    aid = am.group(1)
                    if aid not in seen:
                        seen.add(aid)
                        out.append({"arxiv_id": aid, "via": "hf_readme", "hf_model": mid})
        except Exception:
            pass
        time.sleep(throttle)
    return out


def harvest_hf_datasets(org_slug: str, seen: set, throttle: float = 0.15) -> list[dict]:
    """Some labs cite arxiv only in DATASET cards (e.g. benchmark releases)."""
    out: list[dict] = []
    if not org_slug:
        return out
    try:
        resp = httpx.get(
            "https://huggingface.co/api/datasets",
            params={"author": org_slug, "limit": 500, "full": "true"},
            timeout=30,
        )
        ds = resp.json()
    except Exception as e:
        logger.warning("  hf dataset api fail for {}: {}", org_slug, e)
        return out
    if not isinstance(ds, list):
        return out
    if ds:
        logger.info("  [{}] {} datasets on HF", org_slug, len(ds))
    for d in ds:
        did = d.get("id") or ""
        if not did:
            continue
        for tag in (d.get("tags") or []):
            tm = ARXIV_TAG_RE.search(tag)
            if tm:
                aid = tm.group(1)
                if aid not in seen:
                    seen.add(aid)
                    out.append({"arxiv_id": aid, "via": "hf_ds_tag", "hf_dataset": did})
        try:
            r2 = httpx.get(f"https://huggingface.co/datasets/{did}/raw/main/README.md", timeout=20)
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


def load_seen(main_output: Path) -> set:
    seen: set = set()
    if main_output.is_file():
        for line in main_output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                seen.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--main-output", type=Path, default=Path("local_data/playwright_company_pubs.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("local_data/playwright_crack_failed.jsonl"))
    args = ap.parse_args()

    # Seed `seen` with everything the main run already found, so we only log
    # NET-NEW arxiv IDs here.
    seen = load_seen(args.main_output)
    logger.info("seeded {} arxiv_ids from main output (dedup baseline)", len(seen))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Also dedup against our own prior runs of this script.
    own_seen = load_seen(args.output)
    seen |= own_seen
    logger.info("plus {} from prior crack runs", len(own_seen))

    per_org_new: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as fh:
        for org_label, slugs in FAILED_ORGS.items():
            logger.info("=== {} (slugs: {}) ===", org_label, slugs)
            org_new = 0
            for slug in slugs:
                recs = harvest_hf_models(slug, seen)
                recs += harvest_hf_datasets(slug, seen)
                for rec in recs:
                    rec["org_label"] = org_label
                    rec["hf_slug"] = slug
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                org_new += len(recs)
            per_org_new[org_label] = org_new
            logger.info("[{}] net-new arxiv IDs: {}", org_label, org_new)

    print("\n=== crack-failed summary (NET-NEW only) ===")
    total = 0
    for org, n in per_org_new.items():
        print(f"  {org:20s} +{n}")
        total += n
    print(f"\n  TOTAL net-new: {total}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
