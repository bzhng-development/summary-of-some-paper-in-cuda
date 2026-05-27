"""Re-tag every interested=1 paper with MULTI-tag (V4-Pro structured output).

Why a new script instead of editing daily_papers/tag_papers.py:
  - tag_papers.py emits a single category per paper (CategoryOutput schema).
  - We want MULTI-tag (a paper can be e.g. low-precision AND inference-optimization
    AND kernel-design simultaneously). The user's "Low Precision" view had only
    9 papers because every paper got exactly one bucket — recent FP4 papers
    ended up in inference-optimization, missing from the low-precision view.
  - Schema change: nextjs-ui_paper.tag_categories_v2 text[] (NEW).
  - The single-best primary stays in tag_category_v2 / category for backward compat.

Flow:
  1. Pull every interested=1 paper from Neon (title + abstract).
  2. Skip rows missing abstract — multi-tag requires the abstract.
  3. Concurrent V4-Pro calls (configurable) with a small JSON schema
     {primary: str, categories: list[str], confidence: float, reason: str}.
  4. Validate categories ⊂ known set; trim invalid.
  5. UPDATE Neon: tag_categories_v2 = [...] and tag_category_v2 = primary
     (only if the user hasn't manually overridden category — we never touch the
     `category` column).

Usage:
    DATABASE_URL=... uv run python throwaway_script/multi_tag_via_vllm.py \\
        --base-url http://localhost:8000/v1 \\
        --model deepseek-ai/DeepSeek-V4-Pro \\
        --concurrency 16 \\
        --out local_data/multi_tag_results.jsonl \\
        [--limit N] [--retag-all] [--remote]

By default skips arxiv_ids that already have tag_categories_v2 populated;
--retag-all overrides and re-tags everyone.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from openai import AsyncOpenAI
from psycopg.rows import dict_row
from pydantic import BaseModel, Field, field_validator

from neon_db import NeonDB, TABLE


CATEGORIES: list[str] = [
    "agents",
    "alignment",
    "architecture",
    "code",
    "context-optimization",
    "data",
    "diffusion",
    "distributed-training",
    "evaluation",
    "inference-optimization",
    "llm-systems",
    "low-precision",
    "moe",
    "multimodal",
    "pretraining",
    "prompting",
    "reasoning",
    "retrieval",
    "rl-training",
    "safety",
    "scaling-laws",
    "serving",
    "training-methods",
    "uncategorized",
    "vision",
]
CATEGORY_SET: frozenset[str] = frozenset(CATEGORIES)


class MultiTagOutput(BaseModel):
    primary: str = Field(description="The single best category fit.")
    categories: list[str] = Field(
        description=(
            "All categories the paper genuinely belongs to. 1-4 entries. The primary MUST be "
            "first in this list. Don't include more than 4 — pick the most relevant only."
        ),
        min_length=1,
        max_length=4,
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=400)

    @field_validator("categories")
    @classmethod
    def normalize_cats(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for c in v:
            s = (c or "").strip().lower()
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out


SYSTEM_PROMPT = (
    "You assign a research paper to one or more categories. The available categories are:\n"
    + ", ".join(CATEGORIES)
    + "\n\n"
    "Rules:\n"
    "- Pick 1-4 categories that the paper genuinely belongs to. Most papers belong to 1-2; "
    "papers that introduce a primitive used across multiple domains may need 3-4.\n"
    "- The first entry MUST be the single best fit.\n"
    "- Examples:\n"
    "  * An FP4 quantization paper → primary=low-precision, also inference-optimization\n"
    "  * A GRPO training-recipe paper → primary=rl-training, also training-methods\n"
    "  * A speculative-decoding kernel → primary=inference-optimization, also serving\n"
    "  * A vision-language tokenizer → primary=multimodal, also vision\n"
    "- Only return categories from the provided list; do NOT invent new ones.\n"
    "- If unsure, fall back to 'uncategorized' (it's a real category, single-element list).\n"
)


async def tag_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    arxiv_id: str,
    title: str,
    abstract: str,
) -> tuple[str, MultiTagOutput | None, str | None]:
    user = f"Title: {title}\n\nAbstract: {abstract}\n\nReturn the JSON object."
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "multi_tag", "schema": MultiTagOutput.model_json_schema()},
    }
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format=response_format,
                temperature=0.3,
                top_p=1.0,
                extra_body={"chat_template_kwargs": {"thinking": False}},
            )
        except Exception as e:
            return arxiv_id, None, f"{type(e).__name__}: {e}"
    raw = resp.choices[0].message.content or ""
    try:
        parsed = MultiTagOutput.model_validate_json(raw)
    except Exception as e:
        return arxiv_id, None, f"parse: {e}; raw[:200]={raw[:200]!r}"
    return arxiv_id, parsed, None


def filter_valid_categories(cats: list[str]) -> list[str]:
    """Drop any category that's not in the canonical CATEGORY_SET."""
    return [c for c in cats if c in CATEGORY_SET]


async def go(args: argparse.Namespace) -> int:
    db = NeonDB()
    with db.get_conn() as c, c.cursor(row_factory=dict_row) as cur:
        cur.execute(f"""
            SELECT id, title, abstract, tag_categories_v2
            FROM {TABLE}
            WHERE interested = 1
            ORDER BY id
        """)
        rows = cur.fetchall()
    logger.info("interested=1 total: {}", len(rows))

    valid = [r for r in rows if (r.get("abstract") or "").strip() and (r.get("title") or "").strip()]
    skipped_no_abs = len(rows) - len(valid)
    logger.info("with title + abstract: {} (skipped {} missing one)", len(valid), skipped_no_abs)

    if not args.retag_all:
        valid = [r for r in valid if not r.get("tag_categories_v2")]
        logger.info("after dropping already-multi-tagged: {}", len(valid))

    if args.limit:
        valid = valid[: args.limit]
        logger.info("truncated to {} (--limit)", len(valid))

    if not valid:
        logger.info("nothing to do")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.out.is_file():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["arxiv_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        logger.info("resume: {} already in output file", len(done))

    todo = [r for r in valid if r["id"] not in done]
    logger.info("tagging {} papers via {} (concurrency={})", len(todo), args.model, args.concurrency)

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=None, max_retries=2)
    sem = asyncio.Semaphore(args.concurrency)

    results: dict[str, MultiTagOutput] = {}
    n_ok = n_fail = 0
    out_lock = asyncio.Lock()
    out_fh = args.out.open("a", encoding="utf-8")

    async def one(row: dict) -> None:
        nonlocal n_ok, n_fail
        aid, parsed, err = await tag_one(
            client, sem, args.model, row["id"], row["title"], row["abstract"]
        )
        if err is not None or parsed is None:
            n_fail += 1
            logger.warning("FAIL {}: {}", aid, err)
            return
        # Validate against canonical set
        valid_cats = filter_valid_categories(parsed.categories)
        if not valid_cats:
            valid_cats = ["uncategorized"]
        rec = {
            "arxiv_id": aid,
            "primary": parsed.primary if parsed.primary in CATEGORY_SET else valid_cats[0],
            "categories": valid_cats,
            "confidence": parsed.confidence,
            "reason": parsed.reason,
        }
        async with out_lock:
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            results[aid] = parsed
            n_ok += 1
            if n_ok % 25 == 0:
                logger.info("progress: ok={} fail={} todo={}", n_ok, n_fail, len(todo))

    try:
        await asyncio.gather(*[one(r) for r in todo])
    finally:
        out_fh.close()

    logger.info("done with tagging: ok={} fail={}", n_ok, n_fail)
    if n_ok == 0:
        return 0

    # Bulk-update Neon. Only writes the multi-tag column. tag_category_v2 +
    # category stay unless explicitly overridden by --update-primary.
    logger.info("writing to Neon...")
    n_written = 0
    db2 = NeonDB()
    with db2.batch() as b:
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fields: dict[str, Any] = {"tag_categories_v2": rec["categories"]}
            if args.update_primary:
                fields["tag_category_v2"] = rec["primary"]
            b.save_paper(rec["arxiv_id"], **fields)
            n_written += 1
    logger.info("upserted {} rows", n_written)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("local_data/multi_tag_results.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retag-all", action="store_true", help="Re-tag papers that already have tag_categories_v2 set.")
    ap.add_argument("--update-primary", action="store_true", help="Also overwrite tag_category_v2 with the primary.")
    args = ap.parse_args()

    asyncio.run(go(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
