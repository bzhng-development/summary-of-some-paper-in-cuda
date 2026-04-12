#!/usr/bin/env python3
"""
High-throughput paper tagger. Self-contained — no local imports.

Reads a JSONL of papers (arxiv_id, title, abstract), tags each with a category
using a local LLM, writes results to an output JSONL.

Designed to run on a remote machine against a local/tunneled LLM server.
Handles thousands of concurrent requests via AsyncClientPool.

Usage:
    python tag_papers.py -i papers_to_tag.jsonl -o tagged_papers.jsonl
    python tag_papers.py -i papers_to_tag.jsonl -o tagged_papers.jsonl --concurrency 2048
    python tag_papers.py -i papers_to_tag.jsonl -o tagged_papers.jsonl --base-url http://localhost:30000/v1

"""
from __future__ import annotations

import argparse
import asyncio
import json
import resource
from pathlib import Path

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


# =============================================================================
# Categories
# =============================================================================

CATEGORIES = [
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
    "vision",
]

CATEGORY_DESCRIPTIONS = {
    "agents": "Agent frameworks, tool use, code agents, agentic systems, autonomous agents",
    "alignment": "RLHF, preference optimization, value alignment, instruction tuning for alignment",
    "architecture": "Novel model architectures, attention mechanisms, SSMs, transformers, RNNs, fundamental model design",
    "code": "Code generation models, code LLMs, programming assistants, code benchmarks",
    "context-optimization": "Long context, KV cache compression, context window extension, sparse attention for context",
    "data": "Datasets, data curation, synthetic data generation, data filtering, data quality",
    "diffusion": "Diffusion models, image/video generation, text-to-image, text-to-video",
    "distributed-training": "Parallelism strategies (data/tensor/pipeline), distributed systems for training, ZeRO, FSDP",
    "evaluation": "Benchmarks, evaluation methods, model analysis, leaderboards, probing",
    "inference-optimization": "Speculative decoding, test-time compute, KV cache optimization, inference speed, batching",
    "llm-systems": "End-to-end LLM systems, frameworks, deployment platforms, model releases/tech reports",
    "low-precision": "Quantization, FP8/FP4/INT4 training or inference, mixed precision, low-bit methods",
    "moe": "Mixture of experts architectures, expert routing, sparse expert models",
    "multimodal": "Vision-language models, audio-language, multimodal understanding, VLMs",
    "pretraining": "Pretraining methods, pretraining data strategies, foundation model training, tokenization",
    "prompting": "Prompt engineering, in-context learning, few-shot prompting techniques",
    "reasoning": "Chain-of-thought, reasoning models (o1/R1-style), mathematical reasoning, logical reasoning",
    "retrieval": "RAG, retrieval-augmented generation, dense retrieval, embedding models, search",
    "rl-training": "Reinforcement learning for LLMs, GRPO, PPO for LLMs, reward modeling, RL scaling",
    "safety": "Red-teaming, toxicity, robustness, adversarial attacks, guardrails, content filtering",
    "scaling-laws": "Scaling laws, compute-optimal training, chinchilla, neural scaling predictions",
    "serving": "Model serving infrastructure, request scheduling, GPU cluster management, vLLM/SGLang internals",
    "training-methods": "Fine-tuning, PEFT/LoRA, distillation, curriculum learning, optimization algorithms",
    "vision": "Pure vision models (ViT, CNN, detection, segmentation), image classification, not VLMs",
}


# =============================================================================
# Schema
# =============================================================================


class TagOutput(BaseModel):
    category: str = Field(description="Best matching category")
    confidence: float = Field(description="Confidence 0-1")
    reason: str = Field(description="1-sentence justification")


SYSTEM_PROMPT = f"""\
You are a paper categorizer. Given a paper's title and abstract, assign it to
exactly ONE category from the list below. Pick the MOST SPECIFIC category that fits.

Categories:
{chr(10).join(f'- {cat}: {CATEGORY_DESCRIPTIONS[cat]}' for cat in CATEGORIES)}

Rules:
- Pick the single best category. If a paper spans multiple, pick the primary contribution.
- MoE papers go to "moe" not "architecture" (unless the paper is about a non-MoE architecture that happens to mention MoE).
- Agent/tool-use papers go to "agents" not "llm-systems".
- Code LLM papers go to "code" not "pretraining".
- RL for LLMs (GRPO, PPO, reward modeling) goes to "rl-training". General RL theory goes elsewhere.
- Distributed training (parallelism, ZeRO, FSDP) goes to "distributed-training" not "training-methods".
- Quantization/low-bit goes to "low-precision" not "inference-optimization".
- Reasoning models (o1/R1-style, CoT) go to "reasoning" not "evaluation".
- Vision-language models go to "multimodal". Pure vision (no language) goes to "vision".
- Scaling laws papers go to "scaling-laws" not "pretraining".
- Model tech reports (Qwen, Llama, Gemma, etc.) go to "llm-systems" unless they primarily contribute a specific technique.

Respond with valid JSON matching the schema."""


# =============================================================================
# Client pool (same pattern as hf_daily_papers.py)
# =============================================================================


def _raise_fd_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
    if soft >= target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError) as exc:
        logger.warning("Could not raise fd limit: {}", exc)
    else:
        logger.info("Raised fd limit: {} -> {}", soft, target)


def make_client(
    base_url: str,
    concurrency: int = 4096,
    timeout: float = 300.0,
) -> AsyncOpenAI:
    _raise_fd_limit()
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",
        timeout=timeout,
        max_retries=0,
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=min(concurrency, 100),
            ),
            timeout=timeout,
        ),
    )
    logger.info("Client: max_connections={}, base_url={}", concurrency, base_url)
    return client


# =============================================================================
# Tagging
# =============================================================================


def _coerce_category(raw_category: str) -> str:
    """Normalize an LLM-predicted category against the allow-list."""
    cat = raw_category.strip().lower()
    if cat in CATEGORIES:
        return cat
    for c in CATEGORIES:
        if c in cat:
            return c
    return "uncategorized"


async def tag_one(
    client: AsyncOpenAI,
    paper: dict,
    model: str,
) -> dict | None:
    title = paper["title"]
    abstract = paper.get("abstract", "")
    user_msg = f"Title: {title}\n\nAbstract:\n{abstract[:2000]}"

    for attempt in range(5):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tag_output",
                        "schema": TagOutput.model_json_schema(),
                    },
                },
            )
            raw = resp.choices[0].message.content or ""
            if not raw.strip():
                raise ValueError("Empty response from LLM")
            result = TagOutput.model_validate_json(raw)
            return {
                "arxiv_id": paper["arxiv_id"],
                "title": title,
                "category": _coerce_category(result.category),
                "confidence": result.confidence,
                "reason": result.reason,
            }
        except Exception as exc:
            if attempt == 4:
                logger.error("FAILED {}: {}", paper["arxiv_id"], exc)
                return None
            await asyncio.sleep(2**attempt)
    return None


async def async_main():
    parser = argparse.ArgumentParser(description="High-throughput paper tagger")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSONL")
    parser.add_argument("--base-url", default="http://localhost:30000/v1")
    parser.add_argument("--model", default="default", help="Model name (use 'default' for auto-detect)")
    parser.add_argument("--concurrency", type=int, default=4096)
    args = parser.parse_args()

    # Load input
    papers: list[dict] = []
    for line in args.input.read_text().splitlines():
        line = line.strip()
        if line:
            papers.append(json.loads(line))
    logger.info("Loaded {} papers from {}", len(papers), args.input)

    # Skip already tagged
    already_done: set[str] = set()
    try:
        existing_lines = args.output.read_text().splitlines()
    except FileNotFoundError:
        existing_lines = []
    for line in existing_lines:
        line = line.strip()
        if not line:
            continue
        try:
            already_done.add(json.loads(line)["arxiv_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    if already_done:
        papers = [p for p in papers if p["arxiv_id"] not in already_done]
        logger.info(
            "Skipped {} already tagged, {} remaining",
            len(already_done),
            len(papers),
        )

    if not papers:
        logger.info("Nothing to tag!")
        return

    client = make_client(base_url=args.base_url, concurrency=args.concurrency)

    # Auto-detect model
    model = args.model
    if model == "default":
        try:
            models = await client.models.list()
            model = models.data[0].id
            logger.info("Auto-detected model: {}", model)
        except Exception as exc:
            logger.error("Could not auto-detect model: {}", exc)
            logger.error("Pass --model explicitly, e.g. --model Qwen/Qwen3.5-2B")
            return

    sem = asyncio.Semaphore(args.concurrency)
    completed = 0
    save_lock = asyncio.Lock()

    try:
        with args.output.open("a") as out_f:

            async def _tag_one(paper: dict) -> None:
                nonlocal completed
                async with sem:
                    result = await tag_one(client, paper, model)
                    if result is None:
                        return
                    async with save_lock:
                        out_f.write(json.dumps(result) + "\n")
                        completed += 1
                        if completed % 50 == 0:
                            out_f.flush()
                            logger.info(
                                "Progress: {}/{} tagged", completed, len(papers)
                            )

            tasks = [asyncio.create_task(_tag_one(p)) for p in papers]
            await asyncio.gather(*tasks)
            out_f.flush()
    finally:
        await client.close()

    logger.success(
        "Done: tagged {}/{} papers -> {}", completed, len(papers), args.output
    )


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
