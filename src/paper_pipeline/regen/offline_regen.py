"""Regen all paper summaries with the locked prompts.

Three modes:
  --mode offline-tp     single-process LLM(), tensor parallel (default tp-size 8)
  --mode offline-dp     multi-process spawn, data parallel + expert parallel
                        (true DP=N+EP for MoE — recipe-recommended for B300)
  --mode online         hit a running vllm serve endpoint via AsyncOpenAI

Resume is unconditional: any existing per-section checkpoint is loaded and
that section's batch is skipped (or only the missing papers are run).

Per-section checkpoint files:
  {output}.s{n}.jsonl              (single-rank / merged)
  {output}.s{n}.rank{r}.jsonl      (per-rank, DP mode)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo root setup (needed before importing multi_prompt_pkg)
# ---------------------------------------------------------------------------


def setup_repo_root(repo_root: str) -> None:
    # The summarize prompts/config/schemas now live in the installable `paper_pipeline`
    # package (was `multi_prompt_pkg/`). If it's already importable (uv/editable install or
    # PYTHONPATH), we're done; otherwise add <repo_root>/src to sys.path as a fallback.
    import importlib.util
    import sys

    if importlib.util.find_spec("paper_pipeline") is not None:
        return
    src = Path(repo_root).resolve() / "src"
    if (src / "paper_pipeline").exists():
        sys.path.insert(0, str(src))
        return
    raise SystemExit(
        f"--repo-root {repo_root}: cannot import paper_pipeline (no installed pkg, no {src}/paper_pipeline)"
    )


# ---------------------------------------------------------------------------
# Prompt construction (same shape regardless of mode)
# ---------------------------------------------------------------------------


def build_context_block(prior_outputs: list[str]) -> str:
    if not prior_outputs:
        return ""
    return (
        "<prior_sections>\n"
        "The following sections have already been written for this paper. "
        "Do NOT repeat their content — reference it where needed and expand with new detail.\n\n"
        + "\n\n---\n\n".join(prior_outputs)
        + "\n</prior_sections>\n\n"
    )


def build_section_messages(
    paper_text: str,
    section_prompt: str,
    prior_outputs: list[str],
    system_preamble: str,
) -> list[dict[str, str]]:
    user = f"<paper>\n{paper_text}\n</paper>\n\n{build_context_block(prior_outputs)}{section_prompt}"
    return [
        {"role": "system", "content": system_preamble},
        {"role": "user", "content": user},
    ]


def strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


# ---------------------------------------------------------------------------
# Checkpoint I/O — supports per-rank files
# ---------------------------------------------------------------------------


def ckpt_path(out_path: Path, section: int, rank: int | None) -> Path:
    if rank is None:
        return out_path.with_suffix(f".s{section}.jsonl")
    return out_path.with_suffix(f".s{section}.rank{rank}.jsonl")


def pitch_ckpt_path(out_path: Path, rank: int | None) -> Path:
    if rank is None:
        return out_path.with_suffix(".pitch.jsonl")
    return out_path.with_suffix(f".pitch.rank{rank}.jsonl")


def cat_ckpt_path(out_path: Path, rank: int | None) -> Path:
    if rank is None:
        return out_path.with_suffix(".cat.jsonl")
    return out_path.with_suffix(f".cat.rank{rank}.jsonl")


def load_section_ckpts_into(papers: list[dict[str, Any]], out_path: Path, section: int) -> int:
    """Load any .s{n}.jsonl + .s{n}.rank*.jsonl checkpoints into paper["sections"][n]. Return # loaded."""
    by_id = {p["arxiv_id"]: p for p in papers}
    loaded = 0
    files = [out_path.with_suffix(f".s{section}.jsonl")]
    parent = out_path.parent
    stem = out_path.stem
    files.extend(parent.glob(f"{stem}.s{section}.rank*.jsonl"))
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = rec.get("arxiv_id")
                text = rec.get("text")
                if aid in by_id and isinstance(text, str) and text.strip():
                    by_id[aid]["sections"][section] = text
                    loaded += 1
    return loaded


def load_pitch_ckpts_into(papers: list[dict[str, Any]], out_path: Path) -> int:
    by_id = {p["arxiv_id"]: p for p in papers}
    loaded = 0
    parent = out_path.parent
    stem = out_path.stem
    files = [out_path.with_suffix(".pitch.jsonl"), *list(parent.glob(f"{stem}.pitch.rank*.jsonl"))]
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = rec.get("arxiv_id")
                if aid in by_id:
                    by_id[aid]["_title"] = rec.get("title") or ""
                    by_id[aid]["_pitch_text"] = rec.get("pitch") or ""
                    loaded += 1
    return loaded


def load_cat_ckpts_into(papers: list[dict[str, Any]], out_path: Path) -> int:
    by_id = {p["arxiv_id"]: p for p in papers}
    loaded = 0
    parent = out_path.parent
    stem = out_path.stem
    files = [out_path.with_suffix(".cat.jsonl"), *list(parent.glob(f"{stem}.cat.rank*.jsonl"))]
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = rec.get("arxiv_id")
                if aid in by_id:
                    by_id[aid]["_category"] = rec.get("category") or ""
                    loaded += 1
    return loaded


def write_section_ckpt(papers: list[dict[str, Any]], out_path: Path, section: int, rank: int | None) -> Path:
    path = ckpt_path(out_path, section, rank)
    with path.open("w", encoding="utf-8") as fh:
        for p in papers:
            txt = p["sections"].get(section, "")
            if not txt:
                continue
            fh.write(
                json.dumps({"arxiv_id": p["arxiv_id"], "section": section, "text": txt}, ensure_ascii=False) + "\n"
            )
    return path


def write_pitch_ckpt(papers: list[dict[str, Any]], out_path: Path, rank: int | None) -> Path:
    path = pitch_ckpt_path(out_path, rank)
    with path.open("w", encoding="utf-8") as fh:
        for p in papers:
            if not (p.get("_title") or p.get("_pitch_text")):
                continue
            fh.write(
                json.dumps(
                    {
                        "arxiv_id": p["arxiv_id"],
                        "title": p.get("_title", ""),
                        "pitch": p.get("_pitch_text", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def write_cat_ckpt(papers: list[dict[str, Any]], out_path: Path, rank: int | None) -> Path:
    path = cat_ckpt_path(out_path, rank)
    with path.open("w", encoding="utf-8") as fh:
        for p in papers:
            if not p.get("_category"):
                continue
            fh.write(
                json.dumps(
                    {
                        "arxiv_id": p["arxiv_id"],
                        "category": p["_category"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


# ---------------------------------------------------------------------------
# Paper loading
# ---------------------------------------------------------------------------


def load_papers(input_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("paper_text"):
                continue
            r["sections"] = {}
            papers.append(r)
    if limit:
        papers = papers[:limit]
    return papers


def slice_for_rank(papers: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    return [p for i, p in enumerate(papers) if i % world_size == rank]


# ---------------------------------------------------------------------------
# Final assembly + merge
# ---------------------------------------------------------------------------


def assemble_and_write(papers: list[dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as fh:
        for p in papers:
            full = "\n\n".join(p["sections"][n] for n in sorted(p["sections"]))
            rec = {
                "arxiv_id": p["arxiv_id"],
                "title": p.get("_title", "") or p.get("title", ""),
                "category": p.get("_category", "") or "uncategorized",
                "pitch": p.get("_pitch_text", ""),
                "summary": full,
                "url": p.get("url"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Offline single/multi-process worker shared core
# ---------------------------------------------------------------------------


def run_offline_worker(args, rank: int | None) -> None:
    """Run one offline worker. rank=None means single-process; rank=int means DP worker."""
    setup_repo_root(args.repo_root)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    from paper_pipeline.summarize.config import CATEGORIES, FALLBACK_CATEGORY
    from paper_pipeline.summarize.prompts import SECTION_SPECS, SYSTEM_PREAMBLE
    from paper_pipeline.summarize.schemas import CategoryOutput, PitchOutput

    out_path = Path(args.output)
    all_papers = load_papers(Path(args.input), limit=args.limit)
    papers = all_papers if rank is None else slice_for_rank(all_papers, rank, args.dp_size)
    print(f"[load rank={rank}] {len(papers)}/{len(all_papers)} papers")

    # ---- Resume scan ----
    for spec in SECTION_SPECS:
        n = spec.number
        loaded = load_section_ckpts_into(papers, out_path, n)
        if loaded:
            print(f"[resume rank={rank}] s{n}: loaded {loaded} cached outputs")
    if load_pitch_ckpts_into(papers, out_path):
        print(f"[resume rank={rank}] pitch: loaded cached outputs")
    if load_cat_ckpts_into(papers, out_path):
        print(f"[resume rank={rank}] cat: loaded cached outputs")

    needed_per_section = {n: [p for p in papers if not p["sections"].get(n)] for n in (s.number for s in SECTION_SPECS)}
    pitch_todo = [p for p in papers if not p.get("_pitch_text")]
    cat_todo = [p for p in papers if not p.get("_category")]

    nothing_to_do = (
        all(len(v) == 0 for v in needed_per_section.values()) and len(pitch_todo) == 0 and len(cat_todo) == 0
    )
    if nothing_to_do:
        print(f"[rank={rank}] all cached, nothing to do")
        return

    # ---- DP env setup (only matters when rank is set + dp_size > 1) ----
    if rank is not None and args.dp_size > 1:
        # Standard torch-distributed env vars (required by vllm's external_launcher backend).
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(args.dp_size)
        os.environ["LOCAL_WORLD_SIZE"] = str(args.dp_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        # vLLM-specific DP vars (some code paths still read these).
        os.environ["VLLM_DP_RANK"] = str(rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(rank)
        os.environ["VLLM_DP_WORLD_SIZE"] = str(args.dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = "127.0.0.1"
        os.environ["VLLM_DP_MASTER_PORT"] = "29500"

    # ---- LLM init ----
    print(
        f"[init rank={rank}] loading DeepSeek-V4-Pro (tp={args.tp_size}, dp={args.dp_size}, max_len={args.max_model_len})"
    )
    llm_kwargs: dict[str, Any] = {
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "trust_remote_code": True,
        "tokenizer_mode": "deepseek_v4",
        "tensor_parallel_size": args.tp_size,
        "kv_cache_dtype": "fp8",
        "block_size": 256,
        "max_model_len": args.max_model_len,
        "compilation_config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "custom_ops": ["all"],
        },
        "attention_config": {"use_fp4_indexer_cache": True},
        "enable_expert_parallel": True,
    }
    if args.dp_size > 1:
        llm_kwargs["data_parallel_size"] = args.dp_size
        # Tell vllm we're under an external multiprocess launcher (our spawn).
        # Without this, the single-process safety check refuses dp_size > 1.
        llm_kwargs["distributed_executor_backend"] = "external_launcher"
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=None)
    if args.thinking == "none":
        chat_kwargs = {"thinking": False}
    else:
        chat_kwargs = {"thinking": True, "reasoning_effort": args.thinking}

    # ---- Section loop ----
    for spec in SECTION_SPECS:
        n = spec.number
        todo = needed_per_section[n]
        if not todo:
            print(f"[section {n} rank={rank}] all cached, skipping")
            continue

        messages_batch = [
            build_section_messages(
                paper_text=p["paper_text"],
                section_prompt=spec.prompt,
                prior_outputs=[p["sections"].get(m, "") for m in range(1, n) if p["sections"].get(m)],
                system_preamble=SYSTEM_PREAMBLE,
            )
            for p in todo
        ]
        t0 = time.time()
        outputs = llm.chat(messages=messages_batch, sampling_params=sp, chat_template_kwargs=chat_kwargs)
        dt = time.time() - t0
        for p, out in zip(todo, outputs, strict=False):
            p["sections"][n] = strip_think(out.outputs[0].text)
        path = write_section_ckpt(papers, out_path, n, rank)
        print(
            f"[section {n} rank={rank}] batch of {len(todo)} in {dt:.1f}s ({dt / max(1, len(todo)):.2f}s/paper) -> {path.name}"
        )

    # ---- Pitch (structured output, no thinking to avoid JSON pollution) ----
    no_think = {"thinking": False}
    if pitch_todo:
        pitch_system = (
            "Extract the exact paper title from the PDF, then write a 2-3 sentence pitch designed to make a busy reader open the full summary.\n\n"
            "The pitch is NOT a paraphrase of the executive summary. Lead with the surprising or counter-intuitive finding. Be punchier. "
            "Use concrete numbers in the lead. Hint at conditions in passing.\n\n"
            "Hard rules: 2-3 sentences MAX. No bullets. Don't reuse executive-summary phrasings. Title must be exact. Plain Unicode for math."
        )
        sp_pitch = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=None,
            structured_outputs=StructuredOutputsParams(json=PitchOutput.model_json_schema()),
        )
        messages_batch = [
            [
                {"role": "system", "content": pitch_system},
                {
                    "role": "user",
                    "content": f"<paper>\n{p['paper_text'][:5000]}\n</paper>\n\nPaper Analysis (for context):\n"
                    + "\n\n".join(p["sections"][m] for m in sorted(p["sections"]))[:3000]
                    + "...",
                },
            ]
            for p in pitch_todo
        ]
        t0 = time.time()
        outputs = llm.chat(messages=messages_batch, sampling_params=sp_pitch, chat_template_kwargs=no_think)
        dt = time.time() - t0
        for p, out in zip(pitch_todo, outputs, strict=False):
            raw = strip_think(out.outputs[0].text)
            try:
                parsed = PitchOutput.model_validate_json(raw)
                p["_title"] = parsed.title
                p["_pitch_text"] = parsed.pitch
            except Exception as exc:
                print(f"[pitch parse fail rank={rank}] {p['arxiv_id']}: {exc!r}")
                p["_title"] = p.get("title") or ""
                p["_pitch_text"] = ""
        path = write_pitch_ckpt(papers, out_path, rank)
        print(f"[pitch rank={rank}] batch of {len(pitch_todo)} in {dt:.1f}s -> {path.name}")

    # ---- Category (structured output, no thinking) ----
    if cat_todo:
        cat_system = (
            f"Categorize the paper into one of these categories: {', '.join(CATEGORIES)}. "
            "Respond with ONLY the category name in the JSON."
        )
        sp_cat = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=None,
            structured_outputs=StructuredOutputsParams(json=CategoryOutput.model_json_schema()),
        )
        messages_batch = [
            [
                {"role": "system", "content": cat_system},
                {
                    "role": "user",
                    "content": (
                        f"Title: {p.get('_title', p.get('title', ''))}\n\n"
                        f"Pitch: {p.get('_pitch_text', '')}\n\n"
                        f"Full Summary:\n" + "\n\n".join(p["sections"][m] for m in sorted(p["sections"]))
                    ),
                },
            ]
            for p in cat_todo
        ]
        t0 = time.time()
        outputs = llm.chat(messages=messages_batch, sampling_params=sp_cat, chat_template_kwargs=no_think)
        dt = time.time() - t0
        for p, out in zip(cat_todo, outputs, strict=False):
            raw = strip_think(out.outputs[0].text)
            try:
                parsed = CategoryOutput.model_validate_json(raw)
                cat = parsed.category.strip().lower()
                if cat not in CATEGORIES:
                    cat = next((c for c in CATEGORIES if c in cat), FALLBACK_CATEGORY)
            except Exception as exc:
                print(f"[cat parse fail rank={rank}] {p['arxiv_id']}: {exc!r}")
                cat = FALLBACK_CATEGORY
            p["_category"] = cat
        path = write_cat_ckpt(papers, out_path, rank)
        print(f"[category rank={rank}] batch of {len(cat_todo)} in {dt:.1f}s -> {path.name}")

    # ---- Per-rank or single-process final assembly ----
    if rank is None:
        assemble_and_write(papers, out_path)
        print(f"[done] wrote {len(papers)} records to {out_path}")
    else:
        rank_out = out_path.with_suffix(f".rank{rank}.jsonl")
        with rank_out.open("w", encoding="utf-8") as fh:
            for p in papers:
                full = "\n\n".join(p["sections"][n] for n in sorted(p["sections"]))
                rec = {
                    "arxiv_id": p["arxiv_id"],
                    "title": p.get("_title", "") or p.get("title", ""),
                    "category": p.get("_category", "") or "uncategorized",
                    "pitch": p.get("_pitch_text", ""),
                    "summary": full,
                    "url": p.get("url"),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[done rank={rank}] wrote {len(papers)} records to {rank_out}")


# ---------------------------------------------------------------------------
# Multi-process DP orchestrator
# ---------------------------------------------------------------------------


def run_offline_dp(args) -> None:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(args.dp_size):
        p = ctx.Process(target=run_offline_worker, args=(args, rank))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # Merge per-rank final outputs into one
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as out:
        for r in range(args.dp_size):
            rp = out_path.with_suffix(f".rank{r}.jsonl")
            if not rp.exists():
                continue
            with rp.open("r", encoding="utf-8") as fh:
                for line in fh:
                    out.write(line)
    print(f"[merge] wrote merged output to {out_path}")


# ---------------------------------------------------------------------------
# Online mode (AsyncOpenAI against vllm serve)
# ---------------------------------------------------------------------------


async def online_batch(
    client,
    model: str,
    messages_batch: list,
    sampling_kwargs: dict,
    response_format: dict | None,
    concurrency: int = 22,
    stream_appender=None,
) -> list[str]:
    """Fire requests in parallel via asyncio.gather, bounded by a semaphore.

    Capping concurrent in-flight requests at ~the server's KV-cache concurrency limit
    avoids piling up thousands of pending HTTP connections that just queue and time out.
    The server batches whatever's in-flight; queuing more on top doesn't help and hurts.

    If stream_appender is provided, it's called as `await stream_appender(i, text)`
    each time a request completes. Use it to durably checkpoint per-request before
    the full gather() finishes -- so a crash mid-section preserves what was done.
    """
    sem = asyncio.Semaphore(concurrency)

    async def one(i, messages):
        async with sem:
            kwargs = dict(model=model, messages=messages, **sampling_kwargs)
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = await client.chat.completions.create(**kwargs)
            text = strip_think(resp.choices[0].message.content or "")
            if stream_appender is not None:
                await stream_appender(i, text)
            return text

    return await asyncio.gather(*[one(i, m) for i, m in enumerate(messages_batch)])


def run_online(args) -> None:
    setup_repo_root(args.repo_root)
    from openai import AsyncOpenAI

    from paper_pipeline.summarize.config import CATEGORIES, FALLBACK_CATEGORY
    from paper_pipeline.summarize.prompts import SECTION_SPECS, SYSTEM_PREAMBLE
    from paper_pipeline.summarize.schemas import CategoryOutput, PitchOutput

    out_path = Path(args.output)
    papers = load_papers(Path(args.input), limit=args.limit)
    print(f"[load] {len(papers)} papers")

    for spec in SECTION_SPECS:
        load_section_ckpts_into(papers, out_path, spec.number)
    load_pitch_ckpts_into(papers, out_path)
    load_cat_ckpts_into(papers, out_path)

    needed_per_section = {n: [p for p in papers if not p["sections"].get(n)] for n in (s.number for s in SECTION_SPECS)}
    pitch_todo = [p for p in papers if not p.get("_pitch_text")]
    cat_todo = [p for p in papers if not p.get("_category")]

    # timeout=None: don't time out individual requests. With asyncio.gather of 1274
    # requests, most sit in the server's queue while ~30-50 are processed in parallel.
    # Default 10-minute timeout fires before a queued request gets its turn.
    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY", timeout=None, max_retries=2)
    if args.thinking == "none":
        chat_kwargs = {"chat_template_kwargs": {"thinking": False}}
    else:
        chat_kwargs = {"chat_template_kwargs": {"thinking": True, "reasoning_effort": args.thinking}}

    sampling_kwargs_section = {"temperature": 1.0, "top_p": 1.0, "stream": False, "extra_body": chat_kwargs}
    sampling_kwargs_structured = {
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": False,
        "extra_body": {"chat_template_kwargs": {"thinking": False}},
    }

    only_cat = getattr(args, "only_cat", False)
    max_section = getattr(args, "max_section", 0)
    skip_pitch = getattr(args, "skip_pitch", False)
    skip_cat = getattr(args, "skip_cat", False)
    skip_assemble = getattr(args, "skip_assemble", False)

    async def go():
        if only_cat:
            print("[online] --only-cat: skipping section iteration AND pitch phase")
        if max_section > 0:
            print(f"[online] --max-section={max_section}: stopping after s{max_section}")
        for spec in SECTION_SPECS:
            if only_cat:
                break
            n = spec.number
            if max_section > 0 and n > max_section:
                print(f"[online] skipping s{n} (--max-section={max_section})")
                continue
            todo = needed_per_section[n]
            if not todo:
                print(f"[online section {n}] all cached, skipping")
                continue
            messages_batch = [
                build_section_messages(
                    paper_text=p["paper_text"],
                    section_prompt=spec.prompt,
                    prior_outputs=[p["sections"].get(m, "") for m in range(1, n) if p["sections"].get(m)],
                    system_preamble=SYSTEM_PREAMBLE,
                )
                for p in todo
            ]
            # Stream each completed request into s{n}.jsonl (append mode) so a
            # crash mid-section preserves progress. write_section_ckpt at the
            # end rewrites the file clean; the loader dedupes by arxiv_id so
            # duplicates from prior partial runs are harmless on resume.
            ckpt = ckpt_path(out_path, n, None)
            stream_lock = asyncio.Lock()
            stream_fh = ckpt.open("a", encoding="utf-8")
            stream_ids = [p["arxiv_id"] for p in todo]
            stream_done = [0]

            async def appender(i: int, text: str, _n=n):
                rec = {"arxiv_id": stream_ids[i], "section": _n, "text": text}
                async with stream_lock:
                    stream_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    stream_fh.flush()
                    stream_done[0] += 1
                    if stream_done[0] % 25 == 0 or stream_done[0] == len(stream_ids):
                        print(f"[online section {_n}] streamed {stream_done[0]}/{len(stream_ids)}", flush=True)

            t0 = time.time()
            try:
                outs = await online_batch(
                    client,
                    args.model,
                    messages_batch,
                    sampling_kwargs_section,
                    None,
                    concurrency=args.concurrency,
                    stream_appender=appender,
                )
            finally:
                stream_fh.close()
            dt = time.time() - t0
            for p, o in zip(todo, outs, strict=False):
                p["sections"][n] = o
            path = write_section_ckpt(papers, out_path, n, None)
            print(f"[online section {n}] {len(todo)} in {dt:.1f}s -> {path.name}")

        if pitch_todo and not only_cat and not skip_pitch:
            pitch_system = (
                "Extract the exact paper title from the PDF, then write a 2-3 sentence pitch designed to make a busy reader open the full summary. "
                "The pitch is NOT a paraphrase of the executive summary. Lead with the surprising finding. 2-3 sentences MAX. No bullets. Plain Unicode for math."
            )
            messages_batch = [
                [
                    {"role": "system", "content": pitch_system},
                    {
                        "role": "user",
                        "content": f"<paper>\n{p['paper_text'][:5000]}\n</paper>\n\n"
                        + "\n\n".join(p["sections"][m] for m in sorted(p["sections"]))[:3000]
                        + "...",
                    },
                ]
                for p in pitch_todo
            ]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "pitch_output", "schema": PitchOutput.model_json_schema()},
            }
            outs = await online_batch(
                client,
                args.model,
                messages_batch,
                sampling_kwargs_structured,
                response_format,
                concurrency=args.concurrency,
            )
            for p, raw in zip(pitch_todo, outs, strict=False):
                try:
                    parsed = PitchOutput.model_validate_json(raw)
                    p["_title"] = parsed.title
                    p["_pitch_text"] = parsed.pitch
                except Exception:
                    p["_title"] = p.get("title") or ""
                    p["_pitch_text"] = ""
            write_pitch_ckpt(papers, out_path, None)

        if cat_todo and not skip_cat:
            cat_system = f"Categorize the paper into one of these categories: {', '.join(CATEGORIES)}. Respond with ONLY the category name in the JSON."
            messages_batch = [
                [
                    {"role": "system", "content": cat_system},
                    {
                        "role": "user",
                        "content": f"Title: {p.get('_title', '')}\nPitch: {p.get('_pitch_text', '')}\n\nFull Summary:\n"
                        + "\n\n".join(p["sections"][m] for m in sorted(p["sections"])),
                    },
                ]
                for p in cat_todo
            ]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "category_output", "schema": CategoryOutput.model_json_schema()},
            }
            outs = await online_batch(
                client,
                args.model,
                messages_batch,
                sampling_kwargs_structured,
                response_format,
                concurrency=args.concurrency,
            )
            for p, raw in zip(cat_todo, outs, strict=False):
                try:
                    parsed = CategoryOutput.model_validate_json(raw)
                    cat = parsed.category.strip().lower()
                    if cat not in CATEGORIES:
                        cat = next((c for c in CATEGORIES if c in cat), FALLBACK_CATEGORY)
                except Exception:
                    cat = FALLBACK_CATEGORY
                p["_category"] = cat
            write_cat_ckpt(papers, out_path, None)

    asyncio.run(go())
    if not only_cat and not skip_assemble:
        assemble_and_write(papers, out_path)
        print(f"[online done] wrote {len(papers)} records to {out_path}")
    elif skip_assemble:
        print("[online done] sections written, skipped pitch/cat/assemble per flags")
    else:
        print(f"[online done] --only-cat: wrote cat ckpt to {out_path.with_suffix('.cat.jsonl').name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    # CWD-relative defaults so you can just `cd /your/work/dir && python3 ... offline_regen.py`.
    ap.add_argument("--input", default="regen_input.jsonl")
    ap.add_argument("--output", default="regen_output.jsonl")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--mode", default="offline-dp", choices=["offline-tp", "offline-dp", "online"])
    ap.add_argument("--tp-size", type=int, default=1, help="tensor parallel (default 1 for DP+EP)")
    ap.add_argument("--dp-size", type=int, default=8, help="data parallel (default 8 for B300)")
    ap.add_argument("--max-model-len", type=int, default=200000)
    ap.add_argument("--thinking", default="none", choices=["none", "high", "max"])
    ap.add_argument("--limit", type=int, default=0)
    # Online-mode-only:
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Online mode: max in-flight requests (semaphore). Match the server's KV-cache concurrency.",
    )
    ap.add_argument(
        "--only-cat",
        action="store_true",
        help="Skip section iteration AND pitch phase; run ONLY category classification. "
        "Useful when sections are already done (cached) and you just want to (re)classify.",
    )
    ap.add_argument(
        "--max-section",
        type=int,
        default=0,
        help="If >0, run sections 1..N only and skip later sections. e.g. 4 = run s1-s4 "
        "and skip s5/s6/s7. Combined with --skip-pitch and --skip-cat for a partial "
        "regen that saves ~50%% GPU time vs the full 7-section pipeline.",
    )
    ap.add_argument(
        "--skip-pitch", action="store_true", help="Skip the pitch generation phase (one-sentence intro per paper)."
    )
    ap.add_argument("--skip-cat", action="store_true", help="Skip the category classification phase.")
    ap.add_argument(
        "--skip-assemble",
        action="store_true",
        help="Skip the final assemble-into-summary phase. Useful for partial "
        "runs where you only want the per-section .jsonl outputs.",
    )
    args = ap.parse_args()

    if args.mode == "online":
        run_online(args)
    elif args.mode == "offline-tp":
        # single-process TP=N
        args.dp_size = 1
        if args.tp_size == 1:
            args.tp_size = 8
        run_offline_worker(args, rank=None)
    elif args.mode == "offline-dp":
        if args.dp_size <= 1:
            # degrade to TP=8 single process
            args.tp_size = 8
            run_offline_worker(args, rank=None)
        else:
            run_offline_dp(args)
    else:
        raise SystemExit(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
