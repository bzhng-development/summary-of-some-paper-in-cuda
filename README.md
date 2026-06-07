# Paper Summaries

A personal arxiv-paper curation + summarization pipeline: ~20k papers in Neon Postgres, LLM-scored, multi-tagged, and long-form-summarized. Public reader: **https://paper-graph-ui.vercel.app**

The Python side is the installable **`paper_pipeline`** package under `src/`. Run `uv sync` once to editable-install it (and the `paper-*` console scripts below). See **[`CLAUDE.md`](CLAUDE.md)** for the full package map, data flow, and the paper-curation playbook.

## Local triage UI (Svelte + FastAPI)

Two processes:

```bash
# 1. JSON/API backend (serves all_scored.json + Neon writes)
uv run paper-server --port 8787

# 2. Svelte frontend (Vite proxies /api/papers etc. to :8787)
cd svelte-ui && pnpm install && pnpm dev      # http://localhost:5173
```

Point at a remote backend with `PAPER_SERVER_URL=http://host:port pnpm dev`. More in `svelte-ui/README.md`.

## Score new papers

DeepSeek-V4-Pro on the local/tunnelled SGLang endpoint by default (`DEEPSEEK_BASE_URL`, default `http://localhost:30000/v1`; set `SCORER_BACKEND=sglang` + `SGLANG_BASE_URL` to point elsewhere).

```bash
uv run paper-fetch \
  --out-dir src/paper_pipeline/ingest/papers_out \
  --from 2026-03-01 --to 2026-04-18 --concurrency 16
```

Append-only: `all_scored.json` is re-loaded on each flush and new rows merged, so date-scoped re-runs don't truncate prior work.

## Push scored papers into Neon

```bash
uv run paper-sync --skip-arxiv    # mirror all_scored.json → Neon (fast)
uv run paper-sync                 # same, plus enrich from the arxiv API
```

Mirrors every field — arxiv metadata **and** scoring columns — via `NeonDB.batch()` (one connection, commit every 500). Idempotent, safe to re-run.

## Summarize a paper

```bash
uv run paper-summarize --url https://arxiv.org/abs/XXXX.XXXXX          # long-form, 7-section
uv run paper-summarize --urls "url1,url2,url3" --concurrency 5
uv run paper-summarize-single --url https://arxiv.org/abs/XXXX.XXXXX   # single-pass variant
```

Summaries follow a 7-section framework (Executive Summary, Context, Technical Approach, Insights, Experiments, Limitations, Implications) and land in Neon + the canonical viewing store `paper-graph-ui/src/content/papers/`.

## Preview

<img width="2946" height="1592" alt="paper viewer" src="https://github.com/user-attachments/assets/7e352c41-ce56-43e9-b283-21e225663ea6" />
