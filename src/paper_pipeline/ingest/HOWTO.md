# Daily Papers — How-To

Pipeline for fetching Hugging Face Daily Papers, LLM-scoring them against your reading history, and browsing/marking the results in a local web UI.

## Moving parts

| File | Role |
|---|---|
| `hf_daily_papers.py` | Fetch HF Daily Papers (+ arxiv metadata), score each via local SGLang LLM, write `all_scored.json` |
| `papers_by_score.py` | Split `all_scored.json` into per-score markdown buckets for manual LLM re-verification |
| `tag_papers.py` | Standalone batch tagger: JSONL in → categorized JSONL out (no DB) |
| `paper_server.py` | FastAPI JSON backend (`paper-server`): serves the scored feed + handles "mark interested" POSTs via `NeonDB.mark_interested`. The Svelte app in `svelte-ui/` is the frontend. |
| `examples.py` | Shared reading-history loader (Neon + `../core/external_papers.db` + repo-root `docs/**/*.md`) — used by `hf_daily_papers.py` and `papers_by_score.py` |
| `../core/neon_db.py` | Neon Postgres client (`NeonDB`). `init_schema`, `save_paper`, `mark_interested`, etc. |
| Neon `"nextjs-ui_paper"` | Main DB. `interested=1` rows are the reading history. Connection from `DATABASE_URL` |
| `../core/external_papers.db` | Secondary SQLite DB scanned for more history examples (read by `../core/external.py`) |
| repo-root `docs/<category>/*.md` | Third history source — H1 of each file becomes an example title under `<category>` |

## Data flow

```
HF Daily Papers API ──┐
                      ├─► hf_daily_papers.py ──► papers_out/all_scored.json
arxiv.org API ────────┘        │                         │
                               │ (examples)              │
Neon nextjs-ui_paper ──────────┤                         ▼
external_papers.db ────────────┤              paper_server.py ◄── browser
docs/**/*.md ──────────────────┘                         │
                                                         ▼
                                      Neon nextjs-ui_paper (interested=1)
```

## One-time setup

1. Start a local SGLang server (default `http://localhost:30000/v1`). Override with `SGLANG_BASE_URL` env var. Modal endpoints are commented at the top of `hf_daily_papers.py`.
2. `uv sync` at the repo root.
3. Make sure `DATABASE_URL` is discoverable by `src/paper_pipeline/core/neon_db.py` (repo `.env` or `../company-scraper/nextjs-ui/.env`), then `uv run python -m paper_pipeline.core.neon_db init-schema` once.

## Daily workflow

```bash
# 1. Fetch + score today's papers
uv run paper-fetch --out-dir src/paper_pipeline/ingest/papers_out

# Optional: date range (backfill)
uv run paper-fetch \
    --out-dir src/paper_pipeline/ingest/papers_out \
    --from 2026-01-01 --to 2026-04-12

# Flags:
#   --threshold 6     min score written to relevant.md (default 6)
#   --concurrency N   max in-flight scoring requests (default 4096)
#   --date YYYY-MM-DD single day instead of today
```

Outputs in `papers_out/`:
- `all_scored.json` — every paper + score/reason (cache-resumable; rerun is cheap)
- `relevant.md` — score ≥ threshold, grouped/ranked for skimming

`all_scored.json` is checkpointed every 25 papers, so Ctrl-C is safe.

## Browse + mark interested

```bash
uv run paper-server
# or:
uv run paper-server --port 8787 \
    --json ./src/paper_pipeline/ingest/papers_out/all_scored.json
```

Then run the Svelte UI (`svelte-ui/`, see its README) — its Vite dev server proxies `/api/papers`, `/interested`, etc. to this backend on :8787. Selecting papers and hitting "mark" issues `POST /interested`, which calls `NeonDB.mark_interested` to flip `interested=1` in Neon. Those rows become reading-history examples on the **next** `paper-fetch` run, so the scorer gets sharper over time.

## Scoring model in one paragraph

`SYSTEM_PROMPT` in `hf_daily_papers.py` tells the LLM to score 1–10 on whether the candidate solves the **same specific problem** as any paper in the reading history (loaded from Neon `interested=1`, `src/paper_pipeline/core/external_papers.db`, and `docs/**/*.md` via `examples.py`). It also applies a -2 penalty for unknown-author papers with no org, <5 upvotes, and <20 GitHub stars. Output is JSON-schema-validated into `ScoreOutput { score, similar_paper, reason }`. Edit the prompt or the `AUTHOR QUALITY PENALTY` block there if scores look off.

## Common tweaks

- **Change the scoring model**: edit `MODEL` at top of `hf_daily_papers.py`.
- **Change history sources**: edit `examples.py:load_examples()`.
- **Per-score review buckets**: `uv run python -m paper_pipeline.ingest.papers_by_score` to split `all_scored.json` into markdown files per score.
- **Standalone tagging of external JSONL**: `tag_papers.py` / `paper-tag` (no DB, self-contained).

## Troubleshooting

- **`arxiv API ... 429`**: the arxiv enrichment has 7 retries with exponential backoff; it'll recover. Range fetches skip arxiv enrichment entirely (see XXX note in `fetch_papers_range`).
- **`fd limit`**: on macOS the script auto-raises soft fd limit; if scoring hangs at high concurrency, that's the culprit.
- **Empty history**: check Neon has rows with `interested=1` (`SELECT count(*) FROM "nextjs-ui_paper" WHERE interested = 1`) — otherwise the scorer has nothing to compare against and everything scores low.
- **`DATABASE_URL is not set`**: `src/paper_pipeline/core/neon_db.py` couldn't find `.env`. Export it manually or drop it into this repo's `.env`.
- **Server health check fails**: `SGLANG_BASE_URL` wrong, or server not up yet. Scoring still runs but will error per-request.
