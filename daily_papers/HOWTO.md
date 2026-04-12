# Daily Papers — How-To

Pipeline for fetching Hugging Face Daily Papers, LLM-scoring them against your reading history, and browsing/marking the results in a local web UI.

## Moving parts

| File | Role |
|---|---|
| `hf_daily_papers.py` | Fetch HF Daily Papers (+ arxiv metadata), score each via local SGLang LLM, write `all_scored.json` |
| `papers_by_score.py` | Split `all_scored.json` into per-score markdown buckets for manual LLM re-verification |
| `tag_papers.py` | Standalone batch tagger: JSONL in → categorized JSONL out (no DB) |
| `paper_viewer.py` | Pure HTML generator — turns `all_scored.json` into a static interactive page |
| `paper_server.py` | FastAPI wrapper: serves the viewer, handles "mark interested" POSTs into `papers.db` |
| `../database.py` | SQLite schema + `save_paper_sync` used by the server to persist interested papers |
| `../papers.db` | Main DB. Table `papers`; `interested=1` rows are the reading history |
| `../external_papers.db` | Secondary DB scanned for more history examples |
| `../docs/<category>/*.md` | Third history source — H1 of each file becomes an example title under `<category>` |

## Data flow

```
HF Daily Papers API ──┐
                      ├─► hf_daily_papers.py ──► papers_out/all_scored.json
arxiv.org API ────────┘        │                         │
                               │ (examples)              │
papers.db ─────────────────────┤                         ▼
external_papers.db ────────────┤              paper_server.py ◄── browser
docs/**/*.md ──────────────────┘                         │
                                                         ▼
                                              papers.db (interested=1)
```

## One-time setup

1. Start a local SGLang server (default `http://localhost:30000/v1`, model `Qwen/Qwen3.5-122B-A10B`). Override with `SGLANG_BASE_URL` env var. Modal endpoints are commented at the top of `hf_daily_papers.py`.
2. `uv sync` at the repo root.

## Daily workflow

```bash
# 1. Fetch + score today's papers
uv run python daily_papers/hf_daily_papers.py --out-dir ./daily_papers/papers_out

# Optional: date range (backfill)
uv run python daily_papers/hf_daily_papers.py \
    --out-dir ./daily_papers/papers_out \
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
uv run python daily_papers/paper_server.py
# or:
uv run python daily_papers/paper_server.py --port 8787 \
    --json ./daily_papers/papers_out/all_scored.json
```

Open `http://localhost:8000`. Click papers to select, hit the mark button — `POST /interested` saves them into `papers.db` with `interested=1`. Those rows become reading-history examples on the **next** `hf_daily_papers.py` run, so the scorer gets sharper over time.

Static (no server) alternative:

```bash
uv run python daily_papers/paper_viewer.py \
    --json ./daily_papers/papers_out/all_scored.json \
    --out  ./daily_papers/papers_out/viewer.html \
    --min-score 5
```

## Scoring model in one paragraph

`SYSTEM_PROMPT` in `hf_daily_papers.py` tells the LLM to score 1–10 on whether the candidate solves the **same specific problem** as any paper in the reading history (loaded from `papers.db interested=1`, `external_papers.db`, and `docs/**/*.md`). It also applies a -2 penalty for unknown-author papers with no org, <5 upvotes, and <20 GitHub stars. Output is JSON-schema-validated into `ScoreOutput { score, similar_paper, reason }`. Edit the prompt or the `AUTHOR QUALITY PENALTY` block there if scores look off.

## Common tweaks

- **Change the scoring model**: edit `MODEL` at top of `hf_daily_papers.py`.
- **Change history sources**: edit `load_examples_from_repo()`.
- **Per-score review buckets**: `uv run python daily_papers/papers_by_score.py` to split `all_scored.json` into markdown files per score.
- **Standalone tagging of external JSONL**: `daily_papers/tag_papers.py` (no DB, self-contained).

## Troubleshooting

- **`arxiv API ... 429`**: the arxiv enrichment has 7 retries with exponential backoff; it'll recover. Range fetches skip arxiv enrichment entirely (see XXX note in `fetch_papers_range`).
- **`fd limit`**: on macOS the script auto-raises soft fd limit; if scoring hangs at high concurrency, that's the culprit.
- **Empty history**: check `papers.db` has rows with `interested=1` — otherwise the scorer has nothing to compare against and everything scores low.
- **Server health check fails**: `SGLANG_BASE_URL` wrong, or server not up yet. Scoring still runs but will error per-request.
