# Environment

This is a uv-managed project. Use `uv run python` to run scripts (auto-uses the `.venv`).

```bash
uv run python script.py      # Run a script
uv add <package>              # Add a dependency
uv sync                       # Reinstall/sync all deps
```

Do not use `pip install` or `source .venv/bin/activate`.

# Generating paper summaries (runbook)

Full long-form paper summaries are produced by **`multi_prompt.py`**, NOT `main.py`. `multi_prompt.py` is a thin shim that re-exports the public API from the `multi_prompt_pkg/` package and dispatches the CLI via `multi_prompt_pkg.cli.main`. It fetches an arxiv PDF, extracts text with PyMuPDF, and sends it to a local LLM for the full 7-section teach-through (Executive Summary → Context → Technical Approach → Insights → Experiments → Limitations → Implications). Writes into the Neon Postgres `"nextjs-ui_paper"` table by default.

## Prereqs

1. Start a local OpenAI-compatible server at `http://localhost:30000/v1` (defined as `LOCAL_BASE_URL` in `multi_prompt_pkg/config.py`). SGLang, vLLM, llama.cpp server — anything OpenAI-compatible. The script sends `model="default"`; the server serves whatever it loaded.
2. Have `DATABASE_URL` set to a Neon Postgres connection string. `neon_db.py` auto-loads `.env` from this repo and falls back to `../company-scraper/nextjs-ui/.env` (see `_load_env`).

## Single paper

```bash
uv run python multi_prompt.py --url https://arxiv.org/abs/2312.07104
```

## Batch

```bash
uv run python multi_prompt.py --urls "url1,url2,url3" --concurrency 5
```

## Dynamic mode (continuous worker)

Polls Neon every `--interval` seconds for stub rows missing a summary (`NeonDB.get_stubs_without_summary`) and processes them. Run it in a tmux pane and let it chew through the backlog:

```bash
uv run python multi_prompt.py --dyn --concurrency 4096 --interval 30
```

`--concurrency` here is the semaphore ceiling for concurrent papers; the actual network cap is set by the single shared httpx client inside the script.

## Backfill

Re-summarize every `interested=1` paper currently missing a `summary`:

```bash
uv run python multi_prompt.py --backfill --concurrency 8
```

## Remote runs → JSONL sidecar

When running on a remote box that can't reach Neon, write results to JSONL and absorb later:

```bash
uv run python multi_prompt.py --dyn --jsonl ./runs/remote.jsonl
```

Then import the JSONL into Neon via `sync_db.py`'s absorb helpers (`import_jsonl.py` also exists for one-off imports).

## Model overrides

- `--gemini` — use Gemini 3 Flash via OpenRouter/Google AI Studio free tier (bypasses the local server).
- `--model <name>` — override the `"default"` model string sent in the chat completion request. Only meaningful if your local server dispatches by name.
- `--many-pass` — use the old 7-section **sequential** pipeline (one LLM call per section, with prior sections in context). Default is a single big call; `--many-pass` is for smaller models that choke on the full context.

## Where summaries land

- **Default:** `local_data/papers.db`, column `summary`, keyed by arxiv_id. `paper_server.py` and the docs build read from here.
- **`--jsonl`:** append-only JSONL file; re-imported into the DB later.
- **`docs/<category>/*.md`:** NOT written by `multi_prompt.py`. Those are hand-curated / one-off exports. Don't expect `multi_prompt.py` to populate them.

## Tuning for port-forwarded servers

If hitting the LLM through VSCode port-forward or SSH -L, high concurrency causes connection resets. Drop `--concurrency` first (try 4–8); if still flaky, the httpx client limits live inside `multi_prompt_pkg/llm.py` — look for the `AsyncOpenAI(...)` construction and cap `max_connections` there. See `daily_papers/hf_daily_papers.py` for the analogous single-client pattern.

# Data paths

The source of truth is the Neon Postgres `"nextjs-ui_paper"` table (see `neon_db.py`). Several committed input files feed it; `local_data/papers.db` is a legacy SQLite snapshot kept around for migration/backup only.

## The Neon DB (source of truth)

- Table: `"nextjs-ui_paper"` in a Neon Postgres database. Connection comes from `DATABASE_URL`.
- Schema: defined in `neon_db.py` (`NeonDB.init_schema` + the `SCHEMA_COLUMNS` tuple). Includes scoring columns `score`, `similar_paper`, `score_reason`, `tag_category_v2`, `tag_confidence`, `tag_reason`, `score_source`.
- `DATABASE_URL` discovery: `neon_db._load_env` reads this repo's `.env` first, then walks up to `../company-scraper/nextjs-ui/.env` as a shared fallback. Export in the shell to override.
- CLI: `uv run python neon_db.py init-schema` (idempotent) and `uv run python neon_db.py migrate --sqlite local_data/papers.db` (copies the legacy SQLite snapshot into Neon; uses partial upsert so it's safe to re-run).

## Legacy SQLite snapshot

`local_data/papers.db` — ~130 MB, **gitignored**, NOT in any commit. This is a frozen copy of the pre-refactor state (13,448 rows, 964 `interested=1`, 12,664 scored, 13,023 tagged) and is the source that `neon_db.py migrate` reads from. Nothing in the live pipeline writes to it anymore. `local_data/` is gitignored so it can also hold the `*_backup.db` recovery snapshots.

## Rebuilding from the committed inputs

```bash
uv run python neon_db.py init-schema      # ensure the table exists
uv run python sync_db.py                  # full pipeline (uses arxiv API)
uv run python sync_db.py --skip-arxiv     # fast — use committed data only
```

## Committed input files (tracked in git)

| Path | Size | Role |
|---|---|---|
| `daily_papers/papers_out/all_scored.json` | 40 M | LLM scoring pass output. 12,664 papers with `score` (1-10), `similar_paper`, `reason`, `tag_category`, `tag_confidence`, `tag_reason`. Produced by `daily_papers/hf_daily_papers.py` against the reading history. This is the source of truth for scores. |
| `tagged_papers.jsonl` | 8.6 M | Tagger output. 13,023 rows: `{arxiv_id, title, category, confidence, reason}`. Consumed by `sync_db.py --import-tags`. |
| `papers_to_tag.jsonl` | 19 M | Tagger input: 13,025 rows of `{arxiv_id, title, abstract}`. Produced by `sync_db.py --export`, fed into `daily_papers/tag_papers.py`. |
| `results_modal3.jsonl` | small | Older run of LLM-generated long-form summaries (441 rows). Sidecar reference. |
| `external_papers.db` | 560 K | Hand-seeded classical papers (GANs, Deep Learning, etc.) with non-arxiv IDs. Still a SQLite file — read directly by `external.py` and merged into Neon by `sync_db.py`. |
| `arxiv_index.txt` | small | Free-form list of arxiv IDs to ensure-exist in Neon as stubs. |
| `docs/**/*.md` | many | Per-paper summary markdown (organized by category dir). `sync_db.py` parses these to backfill `summary` on stub rows; `daily_papers/examples.py:load_examples_from_repo()` also uses filenames as a reading-history source. |
| `links.txt`, `arxiv_index.txt` | small | Reading-list text files. |

## Scratch / backup files (local only — gitignored)

Everything under `local_data/` is gitignored:

| Path | Role |
|---|---|
| `local_data/papers.db` | Legacy SQLite snapshot. Read-only in the live pipeline; the source `neon_db.py migrate` reads from |
| `local_data/papers_unified.db` | Pre-scoring v1 unified DB (stash3 ∪ fresh_310pm merge) |
| `local_data/papers_unified_v1_backup.db` | Read-only immutable snapshot of the above |
| `local_data/papers_stash3.db` | Extract of `stash@{3}^2:papers.db` (writable working copy) |
| `local_data/papers_stash3_backup.db` | Read-only immutable copy |

The `*_backup.db` files have `chmod -w` set — treat them as recovery points.

## Reading-history sources (what `load_examples_from_repo` reads)

`daily_papers/examples.py:load_examples_from_repo()` (shared by `hf_daily_papers.py` and `papers_by_score.py`) unions three sources:

1. Neon `"nextjs-ui_paper"` via `NeonDB.get_interested()` — the `interested = 1` rows
2. `external_papers.db` — `SELECT title, category FROM papers` (still SQLite; read by `external.py`)
3. `docs/**/*.md` — H1 (`# Title`) of each file, categorized by parent dir

These are merged by title dedup. Keeping Neon populated with `interested=1` rows (via `paper_server.py`'s "mark interested" POST) is what makes the scorer sharper over time.

## Data flow — full picture

```
                                                              ┌─────────────────────────┐
HF Daily Papers API ──┐                                       │  docs/<cat>/*.md        │
                      ├─► hf_daily_papers.py ──► papers_out/  │  (hand-curated          │
arxiv.org API ────────┘       all_scored.json ────────────┐   │   summaries per paper)  │
                                                          │   └──────────┬──────────────┘
                                                          │              │
                                          ┌───────────────┴───────┐      │
                                          │                       │      │
tagged_papers.jsonl ◄─── tag_papers.py ◄──┤   papers_to_tag.jsonl │      │
                                          │   (from sync_db --export)    │
                                          │                              │
                                          ▼                              ▼
                                    ┌──────────────────────────────────────────┐
                                    │  sync_db.py  (all writes via NeonDB)     │
                                    │    1. init_schema (Neon)                 │
                                    │    2. backfill from docs/                │
                                    │    3. absorb arxiv_index.txt             │
                                    │    4. merge external_papers.db           │
                                    │    5. absorb all_scored.json             │
                                    │    6. enrich from arxiv API              │
                                    │    7. import tags from jsonl             │
                                    └──────────────┬───────────────────────────┘
                                                   ▼
                                    Neon "nextjs-ui_paper"   ◄── paper_server.py
                                    (DATABASE_URL)               writes interested=1
                                                   │             via the web UI
                                                   ▼
                                         (read back as reading
                                          history on next scoring run)
```

## Legacy / recovery refs

If something goes wrong, these git refs preserve earlier states:

- `backup/main-mintlify-era-*` → the 9-commit mintlify attempt (0b6227b). Also on `public-push` branch.
- `backup/first-restore-*` → a first restore attempt that kept 13 one-shot curation scripts (74f3085). Also on `inspect-local-stash` branch.
- `backup/pre-mintlify-rebuild-*` → the canonical restore commit (4bc4943).
- `backup/stash3-papers-db-*` → the stash that held the original 106 MB `papers.db` with 13,255 rows.
- `backup/stash0-curation-*` → stash with the `insert_batch*.py` one-shot classical-paper loaders.
- `stash@{0}..stash@{6}` — never drop these without checking.

