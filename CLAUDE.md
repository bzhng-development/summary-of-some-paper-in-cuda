# Environment

This is a uv-managed project. Use `uv run python` to run scripts (auto-uses the `.venv`).

```bash
uv run python script.py      # Run a script
uv add <package>              # Add a dependency
uv sync                       # Reinstall/sync all deps
```

Do not use `pip install` or `source .venv/bin/activate`.

# Data paths

There's one built artifact (`papers.db`) and several committed input files that feed it. Understanding where each lives matters because the built DB is too large for GitHub and is rebuilt from the inputs.

## The built DB

`local_data/papers.db` — 130 MB, **gitignored**, NOT in any commit.

- Schema: defined in `database.py` (`_migrate_sync` + `_EXTRA_COLUMNS`), extended with scoring columns: `score`, `similar_paper`, `score_reason`, `tag_category_v2`, `tag_confidence`, `tag_reason`, `score_source`.
- Current content: 13,448 rows, 964 `interested=1` (reading history), 12,664 with LLM scores, 13,023 tagged.
- Why gitignored: GitHub rejects files >100 MB. Rebuild locally.
- Resolved path: `database.py` sets `DB_PATH = <repo>/local_data/papers.db` relative to its own `__file__`, so scripts work from any cwd. Override via `PAPERS_DB_PATH` env var.

To rebuild from scratch:

```bash
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
| `external_papers.db` | 560 K | Hand-seeded classical papers (GANs, Deep Learning, etc.) with non-arxiv IDs. Merged into `papers.db` by `sync_db.py`. |
| `arxiv_index.txt` | small | Free-form list of arxiv IDs to ensure-exist in the DB as stubs. |
| `docs/**/*.md` | many | Per-paper summary markdown (organized by category dir). `sync_db.py` parses these to backfill `summary` on stub rows; `hf_daily_papers.py`'s `load_examples_from_repo()` also uses filenames as a reading-history source. |
| `links.txt`, `arxiv_index.txt` | small | Reading-list text files. |

## Scratch / backup files (local only — gitignored)

Everything under `local_data/` is gitignored. Contents beyond the built DB:

| Path | Role |
|---|---|
| `local_data/papers.db` | **THE** built DB used by every script |
| `local_data/papers_unified.db` | Pre-scoring v1 unified DB (stash3 ∪ fresh_310pm merge) |
| `local_data/papers_unified_v1_backup.db` | Read-only immutable snapshot of the above |
| `local_data/papers_stash3.db` | Extract of `stash@{3}^2:papers.db` (writable working copy) |
| `local_data/papers_stash3_backup.db` | Read-only immutable copy |

The `*_backup.db` files have `chmod -w` set — treat them as recovery points.

## Reading-history sources (what `load_examples_from_repo` reads)

`daily_papers/hf_daily_papers.py:load_examples_from_repo()` and `daily_papers/papers_by_score.py:load_examples_block()` both union three sources:

1. `local_data/papers.db` (fallback: top-level `papers.db` for legacy checkouts) — `SELECT title, category FROM papers WHERE interested = 1`
2. `external_papers.db` — `SELECT title, category FROM papers`
3. `docs/**/*.md` — H1 (`# Title`) of each file, categorized by parent dir

These are merged by title dedup. Keeping `papers.db` populated with `interested=1` rows is what makes the scorer sharper over time.

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
                                    │  sync_db.py                              │
                                    │    1. migrate schema                     │
                                    │    2. backfill from docs/                │
                                    │    3. absorb arxiv_index.txt             │
                                    │    4. merge external_papers.db           │
                                    │    5. absorb all_scored.json             │
                                    │    6. enrich from arxiv API              │
                                    │    7. import tags from jsonl             │
                                    └──────────────┬───────────────────────────┘
                                                   ▼
                                        local_data/papers.db   ◄── paper_server.py
                                        (built, gitignored)          writes interested=1
                                                   │                 via the web UI
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

