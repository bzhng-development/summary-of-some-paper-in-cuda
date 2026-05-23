# TL;DR — where the papers live

- **Source of truth:** Neon Postgres, table `"nextjs-ui_paper"` (via `$DATABASE_URL`, see `neon_db.py`).
- **Legacy SQLite snapshot:** `local_data/papers.db` (~130 MB, **gitignored**, 13,448 rows). Read-only in the live pipeline; only `neon_db.py migrate` reads from it.
- **Hand-seeded classical papers:** `external_papers.db` (~560 KB, tracked in git). Merged into Neon by `sync_db.py`; read directly by `external.py`.
- Scored paper feed (JSON cache): `daily_papers/papers_out/all_scored.json` (~41 MB, tracked). Produced by `daily_papers/hf_daily_papers.py`.

# Paper curation playbook (any time window)

How to pick papers to mark `interested=1` from a given month, week, or arbitrary date range. Reuse for "recs from YYYY-MM", "what did I miss in $WINDOW", etc.

## 1. Make sure the window is absorbed + enriched

`all_scored.json` has every paper's score/reason, but **not** always authors/abstract/published. Without authors you can't apply the authorship filter, so enrich first. Scope to the arxiv_id YYMM prefix(es) covering your window — e.g. April 2026 = `2604.*`.

```python
# uv run python - <<'PY'
import json
from pathlib import Path
from neon_db import NeonDB
from daily_papers.hf_daily_papers import fetch_arxiv_metadata

PREFIXES = ("2604.",)  # adjust for window
db = NeonDB()
data = json.loads(Path("daily_papers/papers_out/all_scored.json").read_text())
targets = [e for e in data if e.get("arxiv_id","").startswith(PREFIXES) and e.get("title")]

with db.batch() as b:
    for e in targets:
        aid = e["arxiv_id"]
        b.save_paper(aid, title=e["title"], abstract=e.get("summary") or None,
                     url=f"https://arxiv.org/abs/{aid}", upvotes=e.get("upvotes"),
                     github=e.get("github"), github_stars=e.get("github_stars"),
                     authors=e.get("authors") or None, affiliations=e.get("affiliations") or None,
                     organization=e.get("organization"), org_fullname=e.get("org_fullname"),
                     categories=e.get("categories") or None, primary_category=e.get("primary_category"),
                     published=e.get("published"), score=e.get("score"),
                     similar_paper=e.get("similar_paper"), score_reason=e.get("reason"),
                     tag_category_v2=e.get("tag_category"), tag_confidence=e.get("tag_confidence"),
                     tag_reason=e.get("tag_reason") or None,
                     score_source="all_scored.json" if e.get("score") is not None else None)

ids = [e["arxiv_id"] for e in targets]
meta = fetch_arxiv_metadata(ids)
with db.batch() as b:
    for aid, m in meta.items():
        b.save_paper(aid, title=m.title or None, abstract=m.abstract or None,
                     authors=m.authors or None, affiliations=m.affiliations or None,
                     categories=m.categories or None, primary_category=m.primary_category,
                     arxiv_comment=m.comment, published=m.published,
                     journal_ref=m.journal_ref, doi=m.doi)
PY
```

This runs in under a minute for ~1k papers (arxiv API, 100/batch, 1s between).

Note: `published` is unreliable as a filter (many rows null from the JSON feed). Prefer `id LIKE 'YYMM.%'` as the window selector — arxiv IDs encode submission year+month and never lie.

## 2. Walk scores top-down, with tiered strictness

Query Neon filtered to `interested=0 AND id LIKE 'YYMM.%' AND score=N`, ordered by `upvotes DESC`:

| Tier | Action |
|---|---|
| **score=10** | Read every one. These are rare — April 2026 had exactly 1 across 360 papers. |
| **score=9** | Filter by domain fit AND author signal. Expect ~50% cut. |
| **score=8** | Stricter — only pick if BOTH established authors AND clear domain hit. |
| **score=7 and below** | Skip by default. The good stuff concentrates at 9–10. Open exceptions only for explicit keyword searches (e.g. "anything on speculative decoding this month"). |

## 3. Signal weighting per candidate

Apply in this order, each one can veto:

1. **Domain fit** (for this repo): RL-training/RLVR/GRPO, CUDA kernels, inference-optimization, architecture (attention, linear, KV cache), LLM systems (SGLang, vLLM), pre-training, agents with a systems flavor.
2. **Authorship**. Scan the full author list, not just first+last. Strong positives:
   - **Top-tier names**: Fei-Fei Li, Yejin Choi, Manling Li, Xipeng Qiu, Nan Duan, Wayne Xin Zhao, Jiwei Li, Quanshi Zhang, Jing Shao, Deyi Xiong, Sung Ju Hwang, Joyce Chai, Chuang Gan, Ming Zhou, Ngai Wong, Shanchuan Lin, Haoqi Fan, Dawn Song, Xia Hu, Li Fei-Fei. (Running list — extend as you discover.)
   - **Institutional signal**: FAIR/Meta, DeepMind, Google Research, Microsoft Research, NVIDIA, OpenAI, Anthropic, ByteDance Seed, Qwen/DAMO, DeepSeek, Moonshot/Kimi, Stanford/Berkeley/CMU/MIT, Tsinghua/PKU/SJTU/Fudan/ZJU/RUC, KAIST.
3. **Fundamental > incremental.** Prefer papers introducing a new mechanism, primitive, loss, or architecture over 3rd-gen iterations (`-v2`, `++`, `Enhanced`, `Improved`) — UNLESS the series is in a current focus area where iteration matters:
   - RLVR/GRPO-family follow-ups (critical, the method is evolving)
   - Attention architecture refinements (KV cache, sinks, linear variants)
   - Widely-adopted methods getting practical improvements (EAGLE → EAGLE-2 → EAGLE-3, Self-Forcing → Self-Forcing++)
   - Tech reports for models you're tracking (DeepSeek, Qwen, Gemini, GLM, EXAONE, Kimi, MiniMax, LongCat)

## 4. Red flags — skip regardless of score

- **Benchmark clusters** from one pseudonymous ecosystem (e.g. the `*Claw*` family in April 2026: `SkillClaw` / `ClawBench` / `ClawGUI` / `Claw-Eval` — four adjacent papers, same fictional runtime, no name-recognizable authors). Usually landgrabs.
- **Suspicious author anonymization** mid-list (e.g. "Z. L." as 2nd author while the 1st author name contains the paper's gimmick, like "Hongyuan *Adam* Lu" on "Adam's Law").
- **Vanity author explosions** (30+ authors, no recognizable names, "DataFlow Team" / "HY Vision Team" / "Xpert Team" kitchen-sink preprints).
- **Single-author preprints** with no institutional email and no github.
- **"Survey of surveys"** or **meta-benchmarks** unless the senior author is a known taxonomist.

## 5. Quick query template

```python
# uv run python - <<'PY'
import json
from neon_db import NeonDB, TABLE
from psycopg.rows import dict_row
YYMM = "2604"; SCORE = 9; LIMIT = 30
db = NeonDB()
with db.get_conn() as c, c.cursor(row_factory=dict_row) as cur:
    cur.execute(f"""
        SELECT id, title, upvotes, org_fullname, organization, authors,
               primary_category, similar_paper
        FROM {TABLE}
        WHERE interested=0 AND id LIKE '{YYMM}.%' AND score={SCORE}
        ORDER BY upvotes DESC NULLS LAST LIMIT {LIMIT}
    """)
    for r in cur.fetchall():
        al = json.loads(r["authors"] or "[]")
        names = ", ".join(al[:5]) + (f" +{len(al)-5}" if len(al)>5 else "")
        org = r["org_fullname"] or r["organization"] or "-"
        print(f"[{r['upvotes']:>3}] {r['id']} | {r['primary_category']:8s} | {org[:22]:22s} | {r['title'][:80]}")
        print(f"     ~ {(r['similar_paper'] or '')[:90]} | auth({len(al)}): {names[:120]}")
PY
```

## 6. Why this beats reading everything

The score filter is a real token lever: for a ~360-paper month, running `multi_prompt.py` on everything is ~3.8M tokens of input + 900k output. Running the reranker first (cheap small model, ~500 tokens fresh per paper with cached examples block) + `multi_prompt` on the ~5 survivors is ~800k total — ~5× on tokens, ~15–25× on cost because the reranker uses Gemini 3 Flash while multi_prompt uses the local large model.

The whole point: don't delegate the authorship + fundamental-vs-incremental check to the scorer. The scorer is a harsh relevance filter; the taste layer lives in this playbook.

# Paper viewer UI (Svelte)

The browse-and-mark-interested frontend lives in **`svelte-ui/`** — a SvelteKit 2 + Svelte 5 (runes) + TypeScript app. It replaces the old monolithic HTML that `daily_papers/paper_viewer.py` used to `generate_html()` as one 58 MB blob.

**Architecture (two-process dev):**

```
svelte-ui (Vite :5173)  ──proxy──>  daily_papers/paper_server.py (FastAPI :8787)  ──>  Neon + all_scored.json
```

The FastAPI server is the JSON/data backend. The Svelte app is the UI. `svelte-ui/vite.config.ts` proxies `/api/papers`, `/interested`, `/interested-ids`, `/add-paper` to `$PAPER_SERVER_URL` (default `http://localhost:8787`). Override the target with `PAPER_SERVER_URL=... pnpm dev` when paper_server is on another host.

**Run:**

```bash
# Terminal 1 — backend
uv run python daily_papers/paper_server.py --port 8787

# Terminal 2 — frontend
cd svelte-ui && pnpm install && pnpm dev
# open http://localhost:5173
```

**Key files:**
- `svelte-ui/src/routes/+page.svelte` — the whole viewer (filters, date picker, cards, mark-interested POSTs). Single-file on purpose for now.
- `svelte-ui/src/routes/+layout.svelte` — Google Sans / Material Symbols fonts, page shell.
- `svelte-ui/src/lib/types.ts` — `Paper` type mirroring `all_scored.json` rows.
- `svelte-ui/vite.config.ts` — the dev-server proxy config.
- `daily_papers/paper_server.py` — FastAPI. Exposes `GET /api/papers` (scored feed) and the legacy interested/add-paper routes. Still serves the old HTML at `GET /` as a fallback; remove once the Svelte UI feels good.

**Typecheck:** `cd svelte-ui && pnpm check` (uses svelte-check).

**Do NOT re-run `npx sv create`.** This was forked from `../company-scraper/svelte-ui` (copied with rsync, boilerplate stripped). Edit the files in place.

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

Step 5 (`absorb_scored_json`) mirrors **every** column out of `all_scored.json` into Neon — arxiv metadata (`title`, `authors`, `github`, …) **and** scoring columns (`score`, `similar_paper`, `score_reason`, `tag_category_v2`, `tag_confidence`, `tag_reason`, `score_source`). After `sync_db.py` runs, the Neon row for a given `arxiv_id` mirrors the JSON — no separate scoring-import step.

### Batched writes (`NeonDB.batch`)

All hot write loops in `sync_db.py` use the shared-connection batch API:

```python
with db.batch() as b:
    for row in rows:
        b.save_paper(row["arxiv_id"], title=row["title"], score=row["score"])
```

`batch()` opens one psycopg connection, commits every 500 statements, commits the tail on clean exit, and rolls back on exception. Single-call `db.save_paper(...)` still works for ad-hoc writes but opens a fresh connection per call — in tight loops that trips Neon's SSL reaper after a few thousand rows. **Always prefer `batch()` when writing >100 rows.**

## Committed input files (tracked in git)

| Path | Size | Role |
|---|---|---|
| `daily_papers/papers_out/all_scored.json` | 40 M | LLM scoring pass output. 13,739 papers with `score` (1-10), `similar_paper`, `reason`, `tag_category`, `tag_confidence`, `tag_reason`. Produced by `daily_papers/hf_daily_papers.py` against the reading history. This is the source of truth for scores; `sync_db.py` step 5 mirrors it into Neon (incl. the scoring columns). |
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

