# What this repo really is

A personal arxiv-paper curation + summarization pipeline. The Python side handles ingest, scoring, tagging, and long-form summary generation; two separate frontends ride on top of the same Neon DB. Live counts: ~34k papers in Neon (grew from ~21k after the 2026-06 OpenAlex company backfill), ~5.7k marked `interested=1`, ~17k company-flagged, ~6.1k with a stored `summary`, ~9.5k with full-text `markdown`, ~13k with OpenAlex `cited_by_count`, multi-tag taxonomy via `tag_categories_v2 TEXT[]`. Some signals live in dedicated files, not the DB:

- **`src/paper_pipeline/discovery/`** — company-scrape pipelines (OpenAlex by-institution, S2 author-affiliation, playwright over AI-lab publications + HuggingFace model-card pages, arxiv affiliation search). Only **4 are repeatable "canon"**; the rest are completed one-shots — see **§ Company-scrape: canon vs legacy** below. Outputs land in Neon as new stub rows. (`probes/` holds exploratory one-offs that aren't pipeline steps.)
- **`src/paper_pipeline/substack/`** — a sibling side-project: two-pass DeepSeek-V4-Pro summaries of every Ryan Peterman ("The Peterman Post") podcast/essay. Outputs in `out/bulk/<slug>/summary.md` (160 articles). Synced to the nextjs-ui repo via `scripts/sync-peterman.mjs` over there.
- **`src/paper_pipeline/summarize/`** — the long-form 7-section paper-summary pipeline; `src/paper_pipeline/cli/multi_prompt.py` is the shim. `src/paper_pipeline/regen/offline_regen.py` is the cluster-deployed batch variant (per-section streaming JSONL, resume-safe). Sections 1–4 only when the user wants the abridged version (`--max-section 4 --skip-pitch --skip-cat --skip-assemble`).
- **`local_data/`** is gitignored — all backups + the legacy SQLite snapshot live there.
- **Three frontends, distinct purposes:**
  - **`svelte-ui/`** — local mark-interested triage tool (SvelteKit, talks to FastAPI `paper_server.py`).
  - **`paper-graph-ui/`** — public reader at https://paper-graph-ui.vercel.app (Next.js 16, build-time bake from `docs/**/*.md` + Neon snapshot, no runtime DB). The detail page surfaces the full metadata set (citations/fwci/doi/published/abstract/tags). A **hide-by-default toggle** (localStorage `pg.show-summaryless.v1`) reveals the ~12.7k company/interested papers that have metadata but no summary (`hasSummary:false` nodes, full abstract, "summary not generated yet") in the text/timeline/category views — they are EXCLUDED from the `/graph` SVG. Refresh data with `uv run python scripts/pull-neon-metadata.py && node scripts/build-paper-graph.mjs` then build. Do NOT bake the full-text `markdown` column into the JSON (size).
  - **`mobile/`** — Expo SDK 56 React Native app (EAS build pipeline, bundles ~1.4k paper bodies as assets). **Don't touch unless explicitly asked** — has its own `mobile/CLAUDE.md`, `mobile/BLOCKERS.md`, `mobile/EAS_BUILD_LINKS.md`.
- **`configs/vllm/`** — YAML launch configs for vllm. V4-Pro is checked in but V4-Flash is the default for the production scoring pipeline.
- **`prompt_iter/`** — sandbox for iterating section prompts; `iter.py` + `iter_pitch.py` plus per-paper generation diffs. Not the live pipeline.
- **`plans/`** (dir) and `plan.md` (root) — scratch planning docs, not load-bearing.
- **`arxiv.py/`** — vendored clone of `lukasschwab/arxiv.py` for arxiv-search experiments. Not on the import path of the live pipeline.
- **`scripts/data_bucket.py`** (tracked, PEP-723 Typer CLI) — syncs the repo's gitignored data (`docs/`, `local_data/`, …) to a Hugging Face Storage bucket (`hf://buckets/vincentzed-hf/data/cuda/`). Only `data-bucket.manifest.ndjson` (path/bytes/sha256/hf-URI per file, committed at repo root) is in git; `uv run scripts/data_bucket.py pull` restores everything on a fresh clone, so nothing big bloats git and nothing is lost to `.gitignore`. Needs `hf auth whoami`. This is the only tracked file under repo-root `scripts/` (the `pull-neon-metadata.py` / `build-paper-graph.mjs` / `sync-peterman.mjs` referenced elsewhere live under `paper-graph-ui/scripts/` and the nextjs-ui repo, NOT here).
- **Non-pipeline dirs** (gitignored, kept on purpose): `mintlify-ref/` (abandoned mintlify docs reference) + `scratch/` (throwaway). The pre-package stale leftovers (`multi_prompt_pkg/`, `daily_papers/`, `docs_backup/`, `docs_new/`, `throwaway_script/`, `PDF/`) were **deleted 2026-06-14**.
- **vllm-on-remote workflow** lives in `~/.claude/skills/jsonl-remote-job/` — the resume-safe JSONL+SCP+docker-exec pattern these scripts all use.

# `src/paper_pipeline/` package map

Installable package — `uv sync` editable-installs it. Run entrypoints via the console scripts declared in `pyproject.toml` `[project.scripts]`: `paper-summarize`, `paper-summarize-single`, `paper-sync`, `paper-server`, `paper-fetch`, `paper-tag`, `paper-import-jsonl`, `paper-e2e`. **No `sys.path` bootstraps anywhere** — imports are package-qualified (`from paper_pipeline.core.neon_db import NeonDB`). Data is co-located with its consuming code; `local_data/` (gitignored) and `docs/` stay at the repo root and are read CWD-relative, so **run console scripts from the repo root**.

- `core/` — `neon_db.py` (Neon layer; CLI via `python -m paper_pipeline.core.neon_db init-schema|migrate`), `external.py`, `import_jsonl.py` (`paper-import-jsonl`), and the co-located `external_papers.db`.
- `ingest/` — HF/arxiv ingest + scoring + tagging: `hf_daily_papers.py` (`paper-fetch`), `paper_server.py` (`paper-server`, the svelte-ui backend), `tag_papers.py` (`paper-tag`), `papers_by_score.py`, `examples.py` (reading-history loader), `fetch_only.py`. Co-located data: `papers_out/all_scored.json`, `tagged_papers.jsonl`, `papers_to_tag.jsonl`, `arxiv_index.txt`; `papers_out_run100/` (tracked) is an alternate/older scoring run (`all_scored.json` + `relevant.md`), not read by the live pipeline.
- `summarize/` — the long-form 7-section pipeline (was `multi_prompt_pkg/`): `cli.py` (`paper-summarize`), `summarizer.py`, `pipeline.py`, `prompts.py`, `llm.py`, `storage.py`, `config.py`, `schemas.py`, `pdf.py`. Golden-sample one-shot in `summarize/examples/`.
- `ocr/` — fill full-text `markdown` for the KEEP set (company/interested) via VLM OCR on the B300 box. `find_missing.py` (HEAD-scan `arxiv.org/html/{id}` → `out/{arxiv_scan.jsonl,ocr_needed.txt,has_html.txt,keep_ids.txt}`), a two-phase driver pattern (`fetch` PDFs decoupled from `ocr`), and `ingest_ocr.py` (upsert markdown + stamp `markdown_source`, `--ids-file` gates to a subset). **The live model is `glm_ocr_driver.py` (`zai-org/GLM-OCR` on vanilla vLLM `cu130-nightly`, direct full-page → "Text Recognition:" → markdown).** (Earlier model paths chandra-ocr-2 / PaddleOCR-VL were **deleted 2026-06-14**; the `fetch` phase is model-agnostic. `GOAL.md` still describes the original chandra design as history.) NOTE: keep multi-type excepts parenthesized — `except (A, B):`. Ruff is now pinned to `target-version = "py312"` (was py314, which applied PEP 758 and stripped the parens to the bare `except A, B:`, a `SyntaxError` on the box's Python 3.12).
- `cli/` — entrypoints: `main.py` (`paper-summarize-single`, + `main_prompt.txt`), `multi_prompt.py` (the summarize shim), `sync.py` (`paper-sync`, full Neon rebuild).
- `discovery/` — one-shot org/lab scrapers (land Neon stubs): the `playwright_*` sweep (`playwright_extra_companies.py` imports `playwright_pub_scrape.py` as a sibling), `s2_company_scrape.py` (S2 query normalized to `|`, not the literal `OR`; `--save-neon` default on), `openalex_2026_audit.py` (`--years` range, `--exclude` orgs e.g. universities, `--max-pages` per-org cap, `--save-neon` net-new only; captures the FULL OpenAlex Work payload — `cited_by_count`/`fwci`/abstract/topics/pdf_url + raw object in the JSONL — and uses a per-org short-lived DB batch so slow inter-org fetches don't trip Neon's idle reaper), `arxiv_org_search.py`, `exa_search_missing_papers.py`/`import_exa.py`, and `enrich_interested.py`. The 2026-06 run backfilled ~13.9k historical company papers (2000–2024, universities excluded) with citations; OpenAlex-pulled company papers are marked `is_only_important_because_of_company=true`. **Year coverage of the playwright "webpage-scrape" path (verified 2026-06-14 vs Neon):** it is NOT year-filtered — it grabs every arxiv id on each lab's publications / HF-org page; the last run was 2026-05-28/29, so 2025 is fully covered but **2026 stops at ~late May** (no June+ from this path — that edge only fills via the daily HF feed). Across the 6 `playwright_*.jsonl` outputs (unioned + deduped) there are 811 distinct 2025 + 254 distinct 2026 ids; Neon holds 797/811 (2025) and 205/254 (2026) — i.e. **~63 scraped ids never landed** (14×2025, 49×2026), either un-absorbed or extraction noise (reconcile with an import pass if it matters). **IDs are already canonicalized per-run** — every scraper's regex captures `(\d{4}\.\d{4,5})` with the `vN` suffix OUTSIDE the group (version-stripped) and accumulates into a `set` (0 versioned ids in any output) — but the 6 files are separate runs with NO cross-file dedup on disk; only Neon's PK upsert collapses cross-file dups, so on-disk per-file counts overcount distinct papers.
- `probes/` — exploratory, NOT pipeline steps: `probe_deepmind.py`, `probe_eleuther.py`, `probe_linkedin_2026.py`, `probe_all_orgs_2026.py`.
- `regen/` — summary backfill: `offline_regen.py` (cluster batch generator; `--max-section N --skip-pitch --skip-cat --skip-assemble` for the abridged sections-1-to-N variant; per-section streaming JSONL with resume-via-dedup; `--repo-root` default `.`), the `prepare_*_input.py` / `build_regen_input_from_hf.py` / `export_for_regen.py` builders, `absorb_regen_output.py`, `assemble_s14.py`, `add_missing_to_md.py` / `regen_to_md.py` / `regen_md_v2.py`, `generate_example.py`.
- `tagging/` — `multi_tag_via_vllm.py` + `tag_via_vllm.sh` (V4-Pro multi-tag → `tag_categories_v2 TEXT[]`); `V4_PRO_INFERENCE_RUNBOOK.md` lives here.
- `substack/` — independent side-project (no pipeline imports): two-pass Peterman summaries via `bulk_summarize.py` + `retry_failed.py`; `prompts.py` (iter-4 P1/P2 templates), `example_adrien.md` (style few-shot), `llm.py` (SSH-tunneled cluster vllm). Output: `out/bulk/<slug>/summary.md`.

# Company-scrape: canon vs legacy

**One-run E2E:** `uv run paper-e2e` (`cli/e2e.py`) orchestrates the whole company-paper pipeline in one command — **discover** (OpenAlex + S2 + playwright-web + arxiv-meta) → **absorb** (web/meta JSONL → Neon) → **enrich** (`enrich_from_arxiv`) → **OCR** (`--ocr`, off by default) → **summarize**. Each stage is a failure-isolated subprocess (OpenAlex/S2 self-`--save-neon`; web/meta absorbed in-process), S2 auto-skips with no `S2_API_KEY`, summarize auto-skips if the LLM server at `:30000` is down, and a Rich summary table prints rows-added/time per stage. `--dry-run` prints the plan. Re-run safe (net-new/upsert). It just spawns the canon scripts below — reach for those individually for a single source:

| Canon script | Source | Use for |
|---|---|---|
| `openalex_2026_audit.py` | OpenAlex API (by institution) | **default** company backfill; free, no key, date-filtered |
| `s2_company_scrape.py` | Semantic Scholar (author affiliation) | orgs OpenAlex misses (OpenAI/Anthropic/DeepSeek — no OpenAlex DOI links); needs `S2_API_KEY` |
| `playwright_pub_scrape.py` | browser: lab pubs pages + HF org pages | the web-scrape; many primary URLs are dead → leans on HF org pages |
| `arxiv_org_search.py` | arxiv affiliation search | Meta / LinkedIn fallback (their own sites yield 0 arxiv ids) |

Everything else in `discovery/` is a **completed one-shot, not re-run**: `playwright_{org_scrape,extra_companies,user_specified_orgs,crack_failed,hf_probe_sweep}.py`, `semantic_scholar_audit.py`, `exa_search_missing_papers.py`/`import_exa.py`, `s2_enrich_all.py`, `enrich_interested.py`. `probes/` is exploratory, never pipeline. (`firecrawl/` was the pre-playwright scraper, "3.5× worse than playwright" — **deleted 2026-06-14**.)

# Known pipeline holes / gotchas (as of 2026-06-14)

- **[FIXED 2026-06-14] `except A, B:` Python-3.12 `SyntaxError`.** The 2026-06-13 ruff pass (`target py314`) had stripped the parens off multi-type excepts in 13 runtime files (PEP 758 — valid on 3.14, `SyntaxError` on the 3.12 cluster/OCR/B300 box). Now re-parenthesized in all 13 **and** ruff pinned to `target-version = "py312"` so the rewrite can't recur. Don't bump the ruff target back to py314 while the deploy box is 3.12.
- **`playwright_pub_scrape` org URLs — grok-verified + patched 2026-06-14.** 5 orgs whose primary page was dead/missing got real research-page URLs (all with `follow_paper_links` to per-post pages): **Meta-FAIR** (`ai.meta.com/global_search/?content_types[0]=publication` — old `/research/publications/` now HTTP 500; new page is a JS/GraphQL app rendering `/research/publications/{slug}/` links, NOT direct arxiv ids — validate the follow-links yield), **Stability-AI** (`stability.ai/research`), **AI21** (`ai21.com/research/`), **Moonshot-Kimi** (`kimi.com/blog/`), **StepFun** (`chat.stepfun.com/research/`). 7 orgs confirmed to have **no central publications page** → correctly `mode="none"`, HF-org mining only: xAI, Character-AI, **DeepSeek**, 01-AI, Baichuan, LongCat-Meituan, Kakao Brain. For Meta / LinkedIn the robust path remains `arxiv_org_search.py` (their own sites don't expose arxiv ids directly).
- **Leading-edge company coverage gap.** Company-targeted scrapers last ran 2026-05-28/29 (playwright) and 2026-06-13/14 (OpenAlex). June-2026+ company papers enter ONLY via the daily HF feed (`paper-fetch`), which is not company-targeted — re-run OpenAlex / S2 / playwright to close the rolling gap.
- **[CLOSED 2026-06-14] the 63 un-absorbed webpage-scraped ids** — 62 backfilled into Neon (`score_source=playwright_backfill_20260614`), 1 dropped as an affiliation false-positive (`2501.00002`, "LinkedIn Tango" puzzle paper).

# TL;DR — where the papers live

- **Source of truth:** Neon Postgres, table `"nextjs-ui_paper"` (via `$DATABASE_URL`, see `src/paper_pipeline/core/neon_db.py`).
- **Legacy SQLite snapshot:** `local_data/papers.db` (~130 MB, **gitignored**, 13,448 rows). Read-only in the live pipeline; only `python -m paper_pipeline.core.neon_db migrate` reads from it.
- **Hand-seeded classical papers:** `src/paper_pipeline/core/external_papers.db` (~560 KB, tracked in git). Merged into Neon by `src/paper_pipeline/cli/sync.py`; read directly by `src/paper_pipeline/core/external.py`.
- Scored paper feed (JSON cache): `src/paper_pipeline/ingest/papers_out/all_scored.json` (~41 MB, tracked). Produced by `paper-fetch` (`src/paper_pipeline/ingest/hf_daily_papers.py`).

# Paper curation playbook (any time window)

How to pick papers to mark `interested=1` from a given month, week, or arbitrary date range. Reuse for "recs from YYYY-MM", "what did I miss in $WINDOW", etc.

## 1. Make sure the window is absorbed + enriched

`all_scored.json` has every paper's score/reason, but **not** always authors/abstract/published. Without authors you can't apply the authorship filter, so enrich first. Scope to the arxiv_id YYMM prefix(es) covering your window — e.g. April 2026 = `2604.*`.

```python
# uv run python - <<'PY'
import json
from pathlib import Path
from paper_pipeline.core.neon_db import NeonDB
from paper_pipeline.ingest.hf_daily_papers import fetch_arxiv_metadata

PREFIXES = ("2604.",)  # adjust for window
db = NeonDB()
data = json.loads(Path("src/paper_pipeline/ingest/papers_out/all_scored.json").read_text())
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
from paper_pipeline.core.neon_db import NeonDB, TABLE
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

The score filter is a real token lever: for a ~360-paper month, running `src/paper_pipeline/cli/multi_prompt.py` on everything is ~3.8M tokens of input + 900k output. Running the reranker first (cheap small model, ~500 tokens fresh per paper with cached examples block) + `multi_prompt` on the ~5 survivors is ~800k total — ~5× on tokens, ~15–25× on cost because the reranker uses Gemini 3 Flash while multi_prompt uses the local large model.

The whole point: don't delegate the authorship + fundamental-vs-incremental check to the scorer. The scorer is a harsh relevance filter; the taste layer lives in this playbook.

# Paper viewer UI (Svelte)

The browse-and-mark-interested frontend lives in **`svelte-ui/`** — a SvelteKit 2 + Svelte 5 (runes) + TypeScript app. It replaces an old monolithic HTML viewer (a removed 58 MB blob).

**Architecture (two-process dev):**

```
svelte-ui (Vite :5173)  ──proxy──>  paper-server :8787 (ingest/paper_server.py)  ──>  Neon + all_scored.json
```

The FastAPI server is the JSON/data backend. The Svelte app is the UI. `svelte-ui/vite.config.ts` proxies `/api/papers`, `/interested`, `/interested-ids`, `/add-paper` to `$PAPER_SERVER_URL` (default `http://localhost:8787`). Override the target with `PAPER_SERVER_URL=... pnpm dev` when paper_server is on another host.

**Run:**

```bash
# Terminal 1 — backend
uv run paper-server --port 8787

# Terminal 2 — frontend
cd svelte-ui && pnpm install && pnpm dev
# open http://localhost:5173
```

**Key files:**
- `svelte-ui/src/routes/+page.svelte` — the whole viewer (filters, date picker, cards, mark-interested POSTs). Single-file on purpose for now.
- `svelte-ui/src/routes/+layout.svelte` — Google Sans / Material Symbols fonts, page shell.
- `svelte-ui/src/lib/types.ts` — `Paper` type mirroring `all_scored.json` rows.
- `svelte-ui/vite.config.ts` — the dev-server proxy config.
- `src/paper_pipeline/ingest/paper_server.py` (run via `paper-server`) — FastAPI. Exposes `GET /api/papers` (scored feed) and the legacy interested/add-paper routes. Still serves the old HTML at `GET /` as a fallback; remove once the Svelte UI feels good.

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

Full long-form paper summaries are produced by **`src/paper_pipeline/cli/multi_prompt.py`**, NOT `src/paper_pipeline/cli/main.py`. `src/paper_pipeline/cli/multi_prompt.py` is a thin shim that re-exports the public API from the `src/paper_pipeline/summarize/` package and dispatches the CLI via `paper_pipeline.summarize.cli.main`. It fetches an arxiv PDF, extracts text with PyMuPDF, and sends it to a local LLM for the full 7-section teach-through (Executive Summary → Context → Technical Approach → Insights → Experiments → Limitations → Implications). Writes into the Neon Postgres `"nextjs-ui_paper"` table by default.

## Prereqs

1. Start a local OpenAI-compatible server at `http://localhost:30000/v1` (defined as `LOCAL_BASE_URL` in `src/paper_pipeline/summarize/config.py`). SGLang, vLLM, llama.cpp server — anything OpenAI-compatible. The script sends `model="default"`; the server serves whatever it loaded.
2. Have `DATABASE_URL` set to a Neon Postgres connection string. `src/paper_pipeline/core/neon_db.py` auto-loads `.env` from this repo and falls back to `../company-scraper/nextjs-ui/.env` (see `_load_env`).

## Single paper

```bash
uv run paper-summarize --url https://arxiv.org/abs/2312.07104
```

## Batch

```bash
uv run paper-summarize --urls "url1,url2,url3" --concurrency 5
```

## Dynamic mode (continuous worker)

Polls Neon every `--interval` seconds for stub rows missing a summary (`NeonDB.get_stubs_without_summary`) and processes them. Run it in a tmux pane and let it chew through the backlog:

```bash
uv run paper-summarize --dyn --concurrency 4096 --interval 30
```

`--concurrency` here is the semaphore ceiling for concurrent papers; the actual network cap is set by the single shared httpx client inside the script.

## Backfill

Re-summarize every `interested=1` paper currently missing a `summary`:

```bash
uv run paper-summarize --backfill --concurrency 8
```

## Remote runs → JSONL sidecar

When running on a remote box that can't reach Neon, write results to JSONL and absorb later:

```bash
uv run paper-summarize --dyn --jsonl ./runs/remote.jsonl
```

Then import the JSONL into Neon via `src/paper_pipeline/cli/sync.py`'s absorb helpers (`src/paper_pipeline/core/import_jsonl.py` also exists for one-off imports).

## Model overrides

- `--model <name>` — override the `"default"` model string sent in the chat completion request. Only meaningful if your local server dispatches by name.
- `--many-pass` — use the old 7-section **sequential** pipeline (one LLM call per section, with prior sections in context). Default is a single big call; `--many-pass` is for smaller models that choke on the full context.

## Where summaries land

- **Default:** `local_data/papers.db`, column `summary`, keyed by arxiv_id. `paper_server.py` and the docs build read from here.
- **`--jsonl`:** append-only JSONL file; re-imported into the DB later.
- **`docs/<category>/*.md`:** NOT written by `src/paper_pipeline/cli/multi_prompt.py`. Those are hand-curated / one-off exports. Don't expect `src/paper_pipeline/cli/multi_prompt.py` to populate them.

## Tuning for port-forwarded servers

If hitting the LLM through VSCode port-forward or SSH -L, high concurrency causes connection resets. Drop `--concurrency` first (try 4–8); if still flaky, the httpx client limits live inside `src/paper_pipeline/summarize/llm.py` — look for the `AsyncOpenAI(...)` construction and cap `max_connections` there. See `src/paper_pipeline/ingest/hf_daily_papers.py` for the analogous single-client pattern.

# Data paths

The source of truth is the Neon Postgres `"nextjs-ui_paper"` table (see `src/paper_pipeline/core/neon_db.py`). Several committed input files feed it; `local_data/papers.db` is a legacy SQLite snapshot kept around for migration/backup only.

## The Neon DB (source of truth)

- Table: `"nextjs-ui_paper"` in a Neon Postgres database. Connection comes from `DATABASE_URL`.
- Schema: defined in `src/paper_pipeline/core/neon_db.py` (`NeonDB.init_schema` + the `SCHEMA_COLUMNS` tuple). Includes scoring columns `score`, `similar_paper`, `score_reason`, `tag_category_v2`, `tag_confidence`, `tag_reason`, `score_source`; full-text `markdown` + `markdown_source` (provenance, e.g. `glm-ocr-pdf`); and OpenAlex impact `cited_by_count` (INT) + `fwci` (DOUBLE, field/age-normalized → recency-robust ranking).
- `DATABASE_URL` discovery: `neon_db._load_env` reads this repo's `.env` first, then walks up to `../company-scraper/nextjs-ui/.env` as a shared fallback. Export in the shell to override.
- CLI: `uv run python -m paper_pipeline.core.neon_db init-schema` (idempotent) and `uv run python -m paper_pipeline.core.neon_db migrate --sqlite local_data/papers.db` (copies the legacy SQLite snapshot into Neon; uses partial upsert so it's safe to re-run).

## Legacy SQLite snapshot

`local_data/papers.db` — ~130 MB, **gitignored**, NOT in any commit. This is a frozen copy of the pre-refactor state (13,448 rows, 964 `interested=1`, 12,664 scored, 13,023 tagged) and is the source that `neon_db.py migrate` reads from. Nothing in the live pipeline writes to it anymore. `local_data/` is gitignored so it can also hold the `*_backup.db` recovery snapshots.

## Rebuilding from the committed inputs

```bash
uv run python -m paper_pipeline.core.neon_db init-schema      # ensure the table exists
uv run paper-sync                  # full pipeline (uses arxiv API)
uv run paper-sync --skip-arxiv     # fast — use committed data only
```

Step 5 (`absorb_scored_json`) mirrors **every** column out of `all_scored.json` into Neon — arxiv metadata (`title`, `authors`, `github`, …) **and** scoring columns (`score`, `similar_paper`, `score_reason`, `tag_category_v2`, `tag_confidence`, `tag_reason`, `score_source`). After `src/paper_pipeline/cli/sync.py` runs, the Neon row for a given `arxiv_id` mirrors the JSON — no separate scoring-import step.

### Batched writes (`NeonDB.batch`)

All hot write loops in `src/paper_pipeline/cli/sync.py` use the shared-connection batch API:

```python
with db.batch() as b:
    for row in rows:
        b.save_paper(row["arxiv_id"], title=row["title"], score=row["score"])
```

`batch()` opens one psycopg connection, commits every 500 statements, commits the tail on clean exit, and rolls back on exception. Single-call `db.save_paper(...)` still works for ad-hoc writes but opens a fresh connection per call — in tight loops that trips Neon's SSL reaper after a few thousand rows. **Always prefer `batch()` when writing >100 rows.**

## Committed input files (tracked in git)

| Path | Size | Role |
|---|---|---|
| `src/paper_pipeline/ingest/papers_out/all_scored.json` | 40 M | LLM scoring pass output. 13,739 papers with `score` (1-10), `similar_paper`, `reason`, `tag_category`, `tag_confidence`, `tag_reason`. Produced by `paper-fetch` (`src/paper_pipeline/ingest/hf_daily_papers.py`) against the reading history. This is the source of truth for scores; `src/paper_pipeline/cli/sync.py` step 5 mirrors it into Neon (incl. the scoring columns). |
| `src/paper_pipeline/ingest/tagged_papers.jsonl` | 8.6 M | Tagger output. 13,023 rows: `{arxiv_id, title, category, confidence, reason}`. Consumed by `paper-sync --import-tags`. |
| `src/paper_pipeline/ingest/papers_to_tag.jsonl` | 19 M | Tagger input: 13,025 rows of `{arxiv_id, title, abstract}`. Produced by `paper-sync --export`, fed into `tag_papers.py`. |
| `results_modal3.jsonl` | small | Older run of LLM-generated long-form summaries (441 rows). Sidecar reference. |
| `src/paper_pipeline/core/external_papers.db` | 560 K | Hand-seeded classical papers (GANs, Deep Learning, etc.) with non-arxiv IDs. Still a SQLite file — read directly by `src/paper_pipeline/core/external.py` and merged into Neon by `src/paper_pipeline/cli/sync.py`. |
| `src/paper_pipeline/ingest/arxiv_index.txt` | small | Free-form list of arxiv IDs to ensure-exist in Neon as stubs. |
| `docs/**/*.md` | many | Per-paper summary markdown (organized by category dir). `src/paper_pipeline/cli/sync.py` parses these to backfill `summary` on stub rows; `src/paper_pipeline/ingest/examples.py:load_examples()` also uses filenames as a reading-history source. |
| `links.txt`, `src/paper_pipeline/ingest/arxiv_index.txt` | small | Reading-list text files. |

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

`src/paper_pipeline/ingest/examples.py:load_examples()` (shared by `hf_daily_papers.py` and `papers_by_score.py`) unions three sources:

1. Neon `"nextjs-ui_paper"` via `NeonDB.get_interested()` — the `interested = 1` rows
2. `src/paper_pipeline/core/external_papers.db` — `SELECT title, category FROM papers` (still SQLite; read by `src/paper_pipeline/core/external.py`)
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
                                    │  paper-sync  (all writes via NeonDB)     │
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

