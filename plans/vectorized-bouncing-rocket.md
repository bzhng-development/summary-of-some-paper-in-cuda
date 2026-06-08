# Plan: restructure `cuda/` into an installable `paper_pipeline` package

## Context

The repo root is a flat jumble: core importable modules (`neon_db.py`, `external.py`,
`sync_db.py`, `multi_prompt.py`, `main.py`, `import_jsonl.py`), two ad-hoc packages
(`daily_papers/`, `multi_prompt_pkg/`), the just-reorganized `throwaway_script/`, ~20 loose
scratch/data files, and several frontends — all relying on "run from repo root so `.` is on
`sys.path`" plus per-file `sys.path.insert` bootstraps. The goal: a single editable-installed
`src/paper_pipeline/` package grouping scripts by purpose, data co-located with its consuming
script, every import package-qualified, **all `sys.path` bootstraps deleted**, and commands run
via `console_scripts`. Decided with the user: package name `paper_pipeline`; editable install;
console_scripts; in scope = loose root files + backend Python dirs + committed data (co-located)
+ frontends' backend-facing refs only; `mobile/` is OFF-LIMITS.

## Target tree

```
cuda/
  pyproject.toml                      # + [build-system] (hatchling), [project.scripts], src layout
  src/paper_pipeline/
    __init__.py
    core/      neon_db.py  external.py  import_jsonl.py   external_papers.db (co-located w/ external.py)
    ingest/    (was daily_papers/) hf_daily_papers.py examples.py tag_papers.py papers_by_score.py
               fetch_only.py  paper_server.py   papers_out/all_scored.json   tagged_papers.jsonl
               papers_to_tag.jsonl  arxiv_index.txt        # data co-located with ingest scripts
    summarize/ (was multi_prompt_pkg/) cli.py config.py llm.py pdf.py pipeline.py prompts.py
               schemas.py storage.py summarizer.py
    discovery/ (was throwaway_script/discovery/*) + firecrawl/
    probes/    regen/   tagging/      (was throwaway_script/*)
    substack/  (was throwaway_script/substack/ — self-contained side-project)
    cli/       main.py  multi_prompt.py  sync.py (was sync_db.py)   main_prompt.txt
  scratch/     plan.md DATA.md mobile.md README extras, links.txt, *.bak, results_modal3.jsonl
  scripts/data_bucket.py + data-bucket.manifest.ndjson   # infra tooling (stays top-level)
  docs/  configs/  prompt_iter/  arxiv.py/               # unchanged
  svelte-ui/  paper-graph-ui/  mobile/                   # apps: only fix backend-facing refs
  local_data/ (gitignored, 3.8 GB) — STAYS AT ROOT (shared data tier, referenced as local_data/… everywhere)
```

**Note (flag to user at exit):** co-locating shared data feeds *inside* `src/` is non-idiomatic
(src is for code). `all_scored.json` is 40 MB and referenced by the CLAUDE.md curation playbook,
`paper-graph-ui` build, and several scrapers. Per the "data next to its script" decision it lands
in `ingest/papers_out/`; the alternative (a top-level `data/` tier) is cleaner for code/data
separation. Will confirm which at exit.

## Mechanics

### 1. Make it an installable package
- Add to `pyproject.toml`: `[build-system]` (hatchling), `[tool.hatch.build.targets.wheel] packages = ["src/paper_pipeline"]`, and `[project.scripts]` console entries (see §5).
- Update tool configs: `[tool.pyrefly] project_includes`/`search_path`, `[tool.ty.src]`, `[tool.pyright]` — point at `src/` instead of `.`; keep excludes for `arxiv.py`, `prompt_iter`, `.venv`. `probes/`/scratch may stay excluded.
- `uv sync` installs `paper_pipeline` editable into `.venv`.

### 2. Move files (all via `git mv` to preserve history)
- `neon_db.py external.py import_jsonl.py` → `src/paper_pipeline/core/`
- `daily_papers/*` → `src/paper_pipeline/ingest/` ; `multi_prompt_pkg/*` → `src/paper_pipeline/summarize/`
- `throwaway_script/{discovery,probes,regen,tagging,substack}` → `src/paper_pipeline/...`
- `main.py`, `multi_prompt.py`, `sync_db.py`→`cli/sync.py` → `src/paper_pipeline/cli/`
- Add `__init__.py` to every new package dir.

### 3. Rewrite imports (package-qualified) and DELETE all bootstraps
Mechanical map applied repo-wide (the ~40 sites from the import audit):
- `from neon_db import X` → `from paper_pipeline.core.neon_db import X`
- `from external import X` → `from paper_pipeline.core.external import X`
- `from daily_papers.<m> import X` → `from paper_pipeline.ingest.<m> import X`
- `from multi_prompt_pkg.<m> import X` → `from paper_pipeline.summarize.<m> import X`
- `from multi_prompt import X` → `from paper_pipeline.cli.multi_prompt import X` (or move shared API into `summarize/__init__.py`)
- **Delete every `sys.path.insert(...)` + `_REPO_ROOT = …` bootstrap** (incl. the depth-independent one just added to `throwaway_script/*`, and the ones in `daily_papers/hf_daily_papers.py`, `papers_by_score.py`, `fetch_only.py`). Editable install makes them unnecessary. Drop now-unused `# noqa: E402`.
- The sibling import `from playwright_pub_scrape import …` → `from paper_pipeline.discovery.playwright_pub_scrape import …`.
- **`paper-graph-ui/scripts/pull-neon-metadata.py`** (frontend, in scope for ref-fix): `from neon_db import` → `from paper_pipeline.core.neon_db import` (relies on the editable install being present in its run env — verify its invocation).

### 4. Co-locate data + fix path refs
- `external_papers.db` → `core/` (next to `external.py`); `sync.py`'s `EXTERNAL_DB_PATH` becomes package-relative (`Path(__file__).resolve()...` or `importlib.resources`).
- `tagged_papers.jsonl`, `papers_to_tag.jsonl`, `arxiv_index.txt` → `ingest/`; `all_scored.json` stays in its `papers_out/` (now under `ingest/`). Update: `sync.py` (`REPO_ROOT / "…"` consts), `tag_papers.py`, `paper_server.py` (`--json` default + arxiv_index append), `papers_by_score.py`, and the scrapers under `regen/`/`discovery/` that default to `daily_papers/papers_out/all_scored.json`.
- `main_prompt.txt` → `cli/` next to `main.py`.
- **`local_data/…` CWD-relative defaults are NOT rewritten** — `local_data/` stays at root; console scripts still run from repo root so these resolve.
- Update the CLAUDE.md curation playbook code blocks that reference `daily_papers/papers_out/all_scored.json`.

### 5. console_scripts (run via `uv run <cmd>`)
`paper-summarize` → `paper_pipeline.cli.multi_prompt:main`; `paper-summarize-single` → `…cli.main:main`;
`paper-sync` → `…cli.sync:main`; `paper-server` → `…ingest.paper_server:main`;
`paper-fetch` → `…ingest.hf_daily_papers:main`; `paper-tag` → `…ingest.tag_papers:main`;
`paper-import-jsonl` → `…core.import_jsonl:main`. Rewrite the runbook commands in CLAUDE.md, `daily_papers/HOWTO.md`, and the `V4_PRO_INFERENCE_RUNBOOK.md`.

### 6. Frontends / docs / infra (refs only)
- `svelte-ui/vite.config.ts`: proxy target is a URL (`PAPER_SERVER_URL`/`localhost:8787`) — unaffected; just confirm the `paper-server` command still binds 8787.
- `paper-graph-ui`: build reads `docs/**/*.md` (path unchanged) + runs `pull-neon-metadata.py` (import fixed in §3) — verify its run command/env.
- `mkdocs.yml`: confirm `docs_dir`/`nav` paths still valid (docs/ unmoved).
- `scripts/data_bucket.py` + `data-bucket.manifest.ndjson`: stay top-level; manifest indexes `local_data/` which is unmoved — unaffected. Re-verify after.

## Risks / call-outs
- **Blast radius:** ~40 import rewrites + ~30 moves + data refs + configs. Do it as one mechanical pass (a migration script), `git mv` for history, then verify hard.
- **Editable-install dependency:** every entry now requires `uv sync` to have installed the package. Document in README/CLAUDE.md.
- **`paper-graph-ui` + cron/remote jobs** that invoke these scripts by old path/module will break until updated — grep `~/.claude/skills/jsonl-remote-job` and any CI for old invocations is out-of-repo; flag to user.
- **Data-inside-src** awkwardness (see tree note) — confirm vs top-level `data/`.
- `mobile/` untouched.

## Verification
1. `uv sync` → package installs editable, no errors.
2. `uv run python -c "import paper_pipeline, paper_pipeline.core.neon_db, paper_pipeline.ingest.hf_daily_papers, paper_pipeline.summarize.cli"` → all import.
3. `python -m py_compile` (or `compileall`) over `src/` → clean; `uvx ruff check src` clean (E402 gone).
4. Each console script `--help` exits 0 (`paper-summarize`, `paper-sync`, `paper-server`, `paper-fetch`, `paper-tag`).
5. `paper-server --json src/paper_pipeline/ingest/papers_out/all_scored.json` boots; `svelte-ui` dev proxy reaches it.
6. `git status` shows renames (history preserved); no stray top-level core modules remain.
7. Spot-run one discovery scraper `--no-save-neon --help` and `paper-sync --skip-arxiv` dry path.
