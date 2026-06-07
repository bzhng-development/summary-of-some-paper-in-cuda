# Task: OCR the 6,979 arXiv papers that have no HTML twin → markdown → Neon

## TL;DR for the agent

You are a peer engineer landing fresh on the **HF Daily Papers** pipeline (repo:
`~/cuda`, the `paper_pipeline` package — **NOT** `~/binutils`). The job: take the
~6,979 arXiv papers our corpus tracks that have **no machine-readable full text**, OCR
their PDFs into markdown with `datalab-to/chandra-ocr-2` on the B300 box, and write the
markdown back into Neon. This mirrors what Hugging Face themselves did ("How we OCR'ed
30,000 papers" — they Chandra-OCR the ~27k HF-indexed papers that lack an arXiv HTML
page).

**The job in one breath:** finish the two-phase driver → smoke-test it saturates the
GPU → run a ~500 batch and measure throughput → run the full 6,979 → ingest to Neon →
tear the GPU box down.

**The load-bearing design constraint (do not violate):**
1. **Decouple network from GPU.** Download *all* PDFs FIRST (phase `fetch`, no GPU,
   parallel, arXiv-rate-limit-bound). Only then OCR (phase `ocr`). The GPU must never
   stall waiting on arXiv. These are two separate commands writing/reading a shared
   `pdfs/` dir.
2. **The OCR phase must be genuinely concurrent.** Use an **async OpenAI client + a
   global `asyncio.Semaphore`** sized to ~256 in-flight *pages* (not papers) so the
   vLLM server runs hundreds of requests at once. The server holds **~350 concurrent**
   (see Background). A one-paper-at-a-time blocking loop (≈28 reqs) wastes >90% of the
   GPU — that is the failure mode we are explicitly fixing.

**What we do NOT trust / dead ends already ruled out:**
- **The HF `.md` probe is a dead end for *finding* the OCR set.** `huggingface.co/papers/{id}.md`
  429-throttles hard even authenticated, and it only renders FROM arXiv HTML anyway. The
  authoritative OCR criterion is purely `arxiv.org/html/{id}` returning **non-200** (HEAD,
  fast, unthrottled). That scan is already done — see `out/ocr_needed.txt`. Do not re-probe HF.
- **`py_compile` gives false passes** on `except A, B:`-style syntax errors. Always
  syntax-check with `ast.parse` (command in References). Two such bugs were already fixed
  (mirrored from a real one still in `ingest`-adjacent `tag_papers.py:291`).
- **Some `ocr_needed` ids are bad data** (e.g. `1009.6382`, `1105.8907` → 404/406 from
  arXiv). The driver must skip them cleanly and record `status=error`, not crash.

**Out of scope:** do not re-run Stage 1, do not touch GPUs 0–3 (another user's job), do
not build the `html_only` arXiv-HTML-scrape path (separate future task), do not refactor
`neon_db.py` beyond the `markdown` column already added.

## Background you need (no prior context assumed)

**The corpus.** Neon Postgres table `"nextjs-ui_paper"` (~21,715 rows). 21,623 are
arXiv-pattern ids; 92 are `ext:` non-arxiv (excluded — no PDF). A new `markdown TEXT`
column was added (idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS` in
`neon_db.py:init_schema`; also in `SCHEMA_COLUMNS`).

**The three stages** (`~/cuda/src/paper_pipeline/ocr/`):
- `find_missing.py` — **DONE.** Command `arxiv-scan` HEAD-probed all 21,623 ids →
  `out/arxiv_scan.jsonl` and `out/ocr_needed.txt`. Result: **6,979 need OCR (32.3%)**;
  14,644 have HTML (skip). ids sorted ascending → oldest (most likely truly scanned) first.
- `chandra_ocr.py` — **IN PROGRESS (two-phase rewrite).** Self-contained PEP-723 script
  (deps: chandra-ocr, pypdfium2, openai, httpx, typer, rich, pydantic). Two Typer commands:
  - `fetch` — download every arXiv PDF (async httpx, semaphore, retry, resume by file
    presence). Logs hard-404s to `<pdf-dir>/_missing_ids.txt`.
  - `ocr` — async pipeline: render PDF pages with pypdfium2 (in a ThreadPoolExecutor,
    since pdfium is blocking), submit each page to vLLM via `AsyncOpenAI` under a global
    page-semaphore, parse each page, reassemble per paper, append one `OcrRecord` per
    paper to `ocr_out.jsonl`. Resume = skip arxiv_ids already in the output.
- `ingest_ocr.py` — **DONE & verified.** `add-column` (ensures schema) and `ingest
  --jsonl ocr_out.jsonl` (upsert `markdown` by arxiv_id, `--min-chars` filter, `--dry-run`).

**Reuse chandra's primitives** (don't re-derive the prompt/format):
`from chandra.prompts import PROMPT_MAPPING` (use `ocr_layout`),
`from chandra.model.util import scale_to_fit, detect_repeat_token`,
`from chandra.output import parse_markdown`, `from chandra.input import flatten`.
Per-page request recipe: render → `scale_to_fit` → PNG → base64 → content =
`[{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}, {"type":"text","text":PROMPT}]`,
`max_tokens=12384, temperature=0.0, top_p=0.1`. On `detect_repeat_token` or error, retry
with rising temp (→0.8) / top_p (→0.95). This matches chandra's own vLLM client.

**The GPU box (`b100`).** 8×B300 (~275 GB each), driver **580.82.07**. SHARED:
- **GPUs 0–3 are another user's SGLang TP4 job — DO NOT TOUCH.** Only **GPUs 4–7** are free.
- Port **30000 is taken**; use **8000**.
- Host `/root` is **not writable** by the ssh user → `scp` to `/tmp`, then `docker cp`.
- **vLLM serving:** container `chandra-ocr-vllm`, image `vllm/vllm-openai:v0.20.1-ubuntu2404`,
  launched by `scratch/launch_chandra_vllm.sh` (sleep-infinity container + detached
  `vllm serve` exec; params `GPUS`/`DP`/`PORT`). Currently **single replica on GPU 4,
  :8000**, validated (clean markdown: Qwen-VLA 135k chars, white-dwarf paper 70k chars).
  - **GPU mount gotcha:** the script bind-mounts host `libnvidia-ml.so.580.82.07` over
    both `…so.1` and `…so` or vLLM dies with *"Failed to infer device type."*
  - **Server capacity:** KV cache ~214 GiB → ~6.3M tokens → vLLM logs *"Maximum
    concurrency for 18,000 tokens per request: 349.95x"* → **~350 concurrent** on ONE
    B300. This is why the driver must push hundreds of in-flight requests.
- **The driver runs in the `dsv4-sgl-nightly` container** (also `--network=host`, so it
  reaches `localhost:8000`), after `pip install chandra-ocr pypdfium2 openai typer rich
  pydantic` there. (Keeps the serving container clean.)

**Harness reaping:** the local background-bash harness SIGTERMs tasks at ~5 min (hard,
not idle-based — a heartbeat does NOT save it). Run anything long with `nohup … &
disown` to fully detach. `setsid` is unavailable on macOS; macOS `pgrep` lacks `-c` (use
`pgrep -f … | wc -l`).

## Already completed (this session)

**Stage 1 — find / classify (DONE):**
- [x] Cloned reference `NielsRogge/arxiv-ocr` → `~/cuda/scratch/arxiv-ocr`; read its README + `chandra2-arxiv-ocr.py`.
- [x] Confirmed our OCR criterion == HF's: a paper needs OCR iff `arxiv.org/html/{id}` is non-200.
- [x] Investigated Neon schema — found **no** full-text/markdown column existed (only LLM summaries in `paper-graph-ui/.../*.md`).
- [x] Built `find_missing.py` with `run` / `report` / `arxiv-scan` commands (HF auth, `_Transient` handling, throttle, heartbeat).
- [x] Calibrated the signal: detection keys on **HTTP status**, not body length (HF 404 page is a fixed ~49.8k-char shell).
- [x] Ran `arxiv-scan` over all **21,623** arXiv-pattern ids → **6,979 ocr_needed (32.3%)**, 14,644 has_html. Wrote `out/ocr_needed.txt` + `out/arxiv_scan.jsonl`.

**Stage 2 — driver (IN PROGRESS):**
- [x] Wrote `chandra_ocr.py` v1 (blocking, chandra `InferenceManager`) and validated it (2/2 papers, clean markdown).
- [x] Diagnosed the throughput bottleneck: v1 sent ~28 reqs (1 paper at a time) vs server's ~350 capacity → <1% KV use.
- [x] Rewrote as two-phase (`fetch` + `ocr`) with async OpenAI client + global page-semaphore + threadpool render.
- [ ] **REMAINING:** fix the `_fetch_async` `Progress`-in-`async with` bug (Step 1), then Steps 2–7.

**Stage 3 — ingest (DONE):**
- [x] Added `markdown` to `neon_db.py` `SCHEMA_COLUMNS` + CREATE DDL + idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS`.
- [x] Built `ingest_ocr.py` (`add-column`, `ingest`); ran `add-column` → **`markdown` column live on Neon (verified present)**; dry-run ingest works.

**Box / vLLM bring-up (DONE):**
- [x] Mapped the shared box: GPUs 0–3 occupied (another user), **GPUs 4–7 free**, :30000 taken → use :8000, `/root` not writable.
- [x] Downloaded `datalab-to/chandra-ocr-2` (10.6 GB, public, `Qwen3_5ForConditionalGeneration`) to the host HF cache.
- [x] Wrote `scratch/launch_chandra_vllm.sh` (sleep-infinity + detached `vllm serve` exec; `GPUS`/`DP`/`PORT` params; nvml mount).
- [x] Brought up vLLM `chandra-ocr-2` on **GPU 4, :8000** via `vllm/vllm-openai:v0.20.1` + `--trust-remote-code` (loads qwen3_5 on Blackwell); end-to-end validated (Qwen-VLA 135k chars, white-dwarf 70k chars).
- [x] Installed driver deps in `dsv4-sgl-nightly` (`chandra-ocr pypdfium2 openai typer rich pydantic`).

**Bugs found & fixed:**
- [x] HF `.md` anon→429 / authed→200 (added `HF_TOKEN`; treat 429/5xx as `_Transient`, re-probe, never misclassify).
- [x] Pivoted off the throttled HF probe entirely → fast unthrottled arXiv-HEAD scan.
- [x] `except A, B:` unparenthesized syntax bug (×2) — parenthesized; learned `py_compile` false-passes → use `ast.parse`.
- [x] vLLM "Failed to infer device type" → bind-mount host `libnvidia-ml.so.580.82.07` over `.so.1` and `.so`.
- [x] Background tasks SIGTERM'd at ~5 min → `nohup … & disown` (setsid unavailable on macOS).
- [x] `tar -C scratch` relative-path failure → absolute `-C "$PWD/scratch"`; `grep -v` exit-1 breaking `&&` chains.

## Plan (do these in order)

**Step 0 — Environment & access.**
- Repo `~/cuda` (uv, py3.14). SSH to box (inline flags — zsh won't split a spaced var):
  `ssh -o RemoteCommand=none -o IdentityAgent=none -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -T -i "$HOME/.ssh/b100_ed25519" b100 "<cmd>"`
- Confirm GPUs 4–7 free (`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`)
  and vLLM up (`curl -s -m5 localhost:8000/v1/models | grep -q chandra`). If a model
  download 401/403s, STOP and report (chandra-ocr-2 is public, so it shouldn't).
- **Deliverable:** one line confirming GPUs 4–7 free + `chandra` served on :8000.

**Step 1 — Finish & syntax-validate the two-phase driver.**
- Current bug: `_fetch_async` puts a **sync** `Progress(...)` inside `async with` →
  *"'Progress' object does not support the asynchronous context manager protocol."*
  Restructure so only the `AsyncClient` is in `async with`; nest `with Progress(...)`
  (and the `_missing_ids.txt` file) inside. (The `ocr` command's `with (sink, Progress)`
  is already fine — both sync.)
- **Deliverable:** `uv run python -c "import ast; ast.parse(open('src/paper_pipeline/ocr/chandra_ocr.py').read()); print('PARSE OK')"` prints `PARSE OK`; `uvx ruff check`/`format` clean.

**Step 2 — Smoke the `fetch` phase (40 ids).** Ship driver + `scratch/ocr_smoke40.txt`
(tail-40 of `ocr_needed.txt` = recent real papers) into `dsv4-sgl-nightly:/root/ocrjob2`,
run `fetch --ids-file ocr_smoke40.txt --pdf-dir pdfs --concurrency 16`.
- **Deliverable:** N PDFs present in `pdfs/` (≈40 minus any logged in `_missing_ids.txt`),
  and the fetch summary (ok/exists/missing/error counts).

**Step 3 — Smoke the `ocr` phase + PROVE saturation.** Run `ocr --pdf-dir pdfs --out
smoke40_out.jsonl --vllm-base-url http://localhost:8000/v1 --concurrency 128
--paper-workers 16`. While it runs, watch `docker logs chandra-ocr-vllm` for
`Running: N reqs`.
- **Settling test / guiding question:** does `N` climb into the **dozens-to-hundreds**?
  If it stays ≈6–10, the driver is still effectively serial — fix the concurrency before
  proceeding (this is the whole point). 
- **Deliverable:** `smoke40_out.jsonl` with clean markdown (spot-check 2 records: real
  title, >5k chars, `status=success`), AND a quoted vLLM log line showing high `Running:`.

**Step 4 — Batch of ~500, measure throughput.** `head -500 ocr_needed.txt` is oldest ids
(more 404s — good stress test of skip-handling). `fetch` then `ocr` (single replica still
fine). Record wall-clock and pages/sec.
- **Deliverable:** pages/sec on 1 replica + an extrapolated ETA for 6,979 on dp4
  (≈4× replicas). State whether dp4 is worth it or 1 replica suffices.

**Step 5 — Full run (6,979).** If dp4 chosen: tear down the single replica and relaunch
`GPUS=4,5,6,7 DP=4 … launch_chandra_vllm.sh` (wait for ready). `fetch` ALL ids (detached,
`nohup … & disown`), then `ocr` ALL (detached, resume-safe). Poll line counts.
- **Deliverable:** `ocr_out.jsonl` covering all *valid* ids; final ok/partial/error tally;
  `_missing_ids.txt` (the bad-data 404s) noted.

**Step 6 — Ingest to Neon.** Pull `ocr_out.jsonl` back to `~/cuda`. `uv run python -m
paper_pipeline.ocr.ingest_ocr ingest --jsonl ocr_out.jsonl` (run `--dry-run` first).
- **Deliverable:** count of `markdown` upserts; SQL spot-check that `markdown` is non-null
  for ≥3 sampled OCR'd ids.

**Step 7 — Tear down (shared box).** `docker kill chandra-ocr-vllm`; confirm GPUs 4–7
back to ~0 MiB.
- **Deliverable:** `nvidia-smi` showing 4–7 free again.

## Definition of done
- `ocr_out.jsonl` has one record per *valid* `ocr_needed` id; `status` in
  {success, partial_success, error}; success records carry real markdown (title + body).
- Step 3 produced **evidence the GPU was saturated** (vLLM `Running:` in the dozens+), not
  a serial trickle. This gates the whole run.
- `nextjs-ui_paper.markdown` is populated for the OCR'd ids (spot-checked in SQL).
- vLLM container killed; GPUs 4–7 free.

## Deliverables
- Fixed `chandra_ocr.py` (two-phase, concurrent, resume-safe), ruff-clean, `ast.parse` OK.
- `out/ocr_needed.txt` (6,979 — already produced) and `ocr_out.jsonl` (the OCR output).
- Throughput number + ETA from Step 4; final run tally from Step 5.
- Markdown ingested to Neon; teardown confirmation.

## Constraints / notes
- **Never touch GPUs 0–3** or other users' containers. Tear down your own when done.
- Scratch/artifacts under cwd (or the box's `/root/ocrjob2`), never bare `/tmp` on the mac.
- Long ops: `nohup … & disown` (harness reaps at ~5 min). Syntax-check with `ast.parse`.
- `/root` on the box isn't writable by ssh user → `scp` to `/tmp` then `docker cp`.
- chandra-ocr-2 license: OpenRAIL, HF confirmed free for commercial use — OK.

## Useful commands (cheatsheet)

All `ssh`/`scp` use inline flags (zsh won't word-split a spaced var). `b100` = the B300 box.

**SSH / SCP** (host `/root` not writable → stage in `/tmp`, then `docker cp`):
```bash
SSHFLAGS='-o RemoteCommand=none -o IdentityAgent=none -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -T -i '"$HOME"'/.ssh/b100_ed25519'
ssh $SSHFLAGS b100 "docker ps --format '{{.Names}}\t{{.Status}}'"           # NOTE: $SSHFLAGS has no spaces-in-value issue (flags only); the host arg is separate
scp -o RemoteCommand=none -o IdentityAgent=none -o IdentitiesOnly=yes -o BatchMode=yes -i "$HOME/.ssh/b100_ed25519" <local> b100:/tmp/<file>
```

**Local — syntax / lint (cwd `~/cuda`):**
```bash
uv run python -c "import ast; ast.parse(open('src/paper_pipeline/ocr/chandra_ocr.py').read()); print('PARSE OK')"   # py_compile lies; use this
uvx ruff check --fix src/paper_pipeline/ocr/chandra_ocr.py && uvx ruff format src/paper_pipeline/ocr/chandra_ocr.py
```

**Stage 1 (already done; re-run only to refresh):**
```bash
uv run python -m paper_pipeline.ocr.find_missing arxiv-scan --concurrency 32   # authoritative ocr_needed.txt
uv run python -m paper_pipeline.ocr.find_missing report                        # re-emit counts from the log
# tally a scan/sweep log:
uv run python -c "import json,collections; c=collections.Counter(json.loads(l)['html_status']!=200 for l in open('src/paper_pipeline/ocr/out/arxiv_scan.jsonl') if l.strip()); print(c)"
wc -l src/paper_pipeline/ocr/out/ocr_needed.txt   # 6979
```

**Box state / GPUs / vLLM health:**
```bash
ssh $SSHFLAGS b100 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"   # 4-7 must stay free
ssh $SSHFLAGS b100 "curl -s -m5 localhost:8000/v1/models | grep -q chandra && echo UP || echo DOWN"
ssh $SSHFLAGS b100 "docker logs --tail 20 chandra-ocr-vllm 2>&1 | grep -iE 'Running|Waiting|reqs'"  # saturation signal
```

**Launch / scale vLLM** (script already on box at `/tmp/launch_chandra_vllm.sh`; re-scp if changed):
```bash
# single replica (GPU 4):
ssh $SSHFLAGS b100 "HF_TOKEN=\$HF_TOKEN GPUS=4 DP=1 PORT=8000 bash /tmp/launch_chandra_vllm.sh"
# dp4 for the full run (GPUs 4-7):
ssh $SSHFLAGS b100 "HF_TOKEN=\$HF_TOKEN GPUS=4,5,6,7 DP=4 PORT=8000 bash /tmp/launch_chandra_vllm.sh"
# wait until ready (bail if it dies):
ssh $SSHFLAGS b100 'for i in $(seq 1 40); do curl -s -m5 localhost:8000/v1/models | grep -q chandra && { echo READY; break; }; docker ps --format "{{.Names}}" | grep -q chandra-ocr-vllm || { echo DIED; docker logs --tail 40 chandra-ocr-vllm; break; }; sleep 10; done'
```

**Ship the driver + ids into the driver container (`dsv4-sgl-nightly`):**
```bash
cd ~/cuda && COPYFILE_DISABLE=1 tar czf /tmp/ocr2.tgz -C src/paper_pipeline/ocr chandra_ocr.py -C "$PWD/scratch" ocr_smoke40.txt
scp -o RemoteCommand=none -o IdentityAgent=none -o IdentitiesOnly=yes -o BatchMode=yes -i "$HOME/.ssh/b100_ed25519" /tmp/ocr2.tgz b100:/tmp/
ssh $SSHFLAGS b100 "docker exec dsv4-sgl-nightly bash -lc 'mkdir -p /root/ocrjob2' && docker cp /tmp/ocr2.tgz dsv4-sgl-nightly:/tmp/ && docker exec dsv4-sgl-nightly bash -lc 'cd /root/ocrjob2 && tar xzf /tmp/ocr2.tgz'"
ssh $SSHFLAGS b100 "docker exec dsv4-sgl-nightly bash -lc 'pip install -q chandra-ocr pypdfium2 openai typer rich pydantic'"   # one-time
```

**Run the two phases (in `dsv4-sgl-nightly`; single-command Typer → NO subcommand if only one cmd, else use `fetch`/`ocr`):**
```bash
# Phase 1: download ALL pdfs (no GPU). Detached for big runs:
ssh $SSHFLAGS b100 "docker exec -d dsv4-sgl-nightly bash -lc 'cd /root/ocrjob2 && python3 -u chandra_ocr.py fetch --ids-file ocr_needed.txt --pdf-dir pdfs --concurrency 16 > fetch.log 2>&1'"
# Phase 2: OCR (saturate the server). Detached:
ssh $SSHFLAGS b100 "docker exec -d dsv4-sgl-nightly bash -lc 'cd /root/ocrjob2 && python3 -u chandra_ocr.py ocr --pdf-dir pdfs --out ocr_out.jsonl --vllm-base-url http://localhost:8000/v1 --concurrency 128 --paper-workers 16 > ocr.log 2>&1'"
# poll progress (results line count is truth):
ssh $SSHFLAGS b100 "docker exec dsv4-sgl-nightly bash -lc 'wc -l < /root/ocrjob2/ocr_out.jsonl; tail -2 /root/ocrjob2/ocr.log'"
# pull results back:
ssh $SSHFLAGS b100 "docker exec dsv4-sgl-nightly cat /root/ocrjob2/ocr_out.jsonl" > ~/cuda/scratch/ocr_out.jsonl
```

**Stage 3 — ingest to Neon (local):**
```bash
cd ~/cuda && uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl scratch/ocr_out.jsonl --dry-run   # preview
uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl scratch/ocr_out.jsonl
# SQL spot-check:
uv run python -c "from paper_pipeline.core.neon_db import NeonDB,TABLE; db=NeonDB();
import contextlib
with db.get_conn() as c, c.cursor() as cur:
    cur.execute(f\"SELECT count(*) FROM {TABLE} WHERE markdown IS NOT NULL\"); print('markdown rows:', cur.fetchone()[0])"
```

**Teardown (free shared GPUs):**
```bash
ssh $SSHFLAGS b100 "docker kill chandra-ocr-vllm; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -4"
```

**Detach pattern (local long ops dodge the ~5-min harness reaper):**
```bash
nohup uv run python -m paper_pipeline.ocr.find_missing arxiv-scan > scratch/scan.log 2>&1 < /dev/null & disown
pgrep -f find_missing | wc -l   # macOS pgrep has no -c
```

## References
- Reference impl (cloned): `~/cuda/scratch/arxiv-ocr/chandra2-arxiv-ocr.py` (NielsRogge);
  its blog "How we OCR'ed 30,000 papers" = the criterion source.
- Model card: `huggingface.co/datalab-to/chandra-ocr-2`; chandra repo `datalab-to/chandra`.
- Launch script: `~/cuda/scratch/launch_chandra_vllm.sh`.
- Stage files: `~/cuda/src/paper_pipeline/ocr/{find_missing,chandra_ocr,ingest_ocr}.py`.
- Neon access: `paper_pipeline.core.neon_db.NeonDB` (DATABASE_URL from env / `~/nextjs-ui/.env`).
- Syntax check: `cd ~/cuda && uv run python -c "import ast; ast.parse(open('src/paper_pipeline/ocr/chandra_ocr.py').read()); print('PARSE OK')"`
