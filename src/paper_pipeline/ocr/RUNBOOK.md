# OCR pipeline — fill in full-text markdown for papers HF can't render

Goal: every arxiv paper in Neon (`nextjs-ui_paper`) should have a full-text `markdown`
body. HF's `papers/{id}.md` endpoint gives us that body for free **when the paper has an
arXiv HTML twin**. The ones it can't serve are split into "scrape arXiv HTML" (cheap, no
GPU) vs "OCR the scanned PDF" (this pipeline). Built off NielsRogge's
[`arxiv-ocr`](https://github.com/NielsRogge/arxiv-ocr) but sourced from our DB and sunk
into our NDJSON + Neon, not HF datasets.

Model: [`datalab-to/chandra-ocr-2`](https://huggingface.co/datalab-to/chandra-ocr-2) —
tops the olmOCR **ArXiv** benchmark (90.2). License caveat: weights are modified
OpenRAIL-M (free for research/personal/startups < $2M; not for competing with Datalab's
API) — code is Apache-2.0. Confirm this is acceptable for the use case.

```
Stage 1    find_missing.py   local   probe HF .md (+arXiv HTML) -> bucket every paper
Stage 1.5  scrape_html.py    local   html_only bucket: arXiv HTML -> markdown (no GPU)
Stage 2    chandra_ocr.py    b100    OCR the ocr_needed PDFs with chandra-ocr-2 on vLLM
Stage 3    ingest_ocr.py     local   upsert markdown into nextjs-ui_paper.markdown
```

Two fill-in paths converge on one ingest: most missing papers have an arXiv HTML twin
(Stage 1.5, free, local); only the scanned/no-HTML minority need the GPU (Stage 2).

Artifacts live in `src/paper_pipeline/ocr/out/`:
`classification.jsonl` (every paper's bucket), `ocr_needed.txt` / `html_only.txt`
(id lists), `html_out.jsonl` (Stage-1.5 output), and `ocr_out.jsonl` (Stage-2 output).

---

## Stage 1 — classify (local, no GPU)

```bash
uv run python -m paper_pipeline.ocr.find_missing run        # full sweep (~30-60 min)
uv run python -m paper_pipeline.ocr.find_missing run --limit 100   # smoke test
uv run python -m paper_pipeline.ocr.find_missing report     # re-emit lists + table
```

Buckets (keyed on HTTP status — HF's 404 page is a fixed ~49.8k-char shell, so length is
not separable):

- `hf_ok` — `huggingface.co/papers/{id}.md` 200 → already have markdown, skip.
- `html_only` — HF non-200 but `arxiv.org/html/{id}` 200 → scrape HTML, **no GPU**.
- `ocr_needed` — both non-200 → **OCR the PDF (Stage 2)**.

Resume-safe (append-only NDJSON, ids already classified are skipped). Only the hf-missing
subset touches arXiv, throttled (`--arxiv-concurrency` / `--arxiv-min-interval`). Run a
single instance — concurrent runs corrupt the log. Output: `out/ocr_needed.txt`,
`out/html_only.txt`.

---

## Stage 1.5 — scrape arXiv HTML (local, no GPU)

For the `html_only` bucket: fetch `arxiv.org/html/{id}` and convert to markdown locally.
Math survives because LaTeXML keeps the source LaTeX in `<math alttext="...">`, which we
splice back as `$…$` / `$$…$$` before markdownify. No pandoc needed.

```bash
uv run python -m paper_pipeline.ocr.scrape_html run            # all html_only ids
uv run python -m paper_pipeline.ocr.scrape_html run --limit 50 # smoke test
```

Resume-safe; throttled arXiv access (`--concurrency` / `--min-interval`). Output:
`out/html_out.jsonl` (same shape Stage 3 ingests). Then ingest as in Stage 3.

---

## Stage 2 — OCR on b100 (8×B300)

`chandra_ocr.py` is a self-contained PEP-723 script: `uv run` builds its own env (chandra,
vllm, pypdfium2). It downloads each arXiv PDF (rate-limited 3.1s), renders pages, OCRs via
a vLLM endpoint, and appends one resume-safe record per paper to `ocr_out.jsonl`.

### 0. Ship the script + id list into the container

```bash
SSHB='ssh -o IdentityAgent=none -o IdentitiesOnly=yes -i ~/.ssh/b100_ed25519 b100'
CT=dsv4-sgl-nightly                      # the long-lived --network=host container

# copy via the host then into the container
scp -o IdentityAgent=none -o IdentitiesOnly=yes -i ~/.ssh/b100_ed25519 \
  src/paper_pipeline/ocr/chandra_ocr.py src/paper_pipeline/ocr/out/ocr_needed.txt b100:/tmp/
$SSHB "docker cp /tmp/chandra_ocr.py $CT:/work/ && docker cp /tmp/ocr_needed.txt $CT:/work/"
```

### Option A (recommended, reference-validated) — driver-managed dp8

Each process owns one GPU + a disjoint id shard + its own vLLM. This **is** dp8 across the
8 B300s (max throughput) and is exactly the path the reference validates. No router.

```bash
$SSHB "docker exec -d $CT bash -lc 'cd /work && for i in \$(seq 0 7); do \
  uv run chandra_ocr.py run --ids-file ocr_needed.txt --out ocr_out.jsonl \
    --gpu-id \$i --port \$((8000+i)) --num-shards 8 --shard-index \$i \
    > ocr.shard\$i.log 2>&1 & done; wait'"
```

### Option B (single shared endpoint, mirrors the Qwen SMG dp8 launch)

Serve chandra-ocr-2 **once** dp8, then point the driver at it with `--no-launch`. This is
the "like the qwen sglang command" shape. ⚠️ chandra-ocr-2 is a custom OCR VLM — the
reference serves it on **vLLM**; SGLang arch support is unverified, so smoke-test one GPU
before committing all 8.

```bash
# B1 — vLLM dp8 (validated path):
$SSHB "docker exec -d $CT bash -lc 'cd /work && \
  vllm serve datalab-to/chandra-ocr-2 --served-model-name chandra --data-parallel-size 8 \
    --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 18000 \
    --mm-processor-kwargs '"'"'{\"min_pixels\":3136,\"max_pixels\":6291456}'"'"' \
    --host 0.0.0.0 --port 30000 > vllm.log 2>&1'"

# B1-alt — SMG/sglang dp8, same structure as the Qwen3.6 launch (verify arch first):
#   smg serve --backend sglang --connection-mode http \
#     --model-path datalab-to/chandra-ocr-2 --data-parallel-size 8 \
#     --trust-remote-code --mem-fraction-static 0.85 --context-length 18000 \
#     --router-max-concurrent-requests 1024 --router-queue-size 2048 \
#     --host 0.0.0.0 --port 30000

# B2 — run the driver against the shared endpoint (no per-process launch):
$SSHB "docker exec -d $CT bash -lc 'cd /work && \
  uv run chandra_ocr.py run --ids-file ocr_needed.txt --out ocr_out.jsonl \
    --no-launch --vllm-base-url http://localhost:30000/v1 > ocr.log 2>&1'"
```

### Monitor + pull results

```bash
$SSHB "docker exec $CT bash -lc 'cd /work && wc -l ocr_out.jsonl; tail -3 ocr.*.log ocr.log 2>/dev/null'"
$SSHB "docker exec $CT cat /work/ocr_out.jsonl" > src/paper_pipeline/ocr/out/ocr_out.jsonl
```

### Teardown

```bash
# driver(s) exit on their own when the id list is exhausted; if you launched Option B's
# server, stop it (canonical script inside the container):
$SSHB "docker exec $CT bash -lc './scripts/killall_sglang.sh 2>/dev/null; pkill -f vllm; pkill -f chandra_ocr'"
```

---

## Stage 3 — ingest markdown into Neon (local)

`markdown` is already a live column on `nextjs-ui_paper`; `init_schema()` ensures it.

```bash
uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl src/paper_pipeline/ocr/out/ocr_out.jsonl --dry-run
uv run python -m paper_pipeline.ocr.ingest_ocr ingest --jsonl src/paper_pipeline/ocr/out/ocr_out.jsonl
```

Idempotent upsert; rows with `error` status or markdown < `--min-chars` (200) are skipped.
The reader is source-agnostic, so a future arXiv-HTML scrape of the `html_only` bucket can
feed the same `ingest`.
```
