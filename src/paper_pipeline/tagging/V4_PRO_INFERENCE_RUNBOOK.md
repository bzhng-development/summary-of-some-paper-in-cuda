# V4-Pro inference runbook

Everything you need when GPUs free up. Three jobs queued; pick what to run.

## Step 0 — bring up vllm V4-Pro

```bash
ssh -o RemoteCommand=none -T brayden@95.133.253.79 \
  'docker exec vllm-brayden-dsv4-pro bash -lc "rm -f /tmp/vllm_serve.log"; \
   docker exec -d vllm-brayden-dsv4-pro bash -lc \
   "vllm serve deepseek-ai/DeepSeek-V4-Pro \
      --api-server-count 8 --data-parallel-size 8 --enable-expert-parallel \
      --tokenizer-mode deepseek_v4 --reasoning-parser deepseek_v4 \
      --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
      --trust-remote-code --max-model-len 393216 \
      --block-size 256 --kv-cache-dtype fp8 --moe-backend deep_gemm_mega_moe \
      > /tmp/vllm_serve.log 2>&1"'
```

Config reference: `configs/vllm/deepseek-v4-pro.yaml`. Boot takes ~3-5 min (weights cached). Wait for `/v1/models` to return.

## Job A — V4-Pro tagging of remaining ~726 April papers (replaces OpenRouter/Gemini step)

```bash
# After vllm is up, on the remote (no port-forwarding needed):
bash throwaway_script/tagging/tag_via_vllm.sh remote
```

The script:
- Builds `local_data/papers_to_tag_via_vllm.jsonl` by filtering `regen_input_FULL.jsonl` against any already-tagged sources (tagged_may.jsonl + tagged_new.jsonl + tagged_via_vllm.jsonl). Resume-safe.
- Ships it to the container, runs `daily_papers/tag_papers.py` with `OPENROUTER_API_KEY` UNSET so the resolver falls through to `--base-url http://localhost:8000/v1 --model deepseek-ai/DeepSeek-V4-Pro`.
- Fetches `local_data/tagged_via_vllm.jsonl` back.

Alternative: `bash throwaway_script/tagging/tag_via_vllm.sh local` if you've set up an SSH port-forward (`ssh -L 8000:localhost:8000 …`).

ETA: V4-Pro tagging is ~0.5s/paper structured-output, so ~726 / 22 concurrency = ~30s of LLM time, plus network. **Total ~5-10 min on V4-Pro vs ~15 min on OpenRouter Gemini.** No external API cost.

## Job B — V4-Pro summarization on 650 company-flagged papers

Dataset already prepped: `local_data/regen_input_company.jsonl` (650 rows, schema matches `offline_regen.py`).

```bash
# Ship it to remote, kick off, return immediately.
scp local_data/regen_input_company.jsonl brayden@95.133.253.79:/tmp/
ssh -o RemoteCommand=none -T brayden@95.133.253.79 \
  'docker cp /tmp/regen_input_company.jsonl vllm-brayden-dsv4-pro:/root/regen/ && \
   docker exec -d vllm-brayden-dsv4-pro bash -lc \
   "cd /root/regen && python3 -u throaway_script/offline_regen.py \
     --input regen_input_company.jsonl --output regen_output_company.jsonl \
     --mode online --base-url http://localhost:8000/v1 \
     --model deepseek-ai/DeepSeek-V4-Pro --thinking none \
     > /root/regen/regen_company.log 2>&1"'
```

ETA: ~6-8 hours (650 × 9 phases at the prior-run rate). Streaming checkpoints mean any bounce only costs the ~22 in-flight requests.

## Job C — V4-Pro summarization on the curated subset (already done in prior session)

Already done in T9 — `local_data/regen_curated_20260526_021935/regen_output_curated.jsonl`. The 29 strict picks landed in paper-graph-ui via the T11+T12 commit. No re-run needed unless prompts change.

## Order-of-operations tip

Tagging (Job A) finishes fast and lets us populate categories cleanly before summarization. Run A then kick off B in parallel — summarization saturates vllm; tagging is too short to be worth pipelining around.

## Killing the V4-Pro server when done

```bash
ssh brayden@95.133.253.79 \
  'docker exec vllm-brayden-dsv4-pro bash -lc "pkill -9 -f \"vllm serve\""'
```

(Avoid `pkill -f vllm` — it'll match the bash subshell containing "vllm" and SIGKILL the SSH session. The narrower pattern is safer.)
