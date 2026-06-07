#!/bin/bash
# Tag the 726 (or whatever's left) untagged papers in regen_input_FULL.jsonl
# via vllm V4-Pro instead of OpenRouter/Gemini.
#
# IMPORTANT: this script does NOT start vllm. It assumes vllm V4-Pro is already
# serving on localhost:8000 inside the brayden container (or whatever endpoint
# is passed via --base-url). To bring up vllm, see:
#   configs/vllm/deepseek-v4-pro.yaml  (parameters reference)
#   Earlier session log for the docker exec invocation pattern.
#
# Two execution modes:
#
#   LOCAL — runs everything from this laptop, hitting a port-forwarded vllm.
#   First open the tunnel in another terminal:
#     ssh -L 8000:localhost:8000 brayden@95.133.253.79 \
#       -- docker exec vllm-brayden-dsv4-pro tail -f /dev/null
#   Then:
#     bash throwaway_script/tagging/tag_via_vllm.sh local
#
#   REMOTE — ships the input + script to the brayden container and runs there.
#   No port forwarding needed.
#     bash throwaway_script/tagging/tag_via_vllm.sh remote
#
# Both modes:
#   1. Build the to-tag JSONL from regen_input_FULL.jsonl, filtering out any
#      arxiv_id already present in tagged_may.jsonl + tagged_new.jsonl
#      (resume-safe — re-runs only target the remainder).
#   2. Call tag_papers.py with OPENROUTER_API_KEY UNSET so the resolver falls
#      through to --base-url + --model. Default model: V4-Pro.
#   3. Append results to local_data/tagged_via_vllm.jsonl.
set -u
MODE="${1:-local}"
REPO="/Users/vincentzed/Documents/Github/open_source/refs/summary-of-some-paper-in-cuda"
cd "$REPO"

REMOTE="brayden@95.133.253.79"
CONTAINER="vllm-brayden-dsv4-pro"
MODEL="deepseek-ai/DeepSeek-V4-Pro"

INPUT_JSONL="local_data/regen_input_FULL.jsonl"
TAGGED_PRIOR=("local_data/tagged_may.jsonl" "local_data/tagged_new.jsonl" "local_data/tagged_via_vllm.jsonl")
TO_TAG="local_data/papers_to_tag_via_vllm.jsonl"
OUTPUT="local_data/tagged_via_vllm.jsonl"

# Step 1: build the to-tag list (skip anything already tagged anywhere).
echo "[1/2] building to-tag list..."
uv run python - <<PY
import json
from pathlib import Path

tagged: set[str] = set()
for p in ${TAGGED_PRIOR[@]@Q}:
    f = Path(p)
    if not f.is_file():
        continue
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            tagged.add(json.loads(line)["arxiv_id"])
        except Exception:
            pass
print(f"  already-tagged across {len(${TAGGED_PRIOR[@]@Q})} files: {len(tagged)}")

written = 0
bad = 0
with open("$INPUT_JSONL", encoding="utf-8") as fin, open("$TO_TAG", "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if r["arxiv_id"] in tagged:
            continue
        if not r.get("abstract"):
            continue  # no abstract = can't tag
        fout.write(json.dumps({
            "arxiv_id": r["arxiv_id"],
            "title": r["title"],
            "abstract": r["abstract"],
        }, ensure_ascii=False) + "\n")
        written += 1
print(f"  wrote {written} papers to $TO_TAG (skipped {bad} bad json lines)")
PY

TOTAL=$(wc -l < "$TO_TAG" | tr -d '[:space:]')
if [ "$TOTAL" = "0" ]; then
    echo "[done] nothing to tag — all papers already in one of: ${TAGGED_PRIOR[*]}"
    exit 0
fi

# Step 2: run tag_papers.py against V4-Pro.
echo "[2/2] tagging $TOTAL papers via V4-Pro..."

case "$MODE" in
  local)
    # Assumes vllm is reachable at http://localhost:8000/v1 (port-forwarded).
    # UNSET OPENROUTER + MODAL so tag_papers.py falls through to --base-url.
    OPENROUTER_API_KEY="" MODAL_API_KEY="" uv run python daily_papers/tag_papers.py \
        -i "$TO_TAG" \
        -o "$OUTPUT" \
        --base-url "http://localhost:8000/v1" \
        --model "$MODEL" \
        --concurrency 22
    ;;
  remote)
    # Ship the input into the container then run there. Output is fetched back.
    REMOTE_INPUT="/tmp/papers_to_tag_via_vllm.jsonl"
    REMOTE_OUT="/tmp/tagged_via_vllm.jsonl"
    SSH_OPTS="-o RemoteCommand=none -T -o ConnectTimeout=15"

    echo "  shipping $TO_TAG to remote $REMOTE..."
    scp $SSH_OPTS "$TO_TAG" "$REMOTE:$REMOTE_INPUT"
    ssh $SSH_OPTS "$REMOTE" "docker cp $REMOTE_INPUT $CONTAINER:$REMOTE_INPUT && \
        docker exec -e OPENROUTER_API_KEY= -e MODAL_API_KEY= $CONTAINER bash -lc \
        'cd /root/regen && python3 daily_papers/tag_papers.py -i $REMOTE_INPUT -o $REMOTE_OUT \
            --base-url http://localhost:8000/v1 --model $MODEL --concurrency 22'"
    echo "  pulling result back..."
    ssh $SSH_OPTS "$REMOTE" "docker cp $CONTAINER:$REMOTE_OUT $REMOTE_OUT"
    scp $SSH_OPTS "$REMOTE:$REMOTE_OUT" "$OUTPUT"
    ;;
  *)
    echo "unknown mode: $MODE  (use 'local' or 'remote')"
    exit 2
    ;;
esac

NEW=$(wc -l < "$OUTPUT" | tr -d '[:space:]')
echo "[done] $OUTPUT now has $NEW tagged rows"
