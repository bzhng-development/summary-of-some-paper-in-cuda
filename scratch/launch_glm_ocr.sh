#!/bin/bash
# Serve zai-org/GLM-OCR on the B300 via vLLM NIGHTLY on CUDA 13.0 (sm_103a).
# GLM-OCR's glm_ocr arch is new -> needs a nightly vLLM; cu130 covers the B300.
# Sleep-infinity container + detached `vllm serve` exec so the container persists.
# Override: GPUS=4 PORT=8080 ./launch_glm_ocr.sh
set -euo pipefail

GPUS="${GPUS:-4}"
DP="${DP:-1}"             # data-parallel replicas (one per visible GPU)
PORT="${PORT:-8080}"
IMAGE="${IMAGE:-vllm/vllm-openai:cu130-nightly}"
MODEL="${MODEL:-zai-org/GLM-OCR}"
SERVED="${SERVED:-GLM-OCR}"
MAXSEQS="${MAXSEQS:-512}"
MTP="${MTP:-0}"          # MTP speculative decoding: triggered a CUDA device-side assert under dp8 -> default OFF
NAME="${NAME:-glm-ocr-vllm}"
NVML=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.82.07   # REQUIRED: else "Failed to infer device type"

DP_ARG=""
[ "$DP" -gt 1 ] && DP_ARG="--data-parallel-size $DP"
SPEC_ARG=""
[ "$MTP" = "1" ] && SPEC_ARG="--speculative-config.method mtp --speculative-config.num_speculative_tokens 1"

docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" -e HF_TOKEN="${HF_TOKEN:-}" \
  --shm-size=32g \
  -v /root/.cache:/root/.cache \
  -v "$NVML":/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -v "$NVML":/usr/lib/x86_64-linux-gnu/libnvidia-ml.so:ro \
  --network=host --entrypoint /bin/bash "$IMAGE" -c 'sleep infinity'

# Per the GLM-OCR model card + vLLM docs: serve with built-in MTP speculative decoding.
docker exec -d "$NAME" bash -lc "vllm serve $MODEL \
  --served-model-name $SERVED --allowed-local-media-path / $DP_ARG $SPEC_ARG \
  --max-num-seqs $MAXSEQS --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port $PORT > /root/vllm_serve.log 2>&1"

echo "container $NAME up (GPUS=$GPUS PORT=$PORT MODEL=$MODEL MAXSEQS=$MAXSEQS); server -> /root/vllm_serve.log"
