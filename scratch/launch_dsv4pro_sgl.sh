#!/bin/bash
# Bring up DeepSeek-V4-Pro on the B300 box via SGLang (user-provided throughput config).
# Sleep-infinity container (skill pattern) + detached `sglang serve` exec (no trailing & — -d
# already detaches; a trailing & gets the server reaped). Image + serve flags are the user's.
set -euo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260607-5160f791}"
NAME="${NAME:-dsv4pro-sgl}"
PORT="${PORT:-30000}"
NVML=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.82.07

docker rm -f "$NAME" 2>/dev/null || true
docker run --name="$NAME" \
  --gpus all --shm-size=128g --hostname=sgl-b300-inference \
  --volume /root/.cache:/root/.cache \
  --volume "$NVML":/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  --env=NCCL_SOCKET_IFNAME=eth0 --cap-add=CAP_SYS_PTRACE --network=host --privileged \
  --workdir=/sgl-workspace/sglang --runtime=runc --detach=true \
  "$IMAGE" bash -lc 'sleep infinity'

# User-provided serve command (max-throughput megamoe config). Detached, no trailing &.
docker exec -d "$NAME" bash -lc "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320 \
  sglang serve \
  --trust-remote-code \
  --model-path deepseek-ai/DeepSeek-V4-Pro \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --moe-a2a-backend megamoe \
  --mem-fraction-static 0.835 \
  --cuda-graph-max-bs 544 \
  --swa-full-tokens-ratio 0.075 \
  --chunked-prefill-size 65536 \
  --tokenizer-worker-num 8 \
  --enable-prefill-delayer \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --host 0.0.0.0 \
  --port $PORT > /root/serve.log 2>&1"

echo "container $NAME up (IMAGE=$IMAGE PORT=$PORT); server -> /root/serve.log"
