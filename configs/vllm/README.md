# vLLM serve configs

Codified launch configs so we stop typing the long `vllm serve ...` line
by hand every time the container gets bounced.

| Config | Model | GPUs | Notes |
|---|---|---|---|
| `deepseek-v4-pro.yaml` | DeepSeek-V4-Pro | 8 (DP=8 + EP) | What I used for the full s1-s7 regen. ~600B total. |
| `deepseek-v4-flash.yaml` | DeepSeek-V4-Flash | 8 (DP=4 + EP across 8) | New default for follow-up regens. 284B total / 13B active, FP4+FP8, 1M ctx. |

## Launch (inside the brayden container)

```bash
docker exec -d vllm-brayden-dsv4-pro bash -lc \
  'cd /root/regen && vllm serve --config configs/vllm/deepseek-v4-flash.yaml > /tmp/vllm_serve.log 2>&1'
```

Then poll `/v1/models` until it binds (typically 4-10 min for V4-Pro,
should be faster for V4-Flash given the smaller working set).

## Switching back to V4-Pro

Same launch line, swap the config path. Both YAMLs use the same
`max-model-len: 393216`, `kv-cache-dtype: fp8`, `block-size: 256`, and
`moe-backend: deep_gemm_mega_moe`, so the regen pipeline (which sends
`model="default"` or `model=<served-name>`) works against either
unchanged.

## First-time setup on a new node

DeepGEMM FP8 kernels need to be installed once:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh)
```
