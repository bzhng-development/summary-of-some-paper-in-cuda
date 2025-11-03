# PUNICA: MULTI-TENANT LORA SERVING

**ArXiv:** [2310.18547](https://arxiv.org/abs/2310.18547)

## 🎯 Pitch

Punica introduces a novel system and CUDA kernel (SGMV) for serving multiple Low-Rank Adaptation (LoRA) models on a shared GPU cluster using only a single copy of the underlying large language model. By enabling cross-LoRA batching and consolidating multi-tenant workloads with its specialized scheduler, Punica achieves up to 12× higher throughput than state-of-the-art systems with minimal latency overhead. This is transformative for cloud providers and platforms needing to efficiently and scalably serve thousands of custom fine-tuned models without massive GPU over-provisioning.

---

## 1. Executive Summary
Punica is a system for serving many Low‑Rank Adaptation (`LoRA`) models on a shared GPU pool while keeping only a single copy of the large base model in memory. It introduces a CUDA kernel called `Segmented Gather Matrix‑Vector Multiplication (SGMV)` that lets the GPU batch and execute requests targeting different LoRA adapters together, and a scheduler that consolidates multi‑tenant workloads across GPUs, yielding large throughput gains with tiny latency overhead.

## 2. Context and Motivation
- The specific problem
  - Many organizations fine‑tune the same large language model (LLM) for different tasks using `LoRA` (Section 2.2). Each LoRA adapter adds only 0.1–1% parameters but naïvely serving every adapter as a separate model wastes GPU memory and prevents batching across tenants.
  - Text generation has two stages (Section 2.1): `prefill` (process the prompt, build the key–value cache) and iterative `decode` (generate one token at a time). The decode stage dominates latency and GPU under‑utilization because each step processes a single token per request.
- Why this matters
  - Cloud providers and internal platforms must serve thousands of LoRA variants. If they keep separate copies or cannot batch across them, they need k×n GPUs (k per model, n models), which is wasteful in memory and compute.
  - Batching is the dominant lever for decode‑time efficiency (Figure 1 shows decode latency barely increases when batch size grows from 1 to 32—11→13 ms for short contexts), but standard systems only batch when all requests target the exact same model.
- Prior approaches and gaps
  - General LLM serving systems (e.g., HuggingFace Transformers, DeepSpeed, FasterTransformer, vLLM) either do not support LoRA at serving time or can only batch requests for the same adapter. They also suffer from key–value cache (`KvCache`) layouts that couple requests in a batch, wasting compute when some finish earlier (Figure 6).
  - Systems such as Orca and vLLM improve batching and memory management for a single model, but do not enable cross‑LoRA batching.
- Positioning
  - Punica targets multi‑tenant LoRA serving with three guiding principles (Section 1): consolidate workloads on few GPUs (G1), enable batching across different LoRA adapters (G2), and focus on decode performance since it dominates serving cost (G3).

## 3. Technical Approach
Punica has two pillars: a kernel that makes cross‑LoRA batching fast and a scheduler that keeps GPUs highly utilized with low memory overhead.

- High‑level architecture (Figure 2; Section 3)
  - Frontends receive requests containing a `LoRA model ID` and a prompt and pass them to a central scheduler.
  - Each GPU runs a `runner` process that hosts the backbone LLM in GPU memory and loads LoRA weights on demand; most GPU memory is reserved for `KvCache`.
  - Tokens are streamed from GPU runners back to users as they are generated.

- What is LoRA and how inference changes (Section 2.2)
  - LoRA assumes the fine‑tuned weight `W’` differs from the base weight `W` by a low‑rank product `AB`:
    - `W’ = W + A B`, where `A ∈ R^{h1×r}`, `B ∈ R^{r×h2}`, and `r` (the rank) is small (e.g., 16 vs. hidden sizes thousands).
  - During inference, a linear projection becomes: base GEMM with `W` plus a small “adapter” computation with `A` and `B`.

- SGMV kernel: batching across different LoRA adapters (Section 4; Figure 3–4)
  - Core idea in plain terms: group batch items by LoRA ID, then perform all LoRA “add‑ons” for the whole batch with a single fused, high‑throughput kernel—without duplicating base compute.
  - Semantics (Figure 3): treat the batch as segments `s[i]:s[i+1]`, each segment corresponding to one LoRA ID. For each segment, accumulate `Y += X @ W_i` only over that segment’s rows.
  - Two‑phase decomposition matching LoRA math:
    - First apply `A` (shrinks to rank `r`): `v += x A` — called `SGMV‑shrink`.
    - Then apply `B` (expands back to hidden size): `y += v B` — called `SGMV‑expand`.
  - GPU scheduling (Figure 4):
    - Bind the `LoRA index` (adapter ID) to CUDA `blockIdx.y` so each adapter’s subproblem runs in parallel.
    - `Expand` phase: split the output feature dimension and assign each chunk to a thread block; concatenate results.
    - `Shrink` phase: output is very thin (rank r), so use a `Split‑K` strategy—split the input dimension, compute partial sums in parallel, then perform a cross‑block reduction with grid synchronization.
    - Use Tensor Cores for GEMM‑like parts when operational intensity is high.
    - Edge case optimization: when every item in the batch has a distinct LoRA, the math degenerates to many matrix–vector ops (IO‑bound). Punica uses a special schedule focused on maximizing memory bandwidth and avoids Tensor Cores (Section 4).
  - Why this approach: It reuses the single backbone already on GPU and turns many small LoRA computations into one big, hardware‑efficient kernel launch; it maintains high arithmetic intensity whenever there is adapter sharing (Uniform/Skewed/Identical distributions in Figure 7–9).

- Scheduling and execution strategy (Section 5)
  - Mix prefill and decode in one invocation (Section 5): to give dense projections and the LoRA add‑on larger effective batch sizes. Prefill (single request at a time) and batched decode run separate specialized attention kernels; all other layers are batched across both.
  - New request placement (Section 5.1)
    - Keep per‑GPU `working sets` and available `KvCache` memory.
    - Place a new request on the GPU with the largest current batch (to sustain high throughput), subject to two constraints: max batch size not exceeded and sufficient `KvCache` memory. Ties go to the highest GPU UUID. Overflow requests queue FCFS.
    - Cap batch size at 32 on A100 as the best latency/throughput trade‑off (profiled; Section 5.1).
  - On‑demand LoRA loading (Section 5.2)
    - Only copy adapter matrices `A` and `B` to GPU when first needed.
    - The copy is asynchronous and overlaps with compute; measured cost: ~50 µs per layer and ~2 ms for all layers over PCIe Gen4 x16 (Section 5.2).
    - Since each decode step takes ~30 ms (Figure 1), the copy completes before the next batch needs the adapter in practice.
  - Request migration (Section 5.3; Figure 5)
    - When `KvCache` memory pressure grows, migrate requests by cancel‑and‑readd: stop streaming and free the source cache, then “re‑prefill” on the destination GPU using the original prompt plus already generated tokens to rebuild the cache.
    - Rationale: recomputing cache is as fast or faster than moving it in most cases (in line with PagedAttention results).
  - KvCache layout for continuous batching (Section 5.4; Figure 6)
    - Problem with common layouts: batching dimension is nested, so requests in a batch cannot be separated; when short requests finish early, others force extra decode steps (“waste” in Figure 6).
    - Punica uses a paged, separable layout with batching as the outer dimension:
      - Shape: `[sum_i ceil(S_i/P), L, 2, N, P, D]` with page size `P` (Section 5.4).
      - This enables continuous batching (add/remove requests freely) and reduces fragmentation (inspired by vLLM’s PagedAttention).

- Implementation (Section 6)
  - CUDA kernels are exposed via a PyTorch extension (`pybind11`).
  - Uses `FlashInfer` for fast attention and supports paged `KvCache`. LayerNorm is fused, reducing its latency from 110 µs to 4 µs (Section 6).
  - Batch construction details:
    - Concatenate prefill tokens first, then decode tokens; record boundaries in a `BatchLen` struct.
    - Construct `SGMV` segment indices once per invocation (reused across layers) to eliminate per‑layer overhead.

## 4. Key Insights and Innovations
- Cross‑LoRA batching via `SGMV` is the central innovation (Section 4)
  - What’s new: a kernel that computes the LoRA “add‑on” for many adapters and requests at once, grouping by adapter while still running a single base model copy.
  - Why it matters: batching was previously only effective when all requests shared the same model, leaving multi‑tenant scenarios under‑utilized (Figure 1 and Section 2.1). SGMV lifts this restriction, unlocking the main efficiency lever for decode.
- Decoding‑focused consolidation strategy (Sections 3 and 5)
  - What’s new: a scheduler that deliberately routes more work to already busy GPUs (up to a capped batch size) and migrates requests to keep batches large, knowing that decode latency grows only slightly with batch size (Figure 1).
  - Why it matters: this raises cluster‑level utilization and makes scale‑up/scale‑down decisions straightforward.
- Paged, separable `KvCache` layout that makes batching continuous (Section 5.4; Figure 6)
  - What’s new: a layout where the batch dimension is outermost and cache pages decouple request lifecycles, avoiding wasted decode steps.
  - Why it matters: it avoids the “short request holds back the batch” behavior seen in common frameworks, preserving the gains from large batches.
- Practical on‑demand LoRA loading and recomputation‑based migration (Sections 5.2–5.3)
  - What’s new: treating LoRA weight copies as asynchronous micro‑events (~2 ms model‑wide) and cache migration as recomputation (not data movement).
  - Why it matters: keeps GPU memory mostly for the cache and removes adapter‑switching as a bottleneck, enabling multi‑tenancy without preloading all adapters.

## 5. Experimental Analysis
- Setup (Section 7)
  - Models: Llama‑2 7B, 13B, 70B; LoRA rank default 16, applied to all dense projections.
  - Hardware:
    - Testbed #1: single NVIDIA A100 80 GB (for single‑GPU and microbenchmarks).
    - Testbed #2: two HGX A100 40 GB servers with 8 GPUs each and NVSwitch (for tensor parallel and cluster tests).
  - Workloads:
    - Prompt/response lengths drawn from ShareGPT distribution.
    - LoRA popularity distributions (Section 7): `Distinct` (every request different), `Uniform` (≈sqrt(n) adapters), `Skewed` (Zipf‑1.5), `Identical` (single adapter).
  - Baselines (Section 7): HuggingFace Transformers + PEFT, DeepSpeed + PEFT, FasterTransformer (backbone only), vLLM (backbone only). Model‑switching overheads are omitted for baselines (favorable to them).

- Microbenchmarks: SGMV performance
  - Roofline analysis (Figure 7): shows achieved FLOP/s vs arithmetic intensity for the SGMV kernel on A100.
    - In `Identical`, SGMV is memory‑bandwidth bound (line climbs along the 1.935 TB/s slope), as expected when the operation becomes one large BMM.
    - In `Distinct`, arithmetic intensity stays low; increasing batch size improves performance by parallelism, not intensity.
  - Kernel comparisons (Figure 8):
    - SGMV vs `Loop` (per‑adapter for‑loop) vs `Gather‑BMM` (materialize per‑item weight stacks then batched multiply).
    - SGMV is consistently faster; e.g., in `Distinct`, SGMV grows from ~37 µs at batch 1 to ~116 µs at batch 64, while Gather‑BMM degrades steeply due to extra memory I/O. `Loop` is worst because it runs many batch‑1 kernels.
    - In `Identical`, all reduce to BMM; SGMV still matches or slightly beats `torch.bmm()` (≈37–40 µs), indicating an efficient implementation for LoRA’s shapes.
  - Effect of LoRA rank (Figure 9): with ranks 8/16/32/64, SGMV latency is nearly flat across batch sizes when there is adapter sharing (Uniform/Skewed/Identical). In Distinct, batch‑64 latency increases with rank (≈72/75/89/118 µs), reflecting the IO‑bound nature and extra work.

- Transformer layer latency (Figure 10)
  - For Llama‑2 7B and 13B with sequence length 512 or 2048, increasing batch from 1→32 raises layer latency modestly, especially at shorter sequences (≈72% increase at len=512). At longer sequences, attention dominates, and batching benefits at the layer level are smaller.
  - Importantly, layer latency is “LoRA‑agnostic” across popularity distributions (Distinct/Uniform/Skewed/Identical curves overlap), confirming that SGMV removes the penalty for mixing adapters.

- End‑to‑end single‑GPU throughput (Figure 11)
  - 7B model: Punica sustains ≈1,044 tokens/s across popularity patterns, whereas baselines collapse in `Distinct` and `Uniform/Skewed` because they cannot build large batches across adapters. In `Identical`, vLLM (backbone‑only) slightly beats Punica (≈1,140 tokens/s vs Punica’s slightly lower), as expected when no LoRA overhead exists and vLLM exploits continuous batching.
  - 13B model: similar pattern; Punica ≈693 tokens/s; baselines only perform well when all requests share one adapter.
  - Decode latency overhead: Figure 1 shows decode step grows only ≈2 ms (11→13 ms) when batch size goes 1→32 for short contexts, aligning with the claim that batching adds ≈2 ms/token latency.

- 70B with tensor parallelism (8 GPUs per model; Figure 12)
  - Punica maintains ≈441–446 tokens/s across Distinct/Uniform/Skewed/Identical.
  - vLLM (backbone‑only) achieves ≈457 tokens/s in `Identical` but drops to ≈21–25 tokens/s when many LoRA adapters are involved because it cannot batch across them. Punica’s cross‑LoRA batching preserves high throughput.

- Cluster behavior (16 GPUs; Figure 13)
  - Over a one‑hour variable‑load trace (Zipf‑1.5 adapter popularity), Punica keeps active GPUs near the max batch size 32, migrates requests when `KvCache` fills, and allows idle GPUs to remain idle—making scale‑down decisions simple.
  - The middle panel shows high, smoothly varying token throughput that tracks the request rate; the bottom panel shows consolidation (many GPUs at either batch 32 or 0).

- Do results support the main claims?
  - Throughput: yes—Figures 11–12 show order‑of‑magnitude gains vs LoRA‑unaware baselines under multi‑adapter mixes. The abstract summarizes this as:
    > “Punica achieves 12× higher throughput … while only adding 2 ms latency per token.” (Abstract)
  - Latency: micro and layer results (Figures 1 and 10) show minimal per‑token penalty from larger batches, and kernel timings (Figures 7–9) show SGMV overhead is small.

- Ablations and robustness
  - Kernel‑level ablations include different ranks and popularity mixes (Figures 7–9).
  - The system studies several LoRA distributions, three model sizes, and both single‑GPU and tensor‑parallel regimes.
  - Failure cases are not explicitly cataloged, but the Distinct case microbenchmarks clarify limits when every request has a unique adapter (IO‑bound regime).

## 6. Limitations and Trade-offs
- Scope restricted to LoRA‑style adapters
  - The SGMV formulation depends on the AB low‑rank add‑on. Other parameter‑efficient methods (Adapters, Prefix‑Tuning, LoRA variants with non‑linearities) may not map directly without new kernels.
- Decode‑centric optimization
  - Prefill is run one at a time per batch invocation to minimize latency (Section 5). Workloads dominated by very long prompts (prefill heavy) may see less benefit.
- When every request uses a unique adapter
  - SGMV degrades to many matrix–vector ops that are memory‑bandwidth bound (Section 4; Figure 7–9). Punica still batches, but the speedup headroom is smaller than in adapter‑sharing scenarios.
- Scheduling trade‑offs
  - Consolidation (sending work to already busy GPUs) maximizes throughput but may create fairness concerns or longer tails for some requests under strict SLAs; batch size is capped at 32 to control latency (Section 5.1).
- Migration overheads
  - Rebuilding `KvCache` by re‑prefilling costs time proportional to prompt length. Migration is only triggered by memory pressure, but workloads with very long prompts could pay a higher cost.
- Hardware and precision assumptions
  - Results are on A100 GPUs with FP16 Tensor Cores and PCIe Gen4 x16. The 2 ms adapter load and kernel efficiencies may differ on other accelerators or interconnects.
- Security and isolation
  - The paper focuses on performance. Multi‑tenant isolation (e.g., side channels, memory zeroing between adapters) is not discussed.

## 7. Implications and Future Directions
- How this changes the landscape
  - Serving thousands of LoRA variants no longer requires duplicating base models or segregating traffic by adapter. A single GPU can host one backbone and efficiently time‑share many adapters with high throughput, enabling economical “adapter as a service.”
- Practical applications
  - Multi‑tenant inference for enterprises that host per‑customer customizations.
  - Rapid A/B testing of LoRA variants without preloading.
  - Cost‑effective deployment of domain‑specialized chatbots and tools that share a foundation model.
- Follow‑up research enabled or suggested
  - Extend SGMV‑like batching to other adapter families (e.g., prefix/LoRA hybrids, sparse adapters) and to training‑time mixed‑adapter fine‑tuning.
  - Combine with speculative decoding (Section 8) to further raise decode arithmetic intensity.
  - Integrate model/KvCache quantization and compression (Section 8) to increase context length headroom and further reduce latency.
  - SLA‑aware schedulers that balance consolidation with per‑tenant latency fairness; predictive migration strategies that avoid re‑prefill spikes.
  - Heterogeneous clusters (mix of GPU generations) and cross‑node adapter placement to minimize PCIe/NVLink traffic.
  - Safety/isolation work for multi‑tenant environments (e.g., zero‑copy but isolated adapter pools).

In short, Punica contributes a kernel‑level technique (SGMV) and a system design that together unlock cross‑adapter batching for LoRA at scale. The experiments (Figures 7–13) demonstrate that this turns the main source of inefficiency in multi‑tenant LoRA serving—lack of batching—into an advantage, achieving near‑backbone throughput even when many different adapters are active, with only millisecond‑level per‑token latency impact.
