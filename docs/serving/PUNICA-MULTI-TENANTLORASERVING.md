# PUNICA: MULTI-TENANT LORA SERVING

**ArXiv:** [2310.18547](https://arxiv.org/abs/2310.18547)

## 🎯 Pitch

Punica introduces a novel system for serving multiple Low-Rank Adaptation (LoRA) variants of large language models (LLMs) on shared GPU clusters by maintaining just a single copy of the base model in GPU memory. Its key innovation is a new CUDA kernel—Segmented Gather Matrix-Vector Multiplication (SGMV)—that allows efficient batching of requests across different LoRA models during decoding, dramatically boosting GPU throughput. This approach enables up to 12× higher serving throughput compared to existing solutions with minimal added latency, making multi-tenant, cost-effective LLM deployment at scale practical and resource-efficient.

---

## 1. Executive Summary (2-3 sentences)
Punica is a system for serving many Low-Rank Adaptation (`LoRA`) variants of a large language model (LLM) on a shared GPU cluster while keeping only one copy of the base model in GPU memory. It introduces a new CUDA kernel, Segmented Gather Matrix-Vector Multiplication (`SGMV`), that lets the GPU batch requests from different LoRA models during decoding, and a scheduler that consolidates multi-tenant workloads. The result is substantially higher throughput—up to 12× over popular serving stacks with only ~2 ms extra latency per generated token (Abstract; Section 7; Figures 11–12).

## 2. Context and Motivation
- Problem addressed
  - Serving many tenant-specific LoRA models efficiently on limited GPUs. Naively treating each LoRA model as a separate full model wastes GPU memory and compute because every LoRA model shares the same base weights (Introduction).
- Why this matters
  - Multi-tenant providers often host hundreds to thousands of LoRA adapters for the same base LLM. If each variant needs its own GPU instance, the cost scales linearly with the number of tenants. Efficient sharing lowers cost and unlocks practical deployment at scale (Introduction).
- What makes serving LLMs hard
  - Text generation has a prefill stage (process the prompt) and a decode stage (generate tokens iteratively). Decode dominates compute time and is hard to utilize well because each step processes a single token per request. Figure 1 quantifies this: 
    > “Increasing the batch size from 1 to 32, the decode step latency increases from 11 ms to 13 ms for short sequences, and from 17 ms to 34 ms for longer sequences.”  
    This means batching improves GPU utilization during decode dramatically (Section 2.1; Figure 1).
- Prior approaches and their gaps
  - Batching for LLMs (Orca; vLLM) boosts utilization, but only when requests target the same model; they do not batch across different models. Existing systems also don’t support concurrent execution of distinct LoRA adapters with one backbone copy (Sections 1–2; Related Work).
  - Treating each LoRA model as independent leads to repeated base model copies and costly model switches; even with a shared backbone, there was no efficient way to batch the adapter computation itself (Section 2.2 and Section 3).
- Positioning
  - Punica focuses on the decode stage (the main bottleneck), introduces a kernel that makes different LoRA models “batchable,” and adds a scheduler designed to fully pack a small set of GPUs before scaling out (Guidelines G1–G3; Sections 1 and 3).

## 3. Technical Approach
High-level idea: Keep one backbone LLM per GPU, swap in tiny LoRA weights on demand, and run all requests (even from different LoRA models) in one batched decode using a custom kernel for the LoRA additions. Build a scheduler that always prefers to fill up a few GPUs (with large batches) and only allocates new GPUs when necessary.

- Background: What LoRA changes in the model
  - A dense layer’s weight `W` is adapted to `W + A B`, where `A ∈ R^{h1×r}` and `B ∈ R^{r×h2}`, with `r` (LoRA rank) much smaller than layer dimensions. Only `A` and `B` are trained and stored per adapter (Section 2.2).
  - During inference, each adapted layer computes the normal backbone output plus a low-rank “add-on” `x A B`. The backbone part can already be batched; the challenge is batching the add-on for many different LoRA adapters in one kernel launch (Sections 2.2, 4).

- System architecture
  - Components: Frontend (user API), Scheduler (global control), Runner (per GPU host), and a Python LLM process per GPU (Figure 2; Section 3; Section 6).
  - Each GPU loads the base model and reserves most memory for the key–value cache (`KvCache`). LoRA matrices `A` and `B` are loaded on demand when a request arrives (Sections 3, 5.2).
  - Tokens are streamed back as they are generated (Figure 2).

- The SGMV kernel: how batching across adapters works
  - Goal: Compute `y += x A B` for many requests in one batch where requests may use different `(A, B)` pairs.
  - Two-step factorization for efficiency:
    - First compute `v := x A` (reduces to low dimension `r`) — called SGMV-shrink.
    - Then `y += v B` (expands back to layer size) — called SGMV-expand.  
    This separation allows specialized parallelization patterns (Section 4; Figure 3).
  - Segmentation idea: In the batch, group requests by LoRA model. Use segment indices `s` to mark contiguous slices per LoRA in the batched input (Section 4, before Figure 3).
  - Kernel scheduling (Figure 4; Section 4):
    - Map “which LoRA adapter” to `blockIdx.y`, so each threadblock (or group of threadblocks) multiplies features with the correct `(A, B)` for that adapter.
    - SGMV-expand: split the output dimension and assign stripes to different threadblocks; concatenate partial outputs.
    - SGMV-shrink: output is thin; use Split-K style parallelism—split the input feature dimension, compute partial sums in parallel, then do a cross-block reduction.
    - Use Tensor Cores for these matmul-like pieces when compute-bound; when every request has a unique LoRA (matrix-vector case), switch to an IO-optimized schedule without Tensor Cores to saturate memory bandwidth (end of Section 4).
  - Why this design: It maximizes reuse and parallelism when multiple requests share an adapter and still runs efficiently when all adapters differ. It ensures that batching different LoRAs is nearly as efficient as batching identical ones (Sections 4 and 7.1; Figures 7–10).

- Scheduling new requests (Section 5.1)
  - Maintain each GPU’s working set (batched requests) and available KvCache memory.
  - Place a new request on the GPU with the largest current batch that still has enough KvCache memory and is below the max batch size (32 on A100, chosen by profiling):
    > “We profile A100 GPUs and decide to set the maximum batch size to 32.” (Section 5.1)
  - If all GPUs are saturated, queue requests FCFS; if the whole cluster is busy, request more GPUs; if GPUs go idle, release them (Section 5.1).

- On-demand loading of LoRA weights (Section 5.2)
  - Adapter weights are tiny (1% or less of the base model). Copying all adapter weights to GPU memory takes around milliseconds:
    > “On PCIe Gen4 x16, it takes around 50 µs to load a layer and 2 ms to load the entire model.” (Section 5.2)
  - Copy is asynchronous and overlapped with ongoing decode of other requests, so the new request naturally joins the batch as soon as its adapter finishes loading (Section 5.2).

- Request migration between GPUs (Section 5.3; Figure 5)
  - When a GPU runs out of KvCache space, evict the newest request and move it to another GPU. Migration uses recomputation: rebuild the KvCache on the destination by running a single prefill over the original prompt plus already-generated tokens (Figure 5 steps 1–6).
  - Rationale: Moving the whole KvCache is often slower than recomputing it; the paper cites and aligns with the PagedAttention study (Section 5.3).

- KvCache layout for efficient continuous batching (Section 5.4; Figure 6)
  - Problem with typical layouts (e.g., HuggingFace): concatenating along sequence length forces all requests in a batch to stay together until all finish, causing waste when some finish early (Figure 6).
  - Punica uses a paged, separable layout with the batch dimension outermost and virtual pages to avoid fragmentation:
    > KvCache shape is `[sum_i ceil(S_i/P), L, 2, N, P, D]`, where `P` is page size, `S_i` is sequence length, `L` layers, `N` heads, `D` head dimension (Section 5.4).
  - This enables “continuous batching”: remove finished requests and add new ones without reshuffling or re-copying huge tensors (Sections 5.4 and 6).

- Additional implementation details (Section 6)
  - Python library (PyTorch extension with PyBind11) integrates FlashInfer for fast attention and a fused LayerNorm (reduced from 110 µs to 4 µs).
  - Prefill and decode are co-batched in one model invocation: prefill is limited to batch size 1 for latency; dense projections and LoRA add-ons treat all tokens as one batch to maximize compute efficiency. The system builds `BatchLen` metadata and SGMV segment indices once per invocation (Section 6).
  - Control-plane components (scheduler, runner, frontend) are in Rust; runners spawn the Python subprocess per GPU (Section 6).

## 4. Key Insights and Innovations
- SGMV kernel: batch across different LoRA adapters (fundamental)
  - What’s new: A CUDA kernel that factors the LoRA computation (`xAB`) into two matvec/matmul-like stages and schedules them so each batch can include multiple adapters efficiently (Section 4; Figure 4).
  - Why it matters: It removes the core barrier to batching multi-tenant workloads, making heterogeneous requests nearly as efficient to batch as homogeneous ones (Figures 8–10). In the LoRA-Identical case, SGMV matches or outperforms `torch.bmm()`, and in mixed-adapter cases it vastly outperforms naive loops and Gather-BMM (Section 7.1; Figure 8).

- Decoding-first design with continuous batching and paged KvCache (significant systems innovation)
  - What’s new: Co-batching prefill and decode to boost dense/LoRA efficiency, plus a KvCache layout that supports adding/removing requests mid-run without copying the entire cache (Sections 5 and 6).
  - Why it matters: Decode latency dominates; keeping high batch size during decode drives utilization (Figure 1). The paged layout avoids the “wasted steps” seen in common layouts (Figure 6), enabling steady high throughput (Figure 13 lower panel shows GPUs operating at batch size near 32 most of the time).

- On-demand LoRA loading with overlap and LoRA-agnostic scheduling (practical, impactful)
  - What’s new: Treat adapters as millisecond-scale, overlappable transfers so scheduling isn’t constrained by which adapters are already loaded (Section 5.2).
  - Why it matters: Decouple placement from adapter residency; you can route to any hot GPU and still keep high utilization. The measured 2 ms load per full adapter is small compared to ~30 ms per decode step and thousands of steps per request (Section 5.2).

- Simple cluster policy that maximizes utilization while staying within latency “sweet spot” (incremental but effective)
  - What’s new: Place new requests on the GPU with the largest current batch (subject to memory and a profiled max batch=32) so busy GPUs stay busy, idle GPUs stay idle, and scale up/down decisions are easy (Section 5.1).
  - Why it matters: Results show high, stable throughput, with GPUs running near max batch size in production-like traces (Section 7.3; Figure 13).

## 5. Experimental Analysis
- Setup and workloads
  - Hardware:
    - Testbed #1: single NVIDIA A100 80 GB (for single-GPU and microbenchmarks).
    - Testbed #2: two HGX A100 8-GPU servers with NVSwitch (for 70B with tensor parallelism and cluster experiments) (Section 7).
  - Models: LLaMA 2 7B, 13B, 70B; LoRA rank 16, applied to all dense projections (Section 7).
  - Workload characteristics:
    - Prompt/response length distributions from ShareGPT (Section 7).
    - Four adapter-popularity regimes:
      - `Distinct`: every request uses a different adapter.
      - `Uniform`: roughly √n adapters.
      - `Skewed`: Zipf-1.5 popularity.
      - `Identical`: all requests use the same adapter (Section 7).
  - Baselines:
    - HuggingFace Transformers + PEFT (LoRA-capable), DeepSpeed (LoRA-capable), FasterTransformer (backbone-only), and vLLM (backbone-only). Model switching costs are omitted for baselines, which makes the comparison conservative in Punica’s favor for adapter churn scenarios (Section 7).

- Microbenchmarks: kernel and layer behavior
  - Roofline for SGMV (Figure 7):
    - Distinct adapters: increasing batch size raises achieved FLOPs (more parallel work) at roughly constant arithmetic intensity.
    - Identical adapters: performance tracks memory bandwidth, indicating SGMV is memory-bound in this case.
    - Uniform/Skewed: sit between the two extremes—benefit from both higher parallelism and more reuse (Section 7.1; Figure 7).
  - LoRA operator latency (Figure 8):
    - SGMV vs alternatives:
      > In Distinct, SGMV grows modestly with batch size (≈37 µs → 116 µs for batch 1→64), while a naive loop performs poorly and Gather-BMM slows due to extra memory IO.  
      > In Identical, all methods reduce to BMM semantics; SGMV stays ≈37–40 µs and effectively implements this case more efficiently than `torch.bmm()` (Section 7.1; Figure 8).
  - LoRA rank sensitivity (Figure 9):
    > With ranks 8,16,32,64, single-request latency is ≈42 µs across ranks; at batch 64 it increases to ≈72, 75, 89, 118 µs respectively in Distinct. Under Uniform/Skewed/Identical, latency remains ≈42–45 µs across batch sizes, indicating reuse dominates (Section 7.1; Figure 9).
  - Transformer layer latency (Figure 10):
    > For 7B and 13B at len=512, increasing batch size 1→32 increases layer latency by only ~72%; when len=2048, attention costs dominate, so batching helps less. Crucially, layer latency curves are similar across Distinct/Uniform/Skewed/Identical, showing LoRA add-on time is small compared to backbone + attention (Section 7.1; Figure 10).

- Single-GPU end-to-end throughput (Figure 11)
  - 7B model:
    > Punica achieves 1044 tokens/s across workloads; baselines only approach high throughput in Identical (same adapter), and degrade sharply in Distinct/Uniform/Skewed because they cannot batch across adapters. vLLM (backbone-only) hits 1140 tok/s in Identical due to continuous batching (Section 7.2; Figure 11a).
  - 13B model:
    > Punica reaches 693 tok/s; vLLM (backbone-only) reaches 789 tok/s in Identical; other LoRA-capable baselines suffer when adapters differ (Section 7.2; Figure 11b).

- 70B with tensor parallelism on 8 GPUs (Figure 12)
  - vLLM backbone-only:
    > ≈457 tok/s in Identical but only ≈21–25 tok/s when adapters differ (cannot batch across adapters).
  - Punica:
    > ≈441–446 tok/s regardless of adapter popularity, matching backbone-only performance when all adapters are identical and maintaining it when they differ (Section 7.2; Figure 12).

- Cluster deployment on 16 GPUs (Figure 13)
  - Under a Poisson arrival process with Zipf-1.5 adapter popularity:
    > GPUs operate near max batch size (32) most of the time; throughput scales with request rate and consolidates on fewer GPUs when load drops. The bottom panel shows sustained large batches despite request churn, illustrating the scheduler’s consolidation effect (Section 7.3; Figure 13).

- Do the experiments support the claims?
  - Yes, on three axes:
    - Kernel-level: SGMV consistently outperforms naive loops and Gather-BMM and nearly eliminates the cost gap between batching identical and different adapters (Figures 7–9).
    - Model-layer and end-to-end: LoRA add-on cost is small and batching different adapters achieves similar layer latencies; end-to-end throughput remains high across adapter distributions (Figures 10–12).
    - Systems behavior: The scheduler keeps GPUs near the batch-size sweet spot; the cluster plot shows effective consolidation (Figure 13).
  - Caveat on baselines: FasterTransformer and vLLM are used in backbone-only mode; they do not support LoRA multi-adapter batching, which is exactly Punica’s target. Punica’s advantage on multi-adapter cases is therefore expected, but the comparisons still quantify how large the gap is in practice (Section 7 Baselines).

## 6. Limitations and Trade-offs
- Scope limited to LoRA-style adapters
  - The approach depends on the low-rank structure `AB`. Other adapter types (e.g., prefix-tuning, prompt tuning, or arbitrary module deltas) would need their own batched kernels; PetS addresses other adapters but without concurrent multi-adapter execution (Related Work).
- Decode-focused design
  - Prefill is handled with batch size 1 to minimize latency, and the performance focus is on decode (by far the dominant cost). Workloads dominated by extremely long prompts and very short generations would see less benefit (Sections 2.1, 5).
- Simple placement heuristic and SLOs
  - The “send to the busiest eligible GPU” policy maximizes throughput but does not explicitly optimize per-request tail latency or fairness across tenants. There are no explicit SLO guarantees beyond using a max batch size cap (Section 5.1).
- Migration recomputes KvCache
  - Rebuilding KvCache on the destination GPU adds a prefill-equivalent cost. The paper argues this is typically cheaper than copying the cache (Section 5.3), but for very long contexts under tight latency constraints, moving KvCache could sometimes be better.
- Hardware assumptions
  - Results target A100 GPUs, exploit Tensor Cores, and rely on PCIe Gen4 or NVSwitch bandwidth. Performance portability to smaller GPUs or different interconnects is not evaluated (Sections 6–7).
- Reported latency overhead
  - The abstract claims ~2 ms extra latency per token with Punica’s approach at high throughput; the paper demonstrates strong throughput, but a detailed per-token latency breakdown under varied loads is limited (Abstract; Section 7).

## 7. Implications and Future Directions
- How this changes the field
  - Multi-tenant adapter serving becomes a first-class, high-efficiency workload: one backbone per GPU, many adapters concurrently, and batching works even when adapters differ. This removes a key bottleneck for scalable, cost-effective customization of LLMs.
- Practical applications
  - Hosted fine-tuning platforms, enterprise deployments with many departmental adapters, A/B testing of adapter variants, personalization at scale (each user or team can have its own adapter without extra GPUs).
- Research and engineering directions
  - Extend SGMV-like batching to other parameter-efficient fine-tuning methods (adapters beyond LoRA).
  - Combine with speculative decoding and draft-model techniques to compound decode-stage gains (Related Work).
  - Smarter cluster schedulers: per-tenant fairness, SLO-aware batching, elastic adapter caches across nodes, prefetching adapters based on predicted popularity.
  - Explore tighter integration with quantization/compression of model and KvCache to push throughput and memory headroom further (Related Work).
  - Generalize to heterogeneous hardware (e.g., consumer GPUs) and multi-node distributed settings with adapter caching and peer-to-peer sharing.

Overall, Punica shows that the core obstacle to efficient multi-tenant serving—batching across different adapters—can be removed with the right kernel and cache layout. With SGMV and a throughput-first scheduler, it delivers near-backbone throughput even when every request targets a different LoRA model (Figures 11–12), and consolidates load effectively in a cluster (Figure 13).
