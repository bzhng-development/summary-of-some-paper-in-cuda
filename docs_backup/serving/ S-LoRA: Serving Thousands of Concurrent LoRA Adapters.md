# S-LoRA: Serving Thousands of Concurrent LoRA Adapters

**ArXiv:** [2311.03285](https://arxiv.org/abs/2311.03285)
**Authors:** Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou, Banghua Zhu, Lianmin Zheng, Kurt Keutzer, Joseph E. Gonzalez, Ion Stoica
**Institutions:** 

## 🎯 Pitch

S-LoRA revolutionizes large language model deployment by enabling the concurrent serving of thousands of Low-Rank Adaptation (LoRA) fine-tuned variants on a single machine with low overhead. By decoupling base model computation from adapter computation and unifying memory management, it significantly boosts throughput, turning LoRA’s fine-tuning efficiency into practical deployment efficiency, crucial for scalable, personalized applications like chatbots and digital assistants.

---

## 1. Executive Summary
S-LoRA introduces a serving system that can run thousands of Low-Rank Adaptation (LoRA) fine-tuned variants of a single large language model concurrently on one machine with minimal overhead. It achieves this by decoupling base-model computation from adapter computation, unifying GPU memory management for dynamic data, and adding custom kernels and multi-GPU parallelism strategies that keep throughput high (e.g., up to 4× higher than a strong serving baseline in Table 3).

## 2. Context and Motivation
- Problem addressed
  - When organizations fine-tune a single large language model (LLM) for many users or tasks using LoRA adapters, they must serve a huge set of model variants concurrently. Prior practices either merge each adapter into a full copy of the model or swap adapters serially, both of which reduce throughput and scale poorly.
  - Online decoding makes this harder: requests arrive continuously with variable prompt/generation lengths, requiring token-level scheduling and large “KV cache” tensors to store prior token states for attention.

- Why this matters
  - Real services (e.g., personalized assistants) may need to host thousands to millions of personalized adapters while keeping latency and cost low. Efficient multi-adapter serving turns LoRA’s fine-tuning efficiency into practical deployment efficiency.

- Prior approaches and shortcomings
  - LoRA merging: Merge `W' = W + A B` into the base weights to remove runtime overhead for a single adapter (Section 2, Eq. (1)). This breaks down when serving many adapters because it requires multiple full weight copies or frequent add/subtract cycles, which destroys batching and adds latency (Section 4.1).
  - Batch scheduling and KV cache management: Orca’s iteration-level batching (token-level) improves throughput, and vLLM’s PagedAttention reduces KV-cache fragmentation (Section 2.1). However, neither provides a mechanism to also manage dynamic adapter weights or to batch heterogeneous adapter computations.

- Positioning of this work
  - S-LoRA unifies these threads: it keeps one copy of the base model, streams only needed adapters to GPU, generalizes PagedAttention to also manage adapter weights, and uses custom kernels and a new tensor-parallel plan so that thousands of adapters can be served concurrently with high throughput (Sections 3–6).

## 3. Technical Approach
S-LoRA restructures how inference is computed and how memory is managed to make multi-adapter batching practical.

- Core idea: separate and batch differently
  - The base forward pass is split into two parts using LoRA’s formulation:
    - Base model term: `x W` (dominant cost).
    - Adapter term: `x A B` (much smaller because `rank r << hidden size h`; Section 2, Eq. (2)).
  - Strategy:
    - Batch and execute `x W` across all requests—this is easy because all requests share the same base model (Figure 1, “Batched base computation”).
    - Compute `x A B` per-request/per-adapter with specialized kernels that can operate on non-contiguous memory and heterogeneous ranks and sequence lengths (Figure 1, “Batched LoRA computation”; Section 5.3).

- Memory architecture: “Unified Paging” (Section 5.1; Figure 3)
  - Definitions:
    - `KV cache`: tensors storing past token keys/values so each new token can attend to its history. Per layer, shaped `(S, H)` where `S` is sequence length and `H` is hidden size.
    - LoRA adapter weights: low-rank matrices `A (H×r)` and `B (r×H)`; often discussed as a single low-rank update with shape “rank × hidden”.
  - Observation (Section 5.1): KV caches and adapter weights both share dimension `H`; their other dimension (`S` vs. `R`) varies dynamically.
  - Unified memory pool:
    - Reserve a large GPU buffer for all dynamic tensors; store KV cache and LoRA weights together in “pages,” each page being a vector of length `H`.
    - KV cache for length `S` consumes `S` pages; a LoRA tensor of rank `R` consumes `R` pages. Both are stored interleaved and non-contiguously (Figure 3), reducing fragmentation compared to separate allocators.

- Adapter movement between host and GPU (Figure 2; Section 4.1 and 5.2)
  - All adapters reside in main memory (CPU). Only the adapters needed for the currently running batch are fetched to GPU (Figure 2, “Fetch active adapters…”).
  - Prefetching (Section 5.2): While decoding the current batch, S-LoRA predicts which adapters will be used next (from the waiting queue) and asynchronously loads them, overlapping I/O with compute.

- Scheduling to enhance batching (Section 4)
  - Iteration-level scheduling (Section 4.1): adopt token-level batching (from Orca), so arriving requests can be added to the active batch mid-decoding if memory permits.
  - Adapter clustering (Section 4.2; Appendix A.2): simple heuristic that prefers forming batches with fewer distinct adapters to free more memory for KV cache, potentially increasing batch size. There are fairness/latency trade-offs (discussed in Section 4.2 and Appendix A.2).
  - Admission control via “early abort” (Section 4.3; Appendix B): when overloaded, estimate which waiting requests can still meet the SLO for first-token latency; abort those unlikely to meet it and prioritize recent arrivals. Appendix B provides a rationale: if user reward decreases with latency, serving the latest `l` requests maximizes reward (Theorem B.1).

- Custom kernels for non-contiguous, heterogeneous LoRA compute (Section 5.3)
  - Because Unified Paging stores tensors non-contiguously, standard batched GEMM would require heavy padding and copies.
  - Two tailored kernels gather scattered adapter pages and multiply them without padding:
    - MBGMM (Multi-size Batched Gather Matrix–Matrix): for prefill (multiple tokens at once). Implemented in Triton with tiling.
    - MBGMV (Multi-size Batched Gather Matrix–Vector): for decode (single token). Two versions: Triton and an optimized variant derived from Punica’s earlier kernel, extended for non-contiguous memory and multiple ranks. The latter is faster in experiments.

- Multi-GPU tensor parallelism for LoRA (Section 6; Figure 4)
  - Goal: add LoRA computations on top of a Megatron-LM-style tensor-parallel base without introducing large extra communication.
  - Partitioning (Section 6.1; Figure 4):
    - Align LoRA partitions with the base model’s: first MLP linear (`W1`) is column-partitioned; second (`W2`) is row-partitioned. LoRA matrices `A1, B1, A2, B2` are partitioned to match.
    - Communications are scheduled on small intermediate tensors from LoRA and fused with base model communications when possible (e.g., fuse matmul_4 all-gather with final all-reduce; Figure 4, red dashed box).
  - Communication analysis (Section 6.2):
    - Base model: one all-reduce costs `2(N−1) B h / N`.
    - Added LoRA: three all-gathers + one all-reduce over rank-size intermediates cost `5(N−1) B r / N`.
    - Since `r << h`, extra cost is negligible relative to base model communication.

## 4. Key Insights and Innovations
- Unified Paging for both KV cache and adapters (Section 5.1; Figure 3)
  - Novelty: extends vLLM’s PagedAttention concept beyond KV caches to manage LoRA weights in the same paged pool.
  - Significance: reduces fragmentation from dynamic adapter loads/unloads and dynamic KV caches; enables larger token-level batches, which drive throughput.

- Heterogeneous LoRA batching on non-contiguous memory (Section 5.3)
  - Novelty: custom MBGMM/MBGMV kernels gather adapter pages of varying ranks directly from the paged pool without padding, and operate efficiently batch-wise.
  - Significance: avoids the copy-and-pad overhead that would erase the benefits of on-the-fly LoRA compute; supports mixed sequence lengths and ranks.

- Decoupled compute with batched base-model GEMM + on-the-fly LoRA (Section 4.1; Figure 1; Eq. (2))
  - Insight: The heavy `x W` GEMM can be batched across adapters, while the cheaper `x A B` is computed per-adapter. The small extra compute is vastly offset by the batching gains on `x W`.
  - Significance: replaces “merge weights” serving, which prevents cross-adapter batching and multiplies memory footprint.

- LoRA-aware tensor parallelism with fused communications (Section 6; Figure 4)
  - Novelty: partition and communication scheduling that place LoRA’s added compute onto small intermediates, and fuse an all-gather with a final all-reduce.
  - Significance: keeps multi-GPU overhead low and enables scaling to 30B/70B models with near-ideal parallel efficiency (Figure 8).

- Early-abort admission control grounded in a simple optimality argument (Section 4.3 and Appendix B)
  - Idea: when overloaded, serve the latest requests likely to meet the SLO, supported by a proof sketch (Theorem B.1) and a practical heuristic.
  - Significance: improves SLO attainment and “user satisfaction” versus FCFS/LCFS under high variability (Figure 10).

## 5. Experimental Analysis
- Setup (Section 7.1)
  - Models: LLaMA families at 7B/13B/30B/70B with various adapter ranks (Table 1). Example: `S1` = LLaMA-7B with rank 8; `S2` = 7B with ranks {64, 32, 16, 8}; `S4` = 13B with ranks {64, 32, 16}.
  - Hardware: single GPUs (A10G-24GB; A100-40GB; A100-80GB) and multi-GPU (2×/4× A100).
  - Baselines:
    - HuggingFace PEFT server: batches single-adapter requests and swaps adapters per batch; no continuous batching nor paged KV caches.
    - vLLM “m-packed”: since vLLM lacks native LoRA, each adapter is merged into its own full model copy; multiple workers (via NVIDIA MPS) share a GPU.
  - Metrics: throughput (req/s), average request latency, first-token latency, SLO attainment (% with first token under 6s), and a “user satisfaction” score (Appendix B).

- Main results
  - Scaling to thousands of adapters on a single GPU (Section 7.2; Table 3; Figure 5)
    - Capacity: S-LoRA serves 2,000 adapters concurrently on a single A100-80GB (Table 3, rows “S1” and “S2”, n=2000).
    - Throughput vs vLLM-packed:
      - With 5 adapters on 7B (`S1`), S-LoRA: 8.05 req/s vs vLLM-packed: 2.04 req/s (≈4×; Table 3).
      - vLLM-packed runs out of memory beyond a handful of adapters (“OOM” entries in Table 3).
    - Throughput vs PEFT:
      - On 7B (`S1`, n=5): 8.05 req/s vs 0.88 req/s (≈9×; Table 3).
      - PEFT’s throughput collapses as adapters increase; even at low request rates it shows very high latency (Appendix A.1, Tables 4–5).
    - Stability with large n: Figure 5 shows S-LoRA’s throughput stays nearly flat after a modest decline up to around 100 adapters; beyond that, the number of “active adapters” per running batch saturates, keeping overhead constant.

  - Importance of Unified Paging and custom kernels (Ablations; Figures 5–6)
    - “S-LoRA-no-unify-mem” and “S-LoRA-bmm” (which pads and uses batched GEMM) have clearly lower throughput and higher latency (Figure 5, multiple hardware/settings). In Figure 6, their first-token latencies rise sharply with load, while S-LoRA maintains much lower latencies and higher SLO attainment.

  - Real workload trace (Section 7.3; Figure 7)
    - Using a downsampled LMSYS Chatbot Arena trace (~26 adapters; avg input 85 tokens, output 165 tokens), S-LoRA’s throughput and SLO attainment trends mirror synthetic results: throughput increases with request rate until saturation, and SLO attainment stays higher than ablations.

  - Multi-GPU tensor parallelism (Section 7.4; Figure 8)
    - 30B model: moving from 2× to 4× A100-40GB yields superlinear throughput gains (>2×) because the workload is memory-bound; adding GPUs increases both compute and available memory for KV cache/adapters. The “w/o LoRA communication” bars are very close to full S-LoRA, showing LoRA’s comm cost is tiny.
    - 70B model: similar pattern on A100-80GB.

  - When merging is better (Section 7.5; Figure 9)
    - If there is only one adapter, merging (compute `x(W + A B)`) performs best because the merge is a one-time cost. Beyond 2 adapters, throughput drops due to frequent adapter switching and poor batching; on-the-fly compute wins. Skewed adapter popularity (smaller α) further hurts merging because it shrinks batch sizes.

  - Admission control benefits (Section 7.5; Figure 10)
    - Under high variability (larger `cv`), S-LoRA’s early-abort strategy yields higher SLO attainment and “user satisfaction” than FCFS or LCFS. FCFS often wastes cycles on already-too-late requests; LCFS is too greedy for newest requests when variability is high.

- Do the experiments support the claims?
  - The evaluation covers realistic hardware, multiple model sizes, thousands of adapters, synthetic and real traces, and ablations that isolate each design choice. Key claims—serving thousands of adapters, 4× throughput vs a strong vLLM-based baseline under small-n, and major gains over PEFT—are directly evidenced (Table 3, Figures 5–8, 10).
  - Robustness: results hold across GPUs and both synthetic and real traces; ablations pinpoint where the gains come from (unified memory and custom kernels).

## 6. Limitations and Trade-offs
- Assumptions about adapter characteristics
  - The efficiency hinges on LoRA ranks being small relative to hidden size (`r << h`; Section 6.2). Extremely large ranks would increase both compute (xAB) and communication (multi-GPU), eroding advantages.

- Scope limited to LoRA-style adapters
  - While the techniques may generalize, the implementation, kernel paths, and memory layout are tailored to LoRA’s shapes and access patterns (Section 9 mentions future work to support more adapter types).

- Fairness and latency trade-offs
  - Adapter clustering improves throughput by reducing the number of active adapters per batch (Section 4.2; Appendix A.2), but can increase per-adapter waiting time and harm fairness if not tuned carefully.

- Admission control heuristics
  - The early-abort strategy’s practical implementation uses moving averages and thresholds (Appendix B), which may mispredict under non-stationary traffic or extreme burstiness.

- Memory pressure sources
  - Although Unified Paging mitigates fragmentation, the system remains memory-bound at large scales (Figure 8). Very long sequences or many concurrent long generations can still exhaust the unified pool.

- Engineering complexity
  - Maintaining custom Triton/CUDA kernels for non-contiguous gathers, heterogeneous ranks, and multi-GPU fusion adds complexity and may lag behind rapidly evolving hardware/library optimizations.

## 7. Implications and Future Directions
- Impact on the field
  - S-LoRA makes multi-tenant, personalized LLM serving practical on modest hardware by converting LoRA’s training-time savings into serving-time efficiency. It shifts the design pattern from “merge N models” to “one model + many adapters,” with shared compute and shared memory management.

- Practical applications
  - Multi-tenant SaaS for customized chatbots or copilots.
  - A/B testing and rapid iteration across thousands of fine-tuned variants.
  - Academic/enterprise platforms that host community-contributed adapters without provisioning one GPU per adapter.

- Research directions
  - Extending Unified Paging beyond LoRA: prefix/prompt-tuning, (IA)^3, or other adapter families with different shapes and reuse patterns (Section 9).
  - Smarter placement and prefetching: learning-based prediction of adapter demand; cross-node paging for cluster-scale serving.
  - Stronger fairness/SLA management: combine early abort with per-tenant quotas, preemption, or priority queues; explore more rigorous SLO-aware scheduling under token-level batching.
  - Kernel and hardware co-design: deeper fusion with attention kernels (e.g., FlashAttention-2), exploration of CUTLASS grouped GEMM versus bespoke Triton/CUDA for different rank and batch regimes (Section 5.3).
  - Multi-objective optimization: jointly tune adapter clustering, batch size, KV cache layout, and admission control to optimize throughput, latency, and fairness.

> In short, the system’s central insight—separating heavy base-model compute from light adapter compute, and unifying memory for all dynamic data—enables batched, heterogeneous multi-adapter serving at scales previously impractical. Figures 1–4 detail how; Table 3 and Figures 5–8 quantify the benefits; Figure 10 and Appendix B show how to maintain SLOs under load.
