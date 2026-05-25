# S-LoRA: Serving Thousands of Concurrent LoRA Adapters

**ArXiv:** [2311.03285](https://arxiv.org/abs/2311.03285)

## 🎯 Pitch

S-LoRA revolutionizes large language model deployment by enabling a single machine to serve thousands of concurrent LoRA adapters efficiently, making it practical to deliver personalized and task-specific models at scale. By introducing unified memory management, custom GPU kernels for heterogeneous batching, and novel tensor parallelism, S-LoRA slashes latency and boosts throughput up to 4× over state-of-the-art serving systems, empowering organizations to offer massive-scale customization without sacrificing efficiency or cost.

---

## 1. Executive Summary
S-LoRA is a serving system that lets one machine (one or a few GPUs) run thousands of fine‑tuned variants of a large language model at the same time by separating and batching the base model’s computation while computing each LoRA adapter’s contribution on-the-fly. It introduces a unified GPU memory pool, custom heterogeneous-batching kernels, and a new tensor-parallelism scheme so that throughput improves up to 4× over a strong serving baseline and up to 30× over a popular LoRA library, while scaling to thousands of adapters (Abstract; Sections 3–6; Table 3; Figures 5–8).

## 2. Context and Motivation
- Problem addressed
  - Organizations often fine-tune one large base model into many task- or user-specific variants using parameter-efficient methods such as LoRA. The serving challenge is running many adapters concurrently without duplicating the full base model, while keeping latency and GPU memory usage low (Introduction; Section 2).
- Why this matters
  - Real systems may need to serve thousands or millions of custom assistants or domain models. Without efficient multi-adapter serving, costs and latency balloon: GPU memory gets fragmented, and batching opportunities are lost, reducing throughput (Introduction).
- Prior approaches and their limits
  - LoRA during inference can be “merged” into the base weights (i.e., add low-rank matrices to base weights once) so there is no runtime overhead for a single adapter. But this does not support concurrent adapters and destroys batching across adapters, since each merged model becomes a separate full copy or requires serial add/subtract operations (Section 4.1).
  - General LLM serving systems use iteration-level continuous batching and GPU KV-cache paging (e.g., Orca and vLLM’s PagedAttention), but they do not manage dynamic adapter weights or batch heterogeneous adapter computations (Section 2.1 and 3).
  - HuggingFace PEFT can swap adapters between batches but lacks continuous batching and KV-cache paging; it suffers from much lower throughput and cannot scale to many adapters in practice (Appendix A.1; Table 4–5).
- Positioning
  - S-LoRA extends high-throughput LLM serving ideas to the multi-adapter setting by: (1) decoupling base-model compute from adapter compute, (2) managing KV caches and adapter weights in one paged memory pool, and (3) introducing kernels and parallelism specifically for heterogeneous LoRA computations (Sections 3–6).

Key terms used below
- `LoRA adapter`: a pair of trainable low-rank matrices `A` and `B` added to a base weight `W` so that the effective weight is `W' = W + A B`. The adapter’s rank `r` is small compared to the hidden dimension `h` (Section 2; Eq. 1–2).
- `KV cache`: cached key/value tensors that store past token states for autoregressive decoding; it dominates memory during long generations (Section 2.1).
- `PagedAttention`: a paging scheme for KV caches which stores them in fixed-size “pages” to reduce GPU memory fragmentation (Section 2.1).
- `Unified Paging` (new): S-LoRA’s memory pool that pages both KV caches and adapter weights together (Section 5.1).
- `Prefill` vs `Decode`: prefill processes the prompt (many tokens at once), decode generates one token at a time iteratively (Section 2.1; Section 5.3).

## 3. Technical Approach
At a high level, S-LoRA treats the base model as a shared, batchable computation and computes each adapter’s small low-rank update on-the-fly, while carefully managing memory and parallelism so that the extra compute and movement are negligible.

Step-by-step

1) Separate and batch the base and adapter computations
- LoRA algebra: for an input activation `x`, the transformed output is `xW' = x(W + AB) = xW + xAB` (Section 2; Eq. 1–2).
- Design choice: do not merge `AB` into `W` at serve time when serving many adapters. Instead:
  - Batch `xW` across all current requests regardless of adapter because everyone shares `W`.
  - Compute the small correction `xAB` for each request’s adapter separately on-the-fly (Section 4.1; Figure 1).
- Why this choice: `xW` dominates compute and is batch-friendly; `xAB` is cheap because `r << h`, and computing it on-the-fly avoids carrying many merged model copies and enables multi-adapter batching (Section 4.1).
- Implementation detail: naive batched GEMM for many adapters of different ranks and sequence lengths would require padding and lead to poor utilization. S-LoRA instead uses custom kernels (below) that operate on non-contiguous memory without padding (Section 4.1; Section 5.3).

2) Memory management with Unified Paging
- Observation: adapter weights are dynamic and vary in size (rank `r`), similar to how KV caches vary with sequence length. Both share the hidden dimension `H = h` (Section 5.1).
- Unified Paging mechanism:
  - Allocate a single large GPU memory pool statically, after reserving space for base weights and temporary activations (Figure 2 and Section 5.1).
  - Store both KV caches and adapter weights as “pages” where each page corresponds to a vector of `H` elements; a KV cache of sequence length `S` uses `S` pages, and an adapter of rank `R` uses `R` pages (Figure 3; Section 5.1).
  - This interleaved, non-contiguous layout reduces fragmentation and lets the system flexibly trade space between KV caches and adapters as batches evolve (Section 5.1).
- Prefetch and overlap:
  - While decoding the current batch, the system predicts which adapters will be needed next from the waiting queue and prefetches them into free pages, overlapping I/O with compute (Section 5.2).

3) Heterogeneous batching kernels on non-contiguous memory
- Because adapter pages are non-contiguous and ranks differ across requests, S-LoRA implements two gather-and-multiply kernels:
  - `MBGMM` for prefill: multi-size batched gather matrix–matrix multiply that pulls scattered adapter pages for many tokens and ranks (Section 5.3).
  - `MBGMV` for decode: multi-size batched gather matrix–vector multiply for a single token per request, again with mixed ranks and non-contiguous pages (Section 5.3).
- These kernels are written in Triton and a modified Punica kernel, extending to non-contiguous memory, multiple ranks per batch, and finer-grained gathers (Section 5.3).

4) Scheduling to maximize batching and protect latency
- Iteration-level (token-level) continuous batching: new requests are injected into the current decoding batch as soon as memory allows; finished requests exit immediately (Section 4.1).
- Optional `adapter clustering`: prioritize requests that use adapters already present in the running batch to reduce the number of active adapters and free more memory for KV caches, which can increase batch size and throughput (Section 4.2; Appendix A.2).
- `Admission control` with an early-abort heuristic: when overloaded, serve the latest arriving requests that can still meet an SLO for first-token latency, and abort those that are already guaranteed to miss it; this is supported by a simple optimality result for a non-increasing reward function of latency (Section 4.3; Appendix B and Theorem B.1).

5) Multi-GPU tensor parallelism for LoRA-aware serving
- Goal: extend Megatron-LM’s tensor parallelism used for the base model to also cover LoRA computations with minimal extra communication (Section 6).
- Strategy (illustrated in Figure 4):
  - For a 2-layer MLP block, the base weights follow Megatron’s scheme: first weight `W1` column-partitioned, second weight `W2` row-partitioned; a single all-reduce combines partial sums.
  - For the LoRA paths:
    - For `W1`, partition `A1` and `B1` to match column partitions; use an all-gather on the small intermediate `r`-dimension tensor.
    - For `W2`, partition `A2` row-wise and `B2` column-wise; use an all-reduce on the `r`-dimension intermediate.
    - Fuse communications: the all-gather for one LoRA matmul is fused with the final all-reduce that already exists for the base model (red box in Figure 4).
- Communication cost analysis (Section 6.2):
  - Base model: one all-reduce costs about `2 (N−1) B h / N`, where `N` is number of GPUs, `B` tokens, `h` hidden size.
  - Added LoRA: roughly `5 (N−1) B r / N` across Q, K, V, and output projections.
  - Since `r << h`, the LoRA communication cost is negligible relative to the base model and partially fused with it.

## 4. Key Insights and Innovations
- Unified Paging for both KV caches and adapter weights (Section 5.1; Figure 3)
  - What’s new: extends paging from only KV caches (PagedAttention) to also include dynamically loaded LoRA adapter pages in the same pool with the same page size (`H`).
  - Why it matters: reduces fragmentation and lets the system adapt the footprint of KV caches vs adapters as workloads shift, thereby maintaining large batch sizes and throughput.

- Heterogeneous batching kernels on non-contiguous memory (Section 5.3)
  - What’s new: custom MBGMM/MBGMV kernels that directly gather scattered adapter pages and multiply without padding, supporting mixed ranks per batch.
  - Why it matters: avoids the overhead of copying to contiguous buffers or padding to a maximum rank, which would waste memory and GPU cycles; enables efficient multi-adapter batching in practice.

- LoRA-aware tensor parallelism with fused communication (Section 6; Figure 4)
  - What’s new: a partition scheme that aligns LoRA partitions with Megatron’s base partitions and fuses LoRA’s all-gather with the base model’s final all-reduce.
  - Why it matters: keeps LoRA’s extra communication limited to small `r`-dimension tensors, making the added cost negligible and enabling multi-GPU scaling.

- Decouple-and-batch design choice: compute `xAB` on-the-fly (Section 4.1)
  - What’s new vs common LoRA serving: do not merge adapters into base weights when serving many adapters.
  - Why it matters: preserves batching across adapters for the expensive `xW` path and avoids maintaining many merged full models; experiments show it is better once more than one or two adapters need to be served concurrently (Figure 9).

- Scheduling for multi-tenant latency: early-abort admission control (Section 4.3; Appendix B; Figure 10)
  - What’s new: a principled rule grounded in a simple optimality result to drop requests unlikely to meet the first-token SLO, improving SLO attainment and user satisfaction under overload.

## 5. Experimental Analysis
Evaluation methodology
- Models and adapters
  - LLaMA series: 7B, 13B, 30B, 70B with various adapter rank sets (Table 1).
- Hardware
  - Single GPUs: A10G 24GB; A100 40GB and 80GB. Multi-GPU: 2× and 4× A100s (Section 7.1).
- Baselines and S-LoRA variants (Section 7.1)
  - `vLLM m-packed`: run `m` separate processes, each serving a merged adapter; strong system baseline but no native LoRA support.
  - `HuggingFace PEFT`: swaps adapters between batches; no continuous batching or KV paging.
  - Ablations: `S-LoRA-no-unify-mem` (no unified pool), `S-LoRA-bmm` (no unified pool and use padded batched GEMM), and an alternative “merge-then-serve” variant used in ablations (Figure 9).
- Workloads and metrics
  - Synthetic arrivals using a Gamma process with power-law adapter popularity; variable request rates and coefficients of variation (Table 2; Section 7.2).
  - Real traffic sampled from LMSYS Chatbot Arena logs (Section 7.3).
  - Metrics: throughput (req/s), average request latency, first-token latency, and SLO attainment defined as “percentage of requests that return the first token in 6 seconds” (Section 7.1).

Main quantitative results

- Throughput and scalability to thousands of adapters
  - Table 3 shows that with LLaMA-7B on A100 80GB (`S1` setting):
    - For `n=5` adapters: S-LoRA 8.05 req/s vs vLLM-packed 2.04 vs PEFT 0.88.
    - For `n=1000` adapters: S-LoRA 7.64 req/s; vLLM-packed runs out of memory (OOM).
    - For `n=2000` adapters: S-LoRA 7.61 req/s; vLLM-packed OOM.
  - Quote: “S-LoRA can serve 2,000 adapters simultaneously, maintaining minimal overhead for the added LoRA computation” (Section 7.2; Table 3; discussion following Table 3).

- Benefit of unified memory and custom kernels
  - Figure 5 compares S-LoRA vs its variants as number of adapters grows. S-LoRA sustains higher throughput and lower average latency than `S-LoRA-bmm` and `S-LoRA-no-unify-mem` across all model/hardware settings. Throughput initially dips slightly as adapters increase, then stabilizes once the number of active adapters per running batch saturates GPU memory; beyond that, adding more total adapters does not reduce throughput (Figure 5; Section 7.2).

- Request rate sensitivity
  - Figure 6 shows throughput, first-token latency, and SLO attainment vs request rate. S-LoRA retains high attainment and low first-token latency at higher rates than its ablations; `S-LoRA-bmm` often has first-token latency off the chart due to padding overheads (Figure 6; Section 7.2).

- Real workload performance
  - Figure 7 shows similar patterns on downsampled Arena logs: S-LoRA maintains higher throughput and better SLO attainment than its variants as the request rate increases (Section 7.3).

- Multi-GPU scaling and LoRA communication overhead
  - Figure 8 compares throughput for S-LoRA with and without the extra LoRA communication and for the base-only case:
    - The gap between “S-LoRA” and “S-LoRA (w/o LoRA communication)” is small, confirming negligible extra communication.
    - Going from 2× to 4× A100s increases throughput by more than 2× in these settings, attributed to alleviating the memory-bound regime (Section 7.4).

- Design ablations
  - Merge vs on-the-fly compute (Figure 9): with only one adapter, merging wins because it pays a one-time merge cost and then runs like the base model. Once more than two adapters are concurrent, on-the-fly compute surpasses merging due to frequent adapter switches causing GPU under-utilization; skewed adapter popularity (smaller `α`) further hurts merging because batch sizes shrink (Figure 9; Section 7.5).
  - Early-abort admission control (Figure 10): improves SLO attainment and user satisfaction over FCFS and LCFS, especially when arrival variability (`cv`) is large. FCFS tends to keep old requests that already miss SLO; LCFS works only at low variability (Section 7.5; Appendix B).
  - Adapter clustering (Appendix A.2; Figures 11–12): modest but consistent gains in throughput and SLO attainment as the number of allowed clusters decreases, with diminishing returns and small fluctuations likely from scheduler overheads.

- Summary statistic from abstract reinforced by data
  - Quote: “Compared to state-of-the-art libraries such as HuggingFace PEFT and vLLM (with naive support of LoRA serving), S-LoRA can improve the throughput by up to 4 times and increase the number of served adapters by several orders of magnitude” (Abstract; supported by Table 3 and Figures 5–7).

Assessment
- The experiments are extensive: multiple models (7B–70B), single and multi-GPU, synthetic and real traces, baselines and internal ablations, and queueing policies. The core claims—scaling to thousands of adapters, higher throughput than vLLM-packed and PEFT, minimal communication overhead—are supported by specific measurements (Table 3; Figures 5–8; Section 7).

## 6. Limitations and Trade-offs
- Compute overhead of on-the-fly LoRA
  - Computing `xAB` for every step introduces extra FLOPs versus merging. It pays off when many adapters are active, but for a single adapter or very small concurrency, merging can be faster (Figure 9; Section 7.5).
- Assumption that adapter rank `r` is small
  - Communication and compute analyses rely on `r << h`. Very high-rank adapters would reduce the advantage (Section 6.2).
- Prediction for prefetching
  - Prefetching adapters uses a heuristic prediction from the waiting queue; mispredictions could add I/O stalls, though overlapping mitigates this (Section 5.2).
- Fairness and latency trade-offs
  - Adapter clustering improves throughput but can hurt per-adapter fairness or tail latency; the paper notes this trade-off (Section 4.2; Appendix A.2).
  - Early-abort boosts SLO attainment but intentionally drops some requests under overload; systems may need policies to balance fairness and business logic (Section 4.3; Appendix B).
- Scope of system
  - Focused on single-machine, multi-GPU serving. Cross-machine scaling or distributed paging across hosts is not addressed.
  - Implemented for LoRA; while the authors argue techniques generalize to other parameter-efficient methods, those are not experimentally validated here (Related Work, Section 8).
- Quantization and sparsity
  - Orthogonal accelerations (e.g., quantization, sparsity) are not integrated; combining them with Unified Paging and custom kernels could introduce new engineering trade-offs (Section 8).

## 7. Implications and Future Directions
- Practical impact
  - Multi-tenant personalization at scale: a single GPU server can host thousands of adapters, enabling per-user or per-task specializations without duplicating the base model. This lowers cost and operational complexity for fine-tuning-as-a-service.
  - Platform design: the unified pool concept suggests treating all dynamic tensors (KV caches, adapter weights, perhaps prompts or routing buffers) uniformly to fight fragmentation and maximize batch size.
- Research directions
  - Extending Unified Paging: incorporate other dynamic state (e.g., prefix caches, routing states) and integrate with quantization-aware paging to further increase capacity.
  - Broader adapter families: implement kernels and paging for prefix/prompt-tuning, IA^3, or mixture-of-adapters; measure trade-offs across methods under multi-tenant loads (Section 9 mentions “support for additional adapter methods”).
  - Overlapping compute streams: run base and LoRA computations in parallel streams where safe; the paper lists this as future work (Conclusion).
  - Cross-node serving: adapt S-LoRA’s paging and LoRA-aware parallelism to multi-node clusters, exploring communication/computation overlap at rack scale.
  - Scheduling policies: more principled fairness-aware or cost-aware admission and clustering, possibly with learning-based predictors for adapter demand.
- Field-level shift
  - Quote: “S-LoRA enables scalable serving of many task-specific fine-tuned models and offers the potential for large-scale customized fine-tuning services” (Abstract). By making multi-adapter serving efficient, it moves the bottleneck from memory duplication to scheduling and kernel optimization, opening a path to “adapter marketplaces” and highly personalized LLM deployments.

> Selected citations to ground key points:
> - Decouple compute and batch base model vs adapters: Section 4.1; Figure 1; Eq. (1–2).
> - Unified Paging design: Section 5.1; Figure 3; Figure 2 for memory layout.
> - Custom kernels MBGMM/MBGMV: Section 5.3.
> - Tensor parallel strategy and cost: Section 6; Figure 4; Section 6.2 equations.
> - Throughput comparisons and scalability: Table 3; Figures 5–7.
> - Multi-GPU overhead and scaling: Figure 8.
> - Merge vs on-the-fly ablation: Figure 9.
> - Admission control and advantage: Section 4.3; Figure 10; Appendix B (Theorem B.1).
> - Adapter clustering effects: Section 4.2; Appendix A.2 (Figures 11–12).
