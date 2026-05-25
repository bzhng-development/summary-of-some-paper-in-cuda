# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

**ArXiv:** [2303.06865](https://arxiv.org/abs/2303.06865)

## 🎯 Pitch

FlexGen enables high-throughput inference of massive language models using just a single commodity GPU by efficiently orchestrating GPU, CPU, and disk resources via an optimized offloading schedule and advanced 4-bit compression. This breakthrough system dramatically lowers the cost and hardware requirements for running models like OPT-175B—achieving up to 100× greater throughput than previous offloading solutions—thereby democratizing access to powerful LLM capabilities for research and industry without the need for expensive multi-GPU clusters.

---

## 1. Executive Summary (2-3 sentences)
FlexGen is a system for running very large language models (LLMs) on a single commodity GPU by orchestrating GPU, CPU, and disk as a unified memory-compute hierarchy. It introduces a near-optimal offloading schedule, a cost-model-driven policy search, and 4‑bit compression for both model weights and attention caches, achieving up to 100× higher throughput than prior offloading systems on OPT‑175B using one NVIDIA T4 GPU (Figure 1, Table 2).

## 2. Context and Motivation
- Problem addressed:
  - LLM inference for very large models (e.g., 175B parameters) does not fit into a single GPU’s memory and is often run on expensive multi-GPU clusters using complex parallelism.
  - Many real workloads (benchmarking, information extraction, form processing) are latency-insensitive but throughput-oriented (process many prompts, generate many tokens) (Introduction).
  - The core challenge is how to execute large-batch, high-throughput generation using limited GPU memory by “offloading” data to CPU RAM and SSD without unbearable I/O and scheduling overhead.

- Why it matters:
  - Cost and accessibility: Enabling high-throughput LLM inference on a single 16 GB GPU democratizes usage for organizations without large clusters (Introduction; Figure 1).
  - New bottlenecks: With large batches, the attention key-value (`KV`) cache can exceed the size of weights (e.g., 1.2 TB cache vs. 325 GB weights for OPT‑175B with batch 512, input 512, output 32; Section 3, Memory Analysis).

- Prior approaches and gaps:
  - Model compression and quantization reduce memory but often assume the model still fits on GPU and do not address 175B-scale models on 1 GPU (Related Work).
  - Collaborative inference (e.g., Petals) trades computation for network communication and depends on network latency/bandwidth; per-GPU throughput suffers under realistic networks (Section 6.3; Figure 4).
  - Existing offloading systems (DeepSpeed ZeRO-Inference, Hugging Face Accelerate) inherit row-by-row schedules from training and place most intermediate data on GPU; they incur repeated weight I/O and run out of memory when batch size grows (Section 4.2; Figure 3a). As a result, they are constrained to tiny batches (1–2 for OPT‑175B) and achieve very low throughput (Figure 1, Table 2).

- Positioning:
  - FlexGen targets throughput-oriented generative inference with limited resources by:
    - Designing an I/O-efficient compute schedule and placement strategy that reuses weights across many queries (Section 4).
    - Searching placements over GPU/CPU/disk via a linear programming policy search guided by a calibrated cost model (Section 4.3, Eq. (1)).
    - Compressing both weights and `KV` cache to 4 bits with negligible accuracy loss to unlock CPU-only storage and avoid disk I/O (Section 5; Table 5).

## 3. Technical Approach
FlexGen treats LLM inference as a graph traversal with explicit data placement and I/O scheduling across GPU, CPU, and disk, then optimizes for throughput under memory and bandwidth constraints.

- Core notions (defined on first use):
  - `Offloading`: Storing some tensors (weights, activations, `KV` cache) outside GPU memory (in CPU RAM or SSD) and moving them to GPU when needed.
  - `KV cache`: The per-layer key/value tensors saved during the prefill stage and reused during decoding to avoid recomputing attention over the entire prefix.
  - `Prefill` vs. `Decoding`: Prefill processes the input prompt, creates the `KV` cache for each layer; Decoding generates tokens one by one using the cached keys/values (Section 3).
  - `Effective batch size` (or `block size`): The number of sequences processed together across a block of layers/tokens under FlexGen’s schedule. It equals the product of the per-GPU batch size and the number of GPU batches in a block (Section 4.2).

- Step 1 — Formulate inference with offloading as graph traversal (Section 4.1; Figure 2):
  - Represent computation as a grid: rows are layers, columns are generated tokens per sequence, and “squares” are per-layer computations for a GPU batch.
  - A valid execution path must:
    - Respect left-to-right dependencies within each row (token order).
    - Ensure inputs (weights, activations, `KV` cache) for a square are co-located on the target compute device.
    - Keep activations until the next layer consumes them and keep `KV` cache until the row completes.
    - Respect device memory capacity at all times.
  - Objective: Find a traversal and placement that minimizes total time (compute + I/O).

- Step 2 — Choose a compute schedule that minimizes weight I/O while respecting memory (Section 4.2):
  - Why the default “row-by-row” is inefficient (Figure 3a):
    - Adjacent squares (same batch across successive layers and tokens) do not share weights, so weights are repeatedly loaded, causing huge I/O.
  - FlexGen’s “zig-zag block schedule” (Figure 3b; Algorithm 1):
    - Traverse columns of the grid in blocks to reuse the same layer’s weights across many batches/tokens before evicting them.
    - While computing a given layer for one GPU batch, overlap four I/O streams plus compute (Algorithm 1):
      - Load next layer’s weights.
      - Store previous batch’s activations/`KV` cache.
      - Load next batch’s activations/`KV` cache.
      - Compute current batch.
    - This exposes two tunable parameters: `GPU batch size` and `# GPU batches per block`; their product is the `effective batch size`.
  - Near-optimality guarantee (Appendix A.2; Theorem 4.1):
    - The zig-zag block schedule’s I/O is within 2× of an I/O-optimal (but hard-to-implement) “diagonal block schedule,” proven by bounding weight loads per unit of progress while respecting memory.

- Step 3 — Tensor placement at different granularities (Section 4.2):
  - Weights: placed at layer granularity to reduce overhead while allowing flexibility.
  - Activations and `KV` cache: placed at tensor granularity for fine control.
  - Placements are parameterized as percentages across GPU/CPU/disk:
    - Weights: `wg`, `wc`, `wd`.
    - Activations: `hg`, `hc`, `hd`.
    - KV cache: `cg`, `cc`, `cd`.

- Step 4 — Delegate some computation to CPU when I/O dominates (Section 4.2, “Computation delegation”):
  - Observation: If `KV` cache lives on CPU, moving the whole cache to GPU per decoding step is very expensive.
  - Strategy: Compute attention scores on CPU to avoid moving the cache:
    - Move only the current token’s activations `b × h1 × 4` bytes instead of the full cache `b × s × h1 × 4` bytes; this reduces I/O by a factor of the prompt length `s` (often ≥512).
  - When quantization is enabled (Section 5), CPU delegation is disabled because fine-grained de/quantization overhead on CPU outweighs the I/O savings.

- Step 5 — Predict latency with an analytical cost model and search policies via linear programming (Section 4.3; Appendix A.3):
  - Latency decomposition:
    - `T = Tpre * l + Tgen * (n - 1) * l` where `l` = #layers, `n` = #generated tokens.
    - `Tpre` (prefill) and `Tgen` (per-token decoding) each take the max over overlapped components: CPU↔GPU reads/writes, disk↔CPU reads/writes, and compute (matrix multiplies and batched GEMMs).
    - Example term: `dtocg = (wd * weight_bytes + cd * KV_bytes + hd * activation_bytes) / (disk_to_cpu_bandwidth)` (Appendix A.3).
  - Memory constraints:
    - Peak memory on GPU/CPU/disk is computed for both prefill and decoding, considering “home” tensors (the fixed placement percentages) and “working” buffers for current ops (Appendix A.3).
  - Policy search:
    - Enumerate a small set of tuples `(GPU batch size, #GPU batches per block)`.
    - For each tuple, solve a linear program to select `wg, wc, wd, cg, cc, cd, hg, hc, hd` minimizing `T / block_size` subject to memory constraints and `wg+wc+wd=1`, etc. (Eq. (1)).
    - In practice, due to modeling approximations (fragmentation, OS page cache), a small amount of manual adjustment may be needed (Section 4.3).

- Step 6 — Multi-GPU extension via pipeline parallelism (Section 4.4):
  - Split layers equally across GPUs, apply the same single-GPU policy on each stage, and pipeline micro-batches.
  - This reduces per-GPU memory pressure and can yield super-linear gains in decoding throughput by allowing larger batches and avoiding disk I/O.

- Step 7 — Approximate methods to further increase throughput (Section 5):
  - 4‑bit group-wise quantization (weights and `KV` cache) without retraining or calibration (Section 5; Table 5):
    - For each group of `g=64` contiguous elements along a chosen dimension, store min/max and quantize to 4 bits (`round((x - min)/(max - min) * (2^b - 1))`).
    - Weights grouped along output channels; `KV` cache grouped along hidden dimension; dequantize back to FP16 before compute.
    - This reduces memory and I/O enough to keep everything on CPU, eliminating slow disk access for 175B models.
  - Sparse attention during decoding (Section 5):
    - After computing attention scores, keep only the Top‑K keys per query and load the corresponding subset of `V` from storage; experiments use 10% sparsity.

## 4. Key Insights and Innovations
- An offloading schedule with provable near-optimal I/O efficiency (Section 4.2; Theorem 4.1):
  - Instead of row-by-row traversal, FlexGen’s zig‑zag block schedule reuses the same layer’s weights across many batches/tokens before evicting. This reduces repeated weight loads, the dominant I/O in offloaded settings.
  - The schedule is formally analyzed via a graph traversal model and proven within 2× of the optimal I/O complexity (Appendix A.2).

- Unified placement and compute delegation across GPU/CPU/disk (Section 4.2):
  - FlexGen jointly places weights, activations, and `KV` cache and can offload all of them (not just weights) out of GPU to unlock much larger effective batch sizes.
  - It selectively runs decoding attention on CPU when `KV` cache is off-GPU, cutting I/O by ~`s×` for long prompts.

- Cost-model-driven policy search via linear programming (Section 4.3; Eq. (1), Appendix A.3):
  - A compact set of decision variables (`wg,wc,wd,cg,cc,cd,hg,hc,hd` plus batch/block sizes) are optimized using a calibrated cost model.
  - This makes FlexGen portable across hardware (different bandwidths/memory sizes) and tunable under latency/throughput constraints.

- 4‑bit compression of both weights and `KV` cache with negligible accuracy loss (Section 5; Table 5):
  - Prior work typically compresses only weights or uses 8‑bit activations; FlexGen shows both weights and `KV` cache can be stored in 4‑bit group-wise format without retraining, directly reducing offloading I/O.
  - This enables CPU-only storage for 175B models and removes SSD from the critical path, yielding the largest throughput gains (Table 2; Figure 1).

These are largely fundamental innovations in system design for offloaded inference, not merely incremental optimizations.

## 5. Experimental Analysis
- Evaluation setup (Section 6):
  - Hardware (Table 1): single NVIDIA T4 (16 GB), CPU with 208 GB RAM, 1.5 TB NVMe SSD; SSD read ~2 GB/s, write ~1 GB/s.
  - Models: OPT family from 6.7B to 175B parameters.
  - Workload: synthetic datasets with fixed prompt lengths (512 and 1024 tokens) and output length 32 unless stated; metric is “generation throughput” = generated tokens / (prefill + decoding time).
  - Baselines: DeepSpeed ZeRO-Inference (offloading), Hugging Face Accelerate (offloading), and Petals (collaborative inference; per-GPU throughput under constrained networks).

- Main results on one GPU (Table 2; Figure 1):
  - With prompt length 512:
    - OPT‑30B: FlexGen 7.32 tok/s vs. Accelerate 0.62 and DeepSpeed 0.60 (≈12× gain).
    - OPT‑175B: FlexGen 0.69 tok/s vs. Accelerate 0.01 and DeepSpeed 0.01 (≈69× gain).
    - With 4‑bit compression: OPT‑175B reaches 1.12 tok/s (≈112× vs. baselines).
  - With prompt length 1024:
    - OPT‑175B: FlexGen 0.35 tok/s vs. Accelerate 0.01; with 4‑bit compression: 0.42 tok/s.
  - Latency–throughput trade-off (Figure 1; Tables 19–20):
    - At 5,000 s latency, FlexGen achieves >40× higher throughput than baselines by using `effective batch size = 64` vs. baseline batch ≤2.
    - Maximum throughput rises to 69× without compression and 100× with compression (`effective batch size = 144`) when allowing higher latency (e.g., 4,000–12,000 s windows for 175B; Table 19).

- Multi-GPU scaling via pipeline parallelism (Table 3):
  - With 4 T4 GPUs:
    - OPT‑175B decoding throughput improves from 0.83 tok/s (1 GPU) to 3.86 tok/s (≈4.7×, super-linear) due to larger batches and moving from disk+CPU to CPU-only offloading.
    - Overall generation throughput grows from 0.69 tok/s to 2.33 tok/s but has prefill “pipeline bubbles,” so decoding scales better than end-to-end throughput, as expected for short outputs (32 tokens).

- Runtime breakdown and bottlenecks (Table 8):
  - For OPT‑175B (prompt 512):
    - Prefill total 2711 s: compute 2220 s, weights read 768 s, cache write 261 s.
    - Decoding total 11315 s: compute 1498 s, weights read 3047 s, cache read 7046 s, cache write 124 s.
  - Interpretation: Decoding is I/O-bound (GPU compute utilization only 13%), justifying CPU attention delegation and `KV` cache compression.

- Ablations (Table 4; Appendix Tables 21–23):
  - Removing overlapping: 30B drops from 7.32 → 5.86 tok/s; 175B 0.69 → 0.59.
  - Disabling CPU compute: 30B 7.32 → 4.03 tok/s (large drop); 175B 0.69 → 0.62 (smaller drop because SSD I/O dominates for 175B without compression).
  - Using DeepSpeed-style policy within FlexGen (row-by-row; cache on GPU): 30B 1.57 tok/s; 175B 0.01 tok/s — confirms prior policies are suboptimal for inference.
  - Policy details and latency corroborate that large effective batch sizes drive throughput, constrained primarily by `KV` cache memory (Appendix Tables 21–22).

- Accuracy with approximations (Table 5):
  - 4‑bit quantization (weights + `KV` cache) and 4‑bit + 10% sparse attention:
    - OPT‑30B Lambada: 0.725 (FP16) vs. 0.724 (4‑bit) vs. 0.718 (4‑bit‑S).
    - OPT‑175B Wikitext perplexity: 10.82 (FP16) vs. 10.94 (4‑bit) vs. 10.94 (4‑bit‑S).
  - Takeaway: Negligible quality loss under tested settings.

- Offloading vs. collaborative inference (Figure 4; Table 2):
  - Across “good” (10 ms, 1 Gbps) and “slow” (100 ms, 0.1 Gbps) networks, FlexGen’s per-GPU throughput exceeds Petals for OPT‑30B.
  - In slow networks and short generations, FlexGen can even have lower total latency due to Petals’ communication overhead during prefill.

- Real workloads:
  - HELM integration (Table 9): Runs 7 sub-scenarios with OPT‑30B in 21 hours on a single 16 GB GPU, including loading and metric computation.
  - Data wrangling (Tables 10–11): Reports end-to-end throughput on entity matching, deduplication, and error detection tasks.
  - Mixed sequence lengths (Table 25): With simple padding, “actual throughput” is 75–79% of “padded throughput,” highlighting a batching efficiency consideration.

- Sensitivity to SSD bandwidth (Table 24):
  - With slower SSD (0.5 GB/s vs. 1.6 GB/s reads), OPT‑175B throughput drops from 0.69 to 0.30 tok/s, confirming SSD bandwidth is critical when disk offloading remains in the loop.

Assessment: The experiments are broad (single and multi-GPU, multiple models and prompt lengths, other systems, accuracy, ablations, SSD speed, real tasks), and the numbers consistently support the paper’s claims: the new schedule + placement + compression unlock large effective batches and greatly higher throughput under realistic single-GPU constraints.

## 6. Limitations and Trade-offs
- Throughput over latency:
  - FlexGen explicitly targets latency-insensitive workloads. Latency for a large block can be thousands of seconds (Figure 1; Tables 19–20). It is not optimized for interactive chat.

- Dependence on host memory and SSD speed:
  - High CPU RAM (208 GB in Table 1) is essential to hold large `KV` caches and (with 4‑bit) all weights. Slower SSDs severely hurt performance when disk offloading is required (Table 24).

- Decoding remains I/O-bound without compression:
  - Even with CPU attention delegation, decoding time is dominated by `KV` cache movement (Table 8). Without 4‑bit compression, SSD I/O limits maximum throughput for 175B.

- Cost-model approximations and manual tuning:
  - The LP uses relaxed percentages and approximated peak memory; fragmentation and OS page cache can cause deviations, sometimes requiring manual policy tweaks (Section 4.3).

- Quantization runtime trade-offs:
  - Fine-grained (group-wise) 4‑bit quantization introduces (de)quantization overhead; to avoid making CPU a bottleneck, CPU delegation is disabled when quantization is on (Section 5).

- Not all optimal schedules are implemented:
  - The theoretically I/O-optimal “diagonal block schedule” is not implemented due to engineering complexity with non-contiguous `KV` memory; the implemented zig‑zag schedule is within 2× of optimal but may leave performance on the table (Appendix A.2).

- Batching variable-length sequences:
  - Current batching uses padding to the max sequence length, which can waste compute for highly skewed length distributions (Table 25). More advanced bucketing/packing could improve efficiency.

## 7. Implications and Future Directions
- How this changes the landscape:
  - FlexGen makes it practical to run 30B–175B LLMs for high-throughput batch workloads on a single commodity GPU by leveraging the full memory hierarchy, principled scheduling, and compression.
  - It reframes the “can’t fit on one GPU” narrative: with the right offloading policy, much larger models become accessible for offline processing.

- Enabled follow-ups:
  - Implement the diagonal block schedule with efficient attention over non-contiguous caches to close the remaining ≤2× I/O gap (Appendix A.2).
  - Tighter integration with OS and storage (e.g., predictive prefetching, pinned page cache control) and unified memory hardware to smooth I/O jitter.
  - Faster de/quantization kernels and mixed-precision integer compute to keep quantized tensors in low-precision through more of the pipeline.
  - Advanced batching for variable-length prompts (bucketing, packing) and adaptive block sizing to further increase utilization (Table 25 shows room).

- Practical applications:
  - Large-scale evaluation and benchmarking (e.g., HELM; Table 9).
  - Document processing, data wrangling, and information extraction across large corpora (Tables 10–11).
  - Cost-effective deployment of very large models for back-office workloads, where throughput and cost per token matter more than single-query latency.

> Representative headline result: “On a single T4 GPU with 208 GB CPU RAM and 1.5 TB SSD, FlexGen achieves 0.69 tok/s on OPT‑175B with prompt 512 and output 32—≈69× higher than state-of-the-art offloading systems—and 1.12 tok/s with 4‑bit compression by keeping all weights and caches in CPU memory” (Table 2; Figure 1).
