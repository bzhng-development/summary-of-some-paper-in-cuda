# FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

**ArXiv:** [2303.06865](https://arxiv.org/abs/2303.06865)

## 🎯 Pitch

FlexGen introduces a breakthrough engine that enables high-throughput inference for massive language models—like OPT-175B—on a single commodity GPU. By intelligently coordinating memory and computation across GPU, CPU, and disk, and compressing both model weights and the attention cache, FlexGen achieves up to 100× greater throughput than prior offloading systems. This innovation dramatically expands access to state-of-the-art LLM capabilities for throughput-driven, latency-insensitive tasks, making large-scale language models far more practical and affordable outside specialized hardware environments.

---

## 1. Executive Summary (2-3 sentences)
FlexGen is a generation engine that runs very large language models (LLMs) on a single commodity GPU by jointly scheduling memory and compute across GPU, CPU, and SSD, and by compressing both model weights and the attention key–value (KV) cache. It targets throughput-oriented workloads (large batches, relaxed latency) and achieves a new Pareto frontier: for OPT-175B on a 16GB NVIDIA T4, it reaches up to 1.12 tokens/s with 4-bit compression and delivers up to 100× higher maximum throughput than prior offloading systems (Fig. 1; Table 2).

## 2. Context and Motivation
- Problem addressed
  - Large LLMs (tens to hundreds of billions of parameters) exceed the memory of commodity GPUs for inference. For example, OPT-175B needs 325 GB just for weights, and at large batch sizes the KV cache can exceed a terabyte (Section 3: Memory Analysis).
  - Existing offloading-based systems (that spill tensors to CPU or disk) run, but their throughput is poor on a single GPU because of suboptimal I/O scheduling and small feasible batch sizes (Introduction; Fig. 1).

- Why it matters
  - Many practical uses of LLMs are throughput-oriented and latency-insensitive: benchmarking, information extraction, data wrangling, and form processing (Introduction). Being able to run very large models on widely available hardware broadens access and lowers cost for these “back-of-house” tasks.

- Prior approaches and their shortcomings
  - Model compression (quantization/pruning) reduces memory but often assumes the model still fits on GPU, and does not, by itself, enable 175B-scale models on a single GPU (Related Work).
  - Collaborative/decentralized inference (e.g., Petals) spreads compute across volunteer GPUs, but per-GPU throughput is limited by network bandwidth/latency and pipeline communication (Section 6.3; Fig. 4).
  - Training-inspired offloading systems (DeepSpeed ZeRO-Inference; Hugging Face Accelerate) traverse computation “row-by-row” (by batch then token) and keep cache/activations on GPU, causing small feasible batch sizes and repeated weight loads (Section 4.2; Fig. 3a).

- Positioning
  - FlexGen targets high-throughput inference on a single (or few) commodity GPU(s) by:
    - Redesigning the compute schedule to reuse weights across many inputs.
    - Searching over where to place weights, activations, and KV cache across GPU/CPU/SSD under capacity constraints.
    - Compressing both weights and KV cache to 4 bits to eliminate disk I/O in many cases.
    - Delegating some attention computation to CPU when it reduces I/O (Sections 4–5).

## 3. Technical Approach
FlexGen is a full-stack approach covering computation scheduling, tensor placement, compression, and policy search under hardware constraints.

- Background: how LLM generation uses memory (Section 3)
  - Generative inference has two phases:
    - Prefill: process the prompt to build the KV cache per layer (a memory of past keys/values used by attention).
    - Decode: generate tokens sequentially; each step reads and updates the KV cache.
  - Memory drivers:
    - Weights (fixed per layer).
    - KV cache (grows with batch size `b`, prompt length `s`, and generated tokens `n`): for OPT-175B, with `b=512, s=512, n=32` the KV cache reaches ~1.2 TB—3.8× the weight size (Section 3).

- Core scheduling idea: compute order that minimizes I/O (Section 4.2; Fig. 3)
  - Problem with row-by-row (Fig. 3a): it minimizes latency for a single batch but forces frequent weight loads because adjacent squares (layer computations) do not share weights.
  - FlexGen adopts a zig-zag “block schedule” (Fig. 3b):
    - Process multiple GPU batches as a block across tokens and layers so that loaded weights are reused across many inputs before eviction.
    - “Effective batch size” = GPU batch size × number of GPU batches in a block (introduced in Algorithm 1 and Section 4.2). Larger effective batch sizes amortize I/O costs across more tokens.

- Overlapping of I/O and compute (Algorithm 1)
  - Within the innermost loop, FlexGen pipelines:
    - Loading weights of the next layer.
    - Loading cache/activations of the next batch.
    - Storing cache/activations of the previous batch.
    - Computing the current batch.
  - These are launched as logically parallel streams (GPU streams + CPU threads), then synchronized each iteration (Algorithm 1).

- Tensor placement across the memory hierarchy (Section 4.2: “Tensor placement”)
  - Percentages indicate where each tensor type is resident:
    - Weights: `wg, wc, wd` on GPU/CPU/disk.
    - Activations: `hg, hc, hd`.
    - KV cache: `cg, cc, cd`.
  - Granularity choices:
    - Weights at layer granularity (low overhead, enough flexibility).
    - Activations and KV cache at tensor granularity.

- CPU delegation for attention scores (Section 4.2: “Computation delegation”)
  - When KV cache is on CPU, computing attention scores on GPU requires moving the entire KV cache to GPU.
  - Compute attention scores on CPU instead: move only the current activations from GPU to CPU.
  - I/O saved scales with prompt length `s`: moving KV is `b×s×h1×4` bytes vs. moving activations `b×h1×4` bytes (so s× reduction). This is particularly beneficial when `s ≥ 512`.

- Analytical cost model + linear programming (Section 4.3; Eq. (1))
  - Latency per layer for prefill (`Tpre`) and decoding (`Tgen`) is modeled as the maximum of overlapped terms:
    - Disk→CPU reads, CPU→Disk writes, CPU→GPU reads, GPU→CPU writes, and compute time (formulas under “Cost Model”).
  - Tensor sizes:
    - Per-layer weights: `8h1^2 + 4h1·h2` bytes (FP16).
    - Activations for a block: `2·bls·s·h1` (prefill) or `2·bls·h1` (decode).
    - Average KV cache per layer in a block: `4·bls·(s + n/2)·h1`.
  - Objective and constraints:
    - Objective: minimize `T / bls` (equivalently maximize throughput per token).
    - Subject to GPU/CPU/SSD peak-memory constraints and percentage-sum constraints (Eq. (1)).
  - Search procedure:
    - Enumerate `bls` (block size) and `gbs` (GPU batch size) from a small discrete set.
    - Solve a linear program for the percentages (`wg,wc,wd`, `hg,hc,hd`, `cg,cc,cd`).
    - Minor manual adjustments may be needed due to unmodeled fragmentation (Section 4.3).

- Near-optimality guarantee for the schedule (Appendix A.2; Theorem 4.1)
  - A “diagonal block schedule” is introduced in the appendix and shown to be I/O-optimal asymptotically.
  - The implemented zig-zag block schedule is proven to be within 2× I/O-optimal (Theorem 4.1). Intuition:
    - Weight I/O dominates; any schedule must reload weights a certain number of times under memory limits.
    - Zig-zag block schedule’s per-reload peak memory is not perfectly balanced across tokens, causing at most a factor-2 gap vs. the diagonal schedule (Appendix A.2).

- Compression and approximation (Section 5)
  - 4-bit group-wise quantization for both weights and KV cache without retraining or calibration:
    - Groups of 64 contiguous elements are quantized with per-group min/max (Shen et al., 2020).
    - Dequantization happens before matmul; goal is I/O/memory reduction rather than integer matmul speedups.
    - Group dimensions chosen for stability and efficiency: weights grouped along output channels; KV cache grouped along hidden dimension.
  - Sparse attention option:
    - After attention scores, load only the Top-K values (10% of `V`) per query (Section 5 “Sparse Attention”). Used primarily to reduce I/O during offloading.

- Multi-GPU extension (Section 4.4)
  - Pipeline parallelism partitions the layers across GPUs, reducing per-GPU memory pressure.
  - This often enables larger batch sizes or avoids disk offload, yielding super-linear scaling in decoding throughput (Table 3), though prefill still suffers pipeline bubbles.

- Implementation notes (Section 6: “Hardware”, “Implementation”)
  - Built on PyTorch with multiple CUDA streams and CPU threads; tensors offloaded to SSD via memory-mapped files.
  - Target testbed: 1×NVIDIA T4 (16 GB), 208 GB CPU DRAM, 1.5 TB SSD (Table 1). SSD read ~2 GB/s, write ~1 GB/s.

## 4. Key Insights and Innovations
- A unified, search-based offloading strategy across weights, activations, and KV cache
  - What’s new: Prior systems focused on moving weights and left cache/activations on GPU, constraining batch size. FlexGen jointly places all tensor types across GPU/CPU/SSD and searches over this space with an analytical model and linear program (Section 4.3; Eq. (1)).
  - Why it matters: This unlocks large effective batch sizes (e.g., 144 for OPT-175B with compression), which are essential to amortize I/O and raise throughput (Fig. 1; Table 15–16).

- Near-optimal compute scheduling for offloaded inference
  - What’s new: A zig-zag block schedule with overlapping (Algorithm 1; Fig. 3b) that reuses weights and pipelines I/O with compute.
  - Why it matters: It guarantees I/O within 2× of an asymptotically optimal schedule (Theorem 4.1), making the design principled and not merely heuristic.

- 4-bit quantization of both weights and KV cache with negligible accuracy loss
  - What’s new: Compressing the KV cache to 4 bits in addition to weights, without extra calibration/training (Section 5; Table 5).
  - Why it matters: KV cache dominates memory at large batch sizes; compressing it can eliminate disk I/O and make CPU-only offloading feasible, raising throughput (e.g., 1.12 tokens/s on OPT-175B; Table 2).

- CPU delegation for attention scores to reduce I/O
  - What’s new: Opportunistically compute attention scores on CPU when KV cache lives on CPU, reducing movement by a factor of `s` (prompt length) (Section 4.2).
  - Why it matters: When `s ≥ 512`, cutting KV traffic drastically can make CPU-bound attention faster overall than GPU attention with massive transfers. Ablations show nontrivial gains (Table 4: OPT-30B drops from 7.32 to 4.03 tokens/s without CPU compute).

## 5. Experimental Analysis
- Evaluation setup (Section 6)
  - Hardware: 1×T4 (16 GB), 208 GB RAM, 1.5 TB SSD (Table 1).
  - Workloads: synthetic prompts with fixed lengths; default `s=512` or `s=1024`, generate `n=32` tokens (Section 6 “Workload”).
  - Baselines: DeepSpeed ZeRO-Inference and Hugging Face Accelerate (offloading), and Petals for decentralized inference (Sections 6.1, 6.3).
  - Metrics: generation throughput (tokens/s) = total generated tokens / (prefill + decode time). For Petals: per-GPU throughput under specified network settings (Table 2).

- Main results (single T4; Table 2; Fig. 1)
  - Throughput gains over offloading baselines:
    - OPT-30B @ s=512: FlexGen 7.32 vs Accelerate 0.62 and DeepSpeed 0.60 tokens/s.
    - OPT-175B @ s=512: FlexGen 0.69 vs both baselines 0.01 tokens/s.
    - With 4-bit compression: OPT-175B reaches 1.12 tokens/s.
  - Latency–throughput frontier (Fig. 1; Tables 19–20):
    - At ~5000 s latency, FlexGen achieves >40× higher throughput than DeepSpeed (narrated with numbers in Section 6.1).
    - With higher latency allowance and compression, FlexGen hits 100× higher maximum throughput (effective batch size 144; Fig. 1 left; Table 19).

- Scaling to 4 GPUs via pipeline parallelism (Table 3)
  - Decoding throughput scales super-linearly due to ability to avoid disk and increase batch size:
    - OPT-175B decoding: 0.83 (1 GPU) → 3.86 tokens/s (4 GPUs).
  - Overall generation throughput scales sublinearly due to prefill pipeline bubbles:
    - OPT-175B generation: 0.69 → 2.33 tokens/s.

- Ablations and diagnostics
  - Component contributions (Table 4):
    - Removing overlapping: 7.32 → 5.86 tokens/s (30B).
    - Removing CPU compute: 7.32 → 4.03 tokens/s (30B).
    - Using “DeepSpeed policy” (row-by-row with cache on GPU): 7.32 → 1.57 tokens/s (30B).
    - For 175B, reducing block size from 32×8 to 32×1 drops throughput from 0.69 to 0.27 tokens/s (“No policy search”).
  - Runtime breakdown (Table 8):
    - OPT-175B: compute dominates prefill (2220s of 2711s), but decode is I/O-heavy; GPU compute is only 1498s of 11315s on decode. The reported GPU utilization is 82% in prefill, 13% in decode (Section 6.1 “Runtime breakdown”).
  - SSD sensitivity (Table 24):
    - 175B throughput drops from 0.69 to 0.30 tokens/s when SSD read/write fall from ~1.6/1.3 GB/s to 0.5/0.5 GB/s (no OS cache).

- Accuracy with approximations (Table 5)
  - 4-bit quantization and 4-bit + sparse attention (10% of `V`) preserve accuracy:
    - OPT-175B Lambada accuracy: 0.758 (FP16) → 0.756 (4-bit) → 0.756 (4-bit-S).
    - OPT-175B WikiText perplexity: 10.82 → 10.94 → 10.94.

- Decentralized inference comparison (Section 6.3; Fig. 4)
  - Under varying network conditions, FlexGen on 1×T4 yields higher per-GPU throughput than a 4×T4 Petals cluster:
    > “the throughput of FlexGen with a single T4 outperforms the per-GPU throughput of the Petals cluster under all tested network conditions.” (Fig. 4)
  - In slow networks and short generations, FlexGen can even have lower end-to-end latency than Petals due to communication bottlenecks in prefill.

- Real tasks and mixed sequence lengths
  - HELM subset: FlexGen completes 7 sub-scenarios with a 30B model on a T4 in 21 hours, including I/O and metrics computation (Table 9).
  - Mixed-length batching via padding achieves 75–79% efficiency in two examples (Table 25).

- Overall assessment
  - The experiments consistently support the central claim: with careful scheduling, placement, and compression, single-GPU high-throughput inference for 30B–175B models is feasible and significantly faster than prior offloading systems. The analysis also candidly exposes bottlenecks (decode I/O, SSD speed, pipeline bubbles).

## 6. Limitations and Trade-offs
- Throughput over latency by design
  - Effective batch sizes are large, and latencies can be thousands of seconds per block (e.g., 1973s at effective batch 48; Table 19). This is unsuitable for interactive applications.

- Heavy dependence on storage bandwidth and CPU memory
  - SSD speed directly affects 175B throughput (Table 24). Slow disks or small CPU memory (forcing disk use) degrade performance sharply.

- Decode remains I/O-bound
  - Even with scheduling and CPU delegation, decode GPU utilization is low (13% in Table 8), meaning total throughput is capped by data motion.

- Quantization trade-offs
  - 4-bit group-wise quantization introduces compression/decompression overhead. The system disables CPU delegation when quantization is on because CPU-side (de)compression overhead can outweigh I/O savings (Section 5).

- Cost model approximations and tuning needs
  - The analytical model relaxes placement percentages to continuous variables and cannot perfectly predict fragmentation or OS-level caching; manual adjustments are sometimes required (Section 4.3).

- Model and hardware coverage
  - Results are reported on OPT models and primarily on T4 hardware. While the method is architecture-agnostic, performance depends on the specific GPU/CPU/SSD configuration (Appendix A.4 shows an RTX 3090 setup; Table 12).

- Mixed-length batching inefficiency
  - The default batching strategy pads to the maximum length, which can waste compute for skewed length distributions (Table 25).

## 7. Implications and Future Directions
- What this changes
  - FlexGen reframes “single-GPU giant LLM inference” from impossible to practical for offline, large-batch tasks. It shows that smart I/O scheduling and compression, not just more accelerators, can unlock substantial throughput gains.

- Practical applications now feasible
  - Organization-scale data labeling/extraction, document processing, and benchmarks (HELM-style runs) with 30B–175B models on commodity cloud instances.
  - Academic/SME environments without multi-GPU clusters can still run very large models for offline workloads.

- Research avenues
  - Implement the diagonal block schedule (Appendix A.2) with efficient attention on non-contiguous memory to narrow the remaining 2× gap.
  - Combine FlexGen with kernel-level advances (e.g., optimized attention and KV-cache layouts) to raise decode GPU utilization.
  - Smarter mixed-length batching (bucketing, dynamic partitioning) to reduce padding overhead.
  - Joint optimization of compression and scheduling: end-to-end cost models that include quantization (de)compression overhead on different devices.
  - Adaptive policies for heterogeneous or unified-memory systems and emerging storage-class memory.

- Broader ecosystem impact
  - Encourages system designs that treat GPU/CPU/SSD as a single orchestrated memory-compute fabric.
  - Provides a template (LP-based policy search plus near-optimal scheduling) that can be adapted to other sequence models or inference workloads beyond LLMs.

Overall, FlexGen advances the state of the art in offloaded LLM inference by combining theoretically grounded scheduling, principled placement search, and practical compression, yielding order-of-magnitude throughput improvements on accessible hardware (Fig. 1; Tables 2–4, 8).
