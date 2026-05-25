# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**ArXiv:** [2407.08608](https://arxiv.org/abs/2407.08608)

## 🎯 Pitch

FlashAttention-3 revolutionizes Transformer attention by redesigning the kernel to fully leverage asynchrony and low-precision computation on modern Hopper GPUs. By interleaving data movement, computation, and softmax through warp-specialized pipelines and introducing FP8-specific layout and quantization techniques, it achieves up to 2x speedup and dramatically boosts hardware utilization without sacrificing accuracy. This unlocks scalable, efficient long-context processing for large language models and other domains, removing a critical bottleneck for next-generation AI systems.

---

## 1. Executive Summary
FlashAttention-3 redesigns the attention kernel to exploit two Hopper H100 GPU features—true asynchrony between compute and memory engines, and FP8 low-precision Tensor Cores. It introduces warp-specialized pipelines that overlap data movement, matrix multiplications, and softmax, plus FP8-specific layout and quantization techniques, yielding 1.5–2.0× speedups over FlashAttention-2 in FP16 and up to ~1.2 PFLOPs/s in FP8, while improving FP8 accuracy by 2.6× over a baseline FP8 attention (see §4.1, Fig. 5–7; §4.3, Table 3).

## 2. Context and Motivation
- Problem addressed
  - Attention is the main computational bottleneck in Transformers due to quadratic scaling in sequence length. Even with the FlashAttention family (exact attention that avoids materializing large intermediates), FlashAttention-2 under-utilizes new GPUs: on H100, it reaches ~35% utilization vs 80–90% for optimized GEMM kernels (§1).
- Why it matters
  - Faster exact attention unlocks long-context training/inference across text, code, and multimodal domains (e.g., high-resolution image, audio, video) without sacrificing quality (§1). Architectures and libraries increasingly rely on exact attention primitives (e.g., Ring Attention, cuDNN).
- Shortcomings of prior work
  - FlashAttention-2 still assumes a largely synchronous model and does not exploit Hopper’s asynchronous compute/memory engines or FP8 Tensor Cores (§1, §2.2). Existing optimizations (e.g., Triton kernels, cuDNN) improve instruction choice but do not fully restructure the algorithm around asynchrony or FP8-specific constraints.
- Positioning
  - FlashAttention-3 is an exact-attention algorithm designed around Hopper’s execution model. It (1) uses warp specialization to overlap producers/consumers of data, (2) pipelines GEMMs with softmax within and across warpgroups, and (3) adds an FP8 path with accuracy-preserving quantization and layout handling (§3).

Key terms used in this review
- `warp`/`warpgroup`/`CTA`: GPU execution groupings (32 threads = warp; 4 contiguous warps = warpgroup; a cooperative thread array is a threadblock/CTA) (§2.2).
- `SMEM`/`GMEM`: on-chip shared memory vs off-chip global memory (HBM) (§2.2).
- `TMA` (Tensor Memory Accelerator): Hopper unit for asynchronous GMEM↔SMEM transfers (§2.2).
- `WGMMA`: Hopper’s warpgroup-level asynchronous matrix-multiply-accumulate on Tensor Cores (§2.2).
- `warp specialization`: dedicating some warps to moving data (producers) and others to compute (consumers), enabling overlap (§3.1).

## 3. Technical Approach
FlashAttention-3 keeps FlashAttention’s high-level idea—fusing attention to avoid writing large intermediates to HBM—but re-architects execution to exploit Hopper’s asynchrony and FP8 support. The methods below explain how the forward and backward passes are scheduled and what changes are needed for FP8.

A. Producer–consumer asynchrony with a circular shared-memory buffer (§3.1, Algorithm 1)
- Design
  - Split each CTA’s warps into producers and consumers. Producers issue asynchronous TMA loads of `Q_i`, `K_j`, `V_j` tiles from GMEM to SMEM; consumers perform compute on those SMEM tiles with WGMMA (asynchronous GEMMs).
  - Use an s-stage circular buffer in SMEM plus barriers/commits to coordinate when a stage is filled/consumed (Algorithm 1, lines 1, 7–10, 22).
  - Reallocate registers dynamically with Hopper’s `setmaxnreg`: consumer warps get more registers for GEMMs; producer warps use fewer (§3.1).
- Execution flow (one CTA processes a query tile `Q_i`)
  - Producer: load `Q_i` once (lines 4–5), then iterate over `K_j,V_j` tiles, prefetching next tiles while consumers work (lines 6–10).
  - Consumer: wait for data availability, then for each key/value block, perform:
    - GEMM1: compute scores block `S_i^(j) = Q_i K_j^T` (line 17),
    - local softmax update with numerically stable rescaling using per-row running max `m_i` and sum `ℓ_i` (lines 18–19),
    - GEMM2: multiply softmax-weighted block with `V_j` to update `O_i` (line 21).
  - After all blocks: finalize scaling `O_i = diag(ℓ_i)^{-1} O_i` and write out `O_i`, log-sum-exp `L_i` (lines 24–25).
- Why it works
  - As TMA and WGMMA are asynchronous, producers can keep SMEM staged while consumers compute; the circular buffer hides memory latency.

B. Ping–pong scheduling across warpgroups (§3.1, Fig. 1)
- Observation
  - On H100, special functions like `exp` (used in softmax) have much lower throughput than Tensor Core matmuls. For head dim 128, `exp` can take a sizable fraction of time (up to ~50% of matmul cycles) (§3.1).
- Mechanism
  - Use `bar.sync` to schedule GEMMs in warpgroup A ahead of warpgroup B so that B’s softmax is overlapped with A’s GEMMs, then swap roles (“ping–pong”). This pairs the slow MFU operations (exp/fma for softmax) with concurrent high-throughput GEMMs (Fig. 1).
- Effect
  - Improves FP16 forward speed, e.g., from 570 TFLOPs/s to 620–640 TFLOPs/s in a representative setting (sequence length 8192, head dim 128) (§3.1).

C. Intra-warpgroup 2‑stage GEMM–softmax pipeline (§3.2, Algorithm 2, Fig. 2)
- Challenge
  - Within a single iteration, softmax depends on `S = QK^T`, and `O += softmax(S) V` depends on softmax, creating serial waits (Algorithm 1, lines 17 and 21).
- Idea
  - Pipeline across iterations with additional register buffers:
    - Keep two score tiles: `S_cur` and `S_next`.
    - Overlap GEMM2 for iteration j−1 (`O += P̃_cur V_{j−1}`) with GEMM1 for iteration j (`S_next = Q K_j^T`), and interleave softmax for `S_next` while the previous GEMM2 is finishing (Algorithm 2, lines 8–16).
- Execution (simplified)
  - Warm start on `K_0`, compute `S_cur`, softmax and rescale once.
  - For j = 1..T_c−2:
    - Issue `S_next = Q K_j^T` (WGMMA, no wait), and concurrently issue `O += P̃_cur V_{j−1}` (WGMMA, no wait).
    - When `S_next` is ready, compute softmax for `S_next` while `O`-update is still running; after `O`-update finishes, rescale `O` and advance buffers (Algorithm 2, lines 11–16).
  - Finish with the final `V` block and epilogue scaling (lines 18–20).
- Practicalities
  - Extra registers are needed to hold `S_next` and intermediate state; tile sizes must balance register pressure vs throughput (§3.2).
  - SASS analysis (Appendix B.2) confirms the compiler generates overlapped instruction streams: the first WGMMA is interleaved with softmax and FP32→FP16 conversions; the second WGMMA is issued as a packed block with appropriate waits.
  - A 3‑stage variant (Appendix B.3, Fig. 8) was explored but performed worse due to higher register pressure and compiler reordering that limited overlap.

D. Backward pass warp specialization (§B.1, Algorithm 3)
- Structure
  - Add a third “dQ-writer” role to handle atomic accumulation of `dQ` into GMEM while consumers immediately proceed to the next tiles (Algorithm 3, lines 30–34). This avoids blocking on reductions to `dQ`.
  - Consumers recompute local `S` blocks (as in standard FA backprop) and compute:
    - `dP = dO V^T` (GEMM), `P = exp(S − L)` (elementwise), `dS = P ∘ (dP − D)` (elementwise), then update `dV`, `dK` with GEMMs and compute a local `dQ` (Algorithm 3, lines 21–29).
- Benefit
  - Maintains the asynchrony pattern from the forward pass while addressing contention on `dQ`.

E. FP8 path: layout, transpose, and accuracy techniques (§3.3)
- Layout constraints to use FP8 WGMMA (§3.3; §2.2)
  - FP8 WGMMA accepts only `k`‑major operands from SMEM, unlike FP16 which allows both `mn`‑major and `k`‑major (§2.2). Attention’s second GEMM (`P̃ V`) therefore needs `V` tiles arranged contiguous along sequence length.
- Efficient in‑kernel transpose of V (Fig. 4 and text in §3.3)
  - Rather than a costly pre-transpose in GMEM, tiles of `V` are transposed inside the kernel after TMA loads, using Hopper’s `LDSM`/`STSM` (collective SMEM↔register transfers) to perform 128‑byte swizzles with low register overhead; after the first iteration, these transposes are scheduled in the “shadow” of the GEMMs so they cost little wall time.
- Accumulator→operand register relayout (Fig. 3→Fig. 4)
  - The FP32 accumulator layout produced by the first FP8 WGMMA does not match the register layout required for operand A of the second FP8 WGMMA. Byte‑permute instructions reorder each 8‑byte chunk, e.g., `{d0 d1 d4 d5 d2 d3 d6 d7}`, to form the next WGMMA operand; the in-kernel `V` transpose writes a matching row permutation (§3.3).
- Accuracy: block quantization + incoherent processing (§3.3, §4.3, Table 3)
  - Block quantization: use per‑block scales (e.g., per `B_r×d` or `B_c×d` tile) instead of per‑tensor to reduce quantization error. The rescaling integrates naturally into the tiled softmax, incurring negligible extra compute.
  - Incoherent processing: multiply both `Q` and `K` by the same random orthogonal matrix `M` (Hadamard × random signs), so `QK^T` is unchanged but outliers are “spread out,” reducing FP8 quantization error. It costs O(d log d) and can be fused with rotary embedding (§3.3).

Implementation notes
- Built with CUTLASS primitives for WGMMA and TMA (§4).
- Benchmarks fix H100 clock to 1830 MHz and average over 100 runs (§C.1). FLOPs accounting is specified (§4.1).

## 4. Key Insights and Innovations
1) Warp-specialized producer–consumer pipeline with circular SMEM buffers (Algorithm 1; §3.1)
- What’s new: A deliberate division of labor across warps plus pipelined TMA prefetch keeps Tensor Cores busy while hiding GMEM latency.
- Why it matters: Increases effective overlap in a real kernel, improving utilization from FA‑2’s ~35% toward GEMM‑like levels (§1).

2) Cross-warp “ping–pong” scheduling to hide softmax under GEMMs (§3.1, Fig. 1)
- What’s new: Statically schedules warpgroups so that while one group’s Tensor Cores compute, the other’s MFUs execute softmax computations.
- Impact: Empirically improves FP16 forward performance by roughly 9–12% in a representative setting (570 → 620–640 TFLOPs/s) (§3.1).

3) Intra-warpgroup 2‑stage pipelining of GEMM and softmax (Algorithm 2; §3.2)
- What’s new: Breaks iteration-level dependencies by double-buffering scores and interleaves WGMMA instructions with softmax math (validated by SASS in §B.2).
- Impact: Ablations (Table 2, §4.2) show the overlap plus warp-specialization jointly raise throughput from 570 to 661 TFLOPs/s on a fixed setting.

4) FP8 attention that is both fast and accurate (§3.3; Fig. 3–4; Table 3)
- Efficiency innovations: In-kernel SMEM transpose with `LDSM/STSM` and accumulator→operand relayout using byte permutes enable back-to-back FP8 WGMMAs without extra global traffic.
- Accuracy innovations: Block quantization and incoherent processing reduce FP8 RMSE by 2.6× vs a per‑tensor-scale baseline while achieving close to 1.2 PFLOPs/s (Fig. 7; Table 3).
- Significance: Moves FP8 from an attractive theoretical speedup to a practical, accurate attention primitive.

Fundamental vs incremental
- Fundamental: Architectural re-planning around asynchrony (producer–consumer, ping–pong, intra-warp overlapping) and FP8‑aware layout/quantization constitute new algorithmic structures for attention on Hopper.
- Incremental: Engineering choices (e.g., setmaxnreg tuning, specific tile sizes) are important but build on the fundamental ideas.

## 5. Experimental Analysis
Evaluation setup (§4.1, §C.1)
- Hardware/software: H100 80GB SXM5, CUDA 12.3, cuDNN 9.1.1, CUTLASS 3.5, PyTorch 2.3; clock fixed to 1830 MHz; 100× runs averaged.
- Workloads: Sequence lengths 512–16k; total tokens fixed to 16k by adjusting batch size. Hidden size 2048; head dimensions 64/128/256. Both causal and non‑causal settings.
- FLOPs accounting: Forward FLOPs = `4·seqlen^2·headdim·nheads`; causal masks halve FLOPs; backward FLOPs ≈ 2.5× forward (§4.1).
- Baselines: Standard PyTorch attention; FlashAttention‑2; an H100‑optimized FA‑2 Triton kernel; cuDNN 9 attention.

Main results (all TFLOPs/s)
- FP16 forward speedups (Fig. 5)
  - Head dim 64, non‑causal: FA‑3 ranges 333–497; FA‑2 282–324; Triton 382–403; cuDNN 335–413 (Fig. 5a).
    - At 16k tokens: FA‑3 497 vs cuDNN 413 and FA‑2 324 → strong advantage for FA‑3.
  - Head dim 64, causal: FA‑3 197–473 vs cuDNN 225–388; FA‑2 180–299 (Fig. 5b). FA‑3 leads at long sequences.
  - Head dim 128, non‑causal: FA‑3 497→595; cuDNN 467→648; Triton 323→395; FA‑2 309→370 (Fig. 5c).
    - cuDNN slightly edges FA‑3 at long sequences; both far ahead of FA‑2/Triton.
  - Head dim 128, causal: FA‑3 292→616 vs cuDNN 315→539; FA‑2 191→335; Triton 146→378 (Fig. 5d). FA‑3 leads.
  - Head dim 256, non‑causal: FA‑3 482→756 vs cuDNN 470→581; FA‑2 275→326 (Fig. 5e). FA‑3 clearly leads at all lengths.
  - Head dim 256, causal: FA‑3 286→642 vs cuDNN 391→509; FA‑2 208→308 (Fig. 5f).
  - Peak FP16 forward ≈ 740 TFLOPs/s (75% of theoretical max) noted in §4 (matches Fig. 5e trends).
- FP16 backward speedups (Fig. 6)
  - Head dim 64, non‑causal: FA‑3 272→474 vs FA‑2 198→291 and cuDNN 266→433 (Fig. 6a).
  - Head dim 128, non‑causal: FA‑3 316→561 vs FA‑2 214→322 and cuDNN 305→516 (Fig. 6b).
  - Claimed overall: 1.5–1.75× faster than FlashAttention‑2 on backward (§4.1).
- FP8 forward (Fig. 7; full in Fig. 9)
  - Head dim 256, non‑causal: FA‑3 reaches 1171 TFLOPs/s at 16k (≈1.17 PFLOPs/s), competitive with or above Triton/cudnn across most lengths (Fig. 7a).
  - Head dim 256, causal: FA‑3 299→1024 vs cuDNN 304→1099; FA‑3 is close at long sequences but trails cuDNN at some lengths (Fig. 7b).
  - The abstract summarizes: “FP8 reaches close to 1.2 PFLOPs/s.”

Ablation and compiler validation
- 2‑stage pipelining and warp specialization both matter (Table 2, §4.2):
  > “FlashAttention‑3: 661 TFLOPs/s vs No GEMM–Softmax Pipelining: 582 and No Warp‑Specialization: 570.”
- SASS inspection (Appendix B.2) confirms the intended overlap: early softmax and FP32→FP16 conversions interleave with the first WGMMA; the second WGMMA runs as a packed block with proper dependency barriers.

Accuracy study (§4.3, Table 3)
- Stress test with outliers: inputs sampled as `N(0,1) + N(0,100)·Bernoulli(0.001)`.
- FP16 RMSE vs FP64 reference:
  > Baseline 3.2e‑4; FA‑2 1.9e‑4; FA‑3 1.9e‑4 (keeping softmax in FP32 helps both FA‑2 and FA‑3).
- FP8 RMSE:
  > Baseline (per‑tensor scale) 2.4e‑2; FA‑3 9.1e‑3; No block quant 9.3e‑3; No incoherent processing 2.4e‑2.
  - Both block quantization and incoherent processing are needed for the full 2.6× error reduction.

Do results support the claims?
- Yes for speed: Across settings, FA‑3 consistently outperforms FA‑2 by ~1.5–2.0× (forward) and ~1.5–1.75× (backward), and often surpasses cuDNN at longer sequences or larger head dims (Fig. 5e–f). FP8 throughput approaches 1.2 PFLOPs/s (Fig. 7).
- Yes for accuracy: Table 3 shows FP8 error improvements attributable to the proposed block quantization and incoherent processing.

Nuances and conditions
- FA‑3 can trail cuDNN in some FP16 non‑causal cases at head dim 128 (Fig. 5c) and in FP8 with causal masking at smaller lengths (Fig. 7b; discussed in §5 and footnote 10 about persistent kernels).
- Benefits grow with sequence length and larger head dimension where overlapping becomes more effective.

## 6. Limitations and Trade-offs
- Architecture dependence
  - The design assumes Hopper-like asynchrony (TMA and WGMMA) and register reallocation (`setmaxnreg`) (§2.2). On older GPUs, many benefits may not materialize.
- Register pressure vs tile size
  - 2‑stage (and especially 3‑stage) pipelining consumes extra registers to hold double-buffered tiles (`S_next`, intermediate P̃), forcing smaller tiles or lower occupancy (§3.2; Appendix B.3).
- FP8 layout and transpose complexity
  - Achieving FP8 speed requires in‑kernel `V` transposes (LDSM/STSM) and byte‑permute relayouts (Fig. 3–4). These add implementation complexity and can be sensitive to compiler scheduling (§3.3).
- Compiler reordering
  - Some intended overlaps are subject to compiler scheduling. Appendix B.2 shows good overlap for 2‑stage, but Appendix B.3 reports suboptimal reordering that limited benefits for 3‑stage pipelining.
- Small‑sequence and causal FP8 performance
  - FA‑3 FP8 does not yet use a persistent kernel; the paper notes this contributes to weaker performance at small sequence lengths and with causal masking compared to cuDNN (§5, footnote 10).
- Scope
  - The work optimizes exact attention kernels. It does not address higher-level memory management (e.g., paged KV caches) or algorithmic approximations for very large contexts.

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that redesigning attention around hardware asynchrony and low precision yields near‑GEMM efficiency while preserving exactness. This shifts the default assumption from “optimize instructions” to “co‑design the algorithm with the GPU execution model.”
- Practical applications
  - Faster training and inference for long-context LLMs; better utilization in multi-query/grouped-query attention without changing math (§3.1, “Attention variants”); integration into distributed attention frameworks (e.g., Ring Attention) where FA‑3 becomes the faster primitive (§A).
- Follow‑up research
  - Persistent-kernel FP8 attention to improve small‑sequence and causal performance (§5).
  - Extending to newer precisions (e.g., FP4 on Blackwell) and other accelerators with similar async features (§1, §2.2).
  - Automated tile/pipeline autotuning under register and SMEM constraints.
  - Studying training dynamics with FP8 attention at scale (stability, convergence) and broader quantization strategies for KV caches and activations.
  - Compiler–kernel co-design to improve instruction scheduling for deeper pipelines (addressing the Appendix B.3 limitations).

Overall, FlashAttention-3 is a hardware-aware rethinking of exact attention: warp-specialized producer–consumer pipelines, overlapped GEMM–softmax execution, and FP8‑specific layout/quantization turn Hopper’s asynchrony and low-precision support into large, measured gains—up to 740 TFLOPs/s in FP16 and ≈1.2 PFLOPs/s in FP8, with FP8 accuracy improved by 2.6× (Fig. 5–7; Table 3).
