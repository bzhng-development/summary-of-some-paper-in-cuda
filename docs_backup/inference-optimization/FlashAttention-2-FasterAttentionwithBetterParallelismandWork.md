# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**ArXiv:** [2307.08691](https://arxiv.org/abs/2307.08691)

## 🎯 Pitch

FlashAttention-2 radically refines the exact attention algorithm for Transformers to align with modern GPU hardware, introducing smarter work partitioning and improved parallelism across sequence and within compute units. This innovation slashes non-matrix operations and boosts GPU utilization, enabling attention to run up to twice as fast as FlashAttention v1 and reaching up to 73% of theoretical FLOPs, which dramatically lowers the computational barrier for scaling models to longer contexts and faster training. By doing so, it makes efficient, large-context, exact attention feasible for both research and real-world applications, facilitating breakthroughs in language, vision, and multimodal AI.

---

## 1. Executive Summary (2-3 sentences)
FlashAttention-2 redesigns the exact (non-approximate) attention kernel to better match GPU hardware, cutting non-matrix operations and improving parallelism across sequence length and within GPU warps. It delivers roughly 2× speedup over FlashAttention v1 and reaches up to 50–73% of a GPU’s theoretical FLOPs in forward pass and up to 63% in backward pass (A100), enabling end-to-end GPT-style training at up to 225 TFLOPs/s per A100 GPU (Section 4; Figs. 5–6; Table 1).

## 2. Context and Motivation
- Problem being addressed
  - Exact attention scales quadratically in sequence length `N` in both runtime and memory. Standard implementations materialize the score matrix `S = QK^T` and the probability matrix `P = softmax(S)` in high-bandwidth memory (HBM), causing O(N^2) memory and heavy memory traffic (Section 2.2).
  - Even with FlashAttention v1—which reorders computation to avoid materializing `S` and `P`—GPU utilization remains far from matrix-multiply (GEMM) efficiency: forward reaches only 30–50% and backward 25–35% of the device’s theoretical maximum on A100 (Section 1; Figs. 5–6).

- Why it matters
  - Longer contexts unlock key applications (e.g., book-length reasoning, high-resolution images, long audio/video) and are already used by frontier models with 32k–100k tokens (Section 1). Making exact attention substantially faster reduces training and inference cost and enables longer contexts within the same budget.

- Prior approaches and their shortcomings
  - Approximate attention (e.g., Performer, Linformer, Longformer; Section 1, References) reduces complexity but is not the standard in large-scale training, likely due to approximation error or engineering complexity.
  - FlashAttention v1 (Section 2.3) achieves exact attention with linear extra memory via tiling and online softmax, but:
    - It still spends too many cycles on non-matmul operations that GPUs execute much more slowly than Tensor Core matmuls (Section 3.1: A100 example 312 TFLOPs/s for FP16/BF16 matmul vs 19.5 TFLOPs/s for scalar FP32).
    - Its work partitioning across thread blocks and warps leads to low occupancy or excessive shared-memory traffic (Section 1, 3.2–3.3).

- Positioning of this work
  - FlashAttention-2 keeps exactness but redesigns the kernel to:
    - Reduce non-matmul FLOPs (Section 3.1).
    - Increase occupancy via parallelization along the sequence dimension (not just batch and heads; Section 3.2).
    - Repartition work inside thread blocks to avoid costly inter-warp communication (Section 3.3).
  - The result is a kernel whose efficiency approaches optimized GEMM (50–73% forward on A100; Fig. 5).

## 3. Technical Approach
Before diving in, quick definitions of less common terms (Section 2.1):
- `HBM` (High-Bandwidth Memory): Large off-chip GPU memory with high bandwidth but higher latency.
- `SRAM`/shared memory: Small on-chip memory with very high bandwidth, manually managed by the programmer.
- `Thread block` and `warp`: GPU executes kernels via many threads; 32 threads form a warp; several warps form a thread block scheduled on a streaming multiprocessor (SM).
- `Occupancy`: Fraction of the GPU’s hardware resources that are busy doing useful work.
- `Tiling`: Splitting matrices into blocks that fit on-chip to reduce HBM traffic.
- `Online softmax`: A numerically stable way to compute softmax over a row by streaming blocks and keeping running statistics (row-wise max and sum).
- `logsumexp`: Numerically stable computation of log(sum(exp(.))) used to represent softmax denominators.
- `Atomic add`: A thread-safe addition to a memory location from multiple parallel workers.

A. Baseline computations and FlashAttention v1 (Section 2.2–2.3)
- Standard attention:
  - Scores `S = QK^T`, probabilities `P = softmax(S)`, outputs `O = PV`.
  - Backward uses `dV = P^T dO`, `dP = dO V^T`, `dS = softmax’(S) dP`, `dQ = dS K`, `dK = Q^T dS` (Section 2.2).
  - Stores `S` and `P` to HBM, incurring O(N^2) memory and heavy I/O.
- FlashAttention v1:
  - Tiles over keys/values and computes partial contributions while keeping data on-chip (SRAM), using online softmax to maintain correctness without materializing `S` or `P` in HBM (Section 2.3.1; Fig. 1).
  - Backward recomputes needed blocks of `S` and `P` on the fly and avoids storing them, saving memory and I/O (Section 2.3.2).

B. FlashAttention-2: algorithmic tweaks to reduce non-matmul FLOPs (Section 3.1.1; Algorithm 1)
- Goal: Spend more time on matmuls (very fast on Tensor Cores) and less on scalar ops.
- Two key tweaks:
  1) Defer scaling of `O` until the end:
     - Instead of repeatedly re-scaling both terms on each block, maintain an “unscaled” running output `Õ` and the running softmax denominator stats `ℓ`. Only once per row block (after all key/value blocks are processed) apply `O = diag(ℓ)^{-1} Õ` (Algorithm 1, steps 9–12 and the derivation preceding Algorithm 1).
  2) Store only `L = logsumexp` for backward:
     - Rather than store both the running row-wise max `m` and sum of exponentials `ℓ`, store `L = m + log(ℓ)`. This single statistic suffices to reconstruct `P = exp(S − L)` on the fly during backward (Algorithm 1, step 13; Algorithm 2, step 11).

How the forward pass works (Algorithm 1; Fig. 1)
- Partition:
  - Split queries into `T_r` row blocks of size `B_r × d`, and keys/values into `T_c` column blocks of size `B_c × d` (Algorithm 1, steps 1–2).
- For each query row block `Q_i`:
  1) Load `Q_i` into SRAM; initialize `O_i^(0)=0`, `ℓ_i^(0)=0`, `m_i^(0) = -∞` (Algorithm 1, steps 4–5).
  2) Loop over key/value column blocks `j = 1..T_c`:
     - Load `K_j, V_j` (step 7); compute scores `S_i^(j) = Q_i K_j^T` (step 8).
     - Update statistics (step 9): row-wise max `m_i^(j)`, partial exponentials `P̃_i^(j) = exp(S_i^(j) − m_i^(j))`, and softmax denominator `ℓ_i^(j) = exp(m_i^(j−1) − m_i^(j)) ℓ_i^(j−1) + rowsum(P̃_i^(j))`.
     - Update running unscaled output (step 10):
       `O_i^(j) = diag(exp(m_i^(j−1) − m_i^(j)))^{-1} O_i^(j−1) + P̃_i^(j) V_j`.
  3) After processing all `K,V` blocks, finalize:
     - Scale once to get the true `O_i = diag(ℓ_i^(T_c))^{-1} O_i^(T_c)` and store `L_i = m_i^(T_c) + log(ℓ_i^(T_c))` (steps 12–15).

Causal masking optimization (Section 3.1.1)
- For autoregressive models, entries with column index > row index are masked. FlashAttention-2 exploits block structure:
  - Skip any block entirely above the causal diagonal (roughly half the blocks).
  - Apply the mask within at most one block per row (assuming square blocks).
- The design yields a practical wall-clock speedup in causal settings; the FLOPs accounting halves the forward FLOPs (Section 4.1’s FLOPs discussion).

C. FlashAttention-2 backward pass (Algorithm 2)
- Compute a helper vector once: `D = rowsum(dO ∘ O)` (elementwise multiply and row-sum; Algorithm 2, step 4) to factor the softmax derivative efficiently.
- Loop over column blocks `j` (Algorithm 2, steps 5–19):
  - Load `K_j, V_j`, initialize `dK_j, dV_j = 0` on SRAM (steps 6–7).
  - For each row block `i`:
    1) Load `Q_i, O_i, dO_i, L_i, D_i` and recompute scores `S_i^(j) = Q_i K_j^T` (steps 9–10).
    2) Recompute block probabilities via the stored `L`: `P_i^(j) = exp(S_i^(j) − L_i)` (step 11).
    3) Accumulate gradients:
       - `dV_j += (P_i^(j))^T dO_i` (step 12).
       - `dP_i^(j) = dO_i V_j^T` (step 13).
       - `dS_i^(j) = P_i^(j) ∘ (dP_i^(j) − D_i)` (step 14) — this is the softmax Jacobian-vector product in a compact form.
       - Update `dQ_i += dS_i^(j) K_j` (step 15) and `dK_j += (dS_i^(j))^T Q_i` (step 16).
  - Write `dK_j, dV_j` back to HBM (step 18).
- Parallel update to `dQ` uses atomic adds (Section 3.2) when multiple thread blocks contribute to the same `dQ_i`.

D. Parallelization across sequence length (Section 3.2; Fig. 2)
- Forward:
  - Parallelize the outer loop across row blocks `i`: each thread block computes a different `Q_i` slice independently (Fig. 2 left). This raises occupancy when batch size and number of heads are small (common in long-context training).
- Backward:
  - Parallelize across column blocks `j`: each thread block owns a different `K_j,V_j` slice and accumulates contributions. Because all column blocks contribute to the same `dQ_i`, the kernel uses atomic adds for `dQ` (Fig. 2 right).
- This extends beyond v1, which parallelized primarily over batch and heads.

E. Work partitioning within a thread block (warps) (Section 3.3; Fig. 3)
- v1 “split-K” scheme:
  - Split `K,V` across warps while keeping `Q` shared. Warps must exchange partial results via shared memory to form `O`, causing extra shared-memory reads/writes and synchronizations (Fig. 3a).
- FlashAttention-2 “split-Q” scheme:
  - Split `Q` across warps while sharing `K,V`. Each warp computes its score submatrix `Q_part K^T` and immediately multiplies by shared `V` to produce its output slice—no inter-warp accumulation needed (Fig. 3b).
  - This removes a major source of shared-memory traffic and sync overhead, directly improving speed (Section 3.3).

F. Practical engineering details
- Block sizes: typically choose from `{64,128} × {64,128}` depending on head dimension `d` and shared memory limits. Larger tiles reduce shared-memory traffic but risk register spilling or exceeding SRAM capacity (Section 3.3).
- Causal masking: skipping full blocks and masking at most one block per row is built into the tiled design (Section 3.1.1).
- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA): reuse shared `K,V` across multiple query heads; in backward, sum `dK, dV` across the duplicated heads (Section 3.1.2).

## 4. Key Insights and Innovations
- Reduce non-matmul FLOPs so GPU spends time on Tensor Cores (Section 3.1)
  - Why it matters: On A100, matmul throughput (FP16/BF16) can be ~16× higher than scalar FP32 operations (Section 3.1). By deferring scaling and storing only `logsumexp`, the kernel cuts repeated per-block scaling and bookkeeping.
  - Impact: Contributes to the 2× speedup over v1.

- Parallelize across the sequence length dimension (Section 3.2; Fig. 2)
  - New capability: v1 primarily parallelized over batch and heads; v2 also slices the sequence dimension so more thread blocks run concurrently.
  - When it helps: Especially beneficial at long sequences where batch size/head count are small, improving occupancy and throughput.

- Split-Q warp partitioning eliminates inter-warp reductions (Section 3.3; Fig. 3)
  - Difference from prior: v1’s split-K required warps to communicate via shared memory; v2’s split-Q lets each warp compute its output slice independently.
  - Why it’s significant: Reduces shared-memory reads/writes and synchronizations, unlocking higher practical FLOPs/s.

- Exactness with minimal saved state (Algorithms 1–2)
  - Maintain exact attention outputs with no approximation, while storing only per-row `logsumexp` values for backward. This keeps extra memory linear in `N` (Section 3.1.1), preserving the exactness and memory benefits of FlashAttention v1.

These are principally system/kernel-level innovations rather than new attention mathematics. The combination—algorithmic tweaks + parallelization + warp partitioning—is what lifts the kernel toward GEMM-level efficiency.

## 5. Experimental Analysis
Evaluation design (Section 4.1)
- Hardware and setup:
  - A100 80GB SXM4 (primary); H100 80GB SXM5 (secondary, no use of new H100-specific instructions like TMA or FP8; Section 4.1 and Fig. 7).
  - Sequence lengths: 512 to 16k; total tokens per batch fixed at 16k by scaling batch size accordingly (Section 4.1).
  - Hidden size: 2048; head dimension: 64 or 128; with and without causal mask (Fig. 4–6 captions).
- Metrics:
  - Throughput in TFLOPs/s; forward FLOPs counted as `4 * seqlen^2 * head_dim * num_heads`. With causal mask, FLOPs halved to reflect fewer computed entries. Backward FLOPs are 2.5× forward (two forward matmuls vs five in backward due to recomputation; Section 4.1).
- Baselines:
  - PyTorch standard attention.
  - FlashAttention v1 (CUTLASS/xformers).
  - FlashAttention (Triton implementation).
  - FlashAttention-2.

Main quantitative results
- Forward + backward throughput on A100 (Fig. 4):
  - “No causal mask, head dim 128” (Fig. 4b): FlashAttention-2 reaches 151–203 TFLOPs/s across 512–16k tokens, vs Triton 78–95 and v1 76–91. Speedups are roughly 1.7–2.6× over v1 and ~1.6–2.1× over Triton.
  - “Causal mask, head dim 128” (Fig. 4d): FlashAttention-2 reaches 99–189 TFLOPs/s across 512–16k, vs Triton 50–80 and v1 53–89.
- Forward-only throughput on A100 (Fig. 5):
  - FlashAttention-2 reaches up to 224–227 TFLOPs/s without causal mask, head dim 128 (Fig. 5b), representing as much as 73% of the A100 peak, per the paper’s summary (Section 4).
  - With causal mask, head dim 128, FlashAttention-2 reaches up to ~200 TFLOPs/s (Fig. 5d).
- Backward-only throughput on A100 (Fig. 6):
  - FlashAttention-2 reaches up to 196 TFLOPs/s (no causal, head dim 128, 16k; Fig. 6b), consistent with the paper’s “up to 63% of the theoretical max” for backward (Section 4).
- H100 results (Fig. 7):
  - Without using new H100 features, FlashAttention-2 attains up to 335 TFLOPs/s in forward+backward with head dim 128 (Fig. 7b), suggesting headroom from hardware features still unused.
- End-to-end GPT training throughput (Table 1):
  - With 1.3B parameters, 8k context: 72 TFLOPs/s without FlashAttention, 170 with v1, 220 with v2.
  - With 2.7B parameters, 8k context: 80 TFLOPs/s without FlashAttention, 175 with v1, 225 with v2.
  - Quoted peak: “up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization)” (Table 1 and Section 1).

Do the experiments support the claims?
- Coverage and consistency:
  - Results span sequence lengths (512–16k), head dims (64, 128), and masking regimes (with/without causal), with consistent gains across settings (Figs. 4–6).
  - The speedups over v1 are typically around 2×, in line with the abstract and Section 4.
- Kernel-level attribution:
  - The improvements match the design changes: better occupancy (parallelizing along sequence length, Fig. 2) and less shared-memory traffic (split-Q, Fig. 3). The backward’s lower percent of peak vs forward is also consistent with more non-matmul work and synchronization.
- Accounting nuances:
  - With causal masking, the FLOPs are halved in reporting (Section 4.1). Because certain overheads do not halve, TFLOPs/s can appear lower even if wall-clock time improves; the paper notes a practical speedup from skipping half the blocks (Section 3.1.1).
- Ablations and robustness:
  - While there is no explicit ablation isolating each tweak, the systematic throughput improvements across settings, plus the theoretical rationale (Sections 3.1–3.3), make the case persuasive.
  - The paper also reports results on H100 to suggest portability and further headroom (Fig. 7).

## 6. Limitations and Trade-offs
- Still quadratic in time:
  - FlashAttention-2 is exact and does not change attention’s O(N^2) FLOP count. It reduces constant factors and memory traffic; it does not solve the asymptotic runtime scaling.
- Hardware specialization and manual tuning:
  - Performance depends on GPU architecture details (shared memory size, register file limits). Block sizes must be tuned (typically 64/128; Section 3.3). The paper calls out the need for auto-tuning (Section 3.3).
- Synchronization and atomics in backward:
  - Parallelizing backward across column blocks requires atomic adds to `dQ` (Section 3.2). On some workloads, contention can limit scaling.
- Non-matmul ops remain:
  - Even after reductions, backward still has significant non-matmul components (softmax derivative, reductions), explaining the lower utilization vs forward (Figs. 5–6).
- Masking and irregular patterns:
  - The paper focuses on dense attention (with optional causal mask). Highly irregular masks or block-sparse patterns are not implemented here (Section 5 suggests combining with block-sparse in future work).
- Feature completeness on newer hardware:
  - On H100, the reported kernel does not yet use new features like TMA or FP8 Tensor Cores; speedups are underestimates of what may be achievable (Fig. 7; Section 4.1).
- Scope:
  - The work targets NVIDIA GPUs. Portability to other vendors (e.g., AMD) and data types (e.g., FP8) is earmarked for future work (Section 5).

## 7. Implications and Future Directions
- What changes now
  - Training/inference with long contexts becomes substantially cheaper. As the paper summarizes, this makes “16k context for the same price as 8k context” a practical rule of thumb (Section 5).
  - Exact attention remains competitive, reducing pressure to adopt approximations unless asymptotic scaling is required.

- Research and engineering enabled
  - Longer-context model training at mainstream scale (e.g., 16k–32k) with standard attention is more feasible.
  - Kernel design principles—split-Q warps, sequence-parallel thread-blocking, online-softmax bookkeeping—are broadly applicable to other fused GPU kernels.
  - Compiler and auto-tuning research: the paper explicitly calls for making these optimizations more programmable (Section 5).

- Concrete next steps (also listed in Section 5)
  - Hardware-specific optimizations: exploit H100 features (TMA, 4th-gen Tensor Cores) and FP8, likely yielding an additional 1.5–2× on H100 (Section 4.1, Fig. 7).
  - Portability: extend to AMD GPUs and other accelerators.
  - High-level algorithmic combinations: integrate with local/dilated/block-sparse attention to push beyond 16k while staying efficient (Section 5).
  - Auto-tuning: automate tile-size and warp-partition choices to adapt to head dimensions and device constraints (Section 3.3).
  - Expanded masking and layout support: irregular masks, packed variable-length batches, and optimized KV-cache variants beyond the MQA/GQA support already included (Section 3.1.2).

In sum, FlashAttention-2 is a carefully engineered, exact attention kernel that better aligns computation with GPU strengths. By reducing non-matmul overhead, parallelizing across sequence length, and removing inter-warp reductions via split-Q partitioning, it delivers roughly 2× end-to-end speedups over FlashAttention v1 and approaches GEMM-like efficiency on modern GPUs (Sections 3–4; Figs. 4–6; Table 1).
