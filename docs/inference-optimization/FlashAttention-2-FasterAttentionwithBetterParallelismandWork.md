# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**ArXiv:** [2307.08691](https://arxiv.org/abs/2307.08691)

## 🎯 Pitch

FlashAttention-2 introduces a dramatically improved, exact attention kernel for GPUs, reorganizing computation to better parallelize across sequence length and optimize work distribution within each GPU thread block. By reducing non-matrix operations and carefully partitioning GPU tasks, it achieves up to a 2× speedup over FlashAttention v1 and reaches state-of-the-art hardware utilization, enabling efficient training and inference with longer context windows—critical for the next generation of large language and vision models.

---

## 1. Executive Summary
FlashAttention-2 is a drop-in, exact (non-approximate) attention kernel for GPUs that reorganizes how the attention operation is computed to better match the GPU’s hardware. It reduces non-matrix operations, parallelizes across sequence length, and repartitions work inside each GPU thread block—achieving about 2× speedup over FlashAttention v1 and up to 73% of peak device throughput on A100 GPUs, while preserving linear memory usage in sequence length (Sections 3–4).

## 2. Context and Motivation
- Problem addressed
  - Attention’s runtime and memory scale quadratically with sequence length `N` because the full `N×N` score matrix must be computed and (in standard implementations) materialized. This limits training and inference at long context lengths (Section 1, 2.2).
- Why this matters
  - Longer context enables new capabilities (long document reasoning, high-resolution image and video modeling, code/audio/video generation). Recent models target 32k–100k+ tokens, but standard attention’s costs are prohibitive (Section 1).
- Prior approaches and their limitations
  - Approximate attentions (sparse/linear/low-rank; e.g., Longformer, Performer, Linformer, Reformer) reduce complexity but change the computation and may impact quality; many large training runs still prefer exact attention (Section 1).
  - FlashAttention v1 reorders computation to reduce high-bandwidth memory (HBM) traffic using on-chip tiling and “online softmax,” yielding 2–4× speedups and linear extra memory, without approximation (Section 2.3).
  - Gap: Despite large gains, FlashAttention v1 attains only ~25–50% of maximum device FLOPs/s (especially poor in backward pass) due to suboptimal work partitioning, low occupancy, and unnecessary shared-memory traffic (Abstract; Fig. 5–6 discussion in Section 4).
- Positioning of this work
  - FlashAttention-2 keeps the exactness and IO-awareness of FlashAttention but redesigns:
    - The math to reduce non-matrix-multiply work,
    - The parallelization across sequence length to increase occupancy,
    - The warp-level work partition to avoid shared-memory reductions (“split-K”).
  - Result: ~2× further speedup over FlashAttention, approaching GEMM-like efficiency (Abstract; Section 4).

## 3. Technical Approach
This section explains how the computation is reorganized so the GPU does more of what it does best (matrix multiply on Tensor Cores) and less of what it does poorly (scalar ops and synchronization).

Terminology (GPU-specific, used throughout):
- `HBM`: Off-chip high bandwidth memory—large but slow relative to on-chip memory.
- `SRAM`/“shared memory”: On-chip memory—small but very fast; shared within a thread block.
- `Streaming Multiprocessor (SM)`: A GPU compute unit that runs one or more thread blocks.
- `Thread block` and `warp`: A thread block is a group of GPU threads scheduled together on an SM; a warp is 32 threads that execute in lock-step.
- `Occupancy`: The fraction of SM resources actively used; higher occupancy typically improves throughput.
- `GEMM`/matmul: Matrix multiply; on modern GPUs, specialized units (Tensor Cores) make GEMM much faster than non-matmul floating-point ops.

3.1 From standard attention to FlashAttention-2
- Standard attention (Section 2.2)
  - Compute `S = Q K^T`, then `P = softmax(S)` row-wise, then `O = P V`. Backprop requires `S`/`P` or their equivalents. Materializing `S` and `P` causes `O(N^2)` memory traffic and large HBM IO.
- FlashAttention v1 (Section 2.3)
  - Use tiling: load blocks of `Q`, `K`, `V` into SRAM; compute partial scores and outputs block-by-block; avoid writing `S` and `P` to HBM.
  - Use “online softmax” (Section 2.3.1): keep running row-wise maxima and normalization terms when processing column blocks, so at the end the result equals the full softmax without ever forming the full `S` or `P`.
- FlashAttention-2: three coordinated changes (Sections 3.1–3.3)
  1) Reduce non-matmul FLOPs in the math (Section 3.1)
     - Key idea: GPUs perform matmul up to ~16× faster than scalar ops (A100: 312 TFLOPs/s FP16 matmul vs. 19.5 TFLOPs/s FP32 non-matmul; Section 3.1). Spend proportionally more time in matmul; reduce rescaling/masking arithmetic.
     - Two forward-pass tweaks (Section 3.1.1; Algorithm 1):
       - Maintain an “unscaled” running output `Ô` during the K/V block loop, and do the row-wise scaling once at the end instead of at every block. Concretely, step 10 in Algorithm 1 computes `O_i^(j) = diag(exp(m_i^(j−1) − m_i^(j)))^{-1} O_i^(j−1) + exp(S_i^(j) − m_i^(j)) V_j`, and only at step 12 applies the final scaling by `diag(ℓ_i^(T_c))^{-1}`.
       - Store only the row-wise `logsumexp` `L = m + log(ℓ)` for backward, not both the row-wise max `m` and the sum-of-exponentials `ℓ` (steps 13 and the note in Section 3.1).
     - Backward-pass tweak (Section 3.1.2; Algorithm 2):
       - Recompute local `S` and `P` on the fly from `Q`, `K`, and `L` per block; no need to save full `P`. Uses `P = exp(S − L)` (Algorithm 2, step 11).
  2) Increase parallelism by partitioning along sequence length (Section 3.2; Fig. 2)
     - Forward pass: parallelize different row blocks across thread blocks (outer loop over row blocks becomes the parallel dimension). Each thread block processes a distinct set of rows; no inter-block communication needed. This boosts occupancy especially when batch size and number of heads are small but sequences are long (Section 3.2, “Forward pass”).
     - Backward pass: parallelize across column blocks. Each thread block processes one column block and updates `dK`, `dV` locally; updates to `dQ` are combined using atomic adds since many column blocks contribute to the same `dQ` rows (Algorithm 2, steps 15–18; Section 3.2 “Backward pass”).
  3) Repartition work within a thread block to avoid shared-memory reductions (Section 3.3; Fig. 3)
     - Problem in FA v1: “split-K” strategy—split `K`/`V` across warps while all warps share `Q`. Each warp computes a partial `QK^T`, then all warps must write/read partial results from shared memory and synchronize to combine them before multiplying by `V`. This creates extra shared-memory traffic (Section 3.3 “Forward pass”).
     - FA-2 strategy: split `Q` across warps while all warps share `K` and `V`. Each warp produces an independent slice of `QK^T` and immediately multiplies by the shared `V` to get its own slice of the output—no inter-warp reduction required. This removes a major shared-memory bottleneck (Section 3.3; Fig. 3b).
     - Backward pass uses analogous non–split-K partitioning, still with some synchronization due to gradient dependencies, but substantially fewer shared-memory reads/writes (Section 3.3 “Backward pass”).

3.2 Additional engineering choices and correctness
- Causal masking (Section 3.1.1 “Causal masking”)
  - Skip entire column blocks that lie strictly above the causal diagonal (approximately half the blocks for large `N`), and only apply masking within the one boundary block per row. This yields ~1.7–1.8× speedup relative to the non-causal case simply by not computing the upper-triangular half.
- Correctness and complexity (Section 3.1.1 “Correctness, runtime, and memory requirement”)
  - The final `O` equals `softmax(QK^T) V` exactly; algorithms use `O(N^2 d)` FLOPs overall, but require only `O(N)` extra memory (to store `L`) beyond inputs/outputs. This mirrors FlashAttention’s guarantees, now with fewer non-matmul ops.
- Block-size tuning (Section 3.3 “Tuning block sizes”)
  - Typical blocks are `{64, 128} × {64, 128}`. Larger blocks reduce shared-memory transfers but demand more registers/shared memory; too large causes register spilling or exceeds available SRAM (e.g., A100 has 192 KB per SM; Section 2.1).
- Support for multi-query and grouped-query attention (Section 3.1.2)
  - Reuse the same `K`/`V` across multiple query heads by manipulating head indices; in backward, sum the gradients for `dK` and `dV` across the linked heads.

3.3 How the backward pass works on-chip (Algorithm 2; Section 3.1.2)
- For each `K`/`V` block `j`:
  - Load `K_j`, `V_j`; initialize `dK_j`, `dV_j` in SRAM (steps 6–7).
  - For each `Q`/`O`/`dO` row block `i`:
    - Recompute local scores `S_i^(j) = Q_i K_j^T` and probabilities `P_i^(j) = exp(S_i^(j) − L_i)` (steps 10–11).
    - Accumulate `dV_j += P_i^(j)^T dO_i` and `dK_j += (P_i^(j) ∘ (dO_i V_j^T − D_i))^T Q_i` (steps 12–16), where `D = rowsum(dO ∘ O)` is precomputed row-wise (step 4); `∘` is pointwise multiplication.
    - Update `dQ_i` atomically across column blocks: `dQ_i += dS_i^(j) K_j` (step 15).
- Write `dK_j`, `dV_j` back to HBM (step 18); proceed to next column block (step 19).

Analogy: Think of the attention matrix as tiles on a chessboard. FA-2 computes only the tiles it must (lower triangle for causal), processes many rows in parallel (each row group handled by a different “worker”), and ensures each worker keeps as much as possible on-chip while minimizing back-and-forth through the slow main memory.

## 4. Key Insights and Innovations
- Reduce non-matmul FLOPs by deferring scaling and storing `logsumexp` only (Section 3.1.1; Algorithms 1–2)
  - Novelty: A small but surgical change to the online-softmax workflow: keep an “unscaled” `Ô` and compute scaling once; carry only `L = m + log(ℓ)`. This lowers expensive scalar ops while preserving exactness.
  - Significance: Non-matmul FLOPs can be ~16× slower per FLOP on A100 (Section 3.1), so shaving them off materially improves end-to-end throughput.
- Parallelize along sequence length in both forward and backward (Section 3.2; Fig. 2)
  - Novelty: Treat row blocks (forward) and column blocks (backward) as independent work units schedulable to separate thread blocks; handle `dQ` with atomic adds.
  - Significance: Raises occupancy when batch size or head count is small but sequences are long—exactly the regime targeted for long-context models.
- Warp-level partition that avoids “split-K” (Section 3.3; Fig. 3)
  - Novelty: Split `Q` instead of `K` across warps so each warp completes an end-to-end slice without shared-memory reduction.
  - Significance: Cuts shared-memory reads/writes and synchronization—directly addressing the main bottleneck that left FA v1 well below GEMM efficiency.
- Causal-mask-aware block skipping and minimal masking (Section 3.1.1)
  - Incremental but impactful: Skip upper-triangular blocks entirely and only mask one boundary block per row, yielding ~1.7–1.8× speedup relative to the non-causal compute pattern.

Overall, the fundamental innovation is architectural: reshaping the computation to match GPU hardware hierarchies and scheduling, rather than changing the attention formula itself.

## 5. Experimental Analysis
Evaluation setup and metrics
- Benchmarks (Section 4.1; Figs. 4–6)
  - Hardware: NVIDIA A100 80GB SXM4; also H100 80GB SXM5 in Fig. 7.
  - Settings: sequence lengths from 512 to 16k; total tokens fixed at 16k by adjusting batch size; hidden size 2048; head dimension 64 or 128 (i.e., 32 or 16 heads).
  - Metrics: Speed in TFLOPs/s. FLOPs counted as:
    - Forward: `4 · seqlen^2 · head_dim · #heads` (halved for causal mask).
    - Backward: `2.5 ×` forward FLOPs (five matmuls vs. two; Section 4.1).
  - Baselines: PyTorch standard attention; FlashAttention v1; xFormers (“cutlass” impl.); Triton FlashAttention.
- End-to-end training (Section 4.2; Table 1)
  - Models: GPT3-style with 1.3B and 2.7B parameters; contexts 2k and 8k.
  - Hardware: 8× A100 80GB SXM.
  - Metric: Training throughput in TFLOPs/s/GPU, computed using Megatron-LM’s FLOPs accounting (Section 4.2).

Main quantitative results
- Kernel-level throughput on A100 (Figs. 4–6)
  - Overall forward+backward speed (Fig. 4):
    - Head dim 64, no mask: FA-2 reaches 162–176 TFLOPs/s at 2k–16k; FA v1 reaches 104–110; PyTorch 36–46.
    - Head dim 128, no mask: FA-2 reaches 173–203; FA v1 91–110; PyTorch 53–86.
    - With causal mask, head dim 64: FA-2 reaches 140–171; FA v1 70–97; PyTorch 15–18.
    - With causal mask, head dim 128: FA-2 reaches 133–189; FA v1 69–83; PyTorch 23–34.
  - Forward-only speed (Fig. 5):
    - Up to 223–224 TFLOPs/s on A100 (no mask, head dim 128); 152–155 TFLOPs/s (no mask, head dim 64); 181–200 TFLOPs/s with causal mask, head dim 128. This corresponds to as high as 73% of peak theoretical throughput in forward (Abstract; Section 4).
  - Backward-only speed (Fig. 6):
    - Up to 187–196 TFLOPs/s (no mask, head dim 128); 160–170 TFLOPs/s (mask, head dim 64/128). Reported as up to 63% of peak in backward (Abstract; Section 4).
  - Relative improvements:
    - FA-2 is typically 1.7–3.0× faster than FA v1 and 3–10× faster than PyTorch (Section 4.1).
    - FA-2 is ~1.3–1.5× faster than Triton FA in forward and ~2× faster in backward (Section 4.1).
- Kernel-level throughput on H100 (Fig. 7)
  - Forward+backward reaches 320–338 TFLOPs/s at long sequences with head dim 128, without using Hopper-specific features like TMA or 4th-gen Tensor Cores. Authors expect a further 1.5–2× with those features (Section 4.1).
- End-to-end training throughput (Table 1)
  - Quote:
    > GPT3-2.7B, 8k context: Without FA: 80 TFLOPs/s; FA v1: 175 TFLOPs/s; FA-2: 225 TFLOPs/s (Section 4.2).
  - Across 1.3B/2.7B and 2k/8k contexts, FA-2 improves over FA v1 by up to ~1.3× and over no-FA baseline by up to ~2.8×, reaching up to 225 TFLOPs/s per A100 GPU (72% model FLOPs utilization; Table 1).

Ablations, robustness, and conditions
- The paper explicitly attributes gains to:
  - Reduced non-matmul work (Section 3.1),
  - Sequence-length parallelism (Section 3.2),
  - Warp partitioning that avoids shared-memory reductions (Section 3.3).
- Causal masking accelerates further by skipping upper-triangular blocks (Section 3.1.1).
- Block size selection is manually tuned; performance depends on avoiding register spilling and fitting within shared memory (Section 3.3).
- Conditions where gains matter most:
  - Long sequences and/or small batches/heads benefit most from the added sequence-length parallelism (Section 3.2).

Do the experiments support the claims?
- Yes, for speed on A100/H100 and for end-to-end training throughput:
  - Multiple baselines, both framework-native and optimized kernels, are included.
  - Results are consistent across head sizes and causal vs. non-causal settings.
  - The magnitude and stability of speedups across sequence lengths align with the proposed mechanisms (less non-matmul work, improved occupancy, fewer shared-memory reductions).
- What is less explored:
  - Memory usage is described analytically (linear in `N`), but runtime memory numbers are not plotted here.
  - Variance/repeatability and sensitivity to hardware/software versions are not analyzed in depth (common in systems papers but worth noting).

## 6. Limitations and Trade-offs
- Hardware specificity and portability
  - The design tightly targets NVIDIA GPUs and their memory/warp model; results may differ on other accelerators. While FA-2 uses CUTLASS and Triton concepts, portability to AMD or custom accelerators would require engineering (Section 5).
- Manual tuning
  - Block-size choices are hand-tuned per head dimension and device; an auto-tuner would reduce manual labor (Section 3.3).
- Backward pass still lags GEMM efficiency
  - Despite improvements, backward pass achieves up to ~63% of theoretical peak—still less than forward and below highly optimized GEMMs (Abstract; Figs. 5–6).
- Atomic updates in backward
  - Parallelizing by column blocks requires atomic adds to `dQ`, which can limit scalability in extreme contention scenarios (Section 3.2 “Backward pass”).
- Scope
  - FA-2 accelerates exact attention; it does not reduce the quadratic FLOPs themselves. For extremely long contexts where `N^2` FLOPs are infeasible, algorithmic changes (e.g., sparse/local attention) are still needed (Section 5).
- Hopper-specific optimizations left for future work
  - On H100, features like TMA and 4th-gen Tensor Cores are not yet used; the reported gains are thus not the absolute ceiling (Section 4.1).

## 7. Implications and Future Directions
- How this changes the landscape
  - Exact attention at near-GEMM efficiency bridges the gap between practicality and fidelity for long contexts. Training 8k–16k context models becomes cheaper and faster—“train 16k for the price of 8k” (Section 5).
- Research opportunities
  - Combine FA-2’s low-level kernel optimizations with high-level algorithmic sparsity patterns (local/dilated/block-sparse) to unlock much longer effective contexts while retaining high hardware utilization (Section 5).
  - Automated kernel autotuning (block sizes, warp counts) and compiler integration so these techniques become accessible without expert GPU programming (Section 5).
  - Extend to new data types (FP8), devices (H100 with TMA, AMD GPUs), and decoding regimes (optimize KV-cache handling beyond MQA/GQA support already present; Sections 3.1.2 and 5).
- Practical applications
  - Long-document understanding, multi-page retrieval-augmented generation, high-resolution image/video transformers, long-horizon speech/audio modeling, and code modeling with very long contexts all benefit directly from faster exact attention.
  - Training infrastructure: FA-2 is suitable for drop-in acceleration in existing Transformer stacks (e.g., GPT-style models), already delivering up to 225 TFLOPs/s per A100 GPU in training (Table 1).

In sum, FlashAttention-2 is a carefully engineered reorganization of exact attention that maximizes time spent in Tensor Core matmuls, exploits parallelism along sequence length, and trims synchronization. The result is a robust, general-purpose acceleration of attention that preserves exactness while closing much of the efficiency gap to GEMMs—substantially reducing the cost of long-context Transformers (Sections 3–4).
