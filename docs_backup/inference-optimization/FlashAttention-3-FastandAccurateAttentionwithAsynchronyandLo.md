# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

**ArXiv:** [2407.08608](https://arxiv.org/abs/2407.08608)

## 🎯 Pitch

FlashAttention-3 introduces a breakthrough GPU attention kernel and algorithm that leverages hardware asynchrony and low-precision arithmetic (including FP8) to dramatically accelerate Transformer attention without sacrificing accuracy. By exploiting modern GPU features, it achieves up to 2× higher throughput and 2.6× lower numerical error (for FP8) compared to previous approaches, making it pivotal for enabling long-context and high-resolution AI applications. This advancement sets a new bar for efficient large language models and paves the way for broader adoption of long-context Transformers in real-world tasks.

---

## 1. Executive Summary
FlashAttention-3 is a new GPU kernel and algorithm for exact Transformer attention that exploits hardware asynchrony and low-precision arithmetic to dramatically increase throughput without sacrificing accuracy. On NVIDIA H100 GPUs it delivers 1.5–2.0× faster FP16 attention than FlashAttention-2 (up to 740 TFLOPs/s) and approaches 1.2 PFLOPs/s with FP8 while reducing FP8 numerical error by 2.6× versus a common FP8 baseline (Abstract; Fig. 5, Fig. 7; Table 3).

## 2. Context and Motivation
- Problem addressed:
  - Attention is the primary bottleneck in Transformers due to quadratic scaling with sequence length. Speeding it up enables long-context and high-resolution applications (Introduction §1).
  - FlashAttention-2 already eliminates slow global-memory reads/writes by fusing attention into a single kernel and parallelizing along sequence length (Background §2.3; Introduction; [15]). However, on newer GPUs (H100) it utilizes hardware poorly: ~35% utilization compared to 80–90% for optimized GEMMs (Introduction).
- Why this matters:
  - Unlocks long-context LLMs and multimodal models (text, image, audio, video), real applications like long-history chat and long-horizon agents (§1).
  - Modern GPUs provide new capabilities—Tensor Core and memory engines that operate asynchronously, and fast low-precision formats (FP8) (§2.2). Previous attention kernels largely ignore these.
- Shortcomings of prior approaches:
  - FlashAttention-2 assumes a simplified, mostly synchronous model and does not explicitly overlap compute and memory or exploit FP8 (§1).
  - Library implementations and prototypes (e.g., cuDNN 9 and ThunderKittens) demonstrate Hopper-specific speedups but do not provide an integrated, open algorithm that simultaneously exploits asynchrony and low precision for attention (§1).
- Positioning:
  - FlashAttention-3 re-designs exact attention around Hopper’s asynchrony and FP8 hardware, introducing new software pipelining and quantization strategies (§3). It is intended as a drop-in attention primitive for training and inference (Abstract, §4).

## 3. Technical Approach
At a high level, the kernel processes attention in tiles, moving data between GPU memory levels while overlapping compute with data movement. It introduces three pillars: warp-specialized producer–consumer asynchrony, overlapped GEMM–softmax within and across warpgroups, and an FP8 path with layout fixes and accuracy-preserving quantization.

Key hardware concepts (defined when first used):
- `HBM` or global memory: off-chip DRAM; large but slow (§2.2).
- `SMEM` (shared memory): fast, programmer-managed on-chip memory on each streaming multiprocessor (§2.2).
- `TMA` (Tensor Memory Accelerator): dedicated unit for asynchronous copies between HBM and SMEM (§2.2).
- `Tensor Cores` and `WGMMA`: specialized units/instructions for matrix multiply-accumulate; `WGMMA` runs asynchronously at the granularity of a warpgroup (four warps) and can consume SMEM data directly (§2.2).
- `Warp specialization`: assigning warps in a threadblock to different roles (e.g., producers for data movement, consumers for compute), enabling overlapped progress (§2.2).
- `setmaxnreg`: Hopper feature to dynamically redistribute registers between warpgroups so compute-heavy warps can hold larger tiles (§2.2).

Step-by-step forward algorithm (FP16 path; §3.1–§3.2):
1. Tiling and roles:
   - The sequence is divided into query tiles `Q_i` of size `B_r × d` and key/value tiles `K_j`, `V_j` of size `B_c × d` (§3.1).
   - Each Cooperative Thread Array (CTA) computes one output tile `O_i` from one `Q_i` (Algorithm 1).
   - Warps are split into a producer warpgroup and one or two consumer warpgroups (warp specialization). Producers only issue asynchronous TMA loads; consumers compute with WGMMA and softmax (§3.1).
2. Circular SMEM buffer:
   - A ring buffer with `s` stages holds the upcoming `K_j, V_j` tiles. Producers load tiles with TMA into stage `(j % s)`, then “commit” to notify consumers. Consumers “wait” on the needed stage and then release it after finishing with that tile (Algorithm 1, lines 6–23).
3. Local softmax with rescaling:
   - Instead of materializing the full `S = QK^T` and `P = softmax(S)` in HBM, consumers compute per-block contributions and maintain running per-row `m_i` (max) and `ℓ_i` (sum of exponentials) in FP32 for stability (Algorithm 1 lines 18–24; Background §2.3). This enables exact attention while streaming through tiles.
4. Ping-pong scheduling across warpgroups:
   - Because Tensor Cores (GEMM) and special function units (for `exp`) have very different throughputs (H100: ~989 TFLOPs FP16 GEMM vs ~3.9 TFLOPs exp; §3.1), the kernel schedules the softmax of one warpgroup during the GEMM of another (Fig. 1).
   - This is implemented via barriers so one warpgroup’s GEMM0 (QK^T for next tile) and GEMM1 (PV for previous tile) precede the other’s, driving alternating softmax-GEMM overlap (§3.1; Fig. 1).
5. Two-stage pipeline within a warpgroup:
   - Even inside one warpgroup, the kernel overlaps the WGMMA and softmax by pipelining across iterations (Algorithm 2; Fig. 2). Intuition: start computing `S_next = QK_j^T` (WGMMA) while finishing softmax on `S_cur` from the previous iteration. Then overlap `P_cur V_{j-1}` with softmax for `S_next` (Algorithm 2 lines 8–15).
   - Practicalities: ensure the compiler preserves overlap (SASS analysis in Appendix B.2 shows WGMMA interleaving with exp2/row-sum/rescale); manage register pressure due to buffering `S_next` (§3.2).

FP8 path (efficiency and accuracy; §3.3):
1. Efficiency—layout conformance:
   - FP8 `WGMMA` only supports k-major operands in SMEM, but typical `V` is d-major (contiguous in head dimension). The kernel performs an in-kernel transpose of `V` tiles in SMEM using `LDSM/STSM` (collective load/store) to satisfy FP8 k-major requirements (§3.3).
   - Accumulator layout mismatch: the FP32 accumulator layout of the first FP8 WGMMA differs from the required FP8-operand-A layout for the second WGMMA. The kernel applies a repeated byte-permute on 8-byte groups to reorder registers from `{d0, d1, d2, d3, d4, d5, d6, d7}` to `{d0, d1, d4, d5, d2, d3, d6, d7}` (Fig. 3–4 and §3.3), and pairs this with the in-kernel `V` transpose so the subsequent `P V` WGMMA is correct.
2. Accuracy—quantization strategies:
   - `Block quantization`: scale each `Q/K/V` block (size `B_r×d` or `B_c×d`) separately rather than a single scale per tensor; FA-3 already processes blocks, so the scale can be accounted for when computing `S` at no extra compute (§3.3).
   - `Incoherent processing`: multiply `Q` and `K` by the same random orthogonal transform `M` (Hadamard with random ±1 diagonals) before quantization so large “outlier” activations are spread across dimensions. Because `MM^T = I`, `QK^T = (QM)(KM)^T` and attention is unchanged (§3.3).

Backward pass with asynchrony (Appendix B.1):
- Similar warp-specialized design adds a third role: a dedicated `dQ` writer warp that atomically accumulates each CTA’s partial `dQ` to global memory to avoid stalling compute warps (Algorithm 3).

Design choices and rationale:
- Prefer warp specialization over monolithic synchronous kernels to maximize overlap (producer/consumer separation, register reallocation via `setmaxnreg`; §2.2, §3.1).
- Prefer 2-stage intra-warpgroup pipeline: gives strong overlap without excessive register pressure; a 3-stage pipeline exists but underperforms due to compiler reordering and higher register use (Appendix B.3).

## 4. Key Insights and Innovations
1. Producer–consumer asynchrony with warp specialization and a circular SMEM buffer (§3.1; Algorithm 1):
   - What’s new: clean, explicit division of warps into TMA “producers” and WGMMA “consumers” with a staged ring buffer to coordinate.
   - Why it matters: hides both memory and instruction-issue latencies by continuously feeding Tensor Cores; enables register redistribution to where it helps most.
   - Innovation level: fundamental scheduling redesign for attention on Hopper-class GPUs.

2. Overlapping softmax under asynchronous GEMMs at two levels (§3.1–§3.2; Fig. 1–2; Algorithm 2):
   - Ping-pong scheduling across warpgroups plus a 2-stage pipeline within a warpgroup.
   - Significance: mitigates the low-throughput exp/log operations by running them concurrently with GEMMs. The SASS dump (Appendix B.2) confirms that exp2/row-sum/O-rescale and the first WGMMA are interleaved.

3. FP8 attention that is both fast and accurate (§3.3):
   - Efficiency: in-kernel `V` transpose via `LDSM/STSM` and register-level accumulator permutation bridge stringent FP8 WGMMA layout constraints (Fig. 3–4).
   - Accuracy: block quantization plus incoherent processing lowers FP8 error by 2.6× vs a standard per-tensor FP8 baseline (Table 3).
   - Innovation level: combined system solution—layout fixes plus quantization methods—enables high-throughput FP8 attention without a separate preprocessing kernel.

4. Backward pass warp specialization with a `dQ` writer (Appendix B.1):
   - Avoids write-contention stalls by delegating atomic accumulation of `dQ` to a dedicated warp, allowing compute warps to proceed.

## 5. Experimental Analysis
Evaluation setup (§4, Appendix C):
- Hardware and software: H100 80GB SXM5 GPU (clock fixed to 1830 MHz), CUDA 12.3, cuDNN 9.1.1, CUTLASS 3.5, PyTorch 2.3, Triton nightly (Appendix C.1). FP16 FA-3 uses a persistent kernel; FP8 FA-3 currently does not (Discussion and footnote 10).
- Workloads: sequence lengths 512→16k (for FP8 also 4224, 8448, 16896 to align with 132 SMs), hidden dim 2048; head dim 64/128/256; batch chosen so total tokens = 16k (§4.1; Appendix C.2).
- Metrics:
  - Throughput (TFLOPs/s). FLOP accounting: forward = `4·seqlen^2·head_dim·n_heads`; causal versions divide by ~2; backward = 2.5× forward FLOPs (§4.1).
  - Accuracy: RMSE vs FP64 reference under a synthetic outlier distribution `N(0,1) + N(0,100)·Bernoulli(0.001)` (§4.3).
- Baselines: PyTorch “standard attention,” FlashAttention-2, FlashAttention-2 Triton (H100-specific instructions), and cuDNN attention (closed-source) (§4.1).

Main results (all on H100 80GB SXM5):
- FP16/BF16 forward speedups (Fig. 5):
  - Head dim 128, non-causal: FA‑3 reaches 625–648 TFLOPs/s for seqlen 4k–16k versus FA‑2 at 574–609 and cuDNN at 565–638 (Fig. 5c). Peak reported is “up to 740 TFLOPs/s” (Abstract), visible at head dim 256 non-causal reaching 746–756 TFLOPs/s (Fig. 5e).
  - Head dim 64, causal: FA‑3 420–473 TFLOPs/s at 4k–16k vs FA‑2 342–363 and cuDNN 334–388 (Fig. 5b).
  - Summary: “around 1.5–2.0× faster than FlashAttention‑2 in the forward pass,” and up to 3–16× vs standard attention (§4.1; Fig. 5).
- FP16/BF16 backward speedups (Fig. 6):
  - Head dim 128, non-causal: FA‑3 542–561 TFLOPs/s at 4k–16k vs FA‑2 484–539 and cuDNN 410–616 (Fig. 6b).
  - Head dim 64, non-causal: FA‑3 453–474 TFLOPs/s at 4k–16k vs FA‑2 395–433 and cuDNN 334–388 (Fig. 6a).
  - Summary: “1.5–1.75× faster” than FA‑2 in backward (§4.1).
- FP8 forward throughput (Fig. 7; Appendix C.2):
  - Head dim 256, non-causal: FA‑3 reaches 1122–1171 TFLOPs/s for 8k–16k, near 1.2 PFLOPs/s (Fig. 7a).
  - Causal: 960–1024 TFLOPs/s at 8k–16k (Fig. 7b).
  - Relative to cuDNN, FA‑3 is competitive; exact lead/lag depends on head dim and masking (Abstract footnote 2; Appendix C.2).
- Ablation—what creates the speedup? (Table 2; §4.2):
  - Full FA‑3 (warp specialization + 2‑stage pipelining): 661 TFLOPs/s.
  - Removing pipelining: 582 TFLOPs/s.
  - Removing warp specialization: 570 TFLOPs/s.
  - Quote: “our algorithmic improvements … lead to significant speedup, from 570 to 661 TFLOPs” (Table 2).
- Accuracy and robustness (Table 3; §4.3):
  - FP16 RMSE: standard attention `3.2e‑4`, FA‑2 `1.9e‑4`, FA‑3 `1.9e‑4`. Lower error arises because both FA‑2/3 keep softmax intermediates in FP32 (§4.3).
  - FP8 RMSE: baseline FP8 `2.4e‑2`; FA‑3 FP8 with block quantization + incoherent processing `9.1e‑3` (2.6× better). Ablations show the effect of each: “No block quant” `9.3e‑3` (small regression), “No incoherent processing” `2.4e‑2` (reverts to baseline) (Table 3).

Do the experiments support the claims?
- Throughput: Figures 5–7 and Table 2 directly quantify gains vs FA‑2, Triton, and cuDNN. Gains are more pronounced for medium/long sequences and larger head sizes; small sequences and causal FP8 show weaker lead (Fig. 7b; Discussion note 10).
- Overlap actually occurs: the SASS inspection (Appendix B.2) shows the compiler schedules exp2/row-sum and datatype conversions interleaved with the first WGMMA, matching the intended two-stage overlap.
- Accuracy: Table 3 demonstrates parity with FA‑2 in FP16 and a clear FP8 improvement attributable to the proposed quantization strategies.

## 6. Limitations and Trade-offs
- Hardware specificity:
  - The design depends on Hopper features: asynchronous `WGMMA`, `TMA`, `setmaxnreg`, and FP8 layouts (§2.2; §3.3). Portability to older GPUs is limited.
- Compiler sensitivity:
  - Overlap is partly at the mercy of NVCC scheduling. Appendix B.2 shows good two-stage overlap, but the attempted 3-stage pipeline underperforms because the compiler reorders instructions so only the first WGMMA overlaps with softmax (Appendix B.3).
- Register pressure vs. tiling:
  - Overlap requires extra buffering (e.g., keeping `S_next`), increasing per-CTA registers (§3.2). This constrains tile sizes and may reduce occupancy if not tuned carefully.
- FP8 preprocessing/overhead:
  - In-kernel `V` transpose and accumulator permutations add complexity. While they are overlapped after the first iteration (§3.3), this still costs cycles and increases code complexity.
- Small-sequence and causal FP8:
  - FA‑3 FP8 does not yet use a persistent kernel and shows weaker performance on small sequence lengths and causal masking compared to cuDNN (Discussion footnote 10).
- Quantization assumptions:
  - The FP8 error reduction relies on block quantization and incoherent processing. Extremely pathological activation distributions may still challenge FP8 even with these techniques (§3.3, §4.3).

## 7. Implications and Future Directions
- Field impact:
  - Establishes that attention can reach GEMM-like utilization by embracing hardware asynchrony and careful kernel design. This shifts the ceiling for long-context training and inference on modern GPUs (Abstract; §4.1).
  - The techniques are broadly useful for other fused operators or blocks that mix matmul with lower-throughput elementwise ops.
- Practical applications:
  - Faster training and inference for LLMs, vision-language models, and high-resolution or long-horizon tasks; improved throughput makes million-token distributed attention systems (e.g., Ring Attention) more practical (Related Work §A).
  - FP8 path provides a route to large-scale training/inference with lower precision while controlling error (Table 3).
- Research directions:
  - Persistent-kernel design for FP8 and better small-sequence performance (Discussion).
  - Deeper compiler–kernel co-design to realize 3-stage or richer overlaps (Appendix B.3).
  - Extending the approach to next-gen hardware (e.g., FP4 on Blackwell; §1) and non-NVIDIA accelerators that expose similar asynchronous primitives.
  - Studying training dynamics with low-precision attention at scale; integrating quantization-aware training with the proposed FP8 strategies (§5 Discussion).

> Headline numbers: “FP16 reaching up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s… FP8 FlashAttention‑3 achieves 2.6× lower numerical error than a baseline FP8 attention” (Abstract; supported by Fig. 5e, Fig. 7, Table 3).
