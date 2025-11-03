# SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

**ArXiv:** [2411.10958](https://arxiv.org/abs/2411.10958)
**Authors:** Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen
**Institutions:** 

## 🎯 Pitch

SageAttention2 revolutionizes attention mechanisms by leveraging per-thread INT4 and FP8 quantization, achieving up to 4.5× speedups over existing methods without sacrificing accuracy. This innovation significantly reduces latency and costs for applications like long-context LLMs and high-resolution generative models, making it a vital advancement for scalable and efficient AI deployment across modalities.

---

## 1. Executive Summary
SageAttention2 introduces a faster, accuracy-preserving way to compute attention by (i) quantizing the query/key product `QK^T` to 4-bit integers (`INT4`) using a new per-thread scheme and (ii) quantizing the softmax–value product `P·V` to 8-bit floats (`FP8`) while fixing a hidden precision pitfall in NVIDIA’s FP8 matrix multiply-accumulate. It delivers about 3× kernel speedup over FlashAttention2 and ~4.5× over xformers (Figure 5), matches FlashAttention3(fp8) speed on Hopper GPUs with substantially better accuracy (Table 2, Table 14), and maintains end-to-end quality on language, image, video, and audio tasks.

## 2. Context and Motivation
- Problem addressed
  - Attention’s two matrix multiplications, `QK^T` and `P·V`, dominate runtime with quadratic complexity in sequence length. Prior acceleration via quantization is widespread for linear layers but has struggled to keep attention both fast and accurate at very low precision.
- Why it matters
  - Long-context inference for LLMs, diffusion models, and video generators increasingly requires faster attention without degrading output quality (Introduction; Figure 1). Speedups translate into shorter latency and cheaper serving at scale.
- Prior approaches and gaps
  - FlashAttention v1–v3 improve memory locality and scheduling but still rely on relatively high-precision arithmetic (Sec. 2.1).
  - SageAttention (2025) quantized `QK^T` to `INT8` and used `FP16` for `P·V` with an `FP16` accumulator. It also smoothed `K` to curb outliers (Sec. 2.3). Two drawbacks remain (W1–W2, Introduction):
    - W1: `INT8` matmul is only half as fast as `INT4`.
    - W2: `FP16` matmul with an `FP16` accumulator only speeds up certain GPUs (e.g., RTX 4090/3090), offering limited portability (Table 1).
- Positioning of this work
  - SageAttention2 pushes precision lower, where hardware is faster, while adding accuracy-saving techniques:
    - `Q, K` to `INT4` with a new per-thread quantization and additional “smooth Q”.
    - `P, V` to `FP8` with a two-level accumulation that corrects a subtle loss from NVIDIA’s FP8 tensor core accumulator (discovered to be “FP22”, not true FP32; Sec. 3.4).

## 3. Technical Approach
This section explains how SageAttention2 changes both attention matmuls and why those changes preserve accuracy.

- Background: FlashAttention tiling and online softmax
  - Attention is tiled across tokens into blocks `Qi`, `Kj`, `Vj` (Sec. 2.1). For each query block, the algorithm iteratively multiplies by key/value blocks, performs online softmax, and accumulates normalized outputs (Eq. (1)).

- Quantization notation (Sec. 2.2)
  - A matrix `A` is quantized by a function `ψ` into low-precision `Â` with scale `δA`, so products compute as `ψ^{-1}_{δAδB}(Â B̂) ≈ A B`.
  - “Granularity” is how many elements share one scale (e.g., per-tensor, per-block, per-thread).

- Step 1 — Smooth both Q and K before `INT4`:
  - Why: `INT4` has a tiny representable range [-7, 7]. A few large “outlier” values in `Q` or `K` can force most other values to quantize to zero, destroying accuracy (Challenge C1; Sec. 3.1).
  - What “smoothing” means here: subtract a token-axis mean to shrink the dynamic range (Eq. (2))
    - `γ(Qi) = Qi − q̄i`, where `q̄i = mean(Qi)` (1×D vector, per query block)
    - `γ(Kj) = Kj − k̄`, where `k̄ = mean(K)` (1×D vector, global for all keys)
  - How it keeps correctness: decompose the product
    - `Qi K_j^T = γ(Qi) γ(Kj)^T + ΔS_ij + b`
    - `ΔS_ij = q̄i γ(Kj)^T` is a per-row correction vector; `b` is a per-row constant that cancels out after softmax (Sec. 3.1).
  - Implementation pipeline (Algorithm 1; Figure 3):
    - Preprocessing kernel (off-chip read once): smooth `Q` and `K`; per-thread quantize them; compute `ΔS` via a fused GEMV (`q̄i * γ(Kj)^T`).
    - Attention kernel (on-chip): perform `INT4` GEMM for `γ(Qi)γ(Kj)^T`, dequantize, and add the precomputed `ΔS`.

- Step 2 — Per-thread `INT4` quantization for `Q, K` (Sec. 3.2)
  - Problem with finer granularity: Per-token scales give accuracy but slow down dequantization because each GPU thread must handle many scales (vector dot of `δQ` and `δK`; Sec. 3.2).
  - Key idea: exploit the tensor-core instruction’s fragment layout so that each thread’s output fragment corresponds to one quantization scale, avoiding extra overhead.
    - For the `mma.m16n8k64` instruction, the output 16×8 tile is partitioned so each thread holds four elements (Figure 18). By grouping input rows/cols to match this layout, each thread uses one `δQ` and one `δK`.
    - Concrete groupings: within a warp, `Q_w[8k + i]` share a scale; `K_j[8k + 2i]` and `K_j[8k + 2i + 1]` share a scale (Figure 4; Eq. (8)). This yields 32 groups for `Q` and 4 for `K` in a 128×64 block—32× and 4× finer than per-block—with no dequantization-time vector dot.
  - Outcome: near per-token accuracy, near per-block speed (Table 6 vs. Table 19).

- Step 3 — `FP8` for `P·V` and fixing tensor-core accumulation (Sec. 3.3–3.4)
  - Why `FP8` for `P·V`: The unnormalized softmax factors `P̃ = exp(S − m)` are in [0, 1] and often contain many small but important contributions (Sec. 3.3). Integer quantization wastes resolution near zero. `FP8` preserves many small values.
  - Formats and scales:
    - Use `E4M3` (`4` exponent, `3` mantissa) with range ≈ [−448, +448]. Set `δP = 1/448` because `P̃ ∈ [0, 1]` (Sec. 3.3).
    - Quantize `V` per-channel (one scale per feature dimension) to handle channel-wise outliers (Figure 2).
  - Hidden precision pitfall and fix:
    - Finding: NVIDIA’s `mma(f32.f8.f8.f32)` uses an internal accumulator with 1 sign, 8 exponent, and 13 mantissa bits—effectively “FP22”—not full FP32 (Sec. 3.4). This truncates low bits when accumulating `P̃·V`.
    - Two-level accumulation (Sec. 3.4): accumulate partial `R_ij = P̃_ij V_j` in FP22 over a small key block (e.g., `bk = 64`), then add `R_ij` into the main output `O_ij` kept in true FP32. This confines FP22 error to within each block while preserving overall FP32 accuracy.
    - Optional further smoothing for `V` (Appendix A.3, Table 10): subtract per-channel mean `V̄` to center values around zero (better represented by FP22), then add `V̄` back at the very end. Helps especially in diffusion/video models that show large channel biases (Figure 17).

- Implementations (Table 3; Figures 5, 10–16)
  - `SageAttn2-4b`: `INT4` per-thread for `Q, K`; `FP8` per-block/per-channel for `P, V`.
  - `SageAttn2-8b`: `INT8` per-thread for `Q, K` (for Hopper GPUs that lack native INT4 tensor cores); `FP8` for `P, V`.
  - End-to-end pipeline is summarized in Algorithm 1.

## 4. Key Insights and Innovations
- Per-thread `INT4` quantization aligned with tensor-core layout
  - What’s new: quantization groups are defined to match the `mma.m16n8k64` fragment mapping, so each thread uses one `Q` scale and one `K` scale (Figure 4, Eq. (8), Figure 18).
  - Why it matters: achieves per-token–level accuracy without the dequantization overhead of per-token scales. Evidence: per-thread ≈ per-token accuracy (Table 6, Table 15) while keeping kernel speed essentially unchanged versus per-block (Table 19; +0.35% overhead, Table 18).

- Smoothing both `Q` and `K` before `INT4`
  - What’s new: prior SageAttention smoothed only `K`. SageAttention2 also smooths `Q` and explicitly restores the exact contribution via `ΔS` (Sec. 3.1, Eq. (2)).
  - Why it matters: dramatically reduces outlier impact in `INT4`, expanding effective resolution across non-outlier elements. Evidence: average/worst-layer accuracy jumps with `Q+K` smoothing (Table 4, Table 17) and end-to-end metrics recover from severe degradation when smoothing is absent (Table 5).

- Discovery of FP22 accumulation in FP8 tensor-core path and two-level accumulation fix
  - What’s new: identifies that `mma(f32.f8.f8.f32)` masks as FP32 but accumulates with only 13 mantissa bits (Sec. 3.4), and introduces a two-level accumulation to restore FP32-level accuracy.
  - Why it matters: enables using fast FP8 for `P·V` without accuracy erosion, especially for long sequences and generative models that sum many small terms. Evidence: `E4M3` reaches FP16-like accuracy (Table 7, Table 16) when combined with the accumulation strategy; large end-to-end accuracy advantages over FlashAttention3(fp8) on Hopper (Table 2, Table 14; Figure 7 and Figure 9).

- End-to-end, plug-and-play acceleration across modalities
  - What’s new: a single attention kernel family that accelerates LLMs, diffusion image generators, and video generators with negligible metric loss (Table 2), plus audible ASR gains (Table 20).
  - Why it matters: broadens quantized attention from synthetic microbenchmarks to production-style multimodal workloads with measurable user-facing quality.

## 5. Experimental Analysis
- Evaluation setup (Sec. 4.1; Appendix A.7–A.8)
  - Models: Llama2 (7B), Llama3.1 (8B), GLM4 (9B) for text; Flux (schnell) and Stable-Diffusion 3.5 (turbo) for images; CogVideoX (2B and 1.5–5B), HunyuanVideo, Mochi for video; TIMM for image classification; Qwen2-Audio (7B) for ASR.
  - Metrics:
    - Text: WikiText perplexity (lower is better), LAMBADA and MMLU accuracy, LongBench score.
    - Image: FID, sFID (lower), CLIPScore, ImageReward (higher).
    - Video: CLIPSIM, CLIP-Temp, VQA-a, VQA-t, FlowScore (higher).
    - Audio: WER on Librispeech test-clean/dev (lower).
  - Baselines: FlashAttention2, xformers, SmoothAttn (SmoothQuant for Q/K), HadmdAttn (random Hadamard rotation + `INT4`), SageAttention (prior INT8+FP16), FlashAttention3(fp8) on Hopper (Sec. 4.1).

- Kernel-level performance (Figures 5, 10–16; Table 9)
  - RTX 4090 (head dim 128): 
    > SageAttn2-4b reaches up to 481 TOPS and is ≈3× faster than FlashAttention2 and ≈4.5× faster than xformers across sequence lengths (Figure 5).
  - Across GPUs:
    > Average speedups summarized in Table 9 show SageAttention2 ≈2.6–3.1× vs FlashAttention2 on L20/H20/H100, while FlashAttention3(fp8) is 2.6–3.1× but Hopper-only.
  - End-to-end latency (Table 8):
    > CogVideoX (1.5–5B) on RTX4090: 1040 s → 577 s (`SageAttn2-8b`, 1.8×); → 555 s (`SageAttn2-4b`).
    > Llama3.1 (100K) on L20: 39.9 s → 25.4 s (`-8b`) → 23.2 s (`-4b`).

- Accuracy and quality (Table 2; Figures 6–9; Table 14, Figure 19)
  - Text (Llama3.1, Table 2):
    > Full-precision vs SageAttn2-8b: WikiText perplexity 6.013 vs 6.019; LAMBADA 0.815 vs 0.811; MMLU 0.635 vs 0.634; LongBench 49.40 vs 49.59. Very close.
  - Video (CogVideoX 1.5–5B, Table 2):
    > Full-precision vs SageAttn2-8b: CLIPSIM 0.1778 vs 0.1775; VQA-a 70.231 vs 69.492; FlowScore 2.507 vs 2.487. Visual comparisons show minimal differences (Figure 7).
    > FlashAttn3(fp8) degrades VQA metrics substantially (e.g., VQA-a 6.531; VQA-t 2.181), highlighting the impact of FP22 accumulation without corrective strategies.
  - Hopper super-long context (Table 14; Figure 19):
    > On Llama-3-8B-262k, SageAttention2 matches full precision across InfiniBench tasks, whereas FlashAttn3-fp8 shows clear drops (e.g., Eng.MC 64.19 vs 55.90) and NIAH retrieval failures in the heatmap.
  - Images (Flux, SD3.5; Table 2):
    > Metrics remain at parity or slightly better; e.g., Flux FID 10.960 vs 10.927 (`-8b`). 
  - Audio (Qwen2-Audio; Table 20):
    > WER test-clean 1.74 (FP) vs 1.72 (`-8b`), and dev 4.01 vs 4.03—no degradation.

- Ablations and diagnostics
  - Smoothing effectiveness (Tables 4, 5, 17):
    > For INT4 `Q, K` with FP16 `P, V`, end-to-end Llama3.1 LAMBADA rises from 72.6% (no smoothing) to 80.8% with smoothing `Q+K` (Table 5); cosine similarity across layers improves from 80.04% (none) to 99.46% (`Q+K`, Table 4).
  - Quantization granularity (Tables 6, 15, 19):
    > Per-thread ≈ per-token accuracy (CosSim 99.45% both; Table 6) but avoids per-token speed penalty (TOPS 283 per-thread vs 268 per-token; Table 19).
  - Data type for `P, V` (Tables 7, 16):
    > E4M3 nearly matches FP16 (CosSim 99.44% vs 99.45% average; Table 7). Worst-layer CosSim 96.70% (E4M3) vs 96.76% (FP16); `INT8` is unusable here (CosSim 19.52%; Table 16).
  - Overhead of added techniques (Table 18):
    > Per-thread quantization: ~0.35% TOPS change; two-level accumulation: ~0%; smoothing Q: ~3.7%—small compared to 2–3× overall speed gains.

- Overall assessment
  - The evidence is broad (text, vision, video, audio), deep (layerwise accuracy, end-to-end metrics), and hardware-diverse (RTX 4090, L20/L40/H20/H100). The ablations link mechanism to outcomes (e.g., smoothing and granularity), and the Hopper comparison with FlashAttn3 isolates the FP8 accumulation issue and its fix.

## 6. Limitations and Trade-offs
- Hardware specificity and portability
  - Per-thread scales rely on specific tensor-core fragment layouts (e.g., `mma.m16n8k64`). Alternative architectures or future instructions may require re-deriving the grouping (Figure 18; Eq. (8)).
  - Native `INT4` tensor cores are absent on Hopper; the paper provides an `INT8` variant (`SageAttn2-8b`), which is slightly slower than the `INT4` path (Table 9).
- Assumption about activation structure
  - Smoothing leverages token similarity and channel biases (Figure 2). In domains where `Q/K` do not share strong token-wise means or exhibit different outlier structures, smoothing benefits may diminish.
- Complexity vs. simplicity
  - The pipeline adds preprocessing (smoothing, `ΔS` GEMV) and a two-level accumulation strategy. While the measured overhead is small (Table 18), the implementation complexity is higher than plain FlashAttention.
- Residual accuracy–speed trade-off
  - The fastest `-4b` variant (INT4 QK) can introduce small metric drops in some tasks (e.g., Llama3.1 MMLU 0.635 → 0.607; Table 2), whereas `-8b` recovers accuracy at slightly reduced speed.
- Scope
  - The paper targets inference; training-time stability with these precisions is not addressed. The method focuses on exact attention; integration with algorithmic sparsity or linearized attention is orthogonal (Sec. 1).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that attention can run in `INT4 × FP8` regimes at production quality, not just in microbenchmarks. This unlocks low-precision attention as a practical lever for long-context LLMs and high-resolution generative models.
- Practical applications
  - Lower-latency and higher-throughput serving for chat assistants, code copilots, long-context document QA, image/video generation, and ASR—especially where latency budgets are tight or costs must be reduced (Table 8).
  - Drop-in acceleration in frameworks that already support FlashAttention-style tiling; the paper’s CUDA kernels show feasibility (Algorithm 1; Table 3).
- Research avenues
  - Generalizing per-thread quantization to other tensor-core tile shapes and to other kernels (e.g., feed-forward layers, cross-attention) for end-to-end INT4/FP8 pipelines.
  - Learned or dynamic smoothing: replacing fixed mean subtraction with learned affine transforms or lightweight adapters to further tame outliers per layer or per input.
  - Robust FP8 accumulation on future hardware: formal specs and hardware interfaces that expose accumulator precision; evaluating other FP8 formats or stochastic accumulation.
  - Combining with algorithmic sparsity (e.g., Skip-/Sparse-attention) to stack multiplicative gains (Sec. 1 cites sparse attention works). 
  - Training with these precisions (cf. SageAttention3 FP4 insights cited in references) to close the loop between train-time and serve-time quantization.

Overall, SageAttention2 is a carefully engineered combination of quantization granularity, data smoothing, and accumulation control that pushes attention into faster numeric regimes while keeping real-world quality intact. The comprehensive evaluations (Tables 2, 4–8, 10–20; Figures 5–16, 19–20) substantiate both the speed and accuracy claims and provide a clear recipe for practitioners to adopt the method across modalities and hardware.
