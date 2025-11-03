# SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

**ArXiv:** [2411.10958](https://arxiv.org/abs/2411.10958)

## 🎯 Pitch

SageAttention2 introduces a highly efficient attention mechanism for Transformers by aggressively quantizing attention computations: it uses per-thread INT4 for QKᵀ with advanced outlier smoothing, and FP8 for PV alongside a novel two-level accumulation to repair silent precision loss in NVIDIA’s tensor cores. This design delivers up to 3–4.5× faster kernels than FlashAttention2 and xformers, matches FlashAttention3 speeds on Hopper GPUs but with much higher accuracy, and achieves negligible end-to-end metrics loss across diverse tasks, directly enabling faster and more scalable large model inference.

---

## 1. Executive Summary (2-3 sentences)
SageAttention2 proposes a faster, more accurate way to run Transformer attention by quantizing two key matrix multiplications differently and more aggressively: it computes `QKᵀ` with INT4 using per-thread scaling and thorough outlier smoothing, and computes `PV` with FP8 while repairing a hidden precision loss in NVIDIA’s FP8 tensor cores via two-level accumulation. This yields roughly 3× kernel speed over FlashAttention2 and ~4.5× over xFormers (Fig. 5), matches FlashAttention3(fp8) speed on Hopper while preserving accuracy (Table 2; Fig. 7–9), and maintains end-to-end metrics across language, image, video, and audio tasks (Table 2; Appendix Table 20).

## 2. Context and Motivation
- Problem addressed
  - Attention has quadratic cost in sequence length because it involves two matrix multiplications: `QKᵀ` (compute scores) and `PV` (apply attention to values). Even with memory-efficient kernels like FlashAttention, attention remains a major bottleneck for long contexts and large models.
  - Prior quantized attention (SageAttention) sped up `QKᵀ` with INT8 and left `PV` in FP16 with a faster FP16 accumulator on some GPUs, but:
    - INT8 matmul is much slower than INT4 on modern tensor cores (Weakness W1).
    - FP16-with-FP16-accumulation only speeds up on a narrow set of GPUs (e.g., RTX 3090/4090), not L20/L40/H100 (Weakness W2). See Table 1.

- Why this matters
  - Real workloads increasingly use very long sequences (tens to hundreds of thousands of tokens) and high-resolution/video generation, so attention cost dominates both latency and throughput. Faster attention directly lowers end-to-end latency (Fig. 1; Table 8) and compute bill, and enables longer contexts.

- Prior approaches and shortcomings
  - Structure-changing methods (sparse or linear attention) reduce complexity but change model behavior and are task/model-sensitive.
  - Kernel engineering (FlashAttention v1–v3, xformers) keeps exact attention but does not quantize computations as aggressively.
  - SageAttention quantized `QKᵀ` to INT8 and smoothed `K` to remove outliers, but cannot fully exploit INT4 tensor cores and relies on GPU-specific FP16 accumulator behavior.

- Positioning of this work
  - SageAttention2 targets the fast path on modern tensor cores by:
    - Moving `QKᵀ` to INT4 with accuracy-preserving techniques (per-thread quantization and smoothing of both `Q` and `K`).
    - Moving `PV` to FP8 across GPUs, while fixing a hidden hardware precision issue (the “FP22 accumulator”) using two-level accumulation.
  - The result is a plug-and-play, exact attention kernel with broad GPU coverage (Ada/Lovelace via `SageAttn2-4b`; Hopper via `SageAttn2-8b`) that achieves both high speed and strong end-to-end accuracy.

## 3. Technical Approach
At a high level, SageAttention2 follows FlashAttention’s block tiling and online-softmax (Eq. (1)) but replaces the two GEMMs with low-precision variants and adds numerical fixes to keep accuracy high.

Step-by-step view (also see Algorithm 1 and Fig. 3):

1) Background: FlashAttention tiling and online softmax
- Attention is computed in tiles to keep data in fast on-chip memory. For a query tile `Qi` and each key/value tile `Kj, Vj`, it computes:
  - `Sij = Qi Kjᵀ / √d`.
  - Online softmax state `(mij, lij)` and partial output `Oij` are updated per `Kj, Vj` tile (Eq. (1)).
- This keeps memory traffic low but still needs two large GEMMs per tile: `QiKjᵀ` and `Peij Vj`, where `Peij = exp(Sij − mij)` are unnormalized softmax numerators.

2) Make `QKᵀ` both faster and accurate with INT4
- Two problems block naive INT4:
  - INT4’s dynamic range is tiny (−7..7). Outliers in `Q` or `K` cause many normal values to quantize to zero, collapsing accuracy (Sec. 3.1).
  - Finer quantization (e.g., per-token) improves accuracy but makes dequantization expensive because each thread needs multiple scales (Sec. 3.2).

- Two coordinated fixes:
  a. Smooth both `Q` and `K` to remove channel-wise outliers (Eq. (2), “Smooth Q+K”).
     - Subtract the mean across the token dimension from `K` (as in SageAttention) and from each query block `Qi`.
     - Algebraic decomposition (Sec. 3.1):
       - Write `Qi = q̄i + γ(Qi)` and `Kj = k̄ + γ(Kj)`.
       - Then `QiKjᵀ = γ(Qi)γ(Kj)ᵀ + ΔSij + b`, where:
         - `ΔSij = q̄i γ(Kj)ᵀ` is a vector over columns (per-row bias that cannot be dropped),
         - `b = q̄i k̄ᵀ + γ(Qi) k̄ᵀ` is a uniform row bias that cancels under softmax and need not be computed.
       - Implementation: quantize only the smoothed parts `γ(Qi)` and `γ(Kj)` to INT4; compute `γ(Qi)γ(Kj)ᵀ` in INT4; add back `ΔSij` computed efficiently as a GEMV; skip `b`.
     - Why it helps: smoothing centers and shrinks values, so INT4’s small range is used effectively (Fig. 20). Table 4 shows average cosine similarity improves to 99.46% with `Smooth Q+K`, far above other baselines.

  b. Per-thread quantization aligned to tensor-core layouts (Sec. 3.2; Fig. 4; Appendix A.6 Fig. 18, Eq. (8)).
     - Idea: Instead of per-token (accurate but slow) or per-block (fast but coarser) scales, choose quantization groups so that each GPU thread’s accumulator tile corresponds to exactly one `Q` scale and one `K` scale.
     - How: Use the PTX `mma.m16n8k64` layout. A warp (32 threads) computes a 16×8 output tile; each thread holds a fixed 4-element pattern of this tile. By carefully grouping input rows/columns that feed those outputs, each thread can reuse a single `δQ` and a single `δK` during dequantization (Fig. 4).
     - Result: Nearly per-token accuracy with no extra dequantization overhead. Table 6 shows per-thread ≈ per-token accuracy (both ≈99.45% cosine), while Table 19 shows per-thread runs as fast as per-block and much faster than per-token.

3) Make `PV` fast and accurate with FP8 + two-level accumulation
- Why FP8 and not INT for `P˜`? `P˜ = exp(S − m)` lies in [0,1] and often contains many small values whose sum matters. INT quantization spreads levels uniformly and loses small values; FP8 (especially E4M3) keeps fine resolution near zero.
- Quantization choices (Sec. 3.3):
  - Use FP8 E4M3 for both `P˜` and `V`:
    - `P˜` uses a static scale `δP = 1/448` to map `[0,1]` to E4M3’s range `[-448, 448]`.
    - `V` uses per-channel scaling to handle channel outliers (Fig. 2).
  - Accuracy: Table 7 shows E4M3 nearly matches FP16 (cosine 99.44% vs 99.45%) and beats E5M2 and INT8.
- Hidden hardware pitfall and fix (Sec. 3.4):
  - On Ada/Hopper, the FP8 tensor-core instruction `mma(f32.f8.f8.f32)` accumulates in an “FP22-like” format (1 sign, 8 exponent, 13 mantissa bits), not full FP32. This truncation introduces extra error when many FP8 partial products are accumulated.
  - Two-level accumulation: compute `Rij = Peij Vj` in FP22 per value-block (e.g., `bk=64` keys), then add `Rij` into `Oij` in full FP32. This confines FP22 truncation to within each tile; the outer accumulation stays high precision.
  - Optional: smooth `V` by subtracting its per-channel mean and add the mean back to the final output. Because each row of `Pe` sums to 1, adding back the mean keeps correctness (Appendix A.3 & Table 10). This helps models where `V` has strong channel biases (not all models; e.g., unnecessary on Llama3.1).

4) Putting it together (Algorithm 1; Fig. 3)
- Preprocess once per block:
  - Smooth `K`; per-channel quantize `V` to FP8.
- For each query block `Qi`:
  - Smooth `Qi` to get `q̄i` and `γ(Qi)`; per-thread quantize `γ(Qi)` and each `γ(Kj)`; compute:
    - INT4 GEMM for `γ(Qi)γ(Kj)ᵀ`, dequantize with thread-local scales.
    - GEMV for `ΔSij = q̄i γ(Kj)ᵀ`.
    - Sum to form `Sij`; update online softmax state `(mij, lij)`.
    - Compute `Oij += (Peij · 448).to(FP8 E4M3) × Vj` using FP8 matmul with FP22 inner accumulation and FP32 outer accumulation.
- Finalize:
  - Normalize by `li,·`; rescale by `δV / 448`; add back optional `V` mean if used.
- Two kernel variants (Table 3):
  - `SageAttn2-4b`: INT4 per-thread for `Q, K`; FP8 per-block/per-channel for `P, V`. (Use on GPUs with INT4 tensor cores, e.g., RTX 4090.)
  - `SageAttn2-8b`: INT8 per-thread for `Q, K`; FP8 for `P, V`. (Use on Hopper, which lacks native INT4 tensor cores.)

Design choices and why
- Smooth `Q+K` rather than only `K`: INT4 needs more aggressive outlier control; smoothing both halves of `QKᵀ` materially improves accuracy (Table 4; Table 5).
- Per-thread quantization vs per-token: retains per-token-level accuracy with near-zero runtime overhead by aligning to the `mma` data layout (Fig. 4; Eq. (8); Tables 6 and 19).
- FP8 E4M3 vs E5M2/INT8 for `PV`: better small-value fidelity (Table 7).
- Two-level accumulation: necessary once discovering the FP22 accumulator (Sec. 3.4), otherwise FP8 matmul can silently degrade results.

## 4. Key Insights and Innovations
- Per-thread INT4 quantization aligned to tensor-core lanes (Fundamental)
  - What’s new: Define quantization groups so that each thread uses exactly one `δQ` and one `δK` during dequantization, matching `mma.m16n8k64` layouts (Fig. 4; Appendix A.6). This preserves accuracy like per-token but avoids its per-thread multi-scale dot products.
  - Why it matters: Delivers near per-token accuracy (Table 6) with the speed of coarse-grained quantization (Table 19), enabling INT4 `QKᵀ`.

- Smoothing both `Q` and `K` plus analytically correct bias handling (Significant)
  - What’s new: Extend SageAttention’s `K` smoothing to also smooth `Q`, decompose `QiKjᵀ` into a small-intensity INT4 GEMM plus a correction vector `ΔSij`, and provably drop the uniform row bias because softmax cancels it (Sec. 3.1; Eq. (2); Appendix A.5).
  - Why it matters: Drastically reduces INT4 quantization error in `QKᵀ` (Table 4; Table 5; Fig. 20), unlocking INT4 speed.

- Two-level accumulation to fix FP8 tensor-core “FP22” accumulator (Fundamental)
  - What’s new: Identify that `mma(f32.f8.f8.f32)` uses an FP22 accumulator (1/8/13 bits) and not full FP32, which truncates low bits. Add an FP32 outer accumulation buffer so truncation never compounds across many tiles (Sec. 3.4).
  - Why it matters: Allows safe use of FP8 (E4M3) for `PV` across GPUs, achieving 2× FP8 matmul speed (Table 1) without a hidden accuracy penalty.

- End-to-end, plug-and-play quantized attention across modalities (Incremental but impactful)
  - What’s new: A single attention drop-in works for LLMs, diffusion-based image/video generators, and audio models, with negligible metric loss (Table 2; Appendix Tables 11–13, 20).
  - Why it matters: Practical acceleration without retraining or task-specific tuning.

## 5. Experimental Analysis
- Evaluation methodology
  - Workloads: 10 representative models across modalities (Sec. 4.1), including Llama2/3.1/GLM4 (text), CogVideoX (2B and 1.5–5B), HunyuanVideo, Mochi (video), Flux and Stable-Diffusion 3.5 (image), and TIMM (image classification). Appendix adds Qwen2-Audio (speech-to-text).
  - Datasets/metrics:
    - Text: WikiText perplexity, LAMBADA accuracy, MMLU accuracy, LongBench score (Appendix A.7).
    - Video: CLIPSIM, CLIP-Temp, VQA-a (aesthetics), VQA-t (technical), Flow-score (temporal consistency).
    - Image: FID, sFID, CLIPScore, ImageReward.
    - Audio: Librispeech WER (Appendix Table 20).
  - Baselines: SmoothAttn (SmoothQuant α=0.5 on Q,K), HadmdAttn (random Hadamard on Q,K + INT4), SageAttention (INT8 QK + FP16 PV), FlashAttention3(fp8) on Hopper (Sec. 4.1).
  - Kernel speed: batch 4, 32 heads, head-dim 64/128, causal/non-causal, various sequence lengths, across RTX4090, L20, H100, H20 (Sec. 4.2; Fig. 5 and Appendix Figs. 10–16).

- Main quantitative results
  - Kernel speed
    - On RTX4090 (head-d=128), Fig. 5: `SageAttn2-4b` reaches up to 481 TOPS non-causal and ~479 TOPS causal; ~3× vs FlashAttention2 and ~4.5× vs xFormers.
    - Cross-GPU summary (Appendix Table 9): SageAttention2 is 2.46–3.12× faster than FlashAttention2 on L20/H100/H20; FlashAttention3(fp8) is 2.63–3.06× on Hopper but not available elsewhere.
  - End-to-end latency (Table 8)
    - CogVideoX (1.5–5B) on RTX4090: original 1040 s → 577 s (`SageAttn2-8b`) and 555 s (`SageAttn2-4b`), “1.8× speedup” highlighted in Fig. 1.
    - Llama3.1 first-token latency: 9.2 s → 5.7 s (4090, 48k tokens), 39.9 s → 23.2–25.4 s (L20, 100k tokens).
  - End-to-end accuracy (Table 2; plus Appendix Tables 11–13, 20)
    - Llama3.1: `SageAttn2-8b` essentially matches full precision on WikiText/Lambda/MMLU/LongBench; `SageAttn2-4b` has small drops (e.g., MMLU 0.607 vs 0.635).
    - Video: on HunyuanVideo and Mochi, `SageAttn2-8b` tracks full precision closely, while FlashAttention3(fp8) degrades VQA metrics noticeably (Table 2; Fig. 9).
    - Image: Flux and SD3.5 metrics are on par or slightly better/worse within typical evaluation noise (Table 2).
    - Audio: Qwen2-Audio WER near full precision and better than other quantized baselines (Appendix Table 20).
  - Long-context robustness:
    - Needle-in-a-Haystack and InfiniBench on Llama-3-8B-262k (H100): `SageAttention2-8b` matches full precision, whereas FlashAttention3(fp8) shows accuracy drops (Table 14 and Fig. 19).

- Ablations and diagnostics
  - Smoothing effectiveness (Tables 4, 5, 17): `Smooth Q+K` clearly outperforms only smoothing `K` or baseline transforms; without smoothing, INT4 collapses (Table 5 shows 72.6%→80.8% Lambda and large perplexity gain).
  - Quantization granularity (Tables 6 and 15): Per-thread ≈ per-token accuracy, far better than per-block/tensor. Worst-case layer accuracy still holds for per-thread (Table 15).
  - Datatype for `PV` (Tables 7 and 16): FP8 E4M3 ≈ FP16; E5M2 and INT8 degrade.
  - Overhead of added techniques (Appendix Table 18): per-thread quantization +0.35% slowdown, two-level accumulation 0%, smoothing Q ~3.7%. Net benefits dominate.

- Do the experiments support the claims?
  - Yes, the kernel speedups are consistent across GPUs and head-dims (Fig. 5; Appendix Figs. 10–16; Table 9). Accuracy is validated at both kernel level (cosine/L1/RMSE) and end-to-end metrics across many models (Table 2; Appendices). The Hopper comparison isolates bit-width and shows SageAttention2’s accuracy advantage over FlashAttention3(fp8) at similar speeds (Table 14; Fig. 7–9).

- Conditional results and trade-offs
  - `SageAttn2-4b` (INT4) is the fastest but can show small accuracy drops in some LLM metrics (Table 2); `SageAttn2-8b` matches accuracy better while still boosting speed.
  - Optional `V` smoothing helps when `V` has channel biases (Appendix A.3/A.4), common in some diffusion/video models but not necessary in all (e.g., Llama3.1).

## 6. Limitations and Trade-offs
- Hardware assumptions and portability
  - INT4 tensor-core support is not universal (e.g., Hopper lacks native INT4). Hence the need for `SageAttn2-8b` on Hopper, which is slightly slower than INT4 on Ada (Sec. 3; Table 3; Fig. 5 vs Hopper plots).
  - Per-thread quantization relies on specific `mma` tile shapes and data layouts (Appendix A.6). Porting to different tensor-core shapes or other vendors requires re-deriving the grouping.

- Data distribution assumptions
  - The smoothing strategy leverages token-similarity structure in attention (Fig. 2) and assumes that subtracting means significantly reduces outliers (Appendix A.5). If a model/layer produces highly non-stationary activations with little shared structure, benefits could diminish.

- Complexity and engineering burden
  - The full pipeline combines smoothing, fused quantize/dequantize, tile-wise corrections (`ΔS` GEMV), per-thread scalings, FP8 matmuls, and two-level accumulation (Algorithm 1). This increases kernel complexity and may be harder to maintain than simpler FP16/FP8 kernels.

- Residual accuracy sensitivity at extreme settings
  - While robust in reported tests, extremely long sequences, unusual head dimensions, or pathological activation distributions could expose edge cases in scale selection or FP22 truncation behavior. Optional `V` smoothing is not universally beneficial and requires inspection (Appendix A.3/A.4).

- Evaluation scope
  - Speed benchmarks use synthetic random inputs for fairness/consistency (Appendix A.8). Although paired with extensive end-to-end tests, real-world deployment patterns (batching, KV cache behavior, streaming modes) can vary and might affect realized speedups.

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that exact attention can be aggressively quantized (INT4/FP8) with negligible accuracy loss if one addresses outliers (smoothing), granularity (per-thread scales), and hardware quirks (FP22 accumulator). This sets a template for precision-aware kernel design rather than model changes.

- Practical applications
  - Immediate drop-in acceleration for:
    - Long-context LLM serving and RAG systems where prefill dominates latency.
    - High-resolution image and video generation where attention is the main bottleneck (Table 8; Fig. 6–9).
    - Audio ASR/AV models relying on attention (Appendix Table 20).
  - Benefits cloud cost and latency SLAs, enabling longer contexts (100k–262k) at manageable latency (Table 8; Fig. 19).

- Research directions
  - Generalizing per-thread quantization beyond `mma.m16n8k64`, to other matmul shapes and vendors (AMD/Intel/NPU).
  - Adaptive smoothing: learn or predict optimal smoothing statistics per layer/head, or detect when `V` smoothing helps.
  - Mixed-precision schedulers: dynamically choose INT4/INT8 per layer or per head based on runtime activation stats.
  - Training-time compatibility: explore whether training with these quantization paths (or lightweight fine-tuning) can fully eliminate the small residual gaps of `SageAttn2-4b`.
  - Formal analysis of FP22-like accumulators across future architectures and standardized APIs to expose accumulator precision.

- System-level integration
  - Combine with KV-cache compression, paging, and streaming pipelines; pair with long-context data pipelines to push context windows further without throughput loss.
  - Expose these kernels via widely used inference stacks to ease adoption (the paper provides code at the linked repository).

> Representative highlights:
> - “Peak performance of 481 TOPS on RTX4090… ≈3× vs FlashAttention2 and ≈4.5× vs xformers” (Fig. 5; Sec. Performance).
> - “SageAttn2-8b matches the speed of FlashAttention3(fp8) on Hopper while delivering much better accuracy” (Abstract; Table 14; Fig. 7–9).
> - “Smoothing Q+K… Cosine similarity 99.46% vs baselines” (Table 4); “Per-thread quantization ≈ per-token accuracy with no speed loss” (Tables 6, 19).
> - “Two-level accumulation fixes FP22 accumulator error in FP8 matmul” (Sec. 3.4; Algorithm 1).
