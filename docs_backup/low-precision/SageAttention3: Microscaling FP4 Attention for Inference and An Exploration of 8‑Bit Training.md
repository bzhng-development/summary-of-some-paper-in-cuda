# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8‑Bit Training

**ArXiv:** [2505.11594](https://arxiv.org/abs/2505.11594)
**Authors:** Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen
**Institutions:** Tsinghua University

## 🎯 Pitch

SageAttention3 pioneers the use of 4-bit floating point Transformer attention on NVIDIA Blackwell GPUs, achieving a remarkable 5x speedup over existing methods without compromising accuracy. This breakthrough significantly reduces inference latency in generative models and enables faster training, setting a new standard for efficient, scalable AI computation.

---

## 1. Executive Summary
SageAttention3 introduces a practical way to run Transformer attention in 4‑bit floating point on NVIDIA Blackwell GPUs and reaches up to 1038 TOPS on an RTX5090, delivering about 5× kernel speedup over the fastest available FlashAttention baseline for that card while preserving end‑to‑end quality (Figure 1, Figures 4–5, Table 2). The paper also proposes SageBwd, an 8‑bit forward+backward attention for training that matches BF16 accuracy in fine‑tuning while giving 1.2–1.7× kernel speedups on RTX4090, though it converges more slowly in pretraining (Figures 6–8, Table 3).

## 2. Context and Motivation
- Problem addressed
  - Attention is expensive at long sequence lengths due to quadratic cost in sequence length. Even IO‑optimized kernels (e.g., FlashAttention) remain compute‑heavy at scale (Section 1).
  - New NVIDIA Blackwell GPUs provide FP4 Tensor Cores that promise much higher throughput than FP16 (Section 1; NVIDIA doc [4]). The challenge is to use FP4 without hurting accuracy, and to do so in a plug‑and‑play way across models.
  - No prior work has made low‑bit attention usable for training large models end‑to‑end; existing low‑bit attention (FlashAttention3 FP8, SageAttention INT8) targets inference only (Section 1, Related Work Section 6).

- Why it matters
  - Real‑world: Faster attention directly reduces inference latency for text, image, and video generators; the paper shows large end‑to‑end wins for video generation (Figure 1; Table 4a).
  - Training: Attention dominates step time for long‑context training. Any safe low‑bit training method can reduce cost or increase batch size.

- Gaps in prior approaches
  - FlashAttention2/3 provide speed via IO‑awareness and kernel engineering, not via FP4 on Blackwell for general models or for backward pass (Section 6).
  - SageAttention and SageAttention2 quantize to 8‑bit for inference but not to 4‑bit, and not for training (Section 6).
  - FP4 is hard because:
    - C1: FP4 values are extremely coarse (NVFP4 has only 15 distinct non‑zero magnitudes), so naïve per‑tensor or per‑token quantization loses accuracy (Section 1).
    - C2: The attention map `P` mostly contains small numbers in [0, 1]; its scaling factors end up tiny and must themselves be stored in FP8 (`E4M3` format), causing rounding error (Section 1; Section 3.2; Figure 3).
    - C3: During training, the gradient path through `P` is fragile; quantization noise in `dOVᵀ` corrupts `dP` and then `dS`, `dQ`, `dK` along the backward scan (Section 1; Section 4.2, Algorithm 3).

- Positioning
  - The paper contributes the first FP4 attention implementation that is both accurate and “plug‑and‑play” across several real models for inference (Section 3; Table 2; Figure 9). It also presents the first low‑bit attention that supports training by quantizing most backward matmuls to 8‑bit while strategically keeping the most sensitive one in FP16 (Section 4; Table 1c).

## 3. Technical Approach
The paper delivers two components: FP4 inference attention (SageAttention3) and 8‑bit trainable attention (SageBwd). Both are built on the FlashAttention tiling scheme and online softmax.

Background (Section 2):
- FlashAttention tiles query `Q` into blocks `Q_i` of size `Bq × D` and keys/values `K_j`, `V_j` into blocks `Bkv × D`. It computes submatrices `S_ij = Q_i K_jᵀ`, then applies an “online softmax” that maintains running per‑row maxima and partition functions so `S`/`P` never need to be materialized in full (Section 2; lines 7–13 of Algorithm 1 and Algorithm 2).

Key terms used below:
- `microscaling quantization`: group‑wise scaling at very fine granularity. Here each 1×16 group within a matrix row shares one scale `s` (Section 3.1).
- `NVFP4`: NVIDIA’s FP4 microscaling format: data in 4‑bit (`E2M1`), 1×16 groups, with group scale factors stored in FP8 `E4M3` (Section 3.1 “Data type determination”).
- `FP4MM`: the hardware FP4 matrix multiply that accepts low‑bit blocks plus their scale matrices, reconstructs them on the fly, and accumulates in FP32 (Equation 3).
- `online softmax`: streaming softmax that updates row maxima and sums as you scan blocks, avoiding large intermediate tensors (Algorithm 1 lines 9–13).

### 3.1 FP4 attention for inference (SageAttention3)
Goal: Quantize the two attention matmuls `QKᵀ` and `PV` into NVFP4 while preserving accuracy.

Step-by-step (Section 3.1 and Algorithm 1):
1. Smoothing (accuracy trick reused from SageAttention2; Algorithm 1 lines 2, 5, 8)
   - Subtract the mean of `K` globally (“smoothing K”).
   - For each `Q_i` tile, subtract its row mean `q̄_i` and compensate by a small extra `GEMV(q̄_i, K_jᵀ)` term when forming `S_ij`. This reduces outliers before quantization so FP4 can represent values with less clipping.

2. Microscaling NVFP4 quantization operator `ϕ` (Equations 1–2)
   - Partition each matrix `X` row into 1×16 groups `X_ij`.
   - Set one FP8 scale per group: `s_ij = max(|X_ij|)/6`. The constant `6` is the max representable magnitude of NVFP4’s `E2M1` value before scaling.
   - Store `X̂_ij = round_to_FP4(X_ij / s_ij)`.
   - Dequantize on-the-fly during matmul via `FP4MM(Â, s_A, B̂, s_B)` (Equation 3).

3. Apply FP4 in attention (Equation 4)
   - Quantize `Q` and `K` then compute `S = FP4MM(Q̂, s_Q, K̂, s_K) + GEMV(q̄, Kᵀ)` for the smoothing correction (Algorithm 1 line 8).
   - Compute `P_e = OnlineSoftmax(S)` (Algorithm 1 line 9).
   - Quantize `P_e` and `V` then compute `O = FP4MM(P̂, s_P, V̂, s_V)`, with a special two‑level scaling for `P_e` (below) that also appears in Algorithm 1 lines 10–11.

Why NVFP4 over MXFP4?  
Both are FP4 microscaling formats but MXFP4 uses 1×32 groups with `E8M0` (integer‑like) scales. Section 3.1 shows NVFP4 yields much better accuracy on real attention tensors (Table 1a): cosine similarity 99.52% for NVFP4 vs 98.37% for MXFP4 across layers of CogVideoX.

### 3.2 Two‑level scaling for the attention map `P_e` (Section 3.2, Equation 5)
Problem (C2): The attention probabilities `P_e` live in [0, 1]. In NVFP4, the per‑group scale `s_P` must be stored in FP8 `E4M3`. If you directly compute `s_P = max(P_e group)/6`, most `s_P` lie between 0 and 0.167. That narrow low range is poorly represented by `E4M3`, so rounding errors in `s_P` become large relative to its value, degrading `P_e` quantization (Figure 3a–b; 3d–e).

Two‑level solution (Algorithm 1 lines 10–11; Equation 5):
- Level 1 (per‑token/row normalization): Scale each row of `P_e` up so its dynamic range becomes roughly `[0, 448×6]`. This uses an FP32 factor `s_P1 = rowmax(P_e) / (448×6)` and computes `P_e2 = P_e / s_P1`.
- Level 2 (microscaling NVFP4): Quantize `P_e2` with 1×16 microscaling to get `P̂_2, s_P2`.
- Compute `O` with `FP4MM(P̂_2, s_P2, V̂, s_V)` and then rescale by `s_P1` outside the low‑bit core.

Why it works: Upscaling the row first moves subsequent NVFP4 per‑group scales `s_P2` into a healthy range of `E4M3`, dramatically reducing scale representation error (Figure 3c–e). In ablation (Table 1b) cosine similarity jumps from 93.32% (direct quant) to 99.52% (two‑level).

Intuition: Think of `E4M3` as a ruler that has many ticks near 1 but very sparse ticks near 0. Two‑level scaling moves the scale factors closer to 1 before storing them in `E4M3`, so they can be represented precisely.

### 3.3 Blackwell‑specific kernel engineering (Section 3.3; Figures 19–21)
To achieve peak throughput, the paper modifies dataflow to respect register layouts of Blackwell’s FP4 MMAs:
- Permutation for `K` (Section 3.3 “Permutation for K”)
  - FP4 MMA produces FP32 accumulators in a different register layout than one of its operands (Figures 19 vs 20). Instead of expensive thread shuffles, they permute the columns of the `P` tile and correspondingly rearrange `K`’s columns during quantization so the accumulator’s layout matches the next operation (Figure 21). This avoids extra shuffles.

- Reuse shuffle (Section 3.3 “Reuse shuffle”)
  - Both microscaling quantization of `P_e` and online softmax need per‑row maxima. They fuse the “max of 16 items” reduction so it’s computed once and reused, cutting shuffles and max ops by ~50% and giving about 10% kernel speedup overall.

- Producer‑warp epilogue (Section 3.3)
  - Standard warp‑specialized kernels let consumer warps do both matmul and global stores, but FP4 register pressure makes that infeasible. They instead overlap stages by letting producer warps alternately load inputs and store outputs (“ping‑pong” at producer side), while consumer warps only move accumulators to shared memory. This overlaps MMA with global stores under tight registers.

### 3.4 8‑bit attention for training (SageBwd, Section 4; Algorithms 2–3)
Goal: Quantize as much of attention forward+backward as possible to INT8 without hurting gradients.

Quantization basics used here (Section 2 “Quantization”; Equation 7):
- Per‑block INT8 quantization for a FlashAttention tile `X`: `s_X = max(|X|)/127`, `X̂ = round(X/s_X)`. Dequantization multiplies by `s_X` at use time.

Forward (Algorithm 2):
- Quantize `Q`, `K`, `V` per block to INT8 (line 3).
- Compute `S_ij = MM(Q̂_i, K̂_j) × s_Q × s_K` (line 7).
- Online softmax yields per‑row maxima `m_ij` and partial sums `l_ij` (line 8).
- Crucially, `P_e` is quantized per‑token (row) instead of per‑block (line 9): they set `s_P = exp(rowmax(S_ij) − m_ij) / 127`, which reuses the online‑softmax maxima already computed, avoiding extra max reductions.
- Compute `O_ij = previous + MM(P̂_ij, V̂_j) × s_P × s_V` (line 10). The forward also returns `L_i` (log partition terms) needed for backward (line 13).

Backward (Algorithm 3; Equation 8 dependencies):
- There are five matmuls in the backward of attention: `dV = P_eᵀ dO`, `dP = dO Vᵀ`, `dS = P_e ∘ (dP − D)` where `D = rowsum(dO ∘ O)`, then `dQ = dS K`, `dK = dSᵀ Q`.
- Core design choice for stability (Section 4.2):
  - Keep `dP = dO Vᵀ` in FP16 (Algorithm 3 line 8) because quantizing it to INT8 injects noise into `dP` and then into `dS`, which is repeatedly accumulated along sequence blocks (the backward scan), amplifying errors for long sequences. Table 1c shows cosine similarity for `dQ` improves from 97.47% (when quantizing `dOVᵀ`) to 99.77% when `dOVᵀ` stays FP16.
  - Quantize the other four backward matmuls with per‑block INT8 (lines 7, 9–11).

Summary: SageBwd quantizes 6 of the 7 total matmuls across forward+backward (2 forward + 5 backward), leaving only `dOVᵀ` in FP16 for accuracy.

## 4. Key Insights and Innovations
- FP4 microscaling with 1×16 groups for attention matmuls (Section 3.1; Equation 4)
  - Novelty: Applies NVFP4 (E2M1 data + FP8 E4M3 scales) to both `QKᵀ` and `PV` in a full FlashAttention pipeline, including smoothing for `Q`/`K`.
  - Significance: Yields very high throughput on Blackwell (up to 1038 TOPS, Figures 4–5) with minimal quality loss across diverse generative models (Table 2).

- Two‑level scaling for `P_e` to make FP8 scales accurate (Section 3.2; Equation 5; Figure 3)
  - Novelty: Instead of directly microscaling `P_e`, first rescale each row by an FP32 factor so the subsequent FP8 scales live in a well‑represented magnitude range.
  - Significance: Dramatically reduces both the representation error of FP8 scales and the overall quantization error of `P_e` (Figure 3d–e). Ablation: cosine similarity improves from 93.32% to 99.52% (Table 1b).

- Backward‑pass sensitivity analysis and selective precision (Section 4.2; Algorithm 3; Table 1c)
  - Novelty: Identifies `dOVᵀ` as the single most accuracy‑critical matmul in backward attention and keeps it in FP16 while quantizing the rest to INT8.
  - Significance: Achieves near‑BF16 training accuracy in fine‑tuning with substantial speedups, avoiding the degradation seen when quantizing all matmuls (Table 1c, Table 3).

- Blackwell‑aware kernel engineering (Section 3.3; Figures 19–21)
  - Novelty: Permutes operand layouts to match accumulator registers, fuses reductions to reuse shuffles, and introduces a producer‑warp epilogue.
  - Significance: Bridges the gap between theoretical FP4 throughput and realized kernel performance (≈10% extra speed from shuffle reuse; overall 4–5× vs FlashAttention2 in Figures 4–5).

These are fundamental engineering innovations rather than small parameter tweaks. The two‑level scaling for `P_e` and the selective‑precision backward are conceptually simple but have outsized impact on making 4‑bit/8‑bit attention practical.

## 5. Experimental Analysis
- Evaluation setup (Section 5.1 and Appendix A.3)
  - Models
    - Text: `Qwen2.5` and `Llama3.2` (fine‑tuning).
    - Video: `CogVideoX`, `HunyuanVideo`, `Mochi`.
    - Image: `Flux`, `Stable‑Diffusion 3.5`.
  - Datasets and metrics
    - Text: GSM8K, DROP, MMLU, HellaSwag; report accuracy/F1 (Appendix A.3).
    - Video: CLIPSIM, CLIP‑Temp, VQA‑aesthetic, VQA‑technical, Flow‑score (Table 2; Appendix A.3).
    - Image: FID, sFID, CLIPScore, ImageReward (Table 2; Appendix A.3).
  - Baselines
    - PyTorch naive attention, xFormers kernels, FlashAttention2 CUDA; FlashAttention3 is not supported on RTX5090, so FA2 is the fastest baseline on that card (Figure 1 caption; Section 5.1).
  - Hardware and implementation
    - SageAttention3: CUDA + CUTLASS; SageBwd: Triton (Section 5.1 “Implementation”).
    - Throughput measured in TOPS (tera‑ops/s). Head dims 64 and 128; causal and non‑causal variants tested (Figures 4–7).

- Main quantitative results
  - Kernel throughput for FP4 inference (Figures 4–5 on RTX5090)
    - Head dim 128, non‑causal: SageAttention3 reaches 964–1038 TOPS for seq 1K–32K, while FlashAttention2 sits around 173–215 TOPS; ≈4.5–5× speedup.
    - Head dim 64, similar trends: SageAttention3 751–839 TOPS vs FA2 191–220 TOPS; ≈3.5–4.2× speedup.
  - End‑to‑end inference latency (Table 4a; Figure 1)
    - HunyuanVideo on RTX5090: 489 s → 164 s (≈3× faster).
    - CogVideoX (2B): 64 s → 27 s (≈2.4× faster).
  - End‑to‑end quality (Table 2; Figure 9; Appendix Figures 10–14)
    - Across video models (CogVideoX, HunyuanVideo, Mochi) and image models (Flux, SD3.5), FP4 SageAttention3 matches or very slightly changes metrics relative to full precision and SageAttention2 (8‑bit). Example:
      - CogVideoX: CLIPSIM 0.1865 (FP16) vs 0.1881 (FP4); VQA‑t 69.875 (FP16) vs 70.364 (FP4).
      - Flux: FID 162.812 (FP16) vs 162.121 (FP4); sFID 146.980 vs 142.839 (lower is better).
    - Qualitative samples show visually indistinguishable results (Figure 9, Appendix A.1).
  - Training kernel throughput (Figures 6–7; Appendix Figures 15–18) on RTX4090
    - Head dim 128: forward+backward speed up to ≈214–224 TOPS for causal/non‑causal; FlashAttention2 is ≈130–151 TOPS. Gains ≈1.4–1.7×.
    - Head dim 64: forward+backward ≈248–265 TOPS vs FA2 ≈185–217 TOPS (≈1.2–1.4×).
  - End‑to‑end training step time (Table 4b)
    - Llama (8K tokens): 2.1 s → 1.9 s (~1.1–1.2×).
    - Llama (16K): 6.0 s → 5.2 s (~1.15×).
  - Training accuracy (fine‑tuning) (Figure 8b–e; Table 3; Appendix Tables 5–10)
    - Loss curves closely match BF16 for Qwen2.5 (1.5B, 3B) and Llama3.2 (1B) on GSM8K, DROP, MMLU, HellaSwag.
    - Final metrics are essentially equal; e.g., Qwen2.5‑3B MMLU 0.640 (BF16) vs 0.653 (SageBwd); GSM8K 0.601 vs 0.607 (Table 3). Multiple seeds confirm consistency (Appendix Tables 5–10).
  - Training accuracy (pretraining) (Figure 8a)
    - On FineWeb‑Edu with a 400M Llama, loss decreases but converges more slowly than BF16, limiting usefulness for pretraining.

- Ablations and diagnostics
  - NVFP4 vs MXFP4 (Table 1a): NVFP4 clearly better for attention tensors (cosine 99.52% vs 98.37%).
  - Direct vs two‑level `P_e` scaling (Table 1b; Figure 3): two‑level eliminates most scale and value error.
  - Quantizing `dOVᵀ` (Table 1c; Algorithm 3): keeping it FP16 recovers gradient accuracy (cosine 99.77%).
  - Kernel micro‑optimizations improve utilization (Section 3.3), including a measured ≈10% gain from the shuffle‑reuse fusion.

- Do experiments support the claims?
  - Inference: Yes. The paper demonstrates both high kernel throughput and clear end‑to‑end latency reductions with matched quality across multiple, heterogeneous generation models (Figures 4–5, Table 4a, Table 2).
  - Training: Yes for fine‑tuning; the selective‑precision design avoids gradient drift and achieves the same final metrics (Figure 8b–e, Table 3). The pretraining caveat is explicitly shown (Figure 8a).

## 6. Limitations and Trade-offs
- Hardware specificity
  - SageAttention3’s best gains rely on Blackwell’s FP4 Tensor Cores and NVFP4/E4M3 support (Section 3.1). The reported 5× speedups are on RTX5090; Hopper cards don’t run this path, and FlashAttention3 (which uses FP8) is not a suitable baseline on 5090 (Figure 1 caption).

- Scale/format assumptions
  - Two‑level scaling for `P_e` assumes the per‑row upscaling constant (the “448×6” factor in Algorithm 1 line 10) matches the tile sizes and FP4 block granularity. Porting to different tile shapes may require retuning this normalization range (Section 3.2).

- Training coverage
  - SageBwd leaves `dOVᵀ` in FP16. This reduces the theoretical maximum speedup for backward pass (Section 4.2; Table 1c).
  - Pretraining convergence is slower than BF16 (Figure 8a). The method is currently best suited for fine‑tuning, not large‑scale pretraining.

- Implementation complexity
  - Kernel tricks such as operand permutation and producer‑warp epilogue (Section 3.3, Figures 19–21) increase engineering complexity and are sensitive to GPU ISA details and register pressure.

- Memory and precision interactions
  - While FP4 reduces compute cost, scale matrices (`s_Q`, `s_K`, `s_V`, `s_P2`) are FP8 and maintained per 1×16 group; there is still overhead to compute/store scales and to run quantization kernels. The paper offsets this with fused reductions (Section 3.3), but the balance may vary with head sizes and sequence lengths.

## 7. Implications and Future Directions
- Impact on the field
  - SageAttention3 shows that 4‑bit floating attention can be practical and drop‑in for high‑end generative models, not just toy benchmarks. This likely sets a new default for inference kernels on Blackwell‑class GPUs when quality must be maintained.
  - SageBwd opens a path for low‑bit training that does not require end‑to‑end quantized training but still delivers meaningful speedups in fine‑tuning.

- What this enables
  - Faster, cheaper video and image generation at long contexts without architectural changes (Table 4a, Figure 9).
  - Larger effective context or batch sizes at the same training budget for fine‑tuning workloads (Figures 6–7, Table 4b).

- Follow‑up research directions suggested by the paper
  - Improve low‑bit pretraining: investigate gradient‑aware quantizers, error‑feedback, or mixed‑precision schedules to fix the slower convergence observed in Figure 8a (Section 7 “Future Work”).
  - Kernel optimization headroom: the authors note a gap between current and theoretical speed for SageBwd’s Triton kernels; further CUDA/CUTLASS implementations could close it (Section 7).
  - Extend two‑level scaling beyond `P_e`: similar FP8‑scale range‑expansion may help other softmax‑like or normalization‑heavy operators.
  - Portability: study how to adapt FP4 microscaling and the 2‑level `P_e` scheme to other accelerators or to custom ASICs; analyze how tile sizes and group granularity affect the 448×6 choice.

- Practical applications
  - Plug‑and‑play acceleration for popular open models: text‑to‑video (HunyuanVideo, CogVideoX), text‑to‑image (Flux, SD3.5), and LLM inference, with negligible quality impact (Table 2).
  - Fine‑tuning of compact LLMs in resource‑constrained settings where small but safe speedups per step accumulate into substantial training‑time savings (Table 4b; Table 3).

> Representative headline numbers:
> - “Our FP4 attention … achieves 1038 TOPS on RTX5090, which is a 5× speedup than FlashAttention” (Abstract; Figures 4–5; Figure 1).
> - End‑to‑end video generation latency: “490 s → 164 s” for HunyuanVideo on RTX5090 (Figure 1; Table 4a).
> - Fine‑tuning accuracy matches BF16 across GSM8K, DROP, MMLU, HellaSwag (Figure 8b–e; Table 3).
> - Pretraining converges slower with 8‑bit attention (Figure 8a).

Overall, SageAttention3 and SageBwd combine careful quantization design (microscaling and two‑level scaling), selective precision in backward, and low‑level GPU optimizations to make FP4 inference and 8‑bit fine‑tuning practical today on Blackwell‑class hardware.
