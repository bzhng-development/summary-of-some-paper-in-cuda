# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training

**ArXiv:** [2505.11594](https://arxiv.org/abs/2505.11594)

## 🎯 Pitch

SageAttention3 introduces the first practical 4-bit (FP4) 'microscaling' attention kernel for GPU inference, leveraging new Blackwell FP4 Tensor Cores to achieve up to 5× faster performance than state-of-the-art methods—while maintaining model accuracy in demanding text, image, and video generation tasks. It also pioneers trainable 8-bit attention (SageBwd), enabling efficient, lossless fine-tuning for large models, thus lowering both inference and training costs and paving the way for next-generation, low-latency, and ultra-efficient AI systems.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces SageAttention3, an FP4 (4‑bit floating point) “microscaling” attention kernel for inference that reaches up to 1038 TOPS on an RTX 5090—about a 5× speedup over FlashAttention2 on that GPU—while preserving end‑to‑end model quality (Fig. 4–5, Table 2, Table 4a). It also presents SageBwd, the first trainable low‑bit attention (INT8 forward+backward with one key FP16 path) that matches BF16 fine‑tuning performance on several LLM benchmarks (Fig. 8b–e, Table 3), though it converges more slowly for pretraining (Fig. 8a).

## 2. Context and Motivation
- Problem addressed:
  - Attention is compute- and memory-intensive with quadratic cost in sequence length. Leveraging new low‑precision Tensor Cores can accelerate it, but two gaps remain:
    1) A practical FP4 attention for inference that works “plug-and-play” across models on Blackwell GPUs is missing.
    2) Prior low‑bit attention work targets inference only; training with low‑bit attention (especially backward) is largely unexplored.
- Why it matters:
  - Faster attention directly lowers latency and cost for generation tasks (text, image, video). For training, low‑bit attention could reduce time-to-train and energy cost.
- Prior approaches and shortcomings:
  - FlashAttention/2 reduce memory I/O and improve tiling/parallelism but use higher precision (Sec. 2). FlashAttention3 includes FP8 options but targets Hopper GPUs and does not support training/backward, limiting broad applicability (Sec. 6 Related Work).
  - SageAttention/SageAttention2 established accurate 8‑bit attention for inference, but not FP4 and not for training.
  - A core pain point for low-bit attention is quantizing the attention map `P` (softmax output in [0,1]) and handling scale factors within hardware formats without accuracy loss (Sec. 3.2, Fig. 3).
- Positioning:
  - The paper contributes (1) a performant FP4 attention for inference (SageAttention3) with two technical advances—FP4 microscaling and a new two‑level quantization of `P`—and (2) a practical INT8 attention for training (SageBwd) that keeps a single sensitive backward matmul in FP16 to control gradient error (Sec. 4, Algorithm 3).

## 3. Technical Approach
This section explains SageAttention3 (FP4 inference) and SageBwd (INT8 training) step-by-step.

Definitions (selective):
- `microscaling`: Quantize small groups of elements together, each group with its own scale factor. Here, groups are 1×16 elements (Sec. 3.1).
- `NVFP4` / `MXFP4`: Two FP4 schemes from NVIDIA (Sec. 3.1, “Data type determination”). Both use E2M1 value format (2 exponent, 1 mantissa bit), but differ in group size and scale format. NVFP4 uses 1×16 groups with E4M3 (FP8) scales; MXFP4 uses 1×32 groups with E8M0 scales.
- `FP4MM`: The FP4 microscaling matmul instruction that takes quantized operands and their scales and internally dequantizes to multiply (Eq. 3).
- `per-block quantization`: One scale computed per FlashAttention tile (Sec. 2).
- `per-token quantization`: One scale per sequence token (per row), common for handling activations with highly varying magnitudes (Sec. 2).
- `online softmax`: Softmax computed incrementally across key/value tiles to avoid materializing full `S=QK^T` and `P` (Sec. 2; Algorithm 1, Lines 8–13).

A) SageAttention3: FP4 microscaling attention for inference (Sec. 3)
1) Start from FlashAttention’s blocked computation:
   - Compute `S = QK^T` tile-by-tile, apply online softmax to get `P`, then compute `O = PV` (Sec. 2, Algorithm 1).
2) FP4 microscaling quantization (Eq. 1–2):
   - Partition a matrix `X` into 1×n blocks (here n=16). For each block, set scale `s_ij = max(|X_ij|)/6`, quantize `X̂_ij = round_to_FP4(X_ij/s_ij)`. Dequantize with `X′_ij = s_ij·X̂_ij`.
   - The “/6” matches FP4’s E2M1 dynamic range so that values map into the representable set with good coverage (Sec. 3.1).
3) FP4 matmul with scales (Eq. 3):
   - `C = FP4MM(Â, s_A, B̂, s_B)`, equivalent to multiplying dequantized `A′` and `B′`.
4) Apply FP4 microscaling to both attention matmuls (Eq. 4):
   - Quantize `Q` and `K^T`, compute `S = FP4MM(Q̂, s_Q, K̂, s_K)`.
   - Online softmax yields `P_e` (pre-normalized `P` values).
   - Quantize `P_e` and `V`, then compute `O = FP4MM(P̂, s_P, V̂, s_V)`.
5) Smoothing for robustness (Algorithm 1, Lines 2 and 5):
   - `K` smoothing: subtract mean from `K` (Line 2).
   - `Q` smoothing: subtract per-block mean `q̄_i` and compensate via `GEMV(q̄_i, K^T_j)` (Line 8). These techniques are inherited from SageAttention2 (Sec. 3.1).
6) Two-level quantization for `P` to fix scale-factor precision loss (Sec. 3.2, Eq. 5; Algorithm 1, Line 10):
   - Problem: `P_e` is in [0,1]. Direct FP4 microscaling yields tiny FP8 scales (E4M3) in [0, 0.167], which are poorly representable and cause accuracy loss (Fig. 3a–b, d).
   - Solution: First “inflate” each row of `P_e` with per-token scale `s_P1 = rowmax(P_e)/(448×6)`, producing `P_e2 = P_e / s_P1`. Then apply standard FP4 microscaling on `P_e2` to get `P̂_2, s_P2`. Final `O = FP4MM(P̂_2, s_P2, V̂, s_V) × s_P1` (Eq. 5).
   - Effect: Better utilizes FP8 scale range (Fig. 3c), reduces both scale representation error and overall quantization error (Fig. 3d–e; Table 1b).
7) FP4 data type choice (Table 1a):
   - NVFP4 outperforms MXFP4 on real `Q,K,V` tensors (CosSim 99.52% vs 98.37%). Chosen for higher accuracy (Sec. 3.1).
8) Kernel-level optimizations (Sec. 3.3; Figs. 19–21):
   - Permutation for `K`: Aligns the accumulator’s layout with operand `A` via permuting `P` columns and correspondingly rearranging `K` (fused with quantization).
   - Reuse shuffle: Fuse microscaling’s “max-of-16” with online softmax’s rowmax reduction to halve shuffles/max ops, ~10% kernel speedup.
   - Producer warp epilogue: A scheduling design where producer warps overlap loads and global stores while consumers move matmul results into shared memory, increasing throughput under register pressure.
9) Fusing `V` transpose (Appendix A.5):
   - FP4 MMA for `PV` needs `V` contiguous in sequence dimension; they fuse the transpose into the quantization kernel to avoid extra I/O.

B) SageBwd: INT8 attention for training (Sec. 4; Algorithms 2–3)
1) Forward (Eq. 6; Algorithm 2):
   - Quantize `Q,K,V` per block to INT8 with scales `s_X = max(|X|)/127` (Eq. 7).
   - `S = MM(Q̂, K̂) × s_Q × s_K`.
   - Online softmax (Algorithm 2, Line 9) provides both local and global maxima; reuse them to scale `P_e` per token (Line 10): `s_P = exp(rowmax(S)-m)/127`, `P̂ = P_e/s_P`. This avoids extra max-reductions and improves `P` accuracy compared to static per-block scaling.
   - `O` accumulation uses the online softmax recurrence (Line 11).
2) Backward (Eq. 8; Algorithm 3):
   - Five matmuls appear: `S = QK^T`, `dV = P^T dO`, `dP = dO V^T`, `dQ = dS K`, `dK = dS^T Q`.
   - Key design choice: keep `dP = dO V^T` in FP16 (Algorithm 3, Line 8); quantize the other four matmuls per block to INT8 (Lines 7, 10–11).
   - Rationale: `dOV^T` accuracy directly drives `dP` and then `dS = P∘(dP−D)`, which repeatedly feeds into `dQ,dK` across tiles. Quantizing `dOV^T` accumulates error along sequence length (Sec. 4.2). An empirical ablation shows keeping `dOV^T` in FP16 improves gradient accuracy markedly (Table 1c: CosSim 99.77% vs 97.47% for `dQ`).
   - Additional details: reuse forward’s online softmax values `L_i` for stable reconstructions (Algorithm 3, Lines 5, 9), apply smooth‑`K` correction in gradient (Line 10).
3) INT8 vs FP8 for training:
   - INT8 yields more accurate gradients than FP8 for `dQ,dK,dV` (Tables 6–7). Fine‑tuning results after INT8 vs FP8 SageBwd also favor INT8 (Table 8).
4) Implementation:
   - SageAttention3 uses CUTLASS+CUDA; SageBwd is written in Triton (Sec. 5.1 Implementation).

## 4. Key Insights and Innovations
- FP4 microscaling for attention with 1×16 groups and hardware-friendly FP8 scales (fundamental):
  - Moves attention matmuls to FP4 while preserving accuracy via per‑microgroup scaling (Sec. 3.1). This is a new capability on Blackwell GPUs, delivering large speed gains (Fig. 4–5).
- Two‑level quantization for the attention map `P` (fundamental):
  - First per‑token rescale inflates `P`’s magnitude to fully exploit FP8 scale range; then apply FP4 microscaling (Sec. 3.2, Eq. 5). This directly addresses the scale‑in‑FP8 precision bottleneck for small [0,1] values (Fig. 3), producing accuracy on par with high precision (Table 1b).
- Backward‑aware 8‑bit attention for training with a single FP16 exception (fundamental):
  - Quantizes 6 of 7 matmuls in forward+backward to INT8 (per block), but preserves `dOV^T` in FP16 to stop error cascades in gradients (Sec. 4.2; Algorithm 3; Table 1c). This design yields “lossless” fine‑tuning compared to BF16 (Fig. 8b–e; Table 3).
- GPU‑level kernel engineering tailored to FP4 MMA (incremental but important):
  - Accumulator/operand layout alignment, fused reductions (reuse shuffle), and a producer‑warp epilogue to push throughput toward hardware limits (Sec. 3.3; Figs. 19–21).

## 5. Experimental Analysis
- Evaluation methodology (Sec. 5.1; Appendix A.3):
  - Models: LLMs (`Qwen2.5`, `Llama3.2`), text‑to‑video (`CogVideoX`, `HunyuanVideo`, `Mochi`), text‑to‑image (`Flux`, `Stable‑Diffusion 3.5`).
  - Datasets/prompts: open‑sora prompts for video, COCO for images, and GSM8K, DROP, MMLU, HellaSwag for LLM tasks.
  - Metrics:
    - Video: CLIPSIM, CLIP‑Temp (alignment), VQA‑a/t (aesthetic/technical), Flow‑score (temporal consistency) (Table 2).
    - Image: FID, sFID (fidelity), CLIPScore (alignment), ImageReward (preference) (Table 2).
    - LLMs: task accuracies/F1 (Table 3).
  - Baselines: PyTorch attention, xformers, FlashAttention2; for kernel speed, also SageAttention1/2 (Figs. 4–7).
- Main quantitative results:
  - Kernel speedups on RTX 5090 (Figs. 4–5):
    - With head dim 128, non‑causal: SageAttention3 hits 964–1038 TOPS from 1K–32K sequence; FlashAttention2 is 173–214 TOPS over the same range. Roughly ≈4.5–5× speedup.
    - With head dim 64, similar multiplicative gains. xformers is much slower; SageAttention3 is ≈8–11× faster than xformers.
  - Training forward+backward on RTX 4090 (Figs. 6–7):
    - SageBwd achieves up to 1.67× kernel speedup vs FlashAttention2 and ≈3× vs xformers.
  - End‑to‑end inference (Table 4a; Fig. 1):
    - HunyuanVideo latency: 489s → 164s (≈3×).
    - CogVideoX(2B): 64s → 27s (≈2.4×).
  - End‑to‑end quality (Table 2):
    - Video generation: minimal differences. Example—CogVideoX CLIPSIM 0.1865 (FP16) vs 0.1881 (FP4); HunyuanVideo VQA‑a 68.998 (FP16) vs 70.552 (FP4). Some metrics slightly improve; others are statistically close.
    - Image generation: Flux FID 162.812 (FP16) vs 162.121 (FP4), sFID 146.980 vs 142.839 (small improvement), CLIP 31.409 vs 31.450; SD3.5 similarly close (lower sFID with FP4).
  - Fine‑tuning with SageBwd (Table 3; Fig. 8b–e):
    - Qwen2.5‑3B: GSM8K 0.601 (BF16) vs 0.607 (INT8), MMLU 0.640 vs 0.653.
    - Qwen2.5‑1.5B and Llama3.2‑1B are also on par across tasks. Loss curves overlap BF16 (Fig. 8b–e).
  - Pretraining (Fig. 8a):
    - On FineWeb‑Edu with a 400M Llama‑style model, SageBwd converges but more slowly than BF16.
  - FP4 accuracy ablations (Table 1):
    - NVFP4 vs MXFP4: NVFP4 significantly better on real layers (CosSim 99.52% vs 98.37%).
    - Two-level `P` scaling vs direct: CosSim 99.52% vs 93.32%.
    - Keeping `dOV^T` in FP16 during backward: CosSim 99.77% vs 97.47% (INT8) for `dQ`.
  - Additional checks:
    - Error accumulation across layers can be controlled by keeping a few sensitive layers in FP16 (Appendix A.6, Table 15).
    - Smoothing `Q`/`K` materially improves FP4 accuracy (Appendix A.7, Table 16).
    - INT8 vs FP8 for SageBwd: INT8 yields lower `L1` and higher cosine similarity for gradients, and better downstream task accuracy (Tables 6–8).
  - Synergy of INT8 fine‑tune + FP4 inference (Table 5):
    - After fine‑tuning with SageBwd and inferring with SageAttention3, GSM8K/MMLU edge out BF16 fine‑tune + FP4 inference in both Qwen2.5‑1.5B and 3B variants.
- Do the experiments support the claims?
  - Inference: Yes. Kernel and end‑to‑end speedups are large (Figs. 4–5; Table 4a), and quality metrics remain stable (Table 2). Fig. 12 visually shows two‑level `P` quantization avoids artifacts seen with direct scaling.
  - Training: For fine‑tuning, yes—loss curves and final metrics match or slightly exceed BF16 (Fig. 8b–e; Table 3). For pretraining, the paper is upfront that convergence is slower (Fig. 8a).
- Notable caveats and robustness checks:
  - FA3 is unavailable on RTX 5090, so FlashAttention2 is the strongest baseline there (Fig. 1 caption; Sec. 5.1). Theoretical peak throughput comparisons suggest FP4’s upside on Blackwell (Appendix A.8, Table 17).
  - They explore several ablations (Tables 1, 15, 16) and implementation details (Appendix A.5) that strengthen the engineering story.

## 6. Limitations and Trade-offs
- Hardware dependency:
  - The FP4 path targets Blackwell GPUs with FP4 Tensor Cores (e.g., RTX 5090, B200/B300; Appendix A.8). This limits immediate portability; FP4 isn’t widely available on earlier architectures.
- Accuracy/precision design:
  - While average metrics are preserved, per-layer errors can accumulate with depth/length (Appendix A.6). A mitigation keeps a few sensitive layers in FP16, but this adds complexity and slightly reduces theoretical speedups.
- Training scope:
  - SageBwd relies on keeping `dOV^T` in FP16; fully low‑bit backward remains open. For pretraining, convergence is slower than BF16 (Fig. 8a), limiting applicability to large‑scale from-scratch training.
- Implementation maturity:
  - SageBwd is written in Triton; the paper notes a gap to theoretical performance (Conclusions/Future Work). Further kernel engineering could be necessary for production use.
- Model/runtime coverage:
  - Experiments are broad but still representative samples. Additional model families (e.g., speech, multimodal encoders) and extremely long contexts could reveal corner cases.
- Minor quality regressions:
  - Some metrics fluctuate slightly; e.g., in Table 2 for Mochi, the Flow-score decreases (1.8042 FP16 → 1.649 FP4). Differences are small, but not universally positive.

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that FP4 attention can be made accurate and “plug‑and‑play” for inference on Blackwell GPUs, substantially shifting the speed‑accuracy frontier (Figs. 4–5; Table 2). Establishes a practical blueprint for training-time low‑bit attention (SageBwd) by identifying the single most sensitive backward matmul to keep high‑precision (Table 1c; Appendix A.11).
- Follow‑up research:
  - Toward fully low‑bit training: Replace or robustify `dOV^T` computation via error‑resilient quantization, improved scaling schemes, or gradient‑aware training (Appendix A.11 provides a starting analysis).
  - Better pretraining: Investigate quantization‑aware training schedules, adaptive precision policies per layer/length, or alternative softmax parameterizations to recover convergence speed (Fig. 8a).
  - Autotuning mixed precision: Automatically select layers and tiles to run in FP16 vs FP4/INT8 based on runtime statistics (Appendix A.6 indicates only a few layers may need FP16).
  - Broader ops and platforms: Extend microscaling FP4 to other transformer ops (MLP/GEMM/Conv) and to other vendors once compatible low‑precision units are available.
- Practical applications:
  - Faster generation for text‑to‑video and diffusion (Table 4a; Fig. 1, Fig. 9), lower inference cost for LLMs, and faster fine‑tuning cycles for downstream tasks (Table 4b; Table 3). The combined recipe—INT8 fine‑tune + FP4 inference—can yield the best of both worlds in accuracy and speed (Table 5).

> Bottom line: By solving two long‑standing precision bottlenecks—accurate FP4 attention for inference and a practically stable low‑bit backward path for training—this work opens a clear path to large, real‑world speedups without sacrificing quality.
