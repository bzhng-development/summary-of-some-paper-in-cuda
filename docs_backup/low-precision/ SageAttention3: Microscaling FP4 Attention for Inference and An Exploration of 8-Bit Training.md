# SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training

**ArXiv:** [2505.11594](https://arxiv.org/abs/2505.11594)
**Authors:** Jintao Zhang, Jia Wei, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Haoxu Wang, Kai Jiang, Jun Zhu, Jianfei Chen
**Institutions:** Tsinghua University

## 1. Executive Summary (2-3 sentences)

SageAttention3 shows how to run exact attention almost entirely in 4-bit floating point (FP4) on NVIDIA Blackwell GPUs by combining microscaling quantization with a new two-level scaling trick for the softmax probabilities. It hits 1038 TOPS on an RTX 5090 (about 5× faster kernel speed than FlashAttention2 on the same card) and, importantly, keeps end-to-end quality across image/video generation models; the paper also introduces SageBwd, an 8-bit (INT8) attention that supports training with no measurable fine-tuning loss, while revealing why pretraining is still tricky.

Reasoning: Running attention at FP4 precision is hard because (a) FP4 only has 15 values per sign, (b) softmax outputs are small and make FP8 scales for FP4 microscaling numerically lossy, and (c) gradients in the backward pass amplify quantization errors. The paper tackles each of these with: 1×16 microscaling groups (to localize outliers), a two-level scale for softmax probabilities (to properly use the FP8 range), and a targeted mixed-precision backward (keep only dO·V^T in FP16), respectively.

## 2. Context and Motivation

- Problem/gap
  - Attention has quadratic time and memory complexity in sequence length (N×N maps). That’s already a bottleneck for long-context LLMs and becomes crushing for video diffusion models with very long tokenized sequences. Prior speed-ups like FlashAttention reduce memory I/O but still use FP16/FP8 arithmetic at best on many GPUs.
  - Blackwell GPUs add FP4 Tensor Cores (NVFP4) with huge throughput. But no one had an FP4 attention that was accurate enough to be “plug-and-play” across real models, especially video. Also, low-bit attention methods were inference-only—no one had a training-capable low-bit attention with backward support.

- Why this matters
  - Real-world: A 3× end-to-end speedup on a top-tier video generator (HunyuanVideo) cuts generation from 489s to 164s on RTX5090 (Table 4a). That’s hours saved at scale, lower costs, and more iteration speed for users and labs.
  - Theoretical/practical: If FP4 can be made robust for attention, it unlocks a new precision regime with order-of-magnitude higher math throughput. On the training side, if we can push attention into 8 bits without harming learning, we can significantly reduce training time and power.

- Prior approaches and their limitations
  - FlashAttention (and v2/v3) focuses on tiling, asynchrony, and low-precision support but doesn’t provide an accurate, general FP4 path on Blackwell for a wide range of models; FlashAttention3’s FP8 path doesn’t cover backward and isn’t plug-and-play for video (Related Work, Section 6).
  - SageAttention (8-bit) and SageAttention2 (per-thread INT4 plus outlier smoothing) handled inference, not training, and didn’t reach FP4 speeds on Blackwell hardware.

- How this paper positions itself
  - It introduces the first end-to-end usable FP4 attention for inference on Blackwell (SageAttention3) and the first low-bit attention with a working backward pass for training tasks (SageBwd). It also provides careful ablations and HW kernel design to actually reach the advertised throughput.

## 3. Technical Approach

High-level picture (Fig. 2 and Section 3):
- Goal: Replace the two expensive MatMuls in attention (QK^T and P·V) with FP4 microscaling MatMuls, while keeping accuracy.
- Key issue: Softmax probabilities P are small numbers in [0,1], which make the scale factors themselves hard to represent accurately in FP8 (required by NVFP4 microscaling), causing big errors if handled naïvely.
- Solution pieces:
  1) Use 1×16 microscaling groups for FP4 to confine outliers and improve quantization accuracy.
  2) Use a two-level scale for P so that the FP8 scale factor uses most of its representable range.
  3) Add several GPU-kernel tricks (layout permutation, fused operations, warp scheduling) to actually hit Blackwell’s FP4 Tensor Core speeds.

3.A FP4 microscaling for attention MatMuls (Section 3.1)

- Microscaling quantization (ϕ) in words:
  - Break a matrix X into 1×n blocks (n=16 for NVFP4). For each block X_ij, compute a scale s_ij so the block’s maximum magnitude maps to the largest FP4 code (here, 6; E2M1 has 1 mantissa bit, symmetric range with 15 representable magnitudes per sign; Eq. (1)).
  - Quantize entries by dividing by s_ij and rounding to FP4. Store s_ij in FP8 (E4M3) because NVFP4 Tensor Cores expect FP8 scales.
  - Dequantization multiplies the FP4 code by s_ij (Eq. (2)).

- Why 1×16 groups?
  - Per-tensor or per-channel scales are too coarse; outliers dominate and blow up quantization error. Small 1×16 groups “contain” outliers locally, so most values are well represented (Section 1 “Challenges (C1)” and Section 3.1).

- FP4 MatMul primitive (FP4MM; Eq. (3)):
  - Takes FP4-coded matrices plus FP8 scale matrices and computes the FP32-accumulated MatMul of dequantized inputs on the fly in hardware.

- Attention pipeline with FP4MM (Eq. (4), Algorithm 1):
  1) Smooth Q and K (from SageAttention2) to reduce outliers by subtracting means. Concretely:
     - Preprocess K by subtracting its mean (Algorithm 1, line 2).
     - For each Q block, subtract its row mean q̄_i (line 5) and keep q̄_i for a small GEMV correction later.
  2) Quantize Q, K^T to FP4 via ϕ; run S_ij = FP4MM(Q̂_i, s_Q, K̂_j, s_K) + GEMV(q̄_i, K_j^T) (line 8).
     - The GEMV term corrects for the mean subtraction so the result matches full-precision attention (this is the SageAttention2 smoothing trick).
  3) Compute softmax online (m_ij, l_ij as running max/normalizer; line 9).
  4) Quantize P_e (the unnormalized exponentials within the online softmax framework) using a special two-level scheme (line 10; see next subsection).
  5) Multiply O_ij += FP4MM(P̂_ij, s_P2, V̂_j, s_V) × s_P1 (line 11).
  6) Normalize outputs by l_i,Tn (line 13).

- Picking NVFP4 over MXFP4 (Section “Data type determination”):
  - NVFP4: E2M1 data, 1×16 block, scales in FP8 E4M3.
  - MXFP4: E2M1, 1×32 block, scales in E8M0.
  - Ablation with real Q, K, V from CogVideoX (Table 1a) shows NVFP4 improves Cosine Similarity from 98.37% (MXFP4) to 99.52%, with lower L1 and RMSE. The smaller group (1×16) and a more precise FP8 scale (E4M3) are key.

3.B Two-level scaling for softmax probabilities (Section 3.2)

Problem: With online softmax, P_e entries in each 1×16 block lie in [0, 1], so the per-block scale s_P = max(P_e)/6 lies in [0, 0.167]. Representing such a tiny number in E4M3 (FP8) wastes dynamic range and incurs large rounding error, which then corrupts P quantization (Fig. 3a–b; “Challenges (C2)”).

Mechanism (Eq. (5), Algorithm 1 line 10):
- Step 1: Per-token rescale. For each row of P_e, compute s_P1 = rowmax(P_e) / (448×6), and divide: P_e2 = P_e / s_P1.
  - Why 448? 448 is the max finite magnitude representable by E4M3 FP8. Multiplying by 6 ensures that after dividing by s_P1, the per-block max divided by 6 (the FP4 max code) will produce an FP8 scale factor s_P2 that sits well in E4M3’s usable range.
- Step 2: Microscale FP4 quantize the rescaled P_e2 with standard ϕ, producing P̂_2 with per-block FP8 scales s_P2.
- Step 3: Run the MatMul in FP4 with s_P2 and carry s_P1 as a separate FP32 multiplier: O += FP4MM(P̂_2, s_P2, V̂, s_V) × s_P1.

Why it works:
- It shifts the tiny softmax values into a scale regime where the FP8 scale factor s_P2 uses most of E4M3’s dynamic range (Fig. 3c).
- Fig. 3d–e show both the FP8 scale representation error and the P quantization error drop significantly vs direct quantization. Table 1b: cosine similarity jumps from 93.32% (Direct) to 99.52% (Two-level), with L1 dropping from 0.193 to 0.077 and RMSE from 1.103 to 0.201.

3.C Blackwell-kernel design to actually reach FP4 speeds (Section 3.3)

- Permutation for K:
  - FP4 MatMul’s FP32 accumulator layout doesn’t match operand A’s register layout (see FP4 operand A layout Fig. 19 vs FP32 accumulator layout Fig. 20). Shuffling threads to reconcile layouts is slow.
  - They permute the columns of P tiles and correspondingly reorder columns of K (fused into quantization) so data lands in the right layout without costly shuffles (Fig. 21).

- Reuse shuffle:
  - Microscaling P_e needs a max over 16-row segments that are split across 4 threads, which normally requires intra-thread reduction plus inter-thread shuffles.
  - They fuse this with the online-softmax max-reductions and reuse intermediate maxima, cutting redundant shuffles and max ops by 50% and giving about 10% end-to-end kernel speedup.

- Producer warp epilogue:
  - Instead of the usual “consumers do compute+store” schedule (which hit register limits here), they ping-pong producer warps so one loads the next tile while another stores results, and consumer warps only transfer accumulators to shared memory. This overlaps MatMul and global stores within register constraints and increases throughput.

3.D Training-capable 8-bit attention (SageBwd; Section 4)

Why INT8 and not FP4 for training?
- Hardware and tooling: INT8 Tensor Core paths and Triton kernels are mature; FP4 backward is much harder and not yet supported broadly in frameworks. The paper explores feasibility, not a final solution.

Forward (Algorithm 2):
- Quantize Q, K^T, V per block to INT8: s_X=max(|X|)/127, X̂=X/s_X (Eq. (7)).
- Compute S_ij = (Q̂_i K̂_j) × s_Q × s_K (line 7).
- Use online softmax (line 8).
- Quantize P_e per token (per row) for better accuracy: s_P = exp(rowmax(S_ij) − m_ij)/127; P̂_ij=P_eij / s_P (line 9). No extra max pass is needed—reuse online-softmax maxima.
- Compute O_ij += (P̂_ij V̂_j) × s_P × s_V (line 10).

Backward (Algorithm 3; Eq. (8)):
- Five MatMuls appear: QK^T (same as forward), dV = P_e^T dO, dP = dO V^T, dQ = dS K, dK = dS^T Q.
- Key observation: quantization noise in dP = dO V^T directly pollutes dS (since dS = P⊙(dP − D), line 9), then dQ and dK accumulate error across time steps in FlashAttention’s stream (Section 4.2; “Challenges (C3)”).
- Strategy: Keep dP=dO V^T in FP16 (line 8) and quantize the other four MatMuls in INT8 per block (lines 7, 10–11).
- Evidence: Table 1c shows dQ cosine similarity improves from 97.47% (if dO V^T is INT8) to 99.77% (if dO V^T is FP16), with L1 dropping from 0.171 to 0.039 and RMSE from 2.440 to 0.692.

## 4. Key Insights and Innovations

- FP4 microscaling attention that’s actually usable end-to-end (fundamental)
  - What’s new: First attention kernel that runs both QK^T and P·V with FP4 microscaling on Blackwell (Section 3.1), including the non-trivial handling of P and supporting the FlashAttention online-softmax flow.
  - Why it matters: Unlocks up to ~8× MatMul throughput vs FP16 MatMul on RTX5090 (Section 3.1 “FP4 microscaling quantization Matmul”), and 4–5× kernel speedup vs FlashAttention2 (Figs. 4–5), while preserving generation quality (Table 2).
  - Distinction: Prior low-precision attention stopped at INT8/FP8 or was Hopper-only (FlashAttn3 FP8), lacked backward, or wasn’t plug-and-play for video models.

- Two-level scaling for softmax probabilities (fundamental)
  - What’s new: A per-row pre-scale (s_P1) that reshapes P_e so its per-block FP8 scale (s_P2) falls into the sweet spot of E4M3’s dynamic range (Section 3.2, Eq. (5)).
  - Why it matters: Direct FP4 microscaling of P_e is inaccurate because s_P is tiny and FP8 can’t represent it well; two-level scaling fixes this, with a dramatic accuracy jump (Table 1b) and clear visual gains (Fig. 12).
  - Distinction: This is tailored to the NVFP4 requirement that scales be in FP8 E4M3; it’s a clever workaround to a hardware constraint.

- Identify and surgically preserve the critical backward path (incremental but high impact)
  - What’s new: In the INT8 training setting, only dO V^T remains FP16; all other 6/7 MatMuls are INT8 (Algorithm 3).
  - Why it matters: This precise selection avoids the error cascade in dS→dQ/dK while capturing most of the INT8 speed-up. Ablation in Table 1c shows the difference is decisive.

- Hardware kernel design tuned for NVFP4 (incremental but necessary)
  - What’s new: Layout permutation for K (Fig. 19–21), shuffle reuse (10% speed), producer warp epilogue.
  - Why it matters: These make the theoretical FP4 throughput realizable in practice. Without them, you’d leave a lot of speed on the floor.

## 5. Experimental Analysis

- Evaluation methodology
  - Hardware:
    - Inference: RTX 5090 (Blackwell) for FP4 SageAttention3 vs Torch, xformers, FlashAttention2, and earlier SageAttention versions (Figs. 4–5).
    - Training: RTX 4090 for INT8 SageBwd vs FlashAttention (CUDA and Triton) and xformers (Figs. 6–7; also Figs. 15–18 in Appendix).
  - Models:
    - Text-to-video: CogVideoX, HunyuanVideo, Mochi.
    - Text-to-image: Flux, Stable Diffusion 3.5.
    - LLMs: Qwen2.5 (1.5B and 3B), Llama3.2 (1B).
  - Metrics:
    - Kernels: Throughput (TOPS) by sequence length (1K–32K), head dim 64/128, causal vs non-causal.
    - End-to-end (image/video): CLIPSIM, CLIP-T, VQA-a/t, FScore, FID/sFID, CLIP, IR (Table 2).
    - Training: Fine-tuning accuracy on GSM8K, DROP, MMLU, HellaSwag (Table 3; loss curves in Fig. 8b–e); pretraining loss convergence on FineWeb-Edu with Llama 400M (Fig. 8a).
  - Implementation: SageAttention3 in CUTLASS+Cuda; SageBwd in OpenAI Triton (Section 5.1).

- Main quantitative results
  - Kernel-level throughput, FP4 inference (RTX5090):
    - Head dim 128, non-causal (Fig. 4, left): SageAttention3 hits 1038 TOPS at 32K, versus FlashAttention2 around 214–559 TOPS depending on seq length. Roughly 4–5× faster across lengths. xformers is much slower (≈95–215 TOPS).
    - Head dim 64 (Fig. 5): Similar story; SageAttention3 reaches ~839–827 TOPS (causal/non-causal) vs FlashAttention2 ~220–500 TOPS.
    - Note: FlashAttention3 can’t run on RTX5090; FlashAttention2 is the fastest baseline on this card (Figure 1 caption and Section 5.1).
  - End-to-end quality, FP4 inference (Table 2):
    - Text-to-video: For HunyuanVideo, CLIPSIM improves from 0.1838 (FP16) to 0.1866 (SageAttn3), CLIP-T stays at 0.9993; VQA-a/t and FlowScore remain in the same range. Similar negligible deltas for CogVideoX and Mochi.
    - Text-to-image: Flux and Stable Diffusion 3.5 show near-identical FID/sFID and CLIP/IR scores; e.g., SD3.5 FID is 166.421 (FP16) vs 166.102 (SageAttn3).
    - Visuals: Fig. 9 (and Appendix Figs. 10–11, 13–14) show qualitatively indistinguishable images/videos between full precision and SageAttention3.
  - End-to-end speedups, FP4 inference (Table 4a):
    - HunyuanVideo video generation: 489 s → 164 s (≈3×).
    - CogVideoX: 64 s → 27 s (≈2.4×).
    - The realized end-to-end speedups are smaller than the 4–5× kernel speedups, as expected, since other pipeline components also take time.
  - Training speed, INT8 (RTX4090, Figs. 6–7):
    - Forward+backward throughput improves over FlashAttention2 by up to 1.67× and more than 3× over xformers at some points. For example, head dim 128, non-causal: SageBwd hits ~839 TOPS at 8K (Fig. 6 left) vs FlashAttn(CUDA) ~220 TOPS (forward only numbers in Figs. 15–16 show 2× forward and 1.2–1.6× backward speedups).
    - End-to-end training iteration time (Table 4b): Llama (1B) at 8K tokens: 2.1 s → 1.9 s; at 16K: 6.0 s → 5.2 s (~1.15×).
  - Training accuracy:
    - Fine-tuning (Table 3; Fig. 8b–e): Across GSM8K, DROP, MMLU, HellaSwag, SageBwd matches BF16 within noise (often slightly better). For example, Qwen2.5-3B on GSM8K: 0.601 (BF16) vs 0.607 (SageBwd).
    - Stability across seeds (Appendix Tables 5–10): Means and variances for BF16 and SageBwd are essentially equivalent, suggesting robust behavior.
    - Pretraining (Fig. 8a): With Llama-400M on FineWeb-Edu, both BF16 and SageBwd converge, but SageBwd’s curve is consistently slower—i.e., the same loss needs more steps.

- Ablations that justify design choices
  - NVFP4 vs MXFP4 for FP4 microscaling (Table 1a): NVFP4 is clearly more accurate.
  - Two-level vs direct quantization of P_e (Table 1b; Fig. 3): Two-level is crucial; direct quantization hurts a lot (and Fig. 12 shows visible degradation).
  - Backward: Keep dO·V^T in FP16 (Table 1c): Essential for gradient accuracy of Q/K.

- Do the experiments support the claims?
  - Yes on inference: The FP4 kernel reaches extremely high throughput on Blackwell and preserves model outputs across diverse image/video models with quantitative and qualitative checks (Table 2, Fig. 9).
  - Yes on training (fine-tuning): Careful backward design achieves the intended “lossless” behavior on multiple tasks and models (Table 3, Fig. 8b–e).
  - Candid on pretraining: The slower convergence in Fig. 8a is a clear limitation disclosure.

- Any gotchas or caveats?
  - The biggest headline speeds are on RTX5090; the closest baseline (FlashAttention3) doesn’t run there, so comparisons are against FlashAttention2. That said, the kernel-level TOPS numbers are compelling and the end-to-end speedups are real for big video models.
  - The two-level scaling relies on E4M3 FP8 properties (max ≈ 448). If future hardware changes FP8 format (e.g., E5M2) this method needs retuning.

## 6. Limitations and Trade-offs

- Hardware specificity
  - SageAttention3’s FP4 path leans on NVFP4 microscaling (1×16 blocks, FP8 E4M3 scales) and Blackwell Tensor Core instructions (Section 3.1). On older GPUs, you won’t get these speedups; on future ones with different FP8 formats, the two-level scaling constant and behavior need revisiting.

- Softmax-specific trickery
  - The two-level scaling’s constant 448 is tied to E4M3’s max. The success of the approach hinges on this careful dance between FP4 block quantization and FP8 scale representation. It’s elegant, but specialized.

- End-to-end speedups won’t match kernel speedups
  - The 4–5× kernel gains turn into ~2–3× end-to-end on large video models (Table 4a). That’s still excellent, but other parts of the pipeline (diffusion steps, non-attention ops, IO) are bottlenecks.

- Training: pretraining still suffers
  - SageBwd works great for fine-tuning but converges slower for pretraining (Fig. 8a). This points to residual quantization noise in the backward pass—even after keeping dO·V^T in FP16—that harms optimization when you’re learning from scratch.

- Implementation complexity
  - The kernel improvements (layout permutation, producer warp scheduling) and softmax fusion steps require deep systems knowledge and careful engineering (Section 3.3). Porting or maintaining across frameworks isn’t trivial.

## 7. Implications and Future Directions

- Short-term impact
  - Inference: For anyone running long-context or video diffusion, SageAttention3 on Blackwell should be an immediate win. It’s a “plug-and-play” drop-in for QKV attention within FlashAttention’s tiled interface (Algorithm 1 flow), preserving quality (Table 2) and radically cutting latency (Table 4a).
  - Training: SageBwd is a practical speed-up for fine-tuning today, saving ~15% iteration time in the reported setups (Table 4b), with no accuracy loss (Table 3).

- Shifts in practice
  - Precision regimes: FP4 is no longer a novelty—it’s viable for exact attention inference when paired with the right quantization strategy. Expect FP4 kernels to show up in next-gen inference stacks for Blackwell-class hardware.
  - Mixed-precision backward: The paper’s diagnosis that dO·V^T is the sensitive path suggests a general recipe for low-bit training: keep only a few gradients in higher precision, quantize the rest.

- Research directions
  - Make pretraining work at low-bit: The current bottleneck is error accumulation in attention gradients (Section 4.2). Possible avenues:
    - Better gradient quantization (stochastic rounding, error feedback, or per-token/group adaptive scaling in backward).
    - Low-bit-aware softmax gradient formulations to reduce amplification.
    - Dynamic precision schedules (start higher precision early, then decay).
  - Generalize the two-level scaling: Explore variants for different FP8 formats (E5M2) or even hybrid FP6/FP7 to keep scales optimal.
  - Extend beyond attention: Apply similar microscaling+two-level ideas to MLPs or other blocks that suffer from small activations with scale-quantization mismatch.
  - Multi-GPU and sparsity: Combine with RingAttention or sparsity (Section 6) to compound gains—FP4 math plus less math.
  - Closer-to-peak kernels: Authors note a gap to theoretical limits for SageBwd’s Triton kernels (Conclusion–Future Work). Rewriting in CUDA/CUTLASS or using upcoming compiler features could close that gap.

- Practical applications
  - Video diffusion and long-sequence generation (HunyuanVideo, CogVideoX): cut latency 2–3× with retained quality.
  - Fine-tuning small-to-mid LLMs on limited hardware: reduce iteration time without changing training recipes or accuracy outcomes.

Quotes and anchors to the paper for key claims and numbers:
- FP4 kernel throughput and speedups:
  > “Our implementation achieves 1038 TOPS on RTX5090, which is a 5× speedup over the fastest FlashAttention on RTX5090.” (Abstract; Fig. 4–5)
- End-to-end speedups:
  > “HunyuanVideo: 490s → 164s” (Figure 1 caption and Table 4a; 489s vs 164s in Table 4a)
- Two-level scaling wins:
  > “Two-level quantization boosts the accuracy… CosSim 99.52% vs 93.32%” (Section 3.2; Table 1b; Fig. 3)
- Backward: keep dO·V^T in FP16:
  > “CosSim improves from 97.47% to 99.77%” (Section 4.2; Table 1c)
- Fine-tuning parity and pretraining caveat:
  > “8-bit attention achieves lossless performance in fine-tuning tasks but exhibits slower convergence in pretraining tasks.” (Abstract; Fig. 8 and Table 3)

Definitions (selective):
- FP4/NVFP4: 4-bit floating point format used on Blackwell Tensor Cores; NVFP4 uses E2M1 encoding with microscaling groups of 1×16 and FP8 (E4M3) scale factors (Section 3.1).
- Microscaling: Quantization where each small block (here 1×16) has its own scale factor; better handles outliers than global scales (Section 3.1).
- E4M3 / E8M0: FP8 formats with 4 exponent/3 mantissa or 8 exponent/0 mantissa bits, respectively; E4M3 has better precision for scales, E8M0 more dynamic range but no mantissa (Section 3.1).
- Online softmax: Computes softmax over blocks while maintaining a running max and normalizer, avoiding materializing the full N×N score matrix (Section 2; Algorithm 1 lines 9 & 13).
- TOPS: Trillions of operations per second—a throughput metric for matrix cores.

Final reasoning: The paper addresses three real blockers to FP4/INT8 attention—outliers (solved by 1×16 microscaling and smoothing), FP8 scale representation for small-softmax values (solved by two-level scaling), and gradient sensitivity (solved by a single FP16 MatMul in backward). The numbers line up across ablations (Table 1), kernel benchmarks (Figs. 4–7), and end-to-end metrics (Table 2 and 4). The pretraining result (Fig. 8a) is an honest limitation and usefully narrows the problem: further innovations need to target the backward signal path and softmax gradients.
