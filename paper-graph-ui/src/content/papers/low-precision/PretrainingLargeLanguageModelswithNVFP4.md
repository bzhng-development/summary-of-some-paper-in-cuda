# Pretraining Large Language Models with NVFP4

**ArXiv:** [2509.25149](https://arxiv.org/abs/2509.25149)

## 🎯 Pitch

This paper presents the first stable and accurate methodology for pretraining large language models with 4-bit floating point (FP4) precision using the new NVFP4 format—integrating innovations such as 2D weight scaling, Random Hadamard transforms, stochastic rounding on gradients, and selective high-precision layers. The approach is validated at unprecedented scale (training a 12B-parameter model on 10 trillion tokens), achieving training loss and downstream accuracies that match the FP8 baseline, while unlocking major gains in computational speed and memory efficiency on NVIDIA Blackwell hardware—paving the way for faster, cheaper, and more energy-efficient next-generation LLMs.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces a practical recipe for training large language models in 4‑bit floating point using the `NVFP4` data format and a set of stabilizing techniques (2D weight scaling, Random Hadamard transforms on weight‑gradient paths, stochastic rounding on gradients, and a small set of high‑precision layers). Trained at unprecedented scale—a 12B‑parameter hybrid Mamba‑Transformer on 10T tokens—the method matches FP8 baselines on loss trends and downstream accuracy while promising significant compute and memory savings on NVIDIA Blackwell Tensor Cores (Table 1).

## 2. Context and Motivation
- Problem addressed
  - Training frontier LLMs demands tens to hundreds of yottaflops and massive energy (Abstract). Pushing from 8‑bit training (now common) to 4‑bit could further reduce compute and memory, but naïve FP4 training is unstable due to extreme quantization error, limited dynamic range, and implementation pitfalls over long token horizons (Abstract; Section 1).

- Why it matters
  - FP4 arithmetic on Blackwell hardware runs at 2–3× the math throughput of FP8 and halves memory for operands (Section 2; Table 1). If accuracy can be preserved, this translates into faster, cheaper, and more energy‑efficient pretraining.

- What existed before and limitations
  - Microscaling formats (`MXFP8/6/4`) perform blockwise scaling with 8‑bit power‑of‑two scale factors, typically over 32‑element blocks (Section 2; OCP spec). With FP4 (`MXFP4`), coarse power‑of‑two scales can waste representable values and reduce effective dynamic range by nearly one “binade” (the interval between consecutive powers of two), especially when the block maximum forces scale rounding upward (Appendix B.4).
  - Prior FP4 training attempts require careful treatment of outliers, rounding bias, and scaling but lacked public demonstrations of stable, multi‑trillion‑token pretraining of billion‑parameter LLMs (Section 1).

- How this paper positions itself
  - It proposes `NVFP4`, a 4‑bit microscaling format with finer block scaling and an additional tensor‑level scale, and couples it with a training methodology that targets the failure modes of FP4 at scale (Section 2; Section 4). It then validates stability and accuracy on a 12B model trained on 10T tokens and compares with FP8 and with `MXFP4` (Sections 3 and 5).

## 3. Technical Approach
The approach combines a new 4‑bit format (`NVFP4`) with four training techniques tailored to reduce quantization error where it matters.

- NVFP4 data format (Section 2; Appendix B)
  - Core idea: represent each tensor as 4‑bit elements (`E2M1`, i.e., sign + 2‑bit exponent + 1‑bit mantissa) inside small blocks, but store a more precise scale per block plus a global tensor scale.
  - What’s different from MXFP4:
    - Smaller block size: 16 elements vs. 32 in MXFP4, which shrinks the within‑block dynamic range and reduces the chance that many values round to zero (Section 2).
    - More precise block scales: uses FP8 `E4M3` scale factors (fractional mantissa, not just power‑of‑two), so block maxima align closely to FP4’s maximum representable value (Section 2; Appendix B.4).
    - A second FP32 tensor‑level scale keeps block scales within the representable range of `E4M3` (Section 2).
  - How quantization works (Appendix B):
    - Compute a global encode scale `s_enc` so that the largest value in the tensor (the global `amax`) would map, after scaling, to the maximum representable of the combined FP4×FP8 block range (Eq. 1).
    - For each 16‑element block, compute a local decode scale `S_dec,b = amax_block/6` so the block maximum would land at FP4’s max magnitude `±6` (Eq. 2). Store the product `S_dec,b * s_enc` quantized to `E4M3` (Eq. 3), then invert it in FP32 to get the block’s encode scale.
    - Quantize each value `x_i` as `q(x_i * s_enc,b)` to FP4 (Eq. 4). During GEMMs, Tensor Cores multiply partial dot‑products by the stored per‑block `E4M3` scales, and apply the global FP32 decode scales to the final outputs (Eq. 5).
  - Intuition: first “zoom” the entire tensor into a good range (global scale), then “micro‑zoom” each small tile (block scale) so its largest entry exactly fits FP4. This preserves more information from outliers without saturations and reduces small‑value underflow (Section 2; Appendix B.4).

- Mixed‑precision training design (Section 4.1; Figure 5; Appendix A)
  - Where FP4 is used: the three GEMMs in every linear layer—`Fprop` (forward), `Dgrad` (activation gradients), and `Wgrad` (weight gradients)—consume NVFP4 inputs and output in BF16/FP32 (Figure 5).
  - High‑precision “islands”: to maintain stability, keep a small subset of numerically sensitive linear layers (≤15%) in BF16 or MXFP8, especially toward the end of the network (Section 4.1). For the 12B model, the first 2 blocks and last 8 blocks are BF16 (≈16% of linear layers) (Section 4.1; Appendix A.1).
  - Other non‑linear/attention components, embeddings, output head, and optimizer states remain in BF16/FP32 (Section 4.1).

- 2D weight scaling to preserve forward/backward consistency (Section 4.3; Appendix E.5)
  - Problem: when the same weight tensor is scaled/quantized along different axes in forward vs. backward (due to transposition), the two passes effectively “see” different functions, violating the chain rule and harming learning (Section 4.3).
  - Solution: perform 2D scaling for weights in 16×16 blocks (input×output channels) so the same quantized values are used in both passes. For activations/gradients, keep fine‑grained 1×16 scaling (Section 4.3). Figure 4 shows removing 2D weight scaling worsens training loss.

- Random Hadamard Transforms (RHT) for outlier control (Section 4.2; Appendix C, E.4)
  - What it is: an orthogonal rotation using a `d×d` Hadamard matrix (elements ±1) optionally multiplied by a random diagonal sign matrix; applied in tiles to distribute large‐magnitude outliers more evenly (Section 4.2; Appendix C). Because `H H^T = I`, transforms on both GEMM inputs cancel in exact arithmetic.
  - Where it’s used: only on `Wgrad` inputs (the activation and activation‑gradient operands that produce weight gradients) (Section 4.2; Figure 4). Transforming `Fprop`/`Dgrad` did not help at this scale and can add unnecessary quantization error (Appendix E.4.1).
  - Practical settings: `d = 16` balances accuracy and cost (Appendix E.4.2; Figure 12). A single fixed random sign vector across training suffices at 12B scale (Appendix E.4.3; Figure 13).

- Stochastic rounding on gradients (Section 4.4; Appendix E.3)
  - What it is: probabilistically round a value to one of its two nearest representable numbers with probability proportional to proximity—cancelling quantization bias over time.
  - Where it’s used: on gradient tensors only (Section 4.4; Figure 4). Applying it to activations or weights increases forward‑path error and can cause divergence (Appendix E.3; Figure 10). Blackwell hardware supports stochastic rounding for FP4 conversions (Section 2).

- Hardware path and expected efficiency (Section 2; Table 1)
  - Blackwell Tensor Cores natively support NVFP4, including per‑block scale handling, and provide 4×/6× speedups vs. BF16 for FP4 GEMMs (GB200/GB300) and roughly 2×/3× math throughput vs. FP8 (Section 2; Table 1). Memory use is about half of FP8 for operands. The paper focuses on algorithmic stability rather than end‑to‑end runtime optimization (Section 3).

## 4. Key Insights and Innovations
- NVFP4’s two‑level scaling with smaller blocks preserves FP8‑like fidelity where it counts (Section 2; Appendix B)
  - Novelty: 16‑element blocks with `E4M3` block scales plus a global FP32 scale. Unlike `MXFP4`, block maxima do not force power‑of‑two rounding, avoiding wasted FP4 bins and nearly one lost binade (Appendix B.4).
  - Significance: better representation of outliers and fewer zeros at small magnitudes lead to consistently improved training behavior (Section 2; Section 5).

- 2D scaling of weights to avoid chain‑rule violations (Section 4.3; Appendix E.5)
  - Novelty: recognize that per‑axis scaling causes the forward and backward passes to quantize the same weight tensor differently, and fix it via 16×16 2D scaling replicated along the dot‑product dimension.
  - Significance: improves loss curves at scale (Figure 4) and is more impactful than making activations consistent (Appendix E.5; Figure 14).

- Targeted use of Random Hadamard Transforms only on the weight‑gradient path (Section 4.2; Appendix E.4)
  - Novelty: rather than transforming all GEMMs, limit RHT to `Wgrad` operands where outlier distributions most harm FP4 and avoid extra error where FP4 already suffices.
  - Significance: ablation shows that removing RHT degrades convergence (Figure 4), while applying it to `Fprop`/`Dgrad` can hurt (Appendix E.4.1; Figure 11).

- Stochastic rounding only on gradients (Section 4.4; Appendix E.3)
  - Novelty: precisely identify gradients as the main source of rounding bias in FP4 training and restrict stochastic rounding there.
  - Significance: essential for convergence at 12B scale (Figure 4). Using stochastic rounding on forward tensors increases error and can cause divergence (Appendix E.3; Figure 10).

- A practical mixed‑precision recipe with small BF16 “islands” (Section 4.1; Appendix E.2)
  - Insight: the last few blocks are the most FP4‑sensitive; keeping ≤15% of layers high‑precision stabilizes training. Figure 4 and Appendix E.2 show stable convergence even when only the last four blocks are BF16.

## 5. Experimental Analysis
- Evaluation setup
  - Models and data:
    - 12B hybrid Mamba‑Transformer (62 blocks; details in Table 3) trained on 10T tokens using a phased data blend and a Warmup‑Stable‑Decay schedule (constant LR for first 80%, then decay over last 20%) (Section 3; Figure 2; Appendix A.1).
    - 8B hybrid model trained on 1T tokens for NVFP4 vs. MXFP4 comparison (Section 5; Appendix A.2).
    - 1.2B Transformer used for ablations (Appendix A.3).
  - Metrics: validation loss over tokens; downstream tasks including MMLU, MMLU‑Pro (5‑shot), GSM8k (CoT), MATH, AGIEval, coding (HumanEval+, MBPP+), multilingual (Global MMLU, MGSM), and commonsense (ARC‑C, HellaSwag, PIQA, Winogrande). Evaluations are done in BF16 (Table 2).

- Main results at 12B/10T (Sections 3–4)
  - Loss tracking:
    - Quote: “During the stable phase of training, the relative loss error of NVFP4 remains consistently below 1%, and widens to slightly above 1.5% as the learning rate is decayed towards the end of training.” (Figure 2).
  - Downstream accuracy:
    - Quote: “The model attains an MMLU‑pro accuracy of 62.58%, nearly matching the 62.62% accuracy achieved through FP8 pretraining.” (Abstract; Table 2).
    - Table 2 summary (FP8 vs. NVFP4):
      - General: 68.99 vs. 69.82
      - MMLU: 77.36 vs. 76.57
      - GSM8k CoT: 89.08 vs. 92.27
      - MATH: 83.32 vs. 81.48
      - Multilingual (Global MMLU): 74.00 vs. 74.94; MGSM: 81.87 vs. 85.53
      - Coding: 59.52 vs. 56.67 (NVFP4 slightly lower)
  - Interpretation: despite a small late‑stage loss gap, downstream accuracy is comparable across domains; coding evaluations show modest lag that may reflect evaluation noise (Section 3; Figure 3; Table 2).

- NVFP4 vs. MXFP4 at 8B scale (Section 5; Figure 6)
  - Loss comparison:
    - Quote: “MXFP4 has a relative error of around 2.5% compared to 1.5% for NVFP4.” (Figure 6a).
  - Token‑budget trade‑off:
    - Quote: “MXFP4 matches NVFP4 loss when trained on 36% more tokens (i.e., using 1.36T instead of 1T tokens).” (Figure 6b).
  - Interpretation: NVFP4’s finer scaling and smaller blocks reduce the cost (in tokens/compute) needed to reach a given loss.

- Ablation studies and robustness (Section 4; Figure 4; Appendix E)
  - Removing any of the four methodology components (stochastic rounding, RHT, 2D weight scaling, or BF16 islands) degrades convergence at 12B scale (Figure 4).
  - Layer sensitivity: keeping only the last four blocks in BF16 suffices to stabilize training, while first‑block BF16 alone does not (Appendix E.2; Figure 9).
  - Stochastic rounding placement: gradients only helps; applying it to activations or weights causes divergence (Appendix E.3; Figure 10).
  - RHT placement and size: Wgrad helps; Fprop/Dgrad hurt at this scale (Appendix E.4.1; Figure 11). Matrix size `d=16` is a good trade‑off; `d=4` worse, `d=128` slightly better but costlier (Appendix E.4.2; Figure 12). One fixed random sign vector suffices (Appendix E.4.3; Figure 13).
  - Consistency analysis: mismatched forward/backward quantization of weights harms loss; 2D scaling improves over 1D scaling along different axes (Appendix E.5; Figure 14).

- Precision switching late in training (Appendix D; Figure 7)
  - Quote: “Loss matches the FP8 baseline when precisions are switched after 8.2T tokens… and only slightly worse when switched after 10T tokens.” Most of the remaining gap comes from forward‑path quantization; switching only the forward pass to BF16 at 8.2T reduces error from ~1.5% to ~0.5% (Appendix D; Figure 7).
  - Implication: the majority (~82–99%) of training can run in NVFP4 and still recover FP8‑level loss by briefly switching precision near LR decay.

- Convincingness
  - Strengths:
    - Scale: 12B on 10T tokens with long training curves (Figures 2–3) is a stringent test.
    - Breadth: many downstream tasks (Table 2) and extensive ablations (Figure 4; Appendix E) trace causality of each technique.
    - Comparative: head‑to‑head with MXFP4 quantifies NVFP4’s advantage in tokens required (Figure 6).
  - Gaps:
    - End‑to‑end wall‑clock speedups are not the focus; the paper leans on hardware throughput claims (Table 1).
    - Some components (e.g., attention/softmax paths) remain high precision, so the reported recipe is mixed‑precision rather than “FP4 everywhere” (Section 4.1).

## 6. Limitations and Trade-offs
- Hardware and software assumptions
  - Relies on NVIDIA Blackwell Tensor Cores with native NVFP4 support and stochastic rounding in conversion instructions (Section 2; Table 1). Portability to other hardware is not discussed.

- Not fully FP4 end‑to‑end
  - About 15–16% of linear layers remain in BF16 for stability at 12B scale (Section 4.1; Figure 4). Attention, normalization, embeddings, output head, and optimizer states remain BF16/FP32 (Section 4.1). This tempers the maximum theoretical speedup.

- Additional compute/memory passes
  - Computing global `amax` and applying the tensor‑level scale adds an extra memory pass per tensor (Appendix B.1). While the paper suggests potential optimizations (e.g., smaller‑granularity globals), the runtime impact is not quantified.

- Method complexity and tuning
  - The recipe has several moving parts—2D weight scaling, RHT only on Wgrad with `d=16`, stochastic rounding only on gradients, specific BF16 layer placement. Misconfiguration (e.g., stochastic rounding on forward tensors) can cause divergence (Appendix E.3).

- Task‑specific differences
  - Coding metrics are slightly lower for NVFP4 (Table 2: 59.52 vs. 56.67). The paper notes potential evaluation noise, but residual differences may exist.

- Scope and generality
  - Demonstrations cover one 12B hybrid architecture and one 8B model for NVFP4 vs. MXFP4. Broader scaling laws across model sizes/tokens and other architectures (e.g., MoE) remain open (Section 6).

## 7. Implications and Future Directions
- How this changes the landscape
  - Shows that sustained 4‑bit pretraining of a strong, multi‑billion‑parameter LLM over 10T tokens can be stable and accurate when coupled with the right numerics (Section 6). This lowers the barrier to training future frontier models by reducing compute and memory needs (Table 1).

- Practical applications
  - Training: Faster, cheaper pretraining runs on Blackwell clusters; reduced memory footprint lets practitioners increase batch size, context length, or model width for the same memory budget.
  - Serving/finetuning pipelines: A path to unify training and inference numerics around FP4 with improved scales; downstream methods (SFT, RLHF) may inherit the same efficiency with careful application of the recipe.

- Technical follow‑ups enabled or suggested
  - Quantize more of the stack:
    - Reduce or eliminate the remaining BF16 islands without hurting convergence (Section 6).
    - Extend NVFP4 to attention and communication paths and explore fused kernels that hide the cost of global scaling and RHT (Section 6).
  - Broader evaluations:
    - Establish scaling laws comparing NVFP4 and MXFP4 across parameter counts and token budgets (Section 5 discussion).
    - Test on larger models, longer horizons, and architectures such as MoE (Section 6).
  - Algorithmic refinements:
    - Automate selection of high‑precision layers using quantization‑error diagnostics (Appendix E.2 observation).
    - Study theoretical aspects of forward/backward consistency under block scaling and the limits of RHT size vs. benefit (Section 4.3; Appendix E.4–E.5).
  - Training schedules:
    - Systematize late‑phase precision switching (Appendix D; Figure 7) as a standard practice to recover the last 0.5–1% of loss with minimal extra cost.

Overall, the paper provides both a more capable FP4 format (NVFP4) and a concrete, empirically validated training protocol that together make 4‑bit pretraining a practical option at scale. The combination of methodological clarity (Figures 4–5), large‑scale validation (Figures 2–3), and comparative evidence vs. MXFP4 (Figure 6) makes the case compelling while leaving ample room for systems and algorithmic optimization in future work.
