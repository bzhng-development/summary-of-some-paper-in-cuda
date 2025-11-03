# BitNet: Training 1-Bit Transformers from Scratch

**ArXiv:** [2310.11408](https://arxiv.org/abs/2310.11408)
**Authors:** Alex Gu, Ronghang Hu, Zhaoyang Lv, Brandon L. Amos, Kushal Tirumala, Lilian Weng, Yuhuai Wu, Eric Zelikman, Tianjun Zhang, Yi Tay, Mingda Chen, Tianyi Zhang, Hao Tian, Yi Ma, Denny Zhou
**Institutions:** Google DeepMind, Stanford University

## 🎯 Pitch

BitNet introduces a novel approach to train large language models using 1-bit weights, maintaining competitive accuracy while significantly reducing energy and memory costs through its innovative BitLinear layer. This method not only challenges the limitations of extreme quantization but also shifts focus to efficient inference economics, paving the way for more sustainable AI deployment across diverse environments.

---

## 1. Executive Summary
BitNet proposes a way to train large language models whose linear layers use only 1-bit weights, via a drop‑in layer called `BitLinear` that preserves training stability and accuracy while drastically cutting energy and memory costs. Across language modeling and downstream tasks, BitNet matches or nears FP16 Transformer performance but delivers large energy savings, and it scales predictably with model size in both accuracy and energy terms (Figures 1 and 3, Table 1).

## 2. Context and Motivation
- Problem addressed
  - Deploying large language models (LLMs) is expensive because inference is dominated by moving and multiplying large matrices at high precision. Memory bandwidth and inter-device communication become bottlenecks as models scale.
  - Quantization reduces precision to save compute and memory, but most LLM work focuses on post‑training quantization, which often loses accuracy at very low precision (especially at 1–2 bits).
- Importance
  - Real-world: Lowering inference energy and memory can reduce hosting costs and environmental impact and enable on-device or low-resource deployment.
  - Scientific: Shows whether extreme quantization (1‑bit) can be trained from scratch at LLM scale while retaining the favorable scaling laws of Transformers (Section 3.2, Figure 3).
- Prior approaches and shortcomings
  - Post-training quantization (e.g., Absmax, SmoothQuant, GPTQ, QuIP; Section 4): simple but accuracy drops sharply as bits decrease, and models are not optimized for the quantized representation.
  - Quantization-aware training (QAT) exists mainly for CNNs or smaller Transformers (e.g., BERT, translation). Prior work has not shown 1-bit training for autoregressive LLMs at multi‑billion scale.
- Positioning
  - BitNet trains 1‑bit weights from scratch for decoder-only LLMs and keeps selected components (e.g., embeddings, residual connections, layer norms) at higher precision where they matter less for compute but more for stability and sampling (Section 2, Figure 2b). It introduces a new `BitLinear` operator and a model-parallel strategy (group quantization/normalization) and evaluates scaling laws that consider inference energy, not just FLOPs (Sections 2.1–2.3, 3).

## 3. Technical Approach
BitNet keeps the overall Transformer layout but replaces every standard linear projection (`nn.Linear`) inside attention and feed-forward layers with a quantized `BitLinear` module (Figure 2a–b). Key components:

- What is binarization?
  - Binarization maps each weight to either +1 or −1. BitNet centers weights before binarization and rescales afterward:
    - Centering: subtract the mean `α` of the full-precision weight matrix `W` (Eq. 3), then apply `Sign` (Eq. 1–2).
    - Rescaling with `β`: a per‑layer scalar set to the average absolute value `(1/(nm))||W||₁` (Eq. 12) to reduce the L2 error between real and binarized weights.
- Activation quantization via `absmax`
  - Quantizes each activation tensor `x` to `b` bits (8-bit in experiments) by dividing by its maximum absolute value `γ = ||x||∞` and clipping to `[−Q_b, Q_b]` with `Q_b = 2^{b−1}` (Eqs. 4–5).
  - For pre‑nonlinear activations that should be nonnegative, it subtracts the minimum `η` first and quantizes into `[0, Q_b]` (Eq. 6).
  - During training, quantization is per‑tensor; during inference, per‑token for better efficiency and stability (Section 2.1).
- Preserving variance for stable training
  - Low-precision operations can destabilize signal magnitudes across layers. BitNet inserts a LayerNorm before activation quantization—implemented as `SubLN` (pre-activation normalization; Section 2.1)—so that the output variance remains near 1 (Eqs. 8–12). Intuition: by normalizing `x` before quantization and using the scalar `β`, the variance of the `BitLinear` output matches that of full precision, helping gradients and activations stay in a healthy range.
- The `BitLinear` forward pass (Eq. 11)
  - Compute `LN(x)` (SubLN), quantize to `xe = Quant(LN(x))`, multiply with binarized weights `W_f = Sign(W − α)`, and then dequantize by multiplying the result with `(β γ / Q_b)`:
    - `y = W_f · Quant(LN(x)) × (βγ/Q_b)`.
  - All other components (residual connections, layer norms) remain in higher precision (8‑bit in experiments), and embeddings stay high precision to support accurate sampling (Section 2).
- Training 1-bit weights
  - Straight-Through Estimator (STE): During backpropagation, treat the non-differentiable `Sign` and `Clip` operations as identity for gradient flow (Section 2.2). This is a standard trick in binarized networks to enable gradient-based learning.
  - Latent weights: Maintain a high-precision copy of weights for optimization; binarize them on the fly only for the forward pass (Section 2.2). This lets small gradient updates accumulate even though the forward weights are 1-bit.
  - Learning rate: Use a larger learning rate than FP16 training; small updates often don’t flip a binarized weight, so larger steps help optimization make effective changes (Section 2.2; empirically validated in Figure 5).
- Model parallelism without synchronization overhead
  - Challenge: per-layer scalars `α, β, γ, η` and LayerNorm statistics usually require global reductions across partitions, which slows distributed training.
  - Solution—Group Quantization and Group Normalization (Section 2.1):
    - Partition weight or activation tensors into `G` groups along the parallel dimension and compute `α_g, β_g, γ_g, η_g` locally per group (Eqs. 13–14).
    - Apply LayerNorm statistics per group (Eq. 15).
    - Benefit: avoids all-reduce operations for these scalars, enabling efficient model parallelism.
- Efficiency modeling (Section 2.3)
  - Energy model: Uses published per‑op energies for add/multiply at 45nm and 7nm technology nodes (Table 2).
  - For a standard FP Transformer matrix multiply (`m×n` by `n×p`), energy is dominated by multiplications (Eqs. 16–17).
  - For BitNet, weights are 1‑bit; the main work becomes additions, and multiplications are only needed for the final scalar dequantization `(β, γ/Q_b)` (Eq. 18). This yields large savings, especially in multiplies (Table 1).

Implementation details used in experiments:
- Data: English corpora combining The Pile, Common Crawl snapshots, RealNews, and CC‑Stories; SentencePiece vocabulary size 16K (Section 3.1).
- Architectures: 125M → 30B parameter decoders; exact dimensions and hyperparameters in Tables 5–6 (Appendix A).
- Training regime: polynomial LR decay, Adam β=(0.9,0.98), no dropout, weight decay typically 0.01 (Table 6), with higher decay for 13B/30B to stabilize (Table 6 note).

## 4. Key Insights and Innovations
- Training LLMs with 1‑bit weights from scratch is feasible and stable
  - Novelty: Prior binarization work focused on CNNs or smaller Transformer settings (e.g., BERT, translation). BitNet shows direct-from-scratch training for autoregressive LLMs up to 30B parameters (Section 3.1) with competitive perplexity and downstream accuracy (Figures 3–4, Table 3).
  - Why it matters: It removes reliance on post-training quantization and its accuracy loss at extreme precision.
- `BitLinear`: a minimal, drop‑in 1‑bit replacement for linear layers
  - What’s different: Instead of complex per-channel scales and calibration, BitLinear uses:
    - Centering (`α`) + sign + average-L1 scale (`β`) for weights (Eqs. 1–3, 12).
    - `absmax` quantization for activations with SubLN to preserve variance (Eqs. 4–6, 11–12).
  - Significance: Small code changes (replace matrix multiplies with BitLinear), yet substantial energy savings (Table 1) and accuracy close to FP16 baselines (Figures 3–4, Table 3).
- Group Quantization and Group Normalization for model parallelism
  - What’s new: Computes all quantization and normalization statistics locally per parallel shard to avoid communication (Eqs. 13–15).
  - Impact: Preserves parallel scalability; otherwise the many tiny all-reduces would dominate latency (Section 2.1).
- Inference‑Optimal Scaling Law
  - Contribution: Evaluates scaling not just by loss vs. parameter count, but by loss vs. estimated inference energy (Section 3.2; Figure 3, left). Uses the parametric law `L(N) = a N^b + c` (Eq. 19) and shows similar power-law behavior to FP16 when plotting loss vs. size, but substantially better loss for the same energy budget when plotting loss vs. energy.
  - Significance: Shifts focus toward inference economics, which dominate real deployments.

## 5. Experimental Analysis
- Evaluation methodology
  - Setups
    - Scaling study: Train models from 125M to 30B parameters; keep training tokens fixed and vary model size (Section 3.2). Hyperparameters listed in Tables 5–6.
    - Stability tests: Compare convergence at different learning rates (Figure 5; Table 7).
    - Post-training quantization (PTQ) comparison: Apply Absmax, SmoothQuant, GPTQ, QuIP to FP16 baseline models at multiple bit settings and compare to BitNet W1A8 (Section 4; Table 3; Figure 6).
  - Datasets and metrics
    - Language modeling perplexity (PPL) on a validation set from their training corpora (Sections 3–4).
    - Downstream zero-shot and 4-shot accuracy on Hellaswag, Winogrande, Winograd, StoryCloze (Section 3.3; Figure 4; Table 3).
    - Energy estimates using the Horowitz/ZZL energy model (Table 2) to compute inference energy for matrix multiplies (Section 2.3), summarized in Table 1 and Figure 3.
- Main quantitative results
  - Energy savings (Table 1; input length=512):
    - At 7nm:
      - 6.7B model: FP32 multiplies ≈ 12.46 J vs. BitNet ≈ 0.08 J; adds 3.03 J vs. 0.13 J.
      - 30B model: FP32 multiplies ≈ 56.73 J vs. BitNet ≈ 0.20 J; adds 13.80 J vs. 0.53 J.
    - Similar reductions at 45nm. Multiplication energy is the dominant savings.
  - Scaling curves (Figure 3)
    - Loss vs. model size: BitNet follows a power law similar to FP16 Transformers; the loss gap shrinks as size increases.
    - Loss vs. inference energy: For the same energy budget, BitNet achieves substantially lower loss than FP16 (left panel).
  - Downstream capabilities vs. energy (Figure 4)
    - Both zero-shot and few-shot accuracy improve with more inference energy. BitNet’s accuracy climbs faster per unit energy than FP16 across tasks, indicating better energy efficiency for capability scaling.
  - PTQ comparison at 6.7B (Table 3)
    - Average zero-shot accuracy across four tasks:
      - FP16 baseline: 57.8; BitNet W1A8: 55.9.
      - 8-bit Absmax/SmoothQuant: 53.4 / 56.7.
      - 4-bit GPTQ / Absmax / SmoothQuant: 52.9 / 46.2 / 45.1 (note very poor PPL for W4A4 Absmax/SmoothQuant).
      - 2-bit GPTQ / QuIP: 45.2 / 49.0.
      - 1-bit PTQ Absmax/SmoothQuant: ≈ 44–45 average and catastrophic PPL (e.g., 3.5e23), indicating failure at inference.
    - Takeaway: When both weights and activations are very low precision, PTQ struggles. BitNet’s QAT at 1‑bit weights (and 8‑bit activations) yields performance on par with the best 8‑bit PTQ while using far less energy.
  - Stability and optimization (Figure 5)
    - Left: With the same (large) learning rate, FP16 diverges early while BitNet continues to converge, showing better tolerance to high LR.
    - Right: BitNet improves when increasing peak LR from 2e-4 to 8e-4 (PPL drops more), supporting the “use larger LR” recommendation (Section 2.2; Table 7).
  - Ablations (Table 4)
    - Replacing SubLN with Pre-LN or BMT degrades average accuracy by ~0.1–2.3 points (zero-shot and few-shot averages).
    - Using Elastic activation quantization underperforms `absmax` in this setup.
- Do the experiments support the claims?
  - Evidence is consistent that:
    - BitNet is energy-efficient (Table 1).
    - It scales predictably and efficiently (Figure 3).
    - It achieves competitive downstream performance against PTQ baselines at equal or fewer bits (Table 3, Figure 6).
    - Training is stable and benefits from higher learning rates (Figure 5).
  - Caveats:
    - Energy is modeled, not directly measured on hardware (Section 2.3; Table 2). Actual gains depend on implementations that exploit bit operations efficiently.
    - Downstream task set is limited (four popular benchmarks). Broader task coverage would further validate generality.

## 6. Limitations and Trade-offs
- Assumptions in variance and independence
  - The variance analysis (Eqs. 8–10) assumes independence and identical distributions for weights and activations and uses approximations to argue stability when using SubLN and scaling. Real networks may deviate from these assumptions.
- Precision choices are pragmatic, not fully 1‑bit everywhere
  - Only the linear weights are 1‑bit. Activations are 8‑bit; residual paths, layer norms, and embeddings remain higher precision for stability and sampling (Section 2). This is a reasonable trade‑off, but it means BitNet is not “all 1‑bit.”
- PTQ comparison scope
  - PTQ methods are applied to the same FP16 baseline but may benefit from per-layer or per-channel calibrations or data-dependent tuning not explored here. Nevertheless, the large gap at 1–2 bits suggests a real advantage for QAT at extreme quantization.
- Energy and speed on real hardware
  - Savings are computed from op-level energy models (Table 2), not measured wall-clock latency or power on GPUs/ASICs. Actual speedups require kernels that use bit packing, XNOR/popcount, and specialized memory layouts; such engineering is outside the paper’s scope.
- Training regime and data
  - Vocabulary is relatively small (16K; Section 3.1). Training data is large and diverse, but details like total training tokens per model and exact validation setup are summarized at a high level (Section 3.1; Appendix A).
- Attention and embeddings remain high precision
  - While computationally justified (their cost is smaller relative to large FFN projections; Section 2), future full-stack quantization would need to address these parts for maximal savings.

## 7. Implications and Future Directions
- How this changes the field
  - Demonstrates that 1‑bit weight training at LLM scale is not only possible but also yields competitive accuracy with much better energy efficiency. This reframes extreme quantization from an afterthought (PTQ) to a design-time choice (QAT) for large models.
  - Introduces an energy-centric view of scaling (Figure 3), which better reflects deployment economics than FLOPs-only analyses.
- Follow-up research enabled
  - Extending beyond W1A8: explore lower-bit activations (W1A4, W1A2), mixed-precision policies per layer, and adaptive precision schedules during training.
  - Hardware-software co-design: kernels for XNOR/popcount or addition-dominant accumulations, bit-packing for memory bandwidth, and support in accelerators/compilers.
  - Broader architectures: applying the BitLinear recipe to Retentive Networks (RetNet; Section 6), Mixture-of-Experts, or encoder–decoder setups.
  - Theory and optimization: sharper analyses beyond the independence assumptions; alternative gradient estimators to STE; studying why larger learning rates stabilize binarized training (Figure 5).
  - Distributed training: further exploration of group-wise statistics, dynamic grouping, and communication–computation trade-offs in bigger clusters.
- Practical applications
  - Cost-effective inference serving for large models, especially where energy is constrained (edge/cloud).
  - Enabling larger models within fixed inference budgets: for a given power envelope, BitNet’s inference-optimal scaling suggests better achievable accuracy (Figure 3 left).
  - Fine-tuning and continual training: QAT from scratch means models can be adapted, unlike many PTQ pipelines that freeze or break during further training.

> Bottom line: BitNet replaces linear layers with a 1‑bit, variance-preserving `BitLinear` operation and shows that large LLMs trained this way can retain strong accuracy while gaining large energy benefits. Its groupwise parallelization and energy-aware scaling analysis make it both practically attractive and conceptually clarifying for the future of efficient LLMs.
