# SLA: Beyond Sparsity in Diffusion Transformers via Fine‑Tunable Sparse‑Linear Attention

**ArXiv:** [2509.24006](https://arxiv.org/abs/2509.24006)
**Authors:** Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jun Zhu, Jianfei Chen
**Institutions:** Tsinghua University (likely among others)

## 🎯 Pitch

Sparse-Linear Attention (SLA) revolutionizes Diffusion Transformers by integrating sparse and linear attention in a unified GPU kernel, slashing attention computation costs by 95% while maintaining high-quality video generation. This breakthrough enables significantly faster, scalable video diffusion models, paving the way for real-time applications and enhanced performance across long-sequence generative tasks.

---

## 1. Executive Summary
SLA (Sparse–Linear Attention) is a trainable attention mechanism for Diffusion Transformers (`DiT`) that fuses sparse attention with linear attention inside a single GPU kernel. It solves the bottleneck of quadratic-cost attention for long video sequences by computing exact attention only where it matters and low-rank, linearized attention elsewhere, yielding large speedups without degrading generation quality.

The key significance is empirical and systems-level: SLA cuts attention computation by about 95% and achieves a 13.7× kernel speedup and 2.2× end-to-end speedup on the Wan2.1-1.3B video model, while preserving video quality relative to full attention (Table 1, Figure 6).

## 2. Context and Motivation
- Problem addressed
  - Attention in Transformers has quadratic cost in sequence length. For video `DiT` models, sequence lengths are 10K–100K tokens, so attention latency dominates runtime (Section 1).
- Why it matters
  - Video diffusion models must process many frames at high resolution; practical deployment hinges on attention efficiency. Reducing attention cost unlocks shorter latencies and larger/longer-context models.
- Limits of prior approaches
  - Linear attention (Section 2.2) reduces complexity to linear in sequence length, but in diffusion—especially video—it substantially degrades quality. The paper reports “linear attention severely degrades video quality” in their tests, and existing diffusion work with linear attention is largely limited to images (Limitation L1, Section 1; also ablations in Table 2 “Linear Only”).
  - Sparse attention (Section 2.1) typically reaches only 40–60% sparsity below 50K sequence length; even recent 80–85% sparsity results depend on much longer sequences (Limitation L2, Section 1). Figure 1 (right) explains why: removing too many entries causes large errors.
- Positioning
  - The paper’s central observation is that full attention weights can be split into:
    - a small fraction of large weights with high rank, and
    - the vast remainder with extremely low rank (Section 3.2; Figure 3).
  - This explains why sparse-only (targets the few large weights) and linear-only (assumes global low rank) each fail in isolation, and motivates a hybrid that combines them (Section 3.2, Eq. (1)).

## 3. Technical Approach
SLA partitions attention computation into three tiers—critical, marginal, negligible—using a learned, block-level predictor, and executes each tier with the most appropriate mechanism, all inside one fused GPU kernel (Section 4; Figure 4; Algorithms 1–2).

Key components and steps:
1) Predict where attention matters at block level
- Block partitioning: Queries, keys, and values are divided into blocks of size `bq × d` and `bkv × d` for efficiency on GPUs (Section 2.1). In experiments, `bq = bkv = 64` (Section 6.1).
- Compressed attention predictor `Pc` (Eq. (2)):
  - Compute mean-pooled queries and keys, then a compressed softmax attention:
    - `Pc = Softmax(pool(Q) pool(K)^T / sqrt(d))`, where `Pc ∈ R^(N/bq × N/bkv)`.
  - Define a compressed mask `Mc` (Eq. (3)) per block:
    - Top `kh%` per row: critical blocks (`Mc = 1`).
    - Bottom `kl%` per row: negligible blocks (`Mc = -1`).
    - Others: marginal (`Mc = 0`).
  - Default hyperparameters for video: `kh = 5%`, `kl = 10%` (Section 6.1; ablation in Table 2).

2) Compute critical blocks with exact attention (sparse FlashAttention path)
- For each query block `Qi`, visit only key/value blocks `Kj, Vj` with `Mc[i,j] = 1`.
- Perform blockwise attention with online softmax (Eq. (4); Algorithm 1 lines 9–11):
  - `Sij = Qi Kj^T / sqrt(d)`,
  - `Pij = OnlineSoftmax(Sij)`,
  - Accumulate sparse output `Os_i += Pij Vj`.
- Online softmax computes stable softmax statistics across blocks to avoid materializing the full `N×N` scores (Algorithm 1, lines 10–11; Section 4.1).

3) Compute marginal blocks with linear attention (low-rank path)
- Rationale: the many small weights are low-rank (Section 3.2, Figure 3); linear attention computes a low-rank approximation in O(N d^2) time (Section 2.2).
- Precompute per key/value block once (Algorithm 1 line 4):
  - `hj = φ(Kj)^T Vj` and `zj = rowsum(φ(Kj)^T)`.
- For each query block `Qi`, aggregate only over marginal blocks (`Mc[i,j] = 0`) (Eq. (5)):
  - `Hi = Σ hj`, `Zi = Σ zj`, then
  - `Ol_i = φ(Qi) Hi / (φ(Qi) Zi)`.
- This turns many per-block multiplications into a handful of additions (Algorithm 1 line 13; Section 4.2).
- Choice of feature map `φ(·)`: ablation favors `softmax` over `elu+1` and `hedgehog` (Table 2).

4) Skip negligible blocks
- For `Mc = -1`, no computation is performed.

5) Fuse results and learn a small projection
- Final output is the sum of sparse output and a learned projection of the linear output (Eq. (6)):
  - `O = Os + Proj(Ol)`.
- `Proj: R^d → R^d` is learned to mitigate distribution mismatch between softmax attention and linear attention outputs (Section 4.2 “Insight”).

6) Backward pass and kernel fusion
- Gradients for the sparse path follow FlashAttention’s derivation (Eq. (7)); for the linear path, they follow Eq. (8).
- Both paths’ forward and backward computations are fused into a single kernel (Section 4; Algorithms 1–2), minimizing memory traffic and launch overhead.

7) Additional efficiency optimizations (Appendix A.3)
- Lookup tables for very sparse masks to avoid scanning zeros.
- Pre-aggregation for the linear path: compute global sums and subtract contributions for `Mc ≠ 0`.
- Method of Four Russians to accelerate partial subset sums when marginal density is moderate.

Why these design choices?
- Block-level masking matches GPU efficiency patterns and FlashAttention’s IO-aware tiling (Section 2.1, 4.1).
- A coarse `Pc` predictor is cheap (pooled Q/K) and good enough to classify blocks by importance, especially after a short fine-tuning phase (Section 5).
- Treating marginal mass with linear attention leverages the observed low-rank structure of small weights (Section 3.2, Figure 3), unlocking very high overall sparsity without quality loss (Figure 2; Table 1).

Mathematical idea in plain words
- Split the attention weights `P` into two parts using a binary sparse mask `M`: the few big entries and the many small ones:
  - `P = (P ⊙ M) + (P ⊙ (1 − M))` (Eq. (1)).
- Compute the big ones exactly (sparse FlashAttention).
- Replace the small, low-rank remainder with linear attention (a low-rank construction), and then learn a small projection to align distributions (Sections 3.2 and 4.2).

## 4. Key Insights and Innovations
- Sparse-few, low-rank-many structure of attention in diffusion transformers
  - Observation: Less than 10% of attention weights are large and have high rank; the remaining >90% form a matrix of extremely low rank (Section 3.2; Figure 3).
  - Evidence: In one sample, full/stable ranks are “Rank = 6226” for full, “Top-8%, Rank = 6230,” but “Bottom-92%, Rank = 9” (Figure 3).
  - Significance: Explains why linear attention alone fails (full attention is high rank) and why sparse-only struggles beyond ~90% sparsity (the “middle” mass still matters; Figure 1 right).
- Three-way classification of blocks with a compressed attention predictor
  - Novel classification of attention blocks into critical, marginal, negligible using `Pc` (Eqs. (2)–(3)). This is neither hand-crafted nor full-resolution—coarse but trainable.
  - Significance: Enables 95% block sparsity at moderate sequence length (~30K tokens) while preserving quality (Table 1; Section 6.2).
- Learnable compensation rather than strict approximation for the marginal mass
  - The linear component is not asked to exactly reproduce masked-out weights; it learns to compensate for their aggregate influence (Section 4.2 “Insight”).
  - Significance: Overcomes known failures of linear attention in diffusion (Limitation L1; Table 2 “Linear Only” fails) by combining it with sparse exact computation and a small projection.
- End-to-end fused kernel and practical training recipe
  - SLA provides fused forward/backward kernels (Algorithms 1–2) and system-level tricks (Appendix A.3) that translate theoretical savings into wall-clock speedups:
    - “13.7× speedup in the attention kernel” and “2.2× end-to-end speedup” (Figure 6).
  - Fine-tuning cost is small: “2,000 steps with batch size 64,” <0.1% of typical pretraining cost (Section 6.3).

These are fundamental innovations (a new structural decomposition and hybrid mechanism) backed by engineering contributions (kernel fusion and optimizations) rather than incremental tweaks to a single attention variant.

## 5. Experimental Analysis
Evaluation setup
- Models and data
  - Video: Wan2.1-1.3B diffusion transformer; fine-tuned on 20,000 five-second videos at 480p (Section 6.1). Typical sequence length is 30K tokens (Section 1).
  - Image: LightningDiT-1.0B on ImageNet 512×512 (Appendix A.2).
- Metrics (Section 6.1)
  - Video quality: VBench dimensions—Imaging Quality (`IQ`), Overall Consistency (`OC`), Aesthetic Quality (`AQ`), Subject Consistency (`SC`)—plus VisionReward (`VR`), Aesthetic Video (`VA`), and Technical Video (`VT`).
  - Efficiency: attention FLOPs; attention kernel FLOPS; end-to-end latency (Figure 6).
  - Image quality: FID.
- Baselines
  - Trainable sparse methods: `VSA`, `VMoBa`.
  - Training-free sparse (`Sparge-F`) and trainable sparse (`Sparge-T`) variants.
  - Ablations: `Linear Only`, `Sparse Only`, and naïve sum `L+S` (Section 6.1).
- SLA hyperparameters
  - `kh = 5%`, `kl = 10%`; `bq = bkv = 64`; activation `φ = softmax` (Section 6.1; Table 2 ablation).

Main quantitative results
- Quality vs. complexity (Table 1)
  - Full attention: `VA 76.78`, `VT 82.88`, `IQ 62.5`, `OC 23.3`, `AQ 56.1`, `SC 93.0`, `VR 0.059`, `FLOPs 52.75T`, `Sparsity 0%`.
  - SLA (95% sparsity): `VA 76.96`, `VT 83.92`, `IQ 62.2`, `OC 23.6`, `AQ 55.9`, `SC 93.1`, `VR 0.048`, `FLOPs 2.74T`.
  - Competing sparse methods at 84–89% sparsity show worse quality; e.g., `VSA (89%)` has `VR −0.069` and lower `VA/VT/IQ` (Table 1).
- Kernel and end-to-end speed (Figure 6)
  - Forward kernel: “13.7× speedup over FlashAttention2” at 95% sparsity.
  - Backward kernel: “6.8× speedup over FlashAttention2.”
  - End-to-end: attention latency drops “from 97s to 11s,” yielding a “2.2×” overall speedup.
- Why sparse-only can’t push sparsity to 95% without hurting quality (Figure 1)
  - Distributional facts in Wan2.1 attention (Figure 1 left):
    - Only “~8.1% of weights are larger than 1/N.”
    - “~45% are below 1/(100N).”
  - Error analysis (Figure 1 right):
    - Dropping the smallest 45% causes “<3%” relative L1 error, but keeping only the largest 8.1% leads to “>33%” error.
- Qualitative comparisons (Figures 2, 5, 7)
  - SLA at 95% sparsity matches the visual quality of full attention.
  - Linear-only and sparse-only baselines degrade severely (Figure 2; Table 2 “Linear Only”).
- Ablations (Table 2)
  - Fusion matters: `Sparse Only (85%)` has `VA 64.00`, while SLA (95%) recovers `VA 76.96`.
  - Activation in linear path: `softmax` best; `elu+1` and `hedgehog` slightly worse.
  - Critical block fraction: `kh=5%` achieves near-full quality with much less compute; increasing to `kh=10%` or `20%` reduces sparsity and does not improve metrics consistently.
- Image generation (Appendix A.2; Table 3)
  - SLA reaches 87.5% sparsity and slightly improves `FID` over full attention (`31.49` vs. `31.87`), outperforming 2D variants of `VSA` and `VMoBa`.

Strength of evidence
- The paper evaluates both kernel-level speed and end-to-end latency, includes diverse quality metrics (VBench + VR), and provides ablations that isolate each architectural choice (Table 2).
- The claim of “negligible” linear attention cost in video models is supported by a concrete example: “less than 0.5% of full attention” in Wan2.1 (Section 3.1, Figure 2 caption).
- Quote the headline results:
  > “SLA reduces attention computation by 95% without degrading end-to-end generation quality” (Abstract, Section 1).
  > “13.7× speedup in attention computation and a 2.2× end-to-end speedup on Wan2.1-1.3B” (Abstract; Figure 6).

## 6. Limitations and Trade-offs
- Dependence on structure of diffusion attention
  - SLA assumes the “sparse-few, low-rank-many” structure (Section 3.2; Figure 3). If a task/model violates this (e.g., more uniformly distributed attention or higher-rank tails), the marginal mass may not be well captured by linear attention.
- Requires fine-tuning
  - SLA is not training-free. It needs modest fine-tuning (2,000 steps) so the model adapts to the hybrid attention and the learned projection (Section 5; Table 2 shows `Linear Only` fails without this hybridization).
- Hyperparameters and mask prediction
  - The `Pc` predictor uses pooled Q/K and per-row TopK/BottomK thresholds (`kh`, `kl`). These introduce tunables that might need task-specific adjustment (Section 6.1; Table 2).
- Workload balance and dimension dependence
  - Linear path cost scales as `O(N d^2)` (Section 2.2). While “<0.5%” of full attention for Wan2.1 (Section 3.1), for models with much larger `d` this may be more material.
- Hardware and integration scope
  - Measured on RTX5090 with FlashAttention2 as the baseline (Figure 6). Speedups may vary with hardware, kernel libraries, and frameworks.
- End-to-end ceiling
  - Even with attention accelerated, end-to-end speedup is 2.2× because other parts of the diffusion model still consume time (Figure 6b). SLA does not address non-attention bottlenecks.

## 7. Implications and Future Directions
- Broader recipe for fast attention: precision where needed, low-rank elsewhere
  - The decomposition `P = (sparse few) + (low-rank many)` (Eq. (1); Figure 3) suggests a general design pattern for attention in long-sequence generative models beyond video diffusion (e.g., audio or multimodal diffusion, possibly LLMs if similar structure holds).
- Train from scratch with SLA
  - The paper fine-tunes pretrained models. A natural extension is pretraining with SLA (especially with the learnable projection), potentially yielding models intrinsically adapted to hybrid attention.
- Smarter or learned block selection
  - `Pc` is computed from pooled Q/K (Eq. (2)). Future work could explore richer predictors (e.g., lightweight cross-attenders or temporal priors), or reinforcement learning to optimize the compute-quality trade-off.
- Combine with quantization and low-precision kernels
  - SLA’s kernel fusions could integrate with quantized attention (e.g., FP8/INT4) to multiply speedups, especially for training (Appendix A.3 hints at systems-level improvements).
- Dynamic or content-aware budgets
  - The thresholds (`kh`, `kl`) could be adapted per layer, head, or time to reflect changing attention patterns as diffusion timesteps evolve.
- Expanded evaluation
  - The video tests use a 20K sample fine-tuning set (Section 6.1). Larger and more diverse datasets, more models, and robustness to different sequence lengths would further validate generality.
- Practical applications
  - Faster, high-quality video diffusion benefits creative tools, real-time or interactive video generation, and scaling to longer clips or higher resolutions without prohibitive latency.

In short, SLA changes the field’s default assumption that one must choose between sparse attention or linear attention. By explicitly modeling attention as “a few high-rank, many low-rank” and using a simple, trainable block classifier, it attains very high sparsity (≈95%) at typical video sequence lengths while keeping quality intact and delivering tangible end-to-end speedups (Table 1; Figure 6).
