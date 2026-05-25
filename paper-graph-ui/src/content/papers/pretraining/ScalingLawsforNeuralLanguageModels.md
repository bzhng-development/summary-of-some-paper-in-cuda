# Scaling Laws for Neural Language Models

**ArXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)

## 🎯 Pitch

This paper uncovers precise empirical laws that govern how the performance of Transformer language models scales as you grow model size, dataset size, and training compute. By revealing simple power-law relationships across up to eight orders of magnitude, it provides actionable formulas for predicting model performance and, crucially, shows how to optimally allocate compute—favoring much larger models trained on relatively modest datasets, and stopping training early. These insights not only save resources but also reshape how researchers and practitioners design and scale up language models, offering a universal framework that advances both the science and practice of deep learning.

---

## 1. Executive Summary (2-3 sentences)
This paper empirically maps how the test loss of Transformer language models scales with three knobs you can turn during training: model size (`N` parameters), data size (`D` tokens), and training compute (`C` FLOPs). Across 6–8 orders of magnitude, it finds simple power-law equations that predict performance and prescribe how to allocate a fixed compute budget—train very large models on comparatively modest data, stop early, and use batch sizes near a “critical” value.

## 2. Context and Motivation
- Problem addressed
  - Practitioners must decide how big a model to train, how much data to use, how long to train, and how to size batches. Before this work, there was no robust, quantitative recipe that predicted how test loss improves when any of these are scaled, or how to best spend a fixed compute budget.
- Why it matters
  - Practical: Training trillion-parameter models costs millions of dollars and months of time. A predictive “scaling law” lets one forecast returns before training and allocate compute efficiently.
  - Scientific: Power laws that persist across architectures and scales hint at universal behavior—useful for theory-building (akin to thermodynamic laws). The paper frames such regularities for language modeling.
- Prior approaches and gaps
  - Prior empirical hints about scaling with data or model size existed in narrower settings (e.g., [Hestness et al. 2017]), but they did not:
    - Jointly model performance vs. model size, data size, training steps, and compute.
    - Provide a compute-optimal allocation strategy.
    - Establish universality across six-to-eight orders of magnitude with Transformers.
- Positioning relative to existing work
  - Extends and unifies earlier observations into a single framework with explicit equations:
    - Three single-variable laws for `L(N)`, `L(D)`, and compute-optimal `L(Cmin)` (Figure 1; Equations (1.1)–(1.3)).
    - A joint law for loss vs. model size and data size `L(N, D)` (Equation (1.5), Figure 4/9).
    - A training-time law `L(N, Smin)` (Equation (1.6), Figure 4 right; Table 3).
    - A batch size law `Bcrit(L)` (Equation (1.4), Figure 10).

## 3. Technical Approach
This is a large, carefully controlled empirical study backed by simple, interpretable formulas. The key moving parts:

- Core setup (Section 2)
  - Models: Decoder-only Transformers with standard components; sizes from ~10^3 to ~10^9 non-embedding parameters; variants across depth/width/heads/context (Figures 5–6).
  - Non-embedding parameters: The parameter count excludes token and positional embedding matrices. This choice yields clean scaling trends (Figure 6 right) whereas including embeddings confounds depth effects (Figure 6 left).
  - Loss: Cross-entropy loss in nats per token over 1024-token contexts.
  - Compute proxy: `C ≈ 6NBS`, where `N` is non-embedding parameters, `B` is tokens per batch, `S` is parameter update steps. The factor 6 accounts for forward/backward passes (Section 2.1).
  - Data: WebText2 (2.29×10^10 tokens total; 6.6×10^8 tokens held out for test). Additional out-of-distribution test sets include Books, Internet Books, Wikipedia, Common Crawl (Section 2.3; Figure 8).
  - Training: Mostly Adam with warmup+cosine decay, 2.5×10^5 steps, batch size 512×1024-token sequences; Adafactor for the very largest models (Section 2.2).
- The scaling-law program (Sections 3–6)
  1) Measure single-factor laws where other factors are not limiting:
     - `L(N)` at effectively infinite data and long training (Equation (1.1), Figure 1 right).
     - `L(D)` with early stopping to avoid overfitting (Equation (1.2), Figure 1 middle).
     - `L(C)` at fixed batch size; then correct to compute-optimal `L(Cmin)` using a batch-size adjustment (Equations (5.5), (1.3); Figure 13).
  2) Joint laws and training dynamics:
     - Derive and validate a two-argument loss surface `L(N, D)` that reduces to the single-variable limits and captures overfitting (Equation (1.5); Figure 4 left, Figure 9).
     - Characterize learning curves as a function of steps at the critical batch size: `L(N, Smin)` (Equation (1.6); Figure 4 right; Table 3).
  3) Batch size and optimization efficiency:
     - Measure the “critical batch size” `Bcrit(L)`—the point up to which increasing batch size yields near-linear speedups, beyond which returns diminish (Section 5.1).
     - Empirically, `Bcrit(L)` follows a power law in loss (Equation (1.4), Figure 10) and aligns with the “gradient noise scale” as predicted by a prior model of large-batch training [McCandlish et al. 2018]. This enables two conversions:
       - Convert a run at batch `B` and steps `S` to the minimal steps `Smin` it would have needed at very large batch: `Smin(S) = S / (1 + Bcrit(L)/B)` (Equation (5.4)).
       - Convert to the minimal compute `Cmin` it would have needed at very small batch: `Cmin(C) = C / (1 + B/Bcrit(L))` (Equation (5.5)).
  4) Compute-efficient frontier:
     - Using `L(N, Smin)` together with `Bcrit(L)`, derive how to allocate a fixed compute budget across model size `N`, batch size `B`, and steps `S` to minimize loss (Equations (1.7)–(1.8); Figure 14).
     - Key conclusion: spend most compute on model size; increase batch size substantially; increase the number of serial steps very slowly.

- Why these designs
  - Excluding embedding parameters isolates the capacity that directly scales compute and learning dynamics (Figure 6).
  - Early stopping when varying `D` prevents conflating optimization underfitting with data overfitting (Section 4.2; Figure 16).
  - The `L(N, D)` form is chosen to:
    - Reduce to the single-variable laws in the `N → ∞` or `D → ∞` limits.
    - Admit a series expansion in `1/D` (overfitting treated as variance-like, proportional to `1/D`) (Section 4.1).

## 4. Key Insights and Innovations
- Universal power laws with actionable exponents
  - What’s new: Precise power-law relations between test loss and each of model size, dataset size, and compute—each holding over many orders of magnitude (Figure 1).
  - Why it matters: Each exponent quantifies the “return on investment”:
    - Doubling `N` improves loss by a factor `2^{-αN}`; `αN ≈ 0.076` (Equation (1.1)).
    - Doubling `D` improves loss by `2^{-αD}`; `αD ≈ 0.095` (Equation (1.2)).
    - Increasing compute on the optimal frontier improves loss as `Cmin^{-αCmin}`; `αCmin ≈ 0.050–0.054` (Equations (1.3), (6.3)–(6.4); Figure 13).
- A joint law for loss vs. model and data size that quantifies overfitting
  - Innovation: The two-argument formula
    - `L(N, D) = [(Nc/N)^{αN/αD} + (Dc/D)]^{αD}` (Equation (1.5))
    - It accurately predicts early-stopped test loss surfaces (Figure 4 left; Figure 9 left).
  - Significance: It yields a simple overfitting control rule:
    - Overfitting depends primarily on `N^{αN/αD}/D` (Figure 9 right). To keep it constant, scale `D ∝ N^{αN/αD} ≈ N^{0.74}` (Section 4.2; Equation (4.4)).
- Universality of training dynamics and the critical batch size
  - Finding: Learning curves at large batch collapse to a two-term power law:
    - `L(N, Smin) = (Nc/N)^{αN} + (Sc/Smin)^{αS}` with `αS ≈ 0.76`, `Sc ≈ 2.1×10^3` (Equation (1.6); Table 3; Figure 4 right).
  - Batch size law: `Bcrit(L) = B* L^{1/αB}` with `B* ≈ 2×10^8` tokens and `αB ≈ 0.21` (Equation (1.4), Figure 10).
  - Impact: Enables batch/step/compute conversions (Equations (5.4)–(5.5)) and a principled choice of batch size near `Bcrit`.
- Compute-optimal training stops early and favors larger models
  - Empirical and derived allocation (Equations (1.7)–(1.8); Figure 14; Section 6):
    - `N ∝ Cmin^0.73`, `B ∝ Cmin^0.24`, `S ∝ Cmin^0.03`, `D = B·S`.
  - This is a fundamental shift: most additional compute should increase parameters; the number of serial optimization steps grows extremely slowly. Training “to convergence” is compute-inefficient (Appendix B.3).
- Weak dependence on architectural shape
  - At fixed non-embedding parameter count, depth/width/heads/FFN ratios matter little: loss varies by only a few percent over wide ranges (Figure 5). This de-emphasizes architecture search relative to scaling.

## 5. Experimental Analysis
- Evaluation methodology
  - Data: WebText2 (2.29×10^10 tokens) with 6.6×10^8 test tokens; out-of-distribution tests on Books, Internet Books, Wikipedia, Common Crawl (Section 2.3; Figure 8).
  - Metric: Cross-entropy loss (“nats per token”; lower is better).
  - Setups: Systematic sweeps over model size (10^3 to 10^9 parameters), data subsets (2.1×10^7 to 2.2×10^10 tokens), and compute budgets; early stopping when comparing `D` (Figure 9), long training when comparing `N`.
  - Batch scans to measure `Bcrit(L)` and validate the noisy-quadratic “critical batch size” relationship (Figure 18 → Figure 10).
- Main quantitative results
  - Single-factor power laws (Figure 1; Equations (1.1)–(1.3)):
    - `L(N) = (Nc/N)^{αN}` with `αN ≈ 0.076`, `Nc ≈ 8.8×10^13`.
    - `L(D) = (Dc/D)^{αD}` with `αD ≈ 0.095`, `Dc ≈ 5.4×10^13`.
    - Compute-adjusted: `L(Cmin) ∝ Cmin^{-0.050}` empirically (Figure 13, dashed fit), with a theoretical prediction `αCmin ≈ 0.054` (Equation (6.4)).
  - Joint surfaces:
    - `L(N, D)` fits well over two orders of magnitude in `D`; fitted parameters `αN=0.076, αD=0.103, Nc=6.4×10^13, Dc=1.8×10^13` (Table 2; Figure 9).
    - Learning curves: `L(N, Smin)` fits with `αS=0.76, Sc=2.1×10^3` (Table 3; Figure 4 right).
  - Critical batch size:
    - Empirical `Bcrit(L)` doubles for every ~13% decrease in loss; independent of model size at fixed loss (Figure 10).
  - Compute-efficient allocations (Figure 14; Section 6.1):
    - `N(Cmin) ∝ Cmin^0.73` (left panel).
    - `Smin(Cmin) ∝ Cmin^0.03` (right panel).
    - Deviation from optimal size costs little: models 0.6× to 2.2× optimal need only ~20% extra compute to hit the same loss (Figure 12 left).
  - Generalization:
    - Loss on other datasets improves in near-lockstep with training-distribution loss—a roughly constant offset across scales (Figure 8 left).
    - Generalization tracks in-distribution validation loss during training; proximity to convergence and network depth do not matter (Figure 8 right; Appendix D.8).
  - Architecture and baselines:
    - LSTMs match Transformers on early-context tokens but plateau by ~100 tokens; Transformers keep improving across the entire 1024-token context and asymptotically win (Figure 7).
    - Shape invariance at fixed `N` (Figure 5); embedding parameters confound scaling (Figure 6).
- Support strength and robustness
  - Trends span 6–8 orders of magnitude with consistent slopes, survive changes in depth/width/heads/context (Figures 1, 5, 6, 8).
  - Batch-size–adjusted compute and steps align with independent “gradient noise scale” theory (Figure 10; Section 5.1).
  - Caveat: Very small datasets (~2×10^7 tokens) behave differently (poor fits; early overfitting) (Table 2 note; Figure 16).
- Notable ablations or diagnostics
  - Early stopping vs overfitting gap: A lower bound on early-stopping step relates the excess loss from finite data to steps needed (Equation (5.7); Figure 16 left).
  - Per-token loss patterns show power-law dependence on context position and faster short-range learning early in training (Figure 20–21).

## 6. Limitations and Trade-offs
- Assumptions
  - Stationarity of scaling exponents over large ranges. While supported up to the study’s scale, the exponents may drift at larger scales or with different tokenization/data distributions (Section 8; Appendix C).
  - Early stopping and fixed 10% dropout as the primary anti-overfitting mechanism (Section 4.2). Different regularization/augmentation might alter `L(N, D)`.
  - Compute proxy `C ≈ 6NBS` ignores context-length–dependent terms, which grow with `nctx` and could matter when contexts become very long (Appendix C).
- Scope limits and edge cases
  - Small-data regime: Behavior differs when an epoch is only ~40 steps; the `L(N, D)` fit degrades (Table 2; Figure 16).
  - Extrapolation uncertainties:
    - `Bcrit(L)` extrapolated outside observed loss range could shift step/compute trade-offs (Appendix C).
    - The constants `Nc`, `Dc` depend on tokenization and vocabulary size (Section 1.2), limiting cross-setup comparability of absolute numbers (though exponents tend to be robust).
- Potential contradiction at extreme scale
  - If one continually follows the compute-optimal policy, data used per training run grows only as `D(Cmin) ≈ (4×10^10)·(Cmin/PF‑day)^0.26` tokens (Equation (6.7)), much slower than the data needed to keep overfitting constant `D ∝ N^0.74 ∝ Cmin^0.54` (Equation (6.6)). This implies the `L(Cmin)` curve must eventually be capped by `L(D)`, producing an intersection around
    - `C* ~ 10^4 PF‑days`, `N* ~ 10^12`, `D* ~ 10^12` tokens, `L* ~ 1.7 nats/token` (Figure 15; Equation (6.8)).
  - The paper treats this as a likely breakdown point for the simple power laws; the exact location is sensitive to exponents (Figure 15 caption).
- Practical costs
  - Compute-optimal training favors massive models; even with short training, wall-clock time, memory, and parallelism requirements are substantial. Hardware and engineering constraints can force suboptimal allocations (Figure 12; Section 6.1).

## 7. Implications and Future Directions
- How it changes the field’s practice
  - Planning: The power laws give a quantitative playbook for budget allocation. For a fixed compute budget, prioritize parameters, scale batch near `Bcrit`, and stop far short of convergence (Equations (1.7)–(1.8); Figure 14).
  - Expectations: Larger models are more sample-efficient—needing fewer updates and fewer unique tokens to hit the same loss (Figure 2; Figure 19). This reframes “data vs. model size”: big models can do more with less data when trained compute-efficiently.
  - Architecture search de-emphasized: At fixed `N`, shape choices have minor effect (Figure 5).
- Research avenues
  - Theoretical foundations: The persistence of power laws suggests an underlying “statistical mechanics” of optimization and generalization. The learning-curve exponent `αS` likely reflects the Hessian spectrum; making this precise is an open problem (Section 5.2).
  - Cross-domain tests: Do the same laws hold for images, audio, video, or multimodal models? The paper conjectures portability to other maximum-likelihood generative modeling tasks (Section 8).
  - Beyond the intersection point: Investigate how `L(Cmin)` saturates—e.g., by modeling intrinsic “noise floors” or non-text entropy—and whether different training data, curricula, or architectures shift the critical point (Section 6.3).
  - Training systems: Since `S` grows slowly with budget (≈ `Cmin^0.03`), throughput improvements should target parallelism and model/distributed systems (pipeline/tensor parallel, sparsity/mixture-of-experts) to enable much larger `N` (Discussion; [Huang et al. 2018], [Shazeer et al. 2018]).
- Applications and downstream use
  - Forecasting for large model programs: Organizations can use the equations to estimate the loss (and thereby downstream task performance proxies) achievable for a given budget, then choose `N`, `B`, `S`, and `D` accordingly.
  - Dataset strategy: To avoid overfitting penalties when scaling `N`, expand or diversify data roughly as `D ∝ N^0.74` (Equation (4.4)), though compute-efficient training suggests using far less and stopping early when compute-limited.
  - Evaluation: Since out-of-distribution losses track in-distribution with roughly constant offsets (Figure 8), monitoring one can predict the other during training.

Quoted highlights tied to figures/equations
- “Performance depends strongly on scale, weakly on model shape” (Section 3; Figure 5).
- “Smooth power laws” (Figure 1; Equations (1.1)–(1.3)): `αN ≈ 0.076`, `αD ≈ 0.095`, `αCmin ≈ 0.050`.
- “Universality of overfitting” captured by `L(N, D)` and the ratio `N^{0.74}/D` (Equation (1.5); Figure 9 right).
- “Universality of training”: `L(N, Smin)` with `αS ≈ 0.76` (Equation (1.6); Table 3).
- “Compute-efficient training stops far short of convergence” with `N ∝ Cmin^0.73`, `B ∝ Cmin^0.24`, `S ∝ Cmin^0.03` (Equations (1.7)–(1.8); Figure 14; Figure 3).
- “Optimal batch size” grows with lower loss: `Bcrit(L) = B* L^{1/αB}`; `B* ~ 2×10^8`, `αB ~ 0.21` (Equation (1.4); Figure 10).
