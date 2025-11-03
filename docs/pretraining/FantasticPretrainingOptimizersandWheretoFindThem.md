# Fantastic Pretraining Optimizers and Where to Find Them

**ArXiv:** [2509.02046](https://arxiv.org/abs/2509.02046)

## 🎯 Pitch

This paper systematically benchmarks 11 state-of-the-art optimizers for large language model pretraining at scale, revealing that widely cited 1.4–2x speedups over the standard AdamW optimizer vanish after rigorous, fair hyperparameter tuning—with true gains rarely exceeding 1.4x on small models and shrinking to 1.1x at the billion-parameter scale. By exposing how under-tuned baselines and narrow evaluation setups have inflated previous claims, the study provides practitioners and researchers with a robust, scale-aware comparison protocol—ensuring that future optimizer innovations are tested credibly and that pretraining efforts focus on genuinely impactful improvements.

---

## 1. Executive Summary (2-3 sentences)
This paper performs a rigorous, scale-aware benchmark of 11 deep-learning optimizers for large language model (LLM) pretraining and shows that widely claimed 1.4–2× “speedups” over the AdamW baseline largely disappear after fair hyperparameter tuning. Across 0.1B–1.2B parameter models and multiple data-to-model regimes, the real gains top out around 1.4× for small models and shrink to ≈1.1× at 1.2B parameters; matrix-based optimizers (e.g., Muon, Soap) are consistently best on small models, but their advantage diminishes with scale (Figure 1 bottom-left, Figure 3, Figure 4).

## 2. Context and Motivation
- Problem addressed
  - Many recent optimizers promise big pretraining speedups (1.4–2×), yet practitioners still favor AdamW. The paper pinpoints two blockers to fair comparison:
    1) unequal hyperparameter tuning across methods, and
    2) limited or misleading experimental setups (e.g., small models only, judging mid-training checkpoints).
  - Evidence: tuning only the AdamW learning rate in the commonly used GPT‑3 recipe can itself yield “2× speedup” on small models (Figure 1 top-left), implying under-tuned baselines inflate new-method gains.

- Why it matters
  - Pretraining dominates the compute bill for LLMs (e.g., >95% in DeepSeek-V3), so genuine optimizer improvements translate directly to cost savings (Introduction).
  - A dependable comparison protocol informs both research (what’s actually better?) and operations (which optimizer to deploy at scale).

- Prior approaches and gaps
  - Optimizer families:
    - Scalar-based adaptives: AdamW, NAdamW, Lion, Mars, Adam‑mini, Cautious (Table 1).
    - Matrix-based preconditioners: Muon, Soap, Kron/PSGD, Scion (Table 1).
    - Hessian-based approximations: Sophia (Table 1).
  - Where prior work fell short:
    - Baselines often under-tuned or hyperparameters fixed across optimizers (Section 3.2; Figure 1 top-right).
    - Evaluations focused on small models or a single data-to-model ratio, obscuring behavior at realistic scales (Sections 1, 3.4).
    - Mid-training comparisons can flip the ranking later during learning-rate decay (Figure 5 right).

- Positioning
  - This study offers a controlled, end-of-training benchmark across model sizes (≈130M, 300M, 520M, 1.2B; Table 2) and data-to-model ratios (`Chinchilla` ratios 1×–8× and 16× in case studies).
  - It standardizes fair tuning via multi-phase coordinate descent, then extrapolates with fitted hyperparameter scaling laws (Sections 3.2–3.4).

## 3. Technical Approach
This is an empirical benchmark whose credibility rests on: (a) a careful experimental design, and (b) a reproducible, fair hyperparameter tuning protocol.

- Core definitions (paper-specific)
  - `Chinchilla ratio`: number of training tokens relative to the compute-optimal budget from Hoffmann et al. (2022). Here, the optimal tokens are computed as 20× the non-embedding parameter count; training at `n× Chinchilla` means using `n` times that token budget (Section 3.1).
  - `Matrix-based preconditioner`: instead of scaling each parameter update by a scalar (as in AdamW’s per-parameter variance), the optimizer multiplies gradients by a matrix that captures correlations across dimensions (e.g., Muon, Soap, Kron). This can approximate second-order information with manageable cost (Table 1, Appendix A).
  - `Coordinate descent (for tuning)`: optimize one hyperparameter at a time while holding others fixed, accept a change only if the final validation loss improves by more than a small threshold (∆1 = 3e−3 in Phase I; Section 3.2).
  - `Scaling-sensitive` hyperparameters: knobs whose best values change with model size or data budget; identified empirically by tracking near-optimal configurations across regimes (Section 3.3; Table 4).

- Experimental setup
  - Models: Llama-2 style transformers at 130M, 300M, 520M, and 1.2B parameters (Table 2); sequence length 4096; 32 layers (Section 3.1).
  - Data: a large public mix similar to OLMo 2 (DCLM-baseline 3.8T tokens, StarCoder V2 0.25T, ProofPile 2 55B; tokenized with Llama3 tokenizer; Section 3.1).
  - Hardware and precision: JAX on TPU v5; parameters in fp32, activations in bf16 (Section 3.1).
  - Primary metric: final validation loss on C4‑EN (a strong proxy for downstream performance), plus downstream accuracy on ARC-E/C, BoolQ, HellaSwag, LAMBADA, etc. (Section 3.1).

- Three-phase tuning and scaling protocol (Sections 3.2–3.4)
  1) Phase I: exhaustive coordinate descent at small–mid scales
     - Six regimes: 130M/300M/520M models at 1× Chinchilla, and 130M at 2×/4×/8× (Section 3.2).
     - Sweep full hyperparameter grids per optimizer (e.g., for AdamW: learning rate, weight decay, warmup steps, β1, β2, epsilon, grad clip, batch size).
     - Accept changes if final C4‑EN loss improves by > 0.003 (∆1).
     - Output: near-optimal configs per optimizer per regime and identification of which hyperparameters actually matter (Table 4 summarizes scaling-sensitive ones).
  2) Phase II: focused tuning on scaling-sensitive knobs
     - Rationale: most hyperparameters are either insensitive or stable across scales; only re-tune the few that shift (Table 4).
     - Six more regimes (300M/520M at 2×/4×/8×) with sweeps restricted to the sensitive knobs (Section 3.3).
     - Estimating token-efficiency “speedup”: fit an AdamW loss–data scaling curve `L̂_N(D) = α_N D^{-B_N} + β_N` for each model size `N`. Given another optimizer’s achieved loss `L_optimizer` at `D_optimizer`, solve for `D_AdamW` such that `L̂_N(D_AdamW) = L_optimizer`. The ratio `D_AdamW / D_optimizer` is reported as speedup (Section 3.3; Figure 3).
  3) Phase III: hyperparameter scaling to extrapolate
     - Fit scaling laws for each sensitive hyperparameter: `h(N, D) = α N^{-A} D^{-B} + β` (Section 3.4).
     - Validate prediction at 1.2B/1× Chinchilla: predicted vs fully swept; loss within 3e−3 (Section 3.4).
     - Use the scaling laws to run:
       - 1.2B models at 1×–8× Chinchilla for AdamW, NAdamW, Muon, Soap (Figure 4 left/middle; Table 5).
       - 130M/300M at 16× Chinchilla to probe high data-to-model regimes (Figure 4 right; Appendix B.3 Figure 8).

- What’s inside the optimizers (Appendix A; Table 1)
  - Scalar-based (per-parameter) adaptives:
    - `AdamW` (Algorithm 1): first (`m`) and second (`v`) moment estimates; decoupled weight decay.
    - `NAdamW` (Algorithm 2): Nesterov lookahead, which uses a “future” momentum to refine the update.
    - `Mars` (Algorithm 5): variance reduction by combining current and previous gradients `gt` and `gt−1`, controlled by `γ`.
    - `Cautious` (Algorithm 7): mask the update if it disagrees with the current gradient (reduce harmful steps).
    - `Lion` (Algorithm 3): memory-efficient; uses the sign of a momentum blend instead of second moments.
    - `Adam‑mini` (Algorithm 6): second moment tracked per block (one scalar per block) to save memory.
  - Matrix-based preconditioners:
    - `Muon` (Algorithm 8): for each weight matrix (not embeddings/LM head), apply `Newton–Schulz` orthogonalization on a momentum-blended gradient `u` to get a direction with operator-norm control; effectively multiplies by a data-driven matrix that approximates “whitening” of the update.
      - Intuition: transforming gradients by an approximate inverse square-root of a covariance-like matrix to equalize curvature across directions.
    - `Soap` (Algorithm 11): maintains two running gradient covariance estimates (`GA`, `GB`) and factor matrices (`QA`, `QB`) updated by QR steps; preconditions each block as `Q_A^T [m̂ / sqrt(v̂+ϵ)] Q_B`.
      - Intuition: learn left/right whitening transforms that stabilize Shampoo-like preconditioning.
    - `Kron/PSGD` (Algorithm 10): maintains a set of lower-triangular `Q_i` matrices over tensor modes and updates them via sketched curvature terms; preconditions by sequentially multiplying unfolded tensors by `Q_i`.
    - `Scion` (Algorithm 9): uses Muon inside transformer layers but SignSGD on embeddings/LM head.
  - Hessian-based approximation:
    - `Sophia` (Algorithm 4): periodically obtains a diagonal Hessian proxy via randomized Hessian–vector products and clips the step by `max(γ h, ϵ)`.

- Why this approach
  - Enforces optimizer-by-optimizer tuning fairness (Figure 1 top-right shows optimal hyperparameters differ substantially; e.g., Lion’s best weight decay ≈ 0.6 vs AdamW ≈ 0.1).
  - Evaluates end-of-training checkpoints to avoid misleading early-curve rankings (Figure 5 right).

## 4. Key Insights and Innovations
- A. Fair tuning collapses “2× speedups” to ≈1.1–1.4×
  - What’s new: the study shows that a single baseline fix (learning-rate tuning) can “manufacture” 2× speedups on AdamW (Figure 1 top-left), revealing that many prior claims stem from under-tuned baselines rather than algorithmic breakthroughs.
  - Why it matters: establishes a fair, repeatable tuning protocol (Sections 3.2–3.3) and a comparable speedup metric via AdamW-equivalent tokens.

- B. Speedup decays with model size; small models benefit most
  - Evidence: speedups ≤1.4× for 130M–520M (Figure 3) and ≈1.1× at 1.2B (Figure 4 middle).
  - Significance: even the strongest matrix-based methods (Muon, Soap) lose their advantage as models scale.

- C. Matrix-based preconditioners consistently beat scalar methods at small scales and moderate data budgets
  - Evidence: across 130M–520M, matrix-based (solid lines) outperform scalar-based (dashed) for the same token budget (Figure 1 bottom-right; Figure 2 top row). They also yield better token-efficiency growth as data increases (Figure 3).
  - Caveat: the winner changes with data-to-model ratio—Muon at 1–4× Chinchilla; Soap/Kron at higher ratios (Figure 4 right; Figure 1 bottom-right).

- D. Early curves can invert final rankings
  - Evidence: during decay, loss curves cross (Figure 5 right); small LR mis-tuning flips winners (Figure 5 left/middle).
  - Contribution: a concrete methodological warning—only end-of-training comparisons are robust (Section 4.2).

- E. Hyperparameter scaling laws that predict good settings across N and D
  - Method: fit `h(N, D) = α N^{-A} D^{-B} + β` for sensitive knobs (Section 3.4).
  - Impact: enables reliable extrapolation to 1.2B and beyond without re-sweeping everything; validated within 3e−3 of best (Section 3.4).

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and metrics: C4‑EN validation loss as main metric; downstream tasks include LAMBADA, BoolQ, HellaSwag, ARC-E/C, PIQA, WSC273, etc. (Section 3.1; Table 5; Appendix B.4 Tables 6–35).
  - Baselines and competitors: 11 optimizers spanning scalar- and matrix-based categories plus Sophia (Table 1).
  - Setups:
    - Model sizes: 130M/300M/520M (Phase I & II); 1.2B (Phase III) (Table 2).
    - Data budgets: Chinchilla ratios 1×, 2×, 4×, 8× (all sizes) and 16× (case studies for 130M/300M) (Figures 2–4; Appendix B.3 Figure 8).
    - Hardware: TPU v5; mixed precision (Section 3.1).
  - Speedup metric: AdamW-equivalent token ratio using fitted curves per `N` (Section 3.3; Figure 3).

- Main quantitative results
  - Maximum speedups
    - For 130M–520M, the best methods reach at most ≈1.4× token-efficiency vs tuned AdamW (Figure 3, upper color scale).
    - At 1.2B, speedups shrink to ≈1.1× (Figure 4 middle).
  - Matrix vs scalar
    - Consistent advantage of matrix-based at small–mid scales across data budgets 1×–8× (Figure 1 bottom-right; Figure 2 top row).
    - The advantage grows with more data (Figure 3), but declines with model size (Figure 3; Figure 4 middle).
  - Changing winner with data regime
    - At higher data-to-model ratios, Soap (and sometimes Kron) surpasses Muon (Figure 4 right; Appendix B.3 Figure 8).
  - Downstream performance tracks C4‑EN loss
    - HellaSwag accuracy curves mirror the loss trends at 130M/300M/520M (Figure 2 bottom row).
    - At 1.2B/8×, average downstream accuracy shows no improvement for Muon/Soap over AdamW/NAdamW: Table 5 averages are 67.15 (AdamW), 66.70 (NAdamW), 66.98 (Muon). This corroborates that small loss gains at 1.2B do not translate into clear downstream gains.
  - Early-stage comparisons are unreliable
    - Example: in 520M 8×, picking a 2× larger learning rate makes Soap appear worse than Mars mid-training (Figure 5 left); optimizer rankings can reverse during decay (Figure 5 right).

- Ablations and robustness checks
  - Hyperparameter sweeps demonstrate sensitivity and optimizer-specific optima:
    - Learning rate and weight decay optima differ widely between optimizers (Figure 1 top-right).
    - AdamW ablations (Tables 36–41) illustrate how small changes in LR, warmup, weight decay, or batch size move final loss by >0.01–0.05.
    - Scaling-sensitive knobs per optimizer summarized in Table 4 (e.g., `Soap`: learning rate, warmup, block size).
  - Hyperparameter scaling validation: at 1.2B/1×, predicted settings are within 3e-3 loss of a full sweep (Section 3.4).
  - Additional analysis of training dynamics:
    - Parameter norms rise and fall with the learning-rate schedule; gradient norms increase during LR decay without harming loss (Figure 6 middle panels).

- Do the experiments support the claims?
  - Yes. The end-of-training, multi-scale evaluation and the AdamW-equivalent token metric (Figures 2–4) consistently show smaller, scale-dependent gains than prior reports. The 1.2B downstream plateau (Table 5) further tempers expectations.

## 6. Limitations and Trade-offs
- Scope and scaling
  - Model sizes only up to 1.2B (Conclusion). A fitted loss scaling law (Appendix B.1, Eq. (1)) even predicts Muon slightly worse than AdamW at 7B in 1× Chinchilla; this needs direct confirmation.
- Compute accounting
  - Speedup is measured in token efficiency, not wall-clock time. Although matrix-based overhead can be <10% with optimized implementations (Section 3.1), real systems may see higher overheads depending on kernel availability and communication patterns.
- Tuning methodology
  - Coordinate descent finds coordinate-wise local optima, not necessarily the global optimum. Acceptance thresholds (∆1 = 0.003) and grid choices may still leave small gains on the table (Section 3.2).
- Architecture and data specificity
  - Results are for Llama-2–like models with 32 layers and 4096 context, trained on a specific public mixture; behavior may differ for other architectures (e.g., MoE), context lengths, or data curation strategies (Section 3.1).
- Batch-size/hardware regime interactions
  - Concurrent benchmarking differences appear linked to batch size and hardware (Discussion of Semenov et al. [2025], Section “Comparison with concurrent work”): variance-reduction methods (e.g., Mars) can benefit more in small-batch, noise-dominated regimes. This study primarily uses ≥0.4M-token batches on TPU v5 (Section 2, concurrent work comparison).

## 7. Implications and Future Directions
- For practice
  - Expect modest, context-dependent gains: after fair tuning, typical improvements are ≈1.1–1.4× tokens, largest for smaller models and moderate data budgets (Figure 3). At 1.2B, benefits narrow to ≈1.1× and don’t reliably improve downstream metrics (Figure 4 middle; Table 5).
  - Choose optimizer by regime:
    - If training ≤500M parameters at 1–4× Chinchilla, consider matrix-based methods (Muon/Soap/Kron) for consistent gains (Figure 1 bottom-right; Figure 2).
    - At high data-to-model ratios (≥8×–16×), Soap or Kron may beat Muon (Figure 4 right; Appendix B.3 Figure 8).
    - If memory is tight, Adam‑mini/Lion can approximate AdamW with small losses (Figure 2 top row).
  - Always tune: sharing hyperparameters across optimizers is unfair and can flip rankings (Figure 1 top-right; Figure 5).

- For research
  - Scale-stable optimization: design matrix-level preconditioners whose advantage does not decay with model size (Figure 4 middle).
  - Better theory and proxies: unify why gradient norms grow during decay without hurting loss (Figure 6) and relate optimizer choices to generalization more directly at scale (Table 5 shows plateaus).
  - Robust evaluation standards: end-of-training comparisons, AdamW-equivalent token metrics, and multi-regime tests (across `N` and Chinchilla ratios) should become standard.

- Potential applications
  - Cost-sensitive pretraining at small–mid scales (e.g., 0.1–0.5B params) can benefit from matrix-based optimizers for ≈1.3–1.4× token-efficiency (Figure 3).
  - Hyperparameter scaling laws (Section 3.4) enable predictable deployments across sizes and budgets with fewer sweeps.

> Representative highlights:
> - “Tuning LR of AdamW leads to 2× speedup” on a 130M model (Figure 1 top-left).
> - “Optimizers’ speedup … decreases with model size” and is ≈1.1× at 1.2B (Figure 1 bottom-left; Figure 4 middle).
> - “Matrix-based optimizers … consistently outperform scalar-based optimizers” at 130M–520M (Figure 1 bottom-right; Figure 2).
> - At high data-to-model ratios (16×), Soap surpasses Muon (Figure 4 right; Appendix B.3 Figure 8).
> - 1.2B downstream averages show no advantage for Muon over AdamW/NAdamW (Table 5).
