# Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training

**ArXiv:** [2505.23971](https://arxiv.org/abs/2505.23971)

## 🎯 Pitch

This paper introduces a direct, empirical method for measuring the critical batch size (CBS) during language model training, sidestepping the strong assumptions required by previous proxies like the gradient noise scale. By using these CBS measurements to design a 'batch size warmup' schedule, the authors demonstrate that language models can be trained with significantly fewer gradient steps—improving efficiency and scalability—without compromising, and even slightly improving, final model performance. This advance provides both a practical tool for faster, more cost-effective large-batch training runs and a clearer understanding of optimization dynamics in modern large language models.

---

## 1. Executive Summary
This paper introduces a simple, direct way to measure the “critical batch size” (CBS) during large language model (LLM) training and uses those measurements to design a safe “batch size warmup” schedule. The method avoids strong assumptions behind prior CBS proxies, and in controlled OLMo experiments it enables training with 43% fewer gradient steps while slightly improving final loss compared to a small-batch baseline (Table 1).

## 2. Context and Motivation
- Problem addressed
  - Large batch sizes increase training throughput by letting more data be processed in parallel, but if the batch is too large, token efficiency degrades (i.e., more tokens are needed to achieve the same loss). The open question is how to pick a batch size that is as large as possible without sacrificing token efficiency.
  - The concept of a “critical batch size” `B*` is central: up to `B*`, increasing batch size (with an appropriate learning-rate scaling rule) does not harm the loss-versus-tokens trajectory; above `B*`, it does (Critical Batch Size Hypothesis in Section 1).

- Why it matters
  - Real-world impact: Choosing batch size well can reduce wall-clock time and increase data-parallel scaling without inflating the compute budget to reach a target loss.
  - Theoretical significance: Understanding when and how batch size can be safely increased clarifies optimization dynamics in overparameterized, adaptive-optimizer-trained LLMs.

- Prior approaches and their limits
  - Exhaustive measurement: Run many separate training jobs to a common target loss at different batch sizes (e.g., Zhang et al., 2019; 2024). This is accurate but very expensive (Section 2).
  - Gradient noise scale proxy: Estimate `B*` via the ratio of gradient variance to squared mean, e.g., `B_simple = tr(Σ) / ||G||^2` (McCandlish et al., 2018). Two key assumptions limit its validity for LLM pretraining (Section 2):
    - Optimizer assumption: Treats the update as SGD. In practice, LLMs use Adam, for which the principled learning-rate scaling differs (square-root rather than linear; Malladi et al., 2022).
    - Conditioning assumption: Assumes a well-conditioned Hessian (effectively proportional to the identity) so that `B_noise = tr(ΣH) / (G^T H G)` reduces to `B_simple`. This is unlikely to hold at LLM scale.
  - In short, the proxy is attractive but rests on assumptions that are not met in common LLM setups.

- Positioning of this paper
  - The paper proposes an empirical, low-overhead measurement of CBS that:
    - Works with Adam and typical LLM training pipelines.
    - Measures the “local” CBS at different points in a single training run (rather than only at initialization).
    - Directly checks whether larger batches preserve loss after a short recovery window, avoiding the theoretical assumptions needed by noise-scale proxies (Section 3).

## 3. Technical Approach
The paper develops a “branched training” procedure to measure CBS at a chosen checkpoint and then uses the resulting CBS curve to guide batch size warmup.

- Key term definitions (selective)
  - `critical batch size (CBS)`: The largest batch size such that, when the learning rate is scaled appropriately, the loss-vs-tokens trajectory matches that of a smaller batch size (Section 1).
  - `token efficiency`: How much loss reduction is achieved per token seen during training.
  - `branched training`: From a saved checkpoint, start multiple short training runs (“branches”), each with a different batch size and appropriately scaled learning rate, and compare their losses after a fixed token budget.
  - `local recovery assumption`: If a larger-batch branch recovers to the same smoothed loss as the baseline within a short window `Δ` tokens, it will continue to track thereafter (Section 3.1).

- Measuring local CBS via branched training (Section 3.1; Figure 1)
  1. Choose a checkpoint from an ongoing or prior run with base batch size `B` and learning-rate schedule `η`.
  2. For a set of multipliers `k` (e.g., 1, 2, 3, …), create branches with:
     - New batch size `k·B`.
     - New base learning rate `f(k)·η`, where `f(k)` is the optimizer-specific scaling rule:
       - For SGD: `f(k) = k` (linear).
       - For Adam: `f(k) = sqrt(k)` (square-root), supported by SDE analysis (Malladi et al., 2022).
  3. Train each branch for a short fixed token budget `Δ` to allow the optimizer state to adapt:
     - The paper uses `Δ = 2B` tokens, a conservative and cheap window relative to full pretraining budgets (Section 3.1).
  4. Compute the smoothed loss `L_k` using an exponential moving average with parameter `α = 0.5` (to reduce noise).
  5. Tolerance for “no degradation”: Two losses are considered “the same” if they differ by at most `ε = 0.01`.
  6. Select the largest `k*` such that for all `k < k*`, `L_{k*} ≤ L_k + ε`. Define `B* = k*·B` (and the corresponding `η* = f(k*)·η`).

  Intuition: If a larger batch can recover to the baseline loss after `Δ` tokens—even after the transient disruption of changing optimizer statistics—it is deemed safe. If loss rises and stays above the baseline beyond the tolerance, that batch is deemed too large (Figure 1 shows the rise point as the red dotted line).

- Assumptions and design choices (Section 3.1)
  - Only assumption: the local recovery assumption (defined above).
  - `Δ = 2B` tokens: balances allowing optimizer adaptation against added measurement cost.
  - `ε = 0.01`: a pragmatic tolerance. A statistical test could replace this in future work.
  - EMA smoothing (`α = 0.5`): stabilizes noisy pretraining loss series.

- From local CBS to a training policy: batch size warmup (Section 4.1)
  - Idea: Since the CBS starts small and grows then plateaus (observed empirically in Figure 2), begin training with a small batch and double it only when the measured CBS supports it, adjusting the base learning rate with the square-root rule for Adam.
  - Procedure (Section 4.1):
    - Start at `B0` and `η0`.
    - If at time `t` the measured `B*_t` exceeds `2·B_t`, then set `B_{t+1} = 2·B_t` and `η_{t+1} = sqrt(2)·η_t`.
    - This ensures the run never uses a batch size that exceeds the current CBS.

- Experimental setup to measure CBS (Section 3.2; Appendix A)
  - Models: OLMo 1B and OLMo 7B (open weights/data).
  - Measurements: At many checkpoints across training, run branches over multipliers `k` (details and grids differ for 1B vs 7B; Appendix A).
  - Units: CBS is reported in “documents,” where each document contains 4096 tokens (Figures 2 and 5).

- Optional theoretical link to global scaling (Appendix D)
  - If the local CBS over training behaves like `f(t) ~ t^c`, then the single fixed batch size minimizing L2 deviation from the local CBS scales as `B* ~ T^c/(c+1)`. For `c = 1/2`, this yields `B* ~ (2/3)·√T`, consistent with earlier aggregate CBS ∝ √T findings (Appendix D, Proposition 2).

## 4. Key Insights and Innovations
- Direct, assumption-light CBS measurement
  - Innovation: A practical branched-training protocol that empirically identifies the largest safe batch size at any point in training with a small compute overhead (Section 3.1).
  - Why it matters: It avoids the two strong assumptions (SGD-only and well-conditioned Hessian) needed for gradient noise scale proxies (Section 2), making it applicable to Adam-based LLM pretraining.

- Characterization of CBS evolution during training
  - Finding: For both OLMo 1B and 7B, the CBS “starts near 0, grows rapidly but diminishingly, and plateaus around 4096” documents (Figure 2).
  - Significance: This shape suggests a natural warmup schedule—small batch early, then double a few times, then stay constant. It also indicates that smaller pilot runs can forecast the CBS trend for larger models (Section 3.3).

- Empirical evidence that gradient noise scale is unreliable as a CBS proxy in this setting
  - Result: The measured gradient noise scale “underestimates the CBS by several orders of magnitude” and its trend often does not match, especially for OLMo 7B (Figure 3; Appendix B details).
  - Impact: Challenges common practice that relies on noise-scale estimates to set batch size at LLM scale.

- Batch size warmup validated in an LLM pretraining setting
  - Contribution: A two-step doubling schedule (1024 → 2048 at 168B tokens, → 4096 at 503B tokens) trained OLMo 1B to slightly better loss while saving 43% of gradient steps vs. the small-batch baseline (Section 4.3; Figure 4; Table 1).
  - Importance: Demonstrates a compute-efficient path to large-batch training without harming token efficiency.

## 5. Experimental Analysis
- Evaluation methodology
  - CBS measurement (Section 3.2; Figure 2; Appendices A,B)
    - OLMo 1B and 7B checkpoints throughout training.
    - For each checkpoint, multiple branches at different `k`; compute smoothed loss after `Δ = 2B` tokens; pick `B*` via tolerance rule `ε = 0.01`.
    - Also compute gradient noise scale via the McCandlish et al. estimator with `B_big = 64`, `B_small = 1`, averaging over 4096 batches and reporting 95% CIs (Appendix B).
  - Warmup vs. baselines (Section 4.2; Figure 4)
    - Three runs on OLMo 1B:
      - Batch Size Warmup: start at `B=1024`, then double at ~168B and ~503B tokens; Adam square-root scaling; cosine LR schedule remains (Figure 4, left).
      - Small-Batch Control: `B=1024`, base LR `η = √2·0.0004`.
      - Large-Batch Control: `B=4096` from the start, base LR `η = 2√2·0.0004`.
    - Training budget: pretraining for 608B tokens; then a “mid-training” phase of 50B tokens with LR linearly annealed to zero while keeping the final batch size fixed (Section 4.2, “Loss after Mid-Training”).
    - Metrics:
      - Training loss at the end of pretraining and after mid-training (moving average over last 10B tokens).
      - Out-of-distribution losses: cross-entropy on C4 and The Pile; bits-per-byte (BPB) on gold answers of multiple QA/MC tasks following Bhagia et al. (2024) (Table 2; Appendix E lists tasks).

- Main quantitative results
  - CBS evolution (Figure 2)
    - Quote:
      > “The CBS starts near 0, grows rapidly but diminishingly, and plateaus around 4096.”
    - Similar qualitative shape for 1B and 7B, suggesting weak dependence on model size within this range (Section 3.3; Figures 2, 5, and 6 in Appendix).
  - Noise scale comparison (Figure 3; Appendix B)
    - Quote:
      > “The gradient noise scale underestimates the CBS (cf. Figure 2) and the qualitative trend does not clearly match, especially for OLMo 7B.”
    - Confidence intervals shown; estimator details (Appendix B) indicate careful variance reduction and CI construction, yet mismatch persists.
  - Warmup vs. baselines (Figure 4; Table 1)
    - Table 1 (exact numbers):
      > Batch Size Warmup: PT Loss 2.5891; MT Loss 2.5433; Gradient Steps Saved 43%  
      > Small-Batch Control: PT Loss 2.6057; MT Loss 2.5486; Gradient Steps Saved 0%  
      > Large-Batch Control: PT Loss 2.5962; MT Loss 2.5506; Gradient Steps Saved 75%
    - Interpretation:
      - Warmup slightly outperforms both controls in final loss (after mid-training) while saving 43% of gradient steps.  
      - Large-batch control saves more steps (75%) but exhibits worse loss than warmup and small-batch, consistent with exceeding CBS early in training (Section 4.3; Figure 4 right).
  - Out-of-distribution results (Table 2)
    - Quote (selected values):
      > “Batch Size Warmup (Ours) generally performs comparably or better compared to the small-batch control,” e.g., BPB on downstream tasks after mid-training: 1.0076 (warmup) vs 0.9999 (small-batch) vs 1.0193 (large-batch); C4 loss after mid-training: 2.7597 (warmup) vs 2.7622 (small-batch) vs 2.7658 (large-batch); The Pile loss after mid-training: 2.1521 (warmup) vs 2.1471 (small-batch) vs 2.1586 (large-batch).
    - Takeaway:
      - Performance is comparable across conditions; warmup does not degrade held-out cross-entropy or task BPB and is often slightly better than large-batch control.

- Convincingness and robustness
  - The CBS trend is consistent across two model sizes and many checkpoints (Figure 2; Appendix A plots).
  - The warmup policy is grounded in those local CBS measurements and validated end-to-end (Figure 4; Tables 1–2).
  - The gradient noise scale mismatch is substantial and repeated at many checkpoints (Figure 3), undermining its reliability as a proxy under Adam.

- Ablations and sensitivity
  - Method hyperparameters (`Δ=2B`, `ε=0.01`, EMA `α=0.5`) are fixed choices justified pragmatically (Section 3.1). Sensitivity analyses are suggested for future work but not reported.
  - Warmup thresholds are “heuristically” chosen from the CBS curves (Section 4.1, Implementation Details), not via an automated online detector.

## 6. Limitations and Trade-offs
- Reliance on the local recovery assumption (Section 3.1)
  - The method assumes that if loss equalizes within `Δ` tokens after a batch change, trajectories will continue to match thereafter. This is plausible but unproven; corner cases (e.g., long-horizon optimizer dynamics) might violate it.

- Measurement hyperparameters may affect CBS estimates
  - `Δ` (window length): Larger `Δ` might allow recovery at larger batches, increasing estimated `B*`; smaller `Δ` could be overly conservative (Section 3.1).
  - `ε` (loss tolerance) and EMA smoothing can change the pass/fail boundary for “no degradation.”

- Manual thresholding for warmup
  - The two doubling points in the main experiment are chosen manually from offline CBS measurements (Section 4.1). An online, automated detector is feasible but not implemented.

- Scope of validation
  - Models: OLMo 1B and 7B. While these are meaningful LLMs, broader architectural and dataset diversity (e.g., different tokenizers, sequence lengths, or data regimes) is not covered.
  - Training budget: The warmup experiment runs to 608B tokens plus 50B anneal, not the full 4T tokens of the original OLMo 1B recipe (Section 4.2), though mid-training anneal is intended to approximate later loss improvements.

- Compute overhead of measuring CBS
  - Although far cheaper than running full separate trainings, the branched runs still require extra compute. The cost scales with the number of checkpoints and `k` values tested (Appendix A).

- Gradient-noise-scale baseline configuration
  - The noise-scale estimator uses `B_small=1`, `B_big=64`, and assumes distributional forms to compute CIs (Appendix B). While reasonable, alternative estimators (e.g., Gray et al., 2023; 2024) or larger `B_big` might change absolute values. Nonetheless, the observed order-of-magnitude gap and trend mismatch are hard to reconcile.

## 7. Implications and Future Directions
- How this changes practice
  - Provides a practical, optimizer-aware way to choose and adapt batch size during LLM pretraining without relying on questionable proxies. Practitioners can:
    - Run short branched probes early and mid-training to map `B*` over time.
    - Use batch size warmup that never exceeds the measured CBS, preserving token efficiency while scaling data parallelism.

- Theoretical and empirical follow-ups
  - Formalize and test the local recovery assumption, including longer `Δ` windows, different optimizers (e.g., AdamW variants), and schedules.
  - Automate online CBS estimation and triggering (e.g., adaptive `Δ`, statistical hypothesis tests instead of a fixed `ε`, and curve-fitting of CBS vs. tokens).
  - Extend Appendix D’s connection between local CBS curves and global CBS scaling laws; explore asymmetric penalties for training above vs. below CBS.

- Broader applications
  - Training infrastructure: Cluster schedulers could automatically ramp batch size as CBS grows, minimizing idle compute and communication overhead.
  - Hyperparameter transfer: Small-scale pilot runs can estimate CBS growth to plan large-scale runs (Section 3.3 shows similar trends across 1B and 7B).
  - Methodology beyond LMs: The branched-recovery idea can be adapted to other domains where adaptive optimizers are standard (vision transformers, diffusion models), provided loss-vs-tokens (or steps) comparability is meaningful.

- Open questions
  - Can we predict CBS from model and data properties without any branching?
  - How does CBS interact with other axes: sequence length scaling, curriculum learning, data-mixing strategies, or optimizer hyperparameters like β1/β2 and weight decay?
  - What is the safest learning-rate scaling when also changing other training knobs (e.g., gradient clipping thresholds, normalization layers)?

Overall, this work supplies a concrete, low-assumption toolkit for large-batch LLM training: measure CBS locally with short, branched probes; then warm up the batch size only when it is safe. The empirical results (Figures 1–4; Tables 1–2) show this approach can preserve or slightly improve loss while materially reducing the number of gradient steps.
