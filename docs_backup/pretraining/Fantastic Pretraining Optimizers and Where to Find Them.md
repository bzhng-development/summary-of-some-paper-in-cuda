# Fantastic Pretraining Optimizers and Where to Find Them

**ArXiv:** [2509.02046](https://arxiv.org/abs/2509.02046)
**Authors:** Kaiyue Wen, David Hall, Tengyu Ma, Percy Liang
**Institutions:** Stanford University

## 🎯 Pitch

This paper offers a rigorous re-evaluation of 11 optimizers for large language model pretraining, revealing that the claimed speedups over AdamW are often due to unfair baselines and insufficient tuning. By implementing a comprehensive, multi-phase tuning protocol, the study demonstrates that true speedups are modest (≤1.4× for smaller models) and diminish with scale, offering a more accurate and economically significant understanding of optimizer performance for LLM training.

---

## 1. Executive Summary
This paper rigorously re-benchmarks 11 optimizers for large language model (LLM) pretraining and shows that most previously claimed 1.4–2× speedups over AdamW largely disappear under fair hyperparameter tuning and scaled evaluations. Using a three-phase tuning protocol across four model sizes (130M–1.2B parameters) and multiple data-to-model ratios, it finds matrix-preconditioned methods (e.g., Muon, Soap) do offer real gains—but they are modest (≤1.4×) and shrink with model size to ≈1.1× at 1.2B parameters.

## 2. Context and Motivation
- Problem addressed
  - Optimizer papers for LLM pretraining frequently claim large speedups (1.4–2× and beyond), yet production training mostly continues to use AdamW. This work investigates why prior evaluations do not translate into practice and what speedups are actually achievable under fair, scaled comparisons (Introduction; Figure 1).
- Why this matters
  - Pretraining dominates LLM training cost (e.g., >95% in DeepSeek V3), so even a 10–20% true speedup is economically important (Introduction).
  - Methodologically robust optimizer evaluation reduces wasted compute, clarifies which techniques scale, and guides future optimizer design.
- Gaps in prior work
  - Under-tuned baselines: The widely adopted GPT-3 recipe used a low peak learning rate for AdamW; simply retuning that single hyperparameter yields up to a “2× speedup” that some papers attribute to new optimizers (Figure 1, top-left).
  - Narrow evaluation regimes: Many studies test only one small model or only near the “Chinchilla-optimal” data budget, leaving questions about scale-up and overtraining regimes (Introduction).
- Positioning relative to existing work
  - This paper re-centers evaluation on:
    - Equal hyperparameter budgets per optimizer, tuned to (near-)optimum via a multi-phase coordinate descent protocol (Sections 3.2–3.4).
    - End-of-training comparisons across model sizes and across data-to-model ratios, rather than cherry-picked intermediate checkpoints (Sections 3.1, 4.2; Figure 5).
  - It also reconciles differences with concurrent benchmarking (Semenov et al. 2025) by highlighting batch-size regimes and tuning choices that flip which optimizers look best (Related Work: “Comparison with concurrent work”).

Definitions used throughout:
- Data-to-model ratio: total tokens seen per parameter. Higher ratios mean training longer relative to model size.
- Chinchilla ratio: data-to-model ratio normalized by the “Chinchilla-optimal” budget (≈20 tokens per parameter). “8× Chinchilla” means 8 times the optimal data budget (Section 3.1; Hoffmann et al. 2022).
- Matrix-based preconditioner: an optimizer that multiplies gradients by a matrix capturing parameter structure (e.g., per-layer matrices), rather than scaling each coordinate by a scalar, to better condition updates (Sections 2, 4.1).
- Scalar-based optimizer: AdamW, Lion, Mars, etc., which scale each parameter independently using scalar running statistics.

## 3. Technical Approach
This is a large, controlled benchmark built to eliminate tuning and evaluation confounders.

A. Experimental design and setup (Section 3.1)
- Models and architecture
  - LLaMA-2–style transformer with fixed 32 layers and 4096 sequence length; hidden sizes chosen to yield ≈130M, 300M, 520M, and 1.2B parameters (Table 2).
- Data mixture
  - Three public corpora tokenized with the Llama-3 tokenizer: DCLM-baseline (3.8T tokens), StarCoder V2 (0.25T), ProofPile 2 (55B) (Section 3.1).
- Evaluation metrics
  - Primary: validation language-modeling loss on C4/EN (a widely used proxy for downstream performance). Downstream tasks include ARC-E, ARC-C, HellaSwag, LAMBADA, PIQA, WinoGrande, etc. (Section 3.1).
- Optimizers under test (Table 1)
  - Baseline AdamW; scalar variants (NAdamW, Mars, Cautious, Lion, Adam-mini); matrix-based (Muon, Soap, Kron/PSGD, Scion); Hessian-approx (Sophia).
- Why measure tokens rather than wall-clock?
  - Prior work shows well-implemented matrix preconditioners can limit per-step overhead to <10% (Section 3.1). The study therefore uses “tokens needed to reach a loss” as the speed metric, aligning with compute budgets.

B. Three-phase tuning framework (Sections 3.2–3.4)
- Rationale
  - Reported large gains often come from giving the new optimizer more tuning than AdamW or from transferring AdamW hyperparameters blindly to incompatible algorithms. The paper equalizes hyperparameter budgets and isolates which knobs truly matter at scale.

Phase I: Full coordinate descent at small/medium scales (Section 3.2)
- What is coordinate descent here?
  - A simple, reproducible process: hold all hyperparameters fixed, sweep one hyperparameter across a discrete grid, keep the best value if it improves validation loss by at least Δ1 = 3e-3, and iterate through all hyperparameters until no further gains (Section 3.2; Table 3 shows an example).
- Where run?
  - Six regimes: 130M, 300M, 520M at 1× Chinchilla; and 130M at 2×, 4×, 8× Chinchilla (Section 3.2).
- Output of Phase I:
  - A near-optimal configuration per optimizer per regime, and a diagnosis of which hyperparameters are “scaling-sensitive,” i.e., whose best values change with model/data scale (Table 4).

Phase II: Focused tuning on scaling-sensitive hyperparameters (Section 3.3)
- Why?
  - Full sweeps are expensive at larger scales; most hyperparameters are either insensitive or transfer well. The authors retain only scaling-sensitive knobs (Table 4 lists which for each optimizer).
- Additional regimes tuned:
  - 300M and 520M at 2×, 4×, 8× Chinchilla (Section 3.3).
- How is speedup computed?
  - Fit AdamW’s loss-vs-data curve at each model size as L̂_N(D) = α_N D^(−B_N) + β_N. For an optimizer achieving loss L at D tokens, compute the AdamW tokens D_AdamW needed to reach the same loss by inverting L̂_N. The speedup is D_AdamW / D (Section 3.3; Figure 3 visualizes this mapping).

Phase III: Hyperparameter scaling laws and extrapolation (Section 3.4)
- Predicting good hyperparameters at new scales
  - Fit a law h(N, D) = α N^(−A) D^(−B) + β for each scaling-sensitive hyperparameter using the 12 Phase I–II settings. A validation at 1.2B, 1× Chinchilla shows predicted hyperparameters are within 3e-3 loss of the optimum (Section 3.4).
- New experiments enabled
  - 1.2B models at 1×–8× Chinchilla for AdamW, NAdamW, Muon, Soap (Figure 4).
  - High overtraining: 16× Chinchilla at 130M and 300M to see if winners change (Figure 4 right; Appendix B.3).

C. What distinguishes matrix-based optimizers here?
- Example: Muon (Algorithms Appendix A; Algorithm 8)
  - Uses a matrix operation (Newton–Schulz orthogonalization) to precondition and “whiten” updates for matrix-shaped parameters while using AdamW for embeddings and layer norms. Intuition: better conditioning along correlated parameter dimensions can accelerate convergence.
- Soap (Algorithm 11)
  - Maintains factored curvature summaries (Gram matrices) per block, uses QR steps to keep preconditioners well-conditioned; applies Adam-like updates in a whitened space. Intuition: a Shampoo-style preconditioner stabilized with Adam components.
- Kron/PSGD (Algorithm 10)
  - Kronecker-factored preconditioning with a series of blockwise transformations; uses balancing and occasional sketching to update the preconditioner efficiently.

D. Why this approach over alternatives?
- Equalizing tuning eliminates “unfair baseline” effects (Figure 1 top-left shows 2× apparent gains vanish just by retuning AdamW’s learning rate).
- Deciding winners at end-of-training prevents mis-ranking caused by learning-rate decay phase crossings (Figure 5 right and lower-left).
- Covering multiple model sizes and data-to-model ratios reveals shifts in the best optimizer when scaling or overtraining (Figures 2–4).

## 4. Key Insights and Innovations
1) A rigorous, scalable tuning protocol that produces fair, end-of-training comparisons
- Novelty
  - The three-phase process (Sections 3.2–3.4) identifies which hyperparameters truly need per-scale retuning and fits scaling laws to extrapolate to 1.2B. This is a practical recipe for future optimizer work, not just this benchmark.
- Why it matters
  - It demonstrates that unequal tuning was responsible for many prior “2× speedups.” Figure 1 (top-left) shows tuning AdamW’s peak learning rate alone closes that gap.

2) True speedups over a strong AdamW baseline are modest and shrink with scale
- Evidence
  - Speedups cap around 1.4× for 130M models (Figure 3, left) and fall to ≈1.1× for 1.2B models at 8× Chinchilla (Figure 4, middle).
  - Downstream metrics at 1.2B show no clear advantage of Muon or NAdamW over AdamW despite small loss gains (Table 5).

3) Matrix-based preconditioning consistently beats scalar methods at small/medium scales, but the winner depends on data budget
- Evidence
  - At 130M–520M, Muon/Soap/Kron beat scalar variants (AdamW, NAdamW, Mars, Lion) across data budgets (Figure 2 top row; Figure 1 bottom-right).
  - The “best” matrix optimizer flips with data-to-model ratio: Muon tends to win at 1–4× Chinchilla; Soap and Kron take over at 8–16× (Figures 3 and 4 right; Appendix B.3).

4) Early training curves are unreliable for ranking optimizers
- Evidence
  - Curves cross during learning-rate decay; choosing an intermediate checkpoint can reverse the final ranking (Figure 5 right; 5 lower-left).
- Significance
  - Many prior comparisons stopped early or reported “intermediate speedups,” which this study shows can be misleading.

5) Shared training phenomena across optimizers
- Findings
  - Parameter norms rise and then fall with the learning-rate schedule (Figure 6 middle-left).
  - Gradient norms increase during learning-rate decay without causing loss spikes (Figure 6 middle-right).
  - Training-vs-eval loss tradeoffs are similar across optimizers (Figure 6 right).
- Value
  - These observations suggest some general training dynamics are optimizer-agnostic, informing theory and diagnostics.

## 5. Experimental Analysis
A. Evaluation methodology
- Datasets and preprocessing
  - DCLM-baseline, StarCoder V2, ProofPile 2; tokenized with Llama-3 tokenizer; sequence length 4096 (Section 3.1).
- Models
  - LLaMA-2–style at 130M, 300M, 520M, 1.2B parameters (Table 2).
- Optimizers and baselines
  - Eleven total (Table 1), grouped into scalar, matrix-based, and Hessian-approx families; all tuned per the three-phase protocol.
- Metrics
  - Primary: C4/EN validation loss at target budgets; secondary: downstream tasks (ARC, HellaSwag, etc.) (Section 3.1).
- Speedup definition
  - Tokens-equivalent speedup relative to AdamW using a fitted AdamW scaling curve at each model size (Section 3.3; Figure 3).

B. Main quantitative results
- Across 130M–520M, matrix > scalar, but capped gains
  - Figure 2 (top row) shows consistent loss improvements for Muon/Soap/Kron over AdamW/NAdamW/Mars/Lion at 1×–8× Chinchilla.
  - Estimated speedups (Figure 3):
    - 130M: top performers around 1.3–1.4×.
    - 300M: ≈1.2–1.3×.
    - 520M: ≈1.2–1.3×, with speedup growth slowing as model size increases.
- At 1.2B, gains shrink and downstream equalizes
  - Figure 4 (left, middle): All non-AdamW methods still beat AdamW in loss, but speedup drops to ≈1.1×.
  - Table 5 (8× Chinchilla, 193B tokens): average downstream scores are within ~0.5 points among AdamW, NAdamW, and Muon (66.7–67.2 average), confirming little practical difference.
- High data-to-model ratios change the winner
  - At 16× Chinchilla, Soap surpasses Muon at 300M (Figure 4, right) and similarly at 130M (Appendix B.3, Figure 8).
- Sensitivity and ablations
  - Hyperparameter sensitivity can flip rankings:
    - Doubling Soap’s learning rate on 520M/8× makes it lose to Mars (Figure 5 left).
    - Histograms of losses when deviating one hyperparameter show optimizer orderings can reverse (Figure 5 middle).
  - Weight decay in Muon: removing decay can give faster early loss but worse final plateau (Figure 5 right).

C. Additional checks and cases
- Sophia (Hessian-approx) under this setup
  - In this data curriculum and with fully randomized shuffling, Sophia does not outperform a well-tuned AdamW for sub-0.5B models (Appendix B.2; Figure 7). The paper traces earlier reported advantages partly to smaller AdamW learning rates and non-random data loading in prior codebases (Section 4.2).
- Scaling-law extrapolation
  - Appendix B.1 fits joint scaling laws for AdamW and Muon; extrapolation suggests Muon’s advantage may vanish or invert at larger (7B) scales under 1× Chinchilla.

D. Do the experiments support the claims?
- Yes, for the central claims:
  - Properly tuned AdamW closes most of the “2×” gap (Figure 1 top-left).
  - Matrix methods provide consistent but modest improvements at small/medium scale (Figures 2–3).
  - Gains diminish with model size, and early-curve comparisons can mislead (Figures 4–5).
- The study is thorough:
  - Multi-size, multi-budget grid; extensive hyperparameter sweeps; clear, reproducible speedup definition; end-of-training checkpoints.

## 6. Limitations and Trade-offs
- Scale ceiling
  - Experiments top out at 1.2B parameters. Scaling-law fits suggest Muon may stop helping by 7B at 1× Chinchilla (Appendix B.1), but that is an extrapolation, not a measurement (Conclusion).
- Hardware and batch-size regime
  - Runs use 128 TPU v5lite chips and relatively large batches (≥0.4M tokens). In small-batch GPU regimes, variance-reduction methods (e.g., Mars) can look stronger, as noted when contrasting with Semenov et al. (2025) (Related Work: “Comparison with concurrent work”).
- Measure of efficiency
  - Tokens-to-loss abstracts away step-time overheads and communication costs. If a matrix method incurs >10% per-step overhead at a given deployment, its wall-clock speedup may be smaller than its token-efficiency suggests (Section 3.1).
- Coordinate descent tuning
  - Coordinate-wise sweeps can miss hyperparameter interactions (local optima). The paper mitigates this with multiple passes and convergence thresholds, but a global optimizer might find slightly better configurations (Sections 3.2–3.3).
- Dataset mixture specificity
  - Results reflect a specific, modern open-data blend (Section 3.1). Different mixtures or curricula may shift optimal hyperparameters and relative optimizer robustness.

## 7. Implications and Future Directions
- What changes in practice now
  - Expect real, not mythical, gains: plan for ≈1.2–1.4× speedups over a well-tuned AdamW at 0.1–0.5B scales, but only ≈1.1× at ~1B scales (Figures 3–4).
  - Choose by regime:
    - Low-to-moderate data budgets (1–4× Chinchilla) at small/medium scales: `Muon` is often the best matrix option (Figures 2–3).
    - High overtraining (8–16× Chinchilla): `Soap` or `Kron` tend to overtake `Muon` (Figure 4 right; Appendix B.3).
    - If engineering complexity is a constraint and model is ≥1B, a tuned `NAdamW` already captures most attainable gains (Figure 4).
  - Always allocate real tuning budget per optimizer. Do not transfer AdamW hyperparameters blindly (Figure 1 top-right; Figure 5).
  - Compare at end-of-training, not mid-run (Figure 5).
- Research directions enabled
  - Scale-stable preconditioners: Why do matrix-based gains decay with model size? Can preconditioners be designed to retain advantages at ≥1B or 7B+ scales (Appendix B.1)?
  - Batch-size–aware analysis: Formalize how gradient noise and batch size mediate variance-reduction vs matrix preconditioning benefits (Related Work discussion).
  - Learning-rate schedules and generalization: The shared phenomena (Figure 6) invite theory linking learning-rate decay, parameter/gradient norms, and generalization across optimizers.
  - Better automated tuning: Move beyond coordinate descent to joint, compute-efficient hyperparameter optimization transferable across scales (Sections 3.2–3.4).
- Downstream applications
  - Pretraining infrastructure: Production teams can use the provided scaling laws and open-source runs to set initial hyperparameters, then refine per hardware and data (Section 3.4; code and W&B links).
  - Cost forecasting: The token-equivalent speedup framework (Section 3.3) offers a transparent way to convert optimizer choices into compute budgets before committing large runs.

> Key quantitative takeaways
> - “Tuning the AdamW learning rate in the GPT‑3 recipe can yield up to a 2× speedup” (Figure 1, top-left).
> - “Matrix-based optimizers achieve a consistent speedup over scalar-based optimizers” at small/medium scales (Figure 1, bottom-right; Figure 2).
> - “Speedup decays with model size… to only 1.1× for 1.2B” (Figure 1, bottom-left; Figure 4, middle).
> - “Early-stage loss curves can be misleading” because rankings flip during decay (Figure 5).

Overall, this paper replaces mixed anecdotal reports with a careful, end-to-end picture of optimizer behavior across scales and data budgets, providing both a methodological template and concrete, actionable guidance for LLM pretraining.
