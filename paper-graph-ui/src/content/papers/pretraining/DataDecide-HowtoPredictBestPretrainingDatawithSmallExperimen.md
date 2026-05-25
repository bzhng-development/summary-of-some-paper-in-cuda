# DataDecide: How to Predict Best Pretraining Data with Small Experiments

**ArXiv:** [2504.11393](https://arxiv.org/abs/2504.11393)

## 🎯 Pitch

DATADECIDE introduces the most extensive open suite of controlled pretraining experiments—spanning 1,050 model runs across 25 data recipes and 14 model sizes—to rigorously evaluate how well small-scale experiments predict the optimal pretraining data for large language models. The study finds that simply ranking data at a single small model size (like 150M parameters) predicts the best large-scale data choice with about 80% accuracy, outperforming eight more complex scaling-law extrapolation methods. This work empowers practitioners to dramatically reduce computational costs and risk by validating data decisions with small experiments, while providing a powerful public testbed for future research on scaling laws and data selection.

---

## 1. Executive Summary
This paper introduces DATADECIDE, an open suite of 1,050 controlled pretraining runs that makes it possible to rigorously test how well small, cheap experiments can predict the best pretraining data for larger language models. Using these runs, it shows that simply ranking data recipes at a single small scale (around a 150M-parameter model) predicts the best data at a 1B-parameter target scale with about 80% accuracy across pairwise comparisons, and that eight tested scaling-law methods do not beat this compute-to-decision frontier (Figures 1 and 3; Abstract).

## 2. Context and Motivation
- Problem addressed
  - Choosing pretraining data (“what to train on”) is one of the most consequential decisions in building language models, but validating those choices at large scale is prohibitively expensive.
  - Practitioners often rely on small-scale experiments or scaling-law extrapolations to pick data, yet these methods are rarely validated against many counterfactuals at a larger target scale.

- Why it matters
  - Real-world impact: Reducing trial-and-error at large scale saves substantial compute and cost. The suite itself represents “approximately 820K H100 GPU hours” and is released to avoid others repeating that expense (Impact Statement).
  - Scientific impact: It enables measuring not just prediction error but decision quality (who actually wins when two data choices are compared) over many datasets, scales, and seeds.

- Prior approaches and their shortcomings
  - Single-scale selection: Rank small models trained on alternative data and assume the winner scales (e.g., DataComp-LM). This is simple but unvalidated at breadth and depth.
  - Scaling laws: Fit functions that map scale to loss and then to downstream performance (e.g., two-step methods like loss-to-accuracy), but these are usually validated on a few large models and evaluate prediction error, not decision accuracy (Sections 1, 2.2).
  - Existing suites vary data along with modeling/optimizer choices, making data-specific scaling hard to isolate; they also cover few data recipes or scales (Related Work).

- Positioning
  - DATADECIDE is the “most extensive open suite” focused on data differences across scales (Abstract), allowing:
    - Controlled comparisons across 25 pretraining “data recipes” (Table 1).
    - 14 model sizes from 4M to 1B parameters with a fixed token-to-parameter ratio of 100 (5× “Chinchilla” overtraining; Section 2.1).
    - Three random seeds, fully trained at the 1B target scale (Section 2.1).
  - It introduces a decision-focused evaluation (pairwise decision accuracy; Equation 3) to reveal when small-scale predictions actually lead to the right large-scale choice.

## 3. Technical Approach
- What is DATADECIDE?
  - A controlled suite of 1,050 pretraining runs: 25 data recipes × 14 model sizes × 3 seeds (Section 2.1).
  - Data recipes cover major open corpora (e.g., Dolma, DCLM, RefinedWeb, C4, FineWeb) plus interventions like domain ablations, deduplication, quality filtering, and source mixing (Table 1).
  - All models use OLMo’s “model ladder” to standardize architectures and hyperparameters across sizes (Table 2; Section 2.1), targeting 100 tokens per parameter (overtrained regime).

- Experimental pipeline (end to end)
  1) Pretraining
     - For each data recipe and model size, train to a fixed compute budget (FLOPs) with a 100 tokens/parameter ratio (Section 2.1).
     - For 1B models, train three seeds fully; at smaller scales, only one seed is fully trained and the additional two seeds stop at 25% of the target compute to control overall cost (Section 2.1).
     - Rationale: Ensure fair data comparisons and measure run-to-run variation; at 1B, “standard deviation between runs … can be as high as 2 percentage points of accuracy on some tasks” (Section 2.1).

  2) Evaluation
     - Evaluate models on 10 multiple-choice tasks from OLMES (Open Language Model Evaluation Suite): MMLU, ARC Challenge/Easy, HellaSwag, PIQA, CommonsenseQA, SocialIQA, OpenBookQA, BoolQ, WinoGrande (Section 2.4).
     - Default target metric: `ACCURACY` in a cloze-style prompt with task-specific normalization (Section 2.4).

  3) Prediction methods
     - Single-scale ranking (“Single Scale”): Train small models on each recipe; rank recipes by evaluation score; project this ranking to the 1B target (Section 2.2).
     - Multi-scale scaling laws (“Multi Scale”): Fit a two-step function that maps compute to task loss and loss to accuracy using multiple small scales; extrapolate to the target scale (Equations 1–2, Section 2.2). Variants include:
       - 2-, 3-, and 5-parameter loss fits (Equations 1, 4, 5).
       - Single-step fits from compute to accuracy (Equations 6–7).
       - “Helper points” (add an anchor at perfect accuracy) and checkpoint filtering (>50% training; Appendix C).
     - Compute budget is measured as a fraction of target compute, `%C = c/C × 100%`, where FLOPs are approximated as `6 N D` (parameters × tokens; Section 2.3).

  4) Decision quality metric
     - `Decision accuracy` (Equation 3): Across all pairs of data recipes (A, B), the proportion for which the predicted winner matches the observed winner at the target scale. Winners at the target are defined by mean performance over three seeds (Section 2.3).
     - Why this matters: It measures whether a method leads to the right choice, not just low numerical error.

  5) Proxy metrics for small-scale evaluations
     - Besides `ACCURACY`, the study computes continuous, likelihood-based proxies on multiple-choice tasks (Section 2.5; Table 3):
       - `CORRECT_PROB`: average probability assigned to the correct answer.
       - `MARGIN`: probability gap between the correct answer and the most likely incorrect answer.
       - `NORM_CORRECT_PROB`: correct answer probability normalized over the answer set.
       - `TOTAL_PROB`: sum of probabilities of all answers (correct and incorrect), averaged across items.
     - Each proxy is computed with per-character or per-token length normalization; per-character is default in results (Section 2.5).

- Design choices and rationale
  - Single token-to-parameter ratio (100, i.e., 5× overtraining) mirrors common modern practice of overtraining for inference savings (Section 2.1).
  - Focus on multiple-choice tasks to suit the 4M–1B model range (Section 2.4).
  - Two-stage scaling-law fit reflects common practice (loss scaling + link from loss to accuracy; Section 2.2).
  - Decision accuracy centers the practical question: “Will I pick the right data?” rather than “How close is my predicted accuracy?” (Section 2.3).

## 4. Key Insights and Innovations
- A decision-first metric and compute-to-decision frontier
  - Novelty: Evaluating prediction methods by pairwise `decision accuracy` (Equation 3) across many data recipes and scales, rather than only prediction error.
  - Why it matters: It directly answers the practitioner’s question—“Which data should I pick?”—and exposes which methods use compute most effectively (Figure 1, right).

- Single small models are a strong predictor of large-scale winners
  - Finding: “The ranking of models at a single, small size (e.g., 150M parameters) is a strong baseline for predicting best models at our larger target scale (1B) (~80% of comparisons correct)” (Abstract; Figure 1).
  - Significance: A simple approach competes with or exceeds more complex multi-scale fits, up to the measured frontier.

- Scaling-law methods (as instantiated here) do not beat single-scale ranking on compute efficiency
  - Finding: “8 baseline scaling law methods do not exceed the compute-decision frontier of single-scale predictions” (Abstract; Figure 3).
  - Significance: Despite being mathematically sophisticated, these variants do not deliver better decisions per unit of compute in this setting, setting a clear benchmark for future methods.

- Likelihood-based proxies substantially improve small-scale predictability
  - Finding: Using continuous metrics like `CORRECT_PROB` or `TOTAL_PROB` often yields higher decision accuracy at small compute than `ACCURACY` (Figure 4). The abstract highlights: “benchmarks including MMLU, ARC, HellaSwag, MBPP, and HumanEval > 80% predictable at the target 1B scale with just 0.01% of the compute.”
  - Mechanism: Continuous probabilities smooth out discrete jumps in accuracy and better separate data recipes before accuracy becomes informative (Sections 2.5, 3.3).

- Why some benchmarks are predictable and others aren’t (noise vs. spread)
  - Insight: Decision accuracy improves when (a) run-to-run noise is low and/or (b) different data recipes have a wide spread in scores. Figure 5 shows tasks plotted by `noise` (seed-to-seed SD) and `spread` (variance across recipes) at 150M; high accuracy aligns with low noise (e.g., MMLU) or large spread (e.g., ARC Easy).
  - Practical effect: Switching to `CORRECT_PROB` often reduces noise or increases spread enough to enable better decisions (Figure 5).

- New capability: Making code-task decisions at tiny compute
  - Finding: On MBPP and HumanEval, decision accuracy jumps from near-random with `ACCURACY` to roughly 80% with `CORRECT_PROB`, even at very small model sizes (Figure 6).
  - Significance: Enables early, low-cost decisions for domains where discrete accuracy is too sparse at small scales.

## 5. Experimental Analysis
- Evaluation setup
  - Data recipes (Table 1): 25 configurations spanning original corpora (e.g., Dolma 1.7/1.6++, C4, FineWeb Pro/Edu, RefinedWeb), filtered variants (e.g., top-10% or 20% by a quality classifier; multiple classifiers), domain ablations (e.g., remove code or math), and mixtures (e.g., λ% DCLM + (1−λ)% Dolma).
  - Model ladder (Table 2): 14 sizes from 4M to 1B parameters; sequence length 2024; 100 tokens/parameter; standardized batch sizes and learning rates; three seeds (full for 1B; partial for non-target sizes; Section 2.1).
  - Benchmarks (Section 2.4): 10 OLMES tasks; macro-average often used; accuracy is cloze-form with normalization.
  - Compute measure (Section 2.3): `%C` = computed FLOPs divided by target FLOPs, using the standard approximation FLOPs = `6 N D`.

- Main quantitative results
  - Overall compute-to-decision curve:
    - Figure 1 (right) shows a roughly log-linear rise of decision accuracy with compute across all tasks; single-scale ranking at 150M achieves ≈0.8 decision accuracy against 1B targets.
    - Quote: “Targets pretrain 25 datasets @ 150M to predict pairs of 25 datasets @ 1B ~80% correct” (Figure 1 caption).
  - Per-task predictability varies widely (Figure 2):
    - ARC Easy and ARC Challenge are predictable with orders of magnitude less compute than HellaSwag.
    - BoolQ barely exceeds trivial decision accuracy until near-target compute; HellaSwag and SocialIQA show long flat regions before improving.
  - Scaling-law variants vs single-scale (Figure 3; Table 4):
    - No multi-scale variant moves the frontier upward; the best clusters overlap with single-scale results.
    - Average prediction errors (fit to all non-1B models, tested on 1B) are comparable across 2- and 3-parameter two-step variants (e.g., 3-parameter: 6.5% relative, 3.1 absolute), while single-step and 5-parameter variants fare notably worse (e.g., 5-parameter single-step: 42.8% relative; Table 4; Appendix C).
  - Proxy metrics at small scale (Figure 4):
    - `CORRECT_PROB` and `TOTAL_PROB` (both per-character) generally dominate or tie at low `%C`, especially for ARC, PIQA, CommonsenseQA, OpenBookQA, WinoGrande, and MMLU.
    - Toward high `%C`, `ACCURACY`, `MARGIN`, and `NORM_CORRECT_PROB` often catch up or surpass likelihood-only metrics, and sometimes `CORRECT_PROB` declines slightly near the target.
  - Why proxies help (Figure 5):
    - At 150M, tasks that benefit from `CORRECT_PROB` show either reduced seed noise or larger between-recipe spread, improving separability.
  - Code and math tasks (Figure 6):
    - MBPP and HumanEval: `CORRECT_PROB` lifts decision accuracy to ≈80% even at 4M–60M; `ACCURACY` remains near chance at these scales.
    - Minerva and GSM8K: remain near trivial decision accuracy even with `CORRECT_PROB`, unless the target metric is also changed to a continuous proxy (Section 3.4).

- Do the experiments support the claims?
  - Breadth: 25 data recipes and 14 model sizes create a dense grid for testing decision methods, with fully trained 1B models across seeds for robust targets (Section 2.1).
  - Robustness checks:
    - Seeds: 1B targets average over three seeds; small-scale prediction points show mean and standard deviation across three seeds when available (Figures 1–2).
    - Intermediate checkpoints: Using them is as good as compute-equivalent final checkpoints for decision-making (Section 3.1; visible in the dense point clouds in Figures 1–2).
    - Multiple scaling-law variants and fitting choices tested (Figure 3; Appendix C; Table 4).
  - Limitations acknowledged: Only one token-to-parameter ratio; multiple-choice tasks; one family of scaling-law forms (Sections 5, 6 below).

- Ablations/failure cases
  - Long insensitivity regions (e.g., HellaSwag in Figure 2; SocialIQA, WinoGrande): small models provide little decision signal until a threshold compute is reached.
  - Scaling-law crossovers: Single-scale ranking cannot capture crossover where one recipe overtakes another between the small and target scales; while the study observes many empirical crossovers, they are hard to separate from evaluation variance (Section 3.2).
  - Some proxy metrics degrade near target compute: `CORRECT_PROB` and `TOTAL_PROB` sometimes dip as `ACCURACY`-aligned metrics take over (Figure 4).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Single training regime: Fixed token-to-parameter ratio of 100 (5× Chinchilla) throughout (Section 2.1). Findings may differ under data-constrained or undertrained regimes.
  - Task type: Focused on multiple-choice cloze evaluations; generative or interactive tasks may behave differently (Section 2.4).
  - Scaling-law family: Two-step logistic mapping with specific loss forms; other functional families or priors might surpass the baseline variants used (Section 2.2; Appendix C).

- Computational constraints
  - Despite sharing all artifacts, producing the suite required “~820K H100 GPU hours” (Impact Statement). Extending to more sizes or recipes increases cost.

- Variance and crossover ambiguity
  - Even at 1B, run-to-run SD can reach ≈2 accuracy points (Section 2.1), making it hard to label tight matchups.
  - Crossovers (true or noise-induced) limit the ceiling of single-scale ranking (Section 3.2).

- Generalizability
  - The 25 recipes are diverse but finite; future data pipelines might shift patterns of predictability (Section 5).
  - The best small-scale proxy varies by task and compute region (Figure 4), so guidance is conditional.

## 7. Implications and Future Directions
- Practical guidance for data selection with tiny budgets
  - Start simple: Rank candidate data recipes at one small scale; a 150M model’s ranking predicts 1B winners with ≈80% pairwise accuracy (Figure 1).
  - Use continuous proxies early: Prefer per-character `CORRECT_PROB` or `TOTAL_PROB` to extract signal at very low compute (Figure 4).
  - Expect task-specific compute thresholds: Benchmarks like ARC Easy/Challenge are cheap to predict; HellaSwag and SocialIQA typically require more compute (Figure 2).
  - Leverage intermediate checkpoints: They provide decision signal comparable to compute-equivalent final checkpoints (Section 3.1).

- New evaluation lens for scaling research
  - Decision accuracy vs. prediction error: DATADECIDE lets scaling methods be judged by whether they lead to correct choices, not just low numeric error (Equation 3; Figures 1 and 3).
  - A benchmark for better scaling laws: Current multi-scale fits meet but don’t beat the single-scale frontier (Figure 3). Future methods can aim to push this frontier upward.

- Research avenues enabled by the suite
  - Improved loss-to-accuracy links: Explore alternative mappings, uncertainty-aware fits, Bayesian models over crossovers, or transfer from related tasks (Appendix C; Related Work).
  - Task design: Construct or select benchmarks with low seed noise and/or higher recipe spread to improve decision reliability (Figure 5).
  - Metric engineering: For domains where small models are too weak for discrete accuracy (e.g., code), use likelihood-based proxies—even as the target metric remains accuracy (Figure 6).
  - Data-recipe innovation: Rapidly iterate filtering, mixing, and deduplication policies by testing decisions at small scale against the suite’s 1B targets.

- Broader applications
  - Any team curating pretraining data can adopt the recipe: small model + continuous proxies + per-task compute budgeting, then validate decisions with the released checkpoints.
  - The artifacts (models, corpora, and evaluations) are open, so community members can add new evaluations, try new decision methods (e.g., smoothing or curve fitting on released results), or pretrain additional small models without redoing large runs (Abstract; Section 3).

> Key takeaways to operationalize:
> - “The ranking of models at a single, small size … is a strong baseline … (~80% of comparisons correct)” (Abstract; Figure 1).
> - “8 baseline scaling law methods do not exceed the compute-decision frontier of single-scale predictions” (Abstract; Figure 3).
> - “Using continuous likelihood metrics … makes … MMLU, ARC, HellaSwag, MBPP, and HumanEval >80% predictable … with just 0.01% of the compute” (Abstract; Figures 4 and 6).

Overall, DATADECIDE reframes scaling prediction as a decision problem, provides the open infrastructure to measure progress, and offers concrete, low-cost tactics—especially the use of continuous likelihood proxies—to make better data choices for large language models.
