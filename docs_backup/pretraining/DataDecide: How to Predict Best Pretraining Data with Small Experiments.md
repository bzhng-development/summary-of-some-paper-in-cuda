# DataDecide: How to Predict Best Pretraining Data with Small Experiments

**ArXiv:** [2504.11393](https://arxiv.org/abs/2504.11393)
**Authors:** Ian Magnusson, Nguyen Tai, Ben Bogin, David Heineman, Jena D. Hwang, Luca Soldaini, Akshita Bhagia, Jiacheng Liu, Dirk Groeneveld, Oyvind Tafjord, Noah A. Smith, Pang Wei Koh, Jesse Dodge
**Institutions:** Allen Institute for AI (Ai2), University of Washington (implicit for certain authors)

## 🎯 Pitch

DATADECIDE offers a game-changing approach for selecting optimal pretraining data for large language models by utilizing small-scale experiments. By accurately predicting large-scale outcomes with a single scale ranking method that achieves about 80% decision accuracy, it drastically reduces computational costs, enabling faster and cheaper model development without compromising performance.

---

## 1. Executive Summary
This paper introduces DATADECIDE, a large, controlled suite of language models that isolates pretraining data choice from other variables so one can test, at small scale, which data will train the best larger model. Using 1,050 models trained across 25 pretraining “data recipes,” 14 model sizes (4M–1B), and up to 100B tokens per target run, the work measures how well cheap small-scale experiments predict which data win at a larger target scale. The headline finding is practical: simply ranking data by performance at one small model size (e.g., 150M parameters) predicts the 1B-scale winner about 80% of the time, and continuous likelihood-based proxy metrics enable >80% predictability for several benchmarks using as little as ~0.01% of the full compute.

## 2. Context and Motivation
- Problem addressed
  - Training large language models on multiple candidate pretraining datasets to choose the best one is prohibitively expensive. The specific question is: how can one use small, cheap experiments to decide which pretraining data (“data recipes”) will yield the best large model? (§1)
- Why it matters
  - Pretraining data is a key driver of downstream performance and behavior. Making correct data choices with minimal compute reduces cost and accelerates model development without sacrificing quality.
- Prior practice and gaps
  - Two dominant strategies exist:
    - Rank at a single small scale: Run small models on candidate data and pick the best small-scale performer for the large-scale training (e.g., used in DCLM). This is inexpensive but assumes rankings don’t change with scale.
    - Fit scaling laws: Train multiple small models to fit curves that extrapolate performance to a target scale (e.g., Kaplan-style power laws; extended to predict downstream accuracy via a loss→accuracy mapping). This is more expensive and has mostly been validated by low prediction error on a few points rather than proven decision accuracy (§1–§2.2).
  - Missing piece: A broad, controlled, open suite that allows direct, counterfactual validation of “small-to-large” decision-making across many data recipes and tasks, while holding other variables constant (§1–§2.1).
- This paper’s positioning
  - DATADECIDE provides the missing test bed: 25 data recipes (Table 1), 14 model sizes (Table 2), 3 seeds each, and 10 downstream tasks (OLMES) to measure how well small-scale choices predict large-scale winners. It compares simple single-scale ranking versus eight scaling-law variants and investigates which proxy metrics make small-scale predictions more reliable (§2.1–§2.5).

## 3. Technical Approach
At a high level, DATADECIDE measures the “decision accuracy” of small-scale methods: given two pretraining data recipes A and B, does the small-scale method correctly predict which one wins at a target scale?

- The DATADECIDE suite (§2.1; Table 1, Table 2)
  - Data recipes: 25 controlled pretraining corpora (“data recipes”) covering popular sources and interventions: Dolma 1.7 (and ablations), Dolma 1.6++, C4, RefinedWeb (Falcon), FineWeb variants, DCLM variants and thresholds, and mixtures between DCLM and Dolma (Table 1).
    - “Data recipe” = a concrete specification of sources and curation interventions (e.g., mixes, deduplication, quality filters, domain ablations). This isolates “the data choice” while holding models and training setup constant.
  - Model sizes and training scale:
    - 14 sizes from 4M to 1B parameters (Table 2).
    - Target scale: 1B parameters, trained on 100B tokens (token-to-parameter ratio 100; “5× Chinchilla” overtraining favored for inference savings).
    - For each recipe × size: 3 random seeds. For 1B models, all seeds are trained to completion; for smaller sizes, seeds 2–3 are trained only to 25% of the 1B compute (to keep the prediction budget realistic) (§2.1).
    - Hyperparameters come from the OLMo “model ladder,” which programmatically sets reasonable configs per scale to avoid confounding due to poor hyperparameters (sequence length 2024, MLP ratio 8, etc.; Table 2; §2.1).
    - Observed run-to-run variability is non-trivial at target scale: “standard deviation between runs at the 1B 5×C scale can be as high as 2 percentage points of accuracy for some recipes on most tasks” (§2.1).

- Prediction methods compared (§2.2)
  - Ranking single scale (“Single Scale”):
    - Train small models on candidate data recipes, evaluate downstream, rank the recipes, and pick the top one. No curve fitting. This approximates performance at the target scale with a constant (the observed small-scale performance).
    - Benefit: cheapest, easy to run at many sizes and even at intermediate checkpoints (Figure 1-right).
    - Limitation: cannot catch “crossovers,” i.e., data whose learning curves cross so winner changes with scale (§3.2).
  - Extrapolating scaling laws (“Multi Scale”):
    - Two-step fit (Equations 1–2):
      - Step 1 fits the relation between compute C and task loss L using a power-law form: L(C) = A / C^α + E (Equation 1; parameters A, α, E). Fit only on final checkpoints (smoothed by averaging the last 10% of checkpoints) to avoid schedule effects (§2.2).
      - Step 2 maps loss to accuracy with a logistic: Acc(L) = a / (1 + exp(−k(L−L0))) + b (Equation 2; parameters a, k, L0, b). Fit on all checkpoints to exploit more data (§2.2).
    - Variants explored (Appendix C):
      - Remove irreducible loss term E (2-parameter version; Equation 4).
      - Model loss using tokens N and parameters D explicitly (5-parameter: L(N,D)=A/N^α + B/D^β + E; Equation 5).
      - Single-step versions that directly map C (or N, D) to accuracy via a logistic over predicted loss (Equations 6–7).
      - “Helper point” trick for step 2 (anchor top asymptote at L=0→Acc=1), and filtering early checkpoints (>50% steps) to reduce noise in the step-2 fit (§2.2; Appendix C).
    - Compute budget for multi-scale: the sum of training cost (FLOPs) of all model sizes used to fit the scaling law; size subsets range from smallest 3 sizes up to including near-target sizes (§3.2).

- Metrics for assessing predictions (§2.3)
  - “Prediction error” (absolute or relative error between predicted and actual accuracy) is tracked but is not the primary decision goal.
  - “Decision accuracy” is the main metric: over all pairs of data recipes (A,B), does the prediction produce the correct sign of the difference at 1B? Formally (Equation 3), it’s the fraction of pairs where sign(ŷA−ŷB)=sign(yA−yB), with y values being mean performance over 3 seeds at 1B (§2.3).
    - Intuition: even if a method produces some error in absolute accuracy, it can still be useful if it reliably ranks data correctly.
  - Compute accounting: percent of target compute `%C` uses FLOPs = 6ND (N = parameters, D = tokens trained) (Kaplan et al. assumption), reporting c/C × 100% (§2.3).

- Downstream evaluation tasks (§2.4)
  - Use the OLMES suite of 10 multiple-choice benchmarks: MMLU, HellaSwag, ARC Challenge, ARC Easy, PIQA, CommonsenseQA, SocialIQA, OpenBookQA, BoolQ, and WinoGrande (§2.4).
  - Primary target metric is “Accuracy” in a cloze-format evaluation with curated normalizations per task (“ACCURACY”), using all available items (no subsampling) to reduce variance (§2.4).

- Proxy metrics for small-scale prediction (§2.5; Table 3)
  - Motivation: discrete accuracy is coarse and can hide smooth improvements in model likelihoods (a discrete “jumpiness” problem).
  - Proxy metrics (character- or token-normalized):
    - `CORRECT PROB`: average probability assigned to the correct answer.
    - `MARGIN`: average gap between the correct answer’s probability and the highest incorrect option.
    - `NORM CORRECT PROB`: correct answer prob normalized by sum over all answer options.
    - `TOTAL PROB`: sum of probabilities over all candidate answers (correct + incorrect).
    - `ACCURACY`: fraction of items where the correct option is the most probable.
  - Unless otherwise noted, character normalization is used because it empirically worked best for most tasks (§2.5; Figure 4).

- How predictions are produced
  - Single-scale: at a chosen small size (or even at an intermediate checkpoint), compute a metric (e.g., ACCURACY or CORRECT PROB) across all data recipes; rank recipes; compare to the 1B “gold” rankings (averaged over 3 seeds) to compute decision accuracy (Figures 1–2).
  - Multi-scale: select a set of small sizes; fit a scaling law (with step-1/step-2 variants); extrapolate to 1B; rank recipes by predicted 1B accuracy; compute decision accuracy (Figure 3; Table 4).

## 4. Key Insights and Innovations
- A comprehensive decision-centric evaluation protocol and open suite
  - What’s new: Instead of only reporting low prediction error on a few large models, DATADECIDE provides a direct, counterfactual evaluation of decision quality across 25 data recipes at a realistic target scale (1B), averaged over 3 seeds (§2.1–§2.3).
  - Why it matters: It tests the actual question practitioners face—“Which data recipe should I choose?”—with enough coverage to observe crossovers, noise, and variance, and to quantify how much compute is needed to make good decisions (Figure 1-right, Figure 2).
- Single small-scale ranking is a surprisingly strong baseline
  - Insight: Ranking data at a single small size yields high decision accuracy at 1B—about 80%—often matching or beating more complex, multi-scale scaling law fits (Figure 1-right; §3.1–§3.2).
  - Significance: This sets a high compute-to-decision “frontier” that scaling law methods must beat to justify their additional compute (Figure 3).
- Continuous likelihood-based proxy metrics unlock early predictability
  - Insight: At small scales, `CORRECT PROB` and `TOTAL PROB` often predict large-scale ACCURACY better than ACCURACY itself (Figure 4). They can make tasks like MMLU, ARC, HellaSwag, MBPP, and HumanEval >80% predictable at the 1B target using ~0.01% of the compute.
  - Significance: This reduces the compute needed for data decisions by orders of magnitude (§3.3; Abstract).
- Why predictability varies across tasks—and how to fix it
  - Insight: Decision accuracy improves when either run-to-run variance (“noise”) is low or the spread of performance across data recipes is wide. `CORRECT PROB` often reduces noise or increases spread, improving predictability (Figure 5; §3.4).
  - Example: With `CORRECT PROB`, code benchmarks (MBPP, HumanEval) become predictably rankable at small scale (>80% decision accuracy), whereas with ACCURACY they are near trivial (Figure 6).

## 5. Experimental Analysis
- Evaluation setup
  - Models and compute
    - 1,050 models = 25 recipes × 14 sizes × 3 seeds (§2.1; Table 2), with 1B targets trained to completion on 100B tokens and smaller sizes partially replicated (seeds 2–3 at 25% compute).
    - Compute reported as `%C` (proportion of 1B training FLOPs), allowing apples-to-apples comparisons of methods (Figures 1–4).
  - Downstream tasks
    - OLMES suite of 10 multiple-choice benchmarks with cloze-style ACCURACY as the target metric (Figure 2; §2.4).
  - Decision accuracy metric
    - Pairwise accuracy over all recipe pairs using 1B, 3-seed means as the “gold” ranking (Equation 3; §2.3).

- Main quantitative findings
  - Compute vs decision accuracy frontier (aggregate across tasks)
    - There is a roughly log-linear relationship: more compute yields better decision accuracy. Crucially, intermediate checkpoints at a given size deliver similar benefit to training fully at that size for the same cumulative compute (Figure 1-right; §3.1).
    - Quote: “Pretrain 25 datasets @ 150M to predict pairs of 25 datasets @ 1B ~80% correct” (Figure 1-right annotation).
  - Per-task predictability varies widely (Figure 2; §3.1)
    - Example patterns:
      - ARC Easy is predictable with “five orders of magnitude less compute” than harder tasks like HellaSwag (Figure 2).
      - BoolQ stays near trivial decision accuracy until near-target compute; HellaSwag, SocialIQA, and WinoGrande show threshold behavior—flat, then rapid gains after a compute threshold (Figure 2).
  - Scaling laws vs single-scale ranking (Figure 3; Table 4; §3.2)
    - None of eight scaling-law variants exceed the compute–decision frontier of single-scale ranking (Figure 3).
    - The best-fit scaling setups (e.g., “3-parameter with helper points and >50% checkpoints”) achieve relative prediction error ~5.6% and absolute error ~2.6 points on ACCURACY (Table 4), but still do not produce better decision accuracy per unit compute than simple single-scale ranking (Figure 3).
    - Quote: “At best, [scaling-law approaches] reach only the same compute to decision accuracy frontier as ranking single scale experiments” (Figure 3 caption).
  - Proxy metrics enable cheaper, earlier decisions (Figure 4; §3.3)
    - `CORRECT PROB` and `TOTAL PROB` (character-normalized) often dominate ACCURACY at low compute, then ACCURACY and metrics that penalize incorrect options (e.g., `MARGIN`, `NORM CORRECT PROB`) catch up near the target compute.
    - Task-specific examples:
      - For multiple tasks, decision accuracy rises earlier with `CORRECT PROB`/`TOTAL PROB` than with ACCURACY (Figure 4). The benefit is especially visible at very low `%C`.
    - Quote: “At small scales, continuous metrics…serve as better or equivalent predictors…than using the same ACCURACY as used at the target scale” (Figure 4 caption; §3.3).
  - Why tasks become more predictable (Figure 5; §3.4)
    - Plotting “noise” (seed std) vs “spread” (std across recipes) at 150M shows that higher decision accuracy associates with lower noise and/or higher spread. `CORRECT PROB` shifts tasks toward this favorable region.
      - Example: HellaSwag achieves predictability via lower run-to-run variance under `CORRECT PROB`; SocialIQA achieves predictability via wider spread across data recipes (Figure 5).
  - Code and math tasks with proxy metrics (Figure 6; §3.4)
    - With ACCURACY, MBPP and HumanEval predictions are near random at small scale; with `CORRECT PROB`, decision accuracy jumps to ~80% even at 4M–60M sizes (Figure 6).
    - Math tasks (Minerva, GSM8K) do not show the same benefit for ACCURACY prediction, though if one targets `CORRECT PROB` as the endpoint metric, decision accuracy exceeds 80% (§3.4).

- Support and robustness
  - Multiple seeds at target scale reduce the chance of chasing noise (1B trained to completion for all three seeds), with explicit reporting of seed variance (up to 2 points ACCURACY; §2.1).
  - Intermediate checkpoints are leveraged to use compute more efficiently without sacrificing predictive quality (Figure 1-right; §3.1).
  - Eight scaling-law variants, helper points, and early-checkpoint filtering are explored to check whether curve-fitting can beat simple ranking (Figure 3; Table 4; Appendix C).

- Where results are conditional
  - Predictability depends on the benchmark and on the metric used at small scale:
    - Cheaper predictability: ARC and MMLU; harder: HellaSwag (with ACCURACY), BoolQ (Figure 2).
    - Changing the small-scale metric (to `CORRECT PROB` or `TOTAL PROB`) can change a task from “unpredictable” to “predictable” at low compute (Figure 4).
  - Simple ranking cannot capture true crossovers by design; scaling laws could in principle, but the tested variants did not improve the compute–decision frontier (Figure 3; §3.2).

## 6. Limitations and Trade-offs
- Assumptions and scope (§5; §2.1)
  - Fixed token-to-parameter ratio (100; “5× Chinchilla” overtraining). This matches contemporary practice but does not explore under/overtraining trade-offs at other ratios.
  - Fixed model ladder: 14 specific configurations from 4M–1B. While extensive, broader architectural and optimizer variations are not explored.
  - Focused evaluation: primarily multiple-choice, cloze-style OLMES tasks. Other task types (e.g., generative, open-ended) may show different predictability patterns (§2.4, §5).
- Decision metric design
  - Single-scale ranking cannot capture true crossovers where a lower-curve data recipe overtakes another at larger scale (§3.2).
  - Scaling-law accuracy depends on fitting choices (e.g., which checkpoints, helper points); even with improved fits, noisy or non-sigmoidal loss→accuracy relationships can degrade extrapolations (Appendix C; Table 4).
- Compute and data constraints
  - Although this suite amortizes a large up-front cost (~820K H100 GPU hours; Impact Statement), extending to even larger target scales or more recipes remains expensive.
  - Multi-scale scaling-law fits require final checkpoints for step-1 loss fitting; only one seed’s final checkpoints are available for all sizes, limiting multi-seed fits at non-target scales (§3.2).
- Open questions
  - Why exactly do some tasks exhibit threshold effects (flat then rising predictability) with ACCURACY? Are there task-specific structural reasons beyond noise/spread (Figure 2)?
  - Can better functional forms or hierarchical fits (e.g., across tasks or recipes) produce scaling-law methods that surpass the single-scale frontier on DATADECIDE?

## 7. Implications and Future Directions
- Practical guidance for data selection
  - If compute is scarce, start with single-scale ranking using continuous likelihood proxies:
    - Use `CORRECT PROB` or `TOTAL PROB` (character-normalized) at very small sizes or even intermediate checkpoints to get early signal (Figure 4).
    - Expect around ~80% correct pairwise decisions at the 1B target when using, e.g., 150M single-scale runs, and often much earlier for certain tasks (Figure 1-right; Figure 2).
  - Prefer benchmarks with either low seed variance or wide performance spread across recipes, or switch to proxy metrics that create these conditions (Figure 5).
- Research directions enabled by DATADECIDE
  - Scaling-law innovation with decision-centric objectives:
    - DATADECIDE provides a compute–decision curve (Pareto frontier) that future scaling laws must beat (Figure 3). New functional forms, Bayesian fits, uncertainty-aware methods, or meta-learning across recipes/tasks could be tested in a standardized setting.
  - Benchmark design:
    - The noise/spread diagnostic (Figure 5) suggests how to construct or choose small-scale evaluations that are maximally predictive. Designing tasks/metrics that reduce seed variance and increase between-recipe separation should improve small-to-large predictability.
  - Metric engineering for cheap decisions:
    - For code and perhaps other domains, using `CORRECT PROB` can turn “impossible-to-predict” accuracy outcomes into robust decisions at tiny compute (Figure 6). This invites metric learning or automatic proxy selection tuned for decision accuracy.
- Downstream applications
  - Industrial data curation: Organizations can quickly down-select among crawls, filters, and source mixes before committing to costly large-scale runs.
  - Academic reproducibility: Released models, data, and evaluations allow new methods to be evaluated without retraining large models (Abstract; §2).

> Central takeaways, anchored in the data:
> - “The ranking of models at a single, small size (e.g., 150M) is a strong baseline… (~80% of comparisons correct)” (Abstract; Figure 1-right).
> - “8 baseline scaling law methods do not exceed the compute–decision frontier of single-scale predictions” (Abstract; Figure 3).
> - “Using continuous likelihood metrics as proxies… makes MMLU, ARC, HellaSwag, MBPP, and HumanEval >80% predictable… with just 0.01% of the compute” (Abstract; Figure 4, Figure 6).

Overall, DATADECIDE reframes small-scale experiments around decision quality, not just predictive error, and shows that with the right metric design, extremely cheap experiments can reliably choose good pretraining data for larger models.
