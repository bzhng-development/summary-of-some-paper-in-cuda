# The Art of Scaling Reinforcement Learning Compute for LLMs

**ArXiv:** [2510.13786](https://arxiv.org/abs/2510.13786)

## 🎯 Pitch

This paper pioneers a principled, predictive framework for scaling reinforcement learning (RL) compute in large language models (LLMs), demonstrating that RL performance grows along a stable sigmoidal trajectory as compute increases. By validating this model with over 400,000 GPU-hours and introducing the practical ScaleRL recipe, the work empowers practitioners to forecast RL outcomes from smaller runs, unlocking reliable, efficient, and democratized progress in post-training LLM development—an essential advancement as RL fine-tuning becomes a massive, compute-intensive stage in state-of-the-art LLM pipelines.

---

## 1. Executive Summary
This paper introduces a predictive framework for scaling reinforcement learning (RL) compute for large language models (LLMs) and distills a practical training recipe, called `ScaleRL`, that follows those predictions. It shows that RL performance versus compute follows a stable sigmoidal curve (Equation (1)), enabling accurate extrapolation from small runs to very large ones, and validates this with >400,000 GPU-hours of experiments, including a single run scaled to 100,000 GPU-hours (Figure 1).

## 2. Context and Motivation
- Problem addressed
  - RL has become a key stage for unlocking reasoning and agentic abilities in LLMs, but there is no principled, predictive way to scale RL compute the way pre-training now enjoys through scaling laws (Section 1).
  - Practitioners face many choices—loss functions, off-policy setups, normalization, length control—but lack a way to predict which choices will still work as compute increases (Section 1).
- Why this matters
  - Real systems already allocate massive budgets to RL (e.g., 100k H800 GPU-hours in DeepSeek-R1-Zero; Section 1). Misallocating compute or choosing non-scalable recipes wastes resources and hampers reproducibility.
  - A predictive scaling methodology would let both industry and academia evaluate candidates at small scale and forecast their performance at large compute, democratizing progress (Section 1).
- Prior approaches and limitations
  - Pre-training has converged on predictable power-law scaling (Kaplan et al., 2020; Hoffmann et al., 2022), but RL lacks analogous, validated laws (Section 1).
  - Reports like GRPO, DAPO, Magistral, and others document recipe details but not compute–performance predictability or how choices affect asymptotic performance versus efficiency (Sections 1, 6 and Appendix A.1/A.16).
- Positioning
  - The work proposes a concrete compute–performance model for RL (a sigmoid), stress-tests many recipe choices under a unified setup, and shows which choices raise the ceiling (“asymptote”) and which mainly change efficiency. It then composes those choices into `ScaleRL` and validates its predictability to 100k GPU-hours (Figures 1–2, Sections 2–5).

## 3. Technical Approach
This section explains both the modeling framework (the “scaling law”) and the RL training recipe that is ultimately recommended.

- Compute–performance model (Section 2.1; Equation (1))
  - What is fit: mean pass rate on a held-out validation set versus compute (GPU-hours) in log scale.
  - Sigmoid form:
    - `RC` is validation pass rate after compute `C`.
    - `R0` is the initial pass rate.
    - `A` is the asymptotic pass rate (“ceiling” achievable at large compute).
    - `B > 0` controls the steepness/efficiency of improvement (bigger is more efficient).
    - `Cmid` is the compute where half of the total gain is reached (shifts the curve left/right).
  - Intuition: early slow growth, a mid-range with fast improvement, then saturation at a ceiling; the parameters separate “how high you can ultimately get” (`A`) from “how fast you get there” (`B`, `Cmid`) (Figure 3).
  - Why a sigmoid (Appendix A.4): empirically more robust than power-law fits for bounded metrics like accuracy; power laws over-predict in early-to-mid regimes and are very sensitive to fit range.
  - Fitting procedure (Appendix A.5/A.7): fit after the first ~1.5k GPU-hours to avoid early transient regimes; grid-search over `A` and `Cmid` and fit `B`; error margin on `A` is about ±0.02 based on three independent runs (Figure 8a).

- Experimental regimen and system (Section 2; Appendix A.3)
  - Domain: RL for verifiable reasoning (math primarily; later math+code).
  - Prompt format: model produces a hidden “thinking” trace `<think>…</think>` and a final answer.
  - Generation budget (default): 16,384 total tokens (12,288 think + 2,048 answer + 2,048 prompt). A longer 32,768-budget setting is also studied (Section 5).
  - Data: Polaris-53K math RL dataset (An et al., 2025) with a 1,000-prompt held-out validation set; for multi-task, adds DeepCoder for code (Section 2, Appendix A.3).
  - Batch: 48 prompts × 16 generations per prompt = 768 completions per update (Section 2).
  - Reward: +1/-1 pass/fail using automated checkers; pass rate is measured as mean@16 generations on the 1,000 held-out prompts (Section 2.1).
  - Generator–trainer split: a subset of GPUs run fast generation, the rest run training and periodically synchronize weights (Section 2; Figure 4 and Appendix A.11).

- Base RL objectives and critical definitions (Sections 2–3)
  - `Importance sampling (IS) ratio` is the ratio between the new policy probability and the old policy probability for a token or sequence; used to correct for off-policy learning (Equation (2)).
  - `Clipping` limits IS ratios to stabilize updates (prevents very large policy changes). DAPO uses asymmetric upper/lower clipping (Equation (2) and Appendix A.2).
  - `Advantages` quantify how good a rollout is relative to others for the same prompt; normalized either per-prompt or per-batch (Section 2).
  - Several loss families are compared:
    - GRPO-like with DAPO’s asymmetric clipping (token-level IS) (Equations (2)–(3), Appendix A.2).
    - `GSPO`: sequence-level IS ratios (Section 3.2).
    - `CISPO`: REINFORCE with truncated IS factors applied via stop-gradient (“truncated importance-sampling REINFORCE”) (Equation (4), Section 3.2).

- Off-policy training architecture (Section 3.1; Figure 4)
  - `PPO-off-policy-k`: alternate generation and training phases; each “batch” generates rollouts with the old policy then performs `k` gradient updates on mini-batches (Section 3.1).
  - `PipelineRL-k`: a streaming, asynchronous pipeline. Generators keep generating; each time training completes, the updated weights are pushed to generators immediately—even as they continue generation using stale key-value caches. Trainers wait if they get `k` steps ahead (Section 3.1; Appendix A.11).
  - Empirical finding: PipelineRL reaches similar or slightly higher `A` but with much higher `B` (better efficiency), as shown in Figure 4a; best `k` found to be ~8 (Figure 4b).

- Length control (Sections 2, A.10, A.15)
  - `Forced interruptions`: insert a “time is up” end-of-thinking phrase to stop overly long reasoning and prompt the final answer (prevents runaway lengths and instability).
  - `Length penalty`: subtract a penalty for overly long correct solutions within a tolerance window (Equation (9)).
  - In the final recipe, interruptions are preferred; length penalty does not outperform it in the combined setting (Appendix A.10).

- The `ScaleRL` recipe (Section 4)
  - Components chosen after ablations:
    - Architecture: `PipelineRL-8` (Section 3.1).
    - Loss: `CISPO` (Equation (4)) with prompt-level loss aggregation and batch-level advantage normalization (Section 4).
    - Numeric stability: FP32 precision for the language-model head (logits) on both generator and trainer (Section 3.2; Figure 5b).
    - Length control: forced interruptions (Section 2; Appendix A.10).
    - Batch hygiene: drop prompts with zero reward variance (“zero-variance filtering”) because they give zero gradient; do not resample them within the same step (Figure 6a).
    - Curriculum: `No-Positive-Resampling`—permanently stop sampling prompts that are ≥0.9 pass rate historically (Figure 6b).
  - The combined loss is summarized in Section 4 (under `JScaleRL(θ)`): truncated IS coefficients via stop-gradient, prompt-level averaging, batch-level normalization, zero-variance filtering, and no-positive resampling.

- How the ablation logic was run (Sections 3–4)
  - Stage 1: explore many choices at 3.5–4k GPU-hours; only stable variants are extended (Section 3).
  - Stage 2: combine best choices, then run leave-one-out (LOO) ablations for 16k GPU-hours, fitting on the first 8k and extrapolating to 16k (Section 4; Figure 7).
  - Stage 3: scale on multiple axes—batch size, generation length, MoE model scale, math+code multi-task—and test predictability by fitting on half the target compute and extrapolating to the end (Section 5; Figures 1, 9–11; Table 1 in Appendix A.13).

## 4. Key Insights and Innovations
- A predictive, separable model of RL scaling
  - Novelty: models bounded accuracy vs. compute with a sigmoidal curve that cleanly separates asymptotic performance `A` (ceiling) from efficiency (`B`, `Cmid`) (Section 2.1; Figure 3; Equation (1)).
  - Significance: enables reliable extrapolation from small runs to large budgets; for the 8B model, fitting up to 50k GPU-hours accurately extrapolates the 100k trajectory (Figure 1a). For the 17B×16 MoE, fitting up to 16k extrapolates to 45k (Figure 1a).
- What actually raises the ceiling versus what “just” improves efficiency
  - Finding: some design choices change `A` (e.g., loss family and FP32 logits), while others primarily change `B`/`Cmid` (e.g., normalization, aggregation, curriculum) (Sections 3.2, 4; Figures 5–7).
  - Example: FP32 logits lift `A` from 0.52 to 0.61 (Figure 5b). CISPO/GSPO lift `A` relative to DAPO (Figure 5a). LOO studies show many other choices have similar `A` but different `B` (Figure 7).
- PipelineRL as the more scalable off-policy mechanism
  - Difference from prior work: instead of batch-alternating “generate then train” (PPO-off-policy), PipelineRL streams updates to generators immediately, keeping training closer to on-policy (Section 3.1; Appendix A.11).
  - Impact: much better `B` (efficiency) and slightly higher `A` than PPO-off-policy under matched settings (Figure 4a).
- A robust, practical recipe (`ScaleRL`) that remains predictable at scale
  - Composition over invention: `ScaleRL` integrates existing techniques—CISPO, FP32 logits, PipelineRL-8, prompt-averaging, batch normalization, zero-variance filtering, no-positive resampling, interruptions—and validates each component via leave-one-out ablations (Section 4; Figure 7).
  - Predictability: extrapolations from the first half of training consistently match extended runs, including the 100k GPU-hour run (Figures 1, 7–11).
- Hyperparameter robustness of CISPO vs. sensitivity of DAPO
  - Evidence: changing DAPO’s upper clipping (`ϵmax`) shifts the asymptote `A` materially (Appendix A.17.1; Figure 19a), whereas CISPO’s clipping range changes have little effect (Appendix A.17.2; Figure 19b). GSPO is robust to scale once the correct order of magnitude is chosen, but showed mid-training instability in some runs (Appendix A.17.3–A.17.4).

## 5. Experimental Analysis
- Evaluation methodology
  - In-distribution validation: 1,000 held-out Polaris math prompts; pass rate (average over 16 generations per prompt) computed every 100 steps (Section 2.1).
  - “Compute” is GPU-hours; fits exclude the first ~1.5k GPU-hours (Section 2.1; Appendix A.5–A.7).
  - Downstream generalization: AIME-24 (math), LiveCodeBench Jan–Jun 2025 (code) (Figures 1b, 9b, 10b, 18).
  - Stability diagnostics: “truncation rate” (percentage of generations forcibly interrupted), which correlates with instabilities (Appendix A.15).
- Core ablations and comparisons
  - Off-policy algorithm (Section 3.1; Figure 4a–b)
    - PipelineRL-k vs. PPO-off-policy-k: similar `A` (~0.52 in that setup), but PipelineRL’s `B` is substantially larger. Best `k ≈ 8` for PipelineRL (Figure 4b).
  - Loss family (Section 3.2; Figure 5a; Appendix A.17)
    - CISPO and GSPO both raise `A` substantially over DAPO; e.g., in Figure 5a, DAPO fits to `A ≈ 0.52`, while CISPO/GSPO fit to `A ≈ 0.595`.
    - CISPO chosen for better late-training trajectory and robustness to clipping.
  - FP32 logits (Section 3.2; Figure 5b)
    - > “Using FP32 precision in the final layer (LM head) gives a considerable boost in the asymptotic reward,” lifting the fitted asymptote from ≈0.52 to ≈0.61 (Figure 5b).
  - Loss aggregation and advantage normalization (Section 3.2; Appendix A.9)
    - Prompt-level averaging achieves the best or tied best asymptote among aggregation schemes (Appendix Figure 14a).
    - Batch-level, prompt-level, or no normalization behave similarly on asymptote; batch-level chosen for theoretical soundness and slight edge (Appendix Figure 14b).
  - Zero-variance filtering and No-Positive-Resampling (Section 3.2; Figure 6)
    - Dropping zero-variance prompts improves asymptote relative to counting them toward the batch (Figure 6a).
    - Filtering out ≥0.9-pass prompts across epochs improves scalability and asymptote (Figure 6b).
- The `ScaleRL` LOO study (Section 4; Figure 7)
  - Each component is removed one at a time; most LOO runs achieve similar `A` (within ±0.02), but differ in `B`. Re-plotting in a form where slope equals `B` makes efficiency differences explicit; `ScaleRL` has the highest `B` (Figure 7).
  - Variability analysis across three independent runs yields ±0.02 error on `A` (Figure 8a), which is used to judge meaningful differences.
- Scaling experiments and predictability (Section 5; Figures 1, 9–11; Table 1 in Appendix A.13)
  - Model size (MoE): `Llama-4 17B×16` (“Scout”) trained with `ScaleRL` follows a predictable curve and achieves much higher asymptote (`A ≈ 0.71`) than the 8B dense model (`A ≈ 0.61`), while using about 1/6 of the 8B run’s RL compute to surpass its final level (Figure 1a; Table 1).
  - Generation length: 32k-token runs have lower efficiency (`B` decreases; `Cmid` increases) but a higher asymptote (`A` increases to ≈0.645), overtaking 14k runs at large compute (Figure 9a; Table 1).
  - Batch size: larger global batch (e.g., 2,048 prompts) appears slower early but reaches a higher asymptote (`A ≈ 0.645` vs. `≈0.605–0.610`) and better downstream performance (Figure 10; Table 1; Appendix A.14).
  - Generations per prompt (fixed total batch): changing 8/16/24/32 generations per prompt (and adjusting prompts to keep total batch fixed) leaves fitted curves essentially unchanged at this scale (Appendix A.13; Figure 17).
  - Multi-task (math+code): both domains show clean, parallel scaling trends, and math-only curves remain predictive references; fitted asymptotes: code ≈0.615, math ≈0.595 (Figure 11; Table 1).
  - Downstream scaling: AIME-24 tracks the validation scaling trend, confirming transfer; e.g., Figure 1b and Figure 9b.
- Stability observations (Appendix A.15)
  - Truncation rates above ~10–15% often coincide with instability and degradation.
  - `ScaleRL` keeps truncations <5% for >90% of steps at batch 768 and similar low rates at larger scale; larger models and longer budgets reduce truncations further.

Overall, the experiments back three central claims:
- The sigmoidal model is predictive across setups and scales (Figures 1, 7–11).
- `ScaleRL` is competitive or better than prevalent recipes and more predictable at scale (Figure 2).
- Design choices separate into ceiling raisers (loss/precision) and efficiency boosters (off-policy streaming, normalization/aggregation, curricula) (Figures 5–7).

## 6. Limitations and Trade-offs
- Modeling assumptions
  - The sigmoidal curve is empirical; while strongly supported here, there is no theoretical proof it must hold for all RL settings, data, or reward types (Section 2.1; Appendix A.4).
  - Fits exclude the earliest training (first ~1.5k GPU-hours) to avoid transient regimes (Section 2.1; Appendix A.7); extrapolation depends on having entered the predictable region.
- Scope of tasks and rewards
  - Focus is mainly on verifiable math (and a math+code mixture later). Other RL-for-LLM regimes (dialogue preferences, multi-turn planning, dense/structured rewards) are not analyzed here (Sections 2, 5, 7; Appendix A.1). 
- Stability and hyperparameters
  - Some methods (e.g., GSPO) show mid-training instability on larger models (Appendix A.17.4).
  - DAPO’s performance ceiling is sensitive to upper clipping `ϵmax` (Appendix A.17.1); robustness depends on careful tuning.
- Compute and engineering constraints
  - The approach relies on significant compute (individual runs up to 100k GPU-hours), a generator–trainer split, and careful kernel/precision control (Sections 2–5).
  - FP32 at the LM head improves scalability but increases compute/memory for that layer (Section 3.2; Figure 5b).
- Generalization measurement
  - Predictability is established on held-out in-distribution validation; downstream correlations are promising but not a formal generalization study (Section 7; Appendix A.14).

## 7. Implications and Future Directions
- How this changes the field
  - Provides a practical, validated way to do “RL scaling law” analysis—separating asymptotic ceiling from efficiency—and to select recipes that will continue to improve as compute grows.
  - Makes RL experimentation more accessible: small budgets can forecast large-run outcomes, mitigating the “only large labs can know” problem.
- What research it enables
  - Systematic studies of compute allocation across axes: model size, sequence length, batch size, generations per prompt, reward modeling, verifiers, and multi-task mixtures, all within a predictive framework (Sections 5, 7).
  - Cross-regime scaling laws: connecting pre-training compute, model size, and RL compute to predict total system returns (Section 7).
  - Robustness and theory: why truncated IS (CISPO) is both stable and robust; formal conditions under which sigmoidal scaling emerges; extensions to multi-turn RL and agentic interactions (Section 7).
- Practical applications and guidance
  - If you need higher ultimate performance (ceiling), invest in: larger models (MoE), longer generation budgets, and precision fixes at the LM head (Figures 1, 5b, 9; Table 1).
  - If you need faster returns per unit compute, invest in: PipelineRL streaming, batch/aggregation/normalization choices, and curricula like No-Positive-Resampling (Figures 4, 6–7).
  - Monitor truncations; treat rising rates as early instability warnings and adjust generation budgets or length-control mechanisms (Appendix A.15).

Selected quotes grounding key results:
- > “We fit a sigmoid curve (Equation (1)) on pass rate… up to 50k GPU hours and extrapolate to 100k… The extrapolated curve closely follows extended training” (Figure 1a).
- > “CISPO/GSPO achieve a higher asymptotic reward compared to DAPO” (Figure 5a).
- > “Using FP32 precision in the final layer (LM head) gives a considerable boost in the asymptotic reward” (Figure 5b).
- > “ScaleRL surpasses… prevalent recipes… achieving an asymptotic reward of A = 0.61” and shows predictable extrapolations across methods (Figure 2).
- > “Larger batch size is slower in training but settles at a higher asymptote… and avoids downstream stagnation” (Figure 10 and §5).
- > “ScaleRL scales predictably on math and code” with clean, parallel trends (Figure 11).

In short, this work supplies both the scientific tool (a robust compute–performance fit) and the engineering recipe (`ScaleRL`) needed to turn RL for LLMs from an art into a predictable, scalable practice.
