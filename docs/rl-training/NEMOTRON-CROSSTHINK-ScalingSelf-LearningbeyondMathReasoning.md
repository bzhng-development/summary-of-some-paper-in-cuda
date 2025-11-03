# NEMOTRON-CROSSTHINK: Scaling Self-Learning beyond Math Reasoning

**ArXiv:** [2504.13941](https://arxiv.org/abs/2504.13941)

## 🎯 Pitch

NEMOTRON-CROSSTHINK pioneers a systematic framework for training large language models with reinforcement learning across both mathematical and broad, real-world reasoning tasks. By curating diverse multi-domain datasets, applying structured templates for verifiable rewards, filtering unverifiable samples, and optimizing data blending, it achieves significant accuracy boosts—including on non-math benchmarks—while making correct answers more concise and efficient. This unlocks scalable, efficient, and generalizable self-learning in LLMs far beyond mathematical reasoning, paving the way for robust, cross-domain AI reasoning in practical applications.

---

## 1. Executive Summary (2-3 sentences)
NEMOTRON-CROSSTHINK introduces a practical way to train large language models (LLMs) with reinforcement learning (RL) on reasoning tasks beyond math by turning messy, multi-domain data into verifiable RL training signals. It does this via a pipeline that curates diverse datasets, constrains answer formats so they can be auto-checked, filters out unverifiable samples, and blends data sources strategically—yielding accuracy gains across math and non-math benchmarks and producing shorter, more efficient correct answers.

## 2. Context and Motivation
- Problem/gap addressed
  - RL has boosted LLM reasoning, but most real successes center on math and coding where correctness is easy to verify. Extending RL to general reasoning (law, social sciences, history, etc.) is hard because answers are diverse, evaluation is ambiguous, and high-quality RL data is scarce (Section 1).
  - Prior multi-dataset RL efforts diversify data but do not quantify which sources matter most, how to blend them, or how to create verifiable rewards for non-math tasks (Section 1).

- Why it matters
  - Real-world applications require reasoning across domains and formats (multiple-choice vs open-ended). Without verifiable reward signals, RL either stays narrow (math) or risks unstable training and reward hacking (Section 1).
  - A general framework that expands RL beyond math could unlock models that reason accurately across many disciplines and do so efficiently—reducing inference cost (Figure 1; Section 5 “Token efficiency”).

- Prior approaches and limitations
  - Math-centric RL (e.g., DeepSeek-R1, Open-Reasoner-Zero) achieves strong math scores because correctness can be programmatically checked, but these recipes do not generalize well to non-math tasks (Section 1; Table 4 comparison with ORZ).
  - Works that mix data often: (a) don’t test the relative value of each source; (b) don’t offer strategies for blending by domain or format; (c) rarely handle non-math verification, so reward functions don’t scale (Section 1).

- Positioning
  - NEMOTRON-CROSSTHINK contributes a full, reproducible recipe to scale RL-based “self-learning” beyond math: multi-domain curation, templating for verifiable rewards, filtering, and principled blend design; then trains with GRPO RL and evaluates with detailed ablations (Figure 2; Tables 3–7).

## 3. Technical Approach
At a high level, NEMOTRON-CROSSTHINK turns diverse, messy QA data into RL-ready training instances with verifiable rewards, then optimizes a policy with Group Relative Policy Optimization (GRPO). Figure 2 shows the end-to-end pipeline.

Step-by-step pipeline
1) Data curation across domains (Section 2; Table 1)
   - Two sources:
     - `Dsyn`: synthetic QA generated from Common Crawl, guided by MMLU topic coverage.
     - `Dos`: open-source QA datasets.
   - Two domains:
     - `Dgpr` (general-purpose reasoning): Natural Reasoning (100k) and MMLU-Train (99,842), plus synthetic Syn-QA (192,930).
     - `Dmr` (math reasoning): NuminaMath (87,350), PersonaSkill-MATH (100k), Math (8,523).
   - Total training samples: 588,645 (Table 1).
   - Why this matters: mixing domains introduces varied cognitive patterns (symbolic math vs narrative/legal reasoning), which can enhance generalization if reward signals remain reliable (Section 2).

2) Templating to constrain answer space and enable verifiable rewards (Section 2 “Applying Templates…”)
   - Two question templates over `Dgpr`:
     - `TMCQ`: multiple-choice format.
     - `TOpen`: open-ended format (remove answer options).
   - Rationale: MCQ and open-ended elicit different reasoning strategies; converting formats and unifying outputs helps computing programmatic rewards.
   - Practical choice: discard MCQs that become ill-posed without options (e.g., “Which of the following…?”) to avoid uncheckable or ambiguous answers.

3) Filtering for verifiability (Section 2 “Data Filtering and Formatting”)
   - Keep only samples where a simple rule-based reward can be computed:
     - MCQ (`Dmcq`): ensure the ground-truth correct option `a*` is among the listed options; drop if it is not.
     - Open-ended (`Dopen`): keep only short answers (`≤ 10` words) so exact matching remains feasible.
     - Math (`Dmr`): drop entries with missing `a*`.
   - Why: This ensures stable and interpretable rewards and avoids noisy supervision that would destabilize RL.

4) Blending strategies (Section 2 “Data Blending”; Table 2 and Appendix Table 8)
   - The framework builds six blends to test what mix works:
     - Data source emphasis:
       - `Bmr↑`: 2:1 math:gpr.
       - `Bgpr↑`: 2:1 gpr:math.
       - `Bnd`: natural proportions of datasets.
     - Question type emphasis:
       - `Bmcq↑`: 2:1 MCQ:Open-ended.
       - `Bopen↑`: 2:1 Open-ended:MCQ.
     - Data usefulness:
       - `Bscore`: weight datasets by their standalone performance (from Table 3).
   - Also include single-domain controls:
     - `Bonly_mr` (math-only), `Bonly_gpr` (gpr-only).
   - Purpose: quantify how domain mix and question format affect downstream reasoning and efficiency.

5) Reinforcement learning with GRPO (Section 2 “Reinforcement Learning with GRPO”)
   - Base models: `Qwen2.5-7B` and `Qwen2.5-32B` (Section 3).
   - GRPO overview:
     - For each question `q`, sample a group of `G` candidate outputs from the current policy.
     - Compute a reward for each output.
     - Compute a relative advantage by standardizing each reward within the group (z-score).
     - Update the policy with a PPO-style clipped objective and a KL penalty to a reference policy (Eq. 1).
   - Why GRPO:
     - No separate critic model is needed; relative advantages from a group reduce memory and simplify training while keeping PPO’s stability benefits (Section 2).
   - Key hyperparameters (Section 3):
     - Learning rate 1e-6, batch size 128, 8 rollouts per prompt, max context length 5000, KL coefficient 0.001, temperature=1.0, top-p=1.0.

6) Simple, verifiable reward function (Section 2 “Rule Based Reward Modeling”)
   - Final reward `R` is a logical AND of accuracy and format:
     - `Racc`: exact equality between model’s final answer and ground truth.
     - `Rformat`: response must use `<think>...</think>` for rationale and put final answer inside `\boxed{...}`.
   - Why this specific format:
     - It cleanly separates reasoning from the final answer and makes automated checking trivial.
   - Example:
     - MCQ: model must output `\boxed{C}` (or short-form) to get accuracy reward.
     - Open-ended short answers (≤10 words): exact match is feasible.

7) Difficulty-based filtering for scaling (Section 5 “Difficulty Filtering”; Table 7)
   - For larger models (32B), create a filtered blend by removing questions that a smaller model (`Qwen2.5-7B`) solves in zero-shot.
   - Intuition: focus RL compute on “hard” questions that require deeper reasoning rather than knowledge recall.

Analogy for GRPO’s group-relative advantage
- Think of each group of `G` sampled answers as a mini-competition. Instead of needing a judge (critic) to estimate absolute quality, you rank contestants by their relative performance within the group; then nudge the policy toward winners and away from losers.

## 4. Key Insights and Innovations
- Multi-domain RL with verifiable rewards beyond math
  - What’s new: a practical recipe to turn general reasoning tasks into RL-ready, auto-verifiable training instances via templating and filtering (Figure 2).
  - Why it matters: verifiable rewards are the bottleneck outside math. This approach keeps rewards simple and stable without complex evaluators, enabling scalable RL on non-math data.

- Systematic data blending and its quantified impact (Table 4; Appendix Table 8)
  - What’s new: controlled blends across domain emphasis (math vs gpr), question type (MCQ vs open-ended), and usefulness (data weighted by standalone performance).
  - Why it matters: shows that a 2:1 general-purpose-to-math blend (`Bgpr↑`) gives the best average accuracy, outperforming math-only and naturally sampled mixes.

- Template studies that turn format decisions into performance wins (Tables 5 and 6)
  - Question template: converting everything to open-ended improves average accuracy by +1.21% over mixed MCQ+open-ended (Table 5).
  - Answer template: for MCQ datasets, asking for short-form labels (e.g., “A”) is better than long-form label+description (+1.20%; Table 6).
  - Significance: careful formatting reduces answer-space ambiguity and prevents reward noise from string-matching.

- Difficulty filtering scales gains at larger model sizes (Table 7)
  - Simple proxy (unsolved by 7B) to select “hard” questions improves the 32B model by +2.15% average over the unfiltered blend.
  - Significance: compute can be focused on genuinely challenging cases without needing external difficulty labels.

- Response efficiency as an emergent benefit (Figure 3; Appendix Table 9)
  - `Bgpr↑` uses fewer tokens for correct answers on general-purpose tasks (−39.6% vs math-only; Table 9) yet writes longer when math derivations demand it.
  - Significance: multi-domain RL yields dynamic verbosity—concise when possible, detailed when necessary—cutting inference cost.

## 5. Experimental Analysis
Evaluation setup (Section 3)
- Models: `Qwen2.5-7B` and `Qwen2.5-32B`.
- RL framework: veRL (HybridFlow RLHF implementation).
- Benchmarks and metric: accuracy on
  - Math: MATH-500, AMC23
  - General reasoning: MMLU (test), MMLU-PRO, AGIEVAL, GPQA-Diamond, SUPERGPQA
- Inference protocol:
  - For both open-ended and MCQ, read the final answer from `\boxed{...}` and compare to ground truth.
  - For MCQ benchmarks, ground truths include both the option label and its description to match training format.
  - Report accuracy averaged over 3 greedy runs.

Main quantitative results and takeaways

A) Individual datasets under self-learning (Table 3; 7B, 250 RL steps per dataset)
- Baseline `M` (no additional RL): average 44.75.
- Notable performers:
  - NuminaMath: average 53.06 (best overall; strong in math and solid transfer to non-math).
  - Syn-QA: 45.65 (good on MMLU-PRO, AGIEVAL, and MATH-500).
  - Natural Reasoning: 44.82 (surprisingly strong on math tasks like MATH-500 and AMC23).
- Underperformer:
  - MMLU-Train: 34.78 on average, despite breadth; suggests raw MMLU-Train alone is not ideal for RL without blending and formatting.
- Implication: no single dataset dominates across all tasks; math corpora can transfer, but diverse sources are needed for broad gains.

B) Blending strategies (Table 4; 7B unless noted)
- Baseline `M`: 44.75 average.
- ORZ (math-centric baseline): 55.20 average.
- Multi-domain blends:
  - `Bgpr↑` (2:1 gpr:math): average 58.12 (best overall).
  - `Bmr↑` (2:1 math:gpr): 57.72.
  - `Bnd` (natural proportions): 55.66.
- Question-type blends:
  - `Bopen↑`: 57.49; `Bmcq↑`: 56.89.
- Usefulness-weighted blend:
  - `Bscore`: 56.95 (helps over baseline, but worse than domain-aware blends).
- Single-domain controls:
  - `Bonly_mr`: 57.82 (very strong on math; second-best overall).
  - `Bonly_gpr`: 53.30 (improves over baseline, but weaker on math).
- Key quote:
  > Table 4: `Bgpr↑` achieves the highest average accuracy (58.12), outperforming ORZ (55.20) and math-only or gpr-only blends. It also posts large gains on MMLU-PRO (+12.82 points over baseline) and AGIEVAL (+15.12 points over baseline).
- Interpretation:
  - Mixing domains beats single-domain training for broad generalization.
  - GPR-heavy blends do not sacrifice math much, while substantially improving non-math.

C) Token efficiency and dynamic verbosity (Figure 3; Appendix Table 9)
- Quote:
  > Appendix Table 9: On GPR tasks, mean tokens for correct responses are 385 (Bgpr↑) vs 639 (math-only) and 1115 (ORZ). On math tasks, they are 622 (Bgpr↑) vs 731 (math-only) and 1257 (ORZ).
- Takeaway:
  - `Bgpr↑` is more concise when possible and expands only when needed for math, indicating an efficient reasoning style (Figure 3).
  - Across all tasks, `Bgpr↑` uses about 28% fewer tokens for correct answers than the math-only blend (Section 5 “NEMOTRON-CROSSTHINK is token efficient…”).

D) Template ablations
- All-open-ended questions vs mixed (Table 5):
  - Open-ended only: 56.87 average vs 55.66 mixed (+1.21).
  - Reason: open-ended removes the chance of “guessing” in 4-option MCQs (~25% baseline), pushing models to reason rather than exploit options.
- MCQ answer label “Short” vs “Long” (Table 6):
  - Short label only: 54.50 average vs 53.30 long (+1.20).
  - Reason: exact-match reward penalizes minor paraphrases in long-form; short labels reduce reward noise.

E) Difficulty-based filtering at scale (Table 7; 32B)
- Quote:
  > Table 7: `Qwen-2.5-32B` baseline=54.33 average; `Bgpr↑` (unfiltered) = 65.84; filtered `Bf(gpr)↑` = 67.99 (+2.15 over unfiltered).
- Gains are consistent across MMLU-PRO, GPQA-Diamond, AGIEVAL, SUPERGPQA, and math benchmarks.
- Interpretation: focusing on “hard” samples (unsolved by 7B) yields better use of RL compute for larger models.

F) Sub-category analyses (Figures 5–7)
- MMLU-PRO and AGIEVAL:
  - `Bgpr↑` outperforms math-only in many non-math categories (e.g., law, business, psychology), and even slightly in MMLU-PRO’s math category (Figure 5).
  - In AGIEVAL’s olympiad-style math, math-only has a small edge, but `Bgpr↑` dominates in language-heavy domains (Figure 6).
- SUPERGPQA:
  - `Bgpr↑` leads across most professional categories (engineering, economics, education, law, philosophy), with parity only in the science-heavy bucket (Figure 7).
- Conclusion:
  - Multi-domain RL systematically boosts broad reasoning without materially harming math performance.

Do the experiments support the claims?
- Yes, through:
  - Cross-benchmark gains (Table 4, Table 7).
  - Controlled ablations on templates (Tables 5–6).
  - Data-source vs question-type vs usefulness blends (Table 4).
  - Efficiency analysis (Figure 3; Appendix Table 9).
- Caveats:
  - Some design choices (e.g., blend ratios) are empirically motivated rather than theoretically derived.
  - Exact-match rewards may underestimate performance on longer free-form answers (discussed below).

## 6. Limitations and Trade-offs
- Reward design simplifies non-math evaluation
  - Exact equality (`Racc`) and strict formatting (`Rformat`) stabilize RL but can penalize near-correct outputs (Table 6 discussion). This may bias models toward short, rigid answers and discourage valid paraphrasing.

- Open-ended scope is intentionally constrained
  - To ensure verifiability, open-ended answers are capped at ≤10 words (Section 2 “Data Filtering”). This excludes many realistic tasks requiring long-form reasoning or explanations.

- Difficulty proxy may conflate “hard” with “unknown”
  - Filtering with a smaller model’s errors (Table 7) assumes unsolved questions are harder; they may also reflect knowledge gaps or domain coverage differences rather than reasoning difficulty.

- Blend design leaves room for optimization
  - The best-performing ratio (2:1 gpr:math) is found empirically; a principled or adaptive scheduler could perform better. The “usefulness” blend (`Bscore`) weights by average performance, which misses task-specific strengths (Table 4 analysis).

- Compute cost and training details
  - RL with 8 rollouts, 5000-token context, and mixed open-ended reasoning is compute-intensive (Section 3). The paper reports limited hyperparameter tuning, suggesting headroom but also potential sensitivity.

- Generalization beyond the chosen setting
  - The framework focuses on text QA. Multi-turn, retrieval-augmented, multimodal, or long-form generation scenarios are not addressed directly.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that RL for reasoning can be scaled beyond math by engineering the data and output templates to preserve verifiability. This lowers the barrier to deploying RL on diverse tasks and suggests that “rewardable” general reasoning is feasible at scale (Figure 2; Tables 4–7).

- Practical applications
  - Enterprise and professional QA across law, economics, education, and engineering (SUPERGPQA categories; Figure 7).
  - Test prep and tutoring across STEM and humanities (MMLU/MMLU-PRO; Tables 4–5).
  - Cost-sensitive deployments where concise correctness matters (token-efficiency gains; Figure 3, Appendix Table 9).

- Follow-up research
  - Reward design
    - Move beyond exact match: incorporate semantic similarity, programmatic checkers for non-math (e.g., rule templates, retrieval-backed verification), or learned reward models for open-ended reasoning.
    - Reward shaping for process quality (not only final answer), e.g., partial credit for logical steps.
  - Adaptive data scheduling
    - Curriculum or bandit-style blend selection that prioritizes domains or formats with the highest marginal utility; dynamic ratio adjustment during training.
  - Broader formats and tasks
    - Support longer free-form answers by pairing template constraints with better evaluators.
    - Extend to multi-turn dialogue reasoning, tool use, or multimodal inputs where verifiability is still possible (e.g., code execution, table checks).
  - Efficiency control
    - Integrate explicit thinking-length control (cf. L1-style methods) alongside the observed emergent token efficiency to further reduce inference cost without hurting accuracy.

Block-quoted highlights
- Overall blend performance:
  > Table 4: `Bgpr↑` (2:1 general-purpose:math) achieves the top average accuracy (58.12), outperforming ORZ (55.20) and single-domain blends.

- Template advantages:
  > Table 5: Training with all open-ended questions improves average accuracy by +1.21% over mixed MCQ+open-ended.  
  > Table 6: Short-form MCQ answers outperform long-form by +1.20% on average.

- Difficulty filtering at scale:
  > Table 7: On Qwen2.5-32B, filtering to “hard” examples yields 67.99 average vs 65.84 unfiltered (+2.15), and both exceed the base model (54.33).

- Token efficiency:
  > Figure 3 and Appendix Table 9: `Bgpr↑` reduces mean tokens for correct GPR answers by 39.6% vs math-only and 65.4% vs ORZ, while remaining appropriately verbose on math tasks.

In sum, NEMOTRON-CROSSTHINK offers a clear and effective recipe for taking RL-based reasoning beyond math: constrain outputs so rewards are verifiable, curate and filter multi-domain data, blend it strategically, and optimize with GRPO. The result is a more accurate and more efficient reasoner across a wide range of tasks.
