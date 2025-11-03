# Reinforcement Learning for Reasoning in Large Language Models with One Training Example

**ArXiv:** [2504.20571](https://arxiv.org/abs/2504.20571)
**Authors:** Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Lucas Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen
**Institutions:** Microsoft Research, University of Washington, University of Southern California, Allen Institute for AI, University of California, Santa Cruz, Georgia Institute of Technology

## 🎯 Pitch

This paper introduces a groundbreaking approach using reinforcement learning with verifiable rewards (RLVR) to enhance small language models' mathematical reasoning abilities, achieving similar performance with just one training example compared to thousands. By demonstrating that a single well-chosen problem can unlock latent reasoning capabilities, this work significantly reduces data needs, reshaping the constraints of data-heavy model training and highlighting the inherent potential in base models.

---

## 1. Executive Summary
This paper shows that reinforcement learning with a verifiable reward (RLVR) can dramatically improve a small language model’s mathematical reasoning using just one training example. With a carefully chosen single problem, the 1.5B-parameter `Qwen2.5-Math` model jumps from 36.0% to 73.6% accuracy on MATH500 and from 17.6% to 35.7% average across six math benchmarks, matching the outcome of training on 1.2k examples (Fig. 1; Table 8).

## 2. Context and Motivation
- Problem addressed
  - How much data is actually necessary to improve a base language model’s reasoning with RL when rewards are verifiable (e.g., right/wrong math answers)? The paper asks: “To what extent can we reduce the training dataset for RLVR while maintaining comparable performance compared to the full dataset?” (end of Sec. 1).
- Why this matters
  - Practical: RLVR at scale is expensive and bottlenecked by high-quality data. If one or a few examples can activate reasoning, training becomes cheaper and simpler.
  - Scientific: Clarifies whether recent reasoning gains stem from algorithmic changes, large datasets, or latent abilities already present in base models. Results support the view that base models possess substantial untapped reasoning capacity (Sec. 3 and 4; Appendix D.2).
- Prior approaches and gaps
  - Most RLVR work focuses on improving algorithms (e.g., PPO/GRPO variants) and scaling data (Sec. 1; Related Work). Data-centric questions—what data and how much—are underexplored. The closest prior (LIMR) prunes thousands of examples but does not push to a single example (Related Work; Sec. A).
- Positioning
  - The paper contributes a data-efficiency perspective: it demonstrates 1-shot and few-shot RLVR, provides a simple data selection heuristic, analyzes mechanisms (policy gradient vs. entropy vs. KL/weight decay), and documents phenomena (post-saturation generalization, cross-category transfer, self-reflection emergence).

## 3. Technical Approach
This section explains the training recipe, the minimalist data strategy, and how the models are evaluated.

- Core training setup: RLVR with GRPO
  - RLVR uses a “verifiable” reward: a rule-based function returns 1 if the final answer matches the ground truth and 0 otherwise (Sec. 2 “RL Loss Function”).
  - GRPO (Group Relative Policy Optimization) optimizes a policy using group-normalized advantages. For a prompt `q`, the trainer samples `G` responses `{o_i}` from the current (old) policy, computes rewards `{r_i}`, and normalizes them to advantages `A_i` by subtracting group mean and dividing by group std (Eq. 6). The policy gradient loss then pushes up on above-average samples and down on below-average ones (Eq. 4).
  - Two regularizers are used (Sec. 2; Appendix B.1):
    - `KL` loss to a reference model (Eq. 5) to preserve language quality and stabilize updates.
    - `Entropy` loss with a negative coefficient to encourage per-token entropy, i.e., more diverse outputs and exploration (Eq. 7–8). This is not strictly required by GRPO but is on by default in the training stack (verl).
  - Important definitions
    - “Outcome reward”: the binary 0/1 correctness reward described above.
    - “Format reward”: a weaker reward that gives 1 if a final answer can be parsed (e.g., appears in `\boxed{}`) regardless of correctness. This allows isolating “format correction” from true reasoning gains (Appendix C.2.3; Fig. 13–14; Table 14).

- One-example data selection: historical variance score
  - The paper proposes a simple heuristic to rank examples by how much their training accuracy fluctuates early in training (Sec. 2 “Data Selection”).
  - Method: run short RLVR training over a pool (1.2k DeepScaleR subset), record per-epoch training accuracy for each example `i`, compute variance `v_i = var(s_{i,1}, …, s_{i,E})` (Eq. 1), and rank examples by `v_i` (Eq. 2). Examples `π1`, `π13`, etc., come from this ranking.
  - Rationale: higher reward variance is historically linked to useful credit assignment in RL; such items may better stimulate learning (Sec. 2; footnote in Sec. 3.3 adds that this is not necessarily optimal, but works well).

- Practicalities of 1-shot training
  - To satisfy batch-size constraints, the single chosen example is duplicated to fill the batch (footnote in Sec. 3.1).
  - Default training details for `Qwen2.5-Math-1.5B`: KL coefficient `β=0.001`, entropy coefficient `α=-0.001`, rollout temperature `t=0.6`, max prompt length 1024, max response 3072, eight sampled responses per prompt per step, 128 batch/mini-batch (Sec. 3.1 “Training”).
  - Evaluation uses the Qwen Math evaluation pipeline on six benchmarks: MATH500, AIME 2024, AMC 2023, Minerva Math, OlympiadBench, AIME 2025. Non-math generalization is checked on ARC-Easy and ARC-Challenge (Sec. 3.1 “Evaluation”; Table 1).

- What the selected single examples look like
  - `π1` (Table 2) is a moderate algebra/physics problem (wind pressure varies with area and velocity cubed). The base model already performs all steps except precisely computing `∛2048`, often producing “12.7”, “12.8”, or “13” (Sec. 3.2.1).
  - `π13` (Table 21) is a geometry problem requiring solving a circle equation and a line intersection condition.

- Mechanistic probes and diagnostics
  - To understand where the gains come from, the paper compares:
    - Outcome reward vs. format reward (Appendix C.2.3; Fig. 13–15; Table 14).
    - Full GRPO loss vs. removing policy gradient, KL, weight decay, or entropy (Table 5; Fig. 12; Sec. 4.1).
    - Training the entropy term alone (Tables 6 and 13; Fig. 12; Sec. 4.2).
    - Label correctness/robustness (Table 5 rows 11–13; Table 17).
    - Prompt modification that strips problem structure (Table 18).
  - The paper also logs response length, entropy trends, and the frequency of self-reflection words (“rethink”, “recheck”, “recalculate”) during training (Fig. 4).

## 4. Key Insights and Innovations
- A. One example can match thousands in RLVR
  - With `Qwen2.5-Math-1.5B`, training on a single example `π13` reaches MATH500 73.6% and 35.7% average across 6 benchmarks—virtually the same as training on 1.2k examples (73.6% / 35.9%) and a 7.5k MATH set (75.4% / 36.7%) (Fig. 1; Tables 8–9).
  - Two examples (`{π1, π13}`) slightly surpass both 1.2k and 7.5k on the 6-benchmark average (36.6% vs. 35.9% and roughly on par with 36.7%) (Fig. 1; Table 8).
  - Significance: radically reduces data needs, challenging the assumption that large RLVR datasets are necessary to unlock reasoning.

- B. Post-saturation generalization with 1-shot RLVR
  - Training accuracy on the single example saturates before 100 steps, yet test accuracy continues to rise for over 1k steps (Fig. 2). Even after clear overfitting of the training example into multilingual “gibberish,” test responses remain normal and accurate (Fig. 3).
  - Significance: demonstrates that generalization can improve after the memorization point (“post-saturation”), suggesting exploration-induced refinement rather than standard fitting is driving downstream gains.

- C. Policy gradient is the main driver; entropy-guided exploration amplifies gains
  - Ablation (Table 5): adding only the policy gradient loss achieves most of the improvement (Row 2 vs. Row 5), while weight decay and KL have minor effects (Rows 3–4). Adding entropy yields further gains but is coefficient-sensitive (Rows 5–6).
  - Entropy and higher sampling temperature increase exploration and help sustain post-saturation gains (Fig. 5).
  - Significance: clarifies mechanism—policy gradient on a single example provides the core learning signal; controlled exploration helps avoid stagnation.

- D. Not just format correction
  - Format-only RL (rewarding parseable answers) yields large gains (e.g., MATH500 ~65%), but outcome-reward 1-shot RLVR significantly exceeds it (e.g., +7.4% on MATH500; +5.8% average) even when both have similar rates of boxed answers (Table 14; Fig. 13–15).
  - LLM-as-a-judge confirms improvements are not just “put the right number in `\boxed{}`” (Table 15).
  - Significance: the single example is catalyzing broader reasoning improvements, not merely enforcing answer formatting.

- E. Broad viability and cross-task effects
  - The phenomenon holds across models (`Qwen2.5-Math-7B`, `Llama-3.2-3B-Instruct`, `DeepSeek-R1-Distill-Qwen-1.5B`) and RL algorithms (GRPO and PPO) (Table 4; Table 11; Table 12).
  - 1-shot math RLVR also improves non-math ARC scores, sometimes exceeding full-set RLVR (Table 1).
  - Single-example training often boosts categories other than the training example’s category (Table 3; Fig. 6).

## 5. Experimental Analysis
- Evaluation methodology
  - Benchmarks: MATH500, AIME 2024, AMC 2023, Minerva Math, OlympiadBench, AIME 2025 (Sec. 3.1 “Evaluation”; Appendix B.3). For small sets like AIME, pass@1 is averaged over 8 runs (avg@8).
  - Models: primarily `Qwen2.5-Math-1.5B`, with additional runs on 7B Qwen, Llama 3.2 3B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B (Sec. 3.1 “Models”; Table 4).
  - RL settings: GRPO with outcome reward by default; format-reward and entropy-only variants as baselines/ablations (Sec. 2; Sec. 4; Appendix C.2.3). Eight samples per prompt per step; temperature 0.6 unless noted.

- Main quantitative results (Qwen2.5-Math-1.5B)
  - Base vs. 1-shot vs. full-set (best 6-benchmark average; Table 8):
    - Base: 17.6 avg; 36.0 MATH500.
    - 1-shot `{π13}`: 35.7 avg; 73.6 MATH500.
    - 2-shot `{π1, π13}`: 36.6 avg; 74.8 MATH500.
    - 1.2k DSR-sub: 35.9 avg; 73.6 MATH500.
    - 7.5k MATH train: 36.7 avg; 74.4 MATH500.
    - Format-reward baseline: 28.7 avg; 65.0 MATH500.
  - Post-saturation generalization: training accuracy on `π1` and `π13` hits ~100% <100 steps; test curves keep rising for 1k–1.5k steps (Fig. 2). Overfitting artifacts appear only after ~1.4k–1.8k steps, while test outputs remain coherent (Fig. 3).

- Cross-category and robustness (Table 3)
  - Training on many single examples gives large overall gains on MATH500. Most examples deliver ≥30% absolute improvement over base. Exceptions are a mislabeled item (`π1207`) and an extremely hard item (`π1208`).
  - Gains are not confined to the training example’s topic; e.g., a Number Theory training example does not necessarily maximize Number Theory test gains (Table 3).

- Generalization beyond math (Table 1)
  - On ARC-Easy/Challenge, `π13` 1-shot RLVR outperforms full DSR-sub RLVR and base model:
    - ARC-E: 55.8 vs. 42.2 (DSR-sub) vs. 48.0 (base).
    - ARC-C: 33.4 vs. 29.9 (DSR-sub) vs. 30.2 (base).

- Across models/algorithms (Table 4; Table 11; Table 12)
  - `Qwen2.5-Math-7B`: 1-shot reaches 79.2 MATH500, avg 40.2; few-shot (2–16) gets close to DSR-sub; format baseline lags.
  - `Llama-3.2-3B-Instruct`: smaller absolute gains; 2-shot `{π1, π13}` exceeds DSR-sub average (21.0 vs. 19.8).
  - `DeepSeek-R1-Distill-Qwen-1.5B`: few-shot improves but the gap vs. 1.2k remains larger (Table 12); evaluation at 8k vs. 32k context reported.

- Ablations and diagnostics
  - Loss components (Table 5; Fig. 12):
    - Policy gradient alone nearly matches full GRPO (Row 2 vs. Row 5).
    - Entropy helps (Row 5 vs. Row 4), but too high hurts (Row 6).
    - KL and weight decay have modest impact (Rows 3–4).
  - Entropy-only training (Tables 6, 13): a few steps yield moderate gains (e.g., Qwen-1.5B MATH500 63.4) but below format-reward RL and far below outcome-reward RL.
  - Format vs. outcome reward (Table 14; Fig. 13–15): big but insufficient gains from format alone; outcome reward still adds substantial improvements despite similar `\boxed{}` rates.
  - Label robustness (Table 5 rows 11–13; Table 17): small label errors (12.7 vs. 12.8) barely matter; blatant but guessable wrong labels can harm more than completely unguessable wrong labels; high proportions of random wrong labels (90%) in full-set RL can make it worse than 1-shot.
  - Prompt modification (Table 18): training only on a sub-step (`∛2048`) underperforms the full problem `π1`, indicating the value of richer reasoning structure.
  - Behavior shifts (Fig. 4): response lengths and entropy rise late in 1-shot training; self-reflection words in test outputs increase after ~1.2–1.3k steps, coinciding with test gains.

- Do the results support the claims?
  - The paper triangulates with: multiple models and algorithms; strict baselines (format-reward, entropy-only); ablations disentangling loss components; error analyses (wrong labels, sub-prompts); behavioral metrics (boxed ratio, reflection words).
  - Overall, the evidence convincingly supports the central claims: 1–2 well-chosen examples can match 1.2k examples, gains are not mere format fixing, policy gradient is primary, and exploration aids post-saturation generalization.

## 6. Limitations and Trade-offs
- Scope and generality
  - Focuses on math reasoning with verifiable final answers; does not test coding or domains where automatic checking is harder (Limitations, Appendix D.1).
  - Uses relatively small-to-mid models (1.5B/7B). Larger models (e.g., 32B) are not evaluated.
- Data and selection
  - The “historical variance score” requires a brief run over a dataset to rank examples, and it is not guaranteed optimal (Sec. 2–3.3). Many examples work; a few do not (Table 3). Selecting robust single examples for arbitrary models remains open.
- Computational cost
  - Even with one example, training itself is not cheap: duplication to fill batches, millions of rollouts before overfitting (Fig. 3 caption; Appendix D.1). This is a data-efficiency, not necessarily compute-efficiency, result.
- Stability and sensitivity
  - Entropy coefficient and temperature require tuning (Table 5; Fig. 5). Llama-3.2-3B-Instruct shows more unstable training curves (Appendix C.1.5).
- Evaluation nuances
  - Some full-set RLVR settings show pass@8 degradation over time (Table 20). Understanding trade-offs between pass@1 and pass@k remains open.
- Mechanistic understanding
  - The precise theoretical explanation for post-saturation generalization is not fully resolved; the paper hypothesizes that entropy-induced exploration keeps policy gradients non-zero and acts like implicit regularization (Appendix D.3; Fig. 16).

## 7. Implications and Future Directions
- How this changes the field
  - Establishes that RLVR does not need large curated datasets to unlock reasoning: a single, well-chosen problem can suffice. This reframes RLVR as a problem of identifying catalytic examples and managing exploration rather than sheer data scale.
  - Supports the perspective that base models already contain substantial reasoning capabilities that RL can “activate” (Appendix D.2).

- Practical applications
  - Lean RLVR pipelines for small labs: quickly upgrade a base math model with 1–4 examples.
  - Rapid domain adaptation: if an application has a verifiable check (e.g., certain STEM problems), a tiny set of “seed” items may be enough to yield large gains.
  - Data curation tooling: the historical-variance heuristic (Eq. 1–2) offers a simple scoring method; 1-shot runs enable fast per-example evaluation of “training value.”

- Research directions
  - Better example selection: learn-to-rank examples by expected policy-gradient impact; interpretability-driven criteria; meta-learning of catalytic prompts (Sec. D.4).
  - Exploration beyond entropy: algorithms that diversify reasoning paths without destabilizing training; adaptive entropy or novelty bonuses (Sec. C.2.2; D.4).
  - Other domains and reward types: coding with unit tests, scientific QA, program synthesis; tasks without crisp verifiers via learned reward models plus sparse verifiable checks.
  - Theory of post-saturation generalization: connect to double descent, implicit regularization of SGD, and RL credit assignment dynamics (Appendix D.4).
  - Robustness: systematic studies of label noise tolerance (Table 17), format sensitivity (Appendix C.2.3), and pass@k behavior (Table 20).

> Bottom line: With one or two well-chosen problems, RL with a simple 0/1 correctness reward can “ignite” reasoning in small LLMs to essentially the same level as training on thousands of problems—provided the training loss contains a policy gradient component and some mechanism (like entropy) encourages exploration (Fig. 1; Table 5).
