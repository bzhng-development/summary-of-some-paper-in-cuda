# MUTUAL REASONING MAKES SMALLER LLMS STRONGER PROBLEM-SOLVERS

**ArXiv:** [2408.06195](https://arxiv.org/abs/2408.06195)

## 🎯 Pitch

This paper introduces rStar, a novel self-play mutual reasoning framework that dramatically boosts the reasoning abilities of small language models (SLMs) at inference time, without any fine-tuning or stronger teacher models. By combining search-augmented generation (via rich, human-like actions in Monte Carlo Tree Search) and unsupervised peer discrimination (consensus between SLMs), rStar unlocks substantial accuracy gains on complex reasoning tasks. This approach demonstrates that through structured exploration and peer agreement, even smaller, accessible models can attain near state-of-the-art problem-solving skills—enabling practical, efficient deployment in real-world scenarios where resources or supervision are limited.

---

## 1. Executive Summary
This paper introduces rStar, a test-time self-play framework that substantially improves the reasoning accuracy of small language models (SLMs) without any fine-tuning or stronger “teacher” models. It combines a search-based generator (augmented Monte Carlo Tree Search, MCTS) with a peer discriminator (a second SLM) to produce and validate reasoning trajectories, yielding large gains on math and commonsense datasets (e.g., GSM8K jumps from 12.51% to 63.91% for `LLaMA2-7B`; Table 2).

## 2. Context and Motivation
- Problem addressed
  - Small language models underperform on complex, multi-step reasoning tasks even with standard prompting like Chain-of-Thought (CoT). For instance, `Mistral-7B` reaches only 36.46% on GSM8K using few-shot CoT (Table 2).
  - Many recent improvements depend on supervised fine-tuning, often using data distilled by stronger models like GPT-4. The paper targets the harder setting: improve reasoning at inference time, without training or access to superior models.
- Why it matters
  - Practical: SLMs are cheaper, faster, and easier to deploy. If they can reason better at test-time, more applications (education, on-device assistants, privacy-preserving deployments) become feasible.
  - Scientific: Demonstrates how search and peer agreement can unlock latent reasoning ability in smaller models, advancing understanding of test-time compute vs. parameter scaling.
- Prior approaches and their shortcomings
  - Self-consistency (SC): sample many full CoT solutions and majority-vote. It increases robustness but requires many samples and struggles when most samples are low quality (Section 4, Table 2).
  - Tree of Thoughts (ToT): performs breadth-first search over intermediate thoughts but often with a single action type, limiting exploration quality (Related Work; Section 3.2).
  - RAP: an MCTS-based method that decomposes problems into sub-questions, but relies on the model’s “self-rewarding” (self-evaluation), which is unreliable for SLMs (Appendix A.1; Table 6 shows replacing the self-evaluated component with random values barely changes accuracy on GSM8K).
  - Reward/value model training: can work but needs annotated data and risks overfitting to tasks (Related Work).
- Positioning
  - rStar combines a richer search action space with a novel, unsupervised verification step called mutual reasoning consistency. It aims to produce better candidates than SC/ToT and select better answers than self-verification or naive majority voting (Sections 3.2–3.3).

## 3. Technical Approach
At a high level, rStar decouples reasoning into two roles:
- a generator SLM (`M`) that explores many candidate reasoning trajectories using MCTS with a rich set of actions, and
- a discriminator SLM (`M^`) that verifies those trajectories by attempting to complete them consistently from randomly masked partial steps.

Key terms
- `SLM` (Small Language Model): a model with relatively few parameters (e.g., 3.8B–8B) compared to frontier LLMs.
- `Trajectory`: a complete reasoning path from question to final answer, represented as a sequence of intermediate steps.
- `MCTS` (Monte Carlo Tree Search): a search algorithm that iteratively builds a tree by alternating selection, expansion, simulation (rollout), and backpropagation to estimate which actions lead to good outcomes.
- `UCT` (Upper Confidence bounds applied to Trees): the decision rule MCTS uses to balance exploiting high-value actions and exploring under-sampled ones: UCT(s,a) = average value + exploration bonus (Section 3.2).
- `Self-consistency`: sampling multiple answers and picking the majority.

Step-by-step mechanism
1) Search space and actions (generator; Section 3.2, Fig. 3 and Table 1)
   - rStar augments MCTS with a set of five “human-like” actions to diversify and tailor exploration to the current reasoning state, rather than using a single action:
     - `A1`: propose one next thought step (granular step-wise reasoning).
     - `A2`: propose all remaining thought steps (fast completion when the model is confident).
     - `A3`: propose the next sub-question and answer it (problem decomposition; akin to Least-to-Most prompting).
     - `A4`: re-answer the current sub-question with few-shot CoT (a targeted redo when a sub-answer may be unreliable).
     - `A5`: rephrase the (sub-)question by explicitly listing conditions (reduces misinterpretation).
   - Action ordering constraints enforce sensible sequences (e.g., `A4` only after `A3`; `A5` only from the root). An ablation on GSM8K (Table 1) shows each action contributes; using all five yields the best accuracy among tested subsets.

2) Reward design for search (generator; Section 3.2)
   - Problem: SLM self-evaluation is weak (Appendix A.1); training value models is costly and task-specific.
   - Solution: Use terminal answer “confidence” as the only explicit reward signal and backpropagate it to intermediate nodes:
     - For a terminal node, compute reward from the likelihood estimated via self-consistency (i.e., majority vote among rollouts that reach that terminal answer).
     - Backpropagate this scalar along the path so that actions frequently leading to correct answers get higher `Q(s,a)` in future UCT decisions.
   - Intuition: mimic AlphaGo’s idea of scoring by downstream outcomes rather than local self-evaluations.

3) MCTS details (generator; Section 3.2)
   - Start from the root (the question).
   - Repeatedly perform selection (using UCT), expansion (choose and apply an action to generate the next step), simulations/rollouts (default rollout policy), and backpropagation (update `Q` and visit counts).
   - Stop at a terminal node or max depth (5 for most tasks; 8 for MATH as stated in Section 4.1).
   - Collect all complete trajectories sampled during rollouts as candidate solutions.

4) Mutual reasoning consistency (discriminator; Section 3.3, Fig. 2 and Fig. 4)
   - For each candidate trajectory `t = x ⊕ s1 ⊕ … ⊕ sd`, randomly choose a split point `i` between 20% and 80% of steps (Section 4.1).
   - Provide the prefix `x ⊕ s1 ⊕ … ⊕ s(i-1)` to the second SLM `M^` and prompt it to complete the rest.
   - If `M^`’s completion yields the same final answer as the original trajectory, mark `t` as “mutually consistent.” The idea is that agreement between two peers on the same partial reasoning increases the likelihood of correctness (peer verification analogy in Section 3.3).
   - This is a trajectory-level filter—distinct from majority vote over independent samples and from self-verification.

5) Final selection (Section 3.3)
   - Among validated trajectories, compute a final score by multiplying the path’s MCTS reward with the terminal node’s confidence.
   - Pick the highest-scoring trajectory as the final solution.

Design choices and rationale
- Rich action space vs. single-action methods: increases the chance of escaping poor local modes and better aligns with how humans flexibly reason (Table 1 ablation; Fig. 3 illustrates branching behavior).
- Outcome-based rewards vs. self-rewarding: avoids unreliable self-judgment by SLMs (Appendix A.1 shows minimal performance drop when RAP’s self-evaluated component is randomized; Table 6).
- Peer discriminator vs. majority vote/self-verification: majority vote needs many correct samples; self-verification is weak on SLMs (Table 5, left). Mutual consistency turns partial hints into stronger, external unsupervised feedback.

Implementation snapshot (Section 4.1)
- 32 MCTS rollouts per question; tree depth 5 (8 for MATH).
- `A1` and `A3` can branch up to 5 nodes per depth; others default to 1.
- Discriminator is `Phi3-mini-4k` (3.8B) for all experiments except when `Phi3` is the generator, in which case it self-discriminates.
- Discriminator runs in parallel over candidates (amortizing cost).

## 4. Key Insights and Innovations
- Mutual reasoning consistency as unsupervised verification (Section 3.3)
  - What’s new: A second, comparable SLM receives a random prefix of a trajectory and must independently complete it. If the final answers match, the trajectory is deemed more likely correct.
  - Why it matters: It provides targeted, trajectory-aware validation without labeled data or stronger models. Table 5 (left) shows it beats both majority vote and self-verification across generators.

- Rich, human-like action space for MCTS (Section 3.2; Table 1)
  - What’s new: Five complementary actions (incremental step, full completion, decompose-and-answer, re-answer, rephrase) with ordering constraints reflect realistic reasoning moves.
  - Why it matters: Enables broader and more precise exploration than single-action methods like RAP or ToT, improving the chance of generating at least one correct path.

- Outcome-centric reward without training value models (Section 3.2)
  - What’s new: Use terminal answer confidence (via self-consistency at rollouts) as the sole reward; backpropagate through the path.
  - Why it matters: Avoids unreliable self-rewarding (Appendix A.1) and the cost/overfitting risks of training task-specific reward models.

- Strong test-time gains that rival or exceed fine-tuned baselines (Section 4.2, Table 2)
  - Significance: For GSM8K, `LLaMA2-7B` improves from 12.51% (few-shot CoT) to 63.91% with rStar; `Mistral-7B` from 36.46% to 81.88%; `LLaMA3-8B-Instruct` from 74.53% to 91.13%. This reframes the trade-off between parameter count, fine-tuning, and test-time compute.

These are fundamental innovations in verification (mutual consistency) and exploration (rich action space) rather than incremental hyperparameter tweaks.

## 5. Experimental Analysis
Evaluation setup (Section 4.1)
- Models: `Phi3-mini` (3.8B), `LLaMA2-7B`, `Mistral-7B`, `LLaMA3-8B`, `LLaMA3-8B-Instruct`.
- Datasets:
  - Math word problems: `GSM8K`, `GSM-Hard`, `SVAMP`.
  - Advanced math: `MATH-500` subset (Table 3).
  - Commonsense: `StrategyQA`.
- Baselines (Section 4.2):
  - Zero-shot and few-shot CoT.
  - Self-consistency (SC) with 8/64/128 samples and majority voting.
  - `ToT` (search with single-step action, BFS).
  - `RAP` (MCTS with sub-question action and self-rewarding).
- rStar reporting:
  - “`rStar (generator @maj)`” uses the MCTS generator but selects answers via majority vote, isolating the generator’s effect.
  - “`rStar`” adds the mutual consistency discriminator.

Main quantitative results (Table 2; Table 3)
- Large gains on GSM8K:
  - > “rStar boosts GSM8K accuracy from 12.51% to 63.91% for `LLaMA2-7B`, from 36.46% to 81.88% for `Mistral-7B`, from 74.53% to 91.13% for `LLaMA3-8B-Instruct`.”
  - Generator-only already beats ToT/RAP/SC on most settings; adding the discriminator yields the top results (Table 2).
- Harder math:
  - GSM-Hard: rStar improves `LLaMA3-8B-Instruct` to 37.53% and `Mistral-7B` to 37.91%, surpassing SC, ToT, and RAP (Table 2).
  - MATH-500 (Table 3): rStar reaches 42.94% (`LLaMA3-8B-Instruct`) and 48.60% (`Phi3-mini-4k`), beating SC and RAP.
- SVAMP and StrategyQA:
  - SVAMP: consistent gains across SLMs (e.g., `Mistral-7B` 86.40% with rStar vs. 76.70% with SC@64; Table 2).
  - StrategyQA: improvements are smaller but positive (e.g., `Mistral-7B` 70.31% vs. 65.50%–65.65% for SC@8/64/128; Table 2).

Ablations and robustness
- Rollout sensitivity (Fig. 5):
  - rStar improves accuracy even with 2 rollouts and scales with more rollouts; RAP saturates or declines beyond 4 rollouts for `LLaMA3-8B-Instruct`.
- Action-space ablation (Table 1):
  - Each action adds value; using all five is best (75.0% on sampled GSM8K subset vs. 70.5% for RAP-like action only).
- Generator ablations (Table 4):
  - With majority voting or with rStar’s discriminator, the proposed generator outperforms RAP and SC-generated pools.
  - Replacing rStar’s reward with self-evaluation (“Ours + Self-eval”) hurts performance, confirming the reward choice.
- Discriminator ablations (Table 5):
  - Mutual consistency beats majority vote and self-verification on both SC- and rStar-generated candidates.
  - Using a stronger discriminator (GPT-4) yields only a small bump over `Phi3-mini` (91.13% → 92.57% for `LLaMA3-8B-Instruct`), suggesting the mechanism—not raw discriminator strength—is the main driver.
- Self-rewarding analysis (Appendix A.1, Table 6):
  - Randomizing RAP’s self-evaluation term (`r1`) barely changes results on GSM8K and Multiarith, indicating weak self-judgment by SLMs.

Compute cost (Appendix A.2)
- > Average per-question cost on GSM8K: “166.81” calls and “367.1k” generated tokens for `LLaMA2-7B`; similar scale for `Mistral-7B` (“148.90” calls, “348.6k” tokens).
- End-to-end: ~4.5 days per model on a single A100 GPU for the full GSM8K test set (32 rollouts). Parallelization can mitigate this.

Assessment of evidential strength
- The breadth of models and datasets, along with targeted ablations, supports the core claims:
  - Diverse action space improves candidate quality (Table 1, Table 4).
  - Mutual consistency improves selection over alternatives (Table 5).
  - Gains hold across math and commonsense tasks (Table 2; Table 3).
- Evidence is strongest on math word problems and MATH-500; StrategyQA gains are modest, indicating task dependence.

## 6. Limitations and Trade-offs
- Compute intensity at inference time (Appendix A.2)
  - Many model calls and tokens per question; latency may be high for interactive use unless heavily parallelized.
- Reliance on candidate recall
  - The discriminator can only validate what the generator explores. If no candidate is near-correct, mutual consistency cannot rescue the outcome.
- Risk of “agreeing on the wrong answer”
  - Mutual consistency validates agreement, not truth. If both SLMs are biased by the same misleading prefix, they may agree on an incorrect completion.
- Task coverage
  - Evaluations focus on math word problems and a single commonsense dataset (StrategyQA). Knowledge-heavy, multi-hop factual tasks, code-heavy settings, or multimodal reasoning are not explored.
- Prompt and hyperparameter sensitivity
  - Action ordering, branching factors, rollout counts, and split ratios for the discriminator introduce knobs that may require tuning per domain.
- No cost-benefit frontier analysis
  - While Fig. 5 shows scaling with rollouts, there is no explicit Pareto analysis of accuracy vs. tokens/latency for deployment settings.
- Data leakage not discussed
  - As with most LLM evaluations, the risk that models have seen benchmark problems during pretraining is not analyzed.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that test-time compute plus peer agreement can partially substitute for parameter count and supervised fine-tuning. This reframes the engineering trade-off for deploying capable, private, and cheaper models.
- Practical applications
  - On-device tutors and solvers that reason better without external services.
  - Safety-critical settings that require sanity checks: mutual consistency offers a principled, lightweight verifier without curated labels.
  - Pipeline plug-in: rStar’s discriminator can vet outputs from any multi-trajectory generator (e.g., tool-augmented solvers, program-of-thought systems).
- Research directions
  - Learned or adaptive action policies: meta-controllers that pick among A1–A5 based on state.
  - Richer discriminators: hybrid verifiers that combine `M^` with tool-verified steps (e.g., symbolic math, unit checks) while preserving the no-superior-teacher constraint.
  - Cost-aware search: dynamic rollouts that stop early when mutual consistency is strong; prioritized replay of promising prefixes.
  - Beyond agreement: pairwise or multi-peer tournaments, adversarial discriminators, or probabilistic calibration of agreement vs. correctness.
  - Dataset generation: Use mutually consistent trajectories to build high-quality, weakly-labeled process data for later fine-tuning, bridging test-time and train-time improvements.

In short, rStar offers a compelling blueprint for unlocking SLM reasoning through search-time diversity and peer validation. The gains are large and well-supported on math datasets, with a clear path to broader tasks and to more cost-efficient variants.
