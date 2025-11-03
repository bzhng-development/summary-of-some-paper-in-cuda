# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

**ArXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)

## 🎯 Pitch

DeepSeekMath introduces a 7B-parameter open-source language model that delivers breakthrough mathematical reasoning abilities, combining a massive, high-quality, multilingual math corpus with a novel, lightweight Group Relative Policy Optimization (GRPO) reinforcement learning method. Achieving 51.7% on the competition-level MATH benchmark—surpassing much larger models without external toolkits—it sets a new standard for open models, democratizing access to high-level mathematical reasoning and paving the way for affordable, scalable, and transparent AI-driven mathematics.

---

## 1. Executive Summary (2-3 sentences)
This work introduces DeepSeekMath, a 7B-parameter open language model that substantially improves mathematical reasoning by combining large‑scale math-focused pre‑training, targeted instruction tuning, and a new reinforcement learning algorithm called Group Relative Policy Optimization (GRPO). The model reaches 51.7% Top‑1 accuracy on the competition‑level MATH benchmark without external tools or voting (Table 5), matching or surpassing many much larger systems while using an RL method that is significantly lighter than PPO (Figure 4).

## 2. Context and Motivation
- Problem/gap addressed
  - Open models lag well behind top proprietary systems (e.g., GPT‑4, Gemini Ultra) on challenging mathematical reasoning tasks (Abstract; Figure 1).
  - Even within open models, strong performance has typically required very large parameter counts (e.g., Minerva 540B) or extensive tool use/self-consistency voting, which complicate deployment and fairness of comparisons.
  - Standard PPO-based RL for reasoning is compute- and memory-intensive because it trains both a policy and a value (critic) model (Section 4.1.1; Equation (1)), making it difficult to push RL at scale.

- Why this matters
  - Mathematical reasoning is a stringent test of multi-step reasoning and factual precision—capabilities that often transfer to broader reasoning tasks (Section 1). Reliable math reasoning has direct applications in education, science, engineering, and verification.

- Prior approaches and their limits
  - Data: Earlier math corpora were relatively small or English-only (e.g., OpenWebMath 13.6B tokens, Proof-Pile-2 ~52B tokens with heavy arXiv reliance). Scaling and multilingual coverage were limited (Section 2.2; Table 1).
  - Models: Open models like Mistral-7B or Llemma-34B improved but still trailed on MATH (Table 2).
  - RL methods: PPO improves alignment but is resource-heavy due to the critic and per-token advantage estimation, and requires careful KL regularization to avoid reward hacking (Section 4.1.1; Equations (1)-(2)).

- Positioning
  - This work pursues three levers simultaneously:
    1) A large, high-quality, multilingual math corpus mined from Common Crawl (Section 2.1; Figure 2).
    2) Instruction tuning oriented around chain-of-thought (CoT), program-of-thought (PoT), and tool-integrated reasoning (Section 3).
    3) GRPO, an RL algorithm that removes the critic by using group-relative rewards, cutting memory needs while boosting math reasoning (Section 4.1; Figure 4).

## 3. Technical Approach
This section explains what is built and how it works.

- Pre-training data pipeline (Section 2.1; Figure 2)
  - Start from a seed of math pages (OpenWebMath) to train a `fastText` classifier to detect math-like pages.
  - Apply URL-level de-duplication and near-duplicate removal to Common Crawl, yielding 40B HTML pages.
  - Iteratively mine: use the classifier to recall high-scoring pages; discover math-related domains (e.g., high recall rate domains), manually annotate math URL paths within these domains; add the newly found pages to the seed; retrain the classifier; repeat.
  - After four iterations: 35.5M math pages totaling 120B tokens (“DeepSeekMath Corpus”). To prevent benchmark leakage, filter any page with exact 10‑gram overlap (or shorter exact overlaps ≥3‑gram) with benchmark items (end of Section 2.1).

- Model pre-training (Section 2.3)
  - Initialize from `DeepSeek-Coder-Base-v1.5 7B`.
  - Train for 500B tokens with a mixture:
    - 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, 10% general English/Chinese text.
  - Standard AdamW schedule and long context training (details Section 2.2.1; 7B run uses LR max 4.2e‑4 and 10M-token batch).

- Instruction tuning (Section 3)
  - Data: 776K supervised examples across English and Chinese with three formats:
    - `Chain-of-Thought (CoT)`: solutions as step-by-step natural-language reasoning (Wei et al., 2022).
    - `Program-of-Thought (PoT)`: solutions as executable Python snippets that carry out computation (Chen et al., 2022; Gao et al., 2023).
    - `Tool-integrated reasoning`: interleaving language reasoning with tool calls (e.g., Python via `sympy`) (Gou et al., 2023).
  - Sources include GSM8K, MATH (annotated with tool-integrated solutions), a subset of MathInstruct, Lila-OOD, and Chinese K‑12 problems (Section 3.1).
  - Training: 500 steps, batch 256, max 4K tokens, LR 5e‑5 (Section 3.2).

- Reinforcement learning with GRPO (Section 4.1)
  - Background: In PPO, the policy maximizes a clipped objective using an estimated “advantage” `A_t` derived from a learned value function and a reward model, with a per-token KL penalty to a reference model to avoid reward over-optimization (Equations (1)-(2); Figure 4 top).
  - Key idea in GRPO: Remove the value function. For each question `q`, sample a group of `G` candidate answers `{o_i}` from the current (old) policy. Score each answer with a reward model. Use the group’s normalized rewards as the “advantage” for all tokens in that answer. Add a direct KL term between the policy and a reference policy (Equations (3)-(4); Figure 4 bottom).
    - Outcome supervision (Section 4.1.2): Each answer gets one reward at the end. Compute a z‑score within the `G` samples; use it as the per-token advantage.
    - Process supervision (Section 4.1.3): The reward model scores intermediate reasoning steps; the token-level advantage is the sum of normalized future step rewards. This gives finer-grained guidance for long solutions.
  - KL regularization: Instead of subtracting KL in the reward (as in Equation (2)), GRPO adds an explicit, unbiased KL estimate as a loss term (Equation (4)), simplifying advantage computation (Equation (3)).
  - Iterative GRPO (Algorithm 1; Section 4.1.4): Periodically refresh the reference policy to the current policy and continue training. In parallel, continuously train the reward model using newly sampled data plus a 10% replay buffer of historical reward data.

- RL training details (Section 4.2)
  - Policy start: `DeepSeekMath-Instruct 7B`.
  - RL data: only CoT-format GSM8K and MATH SFT questions (~144K), intentionally excluding other SFT data to probe generalization.
  - Sampling: 64 outputs/question, max length 1024, batch size 1024.
  - Optimization: policy LR 1e‑6, KL coefficient 0.04; a single policy update after each exploration stage.
  - Reward models: trained from `DeepSeekMath-Base 7B`, LR 2e‑5; iterative improving with replay.

- Formal math evaluation setup (Table 3; Section “Formal Mathematics”)
  - Task: “informal-to-formal” proving on miniF2F using Isabelle. The model writes proof sketches; Sledgehammer completes details.

## 4. Key Insights and Innovations
- Scaled math pre-training from the open web is both feasible and high quality
  - Different from prior math corpora that were smaller or arXiv-heavy, the 120B-token multilingual DeepSeekMath Corpus produces stronger downstream math performance than MathPile, OpenWebMath, and Proof-Pile-2 at equal training budgets (Section 2.2; Table 1; Figure 3).
  - Significance: Demonstrates that careful mining of Common Crawl, not just arXiv, can richly supply domain-specific training data at scale.

- Code-first initialization boosts math reasoning and tool-using ability
  - Pre-training from a code model and mixing math+code data improves both step-by-step and program-aided math reasoning (Section 5.1.1; Tables 6–7).
  - Practical takeaway: coding skills and math reasoning reinforce each other, particularly when models must write and execute Python to solve problems.

- GRPO: an RL algorithm that removes the critic by using group-relative rewards
  - Instead of a separate value network, GRPO uses the mean and standard deviation of rewards among multiple samples for the same question to compute token-level advantages (Sections 4.1.1–4.1.3; Equation (3); Figure 4).
  - Why it matters: reduces memory and complexity compared to PPO, yet delivers larger gains than online RFT and RFT, especially with process supervision (Figure 5). Iterative GRPO further improves over a single pass (Figure 6).

- RL improves distributional reliability more than raw capability
  - On GSM8K and MATH, RL increases Maj@K (majority vote over K samples) substantially but does not improve Pass@K (whether at least one of K is correct) (Figure 7). This suggests RL aligns the model to produce the correct solution more consistently rather than discovering new solutions it could not already reach.

- ArXiv-heavy pre-training provides limited or negative gains on these math tasks
  - Training solely on arXiv corpora (MathPile or RedPajama arXiv) brings little to no improvement and sometimes hurts performance on math benchmarks and formal proving (Section 5.1.2; Tables 8–9). This challenges a common assumption about the centrality of arXiv for math reasoning.

## 5. Experimental Analysis
- Evaluation methodology (Sections 2.3, 3.2, 4.2; Table 5)
  - Benchmarks span English and Chinese, from grade-school to college/competition level:
    - English: GSM8K, MATH, SAT, OCW courses, MMLU‑STEM, BBH; tool‑aided variants GSM8K+Python, MATH+Python.
    - Chinese: MGSM‑zh, CMATH, Gaokao‑MathCloze, Gaokao‑MathQA.
    - Formal math: miniF2F with Isabelle (Table 3).
  - Metrics: Top‑1 accuracy unless otherwise noted; some experiments also report majority voting over multiple samples (Table 5 gray numbers) and Maj@K/Pass@K (Figure 7).
  - Baselines: Closed systems (GPT‑4, Gemini, etc.) and open models from 7B to 70B/72B (Table 5). For base-model comparisons: Mistral‑7B, Llemma‑7B/34B, and Minerva 7B/62B/540B (Table 2).

- Main quantitative results
  - Base model (no instruction tuning or RL):
    - On MATH, `DeepSeekMath-Base 7B` reaches 36.2% Top‑1, surpassing Llemma‑34B (25.3%) and even Minerva 540B (33.6%) (Table 2). On GSM8K, it reaches 64.2%.
    - With tools: GSM8K+Python 66.9% and MATH+Python 31.4%, beating Llemma‑34B (64.6% and 26.3%) (Table 3).
    - Formal proving: miniF2F-valid/test 25.8/24.6% vs 21.0/21.3% for Llemma‑34B (Table 3).
    - General skills: improves over its coder precursor on MMLU and BBH (54.9% and 59.5%), and remains competitive on code tasks (Table 4).
  - Instruction tuning (CoT/PoT/tool-integrated):
    - `DeepSeekMath-Instruct 7B` achieves 46.8% on MATH and 82.9% on GSM8K (Top‑1 without tools), outperforming many open and closed models of similar or larger size (Table 5).
    - With tool integration, it reaches 57.4% on MATH (Table 5).
  - Reinforcement learning (GRPO):
    - `DeepSeekMath-RL 7B` increases to 51.7% on MATH and 88.2% on GSM8K (Top‑1 without tools) and to 58.8% on MATH with tools (Table 5). These are the strongest open results among 7B–70B open models listed.
    - Iterative GRPO improves steadily across steps and iterations (Figure 6).
    - GRPO with process supervision outperforms GRPO with outcome-only rewards, which in turn surpasses online RFT and offline RFT (Figure 5).

- Data quality and ablations
  - Corpus comparison: Training a 1.3B model on each math corpus shows the DeepSeekMath Corpus yields the steepest and most persistent learning curves (Figure 3) and the best final performance (Table 1).
  - Code vs math training:
    - Two-stage “code→math” training gives higher gains than “general→math”; adding code especially boosts program-aided reasoning (Tables 6–7).
    - One-stage mixed “code+math” helps tool-use reasoning and mitigates forgetting of code skills relative to pure two-stage math finetuning (Tables 6–7).
  - ArXiv ablations: ArXiv-only training did not help and sometimes degraded performance (Tables 8–9).

- Support for claims
  - The evidence is broad (many benchmarks, languages, and settings) and includes ablations on data, training order, and RL algorithms. Where claims are nuanced, the paper provides specific counter-evidence (e.g., limited arXiv benefits) and diagnostics (Maj@K vs Pass@K, Figure 7).

- Notable details and caveats
  - Some headline numbers in Table 5 include majority-vote results (gray); the Top‑1 comparisons remain strong and are reported as such for DeepSeekMath.
  - The abstract also reports a 60.9% Majority@64 on MATH for DeepSeekMath‑7B (self-consistency), consistent with the distributional reliability effect seen in Figure 7.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - RL supervision depends on reward models trained from correctness rules and annotations; errors in reward labels propagate (Section 5.2.3 notes ~20% noise in PRM800K).
  - RL training data only covers GSM8K and MATH CoT questions; generalization to other math domains is empirical rather than guaranteed (Table 5 shows positive OOD gains but scope is limited).

- Computational and data constraints
  - GRPO removes the critic (saving GPU memory), but still requires sampling many candidates per query (G=64 in Section 4.2), which is inference-heavy.
  - The full pre-training uses 500B tokens and a specialized 120B-token math corpus; reproducing at that scale is non-trivial even if the pipeline is described (Sections 2.1–2.3).

- Problem settings not fully addressed
  - Geometry and theorem-proof remain weaker than top proprietary systems; the paper notes failures on triangles/ellipses and fewer gains in formal math compared to closed models (Section 6: Conclusion, Limitation).
  - Few-shot improvements are limited compared to GPT‑4 (Section 6).

- Methodological trade-offs
  - GRPO optimizes distributional alignment (higher Maj@K) more than expanding solution coverage (Pass@K), suggesting improvements in consistency rather than new capability discovery (Figure 7).
  - ArXiv’s limited utility here might be task- and scale-dependent; the paper flags this as provisional (Section 5.1.2).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that careful web-scale data mining plus code-first initialization can make a 7B open model highly competitive on hard math tasks, closing the gap with much larger or closed systems (Tables 2 and 5).
  - Introduces GRPO as a practical alternative to PPO for reasoning-heavy RL, potentially lowering barriers to RL-based alignment in academic and applied settings (Section 4.1; Figure 5).

- Research avenues
  - Data: Extend the iterative mining pipeline (Figure 2) to other domains (e.g., scientific reasoning, law), improve coverage of geometry/visual math, and explore better multilingual balance.
  - RL algorithms: Develop robustness to noisy rewards (“weak-to-strong” alignment), richer uncertainty estimates in reward models, and finer-grained process rewards at scale (Section 5.2.3).
  - Exploration: Replace naive nucleus sampling with advanced search (e.g., tree-of-thoughts), and leverage efficient decoding methods to reduce the cost of G‑sample exploration (Section 5.2.3).
  - Formal math: Tighten the informal–formal bridge, possibly via co-training with proof assistants and better autoformalization datasets (Table 3 shows promise but headroom remains).

- Practical applications
  - Math tutoring and automated homework/grading that relies on verifiable reasoning steps (CoT/PoT).
  - Scientific and engineering assistance where precise numerical reasoning and tool use (Python) are essential.
  - Formal verification workflows that convert informal sketches to machine-checkable proofs (Table 3).

> Representative headline result: “DeepSeekMath‑RL 7B reaches 51.7% Top‑1 on MATH and 88.2% on GSM8K without tools; with tools, 58.8% on MATH” (Table 5).

> Mechanism highlight: “GRPO replaces the critic with a group-relative baseline and adds a direct KL term, outperforming RFT and online RFT and improving further with process supervision and iterative retraining” (Equations (3)–(4); Figure 5; Figure 6).

Overall, the paper combines a strong data contribution, practical training insights (code-first, arXiv cautions), and an efficient RL algorithm to push open mathematical reasoning notably forward.
