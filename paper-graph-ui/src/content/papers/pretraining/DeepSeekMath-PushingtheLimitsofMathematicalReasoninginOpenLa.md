# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

**ArXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)

## 🎯 Pitch

DeepSeekMath introduces the first open 7B-parameter language model that surpasses 50% Top-1 accuracy on the challenging MATH benchmark, achieved by meticulously curating a massive 120B-token math-focused web dataset and pioneering the memory-efficient Group Relative Policy Optimization (GRPO) algorithm for reinforcement learning. This innovation narrows the gap with proprietary giants like GPT-4, offering the research community a scalable foundation for advanced mathematical reasoning and robust multi-step problem solving—critical for science, education, and AI alignment.

---

## 1. Executive Summary
DeepSeekMath builds an open 7B-parameter language model that substantially improves mathematical reasoning by (1) pretraining on a new 120B‑token math-heavy web corpus mined from Common Crawl and (2) introducing a memory‑efficient reinforcement learning algorithm, Group Relative Policy Optimization (GRPO). The result is the first open 7B model to exceed 50% Top‑1 accuracy on the competition-level MATH benchmark without external tools (51.7%; Table 5), approaching much larger closed systems such as GPT‑4 and Gemini Ultra.

## 2. Context and Motivation
- Problem/gap
  - Open mathematical reasoning models lag far behind proprietary systems (Section 1). Prior open models either rely on relatively small math corpora or use RL algorithms that are expensive to train.
  - The field lacks a scalable way to mine high-quality math web data at scale and a practical RL method that improves reasoning without the heavy “critic” model required by PPO.

- Why it matters
  - Strong math reasoning improves quantitative problem solving, scientific workflows, education, and formal theorem proving. It is also a bellwether for general multi-step reasoning ability (Section 1.2).

- Prior approaches and limitations
  - Minerva (closed, 540B) achieved strong MATH performance via massive math text pretraining (Table 2), but is not publicly available.
  - OpenWebMath (13.6B tokens) and Proof-Pile-2 (51.9B) provide filtered math text but are smaller than what is likely needed; performance plateaus quickly when repeatedly cycling through limited corpora (Figure 3).
  - PPO-based RL improves reasoning but requires an additional value (critic) model with high memory/computation and difficult token-level value estimation when only the final token receives a reward (Section 4.1.1).

- Positioning
  - DeepSeekMath contributes both data and algorithms: a scalable web-mined math corpus (120B tokens; Sections 2.1–2.2, Figure 2) and GRPO (Section 4.1, Equations 3–4), a PPO variant that eliminates the critic by using group-relative baselines.

## 3. Technical Approach
The system is a three-stage pipeline: large-scale math pretraining, supervised instruction tuning, and reinforcement learning with GRPO.

1) Math pretraining data: an iterative Common Crawl mining pipeline (Section 2.1, Figure 2)
- Seed and classifier bootstrapping
  - Start with OpenWebMath as positive examples and random web pages as negatives to train a `fastText` classifier (vector size 256, lr 0.1, 3-gram max, min count 3, 3 epochs).
- Large-scale recall and filtering
  - Deduplicate URLs and near-duplicates to reduce Common Crawl to 40B HTML pages; score pages with the classifier; keep only top-ranked math pages; run multiple scale pilots (40B/80B/120B/160B tokens) and select 40B tokens for the first pass.
- Domain discovery and targeted expansion
  - Identify domains with a high fraction of math content (e.g., mathoverflow.net). Human annotators label math-relevant URL paths (e.g., /questions), and uncollected pages from those paths are added as new positives. Retrain the classifier and repeat.
- Convergence and scale
  - After four iterations: 35.5M math pages, totaling 120B tokens (end of Section 2.1). Nearly 98% of data in iteration 4 had already appeared by iteration 3 (stopping criterion).
- Decontamination
  - Remove any web text containing exact matches with benchmark substrings using a 10‑gram (or exact ≥3‑gram for shorter cases) filter for GSM8K, MATH, CMATH, AGIEval, etc. (end of Section 2.1).

2) Base model training: `DeepSeekMath-Base 7B` (Section 2.3)
- Initialization and data mixture
  - Start from `DeepSeek-Coder-Base-v1.5 7B`. Train for 500B tokens with mixture: 56% DeepSeekMath corpus, 4% AlgebraicStack (math code), 10% arXiv, 20% GitHub code, 10% general English/Chinese web text.
- Training setup
  - Same optimizer/schedule as Section 2.2.1, with lr peak 4.2e‑4 and batch size 10M tokens.

3) Instruction tuning: `DeepSeekMath-Instruct 7B` (Section 3)
- 776K examples spanning English and Chinese; problems paired with:
  - `CoT` (chain-of-thought): step-by-step natural language reasoning traces.
  - `PoT` (program-of-thought): solving via Python snippets that compute intermediate steps.
  - Tool-integrated reasoning: natural language + code tool invocation.
- Training: 500 steps, batch 256, constant lr 5e‑5 (Section 3.2).

4) Reinforcement learning with GRPO: `DeepSeekMath-RL 7B` (Section 4)
- Why GRPO?
  - Traditional PPO needs a critic model to compute token-level advantages (`GAE`), which is difficult when only the last token receives reward and doubles memory. GRPO eliminates the critic by using group-relative baselines (Figure 4).
- How GRPO works (Section 4.1.1; Equations 3–4)
  - For each question `q`, sample a group of `G` responses `{o1..oG}` from the current policy (`π_old`).
  - Score each response with a reward model; compute a normalized relative reward inside the group (subtract mean, divide by std).
  - Use this normalized reward to form the token-level “advantage” for each response.
  - Add a KL regularizer with respect to a frozen reference model (the previous policy) using an unbiased KL estimator (Equation 4), and then optimize a PPO-style clipped surrogate without training any value model.
- Two supervision modes (Sections 4.1.2–4.1.3)
  - Outcome supervision: a single normalized reward applied to all tokens of the response (`Â_i,t = normalized_reward_i`).
  - Process supervision: reward a sequence of steps within the response; token advantages sum the normalized rewards of subsequent steps (encourages correct intermediate reasoning).
- Iterative RL (Algorithm 1, Section 4.1.4)
  - Periodically re-train the reward model on new samples from the latest policy using a replay buffer (10% historical data), reset the reference model to the latest policy, and continue GRPO (Figure 6).
- RL training setup (Section 4.2)
  - Start from `DeepSeekMath-Instruct 7B`, use only the CoT-format subsets of GSM8K and MATH (~144K questions).
  - Reward model: trained from `DeepSeekMath-Base 7B` (lr 2e‑5).
  - GRPO policy: lr 1e‑6, KL coefficient 0.04, sample 64 outputs per question, max length 1024, batch size 1024, one policy update per exploration stage.

Supporting framework: a unified view of SFT/RFT/DPO/PPO/GRPO
- Section 5.2.1 formalizes all these methods under a single gradient template (Equation 5), clarifying how they differ by data source, reward signal, and gradient coefficients (Table 10). This highlights why online sampling and magnitude-aware gradient coefficients matter (Figure 5).

Key terms defined
- `CoT` (chain-of-thought): a textual step-by-step reasoning trace.
- `PoT` (program-of-thought): solving by generating and executing code (usually Python).
- `Outcome supervision`: reward only on final answer correctness/quality.
- `Process supervision`: reward at intermediate steps of reasoning.
- `Advantage`: how much better/worse an action (token) is relative to a baseline; in GRPO, the baseline is the group mean reward, not a learned critic.
- `Maj@K` vs `Pass@K`: majority vote across K samples (Maj@K) versus at least one correct among K (Pass@K). Useful for analyzing distributional robustness versus raw capability (Figure 7).

## 4. Key Insights and Innovations
- Scalable, high-quality math web corpus from Common Crawl (Sections 2.1–2.2)
  - Novelty: an iterative domain-and-URL-path mining pipeline guided by a learned `fastText` classifier and human path annotation (Figure 2), yielding 120B math tokens—7× Minerva’s math web pages and 9× OpenWebMath (Section 1.1).
  - Significance: demonstrably higher average quality than Proof-Pile-2 at the same token budget (Figure 3); multilingual coverage strongly boosts Chinese math benchmarks (Table 1).

- Code-first pretraining improves math reasoning and tool use (Section 5.1.1)
  - Novelty: controlled experiments disentangle the effects of code vs general text before math training (Tables 6–7).
  - Significance: code pretraining (even before math) markedly improves program-aided math solving and accelerates later math training. Mixing code+math in one stage mitigates catastrophic forgetting.

- GRPO: RL without a critic, aligned to groupwise rewards (Section 4.1; Figure 4; Equations 3–4)
  - Novelty: replaces token-level value estimation with normalized group rewards; adds a direct KL penalty using an unbiased estimator; supports both outcome and process supervision.
  - Significance: reduces memory/compute versus PPO while improving performance over Online RFT and matching or exceeding PPO-style gains in practice (Figure 5).

- A unified paradigm connecting SFT, RFT, DPO, PPO, GRPO (Section 5.2.1; Equation 5; Table 10)
  - Novelty: expresses all methods as choices of data source (offline vs online), reward function (rule vs learned model), and gradient coefficients.
  - Significance: explains why online sampling and magnitude-aware penalties help, and frames RL as preference-optimized learning.

- Candid negative finding: arXiv papers alone do not help (Section 5.1.2; Tables 8–9)
  - Novelty: detailed ablations show arXiv-only training offers little or even negative gains on GSM8K, MATH, MMLU‑STEM, and miniF2F.
  - Significance: challenges a common assumption and shifts attention to curated web math text and code.

## 5. Experimental Analysis
- Evaluation methodology (Section 1.2; Sections 2.3–4.2)
  - Datasets
    - English reasoning: GSM8K, MATH, SAT, OCW Courses, MMLU‑STEM.
    - Chinese reasoning: MGSM‑zh, CMATH, Gaokao‑MathCloze, Gaokao‑MathQA.
    - Formal math: miniF2F (Isabelle).
    - Coding: HumanEval, MBPP.
  - Settings
    - Without tools: few-shot CoT prompting.
    - With tools: few-shot PoT (Python) prompting using libraries like `math` and `sympy` (Table 3).
    - Some results use self-consistency (Maj@K); gray numbers in Table 5 denote majority vote over 32 samples.

- Main results
  - Base model performance (no RL; Table 2)
    - `DeepSeekMath-Base 7B` achieves 64.2% (GSM8K) and 36.2% (MATH)—surpassing Minerva 540B on MATH (33.6%) and beating all open base models (e.g., Llemma‑34B: 25.3% MATH).
    - Strong Chinese results, e.g., 71.7% on CMATH (Table 2).
  - Tool use and formal proving (Table 3)
    - `DeepSeekMath-Base 7B` reaches 66.9% (GSM8K+Python), 31.4% (MATH+Python), exceeding Llemma‑34B (64.6%, 26.3%).
    - On Isabelle miniF2F: 25.8% (valid), 24.6% (test), outperforming prior base models.
  - General reasoning/coding (Table 4)
    - MMLU 54.9%, BBH 59.5%, while preserving coding proficiency (HumanEval 40.9%, MBPP 52.6%), showing math pretraining does not sacrifice coding.
  - Instruction tuning and RL (Table 5)
    - CoT (no tools): `DeepSeekMath-Instruct 7B` scores 46.8% (MATH), and `DeepSeekMath-RL 7B` reaches 51.7%—outperforming all open models up to 70B and most closed models except GPT‑4/Gemini Ultra.
    - Tool-integrated: `DeepSeekMath-Instruct 7B` achieves 57.4% (MATH), `DeepSeekMath-RL 7B` 58.8%; competitive with or exceeding much larger open models (e.g., ToRA 34B at 50.8%).
  - Quality of the DeepSeekMath corpus (Table 1, Figure 3)
    - With the same 1.3B training budget, the new corpus yields the best curves across 8 benchmarks and sustains improvement longer than smaller corpora (Figure 3).
  - Ablations: code vs math and training schedule (Tables 6–7)
    - Code → Math two-stage training beats General → Math; mixed Code+Math training best preserves coding while maintaining strong math tool use.
  - RL method comparisons (Figure 5)
    - Online RFT > RFT; GRPO > Online RFT; process-supervised GRPO > outcome-supervised GRPO.
  - Iterative GRPO (Figure 6)
    - Each iteration (refresh reward model + reference) lifts accuracy further, with the first iteration giving the biggest jump.
  - Why RL helps (Figure 7)
    - RL increases `Maj@K` (distributional robustness) but not `Pass@K` (raw best-of-K capability), suggesting RL sharpens the probability mass around correct reasoning rather than discovering new solution modes.

- Do the experiments support the claims?
  - Yes. The multi-pronged evidence—cross-benchmark comparisons (Tables 2–5), corpus ablations (Table 1, Figure 3), training schedule ablations (Tables 6–7), arXiv negative results (Tables 8–9), and RL method studies (Figures 5–7)—systematically supports the two central claims: scalable web math pretraining is effective, and GRPO improves reasoning with lower resource costs than PPO.

- Caveats and conditions
  - Some headline results use majority voting (self-consistency), marked in gray in Table 5.
  - RL training used only GSM8K/MATH CoT data (Section 4.2), which makes the out-of-domain gains noteworthy but also indicates the scope of RL data was limited.

## 6. Limitations and Trade-offs
- Data and compute
  - Pretraining involves 500B tokens and a 10M-token batch size (Section 2.3), and GRPO samples 64 outputs per question (Section 4.2), which is compute-intensive despite removing the critic.
- Reward model quality and supervision granularity
  - Outcome supervision provides sparse signals; process supervision needs step segmentation and can inherit labeling noise. Even curated datasets like PRM800K have ~20% annotation noise (Section 5.2.3).
- Coverage and skills
  - Despite strong improvement, geometry and theorem-proof skills still trail top closed models; the paper notes difficulties with triangle/ellipse problems (Section 6).
- Data selection bias
  - The web-mined corpus is shaped by the classifier and annotated URL paths; certain subdomains (e.g., some geometry topics) may be underrepresented (Section 6).
- Generalization vs distribution sharpening
  - RL mainly improves `Maj@K` rather than `Pass@K` (Figure 7), indicating better calibration/robustness more than entirely new problem-solving capabilities.
- ArXiv usage
  - The finding that arXiv alone is ineffective (Tables 8–9) may depend on model scale, integration with other data, or tasks not evaluated (Section 5.1.2).

## 7. Implications and Future Directions
- Field impact
  - A 7B open model reaching >50% on MATH (Table 5) resets expectations for the parameter count required for competitive math reasoning. The work also establishes scalable web data mining and critic‑free RL as practical tools for domain specialization.

- Research enabled/suggested (Section 5.2.3, Section 6)
  - Data
    - Extend the mining pipeline to other domains (e.g., scientific reasoning, formal methods) and languages; refine coverage for geometry and theorem proving.
    - Explore hybrid corpora where arXiv complements curated web/math code with better integration.
  - Algorithms
    - Robust RL under noisy rewards; integrate uncertainty estimates from reward models.
    - Explore stronger online sampling/decoding (e.g., tree-of-thoughts) and efficient inference for exploration (speculative decoding, memory-optimized serving).
    - Iterative co-training of policy and reward models at scale; step-aware (process) rewards learned from cheaper supervision.
  - Evaluations and applications
    - Broaden to multi-modal math (diagrams), symbolic reasoning with tool stacks, and real classroom/tutoring settings.
    - Use GRPO-style alignment for other structured domains (law, finance) where relative judgments and process feedback are available.

> Headline result: Table 5 shows `DeepSeekMath-RL 7B` reaches 51.7% Top‑1 on MATH with chain‑of‑thought reasoning and 58.8% with tool‑integrated reasoning, beating all open models from 7B to 70B and most closed models except GPT‑4 and Gemini Ultra.

> Data quality and scale: Figure 3 and Table 1 demonstrate that models trained on the 120B‑token DeepSeekMath corpus learn faster and longer than those trained on OpenWebMath (13.6B) or Proof‑Pile‑2 (51.9B), with large gains on both English and Chinese benchmarks.

> Methodological advance: Figure 4 and Equations (3)–(4) show how GRPO removes the critic by using group-relative normalized rewards and a direct KL penalty, reducing memory while improving over Online RFT (Figure 5) and benefiting further from iterative updates (Figure 6).
