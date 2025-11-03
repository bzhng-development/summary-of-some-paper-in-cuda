# DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

**ArXiv:** [2402.03300](https://arxiv.org/abs/2402.03300)

## 🎯 Pitch

DeepSeekMath introduces a 7B-parameter open-source language model that sets a new benchmark for mathematical reasoning by pretraining on a colossal, high-quality 120B-token math corpus mined from the web, and pioneering the lightweight Group Relative Policy Optimization (GRPO) algorithm for reinforcement learning. This approach closes much of the gap with leading proprietary AI models on rigorous math challenges, while GRPO delivers substantial gains in accuracy with dramatically reduced memory requirements compared to PPO—making state-of-the-art mathematical reasoning both accessible and efficient for the research community.

---

## 1. Executive Summary (2-3 sentences)
DeepSeekMath introduces a 7B-parameter open model that markedly improves mathematical reasoning by: (a) pretraining on a newly built 120B‑token math corpus mined from Common Crawl, and (b) applying a lightweight reinforcement learning algorithm, Group Relative Policy Optimization (`GRPO`). The model reaches 51.7% Top-1 on the competition-level MATH benchmark without tool use and 58.8% with tool integration (Table 5), while `GRPO` improves accuracy using far less memory than standard PPO by removing the critic/value model (Section 4.1, Figure 4).

## 2. Context and Motivation
- Problem/gap:
  - Open models lag significantly behind top proprietary systems (GPT‑4, Gemini‑Ultra) on rigorous math benchmarks (Section 1; Figure 1). The gap stems from limited high-quality math pretraining data and costly/unstable reinforcement learning pipelines that require large critic models (PPO).
- Importance:
  - Better math reasoning benefits education, STEM problem solving, software verification, and formal methods. It also serves as a proxy for general multi-step reasoning (Section 1).
- Prior approaches and shortcomings:
  - Minerva (closed, 540B params) and Llemma (open) use math-focused corpora such as OpenWebMath, Proof-Pile-2, arXiv. However, open models underperform, and prior corpora are smaller or English-centric (Sections 2.2, 2.3; Table 1).
  - Reinforcement learning with PPO boosts reasoning but doubles memory with a value model and per-token KL-penalized rewards (Equation 2), complicating training (Section 4.1.1).
- Positioning:
  - The paper tackles both data and training: it builds a much larger multilingual math corpus via an iterative web-mining pipeline (Figure 2; Section 2.1) and proposes `GRPO`, a PPO variant that removes the critic by using group-relative rewards as a baseline (Equations 3–4; Figure 4). It also examines when code pretraining helps math and questions the value of arXiv-only text (Sections 5.1.1–5.1.2).

## 3. Technical Approach
The system pipeline has four stages: math data collection, base model pretraining, supervised instruction tuning, and reinforcement learning with `GRPO`.

1) Math data collection (Section 2.1; Figure 2)
- Goal: Mine high-quality math text from the web at scale.
- Steps:
  - Deduplicate Common Crawl to 40B HTML pages (URL and near-duplicate filtering).
  - Train a fastText classifier on a small “seed” math corpus (initially OpenWebMath positives vs random negatives) to score pages for “math-likeness.”
  - Iterative recall and expansion:
    - Rank CC pages by classifier score; keep top-scoring tokens (first 40B, then extended to 120B based on pretraining gains).
    - Discover math-related domains by computing, per domain, the fraction of pages recalled; if >10% are recalled, treat the domain as math-related (e.g., mathoverflow.net). Human labelers mark math-specific URL paths (e.g., /questions), add remaining pages to the positive pool, retrain the classifier, and repeat.
  - After four iterations: 35.5M pages totaling 120B tokens, multilingual (English and Chinese prominent).
  - Decontamination: Remove pages with any exact 10‑gram overlap with benchmark text (GSM8K, MATH, CMATH, AGIEval); for 3–9 grams, exact match removal (end of Section 2.1).

2) Base model pretraining (Section 2.3)
- Start from `DeepSeek-Coder-Base-v1.5 7B` (code-specialized base).
- Train on 500B tokens with the following mixture: 56% `DeepSeekMath Corpus` (web math), 4% `AlgebraicStack` (math code), 10% arXiv, 20% GitHub code, 10% general English/Chinese web text. Standard AdamW schedule with 10M-token batch and 4K context (Sections 2.2.1, 2.3).
- Rationale: initialize from a code-pretrained model to leverage structured reasoning skills; keep some code during continued training to retain coding ability (Table 4; Section 5.1.1).

3) Supervised fine-tuning (SFT) for instruction following (Section 3)
- Data: 776K problems in English and Chinese with solution traces in three formats:
  - `CoT` (Chain-of-Thought): step-by-step natural language reasoning.
  - `PoT` (Program-of-Thought): reasoning expressed as Python code.
  - `Tool-integrated reasoning`: interleaving language and tool use.
- Sources include GSM8K, MATH (annotated with tool-integrated solutions), subsets of MathInstruct and Lila-OOD, and a broad K‑12 Chinese set covering 76 subtopics (Section 3.1).
- Training: concatenate examples up to 4K tokens; 500 steps, batch 256, LR 5e‑5 (Section 3.2).

4) Reinforcement learning with `GRPO` (Section 4)
- Motivation: PPO needs a critic/value model and per-token KL-penalized rewards, which are memory-heavy and unstable when only the final token is rewarded. `GRPO` avoids the critic and uses group-relative baselines.
- Core mechanism (Sections 4.1.1–4.1.3; Figure 4; Equations 1–4):
  - For each question `q`, sample `G` outputs `{o1 … oG}` from the current policy.
  - Score each output with a reward model; normalize the rewards within the group (subtract group mean, divide by group std).
  - Compute advantages `Â_{i,t}` as:
    - Outcome supervision: set `Â_{i,t}` to the normalized final reward for all tokens of output `i`.
    - Process supervision: score intermediate reasoning steps; set `Â_{i,t}` to the sum of normalized future step rewards past position `t`.
  - Optimize the clipped PPO-like objective but:
    - Replace the value-function baseline with group-normalized rewards (no critic to train).
    - Regularize with a direct KL term to a frozen reference policy rather than adding KL to the reward. Use an unbiased per-token estimator of KL (Equation 4), and include it as `−β D_KL(πθ || π_ref)` in the objective (Equation 3).
- Iterative RL (Algorithm 1; Section 4.1.4):
  - Periodically set the reference model to the current policy, resample data, and continue RL.
  - Continually train the reward model using replay that includes ~10% historical comparisons, improving supervision as the policy evolves.
- RL training setup (Section 4.2):
  - Policy: `DeepSeekMath-Instruct 7B`.
  - Data: only CoT-format questions from GSM8K and MATH (~144K); out-of-domain generalization is assessed on other benchmarks.
  - Hyperparameters: LR 1e‑6 for policy; KL coefficient 0.04; `G = 64` samples/question; max length 1024; batch size 1024; one policy update per exploration round.

Definitions used above:
- `Advantage`: how much better a token/action is than a baseline; in PPO it comes from a critic (GAE). In `GRPO`, the baseline is the group average reward.
- `Outcome vs. Process supervision`: reward only at the final answer vs. reward at intermediate reasoning steps.
- `Maj@K` vs `Pass@K` (Figure 7): `Maj@K` picks the majority answer among `K` samples; `Pass@K` checks if any of the `K` samples is correct.

## 4. Key Insights and Innovations
- Large-scale, multilingual math web corpus (Sections 2.1–2.2; Figure 2, Table 1, Figure 3)
  - What’s new: an iterative “domain discovery + path annotation + retraining” pipeline turns Common Crawl into 120B math tokens—several times bigger than prior sets (OpenWebMath 13.6B; Proof-Pile‑2 51.9B).
  - Why it matters: Pretraining a 1.3B model on this corpus outperforms the same model trained on MathPile, OpenWebMath, or Proof‑Pile‑2 across 8 math benchmarks, especially in Chinese (Table 1). The learning curve remains steep and does not plateau early (Figure 3).
- `GRPO`: critic-free PPO variant with group-relative baseline (Section 4.1; Figure 4; Equations 3–4)
  - What’s new: replaces the value model with group-normalized rewards, aligning naturally with comparison-trained reward models and cutting memory use substantially.
  - Why it matters: Improves over strong SFT baselines across in-domain (GSM8K, MATH) and out-of-domain tasks, without extra SFT data (Table 5). Ablations show online sampling and process supervision help (Figure 5).
- Code-before-math curriculum and mixed training (Section 5.1.1; Tables 6–7)
  - What’s new: systematic study on when code pretraining helps math reasoning and tool-use math.
  - Why it matters: Code training boosts both Python-aided math solving and, to a lesser extent, natural-language-only math reasoning. Mixing code and math in one-stage training mitigates catastrophic forgetting and yields strong coding + math-tool-use performance (Tables 6–7).
- ArXiv-only math text appears ineffective in this setting (Section 5.1.2; Tables 8–9)
  - What’s new: controlled experiments at 1.3B and 7B scales on two arXiv corpora show little or negative benefit for math reasoning, multiple-choice STEM, and miniF2F formalization.
  - Why it matters: Challenges a common assumption; suggests web math with the proposed filtering pipeline may be a better use of tokens for this goal. The paper is careful to state caveats (e.g., perhaps benefits emerge at larger scales or for other tasks).

## 5. Experimental Analysis
- Evaluation methodology (Sections 2.3, 3.2, 4.2)
  - Benchmarks:
    - English: GSM8K, MATH, SAT, OCW Courses, MMLU‑STEM, BBH; coding: HumanEval, MBPP; formal math: miniF2F (Isabelle).
    - Chinese: MGSM-zh, CMATH, Gaokao-MathCloze, Gaokao-MathQA.
  - Settings:
    - Without tools: CoT prompting.
    - With tools: PoT prompting that executes Python (math, sympy).
    - Formalization: informal-to-formal proofs with Sledgehammer filling details (Table 3).
    - Some results are Top‑1; grayed entries indicate majority voting over 32 samples (Table 5 notes).
- Main results
  - Base model strength (no instruction tuning) (Table 2; Table 3; Table 4):
    - Without tools: `DeepSeekMath-Base 7B` achieves 64.2% (GSM8K) and 36.2% (MATH), surpassing all open-source base models and even Minerva 540B on MATH (33.6%).
    - With tools: 66.9% (GSM8K+Python) and 31.4% (MATH+Python), beating Llemma 34B (64.6% and 26.3%) (Table 3).
    - Formalization: 25.8%/24.6% on miniF2F-valid/test—strong few-shot autoformalization (Table 3).
    - General abilities: math pretraining also lifts MMLU (54.9%) and BBH (59.5%) relative to DeepSeek‑Coder‑Base‑v1.5 (Table 4), while retaining most coding skill.
  - Instruction tuning and RL (Table 5):
    - `DeepSeekMath-Instruct 7B` (no tools): 82.9% (GSM8K), 46.8% (MATH); with tools: 83.7% (GSM8K+Python), 57.4% (MATH+Python). It already beats many larger open models.
    - `DeepSeekMath-RL 7B` (GRPO, CoT-only RL data): further rises to 88.2% (GSM8K) and 51.7% (MATH) without tools; with tools: 86.7% and 58.8%. It also improves Chinese benchmarks (e.g., CMATH 88.8% vs 84.6% for SFT).
- Do the experiments support claims?
  - Data quality: Table 1 and Figure 3 convincingly show the DeepSeekMath corpus yields stronger and more sustained gains than MathPile/OpenWebMath/Proof‑Pile‑2 at 1.3B scale, and enhances Chinese tasks—evidence for both quality and multilingual coverage.
  - RL effectiveness: Table 5 shows across-the-board improvements from SFT to RL, even on out-of-domain benchmarks where no RL data is used. Figure 5 shows online RFT > offline RFT, and `GRPO` > online RFT, supporting the algorithmic choices. Figure 6 shows iterative RL improves further.
  - Nature of RL gains: Figure 7 indicates RL raises `Maj@K` much more than `Pass@K`, implying RL aligns the output distribution (more consistency) rather than discovering new solutions—an important diagnostic.
- Ablations and robustness
  - Training curricula (Tables 6–7): code→math vs general→math; mixed one-stage vs two-stage; with/without tools.
  - ArXiv ablations (Tables 8–9): two different arXiv corpora at two model sizes; no noted gains.
  - Contamination control: 10‑gram (and 3–9 gram) exact-match filters (Section 2.1). This is a strong but not foolproof measure; acknowledged in discussion.
  - Reward model quality: trained from DeepSeekMath-Base; continuously improved with replay (Algorithm 1), but noise remains a known issue (Section 5.2.3).

> Highlight result: “DeepSeekMath-RL 7B attains 88.2% (GSM8K) and 51.7% (MATH) using chain-of-thought reasoning… surpasses all open-source models from 7B to 70B, as well as the majority of closed-source models.” (Table 5)

## 6. Limitations and Trade-offs
- Assumptions and supervision:
  - Reward models trained from comparison data can be noisy; process rewards rely on step segmentation quality (Section 4.1.3; Section 5.2.3). The paper notes ~20% noisy annotations in PRM800K as context for the general problem, highlighting the need for robust RL against noisy rewards.
- Compute and sampling:
  - Although `GRPO` removes the critic, it samples `G=64` outputs per question during exploration (Section 4.2), which is computationally heavy at inference time during training. Memory is reduced versus PPO, but wall-clock cost can still be high.
- Data bias and coverage:
  - Despite multilinguality, geometry and theorem-proving still lag behind top closed models; qualitative failures include triangle/ellipse problems (Section 6). This suggests coverage bias in the web-mined corpus and SFT sets.
- Scale and few-shot:
  - The 7B model underperforms GPT‑4/Gemini‑Ultra and gains little from few-shot prompting compared to them (Section 6).
- Decontamination limits:
  - n‑gram filtering prevents direct leakage but may miss paraphrased overlaps; also removes some legitimate content if it happens to share 10-grams (Section 2.1).
- ArXiv conclusion scope:
  - The “arXiv seems ineffective” finding is limited to the tested sizes, corpora, and tasks; benefits might emerge at larger scale or in other math tasks (Section 5.1.2 caveats).

## 7. Implications and Future Directions
- Field impact:
  - Demonstrates that data curation plus lightweight RL can push a 7B open model past many larger systems on MATH, lowering the barrier to strong math reasoning. The iterative CC mining pipeline is reusable for other domains (e.g., code) and languages (Sections 2.1–2.2).
  - `GRPO` shows that critic-free, group-relative RL is an effective, resource-friendly alternative to PPO for reasoning tasks (Section 4, Figures 4–6).
- Follow-up research enabled/suggested:
  - Data:
    - Expand the iterative mining pipeline to explicitly target underrepresented math areas (e.g., geometry), more languages, and higher-quality formal math.
    - Study combined recipes: web math + arXiv + textbooks, and scale sensitivity of arXiv benefits (Sections 5.1.2, 6).
  - Algorithms:
    - Robust RL under noisy rewards; quantify reward uncertainty and integrate it into training (Section 5.2.3).
    - Better exploration via tree-search decoding (e.g., Tree-of-Thoughts) and efficient inference (speculative decoding, paged attention) to reduce the cost of `G`-sample exploration (Section 5.2.3).
    - More capable and generalizable process reward models with calibrated uncertainty.
  - Understanding RL gains:
    - Since RL mainly improves `Maj@K` (Figure 7), develop methods that also raise `Pass@K` (true capability), e.g., mixture-of-training-signals from tools, self-debugging, or verifier-guided exploration.
- Practical applications:
  - Education assistants that produce step-by-step math solutions with high reliability; automated graders/verifiers; math research aides; integration with theorem provers (Isabelle evaluations, Table 3); code-and-math co-reasoning tools for engineering and data science.

Overall, DeepSeekMath shows that scaling high-quality web math data and adopting a critic-free, group-relative RL algorithm can substantially elevate open 7B models in mathematical reasoning, while revealing nuanced lessons about code training, arXiv data, and the nature of RL improvements.
