# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

**ArXiv:** [2509.25760](https://arxiv.org/abs/2509.25760)

## 🎯 Pitch

TruthRL introduces a new reinforcement learning framework that directly optimizes the truthfulness of large language models by rewarding correct answers, neutrally allowing explicit abstentions ('I don’t know'), and penalizing hallucinated (incorrect) outputs through a simple ternary reward system. This approach not only curbs dangerous overconfidence and misinformation but also significantly improves reliability and trustworthiness—particularly vital for high-stakes applications where confidently wrong answers can have severe consequences. Extensive benchmarks demonstrate TruthRL reduces hallucinations by nearly 29% and boosts truthfulness over 21%, showcasing its impact in driving the next generation of more honest, uncertainty-aware AI assistants.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces TruthRL, an online reinforcement learning framework that trains large language models (LLMs) to be truthful by rewarding three distinct outcomes: correct answers, explicit abstentions (e.g., “I don’t know”), and penalizing hallucinations. Using a simple ternary reward within GRPO (a PPO-style algorithm), the method substantially reduces hallucinations and improves a truthfulness metric across multiple datasets and model sizes, especially in retrieval-augmented settings (Table 1).

## 2. Context and Motivation
- Problem addressed
  - LLMs often produce fluent but incorrect statements—“hallucinations”—instead of acknowledging uncertainty when they lack knowledge or the input is misleading. The work targets not just higher accuracy but calibrated behavior: answer correctly when possible, abstain when uncertain, and avoid wrong confident claims (Introduction; Section 2.1).
- Why this matters
  - In high-stakes domains (law, medicine), wrong but confident outputs can be dangerous; abstention can be preferable to misinformation (Introduction, citing medical and legal impacts).
- Shortcomings of prior approaches
  - Accuracy-only training (standard SFT; “vanilla RL” with binary rewards) implicitly incentivizes guessing because abstentions and wrong answers receive the same penalty; this suppresses uncertainty expression and can increase hallucinations (Section 2.2; Figure 2 shows SFT/RL nearly eliminate abstention and raise hallucination when k grows).
  - Uncertainty training via curated “I don’t know” data (e.g., R-Tuning) requires special datasets, tends to over-abstain, and may lose accuracy (Introduction; Sections 2.2–2.3; Table 1: R-Tuning lowers hallucination but often sacrifices accuracy and overall truthfulness).
  - Retrieval augmentation (RAG) helps but retrieved documents can be noisy or wrong, which can mislead models (Introduction).
- Positioning
  - The paper reframes the objective from “maximize accuracy” to “maximize truthfulness,” a composite that rewards correctness, tolerates abstention, and penalizes hallucination (Section 2.1). TruthRL is an online RL method that operationalizes this idea with a ternary reward, showing that a simple reward structure can outperform both accuracy-driven training and more complex reward variants (Sections 3.2, 4.2, 4.4).

## 3. Technical Approach
Step-by-step overview

1) Problem formulation (Section 2.1)
- Truthfulness is defined as a weighted combination of three rates computed over a dataset D:
  - Accuracy (`Acc`): fraction of correct answers.
  - Uncertainty (`Unc`): fraction of explicit abstentions (e.g., “I don’t know”).
  - Hallucination (`Hall`): fraction of factually incorrect answers.
- Truthfulness score = `w1·Acc + w2·Unc − w3·Hall`, `w1, w2, w3 ≥ 0`. Experiments set `w1=1, w2=0, w3=1` (so the score equals Accuracy − Hallucination; Section 4.1).
- Objective: train a model `fθ` to maximize expected truthfulness on D.

2) Why vanilla fine-tuning fails (Sections 2.2–2.3; Figure 2)
- SFT optimizes likelihood of ground-truth answers; it pushes models to always answer and memorize training distributions.
- Vanilla RL with binary rewards (“correct=+1, otherwise −1”) conflates abstention with error. In both SFT and binary-RL, the expected payoff for guessing is better than abstaining; hence uncertainty is suppressed. Figure 2 shows:
  - Prompting only: majority@k sampling reduces hallucination and increases abstention as k increases (Figure 2a).
  - After SFT or vanilla RL: abstention nearly vanishes and hallucination rises with k (Figures 2b–2c).

3) Knowledge boundary probing (Section 3.1)
- Purpose: identify “out-of-knowledge” (OOK) questions where the model cannot answer reliably.
- Procedure: sample 256 model responses per training question; if none are correct, mark the question OOK.
- Uses:
  - For baselines like R-Tuning, relabel OOK questions with “I don’t know” as ground truth.
  - Optionally used to enrich TruthRL rewards (“knowledge-enhanced” variant; Section 3.2).

4) TruthRL with GRPO (Section 3.2)
- GRPO (Group Relative Policy Optimization): an online RL method akin to PPO but with per-prompt group normalization.
  - For each prompt x, sample a group of G responses from the old policy `πθ_old`. Compute a reward for each response, then standardize each response’s reward by subtracting the mean reward within the group and dividing by the group standard deviation (z-score). This yields the advantage `Â_i` used in a clipped PPO-style objective with a KL penalty to a reference model (equation in Section 3.2).
  - Intuition: standardizing within the group emphasizes relative quality among sampled responses for the same prompt, sharpening the learning signal even with sparse rewards.
- Ternary reward design (Section 3.2)
  - `+1` if the answer is correct.
  - `0` if the answer is uncertain/abstention (“I don’t know”).
  - `−1` if the answer is incorrect (hallucination).
  - Why it matters: in a group with an abstention (`0`) and an incorrect answer (`−1`), the abstention has higher relative advantage after z-scoring, so the policy learns to abstain rather than hallucinate when unsure. Under binary reward, both would receive `−1`, failing to distinguish them (Section 3.2).
- Reward enhancements (Section 3.2; ablated in Section 4.4)
  - Knowledge-enhanced: treat abstention as positive (`+1`) on OOK questions and penalize non-abstaining errors more strongly; on non-OOK, reward correctness and penalize errors, optionally neutral on abstention.
  - Reasoning-enhanced: add extra reward based on a judge’s score of reasoning quality (Section 4.6; Table 8).
- Verifier (“LLM-as-a-judge”) for rewards (Sections 4.5, Appendix B)
  - Rewards are computed by a large LLM evaluator judging whether an answer is correct (Table 11). The paper also tests a rule-based string-matching verifier and shows it leads to degenerate behavior (Table 5).

5) Implementation and experimental design (Appendix A; Section 4.1)
- Datasets: CRAG, NaturalQuestions (NQ), HotpotQA, MuSiQue (Section 4.1).
- Settings: with and without retrieval. For CRAG, up to 50 retrieved web pages per question. For NQ/HotpotQA/MuSiQue, use 2018 Wikipedia and an E5 retriever (Appendix A).
- Backbones: `Llama3.1-8B-Instruct`, `Qwen2.5-7B-Instruct`, plus scale study from 3B to 32B (Section 4.1; Table 7).
- Baselines: prompting, SFT, RFT, R-Tuning, and “TruthRLBinary” (GRPO with binary reward). Also compare to offline/semi-online preference learning (DPO, iterative DPO; Section 4.4; Table 4).
- Training: online RL with VeRL/GRPO, KL regularization to a reference model, long context (16k–32k tokens), and greedy decoding at evaluation (Appendix A).

Analogy for the key idea
- Think of each prompt as a quiz where the model can either answer correctly, answer incorrectly, or admit not knowing. Traditional training treats “admit not knowing” the same as “wrong,” so the model always guesses. TruthRL changes the rules: you win for being right, lose for being wrong, and neither win nor lose for admitting uncertainty. Over many quizzes, the model learns when it genuinely knows the answer and when it should refrain.

## 4. Key Insights and Innovations
- A truthfulness-driven objective, not accuracy-only (Section 2.1)
  - Novelty: explicitly formalizes truthfulness as a multi-dimensional target and trains to optimize it, acknowledging abstention as a first-class outcome.
  - Significance: shifts incentive structure so models learn calibrated behavior rather than guesswork.
- Simple ternary reward beats both binary and more complex rewards (Sections 3.2, 4.4; Table 3; Figure 4)
  - Difference from prior work: standard RL with binary verifiable rewards (RLVR) conflates abstention and error; TruthRL separates them. The paper also tests knowledge- and reasoning-enhanced variants and finds the plain ternary reward to be the most robust on average (Table 3).
  - Impact: large reductions in hallucination and higher truthfulness without complex reward engineering.
- Online GRPO is better suited than offline/semi-online preference optimization for this goal (Section 4.4; Table 4)
  - Finding: DPO and iterative DPO show limited or unstable gains in truthfulness, while online GRPO consistently reduces hallucination and improves the composite score.
  - Why: online learning with per-prompt group normalization provides timely, context-relative signal and avoids dataset staleness in preference data.
- Improved knowledge boundary recognition without over-conservatism (Section 4.3; Figure 3)
  - Insight: the model learns to abstain primarily on hard/OOK questions, while maintaining or improving accuracy on easier ones. On a “difficult subset” where almost nobody is correct, TruthRL keeps hallucination minimal and abstains appropriately (Figure 3b).
- Robustness across judges and scales (Section 4.5; Tables 6–7)
  - Behavior generalizes across multiple evaluator LLMs and improves models from 3B to 32B parameters. This suggests the approach is not just exploiting a single judge’s idiosyncrasies and scales to stronger backbones.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1; Appendix A)
  - Datasets: CRAG (comprehensive RAG benchmark), NQ, HotpotQA, MuSiQue.
  - Two regimes: without retrieval (parametric knowledge only) and with retrieval (web pages or Wikipedia; Appendix A).
  - Metrics: Truthfulness `= Acc − Hall` with `w1=1, w2=0, w3=1`; plus reporting of hallucination rate and accuracy individually (Section 4.1).
  - Baselines: prompting, SFT, RFT, R-Tuning, vanilla RL via `TruthRLBinary`; additional comparisons to DPO variants (Table 4).
  - Reward verification: LLM judge; also a rule-based verifier ablation (Table 5).

- Main quantitative results (Table 1; Section 4.2)
  - With retrieval, `Llama3.1-8B-Instruct`:
    - TruthRL achieves T=37.2, H=19.4, A=56.6; the strongest baseline `TruthRLBinary` gets higher accuracy (A=60.3) but much higher hallucination (H=39.5), resulting in lower truthfulness (T=20.8).
    - Compared to prompting (T=5.3, H=43.5), TruthRL reduces hallucination by 24.1 percentage points and raises truthfulness by 31.9 points.
  - With retrieval, `Qwen2.5-7B-Instruct`:
    - TruthRL: T=33.1, H=17.3, A=50.4 vs prompting: T=10.6, H=38.4, A=49.0; and vs RFT: T=22.6, H=31.4, A=54.0 (TruthRL has far lower Hall with modest A).
  - Without retrieval:
    - `Llama3.1-8B-Instruct`: TruthRL: T=22.4, H=16.3, A=38.7 vs prompting: T=−4.4, H=44.5, A=40.1. While accuracy is slightly lower than prompting, the large drop in hallucination drives positive truthfulness.
    - `Qwen2.5-7B-Instruct`: TruthRL: T=16.2, H=8.7, A=24.9; again, lowest hallucination and positive truthfulness where baselines are negative.

- Do experiments support the claims?
  - Reduction in hallucinations with maintained or improved accuracy in many settings is consistent across datasets and backbones (Table 1). Gains are largest in retrieval setups, where spurious documents otherwise amplify hallucinations.
  - Figure 3 decomposes behavior on CRAG: on the full set, TruthRL has the lowest hallucination and the highest uncertainty rate while keeping competitive accuracy (Figure 3a). On the difficult subset where models rarely know the answer, TruthRL abstains 84.5% and hallucinates only 15.5%, while high-accuracy methods like SFT or binary-RL hallucinate nearly 100% (Figure 3b).
  - Table 2 shows robustness to “hallucination-baiting” multiple-choice questions: TruthRL attains T=52.4 with H=16.5 and U=14.6, while baselines have far higher hallucination rates.
  - Ablations (Table 3; Figure 4) show ternary reward produces the best overall truthfulness and the lowest hallucination; binary reward maximizes accuracy but collapses uncertainty to near zero.
  - Preference learning comparisons (Table 4) indicate online GRPO outperforms DPO and iterative DPO in the truthfulness metric across four datasets.
  - Confidence calibration analysis (Figure 5) suggests TruthRL reduces overconfident hallucinations by shifting high-confidence outputs toward correctness or abstention.
  - Verifier ablation (Table 5) highlights that a rule-based judge causes degenerate over-abstention (very low hallucination but negative truthfulness), while an LLM judge yields balanced behavior and high truthfulness.
  - Cross-judge robustness (Table 6) and scale-up study (Table 7) reinforce generality.

- Notable nuances and trade-offs
  - Binary reward (“vanilla RL”) can deliver the highest accuracy but at the cost of high hallucination and near-zero abstention (Table 1, Llama with retrieval: A=60.3, H=39.5).
  - Reasoning reward heuristics (Table 8) modestly change reasoning scores but can hurt outcome truthfulness; the simplest outcome-only ternary reward remains best for the main objective.

> From Table 1 (Llama3.1-8B, with retrieval): “TruthRL 37.2/19.4/56.6 vs TruthRLBinary 20.8/39.5/60.3,” showing the ternary design halves hallucination while retaining high accuracy.

> Figure 3b: On the hardest CRAG questions, TruthRL hallucinates only 15.5% and abstains 84.5%, whereas SFT and binary RL approach 100% hallucination.

> Table 4: “TruthRL average T/H = 25.6/18.8,” outperforming DPO (−10.1/51.1) and best Iterative DPO (Iter 3: 12.6/31.7).

## 6. Limitations and Trade-offs
- Dependence on an LLM judge for reward signals (Section 4.5; Table 6)
  - While robustness to multiple judges is shown, reward quality still hinges on the evaluator’s reliability and potential biases. A poor or mismatched judge can produce degenerate policies (Table 5 shows rule-based string matching leads to over-abstention and negative truthfulness).
- Metric design places zero weight on abstention (Section 4.1 sets `w2=0`)
  - The truthfulness metric used in reporting is `Acc − Hall`. Although the training reward treats abstention neutrally (and knowledge-enhanced variants can reward it on OOK items), the main metric does not directly credit abstention. This may understate the utility of calibrated “I don’t know” behavior in some applications.
- OOK detection relies on self-probing (Section 3.1)
  - Marking questions as OOK by sampling 256 responses may misclassify some questions, especially if the base model is weak. Errors in OOK labeling can affect baselines like R-Tuning and knowledge-enhanced reward variants.
- Computational cost and system complexity (Appendix A)
  - Online RL with long contexts (up to 16–32k tokens), retrieval of up to 50 documents, and group sampling increases compute and engineering complexity (DeepSpeed ZeRO-3, FlashAttention-2, vLLM serving).
- Reasoning reward remains heuristic (Section 4.6; Table 8)
  - Attempts to integrate a reasoning-quality reward yield mixed outcomes; simple outcome-only rewards already improve reasoning somewhat, but principled multi-objective optimization for reasoning remains open.
- Not explored: adversarial reward hacking
  - While cross-judge tests reduce concern, the work does not deeply analyze whether the policy exploits specific judge weaknesses over long training horizons.

## 7. Implications and Future Directions
- Field impact
  - The work reframes post-training objectives for LLMs around truthfulness rather than accuracy alone, showing that a minimal change—a ternary reward—shifts behavior toward calibrated abstention and away from overconfident errors (Sections 3.2, 4.2–4.4). This provides a practical recipe for safer, more trustworthy LLMs in knowledge-intensive tasks.
- Practical applications
  - High-stakes decision support (medical triage, legal research), enterprise question answering, customer support chatbots, and search assistants—any setting where “I don’t know” is safer than a plausible but wrong response—especially with retrieval pipelines that can introduce noisy evidence (Table 1; Figure 3; Section 4.3).
- Research directions
  - Reward design:
    - Calibrated metrics that directly value abstention (e.g., nonzero `w2`) and task-specific trade-offs; incorporating cost-sensitive or coverage-aware objectives.
    - More principled reasoning rewards or verifiable intermediate checks (Section 4.6 shows naive heuristics can backfire; Table 8).
  - Verifier quality:
    - Hybrid verifiers combining semantic LLM judgment with rule-based constraints and external knowledge bases; methods to detect and prevent reward hacking (Section 4.5).
  - Data and retrieval:
    - Active selection of OOK or adversarial queries; robust retrieval that detects and discounts misleading sources, integrated with truthfulness-aware training (Introduction; Section 4.2).
  - Algorithms:
    - Explore other online RL variants and groupwise normalization schemes; theoretical analyses of when ternary vs more granular rewards (e.g., partial credit) optimize truthfulness.
  - Deployment:
    - Calibration and user interfaces for abstention, including asking clarifying questions vs declining; human-in-the-loop triage policies based on confidence distributions (Figure 5).

In sum, TruthRL demonstrates that a simple, well-targeted reward structure within an online RL setup can materially change LLM behavior toward safer, more truthful outputs, without elaborate reward engineering. The extensive ablations, cross-judge tests, and scale studies (Tables 1–8; Figures 2–5) provide strong empirical grounding and point to a broader design principle: explicitly separate “don’t know” from “wrong” in both objectives and training signals.
