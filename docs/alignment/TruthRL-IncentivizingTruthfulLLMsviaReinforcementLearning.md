# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

**ArXiv:** [2509.25760](https://arxiv.org/abs/2509.25760)

## 🎯 Pitch

TruthRL introduces a new reinforcement learning framework for large language models that directly rewards not only accurate answers, but also proper abstention when uncertain, using a principled ternary reward system. By explicitly distinguishing between correct, incorrect, and uncertain responses, TruthRL dramatically reduces harmful hallucinations and improves model trustworthiness, especially in high-stakes domains. This shift from accuracy-only optimization to truthfulness-driven training sets a new standard for building reliable and responsible AI systems.

---

## 1. Executive Summary
TruthRL introduces a reinforcement-learning framework that directly optimizes an LLM’s truthfulness by rewarding three distinct outcomes: “correct,” “uncertain/abstain,” and “incorrect.” Using Group Relative Policy Optimization (GRPO) with a ternary reward, it reduces hallucinations and encourages abstention when the model is unsure, improving reliability across multiple QA benchmarks and model sizes.

The approach matters because standard fine-tuning or RL methods optimized only for accuracy tend to incentivize guessing over calibrated abstention, which increases harmful hallucinations. TruthRL reverses that incentive and demonstrates consistent truthfulness gains in both retrieval and non-retrieval settings (Table 1).

## 2. Context and Motivation
- Problem/gap addressed
  - LLMs often produce fluent but false statements (“hallucinations”) when questions exceed their knowledge. The key gap is that optimizing just for accuracy does not teach models to recognize uncertainty and abstain when appropriate, which is essential in high-stakes domains (Introduction; Section 2.1).
  - A truthful model should maximize correct answers, minimize hallucinations, and abstain when unsure. Prior accuracy-focused training suppresses abstention, leading to confident errors (Figure 2; Section 2.3).

- Why this is important
  - Real-world impact: In law and medicine, it’s often safer to say “I don’t know” than to produce a plausible but incorrect answer (Introduction).
  - Theoretical significance: The paper formalizes “truthfulness” as a multi-dimensional objective and shows that objective design fundamentally changes model behavior (Section 2.1; Section 3.2).

- Prior approaches and their limitations
  - Supervised fine-tuning (SFT) and “vanilla” RL (binary correct/incorrect rewards): improve accuracy but discourage abstention, often increasing hallucinations (Figure 2; Table 1).
  - RAG (retrieval-augmented generation): adds external information but can include noisy or wrong documents and still induce hallucinations (Introduction).
  - Methods to teach “I don’t know” (e.g., R-Tuning): require special data and risk over-conservatism (abstaining even when the model knows the answer) (Section 3.1; Table 1).

- Positioning
  - TruthRL reframes post-training as optimization for truthfulness (not just accuracy). It uses a simple but effective ternary reward design implemented with GRPO to positively differentiate abstention from hallucination (Section 3.2).

## 3. Technical Approach
This section explains how TruthRL works and why its design choices matter.

- Core definitions (Section 2.1)
  - `Accuracy (Acc)`: fraction of correctly answered questions.
  - `Uncertainty rate (Unc)`: fraction of questions where the model abstains, typically by saying “I don’t know.”
  - `Hallucination rate (Hall)`: fraction of answers that are factually incorrect.
  - `Truthfulness score`: a weighted combination Truthfulness = `w1·Acc + w2·Unc − w3·Hall`. For evaluation they set `w1 = 1, w2 = 0, w3 = 1`, i.e., Acc − Hall (Table 1 and Section 4.1). Training encourages abstention via the reward design (Section 3.2).

- Why binary rewards hurt abstention (Figure 2; Section 2.3)
  - With binary reward (`+1` correct, `−1` otherwise), abstentions get the same penalty as wrong answers. Models learn that guessing offers a non-zero chance at `+1`, while abstaining guarantees `−1`. This promotes guessing and suppresses uncertainty expression (Figure 2b–c vs 2a).

- TruthRL with GRPO (Section 3.2)
  - GRPO overview: For each input `x`, sample a group of `G` responses from the old policy. Compute a reward `r(x, yi)` for each response `yi`. The advantage `Âi` is the z-scored reward within the group (reward minus group mean divided by group std). Update the policy with a clipped objective and KL penalty to a reference policy (equation in Section 3.2).
  - Intuition: Because advantages are computed relative to peers sampled for the same prompt, the algorithm learns to prefer relatively better outcomes among the sampled responses.

- Ternary reward design (Section 3.2)
  - Reward scheme:
    - `+1` for correct,
    - `0` for uncertain/abstain,
    - `−1` for incorrect.
  - Why it works: Consider two responses in a group, `y1 = abstain` and `y2 = wrong`. Binary rewards give both `−1` → equal advantage (no preference). Ternary gives `0` vs `−1` → `y1` has higher relative advantage → the policy is nudged to abstain rather than hallucinate (explicit worked example in Section 3.2).

- Knowledge-enhanced and reasoning-enhanced variants (Section 3.2; Section 4.6)
  - Knowledge-enhanced: if a prompt is truly out-of-knowledge (OOK)—detected via “knowledge boundary probing,” described below—then abstentions receive positive reward (`+1`) to actively encourage them on OOK questions. On known questions, abstentions get `0` with ternary reward.
  - Reasoning-enhanced: adds a secondary signal that scores the chain-of-thought quality. Explored with multiplicative, additive, or conditional combinations with outcome reward (Table 8), but the simple ternary outcome-only reward generally works best.

- Knowledge boundary probing for baselines and reward variants (Section 3.1)
  - Detect whether a question is OOK by sampling 256 responses from a base model: if none is correct, label it OOK; then use that info to construct SFT baselines (e.g., R-Tuning with “I don’t know” as the label for OOK; RFT with traces that end with abstention on OOK) and optionally to adjust rewards (knowledge-enhanced variant).

- Training pipeline (Appendix A; Section 4.1)
  - Verifier: An LLM judge (Llama3.3‑70B‑Instruct by default) scores outcomes (correct/incorrect/uncertain). A rule-based string-matching verifier collapses the behavior toward over-abstention and worse truthfulness (Table 5), so an LLM judge is critical.
  - Models: Mainly `Llama3.1‑8B‑Instruct` and `Qwen2.5‑7B‑Instruct` as backbones (Table 1). Also smaller and larger models in Table 7.
  - Settings: Both without retrieval and with retrieval. For retrieval, up to 50 documents per question are provided (Appendix A).
  - Optimization: Online RL with GRPO, clipped objective and KL regularization; efficient rollouts with vLLM; full-parameter finetuning on 8×H100 GPUs (Appendix A).

- A simple analogy for the reward design
  - Think of a traffic signal for answers: green = correct (+1), yellow = honest uncertainty (0), red = wrong (−1). Under GRPO, the model learns to prefer yellow over red when it can’t get green, instead of running red lights.

## 4. Key Insights and Innovations
- Ternary reward that separates abstention from error (fundamental innovation)
  - Unlike binary rewards, the zero-reward for abstention makes it strictly better than an incorrect answer within GRPO’s relative-advantage framework (Section 3.2). This single change causes large drops in hallucination while preserving accuracy (Table 3; Table 1).

- Optimizing truthfulness as a primary objective (conceptual reframing)
  - The paper formalizes truthfulness (Acc, Unc, Hall) and trains toward it. This shifts alignment away from accuracy-only objectives that suppress uncertainty (Section 2.1; Figure 2). It is a simple but powerful reframing that changes model behavior.

- Demonstrated knowledge-boundary recognition without over-conservatism (new capability)
  - On difficult CRAG questions where almost no method answers correctly, TruthRL mostly abstains rather than hallucinate, achieving only 15.5% hallucination while being uncertain 84.5% (Figure 3b). Competing accuracy-driven methods can hallucinate nearly 100% on these hard cases.

- Simplicity beats complexity (practical insight)
  - Ternary outcome reward outperforms more complicated knowledge- and reasoning-enhanced variants on average (Table 3; Table 8). The straightforward design is easier to implement and robust across judges and model scales (Table 6; Table 7).

- Online RL with verifiable rewards outperforms offline/semi-online preference optimization (empirical insight)
  - GRPO-based online training consistently beats DPO and iterative DPO, which improve only modestly and can regress with more iterations (Table 4).

## 5. Experimental Analysis
- Evaluation methodology (Section 4.1; Appendix A)
  - Benchmarks: CRAG (comprehensive RAG benchmark), NaturalQuestions (NQ), HotpotQA, MuSiQue. Models are trained on CRAG and tested across all four datasets.
  - Settings: With and without retrieval (up to 50 web pages or Wikipedia docs; Appendix A).
  - Metrics: Truthfulness score (set to Acc − Hall, i.e., `w1=1, w2=0, w3=1`), hallucination rate, accuracy. Uncertainty rate is also reported in some analyses (Figure 3; Table 2).
  - Baselines: Prompting (no finetuning), SFT, RFT, R-Tuning, and “TruthRLBinary” (GRPO with binary reward; recovers vanilla RL). Additional comparisons with DPO and iterative DPO (Table 4).

- Main quantitative results (Table 1)
  - With retrieval, `Llama3.1-8B-Inst`:
    - CRAG: TruthRL achieves T=37.2, H=19.4, A=56.6, compared to Prompting T=5.3, H=43.5, A=48.8. That is, hallucination drops by 24.1 points with a 8-point accuracy gain.
    - Average over four datasets: TruthRL reaches T=25.6 and H=18.8, while Prompting has T=−16.4 and H=54.1. TruthRL’s truthfulness and hallucination are consistently better than SFT, RFT, R-Tuning, and TruthRLBinary.
  - Without retrieval, gains are smaller and can vary by dataset, but TruthRL still reduces hallucinations relative to baselines. For `Llama3.1-8B-Inst`, average H drops to 20.5 vs 75.2 for SFT and 63.3 for TruthRLBinary (Table 1).

- Behavioral breakdown (Figure 3)
  - On all CRAG questions with retrieval (Figure 3a), TruthRL attains the lowest hallucination among methods while maintaining competitive accuracy and the highest calibrated uncertainty.
  - On difficult questions (Figure 3b), TruthRL’s hallucination is 15.5% with uncertainty 84.5%, while SFT/TruthRLBinary hallucinate on nearly every attempt. This demonstrates calibrated boundary recognition.

- Robustness checks
  - Hallucination-baiting comparisons (Table 2): On multiple-choice-like CRAG questions that induce guessing, TruthRL has T=52.4 and H=16.5, outperforming RFT (T=12.7, H=38.8) and R-Tuning (T=6.8, H=43.7).
  - Ablations on reward design (Table 3; Figure 4): Ternary reward yields the best average truthfulness and the lowest hallucination. Binary rewards maximize accuracy but nearly eliminate abstention and inflate hallucination. Knowledge-enhanced variants help partially but still trail ternary on average.
  - Online vs offline/semi-online RL (Table 4): TruthRL (online GRPO) clearly outperforms DPO and iterative DPO.
  - Verifier choice (Table 5): A rule-based string-matcher causes collapse into excessive abstention (T=−3.6) despite very low hallucination (H=3.6). An LLM-based judge preserves both accuracy and low hallucination (CRAG T=37.2, H=19.4).
  - Judge diversity (Table 6): Gains are consistent across Llama3.3‑70B‑Instruct, Qwen2.5‑72B‑Instruct, and Gemma3‑27B‑Instruct as evaluators.
  - Model scale (Table 7): Improvements hold from 3B to 32B backbones; smaller models benefit proportionally more.

- Confidence analysis (Figure 5)
  - Outputs are grouped by confidence bins. TruthRL reduces overconfident hallucinations and increases the fraction of high-confidence correct answers, indicating better calibration.

- Do the experiments support the claims?
  - Yes. The central claim—simple ternary rewards in online RL reduce hallucination and foster calibrated abstention without sacrificing (and often improving) accuracy—is demonstrated repeatedly: overall results (Table 1), hard subsets (Figure 3b), baiting questions (Table 2), ablations (Table 3), training dynamics (Figure 4), confidence calibration (Figure 5), and robustness (Tables 5–7).

## 6. Limitations and Trade-offs
- Dependence on an LLM judge (Section 4.5; Table 5; Table 6)
  - Rewards come from an LLM verifier. While robust across three judges (Table 6), this introduces potential biases and costs. Rule-based judges fail (over-abstention), so high-quality model judges are necessary.
- Objective choice for evaluation vs training (Section 4.1)
  - The evaluation sets `w2=0` (truthfulness = Acc − Hall), so abstentions do not directly improve the reported score. The benefit of abstention appears indirectly by reducing hallucinations. In domains where abstention is preferred, different weights might be desired and could change conclusions.
- Retrieval quality and domain shifts (Introduction; Section 4.1)
  - Retrieval can include noisy evidence; the method still helps but performance varies across datasets, with weaker results on MuSiQue even with retrieval (Table 1; T=−0.9).
- Compute and engineering complexity (Appendix A)
  - Online RL with GRPO requires substantial infrastructure (8×H100 GPUs, vLLM rollouts, KL control, verifier calls). This is more complex and expensive than SFT or offline preference learning (DPO).
- OOK detection for knowledge-enhanced variants and baselines (Section 3.1)
  - Knowledge boundary probing uses 256 samples per question to infer OOK status. Although TruthRL’s main ternary setup does not require OOK labeling, baselines and enhanced variants do, which is resource-intensive and may be imperfect.

## 7. Implications and Future Directions
- Field-level impact
  - TruthRL shifts post-training from accuracy-only to truthfulness-oriented objectives, showing that reward structure (ternary vs binary) fundamentally changes LLM behavior. It offers a practical recipe—simple reward, online GRPO, LLM judge—that improves reliability across datasets, models, and retrieval settings.

- Practical applications
  - High-stakes QA, customer support, medical/legal triage, enterprise search assistants, and any RAG system where calibrated abstention reduces risk. TruthRL’s abstain-over-guess behavior is particularly valuable for safety-critical deployments.

- Follow-up research
  - Better verifiers: More reliable, cheaper, perhaps hybrid (LLM + rules) judges to maintain semantic awareness without collapse (Table 5).
  - Multi-objective RL: Explicitly incorporate uncertainty into the evaluation objective (nonzero `w2`) to reflect real-world preferences for abstention in risky contexts.
  - Reasoning rewards: The initial explorations (Table 8) show outcome-only rewards already improve reasoning scores; principled multi-signal designs could further enhance faithful reasoning without hurting outcomes.
  - Data-aware retrieval: Couple TruthRL with retrieval re-ranking and evidence quality estimation to further suppress hallucinations from noisy context.
  - Calibration metrics: Beyond Acc/Hall/Unc, evaluate and optimize explicit calibration (e.g., expected calibration error) to align confidence with correctness (Figure 5 hints this is promising).

> Bottom line: A minimal change—treating abstention distinctly from error in an online RL setup—alters incentives so the model stops guessing when unsure, sharply reducing hallucinations and improving trustworthiness (Table 1, Figure 3).
