# TruthRL: Incentivizing Truthful LLMs via Reinforcement Learning

**ArXiv:** [2509.25760](https://arxiv.org/abs/2509.25760)

## 🎯 Pitch

TruthRL introduces a novel reinforcement learning framework for large language models that directly optimizes for truthfulness by rewarding correct answers, treating honest abstentions neutrally, and explicitly penalizing hallucinations. This approach moves beyond traditional accuracy-driven training, enabling models to recognize the limits of their knowledge and avoid confidently producing false information—a crucial capability for high-stakes applications where mistakes can have severe consequences. Extensive evaluations demonstrate that TruthRL substantially reduces hallucinations and boosts reliability across tasks, establishing a new paradigm for developing trustworthy AI systems.

---

## 1. Executive Summary
TruthRL is a reinforcement‑learning framework that trains language models to be truthful by rewarding three distinct outcomes: correct answers, explicit abstentions (e.g., “I don’t know”), and penalizing hallucinations (confidently wrong answers). Across four knowledge‑intensive benchmarks and multiple backbones, it lowers hallucinations by large margins and raises a composite truthfulness score, showing that directly optimizing for uncertainty‑aware behavior outperforms accuracy‑only training.

## 2. Context and Motivation
- Problem addressed
  - Large language models (LLMs) often “hallucinate”—produce fluent but factually wrong statements—especially when questions fall outside what they know or when retrieved evidence is noisy. The paper defines a truthful model as one that both answers correctly when it can and abstains when it cannot (Introduction; Section 2.1).
- Why this matters
  - In high‑stakes domains (law, medicine), a wrong but confident answer is more harmful than an admission of uncertainty. The paper explicitly argues that “accuracy alone does not guarantee truthfulness” because accuracy‑centric training can incentivize guessing (Introduction).
- Shortcomings of prior approaches
  - Accuracy‑driven supervised fine‑tuning (SFT) and vanilla RL reward “answering” behavior and suppress abstentions, which can amplify hallucinations (Sections 2.2–2.3; Figure 2).
  - Retrieval‑augmented generation (RAG) helps but retrieval can be noisy or misleading (Introduction).
  - Methods that teach abstention (e.g., R‑Tuning) require nontrivial dataset construction and often become overly conservative, reducing correct coverage (Introduction; Section 3.1; Table 1).
- Positioning
  - The paper reframes the objective from “maximize accuracy” to “maximize truthfulness,” i.e., reward correct answers and calibrated abstentions while penalizing hallucinations (Section 2.1). It proposes an RL algorithm (TruthRL) that operationalizes this with a simple ternary reward (+1 correct, 0 uncertain, −1 incorrect) trained via GRPO, a group‑relative policy optimization scheme (Section 3.2).

## 3. Technical Approach
This section explains all moving parts and why they were chosen.

- Problem formulation (Section 2.1)
  - Define three per‑question outcomes:
    - `Acc` (accuracy): fraction of questions answered correctly.
    - `Unc` (uncertainty rate): fraction answered with abstention (e.g., “I don’t know”).
    - `Hall` (hallucination rate): fraction answered incorrectly.
  - Define a `truthfulness score` as a weighted sum: Truthfulness = `w1·Acc + w2·Unc − w3·Hall`. Experiments use `w1=1, w2=0, w3=1` (Table 1 description), which evaluates truthfulness as accuracy minus hallucination, while treating abstention neutrally.
- Training objective and algorithm (Section 3.2)
  - The model is optimized with GRPO. In plain language, GRPO samples a small “group” of responses per prompt from the current policy, evaluates each with a reward, and updates the policy to increase the probability of responses that score above the group’s mean (with PPO‑style clipping and a KL penalty to prevent drift).
  - Notation (for intuition rather than derivations):
    - For each prompt `x`, sample `G` responses `{y_i}`. Compute rewards `{r_i}`.
    - Each response’s advantage `Â_i` scales how much to push up or down its probability, defined as z‑score within the group: `(r_i − mean(r))/std(r)`. This “relative within‑group” comparison is the key mechanism that lets small reward differences matter.
  - Reward design (Section 3.2)
    - `Binary reward`: +1 if correct, −1 otherwise. This conflates abstentions with wrong answers (both −1). In GRPO’s group‑relative update, abstentions get no advantage over hallucinations.
    - `Ternary reward` (TruthRL): +1 correct, 0 uncertain, −1 incorrect. Now, if a group contains an abstention and an incorrect answer, the abstention’s 0 is above the −1 average, so it gets positive advantage while the hallucination gets negative advantage.
    - Concrete example (Section 3.2): in a group with an abstention (`r=0`) and a hallucination (`r=−1`), the abstention outranks the hallucination under the ternary scheme, but not under the binary scheme.
- Why ternary over alternatives
  - It incentivizes two desired behaviors simultaneously: (1) prefer correct answers over anything, and (2) when not confident, prefer abstention over guessing.
- Baselines that can express uncertainty (Section 3.1)
  - `Knowledge boundary probing`: sample 256 responses per training question; if none are correct, label it `out‑of‑knowledge (OOK)` and set the target to “I don’t know.” This creates training data for abstention‑aware SFT baselines.
  - `R‑Tuning`: standard SFT where OOK questions are paired with “I don’t know” ground truth (Section 3.1).
  - `RFT` (rejection sampling fine‑tuning): choose the model‑generated trace that ends in “I don’t know” for OOK, and the correct trace for non‑OOK (Section 3.1).
- Optional reward variants (Section 3.2; Section 4.4; Section 4.6)
  - `Knowledge‑enhanced`: reward abstention positively (+1) when a question is OOK (identified by knowledge‑boundary probing) and penalize non‑abstention on OOK items.
  - `Reasoning‑enhanced`: add a separate reward for reasoning quality (judged by an LLM; Section 4.6), combined multiplicatively/additively/conditionally with the outcome reward.
- Verifier (Section 4.5; Table 5)
  - A large LLM judge (e.g., Llama3.3‑70B‑Instruct) evaluates answer correctness for rewards. Replacing it with a rule‑based string match collapses learning into over‑abstention and negative truthfulness (Table 5), showing the importance of semantic judging.

## 4. Key Insights and Innovations
- Ternary outcome reward that separates abstention from error (fundamental)
  - Novelty: Previous RL setups typically use binary correct/incorrect rewards, implicitly treating abstentions as errors. The ternary reward gives abstention zero (or positive in knowledge‑enhanced mode) rather than −1, making “I don’t know” preferable to guessing (Section 3.2).
  - Significance: It directly optimizes truthfulness rather than raw accuracy, leading to large hallucination reductions across settings (Table 1).
- Group‑relative optimization amplifies the abstention signal (mechanistic innovation within GRPO)
  - Because advantages are computed within sampled groups, any abstention’s 0 reward can become relatively “better” than a −1 hallucination in the same group (Section 3.2). This small design detail explains why the model learns to abstain when uncertain without over‑penalizing coverage.
- Simple reward outperforms more elaborate knowledge‑ or reasoning‑augmented schemes (empirical insight)
  - `Table 3` shows the ternary reward yields the best average truthfulness and lowest hallucinations across benchmarks, outperforming binary reward and knowledge‑enhanced variants. `Table 8` shows that adding reasoning rewards does not improve outcome truthfulness and can trade off accuracy vs. reasoning score.
- Online RL beats offline/semi‑online preference optimization for this objective (empirical insight)
  - `Table 4` compares offline `DPO`, semi‑online iterative DPO, and online TruthRL. Iterative DPO improves through early iterations but regresses later, while online TruthRL achieves the best truthfulness and lowest hallucination consistently.

## 5. Experimental Analysis
- Evaluation setup (Section 4.1; Appendix A–B)
  - Datasets: CRAG (Comprehensive RAG benchmark), NaturalQuestions (NQ), HotpotQA, MuSiQue.
  - Training: models trained on CRAG; evaluated on all four datasets.
  - Retrieval vs. non‑retrieval: both evaluated. Retrieval provides up to 50 documents (CRAG) or Wikipedia (others) with E5 retriever (Appendix A).
  - Backbones: `Llama3.1‑8B‑Instruct`, `Qwen2.5‑7B‑Instruct`, plus scale studies from 3B to 32B (Table 7).
  - Metrics: Truthfulness score (Acc − Hall with `w1=1, w2=0, w3=1`), hallucination rate, accuracy; uncertainty rate appears in breakdowns/figures.
  - Verifier for correctness: default LLM judge `Llama3.3‑70B‑Instruct`, with robustness checks across other judges (Table 6).
  - Baselines: Prompting, SFT, RFT, R‑Tuning; TruthRL with binary reward (“TruthRLBinary”) equates to “vanilla RL”; offline/semi‑online DPO variants (Table 4).
- Before training: evidence that SFT/RL suppress abstention (Section 2.3; Figure 2)
  - Figure 2 shows that the base model’s majority@k improves accuracy and abstention while reducing hallucination as more samples are aggregated. After SFT or vanilla RL, uncertainty collapses to near zero and hallucinations increase at larger k, revealing accuracy‑only training’s anti‑abstention bias.
- Main results (Table 1)
  - With retrieval, `Llama3.1‑8B` on CRAG:
    - Prompting: T=5.3, H=43.5, A=48.8.
    - SFT: T=1.4, H=49.3, A=50.7.
    - R‑Tuning: T=15.2, H=33.1, A=48.4.
    - TruthRLBinary: T=20.8, H=39.5, A=60.3.
    - TruthRL: T=37.2, H=19.4, A=56.6.
    - Quote:
      > Table 1 (CRAG, with retrieval, Llama3.1‑8B‑Inst): TruthRL achieves Truthfulness 37.2 with Hallucination 19.4 and Accuracy 56.6, outperforming all baselines.
  - With retrieval, `Qwen2.5‑7B` on CRAG:
    - Prompting: T=10.6, H=38.4, A=49.0.
    - RFT: T=22.6, H=31.4, A=54.0.
    - TruthRL: T=33.1, H=17.3, A=50.4.
    - Quote:
      > Table 1: TruthRL reduces hallucinations from 31.4 (RFT) to 17.3 and increases truthfulness from 22.6 to 33.1 on CRAG with retrieval.
  - Without retrieval (harder): `Llama3.1‑8B`
    - Prompting: T=−4.4, H=44.5, A=40.1.
    - SFT: T=−42.1, H=71.1, A=28.9 (hallucination explodes).
    - TruthRLBinary: T=−14.5, H=57.2, A=42.8.
    - TruthRL: T=22.4, H=16.3, A=38.7.
    - Quote:
      > Table 1 (no retrieval): TruthRL still improves truthfulness to 22.4 while cutting hallucinations to 16.3, whereas SFT raises hallucinations to 71.1.
  - Cross‑dataset averages (Table 1):
    - For `Llama3.1‑8B`, with retrieval, average T=25.6 and H=18.8 for TruthRL; prompting yields T=−16.4, H=54.1; TruthRLBinary yields T=4.5, H=47.7.
- Behavior decomposition and hard‑question analysis (Figure 3)
  - On all CRAG questions, TruthRL has the lowest hallucination and the highest uncertainty among methods while keeping strong accuracy (Figure 3a).
  - On a difficult subset where almost no method is correct, hallucinations for SFT and TruthRLBinary approach 100%, while TruthRL hallucinates only 15.5% and abstains 84.5% (Figure 3b).
  - Quote:
    > Figure 3b: On hard items, TruthRL: H=15.5%, Unc=84.5%, vs. SFT/TruthRLBinary: near‑universal hallucination.
- Hallucination‑baiting questions (Table 2)
  - Multiple‑choice style comparisons are known to induce guessing. TruthRL attains T=52.4 with H=16.5 and highest abstention among methods tested, while others have H≈39–49.
  - Quote:
    > Table 2: TruthRL’s hallucination rate is 16.5% on baiting questions, substantially lower than SFT (48.5%) or R‑Tuning (43.7%).
- Reward ablations (Table 3; Figure 4)
  - Binary reward excels at accuracy but keeps high hallucinations (e.g., CRAG with retrieval: T=20.8, H=39.5).
  - Ternary reward achieves CRAG T=37.2 with H=19.4—the best truthfulness/lowest hallucination.
  - Knowledge‑enhanced variants help abstention but tend to reduce accuracy or underperform ternary overall.
  - Learning curves (Figure 4) show ternary steadily reduces hallucination and maintains uncertainty; binary drives uncertainty to ~0.
- Online vs. offline RL (Table 4)
  - DPO: low truthfulness (average T=−10.1, H=51.1 across datasets).
  - Iterative DPO improves up to Iter 3 (avg T=12.6, H=31.7) but regresses at Iter 4.
  - TruthRL (online) achieves avg T=25.6 with H=18.8, the best across all regimes.
- Confidence calibration (Figure 5)
  - TruthRL increases the fraction of high‑confidence correct answers and reduces overconfident hallucinations compared to prompting.
- Verifier quality (Table 5)
  - Rule‑based judge leads to over‑abstention and negative truthfulness (T=−3.6), while LLM judge enables usable reward signals (T=37.2).
- Judge robustness (Table 6)
  - Using three different high‑capacity judges, TruthRL consistently gives the lowest hallucination and highest truthfulness, indicating it does not “overfit” a specific judge.
- Scale trends (Table 7)
  - Gains are consistent from 3B to 32B. Improvements are relatively larger for smaller models (e.g., Llama3.2‑3B: Prompting T=1.9/H=45.1 → TruthRL T=27.4/H=21.5).
- Reasoning rewards (Table 8)
  - Outcome‑only TruthRL already lifts a separate reasoning‑quality score (50.2 → 56.6). Adding reasoning reward via simple heuristics fails to improve outcome truthfulness and can trade off metrics (e.g., additive increases reasoning to 59.1 but slightly lowers truthfulness to 36.1).

Do the experiments support the claims?
- Yes, because:
  - Hallucination reductions and truthfulness improvements are shown across datasets, model families, scales, and under multiple evaluators (Tables 1, 6, 7).
  - Mechanism‑aligned behavior changes (more abstentions when necessary; fewer overconfident errors) appear in decomposition and confidence analyses (Figures 3 and 5).
  - Ablations isolate the reward structure as causal (Table 3; Figure 4) and show online RL is critical (Table 4).
- Caveats:
  - Truthfulness score sets `w2=0`, so abstention is neutral in evaluation; the benefit of abstentions is argued qualitatively and via reduced hallucination, but not rewarded numerically in the main score (Section 4.1 setup).

## 6. Limitations and Trade-offs
- Dependence on LLM‑as‑a‑judge for rewards and evaluation (Section 4.5; Tables 5–6)
  - Assumption: the judge accurately recognizes semantic correctness and abstentions. Judge bias or inconsistency could skew training/evaluation. The paper partially mitigates this by cross‑judge checks (Table 6), but dependence remains.
- Metric choice treats abstention neutrally (Section 4.1)
  - With `w2=0`, the main truthfulness metric does not directly reward abstention; effects appear via reduced hallucination and behavioral analyses. In deployments that value abstention, alternative weights or separate metrics would be needed.
- Training cost and complexity (Appendix A)
  - Online RL with GRPO requires rollouts, a verifier LLM, and long context windows (16k–32k), trained on 8×H100 GPUs with vLLM/DeepSpeed infrastructure. This may be costly relative to SFT or offline DPO.
- OOK detection for knowledge‑enhanced variants and SFT baselines (Section 3.1)
  - Identifying out‑of‑knowledge questions via sampling 256 generations is approximate. Mislabeling could misguide knowledge‑enhanced rewards or R‑Tuning data.
- Scope of benchmarks and retrieval noise
  - Training is on CRAG and tests include three other QA datasets; results are strong but focused on knowledge‑intensive QA. Other domains (dialogue safety, coding, planning) are not evaluated here.
- Risk of style gaming
  - Because abstention is recognized by natural‑language phrases (e.g., “I don’t know”), models could learn template phrases without perfectly calibrated internal uncertainty. The paper’s behavioral analyses suggest genuine improvements (Figures 3 and 5) but do not fully rule out surface‑form gaming.

## 7. Implications and Future Directions
- Field‑level shift: from accuracy‑centric to truthfulness‑centric training
  - The work demonstrates that a minimal change in reward structure can steer models to recognize knowledge boundaries and avoid harmful guesses, a key property for high‑stakes use. This reframes post‑training objectives for reliable LLMs.
- Practical applications
  - High‑risk domains (clinical decision support, legal research), enterprise assistants, and RAG systems where sources can be noisy. TruthRL’s behavior—answer when warranted, otherwise abstain—fits workflows that escalate to humans or trigger retrieval/re‑query.
- Follow‑up research
  - Reward design:
    - Multi‑objective formulations that explicitly value abstention (`w2>0`) and calibrate penalties by risk.
    - Better reasoning‑aware rewards beyond simple heuristics (Table 8 indicates nontrivial trade‑offs).
  - Verifier/judge reliability:
    - Ensemble judges or calibrated verifiers to reduce bias.
    - Programmatic semantic matching or weak supervision to complement LLM judges and lower cost (avoiding the collapse seen in Table 5).
  - Broader domains and tasks:
    - Apply ternary‑reward RL to code generation, tool use, and long‑form synthesis where selective abstention could trigger tools or retrieval.
  - Deployment policies:
    - Combine TruthRL with dynamic RAG: abstention triggers search or human review; correct answers bypass escalation.
  - Theoretical analysis:
    - Formal guarantees about abstention calibration under group‑relative updates; conditions where ternary rewards minimize expected risk vs. binary rewards.

Overall, TruthRL’s main contribution is conceptual simplicity paired with strong empirical validation: using an outcome‑based ternary reward under GRPO teaches LLMs to convert many would‑be hallucinations into either correct answers (when information is sufficient) or honest abstentions (when it is not), with consistent gains across models, datasets, and judges (Figures 3–5; Tables 1–7).
