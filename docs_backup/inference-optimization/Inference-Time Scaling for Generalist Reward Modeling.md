# Inference-Time Scaling for Generalist Reward Modeling

**ArXiv:** [2504.02495](https://arxiv.org/abs/2504.02495)
**Authors:** Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, Yu Wu
**Institutions:** (Not explicitly stated in the arXiv abstract—likely DeepSeek and Tsinghua University)

## 🎯 Pitch

This paper introduces Self-Principled Critique Tuning (SPCT), a method that enhances large language models into scalable generalist reward models by leveraging inference compute to improve accuracy through sampling. By generating adaptable principles and critiques to score diverse responses, SPCT ensures more informed decision-making, paving the way for safer, more reliable automated evaluations across various domains without the need for extensive custom supervision.

---

## 1. Executive Summary (2-3 sentences)
This paper proposes Self‑Principled Critique Tuning (SPCT), a training method that turns a large language model into a generalist reward model (RM) which can get better simply by using more inference compute (sampling more outputs). The key idea is a pointwise generative RM that first generates “principles” (criteria) and “critiques” (reasoned assessments) and then assigns discrete scores to one or many responses; with parallel sampling and a meta‑RM to filter low‑quality samples, accuracy scales with the number of samples (Figures 1, 3; Eq. 14).

## 2. Context and Motivation
- Problem addressed
  - Reward models are used to grade or rank language model outputs and are central to reinforcement learning from human feedback (RLHF) and search‑at‑inference (Section 1). Existing RMs work well in narrow, verifiable domains (e.g., math or code with ground truth) but struggle as “generalist” judges across diverse tasks where criteria are varied and implicit (Introduction; Section 2.2).
  - A second gap: Most RMs do not improve when you spend more inference compute (more samples). This “inference‑time scalability” matters for practical systems that trade extra compute for higher quality at run time (Figure 1; Section 4).

- Why it matters
  - Practically, scalable RMs enable stronger alignment, better search, and more reliable automated evaluation across domains such as safety, helpfulness, and open‑ended chat (Introduction; Section 5.2).
  - Conceptually, the work suggests that the right learning objective can teach a model behaviors that are specifically “scalable” at inference, analogous to recent work on reasoning‑time scaling.

- Prior approaches and shortcomings (Section 2.1; Figure 2)
  - Scalar RMs produce a single number (e.g., Bradley–Terry models). They are efficient but offer little variance across samples, so sampling does not improve results (no inference‑time scaling) and they struggle with input flexibility (e.g., rating a single response vs. a list).
  - Semi‑scalar RMs add text explanations before outputting a scalar, but the scalar still limits sampling gains.
  - Generative RMs output textual judgments, often pairwise (“which of two is better?”). They allow richer feedback but can’t directly score a single response or a list without extra machinery.
  - Across these, inference‑time scaling is weak, and there is little work that explicitly teaches an RM to become more reliable when we sample and aggregate more.

- How this paper positions itself
  - It chooses a pointwise generative RM (“GRM”) that outputs a discrete score for each response in natural language (Figure 2(c)+(i); Eq. 7). This gives both input flexibility (single, paired, multiple) and the ability to scale via sampling and voting (Section 2.1).
  - It introduces SPCT, an online RL method that teaches the GRM to write good principles and critiques that lead to correct scores, so that sampling more such judgments increases accuracy (Sections 3 and 4).

## 3. Technical Approach
The system has three parts: a pointwise generative RM, a two‑stage training method (SPCT), and an inference‑time scaling procedure with meta‑filtering.

1) Pointwise Generative Reward Modeling (Section 2.1; Figure 2; Eqs. 1–3, 7–8)
- What the model emits
  - For a query `x` and `n` candidate responses `{y_i}`, the GRM generates:
    - A set of natural‑language “principles” `{p_i}`: criteria and their weights tailored to the query.
    - A “critique”: step‑by‑step evaluation based on these principles.
    - Pointwise scores `S_i` (integers by default in [1, 10]) for each response extracted from the critique text (Eq. 7).
- Why pointwise and generative?
  - Pointwise means each response gets its own score, which supports single responses (`n=1`), paired responses, and lists (`n>2`) in one format (Figure 2).
  - Generative means the model explains itself and can vary across samples; this variability is crucial for sampling‑based scaling (Section 2.1).

2) Unpinning principles from “understanding” to “generation” (Section 3.1; Eqs. 8–9)
- Instead of pre‑writing criteria, the model learns to generate principles on the fly given the query and responses (Eq. 9). This makes criteria adaptive to the task and sampleable.
- Motivation from a preliminary study (Table 1; Section 2.2): High‑quality “filtered” principles (those that led to a correct decision) improved accuracy, while unrestricted self‑generated principles did not. This motivates training the model to discover and use the “right” principles.

3) Two‑stage training: SPCT (Figure 3; Section 3)
- Stage A: Rejective Fine‑Tuning (RFT) — a “cold start” (Section 3.2, “Rejective Fine‑Tuning”)
  - Data: preference datasets where the best response is known; the model samples `N_RFT` trajectories (principles + critique + scores) per example.
  - Rejection rule (Eq. 10): keep only trajectories whose extracted pointwise scores pick the known best response (or the correct label for single‑response data). Also discard “too easy” items where all `N_RFT` samples are already correct (Figure 3).
  - Optional “hinted sampling”: append a hint of which index is best to help produce correct‑format trajectories when the pretrained model struggles (Section 3.2).
  - Purpose: teach the model the correct output format and initialize it to generate usable principles and critiques.

- Stage B: Rule‑based Online RL (Section 3.2, “Rule‑Based RL”)
  - Rollout with GRPO (a PPO‑style objective) using a simple accuracy‑based outcome reward:
    - `+1` if the extracted pointwise scores identify the ground‑truth best response; `-1` otherwise (Eq. 11).
  - No extra “format” reward is used; a larger KL penalty keeps outputs in the desired format and mitigates bias (Section 3.2; Appendix C.1: best stability at KL coefficient β=0.08 for the 27B model).
  - Objective (Eq. 15) is the standard clipped policy gradient with per‑token advantages, group size G=4 (Appendix C.1).
  - Effect: directly reinforces the generation of effective principles and high‑fidelity critiques that lead to correct scoring.

4) Inference‑time scaling with sampling and voting (Section 4)
- For each query, sample the GRM `k` times to get `k` different principles, critiques, and pointwise score vectors.
- Voting for pointwise GRM (Eq. 14):
  - Sum scores across the `k` samples for each response: `S*_i = sum_j S_{i,j}`.
  - Because each `S_{i,j}` is in a small discrete range (e.g., 1–10), summing “expands” the effective score space by `k`, producing finer‑grained distinctions as `k` grows (Section 4).
  - Responses are shuffled per sample to avoid positional bias (Section 4).
- Why this scales: each sample represents a different set of principles (different “judging perspectives”). Aggregating more independent perspectives better approximates the true preference signal (Section 4).

5) Meta‑RM–guided voting (Section 4, “Meta Reward Modeling Guided Voting”)
- Train a small pointwise scalar “meta RM” (binary classifier with cross‑entropy) to score the quality of each sampled trajectory (principles + critique) as likely‑correct vs. likely‑incorrect (Section 4; Appendix C.1).
- Training data: trajectories from RFT sampling and from the target GRM itself, labeled with the same correctness rule as Eq. 10 (to reduce train–inference policy mismatch; Section 4).
- At inference, keep only the top `k_meta` out of `k` samples with highest meta‑scores, then vote over those. This filters low‑quality, off‑policy, or noisy samples and boosts scaling (Tables 2, 3, 6).

Implementation notes (Appendix C.1):
- Models: Gemma‑2‑27B is the main backbone; additional runs with DeepSeek‑V2‑Lite (16B MoE), DeepSeek‑V2.5 (236B MoE), and DeepSeek‑V3 (671B MoE).
- Training data: 1.256M examples for RFT (1.07M general instruction data + 186K rejective‑sampled RM data) and 237K for RL; sources include MATH, UltraFeedback, OffsetBias, Skywork‑Reward, HelpSteer2‑Preference (Appendix C.1).
- Compute: for the 27B model, RFT 19.2 hours and RL 15.6 hours on 128×A100 GPUs (Table 5).

## 4. Key Insights and Innovations
- A. Pointwise generative RM that is both input‑flexible and sampling‑friendly (Figure 2; Section 2.1)
  - Novelty: unifies scoring for single, pairwise, and list inputs within one textual format that outputs per‑response scores (Eq. 7), unlike pairwise‑only or scalar models.
  - Significance: enables inference‑time scaling by sampling, because the model’s textual principles and critiques vary across samples, producing diverse, aggregatable scores.

- B. “Principle generation” as a first‑class behavior (Sections 2.2, 3.1)
  - Novelty: the model explicitly generates tailored principles (criteria with weights) before critique and scoring (Eq. 9).
  - Evidence: preliminary study (Table 1) shows that when principles correlate with correctness (“filtered principles”), accuracy improves; SPCT then teaches the model to generate such effective principles on its own.

- C. SPCT: rejective fine‑tuning + rule‑based online RL for scalable judgment (Sections 3.2; Figure 3; Eqs. 10–11, 15)
  - Novelty: a simple, domain‑agnostic online reward that directly grades whether the extracted pointwise scores match ground truth, with a large KL to preserve format, inducing sampling‑friendly behaviors.
  - Significance: drives improvements even without heavy curated supervision; ablations show online RL is crucial (Table 4/7).

- D. Meta‑RM–guided voting for effective inference‑time scaling (Section 4; Tables 2, 3, 6)
  - Novelty: train a small scalar meta‑RM to score trajectory quality and filter samples before voting.
  - Significance: consistently yields larger gains than naïve voting at the same `k` (e.g., Overall score 72.0 with MetaRM@8 vs 70.6 with plain Voting@8 in Table 3).

- E. Inference‑time scaling can beat training‑time scaling in this setting (Figure 4)
  - Insight: With 32 samples, the 27B GRM approaches or surpasses much larger models; with MetaRM@8 it achieves the best results on RewardBench among the tested configurations (Figure 4(a)), showing compute‑at‑inference is a powerful lever.

## 5. Experimental Analysis
- Benchmarks, metrics, and setup (Section 5.1; Appendix D.2)
  - RewardBench (RB): pairwise judging across Chat, Chat‑Hard, Safety, Reasoning; metric is accuracy of picking the better response.
  - PPE: two parts—Preference (crowd preferences) and Correctness (verifiable tasks like MMLU‑Pro, MATH, GPQA, MBPP‑Plus, IFEval); metric is accuracy.
  - RMB: helpfulness and harmlessness with pairwise and best‑of‑N; metric is accuracy; when `n>2`, success requires selecting the single best response.
  - ReaLMistake: single‑response error detection; metric ROC‑AUC (Appendix E.1.2; Table 13).

- Baselines (Section 5.1; Appendix C.2)
  - Scalar: Bradley‑Terry RM (“DeepSeek‑BTRM‑27B”), several public scalar RMs (e.g., Nemotron‑4‑340B‑Reward).
  - Semi‑scalar: CLoud‑Gemma‑2‑27B.
  - Generative pairwise: LLM‑as‑a‑Judge; also a variant that votes using token probabilities.
  - Public instruction‑tuned LLMs (e.g., GPT‑4o) as reference points.

- Main quantitative results (Tables 2–3, 6–10; Figures 1, 4, 6)
  - Overall performance across RM benchmarks (Table 2)
    - Greedy decoding (k=1): `DeepSeek‑GRM‑27B` achieves 69.9 Overall, better than reproduced generative baselines (LLM‑as‑a‑Judge 67.8; DeepSeek‑PairRM‑27B 69.0) and close to strong public models (GPT‑4o 71.3, Nemotron‑4‑340B‑Reward 70.5).
    - With inference‑time scaling:
      - Voting@32: 71.0 Overall.
      - MetaRM‑guided Voting@32: 72.8 Overall, surpassing GPT‑4o (71.3) and Nemotron‑4‑340B‑Reward (70.5).
  - Inference‑time scaling curves (Figure 1; Table 3; Figure 6b)
    - With `k≤8`, `DeepSeek‑GRM‑27B` gains +2.7 points (70.6 vs 67.9 in Table 6) using Voting, and +4.1 (72.0) with MetaRM@8 (Table 3).
    - Extending to k=32 continues to help: 72.8 with MetaRM@32 (Table 3).
  - RewardBench breakdown (Table 8; Figure 6a)
    - Greedy: 86.0.
    - Voting@8: 87.7; MetaRM@8: 89.8; MetaRM@32: 90.4.
    - Public reference Nemotron‑4‑340B‑Reward is 92.0 on RB; the 27B GRM narrows the gap substantially via sampling.
  - PPE Correctness (verifiable tasks) (Table 9)
    - Greedy: 59.8; Voting@8: 60.3; MetaRM@8: 63.0; MetaRM@32: 63.2.
    - With a provided reference answer, the same GRM hits 91.6 on PPE Correctness (Table 9, “w/ Reference”), revealing that difficulty here is judging correctness without reference.
  - RMB (Table 10)
    - Greedy: 69.0; Voting@8: 69.5; MetaRM@32: 70.3.
  - ReaLMistake (single‑response scoring; Table 13)
    - `DeepSeek‑GRM‑27B` achieves ROC‑AUC 72.2 with greedy and 74.4 with Voting@8, outperforming Gemma‑2‑27B‑it (65.8) and `DeepSeek‑V2‑Lite‑Chat` (61.9) at similar sizes.

- Training‑time vs inference‑time scaling (Figure 4)
  - On RewardBench, Voting@32 with the 27B GRM approaches the 671B MoE model that only had RFT (Figure 4(b)), and MetaRM@8 attains the best point on the curve in Figure 4(a). This supports the claim that for this RM design, adding inference compute can rival scaling model size.

- Ablations and diagnostics
  - Removing principle generation hurts both greedy and scaling performance (Overall 69.9 → 67.5 greedy; 70.6 → 68.0 at Voting@8; Table 4/7), confirming principles are not cosmetic.
  - Online RL matters: even without the RFT cold start, online RL lifts performance (66.1 → 68.7; Table 4/7).
  - Non‑hinted trajectories are especially useful; hinted‑only underperforms (Table 4/7).
  - General instruction data is necessary for good GRM behavior (Overall drops to 63.3 if removed; Table 4/7).
  - Meta‑RM robustness: performance is good across different `k_meta` values (72.7–72.8 at k_meta=8 or 16 with k=32; Table 4/7).
  - Response lengths after RL increase mainly on reasoning tasks (Figure 7), suggesting the model learned to allocate more “thinking” where needed.
  - Failure mode analysis (Appendix F.2; Figure 8): major errors come from incorrect critiques (reasoning or domain knowledge gaps) or imbalanced principle weights, rather than formatting issues.

- Do the experiments support the claims?
  - Yes for the central claims: (i) the GRM is input‑flexible (Appendix E.1 shows negligible difference between pair vs list input on RMB BoN; Table 11) and can score single responses (Table 13); (ii) SPCT improves greedy performance vs. baselines (Table 2); (iii) inference‑time sampling plus meta‑filtering yields consistent, sizable gains (Tables 3 and 6; Figures 1 and 6).
  - The work is candid about mixed results on strictly verifiable tasks without references (PPE Correctness; Table 9), and offers mitigations (reference‑based judging lifts to 91.6).

## 6. Limitations and Trade-offs
- Efficiency and latency
  - Generative judgment is longer than scalar scoring. Although parallel sampling keeps wall‑clock latency reasonable for `k≈8` (Section B, “Limitations”), total token generation cost grows with `k`.
  - Training also relies on substantial compute (128×A100; Table 5), and the RL stage uses a high KL penalty to preserve format (Appendix C.1).

- Performance on verifiable tasks without references
  - On PPE Correctness, the GRM trails scalar baselines (Table 2; Table 9) unless a reference is provided (then 91.6). This suggests current GRM critiques can miss subtle factual/logic checks that scalar verifiers implicitly learn.

- Quality control and bias
  - The meta‑RM filters low‑quality critiques but can inherit labeling biases from the correctness rule in Eq. 10 and the training datasets (Ethics Statement; Appendix F.2).
  - Principle weights are learned behaviors; imbalances sometimes lead to wrong outcomes (Appendix F.2; failure categories in Figure 8).

- Domain expertise and tool use
  - The method does not integrate external tools (e.g., calculators, web search). Failure cases include tasks needing real‑time data or precise numeric checks (Table 18).

- Applicability to online RL pipelines
  - Generative length and sampling requirements may limit throughput when RMs need to score many candidates per step. The paper notes this trade‑off and proposes parallel sampling and future efficiency work (Section B).

## 7. Implications and Future Directions
- How this changes the landscape
  - It shows that an RM can be deliberately trained to be inference‑time scalable. By making principles and critiques explicit and optimizing them with an accuracy reward, the RM benefits directly from “more samples → better decision,” which has been less clear for prior RMs (Figures 1, 4).

- Follow‑up research enabled or suggested
  - Tool‑augmented GRMs: integrate search, code execution, or retrieval to strengthen critiques on verifiable or knowledge‑intensive tasks (Appendix B, Future Directions (1)).
  - Two‑stage pipelines: pre‑compute high‑quality principle sets per domain/query and reuse them to reduce inference cost (Appendix B, (2)).
  - Co‑scaling with policies: use scalable GRMs during search or RL to lift policy performance at inference time (Section 6; Conclusion).
  - Process supervision: extend pointwise GRM to act as a process reward model (Appendix B) by evaluating intermediate reasoning steps, not only final outcomes.
  - Better meta‑judgers: explore stronger or self‑improving meta‑RMs and inference‑aware training (Section 4; Appendix C.1; citation to inference‑aware fine‑tuning in References).

- Practical applications
  - Safer, more reliable automatic evaluation for LLM development (model comparisons, A/B ranking, red‑teaming triage).
  - Scalable reward signals for RLHF, direct preference optimization, and test‑time search (sampling larger `k` when quality matters).
  - Analytics on model weaknesses: aggregating principles over many comparisons gives interpretable “what went wrong” diagnostics (Appendix B, (3)).

> Representative results to keep in mind:
> - Overall across benchmarks, `DeepSeek‑GRM‑27B` improves from 69.9 (greedy) to 72.8 with MetaRM‑guided Voting@32 (Table 2).
> - On RewardBench, it rises from 86.0 to 90.4 with MetaRM‑guided Voting@32 (Table 8).
> - With reference answers on verifiable tasks, accuracy reaches 91.6 (Table 9), clarifying where current limitations lie.

Overall, SPCT plus pointwise GRM provides a concrete recipe for generalist reward modeling that is both flexible across input types and predictably improves with more inference compute, establishing a strong baseline for inference‑time scalable evaluators.
