# Inference-Time Scaling for Generalist Reward Modeling

**ArXiv:** [2504.02495](https://arxiv.org/abs/2504.02495)

## 🎯 Pitch

This paper introduces Self-Principled Critique Tuning (SPCT), a novel method that enables reward models to generate and adapt their own evaluation principles and critiques at inference time. By leveraging diverse, self-generated judgments and aggregating them with a meta reward model, SPCT turns language models into flexible, 'generalist' reward models that scale in accuracy as more computation is invested—often outperforming much larger models. This breakthrough matters because it democratizes high-quality alignment for LLMs across open-ended tasks, making safe and robust model evaluation feasible even outside narrow, rule-based domains.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Self‑Principled Critique Tuning (SPCT), a training method that turns a language model into a “generalist” reward model (RM) that can judge one, two, or many responses and gets better when you spend more compute at inference time. It shows that sampling multiple judgments guided by self‑generated principles and filtering them with a small “meta RM” scales performance better than simply using a larger model, achieving state‑of‑the‑art results across several reward‑modeling benchmarks (Table 2, Figure 1, Figure 4).

## 2. Context and Motivation
- Problem addressed
  - Large language models (LLMs) benefit from reinforcement learning that relies on reward models, but building accurate rewards outside narrow, verifiable domains (like math or code) is hard. The paper targets “generalist reward modeling”: scoring responses in broad, messy domains where no ground truth or rules are provided.
- Why it matters
  - Better generalist rewards improve alignment and decision quality in LLMs during both training (post‑training RL) and inference (e.g., search, best‑of‑n sampling). The paper argues that, like reasoning models, reward models should benefit from more inference compute (“inference‑time scaling”).
- Where prior approaches fall short (Figure 2; §2.1)
  - Scalar RMs output a single number per response; they are efficient but:
    - have little diversity across samples, so multiple samples don’t help much (§2.1; Eq. 1–4),
    - and can be biased to specific domains (Table 2).
  - Pairwise RMs excel at choosing the better of two responses (Eq. 3, 5–6), but are inflexible for single or many responses and force a strict winner (no ties), which can bias aggregation.
  - Semi‑scalar methods add a critique but still reduce to one scalar head, limiting diversity for scaling (Eq. 12; Table 3 shows small gains).
- Positioning
  - The paper chooses a pointwise generative reward model (GRM): the model writes a textual “principle + critique,” then outputs discrete scores for each candidate response (Si in [1..10]) that are parsed from text (Eq. 7). This unifies single, pairwise, and multi‑response scoring and, crucially, supports diversity for inference‑time scaling (§2.1–2.2, Figure 2).

## 3. Technical Approach
The system has three pillars: a pointwise GRM, a training method (SPCT) that teaches the GRM to generate effective principles and critiques, and an inference‑time scaling pipeline with voting plus a meta RM.

1) Pointwise Generative Reward Modeling (GRM)
- What it produces
  - For a query and n candidate responses, the GRM generates:
    - a set of “principles” (criteria with optional weights) tailored to the query,
    - a “critique” that applies those principles to each response,
    - pointwise integer scores Si ∈ {1,…,10} for each response, extracted from the text (Eq. 7).
- Why this design
  - Input flexibility: the same prompt format works for single, pairwise, and many responses (§2.1, Appendix E.1; Table 11 and Table 13).
  - Inference‑time diversity: principles and critiques can vary across samples, enabling useful voting (Eq. 14).

2) Principles as first‑class, generated objects (§2.2, §3.1)
- Principle (definition): a short description of what matters for judging the current query (e.g., “Accuracy 30%, Clarity 20%…”), often with weights.
- Insight: good principles dramatically improve judgments.
  - Preliminary study (Table 1): filtering to “correct” principles improves accuracy on RewardBench–Chat Hard from 76.1 to 77.8 for GPT‑4o and from 59.1 to 68.0 for Gemma‑2‑27B‑it.
- Mechanism shift: “unpinning” principles from a static guideline to a generated part of the judgment.
  - The same model first samples principles `pi` and then produces a critique conditioned on those principles (Eq. 9). Both are emitted by the LLM’s language head.

3) SPCT: Self‑Principled Critique Tuning (§3)
SPCT has two stages that teach the GRM to generate useful principles and reliable critiques:

A. Rejective Fine‑Tuning (RFT) – the cold start (§3.2)
- Data: a mix of general instruction data and RM data covering single, pairwise, and multi‑response cases (§C.1).
- Sampling: for each RM datapoint, sample NRFT trajectories (principles, critique, scores). Keep only trajectories whose extracted scores identify the benchmark’s labeled “best” response; discard trajectories that are wrong, and also discard datapoints that are “too easy” (when all NRFT samples are right) (Eq. 10).
- Hinted vs. non‑hinted sampling:
  - Non‑hinted: the model sees only the inputs and must judge.
  - Hinted: the input includes the correct best index (“The best response is: Response j”) to help elicit correct formatting and reasoning (§3.2). These hinted samples are kept only if the outputs are correct.
- Purpose: teach the model the output format and basic skill to tie principles → critique → scores across input types.

B. Rule‑Based Online RL with GRPO (§3.2; Eq. 11 and Eq. 15)
- Rollout: the GRM generates principles, a critique, and scores; the extracted scores are compared to the ground truth label from preference data.
- Reward: +1 if the pointwise score vector ranks the labeled best response above all others (or equals the single‑response label), else −1 (Eq. 11). No “format reward” is used.
- Stability: a relatively large KL penalty toward the reference policy (β = 0.08 for 27B) keeps outputs on‑format and reduces bias/collapse (Eq. 15; §C.1). Group size G=4.

4) Inference‑Time Scaling (§4)
- Why scaling helps: each sample can produce different principles and critiques, offering diverse “evaluation perspectives.” Summing k sets of discrete scores expands the effective reward range from 10 levels to roughly 10k levels (Eq. 14).
- Aggregation schemes (Figure 2; §4):
  - Semi‑scalar: average scalar scores (Eq. 12) – limited variance.
  - Pairwise generative: majority vote on the chosen index (Eq. 13) – no ties, coarse.
  - Pointwise generative (this work): sum per‑response scores over k samples (Eq. 14); shuffle response order to avoid positional bias.
- Meta Reward Model (“meta RM”)
  - Purpose: filter out low‑quality sampled judgments before voting.
  - What it is: a small pointwise scalar classifier trained to predict whether a sampled trajectory’s scores are “correct” (binary label from Eq. 10) using the query, responses, generated principles, and critique as input (§4).
  - Training data: trajectories from RFT non‑hinted sampling plus additional trajectories from the deployed GRM (to reduce policy shift; §4, Appendix C.1).
  - Use at inference: score all k samples, keep top kmeta (default kmeta = k/2), then vote over that subset (Table 2, Table 3, Table 6).

5) Implementation & resources (Appendix C)
- Models: built on Gemma‑2‑27B; also reported for 16B, 230B, and 671B variants (Table 8; Figure 4).
- Data: 1.256M RFT examples (1.07M general instruction + 186K rejectively sampled) and 237K RL examples; sources include UltraFeedback, HelpSteer2‑Preference, OffsetBias, MATH, and internal preference sets (§C.1).
- Training: 900 steps for RFT (LR 5e‑6, batch 1024) and 900 steps for RL (LR 4e‑7, batch 512), on 128×A100; time 19.2h (RFT) + 15.6h (RL) for the 27B model (Table 5).
- Inference settings: temperature 0.5 for scaling experiments; scores in [1..10].

## 4. Key Insights and Innovations
1) Pointwise GRM as a unifying interface (Figure 2; §2.1)
- Novelty: treats reward generation as text generation with structured extraction, enabling single/pair/multi‑response judging without changing format (Appendix E.1).
- Significance: unlocks inference‑time scaling via sampling diversity (Eq. 14), unlike scalar heads which barely vary across samples (Table 3: CLoud’s +0.3 overall vs GRM’s +2.7 to +4.9).

2) Principles moved from “understanding” to “generation” (§3.1; Eq. 9)
- Novelty: the model first proposes task‑specific principles (with weights), then applies them. Preliminary evidence (Table 1) shows that better principles materially improve accuracy; SPCT teaches the model to generate such principles itself.
- Significance: principles act as controllable, interpretable “evaluation lenses,” improving robustness and making scaling by sampling meaningful (Figure 3 examples).

3) SPCT training recipe: rejective fine‑tuning + rule‑based RL (§3.2)
- Novelty: a unified reject‑and‑keep strategy (Eq. 10) across input types, plus online RL with a simple “correctness” outcome reward (Eq. 11) and a strong KL penalty for stability (Eq. 15).
- Significance: markedly improves both base quality and scaling behavior (Table 4/7 ablations), with non‑hinted data especially helpful.

4) Meta RM–guided voting (§4)
- Novelty: a small classifier filters sampled judgments using the generated principles and critiques themselves.
- Significance: scaling gains improve further and require fewer samples: with 8 samples and kmeta=k/2, the 27B GRM reaches an overall 72.0 vs 70.6 with naive voting and 67.9 with no scaling (Table 3/6).

5) Inference‑time scaling can outperform size scaling (Figure 4)
- Finding: 27B GRM with 32‑sample voting approximates 671B performance on RewardBench, and meta‑guided voting with only 8 samples outperforms all tested sizes in that comparison (Figure 4a vs 4b).

## 5. Experimental Analysis
- Benchmarks & metrics (§5.1; Appendix D.2)
  - RewardBench (RB): preference accuracy across Chat, Chat‑Hard, Safety, Reasoning subsets.
  - PPE: Preference (crowdsourced) and Correctness (verifiable tasks like MATH, MBPP, GPQA); accuracy.
  - RMB: Helpfulness/Harmlessness; both pairwise and best‑of‑N (BoN); accuracy.
  - ReaLMistake: single‑response error detection; ROC‑AUC.
- Baselines (§5.1; Table 2)
  - Scalar or semi‑scalar: DeepSeek‑BTRM‑27B (Bradley‑Terry), CLoud‑Gemma‑2‑27B, Skywork‑Reward‑Gemma‑2‑27B, Nemotron‑4‑340B‑Reward, InternLM2‑20B‑Reward, ArmoRM‑8B.
  - Pairwise generative/scalar hybrids: DeepSeek‑PairRM‑27B, LLM‑as‑a‑Judge (with and without token‑probability scoring).
  - General LLMs: GPT‑4o, Gemini‑1.5‑Pro, Claude‑3.5‑sonnet, LLaMA‑3.1‑70B‑Instruct.

- Main results (Table 2; Figure 1)
  - Base (greedy): 
    > “DeepSeek‑GRM‑27B: Overall 69.9 vs. CLoud 68.7, DeepSeek‑BTRM 68.6, DeepSeek‑PairRM 69.0.”
  - Scaled (Voting@32):
    > “DeepSeek‑GRM‑27B (MetaRM): Overall 72.8; naive voting 71.0.”
  - By benchmark (Table 2):
    - RB: 90.4 with MetaRM@32 vs GPT‑4o 86.7 and Nemotron‑4‑340B‑Reward 92.0.
    - PPE‑Preference: 67.2 with MetaRM@32 vs GPT‑4o 67.1.
    - PPE‑Correctness: 63.2 with MetaRM@32 (still below scalar BTRM’s 66.7; see trade‑offs below).
    - RMB: 70.3 with MetaRM@32 (best among listed).

- Inference‑time scaling behavior (Table 3/6; Figure 6)
  - Overall (all benchmarks):
    > “GRM‑27B improves from 67.9 (k=1) → 70.6 (k=8) → 71.0 (k=32). MetaRM lifts to 72.0 (k=8) and 72.8 (k=32).”
  - RewardBench specifically (Table 8):
    > “GRM‑27B: 86.0 (greedy) → 87.7 (k=8) → 88.5 (k=32); MetaRM@32: 90.4.”
  - CLoud and LLM‑as‑a‑Judge show much smaller gains: e.g., CLoud’s overall +0.3 at k=8; LLM‑as‑a‑Judge +0.6 at k=8 (Table 6).

- Domain‑specific patterns (Tables 8–10)
  - Scalar/semi‑scalar models excel on verifiable tasks (PPE‑Correctness): DeepSeek‑BTRM‑27B gets 66.7 vs GRM‑27B greedy at 59.8 (Table 9). However, they underperform elsewhere and show domain bias (Table 2).
  - GRM narrows the gap with reference access: with explicit solution references, GRM‑27B jumps to 91.6 on PPE‑Correctness (Table 12), indicating the judgment mechanism is capable but blind to hidden ground truth.

- Input flexibility checks (Appendix E.1)
  - Many responses (RMB BoN): pair‑wise vs list‑wise inputs differ by <1 pp (Table 11), confirming the unified interface works.
  - Single responses (ReaLMistake): GRM‑27B reaches 72.2 ROC‑AUC with greedy, 74.4 with k=8; competitive among similarly sized models (Table 13).

- Ablations and diagnostics (Table 4/7; Appendix E.4)
  - Principle generation matters:
    > “Removing principle generation drops overall from 69.9 → 67.5 (greedy) and from 70.6 → 68.0 (k=8)” (Table 4/7).
  - Online RL is essential:
    > “Without RFT but with RL, performance rises from 66.1 → 68.7” (Table 4/7), and non‑hinted samples help more than hinted.
  - Meta RM robustness:
    > “At k=32, kmeta of 8 or 16 yields 72.7–72.8 overall” (Table 4/7).
  - Response length analysis:
    > “RL increases output length mainly on Reasoning subset; lengths stay similar or drop for Chat/Safety (Appendix E.4, Figure 7).”

- Failure modes (Appendix F.2; Figure 8; case studies Tables 16–18)
  - Common causes:
    - Incorrect critiques in complex domains (pattern matching, counting).
    - Imbalanced principle weights.
    - Occasionally, dataset label disagreements.
  - Example (Table 18): task requires real‑time prices; GRM prioritizes analysis clarity over factual recency, mis‑ranking answers even though the ground truth prefers the price‑correct one.

- Are the experiments convincing?
  - The paper evaluates across four diverse benchmarks with strong baselines, shows scaling curves (Figures 1, 6), ablations (Tables 4/7), and failure analyses. Evidence supports two core claims:
    - SPCT improves base GRM quality and generality (Table 2).
    - Inference‑time scaling with voting and the meta RM yields meaningful, consistent gains (Table 3/6, Figure 1).
  - The gap on verifiable tasks remains when no reference is provided (Table 9), clarifying the method’s current trade‑off.

## 6. Limitations and Trade-offs
- Efficiency
  - Generative judging emits long text (principles + critique) before scores; slower than scalar heads (§B). Parallel sampling helps latency, but total compute grows with k.
- Verifiable domains
  - Without a reference or external tools, GRM underperforms scalar RMs on correctness‑heavy tasks (Table 9). With references, it excels (Table 12), suggesting integration with tools/ground truth could fix this (§B; Appendix E.1.3).
- Dependence on labeled preferences
  - RL reward is a binary correctness signal w.r.t. labels (Eq. 11). Quality depends on label reliability; the paper notes cases where annotator preferences disagree with labels (Figure 8).
- Stability controls
  - The method relies on a strong KL penalty (β=0.08 for 27B) to avoid collapse and bias (§C.1). Tuning may vary across sizes (β=0.002 for 16B).
- Data and compute constraints
  - Larger models (230B, 671B) did not receive online RL due to resource limits (§C.1), complicating pure size comparisons.
- Principle weight learning
  - Weight attribution emerges from learning and can be unbalanced in edge cases (Appendix F.2, Figure 8), impacting final scores.

## 7. Implications and Future Directions
- How this changes the landscape
  - The work reframes reward modeling as an interpretable, sample‑efficient, and scalable generation task. It shows that spending inference compute on diverse, principle‑guided judgments can rival or beat scaling parameters (Figure 4), echoing broader trends in inference‑time scaling for reasoning.
- Follow‑up research it enables
  - Tool‑augmented GRMs: connect to code interpreters, search, calculators, or on‑chain data to strengthen fact‑checking and verifiable criteria (§B; related work §6; Li et al., 2024b).
  - Two‑stage pipelines: pre‑generate reusable principle sets per task/domain, then run fast critiques to reduce latency (§B).
  - Joint scaling with policies: co‑scale policy sampling and GRM judging (“inference‑time co‑scaling”) (§6 Related Work).
  - Better meta‑evaluation: stronger meta RMs and noise‑robust training (Appendix F.2; Chow et al., 2025) to filter biased or low‑quality critiques more reliably.
  - Process‑level judging: extend the pointwise GRM to step‑wise “process rewards” for long‑horizon reasoning (Appendix B; Table 8 Reasoning subset and DeepSeek‑R1 comparison in Appendix D.3).
- Practical applications
  - Safer, more general post‑training RL pipelines where a single RM can evaluate varied tasks.
  - Test‑time selection and reranking (best‑of‑n, beam search rescoring) with interpretable rationales.
  - Offline evaluation of foundation models: aggregating principle‑level weaknesses across models to produce actionable diagnostics (§B, Future Directions (3)).

> Bottom line: SPCT turns reward modeling into a scalable, interpretable generation problem and shows that “more thinking at inference” (via diverse principles and critiques) is a powerful lever—often more impactful than just training a larger model (Figure 4).
