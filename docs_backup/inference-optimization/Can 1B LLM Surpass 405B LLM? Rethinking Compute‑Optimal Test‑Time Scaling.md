# Can 1B LLM Surpass 405B LLM? Rethinking Compute‑Optimal Test‑Time Scaling

**ArXiv:** [2502.06703](https://arxiv.org/abs/2502.06703)
**Authors:** Runze Liu, Junqi Gao, Jian Zhao, Kaiyan Zhang, Xiu Li, Biqing Qi, Wanli Ouyang, Bowen Zhou
**Institutions:** 

## 🎯 Pitch

This paper introduces Test-Time Scaling (TTS), a method that optimizes inference-time computation to enable smaller language models to outperform models up to 100 times larger in solving complex math problems. By integrating a reward-aware approach that considers verifier biases and problem difficulty, it provides a cost-effective alternative to model scaling, reshaping how inference-time strategies can rival extensive pretraining efforts with reduced computational resources.

---

## 1. Executive Summary
This paper investigates how to allocate “extra thinking” at inference time—called Test-Time Scaling (TTS)—so that small language models can solve difficult math problems competitively with, and sometimes better than, much larger models. It introduces a reward-aware notion of compute-optimal TTS and shows, through extensive experiments on MATH-500 and AIME24, that with the right search strategy and verifier, a 1–3B model can outperform models 100× larger, while using far less total compute (Tables 3–4, Figure 1).

## 2. Context and Motivation
- Problem addressed
  - How to use extra computation during inference (not training) to improve reasoning, and how to allocate that compute optimally for a given model, verifier, and problem difficulty.
  - Prior TTS work rarely studies the interaction among three factors that strongly shape outcomes: the generator (“policy model”), the verifier (“process reward model,” PRM), and problem difficulty (Section 1).

- Why it matters
  - Practical: If small models can be boosted at inference time to match or beat large models, deployment becomes cheaper and faster without retraining.
  - Scientific: Clarifies what, exactly, drives reasoning improvements at inference time and sets a methodology for choosing TTS strategies problem-by-problem.

- Prior approaches and their gaps
  - Internal TTS: train models to think more slowly with long Chain-of-Thought; powerful but requires access to training pipelines (Section 1).
  - External TTS: sample-and-vote (Best-of-N), beam search, or diverse tree search, often scored by a PRM (Figure 2). Past work did not systematically examine how policy/PRM mismatch, PRM quality, and difficulty control which TTS strategy is best (Sections 1, 3).

- Positioning
  - Provides a compute-optimal, reward-aware formulation of TTS (Equation 3) and a comprehensive empirical study across policy families (Llama-3.x, Qwen2.5; 0.5B–72B), multiple PRMs (1.5B–72B), and tasks (MATH-500, AIME24), with detailed analyses of PRM bias and difficulty effects (Sections 3–5; Figures 4–11; Tables 1–6).

## 3. Technical Approach
- Key terms (defined on first use)
  - `Test-Time Scaling (TTS)`: spending more compute during inference by, e.g., sampling multiple solutions or searching multiple partial solutions, then verifying and selecting the best (Section 2.2; Figure 2).
  - `Policy model`: the generator LLM that proposes steps and final answers (footnote 1; Section 2.1).
  - `Process Reward Model (PRM)`: a verifier LLM that scores each intermediate step (“process supervision”) or full solution to guide search and final selection (Section 2.2).
  - `Best-of-N (BoN)`: generate N full solutions, score or vote, return the winner (Section 2.2).
  - `Beam search`: expand multiple partial solutions in parallel; at each depth, keep the top-scoring beams for further expansion (Section 2.2).
  - `Diverse Verifier Tree Search (DVTS)`: run several independent PRM-guided beam searches to increase diversity; more effective than increasing beam width alone at large budgets (Section 2.2; Figure 2).
  - `Pass@k`: accuracy when allowed up to k attempts; Pass@1 is standard accuracy (Figures 4–7).

- Problem formulation as a process with rewards
  - The generation process is seen as a Markov Decision Process where the state is the text so far, the action is the next step, and the reward at each step is given by the PRM (Section 2.1; Eq. 1).
  - TTS affects the distribution over outputs by changing how many candidates are explored and how they are scored/selected.

- Compute-optimal scaling (what and how)
  - Prior notion (Eq. 2): for a fixed compute budget `N` on a prompt `x`, choose the TTS strategy (e.g., BoN vs beam, sampling temperature, etc.) that maximizes the probability of producing the correct answer.
  - New refinement—reward-aware compute-optimal TTS (Eq. 3): explicitly include the reward function `R` (PRM) in the target distribution Target(θ, N, x, R) because the PRM is what guides search and final selection. For sampling-only methods (BoN), Target does not depend on R, but for search methods it does (Section 3.1).

- Why reward-awareness is needed
  - Case study shows different PRMs steer search differently: one PRM (RLHFlow-Mistral-8B) pushes for short but wrong solutions; another (RLHFlow-Deepseek-8B) finds the correct solution but uses many more tokens (Figure 12; discussion in Section 3.1).
  - Hence, the “optimal” TTS choice depends not only on compute budget and policy model but also on which PRM provides rewards.

- Difficulty stratification: absolute, not quantiles
  - Instead of splitting problems by quantiles of a particular model’s Pass@1 (which changes with the model), they define fixed absolute bins: easy (50–100%), medium (10–50%), hard (0–10%) using a strong model’s distribution (Figure 3; Section 3.2). This avoids misleading comparisons across policy models.

- Scoring and voting details used in experiments
  - Step-level scoring for a trajectory of H steps: `PRM-Min` (min step score), `PRM-Last` (last step score), `PRM-Avg` (average step score).
  - Final answer selection: `Majority Vote`, `PRM-Max` (highest single score), `PRM-Vote` (sum scores per unique answer, then choose highest) (Section 4.1).

- Experimental setup (how they run everything)
  - Datasets: MATH-500 (500 math problems) and AIME24 (more challenging) (Section 4.1).
  - Policies: Llama-3.x Instruct and Qwen2.5 Instruct, sizes 0.5B→72B (Section 4.1).
  - PRMs: Math-Shepherd-7B; RLHFlow PRMs (Mistral-8B–based and Deepseek-8B–based); Skywork PRMs (1.5B, 7B); Qwen2.5-Math PRMs (7B, 72B—the strongest open PRM in this study) (Section 4.1).
  - Compute budgets: N ∈ {4, 16, 64, 256}; beam width 4; DVTS divides search into subtrees; temperature 0.0 for CoT, 0.7 otherwise; token limits 8192 overall and 2048 per step for search (Section 4.1; Figures 4–5).

## 4. Key Insights and Innovations
- Reward-aware compute-optimal TTS
  - Novelty: Makes the PRM an explicit part of the optimization target (Eq. 3), acknowledging that PRMs can dramatically change both path selection and final outcomes (Section 3.1; Figure 12).
  - Significance: Explains why the same TTS hyperparameters can succeed with one PRM and fail with another; provides a principled way to pick TTS strategies per PRM.

- Absolute difficulty criterion
  - Novelty: Use fixed Pass@1 thresholds to define easy/medium/hard rather than per-model quantiles (Section 3.2; Figure 3).
  - Significance: Enables fair, model-agnostic comparison of TTS gains across policy models with different base strengths.

- Systematic mapping from model/PRM/difficulty to the “right” TTS strategy
  - Finding: For small policy models, search-based methods (beam/DVTS) shine; for large models, BoN often works best; and optimal choices change with difficulty (Figures 7–9).
  - Significance: Moves beyond one-size-fits-all recipes; provides practical guidance for deploying TTS in the wild.

- Small models beating frontier models under compute-optimal TTS
  - Finding: With the right PRM and search, small models can outperform much larger ones and even specialized reasoning models, with far less total compute (Figure 1; Tables 3–4).
  - Significance: Demonstrates a compelling alternative to scaling model parameters—scale inference-time computation smartly.

- Diagnosis of PRM biases and their downstream effects
  - Evidence: Training data length correlates with PRM length bias (Table 1); choice of voting rule matters for some PRMs (Table 2); qualitative failure modes include over-criticism and error neglect (Figures 13–18).
  - Significance: Helps practitioners anticipate and mitigate verifier-induced artifacts that can derail search.

## 5. Experimental Analysis
- Evaluation design
  - Datasets and metrics
    - MATH-500 (500 representative math problems) and AIME24 (harder Olympiad-style problems) (Section 4.1).
    - Metrics: Pass@k, primarily Pass@1 for final reporting (Figures 4–7; Tables 3, 5, 6).
  - Policies and PRMs
    - Policies span 0.5B to 72B across Qwen2.5 and Llama-3.x families (Section 4.1).
    - PRMs include 1.5B, 7–8B, and 72B variants from several model families (Section 4.1).
  - TTS methods compared: CoT (no TTS), Majority, BoN, Beam, and DVTS, with multiple scoring/voting options (Sections 2.2, 4.1).

- Main quantitative results
  - Small models surpass large models when compute is allocated optimally
    - > “Llama-3.2-3B-Instruct (TTS) reaches 75.6 on MATH-500 and 30.0 on AIME24, surpassing Llama-3.1-405B-Instruct (CoT) at 71.4 and 23.3” (Table 3; Figure 1a, 1d).
    - > “Qwen2.5-0.5B-Instruct (TTS) scores 76.4 on MATH-500 and 10.0 on AIME24, beating GPT-4o at 74.6 and 9.3” (Table 3; Figure 1a, 1d).
    - > “DeepSeek-R1-Distill-Qwen-1.5B (TTS) achieves 91.6/63.3 vs o1-mini 90.0/63.6 and o1-preview 85.5/44.6” (Table 3; Figure 1b, 1e).
    - > “DeepSeek-R1-Distill-Qwen-7B (TTS) hits 95.2/83.3, beating o1 (94.8/79.2) and DeepSeek-R1 (97.3/79.8 on MATH-500/AIME24 respectively; it beats R1 on AIME24)” (Table 3; Figure 1c, 1f).
    - With more budget (N=512), even the 1B Llama exceeds the 405B Llama on MATH-500 (72.2 vs 71.4) though it lags on AIME24 (10.0 vs 23.3) (Table 3).
  - Compute cost comparison
    - > “Llama-3.2-3B (TTS) vs Llama-3.1-405B (CoT): Total FLOPS 1.62e23 vs 3.65e25 (≈225× less); inference FLOPS also lower” (Table 4).
    - > “DeepSeek-R1-Distill-7B (TTS) vs DeepSeek-R1 (CoT): 7.56e23 vs 5.96e25 total FLOPS (≈79× less)” (Table 4).
  - When does which TTS method work best?
    - PRM dependence: Search-based TTS scales well with Skywork and Qwen2.5-Math PRMs, but poorly with Math-Shepherd and RLHFlow PRMs (Figure 4 on MATH-500; Figure 5 on AIME24). BoN often wins when the PRM is mismatched or weak.
    - Policy size dependence: For Qwen2.5 family from 0.5B→72B, small models benefit more from search; large models favor BoN (Figure 7).
    - Difficulty dependence: For small models, BoN is best on easy, beam is best on hard. For 7–32B, DVTS helps on easy/medium, beam stays best for hard. For 72B, BoN is best across the board (Figures 8–9).
  - Gains vs CoT and Majority
    - > “On MATH-500, compute-optimal TTS improves Llama-3.2-1B from 26.0 (CoT) to 66.2 (+154.6%) and is >256× more efficient than Majority at matched performance” (Table 5).
    - > “For Qwen2.5-7B, TTS improves from 76.8 (CoT) and 83.6 (Majority) to 91.0 (TTS)” (Table 5).
    - Gains diminish with model size: improvement shrinks from +154.6% (1B) to +9.5% (72B) (Table 5).
  - PRM quality correlates with TTS performance
    - > “ProcessBench capability vs TTS performance follows Y = 7.66 log(X) + 44.31” (Figure 6).
  - TTS vs long-CoT training methods
    - > “Qwen2.5-7B-Instruct with TTS reaches 91.0 on MATH-500 and 36.7 on AIME24 (with 72B PRM), outperforming rStar-Math-7B, Eurus-2-7B-PRIME, SimpleRL variants, and Satori-Qwen-7B” (Table 6). Distilled reasoning models like DeepSeek-R1-Distill-7B still hold an edge on average.

- Robustness checks and diagnostics
  - PRM bias analysis
    - Length bias linked to training data: RLHFlow-PRM-Deepseek-8B’s data has longer steps and responses than RLHFlow-PRM-Mistral-8B (Table 1), aligning with observed longer inference traces for the Deepseek PRM (Section 4.4).
    - Voting sensitivity: Skywork-PRM-7B prefers PRM-Vote over PRM-Max, while Qwen2.5-Math-PRM-7B is insensitive (Table 2).
  - Failure cases catalog
    - Over-criticism (correct steps scored low), error neglect (wrong steps scored high), error localization bias, and scoring bias (Figures 13–18). These concretely illustrate why a PRM-aware compute-optimal strategy is necessary (Section C).

- Do the experiments support the claims?
  - Breadth: Many policy sizes and families, many PRMs, multiple TTS methods, and two datasets (Figures 4–11; Tables 3–6).
  - Depth: Correlation with PRM quality (Figure 6), difficulty-conditioned results (Figures 8–9), and qualitative PRM failure analyses (Figures 13–18) triangulate the central claim that “optimal TTS is PRM- and task-dependent.”
  - Cautions: Closed-source baselines (e.g., GPT-4o, o1) are compared without TTS; results hinge on chosen prompts and budgets (Section 4.1, Table 3). Nonetheless, comparisons against strong open models and distillations (e.g., DeepSeek-R1) are compelling.

## 6. Limitations and Trade-offs
- Dependence on verifiers
  - The approach assumes access to a PRM that generalizes to the policy’s outputs. Mismatch causes search to select locally optimal but wrong paths (Figures 4–5, 12). Strong PRMs (e.g., Qwen2.5-Math-PRM-72B) are especially beneficial but add compute and may be unavailable in some settings.

- Data and bias issues
  - PRMs inherit biases from their training data (length preferences; Table 1) and can be sensitive to voting rules (Table 2). Diagnosed failure modes (Figures 13–18) suggest that PRMs sometimes reward the wrong evidence.

- Diminishing returns with stronger policies
  - As policy size increases, TTS gains shrink (Table 5). Very large models may not justify complex search overheads, and simple BoN suffices (Figure 7).

- Task scope
  - Focus is on math reasoning. It remains to be tested whether the same conclusions hold for coding, science QA, or multimodal domains (Limitations section).

- Compute budget dependence
  - While total FLOPS drop drastically vs training larger models (Table 4), search-based TTS still increases inference latency and token usage, sometimes substantially (Figure 12 notes 2.4k tokens vs 0.9k in the case study).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes a practical path to frontier-level reasoning without scaling parameters: pair small–mid models with the right PRM and TTS strategy. This reframes “compute scaling” from pretraining-only to inference-time allocation with verifiers.
  - Encourages reward-aware TTS as a standard: optimal search settings should be chosen jointly with the verifier, not in isolation.

- What it enables next
  - Weak-to-strong supervision: The paper shows a 7B PRM can supervise a 72B policy effectively (Conclusion). This motivates research on training policies with weaker verifiers and closing the loop between inference-time search and training-time improvement.
  - Better PRMs and diagnostics: Address over-criticism, error neglect, and localization bias with improved data curation (e.g., LLM-as-a-judge filtering mentioned in Section 4.4), model objectives (advantage modeling, entropy regularization; Related Work), and calibration tools (ProcessBench/PRMBench correlations; Figure 6).

- Practical applications
  - Cost-efficient deployment of reasoning systems in education, automated grading, and competition math assistants: small models plus PRM-guided TTS can meet accuracy targets with vastly lower total compute (Tables 3–4).
  - Adaptive inference controllers: integrate the reward-aware compute-optimal logic into production systems that pick BoN vs beam vs DVTS per instance based on policy size, PRM identity, and estimated difficulty (Sections 3.1–3.2, 4.3).

- Concrete future directions proposed
  - Extend the analysis beyond math to coding and chemistry (Limitations).
  - Explore more effective compute-optimal controllers and PRM-training schemes that minimize OOD mismatch or reduce reliance on very large PRMs (Conclusion & Discussion).

> Bottom line: With a reward-aware, difficulty-sensitive choice of search method and verifier, inference-time compute can substitute for model size. The study offers actionable recipes—when to use BoN vs beam vs DVTS with which PRM and policy size—and documents both the upside (small models surpassing 100× larger ones) and the pitfalls (PRM biases).
