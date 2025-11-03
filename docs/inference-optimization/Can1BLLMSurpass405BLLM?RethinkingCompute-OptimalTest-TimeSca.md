# Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling

**ArXiv:** [2502.06703](https://arxiv.org/abs/2502.06703)

## 🎯 Pitch

This paper demonstrates that with carefully chosen, reward-aware test-time scaling (TTS) strategies, small language models (as small as 1B–3B parameters) can outperform much larger models—even those over 100 times bigger—on advanced mathematical reasoning tasks. By systematically analyzing how the optimal use of inference-time computation depends on the interplay between model size, verifier (PRM), and problem difficulty, the authors show that efficient, tailored TTS can dramatically boost reasoning accuracy while reducing computational cost by up to 1000×, reshaping the path to cost-effective and scalable AI systems.

---

## 1. Executive Summary
This paper asks when extra test-time compute can let a small language model outperform a much larger one on hard reasoning tasks. It builds a reward‑aware framework for “compute‑optimal” test‑time scaling (TTS) and shows, with extensive experiments on math benchmarks, that the best TTS strategy depends strongly on the policy model, the process reward model (PRM), and problem difficulty. Notably, with the right strategy a 1B–3B model can surpass models 100× larger on MATH‑500, and a 7B model beats frontier “long‑thinking” systems on AIME24 (Figure 1, Table 3).

## 2. Context and Motivation
- Problem addressed
  - How to allocate extra computation at inference time to boost reasoning accuracy most efficiently (“test‑time scaling,” TTS).
  - Existing TTS work does not systematically analyze how three interacting factors—policy model, verifier (PRM), and problem difficulty—determine the compute‑optimal strategy (Introduction; §1).

- Why this matters
  - Practical: If small models can match or exceed large ones via smarter inference, we can reduce total FLOPS by 100–1000× while retaining accuracy (Table 4).
  - Scientific: Clarifies when and why verifier‑guided search helps or hurts different models and tasks, providing principled guidance rather than one‑size‑fits‑all recipes.

- Prior approaches and gaps
  - Internal TTS (“long CoT”): train the model to think longer (e.g., o1; DeepSeek‑R1) but requires costly training and can be inefficient on easy problems (Introduction).
  - External TTS: sampling and search guided by a verifier/PRM (e.g., Best‑of‑N, beam search, verifier tree search) but prior studies pick one verifier and do not analyze cross‑model generalization or difficulty‑aware scaling (Figures 2, 4–5; §2.2).
  - Compute‑optimal TTS (Snell et al., 2024) formalizes per‑prompt optimality but ignores that different PRMs change the search distribution—this paper argues optimality must be reward‑aware (§3.1, Eq. 3).

- Positioning
  - Provides the first broad, controlled study of TTS across multiple policy families (0.5B–72B), many PRMs (1.5B–72B), several scaling strategies (BoN, beam, DVTS), and difficulty strata on two math benchmarks (MATH‑500, AIME24) with unified setups (§4.1; Figures 4–11).
  - Introduces two methodological refinements: reward‑aware compute‑optimality (§3.1) and absolute difficulty thresholds rather than dataset quantiles (§3.2, Figure 3).

## 3. Technical Approach
This section explains the framework, the TTS algorithms, and how the study is run.

- Problem formalization (§2.1; Eq. 1)
  - Each reasoning instance is an episode in an MDP: the policy model (LLM) generates a sequence of actions (tokens/steps) given state (prompt + prior steps). A PRM provides step‑level rewards ℛ(s, a) (a scalar score) for process supervision.
  - “Policy model” is the LLM that proposes steps; “process reward model (PRM)” is a separate model scoring the quality of intermediate steps (different from outcome‑only verifiers).

- Test‑time scaling methods (§2.2; Figure 2)
  - Best‑of‑N (BoN): Sample N complete solutions from the policy, then choose an answer using a vote or PRM‑based selection.
  - Beam search with PRM guidance: At each depth, expand candidates; the PRM selects the top N/M partial steps to continue (beam width N, branching M).
  - Diverse Verifier Tree Search (DVTS): Run multiple independent PRM‑guided subtrees (N/M of them), increasing diversity compared to a single beam (Figure 2).

- Scoring and voting (§4.1)
  - For a trajectory of length H, compute a PRM score per step; aggregate by:
    - `PRM-Min`: minimum step score; `PRM-Last`: last step score; `PRM-Avg`: mean score.
  - Final answer selection across candidates uses:
    - `Majority Vote`: most frequent answer.
    - `PRM-Max`: single candidate with highest score.
    - `PRM-Vote`: sum scores over identical answers, pick the highest total.

- Compute‑optimality and reward awareness
  - Classical per‑prompt compute‑optimality (Eq. 2): for budget N, choose TTS hyperparameters θ that maximize the probability the output equals the correct answer y*(x).
  - Key addition (§3.1; Eq. 3): make the “target” distribution explicitly depend on the reward function ℛ because search‑based methods change the generation distribution via PRM scores:
    - Target(θ, N, x, ℛ): distribution over outputs produced when the PRM guides search or is used for selection. This turns the optimization into a reward‑aware selection of strategy and hyperparameters per prompt.

- Difficulty‑aware design (§3.2)
  - Instead of difficulty quantiles tied to a specific model (which shift as models improve), define absolute bins by Pass@1 accuracy on a strong reference policy: easy (50–100%), medium (10–50%), hard (0–10%) (Figure 3).

- Experimental setup (§4.1)
  - Datasets: MATH‑500 (500 representative math problems) and AIME24 (hard Olympiad‑style problems).
  - Policy models: Llama‑3.x and Qwen2.5 families from 0.5B to 72B, instruct variants.
  - PRMs: Math‑Shepherd‑7B; RLHFlow PRMs (Mistral‑8B base; DeepSeek‑8B base); Skywork PRMs (1.5B, 7B); Qwen2.5‑Math PRMs (7B, 72B). Qwen2.5‑Math‑PRM‑72B is the strongest open‑source PRM tested (§4.1).
  - Search budgets: N in {4, 16, 64, 256}; one 1B case uses N=512 (Table 3 note).
  - Beam/DVTS beam width=4; max new tokens per response 8192; per‑step token cap 2048; temperature 0.7 for search and 0.0 for vanilla CoT (§4.1).
  - Code: OpenR2 reasoning framework.

- How PRM choice changes search behavior (mechanism)
  - Because the PRM ranks partial steps, it decides which branches survive; different PRMs can prefer shorter or longer steps and thereby change both accuracy and token usage.
  - A toy example (Figure 12) shows two PRMs scoring different partial steps; one leads to a short but wrong solution (660), the other explores longer branches and finds the correct answer (2220), producing nearly 3× more tokens.

## 4. Key Insights and Innovations
1) Reward‑aware compute‑optimal TTS (§3.1; Eq. 3)
- What’s new: Treat the verifier’s reward as part of the generative process—optimal hyperparameters depend on both the policy and the PRM because the PRM changes the search distribution.
- Why it matters: Explains why the best strategy varies across PRMs and why on‑policy vs off‑policy PRMs behave differently. Empirically, the same policy with different PRMs yields different accuracy‑vs‑compute curves (Figures 4–5).

2) Absolute difficulty thresholds (§3.2; Figure 3)
- What’s new: Replace “quantile‑based” difficulty splits (which shift with model strength) with fixed Pass@1 ranges: easy (≥50%), medium (10–50%), hard (<10%).
- Why it matters: Avoids misleading conclusions when a strong model makes most problems “easy” under quantiles (Figure 3 shows Qwen2.5‑72B solves >80% of MATH‑500 at Pass@1 on 76.2% of problems).

3) Empirical mapping from policy/PRM/difficulty to optimal TTS method (§4.2–§4.3; Figures 4–9)
- Novelty: A comprehensive matrix of results that yields actionable rules of thumb:
  - Small policies (<7B): search‑based methods > BoN (Figure 7).
  - Large policies (≥32B): BoN > search (Figure 7).
  - With strong PRMs (Skywork‑7B, Qwen2.5‑Math‑7B/72B): search scales well; with weaker or OOD PRMs (Math‑Shepherd, RLHFlow for some settings): BoN often wins (Figure 4).
  - By difficulty: for small policies, BoN works best on easy items, beam search on hard ones; for 7B–32B, DVTS is strong on easy/medium and beam on hard; at 72B, BoN is best across all levels (Figure 8; Figure 9).

4) Diagnosing PRM biases and sensitivities (§4.4; Table 1–2; Figures 13–18)
- Findings:
  - Length bias: PRMs trained on longer steps favor longer generations and consume more tokens at the same budget (Table 1; narrative around RLHFlow‑DeepSeek vs RLHFlow‑Mistral).
  - Voting sensitivity: Skywork‑PRM‑7B benefits from `PRM‑Vote` over `PRM‑Max`, while Qwen2.5‑Math‑PRM‑7B is relatively insensitive (Table 2).
  - Error modes with case studies: Over‑criticism (penalizing correct steps), error neglect (missing clear errors), error localization bias (penalizing the wrong step), and scoring bias by token length (Figures 13–18). These explain search failures and suggest training/labeling issues.

These insights go beyond incremental tweaks: they shift the recommended practice from “pick a search method and PRM” to “tune TTS jointly with the policy, PRM, and difficulty, and make PRMs part of the compute‑optimal formulation.”

## 5. Experimental Analysis
- Evaluation methodology (§4.1)
  - Datasets: MATH‑500 and AIME24 (harder Olympiad‑style).
  - Metrics: accuracy (Pass@k plots show upper bounds when one of k samples is correct), final answer accuracy under each selection method, and compute budgets N.
  - Baselines: Vanilla CoT for many models; frontier systems (GPT‑4o, o1 family, DeepSeek‑R1); open‑source long‑CoT/RL methods (rStar‑Math, Eurus‑2, SimpleRL, Satori) in §5.3.
  - Fairness notes: Small models use external TTS with compute budgets up to N=256 (N=512 for one 1B case). Large proprietary baselines are evaluated with their standard long‑CoT (no external search). FLOPS accounting separates pre‑training and inference (Table 4).

- Main quantitative results
  - Small beating large (Figure 1; Table 3):
    - 3B vs 405B: “Llama‑3.2‑3B‑Instruct (TTS) = 75.6 on MATH‑500, 30.0 on AIME24” vs “Llama‑3.1‑405B‑Instruct (CoT) = 71.4 and 23.3.”
    - 1B vs 405B: with larger budget N=512, “Llama‑3.2‑1B‑Instruct (TTS) = 72.2 on MATH‑500,” exceeding 405B’s 71.4 (but falls short on AIME24: 10.0 vs 23.3) (Table 3).
    - 0.5B/3B vs GPT‑4o: “Qwen2.5‑0.5B‑Instruct (TTS) = 76.4 on MATH‑500, 10.0 on AIME24” and “Llama‑3.2‑3B‑Instruct (TTS) = 75.6 / 30.0,” both surpass GPT‑4o’s 74.6 / 9.3 (Table 3).
    - 7B vs frontier long‑thinkers: “DeepSeek‑R1‑Distill‑Qwen‑7B (TTS) = 95.2 on MATH‑500, 83.3 on AIME24,” beating “o1 (94.8 / 79.2)” and beating “DeepSeek‑R1 (79.8 on AIME24)” while slightly below it on MATH‑500 (97.3) (Table 3, Figure 1c,f).

  - FLOPS savings (Table 4):
    - “Llama‑3.2‑3B (TTS) total FLOPS ≈ 1.62×10^23” vs “Llama‑3.1‑405B (CoT) ≈ 3.65×10^25.” This is ≈225× smaller.
    - “DeepSeek‑R1‑Distill‑7B (TTS) total FLOPS ≈ 7.56×10^23” vs “DeepSeek‑R1 (CoT) ≈ 5.96×10^25” (≈79× smaller).

  - How PRMs change outcomes (Figures 4–5):
    - For Llama‑3.1‑8B, search with Skywork and Qwen2.5‑Math PRMs improves steadily with budget, but search with Math‑Shepherd and RLHFlow PRMs performs poorly—even worse than Majority Vote in some regimes (Figure 4).
    - For Qwen2.5‑7B, Skywork‑7B and Qwen2.5‑Math PRMs scale well; others lag (Figure 4). On AIME24, gains are smaller across the board (Figure 5).

  - Which TTS method is best (Figure 7; §4.2):
    - Small policies (0.5B–3B): beam/DVTS > BoN.
    - Larger policies (32B–72B): BoN > search methods.

  - Difficulty‑aware findings (Figures 8–9; §4.3):
    - For small policies: BoN best on easy; beam best on hard.
    - For 7B–32B: DVTS shines on easy/medium; beam on hard.
    - At 72B: BoN dominates across difficulty levels.

  - Efficiency and effectiveness vs CoT and Majority (§5.2; Table 5):
    - Example: “Llama‑3.2‑1B‑Instruct: CoT=26.0, Majority=39.0, TTS=66.2,” a 155% improvement over CoT and >256× efficiency gain vs Majority under their budget metric.
    - Gains shrink as the policy gets stronger (e.g., Qwen2.5‑72B: CoT 83.8 → TTS 91.8, +9.5%).

  - TTS vs long‑CoT training (§5.3; Table 6):
    - With Qwen2.5‑7B policy and strong PRM, TTS gets “91.0/36.7,” outperforming rStar‑Math, Eurus‑2, SimpleRL, and Satori on both datasets.
    - Distilled DeepSeek‑R1‑7B reaches “92.4/63.3” with CoT alone—higher on AIME24 than TTS using a plain 7B policy without distillation—indicating specialized long‑CoT distillation still wins on the hardest problems.

- Robustness and diagnostics
  - Correlation between PRM process quality and TTS accuracy: fitted curve Y = 7.66 log(X) + 44.31, where X is ProcessBench score and Y is TTS performance (Figure 6).
  - Failure analyses exposing PRM error modes (Figures 13–18) substantiate why some searches derail.
  - Token‑length bias traced to PRM training data (Table 1) explains different compute consumption.

- Do the experiments support the claims?
  - Yes for the conditional headline: with reward‑aware, compute‑optimal TTS, small models can beat larger CoT‑only models on MATH‑500, and a 7B model can beat o1 and DeepSeek‑R1 on AIME24 (Figure 1; Table 3).
  - The support is strongest when PRMs are well‑matched and budgets are tuned; it is weaker on AIME24 for plain small policies without distillation (Table 3, 6).

## 6. Limitations and Trade-offs
- Dependence on PRM quality and match
  - PRMs generalize poorly across policy families or tasks; off‑policy PRMs can push search into local optima (Figures 4–5; §4.2). Reward‑aware optimality addresses this conceptually but does not remove the underlying brittleness.
- Sensitivity to hyperparameters and voting
  - Best scoring/voting combos differ by PRM (Table 2). This increases tuning burden and may complicate deployment.
- Compute budgets vs fairness
  - Comparisons pit “small+TTS” vs “large+CoT”; both are realistic usage modes but not identical compute regimes. The FLOPS analysis (Table 4) argues in favor of small+TTS on total compute, yet wall‑clock latency and memory haven’t been exhaustively profiled.
- Task scope
  - Evaluation is limited to mathematical reasoning; coding, science, and multi‑modal tasks remain open (Limitations, §7 Discussion).
- Failure modes in PRMs
  - Over‑criticism and error neglect (Figures 13–15) can invalidate search even when the policy proposes a correct path. These indicate data/labeling issues and limit reliability.
- Diminishing returns with strong policies
  - As base models get better, TTS gains shrink (Table 5), so the cost‑benefit must be reconsidered for very strong LLMs.

## 7. Implications and Future Directions
- How this changes the landscape
  - Moves the field from “TTS is good in general” to “TTS must be reward‑aware and tailored to the policy, the PRM, and problem difficulty.” It demonstrates that careful inference strategy selection can flip performance orderings across 100× parameter gaps (Figure 1; Table 3).
  - Establishes empirical rules: use search for small policies (with strong PRMs), favor BoN for large policies, and adapt the method by difficulty (Figures 7–9).

- Follow‑up research enabled
  - PRM research:
    - Training data curation to reduce over‑criticism/error‑neglect and length biases (Figures 13–18; Table 1).
    - Weak‑to‑strong supervision: the paper shows a 7B PRM effectively supervises a 72B policy (§7 Conclusion), motivating scalable verifiers rather than ever‑larger PRMs.
    - More robust scoring/voting designs that are less sensitive across policy families (Table 2).
  - TTS algorithms:
    - Adaptive budget allocation “mid‑generation” based on self‑predicted uncertainty (related to §6 Related Work; could be combined with reward‑aware search).
    - Difficulty predictors to route problems to BoN vs beam vs DVTS automatically, following the empirical mapping in Figures 8–9.
  - Beyond math:
    - Apply the reward‑aware framework to coding (unit tests as rewards), scientific question answering (symbolic checkers), or multimodal reasoning (vision value models; see §6 Related Work).

- Practical applications
  - Cost‑effective deployment: small on‑device or edge models augmented with PRM‑guided TTS for high‑accuracy math tutoring, homework checking, or exam prep.
  - Cloud inference optimization: dynamically choose BoN vs search based on prompt difficulty and model size to minimize latency/compute for a target accuracy.

---

Key citations to ground claims:
- Reward‑aware compute‑optimal formulation: §3.1, Eq. (3).
- Difficulty thresholds: §3.2, Figure 3.
- TTS methods: §2.2, Figure 2.
- Cross‑matrix results: Figures 4–11.
- Small vs large comparisons: Figure 1; Table 3.
- FLOPS comparisons: Table 4.
- Gains vs CoT and Majority: §5.2, Table 5.
- TTS vs long‑CoT training: §5.3, Table 6.
- PRM biases and failures: §4.4, Table 1–2, Figures 12–18.
