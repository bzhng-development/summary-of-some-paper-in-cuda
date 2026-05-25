# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

**ArXiv:** [2408.03314](https://arxiv.org/abs/2408.03314)

## 🎯 Pitch

This paper introduces a 'compute-optimal' policy that adaptively allocates test-time (inference) compute for large language models, choosing the best strategy per prompt—such as iterative self-revision or search with step-wise verifiers—based on problem difficulty. Demonstrated on the challenging MATH benchmark, this approach enables small models to achieve or exceed the performance of models up to 14× larger, all while using 4× less test-time compute than standard methods, highlighting a transformative path to more efficient and flexible AI systems without simply scaling model size.

---

## 1. Executive Summary
This paper asks whether spending more computation at test time (during inference) can improve large language model (LLM) performance as much as, or more than, training a much larger model. It introduces a “compute‑optimal” policy that adaptively chooses how to use inference‑time compute on a per‑question basis and shows on the MATH benchmark that this adaptive use of compute can (a) beat standard best‑of‑N sampling with roughly 4× less test‑time compute (Figures 4 and 8) and (b) sometimes outperform a ∼14× larger model in a FLOPs‑matched comparison (Figure 9).

## 2. Context and Motivation
- Problem addressed
  - LLMs often improve when you let them “think longer” (use more tokens or more search) during inference, but it is unclear how to best spend that extra compute and when it is more effective than training bigger models.
  - The paper focuses on two families of inference‑time methods:
    - Improving the “proposal distribution” (the model’s own answers) by training the model to revise its answers iteratively (“revisions”).
    - Searching over answers using a verifier that scores candidates, with both outcome‑based verifiers (ORMs) and process‑based verifiers (PRMs) that score each reasoning step (Sections 2, 5, 6).
- Why this matters
  - If inference‑time compute can substitute for model size, smaller models could deliver strong reasoning on devices or in cost‑sensitive deployments, and agentic systems could self‑improve with less human supervision (Introduction).
- Prior approaches and gaps
  - Best‑of‑N: sample N answers and pick the “best” via a verifier [7, 22]. Simple but may be inefficient and not adaptive to question difficulty.
  - Self‑critique / iterative refinement can help, but results are mixed, especially on harder reasoning tasks [15, 37, 43].
  - Process‑reward models (PRMs) enable step‑level scoring and search [22, 45], but scaling behavior across budgets and task difficulty has been unclear.
- Positioning
  - The paper unifies inference‑time methods through a proposer‑verifier lens (Section 2) and introduces an adaptive “compute‑optimal” strategy that chooses how to spend test‑time compute per prompt based on an estimated difficulty signal (Sections 3.1–3.2).

## 3. Technical Approach
The work has three pillars: (A) a proposer‑verifier framework, (B) a per‑prompt compute‑optimal policy based on difficulty estimation, and (C) concrete implementations for both verifier‑guided search and proposal‑refinement via revisions.

A. Proposer‑Verifier view (Section 2)
- Proposal distribution
  - The base LLM generates candidate solutions. The paper modifies this distribution at inference time by training a `revision model` that takes previous attempts and produces an improved revision (Section 6.1).
- Verifier
  - A model that scores solutions for selection or search. Two types:
    - `ORM` (Outcome Reward Model): scores only the final answer.
    - `PRM` (Process Reward Model): scores each intermediate reasoning step, enabling tree/search methods (Section 5).

B. Compute‑optimal scaling policy (Sections 3.1–3.2)
- Goal
  - For a given prompt `q`, compute budget `N`, and method hyperparameters `θ` (e.g., how many parallel samples vs. sequential revisions; beam width; lookahead depth), choose the `θ` that maximizes accuracy.
- Formalization (Equation 1)
  - Define `Target(θ, N, q)` as the output distribution induced by method `θ` under budget `N`. Choose `θ*(N)` that maximizes the chance of sampling the correct answer for `q`.
- Practical approximation: predict difficulty, then pick `θ`
  - Difficulty “bins” (five levels) are computed per question. Two variants:
    - Oracle difficulty: uses pass@1 from 2048 samples with ground‑truth checking (used to study scaling behavior; not available at deployment).
    - Model‑predicted difficulty: average PRM final‑answer scores over 2048 samples; avoid ground truth but still compute‑heavy (Section 3.2).
  - For each difficulty bin and budget, select the best `θ` on validation (2‑fold cross‑validation within the test set prevents leakage), then apply it to the held‑out fold (Section 3.2).

C. Concrete methods

1) PRM search (Section 5)
- PRM training without human labels (Section 5.1; Appendix D)
  - For each sampled step, estimate its “value” via 16 Monte‑Carlo rollouts using the same base model (soft labels reflecting future correctness probability). Train a binary classifier (PaLM 2‑S* backbone) with BCE loss (AdamW, lr=3e‑5, batch 128, dropout=0.05). 16 samples per question for training; filter invalid final answers.
- Aggregation when using PRM as a selector (Section 5.1; Appendix E)
  - Step‑wise aggregation: use the PRM’s prediction at the last step as the full‑answer score (“PRM last”), which outperforms taking min or product across steps (Figure 13).
  - Inter‑answer aggregation: “best‑of‑N weighted” (sum PRM scores for all solutions with the same final answer) per Li et al. [21].
- Search algorithms (Section 5.2; Figure 2)
  - Best‑of‑N weighted: sample N complete solutions; pick by PRM‑weighted vote.
  - Beam search: at each step, keep the top `N/M` prefixes (by PRM value), expand each with `M` next‑step samples; repeat up to 40 expansions; finalize N complete candidates and pick via PRM‑weighted vote.
  - Lookahead search: like beam search but, for each candidate step, roll out `k` steps with temperature 0 to get a better value for the current step. Cost is higher: if generation budget is `N`, cost is `N × (k+1)` samples. Viewed as MCTS without exploration when the value function is fixed.

2) Proposal refinement with a revision model (Section 6)
- Data generation (Section 6.1; Appendix H)
  - Sample 64 answers per training question at higher temperature. Post‑hoc construct “revision trajectories” by pairing each correct answer with up to four incorrect but similar answers (chosen via character‑level edit distance) as in‑context attempts the model should fix. Train with supervised fine‑tuning (AdamW, lr=1e‑5, batch 128).
- Inference‑time use (Sections 6.1–6.2; Figures 5–7)
  - Sequential revisions: generate a chain where each revision conditions on previous attempts (truncate to most recent four attempts if chain grows longer).
  - Parallel sampling: generate independent attempts in parallel (like best‑of‑N).
  - Selection within and across chains:
    - Verifier‑based: a dedicated `ORM` trained on the revision model’s outputs (Appendix J) selects the best answer first within each chain (best‑of‑N weighted), then across chains.
    - Majority voting: take all produced final answers and choose the majority (no hierarchical voting).
  - Hybrid allocation: split the generation budget between parallel attempts and sequential revisions. The paper sweeps the ratio to find compute‑optimal mixes.

D. Test‑time vs. pretraining compute trade‑off (Section 7)
- FLOPs accounting
  - Pretraining FLOPs: `X = 6 N D_pretrain`
  - Inference FLOPs: `Y = 2 N D_inference` (from [29])
  - Scaling parameters by `M` multiplies both `X` and greedy inference `Y` by `M`.
  - To match FLOPs with a smaller model using extra test‑time compute, they scale inference compute by a factor dependent on `R = D_inference / D_pretrain` (Section 7).
- Comparison protocol
  - Compare a PaLM 2‑S* model using compute‑optimal test‑time policies vs. a ∼14× larger model run greedily, at FLOPs‑matched points for three regimes of `R`: `R << 1`, `R ≈ 1`, `R >> 1` (Figure 9).

## 4. Key Insights and Innovations
- Adaptive compute‑optimal policy tied to difficulty (Sections 3.1–3.2; Figures 4 and 8)
  - Novelty: instead of a fixed test‑time strategy for all prompts, choose per‑prompt hyperparameters (e.g., search type, beam width; sequential vs. parallel ratio) using a predicted difficulty bin.
  - Significance: With the same or less compute, this adaptive policy approaches or beats strong baselines. For both PRM search and revisions, it nearly matches or exceeds best‑of‑N using roughly 4× fewer generations (e.g., 16 vs. 64), especially at low to medium compute budgets (Figure 4) and sustains gains at higher budgets for revisions (Figure 8).
- When to search vs. when to revise (Sections 5.3, 6.2; Figures 3 and 7)
  - Insight: Search (beam) helps more on harder questions and at low budgets; sequential revisions help more on easier questions and at larger budgets. Over‑optimization with search can hurt on easy items at high budgets (Figure 3, right).
  - Impact: Guides practitioners to route compute: use search to explore on hard problems, and revisions to exploit and polish on easy/medium ones.
- PRM design choices that matter (Section 5.1; Appendix E)
  - The “last‑step” aggregation for PRM scoring beats min/product over steps (Figure 13). This departs from previous defaults and yields stronger selection and search.
- FLOPs‑matched exchange rate results (Section 7; Figure 9; Figure 1 right)
  - Finding: For easy/medium questions or when the inference load is small (`R << 1`), test‑time compute on a small model can outperform training a ∼14× bigger model; for the hardest questions or large inference loads (`R >> 1`), more pretraining (bigger model) is preferable.
  - Example numbers from summary plots (Figure 1, right): revisions show relative gains up to +27.8% in favorable regimes and large drops (e.g., −24.3%) for hard items under high `R`. PRM search shows +19.1% in the best case but large negatives (e.g., −52.9%) in unfavorable regimes.

## 5. Experimental Analysis
- Setup
  - Dataset: MATH (12k train, 500 test; high‑school competition problems) with the split from Lightman et al. [22] (Section 4).
  - Base model: PaLM 2‑S* (Codey), fine‑tuned into a revision model and separate verifiers (Section 4).
  - Metrics: MATH test accuracy; pass@1 analysis for revisions; generation budget measured in “samples”; budgets up to 256 (Figures 3, 6–8).
  - Difficulty bins: 5 quantiles per question via oracle correctness or PRM‑predicted scores (Section 3.2). Two‑fold cross‑validation within the test set to avoid tuning on the same fold (Section 3.2).
- Main quantitative results
  - PRM search vs. best‑of‑N (Section 5.3; Figure 3 left)
    - At low generation budgets, beam search outperforms best‑of‑N; as budget grows, gains diminish and can fall below best‑of‑N. Lookahead search underperforms at equal budgets due to extra compute cost in rollouts.
  - Difficulty‑conditioned behavior (Figure 3 right)
    - Easy bins (1–2): beam search degrades with more compute (over‑exploits PRM biases); best‑of‑N remains stable or better.
    - Medium bins (3–4): beam search consistently improves over best‑of‑N as budget increases.
    - Hardest bin (5): little progress from any method.
  - Compute‑optimal PRM search (Figure 4)
    - “Compute‑optimal” policy (both oracle and predicted difficulty) nearly matches best‑of‑N using up to 4× fewer generations in the low‑budget regime (e.g., 16 vs. 64).
    - With higher budgets, oracle‑difficulty policies continue to improve; predicted‑difficulty policies see reduced gains but remain competitive.
  - Revision model behavior
    - Pass@1 improves with each revision step beyond training horizon (trained with up to 4 revisions, but keeps improving out to ~64 steps; Figure 6 left).
    - Sequential revisions vs. parallel sampling: sequential slightly but consistently better under both verifier‑based and majority selection (Figure 6 right).
    - Optimal sequential/parallel mix depends on budget and difficulty (Figure 7):
      - For a fixed budget, there is an ideal ratio; easy questions favor fully sequential; harder questions favor a balance (right panel at budget 128).
    - Compute‑optimal revisions (Figure 8): with difficulty‑conditioned ratio, outperforms best‑of‑N using up to 4× less compute (e.g., 64 vs. 256), with continued improvements at high budgets where parallel baselines plateau.
  - Test‑time vs. pretraining compute (Figure 9; Figure 1 right)
    - FLOPs‑matched comparisons at three `R` regimes show:
      - For easy/medium bins or `R << 1`, test‑time compute (both revisions and PRM search) is often better than training a ∼14× larger model.
      - For the hardest bin or `R >> 1`, scaling pretraining is more effective.
    - Summary numbers shown in Figure 1 (right):
      - Revisions: relative gains up to +27.8% in best regimes, but large negatives on hard questions at high `R` (−24.3%).
      - PRM search: best case +19.1%, but sizable negatives (e.g., −52.9%) in unfavorable regimes.
- Ablations and diagnostics
  - PRM aggregation (Appendix E; Figure 13): “last‑step” > min/product.
  - PRM vs. ORM (Appendix F; Figure 14): the PRM outperforms an ORM baseline at larger N.
  - Revision verifier adaptation (Appendix J; Figure 15): an ORM trained on revision outputs outperforms using the PRM trained on base LM outputs; including revision history in the verifier context gives a small extra boost.
  - Alternative revision training with ReSTEM (Appendix K; Figure 16): this simplified RL optimization harmed sequential revision performance, likely due to spurious correlations in on‑policy data.
  - Failure/exploitation examples (Appendix M; also Section 5.3): search sometimes finds short or low‑information repetitive steps, signaling PRM over‑exploitation.

Assessment
- The experiments are thorough for the targeted setting (MATH) and carefully analyze compute scaling as a function of difficulty and budget. The two‑fold cross‑validation for difficulty‑conditioned policy selection limits overfitting (Section 3.2).
- Claims are conditional and nuanced:
  - The 4× efficiency claim is evidenced by Figures 4 and 8 (compute‑optimal vs. best‑of‑N at 16 vs. 64 or 64 vs. 256).
  - The “test‑time can beat 14× larger model” claim holds in specific regimes (Figure 9), but not on the hardest questions or when `R` is large.

## 6. Limitations and Trade-offs
- Reliance on capability‑specific fine‑tuning (Footnote 1; Section 4)
  - The base model is fine‑tuned for revisions and for PRM/ORM verification; off‑the‑shelf LLMs may lack these capabilities on math today. Future pretraining could reduce this need, but current results depend on it.
- Difficulty estimation cost (Section 3.2; Appendix C)
  - Model‑predicted difficulty uses many samples (2048) per question, which is itself compute‑heavy. The experiments do not charge this cost to the test‑time budget (a pragmatic simplification), so real‑world deployments need cheaper difficulty predictors.
- Limited domain and metrics
  - All results are on the MATH benchmark; generalization to other reasoning domains (code, science, multimodal) is not shown.
- PRM brittleness at high optimization pressure (Section 5.3; Figure 3 right; Appendix M)
  - Search can over‑exploit verifier quirks (short or repetitive steps), hurting easy‑question performance as compute grows. Lookahead search did not help at equal compute budgets.
- Hardest problems remain unsolved (Figures 3, 9)
  - For bin 5, gains from test‑time compute are small; increasing pretraining compute works better.
- Compute accounting assumptions (Section 7)
  - FLOPs formulas (e.g., `6ND_pretrain`, `2ND_inference`) approximate costs and ignore overheads such as KV‑cache costs or verifier‑scoring cost outside generation tokens, which may matter in practice.
- Distribution shift across modules
  - A PRM trained on base‑LM outputs transfers poorly to revision‑model outputs (Appendix J; Figure 15 left), necessitating an extra ORM for revised solutions.

## 7. Implications and Future Directions
- Practical guidance for deploying reasoning LLMs
  - Adapt test‑time compute per prompt difficulty. Route easy questions to sequential revisions; route hard ones to search (beam) with conservative budgets. This can save ∼4× inference compute without losing accuracy (Figures 4, 8).
  - When inference workload is small (`R << 1`) or problems are mostly easy/medium, prefer extra test‑time compute on a smaller model rather than training a much larger model (Figure 9).
- Systems and product impact
  - On‑device or cost‑sensitive deployments may get strong reasoning by spending more inference compute adaptively instead of hosting very large models.
  - Agentic/self‑improving systems can use these policies to allocate compute among search, revision, and selection steps.
- Research directions
  - Cheaper and more accurate difficulty prediction (Section 8): train a dedicated difficulty predictor; learn to interleave “assess difficulty” with “solve” to balance exploration vs. exploitation.
  - Stronger verifiers and safer optimization: improve PRM robustness to over‑optimization; calibrate step‑scores; regularize search to avoid degenerate short or repetitive solutions (Section 5.3; Appendix M).
  - Combining axes of test‑time compute (Section 8): integrate PRM tree search with revision models and critique‑and‑revise pipelines; study more sophisticated adaptive policies that decide when to switch between modes.
  - Distillation loops: feed the products of compute‑optimal inference back into training (self‑improvement with distillation), closing the loop between test‑time and train‑time compute (Section 8).
  - Broader benchmarks: replicate analyses on code reasoning, scientific QA, and multimodal tasks to establish generality.

> Bottom line: Section 3 formalizes a per‑prompt, budget‑aware selection of inference strategies; Sections 5–6 show how to instantiate it with PRM search and revision models; Figures 4 and 8 demonstrate 4× efficiency gains over best‑of‑N; and Figure 9 maps the regimes where test‑time compute can beat scaling parameters by 14×. The approach reframes “thinking longer” at inference from a blunt tool into a policy decision that can rival, and sometimes replace, brute‑force model scaling.
