# Scaling LLM Test‑Time Compute Optimally can be More Effective than Scaling Model Parameters

**ArXiv:** [2408.03314](https://arxiv.org/abs/2408.03314)
**Authors:** Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar
**Institutions:** Google DeepMind, UC Berkeley

## 🎯 Pitch

This paper introduces an innovative compute-optimal policy for large language models (LLMs) that dynamically allocates inference-time resources based on prompt difficulty, achieving up to 4× compute efficiency over traditional best-of-N sampling. By effectively balancing search and self-revision strategies, this approach not only reduces computational costs but demonstrates that smaller LLMs with adaptive test-time compute can outperform much larger models in FLOPs-matched scenarios, revolutionizing model deployment strategies.

---

## 1. Executive Summary
This paper presents a principled way to spend extra inference-time computation (“test-time compute”) on large language models (LLMs) so that each prompt gets the kind of extra thinking it most benefits from. It introduces a compute-optimal policy that adaptively chooses between search against a verifier and iterative self-revision, showing that this adaptive allocation can match or beat standard best-of-N sampling with up to 4× less compute (Figures 4 and 8), and that, in FLOPs-matched settings, test-time compute with a smaller model can outperform simply using a model with ~14× more parameters on many problems (Figure 9).

## 2. Context and Motivation
- Problem the paper tackles
  - LLMs improve when they “think longer” at inference, but we lack a clear recipe for how to spend this extra compute on each prompt most effectively. Prior test-time strategies (e.g., best-of-N) show mixed results, especially on complex reasoning tasks like math (Section 1).
- Why this matters
  - Practical: If extra test-time compute can substitute for pretraining or parameter count, organizations can deploy smaller, cheaper models on-device or in latency-tolerant settings (Sections 1 and 7).
  - Scientific: Understanding how inference-time reasoning scales clarifies when search, self-correction, or other mechanisms actually help (Sections 2 and 3).
- What existed before and their gaps
  - Best-of-N: sample N full responses and pick the best via a verifier or rule (Section 1). Simple but may waste compute on easy prompts and is often insufficient on hard ones.
  - Verification: outcome reward models (ORMs) score only final answers; process reward models (PRMs) score each step, enabling tree search but are sensitive to over-optimization (Sections 2 and 5).
  - Self-correction/self-refinement: prompting an LLM to critique and revise its output; effective in some settings, weak or inconsistent on math reasoning without finetuning (Sections 1 and 6).
- How this paper positions itself
  - Unifies test-time methods via a “proposer–verifier” view (Section 2).
  - Studies two orthogonal knobs:
    1) Improve the verifier and search against it (Section 5).
    2) Improve the proposal distribution by training an LLM to iteratively revise its own answer (Section 6).
  - Adds an adaptive selection mechanism that chooses the best knob settings per prompt difficulty and compute budget (Sections 3.1–3.2).

## 3. Technical Approach
The core idea is to allocate the available test-time compute to the method that will most help the specific prompt. The paper formalizes and operationalizes this in four steps.

1) A unified “proposer–verifier” lens (Section 2)
- Proposal distribution: how the LLM produces candidate answers given a prompt. It can be modified:
  - Input-side, by adding guidance tokens (e.g., instructions to revise).
  - Output-side, by sampling many candidates and post-selecting.
- Verifier: a learned scorer that estimates correctness and helps pick among candidates.
  - ORM scores only final answers.
  - PRM scores each reasoning step, enabling search over partial solutions.

2) Compute-optimal scaling as a selection problem (Section 3.1; Eq. 1)
- Goal: for a prompt `q` and compute budget `N`, choose hyperparameters `θ` (e.g., best-of-N vs beam search; number of revisions vs parallel samples) to maximize expected correctness of the final answer.
- Formalization (Eq. 1): choose `θ*_q(N)` that maximizes the probability that a sample from the induced distribution `Target(θ, N, q)` equals the ground truth.
- Intuition: treat “how to use compute” as a per-prompt decision with a budget that can be split among methods.

3) Estimating prompt difficulty and mapping difficulty → policy (Section 3.2)
- Why difficulty: The best use of compute depends on whether a prompt is easy (already on the right track) or hard (requires exploring new strategies).
- Two difficulty estimators:
  - Oracle difficulty: bin questions by pass@1 rate of the base LLM, computed from 2048 samples per question (requires ground-truth checking; used for analysis).
  - Predicted difficulty: bin by averaging a verifier’s scores over 2048 samples per question—no ground-truth answers required (Section 3.2). The curves using predicted difficulty largely overlap the oracle curves (Figure 4).
- Practical note: Difficulty estimation itself costs compute. The paper treats this cost as amortizable or overlapping with verifier use and defers efficient estimation to future work (Section 3.2).

4) Two concrete axes for spending test-time compute

A. Search against a PRM verifier (Sections 5.1–5.3)
- PRM training (Section 5.1; Appendix D)
  - Labels per-step “value” without human annotators by Monte Carlo rollouts from each step: generate many continuations, estimate the probability of eventually reaching a correct final answer, and train a step-level regressor (PRM) to predict this value.
- Aggregation to pick a final answer (Section 5.1; Appendix E)
  - Step-wise: score an entire solution by the PRM’s last-step prediction (“PRM last”). This outperforms “min” or “product” aggregations across steps (Figure 13).
  - Inter-answer: “best-of-N weighted” (sum scores over all solutions sharing the same final answer) rather than standard best-of-N; this helps when the same final answer is reached via multiple derivations (Section 5.1).
- Search methods compared (Section 5.2; Figure 2)
  - Best-of-N weighted: sample N full solutions independently and pick the best.
  - Beam search: maintain N candidates at each step; keep top-M by PRM score and expand them.
  - Lookahead search: at each step, simulate k steps with temperature 0 to get a more accurate value estimate; cost N × (k + 1) “generations” (Section 5.2). Think of it as MCTS without exploration noise.
- Compute budget accounting
  - A “generation” is one model sample of a step or full answer. Lookahead adds k more generations per step.

B. Refining the proposal distribution with a revision model (Section 6)
- Training data construction (Section 6.1; Appendix H)
  - Sample 64 solutions per question in parallel at high temperature.
  - For each correct solution, pair it with up to 4 incorrect but “similar” solutions (chosen by character-level edit distance) to form synthetic multi-turn revision trajectories.
  - Supervised finetuning teaches the model to produce the final correct answer given the previous attempts as context.
- Inference-time usage (Section 6.1)
  - Sequential revisions: generate a chain of revised answers, each conditioned on the last few attempts; accuracy at each step improves as chains get longer (Figure 6 left).
  - Selection within and across chains: use a verifier (ORM trained on the revision model’s outputs; Appendix J) with best-of-N weighted selection, or use majority voting over final answers (Section 6.1; Appendix I).
- Parallel vs sequential sampling (Section 6.2; Figure 5)
  - Parallel: generate N independent answers.
  - Sequential: generate N revisions in a single chain.
  - Hybrid: split budget; e.g., allocate √N to initialize multiple chains (parallel) and √N to deepen each chain (sequential).
  - There exists a compute-optimal split that depends on both total budget and difficulty (Figure 7).

5) FLOPs-matched tradeoff with pretraining (Section 7)
- FLOPs approximations: pretraining FLOPs `X = 6 N D_pretrain`; inference FLOPs `Y = 2 N D_inference` (Section 7).
- Scaling parameters by factor `M` multiplies both training and inference FLOPs by `M`.
- To match that budget using the smaller model via test-time compute, the allowable inference-time multiplier depends on the ratio `R = D_inference / D_pretrain` (Section 7).
- The paper compares compute-optimal test-time scaling against a ~14× larger model under three R regimes: R ≪ 1, R ≈ 1, R ≫ 1 (Figure 9).

## 4. Key Insights and Innovations
- Difficulty-conditioned, compute-optimal allocation is the main innovation
  - What’s new: Instead of a one-size-fits-all test-time strategy, the method selects the best hyperparameters per prompt difficulty and compute budget (Eq. 1; Sections 3.1–3.2).
  - Why it matters: It extracts more value from the same compute. Results show near parity with best-of-N using 4× less compute for PRM search (Figure 4) and clear gains for revisions (Figure 8).
- Sequential revisions vs parallel sampling are complementary
  - Insight: Sequential revisions act like local refinement (good when initial attempts are on track); parallel sampling explores diverse approaches (good on harder problems). There is a best split between them that depends on difficulty and budget (Figure 7).
  - Significance: Provides a practical recipe for combining “think longer” styles rather than choosing one.
- PRM training and aggregation choices matter
  - Using MC-estimated per-step returns (not human PRM labels) and scoring entire solutions by the last-step PRM score performs best (Figure 13), and PRM outperforms ORM as N grows (Figure 14).
  - This is not just tuning; it changes how the verifier’s signal is exploited during search.
- When test-time compute beats bigger models
  - FLOPs-matched analysis (Section 7; Figure 9) shows that, especially for easy/medium questions or when inference load R is small, extra test-time compute with a smaller model can beat simply switching to a ~14× larger model. This reframes the pretraining vs inference budget tradeoff.

## 5. Experimental Analysis
- Setup (Section 4)
  - Dataset: MATH benchmark; split of 12k train, 500 test (from Lightman et al. [22]) spanning competition-style problems with varied difficulty.
  - Base model: `PaLM 2-S*`. The team finetunes it into two derived models: a revision-capable proposer (Section 6.1) and a PRM verifier (Section 5.1). A separate ORM is trained for selecting among revision outputs (Appendix J).
  - Metrics: main metric is MATH test accuracy; pass@1 is used for difficulty binning and step-wise revision curves (Figure 6 left).
  - Difficulty estimation: both oracle and predicted difficulty bins are used; predicted bins come from averaging PRM final-answer scores over 2048 samples per question (Section 3.2). Two-fold cross validation avoids selecting and testing policies on the same fold (Section 3.2).

- Results: PRM search (Sections 5.2–5.3)
  - Which search works when?
    - Beam search dominates at small budgets but loses to best-of-N as budget grows; lookahead search underperforms at matched budget due to extra compute cost (Figure 3 left).
    - By difficulty, beam search improves over best-of-N on medium bins (3–4), but on easy bins (1–2) aggressive search degrades performance at higher budgets, consistent with over-optimizing spurious PRM features (Figure 3 right; Appendix M shows repetitive or very short “exploited” solutions).
  - Compute-optimal PRM search
    - Quote: “Compute-optimal scaling can nearly outperform PRM best-of-N using up to 4× less test-time compute (e.g., 16 vs 64 generations)” (Figure 4).
    - Importantly, curves using predicted difficulty nearly overlap oracle-difficulty curves (Figure 4), indicating practicality without labels.

- Results: Revisions (Sections 6.1–6.2)
  - Step-wise improvement: pass@1 improves across revision steps even beyond the 4-step training horizon (Figure 6 left), indicating the model learned a robust revision behavior.
  - Sequential vs parallel: for a given budget, generating N revisions sequentially slightly beats generating N independent attempts (Figure 6 right), both with verifier selection and majority voting.
  - Optimal split depends on difficulty
    - At a fixed budget (e.g., 128 generations), easy questions do best with fully sequential compute; harder ones prefer a mixed split, proving the complementarity (Figure 7 right).
    - Across budgets, there exists an ideal sequential/parallel ratio at each compute level (Figure 7 left).
  - Compute-optimal revisions
    - Quote: “We can outperform best-of-N using up to 4× less test-time compute (e.g., 64 samples vs 256)” (Figure 8). The gap widens at higher budgets, where parallel best-of-N plateaus but compute-optimal strategies keep improving (Figure 8).

- FLOPs-matched comparison with a ~14× larger model (Section 7; Figure 9)
  - The paper compares compute-optimal curves (per difficulty) against the accuracy of a ~14× larger model (plotted as stars placed at FLOPs-equivalent x-positions for R ≪ 1, R ≈ 1, R ≫ 1).
  - Findings:
    - For easy and many medium questions, especially when inference load is small (R ≪ 1), the compute-optimal test-time approach with the smaller model beats or matches the larger model.
    - On the hardest bin or when R ≫ 1 (heavy inference per training token), scaling pretraining/parameters is more advantageous.

- Ablations and robustness checks
  - Aggregation strategies: PRM-last outperforms min/product (Figure 13).
  - PRM vs ORM: PRM wins and the gap increases with more samples (Figure 14).
  - Predicted vs oracle difficulty: similar trends for both search and revisions (Figures 4 and 11–12).
  - Revision verifier details: an ORM trained on revision outputs performs better for revision selection than the PRM trained on base-LM outputs (Figure 15 left); including revision history in the verifier’s context helps slightly (Figure 15 right).
  - Negative control: a ReSTEM-optimized revision model actually worsened with more sequential revisions, likely due to spurious correlations from on-policy data (Figure 16), underscoring the importance of data construction for revisions.

- Convincingness
  - The study is thorough for math reasoning on MATH with PaLM 2-S*, with careful budget accounting (e.g., lookahead cost N × (k + 1)) and multiple baselines. The difficulty-conditioned analyses (Figures 3, 7, 9) are particularly compelling because they reveal when and why methods help or hurt.

## 6. Limitations and Trade-offs
- Reliance on capability-specific finetuning
  - Both the revision capability and the PRM verifier require finetuning tailored to math reasoning (Section 4; footnote in Section 2). Off-the-shelf chat models may not benefit in the same way without such training.
- Cost of difficulty estimation
  - Predicted difficulty uses 2048 samples per question for scoring (Section 3.2). While the paper argues this can be amortized or reused with search, it is still substantial, and this cost is not included in the main compute comparisons (Section 3.2).
- Sensitivity to verifier exploitation
  - Beam and lookahead search can overfit PRM quirks on easy questions at large budgets, generating low-information or too-short solutions (Figure 3 right; Appendix M). This limits the robustness of aggressive search.
- Scope of evaluation
  - The study focuses on one benchmark (MATH), one base model family (PaLM 2-S*), and two test-time mechanisms (verifier search and revisions). Generalization to other tasks (e.g., code, commonsense), modalities, or more diverse models is not assessed.
- FLOPs tradeoff assumptions
  - The FLOPs model fixes data scaling and varies parameters (Section 7), whereas other pretraining regimes (e.g., data–parameter co-scaling) could yield different crossovers. Also, matching FLOPs ignores latency/throughput constraints that matter in deployments.
- Hardest problems remain hard
  - For the top difficulty bin, neither revisions nor PRM search delivers sizable gains (Figures 3 right and 9). This suggests current test-time strategies cannot fully substitute for additional pretraining or architectural improvements on truly challenging problems.

## 7. Implications and Future Directions
- Rebalancing where we spend compute
  - The results encourage shifting some budget from pretraining to test-time reasoning, particularly for deployments dominated by easy/medium tasks or for self-improvement pipelines with low R (Figure 9). Smaller models with adaptive inference can be competitive.
- Practical recipes for deployment
  - Use a difficulty estimator (or a lightweight proxy) to select:
    - For easy prompts: prioritize sequential revisions.
    - For medium prompts: do PRM-guided beam search or mixed sequential/parallel revisions.
    - For hard prompts: either spend more parallel exploration or accept that pretraining helps more.
- Research directions opened
  - Better, cheaper difficulty prediction: train a model to forecast difficulty from the prompt alone, or learn to interleave “assess difficulty” with “attempt to solve” (Section 8).
  - Combine axes more tightly: integrate PRM-guided tree search inside each revision chain (Section 8), or add critique-and-revise loops, tools, or program synthesis to the proposer.
  - Close the loop: distill the outputs of compute-heavy test-time strategies back into the base model to create an iterative self-improvement cycle (Section 8).
  - Make verifiers robust: mitigate over-optimization by regularizing PRMs, using ensembles, or designing search that avoids low-information loops.
  - Go beyond math: test the compute-optimal framework on coding, scientific QA, theorem proving, and agentic tasks where test-time compute is natural.

Overall, the paper’s main contribution is methodological: treat “how to spend test-time compute” as an adaptive decision problem tied to prompt difficulty and budget. The empirical finding that this adaptation yields 2–4× compute efficiency improvements over best-of-N (Figures 4 and 8) and can rival a ~14× larger model in FLOPs-matched settings (Figure 9) makes a strong case that inference-time scaling—done smartly—can be as important as pretraining scale.
