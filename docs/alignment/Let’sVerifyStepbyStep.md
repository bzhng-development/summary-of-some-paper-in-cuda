# Let’s Verify Step by Step

**ArXiv:** [2305.20050](https://arxiv.org/abs/2305.20050)

## 🎯 Pitch

This paper introduces 'process supervision' for training reward models that evaluate each step of a language model's reasoning, rather than just the final answer. On the challenging MATH dataset, process-supervised models dramatically outperform traditional outcome-supervised ones in reliably identifying correct solutions, offering better data efficiency and interpretability. By releasing a large human-labeled dataset and demonstrating process supervision’s superiority at scale, this work sets a new standard for building safer, more robust, and more transparent reasoning systems—a critical advancement for trustworthy AI.

---

## 1. Executive Summary
This paper introduces process supervision for training reward models that judge each step of a model’s reasoning rather than only the final answer. On challenging math problems (MATH dataset), a process-supervised reward model (`PRM`) markedly outperforms an outcome-supervised model (`ORM`) in selecting correct solutions from many candidates and does so with better data efficiency and interpretability.

## 2. Context and Motivation
- Problem addressed
  - Large language models can generate multi-step solutions (chain-of-thought), but a single incorrect step can derail the entire reasoning. Detecting such errors is hard if supervision only evaluates the final answer.
  - Two supervision paradigms are compared:
    - `Outcome supervision` (ORM): label each whole solution as correct/incorrect based on the final answer.
    - `Process supervision` (PRM): label each intermediate step in the solution as correct/incorrect (plus a “neutral” option).
- Why important
  - Real-world tasks often involve long reasoning chains; final-answer correctness alone can be misleading when models get the right answer for the wrong reasons.
  - Reliable reward models are crucial because they are used to guide search (rejection sampling) or to train models via reinforcement learning; if the reward model is unreliable, the overall system fails (Section 1).
- Prior approaches and gaps
  - Outcome-supervised verifiers and reward models are common and work reasonably well but struggle with credit assignment: locating which step went wrong in long solutions.
  - Prior work on process vs outcome feedback (Uesato et al., 2022) found similar end performance on easier grade-school math. This left open whether process supervision scales and whether it can surpass outcome supervision on harder domains.
- Positioning
  - This work scales up process supervision on a harder benchmark (MATH), uses a stronger base model (GPT‑4 family), collects a large human-labeled step dataset (`PRM800K`), evaluates at large scale, and performs controlled small-scale ablations with synthetic supervision to disentangle confounds (Sections 2–4).

## 3. Technical Approach
The pipeline has four main components.

- Generator (Section 2.3)
  - `Generator` = the model that produces candidate solutions. It is finetuned to format its outputs as newline-delimited steps so that step boundaries are easy to parse.
  - Training of the generator is minimal and focused on formatting: few-shot solutions are generated, filtered to correct ones, and the base model is finetuned for one epoch so it consistently emits step-by-step solutions.

- Data and base models (Section 2.2, Appendix A)
  - Large-scale models are finetuned from a base GPT‑4 model (pretrained by next-token prediction only).
  - Small-scale models share the same architecture family but are pretrained with ~200× less compute.
  - All models do a light additional pretraining on a 1.5B-token math corpus `MathMix` to improve mathematical reasoning.

- Process-supervised data collection (Sections 2.4, 2.6; Appendix B, D)
  - Human labelers see a full candidate solution and label each step `positive` (correct/reasonable), `negative` (incorrect/unreasonable), or `neutral` (ambiguous/technically valid but potentially unhelpful).
  - Labeling stops at the first incorrect step. This keeps costs similar to outcome grading of wrong solutions and ensures a fair comparison: for correct solutions, both paradigms agree that all steps are correct; for incorrect ones, both indicate at least one error, while PRM also pinpoints where it happens (Section 2.6).
  - Active learning strategy (Section 2.4): the system prefers to surface “convincing wrong-answer solutions” to labelers—
    - `Convincing` = ranked highly by the current PRM.
    - `Wrong-answer` = final answer mismatches the ground truth (checked automatically).
    - The PRM is periodically retrained with the latest labeled data to iteratively improve which samples are shown.

- Reward models and scoring (Sections 2.5–2.6; Appendix E–F)
  - ORM training
    - Uniformly sample solutions per problem from the generator.
    - Use final-answer checking to label each whole solution as correct/incorrect.
    - Train a token-level verifier to predict the solution’s correctness; at test time, use the score at the final token as the solution score (Appendix E).
    - Caveat: final-answer checking can mislabel “lucky” solutions that reach a correct answer via flawed reasoning (false positives).
  - PRM training
    - Train a classifier at each step boundary to predict `positive/negative/neutral` for that step (Appendix F.1). Low learning rates are important for stability because this is a distribution shift from language modeling.
    - Test-time reduction to a single solution score: compute per-step probabilities that the step is correct and combine across steps.
      - The best strategy is to multiply step-level correctness probabilities and treat neutral steps as positive (Appendix F.2; Table 4). This effectively estimates “the probability that all steps are correct.”
  - Search procedure (`best-of-N`)
    - For each test problem, sample N solutions from the generator.
    - Rank them using ORM or PRM scores.
    - Select the top-ranked solution, grade it by final answer, and report accuracy as “% solved.” This isolates reward-model reliability in searching among many candidate solutions (Section 2.1).

- Synthetic supervision at small scale (Section 4; Appendix H)
  - To run controlled ablations without expensive human labels, the best large PRM (`PRMlarge`) is used as a labeling oracle:
    - For process supervision: use `PRMlarge` to label individual steps and stop at the first predicted mistake.
    - For outcome supervision: label the entire solution as correct only if `PRMlarge` thinks every step is correct (a thresholded rule; Appendix H).
  - An auxiliary small PRM (`PRMselector`) is also trained on 1 sample/problem, then used to score 1000 samples/problem to implement an active-learning style selection for later training (Section 4.2).

Key definitions used throughout
- `Reward model (RM)`: a model that scores candidate solutions by “how good” they are; higher is better.
- `Outcome-supervised RM (ORM)`: an RM trained with final-answer labels for whole solutions.
- `Process-supervised RM (PRM)`: an RM trained with step-level labels.
- `Best-of-N search`: sample N solutions, score them, pick the top one.
- `Majority voting`: pick the answer most frequently produced by N samples (ignores chain-of-thought).
- `Convincing wrong-answer`: a wrong final answer that the current PRM scores highly—useful for finding where the PRM is fooled.
- `PRM800K`: the released dataset with ~800k step-level labels across 75k solutions for 12k problems (Section 2.4; Appendix B).
- `MathMix`: a 1.5B-token math-focused pretraining corpus (Appendix A).

## 4. Key Insights and Innovations
- Process supervision scales and surpasses outcome supervision on hard math
  - Novelty: The study shows a strong, widening gap in favor of PRM when searching over many candidate solutions on MATH—a harder benchmark than prior comparisons (Section 3).
  - Significance: At `best-of-1860`, PRM solves 78.2% vs ORM’s 72.4% and majority voting’s 69.6% (Figure 3, table above the plot).
  - The advantage holds across difficulty levels and grows with more samples (Appendix G, Figure 6).

- Active learning makes process supervision more data-efficient
  - Innovation: Use of “convincing wrong-answer” sampling plus periodic PRM retraining to focus human labeling effort where the model is most confused (Section 2.4).
  - Result: In small-scale synthetic experiments, this strategy yields roughly 2.6× better data efficiency than uniform labeling (Figure 4a; slope comparison of lines of best fit).

- Synthetic supervision methodology for controlled comparisons
  - Contribution: Using `PRMlarge` as a teacher enables apples-to-apples comparisons of process vs outcome supervision and ablations of data selection strategies—otherwise infeasible with human labeling costs (Section 4).
  - Finding: Even when outcome labels are provided by a strong `PRMlarge` (so final-answer “false positives” are reduced), process supervision still consistently wins across dataset sizes (Figures 4a–4b).

- Public dataset release enabling reproducibility and follow-up work
  - `PRM800K` is released, containing the complete set of step-level labels used to train the best PRM (Abstract; Appendix B).
  - Scope and composition: ~800k step labels over ~75k solutions; active learning–shaped distribution with many wrong-answer solutions but numerous correct steps (Appendix B; Table 3).

- Interpretability and alignment relevance
  - Process supervision reduces credit-assignment ambiguity by specifying where errors occur (Section 6.1).
  - It directly rewards good reasoning rather than only good outcomes, which can mitigate misaligned strategies and decrease the “alignment tax” (Section 6.2).

## 5. Experimental Analysis
- Evaluation setup and datasets
  - Primary benchmark: MATH (Hendrycks et al., 2021).
    - The PRM’s training uses `PRM800K`, which includes labels from both MATH train and 4.5k MATH test problems; to avoid contamination, evaluation is on a uniformly sampled subset of 500 held-out MATH test problems (Appendix C; Figure 5 shows this subset matches the original test set’s difficulty and subject distributions).
  - Out-of-distribution (OOD) test: 224 fresh problems from AP Calculus, AP Chemistry, AP Physics, AMC10/12 exams released after pretraining (Section 5; Table 1).
  - Metrics: “% Problems Solved (Best-of-N)” where N varies, using final-answer grading to mark correctness (Sections 2.1, 3).

- Main quantitative results
  - Large-scale comparison (Section 3; Figure 3)
    - Quote:
      > At best-of-1860, ORM = 72.4%, PRM = 78.2%, Majority Voting = 69.6%.
    - The PRM’s advantage widens as N increases, indicating better search capability among many candidates.
    - Attempts to combine PRM with majority voting (RM-weighted voting) bring no noticeable improvements (caption of Figure 4; discussion after Figure 3).
  - Difficulty breakdown (Appendix G; Figure 6)
    - PRM outperforms ORM across all quintiles of generator pass rate. For the easiest quintile, ORM can be fooled as N grows (performance slightly drops), while PRM remains robust.
    - The benefit of increasing N is largest for the hardest quintile, consistent with requiring more samples to find a valid reasoning chain.
  - OOD generalization (Section 5; Table 1)
    - Quote (aggregate over 234 problems):
      > Aggregate Best-of-100: ORM = 63.8%, PRM = 72.9%, Majority Voting = 61.3%.
    - Per subject, PRM leads in all: e.g., AP Calculus 86.7% vs ORM 68.9%; AP Physics 86.7% vs 77.8%.
    - This shows the PRM maintains a strong edge under modest domain shift.
  - Scoring strategy ablation (Appendix F.2; Table 4)
    - Treating neutral as positive and multiplying step scores yields the best 78.2%, but all four strategies are close (77.4–78.2%), indicating robustness to the scoring reduction.
  - Small-scale, synthetic supervision (Section 4; Figures 4a–4b)
    - With matched datasets and teacher supervision from `PRMlarge`, process supervision consistently outperforms outcome supervision across 1–200 samples/problem (Figure 4a).
    - Even when the outcome labels come from `PRMlarge` (mitigating final-answer false positives), process supervision remains superior (Figure 4b).
    - Active learning yields ~2.6× data efficiency over uniform selection (Figure 4a). An attempt to iteratively retrain the selector during data collection showed instability and no clear gains (Section 4.2).
  - Qualitative visualizations (Section 2.6; Figure 2; Appendix I)
    - Figure 2 shows side-by-side solutions where the PRM highlights (green/red) steps and isolates the precise step error in the incorrect solution.
    - Appendix I collects true positives/negatives and false positives: typical PRM misses include subtle counting mistakes or algebraic slips that “look” reasonable.

- Do the experiments support the claims?
  - Yes, convincingly in this domain:
    - Multiple scales (large human-labeled, small teacher-labeled) point to the same conclusion: process supervision produces more reliable reward models for best-of-N search.
    - The widening gap with larger N (Figure 3) strongly suggests PRM is better at ranking many candidate chains-of-thought.
    - OOD results (Table 1) reduce concerns that the gains are idiosyncratic to the MATH test distribution.

## 6. Limitations and Trade-offs
- Domain scope
  - Experiments focus on math with checkable final answers. The approach’s benefits in domains without unambiguous final answers remain to be demonstrated (Sections 2.1, 6.2).
  - Process labels stop at the first error (Section 2.6). While fair for comparison and labeling cost, this omits supervision on later steps, which might matter in tasks where recovery from errors is important.
- Data and bias considerations
  - `PRM800K` is actively curated toward convincing wrong answers. While this improves PRM training, it means the ORM cannot be fairly trained on the same set without mixing uniform samples (Section 3). The paper addresses this with separate small-scale matched-data experiments, but the large-scale results remain incomparable in training distribution specifics.
  - Automatic final-answer grading for ORM has known imperfections (false positives) that can degrade ORM targets (Section 2.5).
- Stability and scalability of active learning
  - Iterative retraining of the selector (`PRMselector`) during collection was unstable in preliminary tests (Section 4.2).
  - Active learning might reduce sample diversity when selection approaches the pool size (200 selected out of 1000), slightly undercutting gains (Section 4.2).
- Potential test contamination
  - The MATH test set likely exists online; although precautions were taken (Appendix A decontamination; Section 6.3 discussion), undetected overlap could inflate absolute scores. The relative comparisons are less likely to be affected, and OOD results mitigate this concern (Section 6.3).
- Computation and data cost
  - Collecting step-level human labels is expensive; while active learning improves efficiency (~2.6×), deploying this widely requires significant infrastructure and labeling expertise (Sections 2.4, 4.2).
- Scoring bias with product reduction
  - Multiplying step probabilities slightly penalizes longer solutions (Appendix F.2). Table 4 suggests minimal effect here, but different domains with longer chains might be more sensitive.

## 7. Implications and Future Directions
- How this changes the landscape
  - It provides strong evidence that supervising the reasoning process—not just outcomes—yields reward models that better guide search through chains-of-thought, especially as the number of candidate solutions grows (Figure 3).
  - For alignment and safety, process supervision directly rewards transparent, human-endorsed reasoning steps, which may reduce incentives for shortcut strategies and lowers the “alignment tax” (Section 6.2).
- Enabled follow-ups
  - Integrate PRMs into reinforcement learning to improve the generator itself, not just search over its samples (Section 2.1 notes this as a natural next step).
  - Explore iterative, stable active-learning loops with stronger theoretical grounding and better selectors (Section 4.2).
  - Extend to domains beyond math: programming, scientific reasoning, medical decision-making—especially where final answers are ambiguous or uncheckable.
  - Go beyond “first error” supervision to shape entire reasoning trajectories, including error recovery and robustness to detours (Section 2.6 suggests this would give process supervision an even larger information advantage).
  - Improve calibration and uncertainty estimation of PRMs to reduce the remaining false positives/negatives observed in Appendix I.
- Practical applications
  - Deploy PRM-guided search for math tutoring, automated theorem proving assistance, and educational tools where step-by-step validation is critical.
  - Use PRM800K as a benchmark and pretraining resource for step-level feedback models across math-related tasks.

Overall, the paper demonstrates—empirically and at scale—that supervising the reasoning process enables more reliable selection among chains-of-thought than supervising outcomes alone. The open-sourced PRM800K dataset and the synthetic supervision methodology provide a foundation for broader, reproducible research on reasoning-aware reward models.
