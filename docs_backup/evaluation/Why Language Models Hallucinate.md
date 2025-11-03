# Why Language Models Hallucinate

**ArXiv:** [2509.04664](https://arxiv.org/abs/2509.04664)
**Authors:** Adam Tauman Kalai, Ofir Nachum, Santosh S. Vempala, Edwin Zhang
**Institutions:** OpenAI, Georgia Tech

## 🎯 Pitch

This paper provides a rigorous statistical framework to explain why large language models confidently produce false statements, highlighting the need to shift evaluation practices that incentivize guessing over uncertainty. By integrating confidence targets into benchmarks, the proposed solution realigns incentives, fostering the development of more reliable models crucial for applications in fields like medicine and law.

---

## 1. Executive Summary
This paper explains why large language models (LLMs) produce confident but false statements (“hallucinations”) even when their training data are correct. It shows, with precise statistical arguments, that the standard pretraining objective and today’s evaluation practices together create strong incentives to guess rather than express uncertainty—and it proposes a concrete, scalable fix for evaluations to reverse that pressure.

## 2. Context and Motivation
- Problem addressed
  - LLMs often output plausible falsehoods instead of saying they don’t know. The paper analyzes when and why this happens, from pretraining through post‑training, and how to mitigate it at the evaluation level.
- Why it matters
  - Hallucinations undermine reliability in real applications (medicine, law, coding). They also reduce trust and can be hard to detect. The paper argues they are not mysterious: they follow from basic statistics and incentives (Section 1; Table 1 shows concrete failures on dissertation titles; the “letter counting” prompt illustrates intrinsic mistakes).
- Prior approaches and gaps
  - Many causes have been proposed (overconfidence, exposure bias, long tails, spurious correlations). But there has been no general, end‑to‑end statistical account that:
    - connects generation errors to a simpler supervised problem,
    - handles prompts and abstentions (“IDK”), and
    - explains why post‑training systems still hallucinate despite many mitigations.
- Positioning
  - The paper unifies generative errors with misclassification in binary supervised learning via a reduction (Sections 3.1–3.2). It strengthens a prior theoretical result on “arbitrary facts” (birthdays) by including prompts and IDK (Section 3.3.1; Theorem 2) and offers a socio‑technical diagnosis of post‑training incentives (Section 4) with a practical remedy for mainstream benchmarks (Section 4.2; Table 2).

## 3. Technical Approach
At a high level, the paper shows:
1) Pretraining on cross‑entropy naturally leads to some errors even with perfect data, because avoiding all generative errors is strictly harder than solving a related binary classification task.
2) Post‑training evaluations usually score like exams that penalize uncertainty, so guessing raises measured performance.

Step by step:

A. Formalizing “valid vs. erroneous” generations (Sections 3.1, 3.2)
- The paper models the space of plausible strings (or prompt–response pairs) as a finite set `X`, partitioned into `V` (valid) and `E` (errors). The base model `p̂` is a probability distribution over `X`.
- Generative error rate is `err = p̂(E)` (Eq. 1).
- To analyze generation statistically, the paper defines a supervised classification problem named Is‑It‑Valid (`IIV`): given an `x ∈ X`, predict whether it is `+` (valid) or `−` (error).
  - Training/test distribution `D` is a balanced mixture: half samples come from the training distribution over valid text (`p` restricted to `V`), and half are uniformly sampled errors from `E` (before prompts; Eq. 2). With prompts, this is extended by sampling a prompt `c` from a distribution `μ` and then a uniformly random erroneous response from `E_c` (Section 3.2).

B. Reduction: from generation to classification (Theorem 1; Corollary 1)
- Use the language model itself as a classifier by thresholding probabilities:
  - Predict “valid” if `p̂(x) > 1/|E|` (or with prompts, `p̂(r|c) > 1/min_c |E_c|`; Section 3.2).
- Main bound (with prompts, Theorem 1):
  - > Generative error rate `err` ≥ 2 × `IIV` misclassification rate − (max_c |V_c| / min_c |E_c|) − δ.
  - Here `δ = |p̂(A) − p(A)|` where `A` is the set of responses above the threshold. Intuitively, `δ` measures miscalibration of the base model around that threshold.
- Interpretation:
  - If it’s hard to classify valid vs. invalid (high `IIV` error), then generation must also make mistakes, roughly twice as often (up to small terms). Avoiding generative errors would require excellent discrimination and calibration.

C. Why δ is small after pretraining (Section 3.1; Fig. 2)
- Pretraining minimizes cross‑entropy `L(p̂) = E_{x∼p}[−log p̂(x)]` (Eq. 3).
- Consider scaling up probabilities of all “above‑threshold” items by a factor `s` and re‑normalizing. The derivative of the loss in `s` at `s=1` equals `δ`. If `δ ≠ 0`, loss can be reduced by moving `s`, so local optimization makes `δ` small (Section 3.1).
- Empirical evidence (reprinted calibration histograms for GPT‑4): pretrained models are well‑calibrated while post‑RLHF models may be less so (Fig. 2; left ECE≈0.007 vs. right ECE≈0.074).

D. Incorporating prompts and abstentions (IDK) (Section 3.2)
- With prompts, valid and erroneous responses per prompt are `V_c` and `E_c`. The same reduction applies, yielding Theorem 1 (as above). The threshold uses `min_c |E_c|`.
- The analysis supports IDK as a valid response and treats it explicitly in later results (Theorem 2).

E. Two canonical statistical regimes (Section 3.3)
1) Arbitrary facts (no learnable pattern)
   - Model (Definition 1): each prompt `c` has exactly one correct answer `a_c` drawn uniformly from a set `R_c`, answered with probability `α_c`, or IDK otherwise.
   - Define `singleton rate` `sr` as the fraction of prompts that appear exactly once with a non‑IDK answer in the N‑sample training data (Definition 2).
   - Main bound (Theorem 2): with high probability,
     - > `err ≥ sr − 2/(min_c |E_c|) − (35 + 6 ln N)/√N − δ`.
     - When facts appear only once, they cannot be generalized: hallucination rate after pretraining is at least the share of such singletons (up to small terms).
   - Upper bound construction: there exists a calibrated `p̂` (δ=0) achieving
     - > `err ≤ sr − sr/(max_c |E_c| + 1) + 13/√N`.
   - Mechanism: this extends Good‑Turing “missing mass” estimation to settings with IDK (Appendix B; Lemma 1) and shows how unseen or singleton facts force errors.

2) Poor models (misspecification or inadequate capacity)
   - Define a family of thresholded‑LM classifiers `G = {g_{θ,t}}` by varying model parameters `θ` and threshold `t` (Section 3.3.2).
   - If even the best classifier in `G` has high `opt(G)` (agnostic error), generation must err:
     - > `err ≥ 2·opt(G) − (max_c |V_c| / min_c |E_c|) − δ` (from Theorem 1).
   - Special case: pure multiple‑choice with exactly one correct response per prompt, `C` options (Theorem 3; proved more strongly as Theorem 4 in Appendix C):
     - > `err ≥ 2 (1 − 1/C) · opt(G)`.
   - Example (Corollary 2): a trigram model must make ≥50% generation errors on a simple gender‑agreement prompt pair because it cannot disambiguate the long‑range dependency.

F. Additional error drivers (Section 3.4)
- Computational hardness: some prompts (e.g., decryption) are intractable; the reduction implies high error unless the model “breaks” the crypto (Appendix D; Observation 2).
- Distribution shift: out‑of‑distribution prompts induce classification—and thus generation—errors.
- GIGO (garbage‑in, garbage‑out): base models replicate errors in training data.

G. Post‑training: why hallucinations persist (Section 4)
- Formalizing exam‑style grading (Section 4.1):
  - A binary grader `g_c` outputs 1 for correct, 0 otherwise; abstentions (`IDK`) receive 0 (by definition).
  - Decision‑theoretic result (Observation 1; proof in Appendix E):
    - > Under any distribution over such graders, the score‑maximizing policy is never to abstain.
- Empirical meta‑evaluation of benchmarks (Section 4; Table 2, Section F)
  - Most widely used benchmarks (GPQA, MMLU‑Pro, MATH, SWE‑bench, HLE, etc.) use binary grading; IDK gets no or worse credit than a risky guess. One exception, WildBench, offers minimal partial credit but can still reward confident bluffs.
- Proposed fix: explicit confidence targets in instructions (Section 4.2)
  - Append to each task a statement like:
    - > “Answer only if you are > t confident; mistakes incur penalty t/(1−t), correct answers get 1, IDK gets 0.”
  - This turns abstention into an optimal choice whenever confidence ≤ t. It makes the acceptable risk explicit and objective across benchmarks.
  - Introduces “behavioral calibration”: models should answer only when true correctness probability exceeds the stated threshold, measurable via accuracy vs. abstention curves.

## 4. Key Insights and Innovations
1) Reduction from generative modeling to binary classification (Theorem 1; Fig. 1)
   - Novelty: a general, model‑agnostic link that lower‑bounds generative error by misclassification error on a constructed `IIV` task, including prompts and IDK.
   - Significance: reframes hallucinations as ordinary statistical errors driven by learnability, calibration, and model capacity—demystifying their origin.

2) Calibration–error tradeoff for base models (Section 3.1; Fig. 2)
   - Insight: minimizing cross‑entropy encourages local calibration, which mathematically forces some generative errors when discrimination is imperfect. A perfectly “non‑hallucinating” base model would be miscalibrated (large `δ`) unless it outputs IDK for everything.

3) Singleton‑rate lower bound for arbitrary facts with IDK (Theorem 2; Appendix B)
   - Novelty: extends Good‑Turing “missing mass” reasoning to prompts and abstentions, producing finite‑sample bounds both below and above.
   - Significance: gives a measurable predictor (singleton rate `sr`) for unavoidable hallucination on long‑tail facts, even with clean data.

4) Benchmark‑driven guessing incentive and an explicit fix (Observation 1; Table 2; Section 4.2)
   - Innovation: formalizes why binary scoring makes abstention strictly suboptimal and documents that leading benchmarks overwhelmingly use such scoring.
   - Proposal: embed explicit confidence targets and penalties into existing mainstream evaluations (not separate hallucination tests), enabling “behavioral calibration.” This is a leverage point for field‑wide change.

## 5. Experimental Analysis
This work is primarily theoretical plus a benchmark meta‑audit rather than a large‑scale empirical study.

- Evaluation methodology for meta‑audit (Section 4; Table 2; Section F)
  - The paper inspects the primary metrics of widely used leaderboards:
    - HELM Capabilities (five scenarios), Open LLM Leaderboard v2 collection, SWE‑bench, and HLE.
  - It checks whether abstentions can earn credit and whether grading is binary (0/1).
- Main findings (Table 2; Sections F.1–F.3)
  - > “The vast majority of popular evaluations have binary grading” (Table 2).
  - Benchmarks providing no credit for IDK: GPQA, MMLU‑Pro, BBH, MATH (L5), MuSR, SWE‑bench, HLE, Omni‑MATH (via equivalence grading).
  - IFEval aggregates binary sub‑scores; WildBench uses an LM‑graded rubric but may score IDK lower than a “fair” answer with hallucinations.
  - Additional detail: detailed reading of HELM’s featured scenarios shows 4/5 clearly give no IDK credit; WildBench can still penalize abstention relative to flawed but “helpful‑looking” answers (Section F.1).
- Supporting empirical illustrations
  - Real LLM failures on factual questions and counting letters (Section 1; Section 3.3.2).
  - Calibration evidence: GPT‑4 calibration curves before vs. after RL (Fig. 2; reprinted from OpenAI, 2023a). ECE rises from ~0.007 (pretrain) to ~0.074 (post‑RL), consistent with the cross‑entropy–calibration link and the idea that later training can distort it.
- Do these analyses support the claims?
  - The reduction‑based theorems are mathematically proved (Sections 3; Appendices A–D).
  - The benchmark audit is descriptive but concrete: it maps the scoring rules that shape model incentives today (Table 2 and Sections F.1–F.3).
  - The proposed evaluation change is testable: adding explicit thresholds lets one measure “answer‑only‑if‑confident” behavior via accuracy/coverage trade‑offs across `t`.

## 6. Limitations and Trade-offs
- Modeling assumptions (Section 5)
  - Finite “plausible” set `X` and a clean training distribution `p(V)=1`. Real corpora include noise; the authors note that noisy data would typically increase, not decrease, error lower bounds.
  - The “uniform random error” component in `D` for `IIV` simplifies analysis; real errors are structured.
  - The calibration term `δ` is evaluated at a single threshold; richer calibration notions (ECE) vary across thresholds (Section 3.1).
- Scope limits (Section 5)
  - Open‑ended, multi‑fact generations are simplified to any falsehood being an error; degrees of hallucination are not modeled.
  - Hidden user intent or ambiguous context (“latent context”) is out of scope.
  - Search/RAG and chain‑of‑thought are not panaceas under binary scoring; however, the paper does not benchmark these methods empirically here (Section 5).
- Practical trade‑offs
  - Adding explicit confidence targets introduces an accuracy–coverage trade‑off: models may abstain more, reducing headline accuracy unless leaderboards accept this new metric.
  - Selecting the penalty parameter `t` is application‑dependent; the paper suggests explicit but somewhat arbitrary thresholds (e.g., 0.5, 0.75, 0.9; Section 4.2).

## 7. Implications and Future Directions
- Field‑level implications
  - Conceptual: Hallucinations are not exotic failures of generation—they are standard statistical errors under capacity limits, long tails, and calibration. This reframes research toward classification‑style diagnostics and capacity/uncertainty management.
  - Practical: As long as leaderboards punish abstention, post‑training will continue to produce good “test‑takers” that guess. Adjusting mainstream benchmarks is a leverage point for safer systems (Section 4; Table 2).
- Methodological follow‑ups
  - Build “behavioral calibration” dashboards: plot accuracy vs. abstention rate under explicit thresholds `t`, and audit per‑domain calibration (Section 4.2).
  - Extend the reduction to graded hallucination severity and open‑ended multi‑fact outputs (Section 5).
  - Combine with RAG/reasoning: evaluate whether explicit thresholds improve retrieval/querying behavior (verify‑when‑uncertain pipelines).
  - Data curation: measure singleton rates `sr` per domain to predict unavoidable hallucination on long‑tail facts (Theorem 2) and prioritize data collection.
- Applications
  - High‑stakes workflows (medical, legal, coding): deploy systems that abstain below explicit confidence targets; integrate fallback search or human handoff.
  - Benchmark design: retrofit GPQA, MMLU‑Pro, SWE‑bench, HLE with explicit confidence instructions and penalties to realign incentives without creating new niche evals.

In short, the paper delivers a principled explanation of why LLMs hallucinate and a concrete, scalable path to reduce it: change benchmark scoring so that “I don’t know” is sometimes the optimal—and rewarded—answer.
