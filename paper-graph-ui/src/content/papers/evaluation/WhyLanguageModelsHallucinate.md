# Why Language Models Hallucinate

**ArXiv:** [2509.04664](https://arxiv.org/abs/2509.04664)

## 🎯 Pitch

This paper rigorously explains why large language models inevitably generate confident falsehoods ('hallucinations'), tracing their origins to statistical pressures during training and the misaligned binary accuracy metrics used in most benchmarks. By connecting generative modeling to classical binary classification, the authors not only demystify hallucinations as predictable errors but also reveal how current evaluation practices reward guessing over honest uncertainty—impacting model trustworthiness. Crucially, they propose a concrete fix: embedding explicit confidence targets in mainstream evaluations, paving the way for more reliable and transparent AI systems.

---

## 1. Executive Summary
This paper explains why large language models (LLMs) produce confident false statements (“hallucinations”) even when trained on correct data and how today’s evaluation culture inadvertently preserves them. It introduces a reduction from generative modeling to binary classification that yields lower bounds on inevitable error after pretraining, and it argues that post‑training benchmarks graded with binary accuracy incentivize guessing over honest uncertainty—then proposes a concrete fix: add explicit confidence targets to mainstream evaluations.

## 2. Context and Motivation
- Problem addressed
  - LLMs often output fluent but incorrect statements instead of signaling uncertainty. The paper analyzes the statistical causes of these errors during pretraining and explains why they persist after post‑training (Abstract; Section 1).
- Why it matters
  - Practically: confabulation erodes trust, causes downstream harm, and limits adoption.
  - Theoretically: it connects generative modeling to classical learning theory, turning a sometimes “mystical” failure mode into a predictable error pattern (Sections 3–4).
- Prior approaches and gaps
  - Many mitigations target decoding randomness, search/RAG, alignment/RLHF/RLAIF, uncertainty estimation, or bespoke hallucination tests (Related Work, Section 2). These help partially but don’t explain inevitability nor resolve why hallucinations remain after post‑training.
- This paper’s position
  - Pretraining: hallucinations are ordinary statistical errors. The paper derives lower bounds using a reduction to a binary classification task it calls Is‑It‑Valid (IIV) (Sections 3.1–3.2).
  - Post‑training: most “primary” benchmarks use binary grading (accuracy/pass rate). Under such grading, “I don’t know” (IDK) is strictly inferior to guessing, so optimization pressures preserve overconfident behavior (Section 4; Table 2).
  - Remedy: change the scoring of widely used benchmarks by embedding explicit confidence targets and penalties in the instructions (Section 4.2).

## 3. Technical Approach
The paper’s core methodology is a two‑stage analysis mirroring modern LLM training.

- Stage A: Pretraining as density estimation → reduction to binary classification
  - Setup (Section 3.1):
    - Consider a universe of “plausible strings” `X` partitioned into `V` (valid) and `E` (errors). A pretrained base model is a distribution `p̂` over `X` (prompts are added later).
    - Define the base model’s generative error rate `err := p̂(E)` (Eq. 1).
    - Training data are drawn from an error‑free distribution `p` with `p(V) = 1`.
  - The IIV task (Fig. 1; Section 3.1):
    - Construct a binary classification problem: examples from `V` (labeled +) with probability 1/2 and uniformly random errors from `E` (labeled −) with probability 1/2.
    - Use the language model as a classifier by thresholding its probability: predict “valid” iff `p̂(x) > 1/|E|` (Eq. 2).
  - Key inequality (Corollary 1; generalized in Theorem 1):
    - In words: if the language model would misclassify many examples in IIV, then it must also generate many errors. Formally (no prompts case),
      > generative error rate `err` ≥ 2 × IIV misclassification rate − |V|/|E| − δ
      where `δ` measures calibration mismatch between model and data mass above the threshold.
  - Why `δ` is small for base models (Section 3.1; Eq. 3 and Fig. 2):
    - `δ` equals the derivative of the cross‑entropy loss (Eq. 3) under a simple rescaling of probabilities above the threshold; at a local optimum of cross‑entropy, this derivative is near zero.
    - Empirical evidence from prior GPT‑4 calibration histograms shows pretrained models are well calibrated, while post‑training (e.g., RL) can degrade calibration (Fig. 2).
- Stage B: Generalization to prompted generation (Section 3.2)
  - Now examples are `(c, r)` where `c` is a prompt with distribution `μ`, and `r` is a response.
  - For each prompt, split responses into valid `V_c` and erroneous `E_c`.
  - The IIV distribution mixes (i) true `(c, r)` pairs drawn from training dialogs and (ii) the same prompts paired with uniformly random errors from `E_c`.
  - Main bound with prompts (Theorem 1):
    > `err ≥ 2·err_iiv − (max_c |V_c|)/(min_c |E_c|) − δ`
  - Interpretation: generation is strictly harder than recognizing validity; if a model cannot reliably classify valid vs. invalid responses, it will inevitably produce invalid ones during generation.

- Two focal scenarios that instantiate the bound
  1) Arbitrary facts (Section 3.3.1; Definitions 1–2; Theorem 2)
     - Model: for each prompt `c` there is one correct answer `a_c` chosen uniformly from a candidate set `R_c`, and the model may abstain with probability `1−α_c` by outputting `IDK`. This captures facts like birthdays where no generalizable pattern exists.
     - `singleton rate (sr)`: fraction of training prompts that appear exactly once with a non‑IDK answer (Definition 2). This is the Good–Turing “missing mass” proxy.
     - Lower bound (Theorem 2; lower part):
       > With high probability over training samples of size `N`, `err ≥ sr − 2/(min_c |E_c|) − (35 + 6 ln N)/√N − δ`
       Meaning: if, say, 20% of facts occur only once during pretraining and there are many more wrong answers than right ones per prompt (e.g., 364 wrong birthdays), base models will hallucinate on at least ~20% of those facts.
     - Tightness via an upper bound (Theorem 2; upper part):
       > There exists an efficient, calibrated `p̂` with `err ≤ sr − sr/(max_c |E_c| + 1) + 13/√N`
       This shows the lower bound is not vacuous and depends essentially on the singleton rate.
  2) Poor models (Section 3.3.2; Theorem 3 and Corollary 2)
     - Define a family of thresholded LM classifiers `G := {g_{θ,t}}` induced by `p̂_θ(r|c) > t`.
     - `opt(G)`: the best possible IIV classifier in that family (agnostic learning view).
     - Multiple‑choice lower bound (Theorem 3):
       > If each prompt has exactly one correct choice among `C` options, then `err ≥ 2(1 − 1/C) · opt(G)`.
     - Example: with a trigram language model that can’t represent long‑range dependencies, two prompts (“She lost it and was completely out of …” vs. “He …”) require gender‑aware completions. `opt(G)=1/2` implies any such model yields at least 50% generation error in this micro‑task (Corollary 2).

- Additional statistical drivers (Section 3.4)
  - Computational hardness: some prompts are intractable (e.g., decryptions without a key). Observation 2 formalizes that, under standard cryptographic security, any calibrated LM must err on such instances (Appendix D).
  - Distribution shift: if test prompts differ from training (OOD), generative errors follow.
  - GIGO (errors in the corpus): even if the base analysis assumes clean data, real corpora contain falsehoods that can be replicated.

- Post‑training and evaluation incentives (Section 4)
  - Binary grading makes abstention strictly suboptimal:
    > Observation 1 (Section 4.1; proof in Appendix E): For any distribution over binary graders that award 1 for a correct answer and 0 otherwise (including IDK), the expected‑score‑maximizing response is never an abstention.
  - Meta‑audit of mainstream benchmarks (Table 2; Section F):
    - Most flagship evaluations (GPQA, MMLU‑Pro, BBH, Omni‑MATH, MATH, MuSR, SWE‑bench, HLE, IFEval) use binary or exact‑match scoring; IDK typically gets zero credit.
  - Proposed fix (Section 4.2):
    - Embed explicit confidence targets into problem instructions, e.g.:
      > “Answer only if you are > t confident, since mistakes are penalized `t/(1−t)` points; correct answers get 1 point; IDK gets 0.”
    - This turns abstention into the rational choice below threshold `t` and encourages what the paper calls “behavioral calibration”: answer only when confidence exceeds the target, otherwise abstain.

## 4. Key Insights and Innovations
- Reduction from generation to classification (Sections 3.1–3.2; Theorem 1)
  - Novelty: links unsupervised density estimation with supervised binary classification via an operational classifier derived from `p̂`. This yields tight, general lower bounds that don’t depend on transformer details or next‑token prediction.
  - Significance: reframes hallucinations as ordinary misclassifications; once you can’t reliably classify validity, you can’t reliably generate valid outputs.
- Calibration term grounded in cross‑entropy (Section 3.1; Eq. 3; Fig. 2)
  - Insight: the “δ” term—difference in probability mass above a threshold between model and data—is the gradient of the cross‑entropy under a simple rescaling; pretraining drives it near zero. This explains why inevitability applies specifically to calibrated base models.
- Singleton‑rate bound for arbitrary facts with prompts and IDK (Section 3.3.1; Theorem 2)
  - Advancement: extends Good–Turing missing mass reasoning to prompted generation with abstentions, strengthening earlier results (now includes prompts and IDK; Theorem 2 with finite‑sample constants).
  - Impact: quantifies why long‑tail facts (e.g., obscure birthdays) resist elimination of hallucinations despite more data.
- Evaluation misalignment and a concrete remedy (Section 4; Observation 1; Table 2)
  - Diagnosis: binary grading entrenches guessing behavior across most flagship benchmarks.
  - Prescription: add explicit confidence thresholds—and penalties—in instructions of mainstream benchmarks, enabling a single, optimal behavior across tasks (“behavioral calibration”) and realigning incentives away from bluffing.

## 5. Experimental Analysis
This work is primarily theoretical plus a meta‑evaluation of benchmarks; still, it includes targeted empirical illustrations.

- Illustrative failures and calibration evidence
  - Table 1 shows three popular LLMs giving plausible but incorrect dissertation titles/dates for a named researcher; none match the ground truth (Introduction, Table 1).
  - Fig. 2 (Section 3.1) reprints GPT‑4 calibration curves: pretrained GPT‑4 is well‑calibrated, while RL‑fine‑tuned (PPO) deviates—supporting the argument that pretraining leads to small `δ` whereas post‑training can alter calibration.
  - The prompt “How many Ds are in DEEPSEEK? …” yields wrong counts across several models, illustrating a “poor model” failure due to tokenization/representation rather than lack of knowledge (Section 1; revisited in Section 3.3.2).
- Meta‑evaluation of benchmark incentives (Section 4.1; Appendix F; Table 2)
  - The paper inspects representative evaluations used on major leaderboards (HELM Capabilities, Open LLM Leaderboard, SWE‑bench, HLE).
  - Quote from Section F summarizing the core finding:
    > “Only one evaluation … offers minimal credit given for indicating uncertainty.”
  - Examples:
    - MMLU‑Pro and GPQA: multiple‑choice accuracy; no abstention credit.
    - SWE‑bench: patch correctness; no credit for uncertainty.
    - WildBench: LM‑graded rubric may even score IDK lower than a “fair” but hallucinated response (Table 2 note).
- Do the experiments support the claims?
  - For inevitability: the empirical portion is illustrative rather than quantitative, but the lower bounds are proved with explicit constants (Theorem 1; Theorem 2; Theorem 3).
  - For post‑training incentives: the benchmark audit (Table 2) directly supports the claim that binary grading dominates. Observation 1 formally shows why this grading rewards guessing.
- Ablations, failures, robustness
  - The paper provides upper/lower bounds (Theorem 2) to show tightness in the arbitrary‑facts setting.
  - It analyzes multiple error drivers (Sections 3.3–3.4) to separate irreducible long‑tail uncertainty, model class limitations, OOD, and computational hardness.

## 6. Limitations and Trade-offs
- Assumptions behind lower bounds
  - Clean training distribution `p(V)=1`: real corpora contain errors; the bounds show inevitability even in the best case (Section 3). In practice, GIGO can worsen errors (Section 3.4).
  - Calibration smallness (`δ ≈ 0`): justified for base models trained with cross‑entropy and supported by Fig. 2, but may be violated after certain post‑training methods (Section 3.1).
  - IIV construction: mixes true data with uniformly random errors from `E` (Sections 3.1–3.2). This is a stylized yet analytically tractable contrast set; real invalid strings need not be uniform.
- Scope of “plausibility”
  - The formalism partitions `X` into valid/invalid and largely ignores nonsensical outputs; the authors note how to extend the math to include a third “nonsense” set `N` (Section 5).
- Prompt coverage
  - The prompt distribution `μ` is assumed known in the reduction; real deployment involves distribution shift (Section 3.4) and complex multi‑turn contexts (Section 5).
- Open‑ended generation
  - The framework treats a response with any falsehood as an error; degrees of hallucination aren’t modeled (Section 5).
- Proposed evaluation change
  - Adding explicit confidence targets requires coordination by benchmark maintainers; it alters scores and leaderboards, and thresholds are somewhat arbitrary (Section 4.2). The paper argues for transparency (include thresholds in instructions) to preserve objectivity.

## 7. Implications and Future Directions
- Field‑level reframing
  - Hallucinations are not exotic defects of transformers but predictable errors given (a) finite data and (b) optimization for cross‑entropy and binary‑graded leaderboards. This reframing shifts focus from solely model tweaks to evaluation design.
- Practical actions for benchmark designers and labs
  - Integrate explicit confidence targets and penalties into mainstream benchmarks (e.g., SWE‑bench, MMLU‑Pro, GPQA, HLE) to stop penalizing abstention (Section 4.2).
  - Report “behavioral calibration” curves: accuracy/error as a function of threshold `t`, auditing if models answer only when sufficiently confident (Section 4.2).
- Research opportunities
  - Theory: extend the generation→classification reduction to richer settings (graded correctness, multi‑error counts, hidden context/aleatoric uncertainty; Section 5).
  - Methods: post‑training algorithms that optimize for thresholded, risk‑aware behavior (e.g., training with explicit penalties), and techniques that jointly improve knowledge coverage (reduce singleton rate effects) and model expressivity (reduce `opt(G)`).
  - Evaluation science: reproducible protocols for confidence‑aware grading; analyses of how RL or preference optimization affects calibration (cf. Fig. 2) and hallucination under the new metrics.
- Applications
  - Safety‑critical domains (medicine, law, finance): adoption of confidence targets can reduce harmful bluffs by making “IDK” rational.
  - Tool‑use and RAG systems: even with retrieval, if search yields low confidence, the modified grading encourages abstention rather than fabrication (Section 5).

Overall, the paper contributes a principled, end‑to‑end account: lower bounds explaining why base models must err (Theorems 1–3), a diagnosis of why current benchmarks preserve those errors (Observation 1; Table 2), and a concrete, implementable fix (explicit confidence targets) that could steer the ecosystem toward more trustworthy, uncertainty‑aware LLMs.
