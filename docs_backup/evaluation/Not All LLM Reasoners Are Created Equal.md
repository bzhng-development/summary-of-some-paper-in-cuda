# Not All LLM Reasoners Are Created Equal

**ArXiv:** [2410.01748](https://arxiv.org/abs/2410.01748)
**Authors:** Arian Hosseini, Alessandro Sordoni, Daniel Toyama, Aaron Courville, Rishabh Agarwal
**Institutions:** Mila, Google DeepMind, Microsoft Research

## 🎯 Pitch

The paper introduces Compositional GSM, a novel benchmark that reveals significant weaknesses in large language models by testing their ability to solve two simple math problems in sequence. This work is crucial as it highlights the gap between high performance on single-question benchmarks and actual multi-step reasoning skills, urging a reevaluation of model evaluation and training for tasks requiring sequential reasoning.

---

## 1. Executive Summary
This paper introduces Compositional GSM, a two-step (“two-hop”) version of the popular GSM8K grade‑school math benchmark, and a metric called the reasoning gap that measures how much a model’s performance drops when two easy problems must be solved together. Across 20+ open and closed LLMs, the study shows that many models—especially small, cost‑efficient, and math‑specialized ones—perform far worse than expected on this simple composition, largely due to distraction from extra context and failure on the second reasoning hop (Figures 3–4, 10–12).

## 2. Context and Motivation
- Problem addressed
  - Many LLMs score very highly on single‑question grade‑school math benchmarks like GSM8K. The open question is whether these models truly reason or whether they exploit patterns in question format. This work probes that gap by testing whether models can combine two familiar problems into one small composition without increasing math difficulty (Section 1, Figure 2).

- Why it matters
  - Real tasks often require multi‑step reasoning: solving a subproblem and correctly carrying its result into the next step. If LLMs fail on such basic composition, their reliability on workflows, tutoring, planning, and tool use is overestimated by single‑question benchmarks.

- Prior approaches and shortcomings
  - Robustness work has explored test‑set leakage, adversarial phrasing, or functional rewrites of math problems (Section 4). These show brittleness but do not directly isolate the skill of chaining two easy, familiar problems in one prompt.
  - Multi‑hop reasoning analyses often use knowledge retrieval tasks; here the authors use a numerically precise, easily verifiable math setting to pinpoint where multi‑hop breaks.

- Position relative to existing work
  - The paper positions Compositional GSM not as “just another benchmark” but as a controlled case study of two-hop reasoning at the same difficulty level as GSM8K (Section 1). It pairs this with a simple expected‑performance model and a diagnostic metric (Equation 1) to quantify how far models fall below what should be achievable.

## 3. Technical Approach
- Core setup
  - GSM8K: A dataset of grade‑school math word problems with short, verifiable answers.
  - Compositional GSM: Each test item joins two GSM8K questions, Q1 and Q2, so that the numerical answer to Q1 becomes a variable `X` that must be substituted into Q2 (Figure 2). Both subproblems are grade‑school level; only composition is new.

- How the dataset is built (Section 2; Appendices A, E, F)
  - Start from 1,200 GSM8K test questions to serve as Q1.
  - Build a modified set of 1,200 Q2 questions by taking other GSM8K items and editing a single number in their code‑form solutions so the final answer changes, remains a positive integer, and stays close in magnitude to the original. The substitution location is chosen so Q2 remains sensible after replacing that number with `X` (Figure 2; Appendix A shows resulting answer‑magnitude distributions remain similar to GSM8K).
  - Sanity checks: for each modified Q2, generate 16 candidate solutions using two strong models and keep or fix items that do not yield consistent correct answers (about 25% required manual edits).

- How accuracy and the “reasoning gap” are defined (Section 2)
  - Measure three accuracies (each on 1,200 items):
    - `S1`: accuracy on the original GSM8K test split (used as Q1).
    - `S2`: accuracy on the modified GSM8K test split (the standalone Q2 variants).
    - `Scomp`: accuracy on the compositional set where Q1 and Q2 appear together and Q2 depends on `X`.
  - Expected compositional performance if the two steps were independent is `S1 × S2`. The reasoning gap is
    - “Reasoning gap: Δ = Scomp − S1 × S2” (Equation 1).
  - Intuition: If a model can solve Q1 and Q2 independently with probabilities `S1` and `S2`, it should solve both in composition with probability `S1×S2`. Deviations indicate difficulty caused by composition itself.

- Evaluation protocol (Section 3; Appendices B–F)
  - Models: GPT‑4o/mini, Gemini 1.0/1.5 (Flash/Pro), Llama‑3 (8B/70B, PT/IT), Gemma2 (9B/27B, PT/IT), Mistral‑7B (PT/IT), Mixtral‑8×7B (PT/IT), Phi‑2, Phi‑3‑mini‑IT, Mathstral‑7B, NuminaMath‑7B‑CoT, Qwen2.5‑Math‑7B/72B‑IT (Section 3).
  - Prompting: standardized 8‑shot prompts for all three splits; a short preamble is added if a model needs formatting guidance (Appendix B, D, E).
  - Decoding: temperature 0, `pass@1` (i.e., the first output must be correct; no sampling for multiple tries).
  - Two solution modes: natural‑language chain‑of‑thought (`CoT`) vs Python code generation that explicitly defines `solve_q1()` and then calls it inside a `solution()` function for Q2 (Appendix F).

- Additional experiments to diagnose causes (Section 3.6)
  - Leakage check: Compare `S1` vs `S2` to see if modified standalone Q2s are harder or contaminated; they mostly line up on the x=y line (Figure 9).
  - Distraction check: Compare accuracy on a Q1 alone vs the same Q1 when embedded at the start of a compositional item (Figure 10).
  - Second‑hop check: Compare accuracy on a Q2 alone vs Q2 in composition conditional on Q1 being solved correctly (Figure 11).
  - Two‑questions capacity: Compare Q2 alone vs Q2 with Q1 in context but independent vs Q2 in composition (Figure 12).

- Design rationale
  - Two-hop composition focuses on the simplest form of compositional generalization: correctly solve an easy subproblem and carry its result forward.
  - The controlled code‑based editing ensures Q2 stays grade‑school, keeps answer magnitudes comparable to GSM8K (Appendix A), and lets the study attribute failures to composition rather than a shift in difficulty.

## 4. Key Insights and Innovations
- A simple, controlled test exposes big hidden weaknesses
  - Innovation: the Compositional GSM construction plus the reasoning‑gap metric (Equation 1) isolates the cost of composition itself.
  - Significance: Many models that ace GSM8K drop sharply when asked to solve two easy steps in one prompt (Figures 1 and 3). This reveals a gap between benchmark performance and actual multi‑step reliability.

- Size and cost matter—in the wrong direction for deployment
  - Finding: Small and cost‑efficient models show much larger negative reasoning gaps than their larger counterparts, despite similar GSM8K scores (Figure 4).
  - Significance: For practitioners optimizing cost, single‑benchmark scores are misleading; multi‑step reliability may collapse.

- Task formatting and training recipes interact with model size
  - Finding: Instruction tuning (“IT”) substantially boosts GSM8K accuracy for small models but yields only modest gains on compositional GSM; for larger models the pattern is weaker or reversed (Figure 5).
  - Significance: The same IT recipe can overfit smaller models to standard formats while not improving compositional reasoning.

- “Math‑specialized” does not equal “compositional”
  - Finding: Models trained heavily on math (e.g., Qwen2.5‑Math‑7B‑IT) still show substantial gaps and even signs of overfitting to benchmark style (Figure 6).
  - Significance: Specialized training can raise single‑problem scores without transferring to simple two‑hop composition.

- Where the failure comes from: distraction and second‑hop errors
  - Evidence: Models often miss details in Q1 when Q2 is present (Figure 10) and frequently fail Q2 even when Q1 is correct (Figure 11). When Q2 is independent of Q1, simply adding Q1 to the context causes little harm (Figure 12).
  - Significance: The bottleneck is not handling two questions per se; it is correctly using Q1’s result inside Q2.

- Code helps, especially for small models
  - Finding: Switching from natural‑language CoT to code yields large relative gains on compositional GSM for smaller models—for example, Llama‑3‑8B (+69%), Gemma2‑9B (+74%), and Mistral‑7B (+149%)—with smaller effects for big models like Llama‑3‑70B (+2%) (Figure 8).
  - Significance: Externalizing intermediate computation into code scaffolds the second hop for weaker models.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets: Three 1,200‑item test sets (original GSM8K as Q1; modified GSM8K as standalone Q2; compositional GSM combining Q1→Q2) (Section 3).
  - Metric: exact‑match accuracy; `pass@1`.
  - Prompting: 8‑shot exemplars for each split (Appendices D–F).
  - Models: A broad sweep of open and closed models, pretrained (PT), instruction‑tuned (IT), and math‑specialized (Section 3).

- Main quantitative results
  - Overall reasoning gaps
    - Figure 1 plots compositional accuracy vs the geometric mean of `S1` and `S2` with a y=x² expectation line; most points lie well below the curve, showing large negative Δ.
    - Figure 3 ranks models by Δ: small/cost‑efficient and math‑specialized models have the largest negative gaps.
  - Cost‑efficient vs high‑end (Figure 4)
    - GPT‑4o vs GPT‑4o mini: mini shows a far larger negative Δ (≈−14 points vs ≈−1).
    - Gemini 1.5 Pro vs 1.5 Flash: Flash has a much larger gap (≈−11 points vs ≈−6).
    - Llama‑3‑70B‑IT vs 8B‑IT: 8B‑IT gap (~−27.5) dwarfs 70B‑IT (~−4.9).
    - Gemma2‑27B‑IT vs 9B‑IT: 9B‑IT gap (~−37.3) is much larger than 27B‑IT (~−18).
    - Quote: “Although the cheaper models perform similarly on the original GSM8K test, they show a significant decline in performance on the compositional GSM test” (Figure 4 caption).
  - Instruction tuning across sizes (Figure 5)
    - Small models: IT boosts GSM8K a lot more than compositional GSM (e.g., Mistral‑7B: +14.1 vs +4.3; Llama‑3‑8B: +25.1 vs +12.6; Gemma2‑9B: +22.8 vs +4.8).
    - Large models: pattern weak or reversed (e.g., Llama‑3‑70B: +8.6 GSM8K vs +19.0 compositional).
    - Quote: “For smaller models… instruction‑tuning results in substantial improvements on the original GSM8K test set, but a much smaller improvement on the compositional GSM test” (Figure 5 caption).
  - Math‑specialized models (Figure 6)
    - Large negative gaps remain: Numina‑7B‑CoT (~−12), Mathstral‑7B (~−14), Qwen2.5‑Math‑7B‑IT (~−22), while 72B‑IT is closer to parity (~−3).
    - Text highlights: “Qwen2.5‑Math‑7B‑IT… solves less than 60% of the compositional grade‑school math problems” despite strong MATH performance (Section 3.3).
  - Finetuning and overfitting (Figure 7)
    - Fine‑tuning Gemma2‑27B on GSM8K solutions (human or synthetic) improves GSM8K accuracy steadily, but compositional accuracy increases only up to ~100 steps, then drops by 400 steps.
    - Quote: “After 100 training steps, compositional GSM test performance drops while GSM8K test performance keeps improving… [suggesting] task‑specific overfitting” (Figure 7 caption).
  - Natural language CoT vs code (Figure 8)
    - Relative gains for small models are striking: Mistral‑7B (+149%), Llama‑3‑8B (+69%), Gemma2‑9B (+74%). Big models benefit less (e.g., Llama‑3‑70B +2%).
    - Quote: “Smaller models benefit more from generating code rather than natural language CoT” (Figure 8 caption).

- Diagnostic analyses on causes (Section 3.6)
  - Leakage not the culprit (Figure 9)
    - Plot of `S1` (original) vs `S2` (modified) hugs the x=y line: “test set leakage is not a major concern.”
  - Distraction on Q1 (Figure 10)
    - Many models perform worse on the same Q1 when it appears at the start of a compositional item; responses often “overlook important details” when Q2 sits below.
  - Second‑hop difficulty (Figure 11)
    - Even when Q1 is correct, many models fail Q2 more often than when Q2 is asked alone; they have “become too specialized in handling GSM8K‑style questions.”
  - Two‑question capacity (Figure 12)
    - When Q2 does not depend on Q1, adding Q1 barely harms performance. The failure is specifically on using Q1’s answer inside Q2.

- Do the experiments support the claims?
  - Yes. The expected‑performance model (`S1 × S2`) plus the triad of plots (Figures 10–12) triangulate the mechanism: extra context alone is not the main problem; rather, composing the second hop using the first hop’s result is the failure point. Size, cost, and training mode systematically modulate the gap (Figures 4–6), and code scaffolding mitigates it in smaller models (Figure 8).

## 6. Limitations and Trade-offs
- Assumptions in the expected‑performance model
  - The baseline expectation `S1 × S2` treats Q1 and Q2 as independent subtasks. In practice, solving Q1 in the compositional setting might be easier or harder than alone (Figure 10 shows it is often harder), which builds the gap by design. That is a feature for diagnosis, but it also means Δ conflates second‑hop failure with Q1 distraction.

- Scope of tasks
  - The study focuses on grade‑school math. While chosen for verifiability and control, results may differ in other domains (e.g., commonsense, programming beyond arithmetic).

- Single prompting/decoding regime
  - All models use 8‑shot prompts and temperature 0 `pass@1`. Some models might improve under different prompting or sampling strategies (e.g., reruns, self‑consistency). The uniform regime strengthens comparability but may understate a model’s best achievable performance.

- Dataset construction choices
  - Q2s are edited via code‑form solution changes and curated with model agreement plus manual checks (Section 2). This yields a clean test but still relies on the availability and correctness of code‑form solutions and editorial choices that keep the final answer “not too far” from the original.

- Finetuning study breadth
  - Overfitting results are shown for one base model (Gemma2‑27B PT) and short training runs (50–400 steps). The pattern is suggestive but not exhaustive across models or curricula.

- Evolving closed models
  - Results for proprietary models (e.g., GPT‑4o, Gemini 1.5) reflect specific versions and may shift as APIs update.

## 7. Implications and Future Directions
- Implications for evaluation and deployment
  - Single‑question math benchmarks can overestimate real reasoning reliability. A basic two‑hop composition reveals sizable weaknesses, especially in the small/cost‑efficient regime (Figures 3–4). Practitioners should include compositional tests when selecting models for workflows that chain steps, even if each step is simple.
  - Instruction tuning can overfit small models to familiar formats (Figure 5). Training and evaluation should explicitly include multi‑step compositions to avoid brittle behavior.
  - Code‑based reasoning scaffolds (Figure 8) can notably boost small models; tool‑use or program‑aided agents may be preferable for low‑cost deployments.

- Research directions
  - Extend compositional testing beyond GSM8K: to harder math (MATH), non‑math reasoning, and multimodal settings (discussion in Section 5).
  - Develop training curricula that directly target second‑hop grounding—learning to carry forward intermediate results faithfully rather than just producing fluent single‑problem solutions.
  - Diagnose and mitigate distraction: techniques that preserve focus across multiple subquestions (e.g., structured memory, explicit sub‑goal tracking, verifier‑aided decomposition).
  - Explore scalable code‑oriented scaffolds and hybrid NL‑code strategies that retain interpretability while improving reliability for small models.
  - Rethink progress metrics: incorporate expectation‑based gaps like Δ and conditional plots (Figures 10–12) into standard evaluation suites to distinguish genuine reasoning from format familiarity.

- Practical applications
  - Education/tutoring systems that assign multi‑part problems.
  - AI agents that must execute sequences of subgoals or tool calls where later steps consume outputs from earlier steps.
  - Quality assurance for cost‑sensitive deployments: compositional smoke tests can prevent over‑reliance on high GSM8K scores.

> Bottom line: Even at grade‑school difficulty, two‑hop composition remains a stumbling block for many LLMs. Measuring and training for this capability is essential if we want models that not only answer isolated questions but also carry correct intermediate results through to the end.
