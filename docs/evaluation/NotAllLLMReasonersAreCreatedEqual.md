# Not All LLM Reasoners Are Created Equal

**ArXiv:** [2410.01748](https://arxiv.org/abs/2410.01748)

## 🎯 Pitch

This paper introduces Compositional GSM, a novel extension of the GSM8K math benchmark designed to rigorously test large language models’ (LLMs) ability to perform true multi-step (two-hop) reasoning, where the output of one problem must be used to solve the next. By evaluating 24 popular LLMs, the authors reveal that many models—especially smaller, cost-efficient, and even math-specialized ones—exhibit a significant 'reasoning gap,' performing well on standard benchmarks but faltering when required to chain simple reasoning steps. This work challenges the prevailing notion that high GSM8K scores equate to genuine mathematical understanding, highlighting crucial weaknesses and providing a more realistic lens for assessing and improving LLM reasoning in real-world, compositional tasks.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Compositional GSM, a two-step (“two-hop”) version of the popular GSM8K grade‑school math benchmark, and uses it to measure a model’s ability to solve problems where the answer to the first question must be correctly reused to solve a second. Across 24 open and closed LLMs, the study finds large, systematic “reasoning gaps”: many models that do well on standard GSM8K struggle when the same skills must be composed, with the gap especially large for smaller, cost‑efficient, and even math‑specialized models. 

## 2. Context and Motivation
- Problem addressed
  - State-of-the-art LLMs score highly on GSM8K, a benchmark of grade‑school math word problems. This has led to a perception that such models “master” basic math reasoning. The paper asks a deeper question: can models combine (“compose”) simple, familiar skills to solve a multi-step problem that requires using the result of one step as input to another?
  - The study focuses on “two-hop” reasoning: solve `Q1`; use its answer `X` inside `Q2`. If models really understand GSM‑level math, they should perform close to the expected accuracy obtained by multiplying their independent accuracies on `Q1` and `Q2`.

- Why it matters
  - Real-world tasks frequently require multi-step, dependent reasoning (e.g., computing a subtotal and then using it to calculate tax). A model that performs well on isolated questions but degrades when steps must be chained can be unreliable in practice.
  - The work helps separate genuine reasoning from pattern matching and format memorization, which improves how we evaluate and train LLMs for robust reasoning.

- Prior approaches and limitations
  - Robustness studies show models can degrade under paraphrases or adversarial edits of math problems, and raise concerns about benchmark leakage. However, these works mostly keep the “one-hop” format.
  - Multi-hop reasoning has been examined in other domains, but results on whether larger models close the compositionality gap are mixed. The field lacks a targeted, same-difficulty compositional test aligned to GSM8K.

- Positioning
  - The paper contributes a controlled, same-difficulty, two-hop variant of GSM8K and a principled metric for the “reasoning gap.” It then analyzes why gaps occur (distraction, second‑hop failures) and how factors like instruction tuning, code generation, and fine‑tuning affect compositional performance.

## 3. Technical Approach
- Benchmark construction: Compositional GSM (Section 2; Figure 2)
  - Each item contains two questions:
    - `Q1`: a question from the original GSM8K test set.
    - `Q2`: a modified GSM8K test question where one number in its solution is replaced by the variable `X`, defined as the answer to `Q1`.
  - The model must:
    1) correctly solve `Q1` to compute `X`,
    2) substitute `X` into `Q2`,
    3) solve `Q2` to produce the final answer.

- How `Q2` is created (Section 2; Appendix A; Figure A.1)
  - Start from an original GSM8K problem that already has a code-form solution (from Gao et al., 2023).
  - Replace one numeric constant in that code with `X` to produce a new problem whose final answer remains a “sensible” positive integer and not too far from the original answer.
  - Execute the modified code to compute the new ground-truth answer.
  - To check problem quality, generate 16 solutions with frontier models (GPT‑4o and Gemini 1.5 Pro); if fewer than 4 of 16 reach the code-computed answer, manually revise the item (≈25% of items were edited).

- Datasets and splits (Section 3: Setup)
  - Three 1,200‑question test sets:
    1) Original GSM8K test split (`Q1 set`).
    2) Modified GSM8K test split (`Q2 set`, i.e., the same questions with one number replaced by `X` in their solution logic).
    3) Compositional GSM (paired `Q1 + Q2`, where `Q2` depends on the answer to `Q1`).
  - The distributions of final answer magnitudes in original and compositional sets are matched (Figure A.1).

- Core metric: “Reasoning gap” (Equation 1)
  - If a model’s independent accuracies are `S1` on `Q1` and `S2` on `Q2`, then the expected compositional accuracy—assuming independence and correct reuse—is `S1 × S2`.
  - The measured compositional accuracy is `Scomp`. The gap is:
    - Δ = `Scomp − S1 × S2` (Eq. 1).
  - Visualization: Figure 1 places each model with x-axis as `sqrt(S1 × S2)` (labeled “GSM8K accuracy” for simplicity) and y-axis as `Scomp`. The diagonal curve `y = x^2` represents the expectation `S1 × S2`. Points lying below that curve have negative gap.

- Evaluation protocol (Section 3: Setup; Appendices B–F)
  - Prompting: 8-shot exemplars for original and modified GSM8K (Appendix D), and a matched 8-shot prompt for Compositional GSM (Appendix E). Preambles enforce answer format (Appendix B).
  - Decoding: temperature 0, `pass@1` (i.e., single deterministic sample must be exactly correct).
  - Models: 24 LLMs covering closed and open families (GPT‑4o/mini; Gemini 1.0 Pro, 1.5 Flash/Pro; Llama‑3 8B/70B PT/IT; Gemma‑2 9B/27B PT/IT; Mistral 7B PT/IT; Mixtral‑8x7B PT/IT; NuminaMath‑7B‑CoT; Mathstral‑7B; Qwen2.5‑Math 7B/72B IT).
  - Reporting: when models need a preamble for formatting, results reflect their best of with/without preamble.

- Additional analyses (Sections 3.2–3.6)
  - Instruction tuning: Compare pretrained (`PT`) vs instruction-tuned (`IT`) variants across sizes (Figure 5).
  - Math specialization: Evaluate math‑specialized LLMs (Figure 6).
  - Fine‑tuning on GSM8K: Study overfitting with human vs synthetic (self-generated) data using Gemma‑2 27B PT (Figure 7; Appendix C).
  - Code vs natural language CoT: Use a code‑generation prompt (Appendix F) where `solution()` calls `X = solve_q1()`; compare to natural-language chain‑of‑thought (Figure 8).
  - Why gaps happen: check test leakage (Figure 9), distraction on `Q1` (Figure 10), failures on `Q2` even when `Q1` is correct (Figure 11), and whether merely co-presenting two independent questions causes trouble (Figure 12).

## 4. Key Insights and Innovations
- A same-difficulty, compositional benchmark grounded in GSM8K (Sections 2–3; Figures 1–2)
  - Novelty: Instead of making problems harder, the benchmark composes two familiar GSM8K questions so that the second depends on the first. This isolates compositional reasoning itself from pure math difficulty.
  - Significance: It provides a clean test of whether “skills + reuse” hold when models must carry intermediate results across steps.

- A principled “reasoning gap” metric (Equation 1; Figure 1)
  - Novelty: Uses independent accuracies on `Q1` and `Q2` to set an expected ceiling (`S1 × S2`) for the compositional task, then measures deviation (Δ). This avoids ambiguities from raw accuracy alone.
  - Significance: The metric quantifies whether models can reuse intermediate results without being distracted or format‑overfitting.

- Diagnosis of failure modes: distraction and second-hop weakness (Section 3.6; Figures 10–12)
  - Finding: Many models answer `Q1` less accurately when `Q2` is present (distraction), and even when they get `Q1` right, they often fail to correctly perform `Q2` with the `X` substitution (poor second-hop reasoning).
  - Significance: The analysis pinpoints why performance collapses, guiding training and prompting strategies beyond merely scaling.

- Evidence that small, cost‑efficient, and even math‑specialized models have larger gaps (Sections 3.1, 3.3; Figures 3–6)
  - Novelty: Systematically compares model families and sizes under identical prompts and decoding.
  - Significance: Challenges the assumption that high GSM8K scores reflect robust reasoning, especially in widely deployed, lower-cost models.

- Training levers have nontrivial, size-dependent effects (Sections 3.2, 3.4, 3.5; Figures 5, 7, 8)
  - Instruction tuning improves GSM8K much more than compositional GSM for small models (Figure 5 top), but not necessarily for large models (Figure 5 bottom).
  - Fine‑tuning on GSM8K improves GSM8K accuracy but can reduce compositional GSM after more steps (overfitting; Figure 7).
  - Generating code markedly helps smaller models on compositional tasks (Figure 8), suggesting tooling/scaffolding can partly compensate for weaker internal composition.

## 5. Experimental Analysis
- Evaluation methodology (Section 3: Setup)
  - Datasets: three 1,200‑question sets (original GSM8K; modified GSM8K with `X`; compositional pairs).
  - Metrics: exact-match accuracy; `pass@1` at temperature 0.
  - Protocol: 8-shot prompting with standardized preambles and answer formats; best of with/without preamble used per model.

- Main quantitative results
  - Large, pervasive reasoning gaps (Figures 1 and 3)
    - > “Most models demonstrate a noticeable gap between their reasoning performance on GSM8K and compositional GSM” (Figure 1 caption).
    - Figure 3 plots Δ for each model; many are substantially negative. Smaller, cost‑efficient, and math‑specialized models cluster with larger negative gaps.

  - Cost-efficient vs high-cost (Figure 4)
    - GPT family: `GPT‑4o` has Δ ≈ −1.1, while `GPT‑4o mini` (25–35× cheaper) has Δ ≈ −14.2 despite similar GSM performance.
    - Gemini family: `Gemini 1.5 Pro` Δ ≈ −5.8 vs `Gemini 1.5 Flash` Δ ≈ −11.3.
    - Llama‑3 IT: `70B‑IT` Δ ≈ −4.9 vs `8B‑IT` Δ ≈ −27.5.
    - Gemma‑2 IT: `27B‑IT` Δ ≈ −18 vs `9B‑IT` Δ ≈ −37.3.
    - Takeaway: Cheaper/smaller models perform close to larger ones on GSM8K but drop steeply on compositional GSM.

  - Instruction tuning varies by size (Figure 5)
    - Small models (top row): IT provides big gains on GSM8K but much smaller gains on compositional GSM—for example, `Llama‑3 8B` improves +25.1 points on GSM8K vs +12.6 on compositional GSM; `Gemma‑2 9B` +22.8 vs +4.8.
    - Large models (bottom row): Pattern can reverse or narrow—for example, `Llama‑3 70B`: +8.6 on GSM8K and +19.0 on compositional GSM.
    - Interpretation: The same instruction-tuning recipe does not generalize equally across scales; smaller models may overfit to GSM8K-style formats.

  - Math-specialized models still show gaps (Figure 6)
    - `NuminaMath‑7B‑CoT`: Δ ≈ −12.1; `Mathstral‑7B`: Δ ≈ −14.0; `Qwen2.5‑Math‑7B‑IT`: Δ ≈ −21.9; `Qwen2.5‑Math‑72B‑IT`: Δ ≈ −2.6.
    - Example: `Qwen2.5‑Math‑7B‑IT` scores above 80% on much harder MATH questions (as cited in Section 3.3) yet solves <60% of compositional GSM—evidence of overfitting to standard math benchmarks rather than robust composition.

  - Fine‑tuning can overfit (Figure 7; Appendix C)
    - Setup: Fine‑tune `Gemma‑2 27B PT` on GSM8K training solutions using human-written or synthetic (self-generated) solutions.
    - Observation: Compositional GSM accuracy rises initially (to 100 steps) but drops by 400 steps, while GSM8K test accuracy keeps improving. Quote: 
      > “In both settings … after 100 training steps, compositional GSM test performance drops while GSM8K test performance keeps improving” (Figure 7 caption).
    - Implication: More training on GSM8K-format data can increase format-specific performance at the expense of general compositional ability.

  - Code vs natural language CoT (Figure 8; Appendix F)
    - Implemented as two Python functions: `solve_q1()` and `solution()` with `X = solve_q1()` in `solution()`.
    - Relative improvements on compositional GSM over natural language CoT:
      - `Llama‑3 8B‑IT`: +69%; `Llama‑3 70B‑IT`: +2%.
      - `Gemma‑2 9B‑IT`: +74%; `Gemma‑2 27B‑IT`: +27%.
      - `Mistral 7B‑IT`: +149%; `Mixtral‑8x7B‑IT`: +71%.
    - Interpretation: Smaller models benefit most from explicit code scaffolding; very large models gain little, suggesting they already internalize some composition ability.

  - Ruling out test-set leakage; isolating failure modes (Section 3.6; Figures 9–12)
    - Leakage check (Figure 9): Accuracies on original vs modified GSM8K align close to the `x=y` line, indicating that replacing a number with `X` did not itself create a harder distribution or reveal memorization effects.
    - Distraction on `Q1` (Figure 10): Many models solve fewer `Q1`s when `Q2` is present than when `Q1` is asked alone; they “get distracted” by extra context.
      > “Models below the trend-line get distracted and cannot answer `Q1` in the compositional format even though solving it does not depend on any other question” (Figure 10 caption).
    - Second-hop weakness (Figure 11): Even conditioned on `Q1` being correct, models underperform on `Q2` compared to solving `Q2` alone.
      > “While models may correctly answer the first question, they frequently make subtle errors … when solving the second question” (Figure 11 caption).
    - Two questions in context (Figure 12): When `Q2` is independent of `Q1`, co-presenting them causes minimal degradation, confirming that the main difficulty is using `X` from `Q1`—i.e., second-hop composition—not merely handling longer input.

- Do the experiments support the claims?
  - Yes: The study triangulates the reasoning gap with a well-defined expected baseline (Eq. 1), multiple families/sizes of models, controls for leakage (Figure 9), and targeted analyses separating distraction from second-hop failures (Figures 10–12). The findings are consistent across metrics and manipulations (instruction tuning, code generation, fine‑tuning).

- Robustness checks and ablations
  - Prompt formatting controls via preambles (Appendix B).
  - Multiple models and sizes; both open and closed.
  - Code-executed ground truth for modified `Q2` and manual fixes for ≈25% problematic items.
  - Synthetic vs human fine‑tuning data.
  - Natural language vs code solutions.

## 6. Limitations and Trade-offs
- Assumptions baked into the expected baseline
  - The expected compositional accuracy uses `S1 × S2`, which assumes independent errors and perfect transfer of `X` from `Q1` to `Q2`. Real models may have correlated errors or propagate uncertainty differently, so Δ can conflate multiple effects.

- Construction choices for `Q2`
  - Only one number is replaced with `X`, and the new answer is kept “not too far” from the original (Section 2). This ensures comparability but restricts the space of possible compositions. Other forms of dependence (e.g., structural changes or multiple substitutions) are not explored.

- Evaluation at `pass@1`, temperature 0
  - This strict setting reflects deterministic inference in many applications, but may understate models that succeed with simple sampling or self-consistency. Conversely, allowing sampling could blur whether the model “really” composes or just gets lucky.

- Prompt-specific generalization
  - Results depend on 8-shot prompts and specific format preambles. Although the authors test both with and without preambles and select the best (Section 3: Setup), other prompting styles (e.g., least-to-most, tool use) might shift absolute scores.

- Scope: math word problems
  - The study targets GSM‑style arithmetic/algebraic composition. Compositional reasoning in other domains (symbolic logic, commonsense, multimodal) might show different patterns. 

- Fine‑tuning study is limited in breadth
  - The overfitting analysis uses one base model (Gemma‑2 27B PT), a specific training protocol, and up to 400 steps (Figure 7). Broader hyperparameter sweeps or other objectives (e.g., preference optimization, verifier‑guided training) could paint a more nuanced picture.

## 7. Implications and Future Directions
- How this work reshapes the field
  - High performance on popular one-hop math benchmarks does not guarantee robust compositional reasoning. The introduction of Compositional GSM and the “reasoning gap” metric provides a sharper lens for evaluating reasoning reliability, particularly in cost‑efficient models that are attractive for deployment.

- Practical applications
  - Model selection: Teams should test candidate models on compositional tasks similar to their target use cases, not just on single-hop benchmarks.
  - Prompting and tooling: For smaller models, code generation or explicit function scaffolding (`solve_q1()` then `solution()`) can markedly improve composition (Figure 8).
  - Training strategy: Be cautious with extended fine‑tuning on benchmark-style data; monitor compositional metrics to avoid overfitting (Figure 7).

- Suggested research directions
  - Better second-hop training: Develop training curricula or objectives that explicitly encourage using intermediate results (e.g., supervised traces that require binding the symbol `X` and reusing it).
  - Distraction-robust prompting: Explore prompts that segment subproblems, enforce intermediate variable naming, or use structured memory to carry values between steps.
  - Beyond two hops: Extend the framework to three or more dependent steps, different dependency types (multiple substitutions, functional transformations), and other domains (e.g., MATH dataset, multimodal reasoning), as hinted in the Discussion.
  - Verification and tool use: Combine LLMs with lightweight program interpreters or verifiers to check and reuse intermediate results, possibly reducing second-hop errors without sacrificing interpretability.
  - Scaling studies with principled metrics: Use Δ as a standard metric when reporting reasoning results, to prevent over-optimism from single-hop accuracies.

> Bottom line (Discussion; Figures 1, 3–6, 10–12): “Not all LLM reasoners are created equal.” Many models—especially smaller, cheaper, and even math‑specialized ones—exhibit substantial deficits when knowledge must be composed across steps. Measuring and training for composition, not just single-step accuracy, is essential for trustworthy reasoning systems.
