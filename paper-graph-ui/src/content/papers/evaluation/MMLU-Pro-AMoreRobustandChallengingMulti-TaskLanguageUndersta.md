# MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark

**ArXiv:** [2406.01574](https://arxiv.org/abs/2406.01574)

## 🎯 Pitch

MMLU-Pro is a next-generation benchmark for large language models that tackles the looming saturation and instability of MMLU by introducing more challenging, reasoning-focused questions, expanding answer choices from four to ten, and eliminating trivial or noisy items. By making model scores more discriminative and robust to different prompts, and by rewarding genuine multi-step reasoning (as seen by the substantial gains from chain-of-thought approaches), MMLU-Pro provides the research community with a vital, future-proof tool for tracking real progress in AI language understanding.

---

## 1. Executive Summary (2-3 sentences)
MMLU-Pro is a new benchmark for evaluating large language models (LLMs) that replaces the widely used MMLU with a harder, more reliable, and more discriminative test. It increases multiple-choice options from 4 to 10, focuses more on reasoning (not just recall), and removes noisy or trivial items—yielding clearer separation among strong models, lower prompt sensitivity, and large gains from chain-of-thought reasoning.

## 2. Context and Motivation
- Problem the paper addresses
  - Frontier LLMs have nearly saturated the original MMLU benchmark: top models cluster at 86–87% accuracy, making it hard to tell them apart or track progress (Introduction; Figure 4). GPT-4o shows big gains elsewhere (e.g., MATH, Chatbot Arena) but only +1% on MMLU (Introduction).
  - Scores on MMLU are sensitive to prompt phrasing and scoring details, causing unstable rankings (Introduction; Section 6.3).
  - Suspected causes (Introduction):
    - Only three distractors per question (4 options total), allowing shortcut strategies without deeper understanding.
    - Items are “mostly knowledge-driven” and often solvable without multi-step reasoning; chain-of-thought (CoT) tends not to help on MMLU.
    - Nontrivial noise: unanswerable or mislabeled items lower the performance ceiling.

- Why this matters
  - Practically: model choices hinge on benchmarks. Saturation and instability reduce trust in leaderboards and hinder targeted improvement.
  - Scientifically: to study reasoning improvements, a benchmark must elicit and reward multi-step reasoning, not chance or shallow heuristics.

- Prior approaches and gaps
  - Widely used evaluations (e.g., GLUE, SuperGLUE, BigBench, ARC, MMLU) cover language understanding, factual knowledge, and synthetic reasoning (Related Work). But MMLU’s difficulty and robustness are no longer sufficient for today’s frontier models.

- How this work positions itself
  - MMLU-Pro builds directly on MMLU (same spirit: diverse, exam-like questions) but redesigns three aspects: more distractors (10 options total), more reasoning-heavy content, and two rounds of expert validation to reduce noise (Abstract; Section 3). The result is a benchmark that is harder, more stable across prompts, and more discriminative among strong models (Sections 5–6; Figures 4–5; Table 3).

## 3. Technical Approach
MMLU-Pro is a curated, multi-domain multiple-choice benchmark emphasizing reasoning. Its construction pipeline (Figure 2; Section 3.2) has four stages:

1) Initial filtering (clean and raise difficulty)
- Merge 57 MMLU subjects into 14 broader disciplines to reduce redundancy and emphasize core areas (Section 3.2).
- Identify and remove trivial items from MMLU by running eight mid-sized models (e.g., `Llama-2-7B`, `Mistral-7B`, `Gemma-7B`, `Yi-6B`) and dropping questions answered correctly by more than four of them (Section 3.2).
  - Outcome: 5,886 questions removed as “too easy” (Section 3.2). Filtering percentages by discipline are reported in Appendix A.1, Table 4 (e.g., >50% filtered for Business, History, Other, Psychology).

2) Question collection and integration (increase breadth and reasoning demand)
- Sources (Figure 3b; Section 3.1): original MMLU (56.6%), STEM websites (33.9%), TheoremQA (5.0%), and SciBench (4.5%). Total: 12,032 questions across 14 disciplines (Figure 3a).
- Adaptation to multiple choice:
  - For STEM Website and TheoremQA items (often short-answer or solution-based), use `GPT-4-Turbo` to (i) extract a short, correct answer from solutions and (ii) generate three plausible distractors (Section 3.2, “Question Collection and Integration,” Table 6 prompts).
  - Manual comparison checks remove items with incomplete or incorrect answer extraction (Section 3.2).

3) Option augmentation to 10 choices (raise difficulty and reduce guessability)
- Expand each 4-option item to 10 options (A–J) using `GPT-4-Turbo` to generate six additional, plausible distractors (Section 3.2 “Option Augmentation”; Table 6 prompt).
  - Rationale: a 10-option question reduces guess probability from 25% to 10%, and more “plausible” distractors pressure models to reason rather than pattern-match.
  - Authors report that this augmentation does not give `GPT-4-Turbo` a special advantage in evaluation (Section 3.2).
  - Post-review dataset stats: 83% of items have 10 options; the remainder have fewer due to filtered distractors; average options per item = 9.47 (Section 3.2).

4) Two-phase expert review (improve correctness and reliability)
- Phase 1: Human experts verify answer correctness and remove “bad questions” (need images/tables, lack sufficient information, or are unsuitable for MCQ) (Section 3.2).
- Phase 2: `Gemini-1.5-Pro` re-evaluates options to flag potential “false negatives” (i.e., an option that is actually correct but was labeled incorrect). Human experts then adjudicate (Section 3.2).
- Issues found (Table 1):
  - “Incorrect Answer” corrections (e.g., MMLU: 350; STEM: 483).
  - “False Negative Options” removals (e.g., MMLU: 1,953).
  - “Bad Questions” removals (e.g., STEM: 862).

Evaluation protocol (Section 4)
- Prompting: 5-shot chain-of-thought (`CoT`) for most models (examples per discipline; Appendix A.2), except `Gemini-1.5-Pro/Flash` which use 0-shot (Table 2 note).
  - `CoT` prompting: demonstrations include step-by-step reasoning and end with a formatted final choice (“The answer is (X)”) to encourage deliberate reasoning and consistent extraction (Table 7).
- Answer extraction: regex looks for the formatted final choice; a backup regex handles deviations; if both fail, a random option is chosen (Section 4 “Answer Extraction”).
- Robustness check: evaluate with 24 reasonable prompt variants to measure sensitivity (Section 6.3; Figure 5).
- Metric: accuracy (percentage of correct choices).

Key terms defined
- `Chain-of-Thought (CoT)`: prompting models to write out intermediate reasoning steps before producing the final answer, which can improve reasoning reliability.
- `Distractor`: an incorrect answer option designed to be plausible.
- `False negative option`: an option mistakenly labeled as incorrect when it is actually correct (Phase 2 of expert review targets these).

Design choices and their motivations
- 10 options per question: lower chance guessing; forces finer discrimination among options; reduces prompt-induced score volatility (Section 3; Abstract; Section 6.3).
- More reasoning-heavy sources (TheoremQA, SciBench, STEM): increases the proportion of questions requiring multi-step derivation, calculation, or theorem application (Section 3.1).
- Two-stage review: addresses MMLU’s known noise and ensures distractors are actually incorrect and distinct from the true answer (Section 3.2; Table 1).
- CoT evaluation: reflects the benchmark’s emphasis on reasoning; the paper also compares CoT vs. direct answering (Section 6.2; Table 3).

## 4. Key Insights and Innovations
1) Ten-option multiple choice as a simple but powerful difficulty lever
- What’s new: Expand from 4 to 10 options with carefully crafted distractors (Section 3.2).
- Why it matters:
  - Performance drops substantially on MMLU-Pro compared to MMLU for many models, increasing headroom for progress (Figure 4; Abstract: “16% to 33%” drop).
  - Prompt sensitivity shrinks from ~4–5% to ~2% (Section 6.3; Figure 5), suggesting fewer “accidental” wins due to prompt phrasing.

2) Explicit focus on reasoning over recall
- What’s new: Integrating reasoning-centric sources (TheoremQA, SciBench, STEM) and curated harder items from MMLU (Sections 3.1–3.2).
- Why it matters:
  - CoT helps significantly on MMLU-Pro but not on MMLU. Example: `GPT-4o` sees +19.1% with CoT on MMLU-Pro (72.6 vs. 53.5) but only +1.5% on MMLU (Table 3; Section 6.2).
  - Subject-specific patterns align with reasoning difficulty (Section 5.2): Engineering and Law are especially challenging; Math and Physics show wide model spread (70%+ down to ~20%).

3) Two-phase expert review reinforced by LLM-assisted error finding
- What’s new: After human checks, `Gemini-1.5-Pro` is used to detect “false negative options,” which are then human-verified (Section 3.2).
- Why it matters:
  - Reduces mislabeled answers and unsuitable items (Table 1 reports hundreds to thousands of corrections by source), addressing the “lower ceiling” and noise issues suspected in MMLU.

4) A benchmark that discriminates among strong models
- Evidence:
  - The gap between top models expands: on MMLU, `GPT-4o` vs. `GPT-4-Turbo` differs by ~1–2%; on MMLU-Pro the gap is ~9% (Figure 4; Table 2).
  - Open-source leaders (`Llama-3-70B-Instruct`) approach mid-tier closed models but still trail top closed models (Table 2), enabling clearer placement in capability tiers.

These are primarily fundamental design changes (10 options, data-source shift, expert review) rather than incremental tweaks.

## 5. Experimental Analysis
Evaluation setup
- Models: 50+ LLMs across closed-source (e.g., `GPT-4o`, `Claude-3-Opus`, `Gemini-1.5-Pro`) and open-source families (`Llama-3`, `Phi-3`, `Qwen`, `Mixtral`, etc.) (Section 5; Appendix A.3).
- Prompting: mostly 5-shot CoT; `Gemini-1.5-Pro/Flash` use 0-shot (Table 2 note).
- Metric: accuracy.
- Answer parsing: two regex passes; random guess fallback if both fail (Section 4).
- Robustness: 24 prompt variants (Figure 5).

Main results (Table 2; Figures 4–5; Sections 5–6)
- Overall ranking and separation
  - > “GPT-4o [achieves] 72.6%” (Table 2).
  - `Gemini-1.5-Pro`: 69.0%; `Claude-3-Opus`: 68.5%; `GPT-4-Turbo`: 63.7% (Table 2).
  - Best open-source: `Llama-3-70B-Instruct` at 56.2% (Table 2).
  - Separation improves relative to MMLU: Figure 4 shows the same models cluster around 78–82% on MMLU but spread to ~53–73% on MMLU-Pro, widening gaps among top systems.

- Subject-specific insights (Section 5.2)
  - Reasoning-heavy areas (Math, Physics): large disparities (e.g., models range from >70% to low 20%); strong discriminativeness.
  - Engineering and Law: lower scores overall. Engineering is harder due to many multi-step, formula-based questions from STEM sources; Law requires nuanced legal reasoning with more options.
  - Knowledge-heavy areas (History, Psychology): higher floors overall, but some models (e.g., `DeepSeek-V2-Chat`) appear relatively stronger in reasoning than in knowledge retrieval.

- Chain of Thought vs. Direct Answering (Section 6.2; Table 3)
  - CoT boosts on MMLU-Pro are substantial:
    - `GPT-4o`: 72.6% (CoT) vs. 53.5% (Direct) → +19.1%.
    - `GPT-4-Turbo`: 63.7% vs. 48.4% → +15.3%.
    - Smaller but positive for `Phi-3-medium` (+8.2%), `Llama-3-8B` (+3.9%), `Gemma-7B` (+6.7%).
  - On MMLU, CoT offers small or even negative changes for some models (e.g., `Llama-3-8B`: –3.9%).

- Prompt robustness (Section 6.3; Figure 5)
  - With 24 prompt variants:
    - On MMLU: typical score variance is 4–5%, max up to 10.98%.
    - On MMLU-Pro: typical variance ~2%, max 3.74%.
  - Interpretation: more options and harder distractors dampen prompt-induced volatility.

- Error analysis on `GPT-4o` (Section 5.3; Appendix A.6)
  - Of 120 mistakes analyzed:
    - 39% due to flawed reasoning chains (e.g., adding rather than subtracting pressures; Table 10).
    - 35% due to lack of specific domain knowledge (e.g., optics refractive index ratios; Table 9; finance cash-balance definition; Table 8).
    - 12% due to calculation errors (e.g., molar mass addition mistake; Table 11).
    - Remaining: no selection made (5%), question understanding (4%), generation issues (2%), annotation errors (2%), answer extraction errors (1%).
  - These categories validate that MMLU-Pro stresses reasoning and precise domain application, not just recall.

- Dataset curation statistics (Section 3.2; Table 1)
  - Human and LLM-assisted reviews found substantial numbers of incorrect answers, false-negative distractors, and bad questions—especially in MMLU and STEM-derived items—underscoring the need for cleaning.

Do the experiments support the claims?
- Harder and more discriminative: Yes. Figure 4 and Table 2 show large accuracy drops vs. MMLU and wider gaps among top models.
- More reasoning-centric: Yes. CoT boosts on MMLU-Pro (Table 3) are large and consistent, unlike on MMLU.
- More robust to prompt variation: Yes. Figure 5 shows reduced variance across 24 prompts.

Caveats in the setup that readers should keep in mind
- Two Gemini models are evaluated 0-shot while others use 5-shot CoT (Table 2), which may depress their scores relative to peers.
- The answer-extraction fallback to a random guess can penalize models that violate output format; in the error analysis, “No selection made” accounts for 5% of `GPT-4o`’s examined errors (Appendix A.6).
- `GPT-4-Turbo` is used to generate distractors; the paper reports “no additional advantage” for `GPT-4-Turbo` from this (Section 3.2), but this is not shown via an ablation.

## 6. Limitations and Trade-offs
- Multiple-choice format only (Section 7)
  - Trade-off: standardizes scoring and supports large-scale evaluation, but cannot capture richer, open-ended reasoning, proof quality, or explanation faithfulness.

- No multimodal items (Section 7)
  - Excludes tasks requiring images/tables/figures by design; this narrows applicability for multimodal LLMs.

- LLM-in-the-loop data construction
  - Risks: stylistic artifacts in distractors (generated by `GPT-4-Turbo`) might align with some model families; the paper asserts no measured advantage but lacks a formal ablation (Section 3.2).
  - Mitigation: human review plus `Gemini-1.5-Pro` checking of false negatives (Section 3.2; Table 1).

- Screening “easy” items with mid-sized models (Section 3.2)
  - Assumption: questions solved by >4 of 8 mid-sized LLMs are too easy for frontier evaluation. This may bias the retained set toward items that defeat mid-sized models in specific ways (e.g., format peculiarities) rather than purely “hard” content.

- Prompting and parsing constraints
  - The evaluation depends on specific output formatting and regex-based extraction (Section 4); formatting deviations lead to random guesses. For some models, this contributes to errors (Appendix A.6).

- Shot-count inconsistency across models
  - Gemini models evaluated in 0-shot while others use 5-shot CoT (Table 2). This complicates direct fairness comparisons.

## 7. Implications and Future Directions
- Field impact
  - MMLU-Pro re-establishes headroom and reliability for general-purpose LLM evaluations. It turns CoT into a clear advantage on a mainstream benchmark, re-focusing the community on deliberate reasoning quality (Section 6.2; Table 3).
  - The reduced prompt sensitivity (Figure 5) encourages more stable leaderboards and fairer comparisons across prompt templates.

- Practical applications
  - Better model selection for high-stakes domains where reasoning drives performance (engineering, law, quantitative finance, clinical reasoning).
  - Training and alignment targets: since error analysis quantifies reasoning, knowledge, and calculation failures (Section 5.3), model developers can prioritize the largest error modes for improvement.

- Follow-up research directions
  - Open-ended and verifiable reasoning: move beyond multiple choice to proofs, structured derivations, or program-assisted solutions (acknowledging Section 7’s MCQ limitations).
  - Multimodal extension: incorporate diagrams, tables, and figures for science and engineering—currently excluded by design (Section 7).
  - Distractor generation ablations: systematically test how distractor sources/styles affect fairness (e.g., generate with diverse LLMs; human-only variants).
  - Robustness beyond prompting: study adversarial phrasing, option order randomization, and cross-lingual variants to further validate stability.
  - Tool use integration: given that 12% of top-model errors are calculation-related (Appendix A.6), evaluate under calculator or code-execution assistance to isolate reasoning vs. arithmetic limitations.

> Headline results to remember:
> - Table 2: `GPT-4o` 72.6% overall on MMLU-Pro; `GPT-4-Turbo` 63.7%; `Llama-3-70B-Instruct` 56.2%.
> - Figure 4: accuracy gap between strong models widens on MMLU-Pro; overall scores drop by 16–33% vs. MMLU.
> - Table 3: CoT boosts up to +19.1% (`GPT-4o`) on MMLU-Pro, but offers little or negative change on MMLU.
> - Figure 5: prompt sensitivity: ~2% on MMLU-Pro vs. ~4–5% on MMLU (max 3.74% vs. 10.98%).
> - Section 5.3: error modes for `GPT-4o`: 39% reasoning, 35% knowledge, 12% calculation, with smaller shares from formatting/parsing and annotation issues.

Overall, MMLU-Pro is a carefully constructed step toward a harder, fairer, and more informative evaluation of LLM reasoning, while acknowledging that future work should broaden beyond multiple-choice and into multimodal and open-ended settings.
