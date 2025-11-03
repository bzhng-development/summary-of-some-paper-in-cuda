# MMLU‑Pro: A More Robust and Challenging Multi‑Task Language Understanding Benchmark

**ArXiv:** [2406.01574](https://arxiv.org/abs/2406.01574)
**Authors:** Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, Wenhu Chen
**Institutions:** —

## 🎯 Pitch

MMLU-Pro introduces a robust benchmark for large language models by expanding multiple-choice options to 10, focusing on college-level reasoning problems, and implementing rigorous error audits. This innovation resolves issues of saturation and prompt sensitivity in previous benchmarks, offering a more reliable and discriminative evaluation tool essential for advancing and differentiating LLM capabilities effectively.

---

## 1. Executive Summary
MMLU-Pro is a new benchmark for evaluating large language models (LLMs) that replaces the widely used MMLU when models cluster at the top and small prompt tweaks reorder the leaderboard. It makes evaluation harder and more reliable by (a) expanding multiple-choice questions from 4 to 10 options, (b) shifting the mix toward reasoning-heavy, college-level problems, and (c) aggressively auditing questions and answer options for errors. Results on 50+ models show large, informative gaps between systems, strong benefits from Chain-of-Thought (CoT) reasoning, and far lower sensitivity to prompt phrasing.

## 2. Context and Motivation
- The gap addressed
  - Existing general-knowledge benchmarks—especially `MMLU`—are saturated: many frontier models score ~86–87% with tiny differences that don’t clearly separate capabilities. The paper documents this plateau since early 2023: GPT‑4 at 86.4% (Mar 2023) and subsequent frontier models around 86–87% with GPT‑4o only at 87.4% (Introduction).
  - Scores are unstable: small prompt or scoring changes shift rankings by several percent. Prior work finds high prompt sensitivity; this paper measures 4–5% range on MMLU across 24 prompt styles, peaking at 10.98% (Section 6.3, Figure 5).

- Why it matters
  - Practitioners and researchers need benchmarks that (1) discriminate meaningfully between strong models, (2) reflect reasoning ability rather than memorization or shortcut use, and (3) are robust to prompt wording so leaderboards are trustworthy.

- Root causes in existing MMLU (Introduction)
  1. Only 3 distractors per question. With 4 options, elimination heuristics and lucky guesses inflate scores.
  2. Questions are largely knowledge-driven; CoT often hurts or doesn’t help on MMLU (Section 6.2, Table 3), suggesting limited reasoning burden.
  3. Nontrivial noise: some questions are unanswerable or mislabeled, depressing ceilings and adding variance.

- Positioning relative to prior work
  - MMLU-Pro retools the MMLU format and curation to tackle the above causes. It also integrates additional sources of college-level STEM questions and performs a two-stage expert audit (Section 3; Table 1).
  - Compared with other broad benchmarks (AGIEval, ARC, BIG-bench, HELM), MMLU-Pro focuses on discriminative, reasoning-centric multiple-choice assessment at scale.

## 3. Technical Approach
The benchmark is a dataset and an evaluation protocol designed to increase difficulty, reduce noise, and stabilize scoring.

- Dataset composition and scope
  - 12,032 questions across 14 disciplines (math, physics, chemistry, law, engineering, psychology, health, business, economics, biology, philosophy, computer science, history, other). See distribution in Figure 3a.
  - Sources: 56.6% from MMLU (curated), 33.9% STEM websites, 5.0% TheoremQA, 4.5% SciBench (Figure 3b).

- Pipeline overview (Figure 2)
  1. Initial filtering of original MMLU
     - Merge 57 subjects into 14 categories for coverage without redundancy.
     - Remove “too easy” items: evaluate eight modest LLMs; if >4 models answer a question correctly, drop it. This excludes 5,886 questions (Section 3.2; Appendix A.1 Table 4 details per-discipline filtering rates).
     - Rationale: quickly eliminate trivial items using a model committee rather than ad-hoc heuristics.
  2. Question collection and integration
     - Add higher-difficulty STEM problems from online sources, TheoremQA (theorem-application questions), and SciBench (college exam problems). Many arrive as problem+solution (not MCQ).
     - Convert to MCQ: use `GPT-4‑Turbo` to extract a short, unambiguous answer from the solution and generate three plausible distractors; then human reviewers verify the extraction and discard items where the extracted answer is incomplete or incorrect (Section 3.2).
  3. Option augmentation from 4 to 10
     - Use `GPT-4‑Turbo` to produce six additional plausible—but wrong—options for each MCQ, raising options to A–J (Section 3.2).
     - Purpose: lower chance accuracy (25% → 10%) and require finer reasoning to distinguish close contenders.
     - Mitigation of fairness concerns: experiments suggest GPT‑4‑Turbo does not get extra advantage from having generated distractors (Section 3.2).
  4. Two-phase expert review (Section 3.2; Table 1)
     - Phase 1: Human experts verify correct answers; remove unsuitable questions (needing images/tables, missing info, proofs/open-ended forms).
     - Phase 2: Model-assisted “false negative” sweep. Use `Gemini‑1.5‑Pro` to flag any distractor that might also be correct; human experts adjudicate and remove those. Outcomes (Table 1):
       - Incorrect answers found: 350 (MMLU), 11 (SciBench), 483 (STEM Website).
       - False negative options: 1,953 (MMLU), 293 (STEM Website), etc.
       - Bad questions removed (needing non-text, insufficient info, etc.): 385 (MMLU), 862 (STEM Website), etc.
     - Final option counts: 83% of items have 10 options; 17% fewer; average 9.47 options (Section 3.2).
- Evaluation protocol (Section 4)
  - Prompting style: 5-shot Chain-of-Thought (CoT) for all except `Gemini-1.5-Pro/Flash` (0-shot). CoT demos are adapted from Chain-of-Thought Hub, with 5 representative examples per discipline (Appendix A.2, Table 7 shows a full physics prompt and five step-by-step exemplars).
  - Why CoT: emphasize reasoning rather than language fluency; paper later shows CoT confers large gains specifically on MMLU-Pro (Section 6.2; Table 3).
  - Answer extraction: regex to capture “The answer is (X)”. If format deviates, a backup regex is used; failing both, a random option is chosen to avoid missing outputs (Section 4).
  - Robustness audits: evaluate under 24 reasonable prompt variants and report score ranges (Section 6.3; Figure 5).

## 4. Key Insights and Innovations
- 10-option multiple choice with carefully crafted distractors
  - What’s new: expand from 4 to 10 options and ensure plausibility via LLM generation + human verification (Section 3.2).
  - Why it matters: lowers guess accuracy from 25% to 10% and reduces shortcutting by elimination. Empirically, this widens performance gaps among models (Figure 4) and lowers prompt sensitivity (Figure 5).

- Curated difficulty shift toward reasoning
  - What’s new: systematically filter out trivial MMLU items using a committee of 8 LLMs; add many college-level STEM problems requiring derivations/multi-step calculation (Sections 3.2 and 5.2).
  - Why it matters: forces models to actually reason. Evidence: CoT boosts accuracy strongly on MMLU-Pro (+19.1% for `GPT‑4o`) while it barely helps or even hurts on MMLU (Table 3, Section 6.2).

- Two-stage error auditing pipeline with model-assisted “false negative” detection
  - What’s new: after human correctness checks, use a strong LLM (`Gemini-1.5-Pro`) to scan distractors for potentially correct alternatives and return them for human adjudication (Section 3.2; Table 1).
  - Why it matters: reduces dataset noise (mislabels, ambiguous distractors), raising the reliability ceiling and stabilizing evaluation.

- Demonstrated robustness to prompt style
  - What’s new: 24-prompt evaluation and explicit reporting of score ranges on both MMLU and MMLU-Pro (Section 6.3; Figure 5).
  - Why it matters: on MMLU-Pro, ranges shrink to about 2% on average (max 3.74%), compared to 4–5% on MMLU (max 10.98%), improving leaderboard trustworthiness.

These are more than incremental changes to MMLU formatting; they alter the evaluation regime to emphasize verifiable reasoning and dataset reliability.

## 5. Experimental Analysis
- Setup overview (Sections 4–5; Appendix A.3)
  - 50+ LLMs evaluated: closed-source (`GPT‑4o`, `GPT‑4‑Turbo`, `Claude‑3‑Opus/Sonnet`, `Gemini‑1.5‑Pro/Flash`, `Yi-Large`) and open-source (`Llama‑3‑70B/8B`, `Qwen`, `Phi‑3`, `DeepSeek‑V2`, `Mixtral`, etc.).
  - Metric: accuracy on multiple-choice selection (A–J).
  - Prompting: 5-shot CoT by default, with per-discipline exemplars; answer parsed by regex.
  - Additional checks: 24 prompt variants for robustness; direct-answer vs CoT comparisons.

- Main quantitative results
  - Overall accuracy (Table 2, CoT setting):
    - “Frontier” closed systems: `GPT‑4o` 72.6% (best), `Gemini‑1.5‑Pro` 69.0%, `Claude‑3‑Opus` 68.5%, `GPT‑4‑Turbo` 63.7%.
    - Best open-source: `Llama‑3‑70B‑Instruct` 56.2%, `Phi‑3‑medium‑4k‑instruct` 55.7%, `DeepSeek‑V2‑Chat` 54.8%.
  - Per-subject patterns (Table 2; Section 5.2):
    - Reasoning-heavy (Math/Physics): large spread—from ~70%+ for `GPT‑4o` to ~20–30% for smaller models; strong discriminative signal.
    - Knowledge-heavy (History/Psychology): higher floors; `DeepSeek‑V2‑Chat` underperforms relative to its reasoning abilities.
    - Engineering and Law: notably hard; Engineering difficulty driven by new STEM Website items requiring derivations and multi-step calculations; Law requires nuanced legal reasoning.
  - Discriminativeness vs MMLU (Section 6.1; Figure 4):
    - On MMLU, multiple top models cluster in the 78–82% band; on MMLU‑Pro those same models spread roughly 10% apart.
    - Example gap: `GPT‑4o` vs `GPT‑4‑Turbo` is ~1% on MMLU but 9% on MMLU‑Pro.
    - Difficulty increase: top score drops from ~88.7% (MMLU, `GPT‑4o`, CoT) to 72.6% (MMLU‑Pro), leaving 27.4% headroom (Section 6.1).
  - CoT vs direct answering (Section 6.2; Table 3):
    > `GPT‑4o`: 88.7% (MMLU, CoT) vs 87.2% (direct), +1.5%; but 72.6% (MMLU‑Pro, CoT) vs 53.5% (direct), +19.1%.
    > `GPT‑4‑Turbo`: −0.2% change on MMLU with CoT, but +15.3% on MMLU‑Pro.
    - Pattern: CoT helps far more on MMLU‑Pro, evidencing the benchmark’s reasoning demands.
  - Prompt robustness (Section 6.3; Figure 5):
    > MMLU prompt variation: typically 4–5% range; worst 10.98%.
    > MMLU‑Pro prompt variation: around 2% range; worst 3.74%.
    - Interpretation: broader choice sets and higher-quality distractors dampen format sensitivity.

- Error analysis (Section 5.3; Appendix A.6)
  - 120 random `GPT‑4o` mistakes categorized:
    > 39% reasoning flaws, 35% lack of specific domain knowledge, 12% calculation errors. Remaining: no final selection (5%), question understanding (4%), generation issues (2%), annotation errors (2%), answer-extraction errors (1%).
  - Concrete examples (Appendix A.6):
    - Financial principal misapplied for installment interest (Table 8).
    - Optics mis-formula: subtracting indices instead of using an index ratio for lenses in media (Table 9).
    - Pressure difference added instead of subtracted in a piston/boiling problem (Table 10).
    - Molecular weight miscalculation in Graham’s law application (Table 11).
    - Generation loop producing repeated sentences (Table 12).
    - Misreading Singer’s equality principle scope (Table 13).
  - Takeaway: even the strongest model fails on multi-step reasoning chains and precise domain formulas, which aligns with the benchmark’s design goals.

- Do the experiments support the claims?
  - Yes for discriminativeness and reasoning: widened model gaps (Figure 4), strong CoT gains on MMLU‑Pro (Table 3), and detailed failure cases back the reasoning emphasis.
  - Yes for robustness: measured score ranges over 24 prompts shrink markedly on MMLU‑Pro (Figure 5).
  - Caveats:
    - Two `Gemini‑1.5` models were run 0-shot CoT (Table 2 note) while others used 5-shot; this might slightly understate their peak performance.
    - No ablation isolating the contribution of each curation step (e.g., effect of 10 options vs expert sweep alone).

## 6. Limitations and Trade-offs
- Multiple-choice format remains a simplification (Section 7)
  - Even with 10 options, MCQ can’t fully measure generative reasoning, solution explanations, or open-ended creativity.
  - Some elimination heuristics may still work; question framing can advantage models trained on MCQ-style corpora.

- Modality restriction
  - Non-textual questions (images/tables) are removed; the benchmark is text-only (Section 7, Expert Review Phase 1 criteria). Real-world problem-solving often needs multimodal reasoning.

- Potential generation bias
  - Distractors are partly LLM-generated. While human-verified and checked for false negatives (Section 3.2; Table 1), LLM generation could imprint stylistic artifacts that certain models exploit—or penalize models aligned differently.

- Evaluation protocol choices
  - Regex parsing of answers introduces a small failure mode (1% extraction errors among analyzed mistakes; Appendix A.6). Although fallback random choice avoids missing values, it injects noise at a very low rate.
  - Prompting differences (5-shot for most, 0-shot for some) can complicate absolute comparisons across models.

- Scope of difficulty
  - Engineering and Law are difficult partly due to newly added, derivation-heavy or intricate items (Section 5.2). This is desirable for stress-testing but may diverge from standard curricula in some regions.

- Computational and data considerations
  - Large-scale evaluation is nontrivial: closed-model API usage involved ~20M input and ~5M output tokens; open-source runs use A100s with vLLM acceleration (Appendix A.4).

## 7. Implications and Future Directions
- Shifting the field’s evaluation baseline
  - MMLU-Pro demonstrates a practical recipe for a modern, discriminative LLM benchmark: increase option count, emphasize reasoning, and systematically de-noise. Its prompt robustness and headroom make it suitable as a default leaderboard for near-term progress tracking.

- Research enabled
  - Reasoning methods: Because CoT is clearly beneficial on MMLU‑Pro (Table 3), the benchmark is well-suited for evaluating advanced reasoning strategies (self-consistency, tool use, program-aided reasoning).
  - Error-targeted improvements: The taxonomy of `GPT‑4o` mistakes (reasoning vs. domain knowledge vs. calculation; Section 5.3) points to targeted interventions—external calculators/solvers, retrieval-augmented domain knowledge, and verification layers.
  - Data quality frameworks: The two-phase expert+LLM audit (Section 3.2) can guide dataset construction beyond benchmarks, e.g., in education or compliance QA, to root out ambiguous or mislabeled items.

- Practical applications
  - Model selection for enterprise: With clearer separation and lower prompt sensitivity (Figure 5), MMLU‑Pro scores can better inform procurement/QA decisions.
  - Curriculum and tutoring systems: The reasoning-heavy questions reflect college-level problem-solving; models performing well on MMLU‑Pro are more likely to support step-by-step tutoring and exam prep reliably.

- Future work directions suggested by the paper (Section 7 and overall discussion)
  - Beyond MCQ: complementary benchmarks with open-ended reasoning, structured proofs, or code-executed solutions for verifiability.
  - Multimodal extension: incorporate images/tables to evaluate realistic STEM workflows.
  - Deeper ablations: quantify the individual effects of 10-choice expansion, specific sources (STEM/TheoremQA/SciBench), and each review phase.
  - Adversarial and dynamic distractors: iterative, model-in-the-loop generation to maintain difficulty as models improve.

In sum, MMLU‑Pro is a carefully engineered step toward measuring genuine reasoning in LLMs at scale: it makes guessing harder, surfaces real model differences, puts pressure on step-by-step thinking, and yields more stable leaderboards—all substantiated by the reported figures and tables (Figures 4–5; Tables 1–3, 7; Section 5.3 error analyses).
