# Premise Order Matters in Reasoning with Large Language Models

**ArXiv:** [2402.08939](https://arxiv.org/abs/2402.08939)

## 🎯 Pitch

This paper reveals a critical weakness in large language models (LLMs): their reasoning accuracy drops drastically when the order of premises is altered, even though the logical outcome should remain unchanged. By introducing new benchmarks for logical and mathematical reasoning, the authors show that LLMs’ reliance on left-to-right sequential processing leads to brittleness, undermining their reliability in real-world scenarios where information order is uncontrolled—highlighting the urgent need for models that can reason based on structure, not sequence.

---

## 1. Executive Summary
This paper shows that large language models (LLMs) are surprisingly sensitive to the order in which premises are presented, even when order should not affect the correct conclusion. By building two benchmarks—one for synthetic logical proofs and one for math word problems (R-GSM)—the study demonstrates large accuracy drops (often 20–40% and sometimes more) when premises are shuffled, and it explains why: models tend to reason left-to-right and struggle when they must jump around.

## 2. Context and Motivation
- Problem addressed
  - In deductive and mathematical reasoning, the truth of a conclusion does not depend on the order of the premises (e.g., “If A→B, If B→C, A” should yield C in any order). The paper investigates whether LLMs are order-invariant on such tasks and, if not, how and how much order matters.
- Why it matters
  - Practical: Users and applications cannot control the order of evidence in long contexts (documents, tool outputs, retrieval results). If order harms reasoning, systems will be brittle.
  - Scientific: Reasoning should be structural, not positional. Order sensitivity suggests LLMs rely on sequential heuristics shaped by autoregressive training rather than robust logical inference.
- Prior approaches and gaps
  - Known weaknesses include the Reversal Curse (“A is B” ≠ “B is A”), distractibility by irrelevant context, and long-context position biases. But a comprehensive, controlled study of premise order—in settings where order should be irrelevant—was missing.
- Positioning relative to existing work
  - The study isolates order effects:
    - Uses only `modus ponens` problems in logic (a simple, human-easy form of deduction) to avoid confounds from complex logic (Section 2.1).
    - Creates R-GSM by reordering sentences of GSM8K math problems while keeping answers unchanged (Section 2.2).
  - It also distinguishes premise-order effects from long-context “lost-in-the-middle” by keeping inputs short and running position ablations (Appendix D, Table 5).

## 3. Technical Approach
Two complementary evaluations are constructed.

1) Logical reasoning benchmark (Section 2.1)
- Problem format
  - Each instance contains: (a) a set of true `facts` (e.g., `A1 … An`), (b) a set of `rules` of the form “If X then Y” or “If X0 and X1 (and X2) then Y,” and (c) a `conclusion` (“C is True”) that is always derivable.
  - Only `modus ponens` is required: from “If P then Q” and “P,” infer “Q.”
- Why this design
  - Limits confounds to isolate ordering effects: no need for complex logical theorems or proof search beyond chaining definite clauses.
- Order manipulation
  - `Forward order` (τ = 1): rules listed in the same sequence they are used in a correct forward-chaining proof (proof can be written on-the-fly as you read; see Figure 1 left).
  - `Backward order` (τ = −1): exactly reversed order, aligned to backward chaining (start from goal and work backward).
  - `Shuffled/Intermediate orders`: categorized using normalized `Kendall tau distance` τ ∈ [−1, 1], a measure of similarity between two orderings. τ≈0 means no correlation with the proof order (Section 2.1).
  - `Distracting rules`: additional rules not needed for the proof; set to 0, 5, or 10 per instance to study distractibility (Section 2.1).
- Dataset scale
  - 200 base problems for each number of required rules (4–12). Each base problem is expanded into 15 variants combining different τ values and distractor counts, totaling 27K problems (Section 2.1).
- Prompting and scoring
  - Zero-shot prompts require the model to produce a step-by-step derivation and to indicate which premise is used at each step (Section 3.1; Figure 1).
  - An answer is correct only if the entire proof is valid and uses only given facts and rules. Hallucinated rules/facts or unjustified steps are marked wrong (Section 2.1; Table 1 shows error taxonomy).

2) R-GSM mathematical reasoning benchmark (Section 2.2)
- Construction
  - Start from GSM8K test problems with ≥5 sentences. Keep the final question sentence fixed.
  - Reorder the other sentences while preserving the correct answer. Minor wording edits are allowed to keep grammar natural.
  - A simple enumeration function suggests candidate reorderings; humans verify that the answer is unchanged and that the reordering is natural (Section 2.2).
  - Final dataset: 220 pairs (original vs. reordered). Dataset statistics in Appendix A, Table 4.
- Why this design
  - Tests order effects in natural language math problems, not just synthetic logic. Figure 2 shows a typical failure where the required computation does not follow the sentence order.

3) Models and evaluation setup (Section 3.1)
- Models: `GPT-4-turbo`, `GPT-3.5-turbo`, `PaLM 2-L`, `Gemini 1.0 Pro`.
- Decoding: greedy (temperature 0).
- Prompts: zero-shot; logic tasks include derivation instructions; R-GSM uses only the problem text.

Analogy to clarify the mechanism
- Think of the model as reading a list left-to-right while trying to build a proof. If the next rule in the list can be applied immediately, the model moves forward smoothly. If not (because the needed antecedent appears elsewhere), a robust reasoner would search globally. Instead, these LLMs often “force” progress—sometimes hallucinating a needed fact—or they give up, revealing a bias toward sequential, local reasoning.

## 4. Key Insights and Innovations
- A. Order-invariance is broken—and it matters a lot.
  - Across models and tasks, accuracy is highest when premises follow the proof order (`τ = 1`) and drops markedly when shuffled or reversed (Figures 3–6; Tables 6–11). This reveals a fundamental sensitivity that standard evaluations masked.
- B. Order interacts with model-specific reasoning styles.
  - `GPT-4-turbo` often does second-best on `backward order` (`τ = −1`), consistent with backward chaining (Figure 5a; Table 9a). `PaLM 2-L` performs worst on backward order (Figure 5b; Table 9b). Thus, not all non-forward orders are equally harmful; some align with an LLM’s implicit strategy.
- C. New benchmarks that isolate ordering effects.
  - A large synthetic logic suite (27K variants) with controlled τ, distractors, and proof length (Section 2.1), plus `R-GSM`—220 natural math problem pairs where only sentence order changes (Section 2.2).
- D. Mechanistic error analysis: “sequential pressure” drives hallucinations.
  - With decreasing τ, errors shift toward `fact hallucination`—inventing just-in-time facts to apply the next seen rule (Table 1 and Figure 10). This highlights a left-to-right reasoning bias arising from autoregressive training.
- E. Not just long-context “lost-in-the-middle.”
  - Position ablation holding order constant shows little variation (PaLM 2-L; Table 5), and total tokens are short (<300). The effect is due to ordering relative to proof structure, not proximity to the beginning or end (Appendix D).

## 5. Experimental Analysis
- Evaluation methodology
  - Logic tasks: accuracy = percentage of fully valid proofs (no hallucinations) produced with step references (Section 3.1).
  - R-GSM: standard exact-answer accuracy on original vs. reordered versions (Table 2).
  - Zero-shot prompting; greedy decoding; no chain-of-thought exemplars beyond the derivation instruction for the logic tasks.
- Main quantitative results
  - Logical reasoning without distractors (Figure 3; Table 6):
    - `GPT-4-turbo` at 12 rules: Forward 96.5%, Backward 84.0%, Shuffled 80.8% (Table 6a).
    - `PaLM 2-L` at 12 rules: Forward 88.0%, Backward 57.5%, Shuffled 66.5% (Table 6b).
    - `Gemini 1.0 Pro` at 12 rules: Forward 16.5%, Backward 0.5%, Shuffled 0.2% (Table 6c).
    - `GPT-3.5-turbo` at 12 rules: Forward 30.0%, Backward 1.0%, Shuffled 1.2% (Table 6d).
    - Trend: accuracy gap grows with more rules; forward order dominates.
  - Logical reasoning with distractors (Figures 4 and 6; Tables 7–8, 10–11):
    - With 10 distractors and 12 rules: `GPT-4-turbo` drops to 57.5% (forward), 46.5% (backward), 40.0% (shuffled) (Table 8a). `PaLM 2-L` drops to 36.5% (forward), 15.5% (backward), 18.2% (shuffled) (Table 8b).
    - Distractors magnify the order effect (compare Figure 3 vs. Figure 4, and Figure 5 vs. Figure 6).
  - Breakdown by τ without distractors (Figure 5; Table 9):
    - `GPT-4-turbo` degrades roughly with |τ| decreasing; backward often second-best (Table 9a).
    - `PaLM 2-L` is particularly harmed by backward order (Table 9b).
  - Error analysis on logic (Table 1; Figure 10):
    - For `GPT-4-turbo` at 12 rules: `fact hallucination` increases from 1.5% at τ=1 to 12.5% at τ=−1; `wrong refutation` is lowest at τ=−1 (0.0%) (Table 1).
    - For `GPT-3.5-turbo`, the failure rate is high at non-forward orders; `fact hallucination` reaches 47.0% at τ=−1 (Table 1).
  - R-GSM overall (Table 2a):
    - Accuracies drop on reordered versions: `GPT-4-turbo` 94.1%→85.0%; `PaLM 2-L` 86.4%→79.5%; `Gemini 1.0 Pro` 80.5%→69.1%; `GPT-3.5-turbo` 67.3%→51.8%.
    - On the subset each model solves initially (Table 2b), at least 10% of those are lost after reordering; for `GPT-3.5-turbo`, accuracy falls from 100% to 64.9%.
  - R-GSM by task complexity (Figures 7–8; Tables 12–13):
    - More reasoning steps and more sentences increase the gap.
    - Example: for `GPT-4-turbo`, ≥6 steps: 89.8%→73.5% (Table 12a). For ≥7 sentences: 86.4%→68.2% (Table 13a).
  - R-GSM error analysis (Table 3; Figures 2 and 9):
    - Most common failure: ignoring `temporal order`—using numbers in textual order rather than real-world time order (45% of failures for GPT-4-turbo).
    - Another failure: acting as if required quantities are known when they are introduced later (Unknown-variables category, e.g., Figure 9).
- Robustness and diagnostics
  - The “lost-in-the-middle” confound is ruled out by short contexts and position ablations (Appendix D, Table 5).
  - Qualitative examples illustrate rule/fact hallucinations (Figure 10) and different τ configurations (Figure 11).

Assessment
- The experiments are well-controlled for the target question (order sensitivity) and span both synthetic and natural data. The quantitative gaps are large and persistent across models and settings, convincingly supporting the claim that premise order meaningfully affects LLM reasoning. The main caveat is generalization beyond `modus ponens` logic and the limited set of models; still, parallel results on R-GSM mitigate the “synthetic-only” concern.

## 6. Limitations and Trade-offs
- Scope of logic
  - Only `modus ponens` with definite clauses is tested (Section 2.1). Real proofs often require diverse inference patterns (e.g., modus tollens, contradiction) and quantifiers.
- Proof evaluation
  - Correctness requires clean step attribution and no hallucinations. While strict, it may penalize semantically correct but differently structured proofs (though the study’s goal is consistency with given premises).
- Data generation and naturalness
  - Logical tasks are synthetic, which might not reflect real-world language variability; however, R-GSM addresses naturalness.
- Prompting mode
  - Zero-shot only (Section 3.1). Few-shot demonstrations or deliberate reasoning prompts might reduce order sensitivity; this is not explored.
- Model coverage
  - Four prominent LLMs are tested, but conclusions might shift with more recent or specialized models, or with tool-augmented setups.
- No proposed mitigation
  - The work diagnoses but does not introduce a training or architectural fix (Conclusion).

## 7. Implications and Future Directions
- How this changes the landscape
  - Premise order should become a first-class factor in evaluating and deploying LLM reasoners. Benchmarks that assume order irrelevance may overestimate reliability.
  - The findings point to an inherent left-to-right bias: current LLMs prefer reasoning paths that align with sequential reading and struggle with global search across premises.
- What research it enables or suggests
  - Order-robust training: objectives that encourage permutation invariance or multi-step search over unordered sets; e.g., contrastive training across premise permutations.
  - Architectural changes: modules for explicit premise selection, graph-based intermediate representations, or bidirectional planning (forward/backward chaining hybrids).
  - Prompting and tooling: reordering inputs to match a discovered proof order; retrieval systems that sort evidence by estimated utility for the next step; graph-of-thought or planner-guided CoT.
  - Error-aware decoding: detect stalled steps where no rule applies; trigger search or look-ahead rather than hallucinating facts (motivated by Table 1 patterns).
- Practical applications
  - Document QA, legal/medical reasoning, and multi-hop retrieval pipelines should:
    - Normalize or reorder evidence to align with an intended reasoning plan.
    - Avoid mixing relevant and irrelevant premises without structure; distractors not only degrade accuracy but amplify order sensitivity (Figures 4 and 6).
    - Use verification and step-checking tools to catch fact/rule hallucinations (Figure 10).

Key takeaway
> Premise order, though logically irrelevant, materially shapes the success of LLM reasoning. Aligning inputs with the intended proof sequence (or equipping models to discover that sequence) is a practical, high-leverage step toward more reliable reasoning.
