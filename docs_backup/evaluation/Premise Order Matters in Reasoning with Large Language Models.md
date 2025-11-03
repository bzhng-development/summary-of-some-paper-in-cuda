# Premise Order Matters in Reasoning with Large Language Models

**ArXiv:** [2402.08939](https://arxiv.org/abs/2402.08939)
**Authors:** Xinyun Chen, Ryan A. Chi, Xuezhi Wang, Denny Zhou
**Institutions:** Google DeepMind, Stanford University

## 🎯 Pitch

This study exposes a critical vulnerability in large language models, revealing that the order of premises significantly impacts reasoning accuracy, with deviations causing up to 40% performance drops. By establishing new benchmarks and analyzing error modes, it underscores the fragility of LLMs to input order, a pivotal insight for enhancing model reliability and robustness in real-world applications.

---

## 1. Executive Summary (2-3 sentences)
This study shows that large language models (LLMs) are highly sensitive to the order in which premises are presented, even when reordering does not change the task’s logical content. By constructing controlled logical benchmarks and a reordered math dataset (R‑GSM), the work demonstrates accuracy drops of up to 30–40% when premises are shuffled, analyzes error modes, and reveals ordering preferences that mirror human “read-as-you-reason” habits.

## 2. Context and Motivation
- Problem addressed
  - The paper examines whether the order of premises in a problem statement affects an LLM’s ability to reason, even when the logical conclusion is order-invariant. Section 1 motivates this with simple modus ponens chains (e.g., “If A→B; If B→C; A; therefore C”), where permutations do not alter truth.
- Why it matters
  - For practical systems, inputs are rarely curated to align with the “ideal” reasoning order. If ordering alone drives large swings in accuracy, deployed LLM systems can be unreliable, brittle to paraphrase, and vulnerable to adversarial or accidental reordering.
  - Theoretically, the phenomenon probes how autoregressive models perform multi-step inference: do they reason globally over the set of premises, or do they rely on a left‑to‑right, stepwise pattern?
- Prior work and gaps
  - Related failure modes include distractibility by irrelevant context (Shi et al., 2023), the “reversal curse” (Berglund et al., 2023), position bias (Liu et al., 2024), and limits of formal logic reasoning. However, those lines either focus on different kinds of order effects (e.g., “A is B” vs. “B is A”) or on long-context placement effects. This work isolates premise order while holding content and length constant, and it demands explicit proofs rather than yes/no answers.
- Positioning
  - The study builds tightly controlled logical tasks and a reordered math benchmark to show that “premise order matters,” quantifies the effect across multiple LLMs, and analyzes why errors occur (Sections 2–3). It also contrasts forward-chaining and backward-chaining preferences using Kendall tau correlations (Section 2.1 and Figures 5–6).

## 3. Technical Approach
- Two complementary evaluations
  1) Logical reasoning with controlled synthetic tasks
  2) Mathematical word problems with reordered sentences (R‑GSM)

- Key terms
  - `modus ponens`: a basic deduction pattern—“If P then Q; P; therefore Q.”
  - `definite clause`: a rule of the form “If X then Y,” where X can conjoin 1–3 facts (Section 2.1).
  - `forward order`: the premises are listed exactly in the sequence they are used by a forward‑chaining proof (Section 2.1).
  - `backward order`: the reverse of the forward order; aligns with backward chaining (goal first, then supporting rules).
  - `distracting rules`: premises that are unnecessary for the proof (irrelevant) (Section 2.1).
  - `Kendall tau distance` (`τ`): a correlation score between two orderings; normalized to [−1, 1], where 1 means identical to forward order, −1 is exact reverse, and ~0 is uncorrelated (Section 2.1).

- Logical reasoning benchmark (Section 2.1)
  - Problem format:
    - A set of true facts A1…An.
    - A set of rules: “If X then Y,” optionally with 2–3 antecedents.
    - A conclusion “C is True” that is provable.
  - Evaluation requires a proof: models must output step‑by‑step deductions, citing which fact or rule is used at each step. Any use of nonexistent rules/facts is an error. Every problem’s ground-truth label is True; correctness requires a fully valid proof.
  - Premise ordering control:
    - For each logical problem, variants are generated with different `τ` values: `1` (forward), `0.5`, `0`, `−0.5`, `−1` (backward). Figure 11 provides a concrete example of how the same rule set looks at different `τ`.
  - Difficulty knobs:
    - Number of rules actually used in the proof: 4–12.
    - Number of distracting rules: 0, 5, or 10.
  - Scale: 200 base problems for each rule-count, each expanded into 15 variants (5 orderings × 3 distractor settings) = 27,000 instances total (Section 2.1).

- R‑GSM (Section 2.2)
  - Built from GSM8K test problems with at least 5 sentences.
  - Sentences are reordered (last sentence kept), and light wording edits ensure grammatical correctness. The answer is manually verified to remain unchanged.
  - Construction procedure uses simple enumeration to find alternative orderings, then human verification/editing; total of 220 paired items (original and reordered) (Section 2.2; Appendix A for counts by steps/sentences).
  - Figure 2 illustrates a case where all models solve the original but fail the reordered version because correct computation requires non-sequential references.

- Models and inference setup (Section 3.1)
  - Evaluated LLMs: `GPT-4-turbo`, `GPT-3.5-turbo`, `PaLM 2‑L`, `Gemini 1.0 Pro`.
  - Decoding: zero-shot prompting, greedy decoding with temperature 0.
  - Logical tasks include an instruction to produce step‑by‑step derivations (Figure 1). R‑GSM uses only the problem text.

- Why these design choices?
  - Restricting to definite clauses and modus ponens removes confounds from advanced theorems and focuses on premise order (Section 2.1).
  - Requiring proofs prevents lucky guesses and exposes hallucinated steps (Section 2.1).
  - Kendall tau bins provide a principled spectrum from forward to reverse orders (Section 2.1).
  - R‑GSM tests whether ordering effects persist in natural language math problems beyond synthetic logic (Section 2.2).

## 4. Key Insights and Innovations
- A. Ordering sensitivity is large and systematic
  - Across models, accuracy is highest when premises follow the forward proof order. As premises deviate from that order, accuracy drops—often steeply—especially as the number of rules increases (Figures 3–5). This isolates ordering as a key variable independent of content or length.
- B. Backward order is not equivalent to random order
  - Some models prefer the backward order over random permutations. GPT‑4‑turbo in particular shows “forward > backward > others,” suggesting a capacity for backward chaining when the order explicitly supports it (Figure 5). In contrast, PaLM 2‑L performs worst on backward order.
- C. New benchmarks to probe order effects
  - The logical reasoning suite systematically varies `τ`, rule counts, and distraction; R‑GSM provides order-controlled GSM8K variants where the answer is unchanged (Section 2). Together, they constitute reusable tools for future research on inference robustness.
- D. Error taxonomy connects order to failure modes
  - By forcing explicit proofs, the study shows that decreasing `τ` increases “fact hallucination”—inventing missing intermediate facts to make the next listed rule “applicable” (Table 1 and Section 3.2). This links ordering sensitivity to greedy, sequential reasoning heuristics in autoregressive LLMs.
- E. Not a long-context artifact
  - Appendix D shows the “lost‑in‑the‑middle” effect is unlikely here: inputs are <300 tokens and moving relevant rules to the beginning/middle/end at fixed order barely changes accuracy (Table 5).

## 5. Experimental Analysis
- Evaluation design
  - Logical tasks: accuracy measured as proportion of fully correct proofs. Variables: number of relevant rules (4–12), number of distracting rules (0/5/10), and `τ ∈ {1, 0.5, 0, −0.5, −1}` (Sections 2.1 and 3.2). Figure 1 provides a representative failure where shuffling makes GPT‑4‑turbo reject a provable conclusion.
  - R‑GSM: measure exact-answer accuracy on original vs. reordered versions (Section 3.3; Tables 2, 12, 13). Also analyze errors by category: temporal-order mistakes, use of unknown variables, and other issues (Table 3).

- Core quantitative results
  - Forward order dominates without distractors (Figure 3; Table 6):
    - With 12 relevant rules and no distractors:
      - GPT‑4‑turbo: forward 96.5% vs backward 84.0% vs shuffled 80.8%.
      - PaLM 2‑L: forward 88.0% vs backward 57.5% vs shuffled 66.5%.
      - Gemini 1.0 Pro: forward 16.5% vs backward 0.5% vs shuffled 0.2%.
      - GPT‑3.5‑turbo: forward 30.0% vs backward 1.0% vs shuffled 1.2%.
    - The gap widens as rule count increases. As summarized in Section 3.2: 
      > “the performance drop caused by alternative orderings becomes more significant when the number of rules increases.”
  - Ordering preferences (Figure 5; Table 9):
    - GPT‑4‑turbo generally ranks `τ=1` (forward) highest and `τ=−1` (backward) second; accuracy decreases as |`τ`| shrinks. Example with 10 rules: 99.0% (τ=1) → 92.5% (τ=−1) → 82.5% (τ=0).
    - PaLM 2‑L is sensitive to reverse order: with 12 rules, 88.0% (τ=1) vs 57.5% (τ=−1).
  - Distractors magnify the order effect (Figures 4 and 6; Tables 7–8, 10–11):
    - With 12 relevant rules and 10 distractors:
      - GPT‑4‑turbo: forward 57.5% vs backward 46.5% vs shuffled 40.0% (Table 8a).
      - PaLM 2‑L: forward 36.5% vs backward 15.5% vs shuffled 18.2% (Table 8b).
    - Section 3.2 notes:
      > “adding distracting rules further decreases the reasoning performance and magnifies the effect of different premise orders.”
  - Error analysis on logical tasks (Table 1):
    - Fact hallucination rises sharply as `τ` decreases. For GPT‑4‑turbo, hallucinated facts go from 1.5% (τ=1) to 12.5% (τ=−1). GPT‑3.5‑turbo shows low correct rates overall and high wrong‑refutation rates when `|τ|<1`.
  - R‑GSM accuracy drops (Table 2):
    - Overall accuracy: GPT‑4‑turbo 94.1% → 85.0%; PaLM 2‑L 86.4% → 79.5%; Gemini 80.5% → 69.1%; GPT‑3.5‑turbo 67.3% → 51.8%.
    - On the subset each model originally solves (Table 2b), at least 10% of reordered problems fail for every model; for GPT‑3.5‑turbo the drop is from 100% to 64.9%.
  - R‑GSM breakdowns (Figures 7–8; Tables 12–13):
    - More reasoning steps and longer problems exacerbate the gap for GPT‑4‑turbo and Gemini. For instance, GPT‑4‑turbo’s accuracy on problems with ≥6 steps falls from 89.8% (original) to 73.5% (reordered) in Table 12a.
  - R‑GSM error taxonomy (Table 3; Figures 2, 9, 12–13):
    - Most dominant error: temporal-order mistakes—using numbers in the order of appearance instead of the correct causal/temporal dependency. Table 3 shows this is the largest category for GPT‑4‑turbo (45%) and substantial for others.

- Do the experiments support the claims?
  - Yes. The controlled logical setting carefully isolates order, and the proof requirement reveals mechanism-level failures (hallucinations, wrong refutations). The natural-language R‑GSM confirms that the effect extends beyond synthetic logic. The ablation in Appendix D addresses the confound of long‑context placement (Table 5).

- Robustness and failure cases
  - Failure examples in Figures 1–2, 9–13 show characteristic behaviors: sequential overreach (using the next rule prematurely), temporal misalignment in math problems, and hallucination when the next rule isn’t yet applicable. The consistent amplification of errors with more rules/distractors underscores brittleness under increased reasoning complexity.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Logical tasks are limited to propositional reasoning with definite clauses and modus ponens (Section 2.1). This focuses the study but leaves open whether the same effects hold for richer logics (e.g., quantifiers, negation-heavy proofs beyond simple cases).
  - R‑GSM includes 220 pairs (Appendix A), a carefully curated but relatively small dataset that emphasizes clarity of the phenomenon over breadth.
- Selection bias in R‑GSM construction
  - The creation process enumerates alternate orderings and keeps those that preserve the answer and sometimes induce failure (Section 2.2). This produces informative “order stress tests” but may overrepresent fragile configurations relative to random real-world phrasing.
- Model and prompting choices
  - Zero-shot, temperature‑0 decoding (Section 3.1) is a stringent setting that avoids sampling variance and elaborate instructions. Different prompting (e.g., ask the model to reorder or plan first) might mitigate some failures; the work intentionally does not pursue mitigation techniques.
- Scale and context length
  - Logical inputs are short (<300 tokens; Appendix D). Results do not address very long contexts or retrieval-augmented settings.
- Generality across tasks and languages
  - The paper studies English tasks and specific LLM versions. Cross-lingual and domain-general generalizations are not evaluated.

## 7. Implications and Future Directions
- How this changes the landscape
  - The work reframes “reasoning” evaluation: not only what premises are given matters, but also how they are ordered. It highlights a matching problem between a model’s internal inference style (largely left‑to‑right, greedy) and the presentation of information.
- Practical applications and mitigations
  - Input engineering: pre-process problem statements to a forward-chaining order, when possible, or offer both forward and backward chains.
  - Tool-assisted planning: before reasoning, build a lightweight graph over facts/rules and reorder or retrieve only needed premises (cf. Section 4 referencing methods like “Concise and Organized Perception”).
  - Prompt patterns: encourage planning or premise selection steps (“find relevant rules, then order them”) to reduce hallucinations and sequential overreach.
  - Evaluation: include reordered variants in benchmarks to measure robustness (R‑GSM-style stress tests).
- Research directions
  - Training objectives that penalize hallucinated intermediate facts and discourage overreliance on sequential proximity.
  - Architectures that support non-sequential reasoning (e.g., explicit memory, graph-based intermediates, or bidirectional planning with verification).
  - Systematic study of backward chaining: GPT‑4‑turbo’s relatively strong performance on reverse order suggests learnable strategies for goal‑driven inference (Figures 5–6).
  - Extending beyond propositional logic: test quantifiers, negation, and proofs that demand nonmonotonic reasoning or case splits.
  - Larger and multilingual reordered datasets to test generality.

> Key takeaway, echoed across figures and tables: arranging premises to align with the proof’s step sequence yields the highest accuracy, while deviations—especially with more steps or added distractors—cause large performance drops (Figures 3–6; Tables 6–8, 10–11). The observed error modes indicate that current LLMs often “read to reason” sequentially rather than reasoning over a set, explaining why premise order matters so much.
