# GENERALIZATION V.S. MEMORIZATION: TRACING LANGUAGE MODELS’ CAPABILITIES BACK TO PRETRAINING DATA

**ArXiv:** [2407.14985](https://arxiv.org/abs/2407.14985)

## 🎯 Pitch

This paper introduces a novel, scalable methodology—'task-gram language models'—to quantify how much large language models' (LLMs) predictions depend on memorization versus generalization, by explicitly linking their outputs to specific patterns in pretraining data. By correlating semantically matched n-gram input–output pairs from pretraining corpora with LLM predictions across diverse tasks, the work shows that factual question answering is driven primarily by memorization while tasks like translation and reasoning rely more on generalization. This insight not only clarifies the origins of key LLM capabilities but also provides practical tools to guide dataset curation, prompt design, and model deployment for different applications.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces a scalable way to quantify how much large language models (LLMs) rely on memorization versus generalization by tying model predictions back to specific statistics of their pretraining corpora. It builds a “task-gram language model” that counts semantically matched input–output n-gram pairs in the pretraining data and measures distributional alignment (via rank correlation) between those counts and the LLM’s predicted probabilities (Definitions 1–3; Figure 1). Applied to Pythia and OLMo models across translation, factual QA, world knowledge, and math reasoning, the analysis shows: factual QA is driven by memorization, while translation and reasoning depend mainly on generalization (Figures 3–4, 5; Table 1).

## 2. Context and Motivation
- Problem addressed
  - When an LLM performs a task, is it recalling pretraining content (“memorization”) or synthesizing new outputs (“generalization”)? Existing probes mostly check for verbatim copying (exact spans reappearing) or run counterfactual retraining (removing an example to see its effect), neither of which scales or reflects typical task outputs like short answers or reasoning steps (Introduction; Related Work).
- Why it matters
  - Practical: Understanding whether success comes from recall or skill transfer informs dataset curation, privacy risk, prompt design, and model selection for different applications (Section 7; Ethics).
  - Scientific: It clarifies which abilities (knowledge recall vs. reasoning) emerge from which parts of training data and how they evolve with scale (Figures 3–4).
- Where prior approaches fall short
  - Verbatim memorization is rare for many tasks and misses subtler training-data influences (Introduction).
  - Counterfactual memorization requires retraining, infeasible at LLM scale (Introduction).
  - Classic n-gram LMs and even ∞-gram models capture local context but miss long-range, task-specific dependencies across input and output (Section 2; discussion of Liu et al., 2024 in Section 2).
- Positioning
  - The paper defines distributional memorization/generalization at scale by correlating an LLM’s probability distribution with a task-specific distribution estimated directly from the pretraining corpus via a new “task-gram” construction (Definitions 1–3; Figure 1). It complements correlation-based analysis with a gradient influence estimate that traces training documents’ effects on test examples (Section 6; Figure 5).

## 3. Technical Approach
The core idea is to build a task-specific probabilistic model of the pretraining data and compare it to the LLM’s output probability distribution on test examples.

Step 1 — Build a task-gram table (Definition 1; Figure 1, left)
- What is an n-gram? A contiguous sequence of n tokens.
- What is a task-gram pair? An input n-gram `s_x` from a task example’s input `x` and an output n-gram `s_y` from its target `y` that are semantically aligned.
- How are pairs mined?
  - For each supervised example `(x, y)` from a task dataset `D_T`, enumerate candidate `n`-grams in `x` and `y`.
  - Compute embedding-based cosine similarity between each `(s_x, s_y)` and keep those above a threshold `γ_T`, excluding identical strings to avoid trivial matches (Definition 1).
  - Embeddings: LASER (for cross-lingual in MT) and E5 (for other tasks) are used for the semantic similarity step (Appendix D).
- Why document-level, long-range pairs? For many tasks, the input cue and output answer co-occur in the same document but far apart; counting such co-occurrences captures task-relevant long-range dependencies (Figure 1; Section 2).

Step 2 — Search the pretraining corpus for pair counts (Figure 1, middle)
- Use WIMBD to find document-level co-occurrence counts `C((s_x, s_y), D)` because pair co-occurrences are sparse and need exact counts; use ∞-gram for frequent single n-gram counts `C(s, D)` (Section 3, “Searching over Pretraining Data at Scale”).
- Decontamination check: ensure test sets are not copied into the pretraining corpus by verifying no large n-gram (n=8,14) overlap (Section 4).

Step 3 — Construct the task-gram language model (Definition 2)
- Define a conditional probability from pretraining data:
  - P_n,D(s_y | s_x) = C((s_x, s_y), D) / C(s_x, D)
  - Interpret: When `s_x` appears in a document, this is the empirical chance that `s_y` also appears somewhere in that document (Equation 1).

Step 4 — Compute LLM probabilities for the same n-gram outputs (Equation 2)
- Zero-shot inference with a minimal instruction `u` plus the input `x`.
- For each aligned output n-gram `s_y` that occurs in the gold target `y`, compute its probability under the LLM as the product of token probabilities given the growing context `u ⊕ x ⊕ y[1:m−1]`, where `m` is the start position of `s_y` in `y` (Equation 2).

Step 5 — Quantify distributional memorization (Definition 3)
- Collect all aligned pairs observed in the test set (Φ).
- Compute the Spearman rank correlation ρ between:
  - The log task-gram probabilities log P_n,D(Y | X) = {log P_n,D(s_y | s_x) for all pairs in Φ}, and
  - The LLM’s log probabilities log P_LLM(Y | X) = {log P_LLM(s_y | s_x) for the same pairs}.
- Interpretation:
  - High ρ: When the pretraining corpus makes `s_y` likely given `s_x`, the LLM also makes it likely—evidence of distributional memorization.
  - Low or insignificant ρ: The LLM’s predictions diverge from pretraining frequencies—evidence of distributional generalization.
- Statistical significance: Marked by p-value < 0.05 in plots (solid circles) vs. insignificant (gray stars) in Figure 4.

Alternative baseline — ∞-gram LM alignment (Equation 5)
- Also correlate the LLM with an ∞-gram LM that assigns token probabilities by backing off from the longest seen prefix in the corpus (Equation 5; Section 5). This is a “local-context” baseline.

Complementary causal signal — Training influence (Section 6; Figure 5)
- For additional evidence, estimate how much a pretraining document `d` influenced a test example `(x, y)` by summing, across checkpoints, the dot product between:
  - The gradient of the training loss of `d` at the position of `s_y`, and
  - The gradient of the test loss for `s_y` in `(x, y)`.
- Aggregate this over retrieved documents that either contain the full pair `(s_x, s_y)` or only the output `s_y` (R=50 per test example).
- Interpretation: Larger positive influence suggests the document contributed more to the model’s ability to predict `s_y` (Section 6).

Prompt optimization based on similarity to pretraining (Section 7; Table 1)
- Hypothesis: If a task benefits from memorization, make the prompt “look” more like pretraining text (higher n-gram counts); if it benefits from generalization, do the opposite.
- Implementation: Iteratively rewrite prompts using GPT4o, using WIMBD n-gram counts as a reward to maximize (memorization) or minimize (generalization), then evaluate zero-shot accuracy (Section 7; Appendix E; Table 1).

## 4. Key Insights and Innovations
- New notion: distributional memorization vs. generalization
  - Innovation: Memorization is defined as rank correlation between an LLM’s probabilities and a task-specific distribution estimated from the pretraining corpus (Definition 3). This captures non-verbatim, task-relevant dependence.
  - Why it matters: It scales (no retraining), applies to short outputs, and distinguishes knowledge recall from reasoning behavior (Sections 2, 5).
- Task-gram language model for long-range, cross-text dependencies
  - Innovation: Counts document-level co-occurrence of semantically matched input–output n-grams, not just local token contexts (Definition 2; Figure 1).
  - Significance: Outperforms ∞-gram LM at explaining LLM behavior on knowledge tasks (higher correlations in Figure 4), indicating that task-specific long-range structure in pretraining data better accounts for what LLMs “memorize.”
- Capability-level mapping across tasks
  - Finding: TriviaQA (factual QA) shows strong memorization that increases with model size; WMT (translation) and GSM8K (math reasoning) show insignificant memorization and stronger generalization; MMLU splits: knowledge-heavy subsets show moderate memorization that decreases with size, reasoning-heavy subsets show negligible memorization (Figure 4).
  - Impact: Clarifies that task type and difficulty mediate how scaling translates into performance—simple/knowledge tasks gain via better recall, hard/reasoning tasks via better generalization (Sections 4–5).
- Gradient-based training influence validates pair-based retrieval
  - Innovation: A checkpoint-traced gradient dot-product influence confirms that documents containing full n-gram pairs matter more than those with only the output phrase, and that influence is largest for TriviaQA, smallest for WMT (Figure 5).
  - Significance: Moves beyond correlation to provide causal-leaning evidence consistent with the distributional analysis (Section 6).
- Practical prompt optimization knob
  - Contribution: A simple optimization loop that nudges prompts toward or away from the pretraining distribution improves task performance in the expected direction: more “memorization-like” prompts help TriviaQA; more “generalization-like” prompts help GSM8K (Table 1; Section 7).

## 5. Experimental Analysis
Evaluation setup
- Models and pretraining corpora
  - Pythia family (13M–12B) trained on The Pile (≈207B tokens) (Section 3).
  - Additional GSM8K runs with OLMo (1B, 7B) trained on Dolma (≈3T tokens) (Section 3).
- Tasks and test sets
  - WMT-09 machine translation: six European languages; 2.5K test examples (Section 3, “Downstream Tasks”).
  - TriviaQA factual QA: 10K test examples; ground-truth answers are short strings (Section 3).
  - MMLU world knowledge and reasoning: 57 multiple-choice subjects; split into knowledge-intensive vs. reasoning-intensive (Appendix D lists subjects).
  - GSM8K math reasoning: grade-school word problems with chain-of-thought (CoT) solutions (Section 3).
- Metrics
  - WMT: BLEU between greedy translations and references (Section 4).
  - TriviaQA: exact-match accuracy (Section 4).
  - MMLU: accuracy by selecting the option with highest LM probability (chance = 25%) (Section 4).
  - GSM8K: due to low Pythia accuracy (<5%), CoT similarity via BERTScore precision (Section 4).

Main quantitative findings
- Performance vs. pair count (proxy for task-relevant data availability)
  - Figure 3 shows that for sufficiently large models (>410M), performance generally increases with the number of task-gram pair counts found in the pretraining corpus across WMT, TriviaQA, MMLU, and (for GSM8K) CoT BERTScore.
  - Caveat highlighted in Section 4: this trend could reflect either memorization or better generalization from richer, relevant pretraining data—necessitating the correlation analysis in Section 5.
- Distributional memorization/generalization (Figure 4)
  - WMT (top-left panel): No statistically significant Spearman correlation between task-gram probabilities and LLM probabilities; instead, larger models generate more novel n-gram pairs (left bar plot), indicating increasing distributional generalization.
  - TriviaQA (top-middle panel):
    - Strong and significant memorization: “Mem_n=3 > 0.35” while “Mem_n=5 < 0.25” and “Mem_∞ < 0.25,” with memorization increasing as model size grows (text immediately under Figure 4).
    - Interpretation: LLMs rely on memorizing small, long-range input–answer associations found in pretraining, and they do so more as they scale.
  - MMLU (right panels):
    - Split by type:
      - Knowledge-intensive tasks: most pronounced at n=3; memorization decreases with model size (top-right panel and text “Mem_n(LLM,D|T) decreases as the LLM size increases”).
      - Reasoning-intensive tasks: minimal memorization overall; when significant, larger n-grams (n=5) matter more; memorization decreases with model size (bottom-right panel).
    - Interpretation: Specialized, rarer knowledge requires the model to adjust away from raw pretraining frequencies (decreasing memorization), while reasoning depends little on memorization.
  - GSM8K (bottom-left panel):
    - No significant memorization across Pythia or OLMo.
    - Distributional generalization increases with model size, quantified via normalized Kendall tau ranking distance (fraction of pairwise rank disagreements), and task-gram LM explains LLM probabilities better than ∞-gram (dashed vs. solid lines).
- Gradient-based training influence (Figure 5; Section 6)
  - For documents retrieved from pretraining:
    - Those containing full n-gram pairs (green) exert consistently higher average influence than those with only the output phrase (blue), across model sizes and tasks.
    - Influence magnitude ranking: TriviaQA highest (and slightly increasing with size), MMLU moderate, WMT lowest. This mirrors Figure 4’s memorization profile.
- Prompt optimization results (Table 1; Section 7)
  - TriviaQA:
    - For Pythia-12B: memorization-encouraging prompt yields 28.7% accuracy vs. 23.2% with generalization-encouraging prompt.
    - For OLMo-7B: 36.4% vs. 29.8%.
  - GSM8K:
    - Improvements are small but consistent when encouraging generalization. For OLMo-7B: 7.9% vs. 6.3%. For Pythia-Instruct-6.9B: 7.3% vs. 6.3%.
  - These shifts align with the per-task memorization/generalization profiles in Figure 4.

Ablations and checks
- Cosine similarity threshold sensitivity for mining pairs: Varying γ among {0.70, 0.75, 0.80} for TriviaQA leaves the memorization trends unchanged (Appendix D; Figure 6).
- Decontamination: No large (n=8,14) n-gram overlaps between The Pile and evaluated test sets (Section 4).

Assessment
- The experiments are consistent and multi-pronged: (i) performance vs. counts (Figure 3), (ii) distributional correlations vs. ∞-gram baseline (Figure 4), (iii) gradient influence (Figure 5), and (iv) prompt manipulation (Table 1).
- Together they convincingly support the central claims: knowledge-intensive tasks hinge on memorization, while translation and reasoning hinge on generalization.

## 6. Limitations and Trade-offs
- Dependence on supervised task data to build task-gram tables
  - Requires labeled input–output examples to mine pairs. For MMLU and GSM8K they mine from test sets due to limited or absent training splits (Appendix D), which is acceptable for analysis but would not be suitable for training-time interventions.
- Pair mining quality and thresholds
  - Embedding similarity can admit spurious matches or miss paraphrases; performance depends on threshold `γ_T` and chosen embedding model (LASER/E5) (Appendix D). Figure 6 shows robustness for TriviaQA, but broader sensitivity analyses would be useful.
- Document-level co-occurrence as a proxy for dependency
  - Counting co-occurrence in the same document does not prove a functional relation; unrelated mentions in long documents can inflate counts. The gradient influence analysis (Section 6) partially addresses this but is approximate.
- Corpus and tooling scope
  - Analyses are on The Pile (≈207B tokens) with Pythia, and Dolma/OLMo only for GSM8K. The paper notes current tooling (WIMBD/∞-gram) limits scaling to even larger corpora like full Dolma by default (Appendix A).
- Causality and compute constraints
  - Influence tracing uses a limited number of retrieved documents (R=50) and available checkpoints (Section 6), so results are indicative rather than exhaustive.
- Metrics for GSM8K
  - Because Pythia’s GSM8K accuracy is very low, the paper reports BERTScore on CoT (Section 4). This measures surface similarity to reference solutions, not correctness; it’s a proxy for distributional tendencies.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a scalable, task-aware attribution lens for LLM behavior—moving beyond verbatim checks to distributional alignment grounded in pretraining data (Definitions 1–3; Figures 1, 4). This reframes capability analysis as a comparison of probability distributions, not just text overlaps.
- Practical applications
  - Prompt design: Encourage or discourage memorization by shifting prompt n-gram statistics toward or away from pretraining distributions (Section 7; Table 1).
  - Data curation: Identify which subsets of pretraining data likely drive knowledge vs. reasoning capabilities, guiding targeted augmentation or filtering.
  - Privacy and safety: Since memorization is strongest for knowledge-intensive tasks (Figure 4, TriviaQA), auditing and red-teaming should prioritize these regimes for potential leakage.
  - Evaluation: Use task-gram alignment as an auditing tool to detect when test performance is driven by corpus frequency vs. genuine generalization.
- Research directions
  - Better pair mining: Replace static embedding thresholds with learned alignment models, multilingual paraphrase mining, or causal discovery to reduce spurious pairs (Appendix A).
  - Beyond document-level counts: Incorporate structural cues (e.g., cross-sentence dependency parsing, layout, citations) to capture richer relations.
  - Larger corpora and newer models: Extend WIMBD-like search to multi-trillion-token corpora (Dolma) and modern models; compare instruction-tuned vs. base models systematically.
  - Finer-grained influence: Integrate the gradient influence estimator with retrieval to attribute predictions to specific documents/snippets in real time.
  - Task generalization map: Build broader taxonomies linking task characteristics (rarity, compositionality) to memorization/generalization profiles; use them to predict scaling behavior a priori.

> In short: Figure 4 and Figure 5 together show that factual QA aligns tightly with pretraining distributions and training influence, while translation and math reasoning do not—indicating that LLMs recall for simple knowledge tasks but must generalize for harder, reasoning-centric ones. The task-gram LM (Definitions 1–2; Figure 1) is the key mechanism that makes this analysis feasible and more explanatory than ∞-gram baselines.
