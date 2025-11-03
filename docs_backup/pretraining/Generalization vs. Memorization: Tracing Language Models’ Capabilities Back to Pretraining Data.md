# Generalization vs. Memorization: Tracing Language Models’ Capabilities Back to Pretraining Data

**ArXiv:** [2407.14985](https://arxiv.org/abs/2407.14985)
**Authors:** Xinyi Wang, Antonis Antoniades, Yanai Elazar, Alfonso Amayuelas, Alon Albalak, Kexun Zhang, William Yang Wang
**Institutions:** 

## 🎯 Pitch

This paper introduces a novel method to trace the influence of pretraining data on large language model outputs through 'distributional memorization,' quantifying how models balance memorization versus generalization across tasks. By leveraging task-specific n-gram models, it provides insights into LLM behavior, revealing crucial differences in handling factual QA versus tasks like translation and math reasoning, which has significant implications for model evaluation, prompting, and data privacy management.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces a scalable way to trace what large language models (LLMs) “use” from their pretraining corpus when solving downstream tasks. It defines distributional memorization and builds a task-specific n-gram language model (“task-gram LM”) to quantify how closely an LLM’s output probabilities align with pretraining data statistics, revealing that factual QA leans heavily on memorization while translation and math reasoning rely more on generalization (Figures 3–4, Definitions 1–3).

## 2. Context and Motivation
- Problem addressed
  - Whether LLMs’ impressive capabilities stem from genuine generalization or largely from memorizing pretraining data remains unsettled (Abstract; Introduction §1). Existing work often measures memorization as verbatim copying (exact string regurgitation), which is rare for short outputs like answers in QA or the final number in math reasoning.
  - The community lacks scalable methods to connect model behavior on real tasks to specific pretraining data patterns without retraining models.

- Why this matters
  - Practical: Understanding when to encourage memorization (e.g., knowledge retrieval) vs. when to reduce it (e.g., reasoning) can improve prompting, evaluation, and data curation, and inform privacy-risk mitigation.
  - Scientific: It clarifies how model capabilities arise from data distributions and model scaling, enabling principled interpretations beyond anecdotal “it memorized” or “it generalized.”

- Prior approaches and their gaps
  - Verbatim recall audits detect near-duplicate output (e.g., long exact sequences), which misses non-exact but distribution-driven behavior (Introduction §1).
  - Counterfactual memorization (ablate a training example and re-train) yields causal signals but does not scale to LLMs due to retraining expense (Introduction §1).
  - ∞-gram (Infini-gram) LMs approximate pretraining distributions with unbounded n-grams but primarily capture local context (Introduction §1; §5).

- Positioning
  - The paper proposes “distributional memorization”: a correlation-based measure aligning LLM output probabilities with task-relevant pretraining frequencies (Definitions 2–3). It introduces a “task-gram LM” that counts semantically matched input–output n-gram pairs (long-range, task-relevant co-occurrences in the same document), enabling scalable, task-aware analysis across the full pretraining corpus (Figure 1; §2).

## 3. Technical Approach
The pipeline has four main steps (Figure 1; §2–§3):

1) Build a task-gram table from supervised task data
- Goal: capture task-specific supervision as input–output n-gram pairs.
- What is an n-gram? A contiguous sequence of n tokens.
- How pairs are formed:
  - For each task example `(x, y)` (input, ground-truth output), enumerate all input n-grams `G_n(x)` and output n-grams `G_n(y)` and embed both sides with a sentence embedding model.
  - Keep pairs `(s_x, s_y)` whose cosine similarity exceeds a threshold `γ_T`, with `s_x ≠ s_y`.
  - Formal definition (Definition 1): the task-gram table `H_n(T)` is all such pairs filtered by semantic similarity greater than `γ_T`.
- Why embed-based matching? It aligns semantically related n-grams that may not be exact matches—critical for tasks like translation or paraphrastic QA (§2, Definition 1; Appendix D gives thresholds and a sensitivity check in Figure 6).

2) Search the pretraining corpus for task-gram evidence
- Tools:
  - `WIMBD`: a large-scale n-gram search engine used here to count low-frequency co-occurrence of `(s_x, s_y)` within the same document (not necessarily adjacent) (§3 “Searching over Pretraining Data at Scale”).
  - `∞-gram` API: used to count frequent single n-grams and to obtain ∞-gram probabilities for tokens (§3; §5).
- Counts computed:
  - `C((s_x, s_y), D)`: number of documents in the pretraining corpus `D` (Pile) where both n-grams appear (co-occur) in the same document.
  - `C(s_x, D)` and `C(s_y, D)`: occurrence counts for the individual n-grams (§2).

3) Construct a task-gram language model (task-gram LM)
- Conditional probability (Definition 2; Eq. (1)):
  - `P_n,D(s_y | s_x) = C((s_x, s_y), D) / C(s_x, D)`.
  - Interpretation: given the input-side n-gram `s_x` appears in a document, how often does the output-side n-gram `s_y` appear in the same document?
- Why this helps: Unlike classical n-gram LMs focused on adjacent tokens, this models long-range, task-relevant dependencies (e.g., cross-lingual phrase pairs or input-question to answer-term co-occurrences) (§2).

4) Compare the task-gram LM with the LLM’s predictive distribution
- Compute the LLM probability of the gold output n-gram:
  - For a prompt with instruction `u`, test input `x`, and true output `y`, define `PLLM(s_y | s_x)` as the product of token probabilities of `s_y` conditioned on the full context `u ⊕ x ⊕ y[1:m−1]` (Eq. (2)), where `m` is the start position of `s_y` in `y` (§2).
- Define distributional memorization (Definition 3; Eq. (3)):
  - Collect all task-gram pairs found in the test set Φ, compute two vectors: `log P_n,D(Y|X)` and `log P_LLM(Y|X)`, and take their Spearman rank correlation `ρ`.
  - Spearman correlation measures whether two quantities increase together in a monotonic way, without assuming linearity.
  - High positive `ρ` means the LLM assigns higher probabilities to output n-grams that are more frequent (in a task-conditioned sense) in pretraining data—i.e., stronger distributional memorization.
- Alternative baseline: ∞-gram memorization (Eq. (5); §5)
  - Compute `P_∞,D(s_y | u ⊕ x ⊕ y)` as the product over tokens using the longest n-gram prefix found in the corpus (with back-off), then correlate these log-probabilities with `log P_LLM`.
  - Purpose: compare task-aware, long-range modeling (task-gram LM) vs. generic local-context modeling (∞-gram LM).

Additional analyses

- Performance vs. task-gram coverage (Eq. (4); Figure 3; §4)
  - Estimate how likely a test pair `(x, y)` is “represented” in pretraining by summing counts of all matched task-gram pairs from `(x, y)` and examine how task performance varies with this coverage.
- Training influence estimate (gradient tracing; §6)
  - For a document `d` in pretraining that contains `(s_x, s_y)`, define training loss `ℓ(θ, d, s_y)` where `s_y` occurs in `d`; define test loss `ℓ(θ, (x,y), s_y)` for the test example.
  - At each stored training checkpoint `θ_i`, compute the dot product of loss gradients `∇θ_i ℓ(θ_i, d, s_y) · ∇θ_i ℓ(θ_i, (x,y), s_y)`; sum over checkpoints and relevant pairs (formula in §6).
  - Average this influence over the test set and over R=50 retrieved documents per test example for two retrieval schemes: documents containing the pair `(s_x, s_y)` vs. documents containing only `s_y` (Figure 5).
- Prompt optimization to encourage memorization vs. generalization (§7; Table 1; Appendix E)
  - Use GPT-4o to rewrite prompts, with a “reward” equal to the average n-gram count in the pretraining corpus (as measured via WIMBD). Two directions:
    - Maximize reward to encourage memorization (make the prompt “look like” pretraining).
    - Minimize reward to encourage generalization (make the prompt less like pretraining).
  - Evaluate zero-shot accuracy changes (Table 1).

Design choices, and why
- Use document-level co-occurrence counts for task-gram pairs to capture long-range dependencies (Figure 1; §2), because local token windows miss input–output relationships such as source/target phrases spread apart in a document or question vs. answer references.
- Prefer WIMBD for pair co-occurrence (more accurate at low frequency) and ∞-gram for single n-gram counts (fast and accurate for frequent events) (§3).
- Spearman correlation on log-probabilities provides a robust, scale-insensitive, monotonic association measure for “memorization” (§2, Definition 3).

## 4. Key Insights and Innovations
- A scalable definition of distributional memorization anchored in task semantics
  - Novelty: Defines memorization as a rank correlation between LLM probabilities and a task-aware pretraining distribution, not as verbatim copying (Definitions 2–3).
  - Significance: Works at LLM scale (no retraining), and captures non-exact yet semantically aligned reuse from pretraining (§2).

- Task-gram language model: document-level, long-range, task-conditioned counts
  - Novelty: Counts co-occurrence of semantically matched input–output n-gram pairs within documents (Definition 2; Figure 1), unlike classical or ∞-gram LMs that model local lexical dependencies.
  - Significance: Better explains LLM behavior on complex tasks where input cues map to output concepts across long spans (§5; Figure 4 shows task-gram correlations exceed ∞-gram).

- Clear, task-dependent mapping of memorization vs. generalization
  - Insight: Factual QA (TriviaQA) shows strong, scaling memorization; translation (WMT) and reasoning (GSM8K; reasoning-heavy MMLU subsets) show weak or decreasing memorization and stronger generalization (Figure 4).
  - Significance: Provides a principled diagnosis that aligns with task nature—retrieval vs. reasoning.

- Gradient-based training influence that complements correlation
  - Novelty: Traces test-time behavior back to training-time documents using gradient dot products across checkpoints (formula in §6), without retraining.
  - Significance: Corroborates that documents containing task-gram pairs affect test predictions more than documents sharing only output n-grams (Figure 5).

- Prompt optimization guided by distributional statistics
  - Novelty: Uses pretraining n-gram counts as a feedback signal to make prompts more or less “pretraining-like” (Table 1; Appendix E).
  - Significance: Demonstrates actionable prompting strategies: encourage memorization improves QA, encourage generalization improves math reasoning (§7).

## 5. Experimental Analysis
- Setup
  - Models and data
    - `Pythia` family (13M–12B), trained on the `Pile` (≈207B tokens) (§3).
    - Some results with `OLMo` (1B, 7B), trained on `Dolma` (3T tokens) (Figure 4; §3).
  - Tasks
    - Translation: WMT09, 2.5K tests across 6 European languages (Hungarian, Czech, German, Italian, Spanish, French) (§3).
    - Factual QA: TriviaQA, 10K test set; treat whole answer as the output n-gram (§3).
    - World knowledge/Reasoning: MMLU (57 tasks) split into knowledge-intensive vs. reasoning-intensive subsets (Appendix D lists categories).
    - Math reasoning: GSM8K; evaluate chain-of-thought similarity (BERTScore precision) due to low exact-answer accuracy for Pythia (<5%) (§4).
  - Metrics
    - WMT: BLEU (greedy decoding).
    - TriviaQA: exact-match accuracy.
    - MMLU: multiple-choice accuracy (random baseline 25%).
    - GSM8K: BERTScore precision on chain-of-thought (CoT).
  - Decontamination: verify no large n-gram (n=8, 14) overlaps between Pile and test sets (Section 4).
  - Task-gram filtering thresholds: e.g., WMT uses cosine similarity 0.85/0.8/0.75/0.7 for n=2–5; MMLU/TriviaQA use 0.75 (n=3), 0.65 (n=5). Sensitivity analysis shows trends stable across thresholds (Appendix D, Figure 6).

- Main results

  1) More task-gram coverage ↔ better performance (Figure 3; §4)
     - Quote:
       > “In general, all task performance increases when the number of task-related n-gram pairs increases when the model size is large enough (> 410M).”
     - Exceptions/nuances:
       - Small models (<410M) show near-zero performance for WMT/TriviaQA and sub-random MMLU performance around certain coverage bins, especially on reasoning-heavy items (§4).
       - GSM8K trend is noisy for large models; 2.8B shows strongest increase in CoT BERTScore with more pairs (§4).

  2) Distributional memorization vs. generalization (Figure 4; §5)
     - WMT (translation):
       - No significant memorization: Spearman correlations of task-gram/∞-gram probabilities with LLM probabilities are statistically insignificant across sizes (Figure 4, top-left).
       - Generalization evidence: larger models generate more novel n-gram pairs unseen in the pretraining corpus (Figure 4, top-left panel), indicating distributional divergence from pretraining text.
     - TriviaQA (factual QA):
       - Strong memorization: 
         > “Mem_n=3(LLM, D|T) (> 0.35) is significantly more profound than Mem_n=5(<0.25) and Mem_∞(<0.25)” (Figure 4, top-middle).
       - Scaling: memorization strengthens as model size increases; aligns with improved TriviaQA accuracy (Figure 3).
       - Interpretation: small, long-range input–output associations (n=3) drive factual recall.
     - MMLU:
       - Knowledge-intensive subset: 
         > Task-gram memorization is most pronounced for n=3 (Mem_n=3 > 0.25), but decreases as model size grows (Figure 4, top-right).
         - Interpretation: specialized/rare knowledge may require adjusting recall probabilities as models scale, reducing correlation with raw pretraining frequencies (§5).
       - Reasoning-intensive subset:
         > Significant memorization appears at larger n (n=5), but also decreases with scale (Figure 4, bottom-right).
         - Interpretation: larger text chunks align with reasoning concepts, but scaling shifts away from memorized patterns toward generalization (§5).
     - GSM8K (math reasoning):
       - No significant memorization with either Pythia or OLMo (Figure 4, bottom panels).
       - Generalization quantified via normalized Kendall tau ranking distance (higher distance = more disagreement with pretraining-based rankings):
         > Generalization increases with model size; LLM probabilities agree more with task-gram probabilities than ∞-gram probabilities (Figure 4, bottom).
       - Interpretation: reasoning relies on constructing novel reasoning patterns rather than reusing pretraining distributions (§5).

  3) Task-gram LM vs. ∞-gram LM (Figure 4; §5)
     - Across TriviaQA and MMLU, `Mem_∞(LLM, D|T)` is consistently lower or equal to `Mem_n(LLM, D|T)`.
     - Takeaway: task-gram LM (long-range, task-aware) better explains LLM predictive behavior than local ∞-gram statistics.

  4) Gradient-based training influence (Figure 5; §6)
     - Documents containing full pairs `(s_x, s_y)` exert more influence on test losses than documents containing only `s_y` (blue vs. green curves). This holds across WMT, TriviaQA, MMLU.
     - Scaling trends:
       > Influence decreases with model size for WMT and MMLU, slightly increases for TriviaQA, and is highest for TriviaQA at the largest size (Figure 5).
     - Interpretation: aligns with the correlation results—factual QA derives more from pretraining pairwise associations as models scale.

  5) Prompt optimization guided by n-gram counts (Table 1; §7; Appendix E)
     - TriviaQA: Memorization-encouraging prompts improve zero-shot accuracy (e.g., `Pythia-12B`: 28.7% vs. 23.2%; `OLMo-7B`: 36.4% vs. 29.8%).
     - GSM8K: Generalization-encouraging prompts improve accuracy (e.g., `OLMo-7B-instruct`: 7.9% vs. 6.3%; `Pythia-6.9B-instruct`: 7.3% vs. 6.3%).
     - The optimized prompts differ in phrasing but not length; see examples in Appendix E (Table 2).

- Do the experiments support the claims?
  - Yes, for correlation claims: Figures 3–4 demonstrate systematic, task-dependent relationships between pretraining distributions and LLM predictions that vary with model size.
  - Causality is supported indirectly: the gradient-based influence analysis (Figure 5) points to training-time influence of documents containing task-gram pairs on test-time predictions.
  - Robustness checks:
    - Decontamination against long exact overlaps (§4).
    - Similarity-threshold sensitivity (Appendix D, Figure 6) shows trends persist under reasonable filtering changes.

- Caveats
  - Some results are significance-based (e.g., “not statistically significant” for WMT correlations), and precise ρ values beyond thresholds are summarized qualitatively in text rather than tabulated.
  - GSM8K relies on BERTScore of CoTs due to low exact accuracy; still, generalization patterns align with expectations for reasoning tasks (§4, Figure 4).

## 6. Limitations and Trade-offs
- Data and model coverage
  - Primary analysis is on `Pythia` (Pile, 207B tokens), which is not state-of-the-art; WIMBD currently limits full-scale investigation on larger corpora like `Dolma` (Appendix A).
  - Some GSM8K/OLMo results are included, but comprehensive cross-corpus generality remains to be fully demonstrated.

- Task-gram construction sensitivity
  - The quality of `H_n(T)` depends on the embedding model and cosine-threshold `γ_T` (Appendix D). Although Figure 6 shows stability across thresholds 0.70–0.80, different embedding backbones or multilingual settings could shift matches.
  - Co-occurrence within a document can include spurious matches (same document, unrelated context). The embedding filter mitigates but does not eliminate this risk.

- Use of test sets to mine pairs for some tasks
  - For MMLU and GSM8K, limited training data led to mining pairs directly from test sets (Appendix D). This is acceptable for analysis (no model retraining uses these pairs), but it can bias which n-grams are analyzed and could inflate perceived alignment between task-grams and test outputs.

- Definition of generalization
  - Generalization is operationalized as the “opposite” of memorization (lack of positive correlation), plus a ranking-distance diagnostic for GSM8K (Figure 4). A more direct or orthogonal operationalization (e.g., explicit novelty or out-of-support generation statistics) could further sharpen conclusions.

- Training influence estimation constraints
  - Influence uses gradient dot products over a limited number of stored checkpoints and only `R=50` retrieved documents per test example (§6). This provides directional evidence rather than a complete causal audit.

- Computation and retrieval trade-offs
  - Accurate pair co-occurrence requires WIMBD (more expensive) while frequent single n-grams use ∞-gram (faster). Extremely low-frequency pairs might still be undercounted or missed.

## 7. Implications and Future Directions
- What this changes
  - Provides a practical, scalable framework to quantify how much downstream performance comes from “matching pretraining distributions” vs. “departing from them,” in a task-aware way (Definitions 2–3; Figures 3–5).
  - Reframes memorization as a distributional alignment signal tied to task semantics—not just verbatim copying—opening more nuanced evaluations of model behavior.

- Practical applications
  - Prompting: Choose prompt styles based on task type—encourage memorization (more pretraining-like phrasing) for factual QA; encourage generalization (less pretraining-like phrasing) for reasoning (Table 1; Appendix E).
  - Data curation: For knowledge-heavy tasks, curate pretraining corpora to include more documents with strong `(input n-gram, output n-gram)` co-occurrence. For reasoning, diversify contexts and reduce templatic regularities that might anchor the model to shallow patterns.
  - Model audits and safety: Use task-gram analysis to identify where models are likely relying on memorized associations, informing privacy-risk assessments and mitigation strategies.

- Research directions
  - Richer task-grams: Move beyond n-gram pairs to multi-hop or structured relations (e.g., graph-based “question→fact→derivation” chains) to analyze reasoning pathways.
  - Better pair mining: Improve semantic filtering with cross-encoder rerankers, multilingual alignment, or task-specific encoders; evaluate more granular distance measures beyond cosine.
  - Larger corpora and models: Apply the method to state-of-the-art open models and corpora (e.g., full Dolma; multilingual settings), and compare across training recipes.
  - Stronger causal tools: Combine influence functions with data ablations targeted at high-influence documents; use efficient fine-tuning/”unlearning” methods to simulate removals without full retraining.
  - Generalization metrics: Develop formal, distribution-aware generalization measures that are orthogonal to memorization (e.g., novelty under the task-gram LM; support size; divergence metrics).

In summary, the paper offers a principled and scalable toolkit—task-gram LMs plus correlation and influence analyses—that reveals a consistent, task-dependent pattern: factual QA is driven by distributional memorization of pretraining data, while translation and reasoning benefit more from distributional generalization (Figures 3–5; Definitions 2–3; Eq. (5)), and this distinction can be operationalized to improve prompting and data strategy (Table 1; §7).
