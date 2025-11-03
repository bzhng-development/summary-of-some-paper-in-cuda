# Infini-gram: Scaling Unbounded n‑gram Language Models to a Trillion Tokens

**ArXiv:** [2401.17377](https://arxiv.org/abs/2401.17377)
**Authors:** Jiacheng Liu, Sewon Min, Luke Zettlemoyer, Yejin Choi, Hannaneh Hajishirzi
**Institutions:** University of Washington, Allen Institute for AI

## 🎯 Pitch

The paper presents `∞-gram`, an innovative unbounded n-gram language model, and `infini-gram`, a powerful engine capable of handling queries over 5 trillion tokens with millisecond latency. This breakthrough offers a nonparametric alternative to neural LLMs, enhancing interpretability, data attribution, and supporting deeper analyses of text, while achieving up to a 73% reduction in perplexity, making it invaluable for large-scale language modeling applications.

---

## 1. Executive Summary
This paper introduces `∞-gram`, an unbounded n‑gram language model, and `infini-gram`, a suffix‑array–based engine that serves it at web scale. By indexing 5 trillion tokens and answering count/probability queries with millisecond latency, the work turns classic n‑gram modeling into a practical nonparametric complement to neural LLMs, cutting perplexity by up to 73% in some settings and enabling new analyses of human and machine text.

## 2. Context and Motivation
- Problem addressed
  - Can classic n‑gram language models still help at the trillion‑token scale of modern LLMs, and if so, how can we make them practical and useful? The paper targets two historical bottlenecks (§1, §2, §3):
    1) n‑gram models are usually limited to small `n` (e.g., 5), which discards long context.
    2) Precomputing large n‑gram count tables becomes infeasible as `n` and data scale grow.
- Why this matters
  - Practical: A count‑based, data‑centric LM is interpretable, auditable, and directly tied to source documents—useful for attribution, decontamination, and copyright‑sensitive settings (§E; §5, Table 2 for SILO).
  - Scientific: It provides a new lens on how much of next‑token prediction is “seen‑before text” vs. neural generalization (§4, Figures 3–5).
- Prior approaches and shortcomings
  - Large n‑gram tables reached 2T tokens but still capped at 5‑grams (Brants et al., 2007). Suffix tree/array approaches existed, but not at trillion‑token scale and often without valid probability distributions (§6, §F).
  - Interpolating small n‑grams with neural LMs has had mixed results and used far less data (e.g., Khandelwal et al., 2020; Li et al., 2022) (§6).
  - Retrieval‑based LMs (e.g., kNN‑LM, RETRO) store vectors per token/chunk, leading to massive storage/compute; scaling to trillions of tokens is challenging (Table 6, §F).
- Positioning
  - The paper reframes n‑grams as a nonparametric LM with unbounded context (`∞-gram`) and provides `infini-gram`, a storage/serving layer that makes counting and probability queries fast on trillions of tokens. It modernizes n‑grams in both data scale (5T tokens) and context length (unbounded) and shows strong complementary gains with large neural LMs (§1, §2, §3, §5).

## 3. Technical Approach
This section explains what `∞-gram` is, how `infini-gram` works, and how the two are combined with neural LMs.

- 3.1 The `∞-gram` language model (§2)
  - Core idea
    - For each prediction step, use the longest suffix of the prefix that appears at least once in the corpus. Define `effective n` as one plus the length of that suffix.
  - Probability definition
    - Plainly: look at all times the chosen context occurred in the corpus; among those, count how often each next token followed; the ratio is the probability of that next token.
    - Notation (from §2): Let `w1:i-1` be the prefix and `n = max{n' | cnt(wi-(n'-1):i-1 | D) > 0}`. Then
      `P∞(wi | w1:i-1) = cnt(wi-(n-1):i-1 wi | D) / cnt(wi-(n-1):i-1 | D)`.
  - How it differs from standard backoff
    - Traditional backoff reduces `n` until the numerator (joint count) is nonzero, then uses discounting (e.g., Katz). `∞-gram` instead reduces `n` until the denominator (context count) is nonzero and does not require discounting, because the effective `n` depends only on the context, not on the candidate token, making `P∞(·|context)` a proper distribution (§2).
  - Special notions
    - `effective n`: 1 + longest seen suffix length for the current prefix (§2).
    - `sparse estimate`: the `∞-gram` distribution places probability 1 on exactly one next token (and 0 on all others). Intuitively, the corpus always continues this context in one way (§2). The paper shows these cases are especially reliable (§4.1, Figure 3 right).
  - Why perplexity is not reported for pure `∞-gram`
    - Zero probabilities cause infinite perplexity; instead, the paper reports perplexity for an interpolation of `∞-gram` with a neural LM (§2, §5).

- 3.2 The `infini-gram` engine (§3; Appendix A)
  - What is a `suffix array` (definition, §3)
    - An array of pointers to all suffixes of a sequence, sorted lexicographically. It lets you find how many times a query string appears as a substring in `O(L + log N)` time (L = query length; N = corpus length).
  - How the index is built and stored
    - The input corpus is tokenized. A `token array` stores token IDs as bytes; documents are separated by a special 0xFFFF token (§3, Figure 2 right).
    - The `suffix array` stores for each suffix its starting byte offset into the token array. With 2 bytes per token ID and 5 bytes per pointer (for corpus sizes considered), total storage is about 7 bytes per token (§3).
    - Example scale: indexing 1.4T tokens took ~48 hours on one 128‑CPU, 1 TiB RAM node and ~10 TB disk (§3).
    - Indexes for Dolma (3T), RedPajama (1.4T), Pile (380B), and C4 (200B) are built; these are additive to reach 5T tokens (§3, §A.2).
  - How counting and probabilities are computed
    - For a query n‑gram, all its occurrences form a contiguous slice in the suffix array; binary searching for the slice boundaries gives the count. This underlies all `COUNT`, `NGRAMPROB`, and `INFGRAMPROB` queries (§3, §A.4).
    - For `∞-gram`, a “binary‑lifting + binary‑search” finds the longest suffix length with nonzero count in `O(log L)` counting calls (§A.4).
  - Latency and complexity
    - On‑disk, memory‑mapped access with prefetching, shard‑parallelism, and amortization yields sub‑second latency for all queries (§A.4–A.5).
    - Table 3 reports average latencies (single 8‑core CPU):
      > Counting an n‑gram: ~13–20 ms; `n` from 1 to 1000 has similar time (Pile‑train vs. RedPajama).  
      > `∞-gram` token probability: 90–135 ms; full `∞-gram` next‑token distribution: 88–180 ms (Table 3).
  - Extra capabilities
    - Document retrieval: given an n‑gram or a CNF expression (AND/OR over multiple n‑grams), retrieve all matching documents (`SEARCHDOC`; §A.5, Figure 16).
    - Index additivity/subtractivity enables composing/shrinking corpora without rebuilding (§A.2).

- 3.3 Combining with neural LMs (§2, §5)
  - Simple linear interpolation:
    `P(y | x) = λ P∞(y | x) + (1 − λ) Pneural(y | x)`.
  - Two hyperparameters are used in practice (§5): `λ1` for contexts with `sparse` `∞-gram` estimates (high confidence) and `λ2` otherwise. Values are tuned on validation to minimize perplexity.
  - For time‑shifted data, a Random Forest selects an instance‑wise interpolation weight using features such as suffix lengths and frequencies, further improving perplexity (§D.2, Table 5).

- 3.4 Datasets and decontamination (§4, §B)
  - To avoid trivial copy‑through, the training corpora for `∞-gram` are decontaminated against evaluation sets using the Big Friendly Filter: remove a document if ≥80% of its 13‑grams appear in evaluation (§4, §B).
  - Table 4 shows filtering stats; e.g., 0.6% of Pile‑train documents removed overall, with high removal in GitHub (5.3%).

- 3.5 A running example (Figure 1)
  - A 5‑gram LM fails to predict the next token in a snippet, while `∞-gram` finds the longest matching suffix (`n = 16` here) and predicts correctly by counting continuations in the corpus.

## 4. Key Insights and Innovations
- Unbounded n‑gram LM with a valid distribution (§2)
  - Innovation: `∞-gram` starts from arbitrarily long context and backs off until the context (denominator) is seen; unlike standard backoff, it yields a proper probability distribution without discounting.
  - Significance: It preserves as much context as the corpus can support on each instance. This is fundamentally different from fixed small‑`n` models that necessarily truncate context.
- Trillion‑token suffix‑array engine with millisecond latency (§3; §A.4–A.5)
  - Innovation: A compact (7 bytes/token) on‑disk index with memory‑mapped, sharded, hinted binary search and amortized computation.
  - Significance: Enables interactive n‑gram/`∞-gram` probability and document retrieval over 5T tokens. Prior suffix‑based language modeling did not reach this scale or latency (Table 6; §6, §F).
- Empirical finding: `∞-gram` is highly predictive where the corpus “commits” (§4.1)
  - Novel result: On decontaminated Pile validation data, `∞-gram` agrees with ground truth on 47% of tokens overall, and exceeds 75% agreement for tokens with `effective n ≥ 16` (Figure 3 middle). When the estimate is `sparse`, overall agreement rises to 75%, and to >80% for `effective n ≥ 14` (Figure 3 right).
  - Significance: Large‑context exact‑match statistics are powerful signals, different from neural probabilities (Figure 4).
- Complementarity with strong neural LMs (§5)
  - Novel result: Interpolating `∞-gram` with Llama‑2 70B reduces perplexity from 4.59 to 3.96 on Pile validation (−18%), and to 3.95 on test (−19%) when using Pile‑train + RedPajama as the reference data (Table 1).
  - Significance: This contradicts the conventional wisdom that n‑grams no longer help large LMs; the key appears to be both data scale and unbounded context.
- Diagnostic lens on decoding and positional effects (§4.2)
  - Novel observation: Greedy decoding exhibits strong, sometimes periodic, fluctuations in `∞-gram` agreement as `effective n` grows (e.g., Llama‑2 7B dips at n = 20, 24, 28, 32 with p < 10⁻⁹⁹), unlike nucleus sampling which resembles human text distributions (Figure 5). The paper hypothesizes links to positional embeddings.

## 5. Experimental Analysis
- Evaluation methodology
  - Human text: Pile validation/test. Token‑wise `∞-gram` agreement is measured by checking whether `P∞(true token | prefix) > 0.5` (a lower bound on argmax accuracy), binned by `effective n` (§4.1; Figure 3).
  - Machine text: Continue 50‑token prompts from Pile‑val across model sizes (Llama‑2 7B/13B/70B; GPT‑J 6B; GPT‑Neo 125M/1.3B/2.7B) and decoding schemes (greedy, temperature, nucleus). Analyze agreement vs. `effective n` (§4.2; Figure 5, Figure 9).
  - Perplexity: Only for the interpolated model (`neural + ∞-gram`) since `∞-gram` has zeros. Evaluate on Pile validation/test and time‑shifted Wikipedia (April–Aug 2023) (§5; §D.2).
  - Decontamination: Big Friendly Filter; statistics in Table 4 (§B).
  - Tokenizers: Separate `infini-gram` indexes for GPT‑2/Neo/J, Llama‑2, and SILO tokenizers (§5.1).
- Main quantitative results
  - Predictiveness of `∞-gram` on human text (§4.1)
    > Overall agreement 47%; >75% when `effective n ≥ 16` (Figure 3 middle).  
    > With `sparse` estimates: 75% overall; >80% for `effective n ≥ 14` (Figure 3 right).
    - 5‑gram baselines have much lower agreement because most tokens require context longer than 5 (Figure 3 left; median `effective n` is 7 and mean is 9.1).
    - Qualitatively, `∞-gram` excels at continuing multi‑token words, common phrases, and entity tails but struggles to recall the first token of names (Figure 3 discussion).
  - Complementarity with neural LMs (§4.1; Figure 4)
    > When Llama‑2 assigns very low probability, `∞-gram` still has >20% agreement, rising to ~50% on `sparse` cases (Figure 4).
  - Perplexity gains from interpolation (Pile val/test; Table 1)
    > Llama‑2 70B: 4.59 → 3.96 (−18%) on val and 4.65 → 3.95 (−19%) on test with Pile‑train+RedPajama.  
    > Llama‑2 13B: 5.30 → 4.41 (−21%) on val and 5.43 → 4.42 (−23%) on test, outperforming Llama‑2 70B baseline (§5.2, Table 1).  
    > GPT‑2 1.6B: 14.42 → 9.93 (−33%) on val; 14.61 → 9.93 (−34%) on test.  
    > GPT‑J 6.7B: 6.25 → 5.75 (−10%) on val; 6.51 → 5.85 (−12%) on test.
    - Gains are larger when the neural LM’s pretraining data differs more from the `∞-gram` reference (e.g., GPT‑2 vs. GPT‑Neo on Pile; §5.2).
  - SILO and comparisons to other retrieval methods (§5; Table 2)
    > On Enron Emails (domain out‑of‑distribution for SILO), SILO‑PD PPL drops 19.56 → 6.31 (−70%) on val and 20.62 → 4.85 (−73%) on test with `∞-gram`, outperforming kNN‑LM and RIC‑LM lines reported in Min et al. (2023a) (Table 2).  
    > On Wikipedia and NIH ExPorters, `∞-gram` consistently improves SILO variants, often more than kNN‑LM / RIC‑LM.
  - Time‑shifted Wikipedia (April–Aug 2023; Table 5)
    > Simple interpolation gives 0–6% relative gains; a Random Forest gating over suffix features raises this to 3–20% (Table 5).
  - Scaling and domain ablations (§D.3; Figure 10)
    > Gains grow roughly log‑linearly with reference data size; using only in‑domain slices performs similarly to using the full reference set, suggesting most benefit comes from in‑domain matches.
  - Machine text decoding analysis (§4.2; Figure 5)
    > Nucleus sampling most closely matches human `effective n` distribution and yields smoother agreement curves.  
    > Greedy decoding shows strong oscillations in agreement as `effective n` increases; smaller models and Llama‑2 7B show pronounced periodicity (p < 10⁻⁹⁹).
- Efficiency results
  - Query latency and complexity summarized in Table 3 and §A.4–A.5:
    > COUNT: ~13–20 ms; `NGRAMDIST (n=5)`: 31–39 ms; `INFGRAMPROB`: 90–135 ms; `INFGRAMDIST`: 88–180 ms.
  - Storage: ≈7 bytes/token; e.g., RedPajama’s 1.4T tokens indexed in ~2 days on a single 128‑core CPU node using ~10 TB disk (§3).
- Do the experiments support the claims?
  - Yes, convincingly for “complementarity” and “practicality”: sizable PPL reductions across model families (Table 1–2), strong on OOD domains (SILO Enron), and measurable millisecond‑level latency at trillion‑token scale (Table 3).
  - The human/machine agreement analyses (Figures 3–5) provide granular evidence that long‑suffix counts carry predictive power not captured by neural probabilities alone (Figure 4).
- Robustness and caveats
  - Careful decontamination reduces trivial copy effects (Table 4), and time‑shifted tests confirm continued value (Table 5).
  - The model’s own zero‑probability nature prevents direct perplexity reporting for `∞-gram`; interpretation relies on interpolation choices (§2, §5).
  - A noted failure mode: naive interpolation may harm open‑ended generation by steering the model into irrelevant continuations; gating/learning when to trust `∞-gram` is important (§5.2, “A note on text generation”).

## 6. Limitations and Trade-offs
- Dependence on exact surface forms
  - `∞-gram` only “knows” what appears verbatim in the reference data. It cannot generalize semantically beyond exact counts, so it struggles with novel paraphrases, rare entities’ first tokens, or reasoning beyond memorized strings (qualitative notes in §4.1).
- Zero probabilities
  - As a count‑based model, `∞-gram` assigns zero to unseen continuations, precluding standalone perplexity evaluation and risking harmful guidance in generation. Interpolation and gating are necessary (§2, §5.2).
- Data and storage requirements
  - While compact for its scale (7 bytes/token), a 5T‑token index is ≈35 TB. Building and serving such indexes require significant CPU time and fast SSDs, though still far below vector‑retrieval footprints at comparable scales (Table 6).
- Tokenization and vocabulary constraints
  - The storage layout assumes 2 bytes per token ID (|V| < 65,536). Other tokenizers must be separately indexed; cross‑tokenizer use is not supported (§5.1).
- Domain shift and coverage
  - Gains are largest with in‑domain or overlapping distributions (Figure 10). For strongly out‑of‑domain text with little surface overlap, `∞-gram` contributes less.
- Open questions
  - The cause of periodic agreement dips under greedy decoding (Figure 5) is hypothesized to involve positional embeddings, but this remains to be rigorously tested (§4.2).
  - Best practices for using `∞-gram` in generation (how to gate, when to trust) remain to be developed (§5.2).

## 7. Implications and Future Directions
- Shifting the role of data in language modeling
  - This work re‑establishes nonparametric, data‑centric modeling as a first‑class complement to large neural LMs, at the actual pretraining scale (5T tokens). The ability to query what the corpus literally contains—fast and precisely—has far‑reaching implications for attribution, compliance, and debugging (§E).
- Practical applications enabled by `infini-gram` (§E; Appendix G)
  - Retrieval and attribution at pretraining scale (e.g., “which documents contain this phrase?” via `SEARCHDOC`).
  - Data curation and decontamination loops using additive/subtractive indexing (§A.2; Figures 11–16).
  - Hallucination mitigation by biasing toward observed continuations in factual contexts; copyright‑risk mitigation by diverting away from long n‑grams that uniquely occur in copyrighted sources (§E).
  - Memorization/plagiarism detection, novelty/creativity measurement, and entity popularity quantification using counts (§E).
- Research directions
  - Learned gating for interpolation: Beyond two global `λ`s, learn instance‑wise trust policies (Random Forest results in Table 5 are a first step).
  - Hybrid retrieval+`∞-gram`: Combine neural semantic retrieval with exact suffix‑based counts, using `∞-gram` for local continuation probabilities.
  - Nonparametric speculative decoding: Use `∞-gram` as the fast proposer in speculative decoding pipelines (§E), similar in spirit to retrieval‑based speculative approaches.
  - Understanding positional representations: Use the `effective n` agreement curves to probe LLM positional encoding behaviors (Figure 5).
  - Multilingual and code domains: Extend indexes and analyses to diverse tokenizers and domains; exploit in‑domain advantage observed in Figure 10.

> Most important takeaways, grounded in the paper’s figures and tables:
> - `∞-gram` is surprisingly predictive on real text, especially when it finds long seen suffixes: 47% overall agreement; >75% at `effective n ≥ 16` (Figure 3).  
> - It complements large LLMs substantially: Llama‑2 70B perplexity improves 4.59 → 3.96 (−18%) on Pile‑val with Pile+RedPajama (Table 1), and SILO gains up to −73% on OOD domains (Table 2).  
> - The `infini-gram` engine makes trillion‑token, unbounded n‑gram modeling practical with 13–180 ms per query (Table 3), 7 bytes/token storage, and additive indexing (§3, §A.2, §A.5).
