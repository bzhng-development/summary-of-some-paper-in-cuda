# OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens

**ArXiv:** [2504.07096](https://arxiv.org/abs/2504.07096)
**Authors:** Jiacheng Liu, Taylor Blanton, Yanai Elazar, Sewon Min, YenSung Chen, Arnavi Chheda‑Kothary, Huy Tran, Byron Bischoff, Eric Marsh, Michael Schmitz, Cassidy Trier, Aaron Sarnat, Jenna James, Jon Borchardt, Bailey Kuehl, Evie Cheng, Karen Farley, Sruthi Sreeram, Taira Anderson, David Albright, Carissa Schoenick, Luca Soldaini, Dirk Groeneveld, Rock Yuren Pang, Pang Wei Koh, Noah A. Smith, Sophie Lebrecht, Yejin Choi, Hannaneh Hajishirzi, Ali Farhadi, Jesse Dodge
**Institutions:** University of Washington, Allen Institute for AI, Allen School at University of Washington (implied)

## 🎯 Pitch

OLMOTRACE represents a breakthrough in AI transparency by enabling real-time tracing of large language model outputs to their exact sources within trillion-token training datasets. This tool not only enhances interpretability and supports fact-checking, but also addresses potential copyright and PII issues, paving the way for more transparent and accountable AI development.

---

## 1. Executive Summary
OLMOTRACE is a real-time system that traces an LLM’s generated text back to exact, verbatim matches in its full training data—even when the data spans trillions of tokens. It achieves interactive latency by indexing the training corpus with a suffix-array–based engine (“infini-gram”) and introducing a parallel algorithm that finds all maximal verbatim matches in an output with only O(L log N) work, then surfaces the enclosing source documents ranked by topical relevance.

## 2. Context and Motivation
- Problem addressed
  - Users and developers need to understand why a language model says what it says. The concrete goal: for any generated answer, quickly locate where in the training data the same wording appears so people can inspect the context, verify facts, check potential memorization, and study model behavior (Introduction; Figure 1).
- Why this matters
  - Real-world impact: supports fact checking, spotting potential copyright/PII risks, and diagnosing hallucinations (Abstract; §5).
  - Scientific significance: enables empirical study of how training data relates to model behavior at the scale of trillions of tokens, which prior attribution methods could not handle (§1, §6).
- Prior approaches and gaps
  - Influence-function–style tracing uses gradients to attribute outputs to training points, but is computationally intractable at trillion-token scale (Related Work; Koh & Liang 2017; Han et al. 2020).
  - Training models to cite sources changes the model or training setup (Khalifa et al., 2024), while OLMOTRACE is post-hoc and model-agnostic (§6).
  - RAG and web search retrieve from external indexes that do not reflect the exact model training set; OLMOTRACE restricts retrieval strictly to the training data for the target model (§6).
- Positioning
  - OLMOTRACE offers a practical, scalable, post-hoc tracing tool anchored in verbatim matching. It works over the full pre-, mid-, and post-training datasets of OLMo models (Table 1), returns results in seconds (§3.2), and exposes an interactive UI (Figure 1; Appendix Figure 6).

## 3. Technical Approach
At a high level, OLMOTRACE takes a model response and returns:
1) highlighted spans in the response that occur verbatim in the training data; and
2) source documents (with context) that contain those spans (System Description; §3; Figure 2).

Key terms used once and defined:
- `Suffix array (SA)`: a data structure that stores all suffixes of a text in lexicographic order, enabling fast substring search.
- `FIND`: the infini-gram query that returns the SA segment covering all corpus positions of a search term; if the term does not exist, it returns an empty segment plus the neighboring SA indices.
- `Longest common prefix (LCP)`: the maximum-length prefix two strings share.
- `Span unigram probability`: product of per-token frequencies in the training data for tokens in a span; lower values indicate longer/rarer spans.
- `BM25`: a standard relevance-scoring function used in information retrieval.

End-to-end pipeline (Figure 2; §3)

Step 1: Find maximal matching spans efficiently (§3 Step 1; §3.1; Figure 3; Algorithm 1)
- Objective: identify every “maximal” text span in the output that appears verbatim in the training data.
  - A span must: exist in the corpus; be “self-contained” (no period/newline inside unless at the end; start and end on whole words under the tokenizer); and not be a subspan of a longer qualifying span.
- How it works:
  - Tokenize the output with the Llama-2 tokenizer (§3 Step 1).
  - For each suffix of the output (i.e., the sequence starting at position b), find the length of its longest prefix that occurs in the corpus.
  - Crucial speedup: use only one `FIND` call per suffix.
    - If the full suffix is not found, `FIND` returns an empty SA segment with a left and right bound that coincide. The two neighboring suffixes in SA (at those bounds) are the closest lexicographic neighbors. One of them shares the maximum LCP with the query suffix; by directly comparing against each, the algorithm gets the exact length of the longest matching prefix (Figure 3; Algorithm 1 GETLONGESTPREFIXLEN).
  - Parallelization: all suffixes are processed in parallel; the bottleneck is disk I/O, so the index files are kept on high-IOPS SSDs (Appendix §B).
  - After computing a candidate span for each suffix, remove those that are subspans of another (Algorithm 1 SUPPRESSNONMAXIMALSPANS).
- Complexity: O(L log N) total work (one binary search over SA per suffix; N is corpus size in tokens), and O(log N) latency with full parallelization (§3 Step 1; §3.1).

Step 2: Filter to keep the “interesting” spans (§3 Step 2)
- To avoid clutter, retain only K = ceil(0.05 × L) spans with the smallest `span unigram probability`.
- Rationale: this favors spans that are both longer and rarer in the full training data. During development, using span length alone yielded worse downstream document relevance than the unigram-probability metric (App. §C; Table 3).

Step 3: Retrieve enclosing documents (§3 Step 3)
- For each retained span, retrieve up to 10 document snippets that include the span. If a span appears more than 10 times, randomly sample 10 to control latency and UI overload.

Step 4: Merge spans and documents (§3 Step 4)
- Merge overlapping spans into one highlight to reduce visual clutter.
- Merge snippets from the same document.

Step 5: Rank and color by relevance (§3 Step 5)
- Compute BM25 between the concatenated “user prompt + model response” (as the query) and each retrieved document snippet (treat retrieved documents as the corpus).
- Normalize BM25 scores by an empirically observed length effect (max score ≈ 0.18 × response characters), then bucket into high/medium/low relevance (Figure 4, middle; §4).
- Use color saturation to visually emphasize more relevant documents and their associated spans (§3 Step 5).

Training data coverage and indexing
- OLMOTRACE indexes the full OLMo training sets: pre-training (4.575T tokens), mid-training (34B), and post-training (1.6B), totaling 4.611T tokens across 3.164B documents for OLMo-2-32B-Instruct (Table 1).
- Indexing uses infini-gram’s SA on Llama-2 tokens; the index is sharded (≤500B tokens per shard), and `FIND` runs on all shards in parallel (§3.1).

System implementation and latency
- Deployment: CPU-only nodes (64 vCPUs, 256 GB RAM) with 40 TB SSDs; two replicas; page tables mmap’ed; prefetching disabled to maximize throughput (Appendix §B).
- Disk I/O analysis: ≈960 random reads per output token (12 shards × ~40 SA steps × 2 reads/step). With 80K IOPS, a 100-token output processes in ~1.2s (Appendix §B).
- Observed latency: on 98 Playground conversations (avg 458-token responses), steps 1–3 complete in 4.46 s on average (§3.2).

User experience (Figure 1; Appendix Figure 6)
- Highlights spans in the model response, lists matching training documents with 80-token snippets, and lets users open 500-token context. Users can filter by span or by document and see relevance-coded coloring.

## 4. Key Insights and Innovations
- Single-query longest-prefix matching per suffix (fundamental)
  - Instead of repeated searches or binary lifting over span lengths, OLMOTRACE shows that a single `FIND` per suffix plus comparison with the two neighboring SA entries suffices to recover the longest matching prefix (Figure 3; Algorithm 1). This is the core algorithmic insight enabling O(L log N) work with strong parallelism.
- Scalability to trillions with SSD-resident SA (fundamental)
  - By keeping suffix-array shards on low-latency SSDs and optimizing I/O (e.g., disabling prefetch, batching document retrieval), the system attains interactive latency without massive RAM (Appendix §B). This makes tracing feasible for multi-trillion-token corpora.
- “Uniqueness-aware” span selection (incremental but impactful)
  - Ranking spans by minimal `span unigram probability` rather than raw length yields more relevant retrieved documents, improving end-user utility (App. §C; Table 3).
- Practical relevance normalization and visualization (incremental)
  - BM25 scores are normalized by response length (empirical cap ≈ 0.18 × characters), then bucketed into three levels for intuitive, color-coded inspection (Figure 4, middle/right; §4). This design choice improves navigability and trust.

## 5. Experimental Analysis
Evaluation setup
- Latency and scale
  - 98 real Playground conversations; average response length 458 tokens. Tracing (steps 1–3) averages 4.46 s per response (§3.2).
  - Training data size and composition in Table 1; index sharded across 12 parts (Appendix §B).
- Span characteristics and selection
  - After filtering (Step 2) and before merging (Step 4), spans have mean length 10.4 tokens; distribution in Figure 4 (left). This indicates the system typically surfaces multi-token phrases with substantive content (§4).
- Document relevance scoring and thresholds
  - Observed that max BM25 score increases roughly linearly with response length; normalized with factor 0.18 per character (Figure 4, middle). Thresholds: high ≥0.7, medium 0.5–0.7, low <0.5 (§4).
  - After normalization, 14% of documents are “high relevance” and 19% of spans have at least one “high” document (Figure 4, right; §4).

Human and LLM-judge assessment of relevance (Appendix §C; Table 2; Table 3)
- Human rubric (0–3 scale) focuses on topical match and scope (Appendix Table 2, left). LLM-as-a-Judge uses GPT-4o with a prompt mirroring the rubric (Appendix Table 2, right).
- Agreement: Spearman 0.73 between human and LLM-judge (§4).
- Final configuration performance:
  - “First document” average relevance: 1.82 (0–3 scale).
  - “Top-5 documents” average relevance: 1.50 (Appendix Table 3).
- Ablations (Appendix Table 3) show:
  - Switching span ranking from “length” to “unigram probability” improves average top-5 relevance (1.37 → 1.50 across iterative changes).
  - Including the user prompt (not just the response) in BM25 improves relevance slightly (1.49 → 1.50).
  - Using longer document context (500 vs. 100 tokens) improves scores (1.44 → 1.50).
  - Removing earlier frequency-based dropping of spans (>10 occurrences) improves top-5 coverage but can lower the “first doc” slightly—highlighting a precision–recall trade-off.

Where matches come from (§4)
- Training stage distribution of retrieved documents:
  - 96.7% pre-training, 0.9% mid-training, 2.4% post-training; within post-training: 0.9% SFT, 1.5% DPO, 0% RLVR (§4). This varies by topic (e.g., math-heavy cases can draw more from SFT/RLVR).

Case studies (Figure 5)
- Fact checking
  - Example span “The space needle was built for the 1962 World Fair” is highlighted with links to training documents and source URLs when available (Figure 5a).
- Tracing “creative” language
  - Tolkien-style story includes spans like “I’m going on an adventure” that appear verbatim in fan fiction within training data (Figure 5b).
- Tracing math capabilities
  - AIME-style calculation “binom{10}{4} = 10!/(4!(10-4)!) = 210” is traced to post-training data (Figure 5c).

Do the experiments support the claims?
- Latency and scale are convincingly demonstrated with concrete system specs and I/O analysis (Appendix §B) and measured latency (4.46 s; §3.2).
- Relevance is supported by both human and LLM-judge assessments, with iterative ablations showing sensible gains from design choices (Appendix Table 3).
- The case studies illustrate diverse uses (fact-checking, creativity, math), though they are qualitative.

Selected direct citations of key results
- Latency:
  > “On average, each LM response has 458 tokens, and the OLMOTRACE inference latency per query is 4.46 seconds.” (§3.2)
- Span length:
  > “The spans have a mean length of 10.4 tokens.” (Figure 4, left; §4)
- BM25 normalization:
  > “The maximum attainable BM25 score is roughly capped by 0.18 times the number of characters in the LM output.” (Figure 4, middle; §4)
- Relevance scores:
  > “Final setting achieved average LLM-as-a-Judge scores of 1.82 (first documents) and 1.50 (top-5).” (Appendix §C; Table 3)

## 6. Limitations and Trade-offs
- Verbatim-only, not causal
  - OLMOTRACE reports exact string matches; it does not detect paraphrases, near-duplicates, or semantic influence, and it does not establish that a retrieved document caused the model to generate the span (§7 Limitations).
- Heuristics on span boundaries
  - “Self-contained” requires spans to avoid internal periods/newlines and to start/end on whole words in the tokenizer. This can exclude valid cross-sentence or punctuation-rich matches (§3 Step 1).
- Sampling and k-limits
  - Only up to 10 occurrences per span are retrieved; when spans are frequent, random sampling may miss the most informative instances (§3 Step 3).
- Relevance scoring scope
  - BM25 is computed only over the already retrieved snippets, not the full corpus, so ranking cannot rescue missed retrievals (§3 Step 5).
- Data and deployment constraints
  - Requires full access to the model’s training data and large SSD-backed indexes. The production setup uses 40 TB SSD, 64 vCPUs, and careful I/O tuning (Appendix §B). This may be costly or infeasible for closed models or smaller teams.
- Language/tokenization dependencies
  - Uses the Llama-2 tokenizer for indexing and output tokenization; boundary heuristics (begin-of-word tokens) may behave differently across languages and scripts (§3; Algorithm 1).
- Coverage skew
  - Most matches originate from pre-training data (96.7%), which shapes what users see, and topic variation can bias which training stages are surfaced (§4).
- Risk exposure and governance
  - Surfacing training data can expose copyrighted or sensitive content; the system includes takedown and regex-based PII filters, but residual risks remain (§7 Limitations).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that training-data tracing at trillion-token scale is feasible with interactive latency. This unlocks practical interpretability workflows that were previously out of reach at modern data scales.
- Enabled research directions
  - Beyond verbatim: add fuzzy/semantic matching to capture paraphrases and near-duplicates while keeping latency manageable (e.g., SA-augmented approximate matching or hybrid semantic filters).
  - Toward causal attribution: combine efficient lexical retrieval with gradient- or counterfactual-based analyses on shortlists of candidates to estimate influence with tractable cost.
  - Dataset governance at scale: use tracing to audit training sets for copyright, PII, toxicity, and duplication; support differential takedowns (the paper notes an efficient removal path without reindexing; §7).
  - Evaluation science: build standardized benchmarks for “traceability quality” (e.g., human judgments beyond topicality, context faithfulness, and utility for fact checking).
- Practical applications
  - Compliance and risk: rapid copyright and PII auditing for outputs.
  - Product trust features: “show training sources” panels for enterprise deployments where training sets are known.
  - Education and research: explore how models internalize formulas, code idioms, or stylistic patterns (cf. math and creative writing case studies in Figure 5).
  - Data curation: identify overrepresented phrases or sources in pre-training that may drive memorization.

In sum, OLMOTRACE contributes a concrete, scalable mechanism for tracing LLM outputs to their training data, supported by a key algorithmic insight on suffix-array queries (Figure 3, Algorithm 1) and a set of pragmatic choices—uniqueness-weighted span selection, BM25-based relevance, and SSD-optimized deployment—that together deliver real-time tracing at multi-trillion-token scale.
