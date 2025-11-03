# OLMOTRACE: Tracing Language Model Outputs Back to Trillions of Training Tokens

**ArXiv:** [2504.07096](https://arxiv.org/abs/2504.07096)

## 🎯 Pitch

OLMOTRACE introduces the first real-time system capable of tracing generated text from language models back to their multi-trillion-token training data by identifying verbatim overlaps at unprecedented scale. This innovation empowers researchers and end users to directly inspect the origins of model outputs—enabling robust fact-checking, transparency, and deeper understanding of model behavior, factuality, and originality, which is crucial for trustworthy AI deployment and scientific study of large-scale language models.

---

## 1. Executive Summary
OLMOTRACE is a real-time system that traces segments of a language model’s generated text back to the model’s full, multi-trillion-token training data by finding verbatim matches. It matters because it gives end users and researchers a concrete way to inspect where specific wording in an LM’s output may have been seen during training, enabling analysis of factuality, originality, and training-data provenance at unprecedented scale.

## 2. Context and Motivation
- Problem addressed
  - Modern LMs are trained on massive, often opaque corpora, which makes it hard to understand why a model says what it says. The practical question is: can we connect a model’s generated words to concrete places in its training data?
  - Prior “behavior tracing” methods (e.g., influence functions) are computationally infeasible at trillion-token scale (see §1 and §6).
- Why it’s important
  - Real-world impact: Supports fact checking, IP/copyright audits, PII and toxicity monitoring, and transparency for high-stakes deployments.
  - Scientific significance: Provides a scalable window into the relationship between data and model behavior, which is essential for understanding memorization and generalization.
- Prior approaches and limitations
  - Influence-function-based techniques (Koh & Liang, 2017; Han et al., 2020; §6) require gradients and are extremely costly at this scale.
  - Training-time or in-context citation methods (Khalifa et al., 2024; Huang et al., 2024; Chuang et al., 2025; §6) either modify training or rely on context retrieval, not the original training data itself.
  - Search engines and RAG: retrieve web or external sources at inference time; they don’t reveal connections to the model’s actual training set (§6).
- Positioning
  - OLMOTRACE takes a post-hoc, data-driven approach: it does not modify model training or generation. It directly indexes the full training corpora and surfaces exact (verbatim) overlaps with generated outputs in real time (Abstract; §1–§3).

## 3. Technical Approach
At a high level, OLMOTRACE takes an LM response and surfaces spans from that response that appear verbatim somewhere in the model’s training data, then shows the enclosing documents and ranks them for topical relevance. The pipeline (Figure 2; §3) has five steps.

Key terms (paper-specific or uncommon):
- `span`: a contiguous sequence of tokens within the model’s output.
- `suffix array (SA)`: a data structure that lexicographically sorts all suffixes of a large text corpus and stores pointers to their positions. It supports fast substring search by binary search.
- `infini-gram`: a trillion-token–scale search engine that builds SA-based indexes and exposes fast queries for counting and locating substrings in massive corpora (§3.1).
- `FIND` query: an infini-gram operation that returns the SA segment covering all occurrences of a search string; if the string is absent, the segment is empty but pinpoints neighboring suffixes (§3.1, Figure 3).
- `LCP` (longest common prefix): the longest initial substring shared between two strings.
- `span unigram probability`: the product of unigram token probabilities estimated over the entire training corpus; lower values correspond to longer and rarer spans (§3, Step 2).

Step-by-step pipeline (Figure 2; §3):
1) Find maximal matching spans (Step 1; §3 and §3.1)
   - Goal: identify all spans in the output that appear verbatim somewhere in the training data and are “maximal” (i.e., not contained in a longer matching span).
   - Constraints for spans (§3, Step 1):
     - Existence: must occur at least once in training data.
     - Self-contained: does not include a period or newline unless that delimiter is at the end; and begins/ends at word boundaries (using begin-of-word token indicators).
     - Maximality: not a subspan of another candidate that meets the above.
   - How it works efficiently:
     - For each suffix of the output (i.e., every starting token position), OLMOTRACE finds the longest matching prefix that exists in the training data.
     - Naively, this would require O(L) or O(log L) substring queries per position. Instead, OLMOTRACE uses a single `FIND` over the entire suffix (Algorithm 1; Figure 3).
       - If the suffix exists verbatim (non-empty `FIND` result), the longest prefix equals the whole suffix (up to delimiters/word-boundaries trimming).
       - If not found, the `FIND` returns an empty segment but gives the two neighboring SA entries; the one with the longest common prefix with the search suffix bounds the longest matching prefix. Only these two neighbors need to be compared to determine the LCP length (§3.1; Figure 3).
     - Because `FIND` relies on binary search over the SA, each query costs O(log N) disk lookups; by parallelizing across all suffixes, total latency scales with O(log N) (§3 and §3.1).
     - Implementation detail: the index is sharded (each ≤500B tokens). OLMOTRACE queries all shards in parallel and takes the max LCP (§3.1).
     - After computing lengths, it trims spans to satisfy the delimiter/word-boundary constraints and then suppresses non-maximal spans with a single pass (Algorithm 1: `SUPPRESSNONMAXIMALSPANS`).

2) Keep only long and unique spans (Step 2; §3)
   - To reduce clutter, spans are ranked by `span unigram probability` (lower is better). OLMOTRACE keeps K = ceil(0.05 × L) spans with the smallest probability (§3, Step 2).
   - Why not just longest spans? Using length alone produced worse document relevance downstream; ranking by span unigram probability improved relevance (App. §C; Table 3, rows comparing “span ranking w/ length” vs. final setting).

3) Retrieve enclosing documents (Step 3; §3)
   - For each kept span, retrieve up to 10 training-document snippets that contain it (randomly sample if more than 10 occurrences to keep latency and UI manageable).
   - Because spans are maximal, most appear ≤10 times (§3, Step 3).

4) Merge spans and documents (Step 4; §3)
   - Merge overlapping spans in the output, and merge snippets from the same training document to a single document entry for display.

5) Rerank and color by topical relevance (Step 5; §3)
   - Use BM25 (standard lexical relevance scoring) to rank retrieved documents, treating “prompt + response” as the query and the set of retrieved documents as the corpus (§3, Step 5; footnote 4).
   - Normalize BM25: the maximum score scales with response length; empirically, a cap around 0.18 × character count was observed (Figure 4, middle). Scores are normalized and bucketed into high (≥0.7), medium (0.5–0.7), and low (<0.5) relevance. Colors in the UI reflect this, both for documents and for spans whose colors inherit the best relevance among enclosing documents (§3, Step 5).

Engineering for real-time performance (§3.2; App. §B):
- Hosted on GCP: 64 vCPUs, 256 GB RAM, 40 TB SSD; two VM replicas, multi-mounted disks for availability (§3.2; App. §B).
- Index files are memory-mapped from SSDs (not loaded into RAM) to balance capacity and latency; prefetching disabled to avoid wasting I/O (§B).
- Disk-I/O analysis estimates ~960 random reads per generated token for span computation across 12 shards, enabling ~1.2 s for a 100-token output at 80k IOPS (App. §B).

Training data indexed (Table 1; §2):
- OLMo-2-32B-Instruct’s full training mixture: 3.164B documents and 4.611T tokens.
  - Pre-training: 3081M docs, 4575B tokens (`allenai/olmo-mix-1124`)
  - Mid-training: 81M docs, 34B tokens (`allenai/dolmino-mix-1124`)
  - Post-training: 1.7M docs, 1.6B tokens (SFT, DPO, RLVR)
- OLMOTRACE matches against an LM’s entire training data, including pre, mid, and post-training (§2; Table 1).

## 4. Key Insights and Innovations
- Real-time, trillions-scale verbatim tracing
  - Innovation: Extends the infini-gram SA-based engine to compute, in parallel, the longest matching prefix for every suffix of the model output using a single `FIND` per suffix and neighbor-LCP logic (§3.1; Figure 3; Algorithm 1).
  - Why it matters: Reduces the number of corpus lookups from O(L) per suffix to O(1) `FIND` (each costing O(log N) due to SA binary search), enabling interactive latency over multi-trillion-token corpora (§3.1; §3.2).
- Span selection that balances length and rarity
  - Insight: Selecting spans by “low span unigram probability” (rarity × length proxy) retrieves more topically relevant documents than simply choosing the longest spans (§3, Step 2; App. §C; Table 3).
- Practical, interpretable UI with relevance-driven coloring
  - Mechanism: Use BM25 to rank retrieved documents and map to three relevance buckets, mirrored in span highlighting (§3, Step 5). This guides users to higher-signal matches.
- End-to-end systemization for public use
  - Contribution: A production deployment that indexes the full training data for multiple OLMo models and exposes interactive inspection with document-level context (Figure 1; §2; App. Figure 6). The core components are open-sourced.

These are more than incremental improvements: the neighbor-LCP trick over SA for one-query-per-suffix matching is a fundamental practical advance for latency at this scale, while the rest translate that capability into a usable product.

## 5. Experimental Analysis
Evaluation methodology
- Workload and latency (§3.2)
  - Dataset: 98 real conversations collected from internal OLMo model use in AI2 Playground.
  - Each response averages 458 tokens.
  - Measurement: end-to-end OLMOTRACE inference latency for Steps 1–3 (span finding and document retrieval).
  - Result: 
    > “On average, … latency per query is 4.46 seconds.” (§3.2)

- Span and relevance statistics (§4)
  - Span lengths (after Step 2 filtering): mean 10.4 tokens, median 10 (Figure 4, left).
  - BM25 normalization: maximum attainable scores scale with response length (~0.18 × characters; Figure 4, middle). After normalization, spans/docs are bucketed into high/medium/low relevance; 14% of documents and 19% of spans land in the “high” bucket (§4; Figure 4).

- Human and LLM-as-Judge relevance evaluation (§4; App. §C; Table 2; Table 3)
  - Human rubric: 0–3 scale (Table 2, left) judging topical relevance of the top-5 displayed documents per conversation.
  - Agreement: A later LLM-as-Judge setting (gpt-4o-2024-08-06; Table 2, right) shows strong correlation with human scores (Spearman 0.73; §4).
  - Final configuration results (LLM-as-Judge):
    > “Average … 1.82 on first documents and 1.50 on top-5 documents.” (App. Table 3, “our final setting” row)
  - Ablations in App. Table 3:
    - Using span length instead of unigram probability reduces scores (1.56 first-doc, 1.37 top-5).
    - Shortening document context (500 → 100 tokens) reduces top-5 average (1.44).
    - Ignoring the prompt in BM25 slightly hurts first-doc average (1.78 vs. 1.82).
    - Early setting with human annotation: 1.90 first-doc, 1.43 top-5, but subsequent LLM-as-Judge tuning led to a better balanced final setting.

- Where matches come from (§4)
  - Distribution by training stage:
    > 96.7% pre-training, 0.9% mid-training, 2.4% post-training (0.9% SFT, 1.5% DPO, 0% RLVR). (§4)

Case studies (Figure 5; §5)
- Fact checking: tracing factual claims to pretraining sources; users can open the source URL when available.
- “Creative” expressions: even stylistic output (e.g., Tolkien-esque) can include verbatim fragments present in training data (e.g., “I’m going on an adventure”).
- Math tracing: exact arithmetic steps (e.g., “binom{10}{4} = … = 210”) can appear verbatim in post-training data.

Assessment of support for claims
- The latency and engineering claims are supported by concrete system measurements (§3.2) and a disk I/O analysis (App. §B).
- Relevance claims are backed by:
  - A clear rubric (Table 2), a human evaluation round, and an LLM-as-Judge replication with reasonable agreement (Spearman 0.73; §4).
  - Sensible ablations demonstrating the value of key design choices (App. Table 3).
- Limitations are acknowledged explicitly (verbatim-only, no causal inference; §7 “Limitations”).

## 6. Limitations and Trade-offs
- Verbatim-only matching
  - OLMOTRACE does not surface paraphrases, semantic similarity without lexical overlap, or causal influence. It finds exact token sequences only (§7 “Limitations”).
  - Implication: Many true “influences” on generation will not be surfaced.

- No causal attribution or citation guarantee
  - Matches are not evidence the model “used” the specific document, nor are they citations supporting the content’s truth (§7 “Limitations”).

- UI and retrieval truncation
  - At most 10 snippets per span (§3, Step 3). Very frequent spans are randomly subsampled, which may hide some contexts.

- Dependence on tokenizer and boundaries
  - Maximal spans are constrained by begin-of-word markers and delimiter handling (§3, Step 1). Subword tokenization can split words in ways that slightly affect span discovery.

- Computational constraints and infrastructure dependencies
  - Real-time performance depends on high-IOPS SSDs, careful mmap, and parallelism (App. §B). Lower I/O environments will degrade latency.

- Evaluation scope
  - The main quantitative study is on 98 conversations (§3.2; §4). While sufficient to demonstrate latency and provide initial relevance statistics, broader domain coverage (e.g., scientific, legal, multilingual) is not reported.

- Social/legal risks and mitigations
  - Copyright: News/song lyrics can appear; a takedown mechanism exists; efficient deletion in the index is supported (§7 “Mitigating…”).
  - PII: None observed in their tests, but a regex-based filter is applied (§7).
  - Toxicity: relies on existing AI2 Playground moderation for prompts (§7).

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes that trillion-scale, training-data-grounded tracing can be done in seconds. This turns training-data provenance from an offline research task into an interactive capability for end users and auditors (Figure 1; §3.2).
  - Offers a concrete tool to interrogate memorization and originality: which parts of an output are reproduced vs. composed?

- Follow-up research enabled or suggested
  - Beyond verbatim: add approximate or semantic matching (e.g., edit distance, paraphrase retrieval) to capture non-exact influences while keeping latency manageable.
  - Causality: combine tracing with training logs or gradient-based approximations to estimate influence while respecting compute constraints.
  - Memorization studies: quantify rates and types of verbatim reuse across domains, model sizes, and training regimes; connect to safety topics (PII leakage, copyrighted content).
  - Better relevance modeling: move from BM25 to hybrid lexical–semantic ranking while preserving speed; incorporate prompt vs. response contributions more explicitly (App. Table 3 suggests benefits of including the prompt).
  - Provenance auditing: integrate document-level source integrity checks, crawl dates, and licensing metadata for compliance.

- Practical applications
  - Enterprise governance: audit outputs for potential IP risks and provide tracebacks for compliance.
  - Fact checking and education: let users inspect supporting text in the training data and understand context (Figure 5a).
  - Creative assistance with transparency: show which phrases are novel vs. seen before (Figure 5b).
  - Math and code pedagogy: surface canonical derivations/snippets present in training data (Figure 5c).

Quoted highlights for quick reference
- Latency: 
  > “OLMOTRACE … completes tracing … (avg. ∼450 tokens) within 4.5 seconds on average.” (§1; §3.2)
- Scale:
  > “Total 3164 M documents, 4611 B tokens.” (Table 1)
- Span properties:
  > “mean length of 10.4 tokens” (Figure 4, left)
- Relevance evaluation:
  > “Spearman correlation … 0.73” between human and LLM-as-Judge; final scores “1.82 (first), 1.50 (top-5).” (§4; App. Table 3)
- Limitation:
  > “Retrieved documents should not be interpreted as having a causal effect … or as supporting evidence.” (§7 “Limitations”)

Overall, OLMOTRACE contributes a principled, scalable, and user-facing method to connect LM behavior to its training data. The core algorithmic insight (single `FIND` per suffix + neighbor LCP over a suffix array) and the end-to-end system engineering make real-time tracing feasible at trillions of tokens, opening a practical path toward data-aware LM analysis and governance.
