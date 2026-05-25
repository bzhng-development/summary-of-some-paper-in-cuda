# OLMOTRACE: Tracing Language Model Outputs Back to Trillions of Training Tokens

**ArXiv:** [2504.07096](https://arxiv.org/abs/2504.07096)

## 🎯 Pitch

OLMOTRACE introduces the first real-time system capable of tracing generated text from language models back to their multi-trillion-token training data by identifying verbatim overlaps at unprecedented scale. This innovation empowers researchers and end users to directly inspect the origins of model outputs—enabling robust fact-checking, transparency, and deeper understanding of model behavior, factuality, and originality, which is crucial for trustworthy AI deployment and scientific study of large-scale language models.

---

## 1. Executive Summary

This paper introduces **OLMOTRACE**, a real-time system that traces language model outputs back to their full, multi-trillion-token training data by locating verbatim text matches. Deployed on OLMo models—including OLMo-2-32B-Instruct—trained on 4.6 trillion tokens across 3.2 billion documents, the system retrieves matching spans and their source documents within an average of 4.5 seconds per response. The core mechanism is a novel parallel algorithm built on infini-gram suffix arrays that computes maximal matching spans using a single FIND query per suffix position (reducing time complexity to O(L log N) rather than enumerating all O(L²) substrings), followed by unigram-probability filtering to retain long, unique spans and BM25 reranking to surface topically relevant documents. The system displays 1.82 average relevance among first-ranked documents and demonstrates utility across fact checking, creative expression tracing, and math capability attribution—establishing that verbatim tracing can connect LM outputs to training data at trillion-token scale, though only for exact lexical matches rather than semantically equivalent or paraphrased content.

## 2. Context and Motivation

### The Core Problem: We Cannot Trace Why LMs Generate Specific Outputs at Scale

Modern language models are trained on trillions of tokens—OLMo-2-32B-Instruct alone consumed 4.6 trillion tokens across 3.2 billion documents (Table 1)—yet when a model produces a response, we have no practical way to determine whether that response originates from something it encountered during training. This is not merely an academic curiosity. As the authors state:

> "As LMs gain adoption in higher-stakes scenarios, it is critical to understand why they generate certain responses."

The verb "understand" here carries significant weight. In deployment contexts ranging from medical advice to legal analysis to educational tools, an LM's output is only as trustworthy as our ability to verify its provenance. If a model states a factual claim, users need to know whether that claim reflects information learned from reliable training sources, or whether the model is confabulating. If a model generates text that appears creative or novel, researchers need tools to investigate whether that creativity is genuine recombination or verbatim reproduction from training data. Currently, for trillion-token-scale models, neither users nor researchers have such a tool.

The scale of the problem is what makes it genuinely hard. The training corpora are not merely large—they are massive in a way that breaks conventional information retrieval assumptions. At 4.6 trillion tokens, even an O(n) scan of the training data would be prohibitively expensive per query. Influence functions (Koh and Liang, 2017), which compute how individual training examples affect a specific model output by leveraging gradient information, scale with model size and training set size in ways that become intractable well before reaching trillion-token scale. The gap this paper identifies is not that attribution is theoretically impossible, but that **no existing method had been demonstrated to work at the scale of contemporary LLM training data in real time**.

### Why the Gap Matters: From Scientific Understanding to Practical Trust

The motivation operates on multiple levels, which the paper addresses implicitly through its design choices and explicitly through its case studies (Section 5).

**Scientific understanding of LM behavior.** The relationship between training data and model outputs is a central question in language model research. Do models primarily memorize and regurgitate, or do they genuinely combine learned patterns in novel ways? Under what conditions do models reproduce training examples verbatim versus paraphrase them? These questions have been studied extensively at smaller scales (Carlini et al., 2021; Kandpal et al., 2022; Lee et al., 2022), but **empirical investigation has been bottlenecked by tooling**—researchers simply cannot search trillion-token corpora interactively. OLMOTRACE enables a new mode of exploration where a researcher can probe a model with prompts and immediately see which training documents match its outputs, facilitating qualitative and quantitative analyses that were previously infeasible.

**Trust and verification for end users.** The paper positions OLMOTRACE as a tool for non-expert users as well, demonstrated by its integration into the Ai2 Playground with an interactive UI (Figure 1, Figure 6). When a model makes a factual claim—"The space needle was built for the 1962 World Fair" (Figure 5a)—a user can click to see the exact training document containing that claim, including its source URL. This transforms the user's relationship with the model from blind trust (or blind skepticism) to **verifiable provenance**. The authors are careful to avoid overclaiming: they explicitly state that "the retrieved documents should not be interpreted as having a causal effect on the LM output, or as supporting evidence or citations" (Limitations). But even with this caveat, seeing where a phrase appeared during training gives users a concrete basis for judgment that is otherwise absent.

**Transparency in the open-source ecosystem.** The paper is situated within the broader movement toward fully open language models. The OLMo family (OLMo et al., 2024; Muennighoff et al., 2024) is designed to make every aspect of model development transparent—training data, code, weights, and logs are all publicly available. But transparency of data is insufficient without **accessibility** of that data. A researcher could, in principle, download the 3.2 billion documents in OLMo's training set and grep through them, but this is impractical for interactive use. OLMOTRACE completes the transparency loop: it makes the training data not just available but *queryable in real time*, turning a theoretical possibility into a practical tool. This aligns with the paper's broader institutional context at the Allen Institute for AI, which has invested heavily in open model development.

### Prior Approaches and Their Limitations at Scale

The paper identifies several categories of prior work and explains why each is insufficient for real-time verbatim tracing at trillion-token scale.

**Influence functions and gradient-based attribution.** The classical approach to tracing model behavior to training data is influence functions (Koh and Liang, 2017), which estimate how removing or upweighting a training example would affect the model's loss on a given test input. Subsequent work extended these ideas to language models (Han et al., 2020; Han and Tsvetkov, 2022), demonstrating that influential training examples can be identified for specific LM predictions. The fundamental limitation, as the paper states, is computational:

> "While effective on a small scale, influence functions are intractable for trillion-token training data due to their high computational cost."

This intractability is structural, not merely a matter of engineering optimization. Influence functions require computing inverse Hessian-vector products and per-example gradients, operations that scale linearly or super-linearly with both model parameters and dataset size. For a model the size of OLMo-2-32B-Instruct trained on 4.6 trillion tokens, even approximate influence function methods would require computation orders of magnitude beyond what could be delivered in seconds. Moreover, influence functions answer a different question: they identify examples that *causally influenced* a prediction, which is a stronger claim than the association-based matching that OLMOTRACE provides.

**Training-based citation approaches.** Khalifa et al. (2024) proposed training LMs to explicitly cite training documents in their outputs by modifying the training objective. This is a fundamentally different intervention: it changes what the model learns during training rather than providing a post-hoc analysis tool. The limitation is practical: it requires retraining models with a modified objective, which is not applicable to already-trained models and may interfere with other capabilities. The paper acknowledges this work but positions OLMOTRACE as complementary—it works on already-trained models without modifying their training process.

**Retrieval-augmented generation (RAG).** The paper explicitly distinguishes OLMOTRACE from RAG systems like Bing Chat, Google AI Overview, and Perplexity AI. The key difference is temporal and functional: RAG retrieves documents *during generation* and feeds them into the model's context to improve the output, whereas OLMOTRACE retrieves documents *after generation* to explain the output. The authors state:

> "Despite looking similar, OLMOTRACE is fundamentally different from RAG: OLMOTRACE retrieves documents post-hoc and does not intervene with the LM generation."

There is also a difference in the retrieval corpus: RAG systems search live web indexes or curated knowledge bases, not the model's actual training data. A RAG-retrieved document might be factually correct and topically relevant but has no necessary relationship to what the model learned during training.

**Search engines as content checkers.** Google's Gemini app includes a "double-check response" feature that highlights response segments and shows similar results from Google Search. Gao et al. (2022) proposed RARR, which retrieves evidence from Google Search to verify and revise LM outputs. The paper identifies that these tools search the *live web*, which is "updated in real time and thus not identical to Gemini's training data, making it less useful for scientific exploration." A training document from 2023 may be a more relevant explanation for model behavior than a 2025 web search result, even if the latter is more factually current.

### How OLMOTRACE Positions Itself

Given this landscape, the paper positions OLMOTRACE around a specific, deliberately scoped capability: **finding and displaying verbatim matches between LM outputs and training data, in real time, at trillion-token scale.** It is not claiming to provide causal attribution, semantic matching, or factual verification. The key design decisions follow from this positioning:

**Verbatim matching as a tractable proxy.** By focusing on exact lexical matches rather than semantic similarity or causal influence, the system sidesteps the computational intractability of influence functions and the engineering complexity of training-time interventions. Verbatim matching with suffix arrays can be made extremely fast—O(log N) per query—enabling the real-time performance that makes the tool interactive. The implicit claim is that verbatim overlap is a useful signal for understanding data-to-output relationships, even though it is not the only signal and certainly not the ground truth of causality.

**Scaling up an existing primitive (infini-gram) with a novel parallel algorithm.** The core technical challenge is not the data structure—suffix arrays are well-established—but rather making suffix array queries fast enough for real-time interactive use across trillion-token corpora when processing potentially hundreds of suffix queries per LM response. The paper builds on infini-gram (Liu et al., 2024) but develops a new parallel algorithm (Section 3.1, Algorithm 1) that processes all suffix positions simultaneously using a single FIND query per position, exploiting the observation that when a search term does not exist in the corpus, the suffix array's neighboring entries reveal the longest matching prefix. This is a genuine algorithmic contribution that makes the difference between a theoretically possible system and a practically usable one.

**User-facing design as a first-class concern.** Unlike many academic tracing systems that produce ranked lists of training examples, OLMOTRACE emphasizes interactive exploration. The UI shows spans highlighted in the response, documents with color-coded relevance, and bidirectional navigation—clicking a span filters to its documents, and clicking a document shows its matching spans (Figure 6). The span filtering criteria (Step 2) and document reranking (Step 5) are explicitly designed to "declutter the UI" and present the most "interesting" matches first. The unigram-probability metric for span selection—prioritizing spans with lower product of token frequencies—reflects an understanding that common phrases ("the United States," "in order to") are uninformative even though they appear verbatim in training data.

**Scope limited to fully open models.** OLMOTRACE is deployed on OLMo models because their training data is publicly available and can be indexed. The authors note that "OLMOTRACE can be applied to any LM as long as the service provider has access to its full training data," but the practical constraint is that the training data must be accessible for indexing. This positions the work as part of the open-source model ecosystem rather than as a general-purpose tool applicable to proprietary models like GPT-4 or Claude, where training data access is unavailable. The paper implicitly argues that this kind of transparency is a benefit of the open model paradigm.

## 3. Technical Approach

### 3.1 Reader Orientation

OLMOTRACE is a production system that, given a language model's generated response, finds every substring of that response that appears verbatim (exactly, character-for-character) in the model's training data, and presents those matches along with their source documents to the user within seconds. The problem it solves is the scale mismatch: the training data contains trillions of tokens, so naively searching for all possible substrings of an LM output against that corpus would be computationally absurd. The shape of the solution is to pre-index the training data into a suffix array (a data structure that sorts all suffixes of the corpus lexicographically), then use a fast parallel algorithm to query every suffix position of the LM output against this index simultaneously, leveraging the property that a single FIND query—even when it returns "not found"—reveals the longest matching prefix by inspecting the suffix array's neighboring entries.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five sequential stages, each feeding into the next:

1. **Infini-gram Index (pre-built, off-line):** The tokenized training data is stored as a suffix array sharded across disks on a cloud VM. This index supports two operations: FIND (return the segment of the suffix array where a query string appears) and GETDOCBYPTR (retrieve the document containing a given position). The index is built once and queried at inference time—it never changes between queries.

2. **Span Computation Engine (Step 1):** Takes a tokenized LM response and the infini-gram index as input. For every suffix position in the response (every token offset), it computes the longest prefix of that suffix that appears verbatim in the training data. These become candidate maximal matching spans. The computation is fully parallelized across suffix positions, each requiring a single FIND query plus inspection of the suffix array neighbors.

3. **Span Filtering (Step 2):** Takes the set of maximal matching spans and selects a subset to display. The filtering criterion is span unigram probability—the product of individual token frequencies in the training data—with lower probability (indicating longer, rarer spans) being preferred. The system keeps the top K spans, where K is proportional to response length.

4. **Document Retrieval and Merging (Steps 3–4):** For each kept span, retrieves up to 10 document snippets containing that span from the training data via the infini-gram index. Overlapping spans are merged into unified highlights. Documents appearing multiple times (because they contain multiple matching spans) are deduplicated, and their snippets are merged.

5. **Relevance Reranking and UI Rendering (Step 5):** All retrieved documents are scored with BM25 using the concatenation of user prompt and LM response as the query. Documents are ranked by this score, bucketed into three relevance tiers (high/medium/low), and displayed with color-coded highlights on both the response text and the document sidebar. Users can interactively click spans to see their documents or click documents to locate their matching spans.

Information flows strictly forward: LM response → tokenization → FIND queries on the index → span list → filtered span list → document retrieval → BM25 reranking → UI rendering. There is no feedback loop or iterative refinement—the pipeline is deterministic given the response and the pre-built index.

### 3.3 Roadmap for the Deep Dive

- **First, the infini-gram index (the pre-built foundation).** Understanding how the training data is stored and queried is prerequisite to understanding why the span computation algorithm works. We will cover suffix arrays conceptually, sharding, and the two core query operations (FIND and GETDOCBYPTR).

- **Second, the maximal matching span algorithm (the core technical contribution).** This is where the paper claims its speed advantage—replacing an O(L²) or O(L log L) enumeration of substrings with O(L) FIND queries, each executing in O(log N) time. We will walk through the algorithm, the FIND-with-neighbor-inspection trick, and the suppression of non-maximal spans.

- **Third, the span filtering logic (Step 2).** The paper must decide which of potentially many matching spans to show the user. We will examine the unigram probability metric, why it was chosen over span length or bigram probability, and the hyperparameter K that controls how many spans are kept.

- **Fourth, document retrieval, merging, and reranking (Steps 3–5).** Once spans are selected, the system must find their enclosing training documents, merge redundant spans and documents, and order documents by relevance. We will cover the BM25 relevance scoring, the color-bucketing scheme, and the bidirectional UI interaction model.

- **Fifth, the production deployment architecture.** The system runs on a specific hardware configuration (64 vCPUs, 256GB RAM, 40TB SSD) with specific disk I/O characteristics. Understanding the deployment reveals how the algorithmic design choices translate into real-world latency (4.5 seconds average).

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and engineering paper** whose core idea is that verbatim substring matching between an LM output and its trillion-token training data can be made fast enough for interactive use by combining a pre-built suffix array index with a parallel algorithm that computes longest matching prefixes in a single query per suffix position.

---

#### The Infini-gram Index: How the Training Data Is Pre-Built for Fast Querying

Before any user query arrives, the entire training data of the target LM is processed into a queryable index. The paper builds on infini-gram (Liu et al., 2024), which is a text search engine designed for trillion-token-scale corpora. Understanding its data structure is essential because the span computation algorithm exploits specific properties of how infini-gram handles queries that do not match.

**What is a suffix array?** Given a text corpus of N tokens, consider every possible suffix—every substring that starts at some position i and continues to the end of the corpus. There are exactly N such suffixes (one starting at each position). A suffix array is simply a sorted list of these N positions, ordered lexicographically by the text that follows each position. If two suffixes share a common prefix, they will appear adjacent in the sorted array. This adjacency property is the key insight: to find whether a query string appears in the corpus, one can binary-search the suffix array—at each step, compare the query string against the text starting at the middle suffix position. If the query matches a prefix of that suffix, the query exists in the corpus, and the matching positions form a contiguous segment of the suffix array.

**Why suffix arrays enable fast substring search.** For a corpus of N tokens, a binary search over the suffix array takes O(log N) steps. Each step requires comparing the query against the corpus at a specific position, which requires O(query length) character comparisons in the worst case. In practice, infini-gram optimizes this with additional data structures (the Longest Common Prefix array, which pre-computes shared prefix lengths between adjacent suffixes, allowing comparisons to skip already-matched characters). The total complexity for a single FIND query is therefore O(query length + log N), but the log N term dominates for the typical query lengths in OLMOTRACE (tens to hundreds of tokens). For the trillion-token corpus, log N ≈ 40 (since log₂(4.6 × 10¹²) ≈ 42), meaning each FIND query requires roughly 40 steps of binary search.

**Sharding.** The infini-gram index is partitioned into shards because a single suffix array for trillions of tokens would exceed practical file size limits and memory addressing. The paper states that each shard is limited to 500 billion tokens. For the 4.6 trillion token corpus (Table 1), this gives approximately 9–10 shards. The paper mentions 12 shards in the disk I/O analysis (Appendix B), which accounts for data growth across all training stages. When querying, FIND is executed on each shard in parallel, and results are aggregated. The paper notes:

> "In case there are multiple shards, we run FIND on each one in parallel, and take the maximum of LCP length from all shards."

**Tokenization.** All text—both the training data and the LM output—is tokenized with the Llama-2 tokenizer before any matching occurs. This is a critical design choice: it means "verbatim" matching operates at the token level, not the character level. A consequence is that the system is sensitive to tokenization boundaries—the span criteria explicitly require that spans "does not begin or end with incomplete words" (Step 1, criterion 2), which is enforced at the token level by checking whether the span boundaries fall at token positions that correspond to word beginnings and endings. The authors implement this by checking for "begin-of-word" tokens in Algorithm 1.

**Two core operations.** The infini-gram engine exposes two operations that OLMOTRACE uses:

1. **FIND(query_string):** Returns the contiguous segment [l, r) of the suffix array where all suffixes begin with the query string. If the query string does not exist in the corpus, the segment is empty (l = r). Crucially—and this is the property the algorithm exploits—even when the segment is empty, the position l (which equals r) points to the location in the suffix array where the query string *would* be inserted if it existed. The suffixes immediately before and after this insertion point are the lexicographically closest matches to the query. These neighboring suffixes are accessible via the infini-gram API, and their text tells us the longest common prefix between the query and the corpus.

2. **GETDOCBYPTR(position):** Given a position in the tokenized corpus, returns the document that contains that position, along with surrounding context. This is used after spans are identified, to retrieve the enclosing document snippets. The paper implemented a batched version to reduce latency for multiple position lookups.

**Disk-based storage.** The suffix array files are kept on SSD disks (not in RAM), because storing the full index for trillion-token corpora in RAM would be cost-prohibitive. The paper states:

> "At inference time [infini-gram] keeps the huge index files on low-latency SSD disks to avoid loading them into RAM."

The VM is allocated 256GB RAM, but this is used for "the fully-materialized page tables of the mmap'ed index files (0.2% the full file size)," not for the index data itself. This means every FIND query triggers physical disk reads. The paper's engineering contribution is making this disk-I/O-bound workload fast enough for interactive use through parallelization. With 12 shards, 40 binary search steps per shard, and 2 disk reads per step (one for the suffix array, one for the text corpus), the system performs 960 disk reads per suffix position in the LM output. With SSDs providing 80,000 IOPS, the disk I/O analysis (Appendix B) concludes:

> "OLMOTRACE can process, for example, a 100-token LM output within 1.2 seconds."

**Prefetching is disabled.** The paper explicitly notes:

> "In the infini-gram engine, we turn off prefetching (setting all prefetch depth to 0) because it would slow down the overall inference."

The reasoning is that prefetching performs speculative disk reads to reduce single-query latency, but when many queries are executed in parallel and disk I/O throughput is the bottleneck, these speculative reads consume bandwidth without providing value—they slow down the queries that actually need the I/O capacity.

---

#### Step 1: Finding Maximal Matching Spans (The Core Algorithm)

This is the most technically novel component of the paper. Given a tokenized LM output of length L (where typically L ∈ [10², 10³]), the goal is to find all text spans in the output that satisfy three criteria: they appear verbatim somewhere in the training data (Existence), they respect word and sentence boundaries (Self-contained), and they are not contained within any larger span that also satisfies the first two criteria (Maximality).

**Why this is computationally challenging naively.** An LM output of length L has O(L²) possible substrings. For L = 450 tokens (the average in the paper's evaluation), that is approximately 100,000 substrings. Checking each substring against the training data via FIND queries would require 100,000 × O(log N) operations, which is already expensive. But the real cost is I/O: each FIND query requires ~960 disk reads across all shards, and 100,000 × 960 = 96 million disk reads, which would take over 1,000 seconds even at 80,000 IOPS. The paper's key insight is that **we only need to query suffixes, not all substrings**, because any matching substring is a prefix of some suffix.

**The algorithmic approach: query every suffix position.** For each token position b in the LM output (where b ranges from 1 to L), the system considers the suffix S[b:L] (the text from position b to the end). It computes the longest prefix of S[b:L] that appears verbatim in the training data. This gives, for each starting position b, exactly one span [b, b + len) where "len" is the length of the longest matching prefix. This is the unique maximal matching span starting at position b—if any longer span starting at b existed in the training data, it would have been found as a longer prefix.

**The single-FIND trick for longest prefix computation.** The naive approach to finding the longest matching prefix of S[b:L] would require multiple FIND queries: try progressively longer prefixes until one fails to match, or use binary search over prefix lengths (which would take O(log L) FIND queries per suffix). The paper's innovation—and the reason the system achieves real-time performance—is that the longest matching prefix can be found with **a single FIND query** by exploiting what happens when FIND returns an empty segment.

Here is the critical property: when FIND(S[b:L]) is called and the entire suffix S[b:L] does not appear in the corpus, infini-gram returns an empty segment bounded by position l (where l = r). At this position l in the suffix array, the suffixes in the corpus that lexicographically precede and follow S[b:L] are adjacent. By inspecting these two neighboring suffixes—specifically, by computing the Longest Common Prefix (LCP) between S[b:L] and each of these two corpus suffixes—the system determines the longest prefix of S[b:L] that *does* appear in the corpus. The formal guarantee is that the maximum of these two LCP values equals the length of the longest matching prefix.

The paper's Algorithm 1 encodes this logic in the GETLONGESTPREFIXLEN subroutine:

> "We use the fact that when the search term does not exist in the text corpus, FIND would return a 0-length segment... where the previous (or next) SA element corresponds to the suffix in the text corpus that lexicographically precedes (or succeeds) the search term. Consequently, the suffix in the text corpus that shares the longest common prefix (LCP) with the search term must come from one of these two neighboring suffixes, and inspecting these two suffixes would tell us the length of the longest matching prefix for this search term."

There is a subtlety: when the query *does* exist in the corpus (i.e., FIND returns a non-empty segment), the longest matching prefix is simply the full query length—the entire suffix appears verbatim. Algorithm 1 handles this case explicitly: `if l ≠ r then len ← |s|`, setting the match length to the full query length.

**The self-contained and word-boundary constraints.** After finding the raw longest matching prefix, Algorithm 1 applies additional constraints related to word boundaries and sentence delimiters. Specifically, the while loop at the end of GETLONGESTPREFIXLEN decrements `len` until the span satisfies:

- The span does not contain a period token (.) or newline token (`\n`) internally—they are only allowed at the very end of the span,
- The span does not begin or end with incomplete words (checked via begin-of-word token positions).

These constraints are applied post-hoc: the algorithm finds the raw longest prefix, then truncates it from the right until the constraints are satisfied. The constraint about period and newline tokens being only at the end of the span is designed to keep spans within sentence boundaries, avoiding spans that cross sentence breaks (which would be less meaningful as coherent textual units).

**Parallelization.** All L suffix positions are processed independently, making the computation embarrassingly parallel. The paper implements this as a parallel for-loop over beginning positions:

> `for b = 1, . . . , L do`  ▷ `execute in parallel`

In practice, this means all L FIND queries are issued simultaneously to the infini-gram engine. Since the disk I/O subsystem can handle many concurrent read requests (80,000 IOPS), and each query involves sequential dependency within its binary search but no dependency between queries, the total latency is dominated by the slowest single query (plus some queuing overhead) rather than the sum of all query latencies.

**Suppressing non-maximal spans.** After computing the longest matching prefix for each suffix, the system has L spans, one starting at each position. However, many of these are subsumed by longer spans starting at earlier positions. For example, if the LM output contains "the quick brown fox" and "the quick brown" appears in the training data, then the span starting at word "the" might be 4 tokens, while the span starting at "quick" might be 3 tokens. The second span is entirely contained within the first and is therefore not maximal.

The paper's SUPPRESSNONMAXIMALSPANS procedure (Algorithm 1) processes spans in order of increasing beginning position and only keeps a span if its ending position exceeds the maximum ending position seen so far:

```
sort spans by beginning position in ascending order
maxend ← 0
for (b, e) in spans do
    if maxend < e then
        maxend ← e
        keep span (b, e)
```

This is a standard greedy interval-covering algorithm: it keeps the "rightmost-reaching" spans and discards any span that is entirely covered by previously seen spans. The result is a set of spans that collectively cover all token positions that appear in any matching span, with each span being maximal (not contained in any other matching span). This reduces the output from O(L) spans to a much smaller number—typically the number of distinct verbatim regions in the output.

**Time complexity.** The serial time complexity for a single suffix is O(log N + query length) for the FIND query plus O(LCP) for inspecting the neighboring suffixes. With full parallelization and sufficient disk I/O bandwidth, the wall-clock latency is dominated by the O(log N) binary search depth (≈40 steps) multiplied by the per-step disk read time, which is limited by IOPS. The paper's empirical measurement shows this translates to **4.46 seconds for an average 458-token response** (Section 3.2), which is consistent with the disk I/O analysis.

**The two-FIND per suffix detail.** The paper notes that retrieving the actual document positions for the longest matching prefix requires a second FIND query:

> "Note that to retrieve documents containing the longest matching prefix, we need to run a second FIND query to locate all its occurrences in the SA. In practice, we run this query immediately after the first one to leverage temporal locality in the disk cache."

The first FIND (with the full suffix) identifies the length of the longest matching prefix. The second FIND (with the specific longest matching prefix string) retrieves all positions in the corpus where this prefix occurs, enabling document retrieval in later steps. Running them sequentially exploits the fact that nearby disk blocks are likely still in the SSD's cache after the first query.

---

#### Step 2: Filtering Spans by Unigram Probability

After Step 1, the system has a set of maximal matching spans—typically dozens for a response of hundreds of tokens. Not all of these are informative. Common phrases like "the United States" or "in order to" appear verbatim in the training data billions of times and are matched frequently, but they reveal nothing interesting about the relationship between this specific output and the training data. The filtering step aims to select a subset of spans that are more likely to be "interesting"—long, distinctive, and potentially informative about the provenance of the content.

**The span unigram probability metric.** For each span, the system computes:

$$\text{span\_unigram\_probability} = \prod_{t \in \text{span}} p_{\text{unigram}}(t)$$

where `$p_{\text{unigram}}(t)$` is the unigram probability of token t in the LM's training data—simply the frequency of token t divided by the total number of tokens in the training corpus.

**What it computes:** the product of individual token probabilities across all tokens in the span. A span consisting entirely of common tokens (like "the", "of", "in") will have a high product; a span containing rare tokens (proper nouns, technical terms, specific numbers) will have a low product. Since probabilities are always ≤ 1, longer spans also tend to have lower products than shorter spans (more multiplicands, each ≤ 1, reduce the product). This means the metric implicitly captures both length and rarity.

**Why this form was chosen over alternatives.** The paper compared this metric against ranking spans by their raw length (number of tokens) and found that unigram probability produced more relevant documents downstream (Table 3). When the system ranked spans by length instead, the average LLM-as-a-Judge relevance score for first documents dropped from 1.82 to 1.56. The authors explain:

> "We found that ranking with the span length metric leads to worse relevance level on documents retrieved from the filtered spans."

The intuition is that pure length ranking can select long but uninformative boilerplate phrases (e.g., a 15-token legal disclaimer that appears identically in thousands of documents), while unigram probability differentiates between a long common phrase and a long distinctive phrase by penalizing spans full of common tokens.

**Why unigram rather than bigram or trigram.** The paper explicitly addresses this:

> "We chose unigram over bigram or trigram because computing them (either online or pre-caching) takes a lot of time."

Higher-order n-gram probabilities would better capture multi-token collocations (e.g., "New York" is more informative than its unigrams suggest), but computing and storing bigram or trigram probabilities for a trillion-token corpus with a vocabulary of 32,000 tokens would require enormous memory (bigram: ~1 billion entries; trigram: ~30 trillion entries, most of which are zero). The unigram approximation is fast—token probabilities can be pre-computed once from training data statistics and cached as a simple lookup table of size equal to the vocabulary—and the paper found it to be effective enough.

**Pre-computation.** The token unigram probabilities are "pre-computed from statistics of the LM's entire training data" and cached. At inference time, computing a span's unigram probability is a simple table lookup for each token followed by multiplication, which is negligible in cost compared to the suffix array queries.

**The selection threshold K.** The system does not apply an absolute probability threshold; instead, it keeps the K spans with the *smallest* (lowest probability = most distinctive) unigram probability, where K is defined as:

$$K = \lceil 0.05 \times L \rceil$$

where L is the length of the LM output in tokens. For a typical 458-token response, K ≈ 23 spans are kept.

**What it computes:** ceiling of 5% of response length. For a 100-token response, 5 spans are kept; for a 500-token response, 25 spans are kept. The proportion (5%) is a hyperparameter chosen during development.

**Why proportional to response length.** Longer responses tend to contain more distinct topics and more potential matching spans. A fixed K would either overload the UI for short responses (showing too many uninteresting spans) or under-represent long responses (missing potentially informative matches). The proportional threshold adapts to response verbosity.

---

#### Step 3: Retrieving Enclosing Documents

For each span surviving the filtering step, the system must now find the training documents that contain that span.

**The frequency cap.** The paper notes a practical constraint:

> "Due to the maximality criterion in step 1, most spans appear no more than 10 times."

The maximal matching spans are typically long enough (mean 10.4 tokens, Section 4) and distinctive enough that they do not appear in hundreds or thousands of training documents. This is a consequence of the maximality property: a span that appears very frequently would likely be subsumed by an even longer frequent span, or it would be a common phrase that the unigram filtering step already deprioritized. If a span exceeds 10 occurrences in the training data, the system randomly samples 10 of them:

> "If a span exceeds this limit, we randomly sample 10 to keep retrieval time manageable and avoid UI overload."

This random sampling is a pragmatic choice to bound retrieval cost and prevent the document panel from being dominated by documents for a single common span.

**Retrieval mechanism.** The system uses infini-gram's document retrieval capability (the GETDOCBYPTR operation, batched for performance) to fetch 80-token snippets centered on each span occurrence. The 80-token context window is a fixed width that provides enough surrounding text for a user to understand the document's topic and the span's role within it, without overwhelming the UI. The extended document view (500 tokens) is available on demand when a user clicks "View Document."

---

#### Step 4: Merging Spans and Documents

**Span merging.** Overlapping spans are merged (union) into single highlighted regions in the UI. This is a cosmetic operation: if two matching spans in the LM output are adjacent or overlap (e.g., "the quick brown" and "brown fox jumps"), they are displayed as a single continuous highlight rather than two adjacent highlights. This reduces visual clutter.

**Document deduplication.** Since multiple spans may appear in the same training document, the system merges document entries that refer to the same source:

> "If two snippets are retrieved from the same document, we merge them into a single document to be displayed in the document panel."

The merged document shows the union of its matching spans, avoiding redundant entries in the document list.

---

#### Step 5: BM25 Reranking and Color-Coded Relevance

The document panel must present documents in a useful order, and the response highlights must communicate which spans are associated with the most relevant documents. This is a user experience problem: with potentially dozens of documents spanning multiple relevance levels, the order and visual weight matter.

**BM25 scoring.** Each retrieved document is assigned a BM25 score using the concatenation of the user prompt and the LM response as the query, and treating the set of all retrieved documents for this query as the corpus:

> "The per-document BM25 score is computed by treating the collection of retrieved documents as a 'corpus', and the concatenation of user prompt and LM response as the 'query'."

BM25 is a bag-of-words retrieval function that scores documents based on term frequency (how often query terms appear in the document), inverse document frequency (how rare those terms are across the corpus), and document length normalization. The paper uses an existing implementation (`rank_bm25`).

**Why BM25 was chosen.** The paper states:

> "We use this BM25 score because it has fairly high agreement with human judgment on topical relevance (§4), and can be quickly computed using CPUs."

BM25 is computationally lightweight (no neural model inference, no GPU required) and well-understood. For the scale of OLMOTRACE—reranking perhaps 50–200 documents per query—BM25 adds negligible latency. The human evaluation validation (Section 4, Appendix C) confirmed that BM25 scores correlate with human relevance judgments (Spearman correlation of 0.73 between human and LLM-as-a-Judge scores, and the judge was tuned to agree with humans).

**Score normalization.** The paper discovered that the maximum achievable BM25 score is roughly proportional to the response length:

> "We found that the maximum attainable BM25 score is roughly capped by 0.18 times the number of characters in the LM output."

The authors normalize each BM25 score by this response-length-dependent ceiling, producing scores in [0, 1]. Without this normalization, longer responses would naturally accrue higher BM25 scores (more query terms to match), making cross-response comparisons inconsistent and the relevance thresholds arbitrary.

**Relevance bucketing.** Normalized scores are bucketed into three tiers:

- **High relevance:** normalized score ≥ 0.7
- **Medium relevance:** normalized score between 0.5 and 0.7
- **Low relevance:** normalized score < 0.5

The thresholds were set empirically to align with human expectations. In the evaluated corpus, this assigns 14% of documents to high relevance, with the majority falling into medium and low categories.

**Two-directional coloring.** The relevance tiers are reflected in the UI in two complementary ways:

1. **Document sidebar:** Each document entry has a colored sidebar indicating its relevance tier—most saturated for high relevance, least saturated for low relevance.

2. **Response span highlights:** Each span's relevance level is set to the maximum relevance among all documents enclosing that span:

> "A span's relevance level is computed as the maximum relevance level among documents enclosing the span."

If a span appears in one high-relevance document and three low-relevance documents, the span is highlighted with the saturated "high relevance" color. This design choice means users see the "best" connection for each span, which is the most useful signal for exploration—a span that has *any* highly relevant training document is worth investigating.

The color encoding is functional, not merely decorative: it guides the user's attention to the most promising entry points for exploration. The paper states:

> "As a result, users are more likely to find highly relevant documents for spans highlighted with the most saturated color."

**Reranking order.** Documents in the sidebar are sorted by BM25 score in descending order, so the highest-relevance documents appear at the top regardless of which spans they correspond to. This is a design choice that prioritizes showing the most topically relevant documents first, even if they correspond to spans that are not the longest or most distinctive.

**A design note: BM25 uses only response (not prompt) in final configuration.** Table 3 tracks an incremental change where the BM25 scorer was initially configured to use only the LM response as query, then changed to use "user prompt + LM response." The inclusion of the prompt improved first-document relevance from 1.78 to 1.82 (LLM-as-a-Judge score), suggesting that the user's question provides useful topical context for ranking documents.

---

#### Interactive UI Features (Bidirectional Exploration)

The OLMOTRACE UI (Figure 1, Figure 6, Appendix Figure 6) supports two interaction patterns that enable non-linear exploration:

**Span-first exploration (click on a highlight).** When a user clicks on a highlighted span in the response, the document panel is filtered to show only documents that contain that specific span. This allows a user to ask: "Where did the model learn to say *this specific phrase*?" and immediately see the source documents.

**Document-first exploration (Locate Span button).** When a user clicks "Locate Span" on a document in the sidebar, the span highlights in the response are filtered to show only spans that appear in that document. This allows a user to ask: "What parts of the response came from *this particular source*?" The operation is reversible—clicking the same button again or the "Clear Selection" button restores the full view.

**Extended document view.** The initial document display shows an 80-token snippet centered on the matched span. Clicking "View Document" expands this to an extended context of 500 tokens. This accommodates two use cases: quick scanning (80-token view) and deep investigation (500-token view). For web-crawled documents, the extended view may include the source URL, enabling users to visit the original webpage.

---

#### Summary of Key Design Decisions and Their Rationale

- **Token-level matching with Llama-2 tokenizer:** ensures exact correspondence with how the model processes text during training, but means "verbatim" is tokenization-dependent rather than character-level.
- **Single-FIND longest prefix computation:** replaces O(L) or O(log L) queries per suffix with O(1) by exploiting the suffix array's lexicographic ordering property and the neighbor-inspection trick—this is what makes real-time performance possible at trillion-token scale.
- **Maximality criterion (suppress non-maximal spans):** reduces O(L) raw spans to the minimal set covering all matched regions, a standard greedy interval-covering solution that is both optimal and simple.
- **Unigram probability for span ranking:** a practical compromise between effectiveness (better than raw length) and computational feasibility (much cheaper than bigram/trigram models for trillion-token-scale pre-computation). The 5% selection threshold (K = ⌈0.05L⌉) is a tuned hyperparameter validated by downstream document relevance.
- **Sampling cap of 10 documents per span:** a pragmatic bound to maintain fast retrieval and usable UI when spans appear in many training documents (which is rare for maximal spans but can occur).
- **BM25 with prompt+response as query:** provides fast, CPU-based topical relevance ranking validated against human judgments, with response-length normalization to enable consistent thresholding.
- **Three-tier bucketed coloring (high/medium/low):** reduces a continuous score to an interpretable visual signal, with the "maximum relevance across documents" rule for spans ensuring users can find the best connection for each phrase.
- **Sharded, disk-resident infini-gram with disabled prefetching:** reflects an engineering optimization where parallel query throughput (IOPS) is the bottleneck, not single-query latency, so speculative reads are counterproductive.
- **Batched GETDOCBYPTR:** reduces per-document retrieval latency by issuing multiple position lookups simultaneously, exploiting the disk subsystem's ability to handle concurrent requests.

## 4. Key Insights and Innovations

### Innovation 1: Verbatim Matching as a Tractable, Scalable Proxy for Data Attribution at Trillion-Token Scale

The paper's most fundamental intellectual move is a deliberate *lowering of ambition* that paradoxically unlocks far more practical capability than any prior approach. The field's dominant paradigm for linking model outputs to training data has been influence functions (Koh and Liang, 2017; Han et al., 2020; Han and Tsvetkov, 2022), which attempt to answer the causal question: "which training examples *made the model produce* this output?" This is the most scientifically rigorous framing—it asks about mechanism, not just correlation—but it imposes computational requirements that scale with both model parameters and dataset size, making it categorically impossible at trillion-token scale in real time.

OLMOTRACE abandons causality entirely and asks a simpler question: "which training documents *contain the exact same text* as this output?" This is not a claim about influence. The authors are explicit: "the retrieved documents should not be interpreted as having a causal effect on the LM output." What makes this move intellectually distinctive is that it trades theoretical completeness for empirical utility at a scale where the complete answer is unavailable. The insight is that at multi-trillion-token scale, **even correlation-level tracing was previously impossible**, and making it possible—even in a deliberately limited form—enables a qualitatively new mode of interaction with models.

This is a **fundamental reframing**, not an incremental improvement. Prior work implicitly assumed that the gap between "verbatim match" and "training influence" was too large for the former to be useful. OLMOTRACE argues—through its deployment rather than through theoretical argument—that verbatim tracing is useful enough to justify the tool, and that the interactive, exploratory use cases (fact checking, creativity tracing, capability attribution) do not require causal claims. The ~4.5-second latency and 1.82 average first-document relevance score (Table 3) demonstrate that the tradeoff pays off in practice. The paper essentially defines a new point on the precision-vs-scale Pareto frontier that the field didn't know existed: real-time, trillion-token, verbatim-only attribution with moderate relevance quality, as opposed to small-scale, high-fidelity causal attribution or large-scale but training-data-agnostic search engine verification.

### Innovation 2: Suffix Array Neighbor Inspection Eliminates the Per-Suffix Query Multiplier

The algorithmic contribution—using a single FIND query per suffix to compute the longest matching prefix by inspecting neighboring suffix array entries when the query fails—is a genuinely clever exploitation of a well-known data structure property for a purpose its designers did not anticipate. Suffix arrays have been used for decades in string search (Manber and Myers, 1993), and infini-gram (Liu et al., 2024) had already scaled them to trillion-token corpora. The standard approach for finding the longest substring match would require either O(L) queries (trying incrementally longer prefixes) or O(log L) queries (binary search over prefix lengths) for each of L suffix positions.

What OLMOTRACE recognizes is a specific property of the infini-gram FIND operation: when a query string does not exist in the corpus, the returned empty segment's boundary pointer sits at the lexicographic insertion point, and the two neighboring suffixes are guaranteed to contain the longest common prefix. This reduces each suffix position's processing from O(L) or O(log L) FIND queries to exactly one FIND query plus a constant-time inspection of two neighboring suffixes. The paper states this plainly:

> "We use the fact that when the search term does not exist in the text corpus, FIND would return a 0-length segment... where the previous (or next) SA element corresponds to the suffix in the text corpus that lexicographically precedes (or succeeds) the search term."

This is an **incremental algorithmic insight** with **fundamental practical consequences**. It is the difference between a system that completes in hours (O(L² log N) or O(L log L log N)) and one that completes in seconds (O(L log N) with parallelization). Without this insight, the entire system concept—real-time interactive tracing at trillion-token scale—is computationally infeasible regardless of hardware. The paper's deployment numbers make this concrete: each suffix position triggers ~960 disk reads, and with 80,000 IOPS SSDs, a 100-token output processes in 1.2 seconds. If each suffix required even 2 FIND queries (a modest alternative), the latency would double; at 10 FIND queries, it would exceed 10 seconds and lose interactivity. The fact that the paper found an O(1)-queries-per-suffix solution is what makes the engineering feasible within a reasonable hardware budget.

### Innovation 3: Unigram Probability as a Cheap, Effective Proxy for Span Interestingness

At first glance, ranking matching spans by their unigram probability—the product of individual token frequencies—seems almost too simplistic to work. It ignores token order entirely (treating "dog bites man" and "man bites dog" identically) and uses only first-order frequency statistics from the training corpus. Yet the paper demonstrates empirically that this metric outperforms the most obvious alternative (ranking by raw span length) for downstream document relevance, with a first-document LLM-as-a-Judge score of 1.82 for unigram ranking versus 1.56 for length ranking (Table 3).

The conceptual insight is that **the metric captures an interaction between length and rarity that neither alone captures well**. Pure length ranking selects long boilerplate phrases—a 20-token legal disclaimer that appears in thousands of documents would rank highly but is uninformative about any specific output's provenance. Pure rarity ranking (minimum token frequency) would select short spans containing a single rare word, missing the contextual information that longer distinctive spans provide. The multiplicative form of unigram probability naturally penalizes spans full of common tokens while rewarding spans with rare vocabulary, and the accumulation of multiplicands (each ≤ 1) means length contributes to distinctiveness only when the tokens themselves are not overwhelmingly common.

This is an **incremental refinement** with **practical significance** for system design. The paper does not claim this metric is theoretically optimal; they chose it over bigram and trigram alternatives explicitly because those are computationally prohibitive ("computing them (either online or pre-caching) takes a lot of time"). In a system where span computation already relies on careful I/O optimization, adding a second expensive computation for span ranking would undermine the real-time guarantee. The unigram approach requires only a vocabulary-sized lookup table pre-computed once from training data statistics, making it essentially free at inference time. The fact that it also empirically outperforms the natural baseline (length ranking) suggests it is not merely a compromise but actually captures something meaningful about what makes a span informative.

The selection threshold (K = ⌈0.05 × L⌉, keeping 5% of spans proportional to response length) is a tuned hyperparameter, not a derived optimum, but the design logic is sound: longer responses contain more distinct topics and more matching spans, so a fixed threshold would either overwhelm short-response UIs or starve long-response exploration.

### Innovation 4: Difficulty-Agnostic Tracing as a Democratic Research Tool

This innovation operates at the level of **system philosophy and usage paradigm** rather than algorithm design. Prior tools for training data attribution—whether influence functions, training-time citation models (Khalifa et al., 2024), or search-engine-based verification (Gemini's "double-check")—are designed around a specific analytical goal: proving influence, providing citations, or fact-checking respectively. Each imposes a particular interpretive frame on the relationship between output and training data.

OLMOTRACE deliberately avoids imposing a frame. It provides raw verbatim matches with color-coded topical relevance and bidirectional navigation (click span → see documents, click document → see spans), and explicitly disclaims any causal or factual interpretation: "the retrieved documents should not be interpreted as having a causal effect on the LM output, or as supporting evidence or citations for the LM output." The tool does not tell the user what the matches *mean*; it enables the user to *explore* and decide.

This is a **reframing of the tracing problem** from a classification task (is this output supported/memorized/influenced?) to an exploration task (what training data does this output resemble?). The case studies in Section 5 illustrate this breadth: fact checking (Figure 5a), creativity tracing (Figure 5b), and math capability attribution (Figure 5c) are three qualitatively different analytical goals, all served by the same underlying verbatim matching mechanism without per-use-case customization.

The significance is that OLMOTRACE is not optimizing for one metric or one user persona. A fact-checker uses it differently from a copyright researcher, who uses it differently from a model developer debugging memorization. The tool's value lies in enabling these diverse inquiries at a scale where they were previously impossible, not in providing definitive answers to any single one of them. This **general-purpose design philosophy**—enabling rather than answering—is unusual in the attribution literature and positions OLMOTRACE as infrastructure rather than as a solution to a specific problem.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The evaluation uses 98 conversations collected from internal usage of OLMo models in the Ai2 Playground. These are not from a standardized benchmark—they represent organic user interactions with the deployed models. The paper does not specify filtering or selection criteria beyond "internal usage," so the dataset reflects whatever queries Ai2 users happened to pose. This is a deliberate choice: evaluating on real user conversations tests the system's performance under realistic conditions rather than on curated test suites. The average LM response length in this set is 458 tokens (Section 3.2), giving a concrete sense of the typical workload.

- **Base model(s).** The system is deployed on three OLMo models (OLMo et al., 2024; Muennighoff et al., 2024), with OLMo-2-32B-Instruct as the flagship. The training data index covers all stages—pre-training (4,575 billion tokens across 3,081 million documents), mid-training (34 billion tokens, 81 million documents), and post-training SFT/DPO/RLVR (1.6 billion tokens, 1.7 million documents)—totaling 4,611 billion tokens across 3,164 million documents (Table 1). The other two OLMo models have comparable training data sizes. The models were chosen because their training data is fully open, enabling the construction of a complete index, which is a hard prerequisite for the system.

- **Metrics.** The evaluation uses two categories of metrics, measured on the 98-conversation sample:

  **Latency:** End-to-end wall-clock time for steps 1–3 (span computation through document retrieval), measured in seconds per query. The paper reports average latency (4.46 seconds) and characterizes the LM response length distribution (mean 458 tokens). Latency is measured on the production hardware described in Appendix B (64 vCPUs, 256GB RAM, 40TB SSD with 80,000 IOPS). Steps 4–5 (merging, BM25 reranking, UI rendering) are not included in the latency benchmark, which is a minor undercount—but these steps add negligible time relative to the I/O-bound span computation.

  **Document relevance:** A human-annotated Likert-scale score from 0–3 measuring how topically relevant each retrieved document is to the LM response, using the rubric in Table 2 (left). The rubric operationalizes relevance as topical alignment: 0 = different topic, 1 = broader topic or insufficient information, 2 = right topic but different context or overly specific, 3 = direct match in topic and scope. The paper reports average scores for the first displayed document and for the top-5 displayed documents, plus the percentage of documents scoring 2 or 3 (considered "relevant"). Human annotation was performed by a single expert (the paper uses the singular "a human expert" in Appendix C), and was later supplemented with LLM-as-a-Judge evaluation using gpt-4o-2024-08-06 with a prompt closely following the human rubric (Table 2, right). The LLM judge achieved a Spearman correlation of 0.73 with the human annotator, which the paper treats as sufficient agreement to use LLM-as-a-Judge for hyperparameter tuning. The LLM judge assigned slightly lower scores overall than the human annotator (1.73 vs. 1.90 on first documents under the same hyperparameter setting; Table 3).

- **Baselines.** OLMOTRACE does not compare against alternative tracing systems (influence functions, training-time citation models, or search engine verification) on quantitative metrics. The comparisons in the evaluation are **internal ablations**—different hyperparameter configurations of OLMOTRACE measured against each other rather than against external baselines. The paper tracks incremental changes across five configurations in Table 3:

  1. **Early setting + human annotator:** The initial hyperparameter configuration, evaluated by the human expert. This setting dropped maximal matching spans appearing more than 10 times in the training data, ranked spans by length, used BM25 with only the LM response (no user prompt), and limited document context to 100 tokens. Achieved 1.90 average first-document score and 1.43 top-5 score under human evaluation.

  2. **Same setting + LLM-as-a-Judge:** The identical configuration evaluated by gpt-4o, establishing the 0.73 correlation baseline and producing scores of 1.73 (first) and 1.28 (top-5).

  3–6. **Incremental improvements:** Each subsequent row in Table 3 applies one change and re-evaluates with LLM-as-a-Judge: removing the frequency-10 drop filter, switching from length ranking to unigram probability ranking, extending document context from 100 to 500 tokens, and incorporating the user prompt into the BM25 query. The final configuration ("our final setting") achieves 1.82 (first) and 1.50 (top-5) under LLM-as-a-Judge.

  There is no external baseline (e.g., a grep-based approach, a random document retrieval, or a retrieval using standard search engine APIs on the training data) to contextualize these absolute relevance scores. The paper implicitly treats the early hyperparameter setting as the baseline for improvement, but this is an internal comparison, not a comparison against alternative methodological approaches.

- **Generation budget / compute accounting.** There is no "generation budget" in the traditional sense, because OLMOTRACE does not generate text—it analyzes already-generated text. The relevant resource constraint is **latency** (time to complete the tracing pipeline) and **disk I/O** (the number of random reads required). The paper's compute accounting operates in terms of:

  **Per-suffix FIND queries:** 1 query per token position in the LM output, with each query requiring approximately 960 disk reads (40 binary search steps × 2 reads per step × 12 shards; Appendix B). For a 100-token response, this is 96,000 disk reads, which at 80,000 IOPS yields approximately 1.2 seconds.

  **Shard-level parallelism:** The 12 shards are queried in parallel, meaning the per-shard binary search depth (log(500B/12) ≈ 35 steps rather than log(4.6T) ≈ 42) dominates latency.

  The "budget" is not configurable—it is determined by response length and fixed hardware characteristics. The paper does not explore how latency scales with shard count, disk speed, or response length beyond the single empirical measurement at the production configuration.

- **Cross-validation / statistical protocol.** There is no cross-validation, train/test split, or statistical significance testing reported. The 98 conversations serve as a single evaluation set for latency and relevance. For hyperparameter tuning (Table 3), the paper uses LLM-as-a-Judge on the same 98 conversations rather than a held-out set, which means the final hyperparameter configuration may be overfit to this specific conversation sample. The paper does not discuss this limitation. For the human evaluation, a single annotator was used, and no inter-annotator agreement statistics are reported—the Spearman correlation of 0.73 is between the single human's scores and the LLM judge's scores, not between multiple human annotators. This means the reliability of the human relevance judgments cannot be assessed independently.

---

### Main Quantitative Results

The paper's quantitative evaluation is organized around two axes: system latency (can it run in real time?) and document relevance (are the retrieved documents useful?). There are no accuracy or F1 metrics because OLMOTRACE does not make predictions—it retrieves matches, and the quality of those matches is measured by human (and LLM) judgments of relevance.

#### System Latency: Real-Time Performance at Trillion-Token Scale

The headline latency result is reported in Section 3.2:

> "On average, each LM response has 458 tokens, and the OLMOTRACE inference latency per query is 4.46 seconds."

This is measured on the 98-conversation sample for steps 1–3 of the pipeline (span computation through document retrieval). With an average response length of 458 tokens, the system processes approximately 103 tokens per second of latency. The paper notes this is "in line with our disk I/O analysis in App. §B," which predicts 1.2 seconds for a 100-token response (linear scaling would give 1.2 × 4.58 ≈ 5.5 seconds for 458 tokens; the actual 4.46 seconds is slightly better, likely due to caching effects or the fact that many suffixes share prefix computations in practice).

The latency measurement establishes that the system achieves its core design goal: "in real time" and "within a few seconds" (Abstract). However, the paper does not report latency distribution (minimum, maximum, variance), so it is unclear whether the 4.46-second average hides significant tail latency for unusually long responses or whether the latency is tightly clustered. The absence of percentiles or standard deviations is a notable omission for a systems paper making real-time performance claims.

**Figure 2** (the inference pipeline diagram) illustrates the five-step architecture visually but contains no quantitative data. **Figure 3** (the parallel algorithm illustration) is an explanatory diagram, not a performance plot. There is no latency-vs-response-length scatter plot or latency distribution histogram, which would be standard in a systems evaluation.

#### Document Relevance: How Useful Are the Retrieved Matches?

The document relevance evaluation is the paper's primary quality metric and is reported across multiple configurations in **Table 3**. The key results for the final configuration are:

| Metric | Final Setting (LLM Judge) |
|---|---|
| Avg score, first document | 1.82 |
| Avg score, top-5 documents | 1.50 |
| % relevant (score ≥ 2), first document | 63.3% |
| % relevant (score ≥ 2), top-5 documents | 55.1% |

Recall that the relevance scale is 0–3 where 0 = unrelated, 1 = broader topic, 2 = right topic but different context, and 3 = direct match. An average first-document score of 1.82 means the top-ranked training document is typically "on the right topic" but may be in a slightly different context or overly specific (more 2s than 3s, with some 1s pulling the average down). The 63.3% relevant rate for first documents means nearly two-thirds of the time, the very first document a user sees is at least partially relevant (scoring 2 or 3). For the top-5 documents, the 1.50 average score and 55.1% relevant rate indicate that more than half of the top-ranked documents are relevant, but there is a noticeable quality drop-off from the first to subsequent positions.

**Incremental improvement tracking (Table 3, read bottom-to-top for chronological order).** The paper's tuning process reveals which design decisions matter most:

1. **Removing the frequency-10 drop filter** (first change, row 3 in Table 3 read from bottom): Dropped first-document score from 1.73 to 1.56 but improved top-5 from 1.28 to 1.37, and improved top-5 % relevant from 47.0% to 49.4%. The mixed results suggest that frequent spans sometimes provide useful connections (improving top-5) but can also displace more relevant documents from the top position (hurting first-document score). The final decision to keep all spans regardless of frequency was a choice to prioritize recall over precision at rank 1.

2. **Switching from length ranking to unigram probability ranking** (second change, row 4): This was the single largest improvement in the tuning process. First-document score improved from 1.56 to 1.74, top-5 from 1.37 to 1.44, first-document % relevant from 57.1% to 64.3%, and top-5 % relevant from 49.4% to 52.9%. This is strong empirical validation that unigram probability—despite its simplicity—provides a substantially better signal than raw span length for identifying which matching spans will lead to relevant documents.

3. **Shortening document context from 500 to 100 tokens** (third change, but moving from row 4 to 3 in Table 3 actually increases context, so the row ordering is slightly confusing): The direction of improvement is consistent—longer context (500 tokens vs. 100 tokens) improves both first-document score (1.78 vs. 1.74) and top-5 score (1.49 vs. 1.44), though the gains are modest. The paper does not explore even larger context windows or dynamic context sizing.

4. **Adding user prompt to BM25 query** (fourth change): Improves first-document score from 1.78 to 1.82 and top-5 from 1.49 to 1.50. The small magnitude suggests that the LM response itself is the dominant relevance signal, but including the user's question provides a slight additional topical grounding.

The cumulative effect of these changes (from the early setting to the final setting, evaluated by LLM judge) is a first-document score improvement from 1.73 to 1.82 (+0.09) and a top-5 improvement from 1.28 to 1.50 (+0.22). The larger gain in top-5 versus first-document suggests that the improvements primarily help surface relevant documents that were previously buried rather than finding a better single top document.

#### Span Length Statistics

**Figure 4 (left)** reports the distribution of span lengths after Step 2 filtering (before merging). The spans have a mean length of 10.4 tokens and a median of 10 tokens. This is not presented as a quality metric per se but as a characterization of what the system finds in practice: the maximal matching spans between LM outputs and training data are non-trivially long—10 tokens is roughly a short phrase or partial sentence—rather than being dominated by 2–3 token common phrases. The paper interprets this as validation that "there are many long pieces of text shared between the LM output and its training data, which are revealed by OLMOTRACE" (Section 4).

#### Relevance Score Distribution and Threshold Calibration

**Figure 4 (middle and right)** shows the relationship between BM25 scores and the bucketing thresholds. The key empirical finding is that the maximum attainable BM25 score for a document is roughly proportional to the response length: "capped by 0.18 times the number of characters in the LM output." The paper uses this relationship to normalize scores by response length before applying fixed thresholds (≥0.7 high, 0.5–0.7 medium, <0.5 low). After normalization and bucketing, 14% of documents fall into the high-relevance category and 19% of spans into the high-relevance category. These percentages are not derived from an optimization—they reflect empirically chosen thresholds that "aligned with human expectations."

#### Training Stage Distribution of Retrieved Documents

The paper reports that 96.7% of retrieved documents come from pre-training data, 0.9% from mid-training, and 2.4% from post-training (0.9% SFT, 1.5% DPO, 0% RLVR). This is not a performance metric—it is a characterization of where verbatim matches originate—but it has practical implications. The overwhelming dominance of pre-training matches suggests that even for an instruction-tuned model like OLMo-2-32B-Instruct, most verbatim traces lead back to the massive pre-training corpus rather than to the smaller, curated post-training datasets. The paper notes this "heavily depends on the topic of the conversation" and gives the example that math-heavy outputs retrieve more from SFT and RLVR datasets—but the 0% RLVR retrieval rate across the 98-conversation sample is striking and perhaps indicates that RLVR training data is either too small (it is presumably a subset of the 1.7 million post-training documents) or too domain-specific to produce matches for the general user queries in this evaluation set.

---

### Ablation Studies and Robustness Checks

The paper's ablation studies are embedded in the incremental hyperparameter tuning tracked in **Table 3**. Each row change represents an ablation of one design choice. Here I extract them as structured findings:

**Frequency-based span dropping (dropping spans with >10 occurrences) vs. keeping all spans:** Removing the frequency filter decreased first-document relevance (score 1.73 → 1.56, first-doc % relevant 62.2% → 57.1%) but improved top-5 relevance (score 1.28 → 1.37, % relevant 47.0% → 49.4%). This is a precision-recall tradeoff: frequent spans sometimes connect to relevant documents (improving the aggregate top-5) but tend to push less discriminative content into the top rank. The final configuration keeps all spans, prioritizing the ability to surface relevant documents anywhere in the list over maximizing rank-1 precision.

**Span ranking by length vs. unigram probability:** This is the most important ablation. Ranking by length produced first-document score 1.56, while ranking by unigram probability produced 1.74 (+0.18, a 12% relative improvement). Top-5 scores improved from 1.37 to 1.44. The paper's rationale—that unigram probability penalizes long but common boilerplate phrases while rewarding long spans with distinctive vocabulary—is supported by these numbers. However, the ablation does not isolate the interaction between the ranking metric and the number of spans kept (K = ⌈0.05L⌉). It is possible that unigram probability would still outperform length at different K values, but this is not tested.

**Document context window: 100 tokens vs. 500 tokens (implicit ablation, rows 4→3):** Extending context from 100 to 500 tokens improved first-document score from 1.74 to 1.78 and top-5 from 1.44 to 1.49. The modest improvement suggests that the additional context helps the BM25 scorer make slightly better relevance distinctions, but the dominant relevance signal is already present in the 100-token window. The paper does not explore whether further increases (e.g., 1000 tokens) would yield additional gains or plateau. The choice of 500 tokens for the "View Document" extended view is a UI decision, not an evaluated parameter.

**BM25 query formulation: LM response only vs. user prompt + LM response:** Adding the user prompt to the BM25 query improved first-document score from 1.78 to 1.82 and top-5 from 1.49 to 1.50. The near-negligible improvement suggests that for the types of user interactions in the evaluation set, the LM response's text is the primary driver of topical relevance, and the prompt adds only marginal value. This is consistent with the system's design: the response is what contains the verbatim matches, and the prompt provides context about the user's intent but does not contribute additional matching text.

**Human evaluation vs. LLM-as-a-Judge:** This is not an ablation of the system but a validation of the evaluation methodology. Under the identical early hyperparameter setting, human evaluation produced first-document score 1.90 and top-5 score 1.43, while LLM-as-a-Judge produced 1.73 and 1.28. The LLM judge is systematically more conservative (lower scores). The Spearman correlation of 0.73 indicates moderate agreement—the relative ordering of documents by relevance is broadly consistent between human and LLM judge, but the absolute score calibration differs. This is adequate for hyperparameter tuning (where relative improvements matter) but should caution against interpreting the absolute scores as ground truth.

**Absent ablations.** Several design choices that would benefit from explicit ablation are not evaluated:

- **The number of difficulty/span buckets (K parameter).** The threshold K = ⌈0.05L⌉ is stated but never varied. Would K = ⌈0.10L⌉ (keeping twice as many spans) improve top-5 relevance by surfacing more matches, or degrade it by adding noise? The paper provides no sensitivity analysis.
- **The BM25 relevance thresholds (0.5 and 0.7).** These were "empirically found... to be aligned with human expectations," but the tuning process is not described. An ablation showing how different thresholds change the distribution across relevance tiers (and impact user perception, if measured) would strengthen the claim that these specific thresholds are well-calibrated.
- **The number of shards (12).** The latency is directly proportional to the number of shards (each shard is queried in parallel, but more shards means more total disk reads). An ablation showing latency vs. shard count would help users deploying OLMOTRACE on different hardware configurations understand the scaling properties.
- **The effect of disabling prefetching.** The paper states this decision without empirical evidence that it actually improves throughput. A side-by-side comparison of latency with and without prefetching at different load levels would substantiate the claim.

---

### Critical Assessment

The experiments in this paper are best understood as a **feasibility demonstration** rather than a rigorous comparative evaluation. This is not necessarily a weakness—the paper's primary contribution is the system itself, and demonstrating that it works at trillion-token scale in real time with reasonable relevance is sufficient to establish its value. However, the evaluation leaves several important questions unanswered, and the claims should be understood within the boundaries of what was actually tested.

**Claim: "The first system that traces the outputs of language models back to their full, multi-trillion-token training data in real time" (Abstract).**

This claim is supported by the latency measurement (4.46 seconds average) on the production hardware. However, "real time" is context-dependent: 4.5 seconds is fast enough for interactive exploration (a user can click, wait briefly, and explore results) but would be too slow for, say, streaming response-by-response tracing in a high-throughput chatbot deployment. The paper does not explore throughput—how many concurrent queries the system can handle, or how latency degrades under load. For a single-user interactive tool (which is what the Ai2 Playground provides), 4.5 seconds is adequate. For batch processing of thousands of responses, the throughput characteristics matter but are not reported. The claim also implies uniqueness ("the first system"), which is difficult to verify but is plausible given the paper's literature review and the absence of competing systems operating at this scale.

**Claim: Document relevance is high enough to be useful, with 1.82 average first-document score and 63.3% first-document relevance rate (Table 3).**

The absolute numbers are moderate, not strong. A 1.82 average on a 0–3 scale means the typical top-ranked document is between "broader topic" (1) and "right topic, different context" (2)—it is related to what the model said but is often not a perfect match in topic and scope. The 63.3% relevance rate means that more than one-third of the time, the first document a user sees is *not* relevant (scoring 0 or 1). For an exploratory tool, this may be acceptable—users can scan multiple documents and use the bidirectional UI to find relevant connections. But the paper's framing sometimes implies stronger relevance than the numbers support (e.g., the case studies in Figure 5 show documents that are clearly connected to the highlighted spans, which may represent the 63% of cases where relevance is good rather than the 37% where it is not).

A critical missing analysis is **per-query relevance breakdown.** The paper reports averages across 98 conversations but never shows the distribution of relevance scores per query. If some queries have excellent relevance (score 3 documents) and others have near-zero relevance (score 0–1 documents), the tool's utility is highly variable depending on what the user asks. This is plausible: queries about widely-covered topics (science facts, famous events) likely retrieve more relevant training documents than queries about obscure or idiosyncratic topics. Without a distributional analysis, users cannot know when to trust the tool's results and when to disregard them.

**Claim: The unigram probability metric outperforms span length for ranking (Table 3).**

This is the best-supported quantitative claim in the paper, with a clear 0.18-point improvement in first-document score (1.56 → 1.74). However, the ablation compares only two alternatives (length vs. unigram probability), leaving open whether even better metrics exist. For instance, a combination metric (e.g., length × unigram probability, or a metric incorporating bigram statistics for common collocations) might outperform either alone. The paper's justification for not trying bigram models (computational cost) is reasonable, but it means the claim should be "unigram probability outperforms the simple length baseline" rather than "unigram probability is the best tractable metric."

**Missing evaluation: Does OLMOTRACE help users accomplish tasks?**

The paper's case studies (Section 5) are illustrative, not evaluative. Figure 5 shows three examples where relevant training documents are surfaced, but these are cherry-picked to demonstrate the system working well. There is no user study measuring whether OLMOTRACE *improves* users' ability to fact-check, understand model creativity, or attribute capabilities, compared to not having the tool or compared to alternative approaches (e.g., a standard web search). This is understandable for a first system paper, but it means the claims about utility ("OLMOTRACE can help users understand the behavior of language models through the lens of their training data") are aspirational rather than demonstrated.

**Missing evaluation: How often do verbatim matches *not* explain behavior?**

The system only finds exact matches. An LM output that is a semantic paraphrase of training content (same meaning, different words) will produce no matches, even though the training data may be highly relevant to understanding the output's origin. The paper acknowledges this limitation ("verbatim matches... should not be interpreted as having a causal effect"), but the evaluation does not quantify how frequently verbatim matches fail to surface relevant training connections. A study comparing OLMOTRACE's retrieval against a semantic retrieval baseline (e.g., embedding-based similarity search over the training data, even if limited to a small subset due to scale) would help characterize this blind spot. Without it, users cannot know whether the absence of highlighted spans means "this output is novel and not derivable from training data" or "this output is semantically derived from training data but expressed in different words."

**Missing evaluation: Reproducibility across models and training data compositions.**

All evaluation is on OLMo models with the Ai2 training data mix. The system architecture is general—any tokenized corpus can be indexed with suffix arrays, and the algorithm is training-data-agnostic—but the *relevance quality* likely depends on properties of the training data: its size, its domain composition, its deduplication, its language distribution. A corpus with heavy deduplication might produce fewer matching spans (because duplicates are removed) but those matches might be more informative. A corpus dominated by one domain (e.g., scientific papers) would produce different relevance patterns from a web crawl corpus. The paper cannot be expected to evaluate on multiple corpora, but the absence of this analysis means the reported relevance numbers (1.82, 63.3%) should be understood as specific to the OLMo training data, not as universal characteristics of the approach.

**Missing evaluation: Stability of the hyperparameter choices across the 98-conversation sample.**

The tuning in Table 3 uses the full 98 conversations for both development and evaluation. With approximately 5 hyperparameter decisions and 98 data points, the risk of overfitting to this specific sample is non-trivial. A held-out set of conversations (or a cross-validation procedure, which would be straightforward with 98 samples) would provide a more honest estimate of generalization. The paper does not mention this risk or any attempt to mitigate it.

**What works:** The latency numbers (4.46 seconds average) are credible and represent a genuine engineering achievement—finding arbitrary substrings in 4.6 trillion tokens in seconds is not trivial. The span length statistics (mean 10.4 tokens) confirm that non-trivial verbatim overlaps exist between model outputs and training data, validating the basic premise. The incremental tuning (Table 3) shows systematic improvement from well-motivated design changes, and the 0.73 Spearman correlation between human and LLM judge is adequate for tuning purposes.

**What is missing:** Distributional analyses (latency variance, per-query relevance spread), sensitivity to hyperparameter choices (K value, relevance thresholds), comparison against any external baseline (random retrieval, BM25 on random training snippets, semantic retrieval on a subset), user studies demonstrating task-level utility, and evaluation on a held-out query set to verify generalization. These omissions are moderate for a systems paper introducing a new capability, but they mean the quantitative claims should be interpreted as feasibility bounds rather than as precise performance guarantees.

## 6. Limitations and Trade-offs

### Only Verbatim Matches — Semantic and Paraphrased Derivations Are Invisible

**The assumption or constraint.** OLMOTRACE finds *exact lexical matches* between LM outputs and training data — spans that appear character-for-character (token-for-token under the Llama-2 tokenizer). The system fundamentally cannot detect training documents that are semantically related but lexically distinct. The authors acknowledge this explicitly in the Limitations section:

> "OLMOTRACE finds lexical, verbatim matches between an LM's output and its training data. The retrieved documents should not be interpreted as having a causal effect on the LM output."

**The consequence.** This is not a minor implementation limitation — it is a **categorical blind spot**. An LM output that paraphrases training content (same meaning, different words) produces zero highlighted spans, even though the training data may be the direct source of the model's knowledge. Conversely, a verbatim match to a document does not guarantee that document caused the output — the same phrase may appear in thousands of training documents, and the model may have learned it from any of them or from their aggregate statistical pattern. The tool cannot distinguish between genuine memorization, coincidental production of a common phrase, and learning that generalizes beyond specific training examples. Users inspecting an output with no highlighted spans cannot know whether the output is genuinely novel or merely expressed in words that happened not to match any training document exactly.

**What evidence exists in the paper.** The paper provides no quantitative measurement of how frequently semantically-related-but-lexically-distinct training documents exist for LM outputs. There is no comparison against an embedding-based retrieval baseline (even on a small subset of the training data) to estimate the recall gap. The span length statistics (Figure 4, left: mean 10.4 tokens) show that when matches exist, they are non-trivial in length — but they say nothing about how often matches *fail to exist* when training data is relevant. The case studies (Figure 5) are selected to show matches, not to illustrate the absence of matches for semantically derived content.

**Mitigation status.** Not addressed. The paper accepts this as an inherent characteristic of the verbatim-matching approach and does not propose any extension toward semantic matching. The disclaimer "should not be interpreted as having a causal effect" is a caveat, not a mitigation. Future work on scaling embedding-based search to trillion-token corpora could address this, but the paper does not outline a path.

---

### Training Data Access Is a Hard Deployment Constraint

**The assumption or constraint.** OLMOTRACE requires building a suffix array index over the *complete* training data of the target model. This means the service provider must have full access to the training corpus. The paper is transparent about this:

> "OLMOTRACE can be applied to any LM as long as the service provider has access to its full training data."

The system is deployed on OLMo models precisely because their training data is open. For proprietary models (GPT-4, Claude, Gemini, Llama-3 with undisclosed data), OLMOTRACE cannot function at all — the index cannot be built.

**The consequence.** This limits OLMOTRACE's applicability to the **open-source model ecosystem**, which represents a minority of deployed LM inference traffic. The models where tracing would be *most* valuable to external researchers and auditors — large proprietary models with opaque training pipelines — are precisely the ones OLMOTRACE cannot serve. Even for semi-open models that release model weights but not training data (e.g., Llama models from Meta prior to full data disclosure), OLMOTRACE is inapplicable. The tool's value proposition — transparency into model behavior through training data inspection — is thus restricted to models that are already relatively transparent.

Furthermore, even for OLMo models, the constraint means that only Ai2 (the data owner) can deploy OLMOTRACE. A third party wanting to trace their own fine-tuned OLMo variant would need to host the full index (40TB of SSD storage, 64 vCPUs, 256GB RAM per the production setup in Appendix B), which represents a substantial infrastructure commitment beyond most academic or hobbyist budgets.

**What evidence exists in the paper.** Table 1 shows the scale of the index required: 4.6 trillion tokens, 3.2 billion documents. Appendix B describes the production hardware: 64 vCPUs, 256GB RAM, 40TB SSD with 80,000 IOPS on Google Cloud Platform. The paper does not discuss deployment cost or whether smaller-scale deployments (e.g., indexing only pre-training data, or using fewer shards) are feasible.

**Mitigation status.** Not addressed as a limitation that needs solving. The paper frames OLMOTRACE as a feature of the open model paradigm, implicitly accepting that it cannot serve proprietary models. The open-sourcing of the core system (Apache 2.0 license, Section 1) enables others with training data access to deploy their own instances, but this does not expand the set of models to which tracing can be applied.

---

### The ~4.5-Second Latency Hides Throughput Limits and Response-Length Scaling

**The assumption or constraint.** The paper reports an average latency of 4.46 seconds per response (average 458 tokens) on the 98-conversation sample (Section 3.2). This measurement captures a single-query scenario — one user submitting one response for tracing. The production system is evaluated as an **interactive single-user tool**, not as a batch processing pipeline or high-throughput service.

**The consequence.** The ~4.5-second figure does not characterize the system's behavior under load, which matters for two distinct reasons:

1. **Throughput under concurrent queries.** The disk I/O analysis (Appendix B) shows that each suffix position in the LM output requires ~960 disk reads across 12 shards. With 80,000 IOPS, processing a 458-token response consumes approximately 458 × 960 = 440,000 I/O operations. At 80,000 IOPS, that is ~5.5 seconds of dedicated disk time per query, implying that a single VM can service at most ~0.2 queries per second before latency degrades. The paper does not measure latency under concurrent load or report throughput. For batch use cases — tracing thousands of model outputs for a research study, for instance — the relevant metric is throughput (queries per hour), not single-query latency. The 4.46-second figure understates the wall-clock time for batch processing by orders of magnitude if queries are serialized.

2. **Latency scaling with response length.** The paper reports a single average latency number but does not show a latency-vs-response-length curve. The per-suffix cost is roughly constant (one FIND query + neighbor inspection), so latency should scale approximately linearly with response length L — a 900-token response would take roughly twice as long as a 450-token response. But the maximum response length tested is not reported (only the mean of 458 tokens), and the upper tail of latency — for unusually long or short responses — is invisible. A user generating a 2,000-token response could face 15–20 seconds of latency, breaking the "real time" experience.

**What evidence exists in the paper.** The disk I/O analysis in Appendix B estimates 1.2 seconds for a 100-token output, which linearly extrapolates to ~5.5 seconds for the 458-token average. The actual measured 4.46 seconds is slightly better than this linear extrapolation, suggesting some caching benefits, but this is a point estimate not a scaling curve. The absence of latency distribution statistics (variance, percentiles, min, max) is a notable gap for a systems paper making real-time claims.

**Mitigation status.** Partially addressed through the production architecture (2 VM replicas, multi-mounted disks, separate worker processes — Appendix B), which improves availability but not per-query latency. The paper does not propose or evaluate techniques for reducing latency under load (e.g., query batching across concurrent users, limiting shard count, or pre-computing results for common outputs). The 4.46-second headline number is best understood as a **best-case single-query latency** on unloaded hardware.

---

### Relevance Quality Is Moderate and Unevaluated Across Query Types

**The assumption or constraint.** The paper reports document relevance using human and LLM-as-a-Judge scores on a 0–3 scale (Table 3, Appendix C). The final configuration achieves average first-document scores of 1.82 (LLM judge) and 1.90 (human, earlier setting). These scores represent "between broader topic (1) and right topic in different context (2)" — the retrieved documents are *related* to the LM output but often are not *direct matches* in topic and scope. The percentage of documents scoring 2 or 3 ("relevant") is 63.3% for first documents and 55.1% for top-5 documents.

**The consequence.** These numbers mean that approximately **37% of the time, the first document a user sees is not relevant** (scoring 0 or 1) to the query and response. For an exploratory tool, a 63% first-document relevance rate may be acceptable — users can scroll past irrelevant results — but it fundamentally differs from the experience implied by the case studies in Figure 5, where every shown document is on-topic. The paper reports *average* relevance but never shows how relevance varies across query types. If some categories of user queries (e.g., factual questions about well-covered topics) produce highly relevant documents while others (e.g., creative writing, niche technical questions) produce very few relevant matches, the tool's utility is highly variable in ways a user cannot predict.

Furthermore, the 0–3 rubric (Table 2, left) sets a relatively low bar for "relevance": score 2 means "on the right topic... but in a slightly different context or is too specific." A document that discusses the same general subject as the user's query but in a substantively different way is counted as "relevant" under this rubric, even though it may not help a user understand why the model produced a specific output. The 55–63% relevant rates should be interpreted with this permissive rubric in mind.

**What evidence exists in the paper.** Table 3 provides aggregate relevance scores across five hyperparameter configurations. Figure 4 (middle, right) shows the distribution of BM25 scores and the bucketing thresholds, confirming that only 14% of documents and 19% of spans fall into the "high relevance" category — the majority of matches are medium or low relevance. However, there is **no per-query relevance breakdown**, no analysis of which query characteristics predict high vs. low relevance, and no user study measuring task completion rates with vs. without the tool. The 98-conversation evaluation set is described only as "from internal usage of OLMo models in the Ai2 Playground" (Section 3.2) with no characterization of its topical distribution, difficulty, or representativeness.

**Mitigation status.** Partially addressed through the color-coding system (high/medium/low relevance tiers, Section 3, Step 5), which helps users visually identify more promising documents. The bidirectional UI (click span → filter documents, click document → locate spans) enables users to pivot between views and find relevant connections even when the top-ranked document is not ideal. But these are UI mitigations for a retrieval quality problem — they make it easier to find good documents when they exist, but do not make bad documents better. The paper does not propose improving the underlying retrieval quality through better ranking models or hybrid (lexical + semantic) approaches.

---

### Difficulty Estimation Cost Is Unaccounted for in the Headline Latency

**The assumption or constraint.** The latency measurement of 4.46 seconds covers Steps 1–3 of the pipeline: finding maximal matching spans and retrieving document snippets. However, this measurement is taken on a specific hardware configuration (64 vCPUs, 256GB RAM, 40TB SSD) and for a specific average response length (458 tokens). The latency depends on both factors, and the paper provides no guidance for how latency would change under different configurations.

**The consequence.** For practitioners considering deploying OLMOTRACE, the latency cost has two hidden dimensions:

1. **Hardware cost not discussed.** The production system uses 40TB of SSD storage and a 64-vCPU VM — these are not trivial resources. The paper does not discuss the dollar cost of this infrastructure, the cost scaling if the training data grows (as it inevitably will for future model versions), or whether cheaper configurations (fewer shards, slower disks, less RAM) are viable with degraded latency. A research lab wanting to deploy OLMOTRACE for their own open model cannot determine from the paper whether this is a $500/month or $5,000/month proposition.

2. **Index construction cost not discussed.** Building the infini-gram suffix array index for 4.6 trillion tokens is itself a substantial computational undertaking. The paper describes the index as pre-built ("we build an infini-gram index on the tokenized version of the LMs' training data," Section 3.1) but never reports the time, cost, or hardware required to construct it. For any new model or updated training data, this one-time indexing cost must be paid before any tracing queries can be served. The paper's "real-time" guarantee applies only to queries against an already-built index, not to the end-to-end process of preparing the system for a new model.

**What evidence exists in the paper.** Appendix B provides hardware specifications and the disk I/O analysis, which gives a per-token cost model (960 disk reads per token position). But there is no total cost of ownership analysis, no index construction time measurement, and no scaling analysis for growing training data sizes (e.g., how does latency change if the next OLMo model is trained on 10 trillion tokens instead of 4.6 trillion?).

**Mitigation status.** The paper open-sources the core system (Apache 2.0 license) and makes OLMOTRACE available as a hosted service on the Ai2 Playground, which eliminates the deployment burden for end users. But for anyone wanting to deploy their own instance — the stated use case per "OLMOTRACE can be applied to any LM as long as the service provider has access to its full training data" — the missing cost analysis is a practical barrier. The paper does not acknowledge this as a limitation.

---

### Single Model Family, Single Domain — No Evidence of Generalization

**The assumption or constraint.** Every result in the paper is measured on OLMo models (OLMo-2-32B-Instruct and two others) using the Ai2 training data mix described in Table 1. The 98-conversation evaluation set comes from internal Ai2 Playground usage, reflecting whatever queries Ai2 users happened to pose to these specific models. There is no evaluation on other model families (e.g., Pythia, Llama open variants, or any encoder-decoder architecture), other training data compositions (monolingual, code-heavy, multilingual), or other interaction patterns (the Playground usage likely skews toward certain types of queries based on who uses Ai2's tools).

**The consequence.** Three important generalization questions are left completely unanswered:

1. **How does tracing quality depend on training data composition?** The OLMo training data is heavily English web text (pre-training is 4.6 trillion tokens of primarily web-crawled documents). A model trained primarily on code, scientific papers, or multilingual data would have different verbatim overlap patterns with its outputs. The span length distribution (mean 10.4 tokens; Figure 4, left) and document relevance (1.82 average first-document score; Table 3) may be specific to this data mixture. A code model might show longer exact matches (code snippets are often reproduced verbatim); a multilingual model might show shorter or fewer matches if the training data is more diverse.

2. **How does relevance quality vary with model capability?** OLMo-2-32B-Instruct is a capable instruction-tuned model. A weaker model might produce noisier, less coherent outputs that generate fewer meaningful training data matches. A stronger model might produce outputs that are better paraphrases of training data, actively reducing verbatim overlap and making OLMOTRACE *less* useful as models improve — an ironic possibility where better generation quality makes tracing harder.

3. **How representative are the 98 Ai2 Playground conversations?** The evaluation set is described only by its source ("internal usage") and its average response length (458 tokens). No information is provided about the topical distribution, the types of prompts, or whether the conversations represent typical or unusual usage. If Ai2 Playground users are predominantly AI researchers asking technical questions, the document relevance scores may not generalize to a general-public user base asking about news, entertainment, health, or other common topics.

**What evidence exists in the paper.** None for cross-model or cross-domain generalization. The paper acknowledges its scope is limited to OLMo models with open training data, but does not frame this as a limitation to be addressed, only as a design parameter. There is no discussion of how the approach might need to change for different model architectures, training data regimes, or deployment contexts.

**Mitigation status.** Not addressed. The paper positions itself as a system for the OLMo ecosystem specifically, which is a reasonable scope for a first deployment. But the abstract's claim of a general system ("OLMOTRACE can be applied to any LM as long as the service provider has access to its full training data") implies broader applicability than the evaluation supports. A practitioner considering deploying OLMOTRACE for a non-OLMo model would need to replicate the latency and relevance evaluation from scratch, without guidance from the paper on what to expect. The authors do not suggest a systematic evaluation protocol for new model-training data combinations.

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
