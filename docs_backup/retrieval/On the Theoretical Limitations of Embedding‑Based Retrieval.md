# On the Theoretical Limitations of Embedding‑Based Retrieval

**ArXiv:** [2508.21038](https://arxiv.org/abs/2508.21038)
**Authors:** Orion Weller, Michael Boratko, Iftekhar Naim, Jinhyuk Lee
**Institutions:** Google DeepMind

## 🎯 Pitch

This paper establishes a fundamental limit of single-vector embedding retrieval systems by linking embedding dimension constraints to sign-rank, proving that certain query-document relevance patterns can't be realized within fixed dimensions. This insight is crucial as it highlights the hidden capacity limitations affecting instruction-following systems, challenging the scalability of dense retrieval models and prompting a reevaluation of current embedding architectures in handling complex queries.

---

## 1. Executive Summary
This paper proves and demonstrates a fundamental limit of single‑vector embedding retrieval: with a fixed embedding dimension `d`, there exist legitimate query–document relevance patterns that no dot‑product embedding model can realize. It connects retrieval to the mathematical notion of sign‑rank to bound the minimum dimension needed, validates the bound in best‑case optimization, and releases a simple natural‑language dataset (LIMIT) where state‑of‑the‑art embedders fail despite the task’s trivial semantics.

## 2. Context and Motivation
- Problem addressed
  - Modern retrieval systems often use single vectors for queries and documents and select top‑k documents by dot‑product similarity (“dense retrieval”). Benchmarks now require handling arbitrary instructions, reasoning, and logical combinations, implicitly asking embedders to represent any “set of relevant documents” a query might define (§1).
  - The question: Can a finite‑dimensional single‑vector embedding represent all possible top‑k relevance combinations users might request?

- Why it matters
  - Practical: Instruction‑following and agentic search can synthesize complex, hyper‑specific queries that implicitly pick arbitrary combinations of documents. If embeddings cannot represent certain combinations, some user intents are unretrievable no matter how we train (§1, §5.1).
  - Theoretical: The paper ties representational limits of vector spaces to classic results in communication complexity via sign‑rank, making limits precise rather than heuristic (§3).

- Prior approaches and gaps
  - Empirical studies have hinted at dimensionality issues (e.g., higher false positives with lower‑dimensional embeddings in large corpora; §2.2).
  - Geometric bounds from order‑k Voronoi regions conceptually relate to top‑k retrieval but are hard to compute tightly in high dimensions and offer little actionable guidance for IR (§2.3, Appendix §8).
  - No prior work links the exact realizability of top‑k sets in embedding retrieval to a formal lower bound on dimension via sign‑rank and then shows the effect empirically.

- Positioning
  - The work (1) formalizes retrieval as a matrix order/threshold preservation problem, (2) proves tight dimension bounds using sign‑rank (within ±1), (3) verifies best‑case realizability by directly optimizing free embeddings on the test qrels, and (4) provides a natural‑language dataset (LIMIT) that operationalizes the theory and exposes failures of current embedders (§§3–5).

## 3. Technical Approach
The paper proceeds in three layers: a formal theory, a best‑case optimization test, and a realistic dataset that instantiates the theoretical difficulty.

- Formalization of retrieval as matrix ordering (Section §3.1)
  - Setup:
    - `m` queries, `n` documents, and a binary relevance matrix `A ∈ {0,1}^{m×n}` where `A[i,j]=1` iff document `j` is relevant to query `i`.
    - An embedding model maps queries and documents to `d`‑dimensional vectors. Scores are dot products. Let `U ∈ R^{d×m}`, `V ∈ R^{d×n}`, and `B = U^T V` be the score matrix.
  - Goal: For each row (query), ensure all relevant docs score above all irrelevant docs (or at least appear before them in order).

- Paper‑specific notions (defined and used to tie retrieval to linear algebra; §3.1)
  - `row‑wise order‑preserving rank` (`rank_rop A`): smallest rank of a score matrix `B` that preserves the within‑row ordering given by `A`.
  - `row‑wise thresholdable rank` (`rank_rt A`): smallest rank enabling a per‑row threshold `τ_i` that separates relevant from irrelevant entries in row `i`.
  - `globally thresholdable rank` (`rank_gt A`): smallest rank enabling a single global threshold `τ` that separates all 1s from 0s in `A`.

- Key equivalence and bridge to sign‑rank (Section §3.2)
  - Equivalence for binary matrices:
    - Proposition 1 shows `rank_rop A = rank_rt A`. Intuition: if in each row all relevant scores are above all irrelevant scores (order‑preserving), there exists a separating threshold per row, and vice‑versa.
  - Connection to sign‑rank:
    - Sign‑rank (Definition 3) of a ±1 matrix `M` is the minimum rank of a real matrix whose entrywise signs match `M`.
    - Construct `M = 2A − 1_{m×n} ∈ {−1, 1}^{m×n}` (map relevant to +1, irrelevant to −1).
    - Proposition 2 provides the core chain of inequalities:
      > rank±(2A−1) − 1 ≤ rank_rt(A) = rank_rop(A) ≤ rank_gt(A) ≤ rank±(2A−1)
    - Plain‑language meaning:
      - The minimum dimension needed by a dot‑product embedder to realize the relevance constraints (with per‑row thresholds or order preservation) is sandwiched within one of the sign‑rank of `2A−1`.
      - Therefore, if you know the sign‑rank, you know the minimum embedding dimension up to ±1. Conversely, if you can realize `A` in `d` dimensions (e.g., by gradient‑descent optimizing free vectors), you bound the sign‑rank to either `d` or `d+1` (§3.3).

- Consequences (Section §3.3)
  - Lower bound on required dimensionality:
    > “we need at least rank±(2A − 1) − 1 dimensions to capture the relationships in A exactly” (§3.3).
  - Practical mechanism:
    - If free embeddings can realize `A` in `d` dims, then the sign‑rank is ≤ `d+1`. This yields a constructive, optimization‑based upper bound on sign‑rank—useful because sign‑rank is hard to compute exactly.

- Best‑case “free embedding” optimization (Section §4)
  - Idea: Remove all language‑modeling constraints. Directly treat each query/document as its own learnable vector (“free embeddings”) and optimize them against the target qrels with full‑batch contrastive loss.
  - Setup:
    - Build a toy world where for `n` documents and `k=2`, include all “choose‑2” query sets, i.e., `m = C(n,2)` queries, each requiring two specific documents to be top‑2.
    - Optimize query/document vectors (unit‑normalized after each update) using Adam and full in‑batch negatives with InfoNCE (§4; footnote 7). Early stop if no loss improvement for 1,000 steps.
    - Increase `n` gradually for a fixed dimension `d` until the optimizer can no longer reach 100% accuracy. The largest solvable `n` at that `d` is the “critical‑n” point.
  - Why this matters: If even this unconstrained, test‑set‑optimized procedure fails beyond a certain `n` for dimension `d`, then real embedders (which must encode language, generalize, and use finite data) cannot hope to realize those qrels at that dimension.

- Resulting empirical law (Figure 2; Table 6)
  - The critical‑n vs `d` curve fits a cubic polynomial:
    > “y = −10.5322 + 4.0309d + 0.0520d^2 + 0.0037d^3 (r^2=0.999)” (Figure 2).
  - Extrapolation yields best‑case limits for k=2:
    > “critical‑n values (for embedding size): 500k (512), 1.7m (768), 4m (1024), 107m (3072), 250m (4096).” (Figure 2 caption text and §4 Results)
  - Interpretation: Even in the friendliest setting, finite `d` caps how many distinct top‑2 combinations can be realized.

- LIMIT dataset: a natural‑language instantiation (Section §5.2; Figure 1)
  - Mapping idea:
    - Queries: “Who likes X?” where `X` is an attribute (e.g., “Apples”, “Quokkas”), keeping the query language trivial.
    - Documents: Short profiles like “Jon Durben likes Quokkas and Apples.” Each query has exactly two relevant documents (k=2).
  - Construction details:
    - 1,850 attribute types curated by iterative de‑duplication and overlap checks (§5.2).
    - Choose the largest `n` such that “n choose 2” slightly exceeds 1,000; that is `n=46` (since `C(46,2)=1035`) so every pair of these 46 documents appears as a relevant set to one query (§5.2).
    - Two settings:
      - LIMIT‑small: just these 46 documents and the 1,000 queries built from all choose‑2 pairs.
      - LIMIT‑full: a 50k‑document corpus where only those 46 are ever relevant; the rest are realistic distractors (§5.2).
  - Why this is hard (and realistic):
    - The difficulty stems from how many distinct “pairs” across the same 46 items must be representable—high “qrel graph density.” Table 1 shows LIMIT’s Graph Density is 0.085 and Average Query Strength is 28.47, orders of magnitude higher than common IR test sets.

## 4. Key Insights and Innovations
- A tight theoretical bound linking embedding dimension and sign‑rank (Fundamental)
  - Proposition 2 pins the minimum dimension to realize a binary relevance pattern to the sign‑rank of a derived ±1 matrix, within ±1. This is stronger and more operational than prior geometric analogies (order‑k Voronoi) that lack tight, computable bounds in high dimensions (§§3.2–3.3, Appendix §8).

- “Free embeddings” as a constructive tool to probe realizability (Methodological)
  - By directly optimizing vectors on the target qrels, the study offers a practical way to estimate whether a pattern is representable in `d` dimensions and to bound its sign‑rank from above (to `d+1`). This bypasses the intractability of exact sign‑rank computation (§3.3, §4).

- An empirical law for capacity collapse: the critical‑n curve (Empirical insight)
  - The cubic fit between capacity (`n`) and dimension (`d`) in the k=2 setting quantifies where even best‑case embeddings will start failing (Figure 2; Table 6). This turns a qualitative intuition into a usable predictive model.

- LIMIT: a deceptively simple, natural‑language benchmark that stress‑tests combination capacity (Benchmark contribution)
  - LIMIT shows that instruction‑following embedders struggle not with reasoning or linguistic nuance, but with the sheer number of top‑k combinations required for even simple “Who likes X?” queries. Results reveal large gaps versus sparse and multi‑vector systems (§5.2–§5.6; Figures 3–6; Tables 3–5).

- Diagnosing the root cause: qrel density matters more than domain shift (Diagnostic)
  - Training on in‑domain LIMIT‑train barely helps, while training on LIMIT‑test allows overfitting to near‑perfect scores (Table 2; Figure 5). Ablations across qrel patterns show that the “dense” pattern (maximizing combinations) is uniquely hard (Figure 6; Table 3).

## 5. Experimental Analysis
- Evaluation setup (Sections §4–§5)
  - Metrics: Recall@k (e.g., Recall@2, @10, @20, @100) using the MTEB evaluation framework (§9).
  - Baselines and systems (Figure 3; Table 5):
    - Single‑vector embedders spanning 1,024–4,096 dims: `GritLM 7B`, `Qwen3 Embedding`, `Promptriever Llama3 8B`, `Gemini Embedding`, `Snowflake Arctic 2.0`, `E5‑Mistral`.
    - Alternatives: `BM25` (sparse lexical), `GTE‑ModernColBERT` (multi‑vector).
    - Embedding dimension truncation is tested, including via Matryoshka Representation Learning (MRL) where available; stars in figures denote models trained with MRL (§5.2, Figure 3 caption).
  - LIMIT datasets:
    - LIMIT‑full: 50k documents, 1k queries, k=2 (§5.2).
    - LIMIT‑small: 46 documents (every pair queried), 1k queries, k=2 (§5.2).
  - Free‑embedding optimization:
    - All‑combinations setup with k=2; increment `n` until failure at 100% accuracy; InfoNCE loss; full‑batch negatives; normalized vectors; Adam; early stopping (§4).

- Main quantitative findings
  - Capacity limit law (Figure 2; Table 6)
    > “y = −10.5322 + 4.0309d + 0.0520d^2 + 0.0037d^3 (r^2=0.999).”  
    > “critical‑n … 500k (512), 1.7m (768), 4m (1024), 107m (3072), 250m (4096).”
    - This is the best‑case capacity when the model can directly optimize query and document vectors for the test qrels.

  - LIMIT‑full results (Figure 3; Table 5)
    - Single‑vector embedders struggle to even surface the two relevant docs among 50k:
      > `Promptriever Llama3 8B` at 4096 dims: Recall@100 = 18.9; `GritLM 7B` at 4096 dims: Recall@100 = 12.9; `E5‑Mistral 7B` at 4096 dims: Recall@100 = 8.3 (Table 5).
      > The caption emphasizes “models perform poorly, scoring less than 20 recall@100” (Figure 1).
    - Alternatives fare better:
      > `GTE‑ModernColBERT`: Recall@100 = 54.8; `BM25`: Recall@100 = 93.6 (Table 5).
    - Trend: Performance improves with dimension but remains far from acceptable at scale (Figure 3 shows monotonic gains with dimension for most embedders).

  - LIMIT‑small results (46 docs; Figure 4; Table 4)
    - Even with tiny `n=46`, single‑vector embedders cannot perfectly realize all 1,035 choose‑2 combinations:
      > `Promptriever Llama3 8B` at 4096 dims: Recall@2 = 54.3; Recall@20 = 97.7 (Table 4).
      > `GTE‑ModernColBERT`: Recall@2 = 83.5; Recall@20 = 99.1 (Table 4).
      > `BM25`: achieves 100.0 Recall@10 and Recall@20 (Table 4).
    - Takeaway: The difficulty is not only massive corpus size; even small, dense combination spaces stress single‑vector capacity.

  - Domain‑shift check via finetuning (Figure 5; Table 2)
    - Training a modern embedder on LIMIT‑train hardly helps:
      > Best Recall@10 on `Train` split is 2.8 (1024 dims), while most settings are <1.0 (Table 2).
    - Overfitting on LIMIT‑test succeeds (as expected if capacity exists for that exact matrix):
      > On `Test` split, Recall@10 > 98 for embedding dims as small as 32 (Table 2).
    - Interpretation: The failure is not due to vocabulary/domain mismatch; it is the combination density that matters (§5.3).

  - Qrel‑pattern ablations (Figure 6; Table 3)
    - When the 1k queries are sampled to form “Random”, “Cycle”, or “Disjoint” patterns, scores rise substantially vs “Dense” (maximizing unique pairs):
      > `E5‑Mistral 7B` (4096 dims): Recall@100 = 40.4 (Random) vs 4.8 (Dense).  
      > `GritLM 7B` (4096 dims): 61.8 (Random) vs 10.4 (Dense).  
      > `Promptriever` (4096 dims): 62.0 (Random) vs 19.4 (Dense). (Table 3)
    - Conclusion: The dominant difficulty is the number of distinct top‑k combinations that must be realized, not linguistic complexity.

  - Dataset density metrics (Appendix §10; Table 1)
    > LIMIT Graph Density = 0.0855 and Average Query Strength = 28.47, while standard IR sets have near‑zero density/strength (e.g., HotpotQA density 0.000037; average strength 0.1104).
    - This quantifies why LIMIT stresses embedding capacity: many queries share and recombine document pairs.

  - Cross‑encoder sanity check (Section §5.6)
    - A long‑context reranker (`Gemini‑2.5‑Pro`) can solve LIMIT‑small perfectly when given all 46 docs and all 1,000 queries in a single pass:
      > “successfully solve (100%) all 1000 queries in one forward pass” (§5.6).
    - This supports that the task itself is trivial semantically; it is the single‑vector constraint that bites.

  - BEIR vs LIMIT (Figure 7; Table 7)
    > `Qwen3 Embedding`: BEIR = 62.76 vs LIMIT R@100 = 4.8;  
    > `Promptriever`: BEIR = 56.40 vs LIMIT R@100 = 18.9.  
    - There is no clear correlation, indicating that standard IR benchmarks do not expose this capacity limit (§5.5).

- Do the experiments support the claims?
  - Yes. The theory predicts capacity limits governed by dimension; free‑embedding runs expose the limit as a tight empirical curve; LIMIT demonstrates the functional consequence with natural language; density ablations and domain‑shift tests isolate the cause to combination density, not language.

## 6. Limitations and Trade-offs
- Scope of theory
  - The proofs target single‑vector models with dot‑product scoring and binary relevance; they do not cover multi‑vector architectures (e.g., ColBERT’s MaxSim) or cross‑encoders (§Limitations).
  - The results address exact separability/ordering; approximate retrieval with tolerated errors is not bounded here (cited as future work; see also Ben‑David et al. 2002).

- Computing sign‑rank
  - Exact sign‑rank is difficult to compute; the paper instead brackets it via Proposition 2 and a constructive upper bound from free‑embedding realizability (§3.3). This gives guidance but not closed‑form answers for arbitrary qrels.

- Dataset construction choices
  - LIMIT is synthetic in structure (likes‑attributes mapping) though expressed in fluent natural language (§5.2). While this isolates the combination phenomenon, it does not measure other retrieval skills (e.g., multi‑hop reasoning beyond set membership).

- Computational trade‑offs
  - Cross‑encoders solve LIMIT‑small but are too expensive for first‑stage retrieval at web scale (§5.6). Multi‑vector and sparse models perform better but come with indexing and storage costs, and unclear transfer to instruction‑following reasoning tasks (§5.6).

- What is not addressed
  - Which specific combinations fail for a given model/dimension remains uncharacterized; the theory certifies the existence of failures but does not enumerate them (§Limitations).
  - Triangle inequality–based arguments do not apply for cosine similarity (non‑metric), so they cannot provide alternative bounds (§7).

## 7. Implications and Future Directions
- How this changes the landscape
  - Embedding dimension is not merely a performance knob; it is a hard capacity bound on the set of top‑k combinations a single‑vector retriever can realize. As benchmarks move toward instruction‑following and compositional querying, these bounds will be hit unless architectures evolve (§1, §5.6).

- Practical guidance
  - For systems that must handle many recombinations of a small set (e.g., catalog filters, attribute queries, facet combinations), single‑vector embedders may be insufficient even with large `d`. Consider:
    - Multi‑vector retrievers (e.g., ColBERT variants), which showed large gains on LIMIT (Table 5).
    - Sparse retrieval or hybrid dense+sparse; `BM25` nearly solved LIMIT‑full (Table 5).
    - Cross‑encoder reranking as a second stage, which trivially solved LIMIT‑small (§5.6).
  - When designing evaluations, include dense combination patterns; metrics like qrel Graph Density and Average Query Strength (Appendix §10; Table 1) help quantify stress on embedding capacity.

- Research directions
  - Theory: Extend sign‑rank–style bounds to multi‑vector architectures and to approximate retrieval where some errors are allowed (§Limitations).
  - Learning: Develop training strategies that better allocate dimensional capacity across many independent combination axes, or adaptive per‑query subspaces.
  - Indexing: Explore hybrid and learned sparse methods that retain high effective dimensionality while supporting instruction following.
  - Benchmarking: Build more datasets in the style of LIMIT that vary qrel density systematically (Figure 6), to chart capability frontiers and drive architectural innovation.

> Core takeaway: A single dot‑product vector has finite “combination bandwidth.” Proposition 2 makes that precise via sign‑rank; the free‑embedding curve shows where capacity collapses; and LIMIT proves that even today’s strongest embedders fail on a conceptually trivial—but combination‑dense—task.
