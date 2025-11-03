# On the Theoretical Limitations of Embedding-Based Retrieval

**ArXiv:** [2508.21038](https://arxiv.org/abs/2508.21038)

## 🎯 Pitch

This paper rigorously proves and empirically verifies that single-vector embedding retrieval models have a fundamental limitation: no matter how they are trained, for any fixed embedding dimension, there exist realistic query–top‑k relevance patterns they simply cannot represent or retrieve exactly. By connecting this limit to the sign-rank of the relevance matrix and introducing the LIMIT dataset—a simple, natural-language benchmark on which state-of-the-art dense retrievers fail sharply—the authors provide a decisive bridge between learning theory and practical IR, highlighting the urgent need for new architectures as retrieval tasks diversify beyond what conventional embeddings can capture. This work matters because it exposes irreducible recall failures that will persist even as models get larger, guiding both researchers and practitioners in designing retrieval systems robust to the increasingly complex demands of modern search and AI agents.

---

## 1. Executive Summary
This paper proves and empirically demonstrates a fundamental capacity limit of single‑vector embedding retrieval: for any fixed embedding dimension `d`, there exist realistic top‑k retrieval targets that no scoring by dot‑product in `R^d` can realize exactly. It connects this limit to the sign‑rank of the relevance matrix, verifies it with “best‑case” optimization where embeddings are directly learned on the test labels, and releases LIMIT, a simple natural‑language dataset that stresses these limits and causes sharp failures in state‑of‑the‑art (SoTA) dense retrievers.

## 2. Context and Motivation
- Problem addressed
  - Modern retrieval increasingly asks dense embedding models to handle arbitrary notions of relevance, including instruction‑following, reasoning, and logical combinations (“X or Y”) of previously unrelated documents (Introduction; §1). The paper asks: is a single fixed‑dimensional vector per query/document expressive enough to represent all such top‑k combinations?

- Why this matters
  - Practical: As systems broaden to open‑ended queries and instructions, the space of possible top‑k sets explodes (e.g., QUEST has 325k docs and k=20; the number of possible top‑20 sets is ~7.1e+91, §5.1). If embeddings cannot represent many such combinations, user queries will systematically fail even with large models.
  - Theoretical: While earlier works note geometric limits (e.g., Voronoi order‑k regions), tight, actionable bounds for IR are elusive (§2.3, §8). This paper supplies a concrete, learnability‑style bound using sign‑rank from communication complexity.

- Prior approaches and gaps
  - Dense single‑vector retrieval dominates practice but is known to struggle in high‑cardinality settings (e.g., false positives increase with larger corpora; Reimers & Gurevych 2020, §2.2).
  - Theoretical tools like order‑k Voronoi region counts are hard to compute and not practical for IR (§2.3, §8).
  - There was no clear bridge from theory to an empirical stress test that is simple yet not solvable by stronger training or bigger models.

- Positioning
  - The paper provides: (i) a formal lower bound linking the representability of a retrieval target to the sign‑rank of the binary relevance matrix (§3); (ii) “free embedding” experiments that optimize embeddings directly on the test labels (no language modeling constraints) to establish best‑case feasibility (§4); and (iii) LIMIT, a natural‑language dataset engineered to maximize the number of required top‑k combinations while keeping queries and docs trivial (§5.2).

## 3. Technical Approach
Step 1 — Formalization of retrieval as a matrix ordering problem (§3.1)
- Setup
  - Let there be `m` queries and `n` documents, and a binary relevance matrix `A ∈ {0,1}^{m×n}` where `A[i,j]=1` iff document `j` is relevant to query `i`.
  - A single‑vector embedding model maps each query `i` to `u_i ∈ R^d` and each document `j` to `v_j ∈ R^d`. Scores are dot products; stacking gives `B = U^T V` where `B[i,j] = u_i · v_j`.
- Retrieval goal as row‑wise ordering/thresholding
  - The model should score all relevant docs ahead of irrelevant ones for each query.
  - Two equivalent formulations are introduced (Proposition 1, §3.2):
    - Row‑wise order‑preserving rank `rank_rop(A)`: minimal rank of a real matrix `B` that preserves the per‑row ordering in `A`.
    - Row‑wise thresholdable rank `rank_rt(A)`: minimal rank of `B` for which a per‑row threshold separates relevant from irrelevant entries.
  - A stricter variant uses one global threshold `τ` for all rows (global thresholdable rank `rank_gt(A)`).

Step 2 — Connecting to sign‑rank (§3.2–§3.3)
- Key definition (Definition 3): For a sign matrix `M ∈ {−1,1}^{m×n}`, the sign‑rank `rank_±(M)` is the minimum rank of a real matrix whose entry signs match `M`.
- Central inequality chain (Proposition 2):
  - Let `J` be an all‑ones matrix of shape `m×n`. Then:
    > rank_±(2A − J) − 1 ≤ rank_rop(A) = rank_rt(A) ≤ rank_gt(A) ≤ rank_±(2A − J)
  - Intuition:
    - Convert binary `A` to a sign matrix `2A − J` (relevant becomes +1, irrelevant −1).
    - If a `B` can separate relevant/irrelevant per row (row‑wise thresholding), then after shifting by the row thresholds (i.e., `B − τ 1^T`), the sign pattern matches `2A − J`, so the needed rank is at least sign‑rank minus one (§3.2, proof step 3).
- Consequences (§3.3)
  - Lower bound: any embedding in `R^d` must have `d ≥ rank_±(2A − J) − 1` to exactly realize `A`’s per‑row separation.
  - Existence of hard tasks: There are `A` with arbitrarily large sign‑rank, so for any fixed `d`, some retrieval targets cannot be represented exactly by single‑vector embeddings.
  - Practical diagnostic: If one can find embeddings that realize the ordering in `d` dimensions (by optimization), that also upper‑bounds the sign‑rank.

Step 3 — “Free embeddings” as best‑case feasibility test (§4)
- Idea
  - Remove language modeling constraints and directly optimize a learned vector per query and per document to fit the desired relevance (`A`) using contrastive loss (InfoNCE).
  - If even this unconstrained optimization fails for a given `(n, k, d)`, real embedding models will be at least as limited.
- Experimental setup (all in §4)
  - Generate unit‑norm document vectors for `n` docs; create all top‑`k` combinations as queries (`m = binom(n, k)`), with `k=2`.
  - Optimize query and document vectors jointly with Adam on the full set of positives per query and all other docs as negatives (full‑dataset batch); re‑normalize vectors each step; early stop after no loss improvement for 1000 iterations.
  - Increase `n` for a fixed `d` until 100% training accuracy becomes impossible; the last solvable `n` is the “critical‑n” for that `d`.
- Results summary
  - The measured critical‑n grows as a cubic polynomial in `d`: Figure 2 fits `y = −10.5322 + 4.0309 d + 0.0520 d^2 + 0.0037 d^3` with `r^2=0.999`.
  - Extrapolated “best‑case” upper limits for k=2 combos: critical `n ≈ 500k (d=512), 1.7M (768), 4M (1024), 107M (3072), 250M (4096)` (Figure 2, §4).

Step 4 — LIMIT: a simple natural‑language instantiation (§5.2)
- Goal
  - Create a dataset where queries are trivial (“Who likes X?”) but the induced qrels require many distinct top‑k combinations, stressing the sign‑rank bound in a realistic form.
- Construction
  - Attributes: collect ~1850 “things a person could like” (cleaned list; §5.2).
  - Choose `k=2` relevant documents per query and select a document set that maximizes the number of distinct 2‑way combinations just over 1000:
    > use `n=46` documents since `binom(46,2)=1035` (Figure 1; §5.2).
  - Full vs small versions:
    - LIMIT full: 50k documents, 1000 queries. Only 46 docs are ever relevant; the rest are distractors (non‑relevant to any query).
    - LIMIT small: only those 46 documents (easier to evaluate re‑ranking behavior; Figure 4).
  - Naturalization: assign random names to documents and populate each with a fixed number of liked attributes; queries ask “Who likes X?”; relevant docs are those whose attribute list includes X (Figure 1).

## 4. Key Insights and Innovations
- Theoretical lower bound via sign‑rank for single‑vector retrieval (§3.2–§3.3)
  - Novelty: A clean inequality chain ties the minimal embedding dimension needed to exactly realize a binary relevance pattern to the matrix’s sign‑rank. This reframes dense retrieval expressivity as a matrix factorization problem with known hardness properties.
  - Significance: It proves the existence of realistic top‑k sets that cannot be represented for any fixed `d`, even in principle, when using dot‑product scoring with one vector per query/document.

- Best‑case empirical “free embedding” methodology (§4)
  - Novelty: Instead of testing trained language encoders, the work directly optimizes per‑item vectors on the test `A` with full‑batch InfoNCE. This isolates geometric capacity limits from modeling or training data issues.
  - Significance: The resulting critical‑n vs. `d` curve (Figure 2; Table 6) sets an empirical bar that real models cannot surpass; any gap between these curves and model performance is due to practical constraints, not capacity in `R^d` itself.

- LIMIT dataset: simple language, maximally demanding qrels (§5.2)
  - Novelty: A natural‑language dataset that is trivial to read and query but deliberately dense in distinct top‑k combinations, achieved by choosing `n` and `k` to maximize `binom(n,k)` within 1000 queries (Figure 1).
  - Significance: State‑of‑the‑art dense embedders fail sharply on LIMIT, showing that the theoretical limit surfaces in realistic settings, not just contrived math.

- Empirical demonstration that “combinational density” of qrels drives difficulty (§5.4)
  - Novelty: An ablation that instantiates four qrel connectivity patterns—random, cycle, disjoint, and dense—and shows the dense pattern (max combinations) is far harder (Figure 6; Table 3).
  - Significance: It isolates the root cause: not language, domain shift, or negatives, but the number of distinct top‑k sets the model must represent.

## 5. Experimental Analysis
- Evaluation protocol
  - Metrics: Recall@2/10/20/100 depending on dataset size (Figures 3–4; Tables 4–5).
  - Models: Multiple SoTA dense retrievers varying in embedding dimension (1024–4096), training style (instruction‑tuned vs. others), and MRL truncation; plus non‑single‑vector baselines (BM25; ModernColBERT) (§5.2).
  - Ablations:
    - Domain shift: fine‑tune a modern encoder on LIMIT‑train vs LIMIT‑test (§5.3; Figure 5; Table 2).
    - Qrel pattern: random, cycle, disjoint, dense (§5.4; Figure 6; Table 3).
    - Cross‑benchmark correlation: LIMIT vs BEIR (§5.5; Figure 7; Table 7).

- Main results on LIMIT full (50k docs; Figure 3; Table 5)
  - Dense single‑vector models perform poorly despite large dimensions:
    - Example Recall@100 (full dimension, unless noted): `E5‑Mistral‑7B`: 8.3; `GritLM‑7B`: 12.9; `Promptriever‑Llama3‑8B`: 18.9; `Qwen3‑Embed`: 4.8; `Gemini‑Embed`: 10.0 (Table 5).
    - The same models often have Recall@2 below 2–3% on LIMIT full (Table 5).
  - Alternatives fare better:
    - BM25 (sparse lexical) reaches 93.6 Recall@100 (Table 5).
    - `GTE‑ModernColBERT` (multi‑vector late interaction) achieves 54.8 Recall@100 (Table 5).
  - Interpretation: As embedding dimension grows, performance increases somewhat (Figure 3), but remains far from satisfactory; sparse and multi‑vector methods exploit higher effective dimensionality or more flexible matching.

- LIMIT small (46 docs; Figure 4; Table 4)
  - Even with only 46 docs, dense retrievers do not solve the task at top‑20:
    - Example Recall@20 at max dim: `E5‑Mistral‑7B`: 85.2; `GritLM‑7B`: 90.5; `Promptriever‑Llama3‑8B`: 97.7; `Qwen3‑Embed`: 73.8; `Gemini‑Embed`: 87.9 (Table 4).
  - BM25 is near perfect (97.8/100.0/100.0 for R@2/10/20, Table 4). `GTE‑ModernColBERT` is also very strong (R@20=99.1).
  - Takeaway: Single‑vector dense models still leave substantial errors even in tiny corpora when many distinct top‑k sets must be represented.

- Domain shift ablation (§5.3; Figure 5; Table 2)
  - Training on LIMIT‑train barely helps:
    > At 1024 dims, training on train reaches only 1.0/2.8/11.2 for R@2/10/100 (Table 2).
  - Training on LIMIT‑test (overfitting) nearly solves it across dims:
    > At 32 dims, test‑trained model attains 85.5/98.4/100.0 for R@2/10/100 (Table 2).
  - Conclusion: The difficulty is not domain shift; models can memorize when allowed to overfit specific tokens, echoing the free‑embedding results that feasibility depends on fitting the exact `A`.

- Qrel‑pattern ablation (§5.4; Figure 6; Table 3)
  - Dense pattern (max combinations) is decisively hardest:
    - `GritLM‑7B` Recall@100 drops from 61.8 (random) to 10.4 (dense); `E5‑Mistral‑7B` from 40.4 to 4.8; `Promptriever` from 62.0 to 19.4 (Table 3).
  - Random/cycle/disjoint are all noticeably easier and fairly similar to each other.

- Cross‑benchmark comparison (§5.5; Figure 7; Table 7)
  - BEIR performance does not predict LIMIT performance:
    > e.g., `Qwen3‑Embed` scores 62.76 on BEIR but 4.8 Recall@100 on LIMIT (Table 7).
  - Implication: LIMIT reveals a distinct capability not measured by standard retrieval benchmarks.

- Reranking sanity check (§5.6)
  - A long‑context cross‑encoder reranker (`Gemini‑2.5‑Pro`) can solve LIMIT‑small perfectly by scoring all 46 docs at once (§5.6).
  - This validates that the task itself is simple—the error source is the single‑vector constraint, not semantic understanding.

- Support for claims
  - The empirical findings align with the theory: performance collapses as the dataset requires many distinct top‑k combinations, and increases with effective dimensionality (BM25, multi‑vector; Figures 3–4, 6).
  - The free‑embedding upper bounds confirm that even perfect optimization in `R^d` hits limits at finite `n` (Figure 2; Table 6).

## 6. Limitations and Trade-offs
- Scope: Single‑vector, dot‑product retrieval
  - The formal bounds and most experiments target one‑vector‑per‑sequence with dot‑product scoring. Multi‑vector methods and cross‑encoders are not covered by the theory, though they are explored empirically (§5.6).
- Exact realization vs. approximate performance
  - The sign‑rank connection concerns exact separation (row‑wise thresholding/order). The paper does not provide formal bounds for approximate retrieval (e.g., “get most but not all combinations”), though it cites learning‑theory directions for future work (Limitations section; referencing Ben‑David et al. 2002).
- Computing sign‑rank
  - Sign‑rank is hard to compute in practice. The paper relies on the inequality and the free‑embedding upper‑bound heuristic rather than computing sign‑rank of specific datasets (§2.3, §3.3).
- Dataset design choices
  - LIMIT is engineered to create many distinct top‑k combinations using simple content. While this isolates the capacity issue, it is not a comprehensive benchmark for all kinds of instruction‑following retrieval.
- Scalability of alternatives
  - Cross‑encoders solve the small version but are computationally expensive for first‑stage retrieval at web scale (§5.6).
  - Multi‑vector models improve results but have their own trade‑offs (index size, latency) and are less explored for instruction‑following (§5.6).

## 7. Implications and Future Directions
- Field impact
  - Recalibrates expectations for dense single‑vector retrieval: it cannot realize all top‑k combinations for arbitrary instructions at fixed `d`. As instruction‑following and reasoning‑based retrieval expand, systems must plan around this ceiling (§1, §6).
  - Evaluation design: Benchmarks with few queries miss these limits; LIMIT shows how small, high‑combination datasets can stress real capacity (§5.1–§5.2, Figure 1).

- Research directions
  - Theory extensions:
    - Approximate representation: formalize bounds when small fractions of combinations can be wrong (Limitations section).
    - Multi‑vector theory: extend sign‑rank‑style arguments to MaxSim late‑interaction models (§5.6).
    - Links to geometric constructs beyond Voronoi that are tractable for `d≫3` (§8).
  - Modeling advances:
    - Hybrids: use sparse or multi‑vector first‑stage retrieval to increase effective dimensionality, then re‑rank with cross‑encoders (§5.6).
    - Adaptive dimensionality: dynamically allocate more vectors or dimensions for queries likely to induce many combinations.
    - Instruction‑aware indexing: store multiple sub‑embeddings per document keyed to attribute/operator types to hedge against combination blow‑ups.
  - Benchmarking:
    - Systematic control of qrel “graph density” (Appendix §10) to map performance vs. combination complexity.
    - Broader instruction‑following datasets that vary `binom(n,k)` regimes and include range/logic operators (connections to BrowseComp and QUEST; §5.1).

- Practical applications
  - Production search/QA systems should incorporate:
    - Multi‑stage retrieval (e.g., BM25/ColBERT first stage, cross‑encoder rerank).
    - Query analysis to detect high‑combination cases and switch to higher‑capacity routes.
    - Caching or precomputation for frequent combination patterns.
  - Risk management: Recognize that adding arbitrary logical operators (“X and Y or Z”) can silently exceed single‑vector capacity even when language appears simple.

> Core takeaway: “We need at least rank_±(2A − 1) − 1 dimensions to capture the relationships in A exactly” (§3.3). LIMIT shows that this lower bound is not merely theoretical—simple, realistic queries can push single‑vector dense retrievers past their representational limits (Figures 3–4, 6), while even best‑case free embeddings hit hard thresholds that grow only polynomially with `d` (Figure 2; Table 6).
