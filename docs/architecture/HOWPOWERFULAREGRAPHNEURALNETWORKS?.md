# HOW POWERFUL ARE GRAPH NEURAL NETWORKS?

**ArXiv:** [1810.00826](https://arxiv.org/abs/1810.00826)

## 🎯 Pitch

This paper develops a rigorous theoretical framework to precisely characterize the expressive power of message-passing Graph Neural Networks (GNNs), proving they can only be as powerful as the Weisfeiler–Lehman (WL) graph isomorphism test in distinguishing graph structures. By deriving conditions for maximum expressiveness and introducing the Graph Isomorphism Network (GIN) architecture that meets this bound, the work delivers both profound theoretical insights and practical advances—clarifying the limits of popular GNNs and enabling state-of-the-art performance in graph classification tasks across diverse domains.

---

## 1. Executive Summary (2–3 sentences)
This paper builds a formal theory for the expressive power of graph neural networks (GNNs) that use neighborhood aggregation (message passing). It proves that such GNNs are at most as powerful as the 1-dimensional Weisfeiler–Lehman (WL) graph isomorphism test, derives exact conditions to match WL’s power, and introduces a simple architecture—`Graph Isomorphism Network (GIN)`—that achieves this bound while delivering state-of-the-art graph classification accuracy (see Sections 3–5; Eq. 4.1; Table 1).

## 2. Context and Motivation
- Problem addressed
  - Message-passing GNNs iteratively update node representations by aggregating neighbors, but there has been little theory about what structures they can distinguish and what they fundamentally cannot (Section 1).
  - Without such understanding, new GNN designs are driven largely by heuristics, making it unclear when and why some variants fail on simple graphs (Section 1).

- Why it matters
  - Theoretical significance: Knowing the representational limits of GNNs clarifies which graph distinctions are learnable and which are provably impossible within the message-passing paradigm.
  - Practical impact: Graph tasks in chemistry, social networks, biology, and finance rely on structural distinctions. If a GNN cannot differentiate certain structures, it can underfit even with abundant data (Sections 1, 5; Figure 4).

- Prior approaches and their shortfalls
  - Popular GNNs such as `GCN` (mean aggregator) and `GraphSAGE` (max/mean aggregators) achieve strong empirical results but sometimes fail to distinguish basic graph patterns (Eq. 2.2, 2.3; Section 5.2; Figures 2–3).
  - WL kernels (e.g., WL subtree kernel) are powerful but use discrete hashing; they do not learn continuous embeddings that capture similarities between substructures (Section 4, “benefit beyond distinguishing different graphs”).

- Positioning of this work
  - Establishes a general, model-agnostic framework to analyze expressiveness of any neighborhood-aggregation GNN via its ability to form injective functions over multisets of neighbor features (Sections 3–4).
  - Provides constructive conditions under which a GNN matches the WL test’s discriminative power (Theorem 3, Section 4).
  - Designs `GIN` to meet these conditions using sum aggregation plus MLPs (Eq. 4.1), and empirically validates the theory across nine benchmarks (Sections 4.1, 7; Table 1; Figure 4).

## 3. Technical Approach
This section builds the analysis step-by-step and then introduces the `GIN` architecture that realizes the theoretical conditions.

- Preliminaries: how message-passing GNNs work
  - Generic layer (Eq. 2.1): at layer `k`, each node `v` aggregates neighbor embeddings `{h_u^(k-1): u ∈ N(v)}` using an aggregation function and then combines it with its own previous embedding `h_v^(k-1)`:
    - `a_v^(k) = AGGREGATE^(k)({h_u^(k-1) : u ∈ N(v)})`
    - `h_v^(k) = COMBINE^(k)(h_v^(k-1), a_v^(k))`
  - Graph-level output uses a permutation-invariant `READOUT` over node embeddings (Eq. 2.4).

- Key concept: neighbors as a multiset
  - A `multiset` allows repeated elements—critical because multiple neighbors can share identical features (Definition 1, Section 3).
  - The core question: does the aggregator map different multisets to different embeddings? If yes (injective), the GNN preserves structural distinctions needed to be as powerful as WL (Section 3).

- Benchmark for expressiveness: the WL test
  - The 1D WL test repeatedly relabels nodes by hashing the current label with the multiset of neighbor labels. It distinguishes many non-isomorphic graphs efficiently (Section 2, “Weisfeiler-Lehman test”).
  - Alignment with GNNs: both do iterative neighbor-based updates; WL’s strength comes from an injective multiset update (hashing) (Figure 1).

- Upper bound: message-passing GNNs cannot beat WL
  - Lemma 2 (Section 4): if a GNN maps two graphs to different embeddings, WL would also distinguish them. Hence, aggregation-based GNNs are at most as powerful as WL.

- Conditions to match WL: injectivity at node and graph levels
  - Theorem 3 (Section 4): a GNN can match WL’s power if:
    - Node update is of the form `h_v^(k) = φ(h_v^(k-1), f({h_u^(k-1)}))` where both `f` (a function on multisets) and `φ` are injective; and
    - Graph-level `READOUT` is injective on the multiset of node embeddings `{h_v^(K)}`.
  - Intuition: injectivity ensures that different neighborhoods (multisets) never collapse to the same embedding, mirroring WL’s unique relabeling.

- How to build injective multiset functions (“deep multisets”)
  - Lemma 5 (Section 4.1): with countable input feature space (assumption formalized in Lemma 4), there exists a function `f` such that `Σ_x∈X f(x)` is unique for every multiset `X` of bounded size; moreover, any multiset function `g(X)` can be written as `φ(Σ f(x))` for some `φ`.
    - Plain-language summary: sum can serve as a universal, injective summary of a multiset if combined with a powerful function approximator (`MLP`) to learn `f` and `φ`.
  - Corollary 6 (Section 4.1): for node updates, using `(1+ε)*f(c) + Σ_x∈X f(x)` yields injectivity for infinitely many `ε` (including all irrational numbers). This makes the “self node” distinguishable from its neighbors.

- From theory to architecture: Graph Isomorphism Network (GIN)
  - GIN layer (Eq. 4.1): 
    > `h_v^(k) = MLP^(k)( (1 + ε^(k)) * h_v^(k-1) + Σ_{u∈N(v)} h_u^(k-1) )`
  - Design choices:
    - `Sum` aggregator to realize injective multiset encoding (Lemma 5).
    - Multiply self-embedding by `(1+ε)` to respect Corollary 6; `ε` may be learned (`GIN-ε`) or fixed to 0 (`GIN-0`).
    - Use `MLPs` (not 1-layer linear maps) to ensure universal approximation of the needed injective functions (Section 5.1; Lemma 7).
  - Graph-level readout across layers (Eq. 4.2):
    > `h_G = CONCAT( READOUT( { h_v^(k) | v ∈ G } ) for k = 0..K )`
    - Rationale: earlier layers capture local subtrees; deeper layers capture larger neighborhoods. Concatenating across depths preserves multi-scale structure (Section 4.2).

- Why not mean or max?
  - Figure 2 ranks aggregators by expressiveness: `sum > mean > max`.
    - `sum` captures the full multiset (counts included).
    - `mean` only captures proportions (the distribution).
    - `max` collapses multiplicities and only keeps the underlying set.
  - Concrete failure cases (Figure 3, Section 5.2):
    - Two different neighborhoods with repeated colors (features) can look identical to `mean` or `max`, but `sum` separates them because counts differ.

- Why not 1-layer perceptrons?
  - Lemma 7 (Section 5.1): there exist different multisets that any 1-layer perceptron (linear map + ReLU) maps to the same aggregated result, i.e., they cannot provide universal multiset injectivity. Hence, GIN uses `MLPs`.

## 4. Key Insights and Innovations
- A unifying expressiveness framework for message-passing GNNs (fundamental)
  - Novelty: Formalizes GNN neighborhood aggregation as functions over multisets and ties GNN expressiveness tightly to the 1D WL test (Sections 3–4; Lemma 2; Theorem 3).
  - Significance: Provides clear necessary/sufficient-like conditions (injective aggregation and readout) for WL-level power.

- Deep multisets: sum + MLP is universal and injective for multisets (fundamental)
  - Novelty: Extends Deep Sets insights to multisets with multiplicities (Lemma 5) and to node-plus-neighborhood with `(1+ε)` (Corollary 6).
  - Significance: Explains why `sum` is strictly more expressive than `mean` and `max`, and when `mean/max` can be appropriate (Section 5.3–5.4).

- The `GIN` architecture that attains WL expressiveness in practice (applied innovation)
  - Novelty: A minimal design—`sum` aggregator, `(1+ε)` self-weighting, `MLP`—that matches WL’s discriminative power (Eq. 4.1; Theorem 3).
  - Significance: Achieves near-perfect training fit where others underfit (Figure 4) and state-of-the-art test accuracy on many benchmarks (Table 1).

- Precise characterization of common aggregators (clarifying insight)
  - `mean` learns distributions of neighbor features (Corollary 8), not counts.
  - `max` learns the underlying set of distinct neighbor features (Corollary 9), ignoring multiplicity.
  - Significance: Explains when `GCN`/`GraphSAGE` can work well (e.g., rich, non-repeating node features) and when they systematically fail (graphs with repeated features; unlabeled graphs) (Section 5; Figure 3).

## 5. Experimental Analysis
- Evaluation setup
  - Datasets (Section 7; Table 1):
    - Social networks: `IMDB-BINARY`, `IMDB-MULTI`, `REDDIT-BINARY`, `REDDIT-MULTI5K`, `COLLAB`.
      - For Reddit: all nodes share the same feature (intentionally uninformative), stressing structure learning (Section “Datasets”).
      - For other social graphs: degree one-hot encodings as node features.
    - Bioinformatics: `MUTAG`, `PTC`, `PROTEINS`, `NCI1` with categorical features.
  - Models compared (Sections 4–5, 7):
    - `GIN-ε` and `GIN-0` (Eq. 4.1; Eq. 4.2).
    - Less-powerful GNN variants: `SUM–1-LAYER`, `MEAN–MLP`, `MEAN–1-LAYER (GCN)`, `MAX–MLP`, `MAX–1-LAYER (GraphSAGE)`.
    - Non-GNN baselines: `WL subtree kernel`, `DCNN`, `PATCHY-SAN`, `DGCNN`, `AWL`.
  - Protocol: 10-fold cross-validation with fixed GNN depth (5 layers), 2-layer MLPs, batch norm, Adam optimizer; shared hyperparameter search where applicable (Section 7).
  - Readout choice: sum readout for bio datasets; mean readout for social datasets for better generalization (Section 7).

- Main quantitative results
  - Training fit (Figure 4):
    > GIN-ε and GIN-0 “almost perfectly fit all the training sets,” while `mean/max` variants and `1-layer` variants “severely underfit on many datasets.”
    - Training accuracy never exceeds the WL subtree kernel, consistent with Lemma 2 (upper bound).
  - Test accuracy highlights (Table 1):
    - Social graphs:
      - `REDDIT-BINARY`: `GIN-0 = 92.4 ± 2.5%`; `MEAN–MLP = 50.0 ± 0.0%`; `GCN = 50.0 ± 0.0%`.
      - `REDDIT-MULTI5K`: `GIN-0 = 57.5 ± 1.5%`; `MEAN–MLP = 20.0 ± 0.0%`; `GCN = 20.0 ± 0.0%`.
        - These 50%/20% numbers correspond to random guessing for 2/5 classes—matching the theoretical failure of `mean` on unlabeled graphs with repeated features (Section 5.2).
      - `IMDB-BINARY`: `GIN-0 = 75.1 ± 5.1%`, competitive with best baselines (`AWL 74.5 ± 5.9%`).
      - `COLLAB`: `GIN-0 = 80.2 ± 1.9%`, outperforming WL subtree kernel `78.9 ± 1.9%`.
    - Bioinformatics:
      - `PTC`: `GIN-0 = 64.6 ± 7.0%`, outperforming WL subtree `59.9 ± 4.3%` and DGCNN `58.6%`.
      - `PROTEINS`: `GIN-0 = 76.2 ± 2.8%`, comparable to top GNN variants.
      - `MUTAG`: `GIN-0 = 89.4 ± 5.6%`, close to WL subtree `90.4 ± 5.7%` and `PATCHY-SAN 92.6 ± 4.2%` (the best here; Table 1 marks significantly better baselines with an asterisk).
  - `GIN-0` vs `GIN-ε`:
    > “GIN-0 slightly but consistently outperforms GIN-ε” on test accuracy, despite both fitting training data equally well (Section 7.1; Table 1).

- Ablations and diagnostics
  - Aggregator effect:
    - `sum`-based models consistently outperform `mean/max` on datasets where neighbor multiplicity matters or features are repetitive/uninformative (Reddit).
    - `mean` can be competitive when node features are diverse and rarely repeat (Section 5.3).
  - Depth and readout:
    - Using concatenated readouts over layers (Eq. 4.2) helps capture multi-scale subtrees (Section 4.2).
  - Computational notes:
    - `max`-pooling runs for some large social graphs were omitted due to GPU memory constraints (footnote in Section 7).

- Do the experiments support the claims?
  - Yes, convincingly:
    - The training curves substantiate the theoretical ranking of expressiveness: `GIN > sum–1-layer > mean/max variants` (Figure 4; Sections 5.1–5.2).
    - The catastrophic failures of `mean` on Reddit (50% and 20% accuracy) match the formal characterization that `mean` collapses multiplicities and cannot distinguish unlabeled structure (Figure 2; Corollary 8).
    - GIN’s strong performance across datasets, especially structural ones, aligns with WL-level expressiveness (Theorem 3; Eq. 4.1).

## 6. Limitations and Trade-offs
- Message-passing ceiling (structural limit)
  - All results are within the standard neighborhood-aggregation (message-passing) framework; expressiveness is upper-bounded by the 1D WL test (Lemma 2). Graphs that WL cannot distinguish (e.g., some regular graphs) also cannot be separated by any such GNN.

- Assumption on feature space
  - Key injectivity proofs assume node features come from a `countable` space and multisets have bounded size (Lemma 4; Lemma 5). Real-valued continuous inputs are common; while MLPs can approximate functions, the exact injectivity guarantees rely on this assumption.

- Aggregator choices and task fit
  - `sum` is most expressive but may not always be ideal:
    - When the task cares about distributions or representative elements rather than counts, `mean` or `max` may be preferable (Sections 5.3–5.4).
    - The paper does not study robustness to noise or degree scaling; `sum` can amplify large-degree neighborhoods (not analyzed here).

- Optimization vs. theory
  - Theorem 3 asserts existence of injective functions and WL-level expressiveness “with sufficient layers,” but it does not guarantee easy optimization or generalization. The paper explicitly leaves generalization properties and optimization landscape to future work (Conclusion, Section 8).

- Computational aspects
  - While GIN is simple, the paper does not provide detailed runtime/memory analysis. Some variants (e.g., max-pooling on large graphs) faced GPU memory limits (Section 7).

## 7. Implications and Future Directions
- How this changes the field
  - Provides a principled target for GNN design: use injective multiset functions (sum + MLP, with explicit self-weighting) and injective readouts to reach WL-level expressiveness (Sections 4–4.2).
  - Clarifies when popular models (GCN/GraphSAGE) will fail—enabling informed model selection based on the nature of node features and required structural distinctions (Sections 5.2–5.4; Figures 2–3).

- Enabled follow-up research
  - Beyond message passing: motivates exploring architectures that surpass 1D WL, e.g., higher-order GNNs, subgraph-based models, or mechanisms that encode non-local structure (Conclusion, Section 8).
  - Multiset learning: further theoretical characterization of permutation-invariant functions over multisets with continuous inputs; analyzing attention and other weighted aggregations within this framework (Section 5.5).
  - Generalization theory: understanding when WL-level expressiveness translates to better out-of-sample performance and how to regularize injective aggregators (Conclusion, Section 8).

- Practical applications
  - Molecular property prediction and drug design: distinguishing subtle structural motifs requires WL-level expressiveness; GIN’s architecture is directly applicable.
  - Social and information networks: in settings with weak or repetitive node features (e.g., Reddit datasets), expressive aggregators are crucial to avoid collapse to random guessing (Table 1).
  - Any graph classification or node classification task where multiplicities of neighbor features carry signal (sum > mean/max).

> Key equations and figures:
> - Message-passing template: Eq. 2.1; GCN example: Eq. 2.3; Graph readout: Eq. 2.4.
> - WL/GNN connection and multiset framing: Figure 1; Definition 1; Lemma 2; Theorem 3.
> - Deep multiset universality and GIN update: Lemma 5; Corollary 6; Eq. 4.1; multi-layer readout Eq. 4.2.
> - Aggregator expressiveness and failure cases: Figures 2–3; Section 5.2–5.4.
> - Empirical evidence: training curves Figure 4; cross-dataset test results Table 1.
