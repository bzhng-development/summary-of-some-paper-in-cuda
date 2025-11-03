# In-Context Pretraining: Language Modeling Beyond Document Boundaries

**ArXiv:** [2310.10638](https://arxiv.org/abs/2310.10638)
**Authors:** Weijia Shi, Sewon Min, Maria Lomeli, Chunting Zhou, Margaret Li, Xi Victoria Lin, Noah A. Smith, Luke Zettlemoyer, Scott Yih, Mike Lewis
**Institutions:** 

## 🎯 Pitch

In-Context Pretraining (ICLM) revolutionizes language model training by smartly ordering semantically related documents, improving cross-document reasoning and yielding significant performance boosts in reading comprehension and retrieval-augmented QA. This method elevates long-context learning without changing core model objectives or architecture, making it a scalable solution with broad implications for tasks requiring complex contextual understanding across various domains.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces In-Context Pretraining (ICLM), a way to pretrain language models by concatenating semantically related documents—rather than random ones—within each training context window. By only changing how documents are ordered (everything else in the training recipe stays the same), the method improves reasoning across document boundaries and yields sizable gains on in-context learning, reading comprehension, long-context reasoning, and retrieval-augmented QA (e.g., +14–15% average gains on reading comprehension; Table 2).

## 2. Context and Motivation
- Problem/gap
  - Standard large language model (LLM) pretraining forms long inputs by randomly concatenating short documents until the context limit is reached (Section 2, Figure 1). In this setup, earlier documents in a context often provide no useful signal for predicting tokens in later documents, wasting computation and failing to teach models how to reason across documents.
  - LLMs often struggle with complex contextual understanding: following instructions precisely, reasoning over long or multi-document contexts, and being faithful to provided context rather than over-relying on memorized knowledge (Introduction, citing McKenzie et al., 2023; Liu et al., 2023; Shi et al., 2023a).

- Why it matters
  - Many real-world tasks naturally span multiple related documents (news, Wikipedia pages with links, research paper sections, code repositories). Teaching models to leverage cross-document information during pretraining should better prepare them for in-context learning, retrieval-augmented generation, and long-context reading comprehension.

- Prior approaches and limitations
  - Random concatenation: easy and scalable, but earlier documents seldom help predict later ones (Section 1).
  - Retrieval-augmented pretraining-by-kNN: directly prepend a document’s top-k neighbors to it during training (Section 2.2). This can overexpose popular documents and repeat data across contexts, reducing diversity and leading to overfitting (the “data repeating” problem).
  - Small-scale or metadata-dependent multi-document pretraining (Related Work, Section 5): e.g., using hyperlinks/citations (LinkBERT; Yasunaga et al., 2022) or human-curated collections. These do not scale universally to web-scale corpora.

- Positioning
  - ICLM keeps the usual language modeling objective intact but changes the document order to ensure each training context is filled with related documents, without repeating documents across contexts (Sections 2, 2.2). It provides a general, scalable pipeline to find related documents at web scale and to sort them into coherent, non-overlapping sequences.

## 3. Technical Approach
ICLM has two main steps (Figure 2): (1) find related documents at scale, then (2) construct fixed-length input contexts by traversing a document graph so that each document appears exactly once.

Step 1: Finding related documents at scale (Section 2.1)
- Goal: For each document `di` in corpus `D`, retrieve a set of nearest neighbors `N(di)` that are semantically similar.
- Embeddings and similarity
  - Use the `contriever` dense retrieval model to embed documents (first 512 tokens per document) into vectors (Izacard et al., 2022).
  - Similarity is cosine similarity:
    - Equation (1): s(di, dj) = cos(E(di), E(dj))
- Scalable nearest-neighbor search
  - Use FAISS with product quantization (PQ) and an inverted file (IVF) index, plus large-batch offline search (“OIVFBBS”; Section 2.1 and Appendix A.2).
  - Practical setup:
    - Top-10 neighbors per document (k=10).
    - 235M documents, 306B tokens total (Section 3.1).
    - Index: IVFPQ with 32,768 inverted lists, 62 GB index size; trained on 1.57M vectors (Appendix A.2).
    - Search is done in 50M-vector batches on 32 GPUs, taking ~6 hours. Document graph traversal then takes ~12 hours on 20 CPUs (Section 3.1).
- Semantic deduplication
  - The corpus contains many near-duplicates. High-similarity pairs are removed, because keeping duplicates in the same context encourages copying and destabilizes training (Section 2.1; Appendix A.1).
  - Ablation shows dedup is crucial: removing it worsens perplexity (Figure 5, “No dedup” PPL 8.3 vs “Dedup” 7.3).

Step 2: Creating input contexts via document graph traversal (Section 2.2)
- Build a document graph
  - Nodes are documents in `D`. Add an undirected edge between `di` and `dj` if either is in the other’s top-k neighbor list; weight = cosine similarity s(di, dj) (Equation 1).
- Objective: order all documents into a single path so that adjacent documents are as similar as possible, then split the path into fixed-size context windows (e.g., 8,192 tokens per context; Figure 2).
  - Formulated as a maximum Traveling Salesman Problem (TSP): find a path visiting each node once with maximal sum of edge weights (Section 2.2). Exact solution is intractable at this scale (NP-hard), so use a greedy approximation.
- Greedy traversal algorithm (Algorithm 1; Section 2.2)
  - Start node: pick an unvisited document with minimum degree (fewest edges) to reduce the chance it gets forced to connect to a poor neighbor later.
  - Extension rule: repeatedly move to the unvisited neighbor with the highest similarity (Algorithm 1 text says “highest weight,” and Figure 2 illustrates this).
  - If stuck (no unvisited neighbors): “teleport” to another unvisited minimum-degree document via a zero-weight edge and continue.
  - Continue until all documents are placed on a single path `P`. Then concatenate documents along `P` and split into fixed-length contexts. Batch sampler ensures diversity across different contexts within the same batch (final paragraph, Section 2.2).
- Why this design vs alternatives?
  - Simple kNN concatenation repeats documents across many contexts, reducing diversity and potentially overfitting to popular documents (Section 2.2).
  - Graph traversal enforces a global constraint: each document appears exactly once, while still maximizing local similarity between consecutive documents.

Pretraining setup (Section 3.1)
- Models: LLaMA-like architectures with 0.3B, 0.7B, 1.5B, and 7B parameters; 8,192-token context windows; FlashAttention for memory efficiency.
- Data: 235M English CommonCrawl documents, 306B tokens total; same data for all models and baselines.
- Optimization: AdamW with β1=0.9, β2=0.95; cosine LR schedule.
- Hardware/time: The 7B model is trained on 128 A100 GPUs over 9 days with 4M tokens/batch (Section 3.1).

Baselines (Section 3.2)
- `Standard`: Random document concatenation (current common practice).
- `kNN`: For each document, directly append its top-k retrieved neighbors to form a context. This causes document repetition across contexts.

## 4. Key Insights and Innovations
- Changing only the ordering of training documents can produce broad, meaningful gains.
  - Significance: No change to the language modeling objective, architecture, or loss—just smarter batching—yet improvements appear across LM perplexity, in-context learning, reading comprehension, long-context reasoning, and retrieval-augmented QA (Sections 3.3.1–3.3.6; Figures 3–6; Tables 1–5).

- A scalable, web-scale pipeline for building coherent training contexts
  - Novelty: Combines dense retrieval (Contriever), FAISS big-batch approximate nearest-neighbor search, semantic deduplication, and a maximum-TSP-inspired greedy traversal to form a single, diversity-preserving, coherent ordering of billions of tokens (Sections 2.1–2.2; Appendix A.2).
  - Why it matters: Prior “related-document” pretraining often relied on metadata (e.g., hyperlinks, dates) or small curated corpora (Section 5). This pipeline works at web scale without special metadata.

- Avoiding the “data repeating” pitfall of naive retrieval-augmented pretraining
  - The traversal ensures each document is used once while maximizing adjacency similarity. This reduces overfitting to popular documents and improves generalization over kNN-style concatenation (Section 2.2; kNN baseline results in Tables 1–5).

- Empirical insight: higher intra-context relevance and deduplication both reduce language modeling perplexity
  - Figure 5 shows a monotonic trend: PPL improves from Random (8.2) → Clustering (7.9) → Link-based ICLM (7.3); and dedup is necessary (8.3 without vs 7.3 with dedup). This isolates the contributions of relevance and dedup, beyond end-task results.

## 5. Experimental Analysis
Evaluation methodology (Section 3)
- Datasets and metrics
  - Language modeling: Wikipedia, arXiv, and Books; measure perplexity (Figure 3).
  - In-context learning: Seven text classification datasets (Amazon, SST-2, Yelp, Hate, Offensive, AGNews, DBPedia) with 32 in-context examples; accuracy (Table 1).
  - Reading comprehension: RACE-High, RACE-Middle, BoolQ, SQuAD, HotpotQA, DROP; 2-shot prompting; EM for SQuAD/HotpotQA, accuracy for others (Table 2).
  - Retrieval-augmented QA: Natural Questions (NQ) and TriviaQA (TQA); closed-book vs open-book (prepend top-10 Wikipedia docs); EM (Table 3).
  - Factuality under knowledge conflict: NQ-Swap, MemoTrap; EM (Table 4).
  - Long-context reasoning: SCROLL benchmark (NarrativeQA, Qasper, ContractNLI, QMSum, GovReport); F1 or ROUGE-1 after fine-tuning on task training sets (Table 5).
- Baselines
  - `Standard` random concatenation and `kNN` retrieval-augmented pretraining, trained on the same data with the same compute (Section 3.2).

Main quantitative results
- Language modeling (Figure 3)
  - ICLM has lower perplexity than both baselines across datasets and model sizes, despite evaluation using randomly ordered documents (not sorted to favor ICLM). This indicates better general language modeling, not just a reliance on sorted contexts.
- In-context learning (Table 1)
  - Average accuracy: ICLM 71.3 vs Standard 66.0 vs kNN 61.8 (+8% average over Standard).
  - ICLM wins on all seven datasets; e.g., SST-2: 93.2 (ICLM) vs 83.7 (Standard); AGNews: 76.0 vs 68.3.
- Reading comprehension (Table 2)
  - Average: ICLM 43.2 vs Standard 37.6 vs kNN 36.0 (+14–15% over Standard).
  - Notable gains on HotpotQA (multi-hop): 21.9 vs 10.5 EM.
- Retrieval-augmented QA (Table 3)
  - Open-book: ICLM improves substantially: NQ 32.2 vs 28.5; TQA 51.6 vs 48.1 (+9% absolute average over baselines discussed in Abstract/Introduction; see Table 3).
  - Closed-book: ICLM is equal on NQ (17.0 vs 17.0) and slightly lower on TQA (48.0 vs 49.3), suggesting less reliance on memorized knowledge and more on contextual reasoning (Section 3.3.4).
- Factuality under conflict (Table 4)
  - ICLM performs best on both NQ-Swap and MemoTrap: 45.8 vs 39.6 (Standard) on NQ-Swap; 56.2 vs 48.4 (Standard) on MemoTrap. This indicates improved faithfulness to provided context over parametric memory (Section 3.3.5).
- Long-context reasoning (SCROLL; Table 5)
  - Average: ICLM 34.1 vs Standard 32.5 vs kNN 32.3 (~+5% relative). Gains persist despite subsequent task-specific fine-tuning.
- Training dynamics (Figure 4)
  - ICLM shows consistently lower training loss than Standard, and after ~150B tokens maintains higher EM on RACE-High/Middle throughout the remainder of pretraining (Figure 4 b,c).
- Ablations and robustness
  - Document relevance and dedup (Figure 5) confirm both design choices materially contribute to lower perplexity.
  - Number of in-context examples (Figure 6): ICLM’s advantage over Standard persists as examples increase; performance plateaus after 32 examples.
  - Near-duplicate removal is critical to avoid trivial copying and instability (Appendix A.1; Figure 5).

Assessment of evidence
- The evaluation is broad and aligns with the claim that coherent, related-context pretraining improves cross-document reasoning:
  - Strong gains on tasks requiring synthesizing context (HotpotQA, DROP, SCROLL).
  - Clear benefit when additional relevant text is provided at inference (open-book QA).
  - Improvements appear across scales (Figure 3), and learning curves show early and sustained gains (Figure 4).
- The main mixed result is in closed-book QA (Table 3), where ICLM does not outperform Standard, consistent with the idea that it emphasizes context use over memorization.

## 6. Limitations and Trade-offs
- Retrieval quality and coverage
  - Only the first 512 tokens per document are embedded for retrieval (Section 3.1), which can miss relevant content later in long documents.
  - Top-10 neighbors (k=10) and cosine similarity in a fixed embedding space (Contriever) may not capture all forms of relatedness (e.g., causal, temporal, or argumentative relations).
- Approximate sorting heuristic
  - The maximum-TSP-inspired greedy traversal is not optimal; it may create suboptimal paths in regions of the graph, especially where the graph is sparse or noisy (Section 2.2).
- Compute and preprocessing costs
  - Although a one-time cost, the pipeline adds nontrivial preprocessing: 6 hours of GPU time for search and 12 hours CPU for traversal at this scale (Section 3.1). Scaling to trillions of tokens will increase these costs.
- Data and domain coverage
  - Experiments use English CommonCrawl only (Section 3.1). It is unclear how well the pipeline transfers to multilingual corpora or domain-specific corpora like code or biomedical text without adapting the retriever.
- Potential memorization–context trade-off
  - Closed-book QA shows no gain or a slight drop (Table 3). Emphasizing cross-document context may reduce rote memorization, which can hurt tasks that rely on parametric knowledge without extra context.
- Interaction with downstream fine-tuning
  - Gains on long-context tasks persist but may be modest after fine-tuning (Table 5), suggesting some benefits can be attenuated by heavy task-specific adaptation (Section 3.3.6).

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes “long-context training” from merely increasing window size to increasing the meaningfulness of what fills the window. Simply reordering training data can teach LLMs to use context more effectively without changing architectures or objectives (Sections 1–2).
- Follow-up research enabled
  - Better relatedness functions: task-aware or structure-aware retrieval (link graphs, citations, code dependency graphs). The paper hints at domain-specific structure like code repositories (Conclusion).
  - Multilingual extensions: use multilingual retrievers to group cross-lingual equivalents (Conclusion).
  - Dynamic or curriculum-based ordering: adapt the traversal during training, or refresh neighborhoods as the model’s representation space evolves.
  - Segment-level or hierarchical methods: move beyond document-level embeddings to paragraph/sentence granularity to capture fine-grained relations while preserving the “no repetition” constraint.
  - Jointly learned retriever: co-train the retriever with the LM so that neighborhoods reflect the LM’s evolving notions of usefulness.
- Practical applications
  - Retrieval-augmented systems: ICLM-pretrained models show larger gains when given retrieved passages at inference (Table 3), making them strong backbones for RAG pipelines.
  - Multi-document question answering and summarization: Improvements on HotpotQA and SCROLL (Tables 2 and 5) point to better synthesis across documents.
  - Instruction following and in-context learning: Consistent gains across seven classification tasks with 32-shot prompts (Table 1), and robustness as the number of demonstrations grows (Figure 6).

> Key takeaway: By turning random document concatenation into a principled, similarity-aware, non-repeating ordering problem (Sections 2.1–2.2), In-Context Pretraining makes every token in a long context more likely to help predict the next ones—and that simple change propagates into broad, measurable gains across multiple families of tasks.
