# IN-CONTEXT PRETRAINING: LANGUAGE MODELING BEYOND DOCUMENT BOUNDARIES

**ArXiv:** [2310.10638](https://arxiv.org/abs/2310.10638)

## 🎯 Pitch

This paper introduces In-Context Pretraining (ICLM), a novel approach that improves large language models by reorganizing pretraining data so that each context window contains a sequence of semantically related documents instead of random ones. This simple change in document ordering (with no modification to model architecture or loss) significantly enhances the models' ability to reason across multiple documents, leading to large gains in tasks like reading comprehension, in-context learning, and retrieval-augmented generation—crucial for advanced, knowledge-intensive applications.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces in-context pretraining (ICLM), a way to pretrain language models by ordering the training corpus so that each training context contains multiple semantically related documents rather than random, unrelated ones. By only changing document ordering (not the loss, model, or tokenizer), the method consistently improves tasks that require reasoning over long or multi-document contexts, including in-context learning, reading comprehension, retrieval-augmented QA, and long-context synthesis (see §3.3 and Tables 1–5).

## 2. Context and Motivation
- Problem/gap
  - Standard pretraining creates long input sequences by randomly concatenating documents until the context window is full. The earlier documents in the window usually have no predictive value for the next document’s tokens, so the model wastes compute attending to irrelevant text and does not learn to connect information across document boundaries (§1).
  - Many downstream tasks depend on multi-document reasoning: open-domain question answering with retrieved passages, multi-hop reading comprehension, or long-context synthesis (§1).
- Why this matters
  - Practical impact: Better cross-document reasoning boosts performance for search-assisted assistants, evidence-grounded generation, and long-context tasks (§1, §3.3.3–§3.3.6).
  - Theoretical significance: It tests the hypothesis that next-token prediction benefits from exposing the model to coherent, related context beyond single-document scope (§1).
- Prior approaches and shortcomings
  - Random concatenation (the de facto standard) introduces little to no useful cross-document signal (§1).
  - Retrieval-augmented pretraining that packs each document with its top-k neighbors (`kNN`) can repeat popular documents across many contexts, reducing corpus diversity and risking overfitting (§2.2).
  - Link- or metadata-based grouping (e.g., hyperlinks, dates, curated multi-document datasets) exists but does not scale broadly or requires metadata not available for most web text (§5, “Pretraining with related documents”).
- How this work positions itself
  - ICLM reframes the problem as a data-ordering task: efficiently sort billions of documents so that consecutive items in the training stream are semantically related (§2). It uses scalable approximate nearest neighbor retrieval to find related documents (§2.1; Appendix A.2) and a graph traversal algorithm to build a single long path that visits each document once, maximizing local relatedness while avoiding repetition (§2.2; Algorithm 1).

## 3. Technical Approach
High-level idea: Keep the usual next-token prediction objective and architecture, but change how training contexts are formed: instead of random concatenation, concatenate related documents. The pipeline has three core steps (see Figure 2 and §2):

1) Find related documents at scale (retrieval; §2.1)
- Each document `d_i` is embedded with the `contriever` model (mean-pooled final-layer token embeddings). Only the first 512 tokens of each document are encoded to reduce cost (§3.1).
- Similarity between documents is the cosine similarity of their embeddings, `s(d_i, d_j) = cos(E(d_i), E(d_j))` (Equation 1).
- For each `d_i`, retrieve the top-k neighbors `N(d_i)` using FAISS with an IVF-PQ index and big-batch offline search (OIVFBBS). Practical details:
  - Scale: 235,266,464 documents (768-dim float32 embeddings) from CCNet/English; index size ≈62 GB (§A.2).
  - Indexing/search: IVFPQ (32,768 lists), 256-byte codes, nprobe=64; batches of 50M embeddings; search time ≈6 hours on 32 GPUs; traversal ≈12 hours on 20 CPUs (§3.1; §A.2).
- Semantic deduplication: The retrieval scores are reused to detect near-duplicates; highly similar pairs are pruned so a context does not contain trivial paraphrases that would encourage copying (§2.1; Appendix A.1).

2) Construct a single, coherent document path (graph traversal; §2.2; Algorithm 1)
- Build an undirected weighted graph `G=(D, L)` where nodes are documents and an edge exists if either document is in the other’s `k` nearest neighbors. Edge weights are the cosine similarities from Equation 1 (§2.2).
- Goal: visit each document exactly once while making consecutive documents as similar as possible; formulated as a maximum traveling salesman problem (TSP) on `G` (§2.2).
- Exact TSP is intractable at this scale, so a greedy traversal is used (Algorithm 1):
  - Start from an unvisited node with minimum degree (“hard-to-connect” document). Intuition: such nodes are most at risk of being attached to a poor neighbor if left for later (§2.2).
  - Repeatedly extend the path by moving to the highest-weight unvisited neighbor (most similar document). This produces locally coherent runs (Figure 2).
  - When reaching a node whose neighbors are all visited (graph is sparse), add a zero-weight jump to a random unvisited minimum-degree node and continue. This creates several coherent segments stitched together without repeating any document (§2.2).
- Finally, slide along this path and chunk it into fixed-length training contexts (e.g., 8192 tokens) for pretraining (§2.2; Figure 2). Batches are formed to keep contexts within a batch diverse (to avoid batch-level redundancy).

3) Pretrain as usual
- Architecture: LLaMA-style transformer; sizes 0.3B, 0.7B, 1.5B, 7B; context length 8192; AdamW optimizer; cosine LR schedule; FlashAttention for memory (§3.1).
- Data: 235M CCNet English documents, 306B tokens; same corpus and number of updates across all compared methods for fair compute (§3.1; §3.2).
- Methods compared:
  - `Standard`: random document order, the common practice (§3.2).
  - `kNN`: pack each document with its top-k retrieved neighbors directly; allows repeats across contexts (§3.2).
  - `ICLM`: the proposed sorted-order scheme with graph traversal and no repeats (§2.2, §3.2).

Illustrative example (Figure 1):
- Predicting “For 2022, FIFA set the prize money at $42m … the highest so far.” becomes easier if earlier in the same context there is a document noting “World Cup never awarded more than $10M before 2022.” This shows why semantically related prior documents provide useful predictive signal across document boundaries (§1; Figure 1).

## 4. Key Insights and Innovations
- Reframing pretraining as a document-ordering problem
  - What’s new: Treats “how to pack the context window” as the central lever, not the loss or model. The method uses only reordering of existing data (§1–§2).
  - Why it matters: It produces cross-document predictive signal without changing the objective and scales to web corpora; the training recipe remains compatible with standard pipelines (§2; §3.1).

- Scalable relatedness graph + greedy maximum-TSP traversal
  - What’s new: Build a nearest-neighbor graph over hundreds of millions of documents and extract a single, non-repeating path that locally maximizes similarity (§2.1–§2.2; Algorithm 1; Appendix A.2).
  - Why it matters: Avoids the “data repeating problem” of `kNN` packing, where popular documents appear in many contexts and reduce training diversity (§2.2).

- Retrieval-driven semantic deduplication as a stability and quality control
  - What’s new: Use the same similarity signals to remove near-duplicates before packing contexts (§2.1; Appendix A.1).
  - Why it matters: Ablation shows dedup is crucial; removing it worsens perplexity (PPL 8.3 without vs 7.3 with dedup in Figure 5), likely because near-duplicate contexts encourage copying and destabilize training (§4.2).

- Strong multi-task gains specifically tied to cross-document reasoning
  - What’s new: A single data-ordering change yields consistent improvements on diverse evaluations that require using information from prior context: +8% in in-context learning (ICL), +14–15% in reading comprehension, +9% in retrieval-augmented QA, +5% in long-context reasoning, and +16% in context faithfulness (Tables 1–5; §3.3.2–§3.3.6).
  - Why it matters: Pinpoints that multi-document exposure during pretraining strengthens abilities central to modern LLM usage (RAG, ICL, long context).

## 5. Experimental Analysis
- Evaluation setup (§3)
  - Data/model: All methods use the same 306B-token CCNet corpus and identical LLaMA-style models at 0.3B–7B with 8192 context (§3.1–§3.2).
  - Baselines: `Standard` random concatenation; `kNN` direct top-k neighbor packing that allows repeats (§3.2).
  - Metrics and tasks:
    - Language modeling: Perplexity (lower is better) on Wikipedia, ArXiv, Books; documents randomly ordered at evaluation time (Figure 3; §3.3.1). Perplexity measures how well the model predicts text; lower PPL indicates better predictive fit.
    - In-context learning: 32-shot classification across 7 datasets; accuracy (Table 1; §3.3.2).
    - Reading comprehension: 2-shot on RACE-High/ Middle, SQuAD, BoolQ, DROP, HotpotQA; EM or accuracy as standard (Table 2; §3.3.3).
    - Retrieval augmentation (RAG): Natural Questions (NQ) and TriviaQA (TQA), closed-book (no retrieval) vs open-book (prepend top-10 Wikipedia passages); Exact Match (Table 3; §3.3.4).
    - Context faithfulness under knowledge conflict: NQ-Swap and MemoTrap; EM (Table 4; §3.3.5).
    - Long-context reasoning: SCROLL benchmark (NarrativeQA, Qasper, ContractNLI, QMSum, GovReport); F1 or ROUGE-1 after fine-tuning (Table 5; §3.3.6).
  - Training dynamics: Learning curves and downstream performance during pretraining (Figure 4; §4.1). Ablations on relevance strategy and dedup (Figure 5; §4.2). Effect of number of ICL examples (Figure 6; §4.3).

- Main quantitative results
  - Language modeling:
    - Figure 3: Across all sizes and datasets, `ICLM` has lower perplexity than `Standard` and `kNN`. This holds even though evaluation uses random ordering, showing benefits generalize beyond sorted inputs (§3.3.1).
  - In-context learning (32-shot; Table 1):
    - Average accuracy: `ICLM` 71.3 vs `Standard` 66.0 and `kNN` 61.8. Gains are uniform across sentiment, hate-speech, and topic classification (§3.3.2).
    - Quote:
      > Table 1: ICLM outperforms baselines on all seven datasets; average +8% over Standard.
  - Reading comprehension (2-shot; Table 2):
    - Average: `ICLM` 43.2 vs `Standard` 37.6 and `kNN` 36.0. Largest relative gains on HotpotQA (21.9 vs 10.5, +11.4 absolute) and DROP (35.7 vs 27.2, +8.5 absolute) where multi-hop reasoning and numerical operations rely on context (§3.3.3).
    - Quote:
      > Table 2: ICLM > Standard and kNN on all six datasets; average improvement ≈14–15%.
  - Retrieval augmentation (Table 3):
    - Closed-book: similar or slightly worse (NQ: 17.0 vs 17.0; TQA: 48.0 vs 49.3). Open-book: strong gains (NQ: 32.2 vs 28.5; TQA: 51.6 vs 48.1), ≈+9% (§3.3.4).
    - Interpretation: Better use of provided passages; slight reduction in pure parametric memorization.
    - Quote:
      > Table 3: In open-book settings, ICLM outperforms Standard by +3.7 EM on NQ and +3.5 EM on TQA.
  - Context faithfulness under conflict (Table 4):
    - NQ-Swap: 45.8 vs 39.6; MemoTrap: 56.2 vs 48.4. This shows improved adherence to provided context when it contradicts pretraining memory (§3.3.5).
    - Quote:
      > Table 4: ICLM yields +6.2 EM (NQ-Swap) and +7.8 EM (MemoTrap) over Standard.
  - Long-context reasoning (SCROLL; Table 5):
    - Average: `ICLM` 34.1 vs `Standard` 32.5; gains across all five datasets even after fine-tuning (§3.3.6).
    - Quote:
      > Table 5: ICLM improves NarrativeQA F1 (17.1 vs 16.5), Qasper ROUGE-1 (36.7 vs 34.2), ContractNLI F1 (80.7 vs 78.6), QMSum ROUGE-1 (26.8 vs 25.1), GovReport ROUGE-1 (9.1 vs 8.2).
  - Training dynamics (Figure 4):
    - Lower training loss throughout and stable downstream improvements emerging after ≈150B tokens (§4.1).
  - Ablations (Figure 5):
    - Increasing document relevance improves PPL: Random 8.2 → Clustering 7.9 → Link-based (ICLM) 7.3 (§4.2).
    - Dedup is crucial: No dedup 8.3 PPL vs Dedup 7.3 (§4.2).
  - ICL example scaling (Figure 6):
    - Gains persist as shots increase, with diminishing returns after 32 examples (§4.3).

- Do the experiments support the claims?
  - The breadth of tasks (LM perplexity, ICL, RC, RAG, conflict tests, long-context) and consistent superiority of `ICLM` over two strong baselines support the central claim that cross-document coherence during pretraining improves multi-document reasoning (§3.3).
  - The open-book vs closed-book contrast (Table 3) provides a nuanced view: ICLM excels when external context is provided, aligning with the method’s training signal; slight closed-book decreases suggest a trade-off discussed below.
  - Ablations isolate two key design choices—document relevance and dedup—as causal factors (Figure 5).

- Failure cases and trade-offs visible in results
  - Slightly worse closed-book QA on TQA (48.0 vs 49.3) suggests reduced reliance on parametric memory when no context is provided (Table 3).
  - Long-context gains after fine-tuning are smaller than in zero-shot/ICL settings (Table 5), indicating partial wash-out during task-specific training (§3.3.6).

## 6. Limitations and Trade-offs
- Dependence on retrieval quality (§2.1; Appendix A.2)
  - The neighbor sets come from the `contriever` embedding of the first 512 tokens. If a document’s most informative content occurs later or if the embedding model misses topical nuance, the graph may connect suboptimal neighbors.
- Approximate path construction (§2.2; Algorithm 1)
  - The greedy maximum-TSP traversal is not optimal and includes zero-weight jumps when stranded nodes are reached (graph sparsity). This can stitch together unrelated segments and occasionally insert weak transitions.
- Diversity vs coherence
  - While traversal avoids repeated documents across contexts, extended runs of highly similar documents might reduce topical breadth in a training step. The batching strategy shuffles contexts to reintroduce diversity (§2.2), but global effects were not deeply audited.
- Compute and engineering overhead
  - One-time costs include embedding billions of tokens, building a 62 GB FAISS index, GPU-based big-batch search, and CPU traversal (hours to a day-scale; §3.1; Appendix A.2). This is tractable at lab scale but adds nontrivial infrastructure requirements.
- Closed-book knowledge trade-off
  - Small drops on some closed-book tasks (Table 3) suggest a shift from memorization toward context use. Depending on application (e.g., when retrieval is not available), this may be a disadvantage.
- Scope
  - Experiments are on English web text (CCNet) with LLaMA-like models up to 7B and 8k context. Behavior at larger scales (100B+ parameters), different domains (code, legal), languages, and much longer windows remains to be tested (§6 “Conclusion” hints at future avenues).

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that the ordering of pretraining data—often treated as an implementation detail—is a first-class lever for improving cross-document reasoning. It complements architectural and objective innovations by exploiting context packing alone (§1–§2).
- Follow-up research enabled/suggested
  - Better retrieval for ordering:
    - Train domain-adaptive or multilingual retrievers; include more than the first 512 tokens; hierarchical or dynamic neighbor selection (§6).
  - Alternative graph objectives:
    - Explore global tour-search heuristics (e.g., Lin–Kernighan variants) or beam/path-merging strategies to reduce zero-weight jumps; consider community detection and Hamiltonian path approximations tailored to sparse k-NN graphs (§2.2).
  - Curriculum design:
    - Mix coherence levels (local vs global topicality), interleave random and linked segments, or progressively increase run lengths to balance diversity and signal.
  - Integration with instruction/multitask finetuning and long-context scaling:
    - Combine with instruction-tuning datasets that already concatenate demonstrations (§5 “Multitask finetuning…”); pair with position-scaling methods to exploit even longer windows.
  - Cross-modal and code repositories:
    - Leverage inherent link structures (citations, code repos) to form even more meaningful multi-document contexts (noted in §6 “Conclusion”).
- Practical applications
  - Retrieval-augmented assistants: measurable gains in open-book QA (Table 3) translate directly to improved search-grounded question answering and fact-checking.
  - Multi-document reading and summarization: stronger performance on HotpotQA, DROP, and SCROLL indicates better synthesis over multiple sources (Tables 2 and 5).
  - Safer, more faithful generation: improvements on NQ-Swap and MemoTrap show increased willingness to follow provided evidence even when it conflicts with parametric knowledge (Table 4).

Overall, in-context pretraining is a simple but powerful modification—just reorder documents—that consistently yields better cross-document reasoning. The method is easy to adopt in existing pipelines, and the paper provides concrete algorithms and scaling notes (Algorithm 1; Appendix A.2) that make replication feasible.
