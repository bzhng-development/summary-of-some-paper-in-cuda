# Scaling Retrieval-Based Language Models with a Trillion-Token Datastore

**ArXiv:** [2407.12854](https://arxiv.org/abs/2407.12854)

## 🎯 Pitch

This paper introduces datastore size as a crucial and previously underexplored scaling dimension for language models, complementing model size and pretraining data. By developing MASSIVEDS—the largest and most diverse open-source retrieval datastore at 1.4 trillion tokens—and a compute-efficient pipeline, the authors show that increasing datastore size yields monotonic gains in language modeling and broad downstream tasks, allowing smaller models with large datastores to outperform much larger LM-only models for the same training compute. This work fundamentally expands the roadmap for language model scaling and paves the way for more efficient, knowledge-rich, and broadly applicable AI systems.

---

## 1. Executive Summary
This paper treats the amount of external information available at inference time—the size of a retrieval datastore—as a new scaling dimension for language models, alongside model parameters and pretraining data. It introduces MASSIVEDS, a 1.4-trillion-token, multi-domain, open-source datastore and an efficiency-oriented pipeline that makes studying “datastore scaling” feasible; results show monotonic improvements in language modeling and several downstream tasks as the datastore grows, with compute-optimal analyses indicating retrieval-augmented models can outperform LM-only models for the same training compute.

## 2. Context and Motivation
- Problem/gap addressed
  - Scaling laws have focused on two axes: model size and pretraining data size. What has been largely missing is a systematic study of a third axis: the quantity of information accessible at inference time via retrieval (Section 1).
  - Existing retrieval-augmented systems typically use small, single-domain datastores (e.g., Wikipedia, a few billion tokens). Larger prior efforts (e.g., RETRO at 1.7T tokens) are proprietary and mainly evaluate language modeling, not broad downstream tasks (Table 1).
- Why this matters
  - Retrieval can improve factuality, domain adaptation, and data attribution, and can make models more parameter-efficient. If datastore size reliably improves performance, practitioners could achieve better accuracy for the same training compute by shifting some “knowledge” from parameters to a non-parametric memory (Sections 1, 4.3).
- Prior approaches and their limitations
  - Small, single-source datastores (Wikipedia) limit coverage and generality.
  - Large-scale retrieval (e.g., RETRO, RETRO++) uses custom architectures and closed datastores, limiting reproducibility and breadth of evaluation (Table 1).
  - SPHERE (90B tokens) is open but sometimes underperforms small in-domain stores on downstream tasks (Table 1 discussion).
- Positioning of this work
  - Provides a fully open, trillion-token datastore, covering eight diverse domains (Table 2).
  - Designs a pipeline that makes datastore scaling experiments computationally accessible while being equivalent to naive construction with high probability (Section 3.2, Appendix A.5).
  - Offers broad evaluations: language modeling (perplexity) and multiple downstream tasks, plus compute-optimal scaling analyses across model families (LLAMA-2/3, PYTHIA, OLMO) (Sections 4.1–4.3).

## 3. Technical Approach
This work has two pillars: (1) building a very large, multi-domain datastore and (2) an efficient pipeline to evaluate how scaling that datastore affects performance.

- Core concepts (selectively defined)
  - `Datastore`: a large collection of text passages indexed by a retriever; documents are retrieved at inference time and prepended to the model’s input.
  - `Retrieve-in-context language model (RIC-LM)`: a standard LM that reads retrieved passages as additional context; no model architecture changes are required (Section 2, “RIC-LM”).
  - `Retriever`: a separate model that maps queries and documents to vectors and returns the most similar documents. Here, primarily `CONTRIEVER-MSMARCO` is used (Section 4.1).
  - `Reranker`: an optional model that re-sorts retrieved documents with a stronger but costlier scoring function (Section 5.2).
  - `Perplexity (PPL)`: a standard language modeling measure; lower is better.
  - `Decontamination`: removal of documents that overlap too closely with evaluation data to avoid test leakage (Section 5.3; Appendix B.1).

A) MASSIVEDS: building the trillion-token datastore (Section 3.1; Table 2)
- Composition (1.44T tokens total):
  - General web (CommonCrawl snapshots and C4): 1,191.7B tokens.
  - Domain-specific sources: `Books` (26.3B), `STEM` (arXiv, peS2o; 97.7B), `Encyclopedia` (DPR/RedPajama Wikipedia; 31.9B), `Forum (StackExchange)` (20.2B), `Code (GitHub)` (52.8B), `Math` (OpenWebMath, NaturalProofs; 14.1B), `Biomedical (PubMed)` (6.5B). See Table 2.

B) The efficiency-oriented pipeline (Section 3.2; Figure 2; Appendix A)
Goal: Avoid repeatedly re-indexing trillions of tokens for every experiment variant (datastore size, filtering options, random seeds).

Step-by-step (Appendix A):
1) Distributed indexing (A.1):
   - Split each domain into shards; embed each 256-word chunk with the retriever to create vectors; store as sharded indices.
2) Distributed document retrieval (A.2):
   - For each query, retrieve top-`K` candidates independently from each shard in parallel.
3) Domain merging (A.3):
   - Merge shard results and keep the overall top-`K`. Lemma A.1 proves that “m-shard distributed element-wise top-`K` retrieval” yields the same results as retrieving from a single monolithic index.
4) Post-hoc data filtering and optional reranking (A.4):
   - Deduplicate retrieved candidates using 13-gram Jaccard similarity (≥80%) and remove tiny fragments (<13 words).
   - Decontaminate against test data using 13-gram Jaccard (≥80%) and, for perplexity, an additional longest-overlap threshold (32-gram by default).
   - Reranking (optional): apply a stronger model (e.g., a cross-encoder) to reorder the top-`K` candidates.
   - Lemma A.2 shows that performing de-duplication and decontamination post-retrieval is equivalent to applying them globally before retrieval.
5) Subsampling (A.5):
   - To simulate different datastore sizes, sample each retrieved document with probability `p` (the datastore “scale”), then take the final top-`k` for the model input.
   - Crucial optimization: only subsample from the per-query top-`K` pool (with `K` ≫ `k`) rather than from the entire datastore. Algorithm 2 shows this reduces compute by an order of magnitude compared to the naive Algorithm 1, which would rebuild indices for every `(p, seed)` pair.
   - Lemma A.3: With high probability, subsampling the top-`K` pool and then taking top-`k` yields the same results as indexing a fresh datastore subsampled at rate `p`—provided enough candidates remain.
   - Lemma A.4: Independent element-level operations (e.g., reranking, subsampling) commute; set-level operations (e.g., de-duplication) do not.
6) Evaluation (A.6):
   - Prepend the top-`k` documents to the query and (for downstream tasks) few-shot examples; then run the LM.

Design choices and why:
- Retrieve first, filter/subsample later: Indexing and search are the costliest steps; doing them once and sharing results across variants saves compute (Figure 2).
- Large `K`, small `k`: Use `K=1000` to ensure there are enough candidates after filtering/subsampling; use `k=3` to keep prompts short and focus on high-quality evidence (B.1).
- Post-hoc de-duplication/decontamination: Avoids global preprocessing over trillions of tokens while preserving equivalence guarantees (Appendix A.4–A.5).

C) Models, retrievers, and evaluation protocol (Section 4.1; B.2–B.3)
- Retrievers: Main—`CONTRIEVER-MSMARCO` (dense, 177M). Ablations with DRAGON-RoBERTa and GTR-T5-Base show similar performance (Appendix E.1, Table 6), but Contriever is faster.
- Readers (LMs): `LLAMA-2` (7B, 13B), `LLAMA-3` (8B), `PYTHIA` (1B, 2.8B, 6.9B, 12B), `OLMO-1.7` (1B, 7B).
- Prompting: 5-shot for downstream tasks; retrieved documents are prepended before the few-shot examples, then the question (B.3).
- Metrics:
  - Language modeling: perplexity on RedPajama (general web) and S2ORC (scientific papers) (B.2).
  - Downstream tasks: Exact Match for TriviaQA (TQA) and Natural Questions (NQ); accuracy for MMLU and MedQA (B.3).

D) Compute-optimal scaling analysis (Section 4.3; B.4)
- Use intermediate checkpoints of `PYTHIA` (trained up to 300B tokens) and `OLMO-1.7` (trained up to 2–3T tokens) to approximate models trained on different corpus sizes.
- Compute accounting:
  - Pretraining FLOPs ≈ `6 * N_LM * D_pretrain` (2 forward + 4 backward “units” per token).
  - Datastore construction FLOPs ≈ `2 * N_retriever * D_datastore` (forward-only embedding).
  - Since `N_retriever` is small (177M) relative to LM sizes, and embedding is forward-only, building a huge datastore is much cheaper than pretraining a large LM on the same token count (B.4).

## 4. Key Insights and Innovations
1) Datastore size is a powerful third scaling axis
- What’s new: Treats inference-time memory size (datastore tokens) as a monotonic scaling dimension, analogous to model size and pretraining data.
- Why it matters: Results show consistent gains in perplexity and knowledge-intensive QA as the datastore grows, with no clear saturation in the explored range (Figure 3a–d). This reframes how to invest compute: more into datastore building (cheap) rather than into ever-larger models (expensive).

2) A trillion-token, multi-domain, open datastore and a reproducible pipeline
- What’s new: MASSIVEDS (1.4T tokens, 8 domains; Table 2) is the largest open datastore of its kind, accompanied by code and indices.
- Why it matters: Prior trillion-scale datastores were closed (RETRO; Table 1). The pipeline’s rearrangement (retrieve-then-filter-then-subsample) reduces compute by >10× while remaining equivalent with high probability (Section 3.2; Algorithms 1–2; Lemmas A.1–A.4).

3) Compute-optimality: Retrieval beats LM-only at the same training compute
- What’s new: Pareto curves (Figure 4) show retrieval-based models dominate LM-only models for a fixed training compute budget across knowledge-intensive tasks (TriviaQA, NQ).
- Why it matters: Shifting “knowledge storage” from parameters to a datastore can be more compute-efficient (Section 4.3). This supports a design where smaller LMs with big datastores outperform larger LMs without retrieval.

4) Broad, multi-domain retrieval works—and the retriever finds the right domains
- What’s new: A single multi-domain datastore matches or outperforms single-domain stores on most tasks (Table 3).
- Why it matters: Practically, you often don’t know the “right” domain. Figure 5 shows that retrieval naturally pulls more from relevant sources (e.g., peS2o/PubMed for MedQA; Wikipedia/web for NQ), enabling a general-purpose datastore strategy.

5) Quality controls and retrieval improvements significantly modulate gains
- Data hygiene: Decontamination strongly affects perplexity curves (Figure 7); deduplication helps avoid saturation as `p` increases (Appendix E.2, Figure 13e).
- Retrieval quality: Better reranking (cross-encoder) boosts performance; a lexical oracle bound indicates more headroom (Figure 6).

## 5. Experimental Analysis
A) Evaluation methodology (Section 4.1; B.2–B.3)
- Language modeling:
  - RedPajama (multi-domain web): PPL measured over 1024-token windows (stride 512); the first half of each window is the retrieval query and prefix (B.2).
  - S2ORC (scientific papers): same setup.
- Downstream tasks:
  - TriviaQA and Natural Questions: Open-domain QA; Exact Match metric; 5-shot; retrieved top-3 passages prepended.
  - MMLU: multi-subject reasoning; accuracy.
  - MedQA: medical exam Q&A; accuracy.
- Default retrieval configuration: `k=3` documents; `K=1000` prefiltered candidates; no reranking unless specified (Section 4.1; B.1).

B) Main quantitative results
- Monotonic scaling on language modeling and QA
  - Figure 3a–b: Perplexity consistently decreases as datastore size increases for LLAMA-2 7B/13B and LLAMA-3 8B—no clear saturation.
  - Figure 3c–d: On TriviaQA and NQ, retrieval-based models substantially outperform LM-only baselines at all scales; performance increases with datastore size.
- Concrete numbers (LLAMA-2 7B, MASSIVEDS vs LM-only; Table 3):
  > “TriviaQA EM: 77.0 vs 64.1; NQ EM: 34.6 vs 26.6; MMLU Acc: 49.3 vs 45.8; MedQA Acc: 39.4 vs 36.6. RedPajama PPL: 3.50 vs 4.09; S2ORC PPL: 6.57 vs 7.18.”
  - These improvements are large for knowledge-intensive QA and moderate but consistent for reasoning-heavy tasks.
- Small with retrieval vs large without retrieval (Section 4.2; Figure 3)
  - Example: LLAMA-2 7B + retrieval surpasses LLAMA-2 13B LM-only on multiple metrics, indicating that datastore size can substitute for parameter count on knowledge-centric tasks.
- Compute-optimality (Figure 4; Section 4.3)
  - Across `PYTHIA` and `OLMO`, Pareto frontiers for retrieval-augmented models dominate LM-only on TriviaQA and NQ at matched training compute.
  - On more reasoning-oriented tasks (MMLU, MedQA), `OLMO` (trained on more and higher-quality data) benefits from retrieval, while `PYTHIA` may not, suggesting the LM must have sufficient reasoning ability to exploit retrieved text (Finding 5).
- Multi-domain vs single-domain (Table 3; Section 5.1)
  - MASSIVEDS outperforms single-domain datastores on language modeling, TriviaQA, and MMLU; it matches the in-domain best for NQ and MedQA:
    > “On TriviaQA, MASSIVEDS: 77.0 vs DPR Wiki: 72.6; on MMLU, MASSIVEDS: 49.3 vs DPR Wiki: 48.3; on NQ, MASSIVEDS ties DPR Wiki at 34.6; on MedQA, MASSIVEDS: 39.4 is slightly below RedPajama-Wiki: 39.8.”
  - Figure 5 shows retrieved top-1 documents for NQ are predominantly from Wikipedia/web; for MedQA from peS2o/PubMed.
- Retrieval quality ablation (Section 5.2; Figure 6)
  - Cross-encoder reranking improves over no reranking on both TQA and NQ.
  - A “lexical oracle” (reorders by overlap with the gold answer) reveals additional headroom beyond the cross-encoder, implying better retrievers/rerankers could further enhance scaling trends.
- Data decontamination ablation (Section 5.3; Figure 7)
  - On language modeling, less decontamination yields better (lower) perplexity—evidence that PPL benefits from lexical overlap. Even with aggressive decontamination (8-gram), retrieval still helps.
  - On NQ, decontamination has little effect, suggesting low contamination in that setup.
- Deduplication and quality filters (Appendix E.2; Figure 13)
  - Global deduplication is important to avoid saturation as `p` increases in NQ (Figure 13e).
  - DOLMA-style quality filters show limited effect, likely because sources like RedPajama were already filtered (Figure 13c,f).
- Retriever ablation (Appendix E.1; Table 6)
  - Contriever, DRAGON, and GTR-Base perform similarly on PPL, NQ, MMLU using 10% MASSIVEDS; Contriever is faster, motivating its choice for full-scale runs.

C) Are the experiments convincing?
- Breadth: Multiple model families and sizes; both upstream (PPL) and downstream tasks (QA, reasoning).
- Rigor:
  - Strong controls: LM-only baselines, single-domain comparisons, graded decontamination, deduplication, reranking ablations.
  - Compute accounting and Pareto analyses (Figure 4; B.4) speak to resource trade-offs.
- Caveats:
  - Intermediate checkpoints for compute-optimality are proxies; not independently tuned for shorter training schedules (Section 4.3, “Use of intermediate checkpoints”).
  - Reranking is not used in main results; better retrieval could further shift curves upward (Figure 6).

D) Mixed/conditional findings
- Reasoning-heavy tasks (MMLU, MedQA) benefit less from datastore scaling unless the base LM is strong and the datastore contains sufficient domain coverage (Section 4.2; Figure 4 right).
- LLAMA-3 8B shows worse PPL than LLAMA-2 7B on RedPajama (Figure 3a), possibly due to differences in training data weighting and post-training that do not optimize PPL (Appendix D).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The base LM must have sufficient reasoning ability to use retrieved evidence; otherwise, retrieval gains are limited (Figure 4, PYTHIA on MMLU/MedQA).
  - The datastore must contain relevant content; limited medical or textbook coverage may bottleneck MedQA/MMLU performance (Sections 4.2, 5.1).
- Computational factors
  - Training-time compute: Favorable—datastore construction is far cheaper than LM pretraining (B.4). But changing the retriever requires re-embedding/re-indexing the full datastore (Section 6).
  - Inference-time compute: Retrieval adds search cost and increases prompt length; however, one can offset this by using a smaller LM (Section 4.3, “Discussion on inference cost”).
- Methodological trade-offs
  - Post-hoc de-duplication/decontamination is provably equivalent with high probability, not deterministically in all cases (Appendix A.5). Using large `K` mitigates failures.
  - Intermediate checkpoints used for compute-optimal curves are not fully compute-tuned (Section 4.3).
- Coverage gaps
  - Reasoning-oriented resources (e.g., structured textbooks, high-quality medical corpora) are limited in MASSIVEDS; results suggest adding these could help (Sections 4.2, 6).

## 7. Implications and Future Directions
- How this changes the landscape
  - Datastore size emerges as a first-class scaling axis. Practitioners can target better cost–performance by moving factual knowledge into a large non-parametric memory while keeping LMs smaller.
  - Open, trillion-scale retrieval is now practically accessible and reproducible, thanks to the released datastore, embeddings, indices, and the compute-efficient pipeline.
- Follow-up research enabled
  - Retrieval quality: The sizable gap between cross-encoder and oracle reranking (Figure 6) motivates better dense retrievers, hybrid lexical–dense strategies, or task-aware rerankers.
  - Datastore curation: Add high-quality, reasoning-focused sources (textbooks, verified medical literature) and explore more advanced filtering (semantic deduplication, topic balancing).
  - Training strategies: Jointly train LMs to better use retrieved context (e.g., retrieval-aware pretraining or instruction tuning), and study inference-compute-optimal trade-offs (Section 4.3, future work).
  - Systems and serving: Efficient search, caching, and selective augmentation (e.g., compress/retrieve only when helpful) to reduce inference latency and cost.
- Practical applications
  - Knowledge-intensive assistants: Legal, biomedical, and enterprise QA with stronger attribution and freshness via datastore updates.
  - Domain adaptation: Swap or augment domain-specific slices of MASSIVEDS without retraining the LM.
  - Governance and attribution: Easier provenance tracking and credit assignment by pointing to retrieved sources (Section 2 discussion; related works).

Overall, the study demonstrates that enlarging a retrieval datastore systematically improves performance and can be a more compute-efficient way to add knowledge than growing model size or pretraining tokens. The combination of an open trillion-token datastore and a carefully engineered, provably equivalent pipeline makes this direction immediately actionable for both research and production.
