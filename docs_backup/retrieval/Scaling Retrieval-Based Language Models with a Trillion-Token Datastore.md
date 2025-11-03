# Scaling Retrieval-Based Language Models with a Trillion-Token Datastore

**ArXiv:** [2407.12854](https://arxiv.org/abs/2407.12854)
**Authors:** Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min, Luke Zettlemoyer, Pang Wei Koh
**Institutions:** University of Washington, Allen Institute for AI

## 🎯 Pitch

This paper introduces a transformative third scaling dimension for large language models by integrating vast external retrieval datastores, specifically MASSIVEDS, enhancing their performance on knowledge-intensive tasks without the limitations of parameter scaling. By demonstrating how small models paired with extensive datastores can outperform larger, model-only counterparts, this approach promises a more efficient compute-performance trade-off, marking a significant leap in factuality and domain adaptation while optimizing resource allocation in AI development.

---

## 1. Executive Summary
This paper shows that a third scaling dimension for large language models—how much external text a model can retrieve at inference time—improves performance steadily without saturation. It builds a 1.4-trillion‑token, multi‑domain retrieval datastore (`MASSIVEDS`) and an efficient, provably equivalent pipeline to study datastore size, demonstrating that small models plus large datastores can beat larger “LM‑only” models on knowledge-heavy tasks, and that retrieval achieves better compute–performance trade‑offs than pretraining alone (Figures 1, 3, 4; §3–§4).

## 2. Context and Motivation
- Gap addressed
  - Most scaling laws optimize two dimensions: number of parameters and amount of pretraining data (e.g., Kaplan et al. 2020; Hoffmann et al. 2022). This work adds a third: the size of a retrieval `datastore` (the external corpus the model can search and copy into its context at inference). See §1 and Figure 1.
  - Prior retrieval systems usually used small, single‑domain stores (e.g., Wikipedia with a few billion tokens) and custom architectures (e.g., RETRO) with proprietary corpora, leaving open how retrieval scaling behaves with modern “retrieve-in-context” setups on diverse downstream tasks (Table 1; §2).

- Why it matters
  - Retrieval can improve factuality, domain adaptation, and reduce parametric memorization. Indexing a datastore is much cheaper than pretraining on the same text (§4.3; Appendix B.4), so adding a larger datastore may be a more compute‑efficient way to add knowledge.

- Prior approaches and limitations
  - RETRO (1.7T tokens) evaluated language modeling with a proprietary datastore, but used small task‑specific stores for downstream evaluation (Table 1).
  - SPHERE (90B tokens; open) did not consistently beat smaller, in-domain stores on downstream tasks.
  - Many works focus on single domains or lack open resources for trillion‑token retrieval (§2; Table 1).

- Positioning
  - This paper introduces the largest open retrieval datastore to date, `MASSIVEDS` (1.4T tokens across eight domains; Table 2; §3.1), and an efficiency‑oriented pipeline (§3.2; Figure 2) that makes trillion‑token retrieval studies feasible and repeatable. It then analyzes scaling on language modeling and multiple downstream tasks (§4–§5), including compute‑optimal curves (Figure 4).

## 3. Technical Approach
This is a retrieve‑in‑context approach (RIC‑LM): at inference, the system retrieves documents and concatenates them to the prompt (no model architecture changes). Key components:

- What the system is
  - `Datastore`: a very large corpus chunked into fixed-length passages that can be retrieved at inference time. MASSIVEDS includes general web and domain‑specific sources (Table 2; §3.1).
  - `Retriever`: a model that maps text to vectors and finds nearest neighbors; here `CONTRIEVER‑MSMARCO` (177M parameters) is used by default (§4.1; Appendix E.1).
  - `RIC‑LM`: a standard LM (e.g., `LLaMA‑2/3`, `Pythia`, `OLMo`) that reads the concatenation of retrieved documents plus the task prompt (§4.1).

- Datastore composition (Table 2; §3.1)
  - 1.4T tokens (LLaMA‑2 tokenizer), mixing:
    - General web: Common Crawl (2019–2023) and C4 (≈1.19T tokens).
    - Domains: Books (26.3B), STEM (97.7B, including arXiv and peS2o), Encyclopedia (31.9B, incl. Wikipedia), StackExchange (20.2B), Code (52.8B), Math (14.1B), Biomedical (6.5B).

- Efficient scaling pipeline (Figure 2; §3.2; Appendix A)
  - Challenge: naively rebuilding indices for every datastore variant (size, seed, filters) is prohibitively expensive at trillion scale (Figure 2, top).
  - Strategy: do the most expensive steps once (indexing and initial retrieval), then apply experimental variants only to the much smaller set of retrieved candidates (Figure 2, bottom).
    1) Distributed indexing: split data into shards; embed every document once; store in a flat inner‑product index (FAISS `IndexFlatIP`) (Appendix A.1).
    2) Distributed retrieval: for each query, get top‑`K` candidates per shard, then merge by score (Appendix A.2–A.3). A lemma proves this is equivalent to retrieving over a single unsharded index (Lemma A.1).
    3) Post‑hoc filtering over retrieved candidates only:
       - Deduplication (near‑duplicate removal) using 13‑gram Jaccard ≥80% (Appendix A.4.1; B.1).
       - Decontamination (to avoid test leakage) using 13‑gram Jaccard and/or longest‑overlap thresholds; stricter for language modeling (§5.3; Figure 8; Appendix A.4.2; B.1).
       - Optional reranking with a stronger model (cross‑encoder `MiniLM‑L12‑v2`), or oracle lexical reranker for an upper bound (§5.2; Appendix A.4.3).
    4) Subsampling to simulate smaller datastores: sample each of the `K` candidates independently with probability `p`, then keep the top‑`k` by original retrieval score (Appendix A.5). A lemma shows this is equivalent—with very high probability—to building a smaller datastore and rerunning retrieval (Lemma A.3). Failure probability (not enough remaining docs) is exponentially small in `K`; with `K=1000`, `k=3`, even `p=0.01` succeeds ≥0.997 (Table 4).
    5) Evaluation: concatenate final top‑`k` docs (default `k=3`) before the few-shot prompt (§4.1; B.3).

  - Commutativity and correctness: the paper proves which operations commute and when the re‑ordering is equivalent to the naive pipeline (Lemmas A.1–A.4; Proposition A.1).

- Retrieval and prompting details (§4.1; B.2–B.3)
  - Chunking at retrieval granularity: 256 words.
  - For language modeling perplexity, use 1,024‑token windows with a 512‑token stride; the first 512 tokens form the retrieval query; the next 512 tokens are the target (Appendix B.2).
  - For downstream tasks, prepend retrieved docs, then the few‑shot examples, then the question (Appendix B.3). Default `k=3`, reverse order so higher‑ranked docs are closer to the question (§4.1).

- Compute accounting for scaling (Appendix B.4; §4.3)
  - FLOPs approximations:
    - Pretraining: `FLOPspretrain ≈ 6 * N_LM * D_pretrain`.
    - Datastore construction (embedding): `FLOPsdatastore ≈ 2 * N_retriever * D_datastore`.
  - Indexing with a flat index adds no extra construction FLOPs (§4.3).
  - Compute‑optimal curves use intermediate checkpoints from `Pythia` and `OLMo` to approximate different `D_pretrain` (§4.3; Appendix B.4).

## 4. Key Insights and Innovations
- Datastore size is a real scaling axis with monotonic gains
  - Insight: Increasing the datastore size consistently lowers perplexity and boosts accuracy on knowledge‑intensive tasks, with no obvious saturation up to 1.4T tokens (Figure 3a–f; §4.2).
  - Significance: A smaller LM + large datastore can outperform a larger LM without retrieval on tasks like TriviaQA and NQ (Figure 3c–d).

- An efficient, provably equivalent scaling pipeline
  - Innovation: Retrieve a large candidate set once (`K ≫ k`) and apply subsampling, deduplication, decontamination, and reranking only on those candidates. Theoretical results show equivalence to naive construction with high probability (Appendix A; Figure 2).
  - Impact: Reduces compute by more than an order of magnitude (§3.2), making trillion‑token retrieval studies feasible and reproducible.

- Compute‑optimal scaling with retrieval
  - Finding: At equal training compute, retrieval‑augmented systems achieve better Pareto‑optimal performance on downstream tasks than LM‑only baselines (Figure 4; §4.3).
  - Reason: Indexing is much cheaper than pretraining the LM on the same data (Appendix B.4), so shifting “knowledge storage” into the datastore is more compute‑efficient—provided the LM can use retrieved text (§4.3).

- Multi‑domain datastore that generalizes across tasks
  - Innovation: `MASSIVEDS` covers eight domains (Table 2) and is open‑sourced (abstract; §6). Experiments show it matches or outperforms single‑domain datastores across tasks (Table 3; §5.1).
  - Mechanism: The retriever automatically pulls from relevant sub‑domains; e.g., more Wikipedia for `Natural Questions`, more scientific papers for `MedQA` (Figure 5).

## 5. Experimental Analysis
- Setup overview (§4.1; B.2–B.3)
  - Models: `LLaMA‑2 (7B, 13B)`, `LLaMA‑3 (8B)`, `Pythia (1B, 2.8B, 6.9B, 12B)`, `OLMo‑1.7 (1B, 7B)`.
  - Retriever: `CONTRIEVER‑MSMARCO` by default; ablations with `DRAGON` and `GTR‑Base` show similar performance (Appendix E.1, Table 6).
  - Datastore sizes simulated by subsampling probabilities `p = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]`, with three random seeds (Appendix B.1).

- Datasets and metrics (§4.1; B.2–B.3)
  - Language modeling: RedPajama (multi‑domain web) and S2ORC (scientific papers); metric: perplexity (PPL).
  - Downstream:
    - `TriviaQA` and `Natural Questions`: exact match accuracy.
    - `MMLU` and `MedQA`: accuracy (multiple‑choice).
    - All downstream tasks evaluated 5‑shot with retrieved docs prepended (Appendix B.3).

- Main quantitative results
  - Language modeling improves monotonically with datastore size (Figure 3a–b; §4.2):
    - Example with `LLaMA‑2 7B`: RedPajama PPL drops below the LM‑only baseline; Table 3 reports 4.09 (LM‑only) → 3.50 with MASSIVEDS. On S2ORC: 7.18 → 6.57 (Table 3).
    - Gains persist even after aggressive decontamination (Figure 7, left): removal of exact lexical overlap reduces the benefit but does not eliminate it, indicating semantic help from retrieved docs.
  - Knowledge‑intensive QA gains (Figure 3c–d):
    - `TriviaQA` with `LLaMA‑2 7B`: LM‑only 64.1% vs 77.0% with MASSIVEDS (Table 3).
    - `Natural Questions` with the same model: 26.6% → 34.6% (Table 3).
    - Benefits grow with datastore size; small models with retrieval beat larger LM‑only models (Figure 3c–d).
  - Reasoning‑heavy tasks are mixed (Figure 3e–f; §4.2):
    - `MMLU`: retrieval helps steadily but doesn’t flip ordering (smaller models don’t surpass larger ones). For `LLaMA‑2 7B`, LM‑only 45.8% → 49.3% with MASSIVEDS (Table 3).
    - `MedQA`: limited gains and mostly for weaker models (Figure 3f; Table 3).
  - Compute‑optimal curves (Figure 4; §4.3):
    - For `TriviaQA` and `NQ`, `Pythia` (≤300B pretraining tokens) and `OLMo` (≤2–3T) show similar Pareto trajectories when retrieval is used—suggesting simple factual extraction is learned early (§4.3, Finding 4).
    - For `MMLU/MedQA`, `OLMo` benefits from retrieval, `Pythia` mostly doesn’t (right half of Figure 4; §4.3, Finding 5).

- Reranking and retrieval quality (§5.2; Figure 6)
  - Replacing no reranker with a cross‑encoder boosts both `TriviaQA` and `NQ`, but a large gap remains to the “lexical oracle” (which knows the answer and ranks by overlap). This indicates ample headroom from better retrievers/rerankers.

- Single‑ vs multi‑domain datastores (§5.1; Table 3; Figure 5)
  - `MASSIVEDS` outperforms or matches single‑domain stores across tasks:
    - On `TriviaQA`, it beats Wikipedia and other domain stores (77.0% vs next best 72.9% with RedPajama web; Table 3).
    - On `MMLU`, `MASSIVEDS` ties the best single‑domain score (49.3% vs 48.3%–48.3% ranges in Table 3).
  - Domain‑adaptive retrieval (Figure 5): Top‑1 retrieved docs for `NQ` skew to Wikipedia; for `MedQA`, they skew to scientific papers, even though the underlying datastore is broad.

- Data filtering and decontamination (§5.3; Figure 7; Appendix E.2)
  - Decontamination strongly affects PPL but not NQ accuracy (Figure 7): suggests PPL gains partly stem from lexical overlaps, but retrieval still helps after strict filtering.
  - Global deduplication mitigates saturation on NQ as datastore grows (Appendix E.2, Figure 13e). Dolma‑style quality filters have small effect here (Figure 13c,f), likely because inputs were already filtered (Appendix E.2).

- Additional observations and ablations
  - Removing “short chunks” (<13 words) avoids unhelpful lexical matches and improves NQ at large scales (Appendix E.2; Figures 14–15).
  - LLaMA‑3 8B shows worse PPL than LLaMA‑2 7B on RedPajama (Figure 3a; Appendix D), plausibly due to domain mismatch or post‑training that prioritizes instruction‑following over PPL on this corpus.

- Do the experiments support the claims?
  - Yes for knowledge recall: strong, consistent, monotonic gains and superiority at equal compute (Figures 3–4; Table 3).
  - Mixed for reasoning-heavy tasks: benefits depend on LM capability and datastore domain coverage (Figure 4 right; §4.2; §6).

## 6. Limitations and Trade-offs
- Assumptions and dependence on components (§6; §5.2–§5.3)
  - The LM must be capable enough to use retrieved evidence (Figure 4 right); weaker models may not convert retrieval into reasoning gains.
  - Results hinge on retriever quality; notable headroom remains between cross‑encoder reranking and the lexical oracle (Figure 6).
  - Post‑hoc decontamination is applied to retrieved candidates; while provably equivalent to global decontamination under their setup (Lemma A.2), it relies on retrieving a large enough `K`.

- Coverage and data quality (§6)
  - `MASSIVEDS` is broad but may still lack the specialized content needed for some reasoning tasks (e.g., textbooks for `MMLU`, biomedical knowledge for `MedQA`; §4.2, Finding 5).

- Compute and latency trade‑offs (§4.3, “Discussion on inference cost”)
  - Training compute improves with retrieval, but inference can be costlier: longer prompts (retrieved docs) and retrieval latency. However, switching from a larger to a smaller LM partly offsets this.

- Methodological constraints (§4.3)
  - Compute‑optimal curves use intermediate checkpoints; the training schedules are not re‑tuned per token budget, so some points may be suboptimal (Appendix B.4).

- Reproducibility and scope
  - Full scaling with many retrievers would require re‑indexing; this study primarily uses `CONTRIEVER` (Appendix E.1).
  - Downstream evaluations focus mainly on QA and short‑form answers (§6), not long‑form generation or complex math proofs.

## 7. Implications and Future Directions
- How this changes the landscape
  - Retrieval datastore size should be treated as a first‑class scaling axis alongside parameters and pretraining tokens. For knowledge‑intensive tasks, investing compute in a large datastore plus a capable retriever can be more efficient than further pretraining (Figures 1, 4; §4.3).

- Practical applications
  - Domain adaptation without retraining: swap or extend datastores for new domains (legal, medical, code) while keeping the LM fixed.
  - Compliance and attribution: retrieval enables citing sources and controlling data provenance (highlighted in §1–§2).
  - Cost‑sensitive deployment: run smaller LMs with large datastores for competitive performance on factual QA.

- Research directions
  - Better retrievers and rerankers: the gap to the lexical oracle (Figure 6) suggests substantial unrealized gains.
  - Datastore curation: add targeted high‑quality sources (e.g., textbooks, curated biomedical corpora) for reasoning‑heavy tasks (§4.2, Finding 5).
  - Inference efficiency: optimize retrieval latency and context usage (e.g., compression, selective augmentation; see cited works in §4.1 and §4.3).
  - End‑to‑end scaling studies: jointly choose LM size, pretraining tokens, and datastore scale to meet compute and latency budgets, and extend compute‑optimal analysis to inference cost (§4.3).
  - Broader evaluation: long‑form generation, fact‑checking with citations, and mathematical reasoning at scale (§6).

> Key takeaway (Figure 4; §4.3): For the same training compute, retrieval‑augmented systems reach better Pareto points than LM‑only models, and the retrieval scaling curve shows no sign of saturating within a 1.4T‑token datastore.

> Resource: MASSIVEDS (raw passages, embeddings, and index) and the full scaling pipeline are open‑sourced: https://github.com/RulinShao/retrieval-scaling (abstract; §6).
