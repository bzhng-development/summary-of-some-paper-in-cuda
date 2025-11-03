# Gemini Embedding: Generalizable Embeddings from Gemini

**ArXiv:** [2503.07891](https://arxiv.org/abs/2503.07891)
**Authors:** Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Koert Chen, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, Tom Duerig
**Institutions:** 

## 🎯 Pitch

Gemini Embedding introduces a groundbreaking method for creating versatile text embeddings by leveraging Google's Gemini LLM both as an initializer and data curator. This innovation achieves unparalleled performance across languages and tasks while reducing the need for multiple models, significantly streamlining search engines, multilingual systems, and code retrieval processes.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces Gemini Embedding, a single, general‑purpose text embedding model initialized from Google’s Gemini LLM and trained to produce high‑quality vector representations across languages, tasks, and code. It combines LLM‑guided data generation/filtering, a two‑stage training recipe, and model‑soup averaging to achieve state‑of‑the‑art results on the Massive Multilingual Text Embedding Benchmark (MMTEB) and strong cross‑lingual retrieval, enabling one embedding space to power classification, clustering, ranking, retrieval, and code search.

## 2. Context and Motivation
- Problem addressed
  - General‑purpose embeddings should place semantically similar texts near each other in vector space and support many tasks (retrieval, clustering, classification). However, existing models often overfit to particular benchmarks or languages and struggle to generalize across task types and the long tail of languages.
- Why it matters
  - Practical impact: embeddings are precomputable, cacheable building blocks for search engines, recommendation, deduplication, analytics, multilingual assistants, and code retrieval. A single model that works well for many languages and tasks lowers engineering and operational cost.
  - Scientific impact: demonstrates how to leverage a strong LLM as both initializer and data curator for broad generalization.
- Prior approaches and gaps
  - Encoder‑based models like USE, LaBSE, Sentence‑T5, GTR, E5 and modern Mistral‑based embedders (E5‑Mistral, SFR‑Mistral, BGE‑ICL, NV‑Embed) improved quality but often rely on extensive in‑domain datasets and can overfit to specific benchmarks (Section 2).
  - Some works initialize from large decoder LLMs (e.g., GPT-3 embeddings) but require heavy compute and do not always handle multilingual and task diversity robustly (Section 2).
- Positioning
  - Gemini Embedding starts from Gemini’s multilingual/code knowledge and uses Gemini again to curate training data (synthetic generation, data filtering, hard‑negative mining) and then trains a single model to excel across heterogeneous tasks (Sections 3–4).

## 3. Technical Approach
Terminology used once:
- `embedding`: a dense vector representing input text such that cosine similarity reflects semantic closeness.
- `hard negatives`: texts that are close to the query in meaning but are not correct answers; they force the model to learn fine distinctions.
- `MRL (Matryoshka Representation Learning)`: training a single embedding to work at multiple output sizes by applying separate losses to nested prefixes of the embedding vector.

Step‑by‑step methodology

1) Architecture (Section 3.1; Figure 1)
- A single bidirectional Transformer `M` (initialized from Gemini) encodes sequences.
- Token representations are reduced by mean pooling: average the token embeddings across the sequence.
- A linear projection `f` maps the pooled vector to the final embedding.
  - Output dimension is 3,072, with built‑in support for 768 and 1,536 via MRL (Section 3.2).
- One shared encoder (“siamese” setup) is used for both queries and passages.

2) How embeddings are computed (Equation 1)
- Each training example has a task string `t` (e.g., “question answering” or “fact checking”), a query `q_i`, a positive passage `p_i+`, and optionally a hard negative `p_i−`.
- Embeddings:
  - q_i = f(mean_pool(M(t ⊕ q_i)))
  - p_i± = f(mean_pool(M(p_i±)))
  - The task string prepended to the query acts like a prompt to steer representations to the right sub‑task.

3) Training objective (Equations 2–3; Section 3.2)
- Noise‑Contrastive Estimation (NCE) with in‑batch negatives: for each query, the model maximizes similarity to its positive and minimizes similarity to other in‑batch positives (and an optional hard negative).
- Similarity is cosine similarity, scaled by temperature τ.
- A `mask(i, j)` term excludes trivial duplicates (same query or same positive across examples) from the denominator to avoid false negatives, which is especially important in classification where label sets are small.
- Design choice: unlike Gecko, “same‑tower negatives” are omitted after finding they reduce performance due to false negatives (Section 3.2).

4) Multi‑dimension support with MRL (Section 3.2)
- The 3,072‑dimensional vector is trained with multiple simultaneous losses over overlapping sub‑vectors (first 768 dims, first 1,536 dims, full 3,072). This makes the single model usable at different dimensions without retraining.

5) Two‑stage training recipe (Section 3.3)
- Pre‑finetuning:
  - Large‑scale adaptation from autoregressive generation to encoding using simple (query, positive) pairs and large batches.
  - Uses title–passage pairs from a billion‑scale web corpus; no hard negatives; many more steps than the second stage.
- Finetuning:
  - Train on diverse mixtures of datasets containing (query, positive, hard negative) triples.
  - Use smaller batches (<1024), and keep each batch from a single dataset to make in‑batch negatives meaningfully “similar.”
  - Grid‑search mixture weights and hyperparameters to create several strong checkpoints.
- Model Soup:
  - Average parameters across multiple fine‑tuning runs (and sometimes checkpoints within a run) to improve generalization without increasing inference cost (Section 3.3).

6) Data pipeline and LLM‑assisted curation (Section 4)
- Pre‑finetuning data: scale up diversity with noisy title–passage web pairs.
- Finetuning mixtures:
  - Three families for (i) task diversity, (ii) language diversity, and (iii) code retrieval.
  - To reduce leakage/benchmark overfitting, many in‑domain MTEB datasets are intentionally excluded (Section 4.1).
- Using Gemini to improve training data (Section 4.2):
  - Synthetic data generation:
    - Retrieval: enhanced FRet and SWIM‑IR pipelines with few‑shot prompting; a Gemini “auto‑rater” filters poor synthetic queries.
    - Classification: multi‑stage prompting (e.g., generate users/products/movies first; then generate reviews or counterfactuals) and sample from the long tail of generations to boost diversity.
  - Data filtering:
    - Use few‑shot prompting to judge and remove noisy or mislabeled retrieval examples in human‑annotated datasets.
  - Hard negative mining:
    - Train an initial embedding model without hard negatives; retrieve top‑k nearest neighbors for each query; rescore candidates with Gemini using two prompts (graded classification and query likelihood); combine scores via Reciprocal Rank Fusion (RRF); pick the lowest‑scoring nearest neighbors as hard negatives.

Practical example (how retrieval works at inference time)
- Encode a query and all candidate passages with the same encoder, compute cosine similarities, and rank. Figure 2 shows that the model can encode an Assamese or Hindi query and retrieve relevant English passages without translation.

## 4. Key Insights and Innovations
- LLM‑as‑initializer and LLM‑as‑data‑curator in one pipeline
  - Different from prior embedders that either initialize from smaller encoders or from LLMs without heavy LLM‑guided curation, this work uses Gemini both to initialize the encoder and to synthesize/filter data and mine hard negatives (Sections 3–4). This synergy is central to the model’s generalization across tasks and languages.
- Task‑prompted, single‑tower encoder with selective negatives
  - Encoding queries with explicit task strings (Equation 1) and omitting “same‑tower negatives” (Section 3.2) are simple but impactful design choices that reduce false negatives and let one model cover retrieval, classification, clustering, and reranking.
- Multi‑resolution embeddings via MRL
  - A single checkpoint supports 768, 1,536, and 3,072 dimensions (Section 3.2). This is practically significant: the same API can serve latency‑sensitive scenarios at 768‑d and high‑accuracy scenarios at 3,072‑d.
- Two‑stage training + model soup for generalization
  - Pre‑finetuning aligns the LLM encoder to embedding behavior at scale; dataset‑wise batching in finetuning sharpens task signals; parameter averaging (model soup) increases robustness without inference overhead (Section 3.3).
- Empirical finding: task diversity > language diversity in finetuning
  - Table 6 shows the “English‑only (diverse tasks)” mixture generalizes surprisingly well to multilingual evaluations, whereas “Multilingual‑only (retrieval)” lags on overall MMTEB due to limited task types.

## 5. Experimental Analysis
Evaluation setup
- Benchmarks and tasks (Section 5.1)
  - MMTEB (Enevoldsen et al., 2025): 164 evaluation tasks total: 132 multilingual tasks, 41 English v2 tasks, 12 code retrieval tasks.
  - XOR‑Retrieve: cross‑lingual retrieval where queries are in 7 languages, passages in English.
  - XTREME‑UP: 20 under‑represented languages; queries in those languages, passages in English.
- Metrics
  - MMTEB: “Task Mean” (average over tasks), “Task Type Mean” (average over task types), and Borda rank (official leaderboard metric).
  - XOR‑Retrieve: Recall@5k_t (Table 1).
  - XTREME‑UP: MRR@10 (Tables 1 and 5).
- Baselines
  - Multiple strong embedders including gte‑Qwen2‑7B‑instruct, multilingual‑e5‑large‑instruct, Cohere‑embed‑multilingual‑v3.0, Gecko embeddings, NV‑Embed v1/v2, and others (Tables 1–4).

Main quantitative results
- Overall SOTA across three MMTEB tracks (Table 1)
  - “MTEB(Multilingual) Task Mean”:
    > Gemini Embedding: 68.32 vs multilingual‑e5‑large‑instruct: 63.23; gte‑Qwen2‑7B‑instruct: 62.51.
  - “Task Type Mean”:
    > 59.64 vs 56.00 (gte‑Qwen2‑7B‑instruct) and 55.17 (multilingual‑e5‑large‑instruct).
  - MTEB(Eng, v2) Task Mean:
    > 73.30 vs 69.53–70.72 for top baselines.
  - MTEB(Code) Mean:
    > 74.66 vs 65.40 (gte‑Qwen2‑7B‑instruct) and 58–59 for several others.
  - Cross‑lingual:
    > XOR‑Retrieve Recall@5k_t: 90.42 vs 68.76 (text‑embedding‑3‑large).
    > XTREME‑UP MRR@10: 64.33 vs 34.97 (gte‑Qwen2‑7B‑instruct) and 18–19 for some other baselines.
- Task‑type breakdown on MTEB(Multilingual) (Table 2)
  - Gemini Embedding achieves Borda rank #1 and leads strongly on:
    > Classification: 71.8 (+9.6 over next best 62.2),
    > Clustering: 55.0 (+3.7 over next best 51.3),
    > Retrieval: 67.7 (+9.0 over next best 58.7).
- Task‑type breakdown on MTEB(Eng, v2) (Table 3)
  - Borda rank #1 with large gains:
    > Classification: 90.1 (+7.1 over #2),
    > Clustering: 59.4 (+5.3),
    > Retrieval: 87.7 (+4.3).
- Code retrieval (Table 4)
  - Across eight code tasks:
    > Mean All: 75.5 (Gemini Embedding) vs 62.9 (next), with strong AppsR (93.8), CSNR (84.7), CTOC (91.3).
- Cross‑lingual qualitative (Figure 2)
  - Queries in Assamese and Hindi (with a typo) retrieve correct English passages, showing robustness to language and noise.

Ablations and robustness checks
- Training mixtures (Table 6)
  - Pre‑finetuning only:
    > MTEB(Multilingual) Task Mean 48.89 (vs 68.32 final).
  - No training (Gemini init only):
    > 30.55 — confirms training is essential.
  - English‑only finetuning (diverse tasks):
    > 66.75 on MTEB(Multilingual) and 85.70 on XOR‑Retrieve—surprisingly strong multilingual generalization.
  - Multilingual‑only finetuning (retrieval tasks only):
    > Best on XTREME‑UP (65.06) but weaker on overall MMTEB—task diversity matters.
- Synthetic classification data (Table 7)
  - Training on Gemini‑generated classification datasets yields:
    > Average +17.6 point gain (75.17 vs 57.57 without synthetic), with large boosts on AmazonCounterfactual (91.30) and AmazonReviews (96.51).
- LLM‑based filtering for retrieval (Table 8; MIRACL)
  - Filtering raises average from 59.8 to 63.7 (+3.9), with consistent improvements across many languages.
- Hard negatives (Figure 3)
  - Adding a few Gemini‑selected hard negatives improves nDCG@10 on FEVER, HotpotQA, NQ, SciFact; too many hard negatives overfit and degrade performance—there is a sweet spot.

Do results support the claims?
- Yes. The combination of cross‑benchmark SOTA (Table 1), fine‑grained leaderboards (Tables 2–4), cross‑lingual wins (Tables 1 and 5), and ablations on training stages, mixtures, synthetic data, filtering, and negatives (Tables 6–8; Figure 3) credibly supports the central claim of strong, generalizable embeddings. Full per‑task scores in Tables 9–11 provide additional transparency.

Where results are mixed
- Some instruction‑retrieval tasks remain challenging: e.g., “Robust04InstructionRetrieval: −2.41” in Table 9.
- Multilabel classification lags absolute scores compared to single‑label classification on MMTEB(Multilingual) (Table 1: 29.16 vs 71.84).
- XTREME‑UP long‑tail languages still include low MRRs (e.g., brx: 25.66 in Table 11).

## 6. Limitations and Trade-offs
- Dependence on a large proprietary LLM
  - Initialization, synthetic data generation, filtering, and hard negative mining all rely on Gemini. Reproducibility outside Google may be limited without access to similar LLM capabilities and compute (Sections 3–4).
- Compute and data intensity
  - Pre‑finetuning uses a billion‑scale web corpus with large batch sizes and “substantially greater number of steps” (Section 3.3). This implies high compute and memory costs.
- Sensitivity to hard negative count
  - Figure 3 shows overfitting if too many hard negatives are used; tuning is required per dataset.
- Uneven performance across task types and languages
  - Instruction retrieval remains low in absolute terms across models (Table 1), and some under‑represented languages still see modest MRR (Table 11).
- Limited modality coverage (current work is text/code)
  - Section 7 explicitly treats multimodal embeddings as future work.
- Mixture design and model soup selection
  - Finetuning involves grid search and manual checkpoint selection for soup ingredients (Section 3.3). This engineering know‑how may be non‑trivial to replicate and raises questions about sensitivity to mixture choices.
- Potential dataset bias and synthetic artifacts
  - While LLM‑based filtering and rating help, synthetic data choices (e.g., prompts, sampling) can imprint LLM biases; the paper notes diversity strategies but does not quantify bias shifts.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates a practical recipe to turn a strong multilingual LLM into a highly generalizable embedding model: initialize from the LLM, adapt with pre‑finetuning, curate data with the LLM, and stabilize with model soup. The result is a single model that is SOTA on multilingual, English, and code tracks simultaneously (Table 1).
- Follow‑up research enabled/suggested
  - Multimodal embeddings: Section 7 proposes extending the method to images, video, and audio using Gemini’s multimodal capabilities, training a single space that supports various modality pairings.
  - Better negative sampling and regularization: mitigate overfitting when scaling the number/strength of hard negatives (Figure 3).
  - Systematic studies on task prompts: quantify how much prompt phrasing contributes to gains across task types.
  - Bias and fairness analysis: leverage synthetic pipelines for counterfactual and debiasing data generation (Table 7 hints at controllable generation).
  - Open benchmarking on long‑tail languages and low‑resource tasks: XTREME‑UP results (Table 5/11) are strong on average but still uneven across some languages.
- Practical applications
  - Search and retrieval across 100+ languages and codebases (Figure 2; Tables 1, 4).
  - Classification and clustering at scale (Tables 2–3).
  - Cross‑lingual assistants: queries in under‑represented languages retrieving English resources (Tables 5 and 11).
  - Enterprise analytics: deduplication, topic discovery, and document routing, with dimensionality trade‑offs enabled by MRL (Section 3.2).

In short, the paper provides a replicable blueprint—albeit compute‑heavy—for converting a large multilingual LLM into a unified, high‑performing embedding model by combining LLM initialization, LLM‑assisted data curation, staged contrastive training, multi‑resolution learning, and weight averaging, and then validates it across an unusually broad and rigorous set of benchmarks.
