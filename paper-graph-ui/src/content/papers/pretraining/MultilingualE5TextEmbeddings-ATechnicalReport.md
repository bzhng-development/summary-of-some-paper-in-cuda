# Multilingual E5 Text Embeddings: A Technical Report

**ArXiv:** [2402.05672](https://arxiv.org/abs/2402.05672)

## 🎯 Pitch

This report introduces the multilingual E5 text embedding models—a family of open-source models trained on an unprecedented scale of 1 billion multilingual text pairs, followed by supervised fine-tuning, and including an instruction-tuned variant. By demonstrating state-of-the-art or highly competitive performance across English and over 100 languages on retrieval, semantic similarity, and bitext mining tasks, these models bridge a crucial gap for real-world multilingual information retrieval systems—empowering global applications where language diversity and performance at scale are essential.

---

## 1. Executive Summary (2-3 sentences)
This report releases a family of multilingual text embedding models (`mE5-small`, `mE5-base`, `mE5-large`) and an instruction-tuned variant (`mE5-large-instruct`) trained with a two-stage pipeline: large-scale multilingual contrastive pre-training on ~1 billion text pairs, followed by supervised fine-tuning on ~1.6 million labeled examples. The models deliver state-of-the-art or highly competitive performance on English and multilingual benchmarks, notably surpassing strong English-only and multilingual baselines on the English portion of MTEB and achieving strong cross-lingual retrieval and bitext mining results (Tables 3–5).

## 2. Context and Motivation
- The specific gap addressed:
  - Most widely used text embedding models are trained only on English (e.g., Sentence-BERT, GTR/ColBERT variants), which limits their utility in multilingual retrieval, similarity, and clustering tasks.
  - Translation-pair-focused multilingual models (e.g., `LaBSE`) excel at matching translations but are less optimized for the broad mix of tasks encountered in retrieval-augmented systems.

- Why this matters:
  - Text embeddings underpin search engines, recommendation systems, and retrieval-augmented language models (RAG). In multilingual settings (global search, cross-lingual customer support, knowledge base retrieval), English-only embeddings degrade performance and coverage.

- Prior approaches and their limitations:
  - English-only dual encoders are strong on English benchmarks but do not generalize across languages without machine translation.
  - `LaBSE` (trained on translation pairs) is great for bitext mining but not tuned broadly for retrieval/similarity across varied tasks and domains.

- Positioning of this work:
  - Extends the English E5 recipe to multilingual contexts by:
    - Scaling weakly-supervised contrastive pre-training to ~1B multilingual pairs (Table 1).
    - Supervised fine-tuning with task-diverse multilingual datasets (Table 2), including hard negatives and cross-encoder distillation.
    - Releasing an instruction-tuned embedding model (`mE5-large-instruct`) that incorporates synthetic, multilingual instruction data (Wang et al., 2023), improving task adherence and overall quality.

## 3. Technical Approach
This section explains HOW the models are built and trained, step-by-step.

- Model family and initial backbones (Appendix: Training Hyperparameters):
  - `mE5-small`: initialized from multilingual `MiniLM`.
  - `mE5-base`: initialized from `xlm-roberta-base`.
  - `mE5-large`: initialized from `xlm-roberta-large`.
  - Rationale: start from strong multilingual encoders to ensure broad language coverage and transfer.

- Stage 1 — Weakly-supervised contrastive pre-training (Section 2; Table 1; Appendix A):
  - Goal: Learn a universal embedding space where semantically related texts (across many languages) are close.
  - Core mechanism: Contrastive learning with `InfoNCE` loss using only `in-batch negatives`.
    - `InfoNCE`: encourages a model to assign high similarity to the true pair (anchor, positive) and lower similarity to all other (anchor, negative) pairs in the batch.
    - `In-batch negatives`: the other examples in the same training batch serve as negatives; this scales contrastive learning efficiently without external negative mining during pre-training.
  - Scale and setup:
    - Batch size: 32k; Steps: 30k → roughly “goes over ∼ 1 billion text pairs.”
    - Hyperparameters: LRs of {3, 2, 1}×10^-4 for {small, base, large}.
  - Data construction for pairs (Table 1, Appendix A):
    - Wikipedia: `(section title, section passage)`
    - mC4: `(title, page content)`
    - Multilingual CC News: `(title, news content)`
    - NLLB: translation pairs
    - Reddit: `(comment, response)`
    - S2ORC: `(title, abstract)` and citation pairs
    - Stackexchange: `(question, answer)`
    - xP3 (multitask prompts): `(input prompt, response)`
    - Misc. SBERT training data: a collection of diverse datasets (e.g., SimpleWiki, AGNews, Specter, XSum) to broaden task coverage
  - Why this design:
    - Diverse pair types go beyond translation alignment; they expose the model to many “semantic relations” (titles-to-contents, questions-to-answers, comments-to-replies), making the embedding space useful for retrieval, similarity, and clustering—not just direct translation matching.

- Stage 2 — Supervised fine-tuning (Section 2; Table 2; Appendix A):
  - Goal: Improve task adherence and retrieval quality beyond general pre-training.
  - Methods:
    - Incorporate `mined hard negatives`: non-matching texts that are deceptively similar to the query; learning to push these away refines ranking quality.
    - `Knowledge distillation` from a `cross-encoder` teacher: a cross-encoder jointly reads query and candidate text to produce a high-quality relevance score; using it as a teacher helps the embedding model (bi-encoder) approximate strong relevance judgments while retaining efficient inference.
  - Data mixture (~1.6M examples; Table 2):
    - Retrieval and QA-related: MS MARCO Passage (500k), MS MARCO Document (70k), NQ, TriviaQA, SQuAD (220k), DuReader Retrieval (86k), HotpotQA (70k), MIRACL (40k), Mr. TyDi (50k)
    - Reasoning/fact checking: FEVER (70k)
    - NLI: 275k
    - Community QA & duplicates: ELI5 (100k), Quora Duplicate Questions (15k)
    - NLLB: 100k (cross-lingual signal)
  - Hyperparameters: batch size 512; LRs {3, 2, 1}×10^-5 for {small, base, large}; 2 epochs.

- Instruction-tuned variant (`mE5-large-instruct`) (Section 2):
  - What is instruction tuning for embeddings? The model receives a short natural-language “instruction” that describes the embedding task (e.g., “retrieve passages relevant to this question”), which conditions the embedding to the specific use-case.
  - Data: 500k synthetic, multilingual instruction examples from GPT-3.5/4 with 150k unique instructions across 93 languages, reusing instruction templates from Wang et al. (2023).
  - Why this helps: Instructions explicitly tell the model “what similarity means” for a given task, reducing mismatches between generic embeddings and task-specific needs.
  - Training: same hyperparameters as `mE5-large`, but fine-tuned on the new instruction mixture.

- How the embedding is used at inference time:
  - The model maps any input text (query, sentence, passage) to a dense vector.
  - Retrieval: nearest-neighbor search in the vector space finds semantically similar texts across languages.
  - Similarity and clustering: cosine similarity or other distance metrics compare vectors.

## 4. Key Insights and Innovations
- Large-scale, diverse, multilingual contrastive pre-training beyond translation pairs:
  - Novelty/significance:
    - Rather than relying only on translation pairs (as in `LaBSE`), this approach combines many relation types (title–content, question–answer, comment–reply) and sources (Table 1; Appendix A), cultivating an embedding space that generalizes to retrieval, QA, and clustering.
    - This diversity likely explains the strong multilingual retrieval results in MIRACL (Table 4) and robust bitext mining (Table 5).

- Two-stage pipeline with hard negatives and cross-encoder distillation:
  - Distinguishing factor:
    - Supervised fine-tuning uses both mined hard negatives and a cross-encoder teacher (Section 2), a combination known to produce sharper ranking behavior than contrastive learning alone.
  - Impact:
    - Large gains over BM25 and `mDPR` on MIRACL (Table 4) suggest the ranking refinement is effective.

- Instruction-tuned multilingual embeddings:
  - Difference from prior work:
    - Instruction-tuned embedding models have gained traction in English; here, the variant covers 93 languages with synthetic instructions (Section 2).
  - Impact:
    - On the English portion of MTEB (Table 3), `mE5-large-instruct` achieves 64.4 average, edging out strong baselines like `BGE-large-en-v1.5` (64.2) and `Cohere-multilingual-v3` (64.0).
    - In bitext mining, `mE5-large-instruct` reaches 99.0 (BUCC) and 83.8 (Tatoeba), surpassing `LaBSE` (98.8, 81.1) in both (Table 5).

- Practical model size spectrum:
  - Contribution:
    - Releasing `small`, `base`, and `large` models offers flexible trade-offs between speed/memory and quality (Table 3, Table 4).
  - Impact:
    - For many production systems, a strong `base` model can be preferable if it achieves most of the quality at significantly lower cost.

## 5. Experimental Analysis
- Evaluation methodology:
  - English MTEB (Muennighoff et al., 2023) — multi-task benchmark of 56 datasets; reported as a single average score per model (Table 3). Detailed per-dataset results are in Appendix Table 7 (e.g., STS, classification, clustering, and retrieval tasks).
  - MIRACL multilingual retrieval (Zhang et al., 2023) — evaluated on 16 languages with ranking metrics `nDCG@10` and `R@100` (Table 4, Appendix Table 6).
    - `nDCG@10`: a ranking quality metric emphasizing correct items at higher ranks (normalized to facilitate cross-dataset comparison).
    - `R@100` (Recall@100): fraction of relevant items retrieved in the top 100.
  - Bitext mining (BUCC 2018 across 4 languages; Tatoeba across 112 languages) — cross-lingual sentence pair retrieval where matched pairs often have little lexical overlap (Table 5).

- Baselines and comparators:
  - English MTEB: `LaBSE` (trained on translation pairs), `Cohere-multilingual-v3` (details not fully public), and `BGE-large-en-v1.5` (English-only).
  - MIRACL: sparse BM25 and dense `mDPR` (fine-tuned on MIRACL).
  - Bitext mining: `mContriever-msmarco` and `LaBSE`.

- Main results and comparisons:
  - English MTEB (Table 3):
    - > “mE5-large-instruct 64.4; BGE-large-en-v1.5 64.2; Cohere-multilingual-v3 64.0; mE5-large 61.5; mE5-base 59.5; mE5-small 57.9.”
    - Takeaway: a multilingual, instruction-tuned model matches or slightly exceeds strong English-only and multilingual baselines on a broad English suite.
    - Per-task nuances (Appendix Table 7):
      - Instruction tuning strongly boosts many STS and classification tasks (e.g., `STS13` from 81.5 to 87.2; `AmazonPolarity` from 93.5 to 96.3).
      - Some retrieval tasks see slight regressions with instruction tuning (e.g., `MSMARCO` mE5-large 43.7 vs `mE5-large-instruct` 40.4; `NQ` 64.1 vs 57.8), indicating a trade-off between task-general gains and certain retrieval-style tasks.

  - MIRACL multilingual retrieval (Table 4; averaged over 16 languages):
    - > “BM25: 39.3 nDCG@10 / 78.7 R@100; mDPR: 41.5 / 78.8.”
    - > “mE5-small: 60.8 / 92.4; mE5-base: 62.3 / 93.1; mE5-large: 66.5 / 94.3; mE5-large-instruct: 65.7 / 94.6.”
    - Takeaway: all `mE5` models far exceed both BM25 and `mDPR` on both metrics, with `mE5-large` achieving the best `nDCG@10` and `mE5-large-instruct` slightly higher `R@100`.
    - Per-language detail (Appendix Table 6) shows consistent gains; examples:
      - `zh` (Chinese): `nDCG@10` from 45.9 (`small`) → 56.0 (`large`) → 56.2 (`large-instruct`).
      - `ar` (Arabic): 71.4 → 76.0 → 76.8; `R@100` reaches 97.5–97.6 in top models.
      - These illustrate both cross-lingual strength and the quality-size scaling trend.

  - Bitext mining (Table 5):
    - > “BUCC (4 langs): mE5-large-instruct 99.0; LaBSE 98.8; mE5-large 98.6; mE5-base 98.1; mE5-small 93.2; mContriever-msmarco 93.7.”
    - > “Tatoeba (112 langs): mE5-large-instruct 83.8; LaBSE 81.1; mE5-large 75.7; mE5-base 68.1; mE5-small 64.2; mContriever-msmarco 37.7.”
    - Takeaway: `mE5-large-instruct` surpasses `LaBSE` on both datasets, showing that multilingual instruction tuning and broad pre-training can rival and exceed specialized translation-focused models on cross-lingual matching.

- Do the experiments support the claims?
  - Strengths:
    - Broad benchmarks across 100+ languages/tasks, strong numerical results, and clear scaling trends (Tables 3–6).
    - Instruction tuning yields consistent gains on many tasks and does not harm multilingual retrieval overall.
  - Caveats:
    - MIRACL is included in fine-tuning (Table 2: 40k), and the evaluation is on the MIRACL dev set (Table 4). While this reflects a realistic training regimen (multi-dataset fine-tuning), it is not pure zero-shot on MIRACL.
    - No ablations isolate how much each component (e.g., hard negatives, distillation, specific datasets) contributes to gains.

## 6. Limitations and Trade-offs
- Training cost and complexity:
  - Contrastive pre-training at batch size 32k over 30k steps (~1B pairs) requires significant compute and infrastructure (Section 2), which may limit reproducibility for smaller labs.

- Negative sampling choices:
  - Pre-training uses only `in-batch negatives`; while efficient, it may under-sample “hard” negatives during this stage. The approach compensates in fine-tuning by mining hard negatives, but an ablation comparing alternative negative strategies is absent.

- Data mixture and evaluation coupling:
  - Fine-tuning includes MIRACL and Mr. TyDi (Table 2), which can help multilingual retrieval but complicates pure generalization claims on those benchmarks.

- Instruction-tuning trade-offs:
  - `mE5-large-instruct` improves the MTEB average and bitext mining but shows regressions on some retrieval tasks (Appendix Table 7: `MSMARCO`, `NQ`, `HotpotQA`, `FEVER` retrieval). This suggests instruction conditioning can bias embeddings toward certain task notions of similarity at the expense of others unless carefully balanced.

- Coverage and domains:
  - While language coverage is broad (over 90 languages via instruction data; Table 5 shows 112 langs in Tatoeba), the report focuses on general-domain tasks. Highly specialized domains (legal, biomedical beyond SciDocs) may still need domain-specific fine-tuning.

- Baseline comparability:
  - Limited public details for `Cohere-multilingual-v3` (Table 3 note), making exact apples-to-apples comparisons difficult.

- Ethical/data concerns:
  - Synthetic instruction data from GPT-3.5/4 may inherit biases or stylistic artifacts; the report does not include audits of bias, toxicity, or fairness.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Demonstrates that a single multilingual embedding model can be competitive with top English-only models on English tasks (Table 3) while excelling in multilingual retrieval and cross-lingual matching (Tables 4–5). This reduces the need for English-only embeddings plus translation pipelines.

- Practical applications:
  - Multilingual search and retrieval: cross-language question answering, knowledge base lookups, support ticket routing.
  - Retrieval-augmented generation (RAG): language-agnostic document retrieval feeding LLMs.
  - Bitext mining at scale: corpus alignment, machine translation training data mining (Table 5).
  - Deduplication and clustering across languages: content moderation, topic tracking, and news aggregation.

- Concrete guidance for deployment:
  - Choose model size based on latency and memory budgets:
    - `mE5-small`/`base` provide strong quality-speed trade-offs (Tables 3–4).
    - `mE5-large` yields best multilingual retrieval `nDCG@10` (Table 4).
  - Prefer `mE5-large-instruct` when:
    - You can provide clear task instructions with the text to embed.
    - Your pipeline benefits from better task adherence across diverse tasks and languages, and minor retrieval trade-offs are acceptable (Appendix Table 7).
  - Use multilingual evaluation sets similar to your target languages (Appendix Table 6 shows variance across languages; ensure coverage for your user base).

- Research directions:
  - Ablations and component analysis:
    - Quantify contributions of each dataset, hard-negative mining, and cross-encoder distillation.
    - Study the sensitivity of instruction templates and their multilingual phrasing.
  - Dynamic task conditioning:
    - Methods to interpolate between generic and instruction-tuned embeddings at inference time to avoid regressions on some retrieval tasks.
  - Negative sampling improvements:
    - Memory-bank or curriculum-based negatives during pre-training to complement in-batch negatives.
  - Fairness and bias evaluation:
    - Systematic audits across languages to detect and mitigate cultural/linguistic biases.
  - Domain specialization:
    - Lightweight adapters or LoRA for domain-specific multilingual embeddings without full retraining.

Key citations within the report:
- Training data mixtures: Table 1 (pre-training), Table 2 (fine-tuning); Appendix A (pair construction and full dataset list).
- English MTEB results: Table 3 and Appendix Table 7 (per-dataset).
- MIRACL multilingual retrieval: Table 4 (averages), Appendix Table 6 (per language).
- Bitext mining: Table 5 (BUCC, Tatoeba).
- Training hyperparameters and initialization: Appendix A (Training Hyperparameters).
