# Gemini Embedding: Generalizable Embeddings from Gemini

**ArXiv:** [2503.07891](https://arxiv.org/abs/2503.07891)

## 🎯 Pitch

Gemini Embedding introduces a unified embedding model derived from Google’s Gemini large language model, capable of producing state-of-the-art representations for text and code across over 250 languages and diverse task types. By combining LLM-driven data curation, extensive multilingual and multitask training, and effective model ensemble techniques, it sets a new standard for general-purpose embeddings—eliminating the need for multiple task- or language-specific models. This innovation is crucial, as it empowers a wide spectrum of real-world applications—ranging from retrieval and classification to clustering—with robust, reusable, and precomputable embeddings that generalize reliably across domains, languages, and modalities.

---

## 1. Executive Summary
Gemini Embedding is a single, general-purpose text (and code) embedding model initialized from Google’s Gemini LLM and trained with a two-stage, LLM-in-the-loop pipeline. It achieves state-of-the-art results across multilingual, English, and code benchmarks by combining simple architecture choices (mean pooling, cosine similarity) with careful data curation (synthetic data, LLM-based filtering, and hard-negative mining) and “model soup” parameter averaging.

It matters because many applications—from search and retrieval to clustering and classification—depend on robust embeddings that generalize across tasks, domains, and 250+ languages; this model provides a unified, strong default without task- or language-specific variants.

## 2. Context and Motivation
- Problem addressed
  - Build a single, robust embedding model that generalizes across:
    - Many task types (retrieval, classification, STS/similarity, clustering, reranking, etc.),
    - Many languages (250+ in MMTEB),
    - And code.
  - The model should be practical: embeddings can be precomputed and reused in latency-sensitive systems (Section 1; Figure 1).

- Why it is important
  - Real-world systems require consistent behavior across heterogeneous inputs and languages, often without per-task fine-tuning.
  - Previous “general-purpose” embedders often performed very well on some benchmarks but failed to generalize (overfitting, domain bias) or needed separate models for English, multilingual, and code (Section 2).

- Prior approaches and their gaps
  - Classic encoder backbones (e.g., BERT, T5) adapted for embeddings: Sentence-BERT, LaBSE, GTR, E5 (Section 2). These struggle with broad task and language transfer, especially on new evaluation suites like MMTEB.
  - LLMs used either:
    1) To generate or curate training data (hard-negative mining, synthetic data), or
    2) As initialization for embedding models (e.g., GPT-3-based embeddings, Mistral-based embedders) (Section 2).
  - However, many recent models rely on in-domain training data that inflate benchmark scores but reduce out-of-domain generalization (Section 2; “overfitting to specific benchmarks”).

- Positioning of this work
  - Initializes from Gemini (a strong, multilingual/code-capable LLM), then:
    - Uses Gemini to curate training data: synthetic data generation, quality filtering, and hard-negative mining (Sections 4.2).
    - Employs a two-stage training recipe (pre-finetune → fine-tune) and “Model Soup” averaging for generalization (Section 3.3).
  - Intentionally excludes many in-domain MTEB datasets from training to avoid leakage and benchmark overfitting (Section 4.1, Fine-tuning).

## 3. Technical Approach
At a glance: a same-tower encoder (query and passage share parameters) initialized from Gemini, mean-pooling, cosine similarity, contrastive learning, and multi-loss training to support multiple embedding sizes.

- Architecture (Section 3.1; Figure 1)
  - Input tokens T are encoded by a transformer `M` (initialized from Gemini) with bidirectional attention, producing token embeddings `T_embed ∈ R^{L×d_M}`.
  - Pooling: simple mean pooling across tokens: `P_embed = mean_pool(T_embed)`.
  - Projection: a linear layer `f` maps to the desired embedding size `d`: `E = f(P_embed)`.
  - Why mean pooling? Prior work finds simple pooling effective when adapting decoder-based LLMs to encoders (cited in Section 3.1), and it avoids extra parameters/complexity.

- Task conditioning via prompts (Equation 1; Section 3.2)
  - Each training example carries a short “task string” `t` (e.g., “question answering”, “fact checking”).
  - The query embedding includes this task string, concatenated to the query text:  
    `q_i = f(mean_pool(M(t ⊕ q_i)))`.  
    The positives/negatives do not include `t`:  
    `p_i^± = f(mean_pool(M(p_i^±)))`.
  - Intuition: conditioning the query on task semantics helps a single model separate “what counts as similar” across different tasks.

- Training objective: NCE with in-batch negatives (Equations 2–3; Section 3.2)
  - Goal in plain language: push the query vector close to its true positive and far from negatives.
  - Loss per query `i` uses cosine similarity with temperature `τ`, considering:
    - The positive `p_i^+`,
    - An optional “hard negative” `p_i^-`, and
    - All other batch positives as additional negatives (in-batch negatives).
  - A mask avoids treating “duplicates” as negatives (Equation 3), which matters when labels are few (e.g., classification).
  - Design choice: unlike Gecko, they omit “same-tower negatives” due to false negative risk in general-purpose settings (Section 3.2).

- Multi-size embeddings via MRL (Section 3.2)
  - Matryoshka Representation Learning (MRL) trains multiple overlapping subspaces simultaneously (e.g., 768, 1536, 3072 dims).
  - In practice: the model exposes 3072-d embeddings, and the first 768 or 1536 dimensions are also trained to be strong. This lets users downscale storage/latency without retraining.

- Two-stage recipe (Section 3.3)
  - Pre-finetuning:
    - Large, noisy, weakly-labeled pairs (e.g., title–passage) at web scale; no hard negatives; very large batch sizes.
    - Purpose: adapt a generation-optimized LLM to an encoder setting and stabilize training.
  - Fine-tuning:
    - Mixtures targeting task diversity (retrieval/classification/etc.), language diversity, and code.
    - Batch construction: small batches (<1024) and single-dataset batches to keep negatives “in-task,” which provides better learning signal.
    - Extensive grid search over hyperparameters and dataset mixture weights.
  - Model Soup:
    - Average parameters across multiple fine-tuning runs/checkpoints to improve generalization (Section 3.3).
    - Includes both “within-run” averaging (SWA-style) and “across-run” soups; final ingredients chosen by manual experimentation.

- Data pipeline (Section 4)
  - Pre-finetuning data: billion-scale web corpus of (title, passage) pairs (Section 4.1).
  - Fine-tuning mixtures:
    - Task-diversity mixture (incl. some academic datasets and synthetic sets),
    - Multilingual retrieval mixture,
    - Code retrieval mixture (Section 4.1).
    - Explicitly excludes many in-domain MTEB datasets to reduce leakage and benchmark overfitting.
  - Gemini-in-the-loop curation (Section 4.2):
    - Synthetic data:
      - Retrieval: Gemini uses few-shot prompting to generate queries for web passages; an “auto-rater” filters low-quality generations. Method builds on FRet and SWIM-IR (Section 4.2).
      - Classification: multi-stage prompting generates counterfactual, sentiment, and review datasets; adds controllable diversity (e.g., sampling the tail of long lists).
    - Data filtering:
      - Gemini evaluates and filters noisy human-annotated retrieval pairs (common issue: wrong positives/negatives).
    - Hard negative mining:
      - Train an initial embedder without hard negatives; retrieve top-k nearest neighbors per query.
      - Have Gemini score each neighbor with two prompts: ‘graded classification’ and ‘query likelihood’; combine with Reciprocal Rank Fusion (RRF).
      - Select the lowest-scoring among the close neighbors as the “hard negative” (Section 4.2).

Definitions (selective):
- `Embedding`: a numeric vector representing text such that semantically similar items are close in vector space.
- `Contrastive learning`: training that pulls matched pairs together and pushes mismatched ones apart.
- `Noise-Contrastive Estimation (NCE) loss`: a contrastive loss that treats non-positives as “noise” to be distinguished from the true positive; here implemented with in-batch negatives (Equation 2).
- `Hard negatives`: near-miss examples that look similar to the query but are actually incorrect; they improve discriminative power.
- `Model Soup`: averaging parameters from several fine-tuned checkpoints to improve generalization without inference overhead.
- `MRR@10` (Mean Reciprocal Rank @ 10): evaluates how high the first correct item appears in a ranked list (higher is better); common in retrieval.
- `Recall@5k`: fraction of queries for which the correct item appears in the top-5000 retrieved items.

## 4. Key Insights and Innovations
- A. Strong generalization through Gemini-initialization plus a simple encoder head
  - What is new: initialize from Gemini (decoder-based LLM) and adapt to a bidirectional encoder with minimal architectural additions (mean-pooling + linear projection).
  - Why it matters: It leverages Gemini’s multilingual and code knowledge to produce robust embeddings across tasks and languages without complex heads (Section 3.1).
  - Evidence: Pre-finetuning alone jumps from “No Training” to strong scores (Table 6: MTEB(Multilingual) 30.55 → 48.89).

- B. LLM-in-the-loop data curation at scale (Section 4.2)
  - Synthetic generation: handles both retrieval and classification; multi-stage prompting yields realistic, diverse data.  
    - Evidence: Table 7 shows large gains on classification when trained on synthetic data:  
      > “Average +17.6” points over w/o synthetic; AmazonCounterfactual 65.43 → 91.30, Emotion 48.70 → 55.90.
  - Filtering: LLM removes mislabeled or low-quality pairs in multilingual retrieval datasets.  
    - Evidence: Table 8 (MIRACL) shows average +3.9 points from filtering (59.8 → 63.7), with broad gains across languages.
  - Hard-negative mining: uses Gemini scoring with RRF to select near-but-wrong neighbors.  
    - Evidence: Figure 3 shows that adding a few hard negatives generally improves nDCG@10 across FEVER, HotpotQA, NQ, SciFact; too many can overfit.

- C. Training recipe that prioritizes task diversity over language diversity during fine-tuning
  - Novel claim: in this setting, task-diverse English-only fine-tuning generalizes surprisingly well to multilingual benchmarks.  
    - Evidence: Table 6: English-only mixture reaches MTEB(Multilingual) 66.75 (close to full model 68.32) and strong XOR-Retrieve 85.70; meanwhile, “Multilingual Only (Retrieval)” achieves XTREME-UP 65.06 (best for long-tail languages) but lags on task-diverse English metrics.
  - Insight: the Gemini foundation handles language transfer; fine-tuning should cover diverse task formats to maximize general-purpose utility.

- D. Multi-resolution embeddings via MRL and generalization via Model Soup
  - MRL: train one model to yield strong 768-, 1536-, and 3072-dim embeddings without separate training runs (Section 3.2).
  - Model Soup: averages multiple fine-tuned runs to improve out-of-sample performance (Section 3.3). This is incremental but impactful, enabling the final state-of-the-art results.

## 5. Experimental Analysis
- Evaluation setup (Section 5)
  - Benchmarks and tasks:
    - MMTEB (MTEB(Multilingual), MTEB(Eng, v2), MTEB(Code)) covering 164 tasks; 10 task types (Bitext Mining, Classification, Clustering, Instruction Retrieval, Multilabel Classification, Pair Classification, Reranking, Retrieval, STS, Summarization).
    - Cross-lingual: XOR-Retrieve (queries in 7 languages, English passages; Recall@5k) and XTREME-UP (20 underrepresented languages, English passages; MRR@10).
  - Baselines: strong public and commercial embedders (e.g., multilingual-e5-large-instruct, gte-Qwen2-7B-instruct, Cohere multilingual-v3.0, Google Gecko family).
  - Metrics:
    - Leaderboards use Task Mean, Type Mean, and Borda rank (official).
    - Retrieval tasks include MRR@10, Recall@K.

- Main results (Tables 1–5)
  - MMTEB overall (Table 1):
    - > “MTEB(Multilingual) Task Mean 68.32” and “Type Mean 59.64,” both SOTA, with a “+5.09” Task Mean gap over the next best (multilingual-e5-large-instruct 63.23).
    - MTEB(Eng, v2) Task Mean 73.30; Type Mean 67.67; MTEB(Code) 74.66 (averaged over seven code tasks).
    - Strong cross-lingual: XOR-Retrieve Recall@5k = 90.42 and XTREME-UP MRR@10 = 64.33.
  - Per-task-type on MMTEB(Multilingual) (Table 2):
    - Largest advantages over the second-best model:
      - Classification: 71.8 vs 62.2 (+9.6),
      - Retrieval: 67.7 vs 58.7 (+9.0),
      - Clustering: 55.0 vs 51.3 (+3.7).
    - Instruction Retrieval remains challenging across models (values are low/near 0 for many systems, including negatives for some baselines); Gemini Embedding is 5.2 on this type.
  - MTEB(Eng, v2) (Table 3):
    - Highest Borda rank with notable gains on Classification (90.1 vs 83.0, +7.1) and Clustering (59.4 vs 54.1, +5.3) compared to the second-ranked model.
  - MTEB(Code) (Table 4):
    - SOTA mean with full coverage across 8 tasks. Even when excluding COIR (a task many baselines omit), it remains #1 (Mean -COIR 75.5).
  - XTREME-UP (Table 5):
    - > “Average 64.3 MRR@10” vs next best 39.2–35.0 range for strong baselines—substantial gap.  
    - Two qualitative examples (Figure 2) show correct English passage retrieval for Assamese and noisy Hindi queries without translation.

- Ablations and diagnostics
  - Training mixtures (Table 6):
    - Pre-finetuning is essential: “No Training” ⇒ 30.55 (MTEB(Multilingual)); “Pre-finetuning Only” ⇒ 48.89.
    - English-only, task-diverse fine-tuning generalizes well: 66.75 on MTEB(Multilingual), 72.77 on MTEB(Eng, v2), and 85.70 on XOR-Retrieve; Multilingual-only (retrieval-focused) best boosts XTREME-UP (65.06) but lags on broad tasks.
  - Synthetic classification data (Table 7):
    - > “Average +17.6” improvement when adding synthetic datasets, with big jumps on AmazonCounterfactual and Emotion.
  - LLM-based filtering (Table 8):
    - > “+3.9” average on MIRACL across 18 languages; broad, consistent gains.
  - Hard negatives (Figure 3):
    - Adding 1–3 hard negatives typically improves retrieval (nDCG@10) on FEVER/HotpotQA/NQ/SciFact; too many can overfit and reduce performance.

- Do the experiments support the claims?
  - Yes. The model is tested across 100+ tasks, 250+ languages, English-only and code settings, and cross-lingual retrieval. Broad SOTA metrics (Tables 1–4) plus careful ablations (Tables 6–8, Figure 3) substantiate the design choices (Gemini initialization, task conditioning, LLM-curated data, two-stage recipe, Model Soup).
  - Nuance: Some task types remain hard for all models (e.g., Instruction Retrieval), and there are isolated low scores (e.g., Table 9 lists “Robust04InstructionRetrieval -2.41” and “TempReasonL1 2.96”), showing room for improvement on instruction-style and temporal reasoning tasks.

## 6. Limitations and Trade-offs
- Computational cost and reproducibility
  - Pre-finetuning on a billion-scale corpus with large batches, LLM-powered filtering, and multiple fine-tuning runs for Model Soup imply significant compute and orchestration complexity (Sections 3.3, 4.1–4.2).
  - Initialization from a proprietary LLM (Gemini) and absence of full training recipe details limit exact reproducibility; the model is available via API (Section 1 footnote), not necessarily as open weights.

- LLM-in-the-loop biases and coverage
  - Using Gemini to generate/filter data and to pick hard negatives could propagate or amplify LLM biases (Section 4.2). While multi-stage prompting and auto-rating aim for quality, the paper does not include a fairness/bias analysis.

- Overfitting risk with hard negatives
  - Figure 3 shows that adding too many hard negatives can overfit and hurt retrieval performance; careful tuning or regularization is required.

- Task pockets of weakness
  - Some instruction retrieval benchmarks remain low across the board; Gemini Embedding is better than peers but still single-digit (Tables 1–2). Temporal reasoning tasks (Table 9 “TempReasonL1 2.96”) and certain specialized datasets are not yet strong.

- Mixed message on language vs task diversity
  - While the English-only task-diverse mix generalizes impressively (Table 6), best long-tail language performance on XTREME-UP still benefits from multilingual training (65.06 vs 49.34 for English-only). This suggests a trade-off: broad task diversity for general utility versus explicit multilingual data to push long-tail languages further.

- Practical footprint
  - Default 3072-d embeddings are large; MRL supports 768/1536 but the paper does not report the specific performance drop at those sizes. Storage and latency trade-offs may matter for production-scale retrieval systems.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a unified embedder initialized from a strong LLM, combined with LLM-curated training data and simple architectural choices, can outperform specialized models across languages and tasks (Tables 1–4). This sets a new baseline for “one model fits many tasks.”

- What it enables
  - Practical deployments:
    - Multilingual and cross-lingual search/retrieval (e.g., global search over English corpora with queries in low-resource languages; Table 5, Figure 2).
    - Classification, clustering, and pair classification out-of-the-box for multilingual analytics (Tables 2–3).
    - Code search and retrieval with strong performance across many code tasks (Table 4, Table 10 Right).
  - Research directions:
    - Multi-modal embeddings: extend the recipe to images, video, and audio (Section 7 Future Work).
    - Better hard-negative strategies and regularization to avoid overfitting (Figure 3 suggests diminishing returns).
    - Bias, safety, and robustness studies for LLM-generated/filtered training data.
    - More systematic study of the task-diversity vs language-diversity trade-off (Table 6).
    - Compactness vs performance curves using MRL (reporting 768/1536-d results would guide production users).

- Concrete next steps (from the paper and beyond)
  - Multi-modal embedding space leveraging Gemini’s multimodal capabilities (Section 7).
  - Curating multi-modal tasks for generalizable representations (Section 7).
  - Exploring training recipes that balance uni- and multi-modal performance in a single model (Section 7).
  - Public reporting on fairness and domain robustness when using LLM-generated data.

> Headline results to remember (Table 1):  
> “MTEB(Multilingual) Task Mean 68.32 (+5.09 vs 2nd best), Type Mean 59.64 (+3.64), #1 Borda rank;  
> MTEB(Eng, v2) Task Mean 73.30; MTEB(Code) 74.66; XOR-Retrieve 90.42; XTREME-UP 64.33.”

In sum, Gemini Embedding combines a strong LLM foundation with disciplined training and data curation to deliver a single embedder that is both broadly capable and state-of-the-art across diverse, multilingual settings.
