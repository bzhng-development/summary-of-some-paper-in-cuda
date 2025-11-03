# Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models

**ArXiv:** [2506.05176](https://arxiv.org/abs/2506.05176)
**Authors:** Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, Jingren Zhou
**Institutions:** 

## 🎯 Pitch

The Qwen3-Embedding series introduces a novel approach combining large-scale synthetic weak supervision, fine-tuning, and model-parameter merging to set new benchmarks in multilingual and code retrieval tasks. This innovation not only enhances retrieval precision and reduces hallucinations in LLMs but also ensures broad applicability across tasks and languages, supported by compelling experimental results and significant performance improvements in real-world applications.

---

## 1. Executive Summary
This paper introduces the Qwen3-Embedding series—three instruction-aware text embedding models (0.6B, 4B, 8B) and three rerankers of the same sizes—built on the Qwen3 foundation models. The core contribution is a multi-stage recipe that marries large-scale synthetic weak supervision, supervised fine-tuning, and model-parameter merging to deliver state-of-the-art multilingual and code retrieval performance across MTEB/MMTEB benchmarks (e.g., Table 3 and Table 2).

## 2. Context and Motivation
- Problem addressed
  - How to produce embedding and reranking models that are both broadly general (across tasks, languages, and domains) and strongly effective for real-world retrieval pipelines (web search, RAG, recommendation), while remaining efficient enough for deployment (Section 1; Table 1).
  - Two recurring obstacles:
    - Labeled data scarcity and imbalance across languages/tasks.
    - Alignment of embeddings with task instructions and user intent (instruction-following behavior) for better retrieval and reranking.

- Why it matters
  - Embeddings underpin semantic search, RAG systems, clustering, and classification. Higher-quality embeddings and stronger rerankers directly improve retrieval precision, reduce hallucinations in LLM RAG, and enable cross-lingual/cross-domain applications (Section 1).

- Prior approaches and gaps
  - Pre-LLM methods largely used encoder-only backbones (e.g., SBERT) and curated datasets (Section 1; Reimers & Gurevych 2019).
  - Recent general-purpose embedders (E5, BGE, GTE) rely heavily on weakly supervised web/Q&A corpora with limited control over quality, domain coverage, and language balance (Section 3.2).
  - Reranking with LLMs either uses zero-shot prompting (good generality, unstable) or SFT (better stability, limited coverage), with less integration across training stages (Section 1).

- Positioning
  - This work leverages Qwen3 LLMs not only as backbones but also as data generators to synthesize large, controllable, multilingual training pairs, then consolidates performance via model merging. It targets both embedding and reranking with instruction-aware inputs and supports practical deployment features like flexible output dimensions (Table 1; Sections 2–3).

## 3. Technical Approach
- Overview of the system
  - Two model families:
    - `Qwen3-Embedding-{0.6B,4B,8B}`: instruction-aware embedding models with customizable output dimensions (Table 1).
    - `Qwen3-Reranker-{0.6B,4B,8B}`: instruction-aware point-wise rerankers that score a single query-document pair per pass (Figure 1 right).
  - Three-stage training pipeline for embeddings and two stages for rerankers (Figure 2).

- Architecture and I/O formatting (Figure 1; Table 1)
  - Backbones: dense Qwen3 LLMs (0.6B, 4B, 8B), 32k context length, hidden sizes scaled by model size.
  - Embedding models:
    - Input: concatenate `Instruction` and `Query` into one string; `Document` is encoded independently. Queries use the format `{Instruction} {Query}<|endoftext|>` (Section “Embedding Models”).
    - Representation: use the last-layer hidden state at the end-of-sequence token `[EOS]` as the vector embedding.
    - “MRL Support” in Table 1 indicates the embedding dimension can be customized (e.g., 1024/2560/4096 by default).
  - Reranker models:
    - Format: a chat-style prompt where the system instructs the model to answer “yes” or “no” about the relevance of `Document` given `Query` and `Instruction`. The template is shown in Section “Reranking Models.”
    - Scoring: compute a probability-like score by contrasting next-token likelihoods for “yes” vs “no”:
      - score(q, d) = exp P(yes|I,q,d) / [exp P(yes|I,q,d) + exp P(no|I,q,d)] (Section “Reranking Models,” equation under Figure 2).

- Training objectives (Section 3.1)
  - Embeddings: an enhanced contrastive loss based on InfoNCE (Equation 1) with a rich negative set:
    - Positives: the paired relevant document `d_i^+` for query `q_i`.
    - Negatives: K hard negatives `d_i,k^-`, other in-batch queries `q_j`, other in-batch documents `d_j` (both against `q_i` and against `d_i^+`).
    - A mask `m_ij` discards likely false negatives when their similarity exceeds the positive by >0.1 or if `d_j` equals `d_i^+`. The similarity function is cosine; `τ` is a temperature (Equation 1 and the definition of `Z_i` and `m_ij`).
    - Intuition: push true pairs together while pushing apart multiple kinds of confounders, but avoid penalizing potential false negatives.
  - Rerankers: supervised fine-tuning loss on the “yes/no” label (Equation 2), i.e., negative log-likelihood of the correct token given the prompt.

- Multi-stage training (Section 3.2; Figure 2; Table 6)
  - Stage 1 (embeddings only): weakly supervised pre-training on ~150M synthetic pairs spanning retrieval, bitext mining, classification, and semantic textual similarity (STS).
  - Stage 2: supervised fine-tuning on high-quality labeled datasets (e.g., MS MARCO, NQ, HotpotQA, MIRACL, MLDR, Mr.TyDi, Multi-CPR, CodeSearchNet; see Table 6) plus a filtered, high-quality synthetic subset (~12M pairs with cosine similarity > 0.7).
  - Stage 3: checkpoint model merging using spherical linear interpolation (`slerp`) across multiple fine-tuning checkpoints to improve robustness and generalization (Section 3.2).
    - `slerp` is a weight-space interpolation that preserves vector norms during interpolation on the hypersphere, often yielding smoother blends than linear interpolation.

- Synthetic data generation (Section 3.3; Appendix A.1)
  - Generator: Qwen3-32B used to synthesize multilingual data with controllable attributes.
  - Retrieval data pipeline (document-to-query):
    1) Configuration stage: given a document and candidate “Characters” (user personas retrieved from Persona Hub; top-5 chosen by a retrieval model), an LLM selects a persona, a `Question_Type` (keywords, acquire_knowledge, summary, yes_or_no, background), and a `Difficulty` (high_school, university, phd) via a JSON output (Appendix A.1).
    2) Query generation stage: conditioned on the chosen persona and configuration, another prompt generates the query with constraints on length and target language, producing realistic, persona-grounded queries (Appendix A.1).
  - Scale and selection:
    - Total weak supervision: ~150M pairs across tasks and languages.
    - High-quality subset for Stage 2: ~12M pairs selected via cosine similarity > 0.7 (Section 3.3).
    - Labeled data for Stage 2: ~7M (Table 6).

- Reranker training specifics
  - No Stage 1; trained with Stage 2 supervised SFT and Stage 3 merging (Section 3.2).
  - Point-wise training (each query-document pair labeled yes/no) simplifies integration with LLM decoding and keeps inference simple.

## 4. Key Insights and Innovations
- Large-scale, LLM-driven synthetic data with fine-grained control
  - What’s new: The paper uses a strong LLM (Qwen3-32B) to synthesize multi-task, multilingual pairs with explicit control over persona, question type, difficulty, length, and language (Section 3.3; Appendix A.1), rather than scraping weak signals from heterogeneous web sources.
  - Why it matters: Improves coverage in low-resource languages and task types, while enabling quality filtering and consistency. Evidence: strong multilingual gains in Table 2 and competitive-to-best English results in Table 3.

- Instruction-aware embeddings and rerankers
  - What’s new: Both embeddings and rerankers condition on an explicit `Instruction` alongside the query, enabling task-aware similarity (Section “Embedding Models” and “Reranking Models”; Table 1 includes “Instruction Aware”).
  - Why it matters: In retrieval, alignment to intent or task instruction often drives large quality improvements (e.g., different instructions for STS vs. passage retrieval). This is especially important for complex instruction following (FollowIR; Table 4).

- Model merging via `slerp` to improve robustness
  - What’s new: The paper merges multiple supervised fine-tuning checkpoints using spherical linear interpolation (Section 3.2).
  - Why it matters: Ablations in Table 5 show a clear boost from merging for the 0.6B embedder, e.g., on MMTEB mean-task: 62.56 without merging vs. 64.33 with merging (+1.77).

- Enhanced contrastive objective with comprehensive negatives and false-negative masking
  - What’s new: The InfoNCE denominator includes diverse negatives (hard negatives, in-batch queries, in-batch documents vs. both query and positive doc) and a mask to remove likely false negatives (Equation 1).
  - Why it matters: Reduces representation collapse and avoids harming semantically similar but unlabeled pairs, improving generalization across tasks and languages (Section 3.1).

- Practical feature: customizable embedding dimension (“MRL Support”)
  - What’s new: The embedding models can output vectors with custom dimension (Table 1).
  - Why it matters: Lets practitioners trade off accuracy vs. latency/storage by choosing dimensions to fit system constraints.

## 5. Experimental Analysis
- Evaluation design (Section 4.1)
  - Benchmarks and coverage:
    - MMTEB (Enevoldsen et al., 2025): the multilingual superset covering >500 tasks across >250 languages; the paper evaluates 216 tasks drawn from MMTEB—131 MTEB Multilingual, 41 MTEB English v2, 32 CMTEB (Chinese), and 12 MTEB Code tasks.
    - Reranking: MTEB-R, CMTEB-R, MMTEB-R, MLDR (Chinese long-doc retrieval), MTEB-Code, and FollowIR (complex instruction retrieval).
  - Baselines: strong open-source embedders (E5, BGE-M3, GTE-Qwen2, NV-Embed-v2, GritLM-7B) and commercial APIs (OpenAI text-embedding-3-large, Google Gemini Embedding, Cohere multilingual) (Section 4.1; Tables 2–3).
  - Reranking setup: all rerankers are evaluated on the same top-100 candidates retrieved by `Qwen3-Embedding-0.6B` to ensure fairness (Table 4 note).

- Main embedding results
  - Multilingual (Table 2):
    - `Qwen3-Embedding-8B`: 70.58 mean-task (best), 61.69 mean-type.
    - `Qwen3-Embedding-4B`: 69.45 mean-task.
    - Both surpass the strongest proprietary baseline listed, Gemini Embedding (68.37 mean-task).
  - English v2 (Table 3):
    - `Qwen3-Embedding-8B`: 75.22 mean-task; `4B`: 74.60; `0.6B`: 70.70.
    - All exceed NV-Embed-v2 (69.81) and Gemini Embedding (73.30).
  - Chinese CMTEB (Table 3):
    - `8B`: 73.83 mean-task; `4B`: 72.26, both above `gte-Qwen2-7B-instruct` (71.62).
  - Code retrieval (Table 3):
    - `8B`: 80.68; `4B`: 80.06; `0.6B`: 75.41, all notably higher than Gemini Embedding 74.66.

- Main reranking results (Table 4)
  - Across basic relevance retrieval and code retrieval:
    - `Qwen3-Reranker-4B` and `8B` are best or near-best on most sets. For example, on MTEB-Code, `4B` = 81.20 and `8B` = 81.22 nDCG@10, well above other rerankers (e.g., jina reranker 58.98).
    - For FollowIR (complex instruction retrieval), `4B` shows a large improvement (14.84) over embedding-based ranking (5.09) and other rerankers; `8B` is lower (8.05), indicating a size-performance trade-off on this task.

- Ablations (Table 5)
  - Stage-1 synthetic pretraining helps:
    - Final `0.6B` (64.33) vs. “w/o synthetic data” (61.21) on MMTEB: +3.12.
    - “Only synthetic data” is weaker (58.49) than the final model, confirming the value of mixing synthetic with supervised and model merging.
  - Model merging helps:
    - “w/o model merge” (62.56) vs. final (64.33) on MMTEB: +1.77.

- Do the experiments support the claims?
  - Yes, the results are broad (216 tasks) and include strong baselines (Tables 2–4).
  - The ablations directly test two central claims: the value of synthetic pretraining and model merging (Table 5).
  - The reranking protocol controls for retrieval by fixing the top-100 candidate pool (Table 4 note), isolating reranker differences.

- Additional detailed results
  - Appendix Tables 7–9 provide per-type breakdowns (e.g., classification, clustering, STS) and detailed code retrieval tasks, where the Qwen3 models are consistently strong (e.g., Table 9 shows `Qwen3-Embedding-8B` ≥ 89 on multiple CodeSearchNet-style tasks).

## 6. Limitations and Trade-offs
- Dependence on a strong LLM for synthesis
  - The synthetic data’s quality relies on Qwen3-32B (Section 3.3). If the generator’s biases or gaps are present, they may propagate into the embedder. While cosine filtering (>0.7) removes some low-quality pairs, it may favor easy pairs and prune hard-but-useful ones (Section 3.3).

- Yes/no point-wise reranking
  - The reranker reduces relevance to a binary decision via next-token likelihoods (Section “Reranking Models,” scoring formula), which simplifies training but may limit graded relevance modeling compared to pairwise/listwise approaches for some IR tasks.

- Model-size vs. task behavior
  - Bigger is not always better: `Qwen3-Reranker-8B` underperforms `4B` on FollowIR (8.05 vs. 14.84 in Table 4). This suggests sensitivity to prompt format, label space, or optimization dynamics for complex instruction following.

- Compute and storage
  - While model sizes are moderate (0.6B–8B), training involves ~150M synthetic pairs plus millions of supervised examples (Table 6) and multiple checkpoints for merging, implying non-trivial compute cost. Inference cost scales with dimension (1024–4096) and context length (32k; Table 1).

- Evaluation scope
  - Although coverage is broad, real-world long-tail domains and adversarial robustness are not specifically analyzed. Also, fairness or bias assessments are not reported.

## 7. Implications and Future Directions
- Impact on the field
  - Demonstrates that foundation models can bootstrap high-quality, controllable synthetic corpora for embeddings and reranking at scale, narrowing the gap with or surpassing proprietary systems (Tables 2–3).
  - Validates a practical recipe—synthetic pretraining + supervised finetuning + `slerp` merging—for robust, instruction-aware embeddings.

- Practical applications
  - Multilingual semantic search and RAG across >250 languages, code retrieval for developer tools, cross-lingual knowledge base search, classification and clustering with better task conditioning, and improved reranking modules for production IR systems.

- Follow-up research
  - Richer relevance targets for reranking: extend beyond binary labels to graded judgments; explore pairwise/listwise objectives compatible with LLM decoders.
  - Hard-negative mining with the LLM: use the model itself to propose adversarial negatives during training, potentially improving robustness beyond static hard-negative sets.
  - More principled synthetic data selection: replace global cosine thresholds with difficulty-aware selection, diversity metrics, or small human-in-the-loop audits.
  - Unified training of embedding and reranking: multi-task objectives that jointly train the vector space and the yes/no decision head could reduce misalignment.
  - Efficient deployment: investigate low-rank adaptation and dynamic dimension selection (“MRL”) to tailor embeddings to latency/memory constraints without re-training.

> In sum, leveraging foundation models as both backbones and data generators, Qwen3-Embedding delivers instruction-aware, multilingual embeddings and rerankers that attain state-of-the-art results across MMTEB/MTEB/CMTEB and code retrieval (Tables 2–4), with ablations demonstrating the importance of large-scale synthetic pretraining and model merging (Table 5). The approach is practical, extensible, and well-suited to modern RAG and search systems.
