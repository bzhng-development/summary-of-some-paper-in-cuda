# LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

**ArXiv:** [2404.05961](https://arxiv.org/abs/2404.05961)
**Authors:** Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, Siva Reddy
**Institutions:** McGill University (McGill‑NLP)

## 🎯 Pitch

LLM2Vec revolutionizes text embeddings by converting decoder-only language models into high-performing text encoders using a simple, unsupervised method involving bidirectional attention and contrastive learning. This approach eliminates the need for large supervised datasets, offering efficiency and versatility that enhance search, classification, and retrieval systems, setting a new state-of-the-art in unsupervised benchmarking.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces LLM2Vec, a three-step, unsupervised method that turns any decoder-only large language model (LLM) into a high-quality text embedding model. It replaces causal attention with bidirectional attention, adapts the model with a targeted training objective (masked next token prediction), and then learns sentence-level embeddings using unsupervised contrastive learning—achieving new state of the art on the unsupervised MTEB benchmark while being parameter- and data-efficient.

## 2. Context and Motivation
- Problem addressed
  - Decoder-only LLMs excel at many NLP tasks but are underused for text embeddings because their causal attention only looks leftward, limiting how much context each token can incorporate when forming representations. This is suboptimal for embeddings, which benefit from full context.
  - The paper targets a method to re-purpose decoder-only LLMs into strong “universal” text encoders (usable across diverse tasks) without needing labeled data or expensive fully supervised pipelines.

- Why it matters
  - Text embeddings underpin retrieval, clustering, semantic similarity, and classification systems. Better embeddings directly improve ranking, search, question answering, and downstream analytics.
  - Decoder-only LLMs have practical advantages: they learn from every token during pretraining (sample efficiency), benefit from a rich open-source ecosystem, and have strong instruction-following skills that translate well to instruction-tuned embeddings (Section 1; Appendix A).

- Prior approaches and their shortcomings
  - Encoder/encoder-decoder models (e.g., BERT, T5) dominate embeddings but often require multi-stage, large-scale supervised or weakly supervised contrastive training (e.g., GTR, E5; see Related Work, Section 6).
  - Earlier attempts with decoder-only LLMs either:
    - Kept causal attention and used last-token pooling (often suboptimal), or
    - Used structural workarounds like Echo Embeddings that duplicate the input to simulate looking right—effective but doubling sequence length and slowing inference (Table 6 in Appendix E.2 shows Echo takes ≈64h vs ≈44h for LLM2Vec on Mistral-7B over MTEB).

- Positioning
  - LLM2Vec offers a simple, unsupervised, and parameter-efficient path to convert any decoder-only LLM into a strong text encoder:
    - No labels; uses Wikipedia only for adaptation/fine-tuning (Section 2.2).
    - Minimal compute: 1000 steps for each phase; LoRA adaptation; single 80GB A100 for 7B/8B models (Section 2.2; Appendix D.1.1–D.1.2).
    - Outperforms prior unsupervised baselines on MTEB and competes favorably with Echo embeddings without doubling sequence length (Table 1; Appendix E.2).

## 3. Technical Approach
LLM2Vec has three steps (Figure 1):

1) Enabling bidirectional attention
- What changes: Replace the usual causal mask (each token can attend only to previous tokens) with an all-ones mask so every token can attend to all tokens in the sequence (Appendix B.1 explains the attention mechanism and masking, Eq. 4).
- Why: For embeddings, tokens should incorporate both left and right context to form richer representations.
- Caveat: Simply turning off the causal mask often hurts performance initially because the model hasn’t learned how to use future tokens (Section 2.1, “Enabling bidirectional attention”; Figures 2 and 3 show an immediate drop without adaptation for most models).

2) Masked Next Token Prediction (MNTP)
- What it is: A training objective combining masked language modeling and next token prediction (Section 2.1, “Masked next token prediction”).
  - Select a fraction of tokens to mask (mask token is underscore `_` because these models lack a dedicated mask token).
  - Predict each masked token using the representation at the previous position `i-1` (not at the masked position `i`). This preserves the decoder-style “predict the next token” framing while letting the model exploit bidirectional context to fill in masks (Figure 1 middle; Appendix D.1.1 for masking strategy).
- Why this choice:
  - Aligns with decoder pretraining (next token prediction), easing adaptation to the new attention pattern.
  - Forces the model to leverage both past and future context to predict masked tokens.
- Training details:
  - Data: Wikitext-103 (Wikipedia) for MNTP (Section 2.2).
  - Hyperparameters: LoRA (r=16, α=32), 1000 steps, batch size 32 on a single 80GB A100; ≈100 minutes for 7B/8B models (Section 2.2; Appendix D.1.1).
  - Masking policy: Chosen by model via a small search (Appendix D.1.1): most models use BERT-style 20% masking; for Mistral-7B, RoBERTa-style 80% masking works best on sentence-level tasks; for word-level probing, a 20% BERT-style variant is used (Appendix D.1.3).

3) Unsupervised contrastive learning (SimCSE)
- What it is: Learn sentence embeddings by maximizing similarity between two dropout-perturbed passes of the same sentence and minimizing similarity to other sentences in the batch (Section 2.1 “Unsupervised contrastive learning”; Appendix B.2 Eq. 5).
- Why: Decoder-only LLMs are not trained to encode whole-sequence meaning explicitly; SimCSE adds a sentence-level training signal without requiring labeled pairs.
- Pooling: Compute sentence embeddings via a pooling function over token embeddings. The paper tests EOS (last token), mean pooling, and weighted mean pooling; mean pooling turns out best for LLM2Vec (Figure 3; Table 5).
- Training details:
  - Data: Wikipedia sentence subset from SimCSE (Section 2.2).
  - Hyperparameters: LoRA (r=16, α=32), dropout p=0.3 (higher than usual for larger LLMs), 1000 steps; for 7B/8B batch size 128, ≈3 hours on a single 80GB A100 (Section 2.2; Appendix D.1.2).

Implementation/data pipeline
- Models adapted: `S-LLaMA-1.3B`, `LLaMA-2-7B-chat`, `Mistral-7B-Instruct-v0.2`, and (for main tables) `Meta-LLaMA-3-8B-Instruct` (Section 2.2).
- Instruction prompting at evaluation: Task-specific query prompts (for MTEB and supervised E5 training) are used; instructions excluded from pooling for mean/weighted mean (Section 3.2 “Setup”; Appendix C.2, Table 10; Appendix G.1–G.2, Table 8).

Pooling choices and why they matter
- EOS pooling is the standard in causal LLMs (use the last token). Results show it is suboptimal for embeddings versus mean pooling once bidirectional attention is enabled (Figure 3; Table 5).

## 4. Key Insights and Innovations
- A minimal recipe to unlock bidirectional representations in decoder-only LLMs
  - Novelty: The combination “enable bidirectional attention → short MNTP adaptation → SimCSE” is simple, unsupervised, and parameter-efficient. Crucially, MNTP predicts the target token from position `i-1`, preserving the generative framing while learning to use future context (Figure 1; Section 2.1).
  - Significance: It avoids heavy supervised pipelines and synthetic data while delivering large gains on both token-level and sentence-level tasks (Figures 2–3; Tables 1, 4, 5).

- Strong unsupervised performance with modest compute
  - Claim supported by numbers: On full MTEB, `Mistral-7B + LLM2Vec` achieves an average 56.80, the best among unsupervised models tested in the paper (Table 1). Training is just 2×1000 steps (MNTP + SimCSE) with LoRA on Wikipedia only (Section 2.2).
  - Importance: Shows that decoder-only LLMs can be powerful embedders without large-scale supervised datasets.

- Efficiency advantage over Echo embeddings at inference
  - Difference: Echo embeddings duplicate the input sequence to simulate right-context; LLM2Vec does not. On MTEB, LLM2Vec is much faster (e.g., `Mistral-7B`: ≈44h vs ≈64h on 8×A100; Table 6, Appendix E.2), while being equal or better in accuracy when SimCSE is included (Table 1).

- Diagnostic finding about Mistral-7B’s “built-in” bidirectionality
  - Observation: Simply enabling bidirectional attention (no training) barely changes Mistral-7B’s internal representations (Figure 5c), unlike LLaMA variants, and sometimes improves performance (Figures 2–3).
  - Interpretation: The paper hypothesizes Mistral was trained with some bidirectional signal (e.g., prefix language modeling), at least in parts (Section 4.2). This is a surprising and practically useful property.

## 5. Experimental Analysis
- Evaluation methodology
  - Word-level probing (chunking, NER, POS): CoNLL-2003; train a linear classifier on frozen embeddings (Figure 2; Table 4; Appendix D.1.3).
  - Sequence-level tasks: Massive Text Embedding Benchmark (MTEB), 56 datasets across 7 categories, with task-specific instructions for queries (Section 3.2; Table 1). A 15-task MTEB subset is used for pooling ablations (Figure 3; Appendix C.1, Table 3).
  - Supervised setting: Contrastive fine-tuning on the public portion of E5 (~1.5M pairs) using LoRA; 1000 steps, batch 512; compares with models trained on public data only (Section 5; Table 2; Appendix G.1–G.3).

- Main quantitative results
  - Word-level (Figure 2; Table 4)
    - Naively enabling bidirectional attention hurts performance for S-LLaMA and LLaMA-2 (e.g., `LLaMA-2-7B` chunking 88.23 → 78.24), but much less so for `Mistral-7B` (87.53 → 85.66). After MNTP, all models improve beyond the causal baseline (e.g., `LLaMA-2-7B` chunking 91.61 vs 88.23; NER 97.16 vs 96.59; POS 92.61 vs 91.53). Adding SimCSE, which is sentence-level, slightly reduces word-level performance as expected.
    - Quote: 
      > Figure 2: “adapting via MNTP improves performance … in the chunking task, improvements are +5% (S-LLaMA-1.3B), +4% (LLaMA-2-7B), +4% (Mistral-7B).”

  - Unsupervised MTEB (Table 1; Figure 3; Table 5)
    - Pooling ablation on the 15-task subset (Figure 3; Table 5): mean pooling outperforms EOS notably once bidirectional attention is used. Weighted mean is best for purely causal models but not after LLM2Vec.
    - Full MTEB averages (Table 1):
      - `Mistral-7B`: Uni+weighted mean 42.46 → Bi+MNTP 49.43 → LLM2Vec (with SimCSE) 56.80 (SOTA among unsupervised tested).
      - `LLaMA-2-7B`: 44.54 → 45.70 → 55.36.
      - `S-LLaMA-1.3B`: 35.05 → 41.43 → 49.42.
      - `Meta-LLaMA-3-8B`: 43.98 → 48.84 → 56.23.
    - Category breakdown for `Mistral-7B + LLM2Vec` (Table 1): Retrieval 38.05, Reranking 53.99, Clustering 40.63, Pair Classification 80.94, Classification 74.07, STS 78.50, SummEval 30.19.
    - Compared to Echo (Table 1): LLM2Vec (w/o SimCSE) is similar or better while being faster at inference (Appendix E.2). With SimCSE, LLM2Vec wins by larger margins.

    - Quotes:
      > Table 1: “Mistral-7B + LLM2Vec [unsupervised] achieves 56.80 average, the best among unsupervised models in this comparison.”
      > Figure 3: “LLM2Vec … works best with mean pooling.”

  - Supervised MTEB (Table 2; Figure 6)
    - Training on public E5-only data (Appendix G.1), LLM2Vec improves over strong Uni baselines across models. Best score: `Meta-LLaMA-3-8B + LLM2Vec (w/o SimCSE)` reaches 65.01, exceeding strong public-data-only models like `BGE large v1.5` (64.23) and `E5Mistral-7b-v1` (64.56).
    - `Mistral-7B + LLM2Vec (w/o SimCSE)` reaches 64.80; `LLaMA-2-7B + LLM2Vec (w/o SimCSE)` 64.14 (Table 2).
    - Sample efficiency (Figure 6): LLM2Vec-initialized models reach higher scores earlier in training than Uni baselines on the 15-task subset.

    - Quotes:
      > Table 2: “Meta-LLaMA-3-8B + LLM2Vec (w/o SimCSE) achieves a new SOTA among models trained only on publicly available data.”
      > Figure 6: “applying LLM2Vec before supervised training leads to better performance with fewer steps.”

- Diagnostics: How LLM2Vec changes representations
  - Using a synthetic prefix dataset, LLM2Vec with MNTP separates positives from negatives clearly for S-LLaMA; for Mistral-7B, separation appears even without training once bi-attention is enabled (Figure 4; Appendix F Figure 7).
  - Representation similarity across layers with and without bi-attention (Figure 5): Mistral-7B’s hidden states barely change (high cosine similarity throughout), unlike S-LLaMA and LLaMA-2 which change substantially; suggests Mistral’s pretraining already exposed it to bidirectional signals (Section 4.2; Appendix F Figure 9).

- Ablations and robustness
  - Component ablation on the 15-task subset indicates all three steps matter: MNTP recovers the performance loss from enabling bi-attention; SimCSE adds substantial gains for sentence-level tasks (Figure 3; Table 5; Appendix D.2.2).
  - Efficiency comparison with Echo embeddings (Appendix E.2, Table 6) demonstrates practical speed advantages.

- Overall assessment
  - The experiments are well aligned with the claims:
    - Token-level probing shows MNTP is necessary to exploit bidirectional attention (Figure 2; Table 4).
    - Sentence-level performance scales strongly after SimCSE (Figure 3; Table 1).
    - Both unsupervised and supervised settings show state-of-the-art or competitive results under fair constraints (Wikipedia-only unsupervised; public-data-only supervised).

## 6. Limitations and Trade-offs
- Model size and embedding dimensionality
  - Decoder-only LLMs tend to be large (7B–8B here) with high embedding dimensions (e.g., 4096), increasing memory and indexing cost versus smaller encoders (Appendix A, “Large size of decoder-only LLMs”).
  - While LoRA reduces training cost, inference indexing over large corpora still costs more per vector.

- Uncertain pretraining data and contamination
  - The exact pretraining mixes of LLaMA-2 and Mistral are not fully public; there may be inadvertent overlap with evaluation sets in MTEB (Appendix A, “Data contamination”). This affects cross-paper fairness broadly in the field, not just this work.

- Language coverage
  - LLM2Vec is demonstrated only in English; the method is language-agnostic but not evaluated on other languages (Appendix A, “Extending to other languages”).

- Task-specific trade-offs
  - SimCSE improves sentence-level tasks but may slightly hurt token-level probing tasks (Table 4), as it focuses on sentence representations rather than token granularity.
  - The paper does not evaluate whether enabling bidirectional attention affects generative capabilities of the same model; in practice, one would maintain a separate embedding variant.

- Assumptions and scope
  - Assumes access to a decoder-only LLM and the ability to modify its attention mask and run LoRA fine-tuning.
  - The gains rely on careful pooling choices (mean for LLM2Vec) and short adaptation stages with Wikipedia-like data.

## 7. Implications and Future Directions
- How it shifts the field
  - Demonstrates that modern decoder-only LLMs can be turned into strong, efficient, universal embedders with minimal, unsupervised adaptation—reducing dependence on massive supervised contrastive datasets.
  - Encourages the community to rethink the “encoder-only for embeddings” assumption and to leverage the tooling and instruction-following ecosystem around decoder-only LLMs.

- Follow-up research directions
  - Pretraining strategies: Investigate hybrid or prefix-style objectives that make decoder-only models natively bidirectional for embeddings (Section 4.2’s Mistral finding).
  - Multilingual and domain-specialized LLM2Vec: Evaluate the recipe across languages and specialized corpora (Appendix A).
  - Efficiency and deployment: Explore dimensionality reduction, quantization, or distillation to reduce 4096-dim vector costs without losing accuracy.
  - Robustness and contamination-free evaluation: Curate new benchmarks clearly outside model pretraining mixtures to measure true generalization (Appendix A).

- Practical applications
  - Drop-in embedding backends for search, retrieval-augmented generation, deduplication, clustering, and semantic monitoring.
  - Instruction-aware embeddings for better domain/task transfer (Sections 3.2 and 5 use instruction prompts and show strong transfer across MTEB categories).
  - Faster alternatives to Echo embeddings at inference time while matching or surpassing accuracy (Appendix E.2; Table 1).

> Key takeaways:
> - A three-step unsupervised recipe—bidirectional attention + MNTP + SimCSE—turns decoder-only LLMs into powerful text encoders (Figure 1).
> - On unsupervised MTEB, `Mistral-7B + LLM2Vec` achieves 56.80 average (Table 1), and with supervised public-data-only training, `Meta-LLaMA-3-8B + LLM2Vec (w/o SimCSE)` reaches 65.01 (Table 2).
> - Mean pooling, not EOS, is the right strategy once bidirectional attention is enabled (Figure 3; Table 5).
> - Mistral-7B shows unusual robustness to enabling bidirectional attention without training, hinting at bidirectional signals in its pretraining (Figures 4–5; Section 4.2).
