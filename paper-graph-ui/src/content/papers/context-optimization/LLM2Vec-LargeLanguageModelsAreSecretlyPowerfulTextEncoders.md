# LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders

**ArXiv:** [2404.05961](https://arxiv.org/abs/2404.05961)

## 🎯 Pitch

LLM2Vec introduces a simple, unsupervised method that transforms any decoder-only large language model (LLM) into a high-performing universal text encoder using three lightweight steps: enabling bidirectional attention, masked next-token prediction, and unsupervised contrastive learning. This approach not only outperforms traditional encoder-only models on major text embedding benchmarks but also unlocks the full potential of popular LLMs for a wide range of embedding tasks—making state-of-the-art, efficient, and instruction-following embeddings accessible without expensive supervision or retraining. Its efficiency and effectiveness promise to simplify NLP pipelines and democratize high-quality text representation for real-world applications.

---

## 1. Executive Summary
LLM2Vec is a three-step, parameter- and data-efficient recipe that turns any decoder-only large language model (LLM) into a strong text encoder without labeled data. It does this by enabling bidirectional attention, lightly adapting the model with a masked next-token prediction objective, and then learning sentence representations via unsupervised contrastive learning (SimCSE). On the Massive Text Embedding Benchmark (MTEB), the method reaches a new unsupervised state of the art (Mistral-7B score 56.80; Table 1) and, with a small amount of supervised contrastive training on public data, achieves the best performance among public-data-only models (Meta-LLaMA-3-8B score 65.01; Table 2).

## 2. Context and Motivation
- Problem addressed
  - Decoder-only LLMs excel at generation and many NLP tasks, but are rarely used as universal text encoders (i.e., producing high-quality fixed vectors for sentences/phrases). The likely reason: their default “causal” attention restricts each token to see only tokens to its left, limiting full-context representations that are ideal for embeddings (Introduction; Section 1).
- Why it matters
  - High-quality text embeddings power search/retrieval, clustering, semantic similarity, and classification pipelines. Being able to re-use widely available decoder LLMs as general-purpose encoders reduces the need for separate encoder models and leverages LLM instruction-following capabilities for instruction-tuned embedding tasks (Section 1).
- Prior approaches and their gaps
  - Encoder/encoder–decoder models (e.g., BERT, T5) dominate embedding pipelines through multi-stage (weakly + fully) supervised contrastive training, but are data- and compute-heavy (Section 1).
  - Recent decoder-only attempts either:
    - Use last-token (EOS) pooling under causal attention, which underutilizes context (Section 3.2, Figure 3).
    - Duplicate the input (“Echo embeddings”) so the second copy can attend to “future” tokens, but this doubles sequence length and slows inference substantially (Appendix E.2, Table 6).
- Position in the literature
  - LLM2Vec is an unsupervised, three-step conversion that:
    - Removes the causal mask to enable bidirectional attention.
    - Adapts the model with a lightweight objective (MNTP).
    - Trains sentence embeddings with unsupervised SimCSE.
  - It matches or exceeds Echo embeddings while being faster at inference and requiring minimal adaptation (Table 1; Appendix E.2).

## 3. Technical Approach
The method has three sequential steps (Figure 1), followed by optional supervised contrastive fine-tuning.

1) Enable bidirectional attention
- What this is: Replace the causal attention mask with an all-ones mask so every token can attend to every other token in the sequence (Section 2.1 “Enabling bidirectional attention”).
- Why: Embeddings benefit when each token representation can use both left and right context.
- Caveat: Simply flipping the mask can hurt; Figure 2 and Figure 3 show immediate performance drops for S-LLaMA-1.3B and LLaMA-2-7B on word- and sequence-level tasks (“Bi” bars), though Mistral-7B is surprisingly robust (more in Section 4.2).

2) MNTP: masked next-token prediction (adaptation to the new mask)
- What it is (Section 2.1 “Masked next token prediction”):
  - Randomly mask some tokens in the input sequence.
  - Train the model to predict each masked token using the hidden state at the previous position (i−1), not the masked position itself. This aligns with decoder-style next-token prediction while exploiting bidirectional context for reconstruction.
  - The models lack a special mask token; an underscore character is used as the mask indicator (Section 2.2, “Masked next token prediction”).
- Why this design:
  - Predict-from-previous-position preserves the left-to-right training alignment common in decoder pretraining, easing adaptation to the newly enabled bidirectional attention.
- Training details (Section 2.2; Appendix D.1.1):
  - Lightweight LoRA fine-tuning (`r=16, α=32`), 1000 steps, batch size 32, single 80GB A100 (~100 minutes for 7B–8B models).
  - Masking strategies chosen by small hyperparameter search; e.g., Mistral-7B used a RoBERTa-style strategy with 80% masking for sentence-level tasks (Appendix D.1.1), but 20% masking for word-level tasks (Appendix D.1.3).

3) Unsupervised contrastive learning (SimCSE) to learn sentence embeddings
- What it is (Section 2.1 “Unsupervised contrastive learning”):
  - For each sentence, generate two “views” by running it twice through the model with different dropout masks.
  - Pull the two views of the same sentence together and push views of other sentences in the batch apart (in-batch negatives).
  - Use a pooling function over token representations to get sentence vectors; the paper ablates EOS, mean, and weighted mean pooling (Section 3.2; Figure 3; Appendix D.2.2/Table 5).
- Why:
  - Decoder LLMs are not trained with a “next sentence” objective; SimCSE supplies an unsupervised signal for sentence-level semantics.
- Training details (Section 2.2; Appendix D.1.2):
  - Merge MNTP LoRA into the base model, initialize new LoRA for SimCSE, 1000 steps, batch size 128 (for 7B–8B), dropout 0.3 (higher than typical 0.1 works better for LLMs), ~3 hours on a single 80GB A100.

Optional 4) Supervised contrastive learning
- Setup (Section 5.1; Appendix G.1–G.2):
  - Train on ~1.5M public pairs (replication of E5 public portion) with hard and in-batch negatives for 1000 steps, effective batch size 512, LoRA. MNTP weights are merged; trainable LoRA initialized from SimCSE for the full LLM2Vec variant.
- Purpose:
  - Further boost performance in supervised settings while retaining parameter efficiency.

Data choices
- MNTP and SimCSE both use English Wikipedia (Wikitext-103 for MNTP; a SimCSE Wikipedia subset for contrastive learning). This is likely already in the pretraining mix of the LLMs; the steps should teach “how to use bidirectional context” rather than new knowledge (Section 2.2 “Training data”).

Pooling choices and instructions
- Pooling: Mean pooling generally works best for LLM2Vec; weighted mean can be stronger for purely causal models (Section 3.2; Appendix D.2.2/Table 5).
- MTEB instructions: Task instructions (only for queries; same as Wang et al., 2023) are used across tasks (Section 3.2; Appendix C.2/Table 10). Instruction tokens are excluded from mean/weighted-mean pooling.

Implementation notes for efficiency
- All adaptation uses LoRA; FlashAttention-2, bfloat16, and gradient checkpointing are used for 7B–8B models (Appendix D.1.1, D.1.2, G.2), making training feasible on modest hardware.

## 4. Key Insights and Innovations
- A minimal, unsupervised, three-step recipe to repurpose decoder LLMs as universal encoders
  - Novelty: The sequence “enable bidirectionality → MNTP adaptation → SimCSE” is simple yet effective; each step addresses a specific shortcoming. MNTP’s predict-from-previous-position design is a clever bridge between decoder pretraining and bidirectional usage (Sections 2.1–2.2).
  - Significance: Achieves new unsupervised SOTA on MTEB (Table 1), with only ~100 minutes (MNTP) + ~3 hours (SimCSE) on a single A100 for 7B–8B models.
- Mean pooling beats EOS pooling once bidirectionality is enabled
  - Evidence: Across models, mean pooling consistently outperforms EOS pooling on MTEB subset once the model can use full context (Figure 3; Appendix D.2.2/Table 5).
  - Impact: Challenges the standard “last token/EOS” pooling for decoder LLM embeddings and provides a general recipe for better sentence vectors.
- Strong efficiency–performance trade-off vs. Echo embeddings
  - Difference: Echo duplicates the input to simulate future-context awareness, doubling sequence lengths at inference. LLM2Vec changes the mask and adapts the model—no inference overhead.
  - Evidence: LLM2Vec matches or outperforms Echo for 3 of 4 tested models after just the first two steps, with markedly faster evaluation times (Table 1; Appendix E.2/Table 6).
- Intriguing property of Mistral-7B: bidirectionality “just works”
  - Observation: Switching to bidirectional attention without training barely changes Mistral’s internal representations (high cosine similarity across layers and tokens in Figure 5c) and can even improve performance (e.g., MTEB subset in Figure 3c).
  - Hypothesis: Mistral was trained with some form of bidirectional objective (e.g., prefix LM) at least part of the time (Section 4.2; Appendix F/Figure 9). This is a new, empirically supported insight about model pretraining dynamics.

## 5. Experimental Analysis
- Evaluation methodology
  - Models: S-LLaMA-1.3B, LLaMA-2-7B-chat, Mistral-7B-Instruct-v0.2, and Meta-LLaMA-3-8B-Instruct (Section 2.2).
  - Word-level tasks: Chunking, NER, POS on CoNLL-2003. Method: freeze the model, train a linear classifier on token representations (Section 3.1; Figure 2; Appendix D.1.3).
  - Sentence-level tasks: MTEB (56 datasets; 7 categories). Also a 15-task subset for pooling ablations (Section 3.2; Figure 3; Appendix C.1/Table 3).
  - Instructions: Task-specific instructions added to queries only (Section 3.2; Appendix C.2/Table 10). Instruction tokens excluded from mean/weighted-mean pooling.
  - Baselines: Unsupervised BERT and BERT+SimCSE (Gao et al., 2021); Echo embeddings (Springer et al., 2024) implemented with same instructions (Section 3.2; Appendix E.1).
- Main quantitative results
  - Word-level improvements after adaptation
    - “Naive Bi” often hurts, but MNTP consistently helps. Example (Figure 2; Table 4):
      > Chunking accuracy (S-LLaMA-1.3B): Uni 86.10 → Bi 76.50 → Bi+MNTP 90.51.  
      > NER accuracy (LLaMA-2-7B): Uni 96.59 → Bi 92.31 → Bi+MNTP 97.16.
    - SimCSE tuned for sentences can slightly degrade token-level performance (e.g., S-LLaMA-1.3B Chunking: 90.51 → 89.33; Table 4), which is expected.
  - Unsupervised MTEB (Table 1; Figure 3)
    - Mean pooling is best for LLM2Vec; weighted mean for purely causal models (Appendix D.2.2/Table 5).
    - LLM2Vec Mistral-7B (w/ SimCSE) achieves:
      > Average 56.80 across 56 MTEB datasets, with category scores like Retrieval 38.05 and STS 78.50 (Table 1).
    - LLM2Vec consistently improves over “Uni + weighted mean” (causal) and often over Echo embeddings with the first two steps alone (Table 1).
  - Supervised MTEB with public data only (Table 2)
    - After supervised contrastive training, MNTP-only variants often slightly outperform those with SimCSE in the supervised setting, but both beat causal baselines:
      > Meta-LLaMA-3-8B + LLM2Vec (w/o SimCSE): Average 65.01 (best among public-data-only models in Table 2).
  - Efficiency vs. Echo (Appendix E.2/Table 6)
    - End-to-end evaluation runtime on 8×A100 for MTEB:
      > LLaMA-2-7B: LLM2Vec ≈ 42h vs. Echo ≈ 63h.
- Ablations and analyses
  - Pooling ablation (Figure 3; Appendix D.2.2/Table 5): mean pooling preferred for LLM2Vec; shows clear gains over EOS pooling.
  - Component ablation on MTEB subset (Appendix D.2.2/Table 5): MNTP and SimCSE both add value; SimCSE is critical for best sentence-level performance.
  - “Prefix sensitivity” test (Section 4.1; Figure 4): With embeddings pooled over a shared prefix, Bi+MNTP separates positives from negatives, demonstrating that future tokens influence prefix representations.
  - Representation stability under mask switch (Section 4.2; Figure 5): Mistral-7B’s hidden states remain strikingly similar between causal and bidirectional masks, unlike S-LLaMA-1.3B and LLaMA-2-7B.
  - Sample efficiency (Section 5.2; Figure 6): LLM2Vec-transformed models reach higher MTEB subset scores earlier during supervised training.
- Do the experiments back the claims?
  - Yes. The paper tests across four LLMs, two granularities (token- and sentence-level), multiple pooling strategies, and both unsupervised and supervised settings, with quantitative improvements and diagnostic analyses that explain why each step matters (Figures 2–6; Tables 1–2, 4–6, 11–12).

## 6. Limitations and Trade-offs
- Computational/resource constraints (Appendix A “Limitations”)
  - Decoder LLMs are large (e.g., 7B–8B with 4096-dim embeddings), making indexing/storage heavier than BERT-size encoders (768-dim). Even with LoRA and FlashAttention-2, training and inference can be slower than small, purpose-built encoders.
- When SimCSE helps vs. hurts
  - SimCSE improves sentence-level tasks but can slightly hurt word-level probing (Table 4), implying a trade-off between token and sentence representation specialization.
  - In supervised settings, MNTP-only sometimes edges out SimCSE-initialized variants (Table 2), so the optimal pre-adaptation may depend on whether supervised training follows.
- Data assumptions and contamination risk
  - MNTP/SimCSE use Wikipedia. The supervised data includes public datasets whose test distributions may overlap with pretraining corpora; precise pretraining mixes are unknown (Appendix A “Data contamination”).
- Language scope
  - The paper evaluates only English. The method is language-agnostic in principle, but multilingual performance is untested (Appendix A “Extending to other languages”).
- Why Mistral is special remains an open question
  - The paper hypothesizes some bidirectional pretraining signal for Mistral but does not confirm it (Section 4.2; Appendix F). Understanding this would guide model selection and adaptation strategies.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that decoder-only LLMs can be turned into universal text encoders with minimal, unsupervised adaptation, often matching or beating specialized encoders and “input duplication” methods (Echo) while being more efficient at inference (Table 1; Appendix E.2). This can simplify stacks by unifying generation and embedding in one model family.
- Practical applications
  - Retrieval and reranking pipelines, semantic search, clustering, deduplication, and zero/few-shot classification benefit from improved general-purpose embeddings, especially when instruction-following (query prompts) is useful (Tables 1–2; Appendix C.2).
- Research directions
  - Understanding pretraining signals: Investigate why Mistral-7B tolerates bidirectional masks without adaptation (Figure 5, Figure 9).
  - Multilingual LLM2Vec: Apply the same recipe to multilingual LLMs and evaluate on multilingual MTEB variants.
  - Smaller models and dimensionality reduction: Explore compressing 4096-dim embeddings for scalable indexing while retaining performance (Appendix A).
  - Task-adaptive trade-offs: Study how to jointly optimize token- and sentence-level quality, possibly with multi-objective training or layer-specific pooling.
  - Data efficiency: Extend sample-efficiency findings (Figure 6) to very low-label regimes and domain adaptation settings (e.g., legal/biomedical).

> Key Takeaway: With three simple, unsupervised steps—remove the causal mask, briefly adapt via MNTP, and train SimCSE with mean pooling—decoder LLMs become high-performing text encoders. This unlocks strong, efficient embeddings from widely available models and reduces reliance on heavy multi-stage pipelines or inference-inefficient tricks.
