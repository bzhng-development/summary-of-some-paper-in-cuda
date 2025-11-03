# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**ArXiv:** [1810.04805](https://arxiv.org/abs/1810.04805)

## 🎯 Pitch

BERT introduces a breakthrough in natural language processing by pre-training deep bidirectional Transformer models using masked language modeling and next sentence prediction. This approach enables the model to learn rich representations conditioned on both left and right context, which can then be fine-tuned for a wide range of NLP tasks with minimal modifications. As a result, BERT achieves state-of-the-art performance across multiple benchmarks, dramatically improving accuracy while simplifying the process of building high-performing NLP systems.

---

## 1. Executive Summary (2–3 sentences)
BERT introduces a way to pre-train “deep bidirectional” language representations using a masked language model (MLM) and a next sentence prediction (NSP) task, then fine-tune the same network for many NLP tasks with minimal task-specific changes. This unlocks strong gains across sentence-level and token-level benchmarks (e.g., GLUE, SQuAD v1.1/v2.0, SWAG), establishing new state-of-the-art results while simplifying downstream model design (Figure 1; Tables 1–4).

## 2. Context and Motivation
- Problem addressed:
  - Existing pre-training strategies either (a) add pre-trained features to task-specific models (feature-based, e.g., ELMo) or (b) fine-tune a pre-trained model with minimal task-specific heads (fine-tuning, e.g., OpenAI GPT). Both rely on unidirectional language modeling (left-to-right or right-to-left), which restricts how context is used during pre-training and can hurt token-level tasks that need both left and right context. See Introduction and Related Work (Sections 1–2).
- Why it matters:
  - Many NLP tasks (question answering, named entity recognition, natural language inference) benefit from leveraging context from both directions and from learning relationships between sentence pairs. A pre-training scheme that provides such bidirectional and pair-aware representations can reduce task-specific engineering and improve performance across diverse tasks (Section 1).
- Shortcomings of prior approaches:
  - Unidirectionality: GPT’s left-to-right Transformer only attends to past tokens during pre-training, which is “sub-optimal for sentence-level tasks” and “very harmful” for token-level tasks where right context is crucial (Section 1).
  - Shallow bidirectionality: ELMo concatenates separate left-to-right and right-to-left models, but this “shallow concatenation” (Section 1) is less expressive than deep bidirectional conditioning in all layers (Figure 3).
  - Limited pre-training objectives for sentence pairs: Standard language modeling does not teach models how sentences relate (Section 3.1).
- How this work positions itself:
  - BERT trains a deep bidirectional Transformer encoder using an MLM objective (to enable true bidirectionality) and an NSP objective (to capture sentence-pair relationships), then fine-tunes the same architecture for many tasks with minimal output-layer changes (Figure 1).

## 3. Technical Approach
Step-by-step overview of BERT’s design, training, and fine-tuning.

- Model architecture (Section 3):
  - A multi-layer Transformer encoder (bidirectional self-attention).
  - Two sizes:
    - `BERTBASE`: `L=12` layers, `H=768` hidden size, `A=12` attention heads, ~110M parameters.
    - `BERTLARGE`: `L=24`, `H=1024`, `A=16`, ~340M parameters.
  - Key difference from GPT: bidirectional self-attention instead of left-to-right masking (Section 3; Figure 3).

- Input representation (Section 3; Figure 2):
  - Vocabulary: 30K WordPiece subword tokens.
  - Special tokens:
    - `[CLS]`: prepended to every sequence; its final hidden state `C ∈ RH` is used as a pooled sequence representation for classification.
    - `[SEP]`: separates segments (e.g., sentence A from B).
  - Segment embeddings: each token receives an added learned embedding indicating whether it belongs to “sentence A” or “sentence B.”
  - Positional embeddings: standard Transformer positional encodings are learned.
  - Final input embedding for each token is the sum of token, segment, and position embeddings (Figure 2).

- Pre-training objectives (Section 3.1; Figure 1):
  - `Masked Language Model (MLM)`:
    - Goal: enable deep bidirectional context by preventing the model from trivially seeing each token during prediction.
    - Procedure: randomly select 15% of input tokens; for each selected token:
      - 80% of the time replace with `[MASK]`,
      - 10% with a random token,
      - 10% leave unchanged.
    - The model predicts the original token at masked positions using a softmax over the vocabulary; loss is cross-entropy over the masked positions.
    - Rationale: mixing `[MASK]/random/same` reduces mismatch between pre-training (where `[MASK]` appears) and fine-tuning (where it does not) (Section 3.1; Appendix A.1, C.2).
  - `Next Sentence Prediction (NSP)`:
    - Goal: teach relationships between sentences for tasks like QA and NLI.
    - Procedure: build pairs (A,B) where 50% of the time `B` is the true next sentence (`IsNext`), and 50% `B` is a random sentence (`NotNext`).
    - The `[CLS]` vector `C` feeds a binary classifier for NSP (Section 3.1; Figure 1).

- Pre-training data and schedule (Appendix A.2):
  - Corpora: BooksCorpus (~800M words) + English Wikipedia (~2.5B words); only text passages (no tables/lists/headers). Document-level sampling is crucial for long contiguous sequences.
  - Sampling: two “sentences” (arbitrary spans) per example, with total length ≤512 tokens; apply masking after WordPiece tokenization.
  - Optimization:
    - Batch: 256 sequences × up to 512 tokens = 128K tokens/batch.
    - Steps: 1,000,000 steps (~40 epochs over 3.3B words).
    - Adam (lr 1e-4; β1=0.9; β2=0.999), weight decay 0.01, 10K-step warmup then linear decay, dropout 0.1, GELU activation.
    - Efficiency: train 90% of steps with sequence length 128, last 10% with 512 to learn longer-range position embeddings.
  - Hardware: `BERTBASE` on 4 Cloud TPUs in Pod configuration (16 chips) in ~4 days; `BERTLARGE` on 16 Cloud TPUs (64 chips) in ~4 days (Appendix A.2).

- Fine-tuning (Section 3.2; Figure 4):
  - Reuse the same architecture; add minimal task-specific layers; fine-tune all parameters end-to-end.
  - Single-sentence classification (e.g., SST-2): feed `[CLS]` vector `C` to a softmax classifier (Figure 4b).
  - Sentence-pair tasks (e.g., MNLI, QNLI, QQP): concatenate sentence A and B with `[SEP]`, mark segments, and feed `[CLS]` to a classifier (Figure 4a). Because self-attention spans both segments, the model implicitly performs cross-attention (Section 3.2).
  - Token-level tasks:
    - NER: use token-level representations `Ti` for label prediction (Figure 4d).
    - SQuAD v1.1: learn two vectors `S` and `E` for start and end; score for token `i` is `S·Ti` (start) and `E·Ti` (end); score of span (i,j) is `S·Ti + E·Tj`, choosing `j ≥ i` with highest score (Section 4.2).
    - SQuAD v2.0: allow “[CLS]” as a special “no-answer” span, compare score of best non-null span to `s_null = S·C + E·C` with a tuned threshold (Section 4.3).
  - Hyperparameters: typically batch size 16–32, learning rate 2e-5 to 5e-5, 2–4 epochs; fine-tuning is “relatively inexpensive,” often ≤1 hour on a single Cloud TPU or a few hours on a GPU (Section 3.2; Appendix A.3).

- Why these design choices?
  - Deep bidirectionality: unidirectional LMs cannot condition on right context; shallow concatenation (ELMo) is strictly less expressive than a model that fuses left and right context in every layer (Section 1; Figure 3).
  - NSP: adds a supervision signal for sentence relations that plain LMs lack, benefiting QA/NLI (Sections 3.1, 5.1).
  - Mixed masking: reduces pre-train/fine-tune mismatch by not always using `[MASK]` (Section 3.1; Appendix C.2).
  - Unified architecture: avoids building specialized cross-attention modules or task-specific encoders by letting self-attention handle interactions when sequences are concatenated (Section 3.2; Figure 4).

## 4. Key Insights and Innovations
- Deep bidirectional pre-training via MLM (fundamental):
  - Unlike left-to-right LMs or shallow biLM concatenation, BERT fuses left and right context in all layers by masking some tokens and predicting them from their context (Section 3.1; Figure 3). Ablation shows a left-to-right-only model underperforms substantially (Table 5).
- Joint pre-training for sentence pairs using NSP (incremental but impactful):
  - NSP adds a simple binary task that meaningfully improves tasks relying on inter-sentence reasoning (QA, NLI). Removing NSP reduces QNLI/MNLI/SQuAD performance (Table 5).
- Unified fine-tuning interface across tasks (practical and impactful):
  - The same encoder can be used for token-level and sentence-level tasks; only a minimal head is added (Figures 1 and 4). This reduces architecture-specific engineering while improving SOTA on many benchmarks (Tables 1–4).
- Scaling pre-trained encoder size benefits even small downstream datasets (empirical insight):
  - Larger `L/H/A` yields consistent improvements across GLUE tasks, including those with only thousands of labels (MRPC), indicating that downstream fine-tuning can exploit very large pre-trained representations (Table 6).
- Effective both as a fine-tuned model and as a fixed feature extractor (practical insight):
  - While fine-tuning is best, BERT’s frozen representations combined with a light BiLSTM achieve near-finetuned performance on NER (Table 7).

## 5. Experimental Analysis
- Evaluation methodology:
  - Datasets and tasks:
    - GLUE benchmark (mix of 8 tasks reported; WNLI excluded due to known issues) spanning entailment, paraphrase, similarity, sentiment, and linguistic acceptability (Section 4.1; Appendix B.1).
    - SQuAD v1.1 and v2.0 for extractive QA (Sections 4.2–4.3).
    - SWAG for commonsense sentence continuation (Section 4.4).
    - CoNLL-2003 NER for sequence labeling and for evaluating feature-based usage (Section 5.3).
  - Metrics: accuracy (MNLI, QNLI, SST-2, RTE), F1 (QQP, MRPC, SQuAD), Spearman correlation (STS-B). GLUE “Average” excludes WNLI (Table 1).
  - Baselines: pre-OpenAI SOTA, BiLSTM+ELMo+Attn, OpenAI GPT on GLUE; contemporaneous QA systems on SQuAD; ESIM+ELMo and GPT on SWAG; top NER methods for CoNLL-2003 (Tables 1–4, 7).
  - Setup: consistent fine-tuning search over learning rates, batch sizes, epochs; random restarts for `BERTLARGE` on small datasets due to instability (Section 4.1).

- Main results with figures/tables:
  - GLUE (Table 1):
    - “BERTBASE … 79.6 avg” and “BERTLARGE … 82.1 avg,” both outperform GPT’s 75.1.
    - On MNLI, `BERTLARGE` reaches “86.7/85.9” (matched/mismatched), a +4.6 absolute improvement over prior SOTA.
    - The paper also reports “BERTLARGE obtains a score of 80.5” on the official GLUE leaderboard (Section 4.1).
  - SQuAD v1.1 (Table 2):
    - Single `BERTLARGE` Dev “EM 84.1 / F1 90.9.”
    - With TriviaQA pre-finetuning and ensembling: Test “EM 87.4 / F1 93.2,” surpassing prior top systems.
    - Even single `BERTLARGE` is competitive with ensemble baselines.
  - SQuAD v2.0 (Table 3):
    - Single `BERTLARGE` achieves Test “EM 80.0 / F1 83.1,” described as a +5.1 F1 improvement over the previous best (Section 4.3; Table 3).
  - SWAG (Table 4):
    - `BERTLARGE` reaches Test accuracy “86.3,” improving over GPT’s “78.0” by +8.3 points.

- Ablation studies and robustness checks:
  - Effect of pre-training tasks (Table 5):
    - Removing NSP (“No NSP”) hurts QNLI (−3.5 points) and MNLI (−0.5) and slightly reduces SQuAD (−0.6 F1) relative to the `BERTBASE` row.
    - Left-to-right only (“LTR & No NSP”) drops MRPC accuracy from 86.7 to 77.5 and SQuAD F1 from 88.5 to 77.8; adding a BiLSTM partially recovers SQuAD to 84.9 but still lags and hurts GLUE tasks.
  - Effect of model size (Table 6):
    - As parameters increase from (L=3, H=768, A=12) to `BERTLARGE` (L=24, H=1024, A=16), MNLI-m improves from 77.9 to 86.6 and SST-2 from 88.4 to 93.7. Gains persist even for small MRPC (from 79.8 to 87.8).
    - Masked LM perplexity on held-out data monotonically decreases with size (from 5.84 to 3.23), aligning with downstream improvements.
  - Feature-based vs fine-tuning on NER (Table 7):
    - Fine-tuned `BERTLARGE` Test F1 “92.8,” close to top systems; best feature-based variant (concat last 4 layers) Dev F1 “96.1” (no test given), only 0.3 behind fine-tuning on Dev.
  - Masking strategy sensitivity (Appendix C.2; Table 8):
    - Fine-tuning is robust across masking variants, but feature-based NER suffers when only `[MASK]` is used (Dev 94.0 vs 94.9 for the mixed 80/10/10 strategy).
  - Training steps and convergence (Appendix C.1; Figure 5):
    - Fine-tuned MNLI accuracy continues to improve up to 1M pre-training steps; MLM converges slightly slower than left-to-right in raw speed but surpasses it almost immediately in accuracy.

- Do the experiments support the claims?
  - Yes. The controlled ablations (Table 5) isolate the importance of MLM’s deep bidirectionality and the added NSP signal. Consistent improvements across diverse benchmarks (Tables 1–4) and scaling analysis (Table 6) strengthen the central claims. Robustness to masking variations (Table 8) and to using BERT as features (Table 7) shows practical flexibility.

## 6. Limitations and Trade-offs
- Compute and data requirements:
  - Pre-training is compute-intensive: 1M steps with 128K tokens per batch on TPUs for ~4 days; `BERTLARGE` uses 64 TPU chips (Appendix A.2). This is a barrier for many practitioners.
  - Requires large document-level corpora (BooksCorpus + Wikipedia) and benefits from long sequences, which some domains lack (Section 3.1; Appendix A.2).
- Pre-train / fine-tune mismatch:
  - `[MASK]` never appears during fine-tuning; the mixed 80/10/10 masking reduces but does not eliminate mismatch (Section 3.1; Appendix C.2).
- Sequence length constraints:
  - Pre-training and fine-tuning use maximum length 512; most steps use length 128 for efficiency. Very long-range dependencies beyond 512 tokens are not modeled (Appendix A.2).
- Stability on small datasets:
  - `BERTLARGE` can be “unstable on small datasets,” requiring multiple random restarts (Section 4.1).
- Task coverage:
  - The framework suits understanding tasks; the encoder-style architecture is not designed for left-to-right text generation. Some tasks may still require task-specific architectures, for which BERT can serve as a feature extractor (Section 5.3).
- Objective design:
  - NSP is simple and beneficial in this paper (Table 5), but it is a coarse supervision signal (binary next vs random); more nuanced inter-sentence objectives might further help, especially for discourse-level reasoning (discussion in Sections 3.1, 5.1 hints at its role).

## 7. Implications and Future Directions
- Field impact:
  - BERT shifts NLP pre-training from unidirectional LMs and shallow bidirectionality to deep bidirectional encoders trained with MLM. It establishes a general recipe: pre-train a large bidirectional Transformer on unlabeled text with token- and sentence-level self-supervision, then fine-tune for many tasks with minimal heads (Figure 1; Sections 3–4).
- Practical applications:
  - Rapidly build high-performing models for:
    - Search and QA (SQuAD v1.1/v2.0, Section 4.2–4.3),
    - Natural language inference and classification (GLUE, Section 4.1),
    - Commonsense sentence continuation (SWAG, Section 4.4),
    - Token-level labeling (NER, Section 5.3).
  - Particularly valuable for low-resource labeled settings, where pre-trained representations substantially lift performance (Table 6).
- Follow-up research directions suggested by the work’s structure and findings:
  - Objectives: Explore alternative or richer sentence-pair objectives beyond binary NSP; design masking schemes that further close the pre-train/fine-tune gap (Appendix C.2 shows there is room).
  - Scaling and efficiency: Investigate more efficient training (e.g., longer sequences, better memory use, adaptive sequence lengths) to reduce the heavy compute budget (Appendix A.2).
  - Domain and multilingual transfer: Pre-train on domain-specific or multilingual corpora to broaden applicability (the framework is agnostic to language/domain; Section 3).
  - Task interfaces: Extend the unified fine-tuning interface to structured prediction tasks that need decoding constraints (e.g., integrating CRFs for NER if needed; Table 7 mentions no CRF was used).
  - Robustness and stability: Methods for more stable fine-tuning on small datasets (Section 4.1 notes instability) and for effectively leveraging BERT as fixed features when fine-tuning is not feasible (Section 5.3).

> Core takeaways supported by the paper’s figures/tables:
> - Deep bidirectionality via MLM is the key enabler (Figure 3; Table 5).
> - Simple, unified fine-tuning heads suffice across tasks (Figure 4).
> - Performance gains are broad and substantial: GLUE average 82.1 (`BERTLARGE`), SQuAD v1.1 Test F1 93.2 (ensemble), SQuAD v2.0 Test F1 83.1, SWAG Test 86.3 (Tables 1–4).
> - Bigger pre-trained encoders consistently help—even for small datasets (Table 6).
> - BERT works both as a fine-tuned model and as a feature extractor (Table 7).
