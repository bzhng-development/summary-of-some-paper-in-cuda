# Deep contextualized word representations

**ArXiv:** [1802.05365](https://arxiv.org/abs/1802.05365)

## 🎯 Pitch

This paper introduces ELMo: deep contextualized word representations that generate word embeddings as a function of the entire input sentence by leveraging all layers of a large bidirectional language model (biLM). This innovation not only models complex, context-sensitive aspects of word meaning—including polysemy and syntax—but also delivers substantial improvements across a wide range of NLP tasks, setting new state-of-the-art results and making deep context integration easily accessible for existing models.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces ELMo (“Embeddings from Language Models”), a way to turn each word token in a sentence into a vector that depends on the whole sentence, not just the word type. By exposing and learning from all internal layers of a large bidirectional language model (biLM), ELMo yields substantial, consistent improvements across six challenging NLP tasks and offers a practical, drop-in method that models both syntax and meaning as they vary with context.

## 2. Context and Motivation
- The specific problem or gap
  - Most pre-trained word vectors (e.g., GloVe, word2vec) assign a single, context-independent vector to each word type. This fails to model polysemy (a word having multiple meanings) and does not adapt the representation to the sentence in which a word occurs (Abstract; Sec. 1).
  - Prior contextual methods often use only the top layer of a neural encoder, missing the different linguistic signals that accumulate at different depths (Sec. 1–2, 5.1).

- Why this matters
  - Contextual meaning is essential for tasks like question answering, entailment, and tagging. Real-world systems need robust understanding of syntax and semantics in context to reduce labeled data needs and improve generalization (Sec. 1; Table 1; Fig. 1).

- Prior approaches and limitations
  - Traditional embeddings: Single vector per word type (Mikolov et al., Pennington et al.), no context adaptation (Sec. 2).
  - Enriched type embeddings (subword, multi-sense vectors): Help but still not fully contextual or token-specific (Sec. 2).
  - Contextual encoders:
    - context2vec: Encodes context around a pivot word but does not deeply expose multiple internal layers (Sec. 2).
    - CoVe: Contextual vectors from a neural machine translation (NMT) encoder; effective but depends on parallel data and primarily uses top-layer features (Sec. 2, 5.1, 5.3).
    - TagLM: Uses top-layer of a bidirectional language model (biLM), but not a deep mixture of all layers (Sec. 1–2).

- How this paper positions itself
  - ELMo provides deep contextualized representations by learning a task-specific weighted combination over all layers of a large, pre-trained biLM trained on abundant monolingual data (Sec. 3.2, 3.4).
  - It’s a drop-in addition to existing models with minimal changes, improving state-of-the-art results across diverse tasks (Table 1; Sec. 4).

## 3. Technical Approach
High-level idea: Pretrain a large bidirectional language model (biLM) on massive unlabeled text; extract all its internal hidden states at each token; learn a small number of task-specific scalar weights to mix these states into a single token embedding (`ELMo`) that you concatenate into your supervised model. This exposes multiple kinds of linguistic information (syntax, semantics) captured at different depths.

- Key terms (defined on first use)
  - `biLM` (bidirectional language model): Two language models over the same sequence—one processes text left-to-right (predicts the next word) and one right-to-left (predicts the previous word). Their internal states capture context from both directions (Sec. 3.1).
  - `ELMo`: A task-specific, learned weighted sum of all layers of the pre-trained biLM, scaled by a learned scalar, producing context-sensitive vectors for each token (Sec. 3.2).
  - `Perplexity`: A standard language-modeling metric; lower means the model is better at predicting the next (or previous) token.

- Step-by-step method
  1. Pretrain a large biLM on unlabeled text (Sec. 3.1, 3.4).
     - For a sequence of tokens `(t1, …, tN)`, the forward LM models `p(tk | t1..tk-1)`; the backward LM models `p(tk | tk+1..tN)` (Sec. 3.1).
     - Token representation `x_k^LM` is computed from characters via a convolutional neural net (CNN), enabling open-vocabulary handling (Sec. 3.1, 3.4).
     - Each of `L` layers in each direction produces a context-dependent vector at each position: forward `h_{k,j}^→`, backward `h_{k,j}^←` (Sec. 3.1).
     - The paper uses `L = 2` layers with 4096 units each and 512-d projections, plus residual connections between layers; character CNN uses 2048 filters, followed by highway layers and a 512-d projection (Sec. 3.4).
     - The forward and backward models share the token representation and softmax parameters (“weight tying”), but have separate LSTM parameters (Sec. 3.1).
     - Trained for 10 epochs on the 1B Word Benchmark (~30M sentences); average forward+backward perplexity ≈ 39.7 (Sec. 3.4).
  2. Build a deep contextual token representation (ELMo) by mixing layers (Sec. 3.2).
     - For token `k`, collect all `2L + 1` layer outputs: token layer (`h_{k,0}^LM = x_k^LM`) and concatenated biLSTM layer outputs (`h_{k,j}^LM = [h_{k,j}^→; h_{k,j}^←]` for `j = 1..L`).
     - Learn a small set of task-specific scalar weights over layers, softmax-normalized (`s_j^task`), and a global scaling parameter (`γ^task`).
     - Equation (1) defines the representation (Sec. 3.2):
       - `ELMo_k^task = γ^task * sum_{j=0}^L s_j^task * h_{k,j}^LM`
     - Optional: layer-normalize each biLM layer before weighting (Sec. 3.2).
  3. Integrate ELMo into a supervised model (Sec. 3.3).
     - Freeze the biLM weights.
     - Concatenate `ELMo_k^task` with the model’s usual token representation `x_k` at the input to the model’s contextual encoder (e.g., biRNN). In some tasks, add a second ELMo pathway at the output of the contextual encoder by concatenating `ELMo_k^task` to `h_k` (Sec. 3.3; Table 3).
     - Regularization: add dropout to ELMo vectors (and optionally add an `L2` penalty `λ ||w||^2 / 2` on the ELMo layer weights to bias them toward averaging) (Sec. 3.3).
     - A practical optimization detail: the learned scale `γ^task` is important for stable training; removing it caused poor or failed training in ablations (Supplement A.2).
  4. (Optional) Domain adaptation of the biLM (Sec. 3.4; Supplement A.1).
     - Temporarily ignore labels, fine-tune the biLM for one epoch on the target task’s training text, choose the fine-tuned checkpoint using dev perplexity, then fix the biLM during supervised training (Supplement A.1; Table 7).

- Design choices and rationale
  - Deep mixture over all layers rather than top-only: Lower layers tend to encode syntax; higher layers, semantics. Tasks benefit by learning how much to use each (Sec. 5.1, 5.3; Table 2; Table 5–6; Fig. 2).
  - Character-CNN input: Handles out-of-vocabulary tokens and subword patterns without a fixed word vocabulary (Sec. 3.4).
  - Freezing the biLM during supervised training: Lets the task focus on learning to use the representations, not change them; reduces overfitting risk when labeled data are limited (Sec. 3.3, 3.4).
  - Optional domain fine-tuning: Improves perplexity and sometimes downstream performance when unlabeled in-domain text is available (Supplement A.1; Table 7).

- A concrete intuition: contextual disambiguation
  - Table 4 contrasts nearest neighbors for the word “play.” GloVe neighbors mix parts of speech and sports senses. In contrast, biLM context embeddings retrieve sentences matching the correct part-of-speech and sense (e.g., theater vs. sports), showing that context is encoded into token-level meaning (Sec. 5.3; Table 4).

## 4. Key Insights and Innovations
- Deep, task-learned mixture of all LM layers (fundamental innovation)
  - What’s new: Instead of using only the top LM layer, ELMo learns a task-specific weighted sum across token, lower-layer, and upper-layer states, with a learned global scale `γ` (Sec. 3.2; Eq. 1).
  - Why it matters: Ablations show mixing all layers outperforms last-layer-only on multiple tasks (Sec. 5.1; Table 2).
    - Quote: “Including representations from all layers improves overall performance over just using the last layer.” (Table 2)

- Expose different linguistic signals to downstream models (fundamental insight)
  - Finding: Lower biLM layers better encode syntax; upper layers better encode context-sensitive semantics (Sec. 5.3; Table 5–6).
    - POS tagging (syntax) is highest with the first layer (97.3% on PTB), while WSD (semantics) is highest with the second layer (69.0 F1) (Table 6, Table 5).
  - Impact: Letting each task learn its own mixture yields consistent gains and reveals interpretable layer preferences (Fig. 2).

- Monolingual LM pretraining beats NMT-based context (incremental but important)
  - ELMo’s biLM (trained on large monolingual data) outperforms CoVe (NMT encoder) on downstream tasks and intrinsic probes (Sec. 5.1, 5.3; Table 1; Table 5–6).
    - WSD: biLM second layer 69.0 vs. CoVe second layer 64.7 (Table 5).
    - POS: biLM first layer 97.3 vs. CoVe first layer 93.3 (Table 6).

- Practical, sample-efficient drop-in gains across tasks (practical innovation)
  - ELMo substantially reduces training updates and labeled data requirements (Sec. 5.4; Fig. 1).
    - Quote: “SRL exceeds the baseline maximum at epoch 10 (vs. 486 without ELMo), a 98% relative decrease in updates.” (Sec. 5.4)

## 5. Experimental Analysis
- Evaluation setup
  - Tasks and datasets (Sec. 4; Tables 8–13):
    - Question Answering: SQuAD (span prediction) with F1 and Exact Match (EM) (Table 9).
    - Textual Entailment: SNLI (accuracy) (Table 8).
    - Semantic Role Labeling (SRL): CoNLL 2012 / OntoNotes (F1) (Table 10).
    - Coreference Resolution: CoNLL 2012 (average F1 across MUC/B³/CEAF-φ4) (Table 11).
    - Named Entity Recognition (NER): CoNLL 2003 (F1) (Table 12).
    - Sentiment Analysis: SST-5 (five-way sentence classification accuracy) (Table 13).
  - Baselines: Strong, near-SOTA architectures per task (Sec. 4; per-task Supplements A.3–A.8).
  - Integration: Add ELMo at input (always), and sometimes also at output, with frozen biLM, per Sec. 3.3.

- Main results (Table 1; task-specific tables)
  - Table 1 (single-model improvements over strong baselines):
    - Quote: “SQuAD: 81.1 → 85.8 F1 (+4.7 abs, 24.9% relative error reduction). SNLI: 88.0 → 88.7. SRL: 81.4 → 84.6. Coref: 67.2 → 70.4. NER: 90.15 → 92.22. SST-5: 51.4 → 54.7.”
  - Leaderboard/SoTA comparisons:
    - SQuAD single model: ELMo-enhanced model achieves 85.8 F1, +1.4 over prior SOTA SAN (84.4 F1); 11-model ensemble reaches 87.4 F1 (Table 9).
      - Quote: “BiDAF + Self Attention + ELMo: 78.6 EM / 85.8 F1; Ensemble: 81.0 EM / 87.4 F1.” (Table 9)
    - SNLI single model: ESIM+ELMo = 88.7 ± 0.17; 5-model ensemble 89.3, exceeding prior ensemble best 88.9 (Table 8).
    - SRL single model: 84.6 F1, surpassing prior single and even prior ensemble results (Table 10).
    - Coreference: 70.4 average F1, surpassing previous single and ensemble systems (Table 11).
    - NER: 92.22 ± 0.10 F1, new SOTA; improvement due to using all biLM layers vs. top-only (Table 12; Sec. 4).
    - SST-5: 54.7 accuracy, improving over BCN+Char+CoVe (53.7) (Table 13).

- Ablation and diagnostic studies
  - Layer mixing matters (Sec. 5.1; Table 2):
    - Quote: “Using all layers improves over last-only across SQuAD, SNLI, SRL; learned weights (small λ) beat uniform averaging.” (Table 2)
  - Where to add ELMo (Sec. 5.2; Table 3):
    - Quote: “SQuAD and SNLI benefit from ELMo at both input and output; SRL (and coref) are best with input-only; adding at output hurts SRL (80.9 vs. 84.7 at input-only).” (Table 3)
  - Intrinsic probes (Sec. 5.3; Tables 5–6; Table 4):
    - WSD: biLM second layer (69.0 F1) competitive with specialized WSD models and above WordNet 1st-sense baseline (65.9) (Table 5).
    - POS: biLM first layer (97.3%) near top-tier task-specific models (97.6–97.8), much higher than CoVe (Table 6).
    - Nearest neighbor example “play” shows context-driven disambiguation (Table 4).
  - Sample efficiency (Sec. 5.4; Fig. 1):
    - Quote: “ELMo enables higher accuracy with smaller training sets; for SRL, 1% with ELMo ≈ 10% without.” (Fig. 1)
  - Learned weights visualization (Sec. 5.5; Fig. 2):
    - Quote: “Input-layer mixtures often emphasize lower biLM layers; output-layer mixtures are more balanced.” (Fig. 2)
  - Stabilization parameter γ (Supplement A.2):
    - Quote: “Without γ, last-only performed poorly for SNLI and failed for SRL.” (A.2)

- Assessment of claims
  - The broad, consistent gains in Table 1, confirmed across multiple architecture families and tasks, support the claim that deep contextualization via ELMo is generally beneficial.
  - Ablations (Tables 2–3) credibly isolate that performance comes from deep layer mixing and judicious placement, not just extra parameters.
  - Intrinsic probes (Tables 5–6) substantiate the interpretation that lower layers encode syntax and higher layers encode semantics in the biLM, justifying the design of learning to mix layers.

- Notable conditions and trade-offs
  - “Where to include ELMo” is task-dependent (Table 3).
  - Choosing regularization strength `λ` for the ELMo weights matters; small `λ` (more flexible weights) typically helps, except on small datasets like NER where results are insensitive (Sec. 5.1).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Sentence-level context: The biLM models sequences token-by-token; document-level or cross-sentence phenomena are not explicitly modeled in pretraining (Sec. 3.1–3.4).
  - Global, task-level mixing: The layer weights `s_j^task` are global scalars for a task (and placement), not dynamically adapted per token or sentence. This limits per-instance customization of which layers to trust (Sec. 3.2).

- Computational and resource considerations
  - Pretraining cost: Training a large biLM with 2×4096-unit LSTMs and character CNN on the 1B Word Benchmark is compute-intensive (Sec. 3.4).
  - Inference overhead: Downstream models must run the biLM to extract all layers for each token, increasing memory and compute versus using a single static embedding (Sec. 3.3).

- Data and domain shift
  - While optional domain fine-tuning helps perplexity (Table 7), its impact on downstream metrics is task-dependent (Supplement A.1). Gains may vary with domain mismatch.

- Architectural choices
  - LSTM-based LM: At the time, this was state-of-the-art for LMs, but other architectures (e.g., deep transformers) might capture different or richer contextual patterns. Extending ELMo’s “deep mixture” idea to other encoders is not explored in this paper (Sec. 3.4 positions towards biLMs).

- Mixed or conditional results
  - Placement sensitivity: Adding ELMo at the output can help (SQuAD, SNLI) or hurt (SRL, coref) (Table 3). Users must validate placement per task.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes “deep contextualization” as a general recipe: pretrain a large bidirectional LM, expose all its layers, and learn task-specific mixtures. The paper demonstrates that this simple, modular step reliably improves diverse NLP systems (Table 1).

- What it enables next
  - Extending the mixing idea: Token- or sentence-specific attention over layers, rather than global scalar weights (`s_j^task`), could further adapt the representation to input-specific needs (Eq. 1 invites such generalization).
  - Alternative encoders: Apply the “deep mixture over layers” principle to other pretraining paradigms (e.g., different architectures or objectives), potentially improving both efficiency and accuracy beyond biLSTMs (Sec. 5.3 trend suggests transferability of the idea).
  - Multi-task and low-resource learning: Given the strong sample efficiency gains (Fig. 1), ELMo-like features can be valuable for low-data settings, domain adaptation, and multi-task setups that share a single pre-trained contextual backbone.

- Practical applications and use cases
  - Drop-in module: For production systems in QA, IE/NER, SRL, coref, and classification, adding ELMo is a high-ROI change: freeze a pre-trained biLM, learn per-task mixing weights, and concatenate ELMo at the model’s input (and optionally at the output) (Sec. 3.3; Table 3).
  - Open vocabulary handling: The character CNN at the LM input makes ELMo robust to rare or unseen tokens, benefiting domains with many variants (Sec. 3.4).
  - Domain transfer: Where unlabeled in-domain text is available, the paper’s simple, one-epoch fine-tuning recipe often reduces perplexity, sometimes improving task performance (Supplement A.1; Table 7).

> Key take-away: Deep, context-sensitive word representations built as a learned mixture over all layers of a biLM (Eq. 1) provide a principled, data-efficient, and broadly applicable way to inject rich linguistic signals—syntax from lower layers and semantics from higher layers—into any neural NLP model, with consistent, state-of-the-art gains documented in Table 1.
