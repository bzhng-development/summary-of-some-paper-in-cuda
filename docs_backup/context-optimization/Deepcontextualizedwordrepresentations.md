# Deep contextualized word representations

**ArXiv:** [1802.05365](https://arxiv.org/abs/1802.05365)

## 🎯 Pitch

This paper introduces ELMo, a groundbreaking approach that generates context-sensitive word representations by leveraging all internal layers of a deep, pre-trained bidirectional language model (biLM). By allowing each downstream NLP task to learn its own optimal combination of these layers, ELMo not only captures both syntactic and semantic nuances of word usage as they vary with context, but also delivers dramatic improvements across a wide range of language understanding benchmarks. Its success marks a paradigm shift from static word embeddings to dynamic, task-adaptive representations, unlocking higher accuracy, better sample efficiency, and richer linguistic modeling for diverse NLP applications.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces ELMo (“Embeddings from Language Models”), a way to represent each word in a sentence with a context-dependent vector built from all internal layers of a large, pre-trained bidirectional language model (`biLM`). By letting each downstream task learn its own weighted mix of these layers (Eq. (1), Sec. 3.2), the approach consistently improves state-of-the-art results across six diverse NLP benchmarks (Table 1), while also increasing data efficiency and revealing that different layers capture different linguistic information (syntax vs. semantics).

## 2. Context and Motivation
- Specific gap addressed
  - Most popular word embeddings (e.g., Word2Vec, GloVe) assign a single, context-independent vector to each word type. This cannot model polysemy (a word with multiple meanings) or context-specific usage. The paper targets this gap by producing token-level vectors that vary with context (Abstract; Sec. 1).
- Why it matters
  - Real-world language understanding depends on context: “play” in “the actors rehearse the play” vs. “the kids play outside.” Context-sensitive meaning impacts many NLP problems—question answering, entailment, information extraction, etc.—and limits the effectiveness of single-vector approaches (Sec. 1).
- Prior approaches and shortcomings
  - Multi-sense embeddings learn multiple vectors per word but still require discrete sense assignments (Related Work, Sec. 2).
  - Contextual methods existed: `context2vec` (predicts pivot words with a biLSTM), CoVe (uses a machine translation encoder), and earlier LM-based features (TagLM), but typically:
    - They used only the top layer of an encoder/LM for representations.
    - They were constrained by data (e.g., CoVe relies on parallel corpora) or did not expose multiple internal levels of representation to downstream tasks (Sec. 2).
- Positioning
  - ELMo derives token representations from all internal layers of a deep `biLM`, and each task learns a linear combination (a weighted sum) of these layers. This “deep contextualization” exposes multiple types of information (lower layers more syntactic, higher layers more semantic) to task models, producing larger and broader improvements than prior methods (Sec. 1, 3.2, 5.1, 5.3).

## 3. Technical Approach
High-level idea: Pre-train a large bidirectional language model (`biLM`) on massive unlabeled text, then, for any supervised NLP task, compute per-token vectors from all internal `biLM` layers and let the task learn how to mix them.

- Step 1: Pre-train a bidirectional language model (Sec. 3.1, 3.4)
  - What a language model (LM) does:
    - A forward LM assigns probabilities to a token `t_k` given the history `(t_1, ..., t_{k-1})`.
    - A backward LM does the reverse, predicting `t_k` from future tokens `(t_{k+1}, ..., t_N)`.
  - What “bidirectional” means here:
    - Train both forward and backward LMs together; share token representation and softmax parameters but use separate LSTM weights in each direction.
    - Joint objective: maximize the sum of forward and backward log-likelihoods over the sequence (Sec. 3.1).
  - Architecture details (Sec. 3.4):
    - Input representation uses a character-level CNN (“char-CNN”) to build token vectors—this supports open vocabulary and captures subword patterns.
    - The `biLM` has L = 2 layers of bidirectional LSTMs; each LSTM has 4096 units with 512-dimensional projections and a residual connection between layers. The token representation is 512-dimensional after projection.
    - Trained for 10 epochs on the 1 Billion Word Benchmark (approx. 30M sentences). Average forward/backward perplexity ≈ 39.7 (lower is better).
    - Fine-tuning on in-domain unlabeled text (ignoring task labels) sometimes reduces perplexity markedly and can boost downstream performance (Supplement A.1; Table 7).

- Step 2: Represent each token using all `biLM` layers (Sec. 3.2)
  - For each token position `k`, the `biLM` yields:
    - `h^LM_{k,0}` = token-layer representation (from the char-CNN).
    - `h^LM_{k,j}` for `j = 1..L` = concatenation of forward and backward LSTM hidden states at layer `j`.
  - ELMo constructs a task-specific vector as a weighted sum of these layers (Equation (1)):
    - `ELMo^task_k = γ^task * sum_{j=0..L} s^task_j * h^LM_{k,j}`,
    - where `s^task` are softmax-normalized scalar weights learned during task training and `γ^task` is a learned scalar that rescales the whole mixture to match the task model’s activation scale.
    - Optional layer normalization can be applied to `biLM` layers before mixing (Sec. 3.2).
  - Why a weighted sum?
    - Different layers encode different information; letting each task learn `s_j` allows it to emphasize what it needs (Sec. 5.3; Figure 2).

- Step 3: Inject ELMo into supervised task models (Sec. 3.3)
  - Freeze `biLM` weights; compute all layer representations for the task’s input tokens.
  - Concatenate `ELMo^task_k` to the task’s token representation `x_k` (the usual word embedding + optional char features) before the task’s encoder (e.g., biRNN). In some tasks, also concatenate ELMo to the task encoder’s output representation `h_k` (Table 3).
  - Regularization and stabilization:
    - Dropout on ELMo vectors often helps (Sec. 3.3).
    - Optionally regularize the `s_j` layer weights to prefer a more uniform mixture (add `λ ||w||^2_2` to the task loss; Sec. 3.3, 5.1).
    - The `γ` scale parameter is practically important for optimization; without it, training can fail in last-layer-only settings (Supplement A.2).

- How it works in practice (intuitive example)
  - The method disambiguates word meaning using context. For the word “play,” static embeddings cluster around sports senses, but ELMo’s context representation retrieves nearest neighbors matching the correct sense and part-of-speech in context (Table 4, Sec. 5.3). This illustrates how the `biLM`’s internal states encode context-sensitive semantics and syntax, which ELMo exposes to downstream tasks.

- Design choices and rationale
  - Character-based token encoding: handles rare/OOV words and morphology (Sec. 3.4).
  - Deep layer mixing: lower layers capture syntactic regularities; higher layers capture semantics (validated by POS vs. WSD analyses in Sec. 5.3; Tables 5–6).
  - Freezing the `biLM` during task training: preserves general-purpose representations and avoids overfitting, especially when labeled data are small (Sec. 3.3; Sec. 5.4).

## 4. Key Insights and Innovations
- Deep, task-specific layer mixing (fundamental)
  - Novelty: Instead of using only the top layer of a contextual encoder, ELMo lets each task learn a weighted mixture of all `biLM` layers (Eq. (1), Sec. 3.2).
  - Why it matters: Ablations show that mixing all layers outperforms last-layer-only across tasks; learning non-uniform weights beats uniform averaging (Table 2, Sec. 5.1). This exposes diverse linguistic signals (syntax vs. semantics) needed by different tasks (Sec. 5.3, Figure 2).

- Simple, general integration into existing models (practical innovation)
  - Novelty: ELMo is added by concatenation at the input (and optionally output) of standard task encoders without redesigning architectures (Sec. 3.3).
  - Why it matters: It consistently improves strong baselines across six tasks, often setting new SOTA as a drop-in feature (Table 1; Sec. 4).

- Large-scale, character-based `biLM` pretraining (incremental but consequential)
  - Novelty: The paper scales a purely character-based, deep `biLM` trained on ≈30M sentences, then fine-tunes it on unlabeled in-domain text when appropriate (Sec. 3.4; Supplement A.1).
  - Why it matters: It avoids dependency on parallel corpora (a limitation for MT-based CoVe) and yields richer, more transferable features (ELMo > CoVe in direct comparisons; Sec. 5.1, 5.3).

- Empirical evidence that different layers encode different linguistic information (fundamental)
  - Novelty: Intrinsic probes show top `biLM` layers excel at word sense disambiguation (WSD), while lower layers better capture part-of-speech (POS) information (Sec. 5.3; Tables 5–6).
  - Why it matters: It validates the core premise for deep layer mixing and explains performance gains across varied tasks.

- Improved sample efficiency (practical)
  - Novelty: With ELMo, models reach baseline performance using fewer updates and less labeled data (Sec. 5.4; Figure 1).
  - Why it matters: It reduces labeling and training costs while improving accuracy.

## 5. Experimental Analysis
- Evaluation methodology
  - Datasets and metrics (Sec. 4; Tables 1, 8–13):
    - `SQuAD` (QA): F1 and Exact Match (EM).
    - `SNLI` (Textual Entailment/NLI): accuracy.
    - `SRL` on OntoNotes CoNLL-2012: F1.
    - `Coreference` on CoNLL-2012: average F1.
    - `NER` on CoNLL-2003: F1.
    - `SST-5` (fine-grained sentiment): accuracy.
  - Baselines: Competitive published baselines for each task (e.g., BiDAF variants for SQuAD, ESIM for SNLI, He et al. 2017 for SRL, Lee et al. 2017 for coreference, biLSTM-CRF for NER, BCN for SST-5). Detailed model configs are in Supplemental Material (Sec. A.3–A.8).

- Main quantitative results (Table 1; Sec. 4)
  - Across six benchmarks, ELMo improves the baseline and/or sets new single-model state of the art:
    - > “SQuAD: baseline F1 81.1 → ELMo 85.8; +4.7 absolute / 24.9% relative error reduction” (Table 1).
    - > “SNLI: baseline 88.0 → ELMo 88.7 ± 0.17; +0.7 absolute / 5.8% relative error reduction” (Table 1).
    - > “SRL: baseline 81.4 → ELMo 84.6; +3.2 absolute / 17.2% relative error reduction” (Table 1).
    - > “Coref: baseline 67.2 → ELMo 70.4; +3.2 absolute / 9.8% relative error reduction” (Table 1).
    - > “NER: baseline 90.15 → ELMo 92.22 ± 0.10; +2.06 absolute / 21% relative error reduction” (Table 1).
    - > “SST-5: baseline 51.4 → ELMo 54.7 ± 0.5; +3.3 absolute / 6.8% relative error reduction” (Table 1).
  - Leaderboard-level outcomes:
    - SQuAD single-model F1 85.8 (EM 78.6) and 11-model ensemble F1 87.4 (EM 81.0), both state-of-the-art at submission time (Table 9).
    - SNLI single-model 88.7; 5-model ensemble 89.3, surpassing a prior ensemble (Table 8).
    - SRL 84.6 single-model beats prior single and ensemble results (Table 10).
    - Coref 70.4 single-model beats prior single and ensemble results (Table 11).
    - NER 92.22 surpasses prior SOTA (Table 12).
    - SST-5 54.7 improves over CoVe-augmented BCN (Table 13).

- Ablations and diagnostics
  - Layer mixing vs. last layer only (Sec. 5.1; Table 2)
    - Using all layers outperforms last-layer-only across SQuAD, SNLI, SRL. Allowing learned, non-uniform weights (small `λ`) beats uniform averaging:
      - > “SQuAD dev F1: last-only 84.7 → all layers (λ=1) 85.0 → all layers (λ=0.001) 85.2” (Table 2).
      - > “SNLI dev acc: 89.1 → 89.3 → 89.5” (Table 2).
      - > “SRL dev F1: 84.1 → 84.6 → 84.8” (Table 2).
  - Where to insert ELMo (Sec. 5.2; Table 3)
    - Including ELMo at both input and output helps in NLI and QA; input-only is best for SRL and coreference:
      - > “SQuAD: input-only 85.1; input+output 85.6; output-only 84.8” (Table 3).
      - > “SNLI: 88.9; 89.5; 88.7” (Table 3).
      - > “SRL: 84.7; 84.3; 80.9” (Table 3).
    - Supplemental A.6: in coreference, adding ELMo at the output actually decreased F1 by ~0.7.
  - What different layers learn (Sec. 5.3; Tables 5–6)
    - Word Sense Disambiguation (WSD) using a simple nearest-neighbor classifier over ELMo layer representations:
      - > “biLM second layer F1 69.0 vs. first layer 67.4; competitive with supervised task-specific models and better than CoVe” (Table 5).
    - POS tagging with a linear classifier:
      - > “biLM first layer accuracy 97.3 vs. second layer 96.8; again, better than CoVe layers” (Table 6).
    - Interpretation: lower layers encode syntax (POS), upper layers encode semantics (WSD).
  - Sample efficiency and training speed (Sec. 5.4; Figure 1)
    - > “SRL ELMo model exceeds baseline’s maximum dev F1 by epoch 10 vs. epoch 486 without ELMo (≈98% fewer updates).” 
    - > “With only 1% of SRL training data, ELMo matches the baseline with 10% of the data” (Figure 1).
  - Learned layer weights (Sec. 5.5; Figure 2)
    - > “At input, tasks tend to favor lower layers; output weights are more balanced, with slight preference for lower layers.”
  - Importance of scaling parameter `γ` (Supplement A.2)
    - > “Without `γ`, last-only SNLI performs poorly (below baseline) and SRL training fails.”

- Do the experiments support the claims?
  - Breadth: Improvements are demonstrated on six heterogeneous tasks with diverse architectures and metrics (Sec. 4; Table 1).
  - Depth: Ablations isolate the effect of deep mixing (Table 2), insertion location (Table 3), and probe what is encoded at each layer (Tables 5–6, Table 4).
  - Robustness: Gains hold across strong baselines and even against prior SOTA/ensembles; some task-specific caveats (e.g., output-layer insertion not always helpful) are identified (Sec. 5.2, A.6).
  - Overall, the methodology and diagnostics align well with the central claims of deep, contextual, layer-mixed representations boosting downstream tasks.

## 6. Limitations and Trade-offs
- Dependence on large-scale pretraining
  - The `biLM` is trained on ≈1B-token corpora; achieving similar quality without such data is uncertain (Sec. 3.4). Pretraining remains computationally intensive (4096-unit LSTMs with projections).
- Inference cost and complexity
  - At task time, computing all `biLM` layers and mixing vectors introduces memory and compute overhead compared to static embeddings. Although the `biLM` is frozen, the forward pass is nontrivial.
- Linear mixing assumption
  - ELMo uses a learned linear combination of layers (Eq. (1)), which may not capture all useful cross-layer interactions. While effective, it may be suboptimal compared to more expressive combination schemes.
- Task-specific integration choices
  - Where to inject ELMo (input vs. output) and how to regularize layer weights (`λ`) are task-dependent (Table 3, Sec. 5.1). Misplaced integration can hurt (e.g., coreference output insertion; A.6).
- Domain adaptation variability
  - Unsupervised fine-tuning of the `biLM` on task-domain text sometimes helps (e.g., SNLI perplexity 72.1 → 16.8), but not always (e.g., CoNLL-2012 showed no listed improvement; Table 7). The benefit is uneven across datasets.
- Language and modality scope
  - Experiments are primarily on English text classification/extraction tasks; applicability to other languages, modalities, or generative tasks is not evaluated here.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a practical and general recipe: pretrain a large `biLM`, expose all internal layers, and let downstream tasks learn how to use them. This reframes “embeddings” as context- and layer-aware features rather than static word types, catalyzing a broader move toward pretrain-and-adapt paradigms in NLP (Sec. 1, 3.2, 4).
- Follow-up research enabled or suggested
  - Richer mixing functions: gating or attention over layers rather than scalar weights; layer-wise adaptation beyond linear sums (building on Eq. (1)).
  - Deeper or alternative architectures for the base model: more layers, different recurrent cells, or other sequence encoders; probing whether the syntax/semantics layering pattern generalizes (Sec. 5.3).
  - Systematic domain adaptation: when and how to fine-tune the `biLM` on unlabeled in-domain data (Table 7), and how that interacts with task-specific mixing.
  - Low-resource settings: exploiting the demonstrated sample efficiency (Figure 1) for few-shot/semi-supervised learning and new languages.
  - Analysis tools: further intrinsic probes beyond WSD and POS to map what is encoded at each layer (syntax trees, coreference cues, discourse relations).
- Practical applications
  - Any pipeline that benefits from robust token-level meaning: question answering, information extraction (NER, SRL), textual entailment, and coreference show immediate gains (Table 1).
  - Industrial NLP: drop-in ELMo features can boost existing systems without architectural overhauls; improved data efficiency can reduce labeling cost.
  - Domain-specific NLP: biomedical, legal, or customer-support text can benefit from unlabeled in-domain fine-tuning of the `biLM` plus task-level ELMo integration (Supplement A.1).

> Representative headline result (Table 1): “Simply adding ELMo establishes a new state-of-the-art result in every considered task,” including “up to 20% relative error reductions,” with SQuAD F1 improving “4.7 points (81.1 → 85.8).”

> Core mechanism (Sec. 3.2, Eq. (1)): `ELMo^task_k = γ^task * sum_j s^task_j * h^LM_{k,j}`, a learned, task-specific mixture over all `biLM` layers that exposes both syntactic (lower layers) and semantic (higher layers) signals (Sec. 5.3; Tables 5–6; Figure 2).
