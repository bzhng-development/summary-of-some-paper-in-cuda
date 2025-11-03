# BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension

**ArXiv:** [1910.13461](https://arxiv.org/abs/1910.13461)

## 🎯 Pitch

BART introduces a powerful and flexible pre-training paradigm by corrupting text with various noise functions and training a standard sequence-to-sequence Transformer to reconstruct the original input. By combining a bidirectional (BERT-like) encoder with an autoregressive (GPT-like) decoder, BART unifies the strengths of both understanding and generation models, achieving state-of-the-art results on a range of text generation tasks—such as summarization and dialogue—while matching top baselines on comprehension benchmarks. This versatility significantly advances the potential for a single pre-trained model to excel across the full spectrum of NLP applications.

---

## 1. Executive Summary (2-3 sentences)
BART introduces a simple but powerful pre‑training recipe: corrupt text with flexible noise and train a standard sequence‑to‑sequence Transformer to denoise it. With a bidirectional encoder (BERT‑like) and an autoregressive decoder (GPT‑like), BART unifies pre‑training for both understanding and generation, achieving state‑of‑the‑art results on several generation tasks (notably summarization) while matching strong baselines on discriminative benchmarks (GLUE, SQuAD).

## 2. Context and Motivation
- Problem/gap:
  - Pre‑training methods before this work were specialized: encoder‑only masked language models (e.g., BERT) excel at understanding but do not directly support generation; decoder‑only language models (e.g., GPT) are good at generation but lack bidirectional context crucial for many comprehension tasks.
  - Existing denoising schemes (e.g., word masking) are narrow and do not exploit richer, document‑level corruption patterns that might better teach models to manipulate and reconstruct text.

- Why it matters:
  - A single pre‑trained model that works across comprehension and generation reduces task‑specific engineering and improves sample efficiency on a wide spectrum of NLP tasks (summarization, dialogue, QA, translation).
  - In practice, stronger generative pre‑training translates into substantially better abstractive summarization and dialogue quality; theoretically, it clarifies what kinds of noise/objectives best transfer to different downstream tasks.

- Prior approaches and shortcomings:
  - BERT (masked language modeling): predicts randomly masked tokens independently; not autoregressive, so there’s a mismatch for generation. See Figure 1a.
  - GPT (left‑to‑right language modeling): can generate, but cannot integrate right context at training time; weaker for tasks needing bidirectional reasoning. See Figure 1b.
  - XLNet (permuted LM): autoregressive over permutations; strong on understanding, but requires architectural additions (relative positions, segment recurrence) and is not a standard seq2seq.
  - MASS/UniLM: seq2seq‑style or attention‑mask variants that partially address generation but are less general or still restrict how encoder/decoder view inputs.
  
- Positioning of this paper:
  - BART is a general seq2seq denoising autoencoder with flexible noise on the encoder input and left‑to‑right generation on the decoder output (Figure 1c). It can emulate BERT‑like behavior (bidirectional encoder), GPT‑like behavior (autoregressive decoder), and several recent objectives (Section 4), offering a unified framework.

## 3. Technical Approach
BART is a denoising autoencoder implemented as a sequence‑to‑sequence Transformer. “Denoising autoencoder” here means: corrupt a clean input (with noise such as masking or shuffling) and train the model to reconstruct the original text. This forces the model to learn both global structure and local content.

- Architecture (Section 2.1; Figure 1c):
  - Standard Transformer encoder‑decoder with cross‑attention (as in neural machine translation).
  - Encoder: bidirectional self‑attention over the corrupted input.
  - Decoder: left‑to‑right autoregressive generation of the original text.
  - Minor choices: `GeLU` activations, parameter initialization N(0, 0.02); base model uses 6 encoder and 6 decoder layers; large model uses 12+12. Compared to BERT, BART adds cross‑attention and omits BERT’s extra output feed‑forward layer; overall ~10% more parameters than a same‑sized BERT.

- Pre‑training objective (Section 2.2):
  - Optimize the negative log‑likelihood of the original (clean) document given its corrupted version.
  - Unlike many prior denoising autoencoders tailored to one corruption scheme, BART supports arbitrary document‑level noise—even length changes—because the encoder input need not align token‑by‑token with the decoder target.

- Noise (“corruption”) functions (Section 2.2; Figure 2):
  - `Token masking`: randomly replace tokens with a `[MASK]` token (BERT‑style).
  - `Token deletion`: randomly delete tokens; the model must infer how many/which tokens are missing.
  - `Text infilling` (key novelty): sample spans with lengths from a Poisson distribution (λ=3); replace each span—regardless of its original length—with a single `[MASK]`. Zero‑length spans correspond to insertions. This forces the model to infer both content and span lengths.
  - `Sentence permutation`: split by sentence (using full stops) and shuffle sentence order; encourages reasoning over document structure.
  - `Document rotation`: pick a random starting token and rotate the document; teaches the model to detect document starts.
  - Noise functions can be composed (e.g., text infilling + sentence permutation).

- Fine‑tuning strategies (Section 3; Figure 3):
  - `Sequence classification` (e.g., MNLI): feed the same uncorrupted input to encoder and decoder; append an EOS token; use the final decoder state to classify (Figure 3a). This mirrors BERT’s `[CLS]` usage but leverages decoder context over the entire input.
  - `Token classification` (e.g., SQuAD span prediction): feed the document to both encoder and decoder; use top decoder states as token representations for start/end classifiers.
  - `Sequence generation` (e.g., summarization, abstractive QA): standard seq2seq fine‑tuning; encoder takes the source text; decoder generates targets autoregressively. Training uses label‑smoothed cross entropy (smoothing=0.1), and decoding uses beam search (beam=5) with trigram blocking and tuned length penalties (Section 5.3).
  - `Machine translation` into English (Section 3.4; Figure 3b):
    - Use the entire pre‑trained BART (encoder+decoder) as a target‑side denoiser by placing a small, randomly initialized source encoder in front. This new encoder maps foreign words into a representation that BART can “de‑noise” into English.
    - Two‑step training: (1) freeze most of BART; train only the new source encoder, BART positional embeddings, and the input projection of BART’s first encoder layer; (2) jointly fine‑tune all parameters briefly.

- Large‑scale pre‑training setup (Section 5.1):
  - `BART‑Large`: 12 encoder + 12 decoder layers, hidden size 1024; batch size 8000; 500K steps; GPT‑2 BPE tokenizer.
  - Noise choice: compose `text infilling` (mask 30% tokens) with `sentence permutation` on all sentences.
  - Training data: same 160GB corpus as RoBERTa (news, books, stories, web text).
  - Dropout disabled for the final 10% of steps to help fit the data.

- Analogy for intuition:
  - Think of BART as learning to “unscramble and fill in” documents: shuffle sentences, punch out variable‑length spans with a single hole marker, or lop off tokens. The encoder reads this messy document; the decoder learns to write the original clean document, step by step.

## 4. Key Insights and Innovations
- Flexible denoising pre‑training on a seq2seq model (fundamental):
  - What’s new: apply arbitrary, document‑level noise to inputs and train a standard encoder‑decoder to reconstruct (Figure 1c; Section 2).
  - Why it matters: unifies bidirectional encoding (strong for understanding) with autoregressive decoding (necessary for generation) in one pre‑trained model. It reduces the objective mismatch seen in encoder‑only models when used for generation.

- `Text infilling` spans replaced by a single mask (novel corruption scheme):
  - What’s new: spans sampled from Poisson(λ=3) are replaced with one `[MASK]`, including 0‑length spans (insertions) (Section 2.2; Figure 2).
  - Why it matters: the model must infer both content and span length, strengthening its ability to perform broader edits and structural reasoning. In ablations, infilling consistently improves generation perplexity and maintains strong performance on understanding (Table 1).

- Fair, within‑framework ablations of pre‑training objectives (methodological contribution):
  - What’s new: re‑implement several objectives—language modeling (GPT‑style), permuted LM (XLNet‑style), masked LM (BERT‑style), UniLM‑style multitask masking, MASS‑style masked seq2seq—under a controlled setup (same data/code/optimization) to isolate the effect of the objective (Section 4.1).
  - Why it matters: clarifies which design choices drive gains. For example, left‑to‑right pre‑training improves generation; bidirectional encoders are crucial for SQuAD (Table 1; Section 4.3).

- Using a pre‑trained seq2seq (BART) as a decoder for MT with a learned source front‑end (conceptual/practical):
  - What’s new: add a small, trainable source encoder to feed BART, effectively using BART as a powerful target‑side language model and denoiser (Section 3.4; Figure 3b).
  - Why it matters: improves WMT16 RO‑EN by +1.1 BLEU over a strong back‑translation baseline without needing bilingual pre‑training (Table 6).

## 5. Experimental Analysis
- Evaluation methodology:
  - Ablations with base‑size models (6+6 layers, hidden 768) trained 1M steps on Books+Wikipedia (Section 4). Tasks span:
    - `SQuAD 1.1` (F1) and `MNLI` (accuracy) for understanding.
    - `ELI5`, `XSum`, `ConvAI2`, `CNN/DM` measured by perplexity (lower is better) to assess generation modeling (Table 1).
  - Large‑scale BART‑Large trained at RoBERTa scale (Section 5.1), evaluated on:
    - `GLUE` and `SQuAD 1.1/2.0` (Table 2).
    - `CNN/DailyMail` and `XSum` summarization (ROUGE‑1/2/L) (Table 3).
    - `ConvAI2` dialogue (Valid F1 and perplexity) (Table 4).
    - `ELI5` abstractive QA (ROUGE) (Table 5).
    - `WMT16 RO‑EN` MT (BLEU) with back‑translation data (Table 6).
  - Generation fine‑tuning uses beam=5, trigram blocking, label smoothing 0.1, tuned length penalties (Section 5.3).

- Main quantitative findings:
  - Ablations (Table 1):
    - BART with `text infilling` attains strong, balanced performance: SQuAD F1 90.8, MNLI 84.0, and best/near‑best generation perplexities (e.g., XSum 6.61, CNN/DM 5.83). Adding `sentence shuffling` further improves CNN/DM perplexity to 5.41.
    - Pure `document rotation` and `sentence shuffling` alone perform poorly (e.g., rotation SQuAD F1 77.2; CNN/DM PPL 10.59).
    - `Token deletion` often beats `token masking` on generation (e.g., CNN/DM PPL 5.87 vs 6.10), suggesting models benefit from inferring missing positions.
    - “Pure language model” is best on ELI5 perplexity (21.40 vs BART‑infilling 24.26), indicating outputs weakly tied to inputs favor LM‑style pre‑training (Section 4.3).
  - Discriminative benchmarks (Table 2):
    - BART matches RoBERTa/XLNet overall (e.g., SQuAD 1.1 F1 94.6; GLUE tasks within small margins), showing that adding an autoregressive decoder does not harm classification performance.
  - Summarization (Table 3):
    - On `CNN/DM`: BART 44.16/21.28/40.90 ROUGE surpasses prior best (BERTSUMEXTABS 42.13/19.60/39.18).
    - On `XSum` (highly abstractive): BART 45.14/22.27/37.25 improves by ~6 ROUGE points over the best prior (≈38.8/16.5/31.3), a substantial advance.
  - Dialogue (Table 4):
    - On `ConvAI2`: BART achieves Valid F1 20.72 and perplexity 11.85, outperforming the “Best System” baseline (F1 19.09, PPL 17.51).
  - Abstractive QA (Table 5):
    - On `ELI5`: BART leads with ROUGE‑L 24.3 vs 23.1 for the best prior, despite the earlier perplexity trend favoring pure LMs in ablations.
  - Machine translation (Table 6):
    - `WMT16 RO‑EN` with back‑translation: `Tuned BART` reaches 37.96 BLEU, beating a strong Transformer‑large baseline (36.80) by +1.16 BLEU; “Fixed BART” (mostly frozen) performs slightly below baseline (36.29), underscoring the importance of the second fine‑tuning step.

- Qualitative analysis (Section 6; Table 7):
  - Outputs are fluent and highly abstractive, integrating evidence across the article. The examples use WikiNews published after pre‑training to avoid train–test contamination and remove the first sentence to prevent trivial extraction. Notably, BART sometimes hallucinates unsupported details (e.g., incorrectly claiming a study appeared in “Science”).

- Do results support the claims?
  - Yes, with nuance:
    - The ablation suite (Table 1) convincingly shows the benefit of span‑based `text infilling` and the necessity of left‑to‑right pre‑training for generation, while also confirming the value of bidirectional encoding for SQuAD.
    - Large‑scale experiments (Tables 2–3) validate that BART improves generation substantially without sacrificing discriminative performance.
    - MT results (Table 6) demonstrate a practical path to using pre‑trained seq2seq as a decoder, though gains rely on careful fine‑tuning and back‑translation data.

- Robustness and caveats:
  - The permuted LM replication underperforms XLNet because architectural extras (relative positions, segment recurrence) are intentionally omitted to isolate objective effects (Section 4.1).
  - The discrepancy between ELI5 perplexity (LM best; Table 1) and fine‑tuned ROUGE (BART best; Table 5) shows that objective‑level modeling ability does not always predict end‑task generation quality.

## 6. Limitations and Trade-offs
- Dependence on large‑scale resources:
  - Matching top discriminative performance requires RoBERTa‑scale pre‑training: 500K steps, batch size 8000, and 160GB of data (Section 5.1), which is compute‑ and data‑intensive.

- Noise design is heuristic and task‑dependent:
  - Sentence splitting by full stops may be brittle across domains/languages; `sentence permutation` helps CNN/DM more than others (Section 5.1).
  - Some noise types (document rotation) are ineffective when used alone (Table 1), highlighting the need for careful selection/composition.

- Decoder overhead on discriminative tasks:
  - For classification, BART uses both encoder and decoder (Figure 3a), increasing inference cost relative to encoder‑only models while achieving similar accuracy to RoBERTa (Table 2).

- Hallucination risk in generation:
  - Qualitative samples show occasional unsupported facts (Section 6; Table 7), a typical issue in abstractive generation that remains unaddressed by the objective alone.

- Translation scope:
  - The MT approach improves target‑English translation by adding a trained source encoder, but relies on back‑translation data and bitext; it does not demonstrate bilingual pre‑training or fully unsupervised MT (Section 5.4).

- Objective comparability caveat:
  - Although many objectives are compared under a unified implementation, some (e.g., XLNet) benefit from architectural enhancements not included here (Section 4.1), so absolute rankings across families should be interpreted with care.

## 7. Implications and Future Directions
- Field impact:
  - BART validates seq2seq denoising pre‑training as a unifying recipe that delivers top‑tier generation while preserving strong understanding performance. It reframes pre‑training as “learn to edit/restore documents,” which better aligns with many text‑to‑text tasks.

- Follow‑up research enabled:
  - Explore richer, task‑targeted corruption processes (e.g., discourse‑aware shuffles, entity‑level masking, syntax‑guided deletions) and learnable/noise‑adaptive schedules.
  - Develop cross‑lingual or multilingual BART variants, possibly sharing a subword vocabulary across languages and investigating source‑ and target‑side pre‑training for MT.
  - Combine BART with factuality constraints (retrieval augmentation, knowledge graphs) to reduce hallucinations in abstractive tasks.
  - Study calibration and controllability in generation (length control beyond penalties, semantic coverage constraints, style/format conditioning).

- Practical applications:
  - Abstractive summarization for news, scientific articles, and enterprise documents (Table 3 gains are especially strong on XSum).
  - Conversational agents and long‑form QA with improved fluency and coherence (Tables 4–5).
  - Target‑side strengthened machine translation into English using monolingual pre‑training (Table 6), potentially valuable for low‑resource source languages when English monolingual data is plentiful.

> Bottom line: BART shows that training a seq2seq model to robustly “undo” flexible, document‑level noise—especially via span‑based `text infilling`—is a broadly effective foundation for both understanding and generation. Figures 1–3 detail how it generalizes BERT/GPT and how to use it; Tables 1–6 demonstrate consistent empirical benefits, with especially large gains on abstractive generation.
