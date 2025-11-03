# Voxtral: Open‑Weight Multimodal Audio Chat Models with Whisper‑Based Encoding and Long‑Context Reasoning

**ArXiv:** [](https://arxiv.org/abs/)
**Authors:** 
**Institutions:** 

## 🎯 Pitch

Voxtral presents groundbreaking open-weight multimodal audio chat models, adept at handling up to 40 minutes of audio while mastering transcription, translation, and cross-modal reasoning. By combining Whisper-based encoding, strategic downsampling, and innovative dual-pattern pretraining, Voxtral delivers state-of-the-art speech comprehension, fostering advancements in privacy-preserving and domain-specific audio applications.

---

## 1. Executive Summary
Voxtral introduces two open‑weights multimodal audio chat models—`Voxtral Mini` (≈4.7B parameters) and `Voxtral Small` (≈24.3B)—that understand speech and text and answer in text. The models achieve state‑of‑the‑art transcription and strong translation and speech‑understanding accuracy while preserving text‑only performance, and they can process up to 40 minutes of audio within a 32K‑token context (Sections 1–2; Table 1; Figure 3–6).

## 2. Context and Motivation
- Problem addressed
  - Building a single model that can both transcribe speech accurately and reason over long audio contexts (e.g., answer questions, summarize, translate), with open weights and strong text capabilities.
  - Evaluation gaps: widely used audio benchmarks emphasize transcription/translation; there is less coverage of QA and long‑context reasoning across speech (Section 1, 3.4).

- Why it matters
  - Real‑world use: customer calls, lectures, meetings, podcasts—often long, multilingual, and requiring comprehension rather than just transcription.
  - Practical deployment: open weights under Apache 2.0 enable on‑premise and edge use, privacy‑preserving setups, and domain adaptation (Abstract; Conclusion).

- Prior approaches and limitations
  - ASR models like Whisper excel at transcription but not general instruction following or reasoning.
  - Some multimodal chat models are closed or optimized for short audio and narrow tasks; evaluation suites are not standardized for speech understanding (Section 1, 3.4).
  - Synthetic “speech versions” of text tasks exist, but coverage and standardization are limited (Section 3.4).

- Positioning
  - Voxtral pairs a Whisper‑based audio encoder with Mistral LLM backbones and introduces a training scheme that balances transcription alignment with cross‑modal reasoning (Sections 2–3).
  - It also contributes new evaluation resources: speech‑synthesized versions of GSM8K, TriviaQA, MMLU; and an internal Speech Understanding (SU) benchmark for long‑context QA with an LLM judge (Section 3.4; Appendix A.2–A.4).

## 3. Technical Approach
Voxtral is a Transformer‑based system comprising an audio encoder, an audio‑language adapter, and a language decoder (Section 2; Figure 1).

- Audio processing and encoder (Section 2.1)
  - Input waveform → `log‑Mel spectrogram` with 128 Mel bins (a common time–frequency representation for audio).
  - The encoder is based on `Whisper large‑v3` and outputs embeddings at 50 Hz.
  - Whisper’s receptive field is 30 seconds. For longer inputs, Voxtral:
    - Computes the spectrogram of the full audio.
    - Splits it into independent 30‑second chunks for the encoder.
    - Resets absolute positional encodings for each chunk.
    - Concatenates the resulting embeddings.
  - This is “chunk‑wise attention”: the encoder attends within each 30‑second window; cross‑chunk reasoning is delegated to the language decoder after embeddings are concatenated.
  - Short audios are padded to the next multiple of 30 seconds. An ablation in Section 5.1 shows removing padding slightly hurts ASR (e.g., +0.5% WER on French in Figure 7), so padding is kept.

- Audio‑language adapter (Section 2.2; 5.2)
  - Purpose: reduce the very long audio sequence length before feeding the decoder.
  - Implementation: an MLP downsampling layer applied to the encoder outputs.
  - Downsampling factor 4× (50 Hz → 12.5 Hz) is selected as the best trade‑off:
    - Minimal ASR degradation; improved speech understanding on Llama QA vs no downsampling (Figure 8).
    - Enables 40‑minute audios to fit within 32K tokens.

- Language decoders (Section 2.3; Table 1)
  - `Voxtral Mini` uses `Ministral 3B` backbone (edge‑friendly, 3.6B decoder params; total ≈4.7B).
  - `Voxtral Small` uses `Mistral Small 3.1 (24B)` backbone (22.9B decoder params; total ≈24.3B).
  - Text embeddings are learned; audio and text tokens are consumed jointly during decoding.

- Three‑phase training (Section 3)
  1) Pretraining (Section 3.1)
     - Goal: teach the decoder to align audio with text and to continue discourse across modalities.
     - Two data patterns built from segmented audio–transcript pairs `(A_n, T_n)`:
       - `Audio‑to‑text repetition`: input `A_n` and target `T_n` (explicit ASR alignment).
       - `Cross‑modal continuation`: interleave segments as `(A1, T2, A3, T4, …)` so each audio segment is followed by the next text segment `T_{n+1}` (forces modality‑invariant continuation, like dialog/QA).
     - The model is told which pattern to follow using special tokens `<repeat>` and `<next>`.
     - “Warm‑up”: initially freeze encoder and decoder; train only the adapter—found beneficial for speech understanding (Section 3.1).
     - A variant trained only with repetition is released as `Voxtral Mini Transcribe` (ASR‑focused).
  2) Supervised finetuning (SFT, Section 3.2)
     - Objective: keep or slightly improve ASR while teaching instruction following and speech understanding.
     - Data creation:
       - `Audio context + text query`: Use long audios (up to ~40 min) with transcripts; prompt a text LLM to generate diverse, speech‑grounded Q/A, including retrieval (“needle‑in‑haystack”) and reasoning; also create summarization and translation tasks.
       - `Audio‑only input`: Convert text SFT datasets (incl. function calling) to speech via TTS; to avoid overfitting to TTS, mine genuine questions from long‑form ASR corpora and pair them with LLM‑generated text answers.
       - A special `transcribe mode` token removes the need for a text prompt for pure ASR.
  3) Preference alignment (Section 3.3)
     - Use `Direct Preference Optimization (DPO)`—a method that directly optimizes the policy to prefer better responses using pairwise comparisons—plus an `Online DPO` variant.
     - For each example, sample two candidate responses (temperature 0.5), replace the audio with its transcript, and score with a text‑only reward model. This captures semantics and style even without raw audio and is simpler to deploy at scale.
     - Online DPO improved response quality most (Section 5.4; Table 2).

- Evaluation infrastructure and new benchmarks (Section 3.4; Appendix A.2–A.4)
  - `Speech‑synthesized benchmarks`: Turn text tasks (GSM8K, TriviaQA, MMLU) into speech by rewriting non‑speakable parts (Appendix A.3 prompt) and synthesizing with diverse TTS voices; model outputs remain text, so standard scoring applies.
  - `Speech Understanding (SU) benchmark`: In‑the‑wild audios up to 19 minutes; LLM judge grades candidate answers using the transcript as context with two metrics:
    - `LLM_JUDGE_SCORE`: 0/1 helpfulness.
    - `GRADE_LLM_JUDGE_SCORE`: 0–5 quality. Prompts and scoring rubric provided (Appendix A.4).

Definitions of less common terms used above:
- `Chunk‑wise attention`: restrict self‑attention to fixed windows (here, 30 s) during encoding; cross‑window integration happens later.
- `WER (Word Error Rate)`: lower is better; measures transcription errors normalized by reference length.
- `BLEU`: a translation quality metric; higher is better.
- `DPO/Online DPO`: preference‑based alignment methods that optimize the model to rank better responses higher, with the “online” variant collecting fresh samples from the current policy during training.

## 4. Key Insights and Innovations
- Balanced dual‑pattern pretraining is crucial (Section 5.3; Figure 9)
  - Novelty: explicitly mixing `audio‑to‑text repetition` (ASR) and `cross‑modal continuation` (reasoning) and signaling them with tokens `<repeat>/<next>`.
  - Evidence:
    - Training only on repetition yields strong ASR but “nearly zero” Llama‑QA performance.
    - Training only on continuation yields good Llama‑QA but ≈60% WER (non‑functional ASR).
    - A 50/50 mix keeps both strong—this is not a minor tweak; it is the core mechanism that makes a single model good at both transcription and understanding.

- Adapter downsampling sweet spot at 12.5 Hz (Section 5.2; Figure 8)
  - Novelty: an MLP adapter reduces audio token rate by 4× without losing accuracy and even improves QA.
  - Significance: makes 40‑min audio feasible in 32K context, and improves understanding (12.5 Hz beats 50 Hz on Llama‑QA by +1.5% absolute, Figure 8 right) with little ASR degradation.

- Practical chunking strategy for long audio with Whisper’s 30s limit (Section 2.1)
  - Design: reset positional encodings per chunk and concatenate embeddings; functionally equivalent to chunk‑wise attention and efficient for long inputs.
  - Importance: avoids retraining a long‑context encoder and offloads discourse integration to the decoder.

- Preference alignment with transcript‑only reward improves helpfulness (Section 5.4; Table 2)
  - With `Online DPO`, Voxtral Small raises SU `LLM_JUDGE_SCORE` from 86.61% to 88.31% and `GRADE` from 4.16 to 4.38, though with a small regression on English short‑form WER.
  - This shows that transcript‑based rewards suffice to enhance dialog quality even for audio tasks.

- New evaluation resources for speech understanding (Section 3.4; Appendix A.2–A.4)
  - Synthesized GSM8K/TriviaQA/MMLU and an SU benchmark with long audios and LLM judging, filling a gap in standardized evaluation of speech comprehension and reasoning.

## 5. Experimental Analysis
- Setup: datasets, metrics, baselines (Sections 3.4–4; Appendix A)
  - ASR: English short‑form (LibriSpeech, GigaSpeech, VoxPopuli, Switchboard, CHiME‑4, SPGISpeech), English long‑form (Earnings‑21/22, segmented to 10‑minute for provider limits), multilingual sets (FLEURS, Common Voice 15.1, MLS); metric = WER (lower is better). Full task breakdown in Table 3 (English) and Tables 4–6 (multilingual).
  - Speech translation: FLEURS speech translation, metric = BLEU (higher is better); results in Figure 4 and Table 7.
  - Speech understanding: Llama‑QA, OpenBook‑QA, plus synthesized MMLU/GSM8K/TriviaQA subsets; internal SU benchmark with LLM judge. Results in Figure 5 and Table 8; SU judge scores in Table 2.
  - Text‑only: five standard text benchmarks (Figure 6; exact tasks not enumerated in the excerpt but compared to Mistral Small 3.1).
  - Baselines: Whisper large‑v3, ElevenLabs Scribe, GPT‑4o mini (Audio/Transcribe), Gemini 2.5 Flash.

- Headline results
  - ASR: strong to state‑of‑the‑art in short‑form and MCV
    - Figure 3 summary:
      > “Voxtral Small outperforms all open and closed‑source models on English Short‑Form and MCV. Voxtral Mini Transcribe beats GPT‑4o mini Transcribe and Gemini 2.5 Flash in every task.”
    - Concrete numbers (Table 3, short‑form examples):
      - LibriSpeech Test‑Clean WER: `Voxtral Small 1.53%` vs `Whisper large‑v3 1.84%`, `Scribe 1.80%`, `GPT‑4o mini Transcribe 1.92%`, `Gemini 2.5 Flash 2.97%`.
      - LibriSpeech Test‑Other: `3.14%` vs `3.66%` (Whisper), `3.44%` (Scribe), `4.70%` (GPT‑4o mini), `6.15%` (Gemini).
      - SPGISpeech: `1.89%` vs `3.15%` (Whisper), `3.16%` (Scribe), `4.51%` (GPT‑4o mini).
    - Long‑form earnings calls (Table 3, 10‑min segments):
      - E21 10m WER: `Voxtral Small 9.55%` vs `Scribe 7.39%`, `Gemini 8.09%`, `Whisper 9.88%`.
      - E22 10m WER: `12.48%` vs `Scribe 9.16%`, `Gemini 10.80%`, `Whisper 13.07%`.
      - Takeaway: short‑form and MCV are clear wins; long‑form ASR remains competitive but not SOTA.

  - Speech translation (Figure 4; Table 7)
    - Consistent SOTA among tested pairs; e.g., Table 7:
      - `fr→en`: `Voxtral Small 54.2 BLEU` vs `GPT‑4o mini Audio 48.2`, `Gemini 42.0`.
      - `en→fr`: `57.3` vs `52.7` (GPT‑4o), `53.9` (Gemini).
      - `de→en`: `56.6` vs `51.8` (GPT‑4o), `39.4` (Gemini).

  - Speech understanding (Figure 5; Table 8)
    - Voxtral Small is competitive with closed models and surpasses GPT‑4o mini Audio on 3/7 tasks (OpenBook‑QA, MMLU*, AU Bench).
      - Table 8 examples:
        - OpenBook‑QA: `Voxtral Small 88.4%` vs `GPT‑4o mini Audio 83.7%` (Gemini 94.7%).
        - MMLU* (speech‑synth): `74.3%` vs `72.6%` (GPT‑4o).
        - AU Bench (internal SU): `86.6%` vs `80.0%` (GPT‑4o), `88.6%` (Gemini).
      - On other synthesized tasks, Voxtral Small trails GPT‑4o mini slightly (e.g., TriviaQA* 79.4% vs 83.7%; GSM8K* 89.7% vs 90.8%).

  - Text‑only performance (Figure 6)
    - > “Voxtral Small performs comparably to Mistral Small 3.1, highlighting its strong text capabilities.”
    - Implication: the audio additions and pretraining do not degrade text‑only skills.

- Alignment gains vs. ASR trade‑off (Section 5.4; Table 2)
  - `Voxtral Mini`:
    - SU `LLM_JUDGE_SCORE`: 83.47% → 85.59% with Online DPO; Grade: 3.92 → 4.08; essentially no WER change (~6.78–6.79).
  - `Voxtral Small`:
    - SU `LLM_JUDGE_SCORE`: 86.61% → 88.31% (Online DPO); Grade: 4.16 → 4.38.
    - English short‑form WER slightly regresses: 6.31 → 6.50 macro average. Hence the released default is the SFT model; an Online‑DPO Small is planned (Section 5.4).

- Ablations and design validations
  - Padding study (Section 5.1; Figure 7): removing 30‑second padding hardly affects FLEURS English ASR but degrades French by ~0.5% WER; Llama‑QA is similar. Padding retained to maximize ASR.
  - Downsampling study (Section 5.2; Figure 8): 12.5 Hz (4× downsampling) offers the best ASR‑understanding trade‑off; 6.25 Hz harms ASR by >1% WER on FLEURS French.
  - Pretraining pattern ratio (Section 5.3; Figure 9): confirms the necessity of mixing repetition and continuation; either alone fails on the complementary task.

- Do the experiments support the claims?
  - Yes for short‑form ASR and translation (clear numerical wins), and broadly for speech understanding where results are competitive with carefully chosen baselines and mixed across tasks (Table 8, Figure 5).
  - The ablations make the design choices traceable to measured trade‑offs, not ad‑hoc.

## 6. Limitations and Trade-offs
- Modeling and data assumptions
  - The encoder only sees 30‑second chunks; fine cross‑chunk acoustic effects must be integrated downstream by the decoder. This could miss prosodic dependencies spanning longer than 30 seconds (Section 2.1).
  - Short inputs are padded to 30 seconds to preserve ASR quality (Section 5.1), increasing compute/time for very short utterances.
  - Synthetic data is heavily used for SFT: QA pairs, summarization, translations, and TTS prompts (Section 3.2). While mitigated by including genuine human questions, the distribution mismatch with real spontaneous speech may persist.

- Preference alignment signal is transcript‑only (Section 3.3)
  - The reward model never “hears” the audio; it cannot evaluate audio‑specific qualities (emotion, speaker identity, tone) and relies on ASR transcriptions that may contain errors.

- Evaluation caveats
  - Long‑form ASR results (Earnings‑21/22) are not SOTA; the inputs were segmented to 10 minutes to fit closed providers’ constraints (Appendix A.1), which changes the task slightly.
  - The SU benchmark uses an LLM judge. Although judged multiple times (10), LLM‑as‑judge can encode biases and may prefer certain linguistic styles.

- Computational footprint and latency
  - `Voxtral Small` totals ≈24.3B parameters (Table 1), plus a 640M‑parameter encoder; inference on 40‑minute audio even with 4× downsampling remains resource‑intensive.
  - The 32K context still limits maximum audio length; beyond ~40 minutes, inputs must be truncated or summarized.

- Coverage and robustness
  - Some languages remain hard (e.g., Common Voice Arabic WERs are high across models; Table 5). Robustness to heavy accents, background noise, overlapping speech, and code‑switching is not deeply analyzed.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides strong, open, end‑to‑end speech understanding models with competitive performance to popular closed systems (Figures 3–5), plus a 32K context for long audio. This lowers the barrier to build privacy‑preserving, on‑device, or domain‑specialized audio assistants.

- What it enables next
  - Domain adaptation: fine‑tune Voxtral on call‑center, medical, or legal audio without vendor lock‑in.
  - Research on speech understanding evaluation: the released speech‑synthesised suites and the SU judging framework can become standardized tests for long‑context reasoning over audio (Section 3.4; Appendix A.2–A.4).
  - Alignment with audio‑aware rewards: extend Online DPO with reward models that ingest audio features or confidence scores rather than transcripts only.
  - Streaming and low‑latency variants: replace fixed 30‑s chunking with true streaming encoders; explore learned downsampling or adaptive compression instead of fixed 4×.
  - Multimodal function calling: the models natively support function calling with audio (Section 1, “Primary contributions”), suggesting agentic pipelines that trigger tools directly from spoken requests.

- Practical applications
  - Meeting assistants that can ingest a full 40‑minute meeting and answer queries or summarize across speakers.
  - Customer‑support analytics: multilingual transcription, translation, and issue extraction with tool calls.
  - Education and accessibility: lecture Q&A, multi‑language translation for hearing‑impaired users.

Block‑quoted highlights supporting key claims:
- Contributions (Section 1):
  > “Two open‑weights audio models with state‑of‑the‑art transcription and multilingual speech understanding for audio durations up to their 32K context window … Native function calling support with audio … Evaluation benchmarks that measure speech understanding and reasoning.”

- Long‑audio handling (Section 2.1):
  > “The audio encoder processes the speech input, attending to 30‑second chunks of audio independently … embeddings computed from each chunk are concatenated … functionally equivalent to chunk‑wise attention.”

- Downsampling choice (Section 5.2; Figure 8):
  > “12.5 Hz surpasses the 50 Hz baseline [on Llama‑QA] … Based on the trade‑off … we select 12.5 Hz as the optimal frame‑rate.”

- Alignment quality gains (Section 5.4; Table 2):
  > “For both Mini and Small, DPO and Online DPO improve response quality … the online variant was more effective.”

In sum, Voxtral’s main technical advance is the integrated recipe—chunked Whisper encoding, a 4× downsampling adapter, balanced dual‑pattern pretraining, and Online DPO—that jointly delivers strong ASR and long‑context speech reasoning in open models, along with useful evaluation artifacts to measure those capabilities.
