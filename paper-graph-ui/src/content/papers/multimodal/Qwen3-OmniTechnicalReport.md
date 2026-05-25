# Qwen3-Omni Technical Report

**ArXiv:** [2509.17765](https://arxiv.org/abs/2509.17765)

## 🎯 Pitch

Qwen3-Omni introduces the first unified, end-to-end multimodal model that achieves state-of-the-art performance across text, image, audio, and video—with no loss relative to single-modality models—via its innovative Thinker–Talker Mixture-of-Experts architecture and a novel AuT audio encoder. By delivering real-time, low-latency speech interaction in up to 119 text and 19 spoken languages, supporting 40-minute audio understanding, and pioneering open-source state-of-the-art results in audio/audio-visual tasks, Qwen3-Omni paves the way for seamless, highly capable AI assistants and sets a new milestone for non-degrading multimodal integration. This breakthrough demonstrates that early, joint multimodal training can consolidate and even enhance core language and vision skills, challenging old trade-offs and enabling more practical, scalable, and interactive AI systems.

---

## 1. Executive Summary
Qwen3-Omni introduces a single end-to-end multimodal model that matches same-sized, single‑modality Qwen models on text and vision while achieving state-of-the-art open-source results on most audio and audio‑visual benchmarks. Its Thinker–Talker Mixture‑of‑Experts architecture, a new `AuT` audio encoder, and a multi‑codebook streaming speech stack enable real-time voice interaction with a theoretical first audio packet in 234 ms (Table 1) and up to 40‑minute audio understanding (Sections 2–5).

## 2. Context and Motivation
- Problem/gap
  - Multimodal LLMs often suffer “modality trade‑offs”: improving one modality degrades others (Introduction). The field lacks a demonstration that a unified model can avoid degradation while also excelling at cross‑modal reasoning and low‑latency speech interaction.
  - There is no widely available general‑purpose audio captioning model for research (Abstract; Section 4.3).

- Importance
  - Practical: Real-time assistants need accurate long-form ASR, low-latency speech synthesis, and strong text/vision reasoning in one system. Latency and concurrency directly affect product usability and cost (Section 2.5; Table 2).
  - Scientific: Shows that early, joint multimodal pretraining can maintain language/vision parity and even improve some visual benchmarks, challenging the assumption that multimodal training inevitably harms core language ability (Section 6; Table 16).

- Prior approaches and shortcomings
  - Pipeline systems (separate ASR → LLM → TTS) incur latency and error compounding.
  - Prior unified models (e.g., Qwen2.5‑Omni) integrated modalities but with more limited audio scale, higher speech synthesis latency, and shorter input limits (Introduction; Section 2.1).

- Positioning
  - Qwen3‑Omni extends the earlier Thinker–Talker design (Qwen2.5‑Omni) with MoE in both modules, a from‑scratch `AuT` encoder trained on 20M hours, multi‑codebook speech generation with an autoregressive + ConvNet pathway, and a training program that mixes unimodal and cross‑modal data early (Sections 2–4). It empirically validates “non‑degrading multimodality” at 30B scale (Section 6; Table 16).

## 3. Technical Approach
Step‑by‑step overview of the system (Figure 2; Section 2):

1) Two-module architecture with Mixture‑of‑Experts (MoE)
- `Thinker` (MoE Transformer, 30B‑A3B; Table 1) handles text generation and multimodal reasoning. MoE means multiple specialized expert sub-networks with a learned router that activates a sparse subset per token, improving throughput for long contexts while controlling compute.
- `Talker` (MoE Transformer, 3B‑A0.3B; Table 1) generates streaming speech tokens conditioned on multimodal features and the conversation state. It no longer consumes `Thinker`’s text embeddings (Section 2.1). Instead, it conditions directly on audio/visual features and a controlled text feed, enabling external interventions (RAG, safety filters) before synthesis and independent style prompts for text vs. audio (Section 2.1).

2) Perception (Section 2.3)
- Text: Qwen tokenizer (151,643 BPE tokens).
- Audio: 16 kHz waveform → 128‑channel mel spectrogram → `AuT` encoder.
- Images/Video: Qwen3‑VL vision encoder initialized from SigLIP2‑So400M (~543M params).
- Temporal alignment: `TM‑RoPE` (Time‑aligned Multimodal Rotary Position Embedding) assigns separate rotary angle budgets to time/height/width (24/20/20) and uses absolute 80 ms temporal IDs for audio and time‑aligned video frames sampled dynamically (Section 2.3). This enables coherent audiovisual fusion, streaming, and long‑sequence extrapolation better than prior M‑RoPE allocations.

3) Audio Transformer (`AuT`) encoder (Section 2.2; Figure 3)
- Trained from scratch on 20M hours: ~80% zh/en pseudo‑labeled ASR, 10% other‑language ASR, 10% audio understanding.
- Conv2D downsampling by 8× before attention yields a 12.5 Hz token rate (one token ≈ 80 ms), enabling efficient real‑time caching. Flash attention uses dynamic window sizes covering 1–8 seconds to balance streaming and offline performance. The `AuT` encoder has ~0.6B parameters (Section 2.2).

4) Streaming speech generation path (Section 2.4 and 2.5; Figure 2)
- Discrete speech tokens: The system uses `RVQ` (residual vector quantization) codebooks (multi‑codebook representation). Each codec frame contains multiple codebook layers that capture different levels of acoustic detail (timbre, prosody).
- Autoregressive top-layer + `MTP` for residuals: `Talker` predicts the zeroth codebook per frame; a lightweight `MTP` (multi‑token prediction) module then predicts remaining residual codebooks for that frame in a fixed-step autoregressive way (Section 2.4–2.5). This concentrates expressivity in discrete codecs while keeping per‑step compute small.
- `Code2Wav` renderer: A causal ConvNet replaces heavier diffusion or DiT vocoders. It can synthesize waveform incrementally from the predicted codebooks, enabling frame‑by‑frame streaming (Section 2.4).

5) Streaming and concurrency optimizations (Section 2.5; Tables 1–2)
- Chunked prefilling: Audio/vision encoders emit temporal chunks. `Thinker` prefills chunk t while `Talker` asynchronously prefills its own chunk with `Thinker`’s last outputs, then `Thinker` starts chunk t+1 (asynchronous pipeline).
- Left‑context‑only codec generation: As soon as `Talker` emits one token, `MTP` fills the frame and `Code2Wav` renders, eliminating the block‑context waiting used in earlier systems (Section 2.5).
- Lightweight modules + MoE: The small `MTP` transformer (~80M params) and ConvNet decoder (~200M) have low FLOPs and batch efficiently; MoE reduces KV‑cache IO for long sequences, improving tokens/sec (Section 2.5; Table 2).

6) Training program (Sections 3–4)
- Pretraining in three stages (Section 3):
  - S1 Encoder alignment: Freeze LLM (initialized from Qwen3). Train audio/vision encoders and adapters against a fixed LLM—but first train adapters, then encoders—to avoid encoders compensating for a frozen LLM in a way that harms perception.
  - S2 General: Unfreeze all and train on ~2 trillion tokens mixed across modalities: text (0.57T), audio (0.77T), image (0.82T), video (0.05T), video‑audio (0.05T).
  - S3 Long context: Raise context to 32,768 tokens; increase long audio/video share.
- Post‑training of `Thinker` in three steps (Section 4.1):
  - Lightweight SFT on ChatML conversations across text/vision/audio/mixed.
  - Strong‑to‑Weak Distillation (off‑policy teacher outputs, then on‑policy KL to teacher logits; teachers include Qwen3‑32B, Qwen3‑235B‑A22B).
  - GSPO reinforcement learning with rule‑based rewards for verifiable tasks (math/coding/IF) and model‑based judges (Qwen3, Qwen2.5‑VL) supplied with references to reduce reward hacking.
- Post‑training of `Talker` in four steps (Section 4.2):
  - Stage‑1: Large mixed multimodal speech data to establish monotonic mapping from context to speech tokens.
  - Stage‑2: Continual pretraining with high‑quality data to reduce hallucinations and strengthen long‑context stability.
  - Stage‑3: Multilingual DPO with preference pairs to improve robustness.
  - Stage‑4: Speaker fine‑tuning for voice control, naturalness, and expressiveness.
- Audio captioner (Section 4.3): Fine‑tune `Qwen3‑Omni‑30B‑A3B` into `…‑Captioner` to generate detailed, low‑hallucination audio captions (Appendix 9.2).

Why these choices?
- Early multimodal data mixing during general pretraining (S2) is presented as the key to “non‑degrading” multimodality (Section 6; Table 16).
- Decoupling `Thinker`’s text from `Talker` lets safety/tools intervene in text before speech, and enables separate system prompts for text and audio styles (Section 2.1).
- Multi‑codebook AR + MTP + ConvNet replaces blockwise diffusion, enabling immediate streaming with low latency and high throughput (Sections 2.4–2.5; Table 1–2).
- `AuT` trained from scratch on 20M hours aims to surpass Whisper-like backbones and to support long-duration, multilingual, general-purpose audio features (Section 2.2).

## 4. Key Insights and Innovations
1) Demonstration of “non‑degrading” multimodal training at 30B scale
- What is new: A controlled comparison shows the 30B multimodal base matches or slightly exceeds same‑size text‑only and vision‑only counterparts on their own modalities while adding audio capabilities (Section 6; Table 16).
- Why it matters: Prior LLM‑centric multimodal models often trade off modalities. Here, early multimodal integration yields parity on language and gains on several vision/OCR benchmarks (e.g., MMMUval 59.33 vs 57.22; InfoVQA 83.31 vs 81.17).

2) Low‑latency, multi‑codebook streaming speech stack
- What is new: A left‑context‑only scheme where `Talker` predicts one token, `MTP` fills remaining codebooks for the frame, and `Code2Wav` streams audio immediately (Section 2.5; Figure 2).
- Why it matters: Theoretical end‑to‑end first‑packet latency is 234 ms for audio (Table 1) and remains streaming with Real‑Time Factor < 1 under varied concurrency (Table 2: RTF 0.47 → 0.66).

3) `AuT` audio encoder trained on 20M hours with 12.5 Hz tokenization
- What is new: A from‑scratch attention encoder‑decoder with Conv2D downsampling, dynamic attention windows (1–8 s), and real‑time prefill caching (Section 2.2).
- Why it matters: Drives strong ASR/S2TT and audio reasoning results across 36 benchmarks; supports up to 40‑minute inputs (Abstract; Section 5.1.2; Table 6–8).

4) Decoupled Thinker–Talker with independent prompts and toolability
- What is new: Talker no longer consumes Thinker’s text embeddings but conditions on multimodal features and controlled text inputs; separate system prompts control textual vs. audio style independently (Section 2.1).
- Why it matters: Improves controllability, safety, and integration with toolchains (RAG, function calling) without sacrificing latency.

5) Release of an audio captioner
- What is new: `Qwen3‑Omni‑30B‑A3B‑Captioner` for detailed, low‑hallucination audio descriptions to fill a gap in general-purpose audio captioning research (Abstract; Section 4.3; Appendix 9.2).
- Why it matters: Enables data generation and evaluation scaffolding for audio‑centric multimodal research.

## 5. Experimental Analysis
Evaluation design
- Modalities and directions:
  - X→Text: text, audio, vision, audio‑visual video to text (Section 5.1).
  - X→Speech: text or cross‑lingual to speech (Section 5.2).
- Datasets/metrics:
  - Text→Text: MMLU‑Redux, GPQA, AIME25, ZebraLogic, MultiPL‑E, IFEval, Creative Writing v3, WritingBench, BFCL‑v3, MultiIF, PolyMath (Tables 4–5).
  - Audio→Text: ASR (Librispeech, Wenetspeech, FLEURS, CommonVoice), S2TT (FLEURS), Music (RUL‑MuchoMusic, GTZAN, MTG‑Jamendo, MagnaTagATune), VoiceBench, MMAU, MMSU (Tables 6–8).
  - Vision→Text: MMStar, HallusionBench, MM‑MT‑Bench, MathVista, MathVision, MMMU/MMMU‑Pro, AI2D, ChartQA, CountBench, Video‑MME, LVBench, MLVU (Tables 9–10).
  - AudioVisual→Text: WorldSense, DailyOmni, VideoHolmes (Tables 11–12).
  - X→Speech: Seed‑TTS test sets, MiniMax multilingual, CosyVoice cross‑lingual (Tables 13–15).
- Latency/concurrency measured with vLLM, torch.compile, CUDA Graph for MTP/vocoder (Table 2).
- Non‑degradation study: controlled pretraining of text‑only, vision‑only, and Omni models with matched size, data, and schedules; only Omni adds audio and audiovisual during pretraining (Section 6; Table 16).

Headline results
- Audio leads
  - ASR/S2TT: On Librispeech test‑clean/test‑other WER, `Qwen3‑Omni‑Instruct` reaches 1.22/2.48, narrowly beating GPT‑4o‑Transcribe (1.39/3.75) and Voxtral‑Small (1.56/3.30) (Table 6).
  - Multilingual ASR/FLEURS avg (19 langs): 5.33 WER for `Omni‑Instruct`, competitive with Gemini‑2.5‑Pro (5.55) (Table 6).
  - VoiceBench “Overall”: `Omni‑Thinking` 89.5, on par with Gemini‑2.5‑Pro 89.6 and ahead of GPT‑4o‑Audio 86.8 (Table 7).
  - Audio reasoning: MMAU 77.5 (`Omni‑Instruct`), surpassing Gemini‑2.5‑Pro 77.4; MMSU edges Gemini‑2.5‑Flash and GPT‑4o‑Audio (Table 7).
  - Music understanding: SOTA on RUL‑MuchoMusic (52.0), GTZAN Acc. 93.0, and strong micro‑F1 across MTG subsets compared with specialists and generalists (Table 8).

- Text and vision parity
  - Text→Text (non‑thinking): `Omni‑Instruct` compares favorably against larger systems on some tasks. Example: AIME25 65.0 (vs Qwen3‑235B‑A22B Non‑Thinking 24.7; vs GPT‑4o‑0327 26.7) and WritingBench 82.6 (Table 4). On other tasks (MMLU‑Redux), it trails GPT‑4o‑0327 (86.6 vs 91.3).
  - Vision→Text: `Omni‑Instruct` matches or exceeds strong baselines on Math/STEM (e.g., MATH‑Vision full 56.3, beating GPT‑4o 30.4 and Gemini‑2.0‑Flash 48.6; Table 9). General VQA is competitive; video understanding mixed (Table 9).

- Audio‑visual reasoning
  - WorldSense: `Omni‑Instruct` 54.0 vs previous open‑source SOTA 47.1 and Gemini‑2.5‑Flash 50.9 (Table 11).
  - DailyOmni and VideoHolmes (Thinking): 75.8/57.3, exceeding Gemini‑2.5‑Flash‑Thinking on VideoHolmes and prior open‑source SOTA on both (Table 12).

- X→Speech generation
  - Zero‑shot TTS (Seed‑TTS): `Omni` achieves 1.07 WER (zh) and 1.39 WER (en). It beats CosyVoice3 on English (1.39 vs 1.45) but not on Chinese (0.71 for CosyVoice3) (Table 13).
  - Multilingual cloning (MiniMax set): Lower is better for “Content Consistency”; higher is better for “Speaker Similarity”. `Omni` excels on Chinese (0.716 WER; SIM 0.772) and English (1.069; 0.773), with competitive performance in other languages (Table 14).
  - Cross‑lingual cloning: `Omni` improves over CosyVoice3 in many any→en and any→ko directions (e.g., zh→en 2.76 vs 2.98; en→ko 4.96 vs 5.87; Table 15). For any→ja, results are comparable despite no kana normalization.

- Latency and throughput
  - First‑packet latency: 234 ms (audio) at single‑request concurrency; 728 ms at 4×; 1172 ms at 6× (Table 2).
  - RTF stays < 1 (0.47 → 0.66), meaning sustained real‑time streaming (Table 2).
  - Token rates: `Thinker` 75→53 tok/s; `Talker` 140→110 tok/s as concurrency rises (Table 2).

- Non‑degradation evidence (Table 16)
  - Language parity: MMLU 81.69 (`Omni‑Base`) vs 81.24 (text‑only base).
  - Vision gains: MMMUval 59.33 vs 57.22 (vision‑only base); InfoVQA 83.31 vs 81.17; AI2D and ChartQA slightly up.
  - Video mixed: LVBench improves (51.07 vs 48.61), MVBench slightly down.

Robustness and ablations
- Thinking vs Instruct for perception tasks: Appendix 9.1 shows `Thinking` underperforms `Instruct` for ASR/S2TT and music understanding, suggesting that explicit chain‑of‑thought adds little to primarily perceptual tasks and may introduce hallucinations.
- Long‑video limitations acknowledged (Table 10 discussion): positional extrapolation and context length constraints.

Overall assessment
- The breadth and depth of benchmarks, plus the controlled non‑degradation study, support the central claims for parity and audio excellence. Where results are mixed (e.g., long video), the paper discusses causes and future fixes (Section 5.1.3).

## 6. Limitations and Trade-offs
- Long video reasoning remains suboptimal
  - The `Thinking` model lags on Video‑MME/LVBench/MLVU compared with Gemini‑2.5‑Flash‑Thinking (Table 10), attributed to limited positional extrapolation and context length (Section 5.1.3).

- Language coverage and balance
  - Speech understanding covers 19 languages and synthesis 10 (Table 3). Text supports 119 languages, but audio modalities are narrower.

- Data and compute intensity
  - Pretraining uses ~2T multimodal tokens and a 20M‑hour audio corpus (Section 3). Such scale may be prohibitive for many labs; provenance/quality control details are not exhaustively enumerated here.

- Streaming design trade‑offs
  - The 12.5 Hz codec rate (80 ms per frame) lowers latency and compute but could limit the finest‑grain prosodic control compared with higher‑rate codecs; the paper argues fidelity is still superior due to multi‑codebook capacity and the ConvNet renderer (Sections 2.4–2.5), but no direct ablation of token rate is shown.

- Reasoning vs perception interference
  - `Thinking` improves complex reasoning but can reduce performance on pure perception tasks (Appendix 9.1), underscoring a need for mode selection or routing between “thinking” and “non‑thinking” behaviors.

- Generality of non‑degradation claim
  - The controlled study is at one scale (30B‑A3B) with matched schedules (Table 16). The paper notes cost prevented a full sweep across sizes (Section 6), leaving open how broadly the finding generalizes.

- Evaluation biases
  - Some post‑training uses model‑as‑judge rewards (Section 4.1). Although references are provided to stabilize judgments, automatic evaluators can encode biases or style preferences.

## 7. Implications and Future Directions
- Field impact
  - Provides concrete evidence that early joint multimodal training can achieve language/vision parity while adding strong audio and audio‑visual abilities (Section 6; Table 16). This challenges the community view that multimodal integration inevitably dilutes core language skills.
  - Establishes a practical recipe for real‑time, low‑latency speech interaction in a unified model (Tables 1–2), likely to become the new baseline for voice assistants.

- Practical applications
  - Real-time multilingual assistants with speech understanding up to 40 minutes and interactive synthesis at sub‑second first packet (Abstract; Sections 2.5, 5.2).
  - Meeting transcription, voice chat, and live translation across 19/10 input/output languages (Table 3).
  - Audio‑visual agents for video understanding, surveillance triage, or media QA, supported by gains on WorldSense and DailyOmni/VideoHolmes (Tables 11–12).
  - Research tooling: an open audio captioner to generate labels and enable benchmarking of audio‑centric tasks (Section 4.3; Appendix 9.2). Models are released under Apache‑2.0 (Abstract).

- Research directions
  - Architecture: Improve long‑video handling via stronger positional extrapolation and longer contexts; explore routing between thinking vs. perception modes to avoid hallucinations on ASR/Music (Section 5.1.3; Appendix 9.1).
  - Audio: Extend to multi‑speaker ASR, diarization, and more languages; explore higher‑rate or adaptive codecs if ultra‑fine prosody matters.
  - Vision: Enhance video OCR and temporal grounding; integrate proactive audiovisual learning (Conclusion).
  - Agentic workflows: Tighter coupling with function calling and retrieval, leveraging the decoupled Thinker–Talker to enforce safety and control (Section 2.1; Conclusion).

> Key quantitative highlights to remember:
> - First packet latency: 234 ms (audio), 547 ms (video) at single concurrency (Table 1).
> - VoiceBench Overall: 89.5 (Omni‑Thinking), nearly tied with Gemini‑2.5‑Pro 89.6 (Table 7).
> - Librispeech WER: 1.22/2.48 (Omni‑Instruct), better than GPT‑4o‑Transcribe 1.39/3.75 (Table 6).
> - WorldSense: 54.0 (Omni‑Instruct) vs 50.9 (Gemini‑2.5‑Flash) and 47.1 prior open‑source SOTA (Table 11).
> - Non‑degradation: MMLU 81.69 (Omni‑Base) vs 81.24 (Text‑only); MMMUval 59.33 (Omni‑Base) vs 57.22 (Vision‑only) (Table 16).

In sum, Qwen3‑Omni is a compelling demonstration that a unified multimodal model can retain single‑modality strength while unlocking low‑latency, real‑time speech and cross‑modal reasoning—backed by broad empirical evidence and a clear systems design that balances accuracy, latency, and concurrency.
