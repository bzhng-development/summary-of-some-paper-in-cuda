# Qwen2.5-Omni Technical Report

**ArXiv:** [2503.20215](https://arxiv.org/abs/2503.20215)

## 🎯 Pitch

Qwen2.5-Omni introduces a unified, end-to-end multimodal model that can perceive text, images, audio, and video, while simultaneously generating real-time text and high-quality speech responses. Its novel TMRoPE (Time-aligned Multimodal RoPE) and Thinker–Talker architecture enable block-wise streaming and synchronized audio-video understanding, breaking the traditional latency and modality barriers of prior systems. This fusion empowers next-generation conversational agents with fluid, low-latency, human-like interactions across modalities, achieving state-of-the-art results on leading multimodal benchmarks and paving the way toward truly intelligent multi-sensory AI assistants.

---

## 1. Executive Summary
Qwen2.5‑Omni is a single end‑to‑end multimodal model that ingests text, images, audio, and video and can stream both text and natural‑speech responses at the same time. It introduces two core mechanisms—`TMRoPE` for time‑aligned audio‑video fusion and a `Thinker–Talker` architecture for concurrent, low‑latency text and speech generation—yielding state‑of‑the‑art performance on multimodal understanding benchmarks while maintaining competitive text and high‑quality speech generation (Figures 1–4; Tables 1–10).

## 2. Context and Motivation
- Problem/gap addressed
  - Most systems handle only a subset of modalities (e.g., audio or vision) or require multi‑stage pipelines (ASR → LLM → TTS) that add latency and error propagation. This work targets a unified, end‑to‑end model that:
    - streams multimodal inputs (audio/video) in real time,
    - aligns audio and video temporally,
    - and generates text and speech simultaneously without mutual interference (Section 1; Figure 2).
- Why it matters
  - Real‑time assistants need to “see” and “hear” while “talking.” Low‑latency, synchronized understanding and response unlocks voice/video dialogue, live video reasoning, and natural conversational agents (Figure 1).
- Where prior approaches fall short
  - LVLMs (vision + language) and LALMs (audio + language) typically do not unify all modalities end‑to‑end or stream outputs; audio‑video timing is often loosely aligned; text and speech decoding interfere when trained together (Section 1).
- Positioning relative to existing work
  - Qwen2.5‑Omni compares to Qwen2.5‑VL on vision and Qwen2‑Audio on audio, and challenges recent omni models (e.g., Gemini 1.5, MiniCPM‑o, Baichuan‑Omni). It aims to match or exceed modality‑specific models while adding synchronized streaming generation (Sections 1–2; Tables 1–8).

## 3. Technical Approach
Step‑by‑step overview (Figure 2 and Section 2):

1) Inputs and encoders (“Perceivation,” Section 2.2; Figure 3)
- Text
  - Tokenized with the Qwen tokenizer (151,643 tokens).
- Audio
  - Resampled to 16 kHz; converted to a 128‑channel mel‑spectrogram (25 ms window, 10 ms hop). The audio encoder (from Qwen2‑Audio) outputs one frame ≈ 40 ms of original audio.
- Vision (images and video)
  - Vision encoder (≈675M‑param ViT from Qwen2.5‑VL) is trained on mixed image/video data. Video frames are sampled dynamically to preserve time, and a single image is treated as two identical frames for consistency with video.

2) Time alignment across modalities with `TMRoPE` (Section 2.2; Figure 3)
- What it is
  - `TMRoPE` (Time‑aligned Multimodal Rotary Position Embedding) decomposes rotary positional embedding into three components: `temporal`, `height`, and `width`.
- How it works
  - Text and audio use identical (1D‑like) position IDs; audio also gets an absolute temporal ID at a granularity of 40 ms per frame.
  - Images: a constant temporal ID per image; distinct `height`/`width` IDs across patches.
  - Video: temporal IDs increase per frame. Because FPS varies, the temporal ID step is dynamically set so that one temporal ID ≈ 40 ms.
  - When mixing modalities, position IDs are offset so the sequence is globally ordered (modality A’s max ID + 1 starts modality B).
- Why it matters
  - It gives the shared attention mechanism a consistent, time‑aware coordinate system across audio and video, enabling fine‑grained, time‑synced fusion (Figure 3).

3) Interleaved audio–video packing (Section 2.2; Figure 3)
- The model segments streams into 2‑second chunks. Within each 2‑second chunk, it places visual representations first and audio representations after, then interleaves such chunks in time. This ensures the LLM receives temporally adjacent audio–video content together.

4) The `Thinker–Talker` architecture (Section 2.1; Figure 2)
- `Thinker` (Transformer decoder)
  - Acts as the “brain.” It consumes text/audio/video encodings and produces high‑level hidden representations and autoregressive text tokens.
- `Talker` (dual‑track autoregressive Transformer decoder)
  - Acts as the “mouth.” It takes: (a) `Thinker`’s high‑level representations and (b) the embeddings of sampled `Thinker` text tokens. It autoregressively emits discrete audio tokens while having access to all `Thinker` historical context. Text and speech streams are generated concurrently.
- Rationale
  - Speech needs prosody and pragmatics before the entire text is known. Feeding `Thinker`’s hidden states gives `Talker` anticipation of tone/emotion while the discrete text tokens remove phonetic ambiguity (Section 2.3).

5) Speech tokenization and streaming vocoder (Sections 2.3–2.4; Figure 4)
- Discrete speech tokens: `qwen‑tts‑tokenizer` encodes speech compactly and supports streaming decoding via a causal audio decoder.
- Token‑to‑waveform decoding:
  - A `Flow‑Matching` DiT maps codes to mel‑spectrograms, then a modified `BigVGAN` reconstructs waveforms (Section 2.4).
  - Sliding‑window block attention restricts DiT’s receptive field to 4 blocks with 2‑block lookback and 1‑block lookahead, reducing initial latency while preserving local context (Figure 4).

6) Streaming inference and prefilling (Section 2.4)
- Encoders are revised for block‑wise streaming. Audio attention is confined to 2‑second blocks; the vision encoder uses FlashAttention and merges adjacent 2×2 tokens to keep compute bounded. These changes enable `chunked‑prefill`—feeding chunks early to build key/value caches so generation can start quickly.

7) Training pipeline (Sections 3–4)
- Pre‑training: three stages
  - Stage 1: Initialize the LLM from `Qwen2.5`, vision from `Qwen2.5‑VL`, audio from `Whisper‑large‑v3`. Freeze LLM; train encoders and adapters on large audio‑text and image‑text data to align them to the LLM.
  - Stage 2: Unfreeze all; train on expansive mixed multimodal data: ~800B image/video‑related tokens, 300B audio‑related, 100B video‑with‑audio, plus text corpora. Max length 8,192 tokens.
  - Stage 3: Extend sequences and include long audio/video to 32,768 tokens for long‑context understanding.
- Post‑training (supervised instruction tuning; Section 4)
  - `Thinker` uses `ChatML` dialogs across text, image, audio, and mixed modalities (Section 4.1–4.2).
  - `Talker` three‑stage training (Section 4.3):
    1) Next‑token continuation on multimodal dialog + spoken responses (learn monotonic mapping from semantics to speech; timbre disentanglement to avoid overfitting specific voices).
    2) Stability optimization with DPO‑style objective. Equation (1) (Section 4.3) shows an LDPO loss that prefers “good” speech samples (`yw`) over “bad” (`yl`) for the same context `x`, based on rewards tied to WER and punctuation‑pause errors.
    3) Multi‑speaker instruction fine‑tuning for naturalness and controllability of voice.

## 4. Key Insights and Innovations
- `Thinker–Talker` for concurrent, low‑interference generation (Figure 2; Sections 2.1–2.3)
  - Novelty: Separates semantic reasoning (`Thinker`) from speech realization (`Talker`), while letting `Talker` directly consume `Thinker`’s hidden states and sampled text tokens. This preserves semantic coherence and reduces interference between text and speech decoding—an issue in prior unified decoders.
  - Why it matters: Enables real‑time voice responses that already reflect intended tone and content while text is still being formed.
- `TMRoPE` for time‑aligned multimodal fusion (Figure 3; Section 2.2)
  - Novelty: Decomposes rotary embeddings into `temporal/height/width` and injects absolute time (40 ms steps) to synchronize audio and video streams—even under variable video frame rates—while maintaining a single attention space for all modalities.
  - Why it matters: Creating a shared, aligned coordinate system is essential for coherent audio‑video reasoning and streaming fusion.
- Interleaved 2‑second chunking + block‑wise encoders for streaming (Section 2.4)
  - Novelty: The encoders themselves are made streaming‑aware (2‑second audio blocks; ViT token merging + FlashAttention), enabling `chunked‑prefill` across modalities. Interleaving keeps audio and visual for the same window adjacent in the sequence.
  - Why it matters: Reduces first‑token latency and avoids quadratic growth in compute/memory with long inputs.
- Sliding‑window DiT with Flow‑Matching for streaming vocoding (Figure 4; Section 2.4)
  - Novelty: Constrains receptive field (2‑block lookback, 1‑block lookahead; 4 total) when mapping tokens → mel, then uses a streaming‑friendly `BigVGAN` to produce waveforms chunk by chunk.
  - Why it matters: Improves robustness and reduces initial delay for speech output; the lookahead helps maintain continuity without waiting for long future context.

These are fundamental architectural choices rather than small tweaks; they reconfigure how a multimodal LLM handles time, streaming, and dual‑output generation.

## 5. Experimental Analysis
Evaluation setup (Section 5; Tables 1–10):
- Modalities evaluated
  - Understanding: `X → Text` where `X ∈ {text, audio, image, video, mixed}`.
  - Generation: `X → Speech` focusing on zero‑shot and single‑speaker TTS‑style metrics.
- Datasets and metrics
  - Text: MMLU‑Pro, MMLU‑redux, LiveBench, GPQA, GSM8K, MATH, HumanEval, MBPP, MultiPL‑E, LiveCodeBench (Table 1).
  - Audio→Text: ASR (LibriSpeech, Common Voice 15, FLEURS, WenetSpeech, VoxPopuli), S2TT (CoVoST2), SER, VSC, music tasks; reasoning on MMAU; voice interaction on VoiceBench (Tables 2–3).
  - Image→Text: MMMU, MMMU‑Pro, MathVista, MathVision, MMBench‑V1.1, MMVet, MMStar, MME, MuirBench, CRPE, RealWorldQA, MME‑RealWorld, MM‑MT‑Bench; OCR tasks AI2D, TextVQA, DocVQA, ChartQA, OCRBench_v2; grounding (RefCOCO family, ODinW, point grounding) (Tables 5–6).
  - Video→Text: Video‑MME, MVBench, EgoSchema (Table 7).
  - Mixed‑modality: OmniBench (Table 8).
  - Speech generation: SEED‑TTS for WER (content consistency) and speaker similarity; NMOS for subjective naturalness on a self‑created set (Tables 9–10).

Main quantitative results
- Text→Text (Table 1)
  - Qwen2.5‑Omni‑7B generally sits between Qwen2‑7B and Qwen2.5‑7B. Examples:
    - MMLU‑redux: 71.0 vs Qwen2‑7B 67.3 and Qwen2.5‑7B 75.4.
    - MATH: 71.5 vs Qwen2‑7B 52.9 and Qwen2.5‑7B 75.5.
    - GSM8K: 88.7 vs Qwen2‑7B 85.7 and Qwen2.5‑7B 91.6.
  - Takeaway: multimodal additions do not collapse text ability; it stays competitive, though not equal to the strongest pure‑text 7B baseline.

- Audio→Text (Tables 2–3)
  - ASR examples:
    - LibriSpeech test‑other: 3.4 WER (vs Qwen2‑Audio 3.6; Whisper‑large‑v3 3.6).
    - Common Voice 15 en/zh: 7.6/5.2 WER (Qwen2‑Audio 8.6/6.9).
    - FLEURS zh/en: 3.0/4.1 (competitive with best models listed).
  - S2TT (CoVoST2): en‑de 30.2 BLEU, de‑en 37.7, en‑zh 41.4, zh‑en 29.4 (Table 2).
  - General audio understanding/reasoning:
    - VSC: 0.939 (ties Qwen2‑Audio best; Table 3).
    - MMAU avg: 65.60 vs Qwen2‑Audio 49.20; per‑subset: Sound 67.87, Music 69.16, Speech 59.76 (Table 3).
  - Voice interaction:
    - VoiceBench average: 74.12, best among listed omni/audio models of similar size (Table 3).

- Voice‑chatting with speech instructions (Table 4)
  - On converted speech prompts, Qwen2.5‑Omni narrows the gap with a text‑prompt LLM:
    - GSM8K (math word problems): 85.4 (voice) vs Qwen2‑7B text 82.3.
    - MMLU: 65.6 (voice) vs Qwen2‑7B text 69.3.
  - > Table 4: “Qwen2.5‑Omni‑7B … GSM8K* 85.4; MMLU* 65.6 … Qwen2‑7B (text) GSM8K* 82.3; MMLU* 69.3.”

- Image→Text (Table 5) and grounding (Table 6)
  - Comparable to Qwen2.5‑VL‑7B and often ahead of other open‑source omni models:
    - MMMUval: 59.2 (vs Qwen2.5‑VL 58.6).
    - MMBench‑V1.1‑EN: 81.8 (vs 82.6).
    - TextVQA: 84.4; DocVQA: 95.2; ChartQA: 85.3.
  - Grounding:
    - RefCOCO(val): 90.5; RefCOCO+(testA): 91.0; ODinW mAP: 42.2 (Table 6).

- Video→Text (Table 7)
  - Competitive with the vision specialist:
    - Video‑MME with subtitles: 72.4 (vs Qwen2.5‑VL 71.6); MVBench: 70.3 (vs 69.6).

- Mixed‑modality understanding (Table 8)
  - State of the art on OmniBench:
    - > Table 8: “Qwen2.5‑Omni‑7B: Speech 55.25%, Sound Event 60.00%, Music 52.83%, Avg 56.13%.”

- Speech generation quality (Tables 9–10)
  - Zero‑shot on SEED:
    - > Table 9 (WER): “Qwen2.5‑Omni‑7B_RL: test‑zh 1.42%, test‑en 2.33%, test‑hard 6.54%.”
    - Similarity: 0.754/0.641/0.752 (zh/en/hard).
    - Competitive vs strong non‑streaming and streaming TTS (e.g., better WER than CosyVoice 2 on zh/en/hard in Table 9).
  - Single‑speaker (after speaker fine‑tuning):
    - > Table 10 (NMOS): “Speakers A–D ≈ 4.46–4.62,” approaching human (zh 4.51).
    - WER remains low (e.g., Speaker A zh/en: 1.29/1.86).

Do the experiments support the claims?
- The breadth of evaluations (Tables 1–10) substantiates: (i) strong multi‑modal understanding (especially audio reasoning and OmniBench), (ii) competitive text ability for a 7B model, and (iii) robust, streaming‑ready speech synthesis.
- What is less quantified
  - The paper details several latency‑oriented design choices (block‑wise encoders, sliding‑window vocoder; Section 2.4) but does not report end‑to‑end latency numbers or ablations isolating `TMRoPE` and `Thinker–Talker` contributions.

## 6. Limitations and Trade-offs
- Streaming design constraints (Sections 2.2, 2.4)
  - 2‑second chunking and 40 ms temporal discretization may limit ultra‑fine timing resolution (e.g., sub‑phonetic cues) or rapid context shifts crossing chunk boundaries.
  - Sliding‑window DiT restricts long‑range acoustic dependencies; prosodic planning beyond the 2‑block lookback + 1‑block lookahead may be limited.
- Missing latency metrics
  - Although multiple mechanisms aim to reduce initial packet delay (Section 2.4), the paper does not report measured latency or throughput for different hardware/batch sizes—a key practical metric for “streaming” claims.
- Data and compute intensity (Section 3)
  - Large‑scale pretraining (hundreds of billions of multimodal tokens) implies heavy compute and data requirements; reproducing the model may be costly.
- Scope of languages and domains
  - Many speech metrics emphasize zh/en; generalization to low‑resource languages, accents, or noisy environments is only indirectly addressed through aggregate benchmarks.
- Ablations and component isolation
  - No explicit ablation shows how much `TMRoPE`, interleaving, or `Thinker–Talker` individually contribute to the gains. This makes it hard to guide minimal implementations.
- Safety and reliability
  - The work mentions improving stability with DPO (Section 4.3), but broader safety, hallucination, and controllability analyses across modalities (e.g., video OCR errors highlighted in Conclusion) are not systematically quantified.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a single 7B‑class model can unify audio‑video understanding with synchronized, streaming text and speech output, moving assistants closer to natural human‑like interaction. Strong OmniBench and MMAU results suggest unified training can surpass specialist models on cross‑modal reasoning (Tables 3 and 8).
- Enabled follow‑ups
  - Component ablations: quantify the independent effects of `TMRoPE`, interleaving, and `Thinker–Talker`.
  - Latency/efficiency studies: report end‑to‑end latency under varied hardware; explore adaptive chunk sizes and dynamic receptive fields.
  - Broader outputs: the paper’s Conclusion points to generating images, videos, and music—natural next steps given the architecture’s multi‑output design.
  - Long‑horizon streaming: extend sliding windows for prosody planning; hierarchical prosody tokens for utterance‑level coherence.
  - Robustness and coverage: evaluate many more languages, accents, spontaneous speech, and real‑world video OCR—an explicit challenge identified in Section 6 (Conclusion).
- Practical applications
  - Real‑time voice and video assistants, live video tutoring/explanation with narrated responses, accessibility tools (describing scenes and conversing in speech), customer service kiosks, and on‑device or edge deployments where low latency is crucial.

In short, Qwen2.5‑Omni combines time‑aligned multimodal perception with concurrent text/speech generation in a single model. Its architectural innovations (`TMRoPE`, `Thinker–Talker`, streaming encoders, sliding‑window DiT) are the main enablers, and the empirical results—especially on OmniBench and audio reasoning—indicate clear benefits. The next steps are to quantify latency, provide ablations, and broaden outputs and language coverage.
