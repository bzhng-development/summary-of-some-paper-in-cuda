# Qwen3‑Omni Technical Report

**ArXiv:** [2509.17765](https://arxiv.org/abs/2509.17765)
**Authors:** Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin
**Institutions:** Alibaba’s Qwen Team (Alibaba Cloud / Alibaba AI Research)

## 🎯 Pitch

Qwen3-Omni revolutionizes multimodal AI by achieving state-of-the-art performance across text, vision, and audio without compromising any modalities, thanks to its innovative Thinker–Talker Mixture-of-Experts architecture. This advancement enables real-time, seamless interaction for applications like multilingual voice assistants and video analysis, setting a new standard for integrated AI systems that excel in complex, cross-modal tasks.

---

## 1. Executive Summary
Qwen3‑Omni is a single end‑to‑end multimodal model that deliberately avoids the usual “modality trade‑off”: it matches same‑size text‑only and vision‑only Qwen models while achieving state‑of‑the‑art results in audio and audio‑visual tasks. It does so with a Thinker–Talker Mixture‑of‑Experts (MoE) architecture, a new audio encoder (`AuT`), and a streaming speech stack (multi‑codebook + ConvNet vocoder) that yields a theoretical first‑packet latency of 234 ms for speech (Table 1, Table 2; Fig. 2).

## 2. Context and Motivation
- The gap addressed
  - Many LLM‑centric multimodal systems improve one modality (e.g., vision) at the expense of others (e.g., text), creating “modality trade‑offs.” The paper targets an integrated training recipe and architecture that avoids degrading any modality while enabling new cross‑modal abilities (Introduction; Fig. 1).
- Why it matters
  - Real‑world agents must hear, see, reason, and speak in real time, often over long contexts (meetings, videos). Qwen3‑Omni: 
    - Handles ASR and spoken‑language understanding up to 40 minutes per instance (Abstract, Sec. 2.5).
    - Supports 119 written languages, 19 spoken for understanding, 10 for speech generation (Table 3).
    - Streams natural speech with sub‑second perceived start time (234 ms theoretical first‑packet latency; Table 1–2).
- Prior approaches and shortcomings
  - Cascaded pipelines (ASR → LLM → TTS) add latency and lose cross‑modal cues; diffusion or block‑wise vocoders delay first audio packet; Whisper‑based audio encoders limit generality and streaming prefill (Sec. 2.1–2.2; Sec. 2.5).
  - Previous Qwen2.5‑Omni used Thinker–Talker but still relied on Whisper and block‑wise vocoding, and split audio‑visual inputs into fixed chunks (Sec. 2.1).
- Positioning
  - Qwen3‑Omni builds on Thinker–Talker but:
    - Upgrades both modules to MoE for higher throughput (Sec. 2.1, Table 1).
    - Introduces `AuT` (trained from scratch on 20M hours of supervised audio) for general‑purpose audio representations with streaming‑friendly attention (Sec. 2.2, Fig. 3).
    - Replaces diffusion/block vocoders with a lightweight causal ConvNet (`Code2Wav`) plus multi‑codebook streaming prediction for immediate synthesis (Sec. 2.4–2.5; Fig. 2).
    - Uses time‑aligned multimodal position embeddings (`TM‑RoPE`) for precise audio‑video alignment and arbitrary‑length streaming (Sec. 2.3).

## 3. Technical Approach
Step‑by‑step architecture and training pipeline.

- System overview (Fig. 2; Sec. 2.1)
  - `Thinker` (text generator): an MoE Transformer that performs multimodal understanding and text generation.
  - `Talker` (speech generator): an MoE Transformer that consumes multimodal features and generates discrete speech codec frames in a streaming manner.
  - Decoupling choice: Talker no longer consumes Thinker’s textual embeddings; instead it conditions directly on audio/visual features and the conversation history (Sec. 2.1). Rationale:
    - Text tokens vs. embeddings are information‑equivalent for content.
    - For tasks like speech translation or voice‑over, conditioning on audio/visual features helps preserve prosody/timbre and sync with video.
    - Enables separate system prompts for text style vs. audio style and permits external modules (RAG, safety) to intervene on text before speech (Sec. 2.1).
- Perception modules (Sec. 2.2–2.3; Fig. 3)
  - `AuT` audio encoder
    - What it is: an attention encoder–decoder trained on 20M hours of supervised audio (ASR + audio understanding). Input: 16 kHz waveform → 128‑channel mel spectrogram; Conv2D downsampling ×8 to a token rate of 12.5 Hz (≈1 token per 80 ms). Uses flash attention with dynamic windows covering 1–8 s to support real‑time prefill (Sec. 2.2; Fig. 3). ~0.6B parameters.
    - Why it matters: stronger, general‑purpose audio features than Whisper; streaming‑oriented prefill via block‑wise windows (Sec. 2.2).
  - Vision encoder
    - SigLIP2‑So400m (~543M params), trained on mixed image+video data; provides image/video features (Sec. 2.3).
  - Time‑aligned Multimodal RoPE (`TM‑RoPE`)
    - What it is: a positional encoding scheme that splits rotary angles into temporal, height, width components and aligns audio/video by absolute time IDs every 80 ms (Sec. 2.3).
    - Why it matters: improves long‑range temporal modeling vs. earlier M‑RoPE allocations and supports arbitrary‑length streaming without fixed 2‑s chunking (contrast with Qwen2.5‑Omni; Sec. 2.3).
- Speech generation stack (Sec. 2.4–2.5; Fig. 2)
  - Discrete speech representation: residual vector‑quantized (`RVQ`) “codebooks.” Thinker provides high‑level context; Talker autoregressively predicts one codec frame per step.
  - `MTP` module (multi‑token prediction): a tiny dense Transformer that, for each frame, predicts the residual codebooks beyond the first “zeroth” codebook predicted by Talker (Sec. 2.4).
    - Purpose: captures fine acoustic details (prosody, timbre) without delaying synthesis.
  - `Code2Wav`: a lightweight causal ConvNet that reconstructs waveform from the multi‑codebook tokens incrementally, with left‑context only attention (Sec. 2.4–2.5). 
    - Key design trade‑off: replace compute‑intensive diffusion/DiT vocoders with a small ConvNet to reduce latency and increase throughput (Sec. 2.4–2.5).
  - Streaming path
    - As soon as Talker emits the first token for a frame, MTP fills residual tokens; `Code2Wav` renders a short waveform segment immediately. With 12.5 Hz codec rate, a single token produces 80 ms of audio, enabling real‑time streaming (Sec. 2.5).
- Concurrency and latency engineering (Sec. 2.5; Table 1–2)
  - Chunked prefilling: audio/vision encoders output temporal chunks; Thinker and Talker prefill asynchronously so each can begin decoding earlier (Sec. 2.5).
  - MoE benefits: lower KV‑cache I/O, higher tokens/s under long sequences and concurrency (Sec. 2.5; Table 2).
  - Measured on vLLM with CUDA Graph and torch.compile: theoretical first‑packet latency is 234 ms for audio, 547 ms for video (Table 1–2). Generation RTF stays <1 across concurrency levels (Table 2).
- Training pipeline
  - Pretraining in three stages (Sec. 3)
    1) `S1` Encoder alignment: initialize Thinker from Qwen3 and bring audio/vision encoders (AuT, SigLIP2) into alignment on a frozen LLM via adapters, then encoders, avoiding the pitfall where encoders compensate for a frozen LLM (Sec. 3).
    2) `S2` General: ~2 trillion tokens spanning text (0.57T), audio (0.77T), image (0.82T), video (0.05T), and video‑audio (0.05T) with mixed prompts from early training to prevent modality siloing (Sec. 3).
    3) `S3` Long context: extend max length from 8,192 to 32,768 tokens; increase long audio/video proportion (Sec. 3).
  - Post‑training Thinker (Sec. 4.1)
    - Supervised fine‑tuning (SFT) with ChatML dialogues spanning text/vision/audio.
    - Strong‑to‑weak distillation: off‑policy (teacher responses) then on‑policy (student aligns logits to teacher via KL; teachers include Qwen3‑32B / Qwen3‑235B‑A22B).
    - RL with GSPO: rule‑based rewards for verifiable tasks; model‑based LLM‑as‑a‑judge for subjective multimodal tasks (Sec. 4.1).
  - Post‑training Talker (Sec. 4.2)
    - Stage 1: large‑scale multimodal‑context speech mapping.
    - Stage 2: continual pretraining (CPT) on high‑quality data + long‑context training to reduce hallucinations and improve stability.
    - Stage 3: multilingual DPO for generalization and stability.
    - Stage 4: speaker fine‑tuning for voice cloning and controllability.
  - Audio captioner (Sec. 4.3; Appx. 9.2)
    - Fine‑tune `Qwen3‑Omni‑30B‑A3B` into `…‑Captioner` to fill a gap in general‑purpose audio captioning.

Definitions of less common terms used above:
- `MoE` (Mixture‑of‑Experts): a model where a router activates only a small subset of expert feed‑forward networks per token, improving throughput at similar quality.
- `RVQ` (Residual Vector Quantization): a way to represent audio as a stack of discrete codebooks; each codebook encodes the residual left by the previous one.
- `MTP` (Multi‑Token Prediction): predicts multiple discrete tokens for a frame in one shot, reducing steps.
- `TM‑RoPE`: a rotary position embedding that separates temporal, height, and width angles and assigns absolute time IDs at 80 ms resolution for audio/video.

## 4. Key Insights and Innovations
- Integrated training without modality degradation (fundamental)
  - Evidence: A controlled 30B‑scale study trains a text‑only baseline, a vision‑only baseline, and `Omni` on identical text/vision corpora and matched FLOPs; `Omni` additionally sees audio/audio‑visual data. `Omni` matches or exceeds unimodal baselines in text and vision and improves some vision/OCR tasks (Table 16). Example: `MMMUval` (college‑level problems) improves from 57.22 (vision‑only) to 59.33 (Omni). Text benchmarks remain on par (e.g., MMLU 81.69 vs 81.24; Table 16).
- Streaming low‑latency speech with multi‑codebook + ConvNet vocoder (fundamental)
  - Immediate per‑frame synthesis: Talker emits first token, MTP predicts residuals, `Code2Wav` streams waveform with left‑context attention only (Fig. 2; Sec. 2.4–2.5).
  - Result: 234 ms theoretical first‑packet audio latency in cold start; RTF < 1 under load (Table 1–2).
  - Significance: removes block dependence and diffusion overhead; enables natural real‑time agents.
- `AuT` audio encoder trained from scratch at scale (novel subsystem)
  - 20M hours supervised audio with dynamic attention windows; 12.5 Hz token rate; block‑wise prefill caching (Sec. 2.2). 
  - Impact: strong general audio performance and long‑form streaming ability; underpins SOTA ASR/S2TT and audio reasoning (Table 6–8).
- Time‑aligned positional encoding for multimodal streams (incremental but impactful)
  - `TM‑RoPE` allocates more temporal angles (24) and ties absolute 80 ms IDs to audio/video frames; eliminates fixed 2‑s chunking used before, enabling arbitrary‑length streaming and better long‑range temporal extrapolation (Sec. 2.3).
- Thinking variant for cross‑modal reasoning + audio captioner (new capability)
  - `…‑Thinking` explicitly reasons over inputs from any modality; excels on audio‑visual reasoning tasks (Table 12) but is not optimal for pure perception benchmarks (Appx. 9.1).
  - `…‑Captioner` supplies detailed low‑hallucination audio descriptions for the community (Sec. 4.3; Appx. 9.2).

## 5. Experimental Analysis
- Evaluation methodology and setup
  - Modalities and tasks (Sec. 5)
    - Text→Text: general knowledge (MMLU‑Redux, GPQA), reasoning (AIME25, ZebraLogic), coding (MultiPL‑E), alignment/creative writing (IFEval, Creative Writing v3, WritingBench), agents (BFCL‑v3), multilingual (MultiIF, PolyMath).
    - Audio→Text: ASR and S2TT (LibriSpeech, WenetSpeech, FLEURS, CommonVoice), voice chat (VoiceBench), audio reasoning (MMAU, MMSU), and music understanding (RUL‑MuchoMusic, GTZAN, MTG‑Jamendo, MagnaTagATune).
    - Vision→Text: general VQA (MMStar, HallusionBench, MM‑MT‑Bench), math/STEM (MMMU, MMMU‑Pro, MathVista, MATH‑Vision), documents/OCR (AI2D, ChartQA), counting (CountBench), and video understanding (Video‑MME, LVBench, MLVU).
    - Audio‑Visual→Text: WorldSense (integration), DailyOmni and VideoHolmes (reasoning).
    - X→Speech: zero‑shot TTS on SEED, multilingual TTS on MiniMax set, cross‑lingual cloning on CosyVoice3 suite (Sec. 5.2).
  - Metrics
    - ASR/S2TT uses WER/BLEU; VoiceBench reports multiple sub‑scores and overall; music uses accuracy or micro‑F1 for multi‑label tagging; text/vision benchmarks use established metrics per dataset (Tables 4–15).
- Main quantitative results (selected highlights; all tables in Sec. 5)
  - Audio and audio‑visual leadership
    - ASR & S2TT (Table 6; `…‑Instruct`):
      - LibriSpeech WER: 1.22% (clean), 2.48% (other), surpassing GPT‑4o‑Transcribe (1.39/3.75) and Voxtral‑Small (1.56/3.30).
      - WenetSpeech (net|meeting) WER: 4.69|5.89 vs Seed‑ASR 4.66|5.69 and far below GPT‑4o‑Transcribe 15.30|32.27.
      - FLEURS‑avg (19 langs) WER: 5.33 vs Voxtral‑Small 8.09 and Gemini‑2.5‑Pro 5.55.
      - S2TT BLEU (FLEURS en→xx/xx→en/zh→xx/xx→zh): 37.50/31.08/25.17/33.13, broadly competitive with Voxtral‑Small and Gemini‑2.5‑Pro (Table 6).
    - VoiceBench overall (Table 7):
      - `…‑Thinking`: overall 88.8, second only to Gemini‑2.5‑Pro (89.6) and ahead of GPT‑4o‑Audio (86.8) and Qwen2.5‑Omni (73.6).
    - Audio reasoning (Table 7):
      - MMAU v05.15.25: `…‑Instruct` 77.5 and `…‑Flash‑Instruct` 77.6, higher than Gemini‑2.5‑Pro (77.4) and far above GPT‑4o‑Audio (62.5).
      - MMSU: `…‑Flash‑Thinking` 71.3, better than GPT‑4o‑Audio (56.4) and Gemini‑2.5‑Flash (70.2).
    - Music understanding (Table 8):
      - RUL‑MuchoMusic: 52.0–52.1 vs Gemini‑2.5‑Pro 49.4 and best specialist 47.6.
      - GTZAN accuracy: 93.0–93.1 (vs GPT‑4o‑Audio 76.5).
      - Multi‑label MTG and MagnaTagATune: Qwen3‑Omni achieves the best micro‑F1 across genre, mood/theme, instrument, top‑50 tags, and MagnaTagATune.
    - Audio‑Visual→Text (integration and reasoning):
      - WorldSense: 54.0–54.1, beating Gemini‑2.5‑Flash (50.9) and Qwen2.5‑Omni (45.4) (Table 11).
      - DailyOmni: `…‑Thinking` 75.8–76.2, exceeding Gemini‑2.5‑Flash‑Thinking 72.7 (Table 12).
      - VideoHolmes: `…‑Thinking` 57.3 vs previous open‑source SOTA 55.6 and Gemini‑2.5‑Flash‑Thinking 49.5 (Table 12).
  - Text and vision parity with same‑size unimodal models
    - Text→Text (Table 4–5):
      - `…‑Instruct` outperforms much larger Qwen3‑235B‑A22B Non‑Thinking on several benchmarks: AIME25 65.0 vs 24.7; ZebraLogic 76.0 vs 37.7; PolyMath 37.9 vs 27.0.
      - `…‑Thinking` is close to Gemini‑2.5‑Flash‑Thinking: GPQA 73.1 vs 82.8; WritingBench 85.5 vs 83.9; AIME25 73.7 vs 72.0 (Table 5).
      - Crucially, vs text‑only Qwen3‑30B models, Qwen3‑Omni matches or is comparable (Table 4–5; Sec. 5.1.1).
    - Vision→Text (Table 9–10):
      - `…‑Instruct` competitive with Qwen2.5‑VL‑72B; especially strong in math/STEM: MATH‑Visionfull 56.3 vs GPT4‑o 30.4 and Gemini‑2.0‑Flash 48.6 (Table 9).
      - `…‑Thinking` gains further on math+reasoning (e.g., MathVista‑mini 80.0; CountBench 88.6–92.5; Table 10). Long video understanding lags Gemini‑2.5‑Flash‑Thinking (Table 10; discussed as a limitation).
  - X→Speech generation (Sec. 5.2)
    - Zero‑shot TTS on SEED (WER; lower is better): `…‑30B‑A3B` achieves 1.07 (zh) and 1.39 (en). It is close to the best zh (CosyVoice3 0.71) and best en (CosyVoice3 1.45), and improves over prior Qwen2.5‑Omni‑7B (1.42/2.33) (Table 13). With RL optimization, English stability/consistency improves further (Table 13 note).
    - Multilingual TTS on MiniMax (Table 14): lower content‑consistency numbers are better (WER‑like); higher SIM is closer voice match. Qwen3‑Omni shows strong WER in Chinese (0.716) and English (1.069) and high SIM (0.77 range), outperforming or matching MiniMax/ElevenLabs on several languages.
    - Cross‑lingual cloning (Table 15): Qwen3‑Omni achieves lower errors in many any→en/ko directions (e.g., ko→en 3.34 vs CosyVoice3 4.19; en→ko 4.96 vs 5.87) and competitive any→ja even without phonetic normalization.
  - Latency and throughput (Table 1–2)
    - First‑packet audio latency 234 ms at 1× concurrency; Talker token rate 140 tok/s, Thinker 75 tok/s; RTF 0.47. Even at 6× concurrency, RTF stays 0.66 (Table 2).
- Robustness, ablations, and nuanced findings
  - Controlled non‑degradation experiment (Table 16) isolates multimodality effects at matched compute. Finding: early multimodal mixing does not harm language and can help vision/OCR; adding audio improves `MMMU` and OCR tasks while language gains from adding vision/audio are not observed (Sec. 6 discussion).
  - Thinking vs Instruct for perception: The `…‑Thinking` variant underperforms `…‑Instruct` on ASR and music tagging (Appx. 9.1; Tables 17–18). Insight: explicit chain‑of‑thought is unnecessary or even harmful for perception‑heavy tasks due to hallucination risk.
  - Failure modes: Long video benchmarks show lag vs top closed models (Table 10). The paper attributes this to limited positional extrapolation and context length (Sec. 5.1.3 narrative).
- Do the experiments support the claims?
  - The breadth of standardized benchmarks (text, vision, audio, audio‑visual) with competitive closed‑source baselines (GPT‑4o, Gemini‑2.5) and specialists (Seed‑ASR, Voxtral, music specialists) supports the SOTA claims in audio/audio‑visual and parity in text/vision.
  - The matched‑compute study (Table 16) is particularly convincing for the “no degradation” claim at 30B scale, though scalability to other sizes remains to be fully swept (Sec. 6).

## 6. Limitations and Trade‑offs
- Long video reasoning lags and scaling constraints
  - Performance on Video‑MME/LVBench/MLVU is below Gemini‑2.5‑Flash‑Thinking (Table 10). The paper attributes this to position extrapolation capacity and context window limitations (Sec. 5.1.3), implying architectural or training adjustments are needed.
- Heavy data and compute demands
  - Pretraining includes ~2T multimodal tokens and 20M hours of supervised audio for `AuT` (Sec. 2.2; Sec. 3). Replication requires significant resources; public checkpoints are 30B‑scale, but broader size sweeps are not reported due to cost (Sec. 6).
- “Theoretical” latency numbers
  - First‑packet latencies are derived under specific server‑side settings (vLLM, CUDA Graph, torch.compile) and reported as “theoretical” in cold‑start (Table 2). Real‑world network delays, device heterogeneity, and warmth of caches may alter observed latencies.
- Coverage of languages and domains
  - Speech understanding/generation is limited to 19/10 languages (Table 3). Low‑resource or code‑switching scenarios beyond this set are not deeply evaluated.
- Reward modeling and evaluation bias
  - RL uses model‑based judges (Qwen3, Qwen2.5‑VL) for non‑verifiable tasks (Sec. 4.1), which can introduce bias or reward hacking if not carefully controlled, despite safeguards like reference‑aware prompting.
- Perception vs reasoning trade‑off
  - The `Thinking` variant shows weaker ASR/music results (Appx. 9.1), suggesting that adding explicit reasoning pathways may increase hallucination or distract from low‑level perception unless gated appropriately.
- Audio codec rate choice
  - The 12.5 Hz (80 ms) codec rate is latency‑friendly but may coarsen ultra‑fine prosodic control in edge cases like rapid phonetic transitions or singing ornaments; although the multi‑codebook setup adds capacity, such trade‑offs aren’t ablated in detail.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that fully integrated, end‑to‑end multimodal training can avoid modality degradation while enabling strong cross‑modal reasoning (Sec. 7; Table 16). This challenges the assumption that best‑in‑class assistants must be cascades or ensembles.
  - Establishes a practical blueprint for real‑time, speech‑centric agents: multi‑codebook streaming, MTP, and ConvNet vocoding for low latency (Fig. 2; Table 2).
- Follow‑up research enabled
  - Position/extrapolation and long‑video context: augment TM‑RoPE or combine with learned time embeddings; extend context windows; evaluate memory‑augmented decoding for hour‑long videos (Sec. 5.1.3 limitation).
  - Adaptive routing between `Instruct` and `Thinking` modes for perception vs reasoning to curb hallucinations on ASR/music (Appx. 9.1).
  - Broaden language coverage for speech, code‑switching ASR/S2TT, and dialectal robustness (Table 3 hints at dialect work in “Flash” variants; Sec. 5).
  - Audio captioning as a foundation: leverage `…‑Captioner` to bootstrap audio‑grounded supervision for multimodal agents (Sec. 4.3; Appx. 9.2).
  - Transparent latency reporting across hardware tiers (edge, mobile) and integration with function calling, RAG, and safety interposers enabled by Thinker–Talker decoupling (Sec. 2.1).
- Practical applications
  - Real‑time multilingual voice assistants and meeting copilots (40‑minute ASR; Abstract; Fig. 1).
  - Audio‑visual reasoning for video understanding, surveillance/event analysis, and media QA (Tables 11–12).
  - Low‑latency speech translation or dubbing that preserves speaker traits and syncs with video via multimodal conditioning (Sec. 2.1, 2.4).
  - Music and sound understanding for media indexing, recommendation, and content moderation (Table 8).

> Bottom line: By carefully co‑training modalities from the start, aligning them in time (`TM‑RoPE`), and engineering an end‑to‑end streaming speech path (multi‑codebook + `Code2Wav`), Qwen3‑Omni shows that a single model can be competitive in text and vision while achieving state‑of‑the‑art audio performance with practical, low‑latency speech interaction (Fig. 2; Tables 1–2, 6–8, 11–12, 16).
