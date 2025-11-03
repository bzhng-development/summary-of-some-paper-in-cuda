# Qwen2.5‑Omni Technical Report

**ArXiv:** [2503.20215](https://arxiv.org/abs/2503.20215)
**Authors:** Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin
**Institutions:** Alibaba Cloud / Qwen Team

## 🎯 Pitch

Qwen2.5-Omni introduces a groundbreaking omni-model capable of synchronizing audio, video, and textual inputs to generate coherent text and speech outputs in real time. This innovation not only surpasses existing systems in multimodal understanding benchmarks but also simplifies engineering complexity by eliminating traditional chained component architectures, paving the way for more efficient and responsive real-world applications in voice assistants and beyond.

---

## 1. Executive Summary (2-3 sentences)
Qwen2.5-Omni introduces a single, end-to-end “omni-model” that can listen (audio), look (images and video), read (text), and respond with both text and natural speech in real time. It tackles two long-standing bottlenecks at once: synchronizing time across audio–video inputs and co-generating text and speech without the two interfering, yielding state-of-the-art results on multimodal understanding (e.g., OmniBench) and competitive zero-shot speech generation while remaining comparable to specialist models on individual modalities (Sections 2–5; Tables 5–9).

## 2. Context and Motivation
- Problem addressed:
  - Build a unified system that can simultaneously perceive multiple input modalities (text, audio, images, video) and generate both text and speech responses in a streaming setting (Abstract; Section 1).
  - Key sub-problems:
    - Efficient, synchronized processing of long, mixed audio–video inputs.
    - Avoiding interference when training one model to output both text and speech at the same time.
    - Reducing initial output latency for real-time interaction (Section 1; Section 2.4).

- Why it matters:
  - Real-world assistants must understand conversations and visual scenes and reply instantly with either voice or text. Prior systems commonly chain separate components (ASR → LLM → TTS) or handle modalities in isolation, adding latency and error propagation (Section 1).
  - Theoretically, a unified architecture can share representations across modalities, improving cross-modal reasoning and reducing engineering complexity.

- Prior approaches and gaps:
  - LALMs and LVLMs extend LLMs to audio or vision, but usually not both together in one end-to-end stream, or they do not co-generate speech natively (Section 1).
  - Existing omni models often lack tight audio–video time alignment, or rely on multi-stage pipelines that increase delay.
  - Speech generation is usually a separate model; when integrated, text and speech often interfere during training, degrading either text quality or speech robustness.

- Positioning:
  - Qwen2.5-Omni aims to be a unified, streaming-capable model. It introduces:
    - A time-aligned positional embedding to synchronize audio and video (TMRoPE, Section 2.2; Figure 3).
    - A `Thinker–Talker` architecture that separates text generation (Thinker) from speech token generation (Talker), while keeping them end-to-end and context-synchronized (Section 2.1; Figure 2).
    - Streaming-oriented encoders and a sliding-window decoder for low-latency speech (Section 2.4; Figure 4).

## 3. Technical Approach
Step-by-step overview of the system in Figure 2:

- Overall decomposition (Section 2.1):
  - `Thinker`: a Transformer decoder LLM that ingests text plus features from an audio encoder and a vision encoder, and produces high-level hidden states and text tokens.
  - `Talker`: a dual-track autoregressive Transformer that reads Thinker’s hidden states and Thinker’s already-sampled text tokens to generate streaming speech tokens (and optionally continue text tokens for training stability).
  - Encoders:
    - Audio encoder from Qwen2-Audio, operating on mel-spectrograms (128 channels, 25ms window, 10ms hop), producing one representation per ~40ms of audio (Section 2.2).
    - Vision encoder from Qwen2.5-VL (ViT ~675M params) for both images and videos (Section 2.2).

- Synchronizing audio–video with TMRoPE (Section 2.2; Figure 3):
  - Background: `RoPE` (rotary positional embedding) rotates token embeddings as a function of position, encoding order information. `M-RoPE` (multimodal RoPE) extends this to 3D (e.g., height/width for vision).
  - New contribution: `TMRoPE` (Time-aligned Multimodal RoPE) adds explicit absolute time to M-RoPE by factorizing positions into three components:
    - `temporal` (absolute time),
    - `height`, and `width` (for spatial layout).
  - How it works:
    - Text and audio tokens use identical position IDs across spatial components; audio receives an absolute temporal ID where 1 temporal unit ≈ 40ms.
    - For images: temporal IDs are constant within an image; height/width IDs vary spatially.
    - For videos: treat frames like images but increment temporal IDs per frame. Because frame rates vary, temporal ID increments are dynamically adjusted so 1 ID ≈ 40ms.
    - For mixed modalities, each modality’s position IDs start after the previous modality’s max ID to avoid collisions.
  - Interleaving for fused perception: For “video with audio,” features are chunked into 2-second windows; within each window, visual tokens are placed before audio tokens, then windows are interleaved sequentially (Section 2.2 and Figure 3). This preserves causality and allows the LLM to “see and hear” the same moments together.

- Text and speech generation (Sections 2.3–2.4):
  - Text path:
    - Thinker autoregressively generates text in the usual LLM way (top-p, repetition penalty supported; Section 2.3).
  - Speech path:
    - `Talker` receives:
      - Thinker’s high-level hidden states (carry semantics, discourse state, and prosodic intent),
      - embeddings of Thinker’s sampled text tokens (to disambiguate phonetics that semantics alone cannot resolve; Section 2.3).
    - `qwen-tts-tokenizer`: a speech codec that discretizes audio into tokens that a causal audio decoder can streamingly reconstruct (Section 2.3). Inference: Talker autoregressively predicts these audio tokens.
    - No enforced word/timestamp alignment to text is required, simplifying training and enabling natural prosody (Section 2.3).
    - Code-to-audio synthesis uses:
      - A Diffusion Transformer (`DiT`) trained with `Flow Matching` to convert code tokens into mel-spectrograms (Section 2.4).
        - Flow Matching learns a continuous transport from noise to data without iterative score estimation complexity.
      - A modified `BigVGAN` vocoder to render mel into waveform (Section 2.4).

- Streaming design for low latency (Section 2.4; Figure 4):
  - Problem: real-time interaction is dominated by (i) encoder delays, (ii) lag between first text token and first speech token, (iii) codec-to-audio delay, (iv) overall compute.
  - Encoder-side changes for `prefill`:
    - Audio encoder uses block-wise attention over 2-second chunks instead of full-context attention (Section 2.4).
    - Vision encoder applies FlashAttention and merges 2×2 neighboring tokens via a small MLP; patch size 14 enables packing variable-resolution images into a sequence (Section 2.4).
    - These enable `chunked-prefill`—ingesting and partially processing input while the user is still speaking or the video is still streaming.
  - Decoder-side `sliding-window` for code-to-audio:
    - Group adjacent audio code tokens into “blocks.” DiT sees only a limited window: current block + lookback of 2 blocks + lookahead of 1 block (Section 2.4; Figure 4).
    - This caps the receptive field to reduce initial output delay while preserving just enough context for audio continuity.
    - Mel and waveform are generated chunk-by-chunk to support continuous playback (Section 2.4).

- Training pipeline (Sections 3–4):
  - Stage 1 (encoders-only): Initialize LLM with `Qwen2.5`; vision encoder from `Qwen2.5-VL`; audio encoder from `Whisper-large-v3`. Freeze LLM; train adapters first, then encoders on large audio–text and image–text pairs (Section 3).
  - Stage 2 (end-to-end): Unfreeze all. Train on a large multimodal mixture: about 800B tokens (image/video related), 300B (audio related), and 100B (video-with-audio) plus pure text (Section 3). This builds cross-modal alignment and multitask skills.
  - Stage 3 (long context): Extend sequence length to 32,768 tokens with long audio/video/text data to improve long-sequence comprehension (Section 3).
  - Post-training (instruction tuning; Sections 4.1–4.3):
    - `ChatML` message format for multi-turn data spanning text, vision, audio, and mixed-modality dialogs (Section 4.1–4.2).
    - `Talker` three-stage process (Section 4.3):
      1) In-Context Learning (ICL) continuation: next-token prediction on speech tokens in dialog contexts with multimodal inputs; includes timbre disentanglement to avoid overfitting voice to text quirks.
      2) Stability via preference optimization: a DPO-style objective (“LDPO”) on pairwise speech generations ranked by WER and punctuation-pause errors (Equation (1), Section 4.3). Intuition: prefer the version with clearer, more accurate words and properly timed pauses.
         - Equation (1) uses a logistic loss over the log-odds between a “winner” `y_w` and “loser” `y_l` under model `P_θ` vs. a reference policy `P_ref`, scaled by β.
      3) Multi-speaker instruction fine-tuning for controllable voice styles and better naturalness.

## 4. Key Insights and Innovations
- Time-synchronized multimodal encoding (`TMRoPE`, Section 2.2; Figure 3):
  - What’s new: absolute time is injected into a 3D RoPE scheme—explicitly modeling temporal alignment across audio and video while preserving spatial structure for vision.
  - Why it matters: it gives the LLM a consistent, cross-modal clock (40ms units), enabling accurate fusion of what is seen and heard at the same moment. This is crucial for sound–vision reasoning and improves mixed-modality tasks (e.g., OmniBench results in Table 8).

- Separation of concerns in output (`Thinker–Talker`, Figure 2; Section 2.1–2.3):
  - What’s new: text generation (Thinker) and speech token generation (Talker) are specialized yet tightly coupled via shared history and direct access to Thinker’s hidden states plus sampled text tokens.
  - Why it matters: avoids interference between text and speech supervision while retaining end-to-end learning and synchronized context. It supports truly concurrent text-and-voice response in a single model.

- Streaming-oriented design throughout (Section 2.4; Figure 4):
  - What’s new: block-wise attention in encoders for `chunked-prefill`, and a `sliding-window` DiT for code-to-audio with lookback/lookahead to reduce initial packet delay.
  - Why it matters: traditional pipelines accumulate latency at every stage; this design reduces latency sources end-to-end to enable responsive voice assistants and live video dialogs.

- Robust speech generation with preference optimization (Section 4.3; Equation (1); Table 9):
  - What’s new: a DPO-style objective (“LDPO”) based on WER and punctuation-pause error ranks to improve stability (fewer misalignments, fewer pronunciation mistakes) in zero-shot, streaming speech.
  - Why it matters: speech generation quality often degrades under streaming and long prompts. The method shows measurable gains on “hard” sets (Table 9).

These are fundamental advances (time alignment, dual-head generation, streaming-friendly decoding) rather than incremental tweaks.

## 5. Experimental Analysis
- Evaluation setup (Section 5):
  - Two use modes:
    - `X→Text`: understanding tasks where input X ∈ {text, audio, image, video} and output is text.
    - `X→Speech`: speech generation given text (TTS-like) or multimodal context.
  - Datasets and metrics:
    - Text→Text: MMLU-Pro, MMLU-redux, LiveBench, GPQA, GSM8K, MATH, HumanEval, MBPP, MultiPL-E, LiveCodeBench (Section 5.1.1; Table 1).
    - Audio→Text: ASR (LibriSpeech, CommonVoice 15, Fleurs, WenetSpeech, VoxPopuli), S2TT (CoVoST2), Speech Entity Recognition, Vocal Sound Classification, Music understanding, Audio reasoning (MMAU), Voice chatting (VoiceBench) (Sections 5.1.2; Tables 2–3).
    - Image→Text: MMMU/Pro, MathVista, MathVision, MMBench, MMVet, MMStar, MME, MuirBench, CRPE, RealWorldQA, MME-RealWorld; OCR tasks like TextVQA, DocVQA, ChartQA, OCRBench_v2 (Section 5.1.3; Table 5). Visual grounding on RefCOCO/+/g and ODinW (Table 6).
    - Video→Text: Video-MME, MVBench, EgoSchema (Section 5.1.4; Table 7).
    - Multimodality→Text: OmniBench (Section 5.1.5; Table 8).
    - X→Speech: SEED for zero-shot speech WER and speaker similarity (SIM); subjective naturalness (NMOS) and single-speaker fine-tuning (Section 5.2; Tables 9–10).

- Main quantitative results:
  - Multimodality (OmniBench):
    - > “Qwen2.5-Omni-7B achieves 56.13% average (Speech 55.25%, Sound Event 60.00%, Music 52.83%), surpassing Gemini-1.5-Pro (42.91%) and other open omni models” (Table 8).
    - This strongly supports the claim of state-of-the-art cross-modal integration.
  - Audio reasoning and understanding:
    - > “On MMAU, Omni scores Sound 67.87, Music 69.16, Speech 59.76, Avg 65.60,” comfortably above Gemini-Pro-1.5 (Avg 54.90) and Qwen2-Audio (49.20) (Table 3).
    - ASR: Comparable to top models on LibriSpeech (test-clean 1.8 WER vs Whisper-v3 1.8; test-other 3.4 vs Whisper 3.6), and strong on CommonVoice/Fleurs (e.g., Fleurs-zh 3.0 WER vs Qwen2-Audio 7.5; Table 2).
    - S2TT (CoVoST2): better on en-de (30.2 vs 29.9), de-en (37.7 vs 35.2), zh-en (29.4 vs 24.4), while slightly worse on en-zh (41.4 vs 45.2) compared to Qwen2-Audio (Table 2).
    - Voice chatting: > “VoiceBench avg 74.12,” beating other similar-size omni models (Table 3).
  - Image understanding:
    - Comparable to a specialist LVLM (`Qwen2.5-VL-7B`): e.g., MMBench-V1.1-EN 81.8 vs 82.6; TextVQA 84.4 vs 84.9; DocVQA 95.2 vs 95.7 (Table 5).
    - Visual grounding: strong across RefCOCO/+/g; ODinW 42.2 mAP (vs 37.3 for Qwen2.5-VL, though still lower than Grounding DINO 55.0) (Table 6). This shows the omni model retains precise localization ability.
  - Video understanding:
    - Equal or better than Qwen2.5-VL: Video-MME w/sub 72.4 vs 71.6; MVBench 70.3 vs 69.6; EgoSchema 68.6 vs 65.0 (Table 7).
  - Text-only benchmarks:
    - Generally between `Qwen2-7B` and the stronger `Qwen2.5-7B`: e.g., MMLU-Pro 47.0 (vs 44.1 and 56.3); GSM8K 88.7 (vs 85.7 and 91.6) (Table 1). This suggests modest trade-offs for being omni while still robust.
  - End-to-end speech instruction following (voice in, text out):
    - Converting text instructions into speech and evaluating textual answers: Omni narrows the gap with text-only inputs and far exceeds previous audio LLMs. For example:
      - > “On GSM8K*, Omni 85.4 vs Qwen2-7B (text) 82.3 and Qwen2-Audio 18.4.”
      - > “On MMLU*, Omni 65.6 vs Qwen2-7B 69.3 and Qwen2-Audio 33.2” (Table 4).
  - Speech generation quality:
    - Zero-shot WER (lower is better):
      - > “SEED test-zh/en/hard: Omni-RL 1.42 / 2.33 / 6.54,” competitive with Seed-TTS-RL (1.00 / 1.94 / 6.42) and better than CosyVoice 2 (1.45 / 2.57 / 6.83) and F5-TTS (1.56 / 1.83 / 8.67) on ‘hard’ (Table 9).
    - Speaker similarity (higher is better):
      - Omni-RL achieves 0.754 / 0.641 / 0.752 (zh/en/hard); still below Seed-TTS-RL (0.801 / 0.766 / 0.782), indicating room to improve perceived voice match (Table 9).
    - Single-speaker fine-tuning:
      - En WER notably improves for some speakers (e.g., Speaker A 1.86 vs base-RL 2.33), and subjective naturalness `NMOS` reaches ~4.5, close to human on zh and strong on en (Table 10).

- Do the experiments support the claims?
  - Yes for multimodal integration: OmniBench, MMAU, and video benchmarks show clear gains (Tables 7–8; 3).
  - Yes for “comparable to specialist” on images/video/audio: close to Qwen2.5-VL and Whisper on many metrics (Tables 2, 5, 7).
  - Yes for end-to-end speech instruction following: strong voice-in performance vs prior audio LLMs, near text-input levels on several tasks (Table 4).
  - For speech generation, results are competitive; stability improves after LDPO/RL, but voice similarity trails the strongest TTS baselines in English (Table 9).

- Missing or limited analyses:
  - No ablation isolating TMRoPE vs. standard RoPE/M-RoPE to quantify its standalone impact.
  - No quantitative latency numbers (e.g., ms for first token, end-to-end real-time factor), despite extensive streaming design (Section 2.4).
  - Limited transparency on the massive mixed datasets (only token counts, Section 3) and their licensing/curation.

## 6. Limitations and Trade-offs
- Assumptions and design constraints:
  - The 40ms temporal unit is a design choice in TMRoPE; benefits might vary with other codecs or encoders (Section 2.2).
  - Interleaving video-first then audio within 2-second windows is one specific scheduling; other orderings or window sizes are not compared (Section 2.2).
- What is not addressed:
  - Precise latency/throughput under different hardware; no end-to-end real-time benchmarks (Section 2.4 discusses mechanisms but provides no ms numbers).
  - Fine-grained alignment between text and speech timestamps is deliberately avoided (Section 2.3). While simplifying training, it may limit applications that require word-level alignment (e.g., karaoke, precise dubbing).
  - Safety, bias, and privacy considerations are not discussed; important for deployment in voice assistants and video analysis.
- Computational and data costs:
  - Training involves hundreds of billions of multimodal tokens (Section 3), implying large compute budgets. Details on training stability, compute hours, and carbon footprint are not reported.
- Performance trade-offs:
  - Text-only benchmarks show a small drop vs. a dedicated text LLM (`Qwen2.5-7B`) (Table 1), suggesting some trade-off for omni capability.
  - Speaker similarity in English is below top TTS systems (Table 9), indicating remaining gaps in voice matching.

## 7. Implications and Future Directions
- How it changes the landscape:
  - Demonstrates that an end-to-end, single model can both understand mixed audio–video streams (aligned in time) and speak back while thinking—without a brittle ASR→LLM→TTS cascade. This simplifies engineering and reduces potential error compounding.
  - The `Thinker–Talker` split is a compelling template for multi-output LLMs (text + speech today; potentially text + video/audio tomorrow), balancing specialization with shared context.

- Follow-up research enabled/suggested:
  - Ablations and theoretical analysis of `TMRoPE`: quantify benefits across different frame rates, window sizes, and alternative time encodings.
  - Latency and stability profiling: standardized, device-level benchmarks for streaming (first-token latency, jitter, recovery from packet loss).
  - Extending Talker to other modalities: direct music generation, sound effects, or even video generation with synchronized speech.
  - Alignment-sensitive variants: optional word/timestamp alignment modes for applications needing precise sync.
  - Data governance: clearer documentation of multimodal data sources, licenses, and bias audits.

- Practical applications:
  - Real-time voice assistants that can watch and discuss videos, provide commentary for live events, or help with meetings by hearing and seeing the room.
  - Accessibility tools: live spoken descriptions of visual content; conversational captioning and translation.
  - Customer support and call centers: instant voice responses grounded in visual or document context.
  - Education and tutoring: multimodal explanations (charts, diagrams, spoken hints) with low latency.

Overall, Qwen2.5-Omni delivers a thoughtfully engineered approach to unified, streaming multimodal understanding and speech generation. Its strongest evidence lies in cross-modal benchmarks (OmniBench, MMAU) and robust, near-specialist performance on single-modality evaluations, with notable advances in end-to-end voice interaction. Key next steps are rigorous latency reporting, ablations for TMRoPE and streaming modules, and deeper analysis of data governance and safety.
