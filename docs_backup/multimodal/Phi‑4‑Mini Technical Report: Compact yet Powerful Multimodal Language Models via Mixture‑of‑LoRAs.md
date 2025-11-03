# Phi‑4‑Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture‑of‑LoRAs

**ArXiv:** [2503.01743](https://arxiv.org/abs/2503.01743)
**Authors:** Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dongdong Chen, Junkun Chen, Weizhu Chen, Yen‑Chun Chen, Yi‑ling Chen, Qi Dai, Xiyang Dai, Ruchao Fan, Mei Gao, Min Gao, Amit Garg, Abhishek Goswami, Junheng Hao, Amr Hendy, Yuxuan Hu, Xin Jin, Mahmoud Khademi, Dongwoo Kim, Young Jin Kim, Gina Lee, Jinyu Li, Yun‑sheng Li, Chen Liang, Xihui Lin, Zeqi Lin, Meng‑Jie Liu, Yang Liu, Gilsinia Lopez, Chong Luo, Piyush Madan, Vadim Mazalov, Ali Mousavi, Anh Nguyen, Jing Pan, Daniel Perez‑Becker, Jacob Platin, Thomas Portet, Kai Qiu, Bo Ren, Liliang Ren, Sambuddha Roy, Ning Shang, Yelong Shen, Saksham Singhal, Subhojit Som, Xiaocheng Song, Tetyana Sych, Praneetha Vaddamanu, Shuohang Wang, Yiming Wang, Zhenghao Wang, Haibin Wu, Haoran Xu, Weijian Xu, Yifan Yang, Ziyi Yang, Donghan Yu, Ishmam Zabir, Jianwen Zhang, Li Lyna Zhang, Yunan Zhang, Xiren Zhou
**Institutions:** Microsoft Research

## 🎯 Pitch

This paper introduces 'Mixture-of-LoRAs,' an innovative training design that adds modality-specific adapters to a frozen 3.8B-parameter language model, seamlessly integrating text, vision, and speech without degrading text performance. By achieving state-of-the-art results on various multimodal tasks, the approach offers a cost-effective solution ideal for edge scenarios, simplifying deployment and maintenance while promising extensibility to future modalities.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces two small, open models: `Phi-4-Mini` (a 3.8B-parameter language model) and `Phi-4-Multimodal` (a unified 5.6B-parameter model that handles text, vision, and speech/audio). The core advance is a “Mixture-of-LoRAs” training/serving design that adds modality-specific adapters while keeping the language backbone frozen, yielding strong multimodal performance without sacrificing text-only quality (Section 2.2; Figure 1) and state-of-the-art results for a model of this size on several vision and speech tasks (Tables 1–3).

## 2. Context and Motivation
- Problem addressed
  - Small language models (SLMs) often trade off capability for efficiency. Adding multimodal capability typically requires fine-tuning the base language model, which can degrade its original text skills, or requires separate models per modality (Section 2.2 “However…” paragraphs).
  - Deployers want a single compact model that:
    - Maintains strong language, coding, and reasoning skills.
    - Adds vision and speech/audio without harming text performance.
    - Runs efficiently (lower memory, long-context support) on limited hardware.

- Why this matters
  - Practical impact: smaller unified models are cheaper to serve, easier to update, and better suited for edge or constrained environments. Maintaining a single checkpoint simplifies deployment and reduces maintenance costs, especially when adding new modalities (Abstract; Section 2.2).
  - Scientific significance: shows a path to modular multimodal training—via `LoRA` adapters—that preserves language performance and approaches the quality of fully fine-tuned, larger models (Section 2.2).

- Prior approaches and shortcomings
  - Fully fine-tuning multimodal models often degrades text-only skills, leading organizations to deploy multiple task-specific models (Section 2.2).
  - Cross-attention add-ons (e.g., Flamingo, Llama-Vision) preserve the backbone but underperform fully fine-tuned models on vision-language tasks (Section 2.2; references [ADL+22], [DJP+24]).
  - Hybrid SFT plus text alignment (e.g., NVLM) narrows the gap, but the literature has limited coverage of post-SFT stages and still doesn’t solve the “one model, many modalities” goal convincingly (Section 2.2; [DLW+24]).

- Positioning of this work
  - The paper proposes a unified architecture using a “mixture of modality-specific `LoRA` adapters” so the base LLM remains frozen and high-quality for text, while each modality is “plugged in” via its own encoder, projector, and `LoRA` branch (Section 2.2; Figure 1).
  - It also advances the training data recipe for a compact LLM—especially for math/coding—and explores a reasoning-enhanced variant trained with large-scale synthetic Chain-of-Thought (CoT) data (Sections 2.1, 3.1, 2.2.2 “Reasoning Training”; Table 9).

Definitions (used selectively):
- `LoRA` (Low-Rank Adaptation): a lightweight adapter that learns low-rank updates to weight matrices while keeping the original model weights frozen.
- `Cross-attention design`: an architectural add-on where the LLM attends directly to features from another modality via separate attention layers.
- `Projector`: a small MLP that maps visual/audio features into the text embedding space.
- `CoT` (Chain-of-Thought): intermediate reasoning steps generated during problem solving.
- `DPO` (Direct Preference Optimization): a preference-based fine-tuning objective using “preferred vs. dis-preferred” responses.

## 3. Technical Approach
Step-by-step overview of models, data, and training.

- Language backbone (`Phi-4-Mini`, 3.8B params; Section 2.1)
  - Architecture: 32 Transformer layers; hidden size 3072; tied input/output embeddings; 128K token context via `LongRoPE` (a position encoding scheme scaled for long sequences) (Sections 2, 2.1).
  - Efficiency features:
    - `GQA` (Group Query Attention): 24 query heads with 8 shared key/value heads to reduce KV-cache by ~⅔ (Section 2.1).
    - Fractional RoPE: only 75% of each head’s dimension is position-sensitive, leaving 25% position-agnostic for smoother behavior on long contexts (Section 2.1).
  - Tokenizer: `o200k_base` with 200,064 vocab to better support multilingual and multimodal outputs (Section 2).
  - Learning rate scaling: peak LR chosen as LR*(D) = B·D^-0.32 with B tuned over token budgets D ∈ {12.5B, 25B, 37.5B, 50B} (Section 2.1).

- Multimodal extension via “Mixture-of-LoRAs” (Section 2.2; Figure 1)
  - Principle: keep the 3.8B language model frozen; attach modality-specific components:
    - A modality encoder (vision or audio) produces dense features.
    - A 2-layer MLP `projector` maps those features to the LLM’s embedding space.
    - A modality-specific `LoRA` adapter (`LoRAV` for vision; `LoRAA` for audio) is added to all linear layers in the decoder, activated only in the relevant inference mode (Section 2.2.1).
  - Motivation vs alternatives:
    - Avoids catastrophic forgetting of text skills (a risk with full fine-tuning).
    - Outperforms cross-attention add-ons on many VL benchmarks while staying modular (Section 2.2; Tables 1–2).

- Vision pathway (Section 2.2.1 “Vision modality”)
  - Encoder: `SigLIP-400M`, further tuned with `LLM2CLIP` on large-scale image-text pairs at 448×448 (Section 2.2.1).
  - Projector: 2-layer MLP to 3072-D text embedding space.
  - Adapter: `LoRAV` (~370M params) applied on all LM linear layers, trained during SFT; vision encoder + projector add ~440M params.
  - Dynamic multi-crop: selects the number of crops based on image H×W vs crop size C; if too many crops, use aspect-ratio matching (inspired by InternVL2) to avoid enlarging tiny images (Section 2.2.1). This is a practical input packing strategy that preserves detail without exploding token count.

- Audio/speech pathway (Section 2.2.1 “Speech and Audio Modality”)
  - Input: 80-dim log-Mel filterbanks at 10 ms frames.
  - Encoder: 3 conv layers → 24 Conformer blocks (1024 attn dim, 1536 FFN dim, 16 heads). Conv sub-sampling rate 8 ⇒ 80 ms per LM token (Section 2.2.1).
  - Projector: 2-layer MLP from 1024-D speech features to 3072-D text embeddings.
  - Adapter: `LoRAA` (rank 320) on all attention and MLP layers; audio encoder + projector ~460M params; `LoRAA` ~460M params (Section 2.2.1).
  - Token rate: 1 minute of audio ≈ 750 tokens; with 128k context ⇒ theoretical ~2.8 hours, but the model was not fine-tuned on such long audio, so real performance may require further fine-tuning (Section 2.2.2 “Speech and Audio Training”).

- Training pipeline (Section 2.2.2; Section 3)
  - Language training (Sections 3.1.1–3.1.2)
    - Pre-training data: 5T-token corpus with improved filtering across languages, boosted math/coding, curated synthetic data (from Phi-4), and re-tuned mixtures emphasizing reasoning (Section 3.1.1).
    - Post-training: heavier function-calling and summarization data; challenging code completion set-ups, including “missing middle” fill-in (Section 3.1.2).
  - Vision training (four stages; Section 2.2.2 “Vision Training”)
    1) Projector Alignment: train only the projector on caption data to align modalities.
    2) Joint Vision Pretraining: train encoder + projector on large-scale vision tasks (OCR, dense understanding).
    3) Generative VL SFT: activate `LoRAV` and train encoder + projector + `LoRAV` on curated SFT to make the model generative for image+text.
    4) Multi-frame SFT: extend to multi-image/video-like tasks (up to 64k context), freezing the vision encoder.
  - Audio training (Section 2.2.2 “Speech and Audio Training”; Section 3.4)
    - Pre-training: align audio encoder + projector to LM on ~2M hours of anonymized speech-text pairs via ASR supervision; encoder/projector updated for 50k steps at 4e-5 LR; LM frozen (Section 3.4.1).
    - Post-training: unlock instruction-following across speech tasks with ~100M weighted SFT examples (ASR, AST, spoken QA, spoken-query QA, speech summarization up to 30-minute clips, audio understanding), 50k steps at 1e-4 LR; audio encoder frozen, update projector + `LoRAA` (Section 3.4.2).
  - Joint vision–speech training: freeze LM + audio pathway; fine-tune vision encoder/projector/`LoRAV` on vision-speech SFT, mixed with language and vision post-training data to maintain performance (Section 2.2.2 “Vision-speech Joint Training”).
  - Reasoning enhancement (three stages; Section 2.2.2 “Reasoning Training”)
    1) Distill pre-training: ~60B CoT tokens generated by frontier LLMs; filter by rejection sampling to remove incorrect chains.
    2) Distill fine-tuning: ~200K curated, diverse CoT SFT samples.
    3) Roll-Out DPO: ~300K preference pairs built from filtered incorrect outputs vs corrected “preferred” responses (Table 9 ablation).

Analogy for “Mixture-of-LoRAs”: think of the base LLM as a universal interpreter that you never rewrite. To teach it new IO languages (vision, audio), you add a translator (encoder+projector) and a small “accent” module (`LoRA`) that tweaks how the interpreter speaks when that IO language is present. The interpreter’s native language skills remain intact because its core weights never change.

## 4. Key Insights and Innovations
- Unified multimodality without catastrophic forgetting (fundamental)
  - Insight: freeze the language backbone; attach modality-specific `LoRA` adapters plus encoders/projectors. This preserves text-only performance, unlike fully fine-tuning (Section 2.2; Figure 1).
  - Significance: enables one checkpoint for text, vision+text, speech-only, and vision+speech; easy to add new modalities by training another adapter without harming existing ones (Abstract; Section 2.2).

- Dynamic multi-crop and staged vision training (incremental but effective)
  - A practical cropping scheme that avoids blowing up tokens for unusual aspect ratios and prevents over-upscaling tiny images (Section 2.2.1). This, coupled with projector alignment → joint pretraining → VL SFT → multi-frame SFT, yields strong OCR, chart, science-reasoning, and multi-image performance (Table 1).

- High-quality speech pipeline with long-form summarization (novel capability at this size)
  - Two-stage audio training aligns representations and then unlocks instruction-following across many tasks, including 30-minute speech summarization—a first among open-source models as claimed (Section 4.1.2; Table 3 “SSUM”).
  - Token-rate-aware design (80 ms/token) and 128k context enable very long audio handling in principle (Section 2.2.1; caveat in Section 2.2.2).

- Reasoning extension for a 3.8B SLM via large CoT distillation + DPO (fundamental for compact reasoning)
  - A three-stage recipe (distill pretrain → fine-tune → Roll-Out DPO) produces a 3.8B model with AIME/MATH-500/GPQA results competitive with popular 7–8B distilled reasoning models (Table 9).

- Engineering choices for efficient long-context generation (incremental, pragmatic)
  - `GQA` to reduce KV-cache memory, `LongRoPE` for 128k tokens, fractional RoPE to keep part of each head position-agnostic (Section 2.1).

## 5. Experimental Analysis
- Evaluation methodology
  - Vision-language: 13 single-image and 2 multi-image/video benchmarks (e.g., MMMU, MathVista, OCRBench, BLINK, VideoMME), consistent internal pipeline across models; some server-evaluated sets (DocVQA, InfoVQA) reported via submissions (Section 4.1.1; Table 1).
  - Vision-speech: four ShareGPT4o benchmarks (AI2D, ChartQA, DocVQA, InfoVQA) using spoken prompts plus images; text instructions added for Gemini to force structured outputs (Section 4.1.1; Table 2).
  - Speech/audio: ASR (CV15, FLEURS, OpenASR leaderboard), AST (CoVoST2, FLEURS), spoken QA (MT-Bench, MMMLU), summarization (Golden3, AMI), audio understanding (AIRBench-chat, MMAU). Zero-shot decoding; CoT decoding for AST with a transcript+translation format separated by <sep> (Sections 4.1.2; Tables 3–6).
  - Language & coding: broad academic suites (MMLU/MMLU-Pro, BigBench-Hard, GSM-8K, MATH, GPQA, ARC-C, BoolQ, etc.), plus code sets (BigCodeBench, HumanEval, MBPP, LiveBench-Code, Spider) (Sections 4.2.1–4.2.2; Tables 7–8).
  - Safety: Azure AI Evaluation SDK across harm categories; jailbreak tests; XSTest for exaggerated refusals; multilingual harm rates; audio safety and fairness; sensitive attribute inference (Section 5; Tables 10–15).

- Main quantitative results (selected highlights)
  - Vision-language (Table 1):
    - Average across 15 VL tasks: `Phi-4-Multimodal` 72.0 vs `Qwen2.5-VL-3B` 68.7 and `InternVL2.5-4B` 68.8; close to `Qwen2.5-VL-7B` 73.3 and `InternVL2.5-8B` 71.1.
    - Examples:
      - MMMU(val): 55.1 vs 47.0 (Qwen2.5-VL-3B) and 48.3 (InternVL2.5-4B).
      - MathVista(testmini): 62.4 vs 60.8 (Qwen2.5-VL-3B) and 51.2 (InternVL2.5-4B).
      - OCR-heavy sets: DocVQA 93.2; OCRBench 84.4.
  - Vision-speech (Table 2):
    - Average across 4 sets: 72.2 vs `InternOmni-8.7B` 62.6. Large margins on AI2D (68.9 vs 53.9) and ChartQA (69.0 vs 56.1).
  - Speech/audio (Tables 3–6):
    - ASR (WER↓): OpenASR average 6.14, outperforming `nvidia/canary-1B` (6.50) by 5.5% relative and topping the HF leaderboard at the time (Tables 3–4).
      > “Phi-4-Multimodal is 5.5% relatively better… now ranks No.1 on the leaderboard as of 1/14/2025.” (Table 3 notes)
    - AST (BLEU↑): Best on CoVoST2 in both X→EN and EN→X; CoT improves 1–2 BLEU (Table 5).
    - Spoken QA: MT-Bench score 7.05 (below GPT-4o/Gemini ~8.1); MMMLU 38.5% vs GPT-4o/Gemini ~72% (Table 6) — strong conversational ability but weaker general-knowledge reasoning via speech.
    - Speech summarization (first open-source with this capability): near GPT-4o on both sets—Golden3 overall 6.28 vs 6.76; AMI 6.29 vs 6.53; low hallucination flags (Table 6).
    - Audio understanding: AIRBench-chat 6.98 avg (beats Qwen2-audio 6.93); MMAU 55.56% vs Qwen2-audio 52.50% (Table 6).
  - Language (Table 7):
    - Average across many tasks: `Phi-4-Mini` 64.9 vs `Llama-3.2-3B` 58.0, `Ministral-3B` 58.3; comparable to `Llama-3.1-8B` 63.9. Strong on math: GSM-8K 88.6; MATH 64.0; MGSM 63.9.
  - Coding (Table 8):
    - Average 49.0 across 9 code benchmarks, beating all 3B baselines and many 8–9B models; e.g., HumanEval 74.4 (higher than Llama-3.1-8B 66.5).
  - Reasoning-enhanced 3.8B (Table 9; ablation):
    - Final model: AIME 50.0, MATH-500 90.4, GPQA-Diamond 49.0—comparable to `DeepSeek-R1-Distill-Qwen-7B` (53.3/91.4/49.5) and above `DeepSeek-R1-Distill-Llama-8B`.
    - Ablation shows stepwise gains from Distill pretraining → Distill FT → Roll-Out DPO.
  - Safety (Tables 10–12):
    - Harm content “Defect Rate” similar to peer SLMs; improved robustness to jailbreaks vs `Phi-3.5-mini` (Table 11).
    - Strong refusal of harmful prompts (IPRR 93.5%), moderate over-refusal on innocuous ones (VPRR 20.8%) (Table 12).
    - Multilingual safety: average 3.91% defect rate across Tier 1 languages, better than `Phi-3.5-mini` 6.31% (Table 13).
    - Audio safety: higher defect rates than GPT-4o but comparable to text-only rates; ISA (inference of sensitive attributes) can be reduced to 0.4% with a system prompt (Section 5.2; Table 14).
    - Vision safety: scores on RTVLM/VLGuard indicate safety on par or better than previous open models (Table 15).

- Are the experiments convincing?
  - The unified multimodal story is well supported by broad benchmarks: strong vision-language averages (Table 1), very strong ASR/AST (Tables 3–5), and legitimate long-form speech summarization (Table 6).
  - The “language does not degrade” claim is plausible because the LM is frozen, but the paper does not show A/B comparisons of text-only metrics before vs after adding modality LoRAs. It argues indirectly via design rather than a direct ablation (Sections 2.2, 4.1.1).
  - The reasoning-enhanced variant’s ablation (Table 9) credibly demonstrates the data/stage contributions.
  - Some evaluation prompts required care (e.g., structured outputs for Gemini; ASR prompts for GPT-4o), which the paper documents (Sections 4.1.1–4.1.2), but could introduce variability across closed baselines.

- Robustness checks and failure cases
  - Ablations: reasoning pipeline (Table 9). Less visible: ablations on dynamic multi-crop, LoRA rank choices, or router behavior across modalities.
  - Failure modes:
    - Spoken QA on knowledge-heavy MMMLU is far below GPT-4o/Gemini (Table 6).
    - Very long audio (>30 minutes) is only “theoretically” supported; not fine-tuned for it (Section 2.2.2).
    - GPT-4o appears prompt-sensitive on audio/music understanding, complicating closed-model comparisons (Section 4.1.2).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The base LLM remains frozen. This preserves text skills but may cap how tightly the multimodal behaviors can be integrated versus joint end-to-end fine-tuning (Section 2.2).
  - The approach assumes modality-specific `LoRA` branches do not interfere; real-world inter-modality interactions could still conflict without a learned router or gating policy.

- Scenarios not fully addressed
  - Ultra-long audio (hours): possible by context length, but the model is not fine-tuned for such durations; performance may degrade (Section 2.2.2).
  - Spoken QA requiring extensive general knowledge or multilingual reasoning lags behind large closed models (Table 6).
  - The vision-speech benchmarks use TTS-generated spoken queries, which may not fully reflect noisy real-world speech (Section 3.3).

- Computational and memory considerations
  - Although the core LLM is small (3.8B), each modality adds sizeable parameters: vision (~440M encoder/projector + ~370M LoRA) and audio (~460M encoder/projector + ~460M LoRA) (Section 2.2.1). This is still compact compared to large models, but the additions are not trivial for edge scenarios.
  - Multi-frame vision and long-context audio increase token counts; throughput depends on KV-cache memory despite GQA optimizations (Section 2.1; 2.2.2).

- Safety/data constraints
  - Audio safety training uses TTS from text-safety datasets (voice-only, no non-speech sounds, no audio-specific jailbreaks), which limits coverage (Section 5.2).
  - Multilingual capability receives fewer data resources than English due to coding emphasis (Section 6).

- Open questions
  - How well does “Mixture-of-LoRAs” scale to additional modalities (e.g., video encoders, sensors) or to duplex real-time constraints?
  - What is the best routing mechanism when multiple modalities are present concurrently?
  - Can text performance be further improved with minor joint fine-tuning without losing the preservation benefits?

## 7. Implications and Future Directions
- How this work changes the landscape
  - Demonstrates that a compact, frozen LLM can be the stable core of a high-performing multimodal system by attaching modality-specific `LoRA` branches—retaining text skills while achieving competitive vision/speech performance (Section 2.2; Tables 1–3).
  - Provides a practical blueprint for modular extensibility: add a new encoder/projector/LoRA for each modality without re-training the backbone (Abstract; Section 2.2 “extensible design”).

- Follow-up research enabled or suggested
  - Adaptive modality routers: learn to dynamically combine multiple `LoRA` branches when inputs mix modalities (beyond the current “activate the relevant LoRA” pattern).
  - Data-efficient multimodal SFT: identify minimal, high-quality instruction data that transfers best through projectors and `LoRA`s.
  - Ultra-long audio finetuning and evaluation to operationalize the 128k context promise (Section 2.2.2).
  - Further reasoning upgrades for 3–4B models, combining CoT distillation with lightweight test-time scaling (cf. S1; Table 9 vs [MYS+25]).

- Practical applications
  - Edge or cost-sensitive deployments requiring a single compact model for:
    - Document, chart, and diagram understanding with OCR (Table 1; strong DocVQA/InfoVQA/OCRBench).
    - Contact-center or meeting analytics via multilingual ASR/AST and long-form summarization (Tables 3, 6; Golden3/AMI).
    - Developer assistants that combine strong coding (Table 8) with visual context (UI screenshots) or spoken requirements.
    - Multilingual voice interfaces that don’t need explicit language prompts for ASR/AST (Tables 4–5).

> Bottom line: The paper’s key contribution is architectural and procedural—a modular “Mixture-of-LoRAs” that cleanly adds vision and speech to a frozen 3.8B LLM, backed by a carefully engineered data/training pipeline. The result is a compact yet capable unified model that leads its size class on many multimodal tasks, achieves best-in-class small-model ASR and competitive AST, and provides a credible recipe for small-model reasoning via large-scale CoT distillation (Figures 1–3; Tables 1–9).
