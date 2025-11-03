# KIMI-VL TECHNICAL REPORT

**ArXiv:** [2504.07491](https://arxiv.org/abs/2504.07491)

## 🎯 Pitch

Kimi-VL sets a new benchmark for open-source vision-language models by fusing a highly efficient Mixture-of-Experts language core with a native-resolution vision encoder, enabling advanced multimodal reasoning, 128K-token long-context processing, and ultra-high-resolution perception—all with just 2.8B parameters activated. Its 'Thinking' variant, fine-tuned with chain-of-thought and reinforcement learning, rivals or outperforms much larger proprietary and open models on demanding benchmarks, making state-of-the-art, cost-effective multimodal intelligence accessible for real-world applications in research and industry.

---

## 1. Executive Summary
Kimi-VL is an open-source vision–language model (VLM) that combines an efficient Mixture-of-Experts (MoE) language model with a native-resolution vision encoder to deliver strong multimodal reasoning, long-context processing (up to 128K tokens), and high-resolution perception while activating only about 2.8B parameters in the language decoder (Figure 3; Table 3). A “thinking” variant trained with long chain-of-thought (CoT) supervision and reinforcement learning (RL) greatly boosts hard reasoning benchmarks, achieving competitive results with far larger models while remaining highly compute- and token-efficient (Table 4; Figure 13).

## 2. Context and Motivation
- Problem addressed
  - Open-source VLMs lag behind top language-only models in efficiency and behind proprietary VLMs in advanced capabilities such as long reasoning, long-context understanding, and high-resolution perception. The gap is practical (cost/latency) and scientific (how to scale capabilities without scaling dense parameters).
- Why it matters
  - Real applications require: 
    - High-resolution UI grounding and desktop automation (agent tasks).
    - Long PDFs/videos and interleaved multimodal contexts (enterprise, education).
    - College-level reasoning and math in visual contexts (STEM workflows).
  - An efficient, open VLM that performs well across these areas lowers cost and widens accessibility.
- Prior approaches and shortcomings
  - Dense VLMs (e.g., `Qwen2.5-VL-7B`, `Gemma-3-12B-IT`) scale with cost and have limited long-CoT reasoning (Introduction; §4.1, Table 3).
  - Early MoE VLMs (e.g., `DeepSeek-VL2`, `Aria`) show promise but lack some combination of: long context, high-resolution native vision encoding, and long-thinking support (Introduction).
  - Many models resize or segment images, losing details crucial for high-resolution UI and documents.
- Positioning
  - Kimi-VL combines:
    - An MoE language model with only 2.8B activated parameters (16B total) for efficiency (Figure 3; §2.1).
    - `MoonViT`, a native-resolution vision encoder inspired by NaViT packing to preserve detail at arbitrary aspect ratios (Figure 3; §2.1).
    - A training curriculum that preserves text quality while adding multimodal breadth and long context (Figures 4–5; Tables 1–2).
    - A long-thinking extension using CoT SFT and RL for deep multimodal reasoning (Figure 5; Table 4).

## 3. Technical Approach
Step-by-step, from inputs to training to deployment.

- Architecture (Figure 3; §2.1)
  - `MoonViT` vision encoder (≈400M params)
    - Native-resolution input: images are patchified and concatenated as variable-length token sequences using NaViT-style “packing,” so the same compute kernels (e.g., FlashAttention) handle any resolution/aspect ratio (MoonViT subsection).
    - Spatial encoding: combines interpolated absolute embeddings (from SigLIP-SO-400M) with 2D `RoPE` (rotary position embedding) over height/width to better preserve fine-grained positions, especially at higher resolutions.
      - RoPE “base” for the LLM is later increased for long context (§2.3, Joint Long-context Activation).
    - Output are continuous visual tokens.
  - MLP projector with pixel shuffle (MoonViT → LLM)
    - Pixel shuffle compresses spatial dimensions (2×2 downsample) while increasing channels; a 2-layer MLP maps to the LLM embedding space.
  - MoE language model (`Moonlight`; 2.8B activated, 16B total)
    - MoE: multiple feed-forward “experts” where a router selects a small subset per token. This keeps computation low (“activated parameters”) while maintaining large capacity (“total parameters”).
    - Initialized from a 5.2T-token text-only checkpoint with 8K context, then jointly trained multimodally (§2.3).
- Training recipe (Figures 4–5; Table 1)
  - Optimizer and scale-out
    - Enhanced `Muon` optimizer with weight decay and distributed ZeRO-1 implementation for memory efficiency (§2.2).
    - 4D parallelism for throughput and long sequences (§2.5): Data Parallelism, Expert Parallelism (across MoE experts), Pipeline Parallelism, and Context Parallelism (sequence split with FlashAttention). Plus ZeRO-1 and selective activation checkpointing for memory (§2.5).
  - Stage 1: Standalone ViT training (2.0T tokens) + alignment (0.1T) (Table 1; §2.3)
    - CoCa-like dual objective: SigLIP contrastive loss + captioning loss with a tiny text decoder (ViT Training Stages).
    - Diverse targets (alt-text, synthetic captions, grounding, OCR). Progressive resolution sampling to cover small to ultra-high resolution.
    - After 2.0T, a 0.1T “alignment” step updates only `MoonViT` and projector to reduce initial perplexity when feeding the LLM.
  - Stage 2: Joint pre-training (1.4T) (Joint Pre-training Stage)
    - Mixes pure text (to preserve LLM skill) with multimodal data (caption, interleaved image–text, OCR, knowledge, video, agent; §3.1).
    - The proportion of multimodal data increases progressively.
  - Stage 3: Joint “cooldown” (0.6T) (Joint Cooldown Stage)
    - High-quality text + multimodal data with curated and synthetic QA pairs (math, knowledge, code) via rejection sampling and verification. The QA portion is kept small to avoid overfitting QA style while “activating” abilities.
  - Stage 4: Joint long-context activation (0.3T) (Joint Long-context Activation Stage)
    - Extends context 8K → 32K → 128K in two sub-stages by resetting RoPE inverse frequency base from 50,000 to 800,000 and quadrupling sequence length each time.
    - Data composition: 25% long data (long text, interleaved multimodal, long videos, long documents), 75% replay of shorter data. This preserves short-context ability while learning long-context.
    - Needle-in-a-Haystack (NIAH) recall up to 128K tokens:
      > Table 2: text haystack 87.0% recall at 128K; video haystack 91.7% recall at 128K.
  - Post-training (Figure 5; §2.4)
    - Joint SFT at 32K (1 epoch) then 128K (1 epoch) with ChatML formatting; mixes pure-text and multimodal dialogue; supervision on assistant outputs only (Joint SFT).
    - Long-CoT SFT: a small, high-quality warmup dataset of verified long reasoning traces that explicitly train “planning, evaluation, reflection, exploration.”
    - Reinforcement learning (RL): online policy mirror descent variant with an answer-only reward:
      - Objective (Eq. 1): maximize expected reward r(x, y, y*) with KL regularization to stabilize updates.
      - Binary rewards from ground truth; auxiliary length penalty to prevent overthinking; curriculum and prioritized sampling by difficulty/success rate for training efficiency (§2.4, Reinforcement Learning).
- “Thinking” variants and high-resolution continuation
  - `Kimi-VL-Thinking`: base model + long CoT SFT + RL; shows test-time scaling with longer “thinking tokens” (Figure 13).
  - `Kimi-VL-Thinking-2506`: further continues `MoonViT` to support up to 3.2 million pixels per image, integrates perception/video/long-doc/agent skills into the thinking model, and reduces output length for efficiency (Tables 4–5; §4.3).

## 4. Key Insights and Innovations
- Native-resolution vision without tiling or lossy resizing (MoonViT; Figure 3; §2.1)
  - What’s new: NaViT-style packing + 2D RoPE + SigLIP initialization enables high-resolution images (including ultra-wide/tall UI screens) to be processed as a single sequence.
  - Why it matters: Improves OCR, document layout understanding, and UI grounding—validated by strong scores on InfoVQA (83.2%), OCRBench (867/1000), ScreenSpot-Pro (34.5% base; 52.8% in the “2506” variant) (Table 3, Table 5).
- Efficiency via MoE with only 2.8B activated parameters (Figure 3; §2.1)
  - What’s new: A small-activated-parameter MoE LLM (“Moonlight”) plus a 400M vision encoder achieves competitive performance with dense 7B–12B VLMs (Table 3).
  - Why it matters: Lower inference cost and higher throughput; enables integration of long context (128K) and long reasoning within practical budgets.
- A staged training pipeline tailored for multimodal long context (Figures 4–5; Table 1; §2.3)
  - What’s new: After ViT pretraining and alignment, joint pretrain/cooldown carefully balances text and multimodal data; long-context activation covers both text and multimodal (long videos/docs), not just text.
  - Why it matters: Kimi-VL succeeds on long video/doc benchmarks—e.g., 64.5 on LongVideoBench (Table 3)—and passes NIAH up to 128K on both text and video (Table 2).
- A compact yet effective long-thinking recipe (Figure 5; Table 4; Figure 13)
  - What’s new: Small warmup CoT SFT that teaches structured reasoning behaviors (planning, reflection, etc.) followed by RL with length penalties and difficulty-aware sampling.
  - Why it matters: Large boosts on math/science reasoning while maintaining or improving perception—e.g., `Kimi-VL-Thinking-2506` reaches 56.9 on MathVision (+20.1 over the first thinking model), 80.1 on MathVista, and 65.2 on VideoMMMU with around 3B activated parameters (Table 4).

## 5. Experimental Analysis
- Evaluation setup (Sections §4–§B; Figures 1–2; Tables 3–5)
  - Breadth: general VLM benchmarks (MMBench, MMStar, MMVet, RealWorldQA, AI2D), multi-image (BLINK), math (MathVista, MathVision), OCR/document (InfoVQA, OCRBench, MMLongBench-Doc), video (Video-MME, MLVU, LongVideoBench, VideoMMMU), and agent tasks (ScreenSpot-V2/Pro, OSWorld, WindowsAgentArena).
  - Metrics: accuracy or Pass@1 for MCQ; customized scores for OCRBench (out of 1000) and InfoVQA (ANLS-based accuracy).
  - Baselines: Efficient open-source VLMs (Qwen2.5-VL, Gemma-3, DeepSeek-VL2, Llama-3.2-11B-Inst.) and proprietary references (GPT-4o, GPT-4o-mini) where available (Table 3).
- Main results for the base `Kimi-VL-A3B` (Table 3; Figure 2)
  - General understanding
    - MMBench-EN-v1.1: 83.1, on par with GPT-4o (83.1), higher than Qwen2.5-VL-7B (82.6) and DeepSeek-VL2 (79.6).
    - AI2D: 84.9, highest among listed models (GPT-4o 84.6; Qwen2.5-VL-7B 83.9).
  - Multi-image: BLINK 57.3, better than GPT-4o-mini (53.6) and Qwen2.5-VL-7B (56.4).
  - Math
    - MathVista: 68.7, exceeding GPT-4o (63.8) and Qwen2.5-VL-7B (68.2).
    - MathVision: 21.4—behind larger dense baselines (e.g., Gemma-3-12B-IT 32.1), indicating room for deeper reasoning at this scale.
  - OCR/document
    - InfoVQA: 83.2, above GPT-4o (80.7).
    - OCRBench: 867/1000, among the top in Table 3.
    - Long docs: MMLongBench-Doc 35.1, above Qwen2.5-VL-7B (29.6), below GPT-4o (42.8).
  - Long video
    - Video-MME (w/o subs): 67.8 (Qwen2.5-VL-7B 65.1; GPT-4o 71.9).
    - MLVU MCQ: 74.2—SOTA among listed models (GPT-4o 64.6; Qwen2.5-VL-7B 70.2).
    - LongVideoBench: 64.5 (second only to GPT-4o 66.7).
  - Video perception: strong on EgoSchema (78.5 vs GPT-4o 72.2) and VSI-Bench (37.4 vs GPT-4o 34.0); slightly below GPT-4o on TOMATO (31.7 vs 37.7).
  - Agents and UI grounding
    - ScreenSpot-V2: 92.8; ScreenSpot-Pro: 34.5 (hard, high-resolution setting).
    - OSWorld Pass@1: 8.22 (GPT-4o 5.03; Qwen2.5-VL-7B 2.5).
    - WindowsAgentArena: 10.4 (GPT-4o 9.4).
  - Long-context reliability
    - NIAH (Table 2): near-perfect recall up to 64K; high at 128K for both text (87%) and video (91.7%).
- “Thinking” models (Table 4; Figure 13)
  - `Kimi-VL-Thinking` vs base (A3B): notable gains
    - MathVista: +2.6 to 71.3
    - MMMU: +4.7 to 61.7
    - MathVision: +15.4 to 36.8
  - Test-time scaling (Figure 13): Accuracy increases with longer “thinking” tokens—e.g., MathVision from 18.7% at 1k to 36.8% at 16k; MMMU from 49.2% at 1k to 61.7% at 16k; MathVista saturates around 4k (70.9%).
- Integrated “thinking” 2506 variant (Tables 4–5; §4.3)
  - Reasoning:
    - MathVision 56.9; MathVista 80.1; MMMU 64.0; MMMU-Pro 46.3; VideoMMMU 65.2—strong improvements over the first thinking model.
  - Perception/long tasks:
    - MMBench 84.4; MMStar 70.4; MMVet 78.1; RealWorldQA 70.0; OCRBench 869.
    - Long docs: MMLongBench-Doc 42.1—first open-source result matching GPT-4o reported in Table 5.
    - Agents/UI: ScreenSpot-Pro 52.8; OSWorld-G 52.5 (full set with refusals).
  - Token efficiency: ~20% shorter responses on hard reasoning (e.g., MMMU from 2.9k → 2.4k tokens; MathVision 5.8k → 4.4k) and only ~180 tokens per answer on MMBench while improving accuracy (Table 5; §4.3).
- Do the experiments support the claims?
  - Breadth and depth of benchmarks, plus NIAH and test-time scaling curves, strongly support claims of:
    - Competitive general multimodal ability (Table 3).
    - Long-context competence across modalities (Table 2; Table 3).
    - Agent/UI grounding at high resolution (Tables 3, 5; Figures 7, 10).
    - Significant reasoning gains from the thinking recipes (Table 4; Figure 13; Table 5).
  - Caveats:
    - Cross-model comparisons may be influenced by data and toolchain differences (Table 3 footnotes note tool usage for GPT-4o variants).
    - Limited ablations on architectural choices (e.g., 2D RoPE vs alternatives, pixel shuffle vs other projectors).

## 6. Limitations and Trade-offs
- Model capacity vs specialization (Conclusion §5)
  - With ~3B activated parameters, the model can lag behind larger models on the most demanding reasoning tasks (e.g., base model on MathVision) or highly specialized domains.
- Long-context at small attention width (Conclusion §5)
  - Although the context window is 128K, attention capacity is comparable to a 3B model, which may limit extraction and reasoning over extremely dense or multi-document contexts.
- Data and training cost
  - The joint recipe uses 4.4T multimodal tokens on top of a 5.2T text-only LLM pretrain (Figure 4; Table 1). This is efficient at inference, but the training pipeline itself is computationally intensive and requires sophisticated infrastructure (§2.5).
- Synthetic data and QA formatting
  - Cooldown and reasoning stages rely partly on synthetic and rejection-sampled QA/CoT. While carefully curated, synthetic distributions can introduce stylistic biases or brittleness if over-relied upon (§2.3, §3.3). The paper mitigates by keeping QA ratios low during cooldown, but residual risks remain.
- Evaluation coverage
  - Few ablations on choices such as 2D RoPE, packing strategies, expert routing details, or alternative long-context scaling methods. More granular analyses would clarify which elements drive which gains.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that an MoE VLM with small activated parameter count can deliver broad, competitive multimodal performance—including long-context and agent capabilities—when combined with native-resolution vision and a carefully staged training curriculum (Tables 3–5, Figures 3–5, 13).
  - Provides an open-source path toward efficient “thinking” VLMs with RL-enhanced CoT that scale at inference time through longer reasoning traces (Figure 13), not just larger dense parameter counts.
- Follow-up research enabled
  - Scaling variants: Larger MoE backbones and broader expert pools to push high-end reasoning while preserving activation efficiency (Conclusion §5).
  - Better long-context mechanisms: Combine 128K context with retrieval-augmented generation or memory-augmented attention to handle denser multimodal corpora.
  - Perception–reasoning fusion: Deeper study of how native-resolution visual features (e.g., fine-grained layout/UI cues) interact with CoT and RL to drive robust tool use and UI action planning.
  - Data strategy ablations: Quantify contributions of each stage (alignment, cooldown, long-context), synthetic data proportions, and video/document mixtures to guide community recipes.
- Practical applications
  - High-resolution document and UI understanding: OCR, form/table extraction, professional UI grounding at 4K+ (Tables 3, 5; Figure 9).
  - Autonomous computer-use agents: Multi-step desktop/web/mobile task completion with grounding and planning (Figure 10; OSWorld/WindowsAgentArena in Tables 3, 5).
  - Education and analysis: College-level multimodal reasoning across math, science, and engineering (MMMU, MathVista/MathVision; Tables 3–4).
  - Long media analytics: Summarization and QA for lengthy videos and multi-page PDFs (Tables 3, 5; Figure 11).

> Bottom line: Kimi-VL shows that careful system design—MoE for efficiency, native-resolution vision for fidelity, a staged multimodal curriculum for breadth, long-context activation across modalities, and a compact yet effective thinking recipe—can produce an open, compute-friendly VLM that competes well with much larger systems across perception, reasoning, and agentic tasks.
