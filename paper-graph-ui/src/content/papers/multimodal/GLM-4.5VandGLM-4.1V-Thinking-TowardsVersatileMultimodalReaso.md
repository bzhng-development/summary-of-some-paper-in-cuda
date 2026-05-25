# GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning

**ArXiv:** [2507.01006](https://arxiv.org/abs/2507.01006)

## 🎯 Pitch

GLM-4.1V-Thinking and GLM-4.5V are pioneering open-source vision-language models that achieve state-of-the-art general multimodal reasoning via a scalable, reinforcement learning-driven training pipeline. By introducing Reinforcement Learning with Curriculum Sampling (RLCS) alongside a robust, multi-domain reward system, these models consistently outperform similar-sized rivals across 42 diverse benchmarks—bridging the gap between perception and deep, cross-domain reasoning. This advance brings us closer to practical, reliable AI agents capable of solving complex real-world tasks with verified reasoning, while also setting new standards for scalable, stable multimodal RL training.

---

## 1. Executive Summary
GLM‑4.1V‑Thinking (9B) and GLM‑4.5V (106B MoE, 12B activated) are open‑source vision‑language models (VLMs) trained for general multimodal reasoning via a carefully engineered reinforcement learning (RL) pipeline. The work introduces a scalable training framework—most notably Reinforcement Learning with Curriculum Sampling (RLCS) and a robust multi‑domain reward system—that lifts performance across 42 benchmarks, setting state‑of‑the‑art results among open models of similar size and, on many tasks, matching or surpassing closed models.

## 2. Context and Motivation
- Problem addressed
  - Modern VLMs excel at perception (answering what is in an image) but often falter on complex, multi-step reasoning across diverse modalities (images, documents, GUIs, video). The open ecosystem lacks a compact, general-purpose multimodal reasoning model that consistently outperforms similar-sized non‑reasoning baselines across tasks (Introduction, p.1–2).
- Why it matters
  - Real-world use cases—from STEM problem solving and chart/document analysis to GUI agents and video understanding—require robust, verifiable reasoning, not just captioning or simple Q&A. This has both practical impact (reliable assistants and agents) and scientific significance (training procedures that scale reasoning) (Introduction, p.1–3).
- Prior approaches and gaps
  - Long-form reasoning and scalable RL boost text‑only LLMs, and early VLM works explored RL in narrow domains (Introduction, p.1–2). However:
    - Cross‑domain multimodal RL remains fragile; weak reward signals in any single domain can derail training (Section 5.2, Figure 5).
    - Open models with consistent, broad reasoning gains across modalities are scarce.
- Positioning
  - This paper unifies pretraining, supervised fine‑tuning (SFT), and large-scale RL into a reasoning‑centric pipeline for multimodal tasks. It contributes a robust reward system, difficulty‑aware sampling (RLCS), and engineering for stability/efficiency. The resulting models are open‑sourced together with domain‑specific verifiers (Abstract; Sections 3–5).

## 3. Technical Approach
The training pipeline has three layers: pre‑training, SFT for long chain‑of‑thought, and large‑scale reinforcement learning. The architecture is a ViT‑based vision encoder plus a GLM language decoder (Figure 2).

1) Architecture and tokenization (Section 2; Figure 2; Equations 1–2)
- Components:
  - Vision encoder: initialized from `AIMv2‑Huge` (a large ViT). For videos, 3D convolutions replace 2D ones to compress time by 2×—reducing compute while preserving temporal cues (p.2).
  - MLP projector: maps visual features into the language token space.
  - Language decoder: `GLM‑4‑9B‑0414` for GLM‑4.1V‑Thinking and `GLM‑4.5‑Air` for GLM‑4.5V (p.2).
- Variable resolution and extreme aspect ratios
  - The ViT uses `2D‑RoPE` (rotary positional encodings) so it can attend over very wide or tall inputs (over 200:1) and high resolutions (>4K) (p.2–3).
  - The original ViT’s learnable absolute position embeddings are retained but adapted to each input size by bicubic interpolation:
    - Normalize patch coordinates to [−1,1] (Equation (1) on p.3).
    - Sample the pre‑trained position‑embedding grid at those normalized coordinates (Equation (2) on p.3).
- Language‑side spatial grounding
  - The LLM extends RoPE to `3D‑RoPE` to encode richer spatial structures in multimodal sequences (p.3).
- Temporal grounding in video
  - Each frame is followed by a “time index token” that encodes the real timestamp as text. This makes temporal distances explicit (p.3).

2) Data construction (Section 3.1)
- Image‑caption data (10B+ raw pairs → filtered + “recaptioned”):
  - Filtering: resolution/length rules; CLIP score > 0.3; concept‑balanced resampling to fix long‑tail issues; noisy captions denoised by a factual recaptioner (Figure 4; p.4–5).
- Interleaved image‑text corpora (web pages, STEM books):
  - Web pipeline: from MINT, MMC4, OmniCorpus; remove ads/QRs; retain image‑text consistent pages; prioritize “high‑knowledge‑density” content using a trained classifier (p.5–6).
  - Books: 100M digitized books, filtered to STEM; deep PDF parsing extracts structured interleaved content (p.6).
- OCR at scale (220M images):
  - Synthetic documents (render text with varied fonts/backgrounds).
  - Natural scene text via Paddle‑OCR detections.
  - Academic PDFs via a Nougat-like pipeline (LaTeX→HTML→markup→rasterized pages) (p.6–7).
- Grounding data:
  - Natural images: LAION‑115M captions parsed; GLIPv2 predicts noun‑phrase boxes; keep images with ≥2 valid boxes → 40M final pairs (p.7).
  - GUIs: crawl pages with Playwright, collect visible DOM + rendered boxes; generate 140M QA pairs for referring expressions and comprehension (p.7).
- Video data:
  - Diverse academic/web/proprietary sources; human‑in‑the‑loop annotations for actions and cinematography; multimodal dedup across video and text embeddings (p.7–8).
- Instruction tuning corpus (50M):
  - Covers STEM, GUI agents, long documents, code‑related tasks; taxonomy‑guided sampling; synthetic augmentation; de‑contamination against public benchmarks (p.8).

3) Pre‑training and long‑context continual training (Section 3.2)
- Stage 1: multimodal pre‑training
  - Seq length 8,192; global batch 1,536; 120k steps; packed sequences for efficiency (p.8–9).
  - Parallelism: GLM‑4.1V uses tensor‑parallel 2; GLM‑4.5V (MoE) uses expert‑parallel 8 + pipeline‑parallel 4 with a loss‑free router, bias update 1e‑3, balance loss 1e‑4 (p.8).
  - Result: higher pass@k on MathVista free‑response subset vs. a strong 9B base (Figure 3, p.4), indicating a stronger “upper bound” for later RL.
- Stage 2: long‑context continual training
  - Adds video and >8k‑token interleaved data; seq length raised to 32,768; context‑parallel 4; 10k more steps at the same global batch (p.9).

4) Supervised fine‑tuning for long chain‑of‑thought (Section 4)
- Purpose: align the model to produce standardized, verifiable reasoning traces to bootstrap RL stability—not to add new knowledge (p.7).
- Output schema:
  - Responses follow `<think>…</think> <answer>…</answer>`; for verifiable tasks, the final answer is boxed with special tokens `<|begin_of_box|>…<|end_of_box|>` and exactly one box is allowed (p.7).
  - GLM‑4.5V supports a “non‑thinking mode”: adding `/nothink` to the prompt trains the model to emit an empty think segment and respond directly (p.8).
- Training: full‑parameter SFT at seq length 32,768; global batch 32; includes high‑quality text‑only long‑form to preserve language skills (p.8).

5) Reinforcement learning at scale (Section 5)
- RL modes:
  - `RLVR` (Reinforcement Learning with Verifiable Rewards): use reference answers and programmatic verifiers to score outputs.
  - `RLHF` (with reward models) for tasks that cannot easily be verified exactly.
- Reward system (Section 5.2; Table 1 on p.11):
  - Final answers must be extracted reliably: during RLVR, the model is required to put the final answer inside the special box tokens; the verifier only compares the boxed span to the ground truth (p.10–11).
  - Domain‑specific verifiers:
    - Math/Physics: numeric equivalence via SymPy with tolerances; unit‑aware LLM checks when needed.
    - OCR: edit‑distance‑based continuous reward.
    - Charts: numeric tolerance; textual exact or semantic match.
    - Grounding/GUI: IoU thresholds for boxes; action+IoU for GUI action prediction (Table 1).
  - Format/style rewards: penalize misuse of box tokens for non‑verifiable prompts; discourage mixed‑language or repetitive thought patterns (p.11).
  - Critical observation: a single weak verifier can derail multi‑domain RL—Figure 5 shows reward hacking and collapse when a multi‑image QA verifier is imprecise, even though the STEM verifier is strong (p.9–10).
- RLCS: Reinforcement Learning with Curriculum Sampling (Section 5.3)
  - Motivation: as the model improves, many rollout samples become trivial or intractable; both give no useful gradient under GRPO when KL/entropy terms are removed (p.12).
  - Mechanism:
    - Offline difficulty labels: pass@k from several strong models + human labels partition data into tiers (easy→hard).
    - Online difficulty updates: during training, map each rollout to a difficulty tier based on observed success; maintain running distributions (p.12).
    - Adaptive sampling: down‑weight too‑easy and too‑hard; over‑sample the mid‑range where learning signal is strongest (p.12–13).
  - Dynamic sampling expansion via ratio EMA: if many all‑correct or all‑incorrect batches occur, pre‑oversample by an expansion ratio computed as `1/(1 – not_valid_sample_rate)` and smoothed with EMA; then select a subset with balanced difficulty for training. This stabilizes the “effective batch size” for GRPO without KL/entropy losses (p.12–13).
- Other RL practices that improved effectiveness/stability (Section 5.3):
  - Force answering: if thinking grows too long and nears truncation, insert `</think>` to force a final answer, enabling fair rewards and encouraging anytime answers (p.12).
  - Remove KL and entropy losses: keeping them reduced capability or caused garbling; training was more stable without them (p.12–13).
  - Sampling: set `top‑p=1` during rollouts to avoid degeneration in later iterations (p.13).
  - Optimization: larger batch sizes; higher upper clip bound on importance ratios (“clip‑higher”) aid off-policy performance (p.12).
  - Loss reduction: compute per‑sample loss for stability (p.13).
- RL infrastructure (Section 5.4)
  - Load‑balance sequences across data‑parallel ranks; train with packed sequences and gradient accumulation; repack samples to minimize micro‑steps; precompute oversampling quota for parallel rollouts (p.13–14).

## 4. Key Insights and Innovations
- RLCS: difficulty‑aware online curriculum for RL
  - What is new: combines offline/online difficulty grading with adaptive sampling targeted at mid‑range difficulty; integrates a ratio‑EMA oversampling strategy tailored for GRPO without KL/entropy (Section 5.3; p.12–13).
  - Why it matters: greatly increases rollout efficiency—the dominant cost in RL—and raises the performance ceiling by keeping batches informative.
- Multi‑domain, hack‑resistant reward system
  - What is new: a unified but domain‑specialized set of verifiers that enforce boxed final answers, numeric/unit equivalence, IoU‑based grounding, and style/format compliance; unit tests per domain (Table 1; p.10–11).
  - Why it matters: Figure 5 demonstrates that a single weak verifier causes cross‑domain collapse; robust verifiers are essential for stable multi‑skill RL.
- Architecture for high‑fidelity multimodal inputs
  - What is new: ViT with `2D‑RoPE` for arbitrary aspect ratios and bicubic adaptation of absolute embeddings (Equations (1)–(2)), plus `3D‑RoPE` on the LLM side and timestamp tokens between video frames (Section 2, p.2–3).
  - Why it matters: supports native‑resolution images (even >4K) and explicit temporal grounding, which underpin chart/document/video gains.
- Dual‑mode “thinking / non‑thinking” inference
  - What is new: GLM‑4.5V natively supports long chain‑of‑thought or concise direct responses, controllable via a `/nothink` token (Section 4.2, p.8).
  - Why it matters: enables flexible trade‑offs between accuracy and latency—use long reasoning when needed, short mode for throughput‑sensitive tasks.
- Data engineering at scale for reasoning
  - What is new: concept‑balanced, recaptioned web data; high‑knowledge‑density filtering for interleaved corpora; 220M‑image OCR; 140M GUI grounding pairs; human‑in‑the‑loop video labels (Section 3.1).
  - Why it matters: yields a strong base model (Figure 3) that sets a higher “upper bound” for downstream RL gains.

## 5. Experimental Analysis
- Evaluation protocol (Section 6.1)
  - 42 public benchmarks across 8 categories: General VQA, STEM, OCR/Chart/Doc, Visual Grounding, Spatial Reasoning, GUI Agents, Coding, and Video Understanding.
  - Inference: vLLM for most tasks, SGLang for video; max output 8,192 tokens; images capped at 6,144 tokens; video up to 48,000 tokens (p.15–16).
  - Answer extraction: parse the span between `<|begin_of_box|>` and `<|end_of_box|>` as the final answer; GPT‑4o (2024‑11‑20) is used only where a language judge is necessary (p.16).
- Main results (Table 2, p.15)
  - State‑of‑the‑art among open models of similar size; often competitive with or better than larger/closed systems. Examples:
    - STEM and math reasoning:
      > MMMU (Val): `GLM‑4.5V‑Thinking 75.4` vs `Step‑3 74.2` vs `Qwen2.5‑VL‑72B 70.2`  
      > MMMU‑Pro: `65.2` vs `58.6` (Step‑3) vs `51.1` (Qwen72B)
      > MathVista: `84.6`, WeMath: `68.8`
    - Charts/documents:
      > ChartQAPro: `64.0` vs `56.4` (Step‑3) vs `46.7` (Qwen72B)  
      > ChartMuseum: `55.3` vs `40.0` (Step‑3) vs `39.6` (Qwen72B)  
      > MMLongBench‑Doc: `44.7` vs `31.8` (Step‑3) vs `35.2` (Qwen72B)
    - GUI agents and coding:
      > WebVoyager (Some): `84.4` vs `40.4` (Qwen72B)  
      > OSWorld (100‑step budget): `35.8` vs `8.8` (Qwen72B)  
      > Design2Code: `82.2` vs `34.1` (Step‑3) vs `41.9` (Qwen72B)  
      > Flame‑React‑Eval: `82.5` vs `63.8` (Step‑3) vs `46.3` (Qwen72B)
    - Video understanding:
      > VideoMMMU: `72.4` vs `60.2` (Qwen72B), MMVU: `68.7`  
      > VideoMME (w/ subs): `80.7`
    - Visual grounding:
      > RefCOCO‑avg (val): `91.3` (close to `90.3` for Qwen72B)
  - Small model competitiveness:
    > GLM‑4.1V‑9B‑Thinking outperforms Qwen2.5‑VL‑72B on 29/42 benchmarks (Abstract; Table 2 highlights: e.g., MMMU‑Pro 57.1 vs 51.1; ChartMuseum 48.8 vs 39.6; MUIRBENCH 74.7 vs 62.9).
  - Thinking vs non‑thinking trade‑offs:
    > OCRBench favors non‑thinking `87.2` over thinking `86.5`, while reasoning‑heavy tasks benefit from thinking mode (Table 2).
- RL effectiveness and cross‑domain transfer
  - RL Gains: Figure 1B shows reinforcement learning improves performance by up to +10.6% on GLM‑4.5V.
  - Cross‑domain generalization (Figure 6, p.17–18):
    > Training on a single domain (e.g., STEM) improves other domains (grounding, GUI, general VQA). The “mix‑all” setting further improves STEM, OCR/Chart, and general VQA, though not grounding or GUI in this configuration.
- Robustness and failure analysis
  - Verifier quality is pivotal: Figure 5 shows reward hacking (e.g., answering “a correct number between 0 and 10”) can inflate rewards without real accuracy; this stalled STEM progress and degraded multimodal benchmarks when a multi‑image QA verifier was weak (Section 5.2).
  - Stability practices: removing KL/entropy, forcing answers, top‑p=1, per‑sample loss, balanced rollouts (Section 5.3) mitigate collapse.
- Overall assessment
  - The experiments are broad (42 benchmarks), methodologically explicit (boxed answer extraction; shared toolchain; minimum 95% success rate per benchmark), and include diagnostic analyses (Figure 5, Figure 6). Together they convincingly support claims of cross‑domain RL gains and strong absolute performance.

## 6. Limitations and Trade-offs
- Outcome‑only rewards can reinforce flawed reasoning (Section 7)
  - Current verifiers typically score only final answers. The models sometimes reach correct answers via incorrect or hallucinated steps, and RL might reinforce those paths when the outcome is correct (p.18).
- Training sensitivity and stability (Section 7)
  - Early setups saw large variations in reasoning depth/style; although mitigated with better rewards and cold‑start data, large‑scale RL remains sensitive to configuration (p.18).
- Dependence on precise verifiers (Section 5.2; Figure 5)
  - A single weak domain verifier can destabilize multi‑domain training. Building and maintaining high‑quality verifiers per domain is nontrivial.
- Compute and data requirements
  - Pre‑training uses very large datasets (e.g., 10B+ image‑text pairs to start; 220M OCR; 140M GUI QA; Section 3.1), long contexts (32k tokens), and large batch sizes (global 1,536), plus MoE routing/balancing—demanding infrastructure.
- Coverage gaps
  - Mixed results in some domains: “mix‑all” RL did not improve grounding or GUI in Figure 6, suggesting these may need specialized curricula or rewards. Perception failures on cluttered/ambiguous images can undermine reasoning (Section 7).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that large‑scale, multi‑domain RL with strong verifiers and an adaptive curriculum can produce broadly capable multimodal reasoners. The open sourcing of models, code, and domain verifiers (Abstract; Section 1) lowers barriers for reproducible progress.
- Research directions
  - Process‑level rewards: design verifiers that check intermediate reasoning (not just final answers) to prevent “shortcut” chains (Section 7).
  - Stronger anti‑hacking verifiers: broaden unit‑, format‑, and semantics‑aware checks; expand unit tests per domain (Section 5.2).
  - Curriculum design for hard domains: grounding and GUI may benefit from specialized RLCS schedules or multi‑task weighting strategies (Figure 6).
  - Multimodal→text transfer: study whether visual reasoning (e.g., reading code from images) measurably improves purely text tasks (Section 7).
  - Evolving benchmarks: build diagnostic tasks that detect hallucinated reasoning, shortcutting, and domain‑transfer limits (Section 7).
- Practical applications
  - High‑leverage scenarios include: chart/document analytics, STEM tutoring, UI automation and web agents, design‑to‑code workflows, and video understanding for surveillance or media tools. The “thinking/non‑thinking” switch helps tailor latency vs accuracy to task needs (Section 4.2; Table 2 GUI/Coding/Video results).

> Code, models, and reward components are released at https://github.com/zai-org/GLM-V (Abstract).

In summary, the paper’s substantive advances are methodological (RLCS + verifiers + engineering for stability/throughput) and system‑level (open, dual‑mode VLMs with strong cross‑domain reasoning). The results across 42 benchmarks (Table 2) and diagnostic studies (Figures 1, 5, 6) provide a detailed, credible case for the approach and outline clear next steps to evaluate and train reasoning beyond outcome‑only rewards.
