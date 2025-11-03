# Gemini: A Family of Highly Capable Multimodal Models

**ArXiv:** [2312.11805](https://arxiv.org/abs/2312.11805)

## 🎯 Pitch

This paper introduces Gemini, a new family of large-scale models natively trained to understand and reason across text, images, audio, and video, while also generating both text and images. By advancing state-of-the-art performance on a wide array of academic and real-world benchmarks—including being the first to exceed human-expert scores on MMLU—Gemini sets a new standard for generalist AI and delivers these capabilities from massive data-center deployments (Ultra, Pro) all the way to on-device efficiency (Nano). This leap in multimodal and cross-modal reasoning unlocks seamless applications in education, analytics, accessibility, and beyond, while incorporating rigorous evaluation and responsible deployment frameworks.

---

## 1. Executive Summary (2–3 sentences)
This paper presents Gemini, a family of “natively multimodal” large models that process and reason over interleaved text, images, audio, and video, and can also generate both text and images. The core significance is twofold: state‑of‑the‑art performance across 30 of 32 evaluated benchmarks (with human‑expert MMLU performance) and a practical pathway from data center models (Ultra, Pro) to efficient on‑device use (Nano) with a responsible deployment framework (Sections 1–2, Tables 1–2).

## 2. Context and Motivation
- Problem/gap
  - Prior foundation models largely excel in one modality or require external adapters (e.g., OCR for text-in-image) and often cannot natively combine modalities or output images. This limits cross‑modal reasoning such as “read a chart, tie it to text, compute an answer, and format the result” (Sections 1–2; Figure 1, Figure 2).
  - Existing evaluation practices face contamination risks and uneven cross‑modal coverage, making it hard to assess generalist capabilities reliably (Section 5.1.1; HellaSwag discussion).

- Why it matters
  - Real‑world tasks frequently mix modalities: documents with charts, screenshots, and audio/video content. A single system that understands all of them reduces orchestration overhead and unlocks workflows in education, accessibility, analytics, and software engineering (Figures 1, 5; Table 13; Section 5.2).
  - On-device inference enables private, low‑latency user experiences (Section 5.1.3; Table 1).

- Prior approaches and shortfalls
  - Vision‑language models like Flamingo/PaLI/CoCa improved image–text tasks but were not trained “from the beginning” to be multimodal across image, audio, video and to emit images (Section 2).
  - Text‑first LLMs extended to vision (e.g., via OCR) miss fine details or layout information and rely on brittle tool chains (Table 7 highlights “pixel only” results without any external OCR).

- Positioning
  - Gemini is trained jointly across modalities, supports interleaved inputs, produces text and images via discrete image tokens, and is evaluated end‑to‑end across text, image, audio, and video, aiming to be both a generalist model and a competitive specialist in each domain (Sections 2, 5; Figure 2; Table 7).

## 3. Technical Approach
- Model family and sizes
  - `Ultra`: flagship, highest capability for complex reasoning and multimodal tasks.
  - `Pro`: performance/latency‑optimized for broad deployment.
  - `Nano‑1` (1.8B) and `Nano‑2` (3.25B): distilled, 4‑bit quantized models for on‑device use (Table 1; Section 5.1.3).

- Core architecture
  - Decoder‑only Transformer with efficient attention (e.g., multi‑query attention) and a 32k‑token context window (Section 2).
  - Native multimodality: one sequence can interleave text, image frames, audio features, and video frames; outputs can interleave text and images using discrete image tokens (Figure 2; Section 2).
  - Video is encoded as a frame sequence at variable resolution to trade compute for detail; audio ingested as 16 kHz features from Universal Speech Model (`USM`) to preserve non‑textual cues (Section 2).

- Training infrastructure and reliability (how training at scale is made practical)
  - Large‑scale training on TPUv4/v5e across multiple data centers using JAX + Pathways; model/data parallelism orchestrated by GSPMD and XLA (Section 3).
  - Reliability innovations:
    - In‑memory redundant model state with rapid replica recovery instead of periodic persistent checkpoints, increasing training “goodput” (time doing useful new steps) from 85% (PaLM/PaLM‑2 scale) to 97% (Section 3).
    - Silent Data Corruption (SDC) detection via deterministic replay and proactive scanners with hot standbys to identify faulty hardware quickly (Section 3).
    - Optical reconfiguration of TPU SuperPods into 3D tori for flexible scaling and maintenance (Section 3).

- Data pipeline
  - Multimodal, multilingual mixture: web documents, books, code, images, audio, video; SentencePiece tokenizer trained on the full corpus for better non‑Latin coverage and speed (Section 4).
  - Quality and safety filtering; decontamination against evaluation data; staged mixture schedules that up‑weight domain‑relevant data toward the end of training (Section 4).

- Post‑training into two productized variants (Section 6)
  - `Gemini Apps` (for conversational services like Gemini and Gemini Advanced) and `Gemini API` (for developers in AI Studio and Vertex AI). Both use a multi‑stage “data flywheel”:
    1) Curate prompts (single/multi‑turn) representative of real use.
    2) Supervised Fine‑Tuning (`SFT`) on high‑quality demonstrations.
    3) Reward Model (`RM`) training on human preference data.
    4) Reinforcement Learning from Human Feedback (`RLHF`) to align outputs with preferences (Figure 7; Section 6.3).
  - Capability‑specific post‑training recipes for instruction following, tool use (code‑as‑tool loops; Figure 8), multilinguality (translationability filtering + human validation), multimodal vision (SFT on curated image‑text), coding (human + synthetic supervision), and factuality (closed‑book accuracy, attribution to provided sources, and “hedging” when unanswerable; Section 5.1.6; Table 6).

- “Uncertainty‑routed chain‑of‑thought”
  - For multiple‑choice reasoning (e.g., MMLU), the model samples `k` reasoning traces; if a consensus exceeds a validation‑tuned threshold, it picks the majority; otherwise it falls back to greedy decoding—improving accuracy beyond plain CoT or greedy alone (Appendix 10.2; Figure 9).

- Evaluation setup (how the model is exercised)
  - 50+ benchmarks organized into capability clusters: Factuality, Long‑Context, Math/Science, Reasoning, Summarization, Multilinguality (Appendix 10.3).
  - Vision: strict “pixel only” inference—no external OCR, zero‑shot or few‑shot instructions; video evaluated on 16 equally spaced frames per clip (Sections 5.2.1–5.2.2; Table 7; Table 10).

## 4. Key Insights and Innovations
- Native, end‑to‑end multimodality (fundamental)
  - The model ingests and reasons over interleaved text/image/audio/video and also generates images as tokens, enabling tasks like “read a chart + produce a table + explain errors in handwritten math + emit illustrative images” within a single forward pass (Sections 2, 5.2; Figures 1–2, 5–6; Table 13). This removes fragile external adapters (e.g., OCR), shown by strong “pixel only” scores in Table 7.

- Training‑at‑scale reliability and efficiency (enabler for capability scaling)
  - Checkpoint‑free replica recovery and SDC detection increase goodput to 97% at unprecedented TPU scale, allowing longer runs and larger models without proportional downtime (Section 3). This is an operational innovation that materially enables Ultra‑scale training.

- “Uncertainty‑routed” CoT for exam‑style tasks (incremental but impactful)
  - Majority‑vote gating over sampled chains boosts MMLU to 90.04% (Appendix 10.2; Figure 9; Table 2), surpassing a reported human‑expert threshold (89.8%) and prior SOTA.

- Tool‑use as code blocks inside the model loop (incremental but practical)
  - Treating tool calls as generated code that executes and returns results to the context allows the model to compose multiple tools per turn and reason over outputs. This delivers large gains in factual retrieval and math (Table 15), and powers `Gemini Extensions` for real products (Section 6.5.2; Figure 8).

- Responsibility stack integrated into the modeling pipeline (fundamental for deployment)
  - Factuality triad (closed‑book accuracy, source attribution, calibrated hedging), adversarial multimodal safety datasets, and multilayer red‑teaming—plus evaluative assurances for “dangerous capabilities”—form a repeatable deployment process (Sections 5.1.6, 7.1–7.4; Table 6).

## 5. Experimental Analysis
- Evaluation methodology
  - Text (Table 2; Sections 5.1.1–5.1.6): MMLU, GSM8K, MATH, BIG‑Bench‑Hard, HumanEval, Natural2Code (held‑out Python code gen), DROP, HellaSwag (with decontamination and 10‑shot reporting), WMT23 (BLEURT). Long‑context tested via synthetic retrieval and NLL vs position across 32k tokens (Figure 4).
  - Vision (Table 7; Section 5.2.1): MMMU (multi‑discipline visual QA), TextVQA, DocVQA, ChartQA, InfographicVQA, MathVista, AI2D, VQAv2; multilingual captioning on XM‑3600 subset (Table 9). All “pixel only,” zero‑shot, greedy, unless noted.
  - Video (Table 10; Section 5.2.2): VATEX (EN, ZH captioning, 4‑shot), YouCook2 (captioning, 4‑shot), and zero‑shot QA datasets (NextQA, ActivityNet‑QA, Perception Test MCQA); 16 frames sampled per clip.
  - Audio (Table 11; Section 5.2.4): ASR on YouTube EN, MLS EN, FLEURS 62 langs, VoxPopuli 14 langs (WER↓); Speech translation on CoVoST2 21 langs (BLEU↑). Note: FLEURS used in training; a no‑FLEURS model still outperforms Whisper (WER 15.8 vs Whisper v3; Section 5.2.4).
  - Tool use (Table 15) and post‑training changes (Table 17 for pre‑ vs post‑trained vision; Table 14 instruction following; Table 6 factuality).

- Headline quantitative results (selected)
  - Text reasoning
    > “`Gemini Ultra` reaches 90.04% on MMLU (CoT@32 with uncertainty routing), 94.4% on GSM8K (Maj1@32), 53.2% on MATH (4‑shot), and 83.6% on BIG‑Bench‑Hard (3‑shot)” (Table 2, Appendix 10.2).
    - HumanEval Pass@1 74.4% (0‑shot, post‑trained API model), and 74.9% on the held‑out Natural2Code benchmark (Table 2).
  - Multilinguality
    > “On WMT23 averaged across directions, BLEURT 74.4 for Ultra vs 73.8 GPT‑4, 72.7 PaLM2‑L” (Table 4).  
    > “MGSM (8‑shot) 79.0 for Ultra vs 74.7 PaLM2‑L” (Table 5).
  - Long‑context
    > “98% retrieval accuracy for key‑value at the end of a 32k context; NLL decreases steadily across positions up to 32k” (Section 5.1.5; Figure 4).
  - Vision (all zero‑shot unless stated)
    > “`MMMU` pass@1 59.4% (Maj1@32 62.4%) for Ultra vs 56.8% GPT‑4V” (Table 7, Table 8).  
    > Strong OCR‑heavy tasks “pixel‑only”: TextVQA 82.3%, DocVQA 90.9% (Table 7).  
    > MathVista 53.0% (test‑mini), AI2D 79.5%, VQAv2 77.8% (Table 7).  
    > Multilingual captioning (XM‑3600 subset): higher CIDEr than PaLI‑X across seven languages (Table 9).
  - Video
    > “VATEX EN captioning CIDEr 62.7 (4‑shot), YouCook2 135.4 (4‑shot); NextQA WUPS 29.9 (0‑shot); ActivityNet‑QA 52.2% (0‑shot); Perception Test MCQA 54.7% (0‑shot)” (Table 10).
  - Audio
    > “ASR WER: YouTube EN 4.9% (Pro), MLS EN 4.8%, FLEURS 7.6% (62 langs), VoxPopuli 9.1% (14 langs); CoVoST2 BLEU 40.1” (Table 11). Qualitative examples show better rare‑word/proper‑noun handling than USM (Table 12).
  - Tool use and instruction following
    > “With tools vs without: GSM8K 80.1% vs 69.7%; MATH 41.8% vs 30.7%; NQ 68.0% vs 59.0%; RealTimeQA 70.8% vs 39.2%” (Table 15).  
    > Instruction following on complex prompts: per‑instruction accuracy 87.4% and full‑response accuracy 54.1% for Gemini Advanced (Ultra) (Table 14).
  - Factuality/Attribution/Hedging
    > “Inaccuracy rate halved (6.7% → 3.8%), attribution AIS up (40.2% → 60.0%), hedging accuracy 69.3% from 0% after factuality‑focused post‑training on Pro” (Table 6).
  - On‑device Nano models
    > “Despite 1.8B/3.25B size, Nano‑2 achieves 0.83× of Pro on NQ‑Retrieved and 0.78× on MMLU; Nano‑1 achieves 0.69× and 0.64× respectively” (Table 3; Figure 3).

- Ablations and robustness checks
  - HellaSwag data‑sensitivity: fine‑tuning on websites corresponding to HellaSwag training set (not used in pretraining) moves 1‑shot validation accuracy to 96.0% (Ultra) and 89.6% (Pro), showing metric sensitivity to pretraining mixtures (Section 5.1.1). To mitigate such effects, the paper reports decontaminated 10‑shot numbers in Table 2.
  - Pre‑ vs post‑trained vision: SFT improves several image benchmarks meaningfully (e.g., +3.3% on VQAv2, +2.9% AI2D, +2.4% InfographicVQA), aligning outputs to task references while the base model is already strong (Table 17).
  - Long‑context synthetic retrieval: explicit 32k stress test (Section 5.1.5).

- Do the experiments support the claims?
  - The breadth (50+ tasks), transparency on contamination risks, and cross‑modal “pixel‑only” protocol together substantiate claims of multimodal competence and state‑of‑the‑art performance in many areas. Caveats remain where comparisons are not apples‑to‑apples (e.g., some baselines are fine‑tuned vs Gemini zero‑shot; section notes this in Table 7), or evaluations depend on model sampling strategies (Appendix 10.2).

## 6. Limitations and Trade‑offs
- Data and evaluation sensitivity
  - Benchmarks like HellaSwag are highly sensitive to training data composition; despite decontamination efforts, the field still lacks uniformly leakage‑free, robust benchmarks (Section 5.1.1).
  - Some reported baselines are via external APIs at a specific time (Table 2 notes “self‑collected via API in Nov 2023”), which can drift and complicate strict comparability.

- Compute and reproducibility
  - Compute requirements and parameter counts for Ultra/Pro are not disclosed in the model card (“Compute Requirements: Not reported”; Appendix 10.1), which limits independent reproduction and precise scaling analyses. The uncertainty‑routed CoT uses many samples (e.g., 32), increasing inference cost (Appendix 10.2).

- Modal coverage specifics
  - Video evaluation samples only 16 frames per clip (Section 5.2.2), potentially missing fine temporal events.
  - Audio: Ultra is not yet evaluated; FLEURS is in training data, making those ASR numbers less comparable (Section 5.2.4), though a control without FLEURS still beats Whisper.

- Safety and reliability
  - Despite improvements, safety assessments flag areas “with particular room for improvement,” e.g., medical advice and harassment in text‑to‑text (Section 7.4.1.1). Image/video tests show the model can make ungrounded inferences about people; no consistent bias pattern observed, but the behavior remains a risk (Section 7.4.1.2).

- Scope limits
  - Context length is capped at 32k tokens; truly long‑form video or multi‑document corpora may exceed this.
  - Tool‑use results are shown for specific tools and tasks; broader compositional tool chains or security‑hardened tool execution are not exhaustively studied (Section 6.5.2).

## 7. Implications and Future Directions
- Field impact
  - A practical template for “one model, many modalities” that competes with specialized systems while simplifying pipelines (Figures 5–6; Table 7). The integrated responsibility stack—factuality triad, adversarial multimodal safety sets, red teaming—sets a deployment bar for generalist agents (Sections 5.1.6, 7.1–7.4).

- Enabled research
  - Agents that combine Gemini‑class reasoning with tools and search (AlphaCode 2 demonstrates top‑15% Codeforces performance by coupling Gemini Pro with search, clustering, and reranking; Section 5.1.7). Future work can extend this paradigm to planning, retrieval‑augmented generation across modalities, and robust tool‑use security.
  - Evaluation science: the paper’s contamination analysis and “pixel‑only” protocols motivate new, leakage‑resistant multimodal benchmarks and standardized sampling/reporting practices.

- Applications and downstream use
  - Education and accessibility: grading handwritten work, captioning images/videos, explaining charts/diagrams, and audio‑visual tutoring (Figures 1, 10, 13, 23; Table 13).
  - Enterprise productivity: long‑context summarization/search over mixed‑media documents; robust chart/table understanding (Figure 10; Section 5.1.5).
  - Software engineering and data tasks: strong coding and math performance, boosted further by tools (Table 2, Table 15); image generation to support content creation (Figure 6).
  - On‑device experiences: summarization, retrieval QA, and reasoning at the edge with `Nano` models (Table 3; Figure 3).

---

> Selected results to remember (with sources):
> - “MMLU 90.04% with uncertainty‑routed CoT@32” (Table 2; Appendix 10.2, Figure 9).
> - “MMMU pass@1 59.4% (Maj1@32 62.4%) ‘pixel‑only’ zero‑shot; GPT‑4V 56.8%” (Table 7, Table 8).
> - “WMT23 average BLEURT 74.4 (Ultra) vs 73.8 (GPT‑4)” (Table 4).
> - “Tool use boosts NQ from 59.0% → 68.0% and RealTimeQA from 39.2% → 70.8%” (Table 15).
> - “Factuality inaccuracy halved (6.7% → 3.8%); attribution up (40.2% → 60.0%); hedging to 69.3%” (Table 6).
> - “Goodput 97% at Ultra scale via in‑memory recovery and SDC scanning” (Section 3).

Definitions of less-common terms used above:
- `BLEURT`: a learned MT quality metric that correlates with human judgments better than BLEU (Table 4).
- `AIS`: “Attributable to Identified Sources,” a human‑rated measure of whether generated text is faithful to provided sources (Section 5.1.6; Table 6).
- `WER`: Word Error Rate, standard ASR metric (Table 11).
- `CIDEr`: Image/video captioning metric measuring consensus with reference captions (Table 10).
- `Goodput`: fraction of total training time spent doing useful new steps, not recovery/overhead (Section 3).
