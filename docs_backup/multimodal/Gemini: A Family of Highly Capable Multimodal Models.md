# Gemini: A Family of Highly Capable Multimodal Models

**ArXiv:** [2312.11805](https://arxiv.org/abs/2312.11805)
**Authors:** Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean‑Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy Lillicrap, Angeliki Lazaridou, Orhan Firat, James Molloy, Michael Isard, Paul R. Barham, Tom Hennigan, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, Ed Chi, Heng‑Tze Cheng, …plus many others (total≈900+ authors)
**Institutions:** Google DeepMind, Google AI (implied by Gemini Team)

## 🎯 Pitch

Gemini reshapes the AI landscape with the first natively multimodal large models able to understand and generate across text, image, audio, and video, demonstrating extraordinary performance by surpassing human-expert levels on the MMLU exam suite. Unifying these modalities in one scalable model facilitates advanced reasoning and accessibility across diverse compute environments, heralding transformative applications in education, coding, and enterprise solutions.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces Gemini, a family of natively multimodal large models (`Nano`, `Pro`, `Ultra`) trained jointly on text, images, audio, and video that can both understand and generate across modalities. The top model, `Ultra`, advances state of the art on 30 of 32 benchmarks, including the first human‑expert performance on the MMLU exam suite (90.04%), while the system design includes scalable training infrastructure, post‑training for safety/factuality, and an on‑device `Nano` line for edge use (Sections 1, 2, 3, 5; Table 2).

## 2. Context and Motivation
- Problem/gap addressed
  - Most prior systems either focus on a single modality (language only, image only) or bolt a vision encoder onto a language model, limiting true cross‑modal reasoning and image generation. Gemini targets a single model that is strong both as a generalist across modalities and as a specialist within each modality (Section 1; Figure 2).
  - There is also a need for models that can be deployed across a spectrum of compute—from data centers to mobile—while meeting alignment, factuality, and safety expectations (Sections 1, 6, 7).
- Why it matters
  - Real‑world tasks (education, coding agents, document/image understanding, translation) are inherently multimodal and multi‑step. Unifying these capabilities promises more fluent agents and broader accessibility, including on-device operation (Sections 1, 5.2, 5.1.3).
- Prior approaches and shortcomings
  - Visual–language systems like Flamingo, PaLI, and CoCa provide strong perception but typically do not natively generate images and/or require external OCR; many LLMs do not ingest raw audio/video sequences. Gemini differs by being multimodal “from the beginning,” handling interleaved inputs and producing images via discrete image tokens, with “pixel‑only” OCR‑free understanding (Section 2; Table 7 notes “pixel only”).
- Positioning
  - Gemini unifies modalities in one decoder‑only Transformer with long context (32k), scales training efficiently across TPU superpods, and emphasizes rigorous post‑training for instruction following, tool use, safety, and multilinguality. It is released as two post‑trained variants tuned for different users: `Gemini Apps` (chat‑centric) and `Gemini API` (developer‑centric) (Sections 2, 3, 6).

## 3. Technical Approach
Step-by-step overview of how Gemini is built and used.

- Core model and multimodality (Section 2)
  - Architecture: decoder‑only Transformer with efficient attention (multi‑query attention) and 32k context support.
  - Native multimodality:
    - Visual encoding inspired by Flamingo/CoCa/PaLI but trained end‑to‑end so the model “can produce text and image outputs,” using discrete image tokens to generate images (Figure 2).
    - Video is treated as a sequence of frames placed into the long context; resolution is adaptive to spend more compute where needed.
    - Audio is ingested as features from the Universal Speech Model at 16kHz, preserving paralinguistic cues often lost in text transcripts.
  - Consequence: a single sequence can interleave text, image frames, and audio chunks—enabling cross‑modal reasoning (Table 13; Figures 5, 6).

- Training at scale (Section 3)
  - Hardware: TPUv4 superpods (4096 chips each) and TPUv5e; synchronous training with model parallelism inside superpods and data parallelism across data centers.
  - Deterministic infrastructure and resilience:
    - Redundant in‑memory model state for fast failure recovery (no heavy persistent checkpoint reload), raising training “goodput” from 85% (PaLM‑era) to 97% (footnote 2; Section 3).
    - Silent Data Corruption (SDC) detection via deterministic replay, proactive scanners, and hot standbys.
  - Result: stable long‑running jobs at unprecedented scale.

- Data and tokenizer (Section 4)
  - Pre‑training data: large, multimodal, multilingual mixture of web documents, books, code, images, audio, and video.
  - Tokenizer: SentencePiece trained on a large corpus sample to improve coverage, especially for non‑Latin scripts.
  - Mixture scheduling: staged training increases domain‑relevant data toward the end; strong emphasis on quality filters, safety filtering, and evaluation set decontamination (e.g., not reporting LAMBADA due to leakage; Section 5.1.1).

- Post‑training “how” (Section 6; Figure 7)
  - Data flywheel:
    1) Curate prompts representative of real use (single and multi‑turn).
    2) Supervised fine‑tuning (SFT) on demonstrations.
    3) Train reward models (RMs) on human preference data (relative rankings and response‑level labels).
    4) RLHF loops that continually improve both policy and RM.
  - Variants:
    - `Gemini Apps` models optimize for conversational use with tool integrations (Google Flights, Workspace, etc.).
    - `Gemini API` models target developer workflows, tool‑use scaffolds, and enterprise settings.

- Instruction following and factuality mechanisms (Sections 6.5.1, 5.1.6)
  - Build instruction‑verifiable datasets (e.g., exact word count) and synthetic prompts to systematically test adherence.
  - Factuality is controlled along three axes: closed‑book accuracy, attributions grounded in provided contexts (measured by `AIS`, a human‑judgment metric for attribution faithfulness), and hedging on unanswerable inputs.

- Tool use mechanism (Section 6.5.2; Figure 8)
  - Tool calls are framed as code blocks that the model writes. The system executes the code, returns tool outputs into context, and the model decides next steps—creating a tight “LLM ↔ tools” loop.
  - This design simplifies composition of multiple tools and seamless reflection on tool results.

- Special reasoning trick: uncertainty‑routed chain‑of‑thought (Appendix 10.2)
  - Define: Sample `k` chain‑of‑thought (CoT) rationales; if the answers agree above a threshold on validation, take the majority; otherwise fall back to a greedy non‑CoT answer. This balances the known variance of CoT sampling against the stability of greedy decoding.
  - Effect: On MMLU, `Gemini Ultra` rises from 84.0% (greedy) to 90.0% with uncertainty‑routed CoT@32, meaning the routing logic—rather than CoT alone—is pivotal (Figure 9).

- On‑device `Nano` models (Sections 2, 5.1.3; Table 3)
  - Sizes: ~1.8B and ~3.25B parameters, distilled from larger Gemini models and 4‑bit quantized for deployment. Despite small size, they retain notable reasoning/retrieval ability relative to `Pro`.

## 4. Key Insights and Innovations
- Native, bidirectional multimodality with image output (Sections 2, 5.2.3)
  - What’s new: The same model both understands pixels (without external OCR—“pixel only” in Table 7) and can generate images via discrete image tokens (Figure 6).
  - Why it matters: Enables fluid multimodal prompting (e.g., “rearrange these subplots and output the code,” Figure 5) and end‑to‑end image‑grounded reasoning.

- Uncertainty‑routed CoT for robust exam‑style reasoning (Appendix 10.2; Figure 9)
  - What’s new: Instead of always using CoT or always greedy, route based on sampled consensus.
  - Significance: Converts CoT variability into a strength, delivering 90.04% on MMLU (Table 2), the first time surpassing reported human‑expert performance (89.8%).

- Systems engineering for stable mega‑scale training (Section 3)
  - What’s new: In‑memory replicas for rapid failover, deterministic replay for SDC triage, and superpod orchestration for synchronous multi‑datacenter training.
  - Significance: Raises goodput to 97% and makes training at Gemini Ultra scale feasible and repeatable—this is an enabling contribution rather than an algorithmic tweak.

- A deliberate post‑training stack that measurably improves safety/factuality and tool‑use (Sections 5.1.6, 6.5.2; Tables 6, 15)
  - What’s new: Safety‑targeted SFT (including multimodal safety sets), safety‑aware RLHF, and tool use trained as code generation.
  - Significance: Cuts factual inaccuracy nearly in half and boosts tool‑augmented math/QA substantially; demonstrates that alignment methods can scale across modalities.

- A full compute‑spectrum product strategy (Sections 1, 6, 5.1.3)
  - What’s new: `Ultra` for hardest tasks, `Pro` for latency/scale, and `Nano` for on‑device.
  - Significance: Broadens reach (e.g., summarization or reading comprehension on mobile) without losing the multimodal advances.

## 5. Experimental Analysis
- Evaluation methodology (Section 5; Appendix 10.3)
  - Broad, multimodal harness:
    - Text: MMLU, GSM8K, MATH, BIG‑Bench‑Hard, HumanEval, Natural2Code (a new held‑out Python set), DROP, HellaSwag (reported in a decontaminated 10‑shot setup), WMT23 MT with BLEURT; long‑context tests up to 32k tokens (Figures 3–4; Table 2).
    - Vision: OCR‑heavy VQA (TextVQA, DocVQA), charts/diagrams (ChartQA, InfographicVQA, AI2D), math visual reasoning (MathVista), and college‑level multi‑disciplinary MMMU (Table 7, Table 8).
    - Video: VATEX (captioning in English and Chinese), YouCook2 (captioning), NextQA, ActivityNet‑QA, Perception Test MCQA (Table 10).
    - Audio: ASR (YouTube EN, Multilingual LibriSpeech, VoxPopuli, FLEURS) and speech translation (CoVoST 2), reported as WER (lower is better) and BLEU (higher is better) (Table 11).
  - Baselines and decontamination:
    - Comparisons include GPT‑4, GPT‑3.5, PaLM 2‑L, Claude 2, LLaMA‑2, and domain SOTA systems where available (Tables 2, 7, 10, 11).
    - The paper documents leakage analyses and avoids reporting some compromised benchmarks (e.g., LAMBADA), and uses decontaminated or held‑out setups when needed (Section 5.1.1).

- Main quantitative results (select highlights)
  - Text (Table 2):
    - MMLU: `Ultra` 90.04% with uncertainty‑routed CoT@32; GPT‑4 (API) 87.29%.
    - GSM8K: `Ultra` 94.4% with self‑consistency; GPT‑4 92.0%.
    - MATH (4‑shot): `Ultra` 53.2%; GPT‑4 (API) 52.9%.
    - BIG‑Bench‑Hard (3‑shot): `Ultra` 83.6%; GPT‑4 (API) 83.1%.
    - HumanEval (0‑shot): `Ultra` 74.4%; GPT‑4 67.0%.
    - Natural2Code (0‑shot, held‑out): `Ultra` 74.9%; GPT‑4 (API) 73.9%.
    - WMT23 BLEURT (all directions): `Ultra` 74.4 vs GPT‑4 73.8; out‑of‑English average 74.8 vs GPT‑4 73.6 (Table 4).
  - Long context efficacy: 98% retrieval accuracy across the full 32k window; negative log‑likelihood decreases with position, indicating usable signal even at the tail (Section 5.1.5; Figure 4).
  - Vision (Table 7):
    - MMMU (val): `Ultra` 62.4% Maj1@32 (“majority‑vote across 32 samples”), SOTA; GPT‑4V 56.8% zero‑shot (Table 8 shows subject‑area breakdown).
    - OCR‑style tasks: `Ultra` TextVQA 82.3% and DocVQA 90.9% in “pixel‑only” mode, surpassing GPT‑4V.
    - Charts/infographics: ChartQA 80.8% (on par with best), InfographicVQA 80.3% (SOTA reported).
    - Natural images: VQAv2 77.8% zero‑shot; still below specialized fine‑tuned PaLI‑X 86.1% (Table 7, “Prior SOTA”).
  - Video (Table 10):
    - VATEX captioning (EN): `Ultra` 62.7 CIDEr vs Flamingo 56.0.
    - YouCook2 captioning: `Ultra` 135.4 CIDEr vs Flamingo 74.5.
    - NextQA (0‑shot WUPS): `Ultra` 29.9 vs Flamingo 26.7.
    - ActivityNet‑QA (Top‑1): `Ultra` 52.2 vs Video‑LLaVA 45.3.
    - Perception Test MCQA: `Ultra` 54.7 vs SeViLA 46.3.
  - Audio (Table 11):
    - YouTube EN WER: `Pro` 4.9% (better than Whisper v3 6.5% and USM 6.2%).
    - FLEURS WER across 62 languages: `Pro` 7.6% vs Whisper v3 17.6%.
    - CoVoST‑2 speech translation BLEU: `Pro` 40.1 vs Whisper v2 29.1 and USM 30.7.
  - Factuality and instruction following:
    - Factuality mitigation (Table 6): inaccuracy rate drops from 6.7% to 3.8%; attribution `AIS` rises from 40.2% to 60.0%; hedging at 69.3% (from 0%).
    - Instruction following on complex prompts (Table 14): `Gemini Advanced (Ultra)` reaches 87.4% per‑instruction accuracy; full‑response exact compliance at 54.1%.
  - Tool use (Table 15): fine‑tuning `Pro` with tools boosts GSM8K 69.7% → 80.1%, MATH 30.7% → 41.8%, NaturalQuestions 59.0% → 68.0%, and RealTime QA 39.2% → 70.8%.
  - Agents: AlphaCode 2 built on Gemini ranks ~85th percentile on Codeforces, solving 43% of problems vs 25% for the prior system (Section 5.1.7).

- Post‑training ablations and deltas
  - Vision SFT vs pre‑train: `Ultra` gains +2.4 on InfographicVQA, +2.9 on AI2D, +3.3 on VQAv2 with SFT, aligning outputs with task references (Table 17).
  - Multilingual post‑training includes translated data filtered for “translatability” and human‑rated for quality (Section 6.5.3), which underpins WMT23 improvements (Table 4).
  - Long‑context synthetic recall and Figure 4 support that the 32k window is usable, not just a nominal limit.

- Do the experiments support the claims?
  - Breadth and depth are unusually strong: multiple modalities, many independent public benchmarks, repeated evidence of pre‑train → post‑train improvements, and careful notes on contamination.
  - Caveats are candid, e.g., HellaSwag sensitivity to data composition (Section 5.1.1) and reporting decontaminated/held‑out numbers.

> “Gemini Ultra achieves new state‑of‑the‑art results in 30 of 32 benchmarks… and is the first model to achieve human‑expert performance on MMLU… 90.04%” (Sections 1, 5.1.1; Table 2).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The approach presumes access to vast multimodal corpora and large‑scale TPU clusters; compute and data requirements are not trivial (Sections 3–4; Model card notes “Compute Requirements: Not reported”).
  - Many reported wins rely on sampling strategies (e.g., uncertainty‑routed CoT@32), which increase inference cost and complicate deployment latency (Appendix 10.2).
- Comparability and contamination concerns
  - Some community benchmarks are susceptible to pre‑training overlap; the paper invests in decontamination, but residual uncertainty remains (Section 5.1.1).
  - For vision, comparisons often juxtapose zero‑shot Gemini with fine‑tuned SOTAs—good for judging generality, but not a strict apples‑to‑apples on every task (Table 7).
- Capabilities still behind specialist SOTA in places
  - VQAv2 is below fine‑tuned PaLI‑X; WikiLingua summarization trails PaLM 2 on some setups (Table 5).
  - Audio is not yet reported for `Ultra` (Table 11 notes only `Pro` and `Nano‑1`).
- Safety and bias remain open‑ended problems
  - Red teaming finds vulnerability classes (prompt injection/jailbreaks); persuasion/deception studies are mixed; representational harms and ungrounded inferences can occur (Section 7.4).
  - “Dangerous capabilities” tests suggest limited risk today, but evaluations are evolving and not a proof of absence (Section 7.4.1.3).
- Reproducibility and transparency
  - Model sizes, exact data mixtures/weights, and compute budgets are not detailed; full replication is unlikely outside large labs.
- On‑device trade‑offs
  - `Nano` models are strong “for their size,” but absolute accuracy is lower than `Pro/Ultra` (Table 3), reflecting the usual capability–latency–memory trade space.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that a single decoder‑only model can be trained “natively multimodal,” do OCR‑free image understanding, generate images, reason across audio‑video‑text sequences, and deliver state‑of‑the‑art language reasoning. This reduces the need for modality‑specific stacks and opens the door to truly generalist agents (Figures 2, 5–6; Tables 7, 10, 11).
  - Establishes a pragmatic blueprint for safe deployment: clear post‑training stages, measurable factuality/hedging improvements, and tool‑use integration that materially improves reasoning benchmarks (Tables 6, 15; Figure 8).

- Follow‑up research enabled/suggested
  - Better evaluations: The paper explicitly calls for more robust decontaminated benchmarks and new bias/representation measures beyond saturated datasets (Section 5.1.1; 7.4.1.2).
  - Data mixture optimization: Staged mixtures and data quality strongly affect outcomes; systematic exploration of optimal multimodal curricula remains open (Section 4).
  - Efficient long‑context and multi‑sample inference: Routing strategies (like uncertainty‑routed CoT) could be generalized to budgeted test‑time compute.
  - Multimodal safety: Build richer multimodal red‑teaming corpora, improve refusal vs helpfulness balance, and develop grounding tools to reduce ungrounded inferences in image/video QA (Sections 7.3–7.4).

- Practical applications
  - Education/tutoring with diagram/math support (Figures 1, 14, 21).
  - Enterprise document, chart, and infographic understanding and retrieval‑augmented QA (Table 7; Section 5.1.6).
  - Coding copilots and competitive programming agents (Section 5.1.7).
  - Translation and multilingual assistants, including very low‑resource languages (Table 4).
  - On‑device summarization and reading comprehension via `Nano` (Section 5.1.3; Table 3).
  - Multimodal creative workflows—e.g., code‑driven plot manipulation and image design in the same interaction (Figures 5–6, 22).

> “Post‑training improves image understanding… SFT yields +3.3 on VQAv2 and +2.9 on AI2D” (Table 17), and “inaccuracy is halved… attribution rises 40%→60%… hedging reaches 69%” (Table 6)—evidence that careful alignment work translates into measurable, product‑relevant gains.

Definitions used inline:
- `AIS`: human‑judged attribution measure—whether model responses faithfully cite/ground to provided sources (Section 5.1.6).
- `BLEURT`: learned metric for translation/summarization quality; higher is better (Table 4).
- `CIDEr`: captioning metric; higher is better (Table 10).
- `WER`: word error rate in speech recognition; lower is better (Table 11).
- `Maj1@32`: accuracy when taking the majority answer across 32 sampled rationales (Tables 7–8).
- `uncertainty‑routed CoT`: decide between majority CoT vote and greedy decoding based on confidence (Appendix 10.2).
- “pixel only”: results obtained without any external OCR system; the model reads text directly from image pixels (Table 7).
- `goodput`: proportion of elapsed training time spent doing useful new update steps (footnote 2; Section 3).
