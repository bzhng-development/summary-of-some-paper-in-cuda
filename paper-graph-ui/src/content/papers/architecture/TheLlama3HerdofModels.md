# The Llama 3 Herd of Models

**ArXiv:** [2407.21783](https://arxiv.org/abs/2407.21783)

## 🎯 Pitch

Llama 3 introduces a state-of-the-art open family of foundation language models at three parameter scales—8B, 70B, and a flagship 405B dense Transformer—delivering robust multilingual, coding, reasoning, and tool-use abilities, all within a 128K-token context window. By leveraging massive, rigorously curated data and a pragmatic, scalable training pipeline, Llama 3 not only matches leading closed models like GPT-4 in quality across diverse tasks, but also pioneers a modular approach to integrating vision, video, and speech, ushering in a new era of accessible, high-performance AI for research and real-world applications.

---

## 1. Executive Summary (2–3 sentences)
Llama 3 is a family (“herd”) of open‑weight foundation language models at three scales (8B, 70B, 405B parameters) with a 128K‑token context window, strong multilingual/coding/reasoning/tool‑use capabilities, and a pragmatic safety stack. It advances the state of open models through (a) massive, higher‑quality data and careful scaling laws to pick a compute‑optimal 405B dense Transformer, (b) a simple but robust post‑training pipeline (SFT + DPO) that delivers GPT‑4‑class results on many tasks, and (c) a compositional path to add vision, video, and speech without degrading text performance.

## 2. Context and Motivation
- Problem/gap addressed
  - High‑quality, widely accessible foundation models are scarce at the flagship scale; prior open releases (e.g., Llama 2, Mixtral) either lag top closed models or require complex sparse architectures. Llama 3 aims to deliver a dense, stable, long‑context, multilingual model that rivals closed models on diverse tasks and publishes both methods and system recipes.
- Why this matters
  - Practical: Better assistants (tool use, coding, long documents, speech I/O) and safer deployment (guardrails, refusal calibration).
  - Scientific: Clear evidence for data/scale/simplicity levers, forecasts for downstream performance from scaling laws, and robust engineering recipes (parallelism, FP8 inference) for very large training.
- Prior approaches and shortcomings
  - Closed models (GPT‑4/o, Claude 3.5) set quality bars but are not open. Mixture‑of‑experts (MoE) models (e.g., Mixtral) increase capacity but complicate training/inference. Many long‑context models lack robust long‑context pre‑/post‑training; tool use and multilingual safety remain uneven.
- Positioning
  - Llama 3 opts for a standard dense Transformer with targeted architectural tweaks and a heavy emphasis on data quality and training stability (Sections 3–4). It also demonstrates a modular route to multimodality (Sections 7–8) and a layered safety strategy (Section 5.4), while releasing both pre‑trained and post‑trained weights (Abstract; Table 1).

## 3. Technical Approach
This section walks through how Llama 3 is built, trained, aligned, and extended.

- Model family and architecture (Section 3.2; Table 3; Figure 1)
  - Three sizes: `8B`, `70B`, `405B`.
  - Dense Transformer with two notable choices:
    - `GQA` (grouped query attention) with 8 KV heads: reduces key–value cache size and speeds decoding (Table 3).
    - `RoPE` (rotary positional embeddings) base frequency increased to 500,000 to better extrapolate to long context; later extended to 128K with continued pre‑training (Sections 3.2, 3.4.2).
  - New 128K‑token vocabulary: 100K from `tiktoken` plus 28K non‑English tokens; improves English compression from 3.17→3.94 characters/token without harming English tokenization and helps multilingual performance (Section 3.2).

- Data pipeline (Section 3.1)
  - Web data undergoes PII and safety filtering, HTML parsing that preserves math/code, and multi‑level dedup (URL/doc/line). Additional heuristics remove repeated content and low‑quality domains; quality and domain‑specific classifiers (trained on Llama 2 judgments) upsample code/reasoning/math (Sections 3.1.1–3.1.2).
  - Final pre‑training mix: ~50% general knowledge, 25% math/reasoning, 17% code, 8% multilingual (Section 3.1.2 “Data mix summary”).
  - Annealing small, high‑quality domain data near the end improves small models; the 405B model’s improvements are minimal, implying strong in‑context learning at scale (Section 3.1.3).

- Scaling laws and sizing (Section 3.2.1; Figures 2–4)
  - Two‑step forecast connects compute→validation log‑likelihood→downstream accuracy, enabling accurate predictions of big‑model performance on tasks like ARC‑Challenge. IsoFLOPs curves identify compute‑optimal trade‑offs; extrapolation to 3.8×10^25 FLOPs recommends ~402B parameters on ~16.55T tokens (Figures 2–3).
  - Key observation: IsoFLOPs minima flatten at high compute, making performance robust to small changes in size vs tokens (Section 3.2.1).

- Training at scale (Sections 3.3–3.4; Table 4; Table 5; Figures 5–6)
  - Hardware and cluster: up to 16K H100 GPUs (80GB HBM3), RoCE fabric with topology‑aware scheduling, and 240PB storage fabric; >90% effective training time over a 54‑day snapshot (Sections 3.3.1, 3.3.4; Table 5).
  - Parallelism: 4D sharding—`Tensor (TP)`, `Pipeline (PP)`, `Context (CP)`, and sharded `Data Parallel (FSDP)`—ordered [TP, CP, PP, DP] to match network constraints (Sections 3.3.2, 3.3.3; Figure 5). Custom pipeline schedule lets the number of continuous micro‑batches `N` be tunable (Figure 6). Achieved 38–43% BF16 Model FLOPs Utilization (Table 4).
  - Long‑context pre‑training: increase context in six stages (8K→128K), require recovery of short‑context performance and perfect “needle in a haystack” detection at each step; ~800B tokens for the long‑context stage (Section 3.4.2).
  - Final annealing: linearly decay LR to zero over last 40M tokens at 128K context; checkpoint averaging (Polyak) yields the final pre‑trained model (Section 3.4.3).

- Post‑training for instruction following and preference alignment (Section 4; Figure 7)
  - Chat protocol supports multiple messages per turn (for tools) via special headers/terminators (Section 4.1.1).
  - `Reward Modeling (RM)`: trained on human preference triples (edited > chosen > rejected); efficiency trick concatenates shuffled responses per prompt in one row (Section 4.1.2).
  - `Supervised Fine‑Tuning (SFT)` with rejection sampling: for each human prompt, sample 10–30 outputs and pick the best using the RM; `PagedAttention` doubles throughput by sharing prompt KV cache across samples (Section 4.1.3; 4.2.2).
  - `DPO` (Direct Preference Optimization): β=0.1; mask formatting tokens in loss to avoid degenerate behaviors; add NLL regularizer (0.2×) on chosen responses to stabilize formats and prevent likelihood collapse (Section 4.1.4).
  - Model averaging across runs/hyperparameters further smooths performance (Section 4.1.5).

- Capability‑specific improvements (Section 4.3)
  - Code: a continued‑pre‑trained “code expert,” plus large‑scale synthetic data with execution feedback (static checks + unit tests + iterative self‑correction), cross‑language translation, and “backtranslation” for documentation/explanations (Section 4.3.1; Figure 8–9).
  - Multilingual: a multilingual expert (90% non‑English continued pre‑training), high‑quality native annotations, rejection‑sampled SFT, and careful control of language/script matching; limited use of translations (only for math) to avoid translationese/bias (Section 4.3.2).
  - Reasoning: generate step‑wise chains with answer‑checking, train outcome/stepwise reward models, use MCTS on hard problems, and interleave text/code with execution as verification (Section 4.3.3).
  - Long context: synthesize QA/summarization/repo‑reasoning data bucketed by length; mixing only ~0.1% long‑context SFT data retained short‑context quality; DPO can remain short‑context if SFT is strong (Section 4.3.4).
  - Tool use: built‑in tools (Brave Search, Python, Wolfram Alpha) and zero‑shot function calling; message‑level human preferences (not only whole replies), plus multi‑step synthetic traces (ReAct‑style) and file‑upload tasks (Section 4.3.5; Figures 10–11).
  - Factuality: “knowledge probe” generates question/answer pairs from pre‑training snippets and trains calibrated refusing when uncertain (Section 4.3.6).
  - Steerability: preference data with diverse system prompts to control tone/format/length/persona; included in RM, SFT, DPO (Section 4.3.7).

- Safety pipeline (Section 5.4)
  - Pre‑training: domain filtering for PII/adult content; low verbatim memorization measured by n‑gram prompts at scale (e.g., 405B shows 1.13% inclusion for 50‑grams; Table 24).
  - Finetuning: pair “adversarial” prompts (to measure violation rate, `VR`) with “borderline” prompts (to measure false refusal rate, `FRR`), and balance safety vs helpfulness through SFT and DPO; small models need higher safety data ratios (Figure 18).
  - System‑level guardrails: `Llama Guard 3` (an 8B classifier) for input/output filtering across 13 risk categories, with per‑category toggles and int8 quantization; two additional components—`Prompt Guard` (prompt‑attack classifier) and `Code Shield` (static insecure‑code detection) (Section 5.4.7; Tables 26–28).

- Inference efficiency (Section 6; Figures 24–27)
  - Pipeline‑parallel inference across 16 GPUs, plus micro‑batching to increase throughput; modest latency increase but better throughput/latency trade‑off (Figure 24).
  - `FP8` low‑precision inference on H100: quantize most MLP matmuls with dynamic row‑wise scales, cap scales to 1200, skip first/last layers; reward‑model score distributions match BF16 (Figure 26), and throughput improves up to 50% in prefill with superior decoding trade‑offs (Figure 27).

- Compositional multimodality (Sections 7–8; Figures 28–29)
  - Vision/video: a pre‑trained image encoder (ViT‑H) and cross‑attention “vision adapter” injected after every 4th LLM layer; for video, add temporal aggregator and video cross‑attention; keep LLM weights frozen during adapter training to preserve text performance (Section 7.2).
  - Speech: a 1B‑parameter Conformer encoder + lightweight adapter that outputs token‑rate embeddings the LLM can consume directly; system prompts select ASR or speech translation modes; a streaming TTS stack uses Llama 3 embeddings to improve text normalization and prosody with low latency (Section 8.2).

## 4. Key Insights and Innovations
1) A practical, accurate scaling‑law method for downstream performance
- What’s new: A two‑step forecast links training FLOPs → validation NLL on benchmarks → final accuracy, using both small scaling‑law runs (≤10^22 FLOPs) and prior Llama‑2 models (Figure 4). The selected 405B size (≈402B predicted) is compute‑optimal for 3.8×10^25 FLOPs, with robustness near the IsoFLOPs minimum (Figures 2–3).
- Why it matters: It lets teams predict end performance and pick model size/data budget before spending huge compute, reducing risk of under/oversized models.

2) Dense‑model scaling with stability and simplicity
- What’s new: Llama 3 stays dense (not MoE), yet matches/approaches closed models by focusing on data quality, careful long‑context adaptation, and robust distributed training (4D parallelism, NCCLX comms, and pipeline scheduling; Sections 3.3–3.4).
- Why it matters: Training/inference are simpler and more stable than MoE, and the published engineering recipes are immediately reusable.

3) A lean but strong alignment pipeline (SFT + DPO) with targeted tweaks
- What’s new: Rejection sampling using an RM, DPO with formatting‑token masking and NLL regularization, model averaging, and carefully curated capability‑specific datasets (Section 4). DPO remains short‑context without harming long‑context performance if SFT is strong (Section 4.3.4).
- Why it matters: It achieves high instruction fidelity and reasoning without complex RL pipelines, and scales reliably to 405B.

4) Layered safety with measurable VR/FRR trade‑offs and deployable guards
- What’s new: Balance adversarial vs borderline data to lower violations while avoiding over‑refusal (Figure 18), demonstrate long‑context jailbreak mitigation (Figure 20), and provide `Llama Guard 3` and `Prompt Guard` as system components with per‑category controls (Tables 25–26, 28).
- Why it matters: It operationalizes safety as a tunable system property rather than a single monolithic setting.

5) Compositional adapters for vision, video, and speech
- What’s new: Cross‑attention “vision adapter” and temporal aggregator let Llama 3 reach competitive VQA/video QA without joint multimodal pre‑training; speech uses a token‑rate interface (no cross‑attention) for ASR/AST/spoken dialog and streaming TTS with Llama‑embeddings (Sections 7–8).
- Why it matters: Adds modalities without regressing text performance or requiring end‑to‑end re‑training of the LLM.

6) FP8 inference that preserves response distribution
- What’s new: Row‑wise dynamic scaling with a capped scale avoids underflow spikes on high‑perplexity tokens; reward‑model score distributions are nearly unchanged vs BF16 (Figure 26) while throughput improves (Figure 27).
- Why it matters: Safe, high‑speed inference for very large models.

## 5. Experimental Analysis
- Evaluation setup
  - Pre‑training and post‑training are evaluated across general knowledge (MMLU/Pro), reasoning (GSM8K/MATH/ARC‑C/GPQA), code (HumanEval/MBPP, MultiPL‑E), long‑context (ZeroSCROLLS, InfiniteBench, needle tests), tool use (Nexus, API‑Bank, API‑Bench, BFCL), multilingual (MGSM, translated MMLU), and proficiency exams, plus adversarial robustness and contamination checks (Sections 5.1–5.2; Table 8/16).
  - Safety uses internal adversarial/borderline sets spanning MLCommons hazard taxonomy; system‑level guards are measured with VR/FRR and category‑wise metrics (Section 5.4; Figures 19–21; Tables 25–26).

- Headline post‑training results (Table 2; all Llama 3.1)
  > On core benchmarks, `Llama 3.1 405B Instruct` reports: MMLU 87.3 (5‑shot), GSM8K 96.8 (8‑shot CoT), HumanEval 89.0 (0‑shot), MBPP EvalPlus 88.6 (0‑shot), ARC‑Challenge 96.9 (0‑shot), GPQA 51.1 (0‑shot CoT).  
  > In many cases it is competitive with GPT‑4 (0125), GPT‑4o, and Claude 3.5 Sonnet; the 8B and 70B models are best‑in‑class within their size brackets (Table 2).

- Pre‑trained model quality (Tables 9–13, 14)
  - `405B` reaches MMLU 85.2, MATH 53.8 (0‑shot CoT), ARC‑C 96.1, HumanEval 61.0/MBPP 73.4. Long‑context: QuALITY 87.6 (5‑shot), many‑shot GSM8K 90.0 (16‑shot) (Table 14).
  - Robustness checks on MMLU show little sensitivity to label variants, few‑shot label biases, answer permutations, and prompt formats—especially at 405B (Figures 13–14).

- Proficiency exams (Table 17)
  - `405B` scores: LSAT 81.1, SAT Reading 74.8, SAT Math 94.9, GMAT Quant 96.0, GRE Quant/Verbal 162/166. The `70B` model is strong and often beats larger open models (e.g., Nemotron 340B).

- Tool use/function calling (Table 22; Figures 10–11; 16)
  - `405B` achieves 92.3% on API‑Bank and 88.5% on BFCL, near GPT‑4/4o; Nexus 58.7% and API‑Bench 35.3% are competitive. In human evals on code execution/plotting/file uploads, `405B` beats GPT‑4o on execution and plotting but lags on file uploads (Figure 16).

- Long context (Table 21)
  > `405B` achieves QuALITY 95.2 (EM, val), InfiniteBench En.MC 83.4 (acc) and En.QA 30.5 (F1), and 98.1 average recall on Multi‑needle across context lengths up to 128K. GPT‑4(o) slightly outperforms on En.QA and Multi‑needle tops out at 100.0 for GPT‑4/4o.

- Multilingual (Table 20)
  - `405B` reaches 91.6 on MGSM (0‑shot CoT) and 83.2 averaged translated MMLU (5‑shot). `70B` and `8B` lead their size classes by wide margins.

- Human evaluations (Section 5.3; Figure 17)
  - Pairwise win/loss vs top closed models varies by capability. `405B` is roughly on par with GPT‑4 (0125), mixed against GPT‑4o and Claude 3.5 Sonnet: it leads on single/multiturn English, trails on coding/reasoning vs Claude 3.5 Sonnet, and is comparable on multilingual.

- Safety outcomes (Section 5.4; Figures 19–21; Tables 25–26, 28)
  - Model‑level vs system‑level: adding `Llama Guard 3` substantially reduces violation rates at the cost of some increased FRR; reductions are strong across hazard categories (Table 26). On multilingual short‑context sets, `405B + LG3` is at least as safe as anonymized commercial systems (Figure 19).
  - Long‑context safety: both DocQA and Many‑shot jailbreaks show markedly lower VR for `405B` (with or without LG) than one commercial system and a Pareto improvement over another (Figure 20).
  - External risk: on prompt injection, `Llama 3` sits between GPT‑4 Turbo/Gemini Pro (less susceptible) and Mixtral (more) (Figure 22). Uplift studies show no significant increase in capability for cyber or CBRNE attacks relative to web search alone (Section 5.4.5).

- Contamination analysis (Table 15)
  - 8‑gram overlap suggests varying levels of potential contamination across datasets (e.g., high in HellaSwag/PiQA) but estimated performance gain is sometimes small (e.g., NaturalQuestions). Some datasets (MBPP, MMLU/Pro) require alternative methods due to high apparent overlap; the study is careful to present uncertainties (Section 5.1.4).

- Multimodality results (Tables 29–30)
  - Vision: `Llama 3‑V 405B` is competitive with GPT‑4V and near Gemini/Claude on MMMU (64.5), ChartQA (85.8), TextVQA (84.8), and DocVQA (92.6) (Table 29).
  - Video: `Llama 3‑V 70B` achieves 60.8 on PerceptionTest (test), 87.9 on TVQA (val), and 56.3 on ActivityNet‑QA (test), competitive with Gemini variants; results are zero‑shot (Table 30).

- Speech results (Tables 31–35; Figure 30)
  - ASR: `70B` reaches WER 4.4 on MLS English, 3.1 on LibriSpeech test‑other; both `8B`/`70B` outperform Whisper v2/v3 and SeamlessM4T v2, and are close to Gemini on MLS English (Table 31).
  - AST: On FLEURS and CoVoST2 (to English), results are competitive with Whisper v2 and SeamlessM4T v2 (Table 32).
  - Safety (MuTox): very low added toxicity (≤0.84% English) and substantial toxicity removal (Table 33).
  - Streaming TTS: adding Llama‑embeddings improves text normalization accuracy and prosody preferences over phone‑only baselines while staying streamable (Tables 34–35).

Overall, the experiments are extensive, cover core and edge capabilities, include robustness and safety, and disclose limitations where performance is mixed (e.g., some tool/file‑upload tasks, some long‑context QA settings).

## 6. Limitations and Trade-offs
- Assumptions and choices
  - Dense architecture chosen for stability; while simpler than MoE, it is inference‑heavier per token than sparse experts (Section 3.2). The 405B model demands significant compute infrastructure.
  - Long‑context capability is achieved by staged continued pre‑training plus a small fraction of long‑context SFT; certain tasks (InfiniteBench En.QA) still leave room to improve (Table 21).
- Safety trade‑offs
  - Reducing VR increases FRR; `Llama Guard 3` provides category toggles, but tuning is still application‑specific (Tables 25–26; Figures 19–21).
  - Prompt‑injection and tool misuse risks are mitigated but not eliminated (Figure 22; Section 5.4.6 tool‑specific red‑team findings).
- Data and measurement
  - Contamination estimates via 8‑gram overlap can be noisy for some benchmarks (Table 15). Some multilingual safety depends on the coverage/quality of non‑English data (Figure 19 discussion).
- Engineering trade‑offs
  - Micro‑batching raises throughput but adds synchronization points that can increase latency (Figure 24).
  - FP8 inference excludes attention layers and needs careful scale capping and row‑wise quantization to avoid rare corruption (Section 6.2).
- Multimodality status
  - Vision/video/speech components are promising but “still under development” and not broadly released; video uses up to 64 frames and temporal aggregation may miss fine events (Sections 7–8).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that carefully trained dense open models can approach closed‑model performance at flagship scale while providing transparent methods (data curation, scaling laws, long‑context training, safety tuning, FP8 inference). This raises the baseline for open research and practical deployments.
- Follow‑up research
  - Long‑context: stronger reasoning over 100K+ tokens (e.g., improved summarization/QA training, retrieval‑augmented long‑context).
  - Safety: adaptive defenses against jailbreaking/prompt injection (e.g., proactive tool‑call validation, multi‑agent verification), improved multilingual/borderline calibration, and standardized contamination auditing.
  - Data: principled annealing and curriculum strategies; richer reasoning/code datasets with verified step traces; better tool‑grounded corpora (especially for file workflows).
  - Inference/systems: broader FP8 coverage (including attention), quantization‑aware training, and heterogeneous clusters; improved pipeline scheduling for interactive workloads.
  - Multimodality: broader release and scaling of adapters; unified training that preserves text without degradation; tighter integration of speech prosody/semantics and vision grounding for tool use.
- Applications
  - Enterprise assistants (analysis of long documents, spreadsheets, PDFs; Section 4.3.5 and Figure 11), developer tools (code gen/debug/review with execution feedback; Section 4.3.1), multilingual support (Table 20), STEM tutoring and exams (Table 17), research assistants (tool use + factuality probes; Sections 4.3.5–4.3.6), and safety‑aware platforms (Llama Guard 3, Prompt Guard; Section 5.4.7).

---

Below are selected, concrete references used in the analysis:
- Architecture & training: Figure 1; Sections 3.1–3.4; Tables 3–4; Figures 2–6; Table 5.
- Post‑training & capabilities: Figure 7; Sections 4.1–4.3; Figures 8–11; Tables 6–7.
- Headline performance: Table 2 (post‑trained), Tables 9–14 (pre‑trained).
- Long context: Section 3.4.2; Table 21.
- Tool use/function calling: Section 4.3.5; Table 22; Figures 10–11.
- Multilingual: Section 4.3.2; Table 20.
- Human evals: Section 5.3; Figures 16–17.
- Safety: Section 5.4; Figures 18–21; Tables 24–28.
- Inference: Section 6; Figures 24–27.
- Vision/speech: Sections 7–8; Tables 29–35; Figures 28–30.
