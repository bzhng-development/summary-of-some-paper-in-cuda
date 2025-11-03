# OLMo: Accelerating the Science of Language Models

**ArXiv:** [2402.00838](https://arxiv.org/abs/2402.00838)
**Authors:** Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Raghavi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, Hannaneh Hajishirzi
**Institutions:** 

## 🎯 Pitch

OLMo sets a new standard for openness in language-modeling by releasing not only the model weights but also the full pretraining dataset, training code, and evaluation framework. This transparency fosters reproducible research, enabling the community to explore how data and design choices impact model capabilities, and reduces environmental costs by facilitating the reuse of robust models.

---

## 1. Executive Summary
- This paper introduces OLMo, a “truly open” language-modeling framework that releases not just model weights, but also the full pretraining dataset (Dolma), the end‑to‑end training and evaluation code, hundreds of intermediate checkpoints, and training logs (Section 1; Sections 2.1–2.4; Section 5).
- The significance is twofold: it enables rigorous, reproducible science on how data, design choices, and optimization shape LMs, and it delivers competitive 1B and 7B models trained on 2T+ tokens with transparent, decontaminated evaluation (Table 1; Figure 2; Section 2.4).

## 2. Context and Motivation
- Problem addressed
  - As LMs become commercially valuable, the most capable systems are gated behind proprietary APIs; details such as training data composition, model design choices, and training traces are undisclosed. This blocks scientific study of model behavior, risks, and biases (Section 1).
  - Even in “open” models, openness varies: e.g., weights without data/code, partial dataset descriptions, or limited checkpoints. The paper catalogues this spectrum and positions OLMo to cover the full stack (Intro: comparisons to Mixtral, LLaMA, MPT, Falcon, Pythia, BLOOM).

- Why it matters
  - Without data and full training artifacts, core questions—How do filters, deduplication, or data mixtures affect capabilities and harms? Which optimization choices prevent divergence?—remain difficult to answer empirically (Sections 1–2.2, 2.4).
  - Openness also lowers duplicated training and environmental cost by allowing re-use of robust base models (Ethics Statement; Appendix B).

- Prior approaches and gaps
  - Pythia and BLOOM came closest by releasing weights and training code, but pretraining data access, exact data order, and exhaustive logs/checkpoints were not always available, especially for later competitive models (Intro; Section 5).
  - Recent “360°” efforts (LLM360) aim for transparency; OLMo narrows the performance gap to stronger families like LLaMA 2 while remaining fully open (Intro).

- OLMo’s positioning
  - A complete framework: open pretraining data (Dolma), reproducible training/inference/evaluation code, 500+ intermediate checkpoints, training logs, decontaminated perplexity evaluation, and freely usable Apache 2.0 licensing (Sections 2.1–2.4; Section 5).
  - Competitive 7B results on common zero-shot benchmarks and strong fit on web-like text distributions (Table 3; Figure 2).

## 3. Technical Approach
This section explains the “how”: the model design, data pipeline, training system, and evaluation methodology.

- Model architecture (Section 2.1; Tables 1 and 5)
  - Base: decoder‑only transformer (standard for LMs).
  - Sizes: `OLMo-1B` (16 layers, 16 heads, hidden size 2048) and `OLMo-7B` (32 layers, 32 heads, hidden size 4096 per Table 5; Table 1 lists the 7B setting and training tokens).
  - Key design choices for stability and throughput:
    - No bias terms anywhere (“No biases”) to improve stability (Section 2.1).
    - Non‑parametric LayerNorm: normalization without learnable gain/bias (“affine”)—faster and stable compared to parametric LayerNorm and RMSNorm in the team’s ablations (Section 2.1).
    - `SwiGLU` activation with expansion ~8/3 of hidden size; increase to nearest multiple of 128 to align with hardware-friendly shapes (e.g., 11,008 for 7B), improving throughput (Section 2.1).
    - `RoPE` (rotary positional embeddings), which encode token positions by rotating query/key vectors—a well-accepted replacement for absolute embeddings that extends generalization to longer contexts (Section 2.1).
    - Tokenizer: Modified GPT‑NeoX BPE with extra tokens for masking personally identifiable information (PII). Vocabulary is 50,280, but the embedding matrix is padded to 50,304 (multiple of 128) for speed (Section 2.1).
    - Weight tying: used in the 1B model, not in the 7B model (Table 1).
  - Comparison to other 7–8B models (Table 5): OLMo uses full attention (not GQA/MQA), RoPE, non-parametric LayerNorm, and a 2048 sequence length with a large global batch (~4M tokens).

- Pretraining data: Dolma (Section 2.2; Table 2)
  - Open, multi-source, and large-scale (≈2.7T tokens in the corpus; OLMo trains on a 2.0–2.46T token sample).
  - Sources and scale (Table 2): Common Crawl web pages (2,180B tokens), GitHub code (342B), Reddit (80B), Semantic Scholar papers (57B), Gutenberg books (5.2B), Wikipedia (3.7B).
  - Curation pipeline: language filtering → quality filtering → content filtering → deduplication → multi-source mixing → tokenization. Tools for fast re-creation and analysis are open-sourced (Section 2.2).

- Training data preparation (Section 3.3)
  - Each document gets an EOS token, then all documents are concatenated and split into 2048-token sequences.
  - The full sequence list is shuffled deterministically (same order across runs) and batched; artifacts to reconstruct data order are released (Section 5).
  - Models train for at least 2T tokens; some continue into a second epoch with a new shuffle. The paper argues repeating at this scale has negligible adverse effect (Section 3.3).

- Distributed training and precision (Sections 3.1–3.2)
  - Parallelism: ZeRO sharding via PyTorch `FSDP`, which shards weights and optimizer state across GPUs to fit larger models (Section 3.1).
  - Micro-batch per GPU: 4096 tokens; global batch ≈4M tokens (sequence length 2048), enabling high-throughput training (Section 3.1).
  - Mixed precision: most compute in `bfloat16` (`bf16`) for speed; numerically sensitive ops (e.g., softmax) in full precision; weights and optimizer states kept in FP32 locally; gradients reduced in FP32 for stability (Section 3.1).
    - `bfloat16` is a 16-bit floating point format with an 8‑bit exponent (like FP32) and 7‑bit mantissa, giving good dynamic range with fewer precision bits—commonly used to stabilize large-model training.

- Optimizer and schedule (Section 3.2; Table 1)
  - `AdamW` with betas (0.9, 0.95), epsilon 1e‑5, weight decay 0.1 (Tables 1 and 5).
  - Learning rate warmup for 5000 steps (~21B tokens), linear decay thereafter down to 0.1×peak LR. Global gradient clipping to L2‑norm 1.0 (Section 3.2).
  - A final phase tunes for 1000 extra steps with LR decayed to 0, which improves performance across tasks (Figure 1; Section 4).

- Hardware and vendor parity (Section 3.4)
  - Two clusters to verify portability and performance:
    - LUMI: up to 256 nodes with AMD MI250X (128GB each device; dual‑chip modules), 800 Gbps interconnect.
    - MosaicML (Databricks): 27 nodes with NVIDIA A100‑40GB, 800 Gbps interconnect.
  - Despite minor batch differences to optimize throughput, the two 7B runs achieved nearly identical results by 2T tokens (Section 3.4).

- Evaluation framework (Section 2.4)
  - Downstream tasks via `Catwalk`—a unified evaluation tool with standard zero-shot rank classification (Brown et al.-style): score each multiple-choice option by likelihood and pick the highest; different normalization choices are used per dataset (Section 4.1).
  - Intrinsic fit via `Paloma`—a perplexity benchmark that reports “bits per byte” (bpb) across 585 domains and many sources (Section 2.4).
    - `bits per byte` is model cross-entropy measured over bytes; lower is better. It enables fair comparison across tokenizers.
  - “Online” in-loop evaluation every 1000 steps (~4B tokens) to steer ablations (Section 2.4; Figure 1).

- Adaptation pipeline (Sections 2.3, 4.3; Appendix D/E)
  - Two-stage post-training to turn the base model into a chat assistant:
    - Supervised instruction tuning (`SFT`) on the TÜLU v2 mixture of instruction data (3 epochs; LR 2e‑6; 3% warmup; linear decay to 0; max length 2048).
    - Preference optimization with `DPO` (Direct Preference Optimization) on UltraFeedback pairs (LR 5e‑7; β=0.1; 3 epochs).
      - DPO directly optimizes the model to assign higher probability to “chosen” responses vs “rejected” ones, without reinforcement learning rollouts (Section 2.3; Appendix D).

- Decontamination (Section 2.4; Figure 2)
  - For Paloma perplexity, pretraining documents that leak evaluation paragraphs are removed. This avoids inflated results on domains that accidentally overlap with training data.

## 4. Key Insights and Innovations
- Full‑stack openness for scientific study (Sections 1–2; Section 5)
  - What’s new: release of training data (Dolma), data-building code, per-step data ordering tools, modeling and training code, 500+ intermediate checkpoints, training logs, and evaluation frameworks with decontamination.
  - Why it matters: enables controlled experiments on how data filters, mixtures, or optimizers change outcomes; supports reproducibility and fair cross-model comparisons.

- A transparent, competitive 7B model trained to ≥2T tokens (Tables 1 and 3; Figure 2)
  - What’s new: a 7B base model with competitive zero-shot results (average 69.3 across 8 tasks in Table 3) and strong perplexity fit on web-like distributions, fully auditable from data to logs.
  - Why it matters: narrows the capability gap with popular semi-open models while preserving openness that those models lack.

- Rigorous perplexity evaluation with explicit decontamination and intermediate checkpoints (Section 2.4; Figure 2)
  - What’s new: OLMo-7B is “the largest LM with explicit decontamination for perplexity evaluation” (Section 2.4). Checkpoints allow trajectory analysis (how fit improves with more tokens and LR schedule).
  - Why it matters: prevents overestimating out-of-sample fit and supports analysis of sample efficiency across domains (e.g., C4 vs Wikipedia vs S2ORC in Figure 2).

- Training stability and throughput choices validated in-loop (Sections 2.1, 2.4; Figure 1; Table 5)
  - What’s different: non‑parametric LayerNorm, no biases, SwiGLU, RoPE, and large global batch with FSDP+bf16 are tuned using frequent online evaluation. A late “LR-to-zero” phase boosts end-task accuracy (Figure 1).
  - Why it matters: the paper documents the design and shows how each choice is baked into a consistent, reproducible training recipe across AMD/NVIDIA.

- Open chat models via the TÜLU pipeline (Section 4.3; Table 4; Appendix D/E)
  - What’s new: releases OLMo+SFT and OLMo+SFT+DPO chat models and their training code/data.
  - Why it matters: demonstrates OLMo’s utility as a base for safe, helpful assistants; transparent post-training enables fair comparisons and future improvements.

## 5. Experimental Analysis
- Evaluation methodology and setup
  - Downstream zero-shot tasks (Section 4.1; Table 3):
    - 8 core benchmarks: `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `openbookqa`, `piqa`, `sciq`, `winogrande`.
    - Rank classification with dataset-specific likelihood normalization: unconditional normalization for ARC and OBQA; per-token for HellaSwag/PIQA/WinoGrande; none for BoolQ/SciQ (Section 4.1).
  - Intrinsic perplexity (Section 4.2; Figure 2):
    - Reports bits per byte on a combined metric over 11 publicly accessible sources (C4, mC4-en, WikiText-103, PTB, RedPajama, Falcon-RefinedWeb, Dolma, M2D2 S2ORC/Wikipedia, C4 100 domains, Dolma 100 Subreddits), plus each source individually.
    - Decontamination applied to OLMo pretraining data.
  - Adaptation evaluation (Section 4.3; Table 4; Appendix E):
    - `MMLU` (0-shot accuracy), `AlpacaEval` (%win vs Davinci-003 judged by GPT‑4), `ToxiGen` (% toxic outputs on adversarial prompts), `TruthfulQA` (% informative and truthful responses judged by Llama‑2 classifiers).
  - Additional analyses:
    - Online in-loop curves every 1000 steps (Figure 1).
    - Extra end-tasks with unstable signal (Figure 4; Table 7).
    - Carbon accounting (Appendix B; Table 6).

- Main quantitative results
  - Zero-shot downstream (Table 3):
    > “OLMo‑7B … avg. 69.3” with task-wise scores “arc_c 48.5, arc_e 65.4, boolq 73.4, hellaswag 76.4, openbookqa 50.4, piqa 78.4, sciq 93.8, winogrande 67.9.”
    - Competitive with `MPT‑7B` (69.8) and `Llama‑2‑7B` (70.5); ahead of `Pythia‑6.9B` (63.0) and `RPJ‑INCITE‑7B` (66.6).
    - `OLMo‑1B` is stronger than `Pythia‑1B` and `TinyLlama‑1.1B` on average (60.4 vs 54.5 and 59.4).
  - Training dynamics (Figure 1):
    > “We can see the benefit of decaying LR to 0 in the final 1000 steps of training on most tasks.”
    - Many tasks show a clear last‑phase jump (e.g., HellaSwag, BoolQ), supporting the LR‑to‑zero tweak’s effectiveness.
  - Perplexity/fit (Figure 2):
    > “While models follow a general data scaling trend, sample efficiency is most favorable on in-distribution data. For example, OLMo‑7B overtakes all other models on C4, perhaps from having 88.8% Common Crawl pretraining data.”
    - OLMo is competitive on the aggregated 11‑source metric and strongest on C4‑like content; it is less sample‑efficient on curated sources like WikiText‑103, M2D2 S2ORC/Wikipedia—domains underrepresented in Dolma’s web-heavy mix.
  - Adaptation (Table 4):
    > Base OLMo: “MMLU 28.3 … ToxiGen 81.4% toxic … TruthfulQA 31.6%”  
    > After SFT: “MMLU 47.3 … ToxiGen 14.4% … TruthfulQA 41.2%”  
    > After SFT+DPO: “MMLU 46.2 … ToxiGen 1.7% … TruthfulQA 52.0% … AlpacaEval 69.3% win”
    - Instruction tuning dramatically improves capabilities and safety; DPO further reduces toxicity and increases truthfulness. While TÜLU‑2 and Llama‑2‑Chat remain stronger on some axes (e.g., AlpacaEval %win), OLMo’s toxicity is especially low after DPO (1.7%).
  - Additional perplexity sources (Figure 3):
    - OLMo is far ahead on “Dolma 100 Programming Languages,” likely due to identical preprocessing and possible contamination in code data being hard to eliminate (Appendix C cautions apply).
  - Additional end-tasks (Table 7; Figure 4):
    > On 6 extra tasks, OLMo‑7B averages 47.5 (best among listed models), but curves are unstable, and several tasks provide weak development signal.
  - Energy and carbon (Table 6; Appendix B):
    > “OLMo‑7B MI250X: 135 MWh, PUE 1.1, carbon intensity 0.000 → 0 tCO2eq (LUMI is 100% renewable). OLMo‑7B A100‑40GB: 104 MWh, PUE 1.1, carbon intensity 0.610 → 70 tCO2eq.”
    - Estimates are conservative lower bounds and exclude embodied emissions.

- Do the experiments support the claims?
  - Competitiveness: Yes—Table 3 shows OLMo‑7B within ~1.2 points of Llama‑2‑7B and tied with MPT‑7B on average. Figure 2 shows strong in-distribution fit.
  - Scientific utility: Strong—open data, exact data order, and 500+ checkpoints allow controlled studies. The decontaminated Paloma evaluation and in-loop curves (Figure 1) demonstrate robust methodology.
  - Adaptation viability: Yes—Table 4 shows large capability and safety gains after SFT and DPO. The models and code are released for reproduction (Section 5).

- Ablations, failure cases, robustness
  - Ablations are performed “in-loop” (Section 2.4) for architecture, initialization, optimizer, schedule, and data mixtures; specific quantitative ablation tables are not included, but final choices (non‑parametric LN, SwiGLU, LR schedule) are motivated by stability/throughput and Figure 1’s gains.
  - Robustness caveats:
    - Fit is weaker on curated non‑web domains (Figure 2).
    - Some benchmarks show unstable signal during training (Figure 4), warning against over-reliance on those tasks for model selection.

- Conditions and trade‑offs
  - Web-heavy data (88.8% Common Crawl for OLMo) boosts C4‑like performance but can reduce sample efficiency on curated or academic text (Figure 2).
  - The LR‑to‑zero trick aids end-task accuracy late in training but costs extra compute.

## 6. Limitations and Trade-offs
- Data and coverage (Limitations; Sections 2.2, 4.2)
  - English-focused; multilinguality is not addressed. Web‑heavy Dolma mix favors web distributions (Figure 2), at the expense of some curated corpora fit.
  - Despite filtering, large-scale pretraining likely includes toxic, personal, or copyrighted content; filters are imperfect (Limitations: Data).

- Modeling and training choices
  - Context length fixed at 2048; no long-context scaling or attention variants like GQA/MQA (Table 5).
  - Full attention may limit scalability to longer contexts compared to GQA/MQA in some contemporaries (Table 5).
  - Some runs trained past 2T tokens by starting a second epoch. The paper argues repeat effects are negligible, but it remains an assumption (Section 3.3).

- Evaluation realism and noise
  - Zero-shot multiple-choice tasks diverge from chat-style real-world use; results can be noisy and sensitive to prompt normalization (Section 4.1; Limitations: Evaluation).
  - Some tasks used as auxiliary signals showed unstable progress and class-imbalance pathologies (Figure 4; Table 7).

- Environmental accounting
  - Carbon estimates are lower bounds and exclude embodied emissions and miscellaneous development costs (Appendix B).

- Adaptation dependencies
  - Post-training relies on TÜLU mixes and distilled preference data; mixtures may be tuned for LLaMA-family models and may not be optimal for OLMo (Limitations: Adaptation).
  - Despite DPO’s benefits, safety and hallucinations are improved but “not perfect” (Limitations: Adaptation).

## 7. Implications and Future Directions
- How this work shifts the landscape
  - Establishes a new standard for openness: fully reproducible pretraining at competitive quality, with data, order, code, logs, checkpoints, and transparent evaluation (Section 5). This enables the community to study causal links between data/process and model behavior—something largely infeasible with closed or partially open releases.

- Research directions enabled
  - Data-centric LM science:
    - Run controlled experiments on Dolma variants: adjust source mixing, dedup strategies, and filters; then quantify capability/safety trade-offs via Catwalk and decontaminated Paloma (Sections 2.2, 2.4).
    - Use “What’s In My Big Data?” (WIMBD) for dataset audits (Section 2.2).
  - Training and optimization:
    - Compare LayerNorm variants, activations, LR schedules, and gradient clipping strategies using the provided code and checkpoints (Sections 2.1–3.2).
    - Study cross‑vendor portability and performance tuning on AMD/NVIDIA stacks (Section 3.4).
  - Evaluation methodology:
    - Expand decontamination methods (especially for code), add robust, low-noise tasks, and track learning dynamics using intermediate checkpoints (Figure 2; Appendix C).
  - Post-training and safety:
    - Develop OLMo‑specific instruction mixtures; reduce reliance on distilled data; explore RLHF or alternative preference methods beyond DPO; assess safety and calibration improvements (Section 4.3; Limitations: Adaptation).
  - Scaling and capabilities:
    - Extend to more sizes, longer contexts, and modalities; explore mixture‑of‑experts architectures or multilingual pretraining (Conclusion: “different model sizes, modalities, datasets, safety measures, and evaluations”).

- Practical applications
  - A strong, inspectable 7B base model for:
    - Enterprise/private deployments where data provenance and reproducibility are required.
    - Domain adaptation with transparent data and training logs.
    - Safety research and policy compliance audits using open evaluation code and decontaminated benchmarks.
  - Released artifacts reduce duplicated compute and environmental costs for downstream users (Ethics Statement; Section 5).

- Notable post‑paper update
  - After improving data and training setup post-release, the team reports MMLU “improved by 24 points to 52%” (Conclusion; footnote 9). This indicates that, within the OLMo framework, data/process refinements can lead to sizable capability gains—precisely the kind of iteration the release is designed to enable.

> Bottom line: OLMo delivers a competitive, fully reproducible LM stack. Its most important contribution is not a single benchmark number, but a research substrate—open data, code, checkpoints, and decontaminated evaluation—that makes rigorous, cumulative LM science possible.
