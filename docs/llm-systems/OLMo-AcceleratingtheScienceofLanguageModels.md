# OLMo : Accelerating the Science of Language Models

**ArXiv:** [2402.00838](https://arxiv.org/abs/2402.00838)

## 🎯 Pitch

OLMo sets a new standard for transparency in AI by releasing not only model weights but the full pretraining dataset (Dolma), curation tools, training and evaluation code, intermediate checkpoints, and logs—making every stage of large language model development accessible and reproducible. This unprecedented end-to-end openness enables rigorous scientific study of how data choices, architecture, and optimization shape model behavior, lowering barriers for researchers worldwide and dramatically advancing the responsible development and understanding of powerful language models.

---

## 1. Executive Summary (2-3 sentences)
OLMo introduces a fully open, end-to-end framework for building and studying large language models (LLMs), releasing not just model weights but also the complete pretraining dataset (Dolma), data curation tools, training/evaluation code, intermediate checkpoints, and logs under Apache 2.0. The 7B and 1B models are competitive with widely used open models, and the release enables rigorous, reproducible science on how data, design choices, and optimization affect model behavior (Sections 1, 2, 5).

## 2. Context and Motivation
- Problem addressed:
  - The most capable LLMs are usually closed: training data, precise architectures, and training details are hidden, limiting scientific understanding of model behavior, biases, and risks (Section 1).
  - Even “open” releases often omit crucial ingredients such as full training data, curation code, and intermediate checkpoints (Introduction; comparisons with Falcon, LLaMA, MPT, Pythia, BLOOM).

- Why this matters:
  - Without data and training details, researchers cannot rigorously study how data composition, scaling, optimization schedules, or ablations change behavior, nor can they run fair decontaminated evaluations (Sections 1, 2.4).
  - Open artifacts reduce duplicated compute and environmental costs—others can build on a common, audited base (Ethics; Appendix B).

- Prior approaches and their gaps:
  - Partial openness: weights + brief reports (e.g., Mixtral 8x7B); weights + adaptation instructions (LLaMA); detailed training setup but not data (MPT); partially released data (Falcon); most open to date: Pythia and BLOOM, which released code, checkpoints, and some data (Section 1).
  - Persistent gap: lack of truly end-to-end reproducibility across data, training, evaluation, and adaptation.

- How this work positions itself:
  - OLMo releases the entire stack—data (Dolma), data curation code and analyzers (WIMBD), training and evaluation frameworks (Catwalk for downstream tasks; Paloma for perplexity), intermediate checkpoints, and logs—so the community can replay, vary, and analyze every stage (Sections 2.2, 2.4, 5).

## 3. Technical Approach
This section explains how OLMo is built, trained, adapted, and evaluated.

- Model architecture (Section 2.1; Table 1):
  - Family: decoder-only transformer, with 1B and 7B variants trained on ≥2T tokens each.
  - Architectural choices aimed at stability and throughput:
    - No bias terms: removes additive biases in layers to improve training stability.
    - Non-parametric LayerNorm: uses LayerNorm without learned gain/bias parameters to reduce overhead and avoid instabilities seen with alternatives such as RMSNorm in their setup.
    - `SwiGLU` activation: a gated activation (two linear projections, one gated by a Swish nonlinearity) that improves expressiveness over ReLU; the activation hidden size ≈ 8/3 times the model dimension, rounded to multiples of 128 for throughput (footnote in Section 2.1).
    - `RoPE` (rotary positional embeddings): position information is encoded as rotations in the query/key space, replacing learned absolute positional embeddings, improving extrapolation to longer contexts in practice (Section 2.1).
    - Tokenizer: a GPT-NeoX BPE with added tokens to mask PII; vocab size 50,280, but embeddings padded to 50,304 (multiple of 128) to maximize GPU throughput (Section 2.1).
    - Weight tying: on for the 1B model, off for 7B (Table 1; “Weight Tying”).

- Optimizer and training schedule (Sections 3.2 and 3.3; Table 1):
  - Optimizer: AdamW with betas (0.9, 0.95) and epsilon 1e-5, weight decay 0.1.
  - Learning rate: linear warmup, then linear decay to 10% of peak LR; in late-stage tuning they further decay to zero over the final 1,000 steps, which improves final performance (Figures 1 and Section 4).
  - Gradient clipping: global L2 norm capped at 1.0 to stabilize updates (Section 3.2).
  - Batch and context: global batch ≈ 4M tokens per step; sequence length 2,048 (Sections 3.1; 3.3; Table 5).
  - Instance construction: concatenate document tokens with an EOS token, chunk into 2,048-token sequences, shuffle in a fixed order reproducibly; at least one epoch over 2T tokens, some models start a second shuffled pass (Section 3.3).

- Data pipeline: Dolma (Section 2.2; Table 2):
  - Dolma is a 3T-token open corpus assembled via language filtering, quality/content filtering, de-duplication, multi-source mixing, and tokenization.
  - Composition (Table 2): Common Crawl (2,180B tokens), GitHub code (342B), Reddit (80B), Semantic Scholar (57B), Project Gutenberg (5.2B), Wikipedia (3.7B)—total ≈ 2,668B tokens with GPT-NeoX tokenizer.
  - Tools: open-source curation and analysis (WIMBD) to reconstruct training order and audit data seen at each step (Sections 2.2 and 5).

- Distributed training and precision (Section 3.1):
  - `ZeRO` via PyTorch `FSDP` shards model params and optimizer state across GPUs to fit 7B models with 4,096 tokens per GPU micro-batch.
  - Mixed precision: most ops in `bfloat16`; numerically sensitive ops (e.g., softmax) in full precision to avoid instabilities. Parameters/optimizer state are stored in FP32 and cast to bfloat16 on-the-fly for computation; gradients reduced in FP32 (Section 3.1).

- Hardware and cross-vendor reproducibility (Section 3.4):
  - Two clusters: AMD MI250X on LUMI (renewable energy) and NVIDIA A100-40GB on MosaicML. Despite minor batch differences, performance converges closely by 2T tokens (Section 3.4).

- Evaluation framework (Section 2.4):
  - `Catwalk`: a public evaluation harness for downstream tasks; OLMo runs periodic “in-loop” evaluations every 1,000 steps (~4B tokens) to guide architectural and optimization decisions (Section 2.4).
  - `Paloma`: a perplexity benchmark with 585 domains across 18 sources; OLMo removes any training documents with leaked evaluation paragraphs (“decontamination”) to prevent inflated scores (Section 2.4).

- Adaptation for chat and safety (Sections 2.3, 4.3; Appendix D/E):
  - Instruction-tuning (SFT) using the TÜLU v2 mix (a curated, mostly open set of instruction–response pairs).
  - Preference optimization with `DPO` (Direct Preference Optimization): trains the model to prefer “chosen” over “rejected” responses using a fixed β (0.1). For DPO, they use a cleaned “UltraFeedback” dataset variant (Appendix D).
  - Hyperparameters: SFT with LR 2e-6, 3 epochs, 3% warmup then linear cooldown; DPO with LR 5e-7, β=0.1, 3 epochs, 10% warmup (Appendix D).

- Definition notes:
  - `Decontamination`: removing training examples that overlap with evaluation test data to avoid unfairly low perplexity or inflated accuracy.
  - `Bits per byte (BpB)`: perplexity-like metric independent of tokenizer; lower is better and reflects better fit to the text distribution (Section 4.2).
  - `Catwalk/Paloma/TÜLU/DPO/WIMBD`: open tools/datasets introduced or adopted here for evaluation (Catwalk, Paloma), instruction tuning (TÜLU), preference optimization (DPO), and dataset analysis (WIMBD).

## 4. Key Insights and Innovations
- End-to-end openness as a scientific instrument (Sections 1, 5):
  - What’s new: OLMo releases everything—data (Dolma), curation code, training/eval code, 500+ intermediate checkpoints, logs—under Apache 2.0, not just final weights.
  - Why it matters: enables replayable, controlled experiments on data composition, optimizer schedules, and architecture, and makes decontaminated, cross-model comparisons feasible.

- Dolma: a large, public, multi-source pretraining dataset with tooling (Section 2.2; Table 2):
  - What’s new: a scaled, openly licensed corpus plus curation/analysis tools to reconstruct exact training orders and run data ablations.
  - Why it matters: most prior open LLMs did not release the exact training data. Dolma enables controlled studies of how filtering, deduplication, and mixing affect learned capabilities.

- Decontaminated, domain-diverse perplexity evaluation with Paloma (Section 2.4; Figure 2):
  - What’s new: explicit removal of test paragraphs from pretraining data and evaluation across 585 domains aggregated into sources.
  - Why it matters: prevents overestimation of out-of-sample fit and reveals nuanced “in-distribution” vs. “out-of-distribution” sample efficiency differences (e.g., strong on web-like text such as C4; weaker on curated sources like WikiText-103).

- Cross-hardware training parity (Section 3.4; Table 6 in Appendix B):
  - What’s new: training on both AMD MI250X and NVIDIA A100 clusters, with nearly identical outcomes.
  - Why it matters: strengthens claims of reproducibility and broadens accessible infrastructure for the community.

- Practical training insight: end-phase LR decay to zero helps (Figure 1, Section 4):
  - Observation: “a sharp upward tick” appears on many downstream tasks between the second-to-last and last checkpoints when linearly decaying LR to zero over the final 1,000 steps.
  - Significance: a simple schedule tweak yields measurable gains without extra data.

These are primarily fundamental enablers for scientific study (data/code/checkpoints/tools), paired with pragmatic training choices rather than brand-new architectures.

## 5. Experimental Analysis
- Evaluation methodology (Section 2.4; 4.1; 4.2; 4.3):
  - Downstream tasks (zero-shot, rank classification): 8 core tasks—`arc_e/challenge`, `boolq`, `hellaswag`, `piqa`, `sciq`, `winogrande`—with dataset-specific normalization strategies (per-token, per-character, unconditional, or none; Section 4.1).
  - Intrinsic modeling: Paloma BpB across 11 public sources (C4, mC4 (en), WikiText-103, PTB, RedPajama, Falcon-RefinedWeb, Dolma, M2D2 S2ORC, M2D2 Wikipedia, C4 100 Domains, Dolma 100 Subreddits) and also individually by source (Section 4.2; Figure 2).
  - In-loop evaluation: run every 1,000 steps to guide design choices (Section 2.4).
  - Adaptation evaluation: MMLU, AlpacaEval, ToxiGen, and TruthfulQA for chat/safety after SFT and DPO; also the full TÜLU suite (Section 4.3; Table 4; Table 8).

- Main quantitative outcomes:
  - Core downstream comparisons (Table 3; 7B scale):
    > “OLMo-7B avg: 69.3; Llama 2 7B: 70.5; MPT-7B: 69.8; LLaMA 7B: 69.6; Falcon-7B: 70.3; RPJ-INCITE-7B: 66.6; Pythia-6.9B: 63.0.”  
    Interpretation: OLMo-7B is competitive with its peers; within 1–1.5 points of the strongest 7B baselines on this suite.
  - Training dynamics (Figure 1):
    > “We can see the benefit of decaying LR to 0 in the final 1000 steps of training on most tasks.”  
    Many tasks (e.g., `arc_c`, `hellaswag`, `piqa`, `sciq`) show upward jumps near the end, supporting the schedule choice.
  - Intrinsic modeling (Figure 2):
    > “Sources Combined: OLMo-7B follows similar scaling trends and is competitive; on C4, OLMo-7B overtakes all other models; less sample-efficient on WikiText-103, M2D2 S2ORC, and M2D2 Wikipedia.”  
    This suggests better fit to web-scraped distributions (where Dolma is 88.8% Common Crawl) and weaker fit to curated, scarcer distributions.
  - Adaptation results (Table 4; Table 8):
    > “OLMo-7B (base) MMLU 28.3. After SFT: 47.3; after SFT+DPO: 46.2. ToxiGen (% toxic) drops from 81.4 (base) to 14.4 (SFT) to 1.7 (SFT+DPO); TruthfulQA Informative+Truthful rises from 31.6 (base) to 41.2 (SFT) to 52.0 (SFT+DPO). AlpacaEval win-rate rises from – to 57.0 (SFT) to 69.3 (SFT+DPO).”  
    Compared against other 7B chat models, `OLMo+SFT+DPO` beats most except TÜLU 2 variants on some axes (Table 4).

- Baselines and fairness:
  - They evaluate against public 7B-ish baselines: LLaMA (7B), Llama-2 (7B), MPT-7B, Falcon-7B, Pythia-6.9B, RPJ-INCITE-7B (Table 3), and chat variants for adaptation (Table 4).  
  - Paloma is decontaminated for OLMo to avoid unfairly low perplexity (Section 2.4); other models may have contamination risks that underestimate their BpB.

- Ablations and robustness:
  - “In-loop” evaluation (every 1k steps) enabled ablations on architecture (LayerNorm choice), optimizers, schedules, and data mixtures (Section 2.4), though detailed per-ablation numbers are not tabulated.
  - Appendix C shows additional tasks with unstable trends (Figure 4), cautioning against over-reliance on those signals.  
    > “The performance of these additional end-tasks was unstable and provided limited signal during model development.”

- Environmental accounting (Appendix B; Table 6):
  > “OLMo-7B MI250X: 135 MWh with carbon intensity ~0 on LUMI (renewables)—0 tCO2eq; OLMo-7B A100-40GB: 104 MWh at 0.610 kg/kWh—~70 tCO2eq.”  
  This transparency supports the paper’s claim of enabling lower duplicated emissions via reuse.

- Do the experiments support the claims?
  - Yes for competitiveness: OLMo-7B is in the same band as LLaMA/Llama-2/MPT on core zero-shot tasks (Table 3).
  - Yes for utility of openness: the combination of decontaminated Paloma, intermediate checkpoints, and in-loop curves provides unusually rich, reproducible evidence (Figures 1–2; Section 5).
  - Adaptation: substantial capability and safety improvements from SFT/DPO are clearly quantified (Tables 4 and 8).

- Mixed or conditional results:
  - OLMo’s Paloma sample efficiency varies by domain—strongest where pretraining distribution matches (C4) and weaker on curated sources (WikiText-103, M2D2), highlighting data–evaluation alignment effects (Figure 2).

## 6. Limitations and Trade-offs
- Data scope and content (Limitations: Data; Section 2.2):
  - Primarily English; multilingual capability is not addressed.
  - Large-scale web and social data likely contain toxic or copyrighted text and personal information despite filtering; no perfect removal method exists.

- Distribution match and sample efficiency (Figure 2):
  - OLMo-7B fits web-like distributions well (C4) but is less sample-efficient on curated sources (e.g., Wikipedia/ArXiv-proximate sources like M2D2), suggesting a trade-off based on data composition.

- Evaluation representativeness and noise (Limitations: Evaluation; Appendix C):
  - Benchmarks are mostly narrow, structured multiple-choice tasks; real user chat behavior is broader.
  - Some tasks have unstable or class-imbalance-driven metrics, making them weak signals for training-time decision-making.

- Adaptation dependencies (Limitations: Adaptation; Section 4.3):
  - TÜLU mixes were designed with LLaMA-family models in mind; performance may not be optimal for OLMo without retuning the mixture.
  - Preference datasets include distilled outputs from other LMs, entangling OLMo’s post-training behavior with those sources.

- Compute and reproducibility costs:
  - While artifacts are open, reproducing full pretraining still requires substantial compute (239 MWh estimated across runs; Appendix B), limiting who can fully replicate training.

- Architectural novelty:
  - The architecture combines established components (RoPE, SwiGLU, non-param LayerNorm) rather than introducing fundamentally new mechanisms; the innovation is in the openness and tooling.

## 7. Implications and Future Directions
- How this changes the landscape:
  - Establishes a new norm for openness: not just weights, but full data, tools, and step-by-step reproducibility, enabling scientific study of LLMs akin to controlled experiments.
  - Provides a decontaminated, domain-diverse perplexity benchmark (via Paloma) and public tools (Catwalk, WIMBD) that others can adopt, improving fairness and rigor in evaluation.

- What research this enables:
  - Data-centric LLM science: quantify how filtering, deduplication, and source mixing alter capabilities; run controlled “swap-in/swap-out” data ablations using Dolma and curation code (Section 2.2).
  - Training dynamics: study how LayerNorm variants, activation functions, LR schedules, or optimizer settings shape learning curves using the released checkpoints and logs (Sections 2.1, 3.2, 5).
  - Robustness and decontamination: standardize decontaminated evaluations and explore how much contamination affects different metrics across domains (Section 2.4; Figure 2).
  - Cross-hardware/cross-stack replicability studies: validate parity across vendors and software stacks, lowering infrastructure lock-in (Section 3.4).

- Practical applications:
  - As-is base models for research and product prototyping—OLMo-7B is competitive on core benchmarks (Table 3).
  - Instruction-tuned chat assistants with improved safety and truthfulness: `OLMo+SFT+DPO` brings ToxiGen toxicity down to 1.7% and boosts TruthfulQA Informative+Truthful to 52.0 (Tables 4, 8).
  - Auditable, compliant deployments: open data lineage and decontamination make it easier to meet transparency and risk auditing requirements.

- Future directions suggested by the paper (Conclusion; Limitations):
  - Broader data and multilingual training; new modalities; improved safety training and evaluations.
  - Continued releases with better results (they note post-paper MMLU gains up to ~52% through improved data/training, Conclusion).
  - More realistic user-centered evaluations that track interactive performance, not only zero-shot multiple choice.

Overall, OLMo’s primary contribution is not a new model trick but a complete, open, and replicable ecosystem for LLM science—demonstrating competitive performance while making it possible for the community to study, stress-test, and improve LLMs with full visibility from data to deployment.
