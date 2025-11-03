# BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

**ArXiv:** [2211.05100](https://arxiv.org/abs/2211.05100)

## 🎯 Pitch

BLOOM introduces a groundbreaking, 176-billion-parameter multilingual language model—openly released to the public and collaboratively developed by hundreds of researchers. By supporting 46 natural and 13 programming languages and providing extensive documentation, training data transparency, and a Responsible AI License, BLOOM democratizes access to cutting-edge large language model technology, enabling the global community to advance research, foster inclusivity, and drive responsible AI innovation far beyond the boundaries set by proprietary alternatives.

---

## 1. Executive Summary
BLOOM is a 176B-parameter, open-access, multilingual language model trained on the 1.6TB ROOTS corpus spanning 46 natural and 13 programming languages (Section 3.1; Table 1; Figure 3). It demonstrates competitive zero/few-shot performance across English and multilingual tasks, and its zero-shot abilities substantially improve after multilingual multitask finetuning into BLOOMZ using the xP3 prompt collection (Section 3.1.4; Figure 11). The project’s significance lies in opening large-scale LLM research and deployment through public release under a Responsible AI License, detailed documentation, and a reproducible engineering stack (Sections 3.4–3.6).

## 2. Context and Motivation
- Problem/gap addressed
  - Most very large language models (LLMs) are developed by a few resource-rich organizations, often closed-source, limiting reproducible research, multilingual coverage, and societal oversight (Introduction; Section 2; “Social Limitations of LLM Development,” pp. 5–6).
  - Existing LLMs over 100B parameters are primarily monolingual or English-centric; multilingual alternatives exist but are less open or smaller (Section 2.1; references to GPT-3, PaLM, Gopher, OPT, XGLM).
- Why this matters
  - Real-world: Multilingual support affects global access to AI capabilities; closed models concentrate power, inhibit oversight, and risk misaligned deployments (Sections 2.1–2.2).
  - Scientific: Open artifacts (code, data processes, training logs) enable verification, ablation, and methodological advances; multilingual benchmarks reveal generalization dynamics not visible in English-only evaluation.
- Prior approaches and shortcomings
  - Closed large models (e.g., GPT-3) showed emergent few-shot behavior but lacked open access, limiting community experimentation (Sections 2.1; related work).
  - Some multilingual models (e.g., XGLM, mT5) exist, but training data governance, transparency, and responsible licensing are often limited (Sections 2.1–2.2; 3.1.1; 3.6).
- Positioning
  - BLOOM offers: a) an open 176B decoder-only Transformer trained on a broad multilingual corpus; b) a documented data governance pipeline (Section 3.1.1; Figure 2); c) careful tokenizer and architecture choices validated by ablations (Sections 3.2–3.3); d) an open evaluation harness with prompts (Sections 4.1.1–4.1.2); and e) a released instruction-tuned variant (BLOOMZ) for zero-shot task generalization (Section 3.1.4; Figure 11).

## 3. Technical Approach
Step-by-step overview of how BLOOM is built and trained:

- Data governance and sourcing (Section 3.1; Figure 2)
  - ROOTS corpus: 498 datasets totaling 1.61 TB across 59 languages (46 natural, 13 programming) (Table 1; Figure 3).
  - Governance: Explicit data provider permissions when possible, traceability by keeping sources separate until late, and “composite release” mirroring source licenses (Section 3.1.1).
  - Sourcing: Community-curated catalog plus pseudocrawl from Common Crawl via OSCAR 21.09 (38% of corpus), and GitHub code via BigQuery (Sections 3.1.2–3.1.3).
  - Quality filtering: Language-expert-tuned filters target “text written by humans for humans” (no blanket “offensive” content filters); near-duplicate removal and PII redaction (regex-based) on higher-risk sources like OSCAR (Section 3.1.3).
  - Why this design: Human-in-the-loop improvements avoid known harms from automatic blocklists and Reddit-based quality heuristics (Section 3.1 Motivation, citing Dodge et al., 2021; Johnson et al., 2022).

- Prompt collections for instruction tuning (Section 3.1.4)
  - xP3: Prompts for 83 datasets across 46 languages and 16 tasks; language distribution mirrors ROOTS (Figure 4).
  - Process: Built with PromptSource, including metadata such as input/target languages; machine-translated prompt variants (xP3mt) created for analysis.
  - Outcome: Finetuning BLOOM on xP3 yields BLOOMZ, a strongly improved zero-shot generalizer across multilingual tasks (Figure 11).

- Model architecture (Sections 3.2–3.2.3; Figure 5)
  - Choice: Causal decoder-only Transformer with autoregressive objective (Eq. 1 in Section 2.1). This design was validated via experiments showing best zero-shot generalization compared to encoder-decoder and masked/prefix objectives at smaller scales (Section 3.2.2; Wang et al., 2022a).
  - Modifications:
    - `ALiBi positional embeddings`: add a distance-based linear bias directly to attention scores, improving training smoothness and downstream performance, and enabling length extrapolation (Section 3.2.3).
    - `Embedding LayerNorm`: an extra normalization after the token embeddings improved training stability during preliminary very-large-scale runs; retained for BLOOM despite potential small zero-shot penalty found at small scale, prioritizing stability (Section 3.2.3).
  - Why this design: Ablations at 1.3B–6.7B guided choices that scale; alternatives like Mixture-of-Experts or state-space models were excluded due to maturity or tooling constraints at the time (Section 3.2.1).

- Tokenization (Section 3.3; Table 2)
  - `Byte-level BPE`: ensures lossless coverage of all bytes; maximizes subword sharing across languages (Section 3.3).
  - Vocabulary size: 250,680 tokens (divisible by 128 for GPU efficiency and by 4 for tensor parallelism), with 200 reserved placeholders (Section 3.3).
  - Pre-tokenizer: regex split preserving whitespace and line breaks (important for code); avoids English-centric contractions and digit splits to prevent issues in Arabic and code (Section 3.3).
  - Validation metric: `fertility` (avg. subwords per word); target is within 10 percentage points of strong monolingual tokenizers; results reported in Table 2.

- Distributed training & numerics (Sections 3.4.1–3.4.5; Figure 6)
  - Hardware: 384 × A100 80GB GPUs on the Jean Zay supercomputer over ~3.5 months (1,082,990 GPU-hours) (Section 3.4.1).
  - Framework: Megatron-DeepSpeed with “3D parallelism”:
    - `DP` (data parallelism): replicate model across nodes;
    - `TP` (tensor parallelism): shard layers horizontally across GPUs;
    - `PP` (pipeline parallelism): split model depth across GPUs (Section 3.4.2; Figure 6).
  - `ZeRO` Stage 1: shards optimizer states to reduce memory (Section 3.4.2).
  - `bfloat16` mixed precision: avoids float16 overflow instabilities; use float32 for sensitive ops; improved stability at 100B+ scale (Section 3.4.3).
  - Fused CUDA kernels: combine operations like scaling/masking/softmax and bias+GeLU to reduce memory-bound overhead (Section 3.4.4).
  - Practicalities: frequent checkpointing; deadlock and IO issues noted but manageable (Section 3.4.5; training chronicles link).

- Training schedule (Section 3.5; Table 3)
  - Pretraining target: ~341B tokens (ROOTS) + 25B repeated after revised scaling laws (Hoffmann et al., 2022).
  - 176B configuration: global batch size 2048; LR 6e-5 with cosine decay and 375M-token warmup (end of decay not reached due to token budget); weight decay 1e-1; gradient clipping 1.0; no dropout (Table 3).
  - Multitask finetuning (BLOOMZ): 13B tokens; LR 2e-5; constant schedule; best checkpoint chosen on validation; benefits plateau after 1–6B tokens (Table 3).
  - Specialized embedding finetunes: SGPT-BLOOM variants for retrieval/STS using contrastive training (Section 3.5; links to models).

- Carbon accounting (Section 3.5.1; Table 4)
  - Training energy: 433 MWh; resulting emissions 25 tons CO2eq due to France’s low-carbon energy (57 gCO2eq/kWh), lower than grids used for comparable models (Table 4).
  - LCA estimate of total project emissions: ~81 tons CO2eq including manufacturing (14%), training energy (30%), and idle cluster consumption (55%) (Section 3.5.1).
  - API deployment example: ~20 kg CO2eq/day on a GCP 16-GPU instance (Section 3.5.1).

- Release & licensing (Section 3.6)
  - Model Card with intended/out-of-scope uses and limitations.
  - Responsible AI License (RAIL): free use with behavioral-use restrictions; code under Apache 2.0.

## 4. Key Insights and Innovations
- Open, multilingual LLM at 176B scale with governance and transparency
  - What’s new: A publicly released 176B multilingual model with a traceable, curated corpus (ROOTS) and composite data release (Sections 3.1.1–3.1.2).
  - Significance: Enables replication, examination of multilingual behaviors, and real-world deployment under RAIL (Section 3.6). This is more than incremental openness; it materially changes who can study and build on an LLM of this scale.

- Architecture choices validated for zero-shot generalization
  - What’s new: Systematic comparison of objectives/architectures (decoder-only causal LM found best for zero-shot immediately post-pretraining) and adoption of `ALiBi` for positional encoding and an extra `Embedding LayerNorm` for stability (Sections 3.2.2–3.2.3).
  - Significance: Provides methodological guidance for future large-model training where ablations at full scale are infeasible; the ALiBi choice is a design-level improvement beyond minor parameter tweaks.

- Engineering recipe for training 100B+ multilingual models on public HPC
  - What’s new: A detailed training stack (Megatron-DeepSpeed, 3D parallelism, bfloat16 mixed precision, fused kernels) and concrete operational lessons (Sections 3.4.1–3.4.5).
  - Significance: Lowers the barrier for other academic consortia to reproduce large-scale training.

- Multilingual instruction tuning at scale (BLOOMZ via xP3)
  - What’s new: A multilingual prompt collection (xP3) mirroring ROOTS and a recipe that substantially boosts zero-shot generalization across many languages and tasks (Section 3.1.4; Figure 11).
  - Significance: Demonstrates that instruction tuning benefits extend robustly to multilingual settings, not just English.

- Carbon-aware training and life-cycle accounting
  - What’s new: Detailed energy/emissions reporting and comparisons to GPT-3, Gopher, and OPT (Table 4), including idle and manufacturing footprints (Section 3.5.1).
  - Significance: Establishes a transparent baseline and shows how grid carbon intensity strongly affects emissions for LLM training.

## 5. Experimental Analysis
- Evaluation methodology (Section 4.1)
  - Prompting: Multiple human-written prompts per task created and peer-reviewed in PromptSource; tests aim to reflect realistic zero/one-shot usage (Section 4.1.1; Table 5 shows MT prompt examples).
  - Framework: An extended Language Model Evaluation Harness integrated with PromptSource; open-sourced (Section 4.1.2).
  - Tasks and datasets:
    - English understanding: SuperGLUE subset (Ax-b, Ax-g, BoolQ, CB, WiC, WSC) (Section 4.1.3).
    - Machine Translation: WMT14 en↔fr, en↔hi (Table 6), DiaBLa bilingual dialogues (Table 7), and Flores-101 devtest across many directions (Table 8).
    - Summarization: WikiLingua in 9 languages using ROUGE-2/L and Levenshtein with a multilingual tokenizer (Section 4.1.3; Figure 9).
    - Code Generation: HumanEval pass@k (Table 9).
    - HELM benchmark: 5-shot English tasks (Figure 10).
    - Multitask finetuning generalization: XNLI, XWinograd, XCOPA, XStoryCloze (Figure 11).
    - Embeddings: MTEB subsets (MASSIVE classification; STS22) (Table 10).
    - Probing: UD-based morphosyntactic probing across 17 languages (Table 12; Figure 12).
    - Bias: CrowS-Pairs (English, French) with AR-LLM adaptation (Figure 13; Table 14).

- Main quantitative results
  - SuperGLUE zero-/one-shot (Figures 7–8):
    > One-shot: BLOOM-176B matches or exceeds OPT-175B on Ax-b, CB, WSC, WiC; both families show slight gains with scale; zero-shot BLOOM slightly trails but narrows in one-shot.
  - Machine Translation:
    - WMT14 (Table 6): One-shot BLEU (best prompts) reaches 34.2 (en→fr) and 35.4 (fr→en); Hindi directions improve with shots but remain lower (e.g., hi→en 25.8).
    - Overgeneration and wrong-language outputs are common in zero-shot but improve with one-shot (Section 4.3.1).
    - DiaBLa (Table 7): Using previous-dialogue context as the single shot improves BLEU/COMET after truncation; qualitative evidence shows context helps disambiguation (Section 4.3.2).
    - Flores-101 (Table 8): One-shot spBLEU competitive with supervised M2M-100 (615M) on many directions; strong results across high-resource pairs and Romance languages; poor for Swahili↔Yoruba due to scarce training data.
  - Summarization (Figure 9):
    > Across 9 languages, BLOOM models consistently outperform OPT-175B in one-shot ROUGE-2; performance increases with BLOOM scale.
  - Code generation (Table 9):
    > BLOOM-176B achieves pass@1 = 15.52%; behind code-finetuned Codex-12B (28.81%) but in range of text-only GPT-NeoX-20B (15.4%). BLOOMZ gains are modest, reflecting limited code tasks in xP3.
  - HELM (Figure 10):
    > BLOOM’s 5-shot English accuracy is comparable to earlier monolingual models (e.g., GPT-3 davinci v1) but trails the very latest instruction-tuned/monolingual models (e.g., InstructGPT, OPT).
  - Multitask finetuning (Figure 11):
    > BLOOMZ (finetuned on xP3) dramatically improves zero-shot accuracy over untuned BLOOM and monolingual T0 on multilingual NLI (XNLI), coreference (XWinograd), and sentence-completion datasets (XCOPA, XStoryCloze).
  - Embeddings (Table 10):
    > SGPT-BLOOM-7.1B-msmarco achieves strong or SOTA scores across multiple languages on MASSIVE classification and STS22; however, it is much larger than common multilingual embedding models (MiniLM/MPNet).
  - Probing (Table 12; Figure 12; Section 4.9.2):
    > BLOOM-1.7B generally matches/exceeds 176B on averaged morphosyntactic probing F1 across languages; both outperform count-based TF-IDF baselines. Stronger signals for features like `Mood` and `Person`; weaker for `Case` and others. Statistical tests show results correlate with language family and dataset sizes; 176B is more stable across languages.
  - Bias (Figure 13; Table 14):
    > CrowS-Pairs prompt-accuracy near 50% in English and French overall; some bias categories deviate slightly (e.g., religion > 53%), but results vary and are limited to two languages with adapted methodology.

- Do experiments support claims?
  - Competitive multilingual performance: Supported by Flores-101 and WikiLingua results (Tables 8; Figure 9). WMT14 shows good one-shot capability but trails strong supervised MT (Section 4.3.1).
  - Improved zero-shot via instruction tuning: Strongly supported by Figure 11 across multiple multilingual tasks.
  - Engineering feasibility and stability: Training completed with one recoverable loss spike; bfloat16 stabilized training (Section 3.4.3–3.4.5).
  - Carbon efficiency: Table 4 shows lower CO2eq than OPT/GPT-3 due to low-carbon grid, not lower energy use per se (BLOOM uses 433 MWh vs OPT’s 324 MWh).

- Ablations and robustness
  - Architecture/Objective ablations at 1.3B–6.7B guided choices (Section 3.2.1–3.2.2); specific zero-shot task aggregates used (EAI-Eval, T0-Eval).
  - Prompt diversity and one-shot reduce variance and improve performance (Figure 7).
  - Failure cases: Zero-shot MT overgeneration and wrong language (Section 4.3.1); weak performance for very underrepresented languages (Table 8a); code generation behind code-finetuned baselines (Table 9).

## 6. Limitations and Trade-offs
- Data and multilingual coverage
  - ROOTS includes many languages, but representation is uneven (Figure 3; Table 1). Performance degrades for extremely low-resource languages (e.g., Swahili↔Yoruba in Table 8a).
  - Composite release means only 223/498 components are directly accessible due to licensing/privacy (Section 3.1.1), complicating full data replication.
- Architecture and training
  - Ablations were at smaller scales (1.3B/6.7B); some choices (e.g., Embedding LayerNorm penalty on zero-shot) may not perfectly extrapolate to 176B (Section 3.2.1, 3.2.3).
  - Training did not reach end of LR decay due to token budget (Table 3), so further gains may be untapped; repeated data after 341B tokens could bias memorization.
- Capabilities
  - English-only benchmarks (HELM) show BLOOM trailing the very latest monolingual instruction-tuned models (Figure 10).
  - Code generation lags code-finetuned models (Table 9); xP3 lacks pure code completion tasks (Section 4.5).
  - Zero-shot MT can fail with overlong outputs or wrong target language; mitigated by few-shot prompting (Section 4.3.1).
- Bias and societal risks
  - Bias evaluation limited to adapted CrowS-Pairs in English/French; methodology differences and dataset validity concerns limit conclusions (Section 4.10, “Limitations”).
  - RAIL adds safeguards but may constrain some uses; compliance monitoring is an open challenge (Section 3.6).
- Compute and access
  - Despite public release, reproducing training remains resource-intensive (384 A100s for months; Section 3.4.1), so full retraining is still out of reach for most labs.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that academic/consortial teams can train and release 100B+ multilingual models with transparent data/processes, shifting the field from closed demos to open artifacts (Sections 2.2, 3.6).
  - Provides a tested recipe—data pipeline, tokenizer, architecture, distributed stack—that others can adopt or extend (Sections 3.1–3.4).
  - Establishes multilingual instruction tuning (xP3 → BLOOMZ) as a scalable route to zero-shot generalization beyond English (Sections 3.1.4; 4.7).

- Follow-up research enabled/suggested
  - Data: Expand ROOTS with deeper coverage for underrepresented languages; experiment with curated domain additions (legal/health) under clear governance (Sections 3.1.1–3.1.2).
  - Modeling: Explore whether the extra Embedding LayerNorm is necessary under bfloat16 at 176B; test alternatives like MoE or state-space models now that tooling has matured (Sections 3.2.1–3.2.3).
  - Objectives: Systematically study conversion from causal decoder-only to other objectives post-pretraining (Section 3.2.2 suggests this is promising).
  - Long-context: Exploit ALiBi’s extrapolation to evaluate/document long-context behavior in real tasks (Section 3.2.3).
  - Instruction tuning: Enrich xP3 for code and tool-use tasks; study multilingual RLHF-style alignment on top of BLOOMZ.
  - Evaluation: Expand probing to more languages/features and longitudinally analyze when linguistic properties emerge during pretraining (Section 4.9.2 “Discussion”).

- Practical applications
  - Multilingual assistants for summarization, information access, and translation—especially in high/mid-resource languages (Figure 9; Tables 6–8).
  - Multilingual retrieval and STS using SGPT-BLOOM embeddings for search, deduplication, and clustering (Table 10).
  - Education and research: A shared, inspectable foundation for studying scaling laws, prompt engineering, multilingual transfer, and socio-technical impacts.
  - Enterprise deployments with responsible-use constraints via RAIL; carbon-aware training/deployment planning using the provided LCA approach (Section 3.5.1).

> In short, BLOOM delivers an open, multilingual foundation model and an end-to-end recipe—data governance to distributed training to multilingual instruction tuning—showing that strong multilingual zero-shot behavior is achievable and responsibly releasable at 176B scale. The main opportunities lie in boosting underrepresented language performance, broadening instruction tuning (e.g., for code/tool use), and deepening evaluation and alignment for multilingual, real-world deployments.
