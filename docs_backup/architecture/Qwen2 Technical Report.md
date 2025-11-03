# Qwen2 Technical Report

**ArXiv:** [2407.10671](https://arxiv.org/abs/2407.10671)
**Authors:** An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, Zhihao Fan
**Institutions:** 

## 🎯 Pitch

Qwen2 introduces a groundbreaking suite of open-weight large language models designed for long-context understanding, coding, mathematics, and multilingual capabilities. By integrating innovative architectures like Grouped Query Attention and Dual Chunk Attention with a curated 7-trillion-token dataset, Qwen2 sets a new benchmark in efficiency and performance, rivaling proprietary systems and empowering developers with robust, adaptable models for diverse applications.

---

## 1. Executive Summary (2-3 sentences)
Qwen2 is an open-weight family of large language models (LLMs) spanning five sizes (0.5B to 72B) plus a Mixture-of-Experts (`MoE`) variant, redesigned for long-context use, coding, mathematics, and multilingual capability. It combines an efficiency-focused architecture (e.g., `Grouped Query Attention`, `Dual Chunk Attention` with `YARN`, and a fine-grained `MoE`), a 7-trillion-token high-quality pre-training corpus, and a scalable post-training pipeline, yielding state-of-the-art results among open models and competitive performance against proprietary systems (e.g., Table 2, Table 6).

## 2. Context and Motivation
- Problem addressed
  - Open-weight LLMs often lag behind top proprietary systems on reasoning-heavy tasks (coding, math), long-context processing (above 32K tokens), and multilingual breadth. Qwen2 targets all of these gaps simultaneously (Section 1; Sections 2–4).
- Why it matters
  - Practical impact: Developers and researchers need strong, license-friendly models that run across a range of hardware (from on-device to multi-GPU), handle very long documents (legal, scientific, codebases), and support many languages (Section 1; 3.2; 5.2.4).
  - Scientific value: The work probes what actually scales capability—data quality vs. volume (Section 3.1), architectural choices for long contexts (Section 2.2.1), and alignment strategies that reduce the “alignment tax” (Sections 4.1–4.3).
- Prior approaches and shortcomings
  - Dense open models (e.g., Llama-3-70B) achieve strong general performance but are not optimized for Chinese or long-context scaling in the way Qwen2 is (Table 2; Sections 3.2, 5.2.4).
  - Open `MoE` models (e.g., Mixtral-8x7B) show good efficiency but often trail strong dense models on coding/math or multilingual breadth (Table 3).
  - Long-context extensions can degrade accuracy at extreme lengths and/or require costly KV cache memory (Sections 2.2.1, 3.2; Table 12; Figure 1).
- Positioning
  - Qwen2 releases a coordinated suite:
    - Dense models at 0.5B, 1.5B, 7B, 72B parameters.
    - A fine-grained `MoE` at “57B total / 14B active” (`57B-A14B`) that aims to match ~30B dense quality at 14B per-token compute (Sections 2.2.2–2.2.3; Table 1).
  - It emphasizes: efficient long context (up to 128K tested, Section 3.2; Figure 1; Table 12), improved coding and math, and broader multilingual coverage (Sections 3.1, 5.2.4).

## 3. Technical Approach
Step-by-step overview of the modeling and training pipeline.

- Tokenizer (Section 2.1)
  - Uses a byte-level byte-pair encoding vocabulary (151,643 regular + 3 control tokens) shared across all sizes for multilingual coverage and compression efficiency.
  - Practical effect: fewer tokens per sentence across languages → faster training/inference and better multilingual handling.

- Dense model architecture (Section 2.2.1; Table 1)
  - Transformer with:
    - `Grouped Query Attention (GQA)`: reuses Key/Value (“KV”) across groups of attention heads to shrink the KV cache and increase throughput, especially at long context. Table 1 shows fewer `# KV Heads` than `# Query Heads` (e.g., 72B uses 64 query heads but 8 KV heads).
    - `Dual Chunk Attention (DCA)`: a training-free mechanism that splits very long inputs into chunks. If the input fits in one chunk, DCA is identical to standard attention; if not, it tracks relative information across chunks to preserve coherence (Section 2.2.1).
    - `YARN`: “Yet Another RoPE eXtension” rescales position encodings to extrapolate to longer sequences without retraining (Section 2.2.1).
    - Other stabilizers: `RoPE` positional embedding, `QKV bias`, `SwiGLU` activations, and `RMSNorm` with pre-norm for stable training (Section 2.2.1).
  - Why this design: Lower KV memory and improved long-context extrapolation are crucial for practical 32K–128K token use (Sections 2.2.1, 3.2; Table 12; Figure 1).

- Mixture-of-Experts (`MoE`) architecture (Section 2.2.2; Equations (1)–(2); Table 1)
  - Replaces a dense layer’s feed-forward network (`FFN`) with multiple small experts. A gate computes probabilities `p = softmax(G(x))` and the output is a weighted sum of the top-k experts `sum_i p_i E_i(x)` (Equations (1)–(2)).
  - Key design choices:
    - Fine-grained experts: many smaller experts are activated together (64 routed experts with 8 activated per token; plus 8 shared experts, Table 1). This increases combinatorial routing variety and specialization (Section 2.2.2).
    - Shared vs. routing-specific experts: shared ones handle broadly useful features; others specialize via routing (Section 2.2.2).
    - Expert initialization via “sparse upcycling”: copy weights from a dense model, shuffle along the intermediate dimension, then randomly reinitialize 50% of each expert’s parameters to diversify behavior (Section 2.2.2 “Expert Initialization”).
  - Why this design: Keeps per-token compute low (14B active) while approaching the quality of ~30B dense models (Table 3).

- Pre-training data and strategy (Section 3.1; Table 1)
  - Data focus: higher quality and more coverage for code, math, and ~30 languages (English, Chinese, Spanish, French, German, Arabic, Russian, Korean, Japanese, Thai, Vietnamese, etc.).
  - Scale: 7 trillion tokens for all dense models except 0.5B (0.5B uses 12T). The MoE adds another 4.5T as “upcycling” (Table 1).
  - Quality vs. quantity: Attempting 12T for large models did not improve over the 7T set; quality mattered more than raw volume (Section 3.1).

- Long-context training (Section 3.2)
  - Context extended from 4,096 to 32,768 tokens late in pre-training, with much more long-form data.
  - `RoPE` base frequency changed from 10,000 to 1,000,000 to reduce position aliasing at long range.
  - `YARN` + `DCA` used at inference to scale beyond 32K (tested up to 131,072 tokens with minimal perplexity degradation in preliminaries).
  - Outcome: Practical processing of 64K–128K contexts in evaluations (Figure 1; Table 12).

- Post-training (Section 4)
  - Data construction (Sections 4.1.1–4.1.2)
    - “Collaborative” (human-in-the-loop) pipeline:
      - `InsTag`-based instruction ontology extraction and refinement (Section 4.1.1).
      - Instruction selection for diversity/complexity, then “self-evolution” to add constraints and difficulty (Section 4.1.1).
      - Human annotation ranks multiple model responses to produce both demonstrations and preference pairs (Section 4.1.1).
    - Automated synthesis to scale high-quality labels:
      - `Rejection sampling` for math: sample multiple chains-of-thought, keep those with correct final answers and plausible reasoning (Section 4.1.2).
      - `Execution feedback` for coding: generate code + tests; run them to verify correctness; also used for instruction adherence via Python verifiers (Section 4.1.2).
      - `Data repurposing`: build instructions from public-domain texts (e.g., role-play from Wikipedia character profiles) to pair with high-quality targets (Section 4.1.2).
      - `Constitutional feedback`: generate aligned and deliberately misaligned responses based on safety/value principles to create preference data (Section 4.1.2).
  - Supervised fine-tuning (`SFT`) (Section 4.2)
    - >500k examples across instruction-following, coding, math, reasoning, role-play, multilingual, and safety; 2 epochs, 32,768-token sequences; learning rate decayed from 7e-6 to 7e-7; weight decay 0.1; gradient clipping 1.0.
  - Preference optimization and online alignment (Section 4.3)
    - Offline `DPO` (Direct Preference Optimization): maximize the log-likelihood gap between preferred and dispreferred responses, using precompiled preference pairs.
    - Online `DPO`: sample multiple responses from the current model; a reward model selects best and worst; train on these pairs iteratively.
    - `Online Merging Optimizer (OMO)`: mitigates the “alignment tax” (the tendency for alignment to hurt core capabilities) by merging aligned improvements while preserving base skills (Section 4.3).

## 4. Key Insights and Innovations
- Fine-grained `MoE` with diversified upcycled experts (Section 2.2.2; Table 1; Equations (1)–(2))
  - What’s new: Many small experts with both shared and routing-specific roles, plus a unique initialization that shuffles and partially re-randomizes weights to diversify behavior.
  - Why it matters: With only 14B active parameters per token, `Qwen2-57B-A14B` approaches or exceeds ~30B dense models on coding and math while retaining efficiency (Table 3).
- Long-context stack that works in practice (Sections 2.2.1, 3.2; Figure 1; Table 12)
  - What’s new: A practical combination—`RoPE` base frequency = 1e6, `YARN` scaling, and `DCA`—that preserves standard attention behavior for ≤ one chunk and maintains useful signal across chunks beyond 32K tokens.
  - Why it matters: Strong performance at 64K–128K in NeedleBench and LV-Eval (Table 12) and near-perfect retrieval across depths up to 128K in Needle-in-a-Haystack for larger models (Figure 1).
- Data-first scaling: 7T high-quality tokens beat 12T looser data (Section 3.1)
  - What’s new: An explicit test showing more data (12T) did not help beyond a curated 7T set for large models.
  - Why it matters: It shifts optimization toward better filtering, domain balancing (code, math, multilingual), and on-the-fly synthesis, not just volume (Section 3.1).
- Scalable, mixed human–automatic alignment pipeline (Sections 4.1–4.3)
  - What’s new: A multi-pronged approach—ontology-driven instruction selection, self-evolution, executable verifiers, rejection sampling, repurposed literature/roleplay, constitutional feedback, and online `DPO` + `OMO`.
  - Why it matters: Delivers strong alignment scores (e.g., MT-Bench 9.12 and Arena-Hard 48.1 for `Qwen2-72B-Instruct`, Table 6) with minimized manual labeling.

## 5. Experimental Analysis
- Evaluation methodology (Section 5; 5.1; 5.2)
  - Base models: few-shot/zero-shot on standard benchmarks of knowledge, reasoning, coding, math, and multilingual (lists in Section 5.1.1).
  - Instruction-tuned: same core tasks plus human-preference benchmarks (MT-Bench, Arena-Hard, MixEval, AlignBench) and instruction-following (`IFEval`).
  - Long context: Needle-in-a-Haystack (depth-sensitive retrieval), NeedleBench (multi-fact reasoning) and LV-Eval (multi-evidence QA); `YARN` and `DCA` used beyond 32K (Figure 1; Table 12).
  - Safety and contamination: multilingual jailbreaking prompts and strict n-gram/LCS filters for decontamination checks (Sections 5.2.5–5.2.6; Tables 14–15).

- Main quantitative results (selected highlights)
  - Base 72B (Table 2)
    - > “MMLU: 84.2, MMLU-Pro: 55.6, GPQA: 37.9, TheoremQA: 43.1, BBH: 82.4.”
    - > “Coding—HumanEval: 64.6, MBPP: 76.9, EvalPlus: 65.4, MultiPL-E: 59.6.”
    - > “Math—GSM8K: 89.5, MATH: 51.1.”
    - Versus Llama-3-70B: +4.7 on MMLU (84.2 vs. 79.5), +18.3 on HumanEval (64.6 vs. 48.2), +6.5 on GSM8K (89.5 vs. 83.0). Slightly behind on HellaSwag/ARC-C by small margins (Table 2).
  - `MoE` 57B-A14B (Table 3)
    - > “HumanEval: 53.0, MBPP: 71.9, GSM8K: 80.7, MATH: 43.0.”
    - Outperforms Mixtral-8x7B on coding/math and matches or exceeds dense 30B baselines in many metrics; very strong Chinese capabilities (C-Eval 87.7, CMMLU 88.5) for its compute budget (Table 3).
  - Dense 7B (Table 4)
    - > “HumanEval: 51.2, MBPP: 65.9, GSM8K: 79.9, MATH: 44.2; MMLU: 70.3.”
    - Beats Llama-3-8B on many coding/math metrics and significantly surpasses prior Qwen1.5-7B (Table 4).
  - Small models (Table 5)
    - `Qwen2-1.5B` outperforms Phi-2 on MMLU (56.5 vs. 52.7) and has much stronger math (GSM8K 58.5 vs. 57.2; MATH 21.7 vs. 3.5). `Qwen2-0.5B` is surprisingly competitive for its size.
  - Instruction-tuned 72B (Table 6)
    - > “MMLU: 82.3, MMLU-Pro: 64.4, HumanEval: 86.0, LiveCodeBench v1: 35.7, GSM8K: 93.2, MATH: 69.0, MT-Bench: 9.12, Arena-Hard: 48.1, IFEval: 77.6.”
    - Competitive with or better than Llama-3-70B-Instruct on most core and alignment metrics; especially strong on MATH (+18.6) and Arena-Hard (+7.0) (Table 6).
  - Instruction-tuned `MoE` 57B-A14B (Table 7)
    - > “HumanEval: 79.9, MultiPL-E: 66.4, MixEval: 82.3, IFEval: 59.9.”
    - Beats Mixtral-8x7B and is competitive with dense 30B models, especially in coding and general alignment metrics.
  - Instruction-tuned 7B (Table 8)
    - > “HumanEval: 79.9, GSM8K: 85.7, MATH: 52.9.”
    - Very strong coding/math for its size, but weaker on instruction following (`IFEval 54.7`) vs. Llama-3-8B (`72.1`). The paper explicitly plans to improve instruction-following data for 7B (Table 8).
  - Long-context (Figure 1; Table 12)
    - Figure 1: `Qwen2-72B-Instruct` retrieves “needle facts” reliably across depths up to 128K tokens; `Qwen2-7B-Instruct` is accurate up to 128K; `Qwen2-57B-A14B-Instruct` up to 64K; smaller models up to 32K.
    - Table 12 shows `YARN + DCA` improves NeedleBench/LV-Eval at large lengths while not changing behavior below 32K.
  - Multilingual human eval (Table 13)
    - > “Average score (1–5 scale): Qwen2-72B-Instruct = 3.93 vs. GPT-3.5-Turbo 3.16, GPT-4-Turbo 3.98, GPT-4o 4.09, Claude-3-Opus 4.15.”
    - Strong, competitive multilingual quality, clearly ahead of GPT-3.5-Turbo.
  - Safety (Table 14)
    - Lower harmful-response rates than GPT-4 and Mixtral-8x22B-Instruct on Fraud and Privacy; ties GPT-4 on Illegal; slightly better than GPT-4 on Pornography (22.91 vs. 23.63).
  - Contamination analysis (Table 15)
    - Strict 13-gram criterion finds varying contamination rates (e.g., HumanEval 75%). However, performance on decontaminated subsets remains similar or even slightly better (e.g., `Qwen2-72B-Instruct` MATH +5.6), suggesting many “contaminated” flags are false positives for common math/code snippets.

- Do the experiments support the claims?
  - Yes for long context (Figure 1, Table 12), coding/math (Tables 2, 4, 6–8), and multilingual quality (Table 13). The combination of base and instruction-tuned results shows consistent gains vs. prior open models and closeness to high-end proprietary models on many axes.
  - Partial caveats: Some general-language benchmarks see small deficits vs. strong baselines (e.g., ARC-C, HellaSwag in Table 2), and instruction-following for the 7B model lags (Table 8).

- Ablations and robustness
  - Long-context ablation is implicit: with vs. without `YARN + DCA` in Table 12.
  - There is no full ablation isolating each architectural or data component (e.g., GQA’s specific effect, or expert-init variants), which leaves some uncertainty about each element’s standalone contribution.

## 6. Limitations and Trade-offs
- Data and training assumptions (Section 3.1)
  - Relies on very large, high-quality corpora with heavy filtering and synthesis. This pipeline may be hard for smaller teams to reproduce. The exact mix proportions are not exhaustively disclosed.
- MoE training budget (Sections 2.2.3, 5.2.2)
  - The `MoE` model received fewer total pre-training tokens than an equivalently scaled dense model; Section 5.2.2 notes a shortfall in knowledge understanding vs. the 32B dense baseline, possibly due to this.
- Long-context performance beyond 128K (Table 12)
  - Accuracy drops at 256K tokens (e.g., `Qwen2-72B-Instruct` NeedleBench 17.13 without YARN/DCA; LV-Eval 2.88), showing practical limits. The approach is strong up to 128K, but 256K remains challenging.
- Instruction following at small scales (Table 8; Table 9)
  - The 7B and sub-2B models still trail larger models on instruction-adherence benchmarks (`IFEval`), suggesting data/technique refinements are needed for smaller models.
- Compute/memory trade-offs
  - Although `GQA` reduces KV cache size and `MoE` cuts per-token compute, 72B remains heavy; even 7B long-context inference at 128K tokens demands substantial memory for KV caches.
- Transparency and reproducibility
  - While weights are open, full training details (compute cost, exact dataset sources/proportions) are not fully enumerated, limiting apples-to-apples replication.

## 7. Implications and Future Directions
- Field impact
  - Qwen2 shifts the open-weight frontier by showing that careful architectural choices (GQA, `DCA` + `YARN`), quality-centric data curation, and a scalable alignment pipeline can deliver proprietary-level capabilities in many areas. It especially moves the needle for long-context and math/coding in open models (Tables 2, 6; Figure 1; Table 12).
- Follow-up research enabled
  - Long-context training: Investigate training-time chunk-aware objectives and data curricula that further stabilize 128K–256K behavior, plus richer retrieval-augmented generation designs that exploit 72B’s demonstrated long-range retrieval (Figure 1; Table 12).
  - `MoE` scaling laws: Systematic studies on expert granularity, shared vs. routed expert ratios, and token budgets for MoE to close the remaining knowledge gaps (Sections 2.2.2–2.2.3; Table 3).
  - Alignment without tax: Extend online `DPO` + `OMO` studies to quantify how general capabilities can be preserved or even improved during alignment at varying model sizes (Section 4.3; Tables 6–8).
  - Data quality over quantity: Formalize filtering/synthesis pipelines that reproducibly beat raw scale (Section 3.1), including open-source verifiers for code and instruction-following.
- Practical applications
  - Enterprise document analysis and legal/compliance review with 64K–128K contexts.
  - Software development assistants that pass both correctness (execution feedback) and style constraints (instruction verifiers) at high rates (Table 6; LiveCodeBench).
  - Multilingual assistants in ~30 languages for customer service, education, and cross-border operations (Table 13).
  - On-device or edge scenarios with `Qwen2-0.5B` and `Qwen2-1.5B`, trading peak accuracy for latency/footprint (Table 5; Table 9).

Overall, Qwen2 demonstrates that a carefully engineered open-weight stack—efficient attention, fine-grained `MoE`, curated 7T-token pre-training, and a hybrid human–automatic alignment pipeline—can deliver strong general, coding, math, multilingual, and long-context performance, making it a compelling base for both research and real-world deployments.
