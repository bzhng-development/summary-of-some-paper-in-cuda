# DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

**ArXiv:** [2401.02954](https://arxiv.org/abs/2401.02954)

## 🎯 Pitch

DeepSeek LLM redefines how open-source language models are scaled by empirically deriving new, practical scaling laws for allocating compute between model size and data size, as well as for key hyperparameters like batch size and learning rate. By introducing a more accurate measure of model scale (non-embedding FLOPs per token) and systematically validating these guidelines at billion-scale across 2 trillion tokens, DeepSeek produces bilingual models that not only surpass LLaMA-2 70B on multiple benchmarks but also approach or exceed GPT-3.5’s performance in open-ended tasks. This work equips the open-source AI community with robust, reproducible strategies to maximize performance and efficiency for future large-scale model training.

---

## 1. Executive Summary
DeepSeek LLM revisits how to scale open‑source large language models by (i) deriving empirical scaling laws for critical hyperparameters and for the trade‑off between model size and data size, and (ii) validating those laws by training bilingual 7B and 67B models on 2T tokens with strong reasoning, math, code, and Chinese capabilities. The central advance is a more accurate way to allocate compute between model and data using a new measure of model scale and an “IsoFLOP” methodology, leading to models that outperform LLaMA‑2 70B on many benchmarks and rival GPT‑3.5 in open‑ended dialogue (Sections 3, 5; Tables 5–8; Figures 4–5).

## 2. Context and Motivation
- Problem addressed
  - Open‑source LLMs typically train a few fixed parameter sizes (7B–70B) without principled guidance on how to allocate compute between model size and data, and without clear tuning rules for batch size and learning rate (Introduction; Section 3). Prior scaling laws disagree on the optimal model/data split: Kaplan et al. favor larger models (coefficients ≈ 0.73 model / 0.27 data), while Chinchilla favors more data (≈ 0.49 / 0.51).
- Why this matters
  - Compute budgets are finite. Misallocating compute (too many parameters, too little data—or vice versa) wastes resources and caps performance. Better scaling guidance directly improves the effectiveness and cost of training large open‑source models (Section 3).
- Shortcomings of prior approaches
  - Conflicting model/data exponents (Table 4) and incomplete reporting of hyperparameters make it unclear whether earlier experiments reached optimal settings at each compute budget (Section 3).
- Positioning of this work
  - DeepSeek LLM contributes:
    - Practical scaling rules for batch size and learning rate across budgets (Eq. (1); Figures 2–3).
    - A new, more faithful measure of “model scale” for decoder‑only Transformers—`non‑embedding FLOPs per token` (`M`)—and a compute identity `C = M · D` that improves the fit of scaling curves (Eq. (2); Table 3).
    - Evidence that the optimal model/data split depends on data quality—higher-quality data should be paired with larger models (Table 4).
    - Two large bilingual models (7B, 67B) trained to 2T tokens and aligned via supervised fine‑tuning (SFT) and direct preference optimization (DPO), with broad evaluation (Sections 2, 4, 5).

## 3. Technical Approach
This work comprises a data pipeline, model design, training setup, a scaling‑law program, and an alignment pipeline.

- Data pipeline (Section 2.1; Table 1)
  - Stages: deduplication → filtering → remixing.
  - Aggressive web deduplication across 91 Common Crawl dumps (not just dump‑internal) removes 89.8% of documents, vs. 22.2% within a single dump (Table 1). Goal: reduce repeated text that can distort learning signals.
  - Filtering uses linguistic and semantic heuristics to improve “information density” (Section 2.1).
  - Tokenizer: byte‑level BPE with pre‑tokenization to avoid merging across newlines, punctuation, and CJK symbols; numbers split into digits. Vocabulary trained to 100,015 tokens (plus special tokens), but the model uses a padded size of 102,400 for compute efficiency (Section 2.1).

- Model architecture (Section 2.2; Table 2)
  - Base design follows LLaMA: Pre‑Norm with `RMSNorm`, `SwiGLU` feed‑forward, `Rotary Embedding` for positions.
  - 67B uses `Grouped‑Query Attention (GQA)` to reduce key/value cache cost at inference; unlike typical GQA deployments that widen layers, DeepSeek scales depth (67B has 95 layers; 7B has 30) to match parameter budgets while easing pipeline partitioning (Section 2.2; Table 2).
  - Context length is 4096; batch sizes are 2304 (7B) and 4608 (67B) tokens per step (Table 2).

- Training setup (Sections 2.3–2.4; Figure 1; Table 2)
  - Optimizer: AdamW with β1=0.9, β2=0.95, weight decay=0.1; gradient clipping=1.0; bf16 compute with fp32 gradient accumulation.
  - Learning‑rate schedule: multi‑step rather than cosine for better continual‑training reuse. Warmup = 2000 steps; then LR drops to 31.6% at 80% of tokens and to 10% at 90% (Section 2.3).
    - Figure 1a shows multi‑step and cosine reach similar final loss; Figure 1b shows the chosen 80/10/10 split balances reuse and performance.
  - Infrastructure: custom `HAI‑LLM` framework with data/tensor/sequence/pipeline parallelism, FlashAttention, ZeRO‑1 optimizer partitioning, fused kernels, overlap of compute/communication, and “in‑place” cross‑entropy to cut memory. Asynchronous checkpoints every 5 minutes limit worst‑case loss from failures (Section 2.4).

- Scaling‑law program (Section 3)
  - Goal: ensure that, at each compute budget `C`, training uses near‑optimal batch size and learning rate and finds the best model/data split.
  - Hyperparameter scaling (Sections 3.1; Figures 2–3; Eq. (1)):
    - Run many small‑to‑medium experiments (C from 1e17 to 2e19 FLOPs) over grids of batch sizes and learning rates. Treat any setting within 0.25% of the best validation loss as “near‑optimal.”
    - Fit power laws:
      - `η_opt = 0.3118 · C^(-0.1250)`
      - `B_opt = 0.2920 · C^(0.3271)`
    - Figures 2a/2b show broad “valleys” of good hyperparameters; Figures 3a/3b show the fitted curves. Larger compute favors larger batches and lower learning rates.
  - Model/data scaling (Sections 3.2; Eq. (2); Table 3; Figures 4–5):
    - Define model scale as `M = non‑embedding FLOPs/token` to include attention compute but exclude vocabulary embedding compute. Prior proxies—`6·N1` (non‑embedding params) and `6·N2` (all params)—systematically misestimate compute, especially in small models and long contexts (Table 3).
    - Use the `IsoFLOP` profile: for each compute budget `C = M·D`, sweep multiple `(M, D)` allocations and identify the allocation with the lowest validation loss. Budgets span 1e17–3e20 FLOPs; ~10 allocations per budget; validation set is 100M tokens with the same distribution as training (Figure 4a).
    - Fit the optimal allocations (Figures 4b–4c) to obtain:
      - `M_opt = 0.1715 · C^(0.5243)`
      - `D_opt = 5.8316 · C^(0.4757)` (Eq. (4))
    - Fit a performance curve of bits‑per‑byte vs. `C` to predict large‑model performance. Figure 5 shows the small‑scale fit accurately predicts the 7B and 67B models (a ~1000× extrapolation).
  - Role of data quality (Section 3.3; Table 4):
    - Repeat the `IsoFLOP` fitting across three datasets: early in‑house, improved in‑house, and OpenWebText2. As data quality improves, the exponent on model scale increases while the exponent on data decreases:
      - Early data: `a=0.450, b=0.550`
      - Current data: `a=0.524, b=0.476`
      - OpenWebText2: `a=0.578, b=0.422`
    - Insight: higher‑quality data can productively feed larger models before data saturation.

- Alignment pipeline (Section 4)
  - Supervised Fine‑Tuning (SFT): ~1.5M instruction examples (31.2% general tasks; 46.6% math; 22.2% code) plus 300k safety prompts. 7B trains 4 epochs (LR 1e‑5), 67B trains 2 epochs (LR 5e‑6). They monitor a “repetition ratio” (share of never‑terminating, looping outputs) and observe it rises with heavy math SFT (Section 4).
  - Two‑stage SFT (Section 5.5, Table 12): for 7B, a second stage without math/code keeps benchmark scores while reducing repetition from 2.0% to 1.4%.
  - Direct Preference Optimization (DPO): a preference‑based alignment method that adjusts the model to prefer higher‑rated responses without training a separate reward model. Train 1 epoch at LR 5e‑6, batch size 512, using model‑generated responses as candidates for both helpfulness and harmlessness. It improves open‑ended quality with little impact on standard benchmarks (Table 17).

## 4. Key Insights and Innovations
- A more faithful compute model for scaling curves (Fundamental)
  - Using `M = non‑embedding FLOPs/token` as the model‑scale variable yields better predictions than parameter counts, because it includes attention cost (which grows with sequence length) and ignores vocabulary compute (which contributes less to model capacity). Table 3 quantifies large errors when using `6·N1` or `6·N2`, especially for small models; Appendix A.2 shows that fits using `6·N1` overestimate large‑model performance, while `6·N2` underestimates it (Figure 6).
- Empirical hyperparameter scaling laws (Practical, immediately usable)
  - Power‑law formulas for batch size and learning rate vs. compute budget (Eq. (1); Figures 2–3) give a recipe for picking near‑optimal settings across budgets. This removes guesswork when changing `C`.
- Data quality shifts the optimal model/data trade‑off (Conceptual)
  - Table 4 shows that better data pushes the optimal allocation toward larger models (higher `a`, lower `b`). This reconciles prior disagreements (Kaplan vs. Chinchilla) by attributing differences to data quality and composition.
- Depth‑heavy 67B with GQA and multi‑step LR (Incremental but effective)
  - The 67B model opts for more layers (95) rather than wider layers, easing pipeline parallelism and showing strong results (Section 2.2; Table 2). The multi‑step LR schedule enables efficient continual training without losing final quality (Figure 1).
- Alignment practices that balance capability and stability (Practical)
  - The two‑stage SFT and DPO reduce repetition and improve open‑ended dialogue without overfitting to multiple‑choice formats (Section 5.5; Tables 12–13, 17).

## 5. Experimental Analysis
- Evaluation setup (Section 5; Appendix A.6)
  - Benchmarks span language understanding (HellaSwag, PIQA, ARC, OpenBookQA, BBH), knowledge QA (TriviaQA, NaturalQuestions), reading comprehension (RACE, DROP, C3), Chinese tasks (CHID, C‑Eval, CMMLU, CMath, CCPM), math (GSM8K, MATH), coding (HumanEval, MBPP), and language modeling (Pile‑test).
  - Protocols:
    - Multiple‑choice: perplexity scoring over answer options, with length normalization (and unconditional normalization for ARC/OpenBookQA).
    - Generation tasks: greedy decoding and programmatic answer parsing.
    - Language modeling: bits‑per‑byte (lower is better).
- Main quantitative results (Base models; Table 5)
  - Relative to LLaMA‑2 70B, `DeepSeek‑67B` improves substantially on math and code:
    - MATH: 18.7 vs. 13.5 (+5.2)
    - GSM8K (8‑shot): 63.4 vs. 58.4 (+5.0)
    - HumanEval (0‑shot): 42.7 vs. 28.7 (+14.0)
    - MBPP (3‑shot): 57.4 vs. 45.6 (+11.8)
    - BBH: 68.7 vs. 62.9 (+5.8)
  - Chinese tasks: large gains over LLaMA‑2 70B
    - C‑Eval: 66.1 vs. 51.4 (+14.7)
    - CMMLU: 70.8 vs. 53.1 (+17.7)
    - CHID: 92.1 vs. 55.5 (+36.6)
  - English understanding is comparable:
    - HellaSwag: 84.0 (tie)
    - MMLU (5‑shot): 71.3 vs. 69.0 (+2.3)
    - Pile‑test BPB: 0.642 vs. 0.649 (lower is better)
- Effects of tuning (Chat models; Table 6)
  - SFT/DPO massively boost math and code:
    - GSM8K (0‑shot): 67B rises from 63.4 (base, 8‑shot) to 84.1 (chat, 0‑shot).
    - HumanEval: 67B from 42.7 to 73.8.
    - MATH: 67B from 18.7 to 32.6.
  - Some multi‑choice style tasks drop after alignment (e.g., HellaSwag and WinoGrande), which the paper attributes to pure LM perplexity scoring favoring base models (Section 5.1.2).
- Open‑ended evaluations (Section 5.2)
  - Chinese AlignBench (GPT‑4 judged; Table 7):
    - `DeepSeek‑67B‑Chat‑DPO`: 6.69 overall, outscoring ChatGPT (6.08) and most open‑source peers; behind GPT‑4 variants.
    - DPO improves nearly all categories over SFT‑only.
  - English MT‑Bench (GPT‑4 judged; Table 8):
    - `67B‑Chat`: 8.35 (≈ GPT‑3.5 turbo at 8.39), improving to 8.76 with DPO; GPT‑4 is 9.26.
- Held‑out evaluations (Section 5.3; Table 9)
  - LeetCode (recent contests): `67B‑Chat` achieves 17.5 pass@1 (vs. Qwen‑72B 12.7).
  - Hungarian National High‑School Exam (math): 58 (vs. Qwen‑72B 52; GPT‑4 68).
  - Instruction Following (IFEval): 55.5 (vs. Qwen‑72B 50.8).
- Safety (Section 5.4; Tables 10–11)
  - Human‑built taxonomy with 2400 prompts shows high safe‑answer rates across categories (Table 10).
  - On “Do‑Not‑Answer” (939 prompts), `67B‑Chat` scores 97.8, slightly above ChatGPT (97.7) and above GPT‑4 (96.5) in that metric (Table 11).
- Ablations and diagnostics (Section 5.5)
  - Two‑stage SFT reduces repetition without hurting code/math (Table 12).
  - Adding 20M Chinese multiple‑choice questions inflates MC benchmarks in both languages (e.g., 7B MMLU +11.5 points; C‑Eval +24.3) but does not help generative QA (TriviaQA unchanged; ChineseQA slightly down)—evidence of format‑specific overfitting (Table 13).
  - Instruction data late in pre‑training vs. in SFT yields similar final capability; they choose not to include instruction data in pre‑training (Section 5.5).
  - System prompts help large models but can slightly hurt small ones: 67B MT‑Bench improves 8.35 → 8.58; 7B drops 7.15 → 7.11 (Table 14).

Assessment: The breadth of benchmarks, inclusion of held‑out datasets, ablations on MC data, and safety checks make a credible case that the models are strong, especially for math/code and Chinese. Some evaluations rely on LLM‑as‑judge (GPT‑4), which is standard but can bias rankings; the paper mitigates this with many programmatic benchmarks and human‑graded tests (e.g., Hungarian exam).

## 6. Limitations and Trade-offs
- Dependence on data quality without a formal metric
  - The claim that “higher‑quality data → more model scaling” (Table 4) is supported by experiments but lacks a quantitative, reusable quality measure. Applying these exponents to other corpora may require re‑estimation (Section 3.3).
- Scope of the compute model
  - `M = non‑embedding FLOPs/token` fits decoder‑only Transformers with standard attention and feed‑forward shapes (Eq. (2)). Variants with very long contexts, flash‑decoding tricks, or architectural changes (e.g., Mixture‑of‑Experts, retrieval‑augmented models) may alter the mapping between parameters and compute, limiting direct transfer of the fitted exponents (Appendix A.2 highlights representation sensitivity at low budgets).
- Hyperparameter scaling coverage
  - Batch size and LR formulas (Eq. (1)) are fitted over 1e17–2e19 FLOPs with reuse of the first training stage. The formulas center in the optimal region (Figure 2b) but do not model effects beyond `C` (e.g., exact `(M, D)` mix also nudges the optimum; Section 3.1 notes this).
- Evaluation trade‑offs
  - Alignment can reduce scores on MC tasks like HellaSwag (Table 6); choosing between conversational helpfulness and raw MC performance is a design trade‑off.
  - Some open‑ended evaluations are judged by GPT‑4 (Tables 7–8), which introduces potential bias; however, many standard, auto‑graded datasets offset this.
- Compute and resource cost
  - Training to 2T tokens with 67B parameters is expensive; while the paper details efficiency techniques (Section 2.4), reproducing the full setup requires substantial hardware.
- Transparency of data
  - The 2T bilingual corpus is described at a high level (Section 2.1) but not fully released in this paper; replicability of exact scaling results across unseen data distributions remains an open question.

## 7. Implications and Future Directions
- Field impact
  - The compute identity `C = M · D` with `M` as non‑embedding FLOPs/token and the fitted exponents (Eq. (4)) provide a practical playbook for future open‑source LLM training. Researchers can plan budgets and predict performance (Figure 5) rather than guess. The observation that better data favors larger models helps reconcile Kaplan‑ vs. Chinchilla‑style guidance (Table 4).
- Follow‑up research
  - Generalize the compute model to other architectures (e.g., Mixture‑of‑Experts, retrieval‑augmented, long‑context attention) and verify whether analogous `M` definitions lead to stable exponents.
  - Formalize “data quality” with measurable proxies (e.g., perplexity filtering, diversity/novelty metrics) and test how each aspect shifts the `a/b` exponents.
  - Extend hyperparameter scaling to include weight decay, warmup schedules, and optimizer variants; analyze sensitivity to `(M, D)` composition (Section 3.1 notes residual dependence).
  - Explore reinforcement learning–based alignment to further improve complex reasoning (Section 6 “Conclusion” hints at positive early results).
- Practical applications
  - The 67B chat model’s strength on math/code (Tables 6, 8, 9, A.4) suggests immediate use in programming assistants, math tutoring, and enterprise Q&A in bilingual (Chinese/English) settings.
  - The strong safety performance (Tables 10–11) and two‑stage SFT recipe (Table 12) offer concrete guidelines for deploying helpful, low‑repetition chat systems without overfitting to multiple‑choice formats (Table 13).

> In short, DeepSeek LLM contributes a tested methodology for compute‑optimal scaling—hyperparameters, model/data allocation, and alignment choices—and validates it with competitive 7B/67B bilingual models trained on 2T tokens. The work offers both actionable recipes (Eq. (1), Eq. (4)) and conceptual insights (data quality’s role) that future open‑source LLM projects can adopt and extend.
