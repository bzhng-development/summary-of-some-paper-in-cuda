# SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model

**ArXiv:** [2502.02737](https://arxiv.org/abs/2502.02737)
**Authors:** Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlíček, Agustín Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan‑Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, Colin Raffel, Leandro von Werra, Thomas Wolf
**Institutions:** 

## 🎯 Pitch

SmolLM2 redefines small-model efficiency by leveraging a novel data-first strategy, combining targeted datasets and a multi-stage, data-rebalancing training schedule. This approach not only makes a 1.7B model rival larger counterparts in reasoning and coding but also democratizes access to powerful language models by reducing compute costs, paving the way for more resource-efficient applications in cost-sensitive environments.

---

## 1. Executive Summary
SmolLM2 is a 1.7B-parameter language model trained with a data-first strategy: it “overtrains” a small model on ~11 trillion tokens using a multi‑stage, continuously rebalanced mixture of carefully filtered web, math, and code data, plus targeted instruction and preference tuning (Sections 3–5). The result is a small model that rivals or beats other 1–2B models on many knowledge and reasoning tasks while staying competitive on coding and math, achieved largely through three new open datasets—`FineMath`, `Stack‑Edu`, and `SmolTalk`—and a structured training schedule (Figure 2; Tables 4–5).

## 2. Context and Motivation
- Problem addressed
  - Small models (≤3B parameters) are attractive for on-device and cost‑sensitive deployment but typically lag behind larger models in knowledge, reasoning, math, and coding. The key bottleneck is not only model size, but the quality, composition, and schedule of training data and how it is mixed over very long trainings (Introduction; Section 2).
- Why it matters
  - Lower‑compute LMs enable broader access (edge devices, privacy‑sensitive domains). Improving them with data/mixing rather than more parameters reduces inference costs and expands their practical utility (Introduction; Section 4).
- Where prior approaches fall short
  - Web corpora alone—even when filtered—underperform on specialized domains like math and code, and small specialized datasets get drowned in large general corpora if mixed naively from the start (Sections 2–3).
  - Running many full training runs to tune mixtures is too expensive for long‑trained small LMs (SmolLM2’s pretraining is ~1e23 FLOPs, ≈$250k GPU; Section 4).
- Positioning
  - SmolLM2 organizes training around “data-centric” interventions:
    - Systematic ablations to choose web, math, and code datasets (Section 3; Table 1; Figure 1; Table 2).
    - New, larger, higher-quality specialized datasets (`FineMath`, `Stack‑Edu`) to fix gaps in public data (Sections 3.3–3.4).
    - A multi‑stage, manually rebalanced schedule to upsample the right data at the right time, with final‑stage “annealing” on the best math and code data (Section 4; Figure 2).

## 3. Technical Approach
The pipeline has four pillars: dataset ablations and construction, multi-stage pretraining with online rebalancing, long‑context extension, and post‑training.

- Definitions used once where needed
  - `token`: a chunk of text (word or subword) the model reads or predicts.
  - `annealing ablation`: start from a mid‑training checkpoint, then linearly decay learning rate to 0 while training on a mixture that includes a candidate dataset; this reveals the dataset’s marginal impact without a full re‑train (Section 3.1).
  - `WSD` (Warmup‑Stable‑Decay): a learning‑rate schedule with warmup, a long flat “stable” phase, and a switchable linear decay to zero, so training length is not fixed up front (Appendix A; Figure 3).
  - `MMLU CF` vs `MMLU MCF`: cloze formulation computes answer likelihood; multiple‑choice formulation requires explicit A/B/C/D output (Section 4.3; Figure 6).
  - `RoPE`: rotary positional embeddings; SmolLM2 raises RoPE base to 130k for long context (Section 4.6).
  - `DPO`: Direct Preference Optimization for preference learning without explicit reward models (Section 5.3).

A) Empirical dataset ablations and new datasets (Section 3)
- English web (Section 3.2; Table 1)
  - Tested `FineWeb‑Edu` (education‑filtered web) vs `DCLM` (DataComp‑LM; web filtered by explain-like‑I’m‑five style).
  - 350B‑token ablations (same architecture and hyperparameters; Section 3.1) show complementary strengths:
    - > “FineWeb‑Edu achieves higher scores on MMLU, ARC, and OpenBookQA, while DCLM performs better on HellaSwag and CommonsenseQA.” (Table 1)
  - Choice: mix 60% `FineWeb‑Edu` + 40% `DCLM` for early stages (Table 1; Section 4.2), later tilt more toward DCLM to improve MMLU MCF (Section 4.3).

- Math (Sections 3.3.1–3.3.2; Figure 1)
  - Public datasets (`OpenWebMath`, `InfiMM‑WebMath`) are too small or skewed toward advanced papers with little step‑by‑step reasoning (Section 3.3.1; Figure 5).
  - Built `FineMath`:
    - Start from 6.5T tokens re‑extracted from 7.1B pages in math‑rich domains; preserve LaTeX; heavy dedup/language filtering (Section 3.3.2).
    - Two classifier passes using Llama‑3.1‑70B‑Instruct “silver” labels: first to find math domains, second to target mid/high‑school level, step‑by‑step reasoning (Appendix C.2–C.3).
    - Variants: `FineMath4+` (10B tokens, scores 4–5) and `FineMath3+` (34B tokens, scores 3–5), with 13‑gram decontamination against GSM8K, MATH, MMLU (Section 3.3.2).
    - Result in ablations: 
      > “FineMath4+ achieves a 2x improvement on GSM8K and a 6x improvement on MATH compared to InfiMM‑WebMath.” (Figure 1 summary)

- Code (Section 3.4; Table 2)
  - Baselines (`StarCoder2Data`, `Stack v2`) are large but contain a lot of non‑pedagogical code.
  - Built `Stack‑Edu`: filter `StarCoder2Data` for educational quality using per‑language classifiers (trained on 500k synthetic‑labeled samples per language; F1 > 0.7 for most; Appendix D.1). Keep 15 most common languages; ~125B tokens after filtering (Appendix D.2).
  - Annealing ablations find threshold 3 works best for most languages; MultiPL‑E improves across languages:
    > Python: 20.7 → 25.6; C++: 16.7 → 24.8; JavaScript: 18.2 → 22.4; Java: 17.6 → 22.7 (Table 2)

B) Multi‑stage pretraining with online rebalancing (Section 4; Figure 2)
- Model and setup (Appendix A)
  - `SmolLM2‑1.7B`: 24 layers, d_model 2048, 32 heads, SwiGLU, RoPE, 2M tokens/batch; AdamW; tokenizer with 49,152 vocab (Appendix A). Trained on 256 H100s using `nanotron` (Section 4.1).
  - Learning rate: WSD with 2k warmup steps, peak 5e‑4, then final 10% linear decay (Figure 3).

- Stage 1 (0–6T tokens; Section 4.2)
  - Mixture: ~90% English web (60/40 FineWeb‑Edu/DCLM), 10% `StarCoderData`. No math yet due to small math corpora.
  - Observation: good knowledge/reasoning; weak code and math (Table 3; Stage 1 row).

- Stage 2 (6–8T; Section 4.3)
  - Mixture: 75% web (still 60/40), 20% code (upsampled), 5% `OWM` math.
  - Observation: code improves; math barely moves; MMLU MCF rises above random for a small model (Figure 6).

- Stage 3 (8–10T; Section 4.4)
  - Mixture changes:
    - Web: flip to 40/60 `FineWeb‑Edu`/`DCLM` (helps MMLU MCF at this point).
    - Code: switch to `Stack‑Edu` (+ Jupyter notebooks).
    - Math: add text‑only `InfiMM‑WebMath` alongside `OWM` to ~10% math.
  - Observation: general improvement; a transient “loss spike” occurs (cause unclear) but most metrics recover by stage end (Section 4.4).

- Stage 4 (10–11T; final decay; Section 4.5)
  - Linear LR decay to zero; “anneal” on the best specialized data:
    - Math: `FineMath4+` + `InfiWebMath‑3+` dominate math portion; tiny `OWM` (0.08%) and `AugGSM8K` (0.02%) for coverage.
    - Code: `Stack‑Edu` at 24%, broader language coverage.
    - Web: 58% (DCLM‑heavy) + 4% `Cosmopedia v2` (synthetic textbooks/stories).
  - Observation: largest gains in math and code show up here (Table 3 and Table 8).

C) Long‑context extension (Section 4.6)
- Raise context from 2k → 8k tokens by taking a late stage‑4 checkpoint, setting RoPE base to 130k, and training on a mixture where 40% are long documents (≥8k) from DCLM, FineWeb‑Edu, and Dolma‑Books, with 60% following the stage‑4 mix.

D) Post‑training (Section 5)
- Supervised fine‑tuning (SFT) on `SmolTalk` (Section 5.1–5.2; Table 9)
  - Motivation: off‑the‑shelf instruction datasets underperform for this base model; build a tailored mix of 1.1M pairs.
  - `MagPie‑Ultra` (431k): three‑turn, system‑prompted conversations generated by Llama‑3.1‑405B and filtered for quality/safety (Section 5.1.1).
  - Task‑specific sets: `Smol‑Constraint` (36k instruction‑following with constraints), `Smol‑Summarization` (101k), `Smol‑Rewrite` (56k) (Section 5.1.2).
  - Math SFT: combine `NuminaMath‑CoT` and `MetaMathQA` (Section 5.1.3).
  - Plus code (Self‑OSS‑StarCoder2‑Instruct), system‑prompt and function‑calling data, and small long‑context SFT (LongAlign) (Section 5.1.4).
  - Train SFT for 2 epochs, 8k context, LR 3e‑4 (Section 5.2).

- Preference learning with `DPO` (Section 5.3)
  - Use `UltraFeedback` as the most effective feedback pool in experiments; 2 epochs, LR 1e‑6, beta 0.5, 1k context during DPO (Section 5.3).

## 4. Key Insights and Innovations
- Multi‑stage, performance‑driven data rebalancing for small models (Section 4; Figure 2)
  - Innovation: Rather than fix a single mixture, SmolLM2 monitors capabilities during training and “intervenes” in later stages (especially the decay stage) by upsampling the most effective specialized data (e.g., `FineMath4+`, `Stack‑Edu`).
  - Significance: It delivers large late‑stage gains in math/code while preserving general knowledge, without multiple end‑to‑end restarts (Table 3; Table 8).

- `FineMath`: targeted, high‑quality math corpus emphasizing step‑by‑step reasoning at appropriate difficulty (Section 3.3.2; Figure 1)
  - What’s different: domain‑level mining from billions of URLs, two‑stage classifier prompts explicitly aiming at middle/high‑school reasoning, heavy dedup and decontamination.
  - Why it matters: Ablations show much stronger math learning than prior open math corpora:
    > “FineMath4+ … 2x improvement on GSM8K and 6x on MATH vs InfiMM‑WebMath.” (Figure 1 summary)

- `Stack‑Edu`: education‑filtered code pretraining (Section 3.4; Table 2)
  - What’s different: per‑language classifiers trained on synthetic labels rate pedagogical quality, not just license/format.
  - Why it matters: Large, consistent MultiPL‑E gains across languages at manageable size, fitting small‑model capacity.

- `SmolTalk`: instruction dataset engineered for small models (Section 5.1; Table 9; Table 10)
  - What’s different: a balanced, quality‑filtered conversational core (`MagPie‑Ultra`) plus targeted components (constraints, summarization, rewriting, math, code, function calling, long‑context).
  - Why it matters: boosts instruction‑following and reasoning after SFT and DPO (Table 5 and Appendix F, Table 10).

- A practical annealing‑ablation protocol (Section 3.1)
  - What’s different: evaluate candidate datasets by resuming from a mid‑training checkpoint and decaying LR on a short burst including that dataset.
  - Why it matters: Enables evidence‑based mixture decisions under tight compute budgets.

## 5. Experimental Analysis
Evaluation design
- Pretraining ablations: identical 1.7B config runs (350B tokens for web; Section 3.1), and annealing ablations for math (60B tokens) and code (200B tokens) starting from a 3T‑token checkpoint (Section 3.1).
- Stage‑by‑stage tracking: category averages and per‑benchmark metrics after each pretraining stage (Table 3; Table 8).
- Final model comparisons: zero‑shot or few‑shot against `Llama3.2‑1B` and `Qwen2.5‑1.5B` for base and instruct models (Tables 4–5).
- Long‑context: Needle‑in‑a‑Haystack (NIAH) and HELMET at 8k context (Appendix G; Figure 7; Table 11).

Main quantitative results
- Web mix choice (Table 1)
  > “FineWeb‑Edu … better on MMLU, ARC, OpenBookQA; DCLM … better on HellaSwag and CommonsenseQA. The 60/40 mix balances both.”  
  This guided early mixing; later stages tilt to DCLM (Section 4.3).

- Math ablations (Figure 1; Section 3.3.2)
  > FineMath subsets “consistently outperform OWM and InfiMM‑WebMath on GSM8K, MATH, and MMLU‑STEM.”  
  Notably, `Infi‑WebMath4+` plateaus after ~10 epochs due to repetition, whereas `FineMath4+` keeps improving (Figure 1), justifying reserving `FineMath4+` for final annealing (Stage 4).

- Code ablations (Table 2)
  > MultiPL‑E improves for major languages after `Stack‑Edu` filtering (e.g., C++ 16.7 → 24.8).  
  This underpins the Stage‑3 switch to `Stack‑Edu`.

- Stage progression (Table 3; Table 8; Figure 6)
  - Category averages (Table 3):
    > Knowledge/Reasoning: 55.50 → 60.24; Math: 3.21 → 22.07; Code: 8.87 → 23.21; Generative: 31.54 → 36.12 (Stage 1 → 4).  
  - Per‑benchmark (Table 8):
    > MMLU MCF: 29.62 → 48.87; GSM8K: 4.32 → 32.60; MATH: 2.10 → 11.54; HumanEval: 10.97 → 22.60.  
  - Training dynamics:
    > “We observed above-random (>25%) MMLU accuracy with MCF after 6T tokens” (Figure 6), unusual for such small models, and a transient loss spike in Stage 3 that mostly recovers (Section 4.4).

- Base model comparison (Table 4)
  - Strengths vs contemporaries (1–2B):
    > HellaSwag: 68.7 (SmolLM2) vs 61.2 (Llama3.2‑1B) and 66.4 (Qwen2.5‑1.5B); ARC: 60.5 vs 49.2 and 58.5; CommonsenseQA: 43.6 vs 41.2 and 34.1.  
  - Held‑out generalization:
    > MMLU‑Pro: 19.4 vs 11.7 (Llama3.2‑1B) and 13.7 (Qwen2.5‑1.5B); TriviaQA: 36.7 vs 28.1 and 20.9.  
  - Math/coding remain competitive but not best:
    > GSM8K: 31.1 (SmolLM2) vs 61.7 (Qwen2.5‑1.5B); MATH: 11.6 vs 34.3; HumanEval: 22.6 vs 37.2 (Qwen2.5‑1.5B).  
    This reflects the strong specialized training Qwen uses; SmolLM2 narrows the gap with `FineMath` and `Stack‑Edu` but does not surpass Qwen2.5 on these two domains.

- Instruct model comparison (Table 5)
  - Instruction following:
    > IFEval avg: 56.7 (SmolLM2‑Instruct) vs 53.5 (Llama3.2‑1B‑Instruct) and 47.4 (Qwen2.5‑1.5B‑Instruct).  
  - Reasoning and math:
    > GSM8K: 48.8 vs 37.4 (Llama3.2‑1B‑Instruct) and 63.3 (Qwen2.5‑1.5B‑Instruct); MATH: 21.0 vs 19.5 and 19.6.  
  - Coding:
    > HumanEval: 28.1 vs 33.5 (Llama3.2‑1B‑Instruct) and 30.5 (Qwen2.5‑1.5B‑Instruct).  
  - Takeaway: strong instruction‑following and solid math for its size; coding remains behind the best small instruct models.

- Long‑context (Appendix G)
  - Needle‑in‑a‑Haystack: consistent retrieval across depths up to 8k (Figure 7 shows near‑perfect detection—green throughout).
  - HELMET (Table 11):
    > SmolLM2 leads on LongQA (33.00 vs 21.99 and 26.23), competitive on RAG (47.17 vs 42.13 and 47.54), but behind on ICL (23.20 vs 51.20 and 52.00).  
    So long‑document QA is strong; few‑shot in‑context learning lags.

Do the experiments support the claims?
- Yes, for the central thesis that careful data design and stage‑wise rebalancing can make a small model broadly competitive:
  - The staged schedule yields step‑wise capability improvements (Table 3 and Table 8).
  - New datasets demonstrably outperform prior open alternatives in targeted ablations (Figure 1; Table 2).
  - Final base/instruct comparisons show clear strengths on knowledge/reasoning and instruction following (Tables 4–5).
- Mixed results:
  - Math and coding are improved but still trail Qwen2.5 on some benchmarks; ICL under HELMET is notably weaker (Table 11).

## 6. Limitations and Trade-offs
- Heavy reliance on synthetic labeling and filtering
  - Many classifiers are trained on labels from large proprietary/open models (e.g., Llama‑3.1‑70B‑Instruct; Section 3.3.2; Appendix C–D). This may encode their biases and stylistic preferences into the datasets.
- Manual, “online” mixture tuning
  - Rebalancing is human‑in‑the‑loop and guided by observed metrics (Section 4). While effective, it can be subjective and less reproducible than a fully automated policy.
- Compute and data scale
  - The model is “overtrained” on 11T tokens (Sections 4, 4.7), deviating from Chinchilla‑style compute‑optimality; this is expensive even for small models (~$250k compute). Others may find it hard to replicate.
- Instability episode
  - A loss spike occurs in Stage 3 that does not have a clear cause even after rewind/skip attempts (Section 4.4), indicating some brittleness to mixture changes.
- Domain/language scope
  - The focus is primarily English text, code, and math; multilingual coverage is not a goal in this release (Sections 3–5).
- Evaluation coverage
  - Zero‑/few‑shot leaderboards are informative but not exhaustive; real‑world application robustness, safety beyond Llama‑Guard filtering, and long‑tail reasoning behaviors are not deeply audited here.
- ICL performance
  - HELMET shows weaker in‑context learning (ICL 23.2) compared to peers (Table 11), which may matter for prompt‑only adaptation use cases.

## 7. Implications and Future Directions
- Field impact
  - SmolLM2 demonstrates that small models can extract significant capability from data quality, curation, and schedule alone—without architectural novelty or parameter growth. The public release of `FineMath`, `Stack‑Edu`, and `SmolTalk` gives the community high‑leverage building blocks (Sections 3 and 5).
- What this enables next
  - Automated mixture controllers: replace manual rebalancing with bandit/RL controllers trained to optimize held‑out metrics during very long runs.
  - Data freshness and continual learning: extend the stage‑wise framework to streaming or periodic updates where new specialized data is annealed in without catastrophic forgetting.
  - Multilingual extensions: apply the same data‑centric recipe to other languages and cross‑lingual transfer.
  - Better math/coding specialization at small scale: explore curriculum schedules that interleave dataset difficulty (e.g., dynamic `FineMath4+` → `3+` cycles) and tool‑use integration for code.
  - ICL improvements: targeted pretraining on ICL‑friendly corpora or synthetic ICL traces to close the HELMET ICL gap.
- Practical applications
  - On‑device assistants with strong instruction adherence (IFEval 56.7; Table 5).
  - Education and tutoring—`FineMath` and `Stack‑Edu` biases toward step‑by‑step reasoning and pedagogical code are well aligned with tutoring use cases.
  - Lightweight coding helpers and math solvers in constrained environments, where slightly lower peak accuracy is acceptable for latency/privacy gains.
  - Retrieval‑augmented systems—strong LongQA/RAG scores (Table 11) suggest good performance when paired with external knowledge.

Overall, SmolLM2 shifts the conversation from “small models are too limited” to “small models benefit disproportionately from the right data and schedule.” The results—and the released datasets—should accelerate research on data‑centric training for efficient, capable LMs.
