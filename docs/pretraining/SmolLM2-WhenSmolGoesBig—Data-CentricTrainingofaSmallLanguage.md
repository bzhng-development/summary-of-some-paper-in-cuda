# SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model

**ArXiv:** [2502.02737](https://arxiv.org/abs/2502.02737)

## 🎯 Pitch

SmolLM2 demonstrates that small language models can achieve state-of-the-art results by prioritizing data quality and mixture over simply increasing size or tweaking architecture. Through multi-stage, performance-driven training and the development of targeted datasets for math, code, and instruction-following, SmolLM2 outperforms peer models on a wide range of benchmarks—all while remaining efficient enough for on-device and resource-constrained deployment. This work is significant because it proves that with meticulous data curation and adaptive training strategies, compact models can deliver competitive capabilities previously thought exclusive to much larger LLMs.

---

## 1. Executive Summary
SmolLM2 introduces a 1.7B-parameter “small” language model trained for a long duration (≈11 trillion tokens) using a data-centric, multi‑stage recipe and three new datasets: `FineMath` (math), `Stack-Edu` (code), and `SmolTalk` (instruction tuning). The core contribution is showing that careful, stage-wise rebalancing of training data—rather than only architectural changes—pushes small models to state-of-the-art performance in their size class on many general benchmarks while remaining deployable on constrained hardware.

## 2. Context and Motivation
- Problem/gap
  - Small LMs (<3B parameters) are attractive for cost and on-device use, but they are highly sensitive to training data quality and mixture, often underperforming on reasoning-heavy tasks (Section 1, 2). Running multiple full-scale training runs to find the right data mix is prohibitive (≈1e23 FLOPs or ~$250k of GPU compute for this model; Section 1).
  - Public “specialized” data (math, code, instruction-following) is either too small, noisy, or not well targeted for step-by-step reasoning, which especially hurts small models’ capabilities (Sections 3.3.1–3.3.2, 3.4, 5.1).

- Why it matters
  - Real-world: Small models power applications on edge devices and less-provisioned servers, but only if they reach acceptable quality on knowledge, reasoning, coding, and math tasks.
  - Scientific: The work probes how far data quality, scheduling, and long training (beyond “Chinchilla-optimal” token counts) can push small models (Section 4).

- Prior approaches and shortcomings
  - Pretraining on large, filtered web scrapes is standard, with specialized domains sometimes added late (Section 2). However, public math and code sets are often too limited or skewed (e.g., towards advanced papers at the wrong level), so their contribution can be diluted by much larger web datasets (Sections 3.3.1, 3.4).
  - Prior small LMs often either undertrain or use fixed mixtures that are not re-optimized during the run (Section 4).

- Positioning
  - SmolLM2 uses “online” mixture updates—monitoring benchmarks during training and rebalancing data sources across four stages (Figure 2; Sections 4.2–4.5).
  - It also releases new, higher-quality datasets purpose-built for small models: `FineMath` (Section 3.3.2), `Stack-Edu` (Section 3.4), and `SmolTalk` (Section 5.1).

## 3. Technical Approach
This is primarily an empirical, system-level contribution with targeted dataset design and staged training.

- Model and training setup (Appendix A)
  - Architecture: 24-layer Transformer, 2048 hidden size, 8192 FFN, 32 attention heads, RoPE positional embeddings, SwiGLU, tied embeddings (Table 6).
  - Tokenizer: 49,152 vocab trained on a mixture emphasizing educational and math/code sources (Section 4.1).
  - Hardware and schedule: Trained on 256 H100 GPUs with AdamW and a Warmup–Stable–Decay (WSD) scheduler (Figure 3). WSD keeps a long plateau at a fixed learning rate and linearly decays to zero over the last 10% of steps, avoiding the need to pre-decide a fixed length.
  - Context length: Trained at 2k tokens; later extended to 8k tokens via continued training with 40% long documents and RoPE scaling (Section 4.6).

- Up-front ablations to choose building blocks (Section 3)
  - English web sources: Compared `FineWeb‑Edu` vs `DCLM` by training 1.7B models for 350B tokens each and measuring knowledge and reasoning benchmarks (Section 3.2). The datasets are complementary: `FineWeb‑Edu` excels on education-like tasks; `DCLM` on conversational/commonsense style. A 60/40 FineWeb‑Edu/DCLM mix gives balanced results (Table 1).
  - Math sources: “Annealing ablations” starting from a 3T-token mid‑training checkpoint (Section 3.1). Annealing here means mixing a candidate dataset into training while linearly decaying the learning rate to zero, to observe its marginal contribution without a full restart. Existing sets (OpenWebMath, InfiMM‑WebMath) were either too small or skewed, prompting creation of `FineMath` (Sections 3.3.1–3.3.2).
  - Code sources: A new filtered dataset, `Stack‑Edu`, is constructed by scoring the “educational value” of code samples (per language) in `StarCoder2Data` and keeping higher-scoring ones (Sections 3.4, D.1–D.2).

- New datasets (how they are built)
  - `FineMath` (Section 3.3.2):
    - Start from Common Crawl pages overlapping `FineWeb` and domains seen in OpenWebMath and InfiMM-WebMath; extract math-preserving text (LaTeX) with the OWM pipeline (Section 3.3.2).
    - Two-stage LLM-based classification for quality:
      1) 3-point scale to find math-like pages,
      2) 5-point scale to upweight mid‑/high‑school level and step-by-step reasoning (Appendix C.2–C.3).
    - Deduplicate with MinHash LSH and restrict to English; produce subsets like `FineMath4+` (scores 4–5, 10B tokens) and `FineMath3+` (scores 3–5, 34B tokens); also re-filtered InfiMM-WebMath into `Infi‑WebMath4+` and `3+`. Decontaminate against GSM8K/MATH/MMLU (13‑gram + LCS threshold).
  - `Stack‑Edu` (Section 3.4):
    - For the 15 largest languages in `StarCoder2Data` (~450B tokens), label 500k samples per language on a 0–5 “educational” scale with an LLM prompt; train a StarEncoder classifier per language and keep samples scoring ≥3 (≥2 for Java).
    - Output: ~125B tokens across 15 languages (Appendix D.2).
  - `SmolTalk` (Sections 5.1–5.1.4, Table 9):
    - A curated instruction-tuning mix (1.1M examples) blending a new multi-turn conversation set (`MagPie‑Ultra` generated with a 405B instruct model), constraint-following (`Smol‑Constraint`), summarization/rewriting (`Smol‑Summarization`, `Smol‑Rewrite`), and targeted math SFT (NuminaMath‑CoT + MetaMathQA), plus code, system prompts, function-calling, and long-context SFT subsets.

- Four-stage, performance-driven pretraining (Sections 4.2–4.5; Figure 2)
  - Terminology: “Web” is a FineWeb‑Edu/DCLM mix; “Code” is StarCoderData or Stack‑Edu; “Math” is OWM, InfiMM‑WebMath, FineMath.
  - Stage 1 (0–6T): 60/40 FineWeb‑Edu/DCLM; 10% code (StarCoderData); no math (Section 4.2).
  - Stage 2 (6–8T): Keep web ratio; raise code to 20%; add 5% OWM as math warm‑up (Section 4.3).
  - Stage 3 (8–10T): Shift web to 40/60 FineWeb‑Edu/DCLM; replace StarCoderData with `Stack‑Edu` (+ Jupyter notebooks); lift math to ~10% by adding InfiMM‑WebMath text (Section 4.4). A loss spike occurs after this mixture change but metrics recover by stage end.
  - Stage 4 (10–11T; LR decay): Introduce highest-quality math (`FineMath4+`, `Infi‑WebMath3+`, small OWM and AugGSM8K), totaling 14% math; expand `Stack‑Edu` to 24%; keep 58% web (with higher DCLM share) and add 4% `Cosmopedia v2` (synthetic textbooks) (Section 4.5). This is the “annealing” end-game designed to make strong domains stick.

- Context-length extension (Section 4.6)
  - From a late stage-4 checkpoint, continue training with 40% long documents (Dolma books + long web) and set RoPE to 130k to support 8k context.

- Post-training (Section 5)
  - Supervised fine-tuning (SFT): Train on `SmolTalk` for 2 epochs with 8k length (Section 5.2).
  - Preference learning: Use DPO (Direct Preference Optimization) on UltraFeedback for 2 epochs (Section 5.3). DPO uses pairwise “chosen vs rejected” responses to optimize the LM toward preferred outputs without a learned reward model.

## 4. Key Insights and Innovations
- Performance-driven, “online” data mixing at scale (fundamental for small LMs)
  - Rather than fixing a mixture, SmolLM2 monitors benchmark scores during training and adjusts the mix across stages (Figure 2; Sections 4.2–4.5). This is pragmatic when full restarts are too costly and appears decisive for gains in code and math by concentrating high-quality data at the end (stage‑4 annealing).
  - Significance: Table 3 shows category averages rising steadily, especially math: 3.21 → 22.07 and code: 8.87 → 23.21 from stage 1 to 4.

- FineMath: math data filtered for step-by-step, appropriately leveled reasoning (novel dataset)
  - Most open math corpora skew toward advanced academic text. `FineMath` explicitly upweights middle/high‑school content with stepwise solutions (Appendix C). In annealing ablations, `FineMath4+` dominates earlier sets on GSM8K/MATH (Figure 1), and its gains do not plateau early, indicating lower harmful repetition.

- Stack‑Edu: code data filtered for “educational value” (novel dataset)
  - Filtering `StarCoder2Data` to prioritize commented, tutorial-like, self-contained code improves code generalization. Table 2 reports MultiPL‑E/HumanEval improvements across languages (e.g., C++ 16.7→24.8; Python 20.7→25.6 after filtering).

- SmolTalk: instruction mix that targets constraints, rewriting, summarization, math, and system prompts (comprehensive SFT set)
  - Beyond standard SFT corpora, the mix uses multi-turn `MagPie‑Ultra` and targeted sub-datasets (Table 9), which lifts instruction-following metrics (Appendix F, Table 10) and leads to strong IFEval and MT-Bench scores after DPO (Table 5).

## 5. Experimental Analysis
- Evaluation setup
  - Pretraining ablations: 1.7B models; for web data, each ablation trained 350B tokens and measured MMLU/ARC/HellaSwag/PIQA, etc. (Section 3.1). For math and code, “annealing ablations” from a 3T-token checkpoint injected 60–200B tokens from candidate datasets (Section 3.1).
  - Final base model comparison: SmolLM2 vs Qwen2.5‑1.5B and Llama3.2‑1B on broad benchmarks, mostly zero-shot (Section 4.7, Table 4).
  - Instruct model comparison: SmolLM2‑Instruct vs their instruct counterparts on instruction-following, reasoning, math, and code (Section 5.4, Table 5).
  - Long-context: Needle-in-a-Haystack and HELMET with 8k context (Appendix G).

- Main quantitative takeaways (with citations)
  - Web mixtures:
    > Table 1: 60/40 FineWeb‑Edu/DCLM nearly matches FineWeb‑Edu on MMLU/ARC/OpenBookQA and approaches DCLM on HellaSwag/CommonSenseQA.
  - Math ablations:
    > Figure 1: `FineMath4+` is consistently strongest; on GSM8K it reaches about 2× the accuracy of InfiMM‑WebMath, and ≈6× on MATH at similar token counts.
  - Code filtering:
    > Table 2: Filtering to `Stack‑Edu` improves MultiPL‑E across languages (e.g., JavaScript 18.2 → 22.4).
  - Stage-by-stage learning:
    > Table 3: Average math rises 3.21 → 22.07; code 8.87 → 23.21; knowledge/reasoning 55.50 → 60.24; generative tasks 31.54 → 36.12 from stage 1 to 4.
    > Table 8: MMLU (multiple-choice formulation, MCF) 29.62 → 48.87; GSM8K 4.32 → 32.60; HumanEval 10.97 → 22.60.
    > Figure 6: MMLU‑MCF surpasses the 25% random baseline after 6T tokens, which small models often struggle to do.
  - Base model comparisons:
    > Table 4: SmolLM2 beats Qwen2.5‑1.5B on HellaSwag (68.7 vs 66.4) and ARC (60.5 vs 58.5) and shows strong held‑out generalization on MMLU‑Pro (19.4 vs 13.7) and TriviaQA (36.7 vs 20.9). It lags the Qwen base on GSM8K (31.1 vs 61.7), MATH (11.6 vs 34.3), and HumanEval (22.6 vs 37.2).
  - Instruct model comparisons:
    > Table 5: SmolLM2‑Instruct excels on IFEval (56.7 vs Qwen 47.4 vs Llama3.2 53.5), HellaSwag (66.1), and ARC (51.7). Math is competitive (MATH 21.0, higher than Llama3.2 19.5 and on par with Qwen 19.6), but GSM8K trails Qwen2.5‑1.5B‑Instruct (48.8 vs 63.3). Code is mixed: HumanEval 28.1 (below Qwen 30.5 and Llama3.2 33.5).
  - Long-context:
    > Appendix G, Figure 7: Needle-in-a-Haystack shows robust retrieval across depths at 8k tokens.
    > Appendix G, Table 11: On HELMET, SmolLM2’s LongQA score is highest (33.00 vs Llama3.2 21.99 vs Qwen2.5 26.23), though averages on ICL and recall favor larger baselines.

- Do the experiments support the claims?
  - The staged mixture changes correlate with clear improvements in domain capabilities (Table 3, Table 8). The targeted dataset designs (`FineMath`, `Stack‑Edu`) yield measurable gains in their domains (Figure 1, Table 2).
  - On general benchmarks, the base model is competitive or superior to peers on many tasks (Table 4), supporting the premise that data-centric training can close the gap for small models. However, Qwen2.5 remains stronger in base math/code, highlighting room for further improvement.

- Ablations and robustness
  - The paper includes pretraining ablations for web/mix ratios (Table 1; Appendix B), math sets (Figure 5, Figure 1), and code filtering thresholds (Section 3.4). Decontamination procedures are described for math (Section 3.3.2).
  - A noteworthy training anomaly—an unexplained loss spike—appears in stage 3 after switching datasets, but metrics largely recover (Section 4.4).

## 6. Limitations and Trade-offs
- Heavy reliance on LLM-based data labeling
  - Key datasets (`FineMath`, `Stack‑Edu`, `SmolTalk`) depend on prompts to large instruction-tuned models to produce “silver labels” or synthetic SFT data (Sections 3.3.2, 3.4, 5.1). This can encode biases (e.g., stylistic preferences) and complicate exact reproducibility.

- Compute- and data-intensive strategy
  - Training to 11T tokens (beyond “Chinchilla-optimal” for 1.7B) is compute-expensive (Section 4), even if inference is cheap. The approach assumes access to trillions of tokens of curated web/code/math text and large GPU clusters.

- Manual, “online” mixture tuning
  - While practical, the schedule involves human-in-the-loop decisions (Section 4). It is unclear how to programmatically determine optimal mixture shifts; results may be sensitive to monitoring choices, benchmark selection, and timing of interventions.

- Domain gaps remain
  - Base math and code still trail Qwen2.5‑1.5B (Table 4). Even after post-training, GSM8K lags Qwen2.5‑Instruct (Table 5). This suggests either the need for more targeted data or stronger math/code training strategies (e.g., tool use, verifier training).

- Training stability and diagnostics
  - The stage‑3 loss spike (Section 4.4) illustrates fragility when switching mixtures mid-run. The exact cause is undiagnosed, pointing to gaps in data quality checks or optimization monitoring.

## 7. Implications and Future Directions
- Shifting emphasis from architecture to data curriculum for small LMs
  - This work shows small models can gain large‑model‑like behaviors by carefully curating what they see and when. The staged, end‑weighted “annealing” of high-quality math/code appears especially effective for cementing reasoning skills (Figure 1; Table 3).

- Open resources for the community
  - Releasing `FineMath`, `Stack‑Edu`, and `SmolTalk` enables new research on:
    - Data scoring for educational value and reasoning quality,
    - Automated mixture scheduling and “online” ablation frameworks,
    - Transfer of “late-stage upsampling” ideas to other domains (e.g., safety, multilingual).

- Practical applications
  - Deployable assistants with good instruction following and general reasoning (Table 5) for edge devices and cost-sensitive servers.
  - Coding aides that benefit from educationally filtered code corpora (Table 2).
  - Math tutors for mid/high‑school levels, where `FineMath` is targeted (Figure 1).

- Next research steps
  - Automate the mixture-control loop (e.g., Bayesian optimization over sampling rates; early-warning metrics to prevent loss spikes).
  - Stronger math/code pipelines: verification, tool-use, program-of-thought, or curriculum learning that increases difficulty progressively.
  - Data quality guarantees: human spot audits, cross-model label agreement, and bias/harms assessments of classifier-filtered corpora.
  - Scaling down/up: The paper also trains 135M/360M variants (Section 6); extending this recipe across sizes would clarify compute–data trade-offs and where diminishing returns begin.

Overall, SmolLM2’s main advance is not a new architecture but a measured, evidence-backed blueprint for how to “go big on data” for small models—what to include, when to include it, and how to construct missing pieces—delivering broad improvements with open, reproducible ingredients.
