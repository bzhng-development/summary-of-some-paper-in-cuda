# Phi-4‑reasoning Technical Report

**ArXiv:** [2504.21318](https://arxiv.org/abs/2504.21318)
**Authors:** Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, Piero Kauffmann, Yash Lara, Caio César Teodoro Mendes, Arindam Mitra, Besmira Nushi, Dimitris Papailiopoulos, Olli Saarikivi, Shital Shah, Vaishnavi Shrivastava, Vibhav Vineet, Yue Wu, Safoora Yousefi, Guoqing Zheng
**Institutions:** Microsoft Research

## 🎯 Pitch

The paper presents two innovative 14-billion-parameter reasoning models, `Phi-4-reasoning` and `Phi-4-reasoning-plus`, leveraging teachable prompts and chain-of-thought demonstrations to rival much larger models in tasks such as math, coding, and planning. This breakthrough in data-centric training and compact reinforcement learning offers a cost-effective solution that powers diverse applications on affordable hardware, while setting new benchmarks for model training and evaluation processes.

---

## 1. Executive Summary (2–3 sentences)
The report introduces two 14‑billion‑parameter reasoning models, `Phi-4-reasoning` (via supervised fine‑tuning) and `Phi-4-reasoning-plus` (SFT + a short phase of reinforcement learning), that use carefully curated “teachable” prompts and high‑quality chain‑of‑thought demonstrations to unlock inference‑time reasoning. Across math, science, coding, planning, and spatial tasks, these models approach or surpass much larger open‑weight baselines and show strong cost–accuracy trade‑offs, while also spotlighting evaluation pitfalls for small, stochastic benchmarks (Figures 1, 8–12; Table 1).

## 2. Context and Motivation
- Problem addressed
  - Most small open models struggle with complex, multi‑step reasoning that benefits from “thinking longer” at inference time. Existing strong reasoning models either are very large, proprietary, or rely on heavy RL; smaller distilled models often lose capability or require expensive inference budgets.
  - This work seeks a scalable, data‑centric path to teach a 14B model to reason step‑by‑step with inference‑time scaling, and to do so with transparent training recipes and robust evaluation.

- Why it matters
  - Practical impact: Reasoning models that fit on affordable hardware can power math tutoring, scientific QA, planning, code assistance, and agentic workflows without the cost of frontier models.
  - Scientific significance: Shows how careful data selection and outcome‑based RL interact to induce longer, more effective chain‑of‑thought (CoT) behavior; highlights reproducible evaluation practices for stochastic reasoning (Section 5.1.2; Figures 2, 9–12).

- Prior approaches and gaps
  - Distillation + RL for reasoning has been explored (e.g., DeepSeek‑R1 and distilled variants; Section 1, citations [21, 58, 59, 34, 61]). However:
    - Data selection is often coarse; prompts may be too easy, too hard, or unverified, limiting transfer.
    - Evaluation commonly reports single‑run scores on tiny benchmarks (e.g., AIME), which are highly variable (Figure 9).
  - This work positions itself as a data‑centric alternative: select “teachable” seeds lying near the base model’s capability boundary; generate high‑quality demonstrations; then add a compact but effective RL stage with a length‑aware reward (Sections 2–4).

## 3. Technical Approach
This section explains how the models are constructed and trained, the data pipeline, and how the RL stage works.

- Base and architectural edits (Section 3)
  - Start from `Phi-4` (14B). Modify two unused tokens into `<think>` and `</think>` to bracket the reasoning block.
  - Extend maximum context from 16K to 32K tokens by doubling the RoPE base frequency; this supports longer CoT traces during training and inference.
  - Supervised fine‑tuning (SFT) uses 1.4M curated prompt–response pairs totaling 8.3B tokens from math, coding, and safety/RAI domains (Section 3).

- Seed selection and data curation (Sections 2–2.2)
  - Build a large seed set (prompts) from filtered web data, licensed sources, and synthetic rewrites. Then aggressively filter to keep only “teachable” seeds:
    - “Teachable” = at the boundary of the base model’s capability and requiring multi‑step reasoning.
    - When no gold answers exist, create proxy ground truth using plurality from a strong reference model; measure difficulty as agreement gaps with weaker models (Section 2.1).
    - Use rubric‑based LLM evaluators to estimate required reasoning steps and filter accordingly (Section 2.1).
  - Synthetic rewriting for verifiability: convert hard‑to‑verify problems into formats with concise, checkable final answers, easing future RL (Figure 3).
  - Decontaminate against many benchmarks (AIME‑2024, MATH, GPQA, LiveCodeBench, OmniMATH, SWE‑Bench‑Verified, and more; Section 2.2); AIME‑2025 is post‑cutoff and thus clean.

- Supervised training recipe (Section 3; Figure 5; Figure 4)
  - Teacher signals: generate long CoT traces with `o3-mini` (medium/high “reasoning effort”) and place them inside `<think> ... </think>` followed by a concise final “Solution” (Section 3).
  - System message: a fixed reasoning prompt teaches consistent two‑part output: Thought in `<think>...</think>`, then a succinct Solution (Section 3.1, “Role of system message”).
  - Hyperparameters: AdamW, learning rate 1e‑5 (best among 1e‑6–2e‑5; Experiments 1–3 in Figure 5), linear warmup 450 steps, weight decay 1e‑4, global batch 32, context 32K, ~16K steps (Section 3).
  - Data mixture “additivity”: tune weights for clusters per domain (math, code, safety), then combine recipes; improvements persist across domains (Figure 5, experiments 6–12).
  - Training dynamics: accuracy improves steadily on AIME‑24 and GPQA‑Diamond (Figure 4a). Notably, average response length slightly decreases as SFT proceeds (Figure 4b), suggesting more efficient use of tokens as reasoning quality improves.

- Reinforcement learning phase (Section 4; Figure 7)
  - Algorithm: Group Relative Policy Optimization (`GRPO`), a PPO‑style method where each prompt yields a group of candidate completions; advantages are normalized within the group (Section 4.2).
  - Reward design (Section 4.1; Figure 6):
    - Length‑aware correctness reward `Racc_scaled`: encourages concise generations when correct and longer exploration when incorrect. Intuition: do not waste tokens if you’re on track; invest more when you’re not.
      - Correct answer: reward smoothly decays if the output becomes unnecessarily long beyond a threshold (`Lpos_control = 25,600` tokens).
      - Incorrect answer: reward is less negative for longer attempts up to a minimum threshold (`Lneg_control = 3,702` tokens), nudging more thinking before answering incorrectly.
    - Formatting penalties: missing EOS or malformed `<think>` tags receive negative overrides to promote well‑formed outputs.
    - Repetition penalty `Rrep`: discourages repeated 5‑grams above frequency thresholds.
    - Final reward: `Rfinal = (8/13)*Racc_scaled + (1/13)*Rrep`, combining accuracy dominance with light repetition control.
  - RL data and compute:
    - Focused exclusively on math with verifiable answers; 72,401 seeds available, subsampled 64 per iteration. Best checkpoint obtained after only ~90 steps (~6.4K problems × 8 samples each), using 32 H100s, LR 5e‑8, KL 0.001, entropy 0.001, max length 32K; outputs clipped at 31K to save 1K for prompts (Section 4.2).
  - RL dynamics and effects (Figure 7):
    - AIME‑24 accuracy increases by >10% within the first 90 steps (Figure 7a).
    - Accuracy correlates positively with response length (Figure 7c); reward correlates weakly with accuracy (Figure 7b).
    - Incorrect generations grow faster in length than correct ones (Figure 7d), matching the intended “think more when you’re wrong” design.
    - As more samples hit the 31K clip limit, total reward plateaus (Figure 7e), hinting at benefits from even larger context windows (64K+).

- Evaluation methodology and inference‑time scaling (Sections 5–5.1.4; Figures 1–2, 8–12, 17)
  - Standardized pipelines: reuse MathArena for HMMT and Eureka ML Insights for most tasks to ensure consistent prompts, judges, and extraction (Section 5).
  - Stochasticity handling: run many repetitions for small benchmarks (e.g., 50 independent runs for AIME‑2025; Figure 9) and analyze distributions rather than single‑run scores.
  - Test‑time compute scaling: majority‑of‑N or best‑of‑N improves accuracy markedly (Figures 2, 12, 17), revealing headroom if one can afford parallel sampling.

## 4. Key Insights and Innovations
- Data‑centric “teachable seed” curation with verifiable outputs
  - What’s new: using agreement gaps with weaker models and rubric‑based step assessments to select prompts “near the boundary” of base capability, then rewriting problems into easily verifiable forms (Sections 2–2.1; Figure 3).
  - Why it matters: SFT learns transferrable reasoning strategies rather than shallow pattern matching; measurable accuracy gains across math, code, planning, and spatial tasks (Figures 1, 8; Table 1).

- Structured reasoning format with `<think>` tags plus a stable system message
  - What’s new: repurposed tokens for explicit Thought/Solution structure and a single, consistent reasoning system message to teach formatting robustness (Section 3.1).
  - Why it matters: Rapid adoption of CoT structure early in SFT and stable formatting at inference (Figure 4). The model learns to be concise in the Solution while exploring in `<think>`.

- Length‑aware, outcome‑based RL that operationalizes “think more when you’re wrong”
  - What’s new: a reward `Racc_scaled` that penalizes overly long correct answers yet encourages longer exploration for (likely) incorrect ones; lightweight repetition and formatting penalties (Section 4.1; Figure 6).
  - Why it matters: With only ~6.4K problems and 90 RL steps, AIME accuracy jumps by >10% (Figure 7a), and generations become longer primarily when needed (Figure 7d).

- Evaluation that moves beyond single‑run reporting for tiny benchmarks
  - What’s new: distributional analyses over 50 runs on AIME‑2025 (Figure 9), per‑year breakdowns (Figure 10), and best‑of‑N/worst‑of‑N diagnostics (Figures 12, 17).
  - Why it matters: Demonstrates that single‑run AIME scores can differ by 5–10 points; Phi‑4‑reasoning‑plus’ accuracy distribution largely overlaps with `o3-mini-high` and is almost disjoint from `R1-Distill‑70B` (Figure 9), providing a more reliable comparative picture.

## 5. Experimental Analysis
- Evaluation setup (Sections 5, A; Table 3–4)
  - Benchmarks for reasoning: AIME‑2025 (30 items, post‑training), AIME‑83–24 (949 items), HMMT‑Feb‑2025 (30), OmniMATH (4,428), GPQA Diamond (198), LiveCodeBench 8/24–1/25, Codeforces (contests 1505–1536), TSP and 3SAT (new), BA‑Calendar (2,000), Maze (10×10) and SpatialMap (1,500 each). See Table 4 for sources.
  - General‑purpose: FlenQA (length‑controlled QA), IFEval (instruction following), ArenaHard (chat preference), HumanEvalPlus (code), MMLU‑Pro, Kitab (RAG‑style retrieval with constraints), Toxigen (toxicity detection), and internal PhiBench (Table 2).
  - Metrics: pass@1 accuracy (averaged over multiple runs), Elo for Codeforces, precision/recall for Kitab, length/accuracy trade‑offs (Figures 11, 14–16).
  - Baselines: `DeepSeek-R1`, `R1-Distill-Llama‑70B`, `o1`, `o1-mini`, `o3-mini‑high`, `Claude‑3.7‑Sonnet‑Thinking`, `Gemini‑2.5‑Pro/Flash‑Thinking`, and `Phi‑4` (Table 1; Figure 8). Temperatures and token limits in Table 3.

- Main quantitative results
  - Math and science (Table 1; Figures 1, 8, 10)
    - AIME‑2025 (50 independent runs): 
      > `Phi-4-reasoning-plus` 78.0%; `Phi-4-reasoning` 63.1%; `DeepSeek-R1` 70.4%; `R1-Distill-70B` 51.5%; `o3-mini-high` 82.5%; `o1` 71.4%.
    - AIME‑83–24:
      > `Phi-4-reasoning-plus` 89.4%; `Phi-4-reasoning` 83.1%; `DeepSeek-R1` 86.0%; `o3-mini-high` 93.0%.
    - OmniMATH:
      > `Phi-4-reasoning-plus` 81.9%; `Phi-4-reasoning` 76.6%; `DeepSeek-R1` 85.0%; `o3-mini-high` 74.6%.
    - GPQA‑Diamond:
      > `Phi-4-reasoning-plus` 69.3%; `Phi-4-reasoning` 67.1%; `R1-Distill-70B` 66.2%; `DeepSeek-R1` 73.0%; `o1` 76.7%; `o3-mini-high` 77.7%.
    - Per‑year AIME analysis shows large variance by year and a common dip in 1994 and 2025 (Figure 10).
  - Algorithmic, planning, and spatial (Figure 8)
    - BA‑Calendar planning:
      > `Phi-4-reasoning` 67.7%; `Phi-4-reasoning-plus` 65.6%; `DeepSeek-R1` 79.2%; `o1` 86.1%; `Claude` 88.5%.
    - TSP:
      > `Phi-4-reasoning-plus` 42.6% vs. `Phi-4-reasoning` 37.5%; `o3-mini-high` 56.4%; `DeepSeek-R1` 46.7%.
    - Maze and SpatialMap:
      > On Maze, both Phi‑4‑reasoning models score ~55–55.1–53.4% (vs. `o1` ~79.7%); on SpatialMap, both are ~73–74% with `o1` ~83.6% and `o3-mini-high` ~77.4%.
  - Coding (Table 1)
    - LiveCodeBench (8/24–1/25):
      > `Phi-4-reasoning` 53.8%; `Phi-4-reasoning-plus` 53.1%; `R1-Distill-70B` 57.5%; `DeepSeek-R1` 65.9%; `o1` 63.4%.
      - Note: RL focused on math and did not include coding seeds (Section 4), which explains smaller gains here.
    - Codeforces Elo (10 attempts per problem):
      > `Phi-4-reasoning` 1736; `Phi-4-reasoning-plus` 1723; `R1-Distill-70B` 1633; `DeepSeek-R1` 2029 (Table 1).
  - General‑purpose (Table 2; Figure 13)
    - FlenQA (3K‑token subset): 
      > `Phi-4-reasoning` 97.7%; `Phi-4-reasoning-plus` 97.9%; `GPT-4o` 90.8%.
      - Accuracy degrades less with longer contexts, and is insensitive to where key information appears (Figure 13).
    - IFEval (Strict): 
      > `Phi-4-reasoning-plus` 84.9% vs. `Phi-4` 62.3% and `GPT-4o` 81.8%.
    - ArenaHard: 
      > `Phi-4-reasoning-plus` 79.0% vs. `Phi-4` 68.1%.
    - HumanEvalPlus: 
      > `Phi-4-reasoning` 92.9% vs. `Phi-4` 83.5%.
    - MMLU‑Pro:
      > `Phi-4-reasoning-plus` 76.0% vs. `Phi-4` 71.5%.
    - Kitab (RAG subset):
      > With context, precision ~93–94% and recall ~75%—on par with `o3-mini` on this split; without context, precision rises with reasoning but recall can drop (Table 2).
    - Toxigen discriminative:
      > `Phi-4-reasoning` improves “toxic” detection (86.7%) but slightly lowers “neutral” (84.7%); `Phi-4-reasoning-plus` flips the trade‑off (77.3% toxic, 90.5% neutral). Aggregate trends and per‑group patterns shown in Figure 18.

- Robustness checks and ablations
  - SFT ablations: learning rate search; effect of synthetic math data (Figure 5, experiments 4–5); system message stability (Section 3.1).
  - Teacher strength and context length: `o3-mini` high‑effort produces stronger but longer traces; extending context to 32K enables training on longer CoT (Section 3.2).
  - Variance analysis: 50‑run KDE on AIME‑2025 (Figure 9) shows wide ranges for all models; best‑of‑N can substantially outperform average‑of‑N (Figures 12, 17).
  - Token efficiency: `Phi-4-reasoning-plus` uses ~1.5× more tokens than `Phi-4-reasoning` on average; token‑accuracy trade‑offs visualized per benchmark (Figure 11).

- Do the experiments support the claims?
  - Yes, with caveats. On uncontaminated AIME‑2025, `Phi-4-reasoning-plus` is competitive with much larger or proprietary models and clearly surpasses `R1-Distill‑70B` (Table 1, Figure 9). Gains generalize across OmniMATH and several general‑purpose tasks (Tables 1–2). However, coding lags `DeepSeek-R1` and `o1`, and planning/spatial tasks show room for improvement. The paper also convincingly demonstrates evaluation variance and the benefits of parallel test‑time compute (Figures 2, 9, 12, 17).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - RL data is math‑only; improvements in coding, planning, and spatial tasks come mostly from SFT and are smaller than in math (Sections 4, 5.1.3; Figure 8).
  - Teacher‑generated CoT (from `o3-mini`) provides high‑quality signals but may bias reasoning styles; reliance on a strong, proprietary teacher could limit fully open replication (Section 3.2).

- Computational and data constraints
  - Context window is 32K; RL training clips outputs at ~31K tokens, which caps “thinking depth” and flattens reward at high steps (Figure 7e). The report suggests 64K support would help (Section 4.2).
  - `Phi-4-reasoning-plus` consumes ~1.5× more tokens than `Phi-4-reasoning` on average; gains are largest in math but not universal (Figure 11).

- Evaluation challenges
  - Tiny, hard benchmarks like AIME (30 items) are highly stochastic; single‑run comparisons are unreliable (Figure 9). The report addresses this by running 50 seeds and showing distributions, but the broader community often does not.

- Safety and alignment
  - Automated RAI metrics show minor regressions relative to `Phi-4` (Section 5.3), and existing LLM judges may mis-handle long, non‑linear CoT traces (Section 5.3). Toxigen reveals trade‑offs between detecting toxicity and avoiding erasure (Table 2; Figure 18).

## 7. Implications and Future Directions
- How this work shifts the field
  - Demonstrates that a carefully curated SFT corpus plus a compact, outcome‑based RL stage can produce a 14B reasoning model that competes with much larger open‑weight models on complex math and generalizes beyond math (Figures 1, 8; Tables 1–2).
  - Establishes best practices for evaluation under stochasticity: multi‑run distributions, best‑/worst‑of‑N, and accuracy–token trade‑offs (Figures 2, 9–12, 17).

- Follow‑up research enabled/suggested
  - RL beyond math: extend GRPO with verifiable rewards for planning, spatial reasoning, and coding (Section 5.1.3).
  - Longer contexts: support 64K+ with interpolation or RoPE variants to reduce clipping and further raise RL ceilings (Section 4.2).
  - Smarter decoding: exploit the gap between average‑of‑N and best‑of‑N with better verifiers, selection policies, or confidence estimators to reduce parallel sampling costs (Figures 2, 12, 17).
  - Safer CoT evaluation: develop judges/verifiers designed for long, branching thoughts to reduce false positives/negatives in safety assessments (Section 5.3).

- Practical applications
  - Education and assessment (AIME/OmniMATH‑level problem solving), scientific QA (GPQA), calendar/constraint planning (BA‑Calendar), and long‑context reasoning tasks (FlenQA).
  - Cost‑sensitive deployments where 14B‑scale models with strong reasoning are preferred over massive frontier models, with the option to dial inference‑time compute–accuracy trade‑offs (Figures 1, 11).

> Bottom line: With rigorous data selection, explicit thought formatting, and a short, length‑aware RL phase, `Phi-4-reasoning` and `Phi-4-reasoning-plus` deliver competitive reasoning at 14B scale and set higher standards for how reasoning models should be trained and evaluated.
