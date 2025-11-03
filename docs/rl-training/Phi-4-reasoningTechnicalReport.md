# Phi-4-reasoning Technical Report

**ArXiv:** [2504.21318](https://arxiv.org/abs/2504.21318)

## 🎯 Pitch

This paper presents Phi-4-reasoning and Phi-4-reasoning-plus, two compact 14B parameter language models designed for deep multi-step reasoning through meticulously curated supervised fine-tuning and outcome-based reinforcement learning. Demonstrating that careful data selection and targeted training can endow small models with robust, transferable reasoning skills, these models achieve or exceed the performance of much larger open-weight competitors across domains like math, science, coding, and planning—showing that smaller models, with the right approach, can rival giants in both specialized and general tasks.

---

## 1. Executive Summary (2–3 sentences)
This paper introduces `Phi-4-reasoning` (14B parameters) and `Phi-4-reasoning-plus`, two small “reasoning” language models trained to produce structured, multi-step solutions that exploit inference-time compute. Through carefully curated supervised fine-tuning (SFT) on “teachable” prompts and a short outcome-based reinforcement learning (RL) phase, the models achieve competitive or superior performance to much larger open-weight models across math, science, coding, planning, and spatial reasoning, while also improving general-purpose capabilities (Figures 1, 8; Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Modern LLMs increasingly rely on “reasoning” behavior: decomposing problems into steps, reflecting, and exploring solutions at inference time. However, small models typically lack such behavior unless trained on long, high-quality reasoning traces and encouraged to use more compute when needed.
  - The paper targets how to build a strong small reasoning model by combining data-centric SFT and a brief RL stage on a carefully filtered set of problems near the base model’s capability boundary (Sections 1–3).

- Why this matters
  - Practical significance: Smaller models with strong reasoning are cheaper to run and easier to deploy on limited hardware (e.g., edge devices). They can improve accuracy on complex, verifiable tasks (math, coding) and transfer gains to general-purpose use (Table 2).
  - Methodological significance: The work shows data curation and SFT can impart robust, transferable reasoning “meta-skills” to a small model, and that a little RL can amplify those gains (Figures 5, 7, 8).

- Prior approaches and gaps
  - Distillation of large reasoning models into smaller ones (e.g., DeepSeek-R1 distilled variants) has been effective, and reinforcement learning further improves them (Section 1; citations [21, 59, 34, 15]).
  - Gaps:
    - Limited clarity on how to choose prompts that are maximally “teachable” for SFT.
    - Lack of transparent, robust evaluation practices for reasoning models—most rely on small benchmarks with high variance (Sections 1, 5.1.2; Figures 2, 9–10).

- Positioning
  - The paper builds on a “data-centric” tradition from the Phi and Orca lines (careful synthetic data curation; [20, 28, 1, 2, 41, 38, 39]).
  - It combines: (a) high-quality, teacher-generated reasoning traces, (b) explicit format tokens for structured thinking, (c) an SFT recipe tuned for reasoning, and (d) a short RL phase with a length-aware, rule-based reward (Sections 2–4).

Definitions used once:
- `inference-time scaling`: allowing the model to spend more compute (e.g., generate longer chains of thought or multiple samples) on harder problems at inference time (Section 1).
- `chain-of-thought (CoT)`: an explicit, step-by-step reasoning trace in the output.
- `GRPO`: Group Relative Policy Optimization, a policy gradient method that normalizes reward within a group of sampled outputs to stabilize RL (Section 4).

## 3. Technical Approach
This section walks through the full pipeline: data, SFT, and RL, explaining what is done and why.

- Data methodology (Section 2)
  - Seeds database
    - Collects diverse, reasoning-heavy prompts from the web, datasets, licensed sources, plus synthetic problems (Section 2.1).
    - “Teachable” prompts are those near the base model’s competence boundary (hard enough to learn from, not impossibly hard). The team identifies them by:
      - Using a strong reference model’s plurality answer as a proxy ground truth when no gold labels exist.
      - Measuring agreement of weaker models (e.g., `Phi-4`, GPT-4o) with that proxy; low agreement signals room to learn.
      - Using rubric-based LLM evaluators to estimate required reasoning steps/difficulty (Section 2.1).
    - Result: a curated set of seeds emphasizing multi-step reasoning over factual recall (Section 2.1).

  - Synthetic augmentation for verifiability
    - Some seeds are rewritten to enable verification and to elicit concise, checkable final answers, which facilitates both SFT and RL. Example: a free-form geometry proof is converted into a numeric question with a definite answer (Figure 3).

  - Teacher generations and format
    - Responses (both reasoning traces and final answers) are generated with `o3-mini` and stored in a structured format that separates a “thinking” block from the final “answer” (Sections 2.2, 3).
  
  - Safety/RAI data
    - Prompts cover safety guidelines (e.g., disclaimers, confidentiality of chain-of-thought). During training, guidelines are removed from the prompt to encourage implicit adherence; models are trained not to reveal the guidelines or CoT in the final “answer” (Section 2.2).

  - Decontamination
    - The SFT data is systematically decontaminated against many evaluation benchmarks (AIME-2024, MATH, GPQA, LiveCodeBench, etc.). AIME-2025 was released after data finalization, ensuring no contamination (Section 2.2).

- Supervised fine-tuning: `Phi-4-reasoning` (Section 3)
  - Base model and architecture changes
    - Starts from `Phi-4` (14B). Two placeholder tokens are repurposed as `<think>` and `</think>` to demarcate the reasoning block (Section 3).
    - Context length doubled from 16k to 32k by doubling RoPE base frequency (a common positional embedding technique) to store longer reasoning traces (Section 3).
  - Training data and recipe
    - 1.4M prompt–response pairs; 8.3B tokens; domains include math, code, and safety (Section 3).
    - Training for ~16k steps, global batch size 32, context length 32k; AdamW; LR 1e-5; 450-step warmup; weight decay 1e-4 (Section 3).
    - A fixed “reasoning” system message prompts the model to structure output into `<think>...` and `</think>` for Thought, followed by a concise Solution (Section 3.1). Variants were tried; the final message improved robustness but too much variation hurt consistency under the chosen evaluation setting.
  - Data mixture optimization
    - Clusters sources by domain and quality, then tunes per-cluster weights.
    - Key finding: an “additive property” — optimize mixtures per domain independently (e.g., math vs. code), then concatenate. Domain-specific gains persist in the combined mixture (Section 3.1; Figure 5).
  - Teacher choice and token budget
    - `o3-mini-high` provides stronger traces than medium-effort but increases token usage; to benefit, the team raises model context to 32k and tests up to 64k at inference in some evaluations (Section 3.2; Table 3).
  - SFT dynamics
    - Accuracy on AIME-2024 and GPQA increases throughout training (Figure 4a).
    - Mean response length slightly decreases over SFT (Figure 4b), suggesting efficiency gains as the model learns to reason more crisply (not just pad).

- Reinforcement learning: `Phi-4-reasoning-plus` (Section 4)
  - Scope and setup
    - RL only on mathematics, using GRPO with 72,401 math seeds; each RL iteration subsamples 64 problems (Section 4).
    - The selected final checkpoint is after 90 RL steps on ~6.4k problems (with 8 sampled trajectories each), chosen by best AIME-2024 score (Section 4.2).
    - Compute: 32 Nvidia H100 GPUs, global batch size 64; LR 5e-8 with short warmup; GRPO group size G=8; KL β=0.001; entropy γ=0.001; max output length 31k tokens (1k reserved for prompt) (Section 4.2).

  - Rule-based reward, designed to shape “thinking effort” (Section 4.1)
    - Primary term is length-aware accuracy `R_acc_scaled`: reward correct answers more when concise and encourage longer “thinking” when wrong.
      - Key thresholds: `Lmax=31744`, `L_pos_control=25600` (no length penalty when correct and shorter than this), `L_neg_control=3702` (no length penalty when incorrect and at least this long).
      - Correct answer range: 0.5 to 1.0; Incorrect: –1.0 to –0.5; cosine scaling adjusts reward smoothly with length (Figure 6; Section 4.1).
    - Format penalties: missing end-of-sequence or malformed `<think>` tags incur default negative rewards (Section 4.1).
    - Repetition penalty `R_rep`: punishes frequent 5-gram repetition based on frequency thresholds (Section 4.1).
    - Final reward: `R_final = (8/13) * R_acc_scaled + (1/13) * R_rep` (Section 4.1).

  - Observed RL dynamics (Figure 7)
    - Accuracy improves >10% on AIME with only 90 steps (Figure 7a).
    - Longer responses correlate with higher AIME accuracy (Figure 7c).
    - As intended, incorrect answers grow longer faster than correct ones, indicating the policy “thinks more” when it’s likely wrong (Figure 7d).
    - However, excessive length can hit the 31k limit and clip outputs before the final boxed answer, capping reward (Figure 7e). Entropy stays “healthy,” indicating continued exploration (Figure 7f).

## 4. Key Insights and Innovations
- Curating “teachable” prompts near the model’s current boundary is crucial (Sections 2.1–2.2).
  - What’s new: a systematic, LLM-assisted filtering pipeline that estimates difficulty and multi-step requirements without ground truth, then rewrites a subset for verifiability (Figure 3).
  - Why it matters: SFT on this subset generalizes far beyond math/coding into planning and spatial tasks (Figure 8), and even boosts general-purpose benchmarks (Table 2).

- Structured “reasoning tokens” with a tuned system message (Section 3)
  - What’s different: explicit `<think>...</think>` tags and a detailed system message yield robust, consistent CoT behavior without sacrificing general skills (Figure 4; Table 2).
  - Significance: The model learns format quickly but improves the substance of reasoning over time; response length decreases while accuracy increases (Figure 4).

- Length-aware, rule-based RL reward (Section 4.1; Figure 6)
  - What’s different: a simple, verifiable, and interpretable reward that encourages the right amount of “thinking” — concise when correct, longer when uncertain.
  - Significance: With only ~6.4k math problems and 90 RL steps, the method yields double-digit AIME gains and longer, more productive reasoning traces (Figure 7a, 7c–7d).

- Evaluation practice emphasizing variance and compute–accuracy trade-offs (Sections 5.1.2, 5.1.4)
  - What’s different: reports distributions over 50 runs for AIME-2025 (Figure 9), best/average/majority-of-N aggregation (Figures 12, 17), and accuracy vs. token usage scatterplots (Figure 11).
  - Significance: Demonstrates that single-run scores on tiny benchmarks are unreliable and highlights headroom if better decoding or verification extracts the best trajectories.

## 5. Experimental Analysis
- Evaluation setup (Section 5)
  - Benchmarks for reasoning (Table 4; Figure 8)
    - Math: AIME 2025 (30 items; contamination-free), AIME 1983–2024 (949), HMMT Feb 2025 (30), Omni-MATH (4,428).
    - Science: GPQA Diamond (198).
    - Coding: LiveCodeBench 8/2024–1/2025, Codeforces (Elo measured from contest IDs 1505–1536).
    - Algorithmic/Planning: 3SAT-Search (800), TSP-Opt (960), BA-Calendar (2,000).
    - Spatial: Maze (1,500; 10×10), SpatialMap (1,500).
  - General-purpose (Table 2): FlenQA (long-context T/F), IFEval (instruction following), ArenaHard (chat preferences), HumanEvalPlus (code), MMLU-Pro, Kitab (retrieval-like constraint satisfaction), Toxigen (toxicity detection), and internal PhiBench 2.21.
  - Baselines and decoding (Section 5.1.1; Table 3)
    - Closed: `o1`, `o1-mini`, `o3-mini-high`, GPT-4o, Claude 3.7 Sonnet, Gemini 2.5 Flash/Pro.
    - Open: DeepSeek-R1, R1-Distill-Llama-70B, EXAONE-Deep-32B, OpenThinker2-32B, QwQ-32B.
    - Temperatures: Phi models at 0.8 for reasoning; DeepSeek family 0.6; others default or 1.0. Max tokens set high where possible; Phi-4-reasoning models sometimes allowed 65,536 at inference on select evals (Table 3).

- Headline results (quantitative)
  - Reasoning (Table 1; Figures 1, 8)
    - AIME 2025 (pass@1, 50 runs; Figure 1 and Table 1):
      > `Phi-4-reasoning`: 63.1% (±6.3), `Phi-4-reasoning-plus`: 78.0% (±4.6) vs. DeepSeek-R1-Distill-Llama-70B: 51.5% (±5.8) and DeepSeek-R1: 70.4% (±4.3). `o3-mini-high`: 82.5% (±4.9), `o1`: 71.4% (±5.7).
    - AIME 1983–2024 (Figure 8):
      > `Phi-4-reasoning`: 83.1%, `Phi-4-reasoning-plus`: 89.4%; DeepSeek-R1: 86.0%; o3-mini-high: 93.0%.
    - Omni-MATH (Table 1; Figure 8):
      > `Phi-4-reasoning`: 76.6% (±0.5), `Phi-4-reasoning-plus`: 81.9% (±0.1) vs. R1-Distill-70B: 63.4% and DeepSeek-R1: 85.0%. `o1`: 67.5%, `o3-mini-high`: 74.6%.
    - GPQA Diamond (Table 1; Figure 8):
      > `Phi-4-reasoning`: 67.1% (±2.7), `Phi-4-reasoning-plus`: 69.3% (±2.1), DeepSeek-R1: 73.0% (±1.7), `o3-mini-high`: 77.7% (±0.6).
    - Coding (Table 1):
      > LiveCodeBench: `Phi-4-reasoning`: 53.8%, `Phi-4-reasoning-plus`: 53.1% vs. DeepSeek-R1: 65.9%.
      > Codeforces Elo (protocol described in Section 5.1.3): `Phi-4-reasoning`: 1736, `Phi-4-reasoning-plus`: 1723 (DeepSeek-R1: 2029).
    - Algorithmic/Planning/Spatial (Figure 8):
      > BA-Calendar: `Phi-4-reasoning`: 67.7%, `Phi-4-reasoning-plus`: 65.6% vs. DeepSeek-R1: 79.2%, `o3-mini-high`: 88.5%.
      > TSP: `Phi-4-reasoning`: 37.5%, `Phi-4-reasoning-plus`: 42.6% vs. DeepSeek-R1: 46.7%, `o3-mini-high`: 56.4%.
      > Maze: `Phi-4-reasoning`: 55.1%, `Phi-4-reasoning-plus`: 53.4% vs. DeepSeek-R1: 47.5%, `o3-mini-high`: 79.7%.
      > SpatialMap: `Phi-4-reasoning`: 73.7%, `Phi-4-reasoning-plus`: 73.3% vs. DeepSeek-R1: 76.7%, `o3-mini-high`: 77.4%.
  - General-purpose (Table 2)
    - FlenQA (3k-token subset): 
      > `Phi-4-reasoning`: 97.7%, `Phi-4-reasoning-plus`: 97.9% vs. GPT-4o: 90.8%; `o3-mini-high`: 96.8%.
    - IFEval (Strict instruction following):
      > `Phi-4-reasoning`: 83.4%, `Phi-4-reasoning-plus`: 84.9% vs. GPT-4o: 81.8%; `o3-mini-high`: 91.5%.
    - ArenaHard (chat preference):
      > `Phi-4-reasoning`: 73.3%, `Phi-4-reasoning-plus`: 79.0% vs. GPT-4o: 69.0%; `o3-mini-high`: 81.9%.
    - MMLU-Pro: 
      > `Phi-4-reasoning`: 74.3%, `Phi-4-reasoning-plus`: 76.0% vs. GPT-4o: 73.5%; `o3-mini-high`: 79.4%.

- Robustness and analysis beyond single scores
  - Variance on small benchmarks
    - Accuracy distributions over 50 runs on AIME-2025 show large variance for all models (Figure 9). For example, R1-Distill-70B ranges 30–70%; `o3-mini-high` 70–100%. `Phi-4-reasoning-plus` overlaps strongly with `o3-mini-high` and is largely disjoint from R1-Distill-70B.
    - Performance varies significantly by year within AIME (Figure 10), with recent years being harder for most models.
  - Best-of-N potential and compute–accuracy trade-offs
    - Majority/best-of-5 often notably exceeds average-of-5 across tasks (Figures 12, 17), revealing latent capability that better decoding/verification could harness.
    - Accuracy vs token usage shows `Phi-4-reasoning-plus` uses ~1.5× more tokens than `Phi-4-reasoning` on average; `Phi-4-reasoning` is comparable to `o3-mini-high` in token use (Figure 11).
    - Increasing parallel test-time compute narrows the gap to the teacher and may surpass it on AIME-2025 under extensive parallelization (Figure 2).

- Ablations and design choices (Figure 5; Section 3.1)
  - Hyperparameters: LR 1e-5 best balanced performance; weight decay had minor effect within variance.
  - Synthetic math (verifiable final answers) materially boosts AIME 2022–2024 by 3–10% (Figure 5, exps 4–5).
  - System message and `<think>` tags improve consistency; removing or randomizing messages increased variability and slightly reduced average scores under the chosen eval setting (Section 3.1).
  - Teacher choice: `o3-mini-high` better than medium but longer traces required extending context to 32k (Section 3.2).

- Where results are mixed or conditional
  - RL emphasized math; improvements outside math (e.g., planning, spatial) are smaller (Figure 8).
  - In science, gains are smaller in Biology and Chemistry vs. Physics (Figure 16).
  - In math subfields, Discrete Math and Geometry remain harder across models (Figure 15).
  - Coding wins on LiveCodeBench are modest; Codeforces Elo lags larger baselines (Table 1), consistent with no coding seeds in RL (Section 4.2).

Assessment: The experiments are thorough and transparent. They report distributions and standard deviations, use multiple benchmarks (many large), and analyze accuracy–length trade-offs and best-of-N potential. The claims about competitive small-model reasoning and the value of careful SFT plus a short RL phase are convincingly supported (Figures 1, 8; Tables 1–2).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - RL only uses math problems with verifiable answers; improvements outside math are less pronounced (Section 4.2; Figure 8).
  - The method assumes high-quality teacher traces are available for SFT; performance depends on teacher choice (`o3-mini-high` vs. medium; Section 3.2).

- Computational and token-length constraints
  - Inference cost: reasoning traces are long; `Phi-4-reasoning-plus` uses ~1.5× the tokens of `Phi-4-reasoning` (Figure 11).
  - RL max length is clipped at ~31k output tokens, sometimes cutting off final answers and capping reward (Figure 7e).
  - Model trained for 32k context; some evals use up to 64k without full training at that length (Table 3).

- Data and domain coverage
  - SFT domains emphasize math, code, and safety; RL focuses on math. Generalization is strong but not universal (Sections 3–4; Figure 8).
  - Coding competitiveness lags the strongest baselines (Table 1), consistent with lack of RL targeting coding.

- Evaluation caveats
  - Small benchmarks (e.g., AIME-2025 with 30 items) exhibit high variance, making single-run comparisons unreliable (Section 5.1.2; Figure 9).
  - Safety evaluation of long CoT traces is challenging; current LLM judges may misinterpret non-linear, lengthy reasoning (Section 5.3).

- Safety and disclosure
  - Reasoning traces can include safety guidelines-like text in the `<think>` block; the model is trained not to reveal CoT in the final answer, but open-weight settings raise open questions about exposure and faithfulness of explanations (Sections 2.2, 6; [55]).

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a 14B model, with the right data-centric SFT and a brief, well-shaped RL phase, can match or beat much larger open-weight baselines on complex reasoning tasks and improve general-purpose ability (Figures 1, 8; Table 2). This raises the bar for efficient small-model reasoning.

- Practical applications
  - Math tutoring and grading (verifiable answers; AIME/Omni-MATH gains).
  - Scientific QA and study aids (GPQA).
  - Code assistance on medium difficulty tasks (LiveCodeBench improvements over base, though not SOTA).
  - Scheduling/planning agents (BA-Calendar successes).
  - Spatial/logic puzzle solvers (Maze, SpatialMap).

- Research avenues
  - Broaden RL beyond math with verifiers for coding, planning, and spatial tasks; strengthen exploration and verification outside math (Sections 4, 5.1.3).
  - Improve decoding/verification to harvest best-of-N behavior without large inference multipliers (Figures 12, 17).
  - Extend context length reliably to 64k+ via principled RoPE interpolation; reduce clipping harms in RL (Section 4.2).
  - Develop safety tooling tailored to long, non-linear CoT, minimizing false triggers and better catching subtle biases (Section 5.3).
  - Establish evaluation standards for reasoning: multiple independent runs, accuracy distributions, length–accuracy trade-offs, and reporting majority/best-of-N (Sections 5.1.2, 5.1.4).

In short, this paper provides a clear, replicable recipe—curated “teachable” SFT with explicit reasoning structure plus a compact, length-aware RL loop—that turns a small model into a capable reasoner, while also modeling good evaluation hygiene for a volatile domain.
