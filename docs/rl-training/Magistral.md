# Magistral

**ArXiv:** [2506.10910](https://arxiv.org/abs/2506.10910)

## 🎯 Pitch

Magistral unveils Mistral’s first reasoning-centric LLMs and a fully scalable reinforcement learning pipeline that enables large language models to acquire advanced chain-of-thought skills without relying on distillation from prior reasoning models or costly critic supervision. By pushing the limits of pure online RL on verifiable math and code tasks, Magistral not only achieves state-of-the-art reasoning performance from scratch but also delivers open-source, multilingual models—marking a significant leap toward accessible and generalizable AI reasoning systems that excel in STEM, multilingual, and multimodal applications.

---

## 1. Executive Summary
Magistral introduces Mistral’s first “reasoning” LLMs and a scalable reinforcement-learning-from-verifiable-rewards (RLVR) stack that trains them without distilling chain-of-thought from earlier reasoning models. Using only online RL on verifiable math and coding tasks, the approach lifts the base `Mistral Medium 3` checkpoint to strong reasoning performance and yields an open‑sourced `Magistral Small (24B)` model.

## 2. Context and Motivation
- Problem addressed
  - How to train large language models to perform long, effective chains-of-thought on complex tasks (math, coding, STEM) without relying on distillation from prior reasoning models or expensive critic networks.
  - The work targets training stability, scale, and multilingual usability during long-form reasoning.

- Why this matters
  - Reasoning-centric models increasingly solve tasks that require multi-step derivations (e.g., competition math, competitive programming). A method that can push reasoning by pure RL reduces dependence on proprietary teachers and can generalize across modalities and languages.
  - Real-world impact includes better assistants for STEM education, software engineering, and scientific workflows; theoretical significance includes insights into RL algorithms (e.g., GRPO) for sequence models.

- Prior approaches and gaps
  - Previous systems (e.g., DeepSeek-R1) popularized RLVR with distillation cold-starts and KL-regularized PPO-like training. Challenges remain in:
    - Reliance on teacher traces.
    - Expensive critic models or KL computation.
    - Training instability and entropy collapse.
    - On-policy vs. throughput trade-offs at scale.
  - This work positions itself as a ground-up RL stack (Section 1) that:
    - Removes teacher traces for the main model (`Magistral Medium`).
    - Eliminates the KL penalty and critic.
    - Runs a fully asynchronous, GPU-to-GPU pipeline (Section 3, Figure 3).
    - Enforces reasoning language to match user language via reward shaping (Section 2.2.4, Figure 2).

## 3. Technical Approach
This section decomposes the system into four parts: RL algorithm, reward shaping, infrastructure, and data pipeline.

- RL algorithm: GRPO with stability-focused modifications (Section 2.1)
  - GRPO (Group Relative Policy Optimization) uses several generations `G` per prompt to compute a baseline from the group’s average reward. The model maximizes a clipped policy-gradient objective over token log-probability ratios, like PPO but without a critic.
  - Key modifications and why they matter:
    - Remove KL penalty:
      - Standard PPO often penalizes divergence from a reference policy via `DKL`. Here, the KL term is dropped to reduce compute and because GRPO diverges substantially in practice even with KL; keeping a reference adds overhead with little stability gain (Section 2.1 “Eliminating KL divergence”).
    - Length-normalized loss:
      - Sum token-wise losses across all generations and divide by the total number of tokens `∑|o_i|` (Section 2.1 “Loss normalization”). This prevents bias toward short or long generations within a group.
    - Advantage computation and normalization:
      - Per-sample advantage `Â_i = r_i − μ` where `μ` is the group’s mean reward; then normalize within each minibatch to zero mean and unit variance (Section 2.1 “Advantage normalization”). This follows large-scale RL practice to stabilize updates under reward-scale drift.
    - Clip-Higher to prevent entropy collapse:
      - Replace symmetric PPO clipping `[1−ε, 1+ε]` with a higher upper bound `ε_high` (e.g., 0.26–0.28 for Medium, 0.3 for Small; Section 2.1 “Relaxing the trust region’s upper bound”). This gives low-probability “insight tokens” more room to gain probability, encouraging exploration during long derivations.
    - Filter zero-advantage groups:
      - If all `G` generations for a prompt are equally correct or incorrect (zero variance in reward), the group contributes no gradient. Such groups are dropped to keep gradients informative (Section 2.1 “Eliminating non-diverse groups”).
  - Resulting loss (end of Section 2.1):
    - The final objective keeps minibatch-normalized advantages, length normalization, asymmetric clipping, and the constraint “use only groups with at least two different rewards.”

- Reward shaping (Section 2.2)
  - Goal: make outputs verifiable, long enough, and in the user’s language.
  - Four axes of evaluation during training:
    1) Formatting (Section 2.2.1)
       - Enforce one `<think> ... </think>` tag pair at the start; for math, require `\boxed{}` around the final answer; for code, require a fenced code block with language (triple backticks).
       - If formatting fails, assign reward `0` and stop; if it passes, assign `+0.1` and proceed.
    2) Correctness (Section 2.2.2)
       - Math: extract the last `\boxed{}` and compare to reference via symbolic normalization with multiple parsers and SymPy. Reward `+0.9` if correct (total 1.0 with formatting).
       - Code: extract the first fenced code block, compile C++20 with a 10s timeout (precompile `<bits/stdc++.h>` for speed), run 20 sampled tests (4s per test, 300MB). Reward `+0.9` if all pass.
    3) Length penalty (Section 2.2.3, Equation (1))
       - Encourage long thinking but discourage hitting hard cutoffs. With two lengths `l_max` and `l_cache`, add a penalty that linearly ramps from 0 to −0.1 as the sequence approaches `l_max`, and caps at −0.1 beyond `l_max`.
    4) Language consistency (Section 2.2.4)
       - Objective: the chain-of-thought and final answer should be in the user’s language.
       - Translate 10% of English problems into French, Spanish, Italian, German, Chinese, Russian.
       - Strip LaTeX/code and run fastText language ID on the problem, thoughts, and answer; if all match, add `+0.1`.
       - The system prompt (Figure 2) explicitly instructs to think and answer in the user’s language and be “as casual and as long as you want,” which was empirically found to raise entropy and exploration during RL.

- Asynchronous infrastructure (Section 3; Figure 3)
  - Roles:
    - `Generators`: produce many rollouts (completions and token log-probs) under the latest policy.
    - `Verifiers`: compute rewards by running the formatting, correctness, length, and language checks.
    - `Trainers`: perform gradient updates.
  - Why asynchronous:
    - Completion lengths are heavy‑tailed and change over training; synchronous batching leaves many GPUs idle. The system streams completions continuously to verifiers and trainers, and trainers broadcast updated weights back to generators via NCCL (GPU-to-GPU) without waiting for long generations to finish.
    - Mid-generation weight updates:
      - In‑flight sequences continue with a “slightly outdated” KV cache; the newest tokens are generated on newer weights. Empirically recomputing the cache is unnecessary, likely because clipped policy gradients compensate for mild off-policy effects (Section 3).
  - Batching and load balancing:
    - A batch is a fixed number of sequences (not tokens). Within each minibatch, sequences are split into token-budgeted microbatches with a greedy collation heuristic that reduces padding by 19% (Section 3 “Trainer optimization”).
    - If trainers bottleneck early (short generations), a bounded queue limits the degree of off-policy drift before updates.

- Data pipeline (Section 4)
  - Math (Section 4.1):
    - Start with ~700k problems; format-filter to 501k; two-phase difficulty filtering down to 38k (Table 1).
      - Phase 1: sample 16 solutions per problem using `Mistral Large 2`; drop problems that are never solved or trivially solved.
      - Phase 2: regrade the entire pool with a stronger, RL-trained 24B model; sample 16 solutions per problem and again drop the too-easy and the still-unsolved. If most generated answers agree but disagree with the reference, mark the reference as likely wrong and remove.
      - Multiple-choice problems are reformulated to open-ended answer statements; proofs/multi-part questions are removed for verifiability.
  - Code (Section 4.2):
    - Aggregate competitive-programming problems with tests; prune items without trustworthy tests; run all known solutions to filter tests, fix inconsistent tests by majority output, and generate extra tests when needed. Duplicate prompts to require Python or C++ solutions. Final: 35k problems.

- Training schedule for `Magistral Medium` (Section 5.2; Figure 4)
  - Keep three invariants across stages:
    1) Data difficulty increases as the model improves (drop solved problems and include harder ones).
    2) Non-penalized completion budget `l_max − l_cache` grows (16k → 24k → 32k) to avoid length stagnation.
    3) Keep KV-cache memory manageable by reducing batch/minibatch sizes (8k → 4k → 2k).

## 4. Key Insights and Innovations
- Pure RL without teacher traces can strongly raise reasoning skill (fundamental)
  - `Magistral Medium` trains only with RL on verifiable math/code—no SFT on reasoning traces—and lifts AIME’24 pass@1 from 26.8% to 73.6% (Table 2), a near-50 point gain. This challenges the belief that small or medium models must be bootstrapped with teacher CoTs.

- A simple, scalable GRPO variant works in practice (methodological)
  - The combination “no KL penalty + minibatch advantage normalization + Clip-Higher + zero-variance group filtering + length-normalized loss” yields stable, entropy-preserving RL on long outputs (Section 2.1). This differs from many PPO/GRPO recipes that rely on KL regularization or explicit entropy bonuses.

- Language-of-thought control via reward and prompting (practical capability)
  - Enforcing `<think>...</think>` and language-consistency rewards makes the model reason and answer in the user’s language (Section 2.2.4; Figure 2), with modest performance drop when evaluating translated benchmarks (Table 4).

- Asynchronous, on-GPU pipeline for online RL (systems)
  - Generators never idle for trainers; weights are broadcast mid-generation (Section 3; Figure 3). This design balances throughput with “on-policyness,” avoiding frequent stalls while keeping the latest tokens aligned with the latest policy.

- RL on text preserves or improves multimodal and tool-use capabilities (unexpected generalization)
  - Despite training on text-only data, vision reasoning benchmarks improve (e.g., MMMU-Pro Vision +12 points to 52.1%; Figure 10), and function-calling/instruction-following remain steady or slightly better (Table 6).

## 5. Experimental Analysis
- Evaluation setup (Section 5.1)
  - Benchmarks:
    - Math: AIME’24/’25, MATH-500.
    - Coding: LiveCodeBench v5/v6, Aider Polyglot.
    - STEM QA: GPQA.
    - General knowledge: text-only subset of Humanity’s Last Exam.
    - Multimodal: MathVista, MMMU, MMMU-Pro (Section 7.2; Figure 10).
  - Decoding:
    - Temperature 0.7 and top‑p 1.0 for math/GPQA; temperature 0.7 and top‑p 0.95 for coding.
    - Max tokens 40k for AIME and LiveCodeBench; 32k otherwise.
  - Statistical rigor:
    - AIME: average over 64 runs (report pass@1 and maj@64).
    - LiveCodeBench: average over 16 runs.
  - Baselines:
    - Base checkpoints (`Mistral Small 3`, `Mistral Medium 3`) and DeepSeek results (Table 2).

- Main quantitative results
  - RL-only on `Mistral Medium 3` → `Magistral Medium` (Table 2)
    - AIME’24: 26.8% → 73.6% pass@1; 43.4% → 90.0% maj@64.
    - AIME’25: 21.2% → 64.9% pass@1; 30.0% → 83.3% maj@64.
    - LiveCodeBench v5: 29.1% → 59.4%.
    - MATH-500: 91.0% → 94.3%.
    - GPQA: 59.6% → 70.8%.
    - Humanity’s Last Exam (text only): 4.4% → 9.0%.
    - Compared with DeepSeek results reported in the paper: performance lands between R1-Zero and R1, with competitive code/math outcomes, despite no reasoning SFT (Table 2; Figure 1 notes 90% maj@64 on AIME-24).
  - Distillation + RL for `Magistral Small (24B)` (Table 3)
    - Three variants (same 24B backbone):
      - SFT on `Magistral Medium` traces only.
      - RL-only from base.
      - SFT + RL (final `Magistral Small`).
    - Highlights:
      - AIME’24 pass@1: 65.4 (SFT) vs 65.8 (RL-only) vs 70.7 (SFT+RL).
      - AIME’25 pass@1: 55.6 vs 51.9 vs 62.8.
      - LCB v5: 52.2 vs 46.4 vs 55.8; LCB v6: 44.6 vs 42.4 vs 47.4.
      - GPQA: 63.4 vs 68.8 vs 68.2.
    - Takeaway: RL meaningfully improves over SFT alone even for 24B; combining SFT + RL is best overall.
  - Multilingual AIME’24 pass@1 (Table 4)
    - English 73.6%; others 63.7–69.3%. Drop is 4.3–9.9 points. All reasoning and answers are in the input language by design.
  - Cross-domain generalization (Table 5)
    - Train on math-only → LCB v5 improves from 22.7 (base) to 38.3; train on code-only → AIME’24 improves from 32.2 to 49.7. RL signals transfer across domains.
  - Capability preservation (Table 6)
    - Function calling internal benchmark: 87.2 → 87.4.
    - Instruction following (internal IFEval): 86.8 → 87.4.
  - Open-source-traces experiment (Figure 13)
    - First SFT on OpenThoughts + OpenR1 code traces, then RL on hardest data: RL adds >12% on AIME’25 and ~5% on LiveCodeBench over SFT, but slightly reduces GPQA Diamond (72.9% → 71.0%).

- Ablations and diagnostics
  - Batch/minibatch size (Section 6.3; Figure 6)
    - With fixed concurrent generation `n_async`, performance is stable when `n_batch = n_minibatch` and degrades when using multiple minibatches per batch (off-policy drift increases). Final training keeps `n_async / n_batch ≤ 2` and `n_batch = n_minibatch`.
  - Advantage normalization (Section 6.4; Figure 7)
    - Minibatch, group, or none: little difference on eval or length growth. The system standardizes on minibatch normalization.
  - Partial rewards for code (Section 7.4.1; Figure 11)
    - Fraction-of-tests-passed rewards speed up training (fewer discarded samples) but yield slightly worse final code performance (−2% on LiveCodeBench) and slower length growth. The system uses binary pass/fail for stronger signals.
  - Entropy targeting (Section 7.4.2; Figure 12)
    - Explicit entropy bonuses behave inconsistently by dataset and can destabilize training; raising `ε_high` is a more reliable way to maintain exploration.
  - Weight-space analysis (Section 7.1; Figures 8–9)
    - PCA around the final checkpoint shows a dominant direction where both mean reward and output length increase until the length penalty kicks in. Raw reward scales roughly logarithmically with mean output length (Figure 9), reinforcing the finding that “more thinking” is the main driver—up to a point.

- Do the experiments support the claims?
  - Yes, convincingly for the core claims:
    - Pure RL can achieve large gains on medium-scale models (Table 2).
    - SFT + RL is best for smaller 24B models (Table 3).
    - Language control, tool use, and multimodal generalization are evidenced by multilingual AIME (Table 4), capability tables (Table 6), and multimodal benchmarks (Figure 10).
  - Cautions:
    - Results rely on verifiable tasks (math/code), potentially limiting generalization to open-ended reasoning beyond those domains.

## 6. Limitations and Trade-offs
- Scope of rewards and data
  - RL uses verifiable tasks with rule-based or test-based graders (Section 2.2). This excludes proofs, multi-part reasoning, and many real-world tasks lacking programmatic verification. The approach may not directly optimize skills needed for open-ended dialogue or creativity.
- Entropy/exploration control without KL
  - Removing the KL penalty simplifies and speeds training, but increases the risk of distribution drift. The system counters this with Clip-Higher and careful `ε_high` tuning (Section 2.1). Still, the absence of an explicit reference constraint can lead to harder-to-predict behavior on out-of-domain tasks.
- Asynchrony and off-policy drift
  - Mid-generation weight updates create slight off-policy mixtures. The system reports good behavior empirically (Section 3), but robustness depends on batching ratios (`n_async / n_batch`) and clip settings (Section 6.3).
- Memory vs. length vs. batch size
  - Longer contexts increase KV‑cache memory, forcing smaller batches later in training (Section 5.2). This trades statistical efficiency for length growth and may limit scaling on smaller clusters.
- Multilingual breadth
  - Language consistency rewards cover six non-English languages, applied to 10% of problems (Section 2.2.4). Performance in languages outside this set is not quantitatively evaluated.
- Mixed outcomes in supplemental experiments
  - While RL after SFT on open-source traces improves math/code, it slightly reduces GPQA Diamond (Figure 13), suggesting possible trade-offs in knowledge-intensive QA.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that pure RLVR at scale—without distillation and without KL—can substantially boost reasoning. This lowers the barrier to training reasoning models in organizations that lack access to powerful teachers.
  - Provides an RL recipe where exploration is sustained via asymmetric clipping rather than explicit entropy/ KL, and shows that long‑context RL training can be stabilized in practice.

- Follow-up research enabled
  - Reward design beyond verifiable tasks:
    - Develop semi-verifiable or weakly supervised graders (self-consistency, external tools) to expand beyond math/code.
  - Better control of “thinking length”:
    - The observed logarithmic reward–length relation (Figure 9) motivates adaptive schedules that optimize marginal returns from longer chains-of-thought without unnecessary verbosity.
  - Off-policy correction in asynchronous pipelines:
    - Formal analysis and methods (e.g., importance weighting across policy updates) could further reduce bias when weights update mid-generation.
  - Multimodal RL without explicit vision rewards:
    - Since text-only RL improved multimodal reasoning (Figure 10), systematic studies could identify when and why cross-modal transfer occurs and how to amplify it.
  - Safety/alignment without KL:
    - Explore targeted constraints (format, tools, language) as lightweight alternatives to KL for maintaining desirable behaviors, with rigorous safety evaluations.

- Practical applications
  - STEM tutoring systems that explain in the student’s language, with explicit reasoning steps (`<think>`).
  - Competitive programming assistants that reliably pass tests and adhere to language/tooling constraints.
  - Agents that combine reasoning with tool calls, where function-calling ability is preserved (Table 6).
  - Multimodal assistants for scientific diagrams and math‑in‑vision problems, benefitting from unanticipated cross‑domain gains (Figure 10).

> Representative headline result (Table 2): “AIME’24 pass@1 improves from 26.8% (Mistral Medium 3) to 73.6% (Magistral Medium) with RL only; majority voting (maj@64) reaches 90.0%.”

> Mechanistic takeaway (Section 7.1; Figures 8–9): “Reward increases logarithmically with output length along a principal direction in weight space, until length penalties and hard cutoffs bind,” suggesting that controlled encouragement of longer chains-of-thought is a key driver of gains.

Overall, Magistral shows a viable, transparent path to reasoning-centric LLMs via pure RL, offers an open 24B model (`Magistral Small`), and contributes concrete systems and algorithmic insights that the community can build upon.
