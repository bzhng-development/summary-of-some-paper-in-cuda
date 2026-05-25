# Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?

**ArXiv:** [2504.13837](https://arxiv.org/abs/2504.13837)

## 🎯 Pitch

This paper fundamentally reevaluates the transformative potential of reinforcement learning with verifiable rewards (RLVR) for large language models, rigorously testing whether RLVR actually expands reasoning abilities beyond those present in the pretrained base models. The authors find that while RLVR boosts sampling efficiency for existing reasoning paths, it does not introduce novel reasoning capabilities—instead, models remain bounded by their initial pretrained knowledge. This finding has profound implications: it challenges the assumption that RLVR alone can drive open-ended LLM self-improvement and guides the field toward new research directions, such as better exploration strategies and multi-turn interactions, to genuinely unlock new reasoning skills.

---

## 1. Executive Summary
This paper asks whether today’s reinforcement learning with verifiable rewards (RLVR) actually grows a language model’s reasoning capacity beyond what the pretrained “base” model already contains. Using large‑k pass@k evaluations across math, coding, and visual reasoning, plus coverage and perplexity analyses, the paper finds that current RLVR mainly reweights the base model’s existing reasoning paths to make correct samples easier to draw at small k, while narrowing the overall set of problems the model can solve with extensive sampling. Distillation from a stronger teacher, by contrast, does expand the reasoning boundary.

## 2. Context and Motivation
- Problem and gap
  - Reasoning‑centric LLMs (e.g., o1/R1‑style models) owe much of their progress to RLVR—RL against automatic verifiers such as unit tests or exact numeric answers (Section 1; Section 2.1). The common belief is that, like classic RL in games, RLVR makes models explore and acquire new strategies beyond what pretraining gave them.
  - The unresolved question: Does RLVR truly create novel reasoning patterns, or does it only exploit ones already present in the base model (Section 1)?
- Why it matters
  - Practical: If RLVR merely reweights existing abilities, the upper bound on reasoning is set by the base model; scaling RLVR alone won’t yield open‑ended improvement.
  - Scientific: Clarifies how much “self‑improvement” current RL training actually yields, and where to invest research effort (better exploration, multi‑turn interaction).
- Prior approaches and shortcomings
  - Instruction tuning and distillation improve reasoning but require curated or teacher‑generated traces.
  - RLVR promises scalable self‑improvement with cheap verifiers, but prior evaluations favor average‑case metrics (greedy decoding) that can hide a model’s latent ability if you give it more attempts (Section 2.2).
- Positioning
  - The paper reframes evaluation around reasoning “coverage” via pass@k at large k (how many unique problems a model can solve if you sample many times), then probes whether RLVR expands coverage beyond the base model (Figure 1; Section 2.2; Appendix A.2).

## 3. Technical Approach
- What is RLVR?
  - An LLM `πθ` generates a solution `y` for a prompt `x`. A deterministic `verifier V` returns reward `r∈{0,1}` (correct/incorrect). The objective is to maximize expected reward over the prompt distribution `D`: `J(θ) = Ex∼D Ey∼πθ [r]` (Section 2.1).
  - Typical algorithm: PPO with a clipped surrogate (Equation 1). Several critic‑free variants are considered (GRPO, RLOO, Reinforce++, ReMax, DAPO; Appendix A.1).
  - “Zero‑RL” means applying RL directly to the pretrained base model without first doing supervised fine‑tuning (SFT). This setting is used for math; coding/vision start from instruction‑tuned models given training instability otherwise (Section 2.1).
- Measuring reasoning capacity boundary
  - pass@k: For each problem, sample k outputs; pass@k=1 if any passes the verifier, else 0. Averaging over the dataset estimates the fraction of problems solvable within k attempts—i.e., coverage of the model’s reasoning boundary (Section 2.2).
  - Low‑variance estimation: Generate `n≥k` samples per item, count `c_i` correct samples, and estimate pass@k by the unbiased estimator in Equation (2): `1 - C(n−c_i, k)/C(n, k)` (Appendix A.2).
  - Why pass@k (large k) rather than best‑of‑N or majority vote? The aim is to test “could the model solve it at all if you try enough times,” not just “what you’d pick automatically” (Section 2.2).
  - Guarding against guesswork in math: Manual audits of sampled chains‑of‑thought (CoTs) on the hardest problems show that correct answers generally come with valid reasoning, not random guessing (Section 3.1; Appendix D.2).
- Experimental design (Table 1; Section 3)
  - Tasks: mathematics (GSM8K, MATH500, Minerva, Olympiad, AIME24, AMC23), coding (LiveCodeBench, HumanEval+, MBPP+), and visual reasoning (MathVista, MathVision).
  - Models and RL systems:
    - Math: Qwen2.5‑Base 7B/14B/32B and LLaMA‑3.1‑8B; RLVR models include SimpleRLZoo zero‑RL (GRPO), Oat‑Zero‑7B, and DAPO‑32B (Sections 3.1, D.1).
    - Code: Code‑R1‑Zero‑Qwen2.5‑7B‑Instruct; DeepCoder‑14B trained on R1‑style RL (Section 3.2).
    - Vision: EasyR1 training of Qwen2.5‑VL‑7B on Geometry3K; evaluated on MathVista‑TestMini and MathVision‑TestMini (Section 3.3).
  - Evaluation protocol: temperature 0.6, top‑p 0.95, max 16,384 tokens; same zero‑shot prompts as used in RL training; no few‑shot for base models to avoid confounds (Section 3).
- Additional diagnostics
  - Accuracy histograms across problems to see whether RLVR shifts accuracy mass (Section 4.1; Figure 5, Figure 14).
  - Perplexity (`PPL`) analysis: compute likelihood of RL outputs under the base model; if low PPL, the base likely already generates those outputs (Section 4.1; Figure 6; Appendix D.4).
  - Coverage set comparison: For each benchmark, enumerate which problem indices are solved by base vs. RL model to test subset relations (Table 2; Appendix D.7).
  - Sampling Efficiency Gap (`ΔSE`): difference between RL pass@1 and Base pass@256 (proxy for upper bound). Lower is better; it measures how close RL’s one‑shot performance comes to the base model’s best‑of‑many upper bound (Section 4.3; Figure 8).

## 4. Key Insights and Innovations
- Insight 1 — RLVR improves sampling efficiency but narrows reasoning coverage
  - What is new: Prior work emphasized improved pass@1 (average case). This work looks at full pass@k curves and shows the base model usually overtakes the RL model as k grows (Figure 2; Figure 4; Figure 12; Figure 13).
  - Why it matters: It distinguishes “being more likely to hit a known good path” from “learning fundamentally new paths.” The right panel of Figure 1 visualizes this: the RL policy concentrates probability on rewarded paths (more black near green), but eliminates other paths the base could explore, reducing coverage.
- Insight 2 — RLVR solutions live inside the base model’s distribution
  - Evidence: Perplexity distributions show that `PPL_base(Y_RL | x)` overlaps with the lower portion (i.e., high‑probability region) of `PPL_base(Y_base | x)`, meaning RL outputs are already likely under the base model (Figure 6). Over training, `PPL_base(Y_RL | x)` decreases further (Appendix D.4).
  - Why it matters: It indicates RLVR sharpens the prior rather than extending it to new reasoning modes.
- Insight 3 — Coverage set analysis: RLVR rarely solves problems the base cannot
  - Evidence: On AIME24 and MATH500, the fraction of problems solvable only by RL is 0.0% and 1.0% respectively at tested k (Table 2). Detailed indices show the RL‑solved set is nearly a subset of the base’s (Appendix D.7, Tables 5–6).
- Insight 4 — Distillation vs. RLVR
  - Distillation from a stronger teacher expands the reasoning boundary (pass@k curve strictly above base across k on Minerva; Figure 7), unlike RLVR which stays bounded by the base model’s potential.
- Insight 5 — Current RL algorithms behave similarly and are far from “upper bound” efficient
  - Using `ΔSE`, diverse algorithms (PPO, GRPO, RLOO, Reinforce++, ReMax, DAPO) differ only slightly and all leave a large gap (Figure 8 top; Table 3). This suggests algorithmic room for improvement in sampling efficiency.

## 5. Experimental Analysis
- Setup and metrics
  - Datasets and models summarized in Table 1; evaluation uses pass@k with large k (up to 1024) and the unbiased estimator in Equation (2).
  - Prompts are matched across base and RL models; base models are evaluated zero‑shot to avoid hidden advantages from few‑shot exemplars (Section 3).
  - Manual CoT validity check shows correct answers for hard math problems typically come with coherent reasoning (Section 3.1; Appendix D.2).
- Core quantitative findings
  - Math (Figure 2; Appendix Figure 10, Figure 11):
    - At small k (e.g., pass@1), RL improves substantially; e.g., Qwen2.5‑7B on Omni‑MATH‑Train rises from pass@1=9.9 (base) to 26–31 (various RL algorithms; Table 3).
    - As k increases, base overtakes: “base models consistently catch up and surpass RL‑trained models across all benchmarks” (Figure 2). Example: On Minerva with a 32B model, the base beats RL by roughly 9% at k=128 (Section 3.1).
    - Training longer exacerbates the pattern: pass@1 increases from 26.1 to 42.5, while pass@256 falls (Figure 1 right; Table 4).
  - Coding (Figure 3; Figure 4 left; Figure 12):
    - Same trend on LiveCodeBench, HumanEval+, MBPP+: gains at low k, base catches up and surpasses as k grows. This is especially compelling because unit tests eliminate the “lucky guess” concern present in math (Section 3.2).
  - Visual reasoning (Figure 4 right):
    - Qwen2.5‑VL‑7B trained with EasyR1 on Geometry3K shows the same pattern on MathVista and MathVision: RL better at low k, base superior at larger k.
  - Accuracy distribution and coverage
    - Histograms show RL increases frequency near accuracy=1.0 but also increases the mass at 0.0 (unsolved), indicating more problems become unsolvable after RL (Figure 5; Appendix Figure 14).
    - Coverage analysis confirms RL‑solved problems are almost entirely within the base‑solvable set (Table 2; Appendix D.7).
  - Perplexity
    - RL outputs have low perplexity under the base model (`PPL_base(Y_RL|x)`), overlapping the base’s own high‑probability region; the overlap tightens as RL progresses (Figure 6; Appendix D.4).
  - Algorithmic ablations and controls
    - Algorithms: Minor differences across PPO/GRPO/RLOO/Reinforce++/ReMax/DAPO; `ΔSE` remains large (Figure 8 top; Table 3).
    - Training steps: pass@1 rises, pass@256 declines (Figure 1 right; Table 4).
    - Rollouts per prompt (`n`): Increasing from 8 to 32 slightly improves high‑k performance but still trails the base (Figure 16).
    - KL regularization: Adding KL=0.001 maintains pass@1 but lowers high‑k coverage (Figure 16).
    - Entropy: RL training reduces output entropy; even after temperature‑matching entropy to the base, RL still underperforms at large k (Figure 18).
  - Larger near‑frontier model check
    - For Magistral‑Medium (pure RL) vs. its base Mistral‑Medium‑3, RL again gives sizable k=1 gains but little/no improvement at high k; e.g., about +7 problems at k=1 on AIME24, gap narrows with larger k (Figure 9).
- Do the experiments support the claims?
  - Yes. The pattern repeats across model sizes, families, tasks, and algorithms, with multiple diagnostics (coverage sets, PPL, accuracy histograms) pointing to “reweighting within the base’s prior” rather than new reasoning.
  - The paper also controls for confounds like guesswork (manual CoT checks; Section 3.1, Appendix D.2) and entropy (Figure 18).

> Coverage set overlap: “only RL‑solvable” problems are 0.0% on AIME24 and 1.0% on MATH500 (Table 2).

> Training dynamics: pass@1 improves from 26.1 → 42.5 while pass@256 drops (Figure 1 right; Table 4).

> Algorithmic parity: base pass@256=67.2 vs. RL pass@1≈26–31 on Omni‑MATH‑Train; `ΔSE` ≈ 0.36–0.44 across algorithms (Figure 8 top; Table 3).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - RL is single‑turn: the verifier gives a one‑shot binary signal; there is no multi‑turn interaction to revise a solution (Section 6; Appendix C). Findings thus apply to current single‑turn RLVR pipelines.
  - Upper bound proxy: The base model’s pass@256 is used as an “upper bound” for coverage; while practical, it is still a finite‑k approximation (Section 4.3).
- Scenarios not addressed
  - Extremely large‑scale RL (e.g., month‑long runs with far larger rollouts/budgets) remains untested; the paper presents early evidence on a near‑frontier RL system (Magistral, Figure 9), but not at the absolute frontier (Section 4.6).
  - Tasks without verifiable rewards (open‑ended reasoning, long‑horizon multi‑step environments) are outside scope.
- Computational and data constraints
  - High‑k evaluations (k up to 1024) are compute‑intensive, limiting coverage on some very large models (Section 4.6).
- Open questions
  - Can fundamentally different exploration methods push the policy outside the base prior?
  - What role do richer reward structures (beyond binary) or multi‑turn feedback loops play in expanding coverage?

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes success for RL‑trained reasoning LLMs: high pass@1 is not sufficient evidence of new reasoning capacity. Large‑k pass@k and coverage analyses are necessary to judge whether the model can solve more distinct problems than before.
  - Distillation’s role is elevated: teacher‑student transfer demonstrably expands the reasoning boundary (Figure 7), while current RLVR largely does not.
- Research directions
  - Exploration beyond the prior
    - Design RL algorithms that explicitly encourage out‑of‑prior exploration without collapsing into nonsense (Appendix C, “priors as a double‑edged sword”).
    - Structured search or trajectory‑level objectives that look beyond the current policy’s neighborhood.
  - Multi‑turn agent–environment interaction
    - Transform verifiable tasks into iterative environments where models can try, get feedback, refine, and accumulate new experience (Section 6; Appendix C).
  - Reward design
    - Move beyond a single binary check; use dense, step‑level, or verifier‑assisted shaping rewards that better guide exploration in the combinatorial language space.
  - Scaling studies that disentangle components
    - Systematic ablations on rollout counts, KL control, curriculum over problem difficulty, and long‑context reasoning to see where coverage expands vs. contracts.
- Practical applications
  - Evaluation: Adopt large‑k pass@k (with the low‑variance estimator in Equation 2) as a routine metric for reasoning coverage in math, code, and multimodal benchmarks.
  - Training: Combine RLVR (for sampling efficiency) with distillation (for new patterns), rather than relying on RL alone for capacity growth.

> Bottom line: The empirical evidence (Figures 1–9; Tables 1–6; Equations 1–2) indicates that today’s RLVR mainly sharpens probabilities over reasoning paths already present in the base model. To truly expand reasoning capacity, future systems likely need better exploration, richer feedback, and multi‑turn interaction, or leverage distillation to import new reasoning patterns.
