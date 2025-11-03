# Agentic Reinforced Policy Optimization

**ArXiv:** [2507.19849](https://arxiv.org/abs/2507.19849)
**Authors:** Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui Zhou, Yutao Zhu, Ji‑Rong Wen, Zhicheng Dou
**Institutions:** Renmin University of China (presumed)

## 🎯 Pitch

Agentic Reinforced Policy Optimization (ARPO) introduces an innovative reinforcement-learning strategy for multi-turn language model agents, focusing on uncertainty-aware tool interactions to improve task accuracy efficiently. By branching rollouts at high-uncertainty moments post-tool feedback, ARPO reduces wasted calls and boosts exploration where needed most, significantly enhancing performance across diverse benchmarks. This advancement addresses the needs of scalable, cost-effective agent training, promising practical improvements in enterprise search, coding assistants, and scientific analysis.

---

## 1. Executive Summary
This paper introduces Agentic Reinforced Policy Optimization (ARPO), a reinforcement-learning algorithm for training large language model (LLM) agents that interact with external tools over multiple turns. ARPO branches rollouts precisely at high-uncertainty moments after tool calls and assigns credit so the model can learn which stepwise tool-use choices improved outcomes, yielding higher task accuracy with substantially fewer tool calls across 13 benchmarks (Figures 1, 5; Tables 1–2).

## 2. Context and Motivation
- The problem
  - LLMs now solve many single-turn reasoning tasks with reinforcement learning from verifiable rewards (RLVR), but real-world “agent” scenarios require multi-turn interaction with tools such as search engines and code interpreters (§1, §2.1, §2.3).
  - Common RL approaches for agents optimize at the whole-trajectory level (e.g., GRPO, DAPO, REINFORCE++; §1, §2.1, §4.2). They sample full tool-use traces and apply a final reward, which overlooks the fine-grained decisions made at individual tool-use steps.

- Why it matters
  - In multi-turn tool use, each tool’s feedback can drastically shift what the model should do next. Training that ignores these per-step dynamics can waste tool calls, under-explore promising actions, and miss the behaviors needed for deep search and knowledge-intensive tasks (§1, §2.2).

- Gap identified by the paper
  - A pilot entropy study shows uncertainty spikes immediately after the model receives tool feedback: the entropy of the next 10–50 generated tokens rises sharply (Figure 2). The effect is stronger for web-search results than for Python outputs, indicating that external textual feedback is especially disruptive to the model’s internal distribution (§2.2; Equation 3 defines token entropy).
  - Trajectory-level RL under-explores these high-uncertainty regions, so it fails to fully align stepwise tool-use behavior (§1, §2.2).

- Positioning
  - ARPO is an agentic RL algorithm designed to:
    - Branch additional rollouts only at high-entropy tool-use steps (“entropy-based adaptive rollout,” §3.1).
    - Attribute advantage at the token level so shared prefixes and branched continuations receive appropriate credit (“advantage attribution estimation,” §3.2).
    - Provide a theoretical foundation that macro-segmented updates are valid for Transformer policies (Generalized Policy Gradient Theorem, Equation 9; §3.3, Appendix D.2).

## 3. Technical Approach
ARPO modifies the standard RLVR setup to match agentic (tool-using) behavior.

- Problem setup (Equation 1, §2.1)
  - Policy LLM `πθ` interacts with tools `T` while generating a reasoning trajectory `R` and final answer `y`.
  - Objective: maximize expected verifiable reward `rϕ(x, y)` minus a KL regularizer to a reference model `πref`.

- How agentic rollouts are factorized (Equation 2, §2.1)
  - Reasoning with tools: `Pθ(R | x; T)` is the sequence of decisions (including tool calls and textual steps).
  - Final answering: `Pθ(y | R, x; T)` is generated after tool-augmented reasoning.

- Key mechanism 1: Entropy-based adaptive rollout (§3.1; Figures 3–4)
  - What is token entropy? A measure of uncertainty in the next-token distribution: `Ht = −∑j pt,j log pt,j` where `pt` is the softmax over logits at step `t` (Equation 3, §2.2).
  - Step-by-step procedure:
    1) Rollout initialization: for a global rollout budget `M`, sample `N` full trajectories (“global samples”), reserving `M−N` for possible branches. Compute an “initial” entropy profile `Hinitial` by generating the first `k` tokens for each trajectory (Figure 3; §3.1(1)).
    2) Entropy monitoring after each tool call: when a tool returns output, append it to the context and generate the next `k` tokens to get a step-level entropy vector `Ht`. Compute normalized entropy change `ΔHt = Normalize(Ht − Hinitial)` (Equation 4; §3.1(2)). Positive `ΔHt` indicates increased uncertainty compared to the initial baseline.
    3) Adaptive beaming (branching): compute a branching probability `Pt = α + β·ΔHt`. If `Pt > τ`, branch `Z` new partial rollouts from that point; otherwise continue the current path (Equation 5; Figure 4a; §3.1(3)). This concentrates exploration where uncertainty spikes after tool feedback.
    4) Termination logic: stop branching when the partial budget reaches `M−N`, or fill remaining budget with new global samples if all branches terminate early (§3.1(4)).
  - Why this design? The pilot analysis (Figure 2) shows tool feedback reliably induces high uncertainty in the immediate next tokens. Branching at precisely those moments expands exploration where the policy is least certain—and most likely to contain valuable alternatives.

- Key mechanism 2: Advantage attribution estimation (§3.2; Figure 4b)
  - The challenge: partial branching creates sets of trajectories that share prefix tokens but differ after a branching point. Credit assignment should:
    - Give shared tokens a group-level signal (they led to a common branching state).
    - Give diverging tokens individual signals (they determine which branch succeeds).
  - Two variants:
    - Hard advantage: explicitly compute advantages for each branch and assign the average advantage to shared tokens and individual advantages to diverging tokens (definitions below Equation 7 and in §3.2).
    - Soft advantage (the default): use the GRPO objective (Equation 6) where the importance ratio `ri,t(θ)` is identical for shared prefixes across branches and differs for tokens after branching (Equation 7). This implicitly aligns shared-token updates to the group average without separate bookkeeping (the paper provides proof sketches in §3.2 and Appendix D.1).
  - Empirical choice: Soft estimation is more stable and achieves higher rewards during training (Figure 5, §3.2).

- Reward design (§3.2, “Hierarchical Reward Design”)
  - Verifiable rewards include:
    - Correctness (e.g., exact or token-level F1).
    - Format adherence.
    - A small bonus `rM = 0.1` if the agent properly uses multiple tools (both `<search>` and `<python>`).
  - Overall reward `R` is a piecewise function (Equation 8). Misformatted outputs receive `−1`, good format with correct content receives `max(Acc. + rM, Acc.)`.

- Theoretical foundation (§3.3; Equation 9; Appendix D.2)
  - Generalized Policy Gradient (GPG) theorem: you can segment a Transformer’s output into “macro actions” (e.g., segments between tool calls or other markers) and the policy gradient decomposes as a sum over macro steps:
    - `∇θ J(θ) = Eτ [ ΣT ∇θ log πθ(MAT | MST) · AT(τ) ]` (Equation 9).
  - Intuition: as long as the segmentation preserves the chain rule (Appendix D.2), updating on macro segments is valid. This legitimizes ARPO’s partial rollouts and group-based updates.

- Complexity note (§3.1)
  - For global expansion size and tokens per trajectory `n`, ARPO reduces rollout complexity from `O(n^2)` to between `O(n log n)` and `O(n^2)` by focusing expansion at selective steps (footnote in §3.1).

- Tool environment (§2.3)
  - Three tools are used in experiments: a search engine, a browser agent (to visit and summarize pages), and a code interpreter.

## 4. Key Insights and Innovations
- Entropy-triggered step-level exploration is effective and economical
  - Novelty: Instead of expanding entire trajectories, ARPO branches only when the token entropy spikes after tool feedback (Equations 3–5; Figures 2, 4a).
  - Significance: It increases behavioral diversity exactly where the agent is least certain and reduces wasted tool calls (Figure 7 shows better accuracy than GRPO while using about half as many tool calls).

- Advantage attribution that respects shared prefixes and branched actions
  - Novelty: ARPO formalizes credit assignment for partial rollouts where many samples share a prefix (Figure 4b). The “soft” variant shows that GRPO’s importance weighting naturally averages shared-token updates (Equations 6–7; Appendix D.1).
  - Significance: This lets the agent internalize which branch-level tool actions were advantageous without manual token tagging or special losses. It improves reward stability (Figure 5).

- Macro-action policy gradient for Transformer agents
  - Novelty: The GPG theorem (§3.3; Equation 9; Appendix D.2) generalizes policy gradient to macro segments of Transformer outputs.
  - Significance: It provides a principled foundation for partial rollouts and step-level updates in LLM agents. This is a conceptual advance beyond ad hoc trajectory splitting.

- Demonstrated tool-call efficiency at scale
  - Novelty: Efficiency is evaluated explicitly during RL training—rare in agentic RL papers.
  - Significance: Figure 7 shows ARPO reaches higher accuracy than GRPO with roughly half the number of tool calls on Qwen2.5‑7B, directly addressing training-time cost.

## 5. Experimental Analysis
- Evaluation methodology (§4)
  - Tasks (13 datasets; §4.1):
    - Mathematical reasoning: AIME’24/’25, MATH500, MATH, GSM8K.
    - Knowledge-intensive QA: HotpotQA, 2WikiMultihopQA, Musique, WebWalker, Bamboogle.
    - Deep search: GAIA, WebWalkerQA, Humanity’s Last Exam (HLE), xBench-DeepSearch.
  - Baselines (§4.2):
    - Direct reasoning models (Qwen2.5, Llama3.1; larger closed/open references like GPT‑4o, DeepSeek‑R1, QwQ).
    - Trajectory-level RL (GRPO, REINFORCE++, DAPO).
    - Workflow agents for deep search (Vanilla RAG, Search‑o1, WebThinker, ReAct).
  - Training protocol (§4.3; §C):
    - Cold-start SFT using open data (Tool‑Star 54k + STILL 0.8k). RL uses VERL; tool outputs are excluded from loss; only reasoning and tool request tokens are optimized (§C.2).
    - For deep search, only 1k mixed hard samples are used for RL, browser fetch up to 6k tokens per page (§4.3.2; §C.3).
  - Metrics (§4.4):
    - F1 for four QA tasks; LLM-as-judge for others; pass@1 with temperature 0.6, top‑p 0.95. Answers are extracted from `\box{}`.

- Main quantitative results
  - Mathematical + knowledge reasoning (Table 1)
    - ARPO outperforms trajectory-level RL across three backbones. Examples:
      > Qwen2.5‑3B: Avg accuracy rises from 50.4 (GRPO) / 50.6 (DAPO) to 52.8 with ARPO (Table 1).

      > Llama3.1‑8B: Avg accuracy improves from 51.1 (GRPO/REINFORCE++) and 50.4 (DAPO) to 55.3 with ARPO (Table 1).

      > Qwen2.5‑7B: Avg accuracy improves from 56.5 (GRPO) to 58.3 with ARPO (Table 1).
    - Prompting-only TIR often hurts or underperforms direct reasoning (e.g., Qwen2.5‑3B: 26.1 → 25.1; Qwen2.5‑7B: 32.0 → 31.0; Table 1), reinforcing that algorithmic training—not prompting—is needed for tool use.

  - Deep search (Table 2; Figure 6)
    - With 1k RL samples, ARPO beats GRPO at both 8B and 14B scales:
      > Qwen3‑8B: GAIA avg 38.8 (ARPO) vs 32.0 (GRPO); WebWalkerQA avg 30.5 vs 29.0; HLE avg 8.8 vs 7.8; xBench avg 25.0 vs 20.0 (Table 2).

      > Qwen3‑14B: GAIA avg 43.7 (ARPO) vs 36.9; WebWalkerQA avg 36.0 vs 30.0; HLE avg 10.0 vs 8.6; xBench avg 32.0 vs 27.0 (Table 2).
    - Pass@K scaling after ARPO further increases success, e.g. Qwen3‑14B reaches pass@5 of 61.2% on GAIA, 24.0% on HLE, and 59% on xBench‑DR (Figure 6).

  - Tool-call efficiency (Figure 7)
    - > “ARPO achieves superior overall accuracy compared to GRPO while using only half the number of tool calls” on Qwen2.5‑7B during RL training (Figure 7). This validates the selective branching design.

  - Ablations and robustness
    - Browser agent importance: Using only search snippets is worst; a stronger browser improves results (Table 3). For Qwen3‑14B, average deep-search accuracy rises from 24.8 (snippets only) → 29.9 (14B browser) → 39.4 (QWQ‑32B browser).
    - Hyperparameter scaling (Figure 8):
      - Entropy weight: performance peaks near 0.4; very high weight (1.0) hurts, indicating over-reliance on entropy reduces diversity (§4.7).
      - Initial sampling size `N`: with rollout size `M=16`, performance peaks at `N=8`, balancing global and partial samples.
      - Larger global rollout size `M` helps overall.

- Do the results support the claims?
  - Yes, across 13 datasets ARPO consistently matches or exceeds trajectory-level RL. Gains are largest in deep search, where tool feedback is frequent and uncertainty spikes are common (Tables 1–2; Figures 2, 6–7).
  - The causal link—exploration targeted at high-entropy post-tool steps—has empirical support from the entropy pilot (Figure 2) and the efficiency analysis (Figure 7). Ablations confirm dependence on a capable browser agent in deep search (Table 3).
  - Caveat: many tasks rely on LLM-as-judge scoring (§4.4), which can introduce evaluation noise. The paper mitigates this by also using verifiable rewards during training and token-level F1 for some QA tasks.

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - The entropy signal must be measured at specific post-tool windows (first `k` tokens); its normalization divides by vocabulary size `V` (Equation 4). The choice of `k`, `α`, `β`, and threshold `τ` is heuristic and requires tuning (Figure 8).
  - The method presumes access to reliable tool boundary markers and structured tool I/O (e.g., `<search>`, `<python>`) to trigger entropy checks (§2.3, §3.1).

- Scope
  - Tools evaluated are search, browser, and Python (§2.3). ARPO’s behavior for other tools (e.g., database transactions, APIs with strong latency or stochasticity) is not evaluated.
  - Most non-QA tasks are judged by LLMs (§4.4), which may overestimate performance or be sensitive to style.

- Computational and data considerations
  - Although ARPO reduces unnecessary branching, it still expands sampling substantially in high-entropy regions. The practical cost depends on rollout budgets `M, N, Z` and the frequency of tool calls (Figure 7).
  - RL training used relatively small deep-search datasets (1k samples; §4.3), which is promising for efficiency but leaves open whether gains persist at very large scales or with noisier logs.

- Open questions
  - How robust is entropy as a universal cue for branching across domains (e.g., robotics, dialogue planning)?
  - Could learned predictors of “branch-worthiness” outperform raw entropy signals?
  - How sensitive are results to the choice of reward bonuses for multi-tool use (Equation 8)?

## 7. Implications and Future Directions
- Field impact
  - ARPO shifts agentic RL from trajectory-level sampling to step-level, uncertainty-aware exploration. This reframes how the community should collect RL experience for tool-using agents: focus exploration where tool feedback destabilizes the model’s beliefs (Figures 2, 4a).

- Follow-up research enabled
  - Learned branching policies: replace the linear `Pt = α + β·ΔHt` rule with a small policy or critic to decide when and how much to branch.
  - Beyond entropy: integrate other uncertainty estimates (e.g., disagreement among ensembles, verifier signals) or learn to predict the value of additional tool calls.
  - Broader tools and environments: evaluate on APIs with constraints (rate limits, costs), multi-agent coordination, or real-time web changes.
  - Credit assignment variants: explore token-level value functions or hierarchical critics aligned with the GPG theorem to improve advantage estimation for long contexts.
  - Offline and hybrid RL: ARPO’s partial rollouts could be combined with logged trajectories to scale training without excessive online tool calls.

- Practical applications
  - Enterprise research assistants that must browse and synthesize information efficiently.
  - Coding copilots that interleave reasoning with program execution and debugging.
  - Scientific and policy analysis agents that perform deep, multi-document evidence gathering while controlling tool costs.

In sum, ARPO contributes an uncertainty-aware sampling strategy and a principled credit assignment scheme for multi-turn LLM agents. It demonstrates consistent accuracy gains and markedly better tool-call efficiency across diverse benchmarks, suggesting a practical path toward scalable, cost-effective agent training (Tables 1–2; Figure 7).
