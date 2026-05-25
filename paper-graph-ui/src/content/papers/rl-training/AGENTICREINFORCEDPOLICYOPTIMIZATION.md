# AGENTIC REINFORCED POLICY OPTIMIZATION

**ArXiv:** [2507.19849](https://arxiv.org/abs/2507.19849)

## 🎯 Pitch

Agentic Reinforced Policy Optimization (ARPO) introduces a novel RL algorithm that dynamically branches rollouts at moments of high uncertainty immediately following tool calls, allowing large language model agents to better explore and internalize step-level tool-use behaviors. By adaptively targeting these critical decision points and attributing advantage credit with fine granularity, ARPO achieves state-of-the-art performance across 13 multi-turn reasoning benchmarks—while using only half the tool-call budget of prior methods. This establishes a new paradigm for efficient, scalable alignment of LLM-based agents with real-world, dynamic, tool-rich environments.

---

## 1. Executive Summary
Agentic Reinforced Policy Optimization (ARPO) is a reinforcement-learning method for training multi‑turn, tool‑using language‑model agents. It adapts the rollout process to branch precisely at high‑uncertainty moments after tool calls and updates the policy so that shared versus branched tokens receive appropriate advantage credit. Across 13 benchmarks in math, knowledge reasoning, and deep search, ARPO outperforms trajectory‑level RL baselines while using roughly half the number of tool calls during training (Figure 1 right; Figure 7).

## 2. Context and Motivation
- Problem addressed
  - Modern reinforcement learning with verifiable rewards (RLVR) has unlocked strong single‑turn reasoning in LLMs. However, real tasks often require multi‑turn interactions with external tools (search engines, browsers, code execution), where the agent must decide when and how to call tools over long horizons (§1–§2.1).
  - Existing “trajectory‑level” RL methods (e.g., GRPO, DAPO, REINFORCE++; §1, §2.1, §4.2) sample complete trajectories and compare whole outputs with final rewards. They under‑explore fine‑grained, step‑level behaviors inside tool‑use loops, especially right after tool feedback arrives.

- Why it matters
  - In multi‑turn settings, tools provide frequent, informative feedback. Effective agents must exploit these signals at the right steps rather than only optimizing end‑to‑end sequences. This has practical importance for search, research assistance, and code‑assisted reasoning (§1, §2.3).

- Observed gap
  - A pilot analysis shows token‑level predictive uncertainty (“token entropy”) spikes immediately after each tool call (Figure 1 left; Figure 2; §2.2). This indicates the model is most unsure right when external information is injected—precisely when exploration is most valuable. Trajectory‑level RL largely ignores this.

- Positioning
  - ARPO introduces an agent‑specific RL algorithm that (a) detects high‑entropy post‑tool steps and branches sampling at those points, and (b) attributes advantages differently to shared versus branched tokens so the policy internalizes which step‑level behaviors were beneficial (§3).

## 3. Technical Approach
ARPO consists of two core components plus a theory justification (Figures 3–4; §3):

1) Entropy‑based Adaptive Rollout (§3.1)
- Key idea in plain terms: Start several “global” samples for a question. Each time the agent receives tool output, briefly generate a few tokens to measure uncertainty (“token entropy”). If uncertainty increased relative to the start, selectively branch more rollouts from that step to explore alternatives right where the model is unsure.
- Definitions
  - `token entropy` H_t measures uncertainty of the token distribution at step t:
    - H_t = −∑_j p_t,j log p_t,j, where p_t = Softmax(z_t / τ) (Eq. 3). Higher H_t means the model’s next‑token distribution is more spread out (more uncertain).
- Step‑by‑step mechanics (Algorithm 1; Figure 4a)
  - Rollout initialization: With a rollout budget `M`, perform `N` global trajectory samples for a question q, and record the initial entropy of the first `k` tokens to form H_initial (§3.1 step 1).
  - Entropy variation monitoring: After each tool call at step t, append the tool’s response to the context and generate `k` tokens to compute a stepwise entropy vector H_t. Compute normalized change ΔH_t = Normalize(H_t − H_initial) (Eq. 4; §3.1 step 2).
  - Adaptive “beaming” (branching): Compute a branching probability
    - P_t = α + β·ΔH_t. If P_t > τ, create Z branched partial rollouts from that node; otherwise continue the current trajectory (Eq. 5; §3.1 step 3; Figure 4a).
  - Termination and budget control: Stop branching when the partial‑sampling budget (M−N) is consumed or when answers end. If branching used less than the budget, top up with extra global samples so total samples equal M (§3.1 step 4).
- Why this design
  - Entropy spikes immediately after tool outputs (Figure 1 left; Figure 2), indicating rich but uncertain decision points. Branching there increases behavioral diversity exactly where it matters.
  - Complexity: With global expansion size and tokens per trajectory n, the partial rollout reduces per‑rollout complexity from O(n^2) (plain trajectory beaming) to between O(n log n) and O(n^2) (§3.1, footnote 2).

2) Advantage Attribution Estimation (§3.2)
- Intuition: When samples share a prefix and then branch, shared tokens should receive a shared “advantage” signal, and branched tokens should receive distinct signals based on their own outcomes (Figure 4b).
- Two realizations
  - Hard setting: For d trajectories that share a prefix, assign each branch i a normalized advantage Â_i,t from its reward R_i; the shared prefix tokens get the average advantage over the d branches Â_shared = (1/d)∑_i Â_i,t (§3.2).
  - Soft setting (default): Use GRPO (Group Relative Policy Optimization; Eq. 6) with importance ratios r_i,t(θ) = π_θ(y_i,t | x, y_i,<t) / π_ref(y_i,t | x, y_i,<t). If two trajectories share a prefix at token t, their r_i,t are equal; thus shared tokens effectively receive the same credit, while post‑branch tokens get different weights (Eq. 7; Appendix D.1 formalizes the equivalence).
- Empirical choice: The soft setting yields higher and more stable rewards during training (Figure 5), so ARPO defaults to it.

3) Hierarchical Reward and Implementation (§3.2, Eq. 8; §C)
- Rewards combine correctness, output format, and a small bonus r_M=0.1 for using multiple tools (both `<search>` and `<python>`) when the answer is correct and formatted (Eq. 8).
- Training excludes tool outputs themselves from loss to avoid bias (§C.2). Tools include: search engine, browser agent, and code interpreter (§2.3; §C.3).
- Evaluation answers are extracted from \box{} markers (§4.4).

4) Theoretical foundation (§3.3; Appendix D.2)
- Generalized Policy Gradient (GPG) Theorem (Eq. 9): Treat any contiguous output segment as a “macro action” (e.g., from one tool call to the next). The policy gradient can be written as a sum over macro steps T:
  - ∇_θ J(θ) = E_τ [ ∑_T ∇_θ log π_θ(MA_T | MS_T) · A_T(τ) ]
- This justifies optimizing with partial rollouts (macro segments) rather than only token‑level actions, aligning with ARPO’s branch‑at‑step design (§3.3; Appendix D.2).

Analogy
- Think of solving a puzzle while occasionally asking an expert. Right after the expert answers, uncertainty jumps: you may go in several directions. ARPO spends extra “tries” at just those moments, then learns which shared reasoning led to good branches and which branch choices paid off.

## 4. Key Insights and Innovations
- Entropy‑triggered, step‑level exploration (fundamental)
  - Novelty: Uses token‑entropy spikes after tool calls as a branching trigger (Figures 1–2; §3.1). Prior RL methods sample full trajectories uniformly or with generic beaming, missing these high‑value moments.
  - Significance: Focuses exploration exactly where external information creates the biggest uncertainty, improving sample efficiency and behavior diversity.

- Advantage attribution that respects shared vs. branched tokens (principled, practical)
  - Novelty: Either explicit (hard) or implicit (soft GRPO) credit assignment so shared prefixes share credit while branches receive distinct credit (Eq. 6–7; Figure 4b; Appendix D.1).
  - Significance: Helps the model internalize which step‑level tool‑use decisions improved outcomes, stabilizing training (Figure 5).

- Macro‑action policy gradient for Transformer agents (theoretical)
  - Novelty: A generalized policy gradient (Eq. 9) that legitimizes partial rollouts over macro segments (Appendix D.2, Eq. 20–21).
  - Significance: Provides a formal foundation for step‑aware agent training instead of only trajectory/token‑level updates.

- Tool‑call–efficient training (practical impact)
  - Observation: ARPO reaches higher accuracy with roughly half as many tool calls as GRPO during training (Figure 7), reducing cost for web‑tool agents.

## 5. Experimental Analysis
- Evaluation setup (§4)
  - Tasks (13 datasets; §4.1)
    - Mathematical: AIME24, AIME25, MATH500, MATH, GSM8K.
    - Knowledge‑intensive QA: WebWalker, HotpotQA, 2WikiMultihopQA, Musique, Bamboogle.
    - Deep Search: GAIA, WebWalkerQA, Humanity’s Last Exam (HLE), xbench‑DeepSearch.
  - Models and baselines (§4.2)
    - Direct reasoning: Qwen2.5, Llama3.1, Qwen3, plus large closed/open models (e.g., DeepSeek‑R1‑671B, GPT‑4o) for reference in Table 2.
    - Trajectory‑level RL: GRPO, DAPO, REINFORCE++.
    - Search agents (for Deep Search): Vanilla RAG, Search‑o1, WebThinker, ReAct.
  - Metrics (§4.4)
    - F1 for four Wikipedia QA tasks; for others, pass@1 with temperature 0.6 and top‑p 0.95; LLM‑as‑Judge uses Qwen2.5‑72B-instruct. Answers are parsed from \box{}.
  - Training protocol (§4.3; §C)
    - Cold‑start SFT on 54k Tool‑Star data + 0.8k STILL for math; RL with 10k samples for deep reasoning and only 1k mixed hard search samples for deep search. Tooling via Bing search and a sandboxed Python interpreter; browser agent is used for deep search.
    - ARPO rollout hyperparameters include entropy weight, base probability α, threshold τ; examples: total rollout size M=16, initial N=8 (§C.2).

- Main quantitative results
  - Mathematical + Knowledge‑intensive (Table 1)
    - Qwen2.5‑3B‑Instruct: ARPO avg 52.8 vs GRPO 50.4, DAPO 50.6, REINFORCE++ 49.7.
    - Llama3.1‑8B‑Instruct: ARPO 55.3 vs GRPO 51.1 and DAPO 50.4.
    - Qwen2.5‑7B‑Instruct: ARPO 58.3 vs GRPO 56.5, DAPO 54.8, REINFORCE++ 54.9.
    - Prompting baseline (TIR) often underperforms or only slightly helps; e.g., on Llama3.1‑8B avg 36.3 vs direct 28.8, but far behind ARPO 55.3.
    - Quote:
      > Table 1 shows ARPO as the top method on all three backbones’ averages, with consistent gains across 10 datasets.
  - Deep Search (Table 2; Figure 6)
    - Qwen3‑8B: ARPO GAIA avg 38.8 vs GRPO 32.0; WebWalkerQA avg 30.5 vs 29.0; HLE avg 8.8 vs 7.8; xbench 25.0 vs 20.0.
    - Qwen3‑14B: ARPO GAIA avg 43.7 vs GRPO 36.9; WebWalkerQA 36.0 vs 30.0; HLE 10.0 vs 8.6; xbench 32.0 vs 27.0.
    - Against non‑RL search agents, ARPO is generally stronger; e.g., on Qwen3‑14B, ARPO’s GAIA 43.7 beats WebThinker 33.0 and Search‑o1 30.1.
    - Reference models struggle on HLE (e.g., DeepSeek‑R1‑671B 8.6; GPT‑4o 2.6), while small ARPO models reach 10.0 (14B) and 8.8 (8B).
    - Scaling in sampling (Figure 6): ARPO improves Pass@3 and Pass@5. Notably:
      > Qwen3‑14B+ARPO achieves GAIA Pass@5 of 61.2%, HLE 24.0%, and xbench‑DR 59%.
  - Training tool‑call efficiency (Figure 7)
    - Quote:
      > ARPO attains higher accuracy than GRPO while using about half the number of tool calls on Qwen2.5‑7B during RL training.
  - Browser ablation (Table 3)
    - No browser (snippets only) performs worst (e.g., Qwen3‑14B avg 24.8). A stronger browser agent (QWQ‑32B) improves averages further (up to 39.4 on 14B).
  - Hyperparameter scaling (Figure 8)
    - Best entropy weight around 0.4; increasing initial sampling size N up to 8 (with M=16) helps but N=16 hurts (it removes partial sampling); larger M improves performance.

- Do the experiments support the claims?
  - Yes, across backbones and domains ARPO beats trajectory‑level RL baselines, especially on deep search where step‑level tool use is critical (Table 2). The efficiency claim is substantiated by Figure 7. The entropy‑driven exploration hypothesis is supported empirically by entropy spikes after tool calls (Figure 1 left; Figure 2) and by improved Pass@K diversity (Figure 6).

- Notable design checks
  - Robustness to tool quality: Browser ablations reveal performance correlates with browser strength (Table 3).
  - Reward shaping: A small multi‑tool bonus (Eq. 8) nudges but does not dominate outcomes.
  - Choice of advantage estimator: Soft > Hard (Figure 5).

## 6. Limitations and Trade-offs
- Dependence on entropy signals
  - Assumes token‑entropy increases are reliable markers of valuable exploration moments (§2.2). If tool outputs are noisy or misleading, entropy may spike for unhelpful reasons.
  - Requires tuning of α, β, τ, Z, k, M, N (Figure 8). Wrong settings (e.g., all‑global N=M) cancel ARPO’s benefits.

- External tool availability and quality
  - Performance, especially in deep search, depends on search/browsing coverage and a capable browser agent (Table 3). Limited APIs or retrieval quality can bottleneck gains.

- Computational considerations
  - Although branching is targeted, partial rollouts still add decoding and entropy computation overhead. Complexity is between O(n log n) and O(n^2) per rollout (§3.1), and latency may grow with many high‑entropy steps.

- Reward and judging assumptions
  - Some benchmarks rely on LLM‑as‑Judge (Qwen2.5‑72B) for accuracy (§4.4). This introduces potential evaluation bias and variance.
  - The multi‑tool bonus (Eq. 8) may favor using both tools, which could be suboptimal in domains where one tool suffices.

- Scope
  - The work optimizes rule‑based RL with GRPO‑style losses; it does not explore value‑based or model‑based RL variants for tool agents.
  - Focuses on textual tools (search, browser, Python). Extensions to richer environments (APIs with stateful effects, robotics) are not studied.

## 7. Implications and Future Directions
- Field impact
  - Shifts agent training from “whole‑trajectory” thinking to “step‑aware” exploration guided by uncertainty. This reframes RL for LLM agents as a sequence of macro decisions aligned to tool‑feedback points (Eq. 9), a useful perspective for many tool‑rich tasks.

- Practical applications
  - Cost‑effective training of research assistants, web agents, and code‑augmented solvers: fewer tool calls during RL (Figure 7) reduce monetary/API costs while improving outcomes (Tables 1–2).

- Research opportunities
  - Adaptive thresholds and learned branching policies: replace hand‑tuned α, β, τ with learned controllers using entropy and other uncertainty proxies.
  - Broader uncertainty signals: combine token entropy with verifier disagreement, calibration, or retrieval confidence (beyond Eq. 3).
  - Value‑based or hierarchical RL: couple ARPO’s partial rollouts with learned value estimates for macro actions; integrate planning over tool‑use graphs.
  - Multi‑agent/tool scheduling: extend advantage attribution to coordinated decisions across multiple tools or agents, possibly with credit assignment across longer horizons.
  - Robust evaluation: complement LLM‑as‑Judge with programmatic verifiers and human audits; study domain shift (languages, domains, low‑resource web).

> Bottom line: ARPO demonstrates that targeting exploration to post‑tool high‑uncertainty steps and crediting shared vs. branched tokens differently yields better, cheaper training for tool‑using LLM agents, with both empirical wins (Tables 1–2, Figures 6–7) and a principled macro‑action gradient foundation (Eq. 9; Appendix D.2).
