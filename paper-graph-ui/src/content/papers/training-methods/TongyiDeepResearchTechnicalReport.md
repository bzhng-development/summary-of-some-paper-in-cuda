# Tongyi DeepResearch Technical Report

**ArXiv:** [2510.24701](https://arxiv.org/abs/2510.24701)

## 🎯 Pitch

Tongyi DeepResearch introduces an open-source agentic large language model, pioneering a unified training framework that integrates a novel 'agentic mid-training' stage with advanced post-training—powered by fully automated synthetic data across customizable environments. This approach equips the model with scalable reasoning, autonomous research workflows, and efficient tool use, enabling it to outperform both open and proprietary systems on complex, long-horizon information-seeking benchmarks. Tongyi DeepResearch democratizes agentic AI research, sets a new state-of-the-art for deep research tasks, and provides a reproducible foundation for building transparent, trustworthy autonomous research agents.

---

## 1. Executive Summary
Tongyi DeepResearch is an open-source “deep research” agentic large language model (LLM) that can plan, search the web, use tools, and synthesize long-form answers for complex, multi-step information-seeking tasks. Its core contribution is an end-to-end training framework that unifies a new agentic mid-training stage with agentic post-training (SFT + RL), powered by a fully automatic synthetic-data engine and stage-specific environments (Section 2; Figure 2). The result is a 30.5B-parameter model with only 3.3B active parameters per token that achieves state-of-the-art results on multiple deep research benchmarks (Figure 1; Table 1).

## 2. Context and Motivation
- Problem/gap addressed
  - Deep research tasks require long-horizon planning, iterative web browsing, tool use, and reliable synthesis—often taking tens of minutes per query (Section 1). Most existing systems are closed, and their research traces are opaque, making them hard to study or reproduce (Section 1).
  - Standard LLM pipelines (pre-train on web text, then instruction-tune) lack “agentic inductive bias”—they do not learn the behaviors and workflows needed for autonomous research (Section 2, “Agent Training Pipeline”).
- Why this matters
  - Real-world: Organizations need trustworthy agents to perform literature reviews, market analyses, technical due diligence, and investigative reporting—tasks that require careful multi-source evidence gathering and reasoning.
  - Scientific: Understanding how to instill durable, scalable agent behaviors (planning, tool use, verification) is central to progress toward general-purpose agents (Section 1–2).
- Prior approaches and shortcomings
  - Closed systems (e.g., OpenAI/Gemini/Kimi DeepResearch) demonstrated capability but lack open models, process transparency, or training recipes (Section 1; Figure 1).
  - Open efforts often focus on post-training only (SFT/RL) without a mid-training stage to inject agentic priors, creating optimization conflicts and suboptimal learning (Section 2).
- Positioning
  - This work provides: (1) a unified training recipe with mid-training and post-training (Figure 2), (2) a fully automated synthetic data pipeline for all stages (Figures 3–4), (3) stable, stage-appropriate environments (Prior World, Simulated, Real-world; Section 2), and (4) open-sourced model, tools, and prompts.

## 3. Technical Approach
The system is a ReAct-style agent enhanced with context management, trained via agentic mid-training and post-training (SFT + RL), supported by a synthetic-data engine and tailored environments.

A. Agent architecture and rollout formulation (Section 3.1)
- Core loop components
  - `Thought (τ_t)`: the model’s internal reasoning at step t—analyze context, plan, reflect.
  - `Action (a_t)`: a tool call (e.g., Search, Visit, Python, Google Scholar, File Parser) or the final answer.
  - `Observation (o_t)`: tool outputs used to update state.
- ReAct rollout (Equation (1)–(2))
  - The agent alternates Thought → Action → Observation until a final Action that produces an in-depth report.
  - Policy `π` generates `(τ_t, a_t)` conditioned on the full (or managed) history.
- Context management (Section 3.1; Equation (3))
  - Challenge: long-horizon tasks can overflow the context window.
  - Solution: “Markovian state reconstruction” with a compressed, evolving report `S_t` that summarizes progress, plus the immediate last tool call and response `(a_t, o_t)`. At each step, the agent conditions on `(S_{t-1}, a_t, o_t)` to produce the next summary and action. This enforces periodic synthesis and keeps the context compact.

B. Training pipeline overview (Figure 2; Sections 3.2–3.4)
- Base: `Qwen3-30B-A3B-Base` (30.5B total, ~3.3B active/forward; Section 1, footnote 1).
- Mid-training: Agentic Continual Pre-training (Agentic CPT; Section 3.3)
  - Two stages:
    - Stage 1: 32K context; NTP (next-token prediction) on a mix of agentic behavior data and a small portion of general data.
    - Stage 2: 128K context with many long sequences (64K–128K) of agentic data to build long-horizon coherence.
  - Goal: inject agentic inductive biases before post-training to avoid learning conflicts and improve sample efficiency (Section 2).
- Post-training (Section 3.4)
  - Synthetic data engine for hard, verifiable QA (Figure 4).
  - Supervised Fine-Tuning (SFT) for cold start with two formulations:
    - ReAct Mode: input partial history `H_t`, output `(τ_i, a_i)`.
    - Context Management Mode: input `(S_{t-1}, a_{i-1}, o_{i-1})`, output `(S_t, τ_i, a_i)`.
    - Two SFT stages by length: up to 40K and then 128K (Section 3.4.2).
  - Agentic Reinforcement Learning (RL) with verifiable rewards (Section 3.4.3; Figures 5, 8, 9).
  - Model merging: weighted interpolation of several variants from the same base to combine strengths (Equation (6); Section 3.4.4).

C. Synthetic data across stages—how it is produced and why it matters
- Agentic CPT data (Figure 3; Section 3.3.2)
  - Multi-style question synthesis grounded in an “entity-anchored open-world memory” assembled from crawled web knowledge and agent trajectories.
  - Planning action data: decompose tasks and predict first steps; quality ensured via rejection sampling using known entities/knowledge.
  - Reasoning action data: two-stage generation of reasoning chains, filtered by length and answer consistency.
  - Decision-making action data: reconstruct multi-branch decision points in trajectories to expose the agent to choices and preferred alternatives.
  - General function-calling via environment scaling: automatically build simulated read–write environments as diverse “function-calling” contexts (Section 3.3.2).
- Post-training data (Figure 4; Section 3.4.1)
  - Three steps: (1) Construct a knowledge graph via random walks with web acquisition and isomorphic tables; (2) Sample subgraphs/tables; (3) Inject uncertainty to raise difficulty.
  - Difficulty is formally controlled via “atomic operations” on entity relations (e.g., merging similar entities) and a set-theoretic formalization of information-seeking to minimize shortcuts and redundancies.
  - Also scales PhD-level questions through iterative complexity upgrades with tooling (Section 3.4.1).

D. Environments—why three kinds and how they’re used (Section 2; 3.4.3)
- Prior World Environment: no live responses; perfect stability and scale; used to cheaply pre-train agent patterns.
- Simulated Environment: local, reproducible web replica (built from 2024 Wikipedia + local RAG) for fast ablation and causal analysis (Section 3.4.3, “Simulated Environment”).
- Real-world Environment: live web APIs; ultimate fidelity but non-stationary and costly.
- Unified sandbox for real tools (Search, Visit, Python, Google Scholar, File Parser) with concurrency control, caching, retries, QPS throttling, and failover (Section 3.4.3 “Real-world Environment”).

E. RL algorithm and data curation (Section 3.4.3; Equations (4)–(5); Figures 5, 8–10)
- Reward: 0/1 “RLVR”—reward if the final answer matches ground truth; no shaping or format reward (Section 3.4.3).
- Objective: on-policy GRPO-style token-level policy gradient with asymmetric clipping and a leave-one-out baseline to reduce variance (Equations (4)–(5); DAPO-inspired modifications; Section 3.4.3).
- Asynchronous rollout system: separate servers for model inference and tool invocation; centralized coordinator; many agents run in parallel (Figure 5).
- Automatic data curation:
  - Start with dataset `D`; sample multiple rollouts per problem with the SFT policy.
  - Build `D'` by filtering out “always fail” and “always succeed” items—retain “moderately difficult” problems.
  - During RL, continuously replace mastered problems with new moderately hard ones sourced by periodically sampling `D` using improving checkpoints.
  - Curation runs independently of training for uninterrupted optimization.

F. Inference-time scaling: Heavy Mode (Section 4.3; Figure 6; Equations (7)–(8))
- Problem: Aggregating whole trajectories from multiple agents quickly exceeds context length.
- Solution: Each of `n` parallel agents produces a compressed final report `S_T^u` (the context-management summary) and an answer. A separate “Synthesis” model aggregates `{(S_T^u, answer_u)}_{u=1}^n` to output the final answer (Equations (7)–(8)). This exploits diversity in exploration without blowing up context length.

## 4. Key Insights and Innovations
1) Unified mid-training + post-training for agents (Figure 2; Sections 2–3)
- What’s new: Introduces “agentic mid-training” (Agentic CPT) between pre-training and SFT/RL to instill agent-specific priors (planning, tool use, decision points) using NTP on large synthetic corpora.
- Why it matters: Avoids learning agent behavior and alignment from scratch during post-training, reducing optimization conflicts and improving stability and sample efficiency (Section 2).

2) Fully automated, verifiable synthetic-data engine spanning all stages (Figures 3–4; Section 3.3.2; 3.4.1)
- What’s new: End-to-end, human-free pipeline that creates diverse agent trajectories and super-human QA with controllable difficulty via knowledge-graph random walks, subgraph sampling, and formal uncertainty injection; also a set-theoretic formalization of information seeking.
- Why it matters: Deep research data is scarce and expensive; this pipeline enables large-scale, targeted, verifiable training data, including long sequences and complex tool-use patterns (Sections 2–3.4.1).

3) Environment-centric training strategy with robust tooling (Section 2; 3.4.3)
- What’s new: A principled triad of environments (Prior World, Simulated, Real-world), plus a unified sandbox that turns volatile web APIs into a deterministic, stable interface.
- Why it matters: Stable, scalable interactions are more critical than algorithmic tweaks for successful agentic RL; the sandbox also cuts costs and reduces data contamination (Section 3.4.3).

4) Context management and Heavy Mode for long-horizon reasoning (Section 3.1; 4.3)
- What’s new: Markovian state reconstruction using an evolving compressed report `S_t`, enabling deep trajectories within bounded context; Heavy Mode parallelizes exploration while synthesizing only compressed summaries.
- Why it matters: This makes test-time scaling feasible and effective on tasks that would otherwise exceed context limits (Figure 6).

5) Pragmatic, stable RL with on-policy GRPO-style updates plus dynamic curriculum (Section 3.4.3)
- What’s new: Token-level objective with asymmetric clipping, leave-one-out baseline, exclusion of degenerate negatives, and continuously refreshed “moderately difficult” problem set.
- Why it matters: Demonstrates stable reward growth and entropy control (Figure 8) and supports robust generalization (Figures 7, 9–10).

## 5. Experimental Analysis
- Evaluation setup (Section 4.1)
  - Benchmarks: Humanity’s Last Exam (HLE), BrowseComp, BrowseComp-ZH, GAIA, xBench-DeepSearch, WebWalkerQA, FRAMES, and xbench-DeepSearch-2510 (Figure 1; Table 1).
  - Protocols use official scripts and specific LLM judges where required (Appendix B).
  - Fixed inference params: temperature 0.85, repetition penalty 1.1, top-p 0.95; max 128 tool calls; 128K context; each benchmark run 3 times; primary metric is Avg@3 (Section 4.1).
- Main quantitative results (Table 1; Figure 1)
  - Avg@3 highlights:
    > Tongyi DeepResearch: HLE 32.9; BrowseComp 43.4; BrowseComp-ZH 46.7; WebWalkerQA 72.2; GAIA 70.9; xBench-DeepSearch 75.0; FRAMES 90.6.
  - Competitive landscape:
    - Beats all listed LLM-based ReAct agents on most benchmarks (e.g., vs DeepSeek-V3.1: HLE 32.9 vs 29.8; WebWalkerQA 72.2 vs 61.2; FRAMES 90.6 vs 83.7).
    - On BrowseComp, OpenAI DeepResearch scores higher (51.5), but Tongyi is strong and open-source (Table 1).
    - On the newly released xbench-DeepSearch-2510, Tongyi ranks just below ChatGPT-5-Pro (Figure 1, bottom-right; text: 55.0 for Tongyi).
- Heavy Mode gains (Figure 6)
  - Pass@1 improves substantially:
    > HLE: 38.3 (Heavy) vs 32.9 (base); BrowseComp: 58.3 vs 43.4; BrowseComp-ZH: 58.1 vs 46.7.
  - Interpretation: Parallel exploration + synthesis of compressed reports effectively leverages additional test-time compute without blowing past context limits.
- Robustness and variance checks (Figure 7)
  - Reports Avg@3, Pass@1 (best-of-3), and Pass@3 for several benchmarks.
  - Example:
    > BrowseComp: Avg@3 43.4; Pass@1 44.2; Pass@3 59.6.
  - Pattern: Pass@3 significantly exceeds Pass@1, indicating high potential when multiple attempts are allowed—consistent with agentic exploration.
- Training dynamics and ablations (Figures 8–10)
  - RL reward rises steadily; entropy stabilizes (Figure 8), supporting claims of stable on-policy RL with automated curation.
  - Context-length study (Figure 9): 64K model achieves highest reward; 32K shows decreasing response length over training—evidence it learns more concise strategies under constraints (right panel).
  - Interaction scaling (Figure 10a): accuracy on BrowseComp increases with larger context/interaction budget (8K → 128K), illustrating “interaction test-time scaling” for agents.
  - Simulation vs reality (Figures 8 vs 10b): reward curves are similar, validating the simulated environment as a fast iteration proxy.
- Data complexity evidence (Section 4.4 “Super-human Level Synthetic Data”)
  - > Over 20% of SFT samples exceed 32K tokens and involve more than 10 tool invocations.
  - This supports the claim that synthetic data provides a challenging, realistic training signal for deep research behaviors.
- General benchmarks (Figure 11)
  - Tongyi significantly improves over base “thinking-only” models on AIME25, HMMT25, and SimpleQA, reaching:
    > AIME25 100.0; HMMT25 100.0; SimpleQA 98.6
  - Likely driven by tool use (Python for math; Search for knowledge), underscoring the value of agent capabilities even outside canonical browsing tasks.

Assessment
- Convincing evidence:
  - Comprehensive benchmarks, consistent gains, Heavy Mode improvements, stability plots, and ablations by context length and environment support the core claims (Figures 1, 6–10; Table 1).
- Caveats:
  - Some evaluations depend on LLM judges (Appendix B), which can introduce bias or variance across systems.
  - BrowseComp shows a gap vs OpenAI DeepResearch in Pass@1 (Table 1), suggesting headroom and task-specific variance.

## 6. Limitations and Trade-offs
- Explicitly noted (Section 5.1)
  - Context limit: 128K may still be insufficient for the hardest, longest tasks; better context management or larger windows needed.
  - Model scale: only the 30B model is released so far; a larger version is “in progress.”
  - Report fidelity and preference alignment: ongoing improvements to ensure faithful, useful outputs.
  - RL efficiency: exploring partial rollouts will require addressing off-policy challenges (distributional shift).
  - Tooling scope: current Deep Research training assumes specific prompts and a fixed tool set.
- Additional trade-offs from the method and experiments
  - Reward sparsity: 0/1 terminal rewards (RLVR) are simple and verifiable but can be sparse, potentially slowing learning for tasks without ground-truth answers (Section 3.4.3).
  - Evaluation dependence on judges: Benchmarks like GAIA and xBench use LLM judges (Appendix B), which may introduce variability or favor certain answer styles or formats.
  - Heavy Mode cost: Parallel exploration at test time improves accuracy (Figure 6) but increases compute and latency—important for production use.
  - Environment smoothing: The unified sandbox abstracts away many real-world failures; while crucial for stable training, it may partially hide robustness issues encountered by naive deployments (Section 3.4.3).
  - Synthetic-data alignment: Although carefully designed and verified, synthetic distributions can still diverge from real web complexity, particularly for open-world ambiguity or rapid knowledge drift.

## 7. Implications and Future Directions
- How this changes the landscape
  - Establishes an open, reproducible recipe for deep research agents: unified mid-training + post-training with automated, verifiable synthetic data, robust environments, and practical RL (Figure 2; Sections 3.3–3.4).
  - Demonstrates that strong agentic capability can be achieved with moderate model size by investing in data/infra design (“3.3B activated per token” efficiency; Section 1; Table 1).
- Follow-up research enabled/suggested
  - Off-policy and partial-rollout RL for efficiency (Section 5.1).
  - Better reward functions (e.g., process or step-level verifiers) to reduce sparsity and encourage truthful intermediate states.
  - Advanced memory and state compression beyond `S_t` to push beyond 128K without losing long-horizon coherence (Sections 3.1, 5.1).
  - Expanded tool ecosystems and robust routing across heterogeneous APIs; learning to select/compose tools dynamically.
  - Richer simulated environments and “wind tunnel” protocols for domains beyond web browsing (e.g., APIs, GUIs, scientific workflows).
  - Stronger, standardized evaluations that reduce judge dependence and measure reliability, cost, and latency alongside accuracy.
- Practical applications
  - Enterprise research copilots (market/competitive analysis), scientific literature reviews (Google Scholar + PDF parsing), legal/financial due diligence (document parsing + evidence trails), education/tutoring on complex topics, and investigative journalism requiring multi-source corroboration.
  - Heavy Mode can serve as an “escalation” path: routine queries run in standard mode; high-stakes questions trigger parallel exploration plus synthesis for higher reliability (Figure 6).

In sum, Tongyi DeepResearch contributes a complete, open blueprint for building reliable deep research agents: inject agentic priors via mid-training, scale with automated, verifiable synthetic data, train in stable environments with pragmatic RL, and deploy with context management plus optional test-time scaling. The empirical results across diverse benchmarks (Table 1; Figures 1, 6–11) and the detailed systems design make this a strong foundation for future open-agent research and real-world deployments.
