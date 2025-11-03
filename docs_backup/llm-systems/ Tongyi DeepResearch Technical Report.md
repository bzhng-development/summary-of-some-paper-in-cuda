# Tongyi DeepResearch Technical Report

**ArXiv:** [2510.24701](https://arxiv.org/abs/2510.24701)
**Authors:** Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, Kuan Li, Liangcai Su, Litu Ou, Liwen Zhang, Pengjun Xie, Rui Ye, Wenbiao Yin, Xinmiao Yu, Xinyu Wang, Xixi Wu, Xuanzhong Chen, Yida Zhao, Zhen Zhang, Zhengwei Tao, Zhongwang Zhang, Zile Qiao, Chenxi Wang, Donglei Yu, Gang Fu, Haiyang Shen, Jiayin Yang, Jun Lin, Junkai Zhang, Kui Zeng, Li Yang, Hailong Yin, Maojia Song, Ming Yan, Peng Xia, Qian Xiao, Rui Min, Ruixue Ding, Runnan Fang, Shaowei Chen, Shen Huang, Shihang Wang, Shihao Cai, Weizhou Shen, Xiaobin Wang, Xin Guan, Xinyu Geng, Yingcheng Shi, Yuning Wu, Zhuo Chen, Zijian Li, Yong Jiang
**Institutions:** Tongyi Lab, Alibaba Group

## 1. Executive Summary (2–3 sentences)

Reasoning: To concisely capture the paper’s essence, I identify the core technical idea that underpins all sections (a unified agentic training pipeline with synthetic data and stage-specific environments), then anchor its significance in the empirical results (state-of-the-art performance with efficient compute).

Tongyi DeepResearch proposes an open, end-to-end training paradigm for agentic large language models (`agentic LLMs`, models that can plan, act via tools, and adapt through multi-step interactions) by unifying `agentic mid-training` (continual pretraining to instill agent-like priors) and `agentic post-training` (supervised fine-tuning plus reinforcement learning with verifiable rewards). This matters because it enables long-horizon, autonomous “deep research” (multi-step internet information-seeking and synthesis) with fewer activated parameters per token (3.3B) while achieving state-of-the-art results across multiple benchmarks (e.g., 32.9 on Humanity’s Last Exam; 75.0 on xBench-DeepSearch; 90.6 on FRAMES; see Figure 1 and Table 1 in Section 4).

## 2. Context and Motivation

Reasoning: I infer the problem setting by synthesizing the Introduction and Design Principle sections, then connect it to real-world constraints (tool reliability, data scarcity) and prior art’s limitations (closed-source systems, lack of agentic inductive bias).

- Problem addressed:
  - The paper targets `deep research agents`, meaning systems that autonomously perform multi-step reasoning and information-seeking (search, browse, compute) over long horizons on the web to produce in-depth reports. Section 1 defines this as tasks “completed in tens of minutes” that otherwise require hours. 
  - The gap: most high-performing deep research systems are closed-source and do not expose intermediate processes; open models typically lack `agentic inductive bias` (a learned predisposition toward planning, tool-use, and decision-making), leading to sub-optimal post-training when they must learn agency and alignment simultaneously (Section 2, “Agent Training Pipeline”).

- Importance:
  - Real-world impact: Tool-using LLMs that can autonomously research have obvious productivity benefits and form a stepping stone toward general-purpose agents (Section 1).
  - Theoretical significance: The work reframes training environments (real vs. simulated vs. “prior world”) as co-designed components of the training pipeline (Section 2, “Learning Through Environmental Interaction”), which advances methodology for stable agent learning.

- Prior approaches and limitations:
  - Closed systems: OpenAI DeepResearch (2025a), Gemini DeepResearch (2025), Grok-3 Deeper Search (2025) perform end-to-end deep research but do not provide open models or transparent training data/processes (Section 1).
  - Open research agents: Preliminary open efforts (e.g., WebWalkerQA, WebSailor) lack a unified methodology for scalable agent training and rely on costly human annotation or limited data synthesis (Section 1 and citations in References).
  - Baseline LLM agents often rely on `ReAct` (reasoning traces + actions; Yao et al., 2023) but miss long-horizon context and robust environment design, resulting in unstable tool-use and limited performance (Section 3.1, Section 3.4.3).

- Positioning relative to existing work:
  - The paper claims an “end-to-end agentic training paradigm” that integrates `agentic mid-training` (continual pretraining to instill agent priors; Section 3.3) and `agentic post-training` (SFT for cold start + RL with verifiable rewards; Section 3.4). 
  - It emphasizes a fully automated synthetic data pipeline for all stages and stage-specific environments (prior, simulated, real) to stabilize interactions and reward signals (Section 2; Figures 3–5).

## 3. Technical Approach

Reasoning: I reconstruct the pipeline step-by-step from Sections 3.1–3.4, explaining each mechanism in plain terms and referencing figures/equations. I define all technical terms (e.g., `ReAct`, `context window`, `Markovian`) and show how components fit together.

- Agent formulation (Section 3.1):
  - Core components at each step `t`:
    - `Thought (τ_t)`: the model’s internal reasoning (analysis, recall, planning, self-reflection).
    - `Action (a_t)`: an external operation using tools; intermediate actions are tool calls, the final action `a_T` is the user-facing report.
    - `Observation (o_t)`: environment feedback to the last action.
  - `ReAct` framework:
    - `ReAct` (Yao et al., 2023) interleaves reasoning and acting. The trajectory `H_T` is a sequence of `(τ_i, a_i, o_i)` triples (Equation (1)), with policy `π` generating `τ_t` and `a_t` conditioned on history `H_{t-1}` (Equation (2)).
    - Rationale (Section 3.1): Simplicity scales better (“The Bitter Lesson”; Sutton, 2019) than complex, rigid designs requiring heavy prompt-engineering (see Section 3.1 remarks).
  - Context management (Section 3.1):
    - Problem: long-horizon tasks risk exceeding the `context window` (the maximum token length the model can attend to).
    - Solution: A `Markovian state reconstruction` where, at each step, the model conditions on a small “workspace”: the original question `q`, an evolving report `S_t` (a compressed memory of prior reasoning), and the immediate last action/observation (`a_t`, `o_t`). This update is formalized as sampling `(S_t, τ_{t+1}, a_{t+1}) ~ π(· | S_{t-1}, a_t, o_t)` (Equation (3)). This reduces context bloat and enforces structured synthesis.

- Overall training recipe (Section 3.2; Figure 2):
  - Initialize from `Qwen3-30B-A3B-Base` (Yang et al., 2025), a 30.5B-parameter model with `A3B` sparse activation—only ~3.3B parameters active per token (Section 1).
  - Two phases:
    - `Agentic mid-training`: `Agentic CPT` (continual pretraining) stages to learn agentic priors, moving from 32K to 128K context length (Section 3.3).
    - `Agentic post-training`: `SFT` to establish a stable policy (cold start), then `RL` to optimize tool-use and planning (Section 3.4).

- Agentic mid-training (Section 3.3; Figure 3):
  - Training configuration:
    - Objective: standard `next-token prediction` (language modeling) but using agentic behavior data (Section 3.3.1).
    - Two stages: 32K context window first, then 128K, adding many long-sequence (64K–128K) agent behavior samples while interleaving some general pretraining data to keep broad language competence.
  - Large-scale agent behavior data synthesis (Section 3.3.2):
    - `Question Synthesis`: Build an entity-anchored open-world memory from web data and trajectories; sample entities and related knowledge to generate diverse research-level questions (multi-hop, numeric computation).
    - `Planning Action`: Decompose problems and predict first actions using open-source models; use rejection sampling tied to entity knowledge to filter poor plans.
    - `Reasoning Action`: Guide large models through two-stage reasoning-chain generation; apply dual filters on reasoning length and answer consistency to ensure quality.
    - `Decision-Making Action`: Explicitly model choice among multiple feasible reasoning/action paths at each step by reconstructing trajectories into multi-step decision sequences.
    - `Function-calling via environment scaling`: Automatically build heterogeneous simulated environments (as read–write databases) to scale function-calling diversity, improving general agentic capability (Section 3.3.2).

- Agentic post-training (Section 3.4):
  - High-quality data synthesis (Section 3.4.1; Figure 4):
    - Pipeline steps: (1) Graph construction via random walks and web search (including isomorphic tables from websites), (2) Subgraph sampling to create initial QAs, (3) `Uncertainty injection`—systematically increasing difficulty via “atomic operations” on entity relationships (e.g., merge similar entities), formalized using set theory to control reasoning structure and reduce shortcuts (see citations to Tao et al., 2025; Li et al., 2025c/b).
    - Verification: The set-theoretic formalization supports efficient automatic correctness verification.
    - Scaling PhD-level tasks: A question-crafting agent iteratively escalates complexity with tool support, producing multi-source reasoning seeds and compounding them (Section 3.4.1).
  - Supervised fine-tuning (SFT) for cold start (Section 3.4.2):
    - Use synthesized high-quality QA trajectories from strong open-source models; apply strict `rejection sampling` to keep only high-quality, diverse patterns.
    - Mixed training modes:
      - `ReAct Mode`: Input `H_t` (history) → output `τ_i`, `a_i`.
      - `Context Management Mode`: Input previous summary `S_{t-1}`, tool call `a_{i-1}`, tool response `o_{i-1}` → output current summary `S_i`, thought `τ_i`, and tool call `a_i`. This trains the model to synthesize observations into concise state summaries for long-horizon stability.
    - Two-stage SFT by context length: Stage 1 at 40K (shorter ReAct samples + all context-managed samples), Stage 2 at 128K (longer ReAct samples plus a small 40K spill for stability).
  - Agentic reinforcement learning (RL) (Section 3.4.3; Figure 5):
    - `RLVR` (reinforcement learning with verifiable rewards): Roll out a complete attempt; reward = 1 if final answer matches ground truth, else 0 (Section 3.4.3; Guo et al., 2025).
    - Tools and environments:
      - Real-world tools: `Search`, `Visit`, `Python Interpreter`, `Google Scholar`, `File Parser` (Appendix D). A unified sandbox layer ensures reliable calls via QPS limits, caching, retries, graceful degradation, and failover (Section 3.4.3).
      - `Simulated environment`: Offline Wikipedia + local `RAG` (Retrieval-Augmented Generation, a paradigm where models retrieve external documents before generating) to mirror web interactions cheaply and deterministically; used to validate algorithms rapidly (Section 3.4.3).
    - On-policy asynchronous rollout framework:
      - Built on `rLLM` (Tan et al., 2025) with separate asynchronous servers for model inference and tool invocation; centralized handler merges outputs (Figure 5). This parallelizes many agents interacting simultaneously, speeding RL.
    - RL training algorithm (Section 3.4.3; Equations (4)–(5)):
      - Tailored `GRPO` (a PPO-like method where advantages are computed relative to group baselines): token-level policy gradient with clipping (clip-higher to encourage exploration, per `DAPO`; Yu et al., 2025).
      - Strict `on-policy` sampling: Trajectories are always from the current policy (importance ratio `r_{i,j}` remains 1.0).
      - Advantage estimation: `Â_{i,j} = R_i – mean({R_i})` with leave-one-out to reduce variance (Chen et al., 2025).
      - Negative sample filtering: Exclude degenerate failures (e.g., exceeded length with no final answer) to avoid instability and policy collapse.
    - Automatic data curation:
      - Start from a large dataset `D`; build `D'` by removing trivially easy/hard problems relative to the current policy to focus on the “learning frontier”.
      - As training progresses, a background process continuously rescales `D'` by replacing mastered problems with newly challenging ones sampled using intermediate checkpoints (Section 3.4.3).
  - Model merging (Section 3.4.4; Equation (6)):
    - Weighted parameter interpolation among several variants derived from the same base (`θ_merged = Σ_k α_k θ^(k)`), preserving strengths and improving generalization without extra optimization.

- Heavy Mode (Section 4.3; Figure 6):
  - `Research–Synthesis` test-time scaling:
    - `Parallel research`: Launch `n` agents using the context management paradigm; each produces a final compressed report `S_T^u` and an answer (Equation (7)).
    - `Integrative synthesis`: A synthesis model aggregates `{(S_T^u, answer_u)}_{u=1..n}` to produce `answer_final` (Equation (8)).
    - Benefit: Compressed summaries `S_T^u` allow aggregation of many trajectories within a manageable context, unlike concatenating full reasoning traces.

## 4. Key Insights and Innovations

Reasoning: I distinguish which contributions are fundamentally new mechanisms versus incremental improvements, referencing where they appear and why they matter for performance and stability.

- Unifying agentic mid-training and post-training (Sections 2–3; Figure 2):
  - Novelty: Introduces `agentic mid-training` (`Agentic CPT`) as a bridge that instills agentic priors before alignment and RL. Most prior deep-research systems only rely on post-training (Section 2).
  - Significance: Reduces optimization conflicts of teaching agency and alignment simultaneously; strengthens long-horizon coherence via larger context windows (32K → 128K) and agentic data (Section 3.3.1). This is a foundational innovation in training recipe design.

- Fully automated synthetic data pipeline across stages (Sections 3.3.2, 3.4.1; Figures 3–4):
  - Novelty: End-to-end generation of research-level questions, multi-step agent trajectories (planning, reasoning, decision-making), and controllable difficulty via set-theoretic operations (“uncertainty injection”).
  - Significance: Avoids costly human annotation, allows construction of “super-human-level datasets” with stable distributions, and supports verifiable rewards critical for RL stability (Section 2 and 3.4.1). This is a fundamental capability enabler rather than an incremental tweak.

- Stage-specific environment design (Section 2; Section 3.4.3):
  - Novelty: Tri-partite environment strategy—`Prior World` (no responses, pure planning), `Simulated` (offline, reproducible), `Real-world` (authentic but noisy)—co-designed with training phases.
  - Significance: Solves non-stationarity and cost constraints; ensures stable and deterministic interactions during data generation and RL (Section 2). This is methodological infrastructure that directly impacts agent reliability.

- Context management with Markovian state reconstruction (Section 3.1; Equation (3)):
  - Novelty: Replaces full-trace conditioning with a dynamic summary `S_t` + last action/observation.
  - Significance: Prevents context overflow and enforces periodic synthesis, critical for long-horizon tasks and enables Heavy Mode test-time scaling by compressing trajectories (Sections 3.1, 4.3).

- On-policy asynchronous RL with curated dynamic curricula (Section 3.4.3; Figure 5):
  - Novelty: Step-level asynchronous rollouts, strict on-policy training, variance-reduced advantages, and automated difficulty refresh.
  - Significance: Yields stable reward growth and entropy convergence (Figure 8), even under tool stochasticity and long contexts, emphasizing data/environment design over algorithmic novelty.

- Model merging for capability interpolation (Section 3.4.4; Equation (6)):
  - Incremental but practical: Combines specialized variants to balance strengths without additional optimization cost; improves generalization.

## 5. Experimental Analysis

Reasoning: I assess the empirical methodology using the setup in Section 4, summarizing results from Table 1 and Figures 1, 6–11. I check whether the numbers and curves plausibly support the claims, and I note caveats (judge models, multiple runs).

- Evaluation methodology (Section 4.1; Appendix B):
  - Benchmarks:
    - `Humanity’s Last Exam` (2,154 text-only questions; Chai et al., 2025).
    - `BrowseComp` and `BrowseComp-ZH` (web-browsing benchmarks; Wei et al., 2025; Zhou et al., 2025).
    - `GAIA` (Mialon et al., 2023), `WebWalkerQA` (Wu et al., 2025b), `xBench-DeepSearch` (Xbench Team, 2025), `FRAMES` (Krishna et al., 2025), plus `xbench-DeepSearch-2510` (new variant reported in Figure 1).
  - Metrics:
    - `Avg@3`: average score across three independent runs (Section 4.1).
    - `Pass@1` and `Pass@3`: best-of-three and any success across three runs (Figure 7).
  - Inference parameters: temperature 0.85, repetition penalty 1.1, top-p 0.95; up to 128 tool calls; 128K context (Section 4.1).
  - Judges: GAIA/WebWalkerQA via `Qwen2.5-72B-Instruct`; xBench variants via `Gemini-2.0-Flash-001`; BrowseComp variants via `GPT-4o-2024-08-06`; Humanity’s Last Exam via `o3-mini` (Appendix B). This introduces dependency on external judges.

- Main quantitative results (Table 1; Figure 1):
  - On `Humanity’s Last Exam`: `Tongyi DeepResearch` scores 32.9 (Avg@3), higher than DeepSeek-V3.1 (29.8), OpenAI o3 (24.9), GLM-4.5 (21.2), OpenAI DeepResearch (26.6). 
  - On `BrowseComp`: 43.4 (Tongyi) vs. 51.5 (OpenAI DeepResearch), 30.0 (DeepSeek-V3.1), 26.4 (GLM-4.5).
  - On `BrowseComp-ZH`: 46.7 (Tongyi) vs. 49.2 (DeepSeek-V3.1), 58.1 (OpenAI o3), 37.5 (GLM-4.5).
  - On `GAIA`: 70.9 (Tongyi) vs. 68.3 (Claude-4-Sonnet), 66.0 (GLM-4.5).
  - On `xBench-DeepSearch`: 75.0 (Tongyi) vs. 69.0 (Kimi Researcher), 65.0 (Claude-4-Sonnet).
  - On `WebWalkerQA`: 72.2 (Tongyi) vs. 71.7 (OpenAI o3), 63.1 (DeepSeek-V3.1).
  - On `FRAMES`: 90.6 (Tongyi) vs. 84.0 (OpenAI o3), 83.7 (DeepSeek-V3.1), 78.9 (GLM-4.5).
  - Visual summary in Figure 1 confirms strong performance across benchmarks, with model efficiency highlighted: “activating only 3.3 billion per token” (Section 1).
  - Note: `xbench-DeepSearch-2510` appears only in Figure 1 (“75+” for ChatGPT-5-Pro and “55+” for Tongyi), not Table 1; the paper states Tongyi ranks just below ChatGPT-5-Pro (Section 4.2).

- Heavy Mode improvements (Figure 6):
  - `Pass@1` gains:
    - Humanity’s Last Exam: 38.3 (Heavy) vs. 32.9 (base).
    - BrowseComp: 58.3 (Heavy) vs. 43.4 (base).
    - BrowseComp-ZH: 58.1 (Heavy) vs. 46.7 (base).
  - Interpretation: Parallel exploration plus synthesis (Equations (7)–(8)) leverages more test-time compute within a compressed context budget (Section 4.3).

- Stability and training dynamics (Figures 7–10):
  - `Pass@3` significantly exceeds `Avg@3` on some tasks (e.g., BrowseComp-ZH 63.7 vs. 46.7; Figure 7), indicating that multiple attempts unlock higher potential, consistent with stochastic environments.
  - RL reward steadily increases; policy entropy converges (Figure 8). This supports claims that environment design and data curation yield stable learning.
  - Context length during RL matters (Figure 9):
    - Larger contexts (64k) reach higher rewards due to curriculum targeting long, complex problems; shorter contexts (32k) learn to produce more concise solutions (decreasing response length) to fit constraints.
  - Interaction scaling (Figure 10a): More tool interactions (which increase effective context) improve BrowseComp accuracy.
  - Simulated environment reward mirrors real environment (Figure 10b), validating the “wind-tunnel” idea.

- General benchmarks (Figure 11):
  - Tongyi DeepResearch shows very high scores on AIME25, HMMT25, and SimpleQA compared to base reasoning-only models; the paper attributes gains to external information retrieval (`Search`) and native computation (`Python Interpreter`). 
  - Caveat: Manual evaluation for math datasets (Appendix B); results are compelling but depend on small-scale manual checks and the SimpleQA official script.

- Do experiments support the claims?
  - Evidence for stability: Reward and entropy curves (Figure 8), on-policy asynchronous framework (Figure 5), simulated vs real consistency (Figure 10b).
  - Evidence for performance and efficiency: SOTA-ish results with 3.3B activated parameters per token; competitive with closed systems (Figure 1, Table 1).
  - Dependence on judge models: Several benchmarks use external LLM judges (Appendix B), which can introduce variance or bias. Reporting `Avg@3` and `Pass@1/3` mitigates some instability.

- Ablations, failure cases, robustness:
  - Ablation-like analyses include context length effects (Figure 9), interaction scaling (Figure 10a), and multi-run metrics (Figure 7). 
  - The paper acknowledges training instabilities when including all negative rollouts and describes mitigation (Section 3.4.3). 
  - Failure cases are not cataloged exhaustively; however, the curriculum pipeline explicitly removes degenerate or trivially mastered items (Section 3.4.3). 

## 6. Limitations and Trade-offs

Reasoning: I extract limitations explicitly noted in Section 5.1 and add trade-offs implied by the methodology (judge dependence, synthetic data bias, tool API variability), grounding each point in the paper’s content.

- Assumptions and dependencies:
  - Synthetic data validity: The pipeline assumes set-theoretic formalizations and verification are sufficient for correctness; while practical (Section 3.4.1), synthetic distributions may diverge from real-world web noise or user preferences.
  - Judge models: Multiple benchmarks use third-party LLM judges (Appendix B), introducing evaluation dependency and potential bias.
  - Sparse activation (`A3B`): Efficiency (3.3B per token) is premised on activation strategies; real-world latency may still be significant due to tool interactions (Section 3.4.3).

- Scenarios not fully addressed:
  - Extreme long-horizon tasks beyond 128K context: The paper states “current 128K context length remains insufficient” (Section 5.1). 
  - Off-policy RL: Author notes future work on partial rollouts will require addressing distributional shift (Section 5.1).
  - Broader tool ecosystems: Current DeepResearch training targets specific prompts and tool sets; generalization to arbitrary tool APIs and domains is planned (Section 5.1).

- Computational, data, and scalability constraints:
  - Real-world environment non-stationarity and QPS limits complicate large-scale RL (Sections 2, 3.4.3). The sandbox mitigates but does not eliminate external failures.
  - Synthetic pipeline scale vs. quality: While automated, difficulty tuning and uncertainty injection rely on heuristics and formal operations; overfitting to synthetic patterns is possible without careful curation.

- Open questions:
  - How robust is Heavy Mode synthesis to contradictory reports from parallel agents? The paper shows gains (Figure 6) but does not detail tie-breaking or conflict resolution mechanisms.
  - What is the sensitivity of performance to specific tool configurations and fallback strategies? Appendix D outlines tool details and open substitutes but does not quantify impact per tool.

## 7. Implications and Future Directions

Reasoning: I extrapolate from the Discussion (Section 5) and the design principles to articulate field-level impact, practical applications, and research paths that logically follow from the paper’s contributions.

- Field impact:
  - Re-centers agent training around environment co-design and synthetic data flywheels (“data flywheel” effect where better agents generate better data; Section 2), shifting the focus away from purely algorithmic RL tweaks toward pipeline engineering.
  - Demonstrates that relatively small active parameter counts can achieve SOTA in complex agent tasks when coupled with strong training recipes (Section 5.2).

- Enabled follow-ups:
  - `Agent foundation models`: The paper hints at developing unified models with scalable reasoning, memory, and autonomy beyond deep research (Section 5.3).
  - `Extended context and memory`: Advances in `context management` (Section 3.1) suggest future work on hierarchical memory, episodic retrieval, or external memory modules to surpass 128K constraints.
  - `Off-policy partial rollouts`: Methods to use partial trajectories safely (Section 5.1), e.g., importance sampling corrections, conservative policy iteration for agents.
  - `Robust preference alignment`: Improving report fidelity and user preference modeling (Section 5.1) could leverage reward modeling beyond binary correctness (RLVR).

- Practical applications:
  - Enterprise research automation: Competitive performance on BrowseComp/WebWalkerQA/xBench implies value for market analysis, due diligence, literature surveys, and strategic intelligence.
  - Scientific assistance: Integration of `Google Scholar` and `Python Interpreter` supports reproducible workflows for data analysis and citation tracking.
  - Educational tools: Humanity’s Last Exam and general benchmarks suggest potential for tutoring systems that can both compute and retrieve, with explainable reports.

- Representative supporting claims:
  - Performance summary (Section 1, Figure 1): 
    > “Tongyi DeepResearch reaches 32.9 on Humanity’s Last Exam, 43.4 on BrowseComp, 46.7 on BrowseComp-ZH, 72.2 on WebWalkerQA, 70.9 on GAIA, 75.0 on xbench-DeepSearch, 90.6 on FRAMES and 55.0 on xbench-DeepSearch-2510.”
  - Pipeline and environment design (Section 2):
    > “Environments should not be passively viewed as external reality, but actively designed as systems deeply coupled with the training process.”

In sum, the paper’s mechanisms—agentic mid-training, synthetic data generation with formal difficulty control, environment co-design, context-managed reasoning, and on-policy asynchronous RL—collectively form a robust and reproducible blueprint for building open deep-research agents that perform competitively with closed systems, while highlighting data and infrastructure as the primary levers for scaling agentic intelligence.