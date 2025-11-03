# Memento: Fine-tuning LLM Agents without Fine-tuning LLMs

**ArXiv:** [2508.16153](https://arxiv.org/abs/2508.16153)

## 🎯 Pitch

Memento introduces a breakthrough agent architecture that enables Large Language Model (LLM) agents to continually learn and adapt by leveraging an external episodic memory and online reinforcement learning, without ever updating the LLM’s parameters. This paradigm allows agents to improve in real time by reasoning over past successes and failures, achieving state-of-the-art results on challenging deep research and QA benchmarks—providing a scalable, efficient pathway to truly adaptable AI systems without the cost or risk of model retraining.

---

## 1. Executive Summary
Memento proposes a way to continually improve Large Language Model (LLM) agents without updating any model weights. It turns “learning” into writing and reading from an external episodic memory and learns a retrieval policy (which past case to consult) with lightweight online reinforcement learning. Implemented as a planner–executor agent for deep research, Memento reaches top-tier results on GAIA and strong performance on multiple open-domain QA benchmarks, while showing clear continual-learning gains from its memory (Figure 1; Tables 1–2, 4–5).

## 2. Context and Motivation
- Problem addressed
  - LLM agents used for complex tasks (planning, tool use, multi-step reasoning) are either:
    - Rigid: fixed pipelines with handcrafted reflection/workflows that don’t adapt after deployment.
    - Expensive to adapt: approaches that fine-tune the underlying LLM via supervised learning or reinforcement learning, which require heavy compute, data, and risk catastrophic forgetting.
  - The paper’s central question (Introduction): how to build agents that learn continuously from changing environments “without the prohibitive cost of fine-tuning the underlying LLMs?”

- Importance
  - Real-world agent deployments (web research, data analysis, multimodal workflows) face constant novelty. Static agents stagnate; continually fine-tuning models is impractical and brittle. A scalable, low-cost, online learning mechanism is valuable both practically (cheaper, faster adaptation) and conceptually (separates agent learning from model weights).

- Prior approaches and gaps (Section 2)
  - Parametric approaches: update LLM weights via SFT or RL (e.g., START, GRPO-based methods). Pros: flexibility; Cons: costly, data-hungry, and prone to forgetting.
  - Non-parametric retrieval (RAG): pulls from static corpora; typically lacks experience-based credit assignment and online improvement.
  - Agent memory systems: various designs (reflection, memory banks, forgetting schedules), but many simply append more experience without principled selection/valuation, causing swamping and inefficient retrieval.

- Positioning
  - Memento unifies case-based reasoning (CBR) with a formal Memory-augmented MDP and a learned retrieval policy via soft Q-learning (Section 3). It avoids LLM fine-tuning by:
    - Storing rich episodic traces (task, plan, success/failure).
    - Learning which past cases to reuse next time.
  - It instantiates this in a practical planner–executor architecture with tool use (Model Context Protocol; Section 4), demonstrating state-of-the-art performance and continual learning.

## 3. Technical Approach
At a high level, Memento is an agent that, at every step, consults a growing bank of past cases; chooses which case to reuse; adapts it with an LLM to produce the next action; executes; observes feedback; and writes the new experience back to memory.

Step-by-step:

1) Formal problem: Memory-augmented MDP (Section 3; Definition 3.1; Figure 2)
- Augment a standard Markov Decision Process with a memory space `M` that contains an episodic “case bank.”
- State `s` and action `a` are sequences of tokens (the agent’s textual context and output).
- Each time step `t`, the agent maintains `M_t = {c_i}^N_{i=1}` of cases `c_i = (s_i, a_i, r_i)`.

2) Case-Based Reasoning agent (Definition 3.2; Eq. (1))
- The agent’s overall policy is a mixture over retrieved cases:
  - Retrieve a case `c` from memory with policy `μ(c | s, M)` (the retrieval policy).
  - Condition the LLM action generator on both current state and retrieved case: `p_LLM(a | s, c)`.
  - Overall action distribution: `π(a | s, M) = Σ_c μ(c | s, M) · p_LLM(a | s, c)`.
- Intuition: for a new problem, find similar past problems, read their successful/failed attempts, and adapt.

3) Learning the retrieval policy with maximum-entropy RL (Section 3; Eq. (3)–(8), Appendix A)
- Treat “selecting a case” as the action of the retrieval policy `μ`.
- Objective adds entropy to encourage diversity: `J = E[ Σ_t (R(s_t, a_t) + α H(μ(·|s_t, M_t))) ]` (Eq. (3)).
- Soft-Q formulation yields a value function (Eq. (4)) and case-value function `Q(s, M, c)` (Eq. (5)).
- Closed-form optimal retrieval policy is a softmax over Q-values (Eq. (7)):
  - `μ*(c | s, M) ∝ exp(Q*(s, M, c) / α)`
- Learn Q by soft Q-learning with temporal-difference updates (Eq. (8)).

4) Two practical Q-learning instantiations (Section 3; Algorithm 1; Section 4.2)
- Challenge: states, cases are natural language strings—difficult value approximation.
- Option A: Kernel-based episodic control (Eq. (9)–(11); Algorithm 1)
  - Maintain an episodic memory `D = {(s, c, Q)}`.
  - Approximate `Q(s, M, c)` by a similarity-weighted average over past `(s', c)` entries using a learnable kernel `k_θ(s, s')` (Eq. (9)).
  - Train θ with TD loss (Eq. (10)), using a target network for stability (Algorithm 1 lines 1, 10–13).
- Option B: Single-step classification for planning (Section 4.2; Eq. (14)–(16))
  - In Memento’s planner, CBR selection is single-step (no bootstrapping), and rewards are binary (success/failure).
  - Train a neural `Q(s, c; θ)` as the probability that case `c` is useful for state `s` using cross-entropy (Eq. (15)).
  - Retrieval uses Top-K by `Q(s, c; θ)` (Eq. (16)) for stability/interpretability.

5) Memory read/write operations (Section 4.2; Eqs. (12)–(16))
- Write (append new case): `M_{t+1} = M_t ∪ {(s_t, a_t, r_t)}` (Eq. (12)); only the final step of a trajectory is stored to avoid redundancy (Section 5.3).
- Read (non-parametric): retrieve Top-K most similar past states using a frozen encoder and cosine similarity (Eq. (13)). Implemented with SimCSE (Section 5.3).
- Read (parametric): retrieve Top-K highest `Q(s, c; θ)` (Eq. (16)), trained online with new cases.

6) Planner–executor architecture with tool use (Section 4; Figure 3)
- Stage 1: Case-Based Planning
  - Planner (e.g., `gpt-4.1`) retrieves K cases from Case Memory, composes a plan and decomposes into subtasks.
  - Subtask Memory tracks subtasks and outcomes; Tool Memory logs tool calls/results.
- Stage 2: Tool-Based Execution
  - Executor (e.g., `o3` for GAIA; `o4-mini` elsewhere) is an MCP client that runs subtasks by invoking tools (search, crawl, code, math, multimodal parsing).
  - Results are written to memory; planner can replan based on outcomes. When finished, the final (state, plan, success) is written as a new case.

7) Tooling (Section 4.3)
- External knowledge: Searxng meta-search; re-ranking; Crawl4AI for page content.
- Multimodal processing: VLM image captioning, ASR for audio, slide parsing, spreadsheets, archives, PDFs via Chunkr AI fallbacks, etc.
- Reasoning tools: sandboxed code execution (Python/shell; whitelisted libraries), math operations.

Why these design choices?
- Learning to retrieve cases (instead of fine-tuning the LLM) keeps compute low, adapts online, and avoids catastrophic forgetting (Sections 1–2).
- Single-step Q for planning turns unstable TD targets into stable supervised updates (Eq. (15)), well-suited to binary success signals (Section 4.2).
- Top-K deterministic retrieval improves interpretability and reduces variance relative to sampling (Section 4.2).
- MCP makes tool integration scalable and model-agnostic (Figure 3).

## 4. Key Insights and Innovations
- Memory-augmented decision process for agents without LLM fine-tuning (Section 3; Figure 2)
  - Innovation: Formalizing agent behavior as “retrieve a case → adapt with LLM → act,” and learning the retrieval policy `μ` with soft Q-learning.
  - Significance: Converts “agent learning” into optimizing which experiences to consult, decoupling it from model weights.

- Case-based reasoning as the core planning prior (Definition 3.2; Eq. (1); Section 4.1)
  - Innovation: The agent’s policy is a mixture over retrieved cases; planning prompts explicitly include retrieved successes and failures.
  - Significance: Empirically improves robustness and generalization, especially out-of-distribution (+4.7 to +9.6 absolute points; Figure 1d).

- Two complementary memory designs (Sections 3–4.2)
  - Non-parametric retrieval: fast, encoder-based similarity (Eq. (13)).
  - Parametric retrieval: learned Q-value of case usefulness (Eq. (15), (16)).
  - Significance: Both deliver continual-learning gains; parametric retrieval yields the strongest curves (Figure 1c; Table 4).

- Practical planner–executor with MCP tools (Section 4; Figure 3)
  - Innovation: A clean separation—planner does stateful CBR-based planning, executor performs tool calls and reasoning. Tool access via MCP provides a unified interface to search, crawl, code, and multimodal tools.
  - Significance: Achieves top-tier performance on realistic long-horizon, tool-heavy tasks (Tables 1–2), and makes the approach readily extensible.

## 5. Experimental Analysis
Evaluation setup
- Datasets (Section 5.1)
  - GAIA: 450 long-horizon questions with 3 difficulty levels; EM metric (Section 5.2).
  - DeepResearcher suite (NQ, TQ, HotpotQA, 2Wiki, MusiQue, Bamboogle, PopQA): open-domain QA with F1 and Partial Match (PM) metrics (Table 1; Section 5.2).
  - SimpleQA: 4,330 fact-seeking questions; accuracy (Figure 4).
  - Humanity’s Last Exam (HLE): 2,500 long-tail academic questions; PM metric (Figure 4).
- Metrics (Section 5.2)
  - GAIA: Exact Match (EM).
  - Others: macro-F1 and Partial Match (PM); PM judged by `gpt-4o-mini` with the DeepResearcher prompt.
- Models and components (Section 5.3)
  - Planner: `gpt-4.1`.
  - Executor: `o3` (GAIA), `o4-mini` (others).
  - Vision: `gpt-4o`; video: Gemini 2.5 Pro; audio: AssemblyAI.
  - CBR non-parametric: SimCSE embeddings + cosine similarity.
  - CBR parametric: SimCSE features → 2-layer MLP for Q(s,c).

Main quantitative results
- GAIA (Table 2; Figure 1a)
  - Validation: “Pass@3” 87.88% (top-1). Difficulty breakdown: Level1 96.23%, Level2 90.70%, Level3 61.54%.
  - Test: 79.40% EM overall (Level1 90.32%, Level2 75.47%, Level3 71.43%). Competitive with leading closed and open frameworks.
  - Takeaway: Memento is strong across levels; Level 3 remains hardest but still competitive.

- DeepResearcher (Table 1; Figure 1b)
  - Average across 7 datasets: 66.6% F1 and 80.4% PM.
  - Large gains over prompt-only baselines (e.g., CoT+RAG at 37.7%/43.2%).
  - Outperforms state-of-the-art training-based agents like DeepResearcher (51.8%/60.5%) by a notable margin.
  - Per-dataset highlights: `2Wiki` 81.4/94.1; `TQ` 85.5/93.9.

- OOD generalization (Figure 1d; Section 5.5.4)
  - After training on NQ, TQ, HotpotQA, 2Wiki, retrieval of 4 cases improves on OOD datasets: MusiQue, Bamboogle, PopQA by +4.7 to +9.6 absolute accuracy.

- Continual learning curves (Figure 1c; Table 4)
  - Over 5 iterations of experience accumulation:
    - No CBR: 78.65 → 84.47.
    - Non-param CBR: 79.84 → 84.85.
    - Parametric CBR: 80.46 → 85.44 (best).
  - Takeaway: memory-based case accumulation yields consistent improvements.

- Component ablations (Table 5)
  - Replacing offline executor with online tools helps SimpleQA (+28.8 F1 / +63.3 PM) and HLE (+4.8 / +7.1) but can hurt DeepResearcher without planning (−18.0 / −2.1), interpreted as evidence of contamination/overreliance on parametric knowledge in those datasets.
  - Adding planning (Memento w/o CBR) yields strong gains on all three benchmarks.
  - Adding CBR on top of planning further improves results consistently.

- SimpleQA and HLE (Figure 4)
  - SimpleQA accuracy: 95.0%, exceeding WebSailor (93.5%), WebDancer (90.5%), DeepSeek-r1-React (72.2%).
  - HLE PM: 24.40%, second overall and within 0.92 points of GPT-5 (25.32%).

- Hyperparameter K (number of retrieved cases) (Table 3)
  - Best at K=4 for the DeepResearcher average (F1=64.5, PM=78.5). Larger K can plateau or degrade due to noise.

- Cost/behavior analysis (Figures 5–6; Table 6)
  - Tool usage: code, search, crawl dominate, especially as difficulty rises (Figure 5).
  - Tokens/costs: input tokens grow sharply with task difficulty (Level1~26k, Level2~48k, Level3~121k average input tokens), while output tokens remain relatively small (Figure 6).
  - Fast vs slow planner: `gpt-4.1` (fast) with `o3` executor outperforms slow planners across levels (Table 6), attributed to better decomposition and less verbosity.

Do the experiments support the claims?
- Yes, for three core claims:
  - Competitive or state-of-the-art performance without LLM fine-tuning (GAIA, SimpleQA, DeepResearcher).
  - Continual learning via case accumulation (Figure 1c; Table 4).
  - Generalization benefits from CBR (Figure 1d).
- The ablations are informative: they isolate the value of planning, tools, and memory. Robustness is examined via OOD tests and hyperparameter K.

Caveats and mixed results
- Online tools without planning can hurt on DeepResearcher (Table 5c), interpreted as contamination effects and the strength of internal parametric knowledge. This underscores that tool use must be planned and integrated carefully.

## 6. Limitations and Trade-offs
- Assumptions about reward and task structure
  - The parametric memory uses binary rewards for single-step case selection in planning (Eq. (15)). Tasks with nuanced, graded outcomes may require richer signals or multi-step RL.
  - Only the final step of a trajectory is stored, potentially losing fine-grained credit assignment within long plans (Section 5.3).

- Memory growth and curation
  - Although Top-K retrieval and learned Q mitigate swamping, the case bank grows online. Long deployments may demand forgetting, summarization, or selective retention strategies beyond what is presented (Section 2.3 discusses swamping in prior systems).

- Compute and token costs shift rather than vanish
  - While no model fine-tuning is required, complex tasks still incur large input-context costs (Figure 6). This is especially true at higher difficulty levels where evidence aggregation dominates.

- Dependency on closed, high-end models and tools
  - The best results use proprietary models (`gpt-4.1`, `o3`, `o4-mini`, `gpt-4o`, Gemini 2.5 Pro) and a non-trivial tool stack (Section 5.3; Section 4.3). Reproducing results exactly may be costly or constrained.

- Upper-bound challenges on very long-horizon tasks
  - Level-3 GAIA remains hard; although Memento performs well, the best gains are still at lower levels. There is room to improve decomposition and long-range credit assignment (Table 2; Figure 1a).

- Evaluation nuances
  - PM is judged by an LLM (Section 5.2), which may introduce evaluation bias. The paper follows prior work to ensure comparability, but this remains a general limitation of LLM-judge metrics.

## 7. Implications and Future Directions
- Conceptual shift: learn “what to recall,” not “how to reweight parameters”
  - By making retrieval policy the locus of learning, Memento offers a scalable path for agents to improve from experience with minimal compute. This complements RAG and parameter tuning by focusing on episodic, task-shaped memory.

- Practical applications
  - Deep research assistants that continually get better at multi-step web tasks; data/analysis copilots that record successful workflows and reuse them; multimodal research agents with stable tool execution and evolving planning priors.

- Research avenues
  - Richer credit assignment: store and value substeps or subplans, not just final outcomes; hierarchical or option-level cases.
  - Memory management: principled forgetting, summarization, and deduplication; safety filters for memory content.
  - Multi-agent settings: share case banks across agents (e.g., specialized executors) with trust and reputation signals.
  - Beyond binary rewards: graded utility signals, learned evaluators for partial success, and multi-objective criteria (accuracy, cost, latency).
  - Open, efficient stacks: reproducing competitive results with open models, smaller context windows, or distilled planners.
  - Robustness: defenses against contamination and drift when online tools and the web evolve.

Quoted highlights from the paper’s figures and tables:
- “Memento attains top-1 on GAIA validation (87.88% Pass@3) and 79.40% on the test set” (Abstract; Table 2; Figure 1a).
- “It reaches 66.6% F1 and 80.4% PM on the DeepResearcher dataset” (Abstract; Table 1; Figure 1b).
- “Case-based memory adds 4.7% to 9.6% absolute points on out-of-distribution tasks” (Abstract; Figure 1d).
- Continual learning curves show steady improvements over iterations, with parametric CBR best (Figure 1c; Table 4).
- Best K for case retrieval is small (K=4) (Table 3), suggesting “a small, curated memory yields optimal results” (Section 7).

Overall, Memento is a clear, well-scoped demonstration that principled, case-based memory with a learned retrieval policy can deliver continual learning for LLM agents—without touching the underlying LLM weights—while remaining practical and performant on demanding, tool-heavy tasks.
