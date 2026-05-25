# Towards General Agentic Intelligence via Environment Scaling

**ArXiv:** [2509.13311](https://arxiv.org/abs/2509.13311)

## 🎯 Pitch

This paper presents AgentScaler, a systematic pipeline that automatically builds large and diverse simulation environments for training language agents in robust function-calling and tool-use, overcoming the bottleneck of scarce, expensive real-world agentic data. By programmatically materializing API environments and employing a two-stage training strategy, AgentScaler enables small- and medium-scale models to achieve state-of-the-art performance on agent benchmarks, marking a major step toward practical, reliable, and scalable agentic intelligence for real-world applications.

---

## 1. Executive Summary
This work introduces AgentScaler, a full pipeline that builds large, verifiable tool-use environments automatically and then trains language agents in two stages (general then domain-specific) on the experiences collected in those environments. The key significance is that it replaces scarce, expensive, and hard‑to‑verify real-world tool data with scalable, programmatically verified simulations, yielding strong function‑calling performance with relatively small models across multiple agent benchmarks (τ‑bench, τ²‑Bench, ACEBench).

## 2. Context and Motivation
- Problem the paper tackles
  - Large Language Models (LLMs) need reliable “agentic” capabilities: deciding when and how to call tools/APIs in multi-step tasks to affect the external world. The bottleneck is the scarcity of high‑quality “agentic data,” i.e., full trajectories of tool calls with arguments, tool responses, and resulting state changes (Abstract; Section 1).
  - Building those trajectories in real environments is costly, fragile (API downtime), and hard to supervise. Synthetic approaches often lack naturalness or verifiability (Section 1).

- Why this matters
  - Real-world deployments (customer support, booking, operations, enterprise automation) rely on precise, robust function calling across heterogeneous APIs. Without consistent tool-use competence, models hallucinate or fail in multi-step workflows.

- Prior approaches and their limits
  - Reverse synthesis: generate user queries from observed tool calls (produces less realistic trajectories; Section 1).
  - Forward simulation with high-level intents and human–agent interplay: more natural but typically requires manual environment construction and is not scalable (Section 1).
  - LLM-simulated tool responses: cheap but can hallucinate and vary inconsistently (Section 6.1).
  - Offline execution environments for evaluation (e.g., τ‑bench): useful but usually manually built and not scaled for large-scale training (Section 6.1).

- Positioning
  - This work centers the environment itself as the lever for scaling agentic intelligence. It programmatically constructs heterogeneous, verifiable, fully simulated environments from ~30,000 APIs, then performs large-scale experience collection and two-stage fine-tuning (Sections 2–3). It claims state-of-the-art open-source performance under 1T parameters on multiple benchmarks (Table 1).

Definitions used selectively:
- Function calling: the model invokes external tools/APIs by emitting a structured call with `function_name` and `arguments`.
- Agentic data: trajectories of multi-turn interactions where the assistant performs tool calls that read or modify an environment.
- Environment (here): a domain-specific, read–write database that tools operate on (Section 2, “Design Principal”).
- pass^k: the accuracy when a model must answer the same question correctly in all k independent trials; lower as k increases indicates instability (Figure 4).

## 3. Technical Approach
The pipeline has two major phases: environment build & scaling (Section 2; Figure 1) and agent experience learning (Section 3; Figure 2).

A. Unifying view: tools as read–write operators over a database
- Core abstraction (Section 2, “Design Principal”):
  - Each tool is an operator of type `read` or `write` on an underlying database `D`. A tool call with arguments `α` is represented as `op(func)(α; D)`.
  - Tools within a domain share a common database schema `S_k`. The challenge becomes: partition tools into domains `{T_1, …, T_M}` and assign each a schema `S_k` that captures shared read–write patterns.

Why it matters
- This makes environment feedback executable and verifiable: you can check tool arguments and confirm the database state transitions caused by `write` operations.

B. Environment automatic build (Section 2.1; Figure 1, left/middle)
1) Scenario collection
   - Aggregate >30,000 APIs from ToolBench, API‑Gen, and internal repositories (Section 2.1 “Scenario Collection”).
   - Filter and rewrite API descriptions to add explicit input–output specs; also construct tool compositions from input–output relationships to form a large API pool `Θ_F` (~30k tools).

2) Tool dependency graph modeling
   - Build a graph where nodes are tools and edges mean compositional compatibility (arguments/outputs align).
   - For two tools `i` and `j`, compute parameter embeddings `ϕ(P_func)` and add an edge if cosine similarity > threshold `τ`:
     - Equation (1): `E = { (i, j) | sim(ϕ(P_i), ϕ(P_j)) > τ, i ≠ j }` (Section 2.1).
   - Cluster this graph into domains with Louvain community detection (a standard graph clustering algorithm).
   - Within each domain, refine edges using an LLM to check pairwise dependencies more carefully (to fix vector-similarity errors). Outcome: `M` domains (over 1,000).

   Design choice rationale
   - Louvain gives scalable, unsupervised community detection.
   - Parameter-embedding similarity quickly proposes edges; a second LLM pass improves precision.

3) Function schema programmatic materialization
   - For each domain, use tool parameter definitions to synthesize a domain-specific database schema (the environment state).
   - Implement each tool as Python code that executes read–write operations over that schema (Section 2.1 “Function Schema Programmatic Materialization”).
   - Manual checks show the produced schemas and code align closely with τ‑bench’s official implementations for overlapping domains (Section 2.1).

   Why this matters
   - It turns abstract tool specs into executable, testable code grounded on a concrete database—enabling deterministic execution and verifiable state transitions.

C. Agentic task construction (Section 2.2; Figure 1, right)
- Initialize an environment state that is diverse across instances.
- Sample a logically coherent tool sequence from the domain’s directed dependency graph:
  - Start at a random tool; do a directed walk until max steps or a node with no outgoing edges (Section 2.2).
- For each tool in sequence:
  - Generate the arguments.
  - Execute the tool against the database, tracking state changes.
- Synthesize the overall user intent that matches the tool sequence and environment trajectory (Figure 1, “Tasks”).
- Two levels of verifiability (Section 2.2):
  - Database-level: final state matches expected “gold” state when `write` tools are used.
  - Tool-sequence level: exact match on tool names and arguments for read-only or mixed sequences.

D. Human–agent interplay for experience collection (Section 3.1; Figure 2)
- Instantiate a simulated user with the overall intent.
- Let a task agent interact through multi-turn conversation and tool calls until the (simulated) user deems the task done.
- Record trajectories of turns, tool calls, tool responses, and environment states (Figure 2).

Data filtering: a three-stage funnel (Section 3.1 “Filtering”)
1) Validity control
   - Remove malformed dialogues and highly repetitive reasoning (n‑gram filtering).
2) Environment state alignment
   - Keep only trajectories whose final database state equals the gold state—verifies effective `write` operations.
3) Function calling exact match
   - For read-only (or read-heavy) trajectories, require exact match of both tool names and arguments to the gold plan—ensures high-fidelity supervision.

Note: They keep trajectories with tool-call errors if the final goal is still achieved, to improve robustness (Section 3.1).

E. Agentic experience learning (Section 3.2)
- Training objective masks out human instructions `h_i` and tool response tokens `ρ_t`, but conditions on them; the loss only supervises assistant tool-calls `τ_t` and assistant replies `y_t`.
  - Equation (2) (Section 3.2): a masked next-token loss over tokens in set `T` (assistant tool calls and natural language replies):
    - Intuition: learn when/how to call tools and how to wrap results in user-facing responses, without being penalized for tool outputs or user text.

- Two-stage experience learning (Section 3.2)
  1) Stage 1: General tool‑use pretraining across many domains to acquire foundational skills (tool selection, argument formation, integrating tool outputs).
  2) Stage 2: Domain specialization on verticals (e.g., retail, airline, telecom) to adapt to domain‑specific goals and constraints.

F. Models
- Train `AgentScaler` models at 4B, 8B, 30B‑A3B scales on Qwen3 backbones (Section 4.1 “Backbones”).

## 4. Key Insights and Innovations
1) Automated, verifiable environment scaling
   - What’s new: A principled way to convert tens of thousands of tool specs into executable, domain‑clustered environments with shared database schemas and Python-implemented tools (Section 2.1; Figure 1).
   - Why it matters: Enables large‑scale, low‑latency, fully simulated interaction with precise verification at both state and tool‑sequence levels (Section 2.2). This is a step change from ad‑hoc, manually assembled environments.

2) Graph‑centric domain discovery and task sampling
   - Novelty: Build a tool dependency graph from parameter embeddings (Equation (1)); cluster with Louvain; then sample coherent tool sequences by directed walks (Sections 2.1–2.2).
   - Significance: Systematically broadens scenario diversity while maintaining internal logic, improving coverage of function‑calling patterns beyond curated datasets.

3) Three‑stage trajectory filtering anchored in ground truth state
   - Innovation: The funnel—validity control → environment state alignment → exact match—balances realism (allowing some errors) with reliability (state‑based verification) (Section 3.1).
   - Impact: Produces high‑fidelity supervision signals without human annotation, critical for scalable training.

4) Two‑stage agent training: general → vertical
   - Distinctive design choice: First acquire generic tool‑use competence, then specialize for domain realism (Section 3.2; Figure 3).
   - Evidence: Ablation shows both stages improve performance; Stage 2 further boosts multi‑step “Agent” tasks in ACEBench‑en (Figure 3).

5) Strong results with compact models
   - Insight: A 4B model trained with this pipeline competes with or surpasses many 30B baselines on tool‑use benchmarks (Table 1), suggesting that high‑quality, verifiable experience can substitute for sheer parameter count.

Fundamental vs. incremental:
- Fundamental: the environment scaling and verifiable simulation framework (Sections 2–3).
- Incremental but effective: masking strategy in the loss (Equation (2)); Louvain clustering; exact‑match filtering. These choices are not entirely new individually but combine into a compelling end‑to‑end system.

## 5. Experimental Analysis
Evaluation setup (Section 4.1):
- Benchmarks and metrics
  - τ‑bench (retail, airline): report pass^1 (Figure/Table references throughout the paper).
  - τ²‑Bench (retail, airline, telecom): report pass^1 and pass^k (Figure 4).
  - ACEBench‑en: report Accuracy on Normal, Special, Agent, and Overall categories (Table 1).
- Baselines
  - Closed source: Gemini‑2.5‑pro, Claude‑Sonnet‑4, GPT‑o3, GPT‑o4‑mini, GPT‑5‑think.
  - Open source: GPT‑OSS‑120B‑A5B, DeepSeek‑V3.1‑671B‑A37B, Kimi‑K2‑1T‑A32B, Qwen3‑Thinking‑235B‑A22B, Seed‑OSS‑36B, Qwen‑Coder‑30B‑A3B, xLAM‑2 variants, and Qwen3 baselines (Table 1).
- AgentScaler models
  - 4B, 8B, 30B‑A3B trained on Qwen3 series (Section 4.1 “Backbones”).

Main quantitative findings (Table 1)
- τ‑bench (pass^1)
  - `AgentScaler‑30B‑A3B`: Retail 70.4; Airline 54.0.
  - `AgentScaler‑4B`: Retail 64.3; Airline 54.0 (notable for a small model).
- τ²‑Bench (pass^1)
  - `AgentScaler‑30B‑A3B`: Retail 70.2; Airline 60.0; Telecom 55.3.
- ACEBench‑en (Accuracy)
  - `AgentScaler‑30B‑A3B`: Normal 76.7; Special 82.7; Agent 60.0; Overall 75.7.
  - `AgentScaler‑4B`: Normal 70.3; Special 76.7; Agent 30.8; Overall 65.9.
- Baseline comparisons from Table 1
  - Against strong open-source Kimi‑K2‑1T‑A32B (≈1T parameters), `AgentScaler‑30B‑A3B` is comparable:
    - τ‑bench Retail 70.4 vs 73.9; Airline 54.0 vs 51.2.
    - τ²‑Bench Retail 70.2 vs 70.6; Airline 60.0 vs 56.5; Telecom 55.3 vs 65.8 (Kimi stronger on Telecom).
    - ACEBench‑en Overall 75.7 vs 77.4 (close).
  - Against base Qwen3 models of similar size:
    - `AgentScaler‑4B` (Overall 65.9) greatly improves over Qwen3‑Thinking‑4B (49.5) on ACEBench‑en.
    - `AgentScaler‑30B‑A3B` (Overall 75.7) vs Qwen3‑Thinking‑30B‑A3B (67.2) shows a large gain.

Summary claim supported by Table 1:
> Across τ‑bench, τ²‑Bench, and ACEBench‑en, AgentScaler models at each comparable scale (4B, 8B, 30B‑A3B) outperform their base Qwen3 counterparts and are competitive with or better than most open‑source models below 1T parameters.

Ablation on two‑stage training (Figure 3)
- Stage 1 and Stage 2 both improve ACEBench‑en on Normal, Agent, and Overall subsets.
- Multi‑step agent tasks (the “Agent” subset) benefit especially from Stage 2’s domain specialization.

Stability via pass^k (Figure 4)
- On τ²‑Bench, `AgentScaler‑30B‑A3B` has higher weighted overall scores than Qwen3‑Thinking‑30B‑A3B across all k.
  - Example from “Weighted Overall” plot: pass^1 ≈ 62.5 vs 45.3; pass^4 ≈ 30.6 vs 27.7.
- Scores decrease as k increases (both models), highlighting remaining instability in LLM tool use:
> “a clear downward trend in scores is observed as k increases” (Figure 4 commentary, Section 5).

Out‑of‑distribution robustness: ACEBench‑zh (Table 2)
- `AgentScaler‑30B‑A3B` achieves Overall 81.5 (+7.3 over Qwen3‑Thinking‑30B‑A3B).
- Large gains for smaller models:
  - `AgentScaler‑4B` Overall 65.6 (+21.7); “Agent” 38.4 (+31.7).
- Mixed results on the Special subset (e.g., −3.4 for 30B), suggesting domain nuances.

Long‑horizon tool‑calling difficulty (Figure 5)
- Accuracy declines as the number of tool calls increases in τ‑bench (both retail and airline).
- The trend lines show a negative correlation between tool‑call count and task accuracy, indicating remaining challenges in extended tool chains.

Do the experiments support the claims?
- The breadth (three benchmarks; English and Chinese), explicit SOTA comparisons for sub‑1T models (Table 1), ablations (Figure 3), stability analysis (Figure 4), and long‑horizon diagnosis (Figure 5) provide a convincing empirical case that the pipeline improves function‑calling competence and stability, especially for compact models.

## 6. Limitations and Trade-offs
Assumptions and scope
- Simulated environments are adequate proxies for real APIs
  - Tools are grounded in a database abstraction. Many real APIs have side effects and nondeterminism not captured in a read–write DB (e.g., network latency, asynchronous jobs, quotas). This can limit transfer to deployment settings.
- Determinism and verifiability
  - The validation relies on deterministic state changes and exact matching (Section 3.1). This may bias training toward a single “gold” sequence, even when multiple valid tool plans exist.

Scenarios not fully addressed
- Long-horizon planning remains weak (Figure 5).
- Complex cross‑domain tasks that require switching among multiple domains/tools with nontrivial dependencies may exceed the directed-walk sampler’s coverage.

Computational and data considerations
- Building the tool graph and refining edges with an LLM across >30k tools and >1k domains can be compute‑intensive (Section 2.1). The paper does not report the cost/time of environment construction.
- Simulated user quality: outcomes depend on how well the simulated user reflects realistic intents and feedback; details of user simulation are not deeply quantified (Figure 2 provides the conceptual flow).

Methodological trade‑offs
- Filtering keeps some trajectories with intermediate errors to improve robustness (Section 3.1). While sensible, it can introduce noisy supervision if the final success masks brittle behavior.
- The masked loss (Equation (2)) does not directly optimize tool‑response interpretation quality; it relies on conditioning on tool outputs without supervising them, which may limit learning to detect tool-response anomalies.

Model scale and learning paradigms
- Current validated upper bound is 30B parameters (Limitations). Effects at 200B+ or trillion‑scale are untested.
- No reinforcement learning (RL) yet, despite the environment being “RL‑ready” (Limitations).

## 7. Implications and Future Directions
How this work changes the landscape
- It reframes agent training around scalable, verifiable environment construction rather than around rare real‑world logs or fragile LLM‑simulated tool responses.
- It demonstrates that compact models (4B–30B) can acquire strong tool‑use competence when trained on high‑fidelity simulated experiences, potentially lowering the barrier for on‑device or latency‑sensitive applications.

What it enables next
- RL on top of simulated environments
  - The pipeline’s determinism and low latency make it ideal for policy-gradient style training, preference learning for tool plans, or curriculum learning over tool-graph difficulty (Limitations; Conclusion).
- Tackling long‑horizon tool use
  - Incorporate planning modules, hierarchical policies, or search over tool graphs; augment the sampler with constraints to encourage diverse yet optimal plans (Figure 5 diagnosis).
- Multi-modality and real‑world deployment
  - Extend schemas and tools to cover vision, speech, and sensor streams; progressively “swap in” real MCP servers for subsets of tools to bridge sim‑to‑real.

Practical applications
- Enterprise operations: customer support, order management, and scheduling (τ‑bench domains) with verifiable action traces.
- Agent evaluation and data generation platforms: a drop‑in environment builder to synthesize diverse, testable tool-use tasks for training/evaluation at scale.
- Safety and reliability: state‑based verification and exact‑match filters can be adapted as guardrails and auditing tools for production agents.

Representative supporting quotes and pointers
- On environment abstraction and verification:
  > “each function … can be assigned an operator type … over the database D … enabling verifiability at both the environment level and the tool‑argument response level.” (Section 2; Figure 1)
- On the graph-based domain scaling:
  > Equation (1) defines edges via parameter‑similarity; domains come from Louvain community detection with an LLM refinement pass. (Section 2.1)
- On training supervision:
  > Equation (2) masks human text and tool responses, optimizing only assistant tool calls and replies while conditioning on the rest. (Section 3.2)
- On main results:
  > Table 1: `AgentScaler‑30B‑A3B` achieves ACEBench‑en Overall 75.7 and τ²‑Bench Retail/Airline/Telecom 70.2/60.0/55.3, competitive with much larger open models.
- On stability and long‑horizon challenges:
  > Figure 4 shows `pass^k` declines with k but AgentScaler is consistently above its Qwen3 base; Figure 5 shows accuracy decreases as tool‑call counts increase.

In sum, AgentScaler delivers a principled, scalable route to train and evaluate tool‑using agents by turning the “environment” into a first‑class, verifiable object. The combination of environment scaling, trajectory filtering anchored in state verification, and two‑stage training yields strong empirical gains, especially for smaller models, and opens the door to RL, longer‑horizon planning, and sim‑to‑real deployment pipelines.
