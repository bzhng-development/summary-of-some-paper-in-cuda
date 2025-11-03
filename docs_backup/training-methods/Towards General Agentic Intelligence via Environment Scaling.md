# Towards General Agentic Intelligence via Environment Scaling

**ArXiv:** [2509.13311](https://arxiv.org/abs/2509.13311)
**Authors:** Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, Xinyu Wang, Xiaobin Wang, Liangcai Su, Zhen Zhang, Shibin Wu, Zhengwei Tao, Yong Jiang, Pengjun Xie, Fei Huang, Jingren Zhou
**Institutions:** 

## 🎯 Pitch

AgentScaler revolutionizes tool-use training in language models by converting tool calls into verifiable, stateful database operations across diverse environments. This innovation propels open-source models to achieve state-of-the-art results, ensuring more robust, reliable function calls crucial for deploying agile AI agents across varied applications.

---

## 1. Executive Summary
This paper proposes a scalable way to train language-model agents to use external tools (“function calling”) by automatically constructing many verifiable, fully simulated environments and then fine-tuning agents in two phases on experience collected in those environments. The result, the `AgentScaler` family (4B/8B/30B parameters), achieves state-of-the-art results among open-source models under 1T parameters on multiple agentic benchmarks and improves stability and generalization, while revealing remaining challenges for long tool-use chains.

## 2. Context and Motivation
- Problem and gap
  - Tool use (invoking APIs, databases, or services from a model) is essential for real-world agents but current progress is constrained by a lack of high-quality “agentic data”—multi-turn trajectories where an agent actually executes tool calls with arguments and observes tool responses (Section 1).
  - Existing synthetic-data approaches either:
    - “Reverse” generation: create user queries to match given tool calls; realistic flow is limited and often unnatural (Section 1; Yin et al., 2025).
    - “Forward” simulated agent–human interplay: better multi-turn realism but environments are not scalable and typically require manual environment construction (Section 1).
  - Using real tools is costly and brittle; LLM-simulated tool responses hallucinate; offline mock environments have mostly been used for evaluation, not scalable training (Related Work, Section 6.1).

- Why this matters
  - Practical deployment of LLM agents depends on robust, precise function calling across very diverse APIs (Abstract; Section 1).
  - The breadth of tool-using ability appears tied to the breadth/diversity of environments an agent learns in, but building such environments at scale with reliable supervision has been hard (Abstract; Sections 1–2).

- Positioning vs prior work
  - The paper shifts from “more data” to “more environments.” It treats any tool call as a read–write operation on a domain-specific stateful database, then automatically builds many such domains, instruments tools as executable code, and verifies outcomes via state changes (Section 2; Figure 1).
  - It combines this with a two-phase training pipeline: general tool-use learning followed by domain specialization (Abstract; Section 3.2), aiming to deliver general agentic intelligence rather than a narrow set of tool scripts.

## 3. Technical Approach
The paper’s pipeline has two major stages (Abstract; Figure 1): (A) environment construction and scaling, and (B) agent experience learning.

A. Environment construction and scaling (Section 2; Figure 1)
1) Core abstraction: environments as read–write databases
   - Treat each tool/function `func` as an operator on an underlying database `D`:
     - Read tools query `D`; write tools change `D` (Section 2).
     - Tool execution is formalized as `API(func, α) ≡ op(func)(α; D)`, where `α` is the argument set.
   - Tools are grouped into “domains” `T1 … TM`. Each domain shares a database schema `Sk`—a specification of the state and its structure that tools in that domain read/write (Section 2).

2) Scenario collection (Section 2.1)
   - Aggregate >30,000 APIs from ToolBench, API-Gen, and internal sources.
   - Clean and normalize API specs, rewriting descriptions to include explicit input–output signatures; construct compositional tool chains using input–output relations, yielding an API pool `ΘF` (>30k tools).

3) Tool dependency graph modeling (Section 2.1; Eq. 1)
   - Build a graph where nodes are tools; edges indicate that outputs/parameters of one tool can feed another.
   - For any pair of tools `i, j`, embed their parameter lists `ϕ(P_i)` and `ϕ(P_j)`; if cosine similarity exceeds a threshold `τ`, add an edge:
     > Eq. (1): `E = { (i, j) | sim(ϕ(P_i), ϕ(P_j)) > τ, i ≠ j }`
   - Partition the graph into domains via Louvain community detection (a fast community-finding algorithm).
   - Refine edges within each domain using an LLM to check pairwise dependencies more precisely (Section 2.1). Outcome: M > 1,000 domains.

4) Function schema programmatic materialization (Section 2.1)
   - From all tools in a domain, synthesize a domain-specific database schema (the environment state).
   - Implement each tool as executable Python code that reads/writes the domain state according to its schema.
   - Manual spot-checks within τ-bench domains showed high consistency with official τ-bench implementations (Section 2.1).

5) Agentic task construction (Section 2.2; Figure 1)
   - Initialize the environment state stochastically per domain to maximize diversity.
   - Sample a coherent tool sequence by traversing a directed dependency graph:
     - Start at a random tool; walk until reaching a max length or a node with no outgoing edges (Section 2.2).
   - For each step, generate arguments, execute the tool on the actual stateful environment, and track the evolving database state.
   - Verifiability is twofold (Section 2.2):
     - Database-level: final state consistency with the “gold” state produced by the sampled sequence.
     - Tool-level: exact match of the tool sequence and arguments.

B. Agent experience learning (Section 3; Figure 2)
1) Collecting trajectories via simulated user–agent interplay (Section 3.1; Figure 2)
   - Given a synthesized “user intent,” spin up:
     - A simulated user that pursues the intent.
     - A task agent that must use domain tools to solve it.
     - The stateful environment from stage A.
   - Run the conversation to completion; record the full trace, including tool calls, tool outputs, and final state.

2) Rigorous trajectory filtering (Section 3.1)
   - Three-stage funnel:
     - Validity control: enforce well-formed alternating turns; remove traces with repetitive “stuck” reasoning via n-gram filtering.
     - Environment state alignment: keep only trajectories whose final database state equals the gold state (verifies write operations).
     - Function-calling exact match: for read-only sequences (where state checks are uninformative), require exact match in tool sequence and arguments.
   - Tool-call errors are not automatically discarded if the trajectory still reaches the goal; this encourages robustness (Section 3.1).

3) Training objective and two-phase schedule (Section 3.2; Eq. 2)
   - Inputs: human turns `h_t`, agent turns `a_t` with three parts: tool-call tokens `τ_t`, tool-response tokens `ρ_t`, and natural-language reply `y_t`.
   - Supervise only tool-call tokens `τ` and the assistant replies `y`; mask human inputs `h` and tool responses `ρ` from loss but keep them visible as context (Section 3.2).
     > Eq. (2): minimize negative log-likelihood only on tokens in set `T` = {tool calls, assistant replies}; all other tokens are context-only.
   - Two-phase experience learning (Section 3.2; Figure 3):
     - Stage 1 (general): diverse domains; learn “when/how to call tools” and how to integrate tool outputs into replies.
     - Stage 2 (vertical): fine-tune in specific target domains; sharpen tool selection and argument generation to domain norms.

Implementation and models (Section 4.1)
- Backbones: Qwen3 family.
- Trained models: `AgentScaler-4B` (Qwen3-Thinking-4B), `AgentScaler-8B` (Qwen3-8B), `AgentScaler-30B-A3B` (Qwen3-Thinking-30B-A3B).

## 4. Key Insights and Innovations
- Scalable, verifiable environment construction from large tool corpora
  - Novelty: treat tools as read/write operators on domain states and automatically derive domain schemas and executable tool code grounded in those schemas (Section 2.1; Figure 1).
  - Why it matters: enables massive, low-latency, verifiable simulation for training—not just evaluation (cf. τ-bench/τ²-bench), overcoming bottlenecks of cost/brittleness (real APIs) and hallucinations (LLM-simulated tools) (Related Work, Section 6.1).

- Graph-based domain partitioning + LLM-refined dependencies
  - Novelty: build a tool dependency graph from >30k APIs using parameter-similarity edges (Eq. 1), then partition with Louvain and refine edges via LLM checks (Section 2.1).
  - Benefit: yields >1,000 coherent domains with structured tool composability; improves the realism and correctness of sampled tool sequences (Section 2.1–2.2).

- Two-level verifiability and a strict filtering funnel
  - Novelty: supervise with both database-state alignment and tool-sequence/argument exact match; additionally retain some errorful traces that still succeed to encourage robustness (Section 3.1).
  - Significance: produces higher-fidelity agentic trajectories than prior forward-simulation methods that lack automated, state-grounded checks.

- Targeted supervision: optimize only what the agent controls
  - Novelty: loss function masks human turns and tool responses, training only on agent-generated tool calls and replies (Eq. 2).
  - Rationale: focuses learning signal on decision points the agent must master (when/which tool, with what arguments, and what to say next).

- Two-phase training curriculum (general → vertical)
  - Contribution: empirically validated schedule that first builds broad competence, then tunes for domain idiosyncrasies (Section 3.2; Figure 3).
  - Impact: better performance and improved stability vs base models at the same scale (Sections 4–5; Figure 4).

Collectively, these are more than incremental tweaks: the environment-as-database view plus programmatic materialization, verifiable simulation, and targeted supervision form a coherent, scalable training methodology for tool-use agents.

## 5. Experimental Analysis
- Benchmarks, metrics, and setup (Section 4.1)
  - τ-bench (retail, airline) and τ²-Bench (retail, airline, telecom): report `pass^1` and analyze `pass^k` (k=1..4), where a sample counts only if all k repeated runs are correct; this measures both accuracy and stability (Section 4.1; Figure 4).
  - ACEBench-en: Normal, Special, Agent categories; accuracy metric and Overall score (Section 4.1; Table 1).
  - Out-of-distribution check: ACEBench-zh (Table 2).
  - Baselines: strong closed-source systems (e.g., Gemini-2.5-pro, Claude-Sonnet-4, GPT-o3, GPT-o4-mini, GPT-5-think) and open-source models up to 1T parameters (e.g., Kimi-K2-1T-A32B, DeepSeek-V3.1-671B-A37B, Qwen3-Thinking-235B-A22B, xLAM-2 variants) (Table 1).

- Main quantitative results (Table 1)
  - Closed-source models still lead overall, but `AgentScaler` is highly competitive at much smaller scales.
  - Highlights:
    - `AgentScaler-30B-A3B`:
      > τ-bench: Retail 70.4, Airline 54.0  
      > τ²-Bench: Retail 70.2, Airline 60.0, Telecom 55.3  
      > ACEBench-en: Normal 76.7, Special 82.7, Agent 60.0, Overall 75.7
    - `AgentScaler-4B` achieves strong small-model performance:
      > τ-bench: Retail 64.3, Airline 54.0; τ²-Bench: Retail 62.3, Airline 56.0, Telecom 48.2; ACEBench-en Overall 65.9
    - Compared to same-scale backbones:
      - `AgentScaler-30B-A3B` vs `Qwen3-Thinking-30B-A3B`: improves across most settings; e.g., τ²-Bench weighted stability (Figure 4) and ACEBench-en Overall (75.7 vs 67.2).
      - `AgentScaler-4B` vs `Qwen3-Thinking-4B`: marked gains on τ²-Bench and ACEBench-en Overall (65.9 vs 49.5).
  - Against large open-source models:
    - Performance is often comparable and sometimes better at far smaller parameter counts. Example from Table 1: on τ²-Bench Airline, `AgentScaler-30B-A3B` scores 60.0 vs `Kimi-K2-1T-A32B` 56.5; on Telecom, `Kimi-K2-1T-A32B` is higher (65.8) than `AgentScaler-30B-A3B` (55.3). This supports “competitive with much larger models,” but not uniformly superior.

- Two-stage training ablation (Figure 3)
  - Figure 3 shows Stage 1 and Stage 2 both improve over the base `Qwen3-Thinking-30B-A3B` on ACEBench-en Normal, Agent, and Overall. Multi-step Stage 2 adds further gains on the Agent subset. This supports the curriculum’s value.

- Stability via pass^k (Figure 4)
  - Across τ²-Bench domains, `AgentScaler-30B-A3B` consistently exceeds the base model at all k. Example (Weighted Overall, Figure 4):
    > Ours: 62.5 (k=1), 48.6 (k=2), 38.5 (k=3), 30.6 (k=4)  
    > Qwen3-30B-A3B: 45.3, 34.1, 30.9, 27.7
  - Both models’ scores drop as k increases, indicating consistency remains challenging for LLM agents.

- Long-horizon tool calling (Figure 5)
  - Scatter plot on τ-bench shows a clear negative correlation between number of tool calls and accuracy in both Retail and Airline domains: as the action chain length grows, success rates fall. The dashed/dotted trend lines quantify this decline (Figure 5). This is a key remaining limitation.

- Out-of-distribution generalization: ACEBench-zh (Table 2)
  - `AgentScaler` improves Overall across scales:
    > 4B: 43.9 → 65.6 (+21.7)  
    > 8B: 71.3 → 73.7 (+2.4)  
    > 30B-A3B: 74.2 → 81.5 (+7.3)
  - Notably, 4B gains strongly on the Agent subset (6.7 → 38.4, +31.7) but drops on Special (85.3 → 70.0), showing trade-offs (Table 2).

- Do the experiments support the claims?
  - Yes for core claims: the pipeline consistently improves tool-use accuracy and stability over same-scale backbones; delivers strong results vs much larger open-source models; and exhibits OOD gains (Tables 1–2; Figures 3–4).
  - Caveats: closed-source leaders still outperform in several settings (Table 1), and long-horizon chains remain problematic (Figure 5).

## 6. Limitations and Trade-offs
- Assumptions and design choices
  - Tools can be abstracted as read/write operators over a domain-specific database (`D`, `Sk`). Real APIs with side effects beyond structured state (e.g., asynchronous services, non-determinism) may not fit neatly (Section 2).
  - Tool dependency edges are initialized by parameter-list embedding similarity (Eq. 1), then refined by an LLM. Both steps can misclassify dependencies, affecting domain partitions and sampled sequences (Section 2.1).

- Simulation–reality gap
  - Environments are simulated; although they execute code against a real state, they do not call live services. Distribution shift may appear when deploying against real APIs (Related Work 6.1). The state-based verification ensures internal consistency, not external service fidelity.

- Long-horizon performance
  - Accuracy declines with more tool calls (Figure 5). The approach does not yet include specific mechanisms (planning, error recovery, or hierarchical control) to stabilize long chains.

- Training regime
  - Current work uses supervised fine-tuning only; no reinforcement learning (RL) on the simulated environments, though the setup is amenable to it (Limitation section). RL might help with exploration, credit assignment across long chains, and robustness.

- Scale and resources
  - Results are shown up to 30B parameters; scaling beyond 200B or 1T is not explored here (Limitation section). While promising at small/medium scales, it is unclear how the approach interacts with very large backbones.

- Category trade-offs
  - On ACEBench-zh, improvements on Normal/Agent sometimes accompany a drop on Special for small models (Table 2), suggesting that gains may depend on the target category/domain mix and data balance.

## 7. Implications and Future Directions
- How this changes the field
  - Establishes a practical path to train tool-using agents by scaling environments, not just instruction data. The environment-as-database abstraction, plus verifiable state changes, makes large-scale, reliable agentic data generation feasible for training, not only evaluation (Sections 2–3; Figures 1–2).

- Follow-up research enabled
  - Reinforcement learning on top of the simulated, verifiable environments to improve planning, recovery from tool errors, and long-horizon performance (Limitation; Section 7).
  - Better dependency modeling: beyond parameter similarity—learned causal graphs over tools, or execution-level dependency discovery.
  - Curriculum and planning strategies that explicitly target long chains (e.g., hierarchical policies, option discovery, or subgoal validation).
  - Extending to multimodal tools and sensors/actuators where state includes images, audio, or real-world signals (Conclusion; Section 7).

- Practical applications
  - Building enterprise assistants that must integrate with many heterogeneous APIs (CRM, order management, airline, telecom), with predictable behavior and improved stability (Table 1; τ-bench/τ²-bench domains).
  - Rapid prototyping of domain-specific agents by auto-constructing domains from existing API catalogs and fine-tuning in Stage 2.
  - Safer deployment pipelines: state-grounded verification during data generation can feed continuous training/evaluation loops for production agents.

In short, this paper introduces a scalable, verifiable way to generate the “experience” that tool-using agents need, and demonstrates that a targeted training objective plus a two-phase curriculum can turn medium and even small models into capable and more stable tool callers. The approach advances the practice of training general agentic intelligence while making the remaining challenges—especially long-horizon robustness—more visible and tractable.
