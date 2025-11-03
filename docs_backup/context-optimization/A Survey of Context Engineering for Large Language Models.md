# A Survey of Context Engineering for Large Language Models

**ArXiv:** [2507.13334](https://arxiv.org/abs/2507.13334)
**Authors:** Lingrui Mei, Jiayu Yao, Yuyao Ge, Yiwei Wang, Baolong Bi, Yujun Cai, Jiazhi Liu, Mingyu Li, Zhong‑Zhi Li, Duzhen Zhang, Chenlin Zhou, Jiayi Mao, Tianze Xia, Jiafeng Guo, Shenghua Liu
**Institutions:** 

## 🎯 Pitch

This paper introduces Context Engineering, transforming LLM interaction into a systematic optimization and systems discipline through mathematical formalization and a comprehensive taxonomy. By addressing the critical comprehension-generation asymmetry, it not only unifies previously fragmented research but also guides the development of efficient, robust AI applications that can dynamically assemble and process complex contexts, ensuring more reliable and context-aware outputs. This innovation promises significant impacts in fields where AI's memory and decision-making capabilities are crucial, such as healthcare and finance.

---

## 1. Executive Summary (2–3 sentences)
This survey defines Context Engineering as the systematic design, selection, assembly, and optimization of all information fed to a Large Language Model (LLM) at inference time, and proposes a unified taxonomy that links foundational components (retrieval/generation, processing, management) to system implementations (RAG, memory systems, tool-integrated reasoning, multi-agent systems). It also formalizes Context Engineering mathematically (Eqs. 1–6), maps the field’s evolution (Figures 1–2), and highlights a critical performance gap: today’s models understand complex, engineered contexts better than they can generate long, coherent, reliable outputs of comparable sophistication (Abstract; §7.1.2).

## 2. Context and Motivation
- Problem addressed:
  - LLM behavior at inference depends heavily on the provided context, yet research and practice have been fragmented—focused on “prompt engineering” or individual subsystems like RAG, tool use, or memory in isolation. The paper addresses the lack of a formal, system-level discipline and unified map connecting these parts (§1–§2).
  - It identifies a core research gap: strong gains in contextual understanding have not been matched by equally strong gains in long-form, reliable generation (“comprehension–generation asymmetry”) (Abstract; §7.1.2).

- Why it matters:
  - Practical stakes: production assistants, research agents, code assistants, and decision-support systems need dynamic access to external knowledge, long-term memory, and tools, all orchestrated efficiently; poor orchestration leads to latency, cost, hallucinations, and brittle behavior (§3.2.1–§3.2.3; §5–§7).
  - Theoretical stakes: without a formal problem definition and taxonomy, it is hard to reason about optimal context composition, scaling laws, or information-theoretic limits (§3.1; §7.1.1).

- Prior approaches and gaps:
  - Prompt engineering: largely a manual or heuristic search over strings, often brittle and task-specific (§3.1, Table 1).
  - Isolated subsystems: RAG mitigates hallucination but struggles with timing and retrieval quality; memory systems are ad hoc and hard to evaluate; tool calling adds capabilities but is difficult to orchestrate; multi-agent systems lack transactional integrity and shared standards (§5; §6.3).
  - Long-context modeling is constrained by O(n²) attention cost, “lost in the middle” retrieval bias, and KV-cache memory pressure (§4.2.1; §4.3.1).

- Positioning:
  - The survey reframes the field as Context Engineering, introduces a formal optimization view (Eqs. 2–6), and provides a comprehensive taxonomy tying foundational components to system implementations (Figure 1), plus an evolution timeline (Figure 2). It synthesizes results across >1,400 papers to surface unifying principles, best practices, bottlenecks, and open problems.

## 3. Technical Approach
This is a methodological survey that contributes both a formal problem framing and a system taxonomy, then analyzes techniques/components in-depth with references to mechanisms, constraints, and measured effects.

- Formalization of Context Engineering (§3.1; Eqs. 1–6):
  - Goal in plain language:
    - Treat the input context not as a single prompt, but as a structured, dynamically assembled package of heterogeneous components: instructions (`cinstr`), retrieved knowledge (`cknow`), tool interfaces (`ctools`), episodic memory (`cmem`), dynamic system state (`cstate`), and the user query (`cquery`).
  - Assembly function:
    - The overall context is built by an assembly function `A` that formats, orders, and concatenates components with templates and policies (Eq. 2: `C = A(c1,…,cn)`; Figure 3’s top strip shows these subcomponents).
  - Optimization problem:
    - Choose the set of context-generating functions `F = {A, Retrieve, Select, …}` to maximize an expected task reward subject to a hard context-length constraint `|C| ≤ Lmax` (Eq. 3). This reframes design choices (retrieval thresholds, summarization, selection, ordering) as an explicit optimization.
  - Information-theoretic retrieval:
    - Retrieval can be cast as maximizing the mutual information between retrieved knowledge and the task answer `I(Y*; cknow | cquery)` (Eq. 4), steering away from “semantic similarity only” toward “maximally informative evidence.”
  - Bayesian context inference:
    - Context selection can also be viewed as inferring `P(C | cquery, History, World)` and maximizing expected reward over possible answers given the chosen context (Eq. 6). This motivates adaptive, uncertainty-aware retrieval and memory updates.

- Taxonomy and end-to-end pipeline (Figures 1 & 3; §4–§5):
  - Foundational Components (§4):
    - Context Retrieval & Generation (§4.1): prompt and reasoning frameworks (e.g., CoT, ToT, GoT), external retrieval (RAG, KG-based, agentic retrieval), and dynamic assembly (templates, priority selection, automated prompt optimization).
    - Context Processing (§4.2): long-sequence mechanisms (LongNet, FlashAttention, Ring Attention, GQA, StreamingLLM, Infini-attention), self-refinement loops (Self-Refine, Reflexion, N-CRITICS; Table 2), multi-modal fusion, and structured/graph reasoning (GraphFormers, StructGPT).
    - Context Management (§4.3): memory hierarchies (MemGPT’s OS-like paging), KV-cache policies (H2O), compression (ICAE; RCC), and storage architectures (hierarchical caches, consolidated memories).
  - Implementations (§5):
    - RAG (§5.1; Figure 4): modular, agentic, and graph-enhanced RAG patterns.
    - Memory Systems (§5.2; Figure 5): sensory/short-/long-term memory classifications, read–write memory integration, memory-augmented agents, evaluation pitfalls.
    - Tool-Integrated Reasoning (§5.3; Figure 6): function calling, reasoning intertwined with external tools, and agent–environment interaction.
    - Multi-Agent Systems (§5.4; Figure 7): communication protocols (KQML, FIPA ACL, MCP/A2A/ACP/ANP), orchestration (a priori vs posterior; 3S orchestrator; serialized/puppeteer), and coordination strategies.

- Design choices and why:
  - Move from prompt-only to assembly `A`: because different component types (retrieved chunks, tool specs, memories) compete for limited tokens and must be prioritized and formatted to align with LLM inductive biases (§3.1; §4.1.3).
  - Information-theoretic retrieval vs similarity: similarity can bring on-topic but unhelpful text; maximizing mutual information targets evidence most predictive for the answer (Eq. 4; §4.1.2).
  - Dual “context scaling” axes (§3.1, “Context Scaling”):
    - Length scaling: address cost (O(n²) attention), cache pressure, and positional extrapolation (FlashAttention, LongNet, LongRoPE).
    - Multi-modal/structural scaling: integrate temporal, spatial, participant, intentional, cultural, and graph-structured context; because real applications need more than text (§4.2.3–§4.2.4).
  - OS-like memory plus retrieval: fixed context windows require paging, summarization, and recall policies; external, structured memories mitigate “lost-in-the-middle” and support persistence (§4.3.2; §4.3.1).

- Example to ground intuition:
  - Think of the LLM’s prompt as your carry‑on bag with a strict weight limit (`Lmax`). Context Engineering is the logistics operation that: (1) selects the most informative items (retrieval with Eq. 4), (2) compresses clothing (summarization/compression), (3) packs with a checklist (assembly `A`), and (4) uses a locker nearby for overflow (external memory with paging), revisiting the locker if needed (agentic retrieval/planning).

## 4. Key Insights and Innovations
- A formal, optimization-based definition of Context Engineering (§3.1; Eqs. 2–6; Table 1):
  - What’s new: breaks from “prompt as a string” to “context as an optimized, structured assembly,” with explicit objectives and constraints. This enables principled analysis of retrieval timing, content selection, and ordering.
  - Why it matters: unlocks information-theoretic and Bayesian tools to reason about what to include, how to compress, and how to adapt context online.

- A unified taxonomy linking components to systems (Figures 1 & 3; §§4–5):
  - What’s new: a single map that shows how prompt/knowledge generation, long-context processing, and memory management compose into RAG, memory-augmented agents, tool-integrated reasoning, and multi-agent systems.
  - Why it matters: reduces fragmentation, showing, for example, that RAG is a form of external memory, tool use is a context acquisition method, and orchestration is context assembly across agents (§2, “Our Contribution”).

- Two-axis “Context Scaling” model (§3.1, end):
  - What’s new: separates length scaling from multi-modal/structural scaling (temporal, spatial, participant, intentional, cultural, graphs).
  - Why it matters: guides research beyond “more tokens” toward richer, structured context that human-like tasks require.

- Diagnosis of the comprehension–generation asymmetry (Abstract; §7.1.2; §6.3.1 with evidence from GAIA, GTA, LongMemEval):
  - What’s new: a field-level finding—the ability to parse and use engineered context is outpacing the ability to generate long, faithful, self-consistent outputs at the same level.
  - Why it matters: reorients priorities toward long-form planning, factual persistence, and transactional integrity in outputs (e.g., tool-based or multi-agent workflows).

- Systemic evaluation challenges and emerging standards (§6; Table 8; §5.4.1):
  - What’s new: identifies missing transactional guarantees in orchestration (§6.3.1, §5.4.3), memory evaluation isolation problems (§6.3.1), and pushes protocol standardization via MCP/A2A/ACP/ANP (§5.4.1).
  - Why it matters: consistent interfaces and robust evaluation are prerequisites for dependable, composable systems.

## 5. Experimental Analysis
Although this is a survey, it aggregates quantitative results and evaluation setups across hundreds of studies. Key measurements and what they show:

- Evaluation methodology and benchmarks (§6):
  - Component-level tests:
    - “Needle in a haystack” for long-context retrieval; chain-of-thought gains; structural reasoning (SRL/Table-to-SQL) sensitivity (§6.1.1).
  - System-level tests:
    - RAG: retrieval precision/recall and factual accuracy with modular pipelines; Agentic RAG adds plan quality and reflection effectiveness (§6.1.2).
    - Memory systems: LongMemEval (500 Qs) measures information extraction, temporal & multi-session reasoning, knowledge updates, abstention; commercial assistants degrade by ~30% over prolonged interactions (§6.1.2; also §5.2.3).
    - Tool-integrated reasoning: BFCL (2,000 cases), T‑Eval (553), API‑Bank (73 APIs, 314 dialogues), ToolHop (995 queries, 3,912 tools) (§6.1.2).
    - Web agents: WebArena/Mind2Web/VideoWebArena; Table 8 lists top systems (e.g., IBM CUGA 61.7% success; OpenAI Operator 58.1%; several open-source models in the 26–52% range).

- Main quantitative outcomes (selected highlights with locations):
  - Reasoning prompts:
    - Zero‑shot CoT boosts MultiArith accuracy from 17.7% to 78.7% (§4.1.1).
    - Tree‑of‑Thoughts raises Game of 24 success 4% → 74% (§4.1.1).
    - Graph‑of‑Thoughts improves quality by 62% and cuts cost 31% vs ToT (§4.1.1).
  - Long‑context processing:
    - FlashAttention reduces memory scaling to linear and accelerates attention; FlashAttention‑2 roughly doubles speed vs FA‑1 (§4.2.1).
    - StreamingLLM achieves up to 22.2× speedup on sequences up to 4M tokens by retaining attention sink tokens and a recent KV slice (§4.2.1).
    - Heavy Hitter Oracle (H2O) improves throughput up to 29× and reduces latency up to 1.9× by evicting low‑impact KV entries (§4.2.1).
    - LongRoPE extends context windows to 2048K tokens via staged finetuning + interpolation; Self‑Extend enables long‑context use without finetuning through grouped + neighbor attention (§4.2.1).
  - RAG and retrieval:
    - Self‑RAG adds retrieval-on-demand with reflection tokens; modular toolkits like FlashRAG expose 5 modules/16 subcomponents for pipeline tuning (§4.1.2; §5.1.1).
    - Graph RAG systems (GraphRAG, LightRAG, HippoRAG) show improved multi‑hop QA and entity‑level retrieval via graphs and hierarchical indices (§5.1.3).
  - Memory:
    - OS-like paging (MemGPT) and forgetting‑curve update policies (MemoryBank) demonstrate persistent knowledge retention and improved retrieval relevance (§4.3.2; §5.2.2).
  - Tool use and agents:
    - ReTool’s RL on code‑interpreter usage reaches 67.0% on AIME2024 after only 400 steps vs ~40% for text‑only RL with extensive training (§5.3.3).
    - GAIA benchmark shows human accuracy ~92% vs advanced models ~15% on general assistant tasks—large headroom (§6.3.1).
    - GTA reveals real tool-use performance remains <50% for strong LLMs in complex real‑world tool orchestration (§5.3.3).

- Do results support the paper’s claims?
  - Evidence is consistent with the “comprehension–generation asymmetry”: strong input‑side gains (retrieval, long‑context, multi‑modal/graph integration) contrast with lower end‑to‑end task completion in open environments (GAIA/GTA/WebArena; §6.2.2, §6.3.1 and Table 8).
  - Long‑context and KV‑cache techniques show substantial efficiency wins (22× speedups, million‑token windows), yet evaluation reveals “lost in the middle” and position interpolation caveats (§4.2.1; §4.3.1; §6.2.1).
  - Memory evaluations (LongMemEval, Minerva; §5.2.3; §6.1.2) expose degradation over time and difficulty with episodic memory, supporting the call for better memory systems and protocols.

- Ablations, failure cases, robustness:
  - Long-context: bias toward beginning and end; middle information is under‑used (§4.3.1).
  - Tool use: over‑triggering or wrong-tool selection; benchmarks (StableToolBench, NesTools) specifically target instability and nesting failures (§5.3.1; §6.2.2).
  - Orchestration: lack of transactional integrity causes cascading failures; SagaLLM proposes mitigations (§5.4.3; §6.3.1).
  - Multimodal: modality bias toward text; visual grounding and fine-grained spatial/temporal reasoning are weak (§4.2.3, “Modality Bias and Reasoning Deficiencies”).

## 6. Limitations and Trade-offs
- Assumptions and constraints:
  - Context length is bounded; computation and GPU memory impose hard limits—vanilla attention is O(n²) (§4.2.1; §4.3.1).
  - Formal optimization (Eq. 3) assumes an available reward signal and a task distribution; in practice, reward design and offline estimates are noisy (§3.1).
  - Information-theoretic retrieval (Eq. 4) presumes access to or estimation of `I(Y*; cknow | cquery)`, which is nontrivial.

- Scenarios not fully addressed:
  - Transactional guarantees in multi-agent workflows; current frameworks often rely on LLM self-validation with no external verification (§5.4.3; §6.3.1).
  - Robust multimodal grounding under distribution shifts; vision/audio/text conflicts and temporal reasoning in long videos remain challenging (§4.2.3).
  - Graph reasoning at scale with dynamic updates; text-only verbalizations trade structural precision for interpretability (§4.2.4; §7.2.3).

- Computational/data trade-offs:
  - Extending windows via interpolation (LongRoPE, YaRN, PoSE) can extrapolate positions but may require further finetuning to preserve accuracy (§4.2.1).
  - Memory compression (ICAE, RCC) and cache eviction (H2O) accelerate inference but can discard rare yet crucial details (§4.2.1; §4.3.3).
  - Agentic RAG and tool use increase latency and token cost (planning, retries, reflection) unless carefully pruned or learned (§5.1.2; §5.3.2; §4.3.3, Table 5).

- Open questions:
  - How to derive principled “context budgets” across modalities and structures under token limits? (§7.1.1)
  - How to close the generation gap (long-form planning, factual persistence, self‑verification) without exploding cost? (§7.1.2; §7.2.2)
  - How to evaluate memory and orchestration with standardized, transaction-aware benchmarks? (§6.3)

## 7. Implications and Future Directions
- How this work changes the landscape:
  - Provides a shared language and optimization view (Eqs. 2–6) to design and analyze context pipelines end-to-end—retrieval, compression, memory, tool interfaces, and orchestration are now first-class, optimizable components rather than ad hoc add‑ons (Figures 1 & 3).
  - Elevates evaluation beyond static QA: emphasizes transactional integrity, multi-tool planning, longitudinal memory, and protocol‑based interoperability (§6; §5.4.1–§5.4.3).

- Follow-up research enabled/suggested (§7):
  - Theoretical foundations: information-theoretic bounds, optimal allocation across context slots, and compositional interaction models for components and agents (§7.1.1).
  - Efficient scaling: sliding/linear attention with strong reasoning preservation; state space models (e.g., Mamba/LongMamba) for long contexts (§7.1.2; §7.2.1).
  - Advanced reasoning & planning: integrate causal, counterfactual, temporal reasoning with tool use; address the comprehension–generation gap (§7.2.2).
  - Graph/structured context: hybrid neural–symbolic approaches, instruction-tuned graph reasoning, and editable memory graphs (§7.2.3; §5.1.3).
  - Intelligent context assembly: learned assembly `A`, multi‑objective selection (informativeness, diversity, cost), and self‑refining meta‑controllers (§7.2.4).
  - Large-scale multi-agent coordination: formal protocols (MCP/A2A/ACP/ANP), transaction-safe orchestration, anomaly detection, and shared external memories (§5.4.1–§5.4.3; §7.3.2).
  - Domain applications: healthcare, scientific research, legal/finance—where memory, verifiability, and safety are critical; benchmarks like Deep Research Bench and MEMENTO show needs and directions (§6.2.2; §7.3.1).

- Practical applications and use cases:
  - Enterprise assistants with persistent memory and secure tool use (function calling + RAG + MemOS) for IT ops, customer support, and data integration (§5.1–§5.3; §4.3.2).
  - Research agents blending web search, graph RAG, and code tools for literature synthesis and data analysis (Figure 4; §5.1.3; §5.3.2).
  - GUI/web agents for complex workflows (WebArena/Table 8; §5.3.3; §5.4), where transaction-aware orchestration and memory are decisive.
  - Safety-critical domains (healthcare/finance): context attribution, abstention, and independent validation become mandatory (§6.3.3; §7.4).

> Overall, Context Engineering reframes inference-time interaction with LLMs as an explicit optimization and systems problem—what to include, how to process it, where to store it, and how to orchestrate actions—supported by a comprehensive taxonomy (Figure 1), formal objectives (Eqs. 2–6), and converging evidence on both capabilities and gaps (e.g., Table 8; GAIA/GTA; LongMemEval).
