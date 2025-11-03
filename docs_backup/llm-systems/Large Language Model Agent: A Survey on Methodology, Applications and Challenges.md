# Large Language Model Agent: A Survey on Methodology, Applications and Challenges

**ArXiv:** [2503.21460](https://arxiv.org/abs/2503.21460)
**Authors:** Junyu Luo, Weizhi Zhang, Ye Yuan, Yusheng Zhao, Junwei Yang, Yiyang Gu, Bohan Wu, Binqi Chen, Ziyue Qiao, Qingqing Long, Rongcheng Tu, Xiao Luo, Wei Ju, Zhiping Xiao, Yifan Wang, Meng Xiao, Chenwu Liu, Jingyang Yuan, Shichang Zhang, Yiqiao Jin, Fan Zhang, Xian Wu, Hanqing Zhao, Dacheng Tao, Philip S. Yu, Ming Zhang
**Institutions:** 

## 🎯 Pitch

This paper introduces a unified framework for Large Language Model (LLM) agents, organized around constructing, collaborating, and evolving, thus creating a systematic lifecycle blueprint. By connecting fragmented research areas, it offers a comprehensive methodology essential for deploying robust, coordinated, and adaptive systems in high-stakes applications like healthcare and scientific discovery, ultimately advancing the field beyond ad hoc solutions.

---

## 1. Executive Summary
This survey proposes a unified, methodology-centered framework for understanding Large Language Model (LLM) agents, organized around three tightly connected dimensions: how agents are built (“Construction”), how they work together (“Collaboration”), and how they improve over time (“Evolution”). It complements this core taxonomy with coverage of evaluation benchmarks, development tools, real‑world security/privacy/ethics issues, and application domains, giving researchers and practitioners a coherent map of a fast‑moving field (Figures 1–2, Sections 2–5).

## 2. Context and Motivation
- Problem the paper addresses
  - Research on LLM agents has exploded, but contributions are fragmented across subtopics: profile/role design, memory, planning, tool use, multi‑agent cooperation, self‑learning, safety, evaluation, and applications. What’s missing is a coherent methodology that links these pieces into a lifecycle view of “how agents are constructed, collaborate, and evolve” (Figure 1; Sections 1 and 2).
- Why this matters
  - Real‑world deployment needs more than clever prompts. It requires robust architectures, collaboration protocols, learning/evolution mechanisms, evaluation standards, and safety practices. Without a unifying lens, it’s hard to compare systems, identify gaps, or transfer best practices to high‑stakes domains such as scientific discovery and healthcare (Sections 1, 3, 4, 5).
- Prior approaches and their gaps
  - Earlier surveys typically focus on narrow slices: gaming agents [11, 12], deployment environments [13, 14], multimodality [15], or security [16]. Broader overviews exist but lack a detailed methodological taxonomy that ties individual agent internals to multi-agent systems and their evolution (Section “Distinction from Previous Surveys” in 1).
- How this paper positions itself
  - It contributes a “Build–Collaborate–Evolve” taxonomy that deconstructs agents into fundamental components—profile, memory, planning, action—and connects these to collaboration styles (centralized, decentralized, hybrid) and to evolution mechanisms (self‑learning, co‑evolution, external knowledge/tools) (Figure 2; Section 2). It also systematizes evaluation (benchmarks, datasets), tools (for/with/by agents), and real‑world issues (security, privacy, social impact) (Sections 3–4).

## 3. Technical Approach
The paper’s “technical approach” is a structured taxonomy that explains how to design and reason about LLM agents end‑to‑end. It is not a single algorithm, but a conceptual architecture backed by concrete exemplars, with a step‑by‑step decomposition (Figure 2; Section 2).

A. Construction: defining a single agent’s internals (Section 2.1)
- Profile definition (Section 2.1.1)
  - What it is: A profile encodes an agent’s role, objectives, capabilities, and behavior constraints.
  - Two implementations:
    - Human‑curated static profiles: Manually specified roles and protocols yield predictable behavior and compliance—e.g., role orchestration in `CAMEL`, `AutoGen`, `MetaGPT`, `ChatDev`, `AFlow` (Section 2.1.1).
    - Batch‑generated dynamic profiles: Parameterized initialization stochastically creates a diverse “population” of agents with varied personas/values, useful for simulating societies or user cohorts—e.g., `Generative Agents`, `RecAgent`, with optional optimization in `DSPy` (Section 2.1.1).
- Memory mechanism (Section 2.1.2)
  - Why memory is split: LLM context windows are limited, so agents need mechanisms to preserve and retrieve relevant information across time.
  - Short‑term memory: Maintains recent dialogue and intermediate thoughts; enables interactive reasoning but is transient and must be compressed or pruned (e.g., `ReAct`, `Graph of Thoughts`, `AFlow`) (Section 2.1.2).
  - Long‑term memory: Converts ephemeral reasoning into persistent skills/knowledge via:
    - Skill libraries (e.g., `Voyager` in Minecraft; `GITM`) 
    - Experience repositories (e.g., `ExpeL`, `Reflexion`) 
    - Tool synthesis/self-expanding toolkits (e.g., `TPTU`, `OpenAgents`) (Section 2.1.2).
  - Knowledge retrieval as memory: Treats `RAG` (Retrieval‑Augmented Generation) and graph retrieval (`GraphRAG`) as an externalized memory layer interleaved with reasoning (`IRCoT`, `Llatrieval`, `KG‑RAR`, `DeepRAG`) (Section 2.1.2).
- Planning capability (Section 2.1.3)
  - Task decomposition strategies:
    - Single‑path chaining: plan‑and‑solve or dynamic next‑step planning; robustness can be improved with multiple reasoning paths (self‑consistency, voting, discussion) (Section 2.1.3).
    - Tree‑based search: `Tree‑of‑Thought (ToT)` explores multiple branches with backtracking and feedback; can integrate Monte Carlo Tree Search for complex domains, including robotics/gameplay (Section 2.1.3).
  - Feedback‑driven iteration:
    - Sources of feedback: environment (embodied settings), humans (labels/guidance), model introspection (self‑critique), and other agents (collaboration) (Section 2.1.3).
    - Mechanism: Regenerate/refine plans in a loop until success criteria are met (e.g., `AdaPlanner`, `AIFP`) (Section 2.1.3).
- Action execution (Section 2.1.4)
  - Tool utilization: Two subproblems—when to use a tool (decision) and which tool to pick (selection). Systems simplify tool docs or leverage tool‑use training to improve reliability (e.g., `EASYTOOL`, `GPT4Tools`, `AvaTaR`, `TRICE`) (Section 2.1.4).
  - Physical interaction: For embodied agents, translate plans to low‑level actions considering hardware, social norms, and multi‑agent coordination (e.g., `DriVLMe`, `ReAd`, `Collaborative Voyager`) (Section 2.1.4).

B. Collaboration: organizing groups of agents (Section 2.2; Table 1)
- Centralized control (Section 2.2.1)
  - Explicit controllers: A central agent (or human) decomposes tasks, allocates subgoals, and integrates results (e.g., `Coscientist`, `LLM‑Blender`, `MetaGPT`) (Section 2.2.1).
  - Differentiation-based control: A high‑capacity model implicitly plays multiple sub‑roles via meta‑prompts and then aggregates (e.g., `AutoAct`, `Meta‑Prompting`, `WJudge`) (Section 2.2.1).
  - Trade‑off: Strong coordination and accountability vs. single‑point bottlenecks and reduced diversity (Section 2.2.1).
- Decentralized collaboration (Section 2.2.2)
  - Revision‑based: Agents iteratively edit/refine a shared output with limited direct discussion; often more deterministic (e.g., `MedAgents`, `ReConcile`, `METAL`, `DS‑Agent`) (Section 2.2.2).
  - Communication‑based: Agents openly debate/critique with structured protocols to avoid “degeneration of thought” and reach consensus (e.g., `MAD`, `MADR`, `MDebate`, `AutoGen`) (Section 2.2.2).
  - Trade‑off: Flexibility and exploration vs. coordination overhead and convergence risks (Section 2.2.2).
- Hybrid architectures (Section 2.2.3)
  - Static hybrids: Predefine central vs. peer‑to‑peer patterns (e.g., `CAMEL` group roles; `AFlow`’s three‑tier planning; `EoT`’s BUS/STAR/TREE/RING topologies) (Section 2.2.3).
  - Dynamic hybrids: Learn collaboration graphs or adapt structures by task importance/complexity (e.g., `DiscoGraph`, `DyLAN`, `MDAgents`) (Section 2.2.3).
  - Trade‑off: Better fit to heterogeneous tasks vs. additional complexity and training/inference cost (Section 2.2.3).

C. Evolution: improving agents over time (Section 2.3; Table 2)
- Autonomous optimization and self‑learning (Section 2.3.1)
  - Self‑supervised adaptation (e.g., `SE`, evolutionary model merging);
  - Self‑reflection/correction (`SELF‑REFINE`, `STaR`, `V‑STaR`, `Self‑Verification`);
  - Self‑rewarding/RL alignment (`Self‑Rewarding`, `RLCD`, `RLC`) (Section 2.3.1).
- Multi‑agent co‑evolution (Section 2.3.2)
  - Cooperative: Intent inference and shared policy improvement (`ProAgent`, `CORY`, `CAMEL`) (Section 2.3.2).
  - Competitive/adversarial: Debate and red‑teaming to strengthen reasoning and robustness (`MDebate`, `MAD`, `Red‑Team LLMs`) (Section 2.3.2).
- Evolution via external resources (Section 2.3.3)
  - Knowledge‑enhanced evolution (`KnowAgent`, `WKM`) to constrain planning and reduce hallucinations;
  - Feedback‑driven evolution via tools/executors (`CRITIC`, `STE`, `SelfEvolve`) (Section 2.3.3).

D. Evaluation and tooling (Sections 3.1–3.2; Figure 3)
- Evaluation frameworks span general agent capability benchmarking, domain‑specific simulations (medicine, driving, data science), and multi‑agent system assessment (Sections 3.1.1–3.1.3).
- Tools are organized as: used by agents (search, calculators, API callers), created by agents (tool creation pipelines), and used to deploy/operate agents (frameworks like `AutoGen`, `LangChain`, `LlamaIndex`, `Dify`; and the `Model Context Protocol`) (Section 3.2).

Definitions of uncommon terms used above:
- `RAG` (Retrieval‑Augmented Generation): a technique where the agent retrieves relevant external documents/graphs during generation to supplement its internal parameters (Section 2.1.2).
- `Tree‑of‑Thought (ToT)`: a tree‑structured reasoning process that explores multiple branches, allows backtracking, and uses feedback to pick better paths (Section 2.1.3).
- `Multi‑agent debate`: a structured dialogue among agents (or multiple runs of one agent) that alternates critique and defense to improve answers/consensus (Sections 2.2.2, 2.3.2).

## 4. Key Insights and Innovations
- A unified lifecycle view: Build → Collaborate → Evolve (Figures 1–2; Section 2)
  - What’s new: Rather than listing techniques, the paper shows how profile, memory, planning, and action form a recursive loop for a single agent, and how collaboration and evolution sit on top of that loop (Sections 2.1–2.3).
  - Why it matters: It connects internal design choices (e.g., long‑term memory) to system‑level properties (e.g., decentralized debate) and long‑horizon improvement (e.g., self‑verification + tool feedback).
- “Knowledge retrieval as memory” (Section 2.1.2)
  - What’s new: Treats RAG/GraphRAG not as “just tools” but as an externalized memory tier with tight reasoning integration (`IRCoT`, `KG‑RAR`, `DeepRAG`).
  - Why it matters: Clarifies architectural implications (e.g., when to store skills vs. when to fetch facts) and helps avoid conflating internal memories with retrieval pipelines.
- Fine‑grained collaboration taxonomy (Section 2.2; Table 1)
  - What’s new: Distinguishes centralized controllers (explicit vs. differentiation‑based), decentralized modes (revision‑ vs. communication‑based), and hybrid systems (static vs. dynamic topology).
  - Why it matters: Provides a vocabulary to compare agent systems by coordination load, robustness, and scalability, not only by task performance.
- Security and privacy reframed for agentic settings (Figure 4; Sections 4.1–4.3; Tables 3–5)
  - What’s new: Splits threats into agent‑centric (adversarial, jailbreak, backdoor, collaboration attacks) vs. data‑centric (prompt injection, external source poisoning, interaction‑level attacks), plus memorization/IP risks.
  - Why it matters: Security thinking shifts from single‑model prompts to system‑level attack surfaces (tools, memories, inter‑agent messages, topology).

These are fundamental framing contributions (not just incremental lists) because they change how we decompose, compare, and secure agent systems end‑to‑end.

## 5. Experimental Analysis
Because this is a survey, “experiments” are curated benchmarks, datasets, and case studies that substantiate coverage and illustrate evaluation practices (Section 3; Figure 3). The paper reports concrete scales and task designs:

- General agent capability benchmarks (Section 3.1.1)
  - AgentBench: 8 interactive environments to test reasoning and acting (Section 3.1.1).
  - Mind2Web:
    > “the first generalist agent for evaluating 137 real‑world websites with different tasks spanning 31 domains” (Section 3.1.1).
  - MMAU: decomposes into five core competencies across 3,000+ tasks (Section 3.1.1).
  - VisualAgentBench: multimodal foundation‑agent evaluation across GUI, visual design, etc. (Section 3.1.1).
  - Embodied Agent Interface and CRAB: fine‑grained error classification and cross‑platform embodied testing (Section 3.1.1).
  - Dynamic/self‑evolving evaluation:
    > “BENCHAGENTS… automatically creates benchmarks through LLM agents” (Section 3.1.1);  
    > “Seal‑Tools (1,024 nested instances of tool calls)” and “CToolEval (398 Chinese APIs across 14 domains)” (Section 3.1.1).

- Domain‑specific and real‑world environments (Section 3.1.2)
  - Medicine: 
    > “MedAgentBench… tasks designed by 300 clinicians in an FHIR‑compliant environment”;  
    > “AI Hospital… simulates clinical workflows through multi‑agent collaboration” (Section 3.1.2).
  - Driving and desktop/web action:
    > “LaMPilot… executable code generation benchmark for autonomous driving”;  
    > “OSWorld… 369 multi‑application tasks across Ubuntu/Windows/macOS” (Section 3.1.2).
  - Data science and ML engineering:
    - DSEval, DA‑Code, DCA‑Bench, MLAgent‑Bench, MLE‑Bench (Section 3.1.2).
  - Planning:
    > “TravelPlanner… 1,225 planning tasks that require multi‑step reasoning, tool integration, and constraint balancing” (Section 3.1.2).
  - Daily‑life multimodal:
    > “EgoLife… a 300‑hour multimodal egocentric dataset… with EgoLifeQA tasks” (Section 3.1.2).
  - Tools‑in‑the‑wild:
    - GTA: general tool agents with real‑world APIs and multimodal inputs (Section 3.1.2).

- Multi‑agent system and collaboration evaluation (Section 3.1.3)
  - TheAgentCompany simulates a software company to test web interaction and code collaboration; MLRB and MLE‑Bench evaluate research/engineering workflows (Section 3.1.3).

- Security robustness benchmarks (Sections 4.1–4.2; Tables 3–4)
  - AgentDojo:
    > “97 realistic tasks and 629 security test cases” (Section 4.1.1).
  - Agent security bench:
    > “across 10 scenarios, 10 agents, 400+ tools, 23 attack/defense methods, and 8 metrics” (Section 4.1).
  - AgentHarm:
    > “440 malicious agent tasks in 11 hazard categories” (Section 3.1.2 and 4.2.2).

- Scientific/medical dataset creation via agents (Sections 5.1.4–5.1.5)
  - PathGen‑1.6M:
    > “1.6 million pathology image‑text pairs generation through multi‑agent collaboration” (Section 5.1.4).

How convincing is the empirical coverage?
- Breadth: The survey spans general capability, domain simulations, embodied settings, tool use, and collaboration—backed by concrete scales and scenarios (Sections 3.1.1–3.1.3).
- Depth: It does not attempt meta‑analysis (e.g., pooled effect sizes) or head‑to‑head bake‑offs across all agents, which would be infeasible given scope; instead, it catalogs evaluation design patterns with examples and numeric scales, sufficient to guide practitioners choosing benchmarks.
- Robustness checks:
  - Security: Attack/defense typologies are tied to testbeds and numbers (Sections 4.1–4.2; Tables 3–4).
  - Ablations per method are out of scope—this is a curation, not a single system’s experiment.

Summary of key quantitative takeaways (selected):
- OSWorld covers “369” tasks across three operating systems (Section 3.1.2).
- Mind2Web spans “137 websites” and “31 domains” (Section 3.1.1).
- TravelPlanner sets “1,225” planning tasks (Section 3.1.2).
- Seal‑Tools: “1,024” multi‑step tool‑use instances; CToolEval: “398 APIs across 14 domains” (Section 3.1.1).
- AgentDojo: “629” adversarial test cases; AgentHarm: “440” harmful tasks (Sections 4.1–4.2).

## 6. Limitations and Trade-offs
- As a survey:
  - Timeliness vs. completeness: The landscape is changing monthly. While the paper includes late‑2024/early‑2025 resources (e.g., OSWorld, EgoLife), new agent forms and safety techniques will quickly emerge (Sections 3–4).
  - No cross‑benchmark synthesis: It catalogs benchmarks and tools but does not normalize task difficulty or compare evaluation outcomes across suites (Sections 3.1–3.2).
- Taxonomy boundaries can blur:
  - “Knowledge retrieval as memory” is insightful, but in practice retrieval, tool calls, and long‑term memory often intertwine, making strict categorization situational (Section 2.1.2).
  - Collaboration categories can overlap in complex pipelines (e.g., hybrid systems that adapt topologies mid‑task) (Section 2.2.3).
- Scalability and cost:
  - Dynamic hybrid collaboration (e.g., learned collaboration graphs) and extensive memory/retrieval introduce computational and engineering overhead (Sections 2.2.3, 6.1).
  - Multi‑agent debate and multi‑path reasoning improve reliability but increase token/latency costs (Sections 2.2.2, 6.1).
- Assumptions and unaddressed scenarios:
  - Many exemplars assume reliable tool APIs and stable environments; failure modes of flaky tools or adversarial APIs remain challenging (Sections 2.1.4, 4.2).
  - Safety defenses are cataloged, but there is no universal defense—e.g., jailbreak/backdoor/collaboration attacks target different layers and require different mitigations (Sections 4.1–4.2; Tables 3–4).
- Evaluation blind spots:
  - Despite growth, dynamic multi‑turn, multi‑agent evaluations with lifecycle tracking (learning over time) are early; benchmark self‑evolution is promising but nascent (Section 3.1.1; Section 6.4).

## 7. Implications and Future Directions
- How this changes the field’s framing (Figures 1–2; Section 6)
  - The Build–Collaborate–Evolve lens turns a list of techniques into an engineering methodology. It encourages explicit design of: (a) internal cognition loops (profile→memory→planning→action), (b) collaboration topology matched to task constraints, and (c) evolution channels (self‑reflection, debate, knowledge/tool feedback).
- What research it enables or suggests (Section 6)
  - Scalable coordination: Hierarchical/decentralized planning with learned or rule‑based scheduling for many agents (Section 6.1).
  - Long‑term memory: Hierarchical (episodic/semantic) memory with autonomous compression to maintain identity and adapt over months (Section 6.2).
  - Reliability and rigor: Built‑in verification—knowledge‑graph cross‑checks, citation‑grounded responses, self‑consistency and audit logs—for high‑stakes domains (Section 6.3).
  - Dynamic evaluation: Benchmarks that evolve as models improve, measure multi‑turn learning, and capture emergent collaboration patterns (Section 6.4).
  - Safety at system level: Topology‑aware defenses, inter‑agent message sanitization, tool‑use firewalls, and constitutional governance for agent collectives (Sections 4, 6.5).
  - Role‑play fidelity: Better modeling of underrepresented roles and culturally diverse behaviors; richer dialogue diversity (Section 6.6).
- Practical applications and downstream use (Section 5; Table 7)
  - Science and engineering: Hypothesis generation, experimental planning/execution, and dataset creation (e.g., `SciAgents`, `Curie`, `ChemCrow`, `PathGen‑1.6M`) (Sections 5.1.1–5.1.4).
  - Healthcare: Simulated hospitals, diagnostic agents, and medically grounded conversation (e.g., `AgentHospital`, `ClinicalLab`, `AIPatient`, `CXR‑Agent`, `MedRAX`) (Section 5.1.5).
  - Productivity: Software engineering with role‑based multi‑agent workflows (`ChatDev`, `MetaGPT`) and recommender systems where agents model users/items (`AgentCF`, `RecMind`) (Section 5.4).
  - Web/OS automation and embodied control: OSWorld‑style assistants, web agents (Mind2Web), and embodied planning with feedback loops (Sections 3.1.2, 2.1.3–2.1.4).

In short, this survey provides an actionable blueprint for designing, evaluating, and securing LLM agents. Its lifecycle perspective helps teams move from ad hoc prompting to principled, evolvable, and safer agent systems, while pointing to key open problems in scalable coordination, long‑term memory, dynamic evaluation, and system‑level safety.
