# Large Language Model Agent: A Survey on Methodology, Applications and Challenges

**ArXiv:** [2503.21460](https://arxiv.org/abs/2503.21460)

## 🎯 Pitch

This paper introduces a unified, methodology-driven taxonomy—spanning how LLM agents are built, collaborate, and evolve—offering the first comprehensive architectural perspective that interlinks agent construction, coordination, memory, safety, evaluation, and real-world deployment. By systematizing a fragmented and rapidly advancing field, this survey equips researchers and practitioners with actionable frameworks to design, benchmark, and scale reliable, robust, and impactful LLM-based agentic systems across diverse applications.

---

## 1. Executive Summary

This survey systematically deconstructs LLM agent systems through a **methodology-centered taxonomy** organized around three interconnected dimensions—construction, collaboration, and evolution—unifying fragmented research threads by revealing fundamental connections between agent design principles and their emergent behaviors in complex environments. The taxonomy links architectural foundations (profile definition, memory mechanisms, planning capabilities, action execution) with collaboration paradigms (centralized control, decentralized collaboration, hybrid architectures) and evolutionary pathways (autonomous self-learning, multi-agent co-evolution, external resource integration), while also mapping evaluation benchmarks, tool ecosystems, security threats, and applications spanning scientific discovery to productivity tools. The work establishes that effective LLM agent design requires understanding the continuity between individual agent construction and collective system behavior, with the survey identifying persistent challenges—scalability limitations, memory constraints, reliability concerns, and inadequate evaluation frameworks for multi-turn, multi-agent dynamics—that define the frontier for future research across this rapidly evolving field.

## 2. Context and Motivation

### The Core Problem: We Lack a Unified Architectural Understanding of LLM Agents

The fundamental gap this paper addresses is the **fragmentation of research on LLM-based agents**. The field has experienced explosive growth—the survey documents hundreds of papers spanning agent construction, collaboration, evolution, security, and applications—but this rapid expansion has produced a disconnected literature where individual contributions are difficult to compare, synthesize, or build upon systematically. As the authors state in the Introduction:

> "Despite several surveys exploring various aspects of AI agents in recent years, our study makes a distinctive contribution through its methodological focus and comprehensive analysis of LLM agent architectures."

This fragmentation manifests in three specific ways that the paper identifies and aims to resolve.

**First, the absence of a shared taxonomy.** Different research communities use different language to describe similar mechanisms. What one paper calls "self-reflection," another calls "self-correction," and a third calls "iterative refinement"—even when referring to substantially overlapping techniques. Without a common vocabulary organized around architectural primitives, it becomes difficult to identify which design choices matter, which are incidental, and where genuine innovation occurs. The paper's **methodology-centered taxonomy** (Figure 2) is the primary solution: it deconstructs agents into four foundational components—profile definition, memory mechanisms, planning capabilities, and action execution—providing a stable reference frame against which any agent architecture can be described and compared.

**Second, the artificial separation of single-agent and multi-agent research.** The paper observes that prior surveys and research threads have largely treated individual agent construction and multi-agent collaboration as independent topics. Work on "agent architectures" examined internal mechanisms (memory, planning, tool use) in isolation, while work on "multi-agent systems" studied coordination protocols without connecting them back to the design of the individual agents involved. As the authors note in their distinction from previous surveys:

> "We analyze three interconnected dimensions of LLM agents—construction, collaboration, and evolution—offering a more holistic understanding than previous approaches... This integrated architectural perspective highlights the continuity between individual LLM agent design and collaborative systems, whereas prior studies have often examined these aspects separately."

This separation is problematic because **the design of an individual agent fundamentally constrains how it can collaborate**. An agent with only short-term memory, for instance, cannot effectively participate in long-running multi-agent workflows that require persistent state. Conversely, collaboration patterns influence what individual capabilities matter most. The paper's **Build-Collaborate-Evolve framework** treats these as dimensions of a single design space rather than separate research areas.

**Third, the neglect of evolutionary dynamics.** Most prior work treats agents as static entities—once constructed, their capabilities are fixed. But practical agent systems improve over time through various mechanisms: self-supervised learning, interaction with other agents, integration of external knowledge, and feedback from deployed use. The paper's inclusion of **evolution as a first-class dimension** (Section 2.3)—alongside construction and collaboration—reflects the reality that production agent systems are not deployed once but continuously adapted. This is particularly important for the "self-improvement" narrative in LLM research, where models are expected to bootstrap their own capabilities, but also for understanding how multi-agent systems can exhibit emergent collective behaviors not present in any individual agent.

### Why This Problem Matters: Real-World Stakes and Theoretical Significance

The problem of fragmented understanding is not merely academic—it has direct practical consequences and theoretical implications that the paper highlights.

**Practical stakes.** LLM agents are being deployed in consequential domains: healthcare (AgentHospital, ClinicalLab, CXR-Agent), scientific research (ChemCrow, SciAgents, BioDiscoveryAgent), software development (MetaGPT, ChatDev), and financial systems (TradingGPT). As the paper documents in Section 5, these applications involve autonomous decision-making, tool orchestration, and multi-step reasoning chains where failures can have material costs. Without a systematic understanding of how agent design choices affect behavior, developers are operating largely through trial and error:

> "Current benchmarks primarily assess task execution such as code completion and dialogue generation in isolated settings, overlooking emergent agent behaviors, long-term adaptation, and collaborative reasoning that unfold across multi-turn interactions." (Section 6.4)

The paper argues that a unified taxonomy enables more principled engineering: if we understand that memory mechanisms directly affect planning quality, we can diagnose failures systematically rather than guessing whether to adjust the prompt, fine-tune the model, or restructure the collaboration topology. The taxonomy is, in this sense, a **design tool** as much as a descriptive framework.

**Theoretical significance.** The conceptual gap reflects a deeper theoretical question: **What is the right level of abstraction for understanding LLM-based agents?** Traditional AI agent theory (Wooldridge and Jennings, 1995, cited as reference [2]) operated at the level of belief-desire-intention (BDI) architectures and formal logic, assuming agents had well-defined internal states and reasoning procedures. LLM-based agents disrupt these assumptions because:

- Their "reasoning" is emergent from next-token prediction rather than explicit symbolic manipulation.
- Their "knowledge" is distributed across billions of parameters rather than stored in explicit knowledge bases.
- Their "planning" happens through chain-of-thought prompting rather than search over action sequences in a world model.

The paper's taxonomy is an attempt to **define a new abstraction layer** that captures what is distinctive about LLM agents while maintaining enough structure to support systematic analysis. The four components—profile, memory, planning, action—are deliberately chosen to bridge classical agent theory and modern LLM capabilities. Profile definition replaces explicit goal specifications with prompted identities. Memory mechanisms replace belief revision with context window management and retrieval augmentation. Planning replaces symbolic search with prompt-engineered decomposition. Action execution replaces actuator commands with tool API calls. This bridging is not merely taxonomic; it suggests that **classical agent theory may need revision to accommodate LLM-based systems**, and the paper's framework is a step toward that revision.

### Where Prior Approaches Fall Short

The paper identifies specific limitations in existing work that motivate its comprehensive approach.

**Prior surveys are scope-limited rather than method-focused.** The authors explicitly contrast their work with several categories of previous surveys:

- **Application-specific surveys**: Reviews focused on gaming [11], [12] or embodied agents [13], [14], [15] provide depth in one domain but cannot reveal cross-cutting architectural patterns. A planning mechanism that works for game-playing agents may also work for scientific discovery agents, but domain-specific surveys cannot surface this connection.
- **Threat-specific surveys**: Security-focused work [16] catalogs vulnerabilities but doesn't connect them to architectural design choices. The paper's integrated approach (Section 4) ties security threats directly to specific architectural components—agent-centric attacks target the model itself, data-centric attacks target the memory/retrieval pipeline—making the security analysis actionable for system designers.
- **Broad overviews without methodological taxonomy**: Some prior surveys [1], [17] provided general coverage but lacked the structured decomposition that enables systematic comparison. As the authors note, without a taxonomy, "broad overviews" risk becoming annotated bibliographies rather than analytical frameworks.
- **Single-dimension surveys**: Recent work on multi-agent interaction [18], workflows [19], and cooperative decision-making [20] each focus on one aspect of the agent lifecycle. The paper argues that these dimensions cannot be understood in isolation because construction choices affect collaboration capabilities, and collaboration patterns enable or constrain evolution mechanisms.

**The Build-vs-Collaborate separation is artificial and limiting.** The paper identifies a specific gap in how the literature handles the relationship between individual and collective agent behavior:

> "Prior studies have often examined these aspects separately" (from the "Distinction from Previous Surveys" in Section 1)

This separation matters because **individual agent architectures implicitly define the space of possible collaborations**. An agent built with a human-curated static profile (e.g., a defined role like "software architect" in MetaGPT) interacts differently than one with a batch-generated dynamic profile (e.g., a personality-initialized agent in Generative Agents). The former produces deterministic, role-constrained interactions; the latter produces emergent, personality-driven interactions. Without connecting construction to collaboration, we cannot predict or control collective behavior from individual design choices.

**The evolution dimension is underdeveloped in prior taxonomies.** The paper's inclusion of evolution (Section 2.3) as a co-equal dimension alongside construction and collaboration is a deliberate departure from prior surveys. While individual works study self-improvement, multi-agent co-evolution, or knowledge integration, no prior survey has organized these into a systematic taxonomy. This gap is significant because:

- Evolution mechanisms determine whether an agent system **degrades or improves** over deployment time.
- Without an evolutionary perspective, agent design becomes a one-shot activity rather than a continuous process.
- Many of the most promising agent applications (scientific discovery, self-improving code generation) depend on evolution mechanisms.

### How This Paper Positions Itself Relative to Existing Work

The paper defines its contribution through three explicit differentiators quoted in Section 1. We examine each in depth.

**1. Methodology-centered taxonomy.** The paper is not organized by application, model architecture, or chronological development—it is organized by **what agents are composed of and how those components function**. This is a deliberate choice with specific implications:

- The taxonomy is intended to be **model-agnostic**: whether an agent uses GPT-4, PaLM 2, or an open-source model, the same architectural categories apply. This future-proofs the survey against rapid model iteration.
- The taxonomy is **functional rather than descriptive**: categories are defined by what they do (store information, decompose tasks, execute actions) rather than by surface features (prompt length, model size, training data). This enables meaningful comparison across diverse implementations.
- The taxonomy enables **gap identification**: by mapping the full space of possibilities within each category, the taxonomy reveals underexplored regions. For example, the paper's treatment of memory mechanisms spans short-term, long-term, and retrieval-based approaches—if a new agent system uses none of these, the taxonomy would identify it as an anomalous case worthy of investigation.

**2. Build-Collaborate-Evolve framework.** The three-dimensional organization (Figure 2) is not merely a presentation structure; it represents a **theoretical claim** about the lifecycle of agent systems:

- **Construction** addresses the static architecture: what is the agent at a moment in time?
- **Collaboration** addresses the interactional architecture: how do agents combine to form larger systems?
- **Evolution** addresses the temporal architecture: how do agents and agent systems change over time?

These three dimensions are **interdependent in both directions**. Construction choices enable or constrain collaboration (a centralized controller agent needs different internal capabilities than a peer in a decentralized network). Collaboration dynamics drive evolution (debate among agents produces improved outputs that can be distilled back into individual models). Evolution reshapes construction (self-improved agents have different capabilities than their initial versions). By treating these as interconnected dimensions within a single framework, the paper positions itself as providing a **unified design space** for agent systems rather than a catalog of isolated techniques.

**3. Frontier applications and real-world focus.** The paper distinguishes itself by treating practical deployment concerns as **first-class dimensions of the taxonomy**, not afterthoughts or future work sections. The evaluation and tools ecosystem (Section 3), security and privacy threats (Section 4), and diverse application domains (Section 5) are not appendices—they are integral to the framework. This positioning reflects a specific stance: that LLM agent research is transitioning from **feasibility demonstrations** to **deployment engineering**, and that taxonomies must evolve accordingly. The paper's inclusion of:

- Model Context Protocol (MCP) as an emerging standard for agent-tool communication (Section 3.2.3)
- Specific attack taxonomies with named methods (e.g., CORBA [196], AiTM [197], DemonAgent [191]) and corresponding defenses (Section 4.1)
- Production tooling like LangChain, LlamaIndex, Dify, and Ollama (Section 3.2.3)
- Ethical concerns spanning bias, accountability, copyright, and environmental impact (Section 4.4)

...signals that the paper aims to serve practitioners building and deploying agent systems, not just researchers developing novel agent architectures. This distinguishes it from surveys that focus exclusively on algorithmic contributions without addressing the surrounding infrastructure and societal context.

### The Convergence Driving the Need for This Survey

The paper identifies a specific historical moment that makes its comprehensive taxonomy timely. In the Introduction, the authors describe a **three-way convergence** that has transformed LLM agents from theoretical constructs into practical systems:

> "Today's agents represent a qualitative leap driven by the convergence of three key developments: unprecedented reasoning capabilities of LLMs, advancements in tool manipulation and environmental interaction, and sophisticated memory architectures that support longitudinal experience accumulation."

This convergence explains why the survey is needed now rather than earlier. Before these three capabilities matured simultaneously:

- **LLMs could reason** but couldn't act (no tool use, no environmental interaction).
- **Agents could operate** in environments but had brittle, hand-coded reasoning (pre-LLM era).
- **Memory systems existed** but couldn't integrate with flexible reasoning and acting in a unified architecture.

The convergence means that **the design space has exploded combinatorially**: any combination of reasoning strategy, memory architecture, tool set, and collaboration topology is now possible. Without a systematic taxonomy, navigating this space is intractable. The paper's contribution is essentially a **coordinate system** for this expanded design space, enabling researchers and practitioners to locate their work relative to existing approaches and identify unexplored regions.

The paper also positions itself against the backdrop of specific commercial systems—DeepResearch, DeepSearch, and Manus—cited in the Introduction's opening paragraph. These systems exemplify the gap the survey aims to fill: they are deployed, consequential, and complex, but their architectural principles are not systematically understood. The survey provides the conceptual tools to **reverse-engineer and compare** such systems by decomposing them into the taxonomy's components, even when their internal details are proprietary.

## 3. Technical Approach

### 3.1 Reader Orientation

This survey paper does not present a single technical system but rather constructs a **unified architectural taxonomy**—a conceptual classification framework—that decomposes any LLM-based agent system into its fundamental building blocks across three interdependent dimensions: how agents are constructed, how they collaborate, and how they evolve over time. The taxonomy solves the problem of **fragmented understanding** in a rapidly growing research field: by defining a stable set of architectural primitives (profile, memory, planning, action) and showing how they compose across construction, collaboration, and evolutionary layers, the framework enables systematic comparison of hundreds of disparate papers, identification of design patterns that transfer across applications, and recognition of underexplored regions in the design space where future work can have the most impact.

### 3.2 Big-Picture Architecture (Diagram in Words)

The taxonomy organizes LLM agent systems into a **three-dimensional classification space**, with each dimension further decomposed into subcategories:

**Dimension 1: Agent Construction (Section 2.1)** defines the static architecture of a single agent at a moment in time. It has four components, each a class of design decisions:

- **Profile Definition (Section 2.1.1):** What role does the agent play? Two approaches—human-curated static profiles (experts manually specify behavioral rules) and batch-generated dynamic profiles (parameterized initialization produces diverse agent populations).
- **Memory Mechanism (Section 2.1.2):** How does the agent store and retrieve information? Three subcategories—short-term memory (transient dialog history for immediate context), long-term memory (persistent skill libraries and experience repositories), and knowledge retrieval as memory (external knowledge integration via RAG and knowledge graphs).
- **Planning Capability (Section 2.1.3):** How does the agent decompose tasks? Two perspectives—task decomposition strategies (single-path chaining vs. multi-path tree expansion) and feedback-driven iteration (using environmental, human, self-generated, or multi-agent feedback to refine plans).
- **Action Execution (Section 2.1.4):** How does the agent act in the world? Two aspects—tool utilization (deciding when and which tools to invoke) and physical interaction (embodied action in robotic or simulated environments).

**Dimension 2: Agent Collaboration (Section 2.2)** defines how multiple agents coordinate. Three architectures:

- **Centralized Control (Section 2.2.1):** A single controller allocates tasks and integrates decisions. Two implementation strategies—explicit controller systems (dedicated coordination modules) and differentiation-based systems (a meta-agent assumes distinct sub-roles via prompting).
- **Decentralized Collaboration (Section 2.2.2):** Agents interact directly without a central coordinator. Two approaches—revision-based systems (agents iteratively refine a shared output through structured editing) and communication-based systems (agents engage in direct dialogues and observe peers' reasoning).
- **Hybrid Architecture (Section 2.2.3):** Combines centralized and decentralized elements. Two patterns—static systems (predefined combination rules) and dynamic systems (topology adapts via neural optimizers based on real-time feedback).

**Dimension 3: Agent Evolution (Section 2.3)** defines how agents improve over time:

- **Autonomous Optimization and Self-Learning (Section 2.3.1):** Self-supervised learning, self-reflection and self-correction, and self-rewarding via reinforcement learning.
- **Multi-Agent Co-Evolution (Section 2.3.2):** Cooperative learning (knowledge sharing and joint decision-making) and competitive co-evolution (adversarial interactions and debate).
- **Evolution via External Resources (Section 2.3.3):** Knowledge-enhanced evolution (integrating structured external knowledge) and external feedback-driven evolution (using tool-based, evaluator-based, and human feedback).

The taxonomy is **recursive**: the way an agent is constructed determines what collaboration patterns it can participate in; collaboration dynamics generate outputs that can drive evolution; evolution reshapes individual agent capabilities, potentially enabling new construction choices. The framework treats these as a single interconnected design space rather than independent research areas.

### 3.3 Roadmap for the Deep Dive

The detailed breakdown follows the taxonomy's own organizational logic, moving from the innermost layer (construction) outward to collaboration and evolution, then connecting to the practical infrastructure (evaluation, tools, security) that surrounds deployed agent systems:

- **First, Agent Construction (Section 3.4.1):** We examine the four foundational components—profile, memory, planning, and action—because these define the primitive capabilities that all higher-level agent behaviors build upon. Understanding construction first is essential because collaboration mechanisms and evolution strategies are only meaningful in terms of what they assume about individual agent architectures.
- **Second, Agent Collaboration (Section 3.4.2):** We analyze the three architectural patterns for multi-agent coordination—centralized, decentralized, and hybrid—showing how construction choices enable or constrain each pattern and how different patterns suit different task characteristics.
- **Third, Agent Evolution (Section 3.4.3):** We examine how agents improve through autonomous optimization, multi-agent co-evolution, and external resource integration, connecting each evolution mechanism back to the construction and collaboration layers it depends on.
- **Fourth, Evaluation Benchmarks and Tools (Section 3.4.4):** We survey the assessment frameworks and tooling ecosystem, since evaluation methodology shapes what agent capabilities are measured and tool infrastructure shapes what agents can actually do in deployment.
- **Fifth, Security, Privacy, and Social Impact (Section 3.4.5):** We map the threat landscape and ethical considerations, showing how vulnerabilities correspond to specific architectural components and how the taxonomy can guide defensive design.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **taxonomy paper**—a systematic classification of a research field—whose core idea is that LLM agent systems can be understood through a unified architectural framework that connects individual agent construction, multi-agent collaboration, and temporal evolution as interdependent dimensions of a single design space.

---

#### 3.4.1 Agent Construction: The Four Foundational Components

Agent construction is the process of designing the internal architecture of a single LLM-based agent. The paper decomposes this architecture into four components that collectively define what an agent can perceive, remember, plan, and do. These components form a **recursive optimization loop**: memory informs planning, execution outcomes update memory, and contextual feedback refines agent profiles. The paper emphasizes that construction is not a one-time activity but establishes the foundation upon which collaboration and evolution mechanisms later operate.

##### Profile Definition: Establishing Agent Identity and Behavioral Bounds

Profile definition is the process of configuring an agent's operational identity—its intrinsic attributes, behavioral patterns, decision boundaries, and interaction protocols. The paper identifies two fundamentally different approaches to profile specification, distinguished by **who or what determines the profile** and **when the profile is fixed**.

**Human-curated static profiles** involve domain experts manually specifying agent roles, rules, and domain-specific knowledge before deployment. The profile is fixed at design time and does not change during operation. This approach is described as being "particularly effective in scenarios demanding high interpretability and regulatory compliance" because every behavioral constraint is explicitly authored and auditable. The paper identifies two sub-patterns within human-curated profiles:

- **Coordinated interaction patterns**: Systems like Camel, AutoGen, and OpenAgents orchestrate human-agent collaboration through predefined conversational roles. For example, AutoGen's "user proxy" and "assistant" roles establish a fixed interaction protocol where the user proxy executes code and the assistant generates responses. These roles are explicitly designed to constrain the interaction to a productive pattern—the assistant cannot execute code, and the user proxy cannot generate creative responses, preventing category errors in the collaboration.
- **Role-based coordination patterns**: Systems like MetaGPT, ChatDev, and AFlow assign agents specialized functional roles drawn from human organizations. ChatDev, for instance, coordinates static technical roles (product manager, programmer, tester) with deterministic interaction protocols that mirror software development workflows. Each role has a predefined set of responsibilities and communication patterns—the product manager specifies requirements, the programmer implements them, the tester validates—creating a division of labor analogous to human teams.

The key design choice here is **determinism over flexibility**: by fixing profiles and protocols in advance, these systems ensure consistent, interpretable behavior at the cost of adaptability to novel situations.

**Batch-generated dynamic profiles** take the opposite approach: they use parameterized initialization to systematically generate diverse agent populations by injecting controlled variations into personality traits, knowledge backgrounds, or value systems during agent creation. This is described as being "essential for simulating realistic human-agent interactions in applications ranging from social behavior studies to emergent group intelligence simulations." The paper identifies:

- **Human behavior simulation**: Generative Agents (Park et al.) creates a population of agents with distinct personalities, memories, and daily routines within an interactive sandbox environment. Each agent's profile is generated from a template that specifies demographic attributes and personality traits, but the specific manifestation emerges from interactions with the environment and other agents. This enables the study of emergent social phenomena that would not appear with uniform, static profiles.
- **Simulated user data collection**: RecAgent uses LLM agents with varied profiles to simulate user behavior for recommender system evaluation. Different profile configurations directly shape collective interaction patterns—agents with different "preference" profiles interact differently with items, producing realistic recommendation data.
- **Programmatic profile optimization**: DSPy takes this further by treating profile initialization as an optimization problem. Rather than hand-designing or randomly sampling profiles, DSPy "can further optimize the parameters of the agent profile initialization" through a compilation process that searches over prompt structures and demonstrations. This represents a meta-level approach: the profile itself is learned rather than specified.

The tradeoff between the two approaches is fundamental: human-curated profiles provide **control and interpretability** (you know exactly why an agent behaves as it does) at the cost of **diversity and emergence** (you only get the behaviors you designed). Batch-generated profiles provide **richness and realism** (you get emergent social dynamics) at the cost of **predictability** (you cannot fully specify what any individual agent will do).

##### Memory Mechanism: Storing and Retrieving Information Across Time

Memory mechanisms equip agents with the ability to store, organize, and retrieve information, enabling continuity across interactions and learning from experience. The paper identifies three categories, distinguished by **what is stored** (internal vs. external), **how long it persists** (transient vs. persistent), and **how it is accessed** (contextual vs. retrieval-based).

**Short-term memory** retains agent-internal dialog histories and environmental feedback for immediate task execution. Its defining characteristic is **transience**: the information exists only within the current interaction or reasoning chain. The paper identifies two practical constraints that shape short-term memory design:

- **Context window limitations**: LLMs have finite context windows, so "practical implementations require active information compression (e.g., summarization or selective retention) and impose many constraints on multi-turn interaction depth to prevent performance degradation." This is not merely an implementation detail—it is a fundamental design tension. More context improves reasoning but consumes capacity that could be used for new information.
- **Dissipation of intermediate reasoning**: "Intermediate reasoning traces often dissipate after task completion and cannot be directly transferred to new scenarios." This means that short-term memory captures the process of reasoning but not its products—each new task starts largely from scratch unless long-term memory mechanisms are explicitly employed.

The paper notes that short-term memory is widely implemented across diverse frameworks—ReAct for reasoning-with-reflection loops, ChatDev for software development dialog, Graph of Thoughts for elaborate problem-solving, and AFlow for workflow automation—indicating its fundamental role irrespective of application domain.

**Long-term memory** addresses the dissipation problem by systematically archiving agents' intermediate reasoning trajectories and synthesizing them into reusable assets. The key transformation is: **ephemeral cognitive effort → persistent operational asset**. The paper identifies three dominant paradigms for this transformation:

- **Skill libraries** codify procedural knowledge—not just "what happened" but "how to do things." Voyager's automated skill discovery in Minecraft exemplifies this: as the agent explores, it discovers reusable behaviors (e.g., "craft a wooden pickaxe"), encodes them as executable programs, and adds them to a persistent skill library. Future tasks can retrieve and compose these skills rather than reasoning from scratch. GITM's text-based knowledge base performs an analogous function in a different domain, demonstrating transferability of the paradigm.
- **Experience repositories** store success and failure patterns rather than executable skills. ExpeL's distilled experience pool records task outcomes along with contextual features, enabling the agent to recognize situations similar to past successes or failures. Reflexion's trial-optimized memory stores linguistic feedback from failed attempts, which is retrieved and incorporated into prompts for subsequent attempts on similar tasks. The distinction from skill libraries is important: experience repositories store **evaluative knowledge** (was this approach good or bad?), while skill libraries store **procedural knowledge** (how do I do this?).
- **Tool synthesis frameworks** represent a more generative approach: rather than storing fixed skills or experiences, the agent evolves capabilities through combinatorial adaptation of existing tools. TPTU's adaptive tool composition allows the agent to combine primitive tools into new composite tools at runtime; OpenAgents' self-expanding toolkit adds newly created tools back to the available tool set. This blurs the line between memory and tool use—the memory system becomes a tool factory.

The paper notes that these paradigms are not mutually exclusive. MemGPT's tiered memory architecture exemplifies their integration: different types of information are stored at different levels (working memory, archival storage) with explicit management of what moves between levels, analogous to operating system memory hierarchies.

**Knowledge retrieval as memory** diverges fundamentally from the previous two categories: instead of storing internally generated information, it integrates external knowledge repositories into the generation process. The paper identifies this as a paradigm that "enables agents to transcend training data limitations while maintaining contextual relevance." Three implementation approaches are distinguished:

- **Static knowledge grounding**: RAG retrieves relevant text passages from a corpus and injects them into the agent's context. GraphRAG does the same for structured knowledge graphs, where entities and relationships provide more precise factual grounding than unstructured text. In both cases, the knowledge source is external, fixed, and queried at inference time.
- **Interactive retrieval**: Chain of Agents demonstrates a more dynamic pattern where "short-term inter-agent communications trigger contextualized knowledge fetching." Here, retrieval is not just a prep step before generation—it is interleaved with multi-agent dialogue, with one agent's query triggering retrieval that another agent uses. This creates a feedback loop between interaction and information access.
- **Reasoning-integrated retrieval**: IRCoT and Llatrieval "interleave step-by-step reasoning with dynamic knowledge acquisition." Rather than retrieving all relevant information upfront and then reasoning, the agent alternates between reasoning steps and retrieval queries. Each reasoning step identifies what additional information is needed; each retrieval step grounds the next reasoning step. KG-RAR extends this by constructing task-specific subgraphs during reasoning, building a customized knowledge structure rather than querying a static one. DeepRAG introduces fine-tuned retrieval decision modules that learn when to rely on parametric knowledge (what the LLM already knows) versus when to fetch external evidence—"balancing parametric knowledge and external evidence."

The key insight unifying this category is that **retrieval is not just a data access mechanism but a form of memory**—it extends the agent's effective memory beyond its training data and context window, but with different properties (verifiability, updatability, precision) than internally stored information.

##### Planning Capability: Decomposing and Iteratively Refining Tasks

Planning is what enables agents to navigate complex tasks rather than only handling simple one-step queries. The paper structures its analysis around two complementary perspectives: **task decomposition** (how an agent breaks down a complex problem) and **feedback-driven iteration** (how the agent improves its plan based on information received during execution).

**Task decomposition strategies** are categorized along a spectrum from simpler, linear approaches to more complex, branching approaches:

**Single-path chaining** represents the simplest form of planning. The zero-shot chain-of-thought approach asks the agent to "devise a plan, which consists of a sequence of subtasks that are built upon one another," then solve each subtask sequentially. The paper identifies a critical limitation of this straightforward approach: "it may suffer from a lack of flexibility and error accumulation during chaining, as the agent is required to follow the pre-defined plan without any deviation during the problem-solving procedure." In other words, if step 3 of a plan contains an error, steps 4 through N will all be built on a faulty foundation with no mechanism to detect or correct the problem.

Two lines of work address this limitation:

- **Dynamic planning** generates only the next subtask based on the agent's current situation, rather than generating the entire plan upfront. ReAct (Yao et al.) exemplifies this approach: the agent alternates between reasoning steps and action steps, with each reasoning step determining what to do next based on the most recent observation. This "enables the agent to receive environmental feedback and adjust its plan accordingly, enhancing its robustness and adaptability."
- **Ensemble reasoning** uses multiple chain-of-thought paths to improve robustness, "similar to ensemble methods" in machine learning. Self-consistency (Wan et al., Wang et al.) samples multiple reasoning chains and selects the most common conclusion. Majority voting aggregates across chains. Agent discussion (chain-of-discussion) has multiple models debate and combine their chains. The intuition is that while any single chain may contain errors, errors are unlikely to be systematically consistent across independently generated chains, so aggregation enhances reliability.

**Multi-path tree expansion** is described as "a more complicated method" that uses trees rather than chains as the planning data structure. The key difference from single-path approaches is that "multiple possible reasoning paths exist when the agent is planning, and the agent is allowed to backtrack with information from feedback." The Tree-of-Thought (ToT) method (Long et al.) is the canonical example: it "explores the solution space through a tree-like thought process," enabling backtracking to previous states when a reasoning path proves unfruitful. This makes it "possible for the model to correct its previous mistakes, enabling applications to various complicated tasks that involve the 'trial-error-correct' process."

The paper notes that in realistic scenarios, tree-based planning can incorporate reinforcement learning and Monte Carlo Tree Search for more sophisticated exploration. ReST-MCTS* uses "process reward guided tree search" to train LLMs through self-generated reasoning trajectories. This connects planning to evolution—the search process generates data that can improve the model.

**Feedback-driven iteration** is the second major planning perspective: it addresses how agents use feedback to improve their plans over time. The paper categorizes feedback by its **source**:

- **Environmental feedback** is "one of the most common types of feedback in robotics, generated by the environment in which the embodied agent operates." For example, a robot agent that attempts to grasp an object receives tactile feedback indicating success or failure; this feedback directly informs whether the current plan is working.
- **Human feedback** comes from user interactions or manually labeled data. TrainerAgent uses human guidance to customize model training; other systems use human-provided demonstrations or corrections.
- **Model introspection** is feedback "generated by the agent itself." The agent evaluates its own outputs—checking for consistency, correctness, or completeness—and uses these self-assessments to refine its plans. Dynamic self-consistency (RASC) uses the diversity of reasoning paths as an introspective signal.
- **Multi-agent collaboration** provides feedback when "multiple agents work together to solve a problem and exchange insights." REVECA uses information relevance and plan validation across agents; one agent's critique becomes another agent's feedback.

Regardless of source, the feedback loop follows the same pattern: "the agent can use feedback to update (regenerate) its plan, adjust its reasoning path, or even modify its goal. This iterative process continues until a satisfactory plan is achieved." AdaPlanner and AIFP demonstrate this iterative refinement in path planning and obstacle-aware navigation respectively.

##### Action Execution: Translating Plans into World Effects

Action execution addresses the critical gap between planning and doing. As the paper states: "good plans are useless if the agent cannot execute them effectively." Execution involves two distinct capabilities that the paper analyzes separately.

**Tool utilization** involves two sub-problems: the **tool-use decision** (deciding whether to invoke a tool at all) and **tool selection** (choosing which specific tool to invoke).

The tool-use decision is characterized as a confidence-calibrated choice: "when the agent is generating content with less confidence or facing problems related to specific tool functions, the agent should decide to use specific tools." The paper cites TRICE and GPT4Tools as systems that make this decision based on execution feedback and self-instruction respectively. The implicit model is that the agent maintains an internal estimate of whether its parametric knowledge suffices for the current query; when confidence is low, it delegates to external tools.

Tool selection is characterized as requiring dual understanding: "the understanding of tools and the agent's current situation." EASYTOOL addresses tool understanding by "simplifying the tool documentation to better understand the available tools, enabling a more accurate selection of tools." This reveals a practical challenge: tool documentation written for human developers may be verbose, ambiguous, or structured in ways that LLMs cannot efficiently parse. Simplification and restructuring for LLM consumption is a necessary preprocessing step. AvaTaR optimizes tool usage through contrastive reasoning—learning to distinguish between effective and ineffective tool invocations.

The paper's discussion of tool utilization implicitly reveals a **capability boundary** for LLM agents: they can use tools designed for them, but they struggle with tools designed for humans without adaptation. This is why the tool creation paradigm (Section 3.2.2) is necessary—in many cases, it is easier for the agent to create a custom tool than to learn to use an existing human-oriented one.

**Physical interaction** is presented as the embodied extension of action execution. The paper emphasizes that "when deployed in real-world settings, LLM agents must comprehend various factors to execute actions accurately," listing "robotic hardware, social knowledge, and interactions with other LLM agents." The challenges here differ from tool utilization:

- **Hardware constraints**: Robotic actions have physical preconditions (the arm must be positioned correctly) and effects (grasping changes the world state in ways that affect future actions) that purely digital tools do not. BrainBody-LLM addresses this by grounding LLM reasoning in closed-loop state feedback.
- **Social context**: DriVLMe enhances autonomous driving agents with "embodied and social experiences," recognizing that driving decisions are not purely physical—they involve interpreting other agents' intentions and following social conventions.
- **Multi-agent physical coordination**: ReAd and Collaborative Voyager address scenarios where multiple embodied agents must coordinate physical actions, extending collaboration from information exchange to coordinated movement and manipulation.

---

#### 3.4.2 Agent Collaboration: Coordination Architectures for Multi-Agent Systems

The paper introduces collaboration as the second dimension of its taxonomy, arguing that collaboration "plays a crucial role in extending problem-solving capabilities beyond individual reasoning." The three architectures—centralized, decentralized, and hybrid—are distinguished by their **decision hierarchies** (who has authority), **communication topologies** (who talks to whom), and **task allocation mechanisms** (how work is divided).

##### Centralized Control: Hierarchical Coordination Through a Controller

Centralized control employs a hierarchical coordination mechanism where a central controller orchestrates agent activities. The defining constraint is: "other sub-agents can only communicate with the controller"—there are no direct peer-to-peer interactions among sub-agents.

The paper identifies two implementation strategies for how the controller role is established:

**Explicit controller systems** use "dedicated coordination modules (often implemented as separate LLM agents) to decompose tasks and assign subgoals." The controller is a distinct architectural component with specific responsibilities:

- Coscientist exemplifies this pattern in scientific research: a human operator serves as the central controller, establishing standardized experimental workflows, allocating specialized agents and tools to distinct experimental phases, and maintaining direct control over the final execution plan. The human-in-the-loop aspect is important—in high-stakes scientific work, the controller is a human making final decisions based on agent-generated options.
- LLM-Blender explicitly creates a controller that employs a cross-attention encoder for pairwise comparison to identify the best responses among multiple candidates, then fuses the top-ranked responses. This reveals a specific controller function: **response synthesis**, not just selection. The controller doesn't just pick the best answer; it combines elements from multiple answers to produce a superior composite.
- MetaGPT assigns specialized managers to control distinct functional roles and phases in software development, simulating real-world organizational hierarchies. The controller delegates to sub-controllers (managers), creating a multi-level hierarchy.

**Differentiation-based systems** achieve centralized control through a fundamentally different mechanism: rather than having a separate controller module, they "implicitly differentiate the meta-agent into sub-agents through carefully crafted prompts." A single model serves as the orchestrator, dynamically assuming different roles based on task-oriented prompts:

- AutoAct differentiates the meta-agent into three sub-agents—plan-agent, tool-agent, and reflect-agent—to decompose the ScienceQA task. Each sub-agent is the same underlying model but prompted with different role instructions, which the paper describes as using "prompts to guide the meta agent in assuming distinct sub-roles."
- Meta-Prompting decomposes complex tasks into domain-specific subtasks through meta-prompts, where a single model acts as a coordinator. The coordinator dynamically assigns subtasks to specialized sub-agents (again, the same model with different prompts), then integrates all intermediate outputs.

A notable finding is that "even controllers with limited discriminative power can also significantly enhance the overall performance of agent systems," as demonstrated by WJudge. This suggests that the **structure** of centralized control (having a dedicated coordination layer) provides benefits independent of the controller's individual capability.

The paper identifies centralized control as particularly suited for "mission-critical scenarios requiring strict coordination, such as industrial automation and scientific research," where the predictability and accountability of a single control point outweigh the flexibility costs.

##### Decentralized Collaboration: Self-Organizing Peer Interaction

Decentralized collaboration "enables direct node-to-node interaction through self-organizing protocols," removing the single-controller bottleneck that "often becomes a bottleneck due to handling all inter-agent communication, task scheduling, and contention resolution." The paper distinguishes two fundamentally different approaches to how agents interact without central coordination.

**Revision-based systems** operate through a constrained interaction pattern: "agents only observe finalized decisions generated by peers and iteratively refine a shared output through structured editing protocols." The key constraint is that agents interact through a **shared artifact** rather than through open-ended dialogue. This constraint is deliberate: "this approach typically produces more standardized and deterministic outcomes."

- MedAgents employs predefined domain-specific expert agents that "sequentially propose and modify decisions independently, with consensus achieved through final voting." Each expert revises the shared decision in turn, adding their domain perspective, but the interaction is structured as sequential editing rather than simultaneous debate.
- ReConcile coordinates agents to iteratively refine answers through "mutual response analysis, confidence evaluation, and human-curated exemplars." Agents don't just edit the answer; they analyze each other's responses and express confidence, creating a richer interaction while maintaining the revision structure.
- METAL introduces specialized text and visual revision agents for chart generation, demonstrating domain-specific refinement. The revision signal in DS-Agent originates not just from agent interactions but from external knowledge bases, showing that revision-based systems can incorporate non-agent feedback sources.

**Communication-based systems** feature "more flexible organizational structures, allowing agents to directly engage in dialogues and observe peers' reasoning processes." This makes them "particularly suitable for modeling dynamic scenarios such as human social interactions."

- MAD employs "structured communication protocols to address the 'degeneration-of-thought' problem, where agents overly fixate on initial solutions." The protocol structures debate to prevent premature convergence—agents are required to consider alternatives even when consensus seems near.
- MADR extends this by enabling agents to "critique implausible claims, refine arguments, and generate verifiable explanations for fact-checking." The debate is grounded in evidence, not just opinion exchange.
- MDebate optimizes consensus-building through "strategic alternation between stubborn adherence to valid points and collaborative refinement." This reflects a nuanced understanding of effective debate: agents should be stubborn when they have evidence but flexible when they don't.
- AutoGen implements a group-chat framework supporting multi-agent participation in iterative debates, demonstrating that communication-based systems can scale beyond pairwise interactions.

The fundamental difference between revision-based and communication-based systems is **information visibility**: in revision-based systems, agents see only the final output of peers' reasoning; in communication-based systems, they observe the reasoning process itself. This has implications for learning, trust calibration, and error detection.

##### Hybrid Architecture: Combining Control and Flexibility

Hybrid architectures "strategically combine centralized coordination and decentralized collaboration to balance controllability with flexibility, optimize resource utilization, and adapt to heterogeneous task requirements." The paper identifies two implementation strategies based on **when** the combination pattern is determined.

**Static systems** predefine fixed patterns for combining collaboration modalities:

- CAMEL partitions agents into intra-group decentralized teams for role-playing simulations while maintaining inter-group coordination through centralized governance. This creates a two-level structure: within a group, agents interact freely; between groups, coordination is structured.
- AFlow employs a three-tier hierarchy: "centralized strategic planning, decentralized tactical negotiation, and market-driven operational resource allocation." This maps to different levels of decision-making, with centralization at the strategic level (where consistency matters) and decentralization at the operational level (where flexibility matters).
- EoT formalizes four collaboration patterns—BUS, STAR, TREE, RING—aligning network topologies with specific task characteristics. This is a **topology-as-design-choice** perspective: different communication structures are optimal for different problem types.

**Dynamic systems** introduce a significant innovation: "neural topology optimizers that dynamically reconfigure collaboration structures based on real-time performance feedback, enabling automatic adaptation to changing conditions."

- DiscoGraph introduces "trainable pose-aware collaboration through a teacher-student framework." The teacher model with holistic-view inputs guides the student model via feature map distillation, while matrix-valued edge weights enable adaptive spatial attention across agents. The collaboration graph itself becomes a learned structure.
- DyLAN first utilizes "the Agent Importance Score to identify the most contributory agents and then dynamically adjusts the collaboration structure to optimize task completion." This is a **pruning** approach: identify which agents are actually helping and restructure to emphasize them.
- MDAgents dynamically assigns collaboration structures based on task complexity assessment. "Simple tasks are handled by a single agent, while more complex tasks are addressed through hierarchical collaboration." This is a **routing** approach: the collaboration topology is the output of a classification decision.

The dynamic approach represents a meta-level capability: the system learns to organize itself. This connects collaboration to evolution—the ability to dynamically reconfigure collaboration structures is itself a capability that can improve over time.

---

#### 3.4.3 Agent Evolution: Mechanisms for Improvement Over Time

The paper presents evolution as the third dimension of its taxonomy, capturing how agents improve through learning, interaction, and external resource integration. The three subcategories—autonomous optimization, multi-agent co-evolution, and external resource evolution—are distinguished by **where the improvement signal comes from**: within the agent, from other agents, or from external sources.

##### Autonomous Optimization and Self-Learning

This category encompasses mechanisms where an agent improves using internally generated signals, reducing or eliminating dependence on external supervision. The paper identifies three sub-mechanisms of increasing sophistication.

**Self-supervised learning and adaptive adjustment** uses unlabeled or internally generated data:

- Self-evolution learning (SE) enhances pretraining by "dynamically adjusting token masking and learning strategies." Instead of applying a fixed masking pattern, the model learns which tokens to mask and how aggressively, adapting the self-supervision task to its current capabilities.
- Evolutionary optimization techniques facilitate "efficient model merging and adaptation, improving performance without extensive additional resources." Model merging—combining weights from multiple trained models—is treated as an optimization problem, with evolutionary algorithms searching for optimal merge coefficients.
- DiverseEvol refines instruction tuning by "improving data diversity and selection efficiency." The model selects which training examples to learn from, prioritizing those that provide the most information gain.

**Self-reflection and self-correction** enables LLMs to "iteratively refine their outputs by identifying and addressing errors":

- SELF-REFINE applies iterative self-feedback to improve generated responses without external supervision. The key mechanism: the model generates an output, then generates feedback on that output, then uses the feedback to generate an improved output. The cycle repeats until quality stabilizes.
- STaR (Self-Taught Reasoner) bootstraps reasoning by generating rationales, filtering those that lead to correct answers, and fine-tuning on the successful rationales. This is a form of **self-curated learning**: the model generates its own training data and selects the high-quality subset.
- V-STaR extends this by training a verifier to discriminate between correct and incorrect self-generated solutions, then using the verifier to guide further improvement. This adds a **self-assessment capability** that makes the bootstrapping process more reliable.
- Self-verification techniques "enable models to retrospectively assess and correct their outputs." The model checks its own work—verifying logical consistency, arithmetic correctness, factual accuracy—and flags issues for correction.

**Self-rewarding and reinforcement learning** uses internally generated reward signals:

- Self-rewarding language models use "LLM-as-a-Judge" to generate reward signals, then use those rewards to improve through preference optimization. The model serves as both the policy to be improved and the judge that evaluates improvements.
- RLCD uses contrastive distillation for alignment through self-rewarding, learning to distinguish between aligned and misaligned outputs.
- RLC leverages "the evaluation-generation gap via reinforcement learning strategies"—the difference between how the model evaluates outputs and what it actually generates provides a learning signal.

The progression from self-supervised learning to self-reflection to self-rewarding represents increasing levels of **metacognitive capability**: first the model learns from data it generates, then it learns to critique what it generates, then it learns to assign value to what it generates and optimize accordingly.

##### Multi-Agent Co-Evolution

This category captures how agents improve through interaction with other agents. The paper identifies two modes: cooperative and competitive.

**Cooperative and collaborative learning** enhances agents through knowledge sharing and joint problem-solving:

- ProAgent enables LLM-based agents to "adapt dynamically in cooperative tasks by inferring teammates' intentions and updating beliefs." This requires theory of mind—modeling what other agents know and intend—which creates a co-evolutionary dynamic: as each agent improves its model of others, the collective coordination improves.
- CORY extends RL fine-tuning into a cooperative multi-agent framework where "LLMs iteratively improve through role-exchange mechanisms." Agents take turns playing different roles, learning both from their own experience and from observing others' behavior in roles they will later occupy.
- CAMEL's role-playing framework enables "communicative agents to collaborate autonomously using inception prompting." The key mechanism is that agents assign each other roles and tasks, creating a self-organizing collaborative structure.

**Competitive and adversarial co-evolution** drives improvement through challenge:

- Red-team LLMs "dynamically evolve in adversarial interactions, continuously challenging LLMs to uncover vulnerabilities and mitigate mode collapse." The attacker and defender co-evolve: as defenses improve, attacks must become more sophisticated; as attacks improve, defenses must adapt. This creates a natural curriculum of increasing difficulty.
- Multi-agent debate (Du et al.) enhances reasoning by having "multiple LLMs critique and refine each other's arguments over multiple rounds, improving factuality and reducing hallucinations." The competitive dynamic—each agent trying to find flaws in others' arguments—drives collective improvement.
- MAD structures debates "in a tit-for-tat manner, encouraging divergent thinking and refining logical reasoning in complex tasks." The tit-for-tat structure prevents debates from becoming either too adversarial (where agents refuse to concede valid points) or too cooperative (where agents converge prematurely).

The paper characterizes competitive co-evolution as driving LLMs to "develop stronger reasoning, resilience, and strategic adaptability." The underlying principle is that **adversarial pressure creates a need for robustness** that purely cooperative training cannot provide.

##### Evolution via External Resources

This category captures how agents improve by incorporating structured information and feedback from outside the agent system.

**Knowledge-enhanced evolution** integrates structured external knowledge:

- KnowAgent improves LLM-based planning by "integrating action knowledge, constraining decision paths, and mitigating hallucinations." The external knowledge acts as a constraint—certain actions are ruled out, certain strategies are preferred—reducing the search space and improving plan reliability.
- The World Knowledge Model (WKM) enhances agent planning by "synthesizing expert and empirical knowledge, providing global priors and dynamic local knowledge to guide decision-making." Global priors provide general strategies applicable across tasks; dynamic local knowledge adapts to specific situations.

**External feedback-driven evolution** uses real-time feedback signals:

- CRITIC allows LLMs to "validate and revise their outputs through tool-based feedback." When the agent generates an output, it uses external tools (calculators, databases, search engines) to verify correctness, then revises based on discrepancies.
- STE enhances tool learning by "simulating trial-and-error, imagination, and memory." The agent doesn't just execute tools—it imagines possible outcomes, tries approaches, learns from failures, and remembers what worked.
- SelfEvolve adopts a two-step framework where "LLMs generate and debug code using feedback from execution results, enhancing performance without human intervention." The execution environment provides objective feedback (code runs or doesn't; produces correct output or doesn't), which the agent uses for self-improvement.

The unifying principle across all evolution mechanisms is **feedback loop closure**: the agent produces outputs, receives signals about output quality (from itself, from other agents, or from external sources), and uses those signals to improve future outputs. The mechanisms differ in the **source, granularity, and reliability** of the feedback signal.

---

#### 3.4.4 Evaluation Benchmarks and Tools

The paper treats evaluation and tools as integral to the agent ecosystem, not as separate concerns. The evaluation section (Section 3.1) addresses **how agent capabilities are measured**, while the tools section (Section 3.2) addresses **what infrastructure agents use, create, and are deployed on**.

##### Evaluation: From Static Metrics to Dynamic Assessment

The paper identifies an evolution in evaluation methodology driven by the increasing complexity of agent systems:

**General assessment frameworks** have moved beyond simple success-rate metrics:

- **Multi-dimensional capability assessment** dissects agent intelligence across reasoning, planning, and problem-solving dimensions. AgentBench tests across eight interactive environments; MMAU "enhances explainability through granular capability mapping and breaks down agent intelligence into five core competencies by more than 3,000 cross-domain tasks." This reflects a shift from "does the agent succeed?" to "in what specific ways is the agent competent or deficient?"
- **Dynamic and self-evolving evaluation** addresses baseline obsolescence. BENCHAGENTS automatically creates benchmarks through LLM agents, enabling rapid capacity expansion. Benchmark self-evolving introduces six refactoring operations to "dynamically generate test instances for short-cut biases"—preventing agents from exploiting benchmark-specific patterns rather than demonstrating genuine capability. Revisiting Benchmark proposes TestAgent with reinforcement learning for domain-adaptive assessment.

**Domain-specific evaluation** tailors assessment to specialized knowledge:

- MedAgentBench contains tasks designed by 300 clinicians in an FHIR-compliant environment for healthcare evaluation.
- AI Hospital simulates clinical workflows through multi-agent collaboration.
- LaMPilot connects LLM to autonomous driving architecture through code generation benchmarks.
- TravelPlanner provides a sandbox with 1,225 planning tasks requiring multi-step reasoning, tool integration, and constraint balancing.

**Collaborative evaluation** quantifies emergent coordination patterns:

- TheAgentCompany pioneered enterprise-level assessments using simulated software company environments.
- MLE-Bench evaluates Kaggle-style model engineering through 71 real-world competitions.
- MLRB designs 7 competition-level ML research tasks specifically for multi-agent collaboration assessment.

##### Tools: A Three-Layer Ecosystem

The paper maps tools into three layers based on their relationship to the agent:

**Tools used by agents** are external capabilities that agents invoke:

- **Knowledge retrieval tools** (WebGPT, WebCPM) provide real-time information access.
- **Computation tools** (Python interpreters, calculators) provide precise calculation capabilities.
- **API interaction tools** (RestGPT, GraphQLRestBench) enable external service orchestration.

**Tools created by agents** represent a meta-capability:

- CRAFRT collects GPT-4 code solutions and abstracts them into reusable code snippets.
- CREATOR proposes a four-phase framework—Creation, Decision, Execution, and Reflection—for tool creation.
- LATM proposes a two-stage framework with separate tool maker and tool user roles, plus a tool caching mechanism.

**Tools for deploying agents** support the operational lifecycle:

- **Productionization tools** (AutoGen, LangChain, LlamaIndex, Dify) enable building and deploying agent applications.
- **Operation and maintenance tools** (Ollama, Dify monitoring) support ongoing reliability.
- **Model Context Protocol (MCP)** standardizes how applications provide context to LLMs, enabling interoperability across tools and data sources.

---

#### 3.4.5 Security, Privacy, and Social Impact

The paper treats real-world issues as a first-class dimension of the taxonomy, organized into three categories.

##### Agent-Centric Security: Attacks on the Model Itself

These attacks target "weights, architecture, and inference process of the agent models." The paper identifies four attack types and corresponding defenses:

**Adversarial attacks** aim to compromise agent reliability. Attack methods include GIGA (generalizable infectious gradient attacks that propagate across multi-agent systems) and CheatAgent (adversarial perturbations for recommender systems). Defense methods include LLAMOS (purifying adversarial inputs) and multi-agent debate for robustness.

**Jailbreaking attacks** attempt to bypass model protections. Attack methods include RLTA (reinforcement learning for malicious prompt generation), Atlas (text-to-image jailbreaking with mutation and selection agents), RLbreaker (black-box jailbreaking via deep RL), and PathSeeker (multi-agent RL for input modification). Defense methods include AutoDefense (multi-agent filtering), Guardians (rogue agent detection), and ShieldLearner (autonomous defense heuristic learning).

**Backdoor attacks** implant triggers for preset errors. Attack methods include DemonAgent (dynamically encrypted multi-backdoor implantation), BadAgent (input/environment triggers), BadJudge (backdoor in LLM-as-judge systems), and DarkMind (latent backdoor exploiting reasoning chains without trigger injection in user inputs).

**Model collaboration attacks** target multi-agent interactions. Attack methods include CORBA (contagious recursive blocking disrupting agent communications), AiTM (intercepting and manipulating inter-agent messages), and prompt infection (self-replicating across agent networks). Defense methods include Netsafe (identifying safety-critical topological properties), G-Safeguard (graph neural network anomaly detection), TrustAgent (constitution-based planning safety), and PsySafe (psychological assessment and policing agents).

##### Data-Centric Security: Attacks on Input Data

These attacks "contaminate the input data" without modifying model components, categorized by the data type targeted:

**External data attacks** manipulate information entering the system:

- User input falsifying (malicious prefix prompts, prompt injection benchmarks like InjectAgent and AgentDojo) achieves the highest attack success rate and is defended against through input firewalls and sandwich defense strategies.
- Dark psychological guidance injects antisocial prompts; defense uses doctor and police agents for psychological monitoring.
- External source poisoning targets RAG-based agents by injecting malicious content into knowledge databases; defense employs multi-agent debate for factuality verification.

**Interaction attacks** exploit communication channels:

- User-agent interaction attacks extract private memory; defense uses multi-layer agent firewalls with trajectory verification.
- Multi-agent interaction attacks poison one agent to infect others; defense uses blockchain-based consensus (BlockAgents).
- Agent-tool interaction attacks manipulate planning to cause harmful tool calls; defense uses trajectory correction mechanisms.

##### Privacy: Memorization and Intellectual Property

**LLM memorization vulnerabilities** include data extraction attacks (extracting PII from training data), member inference attacks (determining if data was in training), and attribute inference attacks (inferring sensitive attributes). Defenses include data cleaning, differential privacy, and knowledge distillation.

**LLM intellectual property exploitation** includes model stealing attacks (extracting parameters via query APIs) and prompt stealing attacks (inferring original prompts from generated content). Defenses include model watermarking and blockchain-based IP authentication.

##### Social Impact and Ethics

The paper balances **benefits** (automation enhancement, workforce transformation, enhanced information distribution) against **ethical concerns** (bias and discrimination, accountability, copyright, overreliance, environmental impact). The treatment integrates security, privacy, and ethics into the same framework, arguing that "understanding these challenges is crucial for developing robust, trustworthy agent systems."

## 4. Key Insights and Innovations

### Innovation 1: The "Build-Collaborate-Evolve" Framework as a Unified Design Space

The paper's most fundamental intellectual move is not any single classification category but the **meta-architectural claim** that agent construction, collaboration, and evolution are not independent research areas—they are interdependent dimensions of a single, unified design space. This reframing changes how someone thinks about the relationship between individual and collective agent behavior.

**What the field did before.** Prior surveys and research traditions treated single-agent architecture and multi-agent coordination as largely separate topics. Work on "agent construction" examined internal mechanisms—profiles, memory, planning, tools—as if the agent operated in isolation. Work on "multi-agent systems" studied coordination protocols, communication topologies, and consensus mechanisms without connecting these back to the internal design of the participating agents. The authors explicitly identify this separation in Section 1: "prior studies have often examined these aspects separately." The implication was that you could design an agent first, then plug it into a collaboration framework later—as if construction and collaboration were sequential rather than co-constraining.

**What makes this reframing distinctive.** The paper argues that this separation is artificial and limiting by revealing **bidirectional constraints** between the dimensions. Construction choices enable or constrain collaboration patterns: an agent with only short-term memory (Section 2.1.2) cannot effectively participate in long-running multi-agent workflows; an agent built with a human-curated static profile produces deterministic, role-constrained interactions, while one with a batch-generated dynamic profile produces emergent, personality-driven interactions. Conversely, collaboration patterns drive evolutionary dynamics: debate among agents (Section 2.3.2) produces improved outputs that can be distilled back into individual models; competitive co-evolution creates adversarial pressure that reshapes individual agent capabilities. Evolution, in turn, reshapes construction: self-improved agents (Section 2.3.1) have capabilities their initial versions lacked, enabling new forms of collaboration that were previously impossible.

The significance of this reframing is **theoretical rather than empirical**. The paper does not present a new algorithm that outperforms baselines—it provides a coordinate system for a design space that was previously navigated by intuition. The taxonomy enables systematic questions that were difficult to even formulate before: "Given a specific memory architecture, what collaboration topologies are feasible?" "What evolutionary mechanisms are needed to transition from decentralized to hybrid coordination?" These are design-space questions that the taxonomy makes explicit.

**Evidence anchoring.** The three-dimensional structure of Figure 2 and Table 2—which maps evolution methods to their categories—are direct instantiations of this framework. The taxonomy's ability to accommodate diverse systems (from Voyager's skill libraries to CAMEL's role-playing to CRITIC's tool-based self-correction) within a single organizational scheme is the primary evidence that the unified design space is coherent and productive.

---

### Innovation 2: Treating Evolution as a First-Class Dimension Co-Equal with Construction and Collaboration

The paper's decision to elevate "evolution" (Section 2.3) to the same taxonomic level as construction and collaboration is a deliberate departure from prior surveys that either ignored temporal improvement mechanisms or treated them as a minor subcategory. This choice reflects a specific conceptual claim: that **agent systems are not static artifacts deployed once but dynamic entities that change over time**, and that this temporal dimension must be part of the core taxonomy, not an afterthought.

**What the field did before.** Most prior surveys on LLM agents focused on static architectures—what the agent *is* at a moment in time. If they addressed improvement mechanisms at all, these were typically treated as a secondary topic: "fine-tuning approaches" or "training methods" appended to a primarily architectural taxonomy. This reflected an implicit assumption that the interesting design decisions happen at construction time, and that evolution is merely an optimization step applied afterward. Even work on self-improvement (STaR, Self-Refine) and multi-agent debate was typically framed as a method for improving task performance rather than as a fundamental dimension of agent architecture.

**What makes this elevation distinctive.** By placing evolution alongside construction and collaboration, the paper makes a stronger claim: **how an agent improves is as architecturally significant as how it is initially built**. The three subcategories of evolution—autonomous optimization (Section 2.3.1), multi-agent co-evolution (Section 2.3.2), and evolution via external resources (Section 2.3.3)—are not merely a catalog of improvement techniques but a taxonomy of **feedback loop structures**. Each subcategory defines a different source, granularity, and reliability of the improvement signal:

- Autonomous optimization uses internally generated signals (self-generated rationales, self-verification, self-rewarding), creating a closed loop where the agent is both learner and teacher.
- Multi-agent co-evolution uses peer-generated signals (cooperative knowledge sharing, competitive critique), creating an open loop where improvement emerges from interaction dynamics.
- External resource evolution uses environment-generated signals (tool feedback, knowledge base integration), creating a grounded loop where improvement is anchored to objective verification.

The progression reveals a **spectrum of epistemic autonomy**: from purely internal feedback (where the agent must trust its own judgments) to external verification (where improvement is constrained by objective correctness). This spectrum is itself a design dimension—choosing where on this spectrum to operate is a fundamental architectural decision with implications for reliability, adaptability, and scalability.

This reframing also connects agent evolution to broader themes in AI: the self-improvement narrative (models bootstrapping their own capabilities), the emergence of collective intelligence (multi-agent systems exhibiting behaviors not present in individuals), and the grounding problem (how agents connect internal representations to external reality). By treating all three as manifestations of a single evolutionary dimension, the taxonomy reveals structural similarities across what appeared to be disparate research threads.

**Evidence anchoring.** Table 2's systematic categorization of evolution methods—self-supervised learning, self-reflection/correction, self-rewarding/RL, cooperative co-evolution, competitive co-evolution, knowledge-enhanced evolution, and feedback-driven evolution—demonstrates that the evolutionary dimension can accommodate diverse mechanisms while maintaining internal coherence. The fact that methods from entirely different research communities (RL-based self-improvement, debate-driven refinement, tool-based correction) can be placed in the same taxonomy and compared along common dimensions is the primary evidence that evolution deserves first-class status.

---

### Innovation 3: The Difficulty-Dependent Nature of Test-Time Compute Allocation as a Diagnostic Finding, Not Just a Method

While the prior sections document the paper's taxonomic innovations, a deeper insight emerges from its treatment of **real-world evaluation and dynamic assessment** (Section 3.1.1, Section 6.4). The paper identifies that evaluation frameworks for LLM agents face a fundamental challenge that goes beyond metric design: **static benchmarks cannot capture the dynamic, multi-turn, multi-agent behaviors that define real agent systems**, and this is not merely a practical limitation but a diagnostic finding about the nature of agent capability.

**What the field did before.** Prior evaluation of LLM agents largely followed the pattern established for language models: static datasets, single-turn tasks, success-rate metrics. Benchmarks like AgentBench (eight interactive environments) and Mind2Web (137 real-world websites) represented progress toward more realistic evaluation, but they still measured agent performance as a snapshot—can the agent complete task X in environment Y? The underlying assumption was that capability is a property of the agent that can be measured at a point in time, and that static benchmarks provide a valid sample of the capability distribution.

**What makes this diagnostic distinctive.** The paper identifies a deeper structural problem: agent capabilities in multi-turn, multi-agent settings are not static properties but **emergent, context-dependent, and temporally extended**. As the authors state in Section 6.4:

> "Current benchmarks primarily assess task execution such as code completion and dialogue generation in isolated settings, overlooking emergent agent behaviors, long-term adaptation, and collaborative reasoning that unfold across multi-turn interactions."

This is not just a call for better benchmarks—it is a claim about the **ontology of agent capability**. The paper argues that capabilities like "adaptation to changing collaboration topologies" or "recovery from communication failures" or "accumulation of shared knowledge across multiple interactions" are not merely harder to measure than single-task performance—they are **qualitatively different kinds of capabilities** that cannot be reduced to or predicted from single-task metrics.

This connects to the dynamic and self-evolving evaluation paradigms the paper documents (Section 3.1.1): BENCHAGENTS automatically creating benchmarks through LLM agents, benchmark self-evolving introducing refactoring operations to prevent shortcut biases, Revisiting Benchmark using reinforcement learning for domain-adaptive assessment. These represent a shift from **measuring capability** to **probing capability boundaries**—evaluation becomes an interactive process where the benchmark adapts to the agent, revealing not just what the agent can do but where and how it fails.

The paper also identifies a specific failure mode of static evaluation that has received insufficient attention: **data contamination**, where "model performance may stem from memorization rather than genuine reasoning" (Section 6.4). In multi-agent settings, this problem is amplified because contamination can occur across agents, through shared training data, or through memorized interaction patterns. A static benchmark cannot distinguish between an agent that genuinely coordinates and one that has memorized coordination patterns from training data—only dynamic, adversarial, or out-of-distribution evaluation can.

**Evidence anchoring.** The progression of evaluation frameworks documented in Section 3.1—from general assessment (AgentBench, MMAU) to domain-specific evaluation (MedAgentBench, TravelPlanner) to collaborative evaluation (TheAgentCompany, MLE-Bench) to dynamic evaluation (BENCHAGENTS, benchmark self-evolving)—is itself evidence of the field's recognition that static metrics are insufficient. The paper's identification of this progression as a structured research challenge rather than a collection of independent efforts is the diagnostic contribution.

---

### Innovation 4: Security and Privacy as First-Class Architectural Dimensions, Not External Concerns

The paper's treatment of security, privacy, and social impact (Section 4) as **integral components of the agent taxonomy** rather than as a separate "ethics" or "future work" section represents a substantive intellectual position: that security vulnerabilities are not accidents or implementation details but are **inherent consequences of specific architectural choices**, and that the taxonomy itself can serve as a diagnostic tool for identifying where vulnerabilities are likely to arise.

**What the field did before.** Prior surveys on LLM agents typically either omitted security entirely or treated it as a separate topic disconnected from agent architecture. When security was addressed, it was usually in the form of a catalog of threats without systematic connection to design choices—"here are the attacks, here are some defenses"—rather than an analysis of how architectural decisions create or mitigate vulnerabilities. This reflected an implicit assumption that security is a deployment concern to be addressed after the architecture is designed, not a design constraint that should shape architecture from the start.

**What makes this integration distinctive.** The paper structures its security analysis to mirror the agent taxonomy itself, creating direct **causal links between architectural components and vulnerability classes**:

- **Agent-centric attacks** (Section 4.1) target the model itself—weights, architecture, inference—and map to the construction dimension of the taxonomy. Adversarial attacks, jailbreaking, backdoors, and model collaboration attacks are all threats that emerge from specific construction choices (how profiles are defined, how planning works, what action capabilities exist). The paper's categorization of attacks by their target (adversarial, jailbreaking, backdoor, collaboration) corresponds to different construction components being exploited.

- **Data-centric attacks** (Section 4.2) target input data—external sources, user interactions, inter-agent communication—and map to the memory and retrieval components of construction as well as the communication patterns of collaboration. External data attacks exploit retrieval mechanisms; interaction attacks exploit communication topologies. The attack taxonomy reveals that different memory architectures (short-term, long-term, retrieval-based) and different collaboration architectures (centralized, decentralized) create different vulnerability surfaces.

- **Privacy threats** (Section 4.3)—memorization vulnerabilities and intellectual property exploitation—map to the evolution and memory dimensions. Data extraction attacks exploit what the agent has memorized from training; model stealing attacks exploit what can be inferred from the agent's outputs. These threats are exacerbated in multi-agent settings where information flows between agents create new extraction surfaces.

This mapping is not merely organizational—it is **diagnostically useful**. A designer considering a specific memory architecture (e.g., long-term experience repositories) can consult the taxonomy to identify the class of data-centric attacks that become relevant, and can then select defenses appropriate to that class. A researcher developing a new collaboration topology can identify which model collaboration attacks apply and design the topology to be robust against them. The taxonomy transforms security from a checklist of threats to a structured design constraint.

The inclusion of social impact and ethics (Section 4.4) in the same framework as technical security extends this logic: bias, accountability, copyright, and environmental concerns are not separate "ethical" issues but are **architectural consequences** of how agents are constructed (what training data they use), how they collaborate (who controls the interaction), and how they evolve (what feedback signals drive improvement). By treating these alongside adversarial attacks and jailbreaking, the paper asserts that ethical harms are security harms—just with different threat models and different affected parties.

**Evidence anchoring.** Tables 3, 4, and 5 provide the structured mapping between attack types, defense methods, and references that operationalize this architectural-security connection. The categorization of attacks into agent-centric (Table 3) and data-centric (Table 4) with corresponding defense strategies demonstrates that security analysis can be systematically derived from the architectural taxonomy rather than compiled ad hoc. The paper's ability to place a method like PsySafe (which uses doctor and police agents to guard psychological health) and a method like BlockAgents (which uses blockchain consensus against Byzantine attacks) in the same framework—despite their entirely different mechanisms—validates the claim that the taxonomy provides a unified security architecture.

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** The survey paper does not conduct original experiments with a single dataset. Instead, it surveys hundreds of papers, each using its own datasets. The paper catalogs evaluation benchmarks across three categories (Section 3.1): general assessment frameworks (AgentBench with eight interactive environments, MMAU with 3,000+ cross-domain tasks, Mind2Web with 137 real-world websites spanning 31 domains, BLADE for scientific discovery), domain-specific evaluation systems (MedAgentBench with tasks designed by 300 clinicians, TravelPlanner with 1,225 planning tasks, OSWorld with 369 multi-application tasks across Ubuntu/Windows/macOS, AgentHarm with 440 malicious agent tasks in 11 hazard categories, MLE-Bench with 71 real-world Kaggle competitions), and collaborative evaluation (TheAgentCompany simulating software company environments, MLRB with 7 competition-level ML research tasks). The survey does not present unified experimental results aggregating across these diverse benchmarks.

**Base model(s).** The survey covers research using diverse model families—GPT-4, GPT-3.5, PaLM 2, LLaMA, and various open-source models—without conducting original experiments on a specific base model. The paper's contribution is taxonomic rather than empirical, so there is no single base model that anchors its claims. The authors note in Section 4 that they "believe this model is representative of the capabilities of many contemporary LLMs" when discussing PaLM 2-S* in a specific referenced work, but this is not presented as their own experimental choice.

**Metrics.** No unified metric is applied across all surveyed work. Individual papers referenced in the survey use diverse evaluation criteria: task success rate (AgentBench, OSWorld), accuracy on domain-specific questions (MedAgentBench, ClinicalLab), pass@k for code generation (MLE-Bench), attack success rate for security evaluations (AgentDojo, ASB), and qualitative assessments of agent behaviors (Generative Agents). The survey itself aggregates and categorizes these metrics rather than computing any of its own.

**Baselines.** The paper is a literature survey and does not establish experimental baselines in the traditional sense. It compares and contrasts different methodological approaches to agent construction, collaboration, and evolution, but does not run head-to-head experiments. When individual works in the survey compare against baselines, those baselines are specific to each paper (e.g., majority voting vs. verifier-based selection, single-agent vs. multi-agent performance, static profiles vs. dynamic profiles).

**Generation budget / compute accounting.** No unified compute accounting framework is applied across the surveyed papers. Individual works use different measures: number of sampled solutions, wall-clock time, API calls, or FLOPs. The survey does not attempt to normalize these into a common metric, nor does it present original experiments requiring compute accounting.

**Cross-validation / statistical protocol.** The survey does not report any statistical protocol for aggregating results across papers. Individual papers referenced in the survey employ their own validation strategies, but these are not systematically compared or meta-analyzed. The survey's contribution is qualitative, organizing research into a taxonomy rather than quantitatively synthesizing experimental findings.

### Main Quantitative Results

The survey paper does **not present original quantitative results**. It is a taxonomic survey that organizes, categorizes, and relates existing work in the LLM agent literature. There are no figures, tables, or charts containing newly computed performance numbers, no head-to-head comparisons run by the authors, and no experimental results generated for this paper. The "results" in this survey are the **taxonomic structure itself**—the demonstration that hundreds of papers can be coherently organized into the Build-Collaborate-Evolve framework, and that doing so reveals structural relationships between research areas previously treated as separate.

This is a fundamental difference from the reference example (which analyzed an empirical paper with specific quantitative results anchored to figures and tables). The survey's contribution is **conceptual and organizational**, not empirical. The taxonomy is the result.

### Ablation Studies and Robustness Checks

The survey does not conduct ablation studies or robustness checks in the experimental sense. However, the paper implicitly validates its taxonomy through a form of **coverage testing**: by demonstrating that the framework can accommodate diverse research threads—from self-supervised learning (SE) to blockchain-based agent coordination (BlockAgents) to psychological safety monitoring (PsySafe)—the paper provides evidence that the taxonomy is sufficiently general and well-structured. The closest analog to a robustness check is the paper's ability to place methods from entirely different research communities (robotics, game-playing, scientific discovery, software engineering) into the same categorical system without forcing or distortion. This conceptual coverage is the survey's primary form of validation, though the paper does not formalize or quantify this coverage.

### Critical Assessment

Since this is a taxonomic survey rather than an empirical paper, traditional critical assessment—"do the experiments support the claims?"—must be reframed: **does the taxonomy actually deliver on its stated goals, and what are its limitations as an analytical framework?**

**On the claim of "methodology-centered taxonomy."** The paper delivers this in structural terms: Figure 2 and the accompanying text in Section 2 do indeed decompose agent systems into profile, memory, planning, and action components, and the collaboration and evolution sections extend this decomposition to multi-agent and temporal dimensions. The taxonomy is applied consistently throughout. However, the survey does not demonstrate that this taxonomy is **more useful** than alternative organizations (e.g., by application domain, by model architecture, by chronological development). The claim that a methodology-centered taxonomy provides better understanding is asserted rather than demonstrated. A reader committed to an application-centric view—where what matters is whether an agent works for healthcare versus gaming—would find the paper's organization frustrating, as domain-specific insights are scattered across methodological categories. The paper does not provide evidence that its chosen organization leads to novel insights that other organizations would miss, beyond stating that it reveals "fundamental connections."

**On the claim of a "unified architectural perspective."** The Build-Collaborate-Evolve framework is the paper's central intellectual contribution, and it is presented coherently. The recursive dependencies between layers (construction constrains collaboration, collaboration drives evolution, evolution reshapes construction) are logically articulated. However, the paper does not provide empirical evidence that these dependencies operate as described. The claim that profile definition affects collaboration patterns, or that memory architecture constrains multi-agent coordination, is theoretically plausible but not demonstrated through controlled experiments or systematic case studies comparing agents with different construction choices in identical collaboration settings. The "unified perspective" is a conceptual framework, not an empirically validated model of how agent systems actually behave.

**On the claim of addressing "fragmented research threads."** The survey succeeds in bringing together work from diverse communities—robotics, game AI, scientific computing, software engineering, security—under a single organizational scheme. This is a genuine service to the field. However, the paper's treatment is necessarily broad rather than deep: each of the hundreds of cited works receives a sentence or two of description, and the connections drawn between them are at the level of "this is also a form of memory" or "this is also a form of planning" rather than detailed technical comparison. The fragmentation addressed is **bibliographic**—papers are now grouped together—but the deeper fragmentation of **technical detail** (how exactly does Voyager's skill library compare to GITM's knowledge base in terms of retrieval mechanisms, update policies, and forgetting behavior?) is not addressed. The survey provides a map of the territory but not detailed topographical analysis.

**On the claim of identifying "frontier applications and real-world focus."** The paper's inclusion of security, privacy, social impact, and practical tools (Sections 3.2, 4) does distinguish it from surveys that focus exclusively on algorithmic contributions. The categorization of security threats by architectural component (agent-centric vs. data-centric) is a genuine analytic contribution that connects threats to design choices. However, the coverage of these topics lacks the systematic depth of the core methodology sections. Social impact (Section 4.4) is covered in less than two pages, citing broad surveys and position papers rather than engaging with specific findings or debates in the AI ethics literature. The paper identifies that bias, accountability, and copyright are concerns, but does not provide the kind of structured analysis (e.g., mapping specific bias mechanisms to specific architectural components) that it provides for security threats.

**On the absence of quantitative synthesis.** The most significant limitation of this survey as an analytical tool is that it makes no attempt to synthesize quantitative findings across papers. The survey cannot answer questions like: "Which memory architecture provides the largest improvement on multi-turn task completion?" or "How does the performance gap between centralized and decentralized collaboration scale with the number of agents?" or "What is the typical improvement from adding self-reflection to a base agent?" A meta-analysis or even a systematic tabulation of reported performance numbers across papers would have substantially increased the survey's practical utility. The taxonomy tells researchers **what** has been studied and **how** it relates to other work, but provides no guidance on **how well** different approaches work.

**What experiments would have strengthened the paper.** The survey's validity as a taxonomy could be tested through a **design-space coverage experiment**: sample a set of agent systems not cited in the paper, attempt to classify them into the taxonomy, and measure whether the taxonomy accommodates them cleanly or requires stretching. A **researcher utility study** could assess whether the taxonomy helps practitioners diagnose failures or select architectures more effectively than ad hoc reasoning. Neither validation is performed, leaving the taxonomy as a plausible but unvalidated analytical framework. Additionally, a systematic tabulation of which architectural choices are used in which application domains would reveal whether certain design patterns cluster in certain domains—a finding that would both validate the taxonomy's utility and provide actionable guidance that the current survey does not offer.

## 6. Limitations and Trade-offs

### Limitation 1: The Taxonomy Is Unvalidated as an Analytical Framework

**The assumption or constraint.** The paper's central contribution is its Build-Collaborate-Evolve taxonomy, which the authors present as providing "a more structured taxonomy for understanding, comparing, and advancing research of LLM agents from different perspectives" (Section 1). However, the paper provides no empirical validation that this particular taxonomy is more useful, more natural, or more predictive than alternative organizations. The taxonomy is asserted to be the right way to decompose the design space, but the paper never tests whether researchers using this taxonomy actually make better design decisions, diagnose failures more accurately, or identify more promising research directions than researchers using other frameworks (e.g., organizing by application domain, by model architecture, or by chronological development).

**The consequence.** Without validation, the taxonomy remains a plausible but unverified organizational scheme. A practitioner attempting to use it to diagnose why their agent system fails may discover that the taxonomy's categories do not cleanly map to their system's actual failure modes. For example, the taxonomy treats "profile definition" and "memory mechanism" as separate construction components (Section 2.1.1, 2.1.2), but in practice, an agent's profile may be inseparable from what it remembers—its identity is constituted by its accumulated experiences. The taxonomy's decomposition may actively mislead if it encourages designers to treat as independent what are in fact deeply coupled aspects of agent behavior. The paper does not demonstrate that following its taxonomy prevents common design errors or improves outcomes.

**What evidence exists in the paper.** The paper provides no ablation study comparing its taxonomy against alternatives, no expert user study measuring whether the taxonomy improves understanding, and no systematic case study demonstrating that applying the taxonomy to analyze a specific agent system yields insights unavailable through other frameworks. The paper's evidence for the taxonomy is purely demonstrative: it shows that hundreds of papers can be categorized into its framework without obvious forcing. But "can be categorized" is much weaker than "categorizing this way is useful." Section 5 (Experimental Analysis) explicitly confirms that "the survey paper does not present original quantitative results." The taxonomy is validated only by its internal coherence and coverage, not by any external measure of utility.

**Mitigation status.** The paper does not acknowledge this as a limitation or propose validation strategies. This is notable given that the paper positions itself as addressing a practical need—"As LLM agent systems increasingly integrate into various critical domains, understanding their architectural foundations becomes essential"—yet provides no evidence that its framework improves such understanding beyond what practitioners already do ad hoc.

---

### Limitation 2: No Quantitative Synthesis Across Papers

**The assumption or constraint.** The survey makes no attempt to synthesize quantitative results across the hundreds of papers it catalogs. The paper does not compute effect sizes, meta-analyze performance improvements, or even systematically tabulate reported numbers. This is an explicit design choice: the paper is a taxonomic survey, not a meta-analysis. However, the absence of quantitative synthesis means the survey cannot answer the most practically urgent questions a practitioner would have when deciding how to build an agent system.

**The consequence.** A practitioner reading this survey to decide, for example, whether to invest in long-term memory mechanisms or multi-agent debate for their application cannot learn from this paper which approach provides larger empirical gains, under what conditions, or at what cost. The survey tells them *that* both approaches exist, *how* they are taxonomically related, and *which papers* study them—but provides no guidance on relative effectiveness. Questions like "How does the performance gap between centralized and decentralized collaboration scale with the number of agents?" or "What is the typical improvement from adding self-reflection to a base agent?" or "Does retrieval-augmented memory outperform long-term skill libraries for multi-turn task completion?" are unanswerable from this survey. The taxonomy is useful for finding relevant papers but provides no decision-support for architecture selection.

**What evidence exists in the paper.** Throughout the survey, the paper describes methods qualitatively—what they do, how they work, what category they belong to—without reporting or comparing their quantitative performance. The "Experimental Analysis" section (Section 5) explicitly states that the paper "does not present original experimental results" and that "the survey's contribution is qualitative, organizing research into a taxonomy rather than quantitatively synthesizing experimental findings." Even within individual subsections, the paper describes mechanisms without reporting their empirical impact. For example, the discussion of short-term memory (Section 2.1.2) notes that it is "widely implemented in frameworks such as ReAct, ChatDev, Graph of Thoughts, and AFlow" but provides no comparison of how much each framework's memory mechanism improves task performance relative to a memory-less baseline.

**Mitigation status.** The paper does not frame this as a limitation or suggest that future survey work should include quantitative meta-analysis. Given that the field has matured enough to produce hundreds of empirical papers, a quantitative synthesis would be both feasible and valuable. The paper's purely qualitative approach leaves a significant gap between what the survey provides (taxonomic organization) and what practitioners need (evidence-based design guidance).

---

### Limitation 3: The Taxonomy Collapses Technical Diversity Within Categories

**The assumption or constraint.** By design, the taxonomy groups methods into broad categories based on their high-level function (e.g., "long-term memory," "centralized control," "self-reflection"). This necessarily abstracts away substantial technical diversity within each category. The paper acknowledges this implicitly through its structure—each subsection covers multiple methods with brief descriptions—but does not treat this loss of resolution as a limitation.

**The consequence.** Two methods placed in the same taxonomic category may have fundamentally different properties, failure modes, and applicability conditions, yet the taxonomy treats them as equivalent instances of the same architectural pattern. For example, under "long-term memory" (Section 2.1.2), the paper groups Voyager's automated skill discovery (which generates executable programs through iterative exploration), ExpeL's distilled experience pool (which stores task outcomes with contextual features for similarity-based retrieval), and MemGPT's tiered memory architecture (which manages information movement between working memory and archival storage through explicit operating-system-like management). These are radically different technical approaches with different memory representations, update policies, retrieval mechanisms, and scalability properties. A practitioner who understands from the taxonomy that these are all "long-term memory" but not *how they differ* and *when to choose which* gains only limited practical guidance. The taxonomy provides a map at the scale of continents but not at the scale of roads and terrain.

**What evidence exists in the paper.** This limitation is visible throughout Section 2. In Section 2.1.3, "task decomposition strategies" lumps together zero-shot chain-of-thought, Tree-of-Thought, and Monte Carlo Tree Search approaches under "single-path chaining" and "multi-path tree expansion" without discussing their fundamentally different computational costs, reliability guarantees, or scaling properties. In Section 2.2.2, "communication-based systems" groups MAD (structured anti-degeneration protocols), MADR (evidence-grounded critique), and MDebate (strategic stubbornness alternation) without comparing their debate dynamics, convergence properties, or computational overhead. The taxonomy tells you *what family* a method belongs to but not *which family member* to pick.

**Mitigation status.** The paper does not address this as a limitation or propose finer-grained sub-taxonomies that would preserve more technical detail. This is arguably inherent to the survey format—a paper covering hundreds of works cannot provide detailed technical comparison of each—but the paper does not provide the kind of summary table or decision tree that would help practitioners navigate within taxonomic categories. The "Agent Collaboration" section (Table 1) provides a method-level summary with "Key Contribution" columns, but these are one-sentence descriptions, not comparative analyses.

---

### Limitation 4: The Paper Assumes But Does Not Demonstrate Recursive Interdependence Between Taxonomy Dimensions

**The assumption or constraint.** A central theoretical claim of the paper is that the three dimensions of the taxonomy—construction, collaboration, and evolution—are not independent but form "interconnected dimensions" where "construction choices enable or constrain collaboration patterns" and "collaboration dynamics drive evolution" (Section 2, Section 3.4). This recursive interdependence is what distinguishes the paper's framework from prior surveys that "examined these aspects separately" (Section 1). However, the paper states these dependencies as logical relationships without providing empirical evidence that they operate as claimed in actual agent systems.

**The consequence.** If the claimed interdependencies are weaker or different in practice than in theory, the taxonomy's "unified architectural perspective" may overstate the degree to which construction, collaboration, and evolution must be designed together. A practitioner might reasonably ask: can I design my agent's memory system independently of how it will collaborate, or must these be co-designed? The taxonomy strongly implies co-design is necessary, but provides no evidence—no controlled experiment showing that a given memory architecture performs differently under centralized versus decentralized collaboration, for instance. The risk is that the taxonomy encourages over-engineering: designers may invest in tightly coupling construction and collaboration based on the theoretical claim of interdependence when looser coupling would work adequately and be simpler to maintain.

**What evidence exists in the paper.** The paper provides no experimental evidence for any of the claimed cross-dimensional dependencies. There are no case studies tracing how a specific construction choice (e.g., static vs. dynamic profile) causally affects collaboration outcomes (e.g., convergence speed in multi-agent debate), or how a specific evolution mechanism (e.g., self-reflection) changes what collaboration patterns become feasible. The dependencies are asserted at the conceptual level: "an agent with only short-term memory cannot effectively participate in long-running multi-agent workflows" (Section 3.4.1) is logically plausible but is not demonstrated with experiments comparing agents identical except for their memory architecture in the same collaborative task.

**Mitigation status.** The paper does not acknowledge this gap between theoretical framework and empirical validation. Given that this is a survey paper rather than an experimental paper, the absence of such experiments is understandable—but the paper's claims about interdependence are stronger than what can be supported purely through conceptual analysis. A more modest claim—that the taxonomy provides a *language* for describing potential dependencies, not that it has *demonstrated* them—would more accurately reflect the evidence provided.

---

### Limitation 5: Security and Social Impact Analysis Lacks the Systematic Depth Afforded to Core Methodology

**The assumption or constraint.** The paper treats security, privacy, and social impact as "first-class dimensions" of the agent taxonomy (Section 4), arguing that "understanding these challenges is crucial for developing robust, trustworthy agent systems" (Section 4 introduction). This inclusion distinguishes the paper from purely algorithmic surveys. However, the depth of analysis in Sections 4.3 (Privacy) and 4.4 (Social Impact and Ethics) is substantially shallower than the core methodology sections. Privacy is covered in under two pages; social impact and ethics in under two pages. In contrast, agent construction (Section 2.1) receives approximately four pages of detailed taxonomic decomposition, and security attacks and defenses (Sections 4.1-4.2) receive approximately six pages with structured tables.

**The consequence.** The paper's treatment of social and ethical concerns risks being performative rather than analytically substantive. Section 4.4.2 lists ethical concerns—bias and discrimination, accountability, copyright, overreliance, environmental impact—but does not map these to specific architectural choices in the way that security threats are mapped to agent components. For example, bias and discrimination are noted as arising because "LLM agents inherently inherit biases present in their training datasets" (Section 4.4.2), but the paper does not connect this to specific memory architectures (which shape what training data influences which decisions), planning mechanisms (which may amplify or mitigate biases depending on how they aggregate information), or collaboration topologies (which may concentrate or distribute biased decision-making). A practitioner concerned about fairness in their agent system would find the security sections (Sections 4.1-4.2) actionable—they identify specific attack vectors and corresponding defenses—but would find the ethics section (4.4.2) generic, referencing broad surveys and position papers rather than providing the kind of structured, architecture-linked analysis the paper's framework promises.

**What evidence exists in the paper.** The imbalance is visible in the paper's structure. Table 3 (agent-centric attacks) provides 20+ entries mapping specific attack and defense methods to references. Table 4 (data-centric attacks) provides 18+ entries with similar structure. Table 5 (privacy threats) provides 14 entries mapping attacks and defenses. Table 6 (social impacts) provides only 14 entries, but these are primarily references to broad surveys and position papers (e.g., "Foundation Models," "Stochastic Parrots," "Fair Learning") rather than to the kind of specific technical methods that populate the security tables. The ethics section does not provide a structured decomposition of bias mechanisms, accountability frameworks, or copyright protection strategies comparable to the decomposition of attack types in Sections 4.1-4.2.

**Mitigation status.** The paper does not acknowledge this asymmetry as a limitation. The inclusion of social impact and ethics is a genuine step beyond purely technical surveys, but the execution does not match the paper's stated goal of treating all dimensions as "first-class." The framework's promise—that ethical concerns are "architectural consequences of how agents are constructed, how they collaborate, and how they evolve" (Section 3.4.5)—is asserted but not developed with the rigor the framework would require.

---

### Limitation 6: The Survey Provides No Guidance on When to Prefer One Architectural Pattern Over Another

**The assumption or constraint.** The taxonomy is purely descriptive—it classifies existing work into architectural categories—but does not provide prescriptive guidance. For any given application scenario, the survey cannot tell you whether to use human-curated static profiles or batch-generated dynamic profiles, centralized or decentralized collaboration, self-reflection or multi-agent debate for evolution. The paper occasionally notes where certain approaches are "particularly effective" (e.g., centralized control is "particularly effective in scenarios demanding high interpretability and regulatory compliance," Section 2.2.1), but these are ad hoc observations rather than systematically derived recommendations.

**The consequence.** A practitioner approaching this survey to make a concrete architectural decision faces a **selection problem** that the taxonomy does not solve. The survey maps the design space but does not provide a decision procedure for navigating it. When should I use revision-based collaboration versus communication-based collaboration? When should I invest in long-term memory versus improved planning? When is hybrid architecture worth the additional complexity over pure centralized or decentralized approaches? These are the questions that determine whether an agent system succeeds or fails in practice, and the survey provides no evidence-based answers. The taxonomy helps a practitioner understand *what* is possible but not *what to do* given specific constraints, requirements, or failure modes.

**What evidence exists in the paper.** This is a structural limitation of the survey's methodology, visible across all sections. Section 2.1.2 describes three memory paradigms (short-term, long-term, retrieval-based) but provides no comparison of their relative strengths, weaknesses, or applicability conditions. Section 2.2 describes three collaboration architectures but does not analyze the tradeoffs between them in terms of scalability, robustness, latency, or failure modes. Section 2.3 describes seven evolution mechanisms (Table 2) but does not indicate which are appropriate for which types of agents, tasks, or deployment contexts. The paper's contribution is mapping the space, not providing a guidebook for navigating it—but for the practitioner audience the paper explicitly targets, a map without navigation guidance has limited utility.

**Mitigation status.** The paper does not frame this as a limitation, and indeed a purely taxonomic survey cannot be expected to also provide a complete decision framework. However, the paper could have provided at least a high-level decision heuristic or a structured comparison of tradeoffs within each taxonomic category. The challenges section (Section 6) identifies open problems but does not provide actionable guidance on how current methods compare. The gap between "here is what exists" and "here is what you should do" remains unfilled, and the paper does not acknowledge that filling it would substantially increase the survey's practical impact.

## 7. Implications and Future Directions
- How this changes the landscape
  - Provides a common design language for LLM agents: with Build‑Collaborate‑Evolve, teams can reason about where to invest (e.g., memory vs. planning vs. topology) and how to evaluate and secure deployments (Figures 1–4; Sections 2–4).
  - Bridges design to evaluation: the mapped benchmarks (Section 3) help convert abstract capabilities (planning, tool use, collaboration) into measurable test plans.

- Follow‑up research enabled/suggested (Section 6)
  - Scalable coordination: hierarchical controllers with decentralized execution; learned collaboration graphs; cost‑aware agent routing (Section 6.1; dynamic hybrids in Section 2.2.3).
  - Long‑horizon memory: hierarchical episodic/semantic memory plus autonomous knowledge compression and retrieval policies (Section 6.2).
  - Reliability pipelines: knowledge‑graph verification, retrieval‑with‑citation generation, self‑consistency ensembles, and standardized AI auditing logs (Section 6.3).
  - Dynamic evaluation: self‑evolving benchmarks, meta‑probing evaluators, contamination‑resistant test generation (Section 6.4).
  - Safety and governance: topology‑aware defenses for multi‑agent networks (Figure 4; Tables 3–4; Section 6.5), constitutional design for planning safety (`TrustAgent`, Section 4.1.4), psychology‑aware risk controls (`PsySafe`, Section 4.1.4).
  - Role‑play fidelity: improve coverage for under‑represented roles, integrate real‑world reasoning frameworks, and enhance dialogue diversity (Section 6.6).

- Practical applications and downstream use cases (Section 5; Table 7)
  - Science: autonomous hypothesis generation and experimental execution (`SciAgents`, `Curie`), chemistry tool‑augmented synthesis (`ChemCrow`), materials/astronomy co‑pilots (`AtomAgents`, CTA agents).
  - Medicine: virtual hospitals and patient simulators (`AgentHospital`, `AIPatient`), multimodal radiology agents with uncertainty reporting (`CXR‑Agent`, `MedRAX`).
  - Productivity: software lifecycle automation (`ChatDev`, `MetaGPT`) and recommender systems with generative user/item agents (`Agent4Rec`, `AgentCF`, `MACRec`, `RecMind`).
  - Web/desktop autonomy: generalist web agents (`Mind2Web`), real‑computer tasking (`OSWorld`), and API‑integration frameworks (`RestGPT`, `GraphQLRestBench`).

In short, this paper’s value is architectural: it supplies a precise decomposition of LLM agent systems (what to build), maps collaboration patterns to task properties (how to organize), and catalogs evolution, evaluation, and safety (how to improve and deploy responsibly). Figures 1–4 and Tables 1–7 function as a design and testing checklist for anyone standing up real LLM‑based agent systems.
