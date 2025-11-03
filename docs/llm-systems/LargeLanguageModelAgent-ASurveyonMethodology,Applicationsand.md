# Large Language Model Agent: A Survey on Methodology, Applications and Challenges

**ArXiv:** [2503.21460](https://arxiv.org/abs/2503.21460)

## 🎯 Pitch

This paper introduces a unified, methodology-driven taxonomy—spanning how LLM agents are built, collaborate, and evolve—offering the first comprehensive architectural perspective that interlinks agent construction, coordination, memory, safety, evaluation, and real-world deployment. By systematizing a fragmented and rapidly advancing field, this survey equips researchers and practitioners with actionable frameworks to design, benchmark, and scale reliable, robust, and impactful LLM-based agentic systems across diverse applications.

---

## 1. Executive Summary (2–3 sentences)
This paper presents a methodology‑centered, end‑to‑end taxonomy of Large Language Model (LLM) agents that unifies how agents are built, how they collaborate, and how they evolve, and connects these to evaluation, tools, security, privacy, ethics, and applications. Its core significance is a coherent architectural lens—the Build‑Collaborate‑Evolve framework (Figures 1–2; Section 2)—that systematizes a fragmented literature into interoperable components and provides actionable guidance for designing, assessing, and deploying modern LLM‑based agent systems at scale.

## 2. Context and Motivation
- Problem/gap addressed
  - The literature on LLM agents is rapidly expanding but fragmented: different works emphasize isolated pieces (e.g., tool use, planning, or multi‑agent protocols) without a cohesive architectural methodology that maps design choices to capabilities and risks (Distinction from Previous Surveys, p. 2; Figure 1).
  - Existing surveys often focus on narrow slices (e.g., gaming, multi‑modality, security) or provide high‑level overviews without a detailed methodological taxonomy that connects construction, collaboration, and evolution (Section “Distinction from Previous Surveys,” p. 2).

- Why this matters
  - Practically: Agentic systems are moving from demos to production (e.g., research assistants, autonomous web agents, software automation). Designing robust, safe, and evolvable systems requires shared abstractions and evaluation scaffolds (Figure 1; Sections 2–4).
  - Theoretically: Understanding how memory, planning, tools, and multi‑agent dynamics generate emergent behavior informs how to achieve reliability, generalization, and safety (Sections 2.1–2.3; 6.3).

- Prior approaches and limitations
  - Prior surveys concentrated on specific domains (e.g., games [11, 12]) or single dimensions (e.g., workflows [19], multi‑agent interaction [18]). They lack an integrated framework that ties together individual agent design, multi‑agent coordination, and learning/evolution (Distinction from Previous Surveys, p. 2).

- How this work positions itself
  - It proposes a methodology‑centered taxonomy anchored in three linked dimensions—Construction, Collaboration, Evolution—supplemented by Evaluation & Tools, Real‑World Issues, and Applications (Figure 1; Section 2).
  - It provides concrete decomposition of agent internals (profile, memory, planning, action) and external structures (centralized/decentralized/hybrid collaboration) with curated exemplars (Figure 2; Sections 2.1–2.2), and ties these to security/privacy risks (Figure 4; Section 4) and benchmarks (Figure 3; Section 3).

## 3. Technical Approach
The paper’s “approach” is a unifying architectural framework (Figures 1–2) plus a curated mapping of methods, tools, and risks. Below, each building block is explained in mechanism‑level detail, with why/when to use it.

- Build: Agent Construction (Section 2.1; Figure 2)
  - `Profile definition` (Section 2.1.1)
    - What it is: The agent’s operational identity—role, goals, domain constraints, and communication protocol.
    - Two mechanisms:
      - Human‑curated static profiles: Manually specified, interpretable roles (e.g., developer/tester) with deterministic interaction rules. Useful for regulated domains and reproducibility. Examples: `CAMEL`, `AutoGen`, `MetaGPT`, `ChatDev`, `AFlow` (Section 2.1.1; Figure 2).
      - Batch‑generated dynamic profiles: Programmatically sample diverse traits (personality, background) to create heterogeneous agent populations for simulation or robustness (e.g., `Generative Agents`, `RecAgent`), optionally optimized with `DSPy` (Section 2.1.1).
    - Why this design: Static profiles yield control and compliance; dynamic profiles yield diversity and emergent behaviors (Section 2.1.1).

  - `Memory mechanisms` (Section 2.1.2)
    - Short‑term memory: Conversation/state within the context window (pros: strong local coherence; cons: context length limits; requires summarization/compression). Used in `ReAct`, `ChatDev`, `Graph of Thoughts`, `AFlow` (Section 2.1.2).
    - Long‑term memory: Persistent stores that transform transient reasoning into reusable assets:
      - Skill libraries (e.g., `Voyager` auto‑discovers Minecraft skills; `GITM` knowledge base).
      - Experience repositories (e.g., `ExpeL` distilled successes/failures; `Reflexion` self‑improvement).
      - Tool synthesis (e.g., `TPTU` composes tools; `OpenAgents` self‑expands toolkits; `MemGPT` tiered memory) (Section 2.1.2).
    - Knowledge retrieval as memory (`RAG`, `GraphRAG`, `IRCoT`, `Llatrieval`, `KG‑RAR`, `DeepRAG`): External corpora/graphs are queried in‑loop, balancing parametric knowledge vs. external evidence (Section 2.1.2).
    - Why this design: Short‑term memory supports local reasoning; long‑term/retrieval overcome forgetting and training‑time limits, enabling cumulative competence (Section 2.1.2).

  - `Planning capability` (Section 2.1.3)
    - Task decomposition:
      - Chain‑based (sequential): Plan‑and‑solve, self‑consistency/ensemble voting, dynamic next‑step planning (`ReAct`). Simple but error accumulation risk (Section 2.1.3).
      - Tree‑based: `ToT`, `Tree‑planner`, `ReAcTree`, `ReST‑MCTS*`, enabling backtracking, exploration (e.g., MCTS) and trial‑error‑correct loops; applied to robotics/embodied tasks (Section 2.1.3).
    - Feedback‑driven iteration: Incorporates environment signals (robotics), human feedback, self‑introspection, and multi‑agent critique to refine plans (`BrainBody‑LLM`, `TrainerAgent`, `RASC`, `REVECA`, `AdaPlanner`, `AIFP`) (Section 2.1.3).
    - Why this design: Complex tasks require decomposition and iterative correction; tree search reduces premature commitment (Section 2.1.3).

  - `Action execution` (Section 2.1.4)
    - Tool use: Two sub‑problems—when to call tools (confidence/need‑based decision) and which tool to select (documentation understanding; e.g., `EASYTOOL` simplifies docs, `GPT4Tools`, `TRICE`, `AvaTaR`) (Section 2.1.4).
    - Physical interaction: Embodied control must factor hardware, social norms, and multi‑agent coordination (e.g., `DriVLMe` for driving, `ReAd`, `Collaborative Voyager`) (Section 2.1.4).
    - Why this design: Tools extend precision/coverage (math, APIs); embodiment requires grounded feedback loops (Section 2.1.4).

- Collaborate: Agent Collaboration (Section 2.2; Figure 2; Table 1)
  - Centralized control (Section 2.2.1): A controller decomposes tasks, assigns subgoals, and integrates results.
    - Explicit controllers: Human/LLM orchestration pipelines (`Coscientist` for experimental workflows; `LLM‑Blender` ranks/fuses responses; `MetaGPT` manages software roles).
    - Differentiation‑based: A single strong agent implicitly plays sub‑roles (`AutoAct` splits into plan/tool/reflect; `Meta‑Prompting` assigns subtasks by meta‑prompts; `WJudge` shows even weak controllers can help) (Section 2.2.1).
    - Trade‑off: Strong coordination and accountability; single‑point bottleneck/failure risk (Section 2.2.1).

  - Decentralized collaboration (Section 2.2.2): Peers interact directly; no single hub.
    - Revision‑based: Agents iteratively edit/critique a shared artifact to reach consensus (`MedAgents`, `ReConcile`, `METAL`, `DS‑Agent`)—more deterministic outputs (Section 2.2.2).
    - Communication‑based: Open dialogues expose reasoning traces; good for dynamic social scenarios (`MAD`, `MADR`, `MDebate`, `AutoGen`) (Section 2.2.2).
    - Trade‑off: More flexible/robust but harder to coordinate and verify (Section 2.2.2).

  - Hybrid architectures (Section 2.2.3)
    - Static patterns: Predefined topologies—e.g., groups with role‑play and governance (`CAMEL`), three‑tier planner/negotiator/market (`AFlow`), canonical patterns (BUS/STAR/TREE/RING in `EoT`).
    - Dynamic systems: Topology adapts by performance feedback—`DiscoGraph` (learned collaboration graphs), `DyLAN` (importance‑aware restructuring), `MDAgents` (route by task complexity) (Section 2.2.3).
    - Trade‑off: Balance controllability with adaptivity; dynamic routing reduces waste and improves fit to task (Section 2.2.3).

- Evolve: Agent Evolution (Section 2.3; Figure 2; Table 2)
  - Autonomous self‑learning (Section 2.3.1)
    - Self‑supervised/pretraining refinements: `SE` (adaptive masking), evolutionary model merging [87], `DiverseEvol` (diverse data sampling).
    - Self‑reflection/correction: `SELF‑REFINE`, `STaR`, `V‑STaR`, self‑verification—iteratively generate, critique, and improve outputs.
    - Self‑rewarding/RL: `Self‑Rewarding`, `RLCD`, `RLC` align models with internally generated rewards.
  - Multi‑agent co‑evolution (Section 2.3.2)
    - Cooperative: `ProAgent` (intent inference), `CORY` (cooperative RL fine‑tuning), `CAMEL` (role‑based collaboration).
    - Competitive/adversarial: `Red‑Team LLMs`, multi‑agent debate (`MAD`, `MDebate`)—stress tests that improve robustness/reasoning.
  - Evolution with external resources (Section 2.3.3)
    - Knowledge‑enhanced planning: `KnowAgent` (action knowledge), `WKM` (world priors + dynamic local knowledge).
    - Feedback‑driven: `CRITIC` (tool‑based self‑correction), `STE` (trial‑and‑error for tool mastery), `SelfEvolve` (generate + debug with execution feedback).

- Evaluation and tools (Section 3; Figure 3)
  - Benchmarks span general capability (e.g., `AgentBench`, `Mind2Web`), domain‑specific (e.g., `MedAgentBench`, `LaMPilot`), real‑world environments (`OSWorld`, `OmniACT`, `EgoLife`), and collaboration (e.g., enterprise‑style `TheAgentCompany`) (Sections 3.1.1–3.1.3).
  - Tools cover: what agents use (search, calculators, interpreters, APIs), what agents create (tool synthesis frameworks like `CRAFT`, `Toolink`, `CREATOR`, `LATM`), and what devs use to deploy/manage agents (`AutoGen`, `LangChain`, `LlamaIndex`, `Dify`, `Ollama`, `MCP`) (Section 3.2).

- Real‑world issues (Section 4; Figure 4)
  - Security: Agent‑centric (adversarial, jailbreak, backdoor, collaboration attacks—Table 3) and data‑centric (prompt/indirect injections, poisoning, interaction‑layer exploits—Table 4).
  - Privacy: Memorization (data extraction, membership, attribute inference) and IP (model/prompt stealing), with defenses like DP, distillation, watermarking, blockchain (Table 5).
  - Social impact: Benefits (automation, jobs, information access) vs. ethical concerns (bias, accountability, copyright, overreliance, environmental cost) summarized in Table 6 (Section 4.4).

## 4. Key Insights and Innovations
- Methodology‑centered Build‑Collaborate‑Evolve taxonomy
  - What’s new: A single framework that decomposes agent internals (profile, memory, planning, action) and externals (centralized/decentralized/hybrid) and ties them to evolution mechanisms (self‑learning, co‑evolution, external resources) (Figures 1–2; Section 2).
  - Why it matters: It connects design choices to emergent behavior and evaluation/security realities, enabling principled system design rather than ad‑hoc pipelines.

- Unification of memory and retrieval as first‑class design axes
  - What’s new: A three‑way classification—short‑term, long‑term (skills/experience/tools), and retrieval‑as‑memory (`RAG`, `GraphRAG`, reasoning‑integrated retrieval) (Section 2.1.2).
  - Why it matters: It clarifies how to transcend context‑window limits (noted constraints in Section 2.1.2) and build cumulative competence—critical for longevity and real‑world deployment.

- Collaboration patterns distilled into explicit, decentralized, and hybrid topologies
  - What’s new: A fine‑grained split of decentralized styles (revision vs. communication) and a catalog of static vs. dynamic hybrid topologies with examples (`CAMEL`, `AFlow`, `EoT`, `DiscoGraph`, `DyLAN`, `MDAgents`) (Sections 2.2.2–2.2.3).
  - Why it matters: Designers can select appropriate organization patterns (determinism vs. flexibility vs. adaptivity) grounded in task needs.

- End‑to‑end bridge from architecture to evaluation, tools, and risk
  - What’s new: A single map that spans benchmarks (Figure 3; Sections 3.1.1–3.1.3), tool ecosystems (Section 3.2), and real‑world risks (Figure 4; Section 4), with concrete scales (e.g., number of tasks, APIs, environments).
  - Why it matters: It enables coverage‑aware testing (e.g., tool use via `Seal‑Tools` with “1,024 nested instances,” Section 3.1.1) and safety‑first deployment planning.

These are primarily integrative innovations (a coherent architecture and mappings) rather than a new algorithm; the novelty is in the synthesis and the actionable decomposition.

## 5. Experimental Analysis
While this is a survey (no new experiments), it assembles evaluation scaffolds with concrete scales, domains, and scenarios. Below are the key evaluation elements and what they support.

- Evaluation methodology (Section 3; Figure 3)
  - General capability
    - `AgentBench` offers “a unified test field across eight interactive environments” (Section 3.1.1).
    - `Mind2Web` is “the first generalist agent for evaluating 137 real‑world websites with tasks spanning 31 domains” (Section 3.1.1).  
      > “Mind2Web … proposing the first generalist agent for evaluating 137 real‑world websites with different tasks spanning 31 domains.” (Section 3.1.1)
    - `MMAU` provides “more than 3,000 cross‑domain tasks” with capability mapping (Section 3.1.1).
  - Scientific/vision/embodied
    - `BLADE` targets scientific workflows; `VisualAgentBench` spans GUI/visual design; `Embodied Agent Interface` gives fine‑grained error classification; `CRAB` enables cross‑platform testing with a unified Python interface (Section 3.1.1).
  - Dynamic/self‑evolving evaluation
    - `BENCHAGENTS` auto‑creates benchmarks via agents; “Seal‑Tools (1,024 nested instances of tool calls)” and `CToolEval` (“398 Chinese APIs across 14 domains”) standardize tool‑use evaluation (Section 3.1.1).  
      > “Seal‑Tools … 1,024 nested instances of tool calls … CToolEval (398 Chinese APIs across 14 domains)” (Section 3.1.1)
  - Domain‑specific and real‑world simulation
    - Healthcare: `MedAgentBench` (tasks by “300 clinicians in an FHIR‑compliant environment”), `AI Hospital` simulates multi‑agent clinical workflows (Section 3.1.2).
    - Autonomy: `LaMPilot` ties LLMs to self‑driving code generation (Section 3.1.2).
    - Data science & ML: `DSEval`, `DA‑Code`, `DCA‑Bench`, `MLAgent‑Bench`, `MLE‑Bench` (Section 3.1.2–3.1.3).
    - Real computers/web: `OSWorld` “supports 369 multi‑application tasks across Ubuntu/Windows/macOS” (Section 3.1.2).  
      > “OSWorld … supports 369 multi‑application tasks across Ubuntu/Windows/macOS.” (Section 3.1.2)
    - Web/task UX: `TurkingBench` (158 micro‑tasks), `OmniACT` (32K desktop/web instances), `EgoLife` (300‑hour egocentric multimodal dataset + EgoLifeQA), `GTA` (real tools + multi‑modal inputs) (Section 3.1.2).
  - Collaboration/system‑level evaluation
    - Enterprise‑style `TheAgentCompany`, ML research/engineering (`MLRB`, `MLE‑Bench`), and comparative frameworks (`AutoGen` vs. `CrewAI`) (Section 3.1.3).

- What the numbers show
  - Breadth and realism are increasing: multi‑OS desktop tasks (369 tasks, `OSWorld`), large web coverage (137 sites, `Mind2Web`), deep tool‑use testbeds (1,024 nested calls, `Seal‑Tools`), and domain‑specific rigor (FHIR‑compliant medical tasks, `MedAgentBench`) (Sections 3.1.1–3.1.2).
  - Safety/stress testing is maturing: `AgentHarm` curates “440 malicious agent tasks in 11 hazard categories” (Section 3.1.2), enabling systematic multi‑step harmfulness assessment.

- Do these convincingly support the paper’s synthesis?
  - Yes for scope and structure: the curated benchmarks map naturally onto the Build‑Collaborate‑Evolve design (e.g., planning and tool use can be stress‑tested with `Mind2Web`, `Seal‑Tools`; collaboration with enterprise/ML research settings). The paper is cautious not to claim state‑of‑the‑art numbers—it focuses on coverage and evaluation patterns (Sections 3.1–3.2).

- Ablations/failure cases/robustness checks
  - Being a survey, there are no new ablations. However, the security sections provide adversarial/poisoning stressors (Tables 3–4), and the reliability challenges (Section 6.3) discuss hallucinations and sensitivity to prompt changes, calling for verification pipelines.

- Conditional results and trade‑offs
  - Centralized vs. decentralized vs. hybrid collaboration each trade off controllability, flexibility, and scalability (Sections 2.2.1–2.2.3; Table 1).
  - Memory choices trade short‑term coherence vs. long‑term accumulation vs. retrieval cost/latency (Section 2.1.2).

## 6. Limitations and Trade-offs
- Assumptions and scope
  - The taxonomy assumes LLM‑centric agents; specialized non‑LLM agents (symbolic planners, classic RL) are treated principally as tools/augmentations (Figure 1; Section 2), not primary citizens.
  - The survey aggregates but does not empirically compare methods head‑to‑head; performance conclusions rely on cited works and benchmark designs (Sections 3–5).

- Scenarios not fully addressed
  - Closed‑loop safety for high‑stakes settings (healthcare, finance) still lacks standardized verification triggers and human‑in‑the‑loop escalation points (Section 6.3).
  - Cross‑modal embodied agents beyond desktop/web (e.g., multi‑robot physical collaboration under uncertainty) are cataloged but not deeply formalized (Sections 2.1.4, 3.1.2).

- Computational and scalability constraints
  - Multi‑agent systems with heavyweight LLMs face orchestration and cost bottlenecks; classic multi‑agent infrastructures are not optimized for billion‑parameter models (Section 6.1).
  - Memory scaling and relevance management (beyond vector DBs and naive summarization) remain open (Section 6.2).

- Open weaknesses and questions
  - Reliability/scientific rigor: susceptibility to hallucinations, sensitivity to prompts, and lack of standardized citation/grounding pipelines (Section 6.3).
  - Dynamic evaluation lags behind fast‑moving models; contamination and overfitting to benchmarks remain concerns (Section 6.4).
  - Governance and ethics: actionable, auditable accountability pipelines and bias diagnostics across cultures and modalities need formalization (Sections 4.4, 6.5).

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
