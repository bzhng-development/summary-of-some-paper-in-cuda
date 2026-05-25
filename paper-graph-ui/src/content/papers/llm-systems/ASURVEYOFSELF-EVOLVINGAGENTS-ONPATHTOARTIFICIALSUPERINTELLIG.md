# A SURVEY OF SELF-EVOLVING AGENTS: ON PATH TO ARTIFICIAL SUPER INTELLIGENCE

**ArXiv:** [2507.21046](https://arxiv.org/abs/2507.21046)

## 🎯 Pitch

This survey delivers the first comprehensive, systematic framework for understanding and building 'self-evolving agents'—AI systems that autonomously adapt and improve through continual updates to their models, memories, prompts, tools, and architectures. By organizing the field along the dimensions of what, when, how, and where agents evolve, and pairing this with concrete evaluation criteria and benchmarks, the paper charts a roadmap for overcoming the static limitations of current agents and propelling AI toward greater adaptability, robustness, and the emergence of Artificial Super Intelligence. This matters because real-world deployments demand agents capable of evolving with their environments; such continual self-improvement is foundational for creating AI systems that go beyond static performance to achieve human-level versatility and resilience.

---

## 1. Executive Summary (2-3 sentences)
This survey builds the first structured, end-to-end framework for understanding and designing “self‑evolving agents”—AI agents that can improve themselves during and between tasks by updating models, prompts/memories, tools, and architectures. It formalizes the problem (Sec. 2.1; Eqs. 1–3), organizes the field along four axes—what to evolve, when to evolve, how to evolve, and where to evolve (Figs. 2–3)—and pairs these with concrete evaluation goals, metrics, and benchmarks tailored to continual adaptation (Sec. 7; Table 5; Figure 9; Table 7).

## 2. Context and Motivation
- Problem addressed
  - Modern large language model (LLM) agents are “static”: they typically do not change their internal parameters, prompts, tools, or workflows once deployed. The survey targets the gap between these static systems and agents that can “self‑evolve”—learn from interactions and feedback to adapt to new tasks and environments in real time and over time (Abstract; Sec. 1).
- Why it matters
  - Practical: Deployed agents operate in open-ended, dynamic settings (coding assistants, web/GUI agents, medical and education agents). Static behavior limits robustness, adaptability, and real-world performance.
  - Conceptual: The path from LLMs to Artificial Super Intelligence (ASI) likely requires agents that autonomously learn and restructure themselves (Fig. 1 conceptual trajectory).
- Prior approaches and their gaps (Sec. 2.2; Table 1)
  - Curriculum learning: orders training data by difficulty but “updates only model parameters” and uses static datasets (Table 1).
  - Lifelong/continual learning: adds knowledge sequentially but again focuses on model parameters and “acquires knowledge passively” (Sec. 2.2).
  - Model editing/unlearning: efficiently modifies or removes specific knowledge but cannot evolve non-parametric components (memory, tools, workflows) and lacks autonomous exploration (Sec. 2.2).
- Positioning of this survey
  - It expands the unit of evolution from parameters to the full agent system—`model`, `context` (prompts and memory), `toolset`, and `architecture`—and adds timing (intra-test vs inter-test) and mechanism (reward-based, imitation, population-based) dimensions (Figs. 2–3; Secs. 3–5). It also systematizes evaluation for adaptive agents (Sec. 7).

## 3. Technical Approach
This paper’s “method” is a formal framework and taxonomy rather than a new algorithm.

1) Formal problem definition (Sec. 2.1)
- Environment as POMDP (partially observable Markov decision process—a standard framework for sequential decision-making under uncertainty):
  - “E = (G, S, A, T, R, Ω, O, γ)” where `G`=goals, `S`=states, `A`=actions (including language and tool calls), `T`=transition, `R`=feedback (scalar or textual), `Ω`=observations, `O`=observation function, `γ`=discount (Sec. 2.1).
- Agent system components: Π = (Γ, {ψi}, {Ci}, {Wi})
  - `Γ` = architecture/workflow graph; at each node `Ni`:
    - `ψi` = underlying LLM/MLLM,
    - `Ci` = context (prompt `Pi` and memory `Mi`),
    - `Wi` = tools/APIs available (Sec. 2.1).
  - The node policy `πθi(·|o)` outputs action distributions; `θi = (ψi, Ci)` (Sec. 2.1).
- Self‑evolving strategy:
  - > “f(Π, τ, r) = Π′ = (Γ′, {ψ′i}, {C′i}, {W′i})” (Eq. 1) — given trajectory `τ` and feedback `r`, the agent updates its components.
- Objective:
  - > “Given tasks (T0, …, Tn) and initial Π0, evolve Πj+1 = f(Πj, τj, rj) to maximize ∑j U(Πj, Tj)” (Eqs. 2–3).
  - `U` is a utility function derived from feedback and metrics.

2) Taxonomy: What, When, How, Where (Figs. 2–3)
- What to evolve (Sec. 3; Table 2)
  - Models: update policies/weights via self-generated data, online feedback, or RL (Sec. 3.1). Example methods: SCA, SELF, SCoRe, PAG, TextGrad (Table 2).
  - Context: Memory evolution (add/merge/delete/update long-term experiences; e.g., Mem0, SAGE) and Prompt optimization (search, gradient-like edits, evolutionary methods; e.g., APE, PromptBreeder, DSPy, TextGrad) (Secs. 3.2.1–3.2.2).
  - Tools: Creation (e.g., Voyager, Alita, SkillWeaver), Mastery/refinement from feedback (LearnAct, DRAFT), and Selection/management at scale (ToolGen, AgentSquare) (Sec. 3.3).
  - Architecture: Single-agent node/workflow optimization (TextGrad; EvoFlow; AgentSquare) and multi-agent workflow generation and co-evolution (AFlow, ADAS, ReMA, GiGPO) (Sec. 3.4).
- When to evolve (Sec. 4; Fig. 5)
  - Intra-test-time self‑evolution: adaptation while solving the current task instance—without (ICL) or with (SFT/RL) weight updates; examples: Reflexion, AdaPlanner, LADDER (Sec. 4.1).
  - Inter-test-time self‑evolution: learning between tasks from past trajectories or datasets via ICL/SFT/RL; examples: STaR, Quiet‑STaR, SiriuS, RAGEN (Sec. 4.2).
- How to evolve (Sec. 5; Figs. 6–7; Table 4)
  - Reward-based: feedback as language critiques, internal confidence, external rewards, or implicit rewards (Sec. 5.1).
  - Imitation/demonstration: self-generated demonstrations (STaR variants), cross-agent demos (SiriuS), hybrid strategies (Sec. 5.2).
  - Population-based/evolutionary: evolve code, prompts, teams, or policies via selection/mutation/self-play (DGM, SPIN, EvoMAC) (Sec. 5.3).
  - Cross-cutting dimensions: online vs offline learning, on-policy vs off-policy, and reward granularity (process vs outcome vs hybrid) (Sec. 5.4; Fig. 7; Table 4).
- Where to evolve (Sec. 6; Fig. 8)
  - General-purpose assistants: memory mechanisms, model–agent co‑evolution, and curriculum-driven training (Sec. 6.1).
  - Specialized domains: coding, GUI, finance, medical, education, others (Sec. 6.2). Representative systems listed for each.

Design choices and why they matter
- Broadening the evolvable surface: Moving beyond model weights to include prompts/memory, tools, and architectures acknowledges that many agent failures arise from insufficient context, capabilities, or workflows, not just parametric deficiencies (Sec. 3; Table 2).
- Two-phase timing (intra vs inter test time): Separating immediate adaptation from retrospective consolidation clarifies method selection and constraints (Fig. 5; Secs. 4.1–4.2).
- Multiple evolution mechanisms: Reward-based, imitation, and population-based approaches capture complementary strengths (sample efficiency vs exploration vs structural innovation) (Sec. 5; Table 4).
- Evaluation tied to adaptivity and safety: The paper aligns metrics and benchmarks with the longitudinal nature of self‑evolution (Sec. 7; Fig. 9; Table 5; Table 7).

## 4. Key Insights and Innovations
- A unifying formalism for self‑evolution (Sec. 2.1; Eqs. 1–3)
  - What’s new: A precise mapping from agent experience (`τ`, `r`) to updates over architecture, model, context, and tools (`Π → Π′`), plus a cumulative-utility objective.
  - Why it matters: It turns “agent improvement” into a well-posed optimization problem that can be analyzed, compared, and implemented.
- The “What–When–How–Where” decomposition (Figs. 2–3; Secs. 3–6)
  - What’s new: A comprehensive taxonomy that simultaneously covers components (what), timing (when), mechanisms (how), and application domains (where).
  - Significance: Prior work surveyed narrower facets (e.g., tool use or model training). Here the interdependencies (e.g., prompt evolution inside auto-generated workflows) become explicit, guiding system design and research roadmaps.
- Cross-cutting evolutionary dimensions (Sec. 5.4; Fig. 7; Table 4)
  - What’s new: A comparative lens—offline/online, on/off-policy, reward granularity—that cuts across reward-based, imitation, and evolutionary methods.
  - Significance: It exposes key trade-offs (stability vs sample efficiency; dense vs sparse rewards) that practitioners must manage when building real systems.
- Evaluation framework tailored to adaptive agents (Sec. 7; Fig. 9; Table 5; Table 7)
  - What’s new: Goal‑driven metrics such as Adaptivity over iterations, Retention (Forgetting/BWT), Generalization across domains, Efficiency, and Safety; paired with static/short‑horizon/long‑horizon paradigms and concrete benchmarks.
  - Significance: Moves beyond one‑shot accuracy to longitudinal performance, enabling rigorous assessment of self‑evolution.

## 5. Experimental Analysis
Because this is a survey, it does not run new experiments. Instead, it specifies how to evaluate self‑evolving agents and compiles the landscape of benchmarks and methods.

- Evaluation methodology and metrics (Sec. 7)
  - Goals (Table 5; Fig. 9):
    - Adaptivity: “Success rate by iteration steps” and “Adaptation speed” track improvement during interaction.
    - Retention: “Forgetting (FGT)” and “Backward Transfer (BWT)” quantify whether learning new tasks degrades or improves prior tasks (Sec. 7.1; equations given).
    - Generalization: Aggregate and out-of-domain performance across task suites.
    - Efficiency: Token/time cost, steps, tool productivity.
    - Safety: Safety/Harm scores, Completion under Policy (CuP), Risk ratio, Refusal and Leakage rates.
  - Evaluation paradigms (Sec. 7.2; Fig. 9; Table 6):
    - Static assessment: one-shot capability on fixed sets (e.g., AgentBench, OSWorld; Table 7).
    - Short-horizon: improvement across attempts/episodes; includes built-in dynamic tasks like MemoryAgentBench’s “Test-Time Learning” (Sec. 7.2.2).
    - Long-horizon/lifelong: sequences of diverse tasks (e.g., LTMBenchmark, LifelongAgentBench) with retention metrics and dynamic benchmarks (Sec. 7.2.3).
- Benchmarks covered (Table 7; Sec. 7.2)
  - External task-solving: WebArena (812 tasks), SWE-bench (2,294 issues), OSWorld (369 tasks), GAIA (466 tasks), TheAgentCompany (175 tasks).
  - Component skills: Planning (PlanBench ~26,250 tasks), Tool use (ToolBench 126,486 examples; T‑Eval 23,305), Memory (StoryBench; MemoryAgentBench 2,200), Multi-agent collaboration (MultiAgentBench, SwarmBench).
  - Safety benchmarks: Agent‑SafetyBench (20,000), ST‑WebAgentBench (235) (Table 7).
- Evidence strength
  - The survey aggregates methods and benchmarks (Tables 2–4, 7) and specifies metrics (Table 5), but it does not perform meta-analysis or standardized cross-method comparisons. Claims are therefore qualitative (e.g., “automatically discovered workflows could outperform human-designed ones” in the workflow-optimization narrative; Sec. 3.4.2), with references to source papers for quantitative details.
- Ablations/robustness/failure modes
  - Not directly provided by the survey; instead, Secs. 7 and 8 highlight evaluation gaps (e.g., catastrophic forgetting, dynamic safety, and cost of dynamic reasoning) and call for long-horizon assessments and safety stress tests.

## 6. Limitations and Trade-offs
Grounded in the survey’s analysis and “Future Direction” section (Sec. 8):

- Assumptions and scope
  - The framework presumes access to feedback signals `r` (scalar or textual) and well-defined task utilities `U` (Sec. 2.1). In many real applications, reliable, low-noise feedback may be costly or delayed (Sec. 5.1 and 7.1 discuss signal design and sparsity).
- Unaddressed scenarios / edge cases
  - Non-stationary or adversarial environments where feedback is strategically misleading are not deeply treated; safety‑critical adversarial threats for agents remain an open challenge (Sec. 8.3; Benchmarks in Sec. 7 focus on controlled settings).
  - Multi-agent ecosystems with evolving norms and incentives need dynamic evaluation and role adaptation beyond current static benchmarks (Sec. 8.4).
- Computational and data costs
  - “The cost of dynamic reasoning” and test-time scaling can be substantial (Sec. 8.2; cites infrastructure concerns in [308]); population-based methods can be resource‑intensive (Table 4 “Scalability” row).
  - Online RL for web/GUI agents needs simulators or live environments; reward models and world models introduce additional training loops (Secs. 5.1, 6.1).
- Methodological weaknesses and open questions
  - Generalization vs specialization tension (Sec. 8.2): agents optimized for a narrow domain often struggle to transfer.
  - Continual learning risks catastrophic forgetting; balancing stability vs plasticity remains unsettled in LLM agents (Sec. 8.2; Retention metrics in Sec. 7.1 address evaluation, not solution).
  - Safety/control: agents still “struggle to accurately differentiate between necessary and irrelevant sensitive information” and handle goals involving unethical methods (Sec. 8.3).
  - Knowledge transfer across agents and emergence of robust world models are not well-understood (Sec. 8.2, “Knowledge Transferability”).

## 7. Implications and Future Directions
- How this changes the field
  - The survey reframes agent improvement as a full‑stack, continual process across model, context, tools, and architecture, anchored in a rigorous formalism (Sec. 2.1) and a multi-axis taxonomy (Figs. 2–3). This provides a common language to design, compare, and evaluate self‑evolving systems and aligns research with long‑horizon, safety‑aware goals.
- Research directions (Sec. 8)
  - Personalized agents at scale: move from heavy post-training to self‑generated preference data and robust “cold‑start” personalization with reliable long-term memory (Sec. 8.1).
  - Generalization: test-time scaling policies, meta-learning for cross-domain adaptation, and mechanisms to prevent catastrophic forgetting while maintaining adaptability (Sec. 8.2).
  - Safe and controllable agents: richer constitutions, better risk detection in ambiguous contexts, privacy-aware memory, and diverse real-world safety datasets (Sec. 8.3; Table 7 lists initial safety benchmarks).
  - Multi-agent ecosystems: dynamic role allocation, knowledge sharing, and workflow/knowledge co‑evolution with live, longitudinal benchmarks (Sec. 8.4).
- Practical applications
  - General domain assistants with persistent memory and curricula (Mobile-Agent‑E, WebRL; Sec. 6.1).
  - Specialized domains:
    - Coding: self-improving and multi-agent code generation and repair (SICA, EvoMAC; Sec. 6.2).
    - GUI/Web: end-to-end computer use and navigation with reflection and RL (OSWorld agents, WebVoyager; Sec. 6.2).
    - Finance: self-improving trading agents with domain knowledge evolution (QuantAgent; Sec. 6.2).
    - Medical and Education: consultation/tutoring agents that accumulate cases and refine strategies over time (Agent Hospital, MedAgentSim, PACE; Sec. 6.2).

Quoted anchors for key definitions and objectives:
- > “f(Π, τ, r) = Π′ …” (Eq. 1; Sec. 2.1)
- > “Πj+1 = f(Πj, τj, rj)” (Eq. 2; Sec. 2.1)
- > “max_f ∑ U(Πj, Tj)” (Eq. 3; Sec. 2.1)
- Comparative scope:
  - > “Self‑evolving Agents ✓ evolve Context and Toolset; Curriculum/Lifelong/Editing do not” (Table 1).
- Taxonomic overviews:
  - > “What–When–How–Where” structure and examples (Figs. 2–3).
- Evaluation blueprint:
  - > “Adaptivity, Retention (FGT/BWT), Generalization, Efficiency, Safety” (Table 5; Fig. 9), and benchmark catalog (Table 7).

Overall, this survey supplies the conceptual scaffolding—and the evaluation toolkit—needed to push from static LLM tools to adaptive, safe, and scalable agentic systems capable of self‑evolution on the road toward stronger general intelligence.
