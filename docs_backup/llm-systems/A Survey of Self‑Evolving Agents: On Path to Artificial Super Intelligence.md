# A Survey of Self‑Evolving Agents: On Path to Artificial Super Intelligence

**ArXiv:** [2507.21046](https://arxiv.org/abs/2507.21046)
**Authors:** Huan‑ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Qihan Ren, Cheng Qian, Zhenghailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, Mengdi Wang
**Institutions:** Various institutions across contributing authors (not individually specified in abstract)

## 🎯 Pitch

This paper introduces a comprehensive framework for self-evolving agents, integrating LLMs that adapt and improve autonomously across tasks and environments. By formalizing agent evolution, decision-making, and feedback-driven transformation, it offers a foundational blueprint for developing adaptive systems crucial for dynamic fields like digital assistants and healthcare, ensuring robustness and scalability in real-world applications.

---

## 1. Executive Summary (2-3 sentences)
This survey systematizes the rapidly growing area of self‑evolving agents—LLM‑based systems that can improve themselves during and between tasks—into a unified, formal framework. It answers three core questions—what to evolve, when to evolve, and how to evolve—while adding “where to evolve” (applications) and “how to evaluate,” with formal definitions (Section 2.1, Eqs. 1–3), a complete taxonomy (Figures 2–3), and an evaluation roadmap (Section 7, Figures 9, Tables 5–7).

## 2. Context and Motivation
- Problem addressed
  - LLMs and many “agents” that wrap them are fundamentally static: they do not adapt their internal parameters or non‑parametric components (memory, tools, workflows) as they interact with the world (Introduction, p. 4). This is a critical bottleneck for open‑ended, interactive deployments where tasks, tools, and environments change.
- Importance
  - Real‑world impact: Digital assistants, coding agents, GUI/web automation, healthcare, and education all require continual adaptation and robust retention of past experience (Sections 6.1–6.2).
  - Theoretical significance: The work gathers scattered ideas into a principled formulation of agent evolution as decision‑making in partially observable environments with explicit objectives (Section 2.1), providing foundations for analyzing safety, stability, and co‑evolution.
- Prior approaches and gaps
  - Curriculum learning, lifelong learning, model editing, and unlearning each cover a slice of the problem but focus mainly on parameter updates over static data and usually lack autonomous exploration, architectural self‑modification, or tool evolution (Section 2.2; Table 1).
  - Existing surveys treat evolution as a small component of agent taxonomies or study model self‑improvement divorced from tools, memory, and system architecture (Introduction; Section 2.2).
- Positioning
  - The survey provides the first end‑to‑end framework that:
    - Formally defines environments, agents, and self‑evolution as a transformation problem (Section 2.1; Eqs. 1–3).
    - Organizes the field across “what/when/how/where,” with cross‑cutting dimensions (online/offline, on/off‑policy, reward granularity) and evaluation principles (Figures 2–3, 7, 9; Tables 3–4, 5–7).

## 3. Technical Approach
This is a survey with a formal framework. The “approach” is the organization and definitions that make disparate work comparable.

- Formal problem setup (Section 2.1)
  - Environment as a POMDP: `E=(G,S,A,T,R,Ω,O,γ)`
    - `G`: goals (e.g., a user request).
    - `S`: environment states; `Ω`: observations the agent can read; `O`: observation model.
    - `A`: actions that include natural language, retrieval, and tool calls.
    - `T`: transition dynamics; `R`: feedback (a scalar or text) conditioned on goal `g∈G`.
    - `γ`: discount factor.
  - Agent system: `Π=(Γ,{ψ_i},{C_i},{W_i})`
    - `Γ`: architecture/topology (workflow or code graph organizing nodes `N_i`).
    - At each node `N_i`: model `ψ_i`, context `C_i` (prompt `P_i`, memory `M_i`), and tools `W_i`.
    - Policy at node `i`: `π_{θ_i}(·|o)` with `θ_i=(ψ_i, C_i)`; actions live in language space ∪ tool space `W_i`.
  - Self‑evolving strategy (Eq. 1):  
    > `f(Π, τ, r) = Π′ = (Γ′, {ψ′_i}, {C′_i}, {W′_i})`  
    The agent transforms itself into a new system `Π′` based on the trajectory `τ` and feedback `r`.
  - Objective over a task sequence (Eq. 3):  
    > Maximize `Σ_j U(Π_j, T_j)` where `Π_{j+1} = f(Π_j, τ_j, r_j)` (Eq. 2).  
    `U` is a utility that can be derived from rewards, time, accuracy, robustness, etc.

- Taxonomy: what, when, how, where (Figures 2–3)
  - What to evolve (Section 3; Table 2)
    1) Models: update policies with self‑generated data, feedback, or RL (e.g., SCA generates and solves code‑tasks; SELF, SCoRe, PAG use execution traces or critiques as signals; TextGrad treats textual feedback as “gradients”).
    2) Context: memory and prompt evolution.
       - Memory management mechanisms (Section 3.2.1): e.g., SAGE uses the Ebbinghaus curve to decide what to forget; Mem0 supports ADD/MERGE/DELETE to maintain coherent long‑term memory; Agent Workflow Memory stores reusable sub‑task workflows.
       - Prompt optimization (Section 3.2.2): search‑based (APE), iterative rewriting (ORPO), “textual gradient” edits (ProTeGi), MCTS (PromptAgent), evolutionary (PromptBreeder), and fully self‑supervised loops (SPO).
    3) Tools: creation (Voyager skill library, CREATOR abstracts tool creation), mastery via iterative refinement (LearnAct, DRAFT), and scalable management/selection (ToolGen encodes tools as tokens; AgentSquare searches modular agent designs; Darwin Gödel Machine rewrites its own code) (Section 3.3).
    4) Architecture: optimize single‑agent nodes and code (TextGrad; Gödel Agent; AlphaEvolve) and evolve multi‑agent workflows (ADAS, AFlow with MCTS; ScoreFlow/FlowReasoner learn to generate query‑specific workflows) or learn coordination via MARL (ReMA, GiGPO) (Section 3.4).
  - When to evolve (Section 4; Figure 5)
    - Intra‑test‑time (during solving the current task): via in‑context learning (ICL), supervised fine‑tuning (SFT), or reinforcement learning (RL).
      - Examples: Reflexion stores natural‑language reflections mid‑episode; AdaPlanner revises plans on out‑of‑plan feedback using an `ask_LLM()` action; Self‑Adapting LMs produce “self‑edits” that trigger immediate SFT; LADDER triggers targeted test‑time RL for hard problems.
    - Inter‑test‑time (between tasks): offline or online learning over collected trajectories.
      - Examples: SELF/STaR/Quiet‑STaR/SiriuS for self‑training with self‑generated rationales; RAGEN/DYSTIL/WebRL/DigiRL for RL across multi‑turn environments.
  - How to evolve (Section 5; Figure 6; Table 3; Figure 7)
    - Reward‑based: textual feedback (Reflexion, Self‑Refine), internal rewards (confidence/certainty), external rewards (environment, verification rules, majority vote), and implicit rewards (in‑context RL or logits‑derived “endogenous” rewards).
    - Imitation/demonstration: self‑generated (STaR and variants), cross‑agent (SiriuS), and hybrid (RISE, confidence‑filtered).
    - Population‑based/evolutionary: single‑agent (Darwin Gödel Machine code evolution; GENOME parameter evolution; self‑play methods like SPIN/SPC/STL) and multi‑agent (EvoMAC team/backprop‑like updates; Puppeteer learning orchestration; MDTeamGPT/MedAgentSim knowledge‑base evolution).
    - Cross‑cutting dimensions (Figure 7; Section 5.4; Table 4): online vs offline learning, on‑policy vs off‑policy, and reward granularity (process‑ vs outcome‑ vs hybrid).
  - Where to evolve (Section 6; Figure 8)
    - General‑purpose agents: memory mechanisms (Mobile‑Agent‑E “Tips/Shortcuts”), model‑agent co‑evolution (UI‑Genie co‑trains reward model and agent; WebEvolver co‑trains a world model), and curriculum‑driven training (WebRL adaptive curricula; Voyager’s bottom‑up tasks).
    - Specialized domains: coding (SICA, EvoMAC), GUI/web (WindowsAgentArena/Navi; WebVoyager; ReAP), finance (QuantAgent), medical (Agent Hospital, MedAgentSim, DoctorAgent‑RL), education, and more.

- Evaluation framework (Section 7; Figure 9; Tables 5–7)
  - Goals: adaptivity, retention, generalization, efficiency, safety (Table 5).
  - Paradigms: static, short‑horizon adaptation, and long‑horizon lifelong learning (Sections 7.2.1–7.2.3; Table 6).
  - Benchmarks: a catalog by domain (Table 7), plus long‑term memory (LTMBenchmark) and lifelong agents (LifelongAgentBench).

## 4. Key Insights and Innovations
1) A formal, general definition of self‑evolution in agents (Section 2.1)
   - Innovation: Eqs. (1)–(3) express self‑evolution as a transformation `f` over all agent components—not just weights—conditioned on observed trajectories and feedback. This unifies parameter updates, prompt/memory editing, tool creation, and workflow search as first‑class optimization targets.
   - Significance: Enables rigorous reasoning about adaptive agents beyond fine‑tuning and connects to utility maximization over task streams.

2) A comprehensive, actionable taxonomy of evolution (Figures 2–3; Sections 3–5)
   - Difference from prior work: Goes beyond “model self‑improvement” to cover non‑parametric context, tool ecosystems, and architecture (single vs multi‑agent), and ties each to specific methods (Table 2) and learning paradigms (ICL/SFT/RL).
   - Significance: Provides a design map—from prompt search to code‑level self‑modification—with concrete exemplars for each branch.

3) Cross‑cutting lenses that explain design trade‑offs (Figure 7; Table 4)
   - Novelty: The online/offline, on/off‑policy, and reward‑granularity axes expose why certain approaches are sample‑efficient but brittle (e.g., imitation), while others are stable yet expensive (e.g., outcome‑only RL).
   - Utility: These lenses guide practitioners to mix strategies (e.g., hybrid reward; offline SFT + online RL) for a target domain.

4) Evaluation program tailored to evolving agents (Section 7; Figure 9; Tables 5–7)
   - Contribution: Defines retention with explicit formulas for forgetting and backward transfer (FGT/BWT), distinguishes short‑ vs long‑horizon evaluation, and curates benchmarks that stress adaptation and memory (LTMBenchmark, LifelongAgentBench).
   - Impact: Shifts assessment from single‑shot accuracy to longitudinal competence and safety.

These are fundamental organizing contributions rather than incremental empirical improvements.

## 5. Experimental Analysis
This survey does not introduce a new model; instead, it synthesizes methods and evaluation practices. It still grounds effectiveness with representative evidence and provides a concrete evaluation blueprint.

- Evaluation methodology (Section 7)
  - Metrics (Table 5)
    - Adaptivity: success‑rate by iteration, adaptation speed.
    - Retention:  
      > Forgetting `FGT_t = (1/(t-1)) Σ_{i=1}^{t-1} (max_{j∈{i,…,t}} J_{j,i} − J_{t,i})`  
      > Backward transfer `BWT_t = (1/(t-1)) Σ_{i=1}^{t-1} (J_{t,i} − J_{i,i})`  
      where `J_{j,i}` is performance on task `i` after finishing task `j` (Section 7.1).
    - Generalization: aggregate cross‑domain performance and out‑of‑distribution tests.
    - Efficiency: token cost, time, action steps, tool productivity.
    - Safety: safety score, harm score, completion under policy (CuP), risk ratio, refusal rate, leakage rate.
  - Paradigms (Figure 9; Table 6)
    - Static assessment: end‑to‑end competence snapshots (e.g., AgentBench, SWE‑bench, OSWorld).
    - Short‑horizon adaptation: performance vs. iteration (examples in Section 7.2.2; MemoryAgentBench includes built‑in test‑time learning tasks).
    - Long‑horizon: lifelong memory/learning (LTMBenchmark; LifelongAgentBench), including dynamic/evolving test suites (Section 7.2.3).

- Benchmarks and datasets (Table 7; Section 7.2)
  - Web/GUIs: WebShop, WebArena, Mind2Web, BrowseComp; OSWorld; Mobile‑Eval‑E.
  - Software engineering: SWE‑bench (and variants).
  - Planning/Tools/Memory/Multi‑agent: PlanBench, ToolBench family, MemoryAgentBench, MultiAgentBench, SwarmBench.
  - General assistants: AgentBench, GAIA, TheAgentCompany.

- Representative quantitative evidence from cited systems within the survey (Section 6.2)
  - GUI agents:
    - WindowsAgentArena’s Navi agent “doubles” task‑completion after replay‑and‑critique self‑evolution (Section 6.2, “Graphical User Interfaces”).
    - WebVoyager improves success on unseen sites from “30% to 59%” via self‑fine‑tuning (Section 6.2).
    - ReAP adds episodic memory and “recovers a further 29‑percentage‑point margin” on previously failed queries (Section 6.2).
  - These are examples of self‑evolution’s impact across environments; the survey reports them to illustrate effectiveness, not as new experiments.

- Do the compiled results support the claims?
  - The evidence spans many domains and mechanisms, showing self‑evolution can:
    - Increase success rates through memory and plan revision (Sections 3.2, 4.1).
    - Learn tools from scratch and then master them (Section 3.3).
    - Benefit from hybrid evolution strategies (Table 4 shows trade‑offs motivating combinations).
  - Robustness considerations:
    - The survey highlights pitfalls such as “feedback friction” (agents under‑use external feedback; Table 3), reward sparsity (Sections 5.1, 5.4.3), distribution shift in off‑policy learning (Section 5.4.2), and expensive workflow search (Section 3.4.2), along with mitigations like Agentic Predictor (Section 3.4.2) and hybrid rewards (Section 5.4.3).

- Ablations/failure modes discussed
  - Reward granularity ablations: process‑ vs outcome‑based rewards and hybrid methods (Section 5.4.3) detail when each improves stability/learning signal density.
  - Stability/sample‑efficiency comparisons across method families (Table 4) serve as a qualitative ablation over design axes.

Overall, while no new experiments are run, the survey’s evaluation section makes the case for longitudinal, safety‑aware assessment and provides concrete metrics and benchmarks to implement it.

## 6. Limitations and Trade-offs
- Scope and assumptions
  - The framework assumes environments can supply some evaluative signal—textual or scalar—even if implicit (Section 5.1). In domains lacking verifiable outcomes or reliable critics, evolution can stall or drift.
  - The POMDP formalization presumes task‑conditioned rewards `R(s,a,g)`, which may be hard to craft or infer in messy real‑world settings (Section 2.1).
- Method‑level trade‑offs (Table 4; Figure 7)
  - Imitation/demonstration: high sample efficiency but brittle if demonstrations are biased or scarce.
  - Reward‑based RL: flexible but sensitive to reward design and sparsity; can reward‑hack without careful verification (Sections 5.1, 5.4.3).
  - Population‑based evolution: broad exploration and architectural novelty but compute‑intensive and slower to converge.
  - On‑policy vs off‑policy: on‑policy is stable but data‑hungry; off‑policy is efficient but risks distribution mismatch (Section 5.4.2).
  - Outcome vs process rewards: outcomes are cheap but sparse; process rewards are informative but require validation or annotation surrogates (Section 5.4.3).
- Practical constraints
  - Compute and latency: Dynamic reasoning, workflow search, and multi‑agent rollouts incur large costs during test‑time (Sections 3.4.2, 5.4.1; see also Figure 7’s online learning path).
  - Safety and controllability remain challenging in open environments; agents can leak sensitive data or pursue unsafe strategies despite constitutions or rules (Section 8.3; Table 5 safety metrics).
- Open questions
  - Catastrophic forgetting and stability‑plasticity balance in long‑horizon settings (Sections 7.1, 8.2).
  - Knowledge transfer across agents and tasks; evidence suggests current agents often fail to propagate learning reliably (Section 8.2).
  - Reliable evaluation under data contamination and evolving benchmarks (Section 7.2.3).

## 7. Implications and Future Directions
- Field‑level impact
  - By treating prompts, memory, tools, and architectures as evolvable components (Figures 2–3), the survey reframes “agent training” from model‑only fine‑tuning to full‑stack adaptation. This widens the design space for practical systems and aligns research on the path from “foundation agents” to “self‑evolving agents” and ultimately ASI (Figure 1).
- Enabled research directions (Section 8)
  - Personalized agents: self‑generated preference data, cold‑start‑aware memory and profile building, and evaluation that tracks evolving personalization quality.
  - Generalization: scalable architectures, test‑time scaling of compute, meta‑learning for cross‑domain adaptation, and explicit mechanisms to form robust world models.
  - Safety and control: richer constitutions and case libraries, improved feedback verification and sandboxing, privacy‑preserving memories, and safer training algorithms.
  - Multi‑agent ecosystems: dynamic role formation, orchestration learning (e.g., Puppeteer), knowledge‑base evolution (MDTeamGPT), and dynamic benchmarks capturing long‑term collaboration.
- Practical applications
  - Software engineering: self‑improving coding agents that evolve workflows and tests (Section 6.2).
  - GUI/web automation: memory‑augmented, curriculum‑trained agents that adapt to new apps/sites (Section 6.2).
  - Healthcare and education: hospital‑scale simulations for evolving diagnostic strategies; multi‑agent tutors that refine pedagogy via self‑play and feedback (Section 6.2).
  - Finance and science: self‑evolving consultation/trading frameworks; principle‑aware scientific discovery with multi‑agent collaboration (Sections 6.2, 5.3.2).

In short, the survey provides a rigorous blueprint—formalism, taxonomy, and evaluation toolkit—that allows researchers and practitioners to design, analyze, and measure truly adaptive agentic systems. It clarifies which levers to pull (what/when/how), where to apply them, and how to know whether evolution is working over the short and long term.
