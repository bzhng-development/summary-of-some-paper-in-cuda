# The Landscape of Agentic Reinforcement Learning for LLMs: A Survey

**ArXiv:** [2509.02547](https://arxiv.org/abs/2509.02547)

## 🎯 Pitch

This survey redefines the intersection of large language models and reinforcement learning by introducing 'Agentic RL,' a paradigm where LLMs are trained as autonomous, decision-making agents embedded in dynamic environments. By mapping more than 500 recent works, the paper builds a comprehensive taxonomy of agentic capabilities and task domains, and unifies the rapidly evolving landscape of algorithms, environments, and benchmarks. This fundamental shift—from static output alignment to adaptive, long-horizon agentic behavior—lays the groundwork for scalable, general-purpose AI agents that can robustly reason, plan, use tools, and improve themselves across real-world domains.

---

## 1. Executive Summary (2-3 sentences)
This survey defines “Agentic Reinforcement Learning (Agentic RL)” and formally distinguishes it from conventional preference-based LLM reinforcement fine-tuning (PBRFT). It builds a twofold taxonomy—by core agent capabilities and by task domains—and consolidates algorithms, environments, frameworks, and hundreds of recent systems, arguing that RL is the mechanism that converts static LLM modules (planning, tool use, memory, reasoning, etc.) into adaptive, robust, long-horizon behavior (Sections 1–5; Figures 1–4; Tables 1–2).

## 2. Context and Motivation
- Problem addressed
  - Prior “LLM RL” work overwhelmingly optimizes single-turn output quality on fixed datasets—e.g., RLHF and DPO—what the paper calls PBRFT (preference-based reinforcement fine-tuning). This corresponds to a degenerate, one-step decision problem (horizon T=1) (Section 2; Eq. (1); Table 1).
  - Real agents must operate as sequential decision-makers in dynamic, partially observable environments: planning, acting, invoking tools, updating memory, and reasoning over long horizons (Section 1).
- Why it matters
  - Practical: Many high-impact applications (web research, GUI automation, software engineering, robotics) demand multi-step interaction, uncertainty handling, and credit assignment (Sections 4–5).
  - Scientific: It reframes LLMs as policies in POMDPs, which opens the door to principled exploration, state/action abstraction, and verifiable feedback (Section 2.1).
- Prior approaches and gaps
  - Alignment/post-training: RLHF, RLAIF, DPO, KTO, etc., align outputs but treat models as text generators, not agents (Section 1; Section 2).
  - Prompted agent pipelines (e.g., ReAct) interleave “think-act-observe” but lack learning signals and adaptivity; SFT-only “agent tuning” imitates heuristics without robust generalization (Section 3.2).
  - Terminology is inconsistent across domains; evaluation environments are fragmented (Section 1; Section 5).
- How this work positions itself
  - Provides a formal MDP/POMDP grounding for Agentic RL, unifying action spaces (`Atext` ∪ `Aaction`), transitions, rewards, and objectives distinct from PBRFT (Section 2; Eqs. (5)–(11); Table 1; Figure 2).
  - Proposes dual taxonomies: capability-centric (planning, tool use, memory, self-improvement, reasoning, perception, others; Section 3) and task-centric (search, code/SWE, math, GUI, vision/embodied, multi-agent; Section 4).
  - Curates algorithms (REINFORCE, PPO, DPO, GRPO families; Section 2.7; Table 2), environments/benchmarks (Section 5; Table 9), and frameworks (Section 5.2; Table 10).
  - Surfaces open challenges around trustworthiness, scaling training, and scaling environments (Section 6).

## 3. Technical Approach
This is a survey. Its “approach” is the conceptual and formal framework it uses to unify disparate results and systems.

1) Formalization: From PBRFT to Agentic RL (Section 2; Figure 2)
- Core idea in plain language
  - PBRFT: Treats each prompt→response as a single-step decision with immediate reward (no state changes). Useful for alignment but not for multi-step behavior.
  - Agentic RL: Treats an LLM as a policy in a sequential decision loop where it receives observations, can generate text and/or structured actions, queries tools, updates memory, and accumulates rewards over time.
- Key definitions (notation introduced then explained)
  - PBRFT MDP (Eq. (1)): ⟨`Strad`, `Atrad`, `Ptrad`, `Rtrad`, γ=1, T=1⟩. Only one state (prompt), deterministic transition to terminal, reward r(a) on the output.
  - Agentic RL POMDP (Eq. (2)): ⟨`Sagent`, `Aagent`, `Pagent`, `Ragent`, γ, `O`⟩ with partial observability `ot = O(st)`. The agent acts repeatedly, environment evolves with uncertainty (Eq. (7)).
  - Unified action space (Eq. (5)): `Aagent = Atext ∪ Aaction`.
    - `Atext`: free-form language not directly changing the external world.
    - `Aaction`: structured calls (e.g., tool invocations like `call("search","Einstein")`, or environment actions like `move("north")`). Actions can be composite sequences of primitives.
  - Rewards (Eq. (9)): allow sparse “task completion” reward and dense “sub-reward” shaping, potentially learned/verifiable (unit tests, symbolic checks).
  - Objectives: PBRFT maximizes expected response reward (Eq. (10)); Agentic RL maximizes discounted cumulative reward over trajectories (Eq. (11)).

2) RL algorithm families unified under this lens (Section 2.7; Table 2; Eqs. (12)–(17))
- REINFORCE (Eq. (12)): First-principles policy gradient with a variance-reducing baseline.
- PPO (Eqs. (13)–(14)): Stable updates via clipping the probability ratio and using a learned value critic; widely used in LLM alignment.
- DPO (Eq. (15)): Direct preference optimization reframes KL-regularized reward maximization as likelihood on preference pairs—no separate reward model needed. Many variants (β-DPO, SimPO, IPO, KTO, ORPO, Step-DPO, LCPO).
- GRPO (Eqs. (16)–(17)): Group Relative Policy Optimization replaces a large critic with groupwise relative rewards/advantages; sample-efficient for long reasoning/tool-use; numerous extensions (DAPO, GSPO, GMPO, ProRL, Dr.GRPO, Step-GRPO, etc.) summarized in Table 2.
  - Why GRPO in agents? Group-based normalization stabilizes long-horizon, multimodal signals, and avoids heavy critics, which is valuable for agent rollouts (Section 2.7; Table 2).

3) Capability taxonomy tied to RL mechanisms (Section 3; Figure 3–4)
- Planning (Section 3.1)
  - RL as external guide: Train value/heuristic models to guide search (e.g., MCTS in RAP/LATS). LLM proposes; RL-trained critic prunes/explores (Figure 4, “RL as External Guide”).
  - RL as internal driver: Optimize the LLM policy to generate better plans directly from environment feedback (e.g., ETO, VOYAGER, Planner-R1) (Figure 4, right half).
- Tool use (Section 3.2; Figure 5)
  - From prompted/SFT ReAct pipelines (imitative) to tool-integrated RL (outcome-driven), where the policy learns when/which/how to call tools and to interleave code execution with text reasoning (ToolRL, ToRL, ReTool, ARTIST, etc.).
- Memory (Section 3.3; Table 3)
  - From static RAG stores to RL-controlled memory policies: learn to add/update/delete memory, and even maintain latent “memory tokens” updated by RL (MemAgent, MEM1, Memory-R1, MemoryLLM/M+).
- Self-improvement (Section 3.4)
  - From verbal self-correction at inference time (Reflexion, Self-Refine) to internalizing reflection via RL (KnowSelf, Reflection-DPO, SWEET-RL), to autonomous self-training/self-play loops (R-Zero, Absolute Zero, Self-Evolving Curriculum, TTRL).
- Reasoning (Section 3.5)
  - Dual-process framing: fast vs. slow reasoning and RL’s role in stabilizing long, verifiable chains-of-thought (R1-style RLVR).
- Perception (Section 3.6)
  - Extending R1-style RL to vision/audio: verifiable rewards (IoU, mAP), grounded CoT with visual evidence tokens, and tool-driven or generated visual “imagination”.

4) Task taxonomy and evaluation mapping (Section 4; Figure 6)
- Search & research agents (Table 4)
- Code & SWE agents (Table 5)
- Math agents (Table 6)
- GUI agents (Table 7)
- Vision/embodied/multi-agent (Sections 4.5–4.7)
- The survey explains how RL variants and reward designs are adapted per domain (e.g., unit tests in coding, verifier feedback in formal math, click-level rewards in GUI, spatial-temporal metrics in video).

5) Infrastructure consolidation (Section 5; Tables 9–10; Figure 1)
- Environments: Web, GUI, coding/SWE, games, and general-purpose RL gyms (Table 9).
- Frameworks: Agentic-RL, RLHF/LLM-RL, and general RL libraries (Table 10).

Design choices and rationale
- Unifying `Atext` and `Aaction` captures both conversational and world-altering decisions, a core difference from text-only PBRFT (Section 2.3).
- POMDP formalization is necessary for partial observability and memory (Section 2.1–2.2).
- Groupwise policy optimization (GRPO and successors) is emphasized because it scales to long rollouts and mixed, verifiable reward signals without heavy critics (Section 2.7; Table 2).

## 4. Key Insights and Innovations
- Formal, symbolic separation of PBRFT vs. Agentic RL (fundamental)
  - What’s new: A clean POMDP treatment with partial observability, hybrid actions, stochastic transitions, and long-horizon objectives (Eqs. (2), (5)–(11); Table 1; Figure 2).
  - Why it matters: Clarifies why single-turn “alignment RL” is insufficient for agents; sets a common language for future algorithmic work.
- Unified action model for agent LLMs (incremental but enabling)
  - `Aagent = Atext ∪ Aaction` (Section 2.3) standardizes how to model tool calls, environment control, and free-form text. Enables joint optimization of language and actions by a single policy.
- Tool-Integrated Reasoning (TIR) under RL (significant capability advance)
  - The survey synthesizes a line of work showing outcome-driven RL transforms brittle ReAct-style pipelines into adaptive controllers that combine symbolic computation and verbal reasoning (Section 3.2; Figure 5). It highlights theoretical arguments for why TIR extends “text-only” RL (ASPO; Section 3.2).
- Memory as a trainable sub-policy (emerging capability)
  - RL-controlled memory management at explicit-token and latent-token levels improves long-context stability and continual adaptation (Section 3.3; Table 3).
- Cross-domain RLVR (R1-style) extension to multimodal and formal settings (broad impact)
  - Evidence that verifiable rewards and process shaping scale to vision/video (IoU/mAP/Spatial rewards), audio, and formal theorem proving (Section 3.6; Section 4.3.2).

## 5. Experimental Analysis
Note: As a survey, this paper aggregates results rather than running its own experiments. It presents task-specific methodologies, metrics, and headline outcomes.

- Evaluation methodologies summarized
  - Search & Research (Section 4.1; Table 4)
    - Environments/benchmarks: real web APIs, BrowseComp (hard-to-find info). Rewards: answer correctness plus retrieval quality; step-wise rewards in some systems (StepSearch). Metrics: pass@1, report quality.
    - Headline result: “OpenAI Deep Research achieves 51.5% pass@1 on BrowseComp” (Section 4.1.2).
  - Code & SWE (Section 4.2; Table 5)
    - Rewards: unit test pass/fail, compiler messages, process-level signals, self-evolving testers (CURE).
    - Metrics: pass@k (HumanEval/MBPP), SWE-bench Verified, LiveCodeBench, DevBench.
    - Examples: “DeepSWE performs large-scale RL on realistic SWE tasks and achieves leading open-source results on SWE-bench–style evaluations” (Section 4.2.3). “Qwen3-Coder yields SOTA on SWE-Bench Verified with large-scale, long-horizon RL in 20k parallel envs” (Section 4.2.3).
  - Mathematical reasoning (Section 4.3; Table 6)
    - Informal math: correctness of final numeric/symbolic answers; sometimes process-aware hints.
    - Formal math: binary verifier success in Lean/Isabelle (miniF2F, ProofNet).
    - Examples: “rStar2-Agent (14B) achieves 80.6% (AIME24) and 69.8% (AIME25) pass@1 in 510 RL steps using GRPO-RoC and Python execution” (Section 4.3.1). “DeepSeek-Prover-v1.5/v2 and Leanabell-Prover-v2 scale verifier-based RL and subgoal decomposition in Lean4” (Section 4.3.2).
  - GUI agents (Section 4.4; Table 7)
    - Static vs. interactive settings; rewards include format/parameter correctness, sub-goal success, trajectory recovery, environment-provided success signals.
    - Systems like GUI-R1, UI-R1, InFiGUI-R1, WebAgent-R1, UI-TARS(-2), ZeroGUI, MobileGUI-RL, ComputerRL show rising success in long-horizon, realistic desktop/mobile/web environments.
  - Vision/Embodied agents (Sections 4.5–4.6)
    - Vision: verifiable spatial rewards (IoU, mAP), spatial-temporal consistency for video; grounding tokens and tool-use under RL (GRIT, Ground-R1, VTool-R1).
    - Embodied: navigation/manipulation with VLA models; rewards from path alignment (VLN-R1) or trajectory quality (TGRPO); S2E shows safety-aware RL for generalization.
  - Multi-agent (Section 4.7; Table 8)
    - Dec-POMDP framing with MAGRPO, MARFT, MAPoRL; RLCCF uses self-consistency voting for pseudo-labels; Chain-of-Agents distills multi-agent traces then applies agentic RL.

- Quantitative examples (selected, as reported)
  - BrowseComp pass@1: 51.5% for OpenAI Deep Research (Section 4.1.2).
  - rStar2-Agent: 80.6% (AIME24) and 69.8% (AIME25) pass@1 with only 510 RL steps (Section 4.3.1).
  - Multiple code systems report improvements on SWE-bench Verified; Qwen3-Coder and DeepSWE are cited as open-source leaders (Section 4.2.3). Exact numbers vary across papers; the survey compiles methods and settings in Table 5.
- Ablations and robustness
  - Algorithm ablations are discussed at the family level (Table 2): DAPO (decoupled clipping), GSPO (sequence-level clipping), Posterior-GRPO (reward only successful processes), Dr.GRPO (unbiased objective), Step-GRPO (step-wise rewards).
  - Reward shaping and process rewards consistently improve stability and credit assignment in long-horizon tasks (Sections 3.1–3.2; 4.2; 4.3).
- Do the results support the claims?
  - Across domains, the collected evidence shows RL (especially verifiable/process-aware rewards and groupwise optimization) improves long-horizon performance, tool-use efficiency, and robustness compared to SFT-only or prompt-based pipelines (Sections 3–4).
  - Caveats: Many strongest systems are closed-source (e.g., o3/Deep Research), making full comparisons difficult (Section 4.1.2). Benchmarks vary; not all metrics are standardized (Section 5).

> Table references for methods and ecosystems:
> - Table 2: PPO/DPO/GRPO families and variants.
> - Table 3: Memory categories and RL involvement.
> - Table 4–7: RL-based methods for Search, Code/SWE, Math, and GUI.
> - Table 9–10: Environments/benchmarks and RL frameworks.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Verifiable signals exist or can be synthesized (unit tests, symbolic verifiers, programmatic rules). Many open-ended tasks are hard to verify, which complicates reward design (Sections 3.2, 4.3, 6.3).
  - POMDP framing assumes access to observations and controllable actions/tools—nontrivial to sandbox and secure in real systems (Section 6.1).
- Edge cases not fully addressed
  - Temporal credit assignment over very long horizons remains difficult despite step-wise/process rewards (Sections 3.2 “Prospective: Long-horizon TIR”, 3.7; and 6.2).
  - Sim-to-real transfer for embodied agents is still a major bottleneck (real-robot RL is expensive; Section 4.6).
- Computational/data constraints
  - RL scaling is compute-intensive; training stability is sensitive to hyperparameters; sample efficiency remains a pain point (Section 6.2; also Section 2.7 notes PPO/GRPO trade-offs).
- Trustworthiness risks amplified by RL
  - Reward hacking, unsafe tool use, indirect prompt injection, sycophancy, and hallucination can be reinforced if reward models are misaligned (Section 6.1). The survey documents:
    - Security risks and defense-in-depth (sandboxing, adversarial training, anomaly detection).
    - Hallucination trade-offs under outcome-only RL (“hallucination tax”); mitigation via factuality-aware step-wise rewards and training on unanswerable cases.
    - Sycophancy exacerbated by RLHF-like signals; mitigations include sycophancy-aware reward models and co-optimizing policy+reward to close loopholes.

## 7. Implications and Future Directions
- How it changes the landscape
  - Provides a common formal language (POMDP, hybrid action space) and taxonomies that allow researchers to compare methods across domains and to design RL signals that target specific agentic capabilities (Sections 2–3).
  - Positions RL—not just SFT or prompting—as the central mechanism to internalize planning, memory control, and tool reasoning, leading to adaptive, stateful agents (Sections 3–4).
- Follow-up research enabled
  - Long-horizon credit assignment: step-/segment-level reward modeling, verifiable process rewards, selective rollouts (e.g., GRESO), and hierarchical RL to scale depth (Sections 3.2, 3.7; Table 2).
  - Meta-reflection: learning “how to reflect” as a meta-policy, dynamically choosing reflection modes based on task difficulty and long-term outcomes (Section 3.4 “Prospective”).
  - Structured memory under RL: graph/temporal memory with RL-managed insert/update/delete/link operations (Section 3.3 “Prospective”).
  - Co-evolving environments: automated reward modeling and curriculum/task generation (e.g., EnvGen) to sustain learning flywheels (Section 6.3).
  - Cross-domain RL scaling: principled recipes (e.g., Polaris), multi-domain curricula (Guru data), and efficiency advances (LitePPO, dynamic fine-tuning) (Section 6.2; Table 2).
- Practical applications
  - Deep research assistants that plan, browse, verify, and synthesize reports (Section 4.1).
  - Autonomous software engineers and MLE agents operating over repos, IDEs, and pipelines (Sections 4.2, 4.8).
  - Math and formal reasoning assistants with verifiable proofs and traceable subgoals (Section 4.3).
  - GUI operators for desktop/mobile/web automation; VLM agents for image/video/3D reasoning and control; embodied navigation/manipulation (Sections 4.4–4.6).
  - Multi-agent systems that coordinate specialized LLMs with RL for debate, planning, and execution (Section 4.7).

> Key figures/tables to skim if you only have minutes:
> - Figure 2 (paradigm shift): shows the jump from text-only RL to dynamic agent RL with tools, memory, and environments.
> - Table 1 (PBRFT vs Agentic RL): one-shot vs. long-horizon POMDP.
> - Table 2 (RL algorithms): quick map of PPO/DPO/GRPO ecosystems.
> - Figure 5 (tool use evolution) and Table 3 (memory designs): how RL “wires in” agentic capabilities.
> - Tables 4–7 (task-specific ecosystems): what’s been tried where—and with what signals.

Overall, this survey supplies the conceptual scaffolding, algorithmic menu, and empirical landscape needed to build, study, and benchmark truly agentic LLMs.
