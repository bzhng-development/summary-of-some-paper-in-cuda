# A Survey of Reinforcement Learning for Large Reasoning Models

**ArXiv:** [2509.08827](https://arxiv.org/abs/2509.08827)

## 🎯 Pitch

This comprehensive survey establishes the first unified framework for applying reinforcement learning (RL) to Large Reasoning Models (LRMs)—advanced language and multimodal models designed for complex, multi-step reasoning. By synthesizing recent advances in reward design, policy optimization, scalable infrastructure, and open problems, it distinguishes RL for reasoning from alignment-focused approaches like RLHF/DPO and clarifies how to scale LLMs into verifiable, agentic, and generalist problem solvers. This work matters because it lays the methodological foundation for transforming LLMs into AI systems capable of planning, tool use, and scientific reasoning, offering a roadmap for the next wave of practical and scientific breakthroughs in AI.

---

## 1. Executive Summary (2-3 sentences)
This survey systematizes the fast‑moving field of reinforcement learning (RL) for Large Reasoning Models (LRMs)—language and multimodal models whose core capability is multi‑step reasoning rather than only instruction following. It builds a unified framework for how to design rewards, optimize policies, sample data, and evaluate progress; clarifies open controversies (e.g., whether RL “discovers” new skills or mainly “sharpens” existing ones); and compiles the resources and infrastructure required to scale RL beyond alignment toward verifiable, agentic, and multimodal reasoning (Figures 1–2, Sections 3–7).

## 2. Context and Motivation
- Problem the paper addresses
  - There is no consolidated playbook for turning general LLMs into LRMs that reliably plan, reason, and act under long horizons. Prior post‑training—RLHF and DPO—focused on aligning behavior to preferences, not on incentivizing correct reasoning (Figure 2). This survey answers how to design and scale RL specifically for reasoning across tasks such as math, code, tools, agents, robotics, and medicine (Sections 1–2, 6).

- Why this matters
  - Practical impact: Verifiable tasks such as competition math and coding benefit from automated rewards, enabling rapid capability gains and reducing reliance on scarce human labels (Sections 3.1.1, 6.1). Agentic and tool‑use systems need turn‑by‑turn feedback to learn to plan, call tools, and correct themselves (Section 6.2).
  - Theoretical significance: RL introduces a new scaling axis—train‑time interactions and test‑time “thinking” budget—that is orthogonal to data/parameter scaling (Section 1; Figure 2).

- Limitations of prior approaches
  - RLHF/DPO rely on learned reward models or preferences; these are noisy outside well‑specified domains and susceptible to reward hacking (Sections 1, 3.1.1, 4.5).
  - Supervised fine‑tuning (SFT) often memorizes surface patterns and can cause catastrophic forgetting when distribution shifts (Section 4.2).

- Positioning relative to existing work
  - The survey centers RL for reasoning, not just alignment. It formalizes how LMs fit into the RL loop (Figure 3; Section 2.1), maps the algorithmic space (Tables 2–3; Section 3), distills foundational debates (Section 4), curates training resources (Tables 4–6; Section 5), and inventories applications in coding, agents, multimodality, multi‑agent, robotics, and medicine (Figure 6; Section 6). It closes with research roadmaps (Section 7).

## 3. Technical Approach
This is a structured survey. Its “method” is an organizing framework and precise formalization of RL for LRMs.

- Mapping LMs to the RL loop (Section 2.1; Figure 3)
  - State `s_t`: the prompt plus tokens generated so far, i.e., `(x, a_1: t-1)`.
  - Action `a_t`: a next token, a segment, or an entire sequence; the “granularity” matters for reward and credit assignment (Table 2).
  - Transition: deterministic string concatenation `s_{t+1} = [s_t, a_t]` until EOS.
  - Reward: can be sequence‑level (sparse) or token/step/turn‑level (dense) (Table 2).
  - Objective: maximize expected return J(θ) over the data distribution (Eq. (1)).

- Reward design taxonomy (Section 3.1)
  - Verifiable rewards (Section 3.1.1): rule‑based correctness/format checks—e.g., boxed math answer equality or unit tests for code. “Verifier’s Law” highlights that tasks are easiest to train when feedback is automatically checkable.
  - Generative rewards (Section 3.1.2): learned “judges” that reason before scoring. Two families:
    - Model‑based verifiers for verifiable tasks to handle formatting brittleness.
    - Assessment‑based generative reward models (GenRMs) for subjective tasks using chain‑of‑thought (CoT) critiques or rubric‑guided evaluation; can co‑evolve with the policy.
  - Dense rewards (Section 3.1.3): step/token/turn‑level signals via process reward models (PRMs), Monte‑Carlo attribution, or explicit per‑turn supervision for tool‑use/agents (Table 2).
  - Unsupervised rewards (Section 3.1.4): no human labels—derive signals from model consistency, internal confidence/entropy, self‑generated knowledge (“self‑rewarding”), or data‑centric heuristics and corpora.
  - Reward shaping (Section 3.1.5): combine verifiers with reward models and structure advantages at the group/set level (e.g., group baselines in GRPO; aligning to Pass@K).

- Policy optimization landscape (Section 3.2)
  - General PPO‑style objective (Eq. (5)): clipped ratio and advantage estimate.
  - Critic‑based algorithms (Section 3.2.2): PPO with value models and GAE (Eqs. (6)–(9)) provides token‑level signals but adds compute/instability under long horizons.
  - Critic‑free algorithms (Section 3.2.3): REINFORCE (Eq. (10)); GRPO (Eqs. (11)–(12)) replaces token‑level values with group‑relative sequence advantages—simple, scalable with verifiable rewards.
  - Off‑policy optimization (Section 3.2.4; Eq. (13)): learn from replay/asynchronous data and offline corpora; also hybrid SFT+RL loss or data mixing.
  - Regularization (Section 3.2.5): KL penalties to a reference or old policy (Eq. (14)); entropy regularization to avoid entropy collapse (Eq. (15)); length penalties to control thinking cost.

- Sampling strategy (Section 3.3)
  - Dynamic sampling (Section 3.3.1): focus rollouts on medium‑difficulty or under‑mastered items; curriculum and prioritized replay; encourage exploration where uncertainty/entropy is high.
  - Structured sampling (Section 3.3.1): tree sampling (MCTS‑like) for node‑level process signals; shared‑prefix/segment rollouts to reuse compute.
  - Hyperparameters (Section 3.3.2): tuning temperature, entropy targets, clipping bounds, and staged context lengthening (e.g., 8k→32k) to balance exploration and cost.

- Empirical scaffolding compiled by the survey
  - Frontier models timeline and coverage (Section 2.2; Figure 4; Table 1).
  - Static corpora (Section 5.1; Table 4) and dynamic environments (Section 5.2; Table 5).
  - RL infrastructure (Section 5.3; Table 6): training runtimes, serving, distributed rollouts.

## 4. Key Insights and Innovations
- A unified formal and practical map of RL for LRMs
  - What’s new: The survey aligns notation (Figure 3; Eq. (1)), clarifies action/reward granularities (Table 2), and connects algorithm families and sampling/regularization choices (Sections 3.1–3.3; Table 3).
  - Why it matters: Practitioners can plug‑and‑play components—verifier/GenRM, GRPO/PPO, dynamic sampling—rather than reinvent pipelines.

- Verifier‑centric scaling principle (“Verifier’s Law”)
  - Content: “The ease of training AI systems to perform a task is proportional to the degree to which the task is verifiable” (Section 3.1.1). Math and code reward pipelines succeed because they offer automatic, precise, and scalable feedback.
  - Significance: Explains why RLVR (rule‑based RL with verifiable rewards) has rapidly advanced math/code reasoning (Sections 1, 2.2, 6.1), and frames the challenge of open‑ended tasks where only GenRMs or rubrics are viable (Section 3.1.2).

- Process‑level credit assignment for long‑horizon reasoning
  - Content: The survey organizes dense reward techniques (token/step/turn in Table 2 and Section 3.1.3) including PRMs, Monte‑Carlo step attribution, turn‑level evaluators for tool calls, and tree rollouts.
  - Significance: These are the mechanisms that reduce variance and improve sample efficiency relative to solely outcome‑level rewards (Section 3.1.3 “Takeaways”).

- Clarity on contentious issues that guide scaling decisions (Section 4)
  - Sharpening vs. discovery (§4.1): Evidence that RL improves Pass@1 but can shrink exploration (Limit‑of‑RLVR), alongside counter‑evidence showing extended RL grows both Pass@1 and Pass@K and enables composition of new skills.
  - RL vs. SFT (§4.2): Empirical patterns—“SFT memorizes, RL generalizes”—with important caveats and unified/alternating paradigms that often work best.
  - Model priors (§4.3): RL responsiveness differs across families (e.g., Qwen vs. Llama); mid‑training/annealing with math/code corpora can make weaker priors more RL‑friendly.

- Complete ecosystem view
  - Datasets (Table 4), gyms/worlds (Table 5), and infra (Table 6) are cataloged so one can reproduce end‑to‑end RLVR/RL‑agent training. Few surveys provide this breadth (Sections 5–6).

## 5. Experimental Analysis
This survey synthesizes results across many works rather than running new experiments. It still specifies evaluation norms, representative findings, and caveats.

- Evaluation methodology compiled
  - Tasks: math and code (verifiable); tool‑use and agents (turn‑level signals); multimodal images/videos/3D; robotics and medicine (Sections 6.1–6.6).
  - Metrics:
    - Pass@1 and Pass@K for problem solving; the survey notes that optimizing for Pass@1 can hurt broad exploration and introduces Pass@K‑aligned objectives and credit shaping (Section 3.1.5; Section 4.1).
    - For agents: per‑turn success, tool correctness, and environment‑level completion (Sections 3.1.3 Turn‑level; 6.2).
    - For generation: aesthetic/consistency/physics scores for images/videos; rubric‑based ratings for subjective tasks (Sections 3.1.2, 6.3).
  - Setups: On‑policy RL (GRPO/PPO), critic‑free vs critic‑based, replay and asynchronous architectures (Sections 3.2–3.3; Table 6).

- Representative quantitative patterns reported from the literature
  - Smooth scaling with more RL compute and test‑time thinking for o1/R1‑style systems (Section 1; Figure 2 discussion).
  - Group‑relative advantages (GRPO, Eq. (12)) stabilize critic‑free training on verifiable tasks; PPO/GAE (Eqs. (6)–(9)) offer finer token‑level control but are heavier and sensitive to noise (Sections 3.2.2–3.2.3; Table 3).
  - Evidence tension in §4.1:
    - Quote (sharpening view): “Pass@K evaluations indicate that RL enhances Pass@1 performance, yet tends to underperform relative to base models when sampling broadly at large‑K Pass@K” (Section 4.1).
    - Quote (discovery view): “ProRL v2 … demonstrates stronger results,” including improved Pass@1 and Pass@K via prolonged RL with engineering advances (Section 4.1); and “LLMs can learn new skills in RL through composition of existing capabilities” (Section 4.1).
  - RL vs. SFT: “SFT memorizes, RL generalizes” across textual and vision settings (Section 4.2 summarizing GeneralPoints/V‑IRL); however, SFT warm‑ups and unified SFT+RL objectives often yield the best stability and transfer (Section 4.2).

- Ablations/failure modes highlighted
  - Reward hacking and “reasoning illusions” when rewards are poorly specified or when learned reward models are brittle (Sections 3.1.2, 4.5).
  - Entropy collapse without either explicit entropy terms (Eq. (15)) or exploration‑promoting tricks such as asymmetric clipping and advantage shaping (Section 3.2.5; 3.3.2).
  - Model‑family dependence and mid‑training sensitivity (Section 4.3).
  - Length‑performance trade‑offs and staged context curricula to prevent wasteful long CoT (Section 3.2.5 Length Penalty; Section 3.3.2).

- Does the evidence support the claims?
  - The survey refrains from a single leaderboard; instead, it triangulates mechanisms and consistent patterns across many sources and explicitly flags points of contention (Section 4). When claims are conditional (e.g., RL helps more on verifiable domains), the conditions and counter‑examples are provided.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Verifiability assumption: many successes hinge on tasks with automatic checkers; subjective domains still need GenRMs/rubrics and are harder to scale (Sections 3.1.1–3.1.2).
  - Stationarity and on‑policy sampling: PPO/GRPO assume manageable drift; large asynchrony, quantized inference, or off‑policy data requires careful importance weighting and replay design (Section 3.2.4).

- Scenarios not fully addressed
  - Open‑ended creativity and safety alignment under sparse or conflicting human preferences remain challenging; rubric/GenRM solutions are evolving (Section 3.1.2).
  - Cross‑domain, long‑horizon, multi‑step tasks without clear intermediate verifiers still lack robust process credit assignment (Section 3.1.3 “Takeaways”).

- Computational and data constraints
  - RL rollouts add a heavy training inference cost; long CoT and large group sizes increase memory/latency (Sections 3.3.2, 4.4).
  - Critic models (PPO/GAE) add compute overhead and instability under noisy rewards (Section 3.2.2).
  - Dynamic environments and tool ecosystems require significant engineering (Tables 5–6).

- Open questions and weaknesses called out
  - Exact role of KL regularization for reasoning RL—some pipelines remove it entirely while others find it essential (Section 3.2.5).
  - Whether RL primarily “sharpens” or truly “discovers” depends on model prior, reward design, and training scale (Section 4.1).
  - Evaluation gaps: inconsistent OOD benchmarks; distinguishing generalization from data contamination (Section 4.2).

## 7. Implications and Future Directions
- How this work changes the landscape
  - Provides a blueprint to build LRMs with RL beyond alignment: choose verifiable or generative rewards; pick critic‑free GRPO for scalable verifiable tasks or critic‑based PPO/PRM when dense token‑level signals are needed; use dynamic sampling and length control; exploit replay/asynchrony when helpful (Sections 3–5).
  - Gives practitioners a catalog of environments, datasets, and frameworks to launch large‑scale projects quickly (Tables 4–6).

- Concrete follow‑ups and research opportunities (Section 7)
  - Continual RL for LRMs (§7.1): lifelong, multi‑stage training that balances stability vs plasticity—experience replay and policy reuse tailored to language agents.
  - Memory‑based RL (§7.2): turn per‑task memory into an experience substrate shared across tasks; learn policies that manage and compose memory.
  - Model‑based RL (§7.3): build world models (text/vision) to simulate environments and generate robust state/reward signals.
  - Efficient reasoning (§7.4): learn compute‑allocation and halting policies—instance‑adaptive reasoning depth rather than uniform long CoT; formalize cost‑performance trade‑offs.
  - Latent‑space reasoning (§7.5): move from token‑space CoT to continuous latent reasoning, then design reward/advantage signals for latent trajectories.
  - RL for pre‑training (§7.6): reframe next‑token prediction as RL with corpus‑derived rewards; explore unsupervised/self‑rewarding at scale (Eq. (13) context).
  - RL for diffusion‑based LLMs (§7.7): address ELBO/likelihood estimation challenges and trajectory‑level rewards during denoising.
  - Scientific discovery (§7.8): couple RL with simulators and domain‑specific verifiers to replace slow/expensive wet‑lab feedback.
  - Architecture–algorithm co‑design (§7.9): treat routing/sparsity/expert activation as actions and jointly optimize capability and hardware efficiency.

- Practical applications
  - Coding and program repair with unit‑test/verifier rewards (Section 6.1).
  - Web, search, and tool‑use agents using turn‑level rewards and asynchronous rollouts (Section 6.2).
  - Multimodal reasoning and generation with visual/temporal verifiers or rubric‑guided GenRMs (Section 6.3).
  - Multi‑agent collaboration with RL‑trained language agents (Section 6.4).
  - VLA robotics trained with success/failure outcomes and replay in simulation (Section 6.5).
  - Medical reasoning with correctness, formatting, and rubric rewards across text and imaging (Section 6.6).

> Overall, Figures 1–2 frame the shift from alignment RL (RLHF/DPO) to reasoning‑oriented RL (RLVR and beyond); Figure 3 and Tables 2–3 specify the mechanics; Tables 4–6 furnish the ecosystem; Sections 4 and 7 articulate the debates and the path forward.
