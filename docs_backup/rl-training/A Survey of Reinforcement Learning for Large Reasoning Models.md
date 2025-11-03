# A Survey of Reinforcement Learning for Large Reasoning Models

**ArXiv:** [2509.08827](https://arxiv.org/abs/2509.08827)
**Authors:** Kaiyan Zhang, Yuxin Zuo, Bingxiang He, Youbang Sun, Runze Liu, Che Jiang, Yuchen Fan, Kai Tian, Guoli Jia, Pengfei Li, Yu Fu, Xingtai Lv, Yuchen Zhang, Sihang Zeng, Shang Qu, Haozhan Li, Shijie Wang, Yuru Wang, Xinwei Long, Fangfu Liu, Xiang Xu, Jiaze Ma, +17 more authors
**Institutions:** Tsinghua University (TsinghuaC3I)

## 🎯 Pitch

This paper introduces a unified framework for leveraging reinforcement learning in large reasoning models by systematically organizing the field into three critical components: reward design, policy optimization, and sampling strategies. By transforming dispersed methodologies into a coherent blueprint, it highlights how reinforcement learning can provide reasoning advantages over traditional supervised fine-tuning, enabling scalable, automated feedback particularly in domains like math and code. This new understanding paves the way for more effective AI models with enhanced problem-solving capabilities and opens avenues for practical applications in fields requiring complex reasoning.

---

## 1. Executive Summary (2-3 sentences)
This survey systematizes the fast‑moving area of “reinforcement learning for large reasoning models (LRMs)” by organizing it into three interacting pillars—reward design, policy optimization, and sampling—plus training resources and applications. Its significance is to turn scattered recipes from recent model releases (e.g., OpenAI o1, DeepSeek‑R1) into a coherent blueprint for scaling reasoning with reinforcement learning (RL), clarifying when and how RL yields genuine reasoning gains beyond standard supervised fine‑tuning (SFT).

## 2. Context and Motivation
- Problem/gap
  - Reasoning‑capable language models (LMs) have surged after systems like o1 and DeepSeek‑R1 used RL with automatically checkable (“verifiable”) rewards. Yet the field lacks a unifying map: What types of rewards actually scale? Which RL algorithms matter? How should we sample trajectories? What infrastructure and datasets are required? Where does RL help more than SFT—and why?
  - Section 1 and Figure 2 frame the shift from alignment‑oriented RLHF/DPO to “RL with Verifiable Rewards” (RLVR) that directly incentivizes task solving (math, code). The survey targets foundational components, open problems, resources, and applications (Figure 1).
- Importance
  - Practical: Verifiable domains (math, code, some scientific tasks) allow scalable automated feedback, unlocking training signals not bounded by human labeling (Sections 3.1.1 and 3.1.4).
  - Theoretical: RL introduces a new scaling axis—“train‑time RL” and “test‑time compute (thinking time)”—orthogonal to model size and pre‑training data (Section 1).
- Prior approaches and shortcomings
  - Alignment methods (RLHF, DPO) optimize preferences and safety but do not directly teach problem solving; they also depend on noisy learned reward models and humans (Section 1; Figure 2).
  - Pure SFT often memorizes solutions and can harm generalization to out‑of‑distribution tasks (Section 4.2).
- Positioning
  - The paper formalizes RL for LLMs (Section 2.1; Figure 3; Eq. (1)), then builds a taxonomy:
    - Reward design (verifiable, generative, dense/process, unsupervised, shaping) in Section 3.1.
    - Policy optimization (critic‑based vs. critic‑free, off‑policy, regularization) in Section 3.2.
    - Sampling strategies (dynamic/structured and hyper‑parameters) in Section 3.3.
  - It then examines five foundational controversies (Section 4), catalogs training resources (Tables 4–6), and synthesizes application domains (Figure 6).

## 3. Technical Approach
This is a survey, but it provides a precise conceptual and algorithmic framework that explains how RL for LRMs actually works.

- Formal problem setup (Section 2.1; Figure 3; Eq. (1))
  - Map language generation to an RL Markov Decision Process (MDP):
    - State `s_t`: the prompt plus tokens generated so far.
    - Action `a_t`: the next token (or segment/response).
    - Transition: deterministically concatenates `a_t` to the context.
    - Reward `R(x, y)`: may be given at sequence‑level (sparse), token‑level, step‑level (a “step” can be a sentence, a reasoning step, or a turn in an agent loop).
  - Objective: maximize expected return over prompts (Eq. (1)) with optional regularization toward a reference policy.
  - Granularity matters (Table 2): trajectory‑level rewards support simple bandit‑style updates; token/step/turn‑level rewards enable denser credit assignment.

- Pillar 1 — Reward design (Section 3.1)
  - Verifiable rewards (Section 3.1.1)
    - How it works: For math, enforce a parseable answer format (e.g., `\boxed{…}`) and compare against ground truth using programmatic checkers; for code, compile or run unit tests. Format constraints (special `<think>` and `<answer>` fields) ensure reliable parsing at scale.
    - “Verifier’s Law”: tasks become easy to train once reliable automated verification exists (Section 3.1.1).
  - Generative rewards (Section 3.1.2)
    - Model‑based verifiers reduce brittleness of rule systems (they judge semantic equivalence, not exact strings).
    - Reasoning reward models (RMs) “think before judging”: they generate critiques/rationales and then output a scalar preference or score; some are themselves trained with RL using verifiable meta‑rewards.
    - Rubric‑based rewards structure subjective evaluation into checklists (e.g., writing quality), enabling RL beyond pure correctness signals.
    - Co‑evolving systems unify policy and reward—either a single model self‑rewards (“self‑judges”) or the policy and RM are co‑optimized in one loop.
  - Dense rewards (Section 3.1.3; Table 2)
    - Token‑level signals: implicit PRMs (process reward models) induce token‑wise rewards from outcomes or learned oracles.
    - Step‑level signals: two strategies:
      - Model‑based PRMs to score intermediate steps (risk: reward hacking).
      - Sampling‑based Monte Carlo—branch the reasoning tree, evaluate outcomes of partial steps, back‑propagate credit; includes tree search (TreeRL, TreeRPO) and “force stopping” at intermediate points to estimate step values.
    - Turn‑level signals for multi‑turn agents: either explicitly reward each action‑result turn or decompose session‑level rewards back to turns (credit attribution).
  - Unsupervised rewards (Section 3.1.4)
    - Model‑specific (no external labels): majority/consensus voting across samples; internal confidence (entropy/probability/attention); self‑rewarding/self‑instruction curricula.
    - Model‑agnostic: heuristic rules (format/length) and data‑centric RL that reframes next‑token prediction as an RL problem (RPT).
  - Reward shaping (Section 3.1.5)
    - Rule‑based mixtures—combine verifiers with RMs to avoid 0/1 rewards and improve gradients.
    - Structure‑based shaping—group‑wise baselines over a set of candidates for the same prompt (e.g., GRPO), or transform rewards to align with Pass@K metrics.

- Pillar 2 — Policy optimization (Section 3.2)
  - Policy gradient objective (Section 3.2.1; Eq. (5))
    - Intuition: increase the probability of above‑average actions (`advantage > 0`) and decrease below‑average ones; PPO uses a clipped ratio to stabilize updates.
  - Critic‑based algorithms (Section 3.2.2)
    - PPO with a value function (“critic”) supplies token‑wise advantages via GAE (Eqs. (8)–(9)). Scales when you can reliably train a critic/reward model, but incurs extra compute and risk of reward hacking.
  - Critic‑free algorithms (Section 3.2.3)
    - REINFORCE and relatives: treat the whole sequence as one action; stabilize via baselines (e.g., greedy baseline in ReMax or leave‑one‑out in RLOO).
    - GRPO (Eq. (11)–(12)): compute a group‑relative advantage by normalizing each response’s reward by the mean/std over G candidates for the same prompt; apply PPO‑style clipping but no learned critic. This is favored in RLVR because rewards are reliable and the method is simpler and cheaper.
    - Enhancements: DAPO (decoupled clipping and dynamic sampling), CISPO (importance weighting), GSPO (sequence‑level clipping), VinePPO (Monte‑Carlo advantages), FlowRL (optimize reward distributions to avoid mode collapse).
    - Importance sampling: needed because rollouts lag parameter updates; most methods approximate token‑wise ratios; newer variants explore sequence‑level ratios (GSPO) or geometric means (GMPO) to reduce variance.
  - Off‑policy optimization (Section 3.2.4)
    - Learn from “older” trajectories or offline datasets; use replay buffers and truncated importance sampling to limit bias; combine SFT‑style losses with RL (UFT, SRFT, mixed‑policy training).
  - Regularization (Section 3.2.5)
    - KL regularization: toward a reference model or prior policy; opinions diverge—some remove KL entirely to let exploration diverge (beneficial in reasoning RLVR), others retain adaptive KL for stability.
    - Entropy regularization: maintain exploration but can destabilize sparse‑reward training; practical recipes include emphasizing high‑entropy tokens or constraining covariance between probabilities and advantages.
    - Length penalties: encourage concise reasoning, sometimes conditioned on difficulty.

- Pillar 3 — Sampling strategies (Section 3.3)
  - Dynamic sampling (Section 3.3.1)
    - Efficiency‑oriented: oversample “medium difficulty” questions that still yield non‑zero advantages; prioritize failure‑prone items; curriculum by category or difficulty; reuse rollouts via replay.
    - Exploration‑oriented: branch at high‑attention or high‑uncertainty steps; add guided prefixes or rubrics; keep “all‑wrong” items but inject intermediate guidance to bootstrap.
  - Structured sampling (Section 3.3.1)
    - Tree‑structured rollouts with Monte Carlo Tree Search (MCTS) and node‑level rewards; shared‑prefix/segment sampling to reuse KV caches and cut compute.
  - Hyper‑parameters (Section 3.3.2)
    - Temperature schedules to control exploration; staged context‑length curricula (e.g., 8k→16k→24k→32k) to teach efficient short reasoning before enabling long chains; mixed strategies for over‑length responses (masking vs. soft penalties).

- Training resources and system plumbing
  - Static corpora (Section 5.1; Table 4): curated math/code/STEM/agent datasets with verifiable outcomes and, increasingly, process traces.
  - Dynamic environments (Section 5.2; Table 5): programmatic logic/maths/code gyms, GUI/web agents, and model‑based arenas that provide interactive, dense feedback.
  - Infrastructure (Section 5.3; Table 6): open RL runtimes (TRL, OpenRLHF, Verl, AReaL, ROLL, slime, RLinf) built atop vLLM/SGLang for serving and FSDP/Megatron/DeepSpeed for training; many support asynchronous rollouts and agentic RL.

## 4. Key Insights and Innovations
- A unifying, mechanism‑level taxonomy of RL for LRMs (Section 3; Figure 5)
  - Novel because it ties concrete design choices (reward granularity, algorithm family, sampling topology) to failure modes (reward hacking, entropy collapse) and scaling levers (group baselines, tree sampling). This moves beyond mere lists of techniques to “how the pieces interact.”
- Centering “Verifier’s Law” for scalable reasoning (Section 3.1.1)
  - Fundamental insight: if a domain admits fast, reliable automated checks, RL can scale without humans. This explains why math and code are the leading edges (rule‑based checkers and unit tests) and why open‑ended writing still struggles (subjective rewards).
- A clear articulation of the RL vs. SFT boundary (Section 4.2)
  - Synthesizes evidence that “SFT memorizes, RL generalizes” under distribution shift, but also explains conditions where SFT plus careful weighting or warm‑up is beneficial and when RL is not a panacea.
- Reconciling the “Sharpening vs. Discovery” debate (Section 4.1)
  - New perspective: RL can both concentrate probability on latent correct modes (sharpening via reverse‑KL dynamics) and, given time and exploration, compose skills into new behaviors (discovery). The survey identifies metrics (Pass@K vs. CoT‑Pass@k) and training recipes that push one or the other.
- End‑to‑end view of resources and systems (Sections 5–6; Figure 6)
  - The compilation of static corpora (Table 4), dynamic environments (Table 5), and RL infrastructure (Table 6) provides a practical, reproducible path from research insight to deployed agentic systems.

## 5. Experimental Analysis
While this is a survey, it aggregates quantitative evidence, recipes, and ablations from many studies and provides structured comparisons.

- Evaluation methodology and scope
  - Models and timelines (Figure 4; Table 1): catalogs public and proprietary LRMs trained with RL (e.g., DeepSeek‑R1 671B MoE; QwQ‑32B; Intern‑S1 241B; Minimax‑M1 456B), along with algorithms such as GRPO, MPO, CISPO, and GSPO.
  - Algorithm comparison (Table 3): contrasts PPO/GRPO variants by advantage estimate, importance sampling, and loss aggregation level (token vs. sequence).
  - Reward/action granularity (Table 2): clarifies how returns are computed at trajectory, token, step, and turn levels.
- Representative quantitative findings cited in the survey
  - Length and efficiency:
    - “S‑GRPO … shortens sequence length by 35–61% across multiple benchmarks, with slight improvements in accuracy” (Section 4.4).
  - Generalization and data efficiency:
    - One‑shot RLVR “more than doubled MATH500 accuracy for a 1.5B model” and improved averages across multiple math benchmarks (Section 4.3).
  - Exploration and stability:
    - Dynamic sampling that filters all‑correct and all‑wrong batches (DAPO) yields “state‑of‑the‑art” AIME24 performance with reproducible recipes (Section 4.4).
  - Pass@K alignment:
    - Set‑level objectives and reward transformations (Walder & Karkhanis; Chen et al.) derive unbiased/low‑variance estimators to optimize Pass@K directly (Section 3.1.5).
- Resources and scale indicators
  - Static corpora contain up to millions of verifiable reasoning traces—for example, OpenMathReasoning at 5.5M (Table 4), AM‑DeepSeek‑R1‑0528‑Distilled at 2.6M, and MegaScience at 2.25M; code datasets like OpenCodeReasoning at 735K and rStar‑Coder at 592K.
  - Dynamic environments span logic puzzles (AutoLogi: 2,458/6,739 puzzles), GUI agents (AgentCPM‑GUI: 55K trajectories), and model‑based TextArena with 99 adversarial games (Table 5).
- Do experiments support the claims?
  - The compiled results consistently show that RL with verifiable rewards (RLVR) improves Pass@1 and tool‑use reliability in math and code and increasingly in agents (Sections 2.2 and 6). Where claims are mixed (e.g., whether RL “discovers” skills), the survey presents both counter‑evidence (e.g., “Limit‑of‑RLVR” observing worse large‑K Pass@K; Section 4.1) and techniques that address it (self‑play synthesis; Pass@K‑aligned objectives).
- Ablations and robustness
  - The survey highlights ablations such as:
    - Removing KL vs. adaptive KL: many RLVR pipelines now omit KL for freer exploration, but several works use adaptive or token‑dependent KL to preserve knowledge (Section 3.2.5).
    - Entropy‑control ablations: high‑entropy token emphasis vs. explicit entropy loss vs. covariance clipping strategies to avoid collapse (Section 3.2.5).
    - Sampling ablations: medium‑difficulty filtering and replay significantly stabilize GRPO (Section 3.3.1; DAPO, PRIME).
- Failure cases and conditions
  - Reward hacking appears when model‑based PRMs are used without strong verifiers (Sections 3.1.3 and 3.1.2).
  - RL does not always beat SFT under severe overfitting or abrupt distribution shifts (Section 4.2).
  - Entropy collapse and length sprawl are recurring issues if not explicitly managed (Sections 3.2.5 and 3.3.2).

> “Outcome rewards provide scalable goal alignment with automated verification, while process rewards offer interpretable dense guidance” (Section 4.5). This duality explains both the successes (math/code) and the remaining brittleness (open‑ended writing, subjective judgments).

## 6. Limitations and Trade-offs
- Assumptions and prerequisites
  - Reliable verifiers or reward proxies exist (Verifier’s Law). Without them, generative/rubric‑based rewards are noisy and prone to gaming (Sections 3.1.1–3.1.2).
  - Group‑based training expects multiple rollouts per prompt (GRPO), increasing inference cost (Section 3.2.3).
- Scope gaps
  - Open‑ended subjective tasks remain hard; rubric‑based rewards help but are not as scalable or robust as rule‑based checks (Sections 3.1.2 and 4.5).
  - Credit assignment for very long chains is still expensive; tree search and step stopping help but raise compute (Section 3.1.3).
- Computational constraints
  - RLVR requires repeated sampling with high temperatures, long contexts, and often multiple candidates per prompt; compute and latency are bottlenecks (Sections 3.3 and 5.3).
  - Asynchronous actors/learners and replay improve utilization but create off‑policy drift that must be controlled (Section 3.2.4).
- Algorithmic trade‑offs
  - Removing KL improves exploration but risks knowledge drift; adding KL can over‑constrain progress (Section 3.2.5).
  - Process rewards increase stability but invite reward hacking if PRMs are weak; outcome‑only rewards are scalable but suffer from credit assignment sparsity (Sections 3.1.3 and 4.5).
- Open questions
  - When does RL truly “discover” vs. “sharpen”? The survey provides hypotheses and metrics but no definitive boundary (Section 4.1).
  - How to generalize RL beyond verifiable domains without heavy human oversight (Section 7).

## 7. Implications and Future Directions
- How this changes the landscape
  - RL is becoming a core mechanism for scaling reasoning, not merely alignment. The field is coalescing around RLVR with critic‑free updates (GRPO family), dynamic sampling, and length‑aware training—all supported by standardized verifiers, open corpora, and asynchronous RL infrastructures (Sections 2.2, 3, and 5).
- Follow‑up research avenues (Section 7)
  - Continual RL (Section 7.1): Lifelong, multi‑stage RL that preserves past skills while learning new tasks; needs replay, policy reuse, and reward shaping tailored to LRMs.
  - Memory‑based RL (Section 7.2): Turn task‑specific memories into general experience repositories; learn memory operations via RL to reuse strategies across tasks.
  - Model‑based RL (Section 7.3): Integrate world models (including video‑trained ones) to provide rich state and synthetic rewards for agents in GUI/web/robotics domains.
  - Efficient reasoning (Section 7.4): Learn compute‑allocation policies (adaptive halting, difficulty‑aware budgets) to minimize overthinking and underthinking.
  - Latent‑space reasoning (Section 7.5): Move from token‑space CoT to continuous latent thought optimized with RL; requires new reward estimators for latent trajectories.
  - RL in pre‑training (Section 7.6): RL as a scalable pre‑training objective (e.g., RPT), potentially reducing dependence on next‑token prediction alone.
  - RL for diffusion LLMs (Section 7.7): Solve ELBO variance and guide multi‑step denoising with intermediate rewards; mixed ODE/SDE sampling for exploration–efficiency balance.
  - Scientific discovery (Section 7.8): Use soft verifiers (biological models, simulations) to scale verifiable rewards beyond math/code; couple lab‑in‑the‑loop agents with in‑silico training.
  - Architecture–algorithm co‑design (Section 7.9): Treat routing/sparsity/expert activation as RL actions to optimize both accuracy and hardware efficiency.

- Practical applications (Figure 6; Section 6)
  - Code generation and repository‑level engineering (unit tests and CI as rewards).
  - Agentic search/deep research on the web (browser and tool use with outcome/process rewards).
  - GUI/computer‑use agents with environment‑derived success signals.
  - Multimodal reasoning and generation (vision/video/3D) with verifiable attributes.
  - Robotics with Vision‑Language‑Action models trained via sparse success signals and GRPO/PPO.
  - Medical reasoning with rule‑based (verifiable) and rubric‑based (non‑verifiable) rewards.

In short, this survey provides a mechanism‑centric map that practitioners can follow end‑to‑end: pick a verifiable task (or build a surrogate verifier), choose critic‑free GRPO‑style optimization with dynamic sampling and length control, reuse open datasets/environments, and scale via asynchronous RL infrastructure—while being mindful of reward hacking, entropy collapse, and generalization trade‑offs.
