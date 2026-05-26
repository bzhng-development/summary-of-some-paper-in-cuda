# AGENTIC REINFORCED POLICY OPTIMIZATION

**ArXiv:** [2507.19849](https://arxiv.org/abs/2507.19849)

## 🎯 Pitch

Agentic Reinforced Policy Optimization (ARPO) introduces a novel RL algorithm that dynamically branches rollouts at moments of high uncertainty immediately following tool calls, allowing large language model agents to better explore and internalize step-level tool-use behaviors. By adaptively targeting these critical decision points and attributing advantage credit with fine granularity, ARPO achieves state-of-the-art performance across 13 multi-turn reasoning benchmarks—while using only half the tool-call budget of prior methods. This establishes a new paradigm for efficient, scalable alignment of LLM-based agents with real-world, dynamic, tool-rich environments.

---

## 1. Executive Summary

This paper proposes **Agentic Reinforced Policy Optimization (ARPO)**, a reinforcement learning algorithm tailored for training multi-turn LLM-based agents that interact with external tools. The method is evaluated on 13 benchmarks spanning mathematical reasoning, knowledge-intensive reasoning, and deep search, using backbone models from the Qwen2.5, Qwen3, and Llama3.1 families. ARPO introduces an **entropy-based adaptive rollout mechanism** that dynamically balances global trajectory sampling with step-level partial sampling at high-entropy tool-use steps (branching additional reasoning paths when the model's token uncertainty spikes after receiving tool-call feedback), paired with **advantage attribution estimation** that assigns distinct advantage values to shared versus individual token segments across branched trajectories (enabling the model to internalize stepwise tool-use advantage differences during policy updates). ARPO consistently surpasses trajectory-level RL algorithms like GRPO and DAPO—achieving, for instance, average accuracy improvements of roughly 4 percentage points across 10 reasoning datasets—while using only **half the tool-call budget** required by existing methods, establishing that step-level exploration guided by post-tool-call entropy signals yields substantial efficiency gains only when the model's uncertainty after tool interactions is explicitly monitored and exploited.

## 2. Context and Motivation

### The Core Problem: Trajectory-Level RL Ignores the Multi-Turn Dynamics of Tool-Using Agents

The fundamental gap this paper addresses is that **existing reinforcement learning algorithms for LLM-based agents treat multi-turn tool interactions as if they were single-turn reasoning problems**. Current methods—GRPO, DAPO, REINFORCE++—sample complete tool-use trajectories and provide a single reward signal based on the final output accuracy. This "trajectory-level" approach ignores a crucial reality of agent-environment interaction: the LLM receives diverse, informative feedback from tools at **every intermediate step**, and its internal state changes in response to that feedback.

The paper frames this as a mismatch between algorithm design and task structure. In single-turn reasoning (e.g., solving a math problem in one continuous generation), sampling complete solutions and comparing them via group-relative advantage makes sense—the model's reasoning unfolds linearly without external perturbation. But when an LLM calls a search engine, receives a snippet, calls a Python interpreter, receives execution output, and then continues reasoning, each tool-call boundary introduces a **distributional shift**. The model must integrate externally generated text or structured data into its ongoing chain of thought. How it handles this integration—which tokens it generates immediately after seeing tool results—is where step-level behavioral differences emerge, and trajectory-level RL is blind to them.

The paper quantifies this blindness through a specific diagnostic: token-level entropy after tool calls. Using the vocabulary distribution entropy $H_t = -\sum_{j=1}^V p_{t,j} \log p_{t,j}$ (where $p_t$ is the softmax over the vocabulary at step $t$), the authors measure how certain or uncertain the model is at each generation step. They find a consistent pattern: **entropy spikes sharply in the first 10–50 tokens following every tool-call**, and this spike is larger for unstructured feedback (search engine text) than for structured feedback (Python interpreter output). This is not an artifact of one model or one dataset—it reproduces across both search-based and code-based agents (Figure 2).

The implication is that tool-call feedback introduces genuine uncertainty into the model's decision process. The model doesn't simply "know" what to do with search results or execution output; it explores multiple possible interpretations in its early post-tool tokens. Trajectory-level RL, which samples complete trajectories independently and compares their final rewards, provides **no mechanism to differentially explore these high-uncertainty moments**. All trajectories are sampled from the same initial conditions, and if the model happens to make a suboptimal choice in those first post-tool tokens, that choice propagates through the rest of the trajectory with no opportunity for targeted exploration or correction.

### Why This Matters: The Practical and Theoretical Stakes

The problem has both an efficiency dimension and a capability dimension.

**The efficiency argument** is about tool-call budgets. Every call to a search engine, web browser, or code interpreter costs money and wall-clock time. In the RL training phase, where models generate thousands of trajectories, the total tool-call count directly determines training cost. The paper shows (Figure 7) that ARPO achieves better accuracy than GRPO while using half the tool calls during training. This isn't a small optimization—for organizations training agents at scale, halving the API costs or infrastructure overhead of tool interactions during RL is a substantial practical gain. The mechanism behind this efficiency is that ARPO selectively expands exploration only at high-entropy steps, rather than independently sampling full trajectories from scratch, each of which incurs its own full complement of tool calls.

**The capability argument** is about escaping local optima in tool-use behavior. Trajectory-level RL tends to reinforce whatever tool-use patterns happen to correlate with final-answer correctness in the initial sampling distribution. If the model's default behavior after receiving search results is to superficially skim snippets (a common failure mode in RAG systems), trajectory-level RL has no structured way to discover that deeper engagement with the retrieved content would yield better answers. The paper's entropy-based adaptive rollout mechanism explicitly creates branching at these post-tool-call decision points, generating additional reasoning paths that explore different interpretations of the same tool output. Some of these branches will be worse than the default path, but some will be better—and the advantage attribution estimation ensures that tokens on better branches receive positive reinforcement, gradually shifting the model's policy toward more effective tool-use behaviors.

**The theoretical significance** extends beyond tool use. The paper is essentially arguing that **for any sequential decision-making problem where the agent receives external observations at intermediate steps, treating the trajectory as a monolithic unit for credit assignment is suboptimal**. This connects to long-standing principles in RL: temporal credit assignment—determining which actions contributed to eventual success or failure—is harder when rewards are sparse and delayed. In the agentic RL setting, the only reward signal is typically at the end (was the final answer correct?), but the decisions that matter most (how to interpret tool output, which subsequent tool to call) happen in the middle. ARPO's combination of entropy-guided exploration branching and shared-vs-individual advantage attribution is a specific solution to this credit assignment problem for Transformer-based policies, formalized through the Generalized Policy Gradient Theorem (Section 3.3) which shows that policy optimization can be effectively conducted on macro-action segments rather than requiring single-token granularity.

### Prior Approaches and Their Limitations

**Trajectory-level RL algorithms (GRPO, DAPO, REINFORCE++).** These methods, originally designed for single-turn reasoning enhancement, have been applied to agentic RL in a straightforward way: sample $G$ complete trajectories for each training prompt, compute rewards based on final answer correctness, and update the policy using group-relative advantage estimation (comparing each trajectory's reward to the group mean). GRPO (Shao et al., 2024) simplifies the PPO objective by dropping the value function and using group statistics for advantage normalization. DAPO (Yu et al., 2025) adds dynamic sampling, token-level loss computation, and overlong reward shaping. REINFORCE++ (Hu, 2025) incorporates baselines and TD estimation for variance reduction.

These methods have demonstrated strong performance on math and coding benchmarks (DeepSeek-R1, Kimi k1.5). But when applied to multi-turn tool use, they exhibit a specific failure mode: they **overemphasize the initial reasoning before any tool calls** while **neglecting the step-level decisions made after receiving tool feedback**. The paper's entropy analysis provides the mechanistic explanation—the high-uncertainty moments that most determine trajectory quality happen *after* tool calls, and trajectory-level sampling, which treats each trajectory as an independent draw from the initial distribution, provides no structured way to explore alternatives at those points. The result is that trajectory-level RL plateaus at a level of tool-use proficiency determined by whatever behaviors happened to be present in the initial SFT model, without discovering superior strategies that require coordinated step-level optimization.

The paper's empirical evidence for this limitation is in Table 1: across 10 datasets using three backbone models (Qwen2.5-3B, Llama3.1-8B, Qwen2.5-7B), the three trajectory-level algorithms achieve broadly similar performance, with DAPO—despite being state-of-the-art for single-turn reasoning—actually underperforming on knowledge-intensive tasks. This supports the paper's claim that the optimization dynamics that work for single-turn reasoning don't transfer straightforwardly to multi-turn agent interactions.

**Prompting-based tool use (TIR, ReAct, Search-o1).** An alternative approach to improving tool-use behavior is through prompting strategies rather than RL training. Tool-Integrated Reasoning (TIR) prompting prepends instructions about tool availability and formats, encouraging the model to interleave reasoning with tool calls. ReAct (Yao et al., 2022) structures outputs as alternating "Thought," "Action," and "Observation" steps. Search-o1 (Li et al., 2025d) adds a Reason-in-Documents module that explicitly re-reads retrieved content.

The paper shows (Table 1) that TIR prompting not only fails to improve over direct reasoning for most model-dataset combinations but **actually degrades performance** in several cases. For Qwen2.5-3B, TIR prompting drops MATH500 accuracy from 63.0 to 52.2, GSM8K from 75.0 to 56.6, and MATH from 71.6 to 62.8. The paper attributes this to prompting methods disrupting the model's inherent reasoning capabilities—the tool-use instructions, imposed from outside, conflict with the model's internal chain-of-thought patterns learned during pretraining and instruction tuning.

This failure of prompting-based methods is important context for why RL-based approaches are necessary: you cannot simply tell the model to use tools better; you need to **train it** through reinforcement signals that reshape its generation distribution at the step level. The paper cites several works (Wang et al., 2025b; Sha et al., 2025; Bai et al., 2025) that have attempted to improve tool-use through reward shaping—designing better reward functions that penalize excessive tool calls or reward efficient tool usage—but argues that these optimizations, while helpful, still operate at the trajectory level and miss the step-level behavioral exploration that ARPO enables.

**Workflow-based search agents (Vanilla RAG, WebThinker).** For deep search tasks, where agents must navigate web pages, extract information, and synthesize across multiple sources, a common approach is to design structured workflows: retrieve-then-read (Vanilla RAG), retrieve-then-reason (Search-o1), or hierarchical planning-then-execution (WebThinker). These systems typically use frozen LLMs within hand-designed pipelines, where the tool-use strategy is determined by the system designer rather than learned by the model.

The limitation here is that **hand-designed workflows are brittle**—they cannot adapt to the specific requirements of individual questions. A question requiring comparison of statistical data across multiple countries needs different search and extraction strategies than a question requiring finding a specific historical fact. Workflow-based agents apply the same strategy to all problems, missing opportunities for efficiency (using simpler strategies for easier questions) and capability (using more aggressive exploration for harder questions). The paper's results in Table 2 show that while workflow-based agents achieve nontrivial performance on deep search benchmarks, they are consistently outperformed by RL-trained agents, even with minimal training data (1K samples for the RL phase). For example, on GAIA, the best workflow-based agent (WebThinker with Qwen3-8B) achieves 22.3% average accuracy, while GRPO-trained Qwen3-8B achieves 32.0% and ARPO-trained Qwen3-8B achieves 38.8%.

**Agentic RL with reward shaping (ToolRL, Tool-Star, OTC).** Recent work has specifically targeted the tool-use setting with RL algorithms. ToolRL (Qian et al., 2025a) uses rule-based rewards to teach multi-tool invocation. Tool-Star (Dong et al., 2025), which ARPO builds on, introduces multi-tool collaboration rewards and the cold-start SFT + RL training paradigm. OTC (Wang et al., 2025b) focuses on optimal tool-call efficiency through reward design.

These works push the frontier of what's possible with RL-trained agents, but they share a common limitation: they apply **trajectory-level** RL algorithms to what is fundamentally a **multi-turn** interaction problem. The RL phase treats each complete tool-use trajectory as the unit of optimization, computing rewards at the end and propagating credit uniformly across all tokens. This means that if a trajectory is successful, *all* token-level decisions in that trajectory receive positive reinforcement, including potentially suboptimal tool-use steps that happened to be rescued by later correct reasoning. Conversely, if a trajectory fails, *all* decisions are penalized, even smart tool-use choices that were undone by a later reasoning error. This credit assignment problem is the central challenge that ARPO addresses through its combination of entropy-guided exploration branching and differentiated advantage attribution.

### How This Paper Positions Itself

The paper positions ARPO as an **algorithmic innovation within the agentic RL paradigm**, not a new training paradigm or a new model architecture. The stated goal is "to validate the effectiveness of ARPO at the algorithmic level compared to traditional RL in training LLM agents, rather than merely pursuing performance improvements" (Section 4.3). This is a deliberately focused contribution: given the existing infrastructure of cold-start SFT followed by RL training (established by Tool-Star and others), how should the RL algorithm itself be redesigned to account for the multi-turn, observation-driven nature of tool-use trajectories?

The paper builds directly on three converging lines of work:

1. **Entropy-based RL analysis** (Wang et al., 2025c;d; Cheng et al., 2025; Zheng et al., 2025), which has shown that high-entropy tokens are disproportionately important for reasoning capability acquisition in single-turn settings. ARPO extends this insight from identifying *which tokens matter* to *actively exploring alternatives at those tokens*.

2. **GRPO and its variants** (Shao et al., 2024; Yu et al., 2025), which provide the group-relative advantage estimation framework that ARPO adapts. The paper explicitly builds on the GRPO loss formulation (Equation 6) but modifies how trajectories are sampled (entropy-based adaptive rollout) and how advantages are computed across branches (advantage attribution estimation).

3. **Tool-use agent training** (Dong et al., 2025; Song et al., 2025), which provides the training pipeline (cold-start SFT on tool-use data, followed by RL with verifiable rewards) and the reward design (correctness + format + multi-tool collaboration). ARPO does not change the reward function or the SFT phase; it only modifies the RL sampling and advantage computation.

The paper's key positioning claim is that **the multi-turn nature of tool-use interactions requires step-level optimization that trajectory-level RL cannot provide**. The entropy analysis in Section 2.2 is positioned as empirical evidence for this claim: by showing that uncertainty spikes at tool-call boundaries, it establishes that these are the points where exploration matters most, and by showing that trajectory-level methods don't explore them differentially, it identifies the specific mechanism by which they fall short. The entropy-based adaptive rollout mechanism is then presented as the natural solution: if tool-call boundaries are high-uncertainty points, branch additional exploration there; if other parts of the trajectory are low-uncertainty (the model is confident), save the compute budget by not branching.

A subtle but important aspect of the positioning: the paper does not claim that ARPO replaces trajectory-level RL, but rather that it **extends** it with step-level exploration capabilities. The global rollout of $N$ trajectories still happens—ARPO just adds partial sampling branches at high-entropy steps, using a reserved budget of $M - N$ partial trajectories. This hybrid design reflects the paper's empirical finding (Section 3.1, scaling analysis in Figure 8) that purely partial sampling (initial sampling size $N = 0$) and purely global sampling ($N = M$) both underperform the balanced intermediate setting, where approximately half the budget goes to global trajectories and half to branched exploration.

The paper also positions itself explicitly relative to the theoretical foundations of policy gradient methods. By introducing the Generalized Policy Gradient Theorem (Section 3.3), it argues that ARPO's macro-action segmentation (grouping consecutive tokens into segments for advantage computation) is not just a heuristic trick but a theoretically justified generalization of the standard Policy Gradient Theorem. This theoretical framing serves to distinguish ARPO from ad-hoc modifications to RL algorithms and to connect it to the broader RL literature on hierarchical credit assignment.

## 3. Technical Approach

### 3.1 Reader Orientation

ARPO is a **reinforcement learning algorithm** specifically designed for training LLM-based agents that use external tools across multiple turns. The system solves the problem that existing trajectory-level RL algorithms (like GRPO) treat complete tool-use trajectories as monolithic units for credit assignment, ignoring the fact that tool-call feedback introduces sharp uncertainty spikes at specific intermediate steps where targeted exploration would be most valuable. The solution takes the shape of a **hybrid sampling strategy**: sample some complete trajectories globally (as in standard RL), but reserve part of the generation budget for **partial branching at high-entropy post-tool-call steps**, then compute advantages that distinguish between shared prefix tokens and divergent branch tokens during policy updates.

### 3.2 Big-Picture Architecture (Diagram in Words)

The ARPO system has five major components that operate in sequence during each RL training iteration:

1. **Policy Model (`$\pi_\theta$`)** — The LLM being trained (e.g., Qwen2.5-7B). Generates reasoning text interleaved with tool-call requests and processes tool-returned results to produce final answers.

2. **Tool Environment (`$\mathcal{T}$`)** — External tools (search engine, web browser agent, Python code interpreter) that receive structured call requests from the policy model and return results (search snippets, parsed web page content, code execution output or error messages).

3. **Entropy Monitor** — A non-trainable diagnostic component that computes token-level vocabulary entropy `$H_t$` from the policy model's output logits at each generation step, tracking how uncertainty changes specifically after tool-call boundaries to identify candidate branching locations.

4. **Adaptive Rollout Module** — The core sampling mechanism. Given a total rollout budget of `$M$` trajectories per prompt, it first generates `$N$` complete global trajectories, then monitors entropy after each tool call. When the normalized entropy change `$\Delta H_t$` exceeds a threshold `$\tau$`, it branches `$Z$` additional partial trajectories from that step, consuming the remaining `$M - N$` budget. This produces a mixture of fully independent trajectories and partially shared trajectories with divergent post-tool-call branches.

5. **Advantage Attribution Estimator** — Computes group-relative advantages for policy updates. For tokens shared across multiple branched trajectories (the prefix before branching), it assigns the same advantage value (derived from the average reward of all trajectories sharing that prefix). For tokens on divergent branches, it assigns individual advantages based on each branch's own trajectory reward.

Information flows as follows: a prompt is sampled → the policy model generates `$N$` initial trajectories with tool calls → the environment returns tool results → the entropy monitor computes uncertainty after each tool call → the adaptive rollout module branches at high-entropy steps → additional trajectories are generated → final answers are produced → rewards are computed (correctness + format + multi-tool collaboration) → advantages are estimated separately for shared vs. branch tokens → the policy model is updated via the GRPO clipped objective.

### 3.3 Roadmap for the Deep Dive

- **First**, the entropy monitoring mechanism (Equation 3–4) — because it is the diagnostic that motivates and drives the entire adaptive rollout design, and understanding what `$H_t$` and `$\Delta H_t$` measure is prerequisite to everything else.
- **Second**, the agentic RL training objective (Equation 1) and trajectory decomposition (Equation 2) — because these define the mathematical framework within which ARPO operates and clarify what makes agentic RL structurally different from single-turn RL.
- **Third**, the entropy-based adaptive rollout mechanism (Section 3.1, Equations 4–5) — the core sampling innovation, including the four-step process (initialization, monitoring, branching, termination) and the branching probability formula.
- **Fourth**, the advantage attribution estimation (Section 3.2, Equations 6–7) — how ARPO modifies the GRPO policy update to account for shared vs. individual token segments, including both hard and soft advantage estimation variants.
- **Fifth**, the hierarchical reward design (Equation 8) — the multi-component reward function that provides the optimization signal.
- **Sixth**, the theoretical foundation (Section 3.3, Equation 9) — the Generalized Policy Gradient Theorem that justifies macro-action segmentation for Transformer policies.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This paper proposes a **novel RL training algorithm** whose core idea is that tool-use trajectories should not be sampled and optimized as monolithic units; instead, the model should branch additional exploration specifically at post-tool-call steps where token entropy spikes, then assign differentiated credit to shared reasoning prefixes versus divergent post-branch continuations during policy updates.

---

#### Agentic RL Training Objective

The paper formulates agentic RL training as a constrained optimization problem. The objective is to maximize expected reward while staying close to a reference policy:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x; \mathcal{T})} \left[ r_\phi(x, y) \right] - \beta \, D_{\text{KL}} \left[ \pi_\theta(y | x; \mathcal{T}) \parallel \pi_{\text{ref}}(y | x; \mathcal{T}) \right]$$

where `$\pi_\theta$` is the policy LLM being trained, `$\pi_{\text{ref}}$` is a frozen reference copy of the initial policy (prevents catastrophic forgetting), `$r_\phi$` is the reward function that scores outputs, `$x$` is a prompt sampled from dataset `$\mathcal{D}$`, `$y$` is the generated output possibly interleaved with tool-call feedback, `$\mathcal{T}$` denotes the set of available tools, and `$\beta$` controls the KL penalty strength.

**What it computes:** the expected reward over the policy's own generated outputs, minus a penalty proportional to how much the policy has diverged from the reference. The expectation is taken over prompts from the training distribution and over outputs sampled from the current policy (on-policy). The reward is computed after the full trajectory completes (final answer verified). The KL term ensures the policy does not drift so far from the reference that it loses general language capabilities — a standard regularization in RLHF and RLVR.

**Why this form:** this is the standard RLVR objective adapted for tool-use agents. The key difference from single-turn RLVR is that `$y$` is not a single contiguous generation but an interleaved sequence of reasoning text, tool-call commands, and tool-returned results. The notation `$\pi_\theta(\cdot|x; \mathcal{T})$` indicates that the policy conditions on available tool definitions during generation — the model knows which tools exist and how to format calls to them, but the actual tool execution happens outside the model in the environment.

The paper notes that this is built on rule-based RL algorithms like GRPO and REINFORCE++, designed specifically to optimize LLM-based agents. The ARPO contribution is not in changing this objective but in changing *how trajectories are sampled* to populate the expectation and *how advantages are computed* within the policy update step.

---

#### Trajectory Decomposition for Multi-Turn Tool Use

The rollout sampling process in agentic RL is decomposed into two phases — agentic reasoning (interleaved with tools) followed by answer generation:

$$P_\theta(R, y | x; \mathcal{T}) = \prod_{t=1}^{t_R} \underbrace{P_\theta(R_t | R_{<t}, x; \mathcal{T})}_{\text{Agentic Reasoning}} \cdot \prod_{t=1}^{t_y} \underbrace{P_\theta(y_t | y_{<t}, R, x; \mathcal{T})}_{\text{Answer Generation}}$$

where `$R$` is the reasoning trajectory of length `$t_R$` (including both the model's own reasoning tokens and the tool-returned results), `$y$` is the final answer with length `$t_y$`, and `$R_t$` denotes the `$t$`-th step of the reasoning process.

**What it computes:** the joint probability of a reasoning trajectory and final answer given a prompt and tool set. The first product runs over the reasoning steps — each step `$R_t$` is conditioned on all previous reasoning steps `$R_{<t}$`, the original prompt `$x$`, and the tool definitions `$\mathcal{T}$`. The second product runs over the answer tokens — each answer token is conditioned on all previous answer tokens, the complete reasoning trajectory `$R$`, the prompt, and the tools. This captures the intuition that the model first works through the problem with tool interactions, then produces a final answer based on everything it has gathered.

**Why this form:** this decomposition makes explicit what trajectory-level RL obscures — that tool-call feedback creates structural breakpoints in the generation. The reasoning phase `$R$` is not one continuous autoregressive sequence; it is punctuated by tool calls where the model stops generating, the environment executes the tool, and the model resumes generating conditioned on the new external information. At each such resumption point, the model's internal state changes, and — as the entropy analysis shows — its uncertainty about what to generate next increases sharply. The decomposition motivates why branching exploration at these specific breakpoints (rather than at arbitrary tokens or only at the beginning) is well-motivated: these are the points where the model has just received new information and is deciding how to interpret it, so alternative interpretations are most worth exploring.

---

#### Token Entropy Calculation and Monitoring

The paper computes token-level generation entropy to quantify model uncertainty. The entropy at generation step `$t$` is:

$$H_t = -\sum_{j=1}^{V} p_{t,j} \log p_{t,j}, \quad \text{where} \quad p_t = \pi_\theta(\cdot | R_{<t}, x; \mathcal{T}) = \text{Softmax}\left(\frac{z_t}{\tau}\right)$$

where `$V$` is the vocabulary size, `$p_{t,j}$` is the model's predicted probability for token `$j$` at step `$t$`, `$z_t \in \mathbb{R}^V$` is the pre-softmax logit vector, and `$\tau$` is the decoding temperature.

**What it computes:** the information-theoretic entropy of the model's next-token probability distribution at a specific generation step. When the model is confident (probability mass concentrated on one or a few tokens), entropy is low. When the model is uncertain (probability mass spread across many plausible tokens), entropy is high. The paper explicitly notes: "this entropy reflects the uncertainty in the token generation distribution, rather than the uncertainty of any particular token" — it is a property of the entire distribution `$p_t$`, not a measure of any single token's probability.

**Why this form:** entropy is a principled measure of distributional uncertainty from information theory with a natural interpretation — it is the expected number of bits needed to encode a sample from the distribution. The paper draws on a line of recent work showing that high-entropy tokens are disproportionately important for RL-based reasoning acquisition. The novelty in ARPO is not the entropy calculation itself but how it is used: rather than just identifying which tokens are high-entropy (as in prior work), ARPO uses entropy as a **branching trigger** — it actively generates additional trajectories starting from high-entropy steps to explore alternative continuations.

The pilot experiment in Section 2.2 provides the empirical motivation: the first 10–50 tokens after each tool call consistently show elevated entropy, with search engine feedback producing larger spikes than Python interpreter feedback. Figure 2 visualizes this across both search-based and code-based agents. The paper attributes the entropy spike to "distributional shift between external feedback and the model's internal reasoning" — the tool-returned text comes from a different distribution than the model's own generation, creating uncertainty about how to integrate it.

To operationalize entropy monitoring, ARPO computes the entropy of the first `$k$` tokens at two key moments:

1. **Initial entropy** (`$H_{\text{initial}} \in \mathbb{R}^{1 \times k}$`): computed from the first `$k$` tokens generated by the model at the start of the trajectory, serving as a baseline uncertainty level.

2. **Step-level entropy** (`$H_t \in \mathbb{R}^{1 \times k}$`): computed from the first `$k$` tokens generated *immediately after concatenating tool-call feedback* at tool-call step `$t$`.

The normalized entropy change is then:

$$\Delta H_t = \text{Normalize}(H_t - H_{\text{initial}})$$

where normalization means summing all values of the difference vector and dividing by the vocabulary size `$V$`.

**What it computes:** a scalar measuring how much the model's uncertainty has changed relative to its initial state after receiving tool feedback at step `$t$`. A positive `$\Delta H_t$` indicates the model has become more uncertain (entropy increased); a negative value indicates it has become more certain.

**Why this form:** differencing from the initial entropy controls for the model's baseline uncertainty level — some prompts naturally elicit higher entropy across the entire generation, and we want to detect *changes* specifically attributable to tool interactions, not pre-existing prompt difficulty. Normalizing by vocabulary size makes the metric comparable across models with different vocabulary sizes and across different points in training when the model's calibration may shift. The paper uses `$k$` tokens (rather than a single token) to get a more stable estimate — a single token's entropy can be noisy due to sampling, but averaging over the first `$k$` tokens after a tool call provides a more reliable signal.

---

#### Entropy-Based Adaptive Rollout Mechanism

This is the core sampling innovation of ARPO, described in Section 3.1 as a four-step process. The mechanism extends traditional trajectory-level sampling by dynamically branching additional partial trajectories at high-entropy tool-use steps.

**Step 1: Rollout Initialization.** Given a total rollout budget of `$M$` trajectories for a prompt (a hyperparameter — the paper uses `$M = 16$` for 7B/8B models), the LLM first generates `$N$` complete trajectories via standard trajectory-level sampling (the paper uses `$N = 8$` as the default initial sampling size). The remaining `$M - N$` trajectories' worth of budget is reserved for partial branching. The entropy of the first `$k$` tokens in each trajectory is computed to form the initial entropy baseline `$H_{\text{initial}}$`.

**Step 2: Entropy Variation Monitoring.** As the model generates each of the `$N$` trajectories, it interleaves reasoning with tool calls. After each tool call at step `$t$`, the model concatenates the tool-returned result and generates `$k$` additional tokens. The step-level entropy `$H_t$` is computed from these tokens, and the normalized change `$\Delta H_t$` is calculated via Equation 4.

**Step 3: Entropy-Based Adaptive Branching.** The core decision: should the model branch additional exploration from this tool-call step? The branching probability is:

$$P_t = \alpha + \beta \cdot \Delta H_t, \quad \text{Action}(P_t) = \begin{cases} \text{Branch}(Z), & \text{if } P_t > \tau \\ \text{Continue}, & \text{otherwise} \end{cases}$$

where `$\alpha$` is a base sampling probability (set to 0.5 in experiments), `$\beta$` is a stability entropy weight (set to 0.2), `$\tau$` is the branching threshold (set to 0.5), and `$Z$` is the number of additional partial reasoning paths to branch from the current tool-call step.

**What it computes:** a branching probability that increases with the entropy change `$\Delta H_t$`. When the model becomes more uncertain after a tool call (`$\Delta H_t > 0$`), the probability of branching increases. The base probability `$\alpha = 0.5$` ensures some baseline level of exploration even when entropy doesn't spike; the entropy term `$\beta \cdot \Delta H_t$` adds an adaptive component that increases exploration at the most uncertain steps. The threshold `$\tau = 0.5$` implements a hard decision boundary: if `$P_t > 0.5$`, branch; otherwise, continue the existing trajectory without branching.

**Why this form:** the linear combination of a fixed base rate and an entropy-driven term balances two competing desiderata. Pure entropy-driven branching (`$\alpha = 0$`) would focus all exploration on the highest-entropy steps, potentially missing other important decision points where the model happens to be confident but wrong. Pure random branching (`$\beta = 0$`) would ignore the entropy signal entirely, wasting exploration budget on steps where the model already knows what to do. The additive form with threshold implements a "soft" gating mechanism — entropy influences but does not deterministically control branching decisions. The scaling analysis in Figure 8 (left) validates this design: performance peaks at moderate entropy values (peak at `$\Delta H_t = 0.4$`), declining when entropy dominates too strongly (at `$\Delta H_t = 1.0$`), "suggesting a trade-off in the weight of entropy in sampling."

When branching is triggered, `$\text{Branch}(Z)$` generates `$Z$` additional partial trajectories from the current node. These trajectories share the prefix up to the branching point but diverge in their subsequent reasoning — they explore different interpretations of the same tool output, different subsequent tool-call decisions, or different reasoning paths. The paper does not specify the exact value of `$Z$` in the main text (it appears to be a per-branch parameter that consumes the remaining budget), but the termination condition (Step 4) implies that `$Z$` is dynamically determined to use the available `$M - N$` budget.

**Step 4: Termination.** The branching process iterates as the model continues generating through multiple tool-call steps until one of two conditions is satisfied:

1. The total number of forked paths `$\hat{Z}$` reaches the partial sampling budget `$M - N$`. At this point, branching stops and sampling continues along all existing paths until each produces a final answer.

2. All paths terminate (produce final answers) before reaching `$M - N$` branched trajectories. In this case, the system supplements with `$M - N - \hat{Z}$` additional trajectory-level samples from scratch to ensure the total budget `$M$` is fully utilized.

The paper notes a computational complexity property of this mechanism: "assuming the global expansion size and the number of tokens per trajectory are `$n$`, ARPO reduces the computational complexity of each rollout from the trajectory-level RL's `$\mathcal{O}(n^2)$` to between `$\mathcal{O}(n \log n)$` and `$\mathcal{O}(n^2)$`." This is because branched trajectories reuse the shared prefix computation — the shared tokens only need to be generated and processed once, while the divergent branches add incremental cost proportional to their length. In the worst case (every step branches), complexity approaches `$\mathcal{O}(n^2)$`; in the best case (no branching), it is `$\mathcal{O}(n \log n)$` due to the tree structure of partial expansions.

The scaling analysis (Figure 8, middle and right) provides guidance on the key hyperparameters:

- **Initial sampling size `$N$`**: performance peaks at `$N = 8$` when total rollout `$M = 16$` (a 1:1 global-to-partial ratio). At `$N = 0$` (all partial sampling) and `$N = 16$` (all global sampling), performance degrades significantly. This underscores the importance of balancing the two sampling modes — pure partial sampling lacks the global diversity needed to explore fundamentally different high-level strategies, while pure global sampling misses the step-level exploration that discovers better tool-use behaviors at specific decision points.

- **Global rollout size `$M$`**: performance increases monotonically with `$M$` up to 16 (the maximum tested), "indicating that the ARPO algorithm is scalable and can improve generalization performance with larger sizes."

---

#### Advantage Attribution Estimation

The adaptive rollout mechanism produces a mixture of trajectory types: fully independent global trajectories and partially shared trajectories that diverge at high-entropy tool-call steps. This mixture creates a challenge for advantage estimation: tokens in the shared prefix before a branching point should not be credited or blamed for the specific outcomes of any single branch — they contributed to *all* branches equally. ARPO addresses this through advantage attribution estimation, offered in two variants.

**Hard Advantage Estimation.** In this variant, shared and individual token segments are explicitly treated differently in the advantage computation. Given `$d$` trajectories that share a common prefix but diverge after some branching point, the advantage for tokens on the individual (branch-specific) segments is computed using the standard GRPO group-relative formula:

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

where `$r_i$` is the reward of trajectory `$i$`, `$G$` is the total number of trajectories in the group, and `$\hat{A}_{i,t}$` is the advantage assigned to token `$t$` of trajectory `$i$`. The mean and standard deviation are computed over all `$G$` trajectories in the group.

For tokens in the shared prefix, the advantage is the average across all `$d$` trajectories that contain that shared segment:

$$\hat{A}_{i,t}^{\text{shared}} = \frac{1}{d} \sum_{i=1}^{d} \hat{A}_{i,t}$$

**What it computes:** a two-level advantage assignment where shared-prefix tokens receive a single averaged advantage signal (reflecting that they contributed to all downstream branches), while branch-specific tokens receive individual advantages (reflecting the specific outcome of each branch). This prevents a common pathology: if one branch happens to produce a correct answer while another branch from the same prefix produces an incorrect answer, trajectory-level RL would assign the same (averaged) advantage to the shared prefix tokens, diluting the signal about which prefix decisions were actually helpful. Hard estimation explicitly gives the shared prefix the average — it neither gets full credit for the successful branch nor full blame for the failed one.

**Soft Advantage Estimation (Default).** The paper's preferred variant integrates the shared-vs-individual distinction implicitly through the GRPO objective rather than through explicit advantage manipulation. The GRPO optimization objective is:

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_{i,t} \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

where `$q$` is a prompt, `$\{y_i\}_{i=1}^G$` is a group of `$G$` responses sampled from the old policy, `$|y_i|$` is the number of tokens in response `$i$`, `$\hat{A}_{i,t}$` is the group-relative advantage for token `$t$`, and `$r_{i,t}(\theta)$` is the importance sampling ratio:

$$r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\text{ref}}(y_{i,t} | x, y_{i,<t})}$$

**What it computes:** the standard GRPO clipped surrogate objective. The outer expectation averages over prompts and over groups of responses. The inner sum averages over tokens within each response. For each token, the loss is the minimum of the unclipped importance-weighted advantage and the clipped version — this is the standard PPO-style clipping that prevents the policy from changing too much in a single update. The KL penalty provides additional regularization.

**Why this form enables soft advantage attribution:** the key insight is in how the importance sampling ratio `$r_{i,t}(\theta)$` behaves for shared versus individual tokens. When two trajectories `$y_i$` and `$y_j$` share a common prefix up to token `$t$`, they have identical prefix tokens: `$y_{i,<t} = y_{j,<t}$`. Since `$r_{i,t}(\theta)$` depends on the conditioning context `$y_{i,<t}$`, shared tokens in both trajectories have the same importance weight:

$$r_{i,t}(\theta) = r_{j,t}(\theta) \quad \text{if} \quad y_{i,<t} = y_{j,<t}$$

For individual tokens after branching, the conditioning contexts diverge, so the importance weights differ:

$$r_{i,t}(\theta) \neq r_{j,t}(\theta) \quad \text{if} \quad y_{i,<t} \neq y_{j,<t}$$

**What this means operationally:** during the policy update, shared-prefix tokens across branched trajectories contribute to the loss with identical importance weights but possibly different advantages (since each trajectory may have a different final reward). The GRPO objective averages over tokens within each group, so shared tokens effectively receive an averaged update signal — closely approximating the hard estimation's explicit averaging. Branch-specific tokens receive trajectory-specific updates. The mathematical derivation showing this equivalence is provided in Appendix D.1, where the GRPO objective is decomposed into a weighted combination of shared-prefix and branch-specific components.

The paper reports (Figure 5) that soft advantage estimation achieves "consistently higher rewards with greater stability during ARPO training" compared to hard estimation, and therefore soft estimation is the default. This is attributed to soft estimation avoiding the additional variance introduced by explicitly computing and assigning separate advantage values for shared segments — the implicit alignment through importance weights is smoother and more stable.

---

#### Hierarchical Reward Design

ARPO uses a multi-component reward function adapted from Tool-Star that provides the optimization signal. The overall reward `$R$` for a trajectory is:

$$R = \begin{cases} \max(\text{Acc.} + r_M, \text{Acc.}) & \text{If Format is Good \& Acc.} > 0 \\ 0 & \text{If Format is Good \& Acc.} = 0 \\ -1 & \text{Otherwise} \end{cases}$$

$$r_M = \begin{cases} 0.1 & \text{If } \exists (\texttt{<search>} \ \& \ \texttt{<python>}) \\ 0 & \text{Otherwise} \end{cases}$$

where `$\text{Acc.}$` is the answer correctness score (token-level F1 for QA tasks, exact match or LLM-as-judge for other tasks), "Format is Good" means the model output follows the correct tool invocation syntax, and `$r_M$` is a multi-tool collaboration bonus.

**What it computes:** a three-level reward structure. **Level 1 (penalty):** if the model fails to follow the required output format (missing tool-call syntax, malformed answers), it receives a reward of `$-1$` regardless of correctness — this strongly penalizes format violations that would make the output unusable in a production system. **Level 2 (neutral):** if the format is correct but the answer is wrong (`$\text{Acc.} = 0$`), the reward is 0 — the model is neither penalized nor rewarded for producing a well-formatted wrong answer. **Level 3 (positive):** if the format is correct and the answer is at least partially correct (`$\text{Acc.} > 0$`), the reward is the accuracy score, with a potential bonus of `$0.1$` if the model used both search and Python tools during reasoning. The `$\max$` operation ensures the bonus never reduces the reward — if `$\text{Acc.} + r_M < \text{Acc.}$` (impossible with `$r_M = 0.1$`, but included for robustness), the base accuracy is used.

**Why this form:** the three-tier structure addresses specific failure modes in tool-use training. The `$-1$` format penalty prevents the model from learning degenerate strategies like ignoring tool calls and directly guessing answers — such trajectories are actively penalized. The 0 reward for well-formatted wrong answers is standard in RLVR and prevents the policy from being pushed toward random outputs. The multi-tool bonus `$r_M = 0.1$` is a small positive incentive — the paper follows Tool-Star's design — that encourages the model to explore using both search and code tools when appropriate, rather than relying on only one tool type. The bonus is deliberately small (0.1 relative to accuracy scores typically in 0–1 range) to avoid the model overusing tools just for the bonus — it acts as a tiebreaker when multiple tool combinations would yield similar accuracy, not as a primary optimization target.

---

#### Theoretical Foundation: The Generalized Policy Gradient Theorem

To justify that policy optimization over macro-action segments (rather than single tokens) is theoretically sound, the paper introduces the Generalized Policy Gradient (GPG) Theorem:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left\{ \sum_{T=1}^K \left[ \nabla_\theta \log \pi_\theta(MA_T | MS_T) \, A_T(\tau) \right] \right\}$$

where `$\tau$` is a trajectory, `$K$` is the number of macro-action segments the trajectory is divided into, `$MA_T$` is the `$T$`-th macro action (a contiguous sequence of output tokens `$\langle OT_m, OT_{m+1}, \ldots, OT_{m+n} \rangle$`), `$MS_T$` is the `$T$`-th macro state (all input tokens plus all previous macro actions: `$\langle MS_{T-1}, MA_{T-1} \rangle$`, with `$MS_1 \triangleq \langle IT_1, \ldots, IT_{|\text{input}|} \rangle$`), and `$A_T(\tau)$` is the advantage for the trajectory at macro step `$T$`.

**What it computes:** the policy gradient expressed in terms of macro actions rather than single tokens. The expectation is over trajectories sampled from the current policy. For each trajectory, we sum over macro steps `$T = 1$` to `$K$`. At each macro step, the gradient is the log-probability of the macro action (the sequence of tokens comprising that segment) multiplied by the trajectory advantage. The key property is that `$MA_T$` can be of **arbitrary length** — it can be a single token, a complete reasoning step, or an entire tool-interaction round.

**Why this theorem matters for ARPO:** the standard Policy Gradient Theorem (Sutton et al., 1999) operates on individual actions `$a_t$` (single tokens for Transformers), requiring per-token advantage estimation. The GPG Theorem generalizes this to macro actions of any segmentation. This provides theoretical justification for ARPO's design: when the adaptive rollout mechanism branches at a tool-call step, it is effectively treating the tokens from that branching point onward as a macro action that can be optimized as a unit. The advantage attribution estimation — which differentiates between shared-prefix macro actions and branch-specific macro actions — is a direct implementation of the GPG Theorem applied to the specific segmentation induced by tool-call boundaries.

**Why this form is justified:** the paper proves (Appendix D.2) that for Transformer-based policies, where the next state is deterministically the concatenation of the current state and action (`$s_{t+1} = [s_t, a_t]$`), the environment transition probability `$P(s_{t+1} | s_t, a_t) = 1$` — there is no stochastic environment dynamics beyond the policy's own sampling. This means the trajectory probability factorizes cleanly into the product of macro-action probabilities, and the standard policy gradient derivation goes through unchanged at the macro-action level. The proof shows (Equations 23–36 in Appendix D.2) that:

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau; \theta) \left[ \sum_{T=1}^K \nabla_\theta \log \pi_\theta(MA_T | MS_T) \right] R(\tau)$$

with the key step being the equivalence `$\prod_{t=1}^H \pi_\theta(a_t | s_t) = \prod_{T=1}^K \pi_\theta(MA_T | MS_T)$`, which holds precisely because the Transformer's state transition is deterministic concatenation. This connects ARPO to the broader RL literature on hierarchical credit assignment and options frameworks, while remaining specific to the autoregressive Transformer architecture.

**Practical implication:** the GPG Theorem means ARPO does not need to assign per-token advantages to every token in the shared prefix and every branch — it only needs to compute advantages at the granularity of the macro-action segments defined by the branching structure. This reduces the variance of the advantage estimates (fewer, longer segments mean fewer advantage computations) while preserving the ability to differentially reinforce shared versus branch-specific behaviors. The "macro actions" naturally align with the entropy-based branching points: the shared prefix before a branching point is one macro action, and each divergent continuation is a separate macro action.

## 4. Key Insights and Innovations

### Innovation 1: Tool-Call Boundaries as the Locus of Behavioral Uncertainty — A Diagnostic Reframing

The paper's most intellectually distinctive move is not the algorithmic mechanism itself, but the **diagnostic insight that motivates it**: that tool-call feedback creates measurable, systematic uncertainty spikes in an LLM's token generation distribution, and that these spikes represent the specific decision points where trajectory-level RL is structurally blind. This is not a claim about what the model *should* do—it is a claim about what the model *actually does* at the distributional level, measured through token-level vocabulary entropy.

Prior work on LLM-based agents focused almost exclusively on *output quality*: does the model produce correct answers, use tools appropriately, follow formats? The dominant diagnostic tools were accuracy curves and tool-call counts. Meanwhile, work on single-turn reasoning RL (Wang et al., 2025c;d; Cheng et al., 2025; Zheng et al., 2025) had begun using entropy to identify "critical tokens" that disproportionately drive reasoning capability acquisition—but these analyses treated generation as a continuous sequence, without accounting for the structural breaks imposed by tool interactions. The field had no systematic way to measure or conceptualize what happens to the model's internal uncertainty at the specific moments when external information enters the reasoning stream.

ARPO's entropy visualization (Figure 2) makes this concrete in a way that changes how one thinks about multi-turn agent training. The pattern—entropy spikes sharply in the first 10–50 tokens after every tool call, with search feedback producing larger spikes than Python feedback—establishes that tool interactions are not merely additional context that the model integrates seamlessly into its reasoning. They are **distributional perturbations**: the model's next-token distribution shifts sharply when it encounters externally generated text or structured output, because those inputs come from a fundamentally different distribution than its own autoregressive generations. The model doesn't "know what it thinks" about the tool output immediately; it explores multiple interpretations in those early post-tool tokens.

This reframing has significant consequences beyond this paper. It means that evaluating agent RL methods solely by final-answer accuracy or tool-call efficiency misses the key mechanism that determines training effectiveness: whether the algorithm provides structured exploration at the specific points where the model is most uncertain about how to process external information. A trajectory-level RL algorithm could achieve identical final-answer accuracy as ARPO on a given prompt while leaving entirely different tool-use behaviors latent in the model's distribution—the accuracy might come from memorizing effective patterns from the SFT phase rather than from discovering genuinely better ways to interpret tool output. The entropy diagnostic makes this latent dimension of agent behavior observable and optimizable.

This is a **fundamental diagnostic contribution**, not an incremental performance finding. It doesn't propose a new metric for leaderboards—it proposes a new *lens* for understanding what happens inside multi-turn agent trajectories, analogous to how attention visualization changed how researchers understood Transformer internals. The fact that the entropy pattern reproduces across different tool types (search vs. code) and different tasks suggests it reflects a general property of how LLMs process externally injected information, not a quirk of a specific model or dataset.

The evidence is anchored in Figure 2, which shows frequent high-entropy tokens clustered immediately after tool-call boundaries, and in the observation that "Search feedback introduces more uncertainty than Python feedback"—a finding that aligns with the intuition that unstructured textual feedback (search snippets) creates more interpretive ambiguity than structured deterministic output (Python execution results). The paper explicitly ties this diagnostic to its algorithmic contribution: "These findings highlight a limitation of trajectory-level RL methods, which focus on initial reasoning while overlooking the uncertainty introduced by tool-call feedback."

---

### Innovation 2: Differentiated Credit Assignment for Shared Prefixes vs. Divergent Branches — Importing Hierarchical RL Principles to LLM Training

The paper's second conceptual contribution is to recognize—and formalize within the GRPO framework—that when trajectories share common prefixes but diverge at tool-use steps, the policy update should treat shared and branch-specific tokens differently. This is not a new idea in reinforcement learning broadly (hierarchical RL and options frameworks have grappled with temporal credit assignment for decades), but its application to LLM-based agent training represents a **nontrivial conceptual bridge** between two largely separate research communities.

Prior work on LLM RL—GRPO, DAPO, REINFORCE++, and their variants—treats every trajectory as an independent unit for advantage computation. When GRPO samples G responses for a prompt, it computes advantages by comparing each response's reward to the group mean, then applies that advantage uniformly to all tokens in the response (with possible per-token clipping through the importance sampling ratio, but no structural differentiation between tokens based on trajectory overlap). This makes sense for independent samples—if each trajectory is generated from scratch, every token is in a unique context. But ARPO's adaptive rollout deliberately creates **structured dependency** between trajectories: multiple branches share a common prefix up to a tool-call boundary, then diverge. Applying uniform per-token advantages in this setting would credit or blame the shared prefix tokens for outcomes that were determined entirely by decisions made *after* the branch point—a classic credit assignment pathology.

What makes ARPO's solution conceptually interesting is not the mechanism itself (which is described in Section 3), but the **recognition that the natural structure of multi-turn tool interactions—sequential reasoning punctuated by external observations—creates exactly the kind of trajectory overlap that hierarchical credit assignment was designed to handle.** The paper doesn't import hierarchical RL machinery wholesale (which would be heavy and potentially incompatible with the simplicity that makes GRPO effective for LLMs). Instead, it observes that the GRPO objective, through its importance sampling ratio, already provides a mechanism for differentiating shared from individual tokens: when two trajectories share a prefix, their importance weights are identical for those shared tokens, and the GRPO group averaging effectively pools their advantage signals. The soft advantage estimation variant (Section 3.2) exploits this property implicitly, while the hard variant makes it explicit.

This is a **fundamental conceptual advance** disguised as an implementation detail. The paper is essentially arguing that temporal credit assignment in multi-turn agent RL should operate at the granularity of "macro-actions" (contiguous token segments between tool-call boundaries) rather than at the granularity of individual tokens. The Generalized Policy Gradient Theorem (Section 3.3) provides the theoretical justification: for Transformer-based policies, any segmentation of the output sequence into macro-actions yields a valid policy gradient, and the segmentation can be chosen to align with the natural structure of the problem (tool-call boundaries). This subsumes the standard per-token policy gradient as a special case where each token is its own macro-action.

The significance of this contribution extends beyond tool use. Any LLM-based agent that interleaves reasoning with external observations—whether those observations come from tools, human feedback, database queries, or sensor readings—has the same structural property: the generation is punctuated by observation boundaries where the model's internal state shifts, and credit should be assigned differently to reasoning that happens before vs. after each observation. The GPG Theorem provides a unified framework for designing RL algorithms for all such settings, with ARPO as a specific instantiation for the tool-use case.

Evidence for the effectiveness of differentiated credit assignment comes from the advantage attribution estimation variants (Section 3.2). The paper compares hard and soft estimation and finds that soft estimation (which implicitly aligns shared-token advantages through importance weights) achieves "consistently higher rewards with greater stability" (Figure 5) compared to hard estimation (which explicitly averages advantages for shared segments). This is a non-obvious finding: one might expect explicit averaging to be more principled, but the implicit mechanism actually trains more stably—likely because it avoids introducing additional variance from separately estimated shared-prefix advantages, allowing the GRPO group normalization to smooth the signal naturally.

---

### Innovation 3: Selective Exploration at Uncertainty Hotspots as a Compute-Efficiency Strategy

The third conceptual move is the operationalization of entropy as a **branching trigger** rather than merely a diagnostic. Prior entropy-based RL work (Wang et al., 2025c;d; Cheng et al., 2025) identified high-entropy tokens as disproportionately important for reasoning acquisition and proposed various mechanisms to up-weight them during training (e.g., modifying the loss to emphasize high-entropy tokens, or using entropy to guide KL penalty application). These approaches modify *how the model learns from* existing trajectories but do not change *which trajectories are generated*.

ARPO makes a qualitatively different choice: it uses entropy to actively decide **where to allocate additional generation budget**. When entropy spikes after a tool call, the model doesn't just pay more attention to those tokens during the update—it generates entirely new trajectories starting from that point, exploring alternative continuations that the initial trajectory didn't sample. This transforms entropy from a passive diagnostic ("these tokens are important") into an active exploration policy ("let's see what happens if we try something different here").

What makes this **fundamental rather than incremental** is that it changes the relationship between exploration and exploitation during RL training in a way that is specifically adapted to multi-turn agent interactions. In standard RL, exploration is typically handled through stochasticity in the policy (temperature sampling) or through explicit exploration bonuses. Trajectory-level RL for LLMs inherits this approach: sample multiple trajectories independently with non-zero temperature, and let the advantage estimation sort out which ones were good. The problem is that **independent sampling provides uniform exploration across all decision points**, regardless of whether those points actually need exploration. If the model is already confident about how to start a math problem (low entropy in first few tokens) but uncertain about how to interpret search results (high entropy after tool calls), independent sampling wastes exploration budget on re-sampling the confident prefix while under-exploring the uncertain post-tool decisions.

ARPO's entropy-based branching solves this implicitly: the confident prefix is generated once and shared across all branches; the exploration budget is concentrated at the specific steps where uncertainty is high. The paper quantifies this efficiency gain concretely: ARPO achieves better accuracy than GRPO while using half the tool-call budget during RL training (Figure 7). This is not a small constant-factor improvement—halving the number of tool calls during training directly halves the API costs for search, the computation for browser agents, and the sandbox overhead for code execution. For organizations training agents at scale, this efficiency gain alone could determine whether RL-based agent training is economically viable.

The scaling analysis (Figure 8) provides the mechanism-level validation: performance peaks when the global-to-partial sampling ratio is balanced (N = 8 out of M = 16), and degrades at both extremes (pure global: N = 16; pure partial: N = 0). Pure global sampling (standard trajectory-level RL) underperforms because it wastes budget on redundant prefix generation. Pure partial sampling underperforms because it lacks the diversity of initial strategies that independent global trajectories provide—if all trajectories share the same initial approach to the problem, branching can only explore variations of that single approach, missing fundamentally different high-level strategies that a separate global sample might discover.

This is an **architectural innovation in RL training design** rather than a model architecture innovation. It doesn't change the Transformer, the reward function, or the policy update rule—it changes how the generation budget is allocated during the rollout phase that produces training data. This makes it complementary to other RL improvements: one could combine ARPO's adaptive rollout with DAPO's dynamic sampling or REINFORCE++'s variance reduction techniques. The paper doesn't explore these combinations, but the modularity of the approach is a conceptual strength.

---

### Innovation 4: Negative Result on Prompting-Based Tool Use as Motivation for Step-Level RL

While not the paper's headline contribution, the finding that prompt-based tool-use strategies (TIR prompting) consistently underperform or even degrade performance relative to direct reasoning (Table 1) serves an important **conceptual negative result** that strengthens the case for learned, step-level optimization. The field has invested substantial effort in designing prompting strategies for tool use—ReAct (Yao et al., 2022), Search-o1 (Li et al., 2025d), and various chain-of-thought + tool integration templates—under the implicit assumption that if you tell a capable model *how* to use tools, it will do so effectively. The paper's results challenge this assumption systematically across three model families (Qwen2.5-3B, Llama3.1-8B, Qwen2.5-7B) and ten datasets.

The pattern is striking in its consistency: TIR prompting drops MATH500 accuracy for Qwen2.5-3B from 63.0 to 52.2, GSM8K from 75.0 to 56.6, and MATH from 71.6 to 62.8. For Llama3.1-8B, the drops are similarly substantial on knowledge-intensive tasks. The paper's interpretation—that "relying solely on prompt engineering is insufficient for guiding LLMs toward optimal tool behaviors and may disrupt their inherent reasoning capabilities"—points to a deeper issue: tool-use behavior is not simply a matter of the model knowing *that* it should use tools or *which* tools are available. It requires the model to learn *when* to call which tool, *how* to interpret the returned results, and *how* to recover from tool-use failures—behaviors that are difficult to specify declaratively in a prompt but can be learned through step-level reinforcement signals.

This is a **significant conceptual reframing** because it shifts the problem from "designing better prompts" to "designing RL algorithms that can discover effective tool-use behaviors through interaction." It establishes that the gap between prompted and trained tool use is not small—it is large enough that prompting cannot serve as a viable alternative to RL training for challenging multi-turn agent tasks. This justifies the entire research direction of agentic RL, including ARPO, as addressing a fundamental capability gap rather than providing marginal improvements over simpler methods.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on 13 benchmarks across three domains: (1) Mathematical Reasoning: AIME2024, AIME2025, MATH500, MATH, and GSM8K; (2) Knowledge-Intensive Reasoning: WebWalker, HotpotQA, 2WikiMultihopQA, MuSiQue, and Bamboogle; (3) Deep Search: GAIA, Humanity's Last Exam (HLE), WebWalker, and xbench-DeepSearch. Training data comes from Tool-Star's open-source dataset of 54K samples (augmented with 0.8K STILL samples for mathematical reasoning), with 10K RL training samples used for deep reasoning tasks and 1K mixed samples from SimpleDeepSearcher and WebSailor for deep search tasks. The test set splits follow Tool-Star for mathematical and knowledge reasoning benchmarks and WebThinker/HIRA for deep search benchmarks.

- **Base model(s).** Experiments use three model families: Qwen2.5 at 3B and 7B scales, Llama3.1 at 8B, and Qwen3 at 8B and 14B. The Qwen2.5 and Llama3.1 models are evaluated on mathematical and knowledge-intensive reasoning; Qwen3 models are evaluated on deep search tasks "given the superior mathematical performance of the Qwen3 series." The choice spans both dense architectures at multiple scales, providing evidence that ARPO's benefits generalize across model families and parameter counts rather than being specific to one architecture.

- **Metrics.** For mathematical reasoning and deep search tasks, accuracy is measured using Qwen2.5-72B-Instruct under LLM-as-Judge with pass@1 evaluation (temperature 0.6, top-p 0.95). For knowledge-intensive reasoning QA tasks (HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle), the metric is token-level F1 score. All tasks extract answers from model outputs enclosed in `\box{}`. For deep search tasks, additional analysis includes Pass@3 and Pass@5 to capture the model's full potential under repeated sampling. During RL training, correctness signals for deep reasoning tasks use token-level F1 scores.

- **Baselines.** The paper compares ARPO against three categories: (1) **Direct Reasoning**: Instruct versions of Qwen2.5, Llama3.1, and Qwen3 without tool use; also references strong reasoning models including QwQ, DeepSeek-R1, GPT-4o, and o1-preview. (2) **Trajectory-level RL Algorithms**: GRPO, DAPO, and REINFORCE++ applied to the same tool-use training pipeline. (3) **LLM-based Search Agents**: Vanilla RAG, Search-o1, WebThinker, and ReAct—workflow-based systems that use frozen LLMs with hand-designed tool-use strategies. Additionally, Tool-Integrated Reasoning (TIR) prompting serves as a prompting-only baseline that instructs the model to use tools without RL training.

- **Generation budget / compute accounting.** The core unit of comparison is the number of trajectories generated per prompt during RL training: global rollout size M (typically 16 for 7B/8B models), with initial sampling size N (typically 8). For tool-call efficiency analysis, the paper measures total tool calls during RL training, showing ARPO achieves better accuracy with half the calls of GRPO (Figure 7). For deep search evaluation, generation budget is measured in number of RL training samples (1K for deep search tasks vs. 10K for deep reasoning tasks). All RL methods are compared at the same total training batch sizes and PPO mini-batch sizes.

- **Cross-validation / statistical protocol.** The paper employs a cold-start SFT followed by RL paradigm to "mitigate reward collapse during the initial RL training phases." The SFT phase uses LLaMAFactory with Tool-Star's 54K training samples. The RL phase uses the VERL framework, with training across 2 epochs for deep reasoning tasks (on 8 NVIDIA H800 GPUs) and 5 epochs for deep search tasks (on 8 GPUs for 8B, 16 GPUs for 14B). All tool invocation results are excluded from loss calculation "to prevent bias towards tool outputs"—only tokens involved in text reasoning and tool requests contribute to the loss. Scaling experiments vary one hyperparameter at a time while keeping others constant at their default values.

---

### Main Quantitative Results

#### Mathematical and Knowledge-Intensive Reasoning (Table 1)

The headline finding across 10 datasets and three model families is that **ARPO consistently outperforms all three trajectory-level RL algorithms** in a fair comparison with identical training data, SFT initialization, and compute budgets.

**Qwen2.5-3B-Instruct results.** ARPO achieves the highest or tied-highest accuracy on 9 of 10 datasets. Specific numbers: AIME24 23.3% (vs. 20.0% for GRPO/DAPO, 16.7% for REINFORCE++), AIME25 20.0% (vs. 13.3–16.7% for baselines), WebWalker 24.5% (vs. 19.5–21.0%), HotpotQA 58.5% (vs. 54.8–56.5%), 2WikiMultihopQA 67.4% (vs. 62.3–64.5%), Bamboogle 66.8% (vs. 64.8–65.7%). The average across all 10 datasets is 52.8% for ARPO versus 50.6% (DAPO), 50.4% (GRPO), and 49.7% (REINFORCE++). On MuSiQue, ARPO (28.7%) is slightly below DAPO (30.0%) and REINFORCE++ (27.9%), showing domain-specific variation.

**Llama3.1-8B-Instruct results.** ARPO achieves the highest accuracy on 8 of 10 datasets with a 55.3% average versus 51.1% for both GRPO and REINFORCE++, and 50.4% for DAPO. Notable improvements: HotpotQA 65.4% (vs. 57.8% GRPO), 2WikiMultihopQA 75.5% (vs. 71.8% GRPO), MuSiQue 34.8% (vs. 31.0% GRPO), Bamboogle 73.8% (vs. 68.2% GRPO). On AIME25, ARPO ties GRPO at 16.7% (matching DAPO's 13.3% and REINFORCE++'s 16.7%). The Llama3.1 results demonstrate ARPO's generalization to a different model architecture.

**Qwen2.5-7B-Instruct results.** ARPO achieves 58.3% average versus 56.5% (GRPO), 54.9% (REINFORCE++), and 54.8% (DAPO). On AIME24, ARPO reaches 30.0% (vs. 23.3–26.7% for baselines); on AIME25, 30.0% (vs. 23.3–26.7%). However, on MATH500, ARPO (78.8%) is slightly below GRPO and REINFORCE++ (both 78.0%) and DAPO (80.4%), demonstrating that ARPO's advantage is not uniform across all individual benchmarks. On 2WikiMultihopQA (76.1%), ARPO ties GRPO.

**Key pattern across model scales.** The gains from ARPO over trajectory-level RL are remarkably consistent: approximately 4 percentage points average improvement across all three model families. The improvement is proportionally larger on smaller models (Qwen2.5-3B: 52.8% vs. 50.4%, a 2.4 percentage point gain) than on larger models (Qwen2.5-7B: 58.3% vs. 56.5%, a 1.8 percentage point gain), suggesting ARPO's step-level exploration is particularly valuable when the base model's tool-use behaviors are less refined.

**The failure of prompting-based methods.** TIR prompting consistently underperforms direct reasoning and substantially underperforms RL-trained models. For Qwen2.5-3B, TIR drops accuracy from 63.0% to 52.2% on MATH500, from 75.0% to 56.6% on GSM8K, and from 71.6% to 62.8% on MATH. The paper interprets this as evidence that "relying solely on prompt engineering is insufficient for guiding LLMs toward optimal tool behaviors and may disrupt their inherent reasoning capabilities." This finding justifies the entire RL training paradigm: tool-use must be learned through reinforcement signals, not merely instructed.

**Trajectory-level RL comparison.** The three trajectory-level algorithms (GRPO, REINFORCE++, DAPO) perform surprisingly similarly across tasks. DAPO, despite being state-of-the-art for single-turn reasoning, "underperforms in multi-turn tool-call interaction, especially in knowledge-intensive scenarios." This supports the paper's central claim that trajectory-level algorithms, regardless of their specific innovations (dynamic sampling, token-level loss, overlong reward shaping), share a fundamental limitation: they treat multi-turn tool interactions as monolithic trajectories and fail to differentially explore post-tool-call decision points.

---

#### Deep Search Tasks (Table 2)

The headline finding is that ARPO, trained with only 1K RL samples, achieves performance competitive with or exceeding much larger proprietary models and substantially outperforming workflow-based search agents.

**Qwen3-8B results.** ARPO achieves 38.8% average on GAIA across three difficulty levels (53.9% Lv.1, 32.7% Lv.2, 16.7% Lv.3), compared to 32.0% for GRPO. On WebWalkerQA, ARPO reaches 30.5% average (26.7% easy, 33.3% medium, 29.6% hard) versus 29.0% for GRPO. On HLE, ARPO achieves 8.8% versus 7.8% for GRPO. On xBench, ARPO reaches 25.0% versus 20.0% for GRPO. The 6 percentage point improvement on GAIA (38.8% vs. 32.0%) and 5 point improvement on xBench (25.0% vs. 20.0%) are particularly notable given the minimal training data.

**Comparison against larger models.** ARPO with Qwen3-8B (38.8% GAIA) outperforms GPT-4o (17.5%), DeepSeek-R1-32B (14.2%), Qwen3-32B-thinking (14.9%), and QwQ-32B (18.9%). On HLE, ARPO's 8.8% exceeds DeepSeek-R1-32B (6.4%), QwQ-32B (9.6% but with lower GAIA performance), and GPT-4o (2.6%). Even DeepSeek-R1-671B achieves only 25.2% on GAIA and 8.6% on HLE—ARPO with a 14B model (see below) substantially exceeds both.

**Qwen3-14B results.** Scaling to 14B parameters: ARPO achieves 43.7% GAIA average (56.4% Lv.1, 40.4% Lv.2, 16.7% Lv.3) versus 36.9% for GRPO—a 6.8 percentage point gain. On WebWalkerQA: 36.0% (vs. 30.0% GRPO). On HLE: 10.0% (vs. 8.6% GRPO). On xBench: 32.0% (vs. 27.0% GRPO). These results demonstrate ARPO scales effectively with model size.

**Comparison against workflow-based agents.** ARPO substantially outperforms all single-enhanced methods: Vanilla RAG (20.4% GAIA for 8B), Search-o1 (21.4%), WebThinker (22.3%), and ReAct (23.3%). At 14B, ARPO's 43.7% GAIA more than doubles the best workflow agent (WebThinker at 33.0%).

**The importance of step-level exploration in deep search.** The 6–7 percentage point gap between ARPO and GRPO on GAIA (across both 8B and 14B) is attributed to "ARPO's algorithmic design, which balances global and step-level sampling. This balance promotes diverse behavior exploration by LLMs during high-entropy tool-use steps, crucial for deep search scenarios involving frequent tool invocation." Deep search tasks involve multiple rounds of search, page browsing, and information synthesis—each round introduces tool-call feedback that creates exploration opportunities ARPO can exploit.

---

#### Sampling at Scale Analysis (Figure 6)

Beyond Pass@1 evaluation, the paper analyzes Pass@3 and Pass@5 for deep search tasks to "capture the model's potential for tool usage" that single-sample evaluation might miss.

**Qwen3-8B with ARPO:** GAIA Pass@1 is 38.8%, improving to 42.8% at Pass@3 and 47.2% at Pass@5. HLE Pass@1 is 8.8%, improving to 14.0% at Pass@3 and 18.4% at Pass@5. WebWalkerQA Pass@1 is 30.5%, improving to 42.8% at Pass@3 and 52.0% at Pass@5. xBench-DeepSearch Pass@1 is 25.0%, improving to 38.0% at Pass@3 and 44.0% at Pass@5.

**Qwen3-14B with ARPO:** The scaling is more dramatic. GAIA reaches 61.2% at Pass@5 (up from 43.7% Pass@1). HLE reaches 24.0% at Pass@5 (up from 10.0% Pass@1). xBench-DeepSearch reaches 59.0% at Pass@5 (up from 32.0% Pass@1).

The paper attributes this consistent Pass@K improvement to "ARPO's ability to explore fine-grained tool-use behaviors more efficiently, thereby expanding the sampling space and achieving both inference efficiency and sampling diversity." The implication is that ARPO-trained models have a broader distribution of effective tool-use strategies—different samples explore genuinely different approaches rather than varying superficial aspects of the same approach.

---

#### Tool-Call Efficiency Analysis (Figure 7)

A central practical claim: ARPO achieves better accuracy than GRPO while using approximately half the tool calls during RL training.

For Qwen2.5-7B, the comparison tracks both overall accuracy and total tool calls throughout the RL training phase. ARPO reaches approximately 55% overall accuracy using roughly 50K total tool calls, while GRPO requires approximately 100K tool calls to reach approximately 53% accuracy. ARPO's accuracy curve is consistently above GRPO's at every point on the tool-call axis, and ARPO plateaus at a higher final accuracy.

This efficiency is attributed to "ARPO's unique entropy-based adaptive rollout mechanism, which selectively explores branches only during high-entropy tool-call steps. This approach significantly expands the exploration space for tool behavior while greatly reducing the number of tool calls." The mechanism-level explanation: in GRPO, every independently sampled trajectory makes its own complete set of tool calls from scratch, so M trajectories incur tool-call cost proportional to M times the average number of calls per trajectory. In ARPO, the N global trajectories incur full tool-call costs, but the M - N branched trajectories share the tool calls in their prefix and only incur additional tool calls for their divergent portions, reducing the total.

---

#### Browser Agent Ablation (Table 3)

To assess how external tool capability interacts with ARPO training, the paper ablates the browser agent used for deep search across three settings: (1) no browser (snippet-only), (2) same-scale browser (Qwen3-8B browser for Qwen3-8B reasoning model), and (3) larger-scale browser (QWQ-32B browser).

**Qwen3-8B results.** Snippet-only: 33.0% GAIA, 7.5% HLE, 29.0% WebWalkerQA, 23.2% average. Same-scale browser: 38.8%/8.8%/30.5%/26.0%. Larger browser: 38.8%/8.2%/33.0%/26.6%. The jump from snippet-only to any browser is substantial (23.2% → 26.0% average). The larger browser adds modest gains on WebWalkerQA (30.5% → 33.0%).

**Qwen3-14B results.** Snippet-only: 35.0%/8.4%/31.0%/24.8%. Same-scale browser: 43.7%/10.0%/36.0%/29.9%. Larger browser: 47.6%/32.3%/38.4%/39.4%. At 14B, the larger browser provides a dramatic improvement on HLE (10.0% → 32.3%) and a substantial average gain (29.9% → 39.4%).

The paper concludes: "the capability of the external browser agent is highly correlated with the accuracy of the Deepsearch task and shows a clear upward trend as its scale increases." This is an important finding because it establishes that ARPO's effectiveness is partly gated by the quality of the tools it interacts with—better browsers extract more useful information, and ARPO's step-level exploration can better leverage that information.

---

### Ablation Studies and Robustness Checks

**Hard vs. Soft Advantage Estimation (Figure 5).** Soft advantage estimation achieves consistently higher rewards with greater stability during ARPO training compared to hard estimation. The soft setting's reward curve is both higher (by approximately 5–10% in mean reward) and less noisy (tighter confidence band) throughout training. This validates the paper's decision to default to soft estimation and provides evidence that the implicit differentiation of shared vs. individual tokens through importance sampling ratios (Equation 7) is more effective than explicit separate advantage computation.

**Entropy value for branching (Figure 8, left).** Using Qwen2.5-7B with ARPO, model performance increases with rising entropy values and peaks at ΔHt = 0.4, then declines at ΔHt = 1.0. The paper interprets this as evidence that "integrating a moderate amount of entropy as a clue for partial sampling substantially enhances the model's ability to explore rare tool-use behaviors" but that "over-reliance on entropy may reduce sampling diversity." This non-monotonic relationship validates the design choice to include both a base sampling probability α and an entropy-driven term β · ΔHt rather than pure entropy-gating.

**Initial sampling size N (Figure 8, middle).** With global rollout M = 16, performance peaks at N = 8 (1:1 global-to-partial ratio). At N = 0 (pure partial sampling), performance drops significantly. At N = 16 (pure global sampling—equivalent to trajectory-level RL), performance also drops. The paper's interpretation: "increasing the size to 16 results in a great performance decline. This is because it leads to complete global sampling, which disrupts the dynamic sampling balance." This ablation directly demonstrates that the hybrid sampling strategy—not just any partial sampling—is responsible for ARPO's gains.

**Global rollout size M (Figure 8, right).** Performance increases monotonically with M from approximately 2 to 16, with no sign of saturation at the tested maximum. This suggests ARPO is scalable and would continue improving with larger rollout budgets, though the paper does not test beyond M = 16.

**Browser agent capability (Table 3).** As discussed above, browser quality substantially impacts deep search performance, particularly for HLE. This ablation establishes that ARPO's performance is sensitive to the quality of external tools, and that tool capability improvements compound with RL training improvements.

**Model backbone generalization.** ARPO's benefits are demonstrated across Qwen2.5-3B, Qwen2.5-7B, Llama3.1-8B, Qwen3-8B, and Qwen3-14B—five model configurations spanning two architectures and 3B to 14B parameters. The consistency of improvement (approximately 4 percentage points average on reasoning tasks, 6–7 on deep search) provides evidence that the method is not model-specific.

**Training data quantity generalization.** For deep reasoning tasks, ARPO uses 10K RL training samples; for deep search, only 1K. The method works effectively in both regimes, suggesting robustness to training data scale. The 1K-sample deep search results are particularly striking because the model achieves strong performance with minimal RL data.

---

### Critical Assessment

**Claim: "ARPO consistently surpasses traditional sample-level RL algorithms in agentic training."** This is well-supported by Tables 1 and 2, which show ARPO outperforming GRPO, DAPO, and REINFORCE++ across 13 benchmarks and 5 model configurations. The magnitude of improvement is consistent but modest—approximately 4 percentage points average on reasoning tasks, 6–7 on deep search. The claim does not overreach: the paper acknowledges that DAPO sometimes outperforms ARPO on individual benchmarks (e.g., MuSiQue for Qwen2.5-3B, MATH500 for Qwen2.5-7B), demonstrating appropriate nuance. However, the comparison against DAPO and REINFORCE++ uses their standard configurations without adaptation to multi-turn settings—it is possible that hyperparameter tuning specific to multi-turn tool use could narrow or close the gap, and the paper does not explore this. Additionally, all three trajectory-level baselines use the same GRPO family of algorithms; comparisons against fundamentally different RL paradigms (e.g., value-based methods) are absent.

**Claim: "ARPO achieves improved performance using only half of the tool-use budget required by existing methods."** This is supported by Figure 7 for Qwen2.5-7B, showing ARPO reaching ~55% accuracy at ~50K tool calls while GRPO requires ~100K calls for ~53%. However, the claim is demonstrated on only one model (Qwen2.5-7B) and one setting (deep reasoning tasks). The paper does not report tool-call efficiency for deep search tasks or for other model scales. This is a significant gap: the half-tool-call-budget claim is one of the paper's headline efficiency results, but it rests on a single data point. The claim would be substantially strengthened by showing similar efficiency ratios across model families and task types.

**Claim: "ARPO enables LLMs to internalize advantage differences in stepwise tool-use interactions."** This claim is about mechanism—ARPO's advantage attribution estimation should cause the model to learn differentiated behaviors at shared vs. individual token segments. The paper provides theoretical justification (GPG Theorem, Appendix D.1 derivation) and an empirical comparison of hard vs. soft estimation (Figure 5), but does not provide direct behavioral evidence that the model actually learns different policies for shared-prefix vs. branch-specific tokens. An experiment comparing the token distributions of shared vs. individual segments in ARPO-trained models versus GRPO-trained models would directly test this claim, but is absent. The claim is theoretically grounded and the performance improvements are consistent with the mechanism, but the causal link is not experimentally isolated.

**Weakness: Single RL paradigm.** All experiments use GRPO-based policy updates. ARPO modifies the sampling and advantage estimation components but retains the GRPO clipped surrogate objective. While this makes comparisons clean, it leaves open whether the entropy-based adaptive rollout mechanism would benefit other RL algorithms (PPO, REINFORCE with different baselines, etc.). The theoretical foundation (GPG Theorem) suggests broad applicability, but this is not tested.

**Weakness: Limited hyperparameter exploration.** The scaling analysis (Figure 8) explores three parameters (entropy value, initial sampling size, global rollout size) for one model (Qwen2.5-7B). Other potentially important parameters—the branching threshold τ, the base sampling probability α, the entropy stability weight β, the number of tokens k used for entropy estimation—are fixed at their default values across all experiments. The sensitivity of results to these choices is unknown. The branching probability formula (Equation 5) has four parameters (α, β, τ, Z) that could interact in non-obvious ways.

**Weakness: No analysis of which tool-call steps actually get branched.** The entropy-based mechanism should branch more at high-uncertainty steps, but the paper provides no statistics on the empirical branching distribution: what fraction of tool-call steps trigger branching, whether branching frequency correlates with eventual trajectory success, whether different tools (search vs. Python) trigger different branching rates. Without this analysis, it is unclear whether the mechanism is working as intended (branching at genuinely high-entropy steps) or whether the performance gains come from some other property of the hybrid sampling strategy (e.g., simply having more total trajectories through prefix sharing).

**Weakness: No comparison against simple partial-sampling baselines.** ARPO's adaptive rollout differs from trajectory-level RL in two ways: (1) it does partial sampling at all (shared prefixes), and (2) it chooses where to branch based on entropy. The paper ablates the global-to-partial ratio (Figure 8, middle) and entropy weight (Figure 8, left), but does not compare against a fixed-ratio partial sampling strategy without entropy gating (e.g., always branch at every tool-call step, or branch at random tool-call steps). Such a comparison would isolate whether entropy-guided branching specifically improves over uniform branching, or whether the gains come primarily from having any partial sampling at all.

**Weakness: Test sets lack statistical rigor for the strongest claims.** The deep search benchmarks have relatively small test sets (GAIA: 466 questions; HLE: unreported but challenging). Pass@1 differences of 6–7 percentage points on GAIA between ARPO and GRPO are meaningful, but without confidence intervals, it is unclear whether these differences are statistically significant given the sample size. The paper reports no standard deviations, confidence intervals, or significance tests for any result.

**Missing experiment: Does ARPO help more on harder problems?** The paper's motivating insight is that tool-call boundaries are high-uncertainty points, and ARPO explores alternatives at those points. If this mechanism is genuinely responsible for the gains, one would expect ARPO's advantage over GRPO to be larger on harder problems (where tool-use decisions matter more) than on easier problems (where the model's default tool-use behavior is already adequate). The paper breaks out GAIA by difficulty level (Lv.1/2/3) in Table 2, but does not systematically analyze whether ARPO's improvement correlates with problem difficulty across the other benchmarks. This is a missed opportunity to connect the mechanism to the outcomes.

**Missing experiment: Direct measurement of behavioral differences.** The paper claims ARPO enables exploration of "diverse tool-integrated reasoning behaviors" but provides no behavioral analysis: do ARPO-trained models use different tool-call strategies, call tools in different orders, spend more time reading tool output, or recover differently from tool errors compared to GRPO-trained models? Qualitative examples (Appendix F) show individual trajectories but do not provide comparative statistics. Behavioral analysis would substantially strengthen the paper's mechanistic claims.

**Conditional validity of the efficiency claim.** The "half the tool-use budget" claim in Figure 7 likely depends on the specific global-to-partial sampling ratio. If the ratio is too skewed toward partial (N too small), the diversity of global strategies suffers; if too skewed toward global (N too large), the tool-call savings from prefix sharing diminish. The paper demonstrates the optimal ratio for Qwen2.5-7B on reasoning tasks (N = 8, M = 16), but this ratio may not generalize to other models, tasks, or rollout sizes. The efficiency claim should be understood as "up to 2x improvement under optimal configuration" rather than a universal property of ARPO.

**Overall assessment.** The experiments provide strong evidence that ARPO outperforms trajectory-level RL baselines across a wide range of benchmarks and model scales. The consistency of the improvement is the paper's strongest empirical contribution. However, the experiments primarily demonstrate *that* ARPO works better, not *why*—the mechanism-level evidence is limited to the entropy visualization (Figure 2) and the hyperparameter scaling analysis (Figure 8), both of which are suggestive but not dispositive. The efficiency claim is demonstrated on only one configuration and would benefit from broader validation. The paper's central mechanistic claims—that entropy-guided branching at tool-call boundaries and differentiated credit assignment drive the improvements—remain plausible interpretations of the data but are not experimentally isolated from alternative explanations (e.g., that any form of partial sampling with shared prefixes would yield similar gains).

## 6. Limitations and Trade-offs

### 6.1 The "Half the Tool-Call Budget" Efficiency Claim Rests on a Single Configuration

The paper prominently claims that ARPO achieves improved performance "using only half of the tool-use budget required by existing methods" (Section 1, Section 4.6, Figure 7). This claim is empirically demonstrated for exactly one experimental setting: Qwen2.5-7B on deep reasoning tasks, comparing ARPO against GRPO. The paper states:

> "To assess the tool usage efficiency of ARPO during training, we compare it with GRPO on Qwen2.5-7B. As shown in Figure 7, ARPO achieves superior overall accuracy compared to GRPO while using only half the number of tool calls."

**Consequence.** The efficiency claim—which is one of the paper's three headline contributions and appears in the abstract—cannot be assumed to generalize. The tool-call savings depend directly on the global-to-partial sampling ratio (how many trajectories share prefixes vs. are generated independently), the average number of tool calls per trajectory, and the branching rate at high-entropy steps. These factors vary with model scale (larger models may make different numbers of tool calls), task type (deep search involves many more tool interactions per trajectory than mathematical reasoning), and rollout size (M). For deep search tasks, where each trajectory can involve 5–10 tool calls and the RL phase uses only 1K training samples, the efficiency ratio may differ substantially from the 7B reasoning-task result. A practitioner cannot estimate training costs from Figure 7 without replicating the measurement for their specific model-task-scale combination.

**Evidence in the paper.** Figure 7 is the sole efficiency comparison. Neither Table 1, Table 2, nor any ablation reports tool-call counts for other model scales, task types, or trajectory-level RL baselines beyond GRPO on Qwen2.5-7B. The scaling analysis (Figure 8) varies M and N but does not report tool-call counts at each point, so the relationship between sampling ratio and tool-call efficiency cannot be inferred from the paper's data.

**Mitigation status.** Not addressed. The paper does not acknowledge the single-configuration nature of the efficiency evidence, nor does it discuss how tool-call savings might vary across settings. No future work on efficiency characterization is suggested.

---

### 6.2 Entropy Monitoring for Branching Requires Generating and Scoring Extra Tokens for Every Tool Call

ARPO's adaptive rollout mechanism requires computing token-level entropy at two points for every trajectory: once at initialization (the first k tokens) and once after every tool call (the first k tokens following each tool-result concatenation). The paper briefly acknowledges the computational cost in a footnote:

> "Neglecting the minor overhead from token-level entropy calculations"

This is presented as negligible (Section 3.1). However, the overhead has two components that scale with the number of tool calls: (1) **generating k additional tokens after every tool call** specifically for entropy measurement—these tokens are produced at temperature-dependent logits and must be processed through the full forward pass, even though they are not used as part of the trajectory content; (2) **computing the full vocabulary entropy** H_t = -∑ p_j log p_j over V entries, which for models with vocabularies of 100K+ tokens requires a non-trivial amount of logarithmic and multiplication operations per generated token.

**Consequence.** In tasks with frequent tool interactions—particularly deep search, where a single trajectory may involve 5–10 search calls plus browser page fetches—the overhead of generating k extra tokens after every tool call and computing their full entropy distribution could become significant relative to the trajectory generation cost. The paper uses k as a hyperparameter (the number of tokens monitored for entropy) but never specifies its value, making it impossible to estimate the overhead. If k is set to 10–50 (the range where the paper observes entropy spikes in Figure 2), the overhead per tool call is an additional forward pass generating those tokens plus the vocabulary-level entropy computation. For a 14B-parameter model with a vocabulary of ~150K tokens, this adds measurable per-step cost that the paper does not account for in its efficiency comparisons or complexity analysis.

**Evidence in the paper.** The footnote in Section 3.1 explicitly dismisses the overhead as "minor." No ablation varies k, no efficiency measurement includes entropy computation cost, and the complexity analysis (O(n log n) to O(n²)) explicitly excludes this cost. The paper never reports the value of k used in experiments.

**Mitigation status.** Not addressed. The paper treats entropy computation as free. A simple mitigation—using the existing generated tokens rather than generating k *additional* tokens only for entropy measurement—is not discussed. If the model simply computes entropy on the first k tokens it would generate naturally after each tool call (rather than generating extra tokens), the overhead reduces to the logit-processing cost alone, but the paper does not clarify whether this is the implemented approach or whether extra tokens are generated.

---

### 6.3 The Entropy-Based Branching Mechanism Is Validated Only Through Correlation, Not Causal Isolation

The paper's central mechanistic claim is that branching at high-entropy post-tool-call steps—and *only* at those steps—drives ARPO's performance gains. The empirical evidence for this claim is the entropy visualization (Figure 2, showing entropy spikes after tool calls) and the hyperparameter scaling analysis (Figure 8, showing that moderate entropy weighting improves performance). However, **the paper never compares entropy-guided branching against alternative branching strategies** that would isolate whether entropy specifically—as opposed to any form of partial sampling at tool-call boundaries—is responsible for the improvement.

**Consequence.** Alternative hypotheses remain untested:

- **Uniform branching**: Branch at every tool-call step, regardless of entropy. If this performs comparably to ARPO, then the entropy signal is unnecessary—the gains come from having any partial sampling that shares prefixes across trajectories, not from entropy-guided targeting.
- **Random branching**: Branch at randomly chosen tool-call steps, with the same total number of branches. If this performs comparably to ARPO, then the exploration benefit comes from *having branches* rather than from branching *at the right places*.
- **Reverse-entropy branching**: Branch at *low*-entropy steps (where the model is confident but potentially wrong). If this performs comparably to ARPO, then entropy is not a useful guide for where to branch.

The scaling analysis (Figure 8, left) shows that performance varies with the entropy weight parameter and peaks at ΔH_t = 0.4, but this only demonstrates that *the degree of entropy influence matters*, not that entropy-based selection is better than alternative selection criteria. A practitioner implementing ARPO cannot know whether the entropy computation is worth the overhead if simpler branching heuristics would work equally well.

**Evidence in the paper.** No experiment compares entropy-guided branching against uniform, random, or reverse branching. The three baselines compared in the scaling analysis (varying entropy value, initial sampling size, global rollout size) all operate within the entropy-guided framework and do not test alternative branching criteria. The paper's theoretical justification (GPG Theorem, Section 3.3) supports the validity of macro-action optimization in general but does not speak to entropy as the correct segmentation criterion.

**Mitigation status.** Not addressed. The paper treats the entropy signal as self-evidently the right branching criterion based on the pilot experiment (Section 2.2), without testing whether the observed correlation between post-tool-call entropy and task performance actually reflects a causal relationship exploitable for exploration. No future work on alternative branching criteria is suggested.

---

### 6.4 The Difficulty Estimation Cost for Adaptive Allocation Is Not Incorporated into the Method

ARPO's adaptive rollout requires monitoring token-level entropy at every tool-call step to decide where to branch. While the paper treats this as a training-time mechanism (entropy is computed from the model's own logits, which are available during generation at no extra forward-pass cost beyond the logit processing), there is a subtle cost that the paper does not address: **the branching decisions themselves require generating partial trajectories that may turn out to be dead ends** (producing incorrect answers and receiving zero reward), and the budget M - N allocated to these branches is consumed regardless of whether the branches are productive.

The paper's complexity analysis (O(n log n) to O(n²)) accounts for the shared-prefix computation savings but does not account for the *wasted* budget when entropy-guided branching explores unpromising directions. If the entropy signal is noisy—which it may be, since it is computed from only k tokens after each tool call—then branches may be triggered at steps where the model is uncertain but where no amount of exploration yields a correct answer (e.g., because the retrieved information is irrelevant to the question, or because the model fundamentally lacks the capability to solve the problem). This wasted budget could have been allocated to additional global trajectories that might discover entirely different solution strategies.

**Consequence.** In the worst case—problems where the model receives tool feedback that induces high entropy but where the correct reasoning path is fundamentally inaccessible to the model—ARPO's adaptive branching could waste a substantial fraction of the rollout budget on unproductive exploration. The paper's result that hard problems (difficulty bin 5 in GAIA, Table 2) remain essentially unsolved by both GRPO and ARPO suggests that this worst case is not just hypothetical: on the hardest problems, entropy spikes may be high (the model is confused by the tool output), but branching provides no benefit because no branch will produce a correct answer. The budget consumed by these branches is a deadweight loss that trajectory-level RL, which allocates all budget to independent global trajectories, avoids.

**Evidence in the paper.** Table 2 shows that on GAIA Level 3 (the hardest tier), ARPO with Qwen3-8B achieves 16.7% (matching GRPO's 8.3% and exceeding it, but still low in absolute terms). The paper does not report how many branches were triggered on hard vs. easy problems, what fraction of branches produced correct answers, or whether branching was beneficial or wasteful on the hardest subset. Without this data, the efficiency of budget allocation cannot be assessed from the paper's results.

**Mitigation status.** Not acknowledged. The paper presents ARPO's branching as universally beneficial and does not discuss the possibility of wasted exploration budget on hard problems. The termination condition in the adaptive rollout mechanism (Step 4 in Section 3.1) ensures the total budget M is always fully utilized, but it does not include any mechanism for early termination of unpromising branches based on intermediate PRM-like verifier scores—a natural extension that could prevent budget waste.

---

### 6.5 The Method Is Evaluated Exclusively on Qwen and Llama Model Families with GRPO-Based Policy Updates

All experiments use either Qwen2.5 (3B, 7B), Qwen3 (8B, 14B), or Llama3.1 (8B) as base models, and all RL comparisons (including ARPO) use GRPO as the underlying policy optimization algorithm. The paper states:

> "this model is representative of the capabilities of many contemporary LLMs" (Section 4)

but provides no evidence beyond these two model families. Several architectural properties could interact with ARPO's mechanism in ways that affect generalizability:

**Entropy distribution properties.** Different model families may have different calibration properties—some models may produce sharper or flatter next-token distributions, affecting both the baseline entropy levels and the magnitude of entropy spikes after tool calls. The optimal entropy threshold τ and weight β identified for Qwen2.5-7B (Figure 8) may not transfer to models with different calibration.

**In-context learning and tool-use capability.** Different base models have different inherent abilities to follow tool-call formats, integrate external information, and recover from tool errors. A model with stronger pre-existing tool-use skills may benefit less from ARPO's step-level exploration; a model with weaker skills may need more aggressive branching. The paper's consistent ~4 percentage point improvement across models suggests robustness, but testing only two architectures leaves substantial uncertainty.

**GRPO-specific assumptions.** ARPO's advantage attribution estimation (both hard and soft variants) is designed within the GRPO framework, exploiting the group-relative advantage normalization and the importance sampling ratio's behavior on shared prefixes. Whether the entropy-based adaptive rollout mechanism would benefit other RL algorithms—PPO with a learned value function, REINFORCE with different baseline estimation, or off-policy methods—is untested. The GPG Theorem (Section 3.3) suggests macro-action optimization is broadly applicable, but the specific integration with GRPO's clipped objective and group normalization may be load-bearing in ways the theorem does not capture.

**Consequence.** A practitioner using a non-Qwen, non-Llama model (e.g., Mistral, Gemma, DeepSeek, or a proprietary model accessed via API with limited logit access) has no direct evidence that ARPO's benefits will transfer. The entropy computation requires access to full vocabulary logits, which API-based models may not expose, making the method unimplementable in black-box settings. The optimal hyperparameters (α, β, τ, N/M ratio) identified through scaling analysis may need to be re-tuned for each new model family, and the paper provides no guidance on how to do so efficiently.

**Evidence in the paper.** All five model configurations in Tables 1 and 2 are from two model families (Qwen and Llama). The paper acknowledges no architectural generalizability limitation. The entropy pilot experiment (Section 2.2) uses two types of agents (search-based and code-based) but does not specify whether the entropy pattern was validated across model families beyond the primary experimental models.

**Mitigation status.** Not addressed. The paper claims representativeness based on the models' "capabilities" but provides no cross-architecture validation. No future work on broader model evaluation is suggested in Section 6.

---

### 6.6 No Combination of PRM-Guided Search or Verifier-Based Selection with the Entropy-Based Rollout

ARPO's adaptive rollout mechanism expands exploration at high-entropy tool-call steps by branching additional partial trajectories from those points. After all branches complete, the RL update uses group-relative advantage estimation across the pooled set of global and branched trajectories. However, **ARPO does not incorporate any mechanism for evaluating branch quality during generation**—it generates all branches to completion, computes final-answer rewards, and retroactively assigns credit.

This is a significant missed opportunity. In the reference paper analyzed earlier (Snell et al., 2025, on compute-optimal test-time scaling), a key finding was that process reward models (PRMs) can guide search during generation, pruning unpromising branches early and reallocating compute to more promising ones. ARPO could analogously use a learned verifier (a PRM or outcome reward model trained on the base model's outputs) to score partial trajectories at branching points and decide whether to continue investing generation budget in each branch. Without such a mechanism, ARPO allocates the same budget to every branch regardless of intermediate quality signals—the branch that immediately misinterprets tool output gets the same continued generation investment as the branch that interprets it correctly.

**Consequence.** The efficiency gains ARPO reports (half the tool-call budget) come from sharing prefixes across trajectories, not from intelligent pruning of low-quality branches. In scenarios where most branches from a high-entropy point produce incorrect answers (which is likely on hard problems—see Limitation 6.4), ARPO wastes budget generating full trajectories for doomed branches. A verifier-guided variant could terminate or de-prioritize low-quality branches early, further improving efficiency. The paper's technique of generating k tokens for entropy measurement after each tool call is already producing logits that could be repurposed for a lightweight quality assessment—the infrastructure for branch evaluation is partially in place but unused for this purpose.

Additionally, ARPO's final answer selection uses only the reward signal for RL credit assignment; it does not incorporate verifier-guided best-of-N selection across branches at inference time. This means that even if ARPO training discovers a diverse set of tool-use strategies through branching, at deployment time the model generates a single trajectory and cannot leverage the branch diversity it learned to produce during training.

**Evidence in the paper.** The paper uses only final-answer correctness as the reward signal (Equation 8, hierarchical reward design) and does not train or employ any intermediate verifier. The ~38% correct-to-incorrect reversion rate discussed in the reference paper's revision model (Section 6.1 there) has a direct analog here: branches from an initially correct reasoning prefix may diverge into incorrect continuations, and ARPO has no mechanism to detect this mid-generation and re-allocate budget. The case studies in Appendix F show trajectories with multiple tool calls, but there is no indication of intermediate quality assessment.

**Mitigation status.** Not addressed. The paper focuses exclusively on improving the RL sampling mechanism and does not discuss integration with verifier-guided search or inference-time best-of-N selection across branches. This is consistent with the paper's scope (algorithmic improvement to RL training), but a practitioner seeking to maximize both training efficiency and inference-time performance would need to combine ARPO with verifier-based techniques that the paper does not explore.

## 7. Implications and Future Directions
- Field impact
  - Shifts agent training from “whole‑trajectory” thinking to “step‑aware” exploration guided by uncertainty. This reframes RL for LLM agents as a sequence of macro decisions aligned to tool‑feedback points (Eq. 9), a useful perspective for many tool‑rich tasks.

- Practical applications
  - Cost‑effective training of research assistants, web agents, and code‑augmented solvers: fewer tool calls during RL (Figure 7) reduce monetary/API costs while improving outcomes (Tables 1–2).

- Research opportunities
  - Adaptive thresholds and learned branching policies: replace hand‑tuned α, β, τ with learned controllers using entropy and other uncertainty proxies.
  - Broader uncertainty signals: combine token entropy with verifier disagreement, calibration, or retrieval confidence (beyond Eq. 3).
  - Value‑based or hierarchical RL: couple ARPO’s partial rollouts with learned value estimates for macro actions; integrate planning over tool‑use graphs.
  - Multi‑agent/tool scheduling: extend advantage attribution to coordinated decisions across multiple tools or agents, possibly with credit assignment across longer horizons.
  - Robust evaluation: complement LLM‑as‑Judge with programmatic verifiers and human audits; study domain shift (languages, domains, low‑resource web).

> Bottom line: ARPO demonstrates that targeting exploration to post‑tool high‑uncertainty steps and crediting shared vs. branched tokens differently yields better, cheaper training for tool‑using LLM agents, with both empirical wins (Tables 1–2, Figures 6–7) and a principled macro‑action gradient foundation (Eq. 9; Appendix D.2).
