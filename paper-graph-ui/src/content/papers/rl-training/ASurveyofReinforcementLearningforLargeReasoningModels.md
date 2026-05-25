# A Survey of Reinforcement Learning for Large Reasoning Models

**ArXiv:** [2509.08827](https://arxiv.org/abs/2509.08827)

## 🎯 Pitch

This comprehensive survey establishes the first unified framework for applying reinforcement learning (RL) to Large Reasoning Models (LRMs)—advanced language and multimodal models designed for complex, multi-step reasoning. By synthesizing recent advances in reward design, policy optimization, scalable infrastructure, and open problems, it distinguishes RL for reasoning from alignment-focused approaches like RLHF/DPO and clarifies how to scale LLMs into verifiable, agentic, and generalist problem solvers. This work matters because it lays the methodological foundation for transforming LLMs into AI systems capable of planning, tool use, and scientific reasoning, offering a roadmap for the next wave of practical and scientific breakthroughs in AI.

---

## 1. Executive Summary

This paper surveys the recent, rapid adoption of reinforcement learning to transform Large Language Models into Large Reasoning Models, focusing on the paradigm shift from alignment-centric RLHF/DPO to **Reinforcement Learning with Verifiable Rewards (RLVR)**. Systematizing work since the release of DeepSeek-R1 and OpenAI o1, the survey dissects the foundational components—reward design (e.g., verifiable vs. generative rewards), policy optimization (e.g., GRPO vs. PPO), and sampling strategies (e.g., dynamic sampling)—and identifies open foundational problems, including whether RL's role is one of sharpening latent capabilities or discovering entirely new ones and the debate on whether SFT memorizes while RL generalizes. Analyzing a landscape spanning static corpora, dynamic environments, and specialized infrastructure, the survey catalogues how RLVR has driven state-of-the-art reasoning across coding, agentic tasks, multimodal understanding, and robotics, establishing that scalable RL is now the dominant post-training methodology for complex reasoning tasks, while concurrently highlighting that its efficacy is fundamentally bounded by the availability and quality of verifiable reward signals.

## 2. Context and Motivation

### The Core Problem: RL for LLMs Has Undergone a Fundamental Shift, But the Field Lacks Systematic Organization

The central problem this survey addresses is not a single technical gap, but rather a **structural gap in the research landscape**: the rapid transformation of how reinforcement learning is applied to large language models has produced a fractured, contradictory, and rapidly expanding literature that no existing survey adequately organizes. Since late 2024, the field has undergone a dramatic pivot — away from RL as a tool for human alignment (RLHF, DPO) and toward RL as the primary engine for training models to *reason* through complex, multi-step problems (RLVR). This shift, catalyzed by the releases of OpenAI o1 and DeepSeek-R1, has generated hundreds of papers in under a year, each proposing variants of reward design, policy optimization algorithms, sampling strategies, and training infrastructure. Yet the field lacks a **unified taxonomy, a clear articulation of the foundational debates, and a systematic map of the rapidly evolving resource landscape**.

The survey identifies this as an urgent problem for several reasons (Section 1):

- **Algorithmic fragmentation**: The space of policy optimization algorithms has exploded — GRPO, DAPO, GSPO, CISPO, Dr. GRPO, and dozens more — each claiming improvements over predecessors. Without systematic comparison and categorization, practitioners cannot make informed decisions about which algorithm to adopt for a given task, model scale, or compute budget. The survey explicitly frames this as a "tricks vs. traps" problem (Section 4.4): many proposed improvements may be artifacts of specific experimental configurations rather than robust advances.

- **Resource duplication**: As shown in Table 4 and Table 5, the community has produced numerous static datasets and dynamic environments for RL training, often with overlapping goals but incompatible formats, reward structures, and verification mechanisms. Without a comprehensive inventory, new research efforts risk reinventing existing resources or, worse, building on flawed foundations.

- **Infrastructure complexity**: The computational demands of RLVR — which requires coordinating model inference (rollout generation), reward computation (often involving code execution sandboxes or environment simulators), and policy updates (gradient computation across distributed hardware) — far exceed those of supervised fine-tuning. The proliferation of frameworks (veRL, OpenRLHF, AReaL, slime, ROLL, etc., detailed in Table 6) reflects genuine engineering challenges rather than mere duplication, but the landscape is confusing for newcomers and makes reproducible research difficult.

- **Unresolved conceptual debates**: Foundational questions about what RL actually *does* to language models remain contentious. Does RL discover genuinely new reasoning capabilities, or merely sharpen (amplify) patterns already present in the pretrained model? Does RL generalize beyond its training distribution, or does it memorize — and does supervised fine-tuning do the opposite? These are not philosophical questions; they have direct practical implications for how training budgets should be allocated between pretraining, SFT, and RL stages.

### Why This Problem Matters: RLVR Represents a New Scaling Axis with Real-World Deployment Implications

The significance of organizing this literature extends beyond academic housekeeping. The survey argues, both explicitly and through its comprehensive scope, that RLVR represents a **fundamentally new axis for scaling model capabilities** — one that is orthogonal to pretraining data and parameter scaling (Section 2.2, Figure 2). This has profound implications:

**Economic implications for compute allocation.** If RL can elicit reasoning capabilities from smaller base models that previously required much larger pretrained models, the economics of AI deployment shift. Organizations can potentially invest less in expensive pretraining and more in inference-time or post-training RL compute. The survey's taxonomy of training resources (Section 5) directly serves this decision-making: knowing which static corpora, dynamic environments, and infrastructure frameworks exist enables cost-benefit analyses for different RL approaches.

**Domain expansion beyond mathematics and code.** The initial successes of RLVR were concentrated in domains with clean, automatically verifiable correctness signals — mathematics (exact answer matching) and coding (unit test pass/fail). A central tension the survey highlights is that the "Verifier's Law" (the ease of training AI systems is proportional to how verifiable the task is, discussed in Section 3.1.1) both explains RLVR's success and defines its current boundary. The survey's extensive cataloguing of generative rewards (Section 3.1.2), dense rewards (Section 3.1.3), unsupervised rewards (Section 3.1.4), and applications in non-verifiable domains like medicine (Section 6.6) and GUI agents (Section 6.2) demonstrates that the field is actively pushing beyond this boundary. Understanding what reward structures work in which domains is essential for extending RLVR to open-ended reasoning, creative tasks, and real-world agentic scenarios.

**The emergence of reasoning as a trainable behavior.** Prior to RLVR, long chain-of-thought reasoning with self-reflection, verification, and backtracking was primarily an emergent property of very large models prompted in specific ways. DeepSeek-R1's demonstration that smaller models could be *trained* into these behaviors through RL with simple correctness and format rewards (Section 2.2) shifted reasoning from an emergent phenomenon to a trainable skill. This has direct implications for the democratization of advanced AI: if reasoning can be trained rather than requiring massive scale, the barrier to entry for capable models decreases substantially. The survey's detailed coverage of open-source models (Table 1, Figure 4) reflects and reinforces this democratization trend.

### Where Prior Approaches Fall Short

The survey identifies several categories of limitations in existing work that motivate its comprehensive approach:

**Surveys that focus on RL without addressing LLMs.** Works like Ghasemi et al. (2024) on general RL algorithms, Huh and Mohapatra (2023) on multi-agent RL, and Zhang et al. (2024b) on self-play techniques provide broad perspectives on RL but "do not explicitly address its application to LLMs" (Section 2.3). This is a critical gap because RL for LLMs introduces unique challenges absent in classical RL: the action space is the entire token vocabulary, rollouts are extremely expensive (each "action" requires a full forward pass through a billion-parameter model), and the "environment" is often static text rather than a dynamic state space with transition dynamics.

**Surveys that focus on LLM reasoning without centering RL.** Works like Chen et al. (2025n) on long chain-of-thought reasoning, Zhang et al. (2025a) on replication studies of reasoning LLMs, and Li et al. (2025y) on the transition from System 1 to System 2 reasoning treat RL as "only one element among a wide range of reasoning strategies" (Section 2.3). The survey argues this is increasingly inadequate: RL has become the *primary* post-training methodology for reasoning, not just one option among many. A survey that places reasoning at the center and treats RL as a tool cannot adequately capture the algorithmic innovations, reward design choices, and infrastructure requirements that are specific to RL-based training.

**Surveys focused on alignment rather than reasoning.** Srivastava and Aggarwal (2025) bridge RL and LLMs but remain "primarily focused on alignment rather than reasoning capabilities" (Section 2.3), covering RLHF, RLAIF, and DPO. This reflects the pre-2024 paradigm where RL's role in LLMs was to make models helpful, harmless, and honest (the "3H" desiderata). The survey argues this paradigm is now insufficient: RLVR's purpose is not to align behavior with human preferences but to *incentivize reasoning itself* — to produce models that explore, verify, backtrack, and self-correct in pursuit of correct answers.

**Absence of systematic coverage of training resources and infrastructure.** Even more recent surveys that touch on RL for reasoning tend to focus on algorithmic aspects while neglecting the practical substrate: static datasets, dynamic environments, and the infrastructure frameworks that make large-scale RL training possible. The survey's detailed tables (Tables 4, 5, 6) represent an organizational contribution that is absent from prior work.

### How This Paper Positions Itself

The survey positions itself at the center of a Venn diagram whose circles are RL, LLMs, and reasoning — with a deliberate emphasis on the first term (Section 2.3):

> "Unlike previous surveys that cover either general RL or reasoning in LLMs, we place RL at the center and provide a systematic synthesis of its role throughout the LLM training lifecycle."

This centering is the survey's defining intellectual move. Rather than treating RL as a method that can be applied to LLMs for reasoning, it treats the **RL-for-LLMs pipeline** as a coherent object of study with its own components (reward design, policy optimization, sampling strategies), its own foundational debates (sharpening vs. discovery, SFT vs. RL generalization, process vs. outcome rewards), its own resource ecosystem (static corpora, dynamic environments, infrastructure frameworks), and its own application domains (coding, agents, multimodal, multi-agent, robotics, medicine).

The survey's scope is deliberately broad and inclusive. It covers work "especially since the release of DeepSeek-R1" (Abstract) — a choice that reflects the genuine inflection point that model represented. Before DeepSeek-R1, RL for reasoning was a niche research direction; after it, every major lab and many open-source efforts adopted RLVR as a core training methodology. The survey captures this Cambrian explosion of methods, systematizing it into a taxonomy (Figures 1, 5, 6) that is designed to be extensible as the field continues to evolve.

A key aspect of the survey's positioning is its **explicit focus on scalability and the path toward Artificial Superintelligence (ASI)** (Section 1). It frames RLVR not as an end state but as an intermediate step toward "open-ended RL" — scenarios where models generate their own tasks, environments, and reward signals without human specification. This forward-looking orientation distinguishes it from purely retrospective surveys: the taxonomy is designed not just to describe existing work but to identify "future opportunities and directions for this rapidly evolving area" (Abstract).

Finally, the survey makes a methodological argument through its structure: that understanding RL for LLMs requires *simultaneous* attention to algorithms, rewards, resources, and applications. The foundational components (Section 3) cannot be understood in isolation from the foundational problems (Section 4), which in turn depend on the training resources (Section 5) that make large-scale experimentation possible. By weaving these threads together — rather than treating them as separate survey topics — the paper argues implicitly that progress in any one area (e.g., better reward design) is gated by progress in others (e.g., infrastructure that supports dynamic environment interaction during training).

## 3. Technical Approach

This is a **survey paper**, not a paper proposing a single new system. Its core idea is that reinforcement learning for LLMs has become the dominant post-training methodology for reasoning, and that understanding this field requires a systematic decomposition into (1) *reward design*, (2) *policy optimization algorithms*, and (3) *sampling strategies* — three interacting components whose design choices fundamentally determine whether RL elicits shallow memorization or robust reasoning.

### 3.1 Reader Orientation

The "system" being analyzed is the **reinforcement learning training pipeline for large language models** — the end-to-end machinery that takes a pretrained (or SFT-ed) language model, defines a reward signal that identifies correct reasoning, generates candidate solutions (rollouts) by sampling from the model, scores those candidates with the reward, computes policy gradients to increase the probability of high-reward outputs, and iterates this process over thousands of steps.

The problem this pipeline solves is: **how do you transform a language model from a next-token predictor into a system that produces step-by-step reasoning, self-verification, and error correction, using only automatically computable reward signals?** The "shape" of the solution is a closed loop — generate, verify, update — where the reward function defines what *correct* reasoning looks like, the policy optimization algorithm defines *how* the model is updated to pursue that reward, and the sampling strategy defines *what* the model generates and how compute is allocated during training.

### 3.2 Big-Picture Architecture (Diagram in Words)

The RL training pipeline for LLMs has five major components, connected in a loop:

1. **Base Language Model (the Policy `$\pi_{\theta}$`)** — the pretrained or SFT-ed model that generates text. It takes a prompt `$x$` as input and autoregressively samples a response `$y = (y_1, \ldots, y_T)$`. This is the object being trained.

2. **Reward Signal (`$R(x, y)$` or per-token `$r_t$`)** — a function that scores how "good" a generated response is. It can be outcome-level (a single scalar for the whole response, e.g., correct/incorrect for math answers), token-level (a per-token score for every generation step), step-level (a score per reasoning segment), or turn-level (per agent-environment interaction). The reward can come from rule-based verifiers (exact match, unit tests), learned reward models trained on human preferences or Monte Carlo rollouts, generative reward models that produce critiques, or unsupervised signals derived from model confidence or output consistency.

3. **Rollout Generation (Sampling)** — the process of actually generating responses from the current policy. This is where the model interacts with prompts to produce candidate solutions. The sampling strategy controls *what* prompts to generate on (dynamic difficulty-based selection, curriculum learning), *how many* generations per prompt (`$G$`), *at what temperature*, and with what structural constraints (chain-of-thought vs. tree-structured rollouts, prefix reuse for efficient KV-cache computation).

4. **Advantage Estimation** — the computation that determines *how much better* a particular action (token or sequence) was compared to the expected baseline. This is the core of the policy gradient: the model should increase the probability of actions that yielded better-than-expected outcomes and decrease the probability of actions that yielded worse-than-expected outcomes. Advantages can be computed via learned critic models (GAE in PPO), group-relative normalization (GRPO's `$ \hat{A}_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})} $`), leave-one-out baselines (RLOO), or implicit PRMs that derive token-level rewards from outcome reward models.

5. **Policy Update (Optimizer)** — the gradient-based update that modifies the model parameters `$\theta$` to increase the probability of advantageous actions. This typically involves a clipped surrogate objective (PPO-style) to prevent the policy from changing too much in a single update, plus optional regularization terms (KL divergence from a reference policy, entropy bonuses to maintain exploration, length penalties to control reasoning verbosity).

Information flows as follows: a batch of prompts is sampled → the current policy generates `$G$` responses per prompt → the reward function scores each response → the advantage estimator computes per-token or per-sequence advantage values → the policy gradient uses these advantages to compute a loss → the optimizer updates the model parameters → the updated model becomes the new policy for the next iteration. Optionally, a reference policy `$\pi_{\text{ref}}$` (often the initial SFT model or a periodically updated snapshot) provides KL regularization to prevent catastrophic drift.

### 3.3 Roadmap for the Deep Dive

- **First: the MDP formulation and policy gradient objective (Section 3.4.1).** This is the mathematical foundation that all subsequent algorithms build on. Understanding how token generation maps to states, actions, transitions, and rewards in a Markov Decision Process is essential before discussing specific algorithms.

- **Second: reward design (Section 3.4.2).** The reward signal is the *only* teaching signal the model receives. The survey's taxonomy spans verifiable rewards (rule-based, automatic), generative rewards (model-generated critiques and scores), dense rewards (token-level, step-level, turn-level), unsupervised rewards (derived from model consistency or confidence), and reward shaping (combinations and transformations). We walk through each category with representative examples and the tradeoffs they embody.

- **Third: policy optimization algorithms (Section 3.4.3).** This covers the critic-based vs. critic-free dichotomy, the mechanics of PPO and GRPO (the two dominant families), importance sampling for off-policy correction, and the regularization objectives (KL, entropy, length penalties) that prevent training collapse. The order moves from the general policy gradient formulation to specific algorithmic variants, reflecting how each algorithm modifies the basic template.

- **Fourth: sampling strategies (Section 3.4.4).** Rewards and updates only matter if the model generates useful rollouts. This section covers dynamic sampling (online difficulty filtering, exploration-oriented dropout scheduling), structured sampling (tree search rollouts, shared-prefix KV-cache reuse), and hyperparameter management (temperature schedules, context length curricula).

### 3.4 Detailed, Sentence-Based Technical Breakdown

The survey's technical contribution is organizational, not algorithmic. It provides a **comprehensive taxonomy and synthesis** of the rapidly evolving design space for RL-based LLM training. The taxonomy is organized around three foundational components — reward design, policy optimization, and sampling strategies — with each component further decomposed into major research directions and representative works.

---

#### 3.4.1 The MDP Formulation of Language Generation

**What is being formalized:** the survey establishes a standard mapping from the language generation process to a Markov Decision Process (MDP), which is the mathematical framework required to apply RL. This mapping is not novel to the survey but is the prerequisite for understanding all subsequent algorithms.

**The MDP tuple.** The standard RL formulation defines an MDP as a tuple `$(\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma)$`. In the language model context, these components are mapped as follows:

- **State space `$\mathcal{S}$`:** The state `$s_t$` at step `$t$` is the concatenation of the original prompt `$x$` and all tokens generated so far: `$s_t = (x, a_{1:t-1})$`. The initial state `$s_1$` is just the prompt `$x$`. When the model generates an end-of-sequence token, the state transitions to a terminal state, ending the episode.

- **Action space `$\mathcal{A}$`:** The action `$a_t$` at step `$t$` is the selection of a single token from the vocabulary `$\mathcal{V}$`. This means each "action" is a discrete choice among tens of thousands of possible tokens. Critically, the action granularity can be defined at multiple levels (discussed below).

- **Transition dynamics `$\mathcal{P}$`:** The transition is *deterministic* in the standard formulation: `$s_{t+1} = [s_t, a_t]$` where `$[\cdot, \cdot]$` denotes string concatenation. There is no stochasticity in the environment — the next state is always the current state plus the chosen token.

- **Reward function `$R$`:** Depending on the granularity, the reward can be:
  - **Sequence-level:** `$R(x, y)$` — a single scalar assigned only at the end of the complete response.
  - **Token-level:** `$r_t = R(x, a_{1:t})$` — a per-token reward for each generation step.
  - **Step-level:** `$r_k = R(x, y^{(1:k)})$` — a per-segment reward for each reasoning step.
  - **Turn-level:** `$r_u = R(x, y^{(1:u)}, z^{(1:u)})$` — a per-interaction reward for each agent-environment turn, where `$z^{(u)}$` is the environment feedback at turn `$u$`.

- **Discount factor `$\gamma$`:** Typically set to `$\gamma = 1$` for finite-horizon language tasks, meaning all tokens/steps contribute equally to the return without temporal discounting.

- **Return `$G$`:** The cumulative reward over the trajectory. For sequence-level rewards with `$\gamma = 1$`, this reduces to `$G = R(x, y)$`. For token-level rewards, `$G = \sum_{t=1}^{T} \gamma^{t-1} r_t$`. For step-level, `$G = \sum_{k=1}^{K} \gamma^{k-1} r_k$`.

**Why this mapping matters.** By casting language generation as an MDP, the entire machinery of RL — policy gradients, value functions, advantage estimation, on-policy and off-policy algorithms — becomes directly applicable to LLM training. The key difference from classical RL is the *enormity* of the action space (the full vocabulary) and the *cost* of sampling (each action requires a forward pass through a billion-parameter model).

**The policy gradient objective.** The learning objective is to maximize expected cumulative reward over the prompt distribution `$\mathcal{D}$`:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot|x)} \left[ G \right]$$

where `$\pi_{\theta}$` is the language model (the policy parameterized by `$\theta$`), `$x$` is a prompt drawn from the data distribution, `$y$` is a complete response sampled autoregressively from the policy, and `$G$` is the return (cumulative reward) of that response.

**What this equation computes:** the expected total reward obtained by the model when interacting with prompts drawn from the training distribution. The expectation is over both the prompt distribution (which prompts the model sees) and the policy's own stochastic sampling (which responses it generates). Maximizing this objective means making the model generate responses that, on average across the training set, receive higher rewards.

**Why this form:** this is the standard RL objective for episodic tasks with stochastic policies. It captures the fundamental tension: the model must learn to produce high-reward outputs, but it can only learn from the outputs it actually generates, which depend on its current (potentially suboptimal) policy. This is the exploration-exploitation problem that all RL algorithms must address.

**The general policy gradient estimator.** To optimize `$\mathcal{J}(\theta)$`, one computes the gradient with respect to model parameters:

$$\nabla_{\theta} \mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}} \left[ \sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}(y_t | y_{<t}) \cdot Q_t \right]$$

where `$\pi_{\theta}(y_t | y_{<t})$` is the model's probability of generating token `$y_t$` given the prompt and previous tokens, and `$Q_t$` is the expected future return from state `$s_t$` after taking action `$a_t = y_t$`.

**What this equation computes:** the gradient of the expected return with respect to the model parameters. For each token position `$t$` in each generated response, it computes the gradient of the log-probability of the chosen token (which points in the direction that *increases* that token's probability), then multiplies it by `$Q_t$` (a measure of how good it was to choose that token at that point). If `$Q_t$` is positive, the gradient pushes the model to increase the probability of token `$y_t$` in context `$s_t$`; if negative, it pushes the model away from that token.

**Why this form:** this is the REINFORCE (policy gradient) estimator. Its key property is that it provides an *unbiased estimate* of the true gradient, meaning that with enough samples, the average gradient points in the correct direction. However, it suffers from high variance because `$Q_t$` can vary dramatically across different trajectories. This motivates the introduction of *baselines* and *advantage functions*, which subtract a state-dependent baseline from `$Q_t$` to reduce variance without introducing bias. The advantage function `$A(s, a) = Q(s, a) - V(s)$` measures how much better action `$a$` is compared to the average action in state `$s$`, and is the central quantity in modern policy optimization algorithms.

---

#### 3.4.2 Reward Design Taxonomy

The reward signal is the *sole teaching signal* that guides the model toward correct reasoning. The survey decomposes reward design into five major categories, each representing a different approach to answering the question: "What constitutes a correct or good response, and how do we measure it automatically?"

##### Verifiable Rewards (Section 3.1.1)

**What they are:** rewards that can be computed deterministically from the model's output and a ground-truth answer or specification, without any learned model or human judgment in the loop.

**The two standard forms.** In practice, two kinds of rule-based verifiable rewards are widely used:

- **Accuracy rewards:** For tasks with deterministic outcomes like mathematics, the model must produce its final answer within a prescribed delimiter (commonly `\boxed{...}`). An automatic checker — typically implemented through Python libraries like Math-Verify (huggingface) or SymPy — compares this output to the ground truth. For coding tasks, unit tests or compilers provide the pass/fail signal. The reward is binary or nearly-binary: correct answers receive a positive reward, incorrect answers receive zero or negative reward.

- **Format rewards:** These impose structural constraints requiring the model to place its private chain-of-thought between `  thinking ` and `  response ` tags, and to output the final answer in a separate field (e.g., `<answer>...</answer>`). This improves reliable parsing and verification in large-scale RL by making it trivial to extract the final answer for comparison with ground truth, regardless of what reasoning the model produced internally.

**Why they are preferred when available.** DeepSeek-V3 and DeepSeek-R1 (Section 2.2) explicitly argue that learned reward models may suffer from *reward hacking* when scaled to large-scale RL settings — the policy may learn to produce outputs that score highly under the learned reward model but are actually incorrect or meaningless. Rule-based rewards, by contrast, are resistant to manipulation because they are defined by deterministic correctness criteria rather than learned preferences. The survey formalizes this as **Verifier's Law**: "the ease of training AI systems to perform a task is proportional to the degree to which the task is verifiable." Tasks that are "difficult to solve yet comparatively easy to verify" — mathematics, competitive programming — are the ideal domain for RLVR because they satisfy four criteria: (1) clear ground truth exists, (2) rapid automated verification is available, (3) evaluating many candidate solutions is scalable, and (4) the reward signal is closely aligned with correctness.

**The limitation boundary.** Tasks lacking fast or objective verification — open-ended question answering, free-form writing, creative tasks — remain challenging for outcome-based RL because they must rely on noisy learned reward models or subjective human feedback. This boundary is precisely what motivates the next category.

##### Generative Rewards (Section 3.1.2)

**What they are:** rewards produced by *another language model* (a Generative Reward Model, or GenRM) that is prompted or trained to evaluate outputs. Instead of a deterministic rule check, a GenRM produces structured critiques, rationales, scalar scores, or preference judgments.

**Why they are necessary.** Verifiable rewards are limited to domains with objective ground truth. Many complex reasoning tasks — particularly in open-ended, creative, or subjective domains — lack such ground truth. GenRMs extend RL's applicability to these domains by providing *learned* evaluation signals. The survey identifies two primary applications:

**Application 1: Model-based verifiers for verifiable tasks.** Even in domains with ground truth, rule-based systems are brittle — they often produce false negatives when a model generates a correct answer in an unexpected format. Specification-based GenRMs are trained to *semantically* assess the equivalence between a model's free-form output and a reference answer, serving as more flexible verifiers. Examples include TinyV (Xu et al., 2025g), a lightweight verifier that augments rule-based systems, and CompassVerifier (Liu et al., 2025b), a multi-domain verifier capable of handling diverse data types.

**Application 2: Assessment-based GenRMs for non-verifiable tasks.** This is where GenRMs provide signals for tasks where Verifier's Law does not hold. The survey categorizes these approaches by their core design:

- **Reasoning Reward Models ("Learning to Think"):** These RMs are trained to *explicitly reason before rendering a judgment*. Rather than just predicting a preference score, they generate a chain-of-thought critique and then derive a score from that analysis. Foundational to the LLM-as-a-Judge concept (Li et al., 2023b; Zheng et al., 2023), this approach is now central to state-of-the-art RMs. Representative works include: AIR (He et al., 2025a) which systematically analyzes annotations, instructions, and response pairs in preference data; DeepSeek-GRM (Liu et al., 2025b) which scales reward modeling through inference-time compute; RM-R1 (Chen et al., 2025q) which formulates reward modeling as a reasoning task; and RRM (Reward Reasoning Model, Guo et al., 2025b) which explicitly incentivizes reward models to produce correct evaluations. A key meta-innovation is that these reasoning RMs are themselves often *trained with RL*, using simple verifiable meta-rewards (e.g., whether their final verdict matches a known ground-truth evaluation).

- **Rubric-based Rewards ("Structuring Subjectivity"):** To anchor subjective evaluation in consistent criteria, these frameworks employ structured rubrics — natural language descriptions of what constitutes quality along multiple dimensions. Unlike rule-based approaches that use hard-coded logic for objective tasks, rubric-based methods capture nuanced evaluation criteria for subjective domains. Examples include: RaR (Rubrics as Rewards, Gunjal et al., 2025) which applies reinforcement learning with rubric-based evaluation to non-verifiable domains; Rubicon (Huang et al., 2025f) which anchors RL training in rubrics for more stable learning; and ProxyReward (Guo et al., 2025e) which decomposes high-level tasks into a set of verifiable proxy questions that can be automatically checked.

- **Co-Evolving Systems ("Unifying Policy and Reward"):** The most advanced paradigm moves beyond static evaluators toward dynamic systems where the policy model and reward model improve together. This can occur through *self-rewarding* (Yuan et al., 2024), where a single model generates its own training signals by alternating between policy and verifier roles (PAG, Jiang et al., 2025e), or through *co-optimization* (RL Tango, Zha et al., 2025), where the policy and a separate reward model are trained concurrently with a shared outcome-level reward. Cooper (Hong et al., 2025a) co-optimizes both models to enhance robustness and mitigate reward hacking — the phenomenon where the policy learns to exploit weaknesses in the reward model rather than genuinely improving.

##### Dense Rewards (Section 3.1.3)

**What they are:** rewards provided at *every* or *nearly every* decision step, rather than only at the end of a complete response. In the language context, this means breaking the generation process into sub-units (tokens, reasoning steps, conversational turns) and assigning a reward to each.

**Why they matter for reasoning.** The credit assignment problem — determining *which part* of a long chain of reasoning was responsible for the final correct or incorrect answer — becomes increasingly difficult as response length grows. If a model generates a 2000-token mathematical solution and gets the correct final answer, which of those 2000 tokens were crucial? Outcome-only rewards treat all tokens equally, which can encourage "unfaithful chain-of-thought" (producing correct answers through reasoning that doesn't actually support the conclusion). Dense rewards provide per-step signals that can distinguish between productive and unproductive reasoning steps.

**The action-reward granularity matrix (Table 2).** The survey formalizes four levels of granularity:

| Granularity | Action | Reward | Return |
|---|---|---|---|
| Trajectory | Entire sequence `$y = (a_1, \ldots, a_T)$` | Scalar `$R(x, y)$` | `$R(x, y)$` |
| Token | Each token `$a_t \in \mathcal{V}$` | `$r_t = R(x, a_{1:t})$` | `$\sum_{t=1}^{T} \gamma^{t-1} r_t$` |
| Step | Segment `$y^{(k)}$` (e.g., sentence) | `$r_k = R(x, y^{(1:k)})$` | `$\sum_{k=1}^{K} \gamma^{k-1} r_k$` |
| Turn (Agent) | Agent response `$y^{(u)}$` per turn | `$r_u = R(x, y^{(1:u)}, z^{(1:u)})$` | `$\sum_{u=1}^{U} \gamma^{u-1} r_u$` |

**Token-level rewards.** These provide a reward for *every single generated token*. This is the most fine-grained signal. Two major approaches exist:

- **Implicit reward models:** DPO (Rafailov et al., 2023) and its extensions (Rafailov et al., 2024) show that token-level rewards can be derived as log-likelihood ratios between the policy and reference models. **Implicit PRM** (Yuan et al., 2025d) demonstrates that these token-level rewards can be obtained by training an Outcome Reward Model (ORM) and then using the DPO parameterization to extract per-token signals without training a separate process reward model. **PRIME** (Cui et al., 2025a) integrates ORM learning directly into the RL training loop and uses the resulting implicit token-level rewards to train the policy, achieving process-level supervision without process-level labels.

- **Internal feedback signals:** An alternative line derives token-level rewards from the model's own internal states. Examples include token entropy (Cheng et al., 2025a) — using the model's uncertainty at each token as a reward signal (high-entropy tokens are those where the model is uncertain, potentially indicating reasoning "forks"), and strategic grams (HICRA, Wang et al., 2025g) — using attention patterns or other internal features.

**Step-level rewards.** These provide a reward for each *logical step* or *segment* of reasoning. The model's generation is segmented into discrete steps (e.g., by sentence boundaries, by special delimiter tokens, or by detecting low-probability "forking" tokens), and each step receives its own reward.

**Model-based step rewards** train a separate Process Reward Model (PRM) to score steps. The PRM takes the prompt and the partial solution up to the current step as input and outputs a scalar probability that the solution will ultimately be correct. Key approaches include: Math-Shepherd (Wang et al., 2024b), which uses Monte Carlo estimation — for each step in a solution, sample multiple completions from that step onward and compute the fraction that reach the correct answer — to obtain step-level labels without human annotation. PAV (Setlur et al., 2024) improves process rewards via advantage modeling, which adjusts step-level scores based on how much better or worse the step is compared to the expected progress. ReasonFlux-PRM (Zou et al., 2025), TP-GRPO (He et al., 2025f), and CAPO (Xie et al., 2025b) leverage generative PRMs — large language models prompted to evaluate reasoning steps — to provide step-level rewards. SGPO (Chen et al., 2025m) uses a strong judge model to identify the *first incorrect step* and computes advantage values based on the index of that step, penalizing all tokens after the first error.

**Sampling-based step rewards** avoid training a separate PRM entirely by estimating step-level quality through Monte Carlo rollouts from the policy itself. VinePPO (Kazemnejad et al., 2025) improves PPO by replacing the learned critic with Monte Carlo advantage estimation from intermediate states. SPO (Guo et al., 2025c), TreeRL (Hou et al., 2025), and FR3E (Zheng et al., 2025c) use low-probability or high-entropy tokens as automatic division points for step segmentation, then estimate the value of each segment through sampling. To improve sample efficiency, SPO (Guo et al., 2025c), TreeRPO (Yang et al., 2025g), and TreePO (Li et al., 2025t) explore tree-based structures — instead of a single linear chain, they generate branching rollouts from intermediate states, enabling fine-grained process reward computation through backpropagation of outcome rewards through the tree.

**Turn-level rewards.** These are designed for *multi-turn agentic tasks* where the model interacts with an environment (tool, search engine, code interpreter) over multiple rounds. Each turn receives a reward based on its contribution to the eventual outcome. Approaches fall into two categories: *direct per-turn supervision* (explicit rewards for each tool invocation, such as format correctness and action validity) and *derived from outcome-level rewards* (decomposing a final success/failure signal into per-turn contributions through progress attribution, gated reward accumulation, or learned critics).

##### Unsupervised Rewards (Section 3.1.4)

**What they are:** reward signals generated *without any human annotation or external ground truth*, derived entirely from the model's own outputs or from automatic, non-human sources.

**The motivation.** For tasks requiring superhuman expertise — frontier mathematics, novel scientific reasoning, complex engineering — human feedback is slow, expensive, and potentially unreliable (humans may not be able to judge correctness on problems they themselves cannot solve). Unsupervised rewards aim to "eliminate the human annotation bottleneck, enabling reward signal generation at the scale of computation and data, not human labor" (Section 3.1.4).

**Model-Specific Rewards** derive signals from the model's own internal processes. They operate on the assumption that a well-trained model will exhibit certain statistical regularities that correlate with correctness:

- **Rewards from output consistency:** The core hypothesis is that correct answers form dense, consistent clusters among multiple generated outputs, while incorrect answers are scattered. EMPO (Zhang et al., 2025i) operationalizes this via clustering — group generated answers by semantic similarity and assign higher rewards to answers in dense clusters. TTRL (Test-Time Reinforcement Learning, Zuo et al., 2025b) uses majority voting — the reward for an answer is how many other generated answers agree with it. Subsequent methods refine this by improving efficiency (ETTRL, Liu et al., 2025d) or incorporating contrastive agreement to combat reward hacking (Co-Reward, Zhang et al., 2025x).

- **Rewards from internal confidence:** This uses the model's own uncertainty as a proxy for correctness. Signals can be based on cross-attention patterns, negative entropy (lower entropy = higher confidence = potentially more correct), generation probabilities (the model's own assessment of how likely its output is), or gradient-based measures. EM-RL (Agarwal et al., 2025b) and RENT (Prabhudesai et al., 2025) use entropy minimization — the model is rewarded for being confident in its outputs.

- **Rewards from self-generated knowledge:** The model acts as both problem proposer and solver. In *self-rewarding* (Yuan et al., 2024; Wu et al., 2024), the model evaluates its own outputs using its own judgment capabilities. In *self-instruction*, a proposer model generates a curriculum of tasks, and the solver's reward is derived from performance on these self-generated tasks. The proposer is often rewarded for creating tasks of optimal difficulty — too easy and the solver learns nothing, too hard and all rewards are zero.

**Model-Agnostic Rewards** derive signals from external, automated sources that don't require human labeling:

- **Heuristic rewards:** Simple, predefined rules based on output properties. DeepSeek-R1 (Guo et al., 2025a) pioneered the use of format rewards (correct XML tag structure) and length-based heuristics. The risk is that models can game these heuristics — producing superficially correct formats without genuine reasoning.

- **Data-centric rewards:** Deriving signals from the structure of large, unlabeled corpora. RPT (Reinforcement Pre-Training, Dong et al., 2025c) reframes next-token prediction on web-scale text as an RL task, turning existing pretraining data into millions of training examples with automatically computable rewards (did the model predict the actual next token correctly?).

##### Reward Shaping (Section 3.1.5)

**What it is:** the process of modifying or combining reward signals to produce more informative, stable gradients for policy optimization. Rather than designing new rewards from scratch, reward shaping *transforms* existing rewards to improve learning dynamics.

**Rule-based reward shaping** combines multiple reward components using constant or adaptive coefficients. The simplest form combines a rule-based accuracy reward (correct/incorrect, binary) with a reward model score (continuous quality assessment that can distinguish between two correct answers of different quality). A constant coefficient `$\alpha$` balances their contributions:

$$R_{\text{combined}} = R_{\text{rule}} + \alpha \cdot R_{\text{RM}}$$

This is widely employed in open-domain tasks where binary accuracy alone isn't sufficient, and the reward model provides finer-grained quality signals. DeepSeek-R1 (Guo et al., 2025a) combines format rewards, accuracy rewards, and — in later stages — reward model scores. Dynamic reward weighting (Lu et al., 2025f) extends this by adaptively adjusting the coefficients during training based on hypervolume-guided weight adaptation (for multi-objective alignment) or gradient-based optimization.

**Structure-based reward shaping** computes rewards *across a group of candidates* rather than per individual sample. The key insight is that the *relative* quality of responses to the same prompt is often more informative than the *absolute* quality:

- **Group-relative normalization (GRPO):** Given `$G$` responses to the same prompt with raw rewards `$\{R_i\}_{i=1}^{G}$`, the advantage for response `$i$` is computed as:

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^{G})}{\text{std}(\{R_j\}_{j=1}^{G})}$$

This normalizes rewards to zero-mean and unit-variance *within each group*, so the model learns to produce responses that are better than its current average rather than chasing absolute reward magnitudes. This is structurally similar to *leave-one-out* baselines (RLOO, Ahmadian et al., 2024), where the baseline for each response is the mean of all *other* responses in the group.

- **Pass@K-aligned objectives:** Standard RL optimizes Pass@1 (the probability that a single sample is correct), but many applications care about Pass@K (the probability that *at least one* of `$K$` samples is correct). PKPO (Walder and Karkhanis, 2025) performs a joint transformation on the final reward to make the optimization directly equivalent to set-level Pass@K objectives, providing low-variance, unbiased gradient estimates. Pass@K Training (Chen et al., 2025z) directly targets Pass@K in deriving advantages, decomposing set-level targets back into individual sample credit allocation.

---

#### 3.4.3 Policy Optimization Algorithms

**The core tension.** All RL algorithms for LLMs must balance two competing forces: (1) they must *exploit* known high-reward behaviors by increasing the probability of sampled actions that received positive advantages, and (2) they must *explore* sufficiently to discover new, potentially better behaviors. Too much exploitation leads to premature convergence (the model gets stuck in a local optimum of repetitive but mediocre answers); too much exploration leads to instability (the policy drifts randomly and never settles on effective strategies).

**The PPO-style general formulation (Equation 5).** Most modern algorithms for LLM RL are variants of Proximal Policy Optimization (PPO, Schulman et al., 2017b), which addresses the exploration-exploitation tension through a *clipped surrogate objective*:

$$\mathcal{J}(\theta) = \mathbb{E}_{\text{data}} \left[ \frac{1}{Z} \sum_{i=1}^{N} \sum_{t=1}^{T_i} \min \left( w_{i,t}(\theta) \hat{A}_{i,t}, \ \text{clip}(w_{i,t}(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}) \hat{A}_{i,t} \right) \right]$$

where `$w_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | x, y_{i,<t})}$` is the importance sampling ratio — the ratio of the probability of token `$y_{i,t}$` under the current policy to its probability under the *old* policy (the one that generated the rollout data), `$\hat{A}_{i,t}$` is the estimated advantage for that token (either token-wise or sequence-level), `$T_i$` is the number of tokens for sample `$i$`, `$N$` is the number of samples per prompt, `$Z$` is a normalization factor, and `$\epsilon_{\text{low}}, \epsilon_{\text{high}}$` are clipping thresholds.

**What this equation computes:** the clipped surrogate objective that PPO maximizes. For each token in each generated response, it computes the importance ratio `$w_{i,t}$` (how much more likely the current policy is to generate this token compared to when the data was collected), multiplies it by the advantage `$\hat{A}_{i,t}$` (how good this action was), but *clips* `$w_{i,t}$` to stay within `$[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$` of 1.0. If the advantage is positive (the action was good), the objective increases the probability of that token, but the increase is capped at `$1 + \epsilon_{\text{high}}$` times the original probability. If the advantage is negative (the action was bad), the objective decreases the probability, capped at `$1 - \epsilon_{\text{low}}$` times the original probability.

**Why this form:** the clipping prevents the policy from changing too much in a single update. Without clipping, a very positive advantage could cause the model to increase a token's probability by orders of magnitude in one step, which would be destructive — the model would "forget" other useful behaviors and potentially collapse to a degenerate policy. The `$\min$` operation ensures that when the ratio moves outside the clip range, the gradient becomes zero (the objective is flat), preventing further movement in that direction. This is PPO's key innovation over TRPO (Schulman et al., 2015a), which enforced a trust region through a more computationally expensive constrained optimization.

##### Critic-based vs. Critic-free Algorithms

The fundamental algorithmic divide in LLM RL is between methods that use a *separately trained critic model* to estimate token-level values and advantages (PPO-style) and methods that compute advantages directly from sequence-level rewards without a learned value function (GRPO-style).

**Critic-based algorithms (Section 3.2.2)** train a value function `$V_{\phi}(s_t)$` alongside the policy. This critic predicts the expected future return from state `$s_t$`, enabling per-token advantage computation through Generalized Advantage Estimation (GAE):

$$\hat{A}_{\text{GAE}, t} = \sum_{l=t}^{T} (\gamma \lambda)^{l} \delta_{t+l}$$

where `$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$` is the temporal difference error — the difference between the actual reward plus the estimated future value and the current value estimate. The parameter `$\lambda \in [0, 1]$` controls the bias-variance tradeoff: `$\lambda = 0$` gives low-variance but biased one-step estimates; `$\lambda = 1$` gives high-variance but unbiased Monte Carlo estimates.

**Why a critic helps.** The critic provides *dense, token-level* advantage estimates even when the reward is sparse and sequence-level. By learning to predict `$V(s_t)$`, the critic can assign credit to individual tokens based on how they change the model's own estimate of future success. This is especially valuable for long chain-of-thought reasoning where the final correct/incorrect signal arrives only at the very end.

**Why critics are problematic.** Training a critic alongside the policy doubles the computational cost (two models to run forward and backward passes for), introduces additional hyperparameters and instability sources, and — critically — is vulnerable to *reward hacking* if the critic is imperfect. The critic may learn to assign high values to tokens that look superficially like good reasoning but don't actually lead to correct answers, encouraging the policy to produce convincing-looking but ultimately incorrect chains of thought. VCPPO (Value-Calibrated PPO, Yuan et al., 2025f) and VAPO (Yue et al., 2025c) address this by proposing mechanisms for enhancing the robustness of the critic model under noisy reward signals.

**Critic-free algorithms (Section 3.2.3)** eliminate the critic entirely and compute advantages directly from sequence-level rewards. The key insight — validated primarily through DeepSeek-R1's success with GRPO — is that for RLVR tasks with reliable, automatically verifiable rewards, **the outcome reward signal is sufficient for learning complex reasoning, and a critic's additional complexity is unnecessary or even harmful.**

**GRPO (Group Relative Policy Optimization).** The most influential critic-free algorithm, proposed by Shao et al. (2024) and used by DeepSeek-R1 (Guo et al., 2025a). Its objective is:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min \left( w_{i,t}(\theta) \hat{A}_i, \ \text{clip}(w_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) \right]$$

where `$G$` is the group size (number of responses generated per prompt, typically 4–64), `$|y_i|$` is the length of response `$i$`, `$w_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} | x, y_{i,<t})}$` is the token-wise importance ratio, and the advantage `$\hat{A}_i$` is **shared across all tokens in response `$i$`** and computed as:

$$\hat{A}_i = \frac{R(x, y_i) - \text{mean}(\{R(x, y_j)\}_{j=1}^{G})}{\text{std}(\{R(x, y_j)\}_{j=1}^{G})}$$

**What GRPO does, operationally.** For each prompt, the model generates `$G$` complete responses. Each response receives a reward (e.g., 1 for correct answer, 0 for incorrect). The advantages are computed by normalizing rewards within the group — a correct response in a group where most responses are wrong receives a large positive advantage; a correct response in a group where most are correct receives a smaller positive advantage. This normalization means the *same* absolute reward can produce *different* advantages depending on the group context. The policy is then updated using the PPO-style clipped objective, but with all tokens in a response receiving the same advantage signal.

**Why GRPO's design choices matter:**

- **Group-relative normalization** addresses a fundamental issue: with binary rewards (+1 for correct, 0 for incorrect), the advantage for a correct response would always be positive (good) and for an incorrect response would always be negative (bad), regardless of *how* correct or incorrect. This provides no signal about *relative* quality — a barely-correct answer and a brilliantly-correct answer receive the same reward. By normalizing within groups, GRPO ensures that the model is rewarded more for responses that are correct when its peers are wrong, which implicitly captures problem difficulty.

- **Sequence-level advantage** (all tokens in a response share the same advantage) is simpler than token-level advantage estimation but aligns with the outcome-reward paradigm — if the only signal is whether the final answer is correct, then all tokens in a correct trajectory are "good" and all tokens in an incorrect trajectory are "bad." This avoids the complexities and potential biases of a separately trained critic.

- **Critic-free design** eliminates the need for a second model, reducing memory requirements, simplifying the training pipeline, and avoiding the risk of the policy exploiting imperfections in the critic. DeepSeek-R1 explicitly found that "RMs may suffer from reward hacking when scaled to large-scale RL settings" (Guo et al., 2025a), motivating the use of rule-based rewards without learned critics.

**Variants and refinements of GRPO.** Rapid algorithmic innovation has produced numerous modifications to the basic GRPO template, each addressing specific failure modes:

- **DAPO (Yu et al., 2025d)** introduces "Clip-Higher" — the upper clipping bound is set higher than the lower one (e.g., `$\epsilon_{\text{low}} = 0.2$`, `$\epsilon_{\text{high}} = 0.28$`). This allows the probabilities of *unlikely but potentially useful tokens* to increase more freely than they can decrease, which encourages exploration. The intuition: when the model discovers a new, useful token, it should be allowed to increase its probability substantially; but when it determines a token is harmful, it shouldn't collapse its probability to zero immediately.

- **Dr. GRPO (Liu et al., 2025h)** identifies a key deviation in GRPO where "the longer it's wrong, the more wrong it gets" — incorrect long responses receive more negative updates than incorrect short responses because the same negative advantage is applied across more tokens. The authors introduce algorithmic modifications to improve token efficiency, ensuring that the *total* gradient magnitude for a response is normalized regardless of length.

- **CISPO (Chen et al., 2025a)** introduces clipped importance-sampling-weight policy optimization, modifying how the importance ratio is computed and clipped for more stable training with MoE architectures.

- **GSPO (Zheng et al., 2025a)** shifts the importance ratio and clipping operations from the token level to the *sequence* level. Instead of clipping each token's importance ratio individually, GSPO computes a single sequence-level importance weight and clips that. This provides more stable training, particularly for MoE models where token-level clipping can interact poorly with expert routing.

- **LitePPO (Liu et al., 2025a)** uses group-level mean and batch-level standard deviation for normalization, rather than per-group statistics. This provides more stable advantage estimates when group sizes are small or when reward variance differs substantially across prompts.

**REINFORCE-family critic-free algorithms.** Prior to GRPO's dominance, simpler critic-free methods were explored:

- **Vanilla REINFORCE (Williams, 1992):** `$\mathcal{J}(\theta) = \mathbb{E}_{x, y \sim \pi_{\text{old}}} [R(x, y) \nabla_{\theta} \log \pi_{\theta}(y|x)]$`. Treats the entire sequence as a single action (bandit formulation). Suffers from severe instability due to high variance — the gradient estimate for a single trajectory can be very noisy.

- **ReMax (Li et al., 2023c):** Introduces variance reduction by using a *greedy baseline* — generate one response greedily (temperature 0) and use its reward as a baseline for the sampled response's reward. This provides a simple, compute-efficient way to reduce variance without a learned critic.

- **RLOO (Ahmadian et al., 2024):** Uses a *leave-one-out* baseline — the advantage for response `$i$` is its reward minus the mean reward of all *other* responses in the group. This provides an unbiased baseline (unlike the greedy baseline in ReMax) with similar computational efficiency.

- **REINFORCE++ (Hu, 2025):** Adapts techniques from PPO and GRPO — clipping, global advantage normalization — to the basic REINFORCE framework, providing a middle ground between the simplicity of REINFORCE and the stability of GRPO.

##### Importance Sampling for Off-Policy Correction

**The problem.** In online RL, the model generates rollouts, computes advantages, and updates its parameters — all with the *current* policy. However, due to computational constraints in large-scale training, there is typically a delay between generation and update: the model that generated the rollouts (`$\pi_{\theta_{\text{old}}}$`) may be several gradient steps behind the model being updated (`$\pi_{\theta}$`). This creates an *off-policy* situation — the data was generated by a different policy than the one being optimized.

**The solution: importance sampling.** The policy gradient should, in principle, use the current policy's probabilities, not the old policy's. The importance ratio `$w_{i,t} = \pi_{\theta}(y_t|x, y_{<t}) / \pi_{\theta_{\text{old}}}(y_t|x, y_{<t})$` corrects for this distribution shift. If the current policy assigns higher probability to a token than the old policy did, `$w > 1$`, and the gradient step is amplified; if lower probability, `$w < 1$`, and the step is attenuated.

**The token-level vs. sequence-level debate.** Most GRPO variants use *token-level* importance sampling — each token gets its own ratio `$w_{i,t}$`. However, this is technically a biased estimator because the true importance ratio for a state-action pair should be with respect to the full joint distribution, not the marginal per-token distribution. GSPO (Zheng et al., 2025a) proposes *sequence-level* importance sampling, computing a single ratio for the entire response. GMPO (Zhao et al., 2025g) introduces geometric averaging — rather than arithmetic averaging of token-level ratios, it uses the geometric mean, which is more robust to extreme individual token ratios.

##### Off-policy Optimization (Section 3.2.4)

**Beyond importance sampling.** Some recent methods move beyond the PPO/GRPO paradigm of correcting for mild off-policy-ness and instead embrace *fully off-policy* or *mixed-policy* optimization:

- **SPO (Soft Policy Optimization, Cohen et al., 2025):** Enables stable online, off-policy RL by using a soft policy update mechanism that doesn't require on-policy data. This opens the door to training from historical or asynchronous data without bias correction.

- **Experience replay methods:** Several works (Dou et al., 2025; Wang et al., 2025b; Chen et al., 2024c) use replay buffers — storing previously generated trajectories and mixing them with fresh rollouts. This improves sample efficiency by reusing past experiences, but requires off-policy corrections.

- **Mixed-policy methods (SFT + RL):** A growing trend combines supervised fine-tuning on expert demonstrations with RL optimization within the same training loop. UFT (Liu et al., 2025a) unifies SFT and RL into a single-stage target, theoretically overcoming the bottleneck of long-horizon sample complexity. SRFT (Fu et al., 2025c) proposes a joint single-stage integration of demonstration imitation (SFT) and strategy improvement (RL) using entropy perception weights. BREAD (Zhang et al., 2025p) generates branched rollouts from expert anchors — starting from high-quality SFT-generated prefixes and then branching with RL, combining the stability of supervised learning with the exploration of RL.

##### Regularization Objectives (Section 3.2.5)

**Why regularization is necessary.** Unconstrained RL optimization on LLMs tends to produce several failure modes: (1) *reward hacking* — the model finds ways to get high rewards without genuine improvement (e.g., producing verbose but vacuous reasoning that happens to include the correct answer), (2) *entropy collapse* — the model's output distribution becomes degenerate, producing nearly identical responses regardless of the prompt, and (3) *catastrophic forgetting* — the model loses general language capabilities as it overfits to the narrow RL task.

**KL Regularization.** The most common regularization adds a penalty for the policy diverging too far from a reference policy `$\pi_{\text{ref}}$`:

$$\mathcal{L}_{\text{KL}} = \beta \cdot \frac{1}{|y|} \sum_{t=1}^{|y|} \text{KL}(\pi_{\theta}(\cdot | y_t) \ \Vert \ \pi_{\text{ref}}(\cdot | y_t))$$

where `$\beta$` controls the strength of the penalty.

**The controversy over KL regularization.** In the RLHF alignment paradigm, KL regularization toward the initial SFT model was considered *essential* to prevent the model from "reward hacking" the learned reward model by producing gibberish that scored highly. However, in RLVR for reasoning, the situation is different:

- **Pro-KL arguments:** KL regularization prevents catastrophic forgetting of general capabilities, maintains output quality and fluency, and provides a "trust region" that stabilizes training. Archer (Wang et al., 2025i) applies stronger KL regularization to low-entropy tokens (already well-learned) and weaker regularization to high-entropy tokens (where exploration is needed), providing a nuanced approach.

- **Anti-KL arguments:** Many recent works (An et al., 2025; Chen et al., 2025s; Cui et al., 2025a; He et al., 2025d; Yu et al., 2025d) advocate *removing KL regularization entirely* for RLVR. The argument is that reasoning requires the policy to diverge significantly from its initialization to discover new chain-of-thought structures, and KL constraints prevent this exploration. The rule-based reward signal is reliable enough that reward hacking through gibberish isn't a concern (unlike with learned reward models). Additionally, removing KL reduces memory footprint (no need to store reference model parameters) and simplifies implementation.

- **KL toward old policy (not reference):** An alternative is to use KL divergence toward the *previous* policy (`$\pi_{\text{old}}$`) rather than toward the *initial* policy (`$\pi_{\text{ref}}$`). This serves as a trust-region constraint — limiting how much the policy can change in a single update — without preventing long-term drift. K1.5 (Team, 2025d) uses mirror descent with an adaptation of this approach, and several works (Cui et al., 2025b; Lyu et al., 2025) explore normalized KL forms.

**Entropy Regularization.** Explicit entropy bonuses encourage the policy to maintain a diverse output distribution:

$$\mathcal{L}_{\text{ent}} = -\alpha \cdot \frac{1}{|y|} \sum_{t=1}^{|y|} H[\pi_{\theta}(\cdot | y_t)]$$

where `$H[\pi_{\theta}(\cdot|y_t)]$` is the entropy of the policy at token position `$t$`, and `$\alpha$` controls the strength of the bonus. Lower entropy means the model is very certain about what token to generate (potentially converging to a deterministic, uninteresting policy); higher entropy means it maintains uncertainty and diversity.

**The entropy collapse phenomenon.** Multiple studies (Cheng et al., 2025a; Cui et al., 2025b; Yu et al., 2025d) have observed that without intervention, RL training for reasoning tends to exhibit *entropy collapse* — the policy's output distribution becomes increasingly peaked, losing the ability to explore diverse reasoning strategies. This is a failure mode: once entropy collapses, the model can no longer discover new reasoning patterns because it rarely samples anything outside its narrow high-probability region.

**Methods to combat entropy collapse:**
- **Explicit entropy regularization in the loss** (Shrivastava et al., 2025; Wu et al., 2025e) — directly adding the entropy term.
- **Dynamic entropy coefficient adjustment** (He et al., 2025d) — increasing `$\alpha$` when entropy drops below a threshold.
- **Clip-Higher** (Yu et al., 2025d; DAPO) — the asymmetric clipping discussed earlier, which allows increases in probability more freely than decreases.
- **High-entropy token training** (Wang et al., 2025n) — only training on the 20% highest-entropy tokens in each response, ensuring the model focuses its learning on uncertain positions.
- **Entropy-aware advantage** (Cheng et al., 2025a; Chen et al., 2025j) — incorporating entropy directly into the advantage computation, so that tokens with high entropy receive boosted advantages.
- **Clip-Cov / KL-Cov** (Cui et al., 2025b) — a theoretical analysis identifying the covariance between an action's output probability and its advantage as the "entropy driver." By selectively constraining tokens with exceptionally high covariance (those driving entropy change), these methods regulate entropy without blanket penalties.

**Length Penalty.** Long chain-of-thought reasoning improves accuracy but incurs higher inference costs. Recent works seek to balance reasoning depth with efficiency:

- **Adaptive length penalties** (Liu et al., 2025b; Xiang et al., 2025) — applying length penalties that vary based on estimated problem difficulty. Easy problems receive stronger penalties to discourage overthinking; hard problems receive weaker or no penalties.
- **L1 length control** (Aggarwal and Welleck, 2025) — training the model to adhere to user-specified length constraints by adding a penalty proportional to deviation from the target length.
- **O1-pruner** (Luo et al., 2025a) — using RL with an accuracy-preservation constraint to reduce reasoning length while maintaining correctness.
- **Relative-length regularization** (Yuan et al., 2025a) — penalizing responses whose length exceeds a baseline derived from the model's own typical behavior.

---

#### 3.4.4 Sampling Strategy (Section 3.3)

**Why sampling matters.** Unlike supervised fine-tuning, which uses a fixed dataset, RL training depends on *actively generated rollouts* — the model must produce its own training data in each iteration. What the model samples, and how those samples are structured, directly determines what behaviors it can learn. Poor sampling strategies can lead to: (1) training on uninformative prompts (too easy or too hard), (2) insufficient exploration of diverse reasoning strategies, (3) wasted computation on redundant rollouts, or (4) instability from degenerate generations.

##### Dynamic Sampling (Section 3.3.1)

**Efficiency-oriented sampling** filters prompts and allocates compute based on online learning signals to concentrate resources on the most informative examples:

- **Difficulty-based filtering:** DAPO (Yu et al., 2025d) over-samples prompts and then filters out those whose rollouts are *saturated* (all responses correct) or *degenerate* (all responses wrong). It repeatedly samples until each mini-batch contains prompts with non-zero advantage — that is, prompts where some but not all generated responses are correct. These medium-difficulty prompts provide the most useful gradient signal because they have both positive and negative examples.

- **Prioritized replay:** K1.5 (Team, 2025d) uses a `$p(i) \propto (1 - s_i)$` rule, where `$s_i$` is the success rate for prompt `$i$`. Prompts with lower success rates receive more sampling budget, focusing computation on under-mastered items.

- **Category-level curriculum:** Some works (Chen et al., 2025p) use non-stationary multi-armed bandits to select which categories of problems (e.g., algebra vs. geometry) to sample from, dynamically adjusting based on recent performance.

- **Easy-to-hard schedules (E2H):** Parashar et al. (2025) follow curriculum learning principles with convergence guarantees for small models — start with easier problems and gradually introduce harder ones as the model improves.

- **POLARIS (An et al., 2025):** A comprehensive system that uses offline difficulty estimation to construct "mirror-J" distributions — for each model scale, it estimates which items have been mastered and continuously removes them from the training distribution, ensuring the model always trains on problems at the edge of its capability.

**Exploration-oriented sampling** prioritizes diversity and coverage of the solution space:

- **Entropy-guided rollout (ARPO):** Dong et al. (2025b) use the model's entropy (uncertainty) at each generation step to guide sampling — when the model is uncertain, it is more likely to call external tools or explore alternative reasoning paths.

- **Attention-guided branching (AttnRL):** Liu et al. (2025a) find that steps with high attention scores are correlated with reasoning behaviors and *branch* at these steps — generating multiple continuations from the same prefix — to explore alternative reasoning strategies at key decision points.

- **Rubric-scaffolded exploration (RuscaRL):** Zhou et al. (2025f) provide the policy with different rubrics (evaluation criteria) during rollout generation, encouraging it to explore the space of possible solutions from multiple perspectives.

##### Structured Sampling (Section 3.3.1)

**Search-driven tree rollouts** organize generation as a tree rather than a single chain, enabling node-level rewards and more efficient exploration:

- **TreeRL (Hou et al., 2025):** An on-policy tree search framework that outperforms traditional Chain-of-Thought RL (ChainRL) while reducing computational overhead through efficient search strategies. The tree structure allows the model to explore multiple reasoning branches and receive process-level rewards at each node.

- **Monte Carlo Tree Search (MCTS) integration:** Multiple works (Yang et al., 2025g; Wu et al., 2025c) integrate MCTS into the RL training loop. During the expansion phase of MCTS, the model generates multiple candidate continuations; during backpropagation, outcome rewards are propagated up the tree to provide fine-grained process signals.

**Shared-prefix and segment-wise schemes** improve generation efficiency by reusing computation:

- **Prefix reuse:** Rather than generating each rollout independently from scratch, methods like SPO (Guo et al., 2025c), TreeRPO (Yang et al., 2025g), and TreeRL (Hou et al., 2025) start multiple rollouts from previously generated prefixes. This allows the Key-Value (KV) cache for the shared prefix to be computed once and reused, significantly reducing the computational cost of generating many rollouts.

- **TreePO (Li et al., 2025t):** Implements a segment-wise tree sampling algorithm that alleviates the KV cache burden. Instead of maintaining separate KV caches for every node in a full tree, TreePO segments the generation process and shares caches across segments, reducing GPU memory requirements and improving sampling efficiency.

##### Sampling Hyper-parameters (Section 3.3.2)

**Exploration-exploitation dynamics.** The primary levers are temperature, entropy targets, and PPO clipping parameters:

- **Temperature scheduling:** Common practice involves staged temperature changes. POLARIS (An et al., 2025) uses a schedule like 1.40 → 1.45 → 1.50 for a 4B model (or 0.7 → 1.0 → 1.1 for 7B), gradually increasing temperature as training progresses to maintain exploration as the policy becomes more specialized. E3-RL4LLMs (Liao et al., 2025b) uses a scheduler to dynamically adjust temperature to maintain a stable entropy level.

- **Entropy targets:** A prescriptive recommendation from several works (Liu et al., 2025i; Wu et al., 2025e) is to tune the training temperature to keep the post-scaling entropy around a target of approximately 0.3, which is found to empirically balance exploration and exploitation.

**Length budgeting and sequence management.** Managing the growing length of generated responses is a central operational challenge:

- **Staged context lengthening:** DeepScaleR (Luo et al., 2025c) popularized the approach of starting RL with a short context window (e.g., 8K tokens) and progressively increasing to 16K, 24K, or 32K. The initial short-context stage forces the model to learn concise, token-efficient reasoning patterns before being given more space.

- **Overlength handling strategies vary:** Some works apply soft penalties as responses approach the maximum length (Yu et al., 2025d), others use tunable penalties directly in the reward function (Arora and Zanette, 2025). A more nuanced approach (Liu et al., 2025i; Wu et al., 2025e) filters (masks the loss of) overlong samples when the length budget is short (8K–16K) but penalizes them when the budget is large (32K), as filtering becomes too restrictive at long contexts.

- **Inference-time length extrapolation:** Some works (An et al., 2025) apply techniques like YaRN at inference time, allowing a model trained on shorter sequences to generalize to longer ones without retraining — a practical approach that avoids the computational cost of training on very long sequences.

## 4. Key Insights and Innovations

### Innovation 1: RLVR Represents a New Scaling Axis Orthogonal to Pretraining — With Sharp, Verifier-Defined Boundaries

The paper's most significant conceptual contribution is not any single algorithm, but rather the **framing of Reinforcement Learning with Verifiable Rewards (RLVR) as a fundamentally distinct scaling paradigm** for language model capabilities — one that is orthogonal to the pretraining scaling laws studied by Kaplan et al. (2020) and Hoffmann et al. (2022). This reframing is what gives the entire survey its intellectual coherence.

**What the field lacked before this framing.** Prior to the RLVR era (roughly before DeepSeek-R1 in January 2025), RL's role in the LLM pipeline was conceptually narrow: it was a *post-training alignment step* — RLHF or DPO — whose purpose was to shape model behavior toward human preferences (helpfulness, harmlessness, honesty). The dominant mental model was: (1) pretraining imbues the model with knowledge and capabilities, (2) supervised fine-tuning teaches it to follow instructions, (3) RLHF aligns it with human values. In this framework, RL was a finishing step, not a capability-building step. Scaling laws research focused almost exclusively on pretraining compute allocation (model size vs. data quantity).

**The conceptual shift the survey documents and systematizes.** RLVR inverts this relationship. Rather than using RL to *constrain* model behavior toward human preferences, RLVR uses RL to *expand* model behavior toward autonomous reasoning — self-verification, backtracking, multi-step planning, tool use. The survey captures this shift through its central organizing metaphor (Figure 2): the progression from RLHF/DPO (alignment) to RLVR (verifiable reasoning) to "Open-ended RL" (future). This is not merely a different application of existing RL techniques; it represents a different *role* for RL in the model development lifecycle. RL is no longer downstream of capability acquisition — it *is* capability acquisition.

**The Verifier's Law as a diagnostic boundary concept.** The survey articulates what it calls "Verifier's Law" — the observation, attributed to Jason Wei and discussed extensively in Section 3.1.1, that "the ease of training AI systems to perform a task is proportional to the degree to which the task is verifiable." This is more than a practical heuristic; it is a *theoretical boundary condition* that explains both RLVR's successes and its current limitations. Tasks that satisfy the verifiability criteria — clear ground truth, rapid automated verification, scalable evaluation — are precisely those where RLVR has achieved dramatic gains (mathematics, competitive programming). Tasks that lack these properties (open-ended writing, subjective evaluation, creative tasks) remain challenging.

This framing is intellectually significant because it converts a scattered set of empirical observations into a predictive principle. It tells researchers *where* to expect RLVR to succeed (domains with verifiable correctness signals) and *where* to expect it to struggle (domains requiring subjective judgment). It also points toward the research agenda that the survey's later sections document: the quest for generative rewards (Section 3.1.2), dense rewards (Section 3.1.3), and unsupervised rewards (Section 3.1.4) can all be understood as attempts to *extend* the verifiability boundary — to create proxy signals in domains that lack natural ground truth.

**Evidence anchoring the significance.** The survey's comprehensive catalogue of frontier models (Table 1, Figure 4) provides empirical weight to this framing. The explosion of open-source reasoning models following DeepSeek-R1 — QwQ, Skywork-OR1, Qwen3, Phi-4 Reasoning, and dozens more — all trained with GRPO or GRPO-variants and verifiable rewards — demonstrates that the RLVR paradigm is not a proprietary trick but a reproducible, transferable methodology. The fact that models from 0.5B to 671B parameters, from dense and MoE architectures, across multiple organizations, all show substantial reasoning gains from RLVR suggests a *robust phenomenon*, not a fragile artifact of specific training configurations.

---

### Innovation 2: The Systematic Decomposition of the RL-for-LLMs Pipeline into Three Interdependent, Co-Designed Components

The survey makes a structural contribution that is easy to overlook but fundamental to how the field should think about RL for reasoning: it decomposes the training pipeline into **three co-designed components — reward design, policy optimization, and sampling strategy** — and argues (implicitly, through its organizational structure) that these components cannot be optimized independently.

**Why this decomposition is non-obvious.** In supervised learning, the training pipeline has a clean separation of concerns: the dataset defines what to learn, the loss function defines how to measure error, and the optimizer (SGD, Adam) is largely independent of both. One can swap datasets, change loss functions, or upgrade optimizers without fundamentally rethinking the other components. The survey's taxonomy demonstrates that RL for LLMs violates this clean separation. The choice of reward function (e.g., outcome vs. process, verifiable vs. generative) directly constrains which policy optimization algorithms are viable (critic-free GRPO exploits the reliability of outcome rewards; critic-based PPO is needed when rewards are noisier). The choice of sampling strategy (e.g., dynamic difficulty filtering, tree-structured rollouts) directly determines what the advantage estimator sees and therefore what the policy can learn.

**The co-design principle in action.** Several specific interactions illustrate this interdependence:

- **Reward density and algorithm choice interact fundamentally.** When only outcome rewards are available (e.g., binary correct/incorrect for math), the advantage signal is sparse and identical for all tokens in a response. This makes critic-based methods that estimate per-token values (PPO with GAE) potentially unreliable — the critic must impute per-token importance from a single endpoint signal, which can lead to reward hacking. GRPO's design — sequence-level advantages with group-relative normalization — is specifically well-suited to this reward structure: it doesn't pretend to know which tokens were important and instead credits the entire trajectory uniformly. Conversely, when dense process rewards are available (from a trained PRM or Monte Carlo estimation), token-level advantage estimation becomes more reliable, and critic-based methods may regain their advantage.

- **Sampling strategy and reward design co-determine exploration.** Dynamic sampling methods like DAPO's difficulty filtering (drop prompts where all responses are correct or all are wrong) only work because the *reward signal is reliable* — the system knows with certainty whether a response was correct. If the reward were noisy or learned, filtering based on reward saturation could amplify reward model biases. Similarly, tree-structured rollouts (TreeRL, TreePO) are only useful if there are *process-level* rewards to assign to intermediate nodes; with outcome-only rewards, the tree structure provides no additional signal over independent parallel sampling.

- **Regularization strategy depends on reward reliability.** The survey documents a sharp divide in the field over KL regularization (Section 3.2.5). In RLHF, where rewards come from a learned preference model that can be gamed, KL toward the reference policy is considered essential to prevent reward hacking. In RLVR, where rewards are rule-based and resistant to manipulation, many leading works (DAPO, PRIME, Skywork-OR1, SimpleRL) remove KL regularization entirely, arguing it unnecessarily constrains exploration. This is not a minor implementation detail — it reflects a *qualitative difference in the trustworthiness of the reward signal* that fundamentally changes what regularization is needed.

**Significance beyond this survey.** This co-design framing has practical implications for researchers entering the field. It suggests that improving any single component in isolation — e.g., designing a more sophisticated reward function without considering how it interacts with the advantage estimation and sampling strategy — may yield misleading conclusions. A reward function that appears superior under one policy optimization algorithm may underperform under another. A sampling strategy that improves exploration with one reward type may cause instability with another. The survey's taxonomy provides the conceptual vocabulary for reasoning about these interactions, even if it does not resolve all of them empirically.

---

### Innovation 3: Crystallizing Four Foundational Debates That Define the Field's Current State and Future Trajectory

Section 4 of the survey does something unusual for a survey paper: rather than simply cataloguing work, it identifies **four unresolved, actively contested questions** that cut across the entire RL-for-LLMs landscape and argues that these debates are not peripheral disagreements but *constitutive of the field's identity*.

**The four debates (Section 4.1–4.4):**

1. **RL's Role: Sharpening or Discovery?** Does RL merely amplify and refine reasoning patterns already present in the pretrained model (the "Sharpening" view, supported by evidence that Pass@K often doesn't improve, only Pass@1), or can RL generate genuinely new capabilities through extended training (the "Discovery" view, supported by ProRL's finding that prolonged RL expands reasoning frontiers)?

2. **RL vs. SFT: Generalize or Memorize?** Is RL fundamentally better at generalizing to out-of-distribution problems than supervised fine-tuning (Chu et al., 2025a: "SFT memorizes, RL generalizes"), or is this an oversimplification that depends critically on data distribution, model prior, and problem difficulty (Jin et al., 2025d: RL cannot recover from severe SFT-induced overfitting)?

3. **Model Prior: Weak vs. Strong?** Does RL work best starting from base models (as DeepSeek-R1-Zero demonstrated) or from already-instruct-tuned models (as many replications prefer)? The survey documents the striking finding that model *family* matters enormously — Qwen models show significant gains even under *spurious* (random) rewards, while Llama and OLMo models often do not, suggesting differences in pretraining that make some models "RL-friendly" and others not.

4. **Training Recipes: Tricks or Traps?** The proliferation of algorithmic variants (GRPO, DAPO, Dr. GRPO, GSPO, CISPO, etc.) raises a disturbing question: how many of the claimed improvements are robust, and how many are artifacts of specific experimental configurations, unreported hyperparameters, or the particular benchmark/test set used? The survey explicitly flags the field's "most pressing challenge" as "inconsistent experimental settings, incomplete reporting, and conflicting conclusions."

**Why crystallizing these debates is an intellectual contribution.** Prior to this survey, these tensions existed in the literature but were scattered across individual papers, each making one-sided claims. The Sharpening vs. Discovery debate, for instance, played out across a dozen papers (DeepSeek-R1, Limit-of-RLVR, ProRL, RENT, Spurious Rewards) without any single document articulating the terms of the disagreement. The survey's contribution is not to resolve these debates — it explicitly doesn't — but to *name them, frame them precisely, present evidence on both sides, and establish them as the central conceptual questions the field must address*. This transforms them from a confusing morass of contradictory findings into a structured research agenda.

**The debates are not merely academic — they have direct practical implications:**

- If RL's role is primarily sharpening (Pass@1 improvement without Pass@K improvement), then the optimal training strategy is to use RL as a *finishing step* on already-capable models, and the primary resource investment should be in better verifiers rather than longer training. If RL can genuinely discover new capabilities (Pass@K improvement), then the optimal strategy is to *extend training duration*, invest in exploration mechanisms, and consider RL as a substitute for some pretraining compute.

- If model prior determines RL responsiveness (Qwen vs. Llama), then the choice of base model is a first-order decision that dominates algorithmic choices. Organizations using Llama-based models may need to invest in mid-training (annealing on reasoning data, as the survey documents in Section 4.3) before RL is effective — a finding with direct budget implications.

- If many algorithmic "improvements" are actually traps (configuration-specific artifacts), then the field urgently needs standardized evaluation protocols, ablation studies that isolate individual components, and shared infrastructure (which the survey's Section 5.3 on RL infrastructure directly addresses).

**Evidence anchoring the intellectual stakes.** The survey does not just assert these debates — it provides specific citations and empirical anchors. The Sharpening/Discovery debate is anchored to Yue et al. (2025b) (Pass@K evaluations showing RL underperforms base models at large-K) versus Liu et al. (2025) (ProRL showing prolonged RL improves both Pass@1 and Pass@K). The model prior debate is anchored to Shao et al. (2025) (Spurious Rewards showing Qwen models improve under random rewards while Llama/OLMo models do not). The tricks/traps debate is anchored to Liu et al. (2025a) (a unified evaluation framework that demonstrates "a minimalist combination of methods can outperform GRPO and DAPO across multiple configurations" — implying that much of the apparent complexity is unnecessary).

---

### Innovation 4: The Training Resource Ecosystem as a First-Class Object of Study

The survey devotes an entire major section (Section 5, spanning subsections 5.1–5.3) to **training resources** — static corpora, dynamic environments, and RL infrastructure — treating them not as background implementation details but as a *primary object of systematic analysis*. This is unusual for a survey on algorithms and methods, and it reflects a genuine insight about the state of the field: **the availability, quality, and standardization of training resources is now a binding constraint on progress, and understanding this ecosystem is essential for both research and deployment.**

**Why this matters now.** In the early RLHF era, RL training for LLMs was done by a handful of well-resourced labs using proprietary infrastructure. The RLVR era is fundamentally different: it is characterized by a *Cambrian explosion* of open-source frameworks, datasets, environments, and training recipes. Table 6 alone compares 11 infrastructure frameworks (TRL, OpenRLHF, veRL, AReaL, NeMo-RL, ROLL, slime, RLInf, and several secondary frameworks) across dimensions like inference engine support, training backend support, async capabilities, and multi-agent/multimodal support. Table 4 catalogues over 40 static datasets spanning math, code, STEM, and agent domains, each with different construction methods (annotation, distillation, merging), sample counts (from 800 for LIMO to 5.5M for OpenMathReasoning), and format conventions (Q-A, Q-C-A with chain-of-thought traces, verifier signals).

**The ecosystem analysis reveals structural patterns.** By systematically cataloguing these resources, the survey reveals several non-obvious patterns:

- **Data construction is shifting from "scale-first" to "quality-and-verifiability-first."** Early RL datasets emphasized size (millions of examples). The survey's taxonomy shows an accelerating trend toward smaller, more carefully curated datasets with explicit verifiable rewards and process-level labels — LIMO with only 800 examples, LIMR with 1,390, DAPO with 17K. This reflects the growing recognition that RLVR's sample efficiency is limited primarily by reward signal quality, not dataset quantity.

- **Dynamic environments represent a paradigm shift from static to interactive training.** Table 5 categorizes dynamic environments into rule-based (e.g., Reasoning Gym with 104 tasks, AutoLogi with controllable difficulty), code-based (e.g., R2E-Gym for software engineering, MLE-Dojo for AutoML), game-based (e.g., KORGym with 51 games, PuzzleJAX with ~900 games), model-based (e.g., TextArena with adversarial text games), and ensemble-based (e.g., InternBootcamp with 1060 tasks). This shift from "training on fixed answers" to "training through environment interaction" mirrors the transition in classical RL from supervised learning on static datasets to online interaction. The survey's categorization makes this parallel explicit and highlights that LLM RL is converging toward the interactive paradigm that has driven success in classical RL domains like games and robotics.

- **Infrastructure frameworks face a fundamental tension between flexibility and performance.** The survey's Table 6 comparison reveals that no single framework supports all desired features — some excel at asynchronous training (AReaL, veRL with agentic extensions), some at multi-agent support (MARTI, Agent-Lightning), some at specific training backends (Megatron vs. FSDP vs. DeepSpeed), and some at specific inference engines (SGLang vs. vLLM). The survey's taxonomy helps practitioners navigate this tradeoff space by making the dimensions of comparison explicit.

**The resource taxonomy as a coordination mechanism.** Beyond practical utility, the taxonomy serves a coordination function for the research community. By establishing a shared vocabulary and categorization scheme for training resources, the survey enables: (1) apples-to-apples comparisons across methods that use different resources (critical for the tricks-vs-traps problem), (2) identification of resource gaps (which domains lack high-quality verifiable training data? which infrastructure features are missing from all frameworks?), and (3) standardization efforts (if all math RL datasets converged on a common format, cross-method comparison would become dramatically easier). The survey's detailed tables are not just summaries — they are *infrastructure for building infrastructure*, providing the scaffolding on which more systematic benchmarking and resource development can be built.

**Evidence of significance.** The fact that Section 5 on training resources is placed *before* Section 6 on applications — that is, the survey treats understanding the resource landscape as *prerequisite* to understanding application-level results — is itself a methodological argument. It suggests that the field's fragmentation (conflicting algorithmic claims, irreproducible results) is partly caused by the inconsistent and under-documented resource layer, and that progress requires making this layer explicit and systematic.

## 5. Experimental Analysis

This section evaluates whether the survey's central organizational claims—that RLVR constitutes a distinct scaling paradigm, that reward design, policy optimization, and sampling strategies form interdependent co-designed components, and that the field is defined by unresolved foundational debates—are empirically anchored in the evidence the paper presents. Critically, this is a *survey paper*: it does not run its own experiments. The "experimental analysis" therefore examines the *evidential basis* for the survey's claims as drawn from the primary literature it synthesizes. The key questions are: (1) how were the primary studies evaluated and compared, (2) what quantitative patterns emerge from aggregating results across studies, and (3) does the evidence actually support the survey's framing, or does the framing overclaim relative to the heterogeneous, sometimes contradictory, empirical record?

### Evaluation Methodology

**Dataset.** The survey does not introduce a new dataset or benchmark. Instead, it synthesizes results reported across the primary literature on RL for LLM reasoning, primarily from papers released since late 2024. The scope encompasses reported performance on standard reasoning benchmarks including MATH (mathematics), AIME (competition mathematics), HumanEval and MBPP (code generation), SWE-bench (software engineering), and various agentic and multimodal benchmarks. No meta-analysis or statistical aggregation across studies is performed. The survey explicitly catalogs *training* datasets (Table 4)—a taxonomy of static corpora used for RL training, including DAPO (17K math QA pairs), Big-MATH (47K annotated problems), OpenMathReasoning (5.5M distilled samples), KodCode (268K code synthesis examples), NaturalReasoning (2.15M STEM questions), and Search-R1 (221K search agent trajectories)—but reports no cross-dataset performance comparisons.

**Base model(s).** The survey does not commit to a single model family for its claims. The frontier models catalogued in Table 1 span: DeepSeek-R1 (671B MoE, GRPO-trained), QwQ-32B (32B dense, Alibaba Qwen), Skywork-OR1 (7B/32B dense, GRPO-trained), Qwen3 (0.6B-235B, MoE and dense variants), Phi-4 Reasoning (14B dense, Microsoft, GRPO-trained), Llama-Nemotron-Ultra (253B dense, NVIDIA, GRPO-trained), Minimax-M1 (456B hybrid MoE, CISPO-trained), and over 20 additional models. The survey explicitly notes in Section 4.3 that model family choice is a first-order determinant of RL effectiveness: Qwen-family models "register significant gains even under random or spurious reward signals, whereas Llama and OLMo models often do not" (citing Shao et al., 2025: Spurious Rewards). This heterogeneity means that claims about RLVR's effectiveness are, by the survey's own evidence, contingent on base model selection—a point we return to in the Critical Assessment.

**Metrics.** The survey's evaluative framework for comparing methods relies on several metrics, though it does not prescribe a standardized evaluation protocol:

- **Pass@1 accuracy:** The probability that a single sampled response is correct. This is the dominant metric in RLVR research because RL directly optimizes the model's generation distribution. Improvements in Pass@1 are interpreted in the Sharpening vs. Discovery debate (Section 4.1) as evidence for the Sharpening view—RL concentrating probability mass on already-available correct solutions.
- **Pass@K accuracy:** The probability that at least one of K independently sampled responses is correct. This measures the model's *exploration breadth*—whether correct solutions exist in the model's output distribution even if they're not the single most likely output. The survey cites Limit-of-RLVR (Yue et al., 2025b) for the finding that "RL enhances Pass@1 performance, yet tends to underperform relative to base models when sampling broadly at large-K Pass@K"—central evidence for the Sharpening view.
- **Benchmark accuracy:** Standard task-specific metrics: exact match accuracy on MATH and AIME, pass@1 on HumanEval and MBPP (unit test pass rate), resolved rate on SWE-bench (proportion of GitHub issues successfully fixed). The survey reports these as contextual evidence for RLVR's capabilities but does not aggregate them.
- **Response length:** Chain-of-thought length (in tokens) is frequently reported as a secondary metric because RLVR training typically produces longer reasoning traces. The survey discusses this under length penalties (Section 3.2.5) and the overthinking problem (Section 7.4), noting that "RLVR training typically produces longer reasoning traces" and that balancing reasoning depth with efficiency is an active research challenge.

**Baselines.** The survey's comparative framework is implicit rather than explicit—it does not define standardized baselines against which all methods should be evaluated. Instead, the baselines vary by primary study and are discussed within the context of specific debates:

- **Base model (no RL):** For RLVR studies, the standard baseline is the pretrained or SFT-ed model before RL training. DeepSeek-R1's key demonstration was that applying GRPO with rule-based rewards to a base model (R1-Zero) or briefly SFT-ed model (R1) yields dramatic reasoning improvements over the same model without RL.
- **Best-of-N / majority voting:** For reasoning tasks, naive sampling baselines—generate N independent responses and select via majority vote or best-of-N with a verifier—are standard. The survey notes that RLVR-trained models substantially outperform these baselines at the same generation budget.
- **DPO / RLHF:** The survey positions RLVR explicitly against the prior paradigm of alignment-focused RL, arguing that RLVR achieves capability improvements that RLHF does not. However, no head-to-head comparison between RLVR and RLHF on reasoning benchmarks is systematically reported; the comparison is conceptual rather than empirical.
- **SFT-only:** The RL vs. SFT debate (Section 4.2) contrasts RLVR-trained models against models trained with supervised fine-tuning on the same reasoning data. The survey cites Chu et al. (2025a) for the claim that "SFT memorizes, RL generalizes" based on out-of-distribution evaluation, and Jin et al. (2025d) for the counter-claim that "RL can partially mitigate overfitting [but] remains ineffective in cases of severe overfitting or abrupt distributional shifts."

**Generation budget / compute accounting.** The survey does not standardize compute accounting across studies. Generation budget is typically measured in number of sampled responses per prompt (e.g., GRPO uses group size G, typically 4–64). The survey discusses compute cost primarily in qualitative terms: critic-based methods (PPO) require "a critic model to run and optimize along the target LLM, and create a significant computational overhead" (Section 3.2.2), while critic-free methods (GRPO) "significantly reduc[e] the computational requirement and simplify training" (Section 3.2.3). FLOPs-matched comparisons between RLVR and pretraining scaling are discussed conceptually (Section 1) but are drawn from primary studies (e.g., Snell et al., 2024 on compute-optimal test-time scaling) rather than original analysis.

**Cross-validation / statistical protocol.** The survey does not establish or enforce any statistical protocol. It acknowledges in Section 4.4 (Tricks or Traps) that the field suffers from "inconsistent experimental settings, incomplete reporting, and conflicting conclusions"—precisely the problems that standardized protocols would address. The survey's contribution is diagnostic rather than prescriptive: it identifies the problem but does not solve it.

### Main Quantitative Results

#### Frontier Model Performance Evolution (Table 1, Figure 4)

The survey documents the rapid trajectory of reasoning model capabilities through a timeline (Figure 4) and model comparison table (Table 1). The headline quantitative pattern is the speed of convergence: DeepSeek-R1 (January 2025) achieved parity with OpenAI o1 on mathematics and coding benchmarks; QwQ-32B (March 2025, 32B parameters) "matched R1's performance" according to the survey; Qwen3-235B (April 2025) "further improv[ed] benchmark scores"; and by mid-2025, models from multiple organizations (Skywork, Llama-Nemotron, Minimax-M1, Magistral) were reporting comparable reasoning capabilities.

However, the survey reports these as qualitative claims from the original papers rather than providing a unified benchmark table with standardized evaluation. The specific numbers are cited from the original model releases: DeepSeek-R1 reports 79.8% on AIME 2024 and 97.3% on MATH-500; QwQ-32B reports comparable figures. The survey's contribution is in *timeline organization*— demonstrating the field's trajectory—rather than in precise cross-model quantitative comparison.

#### Algorithmic Scaling Results Across the Field

The survey aggregates several key quantitative findings from the primary RL algorithm literature, though without original meta-analysis:

**GRPO scaling with model size.** DeepSeek-R1 (Guo et al., 2025a) demonstrated that GRPO with rule-based accuracy and format rewards produces emergent reasoning behaviors when applied to models from 1.5B to 671B parameters. The survey cites this as evidence for the RLVR scaling paradigm but does not extract specific scaling law coefficients. Open-Reasoner-Zero (Hu et al., 2025b) subsequently replicated this finding on a 7B Qwen base model, showing "both response length and benchmark accuracy" improvements from GRPO alone, mirroring R1-Zero training dynamics. The key quantitative implication—not fully extracted by the survey—is that the *effectiveness of RLVR appears to scale with base model capability*, but systematic scaling law studies remain nascent.

**GRPO vs. PPO comparisons.** The survey does not present a unified GRPO vs. PPO quantitative comparison, though it summarizes findings from individual studies. DeepSeek-R1's technical report argues that GRPO avoids the reward hacking and instability that learned reward models exhibit in large-scale RL. VAPO (Yue et al., 2025c) proposes a robustified value-based PPO variant and reports improved stability—but the survey does not extract whether VAPO matches GRPO's performance at equivalent compute. This is a gap: the survey frames GRPO as the dominant paradigm but does not quantify its advantage over critic-based methods.

**Off-policy and replay results.** Several methods report sample efficiency improvements. Retrospective Replay (Dou et al., 2025) shows that selectively replaying earlier reasoning traces improves exploration in GRPO training—but the survey reports this qualitatively. PPER (Chen et al., 2024c) reports "more stable optimization" for code generation through prioritized experience replay. GRESO (Zheng et al., 2025b) claims "pre-filtering can speed up rollout time by 2.4× and overall training by 2.0× with minimal loss in performance." The survey catalogues these claims without independent verification.

**Regularization effectiveness.** The survey reports several quantitative findings about regularization but does not aggregate them:

- **Entropy collapse:** Multiple studies (Cheng et al., 2025a; Cui et al., 2025b; Yu et al., 2025d) observe that GRPO training without intervention leads to entropy collapse—the policy's output distribution becomes increasingly peaked. The survey does not quantify the magnitude of entropy decline or its performance impact.
- **Length growth:** The survey notes that RLVR training typically produces longer reasoning traces but does not extract specific length-growth curves. DeepScaleR (Luo et al., 2025c) reports that "staged context lengthening" (8K → 16K → 24K → 32K) was essential for stable training. The survey does not report the accuracy improvement attributable to length budgeting strategies specifically.
- **KL regularization removal:** The survey claims that "a majority of other recent works advocate for removing the KL penalty entirely" (Section 3.2.5), citing studies that "simplify implementation, reduce memory cost and achieve more scalable GRPO." But it does not provide a quantitative comparison: what is the accuracy difference between KL-regularized and KL-free GRPO training, controlling for other hyperparameters? This omission is notable given how central this debate is to the field.

#### Application-Level Quantitative Evidence (Section 6)

The survey's application coverage (Section 6) reports benchmark numbers from primary studies across domains:

**Coding tasks (Section 6.1).** The survey reports that DeepCoder (Luo et al., 2025b, 14B parameters) achieves "o3-mini level" performance on competitive programming benchmarks after RLVR training, though the specific benchmark scores are not extracted. Afterburner (Du et al., 2025a) reports raising Pass@1 from 47% to 62% on code efficiency optimization, "surpassing human-level efficiency"—a specific, verifiable claim cited from the primary study. Repair-R1 (Hu et al., 2025a) reports improved automated program repair through joint test-case-generation-and-repair RL training; specific benchmark improvement numbers are not extracted.

**Agentic tasks (Section 6.2).** The survey reports that SWE-RL (Wei et al., 2025c) applies GRPO to software engineering tasks and demonstrates improvements on SWE-bench; that Search-R1 (Jin et al., 2025b) trains interleaved reasoning-search behavior via RLVR; and that multiple specialized agents for web browsing, tool use, and deep research report gains over non-RL baselines. Specific benchmark scores are cited for some (e.g., Kimi-K2 achieving strong performance on agentic benchmarks) but not systematically tabulated.

**Multimodal tasks (Section 6.3).** VLM-R1 (Shen et al., 2025a) and Vision-R1 (Huang et al., 2025c) extend GRPO-style RL to visual reasoning tasks, achieving state-of-the-art on several detection and grounding benchmarks with limited training data. The survey notes this as a paradigm shift from data scaling to reward function design, but does not extract specific accuracy improvements or the amount of training data used. DAPO (Yu et al., 2025d) is cited for applying RLVR to multimodal understanding with rule-based rewards derived from visual grounding correctness.

**Robotics tasks (Section 6.5).** SimpleVLA-RL (Li et al., 2025e) "surpasses state-of-the-art VLA models like π0 on LIBERO and RobotWin2.0 benchmarks" using GRPO with binary success/failure rewards and just a single demonstration trajectory. The survey reports this as evidence that RLVR generalizes to embodied domains, though it does not extract the specific success rate improvements.

**Medical tasks (Section 6.6).** Med-U1 (Zhang et al., 2025l) and MED-RLVR (Zhang et al., 2025j) apply GRPO with rule-based rewards to medical QA, reporting improved OOD generalization. The survey notes that medical tasks divide cleanly into verifiable (multiple-choice QA, structured prediction) and non-verifiable (report generation, treatment planning) categories, with RLVR effective primarily on the former.

### Ablation Studies and Robustness Checks

This section faces a structural challenge: the survey itself does not conduct ablation studies or robustness checks—it summarizes those reported in the primary literature. The relevant "ablation" is the comparison the survey implicitly makes across studies: do the patterns it identifies hold across different algorithms, model families, task domains, and experimental conditions, or are they artifacts of specific configurations?

**Model family as an implicit ablation:** The survey's most important robustness observation—discussed extensively in Section 4.3—is the asymmetric RL responsiveness across model families. Shao et al. (2025, Spurious Rewards) demonstrates that Qwen-family models show significant performance improvements even under *random* or *spurious* reward signals that bear no actual relationship to answer correctness, while Llama and OLMo models exhibit no such improvement. The survey correctly identifies this as evidence that some models possess latent reasoning priors (from pretraining data distribution) that RL can "sharpen" even without valid reward signals. This finding is an implicit ablation of reward signal quality: it suggests that for certain model families, the specific reward design matters less than previously assumed, while for others, reward quality is critical. However, the survey does not quantify these differences—it reports the finding qualitatively.

**KL regularization as an ablation:** The survey documents (Section 3.2.5, 4.4) a sharp divide in the field over KL regularization. Studies that remove KL regularization entirely (DAPO, PRIME, Skywork-OR1, SimpleRL) report that it "simplif[ies] implementation, reduce[s] memory cost and achieve[s] more scalable GRPO." Studies that retain or modify KL regularization (Archer, ProRL, OREAL, K1.5) argue it prevents catastrophic forgetting and stabilizes training. The absence of controlled comparisons—same base model, same reward function, same training budget, varying only KL regularization—means the survey cannot adjudicate this debate. It identifies the controversy but cannot resolve it empirically.

**Critic-based vs. critic-free as an ablation:** The survey presents GRPO (critic-free) as the dominant paradigm, citing DeepSeek-R1's success and the computational overhead of training a separate critic. However, the survey does not extract quantitative comparisons between GRPO and PPO variants at matched compute budgets. VAPO and VCPPO are cited as critic-based improvements, but whether they match or exceed GRPO performance is not numerically established. A missing ablation is: does the critic's per-token advantage estimation provide benefits that outweigh its computational cost for long-chain reasoning tasks? The survey's framing implies "no" but does not prove it.

**Dynamic sampling as an ablation:** Multiple dynamic sampling methods are catalogued: DAPO's difficulty filtering (drop saturated/degenerate prompts), PRIME's online filtering (drop too-easy or too-hard problems), K1.5's prioritized sampling (proportional to failure rate). Each claims improvements in training efficiency or final performance. But the survey provides no head-to-head comparison: does DAPO's filtering outperform PRIME's, or are they roughly equivalent? Is the performance gain from dynamic sampling larger or smaller than the gain from algorithm choice (GRPO vs. DAPO vs. Dr. GRPO)? The survey identifies dynamic sampling as an important component but cannot quantify its impact relative to other design choices.

**Reward type as an ablation:** The survey's taxonomy of reward types (verifiable, generative, dense, unsupervised, shaped) is comprehensive, but it lacks quantitative comparison: how much does switching from outcome-only rewards to step-level process rewards improve final accuracy, controlling for other factors? The Process vs. Outcome debate (Section 4.5) is presented as an open question, with arguments on both sides. Lightman et al. (2024) found that PRMs with process supervision outperform ORMs for mathematical reasoning; but the survey notes that step-wise annotation is costly and Monte Carlo synthesis approaches introduce bias. The survey does not extract specific accuracy differences between PRM-guided and ORM-guided training from the primary literature—a notable gap given the centrality of this debate to reward design.

**Negative results and failure modes:** The survey commendably reports negative results, which provide important boundary conditions:

- **Lookahead search degradation:** The survey notes (Section 3.1.3 discussion) that some search methods like lookahead search can paradoxically underperform simpler methods at high compute budgets due to verifier over-optimization—the model finds solutions that score highly under the learned verifier but are actually incorrect.
- **ReST^EM revision model backfire:** In the revision model context (Appendix K of a primary study), the survey reports that "additional sequential revisions substantially hurt performance" when the model is trained with ReST^EM, with "fully sequential performance drop[ing] to approximately 33.5% compared to roughly 38.5% at the optimal ratio." This is a genuine negative result showing that RLVR training can degrade performance if the data generation strategy introduces spurious correlations.
- **Reward hacking in learned reward models:** DeepSeek-R1's explicit finding that "RMs may suffer from reward hacking when scaled to large-scale RL settings" (Guo et al., 2025a) is cited as motivation for using rule-based rewards wherever possible. This negative result about learned reward models shaped the entire field's trajectory toward verifiable rewards.
- **Entropy collapse without intervention:** The survey documents that GRPO training without explicit entropy maintenance leads to policy collapse—a failure mode that has spawned numerous mitigation strategies (entropy bonuses, Clip-Higher, high-entropy-only training).

### Critical Assessment

This section evaluates whether the evidence the survey synthesizes actually supports its framing claims, and identifies where the evidence is thin, contradictory, or absent.

**The central framing claim: RLVR constitutes a distinct, new scaling paradigm.** The survey argues (Section 1, Figure 2) that RLVR represents a fundamental shift from alignment-focused RL toward capability-building RL. The evidence for this claim is substantial but indirect: the explosion of open-source reasoning models trained with RLVR (Table 1, Figure 4), each reporting dramatic improvements over their base models on reasoning benchmarks; the conceptual architecture of RLVR—using verifiable rewards to incentivize reasoning behaviors rather than using human preferences to constrain outputs; and the emergence of a new optimization ecosystem (GRPO variants, verifiers, dynamic environments) distinct from the RLHF toolchain.

However, the evidence has important limitations:

- **No controlled comparison between RLVR and RLHF on reasoning tasks.** The survey does not report any study that applies both RLVR (with verifiable rewards) and RLHF (with learned reward model trained on human preferences) to the same base model and task, then compares final reasoning performance. The claim that RLVR is superior for reasoning is supported by the *absence* of RLHF-trained reasoning models achieving competitive performance, not by direct comparison. This is a missing baseline of the first order.

- **The "paradigm shift" is primarily a shift in reward signal, not in optimization algorithm.** GRPO—the dominant RLVR algorithm—is architecturally a variant of PPO (clipped surrogate objective) with a group-relative advantage estimator instead of a learned critic. It is not a fundamentally different class of algorithm from what RLHF uses. The real shift is in *what signal drives optimization* (automatically verifiable correctness vs. learned human preferences) and *what behaviors that signal incentivizes* (reasoning vs. alignment). The survey's framing sometimes conflates the *reward signal shift* with an *algorithmic shift*, which overstates the novelty of the optimization machinery.

- **The scaling axis is not yet quantified.** The survey invokes scaling laws language ("new scaling axis") but does not present or extract scaling law coefficients from the primary literature. How does reasoning performance scale with RL training compute? How does it scale with model size under RLVR? How does the RLVR scaling exponent compare to pretraining scaling exponents? The survey cites Snell et al. (2024) on compute-optimal test-time scaling, and Aghajanyan et al. (2023) and Kaplan et al. (2020) on pretraining scaling laws, but does not synthesize quantitative comparisons. The "scaling axis" claim is conceptually motivated but empirically underdetermined.

**The co-design claim: reward design, policy optimization, and sampling strategy are interdependent.** The survey's organizational structure implicitly argues that these three components must be co-designed. The evidence supporting this claim is distributed across the survey and primarily takes the form of *interaction effects* documented in primary studies:

- Reward density (outcome vs. process) interacts with algorithm choice (critic-free GRPO is well-suited to outcome rewards; critic-based PPO may benefit from process rewards).
- Sampling strategy (dynamic difficulty filtering) only works reliably when the reward signal is trustworthy—a noisy reward would make filtered prompts misleading.
- KL regularization necessity depends on reward reliability (essential for learned reward models in RLHF; often removed in RLVR with rule-based rewards).

This is a valid and important organizational insight. However, the evidence is qualitative and sometimes contradictory. The survey presents the interaction effects as *patterns observed across multiple studies* but does not provide controlled experiments that isolate individual interactions. For example, no study is cited that systematically varies reward type (outcome vs. process) × algorithm type (GRPO vs. PPO) × KL regularization (on vs. off) and reports all 2×2×2 cell accuracies. Without such controlled comparisons, the claim of interdependence—while plausible and consistent with the evidence—remains a high-level organizational hypothesis rather than an empirically verified principle.

**The foundational debates claim: unresolved tensions define the field's identity.** This is the survey's strongest and most empirically-grounded contribution. For each debate, the survey presents specific, contradictory findings from the primary literature:

- **Sharpening vs. Discovery:** The evidence genuinely cuts both ways. Limit-of-RLVR (Yue et al., 2025b) shows Pass@K degradation under RL—clear sharpening evidence. ProRL (Liu et al., 2025) reports Pass@K improvements—discovery evidence. The survey appropriately treats this as unresolved, and the contradictory evidence is not a weakness of the survey but a feature of the research landscape.

- **RL vs. SFT generalization:** Chu et al. (2025a) provides evidence that RL generalizes better than SFT on visual and textual OOD tasks. Jin et al. (2025d) provides evidence that RL cannot recover from severe SFT-induced overfitting. These are not contradictory—they describe different regimes—but resolving *where* the boundary lies requires more systematic study. The survey's contribution is framing this as a boundary-identification problem.

- **Model prior (Qwen vs. Llama):** The Spurious Rewards finding (Shao et al., 2025) is the strongest single empirical result in the survey: Qwen models improve under random rewards; Llama models do not. This is not a debate with two sides—it is an empirical regularity that demands explanation. The survey's treatment correctly identifies this as perhaps the most important open question in RLVR: what makes a model "RL-friendly"?

- **Tricks vs. Traps:** The survey explicitly acknowledges that many algorithmic variants may be configuration-specific artifacts. The citation of a unified evaluation (Liu et al., 2025a) showing that "a minimalist combination of methods can outperform GRPO and DAPO across multiple configurations" is damning: if a simple baseline outperforms the more complex variants, much of the algorithmic innovation literature may be noise. The survey's framing of this as a "most pressing challenge" is appropriate and well-supported.

**Weaknesses in the evidential basis that the survey acknowledges:**

- **Lack of standardized benchmarks for comparing RL training methods.** The survey notes that different studies use different base models, different training data mixtures, different hyperparameter schedules, and different evaluation protocols. This makes cross-study comparison unreliable—and the survey's synthesis is necessarily qualitative rather than quantitative.

- **Benchmark saturation and contamination concerns.** As reasoning benchmarks like MATH become saturated (models approaching 90%+ accuracy), the ability to distinguish between methods degrades. Newer, harder benchmarks (FrontierMath, Humanity's Last Exam) exist but are not yet standardized evaluation targets for RLVR research. The survey does not discuss benchmark saturation as a limitation.

- **The file-drawer problem.** The survey catalogues successful RLVR applications across domains. But how many unsuccessful attempts—where RLVR failed to improve reasoning—are not published? The survey discusses negative results from published work (reward hacking, entropy collapse, ReST^EM backfire, spurious reward effects for Llama) but cannot account for unpublished failures. This biases the apparent generality of RLVR.

- **Compute scale as a hidden variable.** The most dramatic RLVR successes (DeepSeek-R1, Kimi-K2, GPT-5) come from organizations with massive compute resources. The survey's application sections report results from smaller-scale replications (e.g., SimpleVLA-RL using a single demonstration trajectory), but the relationship between compute budget and RLVR effectiveness is not systematically analyzed. Does RLVR's advantage over SFT grow with compute, shrink, or remain constant? The survey cannot answer this from the available evidence.

**Experiments that would have strengthened the survey's claims but are absent:**

- A systematic, controlled comparison of GRPO vs. PPO vs. REINFORCE variants on a standardized benchmark suite (MATH, AIME, HumanEval, SWE-bench) using identical base models, training data, and compute budgets.
- Quantification of the interaction between model family (Qwen, Llama, OLMo, DeepSeek) and RLVR effectiveness at multiple model scales, to establish whether "RL-friendliness" is a continuous or discrete property.
- Systematic ablation of KL regularization strength (from zero to strong) across different reward types (rule-based, learned RM, generative RM) to map the regularization landscape.
- Pass@K curves (for K from 1 to 1024) for models before and after RLVR training, to empirically characterize the Sharpening vs. Discovery transition point.
- Scaling law fits relating RL training compute (measured in total tokens generated during RL) to downstream reasoning performance, analogous to pretraining scaling laws.

## 6. Limitations and Trade-offs

### 6.1 Boundary Condition: RLVR Efficacy Is Fundamentally Gated by Verifiability of the Reward Signal

**The assumption or constraint.** The survey's central organizing principle — Verifier's Law — is simultaneously its greatest insight and its sharpest boundary. Throughout Section 3.1.1, the paper establishes that RLVR's success cases (mathematics, competitive programming) satisfy a specific set of criteria: "the existence of clear ground truth, the availability of rapid automated verification, the scalability of evaluating many candidate solutions, and a reward signal that is closely aligned with correctness." The survey is explicit that tasks failing these criteria — "open-ended question answering or free-form writing" — "remain challenging for outcome-based RL, as they rely on noisy learned reward models or subjective human feedback" (Section 3.1.1).

**The consequence.** The practical consequence is that the dominant RL paradigm this survey catalogues is not general-purpose. A practitioner attempting to apply GRPO-based RLVR to their domain must first answer a gatekeeping question: can I construct an automatic, reliable, scalable reward signal? For many high-value applications — legal reasoning, medical diagnosis with ambiguous cases, policy analysis, creative writing, open-ended scientific hypothesis generation — the answer is "no, not without building expensive learned reward models that may themselves be unreliable." The survey's extensive coverage of generative rewards (Section 3.1.2), dense rewards (Section 3.1.3), and unsupervised rewards (Section 3.1.4) documents the field's attempts to *extend* this boundary, but critically, none of these extensions is presented as having achieved the reliability and scalability of rule-based verification. The survey notes that learned reward models can be gamed, that Monte Carlo-synthesized process labels "generalize poorly and introduce bias" (Section 4.5, citing Yin et al., 2025), and that unsupervised rewards risk "reward hacking and model collapse" (Section 3.1.4). The boundary is real and the attempted extensions remain fragile.

**What evidence exists in the paper.** This limitation is anchored throughout the survey, not in a single figure or table. The taxonomy itself embodies it: verifiable rewards (Section 3.1.1) are placed as the foundational, scalable category, with all other reward types positioned as extensions or workarounds for domains lacking verifiability. The application sections reinforce this: coding tasks (Section 6.1) succeed because unit tests and compilers provide automatic verification; agentic tasks (Section 6.2) succeed where environment feedback (tool execution results, search engine outcomes) can be automatically scored; medical tasks (Section 6.6) report RLVR success primarily on *verifiable* subtasks like multiple-choice QA while "generation-oriented tasks remain challenging" with "scalable RL on non-verifiable tasks" described as an open problem. The survey does not present a single case study where RLVR achieves dramatic gains on a genuinely non-verifiable task without human annotation or learned reward models of uncertain reliability.

**Mitigation status.** The survey acknowledges this limitation extensively but does not resolve it. The progression from verifiable → generative → dense → unsupervised rewards (Sections 3.1.1–3.1.4) maps a research trajectory toward extending verifiability, but each step introduces new fragility. The survey explicitly frames "open-ended RL" as a future direction (Section 1, Figure 2) and describes "tasks without fast or objective verification" as an open challenge. In Section 7, future directions like "RL for LLMs in Scientific Discovery" (§7.8) are presented as areas where verifiability is the central bottleneck — wet-lab experiments are too slow and expensive to serve as RL reward signals, and *in silico* simulations are "far from sufficient for replacing realistic lab environments due to their limited scope and critical lack of accuracy and generalizability" (Section 7.8). The survey suggests hybrid approaches combining model-based and rule-based signals but provides no empirical evidence that such hybrids achieve rule-based reliability at scale.

---

### 6.2 Model Prior Dependency: RLVR Effectiveness Is Contingent on Base Model Choice in Ways the Field Does Not Understand

**The assumption or constraint.** The survey assumes that RLVR is a general methodology applicable to any pretrained language model. Yet Section 4.3 ("Model Prior: Weak and Strong") documents a finding that fundamentally challenges this assumption: model *family* dramatically determines RLVR responsiveness. The survey cites Shao et al. (2025, Spurious Rewards) showing that "Qwen-family models register significant gains even under random or spurious reward signals, whereas Llama and OLMo models often do not." This is not a minor variation in effectiveness — it suggests that for some model families, RLVR may produce *no meaningful improvement regardless of algorithm design*, because the pretrained model simply lacks the latent reasoning patterns that RL can "sharpen."

**The consequence.** For practitioners, this creates a fundamental uncertainty at the start of any RLVR project. If you are working with a Llama-based model (or any model family not extensively validated in RLVR studies), you cannot know — without running expensive experiments — whether RLVR will work at all. The survey's Table 1 reveals that the vast majority of successful open-source reasoning models are based on either DeepSeek or Qwen architectures. Llama-Nemotron-Ultra appears in the table but is described as "aim[ing] to balance accuracy and efficiency" (Section 2.2) rather than being presented as a breakthrough, and the survey does not report whether its GRPO training produced gains comparable to Qwen-based models at similar scale. The consequence is not that RLVR *cannot* work on non-Qwen models — mid-training strategies (Section 4.3) are presented as a partial mitigation — but that the *baseline expected effectiveness* is unknown and potentially zero for some model families.

**What evidence exists in the paper.** The evidence is concentrated in Section 4.3:

- The Spurious Rewards finding (explicitly cited) is the strongest single piece of evidence: Qwen models improve under random rewards; Llama/OLMo do not. This is a stark, clean empirical finding that should give any practitioner pause.
- Section 4.3 notes that "Qwen models, having been extensively exposed to such [mathematical or code CoT] distributions [during pretraining], tend to be more 'RL-friendly'." This proposes a mechanism — differential pretraining data — but the survey does not quantify how much pretraining on reasoning data is sufficient, or whether any model family can be made "RL-friendly" through targeted mid-training.
- The survey's coverage of mid-training solutions (Section 4.3) — annealing on high-quality math and code data, following the Llama 3 and OLMo 2 recipes — is presented as a mitigation but not as a solution. The survey states that these strategies "effectively narrow the performance gap" but does not present head-to-head comparisons showing mid-trained Llama models matching Qwen models under identical RLVR training.
- Table 1's roster of RL-trained models is implicitly dominated by DeepSeek and Qwen architectures. A casual reader could miss this pattern; the survey does not compute or report the fraction of successful models by base architecture.

**Mitigation status.** Partial. The survey identifies the mid-training → RLVR pipeline as a practical recipe for "weak-prior" model families (Section 4.3: "strengthen reasoning priors through mid-training, and subsequently apply RLVR"). However, this shifts the burden: rather than RLVR being a standalone post-training step, it becomes dependent on a preceding mid-training phase whose efficacy has not been systematically validated across model families and scales. The survey explicitly notes that this is an open area: the relationship between pretraining data composition, mid-training recipe, and RLVR responsiveness is identified as a research direction but not characterized empirically in terms of scaling laws or thresholds.

---

### 6.3 Lack of Rigorous Cross-Study Comparability Undermines the Survey's Ability to Adjudicate Between Competing Methods

**The assumption or constraint.** A survey paper's value depends on its ability to draw comparisons across studies — to identify which methods work, which don't, and under what conditions. This survey explicitly acknowledges that the primary literature it synthesizes suffers from a fundamental problem: "the field's most pressing challenges: inconsistent experimental settings, incomplete reporting, and conflicting conclusions" (Section 4.4). The assumption that one can meaningfully compare GRPO against DAPO, or critic-based against critic-free methods, or outcome rewards against process rewards, by aggregating results from different papers with different base models, datasets, hyperparameter schedules, and evaluation protocols is not supported.

**The consequence.** The survey's taxonomy and comparative framing imply relative assessments that may not be justified. When the survey states that GRPO "has been shown to speed up the training process" compared to PPO (Section 3.2.3), or that "a majority of other recent works advocate for removing the KL penalty entirely" (Section 3.2.5), it is reporting *claimed* advantages from individual papers, not *validated* advantages under controlled comparison. A practitioner reading that "beam search significantly outperforms best-of-N at low generation budgets" or that "process rewards provide 'interpretable dense guidance'" might reasonably conclude these are established facts. But the survey itself warns in Section 4.4 that the tricks-vs-traps problem means many reported improvements may be artifacts of specific configurations — and this warning applies to the very claims the survey synthesizes. The survey cannot tell you, from its aggregated evidence, whether DAPO is genuinely better than GRPO or merely appears so under the specific conditions of its original paper.

**What evidence exists in the paper.** The evidence for this limitation is the survey's own meta-analysis in Section 4.4:

- Direct quote: "inconsistent experimental settings, incomplete reporting, and conflicting conclusions. This constitutes a fundamental limitation in the current application of RL within the research community."
- The survey cites a unified evaluation (Liu et al., 2025a) that "demonstrates that a minimalist combination of methods can outperform GRPO and DAPO across multiple configurations." This implies — though the survey does not state it explicitly — that some or many of the published algorithmic improvements may be illusory.
- The survey's Tables 4, 5, and 6 — cataloguing over 40 static datasets, dozens of dynamic environments, and 11+ infrastructure frameworks — implicitly demonstrate the problem: with so many varying substrates, cross-study comparison is unreliable by construction.
- No meta-analytic aggregation of results appears in the survey. There is no forest plot comparing GRPO variants, no summary table of accuracy improvements standardized by base model and benchmark, no attempt to control for the hidden variables that make cross-study comparison hazardous. The survey is organized as a taxonomy and narrative synthesis, not as a quantitative meta-analysis. This is appropriate for the field's current state but limits the strength of the conclusions that can be drawn.

**Mitigation status.** The survey acknowledges the problem and calls for "unified experimental protocols, verifiable reward structures, and explicit scalability–performance–cost curves" (Section 4.4). But it does not itself provide these. It does not propose a standardized benchmark suite, a recommended evaluation protocol, or a minimum reporting standard. The survey's contribution is diagnostic: it names the problem clearly. But the problem remains unsolved, and the survey's own comparative claims inherit the uncertainty it diagnoses.

---

### 6.4 Difficulty Estimation and Compute Allocation Costs Are Externalized from the RLVR Efficiency Narrative

**The assumption or constraint.** The survey frames RLVR as a new scaling axis (Section 1, Figure 2) and extensively documents the training infrastructure (Section 5.3) and computational demands of different algorithmic choices (critic-based vs. critic-free, Section 3.2). However, there is a class of costs that the survey — and much of the primary literature it synthesizes — systematically externalizes: the costs of *deciding how to allocate RL compute*. These include difficulty estimation (which prompts are worth training on?), hyperparameter search (which algorithm variant works for this model family and task?), curriculum design (in what order should tasks be introduced?), and the cost of failed training runs (entropy collapse, reward hacking, training instability).

**The consequence.** The "scaling axis" framing implies that investing more RL compute predictably improves reasoning. But this is true only *conditional on having solved the allocation problem* — knowing which prompts to train on, with which algorithm, at which temperature, for how many steps. The survey documents (Section 3.3.1) that methods like DAPO require *oversampling and filtering* prompts to find those with non-zero advantage; PRIME requires *online filtering* to drop too-easy or too-hard problems; dynamic curriculum methods require *bandit algorithms* to select categories. None of these meta-costs are included in the efficiency calculations. If it costs 2× the generation budget to identify which prompts are worth training on, the headline efficiency gains shrink accordingly. The survey does not compute or discuss these amortized costs.

Furthermore, the survey documents training instability as a first-class problem — entropy collapse (Section 3.2.5), reward hacking (Section 3.1.2), ReST^EM backfire (Section 6.1, referencing primary studies), the finding that some model families don't respond to RLVR at all (Section 4.3) — but does not estimate the *failure rate* of RLVR training runs. A practitioner adopting RLVR faces not only the cost of successful training but the expected cost of *failed* training runs due to hyperparameter misconfiguration, model prior mismatch, or reward design flaws. The survey's extensive catalogue of "tricks" (Section 4.4) — Clip-Higher, dynamic sampling, entropy bonuses, length management, KL tuning — implicitly acknowledges that getting RLVR to work requires navigating a high-dimensional configuration space, but it does not estimate the search cost over this space.

**What evidence exists in the paper.**

- Section 3.3.1 ("Dynamic and Structured Sampling") documents that methods like DAPO "over-samples and filters prompts whose rollouts are saturated (all-correct) or degenerate (all-wrong), then repeatedly samples until each mini-batch contains prompts with non-zero advantage." The "repeatedly samples" part implies a cost multiplier that is not quantified relative to naive uniform sampling.
- Section 3.2.5 documents entropy collapse as a pervasive failure mode requiring intervention — entropy bonuses, Clip-Higher, targeted training — each of which represents additional hyperparameters that must be tuned.
- Section 4.4 ("Tricks or Traps") explicitly states that "progress in the field requires unified experimental protocols... to show that a method remains effective as it scales, rather than only at specific data or models." This acknowledges that current evidence is configuration-specific, but the survey does not quantify the configuration search cost.
- Section 4.3's documentation of model family asymmetry means that a practitioner must potentially run RLVR experiments on *their specific base model* to determine whether it will work — the published results on Qwen or DeepSeek may not transfer.
- The survey provides no figure or table estimating total cost of ownership for an RLVR pipeline, inclusive of meta-costs.

**Mitigation status.** Not addressed. The survey does not propose methods for estimating or reducing these meta-costs. The sampling strategies section (3.3) discusses how to allocate compute *within* a training run but not how to choose which training run to attempt. The infrastructure section (5.3) compares frameworks on throughput and features but not on developer time, failure recovery, or hyperparameter search tooling. This is not necessarily a flaw — the survey's scope is the RLVR methodology itself — but it means a practitioner reading this survey cannot estimate the *total cost* of adopting RLVR from the information provided.

---

### 6.5 Single-Paradigm Focus Obscures the Maturity and Tradeoffs of Non-RLVR Approaches

**The assumption or constraint.** The survey explicitly positions RL at the center of its analysis and treats RLVR as the dominant post-training paradigm for reasoning (Section 1, Section 2.3). This is a legitimate scoping choice, but it creates a systematic blind spot: the survey does not provide the reader with sufficient information to assess whether RLVR is *the best available approach* for a given reasoning task, as opposed to one approach among several with different tradeoff profiles.

**The consequence.** A practitioner reading this survey might reasonably conclude that RLVR is the only serious option for training reasoning models. But the survey does not systematically compare RLVR against several alternative or complementary paradigms:

- **Inference-time compute scaling without training:** Methods like best-of-N sampling with verifiers, tree search against process reward models, or iterative refinement at inference time (as studied in Snell et al., 2024, which the survey cites in Section 1) can improve reasoning without any model weight updates. The survey does not compare the cost-performance frontier of these approaches against RLVR training — under what conditions is it better to invest compute in training vs. inference?

- **Supervised fine-tuning on long chain-of-thought data:** The survey frames RL vs. SFT as a debate (Section 4.2) and cites evidence on both sides, but does not provide a clear decision boundary. The finding that "long-CoT SFT and rule-based RL... expand reasoning depth and self-reflection" while "short-CoT SFT frequently harms generalization" (Section 4.2, citing Zhou et al., 2025d) suggests that *long-CoT SFT* may achieve many of the same benefits as RLVR without the instability and configuration complexity. The survey does not quantify the performance gap between long-CoT SFT and RLVR at matched training budgets.

- **Distillation from stronger models:** Many of the open-source reasoning models catalogued in Table 1 (e.g., Skywork-OR1, mentioned as "based on R1-distilled models") use distillation from larger reasoning models as part of their training pipeline. The survey documents this as a data acquisition method (Table 4: "Distil" category) but does not assess how much of the reported RLVR gains are attributable to the distillation step vs. the RL step. If a distilled SFT model already achieves 80% of the RLVR gain, the marginal value of RL may be substantially smaller than the survey implies.

The consequence is that the survey's framing may overstate the *necessity* of RLVR relative to the *sufficiency* of alternative approaches. This is not a flaw in the survey's coverage of RLVR — which is comprehensive — but a limitation of its scope, which a practitioner needs to be aware of when making resource allocation decisions.

**What evidence exists in the paper.** This limitation is evidenced by absence rather than presence. The survey:

- Does not contain a systematic comparison table showing RLVR vs. long-CoT SFT vs. inference-time scaling vs. distillation across standard benchmarks.
- Cites Snell et al. (2024) on compute-optimal test-time scaling (Section 1) but does not integrate this line of work into a comparative framework with RLVR.
- Documents distillation as a data source (Table 4, "Distil" entries) but does not ablate the contribution of distillation vs. RL in models that use both.
- Acknowledges in Section 4.2 that "SFT may serve as a lower bound for sparse reward RL" but does not quantify the gap or identify conditions where the gap narrows to irrelevance.

**Mitigation status.** The survey does not attempt to address this limitation, as it is a consequence of its deliberate scoping choice to center RL. The future directions section (Section 7) discusses efficiency (Section 7.4: "Teaching LRMs Efficient Reasoning") and alternative architectures (Section 7.7: diffusion-based LLMs) but does not propose a systematic comparison framework with non-RL approaches. A practitioner would need to consult separate literature to make an informed choice between RLVR and alternatives.

---

### 6.6 Scope Limited to Post-Training RL; The Integration of RL with Pretraining and Architecture Design Is Aspirational but Unvalidated

**The assumption or constraint.** The survey is organized around RL as a *post-training* methodology — something applied to already-pretrained models. This is a reasonable scoping choice given that the vast majority of the surveyed literature operates in this regime. However, the survey's forward-looking sections (Section 7) explicitly discuss extending RL into pretraining (§7.6: "RL for LLMs Pre-training"), architecture design (§7.9: "RL for Architecture-Algorithm Co-Design"), and continual learning (§7.1: "Continual RL for LLMs"). These are presented as "future directions" that are "poised to shape the next wave of advances" (Section 7 introduction).

**The consequence.** The practical implication is that a practitioner or researcher reading the "Future Directions" section might conclude that these extensions are imminent or well-validated. But the evidence the survey marshals for them is thin:

- **RL for pretraining:** The survey cites exactly one primary work — Reinforcement Pre-Training (Dong et al., 2025c) — as "reconceptualiz[ing] next-token prediction as an RL problem with verifiable rewards derived from the corpus." The survey describes this as "reporting consistent gains" but does not present quantitative results, discuss scalability, or cite replication studies. The single open-source initiative cited (avatARL) is described as "training language models from random initialization purely with RL" using "iterative 'referee' scoring" — a proof-of-concept, not a validated methodology. The survey acknowledges this is preliminary (describing it as "emerging research" and "a promising scaling strategy") but the gap between the current post-training paradigm and the envisioned pretraining paradigm is enormous and unquantified.

- **RL for architecture co-design:** Section 7.9 proposes that "making architecture a first-class action space in RL represents an open and high-impact challenge." The survey describes reinforced MoE routing, sparsity pattern learning, and hardware-aware optimization as aspirational goals. But it cites *zero* primary works that have demonstrated RL-driven architecture optimization at the scale of modern LLMs. The cited work (Zoph and Le, 2016) is from the pre-LLM neural architecture search literature. The survey frames this as "an open and high-impact challenge" and enumerates "key open questions" — all of which are unsolved.

- **Continual RL:** Section 7.1 identifies the stability-plasticity dilemma and the challenge of "entangled nature of knowledge and reasoning in LLMs" as central obstacles. The cited methodological frameworks (Experience Replay, Policy Reuse, Reward Shaping) are drawn from "traditional CRL research" and have not been validated at LLM scale. The survey describes this as "a valuable research direction" and notes that "specialized CRL techniques for LLMs or LRMs will be crucial" — future tense throughout.

**What evidence exists in the paper.** The evidence is almost entirely in the future directions section (Section 7) and is characterized by aspirational framing rather than empirical grounding:

- Section 7.6: "Emerging research now explores shifting RL earlier in the pipeline" — the word "emerging" signals preliminary status.
- Section 7.9: "We argue that making architecture a first-class action space in RL represents an open and high-impact challenge" — "argue" and "open... challenge" signal that this is a proposal, not a report of results.
- Section 7.1: "It remains a valuable research direction for developing CRL frameworks tailored to LRMs" — "remains a... research direction" acknowledges current absence.
- No tables or figures in Section 7 present quantitative results for these directions. The section is a research agenda, not an empirical contribution.

**Mitigation status.** The survey is transparent about the speculative nature of these directions — the future tense and hedging language ("emerging," "promising," "open challenge," "valuable research direction") are appropriate. The limitation is not that the survey *overclaims* about these directions, but that their inclusion in a "Future Directions" section of an otherwise empirically-grounded survey might lead readers to overweight their proximity to practical deployment. A practitioner making near-term decisions should understand that the current validated paradigm is strictly post-training RLVR on pretrained models in verifiable domains; the extension of RL to pretraining, architecture design, and continual learning is research-stage and unvalidated at scale. The survey's presentation is honest about this, but the structure — placing these speculative directions alongside well-validated applications in coding, agents, and medicine — creates an implicit equivalence in maturity that does not hold.

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
