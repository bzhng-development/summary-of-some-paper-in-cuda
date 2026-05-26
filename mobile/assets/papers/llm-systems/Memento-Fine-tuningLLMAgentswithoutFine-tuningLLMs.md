# Memento: Fine-tuning LLM Agents without Fine-tuning LLMs

**ArXiv:** [2508.16153](https://arxiv.org/abs/2508.16153)

## 🎯 Pitch

Memento introduces a breakthrough agent architecture that enables Large Language Model (LLM) agents to continually learn and adapt by leveraging an external episodic memory and online reinforcement learning, without ever updating the LLM’s parameters. This paradigm allows agents to improve in real time by reasoning over past successes and failures, achieving state-of-the-art results on challenging deep research and QA benchmarks—providing a scalable, efficient pathway to truly adaptable AI systems without the cost or risk of model retraining.

---

## 1. Executive Summary

This paper introduces **Memento**, a memory-based learning paradigm that enables LLM agents to adapt continually through online reinforcement learning without ever updating the underlying LLM parameters — eliminating the cost of fine-tuning while retaining the ability to improve from experience. The system is instantiated as a planner–executor deep research agent and evaluated on GAIA, DeepResearcher, SimpleQA, and HLE, where it achieves top-1 on the GAIA validation set (87.88% Pass@3) and 79.40% on the private test leaderboard, while reaching 66.6% F1 and 80.4% PM on DeepResearcher — outperforming the state-of-the-art training-based system. The core mechanism is a **Case-Based Reasoning Agent** formalized as a Memory-Based Markov Decision Process (M-MDP), in which a learned neural case-selection policy (parametric CBR, implemented via a Q-function trained with cross-entropy loss against binary success/failure rewards) retrieves relevant past trajectories from an episodic case bank to guide planning, while a simpler non-parametric variant (cosine-similarity retrieval over frozen sentence embeddings) provides a competitive alternative for memory reading. The parametric memory yields a continual learning curve that improves from 80.46% to 85.44% accuracy over five iterations on DeepResearcher, and case-based reasoning adds 4.7 to 9.6 absolute percentage points on out-of-distribution tasks, establishing that memory-based adaptation transfers to unseen domains without parameter updates — but only when the underlying model already possesses relevant domain knowledge, as evidenced by near-flat performance on the hardest GAIA Level 3 tasks and minimal gains on HLE’s long-tail expert problems where the base executor lacks sufficient parametric knowledge to benefit from retrieved cases.

## 2. Context and Motivation

### The Core Problem: LLM Agents Are Either Static or Expensive to Improve

The fundamental challenge this paper tackles arises from a tension at the heart of modern LLM agent design. On one hand, we want agents that get better over time — that learn from their mistakes, adapt to new types of tasks, and accumulate wisdom from experience. On the other hand, the dominant mechanisms for improving LLM behavior are either rigidly fixed at deployment time or require computationally prohibitive retraining. The paper states this tension directly in its framing question:

> "How can we build LLM agents that learn continuously from a changing environment without the prohibitive cost of fine-tuning the underlying LLMs?"

This is not a niche concern. It touches nearly every real-world deployment of LLM agents. Consider a deep research agent deployed to answer user questions by searching the web, crawling pages, analyzing documents, and synthesizing findings. On its first day of deployment, it might struggle with a particular pattern of reasoning — say, questions that require cross-referencing information from a video's thumbnail with a spreadsheet's tabular data. A human assistant would learn from the first few such queries and handle subsequent ones more efficiently. But a typical LLM agent, once deployed, encounters each such query as if for the first time. It cannot accumulate expertise unless someone explicitly retrains it — a process that requires collecting thousands of trajectory examples, running expensive gradient updates, and carefully managing the risk of catastrophic forgetting on previously mastered tasks.

This problem grows more acute when we consider the scale of modern deployments. Agents like OpenAI's Deep Research or Google's Gemini Deep Research serve millions of queries spanning wildly different domains. The cost of fine-tuning a large model after every batch of new experiences would be enormous — not just in compute but in engineering overhead for data curation, evaluation, and deployment. Yet the alternative — never improving — means the agent operates permanently at its initial capability level, never learning from the vast stream of interaction data it generates.

The paper's framing of this as a *learning* problem rather than merely an *architecture* problem is crucial. Prior work on LLM agents has focused heavily on architectural choices: what tools to provide, how to structure the planning–execution loop, how to coordinate multiple agents. These are important, but they produce *static* agents. Memento's contribution is to ask: once the architecture is fixed, how can the agent *get better at it* through experience, without touching model weights?

### Why This Matters: The Economic and Practical Stakes

The practical significance of solving this problem extends in several directions, some explicit in the paper and some implicit in its design choices.

**The cost of fine-tuning is prohibitive for continuous deployment.** Training-based approaches to agent improvement — whether supervised fine-tuning (SFT) or reinforcement learning (RL) — require gradient updates through the entire LLM. For a model the size of GPT-4 or o3, this means thousands of GPU-hours per update cycle. Even with parameter-efficient fine-tuning (LoRA, adapters), the infrastructure overhead of maintaining training pipelines, curating datasets, and validating checkpoints creates a barrier to rapid adaptation. The paper contrasts Memento's approach with this explicitly in Section 2.1:

> "Parametric approaches update the LLM through post-training (e.g., Reinforcement Learning or supervised fine-tuning), achieving high task fidelity at the expense of considerable compute, data, and the danger of catastrophic forgetting."

The "danger of catastrophic forgetting" is particularly significant. When an agent is fine-tuned on new tasks, it may lose competence on previously mastered ones. This creates a whack-a-mole dynamic: improving performance on Task A degrades Task B, requiring expensive mitigation strategies (replay buffers, elastic weight consolidation, multi-task training). Memento sidesteps this entirely by leaving the LLM weights frozen.

**Inference-time compute is abundant; training compute is scarce.** A deep research agent already spends substantial computation at inference time — searching, crawling, planning, analyzing. Memento's key insight is that a modest additional allocation of this inference-time compute to memory operations (retrieving and scoring past cases) can substitute for expensive training-time compute. The episodic case bank grows linearly with usage but requires only vector similarity lookups (non-parametric CBR) or lightweight MLP forward passes (parametric CBR). Neither operation is free — there is an inference cost to encoding states and querying the case bank — but this cost is orders of magnitude smaller than gradient-based fine-tuning.

**Open-ended deployment demands open-ended learning.** The paper motivates its approach by reference to human cognition (Section 2.1):

> "Human intelligence relies heavily on memory systems, especially episodic memory, which supports learning from both successes and failures. Cognitive science suggests that such memories are segmented and selectively replayed to inform future decisions."

This is more than a superficial analogy. Humans do not rewire their entire cortex when they learn from a single mistake. Instead, they encode the experience as an episodic trace and retrieve similar experiences when analogous situations arise — exactly the case-based reasoning (CBR) mechanism Memento implements. For LLM agents deployed in genuinely open-ended scenarios (where the distribution of tasks shifts over time, new tools become available, or the external world changes), the ability to learn incrementally from each interaction without periodic retraining is not merely convenient — it is arguably necessary. No fixed training dataset can anticipate all the scenarios a long-lived agent will encounter.

**Self-improvement pipelines require efficient data utilization.** A growing body of work explores using LLMs to generate training data for themselves (STaR, ReST, rejection sampling fine-tuning). These pipelines are inherently expensive because they must generate large volumes of trajectories, filter for quality, and then run gradient updates. If instead the agent could directly incorporate successful trajectories into an episodic memory and retrieve them during planning — without the intermediate step of distilling them into model weights — the self-improvement loop becomes dramatically cheaper. This is not a replacement for fine-tuning but a complementary mechanism: memory-based learning can provide rapid adaptation between fine-tuning cycles, or in settings where fine-tuning is infeasible.

### Prior Approaches and Their Limitations

The paper organizes prior work into three categories, each with distinct shortcomings that Memento addresses.

#### Static Agent Frameworks: Fixed Workflows, No Adaptation

The first category encompasses the dominant paradigm in deployed LLM agents: building a carefully engineered framework with fixed reasoning workflows, fixed tool interfaces, and fixed coordination patterns. Examples include OWL (Camel-AI, 2025), AutoGen (Wu et al., 2023), and ReAct-style agents (Yao et al., 2023). These systems work well on narrow, well-specified tasks but share a fundamental limitation identified in the Introduction:

> "After deployment, such agents are static: they neither incorporate online information nor adapt to novel situations."

The word "static" is doing heavy lifting here. It means that every query the agent receives is handled by the same reasoning procedure, regardless of whether the agent has seen (and succeeded or failed on) similar queries before. If the agent makes a systematic error — say, consistently failing to extract information from PDF tables — it will make that error on every relevant query forever, because there is no mechanism for the error signal to influence future behavior.

Some frameworks attempt to address this through reflection mechanisms (Shinn et al., 2023), where the agent generates self-critiques and revises its output. But these critiques are generated de novo for each query and are not retained. The paper's critique (Section 2.3) is precise:

> "While some efforts, such as ReAct-style agents and reflective prompting pipelines, demonstrate improvement through feedback, they remain constrained by pre-defined heuristics and do not achieve true lifelong learning."

"True lifelong learning" here means the agent's performance on a task type improves cumulatively as it encounters more instances of that type. Reflection-based methods can improve a single query's answer, but the improvement doesn't carry over to the next query — the agent must rediscover the correction from scratch each time.

#### Parametric Adaptation: Expensive and Fragile

The second category — training-based approaches — attempts to solve the adaptation problem by updating the LLM weights through supervised fine-tuning or reinforcement learning on agent interaction data. Papers like START (Li et al., 2025a), ToolRL (Qian et al., 2025), and Search-R1 (Jin et al., 2025) fall into this category. The paper's assessment is direct (Section 2.1):

> "When tackling long-horizon, complex tasks, LLM agent systems must spend substantial time rolling out trajectories to gather training data, and they additionally depend on large volumes of human-annotated questions."

The limitations here are multi-dimensional:

1. **Data hunger.** Long-horizon tasks (like those in GAIA Level 3, requiring up to 50 steps) produce lengthy trajectories that are expensive to generate and difficult to learn from. The paper notes that training-based systems "depend on large volumes of human-annotated questions" — a constraint that limits scalability to domains where human annotation is feasible.

2. **Catastrophic forgetting.** Fine-tuning on new agent tasks can degrade performance on previously mastered capabilities. The paper cites this explicitly (Section 2.1), connecting to well-documented phenomena in continual learning (Li et al., 2024).

3. **Compute cost.** Full-parameter fine-tuning of large models is expensive, and even parameter-efficient methods require significant GPU infrastructure, structured training pipelines, and careful hyperparameter tuning.

4. **Static after training.** Even after fine-tuning, the model is frozen until the next training cycle. There is no mechanism for per-query improvement based on the specific cases encountered. The model learns a general policy but cannot flexibly retrieve and reuse solutions to specific past problems.

Critically, the paper does not argue that parametric adaptation is useless — it acknowledges that training-based systems achieve strong results (Table 1 shows DeepResearcher achieving 51.8% F1 / 60.5% PM). The argument is rather that parametric adaptation is *inefficient* for the continuous, incremental learning that open-ended deployment demands. It is a batch operation suitable for periodic major updates, not a mechanism for learning from each interaction.

#### Retrieval-Augmented Generation and Memory Systems: Retrieval Without Learning

The third category — RAG systems and memory-augmented agents — might seem to address the problem, since they also store and retrieve information. But the paper draws a sharp distinction (Section 2.1):

> "While modern Retrieval-Augmented Generation (RAG) systems share surface similarities with CBR, they typically query static document corpora and lack mechanisms for continual adaptation."

This is a crucial distinction. A RAG system retrieves passages from a fixed collection of documents — textbooks, Wikipedia, code repositories. The retrieval corpus is static: it doesn't grow from the agent's own experiences, doesn't contain success/failure labels, and doesn't adapt its retrieval weights based on outcomes. A memory-augmented agent like Mem0 (Chhikara et al., 2025) or MemoryBank (Zhong et al., 2024) may store interaction history, but typically for the purpose of maintaining conversation context or recalling user preferences — not for learning which past strategies were effective and should be reused.

The paper also identifies a specific failure mode in existing memory systems: the *swamping problem* (Section 2.3):

> "Most systems keep adding cases without selective curation, leading to the classic swamping problem where retrieval costs outweigh utility."

When every interaction is stored indiscriminately, the memory grows without bound. Retrieval becomes slower (more candidates to compare against) and noisier (irrelevant cases dilute the signal from relevant ones). This is not merely an efficiency concern — it directly impacts the agent's ability to learn. If the retrieval mechanism returns unhelpful cases, the LLM planner may be misled by irrelevant past experiences, producing worse plans than if it had relied solely on its parametric knowledge.

The paper also notes that several systems attempt to extract reusable knowledge from trajectories — ExpeL (Zhao et al., 2024) converts trajectories into natural-language rules, AutoGuide (Fu et al., 2024) compresses logs into conditional guidelines, Agent Workflow Memory (Wang et al., 2024) induces frequent subtask sequences. These approaches add a layer of abstraction, but they still require an explicit distillation step (converting raw trajectories into structured rules) and don't provide a principled mechanism for *selecting* which past experiences are most relevant to the current query.

### How Memento Positions Itself

Memento enters this landscape by combining ideas from four distinct traditions in a novel configuration:

1. **From Case-Based Reasoning (CBR):** the core insight that similar problems have similar solutions, and that storing successful past solutions enables reuse without re-derivation. Memento explicitly follows the classic CBR cycle: Retrieve, Reuse, Revise, Retain — the "four R's" originally formalized by Aamodt and Plaza (1994). But unlike traditional CBR systems that rely on hand-crafted similarity metrics and static case libraries, Memento makes retrieval *learnable* and the case bank *dynamic*.

2. **From Reinforcement Learning:** the formalization of case selection as a sequential decision problem within a Markov Decision Process, with rewards (task success/failure) providing the learning signal. Memento adopts the maximum-entropy RL framework (Haarnoja et al., 2018) to derive a soft Q-learning objective for the retrieval policy, connecting case-based reasoning to a principled optimization framework. This is the paper's deepest theoretical contribution: showing that the CBR cycle can be cast as policy optimization over a memory-augmented state space.

3. **From Neural Episodic Control:** the idea (from Pritzel et al., 2017) that Q-values can be estimated by kernel-weighted averaging over stored experiences, rather than requiring a fully parametric neural network to generalize across the entire state space. Memento explores both parametric (neural Q-function) and non-parametric (cosine similarity) variants of this idea, providing a spectrum between computational cost and retrieval accuracy.

4. **From LLM Agent Architectures:** the planner–executor decomposition that separates high-level strategy (what subtasks to pursue, what tools to invoke) from low-level execution (how to actually use the tools, parse outputs, handle errors). Memento applies CBR only at the planning level, leaving the executor as a general-purpose tool-user. This design choice is motivated by the observation that planning benefits most from episodic memory (since similar tasks often decompose similarly), while execution is more tool- and domain-specific.

The paper's key positioning move is to argue that these four ingredients together enable something that none achieved alone: **continuous, cost-effective adaptation that accumulates with experience**. The non-parametric variant is essentially "free" adaptation — it stores trajectories and retrieves them by similarity, requiring only vector encoding and cosine computation. The parametric variant adds a learned weighting over past cases that selectively emphasizes high-utility memories, trained with a simple binary classification loss against the observed reward signal. Neither variant touches the LLM weights.

This positioning has an important implication that the paper does not state explicitly but that emerges from the experimental results: Memento is best understood not as a replacement for fine-tuning, but as a **complement that operates on a different timescale and cost curve**. Fine-tuning provides broad, generalizable improvements that transfer across many tasks but requires batch processing and significant compute. Memory-based CBR provides rapid, instance-specific adaptation that improves performance on tasks similar to those previously encountered, with near-zero incremental cost per new case. A mature agent system would likely use both: periodic fine-tuning to expand the model's general capabilities, interleaved with continuous memory-based learning to specialize and accumulate expertise between fine-tuning cycles.

The paper also positions itself relative to the broader trend of "agentic RL" — training LLMs through reinforcement learning on interactive environments (Section 2.2):

> "Without explicit planning, deciding when and which tools to invoke remains a major bottleneck for long-horizon tasks. We model planning as a stateful MDP with explicit memory for past cases."

This framing places Memento in conversation with recent work on tool-use RL (ToolRL, GRPO-based optimization) but shifts the optimization target: rather than training the LLM to make better tool-use decisions (which requires weight updates), Memento trains a lightweight memory retrieval policy that *selects which past experiences to show the LLM as reference*, allowing the frozen LLM to benefit from past successes without being explicitly trained on them. The LLM's in-context learning ability is the bridge — the retrieved case serves as a few-shot example that biases the planner toward strategies that worked before.

## 3. Technical Approach

### 3.1 Reader Orientation

Memento is a **planner–executor deep research agent** that improves its own decision-making over time by storing past task solutions (both successful and failed) in an external memory and retrieving the most relevant ones to guide future planning — all without modifying the frozen LLMs that power it. The paper is primarily a **systems-and-theory paper** whose core idea is that case-based reasoning can be formalized as a memory-augmented Markov Decision Process with a learnable retrieval policy, enabling continual, gradient-free adaptation of LLM agents through online experience.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components arranged in an alternating loop:

1. **Planner (GPT-4.1)** — the strategic reasoning module. It receives a user query, consults the **Case Bank** for similar past tasks and their outcomes, decomposes the query into subtasks, and monitors overall progress. It is an LLM-driven Case-Based Reasoning (CBR) agent whose behavior is governed by a retrieval policy `$\mu(c \mid s, M)$` that selects which past cases to show as in-context examples.

2. **Executor (o3 or o4-mini)** — the tactical action module. It receives subtasks from the Planner, consults **Tool Memory** to decide which tools to invoke, executes tool calls via the MCP protocol, and returns results. It operates with full tool access but no direct access to the Case Bank.

3. **Case Bank `$M_t$`** — a growing collection of episodic traces `$c_i = (s_i, a_i, r_i)$` where `$s_i$` is the task state (the user query plus planning context), `$a_i$` is the plan generated, and `$r_i \in \{0, 1\}$` is a binary success/failure outcome. This bank supports two operations: **Write** (append new cases after task completion) and **Read** (retrieve relevant cases using either cosine similarity or a learned Q-function).

4. **Subtask Memory** — a text-based module that records active subtasks and their execution results, orchestrating the handoff between Planner and Executor.

5. **Tool Memory + MCP Server** — a registry of available tools (search, crawl, image/video processing, code execution, etc.) accessible via the Model Context Protocol, plus logs of tool interactions scoped per subtask.

Information flows through this alternating cycle:

1. User query enters → Planner queries Case Bank for top-K relevant past cases →
2. Planner decomposes query into subtasks, guided by retrieved cases, writing them to Subtask Memory →
3. Executor receives a subtask, queries Tool Memory, invokes appropriate tools via MCP, returns result →
4. Planner reviews accumulated results and either replans (back to step 2) or declares the task complete →
5. Upon completion, the full trajectory `$(s_t, a_t, r_t)$` is written to the Case Bank, and if using parametric CBR, the Q-function is updated online using the observed reward.

### 3.3 Roadmap for the Deep Dive

- **First**, the Memory-Based Markov Decision Process (M-MDP) formalization, because it defines the mathematical structure within which all agent behavior — retrieval, planning, execution, learning — is defined and optimized. This is the theoretical backbone.
- **Second**, the Case-Based Reasoning Agent policy (Equation 1 and its trajectory decomposition), because it shows how retrieval and LLM generation combine into a single action distribution.
- **Third**, the soft Q-learning objective for training the retrieval policy, because it derives *which* cases should be retrieved (the optimal policy has a softmax form) and *how* the Q-function can be learned through temporal difference updates.
- **Fourth**, the state-similarity kernel approximation (EC-based Q-learning), because it addresses the practical impossibility of directly learning a Q-function over natural language states by replacing function approximation with kernel-weighted averaging over stored experiences.
- **Fifth**, the implementation choices in the deep research setting — why single-step TD decays to supervised learning, why cross-entropy replaces MSE, and how the parametric and non-parametric Read/Write operations work — because these bridge theory to practice.
- **Sixth**, the tool infrastructure (MCP protocol, tool suite design), because it shapes what the executor can actually do and defines the action space of the agent.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **theoretical-and-systems paper** whose core idea is that LLM agent planning can be formalized as a memory-augmented MDP in which case retrieval is the learnable "action," soft Q-learning provides the optimization objective, and kernel-based episodic control makes the learning tractable over natural language states.

---

#### Memory-Based Markov Decision Process (M-MDP)

The M-MDP formalizes the environment in which the CBR agent operates by extending a standard MDP with an explicit memory space.

**Definition (M-MDP).** The tuple is `$\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \mathcal{M} \rangle$` where:

- `$\mathcal{S}$` is the state space — the set of all finite-length token sequences over the vocabulary `$\mathcal{V}$`. In implementation, a state `$s_t$` is the task instruction plus all accumulated planning context and execution history up to timestep `$t$`.
- `$\mathcal{A}$` is the action space — also sequences over `$\mathcal{V}$`. An action `$a_t$` is the plan (a decomposition into subtasks) generated by the Planner.
- `$\mathcal{P} : \mathcal{S} \times \mathcal{A} \rightarrow \Delta(\mathcal{S})$` is the transition dynamics — given a state and a plan, it maps to a distribution over next states. This is determined by the Executor's tool interactions and the environment's responses.
- `$\mathcal{R} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$` is the reward function — in the deep research setting, this is binary: 1 if the final answer matches the ground truth, 0 otherwise. The reward is received only at the end of the trajectory.
- `$\gamma \in [0, 1)$` is the discount factor.
- `$\mathcal{M} = (\mathcal{S} \times \mathcal{A} \times \mathbb{R})^*$` is the memory space — the set of all finite sequences of state-action-reward tuples. This is the Case Bank: a collection of past trajectories.

**What this defines:** an environment where an agent's decision at each step can depend not only on the current state but also on an arbitrarily large history of past interactions stored in `$\mathcal{M}$`. The key difference from a standard MDP is that `$\mathcal{M}$` is explicitly part of the state description — the agent's policy conditions on both `$s_t$` and `$M_t$`.

**Why this form:** a standard MDP's Markov property requires that the current state be a sufficient statistic for future decisions. In a setting where the agent should learn from experience, the raw environment state (the current query) is insufficient — the agent must also know what similar queries it has seen before and how they turned out. The memory space `$\mathcal{M}$` makes this information explicitly available to the policy. Crucially, `$\mathcal{M}$` grows over time (new cases are appended), so the agent's effective state space expands with experience, enabling continual learning without changing the underlying dynamics `$\mathcal{P}$`.

The graphical model in Figure 2 illustrates the dependencies: at each timestep `$t$`, the state `$s_t$` and memory `$M_t$` jointly influence the case selection `$c_t$` (via the retrieval policy `$\mu$`), which then influences the action `$a_t$` (via the LLM `$p_{\text{LLM}}$`), which produces a reward `$r_t$` and next state `$s_{t+1}$`, and the memory is updated to `$M_{t+1} = M_t \cup \{(s_t, a_t, r_t)\}$`.

---

#### Case-Based Reasoning Agent Policy

The CBR agent's behavior is defined by how it combines retrieval from memory with LLM generation.

**Definition (CBR Agent).** The overall policy `$\pi$` of the CBR agent, giving the probability of taking action `$a$` given state `$s$` and memory `$M$`, is:

$$\pi(a \mid s, M) = \sum_{c \in M} \mu(c \mid s, M) \, p_{\text{LLM}}(a \mid s, c)$$

where `$\mu(c \mid s, M)$` is the **case retrieval policy** — a probability distribution over the cases in the Case Bank `$M$` given the current state `$s$`. It determines which past experience to retrieve. `$p_{\text{LLM}}(a \mid s, c)$` is the **LLM action likelihood** — the probability that the frozen language model generates plan `$a$` when prompted with the current state `$s$` and the retrieved case `$c$` as an in-context example.

**What it computes:** the marginal probability of taking action `$a$`, obtained by summing over all possible cases that could be retrieved, weighted by how likely each case is to be retrieved (`$\mu$`) and how likely the resulting plan is to be generated (`$p_{\text{LLM}}$`). This is a mixture distribution: the retrieval policy picks which "expert demonstration" to show the LLM, and the LLM generates a plan conditioned on that demonstration.

**Why this form:** this decomposition separates two concerns that are optimized differently. The LLM `$p_{\text{LLM}}$` is frozen — it provides a powerful but fixed mapping from (state, example) pairs to plans. The retrieval policy `$\mu$` is the learnable component — it can be optimized online without touching model weights. The summation over `$c \in M$` reflects that the agent's behavior marginalizes over its memory: any case could be retrieved, and the overall policy is an expectation over the retrieval distribution. This formulation also naturally handles a growing memory: as `$M$` expands, the support of the mixture grows, and if `$\mu$` puts mass on useful cases, performance improves.

**Trajectory probability decomposition.** The paper provides an explicit factorization of the probability of a complete trajectory `$\tau = \{M_0, s_0, c_0, a_0, r_0, M_1, \ldots\}$`:

$$p(\tau) = \prod_{t=0}^{T-1} \underbrace{\mu(c_t \mid s_t, M_t)}_{\text{(1) Retrieve}} \; \underbrace{p_{\text{LLM}}(a_t \mid s_t, c_t)}_{\text{(2) Reuse \& Revise}} \; \underbrace{\mathbb{I}[r_t = \mathcal{R}(s_t, a_t)]}_{\text{(3) Evaluation}} \; \underbrace{\mathbb{I}[M_{t+1} = M_t \cup (s_t, a_t, r_t)]}_{\text{(4) Retain}} \; \underbrace{\mathcal{P}(s_{t+1} \mid s_t, a_t)}_{\text{(5) Transition}}$$

**What this decomposes:** the five stages of the CBR cycle as probabilistic operations. (1) RETRIEVE: the retrieval policy selects a case `$c_t$` from the current Case Bank. (2) REUSE & REVISE: the LLM generates a plan `$a_t$` conditioned on the state and retrieved case. (3) EVALUATION: the environment deterministically computes the reward. (4) RETAIN: the new case is deterministically appended to memory. (5) TRANSITION: the environment stochastically produces the next state.

**Why this decomposition matters:** it explicitly shows where learning can be applied. The terms (1) and (2) describe agent behavior — these are under the agent's control. The retrieval policy `$\mu$` is the only part we optimize; `$p_{\text{LLM}}$` is frozen. The terms (3), (4), and (5) describe environment dynamics — the reward function and memory update are deterministic (indicator functions), while the transition `$\mathcal{P}$` is stochastic. The paper notes that "the reward function and memory update can also be probabilistic in some specific cases, which we leave as future work" — this would correspond to stochastic reward signals or probabilistic memory maintenance (e.g., forgetting).

---

#### Soft Q-Learning for the CBR Agent

The retrieval policy `$\mu$` is the component we want to optimize. The paper adopts the maximum-entropy reinforcement learning framework to derive an objective and an optimal policy form.

**Objective function.** The optimization objective for the retrieval policy `$\mu$` is:

$$J(\pi) = \mathbb{E}_{\tau \sim p}\left[\sum_{t=0}^{T-1} \left[\mathcal{R}(s_t, a_t) + \alpha \mathcal{H}(\mu(\cdot \mid s_t, M_t))\right]\right]$$

where `$\mathcal{H}(\mu(\cdot \mid s_t, M_t)) = -\sum_{c \in M_t} \mu(c \mid s_t, M_t) \log \mu(c \mid s_t, M_t)$` is the entropy of the retrieval policy at timestep `$t$`, and `$\alpha$` is the entropy weight hyperparameter.

**What it computes:** the expected cumulative reward over trajectories, plus a bonus proportional to the entropy of the retrieval distribution at each step. Maximizing this objective encourages the retrieval policy to not only select cases that lead to high task rewards but also to maintain diversity in its selections — avoiding collapse to a single case.

**Why this form:** maximum-entropy RL (Haarnoja et al., 2018) provides two practical benefits. First, the entropy term acts as an exploration bonus — it prevents the policy from prematurely locking onto a small subset of cases without exploring alternatives. Second, it yields a particularly clean mathematical form for the optimal policy (a softmax over Q-values), which is both interpretable and easy to approximate.

The paper then defines the value functions under this framework. The **soft value function** for state `$(s_t, M_t)$` is:

$$V_\pi(s_t, M_t) = \sum_{c \in M_t} \mu(c \mid s_t, M_t) \left[Q_\pi(s_t, M_t, c) - \alpha \log \mu(c \mid s_t, M_t)\right]$$

where `$Q_\pi(s_t, M_t, c)$` is the **soft Q-function** — the expected future return from selecting case `$c$` in state `$(s_t, M_t)$` and then following `$\pi$` thereafter. The Q-function itself is defined as:

$$Q_\pi(s_t, M_t, c_t) = \mathbb{E}_{a \sim p_{\text{LLM}}(\cdot \mid s_t, c_t),\; s_{t+1} \sim \mathcal{P}(\cdot \mid s_t,a)}\left[\mathcal{R}(s_t, a) + \gamma V_\pi(s_{t+1}, M_{t+1})\right]$$

where `$M_{t+1} = M_t \cup \{(s_t, a_t, r_t)\}$` is the updated memory.

**What this pair computes:** `$Q_\pi$` estimates the value of retrieving a specific case, accounting for both the immediate reward (via the plan that the LLM generates from that case) and the long-term consequences (via the next state and the expanded memory). `$V_\pi$` is the expected value over the retrieval distribution.

**Why recursive:** this is the standard Bellman backup structure. The Q-value for a case depends on what happens after the LLM uses that case to produce a plan — the reward, the next state, and the fact that the memory now contains an additional case (which may be useful in future steps). The recursion means that the retrieval policy can learn to select cases that are beneficial for long-horizon tasks, not just for immediate rewards.

The expected value objective over the visitation distribution `$d_\pi$` is:

$$J(\pi) = \mathbb{E}_{(s, M) \sim d_\pi}[V_\pi(s, M)] = \mathbb{E}_{(s,M) \sim d_\pi}\left[\sum_{c \in M} \mu(c \mid s, M) \left[Q_\pi(s, M, c) - \alpha \log \mu(c \mid s, M)\right]\right]$$

**Optimal retrieval policy.** The paper derives the closed-form optimal retrieval policy by solving the constrained optimization problem (maximize `$J$` subject to `$\sum_c \mu_c = 1$`) using a Lagrange multiplier. The solution emerges as a softmax over optimal Q-values:

$$\mu^*(c \mid s, M) = \frac{\exp(Q^*(s, M, c) / \alpha)}{\sum_{c' \in M} \exp(Q^*(s, M, c') / \alpha)}$$

where `$Q^*$` is the optimal Q-function — the Q-function that would be achieved by following the optimal policy.

**What it computes:** the probability of retrieving case `$c$` is proportional to the exponential of its Q-value divided by the temperature `$\alpha$`. Cases with higher Q-values (better expected outcomes) are retrieved more often, but the softmax ensures that suboptimal cases still have non-zero probability — preserving exploration.

**Why this form:** this is the standard Boltzmann / softmax policy that arises from maximum-entropy RL. The temperature `$\alpha$` controls the exploration–exploitation tradeoff: as `$\alpha \to 0$`, the policy becomes deterministic (always picking the case with the highest Q-value — pure exploitation); as `$\alpha \to \infty$`, the policy becomes uniform (pure exploration). The paper notes that `$\alpha \to 0$` recovers standard (non-entropy-regularized) Q-learning, establishing the connection to classical RL.

**Temporal Difference (TD) learning for the Q-function.** To learn `$Q^*$` from experience, the paper applies soft Q-learning (Haarnoja et al., 2017), which updates Q-values using the TD error:

$$Q(s_t, M_t, c_t) \leftarrow Q(s_t, M_t, c_t) + \eta \left[r_t + \gamma \alpha \log \sum_{c' \in M_{t+1}} \exp\left(Q(s_{t+1}, M_{t+1}, c_{t+1})\right) - Q(s_t, M_t, c_t)\right]$$

where `$\eta$` is the learning rate, and the TD target is:
- `$r_t$`: the immediate reward (1 for success, 0 for failure),
- `$\gamma \alpha \log \sum_{c' \in M_{t+1}} \exp(Q(s_{t+1}, M_{t+1}, c_{t+1}))$`: the soft value of the next state `$(s_{t+1}, M_{t+1})$`, which is the log-sum-exp of Q-values over all cases in the updated memory, weighted by `$\gamma \alpha$`.

**What this computes:** the update nudges `$Q(s_t, M_t, c_t)$` toward the TD target, which combines the observed reward with the estimated soft value of the resulting state. If the target is higher than the current estimate, the Q-value increases (making that case more likely to be retrieved in the future); if lower, it decreases.

**Why this form:** the log-sum-exp `$\alpha \log \sum \exp(Q/\alpha)$` is the soft maximum — a smooth approximation to the hard maximum that appears in standard Q-learning. It corresponds to the value of having the option to select *any* case from the updated memory, with the entropy-regularized policy. This is the key difference between soft Q-learning and standard Q-learning: the target incorporates the value of future choice diversity.

---

#### Enhancing Q-Learning with State Similarity (Episodic Control)

The TD update above assumes we can directly store and update a Q-value for each `$(s, M, c)$` triple. This is infeasible when states are natural language strings: the state space is combinatorially large, and most `$(s, c)$` pairs are never observed. The paper addresses this by borrowing from **Neural Episodic Control** (Pritzel et al., 2017): approximate Q-values by kernel-weighted averaging over stored experiences, rather than trying to learn a parametric function over the raw state space.

**Episodic memory.** The system maintains an episodic memory `$\mathcal{D} = \{(s, c, Q)\}$` — a collection of tuples recording the state, the retrieved case, and the Q-value from each past interaction.

**Kernel-based Q estimation.** Given a query state `$s$`, the Q-value for retrieving case `$c$` from memory `$M$` is approximated as:

$$Q_{\text{EC}}(s, M, c; \theta) = \frac{\sum_{(s', c', Q') \in \mathcal{D}_c} k_\theta(s, s') Q'}{\sum_{(\hat{s}, \hat{c}, \hat{Q}) \in \mathcal{D}_c} k_\theta(s, \hat{s})}$$

where `$\mathcal{D}_c = \{(s_i, c_i, Q_i) \in \mathcal{D} : c_i = c\}$` is the subset of episodic memory entries that share the same retrieved case `$c$`, and `$k_\theta(\cdot, \cdot)$` is a **learnable kernel function** parameterized by `$\theta$` that measures the similarity between two state representations.

**What it computes:** for a fixed case `$c$`, the estimated Q-value is a weighted average of the historical Q-values `$Q'$` associated with that case, where the weight for each historical state `$s'$` is proportional to its kernel similarity `$k_\theta(s, s')$` to the current query state `$s$`. States more similar to the current query contribute more to the estimate. The denominator normalizes the weights to sum to 1 — this is a Nadaraya-Watson kernel regression estimator.

**Why this form:** this addresses the generalization problem in two ways. First, by conditioning the kernel on `$\mathcal{D}_c$` (only entries with the same case `$c$`), the estimation is case-conditional: we are asking "when this case was used in the past, how well did it work for similar states?" Second, by using a learnable kernel `$k_\theta$`, the system can learn which aspects of state similarity predict transferability of Q-values, rather than relying on a fixed similarity metric like cosine distance. The kernel parameters `$\theta$` are shared across all cases, so learning about state similarity for one case improves estimation for all cases.

**Learning the kernel via TD.** By substituting the kernel estimator `$Q_{\text{EC}}$` into the TD update rule (Equation 8), the learning objective becomes:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, c, r, s', M, M')}\left[\left(Q_{\text{EC}}(s, M, c; \theta) - \left[r + \gamma \alpha \log \sum_{c' \in M'} \exp(Q_{\text{EC}}(s', M', c'; \bar{\theta}))\right]\right)^2\right]$$

where `$\bar{\theta}$` is the target kernel network (updated periodically for stability, following standard DQN practice), `$s'$` is the next state, and `$M' = M \cup \{c\}$` is the updated memory.

**What it computes:** the mean-squared TD error — the difference between the kernel-estimated Q-value for the current `$(s, c)$` pair and the TD target computed from the observed reward plus the kernel-estimated soft value of the next state. Minimizing this loss adjusts the kernel parameters `$\theta$` so that the kernel-weighted Q-estimates become more accurate in predicting future returns.

**Why TD with kernel regression:** this hybrid approach combines the data efficiency of episodic memory (storing exact experiences and reusing them by similarity) with the generalization capability of parametric function approximation (the kernel can learn state representations that transfer across cases). Pure episodic control (using a fixed similarity metric) would require the correct metric to be hand-designed; pure deep Q-learning would require enormous amounts of data to learn a Q-function over natural language states from scratch. The kernel approach provides a middle ground: the episodic memory supplies the raw Q-values (data-efficient, no bootstrapping), and the kernel network supplies the similarity weighting (learnable, generalizes across the state space).

The paper provides the explicit gradient of this loss with respect to `$\theta$` (Equation 11), which takes the form of a weighted sum over episodic memory entries:

$$\nabla_\theta \mathcal{L}(\theta) = 2 \mathbb{E}_{(s,c,r,s',M,M')}\left[(f_\theta(s, c) - y) \sum_{i \in \mathcal{D}_c} w_i(s, c; \theta) (Q_i - f_\theta(s, c)) \nabla_\theta \log k_\theta(s, s_i)\right]$$

where `$f_\theta(s, c) = \sum_{(s_i, Q_i) \in \mathcal{D}_c} w_i Q_i$` is the kernel-weighted Q-estimate, `$w_i = \frac{k_\theta(s, s_i)}{\sum_{s_j \in \mathcal{D}_c} k_\theta(s, s_j)}$` is the normalized weight for episodic entry `$i$`, and `$y$` is the TD target.

**What this gradient does:** for each episodic memory entry `$i$`, the update pushes `$\theta$` in the direction that increases the kernel `$k_\theta(s, s_i)$` if that entry's Q-value `$Q_i$` is above the current estimate `$f_\theta(s, c)$` (making the kernel consider `$(s, s_i)$` more similar), and decreases it if below. The scaling factor `$(f_\theta(s, c) - y)$` ensures that updates are only applied when the TD error is non-zero — i.e., when the current estimate disagrees with the observed outcome.

---

#### Simplifications for the Deep Research Setting

The general M-MDP framework supports multi-step reasoning (the planner plans, the executor executes, the planner replans). However, in the deep research implementation, the CBR agent is applied **only for planning** — the planner retrieves cases and generates a full plan decomposition once. The executor then handles all subsequent tool interactions without further case retrieval.

**Why a single-step setting:** in the planner–executor architecture, the planner's state at each step is largely determined by the initial query and the accumulated execution results. The paper notes that "the CBR planner can be simplified to a single-step setting instead of a multi-step M-MDP" because CBR is applied only at the planning stage. The planner does retrieve cases and generate plans multiple times (replanning after each subtask execution), but each planning step is treated as an independent CBR decision — there is no sequential dependency where retrieving a case at timestep `$t$` affects which cases are available or relevant at `$t+1$` beyond the fact that the state changes.

**Collapse of the TD target.** In the single-step setting, there is no bootstrapping from future states: the planner's action (the plan) is evaluated based solely on whether the overall task succeeds. The TD target therefore collapses to the immediate reward:

$$y = r_t$$

because after the planner produces a plan, the executor runs to completion, and reward is observed only at trajectory end. There is no meaningful next state from which to bootstrap a value estimate, since the "next state" after planning is the beginning of execution, and the value of that state under the frozen executor is exactly the task outcome.

**From soft Q-learning to supervised learning.** With the TD target reduced to the observed reward, the learning objective simplifies dramatically. The Q-function `$Q(s, c; \theta)$` is now trained to predict the expected reward (success probability) of using case `$c$` for state `$s$`. The TD learning loss reduces to:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, c, r)}\left[(Q(s, c; \theta) - r)^2\right]$$

where the tuple `$(s, c, r)$` is stored in a replay buffer `$\mathcal{B}$` (which accumulates experiences across tasks), and `$Q$` is implemented as a neural network.

**Why switch from MSE to cross-entropy:** the paper observes that in the deep research setting, the reward signal is binary (`$r \in \{0, 1\}$`). MSE loss suffers from vanishing gradients when predictions are near 0 or 1 — the derivative `$2(Q - r)$` approaches zero at the extremes, causing slow learning for cases where the model is confidently right or confidently wrong. Cross-entropy loss, by contrast, provides stronger gradients near the boundaries. The paper therefore reformulates the objective as binary classification:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s, c, r)}\left[-r \log Q(s, c; \theta) - (1 - r) \log(1 - Q(s, c; \theta))\right]$$

where `$Q(s, c; \theta)$` is now interpreted as `$p(r = 1 \mid s, c; \theta)$` — the predicted probability that using case `$c$` as a planning reference for state `$s$` will lead to task success.

**What this computes:** the standard binary cross-entropy between the predicted success probability `$Q(s, c; \theta)$` and the observed binary outcome `$r$`. The first term `$-r \log Q$` penalizes the model when it assigns low probability to a case that actually led to success; the second term `$-(1-r) \log(1-Q)$` penalizes the model when it assigns high probability to a case that led to failure.

**Why cross-entropy is correct here:** the Q-function is now a calibrated probability predictor. Under maximum-likelihood estimation for a Bernoulli target, cross-entropy is the proper scoring rule — it is minimized when `$Q(s, c; \theta) = \mathbb{E}[r \mid s, c]$`, the true conditional expectation of success. This means the learned Q-values will be well-calibrated estimates of success probability, which is exactly what we need for the softmax retrieval policy: we want to retrieve cases proportional to `$\exp(\text{success probability} / \alpha)$`.

**Dispensing with kernel-based estimation.** In the single-step setting, the paper dispenses with the EC-based kernel estimator described in Section 3 and instead trains `$Q(s, c; \theta)$` as a parametric neural network directly. The justification is practical: the reduced state space (planner states only, not executor states) and the single-step nature means fewer interactions are needed for a parametric Q-function to generalize. The network takes encoded state and case representations as input and outputs a scalar Q-value. This is simpler than maintaining an episodic memory of `$(s, c, Q)$` tuples and performing kernel-weighted averaging, and it avoids the computational cost of querying a growing episodic memory at inference time.

---

#### Case Memory Management: Write and Read Operations

The Case Bank `$M_t$` grows online as the agent processes tasks. The paper defines two variants of memory operations: non-parametric and parametric.

**Write operation (both variants).** After completing a task at timestep `$t$`, the state `$s_t$` (the task query plus planning context), the action `$a_t$` (the generated plan), and the reward `$r_t$` (binary success/failure) are appended to the Case Bank:

$$\text{Write}(s_t, a_t, r_t, M_t) = M_{t+1} = M_t \cup \{(s_t, a_t, r_t)\}$$

The state `$s_t$` is encoded using a frozen text encoder (SimCSE, specifically) to produce a vector representation for subsequent retrieval. The action `$a_t$` and reward `$r_t$` are preserved in their original text form. The paper notes a practical detail: "the CBR planner's state at each step often contains information inherited from previous states. To avoid redundant storage, only the state, action, and reward from the final step of each trajectory are written to memory, ensuring that the case bank remains both compact and informative." This means the Case Bank stores one entry per completed task, not one entry per planning step — a design choice that keeps the bank focused on whole-task outcomes rather than intermediate replanning decisions.

**Non-parametric Read operation.** The non-parametric variant retrieves cases using a fixed similarity metric — no learned parameters:

$$\text{Read}_{\text{NP}}(s_t, M_t) = \underset{(s_i, a_i, r_i) \in M_t}{\text{TopK}} \; \text{sim}\left(\text{enc}(s_t), \text{enc}(s_i)\right)$$

where `$\text{enc}(\cdot)$` is a pretrained SimCSE encoder that maps natural language states to dense vectors, `$\text{sim}(\cdot)$` is cosine similarity, and `$\text{TopK}$` selects the `$K$` cases with the highest similarity scores.

**What it computes:** for a given query state `$s_t$`, encode it with SimCSE, compute cosine similarity against every stored state's encoding, and return the `$K$` cases `$(s_i, a_i, r_i)$` whose states are most similar to the query. The similarity is a purely geometric operation — no learning occurs.

**Why this form:** this follows the classic CBR retrieval paradigm: similar problems should have similar solutions. Cosine similarity over sentence embeddings is a computationally efficient proxy for semantic similarity, and SimCSE is a strong off-the-shelf sentence encoder trained with contrastive learning. The non-parametric variant provides a strong baseline that requires no training, making it suitable for cold-start scenarios where the agent has no accumulated Q-function.

**Parametric Read operation.** The parametric variant uses the learned Q-function `$Q(s, c; \theta)$` to rank cases:

$$\text{Read}_{\text{P}}(s_t, M_t) = \underset{c_i \in M_t}{\text{TopK}} \; Q(s_t, c_i; \theta)$$

where `$c_i = (s_i, a_i, r_i)$` is a case in the Case Bank, and `$Q(s_t, c_i; \theta)$` is the predicted success probability of using case `$c_i$` as a reference for state `$s_t$`. The `$\text{TopK}$` operator selects the `$K$` cases with the highest Q-values.

**What it computes:** for each case in the bank, evaluate the learned Q-network on the pair `$(s_t, c_i)$` to obtain a predicted success probability. Return the `$K$` cases with the highest predicted probabilities.

**Why this form:** unlike the non-parametric version, this retrieval is *adaptive* — the Q-function has learned from past experience which states are similar in terms of *outcomes*, not just in terms of *surface-level semantic similarity*. Two queries might be semantically dissimilar (asking about different topics) but structurally similar (both require cross-referencing a video with a spreadsheet), and the Q-function can learn to recognize that using the same planning strategy will succeed for both. The `$\text{TopK}$` operator is used instead of sampling from the softmax distribution `$\mu^*$` (Equation 7) to "reduce the randomness of case selection and enhance the interpretability of the agent's decision process." This is a practical deviation from the theoretical optimal policy: rather than sampling proportionally to `$\exp(Q/\alpha)$`, the system deterministically selects the top K.

**Why not sampling from the softmax:** sampling would introduce stochasticity into the planning process, making the agent's behavior harder to debug and reproduce. The paper prioritizes interpretability and determinism for the deployed system. However, this means the exploration benefit of the entropy term in the maximum-entropy objective is lost at inference time — the agent exploits the current Q-estimates without exploring alternative cases. Exploration is implicitly handled during training (the Q-function sees diverse `$(s, c)$` pairs and binary outcomes) but not during deployment.

**Parametric Write operation.** In addition to appending the case, the parametric variant also updates the Q-function online:

1. The case `$(s_t, a_t, r_t)$` is appended to the Case Bank as in Equation 12.
2. The replay buffer `$\mathcal{B}$` is extended with `$(s_t, c_t, r_t)$`, where `$c_t$` is the case that was retrieved at planning time for this task.
3. The Q-network parameters `$\theta$` are updated by minimizing the cross-entropy loss (Equation 15) over a mini-batch sampled from `$\mathcal{B}$`.

The paper also maintains a target network `$\bar{\theta}$` (updated periodically via Polyak averaging: `$\bar{\theta} \leftarrow \beta \bar{\theta} + (1 - \beta) \theta$` with update period `$K$`) for stability, though in the single-step setting this is less critical since there is no bootstrapping.

**Why online updates:** the Q-function is updated continuously as new experiences arrive, enabling the agent to adapt its retrieval behavior in real time. A task that was recently solved successfully using a particular case should immediately increase the Q-value for that `$(s, c)$` pair, making the case more likely to be retrieved for similar tasks in the future — without waiting for a batch retraining cycle.

---

#### Tool Infrastructure: MCP Protocol and Tool Suite

The Executor interacts with the external world through a standardized interface and a suite of purpose-built tools.

**Model Context Protocol (MCP).** MCP is a client-server protocol that provides a model-agnostic interface for tool invocation. The Executor (an MCP client) sends requests to MCP servers that host individual tools. Each server exposes a standardized function-calling interface: the server registers a tool description (name, description, parameter schema in JSON), the Executor determines which tool to invoke and with what arguments, the server executes the tool and returns results. This architecture separates tool implementation from agent logic — new tools can be added by spinning up new MCP servers without modifying the Executor.

**Why MCP:** the paper emphasizes three benefits: (1) unified interface — all tools are accessed through the same protocol, simplifying the Executor's tool-selection logic; (2) safety and scaling — tools run in sandboxed environments (especially code execution) that limit security risks; (3) client-server architecture — tools can be distributed across servers, enabling horizontal scaling.

**Information acquisition tools.** The search toolkit integrates multiple components for finding and retrieving external information. `searxng` is a self-hosted metasearch engine that aggregates results from Google, Bing, Duckduckgo, and Brave — providing diverse coverage without dependence on a single search provider. Retrieved candidates are re-ranked based on semantic similarity to the query context (using the same SimCSE embeddings used for case retrieval). `Crawl4AI` fetches and parses full web content from selected URLs when the search snippet is insufficient for the Executor to extract the needed information. The paper describes this as a two-stage process: "the search tool functions as a coarse filter based on keyword matching in the user query, while the crawler serves as a fine-grained mechanism to extract detailed information from the retrieved sources when necessary."

**Multimodal processing tools.** A document processing toolkit handles heterogeneous file types. Images are captioned using a vision-language model (GPT-4o). Audio is transcribed via automated speech recognition (Assembly AI). PowerPoint files are parsed slide-by-slide with embedded image descriptions. Spreadsheets are converted to row-wise text layout. Archives are unpacked. Plain text, code, JSON, XML, and Word documents are parsed into structured or Markdown formats. Videos receive natural-language summaries from a video-language model (Gemini 2.5 Pro). PDFs and unsupported formats use a fallback extraction via Chunkr AI or plain-text parsing. The toolkit provides a "unified interface for accessing and interpreting content across diverse file types and modalities."

**Reasoning tools.** The Code tool provides a sandboxed Python execution environment with a persistent workspace. Python scripts are validated against a security whitelist (allowing `numpy`, `pandas`, `torch`, and other standard libraries). Shell commands can be executed, files created and managed, and outputs inspected — all within an isolated task directory. The workspace maintains state across steps, enabling iterative development (e.g., loading data, processing it, saving intermediate results, loading them in a subsequent step). The Math tool handles basic arithmetic as a complement to the code environment.

**Tool orchestration in the Executor.** For each subtask, the Executor receives the subtask description from the Planner, consults the Tool Memory (which stores logs of previous tool invocations for the current subtask), determines which tool or sequence of tools to invoke, calls them via MCP, and updates the Tool Memory with results. The Executor can compose multiple tools within a single subtask — for example, searching for a document, crawling the result, downloading an embedded spreadsheet, parsing it, and running a computation on the extracted data. The paper notes that "unlike prior agents, Memento's executor supports rich reasoning and flexible tool composition," meaning it is not constrained to a fixed pipeline of tool calls but can dynamically decide which tools to use based on intermediate results.

---

#### Summary of Design Choices and Their Justifications

- **Planner–Executor separation with CBR only at planning:** concentrates memory-based learning on the strategic level (what to do) rather than the tactical level (how to use tools), where the space of decisions is more abstract, more stable across tool changes, and more amenable to case-based reuse.
- **Soft Q-learning with softmax optimal policy:** provides a principled theoretical framework that connects case-based reasoning to maximum-entropy RL, with the entropy term serving as an exploration mechanism during learning and the softmax providing a smooth, interpretable retrieval distribution.
- **Episodic control with learnable kernel:** addresses the generalization challenge of natural language state spaces by interpolating between stored Q-values based on learned state similarity, combining the data efficiency of non-parametric memory with the transfer capability of parametric function approximation.
- **Single-step simplification for deep research:** collapses the TD target to immediate reward, eliminating non-stationary targets and reducing the learning problem to supervised prediction of case utility — dramatically simpler and more stable than multi-step TD learning.
- **Binary cross-entropy as the Q-learning objective:** replaces MSE for the binary-reward setting because cross-entropy provides stronger gradients near the probability boundaries (0 and 1), leading to faster and more stable convergence when learning to predict success probabilities.
- **TopK deterministic retrieval in deployment:** sacrifices the theoretical exploration benefits of softmax sampling for practical interpretability and reproducibility — a deliberate engineering choice that prioritizes reliable, debuggable agent behavior.
- **SimCSE for state encoding:** provides a strong, frozen sentence embedding that captures semantic similarity without requiring training, serving as both the non-parametric retrieval metric and the input representation for the parametric Q-function.
- **Trajectory compaction (one case per task):** stores only the final planning state and outcome per task to keep the Case Bank compact and avoid redundancy from intermediate replanning steps, ensuring retrieval scales efficiently with the number of completed tasks rather than the number of planning iterations.
- **MCP protocol for tool access:** standardizes the tool interface, decouples tool implementation from agent logic (enabling independent scaling and maintenance), and provides a sandboxed execution environment for safety-critical operations like code execution.

## 4. Key Insights and Innovations

### Innovation 1: Memory-Based Learning as a Substitute for Parameter Updates in Agent Adaptation

**The idea.** The paper's most fundamental conceptual move is reframing agent improvement from a *model modification problem* to a *memory management problem*. Instead of asking "how do we update the LLM's weights to produce better plans?", Memento asks "how do we select better past experiences to show the LLM as in-context examples, such that the frozen model's *existing* generation capabilities are steered toward higher-quality outputs?"

**What the field assumed before.** The dominant paradigm — across both static frameworks and training-based systems — implicitly assumes that improving agent behavior requires changing the policy function itself. Static frameworks (ReAct, OWL, AutoGen) keep the policy fixed and accept its limitations. Training-based systems (START, Search-R1, ToolRL) alter the policy through gradient updates, accepting the computational cost and catastrophic forgetting risk. Even memory-augmented agents (Mem0, MemoryBank, A-MEM) treat memory as a *context extension* mechanism — a way to provide the LLM with more information — rather than as a *learning* mechanism that accumulates improvement over time. The field had not seriously explored whether memory operations alone, without touching model weights, could produce a consistent, compounding improvement curve.

**Why this is a reframing, not just a method.** Memento's formalization of the CBR agent policy (Equation 1) makes explicit something that was previously implicit or absent: the retrieval policy `μ(c ∣ s, M)` is a *learnable component* that sits between the frozen LLM and the environment, and optimizing `μ` alone is sufficient to improve overall agent performance. This is not an incremental improvement over existing memory-augmented agents. It is a category shift: memory goes from being a passive knowledge store to being the *primary locus of learning*. The LLM becomes analogous to a fixed reasoning engine, and the retrieval policy becomes the adaptive "strategy selector" that learns which reasoning templates (past successful plans) to deploy for which situations. This reframing has implications beyond deep research: any LLM agent that decomposes tasks and can evaluate success/failure could potentially apply this pattern, making it a general architectural principle rather than a domain-specific trick.

The significance is reinforced by what the paper *doesn't* need to do: there is no distillation step (unlike ExpeL or AutoGuide, which convert trajectories into explicit rules), no periodic retraining (unlike Search-R1 or START), and no complex credit assignment across multi-step rollouts (the single-step simplification means the learning signal is clean binary feedback per task). The entire adaptation pipeline reduces to storing trajectory traces and training a lightweight Q-function on binary outcomes — operations that are orders of magnitude cheaper than fine-tuning.

**Evidence.** The continual learning curves in Figure 1(c) and Table 4 provide the cleanest demonstration. The baseline (Memento w/o CBR) improves from 78.65% to 84.47% over five iterations — this is the gain from simply running more tasks with the frozen planner-executor system (presumably due to tool-prompt refinements or executor familiarization). The non-parametric CBR variant pushes this to 84.85%, and the parametric variant reaches 85.44%. Critically, the gap between the CBR variants and the baseline *grows* over iterations (from +0.19% at iteration 1 to +0.97% at iteration 5 for parametric vs. baseline), consistent with the interpretation that memory-based learning accumulates improvement that the CBR-free system cannot access. The ablation in Figure 1(b) further confirms this: removing CBR drops performance across all three benchmarks (DeepResearcher: from 79.7% to 72.2% PM; SimpleQA: from 95.0% to 89.7% PM; HLE: from 26.7% to 22.2% PM).

---

### Innovation 2: Formalizing Case-Based Reasoning as Policy Optimization Over a Memory-Augmented State Space

**The idea.** The paper provides a rigorous mathematical framework that connects two fields that have largely operated in isolation: case-based reasoning (CBR) from classical AI and maximum-entropy reinforcement learning from modern deep RL. The key move is formalizing the *selection of which past case to retrieve* as an "action" in an MDP, and then deriving the optimal retrieval policy as a softmax over learned Q-values. This transforms CBR from a heuristic retrieval-and-reuse procedure into a *principled policy optimization problem* with a well-defined objective and a closed-form optimal policy.

**What the field assumed before.** CBR systems have traditionally relied on hand-crafted similarity metrics (e.g., weighted feature matching, nearest-neighbor over structured case representations) with no mechanism for learning which similarity dimensions actually predict case transferability. The retrieval function is fixed by the system designer. Even recent CBR+LLM hybrids (DS-Agent, Agent-K) use fixed similarity metrics — typically cosine similarity over sentence embeddings — to retrieve past cases. The underlying assumption is that semantic similarity (how similar two task descriptions *sound*) is a good proxy for solution transferability (whether the same plan will work for both tasks). The paper's formalization exposes this assumption as testable and potentially suboptimal: the optimal retrieval policy `μ*(c ∣ s, M)` is a function of *expected future reward*, not surface similarity, and these two can diverge.

**Why this is a theoretical advance, not just an application.** Three aspects of the formalization are genuinely novel as contributions to the CBR and agent learning literatures:

First, the M-MDP (Definition 3.1) introduces memory as a *first-class component* of the state description — the agent's effective state is `(s_t, M_t)`, where `M_t` grows over time. This formalizes the intuition that an agent with experience is in a fundamentally different decision situation than a naive agent, even when facing the same raw query `s_t`. The memory is not just context; it changes the value of being in state `s_t` because it provides access to cases that may guide action selection.

Second, the derivation that the optimal retrieval policy has a softmax form (`μ*(c ∣ s, M) ∝ exp(Q*(s, M, c) / α)`) provides a theoretical answer to the question that CBR practitioners have debated for decades: *how should cases be weighted during retrieval?* The answer from maximum-entropy RL is: weight by the exponential of the expected value of retrieving that case, with temperature `α` controlling the exploration–exploitation balance. This is a principled alternative to pure similarity-based ranking, and it's *learnable from interaction outcomes* rather than requiring a predefined similarity function.

Third, the kernel-based Q-estimation approach (Equation 9) provides a bridge between episodic memory (store exact experiences, retrieve by similarity) and parametric function approximation (learn a general Q-function). The kernel `k_θ(s, s')` learns which state dimensions predict Q-value transferability — effectively learning a *task-aware similarity metric* that considers not just whether two queries sound similar, but whether they benefit from the same planning strategy. This is a specific, well-motivated application of Neural Episodic Control (Pritzel et al., 2017) to the language agent domain, and it's particularly appropriate here because natural language states are high-dimensional and semantically structured — exactly the setting where a learnable kernel can capture non-obvious analogies that a fixed embedding would miss.

**Evidence.** The comparison between parametric and non-parametric CBR in Table 4 and Figure 1(c) demonstrates that the learned Q-function does indeed capture transfer structure beyond cosine similarity. At iteration 5, parametric CBR achieves 85.44% vs. 84.85% for non-parametric CBR — a small but consistent advantage that persists across all five iterations. The OOD generalization results in Figure 1(d) further support this: the CBR variant (which uses the learned policy from training datasets) adds 4.7–9.6 percentage points over the CBR-free baseline on out-of-distribution datasets, demonstrating that the retrieval policy has captured task structure that transfers across data distributions — something a fixed similarity metric would be less likely to achieve without task-specific tuning.

---

### Innovation 3: Identifying the Boundary Between Memory-Based Learning and Parametric Knowledge as a Diagnostic Concept

**The idea.** The paper's experimental results collectively surface a finding that goes beyond "our method works well": they reveal a *sharp boundary condition* for memory-based learning that constitutes a diagnostic concept useful for the broader field. Specifically, the paper demonstrates that case-based reasoning improves performance only when the underlying LLM already possesses the *parametric knowledge* to execute the retrieved plan successfully. When that knowledge is absent — as on HLE's long-tail expert questions or GAIA Level 3's most complex tool orchestration — memory retrieval provides minimal or no benefit, because the retrieved cases are instructions the model cannot follow.

**What the field assumed before.** The default assumption in the agent memory literature is that providing relevant past experiences is helpful in proportion to their relevance — that "better memory → better performance" is a monotonic relationship. Systems like Agent-KB, Alita, and A-MEM focus on improving memory *coverage* and *retrieval quality* without distinguishing scenarios where even perfect memory would be insufficient. The implicit model is that agents fail primarily due to *information deficits* (not knowing what to do) rather than *capability deficits* (not being able to do it even when shown how).

**Why this is a diagnostic concept, not just a limitation.** The paper's component-wise ablation (Table 5) cleanly separates three sources of performance: (1) raw parametric knowledge (Offline Executor), (2) real-time tool access (Online Executor), and (3) planning with case-based reasoning (Memento w/o CBR → Memento). The *pattern across benchmarks* reveals the boundary. On SimpleQA — a factual QA task where the base model (o4-mini) has strong parametric knowledge — moving from Offline Executor (21.5% PM) to Online Executor (84.8% PM) to Memento (95.0% PM) shows large, compounding gains, with CBR adding +5.3 PM. On HLE — a long-tail academic reasoning task where the base model has weak domain knowledge — the Offline Executor achieves only 8.7% PM, the Online Executor reaches 15.8% PM (a modest gain from tool use), and Memento reaches 24.4% PM (a meaningful but still modest gain). The absolute improvement from CBR on HLE is +7.0 PM — comparable to SimpleQA (+5.3 PM), but the *ceiling* is far lower because the model cannot reliably execute even when shown successful past plans.

This pattern generalizes across difficulty levels within GAIA. The Level 3 results (Table 2) show Memento achieving 71.43% on the test set — strong but far below the 90.32% on Level 1. The difficulty gradient suggests that the bottleneck shifts from "knowing what plan to make" (which CBR helps with) to "being able to execute the plan" (which CBR cannot help with if the executor lacks the necessary tool-use sophistication). The paper acknowledges this indirectly in the Discussion (Section 6.1): "the most challenging problems increasingly rely on the model's internal reasoning to interpret and aggregate evidence from prior tool outputs, rather than simply calling more tools via MCP."

**Significance beyond this paper.** This finding provides a framework for deciding *when to invest in memory-based learning vs. parametric improvement*. If an agent's failures on a task distribution are primarily due to *planning errors* (choosing the wrong subtask decomposition, invoking tools in the wrong order), memory-based CBR is likely to help — it provides examples of correct plans. If failures are due to *execution errors* (the model cannot reliably parse PDF tables, cannot reason about video content, cannot synthesize cross-modal evidence), memory-based CBR will be ineffective regardless of retrieval quality, and the only path to improvement is a more capable executor model. This is a testable diagnostic that can guide resource allocation in agent development.

**Evidence.** The HLE breakdown by subject (Table 5a) provides the most granular evidence. In Math, CBR adds +6.0 F1 / +7.9 PM — suggesting the executor has math knowledge that CBR helps structure. In Biology/Medicine, CBR adds +4.0 F1 / +7.4 PM — the executor understands biology but benefits from planning guidance. In Engineering, CBR adds only +0.1 F1 / -1.9 PM — the executor lacks sufficient engineering knowledge for CBR to help, and the retrieved cases may even be misleading. This subject-level variation is precisely what the capability-boundary hypothesis predicts: CBR helps where the model knows the domain but struggles with task structure; it doesn't help where the model lacks domain knowledge entirely.

---

### Innovation 4: Demonstrating That Fast, Non-Deliberative Planners Outperform Slow, Deliberative Ones in Modular Agent Architectures

**The idea.** The paper's comparison of fast-thinking (GPT-4.1) vs. slow-thinking (o3) planners (Table 6) produces a counterintuitive result with implications for modular agent design: a fast, non-deliberative planner paired with a strong executor consistently outperforms a more deliberative planner, even when both use the same executor. The diagnosis is that deliberative models tend to collapse the planner–executor distinction — either answering directly without generating a plan, or producing plans so verbose and entangled with execution-level reasoning that they confuse the executor.

**What the field assumed before.** The dominant trend in LLM reasoning has been toward more deliberation: chain-of-thought, tree-of-thought, o1/o3-style extended reasoning, and "slow thinking" paradigms that invest more tokens in reasoning before acting. The intuition is straightforward: more thinking → better decisions. This intuition has been validated in single-model settings (math reasoning, code generation), where the same model both thinks and acts. But the paper's result suggests that in *modular* architectures — where a planner generates instructions for a separate executor — this intuition breaks down.

**Why this is a practical insight, not just an ablation.** The finding is significant because it identifies a *role confusion* problem that arises specifically in multi-model agent systems. When the planner is a strong reasoning model (o3), it has a tendency to *do the executor's job* — to include low-level execution details in the plan, to reason about tool-specific implementation rather than staying at the strategic level, and sometimes to skip plan generation entirely and produce a final answer. This violates the separation of concerns that the planner–executor architecture is designed to enforce, and it degrades performance because the planner's execution-level reasoning may be less informed than the executor's (the planner doesn't have access to tool memory or execution context).

The paper diagnoses this clearly: "the slow planner tends to compress solutions into a single, convoluted chain of thought, while the fast planner effectively decomposes problems into manageable sub-tasks." The fast planner, by being less capable of deep reasoning, is forced to stay in its lane — it generates a structured decomposition because that's what it's prompted to do, and it doesn't try to do the executor's job. The result is cleaner, more actionable plans that the executor can follow more reliably.

**Significance beyond this paper.** This finding has direct implications for agent architecture design. It suggests that the optimal planner model may not be the most capable model available, but rather the model that best adheres to the planning format without overstepping its role. More broadly, it challenges the assumption that "better models → better agents" in modular systems, and suggests that *role adherence* (staying within one's designated function) is a distinct capability from raw reasoning power — and that models may trade off between them. This could motivate research on training models specifically for planning roles where conciseness and structured output are prioritized over exhaustive reasoning.

**Evidence.** Table 6 shows the pattern clearly. GPT-4.1 (fast planner) + o3 (strong executor) achieves 70.91% average across GAIA levels, while o3 (slow planner) + o3 (same executor) achieves only 63.03% — a 7.88 percentage point drop. The gap is largest on Level 3 tasks (61.54% vs. 38.46%), where role confusion is most damaging because tasks are complex enough that the planner's attempt to handle execution-level reasoning leads to errors. The Qwen3-32B comparison reinforces the pattern: the fast variant (Qwen3-32B-Fast) achieves 53.94% while the slow variant (Qwen3-32B-Slow) achieves only 40.61% — a 13.33 point gap — confirming that the fast > slow pattern is not specific to GPT-4.1/o3 but generalizes across model families.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** Four benchmarks spanning distinct evaluation dimensions. **GAIA** (Mialon et al., 2023): 450 questions (150 validation, 300 test) with unambiguous answers, stratified into three difficulty levels (Level 1: ~5 steps, single tool; Level 2: 5–10 steps, multiple tools; Level 3: up to 50 steps, unrestricted tools). **DeepResearcher** (Zheng et al., 2025): a compilation of seven open-domain QA datasets — Natural Questions (NQ), TriviaQA (TQ), HotpotQA, 2Wiki, MusiQue, Bamboogle, and PopQA — with 512 examples each except Bamboogle (125 high-quality samples). **SimpleQA** (Wei et al., 2024): 4,330 fact-seeking questions evaluating factual precision. **HLE** (Phan et al., 2025): 2,500 questions across diverse academic subjects assessing frontier reasoning. The paper also designates MusiQue, Bamboogle, and PopQA as out-of-distribution (OOD) datasets due to their distinct question styles and information distributions, using NQ, TQ, HotpotQA, and 2Wiki for training.

- **Base model(s).** The **Planner** is powered by GPT-4.1 (a fast, non-deliberative model). The **Executor** is o3 for GAIA and o4-mini for DeepResearcher, SimpleQA, and HLE. Image processing uses GPT-4o, video processing uses Gemini 2.5 Pro, and audio processing uses Assembly AI. The paper also experiments with Qwen3-32B (both Fast and Slow variants) as alternative planners in an ablation (Table 6). The model choices reflect a deliberate architectural separation: the planner handles strategic decomposition, the executor handles tactical tool use, and specialized VLMs handle modality-specific processing. The choice of o3 for GAIA reflects the benchmark's higher difficulty and tool-orchestration demands, while o4-mini suffices for the other benchmarks.

- **Metrics.** Three metrics are used depending on the benchmark. **Exact Match (EM)** for GAIA: a prediction is correct only if it exactly matches the ground-truth answer after standard normalization (lowercasing, punctuation and article removal, whitespace normalization). **Macro-F1** for DeepResearcher, SimpleQA, and HLE: evaluates token-level overlap between generated and reference answers. **Partial Match (PM)** for the same benchmarks: a semantic match score computed by using GPT-4o-mini as an answer evaluator, following the same prompt as DeepResearcher (Zheng et al., 2025). PM captures semantic equivalence that EM and F1 miss due to surface-form variation. For DeepResearcher, Table 1 reports both F1 and PM scores, with the weighted average across all seven constituent datasets (weighted by dataset size: Bamboogle contributes 125 examples, all others 512). For GAIA, the paper reports Pass@1 and Pass@3 (Table 2), where Pass@k means the agent is given k attempts and the task is counted as correct if any attempt produces the exact answer.

- **Baselines.** The paper compares against multiple categories. **Prompt-based methods**: CoT (chain-of-thought prompting), CoT + RAG (retrieval-augmented generation), and Search-o1 (Web) (Li et al., 2025c). **Training-based methods**: Search-r1-base and Search-r1-instruct (Jin et al., 2025), R1-Searcher (Song et al., 2025), and DeepResearcher (Zheng et al., 2025) — all of which fine-tune the underlying LLM on agent interaction data. **GAIA leaderboard systems**: Alita, Skywork Super Agents v1.1, Langfun Agent, AWorld, Manus, OWL-Workforce, OpenAI DeepResearch, OWL-Roleplaying, Open Deep Research, Su Zero Ultra, and h2oGPTe Agent (Table 2). **SimpleQA baselines**: WebSailor (Li et al., 2025b), WebDancer, WebThinker, and DeepSeek-r1-React (Figure 4, left). For internal ablation, the paper also defines two lower baselines: **Offline Executor** (the executor model with no planner, no case memory, and no external tools — raw parametric knowledge only) and **Online Executor** (the executor with live search and MCP tools but no planner and no CBR). **Memento w/o CBR** is the full planner–executor system with episodic memory disabled, measuring the gain specifically attributable to case-based reasoning.

- **Generation budget / compute accounting.** The paper does not report generation budgets in the conventional sense (number of sampled completions), since there is no sampling-based search or best-of-N selection. Instead, compute is implicitly measured through the number of tool calls, input/output tokens, and the number of learning iterations. For the GAIA benchmark, Figure 6 reports token costs per difficulty level: Level 1 averages 26k input and 4.7k output tokens per task, Level 2 averages 48k input and 6.9k output tokens, Level 3 averages 121k input and 9.8k output tokens. Figure 5 reports the average number of tool calls per task type across difficulty levels. For the continual learning experiments (Table 4, Figure 1c), the budget is measured in **iterations** — each iteration corresponds to processing the full set of training tasks and accumulating cases in the Case Bank. The parametric CBR variant incurs additional compute for Q-function updates (mini-batch gradient steps on the replay buffer), but this cost is not quantified relative to the base inference cost. The paper does not report wall-clock time, GPU-hours, or total FLOPs for any experiment — a notable omission for a method whose primary selling point is computational efficiency relative to fine-tuning.

- **Cross-validation / statistical protocol.** There is no cross-validation, statistical significance testing, or confidence interval reporting anywhere in the paper. The GAIA validation set (150 questions) is used for case bank accumulation — the Case Bank is initialized from scratch and iteratively populated over three iterations of processing the validation set, storing both successful and failed trajectories. The GAIA test set performance (300 questions) is then evaluated using *only* the case bank accumulated during validation, with no further memory updates. This is a form of held-out evaluation, but there is no k-fold cross-validation to assess variance. For the DeepResearcher continual learning experiments, the training datasets (NQ, TQ, HotpotQA, 2Wiki) are used to populate the Case Bank, and the OOD datasets (MusiQue, Bamboogle, PopQA) are used for evaluation — this is a single train/test split with no cross-validation. The per-subject breakdown for HLE (Table 5a) reports results per category, but no aggregate uncertainty is provided. The small sample sizes — particularly for difficulty-stratified analyses (GAIA Level 3 validation has only ~20–30 questions, depending on the split) — mean that point estimates may have high variance, but this is not acknowledged or quantified.

---

### Main Quantitative Results

#### DeepResearcher: Main Results and Comparison to Training-Based Systems

The headline result is that Memento achieves a weighted average of **66.6% F1 and 80.4% PM** across the seven DeepResearcher datasets, substantially outperforming the training-based state-of-the-art system DeepResearcher (Zheng et al., 2025) at 51.8% F1 / 60.5% PM, as well as all other prompt-based and training-based baselines (Table 1).

Breaking down by individual dataset, the pattern is consistent but with notable variation in the magnitude of improvement:

- **TriviaQA**: Memento achieves 85.5% F1 / 93.9% PM vs. DeepResearcher's 78.4% F1 / 85.0% PM — a +7.1 F1 / +8.9 PM gap.
- **2Wiki**: Memento reaches 81.4% F1 / 94.1% PM vs. DeepResearcher's 59.7% F1 / 66.6% PM — a striking +21.7 F1 / +27.5 PM improvement.
- **HotpotQA**: Memento scores 66.5% F1 / 81.6% PM vs. DeepResearcher's 52.8% F1 / 64.3% PM — a +13.7 F1 / +17.3 PM gain.
- **Bamboogle**: Memento reaches 86.2% F1 / 92.8% PM vs. DeepResearcher's 71.0% F1 / 72.8% PM — a +15.2 F1 / +20.0 PM improvement.
- **Musique**: Memento achieves 40.6% F1 / 53.3% PM vs. DeepResearcher's 27.1% F1 / 29.3% PM — a more modest +13.5 F1 / +24.0 PM gap.
- **NQ**: Memento scores 42.0% F1 / 74.6% PM vs. DeepResearcher's 39.6% F1 / 61.9% PM — the smallest improvement at +2.4 F1 / +12.7 PM.
- **PopQA**: Memento reaches 64.0% F1 / 72.5% PM vs. DeepResearcher's 48.5% F1 / 52.7% PM — a +15.5 F1 / +19.8 PM gain.

The comparison against prompt-based methods is even starker: Memento nearly doubles the CoT + RAG baseline average (66.6% F1 vs. 37.7% F1). The paper interprets this as evidence that "real-time, online retrieval tools can rival or even exceed carefully curated static databases" when combined with planning and case-based reasoning.

It is crucial to note a methodological asymmetry in Table 1: Memento uses **GPT-4.1 + o4-mini (or o3)** as its backbone models, while all training-based and prompt-based baselines use **Qwen2.5 (7B)** — a dramatically smaller and less capable model. The paper states this explicitly in the table caption: "The results of prompt-based and training-based methods using Qwen2.5 (7B) are referred to DeepResearcher (Zheng et al., 2025)." This means the comparison is not between methods at equal model scale, but between (a) a memory-based system using frontier proprietary models and (b) fine-tuned systems using a 7B open-source model. The 14.8 F1 point gap between Memento and DeepResearcher conflates the effect of stronger base models with the effect of the memory-based learning paradigm. The paper provides no ablation where Memento's architecture is tested with a 7B model, nor where the training-based baselines are tested with GPT-4.1/o4-mini — making it impossible to attribute the performance difference to the method versus the model scale. This is arguably the most significant methodological limitation in the paper's experimental design.

---

#### GAIA: Validation and Test Set Results

Memento achieves **87.88% Pass@3 on the GAIA validation set** (Table 2), ranking **1st among all systems** on the leaderboard as of June 26, 2025. On the private test set, Memento achieves **79.40%**, ranking **4th overall** — behind Su Zero Ultra (80.40%) and two versions of h2oGPTe Agent (79.73% and 79.07%), but ahead of Aworld (77.08%) and all other listed open-source frameworks.

The validation set breakdown by difficulty level shows:

- **Level 1**: 96.23% Pass@3 — near-perfect performance on the easiest tier.
- **Level 2**: 90.70% Pass@3 — strong but with a visible drop from Level 1.
- **Level 3**: 61.54% Pass@3 — a sharp decline on the most complex tasks.

Comparing against the second-place system on validation (Alita at 87.27%), Memento's advantage comes primarily from Level 1 (96.23% vs. 88.68%) — a +7.55 point gap. On Level 2, the two systems are nearly tied (90.70% vs. 89.53%). On Level 3, Alita actually outperforms Memento (76.92% vs. 61.54%) — a -15.38 point deficit for Memento on the hardest tier. This Level 3 gap is notable because it suggests that Memento's memory-based planning is most effective when tasks are within a difficulty range where the executor can reliably execute plans given good guidance, but offers diminishing returns on tasks that push the executor's fundamental capabilities.

The test set pattern differs from validation in an important way: Memento's Level 3 performance on test is **71.43%**, substantially higher than the 61.54% on validation. This is unusual — typically test performance is lower than validation. The paper attributes this to the case bank being "accumulated during validation" and then applied to test, so the test set benefits from the full memory built during validation. However, the paper provides no analysis of which specific validation cases were retrieved for which test questions, making it impossible to verify this attribution. A more prosaic explanation — differences in the difficulty distribution between the validation and test Level 3 subsets — cannot be ruled out.

It is also significant that Memento is evaluated at Pass@3 while several competing systems on the leaderboard report Pass@1. The paper explicitly notes "Pass@3" in Table 2 for Memento but does not report Pass@1 for any system except in Table 6 (where Memento achieves 70.91% average Pass@1 across GAIA levels with GPT-4.1 + o3). The Pass@3 metric gives Memento three attempts per question — effectively a 3× inference budget advantage over Pass@1 systems. For fair comparison, results should ideally be reported at both Pass@1 and Pass@3 for all systems, or all at the same k. The paper does not provide a systematic Pass@1 vs. Pass@3 comparison or analyze how much of Memento's gain comes from multiple attempts vs. from the memory mechanism.

---

#### SimpleQA: Factual Accuracy

Memento achieves **95.0% accuracy on SimpleQA** (Figure 4, left), outperforming WebSailor (93.5%), WebDancer (90.5%), WebThinker (77.5%), and DeepSeek-r1-React (72.2%). The paper positions this as establishing "a new state-of-the-art over prior web-agent baselines" and demonstrating "strong factual reliability" that "substantially mitigates hallucination on straightforward single-hop queries."

The component-wise ablation in Table 5b reveals where this performance comes from. Breaking down by subject category, the Offline Executor (raw o4-mini with no tools, no planning, no CBR) achieves 21.5% PM on average. Adding online tool access (Online Executor) jumps to 84.8% PM — a massive +63.3 PM gain, confirming that web search is the dominant source of SimpleQA performance. Adding planning (Memento w/o CBR) further improves to 89.7% PM — a +4.9 PM gain. Adding CBR (full Memento) reaches 95.0% PM — a final +5.3 PM gain. The CBR contribution (+5.3 PM) is meaningful but far smaller than the tool contribution (+63.3 PM), suggesting that for factoid QA, access to live information dominates, and case-based reasoning provides a modest additional improvement — likely by helping the planner structure multi-step verification when the initial search result is ambiguous.

---

#### HLE: Frontier Academic Reasoning

Memento achieves **24.4% PM on HLE** (Figure 4, right; Table 5a), ranking second overall behind GPT-5 (25.32%) and ahead of Gemini-2.5 Pro (21.64%), o3-high (20.32%), and o4-mini-high (18.08%). The paper notes that this places Memento "within 0.92 points of GPT-5."

The per-subject breakdown in Table 5a shows extreme variation in CBR effectiveness:

- **Math**: 24.9% F1 / 16.3% PM for Memento w/o CBR → 30.9% F1 / 24.2% PM for Memento — a +6.0 F1 / +7.9 PM gain from CBR.
- **Humanities/Social Science**: 25.5% F1 / 29.2% PM → 28.4% F1 / 33.0% PM — a +2.9 F1 / +3.8 PM gain.
- **Biology/Medicine**: 10.0% F1 / 18.7% PM → 14.0% F1 / 26.1% PM — a +4.0 F1 / +7.4 PM gain.
- **Engineering**: 15.8% F1 / 8.8% PM → 15.9% F1 / 12.1% PM — a negligible +0.1 F1 gain, though PM improves by +3.3 PM.
- **Chemistry**: 17.4% F1 / 21.1% PM → 18.7% F1 / 22.7% PM — a modest +1.3 F1 / +1.6 PM gain.
- **Physics**: 18.4% F1 / 10.8% PM → 22.9% F1 / 19.1% PM — a +4.5 F1 / +8.3 PM gain.
- **CS/AI**: 25.4% F1 / 12.4% PM → 28.5% F1 / 18.5% PM — a +3.1 F1 / +6.1 PM gain.

The pattern supports the capability-boundary interpretation discussed in Section 4: CBR adds the most value in subjects where the executor (o4-mini) already has strong parametric knowledge but struggles with task decomposition (Math, Biology, Physics), and adds the least value in subjects where the executor lacks foundational knowledge (Engineering). The near-zero gain in Engineering F1 (+0.1) despite a PM improvement (+3.3) is consistent with CBR helping produce answers that are *partially* correct (improving PM) but not fully correct (no F1 gain), because the executor cannot fully execute the retrieved plan.

However, the absolute performance on HLE remains low across all categories — even the best category (Humanities/Social Science) reaches only 33.0% PM. The paper's framing of being "within 0.92 points of GPT-5" should be understood in the context that both systems are scoring in the mid-20s on a 100-point scale, and the difference is well within the range of statistical noise given the 2,500-question test set (no confidence intervals are reported).

---

#### Continual Learning Across Iterations

Table 4 and Figure 1(c) present the core evidence for continual learning. The experiment runs five iterations on the DeepResearcher dataset, where each iteration processes the training datasets and accumulates cases. Three configurations are compared:

- **Memento w/o CBR**: Starts at 78.65%, improves to 84.47% by iteration 5 — a +5.82 point gain from repeated exposure alone.
- **Memento w/ Non-Parametric CBR**: Starts at 79.84%, reaches 84.85% — a +5.01 point gain, ending 0.38 points above the CBR-free baseline.
- **Memento w/ Parametric CBR**: Starts at 80.46%, reaches 85.44% — a +4.98 point gain, ending 0.97 points above the CBR-free baseline and 0.59 points above non-parametric CBR.

The learning curves show monotonic improvement across all configurations, with the CBR variants consistently outperforming the CBR-free baseline at every iteration. The gap between parametric and non-parametric CBR is small but consistent (ranging from +0.62 at iteration 1 to +0.59 at iteration 5), suggesting that the learned Q-function provides a modest but reliable advantage over cosine-similarity retrieval.

The paper notes a practical limitation: "With only about 3k training data, the Case Bank saturates quickly. Each additional iteration, therefore, contains progressively fewer previously unseen (and thus potentially failing) cases. In our simulated, open-ended, but ultimately finite environment, we observe rapid convergence with only marginal gains after a few iterations." This saturation is visible in the diminishing returns: the improvement from iteration 4 to 5 is only +0.44 for baseline, +0.82 for non-parametric, and +0.59 for parametric — substantially smaller than the iteration 1 to 2 gains (+2.28, +2.03, +2.38 respectively).

---

#### Out-of-Distribution Generalization

Figure 1(d) reports the generalization performance on three OOD datasets (MusiQue, Bamboogle, PopQA) when the Case Bank is populated from in-distribution datasets (NQ, TQ, HotpotQA, 2Wiki) and then queried for OOD tasks:

- **Musique F1**: Memento w/o CBR baseline → Memento: +4.7 absolute points.
- **Musique PM**: +8.0 absolute points.
- **Bamboogle F1**: +7.0 absolute points.
- **Bamboogle PM**: +9.6 absolute points.
- **PopQA F1**: +5.3 absolute points.
- **PopQA PM**: +5.5 absolute points.

The gains are substantial and consistent across all three OOD datasets, ranging from +4.7 to +9.6 absolute points. The paper interprets this as evidence that "case-based reasoning enhances generalization to unseen tasks" by transferring planning strategies across task distributions. The mechanism is that the CBR policy has learned which cases (from the training datasets) are useful for which types of queries, and this learned retrieval function transfers to OOD queries that are structurally or semantically similar to training queries — even when the actual task content differs.

This result is the strongest evidence in the paper for the transferability of the learned retrieval policy. Unlike the in-distribution results (which could be explained by the Case Bank simply memorizing successful plans for exact or near-exact query matches), the OOD gains require that the retrieved cases be *relevant* to the OOD queries in a non-trivial way — the training and OOD datasets have "distinct question styles and information distributions." This suggests the CBR mechanism captures abstract planning strategies (e.g., "for multi-hop questions about entities with temporal attributes, first search for the entity, then crawl the result, then cross-reference with a second search") rather than surface-level query matching.

---

#### Hyperparameter Sensitivity: Number of Retrieved Cases (K)

Table 3 sweeps the retrieval count K on the DeepResearcher dataset:

| K | Avg F1 | Avg PM |
|---|--------|--------|
| 0 | 59.9 | 72.2 |
| 1 | 63.6 | 77.9 |
| 2 | 63.7 | 78.1 |
| 4 | **64.5** | **78.5** |
| 8 | 64.1 | 78.2 |
| 16 | 63.9 | 78.1 |
| 32 | 63.9 | 78.1 |

Performance peaks at K = 4 and then plateaus or slightly declines for larger K. The difference between K = 0 (no retrieved cases — equivalent to Memento w/o CBR) and K = 4 is substantial: +4.6 F1 and +6.3 PM. Moving from K = 4 to K = 32 produces a small decline of -0.6 F1 and -0.4 PM.

The paper's interpretation is that "CBR benefits from a small, high-quality memory, unlike few-shot prompting, where more examples often help." This is a non-obvious finding with practical implications: it means the memory system should prioritize *curation* (selecting a small set of highly relevant cases) over *coverage* (retrieving many potentially relevant cases). The degradation at high K likely occurs because irrelevant or misleading cases dilute the signal from the truly relevant ones, confusing the planner. The paper connects this to the swamping problem in classical CBR (Francis and Ram, 1993).

---

#### Component-Wise Analysis Across Benchmarks

Table 5 provides a detailed breakdown of how each architectural component contributes to performance across HLE, SimpleQA, and DeepResearcher. The consistent pattern is:

1. **Offline Executor → Online Executor** (add tools, no planning, no CBR): The effect varies dramatically by benchmark.
   - SimpleQA: +28.8 F1 / +63.3 PM — massive gain, since factual QA depends almost entirely on accessing current information.
   - HLE: +4.8 F1 / +7.1 PM — modest gain, since HLE questions require deep domain reasoning that search alone rarely resolves.
   - DeepResearcher: **-18.0 F1 / -2.1 PM** — a *negative* effect of adding tools. The paper attributes this to data contamination in the offline executor's training data: the model has memorized answers that are more accurate than what real-time search retrieves. The paper states: "simply using external knowledge can sometimes negatively affect the model, while the internal knowledge within the model plays an important role in QA tasks and can even outperform RAG." This is a notable negative result that qualifies the "tools always help" assumption common in the agent literature.

2. **Online Executor → Memento w/o CBR** (add planning, still no CBR):
   - DeepResearcher: +29.1 F1 / +11.5 PM — the largest relative gain across all benchmarks, indicating that task decomposition and tool orchestration are critical for multi-hop open-domain QA.
   - SimpleQA: +32.5 F1 / +4.9 PM — substantial F1 gain, smaller PM gain, suggesting planning helps structure search queries more precisely.
   - HLE: +11.0 F1 / +1.6 PM — modest gain, consistent with planning being helpful but insufficient when the executor lacks domain knowledge.

3. **Memento w/o CBR → Memento** (add case-based reasoning):
   - DeepResearcher: +6.7 F1 / +8.2 PM.
   - HLE: +4.5 F1 / +7.0 PM.
   - SimpleQA: +3.7 F1 / +5.3 PM.
   
   The CBR contribution is remarkably consistent across benchmarks — always positive and in the range of +4–8 points — despite the large variation in baseline performance. This consistency is evidence that CBR provides a general, additive benefit that stacks on top of whatever gains planning and tool use already provide, rather than interacting strongly with benchmark characteristics.

---

#### Planner Model Comparison: Fast vs. Slow Thinking

Table 6 compares planner–executor pairings on the GAIA validation set (Pass@1):

| Planner | Executor | Level 1 | Level 2 | Level 3 | Average |
|---------|----------|---------|---------|---------|---------|
| GPT-4.1 (fast) | o3 | 77.36% | 69.77% | 61.54% | **70.91%** |
| o3 (slow) | o3 | 73.58% | 63.95% | 38.46% | 63.03% |
| Qwen3-32B-Fast | o4-mini | 62.26% | 56.98% | 26.92% | 53.94% |
| Qwen3-32B-Slow | o4-mini | 56.60% | 36.05% | 23.08% | 40.61% |

Three patterns are clear. First, the fast planner consistently outperforms the slow planner when using the same executor — GPT-4.1 beats o3 by +7.88 points on average, and Qwen3-32B-Fast beats Qwen3-32B-Slow by +13.33 points. Second, the gap between fast and slow planners *grows with task difficulty*: for GPT-4.1 vs. o3, the gap is +3.78 on Level 1, +5.82 on Level 2, and +23.08 on Level 3. Third, the slow planner's Level 3 performance collapses dramatically — o3 as planner achieves only 38.46% on Level 3, barely above half of GPT-4.1's 61.54%.

The paper's diagnosis (Section 6.3) attributes this to role confusion: "the planner relying on the o3 model often either answers directly – skipping plan generation altogether – or produces overly verbose plans, which can mislead the executor with incomplete instructions." The Level 3 degradation is particularly instructive because these tasks require the most careful decomposition into manageable subtasks — exactly the skill that the fast planner exhibits and the slow planner violates by attempting to handle everything in a single "convoluted chain of thought."

---

### Ablation Studies and Robustness Checks

- **Number of retrieved cases (K)**: Performance peaks at K = 4 (64.5 F1 / 78.5 PM) and degrades slightly at higher K (K = 32: 63.9 F1 / 78.1 PM), confirming that a small curated memory outperforms exhaustive retrieval. Table 3.

- **Memory design: parametric vs. non-parametric CBR**: Parametric CBR achieves a small but consistent advantage over non-parametric CBR across all five learning iterations (85.44% vs. 84.85% at iteration 5), and both outperform the CBR-free baseline (84.47%). The parametric advantage grows slowly — from +0.62 at iteration 1 to +0.59 at iteration 5 — indicating that the learned Q-function provides a reliable but modest gain over simple cosine similarity. Table 4, Figure 1(c).

- **Component-wise contribution**: The incremental value of tools, planning, and CBR varies dramatically by benchmark. On DeepResearcher, adding tools *reduces* performance (-18.0 F1) due to data contamination in the offline model, while adding planning recovers dramatically (+29.1 F1) and CBR provides a further boost (+6.7 F1). On SimpleQA, tools dominate (+28.8 F1) and planning adds an additional +32.5 F1. On HLE, all components provide modest gains (tools: +4.8 F1; planning: +11.0 F1; CBR: +4.5 F1), with none sufficient to reach high absolute performance. Table 5.

- **Planner model type (fast vs. slow thinking)**: Fast planners (GPT-4.1, Qwen3-32B-Fast) consistently outperform slow planners (o3, Qwen3-32B-Slow) when paired with the same executor. The gap grows with task difficulty, reaching +23.08 points on GAIA Level 3 for GPT-4.1 vs. o3. Table 6.

- **OOD generalization across memory designs**: CBR adds +4.7 to +9.6 absolute points on OOD datasets (MusiQue, Bamboogle, PopQA) when the Case Bank is populated from in-distribution training data, demonstrating that the learned retrieval policy transfers across task distributions. Figure 1(d).

- **Continual learning dynamics**: Performance improves monotonically over five iterations for all configurations, with diminishing returns at later iterations as the Case Bank saturates (~3k training cases). The improvement from iteration 1 to 5 is +5.82 for baseline, +5.01 for non-parametric CBR, and +4.98 for parametric CBR. Table 4.

- **Negative result: data contamination in DeepResearcher**: The Offline Executor (no tools) achieves 48.8% F1 / 62.8% PM, while the Online Executor (with live search but no planning) drops to 30.8% F1 / 60.7% PM — a -18.0 F1 point degradation. This is attributed to training data contamination giving the model memorized answers that are more accurate than real-time web search results. This finding is not explored in depth (no contamination analysis is provided), but it is a practically significant caution for deploying agent systems with tool access: if the base model has strong memorized knowledge, adding web search can sometimes make answers *worse*. Table 5c.

---

### Critical Assessment

**The comparison to training-based baselines is confounded by model scale.** Table 1 compares Memento (GPT-4.1 + o4-mini) against DeepResearcher and other training-based systems using Qwen2.5 (7B). This is a ~100× difference in model scale (7B vs. GPT-4.1's estimated hundreds of billions of parameters). The paper provides no ablation where Memento's architecture is tested with a 7B planner/executor, nor where the training-based baselines are reimplemented with GPT-4.1. The 14.8 F1 point gap between Memento and DeepResearcher cannot be attributed to the memory-based learning paradigm without this control. A fair comparison would require either (a) running Memento with the same 7B base models that the baselines use, or (b) reporting baseline results with comparable frontier models. The paper's central claim of "outperforming the state-of-the-art training-based system" is therefore misleading as stated — it demonstrates that a system using much stronger base models outperforms a system using much weaker base models, which tells us little about the relative merits of memory-based vs. training-based adaptation.

**The GAIA comparison conflates Pass@3 with other systems' Pass@1.** Table 2 reports Memento at Pass@3 (three attempts per question) while many competing systems on the leaderboard report Pass@1. A Pass@3 metric gives Memento 3× the inference budget per question, and the paper provides no decomposition of how much of its top-1 ranking comes from multiple attempts versus from the memory mechanism. The paper does report Pass@1 in Table 6 (70.91% average with GPT-4.1 + o3), but this is not directly comparable to the leaderboard Pass@1 results because those systems may use different model configurations. A systematic Pass@1 vs. Pass@3 analysis would clarify this.

**No statistical significance or confidence intervals are reported.** All results in the paper are point estimates with no error bars, standard deviations, confidence intervals, or significance tests. This is particularly problematic for the smaller data splits — GAIA Level 3 validation has approximately 20–30 questions (13 questions × 3 levels approximately, based on 150 total validation questions), and HLE subcategories have unknown but likely small counts. The +0.59 point gap between parametric and non-parametric CBR at iteration 5 on DeepResearcher (85.44% vs. 84.85%) is presented as evidence of parametric superiority, but without any measure of variability, we cannot determine whether this difference is statistically reliable or within the range of sampling noise.

**The cost of memory-based learning is partially externalized.** The paper's primary motivation is avoiding the "prohibitive cost" of fine-tuning, but it never quantifies the compute cost of the memory operations relative to fine-tuning, nor the inference cost relative to a non-CBR baseline. The token cost analysis in Figure 6 reports absolute token counts per difficulty level but does not compare Memento with CBR to Memento without CBR — so we cannot determine how much additional inference compute the case retrieval and Q-function updates require. The parametric CBR variant requires storing all past cases in a vector database, encoding query states with SimCSE, running forward passes through the Q-network, and periodically updating the Q-network via mini-batch gradient steps. These costs may be small relative to fine-tuning, but they are non-zero and are never quantified.

**The Case Bank saturation limits the continual learning narrative.** The paper frames Memento as enabling "continuous, real-time learning," but the learning curves in Table 4 saturate after ~3 iterations (~3k training cases). The improvement from iteration 4 to 5 is only +0.44 to +0.82 points, compared to +2–2.4 points from iteration 1 to 2. The paper acknowledges this but does not explore whether it reflects a fundamental limitation of CBR (the case bank is finite and eventually covers all task types) or an artifact of the specific datasets (the training distribution is limited). In genuinely open-ended deployments where the task distribution continually shifts, the saturation curve might look different — but the paper provides no evidence on this point. The "continual learning" claim is therefore demonstrated only in a finite, static task distribution, not in the open-ended setting the paper motivates.

**The OOD generalization result, while strong, has limited scope.** The OOD evaluation transfers between datasets that are all within the open-domain QA category — the "distinct question styles and information distributions" are variations on a theme (multi-hop reasoning, entity-centric queries, etc.), not genuinely different task types (e.g., code generation, mathematical proof, creative writing). The paper does not test whether the Case Bank accumulated from QA tasks transfers to non-QA tasks, which would be a stronger test of the CBR mechanism's generality. The claim that CBR "enhances generalization to unseen tasks" is therefore supported only for in-domain distribution shifts, not cross-task transfer.

**No analysis of memory quality or retrieval failure modes.** The paper reports aggregate performance metrics but provides no qualitative analysis of *which* cases were retrieved, *whether* the retrieved cases were actually relevant to the query, or *when* CBR helped versus hurt. The K-sweep in Table 3 shows that larger K sometimes degrades performance, suggesting that irrelevant cases are being retrieved and causing harm — but there is no analysis of what fraction of retrievals are unhelpful, what types of queries confuse the retrieval mechanism, or whether the parametric Q-function makes systematically different retrieval errors than the non-parametric similarity function. A qualitative analysis of retrieval successes and failures would substantially strengthen the paper's claims about the mechanism.

**Missing experiment: Memento with a 7B model to isolate method from scale.** The single most informative ablation that is absent from the paper is running Memento with the same Qwen2.5 (7B) base models used by the training-based baselines. This would directly test whether the memory-based paradigm adds value *over and above* what fine-tuning achieves, at equal model scale. Without this experiment, the paper's headline comparison to DeepResearcher conflates model capability with learning method.

**Missing experiment: comparison to in-context learning baselines.** The Case Bank retrieval essentially provides K in-context examples to the planner. A natural baseline would be to replace the retrieved cases with K randomly selected successful cases, K successful cases from the most similar tasks via a simpler metric, or even K hand-crafted prompt examples. The paper does not compare case-based retrieval to these simpler in-context learning strategies, making it difficult to assess whether the retrieval policy (the paper's core contribution) provides value beyond simply having *some* examples in the prompt. The K = 0 baseline in Table 3 (no cases at all) is a weak baseline — what about K = 4 *random* successful cases?

**Missing experiment: latency and wall-clock analysis.** For a system that is motivated by practical deployment efficiency, the paper provides no latency measurements. How long does a GAIA Level 3 task take with Memento? How much of that time is spent on case retrieval vs. tool execution vs. planning? Is the parametrically retrieved case meaningfully faster or slower than the non-parametric variant? These practical metrics matter for deployment decisions but are absent.

**What the experiments DO convincingly demonstrate:** Despite these limitations, several findings are robust. (1) Case-based reasoning provides a consistent, additive improvement of +4–8 points across benchmarks, components, and memory designs — this result appears in every ablation and is unlikely to be an artifact. (2) The parametric Q-function provides a small but reliable advantage over cosine-similarity retrieval, suggesting that learning from outcomes does capture transfer structure beyond surface similarity. (3) Fast, non-deliberative planners outperform slow, deliberative ones in modular architectures — the effect is large (+7–13 points) and replicates across model families (GPT-4.1/o3 and Qwen3). (4) Memory-based adaptation produces a genuine (if bounded) learning curve without touching model weights — the improvement from iteration 1 to 5 is consistent across configurations and is larger for CBR variants than for the CBR-free baseline. (5) The OOD generalization gains demonstrate that the retrieval policy captures task structure that transfers across data distributions, not just surface-level query matching. These findings collectively establish that memory-based learning is a viable complement to parametric adaptation, even if the paper's headline comparisons overstate its advantage relative to fine-tuning.

## 6. Limitations and Trade-offs

### Model Scale Confounds the Central Comparison to Training-Based Methods

**The assumption or constraint.** The paper's headline result in Table 1 — that Memento achieves 66.6% F1 / 80.4% PM while the training-based DeepResearcher system achieves only 51.8% F1 / 60.5% PM — rests on a comparison between systems using fundamentally different base models. Memento uses GPT-4.1 as the planner and o4-mini (or o3) as the executor, while all prompt-based and training-based baselines use Qwen2.5 (7B), as stated explicitly in the Table 1 caption: "The results of prompt-based and training-based methods using Qwen2.5 (7B) are referred to DeepResearcher (Zheng et al., 2025)." The parameter count difference is likely more than 100× (GPT-4.1 is a frontier proprietary model with estimated hundreds of billions of parameters; Qwen2.5 is a 7B open-source model).

**The consequence.** The 14.8 F1-point gap between Memento and DeepResearcher cannot be attributed to the memory-based learning paradigm versus fine-tuning. It conflates two entirely separate variables: (a) the learning mechanism (memory-based CBR vs. parameter updates) and (b) the base model capability (GPT-4.1/o4-mini vs. Qwen2.5-7B). A practitioner reading Table 1 might reasonably conclude that memory-based adaptation outperforms fine-tuning by a wide margin, but the experiment provides no evidence for that conclusion. It is equally consistent with the interpretation that GPT-4.1 is simply a much stronger model than Qwen2.5-7B, and that Memento with a 7B model would underperform DeepResearcher. The paper's central claim of "outperforming the state-of-the-art training-based system" is therefore unsupported by the experimental design.

**What evidence exists in the paper.** Table 1 presents the comparison without any ablation at equal model scale. Table 5c reports an offline executor baseline for DeepResearcher (48.8% F1 / 62.8% PM using o4-mini with no tools), which is actually *higher* than DeepResearcher's reported 51.8% F1 in terms of PM (62.8% vs. 60.5%), though lower in F1. This suggests that the base model alone (o4-mini, no tools, no planning) is already competitive with or superior to the fine-tuned Qwen2.5-7B system — further evidence that model scale, not learning mechanism, drives the comparison. No experiment tests Memento with a 7B model or the training-based baselines with GPT-4.1-level models.

**Mitigation status.** The paper does not acknowledge this confound as a limitation. It is not mentioned in the main text, the discussion, or the conclusion. The omission is particularly significant because a 7B-model ablation would be straightforward to implement (swap the planner and executor models while keeping the Memento architecture intact) and would directly address the central claim.

---

### GAIA Pass@3 Inflates Performance Relative to the Leaderboard

**The assumption or constraint.** The GAIA results in Table 2 report Memento at Pass@3 — the system is given three independent attempts per question, and the question is counted as correct if *any* attempt produces the exact answer. Most systems on the GAIA leaderboard report Pass@1. The paper does not systematically report Pass@1 for Memento on the validation or test sets in a format comparable to the leaderboard (Table 6 reports Pass@1 for a planner-model ablation but not for the full leaderboard configuration). The Pass@3 metric effectively gives Memento a 3× inference budget advantage over Pass@1 systems.

**The consequence.** Memento's top-1 ranking on the validation set (87.88%) and 4th-place ranking on the test set (79.40%) are not directly comparable to systems reporting Pass@1. The improvement from Pass@1 to Pass@3 can be substantial — if each attempt has independent failure modes, Pass@3 accuracy can be far higher than Pass@1 accuracy. A system with 70% Pass@1 would achieve roughly 97% Pass@3 under independent attempts, while a system with 90% Pass@1 would achieve roughly 99.9% Pass@3. The gap between systems compresses at Pass@3, and Memento's top-1 position may partly reflect the metric rather than superior per-attempt capability. The paper does provide a Pass@1 number in Table 6 (70.91% average across GAIA levels with GPT-4.1 + o3), but this is for a specific ablation configuration, not the full leaderboard submission, and it is not directly comparable because other leaderboard systems' Pass@1 results reflect their full configurations.

**What evidence exists in the paper.** Table 2 explicitly annotates Memento with "Pass@3" in the agent name column, but the column header reads "Average score (%)" without distinguishing Pass@1 vs. Pass@3 across rows. Other systems on the leaderboard (Alita, Skywork, AWorld, Manus, OWL) do not have their Pass@k annotated in the table, making it unclear which metric each uses. Table 6 reports Pass@1 for four planner–executor configurations, ranging from 40.61% to 70.91% — substantially below the 87.88% Pass@3 on validation — but these configurations may differ from the leaderboard submission in unspecified ways (e.g., number of iterations, Case Bank size, tool availability).

**Mitigation status.** The paper acknowledges the Pass@3 metric by explicitly labeling it in Table 2, but does not discuss its implications, report a comparable Pass@1 for the leaderboard configuration, or analyze how much of the top-1 ranking is attributable to multiple attempts versus the memory mechanism. A practitioner evaluating Memento for deployment needs to know whether the gains come from the method or from a larger inference budget, and the paper does not provide that decomposition.

---

### Case Bank Saturation Limits the Continual Learning Narrative

**The assumption or constraint.** The paper motivates Memento as enabling "continuous, real-time learning" and "open-ended skill acquisition" (Abstract, Section 1). However, the continual learning experiments in Table 4 and Figure 1(c) use a finite set of approximately 3,000 training cases (from NQ, TQ, HotpotQA, 2Wiki — 512 each, minus any filtering). The Case Bank saturates quickly, as the paper itself notes in Section 5.5.3: "With only about 3k training data, the Case Bank saturates quickly. Each additional iteration, therefore, contains progressively fewer previously unseen (and thus potentially failing) cases. In our simulated, open-ended, but ultimately finite environment, we observe rapid convergence with only marginal gains after a few iterations."

**The consequence.** The learning curves in Table 4 show diminishing returns: the improvement from iteration 4 to 5 is only +0.44 points for the CBR-free baseline, +0.82 for non-parametric CBR, and +0.59 for parametric CBR — compared to +2.28, +2.03, and +2.38 respectively from iteration 1 to 2. This saturation means the "continual learning" demonstrated is actually *convergence to a fixed dataset bound*, not sustained improvement in an open-ended environment. A practitioner deploying Memento in a genuinely open-ended setting — where the distribution of tasks continually shifts or new task types appear — cannot extrapolate from these curves. The system might continue to improve (if new task types trigger new learning), plateau (if the task distribution is eventually covered), or degrade (if the Case Bank becomes so large that retrieval quality declines due to the swamping problem discussed in Section 2.3).

**What evidence exists in the paper.** Table 4 and Figure 1(c) show monotonic but decelerating improvement over five iterations. The paper acknowledges the saturation in Section 5.5.3, attributing it to the finite environment. However, no experiment tests Memento in a genuinely open-ended or distribution-shifting setting — for example, by gradually introducing new datasets over iterations, by testing on a stream of tasks with a non-stationary distribution, or by measuring whether performance degrades when the Case Bank grows beyond a certain size. The swamping problem mentioned in Section 2.3 ("most systems keep adding cases without selective curation, leading to the classic swamping problem where retrieval costs outweigh utility") is never empirically evaluated for Memento — we do not know at what Case Bank size retrieval latency or quality becomes problematic.

**Mitigation status.** The paper acknowledges the saturation explicitly in Section 5.5.3 but frames it as a property of the finite experimental environment rather than a limitation of the method. It does not propose mechanisms for addressing saturation (e.g., forgetting policies, case pruning, hierarchical memory) or test the system at larger memory scales. The claim of "open-ended skill acquisition" (Abstract) remains aspirational rather than empirically supported.

---

### No Quantification of Computational Cost Relative to the Claimed Efficiency Gains

**The assumption or constraint.** The paper's core motivation is avoiding "the prohibitive cost of fine-tuning the underlying LLMs" (Section 1). Yet it provides no quantification of Memento's actual computational cost — neither the absolute cost of memory operations (encoding, retrieval, Q-function updates) nor the cost relative to fine-tuning. The only compute-related data point in the paper is the per-task token count on GAIA (Figure 6: Level 1 averages 26k input / 4.7k output tokens, Level 3 averages 121k input / 9.8k output tokens), but this does not decompose how much of that cost is attributable to case retrieval versus planning versus tool execution, and it does not compare to a CBR-free baseline.

**The consequence.** A practitioner evaluating whether to adopt memory-based learning over fine-tuning cannot make an informed cost-benefit decision. The parametric CBR variant requires: (a) encoding each new case's state with SimCSE and storing it in a vector database, (b) for each retrieval, encoding the query state and computing cosine similarities against all stored cases (non-parametric) or running a forward pass through the Q-network for each candidate case (parametric), (c) maintaining a replay buffer and periodically running mini-batch gradient updates on the Q-network, and (d) tracking a target network with periodic Polyak averaging. Each of these operations has a cost that scales with the Case Bank size. Meanwhile, fine-tuning a 7B model with LoRA on a few thousand trajectories might cost a few hundred GPU-hours — a one-time cost that amortizes over all subsequent queries. Memento's memory operations cost accumulates *per query* and grows with the memory size. Without quantifying these costs, the paper's central efficiency claim — that memory-based learning is cheaper than fine-tuning — is an untested hypothesis.

**What evidence exists in the paper.** Figure 6 provides token counts but not cost decomposition. The paper does not report wall-clock time, GPU-hours, FLOPs, or dollar cost for any experiment. There is no comparison of Memento's inference cost to Memento w/o CBR (which would isolate the memory overhead) or to the cost of fine-tuning a comparable model. The parametric Q-function architecture (a two-layer MLP, Section 5.3) is lightweight, but its training cost depends on replay buffer size, update frequency, and batch size — none of which are specified. The non-parametric variant requires similarity search over a growing database; the paper does not specify the search algorithm (brute-force vs. approximate nearest neighbor) or its scaling properties.

**Mitigation status.** Not addressed. The paper does not acknowledge the absence of cost quantification as a limitation. The efficiency motivation (avoiding fine-tuning costs) is stated qualitatively but never tested quantitatively.

---

### HLE and GAIA Level 3 Results Reveal a Hard Capability Ceiling That Memory Cannot Breach

**The assumption or constraint.** Memento's CBR mechanism fundamentally assumes that the executor model *can* successfully execute a plan if given the right planning guidance — that failures are due to poor plan selection, not inability to execute. This assumption breaks down when the executor lacks the parametric knowledge required to perform the subtasks, regardless of how good the retrieved plan is. The paper's results on HLE and GAIA Level 3 reveal this ceiling clearly, though the paper does not frame it as a limitation of the approach.

**The consequence.** On HLE, even with full Memento (planning + CBR + tools), the absolute performance remains at 24.4% PM — a score that, while ranking second behind GPT-5, means the system fails on more than 75% of questions. The component-wise ablation in Table 5a shows why: the Offline Executor achieves 8.7% PM, the Online Executor (adding tools) reaches only 15.8% PM, and Memento reaches 24.4% PM. Tools and planning help, but the ceiling is low because the executor fundamentally lacks the deep domain expertise required for HLE's long-tail academic questions. Similarly, on GAIA Level 3 (the hardest tier), Memento achieves 61.54% Pass@3 on validation — substantially below the 96.23% on Level 1 and 90.70% on Level 2. The gap between Level 2 and Level 3 (29.16 percentage points) is far larger than the gap between Level 1 and Level 2 (5.53 points), suggesting a qualitative shift in task demands that CBR cannot bridge. For a practitioner, this means memory-based learning is most valuable in a "middle zone" where the executor is competent but the planner needs guidance — it cannot compensate for a fundamentally insufficient executor.

**What evidence exists in the paper.** Table 5a provides the per-subject breakdown for HLE, showing that CBR adds the least in subjects where the executor has the weakest parametric knowledge (Engineering: +0.1 F1) and the most where it has stronger knowledge (Math: +6.0 F1). Table 6 shows the GAIA Level 3 gap between Memento and simpler configurations. The paper notes in Section 6.1 that "the most challenging problems increasingly rely on the model's internal reasoning to interpret and aggregate evidence from prior tool outputs, rather than simply calling more tools" — implicitly acknowledging that the executor's internal capability, not the planner's strategy, becomes the bottleneck. However, this observation is not framed as a limitation of the memory-based approach.

**Mitigation status.** The paper does not explicitly identify this capability ceiling as a systematic limitation of memory-based learning. It is noted in passing (Section 6.1) but not discussed in the context of when Memento is and is not appropriate to deploy. A practitioner would benefit from clear guidance: Memento adds value when task difficulty is primarily in *decomposition* (choosing the right plan) rather than *execution* (carrying out the plan). When execution is the bottleneck, improving the executor model — through fine-tuning or a more capable base model — is the only path forward, and memory-based planning will provide minimal returns regardless of Case Bank quality.

---

### The Method Adds Value Only When the Base Executor Can Already Execute Retrieved Plans, Making It a Force Multiplier Rather Than a Capability Creator

**The assumption or constraint.** The entire CBR framework assumes that the retrieved case `c = (s_i, a_i, r_i)` contains a plan `a_i` that is *executable* by the current executor for the current query. If the executor cannot successfully carry out the plan — whether due to insufficient tool-use skill, missing domain knowledge, or inability to process certain modalities — then retrieving that case provides no benefit, regardless of how high its Q-value is estimated to be. The paper's experimental results are consistent with this assumption holding for easy-to-medium tasks (GAIA Levels 1–2, SimpleQA, most DeepResearcher datasets) and breaking down for the hardest tasks (GAIA Level 3, HLE).

**The consequence.** Memento is best characterized as a **force multiplier** for an already-capable executor, not a **capability creator**. It can make a competent executor more efficient at choosing the right approach, but it cannot enable an executor to do something it fundamentally cannot do. This has direct practical implications for deployment. An organization considering Memento should assess whether their executor model is already strong enough to handle the target task distribution (if shown the right plan). If yes, memory-based CBR can provide ongoing improvement as the Case Bank accumulates successful trajectories. If no — if the executor frequently fails even with perfect planning guidance — investment in memory infrastructure will not pay off, and resources should instead go toward executor improvement (better base model, fine-tuning on execution traces, better tool interfaces). The paper does not provide this diagnostic framing, but the data support it.

**What evidence exists in the paper.** The component-wise analysis in Table 5 shows the clearest evidence. On SimpleQA, the jump from Offline Executor (21.5% PM) to Online Executor (84.8% PM) is massive — the executor *can* answer factual questions given web access, and CBR adds a further +5.3 PM on top. On HLE, the jump from Offline (8.7% PM) to Online (15.8% PM) is much smaller — adding tools helps only modestly because the executor lacks the deep expertise to use them effectively — and CBR adds +7.0 PM but leaves absolute performance at 24.4%. The diminishing return of CBR on the hardest benchmarks is not a failure of the retrieval mechanism; it is a ceiling imposed by executor capability.

**Mitigation status.** The paper does not explicitly characterize Memento as a force multiplier or discuss the implications of this limitation for deployment decisions. The capability-ceiling pattern is visible in the data but not synthesized into guidance for practitioners. A clear statement of the boundary conditions — when CBR helps, when it doesn't, and how to diagnose which regime you are in — would substantially increase the paper's practical value.

## 7. Implications and Future Directions
- Conceptual shift: learn “what to recall,” not “how to reweight parameters”
  - By making retrieval policy the locus of learning, Memento offers a scalable path for agents to improve from experience with minimal compute. This complements RAG and parameter tuning by focusing on episodic, task-shaped memory.

- Practical applications
  - Deep research assistants that continually get better at multi-step web tasks; data/analysis copilots that record successful workflows and reuse them; multimodal research agents with stable tool execution and evolving planning priors.

- Research avenues
  - Richer credit assignment: store and value substeps or subplans, not just final outcomes; hierarchical or option-level cases.
  - Memory management: principled forgetting, summarization, and deduplication; safety filters for memory content.
  - Multi-agent settings: share case banks across agents (e.g., specialized executors) with trust and reputation signals.
  - Beyond binary rewards: graded utility signals, learned evaluators for partial success, and multi-objective criteria (accuracy, cost, latency).
  - Open, efficient stacks: reproducing competitive results with open models, smaller context windows, or distilled planners.
  - Robustness: defenses against contamination and drift when online tools and the web evolve.

Quoted highlights from the paper’s figures and tables:
- “Memento attains top-1 on GAIA validation (87.88% Pass@3) and 79.40% on the test set” (Abstract; Table 2; Figure 1a).
- “It reaches 66.6% F1 and 80.4% PM on the DeepResearcher dataset” (Abstract; Table 1; Figure 1b).
- “Case-based memory adds 4.7% to 9.6% absolute points on out-of-distribution tasks” (Abstract; Figure 1d).
- Continual learning curves show steady improvements over iterations, with parametric CBR best (Figure 1c; Table 4).
- Best K for case retrieval is small (K=4) (Table 3), suggesting “a small, curated memory yields optimal results” (Section 7).

Overall, Memento is a clear, well-scoped demonstration that principled, case-based memory with a learned retrieval policy can deliver continual learning for LLM agents—without touching the underlying LLM weights—while remaining practical and performant on demanding, tool-heavy tasks.
