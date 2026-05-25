# Scaling Agents via Continual Pre-training

**ArXiv:** [2509.13310](https://arxiv.org/abs/2509.13310)

## 🎯 Pitch

This paper introduces Agentic Continual Pre-training (Agentic CPT), a novel intermediate training stage designed to endow large language models with intrinsic agentic behaviors—such as robust tool use, multi-step reasoning, and adaptive decision-making—before post-training. By synthesizing massive, diverse agentic data (First-order and High-order Action Synthesis) and applying this method to create AgentFounder-30B, the authors demonstrate substantial new state-of-the-art performance on deep research and tool-use benchmarks, surpassing both open-source and commercial competitors. This work redefines how we build agentic AI, closing the performance gap for open-source systems and laying the foundation for more capable and reliable autonomous research assistants.

---

## 1. Executive Summary

This paper introduces **Agentic Continual Pre-training (Agentic CPT)**, a new intermediate stage between standard pre-training and post-training that builds pre-aligned agentic foundation models capable of autonomous tool use and multi-step reasoning. Starting from Qwen3 series models, the authors develop AgentFounder-30B by training on 315B tokens of synthetically generated agent data—produced through First-order Action Synthesis (FAS, which creates (question, planning, action) tuples from entity-anchored knowledge without invoking external tools) and Higher-order Action Synthesis (HAS, which expands single trajectory steps into multi-option decision-making sequences by exploring alternative reasoning and tool invocation paths at each step)—followed by downstream supervised fine-tuning. Evaluated across 10 benchmarks, AgentFounder-30B achieves new state-of-the-art open-source performance, including 39.9% on BrowseComp-en (surpassing DeepSeek-V3.1 by 9.9 percentage points), 31.5% Pass@1 on HLE (the first open-source model above 30%), and 72.8% on GAIA, while demonstrating that Agentic CPT provides consistent gains across diverse post-training configurations and exhibits logarithmic scaling behavior with both data volume and model size, establishing that embedding agentic reasoning capabilities directly into the foundation model through continual pre-training substantially outperforms approaches that rely solely on post-training—but that these benefits concentrate on information retrieval tasks rather than knowledge-intensive reasoning where base model comprehension remains the bottleneck.

## 2. Context and Motivation

### The Core Problem: General-Purpose Foundation Models Are Poor Starting Points for Agentic Training

The central problem this paper addresses is deceptively simple but has profound implications for how we build autonomous AI agents: **when you take a general-purpose foundation model and try to teach it agentic behaviors through post-training alone, you get systematically worse results than if the model had been pre-exposed to agentic patterns during its formative training stages.** This manifests concretely as a large performance gap between open-source implementations and proprietary systems like OpenAI's Deep Research — a gap the authors quantify early in the paper by noting that even the best open-source models at the time (DeepSeek-V3.1 at 30.0% on BrowseComp-en) trail OpenAI's Deep Research (51.5%) by over 20 percentage points.

Behind this gap lies what the authors identify as a **fundamental optimization tension**. When you start with a general-purpose foundation model — one trained primarily on static text from the web, code repositories, books, and academic papers — and then attempt to post-train it for agentic tasks, the model must simultaneously:

1. **Learn new capabilities**: tool invocation patterns, multi-step planning, adaptive reasoning in response to environmental feedback (e.g., search results, web page content, tool failures), and long-horizon decision-making across potentially dozens of steps.
2. **Align to expert demonstrations**: reproduce the specific behavioral patterns shown in supervised fine-tuning trajectories, which are necessarily limited in coverage.

These two objectives collide because the foundation model lacks what the authors call **agentic inductive biases** — baked-in patterns and expectations about how agents operate. Without these biases, SFT and RL training must compensate by providing dense, explicit supervision for every aspect of agentic behavior. But this supervision is inherently limited: agentic trajectories are long, complex, and expensive to generate at scale, and human-designed reward signals or trajectory labels can only cover a tiny fraction of the possible behavioral space. The model consequently memorizes specific demonstration patterns rather than developing generalizable decision-making capabilities.

This is not merely a theoretical concern. The authors ground their argument in two specific failures of post-training-only approaches:

**SFT's reliance on complete, high-quality trajectory data makes comprehensive coverage infeasible.** Deep research agents must navigate vast policy spaces: for any given problem, there are many possible sequences of search queries, web pages to visit, code to execute, and reasoning steps to take. High-quality supervised data (where a human or another model has demonstrated an effective trajectory) necessarily covers only a minuscule fraction of this space. When SFT trains the model to imitate these trajectories, it locks the model into reproducing specific patterns rather than flexibly exploring the space of possible solutions. This is the classic coverage problem in imitation learning, made acute by the combinatorial explosion of possible agent behaviors.

**RL training depends on trajectory-level delayed feedback that is both sparse and coarse.** Complete agent trajectories can involve dozens of tool calls and reasoning steps, but the only clear signal — did the agent arrive at the correct final answer? — comes at the very end. Intermediate steps are difficult to evaluate independently: is a particular search query good or bad? Is this intermediate reasoning step on the right track? Without reliable step-level evaluation (which the authors note is "challenging" and risks model collapse if done naively), RL must propagate credit through long, noisy sequences, making the optimization signal weak and high-variance.

The authors crystallize this diagnosis in a key claim from Section 1:

> "Fundamentally, general-purpose foundation models lack agentic inductive biases, forcing post-training to simultaneously learn capabilities and alignment, creating inherent optimization conflicts."

This framing is important because it reframes the agent training problem from "we need better post-training methods" (the dominant paradigm in open-source agent research) to "we need better starting points for post-training." It is an argument about the training pipeline architecture, not about any specific algorithm.

### Why This Problem Matters

The significance of this gap extends across multiple dimensions:

**The open-source ecosystem is systematically disadvantaged for agentic applications.** If the only pathway to strong agentic performance is starting from proprietary foundation models and applying proprietary post-training recipes (as with OpenAI's Deep Research), then the open-source community is locked out of a critical capability. The authors state that "post-training approaches building upon general-purpose foundation models consistently underperform in agentic tasks, particularly in open-source implementations" — the "particularly" is telling, because open-source implementations cannot rely on undisclosed proprietary training data, infrastructure, or model internals. Agentic CPT provides a recipe that is fully reproducible with open-source base models and synthetic data.

**The deployment economics of AI agents depend on model quality.** Deep research agents that autonomously browse the web, execute code, and synthesize information are among the most commercially valuable AI applications — they power research assistance, market intelligence, scientific literature review, and complex decision support. But their economic viability depends on reliability: an agent that produces correct answers 30% of the time may be an interesting research artifact, but one that produces correct answers 50–70% of the time starts being genuinely useful for knowledge workers. The 10–20 percentage point gaps the paper documents between open-source models and the proprietary frontier represent the difference between prototype and product.

**The problem reveals a gap in our understanding of the training pipeline.** The standard LLM training paradigm — pre-train on broad corpora, then post-train for specific capabilities — was developed in an era where "capabilities" meant things like instruction following, dialogue, and summarization. These are fundamentally different from agentic capabilities: they operate in a single-turn, static-input paradigm where the model generates one output and the interaction ends. Agentic capabilities require the model to maintain coherent behavior across extended interactions with dynamic environments, where each action generates new information that must be integrated into subsequent decisions. The standard pipeline implicitly assumes that post-training is sufficient to bridge this gap; this paper provides systematic evidence that it is not, and proposes a concrete architectural fix.

**The finding has implications for how we think about model specialization.** If agentic CPT works — if exposing models to agentic patterns during (continual) pre-training substantially improves downstream agentic performance — then it suggests that other specialized capabilities might benefit from analogous treatment. Should we have "code CPT" before code-specific fine-tuning? "Math CPT" before mathematical reasoning training? The paper opens a research direction around domain-specific continual pre-training as a general approach to capability specialization, beyond the traditional domain-adaptive pre-training that focuses on knowledge acquisition rather than behavioral capability acquisition.

### Where Prior Approaches Fall Short

The paper identifies limitations in existing approaches along several axes:

#### Post-Training-Only Agent Development

The dominant approach in open-source agent research (and much of the commercial space) is to start from a strong general-purpose model — Qwen, DeepSeek, LLaMA — and apply post-training (SFT, RL, or both) with agent-specific data. The paper cites a long list of such works: WebThinker, ASearcher, WebSailor, WebShaper, AFM, MiroThinker, DeepDiver, WebExplorer, DeepDive, and others (Section 4.1). These works have made genuine progress, achieving impressive results on BrowseComp, GAIA, and other benchmarks. But they all share the same architectural assumption: that the foundation model is a generic substrate, and agentic capabilities are entirely acquired during post-training.

The paper argues this assumption is the bottleneck. Even the most sophisticated post-training methods — iterative data generation, knowledge-graph-based question synthesis, multi-agent distillation, reinforcement learning with carefully designed rewards — are fighting an uphill battle because the foundation model they start from has never seen agentic behavior patterns. Every tool call format, every search-result-interpretation pattern, every multi-step planning structure must be learned from scratch during post-training, competing for model capacity with the need to align to specific demonstration trajectories.

The evidence for this claim is both implicit (the persistent gap between open-source and proprietary models, despite open-source methods being published and reproducible) and explicit (the paper's own experiments in Section 3.3 showing that AgentFounder-Base consistently outperforms Qwen3-Base across three different SFT configurations). If post-training methods alone were sufficient, we would expect the gap to close as methods improve — but it has not, suggesting a structural limitation.

#### Trajectory Data Scarcity and Waste

A related but distinct problem is the inefficiency of trajectory data usage. When post-training agent models, both SFT and RL generate substantial volumes of trajectory data. But quality evaluation operates at the trajectory level: a trajectory either succeeds (reaches the correct answer) or fails. Failed trajectories are typically discarded entirely. Successful trajectories might be used once for SFT or provide a single positive reward signal for RL. The authors frame this starkly:

> "These methods rely heavily on trajectory-level delayed feedback for quality assessment, which results in numerous trajectories being either discarded entirely or utilized only once when they fail to meet stringent quality thresholds. This coarse-grained evaluation approach leads to significant waste of the learning signals embedded within real trajectories."

This is a striking observation. A failed trajectory might contain 47 excellent steps and one critical mistake. A successful trajectory might contain 49 mediocre steps and one inspired insight. But trajectory-level feedback treats the first as worthless and the second as uniformly good, discarding the rich signal available at the step level. The authors note that while "step-level evaluation could theoretically provide better leverage of these signals, precisely assessing intermediate steps remains challenging" — and naive incorporation of uncertain step-level rewards into SFT or RL "risks model collapse."

This creates a data efficiency crisis. Generating high-quality agent trajectories is expensive — it requires running the actual tools (search APIs, web scraping, code execution) with real API costs and latency. If most generated trajectories are discarded and the remainder are used once, the cost per effective training sample is enormous. The paper's HAS method (Section 2.3) is explicitly designed to address this by extracting rich training signal from discarded and underutilized trajectories, transforming them from single-use demonstrations into multi-decision learning experiences.

#### Knowledge-Only Continual Pre-training

Continual pre-training (CPT) is a well-established technique in the LLM literature, but prior work has focused almost exclusively on **knowledge acquisition**. The typical use case is domain adaptation: take a general-purpose model and continue pre-training it on domain-specific corpora (medical literature, legal documents, financial reports) to improve performance on domain tasks. The paper cites Ke et al. (2023), which proposes continual domain-adaptive pre-training for knowledge transfer, and Gupta et al. (2023) and Parmar et al. (2024), which provide guidelines for effective CPT data mixing and learning rate scheduling. Recent work has scaled these approaches to billion-parameter models.

But these approaches are about teaching the model **what to know**, not **how to behave**. The training objective is the same next-token prediction on static text — the domain changes, but the underlying capability being trained (language modeling) does not. Agentic CPT is fundamentally different: it aims to teach the model patterns of behavior — tool invocation formats, reasoning structures, decision-making sequences — through the same next-token prediction objective but with synthetically constructed data that embeds these behavioral patterns.

This distinction is crucial and the authors make it explicitly in Section 2.2.1:

> "Conventional continual pre-training focuses on knowledge adaptation, particularly domain-specific knowledge acquisition. In contrast, Agentic CPT targets the adaptation of agentic capabilities, which are domain-agnostic abilities that transcend specific domains and enable universal tool utilization and multi-step reasoning."

The fact that agentic capabilities are domain-agnostic (you need to search and reason regardless of whether the topic is astrophysics or zoology) is both what makes Agentic CPT broadly useful and what makes it a departure from prior CPT work. Knowledge CPT is narrow but deep; Agentic CPT must be broad but behavioral.

#### The Missing Link: No Agentic Foundation Models

The paper positions its contribution as filling a gap that the field has largely overlooked. The standard training pipeline has two stages: pre-training (acquire knowledge) and post-training (align to tasks). The authors argue that for agentic capabilities, this two-stage pipeline is insufficient and propose a three-stage pipeline: pre-training → Agentic CPT → post-training. Section 1 states:

> "pathways toward developing agentic foundation models themselves remain largely unexplored"

This is the key positioning claim. The paper is not proposing a better SFT method, a better RL algorithm, or a better way to generate post-training data. It is proposing that agentic capabilities should be partially acquired **before** post-training begins, through a dedicated intermediate training stage that embeds agentic behavioral patterns into the model's weights using next-token prediction on synthetic agent data. Post-training then only needs to refine and specialize these pre-existing capabilities, rather than building them from scratch.

The closest prior work the paper acknowledges is GLM-4.5, which "incorporates synthetic agent trajectories during mid-training" (Section 1, footnote). But the authors frame this as an exception that proves the rule — a single model among dozens that has partially explored this direction, and one whose incorporation is limited relative to the systematic Agentic CPT approach proposed here. Similarly, recent open-source general models (Kimi-K2, GLM-4.5, DeepSeek-V3.1) have "begun emphasizing enhanced agentic capabilities, yet the systematic exploration of continual pre-training for agent development remains limited" (Section 4.1).

### How This Paper Positions Itself

The paper positions Agentic CPT as a **pipeline-level innovation**, not a method-level innovation within an existing pipeline. This is a strong claim that requires defending on multiple fronts:

**Against the "better post-training is enough" camp.** The authors must show that Agentic CPT provides gains beyond what can be achieved by improving post-training alone. The experiment in Section 3.3 is designed precisely for this: starting from the same Qwen3 base, models that receive Agentic CPT before post-training consistently outperform those that go directly to post-training, across three different SFT configurations. The average gains (5.75%, 6.13%, 6.45% for SFT-A, SFT-B, SFT-C respectively) demonstrate that the benefit of Agentic CPT is not specific to any particular post-training recipe — it is a genuine pipeline improvement.

**Against the "just do more pre-training" camp.** If agentic capabilities could be acquired through standard pre-training on more web data, there would be no need for Agentic CPT. But the paper argues that standard pre-training corpora do not contain agentic behavioral patterns in sufficient density or quality. Web text contains tool invocations only incidentally (e.g., API documentation, code examples) and rarely contains the kind of multi-step decision-making sequences that characterize agent trajectories. Agentic CPT's synthetic data — FAS for planning and reasoning patterns, HAS for multi-step decision-making — is explicitly constructed to provide these patterns at high density and quality.

**Against the "knowledge CPT is the same thing" camp.** The paper draws a clear line between knowledge acquisition (the goal of traditional domain-adaptive CPT) and behavioral capability acquisition (the goal of Agentic CPT). The data for Agentic CPT is not simply domain-specific text; it is synthetically constructed to embed specific behavioral patterns — tool call formats with reasoning, multi-step planning structures, option selection and decision-making sequences. The training objective is the same next-token prediction, but the signal being transmitted is fundamentally different.

**Within the broader scaling laws framework.** The paper connects to the scaling laws literature (though not as centrally as some other works) by demonstrating that Agentic CPT exhibits logarithmic scaling behavior with both data volume (Section 3.5.2) and model size (Section 3.5.1). This positions Agentic CPT not as a one-off technique but as a principled scaling dimension — you can predictably improve agentic performance by increasing the Agentic CPT budget, just as you can predictably improve language modeling by increasing the pre-training budget.

### What Makes This Paper Different from Prior Agent Training Work

To understand the paper's contribution, it helps to contrast it with the typical approach in open-source agent development. A representative workflow might be:

1. Start with Qwen-72B or DeepSeek-V3.
2. Construct challenging questions using knowledge graphs, entity relationships, or iterative question generation.
3. Generate agent trajectories for those questions by having a strong model (or the model itself) interact with tools.
4. Filter trajectories based on correctness.
5. Fine-tune the model on the filtered trajectories (SFT).
6. Optionally apply RL to further optimize.

This is a **post-training-centric** workflow. The model's exposure to agentic patterns happens entirely in steps 5–6, after the model's core representations have already been formed.

AgentFounder's workflow is:

1. Start with Qwen3-30B-A3B-Base.
2. Generate FAS data: construct entity-anchored knowledge memories, synthesize diverse QA pairs, and generate planning actions and reasoning actions **without executing tools**.
3. Generate HAS data: take discarded and successful trajectories from prior post-training, expand each step into multiple alternative reasoning-action options, and reformulate them as decision-making sequences.
4. Perform Agentic CPT Stage 1: continue pre-training on 200B tokens of mixed FAS+HAS data with 32K context.
5. Perform Agentic CPT Stage 2: continue pre-training on 100B tokens of high-quality HAS data with 128K context.
6. Apply post-training (SFT-A, SFT-B, or SFT-C) on the resulting AgentFounder-Base model.

The critical difference is steps 2–5: **agentic patterns are embedded into the model before any task-specific fine-tuning occurs.** The post-training in step 6 only needs to adapt these pre-existing capabilities to specific task formats, rather than building them from scratch. This is the pipeline innovation the paper claims as its primary contribution.

The paper also distinguishes itself by operating at a larger scale than most prior CPT work. While Ke et al. (2023) worked with million-parameter models, and Ça˘gatay Yıldız et al. (2025) scaled to billion-parameter models, AgentFounder uses a 30B-parameter MoE architecture and trains on 315B tokens of agent data. This scale is necessary because agentic capabilities must generalize across diverse domains and task types — a small-scale CPT run might learn domain-specific patterns but would not develop the broad, transferable agentic behaviors that Agentic CPT targets.

### The Practical Motivation: API Costs and Environmental Constraints

A pragmatic motivation that runs throughout the paper is the **cost of generating agent training data with real tool execution.** Every search API call, every web page visit, every code execution costs money and time. Generating one complete agent trajectory might involve 50+ tool invocations, each of which incurs API latency and (for commercial APIs like Google Search or Jina Reader) monetary cost. Scaling this to billions of tokens of training data would be economically infeasible.

The paper's data synthesis methods — both FAS and HAS — are explicitly designed to operate **without external tool invocations during data generation.** FAS generates planning and reasoning actions using only the LLM's internal knowledge and the constructed knowledge statements (Section 2.2.2). HAS expands trajectories synthetically by generating alternative reasoning-action options at each step, without actually executing those alternatives (Section 2.3). This allows the generation of 315B tokens of training data entirely offline, without API costs or latency constraints. The authors state this motivation clearly:

> "Both synthesis approaches operate without external tool invocations, enabling large-scale data generation in offline environments without API costs."

This is not just an implementation convenience — it is what makes Agentic CPT economically viable as a pre-training-scale operation. If every token of agent data required real tool execution, the approach would be restricted to well-funded industrial labs and would not contribute to the paper's goal of advancing open-source agent capabilities.

### Summary of the Gap and the Proposed Solution

The paper identifies a specific structural gap in the LLM training pipeline for agents: the absence of agentic inductive biases in general-purpose foundation models forces post-training to shoulder the entire burden of capability acquisition and behavioral alignment simultaneously, creating optimization conflicts that limit performance. Prior approaches have attempted to address this through better post-training methods (more sophisticated data generation, better reward design, improved RL algorithms) but have not closed the gap with proprietary systems because they operate within the same pipeline architecture.

Agentic CPT fills this gap by inserting a dedicated intermediate training stage between pre-training and post-training, where the model is exposed to high-density agentic behavioral patterns through next-token prediction on synthetically generated data. This creates a pre-aligned agentic foundation model that already understands tool invocation formats, multi-step planning structures, and decision-making patterns before post-training begins, allowing post-training to focus on refinement and specialization rather than capability acquisition from scratch. The evidence for this approach comes not from a single experiment but from a systematic demonstration that AgentFounder-Base outperforms Qwen3-Base across multiple post-training configurations, across multiple benchmarks, and across multiple model scales.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is primarily a **systems and training methodology paper** whose core idea is that inserting a dedicated "Agentic Continual Pre-training" (Agentic CPT) stage between standard pre-training and post-training produces foundation models with baked-in agentic behavioral patterns, which then serve as substantially better starting points for downstream agent-specific fine-tuning than general-purpose foundation models. The system being built is a **deep research agent** — an LLM that can autonomously search the web, visit pages, execute code, parse documents, and synthesize information across dozens of steps to answer complex, open-ended questions — and the paper's contribution is not a new post-training algorithm but a **pipeline-level architectural change** in how the training process is structured.

### 3.2 Big-Picture Architecture (Diagram in Words)

The AgentFounder training pipeline has five major stages, with the first two being inherited from the Qwen3 family and the latter three being the paper's contributions:

1. **Standard Pre-training (Qwen3)** — The base language model (e.g., Qwen3-30B-A3B-Base) is pre-trained on broad corpora (web text, code, books, academic literature) using standard next-token prediction with cross-entropy loss. This provides general knowledge and language capabilities but contains essentially no agentic behavioral patterns.

2. **Agentic CPT Stage 1 (200B tokens, 32K context)** — The model continues pre-training on a heterogeneous mixture of synthetically generated agent data (FAS + HAS) and knowledge reasoning corpora, still using next-token prediction. This stage embeds foundational agentic patterns: tool invocation formats, planning structures, multi-step reasoning chains, and decision-making sequences. The 32K context window handles most agent trajectories, though some longer HAS data may be truncated.

3. **Agentic CPT Stage 2 (100B tokens, 128K context)** — The model undergoes a second round of continual pre-training focused on high-quality HAS data with extended context windows. The 128K context length allows the model to learn from complete long-horizon trajectories without truncation, developing sophisticated understanding of complex action spaces and long-range planning strategies. Computational cost motivates the two-stage split: training everything at 128K would be prohibitively expensive, so Stage 1 handles the bulk of data at 32K, and Stage 2 refines capabilities on the subset that benefits from extended context.

4. **Supervised Fine-Tuning (SFT)** — The resulting AgentFounder-Base model is fine-tuned using a strategically proportioned mixture of general instruction data and React-style agent trajectory demonstrations. The paper experiments with three SFT configurations (SFT-A, SFT-B, SFT-C) that differ in data ordering, mixing ratios, and trajectory formatting, demonstrating that Agentic CPT's benefits are robust to the post-training recipe.

5. **Inference with Tools** — The final AgentFounder-30B model operates under a single-agent React paradigm with five tools (Search, Visit, Python Interpreter, Google Scholar, File Parser), constrained to 128 tool calls maximum and 128K context length, with decoding parameters temperature=0.85, repetition penalty=1.1, top-p=0.95.

Information flows as follows: raw web/text corpora → FAS data synthesis (entity-anchored memory construction → multi-style QA generation → planning action synthesis + reasoning action synthesis) → Agentic CPT Stage 1; discarded/successful post-training trajectories → HAS data synthesis (step-level scaling → contrastive decision-action synthesis) → Agentic CPT Stage 2; the two CPT stages produce AgentFounder-Base → SFT with agent demonstrations → AgentFounder-30B for inference.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal training objective (Equation 1) that unifies all stages, since every phase — standard pre-training, both CPT stages, and SFT — uses the same core next-token prediction loss with cross-entropy.
- **Second**, the FAS data synthesis pipeline (First-order Action Synthesis), because it produces the bulk of the 200B token Stage 1 corpus and introduces the critical design principle of generating agentic training data without executing real tools. This covers: knowledge-to-question transformation, planning action synthesis with multi-reference generation, and reasoning action synthesis with two-step logical deduction.
- **Third**, the HAS data synthesis pipeline (Higher-order Action Synthesis), because it produces the high-quality data for Stage 2 and represents the paper's solution to trajectory data waste — transforming single-use trajectories into multi-decision learning experiences through step-level scaling and contrastive synthesis.
- **Fourth**, the two-stage training strategy and its rationale, because the split between 32K and 128K context windows is a key engineering decision driven by computational constraints and data characteristics.
- **Fifth**, the post-training configurations (SFT-A/B/C), because they demonstrate the adaptability claim — that Agentic CPT benefits are robust to downstream fine-tuning choices.
- **Sixth**, the inference setup (tools, hyperparameters, constraints), because it defines the operational envelope within which all benchmark results are obtained.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **training pipeline paper** whose core idea is that embedding agentic behavioral patterns into foundation model weights through next-token prediction on synthetically generated agent data — before any task-specific fine-tuning — creates a pre-aligned starting point that substantially improves downstream agent performance compared to starting from a general-purpose foundation model.

---

#### The Unified Training Objective: Next-Token Prediction with Cross-Entropy Loss

Every training stage in the AgentFounder pipeline — standard pre-training, Agentic CPT Stage 1, Agentic CPT Stage 2, and SFT — uses the identical training objective: standard next-token prediction with cross-entropy loss. The paper states this explicitly for the pre-training phase and confirms that Agentic CPT "follows the same next-token prediction paradigm as Eq. 1" (Section 2.1).

The training objective is the standard autoregressive language modeling loss:

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_{t+1} | x_1, x_2, \ldots, x_t)$$

where $T$ is the sequence length, $x_1, \ldots, x_T$ is the tokenized training sequence, and $P(x_{t+1} | x_1, \ldots, x_t) = \text{softmax}(W_o h_t)$ is the model's predicted probability for the actual next token $x_{t+1}$ given all previous tokens. The hidden state $h_t$ at position $t$ is the model's internal representation, and $W_o$ is the output projection matrix that maps from the hidden dimension to the vocabulary size.

**What it computes:** For each position in the training sequence, the model produces a probability distribution over its entire vocabulary of which token should come next. The loss is the negative log of the probability assigned to the token that actually appears next — summed across all positions. If the model assigns probability 0.9 to the correct next token, it contributes $-\log(0.9) \approx 0.105$ to the loss; if it assigns probability 0.01, it contributes $-\log(0.01) \approx 4.605$. The total loss is minimized when the model perfectly predicts every token in the sequence.

**Why this form:** This is the maximum-likelihood objective for autoregressive sequence modeling. It has the crucial property that it is **dense** — every token in the training sequence provides a learning signal, unlike RL or trajectory-level SFT where feedback is sparse (one signal per complete trajectory). This density is what makes pre-training-scale data consumption (200B+ tokens) feasible: the model receives gradient updates from every token it processes, not just from trajectory endpoints. For Agentic CPT, this matters because the synthetic agent data embeds behavioral patterns across thousands of tokens per sequence — tool call formats, reasoning structures, planning decompositions — and every token of those patterns contributes to the gradient, whereas post-training approaches that use trajectory-level correctness signals would only provide a single gradient contribution per complete trajectory. Using the same objective across all stages also means there is no distribution shift in the optimization landscape when transitioning from CPT to SFT.

The hidden state $h_t$ and output projection $W_o$ are standard Transformer components not described further in the paper — the key point is that the model's prediction for token $x_{t+1}$ depends on all previous tokens $x_1, \ldots, x_t$ through the self-attention mechanism, allowing it to condition on arbitrarily long context (up to the context window limit).

---

#### First-Order Action Synthesis (FAS): Generating Agentic Data Without Tools

**The Scale Challenge.** The paper must generate hundreds of billions of tokens of training data that contain agentic behavioral patterns. The naive approach — take questions, have an LLM execute real tool calls (search, browse, code execution) to solve them, and record the resulting trajectories — would be economically infeasible because:

1. **API costs are prohibitively expensive**, particularly for search engine APIs (e.g., Google Search) and web access APIs (e.g., Jina Reader). Each trajectory might require 50+ tool calls; generating 200B tokens of trajectories would require millions of real API invocations.
2. **Trajectory generation is slow**. Each tool call involves network latency; generating complete trajectories at scale would take impractical wall-clock time.
3. **Tool responses are non-deterministic**. Search results change over time; web pages get updated or removed. This makes the generated data non-reproducible and potentially inconsistent.

FAS addresses this by generating agentic training data **without executing any external tools during data creation.** The core insight is that the early stages of agent behavior — problem analysis, decomposition, information requirement identification, and first-step action prediction — can be synthesized using only the LLM's internal knowledge and the structured knowledge base used to construct the questions. The paper explicitly states that "both synthesis approaches operate without external tool invocations, enabling large-scale data generation in offline environments without API costs" (Section 1).

FAS consists of three sub-components: (1) contextual scenario construction via knowledge-to-question transformation, (2) planning action synthesis, and (3) reasoning action synthesis. Each is described in detail below.

##### Knowledge-to-Question Transformation: Building Diverse Training Contexts

**Motivation.** Agentic capabilities are domain-agnostic: an agent needs to search and reason regardless of whether the question is about aviation orders, Olympic visitor statistics, or bedbug outbreaks. To ensure the acquired capabilities transfer broadly, the training data must span many domains. The paper states this principle explicitly in Section 2.2.1:

> "Since these abilities must function effectively across diverse application scenarios, this capability adaptation necessitates training data spanning multiple domains to ensure broad transferability and applicability of the acquired skills."

**Data sources.** The paper collects data from four categories:
- **Discarded trajectories from post-training datasets**: Trajectories that failed quality checks during previous SFT/RL training runs. These contain real tool invocation patterns and search results but were not used for post-training because the final answer was incorrect.
- **Historical tool invocation results**: Search queries and their corresponding responses, web page content retrieved during prior agent runs. These represent real information retrieval interactions.
- **Publicly available corpora**: CommonCrawl, Wikipedia, and similar sources. These provide broad but static knowledge coverage.
- **Offline Wikipedia data**: Structured encyclopedic knowledge that can be used for entity extraction.

**Phase 1: Entity-Anchored Open-World Knowledge Memory.** The raw data sources are transformed into a structured memory system where **entities serve as keys** mapping to collections of **declarative knowledge statements** about those entities. This is deliberately not a traditional knowledge graph with fixed schemas and typed relationships between entities — the paper explicitly contrasts with DBpedia (Auer et al., 2007) and Wikidata (Vrandečić & Krötzsch, 2014) to emphasize that the approach does not focus on inter-entity relationships.

Instead, the goal is to maximize the **density of knowledge statements per entity** through reformulation. Raw text is processed to extract factual claims about specific entities, preserving critical contextual information: temporal markers (when something happened), sources (where the information came from), and original stylistic features. The paper provides a concrete example:

> "web data containing 'The number of tourist arrivals in France increased from 3,793 thousand in May 2025 to 4,222 thousand in June' can be reformulated as: ('France', 'Tourist arrivals in France reached 4,222 thousand in June 2025')"

The reformulation is important because it standardizes the format (entity, claim) while preserving the specificity that makes questions challenging. A traditional knowledge graph might store only "France → capital → Paris"; the entity-anchored memory stores "France → Tourist arrivals reached 4,222 thousand in June 2025" — a claim about France that happens to be true at a specific time and can be used to construct temporally-anchored questions.

The memory is "living" — it continuously expands as new search results and web access outcomes are processed, meaning the system can incorporate fresh, time-sensitive information that would not appear in static knowledge bases. This alignment with "the information distribution of the internet world" (Section 2.2.1) is important because deep research agents must answer questions about recent events, not just encyclopedic facts.

**Phase 2: Multi-Style Question Synthesis.** Given the entity-anchored memory, the system synthesizes diverse questions by:
1. Sampling entity clusters along with their associated knowledge statements.
2. Generating questions spanning multiple styles: factual retrieval (direct lookup of a specific fact), numerical computation (calculations based on retrieved numbers), multi-hop reasoning (combining information from multiple entities), and synthesis tasks (integrating diverse information into a coherent answer).

The key innovation is that questions are generated from **single-entity contexts** or **implicit cross-entity links** induced by the high density of statements per entity, rather than requiring explicit relationship construction between entities. The paper contrasts this with WebSailor's approach, which "requires explicit relationship construction between entities" (Section 2.2.1). By instead relying on the richness of reformulated knowledge statements to create natural knowledge intersections, the approach yields questions that are both more reliable (less likely to contain fabricated relationships) and more novel (less constrained by pre-defined relationship types).

The paper provides an extended example of a synthesized question from the entity "Paris" with three knowledge statements:

> **Question:** "At the biennial aerospace marketplace named after the city whose pyramid-fronted museum recorded high single-digit millions of visitors during a period of global athletic celebration, and where the year before a citywide nuisance led authorities to convene transit operators, which buyer placed a perfectly balanced commitment with firm orders equal to options?"
> **Answer:** Riyadh Air

This question demonstrates the multi-hop reasoning style that FAS targets. Solving it requires: (1) identifying the city from the pyramid-fronted museum clue (the Louvre, hence Paris), (2) identifying the biennial aerospace marketplace (Paris Air Show), (3) identifying the year from the citywide nuisance clue (2023 e-scooter debate, hence 2025 Air Show), and (4) finding the specific buyer at the 2025 Paris Air Show whose order had equal firm orders and options (Riyadh Air with 25+25). The paper notes that "since these facts are recent and fluid, reliable resolution typically requires external retrieval with search tools" — the question is designed to be answerable only through agentic search, not through the model's static knowledge.

This transformation from static knowledge to dynamic problem-solving contexts is the critical bridge: the knowledge statements provide the ground truth needed for reject sampling (discussed below), while the question format necessitates the kind of information retrieval, integration, and reasoning that agentic training aims to develop.

##### Planning Action Synthesis: Multi-Reference Generation Without Tool Execution

**The scalability insight.** Once questions are constructed, the system needs corresponding "actions" — the reasoning steps and tool invocations an agent would perform to answer the question. The paper observes that "the initial analysis of complex problems by LLMs typically involves problem decomposition, information requirement identification, and solution planning, which inherently constitutes high-quality planning data" (Section 2.2.2). Moreover, "the quality of first-step reasoning exhibits strong positive correlation with final task completion rates."

This observation enables a critical simplification: **generate only the initial reasoning and first-step action, without executing those actions or continuing the trajectory.** The generated data thus consists of (question, planning_analysis, proposed_action) tuples, where the proposed action might be a search query, a web page visit, a code execution, or a direct answer.

**Multi-reference generation.** Rather than generating a single reasoning-action pair per question, FAS generates **K diverse problem analyses** with corresponding first-step action predictions. The paper draws a connection to multi-reference learning (Zheng et al., 2018; Banerjee & Lavie, 2005) as inspiration for this approach. The initial formulation generates K analyses for the same question by varying generation parameters (e.g., temperature) to produce diverse outputs.

However, the authors identify two limitations with this naive approach:
1. **The generated analyses may still be similar** despite parameter-based diversity encouragement — genuine diversity in reasoning paths is hard to achieve by temperature tuning alone.
2. **The question text gets repeated K times** in the training data, wasting tokens on content that is not the optimization target (the model already knows how to process questions; it needs to learn actions).

**The improved strategy: question-level diversity expansion.** Instead of generating K reasoning-action pairs for a single question, FAS generates reasoning-action data for **K different questions that share the same knowledge memory but differ in style.** Specifically, from the same entity-anchored knowledge statements, the system constructs multiple stylistically distinct questions — factual retrieval, numerical computation, multi-hop reasoning, synthesis — and generates one reasoning-action pair per question. This approach "better covers training contexts and explores the potential reasoning-action space more comprehensively" because different question styles naturally elicit different reasoning approaches and action types. The paper adopts this question-level diversity expansion as the planning action synthesis method in FAS.

**Reject sampling with knowledge alignment verification.** Since FAS does not execute the proposed actions, it cannot verify correctness through end-to-end trajectory outcomes. However, because the knowledge statements used to construct the questions are accessible, the system can implement **reject sampling based on knowledge alignment verification.** Specifically, an LLM-as-Judge evaluates whether the generated reasoning and proposed actions "have a high probability of acquiring the required knowledge" (Section 2.2.2). The paper reports in Appendix B.1 that this filtering:

> "removes 43.5% of problematic samples, increasing retained trajectory accuracy from 50% to 82%"

The filtered-out errors are dominated by semantic issues (Content Inconsistency: 26.2%, Search Necessity: 6.9%, Logic Discontinuity: 5.7%) rather than syntactic errors (Invalid Tool: 1.2%), indicating that FAS maintains structural validity while requiring refinement in semantic alignment. The 82% accuracy among retained samples is sufficient for Agentic CPT because the training objective is next-token prediction, not trajectory correctness — the model learns the format and structure of reasonable planning actions, and occasional inaccuracies in the generated actions are tolerated because they do not propagate (no actual tools are executed).

##### Reasoning Action Synthesis: Two-Step Logical Deduction

**The information synthesis scenario.** A critical capability for deep research agents is synthesizing acquired information into final answers or reports. After the agent has gathered relevant information through tool invocations, it must integrate that information through logical inference — drawing conclusions that are supported by the gathered facts but not trivially extractable from any single source.

This type of reasoning is fundamentally different from mathematical reasoning. The paper characterizes it as "logic-based inference grounded in factual information, requiring a balance between divergent thinking and convergent thinking guided by contextual clues, while being difficult to verify through formal methods" (Section 2.2.3). Unlike math problems where answers can be verified through computation, logical inference from factual information requires judgment about relevance, consistency, and sufficiency of evidence.

**Two-step synthesis procedure.** Reasoning action synthesis generates data that teaches the model this synthesis capability. The procedure operates on the Question-Answer pairs constructed in the knowledge-to-question transformation (Section 2.2.1), and importantly, the model is **prohibited from invoking any external tools** during both steps:

**Step 1:** The LLM decomposes the question $Q$ into multiple sub-questions, then leverages its internal knowledge to generate reasonable speculations and answers for each sub-question, producing preliminary answer $A_1$.

**Step 2:** Given the question $Q$ and its mapped requisite knowledge (the knowledge statements used to construct the question), the LLM refines answer $A_1$, corrects logical errors, and generates the final answer $A_2$.

**Why two steps?** The paper explains that if the question and necessary knowledge are provided together from the start, "the model tends to mechanically utilize the given knowledge as intermediate reasoning nodes rather than simulating an authentic thinking process" (Section 2.2.3). The two-step design forces the model to first attempt reasoning from its own knowledge (simulating the divergent thinking phase where an agent considers multiple hypotheses), then confront its preliminary answer with the ground-truth knowledge (simulating the convergent thinking phase where an agent verifies conclusions against evidence). The first step develops the capacity for hypothesis formation; the second step develops the capacity for evidence-based correction.

The paper provides an extended example showing how the reasoning proceeds through clue-by-clue analysis:

> "(Clue1: The Location): First, the question mentions ..., which presents several potential candidates: the Louvre in Paris with its iconic glass pyramid entrance, .... However, the crucial filtering criterion is the high visitor numbers 'during a period of global athletic celebration.' ... Among these candidate cities, only Paris simultaneously possesses both a pyramid-fronted museum and hosted a global athletic celebration..."

The reasoning chains are structured as explicit, stepwise logical deductions — the model learns to identify clues, enumerate candidates, apply filtering criteria, and converge to a unique answer. This structured reasoning format is what the next-token prediction objective will learn to reproduce.

**Reject sampling.** After generating the answer $A_2$, an LLM-as-Judge evaluates alignment between $A_2$ and the ground-truth answer from the QA pair. If the final answer is correct, the entire reasoning process contained in $A_2$ is considered reliable and included in the training data. If incorrect, the sample is discarded. This creates a corpus of high-quality logical reasoning chain-of-thought data anchored to verifiable ground truth.

The paper emphasizes that this "logical deduction capability constitutes a fundamental competency required by deep research agents throughout the entire problem-solving lifecycle" — it is not just about final answer synthesis but about the capacity for evidence-based reasoning that agents need at every step, from deciding which search query to issue next to determining whether gathered information is sufficient.

---

#### Higher-Order Action Synthesis (HAS): Transforming Trajectories into Decision-Making Training Data

**The trajectory waste problem.** During post-training of agent models (both SFT and RL), the system generates substantial volumes of trajectory data — complete sequences of (reasoning, tool_call, tool_response) steps. However, quality assessment operates at the trajectory level: a trajectory either reaches the correct final answer (success) or does not (failure). This creates two forms of waste:

1. **Failed trajectories are discarded entirely**, even though they may contain many correct and informative steps. A trajectory with 49 excellent steps and one critical mistake provides no training signal under trajectory-level evaluation.
2. **Successful trajectories are used only once** — as SFT demonstrations to imitate or as positive RL reward signals. The rich structure within the trajectory (the branching decisions at each step, the alternatives that were considered but not chosen) is not extracted.

The paper frames this dramatically: "This coarse-grained evaluation approach leads to significant waste of the learning signals embedded within real trajectories" (Section 2.3). While step-level evaluation could theoretically capture more signal, it is "challenging" because "precisely assessing intermediate steps remains challenging" and "naively incorporating such uncertain reward signals into SFT or RL training risks model collapse."

**Core insight: trajectories are decision processes, not just sequences.** The key conceptual move in HAS is to reframe what a trajectory represents. Rather than viewing it as a sequence of actions to imitate, HAS views each step as a **decision point** — a specific reasoning state defined by the context of the original question and all prior steps, with a broad space of feasible next actions. The paper states:

> "Every step is fundamentally a hidden decision process. However, although agents often generate multiple candidates within a single reasoning-action turn (e.g., alternative queries or exploration directions), these candidates remain internal branches of the same path, and supervision mainly rewards reproducing the full trajectory. Consequently, models learn to imitate a sequence rather than to perform decision-making at critical steps."

This is a profound critique of standard trajectory-based training. Even when agents generate multiple candidates internally (e.g., considering several possible search queries before choosing one), the training signal only reinforces the final chosen path, not the decision-making process that led to that choice. The model learns "go from state A to state B" but not "when in state A, evaluate options X, Y, Z and choose the best one."

HAS addresses this by shifting the training objective "from trajectory imitation to step-wise decision-making, explicitly exploiting the choice space at each step" (Section 2.3).

##### Step-Level Scaling: Expanding the Action Space at Each Decision Point

**Input.** HAS takes as input a trajectory $T = \{(S_1, R_1), (S_2, R_2), \ldots, (S_K, R_K)\}$ where $S_k$ is the "reasoning and tool invocation" at step $k$ (the agent's output), $R_k$ is the corresponding tool/environment response (what the tool returned), and $J \in \{0, 1\}$ is a binary trajectory-level judgment indicating failure (0) or success (1). The trajectory may be from successful or failed post-training runs — HAS is designed to extract value from both.

**Procedure.** For each step $S_k$, define the conditional context $C_k$ as the sequence of everything that occurred before that step:

$$C_k = (Q, S_1, R_1, S_2, R_2, \ldots, S_{k-1}, R_{k-1})$$

where $Q$ is the original question. This context represents the exact reasoning state the agent was in when it decided to produce $S_k$.

Without executing any tools, a strong LLM generates $N$ alternative "thought and invocation" candidates for this context, producing the set:

$$A_k = \{S^{(1)}_k, S^{(2)}_k, \ldots, S^{(N)}_k\}$$

These are alternative reasoning-and-action pairs that the agent *could have* produced at step $k$, given the same context. They might represent different search queries, different web pages to visit, different code to execute, or even a decision to provide a direct answer rather than continue searching.

The original step $S^{(0)}_k \equiv S_k$ is merged with these alternatives to form $N+1$ feasible steps. These are randomly shuffled to form the sequence $\tilde{A}_k$, and the position $n_k$ of the original step in this shuffled sequence is recorded. This step-level expansion transforms the trajectory from a single path with $K$ steps into a decision space with $(N+1) \times K$ potential reasoning-actions.

The critical design choice is that **alternatives are generated without executing tools.** The LLM produces candidate reasoning and tool invocations based on its understanding of the context, but no actual search or browsing occurs. This keeps the generation process offline and cost-free, consistent with the overall FAS/HAS design philosophy.

##### Contrastive Decision-Action Synthesis: Making Decisions Explicit

**Transformation into training text.** After step-level scaling, the trajectory with expanded options is transformed into a **progressive decision-making process.** For each step $k$, the system explicitly simulates a multi-option selection:

1. Enumerate each option in the shuffled sequence $\tilde{A}_k$.
2. Insert a local action decision statement: "I will choose option $n_k$", immediately followed by the corresponding real response $R_k$.
3. Continue to the next step, where the context now includes the chosen option and its real response.

After all $K$ steps are processed, append the judgment text: "My decision is {Correct/Incorrect}" (corresponding to $J$, the trajectory-level success/failure).

The complete synthetic training sample is the concatenation of the problem, the choice-decision process for each step (including the full enumeration of alternatives and the explicit choice), and the final judgment.

The paper's Figure 5 illustrates this transformation with a concrete example. The original trajectory contains a single reasoning-action per step. The HAS version shows, for each step, multiple options (different search queries, different analysis directions), an explicit "I will choose option X" statement, and the real tool response corresponding to the chosen option.

**What the model learns.** This format teaches the model several things simultaneously:

- **Option generation:** Given a context, enumerate plausible next reasoning-actions. This develops the capacity for generating diverse candidates rather than fixating on a single approach.
- **Option evaluation:** The trajectory-level judgment provides implicit feedback on the quality of the chosen options throughout the sequence — successful trajectories validate the decision chain; failed trajectories provide negative signal.
- **Decision-making:** The explicit "I will choose option X" format makes the selection act a learnable behavior, rather than an implicit default of "always take the first thing you think of."
- **Recovery from poor decisions:** Because both successful and failed trajectories are used, the model sees examples where particular decision paths led to failure, developing an implicit understanding of what kinds of choices tend to lead to dead ends.

**Why this avoids the step-level evaluation problem.** The paper explicitly notes that HAS "circumvents the risks associated with directly using uncertain step-level rewards." Step-level rewards would require the system to judge whether each individual step is good or bad — a difficult and error-prone assessment that, if wrong, could actively harm training. Instead, HAS uses the trajectory-level judgment $J$ (which is reliable — the model either got the right answer or didn't) but **spreads it across all steps** by making the decision process explicit. The model sees the full sequence of choices and the final outcome, and can learn the association between decision patterns and success without needing per-step correctness labels.

**Sample efficiency.** The paper claims that through this synthesis strategy, "previously underutilized trajectory data is transformed into rich training signals, significantly improving the sample efficiency of the agentic learning process." A single trajectory that would have been used once (if successful) or discarded (if failed) now generates a training sample with $(N+1) \times K$ reasoning-action options explicitly presented, plus the decision-making structure, plus the final judgment. The number of effective learning signals per original trajectory is multiplied.

---

#### The Two-Stage Training Strategy: Why 32K Then 128K

**Computational motivation.** Training language models with long context windows (128K tokens) is substantially more expensive than training with shorter windows (32K) because the self-attention mechanism's computational complexity scales quadratically with sequence length. For a 200B token corpus, training entirely at 128K would be 16× more expensive in the attention layers than training at 32K (since $(128/32)^2 = 16$), making it economically infeasible.

However, some of the HAS data — specifically, long trajectories with many steps and extensive option enumerations — exceeds 32K tokens in length. Training on this data with a 32K context window would truncate the trajectories, losing the long-range dependencies that make them valuable. The paper acknowledges this tradeoff explicitly in Section 3.4.1:

> "single-stage training on all data where some HAS data may be truncated due to length constraints"

**The two-stage solution.** The paper proposes a progressive two-stage strategy:

**Stage 1 (200B tokens, 32K context):** Train on a broad mixture of FAS data (planning actions + reasoning actions) and shorter HAS data that fits within 32K tokens, along with knowledge reasoning corpora. This stage handles the bulk of the data volume and enables "preliminary acquisition of agentic behaviors including tool invocation patterns and multi-step reasoning chains" (Section 2.1). The diversity of FAS data across many domains ensures broad coverage; the shorter HAS data introduces decision-making patterns; the knowledge corpora maintain general capabilities.

**Stage 2 (100B tokens, 128K context):** Train exclusively on carefully curated, high-quality HAS data with extended context. This stage "allows the LLM to develop a sophisticated understanding of complex action spaces and long-horizon planning strategies" because the model can now see complete trajectories without truncation — all $K$ steps of a long-horizon task, with all $N+1$ options at each step, and the final judgment — in a single contiguous training sequence.

**Why the split rather than just Stage 2?** If Stage 2 alone were sufficient, the paper could have trained only on 100B tokens at 128K. The split is motivated by: (1) FAS data, which constitutes the majority of the training volume, does not benefit from extended context because individual FAS samples (question + planning action or question + reasoning chain) are relatively short; (2) training FAS data at 128K would waste computational resources on context padding; (3) the diversity of FAS data across many domains is essential for developing broad transferable agentic capabilities, but this diversity requires volume (200B tokens), and training at 32K keeps that volume feasible.

The paper validates this choice in Section 3.4.1 (Table 4), showing that the two-stage approach (Stage 1 + Stage 2) yields an average improvement of 3.3% on Pass@1 and 3.7% on Pass@3 across BrowseComp-en, BrowseComp-zh, and GAIA compared to Stage 1 only. The Stage 1 only configuration is described as "single-stage training on all data where some HAS data may be truncated due to length constraints." While the paper acknowledges that resource constraints preclude evaluation of single-stage training with extended context (e.g., 128K), it notes that such an approach "would incur substantially higher computational costs."

**Data composition details.** The paper specifies the Stage 1 corpus as "approximately 200B tokens of agent data and knowledge reasoning corpora" (Section 2.1). The knowledge reasoning corpora are not further detailed but presumably include the logical reasoning chains generated through reasoning action synthesis (Section 2.2.3) and possibly additional curated reasoning data. Stage 2 uses "100B tokens of carefully curated, high-quality agent data with extended 128K context windows," which the paper later clarifies is focused on HAS data specifically.

In the scaling experiments (Section 3.5.2), the authors train models with data volumes ranging from 0B to 315B tokens, with Stage 2 (128K context) being introduced at the 65B and 315B token checkpoints. The paper reports that Stage 2 provides "notable gains at both 65B (+1.8% over 50B) and 315B (+1.0% over 210B)," confirming that long-context training provides consistent improvements even as the scaling curve approaches saturation.

---

#### Post-Training: Three SFT Configurations for Adaptability Validation

**Purpose.** The post-training stage serves two purposes in this paper: (1) it produces the final AgentFounder-30B model for benchmark evaluation, and (2) it validates the central claim that Agentic CPT creates a better starting point for downstream fine-tuning, independent of the specific post-training recipe.

**SFT data composition.** All three SFT configurations use "a strategically proportioned mixture of general instruction data and agent trajectory demonstrations" (Section 2.1). The general instruction data maintains the model's broad conversational and instruction-following capabilities; the agent trajectory demonstrations teach the specific React-style interaction format expected by the evaluation framework.

The challenging information-seeking questions used for constructing agent trajectories are built following methodologies from prior work: WebSailor-V2, WebResearcher, WebWeaver, and AgentScaler (Section 3.1.1). These are separate from the FAS/HAS data used in Agentic CPT — they are task-specific demonstrations formatted for supervised fine-tuning, while the CPT data used next-token prediction to embed general agentic patterns.

**Three configurations.** The paper defines three SFT configurations that differ in data ordering, mixing ratios, and trajectory formatting:

- **SFT-A:** A two-stage training paradigm. First stage: general conversational data. Second stage: specialized React-style agent trajectories with explicit reasoning chains. This is the simplest configuration — learn general instruction following first, then specialize to agent behavior.
- **SFT-B:** An enhanced version of SFT-A that maintains the two-stage paradigm but incorporates a balanced mixture of general conversational data and React-style trajectories in **each stage**. This exposes the model to both types of data throughout training, potentially preventing catastrophic forgetting of general capabilities.
- **SFT-C:** A two-stage paradigm with general conversational SFT data and React trajectories using **summarized reasoning.** Rather than full explicit reasoning chains, the trajectories contain condensed reasoning summaries, reducing the token count per trajectory.

These three configurations are chosen to test different aspects of adaptability: SFT-A represents a standard approach, SFT-B tests whether interleaved training is better than sequential, and SFT-C tests whether the model can work with compressed reasoning formats. The paper does not describe the specific data proportions, training hyperparameters, or optimization details for the SFT stage — the focus is on the CPT innovation, and the SFT stage is primarily a vehicle for demonstrating that the CPT benefits transfer across different fine-tuning recipes.

**Experimental validation of adaptability (Section 3.3, Table 3).** The key experiment compares models fine-tuned from AgentFounder-30B-Base versus models fine-tuned from Qwen3-30B-A3B-Base, each subjected to the same SFT configuration:

| SFT Config | Base Model | BrowseComp-en | BrowseComp-zh | GAIA | HLE |
|---|---|---|---|---|---|
| SFT-A | Qwen3-Base | 26.9 | 29.8 | 67.0 | 23.5 |
| SFT-A | AgentFounder-Base | 31.4 | 35.6 | 72.8 | 30.4 |
| **Δ** | | **+4.5** | **+5.8** | **+5.8** | **+6.9** |
| SFT-B | Qwen3-Base | 28.6 | 35.6 | 71.8 | 27.0 |
| SFT-B | AgentFounder-Base | 39.9 | 43.3 | 72.8 | 31.5 |
| **Δ** | | **+11.3** | **+7.7** | **+1.0** | **+4.5** |
| SFT-C | Qwen3-Base | 24.5 | 36.7 | 68.9 | 27.9 |
| SFT-C | AgentFounder-Base | 38.8 | 44.3 | 71.8 | 28.9 |
| **Δ** | | **+14.3** | **+7.6** | **+2.9** | **+1.0** |

The average gains across all configurations and benchmarks are 5.75% (SFT-A), 6.13% (SFT-B), and 6.45% (SFT-C). These consistent improvements validate the central claim: Agentic CPT provides a better starting point regardless of how post-training is conducted. The gains are particularly pronounced on BrowseComp benchmarks (up to +14.3 on BrowseComp-en with SFT-C) and relatively smaller on HLE (as low as +1.0 with SFT-C), which the authors attribute to HLE being "knowledge-intensive" — requiring not just successful retrieval but also strong comprehension capabilities that Agentic CPT may not sufficiently develop.

---

#### Inference Setup: Tools, Hyperparameters, and Constraints

**Tools.** The final AgentFounder-30B model operates with five core tools, described in detail in Appendix A.1:

1. **Search:** Uses Google Search for large-scale information retrieval. Accepts a list of one or more search queries executed concurrently. Returns top-10 ranked results per query, each with title, snippet, and URL.

2. **Visit:** Designed for targeted information extraction from web pages. Takes a set of URLs, each paired with a dedicated information-seeking goal. Uses Jina (a web content extraction service) to retrieve full page content, then applies a summary model to extract only information pertinent to the specified goal. This goal-directed extraction is important: raw web pages contain extensive navigation, advertising, and boilerplate text; the summary model filters to goal-relevant content.

3. **Python Interpreter:** Executes Python code in a sandboxed environment. Code must be enclosed in `<code>` tags. Captures standard output; results must be explicitly printed via `print()`. Enables dynamic computation, data manipulation, and library usage.

4. **Google Scholar:** Retrieves academic publications. Accepts multiple search queries in a single call. Returns scholarly literature including articles, papers, and citations.

5. **File Parser:** Answers user queries by analyzing documents (PDF, DOCX), web pages, and multimedia files (MP4) from local or URL sources. Two-step process: converts all input to plain text (transcribing audio/video when needed), then a summary model reads the unified text to generate a direct answer.

**Inference hyperparameters.** The paper specifies fixed decoding parameters: temperature = 0.85, repetition penalty = 1.1, and top-p = 0.95. These settings are described as "recommended based on extensive empirical validation to optimize the balance between creativity and consistency in agentic reasoning tasks" (Section 3.1.4). Temperature 0.85 is relatively high, encouraging diverse exploration — important for agent tasks where the model needs to try different search queries and consider alternative hypotheses. Repetition penalty 1.1 discourages token-level repetition (common in tool-use templates). Top-p 0.95 provides a wide sampling distribution while filtering only the very tail of low-probability tokens.

**Operational constraints.** Two hard limits shape agent behavior:
- **Maximum tool calls:** 128 per task. This bounds the agent's exploration depth and prevents infinite loops.
- **Context length:** 128K tokens. This matches the Stage 2 training context window, meaning the model has been trained to handle sequences of this length.

These constraints are enforced at the inference framework level, not learned by the model — the model does not inherently know to stop after 128 calls or at 128K tokens, but the evaluation harness terminates the interaction at these limits.

**Single-agent React paradigm.** The model operates under the React framework, where it alternates between reasoning (analyzing the current state, planning next actions) and acting (issuing tool calls). The paper does not describe the specific formatting of these turns (e.g., how reasoning is separated from tool calls, how tool responses are presented to the model), but the format is consistent across training and inference.

**Benchmark evaluation protocol.** Performance is measured using Pass@1 (a single deterministic answer per question) and Pass@3 (the best of three independent rollouts per question), following standard practice in the agent evaluation literature. The specific answer extraction and grading procedures vary by benchmark and are not detailed in the paper beyond references to the original benchmark publications and the grading function from WebSailor.

## 4. Key Insights and Innovations

### Innovation 1: Reframing the Agent Training Problem as a Pipeline Architecture Defect, Not a Post-Training Optimization Challenge

The paper's most fundamental intellectual move is **diagnostic**: it identifies that the persistent underperformance of open-source agent models relative to proprietary systems is not primarily a failure of post-training methods (SFT, RL, data synthesis), but rather a **structural deficiency in the training pipeline itself.** The dominant assumption in the field — evidenced by the long list of prior works the paper cites (WebThinker, ASearcher, WebSailor, WebShaper, AFM, MiroThinker, DeepDiver, WebExplorer, DeepDive, and others in Section 4.1) — has been that a sufficiently sophisticated post-training recipe applied to a general-purpose foundation model should eventually produce strong agentic behavior. The paper challenges this fundamentally.

The key framing innovation is the concept of **agentic inductive biases** — baked-in patterns and expectations about how agents operate — and the claim that general-purpose foundation models lack them. This is not obvious. One could reasonably argue that a model trained on the entire internet has seen plenty of examples of search queries, API calls, and multi-step procedures. But the paper's implicit argument (supported by the consistent ~5–14 percentage point gaps in Table 3 between Qwen3-Base and AgentFounder-Base across all post-training configurations) is that passive exposure to such patterns in web text is fundamentally different from having them embedded as high-density training objectives during the formative stages of model development.

This reframing has two important consequences for how the field thinks about agent training:

**First, it recasts the optimization tension as a capacity allocation problem.** When a foundation model must simultaneously learn agentic capabilities (tool formats, planning structures, decision-making patterns) and align to specific demonstration trajectories during post-training, these objectives compete for the same model capacity. The paper characterizes this as "inherent optimization conflicts" (Section 1). By moving capability acquisition earlier in the pipeline — to the Agentic CPT stage, where the objective is pure next-token prediction on diverse agentic patterns without the pressure of task-specific alignment — the model can allocate capacity to learning generalizable behaviors before it must specialize. Post-training then only needs to refine and adapt these pre-existing capabilities, rather than building them from scratch.

This is a conceptual contribution that generalizes beyond agents. The paper opens the door to asking: what other capabilities (code generation, mathematical reasoning, structured output formatting) might benefit from dedicated intermediate pre-training stages that front-load behavioral pattern acquisition before task-specific fine-tuning?

**Second, it explains the persistent open-source/proprietary gap without invoking secret sauce.** The paper provides a concrete, reproducible explanation for why even the best open-source agent models (DeepSeek-V3.1 at 30.0% on BrowseComp-en) trail proprietary systems (OpenAI Deep Research at 51.5%): proprietary systems likely incorporate something analogous to Agentic CPT in their training pipelines, but this stage is not publicly disclosed or reproduced. The paper effectively argues that the gap is not primarily about better post-training data or more sophisticated RL — it is about a missing stage in the open-source training recipe. This is a falsifiable claim: if another group replicates Agentic CPT at scale and achieves similar gains, the explanation is validated.

**Evidence anchoring:** Table 3 provides the core evidence. Across three different SFT configurations (representing substantially different post-training recipes), AgentFounder-Base consistently outperforms Qwen3-Base by 5.75–6.45% on average. The gains are not uniform — they range from +1.0 (HLE with SFT-C) to +14.3 (BrowseComp-en with SFT-C) — but they are **universally positive**, meaning there is no post-training configuration where the general-purpose base model catches up. This is exactly what the pipeline defect hypothesis predicts: if the issue were just suboptimal post-training, some SFT configuration should close the gap, but none does.

### Innovation 2: Agentic Data Synthesis at Pre-Training Scale Without Tool Execution

The second key innovation is the demonstration that **agentic behavioral patterns can be taught at pre-training scale (hundreds of billions of tokens) using purely synthetic data generated without executing any real tools.** This is a methodological breakthrough with significant economic and practical implications.

The dominant assumption in agent training has been that high-quality agent data requires actual interaction with tools — you need real search results, real web page content, real code execution outputs to create training trajectories that reflect genuine agent behavior. This assumption is visible in the prior work the paper cites: WebSailor constructs knowledge graphs through actual search and browsing; WebExplorer uses "autonomous model exploration to build information networks" through real web interactions; ASearcher takes an "iterative approach, incrementally adding new information to increase problem complexity" through tool use. These methods produce high-quality data, but they are **fundamentally unscalable** because each training example incurs API costs (Google Search, Jina Reader) and latency.

The paper's FAS and HAS methods break this assumption by showing that agentic patterns can be synthesized offline in two complementary ways:

**FAS generates planning and reasoning actions from structured knowledge without tool execution** (Section 2.2). The critical insight is that the *format* of agentic behavior — how to decompose a problem, how to formulate search queries, how to structure logical deductions from evidence — can be taught using synthetic data where the model's own internal knowledge substitutes for actual tool responses. The knowledge statements from the entity-anchored memory serve as ground truth for reject sampling, ensuring the generated actions are plausible without requiring them to be executed.

**HAS extracts rich training signal from existing trajectories by expanding them into decision-making exercises** (Section 2.3). Rather than treating a trajectory as a single-use demonstration (successful) or waste (failed), HAS reframes each step as a decision point with explicit alternatives, teaching the model the *process* of choosing actions rather than just the *sequence* of actions. This multiplies the learning signal per real trajectory — a single trajectory that would have contributed one training example now contributes a training example with (N+1) × K explicit decision points.

The significance of this goes beyond cost savings (though those are substantial — the paper notes that both approaches "operate without external tool invocations, enabling large-scale data generation in offline environments without API costs"). It means that **agentic capability acquisition is not bottlenecked by tool execution infrastructure.** Any research group with sufficient GPU compute can generate training data at arbitrary scale, without needing API keys, rate limits, or budget for commercial search and browsing services. This is a democratizing insight: it lowers the barrier to entry for agent research from "must have substantial API budget" to "must have substantial GPU budget," which is a more fungible and increasingly accessible resource.

The paper's scaling experiments (Section 3.5.2, Figure 6b) confirm that this synthetic data produces meaningful, predictable improvements: the relationship between Agentic CPT tokens and downstream performance follows a logarithmic scaling law, with gains from 0B to 315B tokens totaling 8.0% in average Pass@3 across benchmarks. This is the kind of scaling behavior one expects from genuine capability acquisition, not from superficial pattern memorization.

**Evidence anchoring:** Table 5 shows that FAS data alone (50B tokens) provides substantial gains over no CPT (+9% Pass@3 on BrowseComp-zh, establishing a "higher performance ceiling for subsequent post-training phases"). Appendix B.1 shows that FAS planning action data, after filtering, achieves 82% accuracy — sufficient for CPT where the training signal comes from next-token prediction on behavioral patterns rather than from trajectory correctness. Figure 6b shows the scaling curve from 0B to 315B tokens with consistent improvements at every checkpoint.

### Innovation 3: Recovering Learning Signal from Failed and Discarded Trajectories

The HAS methodology represents a third conceptual innovation: **recovering training signal from agent trajectories that would otherwise be wasted, by reframing them from action sequences to decision processes.**

This is distinct from Innovation 2. Where FAS is about generating new data from knowledge sources, HAS is about **extracting more value from data that already exists but is currently discarded.** The paper's diagnosis of "trajectory data waste" in Section 2.3 is incisive: trajectory-level feedback for post-training (SFT and RL) creates a binary filter where complete trajectories are either used once (if successful) or discarded entirely (if failed). Intermediate steps — which may be perfectly sensible even in a failed trajectory, or may be mediocre despite a successful outcome — provide no differentiated signal.

The field has been aware of this problem but has struggled with solutions. Step-level reward models or process reward models (PRMs) attempt to provide finer-grained feedback, but the paper correctly notes that "precisely assessing intermediate steps remains challenging" and that "naively incorporating such uncertain reward signals into SFT or RL training risks model collapse." This is a genuine tension: step-level feedback would be valuable but is unreliable; trajectory-level feedback is reliable but wasteful.

HAS resolves this tension through a clever reframing. Rather than trying to assign correctness scores to individual steps (which requires solving the step-level evaluation problem), HAS **transforms the trajectory into a format where the model can learn from step-level patterns without step-level labels.** By presenting each step as an explicit choice among alternatives, with the trajectory-level judgment appended at the end, the model can learn associations between decision patterns and outcomes through next-token prediction — the same dense, token-level objective that drives all pre-training. The trajectory-level judgment provides the supervision, but it is spread across all steps through the decision-making format rather than concentrated at the trajectory endpoint.

This is fundamentally a **data formatting insight** rather than an algorithmic insight. The training objective doesn't change; the model architecture doesn't change; what changes is how the available supervision signal is presented to the model. By making the decision process explicit in the training text, the model learns to evaluate options and make choices as a natural language behavior, rather than as an implicit optimization over a reward function.

The significance extends well beyond this paper. Any domain where trajectory-level feedback is available but step-level feedback is unreliable — robotics, dialogue systems, game-playing agents, code generation — could potentially benefit from similar trajectory-to-decision reformatting. The paper does not explore these extensions, but the concept is clearly generalizable.

**Evidence anchoring:** Table 5 shows that adding HAS data to FAS data provides complementary benefits: FAS+HAS mix shows positive gains across benchmarks compared to FAS alone (e.g., +3.1% Pass@1 on BrowseComp-zh), confirming that the decision-making patterns in HAS contribute capabilities distinct from the planning and reasoning patterns in FAS. Table 4 shows that Stage 2 CPT (which focuses on high-quality HAS data with 128K context) provides consistent improvements over Stage 1 only (+3.3% average Pass@1 gain), validating that learning from complete, un-truncated decision-making sequences matters.

### Innovation 4: Empirical Discovery That Agentic CPT Benefits Are Retrieval-Specific, Not Universal

The paper contains an important **negative result** that functions as a conceptual contribution: Agentic CPT does not improve all agentic capabilities equally. The benefits concentrate on information retrieval tasks and diminish on knowledge-intensive reasoning tasks.

This is visible most clearly in Table 3, where the gains from AgentFounder-Base over Qwen3-Base vary dramatically across benchmarks. On BrowseComp-en (a browsing and retrieval benchmark), gains range from +4.5 to +14.3 percentage points depending on the SFT configuration. On HLE (Humanity's Last Exam, which tests expert-level knowledge across diverse subjects), gains range from +1.0 to +6.9 — consistently smaller and, in the SFT-C case, barely above noise. The authors acknowledge this explicitly in Section 3.3:

> "Compared to HLE, the BrowseComp benchmarks show more pronounced improvements from AgentFounder-30B-Base. We hypothesize that knowledge-intensive tasks like HLE require not only successful information retrieval but also strong reasoning capabilities to correctly utilize the retrieved knowledge."

This is a more nuanced finding than "Agentic CPT helps agent performance." It carves nature at its joints, distinguishing between two capabilities that are often conflated in agent benchmarks: (1) the ability to find relevant information (search formulation, result evaluation, source selection), and (2) the ability to reason correctly with that information once found. Agentic CPT primarily improves the former, with limited impact on the latter.

This distinction has both practical and theoretical implications. Practically, it tells practitioners where to invest: if your agent's bottleneck is retrieval quality, Agentic CPT (or analogous behavioral pre-training) is likely to help; if the bottleneck is reasoning over retrieved information, you need different approaches — perhaps dedicated reasoning CPT, larger models, or specialized reasoning architectures. Theoretically, it suggests that behavioral patterns (how to search, how to plan, how to make decisions) and reasoning capabilities (how to draw valid inferences from evidence) are partially separable in the model's training dynamics, even though they interact closely at inference time.

The finding also refines the paper's central claim. Agentic CPT is not a panacea for agent training; it is specifically a solution to the **behavioral capability acquisition problem** — teaching the model the mechanics of being an agent (tool formats, planning structures, decision-making patterns). The **knowledge comprehension problem** — understanding and reasoning with retrieved information — remains a bottleneck that Agentic CPT does not fully address. This honesty strengthens the paper: it identifies both what the method does and what it does not do, providing clearer guidance for future work than an undifferentiated "Agentic CPT improves everything" narrative would.

**Evidence anchoring:** In Table 3, the HLE column shows gains of +6.9, +4.5, and +1.0 for SFT-A, SFT-B, and SFT-C respectively — consistently the smallest gains among the four benchmarks. The authors' hypothesis about knowledge-intensive tasks is plausible but not experimentally validated (they do not, for example, show that HLE performance fails to improve because the model retrieves correctly but reasons incorrectly). Section 3.3 frames this as suggesting that "enhancing base models' knowledge comprehension abilities represents a promising future research direction," treating the limitation as a roadmap rather than a failure.

The scaling experiments in Section 3.5.1 provide converging evidence: the model scaling curve (Figure 6a) shows that performance continues to improve with model size (20.4% → 32.7% → 48.9% for 1B → 4B → 30B), and the 30B AgentFounder outperforms larger general-purpose models (DeepSeek-V3.1, Kimi-K2), but the paper does not claim that larger AgentFounder models would saturate HLE-type benchmarks — the scaling is about agentic behavioral capabilities, not universal reasoning.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on 10 benchmarks spanning two categories: *general web search benchmarks* (BrowseComp-en, BrowseComp-zh, GAIA, Xbench-DeepSearch, WebWalkerQA) and *scenario-targeted web search benchmarks* (HLE, DeepResearch Bench, Frames, SEAL-0, Academic Browse). GAIA uses the text-only subset of 103 questions. All benchmarks are existing, publicly available evaluation suites sourced from prior work (Wei et al., 2025; Zhou et al., 2025b; Mialon et al., 2023; Xbench-Team, 2025; Wu et al., 2025b; Phan et al., 2025; Du et al., 2025; Pham et al., 2025; Krishna et al., 2024; Zhou et al., 2025a). The paper does not describe creating any new benchmark; all results are on standard test sets.

- **Base model(s).** The primary base model is Qwen3-30B-A3B-Base, a 30B-parameter Mixture-of-Experts architecture with 3B active parameters. The paper also experiments with Qwen3 models at 1B and 4B scales for the model scaling analysis (Section 3.5.1, Figure 6a). Additionally, a Qwen3-235B-A22B model and DeepSeek-R1-0528 are evaluated as "General LLMs with tools" baselines, and a ~14× larger model is used implicitly through comparisons with DeepSeek-V3.1 and Kimi-K2, though the paper does not specify exact parameter counts for commercial or open-source baselines beyond what their original publications report. The choice of Qwen3 is pragmatic — it is "representative of the capabilities of many contemporary LLMs" and available in multiple scales for scaling law experiments.

- **Metrics.** The primary metric is **Pass@1 accuracy** — the fraction of test questions for which the model's first generated answer matches the ground truth. For BrowseComp-en, BrowseComp-zh, and GAIA, Pass@3 is also reported (the best of three independent rollouts per question, scored as correct if any of the three answers matches ground truth). Benchmarks have different grading protocols: GAIA uses exact match against ground-truth answers; BrowseComp uses the evaluation framework from Wei et al. (2025); HLE uses the official grading from Phan et al. (2025); DeepResearch Bench reports "RACE Overall" (a composite metric assessing report quality, Section 3.2). The paper reports these metrics as percentages throughout. For the scaling experiments (Section 3.5), "Average Accuracy" or "Average Pass@3" is computed by averaging Pass@3 across multiple benchmarks (the specific benchmarks included in the average are not explicitly enumerated).

- **Baselines.** The paper evaluates against three categories of strong models (Section 3.1.2):
  - **General LLMs with tools:** Qwen3-30B-A3B-2507, Qwen3-235B-A22B-2507, DeepSeek-R1-0528 (Guo et al., 2025), and Claude-4-Sonnet (Anthropic, 2025).
  - **Commercial deep research agents:** Kimi-Researcher (Team et al., 2025), OpenAI-o3 (OpenAI, 2025a), OpenAI Deep Research (OpenAI, 2025b), Grok Deeper Search (xAI, 2025), Perplexity Deep Research (Perplexity AI, 2025), and Gemini Deep Research (Google, 2025).
  - **Open-source deep research agents:** WebThinker-32B-RL (Li et al., 2025e), ASearcher-Web-QwQ (Gao et al., 2025), WebSailor-72B (Li et al., 2025b), WebShaper-72B (Tao et al., 2025), AFM-32B-RL (Li et al., 2025c), MiroThinker-32B-DPOv0.2 (Team, 2025a), DeepDiver-V2-38B (Team, 2025b), WebExplorer-8B (Liu et al., 2025), DeepDive-32B (Lu et al., 2025), Kimi-K2-Instruct (Team et al., 2025), GLM-4.5 (Zeng et al., 2025), and DeepSeek-V3.1 (DeepSeek-AI, 2025).
  The paper "prioritize[s] official results reported by model providers or benchmark organizers, or scores reported by other published works, and evaluate[s] remaining models under our standardized setup." For works with multiple agent models, only the strongest is reported.

- **Generation budget / compute accounting.** The paper does not normalize across models by FLOPs or inference cost. Comparisons in Tables 1 and 2 are based on final accuracy, not compute-matched evaluation. Each evaluated model may use different inference budgets (number of tool calls, context lengths, sampling strategies). For AgentFounder-30B specifically, the inference budget is fixed: maximum 128 tool calls per task, 128K context length, temperature 0.85, repetition penalty 1.1, top-p 0.95. The paper does not report the average number of tool calls actually used per benchmark (though the tool call distribution analysis in Section 3.6.2, Figure 8, provides histograms for four benchmarks). For the scaling law experiments, training tokens are the unit of compute (0B to 315B tokens of CPT data), and the paper compares models at the same SFT data volume — only the CPT budget varies.

- **Cross-validation / statistical protocol.** The paper does not describe any cross-validation or statistical significance testing for the main benchmark results in Tables 1 and 2. For the adaptability validation (Table 3), each configuration is evaluated once on the test sets of four benchmarks. The scaling law experiments (Section 3.5) train one model per data point. The paper does not report confidence intervals, standard deviations, or statistical tests. Pass@3 (averaging over three rollouts) provides some robustness to sampling variance, but the number of rollouts is fixed at 3 and not validated through bootstrap or other resampling methods. For the training loss analysis (Section 3.6.1), metrics are reported as point values (final loss, minimum achieved loss, average loss over last 100 steps) without error bars. This is a significant methodological limitation — with test sets of varying sizes (GAIA has only 103 questions; other benchmarks may have similarly modest sizes), point estimates can be noisy, and claims of state-of-the-art performance should ideally be accompanied by uncertainty quantification.

### Main Quantitative Results

#### Aggregate Benchmark Performance (RQ1)

The headline results appear in Table 1 (general web search benchmarks) and Table 2 (scenario-targeted benchmarks). AgentFounder-30B establishes new state-of-the-art open-source performance on nearly every benchmark.

**BrowseComp-en: 39.9%.** This surpasses the previous best open-source model, DeepSeek-V3.1 (30.0%), by 9.9 percentage points. It also significantly outperforms strong open-source models including GLM-4.5 (26.4%), WebExplorer-8B (15.7%), and WebSailor-72B (12.0%). Relative to commercial systems, AgentFounder-30B trails OpenAI-o3 (49.7%) and OpenAI Deep Research (51.5%) but narrows the gap substantially compared to prior open-source models — the gap shrinks from 21.5 points (DeepSeek-V3.1 vs. OpenAI Deep Research) to 11.6 points (AgentFounder-30B vs. OpenAI Deep Research). Among general LLMs with tools, the best is Claude-4-Sonnet at 12.2%, highlighting that tool-equipped general models without deep research training are not competitive on this benchmark.

**BrowseComp-zh: 43.3%.** This is the one benchmark where AgentFounder-30B does not clearly lead open-source models — it is comparable to DeepSeek-V3.1 (49.2%) and trails OpenAI-o3 (58.1%). The paper attributes this to two factors: "the relatively limited proportion of Chinese data in our training corpus, and the possibility that the underlying search tool (Google Search) may exhibit suboptimal performance or bias in Chinese contexts" (Section 3.2). Notably, AgentFounder-30B still outperforms GLM-4.5 (37.5%), a Chinese-developed model, and substantially exceeds WebSailor-72B (30.1%) and Kimi-K2 (28.8%).

**GAIA (text-only, 103 questions): 72.8% Pass@1, 82.5% Pass@3.** This is the highest reported single-agent accuracy on GAIA's text subset. It exceeds all open-source models (DeepSeek-V3.1: 63.1%, GLM-4.5: 66.0%, MiroThinker: 64.1%) and commercial systems including OpenAI-o3 (70.5%). The paper does not report GAIA performance on the validation set (which has a different number of questions), making direct comparison with older works that used different GAIA splits potentially imperfect. The per-level breakdown in Appendix B.3 (Figure 11) shows that performance degrades substantially with task complexity: 79.5% Pass@1 on Level 1, 68.2% on Level 2, and 50.0% on Level 3.

**Xbench-DeepSearch: 73.0%.** This exceeds DeepSeek-V3.1 (71.0%), GLM-4.5 (70.0%), and all other open-source models. It surpasses commercial systems OpenAI-o3 (66.0%) and Kimi-Researcher (69.0%). The paper does not specify the number of test questions or the scoring protocol for this benchmark.

**WebWalkerQA: 71.9%.** AgentFounder-30B outperforms all open-source models and matches commercial systems (OpenAI-o3: 71.7%). The next-best open-source model is GLM-4.5 at 65.6%.

**HLE (Humanity's Last Exam): 31.5% Pass@1.** This is the first open-source model to exceed 30% on HLE, surpassing DeepSeek-V3.1 (29.8%), Gemini Deep Research (26.9%), Kimi-Researcher (26.9%), and OpenAI Deep Research (26.6%). The paper notes that this "significantly exceeds all reported closed-source deep research products." However, the absolute performance remains low (68.5% of questions are answered incorrectly in a single attempt), consistent with HLE's design as an extremely challenging benchmark.

**DeepResearch Bench (RACE Overall): 47.9%.** This surpasses both OpenAI Deep Research (46.5%) and all open-source agents. The next-best open-source model is GLM-4.5 at 39.2%. The paper notes this result "confirm[s] the comprehensiveness, readability, and depth of AgentFounder-30B's generated reports."

**Frames: 89.6%.** This substantially exceeds all reported results, including commercial systems (OpenAI-o3: 84.0%, Claude-4-Sonnet: 80.7%) and open-source models (DeepSeek-V3.1: 83.7%, GLM-4.5: 78.9%). The paper attributes this to "superior capacity for multi-perspective reasoning and consistent information synthesis."

**SEAL-0: 43.9%.** AgentFounder-30B outperforms DeepSeek-V3.1 (42.6%) and GLM-4.5 (34.2%). The paper interprets this as "strong resistance to information interference."

**Academic Browse: 75.3%.** This substantially exceeds DeepSeek-V3.1 (65.0%) and GLM-4.5 (55.6%), demonstrating strong scholarly research capabilities.

**Key pattern across all benchmarks:** AgentFounder-30B achieves state-of-the-art open-source results on 9 of 10 benchmarks (the exception being BrowseComp-zh where DeepSeek-V3.1 leads by 5.9 points). It also surpasses at least one commercial system on 7 of 10 benchmarks. The gains over the next-best open-source model are substantial on the hardest benchmarks — +9.9 points on BrowseComp-en, +8.3 points on GAIA — but more modest on already-saturated benchmarks.

#### Adaptability to Post-Training Configurations (RQ2)

Table 3 reports the core experiment validating that Agentic CPT provides a better starting point for diverse post-training approaches. The experiment compares AgentFounder-30B-Base vs. Qwen3-30B-A3B-Base, each fine-tuned with three different SFT configurations, evaluated on BrowseComp-en, BrowseComp-zh, GAIA, and HLE.

**Consistent and substantial gains across configurations.** AgentFounder-Base outperforms Qwen3-Base under every SFT configuration and benchmark combination — 12 comparisons, 12 wins, no exceptions. The average gains are: SFT-A (+5.75%), SFT-B (+6.13%), SFT-C (+6.45%). The wins range from +1.0 (HLE with SFT-C) to +14.3 (BrowseComp-en with SFT-C).

**Largest gains on BrowseComp, smallest on HLE.** On BrowseComp-en, gains are +4.5 (SFT-A), +11.3 (SFT-B), +14.3 (SFT-C) — substantial and varying with SFT configuration. On BrowseComp-zh, gains are +5.8, +7.7, +7.6 — consistent and large. On GAIA, gains are +5.8, +1.0, +2.9 — positive but varying. On HLE, gains are +6.9, +4.5, +1.0 — positive but smaller, and near-zero for SFT-C.

**Post-training data matters independently.** Despite sharing the same AgentFounder-30B-Base, the SFT configuration significantly affects final performance. For BrowseComp-zh: SFT-B (43.3%) substantially outperforms SFT-A (35.6%) and SFT-C (44.3%) — a range of 8.7 points. This confirms that while Agentic CPT provides a better foundation, post-training remains crucial for unlocking full capabilities, and the specific post-training recipe matters.

#### Scaling Laws (RQ5)

**Model size scaling (Figure 6a).** Average accuracy (across benchmarks, specific set not enumerated) increases from 20.4% (1B) to 32.7% (4B) to 48.9% (30B-A3B). This is a positive, roughly linear trend in the log-linear plot, though with only three data points, the functional form cannot be reliably determined. The paper notes that AgentFounder-30B (48.9%) exceeds DeepSeek-V3.1 (43.0%) and Kimi-K2 (29.6%) on this average metric, despite the latter two being larger models — demonstrating "superior scaling efficiency."

**Data volume scaling (Figure 6b).** Average Pass@3 (across benchmarks) improves from 54.2% (0B CPT data) to 62.2% (315B CPT data), a total gain of 8.0 percentage points. The curve is approximately logarithmic: the largest gain occurs in the first 15B tokens (54.2% → 58.0%, +3.8), with diminishing returns thereafter (50B → 65B: +1.3; 210B → 315B: +1.0). Stage 2 CPT (128K context) provides notable boosts at both introduction points: 65B (+1.8 over 50B) and 315B (+1.0 over 210B). The paper claims this "logarithmic scaling law holds for agentic capabilities" and that the training methodology prevents premature convergence.

**Pass@N scaling on BrowseComp-en (Figure 10, Appendix B.2).** Performance scales from 31.5% Pass@1 to 75.8% Pass@16, a gain of +44.3 percentage points. Pass@18 reaches 77.0%, suggesting saturation around 16–18 samples. This demonstrates that AgentFounder produces diverse solutions — the model is not just memorizing a single trajectory pattern for each question but can generate multiple viable approaches.

#### Tool Call Behavior Analysis (Section 3.6.2, Figure 8)

The paper analyzes tool invocation distributions to understand how AgentFounder adapts to task characteristics. The histograms in Figure 8 reveal two distinct patterns:

**Complex research tasks (BrowseComp-en, HLE) show heavy-tailed distributions.** These tasks require extensive tool usage, with non-trivial probability mass extending past 40 tool calls. BrowseComp-en shows the highest tool density at higher ranges, reflecting persistent web browsing behavior. HLE shows extended patterns reflecting combined complex reasoning with search-augmented inference.

**Structured tasks (WebWalkerQA, GAIA) show concentrated distributions.** WebWalkerQA's distribution peaks sharply at low invocation counts (0–5 calls), consistent with efficient text navigation. GAIA-text exhibits a compact distribution suited for well-defined problems with clear solution paths.

The paper interprets this as evidence that AgentFounder "calibrate[s] tool usage based on task complexity" — it uses more tools when the task demands extensive exploration, and fewer tools when the problem structure allows efficient resolution.

#### Accuracy vs. Tool Calls (Appendix B.5, Figure 13)

Across BrowseComp-en, BrowseComp-zh, GAIA, and Xbench-DeepResearch combined, the paper bins trajectories by total tool call count and reports accuracy per bin. The key findings:

- Tasks with fewer tool calls achieve higher accuracy (the leftmost bins show the highest success rates).
- When no tool calls are made, accuracy drops noticeably and sample count is very small, suggesting the model benefits from tool usage in most cases.
- Even for challenging cases with 40+ tool calls, the model maintains non-trivial success rates (average 17.5%), demonstrating persistence on difficult problems.

This analysis is aggregate, not per-benchmark, and the paper does not control for question difficulty — easier questions might require fewer tool calls and have higher accuracy for reasons unrelated to tool-use efficiency.

#### Training Efficiency (Section 3.6.1, Figure 7)

The paper analyzes SFT training loss curves to validate that Agentic CPT reduces the optimization burden during post-training. All models are trained for 1,340 steps on the same SFT-A data.

**AgentFounder variants achieve substantially lower loss.** The baseline (Qwen3-30B-A3B-Base) reaches a final loss of 0.8656. The best AgentFounder variant (315B tokens CPT) achieves 0.7953 — a reduction of 0.0703. The paper reports that "all AgentFounder variants achieve markedly lower loss values compared to the baseline across all metrics" (final loss, minimum achieved loss, average loss over last 100 steps).

**Loss decreases monotonically with CPT data volume.** The FAS-only model has higher loss than FAS+HAS mixed models, which have higher loss than the 210B and 315B models. This monotonic improvement is consistent with the scaling law findings, suggesting that more CPT data consistently makes post-training easier.

**Interpretation caveat.** Lower SFT loss is not necessarily equivalent to better downstream task performance — it could reflect overfitting to the SFT data distribution. The paper does not correlate training loss with benchmark accuracy, making the loss curves suggestive but not definitive evidence that CPT "alleviate[s] the dual-burden problem."

### Ablation Studies and Robustness Checks

**Training strategy: single-stage vs. two-stage (Table 4, RQ3).** Comparing AgentFounder Stage 1 Only (single-stage training at 32K context, 50B tokens) vs. AgentFounder Stage 1 & 2 (two-stage training with 128K Stage 2, 50B tokens total), both followed by SFT-A. The two-stage approach yields average improvements of +3.3% Pass@1 and +3.7% Pass@3 across BrowseComp-en, BrowseComp-zh, and GAIA. The largest gain is +8.0% Pass@3 on BrowseComp-zh. The paper notes that the Stage 1 Only configuration may truncate long HAS data due to 32K context constraints, so the comparison is between (Stage 1 with truncation) and (Stage 1 + Stage 2 without truncation). A fairer ablation — single-stage training at 128K with all data — is not run due to computational constraints.

**Data type: FAS vs. FAS+HAS (Table 5, RQ4).** Both configurations use 50B tokens of CPT data, followed by SFT-A. FAS alone achieves 31.4% BrowseComp-en Pass@1; adding HAS yields no improvement (31.4%). On BrowseComp-zh Pass@1, FAS+HAS (40.1%) outperforms FAS alone (37.0%) by +3.1%. On GAIA Pass@1, FAS alone (72.8%) outperforms FAS+HAS (69.9%) by -2.9%, but Pass@3 shows the reverse: FAS+HAS (82.5%) vs. FAS alone (80.6%), a gain of +1.9. The paper interprets the GAIA Pass@1 dip as "normal evaluation fluctuations rather than indicating systematic degradation." Overall, both data types contribute meaningful improvements, with FAS providing broad base capability and HAS adding complementary decision-making patterns.

**CPT data quality filtering (Appendix B.1, Figure 9).** The paper applies weak-supervision filtering to FAS planning action data. Initial generation is 50% correct/50% incorrect. The filter removes 43.5% of samples, yielding retained data that is 82% accurate. Error types are dominated by semantic issues (Content Inconsistency: 26.2%, Search Necessity: 6.9%, Logic Discontinuity: 5.7%) rather than syntactic errors (Invalid Tool: 1.2%). The absolute amount of correct data decreases slightly (50% → 46.3% of original volume) but the precision improvement (50% → 82% among retained) justifies the volume reduction. This is a critical robustness check: it shows that without filtering, half of FAS data would contain errors, and the filtering is effective at removing the worst samples.

**General tool-use transfer (Table 6, Section 3.6.3).** On ACEBench, a benchmark for general tool-use capabilities (not specific to deep research), AgentFounder-30B achieves 70.0% vs. Qwen3-30B-A3B at 67.2% — a +2.8 point gain. This suggests that Agentic CPT's benefits partially transfer beyond the specific deep research tasks it was trained for, supporting the claim that agentic capabilities are "domain-agnostic."

**MoE expert activation patterns (Appendix B.4, Figure 12).** The paper analyzes router logits on BrowseComp-zh, comparing Qwen3-30B-A3B-Base (no CPT) with AgentFounder-30B-A3B. After CPT, the expert distribution in the final layers becomes "more balanced ... rather than being concentrated." The paper interprets this as potentially reducing the risk of "dead experts" and enabling "more diversified utilization of multiple experts, which empirically leads to greater training stability during the post-training phase." This is a qualitative observation without quantitative metrics of expert utilization balance or direct causal evidence linking expert distribution to training stability.

**GAIA performance by difficulty level (Appendix B.3, Figure 11).** Pass@1 degrades from 79.5% (Level 1) to 68.2% (Level 2) to 50.0% (Level 3). Pass@3 follows the same trend: 87.2% → 75.0% → 58.3%. This validates that GAIA's level assignments correlate with model difficulty and that AgentFounder-30B has substantial room for improvement on the hardest GAIA questions.

### Critical Assessment

**Claim: "Agentic CPT provides consistent and substantial improvements across all post-training configurations."** The evidence in Table 3 supports this claim but with important nuance. The improvements are indeed universally positive (12 of 12 comparisons favor AgentFounder-Base), and the average gains (5.75–6.45%) are substantial. However, the range is wide: +1.0 (HLE, SFT-C) to +14.3 (BrowseComp-en, SFT-C). On HLE, the gains are small enough that with a different random seed or slightly different SFT data, they might vanish — the paper provides no error bars or significance tests. The claim would be stronger if accompanied by multiple training runs per configuration to assess variance. Additionally, only four benchmarks are tested in this experiment, and the SFT configurations differ in ways that are described but not precisely quantified (e.g., the mixing ratio in SFT-B, the summarization strategy in SFT-C). The claim is **supported** but the strength of support varies by benchmark, with BrowseComp showing robust gains and HLE showing marginal ones.

**Claim: "AgentFounder-30B achieves state-of-the-art performance across 10 benchmarks."** The evidence in Tables 1 and 2 supports this for 9 of 10 benchmarks (the exception being BrowseComp-zh where DeepSeek-V3.1 leads). However, several caveats apply:

1. **Incomplete baseline coverage.** The tables contain many "—" entries where baseline results are unavailable. For example, OpenAI Deep Research has no reported GAIA, Xbench-DeepSearch, or WebWalkerQA scores. This makes the "state-of-the-art" claim partially unverifiable — AgentFounder-30B might be outperforming reported numbers while still trailing unreported ones.

2. **Evaluation protocol differences.** The paper uses both officially reported scores and its own standardized setup. If AgentFounder-30B is evaluated under conditions (tool set, hyperparameters, answer extraction) that differ from those used for baseline models, the comparisons may not be fair. The paper does not detail how it ensures protocol consistency across all models.

3. **Single evaluation per benchmark.** With no confidence intervals and test sets of varying sizes (GAIA: 103 questions, BrowseComp-en: unknown but likely modest), a difference of a few percentage points may not be statistically significant. The claim that AgentFounder-30B "surpasses" commercial systems on certain benchmarks (e.g., GAIA at 72.8% vs. OpenAI-o3 at 70.5%) is based on point estimates that could overlap under reasonable uncertainty bounds.

4. **The BrowseComp-zh exception is under-explained.** AgentFounder-30B trails DeepSeek-V3.1 by 5.9 points on BrowseComp-zh. The paper's hypotheses (limited Chinese data, Google Search bias) are plausible but untested. A simple experiment — running AgentFounder-30B with a Chinese search engine on BrowseComp-zh — could potentially identify the bottleneck, but it is not performed.

**Claim: "Scaling laws apply to both data volume and model size in agentic CPT."** Figure 6 provides evidence consistent with scaling behavior, but the evidentiary bar for "scaling law" claims is high:

1. **Three data points for model scaling (Figure 6a).** With models at 1B, 4B, and 30B parameters, the paper fits no functional form and reports no scaling law parameters. The data is consistent with a power law, a logarithmic relationship, or a linear trend. Three points are insufficient to distinguish these.

2. **Six data points for data scaling (Figure 6b).** The points from 0B to 315B tokens show a roughly logarithmic pattern, but again no functional form is fit and no scaling coefficient is reported. The claim of "logarithmic scaling law holds for agentic capabilities" overstates what six irregularly-spaced points can establish.

3. **Averaging across benchmarks obscures per-task behavior.** The y-axis is "Average Pass@3" across an unspecified set of benchmarks. If different benchmarks scale differently (as the adaptability experiment suggests — HLE scales poorly, BrowseComp scales well), the average may not represent any individual benchmark's behavior.

4. **No held-out validation of the scaling trend.** The paper reports a single training run per data point. There is no assessment of whether the improvements are robust to retraining with different random seeds or data shuffles. At a minimum, the scaling curves should be interpreted as illustrative rather than predictive.

**Claim: "Agentic CPT alleviates the dual-burden problem by endowing models with foundational agentic capabilities before post-training."** The training loss curves in Figure 7 show that AgentFounder models achieve lower SFT loss than the baseline. This is consistent with the "dual-burden" hypothesis but does not distinguish it from alternative explanations: perhaps Agentic CPT simply provides more total training, and any additional pre-training (even non-agentic) would lower SFT loss; perhaps the lower loss reflects overfitting to patterns in the CPT data that happen to match SFT data formats; perhaps the loss reduction does not translate to proportionally better task performance. The paper does not include the most informative ablation: a non-agentic CPT control group (same data volume, same training procedure, but with non-agentic text) to isolate whether the *agentic* nature of the CPT data specifically, rather than just *more training*, drives the improvement. Without this control, the "dual-burden" narrative is plausible but unproven.

**Missing experiments that would strengthen the paper:**

1. **Non-agentic CPT control.** Train on an equivalent volume of general-domain text (not agentic data) and compare downstream agent performance. This would isolate the specific contribution of agentic behavioral patterns from the general benefits of additional training.

2. **Step-level evaluation of HAS effectiveness.** The paper claims HAS teaches decision-making rather than sequence imitation. A direct test would be to compare HAS-trained models vs. FAS-only models on tasks requiring novel decision sequences (not seen in training), measuring whether HAS-trained models show greater behavioral diversity or better recovery from initial poor decisions.

3. **Cross-model-family validation.** All experiments use Qwen3 base models. Demonstrating Agentic CPT benefits starting from a different model family (e.g., LLaMA, DeepSeek) would establish generalizability.

4. **Difficulty-controlled scaling analysis.** The paper finds that CPT benefits differ by benchmark (large on BrowseComp, small on HLE). Analyzing scaling behavior separately for retrieval-heavy vs. knowledge-intensive benchmarks would clarify whether the scaling laws apply uniformly or are benchmark-dependent.

5. **Inference budget normalization.** Tables 1 and 2 compare models that may use different numbers of tool calls, different context lengths, and different sampling strategies. A fairer comparison would control for inference compute (e.g., all models limited to 128 tool calls, or all models given equal wall-clock time).

6. **Statistical significance.** For a paper claiming state-of-the-art on 10 benchmarks, reporting confidence intervals or conducting significance tests (e.g., bootstrap over test questions) would substantially strengthen the claims, particularly where margins are small (e.g., GAIA 72.8% vs. 70.5%).

**Negative results and their interpretation:**

The paper includes several findings that function as implicit negative results:

- **HLE gains are small.** Table 3 shows Agentic CPT provides minimal benefit on HLE (as low as +1.0 with SFT-C). The paper interprets this as reflecting HLE's knowledge-intensive nature, but this is a post-hoc hypothesis. It could equally well reflect that HLE questions are out-of-distribution relative to the CPT data, that HLE requires capabilities Agentic CPT does not teach, or that the base model's knowledge is the bottleneck and no amount of behavioral training helps. Without experiments varying the knowledge content of CPT data or measuring knowledge acquisition separately, the explanation remains speculative.

- **Stage 2 gains diminish at scale.** Figure 6b shows Stage 2 CPT provides +1.8% at 65B tokens but only +1.0% at 315B tokens. This suggests long-context training has diminishing returns, consistent with the possibility that the benefits come from seeing complete trajectories (which saturates once all trajectories fit in context) rather than from learning fundamentally new long-range reasoning patterns.

- **Pass@n scaling saturates.** Figure 10 shows gains from additional sampling diminish after n=16. This bounds the practical benefit of increased inference compute for AgentFounder and suggests solution diversity (while present) has limits.

**Overall assessment of experimental rigor:**

The paper's experimental program is ambitious in scope (10 benchmarks, multiple baselines, scaling experiments, ablations) but has notable methodological limitations. The strongest evidence is for the central claim that Agentic CPT improves downstream agent performance (Tables 1-3 consistently show AgentFounder outperforming baselines). The weakest evidence is for the mechanistic claims about *why* it works — the "dual-burden" hypothesis, the "decision-making vs. sequence imitation" framing, the scaling law characterization — all of which are consistent with the data but not rigorously tested against alternative explanations. The paper would benefit from more controlled ablations (especially non-agentic CPT baselines), statistical uncertainty quantification, and more systematic investigation of the benchmark-dependence of CPT benefits. The results are impressive enough to motivate further research, but the paper's explanatory framework should be treated as a productive hypothesis rather than an established mechanism.

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Cost Is Not Accounted for in the Agentic CPT Budget

**The assumption or constraint.** The paper's entire pipeline rests on the assumption that agentic behavioral patterns can be embedded into the foundation model through next-token prediction on synthetically generated data. The FAS and HAS methods produce this data without executing real tools, which the authors frame as a key enabling innovation: "Both synthesis approaches operate without external tool invocations, enabling large-scale data generation in offline environments without API costs" (Section 1). However, this framing hides a substantial computational cost: generating 315B tokens of synthetic agent data using LLMs—the planning actions, reasoning chains, option expansions, and decision sequences—requires running inference on a capable LLM at massive scale. The paper never quantifies this generation cost in FLOPs, GPU-hours, or dollars.

**The consequence.** The headline results—AgentFounder-30B outperforming general-purpose baselines by 5–14 percentage points depending on benchmark (Table 3)—are achieved by consuming 315B tokens of agentic CPT data on top of whatever data the base Qwen3 model already consumed during pre-training. A practitioner evaluating whether to adopt Agentic CPT needs to know: is the improvement from Agentic CPT larger than what they would get by simply spending the same compute budget on more general pre-training, or on more post-training data, or on a larger base model? Without quantifying the CPT generation cost and including it in a matched-budget comparison against alternative uses of the same compute, the paper cannot rule out the possibility that the gains come primarily from **more total training** rather than from the **agentic nature** of the training data specifically. The absence of a non-agentic CPT control group—training on an equivalent volume of general-domain text using the same compute budget—makes this ambiguity impossible to resolve from the reported experiments alone.

**What evidence exists in the paper.** The paper provides no FLOPs accounting, no GPU-hour estimates, and no cost analysis for the data synthesis pipeline. Section 3.5.2 shows that performance scales with CPT data volume (Figure 6b, from 54.2% at 0B to 62.2% at 315B tokens), but this scaling curve is unaccompanied by any control condition showing what an equivalent volume of non-agentic training would yield. The adaptability experiment (Table 3) compares AgentFounder-Base vs. Qwen3-Base, which differ in both total training volume and training data composition—the comparison is between (Qwen3 pre-training + SFT) and (Qwen3 pre-training + 315B agentic tokens + SFT), a confounded contrast where the control group receives substantially less total training. The training efficiency analysis (Section 3.6.1, Figure 7) shows that AgentFounder models achieve lower SFT loss than the baseline, but this too confounds additional training volume with training data type.

**Mitigation status.** The paper does not acknowledge this as a limitation. The authors frame the offline, tool-free nature of FAS/HAS generation as a cost advantage relative to methods that require real API calls during data creation, but they do not account for the LLM inference cost of generating 315B tokens of synthetic data. A matched-budget comparison—Agentic CPT vs. general CPT at equal total compute—is mentioned nowhere in the paper and represents the most important missing experiment for establishing that *agentic* training data specifically, rather than *more* training data, drives the improvements. Until such a comparison is performed, the paper's central claim that "embedding agentic behavioral patterns into foundation model weights ... creates a pre-aligned starting point that substantially improves downstream agent performance" remains confounded with "training the model for longer improves performance."

---

### 6.2 Knowledge-Intensive Reasoning Shows Minimal Benefit from Agentic CPT—and the Paper Cannot Explain Why

**The assumption or constraint.** The paper claims that Agentic CPT creates "pre-aligned agentic foundation models that naturally support agentic behaviors for effective downstream fine-tuning" (Section 1). The implicit assumption is that the behavioral patterns taught during Agentic CPT—tool invocation formats, planning structures, multi-step reasoning chains, decision-making sequences—are the primary bottleneck for agentic performance, and that embedding them in the foundation model will broadly improve agent capabilities across task types.

**The consequence.** The experimental results reveal a sharp boundary on this claim: Agentic CPT provides large gains on information retrieval benchmarks (BrowseComp-en: +4.5 to +14.3 points depending on SFT configuration, Table 3) but near-zero gains on the knowledge-intensive HLE benchmark (+1.0 point with SFT-C, Table 3). The authors acknowledge this explicitly: "Compared to HLE, the BrowseComp benchmarks show more pronounced improvements from AgentFounder-30B-Base. We hypothesize that knowledge-intensive tasks like HLE require not only successful information retrieval but also strong reasoning capabilities to correctly utilize the retrieved knowledge" (Section 3.3). But this hypothesis admits a fundamental gap: Agentic CPT teaches the model **how to find information** but does not teach it **how to reason correctly with that information once found.** For a deep research agent, both capabilities are essential—an agent that retrieves perfectly but reasons incorrectly is no more useful than one that reasons well but cannot find relevant sources. The paper demonstrates that Agentic CPT primarily addresses the retrieval half of the problem while leaving the reasoning half largely untouched.

This has direct practical consequences. On HLE (31.5% Pass@1), AgentFounder-30B outperforms existing models but still fails on 68.5% of questions in a single attempt. The per-level GAIA breakdown (Appendix B.3, Figure 11) shows Pass@1 degrading from 79.5% (Level 1) to 50.0% (Level 3), with even Pass@3 falling to 58.3% on the hardest tier—meaning that even with three attempts, the agent fails on over 40% of the hardest GAIA questions. These are not marginal failures that slightly more CPT data would fix; they represent a qualitative capability gap that Agentic CPT, by its design, does not address.

**What evidence exists in the paper.** Table 3 provides the clearest evidence. On HLE, the gain from AgentFounder-Base over Qwen3-Base ranges from +1.0 (SFT-C) to +6.9 (SFT-A). With SFT-C—the configuration that uses summarized reasoning trajectories—the gain is barely distinguishable from noise given the lack of confidence intervals. The scaling experiments (Section 3.5.2) report only "Average Pass@3 across multiple agentic benchmarks" without breaking out HLE separately, making it impossible to determine whether additional CPT data improves HLE at all or whether the overall scaling curve is driven entirely by retrieval-heavy benchmarks. The model scaling experiment (Figure 6a) shows that larger AgentFounder models perform better on average, but again without per-benchmark breakdown, so a practitioner cannot assess whether scaling model size closes the HLE gap or merely improves retrieval further.

**Mitigation status.** The paper treats this limitation as a research direction rather than a flaw: "enhancing base models' knowledge comprehension abilities represents a promising future research direction" (Section 3.3). This is honest framing but does not help a practitioner who needs a deep research agent that both retrieves and reasons. The paper provides no diagnostic experiment to confirm the hypothesis—for instance, measuring whether HLE failures occur because the agent retrieves irrelevant information (a retrieval failure that Agentic CPT might eventually fix with more data) or because the agent retrieves relevant information but draws incorrect conclusions (a reasoning failure that Agentic CPT does not address). Without such diagnostics, the "knowledge-intensive" explanation remains a plausible but unvalidated story.

---

### 6.3 Single Model Family and Single Language—No Evidence of Cross-Architecture or Cross-Language Generalization

**The assumption or constraint.** All experiments in the paper use Qwen3 series models as the starting point for Agentic CPT: Qwen3-30B-A3B-Base for the main experiments, Qwen3-1B and Qwen3-4B for model scaling. The paper claims that Agentic CPT addresses a fundamental pipeline deficiency applicable to "general-purpose foundation models" broadly (Section 1), and that the agentic capabilities acquired are "domain-agnostic abilities that transcend specific domains and enable universal tool utilization and multi-step reasoning" (Section 2.2.1). The implicit assumption is that the benefits of Agentic CPT would transfer to other model families (LLaMA, DeepSeek, Mistral) and that the approach does not depend on architecture-specific properties of the Qwen3 family or the MoE design of the 30B-A3B variant.

**The consequence.** A practitioner starting from a non-Qwen base model—for instance, LLaMA-3 for a research project, or a proprietary internal model at a company—has no evidence that Agentic CPT would provide comparable gains. The MoE architecture of Qwen3-30B-A3B is particularly relevant: the paper's analysis in Appendix B.4 (Figure 12) shows that Agentic CPT changes the expert activation distribution in the model's final layers, making it "more balanced ... rather than being concentrated," and the authors suggest this "potentially offers the benefit of enabling more diversified utilization of multiple experts, which empirically leads to greater training stability during the post-training phase." If part of Agentic CPT's benefit comes from improving expert utilization in MoE architectures specifically, then dense models or differently-designed MoE models might not benefit to the same degree. The paper provides no way to assess this.

Similarly, the paper's training data and evaluation are predominantly English-focused. The BrowseComp-zh results (AgentFounder-30B at 43.3% vs. DeepSeek-V3.1 at 49.2%, Table 1) represent the only non-English evaluation, and they are the one benchmark where AgentFounder does not achieve state-of-the-art open-source performance. The authors attribute this to "the relatively limited proportion of Chinese data in our training corpus" (Section 3.2), but this explanation itself reveals the limitation: the CPT data synthesis pipeline (FAS and HAS) is not language-agnostic by default; it requires explicit investment in language-specific data generation. A practitioner building an agent for Japanese, Arabic, or Hindi has no evidence that the approach would work and some evidence (from BrowseComp-zh) that it may underperform without substantial language-specific investment.

**What evidence exists in the paper.** The entirety of the experimental evidence comes from Qwen3 models and from benchmarks that are predominantly in English with one Chinese exception. The model scaling experiment (Figure 6a) uses Qwen3-1B, Qwen3-4B, and Qwen3-30B-A3B—all from the same family, same training recipe, same tokenizer, same architecture. The general tool-use transfer experiment (Section 3.6.3, Table 6) compares AgentFounder-30B against Qwen3-30B-A3B on ACEBench—again within the same model family. There are no experiments with dense architectures, no experiments starting from LLaMA or DeepSeek base models, no multilingual evaluation beyond English and Chinese.

**Mitigation status.** The paper does not acknowledge this as a limitation, nor does it claim that Agentic CPT is model-family-agnostic. The abstract states that "post-training approaches building upon general-purpose foundation models consistently underperform in agentic tasks" and proposes Agentic CPT as the solution, which carries the implicit claim of broad applicability. But the experimental scope does not match this claim. The paper does not suggest future work on cross-family validation. For a practitioner, the safest interpretation of the results is: "Agentic CPT works for Qwen3 MoE models on English deep research tasks." Generalization beyond this scope is unproven.

---

### 6.4 The Single-Agent React Paradigm Bounds Performance and Does Not Explore Multi-Agent or Alternative Inference Architectures

**The assumption or constraint.** All evaluation of AgentFounder-30B is conducted under a "single-agent React paradigm" (Section 3.2), where one model instance alternates between reasoning and tool invocation within a single sequential interaction loop. The model has access to five tools (Search, Visit, Python Interpreter, Google Scholar, File Parser), operates with a maximum of 128 tool calls per task and 128K context length, and decodes with fixed hyperparameters (temperature 0.85, repetition penalty 1.1, top-p 0.95). The paper does not explore whether the AgentFounder-Base model could serve as a component in more sophisticated inference architectures—multi-agent ensembles, hierarchical planning-execution frameworks, or inference-time search over action sequences.

**The consequence.** Several baselines that AgentFounder compares against use multi-agent or multi-step inference frameworks. The Cognitive Kernel-Pro framework (Fang et al., 2025b) employs a multi-agent architecture. Tencent Youtu's system uses DeepSeek-V3.1 within a multi-agent framework to achieve 71.47% on WebWalkerQA (Wu et al., 2025b). The paper's comparisons do not control for inference architecture—AgentFounder-30B's single-agent performance is compared against other models' single-agent scores where available, and against multi-agent scores where those are the only reported numbers. This makes it difficult to determine how much of AgentFounder's advantage comes from the CPT innovation versus from differences in inference strategy.

More importantly, the paper's central claim—that Agentic CPT creates a better foundation model—implies that the benefits should compound with more sophisticated inference strategies. If AgentFounder-Base has genuinely superior agentic behavioral patterns embedded in its weights, then using it within a multi-agent framework or with inference-time search should yield even larger gains over general-purpose models used in the same frameworks. But this is untested. A practitioner who plans to deploy a multi-agent system has no evidence that AgentFounder-Base is a better component than, say, DeepSeek-V3.1 within that architecture.

The tool call distribution analysis (Section 3.6.2, Figure 8) and accuracy-vs-tool-calls analysis (Appendix B.5, Figure 13) hint at a related concern: on complex tasks (BrowseComp-en, HLE), AgentFounder uses extensive tool calls with heavy-tailed distributions extending past 40 calls, and accuracy declines sharply with increasing tool calls. This suggests that even within the single-agent paradigm, performance is constrained by the agent's ability to manage long interaction sequences—a constraint that might be alleviated by alternative architectures (e.g., hierarchical planning where a high-level planner delegates sub-tasks to specialized execution agents) but is not tested.

**What evidence exists in the paper.** The paper's Tables 1 and 2 report single-agent scores for AgentFounder and for most baselines, but some baselines (noted by the authors as using multi-agent architectures in Section 4.1) may have been evaluated under different inference conditions. The paper does not provide an ablation comparing single-agent vs. multi-agent configurations using the same AgentFounder-Base model. The Pass@n scaling result (Figure 10, Appendix B.2)—31.5% Pass@1 to 75.8% Pass@16 on BrowseComp-en—shows that diversity exists in AgentFounder's generations and can be exploited through repeated sampling, but repeated sampling is a weak form of inference-time compute scaling compared to structured search or multi-agent coordination.

**Mitigation status.** The paper does not frame the single-agent paradigm as a limitation or suggest exploration of alternative inference architectures. Section 4.1 discusses multi-agent and multi-modal deep research agents in the related work, acknowledging their existence without positioning AgentFounder relative to them. The computational cost and latency implications of the 128-tool-call budget and 128K context window are not discussed, leaving practitioners to infer that a typical BrowseComp-en question might require ~40 tool calls (from Figure 8) with corresponding API latency and cost that could make single-agent deployment impractical for real-time applications regardless of accuracy.

---

### 6.5 No Statistical Uncertainty Quantification for Any Benchmark Result

**The assumption or constraint.** The paper reports all benchmark results as point estimates—single numbers like 39.9% on BrowseComp-en, 72.8% on GAIA, 31.5% on HLE (Tables 1–3). No confidence intervals, standard deviations, bootstrap estimates, or statistical significance tests accompany any result. The test sets vary in size: GAIA uses 103 questions (text-only subset), the sizes of BrowseComp-en, BrowseComp-zh, Xbench-DeepSearch, WebWalkerQA, and other benchmarks are not specified in the paper but are likely in the range of 100–500 questions based on the original benchmark publications. The paper evaluates each model once (Pass@1) or with three rollouts (Pass@3) and reports the observed accuracy without quantifying the uncertainty in that observation.

**The consequence.** Many of the paper's headline claims involve small margins where statistical uncertainty could change the conclusion. AgentFounder-30B achieves 72.8% on GAIA vs. OpenAI-o3 at 70.5%—a margin of 2.3 percentage points. On a 103-question test set, this represents approximately 2–3 questions. If the test set were slightly different, or if the model were evaluated with a different random seed for decoding (temperature 0.85 is not deterministic), the ordering could reverse. The same applies to Xbench-DeepSearch (73.0% vs. DeepSeek-V3.1 at 71.0%, margin of 2.0 points), SEAL-0 (43.9% vs. DeepSeek-V3.1 at 42.6%, margin of 1.3 points), and Academic Browse (75.3% vs. DeepSeek-V3.1 at 65.0%, margin of 10.3 points—the last is comfortably large, but without knowing the test set size or variance, even this could be less robust than it appears).

The adaptability experiment (Table 3) reports gains ranging from +1.0 to +14.3 percentage points. The +1.0 gain on HLE with SFT-C is particularly fragile: if a single question flips from correct to incorrect due to decoding randomness, this gain could disappear entirely. The paper provides no way to assess whether this near-zero gain is meaningfully different from zero or merely reflects sampling noise.

The scaling curves (Figure 6) report average accuracy across benchmarks without error bars, making it impossible to determine whether the observed upward trend is reliable or whether individual data points might shift substantially under retraining with different random seeds. Given that each data point represents a full training run (from 0B to 315B tokens of CPT), retraining for variance estimation is expensive, but some form of uncertainty quantification—even bootstrapping over the test set—would substantially strengthen the paper's claims.

**What evidence exists in the paper.** The paper provides no uncertainty quantification whatsoever. Pass@3 (the best of three rollouts) provides some robustness to sampling variance at inference time but does not address test-set sampling variance or training-run variance. The paper does not report the number of test questions for any benchmark except GAIA (103, mentioned in a footnote). The evaluation protocol (Section 3.1.4) specifies decoding hyperparameters but does not describe any multiple-evaluation or statistical testing procedure.

**Mitigation status.** The paper does not acknowledge the absence of uncertainty quantification as a limitation. This is a standard practice in much of the LLM evaluation literature—papers routinely report point estimates without confidence intervals—but it is particularly consequential here because (1) the paper claims state-of-the-art on 10 benchmarks, many by small margins, and (2) the central claim of the paper is about a training pipeline innovation that requires substantial compute investment, where practitioners need to know whether the reported gains are reliable or could vanish under replication. The lack of even basic bootstrapped confidence intervals over test questions (which would cost almost nothing to compute) makes the quantitative results less actionable than they could be.

---

### 6.6 The Base Model for the FLOPs Comparison Is Never Trained or Evaluated

**The assumption or constraint.** The paper's fundamental claim is that Agentic CPT is a better use of compute than alternative training strategies—specifically, that spending compute on embedding agentic behavioral patterns during continual pre-training yields larger downstream gains than spending the same compute on other training activities. This claim requires a controlled comparison: Agentic CPT vs. some alternative use of the same compute budget, starting from the same base model, evaluated on the same benchmarks.

**The consequence.** The paper never performs this comparison. The adaptability experiment (Table 3) compares AgentFounder-Base (Qwen3-30B-A3B-Base + 315B tokens of Agentic CPT) against Qwen3-30B-A3B-Base (no CPT), with both receiving the same SFT. But these two starting points differ in total training compute by exactly the cost of 315B tokens of CPT. The observed gains—averaging 5.75% to 6.45% depending on SFT configuration—cannot be attributed to Agentic CPT specifically because there is no control condition that received equivalent additional training of a different type. Alternative explanations include:

- **More total training is better regardless of content.** If the model had been trained on 315B additional tokens of general web text, general code, or even repeated epochs of the original pre-training data, it might have achieved similar or larger gains.
- **The Qwen3 base model was undertrained.** If Qwen3-30B-A3B-Base had not reached convergence on its pre-training distribution, additional training of any kind might improve performance.
- **The specific content of CPT data matters less than its diversity.** FAS data spans many domains through the knowledge-to-question transformation. Perhaps the benefit comes from domain diversity rather than agentic behavioral patterns specifically.

Without a non-agentic CPT control group—trained on the same volume of data, with the same context window schedule, using the same training infrastructure—the paper cannot distinguish these explanations from its preferred "agentic inductive biases" narrative.

**What evidence exists in the paper.** No control group exists. The paper's training efficiency analysis (Section 3.6.1, Figure 7) shows that models with more CPT data achieve lower SFT loss, but this is equally consistent with "more training helps" as with "agentic training specifically helps." The scaling experiments (Section 3.5.2, Figure 6b) show that more CPT data improves performance—monotonically, with diminishing returns—but again without a control curve for non-agentic data at equivalent volume.

The closest the paper comes to a control is the implicit comparison with DeepSeek-V3.1 and Kimi-K2 in the model scaling experiment (Figure 6a), where the 30B-A3B AgentFounder outperforms larger models from other families. But these models differ in architecture, pre-training data, pre-training compute, tokenizer, and post-training recipe—they are not controlled comparisons in any meaningful sense.

**Mitigation status.** The paper does not acknowledge this as a limitation, does not discuss alternative explanations for the observed gains, and does not propose or conduct a non-agentic CPT control experiment. For a paper whose central contribution is the claim that *agentic* training data specifically—not just additional training—is what improves agent performance, the absence of this control is the single most important missing experiment. A practitioner deciding whether to invest in building an Agentic CPT pipeline (which requires implementing FAS and HAS data synthesis, managing two-stage training with different context lengths, and generating 315B tokens of synthetic agent data) versus simply doing more general pre-training on their existing data pipeline has no evidence from this paper to inform that decision. The paper demonstrates that AgentFounder-30B performs well; it does not demonstrate that Agentic CPT was necessary to achieve that performance, as opposed to any other form of additional training at equivalent scale.

## 7. Implications and Future Directions
- Field impact:
  - Establishes “agentic foundation models” as a new target: pre‑align agent behavior during CPT so that post‑training can focus on alignment refinements rather than capability acquisition (Figure 2; Section 3.6.1).
  - Demonstrates that large‑scale, tool‑free synthesis (FAS) and trajectory reuse (HAS) can unlock open‑source agents that approach or surpass commercial systems on several tasks (Tables 1–2).
- Immediate follow‑ups enabled by this work:
  - Stronger multilingual CPT: expand entity‑anchored memories and trajectory sources in under‑represented languages to address BrowseComp‑zh gaps (Section 3.2).
  - Verified step‑level supervision: combine HAS with selective live execution or simulator feedback to label alternatives, turning contrastive decisions into grounded causal signals.
  - Better judges and filters: calibrate LLM‑as‑judge with meta‑evaluation or human audits to further increase FAS/HAS data reliability (Appendix B.1 shows a promising 82% retained accuracy).
  - Multi‑modal and multi‑agent CPT: extend FAS/HAS to images/tables/videos (the paper cites a multimodal agent in related work) and to cooperative/competitive agent settings.
  - Safety and robustness: integrate SEAL‑style adversarial retrieval during CPT to harden agents against misinformation and tool failures (Section 3.1.3; Table 2).
- Practical applications:
  - Enterprise research assistants (market analysis, patent/literature reviews), scientific tooling (systematic evidence synthesis), investigative journalism support, and education (explanatory research reports).
  - The reported ACEBench gains (Table 6) suggest portability to general tool‑use agents beyond web research.

In short, the paper’s central move—teaching agentic behavior during continual pre‑training with scalable, mostly offline data—reframes how open‑source communities can train robust research agents. The combination of FAS, HAS, and long‑context two‑stage CPT materially advances both capability and efficiency, while leaving clear avenues for multilingual, multimodal, and causally grounded extensions.
