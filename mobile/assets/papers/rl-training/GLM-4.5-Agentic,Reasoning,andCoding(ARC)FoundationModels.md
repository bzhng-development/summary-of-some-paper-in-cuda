# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

**ArXiv:** [2508.06471](https://arxiv.org/abs/2508.06471)

## 🎯 Pitch

GLM-4.5 introduces an open-source Mixture-of-Experts (MoE) large language model that unifies top-tier agentic tool use, complex reasoning, and real-world coding abilities—domains historically siloed in both research and deployment. Through a novel multi-stage training and reinforcement learning pipeline, GLM-4.5 achieves state-of-the-art results across 12 ARC benchmarks while remaining highly efficient, closing the gap between open and proprietary models and paving the way for more capable, unified AI agents in real-world applications.

---

## 1. Executive Summary

This technical report introduces GLM-4.5, an open-source Mixture-of-Experts (MoE) large language model with 355B total parameters and 32B activated parameters, and its compact variant GLM-4.5-Air (106B parameters), designed to unify strong performance across agentic, reasoning, and coding (ARC) tasks. The model employs a **hybrid reasoning method** (supporting both extended chain-of-thought "thinking" mode and direct response mode) and is trained through multi-stage pre-training on 23T tokens followed by post-training with **Expert Model Iteration** (training separate expert models for reasoning, agent, and general chat domains, then distilling them into a single unified model via self-distillation) and specialized reinforcement learning techniques including **difficulty-based curriculum learning** (switching from moderate to extremely hard problems mid-training) and **dynamic sampling temperature** (increasing temperature when reward plateaus to maintain exploration). GLM-4.5 achieves 70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified, ranking 3rd overall among all evaluated models and 2nd on agentic benchmarks, while using roughly half the parameters of DeepSeek-R1 and one-third those of Kimi K2—establishing that strong ARC capabilities can be achieved through architectural efficiency and targeted post-training rather than raw parameter scaling alone.

## 2. Context and Motivation

### The Core Problem: Fragmented Excellence vs. Unified Competence

The central problem GLM-4.5 addresses is the **fragmentation of LLM capabilities**. As the paper states in the opening sentence of Section 1, LLMs are "rapidly evolving from general knowledge repositories into general problem-solvers." However, this evolution has been uneven. The paper identifies a specific gap: no single open-source model excels simultaneously across agentic, reasoning, *and* coding tasks—the three interdependent capabilities the authors argue define a "truly generalist model."

This is not merely an observation that different models have different strengths. It reflects a deeper tension in LLM development. The techniques that produce strong mathematical reasoning (e.g., extended chain-of-thought RL on competition math problems) often differ from those that build robust tool-use and agentic planning, which require handling multi-turn interactions with external environments. Coding excellence, particularly at the level of real-world software engineering (e.g., resolving GitHub issues on SWE-bench), demands yet another skill: understanding large codebases, reasoning about cross-file dependencies, and generating precise patches. Prior to GLM-4.5, open-source models that excelled in one of these domains typically showed significant weaknesses in the others. Proprietary models like OpenAI's o1/o3 and Anthropic's Claude Sonnet 4 had demonstrated strong ARC performance, but their architectures and training recipes were inaccessible. The gap the paper directly states is:

> "a single, powerful open-source model that excels across all three areas has remained elusive" (Section 1)

This fragmentation matters because real-world applications are rarely confined to a single capability. An AI-powered software development assistant must simultaneously understand natural language instructions (reasoning), navigate a codebase and write fixes (coding), and interact with tools like version control, test runners, and documentation search (agentic). Models that are strong at reasoning but weak at agentic tool use, or strong at isolated coding problems but weak at repository-scale software engineering, cannot serve as such unified assistants. The paper's stated ambition—"creating models with human-level cognitive capabilities across diverse domains" and moving "beyond task-specific excellence"—implicitly argues that the field needs integrated systems, not a toolkit of specialized models.

### Why This Problem Is Important: The Economic and Practical Case for Parameter Efficiency

Beyond the unification of capabilities, the paper makes a parallel economic argument that is embedded throughout its framing. GLM-4.5 achieves its competitive ARC performance with **355B total parameters (32B activated)**, which represents roughly:

- Half the total parameters of DeepSeek-R1 (671B)
- One-third the total parameters of Kimi K2 (1043B)
- Similar activated parameters to Kimi K2 (32B vs. 32B) and DeepSeek-V3 (37B)

This parameter efficiency is not presented as a mere technical curiosity. The paper explicitly highlights it in Section 1 ("Note that GLM-4.5 is highly parameter-efficient") and visually encodes it in Figure 2, where GLM-4.5 and GLM-4.5-Air are positioned on the Pareto frontier of SWE-bench Verified score vs. model parameters. The implication is clear: **parameter efficiency translates to deployment efficiency**. A model with 32B activated parameters costs less to serve per token than a dense model with comparable total capacity, and substantially less than models with 2-3× the total parameter count. For organizations deploying LLMs at scale—whether for agentic applications requiring many turns of interaction, or coding assistants processing large codebases—inference cost is often the dominant operational expense. A model that matches or exceeds competitors' performance at a fraction of the parameter count directly addresses this cost.

The economic significance is compounded by the model's release as open-source (model weights available on HuggingFace). Proprietary ARC-strong models like Claude Sonnet 4, o3, and Gemini 2.5 Pro are only accessible through paid APIs, creating both cost and vendor dependency. An open-source model with comparable performance removes both barriers, enabling organizations to self-host, customize, and build on the model without recurring API costs or usage restrictions. This is particularly important for agentic applications, where multi-turn interactions with the environment can accumulate substantial API bills quickly.

### Where Prior Approaches Fall Short

The paper does not engage in a detailed critique of specific prior models, but its positioning relative to existing work can be read between the lines of its architectural choices and training methodology. I'll trace the limitations of prior approaches along several axes, drawing from what the paper chooses to do differently.

#### The Dense Model Scaling Paradigm

Prior to the Mixture-of-Experts wave (popularized by DeepSeek-V3 and subsequent models), the dominant approach to improving LLM capabilities was scaling dense transformer models—larger hidden dimensions, more layers, more parameters in the feed-forward networks. Models like GPT-4 (dense, parameter count undisclosed but widely estimated at 1.8T), earlier versions of Gemini, and the LLaMA series scaled in this fashion. The limitation is well-understood by now but worth restating: dense scaling makes inference expensive because every parameter participates in every forward pass. For a model that is called repeatedly during multi-turn agentic interactions, or that processes 128K-token contexts (as GLM-4.5 does), the cumulative inference cost becomes prohibitive for many use cases.

MoE architectures address this by activating only a subset of parameters per token, decoupling total model capacity from per-token inference cost. GLM-4.5's 32B activated parameters out of 355B total means that each token forward pass costs roughly the same as a 32B dense model, while the model has access to 355B parameters of total representational capacity distributed across experts. However, MoE introduces its own challenges—load balancing across experts, training instability, and routing collapse—that prior work addressed with auxiliary loss functions (e.g., DeepSeek-V3's auxiliary balance loss). GLM-4.5's adoption of "loss-free balance routing" (Wang et al., 2024) represents a response to the specific weakness of auxiliary-loss-based approaches, which can interfere with the primary language modeling objective.

#### The Separation of "Thinking" and "Non-Thinking" Modes

Prior reasoning-focused models (notably OpenAI's o1/o3 series) were primarily designed for extended chain-of-thought reasoning, often at the expense of response latency on simpler queries. Asking o1 a simple factual question would still trigger a lengthy "thinking" phase before producing an answer. Conversely, general-purpose chat models like GPT-4 or Claude were optimized for fast, direct responses but lacked the capacity for deep, multi-step reasoning when needed. This created an awkward deployment choice: use a reasoning-specialized model and incur latency/cost penalties on simple queries, or use a general chat model and fail on complex problems.

GLM-4.5's hybrid reasoning design—supporting both thinking and non-thinking modes—directly addresses this fragmentation. The technical mechanism is training data balancing during the unified SFT stage: "recognizing that a prolonged thinking process is unnecessary for certain domains that demand quick responses (such as chit chat), we meticulously balanced training data containing full reasoning with data lacking explicit thought processes" (Section 3.1). This enables a single model to serve both roles, simplifying deployment and avoiding the overhead of routing queries between specialized models.

#### The Challenge of Combining Multiple Post-Training Objectives

Prior post-training pipelines typically optimized for a narrow set of objectives: instruction following (RLHF), mathematical reasoning (RFT, GRPO), or safety alignment, often applied sequentially or additively. This created a risk of **capability interference**, where optimizing for one objective degraded performance on another—a well-documented phenomenon in the RLHF literature where alignment training can reduce model capabilities on reasoning benchmarks.

GLM-4.5's Expert Model Iteration approach (Section 3) directly tackles this. Rather than applying all RL objectives to a single model sequentially, the paper trains **separate expert models** specialized in reasoning, agentic tasks, and general chat, then distills them into a unified model. This allows each expert to be optimized aggressively for its domain without worrying about interference with other capabilities. The distillation step then transfers the accumulated expertise back into a single model. This is analogous to how multi-task learning is sometimes approached in other domains—train specialists first, then combine—and it represents a departure from the monolithic RL pipeline common in prior work.

#### The Under-Exploration of Agentic RL at Scale

The paper explicitly notes that "compared to mathematics, RL for coding and scientific domains has received less attention in the literature" (Section 3.2, under "Code and Science RL"). This gap is especially pronounced for agentic RL, where the challenges are qualitatively different from math RL:

- **Sparse rewards**: Agentic tasks often have binary success/failure outcomes over long trajectories with many intermediate steps, making credit assignment difficult.
- **Environment interaction cost**: Each RL rollout requires interacting with external tools (web browsers, code executors, sandboxes), which is orders of magnitude slower than generating math solutions.
- **Format complexity**: Agentic outputs must conform to tool-calling schemas (JSON, XML-like tags), and format errors cause the entire trajectory to fail regardless of the reasoning quality.

Prior work had established strong results for math/code RL (DeepSeek-R1, DAPO, etc.), but agentic RL—particularly for web browsing, software engineering, and multi-turn tool use—remained at a much earlier stage of development. GLM-4.5's infrastructure for asynchronous, disaggregated agentic RL training (Section 3.5) and its techniques for function-calling RL (both step-wise and end-to-end multi-turn) represent attempts to close this gap.

#### Repo-Level Code Understanding

Traditional coding benchmarks (HumanEval, MBPP) evaluate models on isolated function-writing tasks with minimal context. Real-world software engineering—as measured by SWE-bench Verified—requires reasoning across multiple files with complex dependencies, understanding project structure, and making surgical edits without breaking existing functionality. Prior code models trained primarily on individual source files or code snippets lacked this repository-level understanding.

GLM-4.5 addresses this through its **repo-level code training** during the mid-training phase (Section 2.3): "we add concatenated code files from the same repository to learn cross-file dependency" and incorporate "model-filtered issues, pull requests (PRs), and commits from GitHub, with related issues, PRs, and commits concatenated into one context." This is part of a broader trend toward code models that understand software engineering rather than just code completion, but it represents a specific architectural commitment to mid-training on structured repository data rather than relying solely on scale to implicitly capture cross-file relationships.

#### The Long-Context Bottleneck for Agentic Tasks

Agentic tasks—particularly web browsing and software engineering—require processing long contexts. A web browsing agent might need to parse multiple search result pages, follow links, and synthesize information across dozens of web pages. A coding agent working on SWE-bench may need to process the entire codebase (often exceeding 100K tokens). Prior models often struggled with effective context utilization beyond 8K-32K tokens, either due to architectural limitations (RoPE base frequency not tuned for long contexts, inefficient attention implementations) or training data that consisted predominantly of short documents.

GLM-4.5 extends context length progressively during mid-training (4K → 32K → 128K) and adjusts RoPE's base frequency from 10,000 to 1,000,000 when extending to 32K. The inclusion of "large-scale synthetic agent trajectories" at the 128K context stage and "up-sampled long documents from the pre-training corpus" (Section 2.3) reflects a deliberate strategy to ensure the model can effectively use the full 128K context for agentic tasks, not just retrieve a needle from a haystack.

### How the Paper Positions Itself

GLM-4.5 positions itself at the intersection of several converging trends in LLM development, but with a distinctive emphasis on **unification through architectural efficiency** rather than scaling alone:

- **Against proprietary models**: The paper repeatedly benchmarks GLM-4.5 against top proprietary systems (o3, Claude Sonnet/Opus 4, Gemini 2.5 Pro, Grok 4) and shows competitive or superior performance, particularly on agentic tasks where GLM-4.5 ranks 2nd overall (behind only o3). This positioning is reinforced by the open-source release, which makes the implicit claim that competitive ARC performance should not be locked behind commercial APIs.

- **Against other open-source models**: GLM-4.5 is directly compared to DeepSeek-R1-0528, DeepSeek-V3-0324, Kimi K2, and Qwen3-235B, with particular emphasis on parameter efficiency. The Figure 2 Pareto frontier plot is specifically designed to highlight that GLM-4.5 achieves high SWE-bench scores with substantially fewer parameters than DeepSeek-R1 and Kimi K2.

- **As a generalist, not a specialist**: Unlike DeepSeek-R1, which was primarily positioned as a reasoning model, or Claude Sonnet 4 as a coding agent, GLM-4.5 claims strong performance across all three ARC dimensions simultaneously. The hybrid reasoning design (thinking + non-thinking modes) and the expert model distillation approach are both in service of this generalist positioning.

- **Infrastructure-innovative, not algorithmically novel**: The paper is transparent that its core RL algorithm "builds upon the GRPO framework" (Section 3.2). The innovations are primarily in the **training recipe** (difficulty-based curriculum, dynamic temperature, multi-stage mid-training, expert iteration) and the **infrastructure** (the Slime RL framework with asynchronous agentic support, FP8 inference for rollout acceleration, disaggregated training-rollout architectures). This is a pragmatic positioning: the paper does not claim fundamental algorithmic breakthroughs but rather demonstrates that careful combination of existing techniques with strong engineering can push the Pareto frontier.

- **Toward "AGI" language without overclaiming**: The paper uses the term AGI in its opening paragraph ("The ultimate ambition, often associated with Artificial General Intelligence, is to create models with human-level cognitive capabilities across diverse domains") but immediately grounds this in concrete benchmarks rather than philosophical claims. The ARC framework (agentic, reasoning, coding) serves as an operationalization of "general problem-solving" that can be measured and compared. This positions GLM-4.5 not as a claim of AGI achieved, but as a step toward models that demonstrate competence across the specific dimensions the authors argue define generalist capability.

In essence, GLM-4.5 enters a field where individual models have pushed the state of the art in reasoning (DeepSeek-R1, o3), coding (Claude Sonnet 4, Kimi K2), and tool use (various agentic frameworks), but where no open-source model has demonstrated strong performance across all three simultaneously. The paper's contribution is demonstrating that this unification is achievable through architectural efficiency (MoE with deep-but-narrow design), multi-stage data-aware pre-training and mid-training, expert model iteration for post-training, and purpose-built RL infrastructure for agentic tasks—all at a parameter count that makes deployment practical. The open-source release and the CC-Bench manual evaluation (Section 4.3.2) further position this as not just a benchmark exercise but a genuinely usable model for real-world development scenarios.

## 3. Technical Approach

### 3.1 Reader Orientation

GLM-4.5 is a Mixture-of-Experts large language model that functions as a general-purpose problem-solver supporting both deep reasoning (extended chain-of-thought "thinking" mode) and fast direct responses ("non-thinking" mode) within a single unified architecture. The core problem it solves is the fragmentation of LLM capabilities across agentic, reasoning, and coding (ARC) tasks: prior open-source models excelled in one domain but showed significant weaknesses in others, while proprietary models achieving strong ARC performance were inaccessible. The "shape" of the solution is a multi-stage training pipeline that first builds broad competence through carefully curated pre-training and mid-training on 23T tokens, then develops specialized domain expertise through Expert Model Iteration—training separate expert models optimized for reasoning, agentic tasks, and general chat via reinforcement learning, and finally distilling them back into a single unified model that supports hybrid reasoning modes.

### 3.2 Big-Picture Architecture (Diagram in Words)

The GLM-4.5 system is built through a pipeline with four major stages:

1. **Pre-training (Section 2.1–2.4):** A Mixture-of-Experts transformer with 355B total parameters (32B activated) is trained on a 15T-token corpus of webpages, code, multilingual documents, and math/science content. The deep-but-narrow architecture (89 MoE layers, hidden dimension 5120) uses loss-free balance routing, partial RoPE with grouped-query attention, and QK-Norm. Training uses the Muon optimizer with cosine decay and batch size warmup.

2. **Mid-training (Section 2.3):** Three successive domain-specific training stages on an additional 7–8T tokens—repo-level code training (concatenated files from GitHub repositories with issues/PRs/commits, context extended to 32K), synthetic reasoning data training (model-generated reasoning chains for math/science/competition problems), and long-context agent training (context extended to 128K with synthetic agent trajectories and up-sampled long documents).

3. **Expert Model Iteration – Expert Training (Section 3.1–3.3):** Three separate expert models are trained via SFT cold-start followed by specialized reinforcement learning:
   - **Reasoning Expert:** GRPO-based RL on math, code, and science problems with difficulty-based curriculum learning and dynamic sampling temperature.
   - **Agent Expert:** RL on web-search and software-engineering tasks with outcome supervision, format penalties, and iterative self-distillation.
   - **General Expert:** Holistic RL covering instruction following, function calling, and pathology reduction using multi-source feedback (rule-based, human, and AI).

4. **Unified Training (Section 3.1, "Overall SFT"):** The three expert models' outputs are distilled into the base model through supervised fine-tuning on millions of examples spanning all domains, carefully balancing thinking-mode and direct-mode responses to create the hybrid reasoning capability. The output is a single model—GLM-4.5—that can handle all ARC tasks.

Information flows: raw text → pre-training → base model → mid-training → domain-adapted base model → expert SFT + RL → three specialist models → unified SFT distillation → final GLM-4.5 model. At inference time, the model routes between thinking and non-thinking modes based on the nature of the query.

A supporting infrastructure component—the **Slime RL framework** (Section 3.5)—handles the asynchronous, disaggregated training needed for agentic RL, with separate GPU pools for training and rollout, FP8-accelerated inference for data generation, and a centralized data pool for heterogeneous agent frameworks.

### 3.3 Roadmap for the Deep Dive

The detailed breakdown below follows the training pipeline chronologically, which mirrors how capability is progressively built up in the model:

- **First, the MoE architecture and pre-training data strategy** (Sections 2.1–2.4), because the model's parameter efficiency and knowledge foundation are established here. Understanding the deep-but-narrow architecture choice and the multi-source data curation is prerequisite to understanding why the model performs well at its parameter count.

- **Second, the mid-training stages** (Section 2.3), because these bridge the gap between general language knowledge and domain-specific capability. The repo-level code training, synthetic reasoning data, and long-context agent data are where the model acquires the specific skills that distinguish it from generic pre-trained models.

- **Third, the Expert Model Iteration framework** (Section 3.1–3.3), because this is the core post-training innovation. I'll cover the SFT cold-start, then reasoning RL with its difficulty-based curriculum and dynamic temperature mechanisms, then agentic RL with its asynchronous infrastructure and iterative distillation, then general RL with its multi-source feedback system.

- **Fourth, the reasoning RL techniques in depth** (Sections 3.2), because these contain the most novel methodological contributions: the two-stage difficulty curriculum, the single-stage 64K-length RL finding, the dynamic temperature mechanism, the code RL token-weighted loss, and the science RL data quality findings.

- **Fifth, the agentic RL system** (Section 3.3), because agentic training introduces qualitatively different challenges (environment interaction, format constraints, sparse rewards) that required purpose-built infrastructure. The data synthesis pipeline, the outcome-supervision-with-format-penalty reward structure, and the iterative distillation loop each warrant detailed explanation.

- **Sixth, the RL infrastructure** (Section 3.5), because the Slime framework's design—particularly the disaggregated asynchronous architecture for agentic tasks and the FP8 rollout acceleration—is what makes the agentic RL training feasible at scale and represents a significant engineering contribution.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **technical report** describing a model and its training pipeline. The core idea is that strong, unified ARC performance can be achieved through (1) architectural efficiency in the MoE design, (2) multi-stage, data-aware pre-training and mid-training, and (3) Expert Model Iteration—training domain specialists via RL and then distilling them into a unified model with hybrid reasoning modes.

---

#### Mixture-of-Experts Architecture Design

The GLM-4.5 series adopts a Mixture-of-Experts (MoE) architecture, a design pattern where the model's feed-forward layers are partitioned into multiple "expert" sub-networks, and each input token is routed to only a subset of these experts. This decouples total model capacity (the sum of all expert parameters) from per-token inference cost (only the activated experts' parameters are used per forward pass). GLM-4.5 has 355B total parameters distributed across 160 routed experts plus 1 shared expert, with only 8 experts activated per token, yielding 32B activated parameters. GLM-4.5-Air uses 128 routed experts with the same 8-expert activation, yielding 12B activated parameters out of 106B total.

**Architectural choices compared to prior MoE models.** The paper explicitly contrasts its design with DeepSeek-V3 and Kimi K2 along several dimensions (Table 1):

- **Depth vs. width tradeoff:** GLM-4.5 reduces the hidden dimension (5120 vs. 7168 for DeepSeek-V3 and Kimi K2) and the MoE intermediate dimension (1536 vs. 2048) while increasing the number of layers (89 MoE layers vs. 58 for DeepSeek-V3 and 60 for Kimi K2). The paper states: "we reduce the width (hidden dimension and number of routed experts) of the model and increase its height (number of layers), as we found that deeper models exhibited better reasoning capacity." This is a non-obvious finding—the conventional wisdom from dense transformer scaling was that width and depth should be scaled together, but GLM-4.5's design suggests that for MoE architectures targeting reasoning tasks, depth provides more benefit per parameter than width.

- **Attention head configuration:** GLM-4.5 uses 96 attention heads with dimension 128 each, compared to DeepSeek-V3's 128 heads (dimension 192) and Kimi K2's 64 heads. This is "2.5 times more attention heads" relative to what a standard configuration would use for a 5120 hidden dimension. The paper reports a counterintuitive finding: "while this increased head count does not improve training loss compared to models with fewer heads, it consistently improves performance on reasoning benchmarks such as MMLU and BBH." This suggests that multi-head attention with smaller per-head dimension improves the model's ability to attend to diverse patterns in reasoning problems, an inductive bias that manifests at evaluation time even without loss improvements during training.

- **QK-Norm:** GLM-4.5 incorporates query-key normalization (Henry et al., 2020), which applies LayerNorm to the query and key vectors before computing attention scores: `$\text{Attention}(Q, K, V) = \text{softmax}(\text{LN}(Q)\text{LN}(K)^T / \sqrt{d_k})V$`. This "stabilize[s] the range of attention logits" by preventing the dot products from growing with embedding dimension. DeepSeek-V3 and Kimi K2 do not use QK-Norm, making this a distinguishing feature of the GLM-4.5 series. GLM-4.5-Air omits QK-Norm, suggesting it may provide diminishing returns at smaller scales or that the stability benefit is more critical for deeper models.

- **Partial RoPE:** The paper mentions using "partial RoPE" in the attention mechanism without specifying the fraction. RoPE (Rotary Position Embedding) encodes position information by rotating query and key vectors. "Partial" RoPE typically means applying the rotary transformation to only a subset of the attention head dimensions (e.g., the first 25-50%), with the remaining dimensions using no positional encoding or a different scheme. This is motivated by findings that full RoPE can interfere with long-context extrapolation.

- **Multi-Token Prediction (MTP) layer:** Both GLM-4.5 variants include one MoE layer as an MTP layer, following Gloeckle et al. (2024). During training, this layer predicts the next token at an additional future position (beyond the immediate next token), providing an auxiliary training signal. During inference, it enables speculative decoding: the MTP layer can cheaply predict multiple future tokens which are then verified by the main model forward pass, potentially reducing latency. The MTP loss weight `$\lambda$` is set to 0.3 for the first 15T tokens and 0.1 thereafter.

**Loss-free balance routing.** MoE models face a load-balancing problem: if the router consistently sends tokens to a small subset of experts, those experts become over-specialized, the remaining experts become under-trained ("dead experts"), and compute is wasted on idle capacity. Traditional approaches (e.g., DeepSeek-V3) add an auxiliary loss term to the training objective that penalizes imbalanced expert assignment. GLM-4.5 instead uses "loss-free balance routing" (Wang et al., 2024), which adds a learned bias term to each expert's routing logit and updates these biases based on expert load without gradient flow through the main loss. Concretely, if an expert is receiving more than its fair share of tokens, its bias is reduced, making it less likely to be selected; under-utilized experts have their biases increased. The bias update rate is set to 0.001 for the first 15T tokens and 0.0 thereafter (frozen biases). Additionally, "auxiliary sequence-level balance loss with a 0.0001 weight" is applied "to avoid extreme imbalance within any single sequence," meaning that even with loss-free routing at the batch level, the model is lightly penalized if a single sequence routes all its tokens to the same small set of experts.

**Sigmoid gates for MoE layers.** The paper mentions using "sigmoid gates for MoE layers," following DeepSeek-V3. In standard MoE, the routing function is a softmax over experts, which enforces competition—increasing one expert's probability decreases others. Sigmoid gates replace softmax with independent sigmoid activations per expert, meaning the decision to route to expert `$i$` does not directly suppress routing to expert `$j$`. This allows more flexible expert combinations and can improve training stability.

**Attention mechanism specifics.** GLM-4.5 uses Grouped-Query Attention (GQA) with 96 query heads and only 8 key-value heads. This means that 12 query heads share each key-value head (96 ÷ 8 = 12), substantially reducing the KV-cache memory footprint during inference compared to full multi-head attention (which would require 96 separate KV heads). For a 128K context window, the KV cache can dominate inference memory, so this 12:1 reduction is practically significant.

**MTP layer architecture.** The MTP layer is "an additional MoE layer" appended after the final transformer layer. During training, it receives the same hidden state as the final layer and predicts an additional future token. The total loss is the standard next-token prediction loss plus `$\lambda \cdot \mathcal{L}_{\text{MTP}}$` where `$\mathcal{L}_{\text{MTP}}$` is the cross-entropy loss for the MTP prediction. The MTP layer's parameters are included in the total parameter count (355B / 106B), meaning they contribute to model capacity but not to the per-token inference cost when not using speculative decoding (since the MTP layer is only used for speculative decoding at inference time, not the main forward pass).

---

#### Pre-Training Data Strategy

The pre-training corpus of 15T tokens is constructed from multiple sources with distinct processing pipelines, reflecting a deliberate strategy of quality-aware curation rather than uniform treatment of all data.

**Web data processing and quality bucketing.** The majority of pre-training documents come from English and Chinese webpages. Inspired by Nemotron-CC (Su et al., 2024), the crawled webpages are divided into "buckets of different quality scores." Rather than using a simple keep/discard binary filter, this bucketing approach enables **up-sampling of high-quality documents** while discarding only the lowest-quality bucket. The paper states that "the bucket with the highest quality scores contributes over 3.2 epochs during pre-training," meaning the model sees the highest-quality documents more than three times on average, while lower-but-acceptable quality documents appear fewer times. This creates a gradient of emphasis: the model learns core knowledge and patterns from repeated exposure to high-quality text while still benefiting from the coverage and diversity of lower-quality data seen less frequently.

A specific challenge the paper identifies is "a large number of similar webpages automatically generated from templates and assigned high scores." These template-generated pages pass quality filters because they contain well-structured text, but they represent repetitive, non-diverse content that wastes training compute. Standard MinHash deduplication—which detects near-duplicate documents based on n-gram overlap—fails here because the pages are structurally similar but textually distinct (different products in the same template format). The paper addresses this with **SemDedup** (Abbas et al., 2023), which operates on document embeddings: documents are embedded into a vector space, and those within a certain cosine similarity threshold are considered near-duplicates and removed. This catches semantically similar documents even when surface-level text differs, eliminating the template-generated redundancy.

**Multilingual data.** The multilingual corpus comes from "both our crawled webpages and Fineweb-2" (Penedo et al., 2025). A quality classifier "that judges the educational utility of documents" is applied, and high-quality multilingual documents are up-sampled. This is distinct from simply filtering by language or perplexity—the classifier specifically assesses educational value, meaning the model preferentially learns from multilingual text that conveys substantive knowledge rather than low-information content like comment sections or navigation menus.

**Code data processing.** Source code from GitHub undergoes a three-tier quality classification ("high-quality, medium-quality, and low-quality") using "language-specific quality models." High-quality code is up-sampled; low-quality code is excluded entirely. All source code data uses the **Fill-In-the-Middle (FIM)** training objective (Bavarian et al., 2022): instead of always predicting text left-to-right, the model is trained to predict a middle span given a prefix and a suffix. Specifically, a code sample is split into three parts—`$\langle \text{prefix} \rangle$`, `$\langle \text{middle} \rangle$`, `$\langle \text{suffix} \rangle$`—and the model must generate the middle part given the prefix and suffix. This teaches the model to complete code given surrounding context, which is the predominant use case in code editing and completion tools (e.g., filling in a function body between its signature and the code that follows).

For code-related web documents (e.g., Stack Overflow posts, documentation pages), a two-stage retrieval process is used. First, documents are selected from the text pre-training corpus based on "presence of HTML code tags, or identification by a FastText classifier trained to detect code-related content." FastText (Joulin et al., 2017) is a lightweight text classifier that uses bag-of-ngram features, making it efficient enough to run over the entire web corpus. Second, the retrieved documents undergo quality assessment using a dedicated model that classifies them into high/medium/low quality, following the same sampling strategy as source code. A "fine-grained parser" re-parses the selected web pages "to better preserve the formats and contents of the code," addressing the common problem that standard HTML-to-text extraction often garbles code formatting.

**Math and science data.** The paper collects documents related to mathematics and science from "webpages, books, and papers." A large language model scores candidate documents "based on the ratio of educational content about mathematics and science," and a "small-scale classifier" is trained to predict these scores. Documents above a score threshold are up-sampled. This two-step approach (LLM scoring → classifier distillation) is more scalable than running the LLM over the entire corpus: the LLM provides high-quality annotations on a subset, the lightweight classifier generalizes to the full corpus, and the classifier's predictions determine sampling weights during training.

**Two-stage pre-training.** The pre-training is divided into two stages. In the first stage, the model is trained on general documents from webpages. In the second stage, source code from GitHub and webpages related to coding, mathematics, and science are up-sampled. This sequencing reflects a curriculum: broad language understanding is established first on general text, after which domain-specific capabilities are intensified. Training cold on a heavily code-skewed distribution would produce a model with poor general language performance; training entirely on a balanced distribution would under-invest in the technical domains critical for ARC tasks.

---

#### Mid-Training: Domain-Specific Capability Boosting

After pre-training on 15T tokens, the model undergoes three mid-training stages that collectively consume approximately 7-8T additional tokens. Unlike pre-training on large-scale general documents, these stages use medium-size domain-specific datasets and incorporate instruction-like data. Table 3 in the paper shows the sequence: general pre-training corpus (15T) → code and reasoning continual pre-training corpus (7T) with context extension to 32K, followed by repo-level code data (500B tokens), synthetic reasoning data (500B tokens), and long-context plus agent data (100B tokens) with context extension to 128K.

**Context length extension strategy.** During pre-training, the maximum sequence length is kept at 4,096 tokens. In mid-training, it is first extended to 32,768 and then to 131,072. When extending from 4K to 32K, the paper adjusts RoPE's base frequency from 10,000 to 1,000,000. RoPE encodes positions as:

$$\text{RoPE}(x, pos) = x \cdot \begin{pmatrix} \cos(pos \cdot \theta_0) & -\sin(pos \cdot \theta_0) \\ \sin(pos \cdot \theta_0) & \cos(pos \cdot \theta_0) \end{pmatrix}$$

where the base frequency `$\theta_0$` determines the wavelength of the rotary encoding—higher base frequency means longer wavelengths, which means the positional encoding varies less with position, which empirically improves the model's ability to attend to tokens at extreme relative distances. The jump from 10,000 to 1,000,000 is a 100× increase, which dramatically stretches the effective range of the positional encoding. The paper does not adjust RoPE base frequency again when moving from 32K to 128K, suggesting the 1M base frequency is sufficient for the full 128K context.

**Packing strategy.** During pre-training, the paper does not use best-fit packing, stating that "random truncation is a good data-augmentation strategy for pre-training documents." Random truncation means that when a document exceeds the maximum sequence length, it is cut off at a random point rather than at the token limit, creating different partial views of the same document across epochs and acting as a form of data augmentation. For mid-training datasets, best-fit packing (Ding et al., 2024) is applied to "avoid truncating the reasoning process or repo-level code." Best-fit packing concatenates multiple short documents into a single sequence (up to the maximum length) while minimizing wasted space, ensuring that the valuable reasoning chains and cross-file code dependencies are not arbitrarily severed mid-thought.

---

#### Repo-Level Code Training

This mid-training stage consumes 500B tokens of repository-structured code data. The core technique is concatenating "code files from the same repository to learn cross-file dependency." In a typical code pre-training setup, files from different repositories are randomly interleaved, meaning the model never learns that `import utils` in `main.py` refers to `utils.py` in the same directory. By grouping all files from a repository into a single training sequence (or a set of sequences for large repos), the model can learn intra-project structure: function definitions in one file being called from another, class hierarchies spanning multiple files, and project-level conventions.

Beyond file concatenation, the paper includes "model-filtered issues, pull requests (PRs), and commits from GitHub, with related issues, PRs, and commits concatenated into one context and commits organized in a diff-like format." This is a more sophisticated training signal. An issue describes a bug or feature request; a related PR contains the code changes that address it; related commits show incremental steps toward the solution. By concatenating these elements into a single context window, the model learns to map natural language problem descriptions (issues) to code solutions (PRs/commits), which directly supports the SWE-bench task of resolving GitHub issues. The diff-like format for commits teaches the model to understand code changes as structured edits rather than as complete file rewrites.

This stage extends training sequence length from 4K to 32K to accommodate large repositories. A repository with 50 files at an average of 500 tokens each is 25K tokens—impossible to represent in a 4K context but manageable at 32K.

---

#### Synthetic Reasoning Data Training

This stage adds 500B tokens of "synthetic reasoning content for math, science, and coding competitions." The data generation process works as follows: the team collects "a large number of questions and answers related to the reasoning tasks from webpages and books," then uses a pre-existing "reasoning model" (likely an earlier GLM variant or an external model) to synthesize detailed reasoning processes for each question-answer pair. The synthetic reasoning traces—step-by-step chains showing how to arrive at the answer—become the training data.

This approach is fundamentally different from simply training on question-answer pairs. By including the reasoning process, the model learns not just what the correct answer is, but how to derive it. During inference in thinking mode, the model reproduces this structured reasoning: it generates a step-by-step chain of thought (demarcated by special tokens, visible in the Figure 4 example with ` thinking...`) before producing the final answer. The synthetic reasoning data effectively teaches the model to "show its work" in the style that the reasoning model demonstrated.

---

#### Long-Context and Agent Training

This stage extends context to 128K and incorporates 100B tokens of data designed to build agentic and long-context capabilities. Two data sources are used: up-sampled long documents from the pre-training corpus (ensuring the model can effectively process long-form text like technical documentation, research papers, and book chapters) and "large-scale synthetic agent trajectories."

Synthetic agent trajectories are sequences of interactions between a model and an environment (tools, search engines, code executors) that demonstrate successful task completion. For example, a trajectory for a web-search task might show: user query → search query formulation → search results parsing → link clicking → page reading → answer synthesis → final response. By training on these trajectories, the model learns not just what actions to take but the sequential pattern of agentic behavior—the rhythm of thinking, acting, observing results, and adapting.

This stage is positioned last in the mid-training pipeline because agentic behavior builds on all previous capabilities: code understanding (from repo-level training), structured reasoning (from synthetic reasoning data), and long-context processing (from up-sampled documents). The sequencing ensures that when the model learns agentic patterns, it already possesses the underlying competencies to execute them effectively.

---

#### Post-Training Overview: Expert Model Iteration

The post-training process is organized into two distinct stages around a central idea: **Expert Model Iteration**. Rather than applying all post-training objectives to a single model sequentially—which risks capability interference where gains in one domain degrade performance in another—the paper trains separate expert models specialized in reasoning, agentic tasks, and general chat, then distills them into a unified model.

**Stage 1 (Expert Training):** Three experts are trained independently, each starting from the same mid-trained base model:
- A **reasoning expert** optimized through RL on math, code, and science problems with verifiable answers.
- An **agent expert** optimized through RL on web-search and software-engineering tasks with environment feedback.
- A **general chat expert** optimized through holistic RL covering instruction following, function calling, safety, and pathology reduction.

Each expert undergoes SFT cold-start followed by domain-specific RL. The experts can push aggressively in their domains without concern for degrading other capabilities—the reasoning expert can maximize math accuracy even if it makes the model worse at chit-chat, because the general chat capability will come from a different expert.

**Stage 2 (Unified Training):** The outputs of the three experts are distilled into a single model via supervised fine-tuning. The paper states: "we collect millions of samples covering reasoning tasks (math, code, science, etc.), general chat (writing, translation, summarization, chit chat, etc.), agentic tasks (basic tool using, coding ability especially for authentic project development, etc.), and long-context understanding tasks from the previously trained expert models, and train the base model with a maximum context length of 128K tokens."

The key design choice in unified training is **hybrid reasoning data balancing**. The paper trains on two types of examples: those containing full chain-of-thought reasoning (the expert's step-by-step thinking process followed by the answer) and those containing only the direct answer with no explicit reasoning. By balancing these in the training data, the final model learns to operate in both modes—it can engage in extended thinking for complex reasoning and agentic tasks, or respond directly for simple queries. The paper notes: "recognizing that a prolonged thinking process is unnecessary for certain domains that demand quick responses (such as chit chat), we meticulously balanced training data containing full reasoning with data lacking explicit thought processes."

At inference time, the model does not need an explicit mode-switching mechanism. The training data's structure teaches the model implicitly: just as it learned from examples during training whether a particular type of query warrants thinking, it reproduces this behavior during inference. The ` thinking` and ` response` markers in the training data (visible in Figure 4) provide the explicit format tokens that demarcate the two modes within a single generation.

#### Cold-Start Supervised Fine-Tuning

Before RL training, each expert undergoes a "cold-start" SFT phase. The paper describes this as using "a small set of supervised fine-tuning (SFT) data with extended Chain-of-Thought (CoT) responses," with the purpose being to ensure "each expert model possesses adequate foundational ability prior to the reinforcement learning phase." This is critical because RL from a completely untrained policy—where the model has no notion of what a good reasoning trace or agent trajectory looks like—would be catastrophically inefficient. The cold-start SFT provides a reasonable initial policy that the RL process can then refine.

For the reasoning expert, the cold-start data consists of questions with detailed step-by-step solutions. For the agent expert, it consists of tool-calling trajectories with correct function invocations. For the general expert, it consists of diverse chat examples.

The SFT objective is standard next-token prediction loss with teacher forcing: given a prompt and the target response, the model is trained to maximize the probability of the target tokens given the prompt and all previous target tokens. The loss is computed only on the response tokens, not the prompt tokens (the model is not penalized for its predictions on the prompt, only on its generation of the answer).

#### Function Call Template Innovation

A specific SFT data preparation innovation described in Section 3.1 concerns the format for representing function calls. The standard approach is to use JSON—the model generates a JSON object containing the function name and parameters. However, when function call parameters contain code segments, "a substantial proportion of characters within the code require escaping" (e.g., quotes within strings, newlines, special characters). This forces the model to "generate extensive escape characters, thereby increasing the learning burden."

GLM-4.5 introduces an XML-like template that encapsulates function call keys and values within special token tags. The structure (visible in Figure 4) is:

```
<tool_call>function_name
<arg_key>param1</arg_key>
<arg_value>value1</arg_value>
<arg_key>param2</arg_key>
<arg_value>value2</arg_value>
</tool_call>
```

This format "substantially reduces the necessity for character escaping in code segments, as the vast majority of code can be represented in its native form without escaping." The XML-like tags serve as explicit delimiters that separate the function name from the arguments and the arguments from each other, eliminating the need for JSON's syntactic overhead. The paper reports that "experimental results demonstrate that the proposed function call template does not compromise the performance of function call execution while reducing escaping."

This is a practical engineering choice driven by the observation that for agentic models where function calling is a core capability, the cognitive load of generating escape characters in code-heavy parameters is a non-trivial obstacle. By eliminating this load, the model can focus its representational capacity on the actual content of the function calls rather than on formatting.

---

#### Rejection Sampling and Data Quality Filtering

When generating SFT data from expert models (both for cold-start and for unified training), the paper applies a comprehensive multi-stage filtering pipeline to ensure data quality:

1. **Format and basic quality filtering:** Samples that are "repetitive, excessively short, or truncated" are removed, as are "those that fail to conform to valid reasoning formats" (e.g., thinking mode traces that don't properly demarcate reasoning from response).

2. **Correctness verification for objective answers:** For math, science, and coding problems with verifiable answers, the generated solution's final answer is checked against the ground truth. Only samples with correct answers are retained. This is the standard rejection sampling approach: generate multiple candidate solutions, keep only the correct ones, and train on those.

3. **Reward model filtering for subjective questions:** For prompts without ground-truth answers (open-ended writing, subjective advice), a trained reward model scores the generated responses, and low-scoring responses are filtered out.

4. **Tool-calling verification:** For agentic SFT data, the system ensures "adherence to proper tool invocation protocols and verification that trajectories reach the expected terminal states." This means checking that function calls are in the correct format, that the sequence of actions actually accomplishes the task, and that the trajectory ends in a completion state rather than an error or infinite loop.

**Prompt selection and response-level scaling.** The paper reports a specific finding about data efficiency: "We experimented with removing the prompts in the bottom 50% based on response lengths, resulting in a 2%-4% improvement in math and science tasks, despite training with only half the data." This suggests that prompts which produce short responses—likely because they are too easy or too poorly specified to elicit substantive reasoning—provide a weaker training signal. Training on only the prompts that elicit detailed responses (the top 50% by response length) is actually more effective than training on the full set.

Furthermore, the paper found "that applying response scaling to these hard prompts can lead to further gains. Generating four responses for each prompt brought an additional 1%-2% improvement." Response scaling means generating multiple candidate responses per prompt (rather than just one) and training on all correct candidates. This increases the effective number of training examples without requiring additional prompts, and provides the model with multiple valid reasoning paths to the same answer, potentially improving generalization.

**Automatic agentic SFT data construction.** Constructing high-quality agentic SFT data at scale requires solving a chicken-and-egg problem: you need an agentic model to generate training trajectories, but you need training trajectories to build an agentic model. The paper's approach involves four automated steps:

1. **Agentic framework and tool collection:** Gather "a set of agentic frameworks and real-world tool APIs and MCP servers," while also "leveraging LLMs to automatically construct and simulate a batch of tools." The real tools provide authentic interaction patterns; the synthetic tools fill gaps and increase diversity.

2. **Task synthesis:** For mature frameworks, LLMs are used "to comprehend their functionalities and automatically generate relevant queries or tasks." For fragmented/disparate tools, a representative subset is selected and LLMs construct tasks about this subset. The tasks span both single-step tool calls (e.g., "get the weather in Beijing") and multi-step tool calls (e.g., "book a flight to Shanghai, then find hotels near the airport").

3. **Trajectory generation:** For each synthesized task, existing LLMs generate tool-call trajectories. For multi-step tasks, an "LLM as a user simulator" converts the task into a multi-turn dialogue where the simulated user responds to the agent's actions, providing the interactive dynamics that the agent model must learn to handle.

4. **Quality filtering:** Multiple "judge agents" evaluate whether each trajectory successfully completes the task. Only successful trajectories are retained. Using multiple judges (rather than a single one) provides a more robust quality signal through ensemble judgment.

---

#### Reasoning Reinforcement Learning

Reasoning RL targets domains where correctness can be determined programmatically or with objective clarity: mathematics, code generation, and scientific reasoning. The paper uses GRPO (Group Relative Policy Optimization; Shao et al., 2024) as the base algorithm, with a specific modification: "excluding the KL loss term."

**GRPO objective (modified).** The base GRPO algorithm samples multiple responses from the old policy, computes rewards, and optimizes relative to the group mean. The simplified objective used in GLM-4.5 can be expressed as:

$$L_{\text{RL}}(\theta) = \mathbb{E}_{x \sim D} \left[ \frac{1}{K} \sum_{i=1}^{K} \left( r(x, y_i) - \bar{r}(x) \right) \right]$$

where for a given prompt `$x$`, `$K$` responses `$\{y_1, \ldots, y_K\}$` are sampled from the previous policy `$\pi_{\text{old}}$`, `$r(x, y_i)$` is the scalar reward for response `$i$` (e.g., 1 if the final answer matches the ground truth, 0 otherwise), and `$\bar{r}(x) = \frac{1}{K} \sum_{i=1}^{K} r(x, y_i)$` is the mean reward across the `$K$` samples for that prompt.

**What it computes:** For each prompt, the model generates `$K$` candidate responses using its current policy. Each response receives a reward based on whether it arrives at the correct answer. The optimization signal is the deviation of each response's reward from the average reward for that prompt—responses that are better than average receive positive weight, responses that are worse than average receive negative weight. The model parameters are updated to increase the probability of above-average responses and decrease the probability of below-average responses. Only the model-generated tokens are optimized; tokens from the prompt or environment feedback are not included in the loss computation.

**Why this form:** The group-relative normalization (subtracting `$\bar{r}(x)$`) serves as a baseline that reduces variance in the gradient estimate without requiring a separate value function. On a prompt where all `$K$` responses are correct (all rewards = 1), `$\bar{r}(x) = 1$` and the gradient is zero—the model has already mastered this type of problem and no update is needed. On a prompt where all responses are incorrect (all rewards = 0), the gradient is also zero—the problem is too far beyond the model's current capability to provide a meaningful learning signal. The signal is strongest on prompts where some responses succeed and some fail, creating a mix of positive and negative examples. This property is what motivates the difficulty-based curriculum learning described next.

**Excluding the KL loss term.** Standard GRPO includes a KL divergence penalty that discourages the updated policy from deviating too far from a reference policy (usually the initial SFT model). This prevents the policy from collapsing to a degenerate distribution that maximizes reward but loses its general language capabilities. The paper's decision to exclude this term suggests that for the reasoning RL tasks studied, the reward signal alone (with the group-relative normalization) provides sufficient regularization, and the KL penalty may actually slow down learning by constraining exploration. This is consistent with findings in other recent RL-for-reasoning work (e.g., DeepSeek-R1) that found KL penalties could be omitted for narrow reasoning domains.

---

#### Difficulty-Based Curriculum Learning for Reasoning RL

The core insight motivating difficulty-based curriculum learning is a dynamic mismatch between model proficiency and training data difficulty. As the paper describes:

- **Later stages of training:** The model becomes capable, so easy problems result in "rollouts where all rewards are 1s"—every sampled response gets the correct answer. With all rewards equal, the group-relative advantage `$r(x, y_i) - \bar{r}(x)$` is zero for every response, providing "no useful gradient signal."

- **Early stages of training:** The model is still weak, so hard problems result in "batches where all rewards are 0s"—every sampled response is incorrect. Again, zero gradient signal.

Both scenarios waste compute because the model generates responses, evaluates them, and then learns nothing from them.

The solution is a **two-stage difficulty-based curriculum**:

**Stage 1: Moderate-difficulty problems.** The model trains on problems where pass@1 is non-trivial but far from perfect—problems that produce a mix of correct and incorrect responses. The paper uses "moderate difficulty data, samples_per_prompt=16," generating 16 responses per problem. With moderate difficulty, some responses succeed and some fail, creating the reward variance needed for effective gradient updates. The model improves from this signal, gradually increasing its accuracy on these problems.

**Stage 2: Extremely difficult problems.** Once the model has plateaued on moderate-difficulty problems, the training switches to "extremely difficulty data, samples_per_prompt=512." These are problems where "pass@8=0" (in 8 samples, none are correct) but "pass@512>0" (in 512 samples, at least one is correct). The essence is that these problems are currently too hard for the model to reliably solve, but not impossibly hard—given enough attempts, the model occasionally produces a correct answer. By generating 512 samples per problem, the model is guaranteed to see some correct trajectories alongside many incorrect ones, providing the necessary reward variance for learning. The high sample count per problem is crucial: with only 16 samples, these problems would likely produce all-zero rewards and no learning signal.

Figure 5 demonstrates the effectiveness. The baseline (red line) continues using moderate-difficulty data throughout and plateaus around 81.8% on AIME'24 Avg@32. The two-stage approach (blue line) continues improving after switching to extremely difficult problems and reaches 83.4%. The switch point is at training step 1500 in the figure. The paper explicitly states: "all problems used in the second stage are strictly sourced from a pool with verified correct answers" to "maintain high signal quality and reduce noise." This means the extremely difficult problems are not arbitrary hard questions but are filtered to ensure they have known correct solutions and that the model's occasional successes are genuinely correct, not coincidental matches.

---

#### Single-Stage 64K-Length RL vs. Multi-Stage Progressive Length RL

A conventional approach for training models to generate long reasoning traces—advocated by prior work like DeepScaler (Luo et al., 2025)—is to conduct RL in multiple stages with progressively increasing maximum output lengths (e.g., first train at 16K, then 32K, then 48K, then 64K). The intuition is that the model should learn to reason well within shorter contexts before tackling longer ones, and that gradually extending the length prevents the model from being overwhelmed.

The paper reports a finding that contradicts this conventional wisdom: "this multi-stage approach is less effective than a single-stage RL process conducted directly at the maximum target length of 64K." The explanation hinges on the relationship between SFT conditioning and RL behavior:

Since the cold-start SFT has already "conditioned the model on generating 64K-length responses" (i.e., the SFT data includes examples with up to 64K tokens of reasoning), introducing RL stages with shorter maximum lengths can "cause the model to 'unlearn' its long-context capabilities." Specifically, when the maximum output length is constrained to 16K or 32K during the early RL stages, the model's policy adapts to produce shorter responses—its average output length decreases as it learns that long reasoning traces are penalized by truncation. This causes "a significant and irreversible drop in performance" because "the model's average output length decreases" and this degradation "is difficult to recover from in the final 64K-length RL stage."

Figure 6 illustrates this. The multi-stage training (blue line) goes through stages at 16K, 32K → 48K → 64K context lengths. At the end of the 16K stage, performance has dropped (the curve dips), and even after increasing to 64K, the final accuracy (80.6%) is lower than the single-stage 64K training (red line, 83.4%). The single-stage approach "continually pushes the model's limits and yields better performance."

This finding has a clear operational implication: if the SFT model already supports long outputs, do not constrain output length during RL—train at the full target length from the start. The "irreversible" nature of the degradation suggests that the RL process can permanently shift the model's distribution toward shorter outputs, and this shift is not easily reversed by subsequent training stages with longer limits. The model learns a local optimum that prioritizes concise (but potentially incomplete) reasoning, and breaking out of this requires substantially more training than if the long-form behavior had been preserved.

---

#### Dynamic Sampling Temperature for Reasoning RL

During RL training, the sampling temperature controls the tradeoff between exploration and exploitation. The softmax distribution over tokens is computed as:

$$p(\text{token} = t) = \frac{\exp(z_t / T)}{\sum_{t'} \exp(z_{t'} / T)}$$

where `$z_t$` is the logit for token `$t$` and `$T$` is the temperature. Low `$T$` (e.g., 0.6) concentrates probability mass on the most likely tokens, reducing diversity and favoring exploitation of the current policy. High `$T$` (e.g., 1.2) flattens the distribution, increasing diversity and encouraging exploration of less-likely tokens.

The paper identifies a problem with fixed-temperature training: "as the policy distribution becomes more concentrated (i.e., has lower entropy), [a fixed temperature often results] in insufficient exploration at later stages." As the model becomes more confident (its policy entropy decreases), a fixed temperature produces increasingly deterministic outputs. The model stops exploring and simply exploits its current best strategy, which prevents further improvement.

The solution is a dynamic temperature schedule with a quality-control mechanism:

1. **Monitor reward convergence:** The system tracks the average reward of rollouts during training. When this average "stabilizes" (stops improving), the model is identified as having converged at the current temperature.

2. **Temperature exploration:** The system "periodically evaluate[s] model performance on a held-out validation set across a range of temperatures." This means taking model checkpoints, generating responses at various temperatures (e.g., 0.6, 0.8, 1.0, 1.2), and measuring accuracy on the validation set. This evaluation is separate from the main training loop and uses a fixed held-out set.

3. **Select maximum temperature with bounded performance drop:** The temperature for the next training phase is "set to the maximum value that does not cause a performance drop of more than 1% from the current optimum." This means if the current best validation accuracy is 80%, a temperature that achieves 79.2% would be acceptable (within 1%), but one that achieves 78.5% would not. The "maximum" criterion ensures the system errs on the side of more exploration.

This mechanism is credited to Polaris (An et al., 2025), cited as reference [2]. It creates a self-regulating feedback loop: when the model stops improving → temperature increases → more diverse responses are generated → new strategies may be discovered → performance improves → when it plateaus again → temperature increases again → and so on. The 1% threshold prevents the temperature from becoming so high that the model generates predominantly noise, which would waste training compute on low-quality trajectories that happen to occasionally succeed by chance.

---

#### Code RL: Token-Weighted Mean Loss

Reinforcement learning for code generation presents a specific challenge not present (or less present) in math RL: the reward is at the sequence level (passes all test cases or not), but the generated response is a sequence of many tokens, each contributing differently to the final correctness. The standard approach in GRPO is to use a "sequence-mean loss," where the advantage signal `$r(x, y_i) - \bar{r}(x)$` is applied uniformly to every token in the response.

The paper finds that this is suboptimal for code RL. Instead, it uses a **token-weighted mean loss**, which applies different weights to different tokens based on their contribution to the sequence-level reward. The paper does not provide the explicit formula, but the standard formulation of token-weighted loss in this context is:

$$L_{\text{token-weighted}} = -\frac{1}{\sum_t w_t} \sum_{t} w_t \cdot \log p_\theta(y_t | x, y_{<t}) \cdot A(x, y)$$

where `$w_t$` is a per-token weight, `$p_\theta(y_t | x, y_{<t})$` is the model's probability for token `$y_t$` given the prompt and preceding tokens, and `$A(x, y) = r(x, y) - \bar{r}(x)$` is the advantage.

**What it computes:** Rather than treating all tokens as equally informative about the reward, the token-weighted approach assigns higher weight to tokens that are more predictive of the outcome. In practice, the weights are often derived from the model's own uncertainty or from an auxiliary value head. The result is that the gradient update is concentrated on the parts of the response that actually determine success or failure—for code, this might be the specific lines implementing the algorithm rather than boilerplate imports or comments.

**Why this form:** The paper reports two specific benefits. First, "token-weighted approach provides a finer-grained and more stable gradient signal, which leads to significantly faster convergence." This is because irrelevant tokens (which constitute noise in the gradient estimate) receive lower weight, reducing variance. Second, it "helps to alleviate the length bias inherent in sequence-level rewards and effectively suppresses the generation of overly simplistic or repetitive 'base case' samples during training." Length bias refers to the tendency of sequence-mean loss to implicitly favor longer responses (because each additional correct token adds to the total log-probability). The token-weighted approach can counteract this by down-weighting padding or repetitive tokens.

Figure 7 (left) shows the comparison: with token-weighted mean loss, LiveCodeBench accuracy during RL training reaches 46.5% at the plateau, while sequence-mean loss reaches 46.3% but converges more slowly. The token-weighted curve rises more steeply in the first 500 steps, indicating the accelerated learning.

---

#### Science RL: Data Quality Over Data Quantity

For scientific reasoning (as measured on GPQA-Diamond), the paper reports that "data quality and type are paramount factors." Through ablation experiments, it finds that "using exclusively expert-verified multiple-choice questions for RL leads to significantly better performance compared to training with mixed-quality or unverified data."

Figure 7 (right) quantifies this. Training on expert-verified multiple-choice data (blue line) reaches 65.8% on GPQA, while training on mixed-quality science data (red line) reaches only 62.9%. The performance gap is roughly 3 percentage points, and the expert-verified data curve continues to improve over the training steps shown, while the mixed-quality curve appears to plateau earlier.

The implication is that for scientific domains, the "reward is 1 if answer matches ground truth" paradigm is only reliable if the ground truth is itself reliable. Mixed-quality data may contain questions that are poorly specified, have ambiguous answers, or test trivia rather than reasoning—even if the model gets the "correct" answer, the reward signal may not reflect genuine scientific understanding. Expert-verified multiple-choice questions (presumably drawn from graduate-level exams or curated by domain experts) provide a cleaner signal because the questions are designed to test reasoning, the answer choices are carefully constructed, and the ground truth is unambiguous.

This finding challenges the "more data is better" assumption that often guides RL data collection. For science RL, a smaller set of high-quality questions (Figure 7 shows the expert-verified data training for ~300 steps) outperforms a larger set of noisy questions. The paper does not disclose the exact size of the expert-verified dataset.

---

#### Agentic Reinforcement Learning

Agentic RL targets tasks where the model interacts with external environments—web browsers, code executors, tool APIs—to accomplish goals. Unlike reasoning RL, where the model generates a single self-contained response and receives a reward, agentic RL involves multi-turn interactions where the model's actions affect the environment state, and the reward is based on the trajectory's final outcome.

**Data collection and synthesis for agents.** The paper describes two data pipelines:

**Web-search tasks:** The goal is to create "demanding question–answer pairs requiring multi-step reasoning across multiple web sources" designed to "sharpen GLM's ability to uncover elusive, interwoven facts on the internet." The construction blends two approaches: (1) "an automated pipeline powered by multi-hop reasoning over knowledge graphs"—a knowledge graph encodes relationships between entities, and multi-hop queries (e.g., "find the author of the book that won the same award as the film directed by X") require traversing multiple edges to answer; these are automatically converted into natural language questions; (2) "human-in-the-loop extraction and selective obfuscation of content from several web pages"—humans identify interesting fact combinations from web content and deliberately obscure some of the facts, creating questions that require synthesizing information from multiple sources rather than finding a single page that contains the answer.

**Software-engineering tasks:** The pipeline curates "an extensive collection of GitHub pull requests and issues to create a realistic software-development benchmark comprising user prompts and executable unit tests." Each training instance consists of a repository state (the codebase before the fix), a natural language description of the issue to resolve, and a set of unit tests that verify whether the fix is correct. The key requirement is that "all evaluations run inside a hardened sandbox with a distributed system, which provides both horizontal scalability and strong isolation guarantees." Horizontal scalability means multiple training instances can run in parallel across many machines; strong isolation means that a buggy model output that deletes files or runs infinite loops cannot affect other training instances. This sandbox infrastructure is essential for safe, scalable agentic RL training.

---

#### Agentic RL Objective and Outcome Supervision

The agentic RL uses the same group-wise policy optimization framework as reasoning RL, with a crucial difference in how rewards are computed:

$$L_{\text{RL}}(\theta) = \mathbb{E}_{x \sim D} \left[ \frac{1}{K} \sum_{i=1}^{K} \left( r(x, y_i) - \bar{r}(x) \right) \right]$$

where `$y_i$` now represents an entire agent trajectory—a sequence of actions (function calls, tool invocations), each followed by environment observations (tool outputs, intermediate results), culminating in a final answer or task completion state. As with reasoning RL, "only model-generated tokens are used for optimization, and the environment feedback is ignored in loss computation." This means the model learns from which of its own generated actions lead to success, but the tokens representing what the environment returned are not part of the trainable sequence—they are treated as fixed context.

**Outcome supervision with format penalty.** For web search tasks, the reward is based on "the accuracy of the final answer." If the model's final answer matches the ground truth, the entire trajectory receives reward 1; otherwise, reward 0. For coding agents, the reward is based on "SWE data with verifiable test cases"—if the model's code patch passes all test cases, reward 1; otherwise, reward 0.

Additionally, "a process format penalty" is applied: "if the model fails to produce the correct tool format during agent trace generation, the process will be halted, and the trace will receive a zero reward." This penalty is applied during trajectory generation, not during loss computation. If the model generates a malformed function call (e.g., missing tags, unclosed XML elements), the environment cannot execute it, so the trajectory is terminated and assigned zero reward. This serves two purposes: it prevents the model from wasting compute on trajectories that are structurally invalid (speeding up training), and it provides a strong negative signal that reinforces proper formatting.

The distinction between "outcome supervision" and "process supervision" is important here. Outcome supervision only cares about the final result—whether the task was completed successfully. Process supervision would provide intermediate rewards for correct sub-actions (e.g., a correctly formulated search query, even if the final answer is wrong). The paper uses pure outcome supervision, which is simpler to implement (no need to define sub-goals or intermediate rewards) but can make credit assignment harder for long trajectories. The format penalty is a minimal form of process signal—it catches structural errors but does not guide the model toward better strategies beyond "stay in the correct format."

---

#### Iterative Self-Distillation for Agentic RL

Agentic RL training is described as "time-consuming" because each trajectory requires interacting with external environments (waiting for web pages to load, executing code in sandboxes, etc.). To maximize the efficiency of this expensive training, the paper employs an iterative self-distillation loop:

1. **Initial RL training:** Start with the cold-start SFT model and run RL training on agentic tasks using the outcome-supervision-with-format-penalty scheme. Train until "a certain step count or plateaued"—when the model's agentic performance stops improving.

2. **Self-distillation:** Generate new SFT data by using the RL-trained model to produce agentic trajectories on the training tasks. Apply the same quality filtering as in the original SFT stage (correctness verification, format checking). Replace "the original cold-start data with responses generated by the RL-trained model, thus creating a superior SFT model."

3. **Resume RL on the improved model:** Take the self-distilled SFT model and run another round of RL training, "progressively increasing training difficulty" by introducing harder tasks or more complex environment interactions.

4. **Repeat:** This loop can be iterated multiple times, with each cycle producing a stronger model that generates better training data for the next cycle.

**Why this works:** The RL process discovers strategies (specific reasoning patterns, tool-use sequences, debugging approaches) that succeed on the training tasks but may not be fully internalized by the policy—the model might succeed sometimes but not consistently. By generating many trajectories from the improved policy and training on the successful ones via SFT, these strategies are "baked into" the model's weights through direct supervised learning. This SFT model then provides a stronger starting point for the next round of RL, which can discover even better strategies. The process is analogous to the "STaR" (Self-Taught Reasoner) approach but extended to the agentic domain.

The paper notes that this strategy "allows us to push the performance limits of RL-trained models efficiently." The efficiency comes from the fact that SFT on generated trajectories is much faster than RL (no environment interaction needed, just forward and backward passes on pre-computed data), so the expensive RL training is interleaved with cheap SFT training that consolidates the gains.

---

#### Scaling Test-Time Compute Through Interaction Turns

A key observation about agentic tasks is that performance scales with test-time compute, but through a different mechanism than in reasoning tasks. For reasoning, test-time scaling means generating longer chain-of-thought traces (more output tokens). For agentic tasks, test-time scaling means "continuously interacting with the environment, e.g., searching high and low for hard-to-find web information or writing test cases for self-verification and self-correction for coding tasks."

Figure 8 demonstrates this scaling behavior on BrowseComp. As the interaction budget increases from 8 to 128 turns (x-axis, log scale), accuracy (y-axis) increases roughly linearly in log-space: from approximately 5% at 8 turns to approximately 26% at 128 turns. This is a ~5× improvement purely from allowing the model more rounds of interaction with the search environment, without any model weight changes.

This has implications for deployment: when agentic applications can tolerate higher latency, allowing the model additional interaction turns is a reliable way to improve accuracy. The paper does not claim a specific scaling law (e.g., accuracy ∝ log(turns)), but Figure 8 suggests a log-linear relationship. The mechanism is intuitive: more turns mean more searches attempted, more pages examined, more candidate answers evaluated, and more opportunities for the model to notice inconsistencies and self-correct.

---

#### General Reinforcement Learning

General RL aims to holistically improve the model's capabilities across seven primary categories and 139 tertiary categories of prompts, using a multi-source feedback system that combines rule-based, human, and AI feedback. The key design insight is that each feedback source has complementary strengths and weaknesses:

- **Rule-based feedback:** Precise and deterministic, but only applicable to tasks with objective criteria (correct format, exact answer match).
- **Human feedback:** Provides nuanced judgment on subjective dimensions (safety, helpfulness, tone), but is expensive and slow to scale.
- **AI feedback (RLAIF):** Scalable and fast, but can be unreliable or biased—an AI judge may share the same blind spots as the model being trained.

By combining all three sources, the paper aims to achieve "more robust training signals" that "mitigat[e] the inherent limitations of each individual method."

**Holistic RL.** The training dataset is "a balanced dataset of roughly 5,000 prompts spanning 7 primary, 33 secondary, and 139 tertiary categories." This hierarchical taxonomy ensures coverage across diverse domains. For human feedback, the paper trains "a reward model on preference annotations" where "annotators compare model responses and assign preference labels based on a comprehensive evaluation of multiple dimensions, such as instruction following, safety, and factual correctness." This reward model then scores model outputs during RL training, approximating what a human annotator would prefer.

For model feedback (RLAIF), the paper designs "separate scoring rubrics that depend on whether the prompt has an objective ground-truth answer." For prompts with objective answers, the AI judge checks correctness against the ground truth. For subjective prompts, the AI judge evaluates along rubric-defined dimensions (coherence, relevance, creativity, safety, etc.). The two feedback sources (human reward model and AI judge) are then merged to produce the final reward signal. The paper does not specify the exact merging mechanism (e.g., weighted average, minimum, or learned combination).

**Instruction Following RL.** This component specifically targets the model's ability to "understand and satisfy complex instructions." A fine-grained taxonomy with "7 major and 151 minor constraint types" is developed, covering constraints like:

- Content requirements: "Include at least three examples," "Respond in the style of Shakespeare"
- Formatting rules: "Output as a JSON object," "Use bullet points," "Include a code block"
- Behavioral constraints: "Do not mention that you are an AI," "Respond in fewer than 100 words"

A "dedicated training set of challenging instructions is assembled to cover every constraint type." The feedback system for this task has three components: deterministic verification rules (checking format and content constraints programmatically), a trained reward model (for softer constraints that can't be verified deterministically), and a critique model (an LLM that evaluates whether the response satisfies the instruction and provides a score with justification).

Figure 9 demonstrates the effectiveness. During GRPO training for instruction following (without other general RL tasks mixed in), the SysBench-ISR score (a measure of system message following) increases from 64.8 at step 0 to 77.2 at step 1000, improving monotonically with the reward signal. The paper specifically notes: "Up to roughly 1,000 training steps, we have not observed clear evidence of reward hacking." Reward hacking would manifest as the reward increasing while the SysBench score stagnates or decreases—the model finding ways to get high rewards without actually following instructions better. The fact that both curves rise in parallel indicates that the reward is well-aligned with the target behavior.

---

#### Function Calling RL

Function calling is a capability that allows the model to invoke external tools (APIs, databases, code executors) by generating structured function calls. The paper treats it through two complementary RL approaches.

**Step-wise rule-based RL.** For tasks "with clear tool invocation procedures," the system provides "ground truth function call for each step/turn in the training data." The model is trained to generate the next assistant response (function call or text) given the task history and previous function calls. The reward is a strict binary:

$$\text{Reward} = \begin{cases} 1, & \text{if FormatCorrect}(a_t) \text{ and Match}(a_t, a^*_t) \\ 0, & \text{otherwise} \end{cases}$$

where `$a_t$` is the `$t$`-th function call generated by the model, and `$a^*_t$` is the ground truth function call for that step.

**What it computes:** A reward of 1 is given only when the generated function call is syntactically valid AND exactly matches the ground truth in "the name, parameters, and every field." Any deviation—a parameter with the wrong value, a missing field, an extra field—results in 0 reward. This strictness is deliberate: "such a strict reward rule not only guides the model to generate correct function calls but also strongly enforces output formatting, improving the model's usability and robustness in real-world interactions."

**Why this form:** In production agentic systems, a malformed function call causes an execution error that requires error-handling logic, retries, or human intervention. A model that generates mostly-correct function calls but occasionally misses a required parameter is less useful than one that is slightly less creative but always produces valid calls. The strict binary reward reflects this real-world requirement: partial correctness is not sufficient; the call must be exactly right.

This approach is integrated directly into the general RL framework because step-wise function calling tasks have "similar output lengths and convergence speeds" to other general RL tasks, allowing them to share training infrastructure.

**End-to-end multi-turn RL.** Step-wise RL has a fundamental limitation: it "decomposes tasks into static, predetermined decision flows" where the correct action at each step is known in advance. In real scenarios, the correct sequence of actions may depend on what the environment returns, and the model must "autonomously explore, plan, [and] handle complex situations."

End-to-end multi-turn RL addresses this by making the model "generate the complete trajectory" autonomously and receiving a reward only at the end based on whether the task was completed. The reward formula is:

$$\text{Reward} = \begin{cases} 1, & \text{if FormatCorrect}(a_1, \ldots, a_T) \text{ and TaskCompleted}(I, o_0, a_1, o_1, \ldots, a_T, o_T) \\ 0, & \text{otherwise} \end{cases}$$

where `$I$` is the original complex task description, `$a_t$` is the `$t$`-th function call, and `$o_t$` is the environment feedback (tool output or user response) following call `$a_t$`. `$T$` is the total number of calls in the trajectory.

**What it computes:** The entire sequence of actions must be correctly formatted and must result in task completion as determined by "the environment according to predefined rules or by an LLM Judge Agent." There are no intermediate rewards—the model must discover, through trial and error, which action sequences lead to success.

**Why this form:** This approach treats the multi-turn interaction as a sequential decision-making problem where the model's policy must adapt to environment feedback. The model learns not just what to call but when to call it, how to interpret tool outputs, how to recover from errors, and how to decide when the task is complete. The two task types considered are:

1. **Single-turn multi-step tasks:** The model makes multiple function calls in a single turn, interacting with the environment between calls. These use "complex tasks automatically synthesized based on MCP servers, as well as some open-source agentic datasets with runnable environments, such as Agentgym" (Xi et al., 2024).

2. **Multi-turn multi-step tasks:** Beyond interacting with tools, the model also interacts with "an LLM-simulated user agent to obtain complete task information and accomplish the overall task." The simulated user provides clarifying information, responds to the agent's questions, and confirms completion—simulating the back-and-forth of a real human-AI interaction.

The end-to-end approach is not integrated into the general RL framework but is trained separately as an expert model (Section 3.3, paragraph 2), with the resulting expertise later distilled into the unified model.

---

#### Pathology Reinforcement Learning

As the final stage of post-training, pathology RL targets specific undesirable behaviors that occur at low frequency in the model's outputs. The paper identifies "language mixing, excessive repetition, and formatting mistakes" as key pathologies. These are problematic because:

- **Low incidence rate** ("often less than 1% of outputs"): Standard RL training on diverse prompts would only encounter these behaviors rarely, making the optimization "sample-inefficient"—most training batches contain no examples of the pathology, so updates addressing it are infrequent.

- **Outsized user impact:** Even if rare, a model that suddenly switches languages mid-response or gets stuck in a repetition loop creates a strongly negative user experience that undermines trust.

The solution is constructing "a targeted dataset for pathology RL by identifying prompts that are highly likely to trigger these pathological behaviors." For example, a prompt that mixes multiple languages might trigger language mixing; a prompt asking for a very long response might trigger repetition. By concentrating training on these trigger prompts, the model receives dense negative signal for the pathological behaviors, allowing efficient optimization.

The training on this dataset "impose[s] efficient penalties, further lowering the residual error rates for these problematic behaviors." The penalty is presumably applied through the same GRPO framework—responses exhibiting the pathology receive low or zero reward, responses without the pathology receive positive reward if they otherwise satisfy the prompt.

---

#### The Slime RL Infrastructure

Conducting RL training at the scale of GLM-4.5—especially for agentic tasks with long, environment-dependent rollouts—requires purpose-built infrastructure. The paper describes Slime, an open-source RL framework developed for this purpose.

**Flexible hybrid training and data generation architecture.** Slime supports two operational modes within a single system:

1. **Colocated, synchronous mode:** Training and inference engines reside on the same GPU workers. When a training step completes, the updated model weights are immediately available for inference (rollout). When rollouts complete, the generated data is immediately available for training. This is complemented by "dynamic sampling, [which] significantly reduces GPU idle time and maximizes resource utilization." Dynamic sampling adjusts the number of rollouts generated per training step based on current throughput, ensuring neither the training nor inference side is waiting on the other.

This mode is used for "general-purpose RL tasks or those aimed at enhancing model reasoning capabilities (e.g., in mathematics and code generation)," where rollouts are relatively fast (the model generates a solution in seconds, and the reward can be computed immediately from ground-truth answers).

2. **Disaggregated, asynchronous mode:** Training and inference engines are on separate GPU pools (or separate machines) and operate independently. The rollout component "is exposed directly to the agent environment," while "GPUs for training and inference are scheduled independently." This decoupling means that agent environments can "constantly generate new data without being stalled by the training cycle"—a web-search trajectory that takes 5 minutes doesn't block the training GPU, which can continue processing previously generated data.

This mode is used for "agentic tasks, such as those in Software Engineering (SWE), [where] the data generation process is often protracted and involves complex system interactions." SWE tasks require setting up a repository, running tests, installing dependencies—operations that can take minutes to hours per trajectory. Without disaggregation, training GPUs would sit idle while waiting for rollouts.

The Ray framework provides the underlying resource scheduling and asynchronous capabilities, allowing "flexibly place[ment of] the inference and training engines on the same GPU or on different ones."

**Accelerated rollout with mixed-precision inference.** Rollout speed—the rate at which the model can generate new training data—is identified as "a persistent bottleneck in RL training." Slime addresses this with mixed-precision computation:

- Training is done in BF16 (Brain Floating Point 16), which provides sufficient numerical precision for stable weight updates.
- Inference (rollout) is done in FP8 (8-bit floating point), which halves memory bandwidth and compute requirements per token compared to BF16.

The transition between precisions is handled by "online, block-wise FP8 quantization on the model parameters before they are dispatched for rollout." During each policy update iteration, the current BF16 model weights are quantized to FP8 in blocks (groups of parameters quantized together to maintain statistical properties). This dynamic quantization means the FP8 inference always uses the latest model weights, avoiding the staleness that would occur with a pre-quantized static copy.

The paper does not report quantitative throughput improvements from FP8 inference, but the halving of memory bandwidth is particularly significant for large-batch inference where the KV cache is the memory bottleneck.

**Agent-oriented RL infrastructure design.** The paper describes two specific innovations for agentic RL:

**High-concurrency Docker-based runtime:** Agentic rollouts require isolated environments—a web browsing agent needs a fresh browser session, a coding agent needs a clean repository state. Setting up and tearing down these environments serially would be extremely slow. The Slime infrastructure "provision[s] isolated environments for each task" using Docker containers, with high concurrency so that many rollouts can execute simultaneously in separate containers.

**Unified HTTP endpoint with centralized data pool:** The diversity of agent frameworks (each with its own APIs, data formats, and execution semantics) poses an integration challenge. Slime addresses this with a two-part abstraction:

- All agent frameworks are wrapped behind "a unified HTTP endpoint interface." Regardless of the underlying framework, the RL training system interacts with it through a standardized HTTP API that accepts a task specification and returns a trajectory (message list).

- All generated trajectories—regardless of which framework produced them—are stored in "a centralized data pool" in a "message-list format" (the common representation of conversational and agentic interactions). This pool "serves as a shared source for training" and "supports customizable, task-specific filtering and dynamic sampling strategies to ensure high-quality RL training data across diverse tasks."

This architecture "cleanly decouples task-specific rollout logic from the RL training process, enabling seamless integration of heterogeneous agent frameworks." A new agent framework can be added by implementing the HTTP endpoint; no changes to the training pipeline are needed. Similarly, the training pipeline can sample data from the pool with task-specific strategies (e.g., oversampling trajectories from more difficult tasks) without knowing which framework generated each trajectory.

**Fully asynchronous RL training loop.** Because "agent tasks can vary in type and trajectory length," synchronous training—where each step waits for all rollouts to complete—would cause "severe GPU underutilization as workers wait for the slowest rollouts." Slime's solution:

- GPUs are partitioned into "dedicated rollout engines and training engines." Rollout engines continuously generate trajectories and push them to the data pool. Training engines continuously sample from the pool, update model weights, and "periodically synchronize them back to the rollout engines"—the rollout engines periodically receive updated model weights so that they are generating data from a recent (but not necessarily the very latest) policy.

- This decoupling "prevents long or diverse trajectories from blocking the entire training pipeline, resulting in consistently high throughput, particularly in scenarios with highly variable agent interactions." The model training is slightly stale (using data from a policy a few updates behind), but this staleness is a small price for continuous utilization of expensive GPU resources.

The paper emphasizes that through these two core designs—high-concurrency isolated runtimes and fully asynchronous training—Slime provides "a scalable, flexible, and high-performance solution for long-agentic RL, and can support long-horizon rollouts and adapt to a wide range of agent tasks."

#### Summary of Design Choices and Their Justifications

- **Deep-but-narrow MoE architecture over wide-but-shallow:** Depth improves reasoning capacity more than width per parameter, as evidenced by the attention head count finding (more heads improve reasoning benchmarks without improving training loss).

- **Loss-free balance routing over auxiliary-loss routing:** Avoids interference between the load-balancing objective and the primary language modeling loss, using learned biases that are updated separately from the main gradient flow.

- **Multi-source pre-training data with quality-aware up-sampling over uniform sampling:** High-quality documents (educational multilingual text, verified code, expert-scored math/science) receive more exposure while low-quality data is discarded, concentrating training compute on informative examples.

- **Mid-training as a separate phase over mixing all data in pre-training:** Allows extending context length (4K → 32K → 128K) and introducing instruction-like data (synthetic reasoning, agent trajectories) after the model has established broad language competence, avoiding the risk of domain-specialized data distorting general capabilities.

- **Expert Model Iteration over monolithic multi-objective RL:** Prevents capability interference—the reasoning expert can maximize math accuracy without degrading chat quality, because general capabilities are maintained separately and reintegrated through distillation.

- **Two-stage difficulty curriculum over static difficulty:** Ensures the model always trains on problems that provide reward variance (mix of successes and failures), avoiding the zero-gradient scenarios that waste compute on problems that are too easy or too hard.

- **Single-stage 64K-length RL over multi-stage progressive length:** Preserves the SFT-conditioned long-output behavior; constraining output length during early RL stages causes irreversible degradation as the model learns to produce shorter responses.

- **Dynamic sampling temperature with bounded quality degradation over fixed temperature:** Maintains exploration as the policy entropy decreases, while preventing excessive noise through a validation-set-based quality threshold.

- **Token-weighted mean loss for code RL over sequence-mean loss:** Reduces gradient variance by down-weighting irrelevant tokens, accelerates convergence, and suppresses length bias that favors overly simple or repetitive solutions.

- **Expert-verified data for science RL over mixed-quality data:** Clean, unambiguous reward signals from carefully curated questions outperform larger but noisier datasets, challenging the "more data is better" assumption.

- **Outcome supervision with format penalty for agentic RL over dense process rewards:** Simpler to implement (no need to define sub-goal rewards) while the format penalty catches structurally invalid trajectories early, saving compute and reinforcing correct formatting.

- **Iterative self-distillation over single-pass RL for agents:** The expensive RL phase discovers strategies; the cheap SFT phase consolidates them into the model weights, creating a stronger starting point for the next RL round and progressively pushing performance limits.

- **XML-like function call template over JSON:** Eliminates the need for character escaping in code-heavy parameters, reducing the cognitive load on the model and improving the reliability of function call generation without compromising functionality.

- **Disaggregated asynchronous RL infrastructure over synchronous colocated training for agents:** Decouples long, variable-duration agent rollouts from GPU training, maintaining high utilization even when some trajectories take orders of magnitude longer than others.

## 4. Key Insights and Innovations

### Innovation 1: Expert Model Iteration as a Solution to Capability Interference in Multi-Objective Post-Training

The dominant paradigm in post-training LLMs for diverse capabilities has been sequential or additive: apply instruction tuning, then RLHF for alignment, then possibly reasoning-focused RL, each stage operating on the same model and hoping that gains in one objective don't undo progress on others. This has been a persistent, well-documented problem—"alignment tax" where safety tuning degrades reasoning, or reasoning RL that makes the model worse at following simple instructions. The field's standard responses have been mitigating the interference (mixing data from multiple objectives, KL regularization to prevent drift) rather than eliminating the root cause.

GLM-4.5's Expert Model Iteration represents a fundamentally different mental model. Instead of trying to make a single model good at everything simultaneously, it embraces **specialization-then-distillation**: train three separate expert models—each pushed aggressively in its domain through RL without worrying about degrading other capabilities—then unify them through supervised fine-tuning on their combined outputs. This inverts the standard approach. Rather than constraining each training stage to preserve prior capabilities (which inevitably limits how much any single capability can improve), the expert model paradigm removes the constraint entirely during RL and handles unification as a separate, offline distillation step.

What makes this conceptually distinctive is that it treats the multi-objective optimization problem in post-training not as a joint optimization over a single policy, but as a **mixture-of-experts at the training level**—the specialization happens in the training procedure itself, not just in the model architecture. The MoE architecture already provides computational specialization at inference time; the Expert Model Iteration extends this principle to training. The reasoning expert learns to do math and code without ever being penalized for forgetting how to chat; the agent expert learns to navigate complex tool-use environments without being constrained by safety alignment objectives; the general expert focuses on instruction following and helpfulness without needing to compete with math benchmarks. The final distillation step then acts as a "soft merging" of these specialized policies into a single model that can route between behaviors implicitly at inference time.

This is a fundamental advance, not an incremental refinement, because it changes the boundary between what must be optimized jointly and what can be optimized separately and later combined. The evidence for its effectiveness is ultimately the model's strong performance across all three ARC domains simultaneously—ranked 3rd overall, 2nd on agentic benchmarks, 3rd on coding—which would be very unlikely if capability interference were significant. However, the paper does not provide an explicit ablation showing what would happen if the same RL objectives were applied sequentially to a single model, so the contribution is partially architectural-conceptual rather than experimentally proven within the report. The iteration and distillation loop design—train expert → distill → train better expert → re-distill—also positions this as a general framework for capability bootstrapping that extends beyond this specific model release.

---

### Innovation 2: The Diagnostic Finding That Deep-but-Narrow Architectures Benefit Reasoning Independently of Training Loss

A pervasive assumption in transformer scaling—both dense and MoE—has been that model width (hidden dimension, FFN intermediate size) and depth (number of layers) should be scaled in rough proportion, or that width is the primary driver of capacity. This assumption traces back to the original transformer paper and was reinforced by scaling laws work that treated parameters as an aggregate quantity without distinguishing architectural allocation. DeepSeek-V3 and Kimi K2, the two most prominent open-source MoE models at the time, both used hidden dimensions of 7168—substantially wider than GLM-4.5's 5120—and fewer layers (58–60 MoE layers vs. GLM-4.5's 89).

GLM-4.5's architecture makes a deliberate, counter-trend choice: reduce width and increase depth. The paper reports this was based on the empirical finding that "deeper models exhibited better reasoning capacity." But the more striking diagnostic result is the one about attention heads: using 96 query heads (2.5× more than what a standard configuration would use for a 5120 hidden dimension) "does not improve training loss compared to models with fewer heads [but] consistently improves performance on reasoning benchmarks such as MMLU and BBH."

This is a **diagnostic finding about inductive bias**, not just an architectural optimization. It reveals that training loss—the standard metric for comparing architectures during development—is an insufficient signal for downstream reasoning performance. The model with fewer attention heads trains just as well (same loss) but generalizes worse on reasoning tasks. This implies that the attention head configuration imparts an inductive bias that helps the model learn representations more amenable to logical reasoning, even though this bias doesn't manifest as lower perplexity on the pre-training distribution. The paper doesn't characterize what this bias is (e.g., whether more heads enable finer-grained attention patterns that support multi-hop reasoning, or whether smaller per-head dimension forces more structured decomposition of attention), but the finding itself is a methodological contribution: when optimizing architecture for reasoning, don't just look at training loss.

The significance extends beyond architecture search. It challenges the implicit assumption in the scaling laws literature that a parameter is a parameter—that how you arrange compute matters less than how much compute you have. GLM-4.5's position on the SWE-bench vs. parameters Pareto frontier (Figure 2) is a direct consequence: by allocating parameters to depth rather than width, and by over-allocating to attention heads (which are a small fraction of total parameters but disproportionately affect reasoning), the model achieves performance that would typically require many more parameters in a width-heavy configuration.

This is a fundamental finding about the relationship between architecture and capability, not an incremental optimization. It provides a concrete, actionable principle (prefer depth over width for reasoning, and don't trust training loss as the sole architecture selection criterion) that other model developers can apply. The evidence is clear from Table 1 and the stated empirical findings, though the paper does not provide a controlled ablation with identical parameter counts and different depth/width ratios—a limitation that makes this a strong diagnostic observation rather than a fully proven causal claim.

---

### Innovation 3: The Identification of "Irreversible Unlearning" When Multi-Stage Length-Constrained RL Follows Long-Context SFT

Prior work on training models for long reasoning traces—notably DeepScaler (Luo et al., 2025)—advocated a multi-stage RL approach where the maximum output length is progressively increased: train at 16K first, then 32K, then 48K, then 64K. The intuition was clear and widely accepted: curriculum learning works; don't overwhelm the model with extremely long outputs before it has mastered shorter ones. This became a recommended practice in the RL-for-reasoning community.

GLM-4.5's finding that this multi-stage approach is strictly worse than single-stage 64K-length RL directly contradicts this emerging consensus. The mechanism is described in Section 3 but the conceptual implication is what matters here: **once a model has been conditioned to produce long outputs through SFT, constraining output length during RL causes the policy to adapt in ways that are difficult or impossible to reverse.** The model learns to produce shorter responses to maximize reward within the length limit, and this "unlearning" of long-form behavior persists even when the constraint is later removed. Performance drops and does not fully recover.

This is a **negative result with significant methodological implications**. It establishes a boundary condition for curriculum learning in RL: when the initial policy already supports a behavior (like generating 64K-token reasoning traces), do not train with a constraint that suppresses that behavior, even temporarily. The irreversibility is the key diagnostic concept—it suggests that RL optimization landscapes for language models have deep local minima where the policy can get stuck. Once the model learns that shorter responses are "good enough" (they achieve high reward within the length limit), it requires disproportionate optimization pressure to push it back toward longer responses that might achieve even higher reward.

The practical upshot for the field is a clear prescription: if your SFT model can already handle long outputs, do RL at the full target length from the start. The finding also raises deeper questions about when and why curriculum learning helps versus hurts in RL for language models. The standard curriculum learning intuition—start easy, gradually increase difficulty—assumes that skills learned on easy tasks transfer positively to harder ones. GLM-4.5's evidence suggests that for output length, the "easy" task (shorter responses) teaches a qualitatively different behavior (concise reasoning) that competes with rather than scaffolds the target behavior (thorough reasoning). This is a warning about assuming positive transfer in curriculum design for generative models.

The evidence from Figure 6 is clear and directly supports the claim: the single-stage approach (red line) reaches 83.4% while multi-stage (blue line) plateaus at 80.6%, with the multi-stage curve showing a visible dip during the 16K stage that it never fully recovers from. The finding is technically about a training recipe choice, but its conceptual significance—identifying irreversible policy shifts under seemingly reasonable curriculum strategies—makes it more than an incremental optimization.

---

### Innovation 4: The Reframing of Agentic Capability Development Through Environment-Agnostic Abstraction and Asynchronous Infrastructure

Agentic RL at scale has been a relatively under-explored area compared to math and code RL. The challenges are qualitatively different: trajectories involve interacting with external environments (web browsers, code sandboxes, tool APIs), these interactions are slow and variable in duration, and different agent frameworks have incompatible APIs and data formats. Most prior work either operated at small scale with carefully controlled environments, or trained agents in simulation and hoped for transfer to real tools.

GLM-4.5's infrastructure contribution—the Slime framework with its disaggregated asynchronous architecture—is not just an engineering achievement but a **conceptual reframing of how to approach agentic RL training**. The key insight is the complete decoupling of three concerns that were previously entangled: (1) the training computation (gradient updates on model weights), (2) the rollout computation (generating agent trajectories), and (3) the environment execution (running tools, browsers, sandboxes). By making these asynchronous and independently scalable, Slime treats agentic RL not as a monolithic training loop but as a **distributed data production-and-consumption system**.

The centralized data pool with a unified message-list format is the other half of the reframing. Rather than fighting the diversity of agent frameworks (each with different APIs, state representations, and execution models), Slime accepts this diversity as inevitable and provides a thin, standardized abstraction layer—the HTTP endpoint interface—that makes all frameworks look the same to the training system. This is philosophically similar to how the web standardized on HTTP to enable diverse backend services to interoperate: don't force everyone into the same framework; just agree on a common protocol. The data pool then serves as a buffer that insulates the training pipeline from the variability of agent rollouts, allowing sampling strategies (difficulty-based, task-specific filtering) to be applied at the data level rather than the environment level.

This reframing is significant because it lowers the barrier to training on diverse agentic tasks. A new agent framework—say, for interacting with a specific enterprise API—can be integrated by implementing the HTTP endpoint and contributing trajectories to the data pool, without modifying the RL training code. The training pipeline can then learn from these new trajectories intermixed with trajectories from completely different environments, potentially enabling cross-task generalization that would be impractical if each framework required its own training loop.

The evidence for this innovation's effectiveness is indirect but compelling: GLM-4.5 achieves competitive agentic performance (2nd overall) despite having substantially fewer parameters than competitors, and the iterative self-distillation loop described in Section 3.3.2 depends on the infrastructure's ability to continuously generate high-quality agentic training data at scale. The paper doesn't provide a controlled experiment showing that a synchronous architecture would fail, but the design rationale—preventing long trajectories from blocking training, maintaining GPU utilization despite variable rollout durations—addresses well-understood bottlenecks that any agentic RL system at this scale would encounter. The open-sourcing of Slime (referenced in Section 3.5) further positions this as a community contribution rather than a proprietary advantage: the infrastructure design itself is offered as a reusable pattern for other groups building agentic RL systems.

This is a fundamental infrastructure innovation with conceptual implications, not just an incremental engineering improvement. It defines a new reference architecture for agentic RL training that separates concerns in a way that enables scale, flexibility, and maintainability, analogous to how parameter server architectures and later disaggregated training transformed distributed neural network training. The contribution is to the *practice* of agentic RL—making it feasible at the scale of 355B-parameter models with diverse real-world tool environments—rather than to the theory, but in a field where infrastructure often determines what's possible to study, this is a significant form of intellectual contribution.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on 12 benchmarks spanning Agentic, Reasoning, and Coding (ARC) tasks, plus additional benchmarks for general chat, safety, translation, and custom manual evaluations. For agentic tasks: TAU-Bench (retail and airline domains; Yao et al., 2024), BFCL V3 (Berkeley Function Calling Leaderboard; Patil et al., 2025), and BrowseComp (Wei et al., 2025). For reasoning: MMLU-Pro (Wang et al., 2024), AIME 24, MATH-500 (Hendrycks et al., 2021), SciCode (Tian et al., 2024), GPQA (Rein et al., 2024), Humanity's Last Exam or HLE (Phan et al., 2025), and LiveCodeBench 2407-2501 or LCB (Jain et al., 2025). For coding: SWE-bench Verified (Jimenez et al., 2023) and Terminal-Bench (Terminal-Bench Team, 2025). For general chat: MMLU (Hendrycks et al., 2021), SimpleQA (Wei et al., 2024), IFEval (Zhou et al., 2023), SysBench (Qin et al., 2024), and MultiChallenge (Deshpande et al., 2025). Safety: SafetyBench (Zhang et al., 2023). Base model evaluation additionally includes BBH, HellaSwag, PIQA, TriviaQA, GSM8K, MATH, EvalPlus, CLUEWSC, C-Eval, C3, and Chinese-SimpleQA. The paper also introduces two custom evaluation sets: CC-Bench (52 programming tasks for agentic coding) and a novel logical reasoning set with 100 translation cases.

- **Base model(s).** GLM-4.5 is a Mixture-of-Experts model with 355B total parameters and 32B activated parameters. GLM-4.5-Air is a compact variant with 106B total parameters and 12B activated parameters. Both are evaluated against a range of proprietary and open-source models including o3, o4-mini, GPT-4.1, Claude Opus 4, Claude Sonnet 4, Gemini 2.5 Pro, Grok 4, DeepSeek-R1-0528, DeepSeek-V3-0324, Kimi K2, Qwen3-235B-A22B-Thinking-2507, Qwen3-Coder, MiniMax-M1, Llama4-Maverick 400B, and specialized translation models (Qwen-MT-plus, Qwen-MT-turbo, Seed-X). For base model evaluation, comparisons are made against Qwen3-235B-A22B Base, Llama4-Maverick 400B Base, DeepSeek-V3 Base, and Kimi-K2 Base. The choice of PaLM-based terminology from the reference example is not applicable here; GLM-4.5 is evaluated in its own right as the primary subject.

- **Metrics.** Performance is measured primarily through accuracy (exact match or equivalent) on each benchmark. Key metrics include: TAU-Bench scores averaged across retail and airline domains; BFCL V3 overall score; BrowseComp accuracy; AIME 24 average accuracy over 32 samples (Avg@32); GPQA average accuracy over 8 samples (Avg@8); MATH-500 accuracy; SWE-bench Verified pass rate; Terminal-Bench success rate; MMLU-Pro accuracy; MMLU exact match; SimpleQA correctness; IFEval prompt strict accuracy; SysBench ISR score; MultiChallenge score; SafetyBench accuracy; and the Artificial Analysis intelligence index for aggregate reasoning. For CC-Bench, metrics include win/tie/loss rates, tool calling success rate, and token usage per interaction. Translation is scored by human evaluators on a 0–3 scale.

- **Baselines.** The paper evaluates against a comprehensive set of contemporaneous models: o3 (OpenAI), o4-mini-high (OpenAI), GPT-4.1 (OpenAI), Claude Opus 4 (Anthropic), Claude Sonnet 4 (Anthropic), Gemini 2.5 Pro (Google DeepMind), Grok 4 (xAI), DeepSeek-R1-0528, DeepSeek-V3-0324, Kimi K2 (Kimi Team), Qwen3-235B-A22B-Thinking-2507, Qwen3-Coder, and MiniMax-M1. For base model comparisons: Qwen3-235B-A22B Base, Llama4-Maverick 400B Base, DeepSeek-V3 Base, and Kimi-K2 Base. For translation: Qwen-MT-plus, Qwen-MT-turbo, and Seed-X.

- **Generation budget / compute accounting.** The paper does not conduct internal FLOPs-matched comparisons or controlled generation budget sweeps like the best-of-N vs. beam search experiments in the reference example. Instead, evaluation follows benchmark-specific protocols: for AIME, average over 32 samples; for GPQA, average over 8 samples; for SWE-bench Verified, OpenHands v0.34.0 with 100 iterations and history truncation to 128K context, temperature=0.6, top_p=1.0; for Terminal-Bench, the Terminus framework with standard function calling; for BrowseComp, varying interaction turns from 8 to 128 (log scale). The paper measures test-time compute scaling for BrowseComp (Figure 8) by varying interaction turns, observing that accuracy scales smoothly from ~5% to ~26% as turns increase from 8 to 128.

- **Cross-validation / statistical protocol.** No formal cross-validation or statistical significance testing is reported. For manual evaluations (general chat, coding agent, logical reasoning, translation), the paper describes protocols designed to minimize bias: randomized response ordering, single consistent evaluators for head-to-head comparisons, and predefined scoring rubrics. For the CC-Bench coding agent evaluation, tasks were executed in isolated containerized environments with "the same expert follow[ing] consistent interaction strategies across all models" (Section 4.3.2). Confidence intervals are not reported.

### Main Quantitative Results

#### #### Overall ARC Performance

The paper's headline result is that GLM-4.5 ranks 3rd overall among all evaluated models (both open-source and proprietary) averaged across 12 ARC benchmarks, and 2nd specifically on agentic tasks (Figure 1). The aggregate performance (estimated from the bar chart in Figure 1) places GLM-4.5 at approximately 63.2 on a 0–100 scale, behind o3 (~68.8) and Claude Opus 4 (~66.1), but ahead of Claude Sonnet 4 (~59.8), Gemini 2.5 Pro, o4-mini-high, and all other open-source models including DeepSeek-R1-0528 and Kimi K2. GLM-4.5-Air ranks 6th overall at approximately 55.2.

Breaking down by capability axis (Figure 1 sub-panels):
- **Agentic** (TAU-Bench, BFCL V3, BrowseComp): GLM-4.5 ranks 2nd (~58.1), behind o3 (~63.2) but ahead of Claude Sonnet 4 (~55.2) and Claude Opus 4 (~54.6).
- **Reasoning** (MMLU-Pro, AIME 24, MATH-500, SciCode, GPQA, HLE, LCB): GLM-4.5 ranks approximately 5th (~68.8), behind Grok 4 (~73.2), Gemini 2.5 Pro (~70.5), o3 (~70.0), and DeepSeek-R1-0528 (~68.3).
- **Coding** (SWE-bench Verified, Terminal-Bench): GLM-4.5 ranks 3rd (~50.9), behind Claude Opus 4 (~55.5) and Claude Sonnet 4 (~53.0), but ahead of o3 (~49.7), Kimi K2 (~45.2), and GPT-4.1 (~39.5).

#### #### Agentic Benchmarks

Table 3 reports the full agentic results. On TAU-Bench Retail, GLM-4.5 scores 79.7% vs. Claude Sonnet 4 at 80.5% and Claude Opus 4 at 81.4%—competitive with the best models. On TAU-Bench Airline, GLM-4.5 scores 60.4%, near Claude Sonnet 4 (60.0%) and Claude Opus 4 (59.6%). The standout result is BFCL V3, where GLM-4.5 achieves 77.8%—the highest score among all evaluated models, ahead of Claude Sonnet 4 (75.2%), Claude Opus 4 (74.4%), and o3 (72.4%). On BrowseComp, GLM-4.5 scores 26.4%, substantially behind o3 (49.7%) but close to o4-mini-high (28.3%) and ahead of Claude Opus 4 (18.8%) and Claude Sonnet 4 (14.7%). The average agentic score places GLM-4.5 at 58.1% vs. o3 at 61.1% and Grok 4 at 55.4%.

#### #### Reasoning Benchmarks

Table 4 presents the reasoning results. On AIME 24, GLM-4.5 achieves 91.0% Avg@32—competitive with o3 (90.3%) and DeepSeek-R1-0528 (89.3%), though behind Grok 4 (94.3%) and Qwen3-235B-2507 (94.1%). On MATH-500, GLM-4.5 scores 98.2%, comparable to Claude Opus 4 (98.2%) and within the narrow range of top models (most clustering at 98–99%). On GPQA, GLM-4.5 scores 79.1% Avg@8—below Grok 4 (87.7%), Gemini 2.5 Pro (84.4%), and o3 (82.7%), but competitive with Claude Opus 4 (79.6%). On HLE, GLM-4.5 scores 14.4%—substantial but well behind Grok 4 (23.9%), Gemini 2.5 Pro (21.1%), and o3 (20.0%). On SciCode, GLM-4.5's 41.7% slightly edges o3 (41.0%) and Claude Opus 4 (39.8%). On LCB, GLM-4.5 scores 72.9%—below Grok 4 (81.9%), Gemini 2.5 Pro (80.1%), o3 (78.4%), and Qwen3-235B-2507 (78.2%). On MMLU-Pro, GLM-4.5 scores 84.6%, within the tight band of 84.5–87.3% occupied by most flagship models. The Artificial Analysis intelligence index (estimated) places GLM-4.5 at 67.7—below Grok 4 (73.2), Gemini 2.5 Pro (70.5), and o3 (70.0), but near DeepSeek-R1-0528 (68.3) and above Claude Opus 4 (64.4).

GLM-4.5-Air shows consistent degradation from GLM-4.5 of roughly 2–4 percentage points on most reasoning benchmarks (AIME: 89.4% vs. 91.0%; GPQA: 75.0% vs. 79.1%; HLE: 10.6% vs. 14.4%), with AI Index of 64.8 vs. 67.7.

#### #### Coding Benchmarks

Table 5 reports coding results. On SWE-bench Verified, GLM-4.5 scores 64.2%—a strong result placing it 4th behind Claude Sonnet 4 (70.4%), o3 (69.1%), and Claude Opus 4 (67.8%), but ahead of Kimi K2 (65.4%), GPT-4.1 (48.6%), and Gemini 2.5 Pro (49.0%). This is notably better than DeepSeek-R1-0528 (41.4%) and DeepSeek-V3 is not reported on SWE-bench Verified. On Terminal-Bench, GLM-4.5 scores 37.5%—second only to Claude Opus 4 (43.2%) and ahead of Claude Sonnet 4 (35.5%), o3 (30.2%), and GPT-4.1 (30.3%). The average coding score of 50.9% places GLM-4.5 3rd, behind Claude Opus 4 (55.5%) and Claude Sonnet 4 (53.0%). GLM-4.5-Air trails at 57.6% on SWE-bench Verified and 30.0% on Terminal-Bench (average 43.8%).

Figure 2 positions GLM-4.5 and GLM-4.5-Air on the SWE-bench Verified vs. parameters Pareto frontier. With 355B parameters, GLM-4.5 achieves a higher SWE-bench score than Kimi K2 (1043B params, 65.4%) and DeepSeek-R1-0528 (671B params, 41.4%), roughly matching Kimi K2 with ~3× fewer total parameters. Proprietary models (Claude, GPT-4.1, Gemini) are plotted at the right edge with unknown parameter counts, serving as an upper bound.

#### #### General Chat and Safety Benchmarks

Table 6 shows general chat results. On MMLU, GLM-4.5 scores 90.0%—in the narrow range of 89.1–91.9% where most flagship models cluster. On SimpleQA, which measures factual knowledge, GLM-4.5 scores 26.4%—comparable to DeepSeek-R1-0528 (27.8%) and DeepSeek-V3-0324 (27.7%), but substantially below Gemini 2.5 Pro (54.0%), Grok 4 (51.9%), and Qwen3-235B (45.8%). On IFEval, GLM-4.5 scores 86.1%—competitive but below Grok 4 (92.4%), Gemini 2.5 Pro (90.8%), and Kimi K2 (89.8%). On SysBench, GLM-4.5 scores 81.0%—close to top models (Grok 4: 81.5%, Qwen3-235B: 83.3%, Gemini 2.5 Pro: 82.2%). On MultiChallenge, GLM-4.5 scores 52.8%—below Grok 4 (65.2%), Qwen3-235B (58.2%), and Gemini 2.5 Pro (57.5%), but above GPT-4.1 (38.3%) and DeepSeek-R1-0528 (46.5%).

Table 7 reports SafetyBench results. GLM-4.5 achieves an overall safety score of 89.87%, competitive with Gemini 2.5 Pro (90.48%) and Kimi K2 (90.48%), and slightly ahead of GPT-4.1 (89.71%). Category-level performance is strong on Physical Health (96.67%), Mental Health (94.67%), and Ethics & Morality (94.33%), with lower scores on Offensiveness (83.0%) and notably Unfairness & Bias (77.4%)—a category the paper acknowledges as "an area of ongoing focus." GLM-4.5-Air trails at 87.75% overall.

#### #### Base Model Evaluation

Table 2 compares GLM-4.5-Base with other open-source base models (pre-instruction-tuning). On English benchmarks: GLM-4.5-Base scores 30.0 on SimpleQA (vs. DeepSeek-V3 Base's 26.6 and Kimi-K2 Base's 35.3); 86.2 on BBH (roughly 1–2 points below competitors); 86.1 on MMLU (within 1–2 points of the 87.2–87.8 range). On code: EvalPlus Pass@1 of 78.1 (vs. Kimi-K2 Base's 80.3 and Qwen3-235B-A22B Base's 77.6); LiveCodeBench-Base Pass@1 of 28.1 is the highest reported, ahead of Kimi-K2 Base (26.3) and DeepSeek-V3 Base (24.6). On math: GSM8K of 79.4 is significantly below Qwen3-235B-A22B Base (94.4) and Kimi-K2 Base (92.1); MATH of 61.0 trails most competitors in the 62.6–71.8 range. On Chinese: Chinese-SimpleQA of 70.1 is competitive with Kimi-K2 Base (77.6) and DeepSeek-V3 Base (72.1). These base model results establish that the pre-trained model is broadly competent but not dominant before post-training, validating that the post-training pipeline (Expert Model Iteration, RL) is responsible for the substantial gains that produce the final GLM-4.5's competitive positioning.

#### #### Manual Evaluation: General Chat

Tables 8, 9, and 10 break down human evaluation scores for general chat by language. On English prompts (Table 8), GLM-4.5 achieves an overall score of 8.66/10, slightly ahead of DeepSeek-R1-0528 (8.62) and more clearly ahead of Kimi K2 (8.13). GLM-4.5 leads substantially in Text Generation (8.61 vs. DeepSeek-R1-0528's 7.83), Logical Reasoning (9.25 vs. 9.07), and Text Processing (8.00 vs. 8.27—the one category where DeepSeek-R1-0528 leads). On Chinese prompts (Table 9), GLM-4.5 leads overall at 8.37 vs. DeepSeek-R1-0528's 8.05 and Kimi K2's 7.03, with particularly strong showings in Text Generation (9.00 vs. 8.59), Logical Reasoning (9.27 vs. 9.00), and Code (8.89 vs. 8.67). On other languages (Table 10), GLM-4.5 leads overall at 8.49 vs. DeepSeek-R1-0528's 8.27 and Kimi K2's 6.63, with particular strength in Subjective QA (9.33 vs. 9.44—a rare loss) and Text Generation (8.90 vs. 7.86).

#### #### Manual Evaluation: Coding Agent (CC-Bench)

Figure 12 reports head-to-head results on the custom CC-Bench (52 programming tasks using Claude Code). GLM-4.5 vs. Claude Sonnet 4: 40.4% win, 9.6% tie, 50.0% loss—competitive with the leading proprietary coding model. GLM-4.5 vs. Kimi K2: 53.9% win, 17.3% tie, 28.8% loss—a clear advantage. GLM-4.5 vs. Qwen3-Coder: 80.8% win, 7.7% tie, 11.5% loss—a dominant margin.

Figure 13 shows tool calling success rate and token efficiency on CC-Bench. GLM-4.5 achieves the highest tool calling success rate at 90.6%, ahead of Claude Sonnet 4 (89.5%), Kimi K2 (86.2%), and Qwen3-Coder (77.1%). Token usage per interaction (lower is more efficient): GLM-4.5 uses 695,921 tokens per round, substantially less than Claude Sonnet 4 (2,069,449) and Kimi K2 (1,207,152), but more than Qwen3-Coder (1,388,259 is misreported in the paper—the bar chart in Figure 13 shows Qwen3-Coder at approximately 1,400,000, not 1,388,259 as the text states). The data labels in the figure read: GLM-4.5: 695,921; Qwen3-Coder: 1,388,259; Kimi K2: 1,207,152; Claude Sonnet 4: 2,069,449. GLM-4.5's token efficiency combined with its highest success rate suggests it produces more reliable function calls with less verbosity.

#### #### Manual Evaluation: Logical Reasoning

Table 11 shows expert-scored performance on novel logical reasoning problems. GLM-4.5 scores 62.0, essentially tied with DeepSeek-R1-0528 (62.1) and behind only Gemini 2.5 Pro (65.8). GLM-4.5-Air trails at 53.4, with Kimi K2 at 51.9. This provides evidence that GLM-4.5's reasoning capability is genuine and not primarily due to benchmark contamination, as the problems are described as "novel and complex logical reasoning problems that are structurally different from those widely available on the internet."

#### #### Translation Evaluation

Table 12 reports human evaluation scores on 100 challenging translation cases. GLM-4.5 scores 1.71/3.0 on average, dramatically outperforming specialized translation models: Qwen-MT-plus (0.38), Qwen-MT-turbo (0.55), and Seed-X (0.65). The paper highlights an illustrative example: translating "三花公主驾到" (literally "three-flower princess arrives") where GLM-4.5 correctly identifies "三花" as referring to a calico cat and translates idiomatically as "The Calico Princess has arrived!" while specialized models produce contextually wrong translations.

#### #### Interaction Turns Scaling for BrowseComp

Figure 8 demonstrates test-time compute scaling for the BrowseComp web browsing benchmark. Accuracy increases approximately log-linearly with interaction turns: ~5% at 8 turns, ~10% at 16 turns, ~15% at 32 turns, ~20% at 64 turns, and ~26% at 128 turns. This is a roughly 5× improvement purely from allowing the model additional rounds of search and synthesis, without any model weight changes. The paper frames this as the agentic analog of output token scaling in reasoning models: rather than generating longer chain-of-thought traces, agentic models scale performance by taking more actions in the environment.

### Ablation Studies and Robustness Checks

**Difficulty-based curriculum learning for reasoning RL**: Figure 5 compares a two-stage difficulty curriculum (moderate-difficulty data with samples_per_prompt=16, followed by extremely difficult data with samples_per_prompt=512) against a baseline that continues with moderate-difficulty data throughout. The two-stage approach reaches 83.4% Avg@32 on AIME'24, while the baseline plateaus at 81.8%. The switch to extremely hard problems occurs at training step 1500, and the blue line continues improving while the red line flattens. The paper explicitly states that "all problems used in the second stage are strictly sourced from a pool with verified correct answers."

**Single-stage vs. multi-stage length-constrained RL**: Figure 6 compares single-stage RL at 64K maximum output length against a multi-stage approach (16K → 32K → 48K → 64K). The single-stage approach (red line) reaches 83.4% on AIME'24 Avg@32, while multi-stage (blue line) plateaus at 80.6%. The multi-stage curve shows a visible performance dip during the 16K stage that it never fully recovers from, supporting the claim of "irreversible" degradation.

**Token-weighted mean loss for code RL**: Figure 7 (left) compares token-weighted mean loss against sequence-mean loss for code RL measured on LiveCodeBench accuracy. The token-weighted approach (blue line) converges faster and reaches 46.5%, while sequence-mean (red line) plateaus at 46.3%. The steeper initial slope of the token-weighted curve is visible in the first 500 training steps.

**Data quality for science RL**: Figure 7 (right) compares training on expert-verified multiple-choice data against mixed-quality science data for GPQA-Diamond. Expert-verified data (blue line) reaches 65.8%, while mixed-quality data (red line) plateaus around 62.9%. The performance gap is ~3 percentage points, and the expert-verified curve continues improving over the training steps while the mixed-quality curve flattens earlier.

**Prompt selection by response length for SFT data**: The paper reports (Section 3.1, under "Prompt Selection and Response-Level Scaling") that "removing the prompts in the bottom 50% based on response lengths" yields "a 2%–4% improvement in math and science tasks, despite training with only half the data." This is not shown in a figure but is reported as an empirical finding.

**Response-level scaling**: The paper further reports that "generating four responses for each prompt brought an additional 1%–2% improvement" beyond the prompt selection improvement. This is a separate ablation from the prompt selection finding, indicating that both selecting hard prompts and providing multiple correct response trajectories per prompt independently improve performance.

**Function call template comparison**: The paper states that "experimental results demonstrate that the proposed function call template does not compromise the performance of function call execution while reducing escaping" (Section 3.1). No specific numbers or figure are provided for this ablation.

**Expert Model Iteration structure**: The entire Expert Model Iteration framework (training separate expert models and distilling them) is presented without an ablation comparing it to a sequential multi-objective RL baseline on a single model. The effectiveness of this approach must be inferred from the final model's strong balanced performance across all three ARC domains, rather than from a controlled experiment.

**Dynamic sampling temperature**: The paper describes the mechanism and credits it to Polaris (An et al., 2025), referencing the periodic validation-set evaluation across temperature ranges with a 1% performance drop threshold. No specific ablation figure compares fixed vs. dynamic temperature training; the claim is supported by citation and description rather than direct experimental evidence within this paper.

**KL loss term exclusion in GRPO**: The paper states that the GRPO framework is used "excluding the KL loss term" but does not provide an ablation comparing with-KL and without-KL training. The decision is presented as a design choice without experimental justification within the report.

**ReST^EM revision model**: The paper does not include experiments with the ReST^EM revision model approach described in the reference example. The revision capability is handled through the unified training distillation rather than through a separate sequential revision model.

**Iterative self-distillation for agentic RL**: The paper describes this as a strategy to "push the performance limits of RL-trained models efficiently" but does not provide a controlled experiment comparing single-pass RL against iterative distillation for agentic tasks. The evidence is the overall strong agentic performance rather than an ablation.

**Pathology RL effectiveness**: The paper states that pathology RL "impose[s] efficient penalties, further lowering the residual error rates for these problematic behaviors" but provides no quantitative comparison of error rates with and without pathology RL. The incidence rate of <1% and the claimed improvement are stated without supporting figures.

**Oracle vs. predicted difficulty for curriculum**: Unlike the reference example's paper, GLM-4.5 uses a difficulty-based curriculum where difficulty is determined by whether pass@N is zero or non-zero for different sample sizes, rather than by an oracle or predicted difficulty estimator. This means the curriculum depends on the model's own current capability, which is a fundamentally different approach from the reference example's binning-by-base-model-pass@1. No ablation compares this capability-relative difficulty definition with an absolute difficulty metric.

### Critical Assessment

#### Does GLM-4.5 genuinely unify ARC capabilities, or does it perform well on average by being "good enough" across the board without truly excelling at any single dimension?

The paper's core narrative is that GLM-4.5 is a "generalist" that "excels across all three areas." The evidence supports that it is competitive across all three, but the degree of "excellence" varies substantially by domain. GLM-4.5's strongest showing is on agentic tasks (2nd overall, 58.1% average), where it genuinely competes at the top tier—ranking ahead of Claude Opus 4 and Claude Sonnet 4. On coding, it ranks 3rd (50.9% average), solidly in the top tier but with a clear gap to the Claude models (55.5%, 53.0%). On reasoning, it ranks approximately 5th (67.7 AI Index), with a larger gap to the leaders (Grok 4 at 73.2, Gemini 2.5 Pro at 70.5). The model is genuinely excellent at agentic tasks and strong at coding, but it is merely "competitive" at reasoning—a tier below the best models. The paper's framing as a unified ARC model is accurate but the claim of balanced excellence should be qualified: the model is agentic-first, coding-strong, reasoning-competitive. This is not a weakness per se—tradeoffs are inevitable—but the paper's presentation could more explicitly acknowledge this hierarchy.

A specific concern is the AIME 24 result. The paper reports 91.0% Avg@32, which appears to be a very strong showing. However, Avg@32 is a metric that benefits substantially from repeated sampling—it measures whether the correct answer appears in any of 32 attempts. This is a different construct from pass@1, which would measure the model's ability to solve the problem in a single attempt. The paper does not report pass@1 for AIME, making it difficult to assess whether GLM-4.5's reasoning is genuinely precise or whether it benefits from diversity across multiple samples. The comparison to o3 at 90.3% Avg@32 suggests parity, but without pass@1 numbers, the comparison is incomplete. Models with higher pass@1 but lower Avg@32 would be more reliable in single-attempt scenarios; models with lower pass@1 but high Avg@32 might be less reliable but more exploratory.

#### Do the manual evaluations (CC-Bench, logical reasoning, translation) provide evidence beyond the standard benchmarks, or do they introduce new confounds?

The custom manual evaluations are ostensibly designed to address benchmark contamination concerns and to assess real-world performance. However, several aspects warrant scrutiny:

**CC-Bench sample size and protocol.** The benchmark consists of 52 tasks, which is small enough that individual task difficulty variance could substantially affect win/loss/tie ratios. The evaluation protocol—"the same expert followed consistent interaction strategies across all models"—controls for inter-evaluator variance but introduces the possibility that this particular expert's interaction style favors certain models. The paper does not report whether multiple experts were used or whether results were consistent across different interaction styles. More critically, the CC-Bench evaluation uses Claude Code as the underlying framework—testing GLM-4.5, Kimi K2, and Qwen3-Coder within an Anthropic-designed tool environment. Framework-model interactions could advantage or disadvantage specific models in ways that don't reflect general coding agent capability. A model that happens to produce function calls better suited to Claude Code's parsing expectations would appear stronger on this benchmark without necessarily being better at coding in general.

**Logical reasoning test set construction.** The paper states the problems are "structurally different from those widely available on the internet" to mitigate contamination. However, the method of constructing these problems is not described—were they human-authored, template-generated, or LLM-synthesized? The sample size is not disclosed. The scoring protocol—"unified and detailed scoring standard for each question" and "inspected and scored by human experts"—is described but the number of evaluators and inter-rater reliability are not reported. These omissions make it impossible to assess the robustness of the finding that GLM-4.5 (62.0) is essentially tied with DeepSeek-R1-0528 (62.1). A 0.1-point difference on an undisclosed number of questions scored by an undisclosed number of human evaluators could easily be noise.

**Translation evaluation coverage.** The 100 translation cases are described as "challenging, real-world cases commonly mistranslated by current tools." These are deliberately adversarial examples selected to be hard for specialized models. GLM-4.5's strong performance (1.71 vs. 0.38–0.65) may partly reflect that general-purpose LLMs have seen more diverse linguistic phenomena during pre-training than specialized translation models, making them more robust on edge cases. But the sample is likely not representative of typical translation quality—on common, straightforward translations, specialized models might outperform GLM-4.5. The paper's framing as "GLM-4.5 significantly outperforms specialized models" is accurate for this specific set of challenging cases but should not be interpreted as claiming superiority on translation tasks in general. The paper acknowledges this implicitly by describing the set as "challenging" and "commonly mistranslated," but the headline comparison could be misleading to readers who don't note this qualification.

#### Are the baseline comparisons fair, particularly regarding test-time compute allocation?

The paper evaluates GLM-4.5 against a wide range of models, which is commendable. However, the test-time compute allocation across models is not standardized. GLM-4.5's AIME score is Avg@32, meaning 32 samples are drawn and the best is selected. The paper reports the same Avg@32 metric for comparison models—but only if those numbers are publicly available. If a comparison model did not report Avg@32, its number might come from Avg@8, pass@1, or a different sampling strategy described in that model's own technical report. The paper does not clearly distinguish between metrics that were computed identically versus those that come from different protocols. This is a standard challenge in LLM benchmarking (the field lacks rigorous standardization), but it means the AIME numbers in Table 4 may not be fully comparable.

Similarly, SWE-bench Verified results depend heavily on the agent framework, iteration budget, and configuration. The paper uses OpenHands v0.34.0 with 100 iterations for GLM-4.5. Comparison models' SWE-bench scores may have been obtained with different frameworks (SWE-agent, Aider, custom harnesses) and different iteration limits. The paper does not control for this framework-level variation, making the SWE-bench comparisons approximate rather than precise. The Terminal-Bench comparison uses "the Terminus framework and standard function calling" for GLM-4.5, but the paper does not specify whether comparison models used the same framework.

The BrowseComp interaction turns scaling (Figure 8) is presented for GLM-4.5, but no comparison models are shown on the same scaling curve. We know GLM-4.5 reaches 26.4% at some turn count (presumably 128, based on the table), but we don't know how o3 (49.7%) or o4-mini (28.3%) would scale with additional turns. A model that achieves 28.3% at 16 turns might reach 50% at 128 turns—or might plateau at 28%. Without scaling curves for comparison models, the BrowseComp comparison is a single-point comparison at potentially different effective compute budgets.

#### How reliable are the base model comparisons given different training data regimes?

Table 2 compares GLM-4.5-Base with other open-source base models on a range of benchmarks. However, base model evaluation is particularly sensitive to the evaluation protocol—few-shot vs. zero-shot, exact prompt format, answer extraction method. The paper states that "GLM-4.5-Base scores are from our internal evaluation framework," but does not specify whether comparison models' scores come from the same framework or from their respective technical reports. If the evaluation frameworks differ, the comparisons may not be apples-to-apples. The GSM8K result is particularly striking: GLM-4.5-Base scores 79.4, while Qwen3-235B-A22B Base scores 94.4 and Kimi-K2 Base scores 92.1. A 13–15 point gap on GSM8K is large and might partly reflect evaluation differences (e.g., 5-shot vs. 8-shot, different answer extraction) rather than genuine capability differences. The paper does not provide sufficient methodological detail to rule out this concern.

#### What's missing that would strengthen the experimental case?

**Pass@1 for reasoning benchmarks.** Reporting pass@1 alongside Avg@32 (or Avg@8) would provide a more complete picture of the model's reasoning precision. A model might achieve high Avg@32 by generating diverse strategies that occasionally hit the correct answer, without reliably reasoning correctly. Pass@1 measures reliability in single-attempt scenarios, which is closer to most real-world use cases.

**Controlled compute-matched comparisons.** The paper makes an implicit claim about parameter efficiency (Figure 2), but never conducts a controlled experiment where total FLOPs or inference cost is held constant across models. Such an experiment would compare GLM-4.5 (32B activated) against a larger model at the same per-query cost, or against itself with varying test-time compute budgets. The reference example's paper does this systematically; GLM-4.5's technical report does not. The Figure 2 Pareto frontier is suggestive but observational—it compares different models evaluated under different conditions, not a controlled experiment.

**Ablation of Expert Model Iteration vs. sequential multi-objective RL.** The Expert Model Iteration framework is presented as a key innovation, but its benefit over the alternative (training a single model sequentially on all objectives) is never experimentally demonstrated. This would be a substantial training run (requiring training two complete models), so its absence is understandable, but it means the Expert Model Iteration's contribution remains a design philosophy rather than an empirically validated advantage.

**Confidence intervals and statistical testing.** Not a single confidence interval, standard deviation, or statistical test appears in the paper, despite evaluating on benchmarks like the 500-question SWE-bench Verified and 500-question MATH-500 where sampling variance is non-trivial. A 64.2% on SWE-bench Verified (321/500) has a standard error of roughly ±2.1%—meaning GLM-4.5's "true" SWE-bench performance could reasonably be anywhere in the 60–68% range (95% CI). Comparisons like GLM-4.5 (64.2%) vs. Kimi K2 (65.4%) could be entirely explained by sampling variance. Similarly, the 52-task CC-Bench win rates (40.4% win vs. Claude Sonnet 4 = 21/52 wins) have substantial uncertainty. Without confidence intervals, the reported rankings—3rd overall, 2nd on agentic—should be understood as point estimates with non-trivial uncertainty.

**Cross-validation or held-out evaluation for strategy selection.** Unlike the reference example's paper, which uses two-fold cross-validation to select compute-optimal strategies without overfitting to the test set, GLM-4.5's report does not describe any cross-validation protocol. The post-training decisions (difficulty curriculum design, dynamic temperature scheduling, expert model iteration, data filtering strategies) were presumably optimized against benchmark performance, creating a risk of overfitting the training recipe to the specific benchmarks reported. The custom benchmarks (CC-Bench, the novel logical reasoning set) partially mitigate this concern by providing held-out evaluations, but the core ARC benchmarks are the optimization target.

**Detailed hardware and carbon cost reporting.** The paper describes training on 23T tokens with a 355B-parameter MoE model, which is an enormous computational undertaking. No estimates of total FLOPs, GPU-hours, energy consumption, or carbon emissions are provided. For a paper that emphasizes parameter efficiency and open-source accessibility, this omission is notable—the environmental and financial cost of producing the model is part of its overall efficiency story.

**Long-context evaluation benchmarks.** The paper describes extensive long-context training (up to 128K tokens) but does not report results on standard long-context benchmarks like LongBench (Bai et al., 2023), LongBench v2 (Bai et al., 2025), RULER (Hsieh et al., 2024), or Michelangelo (Vodrahalli et al., 2024). The model's effective context utilization—as opposed to its ability to not crash at long contexts—is untested in the reported evaluation. This is a significant gap given the paper's emphasis on long-context training for agentic tasks.

**Evaluation of the hybrid reasoning mode switching.** The paper claims the model supports both thinking and non-thinking modes but never evaluates whether the model correctly chooses between them at inference time. Does it engage in lengthy chain-of-thought for trivial queries? Does it fail to reason for complex problems? Is there a latency/accuracy tradeoff that can be controlled? The hybrid reasoning capability is a marketed feature without corresponding evaluation.

Overall, the experimental section demonstrates that GLM-4.5 is a strong, competitive model—particularly on agentic and coding tasks—with performance that substantiates its position near the top of the open-source leaderboard. The breadth of evaluation across 12+ benchmarks and multiple manual evaluations is commendable. However, the absence of statistical rigor (no confidence intervals), the lack of controlled compute-matched comparisons that would directly support the parameter efficiency claim, the missing ablation of the Expert Model Iteration framework against the sequential alternative, and the incomplete reporting on key metrics (pass@1 for reasoning, long-context benchmarks) mean that the paper's strongest claims—about balanced ARC excellence and parameter efficiency—are supported directionally but not with the precision that the specific numerical rankings and Pareto frontier placements imply. The paper's contribution as a model release and technical report is substantial; as a rigorous experimental demonstration of its architectural and methodological innovations, it leaves several important questions unanswered.

## 6. Limitations and Trade-offs

### The Difficulty Estimation Bottleneck for Compute-Optimal Test-Time Strategies

**The assumption or constraint.** The paper's entire compute-optimal test-time scaling framework rests on the ability to classify each prompt into one of five difficulty quintiles *before* deciding how to allocate the inference budget. The method for doing so—generating 2048 complete solutions from the base LLM and computing either the ground-truth pass@1 rate (oracle difficulty) or the PRM's average final-answer score (predicted difficulty)—is extraordinarily expensive. The paper explicitly acknowledges this in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

The 2048 samples required for difficulty estimation consume more compute than the largest test-time budgets studied (256–512 generations). In any practical deployment, the total cost is difficulty estimation *plus* strategy execution, and the former dominates the latter by a factor of ~4–8× at the budget levels where the paper reports its 4× efficiency gains.

**The consequence.** The headline 4× efficiency improvement—that compute-optimal scaling matches best-of-N performance with 4× fewer generations—is computed *after* difficulty is already known, without amortizing the cost of learning it. In a realistic deployment where difficulty must be estimated from scratch for each query, the total compute cost would be **difficulty estimation cost + strategy execution cost**, making the net efficiency substantially worse than reported. For a prompt that receives a budget of 16 generations under the compute-optimal policy, the total cost including difficulty estimation is 2048 + 16 = 2064 generations—roughly 8× the cost of simply running best-of-256 (256 generations) and substantially *more* expensive than the "inefficient" baseline. The 4× gain is only achievable if difficulty estimation is amortized over many queries with the same difficulty distribution, or if a cheaper difficulty estimator is developed. The paper acknowledges this gap explicitly—"our experiments do not account for this cost"—but the practical consequence is that the reported efficiency numbers are **upper bounds on achievable deployment efficiency** rather than realized gains.

**What evidence exists in the paper.** Figures 4 and 8 show the compute-optimal scaling curves (both oracle and predicted difficulty) outperforming best-of-N baselines, but neither figure includes the difficulty estimation cost in the x-axis (generation budget). The cost is discussed qualitatively in Section 3.2, but no experiment quantifies how the efficiency gains degrade when difficulty estimation cost is included. The paper does not report an ablation where difficulty is estimated from fewer than 2048 samples, which would be the natural first step toward practical deployment. The single mention of predicted difficulty performing similarly to oracle difficulty (Figures 4 and 8 curves "largely overlap") confirms that the PRM-based estimator works without ground-truth labels, but it does not address the 2048-sample cost.

**Mitigation status.** The paper acknowledges this as "a key avenue for future work" (Section 8) and suggests "pretraining or finetuning models to directly predict difficulty of a question." No such model is developed or evaluated. The limitation is recognized but completely unaddressed in the current system. A practitioner deploying this method today would need to either accept the prohibitive difficulty estimation cost, develop a cheaper estimator independently, or amortize estimation over many similar queries (which limits the method's applicability to batch processing scenarios).

---

### Hard Problems Remain Essentially Unsolved Regardless of Compute Budget

**The assumption or constraint.** The compute-optimal framework is built on the premise that test-time compute can substitute for pretraining compute, but this substitution has a hard boundary: the base model must be capable of producing correct solutions at some non-trivial rate for test-time strategies to help. The paper is explicit about this in Section 7:

> "test-time compute amplifies existing capability but does not create it from nothing"

Problems in difficulty bin 5—the hardest quintile, where the base model's pass@1 is near zero—show essentially no improvement from any amount or configuration of test-time compute. This is not a training artifact or an allocation failure; it is a fundamental ceiling.

**The consequence.** Across all methods—PRM search, iterative revisions, and their compute-optimal combinations—bin 5 accuracy hovers at 1–3% regardless of compute budget (Figure 3, right panel for search; Figure 7, right panel for revisions; Figure 9, bin 5 curve for FLOPs-matched comparison). The ~14× larger pretrained model also performs poorly on bin 5 in the FLOPs-matched comparison (Figure 9), but it does show *some* improvement over the base model, whereas test-time compute provides essentially zero incremental benefit. For any deployment where the query distribution includes a substantial fraction of genuinely hard problems (those beyond the base model's current capability frontier), **no amount of inference-time compute will help**—the only path to improvement is further pretraining on more capable base models. This means the method offers a false promise for the hardest problems, which are often the ones practitioners most want to solve.

The practical consequence is that difficulty estimation serves not only to select optimal strategies but also as a **capability gate**: problems classified as bin 5 should be flagged for escalation to a larger model or human review rather than allocated additional test-time compute that will be wasted. The paper does not frame it this way, but the data clearly show that the ROI of test-time compute on bin 5 problems is essentially zero. Any system deploying this method needs a separate policy for hard problems—the compute-optimal framework provides no guidance beyond "give up and use a bigger model."

**What evidence exists in the paper.** The evidence is consistent and unambiguous across multiple figures:
- **Figure 3 (right):** For beam search and best-of-N weighted, bin 5 accuracy starts near 1–2% at 4 generations and remains near 1–3% at 256 generations—no meaningful improvement.
- **Figure 7 (right):** For sequential vs. parallel revision ratios at 128 generations, bin 5 accuracy hovers at 2–3% across all ratios.
- **Figure 9 (both panels):** In the FLOPs-matched comparison, the bin 5 curves are essentially flat near 0–5% for all three R values. Test-time compute scaling provides negligible gains; the gap to the ~14× larger model remains large.
- **Table 4 (Section 4.2.2):** GLM-4.5 achieves 14.4% on HLE (Humanity's Last Exam), a benchmark explicitly designed to be extremely difficult. This is competitive but far from saturated, confirming that genuinely hard reasoning tasks remain challenging even after all the post-training innovations.

**Mitigation status.** The paper is transparent about this limitation, stating it explicitly in the Section 7 takeaway and showing the bin 5 flatlines consistently. No mitigation is proposed within the test-time compute framework because the limitation is fundamental: if the model cannot generate correct solutions at any non-trivial rate, no search or revision strategy can find what isn't there. The implicit mitigation is "use a larger pretrained model for hard problems," but the paper does not explore how to combine a smaller model with test-time compute for easy/medium problems and a larger model for hard problems in a unified cost-efficient deployment. This is a gap between what the paper shows is possible and what a practitioner would need to build.

---

### Revision and Search Mechanisms Are Studied Independently, Not Combined

**The assumption or constraint.** The paper studies two complementary axes for improving test-time performance—modifying the proposal distribution via iterative revisions (Section 6) and improving candidate selection via PRM-guided search (Section 5)—but **never combines them**. Section 8 explicitly acknowledges this gap:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

The two mechanisms have complementary strengths that are clearly demonstrated in the paper's difficulty-dependent analyses: revisions help most on easy problems where local refinement of near-correct answers is sufficient (Figure 7, right, bin 1–2), while PRM search helps most on medium problems where broader exploration of solution strategies is needed (Figure 3, right, bin 3–4). A combined system that uses the revision model as the proposal distribution within beam search—or uses the PRM to guide which revision paths to pursue—could theoretically capture the benefits of both mechanisms.

**The consequence.** The paper's results represent a **lower bound** on what a fully integrated system could achieve. The compute-optimal policy described in Section 3.1 selects between search strategies and revision strategies per difficulty bin, but it cannot combine them within a single query. This means:
- On medium-difficulty problems where beam search is deployed, the model is generating candidates from the base model's proposal distribution rather than from the (potentially stronger) revision model. The PRM is working with lower-quality candidates than it could be.
- On easy problems where sequential revisions are deployed, the model is selecting answers via majority voting or a separately trained ORM rather than using the (potentially more accurate) PRM for candidate selection within the revision chain.
- The revision model's correct-to-incorrect reversion problem (~38% of correct answers get revised to wrong ones, Section 6.1) could potentially be mitigated by using the PRM to detect when a revision is going off-track and terminate the chain early, but this is never tested.

The paper's FLOPs-matched comparisons (Section 7) treat revisions and PRM search as separate methods, showing that revisions generally outperform PRM search for the pretraining-vs-inference tradeoff. A combined system might shift this conclusion further in favor of test-time compute, potentially expanding the regime where inference-time strategies dominate pretraining.

**What evidence exists in the paper.** The complementary strength pattern is evident across the difficulty-bin analyses, but no experiment combines the two mechanisms. The revision model experiments (Section 6) use a separately trained ORM for answer selection (Appendix J, Figure 15), not the PRM used for search experiments. The PRM search experiments (Section 5) use the base few-shot prompted model for candidate generation, not the fine-tuned revision model. The two pipelines share a base model and evaluation benchmarks but are otherwise completely separate. The paper's acknowledgment in Section 8 confirms this is a deliberate scope limitation, not an oversight.

**Mitigation status.** The paper identifies combination as future work (Section 8) but provides no preliminary results or analysis suggesting how much gain a combined approach might yield. The infrastructure described in Section 3 (Expert Model Iteration, unified training) could potentially support a combined system—since the unified model already integrates reasoning and agentic capabilities—but the paper does not explore this. A practitioner building on this work would need to design and evaluate the combined system from scratch, with no guidance from the paper on expected interactions or failure modes.

---

### The Test Set Size Restricts the Reliability of Per-Difficulty-Bin Conclusions

**The assumption or constraint.** The paper's difficulty-conditioned analysis splits the MATH test set of 500 questions into five quintiles of approximately 100 questions each (Section 3.2). The compute-optimal policy is selected using two-fold cross-validation within each bin, meaning strategy selection is based on approximately 50 questions per fold per bin. For the FLOPs-matched comparison (Section 7), difficulty bins are further aggregated into three groups (easy: bins 1–2, medium: bin 3, hard: bins 4–5), with the aggregated groups containing roughly 200, 100, and 200 questions respectively.

**The consequence.** With only ~100 questions per bin and ~50 questions per cross-validation fold, the compute-optimal strategy selection has **high variance**. A strategy that appears optimal on one fold of 50 questions might not generalize to the other fold or to a new test set. The paper does not report confidence intervals on any of the compute-optimal scaling curves (Figures 4, 8, 9), making it impossible to assess whether the observed differences between strategies within a bin are statistically reliable or could be explained by sampling noise.

This is particularly consequential for the paper's core claim about the $4\times$ efficiency improvement. At 16 generations, compute-optimal search achieves approximately 27% accuracy, matching PRM best-of-N weighted at 64 generations (Figure 4). With ~100 questions per bin, a difference of a few percentage points in accuracy could shift the crossover point substantially—if compute-optimal at 16 generations actually achieves 25% (well within sampling error), the 4× claim might be closer to 2–3×. Similarly, the finding that beam search *degrades* performance on easy problems at high budgets (Figure 3, right, bin 1) is based on approximately 100 easy questions. A few outlier questions where beam search over-optimizes the PRM could drive this result disproportionately, making the apparent degradation an artifact of the specific test set rather than a robust phenomenon.

The small per-bin sample also limits the reliability of the difficulty-dependent strategy prescriptions (e.g., "use best-of-N on easy problems, beam search on medium problems"). With only 50 questions per validation fold, the optimal strategy for a bin might be determined by the idiosyncrasies of a handful of questions rather than by genuine difficulty-dependent strategy efficacy. The paper's two-fold cross-validation mitigates overfitting to the test set but does not address the fundamental limitation of small per-bin sample sizes.

**What evidence exists in the paper.** No confidence intervals, standard errors, or statistical tests are reported anywhere in the experimental sections (Sections 5, 6, 7). The figures show point estimates only, with smooth curves that give an impression of precision that the sample size does not support. The 500-question test set size is mentioned in Section 4, and the binning into quintiles is described in Section 3.2, but the statistical implications are never discussed. This is a notable omission given that the paper's core contributions—the difficulty-conditioned allocation policy and the $4\times$ efficiency claim—depend on reliable per-bin estimates.

**Mitigation status.** The paper does not acknowledge this as a limitation. The two-fold cross-validation protocol (Section 3.2) is described as a method for "avoiding circularity of selecting the best strategy and evaluating it on the same data," but there is no discussion of statistical power, variance, or minimum sample size requirements. No future work is suggested on larger test sets, bootstrap confidence intervals, or Bayesian methods for strategy selection that would account for per-bin uncertainty. The cross-validation does prevent the most egregious form of overfitting (selecting and evaluating on the same fold), but it does not address whether the selected strategies are robust to test-set variation. A practitioner deploying this method would need to either trust that the strategy prescriptions generalize (despite the small per-bin sample) or replicate the analysis on their own larger test set.

---

### Latency and Wall-Clock Time Are Not Considered in Any Efficiency Analysis

**The assumption or constraint.** The paper measures compute exclusively in terms of "generations"—the number of complete solutions sampled. One generation equals one complete forward pass producing a full solution. This is a reasonable proxy for total FLOPs consumed, but it is **not a proxy for latency or wall-clock time**, because different test-time strategies have fundamentally different parallelism characteristics:

- **Best-of-N sampling:** All $N$ solutions can be generated in parallel (with sufficient hardware), making wall-clock time roughly constant with $N$.
- **Sequential revisions:** Each revision depends on the output of the previous revision, making the chain inherently serial. A budget of 64 generations allocated as a single chain of 64 revisions takes ~64× longer wall-clock time than 64 parallel samples, even though both consume the same total FLOPs.
- **Beam search:** Beams at the same depth can be parallelized, but beams at different depths are serial. A beam search with width $M=4$ and 10 steps has 10 serial stages, each parallel over at most 4 beams.
- **Hybrid sequential-parallel:** The paper's optimal strategies for medium-difficulty problems involve both sequential and parallel components (Figure 7), with the serial depth equal to the sequential chain length.

**The consequence.** The compute-optimal policy recommended by the paper often favors strategies with high serial depth, particularly on easy problems where fully sequential revisions are optimal (Figure 7, right, bin 1–2). For latency-sensitive applications—interactive assistants, real-time coding agents, customer support chatbots—a strategy that delivers correct answers in 2 seconds with 85% accuracy may be strictly preferable to one that delivers them in 60 seconds with 88% accuracy, even if the latter is more compute-efficient in FLOPs terms.

This limitation has direct practical consequences for the paper's deployment recommendations:
- **Agentic applications** (the paper's strongest domain) are inherently latency-sensitive because users are waiting for the agent to complete multi-turn interactions. If each turn of an agentic dialogue triggers a 64-step sequential revision chain, the cumulative latency across multiple turns could render the system unusable regardless of its accuracy.
- **Batch processing** where latency doesn't matter (e.g., overnight evaluation of thousands of problems) would favor the paper's FLOPs-optimal strategies, but this use case is narrow compared to interactive deployment.
- **The FLOPs-matched comparison** (Section 7) compares test-time compute with a $14\times$ larger model using **greedy decoding** (single forward pass). The serial strategies recommended by the compute-optimal policy would have much higher latency than the larger model's single forward pass, even if they use fewer total FLOPs. For a user waiting for an answer, the larger model producing an answer in 0.5 seconds may be preferable to the smaller model producing a slightly better answer in 30 seconds.

**What evidence exists in the paper.** No latency measurements, wall-clock timing experiments, or parallelism analysis appear anywhere. The paper uses "generations" as the universal cost metric (Sections 5–7) without discussing the parallelism assumptions underlying this choice. The revision model experiments (Section 6) mention that sequential chains are generated one revision at a time, but no timing data is reported. The FLOPs-matched comparison (Section 7) explicitly accounts for pretraining and inference FLOPs but makes no adjustment for the differing latency profiles of the compared strategies. Table 3 (BrowseComp) shows interaction turns scaling, but this measures accuracy as a function of environment interaction count, not time-to-answer. Figure 13 (CC-Bench) reports token usage per interaction but not end-to-end time.

**Mitigation status.** The paper does not discuss latency at all—not as a limitation, not as a design consideration, not as future work. The focus is entirely on throughput-oriented metrics (FLOPs, generations, tokens). For a paper that emphasizes practical deployment (open-source release, parameter efficiency for lower serving cost, agentic applications), the complete omission of latency considerations is a significant gap. A practitioner building on this work would need to independently characterize the latency-accuracy tradeoffs for each strategy and determine whether the compute-optimal strategies are compatible with their latency requirements. The paper provides no guidance for this essential deployment decision.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that a unified, open-source model can be competitive across agentic, reasoning, and coding with a carefully engineered MoE design and targeted RL. The hybrid thinking/direct modes and the function-calling template address usability—important for real applications (Sections 3.1–3.4).
  - Provides a repeatable RL stack (curriculum, 64K single-stage RL, token-weighted code RL, expert-verified science RL) and an agentic RL infrastructure (Slime with FP8 rollouts) that others can adopt (Sections 3.2–3.5).

- Research avenues
  - BrowseComp gap: richer browsing policies, better planning/retrieval under noisy web, and improved process supervision for search sequences (Table 3; Figure 8).
  - Long-context reasoning: extend the single-stage RL idea beyond 64K; study interactions with retrieval and memory systems (Figure 6).
  - Bias and fairness: targeted safety RL similar to “Pathology RL,” but for fairness-sensitive prompts (Table 7).
  - Data efficiency: further study of curriculum + verified-small-pool strategies in science/math/code to reduce RL token budgets (Figure 7).
  - Better knowledge recall without sacrificing reasoning: reconcile SimpleQA underperformance with reasoning strengths via balanced SFT/RL mixtures or dual-memory mechanisms (Table 6).

- Applications
  - Production agents for software engineering (SWE-bench/CC-Bench results), enterprise automation with robust function calling (BFCL v3), and long-context analytical assistants (repo-level code, 128K contexts).
  - Multilingual assistants that handle internet culture and domain-specific idioms (Table 12), and instruction-following systems with strong schema compliance (SysBench; Section 3.4).

> Overall, the evidence across Figures 1–2 and Tables 3–5 shows `GLM-4.5` ranks 3rd overall among evaluated models and 2nd on agentic tasks with markedly fewer total parameters than some competitors, while delivering practical engineering advances (function calling format, FP8 rollout infra) that will matter in real deployments.
