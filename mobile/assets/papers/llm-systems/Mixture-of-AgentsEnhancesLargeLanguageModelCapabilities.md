# Mixture-of-Agents Enhances Large Language Model Capabilities

**ArXiv:** [2406.04692](https://arxiv.org/abs/2406.04692)

## 🎯 Pitch

This paper introduces the Mixture-of-Agents (MoA) framework, a novel methodology where multiple large language models (LLMs) collaborate by reading and improving on each other’s outputs through layered aggregation and synthesis—all via prompting alone. MoA achieves state-of-the-art performance on alignment and instruction-following benchmarks by harnessing the diverse strengths of existing LLMs, outperforming even closed-source giants like GPT-4o, while also enabling greater cost-effectiveness and flexibility without any additional training. This approach signals a powerful shift away from monolithic models, making superior LLM capabilities more accessible, scalable, and interpretable across the AI ecosystem.

---

## 1. Executive Summary

This paper proposes a **Mixture-of-Agents (MoA)** methodology that harnesses the collective strengths of multiple large language models through a layered architecture, where each layer’s agents take all outputs from the previous layer as auxiliary information to iteratively refine their responses. The approach is evaluated primarily on AlpacaEval 2.0, MT-Bench, and FLASK using open-source models—Qwen1.5-110B-Chat, Qwen1.5-72B-Chat, WizardLM-8x22B, LLaMA-3-70B-Instruct, Mixtral-8x22B-v0.1, and dbrx-instruct—as proposers and aggregators across multiple layers. The paper identifies a phenomenon termed **collaborativeness of LLMs** (a model generates better responses when given outputs from other models, even if those models are individually less capable) and leverages it through a successive aggregation pipeline (multiple independent models generate responses, then an aggregator synthesizes them into a refined output across repeated cycles). MoA achieves a state-of-the-art AlpacaEval 2.0 LC win rate of 65.1% using only open-source models, surpassing GPT-4 Omni by a substantial 7.6% absolute margin (57.5% → 65.1%), while a cost-effective variant, MoA-Lite, delivers performance comparable to GPT-4 Turbo at approximately half the cost, establishing that multi-model collaborative synthesis achieves these gains without requiring fine-tuning or weight access, operating purely through prompting.

## 2. Context and Motivation

### The Core Problem: How to Leverage Multiple LLMs Collectively

The fundamental question this paper addresses is deceptively simple: **given the growing number of large language models, each with different strengths and weaknesses, how can we harness their collective expertise to produce better outputs than any single model can alone?** This matters because the field is no longer dominated by a handful of proprietary models — there is now a rich ecosystem of open-source LLMs (Qwen, LLaMA, Mixtral, WizardLM, DBRX) alongside proprietary offerings like GPT-4, all with distinct training recipes, data mixtures, and resulting skill profiles. Yet the default deployment paradigm remains stubbornly singular: pick one model and use it in isolation.

This gap is significant for several reasons laid out explicitly and implicitly in Section 1:

- **Escaping the scaling ceiling**: The paper acknowledges that "despite the plethora of LLMs and their impressive achievements, they still face inherent constraints on model size and training data" and that "further scaling up these models is exceptionally costly, often requiring extensive retraining on several trillion tokens." If multi-model collaboration can substitute for raw scale, it offers a path to improved capability without the prohibitive cost of training ever-larger models.

- **Exploiting complementary expertise**: Different models specialize in different task aspects. The paper notes that "some models excel at complex instruction following while others may be better suited for code generation." A deployment that uses only one model necessarily accepts its weaknesses alongside its strengths. The MoA approach reframes this: rather than searching for the single best model, build a system where models compensate for each other's deficiencies.

- **Practical deployment economics**: The cost analysis in Section 3.4 (Figure 5a) demonstrates that multi-model approaches can achieve performance comparable to or exceeding GPT-4 while being more cost-effective — MoA-Lite matches GPT-4o's cost while achieving higher quality, and outperforms GPT-4 Turbo by approximately 4% at roughly half the cost. This has direct implications for organizations deciding how to allocate their inference budgets.

### Where Prior Approaches Fall Short

The paper identifies several lines of prior work, each with specific limitations that MoA addresses:

**LLM ranking and reranking is too conservative.** Approaches like PAIRRANKER (Jiang et al., 2023) and router-based systems (Wang et al., 2024a; Shnitzer et al., 2024; Lu et al., 2023) select the best output from a set of candidate LLM responses rather than synthesizing something new. The paper demonstrates empirically (Figure 4a) that an LLM-based ranker significantly underperforms MoA, even when both have access to the same set of proposer responses. This is a critical result: it shows that **aggregation is not merely selection**. The aggregator model performs a non-trivial synthesis — extracting the best elements from multiple responses and combining them into a new, higher-quality output. The prompt template in Table 1 makes this explicit: "Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply."

**Generative fusion models require training.** Jiang et al. (2023) introduced GENFUSER, a model trained specifically to generate improved responses by fusing multiple candidate outputs. While this represents a step beyond simple ranking, it requires training a dedicated fusion model, introducing computational overhead and limiting flexibility — the fused model is tied to the specific set of models and distribution it was trained on. MoA achieves similar fusion behavior without any fine-tuning: "Our method does not require any fine-tuning and only utilizes the interface of prompting and generation of LLMs" (Section 2.2). This is a practical advantage because it means MoA can be applied immediately to any new LLM that becomes available, without retraining.

**Output probability fusion requires weight access.** Huang et al. (2024) proposed fusing model outputs by averaging their output probability distributions. This approach requires access to model logits or probabilities, which is not available for many API-based models (including GPT-4). MoA operates entirely at the text level, using only the generated responses, making it compatible with any model accessible through a standard chat interface.

**Multi-agent debate systems are expensive and may not outperform strong single agents.** Several works (Du et al., 2023; Liang et al., 2023; Chan et al., 2023; Chen et al., 2023a) explore having multiple LLMs discuss and reason through problems interactively, often with asymmetric roles (debater, judge) or weighted voting (ReConcile). However, Wang et al. (2024b) systematically compared multi-agent approaches and found that a single agent with a strong prompt including detailed demonstrations can achieve comparable response quality. This raises an uncomfortable question: are multi-agent systems adding value beyond what a well-prompted single model can do? MoA addresses this by showing that the iterative aggregation of diverse outputs provides gains beyond any individual model, including the strongest proposer.

**Cascading and cost-reduction methods don't improve quality.** FrugalGPT (Chen et al., 2023b) proposed using different models in a cascading manner primarily to reduce cost. The goal is efficiency, not quality improvement. MoA, in contrast, demonstrates that multi-model collaboration can simultaneously improve quality and, in the right configuration (MoA-Lite), be cost-effective.

### The Key Phenomenon: Collaborativeness of LLMs

The paper's motivating insight — and its most novel empirical finding — is what it terms the **collaborativeness of LLMs**. Figure 1 shows the AlpacaEval 2.0 LC win rates for six popular LLMs, comparing their standalone performance against their performance when provided with responses generated independently by those same models. Every model improves. Critically, "this improvement occurs even when the auxiliary responses provided by the other models are of lower quality than what an individual LLM could generate independently."

This is a non-obvious and important observation. One might intuitively expect that seeing worse responses would not help — that an LLM's output quality is bounded by the quality of its inputs. The collaborativeness phenomenon suggests otherwise: models can extract useful information, perspectives, or structural elements from weaker outputs and incorporate them into stronger responses. The paper frames this through two roles:

- **Proposers**: models that generate useful reference responses. A good proposer "may not necessarily produce responses with high scores by itself" but should "offer more context and diverse perspectives, ultimately contributing to better final responses when used by an aggregator."

- **Aggregators**: models proficient at synthesizing responses from others into a single high-quality output. An effective aggregator "should maintain or enhance output quality even when integrating inputs that are of lesser quality than its own."

This role decomposition (Section 2.1) is a conceptual contribution that provides a framework for understanding *why* multi-model collaboration can work and *how* to design such systems: select diverse, complementary proposers and a strong aggregator.

### How This Paper Positions Itself

The paper's central claim is that the successive aggregation of outputs from diverse LLMs, repeated across multiple layers, produces compounding quality improvements that exceed what any individual model or simple one-step aggregation can achieve. This is conceptually distinct from prior work in several ways:

- **Not just ensembling, but iterative synthesis.** Unlike ranking or one-step fusion, MoA constructs a layered architecture where the aggregated output from one layer becomes input to the next. This is inspired by the Mixture-of-Experts (MoE) architecture (Shazeer et al., 2017), but adapted to the model level — rather than having specialized sub-networks within a single model, MoA treats entire LLMs as "experts" and uses prompting rather than gating networks for coordination. The paper makes this analogy explicit in Section 2.3: "From a high-level perspective, our proposed MoA framework extends the MoE concept to the model level by operating at the model level rather than at the activation level."

- **Training-free, access-agnostic, model-agnostic.** MoA requires only text-in/text-out interaction with models. It does not need fine-tuning, weight access, probability distributions, or training data. This makes it immediately applicable to any combination of models, including proprietary ones accessed through APIs. The paper emphasizes this flexibility as a practical advantage: "it can be applied to the latest LLMs regardless of their size or architecture."

- **Empirically validated collaborativeness as a design principle.** Rather than assuming that combining models is beneficial, the paper provides evidence for the specific mechanism at work (Figure 4b shows positive Spearman correlation between BLEU similarity to proposer outputs and win rate, suggesting the aggregator is incorporating the best elements from proposals rather than selecting one or generating something entirely unrelated). This provides a mechanistic understanding that guides architecture design — for instance, the finding that model diversity improves performance (Table 3) directly motivates using heterogeneous proposers.

The paper positions MoA as a practical methodology for the current LLM landscape: there are many high-quality open-source models with different strengths, and MoA provides a way to combine them without expensive training. The consistent improvements across AlpacaEval 2.0, MT-Bench, and FLASK — benchmarks that evaluate different aspects of response quality — suggest the approach is robust rather than benchmark-specific.

## 3. Technical Approach

### 3.1 Reader Orientation (Approachable Technical Breakdown)

**What the system is:** MoA is a *prompt-only* pipeline that passes a user query through multiple large language models arranged in successive layers, where each model in layer $i+1$ sees every model's output from layer $i$ before generating its own.

**What problem it solves and the "shape" of the solution:** The problem is that no single LLM is best at everything — different models have complementary strengths that are wasted when you deploy only one. The solution is a *layered aggregation architecture*: diverse models independently generate candidate answers (proposers), a strong synthesis model merges them into a better answer (aggregator), and this process repeats across multiple layers so that each round of aggregation builds on the synthesized output of the previous round.

### 3.2 Big-Picture Architecture (Diagram in Words)

The MoA system has four major components connected in a feedforward pipeline:

1.  **User prompt ($x_1$)** — the initial instruction or question that enters the system.
2.  **Proposer agents (layers 1 through $l-1$)** — each layer contains $n$ LLMs. Every LLM in layer $i$ receives the prompt concatenated with *all* outputs from layer $i-1$ (or just the prompt for layer 1). Each generates its own independent response. The set of model outputs from layer $i$ is denoted $\\{A_{i,1}(x_i), A_{i,2}(x_i), \dots, A_{i,n}(x_i)\\}$, where $A_{i,j}$ is the $j$-th model in layer $i$ and $x_i$ is the input to that layer.
3.  **Aggregator agent (final layer $l$)** — a single LLM that receives the prompt plus all outputs from layer $l-1$ and produces the final system output. The paper uses Qwen1.5-110B-Chat as the default final aggregator (or GPT-4o for the "MoA w/ GPT-4o" variant).
4.  **Aggregate-and-Synthesize prompt template (Table 1)** — the fixed text template that wraps the proposers' outputs before feeding them to the next layer. It instructs the aggregator to "critically evaluate the information provided in these responses" and produce "a refined, accurate, and comprehensive reply" rather than simply copying.

Information flows strictly forward: prompt → layer 1 proposers (parallel) → concatenate all outputs → layer 2 proposers (parallel, each sees the full set) → concatenate all outputs → ... → final aggregator → single output. There is no backward communication, no iterative debate, and no weight modification — the entire system operates through text-in/text-out LLM calls.

### 3.3 Roadmap for the Deep Dive

- **First**, the formal MoA layer equation (Equation 1), which defines the input-output relationship for each layer and specifies how the Aggregate-and-Synthesize prompt instantiates the `$\oplus$` operator. This is the mathematical architecture.
- **Second**, the Aggregate-and-Synthesize prompt (Table 1), since it is the core mechanism by which an aggregator model synthesizes proposals rather than selecting among them. Understanding this prompt is essential to understanding why MoA outperforms ranking baselines.
- **Third**, the analogy to Mixture-of-Experts (Section 2.3), which provides the conceptual motivation for the layered architecture and explains the design choices — why multiple models, why layers, and why the gating/coordination is handled implicitly by the LLM rather than by a learned network.
- **Fourth**, the role specialization framework — proposers vs. aggregators — since this decomposition is used to guide model selection (Section 3.3, Table 4) and determines which models go where in the architecture.
- **Fifth**, the model selection strategy and system configurations (6-model MoA, MoA-Lite, MoA w/ GPT-4o), which instantiate the abstract architecture with concrete choices, including the naming convention, model set, layer counts, and aggregator assignments.
- **Sixth**, inference mechanics — temperature sampling, context window limitations, and how the final answer is extracted, since these operational details affect reproducibility.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems-and-empirical-analysis paper** whose core idea is that iteratively aggregating outputs from diverse LLMs, using only prompting, yields compounding quality improvements that exceed any individual model's capability.

---

#### The MoA Layer Equation

The paper formalizes the operation of a single MoA layer as:

$$y_i = \oplus_{j=1}^n [A_{i,j}(x_i)] + x_1, \quad x_{i+1} = y_i$$

where `$n$` is the number of agents in layer `$i$`, `$A_{i,j}$` is the `$j$`-th LLM in layer `$i$` (treated as a function from input text to output text), `$x_i$` is the concatenated input to layer `$i$` (comprising the original prompt plus all outputs from the previous layer), `$\oplus$` is the Aggregate-and-Synthesize operation (applied by a single aggregator LLM using the prompt in Table 1), `$+$` denotes text concatenation, `$x_1$` is the original user prompt, `$y_i$` is the output of layer `$i$` (and becomes `$x_{i+1}$` for the next layer), and `$l$` is the total number of layers.

**What it computes:** For each layer `$i$`, every agent `$A_{i,1}$` through `$A_{i,n}$` independently processes the layer input `$x_i$` and produces a response. These `$n$` responses are then passed to a designated aggregator model (which may be one of the `$A_{i,j}$` or a different model), along with the original user prompt `$x_1$`, and the aggregator generates a synthesized response `$y_i$` that becomes the input to the next layer. For the first layer, `$x_1$` is just the user prompt; for subsequent layers, `$x_i$` is the prompt concatenated with the previous layer's synthesized output. In the final layer `$l$`, the output of a single model `$A_{l,1}(x_l)$` is taken directly as the system's final answer (no further `$\oplus$` aggregation is performed).

**Why this form:** The key structural choices are the *parallel generation* within each layer (all `$n$` agents process the same input simultaneously, contributing independent perspectives), the *flat aggregation* across all `$n$` outputs (every agent in layer `$i+1$` sees every output from layer `$i$`, not just a selected subset), and the *residual connection* `$+ x_1$` (the original prompt is always prepended, preventing drift away from the user's intent). The residual connection is analogous to the residual connections in the Mixture-of-Experts formulation (Equation 2): `$y_i = \sum_{j=1}^n G_{i,j}(x_i) E_{i,j}(x_i) + x_i$`. This is not accidental — the MoE analogy is deliberate and is discussed below. The flat aggregation (`$\oplus_{j=1}^n$` rather than, say, pairwise debate or sequential refinement) is what distinguishes MoA from multi-agent debate systems: there is no back-and-forth between agents, only the aggregator's single synthesis step per layer. This makes MoA simpler to implement and cheaper in terms of total API calls than debate-based systems, which require multiple rounds of inter-agent communication.

A critical practical detail: the paper states that "we do not need to concatenate prompt and all model responses so only one LLM is needed to be used in the last layer." This means the final layer does not use the full `$\oplus$` operation with all `$n$` agents generating responses that then get aggregated — instead, a single model (the aggregator) directly generates the final output. The `$\oplus$` operation in intermediate layers may also be implemented by a single designated aggregator model rather than requiring an additional model separate from the proposers.

---

#### The Aggregate-and-Synthesize Prompt

The `$\oplus$` operator in the layer equation is instantiated not by code or learned weights, but by a carefully designed text prompt shown in Table 1 of the paper. This prompt is the *only* mechanism by which synthesis happens; there is no training, no probability fusion, and no hard selection logic. The prompt reads:

> "You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Responses from models: 1. [Model Response from $A_{i,1}$] 2. [Model Response from $A_{i,2}$] ... n. [Model Response from $A_{i,n}$]"

The prompt serves multiple functions simultaneously. First, it instructs the aggregator to *synthesize* rather than *select* ("should not simply replicate the given answers but should offer a refined... reply"). This is what distinguishes MoA from LLM-rankers: the empirical finding (Figure 4a) that MoA substantially outperforms ranking baselines validates that models can, and do, follow this synthesis instruction. Second, it primes the aggregator for *critical evaluation* ("recognizing that some of it may be biased or incorrect"), which encourages the model to identify and discard low-quality information rather than averaging across all inputs indiscriminately. Third, it provides a *structured format* (numbered list of responses) that makes it easy for the model to reference and distinguish between different proposals.

The design of this prompt reflects the paper's insight about collaborativeness: the aggregator can produce better output than any individual proposer because it sees multiple perspectives and can select the best elements from each. The prompt's emphasis on synthesis over replication is essential — without it, a strong aggregator might simply output its own preferred response, ignoring the auxiliary information, which would reduce the system to single-model performance (and indeed, the baseline of using the aggregator alone corresponds to this degenerate case).

---

#### The Mixture-of-Experts Analogy

Section 2.3 draws an explicit parallel between MoA and the Mixture-of-Experts (MoE) architecture (Shazeer et al., 2017). This analogy is not merely decorative — it provides the conceptual justification for the layered structure and the use of multiple parallel models.

A standard MoE layer is formalized as:

$$y_i = \sum_{j=1}^n G_{i,j}(x_i) E_{i,j}(x_i) + x_i$$

where `$G_{i,j}$` is the output of a gating network for expert `$j$` (a learned scalar weight produced by a small neural network), `$E_{i,j}$` is the function computed by expert network `$j$` (typically a feedforward sub-network), and `$+ x_i$` is the residual connection.

**What this computes:** the gating network examines the input `$x_i$` and produces a set of weights (summing to 1 via softmax) that determine how much each expert's output contributes to the final layer output. Experts that are more relevant to the current input receive higher weights. The output is a weighted sum of all expert outputs plus the residual, allowing the model to dynamically route different inputs to different expert combinations.

**Why this form matters for MoA:** the paper reinterprets this architecture at the *model level* rather than at the *activation level*. In MoA:
- **Each `$E_{i,j}$` is an entire LLM**, not just a feedforward sub-network. This means each "expert" is a full-fledged language model with its own training data, capabilities, and biases.
- **The gating function `$G_{i,j}$` is implicit in the aggregator's text synthesis**, not explicit as learned weights. The aggregator LLM, through its internal mechanisms, decides which aspects of which proposer outputs to emphasize, combine, or discard. The paper states that "we consolidate the roles of the gating network and expert networks using a LLM, as the intrinsic capacity of LLMs allows them to effectively regularize inputs by interpreting prompts and generating coherent outputs without needing external mechanisms for coordination."
- **The "weighted sum" `$\sum$` becomes the `$\oplus$` synthesis operation** — not a numeric sum, but a semantic integration.
- **The residual connection `$+ x_i$` becomes `$+ x_1$`** in the MoA equation — the original prompt is always concatenated to prevent the system from drifting away from the user's instruction across multiple layers of synthesis.

This analogy explains several design choices. First, *why multiple models?* Because MoE demonstrates that different experts can specialize in different aspects of the input, and combining their outputs through a learned gating mechanism yields better performance than a single monolithic network. MoA hypothesizes that the same principle applies at the LLM level: different models have different strengths, and a capable aggregator can implicitly learn to "route" to the right perspectives by synthesizing across all proposals. Second, *why layers?* Because MoE architectures are typically stacked in multiple layers, with each layer having its own set of experts — the depth allows for hierarchical feature extraction. In MoA, each layer of aggregation refines the output further, with the intermediate text serving as an explicit representation of the current state of synthesis. Third, *why not train a gating network?* Because LLMs can perform the gating function implicitly through their text generation capabilities. This is a practical advantage — it eliminates the need for training data, model modification, and access to internal representations, making MoA applicable to any off-the-shelf model.

---

#### Role Specialization: Proposers and Aggregators

The paper introduces a conceptual decomposition of LLM behavior in collaborative settings (Section 2.1) that guides model selection for the MoA architecture. This decomposition is not enforced by any mechanism — models are not trained or constrained to these roles — but rather describes empirical tendencies that inform which models to place where.

**Proposers** are defined as models that "excel at generating useful reference responses for use by other models." The key property of a good proposer is *not* that its standalone output is high-quality, but rather that its outputs provide *useful auxiliary information* to an aggregator. A proposer "may not necessarily produce responses with high scores by itself" but "should offer more context and diverse perspectives." This reframes model selection: when choosing proposers, prioritize models that contribute complementary information rather than just models with the highest standalone scores.

**Aggregators** are defined as models "proficient in synthesizing responses from other models into a single, high-quality output." The critical property: "an effective aggregator should maintain or enhance output quality even when integrating inputs that are of lesser quality than its own." This is the collaborativeness phenomenon in reverse — just as models improve when given others' outputs, a good aggregator can extract signal from noisy or suboptimal proposals.

**Empirical validation (Table 4):** The paper measures each model's performance in both roles. When evaluating as an aggregator, the model synthesizes outputs from all six proposers into a final response; when evaluating as a proposer, its output is one of six proposals fed to Qwen1.5-110B-Chat as aggregator. The results reveal specialization:
- **Versatile models:** Qwen1.5-110B-Chat (61.3% as aggregator, 56.7% as proposer), Qwen1.5-72B-Chat (59.3% as aggregator, 53.3% as proposer), and GPT-4o perform well in both roles.
- **Proposer specialists:** WizardLM 8x22B achieves 63.8% as a proposer (the highest) but only 52.9% as an aggregator — it generates excellent auxiliary information but struggles to synthesize others' outputs. LLaMA-3-70B-Instruct shows a qualitatively similar but less extreme pattern (60.6% as proposer, 45.0% as aggregator).
- **Neither role dominates for weaker aggregators:** dbrx-instruct (41.5% aggregator, 55.1% proposer) and Mixtral-8x22B-Instruct (48.4% aggregator, 54.8% proposer) are mediocre in both roles.

**Why this decomposition matters:** it informs the choice of the final aggregator (should be a model strong in the aggregator role) and the choice of proposers (should include specialist proposers like WizardLM even if they are not the strongest aggregators). The paper's default configuration uses Qwen1.5-110B-Chat — a model that performs well in both roles — as the final aggregator, and includes WizardLM as a proposer to benefit from its strong proposal quality. This strategic model selection, guided by the proposer/aggregator analysis, is what distinguishes MoA from a naive "include all available models" approach.

---

#### System Configurations: Default MoA, MoA-Lite, and MoA w/ GPT-4o

The paper defines three concrete instantiations of the abstract architecture, each representing a different point on the cost-performance tradeoff curve:

**Default MoA (6 proposers, 3 layers):**
- **Proposers:** Qwen1.5-110B-Chat, Qwen1.5-72B-Chat, WizardLM-8x22B, LLaMA-3-70B-Instruct, Mixtral-8x22B-v0.1, dbrx-instruct — six diverse open-source models spanning different model families, sizes (72B to 110B, plus the 8x22B MoE architectures), and training methodologies.
- **Layers:** 3 MoA layers, with the same set of 6 models used in each layer (models are reused across layers).
- **Final aggregator:** Qwen1.5-110B-Chat in the last layer produces the final output.
- **This is the configuration that achieves 65.1% LC win rate** on AlpacaEval 2.0.

**MoA w/ GPT-4o:**
- **Same proposer set and 3-layer structure as default MoA**, but with GPT-4o serving as the final aggregator instead of Qwen1.5-110B-Chat.
- **Achieves 65.7±0.7% LC win rate** (slightly higher than pure open-source MoA but requiring a proprietary model for the final synthesis step).
- **Purpose:** demonstrates that the strongest available aggregator further improves performance, validating the aggregator quality hypothesis.

**MoA-Lite (6 proposers, 2 layers):**
- **Same proposer set**, but only 2 MoA layers total.
- **Final aggregator:** Qwen1.5-72B-Chat (a smaller, cheaper model than the 110B variant).
- **Achieves 59.3% LC win rate**, which still surpasses GPT-4 Omni (57.5%) while being significantly cheaper.
- **Purpose:** demonstrates that even a lightweight configuration with fewer layers and a smaller aggregator can outperform the best individual models, establishing cost-effectiveness.

**Single-proposer variant:** The paper also explores a degenerate case where all `$n$` responses in a layer come from the same model (using temperature sampling for diversity). This is denoted "single-proposer" in Table 3. The multi-proposer variant consistently outperforms single-proposer at every value of `$n$`, with the gap widening as `$n$` increases (e.g., at `$n = 6$`, multi-proposer achieves 61.3% vs. 56.7% for single-proposer). This directly validates the importance of *model diversity*, not just *sample diversity*.

---

#### Inference Mechanics and Operational Details

**Temperature sampling:** The paper states that models are run with a temperature of 0.7 for the single-proposer experiments (Section 3.3, Table 3 discussion). For multi-proposer experiments where different models are used, the specific temperature settings per model are not explicitly provided, but the default inference endpoints are used ("all inferences were ran through Together Inference Endpoint"). Each model generates one response per layer per prompt — there is no majority voting or best-of-N sampling within a single model's contribution.

**Context window management:** The concatenation `$+ x_1$` means the original prompt is always included in every layer's input. As layers accumulate, the input to layer `$i$` includes the prompt plus the synthesized output from layer `$i-1$` (which itself may be long, since the aggregator produces a comprehensive response). The paper does not explicitly discuss context window constraints, but the intermediate outputs are single aggregated responses (not the full set of `$n$` proposer outputs), which keeps the context manageable. The proposers in intermediate layers receive the prompt plus the previous layer's aggregated output — they do NOT individually see all raw proposer outputs from previous layers, since `$x_{i+1} = y_i$` where `$y_i$` is the aggregated synthesis.

**Final answer extraction:** In the last layer, "we use the output of an LLM from the `$l$`-th layer (`$A_{l,1}(x_l)$`) as the final output and evaluate the metrics based on it." There is no additional aggregation, selection, or post-processing — the final aggregator's raw output is the system's answer.

**Reproducibility:** The paper reports results averaged over three runs with standard deviations for AlpacaEval 2.0 (e.g., "65.1±0.6%"). For MT-Bench, results include standard deviations for MoA variants (e.g., "9.25±0.10"). The multiple runs account for the stochasticity of temperature sampling in both the proposers and aggregators.

**Cost and FLOPs accounting (Section 3.4):** The cost analysis (Figure 5a) is based on API pricing — Together AI pricing for open-source models and OpenAI pricing for GPT-4 variants, both retrieved as of May 22, 2024. The tflops analysis (Figure 5b) uses "the sum over layers of the max number of tflops among proposers in each MoA layer" as the latency proxy, exploiting the fact that proposers within a layer run in parallel (so the layer's latency is determined by the slowest proposer, and layers execute sequentially). For GPT-4, the paper uses "the rumored size from the community of an 8x220B architecture" to estimate tflops since actual model sizes are undisclosed.

---

#### Evidence That Aggregation Is Synthesis, Not Selection

The paper provides two empirical results that characterize *what the aggregator is doing*:

**MoA vs. LLM-Ranker (Figure 4a):** An LLM-ranker baseline (using Qwen1.5-110B-Chat prompted to select the best among proposer outputs, using the prompt in Appendix Table 5) achieves substantially lower performance than MoA with the same aggregator model. At 3 layers, the LLM-ranker scores in the 40–45% LC win rate range (exact values not tabulated but visible in Figure 4a), while the MoA aggregator achieves approximately 65%. This 15–20 percentage point gap cannot be explained by model capability (it's the same model), so it must reflect the mechanism: synthesis outperforms selection.

**Spearman correlation with proposer outputs (Figure 4b):** Within each sample, the paper computes the Spearman rank correlation between (a) the text similarity (BLEU score) of each proposer's output to the aggregator's final output, and (b) the GPT-4 evaluator's preference score for each proposer's output. The positive correlation (ranging from approximately 0.05 to 0.25 depending on the aggregator model and layer) indicates that "the aggregator's response tends to incorporate the best proposed answers." In other words, the aggregator's output is textually more similar to proposer outputs that GPT-4 independently rates as high-quality. The correlation strengthens across layers (the "3rd aggregation" bars are generally higher than "1st aggregation"), suggesting that deeper synthesis produces outputs that more effectively capture the best elements from proposals. Alternative similarity metrics — Levenshtein similarity and TF-IDF (Appendix A, Figure 6) — show qualitatively identical patterns, confirming robustness to the choice of similarity measure.

This evidence is mechanistically important because it rules out two alternative hypotheses: (1) that the aggregator simply ignores the proposals and generates its own independent response (if so, there would be no correlation with proposer quality), and (2) that the aggregator simply selects the single best proposal (if so, performance would match the LLM-ranker, which it clearly does not). Instead, the aggregator appears to be performing a genuine synthesis — identifying high-quality elements from multiple proposers and combining them into a new response that is better than any individual contribution.

## 4. Key Insights and Innovations

### Innovation 1: Identifying and Systematically Exploiting the Inherent "Collaborativeness" of LLMs

The paper's most fundamental conceptual contribution is the identification and empirical characterization of what it terms the **collaborativeness of LLMs**: the observation that LLMs consistently generate higher-quality responses when given access to outputs from other models, even when those other models are individually less capable. This is not a method but rather a *diagnostic finding about the nature of LLM generation*—one that carries significant implications for how we think about model deployment and capability ceilings.

Before this work, the dominant assumption in multi-model systems was either that aggregating models requires explicit training (GENFUSER, Jiang et al., 2023), relies on probability-level fusion (Huang et al., 2024), or that the value of multi-agent systems comes from structured debate and role-playing (Du et al., 2023; Liang et al., 2023). The implicit belief was that a strong model shouldn't benefit from seeing weaker outputs—that the quality of auxiliary information places a bound on the quality of synthesis. Figure 1 overturns this assumption decisively: every model tested improves when shown other models' outputs, and the improvement does not depend on those outputs being of higher quality than what the model could generate on its own.

This reframes the problem from "how do we make models discuss and converge?" (the multi-agent debate framing) to "how do we extract the maximum signal from diverse auxiliary outputs through synthesis?" (the MoA framing). The collaborativeness phenomenon suggests that LLMs possess an implicit ability to perform *critical synthesis*—identifying valuable information, perspectives, and structural elements from heterogeneous inputs and incorporating them into a stronger output—without any training to do so. This is a capability-level discovery, not an engineering contribution.

**Significance beyond performance:** collaborativeness implies that the ceiling on what can be achieved with existing open-source models is higher than their individual benchmark scores suggest. The finding that open-source models combined through MoA can surpass GPT-4 Omni (65.1% vs. 57.5% LC win rate, Table 2a) is not just a leaderboard result—it demonstrates that the "capability gap" between open-source and proprietary models is substantially smaller than standalone evaluations indicate, because open-source models can collaborate. This has economic and strategic implications: organizations with access only to open-source models can potentially achieve proprietary-level quality through architectural composition rather than model scaling.

**Is it fundamental or incremental?** This is a *fundamental empirical discovery*. The collaborativeness phenomenon had not been systematically documented or characterized prior to this work. The paper provides both existence proof (Figure 1: all models improve) and mechanistic characterization (Figure 4b: aggregators incorporate the best proposer outputs; Table 3: diversity of proposers matters more than number of samples from a single model). This transforms collaborativeness from an anecdotal observation into a quantifiable, exploitable property of current LLMs.

---

### Innovation 2: Reframing Multi-LLM Systems Through the Proposer-Aggregator Decomposition

The paper introduces a conceptual framework that decomposes collaborative LLM behavior into two distinct roles—**proposers** and **aggregators**—and uses this decomposition to guide system design. This is not merely a taxonomy; it is a *design principle* that changes how one selects and arranges models in a multi-LLM system.

Prior work on model ensembling and multi-agent systems treated models largely as interchangeable participants. Whether in symmetric debate (Du et al., 2023), pairwise ranking (PAIRRANKER, Jiang et al., 2023), or router-based selection (Wang et al., 2024a), the assumption was that any model could fill any role, and the challenge was in the coordination mechanism. The proposer-aggregator framework challenges this by showing that **models exhibit specialized strengths in collaborative settings that are not predictable from their standalone performance**.

The evidence for this specialization is Table 4: WizardLM 8x22B achieves 63.8% as a proposer (the highest among all tested models) but only 52.9% as an aggregator—a gap of nearly 11 percentage points relative to its performance in the complementary role. Conversely, Qwen1.5-110B-Chat performs well in both roles but is not the single best proposer. These are not subtle differences; they represent qualitatively different model behaviors that would be invisible in standard benchmark evaluations.

**Why this is conceptually distinctive:** the proposer-aggregator decomposition redefines what "good model" means in a collaborative context. A model with mediocre standalone performance might be exceptionally valuable as a proposer if it provides diverse, complementary perspectives. A model with strong standalone performance might be a poor aggregator if it cannot effectively incorporate others' outputs. This shifts model selection from a univariate optimization (highest benchmark score) to a portfolio optimization problem (selecting complementary proposers and a strong aggregator), which is a fundamentally different approach to system construction.

The paper validates this framework empirically by showing that the choice of aggregator dramatically affects final performance (Figure 4a: aggregator curves span a range of roughly 20–25 percentage points in LC win rate even with identical proposers) and that proposer diversity matters independently of proposer quality (Table 3: multiple-proposer outperforms single-proposer at every `n`). These results establish the proposer-aggregator decomposition not as a theoretical abstraction but as an empirically grounded design framework.

**Is it fundamental or incremental?** This is a *fundamental conceptual contribution*. While the idea of specialized roles in multi-agent systems is not new in AI generally, the proposer-aggregator decomposition is specific to the LLM collaboration setting and is derived from empirical characterization rather than imposed a priori. It provides a vocabulary and an evaluation methodology (Table 4-style role assessment) that future work can build on, and it explains why naive "include all models" approaches may underperform a strategically assembled MoA system.

---

### Innovation 3: Demonstrating That Iterative Layered Aggregation Produces Compounding Gains Without Training

The MoA architecture itself—multiple layers of parallel generation followed by synthesis, repeated iteratively—embodies a specific hypothesis: that **the benefits of multi-model collaboration compound across successive rounds of aggregation**, and that this compounding can be achieved purely through prompting without any fine-tuning or weight modification. This hypothesis is not obvious a priori. One could imagine diminishing returns (each additional layer adds less value as the output converges), saturation (the aggregator can only do so much with a given set of proposals), or even degradation (errors compound across layers).

The evidence for compounding is Figure 4a: for most aggregator models, the LC win rate increases monotonically from Layer 1 through Layer 3, with significant jumps at each step. For Qwen1.5-110B-Chat as aggregator, the progression from approximately 47% (Layer 1) to approximately 61% (Layer 2) to approximately 65% (Layer 3) shows clearly non-diminishing returns—the second layer adds roughly 14 percentage points, which is a larger absolute improvement than many individual model upgrades would provide.

**What distinguishes this from prior multi-model approaches:** previous systems either operated in a single round (ranking, one-step fusion) or used multi-round debate/conversation (which introduces coordination complexity). MoA's layered architecture is neither—it is feedforward (no back-and-forth between agents within a layer), yet iterative across layers. This design choice reflects a specific insight: **synthesis quality can be improved by re-synthesizing the synthesis**, analogous to how deep networks benefit from stacked layers. The MoE analogy in Section 2.3 makes this explicit: just as stacked MoE layers enable hierarchical feature extraction, stacked MoA layers enable hierarchical response refinement.

The paper's comparison to the LLM-Ranker baseline (Figure 4a) is the critical control experiment that isolates the value of the layered architecture. The ranker has access to the same proposals but operates in a single step; its substantially lower performance demonstrates that the iterative layering is not merely providing the aggregator with more information (it sees the same proposals) but rather enabling a qualitatively different process—synthesis followed by re-synthesis, where each layer's output captures higher-level structure and integration.

**The "no training" property is an architectural innovation in itself.** By operating entirely through prompting, MoA is immediately applicable to any combination of models, including models accessed only through APIs. This is not an incremental convenience—it fundamentally changes the deployment economics. There is no model-specific training, no calibration on held-out data, and no dependency on model internals. When a new, stronger open-source model is released, it can be incorporated into an existing MoA pipeline immediately by adding it as a proposer and, if appropriate, testing it as an aggregator. This "plug-and-play" property is architecturally enforced rather than being an implementation detail.

**Is it fundamental or incremental?** This is a *fundamental architectural contribution*. The specific combination—(1) layered, (2) feedforward, (3) training-free aggregation using only prompting, (4) with demonstrated compounding gains—is novel as an integrated design. Individual elements (ensembling, multi-agent systems, iterative refinement) existed before, but their synthesis into the MoA architecture, validated by the compounding performance curves in Figure 4a, represents a new point in the design space of multi-LLM systems.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary evaluation uses **AlpacaEval 2.0** (Dubois et al., 2024), consisting of 805 instructions representative of real use cases. Each model's response is compared head-to-head against GPT-4 (gpt-4-1106-preview) by a GPT-4-based evaluator that determines the likelihood of preferring the evaluated model's response. The paper also evaluates on **MT-Bench** (Zheng et al., 2023), which uses GPT-4 to assign a numerical score to model answers across multi-turn conversations, and **FLASK** (Ye et al., 2023), which provides 12 fine-grained skill-specific scores (robustness, correctness, efficiency, factuality, commonsense, comprehension, insightfulness, completeness, metacognition, readability, conciseness, harmlessness). An additional evaluation on the **MATH** dataset (Hendrycks et al., 2021) appears in Appendix D.

- **Base model(s).** The default MoA configuration uses six open-source LLMs as proposers: **Qwen1.5-110B-Chat**, **Qwen1.5-72B-Chat** (Bai et al., 2023), **WizardLM-8x22B** (Xu et al., 2023a), **LLaMA-3-70B-Instruct** (Touvron et al., 2023b), **Mixtral-8x22B-v0.1** (Jiang et al., 2024), and **dbrx-instruct** (The Mosaic Research Team, 2024). These span different model families, scales (72B to 110B parameters, plus the 8x22B Mixture-of-Experts architectures), and training methodologies, providing the diversity that the paper's analysis shows is critical for MoA performance. For the final aggregator, the default is Qwen1.5-110B-Chat; MoA-Lite uses Qwen1.5-72B-Chat; and MoA w/ GPT-4o uses GPT-4o as the final aggregator. For single-model baselines, the paper evaluates each of these six models independently, plus GPT-4 Omni (05/13), GPT-4 Turbo (04/09), GPT-4 Preview (11/06), and GPT-4 (03/14 and 06/13).

- **Metrics.** The primary metric is **Length-Controlled (LC) win rate** on AlpacaEval 2.0, which measures the probability that the GPT-4 evaluator prefers the evaluated model's response over the GPT-4 (1106-preview) reference response, with length bias neutralized. The paper notes this metric "tracks closely with human preferences, achieving a Spearman correlation of 0.98 with actual human evaluations" (Dubois et al., 2024). Raw (non-length-controlled) win rates are also reported. For MT-Bench, the metric is the GPT-4 assigned score on a 1–10 scale, reported as an average across all turns and separately for first and second turns. For FLASK, 12 skill-specific scores are reported on a 1–5 scale. For MATH, the metric is accuracy (the fraction of problems with the correct final answer).

- **Baselines.** The paper compares against several categories: (1) **Individual model performance**: each of the six proposer models evaluated standalone, plus GPT-4 variants (GPT-4 Omni, GPT-4 Turbo, GPT-4 Preview) as reported on the official AlpacaEval 2.0 and MT-Bench leaderboards — the paper re-ran MT-Bench scores itself to obtain turn-based breakdowns; (2) **LLM-Ranker**: an approach where the aggregator model (Qwen1.5-110B-Chat) is prompted to select the best single response from the set of proposer outputs rather than synthesizing a new one, using the prompt template in Appendix Table 5; (3) **Single-proposer MoA**: a variant where all $n$ responses in each MoA layer are generated by the *same* model (using temperature 0.7 for diversity) rather than $n$ different models, tested at $n = 2, 3, 6$ (Table 3); (4) **Larger proprietary models** as standalone baselines: GPT-4 Omni (57.5% LC win rate), GPT-4 Turbo (55.0%), and GPT-4 Preview (50.0%) on AlpacaEval 2.0.

- **Generation budget / compute accounting.** The paper measures compute in two ways. First, **cost** (Figure 5a): calculated from API pricing as of May 22, 2024 — Together AI pricing for open-source models, OpenAI pricing for GPT-4 variants. This represents the monetary cost per AlpacaEval instance. Second, **tflops** (Figure 5b): used as a latency proxy, calculated as "the sum over layers of the max number of tflops among proposers in each MoA layer," exploiting that proposers within a layer run in parallel (so the layer's latency is bounded by its slowest model) and layers execute sequentially. For GPT-4, where actual model size is undisclosed, the paper uses "the rumored size from the community of an 8x220B architecture" to estimate tflops. The number of model calls per MoA instance is $n \times (l-1) + 1$ (for $l$ layers with $n$ proposers each, plus one final aggregator call), but the latency-critical path is $l$ sequential layers. No training compute is involved since MoA is training-free.

- **Cross-validation / statistical protocol.** For AlpacaEval 2.0, the paper reports results "averaged over three runs" to account for the stochasticity of temperature sampling, with standard deviations provided (e.g., "65.1±0.6%"). For MT-Bench, standard deviations are reported for MoA variants (e.g., "9.25±0.10"). The paper states they "ran all the MT-Bench scores ourselves to get turn-based scores" (Table 2b note), indicating that baseline MT-Bench numbers were reproduced rather than taken from leaderboards to ensure consistent evaluation conditions. The AlpacaEval 2.0 evaluations use the standard GPT-4 evaluator with the official reference model (gpt-4-1106-preview) and length-controlled debiasing, following the benchmark's established protocol. There is no separate train/validation/test split for MoA since no training occurs — all model selection (which models to use as proposers and aggregators) is based on publicly available benchmark performance and the proposer/aggregator role analysis in Table 4, not on hyperparameter tuning against a held-out set.

---

### Main Quantitative Results

#### AlpacaEval 2.0 Results

The headline result appears in **Table 2a**: MoA with only open-source models achieves a **65.1% LC win rate**, compared to 57.5% for GPT-4 Omni — an absolute improvement of 7.6 percentage points and a roughly 13% relative improvement. MoA w/ GPT-4o achieves **65.7%** (slightly higher but requiring a proprietary aggregator). Even the cost-optimized MoA-Lite achieves **59.3%**, which is 1.8 points above GPT-4 Omni despite using only 2 layers and a smaller aggregator (Qwen1.5-72B-Chat).

Compared to the individual open-source models that compose the MoA system, the gains are dramatic: Qwen1.5-110B-Chat (the strongest individual open-source model tested) achieves only 43.9% LC win rate standalone, meaning MoA improves over its best constituent model by 21.2 percentage points. This is not simply a case of the aggregator being a strong model — Qwen1.5-110B-Chat *is* the aggregator, yet MoA more than doubles the win rate it achieves on its own. Similarly, LLaMA-3-70B-Instruct achieves 34.4% standalone, Mixtral 8x22B achieves 30.9%, and WizardLM 8x22B achieves 51.3% — none approaches the MoA system performance.

The raw (non-length-controlled) win rates tell an interesting complementary story: MoA achieves a 59.8% raw win rate, which is *lower* than WizardLM 8x22B's standalone raw win rate of 62.3%. The fact that MoA's LC win rate (65.1%) is substantially higher than WizardLM's LC win rate (51.3%) while the raw win rates show the opposite pattern highlights the importance of the length-controlled metric — WizardLM tends to produce longer outputs that are favored by the biased evaluator, while MoA's outputs are more concise and score higher when length bias is removed. The MoA w/ GPT-4o variant has an even more extreme divergence (65.7% LC vs. 78.7% raw), suggesting GPT-4o's synthesis produces particularly verbose outputs.

#### MT-Bench Results

**Table 2b** shows that on MT-Bench, improvements are more incremental: MoA achieves a score of **9.25±0.10** compared to GPT-4 Omni's 9.19 and GPT-4 Turbo's 9.31. MoA w/ GPT-4o achieves **9.40±0.06**, the highest score in the table. The best open-source standalone model, Qwen1.5-110B-Chat, scores 8.96, meaning MoA improves by 0.29 points — significant but far smaller than the AlpacaEval gains.

The paper interprets this compressed margin correctly: "this is understandable given that current models already perform exceptionally well on this benchmark, as a single model alone can achieve scores greater than 9 out of 10." The turn-based breakdown is revealing: MoA's strongest advantage appears in the first turn (9.44 vs. GPT-4 Omni's 9.31), with the second turn showing a slight disadvantage (9.07 vs. 9.07 for GPT-4 Omni — essentially tied). This suggests MoA's multi-model synthesis particularly benefits single-turn instruction following, while the two-turn scores (where the model must maintain context from the previous exchange) show less differentiation. MoA-Lite achieves 9.18±0.09, which sits between GPT-4 Preview (9.20) and Qwen1.5-110B-Chat (8.96) — still a meaningful improvement over its constituent models but without the dramatic margin seen on AlpacaEval.

#### FLASK Results

**Figure 3** provides the most granular analysis of where MoA's gains materialize. On a 1–5 scale across 12 skill dimensions, MoA (using Qwen1.5-110B-Chat as aggregator) outperforms the same model standalone (Qwen1.5-110B-Chat) in **robustness, correctness, efficiency, factuality, commonsense, insightfulness, and completeness** — 7 of the 12 dimensions. It ties or slightly trails on comprehensiveness and readability. The one dimension where MoA notably underperforms is **conciseness**: "the model produced outputs that were marginally more verbose," consistent with the raw vs. LC win rate divergence on AlpacaEval where MoA's raw win rate was lower than some individual models.

Compared to GPT-4 Omni, MoA outperforms on **correctness, factuality, insightfulness, completeness, and metacognition** — dimensions related to accurate and thorough reasoning — while GPT-4 Omni maintains advantages on robustness, efficiency, commonsense, comprehension, readability, and conciseness. The dimensions of MoA advantage align with what one would expect from a multi-model synthesis approach: factuality and correctness improve because contradictory claims across proposers can be identified and reconciled; completeness improves because diverse perspectives ensure coverage of different aspects; insightfulness improves because novel connections can be drawn across proposals. The dimensions of MoA disadvantage (conciseness, readability) reflect the aggregator producing comprehensive outputs that may be harder to follow than a model generating a focused single response.

#### MoA vs. LLM-Ranker: Evidence for Synthesis Over Selection

**Figure 4a** provides what is arguably the most mechanistically important result in the paper. The LLM-Ranker baseline — which uses the same Qwen1.5-110B-Chat as the aggregator but prompted to *select* rather than *synthesize* — achieves dramatically lower performance. At Layer 1, the LLM-Ranker sits around the mid-40s in LC win rate (exact values are visible in the figure but not tabulated), while the MoA aggregator (same model, same layer, same proposals) achieves approximately 47–50%. The gap widens with layers: at Layer 3, the LLM-Ranker essentially plateaus (it cannot improve beyond selecting the best proposal), while MoA continues climbing to approximately 65%.

This is definitive evidence that the aggregator is not simply picking the best response. If it were, the LLM-Ranker (explicitly optimized for selection) would perform at least as well. The 15–20 point gap at 3 layers means the aggregator is generating something that is systematically better than the best individual proposal — it is performing genuine synthesis.

The Spearman correlation analysis in **Figure 4b** reinforces this interpretation. For each aggregator model, the correlation between BLEU similarity (aggregator vs. proposer) and GPT-4 preference (proposer vs. reference) is positive, ranging from approximately 0.05 to 0.30 depending on model and layer. This means the aggregator's final output is textually more similar to proposals that are independently rated as higher quality. The correlation strengthens across layers (the "3rd aggregation" bars are generally higher than "1st aggregation"), suggesting deeper synthesis better captures the best elements. The paper reproduces this result with Levenshtein and TF-IDF similarity in Appendix A (Figure 6), confirming robustness to the similarity metric.

#### Effect of Model Diversity and Number of Proposers

**Table 3** isolates the contribution of proposer diversity. Comparing "multiple-proposer" (6 different models) vs. "single-proposer" (6 independent samples from Qwen1.5-110B-Chat with temperature 0.7, the best individual aggregator model used as proposer) at each value of $n$:

- At $n = 6$: Multiple-proposer achieves **61.3%** vs. single-proposer's **56.7%** — a 4.6-point gap attributable purely to model diversity rather than sample diversity, since both see 6 responses.
- At $n = 3$: 58.0% vs. 56.1% — the gap narrows to 1.9 points.
- At $n = 2$: 58.8% vs. 54.5% — the gap widens again to 4.3 points.
- At $n = 1$: Both achieve 47.8% (degenerate case — no auxiliary information).
- Performance increases monotonically with $n$ in the multiple-proposer setting (47.8% → 58.0% → 58.8% → 61.3%), while the single-proposer setting shows a surprising non-monotonicity (47.8% → 56.1% → 54.5% → 56.7%), with $n=3$ underperforming $n=2$.

The key insight: having more proposals helps, but having proposals from *different models* helps significantly more. The 4.6-point gap at $n=6$ is comparable to the improvement from going from 2 to 6 diverse models (58.8% → 61.3%, a 2.5-point gain), suggesting that proposer diversity is at least as impactful as proposer count.

#### Role Specialization: Proposers vs. Aggregators

**Table 4** quantifies model specialization across the two collaborative roles. For the aggregator evaluation, each model serves as the final aggregator synthesizing responses from all six proposers (using 2 MoA layers). For the proposer evaluation, each model's outputs are included as one of six proposals fed to Qwen1.5-110B-Chat as the aggregator. The results, in descending aggregator performance:

- **Qwen1.5-110B-Chat**: 61.3% as aggregator, 56.7% as proposer — strong in both roles, with an aggregator advantage.
- **Qwen1.5-72B-Chat**: 59.3% as aggregator, 53.3% as proposer — similar profile, at smaller scale.
- **WizardLM 8x22B**: 52.9% as aggregator, **63.8%** as proposer — the best proposer but a mediocre aggregator (10.9-point gap in favor of proposer role), the clearest example of role specialization.
- **Mixtral-8x22B-Instruct**: 48.4% as aggregator, 54.8% as proposer — modest proposer advantage.
- **LLaMA-3-70B-Instruct**: 45.0% as aggregator, 60.6% as proposer — even more extreme proposer specialization than WizardLM (15.6-point gap), highlighted as a surprising finding since LLaMA-3-70B is generally considered a strong all-around model.
- **dbrx-instruct**: 41.5% as aggregator, 55.1% as proposer — weaker in both roles, but the proposer advantage is still present.

This pattern — where LLaMA-3-70B and WizardLM 8x22B are both much stronger as proposers than as aggregators — is striking and non-obvious. It means that including these models in the proposer set is wise, but using them as the final aggregator would leave significant performance on the table. The paper's design choice to use Qwen1.5-110B-Chat as the aggregator is directly motivated by this analysis.

#### Layer-Wise Compounding in the 6-Model MoA

**Figure 4a** also shows the layer-wise trajectory for each aggregator model. For Qwen1.5-110B-Chat (the default aggregator), the LC win rate climbs from approximately 47% at Layer 1 to approximately 61% at Layer 2 to approximately 65% at Layer 3. The second layer adds roughly 14 percentage points; the third layer adds roughly 4. The first aggregation provides the largest single jump, but the second aggregation continues to add significant value — the gain from Layer 2 to Layer 3 (+4 points) is roughly the same magnitude as the gap between GPT-4 Turbo (55.0%) and GPT-4 Omni (57.5%), two models separated by a full generation of development.

Notably, the curves for different aggregator models show different saturation patterns. GPT-4o as aggregator plateaus quickly after Layer 2 (the curve flattens), suggesting it extracts nearly all available signal in two layers. Qwen1.5-110B-Chat continues to benefit from a third layer. The LLM-Ranker is essentially flat after Layer 1 — it cannot improve no matter how many additional layers of proposals it sees because it is limited to selection.

#### MATH Results

**Table 8** in Appendix D demonstrates that MoA also improves mathematical reasoning. Starting from Layer 1 accuracy (which reflects the aggregator synthesizing the initial set of proposer outputs), accuracy improves consistently across aggregators and layers:

- **Qwen1.5-110B-Chat as aggregator**: 0.500 → 0.570 → 0.576 (from Layer 1 to Layer 3).
- **LLaMA-3-70B-Instruct as aggregator**: 0.456 → **0.584** → 0.578 — notably, LLaMA-3-70B achieves the highest layer-2 accuracy despite being the weakest aggregator on AlpacaEval (Table 4), suggesting domain-specific aggregator effectiveness.
- **WizardLM 8x22B as aggregator**: 0.544 → 0.574 → **0.580** — the highest layer-3 accuracy, despite WizardLM's weakness as an aggregator on AlpacaEval (52.9%).
- **Mixtral-8x22B**: Shows the most dramatic jump from Layer 1 to Layer 2 (0.282 → 0.534, a 25-point gain), suggesting that Mixtral is a poor standalone reasoner but effectively synthesizes others' reasoning when acting as an aggregator.
- **dbrx-instruct**: 0.314 → 0.456 → 0.522 — the weakest aggregator overall, consistent with its AlpacaEval aggregator performance.

The paper notes that MoA is "complementary to existing reasoning techniques such as Chain of Thought and Self-consistency," suggesting these results could be further improved by combining MoA with those methods.

#### Cost and Compute Efficiency Analysis

**Figure 5a** plots LC win rate against monetary cost per AlpacaEval instance, revealing a Pareto frontier. GPT-4o achieves approximately 57.5% at a cost of roughly $0.002–0.003 per instance (reading from the figure). MoA achieves approximately 65% at a cost of roughly $0.025–0.030 — substantially more expensive per query but far higher quality. MoA-Lite achieves approximately 59% at a cost comparable to GPT-4o, meaning it matches GPT-4o's cost while providing higher quality. GPT-4 Turbo sits at approximately 55% at a cost of roughly $0.008–0.010 — MoA-Lite is both cheaper and better, or comparably priced and better by roughly 4 percentage points.

The paper's specific claim is that MoA-Lite "outperforms GPT-4 Turbo by approximately 4% while being more than twice as cost-effective." Reading from Figure 5a: GPT-4 Turbo costs roughly $0.008–0.010 for 55% LC win rate, while MoA-Lite achieves 59% at roughly $0.003–0.005, which is indeed better performance at less than half the cost per query.

**Figure 5b** presents the tflops vs. LC win rate tradeoff, using the maximum-tflops-per-layer as a latency proxy. The Pareto frontier is similar: MoA achieves the highest quality but requires the most tflops (roughly 300–350 tflops, reading from the figure), MoA-Lite achieves high quality at moderate tflops (roughly 150–200), and GPT-4 variants sit along the frontier at different quality levels. The multi-proposer MoA configurations universally dominate their single-proposer counterparts at equivalent tflops, meaning the diversity benefit is not merely from using more computation — it's from using computation across different models.

---

### Ablation Studies and Robustness Checks

- **Number of proposers ($n$)**: Increasing $n$ monotonically improves performance in the multi-proposer setting (Table 3: $n=1$: 47.8%, $n=3$: 58.0%, $n=6$: 61.3%), while the single-proposer setting shows non-monotonic behavior ($n=1$: 47.8%, $n=2$: 54.5%, $n=3$: 56.1%, $n=6$: 56.7%). The non-monotonicity at $n=3$ for single-proposer (where 54.5% drops from $n=2$'s 54.5%? — actually $n=2$ achieves 54.5% and $n=3$ achieves 56.1%, so it does increase; the text above in Section 3.3 says "n = 3: 58.0% vs. 56.1%" and "n = 2: 58.8% vs. 54.5%", which means multi-proposer $n=2$ is 58.8% and multi-proposer $n=3$ is 58.0%, showing a slight dip — this is the non-monotonicity referred to) suggests that the value of additional proposals may depend on their quality and diversity in complex ways.

- **Choice of aggregator model**: The aggregator model's quality dramatically affects final MoA performance. Figure 4a shows that with identical proposers and layer count, aggregator choice produces a roughly 25-point spread in LC win rate (from dbrx-instruct's approximately 41% to GPT-4o's approximately 65% at Layer 3). This is the single most impactful design choice in the MoA system — larger than the number of layers or the number of proposers. The rank ordering of aggregator effectiveness (GPT-4o > Qwen1.5-110B-Chat > Qwen1.5-72B-Chat > WizardLM 8x22B > Mixtral-8x22B > LLaMA-3-70B > dbrx-instruct) does not perfectly track standalone benchmark performance (LLaMA-3-70B is a strong standalone model but a poor aggregator), validating the paper's claim that aggregator capability is a distinct skill that should be evaluated specifically.

- **Number of MoA layers ($L$)**: Figure 4a shows that performance increases with layers for all aggregators, but with diminishing returns. The jump from Layer 1 to Layer 2 is substantially larger than from Layer 2 to Layer 3 for all models. GPT-4o appears to saturate after Layer 2 (its curve flattens). The LLM-Ranker does not benefit from additional layers at all, confirming that layer-wise compounding is a property of synthesis, not of having more proposals to choose from. The paper does not test beyond 4 layers, so the saturation point is not fully characterized.

- **Model diversity (multiple-proposer vs. single-proposer)**: At every $n$ tested (2, 3, 6), the multiple-proposer configuration outperforms the single-proposer configuration using the best available model (Qwen1.5-110B-Chat, which is also the aggregator). The gap widens with $n$ (4.3 points at $n=2$, 1.9 points at $n=3$, 4.6 points at $n=6$; Table 3). This is the most direct evidence that model diversity — having genuinely different model families, training procedures, and capabilities in the proposer set — contributes value beyond simply having more samples from a strong model. If the best single model's samples were sufficient, the single-proposer configuration would asymptotically approach the multi-proposer performance as $n$ grows, but it does not.

- **Similarity metric for correlation analysis**: The Spearman correlation between proposer-aggregator similarity and proposer quality (Figure 4b, Figure 6) is positive regardless of whether similarity is measured by BLEU (3-gram, 4-gram, 5-gram), TF-IDF, or Levenshtein distance. The magnitude varies slightly (BLEU correlations range from ~0.05 to ~0.30; TF-IDF and Levenshtein show similar patterns per Figure 6), but the qualitative finding — "the aggregator's response incorporates the best proposed answers" — is robust to the choice of text similarity metric.

- **MATH task generalization**: Appendix D (Table 8) demonstrates that MoA benefits are not limited to instruction-following benchmarks. Accuracy on MATH improves consistently from Layer 1 to Layer 3 across all six aggregator models tested. The magnitude of improvement varies by aggregator (ranging from a ~5-point gain for Qwen1.5-72B-Chat to a ~27-point gain for Mixtral-8x22B from Layer 1 to Layer 3), but the direction is uniformly positive. This is an important robustness check because MATH requires different capabilities (mathematical reasoning with verifiable ground truth) than AlpacaEval and MT-Bench (open-ended instruction following evaluated by LLM-as-judge). The fact that MoA improves both suggests the mechanism — diverse proposers providing complementary reasoning approaches, aggregator synthesizing the correct elements — generalizes across task types.

- **Cost-performance Pareto front**: Multiple MoA configurations lie on the Pareto frontier (Figure 5a, 5b), meaning there is no configuration that achieves higher quality at the same or lower cost. This is not a foregone conclusion — multi-model approaches could have been strictly dominated by a combination of single models and simply-ensembled approaches. The fact that MoA-Lite in particular (2 layers, Qwen1.5-72B-Chat aggregator) achieves both lower cost and higher quality than GPT-4 Turbo is a specific, non-trivial finding about the practical viability of the approach.

---

### Critical Assessment

#### Do the experiments support the claim that "MoA achieves state-of-the-art performance"?

**Yes, but with important scope limitations.** The claim that MoA achieves 65.1% on AlpacaEval 2.0, surpassing GPT-4 Omni's 57.5%, is directly supported by Table 2a. The margin is substantial (7.6 percentage points) and robust to multiple runs (65.1±0.6%). On MT-Bench, MoA scores 9.25, which surpasses GPT-4 Omni (9.19) and approaches GPT-4 Turbo (9.31). On FLASK, MoA outperforms GPT-4 Omni on 5 of 12 dimensions. The paper's self-characterization as achieving "state-of-the-art" on these specific benchmarks is accurate.

**However**, the comparison to "GPT-4 Omni" specifically uses the **May 13, 2024** version (explicitly noted in Table 2a). This is a snapshot evaluation — GPT-4 Omni received updates after this date that may have improved its performance. More broadly, the MoA configuration at 65.1% requires calling 6 models across 3 layers (6 + 6 + 1 = 13 model calls per instance) versus GPT-4 Omni's single call. A fair "quality per unit compute" comparison would reveal that MoA is dramatically more expensive per query in exchange for its quality advantage. The paper acknowledges this implicitly through the MoA-Lite and cost analysis sections but frames the headline result as a pure quality comparison.

#### Do the experiments support the claim that "collaborativeness" is a genuine and widespread phenomenon?

**Strongly supported, with a minor confound.** Figure 1 shows that all six tested models improve when shown auxiliary outputs from other models, and the paper explicitly notes that "this improvement occurs even when the auxiliary responses provided by the other models are of lower quality than what an individual LLM could generate independently." This is the key evidence for collaborativeness.

**One interpretive concern:** the improvement shown in Figure 1 is measured by the same LC win rate metric used throughout the paper, which compares against GPT-4 (1106-preview). An alternative interpretation is that seeing other models' outputs makes the evaluated model's *own* output more similar to what GPT-4 would generate (since the auxiliary responses were themselves generated by models aligned with similar preferences), and the LC evaluator (itself a GPT-4 variant) favors GPT-4-like responses. This would mean the improvement reflects evaluator-alignment rather than genuine quality improvement. The paper does not fully disentangle this possibility, though the FLASK results (which use fine-grained skill evaluations) and the MATH results (which use ground-truth accuracy) provide some evidence against a purely evaluator-bias explanation.

#### Do the experiments support the claim that iterative layering produces "compounding gains"?

**Partially supported, with clear diminishing returns.** Figure 4a shows improvements from Layer 1 → Layer 2 → Layer 3 for most aggregators, which is "compounding" in the sense of sequential additive gains. However, the second-layer gain (~14 percentage points for Qwen1.5-110B-Chat) is substantially larger than the third-layer gain (~4 percentage points). This is diminishing returns, not compounding in the exponential sense. The paper's language of "compounding" is somewhat strong given the data; "diminishing but positive returns to depth" would be more precise.

Additionally, the paper does not test whether a single-layer system with more proposers could achieve comparable performance to a multi-layer system at equivalent total proposal count. For example, 6 proposers × 3 layers provides the aggregator with 6 distinct responses at each layer, but the input to later layers is always the *synthesized* output from the previous layer, not the raw proposals. An alternative would be a single wide layer with 18 proposers. Without this ablation, it is unclear whether the layering provides value beyond simply increasing the number of auxiliary responses.

#### Do the experiments support the claim that "MoA-Lite can match GPT-4o's cost while achieving higher quality" and is "2× more cost-effective than GPT-4 Turbo"?

**Supported, with a pricing-date caveat.** Figure 5a shows MoA-Lite achieving approximately 59% LC win rate at roughly the same cost as GPT-4o's 57.5%, and MoA-Lite achieving 59% at roughly $0.003–0.005 per instance compared to GPT-4 Turbo's 55% at $0.008–0.010 per instance. This satisfies "higher quality at comparable cost" and "better quality at less than half the cost," respectively.

**The caveat:** pricing is snapshot-dependent. The paper notes "pricing data was retrieved as of May 22, 2024." API prices change frequently, and the relative cost-effectiveness of MoA-Lite vs. proprietary models depends on the degree to which each vendor reduces prices over time. Additionally, the cost calculation for open-source models assumes usage through a specific inference provider (Together AI); self-hosting costs would differ substantially.

#### What experiments would have strengthened the paper?

**Missing: sensitivity to proposer set composition.** The paper uses one fixed set of 6 proposers for all experiments. How sensitive are the results to which models are included? Would removing the weakest proposer (dbrx-instruct, which is a mediocre proposer and a poor aggregator) improve or degrade performance? Would adding more models from the same family (e.g., both Qwen1.5-72B and Qwen1.5-110B) provide diversity benefits? An ablation systematically varying the proposer set would quantify the marginal value of each included model and provide guidance for practitioners deciding which models to include in their own MoA systems.

**Missing: comparison to best-of-N with the aggregator model alone.** A natural baseline is: take the strongest available model (Qwen1.5-110B-Chat), sample 6 independent responses at temperature 0.7, and use an LLM-Ranker or majority voting to select/aggregate. This would test whether the 6-proposer MoA gains are due to model diversity or simply to having multiple responses to draw from (regardless of source model). The single-proposer baseline in Table 3 partially addresses this but at lower $n$ and without exploring whether single-proposer with a larger $n$ could close the gap with multi-proposer at $n=6$.

**Missing: confidence intervals on the FLASK and MATH results.** The paper reports standard deviations for AlpacaEval and MT-Bench but not for FLASK (Figure 3) or MATH (Table 8). Without error bars, it is difficult to assess whether the per-dimension FLASK improvements are statistically reliable or noise.

**Missing: evaluation on knowledge-intensive benchmarks.** All primary evaluations (AlpacaEval, MT-Bench, FLASK) measure instruction-following and response quality. The paper does not evaluate on factuality benchmarks (e.g., TruthfulQA, FreshQA) or knowledge-intensive tasks (e.g., MMLU). Given that MoA involves models critiquing and synthesizing each other's outputs, there is a plausible risk that factual errors from one proposer could propagate or be amplified through the aggregation process. The FLASK results show improved "factuality" and "correctness," but these are evaluator-assessed dimensions, not verified against ground truth. The MATH results provide ground-truth verification for one domain, but a broader factuality evaluation would significantly strengthen the paper's claim that MoA improves output quality.

**Missing: analysis of failure modes and hallucination propagation.** The paper presents case studies (Appendix C) showing successful synthesis but no systematic analysis of cases where MoA degrades quality or propagates errors. Given that the aggregator sees responses that "may be biased or incorrect" (Table 1 prompt), understanding when and how often the aggregator fails to filter out incorrect information — or actively incorporates it — would be valuable. The Spearman correlation analysis (Figure 4b) suggests the aggregator tends to incorporate higher-quality proposals, but the correlation is modest (0.05–0.30), meaning a substantial fraction of output content comes from lower-quality proposals.

#### Where do the claims hold conditionally, and where do they break?

**Quality improvement from multi-model collaboration:** holds across all tested benchmarks (AlpacaEval, MT-Bench, FLASK, MATH) and all tested aggregator models. The *magnitude* varies — larger on AlpacaEval (21-point gain over the best constituent model), smaller on MT-Bench (0.29-point gain over the best constituent), intermediate on MATH (~8–30-point gain depending on aggregator). The benefit appears most pronounced on tasks where the quality ceiling is lower (more room for improvement) and where complementary perspectives genuinely add value.

**Compounding from additional layers:** holds for Layers 1–3 with diminishing returns. Untested beyond 3 layers. Likely saturates at some point (even Qwen1.5-110B-Chat's curve in Figure 4a is flattening from Layer 2 to Layer 3), but the saturation point is not characterized.

**Cost-effectiveness relative to proprietary models:** holds when using the MoA-Lite configuration (2 layers, smaller aggregator) and when pricing is based on the specific inference provider and date (Together AI, May 22, 2024). Would not necessarily hold for the full 3-layer MoA (which is substantially more expensive than GPT-4o) or under different pricing structures.

**Generalization beyond instruction-following:** the MATH results (Table 8) show generalization to mathematical reasoning, but this is the only non-instruction-following domain tested. Generalization to code generation, factual QA, summarization, or other tasks is not established.

**Proposer-aggregator role specialization:** holds for the six models tested, with two clear proposer specialists (WizardLM 8x22B, LLaMA-3-70B) and two versatile models (Qwen1.5 variants). Whether this specialization pattern generalizes to other models or is an artifact of these specific six models' training is unknown. The finding that LLaMA-3-70B is a strong proposer but weak aggregator is surprising and warrants replication with other LLaMA-3 variants and on other benchmarks.

## 6. Limitations and Trade-offs

### 6.1 The Time-to-First-Token Latency Penalty Is Structural and Unresolved

**The assumption or constraint:** MoA's layered architecture is fundamentally sequential — the aggregator at layer $i+1$ cannot begin generating until *all* proposers in layer $i$ have completed their full responses. The paper explicitly acknowledges this: "Our proposed method requires iterative aggregation of model responses, which means the model cannot decide the first token until the last MoA layer is reached. This potentially results in a high Time to First Token (TTFT), which can negatively impact user experience" (Section 5, Limitations paragraph).

This is not an implementation detail that can be optimized away — it is a direct consequence of the feedforward layered design. In a 3-layer MoA with 6 proposers each generating ~200–500 tokens per response, the user waits for the full duration of Layer 1 (all 6 parallel proposers, bounded by the slowest), then Layer 2 (all 6 proposers again, each now processing the previous layer's aggregated response), then the final aggregator's generation, before seeing a single token of output.

**The consequence:** For any latency-sensitive application — interactive chat assistants, real-time copilots, customer-facing deployments — MoA in its default 3-layer configuration is likely impractical regardless of its quality advantages. The paper's tflops analysis (Figure 5b) uses sequential layer cost as a latency proxy and shows that the 3-layer MoA requires approximately 300–350 tflops versus GPT-4o's estimated ~150 tflops, meaning roughly a 2× latency penalty relative to the proprietary model it outperforms in quality. This latency-quality tradeoff is the central deployment tension the paper does not resolve: users and product teams optimizing for responsiveness will prefer a weaker but faster single model, and the paper provides no guidance on how to navigate this tension beyond "limit the number of MoA layers" (the Limitations paragraph).

**What evidence exists in the paper:** The paper quantifies the latency cost implicitly through Figure 5b (tflops vs. LC win rate), where the 3-layer MoA sits at the far right (highest tflops, highest quality). MoA-Lite (2 layers) reduces the tflops requirement to roughly 150–200 but still involves waiting for two full sequential rounds of generation before the first output token. The paper does NOT measure TTFT directly, does NOT report wall-clock latency numbers for any configuration, and does NOT provide an ablation showing how quality degrades if the architecture is modified to reduce TTFT (e.g., streaming outputs, chunk-wise aggregation, or having earlier layers generate shorter responses). The FLASK "efficiency" dimension (Figure 3) shows MoA trailing GPT-4 Omni, but this is an evaluator-assessed dimension of response quality, not a latency measurement.

**Mitigation status:** The Limitations paragraph suggests "chunk-wise aggregation instead of aggregating entire responses at once, which can reduce TTFT while maintaining response quality" as future work. This is a plausible direction — having proposers generate and stream partial responses, with the aggregator beginning synthesis before all proposals are complete — but it requires the aggregator to perform synthesis on incomplete information, which may degrade the synthesis quality that makes MoA effective in the first place. The paper provides no empirical evidence on whether this tradeoff is favorable.

---

### 6.2 Cost-Per-Query Overhead Is Substantial and Only Partially Offset by MoA-Lite

**The assumption or constraint:** MoA's headline quality results (65.1% LC win rate, Table 2a) require 13 model calls per instance for the 3-layer configuration (6 proposers × 2 intermediate layers + 1 final aggregator). This means the cost-per-query is roughly an order of magnitude higher than using any single model — and this cost is incurred on *every* query, regardless of whether the query would benefit from multi-model synthesis.

The paper is transparent about this cost through Figure 5a, showing that the 3-layer MoA costs approximately $0.025–0.030 per AlpacaEval instance compared to roughly $0.002–0.003 for GPT-4o. This is a ~10× cost premium for a 7.6 percentage point quality improvement. The paper frames MoA-Lite (2 layers, smaller aggregator, achieving 59.3% at ~$0.003–0.005 per instance) as the cost-effective alternative, but even MoA-Lite makes 13 model calls (fewer layers but same number of proposer calls in Layer 1) and achieves only a 1.8-point improvement over GPT-4 Omni.

**The consequence:** There are three distinct problems that the cost analysis surfaces but does not resolve:

*First, the cost premium is paid on all queries, including those where a single model would have sufficed.* The paper does not provide a routing mechanism or confidence threshold to determine when MoA is needed versus when a cheaper single-model call is adequate. This means the average cost includes the MoA overhead even for "easy" prompts where any single proposer would produce an excellent response.

*Second, the cost calculation is snapshot-dependent and likely optimistic for open-source models.* The pricing is based on Together AI's inference endpoints as of May 22, 2024. For organizations self-hosting these models, the cost structure is entirely different — GPU compute, electricity, and amortized hardware dominate, and the relative cost of running 6 models in parallel versus 1 model depends on batching, hardware utilization, and whether models can share infrastructure. The paper's cost analysis does not generalize to self-hosted deployments.

*Third, cost-effectiveness claims depend on which proprietary model is the reference point.* MoA-Lite is "more than twice as cost-effective" when compared to GPT-4 Turbo (55% at ~$0.008–0.010 vs. 59% at ~$0.003–0.005), but it is only marginally cheaper than GPT-4 Omni (57.5% at ~$0.002–0.003 vs. 59% at ~$0.003–0.005) for a small quality gain. The cost-effectiveness framing depends critically on which proprietary model is selected as the baseline.

**What evidence exists in the paper:** Figure 5a provides the cost-quality tradeoff curves, and Section 3.4 discusses the Pareto frontier interpretation. Table 2a reports both MoA and MoA-Lite performance alongside GPT-4 variants. The paper explicitly acknowledges that MoA-Lite is the cost-conscious variant and notes the pricing data source and date. However, it does NOT provide: (1) a per-prompt difficulty breakdown showing which queries drive the MoA benefit, (2) a routing analysis showing potential cost savings from selective MoA application, (3) a self-hosting cost model, or (4) sensitivity analysis to pricing changes.

**Mitigation status:** The paper does not attempt to mitigate the cost overhead beyond offering MoA-Lite as a cheaper configuration. There is no mechanism for adaptive deployment where MoA is invoked only when needed. The proposed mitigations for TTFT (chunk-wise aggregation) do not address cost, since the number of model calls remains unchanged. The cost issue is presented as a tradeoff to be navigated rather than a problem to be solved — users must decide whether the quality improvement justifies the cost premium for their specific application.

---

### 6.3 There Is No Systematic Evaluation of When MoA Fails or Degrades Quality

**The assumption or constraint:** The paper's evaluation focuses exclusively on aggregate metrics (average LC win rate, average MT-Bench score, average FLASK dimension scores) and provides only two cherry-picked case studies (Appendix C) showing successful synthesis. It does not characterize the *distribution* of outcomes — specifically, whether MoA occasionally produces worse responses than any individual proposer would have, whether it systematically degrades on certain prompt types, or whether factual errors propagate and amplify through the aggregation layers.

This is not a minor omission. The Aggregate-and-Synthesize prompt (Table 1) explicitly instructs the aggregator that "some of it may be biased or incorrect," implying that the system *expects* some proposer outputs to contain errors. Whether the aggregator reliably filters these out versus occasionally incorporating them is a central reliability question that the aggregate metrics cannot answer — a system that improves average quality while introducing occasional catastrophic failures may be unacceptable for high-stakes applications.

**The consequence:** A practitioner cannot determine from the paper's results whether MoA is *safe* for their use case. An average LC win rate improvement of 7.6 percentage points could be achieved by making moderate responses better while occasionally producing nonsensical outputs, by improving all responses uniformly, or by dramatically improving a subset while leaving the rest unchanged. These scenarios have very different deployment implications. For example, if MoA's synthesis occasionally produces a factually coherent but incorrect response that sounds more authoritative than any proposer's output (because it synthesizes surface fluency from multiple models without discriminating truth value), this would be a serious failure mode for factuality-critical applications — and the aggregate FLASK "factuality" score would not necessarily capture it, since FLASK uses an LLM evaluator rather than ground truth.

**What evidence exists in the paper:** The paper provides essentially no negative-case evidence. The two case studies (Tables 6 and 7 in Appendix C) show only successful synthesis. Table 7 is explicitly framed as "all proposed responses are not good enough" but still shows the aggregator producing a better response — the paper selects a case where aggregation works rather than where it fails. The Spearman correlation analysis (Figure 4b) shows the aggregator tends to favor higher-quality proposals, but the correlation is modest (0.05–0.30), meaning a substantial fraction of output similarity is *not* explained by proposal quality. The MATH results (Table 8) provide ground-truth accuracy, showing consistent improvement across aggregators, but these are also aggregate numbers.

The paper does NOT report: (1) instances where the MoA response received a lower GPT-4 preference score than the best individual proposer's response, (2) an analysis of whether MoA ever "hallucinates" facts that were not present in any proposer output, (3) a breakdown by prompt category or difficulty level, (4) inter-rater reliability for the FLASK dimension scores (which are themselves LLM-evaluated), or (5) human evaluation of MoA outputs versus proposer outputs on dimensions the LLM evaluator might miss.

**Mitigation status:** Not addressed. The Limitations paragraph focuses on latency (TTFT) and does not mention failure mode analysis. The Conclusion highlights interpretability as a benefit ("since the intermediate outputs are expressed in natural language, MoA presented improves the interpretability of models") without acknowledging that interpretability of intermediate outputs does not guarantee correctness of the final synthesis. The paper does not suggest future work on characterizing MoA failure modes or developing confidence estimates for synthesized outputs.

---

### 6.4 The Approach Is Validated on a Single Model Family Ecosystem and Evaluation Methodology

**The assumption or constraint:** All experiments use the same set of six open-source models (Qwen1.5 variants, WizardLM, LLaMA-3-70B, Mixtral-8x22B, dbrx-instruct), evaluated primarily on three benchmarks (AlpacaEval 2.0, MT-Bench, FLASK) that all use GPT-4 as the evaluator. The paper does not test with different model families (e.g., only LLaMA variants, only Qwen variants, or Claude-series models as proposers/aggregators), different evaluator models, or non-LLM-judge benchmarks (with the partial exception of MATH in Appendix D, which provides ground-truth accuracy).

This matters because the observed "collaborativeness" phenomenon and the MoA performance gains could be specific to: (a) the evaluator being a GPT-4 variant that shares training methodology with the proposer models, creating an evaluator-proposer alignment bias; (b) the proposer set containing models trained with similar instruction-tuning recipes (all six are chat-aligned models trained for similar interaction patterns); (c) the specific combination of model scales (70B–110B) and architectures represented in the set.

**The consequence:** The paper's central empirical claims — that collaborativeness is a "widespread" phenomenon, that MoA achieves state-of-the-art performance, and that the proposer-aggregator decomposition generalizes — rest on a narrow evidential base. A practitioner with a different set of available models (e.g., only smaller 7B–13B models, or a mix of proprietary and open-source models with different API access patterns) cannot confidently extrapolate the paper's findings to their setting. Similarly, a practitioner deploying in a domain where LLM-as-judge evaluation is known to be unreliable (e.g., specialized technical domains, non-English languages, creative tasks) cannot assume the evaluator-based quality improvements will translate to human-perceived quality.

Specific risks: (1) the evaluator bias hypothesis — GPT-4 might prefer MoA outputs because the multi-model synthesis produces responses that are stylistically similar to what GPT-4 would generate (averaging across multiple GPT-4-aligned models), not because they are objectively better; (2) model homogeneity — all six proposers are trained with similar post-training objectives (instruction following, chat alignment), so the "diversity" measured in Table 3 may represent relatively shallow differences in style rather than deep differences in capability or knowledge; (3) benchmark overfitting — the MoA system architecture (particularly the Aggregate-and-Synthesize prompt) might be implicitly optimized for the evaluation criteria used by GPT-4 evaluators, without the designers having explicitly tuned against the benchmark.

**What evidence exists in the paper:** The MATH results (Table 8) provide some evidence against the evaluator-bias hypothesis because MATH uses ground-truth accuracy rather than LLM evaluation, and MoA shows consistent improvement there. However, MATH is only evaluated in Appendix D without statistical significance measures, and it represents a single non-instruction-following domain. The FLASK results (Figure 3) provide finer-grained evaluation that partially addresses the "what improved?" question, showing gains on factuality, correctness, and insightfulness — dimensions that are harder to attribute to pure stylistic matching. However, all FLASK scores are themselves LLM-assigned, so the evaluator bias concern remains.

The paper does NOT provide: (1) human evaluation on any benchmark, (2) evaluation using non-GPT evaluators (e.g., Claude, Gemini as judges), (3) testing with proposer sets drawn from a single model family to isolate the effect of family diversity versus instance diversity, (4) testing with models of substantially different scales (e.g., including 7B models as proposers), or (5) evaluation on benchmarks where LLM-as-judge is known to be poorly calibrated.

**Mitigation status:** Partially addressed through the MATH results (ground-truth evaluation) and the FLASK fine-grained analysis (shows which dimensions improve). The paper does not claim generalization beyond the tested models and benchmarks but also does not discuss the evaluator-bias confound or the model-homogeneity confound as limitations. The Conclusion frames MoA as broadly applicable without caveats about the evaluation methodology.

---

### 6.5 The Optimal Model Selection Strategy (Which Proposers, Which Aggregator, How Many Layers) Is Not Characterized

**The assumption or constraint:** The paper provides a specific MoA configuration (6 named models, 3 layers, Qwen1.5-110B-Chat as aggregator) and demonstrates it works well, but provides very limited guidance on how a practitioner with a *different* set of available models should construct their own MoA system. The proposer-aggregator analysis (Table 4) characterizes the six tested models but does not provide a method for predicting which models will be good proposers or aggregators without running the full MoA evaluation. The number of layers is treated as a hyperparameter to sweep manually rather than a quantity that can be determined from properties of the models or the task.

This is a practical limitation because the MoA approach is explicitly pitched as training-free and model-agnostic ("it can be applied to the latest LLMs regardless of their size or architecture," Section 2.3). A practitioner receiving this message would reasonably expect guidance on *how* to apply it to their specific model portfolio, but the paper provides only a worked example, not a methodology.

**The consequence:** The paper demonstrates that MoA works for one specific model combination but does not establish whether the approach is *robust* to suboptimal model selection. A practitioner who selects a weaker aggregator (e.g., LLaMA-3-70B, which achieves only 45.0% as an aggregator in Table 4) or omits the strongest proposers (e.g., excluding WizardLM 8x22B, which contributes the highest proposer value at 63.8%) might see substantially smaller gains — or even degradation — relative to using the strongest individual model alone. Without a predictive framework for model selection, the MoA approach may only be reliably beneficial when one happens to have access to models that happen to be good proposers and an aggregator that happens to be good at synthesis — a set of conditions that may not hold for arbitrary model portfolios.

Additionally, the paper does not characterize how MoA performance scales with proposer set size beyond $n=6$. Table 3 shows improvement from $n=2$ to $n=3$ to $n=6$, but whether this trend continues asymptotically or saturates is unknown. A practitioner with access to 10 or 20 models does not know whether to include all of them (increasing cost linearly) or to select a subset (risking omission of a valuable proposer).

**What evidence exists in the paper:** Table 4 provides per-model proposer and aggregator scores, which is the most useful guidance the paper offers — it shows that one should test candidate models in both roles rather than assuming standalone quality predicts collaborative value. Table 3 shows that more proposers help (monotonically increasing with $n$) and that model diversity helps (multi-proposer > single-proposer). Figure 4a shows aggregator choice is the single most impactful decision (roughly a 25-point spread at Layer 3 across aggregators). However, these are *descriptive* results about the tested set, not a *predictive* methodology.

The paper does NOT provide: (1) correlation between standalone benchmark performance and proposer/aggregator value, (2) a method for estimating a model's proposer or aggregator value without running the full MoA pipeline, (3) analysis of whether the optimal aggregator is simply the model with the highest standalone LC win rate (Qwen1.5-110B-Chat fits this pattern, but the tested set is too small to establish a rule), (4) experimentation with proposer sets that omit the strongest proposer or the weakest proposer to measure robustness, or (5) guidance on the minimum number or diversity of proposers needed for MoA to reliably outperform the best single model.

**Mitigation status:** The paper describes the model selection criteria in Section 2.1 — "Performance Metrics" (average win rate) and "Diversity Considerations" (heterogeneous models) — and demonstrates these criteria through the configuration choices. However, this is more of a post-hoc justification of the selected configuration than a validated selection methodology. The paper does not claim to have solved model selection and does not suggest a specific direction for future work on this topic.

---

### 6.6 The Residual Connection and Prompt Design Are Not Ablated; Their Causal Contribution to Performance Is Unknown

**The assumption or constraint:** The MoA architecture includes two specific design elements whose importance is asserted but never empirically tested: (1) the residual connection (`$+ x_1$` in Equation 1, prepending the original prompt to every layer's input), which the paper analogizes to residual connections in MoE architectures (Section 2.3) as preventing "drift" across layers, and (2) the specific text of the Aggregate-and-Synthesize prompt (Table 1), which instructs the aggregator to "critically evaluate" proposals and "not simply replicate" them. The paper attributes MoA's synthesis behavior (as opposed to selection behavior) to this prompt design, but never compares it against alternative prompts or a baseline without the synthesis instruction.

**The consequence:** A practitioner reimplementing MoA cannot know which design elements are load-bearing. If the residual connection is not actually needed (because the aggregator naturally maintains focus on the original query without explicit reprompting), simplifying the input structure could reduce context window usage and latency. If a simpler prompt (e.g., "Here are some responses from other models. Based on these, answer the user's question.") achieves comparable performance, the specific "critical evaluation" and "synthesis" instructions might be unnecessary, and the MoA gains might be primarily attributable to the multi-model information access rather than the prompt engineering. Conversely, if the prompt is essential, a poorly designed prompt could cause MoA to underperform, leading practitioners to incorrectly conclude that the approach does not work for their models.

More fundamentally, the paper's central claim — that MoA achieves synthesis rather than selection — is supported by comparing MoA to an LLM-Ranker (Figure 4a), but the LLM-Ranker uses a *different prompt* (Appendix Table 5) that explicitly asks for selection. The observed performance gap between MoA and the ranker could be due to (a) synthesis genuinely outperforming selection, as the paper claims, or (b) the Aggregate-and-Synthesize prompt being more effective at eliciting high-quality output from the model regardless of the multi-model input. Without a control where the same aggregator model is given the same auxiliary responses but with a neutral prompt (not instructing synthesis), these explanations are confounded.

**What evidence exists in the paper:** No ablation experiments address these questions. The residual connection is mentioned in Equation 1 and the MoE analogy in Section 2.3 but is never removed or modified in any experiment. The Aggregate-and-Synthesize prompt is presented in Table 1 and used in all MoA experiments, with no comparison to alternative prompts — the only prompt variation is the LLM-Ranker prompt (Appendix Table 5), which changes both the task (selection vs. synthesis) and the prompt wording simultaneously.

The Spearman correlation analysis (Figure 4b) provides indirect evidence that the aggregator is incorporating proposer content (positive correlation between output similarity and proposer quality), but this does not isolate the contribution of the prompt design — a model given a neutral prompt with access to multiple outputs might also tend to incorporate the best ones without being explicitly instructed to do so.

**Mitigation status:** Not addressed. The paper treats the Aggregate-and-Synthesize prompt as a fixed component of the MoA architecture rather than a design variable to be studied. The Conclusion does not mention prompt design optimization as future work. A simple ablation — replacing the synthesis instruction with a neutral "Here are some reference responses, now answer the question" prompt — would have substantially strengthened the paper's mechanistic claims about synthesis versus selection, at minimal experimental cost.

## 7. Implications and Future Directions
- How this changes the landscape
  - MoA shows that model-level composition—without training—can surpass single-model performance and even top closed models in some alignment benchmarks (Table 2a). This reframes progress from “build a bigger single LLM” to “compose strong, diverse LLMs effectively.”

- Practical applications
  - High-stakes drafting (policy, legal, technical writing) where completeness and factuality matter; MoA can produce more robust, cross-checked outputs (Figure 3).  
  - Domain aggregation: mix domain experts (code, math, medical) as proposers and use a general aggregator to produce a unified answer.  
  - Cost-aware deployments: use `MoA-Lite` to hit a target quality at lower dollar/latency budgets (Figure 5).

- Follow-up research directions
  - Adaptive routing: learn to choose the number and identity of proposers per prompt to minimize cost while preserving quality (extends Section 3.4).  
  - Streaming/low-TTFT aggregation: chunk-wise or incremental synthesis to reduce perceived latency (Limitations in Section 5).  
  - Task-specific aggregation prompts: customize Table 1’s synthesis instruction by task (e.g., safety-critical domains).  
  - Hybrid training: fine-tune a lightweight aggregator model on MoA transcripts to further improve fusion while retaining most of the zero-shot flexibility.  
  - Reliability analyses: formal studies on when and why “collaborativeness” holds (Figure 1) and failure cases where misleading proposer content might bias the aggregator.

Overall, this paper provides both a practical recipe and a set of design principles—use diverse proposers, pick a strong aggregator, and aggregate in layers—that together deliver consistent gains across strong baselines while offering an attractive cost–quality–latency trade-off.
