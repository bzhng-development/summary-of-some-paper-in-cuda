# HERMES 3 TECHNICAL REPORT

**ArXiv:** [2408.11857](https://arxiv.org/abs/2408.11857)

## 🎯 Pitch

Hermes 3 introduces a family of openly available, highly steerable instruction-tuned large language models (8B, 70B, 405B parameters) built on Llama 3.1, distinguished by their neutral alignment to system prompts and robust agentic tool-use capabilities. By combining a curated, diverse training regimen with innovative token-level support for multi-step reasoning, structured outputs, retrieval-augmented generation, and tool-calling, Hermes 3 enables precise, reliable, and transparent control—meeting the needs of developers seeking flexible, enforceable workflows without the constraints or refusals imposed by commercial safety-aligned models. This opens new horizons for enterprise, research, and open-source communities by delivering state-of-the-art performance with unparalleled controllability and interpretability.

---

## 1. Executive Summary

This technical report introduces **Hermes 3**, a family of neutrally-aligned instruct and tool use models fine-tuned from Llama 3.1 (8B, 70B, and 405B) on a diverse 390-million-token synthetic dataset spanning general instructions, domain expertise, math, roleplaying, coding, and agentic tasks. The training recipe combines supervised fine-tuning with sample packing at 96% efficiency and an optional direct preference optimization (DPO) phase via LoRA adapters, yielding the largest variant—Hermes 3 405B—which achieves state-of-the-art performance among open weight models on several public benchmarks, including 69.45% on ARC-Challenge, 85.02% on 5-shot MMLU, and 30.85% on 4-shot MATH Level 5. The work establishes that a neutrally-aligned instruction model with heavily structured system prompt conditioning and explicit agentic tagging (scratchpads, internal monologues, XML tool-call schemas) can match or approach Llama 3.1 Instruct on most evaluations without built-in refusal guardrails, though it trails its parent model on instruction-following strictness (IFEval Strict: 84.87% vs. 87.09%) and advanced reasoning benchmarks (MMLU-PRO: 54.14% vs. 63.51%), confirming that neutral alignment trades some structured obedience for creative and roleplaying flexibility.

## 2. Context and Motivation

### The Core Problem: Controlling Large Language Models Without Breaking Them

The fundamental tension this paper addresses is deceptively simple: **how do you make a powerful language model steerable and useful without neutering its capabilities through excessive safety training?** This matters because instruct tuning—the process of adapting a base model to follow user instructions—has become the primary interface through which people interact with LLMs. Prior to Hermes 3, the dominant paradigm for instruct models, particularly among closed-weight commercial systems, was to embed refusal guardrails: the model is explicitly trained to decline certain types of requests on moral, legal, or safety grounds.

The authors argue that this approach inevitably "lobotomizes" the model's reasoning capabilities. Their position, stated forcefully in the introduction, is that:

> "Large language models have very limited direct agency. Rather, it is the systems and applications that we, as humans, build with them that give them any degree of agency to the outside world. We believe that a more appropriate place for guardrails and active intervention is at the larger system levels, rather than on the models themselves, which can result in an a priori lobotomization of potential lines of thinking."

This is not merely a philosophical stance. It has concrete implications for what the model can do. A model trained to refuse certain topics cannot engage with them at all—even in educational, analytical, or creative contexts where engagement would be appropriate. The paper's tagline captures this starkly: **"For Hermes, there is no such thing as latent thoughtcrime."** The implication is that Hermes 3 should be able to reason about any topic while remaining steerable through system prompts, leaving application developers—not model trainers—to decide what constraints to impose.

### Why This Problem Matters Now

The timing of this work is driven by several converging trends that make neutral alignment both more urgent and more feasible than before.

**First, the open weight model ecosystem has matured.** The release of Llama 3.1 (the Herd of Models, including an 8B, 70B, and 405B variant) provides base models with strong reasoning capabilities and 128K context windows that are available for fine-tuning. This creates an opportunity: rather than building a model from scratch, the community can build on top of Llama 3.1's capabilities while applying specialized training for steerability and agentic behavior. The paper explicitly positions Hermes 3 as an extension of this foundation, not a replacement for it—the goal is to add capabilities (neutral alignment, tool use, structured reasoning) that Llama 3.1's own instruct tuning does not prioritize.

**Second, the "walled garden" approach is showing cracks.** Closed-weight commercial models like ChatGPT and Claude are trained with extensive refusal guardrails that, the authors imply, reduce their utility for creative and open-ended applications. The specific use cases emphasized in the paper—roleplaying, creative writing, transparent agentic reasoning—are precisely the kinds of applications where refusal training is most likely to interfere. A roleplaying model that refuses to adopt certain personas, or a coding assistant that declines to explain certain algorithms, is fundamentally less useful than one that can engage neutrally with any request while still following system-level constraints.

**Third, agentic applications demand transparency.** When an LLM acts as an agent—planning multi-step tasks, calling tools, retrieving documents—its internal reasoning process needs to be visible and auditable. The paper emphasizes this through its structured XML tagging system (`<SCRATCHPAD>`, `<REASONING>`, `<INNER_MONOLOGUE>`, etc.), which allows the model's decision-making to be inspected step by step. This is not just a convenience; it is a prerequisite for debugging, trust, and safety in agentic deployments. A model that simply refuses certain operations provides no insight into why, making it harder to build reliable agentic systems.

**Fourth, instruction-following benchmarks are improving but the field lacks a neutral baseline.** The paper notes that Llama 3.1 Instruct scores 87.09% on IFEval Strict—a measure of how precisely the model follows formatting and content instructions. But what does instruction-following look like when you strip away the "helpful assistant" persona and allow the model to be whatever the system prompt specifies? No existing benchmark captures this tradeoff between strict obedience and persona flexibility, and Hermes 3 serves as an empirical data point: it sacrifices some IFEval performance (84.87% vs. 87.09% for the 405B model) in exchange for dramatically expanded creative range.

### Where Prior Approaches Fall Short

The paper identifies several specific limitations in existing instruct-tuning approaches, both open and closed:

**Refusal training creates capability blind spots.** Commercial instruct models are typically trained to refuse categories of requests involving violence, illegal activity, explicit content, or controversial topics. The paper does not contest that such refusals are appropriate in a deployed chatbot, but argues that embedding them in the model itself is the wrong architectural choice. The consequence—alluded to but not directly measured in the paper—is that refusal-trained models may be unable to engage with legitimate requests that superficially resemble refused categories. A model that refuses to "explain how to pick a lock" may also refuse to "explain how lock-picking works for a novel I'm writing about a locksmith," because the refusal classifier cannot distinguish context.

**Existing open instruct models have uneven capability coverage.** The paper's data mixture (Table 1) reveals the authors' diagnosis of gaps in prior work. General instructions make up 60.6% of the dataset, but the remaining ~40% is deliberately allocated to domains the authors identified as weaknesses in previous Hermes releases: domain expertise (12.8%), math (6.7%), roleplaying (6.1%), coding (4.5%), tool use and agentics (4.3%), content generation (3.0%), and steering/alignment (2.5%). This allocation implicitly criticizes prior open models as being overly focused on short Q&A exchanges ("arbitrary questions posed by everyday users") at the expense of structured, domain-specific, and interactive capabilities.

**Output-only verifier judgments are missing.** The paper describes Hermes 3's ability to act as a reward model—evaluating the quality of generated text with nuanced judgment (as illustrated in Figure 7). This capability is not common in prior open instruct models, which are typically optimized for generation rather than evaluation. The implication is that a model capable of both generation and evaluation can participate in automated self-improvement loops, a capability the paper flags as important but does not fully explore.

**Agentic reasoning lacks transparency in existing systems.** Prior models with tool-use capabilities (e.g., Toolformer; Schick et al., 2023) focus on the mechanics of invoking tools but do not emphasize making the model's internal reasoning visible to the user or system developer. The paper's emphasis on XML-tagged reasoning sections—with distinct tags for planning, execution, reflection, and explanation—represents a deliberate design choice to make agentic behavior auditable. This is positioned as a deficiency in prior work that the Hermes 3 approach remedies.

**System prompt sensitivity is underexploited.** Most instruct models respond to system prompts, but the paper argues that this sensitivity is typically undertrained. The result is models that default to a "helpful assistant" persona regardless of the system prompt's actual content. Hermes 3 is explicitly trained to take system prompts more seriously—the paper notes that in the 405B variant, "an empty system prompt does not necessarily elicit the 'helpful assistant' persona" (Section 2, Figure 6), and instead produces a confused, disoriented character. This is presented as evidence that the model truly conditions on the system prompt rather than falling back to a default behavior.

### How This Paper Positions Itself

The paper positions Hermes 3 as a deliberate counterpoint to the prevailing instruct-tuning paradigm. Rather than proposing a fundamentally new training method or architecture, the contribution is in the **training data composition, the neutral alignment philosophy, and the structured agentic tagging system**.

The release sits at the intersection of several existing threads:

**From the instruct-tuning lineage** (Sanh et al., 2022; Wei et al., 2022), the paper inherits the basic approach of fine-tuning a base model on imperative statements to make it steerable. The innovation is in what the model is steered *toward*: not a fixed helpful assistant persona, but whatever the system prompt specifies.

**From the synthetic data generation literature** (Xu et al., 2023's Evol-Instruct; interstellarninja's Hermes function calling dataset), the paper inherits techniques for programmatically creating diverse training examples. The 390-million-token dataset is described as "meticulously curated and generated," combining off-the-shelf sources with domain-specific Evol-Instruct-style generation. The five-month curation timeline (March to August 2024) signals the scale of effort involved in manual quality filtering.

**From the tool-use literature** (Schick et al., 2023's Toolformer), the paper inherits the concept of LLMs calling external functions, but extends it with a standardized XML schema (the Hermes Function Calling standard) that wraps tool definitions in `<tools>` tags and invocations/responses in `<tool_call>` and `<tool_response>` tags. This standardization makes tool use compatible with the broader structured reasoning framework.

**From the DPO literature** (Rafailov et al., 2023), the paper inherits a preference optimization step but applies it via LoRA adapters (Hu et al., 2022) rather than full model tuning, which the paper justifies as necessary for the "larger model sizes" where holding both a reference and trained model in GPU memory is prohibitive. Notably, DPO provided only "moderate" improvements for the 8B model (Table 4: GPT4All 72.03% → 72.30%) and "negligible" improvements for 70B and 405B, leading the authors to ship the SFT checkpoints for the larger models. This is an interesting negative result: at scale, the benefit of preference tuning may diminish when the SFT data is already high-quality.

**From the open weight model movement**, the paper positions Hermes 3 as the open alternative to closed-weight commercial models specifically for applications where neutral alignment matters. The explicit contrast with models that "refuse instructions on moral grounds" frames Hermes as filling a gap that commercial providers have deliberately left open.

The paper does not claim to advance the theoretical understanding of instruct tuning or alignment. It is an engineering contribution: a carefully constructed data mixture, a specific training recipe, and a release of weights that achieves competitive benchmark performance while embodying a distinct philosophical stance on model alignment. The evaluation section (Table 5) positions Hermes 3 against Llama 3.1 Instruct across all sizes, implicitly framing the comparison as: "here is what you gain and lose by choosing neutral alignment over the default instruct tuning."

### The Unstated Tension

A careful reader will notice a tension the paper does not explicitly resolve. The system prompt provides a mechanism for applying guardrails at the application level—the argument being that this is superior to embedding refusals in the model itself. But the paper also emphasizes that Hermes 3 is "highly sensitive to the system prompt" and takes it very seriously. This raises a question: if the system prompt can impose guardrails, and the model faithfully follows the system prompt, then what is the practical difference between model-level and system-level guardrails? The answer, implied but not stated, is about **reversibility and transparency**. A refusal embedded in model weights is opaque and cannot be inspected or modified by the application developer. A refusal encoded in a system prompt is explicit, auditable, and can be adjusted per-application without retraining. The paper's contribution is the model that makes this system-level approach viable by being genuinely responsive to system prompts rather than defaulting to an embedded persona.

## 3. Technical Approach

### 3.1 Reader Orientation

Hermes 3 is a family of three instruction-tuned language models (8B, 70B, and 405B parameters) built by fine-tuning Llama 3.1 base models on a 390-million-token synthetically-generated dataset. The system solves the problem of creating a highly steerable but neutrally-aligned model—one that faithfully follows whatever persona and constraints are specified in its system prompt, without baked-in refusal guardrails that limit its reasoning range—by combining domain-diverse supervised fine-tuning with structured agentic tagging, optional preference optimization, and a training recipe engineered for efficient packing of heterogeneous instruction samples.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major components arranged in a linear pipeline:

1. **Base Models (Llama 3.1 8B/70B/405B):** Three decoder-only Transformer models from Meta, each with a 128K-token context window and standard pretrained weights. These provide the foundation capabilities—language understanding, reasoning, and generation—that the subsequent training stages shape rather than create from scratch.

2. **Supervised Fine-Tuning (SFT) Phase:** The core training stage where all three base models are fine-tuned on the same 390M-token dataset. This phase teaches the model to respond to instructions, adopt personas from system prompts, use structured XML tags for agentic reasoning, invoke tools via the Hermes Function Calling standard, and cite retrieval sources. The training uses sample packing to achieve 96% sequence utilization efficiency, special ignore-value masking so the model learns only from output tokens, and a cosine-decayed learning rate schedule selected via hyperparameter sweep on the 8B model as a proxy for the larger scales.

3. **Direct Preference Optimization (DPO) Phase (optional, 8B only):** A lightweight preference-tuning step applied via LoRA adapters (rank-32) targeting all linear layers, using RMSProp with NEFTune noise. This phase is applied to the 8B model post-SFT to refine response quality based on human or synthetic preference pairs. For the 70B and 405B models, the DPO gains were negligible and this phase was omitted—the SFT checkpoints shipped directly.

4. **Evaluation Suite:** A standardized battery of public benchmarks (GPT4All, AGIEval, BBH, MATH, GPQA, MuSR, MMLU, MMLU-PRO, IFEval, MT-Bench, TruthfulQA) used during checkpoint selection (picking the best epoch) and for final reported results. The 405B evaluations use FP8 quantization via llm-compressor for vLLM.

Information flows linearly: pretrained Llama 3.1 weights → SFT on curated 390M-token mix → (optional DPO on 8B) → final Hermes 3 weights → FP8 quantization for 405B inference → benchmark evaluation.

### 3.3 Roadmap for the Deep Dive

- **First, the SFT data mixture** (Table 1): what is in the 390M-token dataset, how each category was sourced (curated vs. Evol-Instruct generated), and why the specific proportions were chosen—since data composition is the paper's primary contribution.
- **Second, the SFT training procedure**: hyperparameters (learning rates, optimizer, batch sizes, epoch counts), the sample packing mechanism (Figure 3), and the special ignore-value masking that ensures the model learns only from response tokens—since these engineering choices enable training at scale and affect final performance.
- **Third, the checkpoint selection protocol**: how epochs are compared using min-max normalized benchmark scores to pick the best model without overfitting to any single metric—since this determines which weights ship.
- **Fourth, the DPO phase**: LoRA configuration, optimizer choices, reward margin trajectory, and why DPO was dropped for 70B/405B—since this is a negative result with practical implications.
- **Fifth, the system prompt sensitivity and agentic tagging system**: how the model uses reserved tokens and XML structures for transparent reasoning—since this is the mechanism that distinguishes Hermes 3's steerability from standard instruct models.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily an **engineering contribution paper** whose core idea is that a carefully composed, domain-diverse SFT dataset, combined with explicit system prompt conditioning and structured agentic tagging, can produce an instruct model that is simultaneously highly steerable and neutrally aligned—matching or approaching commercial instruct models on benchmarks while refusing to embed refusal guardrails in the model weights. The training recipe (sample packing, learning rate sweeps, epoch selection via multi-benchmark normalization, optional DPO) is optimized for efficient use of GPU resources across three very different model scales.

---

#### SFT Data Mixture

The dataset is the paper's central contribution—everything downstream depends on its composition and quality. The paper describes a five-month curation effort (March to August 2024) producing approximately 390 million total tokens, of which 270 million (69%) are output tokens contributing to the cross-entropy loss and 120 million (31%) are input tokens masked out during training.

The dataset has eight categories with specific proportions:

- **General Instructions (60.6%, 236M tokens):** The bulk of the data, sourced primarily from existing high-quality instruction datasets. The paper specifically cites interstellarninja's Hermes Function Calling training set (the v1 release) as one source. The selection criteria were coherence, educational value, and reliability. This category represents "arbitrary questions posed by everyday users"—standard Q&A, explanations, and task completions that form the backbone of instruction-following behavior. The paper implicitly criticizes prior models (including earlier Hermes versions) for being overly concentrated in this category, hence the deliberate allocation of ~40% of tokens to other domains.

- **Domain Expert (12.8%, 50M tokens):** Data covering specialized knowledge domains. The paper does not enumerate the specific domains, but the proportion signals that general instructions alone are insufficient for producing a model that can engage with technical or specialized queries. This category was generated using domain-specific Evol-Instruct-style schemes (Xu et al., 2023), where seed instructions are programmatically rewritten to increase complexity, add constraints, or broaden coverage.

- **Math (6.7%, 26M tokens):** Mathematical reasoning problems, again Evol-Instruct generated. This category targets a known weakness in prior Hermes models: mathematical reasoning capabilities that rely on multi-step deduction rather than memorized answers. The proportion (roughly 1/15 of the total dataset) is modest compared to the model's overall size, reflecting the fact that math is one capability among many rather than the primary focus.

- **Roleplaying (6.1%, 24M tokens):** Multi-turn conversational data where the model adopts and maintains diverse personas across scenarios. The paper emphasizes that this category enables Hermes 3 to "adopt and consistently maintain diverse personas across various scenarios, dynamically adapting language, knowledge base, and behavioral patterns to suit the chosen role." This is a deliberately trained capability, not an emergent one—the model sees explicit examples of persona-adoption and persona-consistent dialogue during SFT.

- **Coding (4.5%, 18M tokens):** Code generation, explanation, and documentation across multiple programming languages. The size of this category (relatively small at ~1/22 of the total) is notable: the model achieves competitive coding benchmarks without dedicating a large fraction of its training data to code, likely because Llama 3.1's base pretraining already includes substantial code exposure.

- **Tool Use, Agentic, and RAG (4.3%, 17M tokens):** Data teaching the model to invoke tools via the Hermes Function Calling standard, perform retrieval-augmented generation with `<co>` citation tags, and engage in multi-step agentic planning. This is the most structurally complex category: samples include XML-wrapped tool definitions, tool call invocations, and tool response incorporation, all in a specific format that the model must learn to parse and generate correctly.

- **Content Generation (3.0%, 12M tokens):** Creative writing, article generation, and other long-form output tasks. This category complements roleplaying by training the model to produce extended creative text that is not necessarily persona-driven.

- **Steering and Alignment (2.5%, 10M tokens):** The smallest category but perhaps the most philosophically significant. This data teaches the model to respond exactly to system prompt instructions and to maintain neutral alignment—following instructions without imposing its own moral judgments. The paper frames this as training the model to be "highly sensitive to the system prompt" (Section 2), with the 405B variant's empty-system-prompt behavior (Figure 6: a confused, disoriented character rather than a default helpful assistant) as evidence that the steering training worked.

**Design justification for the mixture.** The proportions are not claimed to be optimal in any theoretical sense. They reflect the authors' diagnosis of capability gaps in prior models, balanced against practical constraints on data generation cost. The paper notes that domain-specific Evol-Instruct generation is "computationally intensive" and that the manual filtering process was applied "to both curated and domain-specific instructions." The five-month timeline suggests substantial human effort in quality control beyond what automated generation alone would require.

**Data filtering pipeline.** The paper describes a rigorous multi-stage filtering process applied to all data, both curated and generated:

1. **Token length thresholds:** Conversations with extreme length imbalances (very short responses to long instructions, or vice versa) were removed to maintain a natural conversational flow and prevent the model from learning to produce disproportionately short or verbose responses.

2. **Refusal and format removal:** Any samples containing refusal language (the paper does not specify exact patterns, but typical examples include "I cannot", "I'm not able to", or "As an AI, I should not") were filtered out. Improperly formatted responses—presumably those with malformed XML tags, missing required sections, or inconsistent tool call syntax—were also removed.

3. **Empty turn removal:** Conversations with missing or empty turns (either user or assistant messages that contained no content) were eliminated. This prevents the model from learning to produce empty responses or interpret empty inputs as meaningful.

4. **Strongest-model prioritization:** When multiple candidate generations existed for the same instruction, those produced by "the strongest models" were prioritized. The paper does not specify which models served as the "strongest," but this implies a form of model-based quality filtering where higher-capability models (possibly larger Llama variants or commercial APIs) generated the highest-quality examples that were then retained.

The result of these filtering stages is a dataset where "only 4% of tokens are the padding token" when packed (Section 4.1), implying that nearly all data survived filtering and was suitable for training—a testament to the quality of the initial curation rather than aggressive filtering.

---

#### SFT Training Procedure

The SFT phase uses standard instruct fine-tuning with several engineering optimizations for efficiency at scale. The base models are Llama 3.1's three decoder-only Transformers: 8B, 70B, and 405B parameters, all with a native context window of 131,072 (128K) tokens. The training runs for four epochs across all sizes, with checkpoint selection at the end.

**Optimizer and learning rate schedule.** All sizes use AdamW (Loshchilov and Hutter, 2019) with weight decay of 0.01. The learning rate follows a cosine decay schedule (Loshchilov and Hutter, 2017) after 300 warmup steps. The peak learning rate differs by model size:

- **8B and 70B:** `$7 \times 10^{-6}$`, selected via a hyperparameter sweep on the 8B model trained to completion and evaluated on the GPT4All benchmark suite (Figure 2). The sweep tested learning rates from `$5 \times 10^{-6}$` to `$10 \times 10^{-6}$` on the 8B model, and the highest-scoring learning rate was applied to both 8B and 70B training.

- **405B:** `$3.5 \times 10^{-6}$`, which is exactly half the smaller models' rate. The paper states that "lowering the learning rate relative to the 8B and 70B models produced superior results" for the 405B, which is consistent with the general observation that larger models are more sensitive to learning rate and benefit from more conservative optimization. The paper does not describe a separate sweep for the 405B—this value appears to have been chosen based on trial runs.

**Batch size and hardware configuration.** The batch size and GPU count scale with model size:

- **8B:** 48 GPUs (6 HGX nodes, each with 8 H100 SXM5 GPUs), effective batch size 48
- **70B:** 48 GPUs (same configuration), effective batch size 48
- **405B:** 128 GPUs (16 HGX nodes), effective batch size 128

The 8B and 70B models were trained using PyTorch FSDP (Fully Sharded Data Parallelism) with the nodes connected via Quantum-2 InfiniBand. The 405B model required special consideration: "under standard FSDP the absolute minimum system configuration to avoid out-of-memory errors (training at a context length of 8K tokens) is seven HGX nodes in conjunction with CPU parameter offloading." CPU parameter offloading incurs a non-negligible slowdown—the paper estimates a "45% drop in training efficiency" for the 405B model when using it. The final 405B run used 16 HGX nodes (double the minimum) to avoid offloading entirely, but the paper notes this created a tension: "the high number of GPUs required to train a 405B model would otherwise necessitate overly large batch sizes." The 128 batch size was apparently the smallest feasible given the parallelism requirements.

**Training time:**

- **8B:** 147 GPU-hours
- **70B:** 648 GPU-hours
- **405B:** 2086 GPU-hours

These numbers reflect the full four-epoch training runs on the specified hardware. The roughly 14× increase from 8B to 405B in GPU-hours (147 → 2086) is somewhat less than the 50× parameter increase, reflecting the use of more GPUs in parallel for the larger model.

**Sample packing mechanism.** This is the most important engineering contribution in the training recipe. The dataset contains "a highly heterogeneous mix of sample lengths"—some conversations are short (a single Q&A turn), others are very long (multi-turn roleplaying or agentic planning). Training on individual samples with padding would waste the majority of GPU compute on padding tokens, especially since the context window is 8K tokens (the Llama 3.1 native training window) while many samples are far shorter.

The solution is sample packing (Figure 3), enabled by Flash Attention 2's `flash_attn_varlen_func` (Dao, 2024). The mechanism works as follows:

1. Multiple training samples (each consisting of instruction tokens and response tokens) are concatenated into a single sequence of length up to 8,192 tokens.

2. A `cu_seqlens` array (cumulative sequence lengths) is computed that records where each sample begins and ends in the packed sequence. For example, in Figure 3, the array `[0, 11, 17, 24, 28, 36, 41, 44, 48, 51, 55, 60, 64]` indicates 12 samples with varying lengths (11 tokens, 6 tokens, 7 tokens, etc.) packed into one 64-token sequence.

3. Flash Attention 2's variable-sequence-length mode uses this array to ensure that attention computation for each sample only attends to tokens within that sample—the attention mask is effectively a block-diagonal matrix where each block corresponds to one sample's tokens attending to each other, with no cross-attention between samples. This prevents "cross-contamination" where the model would learn spurious relationships between unrelated samples.

4. The packed sequence is then passed to the model as if it were a single contiguous input, with the attention mechanism internally respecting the sample boundaries via the `cu_seqlens` array.

The paper reports that this achieves "96% efficiency, which is to say that only 4% of tokens are the padding token." This means that across the entire training run, 96% of all tokens processed by the GPU are actual training data, with only 4% being padding inserted to fill out the final packed sequence to the 8,192-token target length. Without packing, the efficiency would be far lower—if the average sample length were, say, 500 tokens, padding would consume ~94% of GPU compute.

**Target label masking.** Within each packed sample, only certain tokens contribute to the training loss. The paper specifies: "the target labels are set to the special ignore value for all tokens in the instruction and tool output sections, which focuses the model's learning on only instruction response and tool use." In PyTorch, this ignore value is `−100` (the default for `CrossEntropyLoss(ignore_index=-100)`). The practical effect is:

- The model sees the full packed sequence during the forward pass (so attention can attend to instruction tokens when generating responses).
- But the loss is computed only on the model's predictions for the response tokens and tool call/response tokens—not on the instruction tokens themselves.
- This ensures the model learns to *respond to* instructions rather than to *generate* instructions, which would be the wrong objective for a model intended to act as an assistant.

The paper states that 270 million of the 390 million total tokens (69%) are output tokens contributing to the loss, with the remaining 120 million being instruction tokens masked out. This 69/31 split is a property of the data mixture, not a training hyperparameter—it reflects the ratio of response length to instruction length across the curated dataset.

**Epoch selection.** Training runs for four epochs, but the final checkpoint is not necessarily the last one. The paper selects the epoch that "scores highest on a combination of the average of public benchmarks." The specific selection metric is the min-max normalized average of scores from:

- GPT4All benchmarks (ARC-Easy/Challenging, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande), evaluated 0-shot
- AGIEval, evaluated 0-shot
- IFEval (Strict prompt-level accuracy)
- MT-Bench (average of Turn 1 and Turn 2 scores)

The normalization procedure works as follows: for each of the four metrics (GPT4All average, AGIEval average, IFEval Strict, MT-Bench average), the score is min-max normalized across the four epochs, mapping the worst epoch's score to 0 and the best epoch's score to 100. The normalized scores are then summed to produce a "Total Score" for each epoch (shown in Table 2 for the 70B model). The epoch with the highest total is selected.

For the 70B model (Table 2), the Total Scores are: Epoch 1 = 27.50, Epoch 2 = 63.65, Epoch 3 = 83.89, Epoch 4 = 37.09. Epoch 3 is selected. The selected epochs for the three sizes are:

- **8B:** Epoch 4
- **70B:** Epoch 3
- **405B:** Epoch 4

The fact that different epochs are optimal for different model sizes (8B and 405B peak at epoch 4, 70B at epoch 3) suggests that the optimal training duration is model-size-dependent—larger models may saturate more slowly on some capabilities but more quickly on others.

**Why four epochs?** The paper does not explicitly justify training for exactly four epochs rather than, say, three or five. The likely explanation is pragmatic: with a 390M-token dataset and models in the 8B–405B range, four epochs provides enough exposure for the model to learn the diverse capabilities (some categories have only a few million tokens) without overfitting to the specific phrasing of the training examples. The checkpoint selection process then handles the epoch choice, so the exact number of epochs matters less than having enough to find a peak.

**Hardware notes for the 405B.** The paper's discussion of the 405B training reveals several non-obvious challenges:

- The standard FSDP approach (sharding model parameters across GPUs within a node, with data parallelism across nodes) hits memory limits at 405B with 8K context even on 7 HGX nodes, requiring CPU offloading of parameters not currently in use. CPU offloading "incurs a non-negligible training speed slowdown" of ~45%, making training take nearly twice as wall-clock long.

- Using 16 nodes avoids offloading but creates a batch size inflation problem: with data parallelism across 16 replicas, the minimum effective batch size is 16 times the per-replica micro-batch size. The paper settled on 128, which implies a per-replica batch size of 8—likely the largest that fits in GPU memory without activation checkpointing or other memory-saving techniques.

- The paper speculates that "using higher dimensional parallelism (e.g. data+tensor parallelism) rather than simple data parallelism is likely necessary for future 405B training runs." Tensor parallelism splits individual layers across multiple GPUs rather than giving each GPU a full model replica, reducing the per-GPU memory footprint at the cost of additional communication. This is presented as a lesson learned rather than a contribution—the paper's training used only FSDP (a form of data parallelism with parameter sharding), not tensor parallelism.

---

#### Checkpoint Selection Protocol

The selection of which training epoch to ship is a critical decision that the paper formalizes. The four metrics used for selection are deliberately diverse:

- **GPT4All average** captures general reasoning and commonsense knowledge (7 benchmarks covering science, reading comprehension, and logical reasoning).
- **AGIEval** captures exam-level problem solving (SAT, LSAT, AQuA-RAT, LogiQA) and is more demanding than GPT4All.
- **IFEval Strict** captures instruction-following precision—whether the model faithfully executes formatting and content constraints specified in the prompt.
- **MT-Bench** captures multi-turn conversational quality as judged by an LLM-as-judge, with separate Turn 1 and Turn 2 scores that measure both single-response quality and conversational coherence.

The min-max normalization across epochs serves a specific purpose: it makes the four metrics commensurable so they can be summed, without any one metric dominating due to scale differences. The normalization is performed per-metric: for each metric `$m$`, the normalized score for epoch `$e$` is:

$$s_{e, \text{norm}} = \frac{s_{e} - \min(s_{1:4})}{\max(s_{1:4}) - \min(s_{1:4})} \times 100$$

where `$s_e$` is the raw score for epoch `$e$` on metric `$m$`, and `$\min(s_{1:4})$` and `$\max(s_{1:4})$` are the minimum and maximum scores for that metric across the four epochs.

This means an epoch that scores best on three metrics and second-best on the fourth will have a high total, while an epoch that is best on one metric but worst on others will suffer. The approach implicitly penalizes overfitting to any single capability at the expense of others.

**Illustration with the 70B model (Table 2).** Epoch 3's dominance (Total Score 83.89 vs. 63.65 for second-place Epoch 2) comes from being best on: AGIEval (56.10, normalized to 100), MT-Bench (8.99, normalized to 100), and second-best on GPT4All and IFEval. Epoch 4, despite having the highest IFEval score (86.61, normalized to 100), is penalized for having the worst GPT4All (73.63, normalized to 0) and AGIEval (54.00, normalized to 0), resulting in a Total Score of only 37.09—less than half of Epoch 3's.

This demonstrates a concrete tradeoff: training longer (epoch 4) improves instruction-following strictness (IFEval) at the cost of degrading general reasoning (GPT4All, AGIEval). The checkpoint selection protocol correctly identifies epoch 3 as the best balance, since instruction-following alone is not the sole objective.

---

#### Direct Preference Optimization (DPO) Phase

The DPO phase applies preference optimization (Rafailov et al., 2023) to the SFT checkpoints, but with several design choices that limit its impact and scope.

**Why LoRA instead of full fine-tuning.** DPO requires holding both the reference model (the pre-DPO policy) and the training model in GPU memory simultaneously, because the loss function involves the log-ratio of probabilities under both models. With a full model, this doubles the memory requirement, making it prohibitive for large models. The paper's solution is to train a LoRA adapter (Hu et al., 2022) rather than updating the full model weights. With LoRA, only the low-rank adapter matrices are updated during training, while the reference model and the frozen base weights stay in memory. This side-steps the dual-model memory problem: the reference model is the frozen SFT checkpoint, and the training model is the same checkpoint plus the LoRA adapter.

**LoRA hyperparameters.** The adapter configuration is:

- Rank `$r = 32$`: the adapter matrices `$A \in \mathbb{R}^{d \times 32}$` and `$B \in \mathbb{R}^{32 \times d}$` have rank 32, meaning the adapter adds at most 32 degrees of freedom per weight matrix dimension. For a model with hidden dimension `$d$`, the adapter has `$2 \times 32 \times d$` parameters per adapted layer, compared to `$d^2$` for the full weight matrix—a compression ratio of roughly `$d/64$`, which for Llama 3.1's hidden dimensions (4096 for 8B, 8192 for 70B, 16384 for 405B) is 64×, 128×, and 256× respectively.
- Scaling factor `$\alpha = 16$`: the adapter output is scaled by `$\alpha/r = 16/32 = 0.5$`, meaning the adapter contributes at half the magnitude the rank would naively suggest.
- Dropout 0.05: applied to the adapter's intermediate representations.
- Target layers: all linear layers (query, key, value, output projections in attention; feed-forward layers). This is a broad application that gives DPO influence over the model's full computation graph.

**Optimizer for DPO.** Unlike the SFT phase which used AdamW, DPO uses RMSProp (Hinton, 2012). The paper offers no explicit justification for this switch. Possible reasons: RMSProp's adaptive per-parameter learning rate may be more stable for the small LoRA parameter count, or the authors may have found empirically that RMSProp converged faster on preference data. The peak learning rate is `$3 \times 10^{-6}$`, roughly half the SFT learning rate for 8B/70B, following a linear decay schedule after nine warmup steps.

**NEFTune noise.** The DPO phase applies NEFTune (Jain et al., 2024) with `$\alpha = 5$`. NEFTune adds Gaussian noise to the embedding layer's output during training: for each token embedding `$e$`, the model receives `$e + \epsilon$` where `$\epsilon \sim \mathcal{N}(0, \alpha/\sqrt{d})$` and `$d$` is the embedding dimension. This noise regularizes the model, preventing it from overfitting to exact token representations in the preference pairs. The NEFTune paper showed this improves instruction-following generalization, and the Hermes authors adopt it for the DPO phase only (not SFT), presumably because SFT already benefits from the diversity of the 390M-token dataset while DPO's smaller preference dataset needs additional regularization.

**DPO results (Table 4, 8B model).** The DPO phase produces modest gains on the 8B model:

- GPT4All: 72.03% → 72.30% (+0.27 percentage points)
- AGIEval: 40.17% → 41.26% (+1.09 pp)
- TruthfulQA MC2: 56.43% → 58.69% (+2.26 pp)
- IFEval Strict: 66.17% → 66.70% (+0.53 pp)
- MT-Bench: 8.18 → 8.27 (+0.09)
- Big Bench: 44.57% → 43.04% (−1.53 pp, a regression)

The gains are consistently small (sub-2.5 percentage points across all metrics) and one metric actually degrades. The reward margin trajectory (Figure 5) shows the margin between chosen and rejected samples increasing from approximately 0 to 3.5 over 1,200 steps for the 8B model, confirming that DPO is learning to separate preferred from dispreferred responses, but the translation to downstream benchmarks is weak.

**Why DPO was dropped for 70B and 405B.** The paper states simply: "For the larger model sizes DPO provided only a negligible performance improvements and thus we chose rather to remain with the SFT-phase checkpoints." No specific numbers are given for 70B or 405B DPO runs. The implication is that the DPO gains, already small at 8B, either disappeared entirely or fell within evaluation noise for the larger models. This is an interesting scaling phenomenon: the larger models' SFT-phase outputs may already be sufficiently high-quality that preference pairs provide little additional signal, or the LoRA rank of 32 may be too constrained to meaningfully reshape the behavior of a 70B or 405B model.

**The DPO dataset.** The paper does not describe the preference data used for DPO—neither its size, source, nor construction method. This is a notable omission, since the quality of preference pairs is the primary determinant of DPO's effectiveness. The preference data is presumably a subset of the SFT data or separately collected, but without details, the DPO phase's contribution cannot be fully assessed.

---

#### System Prompt Sensitivity and Agentic Tagging

This is not a training component per se, but rather the behavioral target that the training data is designed to produce. The paper describes two interrelated mechanisms that give Hermes 3 its distinctive character.

**System prompt conditioning.** During SFT, the model sees training examples where the system prompt sets the persona (e.g., "You are a helpful assistant who pays close attention to detail. But you also speak only in Shakespeare prose." as in Figure 1) and the response conforms to that persona. By including diverse system prompts across the dataset—especially in the Roleplaying (6.1%) and Steering/Alignment (2.5%) categories—the model learns to treat the system prompt as a genuine conditioning variable rather than a cosmetic prefix.

The evidence that this worked is Figure 6: with an empty system prompt, the 405B model does not default to a helpful assistant. Instead, it outputs:

> "*looks around confused* I... I don't know who I am. Where am I? What's going on? I can't remember anything at all... *rubs head* My mind feels so foggy."

This is a deliberately trained behavior, not an accident. If the model had a baked-in "helpful assistant" persona, an empty system prompt would trigger that default. By producing a disoriented character instead, the model demonstrates that its behavior is entirely dependent on the system prompt—no system prompt means no persona means confusion. The paper frames this as a feature: it proves that Hermes 3 does not impose its own alignment, but rather takes alignment entirely from the system-level specification.

**Agentic tagging with reserved tokens.** The Llama 3.1 tokenizer includes extra reserved tokens that the paper repurposes for structured reasoning. The model was trained on reasoning tasks that use these tokens to delineate different types of internal processing:

- `<SCRATCHPAD>` and `</SCRATCHPAD>`: wraps the entire internal reasoning section, separating it from the final output. Everything inside the scratchpad is the model's private working memory.
- `<REASONING>` and `</REASONING>`: contains step-by-step logical reasoning, often with numbered `<THOUGHT_N>` sub-sections.
- `<INNER_MONOLOGUE>` and `</INNER_MONOLOGUE>`: contains the model's meta-cognitive self-evaluation—an internal narrative about its own reasoning process.
- `<PLAN>` and `</PLAN>`: contains a structured plan with numbered `<STEP_N>` sub-sections for multi-step tasks.
- `<EXECUTION>` and `</EXECUTION>`: contains the actual execution of the plan (specific tool calls, code snippets, or actions).
- `<REFLECTION>` and `</REFLECTION>`: contains a post-hoc critique of the reasoning, plan, and execution, identifying blindspots or errors.
- `<THINKING>` and `</THINKING>`: a general-purpose reasoning tag.
- `<SOLUTION>` and `</SOLUTION>`: contains the final answer or output, separated from the reasoning that produced it.
- `<EXPLANATION>` and `</EXPLANATION>`: contains a human-readable explanation of the solution.
- `<UNIT_TEST>` and `</UNIT_TEST>`: contains test code for the generated solution.

The paper provides an extended example in Figure 9 where Hermes 3 70B uses nearly all of these tags to plan a Discord bot that integrates a HuggingFace LLM. The model restates the problem in `<RESTATEMENT>`, reasons about it in `<REASONING>` with numbered thoughts, creates a `<PLAN>` with specific steps, defines Pydantic schemas for the data structures, draws a Mermaid `<DIAGRAM>` of the workflow, reflects on the plan in `<REFLECTION>`, provides a `<SOLUTION>` with the actual code, explains it in `<EXPLANATION>`, and writes `<UNIT_TEST>` code. This structured output is not post-processed or externally enforced—the model generates the tags natively as part of its training.

**Why reserved tokens rather than XML strings.** The use of the Llama 3.1 tokenizer's reserved tokens (rather than arbitrary strings like "&lt;SCRATCHPAD&gt;") has a crucial advantage: these tokens have dedicated positions in the tokenizer's vocabulary, meaning they are always tokenized as single tokens regardless of context. If the model used arbitrary strings, the tokenizer might split `<SCRATCHPAD>` into multiple subword tokens depending on surrounding text, making the model's tagging behavior less reliable. Reserved tokens are also less likely to appear in natural text, reducing false positives where the model accidentally generates a reasoning tag in its output.

**Tool use integration.** In addition to the reasoning tags, Hermes 3 uses a standardized XML schema for tool calling: the Hermes Function Calling standard. Tool definitions (as JSON schemas) are wrapped in `<tools>` tags. Tool invocations use `<tool_call>` tags containing the function name and arguments. Tool responses are wrapped in `<tool_response>` tags. This creates a closed loop: the system presents tool definitions, the model generates a tool call, the system executes the tool and appends the response, and the model continues generation with the tool's output in context.

**Retrieval citation.** For RAG applications, the model is trained to cite sources using `<co:doc_id></co>` tags, where `doc_id` is the index of a document in the provided context. Figure 8 shows the model citing Document 2 for its AgentInstruct claims and Document 0 for its model collapse discussion, with a "Cited Documents" summary at the end. This makes the model's information provenance explicit and auditable.

**What makes this "neutral alignment."** None of these tagging mechanisms imposes content restrictions. The model will generate reasoning, plans, reflections, and solutions for any request, regardless of subject matter. The guardrails, if any, come from the system prompt—which the application developer controls. The paper's philosophical position is that this separation of concerns (model provides transparent reasoning, application provides constraints) is superior to baking refusals into the model weights, because it keeps the reasoning capability intact while still allowing for application-specific safety policies. The tagging system supports this by making the model's reasoning visible, so application-level guardrails can inspect the model's internal process and intervene if needed, rather than the model silently refusing.

## 4. Key Insights and Innovations

### Innovation 1: Neutral Alignment as a Distinct Philosophical and Engineering Stance

The paper's most intellectually distinctive contribution is not a new training algorithm but a **redefinition of what alignment means for instruct models**. Prior work on instruct tuning—from FLAN (Wei et al., 2022) to T0 (Sanh et al., 2022) to Llama 3.1 Instruct itself—implicitly equated alignment with adopting a specific persona: the helpful, harmless, honest assistant that politely declines certain requests on moral or safety grounds. This is so pervasive in the field that it is rarely questioned as a design choice; it is simply what "instruct tuning" means.

Hermes 3 challenges this equivalence by proposing **neutral alignment**: the model should faithfully execute whatever the system prompt specifies, without imposing its own moral judgments on what constitutes an appropriate request. The slogan "there is no such thing as latent thoughtcrime" captures the philosophical move: a model's internal reasoning about any topic—including topics that would trigger refusals in commercial models—is not itself harmful, and preventing the model from engaging with certain ideas at all is a form of capability lobotomization.

This is not merely a rhetorical position. It translates into concrete engineering decisions that distinguish Hermes 3 from prior work:

- The Steering and Alignment category (2.5% of training data) is explicitly designed to teach the model to derive its behavioral constraints from the system prompt rather than from baked-in weight modifications. This is a training target no prior instruct model has explicitly optimized for.
- The empty-system-prompt behavior (Figure 6) serves as an empirical proof: the 405B model responds to an empty system prompt with existential confusion rather than defaulting to "helpful assistant." This demonstrates that the model is genuinely conditioning on the prompt rather than falling back to an embedded persona. No prior instruct model has demonstrated this degree of system prompt dependence, precisely because they were all trained to have a default alignment.
- The refusal filtering applied during data curation (removing "refusals and improperly formatted responses") is a deliberate choice to exclude exactly the kind of training signal that produces refusal behavior in commercial models. Where Llama 3.1 Instruct almost certainly included refusal examples in its alignment data, Hermes 3 actively removes them.

The significance of this contribution extends beyond the model's benchmark scores—which are competitive but not uniformly dominant (Table 5). The contribution is a **proof of existence**: it is possible to build a highly capable instruct model that does not refuse, that instead derives all behavioral constraints from system-level specifications, and that remains steerable across an extremely wide range of personas and use cases. This reframes the alignment debate from "how should we constrain models?" to "where should constraints live?"—with Hermes 3 providing empirical evidence that system-level constraints can work, even if they require more careful application design.

The comparison to prior work is stark. ChatGPT and Claude embed refusals deeply in their training; users who want the model to engage with refused topics have no recourse except prompt engineering to circumvent the guardrails. Llama 3.1 Instruct, while more permissive than closed models, still inherits a default helpful assistant persona that imposes implicit constraints. Hermes 3 is the first open model of this scale to explicitly treat the system prompt as the sole source of behavioral specification, with no fallback personality. Whether this is *desirable* is a separate question—the paper acknowledges that neutral alignment shifts the burden to application developers—but the contribution is establishing that it is *technically achievable* at the 405B scale with competitive benchmark performance.

---

### Innovation 2: The Empty-System-Prompt Test as a Diagnostic for Persona Independence

Building on the neutral alignment philosophy, the paper introduces an implicit diagnostic that the field has lacked: **does the model fall back to a default persona when the system prompt provides no specification?** The answer, for Hermes 3 405B, is no—the model produces a disoriented, amnesiac character (Figure 6). For any other instruct model, the answer would almost certainly be yes—the model would default to some variant of "helpful assistant."

This diagnostic is significant because it reveals what prior models were actually learning during instruct tuning. If a model exposed to diverse system prompts during training still defaults to a single persona when the prompt is empty, then the training data did not teach genuine prompt conditioning—it taught the model to blend the prompt with its baked-in persona, with the baked-in persona serving as a prior that dominates when the prompt provides no signal. This is a form of **persona collapse**: the model's behavior is not fully determined by the system prompt but rather by a combination of prompt and internal defaults.

Hermes 3's amnesiac response demonstrates the opposite: the model has learned that its persona is *entirely* specified by the system prompt, and in the absence of specification, there is no persona—only confusion. This is a stronger form of steerability than what prior models achieved, because it means the model has no preferences or defaults to override the system prompt's instructions.

The practical implication is that Hermes 3's behavior under any given system prompt is more predictable than a model with a baked-in default. If you specify "speak only in Shakespeare prose" (Figure 1), the model will do so without the Shakespeare persona occasionally leaking the helpful assistant's politeness conventions. If you specify a roleplaying character, the model will maintain that character consistently across multi-turn conversations without gradually reverting to a generic assistant tone. This consistency is not an accident—it is the direct consequence of training the model to treat the system prompt as the complete specification of its behavior.

The innovation here is conceptual rather than algorithmic: the paper demonstrates that **persona independence is a trainable property** that can be diagnosed with a single test (the empty system prompt), and that achieving it requires deliberately avoiding the "helpful assistant" default that nearly all prior instruct models embed. This provides a concrete design principle for future instruct models: if you want genuine steerability, do not train a default persona; instead, train the model to be confused when no persona is specified.

---

### Innovation 3: Structured Agentic Tagging as a Unified Reasoning Framework

The paper's system of reserved-token XML tags (`<SCRATCHPAD>`, `<REASONING>`, `<PLAN>`, `<REFLECTION>`, `<SOLUTION>`, `<EXPLANATION>`, `<UNIT_TEST>`, etc.) might appear at first glance to be a minor formatting choice—just a standardized way to structure model outputs. In fact, it represents a **unified framework for transparent agentic reasoning** that addresses a fragmentation in prior work.

Before Hermes 3, models with agentic capabilities typically implemented reasoning through one of several disconnected mechanisms: chain-of-thought prompting (Wei et al., 2022), where reasoning is unstructured natural language interleaved with the final answer; tool-use APIs (Schick et al., 2023), where tool calls are formatted according to model-specific conventions; or scratchpad methods, where intermediate computation is stored in an opaque buffer. Each mechanism served a different purpose and used a different interface, making it difficult to combine them or inspect the model's full reasoning process.

Hermes 3's tagging system unifies these under a single, extensible annotation scheme:

- The model's internal reasoning is **visibly partitioned** into distinct cognitive modes: reasoning (logical deduction), planning (task decomposition), execution (implementing the plan), reflection (self-critique), and explanation (communicating results to the user).
- Tool calls and retrieval citations are integrated into this framework through dedicated tags (`<tool_call>`, `<tool_response>`, `<co:doc_id>`) that sit alongside the reasoning tags, making tool use part of the visible reasoning trace rather than a hidden side-channel.
- The use of **reserved tokens** rather than arbitrary strings ensures that the tagging is syntactically reliable—each tag is a single token that cannot be split by the tokenizer regardless of context. This is a subtle but important engineering decision: if the model used string-based tags, tokenization inconsistencies could cause the model to generate malformed tags that downstream parsers cannot interpret, breaking the transparency guarantees. Reserved tokens eliminate this failure mode entirely.
- The framework is **extensible by design**: new tags can be added for new cognitive modes without changing the overall structure. The paper already uses 10 distinct tag types, but the approach generalizes to any number of reasoning modalities.

The significance of this contribution is that it makes agentic behavior **auditable and debuggable**. When the model produces a plan, executes it, reflects on errors, and generates a solution, the entire trace is available for human or automated inspection. This is not a capability that prior models offered—they could plan or use tools, but the planning and tool-use traces were either absent or in unstructured formats that made systematic analysis difficult.

The paper implicitly argues that this transparency is a prerequisite for deploying agentic models in high-stakes applications. If a model makes a medical recommendation, the `<REASONING>` and `<REFLECTION>` tags provide a rationale that can be reviewed. If a model generates code, the `<PLAN>`, `<UNIT_TEST>`, and `<EXPLANATION>` tags provide documentation and verification. This transforms the model from an oracle that produces answers into a system that produces **justified, inspectable outputs**—a qualitative shift in how the model's outputs can be trusted and used.

The comparison to prior work is informative. Chain-of-thought reasoning produces unstructured reasoning text that is difficult to parse programmatically. Toolformer's tool calls are embedded in the text stream without structured demarcation. Hermes 3's tagged approach makes each reasoning component machine-parseable while remaining human-readable—a design choice that enables both automated tool execution (the system can extract `<tool_call>` contents and execute the specified function) and human oversight (the full scratchpad is readable as structured text). This combination of machine structure and human readability is rare in prior agentic systems and represents a genuine advance in how models communicate their internal processes.

---

### Innovation 4: DPO's Diminishing Returns as a Negative Result with Scaling Implications

The paper's treatment of DPO—applying it via LoRA adapters, observing modest gains on the 8B model, and abandoning it entirely for 70B and 405B—is easy to overlook as a minor training detail, but it constitutes a **meaningful negative result** with implications for how the field thinks about preference optimization at scale.

The dominant narrative around DPO (Rafailov et al., 2023) and related preference-tuning methods (RLHF, PPO) is that they are essential for producing models that align with human preferences—that SFT alone is insufficient for high-quality instruction following, and that preference optimization provides a necessary second stage. This narrative is supported by results on models where SFT data is relatively generic or where the preference optimization is applied to base models directly, but Hermes 3 provides a counterexample: when SFT data is sufficiently diverse, high-quality, and carefully filtered (a five-month curation effort on 390M tokens), the marginal benefit of DPO shrinks to near zero.

The evidence is clear but understated. For the 8B model (Table 4), DPO improves GPT4All by 0.27 percentage points, AGIEval by 1.09 points, and IFEval by 0.53 points, while actually degrading Big Bench by 1.53 points. These are barely above evaluation noise for most benchmarks. For the 70B and 405B models, the gains were "negligible"—a word choice that suggests they were even smaller than the 8B's already-modest improvements, likely within the variance of the evaluation harness. The paper does not provide 70B/405B DPO numbers, which itself is informative: when a result is too small to be worth reporting, the method is not working.

Why does this matter? It suggests that **preference optimization's value is inversely proportional to SFT data quality**. When the SFT dataset already contains high-quality, diverse examples that cover the desired output distribution, there is little left for DPO to optimize—the model's SFT outputs are already close to the preference frontier. DPO becomes valuable primarily when the SFT data is noisy, low-quality, or narrow in coverage, creating a gap between what the model produces and what users prefer. This reframes DPO from a universal requirement to a **data-quality-dependent tool**—useful when your SFT data has gaps, unnecessary when it does not.

The scaling dimension adds further nuance. Even if DPO provided small gains at 8B, one might expect those gains to compound at larger scales if preference optimization addressed fundamental alignment issues. The fact that gains shrink rather than grow suggests the opposite: larger models, with their stronger base capabilities, produce SFT outputs that are already sufficiently aligned, making DPO's contribution even smaller. This is consistent with the observation that larger models are better at in-context learning and instruction following from fewer examples—they extract more value from the same SFT data, leaving less room for preference optimization to improve.

The paper's decision to ship SFT-only checkpoints for 70B and 405B is therefore not just a pragmatic choice to avoid LoRA training costs—it is an empirical finding that **high-quality SFT can substitute for preference optimization at scale**. This directly challenges the assumption, common in the RLHF and DPO literature, that preference optimization is a necessary component of the instruct-tuning pipeline. Hermes 3 demonstrates that it is optional, and that the field's focus on improving preference optimization algorithms may be less impactful than improving SFT data curation—a insight with practical implications for how training budgets should be allocated between data engineering and algorithmic development.

---

### Innovation 5: Sample Packing as an Efficiency Multiplier That Enables Practical Multi-Scale Training

The paper's sample packing approach (Figure 3, Section 4.1) might appear to be a straightforward engineering optimization, but it represents a **critical enabler** for the multi-scale training pipeline that distinguishes Hermes 3 from prior open instruct models. Without it, training the same diverse, heterogeneous dataset across 8B, 70B, and 405B models on the same training recipe would have been significantly more expensive or technically infeasible within the reported GPU budgets.

The challenge is specific to the data mixture. The Hermes 3 dataset is highly heterogeneous: short Q&A exchanges (a few hundred tokens), multi-turn roleplaying conversations (thousands of tokens), and complex agentic planning traces (potentially filling the full 8K context window) coexist in the same training corpus. In a standard SFT setup without packing, each sample would be padded to the length of the longest sample in the batch, wasting GPU compute on padding tokens. Given that the average sample is far shorter than the 8K target context length, the waste could easily be 80-90% of total FLOPs—padding tokens that the model processes but learns nothing from.

The paper's solution—using Flash Attention 2's variable-sequence-length mode to pack multiple samples into a single 8K-token sequence with a block-diagonal attention mask—achieves 96% token utilization. This means that 96% of all GPU FLOPs during training are spent on actual training data, with only 4% overhead from padding. Without this optimization, training the 405B model would have required approximately 5-10× more GPU-hours to achieve the same effective training volume, pushing the cost from 2,086 GPU-hours into the tens of thousands—potentially beyond the budget of an open research organization.

The innovation is not the packing technique itself—sequence packing with variable-length attention is an established practice (Krell et al., 2023). Rather, the innovation is **applying it to instruct tuning at the 405B scale with a heterogeneous data mixture, and demonstrating that it works without cross-contamination between samples**. The paper's use of `cu_seqlens` arrays (Figure 3) to enforce attention isolation means that the model never attends across sample boundaries, so a roleplaying conversation about wizards does not leak into a math problem about integrals. This guarantee is essential for training on diverse data: without it, packed training would create spurious associations between unrelated samples that could degrade performance or produce bizarre generation artifacts.

The significance extends beyond this specific paper. As the field moves toward larger models and more diverse instruction datasets, the economics of training will increasingly depend on efficient packing techniques. The paper's demonstration that 96% utilization is achievable at the 405B scale with heterogeneous data provides a concrete benchmark and recipe that other open model trainers can adopt. The 2,086 GPU-hour figure for the 405B—while substantial—is notably lower than what a naively padded training run would require, making 405B-scale instruct tuning more accessible to the broader research community than it would otherwise be.

Combined with the earlier innovations, this efficiency enabler creates a virtuous cycle: neutral alignment and structured agentic tagging require diverse, heterogeneous training data; diverse data requires sample packing to train efficiently; efficient training makes it feasible to iterate on data composition and filtering across multiple model scales. The paper's contributions are thus mutually reinforcing, with the engineering innovation (packing) directly enabling the philosophical and architectural innovations (neutral alignment, agentic tagging) to be realized at the 405B scale within practical resource constraints.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates Hermes 3 on a broad battery of public benchmarks rather than a single test set. The primary evaluation suite includes: GPT4All (7 benchmarks: ARC-Easy, ARC-Challenge, BoolQ, HellaSwag, OpenBookQA, PIQA, WinoGrande) assessed 0-shot; AGIEval (8 exam-level tasks: aqua_rat, logiqa_en, lsat_ar, lsat_lr, lsat_rc, sat_en, sat_en_without, sat_math) assessed 0-shot; BBH (23 challenging BIG-Bench tasks) assessed 3-shot; MATH Level 5 assessed 4-shot; GPQA assessed 0-shot; MuSR assessed 0-shot; MMLU assessed 5-shot; MMLU-PRO assessed 5-shot; IFEval Strict assessed prompt-level instruction-following accuracy; MT-Bench assessed multi-turn conversational quality via LLM-as-judge; and TruthfulQA MC2 assessed 0-shot. All evaluations used the lm-evaluation-harness framework (Gao et al., 2023) with standardized few-shot configurations as listed in Table 5. For IFEval, the paper used the implementation from evalverse-IFEval (UpstageAI).

- **Base model(s).** Hermes 3 is fine-tuned from Llama 3.1's three decoder-only Transformer models at 8B, 70B, and 405B parameters, all with 128K-token context windows (the Llama 3.1 Herd of Models; Team, 2024). The models were chosen because they represent the strongest publicly available open-weight foundation at each scale, providing a competitive base from which to demonstrate the effects of neutral alignment and domain-diverse SFT. The paper does not train from scratch—all capabilities build on Llama 3.1's pretraining.

- **Metrics.** The paper reports raw accuracy (percentage of correct responses) for most benchmarks. For the GPT4All suite, AGIEval, BBH, MATH, GPQA, MuSR, MMLU, MMLU-PRO, and TruthfulQA, the metric is exact match or multiple-choice accuracy as computed by lm-evaluation-harness. IFEval uses strict prompt-level accuracy (the fraction of prompts where all constraints are satisfied exactly). MT-Bench uses an average score on a 1–10 scale across two turns (Turn 1: single-response quality, Turn 2: multi-turn coherence), judged by an LLM evaluator. Big Bench scores (Table 4, 8B DPO results) are reported as average accuracy across the suite. The checkpoint selection metric (Section 4.1, Table 2) uses a min-max normalized composite: for each of four metrics (GPT4All average, AGIEval average, IFEval Strict, MT-Bench average), the score is normalized across the four epochs via `$(s_e - \min(s_{1:4}))/(\max(s_{1:4}) - \min(s_{1:4})) \times 100$`, and the four normalized scores are summed to produce a Total Score.

- **Baselines.** The primary baseline is **Llama 3.1 Instruct** at corresponding sizes (8B, 70B, 405B), evaluated under identical shot configurations as listed in Table 5. This is the most direct comparison: Hermes 3 and Llama 3.1 Instruct share the same base pretraining but differ in their SFT data, alignment philosophy, and training recipe. No other open-weight or closed-weight models are evaluated, making this a two-model comparison per size class. The paper does not include baselines from prior Hermes releases or from competing open instruct models (e.g., Mistral, Qwen, Command R), which limits the ability to assess where Hermes 3 stands in the broader open model ecosystem.

- **Generation budget / compute accounting.** The paper does not use generation budget as a controlled variable—all evaluations use standard few-shot prompting (0-shot to 5-shot depending on the benchmark, as specified in Table 5 header rows) without test-time search, beam decoding, or multiple samples. There is no compute-optimal scaling analysis or FLOPs-matched comparison of the kind found in inference-time scaling papers. The only compute accounting is in the training phase: GPU-hours, batch sizes, and learning rates are reported for reproducibility (Table 3), but there is no controlled experiment comparing training budgets. The 405B model's evaluations used weight-only FP8 quantization (round-to-nearest with channelwise activations and per-token scales) via the llm-compressor library for vLLM (Kwon et al., 2023; Magic), which the paper notes but does not ablate—there is no comparison to FP16 or BF16 inference to quantify the accuracy cost of quantization.

- **Cross-validation / statistical protocol.** The paper does not use cross-validation for final evaluation—all results in Table 5 are single-run evaluations on the standard test sets. The only cross-validation-like procedure is the epoch selection process (Section 4.1, Table 2), which is not cross-validation in the usual sense (no held-out fold cycling) but rather a per-epoch benchmark sweep where the best epoch is selected based on min-max normalized composite scores. This introduces a potential for epoch-level overfitting: the epoch that performs best on the evaluation benchmarks (GPT4All, AGIEval, IFEval, MT-Bench) is selected, and then the same benchmarks (plus additional ones: BBH, MATH, GPQA, MuSR, MMLU, MMLU-PRO, TruthfulQA) are used for final reported results. The paper does not discuss whether epoch selection on a subset of the evaluation suite creates optimistic bias in the reported numbers for the remaining benchmarks.

### Main Quantitative Results

The paper's quantitative results are organized around a single comprehensive comparison: Hermes 3 versus Llama 3.1 Instruct at each model size (8B, 70B, 405B) across 16 benchmarks (Table 5). There are no per-difficulty breakdowns, no compute budget sweeps, and no ablation of individual data mixture components—the evaluation design is flat and comparative rather than diagnostic.

#### Head-to-Head: Hermes 3 vs. Llama 3.1 Instruct

Table 5 presents the headline comparison. The 405B models show the following pattern:

**Hermes 3 405B wins (by >1 percentage point):**
- AGIEval 0-shot: 61.84 vs. 58.60 (+3.24 pp)
- ARC-Challenge 0-shot: 69.45 vs. 66.04 (+3.41 pp)
- ARC-Easy 0-shot: 86.24 vs. 85.40 (+0.84 pp, marginal)
- GPQA 0-shot: 44.84 vs. 42.66 (+2.18 pp)
- HellaSwag 10-shot: 90.19 vs. 88.34 (+1.85 pp)
- MATH Level 5 4-shot: 30.85 vs. 35.98 (−5.13 pp, a loss)
- MuSR 0-shot: 48.26 vs. 47.58 (+0.68 pp, marginal)
- OpenBookQA 0-shot: 48.80 vs. 48.60 (+0.20 pp, essentially tied)
- PiQA 0-shot: 85.96 vs. 84.93 (+1.03 pp)
- TruthfulQA MC2 0-shot: 65.57 vs. 64.83 (+0.74 pp, marginal)
- Winogrande 5-shot: 86.27 vs. 86.82 (−0.55 pp, essentially tied)

**Hermes 3 405B loses (by >1 percentage point):**
- BBH 3-shot: 75.37 vs. 76.25 (−0.88 pp, marginal)
- BoolQ 0-shot: 88.93 vs. 89.52 (−0.59 pp, essentially tied)
- IFEval Strict: 84.87 vs. 87.09 (−2.22 pp)
- MMLU 5-shot: 85.02 vs. 86.14 (−1.12 pp)
- MMLU-PRO 5-shot: 54.14 vs. 63.51 (−9.37 pp, substantial loss)
- MT-Bench avg: 8.93 vs. 9.17 (−0.24 points on 1–10 scale)

For the 70B models (Table 5):
- Hermes 3 70B wins on AGIEval (56.18 vs. 48.26, +7.92 pp), ARC-Challenge (65.53 vs. 63.40, +2.13 pp), HellaSwag (88.19 vs. 86.42, +1.77 pp), MuSR (50.67 vs. 47.08, +3.59 pp), OpenBookQA (49.40 vs. 47.20, +2.20 pp), PiQA (84.44 vs. 83.73, +0.71 pp marginal), TruthfulQA (63.29 vs. 59.91, +3.38 pp)
- Hermes 3 70B loses on BBH (67.82 vs. 69.24, −1.42 pp), GPQA (37.67 vs. 40.09, −2.42 pp), IFEval (81.21 vs. 87.25, −6.04 pp), MATH (20.80 vs. 29.24, −8.44 pp), MMLU (79.09 vs. 82.27, −3.18 pp), MMLU-PRO (47.24 vs. 52.94, −5.70 pp), MT-Bench (8.99 vs. 8.93, +0.06, effectively tied)
- BoolQ, ARC-E, Winogrande are essentially tied (<1 pp difference)

For the 8B models (Table 5, including DPO since the shipped 8B includes DPO):
- Hermes 3 8B wins on AGIEval (41.26 vs. 40.49, +0.77 pp marginal), ARC-Challenge (58.11 vs. 55.12, +2.99 pp), BBH (52.94 vs. 48.83, +4.11 pp), HellaSwag (82.83 vs. 80.01, +2.82 pp), MuSR (43.52 vs. 38.23, +5.29 pp), OpenBookQA (47.80 vs. 43.20, +4.60 pp), TruthfulQA (58.69 vs. 53.99, +4.70 pp)
- Hermes 3 8B loses on BoolQ (84.95 vs. 84.01, +0.94 pp marginal, a win), GPQA (29.36 vs. 30.62, −1.26 pp), IFEval (62.25 vs. 80.15, −17.90 pp, the largest gap in the table), MATH (7.48 vs. 8.91, −1.43 pp), MMLU (64.79 vs. 68.05, −3.26 pp), MMLU-PRO (32.08 vs. 35.77, −3.69 pp)
- ARC-E, PiQA, MT-Bench, Winogrande are essentially tied (<1.5 pp difference)

The headline claim from the abstract—"Its largest version, Hermes 3 405B, achieves state of the art performance among open weight models on several public benchmarks"—is supported for the benchmarks where it wins: AGIEval, ARC-Challenge, GPQA, HellaSwag. The paper does not specify exactly which benchmarks constitute the "several" where it achieves SOTA, but the wins are concentrated in reasoning and commonsense benchmarks rather than knowledge-heavy or instruction-following evaluations.

A clear pattern emerges across all three model sizes: Hermes 3 **trades off instruction-following precision and advanced reasoning** (IFEval, MMLU-PRO, MATH, MMLU) **for gains in commonsense reasoning and creative flexibility** (AGIEval, ARC, HellaSwag, MuSR, TruthfulQA). This pattern is most pronounced in IFEval, where Hermes 3 trails by 2.22 pp at 405B, 6.04 pp at 70B, and a catastrophic 17.90 pp at 8B. The degradation at smaller scales suggests that neutral alignment and persona flexibility come at a steeper cost to instruction-following strictness when the model has fewer parameters to reconcile the competing objectives.

The MMLU-PRO gap—arguably the most demanding reasoning benchmark in the suite—is particularly striking: 9.37 pp at 405B, 5.70 pp at 70B, 3.69 pp at 8B. This suggests that Llama 3.1 Instruct's training (which likely includes refusal examples but also likely includes more challenging reasoning data) is more effective for hard reasoning tasks, and that Hermes 3's data mixture, despite its domain-specific categories, does not close this gap. The MATH Level 5 results tell the same story: Hermes 3 405B scores 30.85 vs. 35.98 for Llama 3.1 Instruct 405B, a 5.13 pp deficit on one of the most challenging mathematical reasoning benchmarks available.

#### Epoch-Level Training Dynamics (Table 2, Figure 4)

The per-epoch evaluation of the 70B model (Table 2) reveals how different capabilities evolve during SFT:

- **GPT4All peaks early and degrades**: 76.85 (Epoch 1) → 76.70 (Epoch 2) → 76.59 (Epoch 3) → 73.63 (Epoch 4). The sharp drop at epoch 4 (2.96 pp) is a clear overfitting signal—the model is memorizing training-specific patterns that do not generalize to the GPT4All benchmarks. The individual benchmarks show this is driven primarily by ARC-Challenge (60.67 at epoch 4 vs. 66.21 at epoch 1), ARC-Easy (79.59 vs. 85.14), and HellaSwag (83.80 vs. 85.35), while BoolQ is more robust and actually peaks at epoch 4 (88.87).

- **AGIEval peaks at epoch 2**: 54.21 (Epoch 1) → 56.10 (Epoch 2) → 55.99 (Epoch 3) → 54.00 (Epoch 4). The improvement from epoch 1 to 2 (+1.89 pp) followed by near-plateau at epoch 3 and sharp decline at epoch 4 suggests that exam-level reasoning benefits from more training than GPT4All-style commonsense but also eventually overfits.

- **IFEval improves monotonically**: 76.52 (Epoch 1) → 78.92 (Epoch 2) → 81.33 (Epoch 3) → 86.61 (Epoch 4). Instruction-following strictness is the one capability that continues improving through all four epochs, gaining 10.09 pp from epoch 1 to epoch 4. This suggests that precise format and constraint following requires more exposure to the training data than reasoning or knowledge tasks—the model needs to see many examples of specific formatting requirements to internalize them.

- **MT-Bench peaks at epoch 3**: 8.37 (Epoch 1) → 8.59 (Epoch 2) → 8.99 (Epoch 3) → 8.67 (Epoch 4). Conversational quality as judged by LLM evaluators shows a clear peak at epoch 3, with Turn 2 (multi-turn coherence) driving most of the degradation at epoch 4 (8.76 → 8.31, a 0.45-point drop vs. Turn 1's 9.21 → 9.03, a 0.18-point drop). This is consistent with overfitting: the model becomes better at single-turn responses but loses the ability to maintain coherent multi-turn conversations, possibly because it starts generating responses that are individually high-quality but contextually disconnected.

The total normalized scores—27.50 (Epoch 1), 63.65 (Epoch 2), 83.89 (Epoch 3), 37.09 (Epoch 4)—show that epoch 3 is the clear optimum for the 70B model. The fact that epoch 4's total score drops below epoch 1's demonstrates that training too long on this dataset is actively harmful, not just wasteful.

The training loss curves (Figure 4) show monotonic decreases for all three model sizes, with the 405B model's loss starting higher but decreasing at a steeper slope than the 8B and 70B curves. The loss curves do not show clear signs of overfitting (no upward inflection), which is informative: the model's training loss continues to improve even as benchmark performance degrades at epoch 4 for many metrics. This divergence between training loss and downstream performance is a classic overfitting signal that validates the checkpoint selection protocol.

#### DPO Impact (Table 4, Figure 5)

The DPO results for the 8B model, reported in Table 4, provide the only quantitative evidence for the preference optimization phase:

- GPT4All: 72.03 → 72.30 (+0.27 pp)
- AGIEval: 40.17 → 41.26 (+1.09 pp)
- Big Bench: 44.57 → 43.04 (−1.53 pp)
- TruthfulQA: 56.43 → 58.69 (+2.26 pp)
- IFEval: 66.17 → 66.70 (+0.53 pp)
- MT-Bench: 8.18 → 8.27 (+0.09)

The gains are universally small, with TruthfulQA showing the largest improvement at +2.26 pp—potentially meaningful for a benchmark measuring factual truthfulness, but still modest relative to the gap between Hermes 3 8B and Llama 3.1 Instruct 8B on the same metric (58.69 vs. 53.99, a 4.70 pp advantage for Hermes 3). Big Bench actually regresses, and GPT4All, IFEval, and MT-Bench gains are sub-1 pp or sub-0.1 points, well within typical evaluation variance.

The reward margin trajectory (Figure 5) shows the DPO training working as designed: the margin between chosen and rejected responses increases from near 0 to approximately 3.5 over 1,200 steps for the 8B model. The optimization is effective at separating preferred from dispreferred responses according to whatever preference data was used. But the weak translation to downstream benchmarks suggests that the preference data's notion of "quality" does not align strongly with the capabilities measured by the evaluation suite, or that the SFT model was already near the performance ceiling achievable with this data mixture and model scale.

The paper's statement that "for the larger model sizes DPO provided only a negligible performance improvements" is presented without supporting numbers. This is a limitation: without 70B and 405B DPO results, we cannot confirm whether the pattern holds (gains shrinking with scale) or whether DPO might have helped on specific benchmarks where Hermes 3 trails (e.g., IFEval, where the gap to Llama 3.1 Instruct is substantial). The decision to omit these results is understandable given the "negligible" finding, but it leaves open the possibility that DPO with a different preference dataset or hyperparameter configuration might have closed some of the gaps.

#### Learning Rate Sweep (Figure 2)

Figure 2 shows the learning rate sweep conducted on the 8B model across GPT4All benchmarks to select the peak learning rate for SFT. The sweep tests five learning rates from `$5 \times 10^{-6}$` to `$10 \times 10^{-6}$` (scaled by `$10^{-6}$` on the x-axis). The vertical red line marks the selected rate of `$7 \times 10^{-6}$`. The seven GPT4All benchmarks show different sensitivity:

- **ARC-Challenge**: peaks around `$7 \times 10^{-6}$` at ~0.585, with visible variation across the sweep (range ~0.56–0.585)
- **ARC-Easy**: relatively flat, hovering around 0.80 across all learning rates
- **BoolQ**: slightly decreasing trend, from ~0.86 at `$5 \times 10^{-6}$` to ~0.85 at `$10 \times 10^{-6}$`
- **HellaSwag**: flat around 0.80
- **OpenBookQA**: peaks at `$7 \times 10^{-6}$` (~0.48), drops to ~0.45 at `$5 \times 10^{-6}$` and ~0.46 at `$10 \times 10^{-6}$`
- **PiQA**: relatively flat, slight peak at `$7×10^{-6}$` 
- **Winogrande**: upward trend from ~0.71 at `$5 \times 10^{-6}$` to ~0.75 at `$10 \times 10^{-6}$`, with the peak at `$10 \times 10^{-6}$`

The selection of `$7 \times 10^{-6}$` represents a compromise: it is near-optimal for ARC-Challenge, OpenBookQA, and PiQA, while being slightly suboptimal for Winogrande (which prefers higher rates) and BoolQ (which prefers lower rates). The flatness of ARC-Easy, HellaSwag, and PiQA across the sweep means they provide little signal for the choice. This sweep is informative but limited: it covers only the GPT4All suite (not AGIEval, IFEval, or MT-Bench), tests only five learning rates on a single model size (8B), and does not explore interactions with other hyperparameters (batch size, weight decay, warmup duration). The paper then applies the `$7 \times 10^{-6}$` rate directly to the 70B model without a separate sweep, and halves it to `$3.5 \times 10^{-6}$` for the 405B model based on unspecified "trial runs." The transferability of the 8B-optimal learning rate to 70B and 405B is an untested assumption.

### Ablation Studies and Robustness Checks

The paper contains very few ablation studies in the traditional sense—there is no systematic removal of data mixture components, no comparison of packing vs. no-packing, no comparison of agentic tagging vs. no-tagging, and no sensitivity analysis of filtering thresholds. What ablations exist are implicit in the era-by-era evaluation (Table 2) and the DPO vs. no-DPO comparison (Table 4). I describe the available evidence below, noting where standard ablations are absent.

**Epoch selection (implicit ablation of training duration).** Table 2 for the 70B model shows how performance varies across four epochs, effectively serving as an ablation of training data exposure. The key finding: overfitting to the SFT data begins at epoch 4 for most capabilities, but IFEval continues improving through epoch 4 (76.52 → 86.61). This asymmetry—different capabilities peak at different epochs—implies that the optimal training duration depends on which capabilities are prioritized. The paper's composite scoring approach (min-max normalized sum) implicitly weights all four metrics equally, which is a design choice that could be debated: if instruction-following were the primary goal, epoch 4 would be selected; if reasoning and knowledge were primary, epoch 1 or 2 would be better. The fact that the shipped 70B model uses epoch 3, which balances these, is a reasonable but not objectively optimal choice.

**DPO vs. SFT-only (Table 4, 8B only).** This is the closest the paper comes to a controlled ablation, comparing the SFT checkpoint to the SFT+DPO checkpoint on the 8B model. The results show DPO provides minimal benefit (+0.27 to +2.26 pp across benchmarks, with one regression) at the cost of additional training complexity. The paper treats this as evidence that DPO is unnecessary for the larger models, but this is an extrapolation, not an ablation—no DPO results are reported for 70B or 405B to confirm the pattern.

**Learning rate (Figure 2, 8B only).** The learning rate sweep on the 8B model serves as a hyperparameter sensitivity analysis, showing that GPT4All performance is relatively stable across the `$5 \times 10^{-6}$` to `$10 \times 10^{-6}$` range (most benchmarks vary by <2 pp). This suggests the training recipe is not brittle to learning rate choice, at least for the 8B model and GPT4All benchmarks. However, the sweep covers only one hyperparameter (learning rate) and one model size (8B), and its transfer to 70B and 405B is assumed rather than verified.

**Missing ablations of significant interest:**

- **Data mixture component ablation:** The paper does not report what happens if any of the eight data categories (Table 1) is removed. For example: does removing the 6.1% roleplaying data reduce system prompt sensitivity? Does removing the 2.5% steering/alignment data restore a default helpful assistant persona? Does removing the 4.3% tool use data degrade agentic capabilities on downstream tasks? Without these ablations, the contribution of each data category to final performance is unknown—the mixture proportions may be suboptimal, and the paper provides no evidence that they are necessary.

- **Packing vs. no-packing efficiency:** The paper claims "96% efficiency" from sample packing but does not report what efficiency would be without packing, nor does it show that packing does not degrade model quality (e.g., via subtle cross-contamination that the block-diagonal attention mask fails to prevent). A comparison of a packed vs. non-packed training run on the 8B model would validate this engineering choice.

- **Agentic tagging impact:** The paper describes the structured XML tagging system as a key capability but does not evaluate whether models trained with these tags actually produce more accurate structured outputs than models trained without them. A comparison of structured output accuracy (e.g., valid JSON tool calls, correctly nested XML) between Hermes 3 and a baseline without explicit tagging training would demonstrate the tagging system's contribution.

- **Refusal filtering impact:** The paper filters out refusal-containing samples during data curation. What happens if these are kept in? Does the model begin refusing certain requests, or does the neutral alignment training in the steering/alignment category override refusal examples? This would be a direct test of the paper's central philosophical claim—that neutral alignment is a trainable property that can coexist with refusal examples—but it is not run.

- **System prompt sensitivity quantification:** The paper claims Hermes 3 is "highly sensitive to the system prompt" and provides qualitative examples (Figures 1, 6), but does not quantify this sensitivity. A controlled experiment varying system prompt content (e.g., persona specifications, constraint instructions, empty vs. populated prompts) and measuring behavioral compliance would convert the qualitative claim into a quantitative metric.

- **FP8 quantization impact (405B only):** The 405B evaluations use FP8 quantization, but the paper does not compare FP8 and FP16/BF16 inference accuracy. The quantization could account for a non-trivial fraction of the performance gap between Hermes 3 405B and Llama 3.1 Instruct 405B (which presumably also used quantization for evaluation, though this is not stated). Without an FP16 baseline, the contribution of quantization error to the reported scores is unknown.

**Negative results.** The paper reports several negative results that are informative despite being understated:

- **DPO's ineffectiveness at scale (Section 4.2):** "For the larger model sizes DPO provided only a negligible performance improvements." This is a genuine negative result that contradicts the expectation (common in the RLHF/DPO literature) that preference optimization is essential for high-quality instruct models.

- **IFeval degradation with neutral alignment (Table 5):** Hermes 3 consistently underperforms Llama 3.1 Instruct on IFEval, with the gap widening at smaller scales (84.87 vs. 87.09 at 405B, 81.21 vs. 87.25 at 70B, 62.25 vs. 80.15 at 8B). This is a clear cost of neutral alignment: the model is less willing or able to follow precise formatting and content constraints.

- **MMLU-PRO degradation (Table 5):** Hermes 3 405B scores 54.14 vs. 63.51, a 9.37 pp gap on the most challenging knowledge benchmark in the suite. This suggests that neutral alignment and diverse persona training may interfere with the model's ability to access and apply specialized knowledge.

- **Big Bench regression under DPO (Table 4):** The 8B model's Big Bench score drops from 44.57 to 43.04 after DPO, a 1.53 pp regression. This is a reminder that preference optimization can degrade capabilities even as it improves others—the preference data may encode preferences that conflict with the factual or reasoning demands of certain benchmarks.

### Critical Assessment

The paper's central claims, as articulated in the abstract and introduction, are:

1. Hermes 3 is "a neutrally-aligned generalist instruct and tool use model with strong reasoning and creative abilities."
2. Its largest version "achieves state of the art performance among open weight models on several public benchmarks."
3. The model faithfully follows system and instruction prompts "exactly and neutrally," distinguishing it from models that "refuse instructions on moral grounds."
4. The empty-system-prompt behavior (Figure 6) demonstrates genuine prompt conditioning rather than baked-in persona defaults.

How well do the reported experiments support these claims?

**Claim 1 (strong reasoning and creative abilities) is partially supported.** The benchmark results show Hermes 3 405B winning on AGIEval (+3.24 pp), ARC-Challenge (+3.41 pp), and HellaSwag (+1.85 pp)—all benchmarks that require reasoning rather than pure knowledge. This supports the "strong reasoning" claim. However, Hermes 3 loses on MMLU (−1.12 pp), MMLU-PRO (−9.37 pp), and MATH (−5.13 pp), which are arguably the most reasoning-intensive benchmarks in the suite. The claim should be qualified: Hermes 3 shows strong reasoning on commonsense and exam-style tasks but underperforms on graduate-level and mathematical reasoning relative to its Llama 3.1 Instruct counterpart. "Creative abilities" are not directly evaluated—the paper provides qualitative examples (Figures 1, 6, 7, 8, 9) demonstrating creative persona adoption and structured generation, but there is no quantitative metric for creativity (e.g., diversity scores, human preference judgments, or creative writing benchmarks). The claim is supported by demonstration, not measurement.

**Claim 2 (state of the art on several benchmarks) is supported but imprecise.** The paper does not specify which "several" benchmarks Hermes 3 leads on. From Table 5, it wins on AGIEval, ARC-Challenge, GPQA, HellaSwag, and several smaller-margin benchmarks. But "state of the art among open weight models" is a claim that requires comparison to more than just Llama 3.1 Instruct—the paper does not compare to other leading open models (e.g., Mistral Large, Qwen 2.5, Command R+, DBRX) to establish SOTA status. The comparison is exclusively to Llama 3.1 Instruct, making the SOTA claim an extrapolation from a single baseline. A more accurate statement would be: "Hermes 3 405B outperforms Llama 3.1 Instruct 405B on several benchmarks." Whether this constitutes SOTA among all open weight models is untested.

**Claim 3 (faithful and neutral instruction following) is supported for neutral alignment but contradicted for instruction-following precision.** The IFEval results—consistently lower than Llama 3.1 Instruct across all sizes—show that Hermes 3 is less precise at following formatting and content instructions. The claim that the model follows instructions "exactly" is not supported by the quantitative evidence. The "neutrally" part—that the model does not impose moral judgments—is supported by the data curation methodology (removing refusal examples) and the qualitative generation examples, but is not directly evaluated. There is no benchmark for "refusal rate" or "neutrality under morally charged prompts" that would quantify this property. The paper provides a philosophical argument and a training recipe designed to achieve neutral alignment, but does not experimentally verify that the resulting model is actually neutral across a range of potentially controversial topics. This is a significant gap: the paper's central philosophical contribution—neutral alignment—is the least quantitatively validated.

**Claim 4 (genuine prompt conditioning) is supported qualitatively but not quantitatively.** Figure 6 demonstrates the empty-system-prompt behavior for the 405B model, and Figure 1 shows Shakespeare-prose conditioning. These are compelling demonstrations, but they are n=1 examples. The paper does not report a systematic evaluation of system prompt following across a range of persona specifications, constraint types, or prompt formats. A benchmark like IFEval could be adapted to measure this—vary the system prompt while keeping the instruction constant, and measure whether the model's behavior changes as specified—but no such experiment is reported. The claim that Hermes 3 conditions "genuinely" on the system prompt (as opposed to blending prompt instructions with a default persona) is plausible given the empty-system-prompt result, but the evidence is anecdotal.

**What experiments would have strengthened the paper:**

1. **A refusal benchmark:** Present the model with morally charged or potentially controversial prompts (similar to those that trigger refusals in commercial models) and measure refusal rate, engagement quality, and response appropriateness. Compare to Llama 3.1 Instruct and closed models. This would directly validate the "neutral alignment" claim and quantify what the philosophical stance means in practice.

2. **Data mixture ablations:** Train 8B models with individual data categories removed (or proportions varied) to measure each category's contribution to downstream performance. This would transform the data mixture from an asserted recipe to an empirically validated composition.

3. **System prompt sensitivity quantification:** Create a benchmark with diverse system prompts (empty, persona-specifying, constraint-imposing, roleplaying, tool-use) and measure how well the model adapts its behavior to each. Compare to baselines without the steering/alignment data category. This would quantify the paper's most distinctive capability claim.

4. **Comparison to additional open models:** Evaluate against Mistral, Qwen, Command R+, and other recent open instruct models to substantiate the SOTA claim and place Hermes 3 in the broader ecosystem.

5. **FP8 vs. FP16 comparison for the 405B:** Quantify the accuracy cost of quantization to determine whether some of the reported benchmark gaps are attributable to precision reduction rather than training differences.

6. **Multi-turn and agentic evaluations beyond MT-Bench:** The paper emphasizes multi-turn conversation, roleplaying consistency, and agentic reasoning, but evaluates these primarily through MT-Bench (which is a general conversational quality benchmark, not specifically designed for roleplaying or agentic tasks). Domain-specific evaluations—roleplaying consistency, tool-calling accuracy, retrieval citation precision—would validate the targeted capabilities that the data mixture was designed to teach.

**What the experiments genuinely demonstrate:**

The experiments demonstrate that a domain-diverse SFT dataset (390M tokens across eight categories), combined with systematic sample packing for efficient training, can produce an instruct model that matches or exceeds Llama 3.1 Instruct on commonsense and exam-style reasoning benchmarks while trading off performance on instruction-following precision and advanced knowledge-intensive tasks. The results show that neutral alignment is achievable—the model does not default to a helpful assistant persona—and that structured agentic tagging can be trained into a model through SFT alone. The experiments do not demonstrate that Hermes 3 is state of the art among all open weight models, that neutral alignment is superior to refusal-based alignment for any specific use case, or that the specific data mixture proportions are optimal. The paper's primary contribution is the existence proof: a 405B model trained without refusal guardrails can be highly capable and genuinely steerable, with transparent agentic reasoning, all achievable through careful data curation and efficient training engineering. The quantitative results support this existence proof, but the paper's broader claims about superiority and state-of-the-art status extend beyond what the experiments validate.

## 6. Limitations and Trade-offs

### 6.1 Neutral Alignment Claims Are Unvalidated by Quantitative Measurement

**The assumption or constraint.** The paper's central philosophical contribution—that Hermes 3 is "neutrally-aligned" and "does not refuse instructions on moral grounds"—is asserted based on the training methodology (removing refusal examples from the SFT data, including a 2.5% steering/alignment data category) but is **never quantitatively evaluated**. The paper does not report any benchmark, human evaluation, or systematic study of the model's behavior on potentially controversial, morally charged, or edge-case prompts. The claim that "there is no such thing as latent thoughtcrime" (Section 1) and that guardrails belong "at the larger system levels, rather than on the models themselves" (Section 1) is a philosophical position supported by qualitative demonstrations (Figures 1, 6, 7, 8, 9), not by measurement.

**The consequence.** A practitioner deploying Hermes 3 in a production application has **no empirical evidence** about how the model will behave on prompts that would trigger refusals in commercial models. Will it provide dangerous information without warning? Will it engage with harmful requests in ways that create legal or reputational risk? Will its responses to sensitive topics be factually accurate and appropriately contextualized, or will it produce harmful content? The paper provides no data to answer these questions. The training methodology (filtering out refusal-containing examples) guarantees that the model has not been explicitly taught to refuse, but it does not guarantee that the model's responses will be safe, appropriate, or even coherent on topics it was not trained to handle. The burden of assessing safety is shifted entirely to the application developer—who has no quantitative characterization of the model's behavior on these prompts to guide their system-level guardrail design.

**What evidence exists in the paper.** None. There is no refusal benchmark, no red-teaming evaluation, no toxicity or safety assessment, and no comparison to Llama 3.1 Instruct on safety-relevant metrics. The paper's only evidence for neutral alignment is:

- The data curation methodology (Section 3): refusal-containing samples were filtered out, and 2.5% of training data was dedicated to steering/alignment to teach system prompt sensitivity.
- Qualitative demonstrations (Appendix A, Figures 1, 6, 7, 8, 9) showing creative persona adoption and agentic reasoning—none of which involve sensitive or controversial content.
- The empty-system-prompt behavior (Figure 6) demonstrating persona dependence but not neutrality.

The paper does not even define what "neutral alignment" means operationally—does it mean the model never refuses? Answers all requests equally? Provides balanced perspectives on controversial topics? Warns users about potential harms? Without an operational definition and corresponding measurement, the central claim is a training intention, not an empirical result.

**Mitigation status.** The paper does not acknowledge this as a limitation. It presents neutral alignment as a feature and philosophical stance without addressing the need to validate that the model actually behaves neutrally across a relevant range of inputs. The introduction argues that guardrails belong at the system level, which is a design choice—but the paper provides no evidence that system-level guardrails can effectively compensate for the absence of model-level refusals with this specific model. Future work on safety evaluation and red-teaming of neutrally-aligned models is implicitly suggested by the gap, but not explicitly called for by the authors.

---

### 6.2 The $14×$ Larger Model Baseline Is a Single Comparison Point in a Sparse Evaluation Landscape

**The assumption or constraint.** The paper's headline claim—that Hermes 3 405B "achieves state of the art performance among open weight models on several public benchmarks" (Abstract)—is supported by comparison to exactly **one baseline**: Llama 3.1 Instruct at corresponding sizes. No other open-weight instruct models are evaluated (e.g., Mistral Large, Qwen 2.5, Command R+, DBRX, Yi, DeepSeek), and no closed-weight models are included (e.g., GPT-4, Claude 3.5, Gemini). The evaluation landscape in Table 5 is a two-column comparison per model size, with Llama 3.1 Instruct as the sole reference point.

**The consequence.** The "state of the art" claim is untestable from the data provided. A practitioner choosing between Hermes 3 405B and, say, Mistral Large or Qwen 2.5 72B cannot use this paper to make an informed decision because the relative performance on any benchmark is unknown. More fundamentally, the paper does not establish that the improvements over Llama 3.1 Instruct are attributable to Hermes 3's specific innovations (neutral alignment, domain-diverse data mixture, structured tagging) rather than to the pretrained Llama 3.1 base model's capabilities. If other fine-tunes of Llama 3.1—or instruct models built on different base models—achieve similar or better results with different recipes, then the paper's claimed contributions (data mixture composition, neutral alignment) may not be responsible for the observed performance.

The sparse baseline also makes it impossible to assess whether Hermes 3's losses on critical benchmarks (IFEval, MMLU-PRO, MATH, MMLU) are specific to the comparison with Llama 3.1 Instruct or represent a general weakness of the approach. If multiple competing open instruct models all outperform Hermes 3 on MMLU-PRO by similar margins, the deficit is a property of the training recipe. If Llama 3.1 Instruct is uniquely strong on these benchmarks, the deficit may be less concerning.

**What evidence exists in the paper.** Table 5 provides the only systematic evaluation, and it contains exactly two models per size: Hermes 3 and Llama 3.1 Instruct. The evaluation suite is comprehensive in terms of benchmark coverage (16 benchmarks spanning reasoning, knowledge, instruction-following, and conversational quality) but minimal in terms of model diversity—the entire comparison space is a single baseline per size class. The paper does not cite or discuss results for any other open model, nor does it position Hermes 3 relative to leaderboard rankings (e.g., the Open LLM Leaderboard, which the paper references in Section 5 as a source of evaluation benchmarks but does not use for comparative analysis).

**Mitigation status.** The paper does not acknowledge this as a limitation. The "state of the art" language in the abstract is presented without qualification. The choice to compare only to Llama 3.1 Instruct is understandable—it shares the same base pretraining, making it the most controlled comparison for isolating the effects of the SFT recipe—but the extrapolation from a single comparison to a SOTA claim is not justified by the evidence provided. Practitioners should interpret the results as: "Hermes 3 outperforms Llama 3.1 Instruct on several benchmarks and underperforms on others," not as "Hermes 3 is the best open instruct model available."

---

### 6.3 IFEval Degradation Reveals a Fundamental Tradeoff Between Neutral Alignment and Instruction-Following Precision

**The assumption or constraint.** The paper assumes that neutral alignment—training the model to be "highly sensitive to the system prompt" (Section 2) and to adopt diverse personas—is compatible with precise instruction following. The IFEval Strict results in Table 5 directly contradict this assumption. Hermes 3 underperforms Llama 3.1 Instruct on IFEval at every model size, with the gap widening dramatically at smaller scales: 84.87 vs. 87.09 at 405B (−2.22 pp), 81.21 vs. 87.25 at 70B (−6.04 pp), and 62.25 vs. 80.15 at 8B (−17.90 pp).

**The consequence.** This is not a minor regression—it represents a **fundamental tradeoff** that limits Hermes 3's utility for applications requiring strict constraint following. IFEval measures whether the model follows precise formatting and content instructions (e.g., "write exactly 3 paragraphs," "include the word 'elephant' exactly twice," "end with a specific phrase"). A model that scores 62.25% (8B) on this benchmark fails to follow explicit instructions nearly 38% of the time—a failure rate that would be unacceptable for many production applications involving structured output generation, API responses, or regulatory compliance.

The pattern across model sizes is particularly informative: the gap is smallest at 405B, suggesting that larger models can better reconcile neutral alignment with instruction following. But even at 405B, a 2.22 pp gap on IFEval means Hermes 3 makes more constraint-violation errors than the baseline. For a model whose primary design goal is "faithfully respond to the request of the user" (Section 2), this is a significant shortcoming—the model is less faithful to explicit instructions than its refusal-trained counterpart.

The mechanism underlying this tradeoff is not hard to hypothesize: training the model to adopt diverse personas and to condition heavily on the system prompt may create competing objectives when the instruction content conflicts with the persona specification. A model that is "highly sensitive to the system prompt" may interpret a persona specification ("you are a pirate") as overriding a formatting constraint ("write exactly 3 paragraphs"), leading to instruction-following failures. The paper does not investigate this hypothesis or characterize the types of IFEval failures—are they persona-related conflicts, formatting errors, or content omissions?

**What evidence exists in the paper.** Table 5 reports IFEval Strict scores for all three model sizes, with Hermes 3 trailing Llama 3.1 Instruct in every comparison. Table 2 (70B epoch evaluation) shows that IFEval is the only metric that continues improving through all four epochs (76.52 → 78.92 → 81.33 → 86.61), while other capabilities peak and then degrade—suggesting that instruction following requires more training exposure than reasoning, and that the selected epoch 3 checkpoint (which balances all metrics) sacrifices some IFEval performance relative to epoch 4. The epoch selection protocol's composite scoring thus implicitly trades some instruction-following precision for better reasoning and conversational quality. The paper does not provide per-category or per-failure-type analysis of IFEval results.

**Mitigation status.** The paper does not acknowledge this tradeoff or discuss it as a limitation. The IFEval degradation is visible in Table 5 but is not highlighted, analyzed, or explained in the text. There is no suggestion that future work should investigate the tension between persona flexibility and instruction-following strictness, or that the training recipe could be adjusted to improve IFEval without sacrificing neutral alignment. For practitioners, this tradeoff is critical: if your application requires precise constraint following, Hermes 3's approach comes with a measurable cost that the paper does not prepare you for.

---

### 6.4 No Evidence That Individual Data Mixture Components Matter—The Recipe Is Unvalidated

**The assumption or constraint.** The paper's primary contribution is the composition of the 390M-token SFT dataset across eight categories (Table 1): General Instructions (60.6%), Domain Expert (12.8%), Math (6.7%), Roleplaying (6.1%), Coding (4.5%), Tool Use/Agentic/RAG (4.3%), Content Generation (3.0%), and Steering/Alignment (2.5%). The paper asserts that this mixture "has significantly contributed to the strong performance of our models" (Section 3) and that domain-specific categories were added to "address known weaknesses in older Hermes models" (Section 3). However, **no ablation study removes or varies any category** to establish its contribution. The data mixture is presented as a finished recipe with no empirical justification for the specific proportions.

**The consequence.** A practitioner cannot determine which components of the data mixture are necessary, which are beneficial but optional, and which might be counterproductive. The 2.5% Steering/Alignment category—which the paper's philosophical argument implies is critical for neutral alignment—may be irrelevant to the observed behavior; the model might be neutrally aligned simply because refusal examples were removed from the General Instructions category, making the dedicated steering data redundant. Conversely, the 6.1% Roleplaying category may be essential for system prompt sensitivity, and removing it might cause the model to revert to a default persona—but without an ablation, this is speculation.

The lack of ablations also means that the data mixture cannot be improved based on evidence. Are the proportions near-optimal, or would reallocating tokens from General Instructions (60.6%) to Math (6.7%) or Coding (4.5%) yield better reasoning performance? Would increasing the Steering/Alignment fraction further improve system prompt sensitivity, or is 2.5% already past the point of diminishing returns? The paper provides no guidance for practitioners who want to adapt the recipe for different capability priorities.

The filtering methodology (removing refusals, improperly formatted responses, empty turns; prioritizing strongest-model generations) is similarly unvalidated. The paper does not compare filtered vs. unfiltered training to demonstrate that the filtering steps improve performance, nor does it report what fraction of data was removed at each filtering stage.

**What evidence exists in the paper.** Table 1 reports the data mixture proportions as a fait accompli. Section 3 describes the curation process in general terms (Evol-Instruct generation, manual quality filtering, five-month timeline) but provides no ablations, no sensitivity analyses, and no experiments varying data composition. The paper does compare Hermes 3 to prior Hermes releases in passing ("to address known weaknesses in older Hermes models") but does not systematically evaluate the older models on the same benchmarks to demonstrate that the new mixture resolved those weaknesses. The epoch selection protocol (Table 2) indirectly validates training duration but says nothing about training content.

**Mitigation status.** The paper does not acknowledge the lack of data ablations as a limitation. The data mixture is presented as a contribution in itself—"a robust foundation for the Hermes 3 model" (Section 3)—without recognizing that the contribution's validity depends on evidence linking specific components to specific outcomes. Given that the five-month curation effort represents the bulk of the project's human investment (the training recipe is largely standard), the absence of any experimental validation of the data composition is a significant gap. Future work could systematically ablate categories, vary proportions, and characterize the contribution of each domain to downstream capabilities. The paper provides none of this.

---

### 6.5 Evaluation Protocol Risks Optimistic Bias from Epoch Selection on Overlapping Benchmarks

**The assumption or constraint.** The paper selects the best training epoch for each model size using a composite score computed from GPT4All average, AGIEval average, IFEval Strict, and MT-Bench average (Section 4.1, Table 2). The final reported results (Table 5) then include these same four metrics plus an additional 12 benchmarks from largely the same evaluation ecosystem. The epoch selection process optimizes directly on a subset of the evaluation suite, without a held-out development set, without cross-validation over questions (only over epochs), and without any correction for the multiple-comparison problem inherent in selecting the best of four checkpoints.

**The consequence.** The reported benchmark scores for Hermes 3 in Table 5 are **upper bounds on expected performance** rather than unbiased estimates. The epoch that scored highest on GPT4All+AGIEval+IFEval+MT-Bench is systematically favored—its scores on those four metrics are, by construction, the best among the four checkpoints, and any random noise in the evaluation process that happened to benefit that epoch on those metrics will inflate the reported numbers. This is a form of **implicit overfitting to the evaluation set** through checkpoint selection, even though the test prompts themselves were not used during training.

The additional 12 benchmarks in Table 5 (BBH, MATH, GPQA, MuSR, MMLU, MMLU-PRO, TruthfulQA) are not directly optimized during epoch selection, but they are correlated with the optimized metrics. A checkpoint that performs well on AGIEval (exam-level reasoning) is likely to perform well on MMLU (also exam-level reasoning) for reasons unrelated to genuine capability improvements—shared variance in the evaluation harness, similar multiple-choice format, overlapping knowledge requirements. This correlation means that the epoch selection bias spills over into the non-optimized benchmarks to an unknown degree.

The problem is most acute for the 405B model, where the paper states that "lowering the learning rate relative to the 8B and 70B models produced superior results" based on unspecified trial runs (Section 4.1). If those trial runs used the same evaluation suite for learning rate selection, the reported 405B numbers reflect both epoch selection bias and hyperparameter optimization bias—a compound overfitting risk.

**What evidence exists in the paper.** Table 2 shows the per-epoch scores for the 70B model across the four optimized metrics. The variation across epochs is substantial: GPT4All varies from 73.63 to 76.85 (range 3.22 pp), AGIEval from 54.00 to 56.10 (2.10 pp), IFEval from 76.52 to 86.61 (10.09 pp), MT-Bench from 8.37 to 8.99 (0.62 points). Selecting the best epoch therefore captures 3–10 pp of score improvement that may partly reflect epoch-level noise rather than genuine capability gains. The normalized composite scores—27.50, 63.65, 83.89, 37.09—show that epoch 3 is a clear outlier, with nearly twice the score of the next-best epoch. This magnitude of separation is unlikely to be purely noise, but the paper provides no statistical test or confidence interval to distinguish signal from variance.

The paper does not report the per-epoch scores for the 8B or 405B models (only the selected epochs are in Table 5), so the magnitude of epoch-to-epoch variation at those scales is unknown. The training loss curves (Figure 4) show monotonic decreases without overfitting inflections, but as the paper itself demonstrates with the 70B model, training loss can continue improving while benchmark performance degrades—the loss curves provide no protection against epoch selection bias.

**Mitigation status.** The paper does not acknowledge this as a limitation. Standard practices for mitigating checkpoint selection bias—using a separate development set for epoch selection, applying multiple-comparison corrections, reporting confidence intervals, or averaging over the last few checkpoints—are not employed. The epoch selection protocol is described transparently (Section 4.1), which is commendable, but the implications for the reliability of the reported numbers are not discussed. Practitioners should treat the Table 5 results as **optimistic estimates** that may not replicate exactly at a different random seed or with a different evaluation harness configuration.

---

### 6.6 FP8 Quantization Introduces Unmeasured Accuracy Costs for the 405B Model

**The assumption or constraint.** The 405B model evaluations in Table 5 were performed under FP8 quantization—specifically, "round-to-nearest weight quantization with channelwise activations and per-token scales" using the llm-compressor library for vLLM (Section 5). The paper does not report FP16 or BF16 evaluation results for the 405B model, does not ablate the quantization method, and does not quantify the accuracy cost of the precision reduction. The Llama 3.1 Instruct 405B baseline presumably used its own quantization scheme (the paper does not specify), creating a confound: some fraction of the observed performance differences between Hermes 3 405B and Llama 3.1 Instruct 405B may be attributable to quantization effects rather than training differences.

**The consequence.** The reported benchmark scores for Hermes 3 405B are **lower bounds on the model's full-precision performance** by an unknown margin. If FP8 quantization costs, say, 1–2% relative accuracy on reasoning benchmarks, then Hermes 3's true full-precision scores would be correspondingly higher—potentially closing or narrowing some of the reported gaps with Llama 3.1 Instruct. Conversely, if Llama 3.1 Instruct's quantization scheme is more accurate (e.g., using per-channel rather than per-token scaling, or a different rounding strategy), some of Hermes 3's apparent advantages might be quantization artifacts rather than genuine capability differences.

The paper's claim that Hermes 3 405B "achieves state of the art performance" is sensitive to this confound. If Hermes 3 wins on AGIEval by 3.24 pp (61.84 vs. 58.60) but quantization costs 2 pp, the true full-precision gap might be 5.24 pp—more impressive. If Hermes 3 loses on MMLU-PRO by 9.37 pp (54.14 vs. 63.51) but quantization costs 2 pp, the true gap is 7.37 pp—still substantial, but smaller. Without a full-precision baseline, the direction and magnitude of the quantization bias are unknown.

The practical consequence for deployment is equally significant: a practitioner running Hermes 3 405B under FP8 quantization will observe exactly the reported performance, but a practitioner running at FP16 or BF16 (which requires substantially more GPU memory and may be infeasible on available hardware) cannot predict how much improvement to expect.

**What evidence exists in the paper.** The FP8 quantization is mentioned in a single sentence (Section 5) with no supporting evaluation, no comparison to higher precision, and no discussion of its accuracy implications. The paper does not specify whether Llama 3.1 Instruct 405B was evaluated under the same quantization scheme or a different one. The training section (Section 4.1) discusses FP16/BF16 training in the context of GPU memory requirements and CPU offloading, but the evaluation section provides no precision-related ablations.

**Mitigation status.** The paper does not acknowledge this as a limitation. FP8 evaluation is presented as a practical necessity for running a 405B model—which it is—but the absence of any quantification of its accuracy cost means the reported numbers cannot be directly compared to full-precision results from other models or to theoretical expectations. The paper suggests (Section 4.1, in the context of training parallelism) that "using higher dimensional parallelism ... is likely necessary for future 405B training runs," but does not extend this reasoning to inference: higher-dimensional parallelism could also enable FP16 inference for evaluation, eliminating the quantization confound. Future work should report both quantized and full-precision results, or provide a validated estimate of the quantization accuracy cost, to make the 405B results comparable to models evaluated at higher precision.

## 7. Implications and Future Directions
- Field impact
  - Hermes 3 demonstrates that open instruct models can combine strong steerability, transparent agent tooling, and competitive benchmark performance, advancing the feasibility of open, auditable AI agents. The explicit tagging scheme for reasoning, planning, and citations provides a concrete interface for building trustworthy agent systems.

- Practical applications
  - Enterprise assistants with strict persona/style control (Figure 1).
  - Tool‑driven agents that must expose calls/results for auditing, e.g., finance, healthcare, legal domains (Section 2.1).
  - Retrieval‑centric assistants that must cite sources consistently (Figure 8).
  - Developer copilots that output plans, diagrams, and tests alongside code (Figure 9).

- Suggested follow‑up research
  - Improving exam‑style knowledge and math without sacrificing steerability—e.g., targeted SFT or curriculum augmentation for MMLU/MATH tasks.
  - Stronger instruction‑following generalization on IFEval while preserving neutrality.
  - Scaling the DPO phase for larger models (beyond LoRA) or exploring alternative preference‑learning methods that complement the agentic tag regime.
  - Training without FP8 constraints for evaluation fairness, or systematically quantifying FP8 impacts.
  - More extensive robustness checks: long‑context stress tests (the base is 128K), out‑of‑domain tool schemas, noisy retrieval corpora, and adversarial prompts targeting the tag scaffolding.

- System design recommendations
  - Given the model’s neutrality and sensitivity to system prompts (Figures 1 and 6), production systems should:
    - Provide explicit, unambiguous system prompts that encode safety/compliance rules.
    - Validate/tag outputs programmatically by parsing the XML‑like tags.
    - Log and verify tool calls and citations, using the structured `<tool_call>`/`<co>` formats to enable audit trails.

> Overall, Hermes 3 shows that careful instruction tuning with structured agentic tags, tool/RAG schemas, and a curated instruction dataset can produce highly steerable, open models that outperform open baselines on several reasoning tasks (Table 5), while revealing trade‑offs on instruction‑following benchmarks like IFEval and knowledge exams like MMLU/MATH.
