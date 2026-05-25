# A Survey of Small Language Models

**ArXiv:** [2410.20011](https://arxiv.org/abs/2410.20011)

## 🎯 Pitch

This paper delivers the first comprehensive survey focused on Small Language Models (SLMs), introducing a novel taxonomy that links concrete architectural, training, and compression techniques to the specific computational, memory, and latency constraints they solve. By systematically organizing and synthesizing the literature across lightweight architectures, efficient attention, and resource-aware model optimization, it equips researchers and practitioners with an essential roadmap for deploying powerful language models in constrained environments—unlocking real-time, on-device, and privacy-preserving applications that traditional LLMs struggle to support.

---

## 1. Executive Summary

This survey systematically catalogs the techniques for building and optimizing **Small Language Models (SLMs)**, organizing methods along a novel taxonomy that categorizes approaches by both the technique used—architecture design, training, or compression—and the constraint being optimized—inference runtime, memory, storage, latency, or communication overhead. The paper covers three main technical axes: lightweight and efficient architectures (e.g., MobileBERT's inverted-bottleneck structure, linear attention mechanisms like Mamba), training techniques for resource-constrained settings (e.g., mixed-precision training, parameter-efficient fine-tuning via LoRA), and model compression methods (e.g., unstructured pruning via SparseGPT, quantization via GPTQ and AWQ, knowledge distillation from white-box teachers). The survey identifies a fundamental boundary condition: progress on any one constraint does not necessarily imply progress on others—memory-efficient training methods like quantization-aware training are often slower than their full-precision counterparts—establishing that SLM optimization is inherently a multi-objective tradeoff space where gains in one dimension (e.g., memory footprint) frequently come at the cost of another (e.g., training speed).

## 2. Context and Motivation

### The Core Problem: We Don't Have a Unified Map of How to Build Small Language Models

The fundamental question this survey tackles is deceptively simple: **given that large language models (LLMs) are too expensive for many real-world deployments, what is the complete set of techniques for making them smaller while preserving their capabilities?** The paper's motivation stems from a tension that has become increasingly acute as LLMs have grown from millions to hundreds of billions of parameters: the capabilities that make these models powerful also make them inaccessible for the vast majority of potential use cases.

This gap matters for several practical reasons the authors highlight throughout Sections 1, 6, and 7:

- **On-device and edge deployment**: LLMs require centralized, specialized hardware for both training and inference. Running a 175B-parameter model on a smartphone, a Raspberry Pi, or an embedded medical device is simply infeasible—not just because of memory constraints, but because of latency requirements, energy budgets, and the need to function without continuous cloud connectivity. SLMs aim to bring language understanding to these contexts.

- **Privacy and data sovereignty**: When inference must happen on a cloud server, sensitive user data (medical records, personal conversations, location history) leaves the device. On-device SLMs enable processing to happen locally, keeping data under the user's control. This is particularly critical for applications in healthcare (e.g., clinical note processing with MIMIC data), digital assistants that access personal information, and any domain governed by privacy regulations like HIPAA or GDPR.

- **Cost and environmental impact**: Training and serving LLMs consumes enormous amounts of energy and compute. The paper notes that reducing model size directly reduces inference-time energy use, which is especially important for battery-powered devices and for large-scale deployments where the aggregate energy footprint of millions of inference calls matters both economically and environmentally.

- **Latency-sensitive applications**: Real-time interaction—voice assistants, chatbots, live translation, augmented reality interfaces—cannot tolerate the round-trip latency of cloud-based LLM inference. SLMs running locally can respond in milliseconds rather than seconds, enabling genuinely interactive experiences.

- **Democratization of AI**: If advanced language capabilities require GPU clusters accessible only to well-resourced organizations, the technology concentrates power. SLMs that can run on consumer hardware lower the barrier to entry for researchers, startups, and users in low-resource settings.

### A Fragmented Literature With No Organizing Framework

The paper identifies a specific gap that motivates its existence as a survey: while there is a vast and rapidly growing body of work on making language models smaller and more efficient, **this literature is fragmented across multiple subcommunities that rarely reference each other**. An architectural innovation like Mamba's selective state space model, a compression technique like SparseGPT's one-shot pruning, and a training method like LoRA may all serve the same ultimate goal—getting capable language models onto resource-constrained devices—but they evolved in different research traditions with different vocabularies, evaluation protocols, and baseline comparisons.

The consequences of this fragmentation are practical:

**No common taxonomy.** Prior to this survey, there was no systematic way to understand how different approaches relate to each other, which constraints they address, or where they trade off against each other. A practitioner wanting to deploy an SLM on a mobile device faces a bewildering array of options—pruning, quantization, distillation, efficient architectures, parameter-efficient fine-tuning—with no framework for deciding which combination of techniques makes sense for their specific constraints. The paper's proposed taxonomy (Tables 1 and 2) fills this gap by categorizing methods along two axes: the *type of technique* (architecture, training, compression) and the *constraint being optimized* (inference runtime, memory, storage, latency, communication overhead).

**Conflicting or incomparable results.** Different subcommunities optimize for different things and report different metrics. A pruning paper might report perplexity and compression ratio; a quantization paper might report memory footprint and inference speed on a specific GPU; an efficient architecture paper might report accuracy on a benchmark suite. Without a unified evaluation framework, it is difficult to know whether, say, a 4-bit quantized version of LLaMA outperforms a purpose-built lightweight architecture like MobileLLM on a given task under a given resource budget. The survey's compilation of benchmark datasets and evaluation metrics in Section 5 provides a starting point for such comparisons.

**No clear picture of tradeoffs.** Perhaps most critically, the paper emphasizes that **progress on any one constraint does not imply progress on others**. The authors state this explicitly in Section 1:

> "It is important to note that progress on any one of these goals does not necessarily imply progress on the others. In fact, there are often trade-offs between them."

They give the concrete example of quantization-aware training (QAT), which enables training with reduced memory but is often slower than full-precision training. This means the SLM design space is fundamentally multi-objective—you cannot simply ask "which method is best"; you must ask "which method is best *for my specific set of constraints*." Prior work had not systematically surfaced these tradeoffs.

### The Shifting Definition of "Small"

The paper confronts a definitional challenge head-on. What counts as a "small" language model is a moving target. The authors note:

> "GPT-2, a 'large language model' in 2019 at 1.5B parameters, is smaller than many 'small' language models covered in this survey."

Rather than fixing a hard parameter-count threshold, the survey adopts a functional definition: an SLM is any model that **aims to retain the accuracy and/or adaptability of large language models while being subject to some constraint**—training or inference hardware, data availability, bandwidth, or generation time. This definition is deliberately broad because it captures the common goal shared across otherwise very different research threads: getting useful language capabilities into environments where a full-scale LLM cannot go.

This functional definition also explains why the survey covers techniques like mixed-precision training and distributed training optimizations (ZeRO, FSDP) that are also used for LLMs. The boundary between "training an LLM efficiently" and "enabling an SLM" is porous—efficient training techniques lower the resource barrier that defines what counts as "small" for a given hardware budget.

### Where Prior Work Falls Short

The paper explicitly positions itself against existing surveys that cover related but distinct territory:

- Surveys on **model compression for LLMs** (e.g., Zhu et al., 2023) focus on post-hoc techniques (pruning, quantization, distillation) applied to already-large models but do not cover architectural innovations purpose-built for efficiency or training-time techniques that reduce resource requirements during model development.

- Surveys on **LLMs and their learning methods** (e.g., Rogers et al., 2020; Min et al., 2021; Shen et al., 2023) cover the full landscape of large-scale language models but center on capabilities and training paradigms rather than the constraint-aware optimization that defines the SLM space.

- Individual research threads—efficient attention mechanisms, neural architecture search for transformers, parameter-efficient fine-tuning—each have their own literature but lack synthesis across the full pipeline from architecture design through training through post-training compression.

The survey fills this gap by covering the **end-to-end pipeline** for creating SLMs, spanning architecture design (Section 2), training techniques (Section 3), and compression (Section 4), organized by the constraints each technique addresses. This is explicitly a survey paper, not a methods paper—its contribution is organizational and synthetic, providing what the authors call "a comprehensive survey of existing work on small language models for practitioners."

### The Taxonomy as Intellectual Scaffolding

The paper's central organizational contribution is the two-axis taxonomy presented in Tables 1 and 2, which serves multiple purposes:

1. **Constraint-awareness**: By tagging each technique with the constraints it primarily addresses (e.g., lightweight architectures address inference runtime and memory; quantization addresses memory and storage), the taxonomy enables practitioners to locate methods relevant to their specific bottleneck. Someone building for a memory-constrained edge device knows to look at lightweight architectures, pruning, and quantization rather than, say, data augmentation for fine-tuning.

2. **Pipeline positioning**: By organizing methods into pre-processing (architecture), training, and post-processing (compression), the taxonomy reflects the temporal flow of model development. Different techniques apply at different stages, and some combinations are natural (e.g., train with knowledge distillation, then apply post-training quantization).

3. **Tradeoff surfacing**: By mapping techniques to constraints in a matrix format, the taxonomy implicitly highlights the multi-objective nature of SLM optimization. A technique may check multiple constraint boxes (e.g., pruning addresses inference runtime, memory, and storage) but may be blank on others (e.g., pruning typically does not help training compute). The blank cells in the matrix are as informative as the filled ones.

4. **Application mapping**: Table 3 extends the taxonomy to specific application domains (real-time interaction, content generation, edge inference), showing which constraints dominate in each use case and implicitly suggesting which families of techniques are most relevant.

### How the Paper Positions Itself

The survey does not propose a new method or advance a novel technical claim. Its contribution is synthetic and organizational: it provides the first comprehensive map of the SLM landscape, with a taxonomy that is "novel" in the authors' framing because it jointly considers technique type and constraint target.

The paper's intended audience is explicitly **practitioners and researchers** who need to navigate this space. The structure reflects this: after covering architectures, training, and compression, the paper provides sections on evaluation (Section 5, with standardized datasets and metrics organized by constraint) and applications (Section 6, with concrete use cases organized by constraint), then concludes with open problems (Section 7) that define the research frontier.

The open problems section is particularly revealing about the paper's positioning. It highlights hallucination, bias, energy efficiency, and data privacy as challenges that are *shared by SLMs and LLMs* but may manifest differently at smaller scales. For example, the paper notes that some studies find larger models exhibit increased measured bias (Touvron et al., 2023a; Zhao et al., 2023a), while other work on hallucination benchmarks finds that larger models reduce certain types of errors (Guan et al., 2024). This means SLM designers cannot simply assume that smaller models inherit the same failure modes as their larger counterparts—the scaling behavior of these phenomena is an open research question, and the survey flags it as such.

In summary, this paper addresses the fragmentation of the SLM literature by providing a unified organizational framework, a comprehensive catalog of techniques across the full development pipeline, and a practical resource for practitioners navigating the multi-objective tradeoff space of building capable language models under resource constraints.

## 3. Technical Approach

### 3.1 Reader Orientation

This is a **survey paper**—it does not build a single system but rather catalogs, organizes, and relates hundreds of existing techniques for creating and deploying small language models. The "system" being described is the collective toolkit that practitioners use to take a large, resource-hungry language model and produce a smaller, more efficient version that retains as much capability as possible while satisfying specific deployment constraints. The problem it solves is the **fragmentation of knowledge** across subcommunities: architectural innovations, training optimizations, and post-hoc compression methods all serve the same ultimate goal but evolved in separate research traditions with different vocabularies. The "shape" of the solution is a two-axis taxonomy that maps every technique to both *what kind of technique it is* (architecture, training, or compression) and *which constraint it primarily addresses* (inference runtime, memory, storage, latency, communication overhead), enabling practitioners to navigate the multi-objective tradeoff space systematically.

### 3.2 Big-Picture Architecture (Diagram in Words)

The SLM development and deployment pipeline has three major stages, each corresponding to a section of the survey:

1. **Model Architecture Design (Section 2):** Before training begins, the model's structural blueprint is chosen—whether to use a lightweight encoder-only design (like MobileBERT's inverted-bottleneck), an efficient decoder-only architecture (like TinyLLaMA with FlashAttention), an alternative attention mechanism with linear complexity (like Mamba's state space model), or an architecture discovered automatically through neural architecture search. This stage determines the model's base parameter count and computational footprint.

2. **Training (Section 3):** Given an architecture, the model must learn from data. This stage covers pre-training techniques that reduce resource requirements during development (mixed-precision training with FP16/BFLOAT16/BF8, memory-efficient optimizers like Adafactor, distributed training with ZeRO and FSDP) and fine-tuning techniques that adapt a pre-trained model to a downstream task without retraining from scratch (parameter-efficient methods like LoRA, data augmentation for limited-data regimes).

3. **Model Compression (Section 4):** Starting from a fully trained model (potentially an LLM), post-hoc techniques reduce its size and inference cost. This stage encompasses pruning (removing redundant weights, either unstructured via SparseGPT or structured via layer removal), quantization (reducing numerical precision from 16-bit to 8-bit, 4-bit, or below via GPTQ, AWQ, or quantization-aware training), and knowledge distillation (training a smaller "student" model to replicate the behavior of a larger "teacher").

Information flows linearly through these stages—a model is first architected, then trained, then compressed—but the taxonomy emphasizes that techniques from different stages address overlapping constraints. For example, both architectural efficiency (stage 1) and post-training quantization (stage 3) reduce inference memory, and a practitioner might combine them.

### 3.3 Roadmap for the Deep Dive

- **First, the taxonomy itself (Tables 1 and 2):** Understanding the organizing framework is prerequisite to understanding why specific techniques are grouped as they are. The taxonomy defines what counts as a "technique category" and what counts as a "constraint," establishing the vocabulary for the rest of the discussion.
- **Second, architectural approaches (Section 2):** This is the natural starting point chronologically—the model's structure is chosen before training. The three subcategories (lightweight architectures, efficient attention, neural architecture search) represent progressively more automated approaches to finding efficient designs.
- **Third, training techniques (Section 3):** With an architecture fixed, training is where the model acquires its capabilities. The survey separates pre-training (where computational cost is dominated by the need to process enormous corpora) from fine-tuning (where the challenge shifts to data efficiency and avoiding catastrophic forgetting).
- **Fourth, compression techniques (Section 4):** These are applied to already-trained models and represent the most direct path from an LLM to an SLM. The taxonomy distinguishes pruning (removing structure), quantization (reducing precision), and distillation (transferring knowledge), each of which operates on a different aspect of the model.
- **Fifth, evaluation (Section 5):** This section maps benchmark datasets and evaluation metrics onto the constraint taxonomy, showing which datasets and metrics are appropriate for evaluating which types of efficiency. This serves as the bridge between the technical techniques and their practical assessment.

### 3.4 Detailed, Sentence-Based Technical Breakdown

---

#### The Taxonomy: A Two-Axis Classification Framework

The paper's central intellectual contribution is a taxonomy that categorizes SLM techniques along two orthogonal axes, presented in Tables 1 and 2. This taxonomy is what transforms a collection of methods into a navigable map.

**Axis 1: Technique Category.** Every method for optimizing SLMs falls into one of three categories based on when it is applied in the model development lifecycle:

- **Model Architectures (pre-processing):** Decisions made about the model's structure *before* any parameters are learned. This includes designing lightweight architectures from scratch (Section 2.1), replacing the standard quadratic self-attention with a more efficient approximation (Section 2.2), or using automated search to discover efficient architectures (Section 2.3). The defining characteristic is that these choices affect the model's innate computational profile—how many parameters it has, how computation scales with sequence length, what operations dominate inference time.

- **Training Techniques (in-processing):** Methods applied *during* the learning phase that reduce resource requirements without necessarily changing the final model's inference footprint. This includes pre-training optimizations like mixed-precision arithmetic and distributed training strategies (Section 3.1), and fine-tuning methods that adapt a model to new tasks with minimal additional computation (Section 3.2). The key distinction from architectural choices: training techniques may reduce the cost of *developing* an SLM even if the resulting model's inference cost is unchanged.

- **Model Compression (post-processing):** Techniques applied *after* a model is trained that reduce its size or inference cost. This includes pruning (removing weights or structures, Section 4.1), quantization (reducing numerical precision, Section 4.2), and knowledge distillation (training a smaller model to mimic a larger one, Section 4.3). The defining characteristic: these methods take a fully trained model (potentially a large one) as input and produce a smaller, faster variant as output.

**Axis 2: Constraint Targeted.** Each technique primarily addresses one or more specific resource limitations, indicated by checkmarks (✓) in Table 1:

- **Training Compute:** The computational cost (FLOPs, GPU-hours) of the training process itself. Addressed by mixed-precision training, distributed training optimizations, and neural architecture search (which can find architectures that converge faster).

- **Dataset Size:** The amount of training data required to reach a given performance level. Addressed by data augmentation techniques during fine-tuning and by knowledge distillation (which can transfer knowledge from a teacher trained on vast data using fewer examples).

- **Inference Runtime:** The wall-clock time needed to process an input and generate an output. Addressed by efficient attention approximations, pruning, and quantization—all of which reduce the number or cost of operations at inference time.

- **Memory:** The peak RAM (GPU or CPU) required to hold the model's parameters, activations, and intermediate states during inference or training. Addressed by lightweight architectures (fewer parameters), pruning, quantization (smaller numerical representation), and memory-efficient training strategies.

- **Storage Space:** The disk footprint of the model's saved weights. Primarily addressed by pruning and quantization, which directly reduce the number of bits needed to represent the model.

- **Latency:** The delay between providing input and receiving output, which is closely related to but distinct from throughput (the number of samples processed per unit time). Addressed by many of the same techniques as inference runtime.

The taxonomy's power comes from its matrix structure. A cell with a checkmark tells a practitioner which techniques are relevant to their bottleneck. A cell without a checkmark highlights tradeoffs: for example, model architectures reduce inference runtime and memory but do not directly reduce training compute (you still need to train the lightweight architecture from scratch), while quantization-aware training reduces training memory but may increase training time. The "blank cells are as informative as the filled ones" because they surface these non-obvious interactions.

**The application-constraint mapping (Table 3).** The survey extends the taxonomy to specific use cases by mapping applications to the constraints that dominate them:

- **Real-time interaction** (chatbots, voice interfaces, translation) demands low latency and low inference runtime—the model must respond in milliseconds.
- **Content generation and processing** (summarization, sentiment analysis, text classification, autocompletion) prioritizes fast inference and minimal resource use.
- **Edge inference and privacy** adds the constraint of on-device operation without cloud connectivity, which elevates memory and storage space as primary concerns.

This mapping closes the loop: a practitioner starts with an application, identifies the dominant constraints, and uses Tables 1 and 2 to locate the relevant families of techniques.

---

#### Model Architectures (Section 2 of the Survey)

The architecture section covers three approaches to building models that are inherently efficient, distinguished by their level of automation and the aspect of the architecture they target.

##### Lightweight Architectures

Lightweight architectures are **hand-designed structural innovations** that reduce parameter count and computational cost while attempting to preserve representational capacity. The survey organizes these by the standard encoder-decoder taxonomy:

**Encoder-only lightweight models** are mostly optimized versions of BERT (Devlin et al., 2019). The encoder-only architecture is naturally suited for understanding tasks—text classification, named entity recognition, question answering—where the model processes the entire input before producing a label. Making these models smaller means reducing the cost of the forward pass through all transformer layers.

MobileBERT (Sun et al., 2020) is the flagship example. It introduces an **inverted-bottleneck structure** that maintains a careful balance between the self-attention and feed-forward network components. In a standard transformer, the feed-forward network typically has a hidden dimension 4× larger than the attention dimension, creating a bottleneck where information is compressed before being expanded. MobileBERT inverts this: it keeps the attention dimension large (preserving the model's capacity to represent complex token interactions) while reducing the feed-forward expansion factor. The result is a model that achieves a **4.3× size reduction** and a **5.5× speedup** compared to BERT-base. The insight here is that the feed-forward layers, not the attention mechanism, dominate the parameter count in standard transformers, so reducing the feed-forward expansion ratio provides disproportionate compression benefits.

DistilBERT (Sanh, 2019) and TinyBERT (Jiao et al., 2019) take a different approach: rather than redesigning the architecture, they use **knowledge distillation** to train smaller models that mimic BERT's output. The survey reports that these achieve "more than 96" (presumably meaning they retain more than 96% of BERT's performance while being substantially smaller—the exact figure is cut off in the text). The distinction matters: MobileBERT changes the architecture's structure; DistilBERT and TinyBERT change the training objective but keep a standard transformer architecture at reduced depth and width.

**Decoder-only lightweight models** follow the autoregressive structure of GPT (Radford et al., 2018, 2019) and LLaMA (Touvron et al., 2023b), where the model generates text token by token, each new token conditioned on all previous tokens. These models are designed for generation tasks, and making them efficient requires addressing both the parameter count *and* the sequential nature of autoregressive decoding, which makes latency particularly challenging.

The survey identifies several sub-125M-parameter decoder-only models, each with a distinct design philosophy:

**BabyLLaMA** (Timiryasov and Tastet, 2023a) and **BabyLLaMA-2** (Tastet and Timiryasov, 2024) distill knowledge from multiple teacher models into a 58M-parameter and 345M-parameter model respectively. A key finding highlighted by the survey is that "distillation can exceed teacher models' performance particularly under data-constrained conditions." This is a non-obvious result: when training data is limited (which it often is for specialized domains), a distilled student can outperform its teacher because the distillation process provides a richer training signal than raw next-token prediction—the teacher's output distribution encodes nuanced relationships that would require far more data to learn from scratch. The mechanism is that the teacher's soft probability distribution over the vocabulary provides information about token similarity and plausible alternatives, not just the single correct next token.

**TinyLLaMA** (Zhang et al., 2024), at 1.1B parameters, achieves efficiency primarily through **memory overhead optimization** rather than architectural novelty. The paper specifically mentions FlashAttention (Dao et al., 2022) as the key enabler. FlashAttention is an IO-aware exact attention algorithm that dramatically reduces the memory footprint of the attention computation by fusing operations and carefully managing data movement between GPU memory hierarchies (HBM and SRAM). It computes the same mathematical function as standard attention but reorganizes the computation to avoid materializing the full `$N \times N$` attention matrix in high-bandwidth memory. TinyLLaMA's design philosophy is that you can achieve competitive performance with a relatively standard architecture if you eliminate the memory bottlenecks that make training and inference expensive—the compression comes from the implementation, not the architecture.

**MobiLLaMA** (Thawakar et al., 2024) introduces a **parameter-sharing scheme** that reduces both pre-training and deployment costs without the complexity of designing a novel architecture. Parameter sharing means the same weight matrices are reused across multiple layers or components. The simplest form is weight tying between the embedding layer and the output projection layer (used in many transformer models), but MobiLLaMA extends this idea to share parameters more aggressively. The model achieves its 0.5B-parameter size not by making layers narrower, but by having fewer *unique* parameters—total capacity is lower, but the model can still learn because the shared parameters are applied in different contexts (different layer positions) where they can serve different functions. The survey emphasizes that this approach reduces "both pre-training and deployment costs," which is significant because many architecture innovations improve inference cost but not training cost. Parameter sharing reduces both because fewer total parameters means fewer FLOPs in both the forward and backward passes.

**MobileLLM** (Liu et al., 2024e) combines three complementary efficiency mechanisms: **embedding-sharing** (tying input and output embeddings, which is common), **grouped-query attention** (where multiple attention heads share a single key-value projection, reducing the KV-cache memory during autoregressive generation), and **block-wise weight sharing** (similar to MobiLLaMA's approach but applied at the granularity of transformer blocks rather than individual layers). Grouped-query attention is particularly important for decoder-only models because during autoregressive generation, the keys and values for all previous tokens must be stored in a KV-cache to avoid recomputing them. Standard multi-head attention stores separate K and V for each head; grouped-query attention reduces this to one K-V pair shared across groups of query heads, dramatically reducing the cache size. The survey's inclusion of MobileLLM alongside the other models illustrates a key point: multiple efficiency techniques can be combined in a single architecture, and the interactions between them matter.

The survey's treatment of these models establishes a pattern that recurs throughout the paper: different models make different choices about *which* constraint to optimize and *how* to achieve the optimization, and no single approach dominates across all axes.

##### Efficient Self-Attention Approximations

This section addresses a specific computational bottleneck: the standard dot-product self-attention mechanism has **quadratic complexity `$O(N^2)$` in the sequence length `$N$`**. This means that doubling the input length roughly quadruples the computational cost of the attention layers. For long documents, multi-turn conversations, or any application where the model must process thousands of tokens, this quadratic cost becomes the dominant factor in inference time and memory usage.

The survey organizes approximation techniques by the complexity class they achieve:

**Reducing to `$O(N \log N)$`:**

Reformer (Kitaev et al., 2020) achieves this by replacing the dot-product attention with **locality-sensitive hashing (LSH)**. In standard attention, every query attends to every key, forming a complete bipartite graph. Reformer's insight is that the softmax in attention concentrates probability mass on the keys with the highest dot products, so computing attention to *all* keys is wasteful. LSH groups similar vectors into the same "bucket" with high probability, and attention is only computed within each bucket. The complexity improvement comes from restricting attention to `$O(\log N)$` keys per query rather than `$N$` keys. The survey notes this reduces the complexity "from `$O(N^2)$` to `$O(N \log N)$`," which is a substantial improvement for long sequences—for a 10,000-token document, this reduces the attention cost by roughly three orders of magnitude.

Roy et al. (2021) use a different grouping strategy: **online k-means clustering** applied to the queries and keys, forming a sparse routing module. Rather than hashing, this approach groups tokens based on their semantic similarity as measured by Euclidean distance in the embedding space. Tokens in the same cluster attend to each other; tokens in different clusters do not. The sparsity pattern is data-dependent—the clusters adapt to the content of the specific input sequence.

**Reducing to `$O(N)$` (linear attention):**

Several works achieve linear complexity by re-expressing the attention computation in a form that avoids the explicit `$N \times N$` multiplication:

Katharopoulos et al. (2020) express self-attention as a **linear dot-product of kernel feature maps**. The key mathematical move is applying a non-linear feature map `$\phi$` to the queries and keys before computing the dot product:

$$\text{Attention}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}$$

where `$Q, K, V$` are the query, key, and value matrices, `$\phi$` is a feature map (typically an element-wise activation like ELU+1), and `$\mathbf{1}$` is a vector of ones.

**What it computes:** Instead of computing `$\text{softmax}(QK^T)V$`—which requires materializing the `$N \times N$` attention matrix `$QK^T$`—this formulation computes `$\phi(K)^T V$` first (an `$d \times d$` matrix where `$d$` is the feature dimension), then multiplies by `$\phi(Q)$`. The key insight is a change in the order of operations: by applying the kernel feature map, the multiplication can be regrouped so that the expensive `$N \times N$` intermediate is never formed.

**Why this form:** The property that makes this work is the associativity of matrix multiplication—`$(QK^T)V = Q(K^TV)$`—but this regrouping only produces the same result as softmax attention if the kernel `$\phi$` approximates the softmax. The specific feature map `$\phi(x) = \text{ELU}(x) + 1$` is chosen because it is non-negative (avoiding destructive interference in the accumulated attention), computationally cheap, and empirically provides a good approximation to the softmax's effect of emphasizing large dot products while suppressing small ones.

The survey further notes that transformers with this linear attention "can be viewed as a recurrent neural network which enables faster inference." This is because once `$\phi(K)^T V$` is computed for the sequence so far, processing a new token requires only a constant-time update to this accumulated matrix, rather than recomputing attention over the entire history. For autoregressive generation, this converts the per-token cost from `$O(N)$` to `$O(1)$`.

**State space models and hybrid architectures:**

Building on the linear attention foundation, the survey identifies two recent architectures that have gained significant attention:

**Mamba** (Gu and Dao, 2023; Dao and Gu, 2024) introduces a **selective state space model**. State space models (SSMs) are a classical framework from control theory that model a system's evolution through time using a hidden state that is updated at each step. Traditional SSMs are linear time-invariant—the transition dynamics are the same regardless of the input. Mamba's innovation is making the SSM **input-dependent**: the transition matrices are functions of the current input token, allowing the model to selectively propagate or forget information based on content. This achieves linear complexity in sequence length while, crucially, retaining the ability to perform **content-based reasoning**—deciding which past tokens are relevant to the current token—that standard linear attention can struggle with. The survey positions Mamba as part of an "ongoing trend towards efficient sequence modeling architectures [that] aims to maintain the expressiveness of attention-based models while significantly reducing computational complexity."

**RWKV** (Peng et al., 2023) combines elements of transformers and RNNs with a **linear attention mechanism**. The name stands for Receptance Weighted Key Value, and the architecture is essentially a linear attention that is reformulated to run efficiently in both training (where it can be parallelized like a transformer) and inference (where it can be unrolled like an RNN). This dual nature addresses a tension in efficient architectures: some methods are fast at training but slow at inference, or vice versa. RWKV aims to be efficient in both regimes.

**Encoder-only efficient attention for long documents:**

Longformer (Beltagy et al., 2020) uses a **combination of local windowed attention and task-specific global attention**. Local windowed attention restricts each token to attending only to a fixed-size window of neighboring tokens (computational cost `$O(N \times W)$` where `$W$` is the window size, typically much smaller than `$N$`), which scales linearly with sequence length. The global attention component designates certain tokens (e.g., the `[CLS]` token in BERT-style models) as "global" tokens that attend to and are attended by every other token, providing a mechanism for long-range information flow that is absent from purely local attention. This design reflects a hypothesis about language: most dependencies are local (words modify nearby words), and the few long-range dependencies can be routed through dedicated global tokens.

Wang et al. (2020a) approximate self-attention using a **low-rank matrix decomposition**. The standard attention matrix is `$A = \text{softmax}(QK^T/\sqrt{d})$`, an `$N \times N$` matrix. The low-rank approach approximates this by projecting the `$N \times d$` key and value matrices to `$k \times d$` (where `$k \ll N$`), computing attention at this reduced dimensionality, and then expanding back. The complexity reduces from `$O(N^2)$` to `$O(Nk)$`. The survey reports that "empirically transformers with linear self-attention matches the performance of the original self-attention mechanism across a variety of downstream tasks," which is an important validation: the theoretical complexity improvement translates to practical performance parity.

Xiong et al. (2021) use the **Nyström method** (Nyström, 1930) for approximating the self-attention operation. The Nyström method is a classical technique for approximating large kernel matrices by sampling a subset of "landmark" points. Applied to attention, it means selecting a small number `$m$` of "landmark" tokens, computing the full attention between all tokens and these landmarks, and then using the resulting low-rank factorization to approximate the full attention matrix. This is similar in spirit to the low-rank approach but uses a different mathematical approximation with different theoretical guarantees. The survey reports "strong empirical performances when compared to traditional transformers."

The survey's treatment of efficient attention illustrates a recurring theme: there is no single "best" approach—the choice between LSH-based, kernel-based, state-space, and low-rank approximations depends on the sequence length distribution, the hardware characteristics, and the acceptable accuracy tradeoff for the specific task.

##### Neural Architecture Search Techniques

Neural Architecture Search (NAS) automates the process of finding efficient architectures rather than relying on human designers. The survey acknowledges a practical challenge: NAS has primarily been applied to vision models and BERT-sized language models because the search process itself requires training and evaluating many candidate architectures, which becomes prohibitively expensive for models with over a billion parameters.

The survey notes that prior NAS work concentrated on "vision tasks (Tan and Le, 2019; Zoph and Le, 2016; Wu et al., 2019; Guo et al., 2020) and BERT models (Xu et al., 2021; Jawahar et al., 2023; Ganesan et al., 2021), as these models have comparatively fewer parameters, which reduces the cost of the search process for efficient architectures." The implication is clear: NAS for language models is currently limited to the scale where training a candidate architecture is cheap enough to afford evaluating hundreds or thousands of candidates.

However, the survey identifies two recent works that push NAS toward larger language models:

**MobileLLM** (Liu et al., 2024e) investigates "the impact of model depth (i.e., number of layers) and width (i.e., number of heads) on performance, effectively conducting a targeted architecture search within a smaller parameter range for language models with millions of parameters." Rather than searching over a complex space of novel architectural components, this work systematically explores the fundamental structural hyperparameters—how many layers, how wide each layer should be—for sub-billion-parameter models. This is a "targeted" search because the space is deliberately constrained to the dimensions that matter most for efficiency-performance tradeoffs. The result is an empirical characterization of the depth-width Pareto frontier for small language models.

**Shen et al. (2024c)** reduce the search space by "exploring an appropriate initialization for the search, which helps expedite the convergence of the search process." This addresses the core expense of NAS: if you can start the search closer to a good architecture, you can evaluate fewer candidates. The paper proposes using a strong hand-designed architecture (or a previously discovered good architecture at a different scale) as the starting point for the search, effectively combining human design intuition with automated refinement.

The survey's discussion of NAS is notably brief and forward-looking, reflecting the current state of the field: NAS for LLM-scale models is still in its infancy, and most efficient architectures (MobileLLM, TinyLLaMA, etc.) are still designed by humans. The open challenge is scaling up the search process itself.

##### Small Multi-Modal Models

The survey dedicates a brief section to an emerging trend: applying the same efficiency principles to models that process both text and images (vision-language models or multi-modal models). This section is included because multi-modal models compound the efficiency challenge: they need a vision encoder (often a large convolutional or transformer model) in addition to the language model, so reducing the language model's size is only part of the problem.

The survey identifies several notable efficient multi-modal models organized by their strategy for reducing the vision component:

**Using efficient language models as the backbone:** LLaVA-Next (Liu et al., 2024a), Idefics2 (Laurençon et al., 2024), and InternVL2 (Chen et al., 2023) achieve smaller overall model sizes partly by using "more efficient, smaller language models like Gemma (Team et al., 2024), phi-3-mini (Abdin et al., 2024)." The insight is compositional: if you can reduce the language model from 7B to 2B parameters while maintaining most of its reasoning capability, the multi-modal model shrinks correspondingly.

**Compressing the vision encoder:** InternVL2 "leverages outputs from intermediate layers of large visual encoders while discarding the later blocks." Large vision encoders (like ViT-L or ViT-H) have many layers, with later layers typically learning high-level semantic features. By using only intermediate layer outputs, InternVL2 reduces the encoder's parameter count and computational cost while still capturing visual features that are sufficient when combined with a strong language model. PaliGemma (Beyer et al., 2024) and Mini-Gemini (Li et al., 2024c) take this further by adopting inherently "lightweight vision encoders" from the start.

**Eliminating the vision encoder entirely:** The most radical approach is to have the model process images without a dedicated vision encoder. Chameleon (Team, 2024a) uses a VQ-VAE (Vector Quantized Variational AutoEncoder) to encode images into discrete tokens—essentially the same kind of tokens as text—and then processes them with the same transformer that handles text. This "early fusion" approach means there is no separate vision tower; the transformer learns to process both modalities natively. Mono-InternVL (Luo et al., 2024a) uses an MLP to generate visual tokens for image patches and incorporates "a modality-specific feed-forward network, termed multi-modal Mixture-of-Experts, to differentiate between modalities." This is a lighter-weight alternative to a full vision transformer: a simple multi-layer perceptron generates token embeddings from image patches, and the mixture-of-experts layers allow some parameters to specialize in visual processing while others handle text.

The survey's inclusion of multi-modal models, though brief, signals an important direction: as language models become more efficient, the bottleneck shifts to the other modalities they interface with, and holistic efficiency requires optimizing all components simultaneously.

---

#### Training Techniques (Section 3 of the Survey)

The training section covers methods that reduce the resource requirements of learning, separated into pre-training (learning from scratch on large corpora) and fine-tuning (adapting a pre-trained model to a specific task).

##### Pre-Training Techniques

Pre-training a language model from scratch is typically the most computationally expensive phase, requiring thousands of GPU-hours for even modestly sized models. The techniques surveyed here aim to reduce this cost without sacrificing final model quality.

**Mixed-Precision Training:**

The fundamental idea is to use lower-precision arithmetic for the bulk of computation while maintaining higher precision where it matters for numerical stability.

Automatic Mixed Precision (AMP), introduced by Micikevicius et al. (2018), "initially keeps a master copy of weights in 32-bit floating-point (FP32) precision while performing arithmetic operations in 16-bit floating-point (FP16) precision." The forward and backward passes compute gradients using FP16, which halves memory bandwidth and enables faster arithmetic on hardware with specialized FP16 units, but the optimizer updates are accumulated in FP32 to preserve small gradient values that would underflow in FP16. The "master copy" is the canonical representation of the model's parameters; the FP16 version used during forward/backward passes is a temporary reduced-precision view.

This matters for SLMs because the memory savings from FP16 arithmetic can mean the difference between fitting a model on a single GPU and requiring model parallelism. For a 1B-parameter model, storing parameters in FP16 instead of FP32 saves 2GB of memory.

However, the survey notes a critical limitation: "recent work (Rae et al., 2021) has observed accuracy losses due to [FP16's] limited numerical range." FP16 can represent values up to 65,504, which is insufficient for some activation values and gradient magnitudes that occur during training. When values exceed this range, they overflow to infinity; when they fall below the minimum representable value, they underflow to zero—both of which corrupt the training signal.

Brain Floating Point (BFLOAT16), proposed by Burgess et al. (2019), addresses this by using the same exponent range as FP32 (8 bits of exponent) but fewer mantissa bits (7 bits instead of FP32's 23). This means BFLOAT16 can represent values over a much wider dynamic range—up to `$3.4 \times 10^{38}$`—at the cost of reduced precision for the values it does represent. The survey states BFLOAT16 "has demonstrated superior training performance and representation accuracy compared to FP16" because the extended dynamic range prevents the overflow and underflow issues without requiring the complexity of loss scaling (a technique used in FP16 training to artificially shift gradients into the representable range).

The survey further notes the hardware evolution enabling these techniques: "Modern GPU architectures have further advanced mixed-precision capabilities through specialized Tensor Cores. For instance, while earlier generations supported FP16 and BFLOAT16, NVIDIA's latest Hopper architecture introduces support for 8-bit floating-point (FP8) precision (Luo et al.)." FP8 represents a further 2× reduction in memory and bandwidth over 16-bit formats, potentially enabling training of models twice as large (or training existing models on half the hardware), though at increased risk of numerical precision issues.

**Optimization and Stability Techniques:**

The survey briefly catalogs the optimizers and stabilization methods used in modern language model training:

Adam (Diederik, 2014) and AdamW (Loshchilov and Hutter, 2019) are the standard adaptive optimizers. Adam maintains per-parameter learning rates based on estimates of first and second moments of the gradients. AdamW's contribution is decoupling weight decay from the adaptive learning rate computation—in standard Adam, weight decay is implemented as L2 regularization, which interacts with the adaptive learning rates in ways that can reduce the effective regularization. AdamW applies weight decay directly to the parameters, which the survey notes results in better generalization.

Memory-efficient variants like Adafactor (Shazeer and Stern, 2018) and Sophia (Liu et al., 2024b) are specifically intended to reduce the optimizer's memory footprint. Standard Adam stores two moment estimates per parameter (the mean and uncentered variance of the gradient), which doubles or triples the memory required beyond the parameters themselves. Adafactor reduces this by using factored estimates of the second moment—instead of storing a full matrix, it stores row and column sums—achieving sublinear memory in the parameter count. Sophia goes further by using a stochastic second-order method that requires only a Hessian diagonal estimate, reducing memory while potentially achieving faster convergence.

Gradient clipping (Zhang et al., 2020) prevents exploding gradients by rescaling any gradient whose norm exceeds a threshold. This is "widely used" because the scale of gradients in deep transformers can vary dramatically across layers and training steps, and a single large gradient update can destabilize training. The clipping threshold is typically set to a small value (e.g., 1.0) and is a hyperparameter that must be tuned.

Careful initialization strategies provide "a good starting point for model training" by setting initial weights such that the signal propagating through the network maintains a stable variance. Poor initialization can cause activations or gradients to vanish or explode in the early stages of training, dramatically slowing convergence.

**Distributed Training:**

The survey identifies parallelism strategies as essential for pre-training models across multiple GPUs or nodes, enabling training with larger batch sizes and faster iteration:

Zero Redundancy Data Parallelism (ZeRO) (Rajbhandari et al., 2020) offers three progressive stages that partition different components of the training state:

- **ZeRO-1** partitions optimizer states across data-parallel processes. In standard data parallelism, each GPU maintains a full copy of the optimizer state (which can be 2-3× the model size for Adam). ZeRO-1 distributes these states, so each GPU stores only `$1/N$` of the optimizer states, reducing per-GPU memory by roughly `$2-3\times$`.
- **ZeRO-2** additionally partitions gradients, further reducing memory.
- **ZeRO-3** additionally partitions model parameters themselves. Each GPU stores only `$1/N$` of the parameters and communicates with other GPUs to gather the parameters needed for the current computation.

The survey notes PyTorch's Fully Sharded Data Parallel (FSDP) (Zhao et al., 2023b) "implements similar concepts," making these techniques accessible within the PyTorch ecosystem.

The key benefit for SLMs is enabling "training with larger batch sizes, significantly improving efficiency and scalability." Larger batch sizes improve GPU utilization and can accelerate convergence (to a point), but they increase memory requirements linearly. ZeRO and FSDP decouple batch size from per-GPU memory, allowing SLM training on clusters of commodity GPUs.

##### Fine-Tuning Techniques

Fine-tuning adapts a pre-trained model to a specific downstream task using a smaller, task-specific dataset. The challenge for SLMs is that full fine-tuning—updating all parameters—can be as expensive as pre-training a small model and risks "catastrophic forgetting," where the model loses its general capabilities while learning the new task.

**Parameter-Efficient Fine-Tuning (PEFT):**

PEFT methods update only a small subset of parameters or inject lightweight trainable modules, keeping the bulk of the pre-trained model frozen. The survey identifies the primary benefits: reduced computational cost, preservation of the model's original knowledge, reduced overfitting on small datasets, and improved flexibility (multiple task-specific adaptations can be stored without duplicating the entire model).

**LoRA** (Low-Rank Adaptation; Hu et al., 2021) is the most prominent PEFT method. The core idea is that the weight updates during fine-tuning have low "intrinsic rank"—they can be represented as the product of two much smaller matrices:

$$W' = W + \Delta W = W + BA$$

where `$W \in \mathbb{R}^{d \times k}$` is a frozen pre-trained weight matrix, `$B \in \mathbb{R}^{d \times r}$` and `$A \in \mathbb{R}^{r \times k}$` are trainable low-rank factors, and `$r \ll \min(d, k)$` is the rank (typically chosen between 4 and 64).

**What it computes:** During fine-tuning, only `$B$` and `$A$` are updated via gradient descent; `$W$` remains frozen. The forward pass computes `$h = Wx + BAx$` (or equivalently pre-computes `$W' = W + BA$` at inference time). The decomposition factorizes each weight update into a composition of two low-dimensional transformations.

**Why this form:** The low-rank assumption is motivated by the empirical observation that the changes needed for task adaptation are much simpler than the full model's representational capacity. If adapting to a new task only requires adjusting behavior in a few "directions" in activation space, then a low-rank update can capture those adjustments while using far fewer parameters—`$r(d + k)$` trainable parameters instead of `$d \times k$`. A full-rank update would be overparameterized for the adaptation, wasting computation and risking overfitting when the fine-tuning dataset is small.

**Prompt Tuning** (Lester et al., 2021) inserts learnable "prompt" embeddings into the input sequence rather than modifying model weights at all. A small number of trainable vectors (typically 5–100 tokens worth) are prepended to the input, and only these embeddings are updated during fine-tuning. The model's entire parameter set remains frozen. This is extremely parameter-efficient (often only tens of thousands of trainable parameters regardless of model size) but can be less expressive than LoRA because the adaptation is limited to what can be encoded in a short prefix.

**Llama-Adapter** (Zhang et al., 2023b; Gao et al., 2023) inserts learnable prompts specifically into LLaMA's attention blocks rather than just at the input. This provides more expressiveness than input-level prompt tuning because the adaptation can influence processing at every layer, while still being parameter-efficient because only the injected prompts are trainable.

**Dynamic Adapters** represent a further development where multiple adapters (task-specific trainable modules) are combined using a mixture-of-experts architecture (Kong et al., 2024; Feng et al., 2024; Gou et al., 2023; Liu et al., 2023b; Luo et al., 2024b). The motivation is multi-task learning: rather than training a separate set of adapters for each task and selecting which one to use, dynamic adapters learn to automatically route inputs to the appropriate adapter experts. This "enables multi-tasking and prevents forgetting" (Han et al., 2024; Yang et al., 2024) by having specialized expert modules that can be selectively engaged without interfering with each other.

The survey's treatment of PEFT establishes an important design spectrum: at one extreme, prompt tuning modifies only the input representation; at the other, LoRA modifies the weight matrices themselves (albeit through a low-rank bottleneck). Dynamic adapters sit in between, modifying intermediate representations through injected modules.

**Data Augmentation:**

Data augmentation during fine-tuning aims to increase the effective size, diversity, and quality of the training data when labeled data is scarce. The survey identifies several generation-based approaches:

**AugGPT** (Dai et al., 2023) rephrases training samples using ChatGPT. The idea is straightforward: for each training example, prompt an LLM to produce a semantically equivalent but syntactically different version, effectively multiplying the dataset size without requiring human annotation.

**Evol-Instruct** (Xu et al., 2023) uses "multistep revisions to generate diverse, open-domain instructions with increased complexity." Rather than simple rephrasing, this approach iteratively rewrites instructions to be more complex, more specific, or more constrained, creating a curriculum of increasing difficulty from a small seed set.

**Reflection-Tuning** (Li et al., 2023a, 2024a) "enhances data quality and instruction-response consistency for instruction tuning by refining both instructions and responses using GPT-4 based on predefined criteria." This is a quality-improvement approach rather than a quantity-improvement approach: the LLM critiques and revises existing instruction-response pairs to fix inconsistencies, improve clarity, or add detail, producing a cleaner dataset from noisy initial data.

**FANNO** (Zhu et al., 2024) augments instructions and generates responses by "incorporating external knowledge sources through retrieval-augmented generation." When the training data is limited, augmenting each example with relevant facts retrieved from a knowledge base can provide the model with richer context, improving its ability to generalize.

**LLM2LLM** (Lee et al., 2024b) takes an adaptive approach: it "generates more hard samples based on model prediction on training data during training." The model's own errors identify which examples it struggles with, and additional training data is synthesized specifically for those difficult cases. This is a form of online curriculum learning where the data augmentation is targeted at the model's current weaknesses.

The survey notes that data augmentation is "also effective for synthesizing new data when training data is limited, such as for low-resource languages (Whitehouse et al., 2023), medical and clinical applications (Chintagunta et al., 2021), and privacy-sensitive data (Song et al., 2024), enabling models to generalize better and perform more robustly in constrained settings." This highlights that data augmentation serves dual purposes: it can improve performance when data is scarce, and it can enable fine-tuning in domains where collecting real data is expensive, regulated, or ethically constrained.

---

#### Model Compression Techniques (Section 4 of the Survey)

Model compression is the most direct path from an LLM to an SLM: take a large, fully trained model and reduce its size or inference cost without retraining from scratch. The survey organizes compression methods into three categories, distinguished by *what* aspect of the model they modify.

##### Pruning Techniques

Pruning removes parameters from a trained model to reduce its size and computational cost. The survey distinguishes between two granularities:

**Unstructured pruning** removes individual weights independently, producing a sparse weight matrix. The advantage is fine-grained control: you can remove exactly the least important weights, keeping all the important ones regardless of their position. The disadvantage is that the resulting sparse matrices require "specialized hardware or algorithms to maximize computational benefits" (Frantar and Alistarh, 2023)—standard dense matrix multiplication on GPUs does not automatically accelerate sparse matrices, and achieving wall-clock speedups requires sparse matrix libraries or hardware support for sparsity.

**SparseGPT** (Frantar and Alistarh, 2023) is the flagship unstructured pruning method for large language models. It "reformulates the pruning task as a sparse regression problem, optimizing both the remaining and pruned weights using a layer-wise approximate regression solver." The core idea is:

1. Process the model one layer at a time.
2. For each layer, collect the inputs to that layer by running a calibration dataset through the model up to that point.
3. Formulate pruning as: given the layer's weight matrix `$W$` and the collected inputs `$X$`, find a sparse weight matrix `$\hat{W}$` that minimizes `$\|WX - \hat{W}X\|^2$`—i.e., produces the same outputs as the original layer given the same inputs.
4. Solve this regression problem approximately using a greedy algorithm that prunes weights one at a time, each time updating the remaining weights to compensate for the error introduced.

The survey highlights that SparseGPT "can efficiently handle large-scale models like OPT-175B and BLOOM-176B," which is remarkable because it means pruning can be applied to models at the very edge of what can be trained, in a single pass without iterative retraining.

**Wanda** (Sun et al., 2023) simplifies the pruning criterion by "incorporating both weights and activations into consideration during the pruning process, and eliminates the need of weight updates." The pruning score for a weight `$W_{ij}$` is `$|W_{ij}| \cdot \|X_j\|_2$`, where `$X_j$` is the norm of the activations corresponding to that weight. This means a weight is pruned if it is small *and* its corresponding input activations are small—a small weight that processes important features is kept. Wanda requires no update step (unlike SparseGPT's regression), making it faster and simpler while achieving competitive sparsity-quality tradeoffs.

**n:m pruning** (Zhou et al., 2021) is a structured form of unstructured pruning: "pruning exactly n weights out of every m, balancing pruning flexibility and computational efficiency for significant speedups." For example, 2:4 pruning keeps 2 out of every 4 weights, achieving 50% sparsity. This pattern is specifically supported by NVIDIA's TensorRT and A100 GPUs, which can execute 2:4 sparse matrix multiplications at double the throughput of dense multiplications, providing actual wall-clock speedup. The survey notes this technique can "also be applied in edge AI applications on NVIDIA Jetson Nano to enhance power efficiency and optimize model size," extending the benefits to low-power devices.

**Structured pruning** removes entire groups of parameters—neurons, attention heads, layers—in a way that preserves the dense structure of the remaining matrices. The advantage is that the pruned model is simply a smaller dense model, requiring no special hardware or software support for acceleration. The disadvantage is coarser granularity: you cannot remove individual weights, only predefined structural units.

The survey identifies several directions within structured pruning:

- **Neuron sparsity:** Li et al. (2023b) "observes prevalent sparsity in feed-forward networks," meaning that a large fraction of neurons in the intermediate layers of transformers have near-zero outputs for most inputs and can be removed with minimal impact. Liu et al. (2023e) proposes "contextual sparsity"—using small neural networks to dynamically decide which neurons to activate based on the specific input, rather than statically removing them. This is sparsity at inference time rather than pruning at model modification time.

- **Activation function modification:** Mirzadeh et al. (2024) change the activation functions in pre-trained models to ReLU (Rectified Linear Unit) and then fine-tune. ReLU naturally induces sparsity because it outputs zero for negative inputs; other activations like GELU used in modern transformers produce small but non-zero outputs. By switching to ReLU and fine-tuning, the models learn to operate with sparser activations, which can then be exploited by pruning.

- **Layer and head redundancy:** Several works investigate redundancy at the level of entire transformer layers (Sajjad et al., 2023; Xia et al., 2022). If some layers contribute little to the final output, they can be dropped entirely, reducing depth. Similarly, Michel et al. (2019) and Voita et al. (2019) find that many attention heads are redundant and can be pruned. This is the most aggressive form of structured pruning because it removes the largest structural units.

- **Input-dependent pruning:** FastGen (Ge et al., 2024) and contextual sparsity methods (Liu et al., 2023e) decide what to prune based on the specific input being processed, rather than pruning once and reusing the same sparse model for all inputs. This is more flexible but "should be considered along with the challenges of efficient implementation" because dynamic sparsity patterns are harder to accelerate in hardware.

The survey's Appendix A provides further discussion, noting that n:m pruning can be applied not just in datacenter GPUs but "also in edge AI applications on NVIDIA Jetson Nano to enhance power efficiency and optimize model size." This extension to edge devices is significant because it means pruning techniques developed for datacenter models can directly benefit on-device deployment where both computational throughput and energy efficiency are constrained.

##### Quantization

Quantization reduces the numerical precision of model parameters and activations, typically from 16-bit floating-point to 8-bit, 4-bit, or even lower. The immediate benefit is reduced memory footprint: a 4-bit quantized model requires one-quarter the storage of its 16-bit counterpart. The secondary benefit is faster inference when hardware supports efficient low-precision arithmetic.

**Weight-only quantization** quantizes just the model parameters while keeping activations in higher precision. GPTQ (Frantar et al., 2022) "focuses on layer-wise weight-only quantization, using inverse Hessian matrices to minimize the reconstruction error." The approach is:

1. Process the model one layer at a time.
2. For each layer, approximate the Hessian matrix `$H$` of the loss with respect to the layer's outputs, using a calibration dataset. The Hessian captures which directions in the output space are most sensitive to weight perturbations.
3. Quantize the weights to, say, 4-bit integers with a per-channel scale factor, sequentially quantizing each weight and adjusting the remaining weights using the inverse Hessian to compensate for the quantization error.

The key insight is that not all weights are equally important—the Hessian identifies which weights affect the loss most, and those weights receive more careful treatment during quantization. The survey notes GPTQ enables running large models like OPT-175B in significantly reduced memory.

**Weight-and-activation quantization** quantizes both model parameters and the activations flowing through the network during inference. This enables using fast integer matrix multiplication kernels that can be 2–4× faster than their floating-point counterparts. The survey identifies several methods that address this more challenging problem:

**AWQ** (Lin et al., 2024) and **ZeroQuant** (Yao et al., 2022) "take activation into account to assess the importance of weights, enabling more effective optimization for weight quantization." Concretely, these methods observe that a small fraction of weights (often corresponding to outlier channels) are significantly more important than the rest because they process large-magnitude activations. By identifying these salient weights via activation statistics and preserving them at higher precision (or scaling them differently), the overall quantization error is substantially reduced.

**SmoothQuant** (Xiao et al., 2023) addresses a specific challenge: "activation outliers that fall outside the typical activation distribution." In transformer models, some activation channels consistently produce values 10–100× larger than the typical range. If quantized naïvely, these outliers dominate the quantization range, causing most values to be quantized to zero. SmoothQuant "smoothes activation outliers by migrating quantization difficulty from activations to weights"—it applies a per-channel scaling factor that reduces the magnitude of outlier activations and compensates by increasing the magnitude of the corresponding weights, achieving a mathematically equivalent computation with an easier-to-quantize activation distribution.

**SpinQuant** (Liu et al., 2024d) "introduces rotation matrices to transform outliers into a new space." Rather than scaling, it rotates the activation space so that outliers are distributed more uniformly across dimensions. The rotation is a learned orthogonal transformation applied before quantization, and its inverse is applied after dequantization, preserving the mathematical equivalence.

**K/V Cache quantization** (Hooper et al., 2024; Liu et al., 2024f; Yue et al., 2024) specifically targets the key-value cache used during autoregressive generation. As noted in the architecture discussion, the KV-cache stores the keys and values for all previous tokens, and its size grows linearly with both sequence length and batch size. For long conversations or documents, the KV-cache can exceed the model parameters in size. Quantizing the cache to 4-bit or even 2-bit precision dramatically reduces this memory, "enabling efficient long-sequence length inference."

**Quantization-Aware Training (QAT)** incorporates quantization into the training process rather than applying it post-hoc. The survey highlights two methods: LLM-QAT (Liu et al., 2023d) and EdgeQAT (Shen et al., 2024b). Both "adopt distillation with float16 models to recover the quantization error"—the quantized model is trained to mimic the full-precision model's outputs, with the quantization operations simulated during training so the model learns to produce outputs that are robust to quantization noise. While more expensive than post-training quantization (since it requires training), QAT typically achieves higher accuracy at very low bit-widths where post-training methods struggle.

The survey also notes recent work implementing quantized LLMs on mobile devices and FPGAs (Shen et al., 2024a,b; Zeng et al., 2024), "demonstrating the effectiveness and efficiency of the weight and activation quantization for LLMs" on hardware far more constrained than datacenter GPUs.

##### Knowledge Distillation

Knowledge distillation trains a smaller "student" model to replicate the behavior of a larger "teacher" model. The survey focuses specifically on distillation where "one or multiple white-box teacher language model[s]" are available—meaning the teacher's internal representations, output probabilities, or both are accessible during training.

The classical form, introduced by Hinton et al. (2015), trains the student to match the teacher's **softened output distribution**:

$$p_i^{\text{teacher}} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

where `$z_i$` are the teacher's logits and `$T$` is a temperature parameter (typically `$T > 1$`). The student is trained to minimize the KL divergence between its own softened distribution and the teacher's.

**What it computes:** The student learns to match not just the teacher's most likely prediction, but the full distribution over the vocabulary, capturing the teacher's uncertainty and the relative plausibility of alternative tokens. The temperature `$T$` controls how much the distribution is "softened"—higher temperatures make the distribution more uniform, emphasizing the teacher's secondary predictions (which carry information about token relationships) over its top prediction.

**Why this form:** Matching only the teacher's hard prediction (argmax) discards most of the information in the teacher's output. The teacher's probability distribution encodes rich knowledge: that "dog" and "puppy" are similar (they receive similar probabilities in many contexts), that certain continuations are syntactically valid but semantically odd (moderate probability), and that other continuations are completely out of distribution (near-zero probability). The KL divergence with temperature scaling captures this full distribution, providing a richer training signal than cross-entropy with hard labels.

**Distillation of small decoder-only models:**

BabyLLaMA (Timiryasov and Tastet, 2023b) is "among the first to develop a compact 58M parameter language model using a Llama model as the teacher." The survey highlights the key finding: "distillation from a robust teacher can outperform traditional pre-training on the same dataset." This is significant because it suggests that for very small models, the bottleneck is not the quality of the training data but the ability to extract complex patterns from it—the teacher's output distribution provides a learning curriculum that the student cannot discover from raw text alone.

Gu et al. (2024) introduce "modifications in the distillation loss, which enables the student models to generate better quality responses with improved calibration and lower exposure bias." The loss modifications address a subtle issue: during autoregressive generation at inference time, the student sees its own potentially erroneous outputs as context, but during distillation training, it sees the teacher's correct outputs (teacher forcing). This mismatch, called exposure bias, causes errors to compound at inference time. The modified loss reduces this discrepancy.

**Sequence-level distillation** can be improved using "a generalized version of f-divergences" (Wen et al., 2023), which provides a broader family of divergence measures beyond KL divergence. Different f-divergences emphasize different aspects of the distributional mismatch—some are more sensitive to mode collapse, others to excessive spread—and choosing the appropriate divergence for the specific student-teacher pair can improve distillation quality.

**Layer-wise distillation** (Liang et al., 2023) extends distillation beyond the output layer to intermediate representations. "Task-aware filters" distill only the "task-specific knowledge from the teacher," meaning the student learns to match the teacher's representations only for the aspects most relevant to the target task, ignoring other representational dimensions that may be task-irrelevant. This is more sample-efficient than distilling all layers and all dimensions indiscriminately.

**Multi-teacher distillation** (Wan et al., 2024a,b) fuses knowledge from multiple teacher models. Rather than distilling from a single teacher, the student learns from "strategically merging their output probability distributions." This can produce a student that combines the strengths of different teachers—for example, one teacher may be strong at factual knowledge, another at reasoning. The fusion strategy (how to weight and combine teachers' predictions) is itself a design choice.

**Distillation beyond label matching:**

The survey identifies a critical practical limitation: distillation strategies are "primarily effective when (1) the teacher and the student language model share the same tokenizer and (2) the teacher's pre-training data is available." When models use different tokenizers, their vocabulary spaces are different, and the teacher's output distribution cannot be directly compared to the student's. Boizard et al. (2024) address this with "a universal logit distillation loss inspired from the optimal transport literature," which maps between different token distributions using optimal transport—finding the minimum-cost way to transform the teacher's distribution over its vocabulary into a distribution over the student's vocabulary.

**Combined pruning and distillation** is an iterative process where "an iterative step of pruning a large language model followed by retraining with distillation losses, can enable strong smaller models" (Sreenivas et al., 2024; Muralidharan et al., 2024). The idea: prune to reduce size, then distill to recover the lost quality, then prune again, and so on. Each distillation step provides supervision that helps the pruned model compensate for its reduced capacity, allowing more aggressive pruning than would be possible in a single pass.

**Distilling reasoning capabilities:**

Recent work has explored using "rationales" or chain-of-thought reasoning as additional supervision during distillation. Hsieh et al. (2023) "find that using 'rationales' as an additional source of supervision during distillation makes it more sample-efficient" and that "the distilled model outperforms large-language models on commonly used NLI, Commonsense QA and arithmetic reasoning benchmarks." The teacher not only provides the correct answer but also explains its reasoning step-by-step, and the student learns to replicate both the answer and the reasoning process. Dai et al. (2024), Magister et al. (2023), Ho et al. (2023), and Fu et al. (2023) similarly "distill the reasoning chain from a larger language model to a smaller language model along with the label information," producing student models with "improved arithmetic, multi-step math, symbolic and commonsense reasoning abilities."

This is a particularly important direction for SLMs because reasoning is one of the capabilities that most distinguishes large models from small ones. If reasoning can be effectively distilled—transferred from a teacher that can perform multi-step deduction to a student that cannot learn it from raw data alone—then the capability gap between SLMs and LLMs narrows substantially for tasks that require structured thinking.

---

#### Summary of the Technical Approach Taxonomy

The survey's technical contribution is not a method but an organizational framework. The taxonomy serves several functions that together make the fragmented SLM literature navigable:

1. **Technique categorization by development stage:** Pre-processing (architecture), in-processing (training), post-processing (compression) provides a chronological map of when each technique applies.

2. **Constraint mapping:** The checkmarks in Table 1 explicitly link each technique to the specific resource bottlenecks it addresses, enabling constraint-driven navigation of the literature.

3. **Application grounding:** Table 3 maps real-world use cases to their dominant constraints, closing the loop between abstract optimization goals and concrete deployment scenarios.

4. **Tradeoff awareness:** The taxonomy's structure—particularly the blank cells in Table 1—surfaces the fact that no technique addresses all constraints simultaneously. This is the survey's key analytical insight: SLM optimization is inherently multi-objective, and the right approach depends on which constraints bind for a particular deployment context.

## 4. Key Insights and Innovations

### Innovation 1: A Constraint-Aware Taxonomy That Transforms Fragmented Literatures Into a Navigable Design Space

The dominant prior approach to small language models was to treat each efficiency technique—pruning, quantization, efficient attention, knowledge distillation—as an independent research thread with its own vocabulary, baselines, and evaluation protocols. A practitioner facing a specific deployment constraint (say, fitting a model into 2GB of mobile device RAM with sub-100ms latency) would encounter dozens of papers with no systematic way to determine which techniques applied to their bottleneck, how they combined, or where they traded off against each other. Existing surveys (Zhu et al., 2023 on model compression; Rogers et al., 2020; Min et al., 2021 on LLM capabilities) covered subsets of the landscape but never connected architectural design, training optimization, and post-hoc compression under a unified framework.

This survey's central conceptual move is its **two-axis taxonomy** (Tables 1 and 2) that classifies every technique by both *what kind of optimization it is* (architecture, training, or compression) and *which specific constraint it addresses* (inference runtime, memory, storage, latency). This is not merely a categorization exercise—it is a diagnostic tool. The checkmark matrix in Table 1 functions as a constraint-driven lookup: a practitioner with a memory bottleneck looks down the "Memory" column and finds that lightweight architectures, efficient attention, pruning, and quantization all apply, but knowledge distillation and data augmentation do not (at least not directly). The blank cells in the matrix are as analytically valuable as the filled ones because they surface what the taxonomy's authors call the central insight of the survey:

> "It is important to note that progress on any one of these goals does not necessarily imply progress on the others. In fact, there are often trade-offs between them."

The concrete example of quantization-aware training being slower than full-precision training despite reducing memory (Section 1) makes this tradeoff tangible: a technique that improves one constraint metric can worsen another. This is a **fundamental reframing** of the SLM problem from single-objective optimization ("make the model smaller") to multi-objective constrained optimization ("make the model smaller while respecting latency budget X, memory ceiling Y, and training compute limit Z"). Prior work largely evaluated techniques on accuracy alone (e.g., perplexity after pruning, benchmark scores after quantization), treating resource savings as a secondary benefit rather than a primary optimization target. By making constraints first-class citizens of the taxonomy, the survey shifts the field's evaluation philosophy from "how much can we compress without losing accuracy?" to "which dimensions of efficiency matter for this specific deployment, and which combination of techniques jointly optimizes them?"

The application-constraint mapping in Table 3 extends this diagnostic function to the practitioner level: real-time chatbots face latency and inference runtime as primary constraints; on-device medical NLP faces memory and privacy as primary constraints; energy-efficient edge AI faces all of these plus power consumption. The taxonomy thus serves as a bridge between abstract optimization goals and concrete deployment requirements—a practitioner can enter through their application, identify their dominant constraints from Table 3, and navigate to the relevant technique families using Table 1.

The significance of this contribution is organizational and synthetic rather than empirical, but it addresses a genuine intellectual gap: without such a taxonomy, the SLM literature is a collection of point solutions; with it, the literature becomes a design space with known tradeoffs and composable components. This is a **fundamental intellectual scaffold** rather than an incremental refinement—it changes how researchers and practitioners think about the problem, not just what methods they apply.

### Innovation 2: The Identification of Multi-Objective Tradeoffs as the Central Challenge of SLM Design, Not a Footnote

Before this survey, the SLM literature largely treated tradeoffs as implementation details or second-order concerns. A pruning paper would report compression ratio and accuracy; a quantization paper would report bit-width and perplexity; an efficient architecture paper would report parameter count and benchmark scores. The implicit assumption was that better compression (higher sparsity, lower precision, fewer parameters) at a given accuracy level was uniformly desirable. The possibility that a 4-bit quantized model might achieve higher accuracy than a pruned model at the same parameter count but with higher latency—or that an efficient architecture might reduce memory but increase training cost—was rarely surfaced or systematically compared.

The survey makes this tradeoff structure **explicit and central** through two mechanisms. First, the taxonomy itself (Table 1's column structure) forces every technique to declare which constraints it helps and, by omission, which it does not. The blank cells are the tradeoffs, and the survey draws attention to them explicitly:

> "For instance, memory-efficient training methods like quantization-aware training (Dettmers et al., 2022a, 2024) are often slower than their full-precision counterparts."

This is a concrete, named tradeoff: QAT trades training speed for memory efficiency. But the structure generalizes: pruning reduces inference memory and storage but does not reduce training compute (you still had to train the full model to prune it); knowledge distillation can reduce the student's inference cost but adds the cost of training the student and may require access to the teacher's training data; neural architecture search can find efficient architectures but is itself computationally expensive. Every technique in the survey has a profile of benefits and costs, and the taxonomy makes these profiles visible at a glance.

Second, the survey's evaluation framework (Section 5) organizes metrics by constraint type rather than by task accuracy alone. Table 2 maps datasets and metrics to the constraints they measure: inference time and throughput for latency, peak memory usage and compression ratio for memory, privacy budget and noise level for privacy, energy efficiency ratio and thermal efficiency for energy. This is a departure from the dominant evaluation paradigm in NLP, where accuracy on benchmark suites (GLUE, SuperGLUE, SQuAD, etc.) is the primary metric and efficiency is reported separately as auxiliary information. By elevating efficiency metrics to equal status with accuracy metrics—and organizing them by the constraint dimension they measure—the survey reframes evaluation from "how well does the model perform?" to "how well does the model perform *given the constraints under which it must operate*?"

The intellectual significance of this move is that it converts tradeoffs from a nuisance to be minimized into the **central design problem of SLMs**. The right question is not "which technique is best?" but "which combination of techniques jointly optimizes accuracy under my specific constraint profile?" This is a **fundamental conceptual shift** analogous to the transition from optimizing for raw performance to optimizing for performance-per-watt in processor design—it redefines what "better" means in a way that is more aligned with real-world deployment constraints where resources are scarce and multi-dimensional.

This insight is not empirically validated by a single table or figure—it is a structural property of the taxonomy and the evaluation framework—but it is the intellectual backbone that gives the entire survey its practical relevance. Without it, the survey would be a catalog of techniques; with it, it becomes a decision-support tool.

### Innovation 3: The Identification of an End-to-End Pipeline Spanning Architecture Through Training Through Compression, With Characterized Interactions

Prior work treated architecture design, training optimization, and model compression as distinct phases addressed by separate research communities. Architecture papers (MobileBERT, Mamba, TinyLLaMA) rarely discussed how their designs interact with post-training quantization. Pruning papers (SparseGPT, Wanda) rarely discussed whether their sparsity patterns are compatible with efficient attention approximations. Knowledge distillation papers rarely discussed whether the distilled student benefits from parameter-efficient fine-tuning for downstream tasks. Each subcommunity optimized its own local objective, and the question of how techniques compose across the full development pipeline was largely unasked.

The survey is the first work to present these three stages as a **continuous pipeline** where choices at each stage constrain and enable choices at subsequent stages. The organization of Sections 2 (architecture), 3 (training), and 4 (compression) is not arbitrary—it reflects the temporal flow of model development, and the survey implicitly argues that optimizations should be considered holistically rather than independently. Several specific cross-stage interactions are surfaced:

- **Architecture-compression compatibility:** Lightweight architectures like MobileLLM already incorporate efficiency mechanisms (grouped-query attention, embedding sharing) that reduce the parameter count. Applying post-training quantization on top of an already-efficient architecture may yield diminishing returns because there is less redundancy to exploit, or it may interact poorly with architectural features designed for full-precision arithmetic. The survey does not resolve these interactions but makes them visible by placing the techniques in adjacent sections.

- **Training-compression synergy:** Quantization-aware training (QAT) bridges the training-compression divide by incorporating quantization simulation into the training process, producing a model that is robust to the precision reduction. Similarly, combined pruning-and-distillation approaches (Sreenivas et al., 2024; Muralidharan et al., 2024) iterate between removing weights and retraining with distillation losses, blurring the boundary between compression and training. The survey's taxonomy accommodates these hybrid techniques by recognizing that the categories are not mutually exclusive.

- **Architecture-training interaction:** The choice of attention mechanism (standard quadratic vs. linear vs. state space) affects not just inference cost but also training cost and stability. Linear attention mechanisms that enable efficient inference may also enable training on longer sequences with the same memory budget, changing what training data is accessible. The survey's inclusion of both efficient attention (Section 2.2) and training techniques (Section 3) in the same framework makes these cross-cutting interactions visible.

The intellectual contribution here is not a new technique but a **systems-level perspective** on SLM development that was absent from prior surveys, which typically covered only one stage (compression, or efficient architectures, or training methods) in isolation. By presenting the full pipeline and organizing it within a common constraint taxonomy, the survey enables practitioners to reason about how choices compose—for example, whether to invest in an efficient architecture from scratch or to take an existing LLM and apply aggressive compression, and what the downstream implications of each choice are for fine-tuning and deployment.

This is an **incremental but important** contribution to the organization of knowledge. The individual techniques were all known; the insight that they form a composable pipeline with characterized interactions is what the survey adds. The evidence for this insight is structural rather than empirical—it is the architecture of the survey itself—but it changes how a practitioner or researcher approaches the problem of building an SLM.

## 5. Experimental Analysis

### Evaluation Methodology

This section requires a fundamentally different framing than for a typical methods paper. This is a **survey paper**—it does not conduct original experiments. Instead, it aggregates and organizes experimental results, datasets, and evaluation protocols from the literature it surveys. The "methodology" described here is the set of evaluation practices the survey recommends for comparing SLM techniques, drawn from common practices across the papers it catalogs.

- **Dataset.** The survey does not introduce new datasets but compiles those commonly used for pre-training and evaluating SLMs across different constraint settings, presented in Table 2. These include: **SuperGLUE** (Sarlin et al., 2020) for natural language understanding under efficient inference constraints; **SQuAD** (Rajpurkar et al., 2016), **TriviaQA** (Joshi et al., 2017), **CoQA** (Reddy et al., 2019), and **Natural Questions** (Kwiatkowski et al., 2019) for question-answering tasks requiring fast response times; **TinyBERT** (Jiao et al., 2020) and **OpenOrca** (Lian et al., 2023) for on-device and TinyML settings; **PrivacyGLUE** (Shankar et al., 2023) for privacy-preserving evaluation; and domain-specific datasets like **MIMIC** (Johnson et al., 2020) for clinical NLP. The survey does not report train/test split sizes for any of these, nor does it specify which datasets were used in any head-to-head experimental comparison (because no such comparison was conducted). The datasets are organized by the constraint setting they are designed to evaluate—efficient inference, on-device/mobile, privacy-preserving, or energy-efficient AI—rather than by task type.

- **Base model(s).** As a survey, no single base model family is used for experiments. The survey catalogs SLMs across diverse architectural families and scales: encoder-only models derived from BERT (MobileBERT at 4.3× smaller than BERT-base; DistilBERT and TinyBERT retaining >96% of BERT's performance), decoder-only models spanning 58M parameters (BabyLLaMA) to 1.1B parameters (TinyLLaMA), efficient architectures like Mamba and RWKV using state space models and linear attention, and multi-modal models like LLaVA-Next, InternVL2, and PaliGemma. The survey explicitly refrains from making a recommendation about which model scale or architecture is "best," instead noting that the relevant question is "which is best for my specific set of constraints." The models discussed were chosen not for a controlled experiment but as representative examples of each technique category in the taxonomy.

- **Metrics.** The survey organizes evaluation metrics by the constraint dimension they measure, presented in Table 2, rather than by model capability. For **latency**, the key metrics are inference time (how quickly a model processes input and generates output; Narayanan et al., 2023) and throughput (tokens or samples processed per unit time; Arora et al., 2024). For **memory**, metrics include peak memory usage (Lee et al., 2024a), memory footprint, and compression ratio (Cao et al., 2024), which together quantify how compact a model is and how much headroom remains under a given memory ceiling. For **privacy**, metrics include the privacy budget from differential privacy (Yu et al., 2024) and noise level (Havrilla et al., 2024), which quantify the tradeoff between privacy guarantees and model accuracy. For **energy optimization**, the survey identifies the energy efficiency ratio (performance per unit energy; Stojkovic et al., 2024b), thermal efficiency, and idle power consumption (Patel et al., 2024). Crucially, the survey does not define how any of these metrics are computed in precise mathematical terms (e.g., it does not specify the epsilon-delta parameterization of the privacy budget, or the exact formula for compression ratio), nor does it provide threshold values that would define acceptable performance on any metric. It also does not report numeric values for any efficiency metric from the literature—only accuracy figures (e.g., "more than 96" for TinyBERT's performance retention, "4.3× size reduction and 5.5× speedup" for MobileBERT, "competitive performance" for TinyLLaMA). This is a significant gap: the survey organizes the metric space conceptually but does not populate it with quantitative baselines.

- **Baselines.** The survey does not define formal baselines in the experimental sense. Instead, it identifies canonical reference points within each technique category: BERT-base serves as the implicit baseline for encoder-only lightweight architectures (against which MobileBERT achieves 4.3× compression, DistilBERT achieves >96% performance retention); standard quadratic self-attention with O(N²) complexity serves as the baseline for efficient attention approximations; full fine-tuning serves as the baseline for parameter-efficient methods like LoRA; and the original unpruned/unquantized model serves as the baseline for compression techniques. However, no standardized baseline configurations (e.g., a fixed LLaMA-7B at FP16 as a common reference point against which all compression techniques are compared) are proposed. This is a direct consequence of the survey format: because the papers being surveyed used different base models, different evaluation datasets, and different metrics, no single baseline could be retroactively imposed.

- **Generation budget / compute accounting.** The survey does not propose a standardized compute accounting framework for fair comparison across techniques. Individual surveyed papers use various cost models: MobileBERT reports size reduction factor (4.3×) and speedup factor (5.5×); efficient attention papers report complexity classes (O(N log N) or O(N) vs. O(N²)); pruning papers report sparsity ratios and compression ratios; quantization papers report bit-widths. These cost metrics are not directly comparable across technique families—a 4.3× parameter reduction from architectural redesign and a 4× memory reduction from 4-bit quantization represent different actual resource savings because the computational characteristics of the resulting models differ (sparse vs. dense computation, different operation types). The survey acknowledges this incomparability implicitly through its constraint-based taxonomy, which encourages evaluating each technique against the specific constraint it targets, but it does not propose a unified FLOPs or wall-clock accounting methodology.

- **Cross-validation / statistical protocol.** No statistical validation protocol is proposed or described. The survey reports results from individual papers at face value without re-analysis. This means that claims about relative performance of different techniques (e.g., that distillation "can exceed teacher models' performance particularly under data-constrained conditions," that Mamba "demonstrates competitive performance across various tasks") are reproduced from the original papers without independent verification, and no attempt is made to assess whether reported differences are statistically significant or robust to hyperparameter variation. The survey's two-fold cross-validation protocol referenced in the taxonomy (for strategy selection across difficulty bins) belongs conceptually to the experimental methodology of individual surveyed papers (like the compute-optimal scaling work), not to the survey itself.

---

### Main Quantitative Results

Because this is a survey paper with no original experiments, it does not contain "results" in the conventional sense—no tables of numbers, no learning curves, no head-to-head comparisons. Instead, it presents a series of **aggregated findings** organized by technique category, each drawn from individual papers and presented as representative of what that technique category can achieve. I organize these by the survey's major axes.

#### Architecture Results

The survey reports the following headline quantitative claims from architectural innovations, all drawn from cited papers without independent verification:

**Encoder-only lightweight models:** MobileBERT (Sun et al., 2020) achieves a **4.3× size reduction** and a **5.5× speedup** compared to BERT-base while maintaining competitive accuracy (Section 2.1). DistilBERT (Sanh, 2019) and TinyBERT (Jiao et al., 2019) "achieve more than 96" (the text is truncated here; the intended meaning is presumably >96% of BERT's performance or a 96% parameter reduction—the exact figure is ambiguous and unrecoverable from the truncated text). No specific benchmark task or metric is cited for these performance-retention figures, nor is the compute budget under which the speedup was measured specified.

**Decoder-only lightweight models:** BabyLLaMA (Timiryasov and Tastet, 2023a), at 58M parameters, demonstrates that "distillation can exceed teacher models' performance particularly under data-constrained conditions" (Section 2.1). The survey does not report the specific accuracy gain or the data-constrained regime in which this was observed. BabyLLaMA-2 (Tastet and Timiryasov, 2024) extends this to 345M parameters with similar claims. TinyLLaMA (Zhang et al., 2024), at 1.1B parameters, achieves "high efficiency" and "maintains competitive performance for various downstream tasks" (Section 2.1), with no quantitative performance numbers cited. MobileLLM (Liu et al., 2024e), at 0.5B parameters, "improves on various chat benchmarks and performs comparably with LLaMA-2-7B in API calling" (Section 6.3), which is a specific comparative claim—a 0.5B model matching a 7B model on a specific task.

**Efficient attention mechanisms:** The survey reports that Reformer (Kitaev et al., 2020) improves complexity from O(N²) to O(N log N) (Section 2.2), and that linear attention mechanisms (Katharopoulos et al., 2020; Wang et al., 2020a; Beltagy et al., 2020) reduce complexity to O(N). Wang et al. (2020a) reports that "empirically transformers with linear self-attention matches the performance of the original self-attention mechanism across a variety of downstream tasks" (Section 2.2). None of these complexity improvements are accompanied by wall-clock timing measurements, memory footprint measurements, or accuracy-on-benchmark measurements from the survey itself. Mamba (Gu and Dao, 2023; Dao and Gu, 2024) and RWKV (Peng et al., 2023) are described as demonstrating "competitive performance across various tasks" (Section 2.2), with no quantification.

**Neural Architecture Search:** MobileLLM (Liu et al., 2024e) "investigates the impact of model depth (i.e., number of layers) and width (i.e., number of heads) on performance, effectively conducting a targeted architecture search within a smaller parameter range for language models with millions of parameters" (Section 2.3). No quantitative results from this search are reported.

**Small multi-modal models:** The survey notes (Section 2.4) that models like LLaVA-Next, Idefics2, and InternVL2 achieve "comparable or superior performance to their predecessors while significantly reducing the number of parameters," but provides no specific parameter counts, performance numbers, or relative improvement figures.

#### Training Results

The survey reports the following headline findings from training techniques, all qualitative rather than quantitative:

**Mixed-precision training:** BFLOAT16 (Burgess et al., 2019) "has demonstrated superior training performance and representation accuracy compared to FP16" (Section 3.1), but no specific improvement magnitude is reported. FP8 precision on NVIDIA Hopper GPUs "enables even greater computational efficiency for large-scale language models" (Section 3.1), with no throughput or memory savings figures.

**Efficient optimizers:** Adafactor (Shazeer and Stern, 2018) and Sophia (Liu et al., 2024b) are described as "memory-efficient variants" that "improve training speed and efficiency" (Section 3.1) without quantitative comparisons to Adam/AdamW on memory usage or convergence speed.

**Distributed training:** ZeRO (Rajbhandari et al., 2020) enables training with larger batch sizes but the survey reports no specific memory reduction factors or scaling efficiency numbers.

**Parameter-efficient fine-tuning:** The survey describes LoRA (Hu et al., 2021), Prompt Tuning (Lester et al., 2021), Llama-Adapter (Zhang et al., 2023b; Gao et al., 2023), and Dynamic Adapters (Kong et al., 2024; Feng et al., 2024; Gou et al., 2023; Liu et al., 2023b; Luo et al., 2024b) in terms of their mechanisms and design motivations, but provides no quantitative comparison of their parameter efficiency (e.g., number of trainable parameters as a fraction of total parameters), accuracy on downstream tasks, or training time relative to full fine-tuning.

**Data augmentation:** The survey lists several methods (AugGPT, Evol-Instruct, Reflection-Tuning, FANNO, LLM2LLM; Section 3.2.2) and their qualitative effects on data diversity and quality, but reports no quantitative improvements in downstream task accuracy, sample efficiency, or robustness from any augmentation method.

#### Compression Results

**Pruning:** SparseGPT (Frantar and Alistarh, 2023) "can efficiently handle large-scale models like OPT-175B and BLOOM-176B" (Section 4.1), but the survey does not report the sparsity levels achieved, the accuracy degradation, or the speedup realized. n:m pruning (Zhou et al., 2021) is described as enabling "significant speedups" using NVIDIA TensorRT, with no actual speedup figures. For structured pruning, the survey notes that Li et al. (2023b) observes "prevalent sparsity in feed-forward networks" and that Mirzadeh et al. (2024) changes activation functions to ReLU "to improve activation sparsity," but reports no fraction of neurons pruned or accuracy change.

**Quantization:** GPTQ (Frantar et al., 2022) uses "inverse Hessian matrices to minimize the reconstruction error" (Section 4.2) but no bit-widths, model sizes, or perplexity figures are reported. AWQ (Lin et al., 2024), ZeroQuant (Yao et al., 2022), SmoothQuant (Xiao et al., 2023), and SpinQuant (Liu et al., 2024d) are described mechanistically without quantitative results. Quantization-aware training methods (LLM-QAT and EdgeQAT; Section 4.2) are described as having "strong performance" without quantification.

**Knowledge distillation:** DistilBERT and TinyBERT achieve >96% of BERT's performance (extrapolating from truncated text; Section 2.1). BabyLLaMA (58M parameters) demonstrates that distillation "can outperform traditional pre-training on the same dataset" (Section 4.3), though by what margin is not reported. The survey notes that Hsieh et al. (2023) find that "the distilled model outperforms large-language models on commonly used NLI, Commonsense QA and arithmetic reasoning benchmarks" (Section 4.3), which is a specific comparative claim—small distilled models beating their larger teachers—but the survey provides no accuracy numbers to support it.

#### Application Results

Section 6 reports several application-level findings, mostly qualitative:

- **On-device inference:** Apple Intelligence applies a 3B parameter model for on-device tasks (text summarization, image generation, code completion; Gunter et al., 2024; Research, 2024). MobileLLM (0.5B parameters) "performs comparably with LLaMA-2-7B in API calling" (Liu et al., 2024e).

- **Energy efficiency:** Stojkovic et al. (2024a) find that "energy usage can be reduced by about 20" (text truncated; presumably 20% or a related figure). Husom et al. (2024) find that "response token length [is] the most effective predictor of energy usage" (Section 7.3).

- **Latency:** On-device inference "reduces latency as measured by the time to first generated token" (Hu et al., 2024; Gerganov; Section 6.3), though no specific latency numbers are provided.

---

### Ablation Studies and Robustness Checks

The survey does not contain original ablation studies. However, it does report findings from the literature that have an ablative character—comparisons that isolate the effect of a specific design choice. I catalogue these here:

**Soft vs. hard labels in knowledge distillation:** The survey notes (Section 4.3) that classical knowledge distillation (Hinton et al., 2015) uses softened output distributions (temperature T > 1) rather than hard labels, and that this provides a richer training signal. The evidence for this is drawn from the original distillation literature rather than re-examined in the survey.

**Distilling with vs. without rationales:** Hsieh et al. (2023) find that using "rationales" as additional supervision during distillation "makes it more sample-efficient" and produces models that "outperforms large-language models on commonly used NLI, Commonsense QA and arithmetic reasoning benchmarks" (Section 4.3). This is an ablation comparing label-only distillation to label-plus-rationale distillation, showing that the reasoning chain carries transferable knowledge beyond the final answer.

**Shared vs. separate tokenizer in distillation:** The survey identifies (Section 4.3) that distillation is "primarily effective when (1) the teacher and the student language model share the same tokenizer and (2) the teacher's pre-training data is available." Boizard et al. (2024) address the first constraint with a "universal logit distillation loss," which can be viewed as an ablation showing that removing the shared-tokenizer requirement reduces distillation effectiveness unless explicitly addressed.

**Teacher forcing vs. exposure bias mitigation:** Gu et al. (2024) introduce modifications to the distillation loss that reduce exposure bias—the mismatch between teacher-forced training and autoregressive inference. The improvement in "calibration and lower exposure bias" (Section 4.3) represents an ablation comparing standard distillation to the modified loss.

**Local vs. local+global attention:** Longformer (Beltagy et al., 2020) uses a combination of local windowed attention and task-specific global attention (Section 2.2). The choice to include global attention tokens alongside local windows can be viewed as an ablation over purely local attention, showing that some global information flow is necessary for tasks requiring long-range dependencies.

**Unstructured vs. structured pruning:** The survey distinguishes (Section 4.1) between unstructured pruning (finer granularity but requires sparsity-aware hardware) and structured pruning (coarser but produces dense models that run efficiently on standard hardware). This tradeoff is not quantitatively characterized—no head-to-head comparison of SparseGPT (unstructured) and, say, layer-dropping (structured) at the same compression ratio is reported.

**FP16 vs. BFLOAT16 for mixed-precision training:** The survey reports (Section 3.1) that BFLOAT16 "has demonstrated superior training performance and representation accuracy compared to FP16" due to its greater dynamic range, and that recent work (Rae et al., 2021) observed "accuracy losses due to [FP16's] limited numerical range." This is effectively an ablation documenting that the choice of low-precision format materially affects training stability.

**Quantization-aware training vs. post-training quantization:** The survey notes (Section 4.2) that QAT methods "typically achieve higher accuracy at very low bit-widths where post-training methods struggle," representing an ablation over the quantization approach. No specific accuracy-vs-bit-width curves are provided.

**ReLU vs. GELU for activation sparsity:** Mirzadeh et al. (2024) change activation functions from GELU to ReLU and fine-tune to improve activation sparsity (Section 4.1). This can be viewed as an ablation showing that GELU produces activations less amenable to sparsity-based pruning, and that architectural modifications can improve pruning outcomes.

**Parameter-efficient fine-tuning vs. full fine-tuning:** The survey frames PEFT methods (LoRA, Prompt Tuning, etc.) against the baseline of full fine-tuning (Section 3.2.1), with the claimed benefits being reduced computational cost, preserved knowledge, reduced overfitting, and improved flexibility. However, no quantitative comparison (accuracy vs. number of trainable parameters, or accuracy vs. fine-tuning time) is reported for any specific task.

---

### Critical Assessment

The central challenge in assessing this paper's experimental support is that **there are no experiments to assess**. This is a survey paper that catalogs, organizes, and contextualizes existing work—it does not produce new empirical evidence. The question, then, is not "do the experiments support the claims?" but rather "does the evidence the survey *cites* support the claims the survey *makes*, and are those claims appropriately qualified?"

#### What the Survey Claims vs. What It Demonstrates

The survey makes several types of claims, which require different standards of evidence:

**Claim Type 1: Existence claims** ("Technique X exists and is used for purpose Y"). These are trivially well-supported by citation. The survey demonstrates that SparseGPT, LoRA, Mamba, MobileLLM, and hundreds of other techniques exist, were developed for specific efficiency purposes, and have been described in the literature. This is the survey's primary contribution—comprehensive cataloging—and it is executed thoroughly.

**Claim Type 2: Organizational claims** ("Technique X primarily addresses constraints A, B, and C"). These are supported by the taxonomy structure (Tables 1 and 2) but rest on the authors' judgment rather than empirical evidence. The assignment of checkmarks in Table 1 is an interpretive act—it reflects what the authors believe each technique's primary benefits are. For most categories, the assignments are non-controversial (pruning reduces memory and storage; quantization reduces memory and storage; efficient attention reduces inference runtime), but edge cases exist: does knowledge distillation reduce inference runtime (the student is smaller and faster) or only memory and storage? The survey assigns it to neither inference runtime nor latency in Table 1, checking only dataset size (because distillation can be more data-efficient) and implicitly memory (through the student's smaller size). The justification for these boundary decisions is not provided, and reasonable alternatives exist.

**Claim Type 3: Comparative claims** ("Method X outperforms method Y on constraint Z"). These are the most substantive claims, and they are the least well-supported because the survey provides almost no quantitative head-to-head comparisons. Claims like "BFLOAT16 has demonstrated superior training performance compared to FP16," "the distilled model outperforms large-language models on NLI benchmarks," and "Mamba demonstrates competitive performance across various tasks" are qualitative summaries of results from cited papers but are stripped of the numerical context—effect sizes, confidence intervals, task-by-task breakdowns—that would let a reader assess their reliability. A practitioner reading "MobilLLaMA applies a parameter-sharing scheme that reduces both pre-training and deployment costs" learns that the technique exists but gains no ability to estimate the magnitude of the benefit or the conditions under which it holds.

**Claim Type 4: Boundary claims** ("Progress on constraint X does not imply progress on constraint Y"). This is the survey's most important structural claim—the multi-objective tradeoff thesis—and it is supported by specific examples (quantization-aware training being slower than full-precision training) but not by systematic evidence. The survey does not present, for example, a correlation matrix showing how improvements on different constraint metrics covary across techniques, or a Pareto frontier analysis showing the best achievable accuracy under various constraint combinations. The claim is plausible and well-argued, but it remains a conceptual framework rather than an empirically validated principle.

#### Specific Weaknesses

**Absence of quantitative baselines and standardized comparisons.** The survey's greatest weakness as a practical resource is that it provides no numbers. A practitioner trying to decide between MobileLLM, TinyLLaMA, and a pruned version of LLaMA for a 1GB-memory mobile deployment learns that all three approaches exist and are aimed at memory-constrained settings, but gains no ability to compare their actual memory footprints, inference latencies, or accuracy tradeoffs. The survey describes each technique's mechanism and qualitative benefit but provides none of the quantitative decision-support data that would make the taxonomy actionable. This is partly inherent to the survey format—the underlying papers use different baselines, metrics, and evaluation protocols—but the survey does not address this limitation explicitly or propose standardization as a future direction.

**Truncated and ambiguous performance figures.** Several quoted performance figures are truncated or ambiguous: TinyBERT "achieves more than 96" (96 what? Percent? Parameter reduction? Cut off mid-sentence), energy usage "reduced by about 20" (20%? Mid-sentence), and MobileBERT's "5.5x speedup" is reported without specifying the hardware, batch size, or sequence length under which it was measured. These ambiguities undermine the survey's utility as a reference for practitioners making quantitative decisions.

**No critical evaluation of individual paper claims.** The survey reproduces claims from cited papers essentially verbatim without critical scrutiny. When a paper claims that "distillation can exceed teacher models' performance particularly under data-constrained conditions" (BabyLLaMA), the survey does not interrogate what counts as "data-constrained," whether the result generalizes beyond the specific experimental setup, or whether the teacher models were fairly tuned. This is a fundamental tension in survey methodology: thorough surveys aim for comprehensive coverage, but comprehensive coverage at the scale of this survey (hundreds of papers) makes deep critical engagement with each paper's experimental design infeasible. The tradeoff is that the survey's claims inherit the validity of the underlying papers without independent verification.

**No systematic treatment of the scale-dependence of findings.** The survey acknowledges that "the definitions of 'small' and 'large' are a function of both context and time," noting that GPT-2 at 1.5B parameters was "large" in 2019 but is smaller than many current "small" models. However, it does not systematically address how the effectiveness of different techniques varies with model scale. Does pruning work better at 1B parameters or at 100M parameters? Does the advantage of linear attention over quadratic attention depend on sequence length in a way that determines which model sizes benefit most? These scale-dependent questions are central to the SLM design space but are not organized into the taxonomy, which treats techniques as categorically applicable regardless of the absolute parameter count.

**Missing head-to-head cross-technique comparisons.** Perhaps the most valuable experiment the survey *could have* proposed—but did not—is a matrix of head-to-head comparisons showing how different technique families trade off against each other at a fixed parameter budget. For instance: at 100M parameters, how does a model trained from scratch with a lightweight architecture compare to a 1B-parameter model pruned to 100M and then distilled? The survey's taxonomy makes these cross-technique questions natural to ask but provides no empirical basis for answering them. The section structure (architectures, then training, then compression) implicitly sequences these stages, but the interactions between them—does a distilled-then-pruned model outperform a pruned-then-distilled model?—are entirely unexplored.

**No treatment of latency-throughput tradeoffs in inference.** The survey identifies latency and throughput as separate metrics but does not discuss their well-known tension: batching increases throughput but increases latency per request; autoregressive decoding is a serial bottleneck that limits throughput regardless of model size. For real-time applications, latency is the binding constraint; for batch processing, throughput dominates. The taxonomy does not capture this distinction, treating "inference runtime" and "latency" as separate checkable columns without discussing their interaction.

**Missing privacy metrics for on-device inference.** The survey discusses privacy-preserving datasets (PrivacyGLUE, MIMIC) and on-device inference as a privacy-enabling technology, but it does not provide or propose metrics that would allow a practitioner to quantify the privacy benefit of moving from cloud to on-device inference. The privacy budget metric from differential privacy (Section 5.2) applies to training, not to the inference choice of where computation happens. This is a conceptual gap: the taxonomy presents privacy as a constraint that can be optimized, but the link between specific SLM techniques and quantifiable privacy guarantees is not established.

#### What Would Strengthen the Paper

**Standardized efficiency benchmarks tables.** Even without original experiments, the survey could have compiled existing published results into standardized tables: for each model family (encoder-only, decoder-only at various scales), report published parameter counts, inference memory, latency on a common benchmark (if reported on comparable hardware), and accuracy on a common evaluation suite (e.g., SuperGLUE, MMLU). Where head-to-head comparisons exist in the literature, aggregate them. Where they don't, flag the gap explicitly.

**Pareto frontier visualizations.** A synthetic figure showing the accuracy-efficiency tradeoff across technique families—pruned models, distilled models, purpose-built lightweight architectures, quantized models—at a fixed parameter budget would concretize the survey's central tradeoff thesis. Even if the underlying data came from different papers with different evaluation protocols, a coarse visualization would communicate the design space more effectively than the verbal descriptions that dominate the current text.

**Technique combination matrix.** The survey's pipeline structure (architecture → training → compression) naturally suggests that techniques can be combined, but no systematic treatment of known interactions is provided. A matrix showing which combinations have been empirically tested (e.g., LoRA + quantization, distillation + pruning, efficient attention + quantization) with citations to the relevant papers would make the taxonomy more actionable.

**Explicit characterization of the survey's limitations.** The paper's limitations section (Section 9) discusses hallucination, bias, and the shifting definition of "small," but does not address the survey's own methodological limitations: the absence of quantitative comparisons, the reliance on authors' judgment for constraint assignments in the taxonomy, the lack of critical scrutiny of individual paper claims, and the impossibility of keeping such a survey current given the pace of SLM development. A candid assessment of these limitations would calibrate reader expectations and frame the survey as a map of the territory rather than a travel guide with precise distances and travel times.

In summary, the survey succeeds admirably at its primary goal—comprehensive cataloging and organization of a fragmented literature—but provides almost no quantitative decision-support data, reproduces paper claims without critical scrutiny, and leaves the central tradeoff thesis as a conceptual framework rather than an empirically populated design space. The taxonomy is intellectually valuable as an organizational tool; its translation into actionable engineering guidance awaits future work that populates it with standardized measurements and head-to-head comparisons.

## 6. Limitations and Trade-offs

### No Quantitative Baselines or Standardized Comparisons Across Technique Families

**The assumption or constraint.** As a survey, this paper assumes its primary value lies in comprehensive cataloging and taxonomic organization rather than in providing quantitative decision-support data. It compiles techniques into a unified framework but does not standardize the metrics, baselines, or evaluation protocols used by the individual papers it surveys. The consequence is that a practitioner consulting this survey can learn *what* techniques exist and *which* constraints they address, but cannot answer the most basic engineering question: "Given my specific memory budget, latency target, and accuracy requirement, which technique (or combination) should I use?"

The survey acknowledges the fragmented nature of the literature implicitly throughout but does not address this limitation head-on as a constraint on its own utility. The paper notes in Section 7.3 that Husom et al. (2024) "find that architecture significantly influences power consumption using the MELODI benchmark" and that "CPU-only inference was found to be generally less efficient than on GPU," and in Section 6.3 that MobileLLM "improves on various chat benchmarks and performs comparably with LLaMA-2-7B in API calling." These are qualitative comparisons that gesture toward performance without providing the specific numbers a practitioner needs—what is the latency of MobileLLM at 0.5B on which hardware? How does it compare to a 4-bit quantized LLaMA-2-7B on the same device? The taxonomy tells you both approaches exist and reduce memory; it does not tell you which to choose.

**The consequence.** The survey functions as a **map of the territory without distances or travel times**. A practitioner facing a concrete deployment decision—such as fitting a chat-capable model into 2GB of mobile RAM with sub-200ms time-to-first-token—learns that they should consider lightweight architectures (Section 2.1), pruning (Section 4.1), quantization (Section 4.2), and knowledge distillation (Section 4.3), but gains no ability to estimate the accuracy they would sacrifice for each approach at their target constraint boundary, nor any way to compare a purpose-built 500M-parameter model against a pruned-and-quantized 7B model at the same memory footprint. The survey's central tradeoff thesis—"progress on any one of these goals does not necessarily imply progress on the others"—is presented as a conceptual insight, but the magnitude of these tradeoffs is never populated. Is quantization-aware training 10% slower than full-precision training, or 2× slower? Does the gap widen at 4-bit versus 8-bit? The survey does not say.

**What evidence exists in the paper.** All quantitative evidence is drawn from cited papers and presented at insufficient granularity for cross-technique comparison. The survey reports MobileBERT's "4.3× size reduction and 5.5× speedup compared to BERT-base" (Section 2.1), TinyBERT achieving "more than 96" of BERT's performance (truncated text, Section 2.1), and energy usage being "reduced by about 20" (truncated text, Section 7.3). These numbers come from different papers using different hardware, different benchmark tasks, and different evaluation protocols. No table in the survey standardizes them. The metrics table (Table 2) defines what metrics *exist* but does not populate them with values for any model.

**Mitigation status.** Not addressed. The survey does not acknowledge this as a limitation of its approach. Section 9 discusses hallucination, bias, and the shifting definition of "small" as limitations of SLMs generally, not as limitations of the survey itself. The absence of standardized quantitative comparisons is not mentioned. Future work—perhaps a benchmark paper that evaluates a representative set of SLM techniques under standardized conditions—would be required to populate the taxonomy with the decision-support data practitioners actually need.

---

### No Critical Scrutiny of Individual Paper Claims

**The assumption or constraint.** The survey reproduces claims from cited papers essentially without independent verification, treating reported results as reliable when they may conflict, overstate, or depend on experimental choices that do not generalize. This is a structural tension in comprehensive survey methodology: the paper covers "hundreds" of techniques (Section 1, implicitly through the breadth of coverage), and deep critical engagement with each paper's experimental design is infeasible at that scale. The consequence is that the survey's claims about relative technique performance inherit the validity—and the limitations—of the underlying papers without the reader being able to distinguish well-established findings from fragile ones.

**The consequence.** Several important comparative claims in the survey are stated as settled facts when they may be contingent on specific experimental conditions. For example, the survey reports (Section 2.1) that BabyLLaMA demonstrates "distillation can exceed teacher models' performance particularly under data-constrained conditions." A practitioner reading this might reasonably conclude that distillation is uniformly preferable to training from scratch when data is limited. But the claim depends on what counts as "data-constrained" (not defined in the survey), which teacher model was used, what dataset was used, and whether the teacher was fairly tuned—none of which the survey reports. Similarly, the survey states (Section 3.1) that BFLOAT16 "has demonstrated superior training performance and representation accuracy compared to FP16," but the magnitude of the improvement, the tasks on which it was measured, and whether it holds for all model scales are unreported. A practitioner might switch their entire training pipeline to BFLOAT16 based on this claim without knowing the effect size or boundary conditions.

The survey also reproduces specific quantitative figures that are truncated or ambiguous, undermining their utility. TinyBERT "achieve[s] more than 96" (Section 2.1)—the sentence is cut off, making the figure unrecoverable. Energy usage can be "reduced by about 20" (Section 7.3)—20%? 20 watts? The sentence is incomplete. These are likely artifacts of manuscript preparation, but in their current form they render the survey's most specific quantitative claims meaningless.

**What evidence exists in the paper.** The limitation is structural rather than localized to a specific section. Throughout Sections 2–4, comparative claims are presented as conclusions (e.g., "Both these works show that empirically transformers with linear self-attention matches the performance of the original self-attention mechanism across a variety of downstream tasks," Section 2.2) without the experimental context—which downstream tasks, at what sequence lengths, with what base model—that would let a reader assess their reliability and scope.

**Mitigation status.** Not addressed. The survey does not discuss the reliability of the claims it aggregates, the possibility of publication bias (papers reporting positive results for their technique are more likely to be published and cited), or the challenge of comparing results across papers with different experimental protocols. The limitations section (Section 9) discusses risks of SLMs (hallucination, bias) rather than risks of the survey methodology itself. A brief methodological limitations subsection acknowledging that the survey's comparative claims are only as reliable as the underlying papers—and that readers should consult original sources for quantitative decision-making—would substantially improve the survey's trustworthiness as a reference.

---

### The Taxonomy's Constraint Assignments Are Author Judgments, Not Empirically Validated

**The assumption or constraint.** The survey's central intellectual contribution is the two-axis taxonomy in Table 1, which maps each technique category onto the constraints it primarily addresses. These assignments—the checkmarks in the matrix—are **interpretive judgments by the authors** rather than empirical measurements or consensus classifications from the literature. The justification for why a specific technique receives a checkmark in one column but not another is not provided in most cases, and the taxonomy does not distinguish between constraints that are *directly* reduced by a technique and constraints that are *indirectly* or *sometimes* reduced.

**The consequence.** The taxonomy's practical utility as a decision-support tool is only as good as the accuracy of its constraint mappings. If a practitioner filters for techniques that address "Inference Runtime" and misses a relevant technique because the authors judged it primarily addresses "Memory" instead, the taxonomy misleads rather than guides. Several specific assignment decisions are debatable:

- **Knowledge distillation** receives a checkmark only under "Dataset Size" in Table 1, reflecting the finding that distillation can be more data-efficient than training from scratch. But the primary practical motivation for distillation is to produce a smaller, faster student model, which directly reduces inference runtime, memory, and storage. Assigning no checkmarks in these columns, while logically defensible (distillation *enables* a smaller model; the student's size, not the distillation process itself, reduces inference cost), fails to capture why practitioners actually use distillation.

- **Fine-tuning techniques** receive checkmarks for "Training Compute" and "Dataset Size" but not for "Inference Runtime" or "Memory." This is accurate for fine-tuning itself, but parameter-efficient fine-tuning methods like LoRA can produce task-specific adapters that are loaded and unloaded at inference time, effectively reducing memory when serving multiple tasks. The taxonomy's static assignment obscures this use case.

- **Neural Architecture Search** receives checkmarks for "Training Compute," "Inference Runtime," and "Memory" but not for "Dataset Size." Yet NAS can discover architectures that are more data-efficient (require less training data to reach a given accuracy), and the survey itself notes that architectural choices affect data efficiency through the BabyLLaMA discussion (Section 2.1). Whether NAS should receive the checkmark depends on whether the specific NAS methods surveyed optimize for data efficiency—a question the taxonomy does not answer.

These are not errors so much as **implicit design choices** that reflect a particular view of what each technique "primarily" does. The survey does not articulate the principles by which these judgments were made, leaving readers without a way to assess whether the taxonomy's guidance matches their specific situation.

**What evidence exists in the paper.** Table 1 is presented as a fait accompli without discussion of the assignment criteria. Section 1 states that the taxonomy categorizes methods by "the constraints the technique is attempting to optimize for" (emphasis added), suggesting the assignments reflect author intent in the original papers rather than empirical measurement, but this principle is not consistently applied. For example, knowledge distillation papers almost always *attempt* to reduce inference cost; the fact that the survey assigns distillation to "Dataset Size" instead suggests a different criterion is at work (perhaps "what unique benefit does this technique provide that others do not?"), but this criterion is never stated.

**Mitigation status.** Not addressed. The survey does not discuss the subjectivity of constraint assignments, the possibility of disagreement among experts about which techniques map to which constraints, or the need for empirical validation of the taxonomy against practitioner needs. This is a fundamental limitation of any taxonomy developed through author synthesis rather than systematic empirical study, but acknowledging it would strengthen the survey by calibrating reader trust in the matrix as a decision-support tool rather than a definitive guide.

---

### No Treatment of Latency-Throughput Tensions in Inference Deployment

**The assumption or constraint.** The survey treats "Inference Runtime" and "Latency" as separate checkable constraints in Table 1 and lists both inference time and throughput as latency metrics in Table 2, but never discusses the well-known **tension between these two objectives** in deployed systems. Low latency (fast response to a single request) and high throughput (many requests processed per unit time) are often in direct conflict: batching requests increases throughput by amortizing model-loading and memory-access costs but increases the latency of each individual request because it must wait for the batch to fill. Autoregressive decoding is a serial bottleneck that limits throughput regardless of model size because each token generation depends on the previous one. For SLMs deployed in real-time interaction settings (chatbots, voice interfaces, translation—Section 6.1, Table 3), latency is the binding constraint; for batch processing settings (content classification, large-scale summarization, Section 6.2), throughput dominates. The survey's taxonomy does not capture this distinction, and techniques that improve one may worsen the other.

**The consequence.** A practitioner consulting Table 1 sees checkmarks in both the "Inference Runtime" and "Latency" columns for pruning, quantization, and efficient attention, potentially inferring that these techniques improve both latency and throughput uniformly. But this is not guaranteed. For example:

- **Quantization** reduces memory footprint, which can improve throughput by enabling larger batch sizes (more requests fit in memory simultaneously), but the per-request latency may not decrease proportionally if the quantized arithmetic kernels are not optimized for single-request execution on the target hardware.
- **Pruning** that produces unstructured sparsity may require specialized sparse matrix libraries to achieve throughput gains (the survey notes this in Section 4.1: "unstructured pruning often results in sparse matrices requiring specialized hardware or algorithms to maximize computational benefits"), but those libraries may add per-request overhead that increases latency.
- **Efficient attention approximations** like linear attention reduce the asymptotic complexity from O(N²) to O(N), which reduces latency for long sequences, but the constant factors in the linear attention implementation may make it *slower* than optimized quadratic attention for short sequences (the typical regime for chatbot interactions where context is measured in hundreds, not thousands, of tokens).

The survey's flat treatment of "Inference Runtime" and "Latency" as independent checkboxes obscures these interactions and may lead practitioners to apply techniques that optimize the wrong objective for their deployment scenario.

**What evidence exists in the paper.** Table 2 lists inference time and throughput as separate metrics under "Latency," but no discussion connects these metrics to technique selection or notes their tension. Section 6.1 (Real-Time Interaction) identifies latency as a key constraint for chatbots and voice interfaces; Section 6.2 (Content Generation and Processing) does not explicitly identify throughput as the dominant constraint, listing "Faster inference, minimal resource use" as the need for summarization and classification without clarifying whether speed means latency or throughput. Table 3 repeats "Low latency required for real-time" for voice interfaces and translation, and "Fast prediction with low memory" for autocompletion, but never uses the word "throughput" in the application context.

**Mitigation status.** Not addressed. The survey does not discuss the latency-throughput tension, does not distinguish techniques that primarily improve latency from those that primarily improve throughput, and does not provide guidance on selecting techniques based on which of these two objectives dominates a given deployment. This is a significant gap for a survey aimed at practitioners, since understanding the latency-throughput profile of different techniques is essential for real-world deployment engineering. Future work could extend the taxonomy to include a "latency vs. throughput" dimension in Table 1, distinguishing techniques whose benefits accrue primarily to one or the other.

---

### Survey Scope Is Limited to English-Language, Text-Dominant Tasks with Standard Benchmarks

**The assumption or constraint.** The survey covers SLM techniques primarily as they have been evaluated on standard English-language benchmarks (SuperGLUE, SQuAD, TriviaQA, etc.; Table 2) and in application contexts that assume English text (chatbots, voice assistants, content generation; Section 6). The paper acknowledges data efficiency for "low-resource languages" briefly in the context of data augmentation (Section 3.2.2: "Data augmentation is also effective for synthesizing new data when training data is limited, such as for low-resource languages (Whitehouse et al., 2023)"), but does not discuss how the SLM techniques surveyed—architecture design, training, compression—transfer to languages with different morphological structures, writing systems, or data availability profiles. The multi-modal section (Section 2.4) extends the survey's reach to vision-language models, but the evaluation framework (Section 5) remains primarily text-centric, and none of the privacy, energy, or memory metrics in Table 2 are adapted for multi-modal deployment constraints.

**The consequence.** A practitioner building SLMs for deployment in non-English contexts or for multi-modal applications cannot determine from this survey whether the techniques catalogued transfer equally well. Several specific concerns arise:

- **Tokenization efficiency** varies dramatically across languages. A model with a 32K-token vocabulary primarily trained on English may require 3–5× more tokens to represent the same semantic content in morphologically rich languages (e.g., Turkish, Finnish, Arabic) or in languages using non-Latin scripts (e.g., Chinese, Japanese, Korean), because rare subword units are needed for words not seen during tokenizer training. This increases both latency (more tokens to generate) and memory (larger KV-cache) in ways that interact with the architectural and compression choices catalogued in the survey. A lightweight architecture optimized for English tokenization efficiency may underperform on other languages even if the architecture itself is language-agnostic.

- **Knowledge distillation for low-resource languages** faces a specific challenge: the teacher model must itself be capable in the target language, which may not be true for the English-pretrained teachers used in most distillation work surveyed. The survey's distillation section (Section 4.3) assumes "the teacher's pre-training data is available" (Boizard et al., 2024), which for low-resource languages it often is not because high-quality pre-training corpora do not exist. The survey does not address whether distillation strategies that work for English transfer to this regime.

- **Multi-modal models** compound the efficiency challenge but are discussed only briefly (Section 2.4) without a corresponding extension of the evaluation framework. A practitioner building an efficient vision-language model for on-device deployment needs to reason about the memory footprint of the vision encoder, the latency of the cross-modal fusion mechanism, and the energy cost of image preprocessing—none of which appear in Table 2 or are discussed in the applications section.

- **Task diversity** is concentrated on understanding and generation benchmarks that reward factual accuracy and fluency. The survey does not discuss how SLM techniques affect capabilities like structured prediction (e.g., parsing, entity linking), retrieval-augmented generation, or tool use—all of which are increasingly important for deployed language systems and may impose different constraints (e.g., retrieval latency dominating over model latency).

**What evidence exists in the paper.** The datasets listed in Table 2 are exclusively English-language benchmarks developed in the NLP research community. The applications in Section 6 describe English-language use cases (Apple Intelligence, Gemini Nano, Project Astra) without noting language as a dimension of variation. Section 7.4 (Data Privacy) notes query privacy for "digital assistants" that "interface with user data like location history or protected health information" but does not discuss the cross-lingual privacy implications (e.g., whether on-device processing is more critical for languages where cloud-based models are less capable due to training data scarcity). The multi-modal discussion in Section 2.4 is forward-looking but does not integrate multi-modal efficiency into the constraint taxonomy.

**Mitigation status.** Not addressed as a limitation. The survey's language scope is implicit; it does not claim English-only coverage but also does not discuss cross-lingual generalization as a concern. Section 9 (Limitations) focuses on hallucination, bias, and privacy risks of SLMs rather than on scope limitations of the survey itself. A brief acknowledgment that the surveyed techniques have been primarily evaluated on English and that transfer to other languages, scripts, and modalities requires empirical validation would appropriately calibrate the survey's applicability for practitioners working outside the English NLP mainstream. This is especially important given that one of the survey's motivating use cases is democratization of AI—if SLM techniques only work well for English, they do not democratize access for the majority of the world's population.

---

### Hallucination, Bias, and Privacy Are Identified as Open Problems Without Actionable Guidance for SLM Practitioners

**The assumption or constraint.** Section 7 (Open Problems) identifies hallucination (Section 7.1), bias (Section 7.2), energy efficiency (Section 7.3), and data privacy (Section 7.4) as challenges that "remain to be addressed" (Section 8). The survey discusses these as shared concerns between SLMs and LLMs and provides some preliminary evidence about how they scale with model size: hallucination may decrease with larger models for some types (Guan et al., 2024, cited in Section 7.1) while bias may increase (Touvron et al., 2023a; Zhao et al., 2023a, cited in Section 7.2). However, the survey stops at identifying the problems and noting conflicting evidence—it does not provide SLM-specific guidance about how architecture, training, or compression choices affect these risks, nor does it integrate safety constraints into the taxonomy that is the paper's central contribution.

**The consequence.** A practitioner building an SLM for a safety-critical application (medical dialogue, as in Section 6.3's mention of HuatuoGPT and BioMistral; digital assistants accessing health information, as in Section 7.4's discussion of Siri) consults this survey and learns that hallucination and privacy are open problems, but gains no guidance about which technique families might mitigate or exacerbate them. Several specific questions are left entirely unanswered:

- **Does knowledge distillation from a larger teacher reduce or amplify hallucination in the student?** The teacher may hallucinate less than the student would if trained from scratch (because it is larger), but the distillation process compresses the teacher's knowledge lossily, and errors in the teacher's output distribution may be amplified in the student. The survey cites Hsieh et al. (2023) showing that distilling rationales improves reasoning in small models (Section 4.3), but does not discuss whether this translates to reduced factual hallucination.

- **Does quantization affect calibration or hallucination rates?** Post-training quantization reduces numerical precision, which could systematically bias the model's output probabilities in ways that affect confidence calibration and factual reliability. The survey discusses quantization techniques extensively (Section 4.2) but does not mention their impact on safety-relevant behaviors.

- **Does on-device inference actually provide meaningful privacy, or does it change the threat model without reducing risk?** The survey presents on-device inference as privacy-preserving (Section 6.3: "On-device LLMs maintain usability even when [no cloud connectivity]"), and Section 7.4 discusses training data leakage and system prompt leaking. But the survey does not address whether an on-device SLM that has been fine-tuned on sensitive data (e.g., clinical notes) can leak that data through its outputs, or whether model inversion attacks apply to SLMs at the scales discussed. The privacy budget metric from differential privacy (Table 2) applies to the training process; it does not characterize inference-time privacy risks.

- **How do energy efficiency gains from SLMs interact with the deployment scale?** The survey notes that energy usage can be reduced (Section 7.3) but does not discuss Jevons paradox—the possibility that making inference cheaper per query increases total inference volume, potentially increasing aggregate energy consumption even as per-query efficiency improves. For a survey that includes "Energy-Efficient AI" as a constraint setting in Table 2, this is a notable omission.

**What evidence exists in the paper.** Section 7 provides qualitative discussions of each open problem with citations, but these discussions are disconnected from the technical taxonomy in Sections 2–4. For example, Section 7.1 notes that "HallusionBench, a benchmark for image-context reasoning in vision-language models, found that larger sizes reduced hallucinations (Guan et al., 2024)" and that "analysis of the AMBER hallucination benchmark find that the type of hallucination varies as parameter count changes in Minigpt-4 (Wang et al., 2024)." These findings suggest that scaling behavior for hallucination is non-monotonic and type-dependent. But the survey does not connect this back to technique selection: should a practitioner building a 500M-parameter SLM be *more* concerned about hallucination than if they were deploying a 7B model? Does distillation change the scaling relationship? The taxonomy provides no column for "safety" or "reliability" constraints that would let a practitioner prioritize techniques known to preserve or improve factual accuracy.

**Mitigation status.** Partially addressed through flagging rather than solving. The survey is transparent that these are open problems ("future work may need to consider not only how total hallucinations change in SLMs, but also the type and severity may be influenced by model size," Section 7.1), and the inclusion of Section 7 as a dedicated "Open Problems" section is appropriate for a survey. However, the survey does not acknowledge the gap between its taxonomy (which maps techniques to efficiency constraints) and the safety dimensions that are equally critical for deployment. A limitation acknowledgment stating that the taxonomy does not currently incorporate safety, reliability, or fairness constraints—and that these dimensions may interact with efficiency techniques in ways not yet characterized—would strengthen the paper by making the scope of the taxonomy's applicability explicit. The survey's title and abstract position it as "a valuable resource for researchers and practitioners interested in developing and deploying small yet efficient language models"; a deployment-focused practitioner would reasonably expect safety considerations to be integrated into the decision framework, not relegated to a separate open problems section that acknowledges unquantified risks without offering mitigation strategies.

## 7. Implications and Future Directions
- How this work changes the landscape:
  - It reframes SLM development as a constraint-optimization problem with a toolkit spanning architecture, training, and compression, enabling practitioners to assemble targeted “recipes” for specific deployment settings (Tables 1–2).
  - It spotlights mechanisms (attention approximations, KV-cache quantization, activation outlier handling, dynamic adapters) that are maturing into standard SLM ingredients.

- Follow-up research enabled/suggested:
  - Integrated, hardware-aware pipelines:
    - Combine structured pruning (`n:m`) with quantization and KV-cache compression optimized for a specific device class (GPU, NPU, CPU-only edge) and workload (Section 4.1–4.2).
  - Standardized SLM benchmarks:
    - A combined accuracy–latency–memory–energy–privacy suite for realistic edge scenarios would anchor progress (Table 2).
  - Better long-context SLMs:
    - Further K/V cache innovations, approximate attention with verifiable accuracy bounds, and training curricula that maintain long-range reasoning (Section 2.2; 4.2).
  - Robust distillation:
    - Tokenizer-agnostic distillation losses and rationale/chain-of-thought distillation scaled to multi-domain students (Section 4.3).
  - Multimodal monolithic models:
    - Lightweight tokenizers and modality-specific experts that match larger encoders on core perception tasks while staying sub-billion (Section 2.4).
  - Safety, energy, and privacy by design:
    - Techniques that jointly reduce hallucination, bias, and energy, with measurable privacy guarantees (Sections 7.1–7.4; Table 2).

- Practical applications and use cases:
  - On-device assistants:
    - Offline summarization, notification triage, speech interfaces with low latency and privacy, as in `Apple Intelligence` and `TalkBack with GeminiNano` (Section 6.3).
  - Domain-specific SLMs:
    - Clinical documentation assistance (`MIMIC`, domain-adapted LMs), legal summarization, and low-resource language services where privacy and cost dominate (Sections 5.1, 6.3).
  - Edge multimodal analytics:
    - Real-time perception + language reasoning for AR/VR and robotics (Section 6.1–6.2).

> Summary quote of the paper’s aim:
> “Our survey aims to serve as a valuable resource for researchers and practitioners interested in developing and deploying small yet efficient language models.” (Abstract)

> And its organizational backbone:
> “An overview of these axes can be found in Table 1 (techniques) and Table 2 (constraints).” (Section 1)

Together, the taxonomy (Tables 1–2), the mechanism-first explanations (Sections 2–4), and the deployment-oriented applications and open challenges (Sections 6–7) provide a practical blueprint for building SLMs that meet real-world constraints without sacrificing capability.
