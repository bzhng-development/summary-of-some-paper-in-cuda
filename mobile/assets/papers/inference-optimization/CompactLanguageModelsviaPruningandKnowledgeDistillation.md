# Compact Language Models via Pruning and Knowledge Distillation

**ArXiv:** [2407.14679](https://arxiv.org/abs/2407.14679)

## 🎯 Pitch

This paper introduces a practical framework for deriving an entire family of compact, high-performance language models from a single large pretrained LLM by unifying structured pruning across multiple axes with highly data-efficient knowledge distillation. By applying this to the 15B Nemotron-4 model, the authors produce smaller MINITRON models (8B and 4B) using up to 40× fewer training tokens than training from scratch, with comparable or superior performance to similar community models. This work significantly lowers the cost and barrier to producing versatile LLM families, accelerating real-world deployment and democratizing access to powerful language models.

---

## 1. Executive Summary

This paper investigates whether structured pruning combined with knowledge-distillation-based retraining can replace training LLM variants from scratch, developing a set of **compression best practices** through empirical exploration across multiple axes—depth pruning (removing entire layers), width pruning (removing attention heads, MLP neurons, and embedding channels), and their combinations—on the Nemotron-4 15B model. The resulting MINITRON models require up to **40× fewer training tokens** per model compared to training from scratch (deriving 8B and 4B models using only 94B retraining tokens versus trillions), yielding **1.8× total compute savings** for training the full model family while achieving up to a **16% improvement in MMLU scores** over similarly-sized models trained from scratch, establishing that pruning plus lightweight distillation is a viable substitute for repeated full retraining only when the smaller models fall within the capability range of the largest sibling—hard problems that the 15B base model cannot solve remain unsolved by its compressed descendants regardless of retraining budget.

## 2. Context and Motivation

### The Core Problem: The Prohibitive Cost of Training LLM Families

The fundamental problem this paper addresses is economic and practical: **currently, every model in a family of LLMs—from the smallest to the largest—is trained entirely from scratch**. When an organization wants to offer models at 15B, 8B, and 4B parameter scales to serve different deployment scenarios (edge devices, consumer GPUs, datacenters), each variant undergoes its own complete pretraining run consuming trillions of tokens. The paper cites the LLaMA-2 family (7B, 13B, 70B parameters) and the Pythia suite (eight sizes from 80M to 12B parameters) as representative examples. Each of these models independently processes the entire pretraining corpus, burning enormous computational resources that scale roughly linearly with model size—and modern pretraining datasets are expanding to 8, 15, or even more trillion tokens.

This is not merely an incremental concern about wasted electricity. The paper's framing in Section 1 elevates it to a question of **strategic resource allocation for LLM development**:

> "training multiple multi-billion parameter models from scratch is extremely time, data and resource-intensive. In this paper, we ask the following question: can we train one big model, and obtain smaller, more accurate (w.r.t. training from scratch) models from it through a combination of weight pruning and retraining, while only using a small fraction of the original training data?"

The answer has direct financial implications: if producing a 4B and 8B model can reuse the knowledge already acquired by training a 15B model, the cost of offering a model family drops dramatically. The paper quantifies this concretely: training the Nemotron-4 family (15B, 8B, 4B) from scratch costs $(4.4\text{e}17 + 2.5\text{e}17 + 1.2\text{e}17) \times \text{steps}$ FLOPs, while the pruning-based approach reduces this to $(4.4\text{e}17 + 2.5\text{e}17/40 + 1.2\text{e}17/40) \times \text{steps}$ — a **1.8× total compute savings** (Section 4.1). These are real dollar savings at datacenter scale, and they compound as families grow larger (more variants) and datasets grow bigger (more tokens per variant).

### Why This Problem Is Difficult: Compression vs. Capability Retention

The challenge is not simply "delete some weights and save compute." Structured pruning of LLMs involves navigating a complex set of tradeoffs:

**1. Multiple interdependent axes of compression.** An LLM can be compressed along depth (removing entire transformer layers), width (removing attention heads, MLP neurons, and embedding channels), or both. Each axis interacts with the others in non-obvious ways—removing a layer changes which attention heads and neurons are most important in surviving layers because the computational pathway through the model shifts. Section 4.2 documents this explicitly: when pruning from 15B to 8B, a pure width-pruned model achieves an LM validation loss of 2.049 after retraining, while a combined depth+width pruned model reaches only 2.062 despite having slightly more parameters (7.91B vs. 7.74B). The cheaper, simpler approach performs better, which is counterintuitive.

**2. Importance estimation at scale.** Figuring out WHICH weights, heads, neurons, or layers to remove requires computing some notion of "importance" or "sensitivity." The standard approach in the pruning literature—computing gradients or Hessian information, as used by LLM-Pruner (Ma et al., 2023) or SliceGPT (Ashkboos et al., 2024)—is prohibitively expensive at 15B+ parameter scales. Gradient-based methods can require multiple backward passes, which for a model of this size can be as expensive as the pruning process itself, defeating the purpose of seeking cost savings.

**3. Accuracy recovery with minimal data.** Pruning without any retraining causes a sharp drop in model quality (Table 1 shows LM validation loss jumping from 2.062 to as high as 12.27 on a from-scratch baseline). Recovering accuracy traditionally requires re-running the model on substantial amounts of data. But if retraining requires a large fraction of the original pretraining corpus, the cost savings evaporate—the entire point is to use a small fraction (<3%) of the original data, as stated in the abstract: "re-training it with a fraction (<3%) of the original training data."

**4. Catastrophic forgetting at high compression rates.** Compressing Nemotron-4 15B to 4B means discarding ~73% of the original model weights. Doing this in a single step—removing 73% of parameters all at once—can cause catastrophic information loss that no amount of lightweight retraining can recover. The paper finds that iterative pruning (15B → 8B → 4B) achieves a **12% improvement in MMLU scores** compared to one-shot compression, demonstrating that intermediate retraining stages help preserve knowledge (Table 11, comparing the last two rows: 37.81% MMLU for one-shot vs. 42.45% for iterative).

### Prior Approaches and Where They Fall Short

The paper surveys the structured pruning landscape and identifies specific limitations that motivate its contributions (Section 5, Related Work).

#### Depth-Only Pruning Methods

Recent work has focused exclusively on removing layers, using metrics like:
- **Perplexity-based importance** (Shortened LLaMA; Kim et al., 2024): remove a layer, measure the increase in perplexity on a calibration set, use that increase as the layer's importance score.
- **Block Importance (BI)** (ShortGPT; Men et al., 2024): compute the cosine distance between the input and output of each layer; layers with high similarity (low BI) are redundant and can be removed.
- **Layer Collapse** (LaCo; Yang et al., 2024): when removing layers, add their residual information into neighboring surviving layers to preserve knowledge.

These methods are simple and computationally lightweight, but they operate on only one dimension. As the paper shows in Table 10, a purely depth-pruned 8B model achieves an LM loss of 2.155–2.177, while a width-pruned 8B model reaches 2.049—a substantial gap. Depth pruning alone leaves significant performance on the table because it cannot exploit redundancy in attention heads, MLP neurons, and embedding channels.

#### Width-Only and Width+Depth Pruning Methods

Several recent methods target attention heads, MLP channels, and embeddings:
- **LLM-Pruner** (Ma et al., 2023): uses learned masks and Taylor-based gradient importance to identify which width dimensions to prune.
- **SliceGPT** (Ashkboos et al., 2024): replaces weight matrices with smaller ones by applying orthogonal transformations to remove redundant rows and columns.
- **Sheared LLaMA** (Xia et al., 2023): uses learnable masks with an Augmented Lagrangian loss formulation to arrive at optimal width masks.

The paper identifies several critical shortcomings in these approaches:

**1. Gradient dependence at scale.** Most width-pruning methods require gradient computation to estimate importance or learn masks. At 15B parameters, this is both memory-intensive (needing to store gradients alongside model parameters) and compute-intensive (requiring backward passes through the full model). The paper states this directly:

> "At LLM scale, this strategy has multiple disadvantages: (1) it requires compute and memory-intensive gradient computations, and (2) it requires a considerable amount of data and fine-tuning to arrive at reasonable masks."

**2. Single-axis focus.** No prior method simultaneously addresses depth AND width dimensions. Depth-only methods miss width redundancy; width-only methods miss depth redundancy. The paper positions itself as the first to provide a unified framework that handles both:

> "To the best of our knowledge, we provide the first pruning strategy that (1) simultaneously targets both width and depth dimensions, (2) works at LLM scale (i.e., uses only forward passes for computing importance and uses a small fraction of pretraining data), and (3) achieves state-of-the-art compression and accuracy."

**3. Retraining data inefficiency.** Critically, no prior structured pruning work—neither depth-only nor width-only—explores **knowledge distillation** as a retraining strategy for accuracy recovery. Prior methods typically use conventional training (next-token prediction against ground truth labels) for retraining, which the paper shows is far less data-efficient. Table 11 demonstrates this concretely: under iso-compute settings, a conventionally retrained pruned model achieves 24.57% MMLU, while a distillation-retrained model reaches 37.81%—a 13.5 percentage point improvement with the same compute budget. The paper explicitly claims this gap:

> "To the best of our knowledge, we are the first to employ distillation from an uncompressed teacher to improve the retraining of structurally-pruned student models."

This is the paper's most significant departure from prior work: it recognizes that the uncompressed parent model contains rich knowledge about token distributions, hidden states, and intermediate representations that can be transferred to the compressed child model much more efficiently than learning from raw text alone. The teacher model's output distributions guide the student toward solutions it would otherwise need many more tokens to discover independently.

### How This Paper Positions Itself

The paper frames its contribution not as a single novel pruning technique but as an **empirically-grounded compression methodology** derived from systematic experimentation across all relevant axes. This is evident in the careful structuring of the work:

**1. It treats pruning strategy selection as an empirical question, not a theoretical one.** Rather than proposing a new importance metric and claiming it is optimal, the paper tests three different aggregation functions (mean absolute value, L2 norm, variance) across two aggregation dimensions (batch, sequence) for activation-based importance—a total of 9 combinations—and discovers that the choice matters substantially (Table 13: zero-shot LM loss varies from 7.18 to 10.55 depending on the metric). It then validates that these differences persist after retraining (Figure 5), ensuring the finding is practically meaningful rather than an artifact of zero-shot evaluation.

**2. It recognizes that interactions between axes are non-trivial and must be explored empirically.** Section 4.2's comparison of width vs. depth+width pruning (Table 10 and Figure 6) reveals a surprising dynamic: the combined pruning approach initially (pre-retraining) looks better, but after ~200 steps of retraining, pure width pruning overtakes it. This is the kind of interaction that would be invisible in a purely analytical or single-axis study. The paper translates this into a concrete best practice: "Prefer width pruning over depth for the model scales we consider (≤15B)" (Best Practice #4).

**3. It positions distillation as a first-class retraining strategy, not an afterthought.** The paper explores distillation-specific design choices—which loss functions to use (KLD vs. MSE vs. cosine vs. reverse KLD in Table 15-16), which intermediate states to distill (logits only, logits+embeddings, logits+encoder block outputs in Table 17-18), and how to map layers between teacher and student when depths differ (Best Practices #6-7). Each of these choices is evaluated empirically, producing concrete guidance.

**4. It tackles the practical complexity of architecture search post-pruning.** When you prune attention heads, MLP channels, and embedding dimensions simultaneously, there are many possible target architectures that meet a given parameter budget. Section 2.3's Figure 3 outlines a lightweight neural architecture search process: enumerate feasible architectures within a ~5% parameter target range, perform brief retraining (~1.8B tokens) to stabilize rankings (Figure 9 shows rankings stabilize after ~300 steps), and select the best candidate for full retraining. This search process is crucial because the optimal distribution of parameters across components—how many heads vs. how many MLP channels vs. what embedding dimension—is not obvious a priori.

**5. It connects to the broader research agenda of compute-efficient model production.** The paper fits into a growing recognition that the "train from scratch" paradigm for every model variant is unsustainable. Related trends include continued pretraining (Parmar et al., 2024a, 2024b), multi-phase training (Hu et al., 2024; Shen et al., 2024), and small-model specialization (Gunasekar et al., 2023; Mitra et al., 2023). However, this paper is distinct in treating **pruning as a direct structural compression of an existing model** rather than as a training efficiency technique applied during pretraining. The resulting MINITRON models are literally smaller versions of the same architecture with weights inherited and refined from their larger parent, rather than independently initialized models trained with some efficiency tricks.

### A Critical Limitation in Scope

It's worth noting what this paper does NOT attempt to solve, because understanding the boundaries clarifies the positioning. The paper explicitly focuses on the regime where **the base model already possesses the necessary knowledge**, and compression is about preserving and reorganizing that knowledge into a smaller footprint. When the 15B model itself cannot solve hard problems, its 8B and 4B descendants will not suddenly gain that capability through pruning and retraining—compression amplifies existing capability but cannot create new capability from nothing. This is analogous to the test-time compute paper's finding that hard problems remain unsolved regardless of inference budget, and it establishes a clear boundary condition: pruning-based model production is suitable when the largest sibling is sufficiently capable to serve as a knowledge source for its compressed variants.

## 3. Technical Approach

### 3.1 Reader orientation (approachable technical breakdown)

This paper develops a **methodology for compressing a large, already-trained language model into smaller models through structured pruning and knowledge-distillation-based retraining**, rather than training each smaller model from scratch. The core problem it solves is the **prohibitive cost of training LLM families**: the solution takes the shape of an empirical recipe book — a set of ten best practices derived from systematic experiments across which pruning axes to use, how to measure importance, what order to prune in, how to retrain efficiently, and how to search for the best compressed architecture — that together enable producing accurate 8B and 4B models from a 15B parent using only ~94B retraining tokens (~1% of the original 8 trillion token pretraining corpus).

### 3.2 Big-picture architecture (diagram in words)

The system has five major stages, executed sequentially for each compression target:

1. **Importance estimation** — The fully-trained 15B model processes a small calibration dataset (1024 samples) through forward passes only (no backpropagation), and activation-based metrics compute an importance score for every layer (depth axis), attention head (width axis), MLP neuron (width axis), and embedding channel (width axis).

2. **Ranking and pruning** — Importance scores are sorted to produce a ranking; the least important elements are removed by directly reshaping weight matrices (for width axes) or deleting layers (for depth). This produces a structurally smaller model that inherits the surviving weights from the parent.

3. **Lightweight neural architecture search (NAS)** — When multiple axes are pruned simultaneously, many target architectures (different combinations of layer count, head count, MLP hidden dimension, and embedding dimension) can meet a given parameter budget. A brief retraining phase (~1.8B tokens) evaluates each candidate architecture to select the best one for full retraining.

4. **Knowledge-distillation-based retraining** — The pruned model (student) is retrained by learning to mimic the output logit distributions and/or intermediate hidden states of the original uncompressed model (teacher), using a small fraction of the original pretraining data. This transfers knowledge from teacher to student far more efficiently than retraining from scratch with conventional next-token prediction.

5. **Iterative compression for large size reductions** — For aggressive compression (e.g., 15B → 4B, a 73% parameter reduction), the process is applied iteratively: first compress 15B → 8B with full distillation retraining, then compress the resulting 8B → 4B. This avoids the catastrophic information loss that occurs with one-shot aggressive pruning.

The output is a family of smaller models (MINITRON 8B, MINITRON 4B) that share the same architectural DNA as the parent but require dramatically less training compute to produce than independently trained models of equivalent size.

### 3.3 Roadmap for the deep dive

- **First**, the formal notation for transformer components (MLP layers, multi-head attention, layer normalization) and the activation-based importance estimation strategy — since all pruning decisions depend on correctly measuring what matters.
- **Second**, the per-axis pruning mechanics: how width pruning (attention heads, MLP neurons, embedding channels) and depth pruning (layers) work, including the specific metrics and aggregation functions for each axis and the critical "head residual preservation" trick.
- **Third**, the lightweight neural architecture search procedure — how architectures are enumerated, evaluated, and selected, since the optimal distribution of parameters across components is not obvious a priori.
- **Fourth**, the knowledge distillation retraining pipeline — loss functions, intermediate state mappings, teacher-student layer alignment, and the critical finding that logit-only distillation suffices when depth isn't reduced significantly.
- **Fifth**, the iterative compression strategy for large reductions — why one-shot aggressive pruning fails and how intermediate retraining stages preserve knowledge.

### 3.4 Detailed, sentence-based technical breakdown

This is primarily an **empirical methodology paper** whose core contribution is a systematic exploration of design choices in LLM structured pruning and retraining, producing a validated set of best practices rather than a single novel algorithm.

---

#### Notation and Transformer Architecture

The paper begins by defining the transformer components that will be pruned, establishing precise notation for the weight matrices and operations involved (Section 2.1). Understanding this notation is essential because pruning physically modifies these weight matrices by removing rows, columns, or entire sub-blocks.

**Multi-Layer Perceptron (MLP) layers** in the transformer consist of two linear transformations with a non-linear activation between them:

> $$\text{MLP}(\mathbf{X}) = \delta\left(\mathbf{X} \cdot \mathbf{W}_1^T\right) \cdot \mathbf{W}_2$$

where `$\mathbf{X}$` is the input tensor with shape `[batch, sequence, d_model]`, `$\mathbf{W}_1, \mathbf{W}_2 \in \mathbb{R}^{d_{hidden} \times d_{model}}$` are the two weight matrices (with `$d_{model}$` as the embedding dimension and `$d_{hidden}$` as the MLP hidden dimension, typically 2.5× to 4× larger than `$d_{model}$`), and `$\delta(\cdot)$` is the non-linear activation function (e.g., SwiGLU, GELU, or ReLU depending on the architecture).

**What it computes:** The first linear transformation `$\mathbf{X} \cdot \mathbf{W}_1^T$` projects each token's `$d_{model}$`-dimensional representation into a `$d_{hidden}$`-dimensional hidden space, the non-linearity `$\delta(\cdot)$` applies element-wise activation to introduce non-linear capacity, and the second linear transformation `$(\cdot) \cdot \mathbf{W}_2$` projects back to `$d_{model}$` dimensions so the output can be added as a residual to the input. The output `$\text{MLP}(\mathbf{X})$` has the same shape as `$\mathbf{X}$`: `[batch, sequence, d_model]`.

**Why this form matters for pruning:** Each row of `$\mathbf{W}_1$` corresponds to one MLP neuron (one hidden dimension). Pruning a neuron means removing the corresponding row from `$\mathbf{W}_1$` and the corresponding column from `$\mathbf{W}_2$`, reducing `$d_{hidden}$`. Since `$d_{hidden}$` is typically 2.5–4× larger than `$d_{model}$`, the MLP layers contain the majority of model parameters, making neuron pruning the most impactful width-pruning axis.

**Multi-Head Attention (MHA)** is defined for an input `$\mathbf{X}$` as:

> $$\text{MHA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_L) \cdot \mathbf{W}^O$$
> $$\text{head}_i = \text{Attn}(\mathbf{X}\mathbf{W}^{Q,i}, \mathbf{X}\mathbf{W}^{K,i}, \mathbf{X}\mathbf{W}^{V,i})$$

where `$\mathbf{W}^{Q,i}, \mathbf{W}^{K,i}, \mathbf{W}^{V,i} \in \mathbb{R}^{d_{head} \times d_{model}}$` are the query, key, and value projection matrices for the `$i$`-th head, `$d_{head}$` is the dimension of each attention head (typically `$d_{model} / L$` for standard multi-head attention, or larger for grouped-query attention), `$L$` is the total number of heads, `$\mathbf{W}^O \in \mathbb{R}^{L \cdot d_{head} \times d_{model}}$` is the output projection matrix that combines all heads, and `$\text{Attn}(\cdot)$` is the scaled dot-product attention. The `$\text{Concat}$` operation concatenates the outputs of all `$L$` heads along the last dimension, producing a tensor of shape `[batch, sequence, L · d_head]`, which `$\mathbf{W}^O$` then projects back to `$d_{model}$`.

**What it computes:** Each head independently computes attention scores between all positions in the sequence using its own learned projections: queries (`$\mathbf{X}\mathbf{W}^{Q,i}$`) are compared against keys (`$\mathbf{X}\mathbf{W}^{K,i}$`) via dot products to produce attention weights, which are then used to aggregate values (`$\mathbf{X}\mathbf{W}^{V,i}$`). The `$L$` heads allow the model to attend to different aspects of the input simultaneously (e.g., syntax, semantics, long-range dependencies). The output projection `$\mathbf{W}^O$` combines these multi-faceted representations into a single `$d_{model}$`-dimensional output per token.

**Why this form matters for pruning:** Pruning an attention head means removing `$\mathbf{W}^{Q,i}$`, `$\mathbf{W}^{K,i}$`, `$\mathbf{W}^{V,i}$`, and the corresponding slice of `$\mathbf{W}^O$` (the column block that processes that head's output). This directly reduces parameter count and also changes the concatenation dimension — `$\text{Concat}$` now operates over `$K$` heads instead of `$L$`, and `$\mathbf{W}^O$` shrinks from `$\mathbb{R}^{L \cdot d_{head} \times d_{model}}$` to `$\mathbb{R}^{K \cdot d_{head} \times d_{model}}$`. The Nemotron-4 15B model uses **grouped-query attention (GQA)** with 48 query heads but only 8 key-value groups (Table 5), meaning the key and value projections are shared across groups of query heads — this sharing must be preserved during pruning.

**Layer Normalization (LayerNorm)** on an input `$\mathbf{X}$` is defined as:

> $$\text{LN}(\mathbf{X}) = \frac{\mathbf{X} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$

where `$\mu$` and `$\sigma^2$` are the mean and variance computed across the embedding dimensions for each token independently, `$\epsilon$` is a small constant for numerical stability, `$\gamma, \beta \in \mathbb{R}^{d_{model}}$` are learnable scale and shift parameters, and `$\odot$` denotes element-wise multiplication. This normalization is applied before both the MHA and MLP sub-layers in each transformer block, and also at the final output before the language modeling head.

**What it computes:** For each token, LayerNorm standardizes its `$d_{model}$`-dimensional representation to have zero mean and unit variance across the embedding dimensions, then applies a learned affine transformation (scale by `$\gamma$`, shift by `$\beta$`) to restore representational flexibility. The result is a `$d_{model}$`-dimensional normalized vector per token.

**Why this form matters for pruning:** The `$\gamma$` and `$\beta$` parameters provide a natural importance signal for embedding channels — each channel's scale factor `$\gamma_i$` can be interpreted as the learned importance of that embedding dimension to the normalization's downstream computation. The paper uses the activations of LayerNorm (specifically `$\text{LN}(\mathbf{X})_i$`, the `$i$`-th channel of the normalized output) as the basis for embedding channel importance estimation, which incorporates both the learned scale and the data-dependent activation pattern.

---

#### Activation-Based Importance Estimation

The paper proposes a **purely activation-based importance estimation strategy** that simultaneously computes sensitivity information for all four pruning axes (depth, neuron, head, and embedding channel) using only forward propagation passes on a small calibration dataset of 1024 samples (Section 2.2). This is the foundation on which all pruning decisions rest, and the choice to avoid gradients is deliberate: computing gradient information for a 15B parameter model is "prohibitively memory and compute-intensive" and would defeat the purpose of seeking cost savings.

**The calibration dataset `$\mathcal{D}$`** consists of 1024 samples drawn randomly from the full 8-trillion-token pretraining dataset, with a batch size and sequence length matching the model's training configuration (batch size 1152, sequence length implicitly determined by the training setup). The small size (1024 samples vs. the 8 trillion training tokens) is essential — if importance estimation required processing a large fraction of the training data, the overall cost advantage of pruning-based model production would diminish.

**Width axis importance — attention heads:**

> $$F^{(i)}_{\text{head}} = \sum_{\mathcal{B}, \mathcal{S}} \left\| \text{Attn}(\mathbf{X}\mathbf{W}^{Q,i}, \mathbf{X}\mathbf{W}^{K,i}, \mathbf{X}\mathbf{W}^{V,i}) \right\|_2$$

where `$\mathcal{B}$` denotes aggregation along the batch dimension (summing over all samples in the batch), `$\mathcal{S}$` denotes aggregation along the sequence dimension (summing over all token positions in the sequence), `$\|\cdot\|_2$` is the L2 norm (Euclidean norm), and `$\text{Attn}(\cdot)$` is the output of the `$i$`-th attention head before the output projection `$\mathbf{W}^O$`. The summation `$\sum_{\mathcal{B}, \mathcal{S}}$` computes a single scalar importance score for each head by accumulating its activation magnitudes across all tokens and all calibration samples.

**What it computes:** For each attention head in each layer, the system runs the calibration data through the model, captures the head's output tensor (shape `[batch, sequence, d_head]`), computes the L2 norm per head output (treating each head's output as a `$d_{head}$`-dimensional vector and computing `$\sqrt{\sum_{j=1}^{d_{head}} (\text{output}_j)^2}$`), and sums these norms across all batch elements and all sequence positions. Heads that produce consistently high-magnitude outputs across many tokens are deemed important; heads that produce near-zero outputs are candidates for removal.

**Why activation magnitude as a proxy for importance:** A head whose output is consistently near zero is contributing little information to the subsequent computation — removing it would not significantly change the model's internal representations. This is a first-order Taylor approximation argument: the change in the model's output when removing a head is approximately proportional to the head's output magnitude (since the derivative with respect to the head's output is the subsequent layer's weight). Activation magnitude captures this leading-order effect without requiring gradient computation — it is a "zeroth-order" importance metric that works because the model has already been trained and the weights encode the sensitivity to each component's output indirectly through their own magnitudes.

**Width axis importance — MLP neurons:**

> $$F^{(i)}_{\text{neuron}} = \sum_{\mathcal{B}, \mathcal{S}} \left\| \mathbf{X} \cdot (\mathbf{W}_1^i)^T \right\|_2$$

where `$\mathbf{W}_1^i$` is the `$i$`-th row of the first MLP weight matrix `$\mathbf{W}_1$` (corresponding to the `$i$`-th hidden neuron), `$\mathbf{X} \cdot (\mathbf{W}_1^i)^T$` computes the pre-activation output of that specific neuron across all tokens (shape `[batch, sequence]`), and the L2 norm and summation aggregate these scalar activations into a single importance score. Note that this is the activation BEFORE the non-linearity `$\delta(\cdot)$` — the paper measures the magnitude of the linear projection, not the post-activation value.

**What it computes:** For each neuron in each MLP layer, the system computes the dot product between each token's `$d_{model}$`-dimensional representation and the neuron's weight vector `$\mathbf{W}_1^i$`, producing a scalar activation per token per batch element. These scalars are aggregated via L2 norm (which emphasizes large activations more than small ones compared to simple averaging) and summed across batch and sequence dimensions.

**Why pre-activation rather than post-activation:** Measuring importance before the activation function captures the raw contribution of the neuron's linear filter, independent of any saturation effects from the non-linearity. If a neuron with a ReLU activation consistently produces large positive pre-activations, it is clearly important; but a neuron that occasionally saturates into the zero-gradient region might still be structurally important. The pre-activation metric avoids this ambiguity.

**Width axis importance — embedding channels:**

> $$F^{(i)}_{\text{emb}} = \sum_{\mathcal{B}, \mathcal{S}} \left| \text{LN}(\mathbf{X})_i \right|$$

where `$\text{LN}(\mathbf{X})_i$` is the `$i$`-th channel (embedding dimension) of the LayerNorm output — specifically, the normalized and scaled activation `$(\mathbf{X}_i - \mu)/\sqrt{\sigma^2 + \epsilon} \cdot \gamma_i + \beta_i$` — and `$|\cdot|$` denotes the absolute value. The absolute value is used here rather than the L2 norm because each channel is a scalar (the L2 norm of a scalar equals its absolute value, but the paper's aggregation experiments in Table 13 test different functions including absolute value, L2 norm applied element-wise which reduces to absolute value, and variance).

**What it computes:** For each of the `$d_{model}$` embedding dimensions in each LayerNorm operation (which exists at the input to each transformer block and at the final output), the system computes the absolute activation magnitude of that channel across all tokens, then sums across batch and sequence dimensions. Channels that consistently carry large-magnitude signals are important; channels near zero are candidates for pruning.

**Why LayerNorm activations for embedding importance:** The LayerNorm operation sits at a critical junction in the transformer — it normalizes the representation before it enters the attention and MLP sub-layers. The `$\gamma$` parameter for each channel is a learned scale that the model can increase to emphasize important dimensions or decrease to suppress noisy ones. The activation `$\text{LN}(\mathbf{X})_i$` incorporates both the learned scale `$\gamma_i$` and the data-dependent signal `$(\mathbf{X}_i - \mu)/\sqrt{\sigma^2 + \epsilon}$`, providing a natural importance measure that reflects how the model actually uses each embedding dimension on real data. Pruning an embedding channel means reducing `$d_{model}$`, which affects ALL subsequent weight matrices (MLP, MHA, LayerNorm) — the weight matrices are trimmed along the embedding dimension, effectively "squeezing out" that channel from the entire model.

**Layer-wise to network-wide aggregation:** The formulas above compute importance scores independently per layer. To obtain a single global importance ranking for each axis (e.g., which heads across ALL layers are least important?), the per-layer scores are summed across layers: a head in layer 5 with importance `$F^{(i)}_{\text{head},\ell=5} = 3.2$` and a head in layer 20 with importance `$F^{(j)}_{\text{head},\ell=20} = 2.1$` would be ranked globally, with the layer-20 head being pruned first. This implicitly assumes that importance scores are comparable across layers — an assumption the paper validates empirically through the quality of downstream results but does not explicitly test.

**Aggregation function experiments (Table 13):** The paper discovers that the choice of HOW to aggregate across batch and sequence dimensions matters substantially. For a sequence of scores `$S$` (the activations across batch items and tokens), three functions are tested:

- **mean(abs)**: `$\frac{1}{n}\sum_{i=1}^n |S_i|$` — average absolute activation magnitude
- **L2 norm**: `$\sqrt{\sum_{i=1}^n S_i^2}$` — Euclidean norm of the activation vector
- **variance**: `$\frac{1}{n}\sum_{i=1}^n (S_i - \bar{S})^2$` — activation variance

The batch dimension and sequence dimension are aggregated SEPARATELY with potentially different functions, producing 9 combinations (3 batch functions × 3 sequence functions). Table 13 reports zero-shot LM validation loss on two datasets (8T blend and WikiText2) for each combination after pruning Nemotron-4 15B to the Nemotron-3 8B architecture with NO retraining:

| Batch Agg. | Seq. Agg. | 8T LM Loss | WikiText2 LM Loss |
|------------|-----------|------------|-------------------|
| L2 | L2 | 8.73 | 8.37 |
| **L2** | **mean** | **7.18** | **7.23** |
| L2 | var | 8.18 | 8.61 |
| mean | L2 | 8.41 | 7.84 |
| **mean** | **mean** | **7.21** | **6.89** |
| mean | var | 7.94 | 8.29 |
| var | L2 | 9.01 | 9.30 |
| var | mean | 8.34 | 8.72 |
| var | var | 10.55 | 11.14 |

The best combinations are **(batch=L2, seq=mean)** with 7.18 LM loss and **(batch=mean, seq=mean)** with 7.21 LM loss. The worst combinations (var/var, var/L2) produce dramatically higher losses (>10 vs. ~7). The paper selects **(batch=L2, seq=mean)** as the primary metric (Best Practice #2), noting its slightly better performance on the 8T dataset.

**Why L2/mean works better than alternatives:** The L2 norm across the batch dimension emphasizes samples that produce large activations — it prevents a single outlier sample from dominating the importance score while still giving more weight to high-activation samples than simple averaging would. The mean across the sequence dimension treats each token position as equally important, which is appropriate because pruning should preserve capability across all sequence positions, not just the most active ones. Variance-based aggregation performs poorly because it measures dispersion around the mean rather than magnitude — a head could have high variance but low average magnitude (oscillating between small positive and negative values), which would be incorrectly marked as important by variance but correctly marked as low-importance by L2 norm or mean absolute value.

**Validation post-retraining (Figure 5):** To ensure the zero-shot rankings persist after retraining (which is when models are actually deployed), the paper prunes the 15B model to 8B using: (1) the best metric (L2, mean), and (2) a poorly-performing metric (L2, L2). Both pruned models are retrained for 400 steps (~1.8B tokens). The (L2, mean) model consistently achieves lower LM validation loss throughout retraining, confirming that the zero-shot rankings are predictive of retrained performance — a property that is not guaranteed (some metrics might be good at identifying components that matter for zero-shot performance but become irrelevant after retraining redistributes function). Figure 5 shows the (L2, L2) model starting at higher loss and remaining above the (L2, mean) curve for all 400 steps, without any crossover.

---

#### Depth (Layer) Importance Estimation

For depth pruning — removing entire transformer layers — the paper evaluates two metrics that operate at the layer level rather than per-component within a layer (Section 2.2).

**Perplexity-based (PPL) importance**, adopted from Shortened LLaMA (Kim et al., 2024):

> The procedure removes a single layer at a time and measures the increase in perplexity of this pruned model on a calibration set. The perplexity increase serves as the "importance" or sensitivity of that layer.

**What it computes:** For each layer `$i$` in the model (32 total in Nemotron-4 15B), the system creates a temporary model variant with layer `$i$` removed (input to layer `$i$` is routed directly to layer `$i+1$`), runs forward passes on the calibration data, computes the average perplexity across all tokens, and records the difference from the unpruned model's perplexity. Layers whose removal causes a large perplexity increase are important; layers whose removal barely affects perplexity are redundant.

**Why how it works:** Perplexity measures how well the model predicts the next token — it is `$\exp$(average cross-entropy loss)`. A layer whose removal causes a large perplexity increase is contributing computations that are difficult for other layers to compensate for. This metric directly measures the downstream impact of removing each layer.

**Why perplexity is expensive:** Computing PPL-based importance requires `$L$` separate forward passes through differently-pruned model variants (32 for Nemotron-4 15B), making it `$L \times$` the cost of a single forward pass. This is feasible but slower than the alternative.

**Block Importance (BI)**, adopted from ShortGPT (Men et al., 2024):

> $$\text{BI}_i = 1 - \mathbb{E}_{\mathbf{X}, t} \left[ \frac{\mathbf{X}_{i,t}^T \mathbf{X}_{i+1,t}}{\|\mathbf{X}_{i,t}\|_2 \|\mathbf{X}_{i+1,t}\|_2} \right]$$

where `$\mathbf{X}_i$` is the input to layer `$i$` (the output of the previous layer's residual connection), `$\mathbf{X}_{i,t}$` denotes the `$t$`-th row of `$\mathbf{X}_i$` (the representation of the `$t$`-th token), `$\mathbf{X}_{i+1}$` is the output of layer `$i$` (after the MHA, MLP, and residual connections within layer `$i$`), `$\|\cdot\|_2$` is the Euclidean norm, `$\mathbf{X}_{i,t}^T \mathbf{X}_{i+1,t}$` is the dot product (inner product) between the input and output representations, and `$\mathbb{E}_{\mathbf{X}, t}$` denotes expectation over the calibration data `$\mathbf{X}$` and token positions `$t$`. The fraction `$\frac{\mathbf{X}_{i,t}^T \mathbf{X}_{i+1,t}}{\|\mathbf{X}_{i,t}\|_2 \|\mathbf{X}_{i+1,t}\|_2}$` is the **cosine similarity** between the input and output of layer `$i$` for token `$t$`, ranging from -1 to 1.

**What it computes:** For each layer, the system runs a single forward pass on the calibration data, capturing the input and output tensors of the layer (both shape `[batch, sequence, d_model]`). For each token, it computes the cosine similarity between the input representation and the output representation. The expectation averages these cosine similarities across all tokens and all calibration samples. BI is then `$1 -$` this average cosine similarity. A low BI score (close to 0) means the input and output are very similar in direction — the layer is doing almost nothing, acting as a near-identity function — and is a candidate for removal. A high BI score (close to 1) means the layer is substantially transforming the representation and is important.

**Why this form:** Cosine similarity measures the change in direction of the representation, ignoring magnitude changes. If a layer simply scales all dimensions uniformly (which would change the L2 distance but not the cosine similarity), the BI would remain 0, correctly identifying that the layer is performing a trivial operation. The expectation across tokens ensures the metric captures the layer's average behavior rather than being dominated by outlier tokens. The `$1 -$` transformation inverts the scale: high cosine similarity → low BI → "unimportant"; low cosine similarity → high BI → "important."

**Why BI is computationally efficient:** BI for all layers can be computed in a SINGLE forward pass. The input `$\mathbf{X}_i$` for each layer is naturally available during the forward computation, and the output `$\mathbf{X}_{i+1}$` is available after the layer processes. No separate model variants or additional passes are needed. Additionally, BI can be extended to estimate importance of several contiguous layers simultaneously (following Gromov et al., 2024): instead of measuring `$\mathbf{X}_i$` vs. `$\mathbf{X}_{i+1}$`, measure `$\mathbf{X}_i$` vs. `$\mathbf{X}_{i+k}$` to assess whether a BLOCK of `$k$` consecutive layers is redundant. This is valuable for depth pruning because removing a block of layers might preserve more architectural coherence than removing scattered individual layers.

**Comparison of PPL vs. BI (Table 10):** When pruning depth alone to 8B parameters (removing 16 layers from the 32-layer model to create a 16-layer model), PPL-based importance achieves an LM validation loss of 2.155 after retraining, while BI-based importance achieves 2.177. The difference is small (0.022 in LM loss), suggesting both metrics are viable, but PPL slightly edges ahead. The paper reports both metrics but does not designate one as strictly superior — the choice likely depends on compute budget for the importance estimation step (BI is faster; PPL may be slightly more accurate).

---

#### Obtaining a Pruned Model: Width Axes

Once importance scores are computed and ranked, the actual pruning operation physically modifies the model's weight matrices (Section 2.3). The procedure differs by axis:

**Pruning MLP neurons:** Given a target MLP hidden dimension `$d_{hidden}^{target} < d_{hidden}^{original}$`, the system ranks all neurons across all MLP layers by their `$F^{(i)}_{\text{neuron}}$` scores, selects the `$d_{hidden}^{target}$` highest-scoring neurons within each layer (not globally across layers — the paper prunes each layer to the same target dimension rather than allowing layer-specific dimensions), and:

1. Removes the corresponding rows from `$\mathbf{W}_1$`: `$\mathbf{W}_1^{new} \in \mathbb{R}^{d_{hidden}^{target} \times d_{model}}$`
2. Removes the corresponding columns from `$\mathbf{W}_2$`: `$\mathbf{W}_2^{new} \in \mathbb{R}^{d_{model} \times d_{hidden}^{target}}$`

The surviving rows of `$\mathbf{W}_1$` and columns of `$\mathbf{W}_2$` are **unchanged** — their weight values are inherited directly from the parent model. No additional initialization or perturbation is applied. This is crucial: the pruned model starts from a subset of the parent's weights, giving it a better initialization than random weights and enabling the data-efficient retraining that follows.

**Pruning attention heads:** Given a target number of heads `$K < L$`, the system ranks all heads across all layers, selects the `$K$` highest-scoring heads within each layer (again, layer-local pruning to uniform head count), and:

1. Removes `$\mathbf{W}^{Q,i}$`, `$\mathbf{W}^{K,i}$`, `$\mathbf{W}^{V,i}$` for pruned heads `$i$`
2. Removes the corresponding column block from `$\mathbf{W}^O$` (since `$\mathbf{W}^O$` expects `$K \cdot d_{head}$` input dimensions instead of `$L \cdot d_{head}$`)

**Head residual preservation — a critical design choice:** When pruning attention heads, the paper introduces a mechanism to preserve knowledge from pruned heads by adding their residual contribution back into the SURVIVING heads. This is an MHA analog of Layer Collapse (Yang et al., 2024) for depth pruning. Formally, given `$L$` original heads being pruned to `$K$` heads, each new (surviving) head `$i$` is modified to:

> $$\text{head}_i^{\text{new}} = \text{head}_i + (\text{head}_i - \text{head}_{2K - i + 1})$$

for `$i \in [K - (L - K), K]`. This applies to the TAIL of the surviving heads — specifically, the last `$L - K + 1$` surviving heads (the ones that have a corresponding pruned head "mirrored" around the midpoint). The term `$\text{head}_{2K - i + 1}$` selects a pruned head whose index is symmetrically opposite to `$i$` relative to the pruning boundary.

**What it computes:** For each surviving head near the pruning boundary, take its own output, subtract the output of the corresponding pruned head, and add this difference to the surviving head's output. This has the effect of "folding" the information from pruned heads into their surviving counterparts, making the surviving heads responsible for representing what the pruned heads previously captured.

**Why this form:** The difference `$\text{head}_i - \text{head}_{2K - i + 1}$` captures the UNIQUE information in head `$i$` that is not present in the pruned head `$2K - i + 1$`. Adding this to head `$i$` amplifies its distinguishing features. The symmetric pairing `$(i, 2K - i + 1)$` pairs the least-important surviving head with the most-important pruned head, the second-least-important surviving head with the second-most-important pruned head, and so on — a heuristic intended to match heads that might represent complementary information.

**Why this matters:** Without residual preservation, pruning an attention head simply deletes it, discarding whatever representational capacity it had developed during pretraining. With residual preservation, the information from pruned heads is partially transferred to surviving heads, giving the model a better starting point for retraining. The paper notes that this provides "a boost to model accuracy in our experiments" but does not quantify the boost with an ablation study comparing pruning with vs. without residual preservation.

**Grouped-query attention (GQA) handling:** The Nemotron-4 15B model uses 48 query heads but only 8 key-value groups (Table 5). In GQA, the key and value projections are shared across groups of query heads — each group of `$48/8 = 6$` query heads shares one key and one value projection. When pruning query heads, the residual preservation strategy "is applied only to the query heads" — the key and value groups remain unchanged (since they are shared, pruning a query head within a group does not necessarily prune the key/value projection). This detail is mentioned but not elaborated: the precise interaction between query head pruning and GQA group structure could affect the viability of residual preservation, but the paper's results suggest it works in practice.

**Pruning embedding channels:** Given a target embedding dimension `$d_{model}^{target} < d_{model}^{original}$`, the system ranks all embedding channels by their `$F^{(i)}_{\text{emb}}$` scores and selects the `$d_{model}^{target}$` highest-scoring channels. Embedding channel pruning is the most structurally invasive width-pruning operation because it affects almost every weight matrix in the model:

1. **MLP layers:** Trim `$\mathbf{W}_1$` and `$\mathbf{W}_2$` along their `$d_{model}$` dimension — `$\mathbf{W}_1$` changes from `$\mathbb{R}^{d_{hidden} \times d_{model}^{original}}$` to `$\mathbb{R}^{d_{hidden} \times d_{model}^{target}}$` (removing columns corresponding to pruned channels), and `$\mathbf{W}_2$` changes from `$\mathbb{R}^{d_{model}^{original} \times d_{hidden}}$` to `$\mathbb{R}^{d_{model}^{target} \times d_{hidden}}$` (removing rows).

2. **MHA layers:** Trim `$\mathbf{W}^{Q,i}$`, `$\mathbf{W}^{K,i}$`, `$\mathbf{W}^{V,i}$` along their `$d_{model}$` dimension (columns removed), and trim `$\mathbf{W}^O$` along its `$d_{model}$` dimension (rows removed). The head dimension `$d_{head}$` remains unchanged — only the embedding dimension that feeds into and out of the attention mechanism changes.

3. **LayerNorm:** Trim `$\gamma$` and `$\beta$` to length `$d_{model}^{target}$`.

4. **Embedding layer:** Trim the token embedding matrix from `$\mathbb{R}^{V \times d_{model}^{original}}$` to `$\mathbb{R}^{V \times d_{model}^{target}}$`, where `$V=256,000$` is the vocabulary size. This is a significant parameter reduction: the embedding matrix alone accounts for `$V \times d_{model}$` parameters, and reducing `$d_{model}$` from 6144 to 4096 (as in MINITRON 8B) saves `$256,000 \times (6144 - 4096) \approx 524$` million parameters.

**Why the order of pruning matters:** The paper notes that when multiple width axes are pruned simultaneously (neurons + heads + embedding channels), the order of operations can affect the final architecture. Trimming embedding channels changes `$d_{model}$`, which changes the dimensions of `$\mathbf{W}_1$`, `$\mathbf{W}_2$`, and the attention projection matrices. This means the importance ranking for neurons and heads computed on the original `$d_{model}$` dimensions may not perfectly reflect their importance in the reduced `$d_{model}$` space. The paper's iterative approach (Section 2.2, described below) partly addresses this by re-computing importance after intermediate pruning steps, but the base procedure applies all width pruning simultaneously using the original model's importance scores.

---

#### Iterative Importance Estimation and Why It Doesn't Help

An intuitive hypothesis is that interleaving importance estimation with pruning should produce better results: prune a little, re-compute importance on the pruned model (which has a different computational structure), prune a little more, etc. The paper tests this explicitly and finds it provides **no benefit** (Section 2.2, Best Practice #3).

Formally, given `$T$` iterations and source/target dimensions `$d_s$` and `$d_t$`, the iterative procedure computes importance on a model with `$d_s - i \cdot \frac{d_s - d_t}{T}$` dimensions and prunes to `$d_s - (i+1) \cdot \frac{d_s - d_t}{T}$` dimensions for `$i \in [0, T-1]$`. For example, pruning the embedding dimension from 6144 to 4096 with `$T=4$` iterations would prune in steps of `$(6144 - 4096)/4 = 512$` dimensions per iteration: prune 512 channels, re-compute importance on the 5632-dimensional model, prune another 512, re-compute, etc.

**Results (Table 14):** When pruning the embedding dimension from 6144 to 4096 with `$T=1$` (one-shot), `$T=2$`, and `$T=4$` iterations, the initial zero-shot validation losses differ (5.43 for `$T=1$`, 5.55 for `$T=2$`, 5.24 for `$T=4$`), with iterative appearing better pre-retraining. However, after lightweight retraining (400 steps, ~1.8B tokens), ALL three variants converge to exactly the same final validation loss of **1.92**. The retraining process fully compensates for any differences in the initial pruning quality.

**Why iterative provides no benefit:** There are two plausible explanations: (1) The retraining process is powerful enough to recover from suboptimal initial pruning decisions — given enough training steps, the model can reorganize its remaining capacity to compensate for which specific channels were removed. (2) The importance metric is sufficiently good in a single shot that iterating doesn't change the ranking meaningfully — the least important channels remain the least important even after intermediate pruning. The paper leans toward explanation (1) based on the convergence pattern in Table 14. Regardless, the practical implication is clear: **single-shot importance estimation is sufficient**, saving the computational cost of `$T$` separate importance estimation passes.

---

#### Lightweight Neural Architecture Search

When pruning along multiple width axes simultaneously — reducing the number of layers, attention heads, MLP hidden dimension, and embedding dimension — the resulting architecture has degrees of freedom beyond simply "remove the least important components." Should we allocate parameter budget toward more heads or a larger MLP hidden dimension? A deeper but narrower model or a shallower but wider one? The paper addresses this with a **lightweight neural architecture search (NAS)** procedure (Section 2.3, Figure 3).

**Search space definition (Table 12):** For each target model size, the paper defines a discrete search space:

- **MINITRON 8B**: layers `$\in$` {29, 30, 31, 32}, heads `$\in$` {32, 48}, MLP expansion factor (the ratio `$d_{hidden}/d_{model}$`) `$\in$` {2.5, 3, 3.5, 4}, embedding dimension `$\in$` {4096, 4680, 5120, 5632, 6144}. Total combinations: `$4 \times 2 \times 4 \times 5 = 160$` possible architectures.

- **MINITRON 4B**: layers `$\in$` {29, 30, 31, 32}, heads `$\in$` {24, 32, 48}, MLP expansion factor `$\in$` {2.5, 3, 3.5, 4}, embedding dimension `$\in$` {2560, 3072, 3584, 4096, 4608}. Total combinations: `$4 \times 3 \times 4 \times 5 = 240$` possible architectures.

These spaces are constructed from "commonly used neuron, head and embedding dimensions" — that is, the paper constrains the search to dimensions that are practical for implementation (e.g., 32 heads rather than 31, embedding dimensions that are multiples of reasonable chunk sizes for tensor parallelism). This pragmatic constraint keeps the search tractable.

**Feasibility filtering:** From all combinations in the search space, only architectures whose total parameter count falls within ±5% of the target (8B or 4B) are retained. This eliminates most combinations: the paper reports obtaining only 15 feasible candidates for the 8B target and 18 for the 4B target (Table 19 lists all 15 for 8B). The ±5% tolerance is tight enough to ensure the resulting model is actually in the desired size class but loose enough to capture meaningful architectural variation.

**Lightweight retraining for architecture evaluation:** Each feasible candidate is pruned from the parent model and retrained for a small number of steps (400 steps, ~1.8B tokens, measured precisely as 400 training steps with the Nemotron-4 training configuration of batch size 1152 and sequence length such that 400 steps corresponds to ~1.8B tokens). The validation loss during this lightweight retraining serves as the fitness metric for architecture selection.

**Ranking stabilization (Figure 9):** The paper observes that the RELATIVE ranking of candidate architectures changes significantly during the first ~300 steps of lightweight retraining, then stabilizes. For example, candidate "id2" might initially have lower validation loss than candidate "id5," but after 300 steps, the order flips. This is the same phenomenon observed in the width vs. depth pruning comparison (Figure 6) — post-pruning zero-shot performance does not predict post-retraining performance. The 400-step lightweight retraining phase is long enough for these rankings to stabilize, enabling selection of the truly best architecture without committing to full retraining of all candidates.

**Why not parameter-efficient fine-tuning for evaluation:** The paper explicitly mentions that "parameter-efficient fine-tuning techniques such as LoRA can also be applied at this stage" but leaves their exploration to future work. LoRA would further reduce the cost of evaluating each candidate during NAS, potentially enabling a larger search space or more candidates. The decision to use full fine-tuning for the lightweight evaluation phase is conservative — it ensures the evaluation metric (full fine-tuning validation loss) matches the final retraining procedure, avoiding any mismatch that LoRA-based evaluation might introduce.

**Why genetic search or Bayesian optimization weren't needed:** The paper states: "while it's possible to further reduce the search space size using strategies such as genetic search and/or Bayesian optimization, we found that sticking to commonly used neuron, head and embedding dimensions, along with a reasonably narrow target parameter range (less than 1 billion) was sufficient to obtain tractable solution sets (less than 20 candidates)." This is an important practical insight: for the specific task of compressing a known architecture to a nearby smaller size, exhaustive enumeration over a discretized space is feasible and simpler than more sophisticated search algorithms. The "less than 20 candidates" figure means that lightweight retraining of all candidates costs at most `$20 \times 1.8\text{B} = 36\text{B}$` tokens, which is negligible compared to the trillions of tokens used for training from scratch.

**Architectures selected (Table 5):** The search produced:

- **MINITRON 8B**: 32 layers, 48 attention heads, MLP hidden dimension of 16384 (expansion factor 4 with `$d_{model}=4096$`), embedding dimension 4096, total 8.27B parameters (6.2B non-embedding). Notably, this preserves the full 32-layer depth from the parent, concentrates parameter reduction in the width dimensions (embedding from 6144→4096, a 33% reduction), but KEEPS 48 heads (same as parent) — the MLP expansion factor of 4 with reduced `$d_{model}$` produces an MLP hidden dimension of 16384 (vs. parent's 24576), which is where most of the parameter reduction occurs.

- **MINITRON 4B**: 32 layers, 24 attention heads, MLP hidden dimension of 9216 (expansion factor 3 with `$d_{model}=3072$`), embedding dimension 3072, total 4.19B parameters (2.6B non-embedding). Again, full 32-layer depth is preserved, with aggressive width reduction: embedding from 6144→3072 (50% reduction), heads from 48→24 (50% reduction), MLP hidden from 24576→9216 (62.5% reduction).

A striking pattern: the search consistently chooses to **preserve all 32 layers** and concentrate compression in width dimensions. This is consistent with Best Practice #4 ("Prefer width pruning over depth") and with the experimental finding in Table 10 that width-pruned models outperform depth-pruned ones.

---

#### Knowledge Distillation Retraining

The retraining phase is where the pruned model recovers accuracy lost during pruning. The paper's key innovation here is using **knowledge distillation from the uncompressed teacher model** rather than conventional next-token prediction training (Section 3).

**Why distillation is necessary:** Table 11 provides the clearest evidence. Under iso-compute conditions (comparing approaches with equivalent FLOPs):

- **4B-Random-Init** (train a 4B model from scratch): 24.36% MMLU, 46.22% HellaSwag at 150B tokens
- **4B-Pruned** (prune 15B→4B, retrain with conventional training): 24.57% MMLU, 50.85% HellaSwag at 150B tokens — essentially indistinguishable from random initialization on MMLU
- **4B-Pruned-Distill** (prune 15B→4B, retrain with distillation): **37.81% MMLU**, 51.04% HellaSwag at 100B tokens — a massive 13.5 percentage point MMLU improvement using fewer tokens

Conventional retraining after aggressive pruning is essentially equivalent to training from scratch — the pruning advantage is almost entirely lost. Distillation preserves the knowledge transfer from teacher to student, enabling data-efficient accuracy recovery.

**The teacher-student setup:** The uncompressed Nemotron-4 15B model serves as the teacher; the pruned model (with varying architecture depending on the compression target) serves as the student. The teacher is frozen during distillation — only the student's weights are updated. The teacher processes the same training data as the student in parallel, providing target distributions that the student learns to match.

**Logit-based distillation loss:** The core distillation signal comes from matching the output probability distributions of teacher and student. For a given token `$x_i$`, the probability distribution over the vocabulary is:

> $$p(x_i, \tau) = \frac{\exp(x_i / \tau)}{\sum_{j=1}^{|\mathcal{V}|} \exp(x_j / \tau)}$$

where `$x_i$` is the unnormalized logit for token `$i$`, `$\tau$` is the softmax temperature parameter, and `$|\mathcal{V}| = 256,000$` is the vocabulary size (the Nemotron tokenizer vocabulary).

**What it computes:** Standard temperature-scaled softmax. The logits `$x_i$` are divided by `$\tau$` before exponentiation. When `$\tau = 1$`, this is the standard softmax. When `$\tau > 1$`, the distribution is softened (moves toward uniform), making low-probability but plausible tokens more visible to the student — this is the classic Hinton distillation setting. When `$\tau < 1$`, the distribution is sharpened (peaks more strongly at the mode). The paper experiments with `$\tau \in \{0.1, 0.5, 1.0, 3.0\}$` (Appendix A.3) and finds **`$\tau = 1.0$` works best**, in contrast to vision models where `$\tau > 1$` is standard. The reasoning: LLM output distributions already have high entropy (many plausible tokens), so softening isn't needed to expose secondary information; `$\tau = 1.0$` preserves the teacher's natural uncertainty calibration.

**The logit distillation loss across a sequence:**

> $$\mathcal{L}_{\text{logits}} = \frac{1}{\ell} \sum_{k=1}^{\ell} \text{Loss}\left(p_k^t(x, \tau), p_k^s(x, \tau)\right)$$

where `$\ell$` is the sequence length, `$p_k^t(x, \tau)$` and `$p_k^s(x, \tau)$` are the teacher and student probability distributions at the `$k$`-th token position, respectively, and `$\text{Loss}(\cdot)$` is a divergence or distance measure to be specified. The loss is averaged over all token positions in the sequence, treating each position as an independent distillation target.

**What it computes:** For each token position in each training sequence, the teacher produces a probability distribution over the 256,000-token vocabulary (the "correct" distribution according to the larger trained model), the student produces its own distribution, and the divergence between them is computed and averaged. This differs fundamentally from conventional training, where the student would be trained to match the one-hot ground-truth next token — distillation provides a RICHER signal with information about the relative plausibility of all tokens, not just the correct one.

**Why this form:** Position-level averaging treats each token's prediction as equally important. The teacher's distribution encodes both what the correct token is (the ground-truth label) AND the relative likelihood of alternative tokens — this "dark knowledge" (Hinton et al., 2015) contains information about token similarity, synonym relationships, and grammatical constraints that is absent from one-hot labels. For example, if the correct next token is "happy," the teacher might assign non-trivial probability to "glad," "pleased," and "joyful," teaching the student about semantic similarity even when those tokens aren't the target.

**Choice of divergence function (Table 15-16):** The paper experiments with four divergence measures for `$\mathcal{L}_{\text{logits}}$`:

- **Kullback-Leibler Divergence (KLD)**: `$\sum_j p_j^t \log(p_j^t / p_j^s)$` — the standard distillation loss, measuring how much information is lost when using the student's distribution to approximate the teacher's
- **Reverse KLD (R-KLD)**: `$\sum_j p_j^s \log(p_j^s / p_j^t)$` — the KL divergence with student and teacher swapped, recently shown to perform better than KLD in SFT/instruction-following settings (Gu et al., 2024; Ko et al., 2024)
- **Mean Squared Error (MSE)**: `$\sum_j (p_j^t - p_j^s)^2$` — direct L2 distance between probability vectors
- **Cosine similarity**: `$1 - \frac{\sum_j p_j^t p_j^s}{\sqrt{\sum_j (p_j^t)^2} \sqrt{\sum_j (p_j^s)^2}}$` — angular distance between probability vectors

Table 15 (using a previous generation Nemotron-3 8B model) shows KLD consistently outperforms all alternatives: `$\mathcal{L}_{\text{logits}}(\text{KLD})$` alone achieves 2.107 LM loss, while `$\mathcal{L}_{\text{logits}}(\text{RKLD})$` achieves 2.140, and `$\mathcal{L}_{\text{logits}}(\text{MSE})$` achieves 2.144. Table 16 confirms this on MINITRON 8B depth-pruned: KLD achieves 2.155 vs. R-KLD at 2.665. The paper adopts KLD as the standard (Best Practice #5).

**Why KLD works better than R-KLD in this setting:** R-KLD encourages the student to place high probability wherever the student already has high probability (mode-seeking behavior — it penalizes the student for assigning probability mass to tokens the student thinks are unlikely, even if the teacher disagrees). KLD encourages the student to place high probability wherever the TEACHER places high probability (mean-seeking behavior — it penalizes the student for missing tokens the teacher considers important). For base model pretraining, where the student needs to learn the teacher's full distribution including secondary candidates, mean-seeking behavior is more appropriate. R-KLD may work better in SFT settings where the distribution is sharper (only a few good responses) and mode-seeking prevents the student from being pulled toward low-quality modes.

**Top-K filtering experiment:** The paper experimented with retaining only the top-K teacher logits (and corresponding student logits) before computing `$\mathcal{L}_{\text{logits}}$`, analogous to top-K sampling during inference. A small K (≤100) causes significant accuracy degradation because it discards useful information from the long tail of the vocabulary distribution. Larger K values are equivalent to not using top-K. The conclusion: for LLM distillation with a large vocabulary (256K tokens), the full distribution matters — there is no benefit to top-K filtering.

**Intermediate state distillation loss:** In addition to logit-level distillation, the paper explores distilling intermediate hidden states from the teacher to the student. The loss across a sequence of transformer-specific hidden states is defined as:

> $$\mathcal{L}_{\text{is}} = \frac{1}{\ell} \sum_{k \in \mathcal{H}} \sum_{i=1}^{\ell} \text{Loss}_k(\mathbf{h}_{ki}^t, \mathbf{h}_{ki}^s)$$

where `$\mathcal{H}$` is the set of chosen intermediate states (e.g., `$\{\text{embedding}, \text{encoder block outputs}, \text{attention outputs}\}$`), `$\mathbf{h}_{ki}^t$` and `$\mathbf{h}_{ki}^s$` are the teacher and student hidden states of type `$k$` at token position `$i$`, respectively, and `$\ell$` is the sequence length. The outer sum runs over all intermediate state types `$k$` in the set `$\mathcal{H}$`, and the inner sum averages loss across all token positions.

**What it computes:** For each type of intermediate state that the paper chooses to distill (embeddings, attention outputs, MLP inputs, encoder block outputs), the teacher and student produce hidden state tensors during their respective forward passes. A loss function (typically cosine similarity) measures how well the student's hidden states match the teacher's, and the average across token positions produces a scalar loss. This provides supervision at intermediate layers, not just at the final output.

**Why intermediate states and why only certain ones:** Distilling intermediate states provides the student with a richer training signal — it must match the teacher's internal representations, not just the final token distribution. This is particularly important when depth is reduced significantly (the student has fewer layers than the teacher), because the student must learn compressed representations that nonetheless encode the same information as the teacher's deeper representations. However, the paper finds that intermediate state distillation provides minimal benefit when depth is NOT reduced significantly (Section 4.3, Best Practices #6-7).

**Handling dimension mismatch:** The teacher and student may have different hidden dimensions (e.g., teacher `$d_{model}=6144$`, student `$d_{model}=4096$`). To handle this, a **shared linear transformation** is learned during distillation to upscale the student hidden state to the teacher hidden state dimension before computing the loss. This is a `$d_{model}^{student} \times d_{model}^{teacher}$` projection matrix trained along with the student's primary weights.

**Layer mapping for depth mismatch:** When the student has fewer layers than the teacher (e.g., a 16-layer student distilled from a 32-layer teacher), the loss must define which student layers correspond to which teacher layers. This is shown in Figure 4: a student with `$N$` layers is mapped to a teacher with `$M$` layers, with specific student block `$S$` aligned to teacher block `$T$`. The paper explores several mapping strategies through ablation (Table 17), using notation like `$\text{Lo}(29:13)$` to mean the student layer 13 is mapped to teacher layer 29 for the encoder block output loss `$\mathcal{L}_o$`.

**Loss components explored (Appendix A.4):** The paper experiments with various combinations of `$\mathcal{L}_{\text{is}}$`:

1. **`$\mathcal{L}_o$` (encoder block output loss):** Distilling the output of each transformer encoder block. Using `$\mathcal{L}_o$` alone with `$\mathcal{L}_{\text{logits}}$` provides a boost, reducing LM loss from 2.155 (logits only) to 2.145 (logits + `$\mathcal{L}_o(29:13)$`) for a depth-pruned model.

2. **`$\mathcal{L}_{\text{emb}}$` (word embeddings loss):** Distilling the token embedding layer output. Adding `$\mathcal{L}_{\text{emb}}$` improves accuracy. The combination `$\mathcal{L}_{\text{logits}} + \mathcal{L}_o + \mathcal{L}_{\text{emb}}$` produces the best results in Table 17 for depth-pruned models.

3. **`$\mathcal{L}_{\text{att}}$` (attention relation loss):** Distilling attention patterns derived from query, key, and value states (following MiniLMv2; Wang et al., 2021). The paper reports this "does not show any improvement" — attention pattern matching doesn't help the student learn better than logit + output state matching.

4. **`$\mathcal{L}_i$` (MLP input loss):** Distilling the input to MLP layers. The paper reports this "makes no difference" — the signal at MLP input is largely redundant with the encoder block output signal.

5. **Layer mapping granularity:** Mapping multiple student layers to a single teacher layer, or vice versa, "either results in no improvement or accuracy degradation" (following findings from Lu et al., 2022).

6. **Cosine similarity for intermediate state loss:** Among loss functions for `$\mathcal{L}_{\text{is}}$`, cosine similarity performs best. This makes sense: cosine similarity measures directional alignment of representations, which is invariant to magnitude scaling that the upscaling projection might introduce.

**Why this detailed exploration matters:** The paper's findings translate into actionable guidance: for depth-pruned models (where the student has significantly fewer layers than the teacher), use logit + embedding + encoder block output distillation. For width-pruned models where depth is preserved (as in both MINITRON 8B and 4B), logit-only distillation is sufficient (Table 18: adding `$\mathcal{L}_{\text{is}}$` to `$\mathcal{L}_{\text{logits}}$` for MINITRON 8B yields 58.0% MMLU vs. 58.3% for logits-only — essentially identical). This is Best Practice #7: "Use logit-only distillation when depth isn't reduced significantly."

**Total loss formulation:**

> $$\mathcal{L} = \mathcal{L}_{\text{CLM}} + \mathcal{L}_{\text{logits}} + \alpha \cdot \mathcal{L}_{\text{is}}$$

where `$\mathcal{L}_{\text{CLM}}$` is the conventional causal language modeling loss (cross-entropy against ground-truth next tokens), `$\mathcal{L}_{\text{logits}}$` is the logit distillation loss, `$\mathcal{L}_{\text{is}}$` is the intermediate state distillation loss (set of chosen components), and `$\alpha$` is a weighting coefficient.

**What it computes:** The total loss is a weighted sum of three terms: standard next-token prediction loss (ensuring the student learns to predict correct tokens), logit-level distillation loss (ensuring the student matches the teacher's full output distribution), and intermediate state distillation loss (ensuring internal representation alignment). The relative weighting `$\alpha$` controls how much importance is placed on matching hidden states vs. matching output distributions.

**Dynamic `$\alpha$` computation — a key practical insight:** The paper finds that "the magnitudes of `$\mathcal{L}_{\text{logits}}$` and `$\mathcal{L}_{\text{is}}$` differ significantly" — the intermediate state loss is typically much larger in magnitude than the logit loss because hidden states are high-dimensional and their cosine distances don't naturally scale to match KL divergence magnitudes. Using a fixed `$\alpha$` would require careful per-model tuning. Instead, the paper computes `$\alpha$` dynamically as:

> $$\alpha = \frac{\mathcal{L}_{\text{logits}}}{\mathcal{L}_{\text{is}}}$$

at each training step. This normalizes the magnitude of `$\alpha \cdot \mathcal{L}_{\text{is}}$` to match `$\mathcal{L}_{\text{logits}}$`, ensuring both terms contribute equally to the gradient regardless of their natural scale differences. The paper reports this "achieves better results compared to using a constant."

**Why this dynamic weighting helps:** In multi-task learning, loss weighting is notoriously sensitive — an improperly weighted auxiliary loss can either dominate training (causing the model to optimize representation matching at the expense of prediction accuracy) or be ignored entirely (providing no benefit). The dynamic `$\alpha$` ensures the intermediate state loss contributes a gradient of comparable magnitude to the logit loss at every step, effectively making the two objectives learn at similar rates. If `$\mathcal{L}_{\text{is}}$` naturally decreases faster during training (as hidden states become easier to match), `$\alpha$` correspondingly increases to maintain the balance.

**Logit-only distillation outperforms combined loss (Table 15):** A critical finding: `$\mathcal{L}_{\text{logits}}$` alone achieves 2.107 LM loss, outperforming `$\mathcal{L}_{\text{CLM}} + \mathcal{L}_{\text{logits}}(\text{KLD})$` at 2.117. Adding the conventional language modeling loss slightly REDUCES performance. The paper speculates that this is because the teacher's logit distribution already encodes the ground-truth information (the teacher knows which token is correct and places high probability on it), and the conventional loss may introduce conflicting gradients by forcing the student to commit to a single token rather than learning the teacher's full distribution.

---

#### Iterative Compression for Aggressive Size Reduction

When the target compression ratio is large (73.3% weight reduction from 15B to 4B), the paper finds that **one-shot pruning causes significant capability loss** that even distillation retraining cannot fully recover (Section 4.3, Best Practice #8). The solution is **iterative compression**: compress through an intermediate size, retrain that intermediate model, and then compress again to the final target.

**Procedure:** For MINITRON 4B:
1. Prune Nemotron-4 15B → MINITRON 8B (~46% weight reduction), retrain with distillation using Nemotron-4 15B as teacher (~94B tokens)
2. Prune MINITRON 8B → MINITRON 4B (~50% weight reduction), retrain with distillation using **Nemotron-4 15B as teacher** (~100B tokens, per Table 11 notation for the 4B-Pruned-Distill rows)

**Why the 15B model is used as teacher for the 8B→4B step:** The paper explicitly notes that "during the final retraining step, we observe that using Nemotron-4 15B as the teacher achieves superior results compared to using MINITRON 8B." Even though the student is being pruned from the 8B model, the ORIGINAL 15B teacher provides a richer distillation signal than the distilled 8B model would — the 15B model has full pretraining quality, while the 8B model has already lost some information through its own compression and distillation.

**Quantitative evidence (Table 11, last two rows):**

| Approach | Tokens | MMLU | HellaSwag |
|----------|--------|------|-----------|
| One-shot: 15B→4B + distill | 100B | 37.81% | 51.04% |
| Iterative: 15B→8B→4B + distill | 100B | **42.45%** | **52.04%** |

The iterative approach achieves 4.64 percentage points higher MMLU (a 12% relative improvement) and 1.0 point higher HellaSwag. The total token budget is roughly equivalent (100B each), but the iterative approach interleaves the tokens with an intermediate architecture change, which helps preserve knowledge through the aggressive size reduction.

**Why one-shot aggressive pruning fails:** When 73% of weights are removed simultaneously, the model loses both explicit knowledge (the removed weights contained learned patterns) and structural knowledge (the computational pathways that remained are disrupted by missing connections). Distillation alone cannot reconstruct this lost information from limited data because the student has no intermediate stable configuration to build from. Iterative compression provides a "stepping stone": the 8B model serves as a stable, partially-compressed intermediate that retains more of the 15B model's knowledge, and the second compression step starts from this better initialization.

**Interaction with architecture search:** The iterative strategy also constrains the architecture search. Instead of searching directly from 15B to an arbitrary 4B architecture, the search first finds the best 8B architecture (from the 15 candidates in Table 19), trains it fully, then searches for the best 4B architecture starting from that 8B model. This reduces the combinatorial complexity — the search for 4B architectures doesn't need to consider the full space of possible 15B→4B paths, but only 8B→4B paths conditioned on the chosen 8B architecture.

---

#### Single-Phase vs. Multi-Phase Retraining Data Strategy

Modern LLM pretraining often uses multi-phase strategies: an initial phase on broad web data followed by a continued training phase on higher-quality, curated data (Parmar et al., 2024a, 2024b; Nemotron-4 15B's own training pipeline). The paper explores which checkpoint to prune and which data to use for retraining (Section 4.3, Best Practice #10).

**Two strategies are compared (Table 20):**

1. **Phase1 + Phase2:** Prune the phase-1 (web data) checkpoint, retrain with a mixture of phase-1 and phase-2 data (total 113B tokens). Results: 54.7% MMLU, 80.3% HellaSwag, 77.2% PIQA, 25.6% HumanEval.

2. **Phase2 only:** Prune the phase-2 (continued training) checkpoint (the final trained model), retrain with phase-2 data only (total 94B tokens). Results: **61.9% MMLU**, 80.1% HellaSwag, 76.7% PIQA, **30.5% HumanEval**.

**Why phase-2 only is better:** The phase-2 checkpoint contains higher-quality representations due to continued training on curated data — it's a better starting point for distillation. Retraining solely on phase-2 data (the cleaner, higher-quality data) is sufficient for the student to recover the teacher's phase-2 capabilities. Trying to replicate both training phases during retraining (strategy 1) is unnecessary and uses more tokens for worse results. The paper also notes this suggests "for further aligned models, it may suffice to prune the aligned model and retrain with a portion of the alignment dataset" — that is, for instruction-tuned models, prune the instruction-tuned checkpoint and retrain with instruction data, rather than pruning the base model and re-running instruction tuning.

**Practical implication:** This finding simplifies the retraining pipeline. Instead of needing access to the full pretraining data mix, teams can prune the best available checkpoint and retrain using only the data from the final training phase. This is particularly valuable when earlier-phase data is proprietary, unavailable, or too large to reprocess, but the final-phase data is manageable (as is often the case when the final phase uses a smaller, curated dataset).

---

#### Summary of Design Choices and Their Justifications

- **Activation-based importance over gradient-based:** avoids memory and compute cost of backward passes at 15B scale; validated by showing zero-shot rankings persist after retraining (Figure 5).

- **(batch=L2, seq=mean) aggregation over alternatives:** L2 batch aggregation prevents outlier samples from dominating; mean sequence aggregation treats all token positions equally; empirically best in Table 13.

- **Single-shot over iterative importance:** Table 14 shows all iteration counts converge to identical retrained loss; iterative estimation adds unnecessary compute cost.

- **Width pruning over depth:** Table 10 and Figure 6 show width-pruned models achieve lower retrained loss despite having fewer parameters; the discovery that depth pruning looks better initially but is overtaken after ~200 retraining steps is a key empirical finding.

- **KLD over R-KLD, MSE, cosine for logit distillation:** KLD's mean-seeking behavior is better suited for base model pretraining where the full teacher distribution matters; Tables 15-16 consistently show KLD superiority.

- **Logit-only distillation when depth is preserved:** Table 18 shows adding intermediate state distillation provides no benefit for MINITRON 8B (same depth as teacher); simplifies retraining significantly.

- **Logit + embedding + encoder block output distillation when depth is reduced:** Table 17 shows this combination reduces LM loss for depth-pruned models; intermediate state signals help the student learn compressed representations.

- **Iterative pruning for large compression ratios:** Table 11 shows 12% MMLU improvement over one-shot for 15B→4B; intermediate retraining preserves knowledge that would be lost in aggressive one-shot pruning.

- **Phase-2-only retraining:** Table 20 shows better results with fewer tokens; pruning the highest-quality checkpoint and retraining on the highest-quality data is sufficient.

- **Lightweight retraining for NAS ranking stabilization:** Figure 9 shows rankings converge after ~300 steps; enables confident architecture selection without full retraining of all candidates.

- **Dynamic `$\alpha = \mathcal{L}_{\text{logits}} / \mathcal{L}_{\text{is}}$` over fixed weighting:** automatically balances loss magnitudes; avoids manual per-model tuning and ensures both terms contribute equally to gradients.

## 4. Key Insights and Innovations

### Innovation 1: Activation-Only Importance Estimation as a Viable Alternative to Gradient-Based Methods at LLM Scale

The paper's most disruptive conceptual move is demonstrating that **you do not need gradients to prune LLMs effectively** — a claim that, if it generalizes, overturns the dominant paradigm in structured pruning research and enables compression at scales where gradient computation is prohibitive.

**What the field did before:** The established approach for identifying which components to prune in neural networks — inherited from pre-LLM computer vision and carried into LLM pruning by methods like LLM-Pruner (Ma et al., 2023) and SliceGPT (Ashkboos et al., 2024) — relies on **gradient or Hessian information**. Taylor expansion-based importance metrics approximate the change in loss when a component is removed by computing the product of the component's activation and its gradient (first-order Taylor) or including second-order Hessian terms. Learnable mask approaches go further, using an Augmented Lagrangian formulation that requires gradient-based optimization to discover optimal pruning masks. These methods are theoretically grounded — the Taylor expansion is a principled local approximation of the loss landscape — but they incur the full cost of backpropagation through a 15-billion-parameter model, which is memory-intensive (storing gradients doubles peak memory) and compute-intensive (backward passes are roughly 2× the cost of forward passes).

**What this paper shows instead:** The paper proposes a purely activation-based strategy — compute importance from forward-pass activations on a tiny calibration set (1024 samples) with no backward passes — and validates that this zero-order approximation works. The critical evidence is not just that activation-based importance produces reasonable zero-shot pruning decisions (Table 13, where the best aggregation metric achieves 7.18 LM loss vs. 10.55 for the worst), but that **post-retraining performance vindicates the zero-shot rankings** (Figure 5: the (L2, mean) metric maintains its advantage over (L2, L2) throughout 400 steps of retraining, without crossover). This is non-trivial: a metric that ranks components well for the untrained, immediately-post-pruning model might fail after retraining redistributes function across surviving components. The fact that rankings persist means the metric is measuring something structural about component importance — not just which components happen to be active on the calibration data.

**Why this is a fundamental shift, not just an engineering convenience:** The conceptual implication goes beyond compute savings. Gradient-based importance measures a component's **marginal contribution to the loss at the current parameter setting** — it is inherently local and can miss components that are important for representational capacity even if they have small instantaneous gradients (e.g., a head that currently produces near-zero output but is structurally positioned to represent an important feature that other heads rely on). Activation-based importance measures a component's **actual contribution to the forward computation on real data** — it captures what the component does, not what it could do if perturbed. The paper's results suggest that for well-trained LLMs at convergence, these two notions largely coincide for pruning purposes, making the more expensive gradient information redundant. This is an empirical finding about the nature of trained transformer representations — components that matter produce visible activation signatures — rather than a theoretical guarantee.

**The aggregation function experiments (Table 13) as a diagnostic contribution:** Beyond the headline result, the paper's systematic comparison of 9 batch×sequence aggregation combinations reveals that the choice of metric matters enormously (LM loss ranging from 7.18 to 10.55, a 47% relative difference) and that the wrong choice can make activation-based pruning appear to fail. This diagnostic insight — that aggregation function selection is a first-order design choice, not a hyperparameter detail — is independently valuable for practitioners. The finding that variance-based aggregation performs worst (10.55 for var/var) while norm-based aggregation performs best (7.18 for L2/mean) suggests that magnitude, not dispersion, is the relevant signal for transformer component importance — a characterization of how redundancy manifests in these models.

**Connection to the iterative importance negative result (Table 14):** The finding that iterative importance estimation provides no benefit — all iteration counts converge to identical 1.92 LM loss after retraining — reinforces the sufficiency of single-shot activation-based estimation. If importance estimates were unreliable or highly sensitive to the specific pruning level, iterative re-estimation would improve results. The fact that it doesn't implies the single-shot estimates are robust and that retraining is powerful enough to compensate for any suboptimality in the initial pruning decisions, making more sophisticated estimation unnecessary.

---

### Innovation 2: The Discovery That Width Pruning Systematically Outperforms Depth Pruning — But Only After Retraining

One of the paper's most counterintuitive empirical findings is a **dynamic reversal in the relative performance of width versus depth pruning**: immediately after pruning, a combined depth+width pruned model achieves lower loss than a pure width-pruned model, but after ~200 steps of retraining, the ranking flips and pure width pruning pulls ahead. This temporal dynamic is invisible to any analysis that only evaluates zero-shot pruned models, and it has significant implications for how pruning strategies should be evaluated.

**The dominant assumption before this work:** Depth pruning has received substantial recent attention (ShortGPT, LaCo, Shortened LLaMA) with compelling results — entire layers can be removed with minimal perplexity degradation, suggesting significant depth redundancy in LLMs. The natural hypothesis would be that combining depth AND width pruning should outperform either alone, since it exploits redundancy on both dimensions simultaneously. The paper's Table 10 initially supports this: the combined depth+width pruned model (7.91B parameters, removing 4 layers plus width pruning) achieves lower post-pruning loss than the pure width-pruned model (7.74B parameters, all 32 layers preserved). This is exactly what the dominant assumption would predict.

**What actually happens (Figure 6, Table 10):** During retraining, the width-pruned model's loss drops faster, and by ~200 steps (~0.8B tokens), it overtakes the combined pruning model. The final gap after 400 steps is small but consistent (2.049 vs. 2.062) — and critically, the width-pruned model achieves this with fewer parameters (7.74B vs. 7.91B). The depth pruning operation that initially looked beneficial turns out to be harmful in the retrained regime.

**Why this is a conceptual advance, not just an empirical curiosity:** The finding suggests that **depth and width redundancy are qualitatively different phenomena**. Layer removal eliminates entire computational stages from the model's processing pipeline — the model loses not just capacity but *depth of computation*, which may be important for the multi-step reasoning that retraining tries to recover. Width pruning reduces capacity within each computational stage while preserving the full pipeline depth, which provides a better structural prior for the retraining process to work with. This is not obvious from any existing theory of neural network redundancy — it is an empirical discovery about how information is organized across the depth and width axes of trained transformers.

**Why the zero-shot vs. retrained ranking discrepancy matters:** The paper explicitly notes this dynamic reversal as motivation for Best Practice #9 (lightweight retraining for architecture evaluation), but the deeper insight is methodological: **evaluating pruning strategies based on zero-shot performance is unreliable**. The community's standard practice of reporting perplexity immediately after pruning (which is cheaper than full retraining) can produce misleading conclusions about which strategies are actually better. Any future pruning research that does not include a retraining phase should be interpreted with this caveat. The paper's lightweight retraining protocol (~1.8B tokens, just enough for rankings to stabilize as shown in Figure 9) emerges as a pragmatic resolution: not as expensive as full retraining, but enough to avoid being misled by zero-shot rankings.

**The constraint on model scale (≤15B) in Best Practice #4 is a hedging acknowledgment:** The paper explicitly qualifies "prefer width pruning over depth for the model scales we consider (≤15B)," leaving open the possibility that at larger scales (70B, 175B, 405B), depth redundancy might become so extreme that depth pruning becomes more favorable. This is not tested, but the qualification is intellectually honest — it frames the finding as scale-dependent rather than universal.

---

### Innovation 3: Knowledge Distillation as a Data-Efficient Alternative to Conventional Retraining — and the Discovery That Logit-Only Distillation Suffices

The paper's third major conceptual contribution is establishing that **knowledge distillation from the uncompressed teacher is categorically more data-efficient than conventional retraining after pruning**, and that — contrary to what one might expect from the knowledge distillation literature — **logit-level distillation alone is sufficient when model depth is preserved**, making intermediate state distillation an unnecessary complication in the common case.

**What the field did before:** Prior structured pruning work on LLMs — including depth-pruning methods (ShortGPT, LaCo, Shortened LLaMA) and width-pruning methods (LLM-Pruner, SliceGPT, Sheared LLaMA) — used **conventional next-token prediction training** for post-pruning accuracy recovery. The teacher model, when it existed, was used only as the source of weights to prune, not as a supervision signal during retraining. This is understandable: conventional training with ground-truth labels is the default, and distillation adds engineering complexity (managing two models during training, designing loss functions, tuning temperatures). But the paper shows this default is substantially suboptimal.

**The magnitude of the distillation advantage (Table 11):** Under iso-compute conditions, a conventionally retrained pruned 4B model achieves 24.57% MMLU — essentially indistinguishable from a randomly initialized 4B model trained from scratch (24.36%). The same pruning procedure followed by distillation-based retraining achieves 37.81% MMLU, using fewer tokens (100B vs. 150B). This is a **13.5 percentage point gap** on a benchmark where random performance is 25% (4-choice multiple choice). The gap isn't marginal — conventional retraining after aggressive (73% weight reduction) pruning is essentially equivalent to throwing away the parent model's knowledge and starting over, while distillation preserves the knowledge transfer.

**Why this is a fundamental finding, not an incremental improvement:** The result reframes what pruning accomplishes. If conventional retraining is equivalent to training from scratch, then pruning without distillation is merely **architecture selection** — you've chosen a good architecture by imitating the parent's shape, but you haven't transferred any of the parent's learned function. Distillation is what makes pruning *knowledge transfer* rather than just *architecture design*. This has implications for how the field thinks about the relationship between model compression and model training: compression without distillation is a weaker primitive than previously assumed.

**Why the sufficiency of logit-only distillation for width-pruned models is surprising:** The knowledge distillation literature — both in computer vision and in NLP (Lu et al., 2022; MiniLMv2) — has consistently found that distilling intermediate representations (hidden states, attention patterns, layer outputs) provides additional benefits beyond logit distillation alone. The intuition is that intermediate states provide richer supervision: they tell the student not just WHAT to output, but HOW to represent information internally. The paper's finding that adding intermediate state distillation to logit distillation for MINITRON 8B provides no benefit (Table 18: 58.0% MMLU with intermediate states vs. 58.3% without) contradicts this intuition for the specific case of width-pruned models where depth is preserved.

**A plausible explanation the paper doesn't fully develop:** When depth is preserved (32 layers in both teacher and student), the student's computational pathway has the same number of processing stages as the teacher. It can learn to match the teacher's output distribution by developing its own internal representations that, while dimensionally smaller, are functionally equivalent. The logit-level supervision provides enough information for the student to discover these compressed representations on its own — intermediate state supervision might even be harmful if it forces the student to match representations that are inherently tied to the teacher's larger hidden dimension. When depth IS reduced significantly, the student has fewer processing stages and may need explicit guidance on how to compress multi-stage computation into fewer stages — hence intermediate state distillation becomes beneficial (Table 17, where it reduces LM loss from 2.155 to 2.141 for depth-pruned models). This pattern — intermediate state distillation is helpful for depth mismatch but unnecessary for width mismatch — is a novel characterization of when and why representation-level distillation matters.

---

### Innovation 4: The Demonstration That Pruning + Distillation Can Produce Models Competitive With Independently Trained Models — Establishing a Viable Third Production Pathway

The paper's headline practical result — MINITRON 8B matches community models trained from scratch with 40× fewer tokens, MINITRON 4B outperforms similarly-sized models — is legitimized as an intellectual contribution because it **establishes pruning + distillation as a legitimate third pathway for LLM production**, alongside training from scratch and continued pretraining. This is not just a strong result; it is a demonstration that a previously underexplored production strategy is viable at competitive quality levels.

**The production landscape before this work:** LLM providers had essentially two options for producing models of different sizes: (1) train each size from scratch on the full pretraining corpus (the LLaMA approach), or (2) train one model and continue pretraining with architectural modifications to produce variants (e.g., extending context length, domain adaptation). Pruning was studied extensively in the academic literature but had not been demonstrated to produce models that could compete with independently trained models on standard benchmarks — the results from LLM-Pruner, SliceGPT, and others (reproduced in Table 4) showed substantial degradation: LLM-Pruner's 9.8B model achieves only 25.2% MMLU, SliceGPT's 9.9B model achieves 37.1%, and LaCo's 9.8B model achieves 45.9%, compared to MINITRON 8B's 63.8%. The gap between pruned models and trained-from-scratch models was large enough that pruning was not a serious production option — it was a research technique for studying redundancy, not a deployment strategy.

**What this paper demonstrates that changes the calculus:** Table 2 shows MINITRON 8B (8.3B parameters, 94B training tokens) achieving 63.8% MMLU — comparable to Llama-3 8B (65.3%, trained on >15T tokens), Mistral 7B (64.1%, 8T tokens), and Gemma 8.5B (64%, 6T tokens). This is not an existence proof that pruning *can* work; it is a demonstration that pruning *competes* at the frontier. The fact that MINITRON 8B uses 40× fewer tokens than Nemotron-3 8B (94B vs. 3.8T) while outperforming it on nearly every benchmark — 63.8% vs. 54.7% MMLU, 51.3% vs. 24.0% GSM8K, 31.6% vs. 20.7% HumanEval — shows that the token savings are not coming at the cost of quality. The model is both cheaper AND better.

**Why the instruction-tuned results (Tables 6-9) are conceptually important beyond raw scores:** The paper goes beyond base model evaluations to show that MINITRON 4B-instruct performs well on instruction-following (IFEval, 73.01% prompt-level accuracy), roleplay (MT-Bench, 6.46 score), RAG (ChatRAG-Bench, 41.11), and function calling (BFCL, 53.09). These are deployment-facing benchmarks that matter for real-world use. The fact that a model produced through pruning + distillation can be instruction-tuned to competitive quality demonstrates that the pruning process does not introduce artifacts or limitations that prevent downstream adaptation — the compressed model is a drop-in replacement for a from-scratch model in the standard LLM production pipeline.

**The critical caveat about capability boundaries:** The paper does not explicitly state this, but the results in Table 2-3 implicitly establish a boundary condition: MINITRON models work because their parent (Nemotron-4 15B) is a strong model. The compression process **preserves and reorganizes existing capability** but cannot create new capabilities that the parent lacks. If the 15B model performed poorly on a particular task, its compressed descendants would inherit that weakness. This is analogous to the test-time compute paper's finding that compute cannot help on problems outside the base model's capability range — and it establishes that pruning-based model production is appropriate when the largest sibling is sufficiently capable to serve as a knowledge source. For organizations that have already invested in training a large frontier model, this paper provides a cost-effective playbook for deriving the rest of the model family from that investment.

**The FLOPs calculation (Section 4.1) as a concrete economic argument:** The paper's 1.8× total compute savings for training the full Nemotron-4 family is not an abstract "our method is efficient" claim — it is a specific, auditable calculation that any organization can replicate for their own model families. The formula `(4.4e17 + 2.5e17/40 + 1.2e17/40) / (4.4e17 + 2.5e17 + 1.2e17) = 1/1.8` makes the economic case in the universal currency of FLOPs, independent of hardware, electricity prices, or training infrastructure specifics. This transforms the paper from a methods contribution into a **cost-model contribution** that can inform real resource allocation decisions.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All pruning and retraining experiments use the Nemotron-4 curated pretraining dataset, which comprises 8 trillion tokens (8T) of base training data, plus a continued training (CT) data blend used for final model retraining (Section 4). Ablation studies default to the 8T blend unless otherwise specified. Downstream evaluation uses established benchmarks: MMLU (Hendrycks et al., 2021, 5-shot), HumanEval (Chen et al., 2021, 0-shot pass@1), MBPP (0-shot pass@1), and several commonsense reasoning datasets — ARC-Challenge (Clark et al., 2018, 25-shot), HellaSwag (Zellers et al., 2019, 10-shot), TruthfulQA (Lin et al., 2022, 0-shot), WinoGrande (Sakaguchi et al., 2021, 5-shot), GSM8K (5-shot), and XL-Sum English (Hasan et al., 2021, 0-shot on 20% of the test set) for summarization.

- **Base model(s).** The primary model is Nemotron-4 15B (Parmar et al., 2024c), a 15.6-billion-parameter transformer with 32 layers, 6144 embedding dimension, 48 attention heads (8 key-value groups via grouped-query attention), and an MLP hidden dimension of 24576 (Table 5). The paper states this model was chosen because it represents "a canonical approach to scaling pretraining" (Section 4) with a standard architecture, making the pruning methodology applicable to other model families. A previous-generation Nemotron-3 8B model (NVIDIA, 2023) serves as a from-scratch training baseline for the 8B size class.

- **Metrics.** The primary training metric is **LM validation loss** (cross-entropy on held-out data from the 8T dataset, computed as negative log-likelihood averaged across tokens). For downstream evaluation, the paper reports **accuracy** for MMLU, ARC-Challenge, HellaSwag, WinoGrande, GSM8K, and TruthfulQA (mc2 metric), **pass@1** for HumanEval and MBPP (with temperature 0.2 and nucleus sampling top-p=0.95), and **rougeL** for XL-Sum. For instruction-tuned models, MT-Bench score, IFEval prompt-level accuracy, ChatRAG-Bench average, and BFCL average are used (Tables 6-9). Model size is reported both as total parameters and non-embedding parameters to account for the significant contribution of the 256K vocabulary embedding matrix (Table 5).

- **Baselines.** The paper compares against: (1) **Nemotron-4 15B** — the uncompressed teacher model; (2) **Nemotron-3 8B** — a previous-generation model trained from scratch with 3.8T tokens (NVIDIA, 2023); (3) **Community models trained from scratch**: Llama-3 8B (>15T tokens), Llama-2 7B (2T tokens), Mistral 7B (8T tokens), Gemma 7B/8.5B (6T tokens) for the 8B class (Table 2), and Phi-2 2.7B (1.4T tokens), Gemma 2.5B (3T tokens), Gemma2 2.6B (2T tokens), Qwen2 1.5B (7T tokens), MiniCPM 2.7B (1.1T tokens) for the 4B class (Table 3); (4) **State-of-the-art pruned models**: LLM-Pruner (Ma et al., 2023, 9.8B), SliceGPT (Ashkboos et al., 2024, 9.9B/4.9B), LaCo (Yang et al., 2024, 9.8B/4.9B), ShortGPT (Men et al., 2024, 9.8B/4.9B), and Sheared LLaMA (Xia et al., 2023, 2.7B) (Table 4). All baselines are evaluated by the authors on the same benchmarks using the same evaluation scripts, except for entries marked with an asterisk in the tables, which are drawn from the corresponding papers.

- **Generation budget / compute accounting.** The standard unit of retraining cost is **number of training tokens processed**. Lightweight retraining for ablations and NAS evaluation uses approximately 1.8B tokens (400 steps at batch size 1152, with sequence length such that 400 steps corresponds to ~1.8B tokens). Full retraining for final MINITRON models uses 94B tokens for MINITRON 8B and a total of ~100B tokens (split across two iterative stages) for MINITRON 4B. Compute cost savings are expressed in raw FLOPs: training the 15B, 8B, and 4B models from scratch costs `(4.4e17 + 2.5e17 + 1.2e17) × steps` FLOPs, while the pruning approach costs `(4.4e17 + 2.5e17/40 + 1.2e17/40) × steps`, yielding 1.8× savings (Section 4.1). Iso-compute comparisons between conventional training and distillation (Table 11) match total FLOPs by adjusting token counts — e.g., 150B tokens of conventional training vs. 100B tokens of distillation training, accounting for the extra forward pass through the teacher model during distillation.

- **Cross-validation / statistical protocol.** For the lightweight NAS phase, all feasible architecture candidates (15 for 8B, 18 for 4B) undergo identical lightweight retraining (400 steps, ~1.8B tokens), and the candidate with the best validation loss is selected for full retraining. The paper does not report confidence intervals, error bars, or statistical significance tests for downstream benchmark results. The test set for downstream evaluation is the standard benchmark test splits (e.g., MMLU's 14,042 test questions, HumanEval's 164 problems), and each evaluation is run once per model with the specified few-shot and sampling configuration. For the calibration dataset used in importance estimation, 1024 samples are drawn randomly from the full 8T training dataset (Section 4) — the paper does not specify whether this sampling is repeated across experiments or whether different random seeds produce different importance rankings.

---

### Main Quantitative Results

#### How Pruning Strategy and Retraining Interact

The paper's first set of core results establishes the fundamental dynamics between what you prune, how much you retrain, and what performance you recover. These are not downstream benchmark results but rather **training dynamics** that justify the later best practices and final model quality.

**Purely activation-based pruning without retraining causes severe degradation, but the right aggregation metric matters substantially.** Table 13 reports zero-shot LM validation loss (no retraining) for different batch×sequence aggregation functions when pruning Nemotron-4 15B to the Nemotron-3 8B architecture (reducing embedding from 6144→4096, heads from 48→32, MLP hidden from 24576→16384). The best combination — **(batch=L2, seq=mean)** — achieves 7.18 LM loss on the 8T validation set, while the worst — **(batch=var, seq=var)** — achieves 10.55, a 47% relative degradation. On WikiText2, the same pattern holds: (mean, mean) achieves 6.89 while (var, var) reaches 11.14. The gap between the best and worst aggregation choices (3.37 LM loss points on 8T) exceeds the degradation from pruning itself relative to the unpruned teacher — meaning that **choosing the wrong aggregation function can be more damaging than the pruning operation itself**.

**These aggregation rankings persist after retraining, validating activation-based pruning as a practical strategy.** Figure 5 compares the retraining LM validation loss curves for two pruned models using the best metric — **(L2, mean)** — and a poorly-performing one — **(L2, L2)** — both retrained for 400 steps (~1.8B tokens). The (L2, mean) model starts at lower loss (approximately 2.17 at step 100 vs. 2.22 for (L2, L2)) and maintains this advantage throughout all 400 training steps, with final losses of approximately 2.05 vs. 2.08. The curves do not cross, confirming that zero-shot importance rankings are predictive of retrained performance — a property that is not guaranteed and must be empirically verified, because retraining could theoretically compensate for poor initial pruning decisions by reallocating function to surviving components.

**Lightweight retraining (~1.8B tokens) causes candidate architecture rankings to stabilize.** Figure 9 plots the LM validation loss trajectories for all 15 feasible 8B architecture candidates during lightweight retraining. The relative ordering of candidates fluctuates substantially during the first ~300 steps, then stabilizes. For instance, a candidate that appears best at step 150 may drop to fifth place by step 350. The paper uses this observation to justify a 400-step lightweight retraining phase for NAS: shorter retraining would risk selecting a suboptimal architecture based on unstable early rankings, while full retraining of all candidates would be unnecessarily expensive. The final selection (candidate with lowest loss at step 400) is the one chosen for full retraining.

**Single-shot importance estimation is sufficient; iterative re-estimation provides no benefit after retraining.** Table 14 reports LM validation loss before and after lightweight retraining for embedding dimension pruning from the original size to 4096 using T=1 (one-shot), T=2, and T=4 iterations. Before retraining, the iterative approaches show different losses (5.43, 5.55, 5.24 for T=1, 2, 4 respectively), with T=4 appearing best. After 400 steps of lightweight retraining, **all three variants converge to exactly 1.92** — an identical final loss. The retraining process fully compensates for any differences in the quality of the initial pruning decisions, making the additional compute cost of iterative importance estimation unnecessary. This result directly supports Best Practice #3 ("Use single-shot importance estimation; iterative provides no benefit") and suggests that, at least for width pruning of this magnitude, the importance metric is sufficiently robust that iterative refinement of which specific channels to remove is irrelevant once retraining is applied.

**Knowledge distillation dramatically outperforms conventional training for post-pruning accuracy recovery.** Table 11 provides the paper's most stark demonstration of distillation's value. Under iso-compute conditions (matching total FLOPs, accounting for the teacher forward pass in distillation), three approaches to producing a 4B model are compared:

| Approach | Tokens | MMLU (%) | HellaSwag (%) |
|----------|--------|-----------|---------------|
| 4B Random Init (train from scratch) | 150B | 24.36 | 46.22 |
| 4B-Pruned (prune 15B, conventional retrain) | 150B | 24.57 | 50.85 |
| 4B-Pruned-Distill (prune 15B, distill retrain) | 100B | **37.81** | **51.04** |

The conventionally retrained pruned model (24.57% MMLU) performs essentially identically to a randomly initialized model trained from scratch (24.36%) — pruning without distillation provides no benefit on MMLU beyond architecture selection. Distillation raises MMLU by 13.24 percentage points over conventional retraining (a 54% relative improvement) while using 33% fewer tokens. On HellaSwag, the gains are more modest but still present: 51.04% vs. 50.85%. This establishes that the knowledge transfer from the teacher model — not the structural initialization from pruning — is the primary driver of accuracy recovery at aggressive compression rates.

#### Width vs. Depth Pruning Dynamics

**Width pruning systematically outperforms depth pruning after retraining, despite initially appearing worse.** Table 10 and Figure 6 together reveal a temporal dynamic that the paper highlights as a key empirical discovery. Four pruning strategies are compared for compressing Nemotron-4 15B to ~8B parameters, each retrained for 400 steps:

| Strategy | Parameters | Final LM Loss |
|----------|------------|---------------|
| Depth only (PPL-based) | 9.39B | 2.155 |
| Depth only (BI-based) | 9.39B | 2.177 |
| Width only (heads + neurons + embeddings) | 7.74B | **2.049** |
| Depth + Width combined | 7.91B | 2.062 |

The pure width-pruned model achieves the lowest retrained LM loss (2.049) despite having the fewest parameters (7.74B) — fewer than any other variant. The depth+width combined model (7.91B) performs worse (2.062) than pure width pruning, and both depth-only models (9.39B each) perform substantially worse despite having 1.65B more parameters. This is counterintuitive: removing layers is more aggressive than removing width dimensions in terms of structural change (it eliminates entire computational stages rather than reducing capacity within stages), so one might expect width pruning to be more damaging. Instead, width pruning preserves the full 32-layer computational depth, which appears to provide a better structural prior for retraining to work with.

**However, this ranking flips during early retraining.** Figure 6 shows the LM validation loss curves for the 32-layer width-only model vs. the 28-layer depth+width model. At the start of retraining (step 100), the 28-layer model has lower loss (approximately 2.24 vs. 2.27). Around step 200, the curves cross, and by step 400, the 32-layer model is clearly ahead. This dynamic reversal is the paper's justification for its lightweight retraining phase in NAS (Best Practice #9): evaluating architectures based on their initial post-pruning loss would have selected the wrong model. The paper does not provide a mechanistic explanation for why width pruning catches up and overtakes depth pruning, but the implication is practical: always retrain enough for rankings to stabilize before making architecture decisions.

#### Distillation Design Choices

**KLD is the best distillation loss function, and logit-only distillation is sufficient when depth is not reduced.** Table 15 (on a previous-generation Nemotron-3 8B model) tests five loss configurations: combined conventional LM loss + logit distillation with different divergence measures, and logit-only distillation. KLD-based logit-only distillation achieves the best LM loss (2.107) and WikiText perplexity (8.720), outperforming all combined loss variants: `L_CLM + L_logits(KLD)` at 2.117/8.791, `L_CLM + L_logits(Cosine)` at 2.134/8.965, and `L_CLM + L_logits(MSE)` at 2.144/9.007. Surprisingly, adding the conventional language modeling loss (`L_CLM`) to the KLD distillation loss **degrades** performance (2.107 → 2.117) — the paper speculates this is because the teacher's logit distribution already encodes the ground-truth information, and the conventional loss may introduce conflicting gradients. Table 16 confirms KLD's advantage on MINITRON 8B depth-pruned: KLD achieves 2.155 LM loss vs. 2.665 for reverse KLD, a 0.51 gap.

**For width-pruned models that preserve depth, intermediate state distillation provides no benefit.** Table 18 reports a direct comparison for MINITRON 8B-width-pruned: adding intermediate state distillation (`L_is`, including encoder block outputs, embeddings, etc.) to logit distillation produces 58.0% MMLU and 73.6% HellaSwag, while logit-only distillation achieves 58.3% MMLU and 73.8% HellaSwag — a negligible difference of 0.3 and 0.2 percentage points respectively, within the range of evaluation noise. This directly motivates Best Practice #7: "Use logit-only distillation when depth isn't reduced significantly." The paper does not ablate why this is the case, but a plausible interpretation is that when the student has the same number of layers as the teacher, it can learn compressed internal representations that functionally match the teacher's solely from the output-level supervision signal.

**For depth-pruned models, intermediate state distillation (particularly `L_o + L_emb`) does help, and thoughtful layer mapping matters.** Table 17 systematically explores `L_is` components for a depth-pruned MINITRON 8B variant (16 layers, student layers indexed 0-15, teacher layers indexed 0-31). The baseline (logit-only) achieves 2.155 LM loss. Adding encoder block output loss `L_o(29:13)` — mapping student layer 13 to teacher layer 29 — reduces loss to 2.145. Adding embedding loss `L_emb` to `L_o(28:12)` further reduces loss to 2.141. However, poor layer mappings can be worse than no intermediate state loss: `L_o(15:15) + L_emb` (mapping student's middle to teacher's middle) produces 2.240, which is 0.085 worse than logits alone. The best combination — `L_logits + L_o(28:12) + L_emb` — achieves 2.141. Adding MLP input loss `L_i` makes no difference (2.141 with vs. 2.141 without), and attention relation loss `L_att` "does not show any improvement" (mentioned in Appendix A.4 but not quantified in the table). The paper's finding that mapping to the **final layers** of the teacher (29:13, 28:12) works best is consistent with the observation from Fu (2024) that "the final 1-2 layers in a Transformer for LLM are highly specialized."

**The temperature parameter does not benefit from softening (τ > 1) in LLM distillation, unlike in vision.** The paper experiments with τ ∈ {0.1, 0.5, 1.0, 3.0} (Appendix A.3) and finds τ = 1.0 is optimal. The reasoning provided: "LLM logit distributions have higher entropy and hence the inspiration for temperature < 1 to reduce the noise" — that is, LLMs already produce relatively flat distributions (many plausible next tokens), so the standard vision-motivated practice of softening with τ > 1 to expose secondary class information is unnecessary. Conversely, τ < 1 would sharpen an already-flat distribution toward one-hot, losing the "dark knowledge" that distillation aims to transfer. This finding is reported without a detailed quantitative ablation table for the Nemotron-4 experiments, but the authors note it was validated on the previous-generation Nemotron-3 models and not repeated.

**Dynamic α weighting outperforms a fixed constant for balancing logit and intermediate state losses.** The paper computes α = `L_logits / L_is` dynamically at each step rather than using a constant. The justification is that "the magnitudes of `L_logits` and `L_is` differ significantly" — intermediate state losses involving high-dimensional hidden state comparisons have naturally larger magnitude than KL divergence over probability distributions. A fixed weight would require per-model, per-configuration tuning, while dynamic weighting automatically maintains gradient balance between the two objectives throughout training. The paper reports this "achieves better results compared to using a constant" but does not provide a direct ablation comparing fixed vs. dynamic α for the same model configuration.

#### Iterative Compression for Large Size Reductions

**For aggressive 15B→4B compression (73% weight reduction), iterative two-stage compression substantially outperforms one-shot pruning.** Table 11 (last two rows) provides the direct comparison:

| Strategy | Total Tokens | MMLU (%) | HellaSwag (%) |
|----------|-------------|-----------|---------------|
| One-shot: 15B→4B + distill | 100B | 37.81 | 51.04 |
| Iterative: 15B→8B→4B + distill | 100B | **42.45** | **52.04** |

The iterative approach achieves 4.64 percentage points higher MMLU (a 12.3% relative improvement) and 1.0 point higher HellaSwag, with equivalent total token budget. Critically, both the intermediate (8B) and final (4B) models use the original Nemotron-4 15B as the teacher during distillation — the paper explicitly notes that "during the final retraining step, we observe that using Nemotron-4 15B as the teacher achieves superior results compared to using MINITRON 8B." This implies that the intermediate model, while useful as a structural stepping stone, has already lost some knowledge relative to the original 15B teacher, and the stronger supervision signal from the uncompressed model is valuable.

**Within a single dimension, iterative pruning is worse than one-shot pruning for depth reduction.** Figure 7 (Appendix A.5) compares three strategies for removing 16 layers from the 32-layer model: (1) iterative ×1: remove one layer at a time, distilling with 1.8B tokens after each removal (total 16 × 1.8B = 28.8B tokens); (2) iterative ×4: similar but with 4 × 1.8B = 7.2B tokens after each removal starting from the 26-layer mark (total 86.4B tokens); and (3) one-shot: remove all 16 layers at once, distill with 28.8B tokens. The one-shot strategy outperforms both iterative variants across MMLU, HellaSwag, and HumanEval, even when the iterative approach uses 3× more total tokens (86.4B vs. 28.8B). The paper reports that accuracy on HellaSwag and PIQA is retained up to 31 layers (removing only 1 layer is essentially lossless), remains near-plateau until ~26 layers, and then drops sharply when compressed to 25 layers and below. MMLU follows a similar pattern but with an earlier sharp drop at 20 layers. The implication is that the first few layers removed are genuinely redundant, but beyond a threshold, each additional layer removal causes disproportionate damage that iterative removal with intermediate distillation cannot mitigate better than one-shot removal with equivalent total compute.

#### Architecture Search Produces Specific Non-Obvious Design Choices

**The NAS consistently selects architectures that preserve full depth (32 layers) while concentrating all parameter reduction in width dimensions.** Table 5 shows the final architectures:

| Model | Layers | Embedding | Heads | MLP Hidden | Parameters |
|-------|--------|-----------|-------|------------|------------|
| Nemotron-4 15B | 32 | 6144 | 48 | 24576 | 15.6B |
| MINITRON 8B | 32 | 4096 | 48 | 16384 | 8.27B |
| MINITRON 4B | 32 | 3072 | 24 | 9216 | 4.19B |

MINITRON 8B preserves all 32 layers and all 48 attention heads, reducing only the embedding dimension (6144→4096, 33% reduction) and MLP hidden dimension (24576→16384, 33% reduction). The head count remains at 48 — the search did NOT select any of the 32-head candidates listed in Table 19. MINITRON 4B also preserves all 32 layers, cutting embedding dimension in half (6144→3072), heads from 48→24, and MLP hidden from 24576→9216 (a 62.5% reduction). This pattern — preserving depth, reducing width — is not an a priori design choice but an **emergent finding from the search process**: of the 15 feasible 8B architectures enumerated in Table 19, layers ranging from 29-32 were all tested, and the 32-layer variant was selected as optimal. This empirically validates Best Practice #4: "Prefer width pruning over depth for the model scales we consider (≤15B)."

#### Final MINITRON Model Quality

**MINITRON 8B matches or exceeds models trained from scratch with orders of magnitude more tokens, while MINITRON 4B selectively outperforms similarly-sized specialized models.** Tables 2 and 3 report the headline downstream benchmark results. For the 8B class (Table 2):

| Model | Tokens | MMLU | HellaSwag | GSM8K | HumanEval | WinoGrande |
|-------|--------|------|-----------|-------|-----------|------------|
| MINITRON 8B | **94B** | 63.8 | 80.7 | 51.3 | 31.6 | 79.0 |
| Nemotron-3 8B | 3.8T | 54.7 | 78.5 | 24.0 | 20.7 | 75.9 |
| Llama-3 8B | >15T | 65.3 | 82.1 | 50.3 | 28.1 | 77.6 |
| Mistral 7B | 8T | 64.1 | 83.2 | 37 | 28.7 | 78.5 |
| Gemma 7B | 6T | 64 | 82 | 50 | 32 | 78 |
| Llama-2 7B | 2T | 46 | 79 | 14 | 12 | 74 |
| Nemotron-4 15B | 8T | 66.6 | 84.6 | 48.5 | 35.4 | 83.6 |

MINITRON 8B achieves 63.8% MMLU — within 2.8 points of the 15B teacher (66.6%), 1.5 points behind Llama-3 8B (65.3%), 0.3 points behind Mistral 7B (64.1%), and 9.1 points ahead of Nemotron-3 8B (54.7%) which was trained from scratch with 3.8T tokens. On GSM8K, MINITRON 8B (51.3%) outperforms ALL compared models including the 15B teacher (48.5%) — the only benchmark where the compressed model exceeds its parent, possibly because distillation on math-heavy data sharpens the student's reasoning in ways the teacher's broader training did not. On HumanEval, it achieves 31.6%, behind Gemma 7B (32%) and the teacher (35.4%) but ahead of Llama-3 8B (28.1%) and Mistral 7B (28.7%).

For the 4B class (Table 3):

| Model | Tokens | MMLU | HellaSwag | GSM8K | HumanEval | WinoGrande |
|-------|--------|------|-----------|-------|-----------|------------|
| MINITRON 4B | **94B** | 58.6 | 75.0 | 24.1 | 23.3 | 74.0 |
| Phi-2 2.7B | 1.4T | 57.5 | 75.2 | 55 | 50 | 74 |
| Gemma 2.5B | 3T | 42 | 72 | 18 | 24 | 67 |
| Gemma2 2.6B | 2T | 51.3 | 73.0 | 23.9 | 17.7 | 70.9 |
| Qwen2 1.5B | 7T | 56.5 | 66.6 | 58.5 | 31.1 | 66.2 |
| MiniCPM 2.7B | 1.1T | 53.5 | 68.3 | 53.8 | — | — |

MINITRON 4B achieves 58.6% MMLU, outperforming all similarly-sized models except the slightly smaller Phi-2 (57.5%, 2.7B parameters, 1.4T tokens). The gap to Phi-2 is marginal (1.1 points) despite MINITRON using 15× fewer training tokens. However, on coding benchmarks, MINITRON 4B is substantially behind Phi-2 (HumanEval: 23.3% vs. 50%) and behind Qwen2 1.5B (31.1%). On GSM8K, it similarly lags (24.1% vs. Phi-2's 55% and Qwen2's 58.5%), suggesting that the Nemotron-4 15B teacher's capabilities in math and code were not as strong as specialized small models like Phi-2, and compression cannot create capability the teacher lacks.

**MINITRON models substantially outperform all prior pruned models of comparable size.** Table 4 provides direct comparison:

- **8B class:** MINITRON 8B (63.8% MMLU, 80.7% HellaSwag) vs. LLM-Pruner 9.8B (25.2%, 67.8%), SliceGPT 9.9B (37.1%, 55.7%), LaCo 9.8B (45.9%, 64.4%), ShortGPT 9.8B (54.7%, 66.6%). MINITRON's MMLU is 9.1 points ahead of the best prior method (ShortGPT) and 38.6 points ahead of LLM-Pruner, despite having 1.5B fewer parameters (8.3B vs. 9.8B). The gap on HellaSwag is similarly large: 14.1 points over the next best.

- **4B class:** MINITRON 4B (58.6% MMLU, 75.0% HellaSwag) vs. LLM-Pruner 4.8B (23.33%, 56.46%), SliceGPT 4.9B (28.92%, 50.27%), LaCo 4.9B (26.45%, 55.69%), ShortGPT 4.9B (43.96%, 53.02%), Sheared LLaMA 2.7B (26.4% MMLU, 70.8% HellaSwag). MINITRON's MMLU is 14.6 points ahead of ShortGPT (the best prior pruned model) and 32.2 points ahead of the others. Even accounting for Sheared LLaMA being smaller (2.7B vs. 4.2B), the gap is enormous.

These gaps are large enough to suggest that prior pruning methods — all of which use gradient-based importance estimation and conventional training for retraining — lose substantially more of the teacher's knowledge than the activation-based pruning + distillation approach. The paper does not ablate whether the improvement comes from activation-based importance, distillation retraining, or both, but the combined effect is transformative: prior pruned models were not competitive with from-scratch models, while MINITRON models are.

**Instruction-tuned MINITRON 4B-instruct performs competitively on deployment-facing benchmarks.** Tables 6-9 show that MINITRON 4B, after supervised fine-tuning on the Nemotron-4 340B instruction-tuning data (NVIDIA, 2024), achieves results that often exceed similarly-sized instruction-tuned models:

- **MT-Bench (Table 6):** 6.46 average score vs. Phi-2 (4.29), Qwen-1.5 Chat (5.29), Gemma-2B-IT (5.19), StableLM 2 Chat 1.6B (5.42), TinyLlama v1.0 Chat 1.1B (3.46). MINITRON 4B-instruct is the only sub-5B model to exceed 6.0 on MT-Bench, a score typically associated with 7B+ models.

- **IFEval (Table 7):** 73.01% prompt-level accuracy (loose) vs. Gemma-2B-IT (28.70%, from their paper). The 44-percentage-point gap is dramatic and suggests strong instruction-following capability transferred from the teacher.

- **ChatRAG-Bench (Table 8):** 41.11 average vs. Gemma-2B-IT (33.31), indicating strong retrieval-augmented generation performance.

- **BFCL v2 function calling (Table 9):** 53.09 average vs. Gemma-2B-IT (41.63) and even Llama-3-8B-instruct (50.51). A 4B model outperforming an 8B model on function calling is unexpected and suggests the instruction-tuning data effectively transferred structured reasoning capabilities from the large teacher.

---

### Ablation Studies and Robustness Checks

**Aggregation function for activation-based importance**: The choice of (batch, sequence) aggregation function has a 47% impact on zero-shot LM loss after pruning, with (L2, mean) and (mean, mean) performing best and variance-based functions performing worst. After retraining, the (L2, mean) advantage over (L2, L2) is maintained throughout all 400 training steps (Figure 5). Table 13 provides the quantitative results. This is not a minor hyperparameter sensitivity — it is a first-order design choice that can make activation-based pruning appear to fail if chosen poorly.

**Iterative vs. single-shot importance estimation**: T=1, T=2, and T=4 iterations all converge to identical retrained LM loss of 1.92 (Table 14), demonstrating that the additional compute cost of iterative importance estimation is wasted. This result is robust: it holds even though the iterative approaches show different zero-shot losses (5.43, 5.55, 5.24), confirming that retraining can compensate for moderate differences in which specific channels are pruned.

**Width-only vs. depth-only vs. combined pruning**: Pure width pruning (7.74B parameters, 2.049 LM loss) outperforms combined depth+width (7.91B, 2.062), pure depth with PPL (9.39B, 2.155), and pure depth with BI (9.39B, 2.177) after retraining (Table 10). The temporal reversal shown in Figure 6 — combined pruning initially looks better but is overtaken by width-only at ~200 steps — validates that lightweight retraining is necessary for reliable architecture comparison.

**Distillation vs. conventional retraining**: Under iso-compute, distillation achieves 37.81% MMLU vs. 24.57% for conventional retraining (Table 11), a 13.24 percentage point gap. This establishes that distillation is not merely an incremental improvement but a qualitatively different retraining regime — without distillation, pruning provides essentially no benefit over random initialization for complex reasoning tasks like MMLU.

**Choice of distillation loss function (KLD vs. R-KLD, MSE, cosine)**: KLD consistently outperforms alternatives across two model generations. On Nemotron-3 8B (Table 15): KLD alone achieves 2.107 LM loss vs. R-KLD at 2.140, Cosine at 2.134, and MSE at 2.144. On MINITRON 8B depth-pruned (Table 16): KLD at 2.155 vs. R-KLD at 2.665 (0.51 gap). The results confirm that mean-seeking divergence (KLD) is more appropriate than mode-seeking (R-KLD) for base model distillation.

**Logit-only vs. logit + intermediate state distillation**: For width-pruned MINITRON 8B with preserved depth (Table 18), adding `L_is` (embeddings, encoder block outputs, attention states) to `L_logits` produces negligible difference: 58.0% vs. 58.3% MMLU, 73.6% vs. 73.8% HellaSwag. This validates Best Practice #7 and significantly simplifies the distillation pipeline for the common case where only width is pruned. For the depth-pruned variant (Table 17), the best `L_is` configuration (`L_logits + L_o(28:12) + L_emb`) improves LM loss from 2.155 to 2.141 — a small but consistent gain — while poor layer mappings can degrade performance (e.g., `L_o(15:15) + L_emb` at 2.240).

**Distillation temperature**: τ = 1.0 is optimal; softening (τ = 3.0) or sharpening (τ = 0.1, 0.5) degrades performance (Appendix A.3). The paper attributes this to LLMs' already-high entropy output distributions, making the standard vision-motivated softening unnecessary. No detailed quantitative ablation table is provided for the Nemotron-4 experiments, though the finding was validated on the previous-generation models.

**Dynamic vs. fixed α for intermediate state loss weighting**: Dynamic computation of α = `L_logits / L_is` outperforms fixed weighting by automatically balancing gradient magnitudes across objectives. No direct ablation is provided, but the paper states it "achieves better results compared to using a constant" based on experiments with the previous-generation Nemotron-3 models.

**Top-K logit filtering**: Retaining only the top-K teacher logits (K ≤ 100) causes significant accuracy degradation; larger K approaches the full-distribution performance but provides no benefit (Appendix A.3). The conclusion: for LLM distillation with a 256K vocabulary, the full distribution matters — there is no free lunch from filtering.

**Single-phase vs. multi-phase retraining data**: Pruning the phase-2 checkpoint and retraining with phase-2 data only (94B tokens, 61.9% MMLU) outperforms pruning the phase-1 checkpoint and retraining with a mix of phase-1 and phase-2 data (113B tokens, 54.7% MMLU) (Table 20). This is a 7.2 percentage point gain with 19B fewer tokens, validating Best Practice #10 and suggesting that retraining on the highest-quality data from the final training phase is sufficient — there is no need to replicate the teacher's full multi-phase training curriculum.

**One-shot vs. iterative pruning across dimensions**: For a two-axis compression (embedding → MLP + attention), one-shot pruning achieves better retrained LM loss than iterative (embedding pruning → retrain → MLP + attention pruning → retrain) on equivalent token budget (Figure 8, Appendix A.5). Combined with the within-dimension result (Table 14, where iterative provides no benefit), this consistently favors one-shot pruning across all tested scenarios.

**One-shot vs. iterative pruning for depth only**: One-shot removal of 16 layers with 28.8B tokens of distillation outperforms iterative removal (one layer at a time, distilling after each) with 28.8B or even 86.4B total tokens (Figure 7, Appendix A.5). This result — iterative being worse even with 3× the compute budget — is counterintuitive and suggests that intermediate distillation after each layer removal does not provide a better optimization landscape than one-shot removal followed by equivalent total distillation.

**Head residual preservation**: The paper mentions that adding residual information from pruned heads back into surviving heads provides "a boost to model accuracy in our experiments" (Section 2.3), but does NOT provide a direct ablation comparing pruning with vs. without this technique. This is a notable gap: while the overall MINITRON results are strong, the specific contribution of head residual preservation is unquantified.

**Teacher choice for iterative compression**: Using Nemotron-4 15B as the teacher for BOTH the 15B→8B and 8B→4B distillation steps achieves better results than using MINITRON 8B as the teacher for the final step. The paper notes this observation (Section 4.3) but does not provide a quantitative ablation table for this specific comparison.

**LR schedule and optimizer**: The paper uses the same optimizer settings and cosine LR decay schedule from 2e-4 to 4.5e-7 as Nemotron-4 15B's original training (Section 4). No alternative LR schedules or optimizers are ablated — the assumption is that the teacher's training configuration transfers to the distillation setting.

**Calibration dataset size**: 1024 samples are used for importance estimation. The paper does not ablate calibration dataset size (e.g., 512 vs. 1024 vs. 2048 samples) to determine whether importance rankings are sensitive to this choice. Given that importance estimation cost grows linearly with calibration set size, this would be a practically relevant ablation.

---

### Critical Assessment

The experimental results broadly support the paper's central claims, but the nature of that support varies across claims, and several important qualifications and gaps emerge on close reading.

**Claim: "Training one big model and obtaining smaller ones from it through pruning + retraining achieves higher accuracy and is extremely cost/compute-efficient compared to training from scratch."**

This claim is **well-supported for accuracy** at the specific model sizes tested, as Tables 2-3 demonstrate MINITRON models matching or exceeding from-scratch baselines with 40× fewer training tokens. However, the claim of "higher accuracy" should be interpreted carefully: MINITRON 8B outperforms Nemotron-3 8B (a previous-generation from-scratch model) by 9.1 points on MMLU, but this compares against a model trained with an earlier training recipe and data mix — it does not control for the fact that Nemotron-4 15B (the teacher) was itself trained with better data and methods than Nemotron-3 8B. The more relevant comparison is against contemporaneous from-scratch models (Llama-3 8B, Mistral 7B, Gemma 7B), where MINITRON 8B is competitive but does not uniformly outperform — it is within 1-2 points on MMLU, behind on HellaSwag (80.7 vs. 82.1-83.2), and ahead on GSM8K (51.3 vs. 37-50). The claim of "higher accuracy" is thus better phrased as "competitive accuracy at dramatically lower cost" rather than "strictly better accuracy."

**Claim: "Up to 40× fewer training tokens" and "1.8× total compute savings for training the full model family."**

The 40× figure is calculated by comparing MINITRON's 94B retraining tokens against the 3.8T tokens used to train Nemotron-3 8B from scratch. This is a valid comparison, but it compares against a different model architecture and training recipe. A more conservative comparison would be against an estimate of what a from-scratch Nemotron-4 8B would cost — likely 8T tokens if following the same recipe as the 15B model — which would make the savings closer to 85× rather than 40×. The 1.8× figure for the family (Section 4.1) is a defensible FLOPs calculation given the assumptions (equal token counts per model when trained from scratch, batch size 1152), but it assumes that the from-scratch models would use the full 8T token budget — if smaller models are typically trained on fewer tokens (as is common practice), the savings would be lower. Neither the 40× nor the 1.8× figure accounts for the cost of importance estimation (the 1024-sample forward passes) or the lightweight NAS phase (15-18 candidates × 1.8B tokens each, so ~27-32B tokens of additional compute), which are small but non-zero.

**Claim: "MINITRON models outperform state-of-the-art compression techniques from the literature."**

This claim is **convincingly supported** by Table 4, with very large margins (9-39 MMLU points, 14+ HellaSwag points). However, the comparison is somewhat unfair to prior work: LLM-Pruner, SliceGPT, LaCo, and ShortGPT all prune LLaMA-based models rather than Nemotron-4, so the base model quality differs. The paper does not control for the teacher model's inherent capability. A more rigorous comparison would apply prior methods (e.g., ShortGPT's depth pruning) to the same Nemotron-4 15B teacher and compare against MINITRON's approach on identical starting conditions. Without this ablation, we cannot distinguish how much of the improvement comes from the pruning methodology vs. simply starting from a better teacher. The paper also does not report whether prior methods were given equivalent retraining budgets — if prior methods used less retraining data, the comparison conflates pruning quality with retraining sufficiency.

**Missing ablation: Activation-based vs. gradient-based importance estimation on the same model.** The paper argues that activation-based importance is superior because it avoids gradient computation, but it never directly compares its activation-based pruning against a gradient-based method (e.g., Taylor expansion or learned masks) applied to the same Nemotron-4 15B model with the same retraining budget. Without this ablation, the claim that activation-based methods are sufficient — rather than simply convenient — is not experimentally validated. Such an ablation would require implementing a gradient-based pruning method (e.g., LLM-Pruner's approach) for Nemotron-4, which is non-trivial but would substantially strengthen the paper's central methodological claim.

**Missing ablation: Distillation vs. conventional retraining for the same pruned architecture with more tokens.** Table 11 shows distillation at 100B tokens outperforming conventional training at 150B tokens under iso-compute. But what if conventional training were given even more tokens? The paper doesn't explore whether conventional training would eventually catch up to distillation. Figure 6 suggests LM loss continues decreasing through 400 steps (~1.8B tokens), but the full retraining runs (~100B tokens) don't compare distillation vs. conventional training at equivalent token counts (only iso-compute). We cannot determine whether distillation provides a permanent advantage or merely accelerates convergence that conventional training would eventually match.

**Missing ablation: Sensitivity to the pruned architecture search seed.** The NAS process selects the best of 15-18 candidates after lightweight retraining. The paper does not report whether running the same architecture search with a different calibration dataset sample (different random seed for the 1024 samples) produces the same architecture selection. If the rankings are sensitive to the calibration data, the approach might require ensemble averaging over multiple calibration sets, adding cost. The paper's observation that rankings stabilize after ~300 steps of retraining (Figure 9) provides some reassurance, but it doesn't test robustness to calibration data variation.

**Single model family limitation.** All experiments use Nemotron-4 15B with its specific architecture (GQA with 48 query heads and 8 KV groups, SwiGLU MLP, 6144 embedding dimension). The paper argues this model is "representative of the capabilities of many contemporary LLMs" (Section 4), but this is an assertion, not a finding. Whether the best practices transfer to models with different architectures (e.g., dense attention, different activation functions, different depth-to-width ratios) is unknown. In particular, Best Practice #4 ("prefer width pruning over depth") was derived on a 32-layer model and might not hold for deeper models (70B+ parameters with 80+ layers) where layer redundancy may be more extreme.

**Small calibration dataset (1024 samples) without sensitivity analysis.** The entire importance estimation pipeline uses only 1024 samples. The paper does not test whether importance rankings are stable with fewer samples (e.g., 256 or 512) or whether additional samples (e.g., 4096) would improve the rankings. Given that this is a critical design choice affecting both cost and pruning quality, the lack of a sensitivity analysis is a gap.

**No statistical significance reporting.** All downstream evaluations are reported as single-point estimates without confidence intervals, error bars, or multi-seed runs. MMLU at 5-shot has known variance depending on the specific few-shot examples selected — state-of-the-art evaluations typically report standard deviations or run multiple prompts. A difference of 1-2 MMLU points (e.g., MINITRON 8B at 63.8 vs. Mistral 7B at 64.1) may not be statistically significant, yet the paper's comparisons rely on point estimates. This is standard practice for many LLM papers but worth noting when evaluating the strength of "comparable to" or "outperforms" claims.

**Instruction-tuning results are underdescribed.** Tables 6-9 report strong instruction-tuning results for MINITRON 4B-instruct, but the paper provides no details about the SFT data mixture, training hyperparameters, or number of SFT epochs. The data used is described only as "instruction-tuning data used for Nemotron-4 340B" (an even larger model from the same team), making these results difficult to reproduce or interpret. Furthermore, the instruction-tuned baselines (Phi-2, Qwen-1.5 Chat, Gemma-2B-IT, etc.) may have been trained with different SFT data and recipes, confounding the comparison.

**Positive framing of moderate results on coding and math for MINITRON 4B.** Table 3 shows MINITRON 4B at 24.1% GSM8K and 23.3% HumanEval — substantially behind Phi-2 (55%, 50%) and Qwen2 1.5B (58.5%, 31.1%). The paper's discussion emphasizes the MMLU comparison (where MINITRON 4B leads at 58.6%) and mentions these coding/math deficits without detailed analysis. This pattern is consistent with the teacher's capabilities: Nemotron-4 15B achieves 48.5% GSM8K and 35.4% HumanEval (Table 2), substantially behind specialized small models on coding. The pruning process preserves but does not improve the teacher's relative weaknesses — a boundary condition that the paper acknowledges implicitly but could make more explicit.

**The paper would be strengthened by:** (1) A direct ablation of activation-based vs. gradient-based importance on the same model with the same retraining budget; (2) reporting whether conventional retraining with more tokens can match distillation; (3) testing sensitivity to calibration dataset size and random seed; (4) applying a prior pruning method (e.g., ShortGPT depth pruning) to Nemotron-4 15B to isolate the effect of the teacher model quality; (5) confidence intervals or multi-seed evaluations; (6) detailed SFT configuration for instruction-tuned models; (7) testing whether the best practices generalize to a model with a substantially different architecture (e.g., a dense non-GQA model or a much deeper model). Despite these gaps, the core experimental results — that distillation-based retraining is dramatically more effective than conventional retraining, that activation-based importance estimation is viable, and that the resulting compressed models compete with from-scratch models at much lower cost — are well-supported by the presented evidence.

## 6. Limitations and Trade-offs

### 6.1 Cost of Importance Estimation and Architecture Search Is Unaccounted for in the Headline Savings

**The assumption or constraint.** The paper reports 40× fewer training tokens and 1.8× total FLOPs savings for the model family without accounting for the compute cost of the importance estimation and neural architecture search phases that precede retraining. The text in Section 2.2 describes the importance estimation as using "a small (1024 samples) calibration dataset and only forward propagation passes," and the NAS procedure in Section 2.3 requires "lightweight retraining (∼1.8B tokens in this work)" for 15-18 candidate architectures (Table 19). The paper does not integrate these costs into the FLOPs calculation in Section 4.1, which compares only the raw training FLOPs of from-scratch models against the retraining FLOPs of pruned models:

> "the corresponding cost savings for training the full Nemotron-4 family using our approach is thus 1.8×"

**The consequence.** The actual cost of producing a MINITRON model includes: (1) 1024 forward passes on the 15B model for importance estimation; (2) lightweight retraining of 15-18 candidate architectures at ~1.8B tokens each (totaling ~27-32B tokens for the 8B search and a similar amount for 4B); (3) full retraining of the selected architecture. The NAS phase alone consumes ~27-32B tokens per model — a non-trivial fraction of the 94B tokens used for final retraining (roughly 25-30% overhead). For the 4B model, the iterative process (15B→8B→4B) would require two separate NAS phases and two importance estimation passes. None of these costs appear in the 1.8× savings calculation. A practitioner evaluating whether to adopt this approach needs to compare total end-to-end FLOPs — including search and estimation — against training from scratch, and the headline 1.8× figure overstates the actual savings by an unknown but potentially significant margin (perhaps reducing effective savings to 1.4-1.5×).

**What evidence exists in the paper.** The paper openly acknowledges that 1024 samples are used for importance estimation (Section 2.2, Section 4) and that 15-18 candidates are retrained for 400 steps each (Section 2.3, Figure 9), but never sums these costs or includes them in the FLOPs comparison. Figure 9 shows the retraining curves for all 15 candidates, making the cost visible but unaccounted-for. The FLOPs comparison in Section 4.1 explicitly only includes training steps for the final models.

**Mitigation status.** The paper does not address this gap. It does not propose a cheaper importance estimation procedure (e.g., fewer calibration samples, a learned difficulty predictor) nor a more efficient NAS strategy (e.g., early stopping, parameter-efficient fine-tuning, or progressive elimination of candidates). The suggestion to explore "parameter-efficient fine-tuning techniques such as LoRA" for the lightweight retraining phase (Section 2.3) is mentioned as future work but not implemented. A practitioner adopting this approach would need to add these overheads to their cost model manually.

---

### 6.2 Hard Tasks Outside the Teacher's Capability Range Remain Unsolvable After Compression

**The assumption or constraint.** The compression methodology fundamentally **preserves and reorganizes existing knowledge** from the teacher model — it cannot create new capabilities that the teacher lacks. The paper states this implicitly through its architecture (all MINITRON models are pruned from Nemotron-4 15B and retrained to mimic it) but never explicitly discusses this as a boundary condition. The teacher model's pass@1 or accuracy on a given task defines an upper bound on what its compressed descendants can achieve after retraining.

**The consequence.** This limitation manifests clearly in the MINITRON 4B results (Table 3). On GSM8K (math reasoning), MINITRON 4B achieves 24.1% — well below Phi-2 (55%, trained from scratch on curated data) and Qwen2 1.5B (58.5%). On HumanEval (code generation), MINITRON 4B scores 23.3% vs. Phi-2's 50% and Qwen2's 31.1%. The Nemotron-4 15B teacher itself achieves only 48.5% GSM8K and 35.4% HumanEval (Table 2) — strong but not state-of-the-art on these specific capabilities. The 4B compressed model inherits this relative weakness, and no amount of retraining or distillation can compensate because the teacher simply does not encode the necessary capabilities in its weight structure to transfer. This is not a failure of the compression methodology — it is a fundamental constraint: pruning amplifies existing capability but cannot create it. For tasks where the teacher is weak, compressed models will be proportionally weaker without additional capability injection (e.g., continued pretraining on domain-specific data post-compression).

**What evidence exists in the paper.** Table 3 directly shows this pattern: MMLU and HellaSwag (where the 15B teacher is strong: 66.6% and 84.6% respectively, per Table 2) compress relatively well to 4B (58.6% and 75.0%), while GSM8K and HumanEval (where the teacher is relatively weaker: 48.5% and 35.4%) compress poorly (24.1% and 23.3%). The paper does not discuss or analyze this pattern. Table 2 shows that MINITRON 8B on GSM8K (51.3%) actually exceeds the teacher (48.5%) — the only benchmark where this occurs — which the paper attributes to "distillation on math-heavy data" sharpening the student's reasoning, but this is not analyzed in detail and may not generalize.

**Mitigation status.** Not addressed. The paper does not discuss capability boundaries, does not suggest continued pretraining on domain-specific data for capabilities the teacher lacks, and does not analyze the relationship between teacher capability strength and compression fidelity. A practitioner evaluating this approach for a specific domain would need to check whether the largest available model is sufficiently capable on that domain before committing to pruning-based production — a guideline the paper does not provide.

---

### 6.3 Single Model Family and Architecture — No Evidence of Generalization

**The assumption or constraint.** All experiments use the Nemotron-4 15B model with a specific architecture: 32 layers, 6144 embedding dimension, 48 query heads with 8 key-value groups via grouped-query attention (GQA), 24576 MLP hidden dimension (Table 5). The paper asserts this model is "representative of the capabilities of many contemporary LLMs" (Section 4), but provides no evidence that the best practices transfer to other architectures, model families, or scales.

**The consequence.** Several of the paper's core findings may be architecture-specific rather than universal:

- **Best Practice #4 ("Prefer width pruning over depth for ≤15B"):** This was derived on a 32-layer model. For deeper architectures (70B+ parameters with 80+ layers), layer redundancy may be more extreme — ShortGPT and LaCo demonstrated effective depth pruning on models with more layers, suggesting the depth-vs-width tradeoff could shift as model depth increases. A practitioner with a 70B, 80-layer model cannot assume width pruning is still superior.

- **Best Practice #7 ("Use logit-only distillation when depth isn't reduced significantly"):** This was validated on a model where student and teacher share identical depth (32 layers). Whether logit-only distillation suffices when depth is moderately reduced (e.g., 32→24 layers, not the extreme 32→16 tested in Table 17) is unknown.

- **Grouped-query attention interaction:** The paper notes that head residual preservation "is applied only to the query heads" (Section 2.3) in GQA, but how this technique interacts with standard multi-head attention or multi-query attention is unexplored. A model with dense attention (all heads independent) might respond differently to head pruning and residual preservation.

- **SwiGLU vs. other MLP activations:** The Nemotron-4 uses a specific MLP structure with a particular expansion factor (4×). Whether the activation-based neuron importance metric works equally well with other activations (GELU, ReLU) or different expansion factors (2.5×, 3.5×) is untested.

**What evidence exists in the paper.** None. The paper does not test any model other than Nemotron-4 15B. The baselines in Table 4 compare against methods applied to LLaMA models, but this compares MINITRON pruned from Nemotron-4 against methods pruned from LLaMA — it does not test MINITRON's methodology on LLaMA or prior methods on Nemotron-4. The paper mentions in the acknowledgments that it "gratefully acknowledge[s] the insightful discussion and feedback" from colleagues, but no cross-architecture validation is reported.

**Mitigation status.** The paper does not claim generalization beyond stating the model is "representative." The limitation is partially mitigated by the fact that Nemotron-4 15B uses a standard transformer architecture without exotic components, making the core techniques (activation-based importance, KLD distillation) plausibly transferable. However, the specific best practices — especially the depth-vs-width preference and the sufficiency of logit-only distillation — should be treated as validated only for this architectural regime until replicated on other model families.

---

### 6.4 Distillation vs. Conventional Training Not Compared at Token Parity — Only Iso-Compute

**The assumption or constraint.** The paper's central finding that distillation is "dramatically more effective" than conventional retraining (Table 11) compares approaches under **iso-compute** conditions (matching total FLOPs by giving conventional training more tokens to account for the missing teacher forward pass). Specifically: 150B tokens of conventional training vs. 100B tokens of distillation training. However, the paper does not compare distillation and conventional training at **equal token counts**, leaving open the question of whether conventional training would catch up given enough tokens, or whether distillation provides a permanent accuracy ceiling advantage.

**The consequence.** This matters for resource allocation. If distillation provides a permanent accuracy advantage — the student can reach a higher plateau than conventional training regardless of token budget — then distillation is the unambiguously correct choice and the paper's recommendation is robust. If conventional training would eventually match distillation performance given, say, 300B tokens instead of 150B, then the choice depends on the relative cost of tokens vs. teacher forward passes. The paper's iso-compute comparison conflates these two scenarios. A practitioner with abundant data but limited GPU memory (teacher forward passes increase peak memory) might prefer longer conventional training, while one with limited data but abundant memory might prefer distillation. The paper does not provide the evidence to make this decision.

**What evidence exists in the paper.** Table 11 provides the iso-compute comparison: 24.36% MMLU (random init, 150B tokens) vs. 24.57% (pruned + conventional, 150B tokens) vs. 37.81% (pruned + distill, 100B tokens). This establishes that distillation reaches higher accuracy with fewer tokens under equivalent FLOPs, but does not disentangle the effects of token count vs. training method. The paper also does not report whether the 24.57% conventional training result was still improving at 150B tokens — if the loss curve was still descending, more tokens might close the gap. Figure 6 shows that LM validation loss for pruned models is still decreasing at 400 steps (~1.8B tokens), but this is too small a scale to extrapolate to the 100B+ token regime.

**Mitigation status.** Not addressed. The paper does not discuss this distinction, does not provide a token-parity comparison, and does not analyze convergence behavior for conventional vs. distillation training at scale. The claim that distillation is "clearly ... superior" (Section 4.3) is supported for the iso-compute setting but could be qualified: distillation achieves dramatically better data efficiency, but whether it achieves a higher accuracy ceiling is unknown. A practitioner who can afford very long conventional retraining (trillions of tokens) might achieve equivalent results without implementing the distillation pipeline.

---

### 6.5 Head Residual Preservation Technique Is Unvalidated by Ablation

**The assumption or constraint.** When pruning attention heads, the paper adds the residual information from pruned heads back into surviving heads (Section 2.3):

> "head_i^new = head_i + (head_i − head_{2K−i+1}) for i ∈ [K − (L − K), K]"

The paper claims this "provides a boost to model accuracy in our experiments" but provides **no ablation comparing pruning with vs. without this technique**. The precise benefit — how many accuracy points it contributes, whether it helps more at certain compression rates, whether it interacts with distillation — is entirely unquantified.

**The consequence.** Since the head residual preservation technique is novel (the paper presents it as "an MHA analog of Layer Collapse for depth pruning"), a practitioner implementing this methodology cannot evaluate whether this specific technique is necessary or merely helpful. It adds engineering complexity: the pruning code must correctly identify the symmetric head pairs, compute the residual, and modify the surviving head outputs — all of which must interoperate with the grouped-query attention structure where key-value heads are shared. If the technique provides only a marginal benefit (e.g., 0.5% on downstream benchmarks), a practitioner might reasonably omit it to simplify the implementation. If it provides a substantial benefit (e.g., 5% on MMLU), it is essential. The paper provides no evidence either way.

**What evidence exists in the paper.** The claim in Section 2.3 is unsubstantiated: no table, figure, or quantitative comparison isolates the effect of head residual preservation. The paper compares MINITRON models (which use the technique) against external pruned models (which do not), but this confounds the technique with all other differences in the methodology (activation-based importance, distillation retraining, NAS). The ablation does not appear in the main text or appendix.

**Mitigation status.** Not addressed. The paper does not acknowledge this as a missing ablation or suggest future work to quantify the technique's contribution. For a paper whose core contribution is empirical best practices, the absence of ablation on a non-obvious design choice is a notable gap — especially since Best Practice #2 carefully ablates aggregation functions (Table 13), Best Practice #3 ablates iterative importance (Table 14), and Best Practice #5 ablates distillation loss functions (Tables 15-16), establishing a pattern of thorough ablation that this specific technique breaks.

---

### 6.6 No Dynamic or Adaptive Compression — All Decisions Are Pre-Computed Before Retraining

**The assumption or constraint.** The entire pruning pipeline is **static**: importance scores are computed once on the original 15B model (or once per compression stage for iterative pruning), a fixed architecture is selected via pre-computed NAS, and retraining proceeds with a fixed distillation strategy. The paper does not explore dynamic strategies where pruning decisions, architecture choices, or distillation loss weights adjust during retraining based on the student's evolving capabilities.

**The consequence.** This static approach may leave performance on the table in at least two ways:

- **Pruning-then-training vs. joint optimization:** The importance of specific heads, neurons, or channels may change as the student is retrained to match the teacher. A component that was unimportant at the start of retraining might become critical as other components specialize. The current approach makes pruning decisions based on the teacher's activations (pre-retraining) and never revises them. If retraining could "reactivate" pruned knowledge pathways by reallocating function, the static approach would miss this opportunity.

- **Fixed distillation strategy:** The paper uses a single distillation loss configuration (KLD, logit-only for width-pruned models) throughout all retraining. Curriculum-based or adaptive strategies — for example, starting with intermediate state distillation when the student is far from the teacher and transitioning to logit-only as the student converges — are not explored. The dynamic α weighting (`α = L_logits / L_is`, Section 3) is the only adaptive element, and it is a normalization trick rather than a strategy-level adaptation.

The practical impact is uncertain — the paper's strong results suggest the static approach works well — but a more dynamic approach might enable: (1) higher compression rates, (2) faster convergence during retraining, or (3) better performance on tasks where the teacher-student capability gap is large.

**What evidence exists in the paper.** The paper indirectly provides evidence that static decisions are reasonable: the finding that iterative importance estimation converges to the same result as single-shot (Table 14) suggests importance rankings are stable; the finding that lightweight retraining stabilizes architecture rankings (Figure 9) suggests the architecture choice is robust. However, these findings validate static decisions against other static alternatives, not against dynamic strategies. There is no comparison of, for example, pruning 50% of a model dynamically during retraining vs. pruning 50% pre-retraining.

**Mitigation status.** Not addressed. The paper does not discuss dynamic or adaptive compression strategies, nor does it suggest them as future work. The iterative compression strategy (15B→8B→4B) is the closest the paper comes to dynamic decision-making, but it operates at coarse granularity (full retraining between stages) and the pruning decisions within each stage are still static. A practitioner exploring more aggressive compression rates (e.g., 6-8× rather than 2-4×) might find that static pruning fails and dynamic strategies are necessary, but the paper provides no guidance on this regime.

## 7. Implications and Future Directions
- Field impact
  - A practical path to economical model families: train one strong model once, then derive smaller variants with minimal extra data. The detailed best practices (Section 4.1) make pruning+KD actionable for teams beyond major labs.
- Applications
  - Deployable small/medium LLMs for edge or latency‑constrained environments.
  - Rapid “spin‑offs” of task‑ or domain‑specific models by pruning a generalist teacher and retraining on small domain data.
  - As shown with `MINITRON 4B‑instruct`, instruction‑tuned compact models can be strong for function calling and RAG (Tables 8–9).
- Research directions
  - Multi‑teacher or mixture‑of‑experts distillation for pruned students.
  - Automatic calibration set selection to improve importance estimates for target domains.
  - Extending best practices to larger teachers and to architectures with long‑context mechanisms.
  - Joint compression with quantization and low‑rank adaptation; the paper notes LoRA could be used during the lightweight search phase (Section 2.3).
  - Theoretical understanding of why width pruning overtakes depth pruning after short retraining (Table 10, Figure 6).

In sum, this paper contributes a carefully tested, end‑to‑end procedure to compress LLMs—spanning importance scoring, pruning across multiple axes, efficient retraining by KD, and practical architecture search—backed by extensive ablations and competitive downstream results.
