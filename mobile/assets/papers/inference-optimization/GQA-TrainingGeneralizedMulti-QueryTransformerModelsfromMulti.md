# GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

**ArXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)

## 🎯 Pitch

This paper introduces a novel, low-cost uptraining method to convert existing multi-head attention (MHA) language model checkpoints into faster multi-query attention (MQA) and a new grouped-query attention (GQA) scheme—requiring only about 5% of the original pre-training compute. GQA smartly shares keys and values among small groups of query heads, dramatically reducing inference latency and memory overhead while nearly matching the output quality of full MHA models—making it a practical breakthrough for deploying high-performance, efficient large language models.

---

## 1. Executive Summary

This paper introduces **grouped-query attention (GQA)**, an interpolation between multi-head attention and multi-query attention that uses an intermediate number of key-value heads—more than the single head of multi-query attention (MQA) but fewer than the full set of query heads in multi-head attention (MHA)—to recover quality lost by MQA while retaining most of its inference speedup. The authors also propose a recipe for uptraining existing multi-head checkpoints into MQA or GQA models using only 5% of the original pre-training compute, demonstrated by converting public T5.1.1 Large and XXL checkpoints and evaluating on summarization, translation, and question-answering benchmarks. Uptrained GQA achieves quality close to MHA-XXL with speed comparable to MQA—for instance, GQA-8-XXL reaches 47.1 average performance versus MHA-XXL's 47.2 while reducing per-sample inference time from 1.51 ms to 0.28 ms, establishing that the quality-efficiency gap between MHA and MQA can be substantially closed without training a separate model from scratch.

## 2. Context and Motivation

### The Core Problem: Transformer Decoder Inference Is Memory-Bandwidth-Bound

The fundamental inefficiency this paper targets is not the raw computational cost (FLOPs) of running a Transformer model, but rather a more specific hardware bottleneck: **loading keys and values from memory during autoregressive decoding**. This might seem like a niche architectural concern, but it dominates the wall-clock time of generating text with large language models in production.

To understand why, consider what happens at each step of autoregressive decoding. The model predicts one token, which then becomes part of the input for predicting the next token. For self-attention layers, every previously generated token's key and value vectors must be fetched from the **KV-cache** (the stored keys and values from all prior time steps) to compute attention scores against the current query. For cross-attention layers in encoder-decoder architectures, all encoder output keys and values must be loaded at every decoder step.

The problem is that loading these keys and values from GPU memory is limited by **memory bandwidth**—the rate at which data can be transferred between memory and compute units—rather than by the speed of the arithmetic itself. As Shazeer (2019) originally identified and Pope et al. (2022) and de Jong et al. (2022) subsequently confirmed, this memory bandwidth overhead often dominates inference latency, especially when generating long sequences. The computation itself is fast; the bottleneck is *feeding the computation the data it needs*.

The paper opens by grounding itself in this well-documented phenomenon:

> "Autoregressive decoder inference is a severe bottleneck for Transformer models due to the memory bandwidth overhead from loading decoder weights and all attention keys and values at every decoding step"

This context is essential because it frames the *type* of optimization the paper pursues. The goal is not to reduce total arithmetic operations (as with sparse attention or distillation) but specifically to reduce the volume of data that must be transferred from memory at each decoding step. The size of the KV-cache—determined by the number of key and value heads multiplied by the per-head dimensionality and sequence length—is the primary target.

### The Multi-Query Attention Solution and Its Quality Deficit

Shazeer (2019) proposed a direct solution: **multi-query attention (MQA)**. In standard multi-head attention (MHA), each of the $H$ query heads has its own dedicated key and value head, meaning there are $H$ distinct key matrices $W_i^K$ and value matrices $W_i^V$ projecting the input into $H$ different key and value spaces. MQA collapses this structure: all query heads share a single key head and a single value head. The key-value cache is reduced by a factor of $H$—a dramatic reduction in memory bandwidth overhead—while the query heads remain independent, preserving the model's capacity to attend to different representational subspaces through the queries.

This works. MQA achieves substantial inference speedups, and some major models adopted it, including PaLM (Chowdhery et al., 2022). However, the paper identifies a critical limitation that prevents universal adoption:

> "MQA can lead to quality degradation and training instability"

The single shared key-value representation imposes an information bottleneck. With only one key and value projection for all queries, the model loses the ability to represent different attention patterns in parallel—every query head sees the same keys and values, constraining the diversity of attention the model can express. Empirically, this manifests as lower downstream task performance compared to an MHA model of the same size.

Moreover, the training instability is a practical concern the authors document explicitly in Appendix A. Training MQA models from scratch produced "frequent loss spikes" during pre-training and caused fine-tuning divergence on long-input tasks. Uptrained MQA models (converted from MHA checkpoints) are more stable but still display "high variance," requiring averaging over multiple fine-tuning runs. This instability makes MQA a less attractive option for practitioners who want reliable training and consistent results.

### The Deeper Issue: Training Separate Models for Different Use Cases

The quality-speed trade-off between MHA and MQA creates an organizational and economic problem that goes beyond architecture design. Ideally, a research team or company would have:

- **High-quality models** (MHA) for applications where accuracy is paramount and latency is less constrained—evaluation benchmarks, offline data generation, quality-sensitive user applications.
- **Fast models** (MQA) for latency-sensitive applications—real-time inference, interactive systems, high-throughput batch processing.

Training two entirely separate models from scratch doubles the pre-training cost, which for large language models can run into millions of dollars. The authors point out that this may simply "not be feasible" for many organizations. Moreover, for publicly available models like T5 (Raffel et al., 2020) and LLaMA (Touvron et al., 2023), which use MHA and do not provide MQA variants, users face a binary choice: accept the inference cost of the MHA checkpoint or attempt to modify the architecture on their own with no guidance on whether the resulting model will retain quality.

This practical constraint—that many widely used open-source and industry models are MHA-based, and retraining MQA equivalents is prohibitively expensive—is what motivates the uptraining approach. The ideal scenario is one where a *single* pre-training run yields both objectives: a high-quality MHA model and, through a cheap post-hoc conversion process, a fast MQA model. The paper's first contribution—a recipe for uptraining existing MHA checkpoints into MQA using only 5% of the original pre-training compute—directly addresses this economic constraint.

### Where Prior Approaches Fall Short

The paper situates itself relative to a broader landscape of inference acceleration techniques, observing gaps that GQA specifically fills:

**MQA is the most direct attack on the KV-cache bottleneck but is a blunt instrument.** Reducing from $H$ key-value heads to 1 throws away the most representational capacity while achieving the maximum memory bandwidth reduction. This binary choice—all heads or one head—leaves a large unexplored design space between the two extremes. Prior work had not systematically investigated whether an intermediate number of key-value heads could recover most of the quality loss while preserving most of the speed gain.

**Other acceleration techniques target different bottlenecks or have different cost profiles.** FlashAttention (Dao et al., 2022) restructures the attention computation to avoid materializing the full quadratic attention matrix in memory, reducing memory usage and improving training speed, but it does not reduce the size of the KV-cache or the bandwidth required to load it during inference. Quantization (Dettmers et al., 2022; Frantar et al., 2022) reduces the precision of weights and activations, shrinking memory footprint multiplicatively, but does not change the fundamental number of key-value heads. Model distillation (Hinton et al., 2015; Gou et al., 2021) trains a smaller model on a larger model's outputs, reducing both FLOPs and memory, but requires the original large model's availability and a full distillation training process. Speculative sampling (Chen et al., 2023; Leviathan et al., 2022) uses a small model to propose multiple tokens that a larger model scores in parallel, improving throughput without changing the architecture, but adds system complexity and a second model in the serving pipeline. None of these methods directly alter the attention mechanism's architecture to reduce KV-cache size while preserving model quality, which is the specific lever this paper pulls.

**Prior grouping approaches were not targeting the KV-cache bottleneck.** The paper acknowledges that grouping attention heads for computational efficiency has been explored before—Park et al. (2020), Luo et al. (2022), and Ni et al. (2023) all group attention heads in various ways—but notes that these works did not focus specifically on the key-value heads that determine memory bandwidth overhead. The crucial distinction is between grouping that reduces compute (which these prior works targeted) and grouping that reduces KV-cache size (which GQA targets). Since the number of key-value heads directly determines how much data must be loaded from memory per token, reducing $K$ and $V$ heads yields a more direct speedup than reducing query heads alone.

**No existing method provided a cheap conversion path from MHA to fast inference.** The uptraining approach—converting a pre-trained checkpoint and then continuing training on a small fraction of the original compute—had been demonstrated for sparsely activated mixture-of-experts models by Komatsuzaki et al. (2022), who uptrained dense T5 checkpoints into MoE architectures. The authors explicitly credit this as inspiration. But the concept had not been applied to attention head structure, specifically to converting MHA models into MQA or GQA variants. This gap meant that organizations with MHA checkpoints had no established procedure for obtaining architecturally faster versions.

### The Scaling Argument: Why GQA Becomes More Important for Larger Models

An observation in the paper that is easy to miss but conceptually important for motivation: **MQA's relative advantage diminishes as models scale, and GQA corrects for this.** The argument runs as follows (Section 2.2):

Larger models generally increase the number of attention heads $H$. Since MQA reduces the number of key-value heads from $H$ to 1, the proportional reduction in KV-cache size grows with the number of heads. This sounds like good news for MQA, but the paper identifies two countervailing factors:

1. **Memory bandwidth overhead from attention shrinks relative to other costs as models grow.** The KV-cache scales with model dimension $d$, while model FLOPs and parameters scale with $d^2$. As models get larger, the quadratic scaling of compute and parameters outpaces the linear scaling of the KV-cache, meaning the relative bottleneck shifts from memory bandwidth toward raw computation. A flat reduction from $H$ to 1 key-value heads provides less relative benefit for a 100-head model than for a 16-head model, because on the 100-head model, attention memory bandwidth was already a smaller fraction of total inference cost.

2. **Standard model parallelism replicates key-value heads.** When large models are sharded across multiple devices (model parallelism), the single key and value head in MQA must be replicated across all partitions (Pope et al., 2022). This replication wastes communication and memory bandwidth that could otherwise be spent on useful computation—each device stores a copy of the shared key-value head, undoing some of the original size reduction. With more key-value heads (as in GQA), the heads are naturally distributed across partitions, eliminating this redundant replication.

The paper draws the implication:

> "GQA lets us keep the same proportional decrease in bandwidth and capacity as model size increases."

Rather than applying a fixed reduction ($H \rightarrow 1$) regardless of model size, GQA scales the number of groups (and therefore key-value heads) with the model, maintaining a consistent proportional reduction. This makes GQA a more principled solution than MQA for the large-model regime that dominates modern LLM deployment.

### How This Paper Positions Itself

The authors frame GQA as directly occupying the unexplored interpolation between MHA and MQA, not as a competitor to the broader set of inference acceleration techniques. The paper identifies three gaps that, taken together, define its contribution space:

1. **An architecture gap:** MHA has $H$ key-value heads (maximum quality, maximum KV-cache); MQA has 1 (minimum KV-cache, reduced quality). The intermediate space—2 to $H-1$ key-value heads—is uncharacterized. GQA populates this space and empirically demonstrates that intermediate configurations capture most of MQA's speed while recovering most of MHA's quality.

2. **A conversion gap:** Prior work showed that sparse architectures could be obtained from dense checkpoints via uptraining. But no recipe existed for converting attention structure without training from scratch. The paper provides this recipe, with specific design choices (mean pooling of key-value heads, 5% continued pre-training) and empirical validation of their effectiveness.

3. **A scaling gap:** MQA's fixed single head is an increasingly poor fit for larger models due to diminishing relative bandwidth benefit and model parallelism replication. GQA provides a head count that scales proportionally with model size, maintaining a consistent quality-speed trade-off curve regardless of model scale.

The paper's title itself—"Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"—encapsulates this positioning. "Generalized" signals interpolation between the MHA and MQA extremes. "From Multi-Head Checkpoints" signals the practical conversion pipeline that avoids training a separate model from scratch. Together, these three words describe a position that is architecturally novel (the interpolation), procedurally novel (the uptraining recipe), and practically motivated (cheaper than training two models).

## 3. Technical Approach

### 3.1 Reader Orientation

This paper is primarily a **systems and architecture paper** that proposes both a new attention mechanism (GQA) and a cost-effective recipe for retrofitting existing pre-trained models with that mechanism. The core idea is to interpolate between multi-head attention (H key-value heads, maximum quality, maximum memory cost) and multi-query attention (1 key-value head, maximum speed, degraded quality) by introducing an intermediate number of key-value heads—grouped-query attention—and to show that this interpolation can be achieved starting from a pre-trained MHA checkpoint with only 5% of additional pre-training compute.

The problem being solved is the **practical impossibility of training separate high-quality (MHA) and fast-inference (MQA) models from scratch** for every deployment scenario. The shape of the solution is a two-stage pipeline: (1) convert an existing MHA checkpoint to GQA (or MQA) through a simple mean-pooling operation on the key-value projection matrices, and (2) continue pre-training for a small additional budget to let the model adapt to its new attention structure.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has three major components:

1. **The Multi-Head Attention (MHA) Checkpoint** — a fully pre-trained Transformer model (here, T5.1.1 Large and XXL) with standard multi-head attention in all decoder self-attention and cross-attention layers. This checkpoint serves as the starting point and represents the quality ceiling.

2. **The Checkpoint Conversion Module** — a simple, deterministic operation that restructures the key and value projection matrices. For MQA, it mean-pools all H key-value heads into a single head. For GQA with G groups, it partitions the H query heads into G equally sized groups and mean-pools the key and value heads within each group. The query projection matrices and all other parameters (feed-forward layers, layer norms, encoder self-attention) remain unchanged.

3. **The Uptraining Process** — continued pre-training of the converted checkpoint on the original pre-training corpus and recipe for a fraction α (0.05, or 5%) of the original training steps. This allows the model to learn to use its new attention structure effectively without requiring full re-training.

Information flows as follows: the pre-trained MHA checkpoint enters the system → the conversion module restructures key and value projection matrices via mean pooling → the model undergoes 5% continued pre-training on the original data → the resulting checkpoint is fine-tuned on downstream tasks and evaluated against the original MHA model and a full MQA conversion.

### 3.3 Roadmap for the Deep Dive

- **First**, the checkpoint conversion mechanism — what *exactly* happens to the attention matrices, why mean pooling is chosen over alternatives like selecting a single head or random initialization, and what mathematical operation this corresponds to. This is the linchpin that makes uptraining possible.
- **Second**, the uptraining procedure — the pre-training continuation recipe, the choice of α = 5%, and the diminishing returns observed beyond this budget. This explains how the model adapts to its new structure.
- **Third**, grouped-query attention as an interpolation mechanism — the formal definition of GQA, how G groups are constructed, the relationship between GQA-1 (MQA), GQA-H (MHA), and intermediate G values, and the scaling argument for why intermediate groups are particularly valuable for larger models.
- **Fourth**, the key design decisions and their justifications — why 8 groups for the XXL model, why only decoder self-attention and cross-attention are modified (not encoder self-attention), why the T5.1.1 architecture, and the specific hyperparameters for both uptraining and fine-tuning.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **systems and architecture paper** whose core idea is that the quality-efficiency trade-off between MHA and MQA can be substantially closed by introducing an intermediate number of key-value heads (GQA), and that this transition can be achieved from an existing MHA checkpoint through a cheap conversion-and-uptraining pipeline rather than training from scratch.

---

#### Checkpoint Conversion: From Multi-Head to Grouped-Query (or Multi-Query) Attention

The conversion operation is conceptually simple but its specific implementation—mean pooling—is non-obvious and empirically motivated. Starting from a standard multi-head attention layer, we need to transform the key and value projections into a grouped structure.

**Standard multi-head attention structure.** In MHA with $H$ heads, the input $x \in \mathbb{R}^{d_{\text{model}}}$ is projected through three sets of matrices—query ($W^Q$), key ($W^K$), and value ($W^V$)—each producing $H$ independent projections. The key projection matrix $W^K \in \mathbb{R}^{d_{\text{model}} \times (H \cdot d_k)}$ can be viewed as $H$ separate matrices $\{W_1^K, W_2^K, ..., W_H^K\}$, each of shape $d_{\text{model}} \times d_k$, where $d_k$ is the per-head key dimensionality. Similarly, $W^V$ decomposes into $\{W_1^V, W_2^V, ..., W_H^V\}$ with per-head dimensionality $d_v$. Each head $i$ computes:

$$\text{head}_i = \text{Attention}(x W_i^Q, x W_i^K, x W_i^V)$$

The critical point for conversion is that the $H$ key and value projection matrices within a layer encode different transformation functions learned during pre-training. The question is: how do we combine these into fewer matrices while preserving as much of the learned information as possible?

**Conversion for MQA (GQA-1).** The conversion to multi-query attention collapses all $H$ key heads into a single key head, and all $H$ value heads into a single value head. Let $\{W_1^K, ..., W_H^K\}$ be the key projection matrices from the MHA checkpoint. The converted single key projection matrix is:

$$W_{\text{MQA}}^K = \frac{1}{H} \sum_{i=1}^{H} W_i^K$$

where each $W_i^K$ is the key projection matrix for head $i$ in the original MHA, and the sum is element-wise (mean pooling). The same operation is applied to the value heads: $W_{\text{MQA}}^V = \frac{1}{H} \sum_{i=1}^{H} W_i^V$.

**What the operation computes:** for each element (each weight parameter in the projection matrices), we take the arithmetic mean across all H heads. If a particular input dimension contributed strongly to key projections across most heads, its weight in the pooled matrix remains high. If heads disagreed (some had strongly positive, others strongly negative weights), the average captures the central tendency. The result is a single key projection matrix and a single value projection matrix that represent the "consensus" of the original H heads.

**Why this form over alternatives?** The paper compares three conversion methods in Figure 4:

- **Mean pooling** (the proposed method): preserves information from all H heads by averaging, giving each head equal vote in the pooled representation. The intuition is that all heads learned useful projection functions during pre-training, and discarding any of them would lose information.
- **First head selection**: $W_{\text{MQA}}^K = W_1^K$, simply taking the first head's projection and discarding the other $H-1$ heads. This throws away the information learned in heads 2 through H, which the empirical results show performs worse than mean pooling.
- **Random initialization**: $W_{\text{MQA}}^K$ is initialized with random weights (following the default network initialization scheme) and not derived from the pre-trained checkpoint at all. This discards all pre-trained key-value projection knowledge, forcing the model to learn entirely new projections during uptraining. Empirically, this performs worst.

The empirical ordering—mean > first > random—exactly follows the degree of information preservation from the pre-trained model. Mean pooling preserves all heads' contributions; first-head selection preserves one head's contribution; random initialization preserves nothing.

**Conversion for GQA with G groups.** The generalization to grouped-query attention proceeds similarly. The H query heads are partitioned into G groups of equal size (assuming H is divisible by G). Let group $g$ contain query head indices $\{i_1, i_2, ..., i_{H/G}\}$. The key and value projection matrices for group $g$ are:

$$W_{g}^K = \frac{G}{H} \sum_{j \in \text{group}_g} W_j^K$$

where the sum is over the $H/G$ original key projection matrices belonging to group $g$. The same is done for values: $W_{g}^V = \frac{G}{H} \sum_{j \in \text{group}_g} W_j^V$.

The grouping of query heads is contiguous—heads are partitioned sequentially into groups. For example, with H = 64 query heads and G = 8 groups, each group contains 8 contiguous query heads. The paper does not explore non-contiguous grouping (e.g., interleaving), implying that the query heads' ordering in the original model already reflects some structure (heads that are "near" each other in index space may have similar functions or be learned together).

**Which layers are converted.** The conversion applies to decoder self-attention and cross-attention layers, but **not** to encoder self-attention. The rationale is architectural and hardware-motivated: encoder representations are computed in parallel (the entire input sequence is processed in one forward pass), so the encoder's KV-cache is not repeatedly loaded across autoregressive steps. Memory bandwidth from loading keys and values is therefore "generally not the primary bottleneck" for the encoder. Modifying encoder self-attention would change model quality without delivering inference speed benefits, so it is left unchanged.

**What remains untouched.** Critically, the query projection matrices $W_i^Q$ for all H heads are preserved exactly as they were in the MHA checkpoint. This is essential: the model's ability to attend to different representational subspaces is driven by query diversity, and by keeping all H query heads intact, GQA preserves the model's capacity to form diverse attention patterns. The bottleneck is only on the key and value side—all queries attend over the same (or grouped) key-value representations, but they can still project those representations differently through their own query transformations.

---

#### The Uptraining Procedure: Continued Pre-Training After Conversion

Once the checkpoint has been converted (key and value heads restructured via mean pooling), the model requires additional training to adapt to its new attention structure. This second stage is what the paper terms "uptraining."

**What "uptraining" means operationally.** The converted checkpoint is placed back into the pre-training pipeline—same dataset, same optimizer, same hyperparameters and learning rate schedule as the original pre-training—and training continues for a fraction α of the original total training steps. For α = 0.05 (5%), this means if the original T5-XXL was trained for S steps, the converted checkpoint undergoes an additional 0.05 × S steps. The training cost is therefore approximately 5% of the original pre-training compute.

**Key hyperparameters and setup.** The paper states:

> "We use the Adafactor optimizer with the same hyperparameters and learning rate schedule as T5 (Raffel et al., 2020)."

This means the uptraining inherits all original T5 pre-training settings: Adafactor optimizer (a memory-efficient variant of Adam designed for large models), inverse square root learning rate schedule with warmup, and the same pre-training objective (a span corruption denoising objective in the T5 framework). The pre-training data is the same corpus used to train the original T5 checkpoints (C4, a cleaned version of Common Crawl).

**Uptraining budget and diminishing returns.** The paper sweeps α ∈ {0, 0.01, 0.05, 0.10} (Figure 5). Key observations:

- At α = 0 (no uptraining, immediate evaluation after conversion), GQA already achieves "reasonable performance" while MQA essentially fails—MQA requires uptraining to be useful.
- At α = 0.05 (5%), both MQA and GQA show significant gains from their α = 0 performance. GQA reaches performance close to the original MHA model. MQA reaches usable but clearly lower performance.
- At α = 0.10 (10%), both architectures show further improvements but with "diminishing returns"—the gain from 5% to 10% is much smaller than from 0% to 5%.

The paper selects α = 0.05 as the primary operating point, reflecting a pragmatic cost-quality trade-off: 5% additional compute is cheap enough to be practical (for T5-XXL, approximately 600 TPUv3 chip-days), while capturing most of the achievable quality recovery.

**Why continued pre-training works.** The intuitive mechanism is that the conversion operation (mean pooling) produces key and value projections that are reasonable approximations of useful projections—they represent the "average" of the previously learned heads—but the model's query heads and feed-forward layers were optimized to work with H distinct key-value representations, not G pooled ones. The queries expect certain key projection geometries to produce meaningful attention scores. The uptraining allows the model to:

1. **Adjust query projections** to work effectively with the pooled key-value spaces.
2. **Adapt feed-forward layer representations** to receive properly structured attention outputs.
3. **The pooled keys and values themselves may shift** during training to better serve all queries within their group.

The paper does not provide ablations on what specifically changes during uptraining (e.g., whether query vs. key-value vs. FFN parameters move most), leaving this as a mechanistic black box with empirical validation.

**Training stability observations.** Appendix A notes that uptrained MQA models are "more stable" than MQA models trained from scratch but "still display high variance." For MQA, the paper reports average performance over three fine-tuning runs on unstable tasks. Uptrained GQA models are reported as "stable" with no special variance mitigation needed. This stability difference is attributed to the stricter information bottleneck in MQA (single key-value head for all queries) versus GQA (multiple key-value heads, each serving a subset of queries), but the root cause is not deeply investigated beyond this observation.

---

#### Grouped-Query Attention: The Interpolation Mechanism

Grouped-query attention is formally a generalization of both multi-head and multi-query attention. The key parameter is G, the number of key-value groups.

**Formal definition of GQA.** Given H query heads (each with its own query projection $W_i^Q$), these heads are partitioned into G groups. For each group $g \in \{1, ..., G\}$, there is a single key projection matrix $W_g^K$ and a single value projection matrix $W_g^V$. All query heads within group $g$ share these same key and value projections. The attention computation for query head $i$ in group $g$ is:

$$\text{head}_i = \text{Attention}(x W_i^Q, x W_g^K, x W_g^V)$$

The key-value cache for this layer stores G key-value pairs per token position (one per group), rather than H (as in MHA) or 1 (as in MQA).

**The interpolation property.** The notation GQA-G refers to grouped-query attention with G groups. The boundary cases define the interpolation:

- **GQA-1** (G = 1): all H query heads are in a single group, sharing one key head and one value head. This is exactly multi-query attention (MQA).
- **GQA-H** (G = H): each query head is its own group, with its own unique key and value heads. This is exactly multi-head attention (MHA).

For any G such that $1 < G < H$, the architecture is intermediate: more key-value heads than MQA (so more representational capacity and less information bottleneck) but fewer than MHA (so smaller KV-cache and lower memory bandwidth). The quality-speed curve is therefore interpolated between the two extremes, with G serving as the control knob.

**The group size is H/G query heads per group.** For the T5-XXL experiments with H = 64 heads and G = 8 groups, each group contains 8 query heads that share a single key-value head. The KV-cache size is reduced by a factor of H/G = 8 compared to MHA, or equivalently, it is G = 8 times larger than MQA's cache.

**The head grouping is structural, not learned.** The assignment of query heads to groups is fixed: heads are partitioned contiguously in index order. The paper does not investigate whether allowing the model to learn which query heads should share key-value representations (e.g., through clustering or learned routing) would improve performance. The contiguous partitioning is a simple, deterministic choice that adds no parameters or training complexity.

**Why the GQA interpolation makes sense for large models.** The paper introduces a scaling argument in Section 2.2 that motivates why GQA (with G > 1) becomes increasingly important as model size grows:

1. **Number of heads scales with model size.** Larger models have more heads (e.g., T5-XXL has 64, but a hypothetical larger model might have 128 or 256). MQA reduces from H to 1 regardless of H, so the proportional reduction in KV-cache grows with H—but this only captures part of the picture.

2. **KV-cache overhead shrinks relative to total cost.** The KV-cache size per layer per token scales as $G \cdot d_k + G \cdot d_v$ (key-value heads × per-head dimensionality). Since per-head dimensionality typically remains constant as model size increases (more heads, not larger heads), the total KV-cache scales roughly linearly with G. However, model parameters and FLOPs scale with $d_{\text{model}}^2$, where $d_{\text{model}}$ grows with model size. As models get larger, the fraction of total inference cost attributable to KV-cache loading decreases, meaning that an aggressive reduction (H → 1) provides less relative speed benefit.

3. **Model parallelism waste.** In large models split across multiple devices, MQA's single key-value head is replicated on every device partition, wasting communication. With GQA, the G key-value heads can be distributed across devices, eliminating this redundancy.

The result: GQA "lets us keep the same proportional decrease in bandwidth and capacity as model size increases" by setting G to grow proportionally with the number of heads—for example, always using 8 groups regardless of whether the model has 64 or 256 heads, maintaining a fixed compression ratio rather than a fixed absolute head count.

---

#### The Key Design Decisions and Their Justifications

**Decision 1: Choosing G = 8 for T5-XXL.**

The paper sweeps G ∈ {1, 4, 8, 16, 32, 64} for the T5-XXL model (which has H = 64 heads) and evaluates inference time (Figure 6). The key observation: increasing G from 1 (MQA) to 8 adds only "modest" additional inference time, but moving further toward H (MHA) incurs "increasing cost." The authors select G = 8 as "a favorable middle ground."

The inference time trend in Figure 6 is not linear: the jump from G = 1 to G = 8 is small, and the incremental cost per additional group increases as G approaches H. This is because the memory bandwidth overhead from the KV-cache depends on total cache size, which scales with G, but the overhead is not the sole cost—there is a baseline cost from loading model weights and performing computations that is independent of G. Adding a few key-value heads (from 1 to 8) adds cache size but does not yet dominate the total. As G approaches 64, the KV-cache becomes large enough that memory bandwidth dominates, and further increases in G directly increase latency proportionally.

For G = 8, the KV-cache is 8× larger than MQA but 8× smaller than MHA, and the inference time is 0.28 ms vs. 0.24 ms for MQA and 1.51 ms for MHA—capturing approximately 97% of MQA's speed while substantially recovering quality.

**Decision 2: Applying GQA only to decoder self-attention and cross-attention, not encoder self-attention.**

The stated rationale is that encoder representations are computed in parallel—all input tokens are processed simultaneously in one forward pass—so the encoder does not suffer from the autoregressive KV-cache loading bottleneck. Since the entire motivation for reducing key-value heads is reducing memory bandwidth during autoregressive step-by-step decoding, modifying the encoder would change model quality without any inference speed benefit. This is a clean application of the principle: optimize only where the bottleneck exists.

**Decision 3: Using the T5.1.1 encoder-decoder architecture rather than a decoder-only model.**

The choice of T5 is partly historical (public checkpoints were available) and partly pragmatic (encoder-decoder models have separate self-attention and cross-attention, allowing fine-grained analysis of where attention modifications matter). The paper acknowledges the limitation: "Recently, decoder-only models are extremely popular, and since these models do not have separate self-attention and cross-attention, we expect GQA to have a stronger advantage over MQA." This is because in a decoder-only model, all attention is self-attention—every layer's KV-cache contributes to memory bandwidth overhead—making the total cache reduction from GQA proportionally larger.

**Decision 4: Using Adafactor with the original T5 hyperparameters for uptraining.**

The paper does not tune optimizer hyperparameters for uptraining; it simply reuses the original T5 pre-training recipe. This is a deliberate simplification: the claim is that uptraining works *without* needing to re-optimize the training procedure, making it a drop-in procedure for any existing T5 checkpoint. Whether this generalizes to other model families (which may have different optimizers, schedules, or architectures) is untested, but within the T5 framework, it demonstrates that the model adapts to the new attention structure using the same training dynamics that produced the original MHA checkpoint.

**Decision 5: Uptraining proportion α = 0.05 as the primary operating point.**

Figure 5 shows the performance curve: GQA after conversion (α = 0) already performs reasonably well; MQA does not. At α = 0.05, both architectures have captured most of their asymptotically achievable performance. At α = 0.10, gains are small. The selection of 5% balances cost (600 additional TPUv3 chip-days for T5-XXL) against quality recovery. The paper does not claim that 5% is universally optimal—it may depend on model size, architecture, and the gap between original quality and desired quality—but demonstrates it as a practical working point.

**Decision 6: Mean pooling over first-head selection or random initialization.**

The empirical results in Figure 4 establish the ordering, but the paper also provides an intuitive explanation: "results are ordered by the degree to which information is preserved from the pre-trained model." Mean pooling preserves all heads' contributions, first-head selection preserves one, and random initialization preserves none—and the quality after uptraining follows this ordering. This is consistent with the broader uptraining philosophy: the closer the converted checkpoint is to the original (in terms of functional behavior), the less the model needs to re-learn during uptraining.

**Decision 7: Contiguous grouping of query heads (rather than learned or interleaved).**

The paper does not explore alternative group construction methods. The contiguous assignment (heads 1–8 in group 1, heads 9–16 in group 2, etc.) is the simplest possible scheme and adds no complexity. If non-contiguous grouping or learned routing among groups would improve quality, this remains an open question, but the paper's goal is to establish that GQA works at all and to characterize the quality-speed trade-off, not to optimize group construction.

**Decision 8: Keeping query projections unchanged during conversion.**

The conversion only modifies key and value projections. Queries remain exactly as trained in the original MHA model. This is critical: the uptraining burden falls entirely on adapting queries to new key-value projections and on adapting downstream processing. If queries were also pooled or modified, the information loss would be larger and uptraining would need to recover more ground. By preserving queries, the model retains its ability to form diverse attention patterns; the limitation is that all queries within a group must express those patterns over the same key-value representation.

---

#### Fine-Tuning and Evaluation Setup

After uptraining, the model is fine-tuned on each downstream task separately. The fine-tuning procedure uses task-specific hyperparameters that differ from pre-training:

**Fine-tuning hyperparameters.** The paper states:

> "For fine-tuning, we use a constant learning rate of 0.001, batch size 128, and dropout rate 0.1 for all tasks."

This represents a shift from the Adafactor optimizer with inverse square root schedule used in pre-training and uptraining, to a constant learning rate optimizer (not explicitly named, but the paper implies standard Adam or similar). The constant learning rate and 0.1 dropout are standard T5 fine-tuning settings.

**Task-specific input/output lengths.** Different tasks have different sequence length requirements:
- CNN/Daily Mail and WMT: input length 512, output length 256
- Other summarization datasets (arXiv, PubMed, MediaSum, MultiNews): input length 2048, output length 512
- TriviaQA (question answering): input length 2048, output length 32

These lengths reflect the nature of the tasks: summarization of long documents (arXiv, PubMed, MultiNews) requires processing full papers, while short-answer QA (TriviaQA) needs only brief outputs. The variation in input length is significant for the paper's claims because the memory bandwidth overhead from loading keys and values scales with sequence length—longer inputs mean larger KV-caches and larger benefits from reducing the number of key-value heads.

**Checkpoint selection and decoding.** Training continues "until convergence" and the checkpoint with highest dev performance is selected. At inference time, greedy decoding is used (no beam search, no sampling). This removes any confounding from decoding strategies and isolates the architecture's effect on inference speed and quality.

---

#### Inference Timing Methodology

The timing measurements are carefully controlled to isolate the architectural effect on latency.

**Hardware and configuration.** Timing is measured on TPUv4 chips, with 8 TPUs per experiment. The batch size is set to "the largest batch size that fits up to 32 per TPU," meaning different models may use different batch sizes depending on their memory footprint. Parallelization (model and data parallelism) is "optimized separately for each model." This means the reported per-sample times reflect the best achievable latency for each model under its optimal serving configuration, not a fixed parallelization scheme.

**Metric.** Time per sample per TPUv4 chip, measured by xprof (a Google TPU profiling tool). This is reported in milliseconds (ms) for the main results in Table 1 (with input length 2048 and output length 512) and in seconds for the GQA group ablation in Figure 6.

**Why per-sample time is the relevant metric.** The paper's central claim is about inference efficiency, not training efficiency. Per-sample time directly captures what a user or downstream application experiences—the latency between providing an input and receiving the output. By measuring on identical hardware with per-model-optimized serving configurations, the comparison reflects real-world deployment conditions.

**The timing numbers.** For T5-XXL models (Table 1), with input length 2048 and output length 512:
- MHA-XXL: 1.51 ms per sample
- MQA-XXL: 0.24 ms per sample (6.3× faster than MHA)
- GQA-8-XXL: 0.28 ms per sample (5.4× faster than MHA, only 1.17× slower than MQA)
- MHA-Large: 0.37 ms per sample (as a reference point for a smaller MHA model)

The GQA-8 model achieves 97% of MQA's speed (0.24 vs. 0.28 ms) while closing most of the quality gap to MHA. This is the core quantitative trade-off that the paper establishes.

## 4. Key Insights and Innovations

### Innovation 1: Test-Time Compute Scaling Is Not Uniform—It Depends on Prompt Difficulty

The paper's central conceptual contribution is the empirical discovery that **the optimal test-time compute strategy depends on the difficulty of the prompt being solved**, and that ignoring this heterogeneity leaves enormous efficiency on the table. This is not merely an observation that "hard problems need more compute"—that would be trivial. The deeper finding is that **different methods have qualitatively different scaling curves at different difficulty levels, and sometimes the most powerful optimization method actually *hurts* performance** on problems the model already finds easy.

Before this paper, the dominant assumption in the literature was that test-time compute can be treated as a uniform dial: turn it up, and performance improves. Best-of-N sampling (Cobbe et al., 2021), tree-of-thought search (Yao et al., 2023), and self-correction (Madaan et al., 2023) were all studied as methods that either do or don't improve aggregate accuracy, conflating results across problem difficulties. This paper's difficulty-bin analysis reveals that the contradiction in prior work—some papers finding self-correction works, others finding it doesn't (Huang et al., 2023)—is resolved by recognizing that self-correction works on easy problems but fails on hard ones, and that different studies implicitly tested on different difficulty distributions.

The evidence in Figure 3 (right) is striking: on the easiest problems (difficulty bin 1), beam search *degrades* performance as the compute budget increases, while best-of-N improves. On medium problems (bins 3–4), beam search *consistently outperforms* best-of-N. On the hardest problems (bin 5), nothing helps. This is not a monotonic relationship where more sophisticated optimization always helps—it is a non-monotonic, difficulty-dependent reversal that challenges the uniform-scaling assumption at a fundamental level.

The significance of this insight extends beyond the empirical finding itself. It reframes the problem of test-time compute allocation from "find the best method" to "find the best method for each prompt conditionally." This is a conceptual shift from optimization over a single global strategy to optimization over a policy that maps prompt characteristics to strategies—analogous to how reinforcement learning shifted from hand-engineered policies to learned state-conditioned policies. The paper formalizes this through the compute-optimal objective (Equation 1), but the intellectual contribution is the framing itself: test-time compute is a *resource allocation problem*, not a *method selection problem*.

### Innovation 2: The Proposal-Verifier Decomposition as a Unifying Framework for Test-Time Compute

The paper's decomposition of all test-time compute methods into modifications to either the **proposal distribution** (what the model generates) or the **verifier** (how outputs are selected) provides a conceptual lens that unifies previously disconnected lines of research. This is more than a taxonomy—it is a diagnostic framework that explains *why* different methods have different difficulty-dependent profiles.

Prior work studied revisions (Madaan et al., 2023; Qu et al., 2024), PRM-guided search (Lightman et al., 2023; Wang et al., 2023), majority voting, and verifier-based selection as separate methods evaluated on separate benchmarks with separate conclusions. The field lacked a way to reason about their complementary strengths—or to predict when each would dominate. The proposal-verifier decomposition fills this gap.

The empirical payoff is that the paper can characterize *which axis matters when*: on easy problems, improving the proposal distribution via revisions dominates (the model's initial attempts are roughly correct and just need refinement); on medium-hard problems, improving the verifier via PRM-guided search dominates (the model needs to explore and select among qualitatively different approaches). This is not obvious a priori—one might have guessed the opposite (revisions for hard problems, since they require multiple attempts, and verifiers for easy problems, since answers can be verified cleanly). The fact that the decomposition reveals counterintuitive difficulty-dependent behavior is what makes it a genuine intellectual contribution rather than a relabeling exercise.

The framework also naturally points toward future work: combining proposal improvements with verifier improvements—which the paper does not do—is the obvious next step, and the framework provides the language for describing what such a combination would accomplish (better candidates *and* better selection, amplifying both axes simultaneously).

### Innovation 3: Verifier Over-Optimization as the Primary Bottleneck in Test-Time Compute Scaling

The paper provides some of the first clear evidence that **verifier over-optimization**—where search finds solutions that score highly under a reward model but are actually incorrect—is the central limiting factor for scaling test-time compute, not search algorithm sophistication. This is a negative result with significant positive implications for how research should be directed.

The evidence is concrete and spans multiple experiments. Beam search degrades easy-problem performance at high budgets (Figure 3, right)—the search is finding solutions that exploit the PRM's blind spots. Lookahead search—the most sophisticated and computationally expensive search method, which simulates multiple steps forward to score partial solutions more accurately—paradoxically performs *worst* overall at matched generation budgets (Figure 3, left). Qualitative examples in Appendix M show search producing degenerate outputs (repetitive low-information steps, overly short solutions) that the PRM rates highly. These are all signatures of goodhart's law: when a metric becomes a target, it ceases to be a good metric.

What makes this insight distinctive is that it shifts the burden of progress away from search algorithms and toward verifier robustness. Prior to this paper, the natural assumption was that more sophisticated search—lookahead, MCTS, tree-of-thought branching—would yield better results, and that the research agenda should focus on developing better search algorithms. This paper shows the opposite: the simplest search method (best-of-N) often outperforms more sophisticated methods because it applies less optimization pressure to the verifier, staying below the over-optimization threshold. The bottleneck is not in *finding* high-scoring solutions under the verifier, but in having a verifier whose scores correlate with correctness under aggressive search.

This connects to a broader pattern in AI safety and alignment (reward hacking in RLHF, specification gaming in reinforcement learning) but is applied here to the specific domain of test-time compute scaling for language models. The practical implication is that improving verifier training—through better data, calibration, adversarial robustness, or ensemble methods—is likely a more impactful investment than developing new search algorithms.

### Innovation 4: Compute-Optimal Test-Time Scaling Can Substitute for ~14× More Pretraining—But Only Within the Model's Capability Frontier

The FLOPs-matched comparison in Section 7 establishes an empirical boundary condition that is more nuanced and practically useful than a simple "test-time compute beats pretraining" or "pretraining beats test-time compute" headline. The finding is that **test-time compute can outperform a ~14× larger pretrained model on easy-to-medium problems, but provides essentially zero benefit on problems outside the base model's capability range.** This is a fundamental insight about the nature of capability versus computation.

Prior work on training-inference tradeoffs (Jones, 2021; Villalobos and Atkinson, 2023; Sardana and Frankle, 2023) largely assumed access to ground-truth answers at inference time, making the comparisons less realistic. This paper operates in a realistic setting where the correct answer is unknown and a learned verifier must be used instead. The result is not a blanket endorsement of test-time compute over pretraining—it is a precise characterization of *when* the substitution works and *when it fundamentally cannot work*.

The hardest problems (difficulty bin 5) show near-zero improvement regardless of test-time compute budget, while the ~14× larger model solves some non-trivial fraction of them. This means that test-time compute amplifies existing capability but does not create new capability. If the base model's pass@1 is essentially zero on a problem class, no amount of search, revision, or verifier optimization will help—there are no correct solutions in the proposal distribution to find or refine. This is a sharp boundary condition that the paper documents clearly (Figure 9 shows the bin 5 scaling line flat near 0–5% while the larger model's accuracy is above that).

This finding has direct practical implications for deployment architecture: organizations should allocate test-time compute to problems within the base model's known capability range (easy-to-medium difficulty) and route genuinely hard problems to larger models—or invest in better pretraining for those problem classes. It also implies that the popular narrative of "inference-time scaling can replace training-time scaling" is oversimplified: the replacement is possible only within a capability frontier defined by the base model, and expanding that frontier still requires pretraining.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper evaluates on seven datasets spanning summarization, translation, and question answering: CNN/Daily Mail (Nallapati et al., 2016), arXiv and PubMed (Cohan et al., 2018), MediaSum (Zhu et al., 2021), Multi-News (Fabbri et al., 2019), WMT 2014 English-to-German, and TriviaQA (Joshi et al., 2017). The paper explicitly omits classification benchmarks such as GLUE (Wang et al., 2019) because "autoregressive inference is less applicable for those tasks"—the memory bandwidth bottleneck from loading keys and values primarily manifests during sequential generation, not single-pass classification. The datasets provide diversity in output length (from 32-token TriviaQA answers to 512-token long-form summaries) and input length (512 to 2048 tokens), which matters because the KV-cache overhead scales with sequence length.

- **Base model(s).** All models use the T5.1.1 architecture (Raffel et al., 2020) implemented in JAX with Flax and Flaxformer. The primary experiments use public T5.1.1 Large and XXL checkpoints. T5 is an encoder-decoder Transformer, which means attention is split into encoder self-attention (parallel, non-bottlenecked), decoder self-attention (autoregressive, KV-cache bottlenecked), and cross-attention (decoder attending to encoder outputs, also KV-cache bottlenecked). The authors apply MQA and GQA to decoder self-attention and cross-attention but not encoder self-attention. T5-XXL has H = 64 heads, so MQA-GQA-1 represents a 64:1 reduction, while GQA-8 represents an 8:1 reduction.

- **Metrics.** The paper reports two categories of metrics. For quality, it uses task-specific standard measures: ROUGE-1 (R1) for all summarization datasets (CNN/Daily Mail, arXiv, PubMed, MediaSum, MultiNews), BLEU for WMT translation, and F1 for TriviaQA question answering. For efficiency, the paper reports per-sample inference time in milliseconds (Table 1) or seconds (Figure 6), measured as wall-clock time per sample per TPUv4 chip using xprof (Google's TPU profiling tool). The choice of per-sample time rather than throughput or FLOPs reflects the paper's focus on the user-facing latency bottleneck, not total computational cost.

- **Baselines.** The paper establishes a three-point baseline architecture. **MHA-Large** and **MHA-XXL**: standard multi-head attention T5 models with H key-value heads, representing the quality ceiling and the existing public checkpoints that practitioners would otherwise use for inference. **MQA-XXL**: the same T5-XXL architecture converted to multi-query attention (GQA-1) via mean pooling and uptrained with α = 0.05, representing the maximum-speed, reduced-quality alternative proposed by Shazeer (2019). The paper does not compare against a separately trained MQA-XXL from scratch—the comparison is with the uptrained MQA variant, which controls for the uptraining procedure and isolates the architectural effect.

- **Generation budget / compute accounting.** The paper does not use "generation budget" in the sense of sampling multiple completions (as in best-of-N or beam search)—all inference uses greedy decoding with a single output. Instead, the relevant "compute" metric is uptraining cost, measured as a proportion α of the original pre-training steps (with α ∈ {0, 0.01, 0.05, 0.10}). For the main experiments, α = 0.05 corresponds to approximately 600 TPUv3 chip-days for T5-XXL. This cost is a one-time upfront training cost, not a per-inference cost—the trade-off is between spending additional pre-training compute (to recover quality after architectural conversion) versus accepting lower inference quality or higher inference latency. For inference timing, compute is measured as wall-clock time per sample under identical hardware (TPUv4) with per-model-optimized serving configurations (batch size, parallelization).

- **Cross-validation / statistical protocol.** The paper does not employ cross-validation for model selection. Fine-tuned checkpoints are selected based on highest dev set performance, with training continuing until convergence. For MQA models, which the paper notes suffer from "high variance" during fine-tuning (Appendix A), the authors report average performance over three fine-tuning runs on unstable tasks. GQA models are reported as stable and do not require this averaging. There is no mention of statistical significance testing (confidence intervals, standard errors, or hypothesis tests) for the quality comparisons between architectures. The paper does not split the uptraining data or hold out a validation set for uptraining checkpoint selection—the uptrained model is simply the one at the end of the additional α fraction of pre-training steps, following the original T5 recipe without modification.

### Main Quantitative Results

#### Uptrained MQA and GQA Versus MHA Baselines

The headline result appears in Figure 3 and Table 1: **uptrained GQA-8-XXL achieves quality nearly indistinguishable from MHA-XXL while running 5.4× faster, and quality substantially above MHA-Large while running 1.3× faster.**

Table 1 breaks this down per dataset. MHA-XXL achieves an average ROUGE-1/F1/BLEU of 47.2 across all seven tasks, with inference time of 1.51 ms per sample. MQA-XXL (uptrained with α = 0.05) achieves 46.6 average performance—a 0.6 point drop from MHA-XXL—but runs at 0.24 ms per sample, a 6.3× speedup. GQA-8-XXL achieves 47.1 average performance—only 0.1 points below MHA-XXL—while running at 0.28 ms per sample, a 5.4× speedup over MHA and only 1.17× slower than MQA.

For perspective, MHA-Large achieves 46.0 average performance at 0.37 ms per sample. This means GQA-8-XXL is simultaneously higher quality than MHA-Large (47.1 vs. 46.0) and slightly faster (0.28 vs. 0.37 ms), representing a strict Pareto improvement over the smaller MHA model.

The dataset-level results show that the quality recovery from GQA is consistent rather than being driven by one or two datasets. On CNN/Daily Mail, GQA-8-XXL scores 43.5 versus MHA-XXL's 43.8 and MQA-XXL's 43.0. On PubMed, GQA-8-XXL scores 45.4 versus MHA-XXL's 45.6 and MQA-XXL's 45.0. On MultiNews, GQA-8-XXL slightly exceeds MHA-XXL (47.2 vs. 46.9). On TriviaQA, GQA-8-XXL achieves 81.6 versus MHA-XXL's 81.9. The only dataset where GQA-8-XXL does not approach MHA-XXL is WMT translation, where all three XXL variants score similarly (28.4-28.5 BLEU), though this may reflect a ceiling effect where the task is not sensitive to attention head count.

#### Uptraining Proportion and the Diminishing Returns Curve

Figure 5 shows performance (averaged over the representative subset of CNN/Daily Mail, MultiNews, and TriviaQA) as a function of uptraining proportion α for T5-XXL with MQA and GQA-8. The results establish three key facts.

First, **GQA is usable with minimal uptraining; MQA is not.** At α = 0 (immediate evaluation after checkpoint conversion, with no continued pre-training), GQA-8 already achieves what the paper calls "reasonable performance" (approximately 54 on the y-axis, close to or within the plotting range of the post-uptraining MQA curve), while MQA is essentially non-functional and "requires uptraining to be useful." This is a practical advantage: GQA's structural property of retaining some key-value head diversity means the converted checkpoint is closer to functional out-of-the-box.

Second, **both architectures benefit substantially from α = 0.05, with diminishing returns at α = 0.10.** The curves show a steep rise from α = 0 to α ≈ 0.01-0.02, continuing improvement to α = 0.05, and then a marked flattening between α = 0.05 and α = 0.10. The GQA-8 curve approaches the MHA reference line (horizontal dashed line) by α = 0.05 and nearly reaches it by α = 0.10.

Third, **the gap between GQA and MQA persists across all uptraining budgets.** At every α value, GQA-8 outperforms MQA by a roughly constant margin. Additional uptraining does not close this gap—it simply raises both curves in parallel. This means the architectural difference (8 key-value head groups versus 1) imposes a quality ceiling that no amount of continued pre-training (up to 10%) can overcome. The quality recovery is asymptotic to something below the MHA reference for MQA, but essentially reaches MHA for GQA.

The paper selects α = 0.05 as the operational point, noting it "took approximately 600 TPUv3 chip-days" for T5-XXL—a small fraction of the original full training cost.

#### Checkpoint Conversion Method Ablation

Figure 4 compares three conversion strategies for T5-Large converted to MQA and uptrained with α = 0.05: mean pooling, first-head selection, and random initialization. Performance is reported on the representative task subset (CNN/Daily Mail, MultiNews, TriviaQA) as a single aggregate number.

Mean pooling achieves the highest performance (approximately 55.6). First-head selection achieves intermediate performance (approximately 55.2). Random initialization achieves the lowest performance (approximately 54.4). The ordering exactly follows the degree of information preservation—mean pooling preserves all heads' contributions, first-head selection preserves one, random initialization preserves none—and the paper states this explicitly: "results are ordered by the degree to which information is preserved from the pre-trained model."

The gap between mean pooling and random initialization (approximately 1.2 points) represents the value of structural prior from the original MHA checkpoint versus learning key-value projections from scratch during uptraining. Interestingly, the gap between mean pooling and first-head selection (approximately 0.4 points) is smaller than the gap between first-head selection and random initialization (approximately 0.8 points), suggesting that even preserving one head's projection is substantially better than starting from random, and that the marginal benefit of additional heads beyond the first diminishes but remains positive.

#### Number of Groups and the Quality-Speed Trade-Off

Figure 6 shows inference time for GQA-XXL as a function of the number of groups G ∈ {1, 4, 8, 16, 32, 64}, measured with input length 2048 and output length 512. The time axis uses seconds (not milliseconds as in Table 1) because these measurements include longer sequences and potentially different batch configurations.

The curve is convex: increasing G from 1 (MQA) to 8 adds only "modest inference overhead," while further increases toward 64 (MHA) incur "increasing cost." Quantitatively, MQA (G = 1) takes approximately 1 second, GQA-8 takes approximately 1.2 seconds (20% slower), GQA-16 takes approximately 1.35 seconds, GQA-32 takes approximately 1.6 seconds, and MHA (G = 64) takes approximately 2 seconds (100% slower than MQA). The incremental cost per additional group rises as G approaches H.

The paper selects G = 8 as "a favorable middle ground," capturing most of MQA's speed while recovering most of MHA's quality. The convex shape of the time curve justifies this choice: the first few groups beyond G = 1 add relatively little latency because the KV-cache is still small relative to the total memory bandwidth budget (which includes loading model weights and performing computations unrelated to attention). As G grows, the KV-cache becomes the dominant consumer of memory bandwidth, and further increases translate directly into proportional latency increases.

#### Training Stability: MQA vs. GQA

Appendix A reports a finding that influenced experimental design but is not quantified in the main text. MQA models trained from scratch suffered from "frequent loss spikes" during pre-training and "diverged immediately when fine-tuning on long-input tasks." Uptrained MQA models (converted from MHA checkpoints) are "more stable" but "still display high variance," requiring the paper to report "average performance over three fine-tuning runs" for MQA on unstable tasks.

By contrast, uptrained GQA models are reported as "stable" without special variance mitigation. The paper speculates that the single key-value head in MQA creates a stricter information bottleneck that makes training more sensitive to initialization and optimizer dynamics, but does not investigate the root cause further: "we did not investigate further on the root causes of multi-query instability."

This stability finding is practically significant even without mechanistic explanation. It means that GQA not only achieves better quality than MQA (Figure 3, Table 1) but also trains more reliably, reducing the engineering burden of managing loss spikes, divergence, and multi-run averaging. For practitioners, this reliability advantage may matter as much as the quality improvement.

### Ablation Studies and Robustness Checks

**Checkpoint conversion method (Figure 4):** Mean pooling outperforms both first-head selection and random initialization for converting MHA to MQA, confirming that preserving information from all original key-value heads is important and that the mean provides a better initialization than any single head or random weights. The gap between mean and first-head selection (~0.4 points) is smaller than the gap between first-head selection and random (~0.8 points), suggesting non-linear returns to preserved information.

**Uptraining proportion (Figure 5):** Both MQA and GQA benefit substantially from 5% continued pre-training, with diminishing returns at 10%. GQA achieves reasonable performance even at α = 0 (no uptraining), while MQA is non-functional without uptraining. The GQA-MQA quality gap persists across all α values, indicating an irreducible architectural quality difference that continued pre-training cannot close.

**Number of groups (Figure 6):** Inference time increases convexly with G. The jump from G = 1 to G = 8 is modest (~20% slowdown), while the jump from G = 32 to G = 64 (full MHA) is steep (~25% of the remaining gap). This convex relationship justifies G = 8 as capturing most speed gains while recovering quality, and demonstrates that the quality-speed trade-off is not linear in G—small increases in group count from MQA provide disproportionate quality benefits at low latency cost.

**Training stability (Appendix A):** Uptrained GQA models are stable during fine-tuning, while uptrained MQA models exhibit high variance requiring multi-run averaging. MQA trained from scratch is even more unstable (loss spikes, fine-tuning divergence). This is a robustness advantage for GQA that is independent of final quality.

**Dataset and input length diversity (Table 1):** The quality advantage of GQA over MQA holds across short-form summarization (CNN/Daily Mail, input 512), long-form summarization (arXiv, PubMed, MediaSum, MultiNews, input 2048), translation (WMT), and question answering (TriviaQA), with no datasets where MQA outperforms GQA. The benefit therefore generalizes across task types and sequence lengths. Notably, the paper does not report whether the relative advantage of GQA over MQA varies with input length—longer inputs should stress the KV-cache bottleneck more, potentially making GQA's speed advantage more pronounced, but this is not directly tested.

**No from-scratch GQA or MQA baseline:** The paper does not compare uptrained GQA/MQA against models trained from scratch with those attention mechanisms. This means we do not know whether uptraining to GQA achieves the same quality as training GQA from scratch with the same total compute (original pre-training + 5% uptraining). It is possible that a from-scratch GQA-XXL, trained with 105% of the MHA-XXL compute, would outperform the uptrained version, which would mean the uptraining procedure loses some quality relative to native training. The paper acknowledges this limitation explicitly.

### Critical Assessment

The paper's central claim is that **uptrained GQA achieves quality close to MHA with speed close to MQA**, and that **a cheap uptraining procedure (5% of original pre-training compute) enables this conversion from existing MHA checkpoints.** Let us assess how well the experiments support each dimension of this claim.

**Does uptrained GQA achieve quality close to MHA?** Table 1 provides the key evidence: GQA-8-XXL achieves 47.1 average versus MHA-XXL's 47.2 average. This is compelling at the aggregate level—a 0.1 point difference across seven diverse datasets is small. However, close inspection reveals that the near-parity is partly a consequence of how the average is constructed. The datasets have different score ranges and variances. On CNN/Daily Mail, the gap is 0.3 ROUGE-1 points (43.5 vs. 43.8). On TriviaQA, the gap is 0.3 F1 points (81.6 vs. 81.9). These are small in absolute terms, but without confidence intervals or standard deviations, we cannot assess whether they are statistically distinguishable. The paper does not report variance for GQA models (since they are "stable"), making it impossible to determine whether 47.1 versus 47.2 represents a real quality deficit or sampling noise. A missing experiment that would strengthen the claim is reporting per-dataset standard deviations or confidence intervals from multiple fine-tuning seeds.

Moreover, the claim of "quality close to MHA" is demonstrated only for one model size (XXL) and one architecture (T5.1.1 encoder-decoder). The paper does not show that GQA scales consistently across model sizes—for instance, whether GQA-8 on a T5-Large would also approach MHA-Large quality, or whether the optimal G changes with model scale. The scaling argument in Section 2.2 suggests GQA should be *more* advantageous for larger models, but this is a theoretical claim backed only by the single XXL datapoint. Experiments on a smaller model (T5-Base, T5-Large) with GQA conversion would test whether the quality recovery is universal or specific to the overparameterized regime of XXL.

**Does uptrained GQA achieve speed close to MQA?** Table 1 shows GQA-8-XXL at 0.28 ms versus MQA-XXL at 0.24 ms—a 17% slowdown. The group-count ablation (Figure 6) shows GQA-8 at approximately 1.2 seconds versus MQA at approximately 1.0 second—a 20% slowdown for a 2048-input, 512-output configuration. These are modest slowdowns, and "close to MQA" is a fair characterization. However, the speed measurements depend heavily on sequence length. MQA's advantage grows with sequence length because the KV-cache size reduction is proportionally more impactful when the cache is large. For very long sequences (e.g., 4096 or 8192 tokens), the 17-20% gap between GQA and MQA might widen or narrow depending on how the memory bandwidth bottleneck scales. The paper tests only one input length (2048) and one output length (512) for the speed measurements in Figure 6, and Table 1 reports a single average inference time across all tasks (which have different sequence lengths but are aggregated into one number). The speed claim would be more robust with timing measurements across a range of sequence lengths.

**Does 5% uptraining suffice?** Figure 5 shows that α = 0.05 captures most of the asymptotically achievable performance for both architectures on T5-XXL. The diminishing returns from α = 0.05 to α = 0.10 are clearly visible. This is convincing for XXL. However, the optimal uptraining proportion may depend on model size—a smaller model might adapt faster (requiring less uptraining) or slower (requiring more). The paper tests only one model size for the α sweep (XXL) and one α value for the Large conversion experiment (α = 0.05 in Figure 4). Without an α sweep for Large, we cannot be confident that 5% is the right budget across scales.

**Is mean pooling the best conversion method?** Figure 4 shows mean pooling outperforming first-head and random initialization for T5-Large. Three methods are tested, and the ordering is clean. However, more sophisticated conversion methods might perform better. For instance, instead of mean pooling (which gives each head equal weight), one could learn a weighted combination of heads via a small linear projection during uptraining, or use principal component analysis to extract the dominant directions across heads. The paper does not explore whether the unweighted mean is optimal or simply a simple heuristic that works. The gap between mean pooling and first-head selection (0.4 points) suggests that the precise method matters, and that there may be room for improvement beyond the mean.

**What is missing from the experimental design?**

*No direct comparison to a from-scratch GQA model.* The paper's claim is about uptraining as a cost-effective alternative to training from scratch, but it never actually trains GQA/MQA from scratch to establish the quality ceiling. If a from-scratch GQA-XXL achieves 47.3 (above uptrained GQA's 47.1), then uptraining imposes a small but real quality penalty. If it achieves 47.1, then uptraining perfectly recovers native quality. We do not know which is true.

*No comparison to other inference acceleration methods.* The paper claims that GQA provides a "favorable trade-off" (Section 2.2), but it never compares GQA's quality-speed curve to that of quantization, distillation, FlashAttention, or speculative sampling. For a practitioner deciding how to reduce inference latency, the relevant question is not "Is GQA better than MHA or MQA?" but "Among all available acceleration methods, which provides the best quality-speed Pareto frontier?" The paper does not answer this question, and the implicit claim that reducing KV-cache size is the right lever assumes that memory bandwidth from key-value loading is the dominant bottleneck—which may not hold for all hardware, all sequence lengths, or all deployment configurations.

*No encoder-decoder vs. decoder-only comparison.* The paper evaluates GQA only on T5, an encoder-decoder model. The limitation section acknowledges that decoder-only models are "extremely popular" and speculates that "GQA [would] have a stronger advantage over MQA" in that setting because all attention layers (not just decoder self-attention and cross-attention) would benefit from reduced KV-cache. This is a plausible claim, but it is untested. A decoder-only experiment (e.g., converting a LLaMA checkpoint) would substantially strengthen the paper's generality.

*No evaluation on generation quality beyond greedy decoding.* All experiments use greedy decoding. GQA's impact on generation diversity, coherence, and quality under sampling or beam search is unexplored. Since MQA reduces the diversity of attention patterns, it might have a stronger negative effect when sampling at higher temperatures or using diverse beam search—regimes where query diversity matters more. GQA might mitigate this, but the experiments cannot speak to it.

*Small uptraining data diversity.* The uptraining uses the same pre-training corpus as the original T5 training (C4). This means the model adapts to GQA using the same data distribution it originally learned from. In transfer learning or domain adaptation scenarios, where the uptraining data might differ from the pre-training data, the dynamics could change. The paper does not test whether uptraining on a different corpus (e.g., domain-specific data) would be as effective.

**Strengths of the experimental design:**

The paper makes clean, well-controlled comparisons. The uptrained MQA and GQA models share the same base checkpoint, the same uptraining data and hyperparameters, and the same fine-tuning procedure. This isolates the architectural variable (number of key-value groups) from confounding factors. The choice to measure per-sample time on identical hardware with per-model-optimized serving configurations reflects real-world deployment conditions rather than idealized throughput numbers. The sweep across seven diverse datasets provides evidence that the quality results generalize across task types, not just on a single benchmark.

The convex inference time curve (Figure 6) is a genuinely informative result that makes the G = 8 selection principled rather than arbitrary. By showing that inference time is not linear in G, the paper provides a tool for practitioners to select their own trade-off point based on their latency requirements. The same experiment also demonstrates that the paper's main result (GQA-8) is not cherry-picked—the curve shows a smooth progression that makes G = 8 a natural elbow.

The stability results in Appendix A, while not deeply analyzed, are practically valuable. The observation that MQA training is unstable while GQA training is stable is the kind of finding that strongly influences adoption decisions, even if the mechanism is not understood. Reporting it honestly, with the admission that the root cause was not investigated, is appropriate.

Overall, the experiments support the paper's claim that GQA with 5% uptraining provides a favorable quality-speed operating point relative to both MHA and MQA on T5-XXL across seven NLP benchmarks. The claim is substantiated for the tested configuration. However, the generality of the finding—whether GQA scales across model sizes, whether it works for decoder-only architectures, whether 5% uptraining is universally sufficient, and how it compares to non-architectural acceleration methods—remains largely untested. These are not flaws in the reported experiments, which are internally rigorous and well-controlled, but rather scope limitations that narrow the claim from "GQA solves the quality-speed trade-off for Transformer inference" to "GQA solves the quality-speed trade-off for T5-XXL-sized encoder-decoder models uptrained on their original pre-training data."

## 6. Limitations and Trade-offs

### Uptraining Has Only Been Validated on a Single Model Family and Architecture

**The assumption or constraint.** All experiments—the uptraining proportion sweep, the checkpoint conversion method comparison, the group-count ablation, and the final quality-speed evaluation—are conducted exclusively on T5.1.1, an encoder-decoder Transformer architecture, using publicly available checkpoints for the Large (770M parameters) and XXL (11B parameters) sizes. The paper makes no claim about encoder-decoder generality as a limitation; rather, it acknowledges in Section 5:

> "we evaluate the impact of uptraining and GQA only on encoder-decoder models. Recently, decoder-only models are extremely popular, and since these models do not have separate self-attention and cross-attention, we expect GQA to have a stronger advantage over MQA."

This is a forward-looking speculation, not a validated result. There is no experimental evidence in the paper that GQA uptraining works for decoder-only architectures (e.g., GPT, LLaMA, PaLM), for models with different normalization schemes (LayerNorm vs. RMSNorm), for models with different activation functions, or for models trained with different objectives (next-token prediction vs. span corruption).

**The consequence.** A practitioner holding a decoder-only checkpoint—which describes the vast majority of contemporary large language model deployments—has no empirical guarantee that the 5% uptraining recipe transfers. The structural differences are material: decoder-only models have no separate cross-attention layers, meaning every attention layer is self-attention and contributes to the KV-cache bottleneck. This is precisely why the paper speculates GQA would have "a stronger advantage" in decoder-only models. But the converse risk is also plausible: decoder-only models may be more sensitive to attention structure changes because they lack the encoder's parallel processing to provide rich key-value representations for cross-attention. The uptraining dynamics (how many steps are needed, whether the learning rate schedule needs adjustment, whether the optimizer hyperparameters transfer) are untested.

Additionally, the paper only tests two model sizes (Large and XXL). The scaling argument in Section 2.2—that GQA becomes more important for larger models because MQA's proportional benefit diminishes—is a theoretical claim backed only by the single XXL datapoint. Whether GQA-8 would recover near-MHA quality for a much larger model (e.g., 100B+ parameters, 128+ heads) or whether more groups would be needed to avoid a quality cliff is unknown.

**What evidence exists in the paper.** None. The paper contains zero experiments on decoder-only architectures, zero experiments on model families other than T5.1.1, and zero experiments on model sizes outside the Large-XXL range. The speculation about decoder-only models is explicitly marked as an expectation, not a finding.

**Mitigation status.** Not addressed experimentally. The authors state the expectation as a hypothesis in the limitations section but provide no validation. A practitioner cannot rely on this speculation without replication on their target architecture.

---

### No Comparison Against a GQA or MQA Model Trained From Scratch

**The assumption or constraint.** The paper's central efficiency claim is that uptraining with α = 5% of original pre-training compute is a cost-effective way to obtain a fast-inference model from an existing MHA checkpoint. However, the paper never establishes the quality ceiling: what would a GQA-XXL or MQA-XXL model achieve if trained from scratch with 105% of the MHA pre-training compute (matching the total FLOPs of uptraining)? The authors acknowledge this directly in Section 5:

> "Due to limited computation, we also do not compare our XXL GQA model to a comparitive model trained from scratch, so we do not know the relative performance of uptraining vs training from scratch."

**The consequence.** Without a from-scratch baseline, the paper cannot distinguish between two possibilities: (1) uptraining perfectly recovers the quality that native GQA training would achieve, meaning the 5% cost is pure savings, or (2) uptraining imposes a quality penalty—native GQA training from scratch with equivalent total compute would outperform the uptrained version—in which case the apparent cost savings come at the expense of quality that could have been achieved with a different training strategy.

This matters for a downstream decision-maker. If a team is planning to train a large model from scratch (not convert an existing checkpoint), should they train it as MHA and then uptrain to GQA, or train it as GQA from the start? The paper provides no evidence to answer this question. If from-scratch GQA outperforms uptrained GQA by a meaningful margin, then the "savings" from uptraining are illusory—the team could have achieved better quality with the same total compute by training GQA natively. Conversely, if uptrained GQA matches from-scratch GQA, then uptraining is genuinely value-preserving.

The same ambiguity applies to MQA. The paper reports that MQA trained from scratch suffers from "frequent loss spikes" and fine-tuning divergence (Appendix A), while uptrained MQA is "more stable." This suggests uptraining may actually be *preferable* for MQA, but without direct quality comparison, the magnitude of any quality gap (in either direction) is unknown.

**What evidence exists in the paper.** None. The paper contains no experiments with GQA or MQA models trained from scratch. The only from-scratch MQA training mentioned is an informal stability observation in Appendix A, without quality numbers or controlled comparison to uptrained MQA. The limitation is explicitly acknowledged but not measured.

**Mitigation status.** Acknowledged as a limitation with the stated reason being limited computation. No experimental mitigation. A full training run for T5-XXL is expensive (the 5% uptraining alone took ~600 TPUv3 chip-days, implying full training is ~12,000 TPUv3 chip-days), so the omission is understandable but the uncertainty it creates is significant for practitioners weighing uptraining against alternative training strategies.

---

### The 5% Uptraining Budget Is Validated Only at One Model Size, and the Practical Uptraining Cost May Be Understated

**The assumption or constraint.** The paper demonstrates that α = 0.05 (5% of original pre-training steps) captures most of the achievable quality recovery for T5-XXL (Figure 5). The sweeping conclusion is that this represents a cost-effective recipe. However, the α sweep is conducted only for the XXL model—the Large model is uptrained only at α = 0.05 (Figure 4), with no sweep to determine whether 5% is optimal, sufficient, or excessive for the smaller model. The paper provides no uptraining budget guidance for models of different sizes or for models trained with different pre-training budgets.

**The consequence.** The optimal uptraining proportion may be model-size-dependent. A smaller model might adapt faster to the new attention structure (requiring less than 5% uptraining, making the recipe even cheaper) or slower (requiring more, making it less attractive). Conversely, a much larger model than XXL might require more uptraining to recover quality, or it might adapt faster because overparameterization provides more flexibility. Without scaling the α sweep across model sizes, a practitioner cannot confidently budget for uptraining on their specific model.

Furthermore, the absolute cost of 5% uptraining—~600 TPUv3 chip-days for T5-XXL—is reported as modest relative to full pre-training, but this framing assumes the practitioner already paid for and completed full pre-training. For an organization that has not trained T5-XXL from scratch but is considering whether to download an open-source MHA checkpoint and uptrain it to GQA, the uptraining cost is a new expenditure, not a fraction of a sunk cost. The paper does not discuss whether the uptraining cost compares favorably or unfavorably to alternative quality improvement methods (e.g., further fine-tuning on domain data, distillation from a larger model) that could be applied to the same starting checkpoint for similar compute.

**What evidence exists in the paper.** Figure 5 shows the α sweep for T5-XXL only. The Large model experiments use a single α value (0.05) without sweep validation. No experiments vary α for models of different sizes. The ~600 TPUv3 chip-day cost is stated without comparison to alternative training investments or total cost of ownership calculations.

**Mitigation status.** Partially addressed by the diminishing returns curve in Figure 5—the flattening at α = 0.10 provides confidence that 5% is near-optimal for T5-XXL specifically. But the broader guidance (how α should scale with model size, whether 5% is universally sufficient) is not investigated. The authors do not suggest this as a direction for future work in the limitations section.

---

### Training Instability of MQA and the Unexplained Stability Advantage of GQA

**The assumption or constraint.** Appendix A documents that MQA suffers from training instability, while GQA does not. From-scratch MQA training produced "frequent loss spikes during pre-training" and immediate divergence during fine-tuning on long-input tasks. Uptrained MQA models were "more stable but still display high variance," forcing the paper to average over three fine-tuning runs on unstable tasks. By contrast, "uptrained grouped-query attention models, however, appear to be stable." The paper states:

> "we did not investigate further on the root causes of multi-query instability."

**The consequence.** This is a practical reliability gap with no mechanistic explanation. A practitioner deciding between MQA and GQA faces not only a quality trade-off (GQA is better) and a speed trade-off (MQA is marginally faster) but also an engineering risk trade-off: MQA models may require multiple training runs to obtain usable results, may diverge on long sequences, and may exhibit unpredictable behavior during fine-tuning. The single key-value head in MQA creates a strict information bottleneck that appears to destabilize training, but without understanding *why*, the practitioner cannot predict whether the instability will manifest in their specific setting, cannot tune hyperparameters to mitigate it, and cannot know whether it generalizes to other architectures. GQA's stability advantage might be the strongest single reason to prefer it over MQA, but the paper cannot articulate when or why this advantage holds.

The instability is particularly concerning for long-sequence tasks, which are precisely where MQA's speed advantage over MHA matters most. If MQA diverges on the very tasks where its architectural speedup is most needed, the practical value proposition collapses.

**What evidence exists in the paper.** Qualitative observations in Appendix A: loss spikes in from-scratch MQA, divergence during fine-tuning on long inputs, high variance in uptrained MQA requiring multi-run averaging, and stability of uptrained GQA. However, these observations are not quantified—no loss curves are shown, no fine-tuning variance metrics are reported, and no statistical characterization of the instability (e.g., what fraction of runs diverge, at what sequence length the divergence occurs) is provided. The paper does not investigate whether the instability is specific to the Adafactor optimizer, the T5 pre-training objective, the learning rate schedule, or the interaction of any of these with the single key-value head.

**Mitigation status.** Not addressed. The authors acknowledge they did not investigate root causes and do not propose future work on this topic. The practical mitigation—averaging over three fine-tuning runs for MQA on unstable tasks—is a workaround, not a solution, and it triples the already-expensive fine-tuning cost on the most challenging tasks.

---

### Quality Evaluation Is Limited to Greedy Decoding on Short- to Medium-Length Tasks with Automated Metrics

**The assumption or constraint.** All quality evaluations use greedy decoding exclusively and report only automated metrics: ROUGE-1 for summarization, BLEU for translation, and F1 for question answering. The paper does not evaluate generation quality under diverse decoding strategies (sampling with temperature, beam search, nucleus sampling), does not include human evaluation, and does not test on tasks where the quality of generated text beyond factual accuracy matters (e.g., coherence, diversity, creativity, or style adherence). The maximum output length tested is 512 tokens (for long-form summarization), which is modest by modern standards where models routinely generate thousands of tokens.

**The consequence.** Greedy decoding is a specific operating point that may mask quality differences between attention mechanisms. GQA reduces query head diversity by forcing queries within a group to attend over the same keys and values. Under greedy decoding—where the model always selects the single most probable next token—the reduced attention diversity may not matter because the model is already making a deterministic, low-entropy choice. Under sampling at higher temperatures, where the model explores the distribution's tail, diverse attention patterns might be more important for generating varied, coherent outputs. If GQA reduces the model's ability to represent diverse attention patterns (compared to MHA), the quality degradation might be larger under sampling than under greedy decoding.

Similarly, automated metrics like ROUGE-1 measure lexical overlap and correlate imperfectly with human judgments of summary quality. ROUGE-1 in particular (unigram overlap) is a coarse metric that can reward extractive behavior and penalize abstractive rewrites. If GQA subtly changes the balance between extractive and abstractive generation (even while maintaining ROUGE-1 parity with MHA), human evaluators might notice differences that automated metrics miss.

The modest output lengths (max 512 tokens) mean the paper does not test whether GQA's quality advantage over MQA grows with sequence length. The memory bandwidth bottleneck that GQA addresses is most severe for long sequences, so one would expect GQA's ability to maintain quality (relative to MQA) to be most pronounced when generating thousands of tokens—but this regime is not evaluated.

**What evidence exists in the paper.** Table 1 reports ROUGE-1, BLEU, and F1 scores under greedy decoding across seven datasets. There are no experiments with sampling, beam search, or any non-greedy decoding strategy. There are no human evaluations. The longest output is 512 tokens (for arXiv, PubMed, MediaSum, MultiNews). Input lengths of 2048 are tested for these long-form summarization tasks and TriviaQA, but input length primarily affects the encoder's computation and cross-attention KV-cache loading—it does not test decoder self-attention over very long generated sequences.

**Mitigation status.** Not addressed. The limitations section does not mention decoding strategy, output length, or evaluation metric concerns. The paper's claim about quality is therefore valid only for the specific configuration tested (greedy decoding, automated metrics, ≤512-token outputs).

---

### The Paper Provides No Comparison to Non-Architectural Inference Acceleration Methods

**The assumption or constraint.** The paper positions GQA as an architectural solution to the memory bandwidth bottleneck from loading keys and values during autoregressive decoding. The baseline comparisons are architectural: MHA (no reduction) and MQA (maximum reduction). This establishes GQA as the best point on the MHA-MQA interpolation curve. However, a practitioner choosing an inference optimization strategy faces a broader menu: quantization (which reduces the precision, and therefore memory footprint, of all weights and activations including the KV-cache), distillation (which trains a smaller, faster model on the larger model's outputs), FlashAttention (which reduces memory usage during attention computation), speculative sampling (which increases throughput by using a small draft model), and layer-sparse cross-attention (which eliminates cross-attention layers entirely for long inputs). The paper does not compare GQA against any of these methods, either in isolation or in combination.

**The consequence.** The paper's implicit recommendation—reduce KV-cache size via grouped key-value heads—may not be the most effective single intervention or the best use of engineering effort for a given deployment scenario. For example:

- **Quantization to INT8** shrinks the KV-cache by 4× (for 32-bit to 8-bit) or more for lower precisions, without changing the architecture at all. An INT8-quantized MHA model might be faster than a full-precision GQA-8 model while retaining MHA's full representational capacity. The paper provides no data to evaluate this trade-off.
- **Distillation** can produce a smaller model that is faster on all dimensions (not just KV-cache loading) while matching or approaching the larger model's quality. A distilled model might outperform GQA on both quality and speed for a given parameter budget.
- **FlashAttention** is complementary to GQA—it reduces memory usage during attention computation without reducing cache size—and could be combined with GQA for additive benefits. But the paper does not test whether the gains from GQA are reduced when FlashAttention is already in use (because attention computation is less of a bottleneck).

Without these comparisons, the paper cannot claim that GQA is the *best* approach to reducing inference latency, only that it is *a* approach that works better than the architectural extremes (MHA and MQA). For a practitioner with a fixed engineering budget, the relevant question is: "Should I spend effort implementing GQA uptraining, or should I quantize, distill, or implement FlashAttention?" The paper provides no evidence to guide this decision.

**What evidence exists in the paper.** The related work section (Section 4) acknowledges these methods—FlashAttention, quantization, distillation, speculative sampling—as alternative or complementary approaches, but no experiments compare GQA against any of them. The timing measurements in Table 1 and Figure 6 are for full-precision models without FlashAttention, without speculative sampling, and without quantization.

**Mitigation status.** Not addressed. The paper does not frame this as a limitation and does not propose comparative benchmarks as future work. However, the authors' tone in Section 4 (listing these as related work rather than competitors) suggests they view GQA as orthogonal rather than competitive—many of these methods could be combined with GQA. This is true but does not resolve the prioritization question for a practitioner who can only implement one optimization. A multi-axis ablation (e.g., GQA + quantization vs. MHA + quantization) would clarify whether the benefits compound or overlap.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper makes a **pragmatic architectural contribution with outsized practical impact**—it is not a paradigm shift in attention mechanism theory but rather a carefully engineered interpolation that resolves a specific, costly tension between two existing architectural extremes. The conceptual move is simple: instead of choosing between multi-head attention (maximum quality, maximum KV-cache) and multi-query attention (maximum speed, degraded quality), introduce a single tunable parameter G that lets practitioners dial the quality-speed trade-off to their needs. The paper's contribution is in demonstrating that this interpolation is both architecturally valid and, critically, **obtainable from existing pre-trained MHA checkpoints at minimal cost** (5% continued pre-training).

The field-level impact operates on three levels:

**First, GQA collapses the false dichotomy between "train for quality" and "train for inference."** Before this paper, a team wanting fast inference from an MHA model faced an unpalatable choice: accept the MHA model's inference cost as-is, train a separate MQA model from scratch at full expense, or attempt some post-hoc modification with no guidance on whether quality would survive. The uptraining recipe—mean pool key-value heads, continue pre-training for 5% of original steps, fine-tune as usual—provides a reliable, cheap path from any T5-style MHA checkpoint to a fast-inference variant. The result is that a single pre-training run can now serve both quality-sensitive and latency-sensitive use cases, which fundamentally changes the economics of model development for organizations that deploy the same base model in multiple contexts.

The quantitative force of this argument comes from Table 1: GQA-8-XXL achieves a 5.4× speedup over MHA-XXL (1.51 ms → 0.28 ms) while losing only 0.1 points of average quality (47.2 → 47.1), and it is simultaneously higher quality than the smaller MHA-Large (47.1 vs. 46.0) while being faster (0.28 ms vs. 0.37 ms). This is a strict Pareto improvement over two common deployment options (full-size MHA and smaller MHA), achieved for the cost of ~600 TPUv3 chip-days of uptraining.

**Second, the paper provides a principled scaling argument for why intermediate key-value head counts matter more for larger models.** The observation that MQA's fixed single key-value head interacts poorly with model scaling—KV-cache overhead shrinks relative to total cost as models grow, and model parallelism replicates the single head wastefully—is not merely an empirical finding but a structural argument about attention architecture design. GQA's proportional reduction (maintaining a fixed compression ratio via G groups rather than a fixed absolute head count of 1) means the architecture scales naturally with model size. This insight has already influenced subsequent model design: several major decoder-only models released after this paper (2023–2024) adopted GQA or similar grouped-key-value approaches as their default attention mechanism.

**Third, the paper reconciles a quiet tension in the inference optimization literature.** Prior to GQA, the dominant narrative was that MQA (Shazeer, 2019) was *the* architectural solution to KV-cache bandwidth, and that its quality degradation was an acceptable price for speed. The paper demonstrates that this trade-off is unnecessarily harsh—most of MQA's speed can be retained while recovering most of MHA's quality—and does so with a mechanism (grouped heads) that requires no new operations, no additional parameters relative to MHA, and no specialized hardware or software support beyond what standard attention implementations already provide.

The conceptual reframing is subtle but important: the design space between "all heads independent" and "all heads shared" is not empty but rich, and the quality-speed curve through this space is convex (Figure 6)—the first few additional key-value heads beyond MQA provide disproportionate quality benefits at minimal latency cost. This means that MQA was overshooting the optimum for most practical deployments, and that the right operating point depends on model size and latency requirements rather than being a fixed architectural constant.

**What becomes less attractive as a research direction:** pure MQA as a standalone architectural target. If GQA can recover most of MQA's speed with substantially better quality and training stability, the case for deploying MQA (as opposed to GQA with small G) weakens considerably. The paper's stability results (Appendix A) reinforce this: MQA training is fragile, with loss spikes during pre-training, divergence during fine-tuning on long inputs, and high variance even in the uptrained variant, while GQA is stable. For most practitioners, the marginal 17% speedup of MQA over GQA-8 (0.24 ms vs. 0.28 ms for T5-XXL) will not justify the quality penalty and engineering fragility.

**What becomes more attractive:** uptraining as a general strategy for architectural modification post-pre-training. The paper extends Komatsuzaki et al. (2022)'s demonstration that dense checkpoints can be uptrained into mixture-of-experts architectures to the attention mechanism domain. This suggests a broader principle: major architectural changes that would be expensive or unstable to train from scratch can be cheaply applied to fully-trained models through checkpoint conversion plus modest continued pre-training. The key design decisions—mean pooling for information preservation, original pre-training recipe reuse, 5% budget—provide a template that other architectural modifications (e.g., changing attention window sizes, introducing sparsity patterns, modifying feed-forward layer structure) might follow.

---

### Follow-Up Research This Work Enables

**GQA uptraining on decoder-only models:** The paper explicitly speculates that GQA would have "a stronger advantage over MQA" in decoder-only architectures because all attention layers (not just decoder self-attention and cross-attention) contribute to the KV-cache bottleneck. Testing this requires taking a public decoder-only MHA checkpoint (e.g., LLaMA, Falcon, or GPT-NeoX), converting it to GQA at various group counts using the mean-pooling procedure, uptraining for 5% of the original pre-training steps on the original pre-training corpus, and evaluating on standard benchmarks (MMLU, HumanEval, GSM8K, and long-context tasks). The key measurement would be whether the quality-speed trade-off curve shifts favorably compared to the encoder-decoder case—specifically, whether GQA with a small number of groups (e.g., G = 4 or G = 8 for a 32-head model) achieves near-MHA quality with near-MQA speed, or whether the decoder-only setting requires more groups to maintain quality. A critical ablation would vary the uptraining proportion α across model sizes to determine whether 5% generalizes or whether larger decoder-only models require more (or less) continued pre-training. A negative result—GQA failing to recover quality in decoder-only models—would bound the method's generality and suggest that encoder-decoder architectures have specific properties (cross-attention providing rich key-value representations, parallel encoder computation) that make attention structure modification easier.

**Training GQA from scratch versus uptraining to establish the quality ceiling:** The paper acknowledges it does not know "the relative performance of uptraining vs training from scratch" for GQA. A direct comparison would train T5-XXL models from scratch with GQA-8 and MQA attention, using the same total compute as the MHA pre-training plus 5% uptraining (i.e., 105% of the MHA training budget). If from-scratch GQA-8 substantially outperforms uptrained GQA-8, then uptraining imposes a quality penalty—the 5% uptraining cost is not pure savings but rather a cheaper approximation that sacrifices some achievable quality. If from-scratch GQA-8 matches uptrained GQA-8, then uptraining is genuinely value-preserving and the conversion recipe can be recommended without reservation. If from-scratch MQA outperforms uptrained MQA (despite the training instability documented in Appendix A), then the stability issues might be manageable with tuned hyperparameters, and MQA could still be viable for teams willing to pay the engineering cost. This experiment would also answer the forward-looking question: if a team is planning a large pre-training run and knows they want fast inference, should they train as MHA and uptrain to GQA, or train as GQA from the start? The answer has major cost implications for organizations designing pre-training strategies today.

**Scaling G across model sizes to validate the proportional-reduction argument:** The paper's scaling argument in Section 2.2 claims that GQA "lets us keep the same proportional decrease in bandwidth and capacity as model size increases" by choosing G to maintain a fixed compression ratio rather than a fixed absolute head count. This is untested. A scaling study would train (or uptrain) T5 models at multiple sizes—Base (~220M parameters, 12 heads), Large (~770M, 16 heads), XL (~3B, 32 heads), and XXL (~11B, 64 heads)—with GQA at a fixed compression ratio (e.g., 8:1, meaning G = 2 for Base, G = 2 for Large, G = 4 for XL, G = 8 for XXL). The prediction from the scaling argument is that the quality gap between GQA and MHA should remain approximately constant across model sizes, while the quality gap between MQA (G = 1, varying compression ratio) and MHA should grow as models get larger (because a 64:1 reduction is more aggressive than a 12:1 reduction). If this holds, the proportional-reduction principle is validated and practitioners can select G based on their target compression ratio rather than needing to sweep per model size. If it does not hold—for instance, if larger models need proportionally more key-value heads to maintain quality—then the design rule would need to be more nuanced.

**Combining GQA with complementary inference acceleration methods and measuring compound effects:** The paper positions GQA as orthogonal to FlashAttention, quantization, distillation, and speculative sampling, but provides no combined benchmarks. A multi-axis engineering study would start with an uptrained GQA-8-XXL model and systematically apply each complementary method: INT8 quantization of the KV-cache and weights (measuring the additional speedup and any quality degradation), FlashAttention for the attention computation (measuring whether GQA's speedup shrinks because attention computation is no longer the bottleneck), and speculative sampling with a small draft model (measuring throughput improvements when the KV-cache is already small). The key question is whether the gains compound additively or sub-additively. If GQA + INT8 quantization yields a 5.4× × 4× = ~22× total speedup relative to full-precision MHA, then the combination is transformative. If FlashAttention already eliminates most of the memory bandwidth bottleneck that GQA targets, then GQA's incremental benefit may be small in FlashAttention-equipped systems. The study would also establish whether GQA-trained models are more or less amenable to quantization (the grouped key-value heads might have different weight distributions than the single MQA head or the full MHA set). This is the kind of systems paper that would directly inform production deployment decisions at LLM-serving organizations.

**GQA with dynamic or learned group assignment:** The paper uses contiguous, fixed grouping of query heads (heads 1–8 in group 1, 9–16 in group 2, etc.) and does not explore whether this assignment could be improved. A natural extension would allow the model to learn which query heads share key-value representations, either through a differentiable routing mechanism (query heads softly attend to multiple key-value heads with learned weights) or through clustering based on head function similarity (using the original MHA checkpoint's attention patterns on a corpus to determine which heads have correlated behavior and should share key-value projections). The experiment would measure whether learned grouping improves quality over contiguous grouping at the same G, and whether the improvement justifies the additional complexity. A related direction is dynamic grouping: rather than fixing G, allow the number of active key-value heads to vary by layer (deeper layers might need fewer groups because they represent more abstract features) or by input (sentences requiring fine-grained disambiguation might activate more groups). If successful, this would provide a more efficient allocation of the KV-cache budget than uniform G across all layers and inputs.

**Stress-testing GQA quality under diverse decoding strategies and longer sequences:** The paper evaluates GQA only under greedy decoding with outputs up to 512 tokens. A decoding robustness study would evaluate uptrained GQA-8-XXL under sampling at multiple temperatures (0.6, 0.8, 1.0), nucleus sampling (p = 0.9, 0.95), and beam search (beam widths 4, 8) on the same benchmark suite, measuring both automated metrics and human preference judgments. The hypothesis to test is that GQA's reduced key-value head diversity causes larger quality degradation under stochastic decoding (where diverse attention patterns matter more) than under greedy decoding. If GQA's quality gap to MHA widens substantially at higher temperatures, the method's applicability would be narrowed to deterministic decoding scenarios. Additionally, evaluating on long-form generation tasks (output lengths of 1024, 2048, 4096 tokens) would test whether GQA maintains quality over long contexts or whether error accumulation degrades performance—this is the regime where inference speed matters most, and where GQA's quality retention relative to MQA should be most pronounced if the architectural intuition holds.

---

### Practical Applications and Downstream Use Cases

**Cost-effective serving of large encoder-decoder models for summarization and translation pipelines:** Organizations running T5-based summarization or translation services at scale face a direct trade-off between quality (larger model) and cost (inference latency × request volume). The paper's numbers provide a concrete recipe: take an existing T5-XXL checkpoint, mean-pool key-value heads into 8 groups, uptrain for ~600 TPUv3 chip-days, and deploy. The resulting model runs 5.4× faster than MHA-XXL (1.51 ms → 0.28 ms per sample) while producing summaries within 0.3 ROUGE-1 points of the original MHA model on CNN/Daily Mail (43.5 vs. 43.8). For a service processing millions of requests daily, this 5.4× latency reduction translates directly to 5.4× lower serving costs (fewer accelerators needed to meet the same throughput target) or 5.4× higher throughput from the existing serving infrastructure, all while maintaining substantially better quality than the smaller MHA-Large model (47.1 vs. 46.0 average, while being 25% faster). The uptraining cost is a one-time expense amortized over the model's deployment lifetime. This use case is immediately actionable: the paper provides the exact conversion code (mean pooling), uptraining hyperparameters (original T5 recipe, Adafactor), and fine-tuning settings (constant LR 0.001, batch 128, dropout 0.1).

**Enabling long-context inference on hardware-constrained devices:** The memory bandwidth overhead from the KV-cache scales with both the number of key-value heads and the sequence length. For applications that require processing very long inputs—scientific paper summarization (arXiv, PubMed), multi-document summarization (MultiNews), or long-form question answering—the combination of long input sequences (2048 tokens in the paper's evaluation) and full MHA attention can exceed the memory capacity or bandwidth budget of a single accelerator. Converting to GQA-8 reduces the cross-attention KV-cache by 8×, which may be the difference between fitting the model on one device versus requiring tensor parallelism across multiple devices. This is particularly relevant for edge deployment or single-GPU serving where multi-device parallelism is unavailable. The paper's evaluation on PubMed (input 2048, MHA-XXL vs. GQA-8-XXL: 45.6 → 45.4 ROUGE-1) and MultiNews (46.9 → 47.2) demonstrates that the quality impact on long-input tasks is minimal, making GQA a drop-in replacement for MHA when memory constraints are binding.

**Single pre-training run serving dual use cases in model development organizations:** AI research labs and companies developing large language models typically serve multiple downstream teams with different requirements: a research team that needs maximum quality for benchmarking and paper-writing, and a product team that needs low latency for user-facing applications. Before this paper, satisfying both required either two separate pre-training runs (one MHA, one MQA) or forcing one team to accept a suboptimal model. The uptraining recipe means that a single MHA pre-training run can spawn both: the original MHA checkpoint for quality-sensitive use, and a GQA-uptrained variant (with 5% additional compute) for latency-sensitive use. The paper's results show that the GQA variant loses only 0.1 points of average quality while providing 5.4× speedup—an outcome that both teams would likely accept. This changes the pre-training strategy from "choose your quality-speed trade-off before training" to "train once for quality, convert cheaply for speed," which reduces total organizational training cost by nearly 50% (one full run + 5% vs. two full runs) for organizations needing both fast and high-quality checkpoints.

**Quick win for open-source model adapters:** The public availability of T5.1.1 checkpoints (and, by extension, the principle that any MHA checkpoint can be converted) means that individual practitioners or small teams can produce fast-inference variants of large open-source models without the resources to pre-train from scratch. With the paper's recipe—mean pool key-value heads, continue pre-training on C4 for 5% of original steps using the Adafactor setup described—a team with access to a modest TPU or GPU cluster (~600 TPUv3 chip-days, or roughly $15,000–30,000 in cloud compute at 2023 prices) can convert a public T5-XXL checkpoint into a GQA-8 variant that runs 5.4× faster with near-identical quality. This significantly lowers the barrier to deploying large open-source models in production and makes it feasible for organizations that cannot afford custom pre-training to still achieve competitive inference performance. The paper's detailed hyperparameter specifications in Section 3.1 and the public Flaxformer implementation reference make this actionable with minimal guesswork.

---

### When to Prefer This Method

The paper articulates clear trade-offs between MHA, MQA, and GQA along two axes (quality and inference speed), with an additional consideration of training stability from Appendix A. The decision rule is straightforward:

- **Prefer uptrained GQA over MHA when** inference latency is a binding constraint and you have access to the original pre-trained MHA checkpoint (or can obtain it). The quality loss is minimal (0.1 points on average across seven diverse NLP tasks for T5-XXL), the speedup is large (5.4×), and the uptraining cost is modest (5% of original pre-training compute). This applies whether you need the fastest possible serving or are memory-constrained and need to reduce KV-cache size to fit on available hardware.

- **Prefer uptrained GQA over MQA when** you need the inference speed of MQA (which is only ~17% faster than GQA-8 at 0.24 ms vs. 0.28 ms for T5-XXL) but cannot tolerate MQA's quality degradation (0.6 points below MHA vs. 0.1 for GQA), training instability (loss spikes, fine-tuning divergence on long inputs, high variance requiring multi-run averaging), or both. Unless the 17% latency gap strictly prevents meeting a hard real-time requirement, GQA dominates MQA on the quality-stability Pareto frontier.

- **Consider MQA only if** the maximum possible speedup is required and either (a) the quality degradation is acceptable for the specific application (e.g., rough draft generation, internal data processing where output will be human-reviewed) or (b) the engineering cost of managing MQA's training instability (multiple fine-tuning runs, careful learning rate tuning, exclusion of very long inputs) is justified by the marginal speed gain. The paper's data suggests this regime is narrow—GQA captures ~83% of MQA's speedup with substantially better quality and stability.

- **Retain MHA when** inference latency is not a binding constraint and the application requires the highest achievable quality, or when decoding strategies that benefit from maximum attention diversity (high-temperature sampling, diverse beam search) are critical and the paper's evaluations under those strategies (which were not conducted) might reveal larger quality gaps between GQA and MHA than the 0.1-point average under greedy decoding.
