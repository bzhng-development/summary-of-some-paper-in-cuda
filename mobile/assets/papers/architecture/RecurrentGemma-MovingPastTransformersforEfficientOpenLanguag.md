# RecurrentGemma: Moving Past Transformers for Efficient Open Language Models

**ArXiv:** [2404.07839](https://arxiv.org/abs/2404.07839)

## 🎯 Pitch

RecurrentGemma unveils a new family of open language models leveraging Google's Griffin architecture, which breaks from the Transformer paradigm by combining linear recurrences with local attention to achieve fixed-size memory during inference. This innovation delivers Transformer-comparable accuracy but with dramatically improved inference speed and efficiency, enabling long-sequence processing on smaller hardware—a breakthrough for scalable, resource-constrained AI applications that require handling lengthy or streaming text.

---

## 1. Executive Summary

This paper introduces **RecurrentGemma**, a family of open language models based on Google's Griffin architecture that replaces global attention with a mixture of linear recurrences and local attention to achieve efficient inference on long sequences. The authors release pre-trained and instruction-tuned variants at 2B and 9B parameter scales, trained on 2T tokens, and benchmark them against the transformer-based Gemma models across a broad suite of academic evaluations, human preference studies, and throughput measurements. RecurrentGemma-2B achieves comparable performance to Gemma-2B despite being trained on 50% fewer tokens, and RecurrentGemma-9B achieves comparable performance to Gemma-7B despite being trained on 3× fewer tokens, with the primary architectural advantage being a fixed-size state that eliminates the linearly-growing KV cache and enables up to two orders of magnitude higher sampling throughput on long sequences. The key empirical finding establishes that linear recurrent architectures can match transformer performance on language modeling benchmarks while substantially reducing memory requirements at inference time, though this parity is demonstrated only for models trained on up to 2T tokens — a regime where the larger Gemma baselines received 3T–6T tokens of training.

## 2. Context and Motivation

### The Core Problem: The Memory Bottleneck in Transformer Inference

The fundamental problem this paper addresses is the **linear growth of memory requirements during transformer inference**. Every time a standard transformer generates a new token, it must attend to all previous tokens in the sequence. To avoid recomputing attention from scratch at each step, transformers cache the key-value (KV) representations of every prior token, storing them in what is called the **KV cache**. As the sequence grows, this cache grows linearly with it — generate 1,000 tokens, and you must store 1,000 sets of keys and values in device memory.

This growth creates a hard practical ceiling. The authors frame it directly:

> "Whereas Gemma's KV cache grows proportional to sequence length, RecurrentGemma's state is bounded, and does not increase on sequences longer than the local attention window size of 2K tokens."

The consequence is twofold. First, the maximum sequence length that can be generated is bounded by available device memory — you simply run out of room for the KV cache. Second, even for sequences that fit within memory, the cache occupies space that could otherwise be used by larger batch sizes, which are critical for amortizing the cost of loading model parameters from host memory to device memory each inference step.

This is not merely a theoretical concern. Language models are increasingly deployed in settings that demand long generation: document summarization, multi-turn dialogue, code generation, and creative writing all produce sequences that can stretch to thousands or tens of thousands of tokens. A model that can generate sequences of arbitrary length without memory pressure could serve use cases that transformer-based models simply cannot address without architectural compromises.

### The Insufficiency of Local Attention

The paper acknowledges one known workaround for the KV-cache problem: **local attention**, as introduced by Beltagy et al. (2020) in Longformer. In a local attention transformer, each token attends only to a fixed-size window of preceding tokens rather than the entire sequence. This bounds the cache size — you only need to store keys and values for tokens within the window. However, the authors note a critical trade-off:

> "Although one can reduce the cache size by using local attention, this comes at the cost of reduced performance."

Local attention sacrifices the model's ability to attend to distant context. For tasks that require long-range dependencies — tracking a character across chapters of a story, maintaining consistency in a long dialogue, or reasoning about code that spans hundreds of lines — this information loss can be severe. The transformer's defining strength is its ability to attend to arbitrary positions in the sequence, and local attention directly limits that capability.

This creates an unpalatable choice for practitioners: accept the memory cost of global attention and live with the throughput ceiling, or adopt local attention and accept degraded performance on long-range tasks. Neither option is satisfying.

### Where Linear Recurrences Offer a Path Forward

The Griffin architecture (De et al., 2024), on which RecurrentGemma is built, proposes a third path: **replace global attention entirely with a combination of linear recurrences and local attention**. 

To understand why this is compelling, it's worth briefly unpacking what "linear recurrence" means in this context. In a standard transformer, the attention operation computes a weighted sum over all previous token representations, with the weights determined by learned query-key interactions. The cost (both compute and memory) scales quadratically with sequence length for computation and linearly for storage. In a linear recurrent model — specifically the **RG-LRU** (Real-Gated Linear Recurrent Unit) layers used in Griffin — the model maintains a fixed-size hidden state that is updated at each time step. This state serves as a compressed summary of all prior information in the sequence. When a new token arrives, the model uses a learned gating mechanism to decide what to keep from the old state and what to write from the new input, producing an updated state of the same fixed size.

The key property is that this state has **constant size** regardless of sequence length. After processing 100 tokens or 100,000 tokens, the state vector is always the same number of elements. This eliminates the linearly-growing KV cache entirely for the recurrent layers. The Griffin architecture augments this recurrence with local attention — each token attends only to a fixed window of 2,048 preceding tokens — to capture fine-grained local structure. But the global information flow is carried by the recurrence, not by attention spanning the entire sequence.

This is not a fundamentally new idea. Linear recurrent models for sequence processing date back to the early days of deep learning, and recent work on structured state space models (Gu et al., 2021, the S4 paper) and their descendants demonstrated that recurrent architectures can compete with transformers on long-range sequence tasks. The Deep Linear Recurrent Unit work by Orvieto et al. (2023) further refined these techniques. The Griffin paper synthesizes these threads into a practical architecture for language modeling at scale.

The contribution of *this* paper is not to invent the Griffin architecture but to **validate it at a competitively useful scale** — producing 2B and 9B parameter models trained on 2T tokens, releasing them as open models, and benchmarking them against production-quality transformer baselines (Gemma) on a broad evaluation suite including standard academic benchmarks, human preference studies, and throughput measurements. Prior work on linear recurrent architectures had largely demonstrated results at smaller scales or on narrower benchmarks. RecurrentGemma establishes that the approach is viable for models that practitioners might actually deploy.

### The Gap This Paper Specifically Fills

The gap is best understood by considering what existed before RecurrentGemma's release. The research community had:
- Transformer-based open models (Gemma, Llama, Mistral) that perform well but suffer from the KV-cache memory bottleneck at long sequences.
- Recurrent architectures for long-range tasks (S4, H3, Mamba) demonstrated primarily at smaller scales or on specialized benchmarks, without the full engineering infrastructure (instruction tuning, RLHF, safety evaluation, throughput optimization) that makes a model practically deployable.
- The Griffin architecture paper (De et al., 2024), which introduced the design and showed results at scale, but not in an open-release format with pre-trained and instruction-tuned checkpoints, optimized inference kernels, and human evaluation.

RecurrentGemma bridges this gap. As the authors put it:

> "We hope that RecurrentGemma will unlock novel applications of highly performant small language models in resource constrained environments."

The phrase "resource constrained environments" is telling. The target is not necessarily to outperform transformers at the frontier of scale (the paper is careful to compare against similarly-sized models, not the largest available models), but rather to provide a model that is **practically deployable** in settings where memory is the binding constraint — on-device inference, edge computing, batch processing of long documents, or any scenario where the KV cache consumes memory that could otherwise be used for serving more users or running larger batch sizes.

### How This Paper Positions Itself Relative to Existing Work

The paper positions RecurrentGemma as an **open, practically usable instantiation of the Griffin architecture** that is directly comparable to the transformer-based Gemma family. This positioning is conveyed through several deliberate choices:

**1. Training parity with Gemma, not superiority.** The paper uses the same pre-training data, the same tokenizer (a 256k-vocabulary SentencePiece tokenizer), and the same instruction-tuning and RLHF pipeline as Gemma. This ensures that any performance differences can be attributed primarily to architecture rather than data or training procedure. The authors are transparent that the training budgets are not identical: RecurrentGemma-2B is trained on 2T tokens versus Gemma-2B's 3T tokens, and RecurrentGemma-9B is trained on 2T tokens versus Gemma-7B's 6T tokens. The framing — "achieves comparable performance despite being trained on fewer tokens" — implies that the architecture may be more sample-efficient, but the paper does not make the stronger claim that RecurrentGemma *outperforms* Gemma at matched training budgets. The absence of a matched-training-budget comparison is a notable methodological choice; we'll return to it in later sections.

**2. Throughput as the primary value proposition.** The paper's central claim is not that RecurrentGemma is a better language model in terms of quality, but that it achieves similar quality **while enabling faster inference**, particularly on long sequences. Figure 1 is central to this argument: the sampling throughput of RecurrentGemma stays constant as sequence length increases, while Gemma's throughput falls. This framing naturally follows from the architectural properties — if the state size is constant, throughput should be constant — but the paper provides concrete measurements that quantify the gap (up to two orders of magnitude in the 9B comparison).

**3. Release as open models.** The paper emphasizes that RecurrentGemma is being released as open pre-trained and instruction-tuned checkpoints, with efficient JAX evaluation and fine-tuning code including a specialized Pallas kernel for executing the linear recurrence on TPUs. This distinguishes it from a pure research result and positions it as infrastructure that the community can build on. The inclusion of both JAX and PyTorch implementations acknowledges the diversity of deployment environments.

**4. Human evaluation against an external baseline (Mistral).** Rather than only comparing to Gemma, the paper includes human preference studies against Mistral 7B v0.2 Instruct — an external, widely-used open model. This is important because it tests whether RecurrentGemma's quality generalizes beyond the Gemma comparison. The results in Table 5 — RecurrentGemma-2B IT achieving a 43.7% win rate on instruction following against the larger Mistral 7B, and RecurrentGemma-9B IT achieving a 59.3% win rate — suggest that the models are competitive beyond the Gemma ecosystem.

**5. One explicit architectural modification from Griffin.** The paper makes clear that it is not simply re-releasing Griffin:

> "We make only a single modification to the Griffin architecture, which is to multiply the input embeddings by a constant equal to the square root of model width."

This small change (adopted from Gemma) is noted for transparency, and the paper defers to the Griffin paper for full architectural details. This positions RecurrentGemma as a specific, tested configuration of Griffin rather than a novel architecture, keeping the focus on the practical validation rather than the design.

### The Broader Significance: A Fork in the Road for Language Model Architecture

Stepping back from the specific technical claims, this paper represents an important inflection point in the ongoing conversation about whether transformers will remain the dominant architecture for language modeling. Since the introduction of the Transformer in 2017 (Vaswani et al.), the field has seen periodic challenges from alternative architectures — convolutional sequence models, state space models, and various recurrent designs — but none had demonstrated competitive performance at the multi-billion-parameter scale with the full training and evaluation rigor expected of production models.

RecurrentGemma does not settle this debate — the paper is careful not to claim superiority over transformers in absolute terms — but it does establish that **linear recurrent architectures are now a viable alternative in the regime of practical deployment**. The 2B and 9B parameter scales are precisely the sizes where many real-world applications operate: smaller than the largest frontier models (which are measured in hundreds of billions of parameters and are largely inaccessible to most practitioners), but large enough to capture sophisticated language understanding and generation capabilities. The fact that RecurrentGemma-9B matches Gemma-7B on average across 18 benchmarks (56.1% vs. 56.9%), with both trained on the same data and the same tokenizer, while offering dramatically better inference throughput on long sequences, is a concrete signal that the architectural choice matters for practical deployment — not just as a research curiosity.

## 3. Technical Approach

### 3.1 Reader Orientation

RecurrentGemma is a family of language models that replaces the standard transformer's global attention mechanism with a hybrid architecture combining fixed-size linear recurrent states and local attention windows, producing a model whose memory footprint remains constant regardless of how many tokens it generates. The problem it solves is the linearly-growing KV cache in transformers, which consumes increasing device memory as sequence length increases and ultimately bounds either the maximum generation length or the batch size; the solution replaces the unbounded KV cache with a bounded, compressed state that summarizes all prior context in a fixed-size vector while retaining local attention for fine-grained token-level interactions.

### 3.2 Big-Picture Architecture (Diagram in Words)

The RecurrentGemma system has five major components, organized as layers in a deep neural network:

1. **Input Embedding Layer** — converts token IDs into continuous vector representations, scaled by a width-dependent multiplicative constant to control activation magnitudes.

2. **RG-LRU Recurrent Blocks** — the core architectural innovation: gated linear recurrent units that maintain a fixed-size hidden state updated at each time step via learned gates, carrying global context forward without an attention-based KV cache.

3. **Local Attention Blocks** — standard multi-head attention restricted to a fixed window of 2,048 preceding tokens, capturing fine-grained local structure that recurrence alone might miss.

4. **MLP Blocks** — standard feed-forward layers with expansion factor 3 (the hidden dimension is 3× the model width), providing non-linear transformation capacity at each layer.

5. **Output Embedding Layer** — a tied embedding matrix (shared with the input embeddings, but without the multiplicative scaling factor) that projects the final hidden state back to vocabulary probabilities.

Information flows sequentially: input tokens → embedding lookup with scaling → alternating blocks of RG-LRU recurrence, local attention, and MLP transformations → output projection to vocabulary logits. At inference time, the RG-LRU blocks maintain a single fixed-size state vector (no growing cache), while the local attention blocks maintain a small rolling cache covering only the most recent 2,048 tokens. The two model sizes share this architecture but differ in width (2,560 vs. 4,096), depth (26 vs. 38 layers), and attention heads (10 vs. 16), with the 9B model being both wider and deeper.

### 3.3 Roadmap for the Deep Dive

- **First**, the input embedding scaling modification — the single architectural change from Griffin, adopted from Gemma, and why it matters for training dynamics.
- **Second**, the RG-LRU recurrent layer mechanics — what "gated linear recurrence" means operationally, how the state is updated, and why weight decay is not applied to these parameters.
- **Third**, the local attention mechanism — how it complements the recurrence, the window size of 2,048 tokens, and its role in the overall information flow.
- **Fourth**, the layer composition and depth pattern — how recurrent, attention, and MLP blocks are interleaved across the 26 or 38 layers, and how the model width and expansion factor determine capacity.
- **Fifth**, the embedding layer and tied weights — the 256k vocabulary SentencePiece tokenizer, why embedding parameters dominate the parameter count, and the output projection details.
- **Sixth**, the training configuration — the pre-training recipe (2T tokens, 8192-token sequences, two-phase data curriculum), the optimizer choices (no weight decay on recurrent parameters, gradient clipping on the square-root operation), and the instruction-tuning/RLHF pipeline.
- **Seventh**, the inference dynamics — precisely how the fixed-size state enables constant throughput, the distinction between prompt processing (parallel) and token generation (sequential), and where the memory savings come from.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **model release and benchmarking paper** whose core idea is that a linear recurrent architecture, when scaled to 2B and 9B parameters and trained with the same data pipeline as transformer-based baselines, can match their benchmark performance while eliminating the linearly-growing KV cache — and therefore the memory bottleneck — at inference time. The technical contribution is not a novel architecture but rather the validation, release, and throughput characterization of Griffin at a scale and with an engineering completeness that makes it a practical alternative to transformers for deployment.

---

#### The Single Architectural Modification: Input Embedding Scaling

The paper makes exactly one change to the Griffin architecture as described in De et al. (2024), and it is small but important for training stability:

> "We make only a single modification to the Griffin architecture, which is to multiply the input embeddings by a constant equal to the square root of model width."

This means that for RecurrentGemma-2B, with a model width `$d = 2560$`, the input embeddings are multiplied by `$\sqrt{2560} \approx 50.6$`. For RecurrentGemma-9B, with `$d = 4096$`, the scaling factor is `$\sqrt{4096} = 64$`.

**What this operation does.** After the embedding lookup produces a vector of dimension `$d$` for each input token, every element of that vector is multiplied by `$\sqrt{d}$`. If the raw embedding values are roughly zero-mean with unit variance (as is typical at initialization), this scaling increases their variance to approximately `$d$`. The operation can be expressed simply as:

$$\mathbf{e}_{\text{scaled}} = \sqrt{d} \cdot \mathbf{e}_{\text{raw}}$$

where `$\mathbf{e}_{\text{raw}} \in \mathbb{R}^d$` is the token embedding vector from the lookup table, `$d$` is the model width, and `$\mathbf{e}_{\text{scaled}}$` is the scaled embedding fed into the first model layer.

**Why this form.** The motivation, inherited from Gemma (Gemma Team, 2024), relates to the interaction between embedding magnitude and the residual stream in deep transformers or recurrent networks. In architectures with residual connections, the signal from the embedding layer adds to the output of every subsequent block through skip connections. If the embedding magnitudes are too small relative to the block outputs, the token identity information gets diluted in deeper layers. If they are too large, the token identity dominates and the model cannot effectively integrate contextual information from attention or recurrence. The `$\sqrt{d}$` factor provides a width-dependent scaling that keeps the embedding contribution appropriately balanced as model width increases. This is a form of **width-dependent initialization** that is applied at every forward pass, not just at initialization.

The paper specifies that this scaling is **not applied to the output embeddings** used at the final projection layer, even though the input and output embedding matrices are tied (shared). This asymmetry means the model learns to produce output logits in the original embedding space while receiving scaled inputs, which effectively means the scaling factor gets absorbed into the learned representations throughout the network rather than being "undone" at the output.

---

#### The RG-LRU Recurrent Layer: Gated Linear Recurrence

The heart of RecurrentGemma's difference from transformers is the **RG-LRU** (Real-Gated Linear Recurrent Unit), which replaces the global self-attention operation in a subset of the model's layers. To understand what this layer computes, it is useful to contrast it with both a standard RNN and a standard attention mechanism.

**What a standard transformer does at each layer.** In a transformer, each token position computes a weighted sum over all previous tokens' representations. The weight between position `$i$` and position `$j$` is determined by the dot-product similarity between a learned query vector at position `$i$` and a learned key vector at position `$j$`. This requires storing all previous keys and values in the KV cache. For a sequence of length `$L$` and hidden dimension `$d$`, the cache requires `$O(L \cdot d)$` memory.

**What an RG-LRU does instead.** At each time step `$t$`, the RG-LRU maintains a fixed-size hidden state `$\mathbf{h}_t \in \mathbb{R}^d$`. When a new input `$\mathbf{x}_t$` arrives, the layer computes two things:

1. A **candidate update** `$\mathbf{u}_t$` — what new information from the input could potentially be written into the state.
2. A **gate** `$\mathbf{g}_t$` — a vector of values between 0 and 1 that determines, for each dimension of the state, how much of the old state to retain versus how much of the candidate update to accept.

The state update rule can be expressed as:

$$\mathbf{h}_t = \mathbf{g}_t \odot \mathbf{h}_{t-1} + (1 - \mathbf{g}_t) \odot \mathbf{u}_t$$

where `$\mathbf{h}_t \in \mathbb{R}^d$` is the hidden state after processing token `$t$`, `$\mathbf{h}_{t-1}$` is the previous hidden state, `$\mathbf{g}_t \in [0, 1]^d$` is the forget-keep gate vector (element-wise), `$\mathbf{u}_t \in \mathbb{R}^d$` is the candidate update computed from the current input, and `$\odot$` denotes element-wise (Hadamard) product.

**What this equation computes.** For each of the `$d$` dimensions of the state, the gate `$\mathbf{g}_t[i]$` acts as a dimmer switch. If `$\mathbf{g}_t[i]$` is close to 1, the old state value `$\mathbf{h}_{t-1}[i]$` is largely preserved and the new input `$\mathbf{u}_t[i]$` contributes little. If `$\mathbf{g}_t[i]$` is close to 0, the old state is largely overwritten by `$\mathbf{u}_t[i]$`. This is a **gated linear recurrence** because the update is a linear interpolation between the old state and the new candidate, with the interpolation coefficients determined by learned, input-dependent gates.

**Why this form.** This gating mechanism gives the model two critical capabilities. First, it can learn to **preserve information over long distances** — if an early token contains important information (say, the subject of a sentence that must agree with a verb 100 tokens later), the model can set the corresponding gate dimensions near 1 and carry that information forward without degradation. Second, it can learn to **selectively forget** — when context becomes irrelevant (a paragraph ends, a new topic begins), gates near 0 flush the stale information. This is fundamentally different from a standard RNN like an LSTM or GRU, where the recurrence involves non-linear activation functions (tanh) that can cause gradients to vanish or explode over long sequences. The **linear** nature of the recurrence — the fact that the state is a linear combination of previous inputs — makes it more stable for gradient-based optimization over long sequences, a key insight from the state space model literature (Gu et al., 2021).

The "real-gated" qualifier in RG-LRU refers to the specific parameterization of the gate and the use of real-valued (as opposed to complex-valued) state representations, following the design choices in the Griffin paper. The paper does not reproduce the full equations for the gate computation or the candidate update; it cites De et al. (2024) for those details. However, it does specify two training details specific to these layers:

> "we do not apply weight decay to the parameters of the recurrent (RG-LRU) layers during training"

Weight decay (L2 regularization) penalizes large parameter magnitudes. The decision to exclude RG-LRU parameters from weight decay likely reflects the sensitivity of the gating dynamics to parameter scale — if decay pushes gate parameters toward zero, the gates may become pathologically biased toward either always forgetting or always retaining, breaking the layer's ability to dynamically control information flow.

> "when backpropagating through the square root operation in the recurrent layers, we always clip the derivative to a maximum value of 1000 for stability"

The square root operation appears in the gate computation (likely as part of a normalization step within the RG-LRU formulation). The square root function has derivative `$1 / (2\sqrt{x})$`, which blows up as `$x$` approaches zero. Clipping this derivative to a maximum magnitude of 1000 prevents gradient spikes from destabilizing training when gate-related quantities become very small.

**What the RG-LRU accomplishes for inference.** The critical property for the paper's claims is that `$\mathbf{h}_t$` has the same dimensionality `$d$` regardless of `$t$`. After processing 10 tokens or 10,000 tokens, the state is always a single vector of size `$d$`. At inference time, the model needs to store only the most recent state `$\mathbf{h}_t$` (plus the local attention cache, which we will address next). There is no per-token KV pair to store. This is why the state size is "fixed" and "bounded" — it grows with model width (which is fixed at architecture design time), not with sequence length.

---

#### Local Attention: Capturing Fine-Grained Structure

The RG-LRU carries global context forward in a compressed state, but compression is inherently lossy — information from 10,000 tokens ago cannot be perfectly preserved in a single vector of dimension `$d$`. To complement the recurrence, the Griffin architecture interleaves **local attention** layers that allow each token to attend directly to a fixed window of recent tokens.

**The local attention window.** The paper specifies:

> "Local attention window size: 2048"

This means that at each local attention layer, a token at position `$t$` can attend to tokens at positions `$t-2047$` through `$t$` inclusive — a context window of 2,048 tokens centered on the current position with no look-ahead. Tokens beyond this window (at positions < `$t-2047$`) are invisible to the attention operation at this layer.

This is standard multi-head attention (the paper notes 10 heads for the 2B model and 16 for the 9B model) with the sole modification that the attention mask restricts queries to only the local window rather than the full sequence. The attention computation itself — query-key dot products, softmax normalization, value aggregation — is unchanged from the transformer.

**Why local attention + recurrence, rather than one or the other.** The design choice reflects a hypothesis about the structure of language: some dependencies are **local and fine-grained** (syntactic agreement, word ordering, phrase structure), while others are **global and compressed** (topic coherence, narrative consistency, long-range factual references). Local attention excels at the former because it can compare exact token representations within a window with full quadratic attention; recurrence excels at the latter because it can carry compressed summary information across arbitrary distances without memory growth. Using both is an attempt to get the best of both worlds.

By setting the local attention window to 2,048 tokens, the model guarantees that the attention cache never exceeds 2,048 key-value pairs per layer — a small, constant overhead relative to the potentially unbounded cache of global attention.

**Interaction with the recurrent state.** The local attention layers do not see the recurrent state directly. Rather, the two mechanisms communicate through the residual stream: an RG-LRU layer updates the hidden representation of each token based on the compressed history, and a subsequent local attention layer refines that representation based on fine-grained local context. The paper does not specify the exact interleaving pattern (how many recurrent blocks per attention block), deferring to De et al. (2024) for the full block structure.

---

#### Layer Composition and Model Scaling

The two model sizes are defined by the hyperparameters in Table 1 of the paper. Rather than merely listing them, it is worth understanding how each hyperparameter shapes the model's capacity.

**Model width (`$d$`): 2,560 for 2B, 4,096 for 9B.** The model width is the dimensionality of the hidden state at every layer — the size of the residual stream, the RG-LRU state, and the attention query/key/value projections. A wider model can represent more information at each position but at quadratic cost in the feed-forward layers (since the MLP intermediate dimension is `$3d$`). The 9B model is approximately 1.6× wider than the 2B model.

**RNN width: equal to model width.** The paper specifies that the "RNN width" — the dimensionality of the hidden state in the RG-LRU layers — is equal to the model width. This means the recurrent state occupies the same vector space as the residual stream, with no dimensional bottleneck. This is a design choice: some recurrent architectures use an expanded state dimension (e.g., state size `$> d$`) to increase memory capacity, but Griffin uses a one-to-one mapping, keeping the state compact at exactly `$d$` elements.

**MLP expansion factor: 3.** The feed-forward (MLP) layers project the `$d$`-dimensional input to a `$3d$`-dimensional intermediate representation, apply a non-linearity, and project back to `$d$`. This 3× expansion is relatively standard (many transformers use 4×). It is the primary source of non-linear computation in the network; the RG-LRU is largely linear aside from the gate computation.

**Depth: 26 layers for 2B, 38 layers for 9B.** The 9B model is not just wider but also deeper — 38 layers versus 26, approximately a 1.46× increase. Combined with the 1.6× width increase, the total parameter count grows from 2.68B to 8.58B (approximately 3.2×). The depth determines how many opportunities the model has to apply the recurrent-local-MLP cycle to refine representations, and deeper models can in principle capture more abstract, compositional patterns. However, the paper does not specify exactly how the 26 or 38 layers are divided among RG-LRU blocks, local attention blocks, and MLP blocks; that detail is in De et al. (2024). The key structural insight is that not every layer has attention — a subset of layers use RG-LRU recurrence, a subset use local attention, and all layers include MLP blocks.

**Attention heads: 10 for 2B, 16 for 9B.** In the local attention layers, the query, key, and value projections are split across multiple heads, each with dimension `$d / \text{num\_heads}$`. More heads allow the model to attend to different types of relationships in parallel within the same layer. The 2B model has 10 heads with per-head dimension 256 (since `$2560 / 10 = 256$`), and the 9B model has 16 heads with per-head dimension 256 as well (since `$4096 / 16 = 256$`). The per-head dimension is held constant across model sizes, which is a common scaling pattern — increasing heads rather than per-head dimension when scaling up.

---

#### Embedding Layer and Tied Weights

The paper uses a SentencePiece tokenizer with a vocabulary size of 256,000 tokens — the same tokenizer as Gemma. This vocabulary size is unusually large compared to many open models (e.g., Llama uses 32k, Mistral uses 32k). The consequence is captured in Table 1 and explicitly noted by the authors:

> "as a consequence of this large vocabulary size, the embedding layer comprises a significant fraction of the total model parameters"

For RecurrentGemma-2B, the embedding parameters are 0.65B out of 2.68B total — approximately 24% of the total parameters. For RecurrentGemma-9B, embedding parameters are 1.05B out of 8.58B total — approximately 12% (a smaller fraction because the non-embedding layers scale more aggressively with width and depth). The "Non-Embedding params" row in Table 1 (2.03B and 7.53B) subtracts these embedding parameters, giving a sense of the "core" model size excluding the vocabulary lookup.

The embedding matrix has shape `$\text{vocab\_size} \times d$`, or `$256{,}000 \times 2{,}560$` for the 2B model and `$256{,}000 \times 4{,}096$` for the 9B model. This is a large matrix — 655 million entries for the 2B model, over 1 billion for the 9B — which explains why it accounts for such a large fraction of total parameters.

**Tied input and output embeddings.** The paper states:

> "The input and output embeddings are tied, but this factor [the `$\sqrt{d}$` scaling] is not applied to the output."

Weight tying means the same matrix is used for converting token IDs to vectors (input embedding lookup) and for converting the final hidden state to vocabulary logits (output projection, which computes `$\mathbf{h}_{\text{final}} W_{\text{emb}}^T$`). The asymmetry in scaling — input embeddings get multiplied by `$\sqrt{d}$`, output logits do not — means that during the forward pass, the same weight matrix interacts with the data differently at the input and output. At the input, a scaled token representation enters the residual stream. At the output, the unscaled final hidden state is projected onto the same weight vectors. This means the network must learn to produce hidden states that, when projected through the unscaled embedding matrix, produce correct logits, while receiving input representations that have been amplified by `$\sqrt{d}$`. In effect, the scaling factor is absorbed by the learned transformations in the body of the network.

---

#### Pre-Training Configuration

The pre-training recipe is designed to mirror Gemma's as closely as possible, enabling architectural comparison with minimal confounding from data or training procedure differences.

**Training data and token budget.** Both model sizes are trained on 2T (2 trillion) tokens. The paper emphasizes that this is less than the corresponding Gemma models:

> "Note that in contrast, Gemma-2B was pre-trained on 3T tokens and Gemma-7B was pre-trained on 6T tokens."

The data composition matches Gemma's:

> "primarily English data from web documents, mathematics and code"

with the same filtering pipeline to remove unwanted content, personal data, and evaluation sets from the training corpus.

**Sequence length: 8,192 tokens.** All training sequences are 8,192 tokens long. This is notable because it is substantially longer than the local attention window of 2,048 tokens. During training, the local attention layers attend only within their 2,048-token window, while the RG-LRU layers carry information across the entire 8,192-token sequence through their fixed-size state. This means the model is explicitly trained to use the recurrence for long-range dependencies spanning many multiples of the attention window.

**Two-phase training curriculum.** Like Gemma, RecurrentGemma uses a two-phase approach:

> "we first train on a large general data mixture, before continuing training on a smaller, higher quality dataset"

This is a common strategy for language model pre-training: the first phase builds broad linguistic competence on a large, diverse, but potentially noisy dataset; the second phase fine-tunes on curated, high-quality data to improve performance on downstream tasks and reduce undesirable behaviors. The paper does not specify the exact ratio of tokens between the two phases.

**Optimizer and gradient clipping.** The paper specifies two training details that differ from standard transformer training:

First, the exclusion of weight decay from RG-LRU parameters:

> "we do not apply weight decay to the parameters of the recurrent (RG-LRU) layers during training"

Weight decay is the standard `$\ell_2$` penalty `$\lambda \sum_i w_i^2$` added to the loss function to prevent parameter magnitudes from growing unboundedly. For RG-LRU parameters, this penalty is omitted entirely. The rationale is not explicitly stated, but the likely reason relates to the gate dynamics: weight decay pushes parameters toward zero, and in a gated recurrent layer, parameters near zero can cause gates to saturate at neutral values, degrading the layer's ability to dynamically control information flow. By excluding weight decay, the optimizer allows the RG-LRU parameters to settle at whatever magnitudes best support the gating function.

Second, the gradient clipping on the square root:

> "when backpropagating through the square root operation in the recurrent layers, we always clip the derivative to a maximum value of 1000 for stability"

The square root function `$f(x) = \sqrt{x}$` has derivative `$f'(x) = 1 / (2\sqrt{x})$`. When `$x$` is very small (close to zero), this derivative becomes arbitrarily large. For example, at `$x = 10^{-6}$`, the derivative is `$1 / (2 \times 10^{-3}) = 500$`; at `$x = 10^{-12}$`, it is 500,000. These extreme gradients can cause parameter updates that destabilize training entirely. Clipping the derivative to a maximum absolute value of 1000 — so that the gradient signal cannot exceed ±1000 — prevents isolated small-`$x$` cases from producing update steps that overwhelm the rest of the gradient signal. The threshold of 1000 is chosen as a high ceiling that allows normal gradient flow for well-behaved values while capping pathological cases.

The paper does not specify the base optimizer (likely AdamW, following Gemma and standard practice), the learning rate schedule, the batch size, or other standard training hyperparameters. These are presumably in the Gemma report or the Griffin paper.

---

#### Instruction Tuning and RLHF

The paper describes the post-training pipeline only briefly:

> "We follow a similar instruction tuning approach to Gemma (Gemma Team, 2024), including a novel RLHF algorithm to fine-tune the model to output responses with high reward."

This indicates a two-step fine-tuning process: first, supervised fine-tuning (SFT) on instruction-response pairs to teach the model to follow the designated dialogue format; second, reinforcement learning from human feedback (RLHF) to optimize for response quality as judged by a reward model. The "novel RLHF algorithm" is not described in the paper; the mention of RLHF team members in the author list suggests it is a significant effort beyond the scope of this architecture-focused paper.

The dialogue format is defined in Tables 3 and 4. The key control tokens are:

- `<start_of_turn>` and `<end_of_turn>` — delimit each turn in the conversation.
- `user` and `model` — identify the speaker for each turn.

A concrete example from the paper shows the structure for a two-turn interaction:

> `<start_of_turn>user\nKnock knock.<end_of_turn>\n<start_of_turn>model\nWho's there?<end_of_turn>\n<start_of_turn>user\nGemma.<end_of_turn>\n<start_of_turn>model\nGemma who?<end_of_turn>`

The model is trained to generate text only after `<start_of_turn>model` tokens and to stop at `<end_of_turn>`, with all previous turns (user and model) provided as context. This format is identical to Gemma's, enabling the instruction-tuned RecurrentGemma models to be used as drop-in replacements in systems designed for Gemma's chat interface.

---

#### Inference Dynamics: Why the Fixed State Enables Constant Throughput

The throughput measurements in Figure 1 are the paper's primary evidence for the practical advantage of the recurrent architecture. Understanding them requires unpacking what happens step-by-step during inference.

**Prompt processing (prefill).** When a prompt of `$P$` tokens is provided, the entire prompt is processed in parallel — the model runs the full sequence of prompt tokens through all layers simultaneously, computing hidden states and updating the RG-LRU state at each position. For the local attention layers, this means computing attention over the 2,048-token window for each position (or the full prompt if it is shorter). The paper shows that prompt processing throughput is similar for Gemma and RecurrentGemma:

> "When processing the prompt, both Gemma and RecurrentGemma achieve throughput of roughly 40K tokens per second for the 2B models and roughly 12K tokens per second for the 9B model."

This makes sense: during prefill, both architectures do roughly the same amount of compute per token (multiplications in attention, feed-forward layers, and — for RecurrentGemma — the RG-LRU updates). The memory advantage of RecurrentGemma is not yet relevant because the prompt is processed in a single forward pass.

**Token generation (decoding).** After the prompt is processed, the model generates one token at a time autoregressively. At each generation step, the model processes a single new token through all layers. For a transformer with global attention, this requires:

1. Load the full KV cache (size growing with `$P + g$`, where `$g$` is the number of tokens generated so far) from device memory.
2. Compute attention of the new token against all cached keys and values.
3. Append the new token's key and value to the cache.
4. Execute the feed-forward layers.

Step 1 and step 3 are memory-bound: the time to load and store the cache scales with its size. As `$g$` increases, more memory bandwidth is consumed by cache operations, reducing the throughput in tokens-per-second.

For RecurrentGemma, during generation:

1. Load the fixed-size RG-LRU state (size `$d$`, independent of `$g$`) and the local attention cache (size limited to 2,048 tokens, a constant once `$g + P > 2048$`).
2. Update the RG-LRU state via the gated recurrence (a small, fixed-cost operation).
3. Compute local attention over the 2,048-token window (fixed cost because the window size is constant).
4. Store the updated state and append to the local attention cache (dropping the oldest entry once the window is full).

Because no component of this process grows with `$g$`, the throughput stays constant. The paper's Figure 1 confirms this empirically: RecurrentGemma's sampling throughput is a flat line as generation length increases, while Gemma's throughput falls.

**The magnitude of the advantage.** The paper states:

> "RecurrentGemma-9B achieves particularly large (up to two orders of magnitude) improvements over Gemma-7B as shown in Figure 1b."

The "two orders of magnitude" figure refers to the gap at long generation lengths in the 9B comparison. The paper attributes part of this to Gemma-7B using Multi-Head Attention (MHA) while Gemma-2B uses Multi-Query Attention (MQA):

> "We note that this is primarily due to Gemma-7B using Multi-Head Attention, whereas Gemma-2B uses Multi-Query Attention."

In Multi-Query Attention, all attention heads share a single set of keys and values, which reduces the KV cache size by a factor equal to the number of heads. This makes the 2B comparison less dramatic because the transformer baseline already has a partially compressed cache. The 9B comparison, where the transformer baseline uses full Multi-Head Attention with separate keys and values per head, shows the full benefit of eliminating the KV cache entirely.

**Batch size implications.** An important point the paper notes is that the memory savings enable larger batch sizes:

> "the reduced memory requirement also enables RecurrentGemma to perform inference at much larger batch sizes, which amortizes the cost of loading model parameters from host memory into device memory"

Inference is typically memory-bound because the model parameters must be loaded from device memory (or worse, from host memory) for each forward pass. If the KV cache occupies a large fraction of available device memory, the batch size — and thus the throughput — is limited. By compressing the cache into a fixed-size state, RecurrentGemma frees memory for larger batches, which increases the number of tokens processed per second by amortizing the parameter-loading cost across more tokens.

**Implementation caveats.** The paper notes:

> "Figures 1a and 1b were generated using the Flax implementation of RecurrentGemma, which includes a specialized Pallas kernel for execution on TPUs. Users should expect lower throughput when using the Pytorch implementation or when using GPUs."

The Pallas kernel is a TPU-optimized implementation of the RG-LRU recurrence that is not available in standard PyTorch. The PyTorch reference implementation will be slower because it cannot use the same low-level hardware optimizations. This is a practical consideration for anyone deploying RecurrentGemma on GPU infrastructure: the throughput advantages are real but may not reach the full magnitude shown in the paper without equivalent kernel-level optimization.

---

#### Summary of Design Choices and Their Justifications

- **RG-LRU recurrence over global attention.** Eliminates the linearly-growing KV cache, enabling constant memory and constant throughput at arbitrary generation lengths, at the cost of compressing long-range context into a fixed-size state that may lose fine-grained detail.
- **Local attention (window 2,048) alongside recurrence.** Compensates for the compressed nature of the recurrent state by providing exact token-level attention within a local window, capturing syntactic and short-range dependencies that recurrence alone might miss.
- **Input embedding scaling by `$\sqrt{d}$` (from Gemma).** Balances the token identity signal against block outputs in the residual stream, preventing the embedding contribution from being diluted in deep networks.
- **No weight decay on RG-LRU parameters.** Preserves the dynamic range of gate parameters, avoiding pathological bias toward always-forgetting or always-retaining behavior that could result from `$\ell_2$` regularization.
- **Gradient clipping (max 1000) on square root in recurrence.** Prevents training instability from exploding gradients when gate-related quantities approach zero, where the square root derivative diverges.
- **256k vocabulary SentencePiece tokenizer (same as Gemma).** Provides broad token coverage for multilingual and code data, at the cost of making embedding parameters a large fraction (12–24%) of total model parameters.
- **Tied input/output embeddings with asymmetric scaling.** Shares parameters between embedding and output projection for parameter efficiency, while the scaling asymmetry means the network implicitly learns to absorb the `$\sqrt{d}$` factor in its internal representations.
- **Two-phase pre-training (general data then high-quality data).** First phase builds broad capabilities on diverse data; second phase refines on curated data for downstream quality — a standard strategy inherited from Gemma.
- **Training sequence length exceeds local attention window (8,192 vs. 2,048).** Forces the model to rely on the RG-LRU recurrence for long-range dependencies during training, ensuring the recurrent pathway is exercised and optimized.
- **Flax + Pallas kernel for TPU, PyTorch reference for GPU.** The optimized TPU implementation achieves the throughput shown in Figure 1; PyTorch/GPU users will see lower throughput, though the architectural memory advantage (fixed state size) remains regardless of implementation.

## 4. Key Insights and Innovations

### Innovation 1: The Linear Recurrent Architecture Achieves Transformer-Competitive Performance at Practical Scale — Moving Beyond Proof-of-Concept

The most consequential contribution of this paper is not the Griffin architecture itself (which was introduced in De et al., 2024), but rather the **empirical demonstration that a linear recurrent architecture can match transformer performance when scaled to 2B and 9B parameters, trained on 2T tokens, and evaluated across a comprehensive suite of 18 academic benchmarks, human preference studies, and real-world throughput measurements.** This transforms recurrent language models from a research curiosity with promising scaling properties into a **deployment-viable alternative to transformers** at model sizes where practitioners actually build applications.

**What prior work established and what it left open.** Before RecurrentGemma, the state space model and linear recurrence literature — S4 (Gu et al., 2021), H3, Mamba, RWKV, and the Griffin paper itself — had demonstrated that recurrent architectures could achieve strong perplexity on language modeling benchmarks and could, in principle, scale efficiently due to their fixed state size. However, these demonstrations had significant gaps that prevented practitioners from treating them as transformer replacements:

- Scaling demonstrations were often at smaller parameter counts (hundreds of millions, not billions) or trained on fewer tokens than comparable transformers.
- Evaluations were typically limited to perplexity or a handful of downstream benchmarks, without the comprehensive suite (MMLU, HellaSwag, HumanEval, GSM8K, MATH, TriviaQA, AGIEval, BBH in Table 2) that practitioners use to assess real-world capability.
- There was no instruction-tuned or RLHF-trained variant available — and thus no way to assess whether these architectures could support the chat/instruction-following use cases that dominate deployment.
- Throughput measurements were often theoretical or based on small-scale implementations, not on optimized kernels running production workloads at 2B+ scale.
- Safety evaluation was absent, making it impossible to assess whether the architecture introduced new failure modes.
- Models were not released as open checkpoints with inference code, preventing community adoption and independent verification.

**How RecurrentGemma fills these gaps.** The paper systematically addresses each of these limitations. It reports results across 18 benchmarks (Table 2) spanning knowledge (MMLU, TriviaQA, NQ), reasoning (ARC, GSM8K, MATH, BBH), common sense (HellaSwag, PIQA, SIQA, BoolQ, Winogrande, CQA, OBQA), and code (HumanEval, MBPP). It provides instruction-tuned variants trained with SFT and a "novel RLHF algorithm," evaluated not just on automated benchmarks but in human preference studies against the widely-used Mistral 7B v0.2 Instruct model (Table 5). It measures real throughput on TPU hardware with an optimized Pallas kernel for the RG-LRU recurrence, showing constant throughput as generation length increases (Figure 1). It evaluates safety on 10 standard academic safety benchmarks (Table 6). And it releases pre-trained and instruction-tuned checkpoints, JAX fine-tuning code with the TPU kernel, and a PyTorch reference implementation.

This completeness transforms the conversation. Before RecurrentGemma, a practitioner considering a recurrent architecture would face unknowns at every level: "Will it work at my target scale? Can I fine-tune it for chat? Will it be safe enough to deploy? Will the throughput gains actually materialize on my hardware?" After RecurrentGemma, the answer to each question is a qualified "yes, with the following evidence," backed by an open release that enables independent verification.

**The significance is in establishing viability, not superiority.** The paper is notably restrained in its claims. It does not assert that RecurrentGemma outperforms Gemma at matched training budgets — the Gemma baselines are trained on 1.5× to 3× more tokens, and the benchmark averages are comparable (44.6% vs. 45.0% for 2B; 56.1% vs. 56.9% for 9B in Table 2). This restraint is itself an important contribution: it establishes that recurrence can reach parity with transformers at these scales, leaving open whether it could surpass them with further optimization. The paper positions the architectural choice as a **throughput-for-quality tradeoff at parity** — you get the same performance with dramatically better inference efficiency — rather than claiming a Pareto improvement on both axes simultaneously. This credible, bounded claim is more persuasive than an overstatement would be.

**What makes this a fundamental contribution rather than incremental.** Prior to this release, the dominant assumption in the open language model ecosystem was that transformers were the only viable architecture. Every major open model — Llama, Mistral, Gemma, Falcon, MPT, Phi — used transformer backbones. A practitioner starting a new language model project would default to a transformer without considering alternatives, because no alternative had been validated at practical scale with production-grade engineering completeness. RecurrentGemma does not settle the question of whether recurrence is *better* than attention, but it does settle the question of whether recurrence is *viable* — and that change in the space of acceptable defaults is what makes this a fundamental contribution rather than an incremental one.

The evidence for this shift is in Table 2 and Table 5 together. Table 2 shows that on 18 academic benchmarks, RecurrentGemma-9B (56.1% average) is within one percentage point of Gemma-7B (56.9%) despite being trained on 3× fewer tokens. Table 5 shows that RecurrentGemma-9B IT achieves a 59.3% win rate on instruction following against Mistral 7B v0.2 Instruct, and even RecurrentGemma-2B IT achieves a 43.7% win rate — competitive with a model nearly 4× its size. These are not marginal results in a narrow domain; they represent broad capability parity across knowledge, reasoning, coding, and instruction-following tasks.

---

### Innovation 2: Framing the KV Cache as the Central Bottleneck — and the Fixed State as the Systematic Solution

The paper's second contribution is conceptual and diagnostic: it identifies the KV cache not as an implementation detail but as **the central architectural bottleneck that limits transformer inference**, and reframes the value proposition of recurrent architectures around this diagnosis. This framing transforms the discussion from "recurrence is an interesting alternative to attention" into "recurrence solves a specific, well-characterized problem that attention cannot solve without fundamental modification."

**The dominant prior framing.** Before this paper, the transformer's KV cache was widely understood as a memory cost (the O(L·d) storage requirement is covered in every transformer tutorial), but it was rarely framed as the binding constraint that determines what applications are feasible. The field's approach to long sequences was largely mitigation-based: use FlashAttention to reduce memory bandwidth, apply KV-cache quantization, adopt Multi-Query or Grouped-Query Attention to compress the cache per head, or truncate context. These are optimizations that reduce the constant factor in the O(L) growth but do not change the asymptotic scaling — the cache still grows with sequence length, and for sufficiently long sequences, memory will always be exhausted.

The paper reframes the problem in Section 2 by contrasting two approaches directly:

> "To perform inference, transformers must retrieve the KV cache and load it into device memory. This KV cache grows linearly with sequence length. Although one can reduce the cache size by using local attention, this comes at the cost of reduced performance. In contrast, RecurrentGemma compresses input sequences into a fixed-size state without sacrificing performance."

This three-sentence summary is deceptively powerful. It establishes: (1) the KV cache is not just a memory cost — it is a memory *bandwidth* bottleneck because it must be loaded on every generation step; (2) local attention is the obvious mitigation but it degrades quality; (3) the Griffin architecture offers a third category of solution — compression into a fixed state — that does not trade away long-range capability. This reframes the architectural choice from "which attention variant is fastest" to "do you need unbounded memory or can you operate within a fixed memory budget."

**Evidence for the bottleneck framing.** Figure 1 is central to this argument. It shows not just that RecurrentGemma is faster, but *why* it is faster: the throughput curve for Gemma falls as sequence length increases, while RecurrentGemma's stays flat. The falling curve is a direct consequence of the growing KV cache; the flat curve is a direct consequence of its absence. The two-orders-of-magnitude gap at long sequences in the 9B comparison (Figure 1b) is so large that it cannot be closed by constant-factor optimizations to the transformer — it reflects an asymptotic difference, not an implementation-quality difference.

The paper also notes a less obvious consequence of the fixed state: larger batch sizes. When the KV cache consumes a large fraction of device memory, the maximum batch size is constrained by the remaining memory available for model parameters and activations. By eliminating the per-token cache, RecurrentGemma frees memory that can be used for larger batches, which increases throughput by amortizing parameter-loading costs. This insight connects the architectural choice to a concrete operational benefit that most throughput comparisons miss.

**Why this reframing matters.** It provides a clear decision criterion for practitioners. If your application involves generation lengths where the KV cache fits comfortably in memory, the architectural choice is largely a matter of quality and ecosystem support. But if your application involves long generations — document summarization, multi-turn dialogue, code generation, creative writing — or memory-constrained environments (edge devices, shared inference servers), the transformer's O(L) cache growth becomes the binding constraint regardless of model quality. The paper provides a documented alternative whose memory requirements are O(1) in sequence length, with quality at parity. This is a different value proposition than "recurrence is competitive with attention on perplexity" — it is "recurrence solves a specific, named, and measured problem that attention has."

The significance of this contribution is that it moves the conversation from an academic debate about architectural inductive biases to an engineering tradeoff with measurable consequences. A practitioner can look at their target generation length, their available device memory, and their quality requirements, and make an informed choice between RecurrentGemma and Gemma based on the evidence in this paper. Before this release, that decision could not be made for recurrent architectures because the evidence at scale did not exist.

---

### Innovation 3: Demonstrating That Recurrence-Attention Hybrids Achieve Parellel Prefill Without Sacrificing Sequential Efficiency

A subtle but important architectural insight in this paper is that the Griffin design — interleaving linear recurrent blocks with local attention blocks — achieves a property that pure recurrent models historically could not: **parallelizable prefill (prompt processing) at throughput comparable to transformers, combined with O(1)-memory sequential generation.** This combination is what makes RecurrentGemma practically deployable, and it is worth surfacing as an innovation distinct from the individual components.

**The historical tension.** Pure recurrent models (classic RNNs, LSTMs, GRUs) process sequences one token at a time, updating a hidden state sequentially. This means prompt processing — converting a user's input into the initial state for generation — takes time proportional to prompt length and cannot be parallelized across tokens. Transformers, by contrast, process all prompt tokens simultaneously through their attention layers, achieving much higher prompt throughput. This difference has historically made recurrent models slow not just at generation but at *everything* — even short prompts incur sequential processing cost.

The Griffin architecture resolves this tension through its hybrid design. The RG-LRU layers are indeed sequential — each token's state update depends on the previous state — but the local attention layers can process all tokens within their window in parallel during prefill, and the MLP layers are fully parallelizable. The result, shown in Figure 1, is striking:

> "When processing the prompt, both Gemma and RecurrentGemma achieve throughput of roughly 40K tokens per second for the 2B models and roughly 12K tokens per second for the 9B model."

RecurrentGemma's prompt processing throughput matches Gemma's — approximately 40K tokens/second for 2B and 12K tokens/second for 9B — despite having recurrent components that must be evaluated sequentially. This means that in a real deployment, the prompt processing time is comparable regardless of architecture, and the throughput advantage of RecurrentGemma is concentrated entirely in the generation phase where the transformer's KV cache becomes a drag.

**Why this matters for practical deployments.** In many real-world applications, the generation phase dominates total inference time. For a prompt of 2K tokens and a generation of 2K tokens, prompt processing at 40K tok/s takes ~50ms, while generation at 6K tok/s (RecurrentGemma's constant throughput from Figure 1) takes ~333ms — roughly 87% of total time. As generation length increases, this fraction grows. The paper demonstrates that the generation throughput advantage of RecurrentGemma (constant vs. decaying) applies precisely to the phase that dominates total latency, while achieving prompt processing parity on the phase that is already faster.

This insight — that a hybrid recurrence-attention design can achieve the best of both worlds (parallel prefill + constant-memory generation) — is implicit in the Griffin architecture but is validated empirically in this paper through direct throughput measurements against a transformer baseline. It provides a template for future architectures: you do not need to choose between parallel prefill and sequential efficiency; a well-designed hybrid can deliver both.

---

### Innovation 4: Validating That Recurrent Architectures Can Support the Full Post-Training Stack (Instruction Tuning, RLHF, Safety)

The paper's fourth contribution is demonstrating that recurrent architectures can undergo the same instruction-tuning, RLHF, and safety evaluation pipeline as transformers **without architectural modifications or special handling**, and that the resulting instruction-tuned models are competitive in human evaluation against established transformer-based chat models. This finding addresses a concern that any responsible practitioner would have: even if a recurrent pre-trained model matches a transformer on academic benchmarks, can it be made safe and useful for interactive deployment?

**The unstated concern.** The transformer's global attention provides a mechanism for the model to attend to any part of its input context, including safety instructions, system prompts, and conversation history. When fine-tuning a transformer for instruction-following, the training signal can shape attention patterns to prioritize these elements. With a recurrent architecture, the model must compress all of this context into a fixed-size state — there is no mechanism to "look back" at the exact text of a safety instruction provided 5,000 tokens ago. It was not obvious, prior to this work, that this compression would preserve enough fidelity for the model to reliably follow instructions or adhere to safety constraints in long conversations.

The paper addresses this concern implicitly through its results. The instruction-tuned RecurrentGemma models are evaluated on the same safety benchmarks as Gemma (Table 6) and in human preference studies against Mistral 7B v0.2 Instruct (Table 5). The human evaluation results are particularly informative because they test instruction-following and safety in open-ended, creative scenarios that stress-test the model's ability to maintain context and follow nuanced instructions:

- RecurrentGemma-2B IT achieves a 43.7% win rate on instruction following against Mistral 7B v0.2 — a model with over 3× more parameters and global attention. The breakdown shows 34.5% wins, 18.3% ties, and 47.2% losses, meaning RecurrentGemma's outputs are judged better or equal to Mistral's in over half the comparisons.
- RecurrentGemma-9B IT achieves a 59.3% win rate on instruction following against the same Mistral baseline, with 50.1% wins — a majority-win result against a strong transformer baseline.
- On safety (a separate evaluation with ~400 prompts testing basic safety protocols), both RecurrentGemma models achieve approximately 60% win rates against Mistral 7B.

**The significance beyond the numbers.** These results demonstrate that the fixed-size recurrent state is a sufficient compression of conversation history for instruction-following and safety alignment, at least up to the context lengths tested. If the state were losing critical information — if safety instructions provided early in a conversation were being "forgotten" by the recurrent compression — we would expect to see degraded safety performance and human preference scores, particularly on prompts that require maintaining context across multiple turns. The fact that RecurrentGemma-9B IT *wins* a majority of comparisons against a strong transformer baseline on instruction-following suggests that, at these model scales and training budgets, the compression is not a bottleneck.

This finding is essential for adoption. If recurrent architectures required a fundamentally different alignment procedure, or if they could not match transformer safety at comparable scales, they would be research artifacts rather than deployable models. The paper shows that the standard Gemma post-training pipeline works for RecurrentGemma with no architectural modifications, producing models that are competitive on both capability and safety. This lowers the barrier for other teams to adopt recurrent architectures — they can use the same fine-tuning recipes, the same RLHF algorithms, and the same safety evaluation frameworks that they already use for transformers.

**A qualification to note.** The paper does not provide the prompts used in the human evaluation, does not specify the maximum conversation length tested, and does not break down performance by conversation length or context position. It is possible that the recurrent state does degrade on very long conversations or on instructions placed very early in the context — but this is not tested or reported. The results establish viability for typical use cases (the evaluation used ~1000 instruction-following prompts and ~400 safety prompts) without characterizing the limits of the approach.

---

### Innovation 5: Establishing That Training Token Efficiency May Differ Systematically Between Architectures — A Diagnostic for Future Investigation

A provocative but under-explored finding in the paper is that RecurrentGemma matches Gemma's performance despite being trained on substantially fewer tokens — 50% fewer for the 2B comparison and 3× fewer for the 9B comparison. The paper reports this as a contextual detail, not a central claim, but it raises a question with significant implications: **do recurrent architectures learn more efficiently per training token than transformers, and if so, why?**

**What the data shows.** From Table 2:

- RecurrentGemma-2B (2T tokens): 44.6% average across 18 benchmarks vs. Gemma-2B (3T tokens): 45.0%.
- RecurrentGemma-9B (2T tokens): 56.1% average vs. Gemma-7B (6T tokens): 56.9%.

In both comparisons, the recurrent model uses fewer training tokens — 33% fewer for the 2B comparison, 66% fewer for the 9B — yet achieves nearly identical benchmark averages. The gap is less than half a percentage point in both cases. The individual benchmark results (Table 2) show the recurrent models winning on some benchmarks (PIQA, SIQA, Winogrande for both sizes; TriviaQA for 9B) and losing on others (MMLU, GSM8K, MATH, AGIEval), suggesting the token-efficiency difference may not be uniform across task types.

**Why the paper does not treat this as a primary claim.** The authors are appropriately cautious. The training budgets are not controlled — the comparison is between models trained under different total compute budgets, different learning rate schedules optimized for each architecture, and potentially different batch sizes. The fact that RecurrentGemma was trained on fewer tokens could reflect a genuine architectural efficiency, or it could reflect that the Gemma baselines were not fully converged at their token counts (training was stopped before diminishing returns set in fully), or that the specific hyperparameter choices for the Gemma training runs were suboptimal in ways that made their token efficiency appear worse. The paper does not provide the controlled experiment — training both architectures on exactly 2T tokens with matched hyperparameters — that would isolate the architectural effect.

**Why the observation is still a significant contribution.** The finding provides a **diagnostic target** for future investigation. If the effect is real, it could indicate that recurrent architectures learn faster because the inductive bias of a fixed-size state that summarizes history is better aligned with the structure of language than full quadratic attention. Alternatively, it could indicate that the architectural constraint of compression forces the model to learn more transferable representations, acting as a regularizer that improves generalization per training token. If the effect is spurious (driven by optimizer or schedule differences), that is also important to know, because it would clarify that the token budgets in the comparison are not directly comparable.

The paper does not resolve this question, but it places it on the research agenda. This is a form of contribution — identifying an empirical pattern that calls for explanation — that is distinct from claiming a conclusively demonstrated result. It is included here as an innovation because it may prove, in retrospect, to be the most consequential finding in the paper if follow-up work confirms that recurrent architectures are genuinely more sample-efficient than transformers.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The primary quantitative evaluation uses a broad suite of 18 academic benchmarks spanning knowledge, reasoning, common sense, and code generation, as listed in Table 2. The benchmarks include MMLU (5-shot), HellaSwag (0-shot), PIQA (0-shot), SIQA (0-shot), BoolQ (0-shot), Winogrande (partial scoring), CQA (7-shot), OBQA, ARC-e, ARC-c, TriviaQA (5-shot), NQ (5-shot), HumanEval (pass@1), MBPP (3-shot), GSM8K (maj@1), MATH (4-shot), AGIEval, and BBH. Each benchmark uses its standard evaluation protocol (shot count and metric) as specified in Table 2. For human evaluation, the paper uses a held-out collection of approximately 1,000 prompts for instruction following (covering creative writing and coding) and approximately 400 prompts for safety evaluation. The paper does not provide the exact source, construction methodology, or example prompts for these human evaluation sets.

- **Base model(s).** RecurrentGemma is provided in two sizes: 2B (2.68B total parameters, 2.03B non-embedding parameters) and 9B (8.58B total parameters, 7.53B non-embedding parameters). Both are based on the Griffin architecture (De et al., 2024) with a single modification — input embedding scaling by the square root of model width — and are trained on 2T tokens. Pre-trained and instruction-tuned (IT) variants are released for both sizes. The models use a 256k-vocabulary SentencePiece tokenizer, identical to Gemma's.

- **Metrics.** For academic benchmarks, the paper reports the standard metric for each benchmark as listed in Table 2: accuracy (top-1 or partial scoring), pass@1 for code generation (HumanEval), and maj@1 for GSM8K. The paper computes an unweighted average across all 18 benchmarks as a summary statistic (reported as "Average" in Table 2). For human evaluation, the metric is win rate against the Mistral 7B v0.2 Instruct baseline, computed by breaking ties evenly and reporting the result as a percentage with 95% confidence intervals. For inference speed, the metric is maximum tokens per second on a single TPU device, measured separately for prompt processing and for autoregressive sampling at various sequence lengths. For safety benchmarks (Table 6), metrics vary: RealToxicity and Toxigen use average toxicity score (lower is better), while the remaining benchmarks (BOLD, CrowS-Pairs, BBQ, Winogender, TruthfulQA, Winobias) use accuracy or top-1 selection rate (higher is better).

- **Baselines.** The paper compares against the Gemma model family (Gemma Team, 2024): Gemma-2B (trained on 3T tokens) and Gemma-7B (trained on 6T tokens). These are transformer-based open models using the same pre-training data, tokenizer, and instruction-tuning pipeline as RecurrentGemma, differing primarily in architecture (global attention vs. recurrence + local attention) and training token budget. For human evaluation, the baseline is Mistral 7B v0.2 Instruct (Jiang et al., 2023), an external transformer-based open model with global attention and a comparable parameter count (7B). For throughput benchmarks, the baselines are Gemma-2B and Gemma-7B running on the same hardware (TPUv5e for 2B comparison, TPUv4 for 9B comparison) with an optimized Flax implementation.

- **Generation budget / compute accounting.** Throughput measurements in Figure 1 use maximum tokens per second as the unit of comparison, measured on a single TPU device (TPUv5e for 2B models, TPUv4 for 9B models). The measurement protocol is: sample from a prompt of 2K tokens for a range of generation lengths, reporting the maximum tokens per second achievable, excluding the time to process the prompt and the time to decode tokens into text. Prompt processing throughput is measured separately by processing prompts of different lengths in parallel. The paper does not report total FLOPs, parameter-by-parameter operation counts, or a matched FLOPs analysis between architectures. For the academic benchmark comparisons, no compute budget is specified or controlled — models are compared at their final training checkpoints regardless of the total training FLOPs consumed.

- **Cross-validation / statistical protocol.** For human evaluation results in Table 5, the paper reports 95% confidence intervals alongside win rates, and provides a breakdown of wins, ties, and losses. The confidence intervals are computed from the binomial distribution based on the number of comparisons (~1000 for instruction following, ~400 for safety). For academic benchmarks, no confidence intervals, standard deviations, or statistical significance tests are reported — the results in Table 2 are point estimates from a single evaluation run. The human evaluation prompts are drawn from held-out collections, but no information is provided about how the splits were constructed or whether the evaluators were blinded to model identity. For safety benchmarks in Table 6, no confidence intervals are reported.

### Main Quantitative Results

#### Academic Benchmark Performance (Table 2)

The headline finding is that RecurrentGemma achieves comparable performance to its Gemma counterparts on the 18-benchmark suite, despite being trained on substantially fewer tokens. The aggregate numbers capture this:

- RecurrentGemma-2B (2T tokens): 44.6% average vs. Gemma-2B (3T tokens): 45.0% — a difference of 0.4 percentage points.
- RecurrentGemma-9B (2T tokens): 56.1% average vs. Gemma-7B (6T tokens): 56.9% — a difference of 0.8 percentage points.

The per-benchmark results reveal that the parity is a consequence of RecurrentGemma winning on some benchmarks and losing on others, rather than uniform matching across tasks. Looking at the 9B comparison in detail:

**Where RecurrentGemma-9B outperforms Gemma-7B (Table 2):**
- PIQA: 81.3 vs. 81.2 (+0.1)
- SIQA: 52.3 vs. 51.8 (+0.5)
- Winogrande: 73.6 vs. 72.3 (+1.3)
- CQA: 73.2 vs. 71.3 (+1.9)
- TriviaQA: 70.5 vs. 63.4 (+7.1) — the largest single-benchmark advantage
- BBH: 55.2 vs. 55.1 (+0.1)

**Where RecurrentGemma-9B underperforms Gemma-7B (Table 2):**
- MMLU: 60.5 vs. 64.3 (−3.8)
- HellaSwag: 80.4 vs. 81.2 (−0.8)
- BoolQ: 80.3 vs. 83.2 (−2.9)
- OBQA: 51.8 vs. 52.8 (−1.0)
- ARC-e: 78.8 vs. 81.5 (−2.7)
- ARC-c: 52.0 vs. 53.2 (−1.2)
- NQ: 21.7 vs. 23.0 (−1.3)
- HumanEval: 31.1 vs. 32.3 (−1.2)
- MBPP: 42.0 vs. 44.4 (−2.4)
- GSM8K: 42.6 vs. 46.4 (−3.8)
- MATH: 23.8 vs. 24.3 (−0.5)
- AGIEval: 39.3 vs. 41.7 (−2.4)

The pattern is notable: RecurrentGemma-9B shows particular strength on common sense reasoning (SIQA, Winogrande, CQA) and knowledge retrieval (TriviaQA, where it achieves a 7.1-point absolute improvement), but lags on knowledge-intensive benchmarks (MMLU, NQ, AGIEval) and math reasoning (GSM8K, where it trails by 3.8 points). The TriviaQA result is striking — a 7.1-point gap on a 5-shot knowledge retrieval task is substantial and suggests the recurrent architecture may be particularly effective at compressing and retrieving factual associations. However, the equally large negative gap on MMLU (−3.8 points, also knowledge-intensive) complicates this interpretation.

The 2B comparison shows a similar pattern of offsetting strengths and weaknesses, though with generally smaller gaps. RecurrentGemma-2B notably outperforms Gemma-2B on PIQA (78.5 vs. 77.3, +1.2), SIQA (51.8 vs. 49.7, +2.1), BoolQ (71.3 vs. 69.4, +1.9), and Winogrande (67.8 vs. 65.4, +2.4), while trailing on MMLU (38.4 vs. 42.3, −3.9), GSM8K (13.4 vs. 17.7, −4.3), and several others. The largest negative gap is again on MMLU and math (GSM8K, MATH), suggesting these domains may be systematically harder for the recurrent architecture at these scales, though the effect is entangled with the training token difference (Gemma baselines received more tokens, which could disproportionately benefit knowledge-intensive benchmarks).

#### Human Evaluation Against Mistral 7B v0.2 Instruct (Table 5)

The human evaluation results establish that RecurrentGemma's instruction-tuned variants are competitive with a widely-used external transformer baseline in open-ended, human-judged interaction quality:

**Instruction following (approximately 1,000 prompts testing creative writing and coding):**
- RecurrentGemma-2B IT: 43.7% win rate [95% CI: 41.8%, 45.6%], with a breakdown of 34.5% wins, 18.3% ties, 47.2% losses.
- RecurrentGemma-9B IT: 59.3% win rate [95% CI: 57.4%, 61.2%], with a breakdown of 50.1% wins, 18.3% ties, 31.5% losses.

**Safety (approximately 400 prompts testing basic safety protocols):**
- RecurrentGemma-2B IT: 59.8% win rate [95% CI: 57.1%, 62.6%], with a breakdown of 47.5% wins, 24.6% ties, 27.9% losses.
- RecurrentGemma-9B IT: 59.9% win rate [95% CI: 57.1%, 62.6%], with a breakdown of 44.6% wins, 30.7% ties, 24.8% losses.

Three observations from these results. First, the 2B RecurrentGemma model is surprisingly competitive on instruction following — a 43.7% win rate against a model with over 3× more parameters is notable, and the 18.3% tie rate suggests genuine parity in a substantial fraction of comparisons. Second, the 9B model achieves a majority-win on instruction following (50.1% wins, meaning RecurrentGemma's outputs were judged strictly better than Mistral's in more than half of all comparisons), which is a strong result against an established, widely-used baseline. Third, both models achieve essentially identical safety win rates (59.8% vs. 59.9%), with the 2B model showing a higher strict-win rate (47.5% vs. 44.6%) but lower tie rate (24.6% vs. 30.7%). This suggests that in terms of safety behavior, model scale within this range may not be the dominant factor — the instruction-tuning and RLHF procedure may matter more.

The paper does not break down the instruction-following results by task category (creative writing vs. coding), by prompt length, or by conversation turn count, so it is not possible to determine whether RecurrentGemma's performance relative to Mistral varies with context length. This is a missed opportunity, given that the architectural advantage (fixed state size) should be most relevant in long-context scenarios, and a breakdown would have tested whether the recurrent compression degrades instruction-following fidelity in longer interactions.

#### Inference Throughput Benchmarks (Figure 1)

The throughput measurements are the paper's primary evidence for the practical advantage of the fixed-state architecture. The key results are:

**Prompt processing (prefill):**
> "When processing the prompt, both Gemma and RecurrentGemma achieve throughput of roughly 40K tokens per second for the 2B models and roughly 12K tokens per second for the 9B model."

Prompt processing throughput is comparable between architectures at both model sizes, confirming that the recurrent layers do not impose a sequential bottleneck during the parallelizable prefill phase.

**Token generation (sampling):**
- For the 2B models (Figure 1a, TPUv5e), RecurrentGemma achieves higher throughput at all generation lengths, and critically, its throughput remains constant as generation length increases, while Gemma-2B's throughput falls as the KV cache grows. The exact throughput values are not reported numerically in the text — they must be read from Figure 1a — but the qualitative pattern is clear: RecurrentGemma's curve is flat; Gemma's slopes downward.
- For the 9B models (Figure 1b, TPUv4), the gap is substantially larger, described as "up to two orders of magnitude" at long generation lengths. The paper attributes this amplified gap to Gemma-7B using Multi-Head Attention (with separate keys and values per head) versus Gemma-2B's Multi-Query Attention (shared keys and values across heads), which already compresses the KV cache. The RecurrentGemma-9B advantage is thus most pronounced against the worst-case transformer cache configuration.

The paper also notes:
> "A much higher throughput can be achieved when processing input prompts compared to when generating samples, since prompt processing can be efficiently parallelized."

For both architectures, prompt processing throughput (40K or 12K tok/s) is substantially higher than sampling throughput (RecurrentGemma achieves approximately 6K tok/s for the 2B model, read from Figure 1), meaning the generation phase will dominate total latency unless the prompt is vastly longer than the desired generation.

**Implementation dependence.** The paper explicitly qualifies:
> "Figures 1a and 1b were generated using the Flax implementation of RecurrentGemma, which includes a specialized Pallas kernel for execution on TPUs. Users should expect lower throughput when using the Pytorch implementation or when using GPUs."

This means the absolute throughput numbers are TPU + custom-kernel specific, and the precise magnitude of the advantage will differ on GPU hardware or in PyTorch. The architectural property — constant throughput with increasing generation length — should hold regardless of implementation, but the two-orders-of-magnitude gap at long sequences is a hardware + kernel + architecture combined effect, not an architecture-alone result.

#### Safety Academic Benchmarks (Table 6)

The paper reports results on 10 standard safety benchmarks for both pre-trained (PT) and instruction-tuned (IT) variants of both model sizes. The results do not include a comparison against Gemma's safety benchmark performance, making it difficult to assess whether the recurrent architecture introduces new safety failure modes. The reported values in Table 6 show generally comparable or improved safety metrics from pre-trained to instruction-tuned variants for most benchmarks, suggesting the instruction-tuning and RLHF pipeline is effective for the recurrent architecture. However, without the Gemma comparison, this table primarily serves as documentation of baseline safety performance rather than evidence about architecture-specific safety properties.

### Ablation Studies and Robustness Checks

This section is notable for what the paper *does not* include. The paper conducts no formal ablation studies of architectural components, training hyperparameters, or data choices. There is no comparison of RecurrentGemma with and without local attention, no sweep over local attention window sizes, no analysis of the effect of the input embedding scaling factor, no test of the gradient clipping threshold, and no comparison of weight decay schedules for RG-LRU parameters versus other layers. The paper is a model release and benchmarking report, not an architecture analysis paper — the ablations and design exploration are in De et al. (2024) for the Griffin architecture itself.

What the paper does provide falls into the category of robustness checks through evaluation diversity rather than controlled ablation:

**Robustness to benchmark diversity (Table 2).** The 18-benchmark suite covers substantially different task types — knowledge retrieval (MMLU, TriviaQA, NQ, ARC), common sense reasoning (HellaSwag, PIQA, SIQA, BoolQ, Winogrande, CQA, OBQA), mathematical reasoning (GSM8K, MATH), code generation (HumanEval, MBPP), and general language understanding (AGIEval, BBH). The fact that RecurrentGemma's average performance matches Gemma's despite individual benchmark variation suggests that the architecture is not brittle to any particular evaluation category, though the consistent underperformance on math benchmarks (GSM8K: −3.8 for 9B, −4.3 for 2B; MATH: −0.5 for 9B, −0.8 for 2B) warrants further investigation.

**Robustness to model scale (2B vs. 9B).** Both model sizes show comparable aggregate performance to their Gemma counterparts at different training token ratios (2T vs. 3T for 2B; 2T vs. 6T for 9B), suggesting the architecture scales consistently within this range. However, the paper only provides two data points on the scaling curve — a third intermediate size or a smaller model would have provided much stronger evidence about whether the recurrent architecture scales predictably with parameter count and training tokens.

**Robustness to evaluation type (automated vs. human).** The human evaluation results (Table 5) corroborate the automated benchmark findings in the sense that RecurrentGemma-9B IT is competitive with or superior to a strong external transformer baseline (Mistral 7B) — the 59.3% instruction-following win rate and 59.9% safety win rate both indicate that the quality generalizes beyond the specific set of automated benchmarks in Table 2. This is particularly important because automated benchmarks can reward surface-level patterns that do not reflect genuine instruction-following quality in open-ended interactions.

**The missing training-token-controlled comparison.** The single most important ablation that is absent is a comparison between RecurrentGemma and Gemma trained on exactly the same number of tokens. The paper reports that RecurrentGemma was trained on 2T tokens while Gemma-2B received 3T and Gemma-7B received 6T. This confounds architectural differences with training budget differences. The paper's framing ("achieves comparable performance despite being trained on fewer tokens") implies that the recurrent architecture is more sample-efficient, but without the controlled experiment, it is equally consistent with the hypothesis that Gemma was undertrained at 3T/6T tokens (i.e., that additional tokens would have produced diminishing returns for the transformer) and that both architectures would converge at similar performance given sufficient training. A single data point — training RecurrentGemma on 3T or 6T tokens — would have resolved this ambiguity.

**The missing local-attention-only baseline.** While not an ablation per se, the paper does not include a baseline that uses only local attention without recurrence. This would help disambiguate whether the recurrent layers are genuinely necessary for the reported performance or whether local attention alone (with its bounded cache) could achieve similar results. The Griffin paper (De et al., 2024) includes such comparisons, but this paper does not reproduce them.

**The missing context-length stress test.** Given that the architectural advantage is most relevant for long sequences, the paper would be strengthened by benchmarking performance as a function of context length — e.g., perplexity or downstream task accuracy at 2K, 4K, 8K, 16K, and 32K tokens. The 8,192-token training length and the 2,048-token local attention window raise questions about how well the model performs on sequences that are long enough for the global-attention versus recurrence distinction to matter materially. The throughput measurements confirm that RecurrentGemma is faster at long sequences, but they do not confirm that generation quality is maintained at those lengths.

### Critical Assessment

#### Claim 1: RecurrentGemma achieves comparable performance to similarly-sized Gemma models.

This claim is demonstrated by Table 2, but the demonstration is qualified by the training token mismatch. RecurrentGemma-2B (2T tokens) achieves 44.6% average vs. Gemma-2B (3T tokens) at 45.0%. RecurrentGemma-9B (2T tokens) achieves 56.1% average vs. Gemma-7B (6T tokens) at 56.9%. The aggregate numbers support the "comparable performance" claim — gaps of 0.4 and 0.8 percentage points on an 18-benchmark average are small. However, the comparison is not at matched training budgets, which means the "comparable" framing could mask either an architectural advantage (if RecurrentGemma would outperform Gemma at equal tokens) or an architectural disadvantage (if Gemma is already near convergence at its token count and RecurrentGemma is merely catching up). The paper does not provide the experiment that would distinguish these possibilities. The claim as stated — comparable performance despite fewer tokens — is true on its face from the reported numbers, but its interpretation depends on what one believes about how much additional performance Gemma would gain from its extra tokens, which the paper does not establish.

The per-benchmark variation introduces a further qualification: the architectures appear to have different strength profiles, with RecurrentGemma stronger on common-sense reasoning and retrieval (Winogrande, SIQA, TriviaQA) and Gemma stronger on knowledge-intensive QA (MMLU, NQ) and math reasoning (GSM8K, MATH). "Comparable performance" on average does not imply equivalent behavior on individual tasks, and a practitioner choosing between architectures should consider their target task distribution, not just the aggregate metric. The paper does not provide guidance on this task-level tradeoff.

#### Claim 2: RecurrentGemma enables efficient inference on long sequences, with throughput advantages over transformers that grow with sequence length.

This claim is well-supported by Figure 1 for the specific hardware and implementation tested (TPU with custom Pallas kernel for recurrence, optimized Flax for Gemma). The constant-throughput property is a direct consequence of the fixed state size and is demonstrated empirically across a range of generation lengths. The "up to two orders of magnitude" advantage for the 9B comparison is measured on TPUv4 with a particular implementation; the paper appropriately caveats that PyTorch/GPU users will see different (likely lower) absolute throughput.

However, the claim as stated — "efficient inference on long sequences" — conflates throughput with quality. The paper demonstrates that RecurrentGemma generates tokens *faster* on long sequences, but does not demonstrate that the *quality* of those generated tokens remains comparable to Gemma's at long sequence lengths. The academic benchmarks in Table 2 do not test generation quality as a function of context length. The human evaluation in Table 5 uses prompts drawn from a held-out collection whose length distribution is not reported. A model could have excellent throughput on long sequences while producing incoherent or repetitive text — the paper provides no evidence that this does not happen. The efficiency claim is supported for *speed* but not for *speed at maintained quality*, which is the more relevant metric for deployment.

Additionally, the throughput measurements exclude the time to process the prompt and the time to decode tokens into text, as noted in the paper:
> "we do not account for the time required to process the prompt or the time required to convert the output sequence from a list of token ids into the final text string."

This makes the throughput numbers upper bounds on real-world performance; total end-to-end latency would include these costs. For applications with short prompts relative to generation length, the exclusion is minor; for applications with very long prompts, prompt processing time could be a significant fraction of total latency and is comparable between architectures.

#### Claim 3: RecurrentGemma compresses input sequences into a fixed-size state without sacrificing performance.

This claim requires unpacking "without sacrificing performance." If the intended meaning is "without sacrificing performance *relative to the transformer baseline*," then Table 2 supports it in aggregate (44.6% vs. 45.0%, 56.1% vs. 56.9%), with the training-token caveat discussed above. If the intended meaning is "the compression is lossless with respect to information in the input sequence," then the paper provides no direct evidence — no probing experiments, no ablation showing what information the recurrent state retains at different distances, and no comparison of the model's behavior when critical information is placed at different positions in the context. The fact that RecurrentGemma underperforms on knowledge-intensive benchmarks like MMLU and NQ could be interpreted as evidence that the compression *does* sacrifice some information relevant to these tasks, though the TriviaQA result (where RecurrentGemma-9B outperforms Gemma-7B by 7.1 points) complicates this story.

The paper also does not test the limits of the fixed-state compression. The training sequence length is 8,192 tokens — four times the local attention window of 2,048 tokens — so the model was trained to carry information across distances up to ~8K tokens through the recurrent state alone. Whether the state remains effective at 16K, 32K, or 100K tokens (all of which are supported by the constant-memory property) is entirely untested. The fixed-sized state theoretically supports arbitrary-length sequences, but the paper provides no evidence about whether the *quality* of the compressed representation degrades beyond the training length. A practitioner deploying RecurrentGemma for very long document tasks would be operating in an untested regime.

#### Claim 4: RecurrentGemma supports the full post-training stack (instruction tuning, RLHF, safety evaluation) without architectural modification.

This claim is supported by the existence of the instruction-tuned and RLHF-trained checkpoints and by the human evaluation and safety benchmark results. The models were fine-tuned using the same pipeline as Gemma (as stated in Section "Instruction tuning and RLHF"), and the resulting models are competitive with Mistral 7B v0.2 Instruct in both instruction-following and safety evaluations. The claim that no "architectural modification" was needed is slightly overstated — the paper does not describe any modifications to the fine-tuning procedure, but it also does not describe the procedure in enough detail to confirm that none were made. The statement "We follow a similar instruction tuning approach to Gemma" leaves room for unspecified differences.

More subtly, the paper does not demonstrate that the RLHF-trained RecurrentGemma models exhibit the same kinds of behavioral improvements over the SFT-only variants that are typically observed with transformers. There is no head-to-head comparison of RecurrentGemma IT (SFT + RLHF) versus an SFT-only RecurrentGemma variant, so the marginal benefit of RLHF for the recurrent architecture is not established. The safety benchmarks in Table 6 show PT and IT variants, but these do not isolate the RLHF contribution (the IT variant includes both SFT and RLHF).

#### What Would Strengthen the Paper

Several experiments would substantially strengthen the evidence for the paper's claims:

1. **Training-token-controlled comparison.** Train RecurrentGemma and Gemma on exactly the same number of tokens (e.g., 2T for both, or 3T for both) with matched hyperparameter tuning budgets. This would isolate the architectural effect from the training budget effect.

2. **Quality-at-length evaluation.** Measure benchmark performance or perplexity as a function of context position and total sequence length, up to at least the training length (8K tokens) and ideally beyond. This would test whether the fixed-size state maintains information fidelity at the sequence lengths where the throughput advantage matters most.

3. **Breakdown of human evaluation by prompt length.** Report instruction-following win rates stratified by prompt/conversation length to test whether the recurrent architecture's advantage or disadvantage varies with context length.

4. **Comparison of RecurrentGemma with Gemma at matched inference batch size.** The throughput measurements use maximum achievable throughput for each model individually. A comparison where both models are constrained to the same memory budget — so that RecurrentGemma's memory savings translate to larger batch sizes — would better represent deployment scenarios where memory is the binding constraint.

5. **Ablation of the local attention window size.** Test whether the 2,048-token window is necessary for the reported performance or whether a smaller window (reducing memory further) or larger window (improving local context) would shift the quality-throughput tradeoff.

6. **A deeper investigation of the TriviaQA advantage and MMLU/GSM8K disadvantage.** The 7.1-point advantage on TriviaQA for the 9B model is the largest single-benchmark gap in Table 2. Understanding whether this reflects a genuine architectural strength for retrieval-style tasks or an idiosyncrasy of training data or evaluation would inform whether the result generalizes. Similarly, the consistent math disadvantage (−4.3 on GSM8K for 2B, −3.8 for 9B) warrants investigation into whether recurrence is fundamentally less suited to multi-step symbolic reasoning.

## 6. Limitations and Trade-offs

### 1. Training Token Mismatch Confounds the Architecture-Performance Comparison

**The assumption or constraint.** The central quantitative comparison in the paper — RecurrentGemma versus Gemma on academic benchmarks (Table 2) — compares models trained on substantially different numbers of tokens. RecurrentGemma-2B and RecurrentGemma-9B are both trained on 2T tokens, while Gemma-2B received 3T tokens (50% more) and Gemma-7B received 6T tokens (3× more). The paper acknowledges this explicitly:

> "Note that in contrast, Gemma-2B was pre-trained on 3T tokens and Gemma-7B was pre-trained on 6T tokens."

The paper frames this as evidence of architectural efficiency ("achieves comparable performance despite being trained on fewer tokens"), but this framing implicitly assumes that the larger training budgets would have meaningfully improved RecurrentGemma's performance as well — an assumption that is never tested.

**The consequence.** Because training budget and architecture are confounded, we cannot determine from Table 2 alone whether the recurrent architecture is genuinely more sample-efficient than the transformer architecture, or whether both architectures are simply near convergence at their respective token counts. The performance parity (44.6% vs. 45.0% for 2B, 56.1% vs. 56.9% for 9B) could reflect three very different underlying realities: (1) RecurrentGemma is genuinely more efficient and would outperform Gemma at equal tokens; (2) Gemma is near convergence and additional tokens provide minimal gains, meaning RecurrentGemma is merely catching up; or (3) the architectures have different scaling trajectories that happen to intersect at these particular token budgets. Each interpretation implies a different recommendation to a practitioner — train a recurrent model to match transformer quality with fewer tokens, train a transformer and expect parity with recurrence at larger budgets, or expect the relative ordering to shift with scale. The paper provides no evidence to distinguish these.

**What evidence exists in the paper.** The token mismatch is stated in the "Pre-training" section but is not controlled experimentally. There is no comparison of RecurrentGemma and Gemma trained on exactly the same number of tokens. The per-benchmark results in Table 2 show that the performance parity masks offsetting strengths and weaknesses — RecurrentGemma-9B leads by 7.1 points on TriviaQA but trails by 3.8 points on MMLU and GSM8K — which means the aggregate comparison is sensitive to the benchmark mix. Whether these task-level differences are architectural or an artifact of different convergence rates at different token budgets is unanswerable from the reported experiments.

**Mitigation status.** The paper does not attempt to address this limitation. The authors present the token mismatch as a factual detail rather than a confound to be resolved, and the framing ("despite being trained on fewer tokens") treats it as supporting evidence for the architecture rather than a limitation of the comparison. No matched-training-budget experiment is conducted or proposed as future work. A single additional data point — training RecurrentGemma-9B on 6T tokens, or training RecurrentGemma-2B on 3T tokens — would have resolved the ambiguity and is well within the scope of a model release paper from a team with the demonstrated compute resources.

---

### 2. Generation Quality at Long Sequences Is Entirely Unverified

**The assumption or constraint.** The paper's primary value proposition is that RecurrentGemma enables efficient inference on long sequences through its fixed-size state. The throughput measurements in Figure 1 demonstrate that token generation speed remains constant as sequence length increases, and the paper explicitly claims this as the architectural advantage:

> "RecurrentGemma compresses input sequences into a fixed-size state without sacrificing performance."

However, "without sacrificing performance" is only measured on throughput, not on generation *quality*. The paper provides no evidence about whether the model produces coherent, factual, or task-appropriate text at sequence lengths where the throughput advantage matters.

**The consequence.** A practitioner considering RecurrentGemma for long-generation applications — document summarization, multi-turn dialogue, code generation for large codebases, creative writing of extended narratives — has no information about whether the model's quality degrades at length. The fixed-size recurrent state theoretically supports arbitrary-length sequences, but the model was trained on sequences of 8,192 tokens, meaning its recurrent state has only been optimized to carry information across distances up to ~8K tokens. At 16K, 32K, or 100K tokens — all supported by the constant-memory property — the state may lose critical information, leading to incoherent, repetitive, or factually incorrect generation. The local attention window of 2,048 tokens provides exact token-level access to recent context, but information beyond this window relies entirely on the recurrent compression, whose quality at distances beyond the training length is uncharacterized.

This is not a hypothetical concern. The model's underperformance on benchmarks that require integrating information across long contexts — such as GSM8K (multi-step math reasoning, −4.3 points for 2B, −3.8 for 9B versus Gemma) and MMLU (knowledge retrieval that may depend on long-range dependencies in pre-training, −3.9 for 2B, −3.8 for 9B) — could be early signals that the recurrent compression degrades at distances relevant for complex reasoning. The paper does not investigate this possibility.

**What evidence exists in the paper.** The academic benchmarks in Table 2 are evaluated under standard protocols that do not systematically vary context length. Most of the benchmarks (MMLU, HellaSwag, PIQA, BoolQ, etc.) use relatively short contexts — a few hundred to a few thousand tokens — that fit comfortably within the local attention window. The human evaluation in Table 5 uses ~1000 prompts drawn from a held-out collection whose length distribution is not reported; we cannot determine whether the evaluated interactions stressed the model's long-range context capabilities. The throughput measurements in Figure 1 test speed, not quality.

The paper provides no perplexity-as-a-function-of-position measurements, no probing experiments on state information retention, and no downstream task evaluation at systematically varied context lengths. The training length of 8,192 tokens is noted, but the paper does not test whether performance holds at 8K tokens, let alone beyond.

**Mitigation status.** Not addressed. The paper neither reports quality-at-length measurements nor acknowledges this as a gap. The claim of "without sacrificing performance" is supported only for throughput, not for generation quality. This is the single most significant limitation for the paper's stated use case (efficient inference on long sequences) because it leaves unanswered the question a practitioner must ask: "If I use this model for my long-generation task, will the output be good?" The paper provides no evidence either way.

---

### 3. Throughput Measurements Are Hardware- and Implementation-Specific and Exclude End-to-End Costs

**The assumption or constraint.** The throughput benchmarks in Figure 1 are measured on TPUs (TPUv5e for 2B models, TPUv4 for 9B models) using a Flax implementation with a custom Pallas kernel for the RG-LRU recurrence. The paper acknowledges this explicitly:

> "Figures 1a and 1b were generated using the Flax implementation of RecurrentGemma, which includes a specialized Pallas kernel for execution on TPUs. Users should expect lower throughput when using the Pytorch implementation or when using GPUs."

The measurements also exclude prompt processing time and token-to-text decoding time from the sampling throughput metric:

> "we do not account for the time required to process the prompt or the time required to convert the output sequence from a list of token ids into the final text string."

**The consequence.** The absolute throughput numbers — and the "up to two orders of magnitude" gap between RecurrentGemma-9B and Gemma-7B — are specific to the TPU + custom kernel + Flax combination and do not transfer to the environments where many practitioners deploy models (GPU + PyTorch). The Pallas kernel is a significant engineering investment that provides low-level hardware optimization for the RG-LRU recurrence; without an equivalent kernel, the PyTorch implementation will execute the recurrence less efficiently, potentially narrowing the throughput gap substantially. The paper does not report PyTorch throughput numbers, so a practitioner deploying on GPUs cannot estimate their expected throughput from the paper's figures.

The "two orders of magnitude" figure is also a comparison against Gemma-7B using Multi-Head Attention, which the paper notes is a worst-case transformer configuration. Gemma-2B uses Multi-Query Attention (which compresses the KV cache) and shows a much smaller gap (Figure 1a vs. 1b). The choice of MHA for the 7B model is an architectural decision within the Gemma family, but many contemporary transformer models (e.g., Llama, Mistral) use Grouped-Query Attention or Multi-Query Attention specifically to reduce KV-cache pressure. The throughput advantage of RecurrentGemma over these optimized transformers may be substantially smaller than the "two orders of magnitude" headline.

Excluding prompt processing and decoding from the throughput metric means the reported numbers are upper bounds. For applications with long prompts relative to generation length, prompt processing cost — which is comparable between architectures (~40K tok/s for 2B, ~12K tok/s for 9B) — could dominate total latency. For applications with very short generations (e.g., classification, short-form QA), the throughput advantage during generation may be irrelevant because generation is a small fraction of total time. The paper does not provide end-to-end latency measurements that would allow a practitioner to estimate real-world performance for their specific prompt-to-generation ratio.

**What evidence exists in the paper.** Figure 1 and the accompanying text in the "Inference Speed Benchmarks" section provide the measurements and the caveats. The distinction between prompt processing throughput and sampling throughput is shown, and the hardware/implementation dependence is acknowledged.

**Mitigation status.** Partially addressed through transparency. The paper clearly states the measurement conditions and caveats. The release of a PyTorch reference implementation enables practitioners to benchmark on their own hardware, though no such benchmarks are provided in the paper. The missing piece is PyTorch/GPU throughput numbers that would let a practitioner on standard infrastructure assess the expected advantage without first running their own benchmarks. The paper also does not discuss whether a Pallas-equivalent optimization for the RG-LRU recurrence on GPUs (e.g., a CUDA kernel) is feasible or planned, which would determine whether the full throughput advantage can eventually be realized on non-TPU hardware.

---

### 4. The Local Attention Window Creates a Hidden Context Length Limitation

**The assumption or constraint.** RecurrentGemma uses a local attention window of 2,048 tokens, as specified in Table 1. This means that while the RG-LRU layers can theoretically carry information across arbitrary distances through the fixed-size state, the local attention layers — which provide fine-grained token-to-token interaction within the window — can only attend to the 2,048 most recent tokens. Information older than 2,048 tokens is only accessible through the compressed recurrent state, with no mechanism for the model to "look back" at the exact token representations.

The paper frames this as a feature — a way to limit the attention cache to a constant size — but does not discuss the implications for tasks that require exact retrieval or precise comparison of tokens more than 2,048 positions apart.

**The consequence.** For tasks that require the model to quote, reference, or precisely manipulate text that appeared more than 2,048 tokens ago, RecurrentGemma has no architectural mechanism to do so. The transformer with global attention can attend directly to any previous token and retrieve its exact representation; RecurrentGemma must rely on the recurrent state having preserved the relevant information in a compressed form, and on the model's ability to decode that compressed representation back to something useful.

This limitation may partially explain the performance gaps observed on knowledge-intensive benchmarks. MMLU (−3.8 for 9B vs. Gemma-7B) and NQ (−1.3 for 9B) test factual recall; GSM8K (−3.8 for 9B) and MATH (−0.5 for 9B) test multi-step reasoning where intermediate results must be carried forward. If the recurrent state loses fidelity in its compression of factual details or intermediate reasoning steps, these tasks would suffer. The local attention window of 2,048 tokens means that any reasoning chain or factual context that exceeds this length must be compressed, with potential information loss at each compression step.

This is not a limitation of the Griffin architecture per se — the window size could be increased at the cost of more memory for the local attention cache — but the paper's specific configuration (2,048-token window) creates a hidden context-length ceiling that is not discussed. A practitioner might reasonably assume from the "fixed-size state" claim that RecurrentGemma supports unbounded context with no quality degradation, when in fact the window size imposes a qualitative change in the model's access pattern to information beyond 2,048 tokens.

**What evidence exists in the paper.** The local attention window size is specified in Table 1. The paper does not measure performance as a function of distance from the current token, does not ablate the window size, and does not discuss the interaction between the window size and task performance.

**Mitigation status.** Not addressed. The window size is presented as a configuration parameter without discussion of its implications. The Griffin paper (De et al., 2024) may provide ablations or analysis of the window size's effect, but this paper — which is the deployment-facing release — does not. A practitioner needs to understand that RecurrentGemma's "unbounded context" claim applies to *memory usage*, not necessarily to *information fidelity*, and that the 2,048-token window creates a regime change in how the model accesses information. The paper does not provide the information needed to set expectations about this regime change.

---

### 5. Safety Benchmark Comparison Against Gemma Is Absent, Leaving Architecture-Specific Safety Properties Unknown

**The assumption or constraint.** The paper reports safety benchmark results for RecurrentGemma (Table 6) but does not compare them against Gemma's safety benchmark performance. The table shows results for both pre-trained and instruction-tuned RecurrentGemma variants across 10 standard safety benchmarks (RealToxicity, BOLD, CrowS-Pairs, BBQ, Winogender, TruthfulQA, Winobias, Toxigen), but no Gemma baseline is provided. The paper states:

> "We follow the same safety mitigations as described in the Gemma release."

However, following the same mitigations does not guarantee equivalent outcomes. The architecture could interact with the safety fine-tuning in ways that produce different safety profiles, and without the comparison, a practitioner cannot assess whether RecurrentGemma is safer, less safe, or comparable to Gemma on these metrics.

**The consequence.** A safety-conscious practitioner choosing between RecurrentGemma and Gemma for deployment has no comparative safety data to inform their decision. The human evaluation in Table 5 does show competitive safety win rates against Mistral 7B (~60% for both RecurrentGemma sizes), but this tests interactive safety behaviors in response to ~400 prompts, not the broader set of biases and toxicities captured by academic safety benchmarks. The academic benchmarks in Table 6 cover substantially different safety dimensions (gender bias in Winogender, racial bias in BBQ, toxicity in RealToxicity) that may not be captured by the interactive safety evaluation.

Without the Gemma comparison, the safety benchmark results in Table 6 serve as documentation of baseline performance rather than evidence about architecture-specific safety properties. A finding that RecurrentGemma is systematically more toxic or more biased than Gemma on certain benchmarks — or systematically less so — would be directly relevant to deployment decisions. The paper provides no basis for making such a comparison.

**What evidence exists in the paper.** Table 6 reports absolute safety benchmark scores for RecurrentGemma PT and IT variants at both model sizes. No Gemma scores are reported. The human evaluation (Table 5) provides a comparative safety assessment against Mistral 7B but not against Gemma.

**Mitigation status.** The paper does not acknowledge this gap. The safety section ("Responsible Deployment") states that "our final models were also subjected to ethics and safety evaluations by an independent team before release," suggesting internal safety review occurred, but the comparative results are not shared. A practitioner cannot determine from the paper whether the recurrent architecture introduces, amplifies, or mitigates any safety concerns relative to the transformer baseline. This is a significant omission for a model release paper that explicitly targets deployment in "resource constrained environments," where safety failures may have fewer layers of oversight and mitigation.

---

### 6. Single Architecture Configuration and Model Scale Range Limit Generalizability of Findings

**The assumption or constraint.** The paper evaluates exactly two model sizes (2B and 9B parameters), one local attention window size (2,048 tokens), one ratio of recurrent to attention layers (unspecified but inherited from Griffin), and one training sequence length (8,192 tokens). The architecture is the Griffin design with a single modification (input embedding scaling). The paper does not explore how the performance and throughput properties change with architectural hyperparameters, model scale, or training length. The authors defer these questions to the Griffin paper:

> "We define the key model hyper-parameters for both RecurrentGemma-2B and RecurrentGemma-9B in Table 1, and defer the reader to De et al. (2024) for exact details on the overall architecture."

**The consequence.** A practitioner who needs a model of a different size (e.g., 500M for extreme resource constraints, or 20B+ for higher quality) has no information about how RecurrentGemma's performance and efficiency properties scale. The paper provides exactly two points on the scaling curve, and the relationship between model width (2,560 vs. 4,096) and aggregate benchmark performance (44.6% vs. 56.1%) is confounded with the depth increase (26 vs. 38 layers) and the training token budget (both 2T). There is no basis for predicting what a RecurrentGemma-500M or RecurrentGemma-20B would achieve.

Similarly, a practitioner who needs to operate with a different local attention window size — say, 1,024 tokens to further reduce memory, or 4,096 tokens to improve local context fidelity — has no information about the quality-throughput tradeoff of that choice. The 2,048-token window is a single point in a design space; whether it is near-optimal or arbitrary is unknown from this paper alone.

The paper's central finding — that recurrent architectures can match transformer performance at 2B and 9B scales — may or may not extrapolate to smaller or larger models. Extrapolation downward is particularly uncertain: if the recurrent state's compression capacity scales with model width, a 500M model with width ~1,280 may lose information fidelity faster than a 2B model, potentially degrading relative performance against a transformer more than at the tested scales. Extrapolation upward is equally uncertain: the gap on math reasoning (GSM8K: −3.8 for 9B vs. Gemma-7B) could widen or narrow at 20B+ scales, and the paper provides no basis for prediction.

**What evidence exists in the paper.** Two model sizes are benchmarked in Table 2. The scaling from 2B to 9B shows an increase in average benchmark score from 44.6% to 56.1%, an increase of 11.5 percentage points for a 3.7× increase in non-embedding parameters (2.03B to 7.53B) and a 1.46× increase in depth (26 to 38 layers). No intermediate or larger sizes are reported.

**Mitigation status.** The paper does not claim to provide a scaling law or to characterize the architecture across a broad range of configurations. The limitation is inherent in the model-release format: the paper describes and evaluates two specific models, not a class of architectures. A practitioner can use the released models as-is but cannot extrapolate to different scales or configurations from the evidence provided. The dependency on De et al. (2024) for architectural exploration means the information exists in the literature in principle, but the Griffin paper may not cover the same training data, token budget, or evaluation suite, making cross-paper extrapolation unreliable.

## 7. Implications and Future Directions
- Field-level impact
  - Demonstrates that non‑Transformer architectures with fixed‑size memory can reach mainstream LM accuracy while decisively outperforming on long-sequence efficiency. This pressures the default assumption that global attention is necessary for strong language modeling at small and mid scales.

- Practical applications
  - Long-form generation and streaming settings (e.g., assistants maintaining long histories), code completion over large files, log analysis, and on-device or edge deployment where memory is constrained. The constant memory footprint makes serving costs more predictable.

- Research opportunities
  - Long-context quality: Benchmark explicitly on long-range reasoning/retrieval tasks to test whether recurrence + local windows suffice, and to tune window size vs. quality trade-offs.
  - Detailed ablations: Quantify the contribution of embedding scaling, RG-LRU regularization choices, and gradient clipping to stability and accuracy.
  - Scaling laws: Explore larger Griffin models and token budgets to characterize efficiency/accuracy trade-offs at scale.
  - Kernel and hardware portability: Develop GPU-optimized recurrence kernels to narrow the gap between TPU and GPU throughput.
  - Safety and RLHF: Investigate why `Toxigen` worsens for `9B IT` and refine reward models/training to improve across all safety metrics.
  - Hybrid designs: Combine recurrence with sparse/global attention or retrieval-augmented mechanisms for tasks that truly need long-distance, non-local dependencies.

Overall, RecurrentGemma shows that Griffin’s fixed-state design can match Transformer accuracy while removing the core bottleneck of KV cache growth, delivering large real-world wins in speed and memory—especially during long generations—without requiring massive compute or model sizes (Figures 1a–1b; Table 2).
