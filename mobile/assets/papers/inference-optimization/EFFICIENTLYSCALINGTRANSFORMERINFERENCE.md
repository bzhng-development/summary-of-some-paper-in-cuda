# EFFICIENTLY SCALING TRANSFORMER INFERENCE

**ArXiv:** [2211.05102](https://arxiv.org/abs/2211.05102)

## 🎯 Pitch

This paper presents a principled, hardware-aware framework for scaling generative inference in very large Transformer models, introducing analytical models and optimal partitioning strategies that maximize efficiency and minimize latency on distributed accelerators like Google TPU v4. By combining multi-dimensional tensor partitioning, advanced memory management, and specific attention mechanisms, the authors set a new Pareto frontier—demonstrating fast, cost-effective inference for models exceeding 500 billion parameters, an achievement crucial for deploying large language models in both latency-sensitive and high-throughput production environments.

---

## 1. Executive Summary

This paper develops a systematic set of engineering principles for partitioning large Transformer models across many accelerator chips to achieve efficient, low-latency generative inference. Using the PaLM family of models (8B, 62B, and 540B parameters) on TPU v4 slices, the authors analyze the tradeoffs between different **multi-dimensional partitioning techniques**—specifically, 1D and 2D weight-stationary layouts for the feedforward layer (where weights remain fixed on each chip and activations are transferred) versus weight-gathered layouts (where activations stay stationary and weights are broadcast), alongside attention-layer partitioning strategies that exploit PaLM's multiquery attention by sharding over the batch dimension rather than the heads dimension. The combination of these strategies with low-level communication-computation overlap optimizations achieves a 29ms per-token latency during generation with int8 weight quantization and a 76% model FLOPS utilization during large-batch prefill processing of 2048-token contexts on the 540B model, while the batch-sharded multiquery attention layout enables up to 32× longer context lengths compared to multihead attention. The framework establishes that the optimal partitioning layout changes fundamentally with batch size—weight-stationary layouts minimize communication at small batch sizes during decoding, but weight-gathered layouts become more efficient as the token count grows during prefill—enabling a Pareto frontier where the 540B model can process 64 tokens of user input and generate a 64-token response in 1.9 seconds on 64 chips for interactive applications while achieving 73% overall FLOPS efficiency for throughput-oriented offline inference.

## 2. Context and Motivation

### The Core Problem: Generative Inference Has Fundamentally Different Scaling Challenges Than Training

The paper addresses a gap that becomes obvious once you compare how Transformers behave during training versus inference. During training, the model processes an entire input sequence in parallel—every token can be computed simultaneously because the full sequence is known in advance. This parallelism maps naturally onto large distributed systems: you can partition the model across thousands of chips and batch hundreds of sequences together, maximizing hardware utilization. This is why training at scales of 500B+ parameters is now practical, as demonstrated by models like PaLM (Chowdhery et al., 2022) and Megatron-Turing NLG (Smith et al., 2022).

**Inference breaks this parallelism.** During generative inference, the model produces output autoregressively—one token at a time—with each step depending on all previously generated tokens. As the authors state in Section 1:

> "the computation for each token sequentially depends on the previously generated tokens"

This means that even if you have thousands of chips available, the fundamental serial dependency between tokens prevents you from using them to parallelize across time. You can only parallelize within each forward pass of the model. This creates a cascade of interconnected problems that the paper sets out to solve systematically.

### Why This Problem Matters: Three Interlocking Bottlenecks

The importance of this problem becomes clear when you examine what happens at inference time for a 500B+ parameter model serving real users. The paper identifies three distinct bottlenecks that compound each other.

**1. The memory footprint bottleneck.** Large models simply don't fit on a single accelerator chip. The 540B PaLM model, stored in bfloat16, occupies roughly 1TB of memory just for parameters. The TPU v4 chips the authors target have 32 GiB of High Bandwidth Memory (HBM) each. This means you *must* partition the model across many chips, which introduces inter-chip communication. But partitioning is necessary even before we consider the additional memory burden of the **KV cache**—the stored key and value tensors from the attention mechanism for each layer you've already processed. 

Here's why the KV cache matters so much: during autoregressive generation, when you're producing token number 1,024 in a sequence, the model needs to attend to all 1,023 previous tokens plus the input context. Without caching, you'd recompute the keys and values for all previous tokens at every step, making inference quadratically more expensive over sequence length. With caching, you store them—but the cache size scales as:

$$2 \times n_{\text{layers}} \times d_{\text{model}} \times L_{\text{context}} \times B$$

where $B$ is batch size and $L_{\text{context}}$ is context length. For PaLM 540B with multihead attention, the authors report that at batch size 512 and context length 2048, the KV cache totals 3TB—**three times the size of the model parameters themselves**. This is the startling number from Section 2.1:

> "for batch size 512 and context length 2048, the KV cache totals 3TB, which is 3 times the size of the model's parameters"

**2. The memory bandwidth bottleneck.** Storing these tensors in HBM is only half the problem. Every single forward pass during decoding must load all model parameters *and* the entire KV cache from HBM into the chip's compute cores. The paper calls this the "memory time"—the time the computational units spend waiting for data to arrive. At small batch sizes during decoding, weight loading dominates the memory time. At large batch sizes with long contexts, the KV cache loading dominates. In both cases, the arithmetic units sit idle while waiting for data from memory.

This creates a cruel dynamic: you need many chips to provide enough aggregate HBM to store the model and KV cache, but adding chips introduces communication overhead that can eat away at any latency improvements. The paper quantifies this directly in Section 3.2.1: for a 1D weight-stationary layout, communication time remains roughly constant regardless of chip count, so as you add chips, the computation and memory loading get faster but communication eventually becomes the bottleneck.

**3. The latency-efficiency tradeoff.** Applications with tight latency requirements (like chatbots, where users expect responses in 1-2 seconds) force small batch sizes, because batching multiple requests increases per-token generation time. But small batch sizes are terrible for hardware efficiency (MFU)—the arithmetic units are even more starved for work, since there are fewer tokens to multiply against the huge weight matrices. The paper demonstrates this tension throughout Figure 1: low-latency configurations achieve 14% MFU during decoding, while high-throughput configurations achieve 33% MFU, a more than 2× difference in hardware utilization for the same model.

The practical consequence? Deploying large models interactively is *expensive*—you're using lots of chips at low efficiency to meet latency targets. Offline throughput-oriented workloads can achieve better efficiency, but they need fundamentally different partitioning strategies to do so, as we'll see throughout the paper.

### Prior Approaches and Where They Fall Short

The paper builds on a substantial body of prior work on model parallelism, but it identifies specific gaps that existing systems don't address for the inference setting.

**Training-focused partitioning strategies don't translate directly to inference.** The dominant paradigm for partitioning large models comes from training systems. Megatron-LM (Shoeybi et al., 2019) introduced the core idea of partitioning transformer layers along the feedforward dimension ($d_{ff}$) and using an all-reduce communication pattern after each MLP block. GSPMD (Xu et al., 2021) generalized this to a multi-dimensional partitioning framework where any tensor dimension can be split over any hardware axis. These approaches work well for training because training has massive parallelism across tokens within each batch.

But the paper highlights a critical asymmetry: during decoding, the batch size in tokens is very small (just $B$ tokens per step, since each sequence generates one new token). The communication patterns optimized for training—where you can amortize communication over hundreds of tokens in flight—become expensive per token. The paper's analytical framework in Section 3 explicitly models this by deriving communication time formulas in terms of $B \times L$ (total tokens in flight) and showing how optimal strategies shift with this quantity.

**FasterTransformer established GPU inference benchmarks but has scalability limits.** NVIDIA's FasterTransformer provides a widely-used benchmark suite for multi-GPU inference across model sizes up to Megatron-Turing NLG 530B. The paper uses FasterTransformer as its primary comparison point (Section 5). The key limitation the authors identify is scaling behavior: FasterTransformer's 32-way tensor parallelism achieves only 33% MFU versus 46% in their 16-way configuration, indicating a communication bottleneck at higher parallelism degrees. The authors attribute this to the nature of 1D tensor parallelism:

> "This likely indicates a communication bottleneck of scaling tensor parallelism beyond this point."

In contrast, the paper's 2D weight-stationary partitioning (Section 3.2.2) is designed specifically to continue scaling efficiently to 64-way parallelism by making communication scale as $O(1/\sqrt{n_{\text{chips}}})$ rather than being constant.

**Existing systems treat inference as a one-size-fits-all problem.** Perhaps the most significant gap the paper identifies is that prior frameworks don't distinguish between the two distinct phases of inference: **prefill** (processing all input tokens in parallel at the start) and **decode** (generating output tokens autoregressively one by one). These phases have dramatically different characteristics:

- During **prefill**, you have $B \times L_{\text{input}}$ tokens to process simultaneously. This is computation-heavy—the total FLOPs are proportional to all input tokens at once, and the KV cache loading cost is amortized because you query all positions against each other. The bottleneck tends to be compute throughput.

- During **decode**, you have only $B$ tokens (one per sequence) per forward pass. This is memory-bandwidth-heavy—you must load the entire KV cache for all previous positions just to process one new token per sequence. The bottleneck shifts to memory bandwidth.

The paper's decision to analyze these phases separately (Section 2.2) and select different partitioning strategies for each (as shown in Table 2, where prefill uses XYZ-weight-gathered layout while decode uses 2D weight-stationary) is a departure from prior work that treats inference as a uniform workload. This is both conceptually important and practically enabling—it allows the system to achieve 76% MFU during prefill and acceptable decode latency simultaneously, something a single-strategy approach cannot match.

**Multiquery attention solves a memory problem but creates a partitioning challenge.** Prior work (Shazeer, 2019; Chowdhery et al., 2022) established that multiquery attention—where query heads share a single key and value head—reduces the KV cache size by a factor of $n_{\text{heads}}$. This is undeniably valuable: for PaLM 540B with 48 heads, it reduces the cache from 3TB to 62.5GB at the same batch size and context length. 

However, the paper identifies a **new problem** that prior work hadn't addressed: multiquery attention removes the heads dimension that was previously used for parallelism. In multihead attention (Figure 4a), you can partition the Q, K, V projections over heads—each chip handles a subset of heads, with its own slice of the KV cache. In multiquery attention, the key and value are single heads shared across all queries, so if you try to partition over heads (Figure 4b), you'd need to replicate the single K and V heads to every chip, **completely losing the memory savings** that multiquery attention provides in the first place:

> "Even though the key and value tensors are shared across all heads, they must be replicated on each chip and the memory cost savings of multiquery attention are lost."

The paper's solution—partitioning over the batch dimension instead, using an all-to-all collective to shuffle the small Q, K, V tensors (one token each during decode) while keeping only a slice of the large KV cache per chip (Figure 4c and 5b)—is a specific technical innovation that enables multiquery attention's memory savings to actually be realized during inference. This is not just a partitioning trick; it's what makes long-context inference practical at scale, achieving up to 32× longer context lengths (Table 1: 43,000 tokens for optimized multiquery vs. 1,320 tokens for multihead at batch size 128).

### Positioning: A Practical Engineering Framework, Not a New Architecture

The paper positions itself not as proposing a new model architecture or training method, but rather as providing **a systematic, analytically-grounded framework for making partitioning decisions**. This is stated directly in Section 1:

> "The primary goal of this paper is to provide a set of engineering principles for how best to partition a model in order to scale Transformer inference."

This matters because the landscape of large model deployment is fragmented. Different models have different numbers of heads, different $d_{ff}/d_{\text{model}}$ ratios, different attention mechanisms (multihead vs. multiquery), and different hardware topologies to run on. A practitioner facing these choices needs to know: *when* should I use 2D over 1D partitioning? *At what batch size* should I switch from weight-stationary to weight-gathered layouts? *How* does the answer change if I'm latency-constrained versus throughput-constrained?

The paper's contribution is to derive these answers analytically (in Section 3 and Appendix A) and validate them empirically (in Section 4), producing **decision rules** rather than just benchmark numbers. For example:

- 2D weight-stationary becomes more communication-efficient than 1D when $\sqrt{n_{\text{chips}}} > d_{ff}/d_{\text{model}}$—which, for typical models with $d_{ff} = 4 \times d_{\text{model}}$, occurs when $n_{\text{chips}} > 16$ (Section 3.2.2).

- Weight-gathered layouts become cheaper than weight-stationary when the batch size in tokens is sufficiently large; the optimal number of chips to all-gather weights over is $N = \sqrt{BL \cdot n_{\text{chips}}/F}$ (Section 3.2.3 and Appendix A.2.2).

- During decode, always use 2D weight-stationary because $B \times 1$ tokens per step is always small; during prefill, switch to weight-gathered when $B \times L_{\text{input}}$ exceeds a threshold (Section 4.1).

This framework also explicitly acknowledges that the "best" choice depends on application requirements. The Pareto frontier in Figure 1 doesn't identify a single optimal point—it maps the tradeoff curve so that practitioners can choose based on their latency and cost constraints. Tables 2 and 3 provide concrete "recipes" for specific scenarios (low-latency chatbot vs. high-throughput offline inference), showing exactly which partitioning layout, batch size, and weight format to use.

### The Underlying Stakes: Democratizing Access to Large Models

While the paper's tone is engineering-focused, the motivation has broader implications. Large language models are becoming central to applications ranging from chatbots (Thoppilan et al., 2022) to code generation to scientific reasoning. But their practical utility is constrained by inference costs—if serving a 540B model requires hundreds of thousands of dollars in accelerator hardware and operates at latencies that frustrate users, the technology remains inaccessible to all but the largest organizations.

The paper's approach—partitioning models to 64+ chips to hit sub-second latency targets, using int8 quantization to reduce memory pressure, exploiting multiquery attention to enable long-context generation, and selecting different strategies for different phases of inference—is fundamentally about making large models **economically deployable**. The 1.9-second end-to-end latency for processing 64 tokens of user input with a 1920-token conversation history and generating a 64-token response (Section 1) is benchmarked against the real-world requirement of interactive applications where users expect near-instantaneous responses.

The paper concludes by acknowledging limitations that point toward future work: dense Transformer models have fundamental FLOP and communication floors that no amount of partitioning can eliminate. Sparsity (mixture-of-experts, Fedus et al., 2022) and adaptive computation (Schuster et al., 2022) are flagged as directions that could reduce per-token FLOPs and shift the Pareto frontier further. But within the constraints of dense models on current hardware, the paper aims to provide the most complete picture yet of what's achievable and how to achieve it.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper develops a *partitioning and execution framework* for running large Transformer models efficiently at inference time on TPU v4 hardware. The system takes as input a model architecture specification, a set of application requirements (latency target, batch size, context length), and a hardware configuration (number of chips, interconnect topology), and produces as output a concrete parallelization strategy—how to split the model's weight matrices, activations, attention tensors, and KV cache across chips—along with scheduled communication operations that minimize end-to-end latency and maximize hardware utilization. The core idea is that **no single partitioning strategy is universally optimal**; instead, the optimal layout depends on where you are in the latency-throughput Pareto frontier and which phase of inference you're in (prefill vs. decode), so the framework provides analytical models to select the right strategy given the operating conditions.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major components that together define how computation and data are distributed across an array of TPU v4 chips:

1. **Feedforward Layer Partitioning Engine** — decides how to split the two large weight matrices in each MLP block ($E \times F$ and $F \times E$, where $E$ is the model dimension and $F$ is the feedforward dimension) across the 3D torus of chips. Offers three strategy families: 1D weight-stationary (split only along $F$), 2D weight-stationary (split along both $E$ and $F$), and weight-gathered (keep activations stationary and broadcast weights). The engine analytically selects between these based on batch size, chip count, and the $F/E$ ratio.

2. **Attention Layer Partitioning Engine** — handles the key, query, value projections and the KV cache for the attention mechanism. For multihead attention, it shards over the heads dimension; for multiquery attention, it **shards over the batch dimension** to avoid replicating the single key/value head, using an all-to-all collective to reshuffle the small per-token Q, K, V tensors while keeping only a fraction of the large KV cache on each chip.

3. **Phase-Aware Strategy Selector** — recognizes that prefill (processing all input tokens in parallel) and decode (generating tokens autoregressively one at a time) have fundamentally different computational shapes. During prefill, $B \times L_{\text{input}}$ tokens are in flight and compute dominates; during decode, only $B$ tokens (one per sequence) are processed per step and memory bandwidth loading the KV cache dominates. The selector picks different partitioning layouts for each phase—typically weight-gathered for prefill and 2D weight-stationary for decode—and can even mix batch sizes between phases (e.g., batch-1 prefill feeding into batch-64 decode).

4. **Low-Level Execution Optimizer** — applies communication-computation overlap via Looped CollectiveEinsum (scheduling all-gathers and reduce-scatters to run concurrently with matrix multiplications), tunes tensor memory layouts to minimize padding during matmuls, and supports int8 weight quantization via the AQT library to reduce memory traffic when bandwidth is the bottleneck.

Information flows as follows: the user specifies a model (size, architecture, attention type), application constraints (latency target or throughput target), and hardware budget (number of chips). The feedforward and attention engines each compute the communication cost for their candidate layouts as a function of batch size and chip count using analytical formulas derived in Appendix A. The phase-aware selector picks the layout that minimizes total time (compute + memory load + communication) for each phase. The low-level executor implements this layout with optimized collectives and fused operations, producing a compiled inference program.

### 3.3 Roadmap for the Deep Dive

- **First**, the **feedforward layer partitioning strategies** (1D weight-stationary, 2D weight-stationary, weight-gathered) — because the MLP blocks dominate the parameter count and computation in large Transformers, the choice of how to split the $E \times F$ weight matrices is the single biggest factor determining inference performance. We'll walk through each layout's communication pattern, derive its cost formula, and explain *why* the optimal choice shifts with batch size.
- **Second**, the **attention layer partitioning** — building on the feedforward foundation, we examine the additional challenge of the KV cache, which is unique per sequence and grows linearly with context length. We'll see why multiquery attention's memory savings can only be realized with batch-dimension sharding, and why the all-to-all collective used to enable this is profitable despite its communication cost.
- **Third**, the **parallel vs. serial Transformer block** — a smaller but practically important design choice that halves the communication in each layer by fusing attention and feedforward projections.
- **Fourth**, the **low-level execution optimizations** — how Looped CollectiveEinsum hides communication behind computation, the specific async APIs used, and the int8 quantization scheme for weight memory reduction.
- **Fifth**, the **analytical communication cost model** — the mathematical framework (detailed in Appendix A) that allows the system to *predict* which layout is optimal without exhaustive search. This is the intellectual core that transforms the paper from a collection of benchmarks into a principled engineering methodology.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems engineering paper** whose core idea is that the optimal multi-dimensional partitioning of a Transformer model for inference is predictable from simple analytical models of communication cost, and that different phases of inference (prefill vs. decode) require different partitioning strategies because their ratio of computation to memory access is fundamentally different.

---

#### The Hardware Model: What We're Optimizing Over

Before diving into partitioning strategies, we need to understand the hardware substrate that all these layouts are mapped onto. The paper targets **Google TPU v4 chips** configured in a **3D torus topology** with dimensions $X \times Y \times Z$, where each chip can communicate with its immediate neighbors along each axis. A slice of $n_{\text{chips}} = X \times Y \times Z$ chips is treated as a single logical machine.

**Key hardware parameters** (Section 4, Methodology):
- **Compute:** 275 TFLOPS per chip in bfloat16 matrix arithmetic
- **Memory capacity:** 32 GiB of High Bandwidth Memory (HBM) per chip
- **Memory bandwidth:** 1,200 GB/s per chip (the rate at which data can be streamed from HBM into the compute cores)
- **Interconnect bandwidth:** 270 GB/s per chip (the rate at which data can be sent between chips over the torus links)

The distinction between memory bandwidth (within a chip, HBM to compute cores) and interconnect bandwidth (between chips) is crucial. Loading a tensor from HBM is roughly 4.4× faster than sending it to another chip (1,200 vs. 270 GB/s), which means that while partitioning reduces per-chip memory pressure, the communication it introduces is relatively *more expensive per byte* than local memory access. This is why the analytical models in Section 3 and Appendix A focus on minimizing communication volume—it's the most precious resource in the system.

**The basic cost model** for an operation has three additive components:

1. **Compute time:** $T_{\text{comp}} = \frac{\text{total FLOPs}}{n_{\text{chips}} \times \text{peak FLOPS per chip}}$
   For an $N$-parameter decoder-only model processing one token, the total matmul FLOPs are $2N$ (one multiply and one add per parameter-token pair), and these are divided across $n_{\text{chips}}$ chips.

2. **Memory time:** $T_{\text{mem}} = \frac{\text{total bytes loaded from HBM}}{n_{\text{chips}} \times \text{HBM bandwidth per chip}}$
   The two dominant contributors are weight loading (model parameters) and KV cache loading (attention state). At small batch sizes, weight loading dominates; at large batch sizes with long contexts, KV cache loading dominates (Section 2, "Memory costs").

3. **Communication time:** $T_{\text{comm}} = \frac{\text{total bytes transferred over interconnect}}{\text{effective network bandwidth}}$
   The effective network bandwidth depends on the specific collective operation (all-gather, reduce-scatter, all-to-all) and the fraction of each chip's 270 GB/s interconnect that the operation can use. The analytical formulas in Appendix A approximate this by assuming the full interconnect bandwidth is available for each collective, and ignoring the $(K-1)/K$ factor for simplicity (valid when the number of partitions $K$ is large).

The total time per forward pass is not simply the sum of these three: the paper's low-level optimizations (Section 3.5) overlap communication with computation, hiding some or all of the communication time behind computation. But the analytical models treat them as additive to get a clean optimization target.

**Notation for tensor sharding** (Section 3.1): The paper uses a subscript notation to indicate which tensor dimensions are partitioned over which hardware axes. For example, `$\text{BLExyz}$` means a tensor of logical shape `[B, L, E]` is split such that the $E$ dimension is partitioned into $X \times Y \times Z$ pieces, with each chip holding a slice of shape `[B, L, E/(X \times Y \times Z)]`. If an axis is omitted from the subscript, the tensor is replicated (not partitioned) over that axis. A suffix `partialsum-x` indicates that a tensor has been locally summed on each chip over some dimension, but still needs to be reduced across chips in the $x$ axis before the result is valid.

---

#### Communication Primitives: What Operations Move Data Between Chips

The partitioning strategies are built from three collective communication operations (Section 3.1, Figure A.1), each with a well-defined cost model:

**All-gather (over axis $a$):** Starts with a tensor sharded over axis $a$, where each chip holds a slice of size $D/K$ (where $K$ is the number of partitions along that axis). Each chip sends its slice to all other $K-1$ chips in the group, resulting in every chip holding a replicated tensor of full size $D$. The communication time is:

$$T_{\text{all-gather}} = \frac{D}{\text{bandwidth}} \cdot \frac{K-1}{K}$$

where $D$ is the size of the output (replicated) tensor on each chip. The $(K-1)/K$ factor accounts for the fact that each chip only needs to receive $K-1$ chunks from other chips (it already has its own); in practice the paper approximates this as 1 when $K$ is large. **What it computes:** This is the time to replicate a tensor across $K$ chips so that every chip has a complete copy. **Why this form:** The total data movement is proportional to the output size per chip times the number of recipients, and inversely proportional to the interconnect bandwidth. The all-gather is used in weight-gathered layouts to broadcast weight shards to all chips that need them, and in weight-stationary layouts to replicate activation shards for the next matrix multiplication.

**Reduce-scatter (over axis $a$):** The inverse of all-gather. Starts with a full-sized tensor $D$ on each chip (typically the output of a matrix multiplication where each chip contributed a partial result). Each chip sums its contribution with the corresponding contributions from other chips (reduction) and then scatters the result so that each chip ends up with only a shard of size $D/K$ of the final summed result. The communication time is:

$$T_{\text{reduce-scatter}} = \frac{D}{\text{bandwidth}} \cdot \frac{K-1}{K}$$

where $D$ is the size of the *input* (pre-reduction) tensor on each chip. **What it computes:** The time to sum partial results across chips and distribute the sharded output. **Why this form:** The reduction phase requires each chip to send its chunk to the designated recipient for that chunk, so the data movement is proportional to the per-chip input size. Reduce-scatter is used after matrix multiplications in weight-stationary layouts to aggregate partial sums into sharded activations ready for the next operation.

**All-reduce:** A compute followed by a broadcast. Typically implemented as a reduce-scatter followed by an all-gather (as recommended by Rajbhandari et al., 2020, cited in Section 3.1), because this halves the total communication compared to a naive all-to-all-then-sum approach. The total time is:

$$T_{\text{all-reduce}} = 2 \cdot T_{\text{all-gather}}$$

**All-to-all:** Reshuffles sharding from one tensor dimension to another. For example, a tensor sharded over heads (each chip holds a subset of heads for all batch elements) can be reshuffled to be sharded over batch (each chip holds all heads for a subset of batch elements). Every chip communicates directly with every other chip, sending and receiving data. Unlike all-gather and reduce-scatter where the cost formula is simple, all-to-all cost depends on the specific dimensions being transposed. The paper uses all-to-all as part of the multiquery attention batch-sharded layout (Figure 5b) to transpose the small Q, K, V tensors from a heads-based sharding to a batch-based sharding.

**Looped CollectiveEinsum** (Section 3.5, citing Wang et al., 2023): This is not a new collective but a technique for overlapping collectives with computation. Rather than executing an all-gather first, then running the matrix multiply, Looped CollectiveEinsum interleaves the communication and computation: as each chunk of the all-gathered tensor arrives, it immediately participates in the ongoing matrix multiplication (which is partitioned into tiles). Under ideal conditions, the communication time is entirely hidden behind computation time—the relevant formula becomes $T_{\text{total}} = \max(T_{\text{comp}}, T_{\text{comm}})$ rather than $T_{\text{comp}} + T_{\text{comm}}$. The paper notes (Section 3.5) that their optimized implementation achieves "about 1.4 times better performance than the simpler compiler-partitioned-and-scheduled implementation" using this technique.

The paper emphasizes (Section 3.1) that they always choose to reduce-scatter into the hidden dimension ($E$ or $F$) rather than the batch or sequence dimension ($B$ or $L$), even though the latter would avoid communication in layernorm (as Korthikanti et al., 2022 chose). The reason: reducing into the hidden dimension "exposes more effective opportunities for Looped CollectiveEinsum"—the communication can be better overlapped with the subsequent matrix multiply when the reduced tensor's sharding matches the weight matrix's sharding.

---

#### Feedforward Layer Partitioning Strategy 1: 1D Weight-Stationary

**What it is.** This is the simplest possible partitioning strategy and serves as the baseline that the more advanced strategies improve upon. The idea is taken directly from Megatron-LM (Shoeybi et al., 2019). In each transformer layer, the MLP block consists of two consecutive matrix multiplications: an "up-projection" $W_{\text{in}}$ of shape $E \times F$ (where $F = d_{ff}$ is typically $4 \times E$) followed by a GELU nonlinearity and a "down-projection" $W_{\text{out}}$ of shape $F \times E$. The 1D weight-stationary layout partitions *only* the $F$ dimension across all $n_{\text{chips}}$ chips.

**What happens, step by step** (Figure 2a):

1. The input activation arrives at each chip with shape `BLE` (full batch, full sequence, full model dimension). Every chip has a complete copy of the input—it is replicated over all chips.

2. Each chip multiplies its input `BLE` by its shard of $W_{\text{in}}$ of shape `EFz` (where the $F$ dimension is split $n_{\text{chips}}$-ways, one slice per chip, and $E$ is full). This produces a partial output of shape `BLFz`—each chip holds a different slice of the $F$ dimension for the entire batch and sequence.

3. The GELU activation is applied independently on each chip to its local `BLFz` slice. No communication is needed here since GELU is element-wise.

4. Each chip multiplies its GELU output `BLFz` by its shard of $W_{\text{out}}$ of shape `FzE`. This produces a tensor of shape `BLE` on each chip, but critically, these are *partial sums* over the $F$ dimension—each chip computed the contribution from its subset of $F$ neurons, and the full result requires summing across all chips.

5. A **reduce-scatter** over all $n_{\text{chips}}$ chips sums the partial `BLE` tensors and distributes the result sharded over $E$ (or equivalently, the all-reduce could produce a replicated `BLE`; the paper uses reduce-scatter to match the next operation's sharding).

**Communication cost derivation** (Section 3.2.1):

The input activation `BLE` must be available on every chip for step 2. In the 1D layout, this happens via an all-gather before the first matmul (so each chip gets the full `BLE`), and the output is aggregated via a reduce-scatter after the second matmul. The communication volume per chip for each operation is `BLE` (all-gather of output size `BLE`, reduce-scatter of input size `BLE`). The total communication time is:

$$T_{\text{comm}} = \frac{2 \cdot BLE}{\text{network bandwidth}}$$

where $B$ is the batch size, $L$ is the sequence length (1 during decode, $L_{\text{input}}$ during prefill), and $E$ is the model dimension. The factor of 2 accounts for one all-gather and one reduce-scatter.

**Why this scales poorly.** The communication volume is **independent of $n_{\text{chips}}$**—whether you use 8 chips or 64 chips, you still have to move the full `BLE` tensor across the interconnect for every MLP block. As chip count grows, the compute time and memory time per chip decrease (linearly with $n_{\text{chips}}$), but the communication time stays flat. Eventually, communication becomes the bottleneck, and adding more chips provides no further latency reduction. This is the "communication bottleneck" that limits scalability.

**The Megatron trick for zero intermediate communication.** There's a subtle but critical optimization in this layout (Section 3.2.1): because $W_{\text{in}}$ is partitioned along its output axis ($F$) and $W_{\text{out}}$ is partitioned along its input axis (also $F$), the output shard of the first matmul on each chip is *exactly the input shard needed* for the second matmul on the same chip. The intermediate activation (after GELU, between the two matmuls) never needs to be communicated—it stays on the chip that produced it. If instead $W_{\text{in}}$ were partitioned along $E$ (input axis) and $W_{\text{out}}$ along $E$ (output axis), you'd need an all-reduce *between* the two matmuls to reshuffle the intermediate tensor, doubling the communication. This is why the choice to partition along $F$ (the "column-wise" dimension) for both matrices is not arbitrary—it's what enables the $n_{\text{chips}} > 16$ threshold we'll see next.

---

#### Feedforward Layer Partitioning Strategy 2: 2D Weight-Stationary

**What it is and why it's needed.** The 1D layout's communication doesn't decrease with chip count because it always moves a full `BLE`-sized tensor across the network. The 2D weight-stationary layout fixes this by partitioning the weight matrices along *both* the $E$ and $F$ dimensions (Section 3.2.2). Each $E \times F$ weight matrix is split into a grid of $X$ partitions along $E$ and $Y \times Z$ partitions along $F$, where $X \times Y \times Z = n_{\text{chips}}$. Each chip stores a roughly square chunk of the weight matrix of size $(E/X) \times (F/(YZ))$.

**Why "2D" helps—the intuition.** In the 1D layout, every chip multiplies its $F$-shard by the *full* activation vector of size $E$. In the 2D layout, each chip multiplies its $(E/X) \times (F/(YZ))$ weight shard by only an $E/X$-sized slice of the activation vector. After the matmul, chips need to aggregate results—but they only need to aggregate over the chips that hold *different* $E$-shards (the $X$ dimension), not over all $n_{\text{chips}}$ chips. The communication volume for this aggregation is $BLE/X$ rather than $BLE$, which decreases as $X$ (and hence $n_{\text{chips}}$) grows.

**How the weights are laid out** (Figures 2b and text: "The partitioning layout for weights is ExFyz"). $W_{\text{in}}$ has shape `ExFyz`—partitioned $X$-ways along $E$ and $(Y \times Z)$-ways along $F$. $W_{\text{out}}$ has shape `FyzEx`—partitioned $(Y \times Z)$-ways along $F$ and $X$-ways along $E$. Notice that the sharding is transposed between the two matrices (input of first = output of second in the $E$/$F$ orientation), matching the "trick" from the 1D case: the output of the first matmul is already sharded correctly for the input of the second matmul.

**Step-by-step execution** (Figure 2b, detailed in Appendix A.2.1):

1. The input activation arrives sharded along $E$ over the $X$ axis, shape `BLEx` on each chip. This means the $X$ axis splits the $E$ dimension, and the $Y$ and $Z$ axes replicate the tensor (each chip on the same $X$ coordinate but different $Y$/$Z$ has an identical copy).

2. Each chip multiplies its `BLEx` activation by its `ExFyz` weight shard. The local result has shape `BLFyz`—each chip on the $F$-partitioned axes ($Y$ and $Z$) has a different chunk of the $F$ dimension, while chips on the $X$ axis (same $Y$/$Z$ but different $X$) hold *partial sums* that need to be combined.

3. A **reduce-scatter over the $Y$ and $Z$ axes** sums the partial results along those axes and produces an output sharded along $E$ (via reduction), shape `BLEx(partialsum-yz)`. Wait—this step needs careful unpacking. The initial activation was `BLEx`. The matmul produces a tensor whose $F$ dimension is split over $Y \times Z$ and whose $E$ dimension has been contracted (producing partial sums along the $X$ axis). The reduce-scatter over $Y$ and $Z$ takes the partial sums from chips that had different $E$-shards but the same $F$-shard and sums them, producing a result still sharded along $E$.

4. GELU is applied element-wise, no communication.

5. Each chip multiplies its GELU output (`BLFyz` after appropriate reshuffling to match $W_{\text{out}}$'s `FyzEx` layout) by $W_{\text{out}}$. The local result is `BLEx(partialsum-yz)`—partial summed along the $Y$/$Z$ axes (the $F$ dimension).

6. A **reduce-scatter over the $X$ axis** sums these partial results and produces the final output sharded over $E$ or replicated, depending on the next layer's requirements.

**Communication cost derivation** (Appendix A.2.1):

The total communication involves two reduce-scatters (or equivalently, one all-gather and one reduce-scatter, as shown in the figure). The first reduce-scatter operates on tensors of per-chip size `BLE/X` (the partial sum output of the first matmul), and the second on tensors of size `BLF/(YZ)` (the partial sum output of the second matmul). The total communication time is:

$$T_{\text{comm}} = \frac{2BL}{\text{network bandwidth}} \cdot \left( \frac{E}{X} + \frac{F}{YZ} \right)$$

where $E/X$ is the per-chip $E$-dimension size and $F/(YZ)$ is the per-chip $F$-dimension size. The factor of 2 accounts for the two reduce-scatters.

**Optimizing the torus axes** (Appendix A.2.1): We have a free choice of how to split $n_{\text{chips}}$ into $X$ (partitions along $E$) and $Y \times Z$ (partitions along $F$), subject to the constraint $X \times Y \times Z = n_{\text{chips}}$. To minimize total communication, we set the derivative of the communication expression to zero, which yields:

$$X = \sqrt{\frac{E}{F} \cdot n_{\text{chips}}}$$

Assuming the standard transformer ratio $F = 4E$ (feedforward dimension is 4× the model dimension, as in PaLM):

$$X = \sqrt{\frac{n_{\text{chips}}}{4}} = 0.5 \times \sqrt{n_{\text{chips}}}$$
$$YZ = \sqrt{4 \cdot n_{\text{chips}}} = 2 \times \sqrt{n_{\text{chips}}}$$

Substituting back, the minimum communication time is:

$$T_{\text{comm}} = \frac{8 \cdot BLE}{\sqrt{n_{\text{chips}}} \times \text{network bandwidth}}$$

**What this formula tells us.** The communication time now scales as $O(1/\sqrt{n_{\text{chips}}})$ instead of $O(1)$ in the 1D layout. The factor 8 reflects the product of constants: 2 (for two reduce-scatters), times the sum of the weighted $E/X$ and $F/(YZ)$ terms evaluated at the optimum. If $n_{\text{chips}} = 64$, then $\sqrt{n_{\text{chips}}} = 8$, so communication is roughly $8 \times$ less per chip than a hypothetical 1-chip system (though the absolute communication volume still grows with total chip count, just sublinearly).

**When 2D beats 1D.** The paper gives a clean criterion (Section 3.2.2): 2D weight-stationary is more communication-efficient than 1D when:

$$\sqrt{n_{\text{chips}}} > \frac{F}{E}$$

For the typical case $F = 4E$, this means $n_{\text{chips}} > 16$. Below 16 chips, the 1D layout's communication overhead is still manageable and its simplicity (fewer collective operations) may make it preferable. At exactly 16 chips, they're roughly tied. Beyond 16, 2D becomes increasingly advantageous. This is validated in Figure 6, which shows 2D weight-stationary outperforming 1D weight-stationary on 64 chips ("Both layouts start to become communication-limited, but the 2D layout performs better because of its asymptotically better scaling with chip count").

**Why this layout is used for decoding.** During decode, the batch size in tokens is $B \times 1$ (one new token per sequence). This is *always small*—even at batch size 512, you're processing only 512 tokens per forward pass. The 2D weight-stationary layout is designed for exactly this regime: the activation tensors being communicated are relatively small ($BLE/X$ in size), and the communication overhead is manageable. The weight matrices stay stationary on their chips, which is efficient because they're reused across many decode steps (one per generated token).

---

#### Feedforward Layer Partitioning Strategy 3: Weight-Gathered Layouts

**The premise: when activations are larger than weights, reverse the stationary assignment.** All previous layouts keep weights "stationary" on each chip and transfer activations between chips. This makes sense when the activation size ($BLE$) is smaller than the weight size ($EF$) per forward pass. But during prefill with large batch sizes and long sequences, $BLE$ can exceed $EF$ significantly—at batch 512, sequence length 2048, and $E = 18432$, the activation tensor is $512 \times 2048 \times 18432 \approx 19$ GB, while the weight matrix $E \times F$ with $F = 4E$ is roughly $18432 \times 73728 \approx 1.4$ GB in bfloat16.

**The inversion.** In weight-gathered layouts (Section 3.2.3, Figure 2c and Figure A.2), the activations stay stationary on each chip, and the *weights* are transferred between chips via all-gather. This is more efficient because now the communication cost is proportional to the weight size ($EF$) rather than the activation size ($BLE$), and weights are shared across all sequences in the batch—they only need to be broadcast once per forward pass regardless of batch size.

**Three variants with increasing weight-sharing** (Section 3.2.3, Figure A.2):

The paper defines three weight-gathered layouts, distinguished by how many torus axes the weights are all-gathered over:

- **X-weight-gathered:** Weights are all-gathered over the $X$ axis only. Activations are sharded as `BxLEyz`—the batch dimension is split $X$-ways, and $E$ is split over $Y \times Z$. Communication volume on weights is $2EF/X$, and communication volume on activations is $BLE/X$ (one reduce-scatter/all-gather pair saved compared to 2D weight-stationary).

- **XY-weight-gathered:** Weights are all-gathered over both $X$ and $Y$ axes. Activations are sharded as `BxyLEz`—batch split over $X \times Y$, $E$ split over $Z$. More weight communication ($2EF/(Z)$) but less activation communication ($BLE/(XY)$).

- **XYZ-weight-gathered:** Weights are all-gathered over *all three* axes. Activations are sharded as `BxyzLE`—batch fully split, model dimension $E$ fully replicated. Maximum weight communication ($2EF$) but no activation communication at all for the feedforward layer.

**The core design constraint: weight layout compatibility.** A crucial practical detail (Section 3.2.3): "weights start in the same ExFyz layout as in 2D weight-stationary, so that we can use the same weight layout for weight-gathered (during prefill) and weight-stationary (during decoding)." This means the weight tensors are physically stored on chips in their 2D weight-stationary sharding (`ExFyz`), and the weight-gathered layout is implemented by *temporarily* all-gathering them before the matmul. After the matmul, the gathered weights are discarded from memory (they were just copies), and the stationary shards remain. This design enables the system to switch between prefill and decode layouts without rewriting model parameters in memory—you just change the communication pattern.

**Communication cost minimization** (Appendix A.2.2): Let $N$ be the number of chips that weights are all-gathered over ($N = X$, $XY$, or $XYZ$ depending on the variant). The weight communication time is:

$$T_{\text{comm}}^{\text{weights}} = \frac{2 \cdot EF \cdot N}{n_{\text{chips}} \times \text{network bandwidth}}$$

The factor $N/n_{\text{chips}}$ accounts for the fraction of total chips participating in each all-gather group: each chip in the group receives the full weight, so the per-chip data received is $EF$, and there are $n_{\text{chips}}/N$ such groups operating in parallel.

The activation communication time (for the remaining reduce-scatter that 2D weight-stationary would have required) is:

$$T_{\text{comm}}^{\text{acts}} = \frac{2 \cdot BLE}{N \times \text{network bandwidth}}$$

because activations are now split $N$-ways over the chips that weights are gathered on.

Total communication $T_{\text{comm}} = T_{\text{comm}}^{\text{weights}} + T_{\text{comm}}^{\text{acts}}$ is minimized by setting:

$$N = \sqrt{\frac{BL \cdot n_{\text{chips}}}{F}}$$

where $BL$ is the total number of tokens in the batch (batch size × sequence length).

Substituting this optimal $N$ back:

$$T_{\text{comm}} = \frac{4E \cdot \sqrt{BLF}}{\sqrt{n_{\text{chips}}} \times \text{network bandwidth}}$$

**What this formula reveals.** The communication time for weight-gathered layouts grows as $O(\sqrt{BL})$ instead of $O(BL)$ for weight-stationary layouts. When $BL$ is small (during decode, $BL = B \times 1$), the $\sqrt{BL}$ term makes weight-gathered communication larger than weight-stationary. But when $BL$ is large (during prefill with big batches and long sequences), the $\sqrt{BL}$ scaling is dramatically better than $BL$ scaling. Figure 3 demonstrates this crossover visually: at 2,000 tokens per batch, weight-stationary has the lowest communication volume; by 200,000 tokens per batch, XYZ-weight-gathered has roughly 0.3× the communication volume of weight-stationary.

**Why there are three variants and not a continuum.** Ideally, you'd set $N$ to exactly $\sqrt{BL n_{\text{chips}} / F}$ for every batch size. But in practice, $N$ is constrained to be a valid subset of the 3D torus axes ($X$, $X \times Y$, or $X \times Y \times Z$), where $X$, $Y$, $Z$ are fixed by the 2D weight-stationary layout (to maintain weight layout compatibility). The three variants correspond to the three discrete choices of which axes to all-gather over, and the system selects whichever minimizes total communication for the current $BL$. Figure 7 demonstrates this switching behavior empirically: at $10^5$ tokens per batch, 2D weight-stationary achieves ~30% MFU; at $3 \times 10^5$ tokens, weight-gathered achieves ~76% MFU as communication overhead becomes nearly negligible.

**A subtlety: why not all-gather activations instead of weights?** The paper's choice to keep activations stationary and move weights is driven by the asymmetry that weights are shared across all tokens in the batch, while activations are unique per token. For a batch of $B$ sequences of length $L$, all-gathering weights costs $O(EF)$ regardless of $BL$, but all-gathering activations would cost $O(BLE)$ which grows with $BL$. When $BL > EF/E = F$ (roughly 73,728 for PaLM 540B), weight-gathering is cheaper—and during prefill with $B=512$, $L=2048$, $BL \approx 10^6$, this is decisively the case.

---

#### Attention Layer Partitioning

**The unique challenge of attention.** Unlike the feedforward layer where only the weight matrices and current activations are in play, the attention layer must also contend with the **KV cache**—stored key and value tensors from all previous token positions that must be loaded from HBM at every decode step. The KV cache size per layer is:

$$2 \times B \times L_{\text{cache}} \times d_{\text{model}}$$

in bfloat16 bytes (2 bytes per element), where $d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}}$ for multihead attention and $d_{\text{model}} = d_{\text{head}}$ for multiquery attention (since key/value are per-head, not per-query-head). The factor of 2 accounts for both keys and values.

For multihead attention on PaLM 540B (with hypothetical multihead, 48 heads × 256 head-dim = 12288, but the paper pads to 64 heads with 256 head-dim = 16384 for partitioning purposes as noted in Section 4), at batch size 512 and context length 2048, the KV cache across all 118 layers is approximately:

$$2 \times 118 \times 512 \times 2048 \times 48 \times 256 \text{ elements} \times 2 \text{ bytes} \approx 3 \text{ TB}$$

This is three times the model parameter size (540B parameters × 2 bytes ≈ 1.08 TB). Loading this cache from HBM at every decode step is the dominant cost for long-context inference.

**Multiquery attention reduces cache size, but creates a partitioning dilemma.** Multiquery attention (Shazeer, 2019) uses $n_{\text{heads}}$ query heads but only a single key and value head shared across all queries. This reduces the KV cache size by a factor of $n_{\text{heads}}$ (48× for PaLM 540B), bringing it from 3TB to ~62.5GB for the same batch and context settings. This is unequivocally good for memory capacity.

However, the paper identifies a problem (Section 3.3): "it also removes an axis otherwise used for parallelism." In multihead attention, the $n_{\text{heads}}$ dimension provides a natural axis for tensor parallelism—you can shard the Q, K, V projection weights across heads, and each chip handles a subset of heads with its own fraction of the KV cache (Figure 4a). The per-chip KV cache size is $2 \times B \times L_{\text{cache}} \times d_{\text{head}} \times (n_{\text{heads}} / n_{\text{chips}})$—each chip stores only $1/n_{\text{chips}}$ of the total cache.

In multiquery attention, there is only one key head and one value head. If you use the "sharded over heads" layout from multihead attention (Figure 4b), you'd need to **replicate** the single key and value to every chip, because every query head needs to attend to the same keys and values. This completely nullifies the memory savings: the per-chip KV cache would be the full $2 \times B \times L_{\text{cache}} \times d_{\text{head}}$, same as if multiquery attention were never used.

**The solution: shard attention over batch instead of heads** (Figures 4c and 5b). The key insight is that during autoregressive generation, the Q, K, V tensors for the *current* token are tiny—one token per sequence, so shape `[B, 1, d]` for each. The KV cache, however, contains thousands of tokens. So the paper proposes:

1. **Keep the KV cache sharded over batch**, with each chip storing keys and values for only $B/n_{\text{chips}}$ sequences. The per-chip cache size becomes $2 \times (B/n_{\text{chips}}) \times L_{\text{cache}} \times d_{\text{head}}$, recovering the $n_{\text{chips}}$-fold reduction.

2. **Use an all-to-all collective to reshuffle the Q, K, V tensors** before the attention computation. Initially, Q, K, V are produced by projection matrices that are partitioned over heads (to match the feedforward layer's weight-stationary partitioning)—so each chip holds a subset of heads for all batch elements. The all-to-all transposes this: after the collective, each chip holds all heads for a subset of batch elements. This matches the KV cache sharding.

3. **Compute attention locally.** Each chip now has the Q (all heads) for its batch subset, and the K, V cache for exactly that batch subset. The dot-product attention $QK^T$ can proceed without further communication.

4. **Another all-to-all reshuffles the output** back to the heads-sharded layout for the output projection $W_O$.

**Why this is profitable despite the all-to-all cost.** During decode, the Q, K, V tensors for the current step are tiny: `[B, 1, n_heads × d_head]` for Q, and `[B, 1, d_head]` for K and V in multiquery attention. With $B = 512$, $n_{\text{heads}} = 64$, $d_{\text{head}} = 256$, the Q tensor is $512 \times 64 \times 256$ ≈ 8 million elements ≈ 16 MB. The KV cache being loaded, by contrast, is `[B/n_chips, L_cache, d_head]` in size per chip per layer. With $L_{\text{cache}} = 2048$, this is $(512/64) \times 2048 \times 256$ ≈ 4 million elements ≈ 8 MB per chip per layer (for the key alone, times 2 for key+value). Across all 118 layers, the total KV cache loaded per decode step is $118 \times 2 \times 8$ MB ≈ 1.9 GB.

The all-to-all cost is proportional to the Q, K, V tensors (16 MB), while the savings is proportional to eliminating the KV cache replication (which would be ~60 GB per step if each chip stored the full cache). Since the KV cache load is 3–4 orders of magnitude larger than the all-to-all transfer, the trade is overwhelmingly favorable:

> "Since the KV cache is orders of magnitude larger than the Q, K, and V tensors, it is very profitable to spend the all-to-all communication time on the small tensors to save the memory time on the large tensors."

**Prefill uses a different attention layout.** During prefill, the Q tensor has $L_{\text{input}}$ tokens (e.g., 2048), not just 1. The memory load of the K and V tensors is now amortized over all $L_{\text{input}}$ query tokens, since the same K, V cache is reused for every query position. The paper states (Section 3.3): "The memory load of the K and V tensors is amortized over all tokens in the Q tensor, and so this memory load is typically not a bottleneck during prefill. Therefore for prefill we use the sharded-over-heads layout." In other words, during prefill, you can afford to replicate the K, V caches (or compute them on the fly from the input tokens, since you're processing all positions simultaneously) because the large Q × K matmul dominates the runtime, not the cache loading.

**Empirical validation of the memory savings** (Table 1). The paper compares maximum context lengths for three attention variants on 64 chips, reserving 30% of total memory for the KV cache:

| Model variant | d_head | Max context length (batch=128) | Max context length (batch=512) |
|---|---|---|---|
| Multihead | 128 | 1,320 | 330 |
| Baseline multiquery | 256 | 660 | 165 |
| Optimized multiquery | 256 | 43,000 | 10,700 |

The optimized multiquery layout (batch-sharded) supports **32–64× longer contexts** than either multihead attention or the baseline multiquery layout. The baseline multiquery (sharded over heads, replicating the single K/V) actually performs *worse* than multihead in terms of max context length (660 vs. 1320 at batch 128), because even though there's only one K/V head, it's replicated to all 64 chips, consuming $64 \times$ the memory per layer compared to the optimized layout. This is a stark demonstration that the architectural advantage of multiquery attention only materializes with the right partitioning.

**Latency scaling with context length** (Figure 8). The paper benchmarks an 8-layer version of PaLM 540B (to isolate per-layer effects) and shows that with the optimized layout, multiquery attention scales gracefully: at sequence length 512, attention is ~8% of total runtime; at sequence length 8192, attention is ~31% of total runtime. For the full 118-layer model, context lengths beyond 512 would exhaust memory under multihead or baseline multiquery (shown by the dotted line in Figure 8). The optimized layout enables scaling to 8192–32,768 tokens while keeping attention's runtime share manageable.

---

#### Parallel Attention/Feedforward Layers

**The standard serialized Transformer block** computes attention and feedforward sequentially:

$$\text{output} = \text{FFN}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x)) + x$$

with two separate LayerNorm operations, two separate input projections, and two separate output projections. This requires two all-reduce operations (or equivalent collectives) per layer: one to aggregate the attention output, one to aggregate the FFN output.

**The parallel formulation** (Wang and Komatsuzaki, 2021; used in PaLM per Chowdhery et al., 2022) fuses these:

$$\text{output} = \text{FFN}(\text{LayerNorm}(x)) + \text{Attention}(\text{LayerNorm}(x)) + x$$

The key difference is that the feedforward and attention layers are computed *in parallel* from the same LayerNormed input, and their outputs are summed. The paper identifies three benefits (Section 3.4):

1. **One LayerNorm instead of two:** This reduces latency at small batch sizes where LayerNorm (a relatively cheap but non-matmul operation) can be noticeable.

2. **Fusion of projection matrices:** The input matrix of the feedforward layer ($W_{\text{in}}$) can be concatenated with the query projection matrix $W_Q$ of the attention layer into a single larger matrix multiply. Similarly, the key and value projections $W_K$ and $W_V$ can be fused, and the output projection $W_O$ can be fused with the feedforward output matrix $W_{\text{out}}$. Larger matmuls run more efficiently on accelerators (better utilization of the systolic array), so this fusion improves MFU.

3. **Halved communication for $d_{ff}$/$n_{\text{heads}}$ parallelism:** This is the most important benefit. In the serial formulation, you need one all-reduce for the attention output and one for the FFN output—two communication operations per layer. In the parallel formulation, since both branches are computed independently from the same input and their outputs are summed, the two all-reduces can be fused into a single all-reduce. "It also eliminates one of the two all-reduce operations in each Transformer layer needed for $d_{ff}$/$n_{\text{heads}}$ parallelism, cutting communication time over this axis in half."

**Empirical impact** (Section 4.3). Comparing a serial PaLM 540B variant against the parallel version during generation (2D weight-stationary, 64 chips, batch 512): "The serial formulation incurs 14% higher inference latency per step than the parallel version because of the increased communication time for activations." During prefill, the gap shrinks because weight-gathered layouts already have less activation communication to eliminate.

The 14% improvement is substantial for a purely architectural choice—it requires no additional hardware, no change in model quality (since the parallel formulation was used during PaLM's training), and applies uniformly across all inference scenarios. It's a "free lunch" that makes the case for preferring the parallel formulation in models designed for deployment.

---

#### Low-Level Execution Optimizations

The partitioning strategies above determine *which* communication happens and *what shape* the tensors have. The low-level optimizations (Section 3.5) determine *how efficiently* those operations execute on real hardware.

**Looped CollectiveEinsum** (citing Wang et al., 2023): The core technique for overlapping communication with computation. A standard einsum (matrix multiplication) is tiled into smaller chunks that fit in the compute cores' local memory. Normally, you'd wait for the full all-gather to complete, then start the einsum. Looped CollectiveEinsum instead interleaves: as each tile of the gathered tensor arrives from the network, it's immediately fed into the ongoing matrix multiplication. The einsum loop iterates over tiles, processing some while the network delivers others.

The paper describes their implementation strategy: "we explicitly match up communication collectives with the matrix multiplies that they should be fused with, to maximize the potential for overlap." This requires careful scheduling because not all collectives can be overlapped with all matmuls—the tensor layout must be compatible. Their choice to reduce-scatter into the hidden dimension ($E$ or $F$) rather than the batch/sequence dimension ($B$ or $L$) is driven by this: it "exposes more effective opportunities for Looped CollectiveEinsum, whereas Korthikanti et al. (2022) chose the former, to avoid communication in layernorm."

The paper quantifies the benefit: "Through such optimizations, we achieved about 1.4 times better performance than the simpler compiler-partitioned-and-scheduled implementation that we started with." They also note that "some of the weight-gathered layouts would exhaust memory without these optimizations"—the overlapping reduces peak memory pressure by not needing to store the full gathered tensor before starting computation.

**Async CollectivePermute APIs** (Wang et al., 2023): The paper uses these low-level APIs to build multiple variants of CollectiveEinsum optimized for different scenarios: "latency versus throughput, different numbers of torus axes, fusing with different input/output collectives." The use of async primitives means that communication can be initiated and then the program can proceed to other work (like computation) without blocking on completion, enabling the overlap.

**Tensor memory layout optimization:** Padding and copying during matrix multiplies are minimized by "better in-memory layout of tensors." On TPUs, matmuls require tensors to be in a specific layout (e.g., a 128×128 tiling for the systolic array). If the natural sharding doesn't align, padding is inserted, wasting memory and compute. By choosing sharding dimensions that are multiples of the hardware tile size, this overhead is eliminated.

**Faster top-k/top-p for decode sampling:** During generation, after computing logits, the system must sample from the token distribution (possibly with top-k or nucleus sampling). The paper implements faster versions of these operations, which become noticeable at low batch sizes where sampling latency can be a non-trivial fraction of the per-step time.

**Incremental prefill processing (FasterTransformer):** "Support for incremental processing of sequences during prefill (FasterTransformer)." This refers to the ability to process input tokens in chunks rather than requiring all $L_{\text{input}}$ tokens to be available at once, which is important for streaming applications.

**Weight quantization via AQT** (Section 3.6, citing Lew et al., 2022): The paper uses int8 weight quantization to reduce the memory cost of 16-bit weight loading. The AQT library "converts [16-bit weights] to int8 without noticeable quality loss." This halves the weight memory traffic during decode—critical because at small batch sizes, weight loading from HBM dominates the memory time. The paper explicitly notes that they have *not* implemented activation quantization (Abdolrashidi et al., 2021), but "are hopeful that it could reduce compute time in large-batch configurations and reduce communication volume of activations in weight-stationary layouts."

**Why int8 helps most at low batch sizes.** At small batch sizes during decode, the compute time per token is small (few tokens to multiply against weights), so the forward pass is memory-bandwidth-bound on weight loading. Halving the weight bytes halves the memory time, directly translating to ~2× throughput improvement. At large batch sizes during prefill, compute time dominates, and weights are in bfloat16 for the matmuls regardless (int8 weights are dequantized to bfloat16 before computation). This is why Figure 1 shows int8 providing "just over a factor of 2" cost improvement at low latency targets but being "more neutral" at large batch sizes: "large-batch cost is dominated by the compute time and the matmuls still use bfloat16 arithmetic."

---

#### The Analytical Framework: Predicting Optimal Layouts

The paper's most distinctive contribution is not any single layout but the **analytical methodology** that predicts which layout is optimal for a given configuration without exhaustive search. This framework is described throughout Section 3 and formalized in Appendix A.

**The key insight: communication cost is analytically tractable.** While compute time and memory time depend on well-known quantities (FLOPs, memory bandwidth, model size, batch size), communication time depends on the specific partitioning layout and can be derived in closed form. The derivations in Appendix A proceed by:

1. Writing down the per-chip tensor shapes under a given partitioning layout.
2. Identifying all collective operations (all-gathers, reduce-scatters) and their input/output sizes.
3. Applying the cost model: $T_{\text{comm}} = \text{size} / \text{bandwidth}$ for each collective.
4. Summing over all collectives in the layer.
5. Optimizing over free parameters (torus axis splits $X$, $Y$, $Z$) subject to constraints.

**The budget equation for 2D weight-stationary** (Appendix A.2.1):

$$T_{\text{comm}} = \frac{2BL}{\text{bandwidth}} \cdot \left( \frac{E}{X} + \frac{F}{YZ} \right)$$

under the constraint $X \cdot Y \cdot Z = n_{\text{chips}}$. This is minimized when $E/X = F/(YZ)$, i.e., when the communication volumes for the $E$ and $F$ dimensions are balanced. Solving yields:

$$X = \sqrt{\frac{E}{F} \cdot n_{\text{chips}}}, \quad YZ = \sqrt{\frac{F}{E} \cdot n_{\text{chips}}}$$

For $F/E = 4$, this gives $X = \sqrt{n_{\text{chips}}}/2$ and $YZ = 2\sqrt{n_{\text{chips}}}$, and the minimum communication is $8BLE / (\sqrt{n_{\text{chips}}} \cdot \text{bandwidth})$.

**The switching condition between weight-stationary and weight-gathered** emerges from comparing their communication formulas:

- Weight-stationary: $T_{\text{comm}} \propto BL$
- Weight-gathered: $T_{\text{comm}} \propto \sqrt{BLF}$

The crossover occurs when $BL$ is large enough that $\sqrt{BLF} < BL$, which simplifies to $BL > F$. For PaLM 540B with $F = 73728$, this crossover is at $BL \approx 74,000$ tokens—batch size 36 at sequence length 2048, or batch size 512 at sequence length 145. Figure 7 confirms this empirically: 2D weight-stationary achieves higher MFU below ~$10^5$ tokens per batch, and weight-gathered dominates above that.

**The optimal weight-gathering degree** $N$ (Appendix A.2.2):

$$N = \sqrt{\frac{BL \cdot n_{\text{chips}}}{F}}$$

This formula captures an intuitive tradeoff: larger $BL$ (more tokens) justifies gathering weights over more chips to reduce activation communication; larger $F$ (larger weight matrices) makes gathering more expensive, so you gather over fewer chips; larger $n_{\text{chips}}$ allows more parallelism in weight gathering, so you can gather over more chips without bottlenecking on a single group.

For a concrete example: with $BL = 1$ million tokens (512 batch × 2048 sequence), $F = 73728$, $n_{\text{chips}} = 64$: $N = \sqrt{10^6 \times 64 / 73728} \approx \sqrt{868} \approx 29$. The nearest valid subset of torus axes would be $XY$ (if $X=4$, $Y=4$, $XY=16$) or $XYZ = 64$. The model would select $XYZ$ since 29 is closer to 64 than to 16, and indeed the paper reports using XYZ-weight-gathered for the high-throughput prefill configuration (Table 2).

**The maximum supported context length** (Section 4.2) is derived from memory constraints. With 64 chips, 32 GiB HBM per chip, and 30% reserved for KV cache: each chip has $0.3 \times 32 = 9.6$ GiB for KV cache. For optimized multiquery (batch-sharded), each chip stores $(B / n_{\text{chips}}) \times L \times d_{\text{head}} \times 2$ (key+value) bytes. With $n_{\text{chips}} = 64$, $B = 128$, $d_{\text{head}} = 256$: solving $128/64 \times L \times 256 \times 2 = 9.6 \times 2^{30}$ yields $L \approx 43,000$, matching Table 1. This back-of-envelope calculation demonstrates that the analytical framework extends to capacity planning, not just performance optimization.

**The Pareto frontier construction** (Section 4.4, Figure 1) is the culmination of the framework: for each model size (8B, 62B, 540B), the system sweeps over batch sizes and chip counts, applies the analytical model to select the optimal partitioning layout for prefill and decode separately, and plots the resulting (latency, cost) pairs. The envelope of these points forms the Pareto frontier—the set of configurations where you cannot improve latency without increasing cost per token, or vice versa. Tables 2 and 3 extract specific points from this frontier that represent realistic deployment scenarios.

---

#### Putting It All Together: End-to-End Inference Flow

The paper doesn't just describe components in isolation—it specifies how they combine into a complete inference system. Here is the end-to-end flow for the 540B model serving a chatbot with a 1920-token conversation history, processing a 64-token user message, and generating a 64-token response (Section 1, Table 2 "Low-latency" configuration):

**Prefill phase** (processing the 1984 input tokens: 1920 history + 64 new):
- **Batch size**: 1 (single user request, though the paper notes this could be pipelined into batch-64 decode)
- **Chips**: 64
- **Feedforward layout**: 2D weight-stationary (because $BL = 1 \times 1984 = 1984$ tokens is well below the ~74,000 crossover point)
- **Attention layout**: Sharded-over-heads (as specified in Table 2: "Head")
- **Weights format**: int8 (to minimize weight-loading time at this small batch)
- **Achieved**: 43% MFU, 0.29 seconds total prefill time

**Decode phase** (generating 64 tokens autoregressively):
- **Batch size**: 64 (the paper explains: "batch size 1 achieves best latency in the prefill phase, but for the generate phase we can increase the batch size up to 64 with negligible latency impact"—this is possible by generating multiple samples or pipelining requests)
- **Chips**: 64
- **Feedforward layout**: 2D weight-stationary (always during decode, because $BL = 64 \times 1 = 64$ tokens per step is tiny)
- **Attention layout**: Batch-sharded (multiquery attention optimized layout, to keep KV cache memory manageable at 2048 context)
- **Weights format**: int8
- **Achieved**: 14% MFU, 1.82 seconds total decode time (28.4ms per token × 64 tokens)

**Total end-to-end latency**: $0.29 + 1.82 = 2.11$ seconds (the paper quotes 1.9 seconds for a slightly different scenario with int8 and possibly optimized pipeline overlap between prefill and decode).

**Why the phase-specific strategy mixing is crucial.** If you used 2D weight-stationary for prefill at batch 512 (the high-throughput scenario in Table 2), you'd get only ~30% MFU instead of 76%—more than 2× worse efficiency. If you tried to use weight-gathered for decode, the all-gather of weights at every step would dominate: at $BL = 64$ tokens per step, $\sqrt{BLF} \approx \sqrt{64 \times 73728} \approx 2172$, which is much larger than $BLE/X \approx 64 \times 18432/4 \approx 295K$ divided by $\sqrt{64} = 8$, making weight-stationary decisively cheaper. The phase-aware switching is not a minor optimization—it's what makes the entire system viable at both latency-constrained and throughput-constrained operating points.

---

#### Design Choices: Why These Specific Strategies?

**Why 2D and not 2.5D partitioning?** Other systems (e.g., Megatron-LM training) sometimes use "2.5D" or "3D" parallelism where different dimensions (data, tensor, pipeline) are partitioned over different hardware axes. The paper focuses exclusively on tensor parallelism (partitioning individual weight matrices and activations) because pipeline parallelism introduces "bubbles" at layer boundaries where chips wait for data, which is acceptable during training (amortized over many microbatches) but unacceptable for low-latency inference with small batch sizes. The paper's approach to scaling to 64 chips is purely within the tensor-parallelism dimension.

**Why partition batch for attention but not feedforward?** The feedforward layer's computation is independent across batch elements—each sequence's MLP activations don't interact with other sequences. In principle, you could partition feedforward over batch (data parallelism) with an all-reduce at the end. But during decoding at batch 64 and one token per sequence, data parallelism would have each chip processing $64/64 = 1$ token—extremely low hardware utilization. Tensor parallelism keeps larger matrix multiplies per chip by partitioning weight matrices rather than batch elements, which is why it's preferred for decode. The attention layer uses batch partitioning only because the alternative (replicating the KV cache) would exhaust memory.

**Why int8 and not int4 or fp8?** The paper uses int8 because it's supported by the AQT library and demonstrated to have no quality loss on PaLM models. Going to int4 or fp8 would further reduce memory traffic but might require quality-aware training (quantization-aware training) that wasn't available. Activation quantization (int8 activations) is mentioned as future work that could reduce compute time and activation communication.

**Why pad heads from 48 to 64?** Section 4 notes: "For the PaLM 540B model we padded the number of attention heads up from 48 to 64 in order to partition more effectively on 64+ chips." With 48 heads and 64 chips, you can't cleanly partition—you'd have 16 chips partially replicating heads, creating load imbalance. Padding to 64 heads costs "18B parameters... which comes at a 3% MFU cost, which was more than recovered by being able to partition more effectively." This is a pragmatic engineering tradeoff: a small model quality/capacity penalty for clean divisibility that more than pays for itself in hardware efficiency.

**Why no pipeline parallelism?** Pipeline parallelism splits layers across chips, which is efficient for throughput (each chip processes a subset of layers for a stream of batches) but adds latency because chip $i$ must wait for chip $i-1$ to finish before starting. For low-latency interactive applications, this serial dependency is fatal—you need all chips working on the same layer simultaneously to minimize per-step time. The paper's exclusive use of tensor parallelism (all chips process all layers) is a conscious choice to optimize for latency, not just throughput. The Pareto frontier in Figure 1 shows that this approach achieves latencies down to 29ms per token, which would be impossible with pipeline parallelism's layer-wise serialization.

## 4. Key Insights and Innovations

### Innovation 1: A Phase-Aware Partitioning Framework That Recognizes Prefill and Decode as Fundamentally Different Workloads

The single most important conceptual move in this paper is the recognition that **prefill and decode are not two minor variants of the same computation—they are fundamentally different workloads with opposite optimal partitioning strategies**, and treating them identically (as all prior inference systems did) leaves enormous performance on the table.

Before this work, the standard approach to inference partitioning was to select one layout and apply it uniformly across the entire request lifecycle. FasterTransformer, the dominant GPU inference benchmark, uses a single tensor-parallelism/pipeline-parallelism configuration for both input processing and output generation. Training systems adapted for inference (Megatron-LM, GSPMD) similarly pick one partitioning scheme and stick with it. The implicit assumption was that the parallelism that works well during training—where all tokens are processed simultaneously—should carry over to inference.

This paper demonstrates that this assumption is wrong in a specific, quantifiable, and exploitable way. The difference comes down to a single quantity: the number of tokens in flight ($B \times L$). During prefill, this is large ($B \times L_{\text{input}}$, potentially millions of tokens for batch-512, context-2048). During decode, this is small ($B \times 1$, just one new token per sequence). This difference flips the relative cost of two communication strategies:

- **Weight-stationary layouts** pay communication proportional to $O(BL)$—the activation size. This is cheap during decode when $BL$ is tiny, but becomes expensive during prefill.
- **Weight-gathered layouts** pay communication proportional to $O(\sqrt{BLF})$—sublinear in the token count, with an additional constant overhead from weight transfer. This is expensive during decode (the constant weight-transfer overhead dominates when $BL$ is tiny), but dramatically cheaper during prefill.

The framework's key insight is that **you don't have to choose**. By maintaining weight tensors in a fixed `ExFyz` sharding that is compatible with both layouts (Section 3.2.3), the system can use 2D weight-stationary during decode and XYZ-weight-gathered during prefill without ever rewriting model parameters in memory—only the communication pattern changes between phases. This is what enables the system to simultaneously achieve 76% MFU during prefill and low-latency decode on the same hardware: the prefill layout is optimized for compute throughput, and the decode layout is optimized for memory-bandwidth-limited small-batch generation. Tables 2 and 3 show this phase-switching explicitly, with "FFN: WS 2D" (weight-stationary) for decode and "FFN: WG XYZ" (weight-gathered) for prefill in the high-throughput configuration.

This insight is **fundamental**, not incremental. It doesn't just improve an existing scheme—it redefines what a complete inference system looks like. Any future inference framework that treats prefill and decode identically is objectively leaving performance on the table, and the paper provides both the analytical framework (the crossover condition $BL \approx F$) and the empirical validation (Figures 6 and 7, Tables 2 and 3) to prove it. The practical consequence is immediate: production inference systems should select partitioning strategies per-phase, not per-model, and the switching mechanism is architecturally straightforward (just changing the communication pattern while keeping weight sharding fixed).

### Innovation 2: Batch-Dimension Sharding for Multiquery Attention That Recovers Memory Savings Otherwise Lost to Parallelism

Multiquery attention (Shazeer, 2019) was proposed as a way to reduce the KV cache size by sharing key/value heads across query heads. This is an architectural innovation that reduces memory capacity requirements—on paper, a 48× reduction for PaLM's head count. The standard assumption in the field was that you'd deploy multiquery models the same way as multihead models: partition attention over heads, and let the reduced per-head KV cache make everything cheaper.

This paper identifies a **non-obvious interaction between architecture and parallelism** that completely undermines this assumption. When you partition multiquery attention over heads (the "baseline" layout in Figure 4b), the single key and value heads must be replicated to every chip, because every query head on every chip needs to attend to them. The per-chip KV cache size becomes the *full* cache—exactly as large as multihead attention's per-chip cache, and in some configurations even worse because $d_{\text{head}}$ is typically larger in multiquery models (256 vs. 128 in the paper's comparison). Table 1 shows that the baseline multiquery layout actually supports *shorter* context lengths than multihead attention (660 vs. 1,320 tokens at batch 128) because of this replication effect.

The innovation is recognizing that the correct axis to partition over is **batch**, not heads. During decode, the Q, K, V tensors for the current token are tiny (one token per sequence), while the KV cache contains thousands of tokens. By sharding the cache over batch, each chip stores keys and values for only $B/n_{\text{chips}}$ sequences, recovering the $n_{\text{chips}}$-fold memory reduction. The cost is an all-to-all collective on the small Q, K, V tensors to reshuffle between heads-sharding (used for the projection matrices, which follow the feedforward layer's partitioning) and batch-sharding (used for the attention computation).

The conceptual leap here is the recognition that **the cost asymmetry between the large, persistent KV cache and the small, transient Q/K/V tensors makes batch-sharding profitable despite introducing a new communication operation**. This is not obvious a priori—all-to-all is generally more expensive per byte than all-gather or reduce-scatter, and adding communication to save memory bandwidth is not an automatic win. The paper's analytical justification (Section 3.3: "Since the KV cache is orders of magnitude larger than the Q, K, and V tensors, it is very profitable to spend the all-to-all communication time on the small tensors to save the memory time on the large tensors") is critical for establishing that this is not just a hack but a principled optimization.

The empirical result is a 32–64× increase in maximum context length (Table 1: from 660 to 43,000 tokens at batch 128), which is not just a quantitative improvement but a **qualitative change in capability**—it makes long-context inference practically feasible on memory-constrained hardware. This is significant because it means the architectural advantage of multiquery attention is not automatically realized at deployment time; it requires the right partitioning strategy to materialize. The paper provides both the diagnosis (why naive deployment fails) and the solution (batch-sharded attention), making this a **fundamental** contribution to the practice of deploying attention-efficient architectures.

### Innovation 3: An Analytical Model That Predicts Optimal Partitioning Without Exhaustive Search, Transforming Partitioning from a Heuristic Art to a Solvable Optimization

Prior work on model parallelism (Megatron, GSPMD, Alpa, DeepSpeed Inference) treats partitioning as a search problem: define a space of possible layouts, then use profiling, integer linear programming, or heuristics to find a good configuration. This works, but it's expensive (requiring many trial runs), opaque (it's hard to understand *why* a particular layout was chosen), and brittle (the search must be rerun for every model architecture and hardware topology change).

This paper's most intellectually distinctive contribution is demonstrating that, for the specific domain of Transformer inference on a torus-connected accelerator mesh, **the optimal partitioning layout can be predicted analytically from a small set of easily-computed quantities—model dimensions ($E$, $F$), batch size in tokens ($BL$), chip count ($n_{\text{chips}}$), and hardware bandwidth numbers—using closed-form formulas derived in Appendix A**.

The key results from this analytical framework are:

- **The 2D vs. 1D criterion:** 2D weight-stationary is preferable when $\sqrt{n_{\text{chips}}} > F/E$. For standard Transformers with $F = 4E$, this means $n_{\text{chips}} > 16$. Below 16 chips, the additional communication from the second partitioning dimension isn't worth it; above 16, the $O(1/\sqrt{n_{\text{chips}}})$ scaling of 2D dominates. This is validated in Figure 6, where 2D outperforms 1D on 64 chips.

- **The weight-stationary vs. weight-gathered crossover:** The transition occurs when $BL \approx F$, derived by comparing $O(BL)$ communication for weight-stationary against $O(\sqrt{BLF})$ for weight-gathered. For PaLM 540B with $F = 73,728$, this crossover is at ~74,000 tokens. Figure 7 shows this empirically: below ~100,000 tokens per batch, weight-stationary achieves higher MFU; above that threshold, weight-gathered dominates, reaching 76% MFU at $10^6$ tokens.

- **The optimal weight-gathering degree:** $N = \sqrt{BL \cdot n_{\text{chips}} / F}$ tells you how many torus axes to all-gather weights over. This captures the tradeoff: larger $BL$ justifies more weight gathering to reduce activation communication; larger $F$ makes weight gathering more expensive per byte, so you gather over fewer chips.

What makes this a genuine **innovation** rather than just good engineering is that it transforms partitioning from a black-box search problem into a **predictable design space**. A practitioner can now answer questions like "Should I switch to 2D partitioning at 32 chips?" or "At what batch size does weight-gathering become cheaper?" without running any experiments—the formulas give the answers directly. The paper validates these predictions empirically (Figures 6, 7; Appendix D comparisons), but the framework's value extends beyond the specific results: it provides an **intellectual toolkit** that generalizes to new model architectures, hardware topologies, and batch size regimes.

This contribution is **fundamental** in the sense that it establishes a scaling theory for inference partitioning that parallels Kaplan et al. (2020)'s scaling laws for training. Just as those laws tell you how to allocate a training compute budget between model size and data without exhaustively trying every combination, these formulas tell you how to allocate a hardware topology (chip count and interconnect) across model dimensions without exhaustively profiling every layout. The paper explicitly positions itself in these terms (Section 1: "enables the user to intuitively understand the tradeoffs and select the best multi-axis tensor partitioning strategy... in contrast to a black-box exhaustive search over partitioning strategies").

### Innovation 4: The Identification of Communication as the Central Scaling Bottleneck for Inference—Not Compute, Not Memory Capacity, but Communication Bandwidth

A subtle but important conceptual contribution is the paper's reframing of what limits inference scalability. The dominant narrative in large model deployment has been that **memory capacity** is the primary constraint—models don't fit on a single chip, so you must shard them across many chips. This is true, but it's only the *first-order* constraint. The paper argues that once you've solved the capacity problem by using enough chips, the **second-order constraint**—and the one that determines whether you can actually achieve low latency—is **inter-chip communication bandwidth**.

This reframing is implicit in the paper's cost model structure (Section 2, Section 3, Appendix A). The paper decomposes inference time into compute, memory, and communication components, and shows that while compute and memory time *both decrease linearly with chip count* (because work and data are divided across more chips), communication time **does not**—or at least, decreases much more slowly. In the 1D weight-stationary layout, communication time is constant with chip count ($T_{\text{comm}} \propto 1$). In the 2D layout, it decreases as $1/\sqrt{n_{\text{chips}}}$, which is better but still sublinear. In the weight-gathered layout, it decreases as $\sqrt{BL}/\sqrt{n_{\text{chips}}}$—better at large $BL$, worse at small $BL$.

The consequence is that as you add chips to reduce latency, you eventually hit a **communication wall** where further chips provide no additional speedup because the interconnect cannot move data fast enough to keep the additional compute units busy. This is visible empirically: FasterTransformer's 32-way tensor parallelism achieves only 33% MFU compared to 46% at 16-way, which the paper attributes to "a communication bottleneck of scaling tensor parallelism beyond this point" (Section 5). The paper's 2D partitioning pushes this wall further out—achieving 44% MFU at 64-way parallelism—but it doesn't eliminate it.

The intellectual contribution here is not the discovery of this bottleneck (Rajbhandari et al., 2020; Korthikanti et al., 2022 have noted similar effects), but rather the **systematic characterization** of how it manifests across different partitioning strategies, batch sizes, and inference phases. The paper provides a unified language for discussing this bottleneck: communication volume measured in bytes transferred per chip per forward pass, expressed analytically in terms of model dimensions and batch size. This makes it possible to compare strategies apples-to-apples and predict when communication will become the limiter.

The practical implication (Section 7) is significant: "FLOP count and communication volume can fundamentally limit inference performance of dense Transformer models." This directs future research toward reducing communication volume (via sparsity, mixture-of-experts, or adaptive computation) rather than simply throwing more chips at the problem. The negative result—that 64-way tensor parallelism is approaching the limits of what the interconnect can support—is itself valuable, because it tells practitioners that further scaling of dense model inference will require either better interconnect technology or model architectures with lower communication requirements (like sparsely-activated experts, Fedus et al., 2022). This is a **fundamental** insight because it bounds what's achievable within the current paradigm, not just what was achieved in this paper.

### Innovation 5: The Demonstration That int8 Weight Quantization Provides a "Free" 2× Cost Improvement at Low Batch Sizes, with Zero Quality Loss

While weight quantization for inference is not a new technique (Dettmers et al., 2022; Abdolrashidi et al., 2021; and others have explored it extensively), this paper makes a **specific diagnostic contribution**: it shows that the benefit of weight quantization is highly regime-dependent, and that the regime where it matters most—low-batch-size, latency-constrained decoding—is exactly the regime where it provides the largest gains with the simplest implementation.

The key observation (Section 3.6, validated in Figure 1) is that at small batch sizes during decoding, the forward pass is **memory-bandwidth-bound on weight loading**. The compute time per token is small (few tokens to multiply against the weight matrices), but every token must load the full model weights from HBM. In this regime, halving the weight data volume (int8 vs. bfloat16) directly translates to roughly halving the memory time, which directly translates to roughly halving the total latency—hence the "just over a factor of 2" cost improvement at low latency targets.

At large batch sizes during prefill, the picture reverses: the forward pass becomes compute-bound because the large number of tokens keeps the arithmetic units busy. Weight quantization doesn't help here because (a) weight loading is no longer the bottleneck, and (b) the matmuls still use bfloat16 arithmetic even with int8 weights (the weights are dequantized on the fly). The paper reports that at large batch sizes, "cost is more neutral between int8 and bfloat16."

This is not just a benchmark result—it's a **conceptual clarification** of where quantization delivers value. Many prior works treat quantization as a uniform optimization that improves all inference regimes equally. This paper shows that its benefit is concentrated in the low-batch-size regime, and that for throughput-oriented workloads, the complexity of quantization may not be worth the marginal improvement. The paper's decision to use int8 for low-latency configurations and bfloat16 for high-throughput configurations (Tables 2 and 3: "int8" for low-latency, "bfloat16" for high-throughput) reflects this regime-dependent reasoning.

The paper is also candid about the limitations: they have not implemented activation quantization, which could provide additional benefits in the large-batch regime (by reducing compute time and activation communication volume). This honesty—specifying what was *not* done and why it might matter—strengthens the contribution by clearly demarcating what the current results do and do not prove.

This is an **incremental** contribution relative to the vast quantization literature, but it's a practically valuable one because it provides clear guidance for practitioners deciding whether and when to deploy quantized models. The rule of thumb—if you're latency-constrained at small batch sizes, quantize weights; if you're throughput-constrained at large batch sizes, bfloat16 is fine—is simple, actionable, and backed by data.

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** The paper does not use a standard benchmark dataset in the ML sense—there is no MATH, ImageNet, or SQuAD equivalent here. Instead, the "dataset" is a set of **inference workload configurations** defined by the application requirements: a specific PaLM model (8B, 62B, or 540B parameters), a context length (typically 2048 tokens, but swept from 32 to 8192+ for attention benchmarks), a batch size (swept from 1 to 1024), and a task specification (prefill-only, decode-only, or full inference with specified input/output token counts). The system's performance is measured on these configurations, not on accuracy on a downstream task—the paper is purely about inference efficiency, not model quality.

**Base model(s).** The experiments use the **PaLM family of models** (Chowdhery et al., 2022) at three scales: 8B, 62B, and 540B parameters. The models are selected because they natively incorporate the two architectural features the paper's partitioning strategies exploit: **multiquery attention** (reduces KV cache size by sharing key/value heads across query heads) and **parallel attention/feedforward layers** (enables fusion of projection matrices and halves communication per layer, as described in Section 3.4). The 540B model has a feedforward dimension $F = 73728$ (4× the model dimension $E = 18432$), 118 layers, and 48 attention heads—but the authors **pad the heads to 64** specifically "in order to partition more effectively on 64+ chips" (Section 4, Methodology), adding 18B parameters at a 3% MFU cost that is "more than recovered by being able to partition more effectively." For the FasterTransformer comparison (Section 5), the paper also benchmarks the **Megatron-Turing NLG 530B** model (Smith et al., 2022) to provide an apples-to-apples comparison with prior GPU-based benchmarks, since both models are ~500B parameters.

**Metrics.** The paper measures inference performance through **three primary metrics**, each serving a distinct evaluation purpose:

- **Latency:** The wall-clock time for an inference operation, measured in milliseconds or seconds. For decode, this is reported both as *total decode latency* (time to generate all output tokens) and *latency per step* (total decode latency divided by number of generated tokens, effectively ms per token). For prefill, latency is the time to process all input tokens in a single forward pass. Latency matters primarily for interactive applications where users wait for responses.

- **Model FLOPS Utilization (MFU):** The ratio of observed throughput (tokens per second) to the theoretical maximum throughput if every chip operated at peak FLOPS with zero memory or communication overhead. Formally, for a forward pass, MFU = (observed FLOPs per second) / (n_chips × 275 TFLOPS per chip, for bfloat16 on TPU v4). MFU captures how effectively the hardware is being used—100% MFU would mean the compute units are never idle waiting for data. This metric normalizes across different chip counts and chip types, enabling fair comparison with the FasterTransformer GPU benchmarks.

- **Cost:** Measured in chip-seconds per token, calculated as `n_chips × latency / (B × L)` where $B$ is batch size and $L$ is sequence length in tokens. This quantity is directly proportional to the dollar cost of inference (more chip-seconds = more accelerator rental time) and inversely proportional to MFU (higher MFU means fewer chip-seconds per token). The Pareto frontier plots in Figure 1 use cost on the y-axis and latency on the x-axis, allowing practitioners to select operating points based on their budget and latency requirements.

Additionally, for the attention partitioning experiments (Section 4.2, Table 1), the paper reports **maximum context length**—the longest sequence that fits in memory under a fixed budget of 30% of total HBM reserved for the KV cache—as a capacity metric.

**Baselines.** The paper evaluates against several baselines that represent the state of prior practice:

- **FasterTransformer** (FasterTransformer GitHub, cited in Section 5): NVIDIA's widely-used multi-GPU inference framework for Transformer models, benchmarked on Megatron-Turing NLG 530B using 16–32 NVIDIA A100 GPUs. The paper compares against three FasterTransformer configurations: TP16 (16-way tensor parallelism), TP32 (32-way tensor parallelism), and PP3/TP8 (3-way pipeline parallelism with 8-way tensor parallelism). This is the primary external baseline and the standard against which inference efficiency is measured in the field.

- **1D weight-stationary layout** (Section 3.2.1, Figure 2a): The Megatron-LM partitioning strategy (Shoeybi et al., 2019) that shards feedforward layers only along the $F$ dimension. This serves as the internal baseline for evaluating 2D weight-stationary and weight-gathered layouts, representing what a practitioner would naively deploy.

- **Multihead attention variant** (Section 4.2): A modified PaLM 540B with standard multihead attention instead of multiquery, keeping parameter count constant by reducing $d_{\text{head}}$ from 256 to 128. This isolates the benefit of multiquery attention from other architectural differences.

- **"Inefficient" multiquery layout** (Figure 4b, Section 4.2): Multiquery attention partitioned over heads (rather than batch), where the single key/value head is replicated to every chip. This represents what a naive deployment of multiquery attention would look like—using the same partitioning strategy as multihead attention—and demonstrates that architectural improvements require matching partitioning choices to materialize their benefits.

- **Serial attention/feedforward formulation** (Section 4.3): A variant of PaLM 540B with the standard serial Transformer block instead of the parallel formulation, used to isolate the communication savings from the parallel architecture.

For the PaLM models themselves, the paper does not compare against different partitioning frameworks (e.g., DeepSpeed Inference or Alpa) since those are GPU-specific and not directly portable to TPU v4. The FasterTransformer comparison serves as the cross-platform benchmark by normalizing to MFU.

**Generation budget / compute accounting.** The paper measures compute in terms of **hardware configuration** rather than FLOP counts per inference. The "budget" is specified as a (n_chips, batch_size, sequence_length) triple, and the system's task is to minimize latency or maximize MFU given that allocation. For prefill, the number of tokens in flight is $B \times L_{\text{input}}$; for decode, it's $B \times 1$ per step, with $L_{\text{gen}}$ steps executed sequentially. The cost metric (chip-seconds per token) converts this into an economic quantity. When comparing int8 vs. bfloat16, the paper accounts for the fact that while weights are stored as int8, the matrix multiplications still use bfloat16 arithmetic (weights are dequantized on the fly), so the compute time is unchanged and only the memory traffic is reduced.

**Cross-validation / statistical protocol.** The paper does **not use cross-validation or statistical testing** in the traditional ML sense, because it is evaluating systems performance (latency, throughput, MFU) rather than model accuracy. Performance numbers are deterministic given a hardware configuration (TPU v4 chips are consistent, there is no random variation in execution time at this scale). The "validation" is instead the sweep over batch sizes, chip counts, and context lengths that establishes the Pareto frontier—each point on Figure 1 represents a distinct experiment, and the envelope of those points defines the best achievable performance. For the FasterTransformer comparison (Section 5, Appendix D), the paper reproduces the exact benchmark specifications from the FasterTransformer documentation (20 input / 8 output tokens, 60 input / 20 output tokens, 128 input / 8 output tokens) to ensure a fair comparison. The specific configurations tested for the Pareto frontier (Figure 1, Tables 2–3) are selected from the sweep to represent points on the tradeoff curve, not cherry-picked results.

---

### Main Quantitative Results

#### Feedforward Layer Partitioning: 2D vs. 1D Weight-Stationary During Decoding

**Headline result:** On 64 chips during text generation (decode) with PaLM 540B at batch size 512, 2D weight-stationary layout outperforms 1D weight-stationary, achieving lower latency per token because of its asymptotically better communication scaling with chip count (Figure 6).

**Details and context:** Figure 6 plots latency per decode step as a function of chip count for both layouts. The paper states: "Both layouts start to become communication-limited, but the 2D layout performs better because of its asymptotically better scaling with chip count." While exact numerical latency values are not provided in the text for Figure 6, the analytical prediction from Section 3.2.2 is that 2D weight-stationary becomes more efficient than 1D when $\sqrt{n_{\text{chips}}} > F/E$. For PaLM 540B with $F/E = 4$, this threshold is $n_{\text{chips}} > 16$. At 64 chips, this condition is satisfied by a factor of 2× ($\sqrt{64} = 8$ vs. the threshold of 4), so the 2D layout's communication advantage is substantial. This result validates the analytical framework's core prediction and establishes that for deployments at scale (64 chips), the more complex 2D partitioning is not just theoretically better but practically necessary.

**Why this matters:** The 1D layout is the standard approach from Megatron-LM and is simpler to implement. A practitioner might reasonably ask whether the additional complexity of 2D partitioning—managing two communication axes, balancing $X$ and $YZ$ torus dimensions—is worth it. Figure 6 demonstrates that at realistic deployment scales, the answer is a clear yes.

---

#### Feedforward Layer Partitioning: Weight-Gathered vs. Weight-Stationary During Prefill

**Headline result:** During prefill on 64 chips with sequence length 2048, the optimal partitioning layout switches from 2D weight-stationary to weight-gathered as batch size (in tokens) increases. At the largest batch sizes (~10^6 tokens), weight-gathered achieves 76% MFU, nearly saturating the hardware, while 2D weight-stationary plateaus at lower efficiency (Figure 7).

**Detailed scaling behavior (Figure 7):** The paper sweeps batch sizes from 125,000 tokens to over 1,000,000 tokens per batch (number of sequences × 2048 tokens per sequence) and reports MFU for both layouts. The exact MFU values read from Figure 7 show:

- At ~125,000 tokens per batch: 2D weight-stationary achieves roughly 30% MFU; 2D weight-gathered achieves roughly 20% MFU. Weight-stationary is superior at this small scale.
- At ~250,000 tokens per batch: Both layouts achieve roughly 35–40% MFU—the crossover point where they become comparable.
- At ~500,000 tokens per batch: Weight-gathered achieves roughly 55% MFU versus ~42% for weight-stationary.
- At ~1,000,000 tokens per batch (512 sequences × 2048 tokens): Weight-gathered achieves **76% MFU** versus ~50% for weight-stationary.

The paper comments: "The weight-gathered layouts are inefficient at low batch sizes, but eventually they become the most efficient at high batch sizes, achieving 76% MFU when the communication overhead is almost negligible." This aligns with the analytical prediction from Appendix A.2.2: weight-stationary communication scales as $O(BL)$ while weight-gathered scales as $O(\sqrt{BLF})$, so at large $BL$, the weight-gathered square-root scaling dominates.

**Operational significance:** The paper also notes that "such large batch sizes would fail from memory exhaustion without multiquery attention, as shown in Section 4.2." This is a crucial coupling between the two partitioning innovations: the weight-gathered layout enables high MFU during prefill, but it only fits in memory because the batch-sharded multiquery attention layout (Section 3.3) keeps the KV cache compact enough to leave room for these large activation batches.

**The "basic strategy" distillation:** Based on these results, the paper articulates a simple decision rule (Section 4.1): "during the prefill phase, we select from weight-stationary and weight-gathered layouts based on the current number of tokens in the batch. During the generate phase, we select the 2D weight-stationary layout because the batch size in tokens is always small." This rule is applied in the specific configurations shown in Table 2, where the low-latency scenario uses WS 2D for both phases (small batch sizes), while the high-throughput scenario uses WG XYZ for prefill and WS 2D for decode.

---

#### Attention Layer Partitioning: Batch-Sharded Multiquery vs. Multihead and Naive Multiquery

**Headline result:** The optimized batch-sharded multiquery attention layout enables **32–64× longer context lengths** than multihead attention or naive multiquery partitioning, while also providing modest latency improvements during decode that become more significant at longer contexts (Table 1, Figure 8).

**Memory capacity results (Table 1):** The paper measures the maximum context length that fits on 64 chips with 30% of HBM reserved for KV cache:

| Layout | $d_{\text{head}}$ | Max context (B=128) | Max context (B=512) |
|---|---|---|---|
| Multihead attention | 128 | 1,320 | 330 |
| Baseline multiquery (sharded over heads) | 256 | 660 | 165 |
| Optimized multiquery (sharded over batch) | 256 | **43,000** | **10,700** |

The optimized layout supports 32.6× longer contexts than multihead at batch 128 (43,000 / 1,320), and 64.8× longer at batch 512 (10,700 / 165). The baseline multiquery actually performs *worse* than multihead (660 vs. 1,320 at batch 128), confirming the paper's diagnosis that naive head-sharded multiquery attention loses the memory savings by replicating the single K/V head to all chips.

**Latency scaling results (Figure 8):** The paper benchmarks an 8-layer version of PaLM 540B (to isolate per-layer behavior) on 64 chips with batch size 256, sweeping context lengths from 128 to 8192 tokens. Key observations:

- At context length 128: All three layouts have similar latency (~20–25ms per step). The feedforward layers dominate runtime at short contexts, so attention layout differences are negligible.
- At context length 512: The optimized multiquery layout has marginally lower latency than multihead and baseline multiquery (exact numbers not quoted in text, visible from Figure 8's plot).
- At context length 2048: The optimized multiquery layout shows approximately 30ms per step, while multihead and baseline multiquery show approximately 40–45ms per step—a roughly 25–35% improvement.
- At context length 8192: Only the optimized multiquery layout is shown (the dotted line indicates the other layouts run out of memory for the full 118-layer model). The latency reaches approximately 60ms per step, with attention comprising "8–31% of total runtime" across the tested range.

The paper directly quantifies the attention overhead: "Multiquery attention scales up to sequence lengths of 8192–32,768 tokens (batch sizes 512 and 128 respectively) with attention taking only 8–31% of total runtime." This means that even at 8192-token contexts, the feedforward layers still account for 69% of runtime—the attention optimization ensures that the KV cache loading doesn't become the dominant fraction.

**Why the batch-512 configuration maxes out at lower context length:** The per-chip KV cache size scales as $(B / n_{\text{chips}}) \times L \times d_{\text{head}} \times 2$. At batch 512 with 64 chips, $B/n_{\text{chips}} = 8$, so each chip stores KV cache for 8 sequences. At batch 128, $B/n_{\text{chips}} = 2$, so each chip stores cache for only 2 sequences—a 4× difference that directly translates to 4× longer max context length (43,000 vs. 10,700 ≈ 4.02×).

---

#### Parallel vs. Serial Attention/FFN Layers

**Headline result:** The parallel Transformer block formulation reduces decode latency by 14% compared to the serial formulation on 64 chips with batch size 512, primarily by eliminating one all-reduce operation per layer (Section 4.3).

**Details:** The paper constructs a variant of PaLM 540B with the standard serialized Transformer block (attention followed by feedforward, with separate LayerNorms and separate communication collectives for each) and compares it against the native parallel formulation using 2D weight-stationary layout during decoding. The 14% figure is attributed specifically to "increased communication time for activations" in the serial variant. During prefill, this gap shrinks because the weight-gathered layouts used at large batch sizes "incur less activation communication," making the halving of all-reduces less impactful. The paper does not provide a separate prefill comparison number for this ablation.

---

#### End-to-End Results: The Pareto Frontier

**Headline result:** The paper establishes a Pareto frontier between latency and cost (chip-seconds per token) for all three PaLM model sizes (8B, 62B, 540B) with both bfloat16 and int8 weights, at a context length of 2048 tokens, by sweeping batch sizes and chip counts and selecting the optimal partitioning strategy at each point (Figure 1, Tables 2 and 3).

**Decoding Pareto frontier (Figure 1, left):** The plot shows cost (chip-milliseconds per token) on the y-axis (log scale) vs. latency per generated token (milliseconds) on the x-axis (log scale), for generating 64 tokens assuming the context is already processed. Key features:

- **PaLM 540B with int8 weights** achieves the lowest absolute latency: **28.5ms per token at batch 64** (the leftmost point on the 540B-int8 curve). The text reports "29ms per token" in Section 1, with the specific 28.5ms figure in Section 4.4. With bfloat16 weights, the minimum decode latency is 36.9ms per token—int8 provides a 23% latency reduction.
- **Larger models achieve better cost at low latency than smaller models would.** The 540B model's curve extends further to the left (lower latency) than the 62B and 8B curves because larger models can be partitioned across more chips before communication-limited scaling kicks in. The paper estimates "an approximately square-root relationship between model size and latency based on Figure 1 (left)."
- **The cost penalty for low latency is steep.** For PaLM 540B, the lowest-latency point (batch 64, 28.5ms/token) costs approximately 64 chip-milliseconds per token. The lowest-cost point on the same curve (batch 512+, further right) costs roughly 4–8 chip-milliseconds per token—a ~8–16× cost increase per token to achieve ~3× lower latency. This is the fundamental tradeoff the Pareto frontier maps.
- **int8 vs. bfloat16:** At the lowest latency targets, int8 improves cost by "just over a factor of 2" because low-batch-size cost is dominated by weight loading time, which int8 halves. At large batch sizes (lower cost, higher latency), int8 and bfloat16 converge because compute time dominates and "the matmuls still use bfloat16 arithmetic."

**Prefill Pareto frontier (Figure 1, right):** The same cost vs. latency tradeoff, but for processing 2048 input tokens (excluding output generation). Notable differences from decode:

- **The batch-size-to-latency tradeoff is less severe.** Even batch-1 prefill runs with "fairly low cost"—the cost penalty for low latency is smaller during prefill because the large number of tokens in flight ($L_{\text{input}} = 2048$) provides natural parallelism.
- **Prefill cost is ~2× lower than decode cost at the same batch size.** The paper reports that "batch-512 prefill is 2 times lower than batch-512 generate because of the increased MFU of the weight-gathered layouts we use during prefill."
- **The Pareto frontier curves are smoother** because the weight-gathered layouts automatically adapt to the token count, providing good efficiency across a wider range of batch sizes.

**Example configurations extracted from the frontier (Tables 2 and 3):**

For PaLM 540B on 64 chips (Table 2):

| Scenario | Phase | Batch | FFN Layout | Attn Sharding | Weights | MFU | Latency |
|---|---|---|---|---|---|---|---|
| Low-latency | Prefill | 1 | WS 2D | Head | int8 | 43% | 0.29s |
| Low-latency | Decode | 64 | WS 2D | Batch | int8 | 14% | 1.82s |
| High-throughput | Prefill | 512 | WG XYZ | Batch | bfloat16 | 76% | 85.2s |
| High-throughput | Decode | 512 | WS 2D | Batch | bfloat16 | 33% | 6.0s |

For PaLM 62B (Table 3):

| Scenario | Phase | Batch | Chips | FFN Layout | Attn Sharding | Weights | MFU | Latency |
|---|---|---|---|---|---|---|---|---|
| Low-latency | Prefill | 1 | 16 | WS 2D | Head | int8 | 36% | 0.16s |
| Low-latency | Decode | 32 | 16 | WS 2D | Batch | int8 | 8% | 0.73s |
| High-throughput | Prefill | 512 | 32 | WG XYZ | Batch | bfloat16 | 73% | 20.2s |
| High-throughput | Decode | 512 | 8 | WS 2D | Batch | bfloat16 | 37% | 5.1s |

Several patterns are informative:

- **Larger models need more chips but similar batch sizes:** The 540B uses 64 chips across both configurations, while the 62B uses 8–32 chips depending on the scenario. The batch sizes are similar (1 vs. 1 for low-latency prefill, 512 vs. 512 for high-throughput). This means the 540B model achieves comparable or better MFU at similar batch sizes by using more chips with better partitioning strategies (2D vs. 1D weight-stationary, enabled by $n_{\text{chips}} > 16$).

- **Low-latency decode MFU is very low:** 14% for 540B, 8% for 62B. This reflects the fundamental difficulty of the decode phase—processing only one token per sequence at a time, the arithmetic units are starved for work. Yet the absolute latencies (28.5ms/token for 540B, roughly 22.8ms/token for 62B at batch 32) are adequate for interactive applications.

- **High-throughput prefill MFU is high for both:** 76% for 540B, 73% for 62B. These near-peak utilizations during prefill compensate for the low decode efficiency, yielding good overall system throughput when prefill dominates the workload (long inputs, short outputs).

- **The batch size mismatch between prefill and decode:** The paper explicitly notes that "this mixture of batch sizes is possible in practice either by generating multiple samples from the same input text, or by pipelining a batch-1 prefill server into a batch-64 decoding server." This is the practical realization of the phase-aware strategy: prefill and decode can be served by different hardware partitions or pipelined with different batch configurations.

**Scaling behavior across model sizes (Figure 1, Appendix C.1):** The paper observes that "in most cases, the larger models achieve higher MFUs than the smaller models, because larger matrix multiplies are more efficient." However, there is an exception: "at long-latency decodes, PaLM 62B achieves higher MFU than PaLM 540B, because the former uses 8-way model parallelism and the latter uses 64-way model parallelism." This suggests that for throughput-oriented decode (where latency tolerance allows larger batch sizes), it may be optimal to use fewer chips for the larger model to reduce communication overhead—a potentially counterintuitive result that the paper flags but does not fully explore.

---

#### Comparison with FasterTransformer

**Headline result:** The paper's implementation of both PaLM 540B and Megatron-Turing NLG 530B on 64 TPU v4 chips achieves a new Pareto frontier on the latency-MFU tradeoff, outperforming FasterTransformer on NVIDIA A100 GPUs across most configurations (Figure 9, Appendix D Tables D.2–D.4).

**Benchmark setup:** The FasterTransformer benchmarks specify three fixed inference scenarios:
1. **20 input tokens, 8 output tokens** (Table D.2)
2. **60 input tokens, 20 output tokens** (Table D.3)
3. **128 input tokens, 8 output tokens** (Table D.4)

For each scenario, FasterTransformer reports total end-to-end latency and MFU for three configurations: TP16 (16-way tensor parallelism on 16 GPUs), TP32 (32-way tensor parallelism on 32 GPUs), and PP3/TP8 (3-way pipeline parallelism × 8-way tensor parallelism, on 24 GPUs?). The paper benchmarks both its PaLM 540B implementation and its Megatron 530B implementation on 64 TPU v4 chips (using 2D weight-stationary partitioning) across batch sizes from 4 to 1024, reporting prefill latency, decode latency, and total latency separately along with MFU.

**Headline from the 60-input / 20-output benchmark (Table D.3, Figure 9):**

- **Best absolute latency:** The paper's PaLM implementation achieves the lowest total latency at each batch size. At batch 4, PaLM achieves 690ms total (50ms prefill + 640ms decode) vs. FasterTransformer's TP32 at 1,110ms—a 38% latency reduction. At batch 64, PaLM achieves 1,218ms vs. FT TP16 at 3,383ms—a 64% reduction.
- **Best MFU:** The paper's PaLM implementation achieves the highest MFU at each latency target. At batch 128, PaLM achieves 35% MFU with 1,814ms total latency. FasterTransformer's best MFU at a comparable latency is 27% (TP32 at 4,099ms). At batch 1024, PaLM achieves 43% MFU; FT's best overall MFU is 40% (TP16 at 5,406ms), but at more than 4× the latency.
- **Megatron vs. PaLM on the same hardware:** The paper's Megatron implementation runs the 530B model on 64 TPU v4 chips and achieves MT-NLG-best MFU of 40% (vs. FT's best of 40%, but at half the latency—3,189ms vs. 5,406ms). The PaLM implementation outperforms Megatron by up to "10% MFU in this benchmark primarily because of the parallel attention/ffn layers."

**Scaling behavior comparison:** The paper identifies a specific limitation in FasterTransformer's scalability: "Their 32-way tensor parallelism achieves a maximum of 33% MFU across all reported benchmarks, compared to 46% MFU in their 16-way tensor parallel configuration. This likely indicates a communication bottleneck of scaling tensor parallelism beyond this point. In contrast, our implementation is able to scale up to 64-way tensor parallelism while still achieving 44% MFU, suggesting superior scalability of our 2D weight-stationary partitioning strategy on TPU v4's larger high-speed interconnect domains."

This is a critical competitive advantage: the paper's 2D partitioning enables scaling to higher chip counts without communication collapse, allowing it to simultaneously achieve lower latency (via more chips) *and* acceptable efficiency (via $O(1/\sqrt{n_{\text{chips}}})$ communication scaling). FasterTransformer's 1D tensor parallelism hits a communication wall around 32 GPUs.

**Result across all three benchmarks (Tables D.2–D.4):** The pattern is remarkably consistent. In the 20-input/8-output benchmark (Table D.2), the paper's PaLM implementation achieves 44% MFU at 4,041ms total latency (batch 1024), vs. FT's best of 46% at 3,341ms (TP16, batch 256)—a tradeoff but with PaLM achieving higher absolute throughput at scale. In the 128-input/8-output benchmark (Table D.4), PaLM achieves 46% MFU at 2,343ms (batch 128) vs. FT's best of 46% at 4,002ms (TP16, batch 64)—same MFU at 42% lower latency.

**Important caveat:** The hardware is not identical—TPU v4 vs. A100—so the MFU normalization is essential for a fair comparison. MFU accounts for different peak FLOPS (275 TFLOPS for TPU v4, 312 TFLOPS for A100 in bfloat16) and different chip counts (64 vs. 16–32). A higher MFU on TPU v4 means the chip's compute units are better utilized, which translates to lower cost per token. But the absolute latency numbers also depend on the total FLOPS available (64 × 275 = 17,600 TFLOPS for TPU v4 vs. 16 × 312 = 4,992 TFLOPS for A100-16-way), so the latency advantage is partly a "more hardware" advantage enabled by the better scalability.

---

### Ablation Studies and Robustness Checks

**1D vs. 2D weight-stationary as chip count scales:** Figure 6 shows that both layouts become communication-limited as chip count grows, but 2D degrades more gracefully. At 64 chips, 2D achieves clearly lower latency, validating the analytical model's prediction that the crossover occurs at $n_{\text{chips}} > 16$. This ablation confirms that 2D is not just theoretically better but practically necessary at the scales targeted in this paper.

**Weight-stationary vs. weight-gathered as batch size scales:** Figure 7 demonstrates the crossover behavior predicted by the analytical model: weight-gathered is worse at low token counts but becomes substantially better above ~3 × 10⁵ tokens per batch, reaching 76% MFU at 10⁶ tokens. This validates Appendix A.2.2's derivation and establishes that the switching threshold ($BL \approx F \approx 74,000$ tokens) is empirically meaningful.

**Multiquery attention: head-sharded vs. batch-sharded:** Table 1 provides a stark ablation: the baseline multiquery layout (sharded over heads) supports only 660 tokens at batch 128—worse than multihead attention's 1,320 tokens—while the optimized batch-sharded layout supports 43,000 tokens. This demonstrates that multiquery attention's architectural advantage is *entirely contingent* on the partitioning strategy; without batch-sharding, it's actually *worse* than multihead attention. Figure 8 further confirms that the optimized layout provides latency benefits that grow with context length.

**Parallel vs. serial Transformer block:** The 14% decode latency reduction (Section 4.3) isolates the benefit of the parallel formulation's halved communication. During prefill, the gap shrinks because weight-gathered layouts have less activation communication to begin with, confirming that the benefit is specific to activation-heavy weight-stationary layouts used during decode.

**int8 weight quantization:** The comparison between int8 and bfloat16 curves in Figure 1 shows that int8 provides ~2× cost improvement at low latency targets (where weight loading dominates) but little benefit at high throughput (where compute dominates). This regime-dependent behavior validates the analytical understanding of when memory bandwidth is the bottleneck.

**Chip count scaling for PaLM 62B vs. 540B:** Appendix C.1 (Figure C.1) reveals a non-obvious result: at long-latency decodes, PaLM 62B (on 8 chips) achieves higher MFU than PaLM 540B (on 64 chips), because excessive model parallelism creates communication overhead. This suggests that the optimal chip count for decode may be lower than what memory capacity requires for loading all parameters—a counterintuitive finding that the paper flags but does not optimize for, noting "We may be able to further optimize PaLM 540B by reducing the model parallelism in the high-throughput (latency-tolerant) regime."

**Batch size sweep for prefill latency at batch=1 (Appendix B, Figure B.1):** The paper sweeps sequence lengths from 32 to 1024 at batch size 1 and finds that prefill cost per token is remarkably flat across sequence lengths—the compute time scales linearly with $L$, and the cost per token (chip-seconds / tokens) remains nearly constant. This demonstrates that prefill efficiency is robust to sequence length at batch 1, which is important for interactive applications with variable-length inputs.

**Full FasterTransformer comparison across three benchmark specifications (Appendix D, Tables D.2–D.4):** The paper reproduces all three official FasterTransformer benchmarks and achieves the Pareto frontier (bold/underline annotation) in every scenario, not just a single cherry-picked configuration. The consistency across different input/output token ratios confirms that the partitioning strategy generalizes beyond a single workload.

**Memory capacity as a function of attention layout and batch size:** Table 1's maximum context lengths are computed with a 30% memory reservation for KV cache. The paper does not ablate this reservation percentage, which could affect the absolute numbers but not the relative ranking of layouts.

**Negative results and limitations acknowledged in the ablation context:**

- **Activation quantization not implemented:** "We have not implemented activation quantization..., but we are hopeful that it could reduce compute time in large-batch configurations and reduce communication volume of activations in weight-stationary layouts" (Section 3.6). This is a genuine gap—if activation quantization provides benefits comparable to weight quantization, the Pareto frontier could shift further.
- **Weight-gathered layouts require careful memory management:** "Some of the weight-gathered layouts would exhaust memory without these optimizations" (Section 3.5), referring to Looped CollectiveEinsum and tensor layout optimization. This means the weight-gathered results depend on low-level engineering that may not be portable to other frameworks.
- **Batch size below 4 is not reported** for FasterTransformer benchmarks because "our partitioning strategy partitions multiquery attention over batch and achieves no speedup for a batch size smaller than 4 (the minimum size of a TPU v4 torus axis)" (Appendix D). This is a hardware-specific constraint.

---

### Critical Assessment

#### Claim 1: The computational strategy achieves 29ms per token decode latency and 76% MFU prefill on PaLM 540B.

**What was demonstrated:** The 29ms figure (specifically 28.5ms in Section 4.4) is achieved with PaLM 540B on 64 TPU v4 chips, int8 weights, batch size 64, and decode phase only—assuming the context is already processed. The 76% MFU figure is achieved during prefill on the same hardware with bfloat16 weights, batch size 512 (1,048,576 tokens total), and the XYZ-weight-gathered layout. Both numbers are reported in Table 2 and Figure 1.

**What these numbers represent and what they don't:** The 29ms/token is a **decode-step latency**, not an end-to-end interactive latency. The full interactive scenario (Table 2, low-latency) involves 0.29s of prefill (batch 1, 2048 input tokens) plus 1.82s of decode (batch 64, 64 output tokens) for a total of 2.11 seconds—not 64 × 0.029 = 1.86 seconds, because the prefill cost is non-trivial. The paper does quote a 1.9-second end-to-end latency for a slightly different scenario in the introduction ("process 64 tokens of text from a user, consult a cached conversation history of 1920 tokens, and generate a 64-token response"), but this uses int8 weights and the specific batch mix described in Section 4.4. The 29ms figure is a clean per-step measurement, not the full user experience.

**The 76% MFU is on prefill only, at a massive batch size.** Batch size 512 with 2048-token sequences is 1,048,576 tokens in flight—this is an offline, throughput-oriented configuration that would never be used interactively. The decode MFU at similar batch sizes is only 33% (Table 2, high-throughput decode). The paper does not report an end-to-end MFU weighted by actual prefill/decode token splits in a real workload. If the workload has a 1:1 prefill-to-decode token ratio, the overall efficiency would be closer to the harmonic mean of 76% and 33% rather than 76%.

**The dependence on specific hardware is absolute.** All latency and MFU numbers are for TPU v4 with 275 TFLOPS/chip and 1,200 GB/s HBM bandwidth. They do not directly translate to other accelerators. The MFU normalization helps for cross-platform comparison (Section 5), but the absolute latency depends on total FLOPS (n_chips × 275 TFLOPS), and other platforms may have different compute/bandwidth ratios that shift the optimal partitioning strategy.

#### Claim 2: 2D weight-stationary significantly outperforms 1D weight-stationary at 64 chips.

**What was demonstrated:** Figure 6 shows both layouts on PaLM 540B during decode with batch 512, sweeping chip count. The 2D layout has lower latency at 64 chips. The analytical derivation in Appendix A.2.1 predicts the crossover at $n_{\text{chips}} > 16$, and the data are consistent with this.

**What is missing:** The paper does not provide a full sweep of batch sizes for this comparison. The analytical model predicts that the 2D advantage should depend on batch size (since the communication volume formulas contain $BL$), and at very small batch sizes (e.g., batch 1 during decode), the absolute communication volume might be so small that the 1D vs. 2D distinction is irrelevant—both would be memory-bandwidth-bound, not communication-bound. The paper's recommendation to use 2D for all decode scenarios may be overly broad if some very-low-batch-size cases don't benefit.

**The empirical evidence for the $n_{\text{chips}} > 16$ threshold is indirect.** Figure 6 shows only 64 chips with a clear 2D advantage, and the text mentions that "both layouts start to become communication-limited" but doesn't show the crossover region (around 16–32 chips) where 1D and 2D should be comparable. The threshold is derived analytically and validated only at 64 chips, not at a range of chip counts that bracket the predicted crossover.

#### Claim 3: Weight-gathered layouts become more efficient than weight-stationary at large batch sizes.

**What was demonstrated:** Figure 7 clearly shows this crossover at ~3 × 10⁵ tokens on 64 chips with sequence length 2048. The 76% MFU at 10⁶ tokens is an impressive demonstration of near-peak utilization. The analytical prediction from Appendix A.2.2 ($N_{\text{optimal}} = \sqrt{BL n_{\text{chips}} / F}$, crossover at $BL \approx F$) is consistent with the data.

**Genuine weaknesses:** First, the paper tests only three discrete weight-gathered variants (X, XY, XYZ) rather than a continuous sweep of $N$, because the variants must be compatible with the fixed 2D weight-stationary sharding. This means the empirically achieved MFU may not be the true optimum—there could be intermediate $N$ values (e.g., all-gathering over $XZ$ but not $Y$) that would perform better but are not realizable under the fixed weight layout constraint. Second, the paper does not explore whether the weight-gathered layouts' memory consumption from all-gathering weight copies could be problematic at intermediate batch sizes where the layout is selected dynamically—the text notes that "some of the weight-gathered layouts would exhaust memory without these optimizations," suggesting a precarious memory situation that isn't quantified. Third, the crossover point is validated only on 64 chips for one model with one sequence length. The formula $N_{\text{optimal}} = \sqrt{BL n_{\text{chips}} / F}$ predicts that the crossover $BL$ should scale as $F$ (model-dependent) and that the optimal $N$ should scale as $\sqrt{n_{\text{chips}}}$, but these dependencies are not ablated.

#### Claim 4: Batch-sharded multiquery attention enables 32× longer context lengths.

**What was demonstrated:** Table 1 shows a 32–64× improvement in maximum context length (43,000 vs. 1,320 at batch 128) for the optimized multiquery layout compared to multihead attention. The numbers are derived from memory capacity calculations with 30% HBM reserved for KV cache. Figure 8 shows that even at these extreme context lengths, attention takes only 8–31% of total runtime, so the approach is not just capacity-feasible but latency-feasible.

**What the numbers rely on:** The maximum context length numbers assume 30% of HBM is available for KV cache, which is a design choice the paper does not vary or justify. If other tensors (activations, temporary computation buffers, weight copies from weight-gathered layouts) consume more memory than expected, the usable KV cache fraction could be lower, reducing maximum context length. The paper also does not test inference quality at 43,000-token contexts—the ability to fit tokens in memory does not guarantee the model produces coherent outputs at that scale.

**The 32–64× claim depends on the multihead baseline's head dimension.** The paper deliberately set $d_{\text{head}} = 128$ for multihead to keep total parameter count constant with multiquery's $d_{\text{head}} = 256$. This halves the multihead KV cache size relative to what it would be with $d_{\text{head}} = 256$. If the multihead variant used the same head dimension, its max context length would be half of 1,320 = 660 tokens, making the optimized multiquery advantage 65× instead of 32×. The paper's choice is defensible (constant parameters), but the absolute improvement factor is sensitive to this design choice.

#### Claim 5: The implementation outperforms FasterTransformer in MFU-latency tradeoff.

**What was demonstrated:** Tables D.2–D.4 and Figure 9 show that the paper's PaLM implementation achieves the Pareto frontier (best MFU at each latency, or best latency at each MFU) across all three benchmark scenarios. The MFU normalization accounts for different peak FLOPS between TPU v4 and A100.

**The comparison is not perfectly controlled.** The paper uses 64 TPU v4 chips vs. FasterTransformer's 16–32 A100 GPUs. More chips provide more aggregate memory bandwidth and more total FLOPS, which improves latency independent of efficiency. The MFU metric controls for total FLOPS, but it doesn't control for memory bandwidth per FLOP—TPU v4 has 1,200 GB/s per 275 TFLOPS = 4.36 bytes/FLOP, while A100 has 2,039 GB/s per 312 TFLOPS = 6.54 bytes/FLOP. This means the A100 has proportionally more memory bandwidth, which should favor workloads that are memory-bandwidth-bound (like small-batch decode). The fact that TPU v4 achieves better MFU despite this disadvantage suggests the partitioning strategy is genuinely better, but it's not a perfectly clean comparison.

**The PaLM vs. Megatron model architectures differ** in ways that affect inference efficiency independent of the partitioning framework: PaLM has multiquery attention and parallel attention/FFN layers, while Megatron has multihead attention and serial layers. The paper acknowledges this (Table D.1) and provides both PaLM and Megatron results on their hardware. The Megatron implementation still outperforms FasterTransformer on the Pareto frontier, so the partitioning strategy is doing real work beyond the architectural advantages. But the head-to-head PaLM vs. Megatron comparison within the paper's framework (up to "10% MFU" advantage for PaLM) shows that architecture matters substantially.

#### The Missing Experiments That Would Have Strengthened the Paper

**Ablation of the 30% KV cache reservation fraction in Table 1.** The maximum context length is highly sensitive to this parameter, and practitioners need to know whether the 32× improvement holds at different memory budgets.

**Direct measurement of end-to-end interactive latency** for the chatbot scenario described in the introduction, including the batch-splitting between prefill and decode, to verify the "1.9 seconds" claim rather than relying on the separate prefill/decode timings in Table 2.

**A sweep of $n_{\text{chips}}$ for both 1D and 2D weight-stationary** to directly observe the crossover around 16 chips predicted by the analytical model, rather than testing only at 64 chips.

**Measurement of memory pressure during weight-gathered prefill** with the "some layouts would exhaust memory" claim quantified: what fraction of HBM is consumed by weight all-gather copies, and how much headroom remains for KV cache?

**A continuous sweep of the sequential-to-parallel ratio in weight gathering** rather than the three discrete variants, potentially by dynamically choosing $N$ per layer instead of uniformly across all layers. This could reveal whether the optimum is genuinely at one of the tested discrete points or somewhere in between.

**Ablation of the padded attention heads** from 48 to 64: what is the performance penalty of using the unpadded 48-head model on a 64-chip slice, where 16 chips would partially replicate heads? The paper claims the 3% MFU cost of padding is "more than recovered," but this comparison is not shown numerically.

**All experiments are on PaLM-family models.** While this provides internal consistency, it means the derived formulas ($F/E = 4$, the specific $n_{\text{chips}} > 16$ threshold for 2D) are validated only on one model architecture. Testing on a model with $F/E = 8$ (like some vision transformers) or $F/E = 2$ would probe whether the analytical model generalizes beyond the narrow architectural regime where it was calibrated.

Overall, the experimental section successfully validates the paper's core analytical claims—the superiority of 2D over 1D at scale, the batch-size-dependent crossover to weight-gathered layouts, the memory savings from batch-sharded multiquery attention—with clean, well-controlled comparisons. The results are internally consistent and mutually reinforcing. However, the experiments are narrower than they appear: one model family, one hardware platform, one 2048-token context length for most measurements, and discrete rather than continuous sweeps of the key optimization parameters. The Pareto frontier is real and impressive, but it's a demonstration that the approach works, not a proof that the analytical model is universally predictive across architectures and hardware. The paper's contribution is best understood as providing a **proven recipe for PaLM-scale Transformer inference on TPU v4**, backed by an analytical model that explains *why* the recipe works, rather than a fully-validated general scaling theory.

## 6. Limitations and Trade-offs

### Limitation 1: All Results Are Validated on a Single Model Family and Hardware Platform

**The assumption or constraint.** The paper's analytical framework and all empirical results are derived and tested exclusively on the PaLM family of models (8B, 62B, 540B) running on TPU v4 accelerators. The model architecture has specific properties that the partitioning strategies exploit: multiquery attention, parallel attention/feedforward layers, and a feedforward-to-model-dimension ratio $F/E = 4$. The hardware has a specific 3D torus topology with 275 TFLOPS/chip, 1,200 GB/s HBM bandwidth, and 270 GB/s interconnect bandwidth. The paper does not evaluate on any other model architecture (e.g., models with different $F/E$ ratios, multihead attention, or different layer counts) or any other hardware platform (GPUs, other TPU generations).

The authors acknowledge this scope limitation only indirectly. In Section 4, they state:

> "Our inference framework is based on JAX (Bradbury et al., 2018) and XLA (XLA, 2019), and our original high-level implementation was based on T5X (t5x, 2021). We use up to 256 TPU v4 chips (Google, 2022) for our benchmarks."

There is no explicit statement that results may not transfer to other hardware or architectures.

**The consequence.** The analytical formulas derived in Appendix A—particularly the 2D vs. 1D crossover threshold $\sqrt{n_{\text{chips}}} > F/E$, the weight-stationary vs. weight-gathered crossover $BL \approx F$, and the optimal weight-gathering degree $N = \sqrt{BL \cdot n_{\text{chips}} / F}$—are expressed in terms of general model parameters ($E$, $F$, $n_{\text{chips}}$) and hardware bandwidths. This suggests generality. However, the validation is entirely within one model family where $F = 4E$ and one hardware platform. A practitioner deploying a model with $F/E = 8$ (common in some vision transformers) or $F/E = 2$ on a GPU cluster with different interconnect topology cannot be confident that the formulas produce correct predictions, because the underlying cost model assumptions (e.g., the $(K-1)/K$ factor in all-gather cost, the effective network bandwidth model) may not transfer.

Specifically, the $n_{\text{chips}} > 16$ threshold for 2D weight-stationary depends on $F/E = 4$. For a model with $F/E = 2$, the threshold becomes $n_{\text{chips}} > 4$, meaning 2D would be preferred at much lower chip counts. For $F/E = 8$, the threshold is $n_{\text{chips}} > 64$, meaning 2D might never be preferred at realistic scales. These predictions are untested.

The FasterTransformer comparison (Section 5) provides some cross-platform evidence via MFU normalization, but it compares the paper's PaLM implementation (with multiquery attention and parallel layers) against FasterTransformer's Megatron implementation (with multihead attention and serial layers). The architectural differences are confounded with the partitioning strategy differences, so the comparison does not isolate whether the partitioning framework would be similarly effective on GPU hardware with a multihead model.

**What evidence exists in the paper.** The paper provides MFU-normalized comparisons with FasterTransformer on A100 GPUs (Figure 9, Tables D.2–D.4), but as noted, these compare different model architectures on different hardware. There is no ablation where the same model architecture is run on both platforms, or where the same hardware runs models with different $F/E$ ratios. Table D.1 explicitly lists the architectural differences between PaLM 540B and Megatron 530B (multiquery vs. multihead attention, parallel vs. serial FFN/attention, different $d_{\text{model}}$, $d_{ff}$, $n_{\text{heads}}$, $d_{\text{head}}$), but the paper does not quantify how much of the performance difference is attributable to partitioning vs. architecture.

**Mitigation status.** The paper frames the analytical model as generally applicable ("The proposed partitioning strategies generalize to many topologies, including single- and multi-node NVLink networks in GPU systems," Section 7), but provides no empirical evidence for this claim. The formulas are derived from first principles that are hardware-agnostic (all-gather and reduce-scatter cost models are standard in the HPC literature, as the paper notes by citing Chan et al., 2007), suggesting that the *form* of the results should transfer. But the *constants* (bandwidth ratios, the effective throughput of different collectives on different interconnects) and the *empirical validation* are entirely TPU v4-specific. A practitioner deploying on a different platform would need to independently validate the analytical predictions.

---

### Limitation 2: The "Low-Latency" Scenario Requires 64 Chips for a Single User Request, Making It Economically Impractical for Most Deployments

**The assumption or constraint.** The paper's headline low-latency results for PaLM 540B—28.5ms per token decode latency, 1.9 seconds end-to-end for a 64-token response (Section 1)—are achieved using **64 TPU v4 chips**. The paper does not explicitly highlight this as a cost concern, but the economic implication is significant: each interactive user request ties up 64 high-end accelerator chips for ~2 seconds. At the time of writing, TPU v4 chips are available on Google Cloud as "TPU v4 Pod slices," and while exact pricing varies, 64 chips represent a substantial allocation that would typically serve hundreds or thousands of concurrent users in a batched serving setup, not a single request.

The paper states (Section 4.4):

> "In the low-latency scenarios we combine batch-1 prefill with batch 32-to-64 decode: batch size 1 achieves best latency in the prefill phase, but for the generate phase we can increase the batch size up to 64 with negligible latency impact, and doing so is dramatically better for generate MFU."

This describes a configuration where batch-1 prefill processes a single user's input on all 64 chips, and batch-64 decode generates output (presumably by generating multiple candidate responses or batching across pipelined requests). The phrase "dramatically better for generate MFU" acknowledges the efficiency concern but does not address the underlying resource allocation problem: 64 chips serving 1–64 requests simultaneously still represents an extremely low utilization rate relative to the hardware's capacity.

**The consequence.** In practice, an interactive chatbot deployment would need to multiplex many concurrent user requests onto the same 64 chips to achieve reasonable cost per query. This requires either (a) batching requests from different users together, which adds latency (requests must wait for a batch to fill) and complexity (the batch-sharded attention layout assumes each chip's KV cache subset aligns with the current batch composition), or (b) accepting that each user request consumes an entire 64-chip slice for its duration, resulting in a cost per query that is prohibitive for all but the most latency-sensitive, cost-insensitive applications.

The paper's high-throughput configurations (Table 2, 76% MFU prefill, 33% MFU decode) use batch size 512 and achieve much better hardware utilization, but at latencies that are unacceptable for interactive use (85 seconds for prefill, 6 seconds for decode at batch 512—though these are amortized over 512 sequences, the per-sequence latency is still 85/512 ≈ 0.17s for prefill and 6/512 × 64 ≈ 0.75s for decode when running batched, which may be borderline for some interactive applications).

The fundamental tension is that the partitioning strategies optimized for low latency (many chips, small batches) are inherently inefficient in hardware utilization (14% MFU for decode in the low-latency configuration, Table 2), while the strategies optimized for efficiency (large batches, weight-gathered prefill) add latency. The Pareto frontier in Figure 1 maps this tradeoff but does not resolve it: moving left (lower latency) always means moving up (higher cost per token).

**What evidence exists in the paper.** Figure 1 (left) shows the decode cost vs. latency Pareto frontier. The lowest-latency point for PaLM 540B (batch 64, int8, 28.5ms/token) has a cost of approximately 64 chip-milliseconds per token. The lowest-cost point on the same curve (batch 512+, bfloat16) has a cost of approximately 4 chip-milliseconds per token, with latency around 100ms/token. This represents a ~16× cost increase to achieve ~3.5× lower latency. The paper reports these numbers implicitly through the plot but does not discuss the economic implications of operating at the low-latency end of the frontier.

Table 2 shows that the low-latency configuration achieves 43% MFU for prefill but only 14% MFU for decode—meaning the decode phase leaves 86% of the available compute idle. The chip-seconds per token cost metric accounts for this inefficiency, but the paper does not translate it into dollar terms or compare it to alternative approaches (e.g., using a smaller model with higher utilization, or using distillation to serve a smaller model with equivalent quality).

**Mitigation status.** The paper does not address the economic cost of its low-latency configurations directly. It provides the cost metric (chip-seconds per token) as an abstract quantity and notes that it is "directly proportional to operational cost" (Section 4.4), but does not discuss whether the resulting costs are practical. The suggestion to pipeline batch-1 prefill into batch-64 decode (Section 4.4) is a partial mitigation—it allows some batching during the decode phase—but still requires dedicating 64 chips to serving at most 64 concurrent requests, which is far below the batch sizes needed for efficient operation (batch 512+ for the high-throughput configuration). The paper does not explore alternative deployment architectures such as using fewer chips with higher utilization, or mixing model sizes (e.g., routing easy requests to a smaller model on fewer chips).

---

### Limitation 3: The Difficulty of Adapting to Dynamic Batching and Variable-Length Sequences in Production Serving Is Not Addressed

**The assumption or constraint.** The paper's partitioning framework assumes static, known-in-advance batch sizes and sequence lengths. The analytical cost models take $B$ and $L$ as fixed parameters, and the experiments sweep over predetermined ($B$, $L$) combinations. However, in a real production serving system, requests arrive asynchronously with variable-length inputs and generate variable-length outputs. A dynamic batching system must continuously form batches from waiting requests, and the batch composition changes over time as requests complete and new ones arrive.

Specifically, the paper's two key partitioning innovations—phase-aware layout switching (weight-gathered for prefill, weight-stationary for decode) and batch-sharded multiquery attention—both rely on knowing the batch size to configure the communication pattern. The weight-gathered layout's optimal all-gather degree $N = \sqrt{BL n_{\text{chips}} / F}$ depends on the total token count $BL$. The batch-sharded attention layout partitions the KV cache over batch, meaning each chip stores keys and values for a fixed subset of batch indices. If the batch composition changes (requests complete, new requests join), the KV cache sharding must be updated, potentially requiring costly data movement between chips.

The paper acknowledges the batch-mixing scenario only in passing (Section 4.4):

> "This mixture of batch sizes is possible in practice either by generating multiple samples from the same input text, or by pipelining a batch-1 prefill server into a batch-64 decoding server."

This describes a static pipeline where prefill and decode use different batch sizes, but does not address the general case of dynamic batching across many concurrent users with independent arrival and completion times.

**The consequence.** In a production serving system, the inability to handle dynamic batching efficiently would force one of two suboptimal choices:

1. **Static batching with fixed batch sizes**, where the system waits until enough requests accumulate to fill a batch (adding queuing latency) and all requests in a batch must have similar sequence lengths (or padding wastes compute on shorter sequences). This is the approach implicitly assumed in the paper's experiments (fixed batch size, fixed sequence length), but it adds significant latency for low-traffic periods and reduces throughput when sequence lengths vary widely.

2. **Continuous batching with repartitioning overhead**, where the batch composition changes incrementally but the KV cache must be periodically resharded to maintain the batch-sharded attention layout. The paper provides no analysis of this overhead—the all-to-all collectives used in Figure 5(b) are configured for a specific batch partitioning, and changing the batch-to-chip mapping mid-generation would require additional communication whose cost is not modeled.

The paper's recommendation to use different partitioning layouts for prefill and decode (Section 4.1: "during the prefill phase, we select from weight-stationary and weight-gathered layouts based on the current number of tokens in the batch. During the generate phase, we select the 2D weight-stationary layout") further complicates dynamic serving: the system must switch layouts between phases for the same request, but in a continuous batching system, a single chip may be simultaneously processing prefill for one batch and decode for another (a technique called "iteration-level scheduling" or "inflight batching" used in systems like vLLM and Orca). The paper's framework does not address how to partition when different requests are in different phases.

**What evidence exists in the paper.** None. The paper provides no experiments on dynamic batching, no analysis of KV cache resharding overhead, and no discussion of how the framework would integrate with a production serving system that handles concurrent asynchronous requests. All experiments use static, predetermined batch sizes and sequence lengths. This is a standard simplification for systems benchmarking papers, but it is particularly consequential here because the partitioning strategy is explicitly batch-dependent—unlike simpler data-parallel schemes where batch size changes affect only compute efficiency, not the communication pattern or memory layout.

**Mitigation status.** Not addressed. The paper does not discuss dynamic batching, continuous batching, or the overhead of reconfiguring partitioning layouts at serving time. The suggestion to pipeline batch-1 prefill into batch-64 decode is the closest the paper comes to addressing serving architecture, but this is a static pipeline, not a general solution to dynamic batching. A practitioner attempting to deploy these techniques in a production serving system would need to independently solve the dynamic batching problem, and the solution could negate some of the paper's reported efficiency gains.

---

### Limitation 4: The Framework Requires Models with Multiquery Attention and Parallel Transformer Blocks to Achieve Its Best Results, Limiting Applicability to Models That Use These Specific Architectural Choices

**The assumption or constraint.** The paper's two most impactful results—the 32-64× context length extension (Table 1) and the 14% decode latency reduction from parallel layers (Section 4.3)—depend on specific architectural features of the PaLM model family. Multiquery attention (Shazeer, 2019) is the enabler for the batch-sharded attention layout, and the parallel Transformer block formulation (Wang and Komatsuzaki, 2021) is the enabler for halved communication in weight-stationary layouts.

The paper states these dependencies explicitly. For multiquery attention (Section 3.3):

> "Multiquery attention has lower memory cost to load the KV cache when sharded over batch."

For parallel layers (Section 3.4):

> "The benefits from the parallel formulation are as follows... it also eliminates one of the two all-reduce operations in each Transformer layer."

However, the paper does not provide guidance for models that lack these features. Many widely-deployed large models—including GPT-3 (Brown et al., 2020), LLaMA (Touvron et al., 2023), and most open-source models—use multihead attention and serial Transformer blocks. For such models, the paper's framework would need to fall back to the multihead attention layout (head-sharded, Figure 4a) and the serial block formulation, losing the specific advantages that drive the most impressive headline numbers.

**The consequence.** The paper's results are not directly transferable to a large fraction of existing deployed models. Specifically:

- **Without multiquery attention:** The KV cache size is $n_{\text{heads}}$ times larger (48× for PaLM's head count). Table 1 shows that multihead attention on 64 chips supports only 1,320 tokens of context at batch 128, vs. 43,000 for optimized multiquery. The 76% MFU prefill result (Figure 7) depends on fitting large batches in memory, which would be impossible with a 48× larger KV cache. The batch-sharded attention layout (Section 3.3) cannot be applied to multihead attention because the heads can be partitioned independently without replication, making head-sharding the natural choice.

- **Without parallel layers:** The 14% decode latency penalty from serial blocks (Section 4.3) would apply, and the benefit of halved all-reduces from fusion (Section 3.4) would be lost. For decode latency-sensitive applications, this 14% is a significant fraction of the total optimization headroom.

A practitioner deploying a model without these architectural features would need to adapt the partitioning framework, and the resulting performance would be substantially worse than the paper's headline numbers. The paper provides no characterization of how much worse, beyond the multihead attention comparison in Table 1 (which shows a 32× reduction in max context length) and the serial block comparison in Section 4.3 (14% decode latency increase). The impact on the Pareto frontier (Figure 1) and the FasterTransformer comparison (Section 5) is not shown for models lacking these features.

**What evidence exists in the paper.** The paper provides several ablations that partially quantify the impact:

- Table 1 shows multihead attention's max context length (1,320 for batch 128) vs. optimized multiquery (43,000), a 32× reduction.
- Section 4.3 reports 14% higher decode latency for serial vs. parallel blocks on PaLM 540B with 64 chips and batch 512.
- Table D.1 documents the architectural differences between PaLM 540B and Megatron 530B, and the FasterTransformer comparison (Section 5) includes Megatron results, showing that the Megatron implementation achieves lower MFU than PaLM (e.g., 39% vs. 44% at batch 1024 in Table D.2). However, this conflates the attention type, block formulation, and other architectural differences.

The paper does not provide a systematic ablation that isolates the effect of each architectural feature on the overall Pareto frontier. For example, it does not show a "PaLM 540B with multihead attention and serial blocks" variant and compare its end-to-end latency/cost tradeoff against the optimized variant.

**Mitigation status.** The paper acknowledges the dependence on PaLM's architecture only indirectly, by noting that the model was selected because it "incorporates the techniques of multiquery attention and parallel attention and feedforward layers" (Section 4, Methodology). There is no discussion of how to adapt the framework for models without these features, and the paper's claim that "the proposed partitioning strategies generalize to many topologies, including single- and multi-node NVLink networks in GPU systems" (Section 7) refers to hardware topologies, not model architectures. The omission of a multihead-attention, serial-block variant in the main Pareto frontier results (Figure 1) means practitioners cannot estimate the performance penalty they would incur when deploying a standard model architecture.

---

### Limitation 5: The Headline 76% MFU Prefill Result and 1.9-Second Chatbot Latency Are Achieved Under Mutually Exclusive Hardware Configurations, and No Single Configuration Excels at Both

**The assumption or constraint.** The paper presents a set of impressive performance numbers—76% MFU prefill, 29ms per token decode latency, 1.9-second chatbot response time, 32× longer context lengths—that a casual reader might assume are achieved simultaneously by a single well-optimized system. In fact, these numbers represent different operating points on the Pareto frontier with different hardware configurations, batch sizes, weight formats, and partitioning layouts.

Table 2 reveals the incompatibility:

| Scenario | Prefill Layout | Decode Layout | Weights | Batch (Prefill/Decode) | Prefill MFU | Decode MFU |
|---|---|---|---|---|---|---|
| Low-latency | WS 2D | WS 2D | int8 | 1 / 64 | 43% | 14% |
| High-throughput | WG XYZ | WS 2D | bfloat16 | 512 / 512 | 76% | 33% |

The 76% MFU number comes from the high-throughput configuration with batch 512 and bfloat16 weights. The 29ms per token number comes from the low-latency configuration with batch 64 and int8 weights. These configurations differ in batch size (8×), weight format (int8 vs. bfloat16), and prefill layout (WS 2D vs. WG XYZ). A system cannot simultaneously achieve 76% MFU and 29ms decode latency—a user request is served by one configuration or the other, not both.

The paper acknowledges this by presenting the Pareto frontier (Figure 1), which explicitly shows the tradeoff. But the introduction and conclusion highlight the extreme points without emphasizing their incompatibility. The introduction states:

> "For a state-of-the-art 540B parameter dense model running on 64 TPU v4 chips, we achieve a low-batch-size latency of 29ms per token during generation (with int8 weight quantization) and a 76% MFU during large-batch-size processing of input tokens while supporting a large context length of 2048 tokens."

The "and" in this sentence could be read as implying simultaneous achievement, though technically it lists two separate results.

**The consequence.** A practitioner designing a serving system must choose where on the Pareto frontier to operate, and the choice has large consequences for both latency and cost. The paper does not provide guidance on how to make this choice for a specific application profile (e.g., a chatbot that expects 100 queries per second with a 2-second latency target). The two example configurations in Table 2 (low-latency and high-throughput) represent extremes, but most real applications fall somewhere in between and would need to interpolate.

More importantly, the phase-specific layout switching that the paper advocates—using weight-gathered for prefill and weight-stationary for decode—is only applied in the high-throughput configuration (Table 2: WG XYZ for prefill, WS 2D for decode). The low-latency configuration uses WS 2D for both phases. This means the phase-aware strategy, which the paper presents as a key insight, is not used in the configuration that achieves the best interactive latency. The reason is that weight-gathered layouts are inefficient at small batch sizes (Figure 7 shows they achieve only ~20% MFU at 125,000 tokens per batch, while the low-latency prefill uses batch 1 × 2048 = 2,048 tokens—far below the crossover). The phase-aware insight is most valuable for throughput, not latency.

**What evidence exists in the paper.** Tables 2 and 3 explicitly show the different configurations for low-latency and high-throughput scenarios, confirming they are distinct operating points. Figure 1 shows the full frontier, making the tradeoff visually clear. The tension is implicit in the MFU numbers: the decode phase achieves only 14% MFU in the low-latency configuration vs. 33% in high-throughput, while prefill achieves 43% vs. 76%. These large gaps indicate that the same hardware cannot be simultaneously optimal for both phases.

The paper does not report what performance a single configuration would achieve if forced to handle both interactive and batch workloads. For instance, if the high-throughput configuration (WS 2D for decode, WG XYZ for prefill, batch 512) were used for a single interactive request (batch 1), what would the latency be? The weight-gathered prefill would be grossly inefficient at batch 1 (the all-gather of weights would dominate), and the batch-512-tuned partitioning might perform worse than a dedicated low-latency configuration. This cross-configuration performance is not measured.

**Mitigation status.** The paper does not address how to build a single serving system that handles a mixture of latency-sensitive and throughput-oriented requests. The suggestion to pipeline "a batch-1 prefill server into a batch-64 decoding server" (Section 4.4) implies two separate hardware pools, but this doubles the total hardware requirement and introduces complexity in routing requests between pools. The paper does not explore whether a single configuration can be "good enough" for both use cases, or whether adaptive batching and layout switching can be done dynamically at serving time to track the workload mix.

---

### Limitation 6: The Communication Cost Model Approximates Effective Network Bandwidth as Constant and Ignores Contention Effects That Emerge at Scale

**The assumption or constraint.** The analytical framework in Section 3 and Appendix A derives communication times using a simplified cost model: for an all-gather or reduce-scatter over $K$ partitions with per-chip data size $D$, the communication time is $T = D / \text{bandwidth} \times (K-1)/K \approx D / \text{bandwidth}$. This model assumes that the full interconnect bandwidth (270 GB/s per chip for TPU v4) is available for each collective operation, independent of how many collectives are running concurrently or how they contend for links in the 3D torus.

The paper acknowledges this simplification implicitly by stating (Appendix A.1):

> "This is a general cost model that holds true for most real-world network topologies (Chan et al., 2007), not just the TPU's torus topology."

And notes the $(K-1)/K$ approximation:

> "In most formulas, we will disregard the $(K-1)/K$ term, approximating it as 1 under the assumption $K \gg 1$, in order to simplify the algebra."

However, the model does not account for link contention when multiple collectives share the same physical torus links, or for the bandwidth degradation that occurs when communication patterns create hotspots. In the 2D weight-stationary layout, for instance, there are separate reduce-scatter operations over the $Y$/$Z$ axes and the $X$ axis (Figure 2b). If these operations overlap or contend for the same links, the effective bandwidth per operation may be lower than the 270 GB/s assumed.

The Looped CollectiveEinsum optimization (Section 3.5) explicitly overlaps communication with computation, which reduces *exposed* communication time but increases the potential for network contention since multiple operations share the interconnect simultaneously. The paper does not model this contention analytically.

**The consequence.** The analytical predictions of optimal torus axis splits ($X$, $Y$, $Z$) and optimal weight-gathering degree ($N$) may be inaccurate at large chip counts where contention effects become significant. The formulas assume that communication cost is simply the sum of individual collective costs, with no penalty for concurrency. In practice, running multiple all-gathers or reduce-scatters on overlapping torus axes will experience throughput degradation that the model does not capture.

This could explain several observations in the paper that are not fully accounted for by the analytical model:

- **The 2D weight-stationary advantage over 1D at 64 chips** (Figure 6) is predicted analytically from the $O(1/\sqrt{n_{\text{chips}}})$ vs. $O(1)$ scaling, but the actual measured latency difference may be influenced by contention effects that the formulas don't distinguish.

- **The performance ceiling at 64-way parallelism** (Figure 9 discussion: FasterTransformer achieves only 33% MFU at 32-way, the paper's system achieves 44% MFU at 64-way) may be partly due to contention rather than the idealized per-collective bandwidth assumptions.

- **The Looped CollectiveEinsum benefit of ~1.4×** (Section 3.5) is measured empirically rather than predicted analytically, suggesting that the interaction between computation, communication, and contention is complex enough that empirical tuning is needed on top of the analytical model.

**What evidence exists in the paper.** The paper does not directly measure network contention or compare predicted vs. actual communication times for individual collective operations. The empirical validation of the analytical model is indirect: the paper shows that 2D outperforms 1D at 64 chips (Figure 6), that weight-gathered outperforms weight-stationary at large batch sizes (Figure 7), and that the Pareto frontier is achieved using the strategy the analytical model recommends. But these are end-to-end latency and MFU measurements that confound compute, memory, and communication effects. There is no microbenchmark isolating communication time to validate the $D / \text{bandwidth}$ formula.

The paper notes (Section 3.5) that the Looped CollectiveEinsum implementation involves "explicitly match[ing] up communication collectives with the matrix multiplies that they should be fused with, to maximize the potential for overlap." This suggests that achieving good overlap requires careful manual scheduling—the analytical model doesn't capture whether a given collective can be fully overlapped, partially overlapped, or not overlapped at all. The 1.4× improvement over compiler-generated scheduling indicates that the gap between analytical prediction and achievable performance is substantial enough to require hand-tuning.

**Mitigation status.** The paper does not attempt to model contention or validate the per-collective bandwidth assumption. The analytical model is presented as a *design tool* for selecting between layout families (1D vs. 2D, weight-stationary vs. weight-gathered) rather than a *performance prediction* tool that accurately forecasts latency. The paper's empirical results (Figures 6, 7) validate that the model makes correct *relative* choices (which layout is better), but do not validate its *absolute* predictions (exactly how much better). A practitioner relying on the formulas to predict latency for a new hardware platform or model architecture would need to calibrate the effective bandwidth constants empirically, since the theoretical 270 GB/s may not be achievable under realistic contention patterns.

The paper's approach to the gap between analytical model and reality is pragmatic: use the analytical model to choose the strategy family, then rely on low-level optimizations (Looped CollectiveEinsum, async collectives, memory layout tuning) to extract the predicted performance. This works for the paper's specific setting (PaLM on TPU v4) but means that reproducing the results on other platforms would require both adapting the analytical model *and* redoing the low-level tuning, without clear guidance on how much of the performance comes from each.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper introduces a **systematic analytical methodology** for partitioning Transformer inference workloads that transforms model parallelism from a profiling-and-heuristics art into a predictable engineering discipline. Before this work, practitioners faced a combinatorial explosion of possible partitioning layouts—which tensor dimensions to split over which hardware axes, whether to keep weights or activations stationary, how to trade prefill throughput against decode latency—and the dominant approach was exhaustive search (Zheng et al., 2022; Xu et al., 2021) or adherence to a single strategy proven to work during training (Megatron-LM's 1D weight-stationary). The paper demonstrates that the optimal choices are not arbitrary but are **determined by a small set of analytically tractable quantities**: the model's feedforward-to-embedding ratio $F/E$, the total number of tokens in flight $BL$, the chip count $n_{\text{chips}}$, and the hardware bandwidth ratios.

This is a **conceptual reframing** of the inference partitioning problem. The paper's specific formulas—the 2D vs. 1D crossover condition $\sqrt{n_{\text{chips}}} > F/E$, the weight-stationary vs. weight-gathered crossover $BL \approx F$, the optimal all-gather degree $N = \sqrt{BL \cdot n_{\text{chips}} / F}$—are the surface-level contribution. The deeper contribution is the demonstration that these formulas *exist and are useful*: that you can sit down with a model spec sheet and a hardware topology, compute a few ratios, and predict which partitioning strategy will minimize communication without running a single profiling experiment. This changes partitioning from a *search problem* (requiring many trial runs and opaque optimization) into a *design problem* (requiring analytical reasoning), in the same way that Chinchilla scaling laws (Hoffmann et al., 2022) changed pretraining resource allocation from trial-and-error to formula-driven optimization. The paper is explicit about this positioning (Section 1): "enables the user to intuitively understand the tradeoffs and select the best multi-axis tensor partitioning strategy... in contrast to a black-box exhaustive search."

The paper also introduces a **phase-aware serving architecture** that recognizes prefill and decode as fundamentally different workloads requiring different partitioning strategies. This idea—that the optimal layout for processing 1,048,576 tokens simultaneously (prefill at batch 512, context 2048) is qualitatively different from the optimal layout for processing 512 tokens (decode at batch 512, one token per sequence)—is obvious in retrospect but was not articulated or exploited in prior inference systems. FasterTransformer, the dominant GPU inference framework, uses a single partition configuration for both phases. Training-adapted systems (Megatron, GSPMD) never had to make this distinction because training always operates in "prefill mode" (all tokens known, processed in parallel). The paper's finding that you can switch layouts between phases by keeping weights in a fixed `ExFyz` sharding and changing only the communication pattern (weight-stationary for decode, weight-gathered for prefill, Section 3.2.3) makes this insight **practically actionable** rather than merely conceptual. The 76% MFU prefill with 33% MFU decode in the high-throughput configuration (Table 2) is only possible because the system uses different strategies for each phase; a single-strategy approach would either sacrifice prefill throughput (if using weight-stationary throughout) or sacrifice decode latency (if using weight-gathered throughout).

The paper also **reconciles an apparent contradiction** between the promise and practice of multiquery attention. Multiquery attention (Shazeer, 2019) was proposed as a way to reduce KV cache memory by sharing key/value heads across query heads—a 48× reduction for PaLM's head count. Yet without the paper's batch-sharded attention layout, deploying multiquery attention naively (partitioning over heads, replicating the single K/V head) actually produces *worse* memory capacity than multihead attention (Table 1: 660 tokens max context for baseline multiquery vs. 1,320 for multihead at batch 128). The paper identifies the mechanism behind this paradox (head-sharding forces K/V replication, nullifying the architectural savings) and provides the solution (batch-sharding with an all-to-all collective on the small per-step Q/K/V tensors). This resolves why some practitioners may have observed disappointing memory savings from multiquery attention: the architectural benefit only materializes with the correct partitioning, and the naive approach is actively harmful. The 32–64× context length improvement from the optimized layout (43,000 vs. 1,320 tokens) demonstrates that the interaction between architecture and partitioning strategy is not additive but multiplicative—the right partitioning unlocks the architecture's latent potential.

Finally, the paper **redirects attention from compute to communication as the central inference bottleneck**. The dominant narrative in large model deployment has emphasized memory capacity (models don't fit on a single chip) and compute throughput (need enough FLOPS for acceptable latency). The paper shows that once you've solved the capacity problem by using enough chips, and once you've achieved adequate compute throughput through batching, the binding constraint becomes **inter-chip communication bandwidth**—and critically, this bottleneck behaves differently across partitioning strategies. The 1D layout's communication time is constant with chip count; the 2D layout's scales as $O(1/\sqrt{n_{\text{chips}}})$; the weight-gathered layout's scales as $O(\sqrt{BL}/\sqrt{n_{\text{chips}}})$. The paper's empirical demonstration that FasterTransformer's 32-way tensor parallelism achieves only 33% MFU (vs. 46% at 16-way) due to a "communication bottleneck of scaling tensor parallelism beyond this point" (Section 5) while the paper's 2D strategy achieves 44% MFU at 64-way parallelism provides concrete evidence that communication scaling, not compute throughput, is the limiting factor for large-model inference. This insight directs future research toward communication-reducing techniques (sparsity, mixture-of-experts, better interconnect topologies) rather than simply throwing more compute at the problem.

The work also makes certain research directions **less attractive**. The paper provides strong evidence that **pipeline parallelism is the wrong tool for low-latency inference**. Adding pipeline stages introduces serial dependencies between chips that fundamentally limit latency reduction, since each chip must wait for the previous stage to complete before processing its layers. The paper's exclusive use of tensor parallelism (all chips process all layers simultaneously) and its achievement of 29ms per token latency demonstrate that tensor parallelism alone, with the right multi-dimensional partitioning, can hit interactive latency targets at 500B+ scale. This suggests that research effort invested in optimizing pipeline parallelism for inference (e.g., micro-batching schemes to hide pipeline bubbles) may be better spent on improving tensor parallelism scalability and communication efficiency.

### Follow-Up Research This Work Enables

**Adaptive difficulty-aware partitioning that selects layouts per-request based on predicted sequence length and available batch size.** The paper's framework assumes static, known-in-advance batch sizes and sequence lengths. A dynamic serving system in production receives requests with variable-length inputs, variable numbers of output tokens, and asynchronous arrival times. The analytical formulas could be extended to select partitioning strategies dynamically: estimate the expected total tokens $BL$ for the upcoming forward pass (based on current queue depth and average sequence lengths in the batch), compute the crossover threshold $BL \approx F$, and switch between weight-stationary and weight-gathered layouts mid-serving. A concrete experiment would deploy the paper's 64-chip PaLM 540B configuration with a production workload trace (e.g., LMSYS-Chat-1M conversation logs with real prompt and generation length distributions) and measure end-to-end throughput and tail latency when the system adapts partitioning per-batch versus using a static configuration. The key metric would be whether the overhead of layout switching (reconfiguring collectives, potential KV cache reshuffling for batch-sharded attention) is amortized by the efficiency gains of using the right layout for each batch's token count.

**Joint optimization of architecture and partitioning for inference efficiency, not just training FLOPs.** The paper demonstrates that architectural choices made during model design—multiquery vs. multihead attention, parallel vs. serial transformer blocks, the $F/E$ ratio, the number of attention heads—have large, quantifiable effects on inference partitioning efficiency that are not visible during training. A natural follow-up would treat these architectural hyperparameters as part of the inference optimization problem: given a target inference latency, context length, and hardware budget, what model architecture minimizes total cost per token while maintaining quality? The paper's analytical formulas provide the inference-cost side of this tradeoff. A concrete study would start with a base architecture (e.g., a standard GPT-3-style model with multihead attention, serial blocks, $F/E=4$), enumerate architectural modifications (switch to multiquery, switch to parallel blocks, vary $F/E$ from 2 to 8, pad heads for divisibility as the paper does), train each variant to equal validation loss, and benchmark inference performance using the paper's partitioning framework. The result would be a Pareto frontier over architectural choices rather than just partitioning choices, with predictions like "multiquery attention is worth the quality cost if target context length exceeds 1024 tokens" or "increasing $F/E$ beyond 4 improves training efficiency but hurts inference latency at small batch sizes."

**Communication-optimal partitioning for heterogeneous hardware topologies with asymmetric bandwidth.** The paper's analytical model treats all torus axes as having equal bandwidth (270 GB/s on TPU v4) and derives optimal axis splits ($X$, $Y$, $Z$) that balance communication volume across dimensions. GPU clusters, however, often have hierarchical interconnects: intra-node NVLink at 900 GB/s, inter-node InfiniBand or RoCE at 200–400 GB/s. This bandwidth asymmetry changes the optimal partitioning: dimensions that communicate over the faster intra-node links should be larger than those that cross nodes. A concrete extension would re-derive the paper's optimization (Appendix A.2.1) with separate bandwidth parameters $B_x$, $B_y$, $B_z$ for each torus axis, yielding $X \propto \sqrt{B_x}$, $Y \propto \sqrt{B_y}$, $Z \propto \sqrt{B_z}$ for the 2D weight-stationary case. The prediction would be tested on a GPU cluster with 8×A100 nodes (NVLink within node, InfiniBand across nodes): models partitioned with more chips along the fast NVLink axis should outperform uniform partitioning, and the analytical model should predict the magnitude of the advantage. This would directly validate the paper's claim (Section 7) that "the proposed partitioning strategies generalize to many topologies, including single- and multi-node NVLink networks in GPU systems"—a claim currently unsubstantiated by any GPU experiments.

**Looped CollectiveEinsum-aware analytical cost models that predict overlap feasibility from tensor shapes.** The paper empirically finds that overlapping communication with computation via Looped CollectiveEinsum provides roughly 1.4× improvement over compiler-generated scheduling (Section 3.5), and that the choice of which dimension to reduce-scatter into ($E$ vs. $B$) affects overlap opportunities. But the analytical model in Appendix A treats communication and computation as additive ($T_{\text{total}} = T_{\text{comp}} + T_{\text{comm}}$) rather than modeling the max-overlap regime ($T_{\text{total}} = \max(T_{\text{comp}}, T_{\text{comm}})$). A precise analytical model would predict, given tensor shapes and hardware tile sizes, whether a specific collective can be fully overlapped with a specific matmul—this depends on whether the tiling granularity of the communication matches the tiling granularity of the computation. A research contribution would derive the conditions under which overlap is feasible, construct a modified cost model that switches between additive and max-of formulations accordingly, and validate it by instrumenting the paper's implementation to measure exposed (non-overlapped) communication time versus total communication time for each collective in each layout. This would close the gap between the paper's analytical predictions and the 1.4× improvement achieved through manual scheduling, making the framework more predictive and reducing the need for per-platform tuning.

**Verifier-guided or learned cost models that predict inference latency on unseen hardware from a small number of profiling runs.** The paper's analytical model requires hardware bandwidth parameters (HBM bandwidth, interconnect bandwidth) that must be measured or obtained from datasheets. In practice, effective bandwidth depends on contention, collective implementation quality, and driver overhead, and varies across platforms. A follow-up could train a lightweight performance model (e.g., a small neural network or gradient-boosted tree) that takes as input the paper's analytically-derived features ($E/X$, $F/YZ$, $BL$, $N$) plus measured bandwidths from a few profiling runs, and predicts end-to-end latency for a new layout or batch size. The paper's framework provides the feature engineering; the learned model would correct for the systematic errors in the analytical approximation (the $(K-1)/K \to 1$ simplification, the ignored contention effects discussed in Limitation 6). A strong evaluation would measure how many profiling runs are needed to achieve prediction error below 10% on held-out (batch size, chip count) configurations, and whether the analytical features significantly outperform generic features like total FLOPs or total communication volume in a pure learned model—validating that the paper's analytical decomposition is genuinely informative, not just descriptive.

**Stress-testing the analytical model on models with extreme aspect ratios ($F/E \ll 1$ or $F/E \gg 4$) to establish its domain of validity.** The paper validates its formulas only on PaLM-family models where $F/E = 4$ exactly. The analytical predictions—that 2D becomes preferable when $\sqrt{n_{\text{chips}}} > F/E$, and that weight-gathered becomes preferable when $BL > F$—imply that for models with very small $F/E$ (e.g., some vision transformers where $F/E = 1$ or $2$), 2D weight-stationary should be preferred at much lower chip counts and weight-gathered should become optimal at much smaller batch sizes. Conversely, for models with very large $F/E$ (e.g., some MoE architectures where $F = 8E$ or $16E$ for the expert FFNs), weight-gathered layouts might never be optimal because the $F$ term in the communication cost makes weight all-gathering prohibitively expensive. A concrete experiment would construct synthetic transformer models with controlled $F/E$ ratios (2, 4, 6, 8, 12), train them to equivalent quality, and benchmark with the paper's framework to test whether the analytically-predicted crossover points match empirical measurements. This is a "negative result" experiment as much as a validation experiment: it would establish the boundaries beyond which the paper's formulas break down (e.g., due to attention becoming the bottleneck at small $F/E$, or due to weight matrix loading dominating at large $F/E$).

### Practical Applications and Downstream Use Cases

**Cost-efficient serving of conversational AI assistants at scale.** The paper's low-latency configuration (Table 2: 1.9 seconds end-to-end for processing 64 input tokens with 1920 tokens of conversation history and generating 64 output tokens on PaLM 540B with int8 weights) provides a concrete recipe for deploying large language models as chatbots. A production system serving hundreds of concurrent users could use the phase-aware pipelining strategy the paper suggests (Section 4.4): a pool of 64-chip slices runs batch-1 prefill for incoming user messages (0.29 seconds each, achieving 43% MFU despite batch size 1 because the weight-stationary layout keeps communication manageable), then aggregates requests into batch-64 groups for the decode phase (1.82 seconds for 64 tokens, 14% MFU). With 64 chips, this pipeline can process approximately 64 requests every 2.11 seconds (prefill and decode pipelined in parallel), for a throughput of roughly 30 requests per second. The cost per request, using the paper's cost metric of approximately 64 chip-milliseconds per token at the low-latency operating point and ~128 tokens per request (64 input + 64 output), is roughly 8.2 chip-seconds per request. On TPU v4 pricing, this translates to a cost per query that is directly computable from cloud pricing and enables organizations to forecast serving costs before deployment. The key practical insight is that the paper's analytical framework lets you recompute these numbers for different model sizes (62B vs. 540B), context lengths, and latency targets without running new benchmarks—you adjust $E$, $F$, $n_{\text{chips}}$, $B$, $L$ in the formulas and the Pareto frontier shifts predictably.

**Long-document processing and retrieval-augmented generation with extended context windows.** Table 1 shows that the optimized multiquery attention layout supports 43,000-token contexts at batch 128—long enough to ingest entire books, legal contracts, or code repositories in a single inference pass. This enables applications that were previously impractical due to KV cache memory exhaustion. A concrete deployment scenario: a legal document analysis system that takes a 30,000-token contract as input (prefill phase, processed with the sharded-over-heads attention layout since the memory load is amortized) and generates a 500-token summary (decode phase, using the batch-sharded multiquery layout to keep the 30,000-token KV cache within per-chip memory). The paper's numbers suggest this would fit comfortably on 64 TPU v4 chips with multiquery attention—the KV cache per chip at batch 1 would be $30,000 \times 256 \times 2 \text{ bytes} \approx 15.4$ MB per layer, times 118 layers ≈ 1.8 GB per chip total, well within the 9.6 GB KV cache budget (30% of 32 GiB). The prefill latency for 30,000 tokens at batch 1 with 64 chips using WS 2D layout would scale roughly linearly from the paper's 0.29 seconds for 2048 tokens—approximately 4.2 seconds—and the decode latency at batch 1 (~29ms per token for 500 tokens ≈ 14.5 seconds) would be dominated by the feedforward layers, with attention contributing only 8–31% of runtime per Figure 8. This represents a practical capability that the paper's numbers make predictable without additional experimentation.

**Offline batch inference pipelines for knowledge distillation and synthetic data generation.** The paper's high-throughput configuration (Table 2: 76% MFU prefill, 33% MFU decode at batch 512) is directly applicable to batch processing workloads where latency is irrelevant and minimizing cost per token is the primary goal. A concrete scenario: distilling a 540B teacher model into a smaller student model by generating 100 million tokens of synthetic training data. With the high-throughput configuration, prefill processes 512 × 2048 = 1,048,576 tokens per forward pass at 76% MFU—nearly saturating the 64 TPU v4 chips' compute capacity. At 64 chips × 275 TFLOPS × 0.76 MFU = 13,376 effective TFLOPS for prefill, and 33% MFU for decode, the paper's cost metric of approximately 4 chip-milliseconds per token (from the right end of the PaLM 540B decode curve in Figure 1 left) translates to about 4 chip-seconds per 1,000 tokens. For 100 million tokens, this is roughly 400,000 chip-seconds ≈ 111 chip-hours. The analytical framework allows a practitioner to recompute this for different model sizes: generating 100 million tokens with the 62B model (Table 3, high-throughput) would use fewer chips (8–32) at slightly lower MFU (37% decode), and the cost would scale roughly with parameter count. This makes the paper's framework a **cost estimation tool** for large-scale LLM data generation projects, where the economics of using a 540B teacher vs. a 62B teacher can be evaluated analytically without running generation benchmarks.

**Deployment configuration planning for new hardware platforms via analytical extrapolation.** A hardware vendor or cloud provider evaluating a next-generation accelerator (e.g., a hypothetical TPU v5 with 500 TFLOPS/chip, 2,000 GB/s HBM, 400 GB/s interconnect) can use the paper's analytical model to predict inference performance for large Transformer models before silicon exists. The key parameters that enter the formulas are peak FLOPS per chip, HBM bandwidth per chip, and interconnect bandwidth per chip. Plugging in the new numbers yields predictions for: (a) the crossover chip count where 2D weight-stationary becomes preferable ($n_{\text{chips}} > F/E$, unchanged since it depends only on model architecture), (b) the crossover batch size where weight-gathered becomes preferable ($BL \approx F$, also unchanged), and (c) the absolute latency and cost numbers (scaling inversely with bandwidth and FLOPS improvements). This allows hardware designers to assess whether interconnect bandwidth improvements or compute FLOPS improvements provide greater inference performance returns—the paper's finding that communication is the primary bottleneck at large chip counts (Section 5 discussion) suggests that interconnect bandwidth should be prioritized over peak FLOPS for inference-optimized accelerators. Similarly, the formulas predict how much memory capacity is needed to support target context lengths with multiquery attention (Table 1 methodology), informing HBM capacity requirements for long-context inference. This application transforms the paper from a retrospective benchmark into a **forward-looking design tool** for hardware-software co-optimization.

### When to Prefer This Method

The paper does not propose a single method against specific named alternatives; rather, it provides a **framework for selecting among partitioning strategies** based on operating conditions. The framework itself implies decision rules, which the paper articulates explicitly:

- **Prefer 2D weight-stationary over 1D weight-stationary when $\sqrt{n_{\text{chips}}} > F/E$** (Section 3.2.2). For standard Transformers with $F/E = 4$, this means $n_{\text{chips}} > 16$. Below this threshold, the simpler 1D layout suffices; above it, 2D's $O(1/\sqrt{n_{\text{chips}}})$ communication scaling dominates.

- **Prefer weight-gathered layouts over weight-stationary when $BL > F$** (Section 3.2.3, Appendix A.2.2). For PaLM 540B with $F = 73728$, this crossover is at roughly 74,000 tokens in flight. During decode ($BL = B \times 1$, always small), always use weight-stationary. During prefill, use weight-stationary for small batches and switch to weight-gathered when the token count exceeds the threshold.

- **Prefer batch-sharded multiquery attention over head-sharded multiquery attention whenever context length exceeds memory capacity under head-sharding** (Section 3.3, Table 1). The head-sharded layout replicates K/V to all chips, nullifying multiquery's memory savings. The batch-sharded layout recovers the savings at the cost of an all-to-all on the small per-step Q/K/V tensors.

- **Prefer parallel attention/FFN blocks over serial blocks for any latency-constrained deployment** (Section 3.4, Section 4.3). The parallel formulation halves per-layer communication and provides a 14% decode latency reduction on PaLM 540B, with no quality cost (the model was trained with the parallel formulation).

- **Prefer int8 weight quantization for low-batch-size, latency-constrained decode; prefer bfloat16 for high-throughput, large-batch prefill** (Section 3.6, Figure 1). Int8 halves weight-loading memory time, providing ~2× cost improvement when weight loading dominates (small batches). At large batches where compute dominates, int8 provides negligible benefit and bfloat16 is simpler.

These rules are all grounded in the paper's analytical cost model and empirically validated on PaLM-family models. A practitioner can apply them directly when deploying a large Transformer on a torus-connected accelerator mesh (TPU v4 or similar topology), using the paper's formulas to compute the relevant thresholds for their specific model dimensions and hardware bandwidths.
