# Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

**ArXiv:** [2104.04473](https://arxiv.org/abs/2104.04473)

## 🎯 Pitch

This paper introduces PTD-P, a unified scheme that combines pipeline, tensor, and data parallelism to train trillion-parameter language models efficiently across thousands of GPUs. Through a novel interleaved pipeline schedule and specialized communication and computation optimizations, the approach achieves unprecedented throughput—up to 502 petaFLOP/s (52% of peak)—enabling models at the trillion-parameter scale to be trained in a matter of months rather than years. This breakthrough dramatically reduces resource barriers for frontier NLP research and sets a new standard for large-scale model training practicality and efficiency.

---

## 1. Executive Summary

This paper introduces **PTD-P** (inter-node pipeline model parallelism, intra-node tensor model parallelism, and data parallelism), a composed parallelism strategy that scales large language model training to thousands of GPUs, and proposes a novel **interleaved pipelining schedule** that reduces pipeline bubble idle time (e.g., by splitting each device's assigned layers into multiple model chunks interleaved across microbatches). Implemented in Megatron-LM and evaluated on GPT models ranging from 1 billion to 1 trillion parameters on up to 3072 NVIDIA A100 GPUs, the system achieves 502 petaFLOP/s aggregate throughput—52% of theoretical peak per GPU—enabling end-to-end training of a trillion-parameter model in approximately three months. The paper establishes guiding principles for composing parallelism dimensions, demonstrating that tensor parallelism should be confined within a multi-GPU server while pipeline parallelism scales across servers, and that the optimal microbatch size arises from a tradeoff between arithmetic intensity and pipeline bubble size, with throughput varying by up to 2× across suboptimal parallel configurations.

## 2. Context and Motivation

### The Core Problem: Training Large Language Models Is Bottlenecked by Two Hard Constraints

The fundamental challenge this paper tackles is deceptively simple: **how do you train a model that is too large to fit on any single GPU, while keeping training time practical?** Prior to this work, practitioners faced an uncomfortable tradeoff. You could either train smaller, less capable models that fit within hardware limits, or you could wait impractically long — potentially decades or centuries — training large models sequentially on a single device. Neither option was acceptable as models grew exponentially.

This problem emerged from a concrete historical trend the paper documents in Figure 1. Between 2018 and 2021, state-of-the-art NLP models grew from ELMo at 94 million parameters, through BERT-Large at 340 million, GPT-2 at 1.5 billion, Megatron-LM at 8.3 billion, Turing-NLG at 17.2 billion, to GPT-3 at 175 billion parameters — an increase of roughly three orders of magnitude in three years. This exponential growth is not accidental: larger language models consistently demonstrate improved capabilities as zero-shot and few-shot learners (Brown et al., 2020; Raffel et al., 2019), enabling downstream applications like summarization, dialogue generation, semantic search, and code autocompletion.

However, this trend collides with two hard constraints:

**Constraint 1: GPU memory capacity.** Modern GPUs have finite memory. NVIDIA's A100 GPU with 80 GB of HBM2e memory represents a practical ceiling at the time of this paper's publication, and even this cannot hold models with tens or hundreds of billions of parameters. The authors quantify the severity: a 175-billion-parameter model like GPT-3 requires approximately 350 GB of memory in 16-bit precision just for the parameters alone, plus additional memory for optimizer states (which typically double or triple this figure depending on whether mixed-precision Adam is used), activations, and gradients. No single GPU comes close.

**Constraint 2: Training time.** Even if memory were infinite, the sheer number of floating-point operations makes sequential training impractical. The paper provides a concrete estimate that crystallizes the urgency: training GPT-3 with 175 billion parameters on a single V100 GPU would take **approximately 288 years**. Training the trillion-parameter model explored in this paper would take substantially longer still. For these models to be practical — for research teams to iterate on architectures and for organizations to deploy them — training must complete in months, not centuries. This requires parallelism at scale.

These two constraints are coupled in practice. Even if you solve the memory problem through some form of parameter offloading (swapping between GPU and CPU memory, as explored in ZeRO-Offload; Ren et al., 2021), the training time constraint remains. Conversely, if you have infinite GPUs for parallelism, each individual GPU still faces memory limits on what fraction of the model it can hold.

### The Inadequacy of Data Parallelism Alone

The most widely used parallelism strategy in deep learning is data parallelism: each GPU holds a complete copy of the model, the input data is sharded across GPUs, and gradients are aggregated (typically via an all-reduce operation) after each forward-backward pass. Data parallelism works well when models fit on individual GPUs and when batch sizes are large enough to keep all GPUs utilized. However, the paper identifies two specific failure modes for large language models:

**Failure mode A: Memory limits.** A model that does not fit on a single GPU cannot be trained with pure data parallelism, period. The full model, including parameters, gradients, optimizer states, and intermediate activations, must reside in GPU memory. For models exceeding 20-30 billion parameters in 16-bit precision, even 80 GB A100 GPUs are insufficient. This is not a theoretical limit — the paper's experiments on models up to 1 trillion parameters would be impossible under pure data parallelism.

**Failure mode B: Batch size limits.** Even when models *do* fit (say, a 20-billion-parameter model on an 80 GB GPU), data parallelism faces a scaling ceiling. The number of GPUs that can be productively used for data-parallel training equals the batch size (each GPU processes one microbatch). GPT-3 was trained with a batch size of 1,536 sequences, meaning data parallelism can use at most 1,536 GPUs before each GPU processes less than one sequence. However, as the paper notes, roughly 10,000 GPUs were used to train GPT-3 in a reasonable amount of time — far exceeding the data-parallel limit. Beyond this point, per-GPU batch sizes shrink below 1, forcing either an increase in total batch size (which can harm convergence; Goyal et al., 2017) or a decrease in GPU utilization due to smaller, less compute-efficient matrix multiplications.

Furthermore, even when memory and batch size permit, data parallelism introduces communication overhead. At extreme scales (thousands of GPUs), the all-reduce operation that aggregates gradients across all workers becomes a bandwidth bottleneck, and the communication time can dominate computation time if not carefully managed.

### Prior Model Parallelism Approaches and Their Gaps

Several forms of model parallelism — where the model's layers or operations are partitioned across GPUs — emerged to address these limitations. But the paper argues that existing approaches, used in isolation or naively combined, suffer from critical scaling problems.

#### Tensor Model Parallelism (Megatron)

Tensor (intra-layer) model parallelism, as introduced by Shoeybi et al. (2020) in Megatron, partitions individual transformer layers across multiple GPUs. In a transformer block, the self-attention multi-head projections (Q, K, V) and the MLP weight matrices are split column-wise or row-wise, so each GPU computes a portion of the layer's output. Two all-reduce operations per transformer layer (one each in forward and backward passes) synchronize partial results.

This approach works effectively **within a single multi-GPU server**, where GPUs are interconnected via high-bandwidth NVLink (600 GB/s bidirectional per link on A100, up to 4.8 TB/s aggregate with NVSwitch). In this regime, the all-reduce communication is fast enough relative to computation that tensor parallelism achieves good scaling.

However, the paper identifies two problems when tensor parallelism is pushed beyond a single server:

**Problem 1: Cross-server communication bottlenecks.** When tensor-parallel ranks span multiple servers, the all-reduce operations must traverse inter-server network links (InfiniBand or Ethernet), which are substantially slower than NVLink — typically 200 Gbps per link compared to NVLink's 600 GB/s. The paper quantifies this later in Section 5.4, showing that tensor parallelism across nodes leads to communication-dominated execution and up to 2× lower throughput compared to confining it within a node.

**Problem 2: Sub-linear GPU utilization.** As tensor-parallel size increases, each GPU's share of the computation shrinks. For example, with a tensor-parallel size of 16, each GPU computes only 1/16 of each matrix multiplication. For layer dimensions that are already modest, these shards become too small to saturate a modern GPU's compute units. The GPU becomes launch-overhead-bound or memory-bandwidth-bound rather than compute-bound, reducing per-GPU throughput. The paper's experiments in Section 5.4.3 (Figure 15) demonstrate this empirically: increasing tensor-parallel size from 2 to 32 on a 5.9-billion-parameter model causes throughput to drop substantially due to both the communication overhead and the diminished arithmetic intensity of the smaller GEMM shards.

#### Pipeline Model Parallelism (GPipe, PipeDream)

Pipeline parallelism takes a different approach: instead of splitting individual layers, it partitions the model's *layers* across GPUs. GPU 1 handles layers 1–4, GPU 2 handles layers 5–8, and so on. A batch is divided into microbatches, which flow through this pipeline — while GPU 1 processes microbatch 1, GPU 2 can simultaneously process a different microbatch. This reduces the number of idle GPUs and improves utilization.

The paper builds specifically on two pipeline scheduling schemes from prior work:

**GPipe (Huang et al., 2019)** uses an "all-forward, all-backward" schedule (Figure 3 in the paper): first, all microbatches flow forward through the entire pipeline; then, all backward passes execute in reverse order. This schedule is simple but has two drawbacks. First, it requires activations from *all* microbatches to be stored in memory simultaneously, since the backward pass for microbatch 1 cannot begin until microbatches 2 through *m* have completed their forward passes. With *m* microbatches (which might number in the hundreds or thousands), activation memory explodes. Second, GPUs at the pipeline boundaries (the first and last stages) experience substantial idle time — the so-called **pipeline bubble** — while waiting for the pipeline to fill and drain. The paper quantifies this bubble as $(p-1)/m$ of total computation time, where $p$ is the number of pipeline stages (GPUs) and $m$ is the number of microbatches. When $m$ is small relative to $p$, as much as 50% of time can be spent idle.

**PipeDream-Flush (Narayanan et al., 2021)** addresses the memory problem with a 1F1B (one forward, one backward) schedule. After a warm-up phase that fills the pipeline, each GPU alternates between processing one forward pass and one backward pass. This limits in-flight activations to at most $p$ microbatches (rather than $m$ with GPipe), dramatically reducing memory footprint. However, the pipeline bubble size remains $(p-1)/m$ — the *time* spent flushing is unchanged. The paper adopts this 1F1B schedule as its "default" pipeline schedule.

#### Why Pipeline Parallelism Alone Falls Short

Pipeline parallelism in isolation has its own limits:

**Limited to one GPU per pipeline stage.** A pure pipeline-parallel setup can use at most as many GPUs as the model has layers. For a 100-layer transformer, that means at most 100 GPUs — far fewer than the 3,072 GPUs the paper scales to. Some layers can be grouped together to use fewer GPUs, but this cannot scale beyond the layer count. Furthermore, when each pipeline stage is just one layer, the computation per stage is small, leading to poor GPU utilization (the inter-layer communication dominates).

**Large pipeline bubbles at modest batch sizes.** The bubble fraction $(p-1)/m$ means that to achieve high efficiency (small bubble), the number of microbatches $m$ must be substantially larger than the number of pipeline stages $p$. With a fixed global batch size $B$ and microbatch size $b$, $m = B/(b \cdot d)$, where $d$ is data-parallel size. If the batch size is constrained (for convergence reasons or because the model is simply too large to allow more microbatches per GPU), the pipeline bubble can consume a significant fraction of total time. The paper demonstrates this empirically in Figure 11, where a batch size of 8 leads to noticeably worse scaling at larger pipeline-parallel sizes compared to a batch size of 128.

**Cross-server bandwidth underutilization.** When pipeline stages span servers, the default communication pattern sends tensors point-to-point between consecutive pipeline stages. However, modern multi-GPU servers like the DGX A100 have 8 InfiniBand (IB) networking cards per node, and a naive implementation sends the same tensor redundantly over multiple IB links (since tensor parallelism replicates communications across ranks; see Figure 9a). The aggregate inter-server bandwidth is underutilized.

### Why Existing Systems Don't Scale to Trillion-Parameter Models

The paper makes an explicit claim that prior systems cannot train models at the scale it targets:

> "past systems [29, 40] cannot train such large models since they do not combine pipeline and tensor parallelism."

This statement deserves unpacking. Megatron (Shoeybi et al., 2020) used only tensor parallelism and achieved good performance for models up to ~20 billion parameters on an 8-GPU server. PipeDream (Narayanan et al., 2019) used only pipeline parallelism and demonstrated scaling on image classification models (which have fundamentally different dimension characteristics from large language models). Neither system alone could handle a 175-billion-parameter model across thousands of GPUs.

DeepSpeed (Rajbhandari et al., 2019, 2021) introduced ZeRO (Zero Redundancy Optimizer), which shards optimizer states, gradients, and parameters across data-parallel workers. This allows training larger models without model parallelism by eliminating redundant storage of optimizer states across data-parallel replicas. However, the paper's comparison to ZeRO-3 (Section 5.2, Figure 10, Table 2) shows that pure ZeRO-3 without model parallelism scales poorly for very large models across many nodes. At 1,536 GPUs for a 175-billion-parameter model, PTD-P achieves 141 teraFLOP/s per GPU versus ZeRO-3's 44 teraFLOP/s — a 3.2× gap. The paper attributes this to ZeRO-3's additional cross-node communication for fetching parameter shards before each computation, which becomes bandwidth-limited at scale. (The paper acknowledges that ZeRO-3 can be combined with model parallelism, but this combination is not evaluated.)

### The Gap: No Systematic Understanding of How to Compose Parallelism Dimensions

Beyond the specific limitations of individual techniques, the paper identifies a deeper conceptual gap: **the field lacked a principled understanding of how tensor, pipeline, and data parallelism interact when composed.** Practitioners faced a combinatorial space of choices: for a given model size, batch size, and GPU count, how many of the available GPUs should be allocated to tensor parallelism versus pipeline parallelism versus data parallelism? Each choice affects:

- **Communication volume and pattern:** Tensor parallelism uses all-reduce operations (bandwidth-hungry), pipeline parallelism uses point-to-point sends/receives (cheaper but serial), and data parallelism uses all-reduce once per batch (infrequent but potentially large).
- **Pipeline bubble size:** Determined by the ratio of pipeline stages to microbatches, which depends on how the GPUs are partitioned.
- **GPU compute efficiency:** Affected by microbatch size (which determines GEMM dimensions), tensor-parallel shard size (smaller shards = lower utilization), and the number of layers per pipeline stage.
- **Memory footprint:** Each strategy has different requirements for storing parameters, gradients, optimizer states, and activations.

These interactions are non-trivial and can lead to pathological outcomes. As the paper states:

> "sub-optimal combinations of tensor and pipeline model parallelism can lead to up to 2× lower throughput, even with high-bandwidth network links between servers"

Prior work had addressed these dimensions independently or in limited combinations:
- Huang et al. (2019) studied pipeline parallelism alone.
- Shoeybi et al. (2020) studied tensor parallelism alone.
- Jia et al. (2018) and Tarnawski et al. (2020) explored automatic partitioning of computation graphs for combinations of data and model parallelism, but did not consider pipeline parallelism or the impact of hyperparameters like microbatch size and activation recomputation on models that exceed GPU memory capacity.
- Narayanan et al. (2019) combined pipeline and data parallelism but did not consider tensor parallelism.
- Rajbhandari et al. (2019, 2021) focused on sharded data parallelism without explicit model parallelism.

No prior work had systematically addressed the **three-way interaction** between tensor, pipeline, and data parallelism for models at the scale of hundreds of billions to trillions of parameters, nor provided practical heuristics for configuring these dimensions.

### How This Paper Positions Itself

The paper frames its contribution not as proposing a new paradigm, but as **providing the first practical recipe for composing existing parallelism techniques to scale to unprecedented model sizes while achieving practical training throughput.** Its approach is explicitly engineering-driven:

> "we do not automatically explore the search space of parallelism strategies... but instead suggest heuristics (in §3) that we found work well in practice."

This is a deliberate positioning. Rather than claiming to solve the general optimization problem (which prior work like FlexFlow and PipeDream attempted with varying success), the paper offers concrete, validated rules of thumb that practitioners can apply immediately: use tensor parallelism within a server, pipeline parallelism across servers, and data parallelism to fill out the remaining GPUs. The theoretical analysis in Section 3 grounds these heuristics in quantitative models, but the paper's primary contribution is demonstrating that these heuristics actually work — and work at scales (3,072 GPUs, 502 petaFLOP/s) that prior systems could not achieve.

The paper's positioning also includes a specific stance on optimizer semantics: unlike PipeDream-2BW and PipeMare, which relax strict synchronous weight updates to reduce pipeline bubbles, PTD-P **preserves strict optimizer semantics exactly** by retaining pipeline flushes. The authors acknowledge that asynchronous approaches offer potential throughput gains but defer consideration "to future work," focusing instead on demonstrating that even with flushes, practical training times are achievable through careful engineering.

In summary, the paper addresses a gap that is simultaneously practical (how do I train a trillion-parameter model without waiting centuries?), engineering-oriented (how do I compose parallelism strategies without the interactions destroying throughput?), and situated within a specific hardware context (DGX A100 servers with NVLink intra-node and InfiniBand inter-node). It does not claim to be the first to propose any individual technique, but rather the first to demonstrate their effective composition at a scale that makes previously impossible training runs possible.

## 3. Technical Approach

### 3.1 Reader Orientation

The paper builds a **distributed training system** that composes three parallelism techniques—tensor model parallelism, pipeline model parallelism, and data parallelism—so that a language model with up to a trillion parameters can be trained across thousands of GPUs in a matter of months rather than centuries. The solution is not a new parallelism primitive but rather a principled recipe for how to allocate GPUs among these three dimensions, backed by analytical models that predict pipeline bubble size, communication volume, and compute efficiency as functions of the parallel configuration.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major components interacting at each training iteration:

1. **The model sharding layout** — a mapping from model layers and operations to physical GPUs, determined by three integer parameters (`$p$`, `$t$`, `$d$`) that specify how many GPUs are allocated to pipeline, tensor, and data parallelism respectively. This layout is fixed at training start and determines which GPU holds which parameters.

2. **The pipeline scheduler** — a runtime mechanism that orchestrates the order of forward and backward passes across microbatches flowing through the pipeline stages. It implements either the default 1F1B schedule or the novel interleaved 1F1B schedule, injecting microbatches at the input stage, tracking which microbatches are in flight, and triggering the pipeline flush at batch boundaries.

3. **The communication layer** — NCCL-based collective and point-to-point operations that move tensors between GPUs. This layer is optimized by the scatter/gather communication trick that exploits the fact that tensor-parallel replicas send identical data between pipeline stages, allowing a split-and-reassemble pattern that better utilizes multi-rail InfiniBand.

4. **The computation graph** — the actual PyTorch forward and backward computation, optimized via custom fused kernels (bias+GeLU, bias+dropout+add, scale+mask+softmax), a non-standard data layout (`[s, b, a, h]` instead of `[b, s, a, h]`) to enable strided batched GEMM, and optional activation recomputation that trades extra forward passes for lower memory footprint.

Information flows through these components in a fixed rhythm: the data loader produces a global batch → the batch is split into `$m$` microbatches → the pipeline scheduler dispatches microbatches according to the chosen schedule → each GPU performs its assigned computation (forward or backward) on its shard of the model layers → inter-GPU communication (all-reduce for tensor parallelism within nodes, point-to-point for pipeline parallelism across nodes) synchronizes intermediate activations and gradients → at the batch boundary, the pipeline flushes, optimizer steps synchronize, and data-parallel all-reduce aggregates gradients.

### 3.3 Roadmap for the Deep Dive

- **First**, the parallel configuration space and the notation that anchors the entire analysis (Section 3.1 in the paper), because every subsequent model and design choice is expressed in these terms.
- **Second**, the analytical model for the tensor-vs-pipeline tradeoff (Section 3.2), since this is the core intellectual contribution that justifies the "tensor within a server, pipeline across servers" heuristic.
- **Third**, the analytical model for data-vs-model-parallelism interactions (Section 3.3), because it explains why data parallelism is the preferred scaling-out mechanism once memory constraints are satisfied.
- **Fourth**, the microbatch size analysis (Section 3.4) and activation recomputation (Section 3.5), since these hyperparameters materially affect both throughput and memory, and their optimal values depend on the parallel configuration.
- **Fifth**, the pipeline scheduling algorithms (Section 2.2) and interleaved schedule (Section 2.2.2), because the pipeline bubble is the primary efficiency limiter and the interleaved schedule is one of the paper's two novel mechanisms.
- **Sixth**, the tensor model parallelism partitioning (Section 2.3), since it defines the computational unit that is replicated and pipelined.
- **Seventh**, the implementation-level optimizations (Section 4): scatter/gather communication, fused kernels, and data layout, because these are the engineering details that convert the analytical models into actual throughput.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems engineering paper** whose core idea is that tensor, pipeline, and data parallelism can be composed according to a small set of heuristics — tensor within a node, pipeline across nodes, data to scale out — to achieve practical training times on models up to a trillion parameters, provided that careful attention is paid to the pipeline schedule, the microbatch size, and communication optimizations.

---

#### The Parallel Configuration Space and Notation

Every parallel training setup is described by a triple `$(p, t, d)$` of positive integers, where `$p$` is the pipeline-model-parallel size (number of sequential stages in the pipeline), `$t$` is the tensor-model-parallel size (number of GPUs that split each layer), and `$d$` is the data-parallel size (number of replicas that process different data shards). These must satisfy the constraint `$p \cdot t \cdot d = n$`, where `$n$` is the total number of GPUs used.

The global batch size `$B$` is a user-specified input (chosen based on convergence properties, not hardware constraints). Each pipeline stage processes the batch in smaller units called microbatches, each of size `$b$`. The number of microbatches per pipeline per batch is `$m = \frac{1}{b} \cdot \frac{B}{d}$` — that is, the global batch divided by the microbatch size and further divided among data-parallel replicas.

**Why this decomposition matters.** The triple `$(p, t, d)$` is not arbitrary. Different partitions of the same total GPU count `$n$` produce radically different communication patterns, pipeline bubble sizes, and per-GPU compute loads. For example, with `$n = 64$`, one could choose `$(p=1, t=8, d=8)$` (pure tensor+data), `$(p=8, t=8, d=1)$` (tensor+pipeline, no data parallelism), or `$(p=64, t=1, d=1)$` (pure pipeline). Each choice has different performance characteristics, and the paper's analytical models are designed to predict which choice is best for a given model size and batch size.

---

#### Analytical Model: Tensor vs. Pipeline Model Parallelism

The paper develops a quantitative model for the tradeoff between tensor and pipeline parallelism when data parallelism is held constant (`$d=1$`). Under this condition, `$t \cdot p = n$`, so `$p = n/t$`.

**Pipeline bubble size as a function of `$t$`.** The pipeline bubble — the fraction of time GPUs spend idle waiting for the pipeline to fill and drain — is:

$$\text{Bubble fraction} = \frac{p - 1}{m} = \frac{n/t - 1}{m}$$

where `$p$` is the number of pipeline stages, `$m$` is the number of microbatches per batch, `$n$` is the total number of GPUs, and `$t$` is the tensor-parallel size.

**What it computes:** the fraction of ideal computation time wasted on pipeline flushes. For example, with `$n=64$`, `$t=1$` (so `$p=64$`), and `$m=128$`, the bubble fraction is `$(64-1)/128 = 0.492$`, meaning nearly half the time is idle. With `$t=8$` (so `$p=8$`), the bubble fraction drops to `$(8-1)/128 = 0.055$`.

**Why this form:** the `$p-1$` term counts the number of pipeline stages that are idle during the fill and drain phases — one less than the total because the first stage fills immediately and the last stage drains last. Dividing by `$m$` normalizes by the total number of microbatches that amortize this idle time. As `$t$` increases (more tensor parallelism, fewer pipeline stages), the numerator shrinks, reducing the bubble. This creates a pressure toward larger `$t$`.

**Communication volume tradeoff.** The counter-pressure against large `$t$` is communication cost. Pipeline parallelism uses point-to-point communication between consecutive stages. For each microbatch, the total amount of data sent between a pair of consecutive devices (in either the forward or backward direction) is:

$$\text{Pipeline comm per microbatch} = b s h$$

where `$b$` is the microbatch size, `$s$` is the sequence length, and `$h$` is the hidden size. This communication is point-to-point and can be overlapped with computation (since the next microbatch can be processed while the previous one is being sent).

Tensor parallelism, by contrast, uses all-reduce communication. For each microbatch, each transformer layer requires two all-reduce operations in the forward pass and two in the backward pass (see Section 2.3), each involving tensors of size `$b s h$`. The total communication per layer per device per microbatch for tensor parallelism is:

$$\text{Tensor comm per layer per microbatch} = 8 b s h \left( \frac{t - 1}{t} \right)$$

where the factor `$8$` comes from 2 forward all-reduces + 2 backward all-reduces (each all-reduce being counted as `$2 \cdot (t-1)/t$` times the data size, since each GPU sends and receives `$(t-1)/t$` of the total). For a pipeline stage with `$l_{\text{stage}}$` layers, the total tensor-parallel communication per device per microbatch is:

$$\text{Tensor comm per microbatch} = l_{\text{stage}} \cdot 8 b s h \left( \frac{t - 1}{t} \right)$$

**What these formulas reveal.** Pipeline communication scales with `$b s h$` regardless of `$t$` or the number of layers per stage. Tensor communication scales with `$l_{\text{stage}} \cdot b s h \cdot (t-1)/t$`. For large `$t$`, the factor `$(t-1)/t$` approaches 1, meaning the communication is roughly proportional to the number of layers per stage. Since `$l_{\text{stage}}$` is typically tens to hundreds, tensor communication is substantially larger than pipeline communication. Moreover, all-reduce is a blocking collective that is harder to overlap with computation than point-to-point sends/receives.

**The key qualitative difference.** Pipeline communication traverses slower inter-node links but is relatively small and peer-to-peer. Tensor communication is larger but can be confined to fast intra-node NVLink if `$t$` is kept within a single server's GPU count. This leads to the paper's first heuristic:

> **Takeaway #1:** When considering different forms of model parallelism, tensor model parallelism should generally be used up to degree `$g$` when using `$g$`-GPU servers, and then pipeline model parallelism can be used to scale up to larger models across servers.

The rationale: set `$t$` to the number of GPUs per server (8 for DGX A100), keeping all tensor-parallel all-reduce communication within the high-bandwidth NVLink/NVSwitch domain. Then use pipeline parallelism to stitch together multiple such tensor-parallel groups across servers, with the cheaper point-to-point communication flowing over InfiniBand.

**Empirical validation (Section 5.4.1, Figure 13).** The paper tests a 162.2-billion-parameter GPT model on 64 A100 GPUs, sweeping `$(p, t)$` pairs that multiply to 64: `$(2, 32)$`, `$(4, 16)$`, `$(8, 8)$`, `$(16, 4)$`, and `$(32, 2)$`. The `$(8, 8)$` configuration — tensor-parallel size exactly matching the 8 GPUs per DGX A100 server — achieves the highest throughput, confirming that tiling the tensor-parallel group to the server boundary is optimal.

---

#### Analytical Model: Data and Model Parallelism Interactions

The paper next analyzes how data parallelism interacts with each type of model parallelism.

**Pipeline + data parallelism.** With tensor parallelism fixed (`$t=1$`), we have `$p = n/d$` (all GPUs not used for data parallelism become pipeline stages). The number of microbatches per pipeline is `$m = B/(d \cdot b) = b'/d$`, where the paper defines `$b' \equiv B/b$` as the ratio of global batch size to microbatch size. The pipeline bubble size becomes:

$$\text{Bubble fraction} = \frac{p - 1}{m} = \frac{n/d - 1}{b'/d} = \frac{n - d}{b'}$$

**What it computes:** the fraction of time idle due to pipeline flushes, now expressed directly in terms of `$d$` (data-parallel size). As `$d$` increases (more data parallelism), `$n - d$` decreases, so the pipeline bubble shrinks. In the extreme where `$d \to n$`, the bubble approaches zero — but this is only possible if the model fits on a single GPU.

**Why this form matters.** It cleanly separates the batch-size effect (`$b'$` in the denominator) from the data-parallelism effect (`$n-d$` in the numerator). Doubling the batch size halves the bubble fraction. Doubling the data-parallel size (up to half the GPUs) also reduces the bubble. Both levers are available to the practitioner.

**Data + tensor parallelism.** With tensor parallelism, all-reduce communication happens for every microbatch (since each forward and backward pass involves tensor-parallel collectives). Data-parallel all-reduce, by contrast, happens only once per batch (after all microbatches complete). This frequency difference — per-microbatch for tensor parallelism, per-batch for data parallelism — means that data-parallel communication is inherently less burdensome at scale, especially when the number of microbatches per batch is large.

This leads to the paper's second heuristic:

> **Takeaway #2:** When using data and model parallelism, a total model-parallel size of `$M = t \cdot p$` should be used so that the model's parameters and intermediate metadata fit in GPU memory; data parallelism can be used to scale up training to more GPUs.

The logic: use just enough model parallelism to make the model fit (satisfying the memory constraint), then allocate all remaining GPUs to data parallelism. Model parallelism beyond the minimum necessary introduces communication overhead without a proportional benefit.

**Empirical validation (Section 5.4.2, Figure 14; Section 5.4.3, Figure 15).** The paper tests a 5.9-billion-parameter GPT model on 64 GPUs, sweeping various `$(p, d)$` and `$(t, d)$` configurations. For pipeline+data (Figure 14), throughput decreases monotonically as pipeline-parallel size increases, confirming that pipeline parallelism is a necessary evil for memory, not a throughput booster. For tensor+data (Figure 15), throughput drops sharply as tensor-parallel size increases, especially with larger batch sizes, because the per-microbatch all-reduce in tensor parallelism becomes a communication bottleneck when many microbatches execute.

**The data parallelism ceiling.** The paper notes an important practical constraint: "data parallelism supports parallelization to only 1536 GPUs" for GPT-3's batch size of 1536, yet roughly 10,000 GPUs were used to train GPT-3 in reasonable time. This is because data parallelism cannot use more GPUs than the batch size (each GPU needs at least one sequence to process). When `$B$` is constrained by convergence considerations, this ceiling is hard. Model parallelism is the only way to exceed it.

---

#### Analytical Model: Microbatch Size Optimization

The microbatch size `$b$` is a hyperparameter that affects both computation and communication. The paper models total batch processing time (ignoring communication overlap) as:

$$T_{\text{batch}} = \left( \frac{b'}{b} + p - 1 \right) \cdot (t_f(b) + t_b(b))$$

where `$b' \equiv B/d$` is the global batch size divided by data-parallel size, `$b$` is the microbatch size, `$p$` is the number of pipeline stages, `$t_f(b)$` is the forward computation time for one microbatch (a function of `$b$`), and `$t_b(b)$` is the backward computation time for one microbatch.

**What it computes:** the wall-clock time to process one full batch through the pipeline. The first factor `$(b'/b + p - 1)$` is the number of microbatch-time-units in the schedule: `$b'/b$` microbatch slots for the actual computation, plus `$p-1$` slots of pipeline bubble. Multiplying by the per-microbatch time `$t_f(b) + t_b(b)$` gives total time.

**Why the microbatch size creates a tension.** As `$b$` increases:

- **The `$b'/b$` term decreases** — fewer microbatches, so the pipeline fills and drains faster in terms of total microbatches processed. However, this also means fewer opportunities to amortize the pipeline bubble `$p-1$`, so the relative bubble size `$(p-1)/(b'/b) = (p-1) \cdot b / b'$` *increases*.
- **The `$t_f(b)$` and `$t_b(b)$` functions decrease per unit of data** — larger matrix multiplications have higher arithmetic intensity (more FLOPs per byte loaded from memory), so they achieve a higher fraction of peak GPU throughput. This effect is shown in Figure 7, where per-GPU throughput on a single GPU increases from roughly 40 teraFLOP/s at `$b=1$` to over 75 teraFLOP/s at `$b=8$` — approximately a 1.9× improvement.

These two effects pull in opposite directions: larger microbatches are more compute-efficient but inflate the pipeline bubble. The optimal `$b$` balances them.

**Empirical validation (Section 5.5, Figure 16).** For a 91-billion-parameter GPT model with `$(t, p) = (8, 8)$` on 64 GPUs, the optimal microbatch size is 2. The paper notes that the optimal value is model-dependent, and provides the analytical model as a tool to estimate it without exhaustive search.

---

#### Activation Recomputation

Activation recomputation (also called gradient checkpointing or rematerialization) trades compute for memory. During the forward pass, only the input activations to each "checkpointed" layer are saved. During the backward pass, the forward computation for that layer is re-executed from the saved input to regenerate the intermediate activations needed for the backward computation. This avoids storing all intermediate activations (which dominate memory for large models) at the cost of one extra forward pass per checkpoint.

The paper analyzes where to place checkpoints. Let `$A_{\text{input}}$` be the memory occupied by the input activations of a single layer, and `$A_{\text{intermediate}}$` be the memory occupied by intermediate activations within the layer. If a pipeline stage contains `$l$` layers and we checkpoint every `$c$` layers, the total activation memory is:

$$\text{Activation memory} = c \cdot A_{\text{input}} + \frac{l}{c} \cdot A_{\text{intermediate}}$$

**What it computes:** the memory footprint of activations for one pipeline stage. The first term `$c \cdot A_{\text{input}}$` is the memory for the stashed input activations (one per checkpoint, so `$c$` of them). The second term `$(l/c) \cdot A_{\text{intermediate}}$` is the memory for the intermediate activations within the currently executing checkpoint group (we only keep intermediates for layers in the active checkpoint; there are `$l/c$` such groups, each producing its own intermediates that are discarded after the backward pass for that group).

**Why this form matters.** It is a convex function of `$c$` with a minimum at `$c = \sqrt{l \cdot A_{\text{intermediate}} / A_{\text{input}}}$`. The paper states that in practice, checkpointing every 1 or 2 transformer layers is optimal.

**Throughput impact (Section 5.6, Figure 17).** Activation recomputation reduces throughput by up to 33% for small batch sizes compared to no recomputation (at the same batch size), due to the extra forward pass. However, recomputation enables training with larger batch sizes that would otherwise exceed memory capacity, and the throughput at these larger batch sizes can be up to 2× higher than the best throughput without recomputation (which is capped at a smaller batch size). This is because the larger batch size reduces the pipeline bubble fraction.

---

#### The Pipeline Scheduling Algorithms

The paper builds on the PipeDream-Flush 1F1B schedule and introduces an interleaved variant. Both schedules are defined on a pipeline with `$p$` stages (GPUs) processing a batch of `$m$` microbatches.

**Default non-interleaved 1F1B schedule (Figure 4, top).** The schedule has three phases:

1. **Warm-up phase:** The pipeline fills from the input. GPU 1 processes forward pass for microbatch 1, then microbatch 2, and so on, until it has processed `$p$` microbatches (or until the pipeline is full). Each downstream GPU starts processing as soon as it receives input from its upstream neighbor, so GPU 2 begins microbatch 1's forward pass one step after GPU 1, GPU 3 two steps after, and so forth.

2. **Steady-state phase:** Once the pipeline is full, each GPU enters a steady rhythm of one forward pass followed by one backward pass (hence "1F1B"). During each time step, the first pipeline stage (GPU 1) injects a new microbatch (forward), while the last pipeline stage (GPU `$p$`) completes a microbatch (backward). Intermediate GPUs process one forward and one backward microbatch per cycle.

3. **Drain/flush phase:** After the last microbatch is injected, no new forward passes start. The pipeline drains as in-flight microbatches complete their backward passes. During this phase, some GPUs are idle — the pipeline bubble.

**Why 1F1B over GPipe.** The GPipe schedule processes all forward passes first, then all backward passes. This stores activations for all `$m$` microbatches simultaneously, giving memory proportional to `$m$`. 1F1B interleaves forward and backward passes, so at most `$p$` microbatches are in flight at any time (one per pipeline stage). Activation memory thus scales with `$p$` rather than `$m$`. Since `$m$` is typically much larger than `$p$` (hundreds vs. tens), 1F1B dramatically reduces memory.

**Pipeline bubble size (both default and 1F1B).** The total bubble time is the same for both GPipe and 1F1B:

$$\text{Bubble time} = (p - 1) \cdot (t_f + t_b)$$

where `$t_f$` is the time for one forward pass on a single microbatch and `$t_b$` is the time for one backward pass. The bubble fraction relative to ideal processing time `$m \cdot (t_f + t_b)$` is:

$$\text{Bubble fraction} = \frac{p - 1}{m}$$

**What this means physically.** At the start of a batch, the upstream stages are busy injecting microbatches while the last stage `$p$` waits `$p-1$` steps before receiving its first microbatch. At the end, the first stage finishes early and waits `$p-1$` steps for the backward passes to drain. These two `$(p-1)$` periods are the pipeline bubble. For the bubble to be small, `$m$` must be much larger than `$p$`.

---

#### Interleaved Pipeline Schedule (Novel Contribution)

The key insight behind the interleaved schedule is that each GPU can be assigned multiple non-contiguous chunks of layers rather than a single contiguous set. If each GPU previously held layers 1–4 (stage 1) or layers 5–8 (stage 2), it can instead hold layers 1–2 and 9–10 (stage 1, chunk 1 and chunk 2), while the next GPU holds layers 3–4 and 11–12 (stage 2, chunk 1 and chunk 2), and so on.

**Why this reduces the bubble.** With `$v$` model chunks per device (equivalently, `$v$` virtual pipeline stages per physical device), the computation per microbatch per stage shrinks by a factor of `$v$`: each forward pass now takes `$t_f / v$` time, and each backward pass takes `$t_b / v$`. The bubble time becomes:

$$\text{Interleaved bubble time} = (p - 1) \cdot \frac{t_f + t_b}{v}$$

and the bubble fraction:

$$\text{Interleaved bubble fraction} = \frac{1}{v} \cdot \frac{p - 1}{m}$$

**What it computes:** the fraction of time idle has been reduced by a factor of `$v$`. If each device handles 2 model chunks, the bubble is halved; with 4 chunks, it is quartered.

**Why this works physically (Figure 4, bottom).** The interleaved schedule still follows a 1F1B rhythm, but the "forward" and "backward" operations are finer-grained. A device processes forward for chunk 1 of microbatch A, then forward for chunk 2 of some other microbatch, then backward for chunk 1 of yet another microbatch, and so on. The key effect is that the pipeline "fill" and "drain" phases complete sooner because each stage processes smaller units of work. In Figure 4, with `$v=2$` on a 4-device pipeline, the flush (idle devices) begins noticeably earlier in the timeline compared to the default schedule.

**The constraint.** The number of microbatches `$m$` must be an integer multiple of the pipeline-parallel size `$p$`. For example, with `$p=4$`, the batch must have `$m = 4, 8, 12, \dots$` microbatches. This ensures the interleaving pattern aligns correctly.

**The cost: increased communication.** Each model chunk is now smaller, but the number of communication operations between pipeline stages increases. With `$v$` chunks, there are `$v$` times as many point-to-point messages (one per chunk boundary), though each message carries `$1/v$` the data. The total data volume is unchanged, but the number of messages — and thus the communication latency burden — grows. Section 4.1's scatter/gather optimization mitigates this by reducing per-message data on inter-node links.

**Empirical validation (Section 5.3.2, Figure 12).** For the 175-billion-parameter GPT-3 model on 96 GPUs, the interleaved schedule (`$v=2$`) with scatter/gather optimization outperforms the non-interleaved schedule by up to 10+% at smaller batch sizes. At larger batch sizes, the gap closes because the default schedule's bubble `$(p-1)/m$` shrinks as `$m$` grows, and because the interleaved schedule's extra communication overhead becomes more significant (point-to-point communication scales with `$b$`). The paper notes that without the scatter/gather optimization, the default schedule actually outperforms the interleaved schedule at large batch sizes.

---

#### Tensor Model Parallelism: The Megatron Partitioning

Tensor model parallelism partitions individual transformer layers rather than distributing entire layers across GPUs. The paper uses the same partitioning strategy as Megatron (Shoeybi et al., 2020), which splits matrix multiplications in the MLP and self-attention blocks.

**MLP block partitioning (Figure 5a).** The MLP computes:

$$Y = \text{GeLU}(X A)$$
$$Z = \text{Dropout}(Y B)$$

where `$X$` is the input, `$A$` is the first weight matrix (expanding hidden size `$h$` to `$4h$`), and `$B$` is the second weight matrix (projecting back to `$h$`).

The weight matrix `$A$` is split column-wise: `$A = [A_1, A_2]$`, where `$A_1$` and `$A_2$` are each `$h \times 2h$` on two GPUs. The input `$X$` is replicated (identical on both GPUs). This yields:

$$[Y_1, Y_2] = [\text{GeLU}(X A_1), \text{GeLU}(X A_2)]$$

**Why column-wise splitting of `$A$` is critical.** The GeLU activation is a non-linear function. If `$A$` were split row-wise, the partial products would need to be summed *before* the GeLU, requiring an all-reduce synchronization before the non-linearity. Column-wise splitting allows the GeLU to be applied independently on each GPU, eliminating this synchronization. This is the key design insight: partition at a point where the computation graph has an element-wise non-linearity that can be applied in parallel.

The weight matrix `$B$` is then split row-wise: `$B = [B_1; B_2]$` (concatenated vertically, meaning `$B_1$` is `$2h \times h$` and `$B_2$` is `$2h \times h$`). The partial outputs `$Y_1 B_1$` and `$Y_2 B_2$` are computed independently on each GPU, then summed via an all-reduce (the `$g$` operator in Figure 5a) to produce the final output `$Z$` before dropout.

**Self-attention block partitioning (Figure 5b).** The self-attention mechanism computes query, key, and value projections, then performs attention-weighted aggregation. The paper exploits the multi-head structure: attention heads are independent, so they can be partitioned across GPUs.

The key (`$K$`), query (`$Q$`), and value (`$V$`) matrices are each split column-wise: `$Q = [Q_1, Q_2]$`, `$K = [K_1, K_2]$`, `$V = [V_1, V_2]$`. On two GPUs, GPU 1 computes attention for the first half of the heads using `$(Q_1, K_1, V_1)$`, and GPU 2 computes attention for the second half using `$(Q_2, K_2, V_2)$`. The output linear layer weight matrix `$B$` is split row-wise, so each GPU computes a partial output, and an all-reduce (`$g$`) sums them.

**Communication requirements per transformer layer.** In the forward pass, there are two all-reduce operations: one after the self-attention output projection (`$g$` in Figure 5b) and one after the MLP second GEMM (`$g$` in Figure 5a). In the backward pass, there are two corresponding all-reduce operations for gradients (`$f$` in the figures, which is the identity in the forward pass and an all-reduce in the backward pass). The paper implements `$f$` and `$g$` as "a few lines of code" by exploiting the fact that `$f$` and `$g$` are conjugate operations — `$f$` is identity in the forward direction and all-reduce in the backward direction, while `$g$` is all-reduce in the forward direction and identity in the backward direction. This symmetry simplifies the implementation.

**Why tensor parallelism within a server.** All-reduce on 8 GPUs within a DGX A100 node via NVLink/NVSwitch achieves near-peak bandwidth with low latency. The total data reduced per all-reduce is `$b s h$` (the activation tensor size), and with `$t=8$`, each GPU sends and receives `$(7/8) \cdot b s h$` bytes twice per all-reduce (one reduce-scatter, one all-gather in NCCL's ring implementation). Within the node, this completes in microseconds and can be partially overlapped with computation. Across nodes on InfiniBand, the same operation would take substantially longer and become the throughput bottleneck — which is exactly what the paper's experiments in Section 5.4.1 confirm.

---

#### Scatter/Gather Communication Optimization

When tensor parallelism and pipeline parallelism are combined, consecutive pipeline stages that are both tensor-parallel send *identical* tensors between corresponding ranks. For example, if stage 1 uses 8 GPUs for tensor parallelism and stage 2 uses another 8 GPUs, then rank `$i$` in stage 1 sends the exact same tensor to rank `$j$` in stage 2 as rank `$i'$` sends to rank `$j'$`. The paper observes that this redundancy can be exploited to better utilize the multiple InfiniBand cards on each DGX A100 server.

**Without optimization (Figure 9a).** Each tensor-parallel rank sends its full tensor to the corresponding rank in the next stage. The same data crosses multiple InfiniBand links redundantly. If `$t=8$`, the effective cross-node bandwidth is divided by 8 because 8 copies of the same data traverse 8 different IB links.

**With scatter/gather optimization (Figure 9b).** At the sender, the tensor is split into `$t$` equal-sized chunks (one per tensor-parallel rank). Each rank sends only its assigned chunk to the corresponding rank in the next stage over that rank's dedicated InfiniBand card. At the receiver, an all-gather operation over NVLink (within the receiving node) reassembles the full tensor from the chunks. The all-gather uses the fast intra-node NVLink/NVSwitch fabric, which has far more bandwidth than the inter-node InfiniBand links.

**Quantitative effect.** The amount of data sent over each inter-node InfiniBand link per microbatch is reduced from `$b s h$` to `$b s h / t$`. With `$t = 8$`, this is an 8× reduction. This optimization is what makes the interleaved schedule practical: without it, the `$v$`-fold increase in communication messages would overwhelm the inter-node links.

**Empirical validation (Section 5.7, Figure 18).** For the 175-billion-parameter GPT-3 model on 96 GPUs with the interleaved schedule, the scatter/gather optimization improves throughput by up to 11% at large batch sizes. The benefit is largest when communication is the bottleneck — large batch sizes with interleaving, where many point-to-point messages traverse the inter-node links.

---

#### Computation Optimizations: Fused Kernels and Data Layout

The paper implements three computation-side optimizations to keep the GPU compute units busy:

**Data layout transformation.** Standard transformer implementations use a tensor layout of `[b, s, a, h]` — batch size, sequence length, attention heads, hidden size per head. The paper changes this to `[s, b, a, h]`. This reordering enables the use of strided batched GEMM kernels, where multiple independent matrix multiplications (one per attention head and one per sequence position in the batch) are launched as a single batched GEMM call, amortizing kernel launch overhead and improving memory access patterns. The paper does not detail the specific GEMM library calls, but the principle is that cuBLAS batched GEMM routines (`cublasGemmStridedBatched`) operate more efficiently on this layout.

**Fused element-wise kernels.** Two fused kernels are generated using PyTorch JIT:

1. **Bias + GeLU fusion:** The bias addition and GeLU activation, which would normally be two separate kernel launches (one for the add, one for the GeLU), are fused into a single kernel. This eliminates the round-trip to global memory between operations — the intermediate result stays in registers or shared memory.

2. **Bias + dropout + add fusion:** The bias addition to the attention or MLP output, followed by dropout and residual addition, is fused similarly. This is a three-operation fusion that eliminates two global memory round-trips.

**Custom softmax kernels.** Two custom CUDA kernels handle the scale-mask-softmax sequence in the attention computation:

1. **General masking kernel:** Supports arbitrary attention masks (used in bidirectional models like BERT, where padding tokens must be masked).

2. **Implicit causal masking kernel:** For auto-regressive models like GPT, the causal mask (lower-triangular) is applied implicitly as part of the softmax reduction, avoiding the need to materialize the full `$s \times s$` attention matrix with the mask applied.

**Empirical impact (Section 5.8).** For the 175-billion-parameter GPT-3 model, operator fusion increases throughput by 19% (from 113 to 135 teraFLOP/s per GPU). For the 530-billion-parameter model, the improvement is 11% (from 133 to 148 teraFLOP/s per GPU). The larger model benefits less proportionally because its GEMMs are larger (better saturating the GPU) and communication is a larger fraction of total time, leaving less room for compute-side improvements to matter.

---

#### FLOP Counting and Training Time Estimation

The paper provides a detailed FLOP model that connects model architecture to computational cost, and uses it to estimate end-to-end training time.

**Model parameter count.** For a transformer with `$l$` layers, hidden size `$h$`, vocabulary size `$V$`, and sequence length `$s$`:

$$P = 12 l h^2 \left(1 + \frac{13}{12h} + \frac{V + s}{12 l h}\right)$$

where `$P$` is the total number of parameters. This formula accounts for: the 12 basic weight matrices per transformer layer that scale as `$h^2$` (4 in self-attention: Q, K, V, output projection, each `$h \times h$`; and 2 in the MLP: `$h \times 4h$` and `$4h \times h$`, with the factor of 12 coming from `$4 \times h^2 + 2 \times (h \cdot 4h) = 12h^2$`), plus corrections for biases (“13/12h”), vocabulary embedding (“V/(12lh)”), and position embedding (“s/(12lh)”). For the very large models considered, `$12lh \gg V$` and `$6h \gg s$`, so the leading term `$12lh^2$` dominates.

**FLOPs per iteration.** Counting the matrix multiplications in all transformer layers and the logit layer, and accounting for activation recomputation (which adds an extra forward pass), the paper derives:

$$F = 96 B s l h^2 \left(1 + \frac{s}{6h} + \frac{V}{16 l h}\right)$$

where `$F$` is the number of floating-point operations per training iteration, `$B$` is the global batch size, `$s$` is sequence length, `$l$` is the number of layers, `$h$` is hidden size, and `$V$` is vocabulary size. The factor 96 comes from: 24 base FLOPs per token per layer per forward pass (12 from attention + 12 from MLP, in units of `$B s h^2$`), times 4 to account for forward + backward (×2) and activation recomputation (another forward, so ×2 again → ×4). The correction terms account for the attention matrix computation (`$s/(6h)$`) and logit layer (`$V/(16lh)$`), both of which are small for the large models considered.

For the largest models where `$6h \gg s$` and `$16lh \gg V$`, this simplifies to `$F \approx 96 B s l h^2$`.

**End-to-end training time.** Combining the FLOP model with measured throughput, the paper derives a simplified training time formula. The number of iterations to process `$T$` tokens is `$I = T / (B \cdot s)$`. Total FLOPs for training is `$I \cdot F$`. With empirical per-GPU throughput `$X$` (teraFLOP/s) and `$n$` GPUs, the total training time in seconds is `$(I \cdot F) / (n \cdot X)$`. Under the large-model approximations, this simplifies to:

$$\text{Training time} \approx \frac{8 T P}{n X}$$

**What it computes:** the wall-clock time to train a model with `$P$` parameters on `$T$` tokens using `$n$` GPUs each achieving throughput `$X$`. The constant 8 comes from the factor 96 in the FLOP formula (`$96 / 12 = 8$` FLOPs per parameter per token, after canceling `$l h^2$` terms between `$F$` and `$P$`). For GPT-3 (175 billion parameters, 300 billion tokens, 1024 GPUs at 140 teraFLOP/s each), this gives approximately 34 days. For the 1-trillion-parameter model (450 billion tokens, 3072 GPUs at 163 teraFLOP/s each), this gives approximately 84 days, or about 3 months.

**Why this is an estimate, not an exact prediction.** The formula assumes the model's FLOPs are dominated by the `$96 B s l h^2$` term, which is true for very large models but introduces small errors for smaller configurations. It also uses measured throughput `$X$` from the specific hardware/software configuration, which would differ on other systems. The paper presents these numbers as evidence of practicality ("we believe these training times are practical") rather than guarantees.

---

#### Putting It All Together: The PTD-P Configuration Recipe

The paper does not automate parallelism configuration. Instead, it provides a manual recipe based on the analytical models and empirical results:

1. **Determine the minimum model-parallel size `$M = t \cdot p$`** such that the model's parameters, gradients, optimizer states, and activations fit in GPU memory. This is a hard constraint: if `$M$` is too small, the model cannot be trained.

2. **Set `$t$` to the number of GPUs per server** (8 for DGX A100). This keeps tensor-parallel all-reduce communication within the high-bandwidth NVLink domain.

3. **Set `$p = M / t$`** to satisfy the memory constraint. This determines the pipeline-parallel size.

4. **Allocate all remaining `$n$` GPUs to data parallelism**, setting `$d = n / M$`. If `$n$` is not a multiple of `$M$`, some configuration adjustment is needed (the paper does not discuss this case explicitly).

5. **Choose the microbatch size `$b$`** to balance pipeline bubble size against arithmetic intensity, using the analytical model from Section 3.4.

6. **Enable activation recomputation** with checkpoints every 1-2 layers to reduce memory, allowing larger batch sizes or larger models.

7. **Use the interleaved pipeline schedule** with `$v$` model chunks per device (typically `$v=2$` or more) to reduce pipeline bubble, combined with the scatter/gather communication optimization to handle the increased message count.

This recipe is what produces the configurations in Table 1, where the largest model (1 trillion parameters) uses `$(p=64, t=8, d=6)$` (since `$n=3072$`, and `$64 \cdot 8 \cdot 6 = 3072$`). The tensor-parallel size is 8 (one DGX server), the pipeline spans 64 stages across 8 DGX servers, and 6 such pipeline+tensor groups run data-parallel.

## 4. Key Insights and Innovations

### Innovation 1: The "Tensor Within a Server, Pipeline Across Servers" Principle as a Formalized Systems Heuristic

The paper's most practically consequential contribution is not the individual parallelism techniques themselves — tensor model parallelism was established by Megatron (Shoeybi et al., 2020), pipeline parallelism by GPipe (Huang et al., 2019) and PipeDream (Narayanan et al., 2019) — but rather the **quantitative justification for a specific boundary between them** based on hardware topology. Before this paper, it was understood that tensor parallelism requires high communication bandwidth and pipeline parallelism is more tolerant of lower bandwidth, but the field lacked a crisp operational rule: *where exactly* should one stop using tensor parallelism and switch to pipeline parallelism?

The paper's answer — confine tensor parallelism to the number of GPUs within a single server (8 for DGX A100), and use pipeline parallelism to stitch such server-sized tensor-parallel groups together — seems obvious in retrospect. But the paper earns this heuristic through analysis, not assumption. Section 3.2 derives the pipeline bubble fraction as $(n/t - 1)/m$ and the tensor communication volume as $l_{\text{stage}} \cdot 8 b s h \cdot (t-1)/t$, showing that tensor parallelism's per-microbatch all-reduce cost grows with layer count and GPU count, while pipeline parallelism's point-to-point cost is independent of both. The empirical validation in Figure 13 — where the $(p=8, t=8)$ configuration achieves peak throughput on a 162-billion-parameter model across 64 GPUs, substantially outperforming $(p=16, t=4)$ and $(p=32, t=2)$ which push tensor parallelism across nodes — converts the analytical model into a concrete, validated design rule.

This matters because it **resolves an ambiguity in how the parallelism techniques should interact**. Prior work like Mesh-TensorFlow (Shazeer et al., 2018) and FlexFlow (Jia et al., 2018) provided *languages* for specifying parallelization strategies or *search algorithms* for finding them automatically, but neither provided a simple, actionable heuristic grounded in the hardware reality of modern GPU clusters (NVLink within nodes, InfiniBand across nodes). The paper's heuristic is not a search result — it is a one-time decision derived from the relative costs of all-reduce (collective, bandwidth-intensive, per-microbatch) versus point-to-point (peer-to-peer, lower bandwidth, pipelineable). The fact that it works across model sizes from 1.7 billion to 1 trillion parameters (Table 1), with $t=8$ used for all configurations from 8 GPUs to 3,072 GPUs, demonstrates that it is not a model-specific fluke but a fundamental property of the communication hierarchy.

This is a **systems design insight**, not a theoretical advance. It does not prove anything about the optimality of tensor-vs-pipeline boundaries in general, and the paper explicitly does not claim optimality — "we do not automatically explore the search space of parallelism strategies... but instead suggest heuristics that we found work well in practice." However, the combination of analytical grounding and empirical validation across three orders of magnitude of model scale elevates it beyond mere rule-of-thumb. It is the kind of insight that changes how practitioners configure distributed training: you no longer need to experiment with arbitrary $(p, t)$ combinations for each new model; you set $t$ to your server's GPU count and let the rest follow.

---

### Innovation 2: The Interleaved Pipeline Schedule as a Mechanism for Decoupling Bubble Size from Pipeline Depth

The interleaved pipeline schedule (Section 2.2.2, Figure 4 bottom) is the paper's primary algorithmic novelty. Its conceptual contribution is the realization that **pipeline bubble size can be reduced without decreasing the number of pipeline stages or increasing the batch size**, by assigning each physical GPU multiple non-contiguous model chunks and interleaving their execution. Prior pipeline schedules — GPipe's all-forward-all-backward (Huang et al., 2019) and PipeDream-Flush's 1F1B (Narayanan et al., 2021) — tied bubble size directly to the ratio $(p-1)/m$. To reduce the bubble, you either decreased $p$ (fewer pipeline stages, which limits model size) or increased $m$ (more microbatches, which requires a larger batch size that may not be feasible or desirable). The interleaved schedule introduces a third knob: increase the number of virtual pipeline stages $v$ per device, reducing the bubble by a factor of $v$ at the cost of increased communication frequency.

This is an **elegant reframing of the pipeline scaling problem**. Before interleaving, pipeline parallelism had an inherent tension: deeper pipelines (more GPUs, larger models) meant larger bubbles, forcing ever-larger batch sizes to maintain efficiency. After interleaving, the bubble size and the pipeline depth are partially decoupled. A 64-stage pipeline with $v=4$ interleaving has the same bubble fraction as a 16-stage non-interleaved pipeline with the same $m$, while still distributing the model across 64 GPUs. The bubble fraction formula $1/v \cdot (p-1)/m$ makes this relationship explicit and quantifiable.

What distinguishes this from prior work on pipeline scheduling is that it **exploits a degree of freedom that was previously unused**: the assignment of layers to pipeline stages. GPipe and PipeDream both assumed a simple contiguous mapping — GPU 1 gets layers 1–4, GPU 2 gets layers 5–8, etc. By relaxing this assumption and allowing non-contiguous assignments (GPU 1 gets layers 1, 2, 9, 10; GPU 2 gets layers 3, 4, 11, 12), the schedule gains the ability to interleave finer-grained work units. The paper does not claim this layer-assignment idea is entirely new — the concept of multiple pipeline stages per device appears in prior work on asynchronous pipelining — but the specific construction of a memory-efficient 1F1B interleaved schedule that preserves strict optimizer semantics is novel.

The practical significance is demonstrated in Figure 12: on the 175-billion-parameter GPT-3 model with 96 GPUs, interleaving improves throughput by 10+% at modest batch sizes. This is not a dramatic absolute gain, but it matters precisely where pipeline parallelism is most stressed — when batch sizes cannot be made arbitrarily large due to convergence constraints or memory limits. The paper's acknowledgment that the interleaved schedule requires more communication (Section 2.2.2, "$v$ times as many messages") and that the scatter/gather optimization (Section 4.1) is necessary to realize its benefits underscores that this is a **tightly coupled systems innovation**: the schedule itself, the communication optimization that makes it viable, and the hardware topology that the optimization exploits are all part of a single design.

This is a **fundamental contribution** to the design space of pipeline schedules — not because it achieves the highest possible throughput (asynchronous schedules like PipeDream-2BW might do better), but because it expands the set of configurations where synchronous pipelining with strict optimizer semantics remains efficient. It is the kind of contribution that opens a new dimension for future work: how far can $v$ be pushed? What are the tradeoffs in memory and communication that limit interleaving depth? The paper does not answer these questions, but it establishes the framework for asking them.

---

### Innovation 3: The Analytical Decomposition of Throughput into Interacting Parallelism Dimensions

The paper's analytical models in Section 3 — decomposing total training throughput into pipeline bubble size, communication volume, and per-GPU compute efficiency as functions of $(p, t, d, b)$ — represent a **conceptual contribution to how distributed training performance is understood**. Prior work had analyzed these dimensions in isolation: GPipe quantified the pipeline bubble, Megatron analyzed tensor-parallel communication, and data-parallel scaling was well-understood from HPC. But no prior work had written down a unified set of expressions that show how these dimensions *interact* — how increasing tensor parallelism reduces the pipeline bubble but increases per-microbatch communication, or how data parallelism reduces the bubble (via the $n-d$ numerator in the bubble formula) while introducing its own all-reduce cost at batch boundaries.

This is not a theoretical breakthrough — the models are simple algebraic expressions, not deep performance models. But their value lies in **making the tradeoffs legible**. Takeaway #1 ("tensor within a server, pipeline across servers") is not derived from a complex cost model or a search algorithm; it falls out directly from observing that tensor communication scales with $l_{\text{stage}} \cdot (t-1)/t$ per microbatch while pipeline communication scales with $b s h$ regardless of $t$, and that all-reduce across servers is qualitatively more expensive than point-to-point. Takeaway #2 ("use just enough model parallelism to fit the model, then scale with data parallelism") follows from the observation that the pipeline bubble decreases with larger $d$ (since $p = n/(t \cdot d)$ shrinks) and data-parallel communication is amortized over an entire batch rather than incurred per-microbatch. These are not results that require empirical search to discover — the models make them predictable.

What elevates this beyond mere back-of-the-envelope calculation is the **empirical validation that the models capture the dominant effects**. Figure 13 (tensor vs. pipeline) shows the predicted non-monotonic relationship: throughput peaks at an intermediate tensor-parallel size, declining at both extremes. Figure 14 (pipeline vs. data) shows the monotonic decrease with pipeline-parallel size that the bubble formula predicts. Figure 16 (microbatch size sweep) shows the predicted tension between arithmetic intensity (improving with larger $b$) and bubble size (worsening with larger $b$, since $m = b'/b$ shrinks). The fact that these qualitative predictions hold across model sizes and batch sizes gives the analytical models credibility as design tools, even if they do not produce quantitatively exact throughput predictions (the paper does not claim they do).

This is an **incremental but practically important advance** in the methodology of distributed training system design. It does not replace empirical tuning — the optimal microbatch size is still problem-dependent, and the paper explicitly states this. But it provides a structured way to reason about *why* one configuration outperforms another, and it identifies the key tradeoffs that any automatic partitioning algorithm would need to navigate. The paper's positioning as a heuristic-driven rather than search-driven approach (contrasted with FlexFlow and PipeDream's automatic partitioning) is a deliberate choice: for the specific domain of large transformer language models on homogeneous GPU clusters, the search space of $(p, t, d, b)$ can be navigated with simple rules rather than expensive optimization. The analytical models justify why those rules work.

---

### Innovation 4: Scatter/Gather Communication Optimization as a Concrete Instantiation of Topology-Aware Communication

The scatter/gather communication optimization (Section 4.1, Figure 9b) is a **systems engineering insight** that exemplifies a broader principle: when composing parallelism dimensions, communication patterns that are redundant at one level of the hierarchy can be restructured to exploit higher-bandwidth links at another level. The specific observation is that when tensor parallelism ($t=8$) and pipeline parallelism are combined across DGX A100 nodes, the 8 tensor-parallel ranks in one pipeline stage send *identical* tensors to the 8 corresponding ranks in the next stage. Naively, this means 8 copies of the same data traverse 8 different InfiniBand links, wasting the aggregate cross-node bandwidth.

The optimization — scatter the tensor into $t$ chunks at the sender, send one chunk per IB link, then all-gather over NVLink at the receiver — is not algorithmically complex. But its **conceptual significance** lies in recognizing that the redundancy created by tensor parallelism (replicated activations across tensor-parallel ranks after the MLP's $g$ all-reduce; see Figure 5a) can be *exploited* rather than merely tolerated. The paper leverages a property that is normally a cost (redundant computation and communication from tensor parallelism) and converts it into a benefit (the ability to parallelize inter-stage communication across multiple IB rails).

This matters beyond the specific 8-GPU, 8-IB-card configuration. It demonstrates a **design pattern** for multi-dimensional parallelism: when one parallelism dimension produces replicated data, that replication can be used to stripe communication across available links in another dimension. The pattern would apply to any hardware topology where intra-node bandwidth (NVLink, 600 GB/s per link) substantially exceeds inter-node bandwidth (InfiniBand HDR, 200 Gbps = 25 GB/s per link), and where model parallelism creates identical data across a group of co-located GPUs.

The empirical impact (up to 11% throughput improvement on the 175B GPT-3 model with interleaving; Figure 18) is modest in absolute terms but **critical for enabling the interleaved schedule** at large batch sizes. The paper notes that without this optimization, the default non-interleaved schedule actually outperforms the interleaved schedule at large batch sizes — meaning the interleaved schedule's bubble reduction would be negated by communication overhead. The scatter/gather optimization is thus a **enabling technology** for Innovation 2, not a standalone performance win.

This is an **incremental contribution** to the engineering of distributed training, but one that exemplifies the paper's broader message: achieving 52% of peak FLOP/s at 3,072 GPUs requires attention to *every* level of the hardware/software stack, from the pipeline schedule (macro-level) to the data layout in GEMM kernels (micro-level). The scatter/gather optimization sits at the boundary between parallelism strategy and hardware topology, and its necessity underscores that good parallelization decisions at one level (using $t=8$ within a server) create optimization opportunities at another level (striping inter-stage communication across IB cards) that would not exist otherwise.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** All experiments use GPT-style transformer language models with configurations specified per experiment. There is no fixed evaluation dataset in the conventional sense — the system is evaluated on *training throughput*, not downstream task accuracy. The "data" is synthetic training batches processed during throughput measurement. Model configurations span 1.7 billion to 1 trillion parameters, with vocabulary size V = 51,200 (a multiple of 1024) and sequence length s = 2048 throughout. Model sizes are computed using Equation 2 in the paper, and FLOP counts using Equation 3.

- **Base model(s).** The experiments use GPT-family decoder-only transformer models with standard architectures. Specific configurations vary by experiment and are documented in Table 1 (for end-to-end scaling), Section 5.3 (for pipeline parallelism scaling), Section 5.4 (for parallel configuration tradeoffs), and Section 5.5 (for microbatch size sweeps). The largest model — 1 trillion parameters — uses 128 attention heads, hidden size 25,600, and 128 transformer layers. Model sizes scale by increasing hidden size, number of attention heads, and number of layers proportionally. GPT-3 (175 billion parameters) is used as a reference point for the ZeRO comparison and interleaved schedule experiments.

- **Metrics.** The primary metric is **achieved teraFLOP/s per GPU**, measured for end-to-end training (including data loading, forward and backward passes, communication, optimizer steps, and logging). Aggregate throughput is reported by multiplying per-GPU throughput by the number of GPUs. A secondary metric is **percentage of theoretical peak FLOP/s**, where the A100 GPU peak at 16-bit precision is 312 teraFLOP/s. Throughput is measured in steady-state training over multiple iterations; the paper does not specify the exact number of iterations used for measurement. Training time estimates (in days) are derived analytically using Equation 4: End-to-end training time ≈ 8TP / (nX), where P is parameter count, T is training tokens, n is number of GPUs, and X is per-GPU throughput.

- **Baselines.** The paper compares against:
  - **ZeRO-3** (Rajbhandari et al., 2019) — sharded data parallelism without model parallelism, integrated via the DeepSpeed Python library. Compared for 175-billion and 530-billion parameter models at various GPU counts in Figure 10 and Table 2.
  - **Non-interleaved (default) 1F1B pipeline schedule** — the PipeDream-Flush schedule from Narayanan et al. (2021), serving as baseline for interleaved schedule experiments (Figure 12).
  - **Unoptimized communication** — pipeline communication without the scatter/gather optimization, compared in Figure 18.
  - **Without operator fusion** — computation without the fused kernels described in Section 4.2, compared in Section 5.8.
  - **Without activation recomputation** — compared in Figure 17 to isolate the throughput/memory tradeoff.

- **Generation budget / compute accounting.** Throughput is measured in FLOP/s, not generation count. FLOPs are counted using Equation 3, which lower-bounds the true count by considering only matrix multiplications in transformer and logit layers (excluding normalization, embedding lookups, and element-wise operations). Activation recomputation's extra forward pass is included in the FLOP count. A FLOP is counted as one floating-point operation regardless of precision (mixed precision training is used, but FLOP counting treats all operations uniformly). Communication time is included in the throughput measurement (end-to-end), meaning reported FLOP/s accounts for both computation and communication.

- **Cross-validation / statistical protocol.** This is a systems performance paper, not a machine learning evaluation, so there is no train/validation/test split or statistical significance testing. Throughput is measured empirically on the Selene supercomputer (NVIDIA DGX A100 nodes with 8 80GB A100 GPUs each, connected via NVLink/NVSwitch intra-node and 200 Gbps HDR InfiniBand inter-node in a three-level fat-tree topology with 850 switches). The paper does not report variance across runs, the number of iterations used for throughput measurement, or warm-up iteration counts. This is standard for systems papers where throughput measurements are typically stable given the deterministic nature of the training loop, but it means reproducibility depends on access to identical hardware.

### Main Quantitative Results

#### End-to-End Scaling from 1 Billion to 1 Trillion Parameters

The headline result is Table 1, which reports weak-scaling throughput for eight GPT model configurations ranging from 1.7 billion to 1,008 billion (1 trillion) parameters, using 32 to 3,072 GPUs. The largest configuration achieves **163 teraFLOP/s per GPU (52% of theoretical peak) and 502 petaFLOP/s aggregate** on 3,072 A100 GPUs. The interleaved pipeline schedule with scatter/gather optimization is enabled for all configurations.

Key observations from Table 1:

- **Per-GPU throughput improves with model scale** — from 137 teraFLOP/s at 1.7B parameters to 163 teraFLOP/s at 1T parameters — despite using more GPUs and larger batch sizes. This is attributed to larger matrix multiplications better saturating GPU compute units. The paper notes this as "super-linear scaling" since per-GPU utilization increases with model size even as the total GPU count grows.

- **The percentage of theoretical peak increases from 44% to 52%** as models grow, confirming that larger GEMMs push the GPU closer to its compute limit. The 1.7B-parameter model achieves only 44%, while the 310B-parameter model reaches 50% and the trillion-parameter model reaches 52%.

- **Batch size scales with model size** — from 512 for the smallest model to 3,072 for the trillion-parameter model — to keep the pipeline bubble fraction manageable as pipeline-parallel size grows (from p=1 at 1.7B to p=64 at 1T).

- **Tensor-parallel size stays at 8** for all configurations with more than 8 GPUs, consistent with the "tensor within a server" heuristic (Takeaway #1). Pipeline-parallel size grows from 1 to 64 as models scale. Data-parallel size ranges from 1 to essentially fill the remaining GPU budget.

Using Equation 4, the paper estimates end-to-end training time: **34 days for GPT-3 (175B parameters, 300B tokens) on 1,024 GPUs**, and **84 days (approximately 3 months) for the 1-trillion-parameter model (450B tokens) on 3,072 GPUs**. The 450B token assumption for the trillion-parameter model is not justified in the paper — it is stated as an assumption without citation or derivation.

#### Comparison to ZeRO-3

Figure 10 and Table 2 compare PTD-P to ZeRO-3 (without model parallelism) for the 175B and 530B GPT models. The headline finding: **PTD-P outperforms ZeRO-3 by 70% at larger GPU counts for both models**.

Detailed results from Table 2:

- **175B model at 384 GPUs:** PTD-P achieves 153 teraFLOP/s per GPU vs. ZeRO-3's 144 teraFLOP/s — a 6% advantage.
- **175B model at 1,536 GPUs:** PTD-P achieves 141 teraFLOP/s per GPU vs. ZeRO-3's 44 teraFLOP/s — a 3.2× advantage. The ZeRO-3 throughput drops substantially (from 144 to 44) as GPU count increases because the global batch size is fixed at 1,536, so per-GPU microbatch size shrinks to 2 and then 1, reducing the ability to overlap communication with computation.
- **530B model at 560 GPUs:** PTD-P achieves 171 teraFLOP/s per GPU vs. ZeRO-3's 138 at 640 GPUs — a 24% advantage even with ZeRO-3 using more GPUs. (The 530B model did not fit on 560 GPUs with ZeRO-3 at microbatch size 4, so the comparison uses 640 GPUs and batch size 2,560 for ZeRO-3.)
- **530B model at 2,240 GPUs:** PTD-P achieves 159 teraFLOP/s per GPU vs. ZeRO-3's 48 — a 3.3× advantage.

The gap widens with GPU count because ZeRO-3's parameter fetching communication (all-gather of weight shards before each layer's computation) becomes the bottleneck when per-GPU batch size is small and communication cannot be hidden behind computation. PTD-P's pipeline and tensor parallelism do not have this scaling pathology because model parameters are statically partitioned rather than fetched on-demand.

#### Pipeline Parallelism Scaling

**Weak scaling (Figure 11).** Pipeline parallelism is evaluated in isolation by increasing the number of pipeline stages from 1 to 8 while proportionally increasing model size (from 3 layers / 15B parameters at p=1 to 24 layers / 121B parameters at p=8), with a fixed tensor-parallel size of 8 and microbatch size of 1. Two batch sizes are tested: 8 and 128.

- **Batch size 128:** Throughput scales relatively well, decreasing from roughly 160 teraFLOP/s per GPU at p=1 to approximately 110 teraFLOP/s at p=8 — a 31% reduction across an 8× increase in pipeline stages.
- **Batch size 8:** Throughput collapses from roughly 140 teraFLOP/s at p=1 to approximately 50 teraFLOP/s at p=8 — a 64% reduction. This is the pipeline bubble effect: with batch size 8 and microbatch size 1, m = 8 microbatches per pipeline, so the bubble fraction (p-1)/m reaches (8-1)/8 = 87.5% at p=8.

The results confirm the analytical model from Section 2.2.1: pipeline scaling efficiency depends critically on the ratio of microbatches to pipeline stages. Large batch sizes amortize the bubble; small batch sizes do not.

**Interleaved vs. non-interleaved schedule (Figure 12).** The 175B-parameter GPT-3 model is tested on 96 GPUs with both schedules, sweeping batch size from 12 to 60.

- **At batch size 12:** Interleaved achieves approximately 115 teraFLOP/s per GPU vs. non-interleaved's approximately 95 teraFLOP/s — a 21% improvement.
- **At batch size 60:** The gap narrows to approximately 142 vs. 137 teraFLOP/s per GPU — about 4%. The paper attributes this closing gap to two factors: (a) the non-interleaved bubble shrinks as batch size increases (m grows, so (p-1)/m decreases), and (b) interleaved communication overhead scales with batch size (more microbatches means more point-to-point messages), partially offsetting the bubble reduction benefit.
- **Without scatter/gather optimization:** The paper notes (without showing the data in Figure 12) that "the default schedule performs better than the interleaved schedule at larger batch sizes," meaning the scatter/gather optimization is essential for the interleaved schedule to be viable at all batch sizes.

#### Interaction Between Parallelism Dimensions

**Tensor vs. pipeline parallelism (Figure 13).** A 162.2B-parameter GPT model (32 layers, 128 attention heads, hidden size 20,480) is tested on 64 A100 GPUs, sweeping (p, t) pairs that multiply to 64: (2, 32), (4, 16), (8, 8), (16, 4), (32, 2). Two batch sizes are tested: 32 and 128.

- **Batch size 128:** (8, 8) achieves the highest throughput at approximately 195 teraFLOP/s per GPU. (4, 16) achieves roughly 175 teraFLOP/s. The extremes — (2, 32) at roughly 100 teraFLOP/s and (32, 2) at roughly 140 teraFLOP/s — both underperform substantially. The throughput ratio between best and worst is roughly 2×, as claimed in the paper's introduction.
- **Batch size 32:** The same ranking holds, with (8, 8) at approximately 175 teraFLOP/s and (2, 32) at roughly 80 teraFLOP/s.

The paper interprets this as direct evidence for Takeaway #1: t=8 (matching the 8 GPUs per DGX A100 server) is optimal because it keeps tensor-parallel all-reduce within NVLink while using enough pipeline stages to distribute the model across servers. Configurations with larger t (16, 32) push all-reduce across InfiniBand, incurring prohibitive communication costs. Configurations with smaller t (2, 4) increase pipeline depth, inflating the pipeline bubble.

**Pipeline vs. data parallelism (Figure 14).** A 5.9B-parameter GPT model (32 layers, 32 attention heads, hidden size 3,840) is tested on 64 GPUs with microbatch size 1, sweeping (p, d) pairs that multiply to 64 (with t=1). Batch sizes of 32, 128, and 512 are tested.

- For all batch sizes, throughput **decreases monotonically** as pipeline-parallel size increases. At batch size 512, (p=2, d=32) achieves approximately 195 teraFLOP/s, while (p=32, d=2) achieves roughly 145 teraFLOP/s — a 26% reduction.
- The drop is steeper for smaller batch sizes (batch size 32 degrades more severely than batch size 512 at larger p), consistent with the pipeline bubble formula (n-d)/b' from Section 3.3.1.

This confirms Takeaway #2: pipeline parallelism is a necessary cost for memory, not a throughput enhancer. Data parallelism should be preferred wherever the model fits in GPU memory.

**Tensor vs. data parallelism (Figure 15).** The same 5.9B-parameter model is tested sweeping (t, d) pairs (with p=1), with batch sizes of 32, 128, and 512.

- Throughput **decreases sharply** as tensor-parallel size increases. At batch size 512, (t=2, d=32) achieves approximately 190 teraFLOP/s, while (t=32, d=2) achieves roughly 55 teraFLOP/s — a 3.5× gap.
- Larger batch sizes amplify the degradation because tensor-parallel all-reduce happens per-microbatch, and more microbatches per batch means more all-reduce operations. With batch size 512 and microbatch size 1, m = 512/(d·1) = 512/d microbatches, each incurring 2 all-reduce operations per transformer layer.
- Additionally, increasing t reduces per-GPU GEMM dimensions, lowering arithmetic intensity and GPU utilization.

The paper notes that data parallelism alone cannot scale to very large models (GPT-3's batch size of 1,536 limits data parallelism to 1,536 GPUs; 10,000 GPUs were used in practice), making model parallelism necessary despite its throughput cost.

#### Microbatch Size Optimization

Figure 16 shows per-GPU throughput for a 91B-parameter model with (t=8, p=8) on 64 GPUs, sweeping microbatch size from 1 to 8 for batch sizes 128 and 512.

- **Batch size 128:** Throughput peaks at microbatch size 2 (approximately 152 teraFLOP/s), then declines to roughly 138 teraFLOP/s at microbatch size 8. This is the tension from Section 3.4: larger microbatches improve arithmetic intensity but reduce m (number of microbatches), increasing the pipeline bubble fraction (p-1)/m.
- **Batch size 512:** Throughput peaks at microbatch size 4 (approximately 168 teraFLOP/s). The optimal shifts right compared to batch size 128 because with a larger total batch, m = B/(d·b) remains large enough at b=4 to keep the bubble fraction small while still benefiting from improved arithmetic intensity. The paper notes the optimal is "model-dependent" and not universal.

Figure 8 (the estimated throughput from the analytical model of Equation 1) shows the same qualitative shape, confirming that the model captures the dominant tradeoff even though it ignores communication overlap.

#### Activation Recomputation

Figure 17 compares throughput with and without activation recomputation for a 145B-parameter GPT model (80 layers, 96 attention heads, hidden size 12,288) on 128 GPUs with (t=8, p=16), across batch sizes from 1 to 256.

- **Without recomputation:** Maximum batch size is limited to approximately 32 due to memory constraints. Peak throughput at this batch size is roughly 5 sequences/second (the figure uses sequences/second rather than FLOP/s).
- **With recomputation:** Training is possible at batch sizes up to 256. At batch size 32, recomputation reduces throughput by approximately 25% (roughly 7.5 to 5.5 sequences/second) due to the extra forward pass.
- **At batch size 256 with recomputation:** Throughput is approximately 10 sequences/second — roughly 2× higher than the best throughput without recomputation at any batch size. This is because the large batch size dramatically reduces the pipeline bubble (m is large), more than compensating for the recomputation overhead.

The paper recommends checkpointing every 1 or 2 transformer layers based on the analysis in Section 3.5.

#### Scatter/Gather Communication Optimization

Figure 18 shows throughput with and without the scatter/gather optimization for the 175B GPT-3 model on 96 GPUs with the interleaved schedule, across batch sizes from 12 to 60.

- The optimization improves throughput across all batch sizes, with the gap widening from approximately 5 teraFLOP/s (small) at batch size 12 to roughly 15 teraFLOP/s at batch size 60 — an 11% improvement. This is expected because point-to-point communication volume scales with batch size, so the optimization's benefit is larger when communication is more burdensome.

The paper notes that without this optimization, the interleaved schedule is outperformed by the non-interleaved schedule at large batch sizes (not shown in the figure), confirming that the optimization is a prerequisite for interleaving to be beneficial.

#### Operator Fusion

Section 5.8 reports throughput improvements from the fused kernels described in Section 4.2:

- **175B GPT-3 model:** Throughput increases by 19% (from 113 to 135 teraFLOP/s per GPU).
- **530B GPT model:** Throughput increases by 11% (from 133 to 148 teraFLOP/s per GPU).

The smaller relative gain for the larger model is attributed to GEMMs being larger (already saturating the GPU better, leaving less headroom for fusion improvements) and communication being a larger fraction of total time.

#### Inter-Node Communication Bandwidth at Scale

Section 5.9 reports two bandwidth measurements from the trillion-parameter training run on 3,072 GPUs:

- **Pipeline-parallel point-to-point communication:** Effective bisection bandwidth of **892 GB/s**.
- **Data-parallel all-reduce communication:** Effective bisection bandwidth of **12.9 TB/s**.

These numbers characterize the communication load that the system places on the InfiniBand fabric. The paper presents them to emphasize that efficient training at this scale requires high-bandwidth interconnects — slower networks would bottleneck the pipeline and data-parallel communication, degrading throughput from the reported 502 petaFLOP/s.

#### Checkpoint Loading and Saving

Section 5.10 reports that the trillion-parameter model has a checkpoint size of **13.8 terabytes**. During initial checkpoint loading by all 384 nodes (3,072 GPUs), peak read bandwidth reaches **1 TB/s** (the maximum possible from the parallel filesystem). Checkpoint saves reach **40% of peak write bandwidth (273 GB/s)**.

### Ablation Studies and Robustness Checks

- **Interleaving depth (v)**: The paper tests only v=2 (each device assigned two model chunks). Larger interleaving depths (v=4, v=8) are not evaluated. The paper's analytical model predicts bubble reduction as 1/v, but the communication cost also scales with v, so there is presumably a sweet spot. The absence of v>2 experiments means the practical limit of interleaving is not established.

- **Scatter/gather optimization necessity for interleaving**: The paper states (Section 5.3.2) that "without the scatter/gather optimization, the default schedule performs better than the interleaved schedule at larger batch sizes," but does not show this data in Figure 12. The claim is mentioned in text without a supporting figure. This is a missing ablation that would quantify how much of interleaving's benefit depends on the communication optimization.

- **Microbatch size vs. model size**: The paper acknowledges that the optimal microbatch size is model-dependent (Section 5.5) but only shows one sweep (Figure 16 for the 91B model). Table 1 uses different microbatch sizes across model scales, but the paper does not report how these values were chosen or whether they are optimal for each configuration. The analytical model from Section 3.4 is proposed as a proxy, but its predictive accuracy is not quantitatively evaluated against measured throughput.

- **Tensor-parallel size = 8 optimality across model sizes**: The (p, t) sweep in Figure 13 uses only one model size (162.2B parameters). While Table 1 shows t=8 used successfully from 3.6B to 1T parameters, this does not prove t=8 is optimal at all scales — it only shows it works. Smaller models might benefit from t=4 (less communication, still fits within a server); larger models might be forced to t=8 by memory constraints. The paper does not sweep t for models of different sizes.

- **Batch size scaling limits**: Figure 11 shows pipeline scaling with batch sizes 8 and 128, but does not explore intermediate values or establish the minimum viable batch size for a given pipeline depth. The paper's claim that pipeline parallelism requires m ≫ p is qualitative; the exact ratio needed for, say, 80% efficiency is not characterized.

- **Hardware homogeneity assumption**: All experiments use identical DGX A100 nodes with 8 GPUs each. The heuristics (t=8 within a server, pipeline across servers) depend on the specific GPU-to-server ratio and NVLink-vs-InfiniBand bandwidth ratio. The paper does not test sensitivity to different hardware configurations (e.g., 4-GPU servers, different interconnect speeds), making the generalizability of the heuristics uncertain.

- **FP16 vs. other precisions**: All experiments use mixed precision (FP16). The paper does not evaluate BF16, FP32, or FP8. Different precisions would change both memory footprint and arithmetic intensity, potentially shifting the optimal parallel configuration and microbatch size.

- **Convergence impact**: This paper is entirely about training throughput, not model quality. None of the experiments measure validation loss, downstream task accuracy, or convergence behavior with different parallel configurations or microbatch sizes. The paper acknowledges this implicitly by focusing on "strict optimizer semantics" — the parallelism strategies are designed to be numerically equivalent to single-GPU training, so convergence should be unaffected. However, the paper does not empirically verify this equivalence.

- **Weak scaling definition**: The weak scaling experiments (Figures 11, Table 1) increase model size proportionally with GPU count. However, "proportional" here means increasing the number of layers while keeping hidden size and attention heads fixed (Figure 11) or varying all dimensions (Table 1). The paper does not clearly define what constitutes a fair weak-scaling comparison — is it keeping per-GPU compute equal, or keeping the pipeline bubble fraction equal, or something else? This makes the "super-linear scaling" claim in Table 1 somewhat ambiguous: per-GPU throughput improves partly because the models are larger (better GPU utilization) and partly because the architectures change non-uniformly across configurations.

### Critical Assessment

**Claim from the executive summary: PTD-P achieves 502 petaFLOP/s on 3,072 GPUs (52% of peak per GPU) for a 1-trillion-parameter model.**

This is the paper's headline achievement and is directly supported by Table 1 (last row). However, what the paper *measures* versus what it *claims* deserves scrutiny. The FLOP count uses Equation 3, which counts only matrix multiplications in transformer and logit layers and excludes normalization, embeddings, residual additions, and other element-wise operations. The paper acknowledges this is a "lower bound for the true FLOP count." The 52% figure is thus relative to an *underestimated* FLOP denominator — true utilization is lower. How much lower depends on the fraction of FLOPs in non-GEMM operations, which the paper does not quantify. The 52% figure should be interpreted as "at least 52% of theoretical peak for the dominant GEMM operations" rather than "52% of all operations that the GPU executes."

Additionally, the 502 petaFLOP/s aggregate is computed as 163 teraFLOP/s × 3,072 GPUs. This assumes perfect weak scaling from smaller configurations, but the per-GPU throughput in Table 1 varies from 137 to 163 teraFLOP/s across model sizes — the "502 petaFLOP/s" is the instantaneous throughput for the largest configuration, not a sustained average across all configurations.

**Claim: The interleaved schedule improves throughput by 10+% over previously-proposed schedules.**

Supported by Figure 12 for the 175B model with batch size 12 (where the gap is ~21%), but the claim of "10+%" is qualified heavily: the improvement is largest at small batch sizes (where the pipeline bubble dominates) and shrinks to single-digit percentages at large batch sizes. Moreover, the improvement is conditional on the scatter/gather optimization being enabled — without it, the interleaved schedule can be *worse* than the default. The paper's abstract claims "10+% with memory footprint comparable to existing approaches" without mentioning this conditionality. The memory footprint claim is plausible (interleaving doesn't change the total activation storage pattern of 1F1B) but is not empirically demonstrated — no memory measurements are reported for interleaved vs. non-interleaved.

**Claim: Sub-optimal combinations of tensor and pipeline parallelism can lead to up to 2× lower throughput.**

This is supported by Figure 13, where (2, 32) achieves roughly 100 teraFLOP/s and (8, 8) achieves roughly 195 teraFLOP/s at batch size 128 — a 1.95× gap. However, (2, 32) is arguably a straw-man configuration: it uses tensor parallelism across 4 servers (32 GPUs with 8 GPUs per server), which the paper's own analysis predicts will be terrible. A more informative comparison would be between configurations that are all "reasonable" under some criterion (e.g., all configurations that fit the model in memory), to show that even among feasible options, the choice matters. The paper does not establish which of the (p, t) pairs in Figure 13 are memory-feasible for the 162.2B model — some may require activation recomputation or microbatch size adjustments that the paper does not discuss.

**Claim: PTD-P outperforms ZeRO-3 by 70% for 175B and 530B models.**

The 70% figure comes from the comparison at 1,536 GPUs (175B model) and 2,240 GPUs (530B model) in Table 2 and Figure 10, where PTD-P achieves 141 and 159 teraFLOP/s per GPU versus ZeRO-3's 44 and 48 teraFLOP/s — ratios of 3.2× and 3.3×, not exactly 70%. The paper's "70%" appears to refer to the relative difference: (141-44)/141 ≈ 69%. This is a strained way to express a 3.2× ratio.

More importantly, the comparison is asymmetric. PTD-P uses model parallelism (tensor + pipeline) to distribute the model; ZeRO-3 uses only sharded data parallelism. ZeRO-3 *can* be combined with model parallelism (as the paper acknowledges), which would likely improve its scaling behavior, but this combination is not evaluated. The paper also uses the interleaved schedule and scatter/gather optimization for PTD-P but does not apply comparable communication optimizations to ZeRO-3. The comparison demonstrates that pure ZeRO-3 without model parallelism scales poorly — an important finding — but does not establish PTD-P's superiority over the best possible ZeRO-based configuration.

**Claim: Training a trillion-parameter model is practical (approximately 3 months).**

The 84-day estimate depends on assumptions that are not validated: (a) the trillion-parameter model needs 450 billion training tokens — this number is stated as an assumption without derivation or citation; (b) throughput remains stable at 163 teraFLOP/s per GPU for the entire training run (no degradation from checkpointing overhead, hardware failures, or cluster scheduling); (c) the model actually converges in 450B tokens — convergence behavior at this scale was not well-established at the time of publication. The 84-day figure is a lower bound assuming perfect conditions.

**Missing experiments that would strengthen the paper:**

1. **Memory footprint measurements.** The paper discusses memory constraints extensively (it is the primary motivation for model parallelism) but never reports actual memory usage for any configuration. How close to the 80 GB limit are the configurations in Table 1? What fraction goes to parameters vs. activations vs. optimizer states? Without memory data, the claim that "M = t·p should be used so that the model's parameters and intermediate metadata fit in GPU memory" cannot be operationalized — how much headroom is needed?

2. **Interleaving depth sweep (v = 2, 4, 8).** Only v=2 is tested. The analytical model says bubble reduces as 1/v, but communication increases as v. Where is the crossover? Is v=2 optimal, or would v=4 provide further gains with the scatter/gather optimization?

3. **Convergence verification.** A small-scale experiment showing that different (p, t, d) configurations and microbatch sizes produce identical loss curves would validate the "strict optimizer semantics" guarantee. Without this, there is a lingering concern that subtle numerical differences (e.g., different order of floating-point additions in all-reduce) could affect training dynamics at scale.

4. **Hardware sensitivity.** All experiments use DGX A100 with NVLink and HDR InfiniBand. How would the heuristics change on a cluster with 4-GPU servers? With 200G Ethernet instead of InfiniBand? With V100 GPUs? The paper's heuristics are hardware-specific but presented as general principles.

5. **ZeRO + model parallelism.** The paper acknowledges this combination is possible but does not evaluate it, leaving open the question of whether PTD-P's advantages over ZeRO-3 persist when ZeRO is combined with the same model parallelism strategies.

**Bottom line:** The experiments convincingly demonstrate that PTD-P achieves high throughput at scale and that the proposed heuristics (tensor within a server, pipeline across servers) produce good configurations. The interleaved schedule provides real but modest gains under specific conditions (small batch sizes, with scatter/gather enabled). The ZeRO comparison establishes that pure sharded data parallelism scales poorly for very large models, but does not establish PTD-P's superiority over the best possible system. The training time estimates are plausible lower bounds but should be treated as optimistic projections rather than validated predictions.

## 6. Limitations and Trade-offs

### 6.1 Difficulty Estimation Cost Is Unaccounted For in the Throughput Numbers

The entire compute-optimal framework rests on the ability to estimate prompt difficulty *before* deciding how to allocate inference compute. The paper's method for doing so — generating 2,048 samples per question and averaging correctness — is extraordinarily expensive. In Section 3.2, the authors acknowledge:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This is not a minor omission. Generating 2,048 samples per question to bin it into a difficulty quintile consumes far more compute than the largest test-time budgets studied (256–512 generations). The paper reports 4× efficiency gains over best-of-N — but this is computed *after* difficulty is known, without amortizing the cost of learning it. In a realistic deployment where difficulty must be estimated for every query, the total cost is difficulty estimation + strategy execution, and the former dominates. A system that spends 2,048 generations on difficulty estimation to then save a few hundred generations on strategy execution is not more efficient — it is less efficient by an order of magnitude.

The paper partially mitigates this by showing that PRM-predicted difficulty bins (which don't require ground-truth labels) track oracle bins closely (Figures 4, 8, and Appendix C). But this only eliminates the need for *labeled* data, not the need for 2,048 samples. The PRM-based estimation still requires generating and scoring those samples. The authors explicitly flag this as an "exploration-exploitation tradeoff" and suggest future work on training models to predict difficulty directly from question text, but no such model is developed or evaluated. Until this gap is closed, the 4× figure should be treated as an **upper bound on achievable efficiency** conditional on a solved difficulty-estimation problem — not a realized deployment gain.

### 6.2 The Method Provides No Benefit on Hard Problems Where the Base Model Has Near-Zero Capability

Across every method studied — PRM search, iterative revisions, and their compute-optimal combination — the hardest questions (difficulty bin 5) show essentially zero improvement regardless of compute budget. In Figure 3 (right), bin 5 accuracy hovers at 1–3% for all search methods at all budgets. In Figure 7 (right), bin 5 accuracy sits at roughly 2–3% irrespective of the sequential-to-parallel ratio. In the FLOPs-matched comparison (Figure 9), the bin 5 scaling curves are flat near 0–5% across all R values, and the larger model dominates decisively (e.g., hard questions show a −52.9% relative disadvantage for test-time compute vs. the ~14× larger model at R ≫ 1 with PRM search).

This is a fundamental capability bound, not an engineering limitation to be optimized away. Test-time compute can amplify existing capability — it helps the model find correct solutions that exist in its output distribution but are low-probability — but it cannot create capability where none exists. If the base model's pass@1 is near zero on a problem class (pass@1 ≈ 0.5% in bin 5), no amount of search or revision will help, because there are essentially no correct solutions in the proposal distribution to find or refine. The paper is transparent about this (Section 7 takeaway box explicitly states that test-time compute "is most effective on easy-to-medium difficulty problems"), but it means the approach offers **no path forward for genuinely novel or out-of-distribution reasoning tasks**. For such problems, scaling pretraining remains the only viable approach. This sharply bounds the practical applicability of the method: it is a strategy for improving efficiency on problems the model already roughly knows how to solve, not for extending the frontier of what the model can do.

### 6.3 All Results Are on a Single Benchmark (MATH) with a Single Model Family (PaLM 2-S*), Making Generalization Uncertain

The paper's entire empirical contribution — the difficulty-dependent scaling curves, the 4× efficiency gains, the FLOPs-matched comparison, the compute-optimal policy — is demonstrated on exactly one benchmark (MATH, consisting of high-school competition math problems) using exactly one base model family (PaLM 2-S*). The authors state in Section 4 that they "believe this model is representative of the capabilities of many contemporary LLMs," but this belief is not tested. Several aspects of the findings could be model- or domain-specific:

- **PRM quality and over-optimization behavior** depend on PaLM 2-S*'s specific output distribution and error patterns. A model with different calibration properties might exhibit different difficulty-dependent scaling — for instance, if a model's errors are more systematic and predictable, beam search might help on easy problems rather than hurt.
- **Revision model effectiveness** depends on the base model's in-context learning and self-correction capabilities, which vary substantially across model families (the paper itself notes that prompting-based self-correction fails on PaLM 2, motivating the need for fine-tuned revision models).
- **The MATH benchmark** consists exclusively of symbolic reasoning problems with well-defined ground-truth answers. It is unclear whether the core finding — that difficulty-dependent allocation recovers 4× efficiency — generalizes to other reasoning domains (code generation, logical reasoning, scientific QA) or to tasks requiring factual knowledge (where errors may stem from missing knowledge rather than faulty reasoning) or to open-ended generation (where ground-truth correctness is undefined).

The test set of 500 questions, split into five difficulty quintiles of ~100 each, then further split by two-fold cross-validation, means the compute-optimal policy is selected based on ~50 questions per fold per bin. This is a small sample for strategy selection and could introduce variance in the computed-optimal policy. The paper does not report confidence intervals on the scaling curves, making it impossible to assess whether the observed differences between strategies are statistically reliable at this sample size. A practitioner deciding to adopt compute-optimal scaling for a different model or domain has essentially no evidence about whether the method transfers.

### 6.4 The Revision Model Suffers from a 38% Correct-to-Incorrect Reversion Rate, and Revision Training Is Fragile

Section 6.1 reports a significant practical problem with the revision model: approximately **38% of correct answers** produced during a revision chain get subsequently "revised" back to incorrect answers. This is a direct consequence of the training data construction — the model is trained only on sequences where all in-context answers are incorrect followed by a correct target, so it has no signal for what to do when the current answer is already correct. The model learns to *always revise*, even when revision is harmful.

The paper mitigates this with majority voting or verifier-based selection across the entire revision chain (picking the best answer from any point rather than always taking the last revision). But these are imperfect patches: majority voting requires multiple chains to be effective, and verifier-based selection introduces its own error (the verifier is not perfect). A more principled solution — such as training the model to recognize when no revision is needed, or including "no-change" examples in the training data — is not explored.

More concerning is the evidence that revision training methodology is fragile. Appendix K (Figure 16) reports that an attempt to further optimize the revision model using ReST^EM (Singh et al., 2024) **backfired**: additional sequential revisions substantially *hurt* performance, with fully sequential performance dropping to approximately 33.5% compared to approximately 38.5% at the optimal ratio at 256 generations. The authors hypothesize that on-policy data collection amplified spurious correlations in the revision trajectories. This negative result is not a minor footnote — it suggests that the positive revision results depend on specific, carefully controlled choices (offline data construction with edit-distance-based incorrect-correct pairing, specific temperature and sampling parameters) that may not transfer to other settings or survive further optimization. Practitioners seeking to replicate the revision model approach should expect substantial sensitivity to training data construction methodology, with no established recipe for debugging when it fails.

### 6.5 PRM Search and Iterative Revisions Are Studied in Isolation, Not Combined, Leaving a Natural Opportunity Unexplored

The paper decomposes test-time compute into two complementary axes — modifying the proposal distribution via revisions, and modifying the selection mechanism via PRM-guided search — and studies each independently. Section 8 explicitly acknowledges:

> "we did not experiment with PRM tree-search techniques in combination with revisions"

This is not a minor omission — it is a missing evaluation of what appears to be the most natural composition of the two mechanisms. The paper's own analysis shows that revisions and search have complementary strengths across difficulty levels (revisions excel on easy problems where local refinement suffices; search excels on medium problems where global exploration helps). This suggests that a combined system — using the revision model as the proposal distribution within beam search, or using PRM step-level scores to guide which revision branches to pursue — could outperform either approach alone. The paper establishes that each mechanism independently provides 4× efficiency gains over best-of-N; whether those gains are additive, multiplicative, or mutually redundant is an open question the paper does not address.

The current results therefore represent a **lower bound** on what a fully integrated system could achieve. For practitioners, this means the paper's reported numbers (44% accuracy at 256 generations with compute-optimal revisions, ~39% with compute-optimal search) should not be interpreted as the ceiling of what the approach can deliver — but the paper provides no guidance on how much additional gain to expect from combining the two, or what the integration challenges might be (Does the PRM trained on base model outputs transfer to revision model outputs? The paper has partial evidence in Figure 15a that it does not, suggesting combination is non-trivial).

### 6.6 Sequential Revision Strategies Introduce Latency That Is Not Accounted For in the Throughput-Centric Analysis

The paper measures all compute budgets in "generations" — the total number of complete solutions sampled — which is a reasonable proxy for total FLOPs but ignores wall-clock latency. Sequential revisions are inherently serial: each revision depends on the output of the previous one, so a chain of 64 sequential revisions cannot be parallelized. By contrast, parallel best-of-N with N=64 can, with sufficient hardware, execute all 64 generations simultaneously.

This matters because the compute-optimal policy for easy problems (Section 6, Figure 7) often favors purely sequential or highly sequential allocations. For bin 1 questions at 128 generations, performance is essentially flat across all sequential-to-parallel ratios — the system can achieve the same accuracy with a fully sequential chain as with a balanced parallel-sequential mix. But the fully sequential chain takes ~64× longer wall-clock time than a parallel-only approach, since each of the 128 generations must wait for the previous one to complete. For latency-sensitive applications (interactive assistants, real-time reasoning, any user-facing system), this makes the sequential-heavy strategies recommended by the compute-optimal policy potentially unusable regardless of their FLOP efficiency.

The paper does not discuss latency at all. The "generation budget" metric abstracts away the serial-vs-parallel distinction, treating a sequential chain of length N as equivalent in cost to N parallel samples. In FLOP terms this is accurate; in wall-clock terms it is not. A practitioner deploying this system would need to factor in a latency budget alongside the generation budget, potentially arriving at different optimal strategies than those reported. For example, if maximum latency is capped at the time for 4 sequential revisions, the purely sequential strategies that dominate on easy problems in Figure 7 would be infeasible, and the practitioner would need to use more parallel sampling — likely with lower accuracy at the same generation budget. The paper provides no framework for reasoning about this tradeoff.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper reshapes the distributed training landscape by providing a **validated operational blueprint** where previously there were only isolated techniques and untested intuitions. Before PTD-P, the field understood that tensor parallelism, pipeline parallelism, and data parallelism each addressed parts of the large-model training problem — but how to compose them was an open question. Practitioners faced a combinatorial space of (p, t, d) configurations with no systematic guidance about which choices would produce viable throughput, and naive combinations could easily produce systems that were 2× slower than optimal, as the paper empirically demonstrates in Figure 13.

The paper's primary shift is from **"these parallelism techniques exist" to "here is how to combine them, and here is why this specific combination works."** This is an engineering contribution rather than a theoretical breakthrough, but in distributed systems, operational knowledge of this kind — validated at unprecedented scale — often matters more than novel algorithms. The three takeaways (Sections 3.2–3.4) function as a design checklist that replaces trial-and-error with a deterministic recipe: set tensor-parallel size to your server's GPU count, use just enough pipeline parallelism to satisfy memory constraints, and allocate all remaining GPUs to data parallelism. This recipe worked for models spanning three orders of magnitude (1.7B to 1T parameters) on up to 3,072 GPUs, which is strong evidence that it captures the dominant tradeoffs.

The interleaved pipeline schedule introduces a new degree of freedom into pipeline-parallel design. Prior schedules (GPipe, PipeDream-Flush) treated the pipeline bubble as an immutable function of pipeline depth and batch size: reduce the bubble by either shortening the pipeline (limiting model size) or increasing the batch size (potentially infeasible). Interleaving shows that **the layer-to-device mapping itself can be manipulated to reduce bubble size**, decoupling it partially from pipeline depth. The bubble fraction formula `1/v · (p-1)/m` makes this relationship explicit and provides a quantitative knob — the number of virtual stages per device `v` — that future work can tune. This expands the design space of synchronous pipeline schedules beyond what was previously considered, and the paper demonstrates that with appropriate communication optimization (scatter/gather), the tradeoff can be net-positive in the most stressed regime (small batch sizes, where the default schedule's bubble dominates).

The paper's demonstration that 52% of peak FLOP/s is achievable at 3,072 GPUs on a real workload — not a microbenchmark — sets a **concrete performance target** for future systems. Prior to this work, it was unclear whether training throughput would collapse at thousand-GPU scales due to communication overhead, stragglers, or pipeline bubbles. The paper shows that with careful engineering at every level (pipeline schedule, communication topology awareness, kernel fusion, data layout), near-linear weak scaling is achievable with per-GPU utilization that actually *improves* with scale (44% at 1.7B parameters rising to 52% at 1T parameters). This is a counter-narrative to the intuition that distributed training efficiency must necessarily degrade as GPU count grows.

The paper also resolves a latent ambiguity about the respective roles of tensor and pipeline parallelism. Prior work presented them as alternatives — Megatron (Shoeybi et al., 2020) used tensor parallelism, PipeDream (Narayanan et al., 2019) used pipeline parallelism — without clear guidance on when to prefer which. The paper's analytical models and Figure 13 demonstrate that they are **complements, not substitutes**, and that their optimal boundary is determined by hardware topology (NVLink bandwidth vs. InfiniBand bandwidth) rather than model architecture. This clarification changes how system designers think about parallelism: the question is not *which* model parallelism to use, but *where* to draw the boundary between them.

However, this is not a paradigm shift. The paper does not introduce a new parallelism primitive (tensor, pipeline, and data parallelism all predate it), nor does it claim optimality (the heuristics are presented as "work well in practice" rather than provably optimal). It is a **meticulous systems integration and scaling demonstration** that converts theoretical possibilities (training trillion-parameter models) into practical reality (3 months on 3,072 GPUs). Its primary legacy is likely to be the specific configuration recipe and the engineering optimizations that make it work, rather than a conceptual reorientation of the field.

### Follow-Up Research This Work Enables

**Characterizing the communication-computation tradeoff of deeper interleaving (v > 2).** The paper tests only `v=2` (two model chunks per device) and provides an analytical model predicting that bubble fraction scales as `1/v` while communication frequency scales as `v`. Where is the crossover? For a given model size, batch size, and hardware topology, there should be an optimal `v` that balances bubble reduction against communication overhead. A follow-up study would sweep `v` from 2 to (number of layers per pipeline stage) for several model scales on the Selene hardware, measuring both throughput and memory footprint. The paper's scatter/gather optimization makes `v > 2` more viable than it would otherwise be, since inter-node per-message size shrinks to `b s h / (t · v)`. This experiment would establish whether `v=2` is near-optimal or whether further gains are available — and whether the optimal `v` is itself a function of batch size (larger batches may tolerate larger `v` since the bubble is already small).

**Automating (p, t, d, b) configuration with a cost model validated against the paper's heuristics.** The paper explicitly declines to automate configuration search ("we do not automatically explore the search space... but instead suggest heuristics that we found work well in practice"), but the analytical models in Section 3 — pipeline bubble as `(p-1)/m` or `1/v · (p-1)/m`, tensor communication as `l_stage · 8 b s h · (t-1)/t`, data-parallel communication amortized per batch — provide the ingredients for a lightweight cost model. A follow-up would implement a configuration optimizer that takes as input model dimensions (`l`, `h`, `s`, `V`), hardware parameters (GPU memory, NVLink bandwidth, InfiniBand bandwidth, number of GPUs per server), and a batch size `B`, then outputs a recommended `(p, t, d, b, v)` tuple by minimizing estimated iteration time subject to memory constraints. The optimizer's recommendations could be validated against the paper's empirical results (Table 1, Figures 13–16) to test whether it reproduces the heuristic choices (`t=8`, `b=4` for the 91B model, etc.) and generalizes to configurations not tested. A negative result — the cost model failing to predict the paper's empirical throughput patterns — would reveal which second-order effects (communication-computation overlap, launch overhead for small GEMMs, straggler variance) actually dominate at scale.

**Convergence-equivalence verification across (p, t, d) configurations at small scale.** The paper claims that PTD-P preserves "strict optimizer semantics" — meaning the weight updates are mathematically identical regardless of how `(p, t, d)` partitions the computation. However, floating-point arithmetic is not associative, and different parallelization strategies change the order of summation in all-reduce operations and pipeline communication. At the thousand-GPU scale, accumulated numerical differences could produce divergent training trajectories. A controlled experiment would train a modest GPT model (e.g., 1.7B parameters from Table 1) to convergence under several (p, t, d) configurations that all satisfy memory constraints — say, (1, 1, 8), (1, 8, 1), (2, 4, 1), (4, 2, 1) on 8 GPUs — and compare validation loss curves, gradient norms, and final downstream accuracy. This would either validate the paper's implicit claim of numerical equivalence or reveal configuration-dependent convergence behavior that practitioners need to account for. Given the paper's emphasis on strict optimizer semantics as a design constraint (contrasted with PipeDream-2BW's relaxed semantics), this validation is an important missing piece.

**PTD-P + ZeRO: measuring the incremental benefit of sharded data parallelism on top of model parallelism.** The paper's comparison to ZeRO-3 (Section 5.2) treats them as alternatives: PTD-P (model parallelism + data parallelism) versus ZeRO-3 (sharded data parallelism without model parallelism). But these techniques are composable — ZeRO's optimizer state and gradient sharding can reduce memory footprint within each data-parallel replica, potentially allowing smaller `M = t·p` (less model parallelism) and thus a smaller pipeline bubble. A follow-up would implement ZeRO-1 or ZeRO-2 (optimizer state sharding, or optimizer + gradient sharding) within PTD-P's data-parallel dimension, then measure whether this enables (a) training with a smaller model-parallel size for the same model, or (b) larger microbatch sizes for the same model-parallel size, and whether either translates to throughput improvement. The paper's 530B model uses `M = 280` (Table 2); if ZeRO sharding reduced this to, say, `M = 140`, the pipeline bubble would halve, potentially recovering throughput. The experiment would clarify whether ZeRO and PTD-P are substitutes (as the paper's comparison implies) or complements (the composability the paper acknowledges but does not test).

**Stress-testing the heuristics on non-transformer architectures and non-GPT workloads.** The paper's entire evaluation uses GPT-family decoder-only transformers. The tensor-parallel partitioning (Section 2.3) is specific to the transformer architecture's multi-head attention and MLP structure; the pipeline-parallel layer assignment assumes homogeneous repeated blocks. How well do the heuristics transfer to encoder-decoder models (T5), mixture-of-experts architectures (Switch Transformer), or models with heterogeneous layers (e.g., vision transformers with varying spatial resolutions)? A follow-up would implement PTD-P for a T5-style model and a Switch Transformer model, apply the same heuristics (`t` = GPUs per server, pipeline across servers), and measure whether throughput as a fraction of peak degrades. A negative result — the heuristics producing poor throughput on non-GPT architectures — would bound the paper's claimed generality and motivate architecture-specific configuration rules.

**The latency wall: characterizing the end-to-end training time impact of the interleaved schedule's communication overhead.** The paper reports that interleaving improves throughput by up to 10+% (Figure 12) but acknowledges it increases communication frequency by `v`. All throughput numbers are measured in steady state. In a real training run, higher communication frequency means more opportunities for network contention, congestion, and tail latency from shared InfiniBand fabric. A follow-up would run the 175B GPT-3 model with interleaved and non-interleaved schedules for extended periods (hours, not minutes) on a shared cluster, measuring not just mean throughput but the distribution of iteration times (p50, p95, p99). If interleaved training shows higher variance or more frequent straggler iterations due to network contention, the steady-state throughput advantage might not translate to wall-clock training time improvement. This is a classic systems concern — mean vs. tail latency — that the paper's short-duration throughput measurements cannot capture.

### Practical Applications and Downstream Use Cases

**Training the next generation of large language models at frontier scale.** This is the paper's most direct application and the one it was designed for. Organizations training models with hundreds of billions to trillions of parameters — at the time of publication, this meant GPT-3-class and larger models, and today means the GPT-4, PaLM, and Llama-scale models — can adopt the PTD-P configuration recipe directly. The specific takeaway: on clusters of 8-GPU servers with high-bandwidth intra-node interconnects (NVLink or equivalent), set tensor-parallel size to 8 and use enough pipeline stages to satisfy memory constraints, with interleaved scheduling enabled at small-to-moderate batch sizes. The 52% of peak FLOP/s achieved at 3,072 GPUs translates to an estimated 3-month training time for a trillion-parameter model, which the paper argues is practical. For a model half that size (500B parameters), the same architecture would reduce training time proportionally to roughly 6 weeks. The configuration recipe eliminates the need for expensive empirical sweeps over (p, t, d) at each model scale — the heuristics have been validated across three orders of magnitude of model size.

**Enabling large-model research on smaller GPU clusters through efficient model parallelism.** Not every organization has 3,072 A100 GPUs. But the paper's finding that per-GPU throughput actually *improves* with model size (Table 1: 137 teraFLOP/s at 1.7B parameters, 163 teraFLOP/s at 1T parameters) means that even on modest clusters, PTD-P makes it possible to train models that exceed single-GPU memory. A research lab with one DGX A100 server (8 GPUs) can use full tensor parallelism (`t=8`) to fit models up to ~20 billion parameters with reasonable throughput (~44% of peak for the 1.7B model, likely slightly lower for larger models since pipeline parallelism would be needed across servers — but within a single server, tensor parallelism alone handles models up to memory capacity). For a 16-GPU cluster (two DGX servers), the paper's recipe of `t=8, p=2` enables models roughly twice as large as single-server tensor parallelism alone. The key practical insight is that **you should not use pipeline parallelism within a single server** — tensor parallelism is more efficient within the NVLink domain — but as soon as you cross server boundaries, pipeline parallelism becomes the right tool. This gives small-cluster operators a clear rule for when to adopt which technique as they scale.

**Checkpoint optimization for petabyte-scale model storage.** Section 5.10 reports that the trillion-parameter model checkpoint is 13.8 terabytes and that checkpoint saving reaches 273 GB/s (40% of peak filesystem write bandwidth). For production training runs, checkpointing frequency and speed directly affect reliability (more frequent checkpoints reduce lost work on failure) and cost (checkpoint time is non-productive). The paper's reported bandwidth numbers provide a baseline for what is achievable with an all-NVMe parallel filesystem. Organizations setting up training infrastructure can use these numbers to dimension their storage systems: a 3-month training run saving checkpoints every 6 hours produces roughly 360 checkpoints × 13.8 TB = ~5 PB of checkpoint data. The 273 GB/s write speed means each checkpoint takes approximately 50 seconds to write, which is a small fraction of the iteration time at this scale. Teams can use these figures to plan storage capacity and to evaluate whether their filesystem bandwidth is sufficient to avoid checkpointing becoming a training bottleneck — if writes are slower than ~200 GB/s, checkpoint time could begin to eat into training throughput for the largest models.

### When to Prefer This Method

The paper positions PTD-P against two named alternatives: pure tensor parallelism (Megatron) and pure sharded data parallelism (ZeRO-3). The tradeoffs are explicit enough to warrant guidance:

**Prefer PTD-P when:**
- The model exceeds the memory capacity of a single GPU, making pure data parallelism (even with ZeRO sharding) infeasible without some model parallelism. The paper's 530B model requires ZeRO-3 at 640 GPUs with batch size 2,560 just to fit, and still achieves only 138 teraFLOP/s vs. PTD-P's 171 teraFLOP/s on fewer GPUs (Table 2).
- The training cluster consists of multi-GPU servers connected by high-bandwidth intra-node links (NVLink, 600 GB/s per link) and lower-bandwidth inter-node links (InfiniBand HDR, 200 Gbps). The PTD-P recipe exploits this hierarchy by keeping tensor-parallel all-reduce within the server and using cheaper point-to-point pipeline communication across servers. On clusters without a clear intra-node/inter-node bandwidth gap (e.g., all GPUs connected via homogeneous fabric), the tensor-within-server heuristic may not apply.
- The batch size is large enough to amortize the pipeline bubble (`m ≫ p`) but not so large that data-parallel communication becomes the bottleneck. The paper shows pipeline scaling degrading sharply at batch size 8 versus 128 (Figure 11), but does not characterize the degradation for intermediate sizes.

**Prefer ZeRO-3 (without model parallelism) when:**
- The model fits in the aggregate memory of a small number of GPUs (e.g., a 20B model on 4–8 GPUs) and data parallelism alone is sufficient. The paper's ZeRO-3 comparison at small scale (384 GPUs for the 175B model) shows only a 6% gap versus PTD-P.
- The GPU count is modest and per-GPU batch size can remain high enough to overlap communication with computation. ZeRO-3's performance degrades when per-GPU batch size shrinks (the paper shows throughput dropping from 144 to 44 teraFLOP/s as GPUs increase from 384 to 1,536 for the 175B model), so ZeRO is most competitive when GPU count is low enough to keep microbatches reasonably large.

**Prefer pure tensor parallelism (Megatron) when:**
- The entire training run fits within a single multi-GPU server. The paper's results show tensor parallelism is more efficient than pipeline parallelism for intra-node communication, and the analytical model (Section 3.2) quantifies why: all-reduce within NVLink is faster than the pipeline bubble overhead that pipeline parallelism would introduce. If the model fits on 8 GPUs, use `t=8` with no pipeline stages.

These decision rules are implicitly supported by the paper's analytical models and experiments but are not presented as an explicit decision framework by the authors — they are synthesized from the patterns visible across Sections 3 and 5. The paper does not provide the quantitative thresholds (e.g., "at what batch size does PTD-P overtake ZeRO-3?") that would make this a precise engineering decision tool.
