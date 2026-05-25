# NanoFlow: Towards Optimal Large Language Model Serving Throughput

**ArXiv:** [2408.12757](https://arxiv.org/abs/2408.12757)

## 🎯 Pitch

NanoFlow introduces a novel serving framework for large language models (LLMs) that maximizes GPU throughput by overlapping compute-, memory-, and network-bound operations within a single device through fine-grained intra-device pipelining. By leveraging 'nano-batching' and an automated pipeline search, NanoFlow smartly splits and schedules workloads to boost compute utilization, achieving up to 1.91× higher throughput than state-of-the-art systems and reaching 50–72% of the theoretical optimal. This breakthrough matters because it significantly lowers the infrastructure cost and increases the serving capacity for planet-scale AI systems, addressing the urgent need for efficient LLM deployment amid global GPU constraints.

---

## 1. Executive Summary

This paper introduces **NanoFlow**, a novel serving framework that exploits **intra-device parallelism** to overlap heterogeneous operations—compute-bound GEMMs, memory-bound decode attention, and network-bound collectives—within a single GPU by splitting input batches into independent nano-batches. Evaluated on popular models including LLaMA-2-70B, Mixtral 8×7B, and LLaMA-3-8B across real-world traces (ShareGPT, LMSYS-Chat, Splitwise), NanoFlow achieves a **1.91× throughput boost** over state-of-the-art serving systems (vLLM, DeepSpeed-FastGen, TensorRT-LLM) and reaches **50% to 72% of theoretically optimal throughput** across evaluated models (68.5% for LLaMA-2-70B on 8×A100 GPUs). The paper establishes that modern LLM serving is predominantly compute-bound—not memory-bound as commonly assumed—but that existing systems leave substantial throughput on the table by executing operations sequentially, a gap that intra-device overlapping closes only when the system's auto-search engine appropriately navigates the tradeoff between increased weight-loading from nano-batching and recovered compute utilization from pipelining.

## 2. Context and Motivation

### The Core Problem: LLM Serving Throughput Falls Far Short of Hardware Capability

The fundamental question this paper tackles is deceptively practical: **if you have a fixed set of GPUs serving large language models, how close to the hardware's theoretical peak throughput can you actually get?** The answer, it turns out, is surprisingly poor. When the authors measured existing state-of-the-art serving systems—vLLM, DeepSpeed-FastGen, and TensorRT-LLM—on LLaMA-2-70B with 8×A100 GPUs, they found that these well-engineered systems achieve only **22.0%, 22.9%, and 37.8%** of optimal throughput respectively (Figure 7). In other words, even the best existing system leaves more than 60% of the hardware's compute capacity idle during end-to-end serving.

This gap matters enormously because of the scale at which LLM serving operates. As the authors note in Section 1, ChatGPT has over 200 million weekly users, API usage doubled after GPT-4o Mini's release, and reports indicate "tens of thousands of GPUs continuously serve hundreds of millions of users." At this scale, a 2× improvement in throughput directly halves the number of GPUs required—translating to millions of dollars in hardware savings and reduced energy consumption. Throughput, measured as tokens per device per second, has consequently become "a key factor in reducing serving costs" (Section 1).

The gap between actual and optimal throughput is not merely an engineering oversight. It stems from a fundamental architectural mismatch in how the field conceptualizes LLM inference workloads versus how modern hardware and models actually behave. The paper identifies this mismatch along two dimensions: a pervasive misconception about what resource constrains LLM serving, and a hidden cost to the way systems orchestrate heterogeneous operations.

### The Misdiagnosis: LLM Serving Is Commonly Assumed Memory-Bound

Perhaps the most significant intellectual contribution of the paper is its challenge to a widely held assumption. The field has largely believed that LLM serving is **memory bandwidth-bound**, meaning that the rate at which data can be moved from GPU memory to compute units limits throughput, not the rate at which computations can be performed. This belief is not unreasonable given several well-known characteristics of transformer inference (Section 1):

- **Massive model sizes.** GPT-3 with 175B parameters requires 5×A100 80GB GPUs just to store weights in 16-bit precision. Each inference iteration loads the entire model weight set.
- **KV-cache memory pressure.** The key-value cache that stores per-request attention state scales quadratically with context length, and the authors note that "its size can surpass the size of model weights." Each decode iteration loads a unique KV-cache per sequence on top of the shared model weights.
- **Token-level decoding.** The decode phase generates only one token per sequence per iteration, yet loads both the full model weights and the full KV-cache—a seemingly catastrophic ratio of memory traffic to useful computation.

If this memory-bound diagnosis were correct, the optimal throughput formula would be dominated by memory bandwidth, and efforts to improve throughput should focus on reducing memory movement (e.g., through quantization, KV-cache compression, or weight sharing). The paper shows that this diagnosis is **wrong for most modern LLM workloads and hardware configurations**.

The authors systematically re-examine the cost model in Section 3. They derive the ratio `TR = T_Mem / T_Compute`, which captures whether memory time or compute time dominates in an iteration. When `TR > 1`, memory is the bottleneck; when `TR < 1`, compute is the bottleneck. Equation 4 shows:

$$TR = \frac{T_{Mem}}{T_{Compute}} \approx \frac{\text{Compute}}{\text{MemBW}} \cdot \frac{\text{MemSize}}{P_{model}} \cdot \frac{1}{2B_{dense}}$$

What makes `TR` drop below 1—pushing the workload into compute-bound territory—is a combination of factors that were less prominent when the memory-bound assumption crystallized:

**Factor 1: Growing batch sizes from grouped query attention (GQA).** Modern models (LLaMA-3, Qwen2, Mistral) widely adopt GQA, where multiple attention heads share a KV-cache. This means that given the same GPU memory budget for KV-cache, the system can hold **more decode requests simultaneously**. The paper makes this concrete: serving LLaMA-2 70B on 8×A100 GPUs with GQA yields a maximum decode batch size on the order of 1024 requests, versus only 256 for a comparable non-GQA model—a 4× difference. Combined with prefill tokens in the batch, the dense batch size `B_dense` (which feeds into GEMM operations) reaches 2048 or more. Since batching amortizes weight loading cost across more tokens, the memory I/O per token drops, shifting the bottleneck to compute.

**Factor 2: Increasing model sizes.** As `P_model` grows while hardware memory characteristics (MemSize, MemBW) remain fixed for a given GPU generation, compute time grows proportionally faster than memory time. The paper's Table 1 shows that across GPU generations from V100 to B200, the ratio `Compute/MemBW` has increased from 139 to 281—modern GPUs have grown their compute capacity faster than their memory bandwidth. Larger models exploit this growing compute headroom.

**Factor 3: Batching prefill with decode.** When prefill and decode phases are served together (as in continuous batching), they share model weight matrices and can be combined into a single dense batch, further increasing `B_dense` and amortizing weight loading. The prefill phase processes all input tokens at once, contributing substantial compute work relative to its memory footprint.

Figure 3 visualizes this shift. For LLaMA-2 70B with 8×A100 GPUs across realistic workload traces (Splitwise, LMSYS-Chat, ShareGPT), `TR` values range from 0.07 to 0.37—meaning compute time is 3× to 14× longer than memory time. The workload is **deeply compute-bound**. Only in the extreme case of LLaMA-3 8B on a single A100 with 512-token input and 1024-token output does `TR` approach 1 (balanced), and even then compute is slightly dominant (Figure 3, bottom-left cell).

The network comparison in Figure 2 paints a similar picture: the ratio `T_Net / T_Compute` is consistently below 0.5 across modern GPUs and model sizes, meaning compute dominates network traffic as well. The high bandwidth of NVLink and comparable interconnects keeps network time well below compute time for tensor-parallel configurations.

The paper validates these analytical derivations empirically in Table 2: for LLaMA-2 70B with a dense batch of 2048 on 8×A100 GPUs, the estimated compute time across all operations is 114.17 ms versus 45.09 ms memory time and 31.33 ms network time—compute is the binding constraint by a factor of more than 2.5×.

This reclassification from memory-bound to compute-bound is not merely academic. It fundamentally changes what an optimizer should target. If the workload is compute-bound, the objective is to **maximize compute utilization**, and memory or network operations can be hidden behind compute as long as they don't delay it. This insight directly motivates NanoFlow's overlapping strategy.

### Where Existing Systems Fall Short: Sequential Execution of Heterogeneous Operations

Even accepting that LLM serving is compute-bound, one might expect existing serving engines—which already use aggressive batching, optimized kernels (CUTLASS), and advanced memory management (PagedAttention)—to achieve high compute utilization. The paper shows they do not, and the reason is subtle.

Figure 4 provides the visual diagnosis. In existing serving frameworks (SGLang, vLLM, DeepSpeed-FastGen), the operations within a transformer layer are executed **sequentially** on a single large batch within a GPU. The sequence looks like:

1. KQV generation (compute-bound GEMM)  
2. Decode attention (memory-bound GEMV loading KV-cache)  
3. Network all-gather for attention outputs (network-bound)  
4. O-projection (compute-bound GEMM)  
5. Network all-gather for O outputs (network-bound)  
6. Up, Gate, Down projections (compute-bound GEMMs)  
7. Network all-reduce (network-bound)  

At any given moment during this sequence, the GPU is using only one type of resource—compute, memory bandwidth, or network bandwidth—while the other two sit idle. The figure marks these idle periods as "WASTED." The result: even though each individual operation achieves **~80% utilization** of its bottleneck resource (Section 6.5 notes this for the non-overlapping baseline), the **total compute utilization across the pipeline is only 40%**. The ~80% → 40% drop is the direct consequence of pipeline bubbles—gaps where compute units are idle because the current operation (e.g., decode attention) is memory-bound and cannot keep them fed, or because the system waits for network synchronization.

This is the central inefficiency NanoFlow targets. The problem is not that individual operations are poorly implemented—they're well-tuned. The problem is that the **heterogeneity** of operations (different resource bottlenecks) is ignored at the scheduling level. Existing systems treat all operations as if they belong to a single monolithic execution stream, when in fact they make disjoint demands on GPU resources that could be serviced simultaneously.

### Prior Approaches and Their Limitations

The paper situates its contribution within a landscape of prior optimization work, each operating at a different granularity but none addressing the same gap:

**Request-level batching (Orca, vLLM).** Orca introduced continuous batching at the iteration level, refilling the on-the-fly batch to maximize batch size and keep the GPU fed. vLLM contributed PagedAttention for efficient KV-cache memory management, enabling larger effective batch sizes. These are critical advances—they increase `B_dense`, which as we saw pushes the workload toward compute-bound territory. But they do nothing about pipeline bubbles within a batch. Even with the largest possible batch, the sequential execution pattern in Figure 4 persists, and compute utilization remains constrained by the memory-bound and network-bound operations that run between GEMM calls.

**Phase-level scheduling (Splitwise, DistServe).** These works disaggregate prefill and decode phases into separate clusters or GPU pools, recognizing that the two phases have different resource profiles. Prefill is compute-intensive, decode is memory-intensive. By separating them, each cluster can be optimized for its dominant resource. While this improves overall efficiency, it is orthogonal to the intra-device pipeline problem—within each phase, the same sequential execution pattern applies. A decode-only cluster still alternates between compute-bound KQV projections and memory-bound attention, leaving compute idle during attention operations.

**Chunked prefill (Sarathi-Serve, DeepSpeed-FastGen).** These systems split prefill requests into smaller chunks and interleave them with decode tokens, forming mixed batches that are more regular in size and more compute-heavy. This smooths out the prefill-decode imbalance and improves utilization. But again, within a mixed batch, operations run sequentially—the same pipeline bubbles remain.

**Operation-level parallelism (Rammer, Unity, ASPEN, Welder).** The closest conceptual relatives to NanoFlow are systems that break the boundary between operations to enable finer-grained parallelism. Rammer remaps operations into different functional units on the GPU. Unity combines parallelism with algebraic transformations of operations. ASPEN and Welder go further, constructing tile-level dataflow graphs that compile operations into a single fused execution plan, exploiting overlap at the tile granularity.

However, these prior operation-level systems face two limitations that NanoFlow overcomes. First, they assume operations have similar resource demands—they don't explicitly model or exploit the fact that different operations are bottlenecked by different resources. NanoFlow's key insight is precisely that heterogeneous resource bottlenecks create overlapping opportunities. Second, these systems require recompilation from scratch for each model architecture, which is labor-intensive and brittle. NanoFlow instead uses an auto-search procedure that profiles kernels once and then solves a mixed-integer linear program to generate the pipeline, making it adaptable to new models without manual rewriting.

**Specific operation optimizations (FlashAttention, quantization).** Many works improve individual kernels—FlashAttention fuses attention computation to reduce memory traffic, quantization reduces weight and activation sizes, speculative decoding generates multiple tokens per iteration. These are complementary to NanoFlow: they shift the absolute values in the cost model but don't address the sequential execution pattern. NanoFlow could (and the authors note it should) incorporate these improved kernels by feeding their performance profiles into auto-search.

### How This Paper Positions Itself

NanoFlow positions itself at an unexplored intersection. It is not a replacement for existing serving systems but a **complementary technique that can be integrated into them**. The existing systems handle the macro-level concerns: request scheduling, KV-cache memory management, inter-device parallelism (tensor and pipeline parallelism), and kernel implementation. NanoFlow handles the micro-level concern: within a single device, given the batch that the macro-scheduler has formed, **how should operations be reorganized to keep the compute units busy?**

The paper frames this as **intra-device parallelism**—parallelism within a single GPU rather than across GPUs. Unlike tensor parallelism (which splits operations across devices) or pipeline parallelism (which splits layers across devices), intra-device parallelism splits batches into nano-batches so that heterogeneous operations processing different nano-batches can overlap. This is a new axis in the parallelism design space, situated below the inter-device parallelism that prior systems optimize.

The paper also positions itself against the memory-bound assumption head-on. Rather than accepting that memory bandwidth is the unavoidable bottleneck and trying to reduce memory movement, it argues that **the bottleneck is compute, and the goal should be to hide memory and network operations behind compute**. This inversion is what makes nano-batching viable: splitting into nano-batches increases weight loading (bad for memory traffic) but enables overlapping (good for compute utilization). In a memory-bound world, this tradeoff would be negative. In the compute-bound world that the paper empirically demonstrates, the tradeoff is positive—the additional memory I/O can be hidden behind compute, and the net effect is higher throughput.

The auto-search engine is central to this positioning. The paper does not claim to have found a single optimal pipeline design for all models. Instead, it provides a **methodology** for automatically generating the pipeline given a model architecture and hardware configuration. This is important because the optimal nano-batch sizes and overlap patterns depend on the specific ratios of compute, memory, and network demands in a model—ratios that vary across architectures. The two-stage MILP approach (Section 4.1) makes this methodology concrete and automated.

Finally, the paper positions its contribution as **measurement-driven and validated**. The analytical cost model is not presented as an abstract framework but is validated against real profiling data (Table 2). The throughput gains are measured on real hardware with real workload traces, not simulated. The auto-search is shown to generalize across diverse models—from 8B dense models to 8×7B mixture-of-experts models to 70B dense models—without manual tuning (Figure 11). This empirical rigor is essential because the central claim—that LLM serving is compute-bound—contradicts a widespread belief and requires convincing evidence.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper presents **NanoFlow**, an end-to-end LLM serving runtime that orchestrates GPU operations at sub-batch granularity to overlap compute-bound, memory-bound, and network-bound work within a single device. The problem it solves is that existing serving systems execute operations sequentially, leaving GPUs with ~40% compute utilization despite the workload being fundamentally compute-bound. NanoFlow's solution is to split input batches into **nano-batches** — independently processable slices — and then co-schedule operations on different nano-batches so that while one part of the GPU is stalled on memory access (e.g., loading KV-cache for decode attention), another part is crunching through matrix multiplications (e.g., GEMMs for feed-forward projections), thereby recovering most of the idle compute capacity.

### 3.2 Big-Picture Architecture

The system has three major components, connected in a design-time-to-runtime pipeline:

1.  **Cost Model and Workload Classifier (Section 3):** An analytical framework that takes hardware specs (GPU FLOPs, memory bandwidth, network bandwidth), model config (parameter count, hidden dimension), and workload statistics (batch size, input/output lengths) as input, and outputs a classification: is this serving scenario compute-bound, memory-bound, or network-bound? This is what justifies the entire approach by proving that for modern LLMs, compute is the bottleneck.

2.  **Auto-Search Engine (Section 4.1):** An offline optimization procedure that takes the model's operation dependency graph, kernel performance profiles, and interference measurements as input, and outputs a concrete execution pipeline — a schedule specifying how many nano-batches to create, what size each should be, in what order nano-operations execute, and what fraction of GPU resources each receives. This runs once per model architecture or significant workload change.

3.  **NanoFlow Runtime (Section 4.2):** The online serving component that executes the auto-generated pipeline. It forms batches by combining prefill and decode requests, chunks them into the nano-batch sizes prescribed by the schedule, launches CUDA kernels according to the prescribed ordering and resource allocation, manages KV-cache with asynchronous offloading to CPU/SSD, and orchestrates the CPU-side scheduling asynchronously to hide overhead.

Information flows as follows: at design time, the cost model validates that the workload is compute-bound → the auto-search engine profiles kernels and solves a two-stage mixed-integer linear program to produce a pipeline schedule → at runtime, incoming requests are formed into dense batches, split into nano-batches, and executed according to the schedule, with compute-bound GEMMs running concurrently with memory-bound decode attention and network-bound collectives on different nano-batches.

### 3.3 Roadmap for the Deep Dive

The explanation of NanoFlow's technical approach follows the logical construction of the system, building from motivation to mechanism to optimization to execution:

- **First, the cost model (Section 3.2) and its key ratio `TR` (Equation 4):** This is the analytical foundation. Understanding why `TR < 1` for modern workloads is what makes the entire overlapping strategy viable — without it, nano-batching's increased weight loading would be strictly harmful.

- **Second, the nano-batching concept and the intra-device parallelism model:** What exactly gets split, how, and what "overlapping" means physically on the GPU. This establishes the core mechanism before diving into how to configure it.

- **Third, kernel profiling and interference modeling (Section 4.1.1):** The auto-search engine needs to know how fast each operation runs at each batch size and how operations slow down when they run concurrently. This grounding in real measurements is what makes the pipeline search practical.

- **Fourth, the two-stage auto-search procedure (Sections 4.1.2 and 4.1.3):** Stage I determines the pipeline structure (number of nano-operations, their batch sizes, and ordering) assuming no interference. Stage II refines by allocating GPU resources and modeling slowdowns. The two-stage decomposition is the key engineering insight that makes the search tractable.

- **Fifth, the runtime system (Section 4.2):** How batch formation, asynchronous scheduling, and KV-cache management translate the static pipeline into an efficient online serving system. This closes the loop from analytical model to deployed system.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems paper** whose core idea is that splitting input batches into finer-grained nano-batches enables overlapping heterogeneous operations within a single GPU, and that an auto-search procedure can automatically find the optimal split sizes, ordering, and resource allocation to maximize compute utilization under the constraints of kernel interference.

---

#### The Cost Model and the Compute-Bound Classification

The paper constructs an analytical cost model to determine which resource — compute, memory bandwidth, or network bandwidth — defines the throughput ceiling for a given (hardware, model, workload) configuration. This model does not directly produce a pipeline; it provides the justification that compute is the binding constraint, which in turn makes overlapping a net positive despite the extra memory I/O from nano-batching.

**Memory perspective.** The time required to load the entire device memory content into GPU caches and registers for one iteration is:

$$T_{mem} = \frac{\text{MemSize}}{\text{MemBW}}$$

where `MemSize` (GB) is the aggregate GPU memory capacity holding model weights and KV-cache, and `MemBW` (GB/s) is the aggregate GPU memory bandwidth.

**What it computes:** the minimum time any iteration must spend on memory access if the entire working set (weights plus KV-cache for all active requests) must be loaded from device memory. This assumes the working set is large enough that caching across iterations is infeasible due to long reuse distance — a standard assumption in modern serving engines where the KV-cache for hundreds of decode requests plus model weights far exceeds cache capacity.

**Why this form:** it captures the physical constraint that even with perfect compute scheduling, memory-bound operations cannot execute faster than the rate at which data can be moved. MemSize/MemBW is the lower bound on memory time per iteration when operating at maximum batch size (the largest batch that fits in GPU memory). This is a throughput-oriented assumption: the system always processes as many requests as memory allows.

**Compute perspective.** The time required for the dense operations (GEMMs — general matrix multiplications that form the bulk of transformer computation) is:

$$T_{Compute} \approx \frac{2 \cdot B_{Dense} \cdot P_{Model}}{\text{Compute}}$$

where `B_Dense` is the total token batch size for dense operations (including both decode tokens from hundreds of requests and prefill tokens from one or a few requests being chunked in), `P_Model` is the total number of model parameters, and `Compute` (GFLOP/s or TFLOP/s) is the aggregate GPU compute capacity in the relevant precision.

**What it computes:** a first-order approximation of the compute time for all GEMM operations across all transformer layers. The factor of 2 accounts for each weight element being involved in one multiplication and one addition (standard FLOP counting for matrix multiplication as `2 * M * N * K`). The product `B_Dense * P_Model` captures that each token in the batch interacts with every model parameter once per layer. Dividing by compute capacity gives the execution time at peak utilization.

**Why this form:** the model explicitly ignores attention operations because they contribute negligible FLOPs relative to dense operations — the paper validates this empirically (Table 2 shows decode attention at 3665.9 GFLOP vs. UG projection at 153931.6 GFLOP). The approximation `∑N_w K_w ≈ P_Model` collapses all weight matrix dimensions into a single scalar, trading precision for analytical tractability without losing the dominant scaling behavior.

**Network perspective.** For tensor-parallel configurations where model weights are sharded across multiple GPUs, collective communication primitives are required to synchronize activations after each operation. The time required is:

$$T_{net} \approx 4 \cdot \frac{N_{GPU} B_{Dense} D_{model} S_{type} L}{\text{NetBW}}$$

where `N_GPU` is the number of GPUs in the tensor-parallel group, `D_model` is the hidden dimension size, `S_type` is the number of bytes per element (e.g., 2 for FP16), `L` is the number of transformer layers, and `NetBW` (GB/s) is the aggregate GPU interconnect bandwidth.

**What it computes:** the total data movement across all layers for one GPU's network operations. Each layer requires two AllGathers and one AllReduce (or two AllReduces depending on partitioning), which collectively transfer approximately 4 times the activation size per layer per GPU. An AllReduce requires gathering outputs, performing local summation, and broadcasting results, thus transferring activations twice; an AllGather transfers once. The factor `N_GPU * B_Dense * D_model * S_type` is the per-layer activation size for one GPU, and multiplying by `L` gives total bytes per iteration.

**Why this form:** the key simplification is treating all network collectives as equivalent in total bytes transferred — the paper notes (Section 4.1.2 constraints on operation transformations) that AllGather can be converted to AllReduce with different weight partitioning, so the aggregate data movement is the right metric rather than per-operation distinctions.

**The central classification ratio `TR` (Equation 4).** The cost model then compares memory time to compute time to determine the bottleneck:

$$TR = \frac{T_{Mem}}{T_{Compute}} \approx \frac{\text{Compute}}{\text{MemBW}} \cdot \frac{\text{MemSize}}{P_{model}} \cdot \frac{1}{2B_{dense}}$$

**What it computes:** the ratio of memory-bound time to compute-bound time for a single iteration. When `TR > 1`, the iteration spends more time moving data than computing — the workload is memory-bound. When `TR < 1`, compute dominates — the workload is compute-bound. When `TR ≈ 1`, the two are balanced.

**Why this form:** it cleanly separates hardware factors from model factors from workload factors. The term `Compute/MemBW` is purely hardware — it captures the compute-to-bandwidth ratio that has been growing across GPU generations (from 139 for V100 to 281 for B200 per Table 1, meaning each new generation adds more compute than memory bandwidth). The term `MemSize/P_model` captures the model's memory footprint relative to capacity — larger models increase `P_model` and drive `TR` down (more compute per byte of memory). The term `1/(2B_dense)` captures batching effects — larger batches amortize memory I/O across more tokens and drive `TR` down. The paper measures `TR` for realistic configurations and finds values between 0.07 and 0.37 (Figure 3), meaning compute time is 3× to 14× longer than memory time. This is the justification for the entire NanoFlow approach.

**The network-to-compute ratio.** The paper also derives:

$$\frac{T_{Net}}{T_{Compute}} = \frac{2 D_{model} L}{P_{model}} \cdot \frac{N_{GPU} \text{Compute}}{\text{NetBW}/S_{type}}$$

and notes that `2 D_model L / P_model ≈ 1/(6 D_model)` because `P_model ≈ 12 D_model^2 L` for standard transformer architectures. For `D_model > 4096` (typical for models requiring tensor parallelism) and `N_GPU * Compute / (NetBW/S_type)` in the range `10^4` to `10^5` for modern datacenter GPUs, this ratio is below 1 — network is not the bottleneck either. Figure 2 visualizes this across models and hardware.

**Optimal throughput (Equation 5).** Under the compute-bound classification, the theoretically optimal throughput (tokens/second/GPU) is simply the compute capacity divided by the compute per token:

$$\text{Throughput}_{optimal} = \frac{B_{Dense}}{T_{Compute}} = \frac{\text{Compute}}{2 P_{Model}}$$

**What it computes:** the maximum possible token throughput if compute units are 100% utilized. For LLaMA-2 70B on 8×A100 GPUs, the profiled peak FP16 compute is 280 TFLOPS, and `P_Model` is 70B, giving `Throughput_optimal = 280 × 10^12 / (2 × 70 × 10^9) ≈ 2000` tokens/second/GPU (the paper quotes 1857, reflecting the fact that the profiled peak is slightly lower than theoretical peak due to real-world kernel efficiency).

**Why this form:** it reveals a crucial property: **optimal throughput depends only on aggregate compute capacity and model size, not on memory size, bandwidth, input/output lengths, or batch size**. All those other factors influence how close you can get to optimal, but the ceiling is set by compute alone. This is what makes the 22-38% achieved by existing systems such a large gap to close — the hardware is capable of nearly 3× more throughput than what is being achieved.

---

#### The Intra-Device Parallelism Mechanism

The central mechanism in NanoFlow is **nano-batching**: splitting a single large input batch into multiple smaller **nano-batches** and duplicating each operation across **nano-operations**, where each nano-operation processes exactly one nano-batch. The key enabling observation is that nano-batches are **independent** — they share no data dependencies — because each request's computation within a transformer layer is independent of other requests in the same layer. This independence allows different nano-operations operating on different nano-batches to execute **concurrently** on the same GPU.

The paper uses a concrete example to illustrate. Consider the original Up projection (a GEMM in the feed-forward network) operating on a batch size of 2048 tokens. NanoFlow might split this into two nano-operations UP1 and UP2, operating on batches of tokens 0-768 and 768-2048 respectively, both performing the exact same matrix multiplication with the same weight matrix `W_up` but on different input slices. These two nano-operations have no dependency, so they can be scheduled independently.

The overlapping pattern emerges because different operation types are bottlenecked by different GPU resources:

- **Compute-bound operations** (dense projections like KQV generation, O-projection, Up/Gate/Down): These are GEMM kernels whose execution time is dominated by the rate at which the GPU's tensor cores or CUDA cores can perform floating-point operations. Their throughput scales with the batch size (more tokens = more work for the same weight loading cost), and at large batch sizes they achieve high compute utilization.

- **Memory-bound operations** (decode attention): These load the KV-cache from GPU memory — a unique, per-request state that cannot be amortized across tokens. The GEMV (matrix-vector) or batched GEMV kernels used for decode attention are limited by memory bandwidth, not compute throughput. Even a small fraction of the GPU's compute units can saturate the memory bandwidth for these operations.

- **Network-bound operations** (AllGather, AllReduce): These wait on data transfer over NVLink or comparable interconnects, using minimal GPU compute or memory bandwidth.

Because these operation types consume different hardware resources, they can execute simultaneously without competing for the same bottleneck. The paper's pipeline design (Figure 6) shows, for example, that while KQV1 (compute-bound GEMM on nano-batch 1) executes, DecAttn4 (memory-bound decode attention on nano-batch 4) and network communications for earlier operations can all run concurrently. The GPU's compute units, memory controllers, and NVLink engines are all active simultaneously — eliminating the pipeline bubbles (WASTED regions in Figure 4) that plague sequential execution.

**The cost of nano-batching.** Splitting into nano-batches is not free. Each nano-operation loads the full model weights independently — there is no weight sharing across nano-operations because they execute as separate kernel launches. If the original batch loaded each weight once (amortized across all tokens), nano-batching with two nano-operations loads each weight twice. This is why nano-batching would be destructive in a memory-bound regime: doubling memory traffic would directly hurt throughput. In the compute-bound regime that the cost model establishes, the extra memory traffic can be **hidden behind compute** — the memory controllers handle the additional loads during the time when compute units would otherwise be idle waiting for the previous operation to finish.

---

#### Kernel Profiling and Interference Modeling

The auto-search engine needs accurate performance predictions for individual kernels and for kernel combinations. This profiling happens offline, once per (model architecture, hardware) configuration.

**Determining maximum dense batch size.** Before profiling, NanoFlow calculates the largest `B_dense` that fits in GPU memory given the model's KV-cache requirements. This is the upper bound for all batch sizes in profiling. For LLaMA-2 70B on 8×A100 GPUs, this maximum is 2048 — constrained by the KV-cache memory capacity after accounting for model weights.

**Interference-free kernel profiling.** NanoFlow profiles each operation type (GEMM, GEMV, network collectives) in isolation for every discrete batch size from 128 to the maximum dense batch size, in multiples of 128 (since 128 is "a hardware-friendly shape for GEMM tiling"). For each (kernel, batch size) combination, NanoFlow explores all possible kernel implementations, varying:
- Number of thread blocks
- Number of warps per block
- Tile size (for GEMM tiling parameters)

The system identifies the implementation with the shortest execution time — the **best interference-free implementation**. The output is a mapping from `(kernel, batch_size)` to the optimal implementation and its execution time `D_best`.

**Why this exhaustive profiling is necessary.** Different batch sizes benefit from different tiling strategies on NVIDIA hardware. A GEMM with batch size 128 has different optimal tile dimensions than one with batch size 2048 because the shape of the matrix multiplication `[batch, hidden_dim] × [hidden_dim, intermediate_dim]` changes the arithmetic intensity. NanoFlow profiles the complete space because the nano-batch sizes in the final pipeline are not known in advance — they are outputs of the auto-search, and the search needs accurate time predictions for any candidate size.

**Kernel interference profiling.** When two kernels run concurrently on a GPU, they experience **slowdown** relative to running in isolation. This interference arises from contention for:
- **Execution units** (CUDA cores, tensor cores): both kernels compete for the same compute resources.
- **Caches** (L1, shared memory, L2): one kernel's data may evict the other's working set.
- **Memory bandwidth**: both kernels issue memory requests, potentially saturating the memory controllers.

Crucially, NVIDIA GPUs do not provide explicit software control over how these resources are divided between concurrent kernels. The paper cannot directly assign, say, "60% of compute units to kernel A and 40% to kernel B." Instead, it must rely on **empirical profiling** to understand the performance consequences of co-scheduling.

**The GEMM-centric resource proxy `R`.** The paper defines a proxy `R` for GPU resource allocation, where `R_A` is the fraction of peak GEMM performance that kernel A achieves when running concurrently with kernel B, normalized to A's best interference-free performance. If a GEMM kernel achieves 40% of its interference-free FLOP/s when co-scheduled, then `R_A = 0.4`. The paper then assumes the remaining resources go to the co-scheduled kernel B, with `R_B = 1 - R_A`, and measures B's performance as a fraction of B's interference-free best — this fraction is denoted `P_B`.

**What `P` captures.** The normalized performance `P` accounts for the fact that different kernel types lose performance differently when resources are constrained. A GEMV kernel (memory-bound) might retain 80% of its peak performance even when `R = 0.4` of the GPU's resources are allocated to a co-scheduled GEMM, because GEMV was never compute-limited to begin with. Conversely, giving the GEMV kernel higher priority (low `R` for GEMM) might drastically slow the GEMM while providing minimal throughput gain to the GEMV.

**Why this formulation matters.** The `R → P` mapping is effectively an **exchange rate** between compute utilization (what the system wants to maximize) and memory/network throughput (what must be maintained to avoid stalling the pipeline). Table 3 shows concrete numbers from the profiling: at `R = 0.2` (GEMM gets 20% of its peak), GEMV achieves `P = 0.3` (30% of its peak), and network achieves `P = 0.5`. At `R = 0.8`, GEMV achieves `P = 0.85`, and network achieves `P = 0.9`. These are non-linear tradeoffs that the auto-search must navigate.

**Reducing the profiling space.** Exhaustively profiling all pairs of kernel implementations would require millions of measurements because each operation type has "a few hundred kernel implementations" and pairings multiply this combinatorially. NanoFlow applies two simplifications:

1. **Limit thread blocks for non-GEMM kernels:** GEMV and network kernels are restricted to 8-128 thread blocks (in steps of 8), because 128 blocks are sufficient to saturate their performance — more blocks would only increase scheduling overhead without meaningful throughput gain.

2. **Filter inefficient GEMM implementations:** The system excludes GEMM kernels with longer execution times that also use more thread blocks, retaining only Pareto-optimal implementations.

3. **Pairwise-only interference analysis:** Rather than profiling three-kernel combinations simultaneously, NanoFlow profiles only compute-memory and compute-network pairs, and assumes the pairwise `R → P` mappings hold when three kernels overlap. This is an approximation validated by the sensitivity analysis showing "standard deviation within 5% of the mean."

**The resulting resource mapping table (Table 3).** After profiling and filtering, NanoFlow produces a lookup table mapping resource allocation `R` (fraction of compute resources assigned) to normalized performance `P` for each kernel type. This table is the key input to Stage II of auto-search. The paper notes that this mapping is "consistent" across different GEMM shapes and batch sizes (standard deviation within 5%), so a single table suffices for all nano-operations in the pipeline.

---

#### Auto-Search Stage I: Pipeline Structure Search

The first stage determines the **macroscopic structure** of the pipeline: how many nano-batches to create, what size each nano-batch should be, and in what order nano-operations execute. This stage **ignores kernel interference** — it assumes each nano-operation achieves its interference-free best time — which makes the search space tractable. Stage II will later correct for interference.

**Inputs to Stage I:**
- `B_dense`: the maximum dense batch size (e.g., 2048)
- The operation dependency graph from the model's PyTorch implementation (which operations must finish before which others can start)
- The interference-free kernel performance profiles from Section 4.1.1: `(kernel, batch_size) → D_best`

**Outputs of Stage I:**
- `N`: the number of nano-operations for each original operation
- `b_i`: the batch size of each nano-operation `i`
- `order`: the execution order of nano-operations, respecting both parent-operation dependencies and the independence of non-overlapping nano-batches

**Formulation as mixed-integer linear programming (MILP).** The pipeline search is cast as an optimization problem with the objective of minimizing total execution time (equivalently, eliminating pipeline bubbles where compute units are idle). The decision variables are the batch sizes for each nano-operation (which determine execution times via the profiled mapping) and the start times of each nano-operation (which determine the schedule).

**Optimization objective.** The MILP's primary goal is to minimize pipeline execution time by removing bubbles — periods where compute-bound operations are not executing. The objective function is the makespan: the time from the start of the first nano-operation in a layer to the completion of the last.

**Constraints on the number of nano-operations.** The search begins with all operations split into exactly two nano-operations. The MILP solver determines the batch sizes and ordering. If the resulting schedule still has compute bubbles (idle time where a compute-bound nano-operation could run but isn't), the system increments the number of nano-operations for operations near the bubble. This iterative refinement continues until the MILP cannot find a schedule with fewer bubbles, or until further splitting would produce batch sizes below 128 (the minimum profiling granularity).

**Why start at two and increment.** Each split into more nano-operations increases weight-loading overhead (more kernel launches, each loading the full weight matrix). The search minimizes the number of nano-operations subject to removing compute bubbles — it finds the coarsest granularity that achieves full compute utilization. This is the engineering tradeoff: too few nano-operations leaves bubbles; too many wastes memory bandwidth.

**Constraints on batch sizes and execution times.** Batch sizes must be multiples of 128 (the profiling granularity) and sum to `B_dense`. Each nano-operation's execution time is fixed to the profiled `D_best` for its `(kernel, batch_size)` pair — there is no freedom to trade off time differently.

**Constraints on dependencies.** A nano-operation `B` depends on nano-operation `A` if and only if both conditions hold:
- Their parent operations are dependent in the original model (e.g., O-projection depends on attention output).
- Their input batch ranges intersect (e.g., if O1 processes tokens 0-767 and Attn1 processes tokens 0-767, then O1 depends on Attn1; but O1 does not depend on Attn2 which processes tokens 768-2048).

This "intersection-based" dependency is what enables independence across nano-batches. Two nano-operations with disjoint batch ranges are independent even if their parent operations are dependent. This is the formal mechanism by which nano-batching creates overlapping opportunities — by partitioning the batch dimension, the dependency graph is refined into finer-grained edges.

**Constraints on overlapping.** The MILP is restricted to overlap only operations that use different bottleneck resources (e.g., a compute-bound GEMM can overlap with a memory-bound GEMV, but two GEMMs cannot overlap because they'd compete for compute units and provide no benefit). This is enforced by a constraint that at any point in time, the sum of resource demands for active operations of the same type cannot exceed available capacity.

**Constraints on operation transformations.** Some network collectives have equivalent alternative implementations. For example, an AllGather can be transformed into an AllReduce by changing the weight partitioning scheme — the same tensor-parallel sharding can be expressed with different collective patterns. NanoFlow's MILP explores these alternatives, treating each transformation as an additional option in the operation dependency graph. The solver selects the combination that minimizes overall makespan.

**Search time and optimality.** The paper acknowledges that finding the globally optimal solution is computationally expensive — "searching for a optimal pipeline would take hours or even days" — but notes that a practical pipeline can be found in approximately 10 minutes by "prioritizing searching a feasible solution instead of finding a provably-optimal solution." This reflects the practical reality that the pipeline is generated once per model architecture and the 10-minute search cost is negligible compared to hours or days of serving.

**Example: 70B pipeline structure.** For LLaMA-2 70B with 8×A100 GPUs, Stage I produces a pipeline with 4 nano-operations for the initial KQV generation and associated attention — because three resources (compute for KQV, memory for decode attention, network for collectives) overlap at the start of the layer, and 4 nano-operations are needed to remove all compute bubbles in this region. The remaining operations (Up/Gate/Down, associated all-reduce) use only 2 nano-operations because only two resource types overlap there. The result is Figure 6, where the first segment interleaves compute-heavy KQV, memory-bound DecAttn, and network-bound collectives across four nano-batches, and the second segment executes two compute-heavy streams with network collectives.

---

#### Auto-Search Stage II: Refining the Pipeline

The pipeline from Stage I assumes no kernel interference — each nano-operation runs at its profiled best speed regardless of what else is running. In reality, co-scheduled kernels slow each other down. Stage II corrects for this by solving a second MILP that allocates GPU resources to each nano-operation, using the interference profiles from Section 4.1.1 to predict actual execution times.

**Inputs to Stage II:**
- The pipeline structure from Stage I (number, batch sizes, and ordering of nano-operations — now treated as fixed constraints)
- The resource mapping table (Table 3): `R → P` for GEMV and network kernels
- The interference-free best times `D_best` for each nano-operation's (kernel, batch_size)

**Outputs of Stage II:**
- `R_i`: the resource allocation (fraction of compute) for each nano-operation `i`
- `T_i = D_best_i / P_i`: the adjusted execution time for each nano-operation, where `P_i` is the normalized performance given its `R_i`

**Optimization objective.** Same as Stage I: minimize the total pipeline makespan. However, the decision variables are now the resource allocations `R_i` rather than batch sizes or ordering.

**Constraints on GPU resource utilization.** The key constraint enforces that the sum of `R` values for all active nano-operations at any given time is `≤ 1.0`. This captures the physical reality that the GPU's compute resources, while fungible across kernels, are finite. If at time `t`, nano-operations A (compute-bound) and B (memory-bound) are running, `R_A + R_B ≤ 1.0`. The paper models this as: "given that A gets `R_A` of the compute resources, B gets the remaining `1 - R_A`, and its performance degrades to `P_B` as determined by Table 3."

**Constraints on execution times.** For a nano-operation assigned resource allocation `R_i`, its execution time is `D_best_i / P_i`. For compute-bound (GEMM) kernels, `P_i = R_i` by definition — if a GEMM gets 60% of resources, it runs at 60% of its peak speed. For memory-bound (GEMV) and network-bound kernels, `P_i` is looked up from Table 3 based on the remaining resources `1 - R_GEMM` allocated to them.

**What Stage II optimizes concretely.** The resource allocation determines the balance between compute throughput and memory/network throughput. Consider the overlapping of KQV (GEMM) and DecAttn (GEMV) in Figure 6:

- If the MILP allocates `R = 0.9` to KQV (prioritizing compute), then DecAttn gets `R = 0.1` and runs at `P ≈ 0.2` (from Table 3) — achieving only 20% of its peak memory throughput. This might cause DecAttn to take too long and delay subsequent operations that depend on it.

- If it allocates `R = 0.4` to KQV, then DecAttn gets `R = 0.6` and runs at `P ≈ 0.8` — the memory operation is faster, but the compute operation is much slower.

The MILP finds the allocation that balances these tradeoffs across all overlapping pairs in the pipeline, minimizing the overall critical path.

**Why a second MILP rather than joint optimization.** The two-stage decomposition is the critical engineering insight that makes the problem tractable. Jointly optimizing nano-batch sizes, ordering, and resource allocation would require exploring a combinatorial space of `(batch_size, order, R)` across all candidate split counts — the paper estimates this would take "hours or even days." The decomposition works because the pipeline structure (Stage I) is primarily constrained by the dependency graph and batch-size granularity, while resource allocation (Stage II) is constrained by interference effects that are relatively independent of the exact batch sizes within the ranges explored. The approximation is that the optimal structure under interference-free assumptions remains near-optimal after interference correction.

**The resulting pipeline for LLaMA-2 70B (Figure 6).** The final output of auto-search for this model is the pipeline visualized in Figure 6:

- **First segment (KQV and attention):** 4 nano-batches (e.g., tokens 0-767, 768-1535, 1536-1791, 1792-2047). KQV nano-operations run with `R = 0.4` each (using 40% of compute, but since they overlap with memory/network operations that use distinct resources, the actual execution time is managed). Decode attention nano-operations also run with `R = 0.4`, reaching 80% of peak attention performance (the "0.4" here refers to the resource budget, with performance derived from Table 3). Network AllGather operations for attention outputs and O-projection inputs run with `R = 0.2` and `R = 0.1`, using minimal compute but sustained network bandwidth.

- **Second segment (Up/Gate/Down projections):** 2 nano-batches. UGD1 runs with `R = 0.9` (near-full compute) on the first batch, while UGD2 runs with `R = 0.9` on the second batch but overlaps with network AllReduce operations that need `R ≈ 0.1-0.2`. Prefill attention runs with `R = 0.6` — a compute-intensive allocation reflecting that prefill attention is also compute-bound (it processes all input tokens simultaneously).

The key property of this pipeline is that **compute utilization is high throughout the entire layer execution** — there are no WASTED bubbles where the GPU's compute units sit idle waiting for memory or network operations. Contrast this with Figure 4, where between each pair of compute-bound operations, there is a visible idle gap filled by memory-bound or network-bound operations running alone.

---

#### The NanoFlow Runtime System

Once auto-search generates the pipeline, the runtime executes it during live serving. The runtime handles four concerns: batch formation, asynchronous scheduling, KV-cache management, and multi-round conversation support.

**Batch formation.** NanoFlow assumes that external infrastructure handles auto-scaling, workload balancing, and priority-aware routing. A NanoFlow instance receives an abundance of requests and treats them with equal priority. The key design decision is maintaining a **constant dense batch size** across iterations:

- The system tracks the number of decode tokens currently active and the token budget available for prefill.
- Unfinished decode requests are prioritized — they are guaranteed a slot in each iteration because their latency is on the critical path for all requests.
- Prefill requests are chunked at a token granularity (following SarathiServe's approach) to fill exactly the remaining capacity up to `B_dense`.
- As decode requests complete (detected by EOS token generation), the freed capacity is filled with prefill tokens from waiting requests.

Because `B_dense` remains constant, all GEMM operations execute with the same batch dimensions iteration after iteration, maintaining consistent compute utilization and avoiding the performance variability that comes with variable batch sizes. The batch of hundreds of requests (e.g., 256 decode + 1 prefill request chunked across iterations) easily reaches the target `B_dense` of ~2048 tokens, which is "easy to attain in real-world large-scale serving" (Section 4.2).

**Asynchronous scheduling.** In traditional serving engines, the CPU-side scheduler runs between GPU iterations: after iteration `i` completes, the CPU detects EOS tokens, removes finished requests, inserts new prefill chunks, adjusts the page table for PagedAttention, and then launches iteration `i+1`. The GPU is idle during this CPU work.

NanoFlow eliminates this idle time by **overlapping CPU scheduling with GPU execution**. The scheduler forms the batch for iteration `i+1` while iteration `i` is still running on the GPU. The consequence is that at the time of forming batch `i+1`, the EOS tokens from iteration `i` are not yet known (the GPU hasn't finished generating them). NanoFlow handles this by deferring the EOS detection: after launching iteration `i+1`, the scheduler forms batch `i+2`, detects EOS tokens from iteration `i`, and removes those completed requests from batch `i+2`.

**Why the one-iteration delay is acceptable.** The paper quantifies this: since typical workloads have average decode lengths surpassing 100 tokens (Table 4: Splitwise avg output 211, LMSYS-Chat avg output 222, ShareGPT avg output 322), the overhead of generating one extra token for a completed request is "`< 1%`" overhead. The extra token is wasted computation (it generates a token that will be discarded), but this tiny overhead is far outweighed by eliminating the GPU idle time between iterations.

**KV-cache management and offloading.** For multi-round conversations, a user may send a follow-up message after receiving the model's response. The LLM needs access to the KV-cache from the previous round to avoid recomputing the entire conversation. NanoFlow implements a hierarchical offloading system:

**Simultaneous offloading.** Rather than waiting for a request to complete and then copying the entire KV-cache to CPU memory, NanoFlow offloads each token's KV vectors **immediately after they are computed** during the KQV generation step — before they are appended to the KV-cache in GPU memory. This provides two benefits:
- The KV vectors at this point are contiguous in memory (they are output tensors from a GEMM), enabling high-bandwidth contiguous copy rather than scattered reads.
- The offloading is spread across iterations rather than causing a burst at request completion.

The offloading uses GPU-initiated device-to-host copy operations (cudaMemcpyAsync on the copy engine, not the compute engine), which consume "minor GPU resources." These copies are scheduled during compute-bound FFN operations, when the copy engines are largely idle.

**KV-cache loading for multi-round.** When a follow-up request arrives for a conversation, the KV-cache from prior rounds must be loaded back to GPU memory. Because PagedAttention stores KV-cache in fragmented pages, naive host-to-device copy to scattered destinations would achieve poor bandwidth (limited by the overhead of many small DMA transfers). NanoFlow instead performs a **two-stage load**: first, copy the full KV-cache to a contiguous GPU buffer (achieving 7-10× higher bandwidth than scattered copy); then, perform a GPU-side scatter operation to copy pages from the contiguous buffer to their fragmented destinations. The scatter uses the GPU's own memory bandwidth, which is much faster than PCIe.

**Hierarchical cache management.** The KV-cache is managed in a two-level hierarchy: CPU memory (fast, limited capacity) and SSD (slow, large capacity). NanoFlow uses LRU eviction: when CPU memory is full, the least recently used KV-caches are evicted to SSD. When a new round arrives, the system checks CPU memory first; if the KV-cache is not present, it loads from SSD. The paper notes that offloading reduces pipeline throughput by 3.0% due to "kernel interference caused by KV-cache movement" but reduces compute for multi-round workloads by 3.02× (because recomputation is avoided).

---

#### Example Pipelines for Different Model Scales

The auto-search engine adapts its output to the model's resource profile. The paper provides examples for three categories:

**70B-scale models (LLaMA-2 70B, LLaMA-3 70B, Qwen2.5-72B, Deepseek-67B).** These models require tensor parallelism across 8 GPUs, introducing network-bound operations. The pipeline (Figure 6, detailed above) uses 4 nano-operations for the KQV/attention region where compute, memory, and network all overlap, and 2 nano-operations for the FFN region. The paper notes that variations across these models (LLaMA-3's larger vocabulary, Qwen2.5's bias terms, Deepseek's different layer/hidden dimensions) do not significantly change the performance characteristics, "leading to similar schedules."

**8B-scale models (LLaMA-3 8B).** These models fit on a single GPU, eliminating network operations entirely. The pipeline is simpler: "auto-search splits all operations into two nano-operations, with decode attention overlapping with the Up Gate Down projection." Only two resource types overlap (compute and memory), so two nano-operations suffice to achieve high compute utilization.

**MoE models (Mixtral 8×7B).** Mixture-of-Experts models have different hidden dimensions and number of layers, plus an additional gate routing operation. The FFN is implemented using grouped-GEMM (different experts compute on different tokens, creating an imbalance between experts). Despite these architectural differences, "auto-search works similarly for an MoE model and automatically produces an efficient pipeline" — the profiling captures the new operations, and the MILP finds an overlapping schedule. This demonstrates the generality of the auto-search approach across diverse model architectures.

## 4. Key Insights and Innovations

### Innovation 1: The Memory-Bound Assumption Is Wrong for Modern LLM Serving, and That Changes Everything

The paper’s most intellectually disruptive contribution is not the system it builds but the diagnostic it performs first. The field has operated under a near-consensus belief that LLM inference is **memory bandwidth-bound** — that the rate at which data can be moved from GPU memory to compute units sets the throughput ceiling, not the rate at which computations execute. This belief is entirely reasonable on its face: model weights are massive, KV-caches scale quadratically with context length and can exceed weight size, and each decode iteration generates only one token per sequence while loading both the full weight matrix and the per-request KV-cache. Given these facts, it seems almost inevitable that memory bandwidth, not compute, should be the binding constraint.

The paper systematically dismantles this assumption, and in doing so, **reclassifies the entire optimization landscape**. The analytical cost model in Section 3.2–3.3 demonstrates that three concurrent trends have shifted modern LLM serving firmly into compute-bound territory. First, **grouped query attention (GQA)** multiplies the effective decode batch size by 4× or more, since multiple attention heads share a single KV-cache, dramatically amortizing weight loading costs. Second, **model sizes have grown** faster than memory bandwidth — the ratio `Compute/MemBW` in Table 1 rises from 139 for V100 (2017) to 281 for B200 (2024), meaning each new GPU generation dedicates relatively more silicon to compute than to memory bandwidth. Third, **mixed prefill-decode batching** aggregates prefill tokens (processed in bulk) with decode tokens into a single dense GEMM batch, further amortizing memory I/O per token.

The quantitative validation is striking. For LLaMA-2 70B on 8×A100 GPUs across realistic workload traces (Splitwise, LMSYS-Chat, ShareGPT), the derived ratio `TR = T_Mem / T_Compute` ranges from 0.07 to 0.37 (Figure 3) — meaning compute time is **3× to 14× longer** than memory time. Table 2 confirms this empirically: total estimated compute time is 114.17 ms versus 45.09 ms for memory and 31.33 ms for network. The workload is deeply compute-bound, not even close to balanced.

This matters intellectually because **it inverts the optimization objective**. If the workload were memory-bound, the correct strategy would be to reduce memory traffic — quantization, KV-cache compression, weight sharing, speculative decoding. But if the workload is compute-bound, the objective shifts to **maximizing compute utilization**, and memory or network operations become obstacles to be hidden behind compute rather than problems to be minimized. This inversion is what makes nano-batching’s extra weight loading (each nano-operation independently loads the full weight matrix) not just tolerable but beneficial — the additional memory traffic can be absorbed by the idle memory bandwidth during periods when compute would otherwise be stalled. In a memory-bound world, nano-batching would be destructive. The paper’s reclassification makes it productive.

This is fundamentally a **diagnostic reframing** rather than a new technique. It changes what systems builders should measure, what they should optimize for, and what tradeoffs are worth making. The subsequent contributions — intra-device overlapping, auto-search — are mechanisms that operationalize this reframing, but the reframing itself is the paper’s most significant intellectual move because it **reinterprets the existing empirical landscape** and makes previously counterintuitive design choices (split batches, reload weights, overlap operations) obviously correct.

---

### Innovation 2: Intra-Device Parallelism as a New Axis in the LLM Serving Design Space

Prior work on LLM serving parallelism operates at two well-established granularities: **inter-device** and **inter-operation**. Inter-device parallelism partitions work across GPUs — tensor parallelism splits weight matrices column-wise or row-wise across GPUs within a node, and pipeline parallelism distributes layers across devices, connecting them via point-to-point communication. The field has invested substantial effort in optimizing these paradigms (Megatron-LM, Alpa, FasterMoE) and in tuning the tradeoffs between them (AlpaServe, FlexGen). Inter-operation parallelism, explored in systems like Rammer, Unity, ASPEN, and Welder, breaks the sequential boundaries between operations within a single device by constructing tile-level dataflow graphs that fuse operations or remap them to functional units.

NanoFlow introduces **intra-device parallelism** — parallelism *within* a single GPU, but at a granularity fundamentally different from both existing categories. Unlike inter-device parallelism, which partitions across the device boundary and incurs network communication costs, intra-device parallelism partitions the **batch dimension** within a single device and co-schedules the resulting sub-operations on different execution resources simultaneously. Unlike prior inter-operation parallelism (Rammer, Welder), which focused on fusing operations to reduce memory traffic within the assumption that operations share similar resource profiles, intra-device parallelism explicitly exploits the **heterogeneity** of resource bottlenecks — compute-bound GEMMs, memory-bound GEMVs, network-bound collectives — by overlapping them rather than fusing them.

This is a genuinely new axis because it answers a question prior systems didn’t ask: **given that different operations within a transformer layer are bottlenecked by different hardware resources, why execute them sequentially?** Existing serving frameworks (SGLang, vLLM, DeepSpeed-FastGen, TensorRT-LLM) form a single large batch and execute each operation to completion before starting the next, producing the pipeline bubbles visible in Figure 4. The insight is that the batch dimension can be partitioned, and that partitioning creates independent sub-batches whose operations on *different* resources can be overlapped without data hazards.

The conceptual framing matters. This isn’t pipeline parallelism applied within a layer — it’s a form of **resource-level multiplexing** where the GPU’s compute units, memory controllers, and NVLink engines are treated as independent service queues that can be fed simultaneously from the same workload by disaggregating at the batch dimension. The paper’s MCMC-inspired framing from Section 2 (proposal distribution vs. verifier as complementary scaling axes) is not used here — this is a systems contribution — but the intellectual structure is parallel: identify independent, complementary resources that existing systems treat as a single sequential pipeline, then design a mechanism to exploit their independence.

The significance is practical as well as conceptual. Intra-device parallelism sits **below** all existing parallelism strategies in the hierarchy — it operates on the batch *after* tensor parallelism has partitioned weights and *before* individual kernels execute. This orthogonality means NanoFlow can be combined with existing inter-device parallelism and operation-level optimizations (FlashAttention, quantization) without conflict, because it schedules at a granularity that doesn’t interfere with those mechanisms. The paper’s 1.91× throughput gain over TensorRT-LLM (Figure 7) represents the *additional* benefit of this new axis beyond what state-of-the-art inter-device and inter-operation optimizations already provide — a substantial complement rather than a replacement.

---

### Innovation 3: The Auto-Search Engine That Makes Intra-Device Overlapping Practical Across Diverse Model Architectures

The idea of overlapping heterogeneous operations is not entirely unprecedented — Welder (Shi et al., 2023) explored tile-graph scheduling, and ASPEN (Park et al., 2023) broke operator barriers for DNN parallelization. What distinguishes NanoFlow’s contribution is the **automated pipeline generation methodology** that makes overlapping practical without per-model manual engineering.

Prior systems requiring fine-grained operation scheduling (Welder, ASPEN) demand **recompilation from scratch** for each model architecture — a labor-intensive process that involves manually constructing tile-level dataflow graphs, specifying dependencies, and tuning resource allocation. For a field where model architectures evolve rapidly (LLaMA-3, Mistral, Qwen2, Deepseek all have different layer counts, hidden dimensions, attention mechanisms, and FFN structures), this manual approach imposes a prohibitive engineering burden. It’s why operation-level scheduling hasn’t been widely adopted in production serving engines despite its theoretical appeal.

NanoFlow’s two-stage auto-search procedure (Section 4.1) solves this portability problem. The key engineering insight is **decomposing the search into structure (Stage I) and refinement (Stage II)**, which reduces the joint optimization space from a combinatorial explosion (millions of kernel implementation pairs, variable nano-batch counts, variable batch sizes, variable resource allocations) to two tractable MILP problems solved sequentially in approximately 10 minutes. Stage I determines the nano-batch structure assuming no interference — a clean, well-defined MILP over batch sizes and ordering constraints. Stage II re-allocates resources using profiled interference data — a second MILP where the fixed structure dramatically reduces the search space.

This decomposition is not just a performance optimization — it’s an **abstraction that separates concerns**. The dependency structure of a transformer (which operations depend on which others, and how batch-range intersection defines nano-operation dependencies) is model-specific but well-defined by the PyTorch computation graph. The interference characteristics (how GEMM and GEMV kernels slow each other down when co-scheduled) are hardware-specific but largely independent of model architecture — the paper’s sensitivity analysis shows the `R → P` mapping has “standard deviation within 5% of the mean” across GEMM shapes and batch sizes. By profiling each concern independently and combining them through sequential MILPs rather than joint optimization, the approach becomes **modular**: a new model requires only a new PyTorch graph traversal, not new interference profiling; a new GPU generation requires only new kernel profiling, not a new pipeline design methodology.

The empirical evidence for this generality comes from Figure 11, which shows NanoFlow automatically generating efficient pipelines for six models spanning three architectural families — 8B dense (LLaMA-3-8B), 70B dense (LLaMA-3-70B, Qwen2-72B, Deepseek-67B), and mixture-of-experts (Mixtral 8×7B) — without per-model manual tuning. Each achieves 50–78.5% of optimal throughput, with vLLM comparisons showing 2.0–4.9× throughput gains. The auto-search produces different pipeline structures for each category — 4 nano-operations for 70B models where three resource types overlap, 2 nano-operations for 8B single-GPU models where only compute and memory overlap — adapting automatically to the model’s op count, hidden dimensions, and attention mechanism.

This is a **fundamentally methodological** contribution rather than a point-solution. The paper doesn’t claim to have found the one true pipeline for LLaMA-2 70B — it claims to have built a pipeline generator that works across models. This shifts the optimization burden from manual engineering (per-model, per-hardware, per-batch-size) to automated profiling and MILP solving, making intra-device parallelism deployable in a field where model architectures and hardware configurations multiply combinatorially.

---

### Innovation 4: Empirical Evidence That Sequential Execution Is the Dominant Throughput Bottleneck, Not Kernel Efficiency or Memory Management

The paper’s analysis in Section 3.6 and Section 6.5 identifies a bottleneck that is **structural rather than implementation-level**. Existing serving systems have invested enormous engineering effort in individual operation efficiency — CUTLASS provides state-of-the-art GEMM kernels achieving ~80% of peak FLOPs; PagedAttention nearly eliminates KV-cache fragmentation overhead; continuous batching maximizes batch sizes; FlashAttention fuses attention computation to reduce memory traffic. Each of these is an optimization within a single operation or across a single resource type.

Yet despite these per-operation optimizations, end-to-end compute utilization across the pipeline is only ~40% (Section 3.6). The paper’s Figure 10 makes this visible: in the non-overlapping baseline, compute utilization oscillates between near-100% during GEMMs and near-0% during decode attention and network collectives, averaging to ~40%. The individual operations are well-tuned — they just don’t overlap.

This finding is significant because it identifies a **scheduling gap** that no amount of per-kernel optimization can close. Making GEMM kernels 10% faster would compress the compute-bound periods slightly but would not fill the idle gaps during memory-bound attention — the pipeline bubbles would shrink proportionally but not disappear. The bottleneck is not *how fast* each operation executes but *that they execute one at a time*. This is a fundamentally different diagnosis from the one implicit in most prior work, which implicitly assumes that improving individual operations (better tiling, better memory management, better quantization) will linearly translate to better throughput. The paper shows this assumption is false: the returns to per-operation optimization are bounded by the sequential execution pattern.

The corollary is that **throughput gains beyond ~40% of optimal require structural changes to the execution model**, not incremental improvements to existing operations. This explains why, despite years of optimization across multiple serving frameworks (vLLM, TensorRT-LLM, DeepSpeed-FastGen), no system exceeds 37.8% of optimal throughput on LLaMA-2 70B (Figure 7) — they all share the same sequential execution pattern, and that pattern is the ceiling. NanoFlow’s 68.5% of optimal throughput doesn’t come from faster GEMMs or better memory management (it uses the same CUTLASS kernels and PagedAttention as baselines) but from breaking the sequential execution model entirely.

This is an **explanatory** contribution as much as a technical one. It provides a unified account for why existing systems underperform despite excellent per-component engineering, and it sets a clear direction for future work: the next frontier in LLM serving throughput is not better kernels but better scheduling across heterogeneous resources. The ablation in Figure 9 confirms this: the nano-batch-only baseline (splitting batches but executing sequentially) reduces performance by 13.2% compared to non-overlapping baseline, while the full overlapping pipeline gains 7–17% over the non-overlapping baseline — the gain comes from concurrent execution, not from the batching strategy itself.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The evaluation uses three real-world conversation datasets: **Splitwise** (a production trace from Microsoft with ~20,000 requests, average input 1155 tokens, average output 211 tokens), **LMSYS-Chat-1M** (a large-scale dataset with 1 million real-world conversations from 25 LLMs, average input 102 tokens, average output 222 tokens), and **ShareGPT** (conversations collected from the ShareGPT API, average input 246 tokens, average output 322 tokens). For LMSYS-Chat and ShareGPT, the paper randomly samples 50,000 requests. Table 4 summarizes the average input and output lengths and their standard deviations for each dataset. Experiments are also run with constant-length inputs and outputs (512-512, 1024-512, 512-1024) to isolate the effects of workload variability.

- **Base model(s).** The primary evaluation is on **LLaMA-2-70B**, described as "one of the most widely-used open-source LLMs," running with 8×A100 80GB SXM GPUs and tensor parallelism of degree 8. Additional models evaluated include **LLaMA-3-70B**, **LLaMA-3-8B** (single A100 80GB SXM), **Qwen2-72B**, **Deepseek-67B**, and **Mixtral 8×7B** (a mixture-of-experts model). All models use FP16 weights and activations, which the paper notes is "standard for data center-scale inference." The 70B-scale models require tensor parallelism across 8 GPUs; LLaMA-3-8B fits on a single GPU. The diversity of architectures—dense 8B, dense 70B with different vocabularies/hidden dimensions, and MoE—tests the auto-search engine's generality.

- **Metrics.** The primary metric is **throughput**, measured as total tokens (prefill input tokens + decode output tokens) processed per second per GPU: `Throughput_total = (total input tokens + total output tokens) / execution time / N_GPU`. The paper also reports **normalized latency** for the latency experiments (Section 6.3), computed as end-to-end request latency divided by output length in tokens (ms/token), then averaged across all requests. This normalization accounts for the fact that requests with longer outputs naturally take longer, and a 200ms SLO for normalized latency is used as the quality-of-service target, following prior work (DistServe). The paper also reports **tail latency** as the 99th percentile of normalized latency.

- **Baselines.** Three state-of-the-art serving frameworks serve as baselines:

  - **vLLM** (v0.5.3.post1, commit 38c4b7e): described as "a state-of-the-art serving system delivering high throughput," implementing PagedAttention for GPU memory utilization and chunked prefill for higher GPU utilization. The paper cites Kwon et al., 2023.

  - **DeepSpeed-FastGen** (v0.2.3, commit 429bc5c): "a serving framework developed by Microsoft" that dynamically composes prefill with decode requests to operate in a high-throughput regime. The `max-ragged-batch-size` parameter is varied to tune batch size for highest throughput. Cited as Holmes et al., 2024.

  - **TensorRT-LLM** (v0.8.0, commit 5955b8a): "a high-performance LLM inference engine built upon NVIDIA's TensorRT SDK." The `max-num-tokens` is set by calculating the maximum capacity for KV-cache in GPU memory. Paged KV-cache and dynamic batching optimizations are enabled at compilation time. Cited from the NVIDIA repository.

- **Generation budget / compute accounting.** The paper does not use a generation budget in the style of best-of-N scaling papers. Instead, the relevant unit of comparison is the **dense batch size** `B_dense` (the total number of tokens—prefill + decode—in the batch used for GEMM operations). For LLaMA-2-70B with 8×A100 GPUs, this is fixed at 2048 tokens, which is the maximum that fits in GPU memory given the KV-cache requirements. All systems are configured to operate at their maximum sustainable batch size to measure peak throughput. For latency experiments, request rate is the independent variable (requests/second arriving at the system), with normalized latency measured at each rate. The **theoretically optimal throughput** is derived analytically in Equation 5 as `Compute / (2 × P_Model)`, and the paper measures the peak compute capacity by profiling the state-of-the-art GEMM vendor library CUTLASS with a token batch size of 2048, yielding 280 TFLOPS for FP16 on 8×A100 GPUs and an optimal throughput of 1857 tokens/s/GPU for LLaMA-2-70B.

- **Cross-validation / statistical protocol.** There is no explicit cross-validation or train/test split, as this is a systems paper rather than a machine learning paper. The throughput measurements are taken from end-to-end serving runs. The latency experiments model request arrival intervals via an exponential distribution (following prior work from the vLLM paper), generate 5 minutes of request traces from each dataset, and measure normalized latency at various request rates. The paper does not report confidence intervals or error bars on throughput measurements. The auto-search pipeline is generated once per model architecture, not per-dataset.

### Main Quantitative Results

#### Throughput Comparison Against Baselines

**Headline result.** NanoFlow achieves the highest throughput across all workload settings and models. For LLaMA-2-70B on 8×A100 GPUs, NanoFlow's throughput reaches 65.5–68.5% of the theoretically optimal throughput (1857 tokens/s/GPU depending on the workload), compared to 37.8% for the best baseline (TensorRT-LLM). The average throughput gains over baselines are summarized in Figure 7.

**Constant-length workloads (Figure 7a).** With fixed input/output lengths, NanoFlow achieves throughput of 1212–1286 tokens/s/GPU across three length configurations:
- Input 512, Output 512: NanoFlow 1286 tokens/s/GPU vs. vLLM 494, DeepSpeed-FastGen 410, TensorRT-LLM 735
- Input 1024, Output 512: NanoFlow 1263 tokens/s/GPU vs. vLLM 552, DeepSpeed-FastGen 490, TensorRT-LLM 817
- Input 512, Output 1024: NanoFlow 1212 tokens/s/GPU vs. vLLM 372, DeepSpeed-FastGen 513, TensorRT-LLM 636

On average for constant lengths, NanoFlow provides **2.62× higher throughput than vLLM**, **2.78× higher than DeepSpeed-FastGen**, and **1.73× higher than TensorRT-LLM**. The optimal throughput line is drawn at 1857 tokens/s/GPU.

**Dataset-driven workloads (Figure 7b).** With input and output lengths drawn from real traces, NanoFlow achieves:
- Splitwise: NanoFlow 1259 tokens/s/GPU vs. vLLM 484, DeepSpeed-FastGen 548, TensorRT-LLM 831
- LMSYS-Chat: NanoFlow 1247 tokens/s/GPU vs. vLLM 251, DeepSpeed-FastGen 293, TensorRT-LLM 560
- ShareGPT: NanoFlow 1272 tokens/s/GPU vs. vLLM 255, DeepSpeed-FastGen 335, TensorRT-LLM 639

On average for dataset-driven workloads, NanoFlow provides **4.18× higher throughput than vLLM**, **3.45× higher than DeepSpeed-FastGen**, and **1.91× higher than TensorRT-LLM**. The larger gap over vLLM on dataset-driven workloads compared to constant-length workloads suggests that vLLM's batch formation is more sensitive to workload variability, while NanoFlow's constant `B_dense` strategy is more robust.

**Why TensorRT-LLM is the strongest baseline.** TensorRT-LLM achieves 37.8% of optimal throughput (735 tokens/s/GPU at 512-512), substantially outperforming vLLM (26.6%) and DeepSpeed-FastGen (22.1%). The paper attributes TensorRT-LLM's advantage to its use of NVIDIA's highly optimized TensorRT compiler stack, but notes that it still suffers from the same sequential execution bottleneck—"TensorRT-LLM achieves only 37.8% of the optimal throughput"—because its operations also execute sequentially, as shown in the general pipeline of Figure 4.

#### Latency Under Varying Request Rates

**Headline result.** NanoFlow sustains higher request rates within the 200ms normalized latency SLO compared to all baselines. At low request rates, NanoFlow has slightly higher latency than baselines due to its large constant batch size (a throughput-oriented design choice). At higher rates, NanoFlow's higher throughput allows it to serve more requests before latency degrades beyond the SLO.

**Latency curves (Figure 8).** For each dataset, the paper plots normalized latency (ms/token) against request rate (requests/second):

- **Splitwise (Figure 8a):** NanoFlow sustains up to 8.2 requests/second within 200ms SLO, compared to 6.6 for vLLM (1.24× higher), with TensorRT-LLM and DeepSpeed-FastGen showing intermediate performance. At 5 requests/second, all systems have low latency; by 10 requests/second, vLLM exceeds 600ms normalized latency while NanoFlow remains under 200ms.

- **LMSYS-Chat (Figure 8b):** NanoFlow sustains up to 32.1 requests/second within 200ms SLO, compared to 17.1 for vLLM (1.88× higher) and approximately 19.5 for TensorRT-LLM (the strongest baseline, giving **1.64× higher request rate**). LMSYS-Chat has shorter inputs (avg 102 tokens) and longer outputs (avg 222 tokens), making it decode-heavy—a regime where NanoFlow's overlapping of memory-bound decode attention with compute-bound GEMMs is particularly beneficial.

- **ShareGPT (Figure 8c):** NanoFlow sustains up to 16.3 requests/second vs. 10.5 for vLLM (1.55× higher). TensorRT-LLM reaches approximately 13.5 requests/second at the SLO boundary.

**Low-rate behavior.** At the lowest request rates shown (2.5–5 requests/second), NanoFlow's normalized latency is comparable but slightly higher than baselines. For Splitwise at 2.5 req/s, all four systems cluster around 50–100ms/token; NanoFlow is at the upper end of this cluster. The paper explicitly notes this tradeoff: "NanoFlow targets throughput-oriented scenarios and therefore employs a large dense batch size." The constant 2048-token batch means individual requests may wait slightly longer for batch formation when request rates are low, but this effect diminishes as rates increase.

**Tail latency.** The paper reports that NanoFlow's "99th-percentile latency is only 1.07× of the average latency at near-maximum throughput." This low tail latency ratio is attributed to the "constant dense batch size" strategy—since every iteration processes the same number of tokens, the per-iteration latency is highly predictable, and individual requests do not experience outlier delays from variable batch sizes. The paper presents this as a qualitative claim without a dedicated tail latency figure, but the mechanism (constant batch size → constant per-iteration time) is consistent with the design.

#### Performance on Other LLMs

**Headline result.** NanoFlow generalizes across diverse model architectures without manual tuning, achieving 50.4% to 78.5% of optimal throughput on five additional models, with a 2.0–4.9× throughput advantage over vLLM.

**Per-model results (Figure 11).** All evaluations use constant input 1024, output 512, on 8×A100 80GB SXM GPUs (except LLaMA-3-8B, which runs on a single A100):

- **LLaMA-3-70B:** NanoFlow 1306 tokens/s/GPU (70.6% of optimal) vs. vLLM 593 (32.0% of optimal). Optimal throughput is computed per-model using Equation 5.
- **Qwen2-72B:** NanoFlow 1213 tokens/s/GPU (67.4% of optimal) vs. vLLM 554 (30.8%).
- **Deepseek-67B:** NanoFlow 1147 tokens/s/GPU (59.1% of optimal) vs. vLLM 532 (27.4%).
- **Mixtral 8×7B:** NanoFlow 5188 tokens/s/GPU (50.4% of optimal) vs. vLLM 997 (9.7%). This model achieves the highest absolute throughput due to its smaller effective parameter count per token (MoE with sparse activation), but NanoFlow reaches a lower fraction of optimal than for dense 70B models.
- **LLaMA-3-8B:** NanoFlow 12756 tokens/s/GPU (78.5% of optimal) vs. vLLM 5187 (31.9%). Running on a single GPU, this 8B model has no network operations, simplifying the pipeline; NanoFlow achieves the highest fraction of optimal throughput among all evaluated models.

The 70B-scale dense models (LLaMA-3, Qwen2, Deepseek) all achieve 59–71% of optimal throughput with NanoFlow, suggesting that the auto-search produces similar-quality pipelines for this architecture class despite variations in vocabulary size (LLaMA-3's 128K vocab vs. LLaMA-2's 32K), hidden dimensions (Deepseek-67B's different `D_model`), and bias terms (Qwen2-72B). The Mixtral MoE model achieves a lower fraction of optimal (50.4%), which the paper attributes to "imbalance between experts" that the auto-search pipeline must accommodate, though the absolute throughput (5188 tokens/s/GPU) is the highest among all models.

#### Resource Utilization Analysis

**Headline result.** NanoFlow achieves 68.5% average compute utilization, compared to ~40% for the non-overlapping baseline, by concurrently utilizing compute, memory, and network bandwidth throughout the pipeline.

**Resource usage patterns (Figure 10).** The paper provides a detailed timeline visualization of resource utilization within a single transformer layer for both the non-overlapping baseline (Figure 10a) and NanoFlow (Figure 10b), showing the percentage of compute, memory bandwidth, and network bandwidth used at each microsecond timestep:

- **Non-overlapping baseline (Figure 10a):** Compute utilization (top panel) oscillates sharply—near 100% during GEMM operations (KQV, O-projection, UGD) but drops to near zero during decode attention and network collectives. Memory utilization (middle panel) spikes during decode attention (loading KV-cache) and is low during GEMMs. Network utilization (bottom panel) is active only during AllGather/AllReduce operations. The three resource types are almost never simultaneously active at high levels. The paper quantifies this as "the total compute utilization is only 40%" across the pipeline.

- **NanoFlow (Figure 10b):** All three resource types show sustained, overlapping utilization. Compute utilization remains elevated throughout the layer execution because GEMM nano-operations overlap with memory-bound and network-bound nano-operations. Memory utilization is less bursty because decode attention nano-operations are spread across the timeline. Network utilization overlaps with both compute and memory periods. The paper reports that NanoFlow achieves "68.5% average compute utilization," which is the basis for the 68.5% of optimal throughput figure—the two percentages match because optimal throughput assumes 100% compute utilization, and the remaining gap to 100% is attributed primarily to kernel interference effects that the auto-search's `R → P` model cannot fully eliminate, plus the overhead of weight re-loading from nano-batching.

**Why 68.5% rather than 100%.** The paper identifies "kernel interference" (Section 6.5) as the primary reason NanoFlow falls short of optimal compute utilization. When multiple kernels run concurrently, they compete for GPU execution units, caches, and memory bandwidth in ways that the `R → P` profiling model approximates but does not eliminate. The nano-batching overhead (loading weights multiple times) also consumes some memory bandwidth that could otherwise serve the compute-bound operations. The 68.5% figure represents the practical ceiling given these irreducible interference effects on current NVIDIA hardware that does not provide explicit resource partitioning control.

### Ablation Studies and Robustness Checks

**Nano-batching overhead (Figure 9, "Nanobatch-only" bar):** Splitting batches into nano-batches but executing them sequentially (without overlapping) reduces throughput by 13.2% compared to the non-overlapping baseline. This is measured by comparing the "Non-overlap" bar (which processes the full 2048-token batch in a single sequential pipeline, similar to existing frameworks) with the "Nanobatch-only" bar (which splits into the nano-batch sizes determined by Stage I of auto-search, but executes nano-operations sequentially). For the Input 512 / Output 512 configuration: Non-overlap achieves 1273 tokens/s/GPU, while Nanobatch-only achieves 1106 tokens/s/GPU. The 13.2% degradation confirms that nano-batching's weight re-loading cost is real and would be harmful if not compensated by overlapping. This ablation is critical because it isolates the contribution of overlapping from the batching strategy itself—the throughput gain of NanoFlow over non-overlap is not from nano-batching per se but from the concurrent execution that nano-batching enables.

**Overlapping of network-bound operations (Figure 9, "NanoFlow" vs. "Non-overlap" on prefill-only workload):** On the Input 512 / Output 0 workload (prefill only, no decode attention), NanoFlow achieves 1402 tokens/s/GPU vs. 1446 for the non-overlap baseline—a 1.07× speedup. This workload isolates the benefit of overlapping network-bound collectives with compute-bound GEMMs, since there are no memory-bound decode attention operations to overlap. The relatively modest 7% improvement suggests that network-compute overlapping alone provides limited benefit; the major gains come from also overlapping memory-bound decode attention, which is absent in this workload.

**Overlapping of both network- and memory-bound operations (Figure 9, decode-heavy workload):** On the Input 512 / Output 1024 workload (decode-heavy, with substantial KV-cache loading), NanoFlow achieves 1290 tokens/s/GPU vs. 1092 for the non-overlap baseline—a 1.17× speedup. This workload includes both network-bound collectives and memory-bound decode attention that can overlap with compute-bound GEMMs. The 17% improvement demonstrates the compound benefit of overlapping all three resource types. Combined with the prefill-only result, the paper decomposes the overlapping benefit as: network-compute overlap contributes approximately 7%, and adding memory-compute overlap contributes an additional ~10%, for a combined ~17% improvement.

**KV-cache offloading overhead (Figure 9, "NanoFlow-offload" bar):** Enabling KV-cache offloading to CPU/SSD reduces throughput by 3.0% compared to the non-offloading NanoFlow configuration. For Input 512 / Output 512: NanoFlow-offload achieves 1244 tokens/s/GPU vs. 1290 for NanoFlow. The paper attributes this to "kernel interference caused by KV-cache movement"—the device-to-host copy operations consume some GPU memory bandwidth and copy engine resources, slightly slowing concurrent compute-bound operations. However, the paper reports that offloading "can reduce 3.02× compute for multi-round LMSYS-Chat workloads" by avoiding recomputation of KV-cache for prior conversation rounds, making the 3.0% throughput penalty a favorable tradeoff for multi-round applications.

**Nano-batching granularity (implicit in the 4 vs. 2 nano-operation structure):** The auto-search's choice of 4 nano-operations for the KQV/attention region and 2 nano-operations for the FFN region (Section 4.1.4) represents an empirically validated optimum. The paper does not provide a detailed ablation of alternative nano-batch counts (e.g., 3, 5, 6) with corresponding throughput numbers. However, the search procedure's design—"auto-search increases the number of nano-operations for operations near the bubble to improve resource utilization until MILP cannot produce better solutions"—implies that the chosen counts are the minimal numbers that eliminate compute bubbles. The ablation section does not quantify the throughput penalty of sub-optimal nano-batch counts, which would have strengthened the claim that the MILP search finds genuinely optimal structures.

**Asynchronous scheduling contribution (Section 4.2.1, but no isolated ablation figure):** The paper describes asynchronous scheduling—forming the batch for iteration `i+1` while iteration `i` executes on GPU—as a key runtime optimization, but does not provide an ablation isolating its throughput contribution. The mechanism is described qualitatively: in existing frameworks, "GPU is under-utilized during this time" (CPU scheduling between iterations), and NanoFlow hides this overhead. Given that batch formation for 256+ decode requests with PagedAttention page table updates likely takes tens to hundreds of microseconds, and a single iteration for LLaMA-2 70B with `B_dense=2048` takes on the order of 2-3 milliseconds (per the cost model in Table 2), the GPU idle time between iterations could represent 5-15% overhead. The absence of a quantified ablation means the contribution of asynchronous scheduling to the overall throughput gain cannot be isolated from the primary overlapping mechanism.

**Prefill-decode ratio stability (implicit in the dataset results):** The consistent throughput of NanoFlow across ShareGPT (1272 tokens/s/GPU), LMSYS-Chat (1247), and Splitwise (1259) despite very different prefill-to-decode ratios (ShareGPT: 246/322, LMSYS: 102/222, Splitwise: 1155/211) demonstrates that the constant `B_dense` strategy and overlapping pipeline are robust to workload composition. The baselines show much wider variation across datasets—vLLM ranges from 251 to 484 tokens/s/GPU (1.93× variation), while NanoFlow varies by only 2.0%. This is an implicit robustness result that the paper does not explicitly label as an ablation but follows directly from the design: the pipeline is constructed for the maximum `B_dense`, and as long as the workload supplies enough requests to fill that batch size (which the external control plane ensures), the throughput remains stable.

**Generalization across model architectures (Figure 11, treated as a main result but functions as a robustness check):** The auto-search engine generates efficient pipelines for six models spanning 8B, 70B, and MoE architectures without manual intervention, achieving 50–78.5% of optimal throughput. This is a strong robustness check for the auto-search methodology—the `R → P` mapping from interference profiling (which the paper claims has "standard deviation within 5% of the mean" across GEMM shapes) transfers across model architectures, and the two-stage MILP formulation handles diverse operation dependency graphs. The lower fraction for Mixtral 8×7B (50.4% vs. 59–78% for dense models) suggests that MoE architectures present additional challenges (expert imbalance, grouped-GEMM overhead) that the current auto-search handles but less optimally.

**Negative result: Nano-batching without overlapping hurts performance.** The 13.2% throughput degradation from the nanobatch-only baseline (Figure 9) is a clear negative result that validates the paper's central tradeoff argument: nano-batching is only beneficial when its extra weight loading can be hidden behind compute via overlapping. In the absence of overlapping, the additional memory I/O from loading weights multiple times directly reduces throughput, confirming that the compute-bound classification (TR < 1) is what makes the strategy viable—in a memory-bound regime, the degradation would be even larger and could not be recovered.

### Critical Assessment

**Claim 1: "NanoFlow provides 1.91× throughput boost compared to state-of-the-art serving systems."** This claim, drawn from the executive summary, is **supported but requires careful qualification**. The 1.91× figure comes from averaging across dataset-driven workloads (ShareGPT, LMSYS-Chat, Splitwise) against TensorRT-LLM, the strongest baseline. Against vLLM, the gain is 4.18×; against DeepSpeed-FastGen, 3.45×. The choice of baseline dramatically affects the reported multiplier. TensorRT-LLM is the appropriate comparison because it is the strongest, but the 1.91× figure should not be interpreted as the gain over "typical" systems—most deployments use vLLM or similar, against which the gain is much larger. For constant-length workloads, the gain against TensorRT-LLM drops to 1.73×, suggesting that workload variability amplifies NanoFlow's advantage.

**Claim 2: "NanoFlow achieves between 50% to 72% of optimal throughput across popular models."** This claim is **supported for the models tested, but the "optimal" baseline deserves scrutiny**. The optimal throughput of 1857 tokens/s/GPU (Equation 5) assumes 100% compute utilization at the profiled peak of CUTLASS (280 TFLOPS for FP16 on 8×A100). This is an idealization—no real GEMM achieves 100% of theoretical peak FLOPs due to tile quantization, pipeline stalls, and memory latency. The paper's own profiling acknowledges this by using "profiled peak" rather than datasheet peak. However, the profiled peak is measured with a single GEMM in isolation; in a full pipeline with kernel launch overheads, CUDA event synchronization, and memory allocation, the achievable peak for a single operation would be lower. The 68.5% figure is thus best interpreted as 68.5% of a single-GEMM peak, not 68.5% of what is achievable in a full serving pipeline. A stronger baseline would be the throughput of a system that executes only the GEMM operations (no attention, no network) at the same batch size—this would give a tighter upper bound on what overlapping can achieve.

**Claim 3: "LLM serving is compute-bound for most common workloads and LLMs."** This claim, which underpins the entire motivation, is **strongly supported but bounded to the specific hardware and models tested**. The analytical `TR` values in Figure 3 show compute dominance for all configurations except the edge case of LLaMA-3 8B with long decode (512-1024), where `TR` approaches 1. The empirical validation in Table 2 shows compute time (114 ms) exceeding memory time (45 ms) and network time (31 ms). However, the "compute-bound" classification depends on the batch size reaching its maximum (`B_dense = 2048` for LLaMA-2 70B). At smaller batch sizes (e.g., during low-load periods or when latency constraints prevent batching), `TR` increases because weight loading is amortized over fewer tokens. The paper's throughput experiments run at maximum batch size (appropriate for throughput-oriented evaluation), but the latency experiments (Figure 8) show that at low request rates, batch sizes may be smaller, and the workload could shift toward memory-bound. The paper does not analyze how NanoFlow's overlapping pipeline performs at sub-maximum batch sizes—a gap that matters for deployments that cannot maintain the ~200 concurrent requests needed to fill `B_dense = 2048`.

**Claim 4: "NanoFlow achieves similar latency compared with the best baseline TensorRT-LLM at a low request rate, while handling up to 1.64× higher request rates within SLO constraint."** This claim is **supported with a nuance**: at the lowest request rates, NanoFlow's latency is slightly higher than TensorRT-LLM's (visible in Figure 8 as the leftmost data points where NanoFlow's curve sits above TensorRT-LLM's). The paper acknowledges this directly: "At lower rates, the NanoFlow instance has comparable but slightly higher latency, because NanoFlow targets throughput-oriented scenarios and therefore employs a large dense batch size." The 1.64× higher request rate within SLO (from LMSYS-Chat) is measured at the point where TensorRT-LLM's normalized latency crosses 200ms. However, the evaluation runs only 5 minutes of request traces per configuration. The exponential inter-arrival distribution models independent requests but does not capture burst patterns or temporal correlations that could affect tail latency. A 5-minute trace limits the number of requests at high rates: at 32 requests/second, the 5-minute trace contains only ~9600 requests, making the tail latency measurement (99th percentile) based on ~96 requests—a small sample for reliable tail estimation.

**Missing experiment: sensitivity to batch size.** The paper's throughput experiments fix `B_dense` at 2048, the maximum that fits in GPU memory. This is appropriate for throughput-maximization, but it leaves open the question of how NanoFlow's pipeline performs at smaller batch sizes. If request rates are insufficient to fill 2048 tokens, the pipeline would need to run with smaller nano-batches, potentially changing the interference characteristics (smaller GEMMs have lower arithmetic intensity, shifting toward memory-bound behavior). An experiment sweeping `B_dense` from 256 to 2048 and measuring throughput and normalized latency would characterize the operating envelope and reveal whether the overlapping benefits diminish at smaller scales. The auto-search would need to re-run for each batch size (since the optimal nano-batch splits depend on total batch size), making this a non-trivial experiment that the paper does not attempt.

**Missing baseline: TensorRT-LLM with maximum tuning.** The paper configures TensorRT-LLM by setting `max-num-tokens` to the KV-cache capacity and enabling paged KV-cache and dynamic batching. However, TensorRT-LLM has additional tuning knobs—kernel selection strategies, CUDA graph capture, multi-stream execution—that could improve its throughput. The paper does not describe an exhaustive tuning process for TensorRT-LLM, leaving open the possibility that the 37.8% of optimal throughput could be improved with additional engineering. This is a common challenge in systems comparisons: it is difficult to ensure that each baseline is configured optimally without deep expertise in each framework. The paper mitigates this by using the baselines' recommended configurations and varying key parameters (`max-ragged-batch-size` for DeepSpeed-FastGen), but a sensitivity analysis showing that further tuning does not close the gap would strengthen the comparison.

**Missing experiment: CPU scheduling overhead quantification.** Section 4.2.1 describes asynchronous scheduling as a key runtime optimization, but the paper provides no ablation isolating its throughput contribution. The mechanism—forming batch `i+1` while iteration `i` runs—hides CPU scheduling latency that would otherwise leave the GPU idle. In the non-overlapping baseline (which reuses NanoFlow's runtime but executes nano-operations sequentially), is asynchronous scheduling enabled? If not, the 13.2% nanobatch-only degradation (Figure 9) may partially reflect scheduling overhead rather than purely the weight re-loading cost. If so, the decomposition of overlapping benefit vs. scheduling benefit is conflated. A dedicated experiment comparing synchronous vs. asynchronous scheduling would clarify this.

**Missing experiment: memory capacity utilization.** The paper focuses on compute utilization as the metric of interest, but the analytical model (Equation 1-4) and the constant `B_dense` strategy both depend on the assumption that the system operates at maximum batch size. What fraction of GPU memory is actually utilized at `B_dense = 2048`? Are there remaining memory resources that could support an even larger batch size, which would further reduce `TR` and push the workload deeper into compute-bound territory? Conversely, does the KV-cache offloading system's 3.0% throughput penalty vary with memory pressure? These questions are not addressed, making it difficult to assess whether the 68.5% of optimal throughput is constrained by compute utilization or by memory capacity limiting the batch size.

**The 78.5% result for LLaMA-3-8B (single GPU) deserves scrutiny.** This is the highest fraction of optimal throughput achieved, yet Figure 3 shows that LLaMA-3 8B with long decode (512-1024) has `TR ≈ 1`—the workload is nearly balanced between compute and memory, not deeply compute-bound. NanoFlow's design is motivated by the compute-bound classification, so why does it achieve the highest fraction of optimal on the least compute-bound model? One hypothesis: on a single GPU, there are no network operations to overlap, so the pipeline is simpler (2 nano-operations only) and the interference model is more accurate (only compute-memory pairs, no compute-network or compute-memory-network triple overlaps). The 78.5% figure may reflect that the two-resource overlap problem is easier to optimize than the three-resource problem, not that the compute-bound assumption is stronger. The paper does not discuss this apparent contradiction.

**Generality beyond NVIDIA A100.** All experiments use NVIDIA A100 80GB SXM GPUs. The analytical cost model (Table 1, Figures 2-3) suggests that the compute-bound classification holds across GPU generations (V100 through B200) and vendors (AMD MI250 through MI325X, Intel Gaudi 2/3), but no experiments validate NanoFlow's actual performance on non-A100 hardware. Different GPU architectures have different interference characteristics (e.g., AMD's CDNA vs. NVIDIA's Ampere), different multi-kernel scheduling behaviors, and different memory hierarchies. The auto-search's `R → P` profiling methodology should transfer, but the achieved fraction of optimal throughput could differ substantially. This is an acknowledged scope limitation rather than a flaw, but it means the 50-72% range applies specifically to the A100 generation.

**Statistical significance and reproducibility.** The paper does not report error bars, confidence intervals, or variance across multiple runs for any throughput or latency measurement. Throughput is measured from a single end-to-end serving run per configuration (with 5 minutes of trace for latency experiments). For a systems paper, variance tends to be low (throughput measurements on dedicated hardware are typically stable within a few percent), but the absence of any variance quantification makes it impossible to assess whether the 68.5% figure is 68.5% ± 1% or 68.5% ± 5%. The paper also does not discuss whether results are reproducible across different A100 units or DGX nodes—while GPU performance is consistent within a generation, thermal throttling, NVLink topology, and driver versions can introduce variance that matters when claiming precise percentages of theoretical peak.

## 6. Limitations and Trade-offs

### 6.1 The Cost Model's Compute-Bound Classification Depends on Maximum Batch Size—and NanoFlow's Performance at Smaller Batches Is Uncharacterized

**The assumption or constraint.** The entire motivation for nano-batching rests on the claim that modern LLM serving workloads are compute-bound rather than memory-bound. The analytical derivation of this claim (Section 3.2–3.3) depends critically on the assumption that the system "always operates at the largest batch size at which the total available memory can hold the model weights and all the KV caches for the requests" (Section 3.1). The key ratio `TR = T_Mem / T_Compute` (Equation 4) includes the term `1/(2B_dense)` in the denominator—as batch size decreases, `TR` increases, and the workload shifts toward memory-bound territory. The paper's experiments all run at maximum `B_dense = 2048` for LLaMA-2 70B on 8×A100 GPUs.

**The consequence.** In any real deployment, the achieved batch size fluctuates with request load. During low-traffic periods, batch sizes will be substantially smaller than the memory-capacity maximum, and the workload may shift toward memory-bound behavior. In this regime, nano-batching's extra weight loading—which each nano-operation performs independently—becomes directly harmful rather than hidden behind compute. The 13.2% throughput degradation measured in the "Nanobatch-only" ablation (Figure 9) shows what happens when the extra memory I/O is not compensated by overlapping; in a memory-bound regime, this degradation would be larger, and overlapping might provide little or no benefit because the compute-bound operations that nano-batching is designed to keep busy would not be the bottleneck.

The practical consequence is that **NanoFlow's throughput advantage is conditional on high utilization**. A deployment that cannot sustain near-maximum batch sizes (e.g., one serving a long-tail of low-traffic applications, or one constrained by latency SLOs that prevent large batches) may see substantially smaller gains—or even throughput regression—compared to existing systems. The paper provides no guidance on the minimum batch size at which the overlapping strategy remains beneficial, and the auto-search procedure does not generate batch-size-dependent schedules.

**What evidence exists in the paper.** The latency experiments (Figure 8) provide an indirect hint. At low request rates (e.g., 2.5 req/s for Splitwise), NanoFlow's normalized latency is slightly higher than baselines. While the paper attributes this to the "large dense batch size" (Section 6.3), it also implies that actual batch sizes at these low rates may be smaller than the configured `B_dense`, potentially operating in a regime closer to the memory-bound boundary. However, the paper does not report the actual achieved batch sizes during the latency experiments, nor does it provide throughput measurements at sub-maximum batch sizes. The sensitivity of `TR` to batch size is analytically clear from Equation 4, but is never empirically validated or swept as a controlled variable. No experiment varies `B_dense` and measures the resulting throughput, latency, or compute utilization to characterize the operating envelope.

**Mitigation status.** The paper does not address this limitation. It acknowledges indirectly that "NanoFlow targets throughput-oriented scenarios" (Section 6.3), which is an admission that the design is optimized for high-load conditions, but it does not discuss what happens when load is insufficient. The authors do not suggest any mechanism for dynamically switching between overlapping and non-overlapping execution based on current batch size, nor do they describe how the external control plane (which manages "auto-scaling, workload balancing, and priority-aware routing" per Section 4.2.1) should handle low-load scenarios. The paper explicitly notes that "when requests are not abundant, the control plane should reduce the number of NanoFlow instances to maintain a sufficiently large per-instance batch size," but this merely routes traffic to fewer instances—it does not solve the problem when aggregate load itself is low.

---

### 6.2 The Difficulty Estimation Cost Is Not Accounted for in the Pipeline Search, and Re-Profiling for New Hardware or Model Architectures Is a Practical Barrier

**The assumption or constraint.** NanoFlow's auto-search engine requires extensive offline profiling before it can generate a pipeline for a given (model, hardware) configuration. Specifically, Section 4.1.1 describes profiling interference-free kernels across all batch sizes from 128 to `B_dense` in multiples of 128, exploring "all possible kernel implementations varying the number of thread blocks, the number of warps, and tile size for GEMM, GEMV, and network kernels." It then profiles pairwise kernel interference to construct the `R → P` mapping in Table 3, testing "∼100 pairs" for GEMM-GEMV alone (Figure 5) after filtering. The paper claims this profiling plus the two-stage MILP search takes approximately 10 minutes for a single pipeline, and that the auto-search is performed "only when the model architecture or workload (input length, output length) undergoes significant changes" (Section 4.1.3).

**The consequence.** The profiling cost—while amortized over long-running deployments—represents a substantial practical barrier to adoption that the paper's headline throughput numbers do not capture. Three concerns compound:

First, **the 10-minute figure is for a single (model, batch size, GPU architecture) tuple**. A serving provider running multiple model variants (e.g., LLaMA-3-8B and LLaMA-3-70B) on multiple GPU generations (e.g., A100 and H100) across multiple workload profiles (different input/output length ratios, since these affect the optimal pipeline structure) would need to run auto-search for each combination. The combinatorial multiplication of these factors could require hours of profiling and search, during which the GPUs are occupied with measurement rather than serving.

Second, **the profiling must be redone when kernels change**. The paper profiles CUTLASS GEMM implementations, but GPU vendor libraries are updated frequently with new kernel implementations. Each update to CUTLASS, cuBLAS, or NCCL potentially invalidates the profiling data, requiring a new round of auto-search. In production environments where kernel libraries are updated for security patches or performance improvements, this creates an ongoing maintenance burden.

Third, **the profiling methodology assumes the hardware's interference characteristics are stable and predictable**. The paper's key simplification—that the `R → P` mapping has "standard deviation within 5% of the mean" across GEMM shapes and batch sizes—may not hold on other GPU architectures with different multi-kernel scheduling policies, cache hierarchies, or memory controllers. The profiling process itself is described only for A100 GPUs; transferring to H100 (with its new Tensor Core design and Thread Block Cluster feature) or to AMD/Intel accelerators may require revisiting the profiling methodology itself, not just re-running it.

**What evidence exists in the paper.** The paper provides no measurement of the total profiling time across all the models evaluated in Figure 11 (six models, which would be ~60 minutes of profiling plus search if each takes ~10 minutes). It also does not quantify the storage cost of profiling data or the engineering effort required to integrate profiling into a deployment pipeline. The claim that "compared to the long-running deployment times, the search time is negligible" (Section 4.1.3) is asserted rather than empirically justified—it depends on deployment durations, model update frequencies, and the number of hardware configurations in a fleet. The paper's sensitivity analysis of the `R → P` mapping ("standard deviation within 5%") is mentioned but the underlying data is not shown, making it impossible to assess whether this stability holds across the full range of GEMM shapes and batch sizes encountered in the evaluated models.

**Mitigation status.** The paper does not attempt to reduce or amortize the profiling cost. It suggests no mechanism for reusing profiling data across similar models (e.g., transferring LLaMA-2-70B profiles to LLaMA-3-70B, which the paper notes have "similar schedules"), for caching profiling results across deployments, or for predicting interference characteristics from hardware specifications rather than measurement. The two-stage MILP decomposition is an optimization of the search phase, not the profiling phase. This is a significant omission because the profiling cost, not the MILP solve time, is likely the dominant practical barrier to deploying NanoFlow across diverse serving environments.

---

### 6.3 The Pipeline Is Static and Cannot Adapt to Changes in Workload Composition Mid-Serving

**The assumption or constraint.** NanoFlow's auto-search generates a single, static pipeline for a given (model, input length, output length) configuration. Section 4.1.3 states that auto-search runs "only when the model architecture or workload (input length, output length) undergoes significant changes." The pipeline structure—the number of nano-operations, their batch sizes, ordering, and resource allocations—is fixed for the duration of a serving deployment. The runtime handles batch formation to keep `B_dense` constant, but it does not modify the nano-operation schedule.

**The consequence.** Real serving workloads are not static. The ratio of prefill to decode tokens varies over time as user behavior changes (e.g., diurnal patterns, shifts between short queries and long conversations). The average input and output lengths can drift as different user populations or applications become dominant. If these shifts are "significant" enough to change the optimal pipeline, NanoFlow would need to stop serving, re-run auto-search, and restart with the new pipeline—a disruptive process. If the shifts are not significant enough to trigger re-optimization, NanoFlow may operate with a suboptimal pipeline for extended periods.

The paper does not define what constitutes a "significant" change in workload. For instance, the pipeline optimized for ShareGPT (average input 246, output 322) might be suboptimal for Splitwise (average input 1155, output 211), since the latter is prefill-heavy and the former is decode-balanced. If a single NanoFlow instance serves a mixture of these workloads whose composition shifts over time, its performance could degrade relative to the static optimum. The fixed pipeline also cannot exploit opportunities that arise from transient workload patterns—for example, a sudden burst of very short requests might benefit from a different nano-batch granularity that minimizes latency rather than maximizing throughput.

**What evidence exists in the paper.** The throughput results (Figure 7b) show that NanoFlow's performance is remarkably stable across the three datasets (1259, 1247, and 1272 tokens/s/GPU for Splitwise, LMSYS-Chat, and ShareGPT respectively), with only 2.0% variation. This suggests that for the particular pipeline generated for LLaMA-2 70B with `B_dense = 2048`, the static pipeline is robust to the workload variations present in these traces. However, this is a single-pipeline evaluation on three specific traces—it does not demonstrate that the same pipeline would remain robust under all workload compositions encountered in production. A more systematic evaluation varying the prefill-to-decode token ratio across a wider range (e.g., 90/10 to 10/90) with the same static pipeline would characterize the robustness envelope. No such experiment is performed.

**Mitigation status.** The paper does not address dynamic pipeline adaptation. It suggests no mechanism for online monitoring of workload composition, no trigger condition for re-running auto-search, and no graceful pipeline transition strategy. The constant `B_dense` strategy in the runtime (Section 4.2.1) stabilizes the batch size but does not adapt the nano-operation schedule. This is fundamentally a static-optimization approach—the auto-search invests substantial profiling and computation upfront in exchange for a fixed schedule, and abandons any ability to adjust online.

---

### 6.4 Kernel Interference Limits Throughput to 68.5% of Optimal, and There Is No Path to Close This Gap on Current Hardware

**The assumption or constraint.** NanoFlow models kernel interference through a GEMM-centric proxy `R` (fraction of peak GEMM performance) and an empirically measured `R → P` mapping (Table 3) that captures how memory-bound and network-bound operations degrade when co-scheduled with compute-bound operations. The paper explicitly acknowledges the limits of this approach: "Unfortunately, kernel interference is unpredictable, as NVIDIA GPUs do not give explicit control over compute, memory, and network bandwidth. Thus, we cannot directly control R_physical" (Section 4.1.1). The `R` proxy is an approximation, and the interference model is simplified to pairwise interactions (compute-memory, compute-network) with the assumption that "the R to P mapping profiled with pairwise interference holds when overlapping three kernels" (Section 4.1.1).

**The consequence.** The gap between NanoFlow's achieved throughput (68.5% of optimal for LLaMA-2 70B) and the theoretical maximum (100%) is **entirely due to kernel interference and nano-batching overhead**, as the paper states in Section 6.5: "Due to kernel interference, NanoFlow provides lower than optimal compute usage." This gap represents a hard ceiling on NanoFlow's performance that cannot be overcome through better pipeline design or smarter auto-search—it is a consequence of the GPU hardware's lack of fine-grained resource partitioning.

The 68.5% figure is therefore not a lower bound that future optimizations could improve beyond, but rather an **upper bound for the current approach on current hardware**. The paper's own ablation shows that overlapping of network-bound operations alone yields only a 1.07× improvement (Figure 9, prefill-only workload), and adding memory-bound overlap yields 1.17× total (decode-heavy workload). These are significant gains but they are bounded—the maximum possible throughput improvement from overlapping is constrained by the fraction of time that compute-bound operations spend waiting for memory/network operations in the sequential baseline, and by how much of that idle time can be recovered given interference effects. The paper does not quantify what fraction of the remaining 31.5% gap to optimal is due to irreducible interference versus potentially recoverable inefficiencies, leaving practitioners unable to assess whether further tuning of the pipeline could yield additional gains or whether they are hitting a hardware-imposed ceiling.

**What evidence exists in the paper.** Figure 10 provides the most direct evidence. The resource utilization timelines for the non-overlapping baseline (Figure 10a) show compute utilization dropping to near zero during memory-bound and network-bound operations. NanoFlow's timeline (Figure 10b) shows compute utilization sustained at high levels throughout, but visibly not at 100%—there are dips where even with overlapping, compute resources are not fully saturated. The ablation in Figure 9 decomposes the overlapping benefit: network-compute overlap contributes ~7% improvement, and adding memory-compute overlap contributes an additional ~10%. The paper does not provide an ablation that varies the aggressiveness of overlapping (e.g., different `R` allocations) to show whether the 68.5% figure is truly optimal for the chosen nano-batch structure or whether alternative resource allocations could push higher.

**Mitigation status.** The paper does not attempt to mitigate kernel interference beyond the auto-search's modeling. It acknowledges the fundamental limitation—"NVIDIA GPUs do not give explicit control over compute, memory, and network bandwidth"—but suggests no hardware-level solution (e.g., GPU virtualization, MIG partitioning, CUDA stream priorities) or software-level workaround (e.g., spatial partitioning of SMs between kernel types). The paper does not discuss whether newer GPU architectures (H100, B100) with features like Thread Block Clusters or improved concurrent kernel scheduling could reduce interference. This is a hardware-imposed limitation that the paper correctly identifies but cannot resolve, making it a genuine boundary condition for practitioners: **NanoFlow will not reach more than ~70% of optimal throughput on A100-class hardware regardless of tuning**.

---

### 6.5 The Evaluation Validates Only Throughput-Oriented, Single-Model, Single-Hardware-Configuration Scenarios

**The constraint.** All experiments in Section 6 are conducted on a single hardware configuration (1× NVIDIA 8×A100 80GB SXM DGX node) with a single model per experiment, and the evaluation metrics focus overwhelmingly on throughput. The paper evaluates six models (Figure 11) and three real-world datasets (Figures 7–8), but:

- **Latency experiments are throughput-oriented.** Even the latency evaluation (Section 6.3, Figure 8) uses NanoFlow configured with its maximum dense batch size (`B_dense = 2048` for LLaMA-2 70B). The paper explicitly states that NanoFlow "targets throughput-oriented scenarios" (Section 6.3). It does not evaluate a latency-optimized configuration with smaller batch sizes, nor does it provide guidance on how to trade off throughput for latency in latency-sensitive deployments.

- **No multi-model serving.** Each experiment runs a single model. In practice, serving platforms often host multiple model variants simultaneously (e.g., a 7B and a 70B model) with requests routed to the appropriate model. The resource contention and scheduling challenges of serving multiple models on shared GPUs are not addressed.

- **No heterogeneous hardware.** All experiments use A100 80GB SXM GPUs. The analytical model suggests the compute-bound classification holds across GPU generations, but actual performance—particularly the interference characteristics that limit throughput to 68.5% of optimal—is hardware-specific and untested on other architectures.

- **No integration with existing serving frameworks.** NanoFlow is evaluated as a standalone system, not as a plugin or module within an existing serving stack. The paper claims that NanoFlow's techniques can be integrated into existing systems (Section 7 discusses prior work as complementary), but provides no integration evaluation—no measurement of how much throughput would improve if, say, vLLM adopted NanoFlow's overlapping scheduling while retaining its request scheduler and memory manager.

**The consequence.** A practitioner evaluating NanoFlow for deployment cannot answer several critical questions from the paper's evaluation:

1. **What latency does NanoFlow achieve with a latency-optimized configuration?** The current evaluation shows NanoFlow matching or slightly exceeding baselines' latency at low request rates, but this is with a 2048-token batch size that is deliberately oversized for latency-sensitive scenarios. With a smaller batch size, would NanoFlow's overlapping still provide benefits, or would the reduced batch size push the workload toward memory-bound territory where overlapping is less effective?

2. **How does NanoFlow perform on H100 or B200 GPUs?** The paper motivates its work partly through the "compute-bound" classification that holds across GPU generations (Table 1), but provides no H100/H200/B200 evaluation. Given that the H100's Compute/MemBW ratio is 295 (vs. 156–200 for A100), the workload should be even more compute-bound on H100, potentially making overlapping even more beneficial. But the H100's different SM architecture and concurrent kernel scheduling behavior might change interference patterns in ways that the A100-tuned auto-search does not capture.

3. **Can NanoFlow be adopted incrementally, or does it require replacing the entire serving stack?** The paper presents NanoFlow as an end-to-end system with its own batch formation, KV-cache management, and scheduling. A practitioner using vLLM or TensorRT-LLM for their production deployment cannot simply "turn on" NanoFlow's overlapping—they would need to migrate their entire serving infrastructure. The paper does not demonstrate that the overlapping mechanism can be extracted and composed with existing systems' schedulers and memory managers.

**What evidence exists in the paper.** All the evidence is within the homogeneous A100, single-model, throughput-oriented envelope described above. The paper acknowledges the throughput orientation ("NanoFlow targets throughput-oriented scenarios and therefore employs a large dense batch size," Section 6.3), but does not frame this as a limitation. The single-hardware evaluation is not explicitly acknowledged as a scope constraint; the analytical model (Figures 2–3, Table 1) implicitly suggests generalizability, but no cross-hardware experiments validate this.

**Mitigation status.** The paper partially mitigates the single-model concern by evaluating six diverse models (Figure 11): 8B dense, 70B dense (three variants), and MoE (one variant). This demonstrates that the auto-search methodology generalizes across architectures within the A100 constraint. The single-hardware and throughput-only limitations are not mitigated—they are inherent in the paper's experimental scope. The integration concern is not addressed; the paper treats NanoFlow as a complete replacement for existing serving engines rather than demonstrating composability. A practitioner interested in adopting NanoFlow's overlapping technique within an existing vLLM or TensorRT-LLM deployment would need to perform this integration themselves without guidance from the paper.

---

### 6.6 Multi-Round Conversation Support Imposes a 3% Throughput Penalty, and the Tradeoff Between Offloading Benefit and Overhead Is Not Fully Characterized

**The assumption or constraint.** NanoFlow's KV-cache offloading mechanism (Section 4.2.2) enables multi-round conversations by copying KV-cache from GPU to CPU/SSD after each token is generated, and reloading it when a follow-up request arrives. The paper acknowledges that "enabling offloading would slow down the pipeline by 3.0% due to kernel interference caused by KV-cache movement" (Section 6.4, Figure 9). However, the evaluation of offloading is limited to a single throughput measurement (Figure 9, "NanoFlow-offload" bar) and a qualitative claim that offloading "can reduce 3.02× compute for multi-round LMSYS-Chat workloads."

**The consequence.** The 3.0% throughput penalty is measured in a single configuration (Input 512 / Output 512, constant lengths) and is attributed to "kernel interference." But the magnitude of this penalty likely depends on:

- **The offloading volume per iteration.** Longer sequences produce larger KV-cache entries to offload. A workload with average output length 1024 would offload 2× more data per iteration than one with output length 512, potentially increasing the interference penalty.
- **The ratio of multi-round to single-round requests.** If only a small fraction of requests are multi-round, the offloading overhead is paid for all requests (since KV-cache is offloaded for every token regardless of whether a follow-up will ever arrive) while the benefit (avoided recomputation) accrues only to the multi-round subset. The paper does not model this cost-benefit tradeoff as a function of multi-round request fraction.
- **The host memory and SSD bandwidth.** The paper uses "NUMA-aware thread-binding" to optimize offloading, but does not characterize how offloading performance degrades when host memory bandwidth is contended (e.g., multiple NanoFlow instances on the same node competing for CPU memory bandwidth) or when SSDs are involved (the LRU eviction to SSD when CPU memory is full).

The "3.02× compute reduction" claim is qualitative and unreferenced—the paper does not explain how this figure is derived, what baseline it compares against (full recomputation from scratch? partial recomputation?), or whether it accounts for the overhead of loading KV-cache from CPU/SSD when a new round arrives.

**What evidence exists in the paper.** The only quantitative evidence is the single throughput bar in Figure 9 ("NanoFlow-offload" at 1244 tokens/s/GPU vs. 1290 for NanoFlow without offloading, a 3.0% degradation) and the 3.02× compute reduction claim in Section 6.4. There is no latency evaluation with offloading enabled (Figure 8 presumably runs without offloading), no measurement of offloading bandwidth or host memory pressure, and no sensitivity analysis varying sequence length or multi-round request fraction. The paper states that "the host and the GPU hold the same copy of on-the-fly requests' KV-cache" (Section 4.2.2), implying that offloading duplicates KV-cache memory, but does not quantify the memory overhead.

**Mitigation status.** The paper does not mitigate the offloading overhead or characterize its sensitivity to workload parameters. It presents offloading as an optional feature that can be enabled for multi-round applications, but provides insufficient data for a practitioner to decide whether the 3.0% throughput penalty (and unknown tail latency impact) is worth the recomputation savings in their specific workload mix. The LRU-based hierarchical cache management is described architecturally but not evaluated—there are no experiments showing cache hit rates, SSD offload latency, or the performance impact of KV-cache eviction under memory pressure. This is a consequential gap because multi-round conversation is a dominant use case for LLM serving (ShareGPT, LMSYS-Chat are both conversation datasets), and a practitioner serving conversational workloads must understand the offloading tradeoff to configure NanoFlow appropriately.

## 7. Implications and Future Directions
- How it changes the landscape
  - Establishes a compute-centric view of LLM serving with a simple optimality benchmark (Eq. 5). This reframes system design around overlapping heterogeneous resources rather than optimizing any single stage in isolation.
  - Demonstrates that intra-device scheduling (below the request/iteration level) is a powerful lever—orthogonal to request-level batching, paged attention, quantization, or cluster-level disaggregation.

- Follow-on research enabled
  - Adaptive, online auto-search: make `R` allocations and nano-batch sizes react to live traffic, mixture-of-lengths, and interference signals.
  - Richer interference models: extend from pairwise to multi-kernel and cross-SM/cache contention models; incorporate MIG/MPS multi-tenancy interference.
  - Integration with model compression: jointly optimize quantization/sparsity (e.g., QServe, ATOM, QUEST) and nano-pipeline overlap to further raise the Eq. 5 ceiling by increasing effective TFLOP/s and reducing `P_model`.
  - Compiler support: capture nano-batch duplication and overlap as first-class IR transformations and enable whole-graph scheduling across layers.

- Practical applications
  - High-throughput batch workloads (batch inference, data labeling, RAG precomputations) where large steady-state batches are available.
  - Multi-round assistants: KV offloading and fast reload benefit chat services that serve many concurrent sessions with pauses (Sec. 4.2.2).
  - Heterogeneous clusters: the auto-search and the Eq. 5 bound provide a principled way to plan capacity and compare hardware (Table 1) independent of specific workloads.

> Bottom line: By proving that modern LLM serving is typically compute-bound and then systematically filling compute bubbles through intra-device overlap, NanoFlow moves practical serving throughput much closer to a clear theoretical limit, with broad applicability across dense and MoE models (Figs. 7 and 11).
