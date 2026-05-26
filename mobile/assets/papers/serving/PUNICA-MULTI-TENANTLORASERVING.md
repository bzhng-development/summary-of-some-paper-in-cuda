# PUNICA: MULTI-TENANT LORA SERVING

**ArXiv:** [2310.18547](https://arxiv.org/abs/2310.18547)

## 🎯 Pitch

Punica introduces a novel system for serving multiple Low-Rank Adaptation (LoRA) variants of large language models (LLMs) on shared GPU clusters by maintaining just a single copy of the base model in GPU memory. Its key innovation is a new CUDA kernel—Segmented Gather Matrix-Vector Multiplication (SGMV)—that allows efficient batching of requests across different LoRA models during decoding, dramatically boosting GPU throughput. This approach enables up to 12× higher serving throughput compared to existing solutions with minimal added latency, making multi-tenant, cost-effective LLM deployment at scale practical and resource-efficient.

---

## 1. Executive Summary

Punica introduces a system for serving multiple LoRA models efficiently on a shared GPU cluster by enabling batched execution of different LoRA models against a single copy of the underlying pre-trained model. Evaluating on Llama 2 models (7B, 13B, and 70B) with A100 GPUs, Punica's core mechanism is a new CUDA kernel called **Segmented Gather Matrix-Vector Multiplication (SGMV)** — which groups requests by LoRA model within a batch and dispatches them to GPU Tensor Cores with specialized schedules for expand and shrink operations — paired with a scheduler that consolidates multi-tenant workloads onto the smallest set of active GPUs and migrates requests to free resources. The system achieves 12× higher throughput over state-of-the-art LLM serving systems while adding only 2ms latency per token, establishing that multi-tenant LoRA serving can approach the throughput of single-model serving even under the most challenging workload where every request targets a distinct LoRA model.

## 2. Context and Motivation

### The Core Problem: Serving Many Specialized Models Wastes GPUs

The central problem Punica addresses is deceptively simple: **how do you efficiently serve hundreds or thousands of fine-tuned model variants that all share the same pre-trained backbone?** Low-Rank Adaptation (LoRA) (Hu et al., 2022) has made it cheap to create these variants — each one adds only 0.1% to 1% of the original model's parameters — but actually deploying them at scale creates a resource allocation nightmare that existing serving systems cannot handle.

To understand why, consider the naive approach. If you need $k$ GPUs to serve one LoRA model at acceptable throughput, serving $n$ different LoRA models would seemingly require $k \times n$ GPUs. For a cloud provider with thousands of tenants, each of whom has fine-tuned their own LoRA variant, this linear scaling is economically infeasible. The GPUs sit idle most of the time because each one is dedicated to a model that receives only sporadic traffic. This is the **multi-tenant serving gap**: the infrastructure cost of serving many specialized models is proportional to the number of models, not the aggregate demand.

The paper identifies this as a structural inefficiency. All $n$ LoRA models share the identical pre-trained weights — the backbone model is literally the same matrix $W$ in every case. Loading $n$ copies of that backbone into $n$ sets of GPUs is redundant in both memory and computation. The challenge is architectural: how do you design a system where one copy of the backbone serves all $n$ LoRA variants simultaneously, without each variant's requests interfering with the others?

### Why This Problem Matters Now

The paper is responding to a specific inflection point in ML deployment. LoRA has become the dominant fine-tuning paradigm for several reasons that make the multi-tenant serving problem both urgent and tractable:

**The economics of specialization.** LoRA reduces the cost of fine-tuning an LLM from "requires a cluster of GPUs and a team of ML engineers" to "runs on a single GPU with a modest dataset." This democratization means that the number of fine-tuned model variants in circulation is exploding. The HuggingFace PEFT library (Mangrulkar et al., 2022), which implements LoRA and similar methods, has been widely adopted, creating a large and growing population of specialized models that need to be served. Every ML provider — from cloud platforms to in-house infrastructure teams — faces the question of how to host these models economically.

**The decode-stage bottleneck.** The paper observes (Section 2.1, Figure 1) that the decode stage dominates serving cost. During decoding, the GPU processes one token at a time per request with an input that is a single vector, leaving the GPU's massive parallel compute capacity severely underutilized. Batching multiple requests together is the standard solution — when batch size increases from 1 to 32, decode latency rises only modestly (e.g., from 11ms to 13ms for short sequences on an A100), meaning throughput per GPU can be increased by roughly an order of magnitude without proportional latency penalties. However, this batching trick only works when all requests in the batch are running the **same model**. Existing systems cannot batch requests that require different LoRA add-ons, because each request would need to traverse a different set of weight matrices $A_i B_i$. This forces a painful choice: either run large batches (good throughput) but only for one LoRA model at a time, or serve multiple LoRA models concurrently (good coverage) but with tiny (or single-request) batches and terrible GPU utilization.

**The memory opportunity.** A LoRA model adds approximately 1% to the backbone model's weight footprint. On a Llama 2 7B model, the backbone consumes roughly 14GB in FP16, while each LoRA add-on is on the order of 140MB. An A100 80GB GPU can easily hold the backbone plus **hundreds** of LoRA weight sets in memory simultaneously. The paper argues that this memory abundance should be exploited: rather than dedicating a GPU to one LoRA model, load the backbone once and swap LoRA weights on demand. However, existing systems provide no mechanism to compute the LoRA add-on efficiently when different requests in a batch need different $A_i B_i$ matrices. The memory opportunity and the batching opportunity are simultaneously present but mutually exclusive in prior art.

### Where Existing Approaches Fall Short

The paper identifies specific limitations across four categories of prior work, each of which fails to address the multi-tenant LoRA serving problem in a different way.

**General LLM serving systems (vLLM, Orca, FasterTransformer, DeepSpeed).** These systems achieve high throughput by batching requests for the **same** model. vLLM (Kwon et al., 2023) introduces PagedAttention for memory-efficient KvCache management and continuous batching, allowing requests to enter and leave batches dynamically without waiting for the longest sequence to finish. Orca (Yu et al., 2022) demonstrates the core insight that batching decode steps dramatically improves GPU utilization. However, these systems operate under a fundamental assumption: every request in a batch executes the identical model computation. They offer no mechanism to inject per-request weight variations like LoRA add-ons. As the paper's experiments show (Section 7.2, Figure 11), when workload diversity increases — for instance, when each request targets a distinct LoRA model — these systems degrade to batch-size-1 execution because they can only group requests for the same model. Throughput collapses. In the Distinct workload case, vLLM achieves only a fraction of its single-model throughput because every request runs in isolation.

**Parameter-efficient fine-tuning serving (PetS).** PetS (Zhou et al., 2022) is the closest prior work conceptually. It batches requests for different adapters (including LoRA variants, adapter layers, and other PEFT methods) on a single GPU, sharing the pre-trained model in memory. However, the paper identifies a critical limitation: PetS **does not enable concurrent execution** of different models. It allows weight sharing in memory — the backbone sits in GPU RAM and different adapters are loaded as needed — but it processes requests for different adapters sequentially, not simultaneously within a batch. This means PetS still suffers from low GPU utilization when serving diverse LoRA models because it cannot mix different models' requests in the same computation step. The paper positions Punica as solving the problem PetS identifies but cannot fully address: true concurrent multi-model execution within a single batch.

**The HuggingFace Transformers KvCache layout problem.** The paper diagnoses a subtler limitation that affects many systems beyond just HuggingFace: the memory layout of the Key-Value cache (Section 5.4). In HuggingFace Transformers, the KvCache is organized with sequence length as an innermost dimension that gets concatenated at each decode step, requiring a full copy of the entire cache on every append. More importantly, the batching dimension is not the outermost dimension, meaning requests that enter a batch together **must stay together** until every request reaches its stopping condition. As Figure 6 illustrates, when four requests with different output lengths are batched together, the shorter requests continue running wasted decode steps after they would naturally terminate, because the inseparable KvCache prevents removing them from the batch individually. FasterTransformer and DeepSpeed suffer from the same structural constraint. vLLM solves this via PagedAttention with an outer batching dimension and paged memory — Punica adopts this approach — but the paper makes clear that even with this fix, the multi-model batching problem remains unsolved by prior systems.

**Lack of specialized CUDA kernels for multi-model batching.** At the lowest level, no prior system provides a GPU kernel that can efficiently compute per-request weight additions within a batch. The naive approach is a for-loop over each LoRA model's requests, which reduces to batch-size-1 execution and is catastrophically slow (as shown in Figure 8a, the Loop baseline's latency grows linearly with batch size because each request is processed sequentially). A more sophisticated baseline, Gather-BMM, stacks the appropriate $A_i B_i$ matrices for each request into a tensor and uses `torch.bmm()` for batched matrix multiplication. But this incurs substantial extra I/O: the Gather step reads $n$ separate weight matrices and writes them into a stacked tensor, and then the BMM step reads that stacked tensor back, effectively doubling the weight data movement compared to a solution that reads each weight matrix once and applies it only to the requests that need it. As the roofline analysis in Figure 7 shows, the Distinct workload (each request has a unique LoRA model) is I/O-bound — making the Gather-BMM's extra memory traffic particularly damaging.

### How Punica Positions Itself

The paper establishes three design guidelines (Section 1) that frame its contribution relative to prior work:

- **(G1) Consolidation:** Multi-tenant LoRA serving workloads should be packed onto as few GPUs as possible, maximizing utilization and minimizing idle resources. This directly opposes the "one model per GPU" deployment pattern.

- **(G2) Cross-model batching:** Batching must work for **different** LoRA models, not just identical ones. The paper identifies this as the central technical challenge — prior systems can batch only identical-model requests, and the paper's primary contribution (SGMV) is the mechanism that breaks this constraint.

- **(G3) Decode-stage focus:** Because the decode stage dominates serving cost (each output token requires a full model forward pass, while the prefill stage is amortized over the prompt length), the system only needs to optimize decode-stage batching. Prefill and other operations can use simpler techniques (e.g., on-demand LoRA weight loading).

Punica's position is not to replace existing LLM serving systems but to **add a capability they lack**: the ability to mix requests for different LoRA models within the same GPU batch without sacrificing throughput. The paper builds on prior work extensively — it adopts PagedAttention from vLLM for KvCache management, FlashInfer for optimized self-attention, Megatron-style tensor parallelism for large models, and continuous batching principles from Orca — and adds SGMV as the missing piece that makes multi-tenant batching possible. The contribution is specific and targeted: solve the per-request weight addition problem at the CUDA kernel level, then build scheduling policies that exploit this capability to consolidate workloads across a GPU cluster.

The paper also implicitly positions itself against the "just scale out" approach. A cloud provider could theoretically handle $n$ LoRA models by provisioning $n$ separate serving instances, but the paper argues this is economically wasteful because it fails to exploit the weight sharing that makes LoRA efficient in the first place. Punica's 12× throughput improvement over state-of-the-art systems (Figure 11) is not an incremental optimization — it represents the difference between linear scaling with the number of models (baseline systems) and near-constant throughput regardless of model count (Punica). This transforms multi-tenant LoRA serving from a capacity-provisioning problem into a scheduling problem, where the constraint is aggregate demand rather than model count.

## 3. Technical Approach

### 3.1 Reader Orientation

Punica is a GPU cluster serving system that lets many different fine-tuned LoRA models share a single copy of the pre-trained backbone model. The problem it solves is that existing LLM serving systems can batch requests only when they target the same model — once you have multiple LoRA variants, throughput collapses because each variant must run in its own tiny batch. The shape of the solution is a new CUDA kernel (SGMV) that makes per-request weight additions fast enough to batch different LoRA models together, paired with a scheduler that packs diverse requests onto the minimum number of GPUs and migrates work to keep them fully utilized.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components:

1. **Frontend servers** — expose a RESTful API to end-users, accept prompts tagged with a LoRA model identifier, and stream generated tokens back.

2. **Scheduler** — the central decision-maker that receives all requests from frontends and dispatches them to specific GPUs; it also periodically cancels and re-adds requests (migration) to consolidate load and free idle GPUs.

3. **Runner** — one per GPU server, communicating with the scheduler and managing the GPU subprocesses. Each GPU gets a Python subprocess that executes the actual model inference.

4. **SGMV CUDA kernel** — the core technical innovation. It computes the LoRA add-on operation `$\vec{y} \mathrel{+}= \vec{x}AB$` for a batch of requests where each request may need a different pair of LoRA matrices `$A_i, B_i$`. It operates in two launches: shrink (`$\vec{v} \mathrel{+}= \vec{x}A_i$`) and expand (`$\vec{y} \mathrel{+}= \vec{v}B_i$`).

5. **Paged KvCache with outer batching dimension** — the memory management subsystem organizes the per-request Key-Value cache as individually pageable blocks with the batching dimension outermost, enabling requests to enter and leave batches independently without wasted decode steps.

Information flows as follows: a user sends a prompt with a LoRA model ID to the frontend → the frontend forwards it to the scheduler → the scheduler selects the GPU with the largest current batch size that still has room for the new request's KvCache → the runner on that GPU adds the request to the next batch invocation → if the LoRA model's weights aren't already on the GPU, an asynchronous host-to-device memory copy loads them (completing before the request's first decode step starts) → the GPU runs a forward pass where the backbone projection is computed with standard batched GEMM and the LoRA add-on is computed with SGMV, which groups requests by LoRA model ID internally → generated tokens stream back through the runner to the scheduler to the frontend to the user → when a GPU runs out of KvCache space, the scheduler migrates the newest request to another GPU by cancelling it on the source GPU and re-adding it (with recomputed KvCache) on the destination GPU.

### 3.3 Roadmap for the Deep Dive

- **First**, the SGMV operator's formal semantics — what it computes, why the two-launch design matters, and how it differs from the naive loop and Gather-BMM alternatives.
- **Second**, the CUDA kernel schedule for SGMV-shrink and SGMV-expand — how the threadblock grid is organized around LoRA indices and why the expand kernel splits on output dimensions while the shrink kernel uses Split-K parallelism.
- **Third**, the batch layout construction — how requests are ordered within a batch so that requests sharing a LoRA model become consecutive, enabling a single SGMV invocation to handle multiple LoRA groups simultaneously.
- **Fourth**, the new request scheduling algorithm — the greedy policy that routes requests to the busiest GPU that can accommodate them, why this consolidates load, and the constraints that bound it.
- **Fifth**, on-demand LoRA weight loading — how asynchronous PCIe copies overlap with GPU computation to hide model-switching latency entirely.
- **Sixth**, the request migration mechanism — why cancellation and recomputation are used instead of copying KvCache, and how the scheduler maintains FCFS semantics under migration.
- **Seventh**, the KvCache memory layout — why the batching dimension must be outermost, how paging eliminates fragmentation, and how this enables continuous batching of requests with different output lengths.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **systems paper** whose core idea is that serving many LoRA models efficiently requires a specialized CUDA kernel that lets different models' LoRA add-ons be computed in the same GPU batch without serializing, and that this kernel enables a scheduling policy that consolidates diverse workloads onto a minimal number of GPUs.

---

#### The SGMV Operator: What It Computes

The computation Punica must perform for every dense projection in every transformer layer, for each request in a batch, is the LoRA-augmented forward pass:

$$\vec{y} = \vec{x}W + \vec{x}AB$$

where `$W \in \mathbb{R}^{h_1 \times h_2}$` is the pre-trained backbone weight (shared by all LoRA models), `$A \in \mathbb{R}^{h_1 \times r}$` and `$B \in \mathbb{R}^{r \times h_2}$` are the LoRA matrices specific to the request's model, `$r$` is the LoRA rank (e.g., 16), `$\vec{x}$` is the input activation vector for this request at this layer, and `$\vec{y}$` is the output. The term `$\vec{x}W$` is the backbone computation — it is identical for every request regardless of which LoRA model they target. The term `$\vec{x}AB$` is the LoRA add-on — it is different for each LoRA model.

When a batch contains requests for different LoRA models, the backbone computation is straightforward: concatenate all `$\vec{x}$` vectors into a matrix `$X$` and compute `$XW$` with a standard batched GEMM call (cuBLAS). The LoRA add-on is the problem: each row of `$X$` needs to be multiplied by a different `$A_i B_i$` pair.

The paper observes that the LoRA add-on `$\vec{y}_{\text{lora}} = \vec{x}AB$` can be factorized into two sequential matrix-vector multiplications:

$$\vec{v} = \vec{x}A \quad\quad \vec{y}_{\text{lora}} = \vec{v}B$$

where `$\vec{v} \in \mathbb{R}^r$` is a low-dimensional intermediate vector and the addition `$\vec{y} \mathrel{+}= \vec{y}_{\text{lora}}$` is performed in-place on the backbone output. This factorization is essential because it separates the operation into two phases with very different computational characteristics:

- **Phase 1 (shrink):** `$\vec{v} = \vec{x}A$` maps from a high-dimensional input (`$h_1$`, typically 4096 or larger) to a low-rank intermediate (`$r$`, typically 16). The output dimension is small — this is a reduction operation.

- **Phase 2 (expand):** `$\vec{y} \mathrel{+}= \vec{v}B$` maps from the low-rank intermediate (`$r$`) back to a high-dimensional output (`$h_2$`, typically 4096). The input dimension is small — this is an expansion operation.

Now consider a batch of `$s_n$` requests belonging to `$n$` different LoRA models. Let the requests be ordered such that all requests for the same LoRA model are contiguous. Define segment boundaries `$s_0 = 0 < s_1 < \cdots < s_{n-1} < s_n$` where `$s_i$` is the index after the last request for LoRA model `$i$`. The requests in `$(s_{i-1}, s_i]$` all share the same weight matrices `$A_i, B_i$`.

For the shrink phase, the computation for model `$i$` is:

$$\begin{pmatrix} \vec{v}_{s_{i-1}+1} \\ \vdots \\ \vec{v}_{s_i} \end{pmatrix} = \begin{pmatrix} \vec{x}_{s_{i-1}+1} \\ \vdots \\ \vec{x}_{s_i} \end{pmatrix} A_i$$

This is a matrix-matrix multiplication between a `$(s_i - s_{i-1}) \times h_1$` input block and an `$h_1 \times r$` weight matrix. For the expand phase:

$$\begin{pmatrix} \vec{y}_{s_{i-1}+1} \\ \vdots \\ \vec{y}_{s_i} \end{pmatrix} \mathrel{+}= \begin{pmatrix} \vec{v}_{s_{i-1}+1} \\ \vdots \\ \vec{v}_{s_i} \end{pmatrix} B_i$$

This is a matrix-matrix multiplication between a `$(s_i - s_{i-1}) \times r$` input block and an `$r \times h_2$` weight matrix.

**What this formulation achieves:** by grouping requests by LoRA model ID and applying the same `$A_i$` (or `$B_i$`) to a block of inputs, the per-model computation becomes a small batched GEMM rather than a sequence of independent matrix-vector products. The operational intensity (FLOPs per byte of memory I/O) increases with the number of requests per model within the batch, because the weight matrix `$A_i$` is read once and reused across multiple input vectors.

**Why this factorization matters:** without the two-phase decomposition, the naive approach would compute `$\vec{x}(A_i B_i)$` as a single operation per request. This is strictly worse for two reasons. First, if `$A_i B_i$` is precomputed into an `$h_1 \times h_2$` matrix (which is the same shape as `$W$`), it loses the low-rank structure and becomes a large matrix-vector multiply with no weight reuse opportunity across requests. Second, even if computed on-the-fly as `$\vec{x}A_i B_i$`, the intermediate `$\vec{v}$` needs to be materialized anyway, so the factorization costs nothing and enables the block-wise batching that SGMV exploits.

---

#### SGMV CUDA Kernel Schedule

The paper classifies SGMV into two variants with different GPU scheduling strategies: **SGMV-shrink** (for `$\vec{v} = \vec{x}A$`) and **SGMV-expand** (for `$\vec{y} \mathrel{+}= \vec{v}B$`).

**Binding LoRA index to BLOCKIDX.Y.** In both kernel variants, the LoRA model index is bound to the CUDA grid's Y-dimension (`BLOCKIDX.Y`). This means that threadblocks in the same Y-row handle a single LoRA model's weight matrix, while different Y-rows handle different LoRA models. Within each Y-row, the computation is a standard matrix multiply between the block of input vectors belonging to that model and the model's weight matrix. This design cleanly separates the computation by model, with no cross-model synchronization needed during the matrix multiply.

**SGMV-expand schedule.** For the expand kernel (`$\vec{y} \mathrel{+}= \vec{v}B$`), the weight matrix `$B \in \mathbb{R}^{r \times h_2}$` has a tiny input dimension (`$r$`, typically 16) and a large output dimension (`$h_2$`, typically 4096). The output dimension is large enough to split across threadblocks for parallelism. The paper partitions `$B$` along the output feature dimension:

$$B = \begin{bmatrix} B^{(1)} & \cdots & B^{(m)} \end{bmatrix}$$

where each `$B^{(j)} \in \mathbb{R}^{r \times (h_2/m)}$` is a column-wise slice. Different threadblocks compute `$\vec{y}^{(j)} = \vec{v}B^{(j)}$` independently for different output feature ranges. The concatenation of `$\vec{y}^{(j)}$` across all threadblocks forms the complete output `$\vec{y}$`. Each threadblock uses GPU Tensor Cores for the matrix multiply, since the `$\vec{v}B^{(j)}$` operation has the shape of a (small-batch) `$\times$` (small-input) `$\times$` (moderate-output) matrix multiplication, which maps well to Tensor Core warp-level matrix multiply-accumulate (WMMA) instructions when the batch of requests belonging to the same LoRA model provides multiple rows.

**SGMV-shrink schedule.** For the shrink kernel (`$\vec{v} = \vec{x}A$`), the weight matrix `$A \in \mathbb{R}^{h_1 \times r}$` has a large input dimension and a tiny output dimension (`$r$`). The output dimension is too thin to split across threadblocks meaningfully — there are only `$r$` elements of `$\vec{v}$` to compute per request. To still achieve adequate parallelism, the paper adopts a **Split-K** strategy (Thakkar et al., 2023). The weight matrix `$A$` is partitioned along the **input** feature dimension:

$$A = \begin{bmatrix} A^{(1)} \\ \vdots \\ A^{(k)} \end{bmatrix}$$

where each `$A^{(j)} \in \mathbb{R}^{(h_1/k) \times r}$` is a row-wise slice. Different threadblocks compute partial sums `$\vec{v}^{(j)} = \vec{x}^{(j)} A^{(j)}$` where `$\vec{x}^{(j)}$` is the corresponding slice of the input vector. After all partial sums are computed, a **grid synchronization** barrier ensures all threadblocks have finished, followed by a **cross-threadblock reduction**:

$$\vec{v} = \sum_{j=1}^{k} \vec{v}^{(j)}$$

This reduction aggregates the partial results into the final `$\vec{v}$`. The Split-K strategy trades extra computation (the reduction step) for increased parallelism — without it, the `$r$`-element output would constrain the kernel to very few threadblocks, leaving most GPU compute units idle.

**Tensor Core usage.** Both SGMV-expand and SGMV-shrink use GPU Tensor Cores for the matrix multiplication — but only when there are multiple requests per LoRA model. Tensor Cores operate on matrix tiles (typically `$16 \times 16$` for FP16), which require the batch dimension within a LoRA group to provide enough rows to form full tiles. When the batch size per LoRA model is 1, the computation degrades to a matrix-vector product with extremely low arithmetic intensity.

**The degenerate case: one request per LoRA model.** When every request in the batch targets a different LoRA model (the "Distinct" workload), each `$(s_i - s_{i-1}) = 1$` and the per-model computation is a pure matrix-vector multiplication. This is completely I/O-bound — the GPU spends essentially all its time waiting for memory transfers, not computing. The paper designs a **separate schedule** for this case that maximizes memory bandwidth utilization and explicitly **does not use Tensor Cores**. Tensor Cores require data organized into matrix tiles, which imposes overhead for layout transformation; when the operational intensity is already at the I/O-bound floor, that overhead is pure waste. Instead, the degenerate schedule uses standard CUDA cores with memory-coalesced access patterns to read the weight matrix and input vector as efficiently as possible.

**Roofline analysis grounding.** The paper's roofline model (Section 7.1, Figure 7) formalizes this distinction. The arithmetic intensity of SGMV is:

$$\text{Intensity} = \frac{\text{FLOP}}{\text{I/O}} = \frac{s_n \cdot h_i \cdot h_o \cdot 2}{(s_n \cdot (h_i + h_o) + n \cdot h_i \cdot h_o) \cdot 2}$$

where `$s_n$` is the total number of inputs, `$n$` is the number of distinct LoRA models, `$h_i$` is the input dimension, `$h_o$` is the output dimension, and the factor of 2 in both numerator and denominator comes from multiply-add FLOP accounting and 16-bit element size, respectively. For the Distinct case (`$n = s_n$`), the weight I/O term `$n \cdot h_i \cdot h_o$` dominates and arithmetic intensity is constant regardless of batch size. For the Identical case (`$n = 1$`), the weight I/O term is negligible and arithmetic intensity grows linearly with batch size. The kernel design handles both regimes — the Split-K and output-splitting strategies provide parallelism when arithmetic intensity is too low for Tensor Cores to matter (Distinct case), and the Tensor Core path kicks in when per-model batch sizes are large enough to benefit.

---

#### Batch Layout Construction

Before each GPU model invocation, Punica constructs the input batch from the working set of active requests. The construction involves two ordering decisions that are critical for SGMV's efficiency.

**Separating prefill and decode.** New requests in the prefill stage and ongoing requests in the decode stage are concatenated into a single batch for dense projections and the LoRA add-on. Prefill requests are placed at the **beginning** of the batch, and decode requests at the **end**. This separation is required by the self-attention computation: the prefill stage needs to compute attention over the full prompt sequence (requiring a `BatchPrefill` kernel), while decode steps attend over cached past states (requiring a `BatchDecode` kernel). However, for the dense projections and LoRA add-on, all tokens — prefill prompt tokens and decode single tokens — are treated as an undifferentiated batch of input vectors. This means the LoRA batching benefits from the combined batch size of both prefill and decode requests, even though the attention computation must handle them separately.

**Grouping by LoRA model.** Within the combined batch, requests are ordered such that those sharing the same LoRA model are **consecutive**. This grouping is what generates the segment indices `$(s_0, s_1, \ldots, s_n)$` that SGMV consumes. The paper notes that the tail of prefill requests and the head of decode requests can share a LoRA model if possible, enabling cross-stage grouping for that model. The segment indices are computed once per batch, before the model invocation starts, and remain constant for the entire forward pass — they are reused at every transformer layer (7 layers × L for the full model, where each layer has Query, Key, Value, Output, and 3 MLP projections, all with LoRA add-ons).

**BatchLen metadata.** Alongside the input tensor, Punica passes a `BatchLen` struct that encodes the boundaries between prefill and decode requests and the starting indices of each individual prefill request. This struct is constructed once per batch and reused at every layer, avoiding `$L$` recomputations. The decode portion is described simply by a count (all decode requests contribute one token each), while the prefill portion needs per-request start indices because prompt lengths vary.

---

#### Scheduling New Requests

The Punica scheduler maintains global state: for each GPU, it tracks the current working set of requests (which determines the batch size and composition), the amount of KvCache memory consumed, and the remaining free memory available for new requests' KvCache.

**Greedy consolidation policy.** When a new request arrives, the scheduler selects the GPU with the **largest current batch size** that satisfies two constraints:

1. The GPU has not yet reached the maximum batch size limit (set to 32, profiled on A100 GPUs to balance throughput and per-token latency — beyond 32, latency grows disproportionately relative to throughput gains).

2. The GPU has enough remaining memory to allocate KvCache pages for the new request's expected sequence length.

If multiple GPUs satisfy these constraints, the one with the highest GPU UUID wins (a deterministic tiebreaker). If no GPU can accept the request, it is queued and serviced in first-come-first-serve (FCFS) order when capacity becomes available.

**Why this policy consolidates load.** The paper makes an explicit argument about the dynamics of this scheduling rule. Because new requests are routed to the busiest GPU, a busy GPU stays busy — it keeps receiving new requests as old ones terminate, maintaining a full batch. A lightly-loaded GPU tends to shed load: its existing requests eventually complete without being replaced by new ones, driving its batch size toward zero. An idle GPU stays idle. This creates a natural consolidation pattern where only a subset of GPUs are active at any given time, and the rest can be released back to the cloud provider. The scheduler does not need an explicit "consolidation" pass — consolidation emerges from the greedy routing rule.

**Cluster auto-scaling.** The paper describes how this consolidation enables straightforward cluster allocation decisions. If no lightly-loaded GPU exists (i.e., all active GPUs are at or near their batch size limit or memory limit), Punica should request more GPUs from the cloud provider. Conversely, any GPU server with zero load (all its GPUs idle) can be returned to the provider. The scheduling policy creates clear signals for scaling up and down.

**Maximum batch size rationale.** The limit of 32 is chosen empirically. Figure 1 shows that at the decode stage, increasing batch size from 1 to 32 adds only 2ms (from 11ms to 13ms) for short sequences, but longer sequences see steeper increases (from 17ms to 34ms). A batch of 32 represents the "sweet spot" where throughput is high and the latency penalty remains acceptable. The paper does not claim this is optimal for all hardware — it is specific to the A100 profile.

---

#### On-Demand LoRA Weight Loading

One of Punica's design guidelines (G3) is that only the decode stage needs optimization; other aspects can use straightforward techniques. On-demand loading of LoRA weights follows this principle.

**Loading cost.** A LoRA model consists of matrices `$A$` and `$B$` for each dense projection in the transformer. At rank 16 and FP16 precision, the total LoRA weight size is approximately 1% of the backbone model weight. Loading a single layer's LoRA weights from CPU main memory to GPU memory takes roughly 50µs over PCIe Gen4 x16 (the paper states this is bounded by PCIe bandwidth, with the entire model loading in about 2ms).

**Overlapping load with computation.** When a request for a new LoRA model is added to a GPU's working set, the GPU issues an asynchronous host-to-device memory copy for that model's weights and **immediately continues executing** the current batch with the models that are already loaded. By the time the current model invocation finishes (a decode step takes approximately 30ms), the asynchronous copy has completed (2ms ≪ 30ms). The new request joins the next batch naturally, with its LoRA weights already resident in GPU memory. This entirely hides the model loading latency — no decode step is ever delayed waiting for weights to load.

**Why not finer-grained loading?** The paper acknowledges that layer-by-layer or matrix-by-matrix loading could theoretically reduce the loading delay further, but argues it is unnecessary because the full-model load time (2ms) is an order of magnitude smaller than a single decode step (30ms). A typical request needs thousands of decode steps, so the one-time 2ms loading cost is amortized to insignificance. The simpler whole-model async copy is preferred for implementation simplicity.

**Memory footprint.** An A100 80GB GPU can hold the backbone model (e.g., ~14GB for Llama 2 7B in FP16) plus KvCache and hundreds of LoRA weight sets. The paper does not give an exact number, but at ~140MB per LoRA model (1% of 14GB), 400 LoRA models would consume ~56GB, which leaves ~10GB for KvCache after accounting for the backbone and overhead. The key point is that GPU memory is abundant relative to LoRA weight storage — the bottleneck is computation, not weight capacity.

---

#### Request Migration

As decode steps proceed, each request's KvCache grows (one new Key and Value vector per layer per token). Eventually, a GPU may run out of memory for KvCache before all its requests complete. Punica handles this by migrating requests between GPUs.

**Eviction policy.** When a GPU's KvCache memory is exhausted, the scheduler evicts the **newest** request — the one that most recently joined the batch. This preserves FCFS semantics because older requests (which have been waiting longer) are not penalized. The evicted request has its execution cancelled on the source GPU: its KvCache is released, and any token generated for it in the most recent batch is discarded (the paper states: "GPU 1 also omits the R3's new token generated in the previous batch").

**Re-addition via recomputation.** The evicted request is immediately re-added to another GPU following the same scheduling rule as a new request. On the destination GPU, the request undergoes a **prefill step on its original prompt plus all previously generated tokens**. This recomputes its entire KvCache from scratch on the new GPU. The paper justifies this choice by citing Kwon et al. (2023)'s finding that "recomputation's latency is equal to or better than moving the KvCache in most cases." KvCache migration would require reading the entire cache from the source GPU's memory, transferring it over the interconnect (NVLink or PCIe), and writing it to the destination GPU's memory — for long sequences, this data volume can exceed the cost of simply recomputing the cache by re-running the prefill.

**Streaming continuity.** After the prefill completes on the destination GPU, the request begins normal decode steps, and new tokens stream back to the user. Figure 5 illustrates the protocol: the scheduler sends a cancellation to GPU 1, GPU 1 releases the KvCache and stops streaming R3's tokens, the scheduler adds R3 to GPU 2, GPU 2 runs a prefill to rebuild the KvCache, and GPU 2 starts streaming R3's tokens. The user sees a brief pause but no data loss — the migration is transparent aside from latency.

**Migration as consolidation.** Migration serves a dual purpose. Beyond handling memory pressure, it is the mechanism by which Punica actively consolidates workloads. When a GPU approaches its memory limit, rather than letting it stall or reject new requests, Punica moves work to other GPUs that have spare memory. This keeps all active GPUs at their maximum effective batch size (constrained by either the batch limit of 32 or available KvCache memory) and allows idle GPUs to reach zero load and be deallocated.

**Cancellation as a primitive.** The paper notes that cancellation (cleanly removing a request from a GPU's working set) is not just for migration — it also handles user disconnections. A disconnected user's request is cancelled, freeing its KvCache and removing it from the batch. This is trivial with Punica's separable KvCache layout (described next) but difficult in systems where the batch dimension is internal to the cache layout.

---

#### KvCache Memory Layout

The paper identifies the KvCache layout as a hidden performance bottleneck that interacts with batching. The standard HuggingFace Transformers layout is:

```[L, 2, B, N, S, D]```

where `$L$` is the number of transformer layers, 2 separates Key and Value projections, `$B$` is the batch size, `$N$` is the number of attention heads, `$S$` is the sequence length (the concatenation dimension that grows with each decode step), and `$D$` is the head dimension. The paper identifies two problems with this layout.

**Problem 1: Inefficient concatenation.** In each decode step, HuggingFace Transformers concatenates a new tensor along the sequence length dimension. This operation reads the entire existing KvCache and writes a new copy — an `$O(S)$` operation where the new data is only `$1/S$` of the total. At `$S = 2048$`, this means reading and writing 2048 times more data than necessary, purely for a layout update.

**Problem 2: Inseparable batching.** The more fundamental problem is that the batching dimension `$B$` is not the outermost dimension. In this layout, the Key and Value tensors for different requests within a batch are interleaved in memory at a fine granularity (per-head, per-layer). This means requests that enter a batch together cannot be cleanly separated later. If one request finishes (generates its end-of-sequence token) while others in the same batch continue, the finished request's KvCache cannot be freed without fragmenting the contiguous tensor or triggering a costly compaction. As a result, finished requests remain in the batch, running **wasted decode steps** until the longest request in the batch finishes. Figure 6 illustrates this waste: a short request that naturally needs 2 decode steps might run 6 if batched with a long request.

**Punica's layout.** Punica adopts a paged, outer-batched layout:

```[Σ_i ⌈S_i/P⌉, L, 2, N, P, D]```

where `$S_i$` is the sequence length of request `$i$`, `$P$` is the page size (a fixed block of sequence positions), and `$\lceil S_i/P \rceil$` is the number of pages allocated to request `$i$`. The first dimension enumerates all pages across all requests, with pages belonging to different requests interleaved arbitrarily. Each request's pages are linked logically (via a page table) but not contiguous in memory.

**What this enables.** The outer dimension being page-based rather than batch-based means that individual requests can be added to or removed from the batch without touching other requests' KvCache. When a request finishes, its pages are simply marked free in the page table — no data movement, no compaction, no effect on other requests. When a new request enters the batch, new pages are allocated from the free pool. This is the **continuous batching** capability that vLLM pioneered (Kwon et al., 2023), which Punica adopts. The paper explicitly credits vLLM for the paged KvCache design and notes that Punica uses FlashInfer (Ye, 2023) for the attention kernel implementation, which supports both paged KvCache and batch decoding without padding.

**Page size and fragmentation.** The paper does not specify the exact page size used, but the design follows vLLM's approach of treating the last dimension as a page of `$P$` sequence positions. Memory fragmentation is minimized because all allocations are multiples of the page size, and partially filled pages at the end of sequences are the only source of waste. For the serving workloads considered (ShareGPT prompt/response distributions), the paper reports no issues with fragmentation.

---

#### Summary of Design Choices and Their Justifications

- **Two-launch SGMV over single-kernel LoRA add-on:** the factorization into shrink and expand enables different GPU scheduling strategies tailored to the output dimension size, while also naturally supporting the same kernel code for both operations (the paper notes that `$\vec{v} \mathrel{+}= \vec{x}A$` and `$\vec{y} \mathrel{+}= \vec{v}B$` are "two launches of the same kernel").

- **Split-K for shrink, output-splitting for expand:** the shrink operation has a tiny output dimension that cannot be parallelized, so input-splitting with reduction provides the parallelism. The expand operation has a large output dimension, so straightforward output splitting works. Both use Tensor Cores when per-model batch size provides enough arithmetic intensity.

- **Separate degenerate schedule for distinct LoRA workloads:** when every request has a unique LoRA model, the computation is I/O-bound matrix-vector multiplication. Using Tensor Cores in this regime would add tile-loading overhead without improving throughput, so a memory-bandwidth-optimized CUDA core schedule is used instead.

- **Greedy busiest-GPU routing:** this creates a positive feedback loop that naturally consolidates load onto a minimal subset of GPUs without requiring a separate consolidation phase or centralized load balancer. The tiebreaker (highest UUID) is deterministic to avoid oscillation.

- **On-demand async weight loading over pre-loading all models:** GPU memory is abundant relative to LoRA weight size, and the 2ms loading time is hidden by the 30ms decode step duration. Pre-loading would require predicting which models will be needed, adding complexity with no benefit.

- **Recomputation over KvCache migration:** moving the KvCache over the interconnect requires reading and writing the full cache, which for long sequences can exceed the cost of recomputing it via a prefill step. PagedAttention's findings support this tradeoff.

- **Paged KvCache with outer batching dimension:** this is adopted directly from vLLM and is essential for continuous batching — without it, requests with different output lengths would waste compute on finished sequences, undermining the throughput gains from SGMV batching.

## 4. Key Insights and Innovations

### Innovation 1: The Multi-Tenant LoRA Serving Problem Is a Batching Problem, Not a Capacity Problem

The dominant assumption in LLM serving before Punica was that serving multiple fine-tuned model variants requires proportional GPU resources — $n$ models means $n$ serving instances, each with its own backbone copy. This is the assumption embedded in every prior system from Clipper (Crankshaw et al., 2017) to vLLM (Kwon et al., 2023): batching works only for identical models, so diversity forces fragmentation. Punica's fundamental reframing is that multi-tenant LoRA serving is not fundamentally a capacity-provisioning problem where you allocate GPUs to models, but a **batching problem** where the challenge is making different models' computations composable within a single GPU invocation.

This is a diagnostic insight, not an optimization. The paper doesn't say "we made model loading faster" or "we compressed weights more aggressively" — approaches that would have treated the symptom (many models consume many resources) rather than the cause (existing systems cannot batch heterogeneous work). Instead, it identifies that the bottleneck is at the CUDA kernel level: the per-request weight addition `$\vec{x}AB$` is the one operation in the transformer forward pass that couples a request to a specific model, and prior systems handle it by serializing across models. The insight is that if you can make this one operation fast under model heterogeneity, the rest of the serving stack — the backbone GEMM, the attention computation, the KvCache management — naturally handles batching exactly as it does for a single model.

The evidence that this is a reframing rather than an engineering improvement is in Figure 11: Punica's throughput is essentially flat across the Distinct, Uniform, Skewed, and Identical workloads (e.g., 1044 tok/s on 7B regardless of model count), while baseline systems show a ~12× collapse from Identical to Distinct. The baselines aren't failing because they lack capacity — a single A100 can easily hold the 7B backbone plus hundreds of LoRA weight sets. They fail because their architecture cannot batch heterogeneous requests, forcing batch-size-1 execution when models differ. Punica's contribution is identifying that the batching barrier, not memory capacity or raw FLOPs, is the binding constraint.

This also explains why PetS (Zhou et al., 2022) is insufficient despite sharing the backbone in memory. PetS recognized the memory-sharing opportunity but didn't solve the concurrent execution problem — it still processes different adapters sequentially. Punica's diagnosis is that memory sharing without batching is only half the solution. The paper's design guidelines (G1–G3) encode this diagnosis explicitly: G2 demands cross-model batching as a hard requirement, not an optimization target.

**Significance beyond throughput:** this reframing transforms the economic calculus for LoRA deployment. If throughput is linear in model count (baseline systems), serving 100 tenants costs 100× what serving 1 costs. If throughput is constant in model count (Punica), serving 100 tenants costs roughly the same as serving 1 — the constraint shifts to aggregate demand, not model diversity. This makes LoRA-as-a-Service economically viable in a way that prior systems could not support, regardless of how optimized their single-model serving was.

---

### Innovation 2: Segmented Gather as a GPU Scheduling Abstraction for Heterogeneous Batched Computation

Before Punica, the standard approach for batched computation with per-sample weight variations was **Gather-BMM**: gather the weights each sample needs into a stacked tensor, then call batched matrix multiply (`torch.bmm()`). This is the approach a PyTorch user would naturally write, and it's what the paper benchmarks as the best non-specialized baseline (Section 7.1, Figure 8). Punica's insight is that this Gather-BMM pattern is architecturally wrong for the multi-tenant LoRA setting because it **doubles the weight data movement**: the Gather step reads each weight matrix from its storage location and writes it into the stacked tensor, then the BMM step reads the stacked tensor back. In the Distinct workload where every request needs a different weight matrix, the weight I/O is the bottleneck — and Gather-BMM makes it twice as bad as necessary.

The **Segmented Gather** abstraction is the conceptual move that fixes this. Rather than physically rearranging weight matrices to match the input order, SGMV keeps weights in place and instead **partitions the input by segment**: the batch is ordered so that all requests for the same LoRA model are contiguous, and the kernel reads each model's weight matrix once, applying it to all inputs in that model's segment. This eliminates the Gather-copy entirely. The weight matrix `$A_i$` is streamed from memory once and multiplied against `$(s_i - s_{i-1})$` input vectors before the next weight matrix is loaded.

What makes this an innovation rather than an obvious optimization is the **GPU scheduling problem it surfaces**. In the degenerate case where every request targets a different model (Distinct workload), each segment has size 1. The computation degrades to a sequence of matrix-vector products, each of which is I/O-bound with arithmetic intensity too low to benefit from Tensor Cores. A single monolithic kernel handling this case naively would leave most GPU compute units idle. Punica's solution — separate schedules for the expand and shrink operations, with output-splitting for expand and Split-K reduction for shrink, plus a non-Tensor-Core path for the size-1 segment case — is a GPU scheduling design that handles the full spectrum from fully distinct to fully identical workloads within a single kernel abstraction.

The roofline analysis in Figure 7 makes this innovation concrete. In the Distinct case, SGMV's arithmetic intensity is constant (the roofline point moves right only because batch size increases parallelism, not operational intensity). In the Identical case, arithmetic intensity grows linearly with batch size, moving the kernel from the memory-bandwidth-bound region into the compute-bound region. The Uniform and Skewed cases interpolate between these extremes. Prior to this paper, there was no analysis showing that the multi-model batching problem maps onto a spectrum of arithmetic intensity regimes, each requiring different GPU scheduling strategies. The SGMV kernel design is the first to handle this spectrum within a single operator.

**Significance beyond LoRA:** the Segmented Gather pattern generalizes to any setting where batched computation involves per-sample weight variations drawn from a shared pool — mixture-of-experts inference, multi-adapter serving for other PEFT methods, or multi-task models with task-specific heads. The paper doesn't explore these generalizations, but the abstraction it provides (segment inputs by shared weight, apply each weight once to its segment) is not LoRA-specific.

---

### Innovation 3: Greedy Busiest-GPU Routing as an Implicit Workload Consolidation Mechanism

The standard approach to cluster scheduling for ML serving is some form of load balancing — distribute requests evenly across GPUs to prevent hotspots, typically using least-loaded or round-robin policies (as in Clipper, Crankshaw et al., 2017; or Symphony, Chen et al., 2023). Punica does the opposite: it routes every new request to the **busiest** GPU that can accommodate it, deliberately creating hotspots. The paper argues that this greedy policy causes a positive feedback loop where busy GPUs stay busy and idle GPUs stay idle, naturally consolidating the workload onto a minimal subset of the cluster.

This is a subtle but important insight about the dynamics of stateful serving. In stateless serving (e.g., CNN inference for image classification), load balancing works because any GPU can handle any request equally well, and spreading load maximizes utilization across the fleet. In stateful LLM serving, each request carries KvCache state that ties it to a specific GPU for the duration of its execution. The cost of moving that state (via migration) is non-trivial, so the system wants to minimize migrations while still achieving consolidation. Punica's busiest-GPU policy achieves consolidation **without explicit rebalancing**: because new requests preferentially join already-busy GPUs, lightly-loaded GPUs naturally drain as their existing requests complete without being replenished. The scheduler doesn't need to periodically scan for underutilized GPUs and trigger migrations — the routing policy handles consolidation as an emergent property of greedy assignment.

The paper's cluster deployment evaluation (Figure 13) demonstrates this empirically: GPUs either run at their maximum batch size (32) or sit idle, with very few GPUs in intermediate states. The "occasional" drops to smaller batch sizes occur only when a GPU exhausts its KvCache memory and migrates requests out, not because of load variation. This bimodal GPU utilization pattern — fully busy or fully idle — is exactly what you want for cloud auto-scaling: idle GPUs can be returned to the provider without hesitation, and new GPUs are only provisioned when all existing ones are saturated.

**Comparison to prior work:** Symphony (Chen et al., 2023) uses a non-work-conserving scheduler for model serving, but Punica deliberately runs batches back-to-back on each GPU (work-conserving) because the KvCache affinity makes preemption expensive. The busiest-GPU policy is work-conserving at the GPU level but non-work-conserving at the cluster level — it leaves GPUs idle even when there is queued work, because routing work to idle GPUs would fragment the batch and reduce overall efficiency. This is a deliberate tradeoff: sacrifice instantaneous utilization of all resources to maximize throughput per active GPU, then return the idle ones.

**Significance:** this is not a novel scheduling algorithm in the theoretical sense — it's a simple greedy heuristic. The innovation is in recognizing that for stateful, batch-sensitive LLM serving, the conventional wisdom of load balancing is counterproductive, and that aggressive consolidation through biased routing produces better system-level outcomes. The paper doesn't prove optimality, but the empirical result (throughput remains high while GPU count scales down with load) validates the design intuition.

---

### Innovation 4: The Two-Launch Factorization as a Mechanism for Handling the Asymmetric Dimensionality of LoRA Operations

The decomposition `$\vec{x}AB \rightarrow (\vec{v} = \vec{x}A, \vec{y} \mathrel{+}= \vec{v}B)$` might appear to be a trivial algebraic identity — and indeed, mathematically it is. The innovation is not the factorization itself but the recognition that this factorization exposes an **asymmetric computational structure** that requires different GPU scheduling strategies for the two halves, and that handling this asymmetry correctly is what makes multi-model batching efficient.

Specifically, the shrink operation `$\vec{v} = \vec{x}A$` maps from `$h_1$` (~4096) to `$r$` (~16). The output dimension is too thin to parallelize directly — there are only 16 elements to compute per request. Without the factorization, a kernel computing `$\vec{x}(AB)$` directly would face a choice: either treat `$AB$` as a single `$h_1 \times h_2$` matrix (losing the low-rank structure and making the weight matrix huge — 4096 × 4096 = 16M elements instead of 4096 × 16 + 16 × 4096 = 131K elements) or compute `$\vec{x}A$` and `$\vec{v}B$` sequentially anyway, which is the factorization. The factorization is forced by the low-rank structure.

But the paper's insight goes further: because the factorization is forced, and because the two resulting operations have opposite computational shapes (shrink has large input, tiny output; expand has tiny input, large output), the **same kernel schedule cannot be optimal for both**. The Split-K strategy for shrink and the output-splitting strategy for expand are not interchangeable — applying output-splitting to shrink would create threadblocks each computing a few elements of a 16-dimensional vector, with terrible utilization. Applying Split-K to expand would require a reduction over the large output dimension, which is expensive and unnecessary when the output can simply be tiled.

Prior work on batched matrix multiplication (e.g., the cuBLAS batched GEMM API, Gather-BMM patterns in PyTorch) treats all dimensions symmetrically — the kernel doesn't care whether the contraction dimension is large or small. Punica's contribution is showing that for the specific dimensional regime of LoRA (`$h \gg r$`), the asymmetry matters enormously and a specialized two-kernel design with different schedules outperforms a generic batched GEMM by avoiding the Gather step and matching the parallelism strategy to each operation's shape.

**Evidence:** Figure 8 shows that SGMV achieves 37–116µs latency across all workloads for a single LoRA operator, compared to 30–290µs for Gather-BMM and 30–270µs for Loop (worse at high batch sizes). The gap is widest in the Distinct case where Gather-BMM's extra I/O and Loop's serialization are most penalized. Figure 9 shows that this advantage holds across LoRA ranks from 8 to 64, with SGMV latency nearly flat across batch sizes when weight sharing exists (Uniform, Skewed, Identical).

**Significance beyond performance:** this insight about asymmetric dimensionality is LoRA-specific but applies to any low-rank adapter method. It suggests a design principle: when a computation decomposes into operations with very different input/output dimension ratios, different GPU schedules should be used for each. This is not a general principle for all matrix multiplications — it's specific to the `$h \gg r$` regime — but within that regime, it's a guideline that future adapter-serving systems should follow.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper does not use a static evaluation dataset in the traditional ML sense. Instead, it uses synthetic workloads derived from **ShareGPT** (ShareGPT, 2023) — a collection of real user-bot conversations from Internet users — to model the distribution of prompt lengths and response lengths in LLM serving traffic. The key workload characteristics are prompt length and response length distributions, which determine prefill cost and total decode steps respectively. For multi-model diversity, the paper defines four request distributions across LoRA models: **(1) Distinct** (each request targets a different LoRA model), **(2) Uniform** (given `n` requests, `⌈√n⌉` models are used, each equally popular), **(3) Skewed** (model popularity follows a Zipf-1.5 distribution, where the i-th most popular model receives `α = 1.5` times the requests of the (i+1)-th), and **(4) Identical** (all requests target the same LoRA model). These four distributions span the spectrum from maximum diversity (Distinct) to zero diversity (Identical), allowing systematic measurement of how throughput degrades as model heterogeneity increases.

- **Base model(s).** All experiments use **Llama 2** (Touvron et al., 2023) at three scales: **7B, 13B, and 70B parameters**. The paper states Llama 2 represents a widely-used open model family, making results broadly applicable. The LoRA rank is fixed at **16** across all experiments, and LoRA is applied to all dense projections in the transformer (Query, Key, Value, Output in attention, plus the three MLP projections). LoRA model weights are randomly generated since "the weight does not affect latency performance" — this is a standard simplifying assumption in systems benchmarking, where the computational cost depends only on tensor shapes and data types, not the specific weight values. The models run in FP16 precision.

- **Metrics.** The primary metric is **throughput**, measured in **tokens per second (tok/s)** — the total number of output tokens generated across all requests divided by the total wall-clock time to serve them. This is the standard metric for serving systems because it directly reflects cost-per-query: higher throughput means more requests served per GPU-hour. Secondary metrics include **per-token latency** (the time to generate a single decode token) and **batch size** over time (to visualize GPU utilization and consolidation behavior). For the microbenchmarks, **kernel latency** in microseconds (µs) is measured for individual operations (the LoRA add-on, a single transformer layer).

- **Baselines.** Since "there is no well-known multi-LoRA serving system," the paper compares against **four state-of-the-art LLM serving systems**, each representing a different approach to inference optimization, with varying degrees of capability for LoRA:
  - **HuggingFace Transformers** (Wolf et al., 2020) with the **PEFT library** (Mangrulkar et al., 2022) for LoRA weight integration — this is the reference implementation that most practitioners use, but it lacks optimized CUDA kernels (no FlashAttention, no continuous batching, problematic KvCache layout as described in Section 5.4).
  - **DeepSpeed** (Aminabadi et al., 2022) with PEFT — a production-grade inference engine from Microsoft that includes various optimizations but inherits HuggingFace's KvCache layout limitations.
  - **FasterTransformer** (Hsueh, 2021) — NVIDIA's highly optimized inference framework. The paper runs it **backbone-only** (without LoRA) because FasterTransformer does not support LoRA models. This is a deliberately favorable baseline: it represents the best-case throughput a system could achieve if multi-model batching were solved, since it runs only the backbone computation without any LoRA overhead.
  - **vLLM** (Kwon et al., 2023) — the state-of-the-art LLM serving system with PagedAttention for continuous batching and efficient KvCache management. Also run **backbone-only** because vLLM does not support batching across different LoRA models. Like FasterTransformer, this represents an upper bound on what a single-model serving system can achieve.

  The paper notes that it **"omits the model switching costs for baseline systems"** — meaning when baseline systems need to switch between serving different LoRA models (loading new weights, reloading or recreating KvCache), those costs are not included in their latency measurements. This makes the baselines **more favorable** than a realistic deployment, strengthening any advantage Punica demonstrates.

- **Generation budget / compute accounting.** The paper does not use "generation budget" in the sense of fixed compute per problem. Instead, it measures throughput over a fixed workload: **1000 requests** generating approximately **101k total tokens** for the single-GPU experiments (7B and 13B). For the 70B experiments with tensor parallelism, the workload size is not explicitly quantified but follows the same methodology. All systems are restricted to batch in a first-come-first-serve (FCFS) manner with a **maximum batch size of 32** (based on A100 profiling from Figure 1, where batch size 32 achieves good throughput while keeping per-token latency within acceptable bounds). The comparison is fair because Punica and all baselines operate under the same workload, same maximum batch size constraint, and same FCFS scheduling rule. The key difference is that Punica can fill its batch to 32 with requests from different LoRA models, while baselines can only batch requests for the same model.

- **Cross-validation / statistical protocol.** The paper does not apply cross-validation or statistical significance testing. This is standard for systems benchmarking: the metrics (throughput, latency) are deterministic given the workload and hardware configuration, modulo minor OS and hardware noise. The paper runs one-hour cluster deployment experiments (Section 7.3) with time-varying load to assess dynamic behavior, but does not report error bars or multiple trials for the throughput measurements. The microbenchmarks (kernel latency) appear to be single measurements per data point, which is common practice for CUDA kernel benchmarking where variance across runs on a quiescent GPU is typically negligible.

**Testbed configuration.** Two hardware setups are used:
- **Testbed #1:** A single server with one NVIDIA A100 80GB GPU. The large memory capacity (80GB vs. 40GB) allows studying LoRA batching effects without KvCache memory pressure dominating results. Used for all microbenchmarks and the 7B/13B single-GPU text generation experiments.
- **Testbed #2:** Two NVIDIA HGX A100 40GB servers with 8 GPUs each (16 GPUs total), equipped with NVSwitch for high-bandwidth GPU-to-GPU communication. Used for the 70B tensor parallelism experiments and the cluster deployment evaluation.

---

### Main Quantitative Results

The paper evaluates Punica along four axes: (1) microbenchmarks of the SGMV kernel and transformer layer performance, (2) single-GPU text generation throughput for 7B and 13B models, (3) multi-GPU text generation for the 70B model with tensor parallelism, and (4) cluster deployment with dynamic load. Each axis isolates a different aspect of the system's behavior.

#### Microbenchmarks: SGMV Kernel and Transformer Layer Performance

**SGMV roofline analysis (Figure 7).** The roofline model characterizes SGMV's performance across the four workload distributions on Testbed #1, using the expand kernel with `hi = 16` (LoRA rank), `ho = 4096` (output dimension), and batch sizes from 1 to 64.

- **Distinct workload:** The arithmetic intensity is constant regardless of batch size because both FLOP and I/O grow at the same rate (weight I/O dominates). The achieved FLOP/s increases with batch size — from approximately 10 GFLOP/s at batch size 1 to approximately 300 GFLOP/s at batch size 64 — which means the kernel is exploiting increasing parallelism (more threadblocks) even though the operational intensity per byte doesn't improve. This is the I/O-bound regime, well below the GPU's peak of 312 TFLOP/s.

- **Identical workload:** The arithmetic intensity grows linearly with batch size (the weight is read once and reused across all inputs). At batch size 64, the achieved throughput reaches approximately 100 TFLOP/s, moving into the compute-bound regime and approaching the memory bandwidth ceiling of 1.935 TB/s on the roofline slope. The paper states SGMV is "bounded by memory bandwidth" in this case.

- **Uniform and Skewed workloads:** These interpolate between the two extremes, benefiting from both increased parallelism (like Distinct) and increased arithmetic intensity (like Identical) to intermediate degrees.

The key takeaway: SGMV's performance depends on the degree of weight sharing within a batch. When models are shared (Uniform, Skewed, Identical), performance improves substantially with batch size. In the worst case (Distinct), performance still improves with batch size because the kernel can exploit more parallelism, but it never reaches the compute-bound regime. This analysis justifies the scheduling design: consolidating requests of the same LoRA model onto the same GPU (which the greedy busiest-GPU policy naturally does) increases weight sharing within batches and thus improves SGMV efficiency.

**LoRA operator microbenchmark (Figure 8).** Compares three implementations of the batched LoRA add-on on Testbed #1 across batch sizes 1 to 64, measuring latency in microseconds:

- **Loop:** A Python for-loop over each LoRA model, running each model's requests separately. In the Distinct case, latency grows linearly from approximately 30µs at batch size 1 to approximately 270µs at batch size 64 — each additional request adds roughly the cost of one matrix-vector product. In the Identical case, loop is equivalent to BMM (all requests use the same weights).

- **Gather-BMM:** Two Gather steps (stacking weights) and two `torch.bmm()` calls. The Gather latency increases with batch size in the Distinct case (from near zero to approximately 180µs at batch size 64) because more distinct weight matrices must be read and stacked. Total Gather-BMM latency ranges from approximately 30µs to 290µs.

- **SGMV:** Two SGMV kernel launches. In the Distinct case, latency increases gradually from 37µs to 116µs — a 3.1× increase over a 64× increase in batch size, demonstrating that the kernel efficiently parallelizes across requests even without weight sharing. In the Uniform and Skewed cases, latency increases only marginally (37µs to 46µs) because weight sharing within groups increases operational intensity. In the Identical case, SGMV latency is nearly flat (37µs to 40µs) — the paper notes this means "SGMV implements BMM more efficiently than `torch.bmm()` in the case of LoRA," which is surprising and suggests the specialized schedule outperforms cuBLAS's generic batched GEMM for these specific matrix shapes.

The SGMV advantage is largest in the Distinct case at high batch sizes: 116µs vs. 290µs for Gather-BMM and 270µs for Loop — a 2.5× and 2.3× improvement, respectively. This is the regime that matters most for multi-tenant serving, where model diversity is high.

**LoRA rank sensitivity (Figure 9).** Evaluates SGMV latency for LoRA ranks 8, 16, 32, and 64 across the four workloads. At batch size 1 (single request), latency is approximately 42µs for all ranks — the cost is dominated by kernel launch overhead and reading the weight matrix, which scales with rank but is small in absolute terms. In the Distinct case at batch size 64, latency increases from 72µs (rank 8) to 118µs (rank 64) — roughly proportional to the weight matrix size, confirming the I/O-bound nature of the Distinct workload. When weight sharing exists (Uniform, Skewed, Identical), latency remains almost flat across batch sizes at approximately 42–45µs for all ranks — the operational intensity is high enough that the GPU's compute throughput absorbs the increased FLOPs from higher ranks without additional latency. This is significant: it means increasing LoRA rank (which improves model quality) has minimal cost in the shared-weight regimes that Punica's scheduling aims to create.

**Transformer layer benchmark (Figure 10).** Measures full transformer layer latency (including self-attention, MLP, and LoRA add-on) for 7B and 13B model configurations on Testbed #1, at sequence lengths 512 and 2048, with batch sizes 1 to 32.

- **Shorter sequences (len=512):** The batching effect is strongest. For the 7B model, layer latency increases from approximately 0.5ms at batch size 1 to approximately 0.85ms at batch size 32 — only a 70% increase for a 32× increase in work. This is because self-attention cost scales with sequence length (which is moderate at 512), and the dense projections dominate, benefiting from batching.

- **Longer sequences (len=2048):** The batching effect is weaker. For the 7B model, layer latency increases from approximately 1.1ms to 1.7ms over the same batch size range. Self-attention, which scales quadratically with sequence length, takes a larger fraction of total time and does not batch as effectively (each request's attention computation is independent).

- **Cross-workload comparison:** Critically, the paper observes that "the layer latency is roughly the same across different workloads" — the four lines for Distinct, Uniform, Skewed, and Identical largely overlap in all four subfigures. The LoRA add-on's contribution to total layer latency is small compared to the backbone dense projection and self-attention. This "LoRA-model-agnostic performance property" is a key empirical finding: it justifies the scheduling design that treats different LoRA models as interchangeable from a performance perspective. The scheduler doesn't need to model per-model compute costs — all models have essentially identical latency characteristics.

#### Single-GPU Text Generation: 7B and 13B Models

**Figure 11** presents the headline throughput comparison on Testbed #1, serving 1000 requests (approximately 101k tokens) with the 7B and 13B Llama 2 models. The four workload distributions are evaluated for Punica and all four baseline systems.

**7B model results (Figure 11a):**

- **Punica:** Achieves **1044 tok/s** across all four workloads — throughput is essentially constant regardless of LoRA model diversity. The system maintains a batch size of 32 throughout, filling batches with requests from different models as needed.

- **vLLM (backbone-only):** Achieves **1140 tok/s** in the Identical case — slightly higher than Punica because it runs backbone-only without LoRA computation overhead. In the Distinct, Uniform, and Skewed cases, vLLM's throughput collapses because it can only batch requests for the same model. With every request targeting a distinct model, vLLM degrades to batch-size-1 execution. The paper does not give exact numbers for the Distinct case, but the bar height in Figure 11a appears to be approximately 80–100 tok/s — roughly a **12× reduction** from the Identical case.

- **FasterTransformer (backbone-only):** Shows similar behavior to vLLM — high throughput in the Identical case (slightly lower, perhaps because it lacks continuous batching) and collapse in the Distinct/Uniform/Skewed cases.

- **DeepSpeed:** Marginally lower than FasterTransformer in the Identical case and similarly poor in diverse workloads.

- **HuggingFace Transformers:** The lowest throughput in all cases, including the Identical case, due to the lack of FlashAttention and the inefficient KvCache layout.

The paper states that **"Punica achieves 12x higher throughput"** compared to state-of-the-art systems. This 12× figure comes from the ratio of Punica's throughput (~1044 tok/s) to the baseline systems' throughput in the Distinct case (~80–100 tok/s). This is the paper's central performance claim.

**13B model results (Figure 11b):** The pattern is identical but with lower absolute throughput due to the larger model. Punica achieves **693 tok/s** across all workloads. vLLM achieves 789 tok/s in the Identical case (again slightly higher due to backbone-only execution) and collapses in diverse workloads. The 12× improvement ratio holds.

**Why vLLM's Identical-case advantage doesn't undermine the claim:** The paper explicitly acknowledges that vLLM's throughput is slightly higher than Punica in the Identical case (1140 vs. 1044 tok/s for 7B, 789 vs. 693 tok/s for 13B). This is expected because vLLM runs backbone-only — it doesn't pay the LoRA add-on cost that Punica always pays. The paper frames this as an acceptable tradeoff: a ~9% throughput reduction in the best case (single model) in exchange for maintaining that throughput across all multi-model workloads where baselines experience a ~12× collapse. In a multi-tenant setting with hundreds of models, the Identical case is irrelevant — the system will almost always operate in the Uniform, Skewed, or Distinct regimes.

**The "12×" figure requires careful interpretation.** The improvement factor depends on the workload distribution. In the Identical case, Punica is slightly worse than vLLM (~0.9×). In the Distinct case, the improvement is approximately 12×. In the Uniform and Skewed cases (which are more realistic for production multi-tenant serving), the improvement is intermediate — perhaps 5–8× based on the bar heights in Figure 11. The paper's "12×" headline number is the maximum advantage, achieved in the worst case for baseline systems. A more nuanced summary would be: Punica provides constant throughput regardless of model diversity, while baseline throughput is inversely proportional to model diversity, creating a 1–12× advantage depending on workload characteristics.

#### Multi-GPU Text Generation: 70B Model with Tensor Parallelism

**Figure 12** evaluates the 70B model on 8 GPUs in Testbed #2, using Megatron-style tensor parallelism (Shoeybi et al., 2019; Narayanan et al., 2021) for both Punica and vLLM. Only Punica and vLLM are compared — the other baselines either don't support tensor parallelism or were already shown to be strictly worse than vLLM. The workload size is not explicitly stated but presumably follows the same methodology as the 7B/13B experiments.

- **vLLM (backbone-only):** Achieves **457 tok/s** in the Identical case. In the presence of multiple LoRA models (Distinct, Uniform, Skewed), throughput drops to approximately **21–25 tok/s** — a roughly 20× reduction. The paper notes this is because vLLM runs with small batch sizes when requests target different models.

- **Punica:** Achieves **441–446 tok/s** across all four workloads — essentially flat, matching vLLM's Identical-case throughput within a few percent.

The paper states that for the Identical case, "Punica and vLLM achieve the same performance because their parallel schemes are the same." This is slightly misleading — vLLM achieves 457 tok/s vs. Punica's ~443 tok/s, a 3% difference. The gap is smaller than in the single-GPU case (8% for 7B, 12% for 13B) because tensor parallelism distributes the backbone computation across GPUs, reducing the relative cost of the LoRA add-on.

The significance of this result is that SGMV composes cleanly with tensor parallelism. The LoRA add-on is computed on the same GPU that holds the corresponding backbone weight shard, using the same SGMV kernel without modification. There is no cross-GPU communication needed for the LoRA computation — each GPU handles its portion of `A_i` and `B_i` independently. This means Punica's multi-model batching advantage scales to large models that require model parallelism, which is essential for practical deployment of 70B+ models.

#### Cluster Deployment with Dynamic Load

**Figure 13** evaluates Punica on 16 GPUs in Testbed #2 over a one-hour experiment with time-varying load. The workload uses the 7B model with LoRA popularity following a Zipf-1.5 distribution (the Skewed workload). The request rate follows a macro-level pattern of gradually increasing then decreasing, with micro-level inter-arrival gaps drawn from an exponential distribution (making arrivals a Poisson process). This simulates a realistic diurnal load pattern with natural variation.

The figure shows three panels:

- **Top panel (request rate):** Varies from 0 to approximately 10 requests per second over the hour, peaking around 1500–2000 seconds before declining.

- **Middle panel (throughput):** Tracks the request rate closely, reaching approximately 10,000 tok/s at peak and declining to near zero at the start and end of the trace. The system's throughput scales with offered load, indicating it is not bottlenecked by anything other than demand.

- **Bottom panel (GPU batch size over time):** This is the key visualization of Punica's consolidation behavior. Each of the 16 GPUs is shown as a horizontal stripe, with color intensity representing batch size (0–32, where 32 is the maximum). The pattern shows that:
  - GPUs are either at **maximum batch size (32)** or **idle (0)**, with very few intermediate states. This is the consolidation effect: the scheduler routes requests to already-busy GPUs, keeping them saturated.
  - The number of active GPUs scales with load. At the peak of the trace, approximately 12–14 GPUs are active. At the start and end, only 1–2 GPUs are active.
  - "Occasionally, a GPU runs with a smaller batch size because it runs out of KvCache space and migrates out a few requests to other GPUs." These transient drops are visible as lighter-colored stripes within otherwise-dark GPU rows.
  - "When a GPU becomes idle (batch size = 0), it is likely that it stays idle, which can then be released to the cloud provider if necessary." This validates the consolidation dynamics: idle GPUs don't spontaneously receive new requests because the scheduler routes to busy GPUs first.

The paper does not provide a quantitative metric for this experiment (no average throughput, no GPU utilization percentage, no latency distribution). The evaluation is qualitative — it demonstrates that the system behaves as designed under dynamic load, with consolidation emerging from the scheduling policy. This is a common pattern in systems papers: the microbenchmarks and single-GPU experiments provide the quantitative evidence, and the cluster deployment trace demonstrates that the system works end-to-end without pathological behavior.

---

### Ablation Studies and Robustness Checks

The paper's ablation methodology is embedded in its microbenchmark comparisons rather than presented as a separate section. The key ablations are:

**SGMV vs. Gather-BMM vs. Loop implementations (Figures 8, 9).** This is the central kernel-level ablation. The Loop baseline ablates batching entirely — it processes each LoRA model sequentially, demonstrating the catastrophic performance of the naive approach. The Gather-BMM ablation evaluates whether a generic batched GEMM approach (without segmentation) can match SGMV. The result (Figures 8 and 9) shows that Gather-BMM is 2–2.5× slower than SGMV in the Distinct case at high batch sizes due to the extra Gather memory traffic. This ablation justifies the Segmented Gather design: simply grouping inputs and calling `torch.bmm()` is not enough; the memory access pattern matters.

**LoRA rank scaling (Figure 9).** Tests SGMV across ranks 8, 16, 32, and 64. In the Distinct case, latency scales approximately with rank (72µs at rank 8, 118µs at rank 64 for batch size 64) because the I/O volume is proportional to the weight matrix size. In shared-weight regimes, latency is nearly flat across ranks — this is a non-obvious finding that suggests the GPU's compute throughput absorbs the additional FLOPs from larger ranks when operational intensity is already high. This means higher-quality LoRA models (larger rank) come at negligible cost in Punica's target regime of consolidated multi-tenant serving.

**Transformer layer workload sensitivity (Figure 10).** The near-identical latency across Distinct, Uniform, Skewed, and Identical workloads is a de facto ablation of the LoRA computation's contribution to end-to-end serving cost. The result shows that SGMV is so efficient relative to self-attention and backbone dense projections that the workload distribution doesn't affect layer latency. This justifies treating different LoRA models as interchangeable from a scheduling perspective.

**Sequence length impact (Figure 10, subfigures a–d).** By comparing len=512 vs. len=2048, the paper ablates the effect of sequence length on the batching benefit. At len=512, batch size 32 costs only 70% more latency than batch size 1. At len=2048, batch size 32 costs 55% more. The batching benefit is weaker at longer sequences because self-attention (which is per-request and doesn't batch as effectively) consumes a larger fraction of total time. This informs the system's practical limits: Punica's throughput advantage is largest for workloads with shorter output sequences (typical of chatbots), and diminishes (but doesn't disappear) for long-form generation.

**Backbone-only vs. LoRA-augmented serving (Figures 11, 12).** The comparison between vLLM (backbone-only) and Punica in the Identical case quantifies the overhead of the LoRA add-on. For 7B, the overhead is ~9% (1140 vs. 1044 tok/s). For 13B, ~12% (789 vs. 693 tok/s). For 70B with tensor parallelism, ~3% (457 vs. 443 tok/s). The overhead decreases with model size because the backbone computation scales with `h₁ × h₂` (quadratically in hidden dimension) while the LoRA add-on scales with `h₁ × r + r × h₂` (linearly in hidden dimension). This means Punica's design is particularly well-suited for large models, where the relative cost of supporting multi-tenancy approaches zero.

**On-demand weight loading latency (Section 5.2, not separately evaluated).** The paper states that loading a full LoRA model takes 2ms over PCIe Gen4 x16, and that this is hidden by the 30ms decode step duration. No experiment directly measures or validates this claim — there is no microbenchmark showing load latency, no measurement of batch execution time with and without concurrent weight loading, and no evaluation of what happens when many new models are loaded simultaneously (which could saturate PCIe bandwidth). This is a significant unvalidated claim. The on-demand loading mechanism is central to Punica's cold-start story (new LoRA models can be served immediately without pre-warming), but the paper provides only a back-of-the-envelope latency estimate, not an empirical measurement.

**Request migration latency (no separate evaluation).** The paper describes the migration protocol (cancel on source GPU, prefill on destination GPU) and cites vLLM's finding that recomputation is equal to or better than KvCache migration. However, Punica's specific migration performance is not benchmarked. There is no experiment showing the latency impact of migration on the migrated request (how long does the user wait for the prefill to complete on the destination GPU?), the impact on other requests in the source and destination batches (does the departure or arrival of a request cause a batch stall?), or the frequency of migrations under different workload characteristics. The cluster deployment trace (Figure 13) shows occasional batch size drops attributed to migration, but no quantitative analysis.

**Baseline model switching costs omitted.** The paper explicitly states that "model switching costs for baseline systems" are omitted — meaning when HuggingFace, DeepSpeed, FasterTransformer, or vLLM need to switch from serving one LoRA model to another (loading new weights, potentially flushing and rebuilding KvCache), these costs are not included in their throughput measurements. This makes the baseline throughput numbers **overestimates** of what those systems would achieve in a realistic multi-tenant setting. The paper does not quantify how large these switching costs are, which makes the 12× improvement figure conservative (the true improvement would be larger if switching costs were included) but also makes it difficult to assess how much of Punica's advantage comes from the SGMV kernel versus from the on-demand loading and migration mechanisms.

---

### Critical Assessment

The paper makes three central claims: (1) that Punica achieves 12× higher throughput than state-of-the-art LLM serving systems for multi-tenant LoRA serving, (2) that this advantage holds regardless of LoRA model popularity distribution, and (3) that the system adds only 2ms latency per token. The experiments provide strong evidence for claims (1) and (2) within the tested scope, but the evidence for claim (3) is indirect, and several important dimensions of the system's behavior are uncharacterized.

**Claim 1: "12× higher throughput."** This claim is demonstrated convincingly for the specific setting of 7B and 13B Llama 2 models on a single A100 GPU, evaluated against four baseline systems, under the ShareGPT-derived workload. The 12× figure represents the maximum advantage (Distinct workload, Punica vs. baseline systems), and the paper presents the full spectrum from 0.9× (Identical case vs. vLLM) to 12× (Distinct case). This is a fair and transparent presentation — the headline number is the best case, but all cases are shown.

However, several caveats apply:

- **The 12× figure is relative to backbone-only baselines that cannot serve LoRA at all in the Distinct case.** vLLM and FasterTransformer are evaluated backbone-only, meaning they provide no LoRA functionality in exchange for their low throughput in diverse workloads. A fairer baseline would be vLLM or FasterTransformer with LoRA weights merged into the backbone (creating independent full models), but this would require loading separate model copies, which the paper argues is exactly the inefficiency Punica solves. The paper's framing is that the 12× figure represents the throughput penalty of serving LoRA models as independent full models using current systems.

- **The workload is synthetic and relatively small (1000 requests, ~101k tokens).** A typical production deployment might serve millions of requests per day. The paper does not evaluate whether throughput remains stable over much longer runs or whether memory fragmentation (in the KvCache page table or the GPU memory allocator) causes degradation over time.

- **Only ShareGPT prompt/response distributions are tested.** Different serving workloads (e.g., document summarization with very long prompts, code generation with very long responses, few-shot prompts with many examples) could have different throughput characteristics that the paper does not explore.

- **The absolute throughput numbers (1044 tok/s for 7B, 693 tok/s for 13B) are not compared against theoretical peak or other published results for Llama 2 serving on A100s.** Without this context, it's difficult to assess whether Punica's single-model throughput is itself fully optimized or whether there is room for improvement that would also benefit the baselines.

**Claim 2: "Advantage holds regardless of LoRA model popularity distribution."** This is the most robustly supported claim. Figures 11 and 12 show throughput that is essentially flat across the four distributions. The transformer layer benchmark (Figure 10) shows that per-layer latency is workload-invariant. The microbenchmarks (Figures 8, 9) show that SGMV latency is within 3× from the best case (Identical) to the worst case (Distinct), and this difference is small relative to total layer latency. The evidence is consistent across model sizes (7B, 13B, 70B).

One qualification: the "regardless of distribution" claim applies to the throughput per active GPU, but the number of GPUs needed to handle a given workload depends on aggregate demand. The paper shows that Punica consolidates workloads onto fewer GPUs (Figure 13), but doesn't provide a formula or model for how many GPUs are needed as a function of request rate and model count. The claim is about throughput per GPU, not total resource efficiency, which also depends on the consolidation factor.

**Claim 3: "Only 2ms latency per token."** This claim is the least well-supported by direct evidence. The 2ms figure appears in the abstract and conclusion but is never directly measured or attributed to a specific experiment. The closest evidence is:

- Figure 1 shows that increasing batch size from 1 to 32 adds 2ms to decode latency for short sequences (len=128: 11ms to 13ms). This is a measurement of the batching overhead, not a Punica-specific measurement.

- The transformer layer benchmark (Figure 10) shows that at len=512 and batch size 32, the 7B layer latency is approximately 0.85ms. With 32 layers, total model latency per token would be roughly 27ms — but this is without distinguishing backbone from LoRA add-on time.

- The LoRA operator microbenchmark (Figure 8) shows SGMV latency of 116µs at worst (Distinct, batch size 64). For the 7B model with 7 projections per layer × 32 layers = 224 LoRA operators per forward pass, the total SGMV time is roughly 224 × 116µs ≈ 26ms in the worst case. However, the LoRA add-on is only the `AB` term, not the backbone `W` computation, so total decode latency would be substantially higher than this.

The 2ms figure appears to be an estimate of the **additional** latency Punica adds compared to running the backbone alone — i.e., the SGMV overhead per token. Reconstructing: at batch size 32 in the Distinct case, each request's share of the SGMV time across all layers might be roughly (per-layer SGMV time × number of layers) / batch size. From Figure 8, SGMV latency at batch size 32 in the Distinct case is approximately 90µs for a single operator. With 224 operators, that's 20.1ms total SGMV time. Divided by batch size 32, that's approximately 0.63ms per request per token — well under the 2ms claim. However, this calculation is speculative; the paper does not provide it.

The 2ms claim is plausible but unvalidated by a direct end-to-end latency measurement comparing Punica (with LoRA) to an identical system running backbone-only at the same batch size.

**Missing evaluations that would strengthen the paper:**

- **End-to-end latency comparison at fixed throughput.** The paper reports throughput at fixed batch size, but in production, operators often care about tail latency at a target throughput. A latency-throughput curve (similar to Figure 1 but for Punica vs. baselines across workload distributions) would show the latency cost of Punica's throughput advantage.

- **Memory consumption analysis.** The paper claims that GPU memory is abundant and can hold hundreds of LoRA models, but provides no memory footprint measurement. How much memory does the SGMV kernel's segment metadata consume? What is the actual maximum number of LoRA models that can be resident on an A100 80GB simultaneously while still having enough KvCache space to serve a useful batch size?

- **Ablation of the Split-K strategy.** The paper describes Split-K for SGMV-shrink and output-splitting for SGMV-expand, but does not show what happens if the same strategy is used for both (e.g., Split-K for expand, or output-splitting for shrink). This would validate the claim that different schedules are needed.

- **Ablation of the non-Tensor-Core path.** In the Distinct case, SGMV uses a non-Tensor-Core schedule. What is the performance if Tensor Cores are used instead? The paper argues it would be worse, but no measurement supports this.

- **Sensitivity to maximum batch size.** The max batch size of 32 is set based on Figure 1, but no experiment shows Punica's throughput and latency at different max batch sizes under multi-model workloads. Would max batch size 64 improve throughput further at acceptable latency?

- **Multi-tenant cold start.** The on-demand loading mechanism is described but not benchmarked. What happens when 100 new LoRA models are requested simultaneously? Does PCIe bandwidth become a bottleneck? Is there any queuing delay before the first token?

**What the experiments genuinely demonstrate vs. what they do not:**

The experiments **demonstrate** that a specialized CUDA kernel for segmented batched matrix-vector multiplication enables a GPU to serve requests for different LoRA models in the same batch without the catastrophic throughput collapse that affects current systems. This is the core technical contribution, and it is well-supported by the microbenchmarks and text generation experiments.

The experiments **demonstrate** that greedy busiest-GPU routing creates consolidation behavior in a cluster, reducing the number of active GPUs needed to serve a given workload. The cluster trace (Figure 13) is qualitative but convincing.

The experiments **do not demonstrate** that Punica achieves optimal or near-optimal throughput for multi-tenant LoRA serving. There is no comparison against a theoretical upper bound, no analysis of whether the SGMV kernel is compute-bound or memory-bandwidth-bound in the shared-weight regime, and no comparison against alternative scheduling policies (e.g., least-loaded, round-robin, or a policy that explicitly balances LoRA model colocation to maximize weight sharing).

The experiments **do not demonstrate** that the 2ms per-token latency overhead claim holds under all conditions. The claim is stated prominently but not directly measured.

The experiments **do not evaluate** Punica's behavior under adversarial conditions: memory pressure (many long sequences), extreme model diversity (thousands of distinct LoRA models), GPU failures, or network partitions between the scheduler and runners. These are standard concerns for production serving systems and their absence limits the paper's practical deployment story.

The experiments **are limited to Llama 2 and A100 GPUs**. The paper does not evaluate other model architectures (e.g., Falcon, Mistral, GPT-style models with different projection layouts) or other GPU architectures (e.g., H100 with different Tensor Core capabilities, or consumer GPUs with different memory bandwidth characteristics). This limits the generality of the SGMV design claims, though the paper's focus on the Llama 2/A100 combination represents a realistic and important deployment scenario.

## 6. Limitations and Trade-offs

### The 12× Throughput Figure Is Defined in a Best-Case Regime That Overstates Typical Gains

**The assumption or constraint.** The paper's headline claim — "Punica achieves 12× higher throughput" — is measured against baseline systems in the **Distinct workload**, where every request targets a different LoRA model. This is the adversarial extreme that maximizes Punica's advantage: baseline systems degrade to batch-size-1 execution because they cannot group any requests, while Punica maintains batch size 32. The paper acknowledges this explicitly in the experimental setup (Section 7) by defining four workload distributions and presenting results for all of them, but the abstract and introduction foreground the 12× figure without the qualifier that it applies to the Distinct case.

**The consequence.** In realistic multi-tenant deployments, the workload is unlikely to be purely Distinct. The Zipf-1.5 Skewed distribution — where a few popular models receive most traffic and many long-tail models receive occasional requests — is the paper's most realistic workload and is explicitly used in the cluster deployment evaluation (Section 7.3, Figure 13). Under Skewed workload, the throughput gap between Punica and baselines narrows substantially because popular models create natural batching opportunities even for baseline systems. Extracting exact numbers from Figure 11a: vLLM's throughput under Skewed appears to be roughly 300–400 tok/s (compared to ~1044 tok/s for Punica), implying approximately a 2.5–3.5× improvement rather than 12×. This is still a meaningful gain, but it is a different order of magnitude from the headline claim. A practitioner evaluating Punica for a production deployment with realistic popularity distributions should expect 2–5× improvement, not 12×. The 12× figure is the worst-case recovery ratio, not the expected gain.

**What evidence exists in the paper.** Figure 11 directly shows all four workload distributions. The Identical case shows Punica is slightly *worse* than vLLM (~0.9×), the Distinct case shows the ~12× gap, and the Uniform and Skewed cases show intermediate values. The paper presents this data transparently but does not provide exact throughput numbers for the intermediate cases, making it difficult to compute precise improvement ratios without reading values off the bar chart. The cluster deployment trace (Figure 13) uses the Skewed workload explicitly, suggesting the authors recognize this as the deployment-relevant distribution.

**Mitigation status.** Not addressed. The paper does not provide a formula or model for estimating Punica's throughput advantage as a function of workload skew (e.g., Zipf α parameter), nor does it discuss what popularity distributions make Punica most vs. least advantageous. A practitioner with a known popularity distribution cannot estimate their expected gain from the paper's results beyond the four discrete points tested.

---

### The 2ms Per-Token Latency Overhead Claim Is Unsupported by Direct Measurement

**The assumption or constraint.** The paper states in the abstract and conclusion that Punica "only adds 2ms latency per token." This claim is **never directly measured** in any experiment. The 2ms figure plausibly derives from Figure 1, which shows that increasing decode batch size from 1 to 32 on an A100 adds approximately 2ms for short sequences (128 tokens: 11ms to 13ms) — but this is a measurement of generic batching overhead on an A100, not a Punica-specific measurement of the SGMV kernel's latency contribution relative to backbone-only serving at the same batch size.

**The consequence.** The latency claim is central to Punica's value proposition: the system achieves higher throughput without meaningfully degrading user experience. If the actual per-token overhead is larger than 2ms — for example, if SGMV adds 5–10ms per token in the Distinct case due to the I/O-bound kernel schedule — then Punica's design represents a throughput-latency tradeoff rather than a pure improvement. A practitioner considering Punica for a latency-sensitive application (e.g., interactive chat where per-token latency above 30–50ms is unacceptable) cannot make an informed decision without this number. The paper's microbenchmarks (Figure 8) show that SGMV latency for a single LoRA operator is 37–116µs at batch sizes 1–64. With 224 LoRA operators per forward pass (7 projections × 32 layers for Llama 7B), the per-token SGMV overhead at batch size 32 would be roughly 224 × 90µs ≈ 20ms total SGMV time, but this is shared across the batch — per-request overhead would be 20ms/32 ≈ 0.6ms. If this back-of-the-envelope calculation is correct, the 2ms claim is actually conservative (the real overhead is lower), but the paper never validates this with an end-to-end latency comparison.

**What evidence exists in the paper.** No experiment directly measures Punica's per-token latency and compares it to backbone-only serving at the same batch size. The microbenchmarks measure kernel-level latency in isolation. The text generation experiments measure aggregate throughput (tokens/second) over 1000 requests but not per-token latency distributions. The transformer layer benchmark (Figure 10) shows total layer latency including SGMV, but doesn't separate the LoRA add-on's contribution from the backbone and self-attention contributions.

**Mitigation status.** The paper does not acknowledge this gap. The 2ms figure appears in the abstract and conclusion as an asserted fact without a citation to any experiment. This is a significant evidentiary weakness for a systems paper where latency claims are central to the deployment argument.

---

### On-Demand LoRA Weight Loading Latency Is Estimated, Not Measured, and Its Behavior Under Concurrent Load Is Unknown

**The assumption or constraint.** Section 5.2 describes the on-demand loading mechanism: when a request for a new LoRA model arrives at a GPU, the system issues an asynchronous host-to-device memory copy for the model's weights and continues executing the current batch. The paper estimates that "on PCIe Gen4 x16, it takes around 50µs to load a layer and 2ms to load the entire model," and argues that "since the memory copy and the GPU computation can overlap, it is feasible to implement... loading to minimize the model loading delay." The 2ms full-model load time is compared against a 30ms decode step to conclude that "by the end of the model execution, the weight already finished loading."

**The consequence.** This 2ms estimate is a back-of-the-envelope calculation based on LoRA weight size and PCIe bandwidth, **not a measurement**. Several plausible failure modes are unexplored:

- **Concurrent loading saturation.** If 32 new LoRA models are requested simultaneously (e.g., during a cold start or a traffic spike to unpopular models), the aggregate PCIe bandwidth demand would be 32 × 140MB = 4.5GB. At PCIe Gen4 x16's theoretical ~32 GB/s, this would take ~140ms — much longer than a single 30ms decode step. The asynchronous copies would serialize on the PCIe bus, and requests arriving mid-copy might wait multiple decode steps before their weights are available.

- **CPU-side memory pressure.** The 2ms estimate assumes LoRA weights are resident in CPU main memory. If hundreds of LoRA models are stored on disk (a realistic scenario for a multi-tenant platform with thousands of tenants), the first access to a model would incur disk I/O latency (milliseconds to tens of milliseconds), which the paper does not account for.

- **PCIe contention with KvCache migration or other I/O.** The paper does not model PCIe bandwidth as a shared resource. If request migration (which involves recomputing KvCache on a new GPU) and LoRA weight loading happen simultaneously, they compete for PCIe bandwidth.

**What evidence exists in the paper.** The 2ms estimate is stated in Section 5.2 with no supporting measurement. The paper provides no microbenchmark of LoRA weight loading latency, no experiment measuring the latency of the first token for a request targeting a not-yet-loaded LoRA model, and no stress test where many new models are loaded concurrently. The cluster deployment trace (Figure 13) does show the system operating under dynamic load, but the Skewed workload with Zipf-1.5 distribution means popular models are likely already loaded on active GPUs — this trace does not exercise the cold-start path heavily.

**Mitigation status.** The paper claims the behavior is correct by construction (2ms < 30ms, therefore hidden) but does not validate it empirically. The paper does not discuss the concurrent loading issue or acknowledge PCIe bandwidth as a potential bottleneck. This is an unvalidated design assumption that directly affects Punica's claimed ability to "allow fast cold-start for model serving" (Section 3) and serve new LoRA models without pre-warming.

---

### The Evaluation Is Restricted to a Single Model Family and GPU Architecture

**The assumption or constraint.** All experiments use Llama 2 models (7B, 13B, 70B) and NVIDIA A100 GPUs. The paper's SGMV kernel design is parameterized by the LoRA rank `r`, input dimension `h₁`, and output dimension `h₂` — but its performance characteristics depend on the interaction between these dimensions and the GPU's memory bandwidth, Tensor Core tile sizes, and SM count. The paper does not evaluate on other model architectures (e.g., Falcon, Mistral, GPT-3 style models with different projection layouts or different ratios of attention to MLP computation) or other GPU architectures (e.g., H100 with different Tensor Core capabilities and memory bandwidth, or lower-end GPUs with less parallelism).

**The consequence.** The SGMV design principles — Split-K for shrink, output-splitting for expand, non-Tensor-Core path for the degenerate case — are claimed as general, but their effectiveness is demonstrated only on one architecture. An H100 has substantially higher memory bandwidth (~3.35 TB/s vs. A100's ~2 TB/s) and different Tensor Core capabilities (FP8 support, larger tile sizes). This could shift the roofline breakpoints: operations that are memory-bandwidth-bound on A100 (like the Distinct case) might become compute-bound on H100, potentially changing the optimal kernel schedule. Conversely, on a lower-end GPU (e.g., T4 with ~320 GB/s memory bandwidth), the I/O-bound regimes would be even more severely bottlenecked, and the advantages of SGMV's efficient memory access patterns might be proportionally larger — or the reduced parallelism might limit the Split-K strategy's effectiveness. Without evaluation across GPU tiers, a practitioner cannot predict whether Punica's 12× advantage (or 2.5× on Skewed workloads) will hold on their specific hardware. Similarly, models with different hidden dimensions or different numbers of attention heads would change the ratio of backbone-to-LoRA computation, affecting the relative overhead of the LoRA add-on. The paper shows this ratio improves with model scale (the LoRA overhead drops from ~9% at 7B to ~3% at 70B), suggesting Punica is *more* advantageous for larger models — but without architectural diversity, the generality of this trend is unconfirmed.

**What evidence exists in the paper.** Figure 10 shows that the LoRA-model-agnostic performance property holds for both 7B and 13B model configurations at two sequence lengths. Figure 12 extends this to 70B with tensor parallelism. This demonstrates scaling within the Llama 2 family but does not test across architectures. The roofline analysis (Figure 7) is specific to A100 memory bandwidth (1.935 TB/s) and peak FP16 throughput (312 TFLOP/s) — these constants would change on other GPUs, shifting the boundaries between I/O-bound and compute-bound regimes.

**Mitigation status.** The paper does not claim GPU-architecture generality and does not discuss this as a limitation. The choice of Llama 2 and A100 is reasonable — they represent a widely-used open model and the dominant datacenter GPU — but the paper's claims are implicitly scoped to this combination without stating that scope. Section 8 acknowledges no generalization limitations.

---

### Request Migration Performance and Its Impact on Latency-Critical Requests Are Not Evaluated

**The assumption or constraint.** Section 5.3 describes Punica's request migration mechanism: when a GPU runs out of KvCache memory, the newest request is evicted, cancelled on the source GPU, and re-added to a destination GPU where its KvCache is recomputed via a prefill step on the original prompt plus all previously generated tokens. The paper cites Kwon et al. (2023)'s finding that "recomputation's latency is equal to or better than moving the KvCache in most cases" to justify this design choice. However, Punica's specific migration behavior is never benchmarked.

**The consequence.** Migration directly impacts user-visible latency. When a request is migrated, the user experiences a pause: the source GPU stops generating tokens for that request, the destination GPU must run a full prefill step over the entire sequence generated so far, and only then does token generation resume. For a request that has already generated 1000 output tokens, the prefill on the destination GPU must process 1000 + prompt_length tokens in a single forward pass. This prefill step is not free — Figure 1 shows that prefill latency for a 2048-token sequence at batch size 1 is roughly 2 seconds on an A100. During this time, the user sees no new tokens. The paper provides no measurement of migration-induced latency, no analysis of how frequently migrations occur under different workload characteristics (prompt length distribution, output length distribution, GPU memory capacity), and no evaluation of whether migration frequency is low enough that tail latency remains acceptable.

Additional unexamined questions: What happens to other requests in the source GPU's batch when a request is cancelled mid-execution? Does the batch shrink immediately, or does the GPU run one more step with the old batch before the cancellation takes effect? The paper says "after GPU 1 finishes running the previous batch, it picks up the cancelation and releases KvCache" — this implies a one-batch delay where the migrated request's slot is still occupied. What is the throughput impact of this transient underutilization?

**What evidence exists in the paper.** The cluster deployment trace (Figure 13, bottom panel) shows occasional drops in GPU batch size, which the paper attributes to migration: "Occasionally, a GPU runs with a smaller batch size because it runs out of KvCache space and migrates out a few requests to other GPUs." This is a qualitative observation, not a quantitative measurement. The trace shows that these drops are transient (batch size recovers within a few time steps), but no numbers are provided: what is the median migration frequency per GPU-hour? What is the 99th percentile latency impact on migrated requests? How does migration frequency scale with GPU memory capacity (40GB vs. 80GB A100s) and with sequence length distribution?

**Mitigation status.** The paper acknowledges that migration occurs and that it is induced by KvCache memory pressure, but treats it as an unremarkable background operation rather than a performance-critical event that warrants measurement. The choice to use recomputation over KvCache copying is justified by citation to prior work, not by Punica-specific benchmarking. This is a significant gap for a system that claims to support production multi-tenant serving, where tail latency and predictable performance under memory pressure are critical operational concerns.

---

### The Interaction Between SGMV and Attention Computation Creates an Unresolved Tension Between Prefill and Decode Batching

**The assumption or constraint.** Punica batches prefill and decode requests together for dense projections and the LoRA add-on (Section 5), placing prefill requests at the beginning of the batch and decode requests at the end. This combined batching increases the effective batch size for SGMV, improving its efficiency. However, the self-attention computation must treat prefill and decode separately: prefill requires attending over the full prompt (using `BatchPrefill`), while decode attends over cached past states (using `BatchDecode`). The paper sets the prefill batch size to 1 ("To minimize latency penalty, we limit the prefill batch size to 1 for each batch") while the decode batch can be up to 32.

**The consequence.** This design creates a structural inefficiency: the decode batch benefits from SGMV's batched LoRA add-on, but the prefill batch (size 1) sees no batching benefit for the LoRA add-on during the prefill step. For a request with a very long prompt (e.g., 4096 tokens), the prefill step processes all 4096 tokens sequentially through the model. At batch size 1, SGMV runs in its degenerate (Distinct-like) mode — each LoRA operator processes a single input vector against a single weight matrix. From Figure 8, this costs ~37µs per operator in the Distinct case. Across 224 operators and 4096 prompt tokens, this is 224 × 37µs × 4096 ≈ 34 seconds of SGMV time for the prefill of one long-prompt request. While the backbone GEMM dominates total prefill time (and benefits from the large inner dimension of the prompt), the LoRA add-on at batch size 1 is pure overhead with no batching benefit. The paper's design guideline (G3) states that "we only need to focus on the decode stage performance" — but for workloads with long prompts, the prefill stage can be a substantial fraction of total serving cost, and Punica provides no mechanism to batch prefill requests for different LoRA models.

Moreover, the paper does not explore whether the prefill batch size limit of 1 is fundamental. Could multiple prefill requests for different LoRA models be batched together, with SGMV handling the per-request LoRA add-ons just as it does for decode? The paper's batch layout puts all prefill requests at the beginning and decode requests at the end, suggesting multiple prefill requests *can* coexist — but the paper sets a hard limit of 1. If the limit is due to self-attention kernel constraints (FlashInfer's `BatchPrefill` may not support batching across requests with different prompt lengths efficiently), that is a real constraint that limits Punica's applicability to prompt-heavy workloads. If the limit is chosen for simplicity, an evaluation showing the latency impact of relaxing it would clarify the tradeoff.

**What evidence exists in the paper.** The prefill batch size limit is stated in Section 5 without justification beyond "to minimize latency penalty." Figure 1 shows prefill latency growing proportionally with batch size (32× batch size = roughly 32× latency), which supports the concern about prefill batching latency — but this is measured for a single model, not for multi-LoRA serving with SGMV. The paper does not evaluate workloads with long prompts (e.g., document QA, summarization) to assess whether prefill overhead becomes a bottleneck.

**Mitigation status.** The paper explicitly scopes its optimization to the decode stage via guideline (G3), so this is a declared scope limitation rather than an unacknowledged one. However, the paper does not quantify what fraction of total serving cost the prefill stage represents for different prompt length distributions, nor does it discuss under what workload conditions (G3) becomes invalid. A practitioner serving long-prompt workloads cannot determine from the paper's evaluation whether Punica's throughput advantage persists or whether prefill becomes the new bottleneck.

## 7. Implications and Future Directions
- How this changes the field
  - Multi-tenant adapter serving becomes a first-class, high-efficiency workload: one backbone per GPU, many adapters concurrently, and batching works even when adapters differ. This removes a key bottleneck for scalable, cost-effective customization of LLMs.
- Practical applications
  - Hosted fine-tuning platforms, enterprise deployments with many departmental adapters, A/B testing of adapter variants, personalization at scale (each user or team can have its own adapter without extra GPUs).
- Research and engineering directions
  - Extend SGMV-like batching to other parameter-efficient fine-tuning methods (adapters beyond LoRA).
  - Combine with speculative decoding and draft-model techniques to compound decode-stage gains (Related Work).
  - Smarter cluster schedulers: per-tenant fairness, SLO-aware batching, elastic adapter caches across nodes, prefetching adapters based on predicted popularity.
  - Explore tighter integration with quantization/compression of model and KvCache to push throughput and memory headroom further (Related Work).
  - Generalize to heterogeneous hardware (e.g., consumer GPUs) and multi-node distributed settings with adapter caching and peer-to-peer sharing.

Overall, Punica shows that the core obstacle to efficient multi-tenant serving—batching across different adapters—can be removed with the right kernel and cache layout. With SGMV and a throughput-first scheduler, it delivers near-backbone throughput even when every request targets a different LoRA model (Figures 11–12), and consolidates load effectively in a cluster (Figure 13).
