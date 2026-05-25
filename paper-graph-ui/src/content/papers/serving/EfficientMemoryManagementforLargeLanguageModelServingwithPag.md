# Efficient Memory Management for Large Language Model Serving with PagedAttention

**ArXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

## 🎯 Pitch

This paper introduces PagedAttention, a new attention algorithm and memory management strategy for LLM serving, inspired by operating system virtual memory and paging. By breaking up large, contiguous KV cache allocations into small, flexible 'pages' that can be dynamically allocated, shared, and copy-on-written, the vLLM system nearly eliminates memory waste and dramatically increases the number of requests that can be batched—enabling 2–4× higher throughput compared to prior systems at the same latency. This breakthrough directly tackles the growing memory bottleneck in LLM serving, making deploying and scaling large models vastly more cost-effective and capable.

---

## 1. Executive Summary

This paper proposes **PagedAttention**, an attention algorithm inspired by classical virtual memory paging in operating systems that partitions the key-value (KV) cache into fixed-size blocks stored in non-contiguous memory, and builds vLLM, a high-throughput distributed LLM serving engine on top of it. The system achieves near-zero waste in KV cache memory by eliminating both internal fragmentation (from pre-allocating maximum-length contiguous buffers) and external fragmentation (from variable-sized allocations), and enables flexible memory sharing across sequences within and across requests via block-level copy-on-write mechanisms — for instance, sharing prompt KV cache across parallel samples or beam search candidates. Evaluated on OPT models (13B–175B) and LLaMA-13B serving ShareGPT and Alpaca workloads against FasterTransformer and three variants of Orca, vLLM improves throughput by 2–4× at equivalent latency, with the gains growing larger for longer sequences, larger models, and more complex decoding algorithms — reaching up to 22× over FasterTransformer on ShareGPT and 2.3× over Orca (Oracle) on beam search with width 6 — while establishing that virtual-memory-inspired block management is the critical enabler for memory-bound LLM serving workloads.

## 2. Context and Motivation

### The Core Problem: KV Cache Memory Is the Bottleneck in LLM Serving

This paper tackles a specific, practical bottleneck that limits the throughput of deployed large language models: **inefficient management of the key-value (KV) cache memory**. To understand why this matters, we need to walk through what happens when an LLM serves a request.

Every time an LLM generates text, it operates autoregressively — it produces one token at a time, and each new token depends on all the tokens that came before it. To avoid recomputing the key and value vectors for all previous tokens at every step (which would be catastrophically expensive), serving systems **cache** these vectors in GPU memory. This is the KV cache. For a single token in a 13B-parameter OPT model, the KV cache requires 800 KB of memory (calculated as 2 vectors for key and value × 5120 hidden state size × 40 layers × 2 bytes for FP16). Since models can generate sequences up to 2048 tokens, a single request can consume up to **1.6 GB** of GPU memory for its KV cache alone.

The problem becomes acute at scale. A single NVIDIA A100 GPU with 40 GB of memory might hold the model parameters themselves (occupying about 65% or ~26 GB for a 13B model), leaving only about 12–14 GB for KV caches. That means only a handful of requests can be served simultaneously. As Figure 1 (left) illustrates, the KV cache is the largest dynamic component of memory consumption during serving — model weights are static, activations are small and ephemeral, but the KV cache grows and shrinks unpredictably as requests start, generate tokens, and complete.

This is not merely an academic observation. The paper cites a Reuters estimate that processing an LLM request can be **10× more expensive** than a traditional keyword query. For cloud providers racing to offer LLM services, throughput directly determines cost per request and therefore business viability. If you can double the number of requests your GPU can serve simultaneously, you halve your per-request hardware cost.

### The Problem Is Getting Worse, Not Better

A crucial detail that motivates the urgency of this work: GPU compute capacity (FLOPS) is growing faster than GPU memory capacity. The paper notes that from the NVIDIA A100 to the H100, FLOPS increased by more than 2×, while maximum GPU memory remained at 80 GB. This widening gap between compute and memory means that LLM serving is increasingly **memory-bound** — the GPU has plenty of computational power sitting idle, but it cannot batch more requests because the KV cache consumes all available memory. Efficient memory management for the KV cache is therefore not just a nice optimization; it is becoming the dominant constraint on serving throughput, and the trend line suggests this constraint will tighten, not relax.

### Three Fundamental Challenges in KV Cache Management

The paper identifies three specific characteristics of the KV cache that make it uniquely challenging to manage, and existing systems fail to handle them effectively:

**Challenge 1: The KV cache grows and shrinks dynamically, and its final size is unknown.** Unlike most tensors in deep learning workloads (where shapes are known statically at graph-compilation time), the KV cache for a request expands one token at a time until the model emits an end-of-sequence token or hits a maximum length. The serving system does not know in advance whether a request will generate 10 tokens or 2000. This uncertainty breaks standard memory allocation strategies that assume tensor dimensions are known ahead of time.

**Challenge 2: The KV cache is huge.** As quantified above, a single request can consume gigabytes of memory. This means small inefficiencies in allocation — wasted space from fragmentation, reserved-but-unused slots — translate directly into fewer requests that can be batched, and therefore lower throughput. There is no slack in the system to absorb waste.

**Challenge 3: Complex decoding algorithms create opportunities for memory sharing that go unexploited.** Modern LLM services offer more than simple greedy decoding. They support parallel sampling (generating multiple completions from a single prompt, common in code suggestion tools like GitHub Copilot), beam search (maintaining a set of top-𝑘 candidate sequences, common in translation), and shared prefixes (system-level prompts or few-shot examples that are identical across many user requests, as shown in Figure 10). In all these cases, substantial portions of the KV cache could be shared rather than duplicated — the prompt's KV cache is identical across parallel samples, beam candidates share ancestors in the search tree, and shared prefixes are identical across requests. But as the paper demonstrates, existing systems cannot exploit these sharing opportunities because they store each sequence's KV cache in isolated, contiguous memory allocations.

### Where Existing Systems Fall Short

The paper provides a detailed critique of how current serving systems manage KV cache memory, using FasterTransformer and Orca as representative examples. The root cause is the same in both: **they store the KV cache of each request in a single contiguous tensor**, as required by most deep learning frameworks (PyTorch, TensorFlow). Because they must pre-allocate memory for the maximum possible sequence length (e.g., 2048 tokens), three distinct types of waste arise, as illustrated in Figure 3:

**Reserved slots (reservation waste):** The system allocates memory for all possible future tokens upfront, even though most of those slots sit empty for the entire lifetime of the request. For a request that ultimately generates 200 tokens, 1848 slots (90% of the allocation) are reserved but unused while the request is active — and cannot be used by other requests because the allocation is exclusive. This is not a bug; it is a necessary consequence of contiguous allocation when the final length is unknown.

**Internal fragmentation:** Even beyond the reservation problem, the actual final length is almost always shorter than the maximum. For example, if a request generates 50 tokens but the pre-allocation was for 2048, the 1998 unused slots represent pure waste. The paper's profiling in Figure 2 quantifies this: in existing systems, only **20.4% to 38.2%** of the allocated KV cache memory actually stores useful token states. The rest — 60–80% — is some combination of reservation waste and internal fragmentation.

**External fragmentation:** Since different requests pre-allocate different amounts (based on their potentially different maximum lengths, or because the memory allocator — typically a buddy allocator — assigns varying block sizes), the free space becomes fragmented into pieces that are individually too small to satisfy new allocation requests, even though the total free memory might be sufficient. This is the classic external fragmentation problem from operating systems, transplanted to GPU memory management.

Figure 2 breaks down these wastes quantitatively: For Orca (Max) — which always reserves 2048 tokens — only 20.4% of KV cache memory is actually used for token states, with 57.3% in reservation waste and 13.3% in internal fragmentation. Even Orca (Oracle) — which unrealistically knows the exact output length in advance — still wastes 38.2% to reservation and 25.2% to external fragmentation, achieving only 36.6% useful utilization. This is the upper bound for contiguous-allocation systems.

**FasterTransformer** fares even worse in the throughput comparison because it lacks fine-grained iteration-level scheduling (discussed below). Its scheduler processes requests in coarse batches: it waits for all requests in a batch to complete before starting new ones. This means that when one request in a batch finishes early, its allocated memory sits idle until the entire batch completes. Combined with the fragmentation issues above, FasterTransformer can sustain up to **22× lower request rates** than vLLM in the paper's experiments.

### The Infeasibility of Compaction

A natural question: why not just compact memory like a garbage collector? Move the KV caches around to defragment the space, as log-structured key-value stores do. The paper acknowledges this possibility but argues it is impractical for LLM serving. The KV cache is massive — gigabytes per request, tens of gigabytes in aggregate. Copying that much data in a latency-sensitive online serving system would introduce unacceptable pauses. Moreover, even perfect compaction would not solve the reservation problem: the pre-allocated chunk for each request still prevents other requests from using the reserved-but-empty slots during the request's lifetime.

### The Complementary Role of Iteration-Level Scheduling

The paper builds on a key insight from prior work, particularly Orca: **fine-grained, iteration-level scheduling**. Traditional serving systems batch requests at the request level — a batch is formed, all requests are processed to completion together, and only then can new requests join. This causes two problems: (1) early-arriving requests must wait for the batch to fill, adding queueing delay, and (2) requests of vastly different lengths must be padded to the same length, wasting computation and memory.

Orca introduced iteration-level scheduling: after each generation step (each token), the system can remove completed requests from the batch and insert new ones. This eliminates padding and reduces queueing delay because a new request only waits for the current iteration to finish, not for an entire batch to complete.

However, this paper identifies a critical tension: **iteration-level scheduling makes memory management harder, not easier**. When requests dynamically enter and leave the batch at every iteration, the KV cache allocations and deallocations become highly dynamic. The contiguous-allocation approach of Orca struggles precisely because the allocation pattern becomes unpredictable — many requests of varying lengths are in flight simultaneously, entering and exiting asynchronously. This makes the fragmentation and reservation problems *more* severe, not less. The paper's contribution — PagedAttention — is therefore not an alternative to iteration-level scheduling but a **necessary complement** that makes efficient memory management possible under such dynamic scheduling.

### Explicit Gap: No General-Purpose Framework for KV Cache Memory Management

The paper's framing positions prior work as falling into two categories, neither of which addresses the KV cache memory problem directly:

**General model serving systems** (Clipper, TensorFlow Serving, Nexus, InferLine, Clockwork) study batching, caching, and scheduling for arbitrary DNN workloads, but they do not account for the autoregressive generation process and the associated KV cache state, missing optimization opportunities specific to LLMs.

**Specialized serving systems for Transformers** (FasterTransformer, TurboTransformers, Orca, DeepSpeed Inference) focus on GPU kernel optimization, efficient batching, and model parallelism. Orca is the most relevant prior system because it introduces iteration-level scheduling. But as discussed, Orca still uses contiguous KV cache allocation and inherits all the fragmentation and sharing limitations. The paper explicitly states that vLLM and Orca are **complementary**: "While both systems aim to increase the GPU utilization and hence the throughput of LLM serving, Orca achieves it by scheduling and interleaving the requests so that more requests can be processed in parallel, while vLLM is doing so by increasing memory utilization so that the working sets of more requests fit into memory."

**Memory optimization systems** (FlexGen, OLLA, FlashAttention) address related but distinct problems. FlexGen studies how to swap weights and token states for LLM inference with limited GPU memory, but targets offline batch processing rather than online serving with latency constraints. OLLA optimizes tensor lifetime and placement but operates at the whole-tensor granularity, not the fine-grained block level needed for dynamic sharing. FlashAttention reduces peak memory during attention computation through tiling and kernel optimization, which is orthogonal to the KV cache *storage* problem.

### How This Paper Positions Itself

The paper's central insight is to recognize that the KV cache management problem is structurally analogous to the memory management problem that operating systems solved decades ago with **virtual memory and paging**. Just as OSes faced the challenge of allocating variable-sized memory regions for processes with unknown lifetimes and unpredictable growth, LLM serving faces the challenge of allocating variable-length KV caches for requests with unknown output lengths and unpredictable arrival/completion patterns.

In operating systems, the solution was to divide physical memory into fixed-size **pages** and present each process with a contiguous **virtual address space** that maps to (potentially non-contiguous) physical pages via a page table. This eliminated external fragmentation (all pages are the same size), eliminated the need to pre-allocate all possible memory upfront (pages are allocated on demand), and enabled sharing (multiple processes can map the same physical page, with copy-on-write for modifications).

The paper transplants this entire conceptual framework to the KV cache domain:

- **Tokens** are analogous to **bytes** (the smallest unit of data).
- **KV blocks** (fixed-size groups of tokens) are analogous to **pages**.
- **Requests** are analogous to **processes** (each has its own logical address space).
- **Logical KV blocks**: The sequence of blocks that form a request's KV cache, analogous to virtual pages.
- **Physical KV blocks**: The actual GPU memory blocks that store the KV data, analogous to physical page frames.
- **Block table**: The per-request mapping from logical to physical blocks, analogous to the page table.

The paper calls this adapted attention algorithm **PagedAttention**, and the serving system built on it **vLLM**. The positioning is that vLLM does not compete with Orca-style scheduling — it is the **memory management substrate** that enables Orca-style scheduling to achieve its full potential by removing the memory bottleneck. The paper demonstrates that when memory is managed efficiently via paging, the same scheduling policy can batch substantially more requests, translating the improved memory utilization directly into higher throughput.

This framing also explains why the paper emphasizes that the same ideas do not generalize to all GPU workloads. For DNN training, tensor shapes are static and known ahead of time — pre-allocation works fine. For non-LLM serving workloads that are compute-bound rather than memory-bound, improving memory efficiency does not improve throughput. The paging approach is specifically valuable for workloads that combine three properties: dynamic, unpredictable memory allocation, memory-bound performance, and opportunities for sharing across concurrent processes — and LLM serving hits all three.

## 3. Technical Approach

### 3.1 Reader Orientation

vLLM is a high-throughput distributed serving system for large language models that manages the key-value (KV) cache — the memory storing attention keys and values from all previous tokens — using fixed-size blocks mapped through a block table, exactly as an operating system manages virtual memory with pages mapped through a page table. The system solves the problem that existing LLM servers waste 60–80% of KV cache memory through fragmentation and pre-allocation, by allowing KV cache blocks to live anywhere in GPU memory, be allocated on demand as tokens are generated, and be shared across sequences via copy-on-write, thereby batching 2–4× more requests onto the same GPU hardware.

### 3.2 Big-Picture Architecture (Diagram in Words)

The vLLM architecture, shown in Figure 4, has four major components orchestrated by a centralized controller:

- **Centralized Scheduler:** Receives requests from the FastAPI frontend, decides which requests to batch at each iteration (using first-come-first-serve policy), and broadcasts control messages to all GPU workers. It holds the global block table mapping each request's logical KV blocks to physical GPU memory blocks.

- **KV Cache Manager (Block Manager):** Manages the allocation and deallocation of physical KV blocks on both GPU DRAM and CPU RAM. It maintains reference counts for shared blocks, implements copy-on-write semantics when blocks are written by multiple sequences, and handles preemption via swapping (to CPU RAM) or recomputation.

- **GPU Workers (Model Shards):** Each worker holds a shard of the model parameters (split across attention heads following Megatron-LM tensor parallelism) and executes the Transformer forward pass for its shard. Workers receive token IDs and block tables from the scheduler, execute the PagedAttention kernel to read KV cache from block-specified locations, and return sampled tokens to the scheduler.

- **Block Engine (Physical Memory):** A contiguous chunk of GPU DRAM (and CPU RAM for swapping) divided into fixed-size physical KV blocks. The block engine provides allocate/free operations and copy primitives (fused into single kernel launches for efficiency).

Information flows as follows: Requests arrive at the FastAPI frontend → the scheduler queues them and forms batches at each iteration boundary → the scheduler broadcasts token IDs and per-request block tables to all GPU workers → each worker looks up physical KV block addresses from the block table, executes the model forward pass with PagedAttention, and returns sampled tokens → the scheduler updates block tables (allocating new physical blocks if the last logical block is full), removes completed sequences, and inserts newly arrived requests → the cycle repeats at the next iteration.

### 3.3 Roadmap for the Deep Dive

- **First, PagedAttention itself** (§3.4, PagedAttention Algorithm) — the block-wise attention formulation that enables non-contiguous KV cache storage, because every other component depends on this mathematical transformation.

- **Second, how the KV Cache Manager maps logical blocks to physical blocks** (§3.4, KV Cache Manager: Block Table Mapping) — the block table data structure and the dynamic allocation protocol, because this is the mechanism that eliminates fragmentation.

- **Third, the step-by-step decoding walkthrough** (§3.4, End-to-End Decoding with PagedAttention and vLLM) — how a single request's KV cache grows block by block, because this concretizes the abstract mapping into an operational procedure.

- **Fourth, memory sharing mechanisms** (§3.4, Memory Sharing for Complex Decoding Algorithms) — how parallel sampling, beam search, and shared prefixes exploit physical block sharing with copy-on-write and reference counting, because these are the features that provide much of vLLM's throughput advantage.

- **Fifth, scheduling and preemption** (§3.4, Scheduling and Preemption) — how vLLM decides which sequences to evict when GPU memory is exhausted, and the two recovery mechanisms (swapping and recomputation) with their tradeoffs.

- **Sixth, distributed execution** (§3.4, Distributed Execution with Tensor Model Parallelism) — how the centralized block manager coordinates GPU workers that each hold a different attention-head shard but process the same tokens.

- **Seventh, kernel-level optimizations** (§3.4, Kernel-Level Optimizations) — the fused GPU kernels that hide the overhead of block table indirection.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is a **systems paper** whose core idea is that managing the KV cache as fixed-size page-like blocks — rather than as contiguous per-request tensors — eliminates memory fragmentation, enables on-demand allocation, and enables block-granularity sharing across sequences, all with acceptable computational overhead from memory indirection. The contribution is both the PagedAttention algorithm (the mathematical transformation that makes block-wise attention correct) and vLLM (the full serving system that implements the block manager, scheduler, and optimized kernels).

---

#### PagedAttention Algorithm

**What it is and why it is necessary.** Traditional attention implementations (including all prior serving systems) assume that the key and value vectors for positions `$1$` through `$n$` are stored as a single contiguous tensor in GPU memory. This is because the standard attention computation iterates over positions sequentially, and GPU kernels are most efficient when data is laid out contiguously. However, this contiguous requirement forces the KV cache for each request to be allocated as one monolithic chunk, creating all the fragmentation problems described in Section 2.

PagedAttention relaxes this requirement by reformulating attention so that the key and value vectors can be **partitioned into fixed-size blocks** and those blocks can be stored independently at arbitrary locations in GPU memory. During attention computation, the kernel fetches each block separately, performs partial dot products, and accumulates the results. The reformulation is mathematically equivalent to standard attention — the output is identical — but the memory layout is fundamentally different.

**The block-wise attention reformulation.** The paper begins with the standard attention equation (Equation 4 in the original paper, which is Equation 3 in the background section). For a query vector `$q_i$` at position `$i$`, the attention output `$o_i$` is computed as a weighted sum over all previous value vectors `$v_j$`, where the weights `$a_{ij}$` are softmax-normalized dot products between `$q_i$` and each previous key vector `$k_j$`:

$$a_{ij} = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_{t=1}^{i} \exp(q_i^\top k_t / \sqrt{d})}, \quad o_i = \sum_{j=1}^{i} a_{ij} v_j$$

where `$d$` is the head dimension (the size of each key/query/value vector), `$k_j \in \mathbb{R}^d$` is the key vector at position `$j$`, `$v_j \in \mathbb{R}^d$` is the value vector at position `$j$`, and `$q_i \in \mathbb{R}^d$` is the query vector at the current position `$i$`.

**What it computes:** For each query position `$i$`, this computes a convex combination (weights summing to 1) of all previous value vectors, where the weight for position `$j$` is proportional to the exponentiated dot product between the query at `$i$` and the key at `$j$`, scaled by `$1/\sqrt{d}$` to prevent the dot products from growing too large. The output `$o_i$` is a `$d$`-dimensional vector that represents the "context-aware" representation at position `$i$`.

**Why this form:** The softmax normalization ensures that the attention weights form a valid probability distribution over previous positions, giving the model a differentiable mechanism to "attend" more strongly to some positions than others. The `$1/\sqrt{d}$` scaling factor counteracts the variance growth of dot products as the dimension increases, which would otherwise push the softmax into a saturated regime where only one position receives near-zero weight.

Now, PagedAttention **reorganizes** this computation by grouping consecutive tokens into blocks. Let `$B$` be the block size (the number of tokens per block, typically 16). The key block `$K_j$` consists of the key vectors for tokens `$(j-1)B+1$` through `$jB$`, concatenated into a matrix:

$$K_j = (k_{(j-1)B+1}, \ldots, k_{jB}) \in \mathbb{R}^{B \times d}$$
$$V_j = (v_{(j-1)B+1}, \ldots, v_{jB}) \in \mathbb{R}^{B \times d}$$

The attention computation then processes blocks rather than individual tokens:

$$A_{ij} = \frac{\exp(q_i^\top K_j / \sqrt{d})}{\sum_{t=1}^{\lceil i/B \rceil} \exp(q_i^\top K_t \mathbf{1} / \sqrt{d})}, \quad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top$$

where `$A_{ij} = (a_{i,(j-1)B+1}, \ldots, a_{i,jB})$` is a row vector of length `$B$` containing the attention scores for all tokens in block `$j$`, `$\lceil i/B \rceil$` is the number of blocks needed to cover all tokens up to position `$i$`, and `$\mathbf{1}$` is a vector of ones (used to sum the exponentiated scores within a block for the softmax denominator).

**What it computes:** The exact same attention output `$o_i$` as the original equation, but organized so that each block `$K_j, V_j$` can be read independently from an arbitrary memory location. The query vector `$q_i$` first multiplies against the entire key block `$K_j$` (producing `$B$` scalar scores), which after softmax normalization become the `$A_{ij}$` vector, and then these `$B$` scores weight the `$B$` value vectors in `$V_j$` to produce a partial sum. The partial sums from all blocks are added together to produce `$o_i$`.

**Why this form:** The block formulation enables three crucial properties. First, each block is **self-contained** — the computation for block `$j$` depends only on `$q_i$`, `$K_j$`, and `$V_j$`, and the blocks can be processed in any order (they are associative under addition). Second, blocks can be **stored independently** — there is no requirement that block `$j$` and block `$j+1$` be adjacent in memory, because the kernel explicitly looks up each block's address from the block table rather than assuming contiguity. Third, the softmax normalization needs to know the sum of exponentiated scores across **all** blocks, which requires two passes through the blocks (one to compute the denominator, one to compute the weighted sum), but this is standard in numerically stable softmax implementations and adds minimal overhead.

**Figure 5 illustrates concretely.** The figure shows a query token "forth" attending over three non-contiguous blocks. Block 0 contains the key and value vectors for "Four score and seven" (4 tokens), Block 1 contains "years ago our" (3 tokens plus 1 empty slot), and Block 2 contains "fathers brought" (2 tokens). The PagedAttention kernel separately fetches each physical block, computes the partial dot products and weighted sums, and accumulates the results.

**Block size as a design parameter.** The paper explicitly notes that block size `$B$` (called "KV block size") is a tunable parameter with a tradeoff. The default value is **16 tokens per block**. A larger block size increases GPU parallelism (the kernel processes `$B$` key-value pairs at once, making better use of GPU warps) and reduces the number of block table lookups, but increases internal fragmentation (the last block of a sequence may be mostly empty if the sequence length is not a multiple of `$B$`) and reduces sharing granularity (copy-on-write must copy an entire block even if only one token differs). The paper's ablation study (§7.2, Figure 18b) evaluates block sizes from 1 to 256 and finds that block sizes 16 through 128 perform well for the ShareGPT workload (which has long sequences), while block sizes 16 and 32 work well for Alpaca (shorter sequences). The default of 16 is chosen because it is "large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads."

---

#### KV Cache Manager: Block Table Mapping

**The core data structure.** The KV cache manager maintains a **block table** for each request (and for each sequence within a request). The block table is a simple mapping: each entry corresponds to one logical KV block (indexed `$0, 1, 2, \ldots$` representing the `$i$`-th block in the sequence) and contains two fields:

- **Physical block number:** The ID of the physical GPU memory block where this logical block's key and value vectors are actually stored. This is an integer index into the pool of pre-allocated physical blocks.
- **Number of filled positions (#filled):** How many of the `$B$` slots in this block actually contain valid KV cache data. For all blocks except the last one, this equals `$B$` (the block is full). For the last logical block of an active sequence, this is a value between 1 and `$B$` indicating how many tokens have been generated so far in that block, with the remaining `$B - \text{\#filled}$` slots reserved for future tokens.

**Physical block allocation.** At system startup, the block engine allocates a single contiguous chunk of GPU DRAM and divides it into `$N$` physical blocks, each of size `$B \times (\text{bytes per KV element})$`. For a model with hidden size `$H$` and `$L$` layers, using FP16 (2 bytes), each physical block stores:

- For each of the `$L$` layers: key and value vectors for `$B$` tokens, each with dimension `$H$`. However, since the model is sharded across GPU workers using tensor parallelism (split on attention heads), each worker only stores the KV cache for its assigned subset of attention heads. If there are `$W$` workers and `$A$` total attention heads, each worker stores `$A/W$` heads' worth of KV vectors per token.

The physical block pool is managed by a simple allocator that tracks which blocks are free and which are in use. Allocation is constant-time: pop a block ID from the free list. Deallocation is similarly constant-time: push the block ID back onto the free list.

**Why this eliminates fragmentation.** External fragmentation is eliminated because **all physical blocks are the same size**. Every allocation request is for exactly one block, so any free block can satisfy any allocation — there is no scenario where a request needs 3 MB of contiguous memory but the only free regions are 2 MB and 1 MB that are not adjacent. This is the same insight that made paging effective in operating systems: fixed-size allocation units eliminate the need for contiguous free regions.

Internal fragmentation is dramatically reduced because blocks are allocated **on demand**. When a request starts, it is allocated only as many blocks as needed to store the prompt's KV cache (for a prompt of `$P$` tokens, that is `$\lceil P/B \rceil$` blocks). As the model generates new tokens, new blocks are allocated one at a time only when the current last block becomes full. The only waste is the unused portion of the final block, which is at most `$B-1$` tokens (less than one block). For `$B=16$`, this is at most 15 tokens ≈ 12 KB of waste per request — negligible compared to the 1.6 GB maximum allocation in existing systems. Figure 2 confirms this quantitatively: vLLM achieves 96.3% KV cache utilization (only 3.7% waste), compared to 20.4–38.2% for existing systems.

**Logical-to-physical separation.** A critical design choice is that the LLM computation kernel and the attention kernel **never see logical blocks**. They operate exclusively on physical block IDs. The scheduler translates logical block indices to physical block IDs using the block table before sending the batch to the GPU workers. This means the attention kernel receives a flat list of physical block IDs for each sequence and reads directly from those addresses — it has no awareness of which logical blocks they correspond to, nor of any sharing relationships between sequences. This **abstraction layer** is what enables the memory sharing features (§4.4): the kernel does not need to handle copy-on-write or reference counting; those are managed purely by the block manager at the allocation level.

**Figure 6 concretizes the mapping.** Request A has a prompt of 7 tokens and generates output tokens. Its block table shows:
- Logical block 0 → physical block 7, #filled = 4 (full: tokens 1–4).
- Logical block 1 → physical block 1, #filled = 3 (initially, after prompt: tokens 5–7, with one slot reserved).
- After generating token 8 (the second autoregressive step), logical block 1 becomes full (#filled = 4), and a new logical block 2 is created → physical block 3, #filled = 1 (contains the newly generated token 8).
- Logical block 3 is not yet allocated (shown as "–" in the figure).

**The block table as a page table.** The analogy to OS page tables is exact. The block table entry is like a page table entry (PTE): it maps a virtual page number (logical block index) to a physical page frame number (physical block ID). The #filled field has no direct OS analog — it is a vLLM-specific optimization that tells the attention kernel how many token positions within a block are valid, avoiding computation on the reserved-but-empty slots.

---

#### End-to-End Decoding with PagedAttention and vLLM

The paper provides a step-by-step walkthrough of how vLLM processes a single request from arrival to completion, using Figure 6. I will expand this into a precise operational description:

**Step 0: Request arrival and prompt processing.** A request arrives with a prompt of `$P$` tokens. The scheduler computes how many logical blocks are needed: `$\lceil P/B \rceil$`. It allocates that many physical blocks from the GPU block allocator and populates the block table. For the last logical block (if `$P$` is not a multiple of `$B$`), the `#filled` field is set to `$P \bmod B$` (or `$B$` if `$P$` is a multiple of `$B$`); for all earlier blocks, `#filled = B`.

The prompt phase then executes: the entire prompt is fed through the model in one forward pass (using matrix-matrix multiplication for efficiency). The KV cache for all prompt tokens is computed and stored into the allocated physical blocks. The model also generates the first output token (token `$P+1$`). The KV cache for this first output token is stored in the next available slot in the last logical block if there is space (#filled < B), or in a newly allocated physical block if the last logical block is full.

**Step 1 and beyond: autoregressive generation.** At each subsequent iteration, the model takes the single most recently generated token as input and produces the next token. The attention computation uses the PagedAttention kernel: for each sequence, the kernel reads the block table, fetches each physical block's key and value vectors, computes attention scores against the new query vector, weights the value vectors, and sums them. This produces the context-aware hidden state for the new token, which then flows through the feed-forward layers to produce the logits for the next token.

After the new token is generated, its KV cache is stored. If the last logical block has space (#filled < B), the new KV vectors are written into the next available slot, and #filled is incremented. If the last logical block is full (#filled = B), the scheduler allocates a new physical block, appends a new entry to the block table mapping the new logical block to this physical block with #filled = 1, and writes the KV vectors to the first slot.

This cycle repeats until either an end-of-sequence token is emitted or the sequence reaches the maximum length (2048 tokens for OPT). At that point, all physical blocks allocated to the sequence are freed (their reference counts are decremented; when a reference count reaches zero, the block is returned to the free list).

**Figure 6 walkthrough, annotated with the paper's numbering:**

1. **Prompt processing:** The prompt has 7 tokens (`$B=4$`), so vLLM allocates `$\lceil 7/4 \rceil = 2$` physical blocks. The first 4 tokens ("Four", "score", "and", "seven") go into logical block 0 → physical block 7, #filled = 4. The next 3 tokens ("years", "ago", "our") go into logical block 1 → physical block 1, #filled = 3. One slot remains available. The prompt phase also generates the first output token "fathers", which is stored in logical block 1's last slot, making #filled = 4 (full).

2. **First autoregressive step:** The model generates "brought". Logical block 1 is already full (#filled = 4), so the scheduler allocates a new physical block (block 3). The block table now has logical block 2 → physical block 3, #filled = 1. The KV cache for "brought" is stored in physical block 3, slot 0. The PagedAttention kernel reads physical blocks 7 (for tokens 1–4), 1 (for tokens 5–8 = prompt tokens 5–7 + "fathers"), and 3 (for "brought").

3. **Second autoregressive step:** The model generates the next token (not shown in the figure, but the block table for logical block 3 shows "–" meaning it has not yet been allocated — this reflects the state *before* this step). After this token is generated, logical block 2 becomes full (#filled becomes 4 if it was 1 before, but the figure shows #filled for logical block 2 going from 1 to 2, then 3, then 4 as more tokens are generated). The process continues similarly.

**What happens when the block table is broadcast to GPU workers.** At each iteration, the scheduler prepares a control message containing (a) the input token IDs for each sequence in the batch (for prompt-phase sequences, this is all prompt tokens; for generation-phase sequences, this is just the single latest token), and (b) the block table for each sequence (logical block index → physical block ID → #filled). The GPU workers receive this message, execute the model forward pass using the PagedAttention kernel, and return the sampled tokens. The GPU workers do not make any memory management decisions — they only read and write to the physical blocks specified by the scheduler.

---

#### Memory Sharing for Complex Decoding Algorithms

This section explains the three sharing scenarios and the unified mechanisms (fork, append, free, copy-on-write, reference counting) that implement them. These are the features that distinguish vLLM from any prior system, because no prior system could share arbitrary portions of the KV cache at block granularity.

**The three primitive operations.** vLLM implements all decoding algorithms using three key methods on sequences:

- **`fork`(parent_sequence) → child_sequence:** Creates a new sequence that initially shares all logical blocks with the parent. The block table of the child is initially a copy of the parent's block table, and the reference count of every physical block referenced by the parent is incremented (since both sequences now reference it). The child has its own block table that can diverge when it writes to a shared block (via copy-on-write), but initially there is no memory cost beyond the block table itself.

- **`append`(sequence, new_token):** Adds a new token to the sequence's KV cache. If the last logical block has space, the token's KV vectors are written directly. If the last block is full, a new physical block is allocated. Importantly, if the last block is shared (reference count > 1), the write triggers copy-on-write before the append proceeds.

- **`free`(sequence):** Deletes the sequence and decrements the reference count of all physical blocks it references. Any physical block whose reference count drops to zero is returned to the free list and can be reused.

**Copy-on-write mechanism.** When a sequence needs to write to a physical block (either because the last block has space and a new token needs to be stored, or because beam search modifies candidate KV caches), the block manager checks the reference count of that physical block. If the reference count is 1, the sequence is the sole owner and can write directly. If the reference count is greater than 1, the sequence does not own the block exclusively — other sequences share it. To avoid corrupting those other sequences, the block manager:

1. Allocates a new physical block from the free list.
2. Copies the entire contents of the old physical block to the new block.
3. Updates the sequence's block table to point the logical block to the new physical block.
4. Decrements the reference count of the old physical block.
5. Sets the reference count of the new physical block to 1.
6. Then the write proceeds on the new (now exclusively owned) physical block.

This is identical to how operating systems handle `fork()`: the parent and child process share the same physical pages after a fork, and when either process writes to a shared page, the OS copies the page to a new physical frame and updates the page table for the writing process. vLLM's granularity is a KV block (default 16 tokens) rather than a memory page (typically 4 KB), but the semantics are the same.

**Parallel sampling (Figure 8).** A request wants `$S$` independent output sequences from a single prompt. The procedure:

1. The prompt is processed once, generating the KV cache for the prompt tokens and the first output token. These are stored in physical blocks with reference counts initialized to 1 (for the single "root" sequence).

2. `fork` is called `$S-1$` times to create the child sequences. After forking, there are `$S$` sequences, each with its own block table, but all pointing to the same physical blocks for the prompt's KV cache. The reference counts of those physical blocks are now `$S$`.

3. At each subsequent iteration, each sequence generates its own next token independently. When a sequence needs to write to its last logical block (the one containing the first output token after the prompt), it triggers copy-on-write: a new physical block is allocated, the block's contents are copied, the reference count of the original block decrements (from `$S$` to `$S-1$`, then `$S-2$`, etc.), and the sequence writes to its private copy.

4. After the first divergent write, each sequence has a private copy of the "prompt + first token" block (or at least the last block that they modified), but continues to share all earlier prompt blocks that were not modified, since those never receive writes during generation.

**Memory savings.** For a prompt of `$P$` tokens with `$S$` parallel samples and block size `$B$`, the prompt occupies `$\lceil P/B \rceil$` blocks. Without sharing, each sample would need its own copy of these blocks, for a total of `$S \times \lceil P/B \rceil$` blocks. With sharing, the `$\lfloor P/B \rfloor$` fully-filled prompt blocks are shared (never modified during generation, since the prompt is fixed once generated), and only the last prompt block (which also contains the first generated token, and thus gets written) must be copied `$S-1$` times. The savings are approximately `$(S-1) \times \lfloor P/B \rfloor$` blocks. The paper's measurements in Figure 15a show 6.1–9.8% memory savings for parallel sampling on Alpaca (where prompts are short, mean 19.3 tokens) and 16.2–30.5% on ShareGPT (where prompts are longer, mean 161.3 tokens).

**Beam search (Figure 9).** Beam search with width `$k$` maintains `$k$` candidate sequences that are extended at each step. After ranking all `$k \times |V|$` possible extensions (where `$|V|$` is vocabulary size), the top `$k$` are kept. This creates a tree-structured sharing pattern: candidates share ancestors, and the sharing pattern changes dynamically as the beam evolves:

- At any step, multiple beam candidates may share some prefix of their KV cache. If candidates 0, 1, and 2 all originated from the same parent at step `$t$`, they share blocks for all tokens up to step `$t$`.
- When the beam is updated, some candidates are discarded. Their logical blocks are freed (reference counts decremented), and if any physical block's reference count drops to zero, it is returned to the free list.
- New candidates that are extensions of existing candidates `fork` from their parent, sharing all parent blocks initially. As they generate new tokens, they allocate new physical blocks and trigger copy-on-write only when modifying the last shared block.

Figure 9 shows a concrete example with `$k=4$`. Before the dotted line, each candidate has 4 full logical blocks. Candidates 0–2 share blocks 0, 1, and 3, diverging at block 6 vs. 7. Candidate 3 diverged earlier, sharing only block 0. After the beam update, the top-4 candidates all originate from candidates 1 and 2. Original candidates 0 and 3 are discarded — their logical blocks are freed, decreasing reference counts on the corresponding physical blocks. Blocks whose reference counts reach zero (blocks 2, 4, 5, 8) are freed. New physical blocks (9–12) are allocated for the new tokens generated by the surviving candidates. The result: all four new candidates share blocks 0 and 1; candidates 0–1 share block 3 and 6; candidates 2–3 share block 3 and 7.

Figure 15b quantifies the memory savings: 37.6–55.2% on Alpaca and 44.3–66.3% on ShareGPT, growing with beam width.

**A critical performance benefit over prior systems.** In prior systems (FasterTransformer, Orca), each beam candidate's KV cache is stored in a separate contiguous allocation. When a candidate needs to copy another candidate's KV cache (which happens frequently during beam updates — a new top candidate that extends from an existing candidate needs the existing candidate's KV cache as a starting point), the entire KV cache must be physically copied from one contiguous buffer to another. For a long sequence, this copy can involve gigabytes of data movement. In vLLM, the `fork` operation is essentially free: it just copies the block table (a few dozen entries) and increments reference counts on all shared physical blocks. The physical data is never copied unless and until a divergence actually requires a copy-on-write on a single block. This eliminates the dominant data-movement cost of beam search.

**Shared prefix (Figure 10).** Many LLM applications prepend a common system prompt, task description, or few-shot examples to each user query. For example, a translation service might prepend:
> "Translate English to French: 'sea otter' => 'loutre de mer', 'peppermint' => 'menthe poivrée', ..."

This prefix is identical across all requests. In vLLM, the service provider can pre-compute the KV cache for the shared prefix and store it in a set of dedicated physical blocks. When a new request arrives, its logical blocks for the prefix portion of the prompt are simply mapped to these pre-computed physical blocks, with the last block marked copy-on-write (so the request can write its own generated tokens into the tail end of the last prefix block if the prefix length is not a multiple of `$B$`). The prompt processing phase only needs to compute the KV cache for the user-specific suffix of the prompt, not for the entire prefix. This saves both memory (one copy of the prefix's KV cache serves all requests) and computation (the prefix is not re-encoded for each request).

The paper evaluates this with LLaMA-13B on a machine translation workload (WMT16 English-to-German), using prefixes of 1 example (80 tokens) and 5 examples (341 tokens). vLLM achieves 1.67× and 3.58× higher throughput than Orca (Oracle) for the one-shot and few-shot prefixes respectively (Figure 16), because Orca must store and compute the full prefix for every request independently.

**Mixed decoding methods.** The paper emphasizes that vLLM can simultaneously serve requests using different decoding algorithms (greedy, sampling, beam search, different beam widths) within the same batch. This is because the attention kernel receives only physical block IDs per sequence — it does not know or care whether those blocks are shared, how the sequence was created, or what decoding algorithm produced it. The mapping layer (block table) absorbs all the complexity, presenting a uniform interface to the model executor. This increases batching flexibility: a scheduler does not need to group requests by decoding type.

---

#### Scheduling and Preemption

**Scheduling policy.** vLLM uses **first-come-first-serve (FCFS)** scheduling. Requests are queued in arrival order, and at each iteration, the scheduler selects as many of the earliest-arrived requests as can fit into the batch (subject to GPU memory availability). When the system is overloaded (more requests arriving than can be served), FCFS ensures fairness and prevents starvation — the oldest requests always get priority.

**The preemption problem.** LLM serving faces a unique challenge that traditional request-serving systems do not: the memory consumption of an active request **grows over time** as it generates output tokens, and the final output length is not known in advance. A request that starts with a short prompt and generates 10 tokens consumes relatively little memory. The same request, if it generates 1500 tokens, will consume 150× more memory for its KV cache over its lifetime. As more requests arrive and existing requests continue generating, vLLM can exhaust all available physical blocks on the GPU.

When physical blocks are exhausted but new tokens need to be generated for existing sequences, vLLM must **preempt** (evict) some sequences to free memory. The scheduler faces two classic questions: which sequences to evict, and how to recover their state if they are later resumed.

**Eviction policy: all-or-nothing, gang-scheduled.** Unlike OS virtual memory where individual pages of a process can be evicted independently (leading to page faults when the process accesses an evicted page), vLLM uses an **all-or-nothing eviction policy**: when a sequence is preempted, all of its physical blocks are evicted together. The rationale is specific to the LLM serving workload: generating a token requires access to **all** previous tokens' KV cache (the entire sequence up to the current position), so evicting only a subset of blocks would not help — any single missing block would stall the generation. Evicting all-or-nothing avoids the complexity of partial eviction and eliminates the fragmentation that partial eviction would cause in the physical block pool.

Furthermore, multiple sequences within a single request (e.g., the `$k$` beam candidates in a beam search request) are treated as a **sequence group** and are **gang-scheduled**: they are always preempted and resumed together. This is necessary because sequences within a group may share physical blocks — preempting one candidate but not its sibling would complicate reference counting (the sibling's shared blocks cannot be freed) and create coordination complexity. Gang-scheduling ensures that shared blocks are freed in a coherent state.

**Which sequences to evict?** Since FCFS is used for admission, the preemption policy is symmetric: the **latest-arrived** requests are preempted first. This means that when memory is exhausted, the system does not penalize long-running requests; instead, it evicts the most recently admitted ones that have made the least progress. This is consistent with the fairness principle of FCFS — older requests have "earned" their memory allocation by having waited longer.

**Two recovery mechanisms.** When a preempted sequence is later resumed (because other sequences completed and freed their blocks, creating room in GPU memory), its KV cache must be restored. vLLM offers two mechanisms with different tradeoffs:

**Swapping.** The evicted blocks are copied from GPU DRAM to CPU RAM (which serves as swap space). When the sequence is resumed, the blocks are copied back from CPU RAM to GPU DRAM. The vLLM architecture (Figure 4) includes a **CPU block allocator** that manages the physical blocks in CPU RAM, mirroring the GPU block allocator. The crucial property that bounds swap space: "the number of blocks swapped to the CPU RAM never exceeds the number of total physical blocks in the GPU RAM, so the swap space on the CPU RAM is bounded by the GPU memory allocated for the KV cache." This is because blocks are swapped out only when GPU memory is full, meaning at most all GPU blocks could be swapped to CPU, requiring at most the same total capacity in CPU RAM.

**Recomputation.** Instead of copying the KV cache to CPU RAM, the system simply discards the evicted blocks. When the sequence is resumed, its KV cache is recomputed from scratch. However, recomputation is **significantly faster** than the original generation because the tokens that were already generated (the output tokens) are known. The scheduler concatenates the original prompt with the already-generated output tokens and feeds this combined sequence through the model in a **single prompt-phase forward pass** (which is parallelized across all positions using matrix-matrix multiplication). This means recovering a sequence that generated 200 tokens requires one parallelized encoding pass, not 200 sequential autoregressive steps.

**Tradeoff between swapping and recomputation.** The paper microbenchmarks both methods in §7.3 (Figure 19). The key findings:

- **Swapping latency scales poorly with small block sizes.** When blocks are small (e.g., 4 tokens), swapping generates many small PCIe transfers between GPU and CPU RAM. The aggregate bandwidth across PCIe is limited, and many small transfers incur per-transfer overhead (DMA setup, kernel launch) that dominates the transfer time. With block size 4, swap-in time plus swap-out time exceeds 120 ms.

- **Recomputation latency is independent of block size.** Recomputing the KV cache uses the GPU's computational throughput, not PCIe bandwidth. The time is proportional to the number of tokens being recomputed and the model size, but is constant across different block sizes. The paper reports recomputation overhead is "never higher than 20% of swapping's latency" for the configurations tested.

- **For medium block sizes (16–64), the two methods exhibit comparable end-to-end performance.** At block size 16 (vLLM's default), recomputation and swapping are roughly equivalent in terms of overall throughput impact.

- **Swapping becomes more efficient at large block sizes** because fewer, larger transfers achieve better PCIe bandwidth utilization. At block size 256, swapping outperforms recomputation.

The practical implication: if the deployment has abundant CPU RAM and a fast PCIe interconnect, swapping is viable and avoids redundant computation. If GPU compute is relatively cheap and CPU RAM is limited, recomputation is the better choice. The paper does not declare a universal winner; it exposes the tradeoff and lets the operator choose.

---

#### Distributed Execution with Tensor Model Parallelism

**Why distribution is necessary.** Large language models (e.g., OPT-175B with 346 GB of parameters) far exceed the memory capacity of a single GPU (40–80 GB for an A100). vLLM supports distributed serving by partitioning the model across multiple GPUs using **Megatron-LM style tensor model parallelism** on the attention heads.

**The parallelism strategy.** The model's linear layers are split column-wise or row-wise to perform block-wise matrix multiplication, and intermediate results are synchronized via all-reduce operations. Crucially, the attention operator is split along the **attention head dimension**: if the model has `$A$` attention heads and there are `$W$` GPU workers, each worker processes `$A/W$` attention heads per layer. Each worker thus stores and computes only its assigned heads' key, query, value, and output vectors.

**Why a single centralized KV Cache manager works.** Even though the model is sharded across GPU workers, **all workers process the exact same set of input tokens** for a given request. This means that each worker needs the KV cache for all tokens, but only for its own subset of attention heads. The mapping from logical blocks to physical blocks is therefore **identical across workers** — the block table (which physical block stores the KV data for which logical block) is the same for all GPU workers. What differs is the *contents* of the physical blocks: worker 0's block 7 stores key/value vectors for heads 0–15, while worker 1's block 7 stores key/value vectors for heads 16–31 (assuming 32 total heads and 2 workers), but both store data for the same token positions in the same logical block.

**The per-iteration execution protocol:**

1. The centralized scheduler prepares a control message containing the input token IDs for each sequence in the batch and the block table for each sequence (logical block index → physical block ID → #filled). This control message is identical for all workers.

2. The scheduler **broadcasts** this control message to all GPU workers simultaneously.

3. Each GPU worker executes the model forward pass independently on its shard:
   - For linear layers: the worker's shard of the weight matrix is multiplied with the (replicated) input activations, producing partial outputs that are later combined via all-reduce.
   - For attention layers: the PagedAttention kernel reads the KV cache from the physical blocks specified in the block table, but only for the attention heads assigned to this worker. The kernel computes attention outputs for those heads.
   - Workers synchronize intermediate results using all-reduce communication primitives without coordination from the scheduler (standard Megatron-LM protocol).

4. After the forward pass completes, each worker samples the next token from the output logits (using random number generators synchronized across workers for reproducibility, or by having one worker sample and broadcast the result).

5. Each worker sends its sampled token back to the scheduler.

**Memory management coordination.** GPU workers do not need to communicate about memory management at all. The scheduler is the sole authority on: which physical blocks are allocated to which sequences, which blocks are free, reference counts, and copy-on-write decisions. The workers receive the block table and execute reads/writes exactly as instructed — they are "dumb" executors for memory operations. This centralized design avoids distributed consensus overhead for memory allocation, which would be a significant bottleneck given that memory allocation decisions happen at every iteration.

**Why this design is SPMD-compatible.** The paper notes that Megatron-LM tensor parallelism follows an SPMD (Single Program Multiple Data) execution model. vLLM's design preserves this: all workers run the same program (the model forward pass), but on different data shards (their assigned attention heads). The block table provides the "single program" part with the memory layout, and the attention head partitioning provides the "multiple data" part.

---

#### Kernel-Level Optimizations

The PagedAttention memory layout introduces new memory access patterns that existing GPU kernels do not efficiently support. To mitigate the overhead of block table indirection and non-contiguous memory access, vLLM implements three fused CUDA kernels:

**1. Fused reshape and block write.** At each Transformer layer, after the attention keys and values are computed for the new token, they must be:
- **Split** into blocks (if the sequence spans multiple logical blocks, only the last logical block's relevant portion needs to be written — typically a single new token's vectors).
- **Reshaped** into a memory layout optimized for efficient block reads during future attention computations (likely interleaving key and value vectors, or laying them out to enable coalesced warp-level reads).
- **Written** to the correct physical block and offset, as specified by the block table (the scheduler tells the worker which physical block ID and which slot within that block to write to).

A naive implementation would launch three separate GPU kernels (split, reshape, write), each incurring kernel launch overhead and requiring intermediate data round-trips through GPU memory. The fused kernel combines all three operations into a single launch, keeping intermediate data in registers or shared memory.

**2. Fused block read and attention.** This is the PagedAttention kernel itself. It adapts the attention kernel from FasterTransformer to:
- **Read KV cache** from physical blocks: for each logical block of the sequence, read the block table entry, determine the physical block ID, and load the key and value vectors from that physical block into GPU registers/shared memory.
- **Compute attention** on the fly: multiply the query vector against the loaded key vectors, compute the softmax (with two-pass numerical stability), and weight the value vectors.
- **Support variable sequence lengths** within a batch: different requests in the batch may have different lengths, so the kernel must handle variable numbers of blocks per sequence. This is achieved by processing sequences in parallel with masking for shorter sequences.
- **Use warp-level assignment** for coalesced memory access: each GPU warp (32 threads) is assigned to read one physical block, ensuring that the 32 threads in the warp access contiguous memory locations within that block. This maximizes memory bandwidth utilization.

The paper reports in the ablation (§7.1, Figure 18a) that this kernel has **20–26% higher latency** than FasterTransformer's highly-optimized contiguous attention kernel, due to the extra overhead of block table access and handling variable-length sequences. However, this overhead is limited to the attention operator alone — the feed-forward layers, layer norm, and other operations are identical. Since attention is only a fraction of the total model computation, the end-to-end throughput gain from improved batching (2–4× more requests) far outweighs the 20–26% attention kernel slowdown.

**3. Fused block copy.** The copy-on-write mechanism requires copying the contents of one physical block to another. When many sequences diverge simultaneously (e.g., all parallel samples writing their first generated token at the same iteration), there can be many such block copies. A naive implementation using `cudaMemcpyAsync` for each individual block copy would require one kernel launch per block, creating launch overhead that dominates the actual data transfer time for small blocks.

The fused block copy kernel batches all copy operations across all sequences and all layers into a single kernel launch. The kernel receives a list of (source physical block ID, destination physical block ID) pairs and executes all copies in a single pass, with each GPU thread block handling one or more copy operations. This is particularly important for beam search, where many candidates may simultaneously diverge from shared ancestors.

**Why these kernels are important.** The paper is transparent that the block table indirection adds overhead — the attention kernel is 20–26% slower than FasterTransformer's. Without these fused kernels, the overhead would be larger and might negate the throughput gains from better memory utilization. The kernels are the engineering mechanism that makes the conceptual elegance of paging into a practical performance win: they amortize the indirection cost across the much larger gain of fitting 2–4× more requests into GPU memory.

## 4. Key Insights and Innovations

### Innovation 1: Recognizing KV Cache Management as a Virtual Memory Problem Is a Conceptual Reframing, Not an Incremental Optimization

The paper's most fundamental contribution is not any specific algorithm or kernel, but rather the recognition that the KV cache management problem in LLM serving is structurally isomorphic to the memory management problem that operating systems solved with virtual memory and paging. This is a conceptual reframing, not an incremental improvement on a known technique. Prior to this work, the dominant assumption in LLM serving systems — shared by FasterTransformer, Orca, and the deep learning frameworks they build on — was that tensors must be stored contiguously in GPU memory. This assumption was so deeply embedded that it went unquestioned: PyTorch, TensorFlow, and every major framework require contiguous tensors for efficient operations, and serving systems inherited this requirement without examining whether the KV cache's unique properties (dynamic growth, unknown lifetime, sharing opportunities) made that requirement a poor fit.

The paper does not merely propose a better allocator within the contiguous-tensor paradigm — it rejects the paradigm entirely. By drawing an explicit analogy between tokens and bytes, KV blocks and pages, requests and processes, and block tables and page tables, the paper maps the KV cache problem onto a solution space with decades of proven techniques: on-demand paging, copy-on-write, reference counting, swapping, and all-or-nothing eviction. Each of these is a direct transplant of an OS concept, but the mapping is not trivial — it required identifying which OS abstractions apply, which do not, and where the LLM serving workload demands modifications (e.g., all-or-nothing eviction because generating one token requires all previous tokens' KV cache, a property with no exact OS analog).

This reframing is significant beyond the immediate performance gains because it changes how future researchers and system builders will think about the problem. Before this paper, KV cache management was seen as a tensor-allocation problem — a question of fitting variable-sized contiguous blocks into GPU memory, with compaction as the natural (if impractical) solution. After this paper, it becomes a page-management problem — a question of mapping logical addresses to physical frames, with fragmentation solved by design rather than by expensive defragmentation. This conceptual shift opens an entire design space: block replacement policies (LRU, clock algorithms), multi-level block tables, huge pages (larger blocks for long sequences), and memory compression (zero-page detection for identical KV vectors) all become natural extensions within the paging framework. The paper's specific mechanisms — 16-token blocks, FCFS eviction, swapping vs. recomputation — are instantiations of this framework, but the framework itself is the lasting contribution.

Evidence for the power of this reframing is in Figure 2: vLLM achieves 96.3% KV cache utilization, compared to 20.4–38.2% for contiguous-allocation systems. This order-of-magnitude reduction in waste is characteristic of a paradigm shift, not a point optimization — it comes from eliminating the root cause (contiguous pre-allocation) rather than mitigating its symptoms (better buddy allocators, compaction).

This is a fundamental shift, not an incremental refinement. The contiguous-tensor assumption was not a minor implementation detail; it was the architectural foundation of every prior LLM serving system. The paper's reframing does not improve contiguous allocation — it makes it unnecessary.

---

### Innovation 2: The Block Table Abstraction Decouples Memory Management from Model Execution, Enabling System Composability

The paper introduces a clean separation between *how memory is managed* (block allocation, reference counting, copy-on-write, eviction) and *how the model executes* (attention computation, feed-forward layers, logit generation). This separation is embodied in the block table, which translates logical block indices to physical block IDs and is the sole interface between the KV cache manager and the GPU workers.

Prior systems tightly coupled memory layout with model execution: the attention kernel assumed that the KV cache for a sequence was a single contiguous tensor, which meant that memory allocation decisions (how much to pre-allocate, where to place it) were baked into the execution path. This coupling made it impossible to change the memory management strategy without rewriting the attention kernel, and conversely made it impossible to share memory across sequences without the kernel being aware of the sharing topology.

vLLM's block table breaks this coupling completely. The scheduler broadcasts the block table along with the input tokens at each iteration, and the GPU workers execute the PagedAttention kernel using only physical block IDs. The kernel has no awareness of whether blocks are shared, how they were allocated, or what decoding algorithm produced the sequence — it simply reads from and writes to the specified physical addresses. This means that complex sharing patterns (parallel sampling, beam search with dynamic ancestry, shared prefixes across unrelated requests) are handled entirely by the block manager through reference counting and copy-on-write, with zero changes to the model execution code. The `fork`, `append`, and `free` primitives (§5.2) are sufficient to implement all supported decoding algorithms, and new decoding algorithms can be added by composing these primitives without touching the attention kernel.

This composability is significant for two reasons. First, it reduces system complexity: the block manager's logic for reference counting and eviction is independent of the attention kernel's logic for block-wise dot products and softmax. Each can be optimized, debugged, and tested independently. Second, it enables the system to simultaneously serve requests with heterogeneous decoding algorithms in the same batch — greedy decoding, parallel sampling, and beam search requests are all just sequences with block tables, and the GPU workers process them identically. Prior systems could not do this because each decoding algorithm required different KV cache management logic that was embedded in the serving pipeline. vLLM's page-table abstraction absorbs this diversity, presenting a uniform execution interface.

The evidence for this composability is not a single figure but the architecture itself (Figure 4) and the range of decoding scenarios evaluated (§6.3–6.5). The same system, with the same kernel, achieves throughput improvements across basic sampling, parallel sampling, beam search, shared prefix, and chatbot workloads — each with different sharing patterns — because the block table hides all the differences from the execution engine.

This is a fundamental architectural contribution. The block table is not an optimization on top of existing memory management; it is a layer of indirection that makes the entire system modular. In OS terms, this is the difference between a system where applications manually manage physical memory (pre-virtual-memory) and one where the MMU provides a virtual address space — the latter enables process isolation, shared libraries, and copy-on-write fork, all of which are impossible without the indirection layer.

---

### Innovation 3: Quantifying and Diagnosing KV Cache Waste Reveals That Reservation, Not Fragmentation, Is the Dominant Problem

The paper provides the first systematic quantification of *where* KV cache memory is wasted in LLM serving systems. Figure 2 decomposes memory waste into three categories — internal fragmentation, external fragmentation, and reservation — and measures each for different allocation strategies (Orca Max, Orca Pow2, Orca Oracle, vLLM). This decomposition is a diagnostic contribution that changes the understanding of the problem.

Prior to this paper, the serving systems community understood that KV cache memory was a bottleneck, but the nature of the bottleneck was vague. Is the problem that allocators leave unusable gaps (external fragmentation)? Or that pre-allocated buffers are too large for actual sequences (internal fragmentation)? Or that memory sits idle waiting for future tokens that may never arrive (reservation)? The answer determines the solution: external fragmentation calls for compaction or better allocators; internal fragmentation calls for tighter output-length prediction; reservation calls for dynamic, on-demand allocation. Without quantifying these categories, system builders were optimizing against the wrong target.

The paper's decomposition reveals a surprising result: **reservation is the dominant source of waste**, not fragmentation. For Orca (Max), which reserves the full 2048-token maximum for every request, reservation waste alone accounts for 57.3% of allocated memory — more than internal fragmentation (13.3%) and external fragmentation (8.9%) combined. Even for Orca (Oracle), which unrealistically knows the exact output length in advance and thus has no internal fragmentation, reservation still wastes 38.2% of memory because the pre-allocated chunks prevent sharing of unused slots. This means that even a perfect output-length oracle cannot eliminate the dominant waste source in contiguous-allocation systems — the fundamental problem is the *pre-allocation itself*, not the accuracy of the pre-allocation size.

This insight directly motivates the paging approach. If external fragmentation were the dominant problem, a better allocator (e.g., a slab allocator or a compaction scheme) might suffice. If internal fragmentation were dominant, better length prediction might help. But because reservation is dominant — because the system must reserve memory for tokens that do not yet exist and whose number is unknown — the only solution is to stop reserving memory in advance. PagedAttention achieves this by allocating blocks on demand, one at a time, as tokens are generated. The waste drops to the unfilled portion of a single block (at most `$B-1$` tokens), which for `$B=16$` is negligible.

The significance of this diagnostic contribution extends beyond vLLM. It tells future system builders that any solution that pre-allocates contiguous KV cache — regardless of how clever the allocation algorithm — will hit a ceiling of roughly 35–40% utilization (the Oracle baseline), because reservation is inherent to contiguous pre-allocation. Breaking through that ceiling requires abandoning contiguity, which is what PagedAttention does. This is a lasting conceptual contribution independent of the specific mechanisms.

The evidence is in Figure 2, which is perhaps the most important figure in the paper for understanding *why* the approach works. The 96.3% utilization of vLLM is not a magic number — it is the direct consequence of eliminating reservation waste, which Figure 2 shows to be the dominant term.

---

### Innovation 4: Copy-on-Write at Block Granularity Makes Memory Sharing Practical Across All Decoding Algorithms Without Per-Algorithm Engineering

Memory sharing across sequences is not a new idea — practitioners have long recognized that parallel samples share the same prompt and that beam search candidates share common ancestors. The paper's contribution is showing that a single, general-purpose mechanism — block-granularity copy-on-write with reference counting — can implement sharing for *all* these decoding algorithms without any algorithm-specific memory management code.

Prior systems handled sharing either not at all (each sequence got a full independent copy of the KV cache, as in FasterTransformer and Orca) or through bespoke, algorithm-specific logic (e.g., custom beam search implementations that manually manage which parts of the KV cache to copy when beams diverge). These bespoke solutions were fragile (tied to one decoding algorithm), complex (required understanding the algorithm's sharing topology), and inefficient (frequent full-cache copies when beams reorganized, as described in Section 4.4).

vLLM's innovation is recognizing that all these sharing patterns are instances of a single abstract operation: multiple sequences need to share a prefix of their logical KV cache, diverging only when they generate different tokens at specific positions. The block table, combined with reference counting and copy-on-write, handles this uniformly. The `fork` operation creates a new sequence that shares all physical blocks with its parent by incrementing reference counts — this works whether the parent is a prompt (parallel sampling), a beam candidate (beam search), or a pre-computed system prompt (shared prefix). The copy-on-write mechanism triggers automatically when any sequence writes to a shared block, allocating a private copy only when and where divergence actually occurs.

What makes this significant is not the mechanism itself (copy-on-write is a standard OS technique) but the recognition that the KV cache's access pattern — append-only writes to the tail, reads spanning the entire sequence — makes copy-on-write particularly effective. The shared prefix blocks (the prompt, the early beam ancestors) are never modified after their initial generation; they are only read. Copy-on-write therefore triggers exclusively on the "divergence boundary" — the last shared block that also contains newly generated tokens that differ across sequences. For parallel sampling with a long prompt, this is a single block out of dozens. For beam search, the sharing topology is more complex (Figure 9), but the mechanism remains the same: reference counts track sharing, copy-on-write handles divergence, and the kernel never needs to know.

The evidence for generality is the range of workloads evaluated in §6.3–6.5: parallel sampling (Figure 14a–c, Figure 15a), beam search (Figure 14d–f, Figure 15b), shared prefix (Figure 16), and chatbot (Figure 17) all use the same `fork`/`append`/`free` primitives. The memory savings range from 6.1% (parallel sampling with short Alpaca prompts) to 66.3% (beam search with ShareGPT), but the mechanism does not change — the savings are an emergent property of the sharing pattern, not hand-tuned per algorithm.

This is a fundamental contribution in the systems sense: it replaces `$N$` algorithm-specific sharing implementations with one general-purpose mechanism. It is the block table abstraction (Innovation 2) that makes this possible — because the kernel is oblivious to sharing, the block manager can implement arbitrarily complex sharing topologies without modifying the execution path.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses two datasets: **ShareGPT** (a collection of user-shared conversations with ChatGPT) and **Alpaca** (an instruction dataset generated by GPT-3.5 with self-instruct). Neither dataset includes timestamps, so request arrival times are synthesized using Poisson distributions with varying request rates. As shown in Figure 11, ShareGPT has substantially longer sequences: mean input length 161.3 tokens vs. 19.3 for Alpaca, and mean output length 338.0 tokens vs. 58.5 for Alpaca — an 8.4× and 5.8× difference respectively, with higher variance in ShareGPT.

- **Base model(s).** The evaluation uses **OPT models at 13B, 66B, and 175B parameters** and **LLaMA-13B** for the shared prefix experiment. OPT-13B and 66B are chosen because they represent popular sizes on the LLM leaderboard; OPT-175B matches the scale of GPT-3; LLaMA-13B is multilingual and used for the translation workload. Server configurations scale with model size: a single A100 40GB for OPT-13B, four A100 40GB GPUs for OPT-66B (160 GB total), and eight A100-80GB GPUs for OPT-175B (640 GB total), as detailed in Table 1.

- **Metrics.** The primary metric is **normalized latency**, defined as the mean of every request's end-to-end latency divided by its output length, following Orca. A higher-throughput system sustains lower normalized latency at higher request rates. The paper also reports **number of batched requests** (average count of requests processed simultaneously) and **memory savings** (percentage of KV blocks saved by sharing, computed as the number of blocks shared divided by the total blocks that would be needed without sharing).

- **Baselines.** The paper compares against two primary systems: **FasterTransformer** (a distributed inference engine optimized for latency, with a custom scheduler implementing dynamic batching at the request level — the scheduler takes up to a maximum batch size of the earliest-arrived requests) and **Orca** (the state-of-the-art throughput-optimized system with iteration-level scheduling, reimplemented by the authors since Orca is not publicly available). Three Orca variants are tested based on how much output space is over-reserved: **Orca (Oracle)** — unrealistically knows exact output lengths, representing the upper bound; **Orca (Pow2)** — over-reserves by at most 2× (e.g., 32 slots for a 25-token output); and **Orca (Max)** — always reserves the maximum sequence length of 2048 tokens.

- **Generation budget / compute accounting.** The paper does not measure inference computation in FLOPs or token-generations. Instead, the experiments compare systems at equal **request rates** (requests per second, following a Poisson arrival process) and measure the resulting normalized latency. Throughput is implicitly compared by identifying the maximum request rate each system can sustain before latency explodes (when queue length grows unbounded). For memory savings experiments, the metric is the **number of KV blocks** allocated, which is a direct measure of memory consumption since all blocks are the same size.

- **Cross-validation / statistical protocol.** The paper uses **1-hour traces** for most experiments and **15-minute traces** for OPT-175B (due to cost constraints). Latency measurements are averaged over all requests in the trace. Statistical significance testing and confidence intervals are not reported. The memory waste profiling in Figure 2 is described as "average percentage of memory wastes ... during the experiment in §6.2."

---

### Main Quantitative Results

#### Basic Sampling Throughput (Single Sequence per Request)

The first row of Figure 12 shows results on ShareGPT across all three OPT model sizes, and the second row shows results on Alpaca. The headline finding is that **vLLM sustains 1.7×–2.7× higher request rates than Orca (Oracle)** and **2.7×–8× higher than Orca (Max)** on ShareGPT while maintaining similar latencies, and **up to 22× higher request rates than FasterTransformer**.

On **ShareGPT with OPT-13B** (Figure 12a, single A100): vLLM sustains approximately 1.7–1.8 requests/second before latency spikes, compared to roughly 0.7 req/s for Orca (Oracle), 0.4 req/s for Orca (Pow2), 0.2 req/s for Orca (Max), and well below 0.1 req/s for FasterTransformer. This is a 2.4× improvement over the best Orca baseline. The batched request count in Figure 13a tells the same story: vLLM averages 30.4 requests in the batch (at 2 req/s), vs. 13.6 for Orca (Oracle), 9.8 for Orca (Pow2), and 7.0 for Orca (Max) — a 2.2× increase over Oracle.

On **ShareGPT with OPT-66B** (Figure 12b, 4× A100): The gap widens. vLLM sustains roughly 0.9 req/s vs. 0.35 req/s for Orca (Oracle) — a 2.6× improvement — and even larger gaps over Orca (Pow2) and Orca (Max). The maximum request rate FasterTransformer can handle is negligible on this scale.

On **ShareGPT with OPT-175B** (Figure 12c, 8× A100-80GB): vLLM sustains roughly 0.8–0.9 req/s, vs. roughly 0.3 req/s for Orca (Oracle) — approximately 2.7× improvement. Orca (Max) cannot sustain any meaningful rate.

On **Alpaca** (Figure 12d–f), the trends are similar but with one notable exception: for OPT-175B on Alpaca (Figure 12f), vLLM's advantage over Orca (Oracle) and Orca (Pow2) is **less pronounced**. The paper explains this explicitly: "This is because the model and server configuration for OPT-175B allows for large GPU memory space available to store KV cache, while the Alpaca dataset has short sequences. In this setup, Orca (Oracle) and Orca (Pow2) can also batch a large number of requests despite the inefficiencies in their memory management. As a result, the performance of the systems becomes compute-bound rather than memory-bound." This is a crucial conditional result: **vLLM's advantage shrinks when the workload is compute-bound rather than memory-bound**. The memory efficiency gains only translate to throughput when memory is the bottleneck.

Figure 13b quantifies this for OPT-13B on Alpaca (at 30 req/s): vLLM batches 132.4 requests on average, vs. 72.8 for Orca (Oracle) — a 1.8× improvement, smaller than the 2.2× seen on ShareGPT.

**FasterTransformer's poor performance** is attributed to two factors: (1) it uses request-level batching rather than iteration-level scheduling, so completed requests cannot be replaced until the entire batch finishes, and (2) it uses contiguous KV cache allocation like Orca (Max), inheriting the full reservation and fragmentation waste. The 22× gap on ShareGPT with OPT-13B is the largest reported improvement.

---

#### Parallel Sampling Throughput

Figure 14 (first row) shows results for parallel sampling on OPT-13B with the Alpaca dataset, varying the number of parallel sequences per request (2, 4, 6). The headline finding: **vLLM's advantage over Orca grows with the number of parallel sequences**, because the memory savings from prompt KV cache sharing compound.

At **parallel size 2** (Figure 14a): vLLM sustains roughly 12 req/s vs. roughly 7 req/s for Orca (Oracle) — approximately 1.7×.

At **parallel size 4** (Figure 14b): vLLM sustains roughly 8 req/s vs. roughly 3.5 req/s for Orca (Oracle) — approximately 2.3×.

At **parallel size 6** (Figure 14c): vLLM sustains roughly 7 req/s vs. roughly 2.5 req/s for Orca (Oracle) — approximately 2.8×.

The increasing gap confirms that vLLM's memory sharing becomes more valuable as the number of sequences per request grows. Orca must store a full independent KV cache for each parallel sample, duplicating the entire prompt KV cache. vLLM shares the prompt's physical blocks across all samples, only copying the last block when sequences diverge. The absolute maximum request rate drops for all systems as parallel size increases (more output sequences means more total tokens generated per request, consuming more compute and memory), but vLLM degrades much more gracefully.

**Memory savings from sharing** are quantified in Figure 15a: 6.1% for parallel size 2, 8.5% for size 4, and 9.8% for size 6 on the Alpaca dataset. The paper notes that on ShareGPT — with its much longer prompts — the savings are 16.2–30.5%. This reveals that the memory sharing benefit is proportional to prompt length: longer prompts mean more shared blocks, and since the prompt constitutes a larger fraction of total KV cache, the savings are larger. The Alpaca dataset's short prompts (mean 19.3 tokens, roughly one or two blocks) limit the sharing opportunity, yet vLLM still achieves meaningful gains.

---

#### Beam Search Throughput

Figure 14 (second row) shows results for beam search on OPT-13B with the Alpaca dataset, varying beam width (2, 4, 6). The headline finding: **vLLM's advantage is even larger for beam search than for parallel sampling**, because beam search involves more extensive memory sharing.

At **beam width 2** (Figure 14d): vLLM sustains roughly 13–14 req/s vs. roughly 7–8 req/s for Orca (Oracle) — approximately 1.8×.

At **beam width 4** (Figure 14e): vLLM sustains roughly 9 req/s vs. roughly 3.5 req/s for Orca (Oracle) — approximately 2.6×.

At **beam width 6** (Figure 14f): vLLM sustains roughly 7 req/s vs. roughly 3 req/s for Orca (Oracle) — approximately 2.3×. The paper states this is 2.3× over Orca (Oracle), compared to 1.3× in basic sampling — a substantial increase.

Figure 15b quantifies the memory savings from sharing in beam search: **37.6% for beam width 2, 53.1% for width 4, and 55.2% for width 6** on the Alpaca dataset. On ShareGPT, the savings are even larger: 44.3–66.3%. These are the largest memory savings reported in the paper, reflecting the deep sharing that beam search enables — candidates share not only the prompt but also intermediate generations along their common ancestry paths in the search tree (as illustrated in Figure 9).

An important subtlety: Orca (Oracle) shows more variability in beam search results than in basic sampling. In several beam configurations, Orca (Pow2) outperforms Orca (Oracle) at certain request rates. The paper does not comment on this specifically, but it suggests that the buddy allocation algorithm in Orca may interact unpredictably with beam search's dynamic memory allocation patterns — when beams reorganize and candidates are discarded, the buddy allocator may leave memory fragmentation that even oracle output-length knowledge cannot prevent.

---

#### Shared Prefix Throughput

Figure 16 evaluates vLLM against Orca (Oracle) on a machine translation workload using LLaMA-13B and the WMT16 English-to-German dataset. The prompts share a common prefix consisting of a translation instruction and few-shot examples. Two prefix configurations are tested: **one-shot** (1 example, 80 tokens) and **few-shot** (5 examples, 341 tokens).

For the **one-shot prefix** (Figure 16a): vLLM sustains roughly 35–40 req/s vs. roughly 20–25 req/s for Orca (Oracle) — approximately **1.67× higher throughput**.

For the **few-shot prefix** (Figure 16b): vLLM sustains roughly 35 req/s vs. roughly 10 req/s for Orca (Oracle) — approximately **3.58× higher throughput**.

The improvement grows with prefix length because longer prefixes mean more KV cache that must be duplicated in Orca but is shared in vLLM. The 5-example prefix (341 tokens) occupies roughly 5–6 blocks that are identical across all requests. In Orca, each request independently computes and stores the full prefix's KV cache — meaning N requests require N copies. In vLLM, the prefix's physical blocks are stored once and mapped into every request's block table, with the last block copy-on-write for divergence. The computation savings are equally important: the prefix is encoded once, not re-encoded for each request. The paper notes that "the prompt phase computation only needs to execute on the user's task input" — the model never reprocesses the prefix tokens.

**Why only Orca (Oracle) is compared.** The paper does not provide results for Orca (Pow2) or Orca (Max) on the shared prefix experiment. This is likely because the prefix sharing experiment tests a different capability — computational reuse, not just memory efficiency — and Oracle represents the most favorable memory configuration for Orca. If Orca (Oracle) cannot match vLLM, the other Orca variants would fare worse.

---

#### Chatbot Workload

Figure 17 evaluates vLLM against all three Orca variants on a chatbot workload using OPT-13B and the ShareGPT dataset. Chatbot requests are constructed by concatenating conversation history (from ShareGPT) with the last user query into a prompt, truncated to the last 1024 tokens, with the model generating at most 1024 tokens.

vLLM sustains roughly **0.57 req/s** vs. roughly **0.28 req/s** for all three Orca variants — approximately **2× higher throughput**. Notably, the three Orca baselines behave nearly identically. The paper explains: "Since the ShareGPT dataset contains many long conversations, the input prompts for most requests have 1024 tokens. Due to the buddy allocation algorithm, the Orca baselines reserve the space for 1024 tokens for the request outputs, regardless of how they predict the output lengths." In other words, because prompts are already at or near the 1024-token context limit, the difference between reserving 1024 (Max), 2048 (Pow2, since 1024 → next power of 2), or the exact length (Oracle) is small — all three must allocate similar amounts. vLLM's advantage comes from avoiding pre-allocation entirely: it allocates blocks on demand as the output is generated, rather than reserving space for up to 1024 output tokens upfront.

This also reveals an interesting edge case in the Orca baseline: when prompt lengths are large relative to the maximum output length, all allocation strategies converge to similar behavior because the prompt KV cache dominates memory consumption and there is less room for output-length prediction to matter.

---

### Ablation Studies and Robustness Checks

**PagedAttention kernel latency vs. FasterTransformer:** Figure 18a shows the latency of vLLM's attention kernel compared to FasterTransformer's highly optimized contiguous kernel, measured for two batch sizes (8 and 32) and varying context lengths (64, 128, 256). Across all configurations, vLLM's kernel is **20–26% slower** than FasterTransformer's. At context length 256 with batch size 8, vLLM takes roughly 230 μs vs. 175 μs for FasterTransformer; with batch size 32, roughly 225 μs vs. 175 μs. This overhead comes from block table access, extra branching for variable sequence lengths, and non-contiguous memory reads. The paper argues this is acceptable because attention is only one operator in the Transformer model (feed-forward layers, layer norm, and other components are unaffected), and the throughput gain from better batching (2–4×) outweighs the per-kernel slowdown.

**Impact of block size on end-to-end performance:** Figure 18b sweeps block sizes from 1 to 256 on both ShareGPT and Alpaca traces at fixed request rates. On ShareGPT, block sizes 16 through 128 achieve the best performance (lowest normalized latency), with block sizes 1–4 and 256 performing worse. On Alpaca, block sizes 16 and 32 perform well, but block sizes 64 and larger significantly degrade performance because "the sequences become shorter than the block sizes." For block size 64, a 58-token average output means the last block has only a fraction of its slots used, and for block size 256, many sequences fit entirely within one block but waste most of its capacity. The default block size of 16 is empirically justified as "large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads."

**Recomputation vs. swapping for preemption recovery:** Figure 19a microbenchmarks the overhead of swapping (GPU→CPU and CPU→GPU) and recomputation across block sizes from 1 to 256. Swapping latency decreases with block size: from over 120 ms at block size 1 to roughly 10 ms at block size 256 (for swap-in plus swap-out). Recomputation latency is constant at roughly 4–5 ms regardless of block size. The paper explains that small blocks cause many small PCIe transfers, limiting effective bandwidth. Figure 19b shows end-to-end performance with OPT-13B on ShareGPT: for medium block sizes 16–64, recomputation and swapping achieve comparable normalized latency (roughly 1.2–1.4 s/token). For very small blocks (1–4), recomputation is better; for very large blocks (128–256), swapping is slightly better. Recomputation's overhead is noted to be "never higher than 20% of swapping's latency."

**Memory waste decomposition (Figure 2):** This is both a diagnostic contribution (discussed in Innovation 3) and an ablation that validates the paging design. vLLM achieves 96.3% KV cache utilization — only 3.7% waste (from the unfilled last block and a small "other" category). This compares to 20.4% for Orca (Max), 13.3% for Orca (Pow2), and 36.6% for Orca (Oracle). The ordering is informative: Orca (Pow2) wastes more than Orca (Oracle) because its 2× over-reservation creates larger reservation waste, and Orca (Max) wastes the most because it always reserves 2048 tokens. The fact that even Oracle — with perfect output-length knowledge — wastes 63.4% of memory (reservation 38.2% + external fragmentation 25.2%) is the strongest evidence that contiguous allocation cannot solve the problem.

**Model scale sweep (Figure 12):** The experiments test three model scales (13B, 66B, 175B) and show that vLLM's advantage is consistent, and in some cases grows with model size. The improvement over Orca (Oracle) goes from 2.4× (13B) to 2.6× (66B) to 2.7× (175B) on ShareGPT. This suggests that larger models — which leave less free memory for KV cache after storing parameters — benefit more from efficient memory management. However, the Alpaca result with OPT-175B (Figure 12f) shows a diminished advantage, establishing that model scale alone does not guarantee improvement; the workload's sequence length distribution matters equally.

**Dataset effect (ShareGPT vs. Alpaca):** The two datasets produce qualitatively different results. ShareGPT (long sequences, high variance) shows larger vLLM advantages because memory is the bottleneck. Alpaca (short sequences, low variance) shows smaller advantages and, at large model scales, can become compute-bound. This is not an ablation per se, but it demonstrates that the workload characteristics determine whether memory management is the binding constraint and therefore whether vLLM's techniques translate to throughput.

**Decoding algorithm sweep (Figure 14):** Testing basic sampling, parallel sampling (2/4/6), and beam search (2/4/6) on the same model and dataset shows that vLLM's advantage is not specific to one decoding method. The improvement over Orca (Oracle) grows from 1.3× (basic sampling) to 2.3× (beam search width 6). This validates the claim that the block-table abstraction generalizes across decoding algorithms — the same kernel, the same memory manager, and the same primitives handle all cases without algorithm-specific code.

---

### Critical Assessment

**Do the experiments support the claim that vLLM improves throughput by 2–4× over state-of-the-art systems?**

The claim is supported, but with important boundary conditions that the paper itself documents. The 2–4× range is not a single number; it varies substantially by configuration:

- The **2×** end of the range occurs in configurations where the workload is less memory-bound: chatbot with OPT-13B (Figure 17, 2× over Orca baselines), Alpaca with OPT-175B (Figure 12f, where the advantage diminishes because the system becomes compute-bound), and parallel sampling with small parallel sizes.
- The **4×+** end occurs in configurations where memory is the clear bottleneck: ShareGPT with OPT-13B (Figure 12a, up to 8× over Orca Max, though only 2.4× over Oracle), OPT-66B and 175B on ShareGPT (2.6–2.7× over Oracle), and beam search with width 6 (2.3× over Oracle).
- The **22×** claim over FasterTransformer (Figure 12a) is technically accurate but reflects FasterTransformer's lack of iteration-level scheduling more than PagedAttention's memory efficiency alone. It is an apples-to-oranges comparison against a weaker scheduler, not a head-to-head comparison of memory management.

The claim is therefore **supported with significant variance** across configurations. The 2–4× figure is a reasonable summary, but a practitioner reading the paper should understand that the realized improvement in their deployment will depend on model size, sequence length distribution, and decoding algorithm complexity.

**A critical nuance: the improvement is relative to Orca (Oracle), not Orca (Max).** The paper benchmarks against three Orca variants, and the Oracle variant represents an **unrealizable upper bound** that knows exact output lengths in advance. The paper's claimed improvements over state-of-the-art systems should properly be measured against Orca (Max) or Orca (Pow2), since those are the actually-deployable configurations. Against Orca (Max), vLLM achieves 4.3× more batched requests for ShareGPT (Figure 13a: 30.4 vs. 7.0) and 2.7–8× higher sustainable request rates. Against Orca (Oracle), the improvement is typically 1.3–2.7×. This distinction matters because it affects how the results should be communicated. The 2–4× headline figure appears to reference the Oracle comparison; the gains against the realistic baselines are substantially larger.

**Do the experiments support the claim that improvements are "more pronounced with longer sequences, larger models, and more complex decoding algorithms"?**

**Longer sequences:** Strongly supported. ShareGPT (mean output 338 tokens) shows consistently larger vLLM advantages than Alpaca (mean output 58 tokens). The shared prefix experiment (Figure 16) explicitly shows that doubling the prefix length roughly doubles the improvement factor. The batched request counts (Figure 13) show 2.2× improvement on ShareGPT vs. 1.8× on Alpaca.

**Larger models:** Supported with a qualification. The improvement over Orca (Oracle) grows from 2.4× (13B) to 2.6× (66B) to 2.7× (175B) on ShareGPT. However, the OPT-175B on Alpaca result (Figure 12f) shows that larger models do not guarantee larger improvements — when memory is abundant relative to sequence lengths, the advantage shrinks regardless of model size. The claim is better understood as "improvements are more pronounced when the KV cache consumes a larger fraction of available memory," which correlates with model size but is not solely determined by it.

**More complex decoding algorithms:** Strongly supported. The paper systematically sweeps basic sampling, parallel sampling (2/4/6), and beam search (2/4/6), and the improvement over Orca (Oracle) grows from 1.3× to 2.3× (on Alpaca with OPT-13B) as decoding complexity increases. The memory savings figures (Figure 15) directly quantify the mechanism: 6–10% for parallel sampling, 38–55% for beam search. However, the improvement is not monotonic and unbounded — beam width 6 shows a slightly smaller improvement than beam width 4 in some configurations (Figure 14e vs. 14f), possibly because at very high beam widths, the computation cost of evaluating k × |V| candidates dominates, making the workload less memory-bound.

**What experiments would have strengthened the paper but were not run?**

1. **Latency percentiles, not just means.** All reported metrics are mean normalized latency. LLM serving is a latency-sensitive online service — tail latency (p95, p99) matters greatly for user experience. The paper does not report tail latency distributions. It is possible that vLLM's dynamic block allocation introduces occasional allocation stalls (when many sequences simultaneously need new blocks during a burst) that increase tail latency even if mean latency improves.

2. **Direct comparison to a compaction-based approach.** The paper argues that compaction is impractical for KV cache (§3.1), but this claim is not empirically tested. A system that uses contiguous allocation with periodic compaction (e.g., during idle GPU cycles or at batch boundaries) could potentially recover some of the fragmentation waste without the overhead of block table indirection. The absence of this baseline means we cannot quantify the exact contribution of paging vs. what simpler defragmentation might achieve.

3. **GPU memory capacity as an explicit independent variable.** The experiments use fixed GPU configurations (A100 40GB and 80GB). Artificially limiting GPU memory (e.g., allocating only 20GB for KV cache on a 40GB card) would test whether vLLM's advantage grows as memory pressure increases, which the paper's model predicts. This would strengthen the "increasingly significant bottleneck" argument (§3).

4. **More model families.** All experiments use OPT models (except LLaMA-13B for the shared prefix experiment). Testing on GPT-style models (with different attention implementations, different layer counts, different hidden dimensions) would establish whether PagedAttention's advantages are architecture-agnostic. The paper claims vLLM "supports popular LLMs such as GPT, OPT, and LLaMA" (§5), but only OPT is benchmarked for most experiments.

5. **Throughput at iso-latency rather than latency at iso-request-rate.** The experiments vary request rate and measure normalized latency. A complementary experiment would fix a latency SLO (e.g., 1 second per token) and measure maximum throughput. This is more directly relevant to service providers who have latency guarantees.

6. **Ablation isolating the contribution of iteration-level scheduling vs. memory management.** vLLM uses both PagedAttention (for memory) and iteration-level scheduling (inherited from Orca). The paper's comparison against FasterTransformer conflates these: FasterTransformer lacks both, so the 22× improvement is not solely due to memory management. A vLLM variant with contiguous allocation but iteration-level scheduling would isolate the memory management contribution, while a vLLM variant with PagedAttention but request-level scheduling would isolate the scheduling contribution. Neither is tested.

7. **CPU memory bandwidth measurements for swapping.** The swapping experiments (Figure 19) report latency but not PCIe bandwidth utilization. Knowing the achieved bandwidth would help practitioners estimate whether their hardware (with potentially different PCIe generations and CPU-GPU interconnects) would see similar results.

**Does the experimental methodology match real-world serving conditions?**

There are several gaps between the experimental setup and production LLM serving:

- **Synthetic arrival times.** Both datasets lack real timestamps, so arrivals are synthesized using Poisson processes. Real-world request patterns may be bursty (correlated arrivals, diurnal patterns), and vLLM's memory management might behave differently under burst loads where many requests arrive simultaneously, exhausting physical blocks quickly and triggering more frequent preemption.

- **No conversation-round KV cache reuse.** The chatbot experiment (Figure 17) explicitly does not store KV cache between conversation rounds, noting "doing this would occupy the space for other requests between the conversation rounds." Real chatbot services might want to cache conversation history to avoid recomputing it when the user sends follow-up messages. The paper does not explore this tradeoff.

- **Fixed model parallelism configuration.** The distributed experiments use a fixed number of GPUs per model size (Table 1). In practice, operators might choose different degrees of tensor parallelism depending on the batch size, which changes the memory-per-GPU dynamics.

- **No batching of heterogeneous decoding algorithms.** While the paper claims vLLM supports mixed decoding methods within a batch (§4.4, Mixed decoding methods), this claim is not experimentally validated. All experiments use a single decoding algorithm per trace. A workload mixing greedy decoding, beam search, and parallel sampling requests would test the composability claim directly.

**The "near-zero waste" claim deserves scrutiny.** Figure 2 shows vLLM achieves 96.3% KV cache utilization. The remaining 3.7% is attributed to "internal fragmentation & others" — presumably the unfilled slots in the last block of each sequence, plus block table overhead. For the default block size of 16, the expected waste from the last block is at most 15 tokens per sequence, which is indeed near-zero. However, this measurement is taken "during the experiment in §6.2" — a specific configuration. The waste percentage could be higher in configurations with many very short sequences (where the ratio of wasted last-block tokens to total tokens is higher) or with very large block sizes. The "near-zero" characterization is fair for the evaluated workloads but is not a universal guarantee.

## 6. Limitations and Trade-offs

### Difficulty Estimation Requires 2048 Samples Per Question — Making the Headline 4× Efficiency Gain Conditional on a Cost the Paper Does Not Account For

**The assumption or constraint.** The entire compute-optimal strategy selection depends on knowing each prompt's difficulty quintile before allocating the inference budget. The paper's method for estimating difficulty — whether using oracle correctness or predicted (PRM-based) scores — requires **generating 2048 samples per question** and averaging either the pass@1 rate or the PRM's final-answer score distribution. The authors are explicit about this in Section 3.2:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This is not an implementation detail — it is a fundamental prerequisite without which the compute-optimal policy cannot be selected. The paper emits the headline claim of 4× efficiency improvement relative to best-of-N, but this improvement is computed **after difficulty is known**, without amortizing the one-time cost of learning it against the per-query savings.

**The consequence.** If the difficulty estimation cost is included, the total compute budget per question is (2048 × estimation samples) + (N generations for the selected strategy). For N = 256 (the largest budget studied), the estimation cost is 8× larger than the strategy execution cost. For N = 64 (where much of the 4× improvement is demonstrated), estimation cost is 32× larger. This does not mean the approach is useless — if difficulty is estimated once and cached for many similar queries (e.g., in a system that receives repeated questions), the amortized cost drops. But for single-use queries or prompt distributions with long tails of unique questions, the unamortized cost overwhelms any per-query savings.

The paper acknowledges this gap and flags it as future work:

> "an important avenue for future work is to develop methods that that estimate the difficulty efficiently, without having to generate a large number of samples per question"

But this is not a minor optimization opportunity — it is the blocking issue that stands between the paper's empirical demonstration and practical deployment. Without a cheap difficulty estimator, the compute-optimal framework is an analytical contribution (showing what is possible in principle) rather than a deployable system.

**What evidence exists in the paper.** The paper provides no measurement of the amortization tradeoff — no experiment where the estimation cost is accounted for, no analysis of how many queries must share the same difficulty estimate to break even, and no comparison of the total compute (estimation + execution) against a uniform-allocation baseline. The difficulty estimation mechanism itself is evaluated only in terms of predictive accuracy (Figure 4 and 8 show predicted bins tracking oracle bins), not in terms of its cost-effectiveness relative to simpler heuristics (e.g., using a single forward pass to estimate difficulty, or using the prompt's length or format as a cheap proxy).

**Mitigation status.** The paper does not attempt to reduce the estimation cost. It suggests future work on "pretraining or finetuning models to directly predict difficulty of a question" (Section 8), but no such model is developed, trained, or evaluated. The exploration-exploitation framing in Section 3.2 — "compute spent assessing difficulty versus compute spent solving the problem" — acknowledges the tradeoff conceptually but does not quantify it or propose a specific mechanism for balancing it. A practitioner reading the paper would not know what minimum number of samples is necessary for acceptable difficulty estimation accuracy, nor how the accuracy of the difficulty estimate degrades as the sample count is reduced.

This is the paper's most consequential practical limitation. The 4× efficiency gain is real and well-supported on a per-query basis conditional on known difficulty, but the cost of acquiring that knowledge is large and unaccounted for. The paper's contribution is primarily analytical — demonstrating that adaptive allocation can recover large efficiency gains — but the mechanism for realizing those gains in a cost-effective deployment remains an open problem.

---

### All Results Come from a Single Benchmark (MATH) and a Single Model Family (PaLM 2-S*) — the Difficulty-Dependent Patterns May Be Specific to This Combination

**The assumption or constraint.** Every experiment in the paper uses the MATH benchmark (500 test questions, high-school competition mathematics problems) and the PaLM 2-S* (Codey) model family. The authors state in Section 4 that they "believe this model is representative of the capabilities of many contemporary LLMs," but this belief is not empirically validated against other model families, architectures, or scale points.

The combination of MATH + PaLM 2-S* produces a specific difficulty distribution: ~10–19% base pass@1, with a substantial fraction of problems in the "medium" difficulty range where test-time compute provides the largest benefits. The paper's central results — beam search degrades on easy problems (bin 1–2), helps on medium problems (bin 3–4), and does nothing on hard problems (bin 5) — are a function of both the model's capability profile and the benchmark's difficulty distribution. A different model (e.g., one with higher base accuracy, or one with different error patterns) on a different benchmark (e.g., code generation, where partial correctness matters) might exhibit qualitatively different difficulty-dependent scaling curves.

**The consequence.** Three specific generalisation risks arise:

**Model family dependence.** PaLM 2-S* may have specific properties — its output distribution calibration, its tendency to produce certain types of errors, the quality of its in-context learning — that influence the difficulty estimation and the effectiveness of different strategies. A model with better calibration might exhibit less PRM over-optimization (changing the easy-problem degradation pattern in Figure 3). A model with weaker in-context learning might not benefit from sequential revisions in the same way, since the revision model's training procedure relies on the base model's ability to condition on previous incorrect answers.

**Benchmark dependence.** MATH consists of competition-level math problems with exact, checkable answers. This enables both the PRM training pipeline (Monte Carlo rollout correctness requires a binary correctness signal) and the difficulty estimation (pass@1 requires knowing which answers are right). Many real-world LLM applications — code generation (where correctness is partial or requires execution), open-ended dialogue, summarization (where quality is multi-dimensional), creative writing — lack such clean binary signals. The paper provides no evidence that the difficulty-dependent strategy selection framework transfers to domains where correctness is ambiguous, and the PRM training procedure is directly tied to the availability of ground-truth answer checking.

**Scale dependence.** PaLM 2-S* sits at a specific capability level (roughly 10–19% pass@1). At this level, there are enough incorrect answers to train a revision model (the training procedure requires incorrect-to-correct trajectories) and enough near-correct answers for revisions to be useful. A much more capable model (e.g., 60%+ pass@1) might see diminishing returns from revisions (most answers are already correct) and might need a different balance of search vs. revision. A much weaker model (e.g., <5% pass@1) might have so few correct trajectories that neither search nor revision can help on most problems, similar to difficulty bin 5 in the current results.

**What evidence exists in the paper.** The paper provides no cross-model or cross-benchmark experiments. The ablation on the ReST^EM-trained revision model (Appendix K, Figure 16) shows that changing the training procedure for the same base model can significantly alter the effectiveness of revisions (making sequential revisions hurt rather than help), which hints at the sensitivity of the results to model and training choices. But this is a within-model-family variation, not a test of generalisation to other architectures or pretraining procedures.

The paper's analysis of difficulty bins is correlational within the PaLM 2-S* + MATH setting. There is no theoretical model predicting how the difficulty-dependent patterns would change as model capability or benchmark characteristics vary — the five difficulty quintiles are an empirical partition of one specific model-benchmark combination, not a principled difficulty taxonomy that would transfer to new settings.

**Mitigation status.** The paper does not address generalisation. It acknowledges the single-model limitation implicitly by stating the belief that PaLM 2-S* is "representative," but makes no attempt to validate this claim. Section 8 suggests "applying our framework to other tasks and domains" as future work, but does not specify which aspects of the results are expected to transfer (the difficulty-dependent phenomenon in general?) vs. which are specific (the exact difficulty thresholds? the optimal strategy per bin?).

A practitioner deploying a different model on a different task would not know whether to expect the same qualitative patterns (beam search helps on medium difficulty, revisions help on easy problems) or whether the optimal strategies would need to be recomputed entirely. The paper demonstrates that compute-optimal allocation works for PaLM 2-S* on MATH, but does not provide the principles or theory to extend the approach to new settings without re-running the full analysis pipeline.

---

### The 14× Larger Model Baseline Is Not Compute-Optimally Trained and Uses No Test-Time Compute of Its Own — Making the Pretraining-vs.-Inference Comparison Asymmetric

**The assumption or constraint.** In the FLOPs-matched comparison (Section 7), the paper compares PaLM 2-S* with compute-optimal test-time scaling against a model with approximately 14× more parameters, using greedy decoding and no test-time compute augmentation. The authors explicitly acknowledge two ways in which this baseline is weaker than it could be:

First, the larger model follows the LLaMA scaling paradigm (scale parameters, fix data), not Chinchilla-optimal scaling (scale both parameters and data equally):

> "We choose this setting as it is representative of a canonical approach to scaling pretraining compute and leave the analysis of compute-optimal scaling of pretraining compute where the data and parameters are both scaled equally to future work." (Section 7)

Second, the larger model uses only greedy decoding — no best-of-N, no PRM search, no revisions, no majority voting. This means the comparison is between (small model + sophisticated inference) vs. (large model + naive inference), which conflates the pretraining-vs.-inference tradeoff with the sophisticated-vs.-naive inference tradeoff. A fairer comparison would allocate some of the larger model's FLOPs budget to test-time compute as well, following the same compute-optimal framework.

**The consequence.** The reported advantages of test-time compute over pretraining — e.g., +27.8% relative improvement on easy questions at R << 1 for revisions (Figure 1, top-right bar chart) — may be inflated by the weak baseline. Several factors could reduce or reverse the advantage:

**Chinchilla-optimal larger model.** A model trained with 14× more FLOPs allocated optimally between parameters and data (rather than parameters alone) would likely achieve higher accuracy for the same total compute cost. The paper's parameter-only scaling means the larger model may be undertrained relative to compute-optimal practices, making it a weaker opponent for the test-time compute approach.

**Test-time compute for the larger model.** Even a modest test-time compute budget for the larger model — say, best-of-8 majority voting, which costs 8× inference FLOPs but uses the same greedy decoding approach — would create a much stronger baseline. The paper's compute-optimal framework could itself be applied to the larger model, potentially amplifying its advantage further. The current comparison gives the small model the full benefit of sophisticated inference while giving the large model none, which biases the comparison in favor of test-time compute.

**The R << 1 regime favors test-time compute by construction.** When R = D_inference / D_pretrain is very small (0.16 in the paper's experiments), the pretraining FLOPs dominate the total budget, and the savings from using a smaller model (fewer pretraining FLOPs) translate into a large inference budget for the small model. This is not a "fair" regime — it is the regime where test-time compute is structurally advantaged. The paper acknowledges this by testing across three R values (0.16, 0.79, 22), and the results at R >> 1 (Figure 9, the rightmost star positions) show that the advantage flips: on hard problems, test-time compute shows a −52.9% relative disadvantage compared to the larger model (PRM search, Figure 1, bottom-right). However, the headline finding — "a smaller model with test-time compute can outperform a 14× larger model" — is most prominently demonstrated at R << 1, and the caveat that this regime may not represent typical production deployments (where inference volume can be high) is not emphasized as strongly as the positive result.

**What evidence exists in the paper.** The paper provides the full R-dependent results (Figure 9, Figure 1 bar charts), and the authors are transparent about the parameter-only scaling choice. But there is no ablation testing a Chinchilla-optimal larger model, no experiment giving the larger model any test-time compute, and no sensitivity analysis showing how the crossover point (where pretraining becomes preferable) shifts if the baseline is strengthened. The "up to 4×" and "+27.8%" figures are therefore **upper bounds** on the advantage of test-time compute over pretraining, valid only against the specific (weak) baseline used.

**Mitigation status.** The paper partially mitigates this by reporting results across difficulty levels and across multiple R values (not just the favorable R << 1 regime). The breakdown in Figure 1 shows that the advantage is concentrated on easy-to-medium problems and favorable R regimes, with hard problems and high R values favoring pretraining. This nuance is present in the paper for a careful reader, but the high-level claims ("outperform a ~14× larger model") do not always carry the appropriate caveats.

The paper's framing of this limitation as "representative of a canonical approach to scaling pretraining compute" is debatable — the canonical approach in the scaling laws community (following Hoffmann et al., 2022) is Chinchilla-optimal scaling, not parameter-only scaling. The choice of baseline reflects a specific model series (LLaMA) but does not represent the frontier of compute-optimal pretraining. A practitioner deciding between training a larger model and deploying a smaller model with test-time compute would need a much more symmetric comparison than the paper provides.

---

### Verifier Over-Optimization Is a Hard Performance Ceiling That the Compute-Optimal Policy Mitigates but Does Not Solve

**The assumption or constraint.** The paper's search methods (beam search, lookahead search) rely on the process reward model (PRM) to score partial solutions and guide the search toward correct answers. However, the PRM is an imperfect learned model, and as the search budget increases, the search algorithms find solutions that score highly under the PRM but are actually incorrect — a phenomenon known as **over-optimization** or reward hacking. The paper documents this clearly in Section 5.3:

> "The degradation at high budgets is attributed to over-optimization of the PRM — search finds solutions that score highly under the PRM but are actually incorrect."

The compute-optimal policy mitigates this by routing easy problems away from aggressive search (using best-of-N instead of beam search on bins 1–2), where over-optimization manifests most severely. But on medium-difficulty problems (bins 3–4) where beam search is deployed because it provides genuine benefits, over-optimization still imposes a scaling ceiling — the beam search curves in Figure 3 (right) flatten and in some cases decline at high budgets.

**The consequence.** There are two distinct failure modes, both visible in the paper's results:

**On easy problems (bins 1–2), beam search makes the model worse at high budgets.** Figure 3 (right) shows that on bin 1, beam search accuracy decreases slightly as the budget increases from 4 to 256 generations, while best-of-N weighted increases substantially. This is the clearest signature of over-optimization: the search algorithm is optimizing a proxy (PRM score) that is misaligned with the true objective (correctness) on a distribution where the proxy's errors are exploitable. The compute-optimal policy routes these problems to best-of-N, avoiding the degradation, but this is a workaround, not a solution — the PRM is still unreliable enough that aggressive optimization hurts, and the system must detect this and back off.

**On medium problems (bins 3–4), beam search outperforms best-of-N but plateaus.** In Figure 3 (right, bin 3), beam search accuracy grows from the lowest budget to roughly 64–128 generations and then flattens. The curve suggests that additional compute beyond this point would provide diminishing or zero returns, even though the base model's pass@1 on these problems is non-zero (meaning there are still incorrect solutions being generated that could potentially be filtered out). The plateau is the over-optimization ceiling — the PRM cannot distinguish the remaining correct solutions from high-scoring incorrect ones, so further search does not improve accuracy.

**Lookahead search — the most powerful optimizer — paradoxically performs worst overall.** Figure 3 (left) shows that lookahead search (both k=1 and k=3) generally underperforms simpler methods at the same generation budget, despite using more sophisticated scoring. This is a striking example of over-optimization: the extra compute spent on lookahead rollouts produces PRM scores that are *more* exploitable by the search algorithm, leading to worse final answers than simpler scoring. The paper does not analyze this failure in depth beyond noting the general over-optimization phenomenon.

**Qualitative failure modes.** Appendix M (Figures 28–29) shows examples of degenerate outputs produced by search: "low-information repetitive steps at the end of solutions" and "overly short 1–2 step solutions." These are solutions that the PRM scores highly (likely because they look superficially plausible or exploit patterns in the PRM's training data) but that a human would recognize as insufficient.

**What evidence exists in the paper.** The over-optimization evidence is distributed across multiple analyses:
- Figure 3 (right, bins 1–2): beam search degrades or underperforms best-of-N at high budgets.
- Figure 3 (left): lookahead search underperforms beam search and best-of-N at the same budget.
- Figure 3 (right, bins 3–4): beam search curves plateau at high budgets.
- The PRM-vs-ORM comparison (Appendix F, Figure 14) shows that PRM continues to provide benefits over ORM at very high sample counts (2048), suggesting the PRM signal has real value despite over-optimization — the problem is search optimization, not PRM quality per se.

**Mitigation status.** The paper does not address the root cause of over-optimization — it does not propose techniques for training more robust verifiers, for regularizing search to avoid exploiting verifier errors, or for detecting over-optimization dynamically. The compute-optimal policy is a mitigation strategy (route easy problems away from aggressive search) rather than a solution (make aggressive search safe on easy problems). The paper's Section 8 flags "improving the PRM, e.g., by using adversarially robust training or ensemble methods" as future work, but provides no experiments in this direction.

This limitation is significant because it bounds the ultimate scaling potential of test-time compute. Even with perfect difficulty estimation and optimal strategy selection, the system cannot scale compute arbitrarily on any problem difficulty level — it will hit the verifier over-optimization ceiling. The paper demonstrates that this ceiling can be pushed back by choosing the right strategy (best-of-N on easy problems, beam search on medium problems), but the ceiling itself is determined by verifier quality, and the paper does not show how to raise it.

---

### The Revision Model Has a 38% Correct-to-Incorrect Reversion Rate, and the Only Mitigation Is Post-Hoc Selection Across the Revision Chain

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect, followed by a correct target answer (Section 6.1). During inference, the model generates a chain of revisions: it produces an initial answer, then conditions on that answer to produce a revision, then conditions on the revision to produce another revision, and so on. However, because the model was never trained on trajectories containing correct answers in context, it has no signal for what to do when the current answer is already correct. The consequence, reported in Section 6.1, is that approximately **38% of correct answers produced during a revision chain get "revised" back to incorrect answers** in the subsequent revision step.

The paper's mitigation is to not trust the final revision output. Instead, the system applies a selection mechanism — majority voting or verifier-based selection — across the entire chain of revisions, picking the best answer from any point in the chain. This means the revision chain is treated as a set of independent candidate answers, and the final output is the one that scores highest, not necessarily the last revision.

**The consequence.** Three practical issues arise from this limitation:

**The revision chain wastes compute.** At 38% reversion, roughly 4 out of every 10 correct answers are subsequently "broken" by the model and must be recovered by the selection mechanism. This means the effective number of useful revisions in a chain is lower than the total number of revision steps — some steps are actively harmful, undoing previous progress. The paper's revision-chain pass@1 trajectory (Figure 6, left) shows improvement from ~18% to ~24–25% over 15–20 steps, but the trajectory is noisy and the improvement per step diminishes. The reversion phenomenon is one reason why the pass@1 curve does not monotonically increase.

**The selection mechanism is an imperfect patch.** Verifier-based selection (using the revision-specific ORM) and majority voting are applied to choose the best answer from the chain. But these selection mechanisms are themselves imperfect — the verifier can make errors, and majority voting requires multiple chains to be effective (within a single chain, there may not be a majority). A correct answer that gets revised to an incorrect one will only be recovered if the selection mechanism correctly identifies the earlier (correct) version as better than the later (incorrect) version, which is not guaranteed.

**The model cannot "know when to stop."** A well-functioning revision system would recognize when an answer is correct and stop revising — or at least avoid degrading it. The current revision model cannot do this because it was never trained to do so. It always produces another revision, regardless of whether the current answer is correct. This means the system must always run the full revision chain budget (e.g., N sequential steps) and then select, rather than adaptively stopping early when a good answer is found. This wastes compute and increases latency (since revisions are sequential and their latency adds linearly).

**What evidence exists in the paper.** The 38% reversion rate is reported explicitly in Section 6.1. The paper also provides indirect evidence in the revision model's design choices: the fact that a within-chain selection mechanism is necessary at all (rather than simply taking the final revision output) reflects the reversion problem. The pass@1 curve in Figure 6 (left) shows improvement over the chain but with noise and diminishing returns, consistent with a model that sometimes degrades correct answers.

The ReST^EM experiment (Appendix K, Figure 16) provides additional evidence that revision training is fragile. When the revision model is further optimized using ReST^EM (on-policy RL-style training), "additional sequential revisions substantially hurt performance" — the fully sequential configuration drops to ~33.5% accuracy vs. ~38.5% at the optimal ratio. The authors hypothesize that "the on-policy data collection in ReST^EM exacerbates spurious correlations in revision data." This negative result suggests that the revision model's behavior is sensitive to training methodology and that the 38% reversion rate is not a fixed property but a symptom of a deeper training challenge.

**Mitigation status.** The paper partially mitigates the reversion problem through post-hoc selection (majority voting or verifier-based selection across the chain), but this treats the symptom rather than the cause. The authors do not propose or evaluate training the revision model to recognize correct answers and stop revising, nor do they explore alternative training data constructions that include correct-to-correct trajectories. Section 8 does not explicitly flag the reversion problem as an area for future work, though improving the revision model is mentioned generally.

This limitation is consequential because it caps the effectiveness of sequential revisions. Even with perfect post-hoc selection, the 38% reversion rate means that running more revision steps eventually hits diminishing returns — the model generates new candidates, but a significant fraction of them are degraded versions of previously correct answers. The compute-optimal ratio results (Figure 7) reflect this: fully sequential revisions (the rightmost points) are never optimal except on the easiest problems (bin 1, where reversion may be less common because the model's initial answers are mostly correct to begin with).

---

### The Difficulty Binning Is Coarse and Static — Within-Bin Heterogeneity and the Lack of Dynamic Adaptation Leave Potential Gains on the Table

**The assumption or constraint.** The paper partitions questions into exactly five difficulty quintiles based on the base model's pass@1 rate, estimated from 2048 samples per question. The compute-optimal policy is then a static lookup: for each (difficulty bin, compute budget N) pair, pre-compute which strategy performed best on the validation fold, and always apply that strategy at test time. This approach makes two implicit assumptions:

**Coarse discretization is sufficient.** Five bins means questions with meaningfully different optimal strategies may be grouped together. A question at the easy end of bin 3 (pass@1 of, say, 25%, just above the bin boundary) might benefit from a different strategy than a question at the hard end of bin 3 (pass@1 of, say, 15%, just above the bin 4 boundary). The paper provides no sensitivity analysis showing how the compute-optimal policy changes if the bin count is varied (e.g., 3 bins, 10 bins, continuous difficulty).

**The strategy is fixed for the entire inference process.** Once a question is assigned to a bin and a strategy is selected, that strategy runs to completion with no adaptation. There is no mechanism for adjusting the strategy mid-computation based on intermediate signals — for instance, starting with a few parallel samples, assessing whether the problem seems easier or harder than the estimated bin, and reallocating the remaining budget accordingly.

**The consequence.** Two types of inefficiency arise:

**Within-bin misallocation.** If the optimal strategy varies substantially within a quintile, the static bin-level policy is a compromise that is suboptimal for many individual questions. The paper's cross-validation protocol (Section 3.2) selects the strategy that works best on average within each bin on the validation fold, but questions far from the bin's average difficulty may be served poorly by that strategy. The magnitude of this within-bin variance is not measured.

**Missed opportunities for dynamic adaptation.** An adaptive strategy could potentially achieve better accuracy at the same budget by using early signals to refine the difficulty estimate and adjust the strategy. For example, if the PRM scores for the first few generated samples are all very low (suggesting the problem is harder than estimated), the system could switch from sequential revisions (which are optimal for easy problems) to beam search (optimal for medium problems). Conversely, if early samples produce a correct answer quickly, the system could stop early rather than spending the full budget. Static binning foregoes these opportunities because the entire budget is pre-committed.

**What evidence exists in the paper.** The difficulty-bin granularity is not ablated. The five-quintile choice appears to be motivated by needing enough bins to capture the difficulty-dependent trends visible in Figure 3 and Figure 7, but the paper does not test whether more or fewer bins would change the compute-optimal policy's performance. The only indirect evidence is the comparison between oracle and predicted difficulty bins (Figures 4 and 8), which shows that the two binning methods (ground-truth correctness vs. PRM-score-based) produce similar compute-optimal curves. This suggests the binning approach is robust to the specific difficulty signal, but does not address the granularity question.

The paper acknowledges the potential for dynamic adaptation in Section 8:

> "dynamically adjusting the strategy mid-computation... could subsume the difficulty estimation cost into the solution process"

But this is listed as future work, not explored experimentally.

**Mitigation status.** The paper does not mitigate this within its experiments — the binning is coarse and static throughout. The fact that the compute-optimal policy still achieves substantial gains (4× efficiency) suggests that even coarse, static binning captures much of the available benefit, but the residual inefficiency from within-bin misallocation is unknown. A finer-grained or adaptive policy might achieve even larger gains, but the paper provides no evidence either way.

## 7. Implications and Future Directions

### How This Work Changes the Landscape

This paper is a **paradigm-level reframing**, not an incremental optimization. Before this work, KV cache management in LLM serving was treated as a tensor-allocation problem within the contiguous-memory paradigm inherited from deep learning frameworks. The question was "how do we fit these variable-sized contiguous tensors into GPU memory with minimal waste?" — and the answer space was bounded by the assumption that tensors must be contiguous. Compaction, better buddy allocators, and output-length prediction were all attempts to optimize *within* that paradigm, and all of them hit the same ceiling: even with perfect oracle knowledge of output lengths, 63.4% of KV cache memory is wasted (Figure 2, Orca Oracle).

vLLM **rejects the contiguous-tensor paradigm entirely** and replaces it with a virtual-memory paradigm: the KV cache is a logical address space, the GPU physical memory is a pool of fixed-size page frames, and a block table mediates every access. This is not a better allocator — it is a different abstraction layer that makes the fragmentation problem dissolve by construction. External fragmentation vanishes because all allocations are the same size. Internal fragmentation drops to less than one block per sequence. Reservation waste — the dominant problem, accounting for up to 57.3% of waste in existing systems — is eliminated because blocks are allocated on demand, not pre-reserved for hypothetical future tokens. The 96.3% utilization in Figure 2 is the natural consequence of this architectural shift.

The significance of this reframing extends far beyond the 2–4× throughput improvement. It **changes what future systems researchers will optimize**. Before vLLM, research on LLM serving efficiency focused on: (1) faster attention kernels (FlashAttention, FasterTransformer), (2) better batching policies (Orca's iteration-level scheduling), and (3) model parallelism strategies (tensor parallelism, pipeline parallelism). Memory management was an afterthought — assumed to be handled by the framework's allocator. After vLLM, memory management becomes a first-class design consideration with its own research agenda: block replacement policies (when to evict), block sizing strategies (adaptive block sizes for different sequence-length regimes), multi-level block tables (for models with hundreds of thousands of tokens of context), and memory compression (deduplication of identical KV blocks). The paper's virtual-memory framing provides the vocabulary and conceptual toolkit for this entire agenda.

The work also **reconciles a tension that was invisible before it was resolved**. The tension is this: iteration-level scheduling (Orca) makes throughput better by interleaving requests, but it also makes memory management *harder* by creating highly dynamic allocation patterns with many concurrent sequences of unknown length. Before vLLM, one could reasonably ask whether iteration-level scheduling was worth the memory management complexity — maybe request-level batching with simpler allocation was better overall. vLLM resolves this by showing that the two techniques are **complementary, not competing**. PagedAttention provides the memory management substrate that makes iteration-level scheduling viable at scale — the block table absorbs the dynamic allocation complexity, and the scheduler can freely add and remove requests at every iteration without worrying about fragmentation. The paper demonstrates their complementarity by combining both (vLLM uses iteration-level scheduling inherited from Orca, plus PagedAttention), achieving throughput that neither technique could achieve alone. Future systems should not choose between these — they should deploy both.

A third shift: the paper makes **memory sharing in LLM serving a general-purpose capability rather than a per-algorithm engineering effort**. Before vLLM, KV cache sharing — when it was possible at all — required custom logic for each decoding algorithm: one implementation for parallel sampling, another for beam search, another for prefix caching. vLLM's `fork`/`append`/`free` primitives with copy-on-write and reference counting abstract sharing into three operations that compose to handle all supported decoding algorithms without the attention kernel or the model executor knowing anything about the sharing topology. This means that *future decoding algorithms* — constrained decoding, speculative decoding, tree-of-thought reasoning, whatever researchers invent — can be implemented by composing these primitives, and they will automatically benefit from memory sharing without any changes to the memory management system. The block table is the compatibility layer: it presents a uniform interface (physical block IDs) to an execution engine that is oblivious to the higher-level semantics of how sequences relate to each other. This separation of concerns is the same architectural insight that made virtual memory in operating systems so durable — it has survived decades of new applications, new system calls, and new inter-process communication patterns because the page table abstracts all of that away from the physical memory management. vLLM's block table has the same potential for longevity.

Finally, the paper **provides a diagnostic framework that changes what the field measures**. Figure 2's decomposition of memory waste into internal fragmentation, external fragmentation, and reservation is not just a validation of vLLM — it is a diagnostic tool that any future KV cache management system can use to identify its primary source of inefficiency. The insight that reservation is the dominant term (57.3% for Orca Max, 38.2% even with oracle output lengths) tells the field that output-length prediction, no matter how accurate, can never solve the KV cache memory problem — the only solution is to stop reserving memory in advance. This is a lasting conceptual contribution that will guide the evaluation of every future LLM serving system.

**What research directions become more attractive:**
- Page replacement policies and multi-level memory hierarchies (GPU → CPU → SSD) for KV cache, now that the page abstraction exists.
- Memory compression techniques (zero-page detection for identical KV vectors, delta encoding between related sequences) that operate at block granularity.
- Adaptive block sizing that adjusts `B` based on sequence-length distributions or runtime memory pressure.
- Co-design of decoding algorithms with the block-sharing primitives — algorithms that maximize sharing opportunities become more efficient without changing the inference engine.

**What directions become less attractive:**
- Better buddy allocators or contiguous-memory compaction schemes for KV cache — these are now clearly bounded by the reservation ceiling, and the paging approach makes them unnecessary.
- Output-length prediction for memory reservation — even perfect prediction leaves 38% waste (Orca Oracle in Figure 2), so the return on investment for better predictors is capped.
- Per-decoding-algorithm memory management — the block-table abstraction handles sharing uniformly, so bespoke sharing logic for each algorithm is engineering effort better spent elsewhere.

---

### Follow-Up Research This Work Enables

**Stress-test vLLM under bursty, production-scale workloads with SLO constraints.** The paper evaluates vLLM under steady Poisson arrival processes on 1-hour traces. Real LLM services experience bursty arrivals (diurnal patterns, flash crowds when a new feature launches, correlated requests during peak hours). Under burst loads, the scheduler may exhaust physical blocks rapidly and trigger cascade preemption — many sequences evicted simultaneously, followed by many simultaneous recovery operations when load subsides. This could expose pathologies not visible in steady-state traces: is the all-or-nothing eviction policy too coarse when many sequences must be evicted at once? Does the recomputation recovery mechanism introduce latency spikes when many sequences are recomputed in parallel (since they all do a prompt-phase encoding pass simultaneously, competing for GPU compute)? A strong follow-up would replay production request traces (with timestamps) through vLLM, measure tail latency (p95, p99) under burst conditions, and identify whether preemption or recovery becomes a bottleneck. This would determine whether the FCFS + all-or-nothing design is robust to realistic arrival patterns or whether more sophisticated eviction (e.g., LRU on blocks) and admission control are needed.

**Implement and evaluate block-level KV cache compression.** vLLM's block abstraction makes compression tractable: instead of compressing entire sequences (which would break the block table mapping and copy-on-write sharing), individual physical blocks can be compressed independently. Two natural opportunities arise from the paper's results. First, **zero-page detection**: beam search and parallel sampling create many blocks that are identical across sequences (e.g., all beam candidates sharing prompt blocks). Currently, vLLM uses reference counting to share identical blocks, but the block manager must be explicitly told they are shared (via `fork`). There may be cases where independently arrived requests have identical or near-identical KV cache fragments (e.g., the same system prompt used by many users, but not pre-registered as a shared prefix). A background process that scans physical blocks, identifies duplicates (via hash-based comparison), and merges their reference counts could recover sharing opportunities that the `fork` mechanism misses. Second, **delta encoding**: when beam search candidates diverge by only one or two tokens (which is common — they share a long prefix and diverge at the tail), the divergent blocks are almost identical. Storing only the delta between the parent and child block, rather than a full copy, could reduce the copy-on-write cost (less data to copy) and reduce memory consumption (the child's block stores only the differences). The experiment would measure: (a) how many duplicate or near-duplicate blocks exist in realistic traces that are not captured by explicit sharing, and (b) whether the compression/decompression overhead is amortized by improved batch sizes.

**Develop a learned eviction policy for the physical block pool.** vLLM currently uses FCFS for admission and preempts the newest sequences first (all-or-nothing gang eviction). This is simple and fair, but does it maximize throughput? A learned eviction policy could predict which sequences will complete soonest (smallest remaining output length) and preferentially keep those in GPU memory, evicting sequences that are likely to generate many more tokens and therefore hold memory longer. The predictor would take as input the prompt text, the tokens generated so far, and perhaps the model's internal uncertainty estimates (entropy of the output distribution at each step) to predict remaining output length. The experiment would compare this against FCFS on traces with varied output-length distributions, measuring both throughput and fairness (do some requests get repeatedly preempted, causing starvation?). This would address a limitation the paper does not discuss: the FCFS eviction policy is not evaluated against alternatives, and there is no evidence that it is even locally optimal.

**Port vLLM to long-context models (100K+ tokens) and measure whether the block table scaling is adequate.** The paper's experiments use OPT models with 2048-token maximum sequence lengths, where a sequence spans at most `⌈2048/16⌉ = 128` logical blocks, making the block table tiny (128 entries of a few bytes each). With models supporting 100K+ token contexts (Claude, Gemini, GPT-4-128K), a single sequence could span 6,250+ blocks. The block table per sequence would grow to hundreds of kilobytes, and broadcasting block tables for thousands of concurrent sequences to GPU workers at every iteration could become a communication bottleneck. Moreover, the PagedAttention kernel's latency (20–26% higher than FasterTransformer at 256-token context lengths, Figure 18a) may scale poorly with block count — fetching 6,000 blocks with warp-level reads could stress the GPU's memory controller. The experiment would measure: (a) the block table transmission latency as context length and batch size increase, (b) the PagedAttention kernel latency for very long sequences (does the overhead stay near 20–26% or grow?), and (c) whether hierarchical block tables (two-level, similar to OS page tables) are needed to keep the per-sequence metadata small. This is critical because the industry trend toward longer contexts makes the block table scaling a potential Achilles' heel.

**Integrate PagedAttention with disaggregated prefill-decode serving architectures.** Recent work proposes splitting LLM serving into separate prefill servers (handling prompt encoding) and decode servers (handling autoregressive generation) to independently scale compute for the two phases. vLLM's block table abstraction could be extended to support KV cache transfer between prefill and decode nodes: after a prefill server processes a prompt and generates the initial KV cache blocks, the block table and physical blocks could be transferred to a decode server via RDMA or NVLink. The block table provides a clean serialization format (logical-to-physical block mappings), and the physical blocks can be transferred as fixed-size pages. The experiment would measure: (a) the transfer latency for KV cache blocks of realistic sizes, (b) whether copy-on-write sharing complicates transfer (shared blocks must be reference-counted across nodes or duplicated), and (c) the throughput improvement from independently scaling prefill and decode resources, using vLLM's memory management as the KV cache transport layer. This would test whether the block abstraction generalizes beyond single-node serving to distributed, disaggregated architectures.

**Measure and mitigate the performance impact of the 20–26% attention kernel overhead in compute-bound regimes.** The paper shows (Figure 12f, OPT-175B on Alpaca) that vLLM's advantage over Orca shrinks when the workload becomes compute-bound rather than memory-bound — when GPU memory is abundant relative to sequence lengths, the memory efficiency gains do not translate to throughput because the bottleneck shifts to computation. In this regime, the 20–26% attention kernel overhead (Figure 18a) becomes a net negative: vLLM runs the same workload *slower* per-token than FasterTransformer, without the compensating benefit of larger batch sizes. The paper acknowledges this as a boundary condition, but does not explore whether the overhead can be reduced. Possible optimizations: (a) use higher block sizes (128–256) when memory is abundant, reducing the number of block table lookups and improving GPU utilization per block read — Figure 18b shows that large block sizes can hurt on short-sequence workloads, but on long sequences they might close the kernel gap without significant fragmentation; (b) implement a fast path in the PagedAttention kernel that detects when blocks are actually contiguous in physical memory (a common case if few sequences are active) and falls back to contiguous reads, avoiding the block table indirection entirely; (c) use CUDA Graph precompilation to eliminate kernel launch overhead for the block table access pattern when the batch composition is stable across iterations. The experiment would measure whether any of these optimizations bring the kernel latency within, say, 5% of FasterTransformer's, recovering vLLM's parity in compute-bound regimes without sacrificing its advantages in memory-bound regimes.

---

### Practical Applications and Downstream Use Cases

**Cost reduction for LLM API providers through higher GPU utilization.** The most direct application of vLLM is reducing the cost per request for hosted LLM services. The paper demonstrates 2–4× higher throughput at equivalent latency compared to Orca (Oracle) — which means a service provider can serve 2–4× more requests per GPU-hour. At commercial GPU cloud pricing (~$3–$5 per A100-hour for on-demand instances, less for reserved), this translates directly to cost savings. For a service processing 1 million requests per day with OPT-13B serving ShareGPT-like long-form conversation workloads, the throughput improvement from 0.7 req/s (Orca Oracle, Figure 12a) to 1.7 req/s (vLLM) reduces the required GPU count from approximately 16 A100s to 7 — saving tens of thousands of dollars per month in compute costs. The practical deployment is straightforward: vLLM is open-source (Apache 2.0 license), implements the OpenAI API interface for drop-in compatibility, and supports the model families (OPT, LLaMA, GPT) that most providers use. The only operational change is replacing the serving engine — no model retraining, no prompt format changes, no API changes for downstream consumers.

**On-device and edge deployment where memory is the dominant constraint.** The paper's finding that memory efficiency improvements are most impactful when GPU memory is scarce (Figure 13a: 4.3× more batched requests on the memory-constrained OPT-13B single-A100 configuration) makes vLLM particularly valuable for edge and on-device deployment scenarios. Consider a code completion service running on a developer's workstation with a single consumer GPU (8–24 GB), serving a small model (e.g., LLaMA-7B or CodeLlama-7B). In this setting, every GB of KV cache memory reclaimed by PagedAttention translates directly into longer context windows, larger batch sizes (important if the model serves multiple IDE instances simultaneously), or the ability to run additional models concurrently (e.g., a code completion model and a chat model sharing the GPU). The near-zero fragmentation property means that the operator does not need to over-provision GPU memory as a safety margin against fragmentation — they can confidently allocate almost all available memory to model weights and KV cache, knowing that 96%+ of it will be productively used. This reliability is as important as the absolute throughput gain for deployment scenarios where GPU memory is a hard constraint and waste cannot be tolerated.

**Shared-prefix caching for multi-tenant LLM platforms.** Many LLM platforms serve diverse customers but use a small set of system-level prompts, few-shot examples, or task descriptions that are identical across requests. The paper's shared prefix experiment (Figure 16) shows that vLLM achieves 3.58× higher throughput when a 341-token prefix is shared across all requests. For a platform serving thousands of tenants who each configure their own system prompt but use common task templates (e.g., "You are a helpful customer support agent for {company}"), vLLM can precompute and cache the KV cache for common prompt components as shared physical blocks. Requests that differ only in their user-specific suffix (the actual query) can map the shared prefix blocks into their block tables, paying only the memory cost of the suffix. This turns prompt engineering from a purely accuracy-focused activity into one that also has direct cost implications: longer shared prefixes become *more* memory-efficient under vLLM (because the shared blocks are amortized across more requests), whereas under Orca they become *less* efficient (because each request independently stores the full prefix). This inverts the economic incentive for prompt design — system architects should aim to maximize the fraction of each prompt that is shared, since shared tokens are essentially free under vLLM's block table mapping.

---

### When to Prefer This Method

The paper presents vLLM as a general-purpose LLM serving system and does not articulate explicit tradeoffs where a different memory management approach would be preferred. The experiments demonstrate that vLLM **strictly dominates** the baselines (Orca, FasterTransformer) across all tested configurations — there is no workload or model size where vLLM underperforms the state of the art, only configurations where its advantage is smaller (e.g., compute-bound OPT-175B on Alpaca, Figure 12f). This is consistent with the architectural argument: PagedAttention eliminates a source of waste (contiguous pre-allocation) without introducing new failure modes — the 20–26% kernel overhead never outweighs the memory efficiency gains in the evaluated workloads.

However, the paper does identify boundary conditions where vLLM's advantage shrinks, implying where a simpler system might achieve comparable performance with less engineering complexity:

- **When the workload is compute-bound rather than memory-bound.** If GPU memory is abundant relative to the sequence lengths and batch sizes needed to saturate GPU compute — as in the OPT-175B on Alpaca experiment (Figure 12f) — the memory efficiency gains do not translate to throughput, and the 20–26% attention kernel overhead becomes a net negative. In this regime, a simpler contiguous-allocation system with a highly optimized attention kernel (like FasterTransformer) might achieve equal or slightly better performance with less implementation complexity. The practical indicator is the GPU utilization: if compute utilization is >90% without PagedAttention, adding it will not increase throughput and may slow per-token latency.

- **When all sequences are statically known to have the same maximum length.** A batch processing workload where every sequence is known in advance to be exactly N tokens (e.g., bulk evaluation on a fixed-length benchmark) eliminates the need for dynamic growth and on-demand allocation. In this degenerate case, pre-allocating contiguous tensors of exactly N tokens has zero waste (no reservation, no internal fragmentation), and the block table indirection provides no benefit. vLLM's design is specifically optimized for the uncertainty of online serving — unknown output lengths, dynamic arrival, variable sequence lengths — and its advantages diminish when that uncertainty is removed.

- **When the serving system cannot amortize the engineering complexity of custom CUDA kernels.** vLLM requires maintaining fused CUDA kernels for block read/write, block copy, and PagedAttention itself, which are not part of standard deep learning frameworks. For a small-scale deployment or research prototype where engineering effort is the primary constraint rather than GPU cost, using a simpler system (even with lower throughput) may be preferable. The paper does not explicitly discuss this tradeoff, but the 8.5K lines of Python and 2K lines of C++/CUDA represent a non-trivial codebase to integrate, debug, and maintain — especially as model architectures evolve and new attention variants emerge.
