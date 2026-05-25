# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

**ArXiv:** [2304.11277](https://arxiv.org/abs/2304.11277)

## 🎯 Pitch

PyTorch Fully Sharded Data Parallel (FSDP) introduces a natively integrated, production-grade solution for training massive models that cannot fit on a single GPU, by efficiently sharding model parameters, gradients, and optimizer states across devices. Co-designed with PyTorch's core systems for optimal performance and usability, FSDP achieves near-linear scalability while preserving the simplicity of local training. This innovation empowers both industry and research to tackle cutting-edge large models with greater efficiency, flexibility, and reduced computational barriers.

---

## 1. Executive Summary
This paper introduces PyTorch Fully Sharded Data Parallel (FSDP), a native, production-grade system for training models too large to fit on a single GPU by sharding parameters, gradients, and optimizer state across devices. It delivers near-linear scaling and performance comparable to conventional data parallelism on small models while enabling much larger models, through a design co-developed with PyTorch’s autograd, tensor, dispatcher, and CUDA memory allocator.

## 2. Context and Motivation
- Problem addressed
  - Training modern large models (e.g., 10B–175B+ parameters; Section 1) exceeds single-GPU memory when using ordinary data parallelism, which replicates all parameters, gradients, and optimizer state on each device.
  - Users also need a solution that:
    - Works across heterogeneous hardware interconnects (intra-node high bandwidth vs. inter-node lower bandwidth).
    - Preserves good “local training” ergonomics (i.e., little model code change).
    - Avoids GPU memory fragmentation and throughput collapse at scale.
- Why it matters
  - Large language models and recommendation systems (Section 1) are central to industry and research; cost and time to train dominate deployment feasibility.
  - A scalable, general, and robust method lowers barriers for the broader community and reduces operational risk and cost.
- Prior approaches and their gaps
  - Data parallelism (DDP; Section 2.1) replicates models and overlaps gradient communication with backward compute. It fails when a single replica does not fit GPU memory.
  - Pipeline and RPC-based model parallelism (Section 2.2) partition computation, but require model restructuring and careful scheduling, and often assume rigid architectures.
  - Zero Redundancy Optimizer (ZeRO) and related sharding (Section 2.3) reduce redundancy but rely on fragile internal hooks and are not tightly integrated with PyTorch’s core (Section 6).
- Positioning
  - FSDP extends the “shard parameters” family but is redesigned as a native PyTorch component co-evolving with autograd, tensor storage, the dispatcher, and the CUDA caching allocator (Sections 1, 3, 4). It emphasizes:
    - Non-intrusive user experience (Section 3.1).
    - Efficiency on realistic datacenter topologies via flexible sharding schemes (Section 3.2.2).
    - Communication-computation overlap and allocator-aware memory management (Sections 3.3–3.4).

## 3. Technical Approach
At a high level, FSDP divides the model into units, flattens and shards parameters within each unit, and materializes (gathers) only one unit’s full parameters at a time during forward/backward. Gradients are reduced and re-sharded immediately after they are computed.

Step-by-step, with mechanisms and design choices:

1) Partition the model into `FSDP units` (Section 3, Fig. 1)
   - An FSDP unit is a contiguous block of layers chosen to balance memory and communication (user-controlled via wrapping/annotation; Sections 3, 4.2).
   - During forward:
     - Before executing a unit’s layers, FSDP runs an `AllGather` to assemble full parameters from shards on all ranks; once layers finish, it frees peers’ shards to save memory (Fig. 1, “gather full params” then “free peer shards”).
   - During backward:
     - Just-in-time `AllGather` re-materializes the same unit’s parameters to match autograd’s needs.
     - After the unit’s gradient is fully computed, FSDP immediately `ReduceScatter`s the gradient to redistribute gradient shards, so each rank only retains its local shard (Fig. 1, “reduce-scatter gradients”).

   Definitions:
   - `rank`: a single process usually owning one GPU.
   - `world size` (`W`): total number of ranks.
   - `AllGather`: each rank receives the other ranks’ pieces to form the full tensor.
   - `ReduceScatter`: sum-reduce across ranks and give each rank a disjoint shard of the result.

2) Efficient model initialization for giant models (Section 3.1)
   - Challenge: initializing parameters of a huge model typically materializes full tensors, which may exceed a single GPU.
   - FSDP adds `deferred initialization`:
     - Construct the model on a “fake” device without allocating real storage while recording init operations.
     - Replay those ops per unit when moving tensors to a real GPU, materializing only one unit at a time and immediately sharding it (Section 3.1; Fig. 1 narrative).
   - Fallbacks (Section 4.1):
     - Initialize the full model unsharded on GPU if it fits (smaller than training footprint), then shard afterwards.
     - Initialize unsharded on CPU and stream units to GPU for sharding (“CPU streaming”), trading speed for capacity.

3) Sharding strategies and `FlatParameter` layout (Sections 3.2–3.2.3)
   - Sharding factor `F`: number of ranks over which each parameter is sharded. `F=1` is full replication (DDP), `F=W` is full sharding, `1<F<W` is `hybrid sharding`.
   - Full sharding (Section 3.2.1)
     - Communication overhead is higher than DDP (about 1.5× volume using ring algorithms), so efficiency matters.
     - FSDP constructs, per unit, a single 1D `FlatParameter` by concatenating and right-padding parameters to ensure even shard sizes across ranks; then splits into `F` equal chunks, one per rank (Fig. 3).
     - Why flatten/pad? Section 3.2.1 and Fig. 2 show:
       - Even input sizes and fewer, larger collectives are much faster. Uneven sizes cause extra copies and use fallback group operations, slowing down `AllGather` (Fig. 2a).
       - Given fixed total volume, small per-call sizes sharply increase total communication time (Fig. 2b; performance drops when each `AllGather` < ~33M elements).
     - Memory model (Section 3.2.1): Peak parameter memory is O(sum_i ψ_i/F + max_i ψ_i), where ψ_i is the numel of unit i’s `FlatParameter`. The sum term is the always-resident local shards; the max term is the transient largest unsharded unit.
     - Trade-off: Using more, smaller units reduces max_i ψ_i (lower peak memory) but increases the number of collectives (lower throughput).
   - Hybrid sharding (Section 3.2.2; Fig. 4)
     - Devices are grouped into sharding groups `S_1…S_{W/F}` and replication groups `R_1…R_F`. Gradients are reduced by a `ReduceScatter` within each sharding group, followed by an `AllReduce` within each replication group (Equation (1) in Section 3.2.2 explains equivalence).
     - Benefit: Map groups to datacenter topology (e.g., shard within node; replicate across nodes) to minimize cross-host traffic and mitigate oversubscribed links. Section 3.2.2 computes per-GPU cross-host traffic: hybrid reduces it to 2M(W−1)/(G·W) for a model of size M and G GPUs per host, versus 3M(W−1)/W (full sharding) or 2M(W−1)/W (full replication).
     - Also valuable for mid-sized models that do not fully utilize memory under full sharding.
   - Autograd integration (Section 3.2.3)
     - Original parameters become views into the unsharded `FlatParameter` via autograd-aware `torch.split` and `view`.
     - Gradients accumulate into the `FlatParameter` gradient at the right offsets, and FSDP registers a post-backward hook on `AccumulateGrad` to launch timely `ReduceScatter`.

4) Communication/compute overlap and prefetching (Section 3.3; Fig. 5)
   - Problem: issuing collectives on the default compute stream induces wrong ordering (collectives wait for compute).
   - Solution: issue `AllGather` on a dedicated NCCL stream to overlap with compute (Section 3.3.1). Synchronization is enforced at the CUDA stream level, not just via a CPU-side wait.
   - Backward prefetch (Section 3.3.2): issue the next unit’s `AllGather` before the current unit’s `ReduceScatter` to avoid two back-to-back exposed communications; uses the recorded forward order as a proxy for backward order.
   - Forward prefetch (Section 3.3.3): for static graphs, the next unit’s `AllGather` is issued early in forward using the prior iteration’s order.
   - Gradient accumulation (Section 3.3.4): with or without cross-rank communication, trading communication for memory when accumulating unsharded gradients locally.

5) Memory management and allocator-aware rate limiting (Section 3.4)
   - PyTorch’s CUDA caching allocator serves allocations per stream and cannot reuse a block until all dependent kernels complete (Section 3.4.1).
   - Fast CPUs can queue many producer-stream `AllGather` allocations while consumer-stream compute still holds buffers—leading to allocator “retry” cycles (blocking `cudaFree`s, defragmentation, sharp slowdowns).
   - FSDP includes a `rate limiter` (Section 3.4.2) that intentionally stalls the CPU to bound the number of in-flight `AllGather`s (at most two), still allowing overlap but preventing allocator thrashing. Practitioners can check `torch.cuda.memory_stats()['num_alloc_retries']` to diagnose fragmentation.

6) Runtime hooks and APIs (Section 4)
   - Two entry points: a `FullyShardedDataParallel` wrapper and a `fully_shard` annotator (Section 4).
   - Hooks to insert communication exactly when needed (Section 4.3):
     - Pre-/post-forward logic per unit to schedule `AllGather` and freeing.
     - Autograd tensor-output hooks to trigger pre-backward actions.
     - `AccumulateGrad` hooks to trigger `ReduceScatter` as soon as gradients are ready.
   - Mixed precision (Section 4.4): Keeps FP32 and low-precision (FP16/BF16) copies of parameters but reduces the transient unsharded peak to `K_low * max_i ψ_i` (instead of `K_full`), and runs collectives in low precision. Because gradients are sharded, a special sharded gradient scaler is provided.

7) Putting it together
   - FSDP’s central design choices—unit boundaries, `FlatParameter` construction, stream-aware collectives, prefetching, and rate limiting—work together to minimize peak memory and communication stalls while preserving the “just write local training code” experience.

## 4. Key Insights and Innovations
- Deferred initialization with record–replay on a fake device (Section 3.1)
  - What’s new: Create models without allocating real GPU memory; replay initializations per unit on GPU and immediately shard.
  - Why it matters: Enables initializing models that do not fit on a single GPU, without modifying model source code.

- `FlatParameter` with padding for even, large collectives (Section 3.2.1; Fig. 2 and Fig. 3)
  - What’s new: A single flat 1D tensor per unit is constructed and evenly chunked across ranks, with minimal right-padding.
  - Why it matters: Empirically shown to maximize NCCL efficiency by ensuring even sizes and large per-call messages (Fig. 2). It also enables zero-copy inputs/outputs for `AllGather`/`ReduceScatter`.

- Hybrid sharding mapped to topology (Section 3.2.2; Fig. 4)
  - What’s new: Flexible grouping that shards within local high-bandwidth “islands” and replicates across islands; mathematically equivalent gradient reduction via Eq. (1).
  - Why it matters: Reduces cross-host traffic and straggler effects, and provides a smooth memory–throughput trade-off for mid-sized models (Section 3.2.2).

- Stream-level communication overlap and prefetch (Section 3.3; Fig. 5)
  - What’s new: Issue collectives on a separate NCCL stream to avoid false dependencies; prefetch the next unit’s parameters during backward/forward.
  - Why it matters: Avoids exposed latency and maintains pipeline fullness, delivering significant speedup (Section 5.2, Fig. 6b shows ~18% gain on GPT‑175B).

- Allocator-aware rate limiting (Section 3.4, Fig. 6c)
  - What’s new: Deliberately cap in-flight `AllGather`s to prevent CUDA allocator fragmentation when CPU outpaces GPU.
  - Why it matters: Can turn unstable runs into fast, stable ones—up to multi-x speedups in cases with allocator retries (T5 in Fig. 6c).

- Native mixed precision and sharded gradient scaling (Section 4.4)
  - What’s new: Full/low precision parameter management aligned with FSDP’s “only-unshard-what-you-need” paradigm and collectives in low precision; a gradient scaler that preserves correctness under sharding.
  - Why it matters: Reduces both memory and bandwidth while maintaining numerical stability.

## 5. Experimental Analysis
- Setup (Section 5.1)
  - Hardware: up to 512 NVIDIA A100 (80 GB) GPUs, 2 Tb/s RoCE network.
  - Models:
    - `T5` at 611M, 2.28B, and 11B parameters; sequences length 512 (Sections 5.1–5.2).
    - `minGPT-175B` (Section 5.2, 5.4).
    - `DHEN` recommendation model with 768B sparse and 550M dense parameters (Section 5.1).
  - Metrics: TFLOPS/GPU, median batch latency, QPS for DHEN, and memory metrics (allocated/active/reserved).
  - Baselines: DDP for smaller models; FSDP variants (full, hybrid) with reshard-after-forward (RAF) vs. no-reshard (NRAF); prefetch on/off; rate limiter on/off.

- Main results
  - Small-to-medium models (Fig. 6a; Section 5.2)
    - Performance parity with DDP where both fit. For T5‑611M and T5‑2.28B:
      - Quote:
        - “TFLOPS/GPU ≈ 15.18 (full sharding) vs 14.61 (DDP) at 611M; 27.40 vs 25.76 at 2.28B” (Fig. 6a).
    - DDP OOMs at 11B, while FSDP trains it efficiently. With BF16:
      - Quote:
        - “T5‑11B reaches ≈148.48 TFLOPS/GPU with full sharding; ≈145.81 with hybrid” (Fig. 6a).
  - Effect of backward prefetch (Fig. 6b; Section 5.2)
    - Quote:
      - “Backward prefetch improves GPT‑175B throughput by ≈18% across 128–512 GPUs; e.g., ~150→~175 TFLOPS/GPU at 128 GPUs” (Fig. 6b).
  - Rate limiting (Fig. 6c; Section 5.3)
    - When allocator retries happen, rate limiting helps a lot:
      - Quote (T5):
        - “T5 (4 machines): median batch latency drops from 15.33 s (no limit) to 5.02 s (limit). T5 (2 machines): 18.61 s→8.36 s.” (Fig. 6c).
    - It can be neutral or slightly harmful when fragmentation is not the bottleneck:
      - Quote (DeepViT):
        - “DeepViT (4 machines): 21.64 s (no limit) vs 22.79 s (limit)” (Fig. 6c).
    - Diagnostic: Check `num_alloc_retries` in `torch.cuda.memory_stats()` (Section 5.3).
  - Very large models: GPT‑175B (Fig. 7b, Fig. 8b; Section 5.4)
    - Quote:
      - “Per‑GPU throughput reaches ≈173 TFLOPS (B=1) and ≈186 TFLOPS (B=2), scaling near‑linearly from 128 to 512 GPUs” (Fig. 7b).
    - Memory defragmentation can dominate in certain settings:
      - Quote:
        - “At 128 GPUs, B=2, the backward pass took 85.56% of the step time due to allocator effects; PyTorch’s reserved memory reached the full 80 GB” (Section 5.4; Fig. 8b).
  - DHEN recommendation (Fig. 7a & Fig. 8a; Section 5.4)
    - Trade-off across modes:
      - RAF (reshard-after-forward) uses the least memory but has lower QPS because it repeats parameter `AllGather`s in backward.
      - NRAF keeps unsharded parameters between forward and backward, increasing memory but improving QPS.
      - Hybrid sharding (smaller sharding groups) further improves QPS by better locality and smaller collectives at the cost of memory.
    - Quote:
      - “Peak memory per GPU consistently decreases as GPU count increases under all modes” (Fig. 8a).
  - T5‑11B scaling (Fig. 7c & Fig. 8c; Section 5.4)
    - Quote:
      - “Per‑GPU TFLOPS regresses by ≈7% from 8 to 512 GPUs (communication starts to dominate), but memory remains well below capacity—no defragmentation indicated” (Fig. 7c, Fig. 8c).

- Assessment
  - Convincing for the claimed goals:
    - Enables models that DDP cannot train (T5‑11B).
    - Achieves strong per‑GPU efficiency on 175B‑scale transformers with near‑linear scaling (Fig. 7b).
    - Ablations show that key mechanisms (prefetch, rate limiting) materially affect performance (Fig. 6b–c).
  - Missing comparisons:
    - No direct head‑to‑head with other ZeRO implementations in the paper; the focus is on PyTorch‑native and DDP comparisons.
  - Conditions/trade-offs:
    - Rate limiting helps only when allocator fragmentation is the bottleneck (Section 5.3).
    - RAF vs. NRAF is a memory vs. communication trade-off (Section 5.4; Fig. 7a & Fig. 8a).

## 6. Limitations and Trade-offs
- Assumption: The largest `FSDP unit` must fit unsharded on a single GPU (Section 3; memory bound O(sum_i ψ_i/F + max_i ψ_i)).
  - If a single layer exceeds GPU memory, you need to pair FSDP with tensor parallelism (Section 7.1.2).
- Communication overhead
  - Full sharding increases communication volume (~1.5× vs DDP with ring; Section 3.2.1). Performance depends on overlap; poor unit boundaries or poor topology mapping can expose more latency.
- Unit boundary selection
  - Too fine-grained: lower peak memory but more frequent collectives (Section 3.2.1).
  - Too coarse: fewer collectives but higher peak memory and risk of allocator pressure (Sections 3.2.1, 3.4).
- Allocator sensitivity
  - Multi-stream allocation patterns can trigger defragmentation and `cudaMalloc` retries if the CPU outruns GPU progress (Sections 3.4, 5.3). The rate limiter fixes this but can slow training when fragmentation is not present (Fig. 6c, DeepViT).
- Mathematical non-equivalence for some optimizers (Section 7.2.1)
  - Because optimizer steps operate on sharded `FlatParameter`s that cut across original tensor boundaries, optimizers relying on per-parameter structure/norms or global states may break equivalence without extra paddings or communications.
- Shared parameter handling (Section 7.2.2)
  - Must ensure a shared weight is assigned to the lowest-common-ancestor unit; otherwise one unit may reshard a parameter another unit still needs unsharded, causing runtime errors or unintended resharding windows.

## 7. Implications and Future Directions
- How this changes the landscape
  - Makes “shard-by-default” feasible inside PyTorch with minimal user code changes, unlocking routine training of 10B–175B+ models on commodity multi-node clusters (Sections 1, 5.4). It brings a principled, allocator- and autograd-aware approach into the core framework.
- Practical applications
  - Large LLMs and multi-billion parameter recommenders (Sections 5.1, 5.4) in research and production where cost, reliability, and ergonomics matter.
  - Hybrid sharding lets practitioners map to real datacenter topologies for better stability and throughput (Section 3.2.2).
- Follow-up research enabled/suggested
  - Automatic unit boundary and sharding-factor selection: autotuners that balance `max_i ψ_i` vs. collective counts based on live profiling (Sections 3.2.1, 4.2).
  - Optimizer co-design: sharding-friendly optimizer math that preserves equivalence without heavy padding or extra communication (Section 7.2.1).
  - Deeper integration with tensor and pipeline parallelism (Sections 7.1.1–7.1.2), using PyTorch’s `DTensor` and `parallelize_module` to go beyond 2D into 3D parallelism with topology-aware routing.
  - Allocator innovations: stream-aware block pooling and proactive defragmentation strategies to reduce reliance on CPU-side rate limiting (Sections 3.4, 5.3).
  - Dynamic prefetch policies: tighter coupling between autograd execution order prediction and prefetch to improve overlap on dynamic graphs (Sections 3.3.2–3.3.3).

> In short, FSDP provides a general, native path to train models far larger than a single GPU’s memory while maintaining high hardware efficiency. Its detailed engineering around communication layout (Fig. 2–5), allocator behavior (Section 3.4), and topology-aware sharding (Section 3.2.2) is what makes it practical at 100s of GPUs, as demonstrated by near-linear scaling up to 512 A100s on GPT‑175B (Fig. 7b) and robust operation across diverse model families (Figs. 6–8).
