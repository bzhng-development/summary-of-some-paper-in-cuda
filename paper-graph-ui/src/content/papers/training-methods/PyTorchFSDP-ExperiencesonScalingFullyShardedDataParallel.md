# PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

**ArXiv:** [2304.11277](https://arxiv.org/abs/2304.11277)

## 🎯 Pitch

PyTorch FSDP (Fully Sharded Data Parallel) introduces a native, production-grade solution for efficiently training enormous deep learning models that cannot fit on a single GPU—by sharding parameters, gradients, and optimizer states across devices, and only gathering full parameters as needed. Co-designed with PyTorch's core internals for plug-and-play usability, FSDP enables users to scale model size seamlessly without sacrificing training speed or requiring intrusive model code changes, making state-of-the-art large model training accessible to a broader community and unlocking the next wave of AI advancement.

---

## 1. Executive Summary
PyTorch Fully Sharded Data Parallel (`FSDP`) is a native, production-grade method to train models that do not fit on a single GPU by sharding parameters across devices, gathering them only when needed, and keeping gradients and optimizer state sharded. Its significance is twofold: it preserves a near “like-local” user experience in PyTorch (no intrusive model rewrites) while providing high efficiency through co-design with PyTorch internals (tensor/dispatcher, autograd, NCCL streams, CUDA memory allocator) and a toolbox of communication and memory optimizations.

## 2. Context and Motivation
- Problem addressed
  - The inability of classic data-parallel training (full model replication) to train very large models because every replica must hold the full parameters, gradients, and optimizer states on a single GPU (Section 2.1). This causes out-of-memory (OOM) errors for models in the billions of parameters.
  - The need for an industry-grade, framework-native solution that scales model size, not just dataset throughput, without rewriting model code (Section 1).
- Why it matters
  - Large models (language, recommendation) drive state-of-the-art quality and production use-cases but quickly exceed the memory of a single GPU (Introduction; Sections 1–2).
  - Practical training must use heterogeneous clusters with bandwidth islands (intra-node vs. inter-node) and avoid wasted GPU time due to communications or memory fragmentation (Section 1, “Hardware Heterogeneity” and “Resource Utilization”).
- Prior approaches and their gaps
  - Model replication (`DDP`): simple and fast but requires the entire model on each GPU; breaks for very large models (Section 2.1).
  - Model partitioning: pipeline parallelism and RPC can split models but require model structure constraints or code changes and tuning (Section 2.2).
  - Parameter sharding families: ZeRO/cross-replica sharding shard state but often depend on internal framework details and may incur uneven workloads or maintenance fragility (Section 6).
- Positioning
  - `FSDP` adopts parameter sharding like ZeRO but re-designs it as a first-class PyTorch feature co-designed with autograd, NCCL process groups, and the CUDA caching allocator. It adds deferred initialization, hybrid sharding tuned to hardware topology, overlapping/prefetching of collectives, and allocator-aware rate limiting (Sections 3–4).

## 3. Technical Approach
Step-by-step architecture (Sections 3.1–3.4; Figures 1–5):

1) Core algorithm: shard-by-unit, gather on demand
- Define an `FSDP unit`: a contiguous submodule (one or more layers) whose parameters are flattened together and collectively sharded (Figure 1, “Wrap & Shard”).
- At runtime:
  - Forward: before executing a unit, issue an `AllGather` to materialize the unit’s full parameters on each rank; compute; then free peer shards to minimize memory (Figure 1, “Forward”: “gather full params” then “free peer shards”).
  - Backward: before entering the unit in the reverse order, `AllGather` again if parameters were re-sharded after forward; compute gradients; immediately `ReduceScatter` to shard gradients and reduce across ranks (Figure 1, “Backward”: “gather full params” → compute → “reduce-scatter gradients”).
  - Optimizer states remain sharded at all times (Section 3).
- Memory bound: peak parameter memory per rank is proportional to the sharded model plus the largest fully materialized unit. With `N` flattened units of sizes `ψ1..ψN` and sharding factor `F`, peak parameter memory is O(sum_i ψi/F + max_i ψi) (Section 3.2.1).

2) Flatten–pad–chunk design for communication and memory
- `FlatParameter`: a 1D tensor created by concatenating all parameters in a unit, with right padding so its length is divisible by the sharding factor `F` (Figure 3; Section 3.2.1).
- Sharding: split the flat tensor into `F` equal chunks; each rank owns one local chunk. Gradients mirror this layout and are reduced-sharded in-place (Section 3.2.1).
- Why flatten and pad:
  - Communication efficiency improves when inputs are evenly sized and large (Figure 2): 
    - > “All-Gather Base with even input size achieved highest efficiency.” (Figure 2a discussion)
    - > “Once the AllGather size decreases below 33M elements, total communication time increases rapidly.” (Figure 2b)
  - Flattening coalesces many per-parameter collectives into fewer, larger collectives; padding guarantees even sizes across ranks, avoiding slower emulated behaviors (Section 3.2.1).

3) Sharding strategies tuned to hardware
- `F = 1`: full replication, equivalent to DDP (Section 3.2).
- `F = W`: full sharding; minimal memory but largest communication (1.5× DDP ring volume if bandwidth optimal) (Section 3.2.1).
- `1 < F < W`: hybrid sharding (Figure 4). Devices are partitioned into sharding groups `S` (within which parameters are sharded) and replication groups `R` (within which shards are replicated and later reduced by all-reduce) (Section 3.2.2).
  - Gradient reduction factorization (Equation 1, Section 3.2.2): split a global all-reduce into per-`S` reduce-scatter followed by per-`R` all-reduce.
  - Mapping to topology: set `F = W/G` for host size `G` to keep `AllGather`/`ReduceScatter` intra-host and only do smaller all-reduces inter-host. This reduces cross-host traffic (Section 3.2.2).

4) Autograd-safe parameter/gradient handling
- Each original parameter becomes a view into its unit’s full `FlatParameter` via autograd-visible `torch.split`/`torch.view`, so gradients accumulate at correct offsets (Section 3.2.3).
- A gradient hook on the `FlatParameter` triggers `ReduceScatter` exactly when the full unit gradient is ready (Section 3.2.3).

5) Deferred initialization for huge models
- Build the model on a “fake device” that allocates no real storage but records initialization ops; then load each `FSDP unit` to GPU and replay recorded ops, sharding immediately after (Section 3.1).
- Alternatives when deferred init is unsafe or cross-module dependent: initialize whole model on one GPU (if it fits) or stream from CPU unit-by-unit and shard on arrival (Section 4.1).

6) Communication overlap and prefetch (Section 3.3; Figure 5)
- Issue collectives on a dedicated NCCL stream (not the default compute stream) to avoid false dependencies. Synchronize at the stream level, not only through `Work.wait()` (Section 3.3.1).
- Backward prefetch: issue the next unit’s `AllGather` before the current unit’s `ReduceScatter` to avoid two back-to-back exposed communications in the single NCCL stream (Section 3.3.2).
- Forward prefetch: when execution is stable across iterations, issue the next unit’s `AllGather` ahead of time using the prior iteration’s order (Section 3.3.3).

7) Allocator-aware rate limiting (Section 3.4)
- Problem: multiple streams + fast CPU can cause the CUDA caching allocator to over-reserve blocks for the comm stream (unsharded parameter buffers) that cannot be reused quickly by the compute stream → allocator “retry” (blocking cudaFree/cudaMalloc) and severe slowdowns (Section 3.4.1).
- Solution: a rate limiter allowing at most two in-flight `AllGather`s (still enough to overlap comm/compute) to encourage block reuse and prevent fragmentation (Section 3.4.2).

8) Practical runtime hooks and APIs (Section 4)
- Two entry points: the `FullyShardedDataParallel` wrapper or the `fully_shard` module annotator (preserves module names/structure).
- Forward/backward integration via module pre/post hooks and autograd hooks: issue collectives at precise times, wait for pending ops before optimizer step, and launch `ReduceScatter` as soon as a unit’s gradients are complete (Section 4.3).
- Mixed precision (Section 4.4): keep low-precision params for compute and full precision for optimizer. Because only one unit is fully materialized, peak memory changes from `K_full * max ψ_i` to `K_low * max ψ_i`, and collectives can run at low precision. A sharded gradient scaler is provided for FP16.

Why these design choices:
- Flatten/pad/chunk maximizes NCCL efficiency (Figure 2) and minimizes useless copies (Section 3.2.1).
- Hybrid sharding maps naturally to datacenter locality while keeping a smooth memory–throughput trade-off for medium-size models (Section 3.2.2).
- Stream-level overlap/prefetch addresses the NCCL stream serialization and eager execution order constraints (Section 3.3; Figure 5).
- Rate limiting works with the PyTorch caching allocator’s behavior across streams under fast CPU issuance (Section 3.4).

## 4. Key Insights and Innovations
- Native, co-designed sharded data parallelism
  - Different from earlier ZeRO-like systems that modify framework internals, `FSDP` integrates with PyTorch autograd, NCCL process groups, and the CUDA allocator to achieve robustness and non-intrusive UX (Sections 3–4; Related Work in Section 6).
- `FlatParameter` + padding for communication-optimal collectives
  - Novel flatten–pad–chunk layout ensures even, large `AllGather`/`ReduceScatter` tensors, which empirical microbenchmarks show are much faster than uneven or small collectives (Figure 2). This both simplifies memory bookkeeping and yields fewer, larger collectives (Section 3.2.1).
- Hybrid sharding aligned to hardware topology
  - By choosing `F` and forming sharding/replication groups, `FSDP` localizes heavy `AllGather`/`ReduceScatter` within hosts and uses smaller all-reduces across hosts, reducing cross-host traffic and straggler effects (Section 3.2.2; Figure 4).
- Stream-level overlap and dynamic prefetch
  - Overlap collectives with compute despite eager execution constraints by issuing on dedicated NCCL streams and prefetching the next unit’s parameters in backward (Figure 5; Section 3.3.1–3.3.2). The paper reports ~18% TFLOPS gain on GPT-175B (Figure 6b; Section 5.2).
- Allocator-aware rate limiter
  - A practical, allocator-level fix for multi-stream fragmentation; when fragmentation occurs, limiting in-flight `AllGather`s yields up to 5× end-to-end speedups on T5-11B (Figure 6c; Section 5.3).
- Deferred initialization at framework level
  - Record–replay of parameter init on a “fake device” allows constructing giant models without GPU memory, then replaying and sharding unit-by-unit on real devices (Section 3.1).

## 5. Experimental Analysis
Evaluation setup (Section 5.1)
- Workloads:
  - Language: `T5-611M`, `T5-2.28B`, `T5-11B` (Figure 6a); `GPT-175B` (Figure 6b, Figure 7b).
  - Recommendation: `DHEN` with 768B sparse + 550M dense parameters; sparse trained by activation communication, dense by FSDP (Figures 7a, 8a).
  - Vision: `RegNet` (9B) and `DeepViT` (8B) for rate limiter sensitivity (Figure 6c).
- Hardware: up to 512× A100 80GB GPUs over 2 Tb/s RoCE (Section 5.1).
- Metrics: TFLOPS/GPU, latency per batch, peak allocated/active/reserved memory, QPS for DHEN (Section 5.1).

Main results and takeaways
- Comparable to DDP on small/medium, unlocks much larger models (Figure 6a)
  - > For 611M and 2.28B, FSDP and DDP achieve similar TFLOPS/GPU.
  - > DDP OOMs above 2.28B; FSDP trains T5-11B with BF16 at “148.48 TFLOPS/GPU.” (Figure 6a)
  - Interpretation: The sharded execution adds little overhead at moderate size while enabling models that do not fit under DDP.
- Communication microbenchmarks guide design (Figure 2)
  - > “All-Gather Base (even sizes) is faster than list-based uneven inputs; batching into larger All-Gather calls is crucial—below 33M elements per call, total time rises quickly.” (Figure 2a–b)
  - This justifies `FlatParameter` and padding choices (Section 3.2.1).
- Overlap and prefetch matter at scale (Figure 6b)
  - > Backward prefetching yields ~18% TFLOPS/GPU gain on GPT-175B across 128–512 GPUs. (Section 5.2; Figure 6b)
- Rate limiter: large gains when fragmentation happens, small regressions otherwise (Figure 6c; Section 5.3)
  - > On T5-11B, limiting in-flight `AllGather`s reduces median batch latency by up to 5×; on DeepViT, it can add ~5% overhead. (Figure 6c and Section 5.3 discussion)
  - Signal to enable it: high `num_alloc_retries` in `torch.cuda.memory_stats()` (Section 5.3).
- Training very large models efficiently (Sections 5.4; Figures 7–8)
  - DHEN RecSys:
    - Throughput–memory trade-offs across sharding strategies and “reshard after forward” (`RAF`) vs. “no reshard after forward” (`NRAF`) (Figures 7a, 8a).
    - > Full Sharding + RAF gives the smallest peak memory but lower QPS; Hybrid Sharding + NRAF gives higher QPS but higher memory. Peak memory per rank drops as GPU count increases. (Sections 5.4, 7a–8a)
  - GPT-175B:
    - > Achieves >173 TFLOPS/GPU (B=1) and >186 TFLOPS/GPU (B=2) up to 512 GPUs, around 55–60% of A100 BF16 peak, with near-linear scaling from 128→512 GPUs. (Figure 7b; Section 5.4)
    - > At 128 GPUs, B=2 triggers allocator defragmentation; reserved memory hits 80GB and backward dominates iteration time (85.56%). (Figure 8b and Section 5.4)
  - T5-11B:
    - > Stable memory well below capacity; per-GPU TFLOPS drops by ~7% going from 8 to 512 GPUs, implying comm starts to dominate and perfect overlap is unattainable at very large scale. (Figures 7c, 8c; Section 5.4)

Assessment
- The experimental suite is broad (language, recsys, vision), uses large GPU counts, and includes allocator diagnostics—convincingly supports claims about scalability, overlap/prefetch, and allocator-aware rate limiting.
- Ablations appear where most impactful: microbenchmarks (Figure 2), backward prefetch (Figure 6b), rate-limiter on/off (Figure 6c), and sharding strategy variants (`RAF` vs. `NRAF`) (Figures 7–8).

## 6. Limitations and Trade-offs
- Largest fully materialized unit must fit per GPU (Section 3)
  - The memory bound O(sum_i ψi/F + max_i ψi) means the largest `FSDP unit` determines the unsharded peak; a single “monster module” still won’t fit unless you combine with tensor parallelism (Section 7.1.2).
- Communication volume vs. memory
  - Full sharding minimizes memory but has the highest communication (e.g., ~1.5× DDP’s ring volume), requiring aggressive overlap/prefetch to hide latency (Section 3.2.1).
  - Hybrid sharding reduces cross-host traffic but increases replication overhead; its benefit depends on topology and model size (Section 3.2.2).
- Optimizer mathematical equivalence
  - Because original parameter boundaries are not preserved inside a `FlatParameter`, optimizers that depend on per-parameter structure, norms, or global parameter states are not strictly equivalent to local training without extra work/communication (Section 7.2.1).
- Shared parameters (weight tying)
  - Must ensure shared tensors live in the lowest-common-ancestor `FSDP unit` so they remain unsharded across all their uses; otherwise you can get missing-storage/size-mismatch errors. This can force keeping a unit unsharded longer than ideal (Section 7.2.2).
- Rate limiter is conditional
  - It helps when allocator fragmentation occurs; otherwise it can slightly hurt comm-bound models (e.g., DeepViT) by delaying `AllGather` (Figure 6c; Section 5.3).
- Eager execution constraints
  - Forward `AllGather`s follow compute in the same pass; overlap requires dedicated comm streams and prefetch heuristics. Dynamic graphs reduce opportunities for forward prefetch (Sections 3.3.1–3.3.3).

## 7. Implications and Future Directions
- Field impact
  - `FSDP` makes “large-model training” a drop-in experience for PyTorch users, extending the familiar data-parallel mental model to models an order of magnitude larger than single-GPU capacity. Its co-design with PyTorch internals sets a template for high-performance, framework-native parallelism (Sections 1, 3–4, 6).
- Practical applications
  - Immediate fit for LLMs, deep recsys with huge embedding tables (dense parts via FSDP), and large vision models. Hybrid sharding enables cost-effective training on heterogeneous clusters by aligning collectives with network locality (Sections 3.2.2, 5.4).
- Composability with other parallelisms
  - 2D parallelism: combine tensor parallelism (intra-node) with FSDP (inter-node) via `DTensor`/`parallelize_module`, letting tensor parallel handle units larger than a single GPU while FSDP handles data parallel sharding (Section 7.1.2).
  - Pipeline parallelism: wrap each stage with FSDP; prefer `NRAF` to avoid per-microbatch `AllGather`s, trading memory for less comm (Section 7.1.1).
- Research directions
  - Optimizer co-design: develop optimizer formulations compatible with `FlatParameter` sharding while preserving mathematical equivalence (Section 7.2.1).
  - Automated unit construction: use observed execution order to coalesce units dynamically after a warm-up iteration (Section 4.2).
  - Smarter prefetch scheduling: predictive or profile-guided prefetch under dynamic graphs.
  - Memory management: tighter integration with the caching allocator (e.g., allocator hints), fragmentation-aware scheduling, or OS/hypervisor-level support.
  - Communication: topology-aware collective planning beyond group selection (e.g., adaptive `F`, congestion-aware scheduling), and wider use of low-precision collectives (Section 4.4).

> Bottom line: By sharding parameters per unit, gathering only when needed, and engineering the runtime to overlap, prefetch, and respect allocator behavior, `FSDP` achieves near-linear scaling on very large models (e.g., GPT-175B at >173–186 TFLOPS/GPU up to 512 A100s; Figure 7b) while preserving a PyTorch-native user experience and enabling training far beyond single-GPU capacity.
