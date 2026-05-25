# TensorFlow: A system for large-scale machine learning

**ArXiv:** [1605.08695](https://arxiv.org/abs/1605.08695)

## 🎯 Pitch

TensorFlow introduces a unified dataflow-based programming model that seamlessly integrates computation and mutable state, enabling scalable training and inference across heterogeneous hardware like CPUs, GPUs, and custom ASICs. Its architecture empowers researchers and engineers to develop and experiment with new models and optimization algorithms flexibly, bridging the gap between research and production at large scales—fundamentally accelerating advances in machine learning.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces TensorFlow, a general-purpose system for large-scale machine learning that represents both computation and mutable state as a single dataflow graph. By unifying stateful operations (e.g., model parameters) with computation and by providing portable execution across CPUs, GPUs, and custom accelerators, it enables flexible experimentation and scalable training/inference with strong performance on real applications.

## 2. Context and Motivation
- Problem/gap addressed
  - Large models and datasets require distributing training across many machines and accelerators while keeping communication efficient and code flexible. Prior “parameter server” systems scale but hard-code model/state handling, making algorithmic experimentation difficult; single-machine frameworks are flexible but do not scale; batch dataflow systems assume immutable data and deterministic re-execution, which conflicts with iterative, state-updating ML training (§2.1–2.2).
- Importance
  - Practically, state-of-the-art models (e.g., image classification, language modeling) are computationally intensive and parameter-heavy; scaling across heterogeneous hardware is essential. Theoretically, supporting mutable state and dynamic control flow alongside data-parallel execution widens the space of trainable models and optimization strategies.
- Prior approaches and shortcomings
  - Single-machine frameworks (Caffe, Theano, Torch) provide flexibility and GPU support but lack distributed execution (§2.2).
  - Batch dataflow systems (MapReduce, Spark, DryadLINQ) handle scale but require immutable datasets and deterministic subcomputations, making frequent model updates expensive and slow to converge with large batches (§2.2).
  - Parameter servers (DistBelief, Project Adam, “Parameter Server”) scale parameter updates but “privilege” the server implementation, making new optimizers or model-parallel patterns hard to implement without modifying low-level C++ (§2.2).
- Positioning
  - TensorFlow keeps the scalability of parameter servers but generalizes them by embedding stateful primitives directly into a portable dataflow graph. This lets users move logic (e.g., gradient aggregation, sharded softmax) to the machines that hold parameters, all from user code, not system internals (§3–4).

## 3. Technical Approach
Step-by-step view of how TensorFlow works.

- Core abstraction: unified dataflow graph (§3, Fig. 1)
  - Nodes are `operations` (atomic computations).
  - Edges carry `tensors` (dense, multi-dimensional arrays with primitive types like `float32`, `int32`, `string`).
  - Unlike classical dataflow, nodes can own and mutate state (making the graph a program for both computation and memory).

- Data model and state (§3.1)
  - `Tensor`: dense n-D array. Sparse data is represented either via encoded strings or as indices + values (e.g., m×n index matrix + length-m value vector).
  - `Operation`: takes m inputs, produces n outputs; has a `type` (e.g., `Const`, `MatMul`) and compile-time attributes to control behavior and types.
  - Stateful ops
    - `Variable`: owns a mutable buffer (e.g., model weights). Produces a reference handle; read via `Read`, update via ops like `AssignAdd`. This enables in-place updates and sharing across computations.
    - Queues (e.g., `FIFOQueue`): concurrent producer–consumer coordination. `Enqueue`/`Dequeue` block when full/empty, providing backpressure and synchronization in input pipelines and across concurrent executions.

- Partial and concurrent execution (§3.2; Fig. 1)
  - A single graph can contain many subgraphs for data input, preprocessing, training, and checkpointing.
  - A `step` is an execution on a selected subgraph defined by feed (inputs) and fetch (outputs). Before running, the runtime prunes unnecessary nodes. Multiple steps can run concurrently, coordinating through shared `Variables` and `Queues`.

- Placement, partitioning, and communication (§3.3)
  - Device placement: operations are assigned to devices (CPU/GPU/other) subject to constraints (e.g., stateful op and its state colocated; user-specified preferences like “any GPU”).
  - Partitioning: after placement and pruning, the runtime splits the graph into per-device subgraphs and inserts communication ops across device boundaries:
    - `Send`/`Recv` pairs exchange tensors using a rendezvous key. Implementations are specialized for device pairs (e.g., CUDA async copies for CPU↔GPU, DMA for GPU↔GPU, and inter-task protocols like gRPC or RDMA; §5).
  - Caching: subgraphs are cached per device/session to minimize overhead across repeated steps (§3.3).

- Dynamic control flow (§3.4; Fig. 2)
  - Conditionals and loops are built from two primitives:
    - `Switch` (demultiplex): chooses one of two outputs based on a control input; the untaken branch propagates a special “dead” value.
    - `Merge` (multiplex): forwards the first non-dead input (or dead if both are dead).
  - Loops layer structural constraints inspired by timely dataflow: multiple concurrent iterations and nesting are allowed, with the restriction that each op produces a single value per output per iteration (simplifies distributed state and memory).

- Extensibility through user-level libraries and ops (§4)
  - Automatic differentiation and optimizers (§4.1): gradients are derived by reverse graph traversal and summation; users can customize gradients and implement optimizers (e.g., Momentum, AdaGrad, RMSProp, Adam, L-BFGS) entirely as graphs using `Variable` + math ops—no runtime changes.
  - Very large models via sparse embeddings (§4.2; Fig. 3): `Part` partitions indices per shard, `Gather` reads sparse rows colocated with parameter shards, `Stitch` reassembles results; with gradients defined for these ops, updates remain sparse and localized.
  - Fault tolerance via user-level checkpointing (§4.3): `Save` writes named tensors; `Restore` reads and `Assign`s them back to `Variable`s. Typically one `Save` per task for IO parallelism. By default, checkpoints may be inconsistent unless coordinated (OK for async training).
  - Synchronous replication with backup workers (§4.4; Fig. 4): barriers and aggregation queues implement synchronous SGD variants. Backup workers mitigate stragglers by taking the first m updates of n workers.

- System implementation details (§5; Fig. 5)
  - Core in C++ (portable across Linux/macOS/Android/iOS; x86 and ARM; NVIDIA GPUs).
  - Distributed master prunes, partitions, optimizes (common subexpression elimination, constant folding), and coordinates subgraphs; per-task dataflow executor schedules kernels with low overhead (~2,000,000 null ops/sec).
  - ~200 standard ops; kernels built with Eigen::Tensor and specialized libraries (cuDNN for convolutions/pooling; gemmlowp for quantized matmul). Hand-fused kernels for hot paths (e.g., activation + gradient).
  - Multi-protocol transport (gRPC/TCP, RDMA); specialized `Send/Recv` for locality and overlapping compute/transfer.

Why these design choices?
- Unifying computation and mutable state within a single graph makes “parameter server” behavior a consequence of the graph, not special system code. That lets users move compute to parameter shards (e.g., softmax/gradient on shards) and try new training schemes (§3–4).
- Explicit device placement, graph partitioning, and `Send/Recv` make communication costs visible and optimizable.
- `Switch/Merge` provide portable dynamic control without embedding a full general-purpose runtime in kernels, keeping the system efficient and analyzable (§3.4).

## 4. Key Insights and Innovations
- Unified stateful dataflow model (fundamental)
  - What’s new: Operations can own/modify state (`Variable`, queues), not just compute on immutable data (§3.1). This generalizes parameter servers inside a dataflow runtime.
  - Why it matters: Users can express optimizers, sharding strategies, coordination, and even consistency schemes directly in the graph and execute parts of the training algorithm where the data live (§4.1–4.4), without modifying privileged system code.

- Portable, high-performance distributed execution with graph partitioning and specialized communication (incremental but impactful)
  - What’s different: Automatic insertion of `Send/Recv` and caching of per-device subgraphs enables low-latency repeated execution; specialized CPU↔GPU and inter-task transports (gRPC, RDMA) and overlapping compute/transfer minimize overhead (§3.3, §5).
  - Why it matters: Enables scaling across many workers/PS tasks and heterogeneous devices; Figure 6 shows sub-10ms synchronization overhead in the best case (scalar), and 5–20ms sparse-embedding steps even with 16GB models.

- User-level implementations of “system” features (innovative in packaging)
  - Differentiation, large sparse embeddings, checkpointing, synchronous training with backup workers are all constructed from primitive ops and queues (§4.1–4.4). This lowers the barrier to experiment with new training algorithms and deployment practices.

- Dynamic control flow on dataflow graphs (incremental but enabling)
  - `Switch/Merge` allow non-strict evaluation for conditionals/loops (with timely-dataflow-inspired constraints). This supports models like RNNs with sequence-dependent control (§3.4).

## 5. Experimental Analysis
Evaluation setup spans microbenchmarks, single-machine baselines, and two real tasks (image classification, language modeling).

- Methodology overview (§6)
  - Focus: system performance (throughput, step time), not time-to-accuracy; results are medians with 10th/90th percentiles on a shared production cluster.
  - Hardware details are given per experiment.

- Single-machine convnet baseline (§6.1; Table 1)
  - Setup: 6-core Intel i7-5930K @ 3.5 GHz + NVIDIA Titan X; training step times for four models.
  - Results (ms/step):
    > Table 1: “TensorFlow 81, 279, 540, 445” for AlexNet, Overfeat, OxfordNet, GoogleNet.
  - Comparison: On this GPU setup, TensorFlow matches Torch within ~6% and substantially outperforms Caffe; Neon is faster on 3/4 models due to hand-optimized conv kernels. Attribution in the text links TensorFlow/Torch similarity to both using cuDNN R4, while Caffe used simpler open-source kernels.

- Synchronous replication microbenchmark (§6.2; Fig. 6)
  - Setup: Workers fetch from 16 PS tasks and send back a trivial update. Three workloads:
    - `Scalar`: fetch one 4-byte value per PS (measures coordination overhead).
    - `Dense 100MB/1GB`: fetch full model (tests bandwidth/scaling).
    - `Sparse 1GB/16GB`: fetch 32 random rows from a large embedding table (tests sparse access).
  - Results:
    > “Scalar median step time 1.8 ms (1 worker) → 8.8 ms (100 workers).”
    >
    > “Dense 100MB: 147 ms (1 worker) → 613 ms (100 workers); Dense 1GB: 1.01 s → 7.16 s.”
    >
    > “Sparse (1GB or 16GB): step times 5–20 ms and do not vary with table size.”
  - Interpretation: Coordination overhead is small; dense model traffic scales roughly with workers due to contention; sparse access keeps steps fast and size-independent—important for embedding-heavy models (§4.2).

- Image classification: Inception-v3 scaling (§6.3; Fig. 7, Fig. 8)
  - Setup: 17 PS tasks (8 IvyBridge cores each); workers each with 1 NVIDIA K40 GPU + 5 IvyBridge cores; up to 200 workers; compare asynchronous vs synchronous updates; evaluate backup workers.
  - Throughput:
    > Fig. 7(a): “Throughput improves to 2,300 images/second at 200 workers,” with diminishing returns as step time rises due to PS contention.
  - Step times and tails:
    > Fig. 7(b,c): “Median synchronous steps ~10% longer than asynchronous with same workers; tail (90th percentile+) is much worse for synchronous due to stragglers.”
  - Backup workers:
    > Fig. 8: “With 50 workers, adding 1–4 backup workers reduces median step time; 4 backups achieve 1.93 s/step (shortest), while 3 backups yield the best normalized speedup (9.5%).”
  - Assessment: The data convincingly show that (i) TensorFlow scales to hundreds of workers on a real model; (ii) synchronous training is feasible but straggler mitigation (backup workers) is important; and (iii) PS contention limits scaling, suggesting future comms/placement optimizations.

- Language modeling: LSTM on One Billion Word (§6.4; Fig. 9)
  - Setup: LSTM-512-512; vocabulary restricted to 40k for experiments; compare full softmax vs sampled softmax; vary number of PS tasks and workers.
  - Sharded full softmax:
    > Fig. 9 (dashed): “Adding PS tasks increases throughput by parallelizing multiplication/gradient on PS shards; adding a second PS is more effective than increasing workers from 4→32 or 32→256.”
  - Sampled softmax:
    > Fig. 9 (solid): “Sampling 512 classes per batch reduces softmax data transfer and compute by 78×,” substantially boosting throughput.
  - Assessment: Demonstrates that (i) shifting compute to PS shards (“model parallelism”) is effective; (ii) algorithmic changes (sampling) drastically reduce communication/computation and are easily expressed in TensorFlow via sparse ops (§4.2).

- Overall support for claims
  - The experiments substantiate the system’s scalability and flexibility: low overhead coordination (Fig. 6), competitive single-GPU performance (Table 1), practical multi-GPU/PS scaling for vision and NLP (Fig. 7–9), and benefits from user-level algorithm/system choices (backup workers, sampled softmax).

- Notable omissions/limitations in evaluation
  - As explicitly stated in §6, the focus is on throughput/step time, not time-to-target accuracy; ablations on placement/communication policies are limited; fault tolerance overheads are not benchmarked.

## 6. Limitations and Trade-offs
- Design assumptions and constraints
  - Tensors are dense by design (§3.1). Sparse data must be encoded or handled via (indices, values) decompositions. This simplifies low-level memory/serialization but can add complexity to user graphs and shape inference for highly sparse/irregular data.
  - Static graphs favored: caching and optimizations presume reusable graphs (§3.3). Dynamic control is available, but some users “chafe at the limitations of a static dataflow graph,” especially for algorithms like deep reinforcement learning where computation unfolds dynamically (§7).
  - Loop constraint: each op produces one value per output per iteration (§3.4). This eases memory management but can complicate certain dynamic-programming-like constructs.

- Fault tolerance and consistency
  - Checkpointing is by default inconsistent if concurrent with training (§4.3). For synchronous training or stronger guarantees, users must add extra coordination. Some applications requiring strong consistency are not directly addressed.

- Scalability bottlenecks
  - Parameter server contention: As workers scale, PS tasks become hot on network/aggregation (Fig. 7). Dense model updates suffer significant step-time growth (Fig. 6).
  - Stragglers in synchronous training: Tails degrade sharply (Fig. 7c). Backup workers help but consume extra resources and can increase PS load (Fig. 8 shows slight degradation beyond 4 backups).

- Evaluation scope
  - System metrics only (§6): no end-to-end convergence speed or accuracy comparisons across async vs sync; single-machine convnet results rely on third-party benchmark harness and specific library/kernel versions (Table 1).

- Portability caveats
  - While the core is portable, performance relies on specialized kernels (cuDNN, fused ops) and transport layers (RDMA). On platforms lacking these, performance will vary (§5).

## 7. Implications and Future Directions
- Field impact
  - TensorFlow reframes distributed ML from “use our parameter server with fixed behaviors” to “express your training algorithm as a stateful dataflow.” This democratizes experimentation with optimizers, sharding, and coordination policies and provides a single codepath from research to production (training and inference) across heterogeneous hardware (§1, §3–4).
  - The device-agnostic abstraction facilitates adopting new accelerators (e.g., TPUs) by adding device-specific kernels without changing user models (§2.1, §5).

- Enabled research and system work
  - Algorithmic: rapid prototyping of new optimizers, gradient estimators, control-flow-heavy models (e.g., RNNs with complex conditioning), and large-sparse model components (embeddings, sampled losses).
  - Systems: auto placement, kernel fusion, memory scheduling, better collective communication and PS alternatives; the paper lists these as active areas (§7).
  - Consistency models at user level: building stronger-consistency checkpointing or transactional updates atop queues and stateful ops (§7).

- Practical applications
  - Production-scale training and inference: vision, speech, language (case studies in §6 demonstrate readiness).
  - Mobile and edge inference: quantization support (gemmlowp) is built-in (§5), enabling efficient deployment on constrained devices.
  - Serving infrastructure and tooling: graph visualizers, profilers, and serving stack mentioned as part of the broader ecosystem (whitepaper [1] referenced in §5).

In sum, the paper’s core contribution is architectural: embedding mutable state and control-flow in a portable dataflow system that scales across heterogeneous clusters. The experiments show this design achieves competitive single-device speeds, efficient distributed coordination, and good scaling on realistic tasks, while making sophisticated training patterns expressible in user-level code. The main open fronts are optimizing placement/communication at scale, strengthening consistency where needed, and easing dynamic-programming-heavy workloads without sacrificing performance.
