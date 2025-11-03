# Ray: A Distributed Framework for Emerging AI Applications

**ArXiv:** [1712.05889](https://arxiv.org/abs/1712.05889)

## 🎯 Pitch

Ray introduces a unified distributed system that integrates stateless tasks and stateful actors in a single dynamic execution engine, tailored to meet the interdependent demands of simulation, training, and serving in modern AI—especially reinforcement learning—applications. By combining a flexible programming model with a scalable, fault-tolerant architecture capable of handling over 1.8 million tasks per second at millisecond latencies, Ray enables large-scale AI workloads previously limited by the inflexibility and bottlenecks of existing frameworks, accelerating both research and practical deployment in fields like robotics, autonomous systems, and real-time decision-making.

---

## 1. Executive Summary
Ray is a distributed execution framework designed to run three tightly coupled workloads that modern reinforcement learning (RL) applications need: simulation, training, and serving. It unifies two programming styles—stateless tasks and stateful actors—on a single dynamic task-graph engine, and achieves high scale through a replicated Global Control Store (GCS), a bottom‑up distributed scheduler, and an in‑memory, shared‑memory object store, demonstrating over 1.8 million tasks/second with millisecond-level latencies (Abstract; Figure 8b).

## 2. Context and Motivation
- Problem addressed
  - RL applications must:
    - generate data through massive parallel simulation (often heterogeneous and short-lived tasks),
    - train models using distributed computation (using CPUs/GPUs),
    - serve policies with low latency in tight control loops (milliseconds).
  - These three steps are interdependent and must be coupled within one application (Section 2; Figure 1 and pseudocode in Figure 2).

- Why it matters
  - Real-world RL scenarios (robotics, games, UAVs) require responsive control while constantly learning from new interactions. This requires a system that can launch millions of fine‑grained, heterogeneous computations per second, react dynamically to intermediate results, and efficiently use mixed hardware (Section 2; footnote estimating 1.28M tasks/sec at cluster scale).

- Where prior approaches fall short
  - Batch/dataflow systems (MapReduce, Spark, Dryad) assume bulk-synchronous stages and struggle with millisecond tasks and dynamic graphs (Section 2).
  - Task-parallel systems (CIEL, Dask) support dynamic graphs but lack strong support for distributed training/serving and do not scale to millions of tasks/second with low-latency scheduling (Sections 2, 6).
  - Deep learning frameworks (TensorFlow, MXNet) excel at static compute graphs but are not designed to orchestrate simulators or embedded serving (Section 2).
  - Serving systems (Clipper, TensorFlow Serving) focus on external prediction serving rather than training and simulation in the same loop (Section 2).
  - Stitching multiple specialized systems introduces latency and data-movement overhead unacceptable for RL’s tight loop (Section 2).

- Ray’s positioning
  - A general-purpose, cluster-scale framework that:
    - Expresses both stateless tasks and stateful actors in one programming model (Section 3; Table 1, Table 2).
    - Executes on a dynamic task graph with lineage to enable fault-tolerant recomputation (Sections 3.2, 4.2.3).
    - Uses a decoupled, replicated GCS so all other components can be stateless and horizontally scalable (Section 4.2.1).
    - Employs a bottom‑up distributed scheduler that keeps the global scheduler off the critical path while still optimizing for data locality and resource constraints (Section 4.2.2; Figure 6).

## 3. Technical Approach
This section explains how Ray’s programming model maps to a runtime architecture that schedules and executes dynamic, heterogeneous workloads at scale.

- Programming model (Section 3.1; Table 1; Table 2)
  - Remote functions as stateless `tasks`
    - Invoked with `f.remote(args)`; returns a `future` immediately.
    - Inputs can be concrete values or futures; outputs are immutable objects stored in the object store.
    - Idempotent: output depends only on inputs, enabling re-execution on failure.
  - `Actors` as stateful, long‑lived processes
    - Instantiated with `Class.remote()`; methods invoked with `actor.method.remote()`.
    - Methods execute serially on the same actor, sharing in‑memory state (e.g., a parameter server or a simulator instance).
  - Control primitives for heterogeneity and dynamism
    - `ray.wait(futures, k, timeout)`: return first `k` completed results (helpful when simulation durations vary).
    - Resource annotations on tasks/actors (e.g., `num_gpus=1`) guide placement.
    - Nested remote calls: remote functions can launch other remote functions/actors, allowing multi-source task submission (essential for scaling Section 4.2.2).

- Computation model: dynamic task graph with three edge types (Section 3.2; Figure 4)
  - Nodes: data objects and invocations (tasks or actor methods).
  - Edges:
    - `Data edges`: object dependencies (producer task → object → consumer task).
    - `Control edges`: nested-call dependencies (caller task → callee task).
    - `Stateful edges`: serialize method calls on the same actor to encode state dependency (Mi → Mj on the same actor). This also becomes part of the lineage used for reconstruction.

- Architecture overview (Section 4; Figure 5)
  - Application layer processes:
    - `Driver`: runs user program and submits tasks.
    - `Workers`: stateless executors of tasks, automatically started and managed.
    - `Actors`: stateful executors instantiated explicitly.
  - System layer components:
    - Global Control Store (GCS): sharded, chain‑replicated key–value store with pub/sub that holds all control-plane metadata: function table, task table, object table, event logs (Section 4.2.1).
    - Bottom‑up distributed scheduler: per-node local schedulers plus a global scheduler pool (Section 4.2.2; Figure 6).
    - Distributed object store: per-node shared memory, zero-copy, immutable objects with replication on demand; Arrow format (Section 4.2.3).

- How scheduling and data movement work (Section 4.3; Figure 7)
  - Example: compute `c = add(a, b)` when `a` on N1, `b` on N2.
    - Step 1–2: Driver submits to its local scheduler; if overloaded or constraints not met, the task is forwarded up to a global scheduler.
    - Step 3–4: Global scheduler queries object locations in GCS and chooses a node that minimizes estimated completion time (queueing + transfer time); here N2 (where `b` resides).
    - Step 5–7: N2’s local scheduler checks/replicates missing inputs (fetches `a` from N1 via object store).
    - Step 8–9: Worker executes `add` with zero-copy access to local inputs; result `c` is stored locally, registered in GCS, and pulled by requesters on demand (ray.get triggers callback once `c` appears; Figure 7b).

- Why these design choices
  - Decouple control metadata (GCS) from compute and scheduling to avoid central bottlenecks and simplify fault tolerance—every other system component is stateless and can restart by re-reading from GCS (Section 4.2.1).
  - Schedule “bottom‑up” to keep the global scheduler off the critical path for most tasks while still enabling global load balancing and locality-aware placement for overflow or constrained tasks (Section 4.2.2).
  - Immutable, shared-memory object store enables zero-copy intra-node access, simple consistency, and lineage-based reconstruction (Section 4.2.3).
  - Actors plus tasks let users pick the right tool per workload segment: actors for stateful, iterative compute (e.g., GPU models, simulators), tasks for fine-grained, stateless parallelism (Table 2).

- Implementation details (Section 4.2.4)
  - ~40K LoC, 72% C++ (system), 28% Python (application).
  - GCS built atop sharded Redis with chain replication; schedulers are single-threaded event-driven processes; large objects striped across multiple TCP connections.

## 4. Key Insights and Innovations
- Unifying tasks and actors on a single dynamic task graph
  - What’s new: most prior systems focus on either tasks (e.g., CIEL) or actors (e.g., Orleans/Akka). Ray integrates both and preserves lineage across them using stateful edges (Section 3.2; Figure 4).
  - Why it matters: RL and similar applications need both fine-grained stateless compute (simulation/data processing) and efficient stateful compute (training/serving, third-party simulators). This unification enables application-level optimizations like hierarchical aggregation with actors and post-processing with tasks on the same data paths.

- Decoupled, replicated Global Control Store (GCS)
  - What’s new: Ray centralizes control-plane state in a horizontally scalable, fault-tolerant store separate from schedulers and object stores (Section 4.2.1).
  - Why it matters: enables:
    - stateless schedulers and object stores that can restart transparently,
    - high-throughput scheduling and data transfer without putting the scheduler on the object-transfer path,
    - system-wide introspection for debugging and profiling.
  - Evidence: failover and recovery with sub-30ms client-observed latency during chain reconfiguration (Figure 10a); periodic flushing to bound memory enables 50M-task runs (Figure 10b).

- Bottom‑up distributed scheduling with locality and constraints
  - What’s new: local-first submission with escalation to global schedulers as needed, combined with a cost model that considers queueing delay and input transfer time, and resource constraints like GPUs (Section 4.2.2).
  - Why it matters: supports millions of short tasks/second while placing tasks close to their data and matching heterogeneous resources.
  - Evidence: near-linear scale to >1M tasks/s at 60 nodes and beyond 1.8M at 100 nodes (Figure 8b); locality-aware placement avoids 10–100× latency spikes for large-input tasks (Figure 8a).

- High-performance in-memory object store with lineage
  - What’s new: per-node shared-memory store (Arrow format) for zero-copy intra-node access; objects are immutable and replicated on demand; lineage used to reconstruct on failure (Section 4.2.3).
  - Why it matters: low latency for micro-tasks and high throughput for large objects.
  - Evidence: single client reaches >15 GB/s write throughput for large objects and ~18k IOPS for small objects (Figure 9).

## 5. Experimental Analysis
- Evaluation setup
  - Hardware: AWS EC2; primarily `m4.16xlarge` CPU and `p3.16xlarge` GPU instances unless noted (Section 5).
  - Benchmarks: microbenchmarks (locality, scalability, object store performance, GCS fault tolerance and flushing, reconstruction, allreduce), building blocks (distributed training/serving/simulation), and end-to-end RL applications (Evolution Strategies, PPO).

- Microbenchmarks
  - Locality-aware scheduling (Figure 8a)
    - Setup: 1000 tasks with one large input scheduled on 2 nodes.
    - Result: locality-aware placement keeps latency nearly flat as input size grows; locality-unaware placement increases latency by 10–100× for 10–100MB inputs.
    - Interpretation: tasks benefit from data-local scheduling; actors can’t move to large remote objects, so tasks are preferred for heavy post-processing.

  - End-to-end scalability (Figure 8b)
    - Setup: embarrassingly parallel empty tasks; varying nodes.
    - Result: linear scaling to >1M tasks/sec at 60 nodes; beyond 1.8M tasks/sec at 100 nodes; 100M tasks completed in 54s.
    - Support for claim: demonstrates the scheduler/GCS architecture can handle extreme task rates.

  - Object store throughput (Figure 9)
    - Result: throughput >15 GB/s for large objects; ~18k IOPS for small objects on a 16-core instance using multithreaded copy for objects >0.5MB.
    - Implication: memory copy dominates at large sizes; serialization/IPC dominates at small sizes.

  - GCS fault tolerance and flushing (Figure 10)
    - Reconfiguration latency: <30ms max client-observed during chain member failure and reintegration (Figure 10a).
    - Flushing: with periodic GCS flushes, memory stays bounded and a 50M-task run completes; without, memory grows linearly and the workload stalls (Figure 10b).

  - Reconstruction under failures (Figure 11)
    - Stateless chains: as nodes fail at 25s/50s/100s, Ray re-executes dependent tasks via lineage and regains throughput when nodes return (Figure 11a).
    - Actor methods: checkpointing reduces re-execution from ~10k to 500 methods during recovery when 2 of 10 nodes die (Figure 11b).

  - Allreduce performance and scheduler ablation (Figure 12)
    - Performance vs OpenMPI: on 16 nodes, Ray completes 100MB allreduce in ~200ms and 1GB in ~1200ms, outperforming OpenMPI v1.10 by 1.5×–2× for large objects (Figure 12a).
    - Sensitivity to scheduler latency: injecting just a few ms per task nearly doubles ring-reduce time (Figure 12b), underscoring the need for low-latency scheduling.

- Building blocks for RL
  - Distributed training with TensorFlow (Figure 13)
    - Setup: data-parallel synchronous SGD, parameter server on actors; ResNet-101 with synthetic data on V100 GPUs.
    - Result: Ray+TF matches Horovod and is within ~10% of Distributed TensorFlow (distributed_replicated mode), benefiting from pipelined gradient compute/transfer and direct writes to the object store via a custom TF op.

  - Embedded serving throughput (Table 3)
    - Setup: co-located client/server on a `p3.8xlarge`; compare Ray actor vs Clipper over REST; batch size 64; two models (5–10ms runtime) with 4KB and 100KB inputs.
    - Results:
      > Small input: Ray 6200 ± 21 states/sec vs Clipper 4400 ± 15; Large input: Ray 6900 ± 150 vs Clipper 290 ± 1.3.
    - Interpretation: co-location plus shared-memory serialization eliminates REST overhead, especially for large inputs.

  - Simulation throughput (Table 4)
    - Task: OpenAI Gym Pendulum-v0 timesteps/second.
    - Results:
      > 1 CPU: MPI 22.6K vs Ray 22.3K; 16 CPUs: MPI 208K vs Ray 290K; 256 CPUs: MPI 2.16M vs Ray 4.03M.
    - Interpretation: asynchrony with `ray.wait` and task-based scheduling improves cluster utilization over BSP-style rounds.

- RL applications
  - Evolution Strategies (Figure 14a)
    - Workload: periodic policy broadcast and aggregation of ~10,000 simulation tasks per iteration.
    - Result: Ray scales to 8192 cores; doubling cores yields ~1.6× speedup; median time to solve Humanoid-v1 is 3.7 minutes—more than 2× faster than the best published result of 10 minutes; the custom reference system fails beyond 2048 cores.
    - Mechanism: hierarchical aggregation using actors (made easy by nested tasks/actors).

  - Proximal Policy Optimization (Figure 14b)
    - Workload: asynchronous scatter–gather; collect 320k simulation steps per policy update; 20 SGD steps of batch size 32,768; ~350KB parameters.
    - Result: Ray PPO outperforms a highly optimized MPI baseline at all scales while using fewer GPUs (at most 8 GPUs and ≤1 GPU per 8 CPUs), enabling ~4.5× lower cost on CPU-heavy instances and, with spot instances and fault tolerance, up to 18× total cost reduction (Section 5.3.2).
    - Mechanism: resource-aware scheduling lets CPU-only simulation scale independently while pinning model objects in GPU memory; single-process multi‑GPU TF support used effectively.

- Overall assessment
  - The experiments span micro, component, and end-to-end levels; they include ablations (scheduler latency), fault-injection (node deaths), and robustness (GCS reconfigurations). Together they substantiate claims of scalability, low latency, fault tolerance, and applicability to demanding RL workloads.

## 6. Limitations and Trade-offs
- Scope and positioning (Section 1 and Discussion)
  - Not a replacement for general data processing (e.g., Spark) or enterprise model-serving platforms (Clipper, TF Serving). Lacks higher-level data APIs (query optimization, straggler mitigation) and full model lifecycle features.

- Scheduling and profiling
  - Ray often must schedule without full knowledge of future tasks in a dynamic graph, so optimal placement may require runtime profiling; sophisticated policies are future work (Discussion).

- Metadata and lineage overheads
  - GCS holds all control state; without flushing, memory can grow linearly with the number of tasks (Figure 10b). Flushing/snapshotting adds operational complexity.

- Object model constraints
  - Objects are immutable and must fit on a single node (“no distributed object” primitive; Section 4.2.3). Very large structures must be manually sharded into collections of futures.

- Actor trade-offs (Table 2)
  - Actors enable stateful efficiency but:
    - do not move to data (poorer locality than tasks),
    - serialize method execution on the same actor (limits intra-actor parallelism),
    - require checkpointing to bound recovery time (Figure 11b).

- Network and serialization
  - For small objects, serialization/IPC overhead dominates (Figure 9); for extremely latency-sensitive micro-tasks, Python overheads may matter unless offset by batching or actor amortization.

- Heterogeneous hardware assumptions
  - Ray relies on resource annotations to match tasks to GPUs/CPUs; incorrect annotations or highly fragmented heterogeneity can reduce utilization.

## 7. Implications and Future Directions
- How this changes the landscape
  - Ray demonstrates that a single, general-purpose runtime can handle RL’s end-to-end loop—simulation, training, serving—at cluster scale with millisecond responsiveness. This reduces bespoke systems engineering and encourages algorithmic innovation that exploits combined task/actor patterns (e.g., hierarchical aggregations, mixed CPU/GPU pipelines).

- Follow-on research directions
  - Smarter schedulers for dynamic graphs
    - Incorporate learned or profiled cost models; anticipate data reuse; integrate predictive placement for long actor chains.
  - Memory and data abstractions
    - First-class “distributed objects” with automatic partitioning; smarter eviction policies across object stores; zero-copy RDMA backends.
  - Fault tolerance enhancements
    - Selective lineage tracking; user annotations for non-mutating actor methods to reduce recovery work; faster checkpointing mechanisms.
  - Higher-level libraries
    - Built-in RL libraries that encode common patterns (parameter servers, replay buffers, distributed rollout collectors); policy evaluation/management services to bridge into production serving systems.
  - Cross-language and ecosystem integration
    - Deeper bindings with ML frameworks; standardized data schemas (Arrow) across components; tighter integration with external serving and dataflow systems where appropriate.

- Practical applications
  - Robotics and real-time control where simulation is interleaved with on-robot policy serving.
  - Large-scale game AI (e.g., AlphaGo-like pipelines), where many simulators feed training continuously.
  - Hyperparameter search, neural architecture search, and distributed optimization algorithms that benefit from fine-grained, heterogeneous task scheduling.
  - Any workload needing fast, fault-tolerant orchestration of short tasks with intermittent stateful components (e.g., parameter servers, streaming feature extractors).

By unifying programming models and re-architecting control and scheduling around a replicated control store and bottom-up placement, Ray delivers both a conceptual simplification and concrete performance/cost advantages for emerging AI systems. The paper’s detailed microbenchmarks and full applications (Figures 8–14; Tables 3–4) provide strong evidence that these design choices lead to practical, scalable systems for RL and beyond.
