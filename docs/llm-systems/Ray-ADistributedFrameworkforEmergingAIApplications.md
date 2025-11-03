# Ray: A Distributed Framework for Emerging AI Applications

**ArXiv:** [1712.05889](https://arxiv.org/abs/1712.05889)

## 🎯 Pitch

Ray introduces a unified distributed system that seamlessly supports simulation, distributed training, and low-latency serving—key workflows for modern reinforcement learning (RL) applications—through a single, flexible programming model combining task-parallel and actor-based abstractions atop a highly scalable execution engine. By integrating a sharded global control store, bottom-up distributed scheduling, and an in-memory object store, Ray achieves massive throughput, transparent fault tolerance, and state-of-the-art RL performance, eliminating the need for brittle, stitched-together pipelines. This enables researchers and practitioners to rapidly build, scale, and innovate on real-time, end-to-end AI systems without the prohibitive complexity and inefficiency of fragmented infrastructures.

---

## 1. Executive Summary
Ray is a distributed system that unifies three workflows critical to modern reinforcement learning (RL) applications—simulation, distributed training, and low‑latency serving—behind one programming model and one execution engine. It introduces a combined task/actor API atop a dynamic task graph, and a scalable backend comprising a sharded global control store, a bottom‑up distributed scheduler, and an in‑memory object store, achieving up to 1.8 million tasks per second and state‑of‑the‑art performance on demanding RL workloads (Figure 8b, Sections 3–5).

## 2. Context and Motivation
- Problem/gap being addressed
  - RL applications run a tight loop: they continuously simulate environments to gather experience, train policies on heterogeneous hardware (CPUs/GPUs), and serve those policies at low latency (Figure 1; Section 2). Existing systems cover individual parts—e.g., TensorFlow for training, Clipper for serving—but not the end‑to‑end loop with tight latency and high task churn.
  - Workloads are highly dynamic and heterogeneous: simulation tasks can last milliseconds to minutes, dependences emerge at runtime, and resource needs vary by task (Section 2). The system must schedule millions of fine‑grained tasks per second at millisecond latencies.

- Why this matters
  - Real‑world RL systems (robotics, games, UAVs) must react in real time, explore via simulation at scale, and retrain continuously. Fragmented stacks introduce prohibitive data movement, coordination latency, and duplicated engineering of scheduling and fault tolerance (Section 1).

- Shortcomings of prior approaches
  - Bulk-synchronous frameworks (Spark, MapReduce, Dryad) assume stages of similar‑duration tasks and do not match fine‑grained, dynamic RL loops; they also lack low‑latency serving (Section 1).
  - Dynamic task graph systems (CIEL, Dask) provide futures and nested tasks, but not efficient primitives for stateful computation (parameter servers, GPU‑resident models) and not a control plane designed for millions of tasks/second (Sections 1, 6).
  - Deep learning frameworks (TensorFlow, MXNet) are optimized for static DAGs of tensor ops; they do not naturally express simulation/serving or dynamic graph mutation on task completion (Section 6).
  - Serving systems (Clipper, TensorFlow Serving) do not cover training or simulation (Section 1).

- Positioning
  - Ray aims to be the “glue” runtime for the RL loop: a single system that expresses both stateless and stateful computations with one dynamic task graph, and a backend tailored to low-latency scheduling, data movement, and lineage-based recovery (Sections 3–4). It is not a replacement for rich analytics stacks or full‑featured serving platforms (Section 1, caveats).

## 3. Technical Approach
Ray’s design has two layers (Figure 5): a minimal application API that builds a dynamic task graph, and a stateless, horizontally scalable backend for control, scheduling, and data.

A. Programming model (Section 3; Table 1, Table 2)
- Core abstractions
  - `Task` (stateless remote function): invoked as `f.remote(args)` and immediately returns a `future` (a placeholder for a result). Futures can be passed to other remote functions, allowing the system to track data dependencies without blocking. Ray provides `ray.get(futures)` (blocking) and `ray.wait(futures, k, timeout)` (return as soon as k results arrive) to handle heterogeneity in task durations (Table 1).
  - `Actor` (stateful remote object): created as `Class.remote(args)`; method calls like `actor.method.remote(args)` execute on the same stateful process, serially. Actor state persists across method calls and can encapsulate third‑party simulators or GPU‑resident models (Section 3.1). Actors are essential for workloads that need mutable state with low update overhead (e.g., parameter servers).
- Resource hints and nesting
  - Each task/actor declares resource requirements (e.g., `num_gpus=1`), enabling the scheduler to place work on heterogeneous nodes (Section 3.1).
  - Nested remote calls are allowed: tasks can spawn tasks, and actors can invoke tasks, enabling distributed submission and high scalability (Section 3.1).

- Why both tasks and actors?
  - Tasks excel at fine‑grained, stateless work (good locality, easy load balancing, simple re‑execution for fault tolerance), while actors excel at repeated small updates and stateful computation (Table 2). Ray unifies both in one execution model.

B. Computation model: dynamic task graph with stateful edges (Section 3.2; Figures 3–4)
- The program constructs, at runtime, a directed acyclic graph (DAG) whose nodes are:
  - Data objects (immutable values materialized in the object store).
  - Executions (task invocations and actor method calls).
- Three edge types encode dependencies:
  - Data edges (task produces/consumes an object).
  - Control edges (nested submission: parent spawns child).
  - Stateful edges (ordering constraint among successive actor method calls on the same actor), which capture implicit state flow and are recorded in lineage (Figure 4).
- Example end-to-end
  - Figure 3 shows a minimal RL training loop using Ray: create a policy (`create_policy` task), create simulator `actors`, parallel `rollout` calls on simulators, then `update_policy` (a GPU task) on the collected rollouts, repeated for multiple iterations.

C. Backend architecture (Section 4; Figures 5–7)
- Application processes (Section 4.1)
  - `Driver` runs user code and submits work.
  - `Workers` execute stateless tasks.
  - `Actors` execute stateful methods, serially.
- Global Control Store (GCS) (Section 4.2.1; Figure 10)
  - A sharded, replicated key‑value store (built on Redis with per‑shard chain replication) holding all control plane state: function registry, task lineage (dependencies), object directory (locations/sizes), and event logs.
  - Design choice: store lineage and object metadata in the GCS rather than in schedulers, keeping schedulers off the critical path for data transfer and enabling stateless components that can be restarted/replicated (Section 4.2.1).
  - Benefits: fast recovery (components reload control state), scalable scheduling (schedulers share global view), and easy instrumentation (debugging/profiling tools consume GCS data).
- Bottom‑up distributed scheduler (Section 4.2.2; Figure 6)
  - Per‑node `local schedulers` receive tasks from drivers/workers and aim to schedule locally, exploiting data locality and avoiding global contention.
  - If overloaded or lacking resources (e.g., no GPU), locals forward tasks to `global schedulers`. Global schedulers:
    - Gather node load/resource info via heartbeats and obtain object locations/sizes from the GCS.
    - Choose a node that satisfies resource constraints and minimizes estimated completion time = queueing delay (queue size × average task duration) + data transfer time (sum of remote input sizes / estimated bandwidth), with exponential averaging for estimates (Section 4.2.2).
  - Horizontally scalable: multiple global scheduler replicas coordinate via the GCS.
- In‑memory distributed object store (Section 4.2.3; Figure 7; Figure 9)
  - Each node hosts a shared‑memory store for immutable objects using Apache Arrow format; tasks read/write via zero‑copy when colocated.
  - On task launch, remote inputs are replicated to the local store; outputs are written locally. Objects are evicted to disk by LRU if needed.
  - Data transfer uses striped TCP connections for large objects (Section 4.2.4). The store is per‑object (not sharded across nodes), simplifying consistency and reconstruction via lineage.
- Dataflow of a simple remote call (Figure 7)
  - Submission: the driver calls `add.remote(a, b)`. The local scheduler forwards to a global scheduler if necessary; the global scheduler picks a node that has data/locality (e.g., node with `b`), orchestrates pulling missing inputs (e.g., `a` from node N1 to N2), and dispatches to a worker (steps 1–9 in Figure 7a).
  - Retrieval: `ray.get(idc)` registers a callback in the GCS’s Object Table if the result is not yet materialized; once `c` is produced and registered, it is pulled and returned (Figure 7b).

D. Fault tolerance (Sections 4.2.1, 4.2.3; Figures 10–11)
- Lineage-based reconstruction for tasks: if a node fails and an object is lost, schedulers re‑execute the upstream tasks based on recorded dependencies (Figure 11a).
- Actor recovery: stateful edges encode the sequence of method calls; actors can be reconstructed from checkpoints plus replay of subsequent methods; user‑provided checkpoints bound recovery cost (Figure 11b).
- GCS robustness: chain replication yields fast failover; client‑observed delays during reconfiguration remain under ~30 ms in tests (Figure 10a). Periodic “GCS flushing” snapshots lineage to disk and caps memory growth (Figure 10b).

## 4. Key Insights and Innovations
1) Unifying stateless tasks and stateful actors in one dynamic task graph
- What’s new: Ray adds an `actor` abstraction (stateful, serial method execution) to a futures‑based task model and integrates both into one dependency graph using `stateful edges` (Section 3.2; Figure 4).
- Why it matters: RL needs both patterns—stateless, fine‑grained simulation/data processing and stateful, low‑overhead updates (e.g., parameter servers, GPU‑resident models). Table 2 clarifies trade‑offs: tasks offer locality and easy recovery; actors offer efficient small updates.
- Difference vs prior work: CIEL/Dask have dynamic tasks but no first‑class stateful actors with lineage; Akka/Orleans have actors but no high‑throughput, lineage‑aware stateless task execution and reconstruction (Section 6).

2) Decoupled, sharded Global Control Store and stateless components
- What’s new: centralize control state (lineage, object directory, function table) in a fault‑tolerant store, not in schedulers or executors, and make all other components stateless (Section 4.2.1; Figure 5).
- Why it matters: keeps scheduling on the fast path while enabling transparent recovery and horizontal scaling; avoids coupling object metadata to a centralized scheduler that would bottleneck data‑intensive patterns like allreduce (Section 4.2.1).
- Evidence: under failures, GCS reconfiguration adds <30 ms delay (Figure 10a); memory can be bounded via flushing (Figure 10b).

3) Bottom‑up distributed scheduling for millions of tasks/second
- What’s new: local‑first scheduling to exploit locality and avoid global contention; escalate to global schedulers only on overload or resource mismatch; globals make decisions using queueing and transfer estimates (Section 4.2.2; Figure 6).
- Why it matters: supports millisecond tasks at scale; scheduler latency is critical for communication‑heavy primitives (Figure 12b).
- Evidence: near‑linear throughput growth to 1.8M tasks/s on 100 nodes; 1M tasks/s at 60 nodes (Figure 8b).

4) High‑throughput, zero‑copy in‑memory object store
- What’s new: per‑node Arrow‑based shared memory, replication on demand, and zero‑copy access between colocated tasks (Section 4.2.3).
- Why it matters: minimizes serialization and IPC overheads; critical for low latency in serving and for high‑volume simulation data.
- Evidence: single client achieves >15 GB/s write throughput on large objects and ~18K IOPS on small objects (Figure 9); serving throughput surpasses Clipper when client/server are colocated (Table 3).

5) A practical path to end‑to‑end RL systems
- What’s new: an API that directly expresses the tight simulation–training–serving loop (Figures 1–3) and supports application‑level optimizations (e.g., hierarchical aggregation for ES, pinned GPU objects for PPO).
- Why it matters: replaces ad‑hoc glue code and specialized systems; enables faster iteration and scale (Sections 5.2–5.3).

## 5. Experimental Analysis
Evaluation setup
- Hardware: AWS clusters. Unless noted, CPU instances m4.16xlarge; GPUs p3.16xlarge (Section 5).
- Methodology spans microbenchmarks, building‑block workloads (training, serving, simulation), and end‑to‑end RL algorithms (Section 5).

A. Microbenchmarks
- Locality‑aware placement benefits (Figure 8a)
  - Experiment: schedule 1000 tasks with a random single object dependency onto two nodes.
  - Result: with locality‑aware placement, mean task latency stays almost constant as input size grows from 100 KB to 100 MB. Without locality, latency increases by 1–2 orders of magnitude.
  - Takeaway: tasks can “move compute to data,” unlike fixed actors.

- End‑to‑end scheduler scalability (Figure 8b)
  - Result: “Ray reaches 1 million tasks per second throughput with 60 nodes” and continues nearly linearly beyond 1.8M tasks/s at 100 nodes; 100M empty tasks complete in 54 seconds (rightmost datapoint).
  - Convincing because: throughput scales with added nodes, and the result holds even as task duration increases (proportionally reducing throughput, as expected).

- Object store performance (Figure 9)
  - Write throughput: >15 GB/s for large objects from a single client; IOPS ~18K for small objects on a 16‑core machine.
  - Implication: serialization and IPC no longer bottleneck large transfers; memcpy dominates for large objects.

- GCS robustness and memory control (Figure 10)
  - Fault tolerance: under shard failure and reconfiguration, client‑observed read/write latencies remain under ~30 ms.
  - Flushing: without flushing, lineage memory grows linearly (50M tasks case stalls); with flushing, memory is bounded and long runs complete.

- Task and actor failure recovery (Figure 11)
  - Tasks: as nodes are removed (at 25s, 50s, 100s), throughput dips as Ray reconstructs dependencies, then stabilizes; when nodes return, throughput recovers (Figure 11a).
  - Actors: with checkpoints, only ~500 methods are re‑executed vs ~10K without, reducing recovery time markedly (Figure 11b).

- Allreduce primitive (Figure 12)
  - Performance vs OpenMPI: on 16 nodes with 25 Gbps networking, ring allreduce of 100 MB completes in ~200 ms; 1 GB in ~1200 ms. This is 1.5×–2× faster than OpenMPI v1.10 on these sizes (Figure 12a). For small objects, OpenMPI’s alternative algorithm is lower‑overhead and faster.
  - Scheduler ablation: injecting a few milliseconds of scheduler delay nearly doubles allreduce time (Figure 12b), highlighting the necessity of low scheduler latency.

B. Building‑block workloads
- Distributed training (TensorFlow ResNet‑101; Figure 13)
  - Setup: synchronous data‑parallel SGD with Ray actors as replicas; parameter‑server mode or allreduce; synthetic data; comparisons to Horovod and distributed TensorFlow using NCCL2 and OpenMPI 3.0.
  - Result: Ray matches Horovod throughput and is within ~10% of distributed TensorFlow across 4–64 V100 GPUs (workers allocate 4 GPUs per node).
  - Mechanism: pipelining gradient compute/transfer/sum; a custom TensorFlow op writes tensors directly to Ray’s object store to overlap GPU compute and network (Section 5.2.1).

- Serving throughput (Table 3)
  - Setup: co‑located client and server; residual network policy (~10 ms per inference) with small inputs and a small fully‑connected policy (~5 ms) with large inputs; compare Ray actor vs Clipper (HTTP/REST).
  - Results:
    - Small input model: Ray 6200 ± 21 states/sec; Clipper 4400 ± 15 states/sec.
    - Large input model: Ray 6900 ± 150 states/sec; Clipper 290 ± 1.3 states/sec.
  - Interpretation: when client and server are on the same machine (common in RL), Ray’s shared memory and low‑overhead serialization dramatically increase throughput (Table 3).

- Simulation throughput (Table 4)
  - Task: OpenAI Gym Pendulum‑v0 timesteps; MPI bulk‑synchronous vs Ray asynchronous tasks.
  - Results: at 1 CPU both ~22.5K steps/s; at 16 CPUs Ray reaches 290K vs 208K; at 256 CPUs Ray reaches 4.03M vs 2.16M.
  - Takeaway: asynchrony and dynamic collection (`ray.wait`) better utilize heterogeneous simulation durations at scale.

C. End‑to‑end RL applications (Figure 14)
- Evolution Strategies (ES) (Section 5.3.1; Figure 14a)
  - Workload: broadcast a policy, collect ~10K simulation tasks per iteration, aggregate results; scale to thousands of cores.
  - Ray implementation adds hierarchical aggregation using actors to avoid driver bottlenecks.
  - Results:
    - Scales to 8192 CPUs; median time to reach target score is 3.7 minutes—over 2× faster than the best published result (10 minutes).
    - Strong scaling: doubling cores yields ~1.6× speedup on average.
    - The reference specialized system fails to complete at 2048 cores due to driver overload, whereas Ray succeeds.
  - Evidence of programmability: initial parallelization changed only 7 lines of code; hierarchical optimization expressed naturally using nested tasks/actors.

- Proximal Policy Optimization (PPO) (Section 5.3.2; Figure 14b)
  - Pattern: asynchronous scatter‑gather—assign rollouts to simulator actors until 320K steps are collected; update policy with 20 SGD steps, batch size 32,768; ~350 KB parameters.
  - Results: Ray outperforms an optimized MPI implementation across CPU×GPU configurations while using fewer GPUs (e.g., MPI needs 1 GPU/8 CPUs; Ray needs at most 8 GPUs across the cluster).
  - Mechanisms: resource‑aware scheduling of CPU‑only simulation on cheaper CPU nodes; TensorFlow’s multi‑GPU in a single process with objects pinned in GPU memory; asynchronous rollouts gathered efficiently. Section 5.3.2 notes total cost reductions by up to 18× when combining CPU/GPU asymmetry with spot instances and Ray’s fault tolerance.

Assessment
- The experiments are broad (micro to end‑to‑end), concrete (with numbers, figures, and ablations), and tied to design claims. Particularly persuasive are:
  - Scheduler scalability and the allreduce latency ablation (Figures 8b, 12b), directly connecting design to performance.
  - Serving and simulation comparisons (Table 3, Table 4), isolating the benefits of shared memory and asynchrony.
  - ES/PPO case studies (Figure 14) demonstrating both performance and programmability benefits.
- Caveats/nuance:
  - For small-message allreduce, OpenMPI is faster (Figure 12a), showing Ray’s ring reduce is not yet optimal for all regimes.
  - Some results depend on co‑location assumptions (Table 3); over the network, general‑purpose serving systems may fare better given their broader feature sets.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Objects are immutable and fit on a single node’s object store (Section 4.2.3). Very large, truly distributed objects (e.g., matrices sharded across nodes) must be decomposed into collections of futures at the application level.
  - Ray is not a replacement for full-fledged analytics/serving platforms; it lacks high-level data APIs, model lifecycle management, and features like query optimization or model versioning (Section 1, caveats).
- Performance trade‑offs
  - Centralizing metadata in the GCS, while sharded and replicated, introduces a dependency on Redis and careful memory management; long‑running jobs require lineage garbage collection or periodic “GCS flushing” to cap memory (Figure 10b; Section 7).
  - Actor fault tolerance requires application‑level checkpoints; without them, recovery can be slow due to method replay (Figure 11b).
  - For workloads with very small messages or extremely low-latency collectives, specialized libraries (OpenMPI/NCCL) can outperform Ray (Figure 12a).
  - Scheduling decisions are made without full knowledge of future graph structure; achieving optimal placement may require runtime profiling and hints (Section 7, “Limitations”).
- Functionality gaps and complexity
  - The API is intentionally minimal; higher‑level libraries are not part of the core and must be built on top (Section 7).
  - While components are stateless, overall correctness depends on accurate lineage; lineage explosion must be controlled via GC policies (Section 7).
  - The object store uses LRU eviction; application‑aware eviction or QoS for mixed workloads is not discussed.

## 7. Implications and Future Directions
- Field impact
  - By making simulation, training, and serving first‑class workloads on one runtime, Ray lowers the systems burden for RL research and deployment. This enables rapid prototyping of complex algorithms (A3C, PPO, DQN, ES, DDPG, Ape‑X were implemented with tens of lines of Ray code; Section 7) and encourages algorithmic designs that exploit dynamic graphs and resource heterogeneity.
  - The control‑plane design—decoupling lineage/object metadata into a sharded, fault‑tolerant store while keeping all other components stateless—offers a blueprint for future distributed runtimes targeting fine‑grained, dynamic workloads.

- Practical applications
  - Robotics and real‑time control: low‑latency serving and fast simulation turnaround are essential.
  - Large‑scale simulation‑based search (e.g., games): hierarchical aggregation and asynchronous rollouts scale naturally (Figure 14).
  - Distributed training pipelines that must co‑locate data preprocessing, simulation, and learning (Figures 1–3).

- Suggested follow‑ups and enhancements
  - Richer APIs and libraries on top of Ray for common RL patterns (e.g., experience replay buffers, distributed rollout orchestration) that can inform scheduling decisions (Section 7).
  - Scheduler enhancements: integrate runtime profiling, predictive placement, and application hints; further reduce tail latency for collective operations (Figure 12b suggests sensitivity).
  - Memory and lineage management: automated lineage GC policies and programmer‑friendly annotations (e.g., methods that do not mutate actor state) to reduce recovery cost (Section 7).
  - Distributed objects: first‑class support for sharded objects with consistency models suitable for RL data structures.
  - Serving beyond co‑location: connectors that combine Ray’s object store with external serving front ends for WAN clients while preserving low‑latency intra‑cluster paths.

Quotes anchoring key claims/results
- Contributions list (Section 1):
  > “We design and build the first distributed framework that unifies training, simulation, and serving… We unify the actor and task-parallel abstractions… [and] propose a system design principle in which control state is stored in a sharded metadata store and all other system components are stateless. [We] propose a bottom-up distributed scheduling strategy.”

- Scalability (Figure 8b):
  > “Ray reaches 1 million tasks per second throughput with 60 nodes … [and] continues to scale linearly beyond 1.8 million tasks per second at 100 nodes.”

- Serving throughput (Table 3):
  > “Small Input: Clipper 4400 ± 15 states/sec; Ray 6200 ± 21 states/sec. Larger Input: Clipper 290 ± 1.3 states/sec; Ray 6900 ± 150 states/sec.”

- ES end‑to‑end result (Figure 14a):
  > “The Ray ES implementation… achieves a median time of 3.7 minutes, over twice as fast as the best published result [10 minutes]. The special-purpose system failed to run beyond 1024 cores.”

- Allreduce sensitivity (Figure 12b):
  > “Ray’s low-latency scheduling is critical for allreduce” (iteration time nearly doubles with a few ms of added scheduler latency).

Together, these mechanisms and results show not just that Ray is fast, but how its design—unified API, decoupled control store, bottom‑up scheduling, and in‑memory data—produces that performance and flexibility for emerging AI applications.
