# Beyond Data and Model Parallelism for Deep Neural Networks

**ArXiv:** [1807.05358](https://arxiv.org/abs/1807.05358)

## 🎯 Pitch

This paper presents FlexFlow, a deep learning system that automatically discovers highly efficient parallelization strategies for training deep neural networks by exploring a broad search space—encompassing Sample, Operation, Attribute, and Parameter (SOAP) dimensions—far surpassing traditional data and model parallelism. By leveraging a novel, orders-of-magnitude faster execution simulator, FlexFlow can rapidly and accurately predict performance for candidate strategies, enabling up to 3.8× faster training with reduced communication and improved scalability on real-world GPU clusters—redefining the frontier of scalable deep learning.

---

## 1. Executive Summary
This paper introduces FlexFlow, a system that automatically discovers fast ways to parallelize deep neural network (DNN) training across multiple devices by exploring a much larger search space than standard data or model parallelism. Its key enabler is a fast execution simulator that predicts the runtime of a parallelization plan within seconds, allowing FlexFlow to search broadly and find strategies that improve training throughput by up to 3.8× while reducing communication and improving scalability (Abstract; Section 8).

## 2. Context and Motivation
- Problem addressed
  - Modern DNNs and datasets require multi-device training, but common strategies—data parallelism and model parallelism—often underperform. Data parallelism struggles with layers having many parameters (e.g., large fully connected layers) due to heavy synchronization; model parallelism reduces synchronization but limits concurrency and adds inter-layer communication (Introduction, p.1).
  - Existing automated methods explore narrow subspaces: some choose only how to split across operations (placement for model parallelism), others only within a single operation (e.g., how to split a convolution), missing faster combinations that mix both (Introduction; Related Work, Fig. 1).
- Why this matters
  - The gap translates to wasted compute and communication, especially at scale. Faster parallelization reduces training time and cost, and improves scalability on modern GPU clusters (Introduction; Section 8).
- Prior approaches and their limits
  - Data parallelism: efficient on compute-heavy layers with few parameters (e.g., convolutions), but suboptimal on parameter-heavy layers (Introduction).
  - Model parallelism: removes parameter synchronization but forces serial dependencies and limits intra-op parallelism (Introduction).
  - Expert-designed hybrids (e.g., “one weird trick” that uses data parallelism for conv layers and model parallelism for fully connected layers): better than pure data/model parallelism but still far from optimal and hard to port across clusters (Related Work; Section 8.2.1).
  - Automated methods:
    - REINFORCE-style device placement: optimizes operation placement only; requires running each candidate plan for reward signals, making exploration slow (Related Work; Section 8.2.3).
    - OptCNN: optimizes how to split operations in linear graphs only; assumes no inter-operation overlap, missing parallelism across branches (Related Work; Section 8.2.3).
- How this paper positions itself
  - Defines a broader parallelization search space called SOAP (Sample, Operation, Attribute, Parameter) that subsumes past strategies (Section 4; Fig. 1).
  - Introduces a fast, accurate simulator to evaluate any plan without executing it, enabling broad, guided search (Sections 5–6).
  - Implements a runtime capable of executing any plan in the SOAP space (Section 7).

## 3. Technical Approach
At a high level (Fig. 2), FlexFlow takes two inputs: an `operator graph` (nodes are operations; edges are tensors) and a `device topology` (devices and interconnects annotated with latency/bandwidth), and returns an executable parallelization plan.

A. The SOAP search space (Section 4)
- Goal: Describe all ways to split work both within and across operations, and map those splits to devices.
- Key concepts
  - `Parallelizable dimensions (P_i)` of an operation’s output tensor:
    - Always includes a `sample` dimension (batch axis).
    - Optional `parameter` dimensions: splitting requires partitioning the model’s learned weights (e.g., output channels of a convolution or columns/rows of a matrix multiplication).
    - Optional `attribute` dimensions: splitting does not partition learned parameters (e.g., image height/width in 2D conv). Table 1 lists examples.
  - `Parallelization configuration (c_i)` for operation `o_i`:
    - Specifies a degree of parallelism along each chosen dimension (e.g., how many ways to split along samples, channels, height/width).
    - Imposes equal-sized partitions to balance loads (Section 4).
    - Assigns each resulting task (`t_i:1, …, t_i:|c_i|`) to a device (Fig. 4).
  - `Operation` dimension: parallelism across different operations (e.g., executing independent branches concurrently).
- Why this is different
  - Supports hybrid intra-op splitting across multiple dimensions simultaneously (e.g., split an op along both samples and channels), plus inter-op overlap—most prior systems consider only one of these (Fig. 1; Fig. 3).
- Example
  - 1D convolution: can split along `sample` (S), `length` (A), and `channel` (P) dimensions, including combinations (Fig. 3).
  - Matrix multiplication `Y = W X`: can split along sample and channel-like parameter axes; FlexFlow infers required input slices for each task (Fig. 4).

B. Execution simulation (Section 5)
To search broadly, FlexFlow must rapidly evaluate the runtime of any plan. It does this via a simulator under four explicit assumptions (A1–A4, Section 5):
- A1: Task execution time is predictable, low-variance, and independent of input contents.
- A2: Link transfers achieve full bandwidth: time = size/bandwidth.
- A3: Each device schedules tasks FIFO.
- A4: Runtime overhead is negligible; a device starts a task as soon as inputs and the device are ready.

How the simulator models a plan:
1) Build a `task graph` (Section 5.1; Fig. 5):
   - Create one `compute task` per partitioned operation output (from `c_i`).
   - Represent every inter-device data move as a `communication task` placed on a distinct “communication device” representing that physical link (lets compute and communication overlap).
   - Add dependency edges so a task starts only after all inputs are ready; data dependencies manifest either as same-device edges (no comms) or via explicit communication tasks between devices.
   - For each task, record:
     - `exeTime` (measured once for each op type and output size on a given device and cached; Section 5.1),
     - placement device/link,
     - predecessors `I(t)` and successors `O(t)`.
2) Simulate the schedule:
   - Full simulation (Algorithm 1; Fig. 5c):
     - Use a global priority queue ordered by `readyTime`. When a task is dequeued, set `startTime = max(readyTime, endTime of previous task on same device)` to enforce FIFO (A3), then `endTime = startTime + exeTime`.
     - Propagate updated `readyTime` to successors and continue until done; the final runtime is `max(endTime)` across tasks.
   - Delta simulation (Algorithm 2; Fig. 5d):
     - When the search proposes a small change (e.g., modifying `c_i` for one operation), update only the affected portion of the graph and re-propagate times through successors and the next tasks on the same device, maintaining FIFO. This yields identical timelines to full simulation but much faster incremental updates.

C. Search procedure (Section 6)
- Goal: Minimize simulated runtime over the SOAP space. The exact optimum is NP-hard (reduction from minimum makespan; Section 6).
- Strategy: Markov Chain Monte Carlo (MCMC) with Metropolis–Hastings acceptance (Sections 6.1–6.2).
  - Probability model: `p(S) ∝ exp(-β · cost(S))` where `cost(S)` is simulated runtime (Eq. (1)).
  - Proposal: pick a random operation and replace its `c_i` with a random valid configuration (symmetry ensures `q(S|S*)=q(S*|S)`, so acceptance is Eq. (2)): accept always if runtime improves; accept with probability `exp(β·(Δ improvement))` otherwise, allowing escape from local minima.
  - Stopping: per-initialization time budget or when no improvement for half the budget (Section 6.2).
  - Initialization: tries both standard strategies (e.g., data parallelism, expert heuristics) and random strategies to diversify starting points (Section 6.2).
- Why MCMC: The search space is exponential in the number of operations (Section 6) and mixes discrete partition choices with device placements; MCMC provides a principled, hill-climbing-with-occasional-escape exploration without needing differentiable objectives or expensive full evaluations (the simulator handles fast evaluations).

D. Executable runtime (Section 7)
- Implemented in the Legion runtime with cuDNN/cuBLAS for kernel execution.
- Capabilities:
  - Parallelize any operation along any combination of its parallelizable dimensions (SOAP) at per-operation granularity.
  - Control task partitioning and placement to match the discovered plan (Section 7).

## 4. Key Insights and Innovations
1) A unified, richer parallelization space (SOAP) that allows hybrid intra-operation partitioning and inter-operation concurrency (Section 4; Fig. 1–4).
   - Why it matters: It subsumes data parallelism, model parallelism, and expert hybrids as special cases, exposing strategies that reduce synchronization and better balance computation.

2) A fast, measurement-grounded simulator with delta updates (Sections 5.1–5.3).
   - Novelty:
     - Measures each operator’s `exeTime` once per output shape/device and caches it, leveraging the fact that most models reuse a few operator types and that execution time is largely data-insensitive (Section 5.1).
     - Represents links as “communication devices” to model compute/communication overlap (Section 5.1).
     - Delta simulation reuses prior timelines, yielding 2.2–6.9× faster search than full simulation as devices scale (Table 4), yet preserves exact ordering and final timeline (Section 5.3).
   - Impact: Enables exploration of a far larger space in seconds rather than hours.

3) A guided randomized search that is both general and efficient (Sections 6.1–6.2).
   - Difference from prior automation: REINFORCE/placement methods must execute each plan to get rewards and only explore operation placement; FlexFlow uses simulated cost to evaluate plans that include intra-operation partitioning and achieves >3× speedups in minutes (Section 8.2.3; Fig. 10a).

4) A practical runtime supporting per-operation hybrid parallelism (Section 7).
   - Significance: Most production frameworks only split along batch (data parallelism) and require manual device placement for model parallelism; FlexFlow can implement the mixed strategies discovered by its search (Section 7).

## 5. Experimental Analysis
Evaluation setup
- Models and datasets (Table 3):
  - CNNs: AlexNet (synthetic), Inception-v3 (ImageNet), ResNet-101 (ImageNet).
  - RNNs: RNNTC (Movie Reviews), RNNLM (Penn Treebank), NMT (WMT En–De). Hidden sizes and unrolling steps match prior work; RNN unrolling set to 40 (Section 8.1).
- Clusters (Fig. 6):
  - P100 cluster: 4 nodes×4 GPUs, intra-node NVLink, inter-node 100 Gb/s IB.
  - K80 cluster: 16 nodes×4 GPUs, PCIe switches, inter-node 56 Gb/s IB.
- Training setup:
  - Synchronous training, batch size 64 (AlexNet uses 256), standard hyperparameters per prior work (Section 8.1).
  - Search time budget typically 30 minutes, though most searches converge in minutes (Section 8.1; Fig. 12).
- Baselines:
  - Data parallelism implemented in TensorFlow, PyTorch, and FlexFlow; FlexFlow’s data-parallel runs match or exceed the others, so reported as baseline (Section 8.2.1).
  - Expert-designed hybrid strategies for CNNs and RNNs from prior literature [27, 42] (Section 8.2.1).
  - Automated methods: REINFORCE device placement (no public code; results reproduced from their paper) and OptCNN (linear graphs only) (Section 8.2.3).

Main results
- Per-iteration throughput (Fig. 7):
  - FlexFlow improves training throughput by 1.3–3.3× over data parallelism and expert-designed strategies across five of six models; ResNet-101 is similar to data parallelism except a single FC layer uses model parallelism (Section 8.2.1).
- Communication and computation (NMT, 64 K80 GPUs; Fig. 8):
  - Quote: “FlexFlow reduces the per-iteration data transfers by 2–5.5×” compared to other approaches (Fig. 8b).
  - FlexFlow also reduces total compute time per iteration by ~20% compared to data parallelism (Fig. 8c). Expert strategy has slightly lower compute time but worse overall runtime due to imbalance and disabled intra-op parallelism; FlexFlow wins on wall-clock (Fig. 8a).
- End-to-end training time (Fig. 9):
  - For Inception-v3 on 16 P100 GPUs, to reach 72% top-1 single-crop accuracy, FlexFlow shortens end-to-end training time by 38% relative to TensorFlow (Section 8.2.2).
- Against automated baselines (Fig. 10):
  - REINFORCE (4 K80s, single node): FlexFlow achieves 3.4–3.8× higher throughput, and finds the strategy in 14–40 seconds versus REINFORCE’s 12–27 hours that required up to 160 nodes for search (Section 8.2.3).
  - OptCNN (16 P100s): For non-linear graphs (Inception-v3, RNNTC, RNNLM, NMT), FlexFlow delivers 1.2–1.6× throughput gains by exploiting inter-operation parallelism that OptCNN’s linear dynamic program cannot (Fig. 10b).
- Simulator accuracy and speed (Fig. 11; Table 4; Fig. 12):
  - Accuracy: Across six models and both clusters, simulated vs. actual runtimes differ by less than 30% and, crucially, preserve the ordering of strategies (Fig. 11).
  - Speed: Delta simulation reduces end-to-end search time by 2.2–6.9× versus full simulation, with larger gains at larger device counts (Table 4). Example: For NMT on 16 P100s, full search completes in 16 minutes; delta completes in 6 minutes (Fig. 12).
- Optimality checks (Section 8.4):
  - For small problems (LeNet on 4 GPUs; a short unrolled RNNLM on 4 GPUs), exhaustive search with A* pruning finds the global optimum, and FlexFlow’s search discovers it.
  - For larger cases (2, 4, 8 GPUs across all six models), the returned strategy is locally optimal among all 1-change neighbors.

Case-study insights (Section 8.5)
- Inception-v3 on 4 P100s (Fig. 13):
  - Strategy mixes intra-operation splitting on the critical path and inter-branch parallelism, reducing parameter synchronization by 75% and total iteration time by 12% versus data parallelism.
- NMT on 4 P100s (Fig. 14):
  - Embedding layers (parameter-heavy, compute-light) run on fewer GPUs to cut synchronization.
  - Softmax layers (parameter- and compute-heavy) split along channels so each GPU touches a subset of parameters, balancing load and reducing sync.
  - Recurrent and attention layers overlap across layers and within ops, jointly reducing sync while keeping balance.

Do the experiments support the claims?
- Yes, across diverse models, two hardware topologies, and multiple baselines, FlexFlow consistently delivers significant speedups, lower communication, and better scaling. The simulator’s accuracy and the optimality checks strengthen confidence that search decisions generalize to actual execution (Sections 8.2–8.4; Fig. 7–14; Table 4).
- Missing pieces: limited ablations on search hyperparameters (e.g., β, initialization diversity), and no sensitivity analysis for simulator assumptions beyond the reported accuracy bounds.

## 6. Limitations and Trade-offs
- Simulator assumptions (Section 5):
  - A1 data-independence and low-variance execution times: suitable for dense linear algebra kernels (cuDNN/cuBLAS), but not for data-dependent or sparse kernels whose runtime varies with input content.
  - A2 full link utilization: ignores congestion and protocol overheads; could overestimate throughput on contended or oversubscribed fabrics.
  - A3 FIFO device scheduling: matches many GPU queues but not all vendor/runtime behaviors (e.g., priority streams, MPS), and ignores multi-tenant interference.
  - A4 negligible runtime overhead: may break at very fine task granularities where kernel launch/coordination costs matter.
- Equal-size partitions (Section 4):
  - Ensures balance but may miss beneficial skewed partitions on heterogeneous devices or for layers with non-uniform compute/communication characteristics.
- Scope limitations
  - Focuses on synchronous training and single-job optimization; no modeling of asynchronous updates or multi-job cluster sharing.
  - Memory capacity/activation checkpointing not explicitly optimized; strategies are derived under the assumption that required slices fit device memory.
  - Evaluation limited to two GPU clusters (P100 NVLink; K80 PCIe) and to six models; results may differ on newer interconnects (e.g., NVSwitch) or very large transformer-style models.
- Search optimality and stability
  - MCMC gives no global optimality guarantee; authors mitigate with multiple initializations and show local/global optimality only on selected cases (Section 8.4).
  - β selection and proposal distribution can affect convergence; details beyond acceptance criterion are not deeply ablated.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that moving “beyond data/model parallelism” requires treating parallelization as a compiler/scheduler problem over a rich search space, not a fixed recipe. This perspective aligns with and informs emerging “3D parallelism” methods in large-scale training by providing a principled way to combine intra-op, inter-op, and tensor-dimension splits.
  - Shows that accurate, fast simulation can replace costly on-hardware profiling during search, cutting exploration from hours to seconds (Section 8.2.3; Table 4; Fig. 12).
- Practical applications
  - Auto-parallelization for production training jobs: FlexFlow can select cluster-specific strategies that reduce training time and cost without manual tuning (Sections 3.2, 8.2.2).
  - Portability across clusters: Automatically adapts to device topology differences (e.g., NVLink vs PCIe) and chooses communication-aware plans (Section 8.5, Inception-v3 case).
- Suggested follow-ups
  - Broader cost models: incorporate memory limits, activation checkpointing, energy, and monetary cost; model congestion and non-FIFO scheduling to relax A2–A4.
  - Richer search: combine MCMC with learned proposal distributions or surrogate models; add multi-objective optimization (speed, memory, energy).
  - Dynamic/adaptive strategies: re-optimize online as batch size or sequence length changes, or under multi-tenant contention.
  - Extended workloads: test on modern large transformer architectures, variable-length sequence models, sparse operations, and newer interconnects (NVSwitch, H100-class GPUs).
  - Integration with cluster schedulers: co-optimizing strategy with placement/packing for multi-job environments (Related Work mentions graph-based schedulers; Section 2).

> Bottom line: By defining the SOAP space and making it searchable via a fast simulator, FlexFlow turns parallelization into an empirical optimization problem that can be solved automatically. The empirical evidence—up to 3.8× throughput gains, 2–5.5× less communication, and 38% end-to-end time reduction for a mainstream CNN—suggests substantial practical value when moving beyond one-size-fits-all parallelism (Abstract; Fig. 7–10).
