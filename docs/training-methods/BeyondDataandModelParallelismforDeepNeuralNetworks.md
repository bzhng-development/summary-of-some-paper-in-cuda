# Beyond Data and Model Parallelism for Deep Neural Networks

**ArXiv:** [1807.05358](https://arxiv.org/abs/1807.05358)

## 🎯 Pitch

This paper introduces FlexFlow, a deep learning framework that automatically finds optimal parallelization strategies for DNN training by exploring the broad SOAP search space—comprising sample, operation, attribute, and parameter dimensions—well beyond conventional data and model parallelism. By combining a fast, accurate execution simulator with a guided MCMC search, FlexFlow delivers up to 3.8× faster training throughput and superior scalability on real-world workloads, transforming the efficiency of large-scale DNN training on modern hardware.

---

## 1. Executive Summary
FlexFlow proposes a general way to parallelize deep neural network (DNN) training that goes beyond the usual “data parallelism” and “model parallelism.” It defines a broad search space (called SOAP) and uses a fast execution simulator with a Markov Chain Monte Carlo (MCMC) search to automatically find efficient strategies for a given hardware setup. Across six real models on two GPU clusters, FlexFlow delivers up to 3.8× higher training throughput and scales better, while keeping search time seconds-to-minutes (Sections 1, 8.2, Figures 7, 10; Abstract).

## 2. Context and Motivation
- Problem addressed
  - Choosing how to split and place DNN computation across multiple devices is now essential, but common strategies are limited:
    - Data parallelism: replicate the whole model on each device and split the batch; synchronize parameters each iteration (Section 1).
    - Model parallelism: split layers/operations across devices; eliminates parameter sync but restricts parallelism within an operation and introduces inter-op data transfers (Section 1).
  - Both are often suboptimal, especially for layers with many parameters (e.g., fully-connected) or complex graphs (Section 1).

- Why this matters
  - Training time scales poorly with model size and dataset size; parallelization is necessary in practice (Section 1).
  - Inefficient parallelization wastes expensive GPU time and network bandwidth and limits scalability (Sections 1, 8.2).

- Prior approaches and gaps
  - Hand-crafted hybrids (e.g., data parallel for conv, model parallel for dense) improve over naive splits but still leave performance on the table (Section 1; baselines described in Section 8.2).
  - Automated systems with restricted search spaces:
    - REINFORCE learns device placement but only within model-parallel placements; each candidate must run on hardware, taking 12–27 hours and large clusters to search (Section 1; Section 8.2.3).
    - OptCNN exploits intra-op parallelism but assumes linear graphs and no inter-op parallelism; cannot exploit concurrent branches (Sections 2, 8.2.3).

- Positioning
  - FlexFlow enlarges the search space to include parallelism across samples, operations, and multiple intra-operation dimensions; then it makes this search tractable with a high-speed, accurate simulator and MCMC search (Sections 3–6).
  - It integrates a runtime (built on Legion) that can realize any strategy from this space at per-operation granularity (Section 7).

## 3. Technical Approach
FlexFlow’s method has four layers: problem formalization (SOAP), strategy representation, performance simulation, and search.

- Formalizing the search space (SOAP, Section 4)
  - Represent the DNN as an `operator graph` G: nodes are operations (e.g., convolution, matrix multiply), edges are tensors flowing between operations (Section 3.1).
  - Represent the hardware as a `device topology` D: devices (e.g., CPU, GPU) and their links with bandwidth/latency (Figure 2; Section 3.1).
  - For each operation `o_i`, define its parallelizable output dimensions `P_i` (Table 1):
    - `Sample` dimension (S): across input examples; always available.
    - `Parameter` dimensions (P): splitting requires partitioning learned parameters (e.g., output channels in conv; rows/cols in matmul).
    - `Attribute` dimensions (A): data attributes that don’t split parameters (e.g., spatial height/width in conv).
  - Parallelization configuration `c_i` for operation `o_i`:
    - Specifies how many partitions (degree) in each available dimension and which device runs each resulting task (Figure 3 and Figure 4).
    - Produces `|c_i|` independent tasks `t_{i:1} … t_{i:|c_i|}` with equal-sized output partitions for load balance (Section 4).
  - A complete `strategy S` assigns a configuration `c_i` to every operation; operations’ configs are independent choices (Section 4).

- Execution simulator (Section 5; Algorithms 1–2; Figures 5, 11)
  - Purpose: estimate end-to-end iteration time for a candidate strategy S on hardware D, without actually running it, enabling fast search (Section 5).
  - Core assumptions (A1–A4, Section 5):
    - A1: Task execution time is predictable with low variance and depends on tensor sizes and op type, not tensor contents.
    - A2: Communication time is tensor size s divided by link bandwidth b (i.e., fully utilized).
    - A3: Each device schedules tasks FIFO.
    - A4: Runtime overhead is negligible; tasks start when inputs are ready and the device is free.
  - Task graph construction (Section 5.1; Table 2; Figures 5a–b):
    - Nodes: computation tasks (from op partitions) and communication tasks (treat each hardware link as a “communication device”).
    - Edges: dependencies (finish-to-start constraints). Data movement is explicit via communication tasks.
    - Each task gets an `exeTime`:
      - Computation: measure once per (op type, output size, device) and cache (leveraging that DNNs reuse few op types; Section 5).
      - Communication: s/b from link bandwidth (A2).
  - Simulation algorithms:
    - Full simulation (Algorithm 1; Figure 5c):
      - A priority queue by `readyTime` repeatedly schedules the next ready task, respecting FIFO on each device (A3) and propagating ready times downstream.
      - Returns the max `endTime` as predicted iteration time.
    - Delta simulation (Algorithm 2; Figure 5d):
      - Most search proposals only change one operation’s config; reuse the previous timeline and incrementally update only affected tasks.
      - Maintains FIFO per device and propagates changes in `readyTime` and `startTime`.
      - Produces identical timelines to full simulation but much faster (Section 5.3; Table 4).

- Search procedure (Sections 6.1–6.2; Figure 2)
  - Objective: minimize simulated execution time (cost).
  - MCMC with Metropolis–Hastings:
    - Convert cost to a sampling distribution: `p(S) ∝ exp(-β · cost(S))` (Eq. 1, Section 6.1).
    - Proposal: pick a random operation and replace its config with a random valid config (symmetric proposal; Section 6.2).
    - Acceptance: always accept if cost improves; otherwise accept with probability `min(1, exp(β · (cost(S) − cost(S*))))` (Eq. 2, Section 6.1).
    - Initialize from diverse seeds (data-parallel strategy and random ones) and run until a time budget or when no improvement for half the budget (Section 6.2).
  - Why MCMC? It behaves like greedy descent when improvements exist but can escape local minima, critical in a combinatorial space that is exponential in graph size (NP-hard by reduction from minimum makespan; Section 6).

- Runtime to execute the found strategy (Section 7)
  - Built on the Legion runtime; uses cuDNN/cuBLAS for kernels.
  - Supports hybrid intra-op partitioning across any combination of S, A, P for each op and fine-grained control at per-op level (Section 7).
  - This capability is what prior DL frameworks generally lack (they mostly support S-only partitioning, i.e., data parallelism; Section 7).

## 4. Key Insights and Innovations
- SOAP search space (Section 4; Figure 1)
  - Innovation: Unifies parallelism across Samples (S), Operations (O), Attributes (A), and Parameters (P), including hybrid intra-op splits (e.g., split by both batch and channels).
  - Why significant: Existing systems consider at most S (data parallelism), or O/P (model parallelism), or partially S/A/P without O (OptCNN). FlexFlow’s broader space contains these as special cases and exposes many faster strategies.

- High-speed, accurate execution simulation (Sections 5, 8.3; Figure 11; Table 4)
  - Innovation: Predict performance without running; measure each op type/size once, cache, and build end-to-end timelines including explicit comms and device FIFO.
  - Accuracy: For all measured runs across six models and two clusters, simulated vs. real execution times differ by <30%, and ordering of candidate strategies is preserved (Section 8.3.1; Figure 11).
  - Speed: Delta simulation accelerates end-to-end search 2.2–6.9× over full simulation, with larger gains as cluster size grows (Table 4). This makes seconds-to-minutes searches feasible.

- MCMC-guided exploration with incremental re-simulation (Sections 5–6; Figure 12)
  - Innovation: A simple, hardware-agnostic search loop that only needs a runtime oracle (the simulator), not analytic models or hand-tuned heuristics about compute/comm balances.
  - Significance: Enables automatic discovery of non-intuitive strategies that mix inter- and intra-op parallelism matched to the topology (e.g., preferring adjacent K80 GPUs to reduce PCIe hops; Section 8.5).

- A runtime capable of any SOAP strategy at per-operation granularity (Section 7)
  - Innovation: Fine-grained control to realize hybrid splits (S/A/P) per op—a capability largely absent in mainstream DL frameworks.
  - Significance: Turns the found strategies into real speedups (e.g., 1.3–3.3× throughput gains across models; Figure 7).

## 5. Experimental Analysis
- Setup (Section 8.1; Figure 6; Table 3)
  - Models: AlexNet, Inception-v3, ResNet-101 (CNNs); RNNTC (text classification), RNNLM (language modeling), NMT (machine translation). Standard datasets and metrics (Table 3).
  - Hardware: Two GPU clusters.
    - P100 cluster: 4 nodes, each with 4× Tesla P100 connected by NVLink; nodes over 100 Gb/s InfiniBand (Figure 6a).
    - K80 cluster: 16 nodes, each with 4× K80 via PCIe; nodes over 56 Gb/s InfiniBand (Figure 6b).
  - Training: Synchronous training; batch size 64 (AlexNet uses 256); default optimizer settings follow prior work (Section 8.1).
  - Search: 30-minute budget; seeded with data-parallel and a random strategy (Section 8.1).

- Baselines and comparators (Sections 2, 8.2)
  - Data parallelism (implemented in FlexFlow; matched or exceeded TF/PyTorch baselines to remove framework confounds; Section 8.2.1).
  - Expert-designed hybrids: data-parallel conv/pool + model-parallel dense for CNNs; cross-node data parallel + intra-node model parallel for RNNs (Section 8.2.1).
  - Automated frameworks: REINFORCE (device placement for model parallelism) and OptCNN (intra-op parallelism, linear graphs) (Section 8.2.3).

- Main quantitative results
  - Throughput vs. devices (Figure 7):
    - FlexFlow achieves 1.3–3.3× speedups over data parallelism and expert heuristics across most models and scales; for ResNet-101 it converges to near data parallelism (similar performance) since that is near-optimal for that architecture.
  - Communication and computation breakdown (NMT on 64 K80 GPUs; Figure 8):
    - > Per-iteration time (Figure 8a): FlexFlow reduces runtime by 1.7–2.4× compared to baselines.
    - > Data transfers per iteration (Figure 8b): 65.8 GB (data parallelism) and 24.2 GB (expert) drop to 12.1 GB with FlexFlow (≈2–5.5× reduction).
    - > Total task compute time (Figure 8c): FlexFlow is ~20% lower than data parallelism (35.7 s → 28.7 s) and comparable to expert’s 28.2 s, but without the expert approach’s load imbalance.
    - Mechanistic explanation (Section 8.2.1): FlexFlow often parallelizes parameter-heavy ops in channel (P) rather than batch (S), reducing both compute and synchronization.
  - End-to-end training time (Figure 9):
    - For Inception-v3 to 72% top-1 single-crop on ImageNet: FlexFlow shortens wall-clock training time by 38% relative to TensorFlow’s data-parallel baseline.
  - Against REINFORCE (Figure 10a; Section 8.2.3):
    - 4 K80 GPUs (single node): FlexFlow’s strategies yield 3.4–3.8× higher throughput.
    - Search efficiency: REINFORCE needs 12–27 hours and up to 160 nodes to search; FlexFlow finds better strategies in 14–40 seconds on a single node (Section 8.2.3).
  - Against OptCNN (Figure 10b; Section 8.2.3):
    - For non-linear graphs (Inception-v3, RNNTC, RNNLM, NMT), FlexFlow is 1.2–1.6× faster by exploiting inter-op parallelism among branches in addition to intra-op partitioning.
    - For linear CNNs (AlexNet, ResNet), both find similar best strategies.
  - Simulator accuracy and search speed (Section 8.3):
    - Accuracy: Simulated vs. real runtime differs by <30% across devices/models; candidate ordering preserved (Figure 11).
    - Speed: Delta vs. full simulation makes search 2.2–6.9× faster end-to-end; e.g., Inception-v3 on 64 GPUs: 8817 s → 1278 s (6.9×) (Table 4).
    - Search trajectory: With delta simulation, better strategies are found earlier (Figure 12).
  - Search optimality checks (Section 8.4):
    - Global optimal found on small problems (LeNet; a reduced-step RNNLM), validated via exhaustive/A* search (took 0.8 h and 18 h offline).
    - Locally optimal (no improving single-op neighbor) for all six DNNs on 2/4/8 devices.

- Case studies: what strategies look like (Section 8.5)
  - Inception-v3 on 4 P100 GPUs (Figure 13):
    - Mixes intra-op parallelism on the critical path and inter-op parallelism across branches; reduces parameter synchronization by 75% and total iteration time by 12% relative to data parallelism.
    - On K80 topology, prefers adjacent GPUs due to PCIe layout (communication-aware placement).
  - NMT on 4 P100 GPUs (Figure 14):
    - Embedding layers (many parameters, little compute): run on fewer devices to cut synchronization.
    - Softmax layers (many parameters, heavy compute): split by channel (P) to balance compute while reducing sync.
    - LSTM + attention: combine inter-layer concurrency and intra-op partitioning to reduce sync and maintain balance.

- Do the experiments support the claims?
  - The breadth (six models; two disparate topologies) and multiple baselines (manual and automated) make a strong case.
  - The mechanistic evidence (compute vs. comm breakdown, case studies) links design to outcomes (Figures 8, 13, 14).
  - Simulator validation (Figure 11) and search speed (Table 4) substantiate feasibility.

## 6. Limitations and Trade-offs
- Core assumptions for simulation (Section 5)
  - A1: Op runtime predictability and data-independence. This holds for dense linear-algebra-heavy ops but may fail for data-dependent control flow or highly irregular sparsity.
  - A2: Full link utilization (s/b). Real systems may suffer contention or protocol overheads, especially at cluster scale.
  - A3: FIFO device scheduling. Some backends/devices may reorder work or overlap kernels in ways that deviate from FIFO.
  - A4: Negligible runtime overhead. For very fine-grained tasks, runtime overheads could matter.

- Search-space and strategy constraints
  - Equal-size partitions per dimension (Section 4) simplify balancing but prevent exploring imbalanced splits that might be better under heterogeneity or stragglers.
  - The search is heuristic MCMC; no global optimality guarantees for large problems (though Section 8.4 shows global/ local optimality in tests).

- Applicability
  - Best suited to static computation graphs with a small set of op types (Section 5). Dynamic graphs with frequent shape changes would require repeated calibration.
  - Focuses on synchronous training; does not study asynchronous or pipeline-parallel training across micro-batches.

- Engineering and environment assumptions
  - Accuracy hinges on the initial per-op calibration on each device type; cross-version kernel or driver changes could alter timings.
  - Communication model assumes independent, dedicated links; complex NUMA effects or shared-bus congestion can reduce fidelity.

## 7. Implications and Future Directions
- How this changes the landscape
  - Recasts parallelization as a general optimization problem over a rich space (SOAP), rather than a choice between canned templates (data vs. model parallelism). This opens the door to hardware-aware, model-aware strategies discovered automatically (Sections 3–6, Figure 1).
  - Demonstrates that accurate, fast simulation can replace costly on-hardware evaluation during search, slashing discovery time from hours to seconds (Sections 5, 8.3).

- Practical applications
  - Turnkey performance tuning for training on any multi-GPU/multi-node cluster; useful for ML platforms that serve many teams and models.
  - Reduced communication and better scaling can lower training costs and carbon footprint, especially for parameter-heavy models.

- Research avenues enabled
  - Integrate memory and activation-checkpointing decisions into the SOAP space, trading compute for memory under GPU RAM constraints.
  - Extend to pipeline parallelism and micro-batching to cover long sequential models (e.g., large transformers), coordinating with S/A/P splits.
  - Learn proposal distributions for MCMC (or employ Bayesian optimization/RL) seeded by simulator feedback to accelerate convergence.
  - Enhance the simulator to model contention, non-FIFO scheduling, or kernel fusion, improving fidelity for next-gen accelerators.
  - Support dynamic graphs and conditional computation by fast on-the-fly calibration and online adaptation.

- Broader impact
  - By automating low-level performance engineering, FlexFlow can make advanced parallelization accessible to non-experts, increasing productivity and portability across hardware (Section 3.1, “Programmability” and “Portability”).

> Key numbers to remember:
> - Up to 3.8× higher throughput over state-of-the-art baselines (Abstract; Figure 10a).
> - 2–5.5× less communication on NMT at 64 GPUs; 1.7–2.4× faster per-iteration (Figure 8).
> - 38% shorter end-to-end training time for Inception-v3 to a fixed accuracy (Figure 9).
> - Simulator within 30% of real runtime, preserves ordering (Figure 11); search sped up 2.2–6.9× by delta simulation (Table 4), with seconds-to-minutes total search time for small to medium scales (Section 8.3.2).
