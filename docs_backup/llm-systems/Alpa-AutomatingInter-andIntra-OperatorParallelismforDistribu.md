# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

**ArXiv:** [2201.12023](https://arxiv.org/abs/2201.12023)

## 🎯 Pitch

Alpa presents a novel compiler and runtime that fully automates the discovery and orchestration of efficient distributed training plans for large-scale deep learning models, unifying data, operator, and pipeline parallelism. By hierarchically decomposing the problem into intra-operator (tensor/weight sharding) and inter-operator (pipeline/model slicing) levels and optimizing each with tractable algorithms, Alpa eliminates the need for manual, expert-intensive parallelization—enabling scalable, out-of-the-box training of massive models across heterogeneous GPU/TPU clusters. This breakthrough accelerates both machine learning research and deployment by removing a pervasive bottleneck in scaling up model size and cluster utilization.

---

## 1. Executive Summary (2-3 sentences)
Alpa is a compiler and runtime that automatically finds and executes efficient distributed training plans for very large deep learning models by jointly combining data, operator/tensor, and pipeline parallelism. Its key idea is to recast the planning problem into two hierarchical levels—“intra-operator” (how to shard each operator across devices) and “inter-operator” (how to slice the model into pipeline stages and map them to device groups)—and to optimize each level with tractable algorithms. This matters because it removes much of the hand-engineering needed to train multi‑billion‑parameter models efficiently on GPU/TPU clusters.

## 2. Context and Motivation
- Problem addressed:
  - Efficiently training very large models requires carefully mixing several parallelization techniques: data parallelism, operator/tensor model parallelism, and pipeline parallelism. Choosing the “right” combination for a specific model and cluster can change performance by an order of magnitude, but it is hard and brittle to do by hand (§1, §2.1).
- Why it’s important:
  - State-of-the-art models (e.g., large language models, Mixture‑of‑Experts) are too big for single devices and too complex for a single parallelization strategy. Automating this planning lowers the barrier for researchers and practitioners to train large models, accelerates iteration, and improves hardware utilization (§1).
- Prior approaches and gaps:
  - Hand-tuned systems (e.g., Megatron-LM’s “3D parallelism”) prescribe a limited plan for specific model families and cluster types; they often assume repeated layers and fixed pipeline splits (§2.1, Fig. 1b).
  - Auto-parallel systems typically optimize only one dimension: intra-operator sharding (e.g., Tofu) or pipeline placement (e.g., DAPPLE), missing cross-technique synergies (§2.1, Fig. 1c–d).
  - Some solutions rely on strong assumptions (uniform models/layers, preassigned devices), or do not scale to large graphs/clusters (§2.1, §5.1).
- Positioning:
  - Alpa reframes the search space as a hierarchical composition of inter- and intra-operator parallelisms (Fig. 1e). It introduces optimization passes and a runtime that jointly plan sharding within stages and pipelining across stages on heterogeneous cluster topologies (§3, §4–§6).

## 3. Technical Approach
Alpa’s method consists of three major components (Fig. 3): an intra-operator pass, an inter-operator pass, and a runtime orchestration pass. You use a simple decorator (`@parallelize`) around your training step (Fig. 4), and Alpa traces the function, compiles, profiles, and executes the distributed plan automatically.

1) Concepts and definitions (used throughout)
- `Device mesh`: a logical 2D grid view of physical devices (e.g., GPUs) used to express collective communications along mesh axes; different logical layouts (e.g., 4×4 or 1×16) can be considered over the same physical cluster (§4.1 “Device mesh”).
- `Sharding spec`: a short code describing how a tensor is partitioned and replicated across mesh axes, e.g., `RS0` means the first tensor axis is replicated (R), the second is sharded (S) along mesh axis 0 (Table 1).
- `Resharding`: changing a tensor’s sharding layout between operators; this may trigger collectives like all‑gather, all‑reduce, or all‑to‑all (Table 2).
- `SPMD`: single program multiple data; all devices execute the same program on different tensor shards (§4).
- `1F1B`: a synchronous pipeline schedule that alternates one forward and one backward microbatch per stage and reduces memory vs. GPipe while keeping latency the same (§2.2).

2) Intra-operator pass (how to shard operators within a stage)
- Goal:
  - For every operator (node) in a stage’s computational graph, choose one parallel algorithm (a specific sharding layout and communication pattern) to minimize total time = compute + communications + resharding across edges (§4.2).
- How the search space is represented:
  - Each operator has a set of candidate parallel algorithms with known output/input sharding specs and communication costs (e.g., batched matmul alternatives in Table 3). Resharding costs between specs are enumerated (Table 2).
- Cost model and ILP formulation:
  - The objective sums per-operator compute/communication costs (`dv + cv`) and edge resharding costs (`Rvu`) (Eq. (1)). Decision variables select one algorithm per node, and pairwise choices determine resharding; quadratic terms are linearized to fit an ILP (§4.2).
  - Compute costs are set to zero in the model for tractability because heavy ops are evenly divided across devices (no redundant compute) and light ops’ compute costs are negligible (§4.2).
- Practicalities:
  - Graph simplification merges trivial ops (like element-wise and transpose) into neighbors to shrink the ILP size (§4.2).
  - After selecting sharding, Alpa applies communication‑reducing rewrites (e.g., replacing all‑reduce with reduce‑scatter + all‑gather) to realize weight‑update sharding (ZeRO) where applicable (§4.2).

3) Inter-operator pass (how to split into pipeline stages and map to meshes)
- Goal:
  - Given the full model graph and a device cluster, find a slicing into stages and assign each to a submesh such that total pipeline latency is minimized (§5.1–§5.2).
- Latency model:
  - Pipeline latency for `B` microbatches is:
    - “Fill‑and‑drain” time (sum of stage times) + “steady‑state” time for remaining microbatches (bounded by the slowest stage):
    - T* = min over stage/mesh choices of Σ ti + (B − 1) · max{ti} (Eq. (2), Fig. 5).
- Device mesh choices:
  - To ensure submeshes tile the full cluster (no idle devices), Alpa restricts candidate submesh shapes to a set that always covers the cluster (proof in Appendix A): (i) 1×(1, 2, 4, …, M) and (ii) (2..N)×M (§5.2).
- Dynamic programming (DP) over `tmax`:
  - The pass enumerates a candidate `tmax = max{ti}` (slowest stage time), then computes the minimal total Σ ti subject to each stage time ≤ `tmax`.
  - Subproblem `F(s, k, d; tmax)` = minimal time to slice operators `ok..oK` into `s` stages using `d` devices with stage times ≤ `tmax`; recurrence in Eq. (3). The final objective is Eq. (4) (§5.2).
- Stage cost queries come from the intra-op pass:
  - For each subgraph candidate (a contiguous operator span) and submesh, the intra-op pass compiles and profiles the stage to get time and memory; the result is only valid if `memstage + s · memact ≤ memdevice` under the 1F1B schedule (Eq. (5)) (§5.2).
- Scaling optimizations:
  - Early pruning of `tmax` enumeration and discretization (ε‑spacing) keeps DP tractable while bounding suboptimality by `B·ε` (§5.2 “Performance optimization #1”).
  - Operator clustering: an auxiliary DP merges neighboring light ops to reduce problem size while controlling per‑layer FLOPs; recurrence in Eq. (6) (§5.2 “Performance optimization #2”).

4) Runtime orchestration (how execution is stitched together)
- Cross‑mesh resharding:
  - Adjacent pipeline stages may use different mesh shapes and sharding specs, so their boundary tensors require many‑to‑many multicast between meshes (§6).
  - Alpa generates a two‑pass plan: (i) build P2P send/recv between source/destination tiles; (ii) when the destination has replication, rewrite to a single inter‑mesh transfer plus a fast intra‑mesh all‑gather (the “local all‑gather” optimization), shifting load to high‑bandwidth local links (Fig. 6b–c, §6).
- Execution model:
  - The runtime is MPMD (multiple programs, multiple data): each mesh receives a static instruction list for its stage(s), including allocations, compute, inter‑stage comms, and sync, avoiding centralized orchestration during steady state (§6).

5) Why these design choices?
- Hierarchical split leverages cluster structure: intra‑op sharding prefers high‑bandwidth local links (within nodes), while pipeline stage edges cross lower‑bandwidth links (across nodes). This maps naturally to typical cluster hierarchies (§1–§3).
- ILP and DP isolate two otherwise entangled problems into tractable subproblems with near‑optimal local solutions that compose well empirically (§3, §4, §5).
- The reduced submesh set trades negligible optimality loss for guaranteed full coverage and faster search (§5.2 and Appendix A).

## 4. Key Insights and Innovations
- Hierarchical parallelism decomposition (fundamental):
  - Re‑categorizes the vast plan space into intra‑operator vs. inter‑operator parallelism (Fig. 1c–e, §2.2). This reframing matches network asymmetry (fast within nodes, slower across nodes), enabling a principled mapping of sharding and pipelining to appropriate links (§1–§3).
- ILP-based auto-sharding for SPMD within stages (significant capability + performance):
  - Formulates operator‑level sharding as an ILP over enumerated algorithm choices, including resharding edges, with an communication‑aware cost model (Eq. (1), Table 2–3). This unifies data parallelism, ZeRO, and Megatron‑style tensor parallelism under one solver (§4.1–§4.2).
- DP-based joint stage slicing and mesh assignment with profiling (significant capability):
  - Introduces a latency‑aware DP that simultaneously chooses stage boundaries, mesh shapes, and feasible intra‑op plans (via queries) under memory constraints (Eq. (2)–(5), Alg. 1). Operator clustering keeps this tractable for large graphs (§5).
- Cross-mesh resharding with “local all-gather” (practical systems innovation):
  - Generalizes the equal‑mesh “scatter‑gather” trick beyond identical meshes (Fig. 6a–c). By pushing replication to the destination mesh, it reduces slow cross‑mesh traffic and leverages fast local collectives (§6).
- MPMD runtime that composes SPMD intra-op with pipelined inter-op (systems design):
  - Generates static instruction streams per mesh to avoid runtime coordination overheads while accommodating different stage shapes and programs (§6).

## 5. Experimental Analysis
- Setup:
  - Cluster: 8× p3.16xlarge nodes (64 GPUs total), NVLink within nodes; 25 Gbps cross‑node bandwidth (§8).
  - Models (Table 4): GPT‑3‑style LMs up to 39B params (FP16), GShard MoE up to 70B (FP16), Wide‑ResNet up to 13B (FP32).
  - Metric: total PFLOPS over the cluster; weak scaling (model size grows with GPU count), warm‑up then measure; variability <0.5% (§8 “Evaluation metrics”).
  - Baselines: Megatron‑LM for GPT (§8.1), DeepSpeed for MoE (§8.1), and a PP‑DP baseline (pipeline+data parallel only) for Wide‑ResNet. Also show “inter‑op only” and “intra‑op only” using Alpa (§8.1).

- Main results (Fig. 7):
  - GPT‑3 (Fig. 7a):
    - Alpa matches or slightly exceeds Megatron‑LM across 1–64 GPUs, with near‑linear or super‑linear weak scaling.
    - Insight: the best Megatron‑LM plans tend to avoid tensor parallelism except when memory‑bound; Alpa rediscovers similar strategies and additionally shards weight updates (ZeRO‑style) inside stages, explaining small gains (§8.1).
  - MoE (Fig. 7b):
    - Alpa scales across nodes and outperforms DeepSpeed substantially: 
      > “3.5× speedup on 2 nodes and 9.7× on 4 nodes” (text in §8.1 and abstract).
    - Reason: DeepSpeed combines intra‑op techniques (expert parallelism + ZeRO + tensor parallel) but lacks inter‑operator pipelining; cross‑node bandwidth becomes the bottleneck. Alpa uses inter‑op stages to contain cross‑node traffic (§8.1).
  - Wide‑ResNet (Fig. 7c):
    - Large heterogeneous CNN without manual plans: Alpa achieves good scaling; 
      > “80% linear scaling efficiency on 32 GPUs” (§8.1 and abstract).
    - Baselines run OOM (PP‑DP and inter‑op only) or fail to scale (intra‑op only), highlighting the importance of mixing stage‑level pipelining with selective operator sharding (§8.1).

- Ablation studies:
  - Intra‑op search (Fig. 8a–c):
    - ILP (“Auto‑sharding”) consistently beats heuristic, ZeRO‑2/3, and vanilla data parallel; the latter often OOMs. When gradients dominate, ZeRO variants communicate large gradient tensors every step and fall behind (§8.2).
  - Inter‑op stage DP (Fig. 9):
    - Full DP outperforms naive “equal operator” clustering and “equal layers,” especially on heterogeneous Wide‑ResNet (2.6× over equal‑operator, 1.6× over equal‑layer on 32 GPUs) (§8.3).
  - Compilation time (Fig. 10, Table 5):
    - Scales roughly linearly with model and GPU count; for GPT‑39B on 64 GPUs, total time ≈ 2393 s (≈40 min) with profiling, down from >40 hours without the accelerations (Table 5). Most time is in profiling stage‑mesh pairs; distributed compilation and a simple cost model accelerate this (§8.4).
  - Cross‑mesh resharding (Fig. 11):
    - The “local all‑gather” optimization yields ≈ 2.0× speedup on 32‑GPU Wide‑ResNet vs. naive send/recv; the “signal send/recv” curve shows the upper bound with negligible inter‑mesh payload (§8.5).
  - Case study (Fig. 12):
    - On 16 GPUs, Alpa splits Wide‑ResNet into 3 stages with 4/4/8 GPUs. Early stages favor batch‑axis partitioning (activations large), later stages shard channels/weights (weights dominate). These non‑uniform, layer‑dependent choices are difficult to design manually (§8.6).

- Assessment of evidence:
  - The evaluation spans homogeneous (Transformers) and heterogeneous (CNN) models, includes competitive baselines where available, and provides ablations isolating both passes. Reported gains are substantial (up to 9.7×) and consistent with the design rationale (placing communication on fast links, balancing memory/time across stages).
  - One caveat: the metric is PFLOPS on synthetic data; convergence behavior is unchanged by design but not empirically verified here (explicitly noted in “Evaluation metrics”).

## 6. Limitations and Trade-offs
- Modeling and search assumptions:
  - Cross‑stage communication is not modeled in the ILP/DP cost; the paper argues it is small compared to intra‑stage collectives, but this may not hold for all architectures/datasets (§7).
  - Microbatch count `B` is a hyperparameter, not optimized inside the DP; different `B` affects both memory and pipeline bubbles (§7).
  - The pipeline schedule is fixed (synchronous 1F1B); dynamic or branch‑parallel schedules are out of scope (§7).
  - The approach targets static computation graphs with known shapes; dynamic control flow or variable shapes are not handled (§7).
- Practical constraints:
  - Profiling many stage‑mesh pairs is still the dominant compilation cost (Table 5), although manageable for the evaluated scales (§8.4).
  - The intra‑op cost model ignores compute time and uses bytes/bandwidth for communication; while justified (heavy ops equally partitioned), architectures with irregular kernels might deviate (§4.2).
  - Submesh shapes are restricted to guarantee coverage ((1×powers of two) or (n×M)); some exotic topologies could prefer other partitions (§5.2 and Appendix A).
- Scope:
  - No explicit optimization for overlapping communication with computation (§7).
  - Heterogeneous device capabilities or highly irregular network topologies are not modeled explicitly.

## 7. Implications and Future Directions
- Impact on the field:
  - Establishes a general, automated path to combine data, tensor, and pipeline parallelism with strong empirical performance, reducing reliance on manual, model‑specific plans. This can democratize large‑model training for diverse architectures (§10).
- What it enables:
  - Out‑of‑the‑box scaling for new models (e.g., heterogeneous CNNs, custom MoE mixes) without bespoke system engineering (Fig. 7c, Fig. 12).
  - A unifying framework that can incorporate complementary techniques (rematerialization, mixed precision, gradient compression) at either level (§9 “Techniques for training large‑scale models”).
- Promising research directions:
  - Optimize microbatch schedule jointly with stage/mesh choices; model and minimize cross‑stage communication directly in the DP.
  - Design cost models that include compute kernels and overlap comm/compute; integrate topology‑aware mapping and heterogeneous resources.
  - Develop optimal cross‑mesh resharding strategies beyond “local all‑gather,” potentially using multicast trees or topology‑aware collectives (§6, §8.5).
  - Extend to dynamic graphs and dynamic schedules; explore branch‑parallel inter‑op planning and asynchronous variants while maintaining convergence guarantees.
- Practical applications:
  - Training and fine‑tuning large LMs/MoEs on commodity multi‑node GPU clusters; scaling vision or multimodal models with uneven layer characteristics; rapid prototyping of novel architectures using JAX via a simple decorator API (Fig. 4).

> In short, the core idea—optimize intra‑operator sharding and inter‑operator pipelining in a hierarchy that mirrors cluster communication asymmetry—proves powerful. The system achieves parity with the best hand‑tuned setups on their home turf (GPT‑style Transformers) and outperforms specialized baselines where those lack a dimension of parallelism (MoE across nodes), while generalizing to architectures that lack manual strategies (Wide‑ResNet).
