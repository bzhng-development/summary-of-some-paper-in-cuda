# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

**ArXiv:** [2201.12023](https://arxiv.org/abs/2201.12023)

## 🎯 Pitch

Alpa introduces the first system to automatically generate and orchestrate execution plans that unify both intra- and inter-operator parallelism for distributed deep learning—covering the entire spectrum from data, operator/tensor, to pipeline parallelism—in a single, hierarchical framework. By removing the need for manual parallelization and optimizing across a vast space of possibilities, Alpa enables scalable, efficient training of massive and heterogeneous models, often outperforming meticulously hand-tuned systems. This breakthrough empowers researchers and practitioners to quickly and reliably train large models on diverse hardware, democratizing access to large-scale deep learning and accelerating innovation.

---

## 1. Executive Summary
Alpa is a compiler and runtime that automatically plans and executes distributed training of large neural networks by combining two kinds of parallelism—within operators and across operators—in a single hierarchical framework. It turns the previously manual, model- and cluster-specific engineering of “3D parallelism” into an automated optimization problem that, on real GPU clusters, matches or exceeds hand-tuned systems for transformers and enables efficient training of heterogeneous models without bespoke plans (Sec. 8.1, Fig. 7).

## 2. Context and Motivation
- Problem/gap:
  - Training modern large models (billions of parameters) requires splitting work across many devices. Doing so efficiently involves choosing among data parallelism, operator/tensor parallelism, and pipeline parallelism, plus their combinations, which interact in complex ways (Sec. 1–2.1; Fig. 2).
  - Prior auto-parallelization systems either search only a narrow space (e.g., only operator partitioning, only pipeline, or only data parallelism) or rely on strong assumptions about model structure or cluster layout (Sec. 1, 2.1). As a result, they fail to scale diverse models efficiently on real, multi-node clusters.
- Why it matters:
  - Poor parallelization can slow training by an order of magnitude (Sec. 1), raising costs and limiting research velocity. Automating this removes the need for deep systems expertise and generalizes across models and clusters.
- Prior approaches and limits:
  - Manual 3D parallelism (e.g., Megatron-LM) tightly hand-crafts combinations of data, tensor, and pipeline parallelism for specific transformer architectures and cluster shapes (Sec. 2.1). This does not generalize to heterogeneous models (e.g., Wide-ResNet) or new cluster interconnects (§8.1).
  - Systems focusing on only one dimension (e.g., Tofu for intra-operator tensor partitioning, DAPPLE for pipeline) miss cross-technique trade-offs (Fig. 1c–d).
- Positioning:
  - Alpa re-categorizes the space into two orthogonal levels—`intra-operator` vs. `inter-operator` parallelism—and maps them to the hierarchical bandwidth structure of clusters (high-bandwidth intra-node vs. lower-bandwidth inter-node). It then compiles end-to-end plans using optimization at each level (Sec. 3; Fig. 1e, Fig. 3).

## 3. Technical Approach
Alpa is both a compiler and runtime. Its core idea is to optimize in two levels that align with cluster topology and communication patterns.

Key definitions (used throughout):
- `Intra-operator parallelism` (intra-op): Partition the computation of a single operator (e.g., matrix multiply) along tensor axes and run partitions in parallel. Typical collectives: all-reduce, all-gather, all-to-all; high communication volume at split/merge points (Sec. 2.2; Fig. 2a–c).
- `Inter-operator parallelism` (inter-op): Split the model graph into stages assigned to different device groups and pipeline microbatches across stages. Communication is point-to-point between stages; volume is typically smaller but there is pipeline idle time (bubbles). The system adopts the synchronous 1F1B schedule to control memory and maintain synchronous semantics (Sec. 2.2; Fig. 2d).
- `Device mesh`: A logical 2D grid view of devices used to express where tensor partitions live and how collectives run along mesh axes (Sec. 4.1).
- `Sharding spec`: An annotation that describes how a tensor is partitioned (“S”) or replicated (“R”) along its axes and which mesh axes carry the partitions (e.g., `S0R` = rows sharded along mesh axis 0, columns replicated) (Sec. 4.1; Table 1).
- `Resharding`: Converting a tensor from one sharding spec to another, possibly with communication (e.g., all-gather, all-to-all) (Sec. 4.1; Table 2).

High-level pipeline (Sec. 3; Fig. 3):
1. Inter-op pass slices the model graph into pipeline stages and slices the device cluster into `device meshes`. It assigns each stage to one mesh.
2. For each (stage, mesh) pair, the intra-op pass chooses per-operator tensor partition strategies (sharding) to minimize stage execution time.
3. A runtime orchestration layer compiles and launches SPMD executables per mesh and coordinates cross-mesh communication between stages.

Step-by-step mechanisms

A) Intra-op pass: choose how to shard operators on a given device mesh (Sec. 4)
- Goal: For a stage’s computational graph, pick one parallel algorithm for every operator that minimizes end-to-end time (compute + communication + intermediate resharding).
- Strategy space: For each primitive op (e.g., batched matmul), Alpa enumerates parallel mappings of loop indices to mesh axes and derives the resulting input/output sharding specs and any required collectives (Sec. 4.1; Table 3 for batched matmul examples).
- Cost modeling and optimization:
  - Formulate selection as an Integer Linear Program (ILP): pick one algorithm per node (`s_v` is one-hot choice) to minimize the sum of per-op compute and communication costs plus edge resharding costs (Eq. (1), Sec. 4.2).
  - Quadratic resharding terms are linearized with auxiliary variables to fit ILP (Sec. 4.2).
  - Communication costs are estimated by bytes transferred divided by bandwidth per mesh dimension; compute costs for heavy ops are treated as equal across strategies (since work is evenly split) and set to zero; lightweight ops are merged away to reduce problem size (Sec. 4.2).
- Post-ILP refinement:
  - Where applicable, replace all-reduce with reduce-scatter + all-gather to avoid replicating tensors unnecessarily, achieving “weight update sharding” behavior akin to ZeRO (Sec. 4.2).
- Why ILP? Unlike randomized search or linear-graph DP, the ILP can reason over arbitrary DAGs with thousands of ops and unify data parallelism, tensor model parallelism, and optimizer sharding in one SPMD framework (Sec. 4, contrast to [25,55]).

B) Inter-op pass: slice model and cluster, decide stage-to-mesh mapping (Sec. 5)
- Objective: Minimize total pipeline latency for B microbatches:
  - Sum of stage times (time of the first microbatch) plus (B−1) times the bottleneck stage time (Eq. (2), Fig. 5).
- Constraints and design:
  - Co-locate each forward op with its corresponding backward op in the same stage/mesh to reduce recomputation and cross-stage traffic (Sec. 5.1).
  - Use submeshes that fully tile the cluster without wasting devices. Alpa restricts shapes to either (1× power-of-two) “row” submeshes or (n×M) submeshes that consume a full second dimension (e.g., all GPUs in a node), with a proof that these tiles can always cover the cluster when M is a power of two (Sec. 5.2; Appendix A).
- How Alpa solves it:
  - Dynamic Programming (DP) over stage boundaries, device counts, and a chosen bottleneck time `t_max`. For each `t_max`, DP finds the minimum sum of stage times with each stage not exceeding `t_max`; the final objective is min over `t_max` of that sum plus `(B−1)*t_max` (Eq. (3)–(4), Sec. 5.2).
  - For a candidate stage on a candidate submesh, Alpa enumerates logical mesh shapes and calls the intra-op pass to get a plan, compiles it, profiles true latency and memory, and checks it fits device memory given pipeline depth `s` via `mem_stage + s * mem_act ≤ mem_device` (Eq. (5), Sec. 5.2).
- Scaling the search:
  - Early pruning: enumerate `t_max` from small to large and stop once `B*t_max` exceeds the best-so-far; only evaluate `t_max` that differ by at least ε to bound suboptimality by `B*ε` (Sec. 5.2).
  - Operator clustering: reduce DP state by grouping adjacent low-cost or tightly coupled ops into L “layers,” balancing FLOPs per layer while minimizing inter-layer communication (DP in Eq. (6); Sec. 5.2, “Performance optimization #2”).

C) Runtime orchestration (Sec. 6)
- Compilation: Each stage compiles to an SPMD executable for its mesh via XLA/GSPMD, which inserts collectives implied by the intra-op plan (Sec. 6).
- Cross-mesh resharding: Adjacent stages can reside on different mesh shapes and expect different sharding specs; Alpa generates many-to-many send/recv plans and then replaces repeated inter-mesh transfers with a single transfer plus fast intra-mesh all-gather (“local all-gather” optimization) when the destination spec includes replication (Fig. 6b–c; Sec. 6).
- Execution model: MPMD (multiple-program multiple-data) across meshes with pre-generated static instruction lists for compute, communication, and synchronization, scheduled with a user-selected pipeline schedule (1F1B used in the paper) (Sec. 6).

Implementation details
- Frontend API: one decorator `@parallelize` over a JAX function; Alpa traces the function once, compiles, and swaps in the parallel version (Sec. 3; Fig. 4).
- Backend: JAX and XLA for IR and codegen; Ray actors for distributed workers; NCCL for communication (Sec. 8).

## 4. Key Insights and Innovations
1. Hierarchical reformulation of parallelism (fundamental):
   - Insight: The main design degrees of freedom in distributed training neatly separate into intra-operator vs. inter-operator choices, which also map to the cluster’s bandwidth hierarchy (intra-node high-bandwidth vs. inter-node lower-bandwidth). This reframing tames the combinatorial explosion and enables specialized optimizers at each level (Sec. 1–3; Fig. 1e).
   - Significance: Enables joint use of all three classic dimensions (data, tensor/operator, pipeline) without manual coupling assumptions.

2. ILP-based intra-op optimizer with unified sharding vocabulary (fundamental):
   - What’s new: Every operator’s strategy is selected through an ILP that accounts for communication and resharding costs across the whole subgraph, using a compact “sharding spec” language and SPMD semantics (Sec. 4.1–4.2; Tables 1–3).
   - Why it matters: Prior automated systems either assumed linear graphs, focused on one operator family, or used heuristic/randomized search; the ILP gives near-optimal plans with tractable compilation time even for graphs with tens of thousands of ops (Sec. 4.2).

3. DP-based inter-op planner that co-optimizes stage boundaries and mesh shapes (substantial, enabling):
   - What’s new: Stage construction is not tied to uniform layer counts nor uniform mesh shapes; the DP searches both, with memory-aware profiling-guided costs from the intra-op pass (Sec. 5.2).
   - Why it matters: Handles heterogeneous models (e.g., Wide-ResNet) and real clusters where best stage sizes and device groupings vary (Fig. 9b, Fig. 12).

4. Cross-mesh resharding with local all-gather (incremental but practical):
   - What’s new: A generic plan generator for many-to-many inter-mesh reshapes, plus an optimization that shifts repeated inter-mesh sends into one inter-mesh transfer followed by fast intra-mesh all-gather on the receiver (Sec. 6; Fig. 6c).
   - Why it matters: Delivers 2.0× throughput gain on Wide-ResNet at 32 GPUs compared to naïve send/recv (Sec. 8.5; Fig. 11).

5. End-to-end system that matches/exceeds hand-tuned baselines and generalizes (substantial empirical contribution):
   - On GPT, matches or slightly surpasses Megatron-LM; on MoE, up to 9.7× faster than DeepSpeed across nodes; on Wide-ResNet, achieves 80% scaling without a manual plan (Sec. 8.1; Fig. 7; Abstract).

## 5. Experimental Analysis
Evaluation setup (Sec. 8)
- Cluster: 8× p3.16xlarge AWS nodes (64 V100 16GB GPUs), NVLink intra-node and 25 Gbps inter-node networking (Sec. 8).
- Models and scaling:
  - GPT-3 style LMs up to 39B params; MoE language models up to 70B params; Wide-ResNet up to 13B params (Sec. 8; Table 4). Model sizes grow with GPU count (weak scaling).
- Baselines:
  - GPT: Megatron-LM v2 (manual 3D parallelism) (Sec. 8.1).
  - MoE: DeepSpeed with expert parallelism combined with ZeRO and tensor parallelism (Sec. 8.1).
  - Wide-ResNet: a “PP-DP” baseline using only pipeline + data parallelism (similar to DAPPLE/PipeDream space) (Sec. 8.1).
- Metric: Aggregated PFLOPS throughput; runs use dummy data after warmup to isolate system performance; variance <0.5% so no error bars (Sec. 8.1).

Main results (Fig. 7; Sec. 8.1)
- GPT (Fig. 7a):
  - Alpa achieves comparable or slightly higher throughput than Megatron-LM across 1–64 GPUs, with near-linear to super-linear weak scaling.
  - Pure intra-op auto-parallelism (“Intra-op only”) degrades beyond 16 GPUs due to heavy cross-node collectives; pure inter-op (“Inter-op only”) scales well but can be memory-limited in some regimes.
  - Analysis in the text links Alpa’s edge to automatic weight-update sharding (post-ILP transform), which Megatron-LM does not natively include (Sec. 8.1).
- MoE (Fig. 7b):
  - On multi-node, Alpa outperforms DeepSpeed by large margins: “3.5× speedup on 2 nodes and a 9.7× speedup on 4 nodes” (Sec. 8.1).
  - DeepSpeed’s lack of inter-op pipeline parallelism on this cluster causes poor scaling beyond a single node; Alpa leverages both levels.
- Wide-ResNet (Fig. 7c; Fig. 12):
  - Alpa delivers scalable performance on 32 GPUs with “80% linear scaling efficiency” (Sec. 8.1), while PP-DP and pure inter-op often run OOM due to inability to shard large weights.
  - Case study shows nontrivial stage sizes and per-layer sharding (e.g., switching to channel-axis partitioning in later WRN stages) that would be hard to craft manually (Sec. 8.6; Fig. 12).

Ablations and supporting studies
- Intra-op ablation (ILP vs. ZeRO-2/3 vs. heuristic; Fig. 8; Sec. 8.2):
  - Across GPT, MoE, and WRN on up to 8 GPUs, the ILP planner maintains near-linear scaling while alternatives either run OOM (Data) or suffer higher communication (ZeRO-2/3, Heuristic). For WRN, alternatives degrade severely; ILP remains best (Fig. 8c).
- Inter-op ablation (DP vs. equal operators vs. equal layers; Fig. 9; Sec. 8.3):
  - On WRN-32 GPUs, DP is “2.6×” faster than equal-operators and “1.6×” faster than equal-layers; on homogeneous GPT, equal-layers approaches DP (Fig. 9a–b).
- Compilation time (Sec. 8.4; Fig. 10; Table 5):
  - Total compile/search time grows roughly linearly with model and cluster size; GPT-39B on 64 GPUs compiles in ~2,393 s with Alpa’s optimizations, versus >40 hours without (Table 5). Most time is in stage-mesh enumeration and profiling.
- Cross-mesh resharding (Sec. 8.5; Fig. 11):
  - Local all-gather optimization improves WRN throughput by 2.0× on 32 GPUs over naïve send/recv. A “signal send/recv” curve (1-byte transfer) acts as an upper bound reference.
- Plan visualization (Sec. 8.6; Fig. 12; Appendix C Fig. 13):
  - On WRN-16 GPUs, stages receive 4, 4, and 8 GPUs respectively, with early layers favoring data-parallel partitioning (activation-heavy) and later layers using channel partitioning (weight-heavy).

Do the experiments support the claims?
- The evaluation spans homogeneous and heterogeneous models, single- and multi-node regimes, and includes strong baselines with grid search where applicable (Sec. 8.1). Combined with ablations isolating intra- and inter-op components (Sec. 8.2–8.3) and a case study (Sec. 8.6), the evidence supports the central claims of performance parity or gains over expert-tuned systems and generalization to new architectures.

## 6. Limitations and Trade-offs
Assumptions and modeling choices (Sec. 7)
- Cross-stage communication is not modeled in the optimization loops (it is implemented in runtime but not part of DP/ILP cost), justified by typically small volume; however, omitting it can miss plans where pipeline communication dominates.
- The number of microbatches `B` is a user-tuned hyperparameter; the DP assumes a fixed schedule and uses `B` only in the latency formula (Eq. (2)). Jointly optimizing `B` is left to search or future work.
- Pipeline schedules are static and linear; parallel branches are not explored for concurrent multi-branch pipelines (Sec. 7).
- Overlap of compute and communication is not explicitly optimized; compilation assumes static graphs with known shapes (Sec. 7).

Scalability and compute constraints
- DP complexity without pruning can be high (Sec. 5.2), prompting several engineering shortcuts: restricted submesh shapes (with a covering proof), operator clustering to shrink the graph, early pruning, and profiling-based selection. These design choices bias the search but make it tractable.
- Profiling stage-mesh candidates dominates compile time (Table 5). Although the cost model accelerates this, profiling remains the main bottleneck for very large search spaces (Sec. 8.4).

Edge cases and scenarios not addressed
- Non-synchronous or elastic training regimes (e.g., asynchronous pipelines, stragglers) are out of scope (Sec. 7).
- Networks with highly irregular or dynamic topologies, or clusters with non-power-of-two device-per-node configurations, may reduce the effectiveness of the preselected submesh shapes, though Appendix A shows covering for many practical cases.
- Dynamic shape models and data-dependent control flow are not supported by the static compilation pipeline (Sec. 7).

## 7. Implications and Future Directions
How this changes the landscape
- Alpa turns distributed training plan design into a compiler optimization problem with clear separation of concerns (intra-op ILP, inter-op DP). This provides a unifying substrate where new parallelization tricks (e.g., optimizer sharding variants, topology-aware collectives) can be plugged in as strategies or cost terms rather than hard-coded into model-specific systems.

Practical applications
- Out-of-the-box efficient training for:
  - Large transformer LMs on commodity GPU clusters (matches Megatron-LM on GPT; Fig. 7a).
  - Large MoE models across nodes (substantial speedups over DeepSpeed; Fig. 7b).
  - Heterogeneous CNN backbones where manual plans are lacking (WRN scaling; Fig. 7c, Fig. 12).
- Cloud and enterprise ML platforms can expose a simple API (e.g., `@parallelize` in Fig. 4) to democratize massive model training without dedicated systems experts.

Follow-up research directions
- Integrate explicit cross-stage communication modeling into the DP/ILP, possibly with network-topology-aware costs and collective fusion.
- Co-optimize microbatch count `B`, schedule variants (e.g., parallel branches, interleaved schedules), and overlap strategies.
- Extend to dynamic-shape or data-dependent graphs via speculative compilation or runtime adaptive partitioning.
- Richer intra-op cost models including compute heterogeneity (e.g., mixed precision kernels, tensor core availability) and kernel fusion interactions.
- Optimal cross-mesh resharding: formalize and solve the many-to-many multicast with replication to minimize total time, beyond the current local all-gather heuristic (Sec. 6, 8.5).

> “Alpa generates parallelization plans that match or outperform hand-tuned model-parallel training systems even on models they are designed for... and also generalizes to models without manually-designed plans.” (Abstract; supported by Sec. 8.1 Fig. 7)

By providing a principled, extensible compiler that spans the full parallelism space, Alpa meaningfully lowers the barrier to training ever-larger models efficiently and consistently across diverse hardware setups.
