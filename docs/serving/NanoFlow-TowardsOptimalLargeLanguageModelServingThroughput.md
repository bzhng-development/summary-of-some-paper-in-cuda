# NanoFlow: Towards Optimal Large Language Model Serving Throughput

**ArXiv:** [2408.12757](https://arxiv.org/abs/2408.12757)

## 🎯 Pitch

NanoFlow introduces a novel serving framework for large language models (LLMs) that maximizes GPU throughput by overlapping compute-, memory-, and network-bound operations within a single device through fine-grained intra-device pipelining. By leveraging 'nano-batching' and an automated pipeline search, NanoFlow smartly splits and schedules workloads to boost compute utilization, achieving up to 1.91× higher throughput than state-of-the-art systems and reaching 50–72% of the theoretical optimal. This breakthrough matters because it significantly lowers the infrastructure cost and increases the serving capacity for planet-scale AI systems, addressing the urgent need for efficient LLM deployment amid global GPU constraints.

---

## 1. Executive Summary
NanoFlow is a serving framework for large language models (LLMs) that maximizes end-to-end throughput by overlapping compute-, memory-, and network-bound operations on the same GPU via fine-grained intra-device pipelining. It introduces “nano-batching” and an automatic pipeline search that jointly determine how to split batches, order operations, and allocate GPU resources, yielding up to 1.91× higher throughput than state-of-the-art systems and reaching 50–72% of a derived theoretical optimum (Fig. 7, Fig. 11; Eq. 5).

## 2. Context and Motivation
- Problem addressed
  - Modern LLM serving involves heterogeneous operations whose bottlenecks differ: matrix multiplications (compute-bound), attention over cached keys/values (memory-bound during decoding), and collective communication (network-bound under tensor parallelism). Existing systems execute these stages sequentially per layer and per batch, leaving the dominant resource—compute—underutilized (Sec. 3.6; Fig. 4 shows “WASTED” pipeline bubbles).
  - Conventional wisdom treats LLM serving as memory-bound due to large model sizes and the key–value (KV) cache. The paper challenges this and shows that, for common models, workloads, and hardware, serving is compute-bound when considered end to end (Sec. 3.3–3.4; Figs. 2–3; Table 2).

- Why it matters
  - Throughput (tokens per second per device) drives cost for planet-scale inference with tens of thousands of GPUs (Sec. 1). With GPU supply constrained, improving utilization directly reduces serving cost and increases capacity.

- Shortcomings of prior approaches
  - Request-level and phase-level scheduling: systems such as vLLM (paged attention), Sarathi/Sarathi-Serve (chunked prefill), and DistServe/Splitwise (prefill/decode disaggregation) improve batching or cluster-level placement but do not overlap heterogeneous operations inside a single GPU execution (Sec. 7).
  - Current engines (e.g., vLLM, DeepSpeed-FastGen, TensorRT-LLM) run operations sequentially per layer, achieving good utilization per operation but poor overall compute utilization (~40%) because different stages bottleneck different resources (Sec. 3.6; Fig. 4).

- Positioning
  - NanoFlow reframes the serving loop as a resource-overlap scheduling problem within a GPU. It introduces (1) nano-batches and duplicated “nano-operations” to permit overlap, and (2) an auto-search procedure that models kernel interference and chooses the number/size/order of nano-operations and their GPU resource shares (Sec. 4).

## 3. Technical Approach
This section explains the system end to end, from the underlying performance model to the pipeline that NanoFlow automatically constructs and executes.

- Background: where time goes in LLM serving (Sec. 2)
  - Two phases per request:
    - `prefill`: process the whole prompt at once; initializes the `KV-cache` (per-token key/value tensors reused at decode).
    - `decode`: generate output tokens one by one; each step attends to the entire per-request KV-cache.
  - Key operation classes (Fig. 1):
    - `Dense` (compute-bound): GEMMs for projections (K/Q/V, O) and MLP (Up/Gate/Down).
    - `Attention–decode` (memory-bound): loads per-request KV-cache and performs GEMV-like operations over past tokens.
    - `Network` (network-bound): collectives (`AllReduce`, `AllGather`) under tensor parallelism to combine shards across GPUs.

- A cost model that predicts the bottleneck and a theoretical optimum (Sec. 3)
  - The paper models the time for one serving iteration at maximum feasible batch size (so batching effects are fully amortized).
  - Memory time (Eq. 1): `T_mem = MemSize / MemBW`. Intuition: at large batches, each iteration touches all model weights + KV-cache in device memory.
  - Compute time (Eq. 2): for dense ops, `T_comp ≈ 2 * B_dense * P_model / Compute`, where `P_model` is parameter count (approx the total weight multiplications), and `B_dense` is the token batch size over which GEMMs run (includes decode tokens across many requests plus prefill tokens).
  - Network time (Eq. 3): `T_net ≈ 4 * NGPU * B_dense * D_model * S_type * L / NetBW` for the collectives needed under tensor parallelism.
  - Key ratios:
    - Network vs compute: `T_net / T_comp ≈ (2 D_model L / P_model) * (NGPU * Compute / (NetBW / S_type))` (Sec. 3.3). With typical `D_model ≥ 4096` and modern interconnects, this ratio is usually < 1 (Fig. 2), meaning network is not the end-to-end bottleneck.
    - Memory vs compute: `T_mem / T_comp ≈ (Compute / MemBW) * (MemSize / P_model) * (1 / (2 B_dense))` (Eq. 4). Grouped-Query Attention (GQA) increases `B_dense` by reducing KV-cache size per request, pushing the system toward compute-bound (Fig. 3).

  - Optimal throughput (Eq. 5): when compute is the bottleneck and fully utilized,
    - `Throughput_opt (tokens/s/GPU) = Compute / (2 * P_model)`.
    - This upper bound depends only on GPU FLOP/s and model size; it is independent of memory size/bandwidth and sequence lengths at large batch sizes (Sec. 3.5).

  - Validation: LLaMA‑2‑70B on 8×A100 (Sec. 3.4; Table 2)
    - Summing estimated per-op times shows compute dominates. Measured runtimes match estimates except prefill attention (kernel launch overhead dominates there).
    - This underpins the design choice to maximize compute utilization by overlapping other resources underneath.

- Why sequential execution wastes compute (Sec. 3.6; Fig. 4)
  - Existing systems process, per layer: K/Q/V GEMMs → decode attention → O/MLP GEMMs → collectives, one after another. Because each stage stresses a different resource, the compute units sit idle during memory-bound and network-bound phases (“pipeline bubbles” in Fig. 4), capping overall throughput well below the Eq. 5 bound.

- NanoFlow’s core idea: intra-device parallelism via nano-batching (Sec. 3.7; Sec. 4)
  - `Nano-batch`: split the large batch processed by a layer into 2–4 smaller, non-overlapping ranges of tokens/requests.
  - `Nano-operations`: duplicate each layer operation (e.g., KQV GEMM) so that each copy runs on a different nano-batch.
  - Because nano-operations process disjoint data, heterogeneous operations can run concurrently: while part of the GPU computes a GEMM on nano-batch A, another part can run memory-bound decode attention on nano-batch B, and a third part can issue network collectives on nano-batch C. This overlaps resource usage and fills the pipeline bubbles that previously idled compute.
  - Trade-off: more weight reads (I/O) due to duplication. The cost model argues this is acceptable in compute-bound regimes because the extra memory I/O can be hidden under compute (Sec. 3.7).

- Handling kernel interference with resource shares (Sec. 4.1)
  - When multiple kernels run concurrently, they slow each other down due to contention for GPU execution units, caches, and memory bandwidth. NanoFlow:
    - Profiles “interference-free” times for candidate kernels across batch sizes (Sec. 4.1.1).
    - Profiles pairwise interference between compute kernels (GEMM) and memory/network kernels (GEMV for decode attention, collectives) and represents it with a simple mapping:
      - `R`: the fraction of GEMM throughput retained when overlapped (GEMM-centric resource share).
      - `P`: the normalized performance achieved by the other kernel at that `R`.
      - Table 3 captures typical `R → P` exchange rates (e.g., allocating `R=0.8` of “GEMM resources” leaves enough for decode attention to reach `P≈0.85`).
    - Fig. 5 visualizes measured trade-offs and discards dominated kernel pairs (those that simultaneously harm GEMM more and help GEMV less).

- Automatic pipeline search (Sec. 4.1)
  - Stage I: pipeline structure search (MILP; ignores interference)
    - Inputs: max dense batch size, op dependencies (from the model graph), profiled per-kernel times.
    - Outputs: how many nano-operations per layer op, each nano-batch’s size, and the execution order, under constraints:
      - At least two nano-operations per op to allow overlap, but keep the count small to preserve batching efficiency.
      - Respect model dependencies and input-range overlaps (two nano-ops are dependent only if their parent ops are dependent and their input ranges intersect).
      - Only overlap operations with different resource bottlenecks (no benefit in overlapping two compute-bound kernels).
      - Explore functionally equivalent network transformations (e.g., AG ↔ AR variants) that change dependencies and performance (Sec. 4.1.2).
  - Stage II: resource allocation with interference (MILP; uses Table 3)
    - Fix the Stage I structure. Choose a time-varying `R` (GEMM share) per nano-operation such that concurrent `R` sums ≤ 1.0.
    - Compute each nano-op’s runtime as `Dbest / P`, where `Dbest` is its solo best time and `P` comes from the profiled `R → P` mapping (Sec. 4.1.3).
    - Objective: minimize total pipeline time (remove or shrink compute bubbles).
  - Example pipeline for 70B models (Fig. 6)
    - KQV and decode attention at the start of a layer run in four nano-operations with each decode attention using `R≈0.4` (achieving ~80% of its solo performance).
    - Later GEMMs are prioritized (higher `R`); two nano-ops suffice. Collectives are scheduled where they overlap profitably with compute.

- Runtime mechanisms (Sec. 4.2)
  - Batch formation aimed at a fixed high-throughput `B_dense`:
    - Mix in-progress decode requests with chunked pieces of long prefills so GEMMs always see a large batch (follows Sarathi-style chunking but pushes further to stabilize token-level batch sizes).
    - Predict near-future memory usage from decoded tokens and average lengths; if needed, temporarily offload a request to CPU memory to avoid OOM (Sec. 4.2.1).
  - Asynchronous scheduling:
    - While the GPU executes iteration `i`, the CPU prepares iteration `i+1` (including page-table updates for paged attention). EOS detection is one iteration behind; the extra decode token per finished request is negligible for typical lengths (>100 tokens; Table 4) and hides host-side overheads (Sec. 4.2.1).
  - KV-cache offloading to CPU/SSD for multi-round chats (Sec. 4.2.2):
    - After KQV computation at each layer, the KV for new tokens is immediately offloaded device→host (GPU-initiated copies during compute-heavy FFN time).
    - Use LRU to manage CPU/SSD tiers; when resuming a prior conversation, load KV back to GPU via a gather-then-scatter strategy that avoids writing into fragmented page destinations, boosting H2D bandwidth 7–10×.

## 4. Key Insights and Innovations
- A principled throughput model that flips the prevailing assumption (Sec. 3)
  - Insight: At large, practical batch sizes, modern LLM serving is compute-bound end to end, not memory-bound. Heatmaps show compute dominates network on most accelerators and models (Fig. 2), and compute dominates memory especially with GQA and large `B_dense` (Fig. 3). Operation-level accounting (Table 2) empirically validates the model.
  - Significance: Justifies spending extra memory bandwidth (duplicated weight loads) to hide non-compute stalls by overlapping operations.

- A clean, hardware-agnostic optimum (Eq. 5)
  - Insight: `Throughput_opt = Compute / (2 P_model)` offers a simple, interpretable ceiling; it depends only on GPU TFLOP/s and parameter count.
  - Significance: A common yardstick to quantify “how far from ideal” serving engines are, independent of workload lengths or memory size. The paper reports baselines at 22–38% of this bound and NanoFlow closing the gap to as much as 68.5% (Sec. 3.6; Fig. 7).

- Intra-device pipelining via nano-batching (Sec. 3.7; Fig. 6)
  - Innovation: Split each layer’s batch into nano-batches and duplicate the layer’s operations over them, enabling overlap of compute-, memory-, and network-intensive work streams on one GPU.
  - Difference vs prior work: Prior systems schedule at the request/iteration level; NanoFlow operates inside the iteration at the operation level.

- Auto-search with interference-aware resource allocation (Sec. 4.1; Fig. 5; Table 3)
  - Innovation: A two-stage MILP that first finds a feasible pipeline structure and then assigns time-varying resource shares using profiled `R → P` trade-offs. It also explores algebraically equivalent collective patterns (AG→AR) that alter overlap opportunities (Fig. 6).
  - Significance: Turns a vast, hardware-dependent search space into a practical 10-minute planning step that adapts to different models (70B dense, 8B single-GPU, Mixture-of-Experts) and hardware.

- Practical runtime engineering to sustain high throughput (Sec. 4.2)
  - Asynchronous batch formation hides CPU overhead (noted to be non-trivial for paged attention in Sec. 4.2 and [42]).
  - KV-cache offload/load is overlapped with compute, and a gather-then-scatter trick restores fragmented pages efficiently.

## 5. Experimental Analysis
- Setup (Sec. 6.1)
  - Hardware: 8×A100 80GB SXM with NVLink.
  - Models: Detailed study on `LLaMA-2‑70B`; additional results on `LLaMA‑3‑70B`, `Qwen2‑72B`, `Deepseek‑67B`, `Mixtral 8×7B` (MoE), and `LLaMA‑3‑8B` (single GPU).
  - Baselines: vLLM (paged attention), DeepSpeed-FastGen (dynamic prefill+decode), TensorRT-LLM (TensorRT-based engine with dynamic batching and paged KV).
  - Workloads: Real traces from Splitwise, LMSYS-Chat‑1M, ShareGPT (Table 4). Also constant-length experiments (e.g., 512/1024-token prompts and outputs).
  - Metric: total token throughput (prefill + decode) per GPU; optimal throughput computed via Eq. 5 using CUTLASS-measured peak GEMM performance (Sec. 6.2).

- Main throughput results (Fig. 7)
  - Constant lengths (Fig. 7a, LLaMA‑2‑70B, TP=8):
    - NanoFlow achieves on average 2.62× vLLM, 2.78× DeepSpeed-FastGen, and 1.73× TensorRT‑LLM.
    - Best case reaches 68.5% of the theoretical optimum (1857 tokens/s/GPU computed from Eq. 5; Sec. 6.2).
  - Real traces (Fig. 7b):
    - Average gains rise to 4.18× over vLLM, 3.45× over DeepSpeed-FastGen, and 1.91× over TensorRT‑LLM.
  - Interpretation: These gains align with the compute-bound analysis and the removal of pipeline bubbles by overlapping heterogeneous kernels (compare Fig. 4 vs. Fig. 6).

- Latency under load (Fig. 8; Sec. 6.3)
  - Method: Poisson arrivals; report mean latency normalized by output length (ms/token); SLO of 200 ms/token (typical human reading speed; [8]).
  - Results:
    - At low request rates, NanoFlow’s latency is slightly higher than the best baseline due to its larger steady-state batch size (throughput-oriented design).
    - Within a 200 ms/token SLO, NanoFlow sustains higher arrival rates; for LMSYS-Chat‑1M, it admits 1.64× higher request rate than TensorRT‑LLM (Fig. 8b).
    - Tail latency: 99th percentile is 1.07× the average near maximum throughput because token-level batch size is stabilized (Sec. 6.3).

- Ablations and mechanism validation (Fig. 9; Sec. 6.4)
  - Splitting into nano-batches without overlap (“nano-batch only”) reduces throughput by 13.2%; this quantifies the overhead of duplicated weight loads and losses in GEMM efficiency.
  - Overlapping benefits:
    - Prefill-only (compute + network): +1.07× over non-overlapped baseline.
    - Decode-heavy (compute + memory + network): +1.17×.
  - KV offloading: ~3% slowdown due to added data movement, but reduces compute by 3.02× for multi-round LMSYS‑Chat (useful when serving multi-turn chats at scale).

- Resource usage (Fig. 10; Sec. 6.5)
  - Non-overlap baseline shows serial utilization of compute/memory/network.
  - NanoFlow sustains ~68.5% average compute utilization across the layer pipeline while concurrently exercising memory and network. Remaining gap stems from kernel interference (as modeled in Stage II).

- Generality across models (Fig. 11; Sec. 6.6)
  - NanoFlow achieves 50–72% of the optimal throughput across `LLaMA‑3‑70B`, `Qwen2‑72B`, `Deepseek‑67B`, `Mixtral 8×7B`, and `LLaMA‑3‑8B`, and consistently outperforms vLLM.
  - For the single‑GPU 8B model (no network collectives), auto-search reduces to overlapping compute with memory-bound decode attention.

- Do the experiments support the claims?
  - The compute-bound claim is backed by:
    - Analytical ratios (Sec. 3.3), heatmaps across models/hardware (Figs. 2–3), and operation-level measurement/estimation agreement (Table 2).
  - Throughput gains are broad (three datasets; constant vs real lengths), and ablations isolate the contribution of overlap vs nano-batching overhead.
  - Latency trade-offs are measured; the SLO study shows throughput benefits do not come at the expense of unacceptable latency under realistic loads.

## 6. Limitations and Trade-offs
- Dependence on the compute-bound regime (Sec. 3.3; Fig. 3)
  - If workloads become memory-bound (e.g., very long decodes on small models or tiny batch sizes), duplicating operations can exacerbate memory traffic and may not yield gains.
  - The analysis assumes large enough `B_dense` (hundreds–thousands of tokens) and abundant requests; at low traffic, NanoFlow’s large-batch strategy can increase per-request latency (Fig. 8).

- Interference modeling simplifications (Sec. 4.1.1–4.1.3)
  - Uses pairwise interference (compute–memory, compute–network), then assumes these mappings hold when three kernels overlap. The `R→P` table is derived empirically per hardware family and may need re-profiling across GPUs or driver versions. Standard deviation is within 5% across measured shapes, but edge cases can deviate.

- Search and portability
  - Auto-search takes about 10 minutes per model/workload configuration; while negligible for long deployments, it is an extra step and relies on accurate profiling of candidate kernels.
  - The implementation and interference profiles are developed on NVIDIA GPUs. Porting to other accelerators requires rebuilding the profiling database and possibly adapting kernel choices.

- Memory overheads
  - Nano-batching increases weight loading frequency and temporarily increases host-memory pressure due to KV offloading. Although largely hidden under compute, the ~3% runtime cost (Fig. 9) is measurable.

- Scheduling assumptions
  - The runtime assumes the broader control plane keeps per-instance batches sufficiently full (Sec. 4.2). In multi-tenant or bursty environments without such coordination, throughput benefits may shrink.

## 7. Implications and Future Directions
- How it changes the landscape
  - Establishes a compute-centric view of LLM serving with a simple optimality benchmark (Eq. 5). This reframes system design around overlapping heterogeneous resources rather than optimizing any single stage in isolation.
  - Demonstrates that intra-device scheduling (below the request/iteration level) is a powerful lever—orthogonal to request-level batching, paged attention, quantization, or cluster-level disaggregation.

- Follow-on research enabled
  - Adaptive, online auto-search: make `R` allocations and nano-batch sizes react to live traffic, mixture-of-lengths, and interference signals.
  - Richer interference models: extend from pairwise to multi-kernel and cross-SM/cache contention models; incorporate MIG/MPS multi-tenancy interference.
  - Integration with model compression: jointly optimize quantization/sparsity (e.g., QServe, ATOM, QUEST) and nano-pipeline overlap to further raise the Eq. 5 ceiling by increasing effective TFLOP/s and reducing `P_model`.
  - Compiler support: capture nano-batch duplication and overlap as first-class IR transformations and enable whole-graph scheduling across layers.

- Practical applications
  - High-throughput batch workloads (batch inference, data labeling, RAG precomputations) where large steady-state batches are available.
  - Multi-round assistants: KV offloading and fast reload benefit chat services that serve many concurrent sessions with pauses (Sec. 4.2.2).
  - Heterogeneous clusters: the auto-search and the Eq. 5 bound provide a principled way to plan capacity and compare hardware (Table 1) independent of specific workloads.

> Bottom line: By proving that modern LLM serving is typically compute-bound and then systematically filling compute bubbles through intra-device overlap, NanoFlow moves practical serving throughput much closer to a clear theoretical limit, with broad applicability across dense and MoE models (Figs. 7 and 11).
