# NanoFlow: Towards Optimal Large Language Model Serving Throughput

**ArXiv:** [2408.12757](https://arxiv.org/abs/2408.12757)
**Authors:** Kan Zhu, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Yufei Gao, Qinyu Xu, Tian Tang, Zihao Ye, Keisuke Kamahori, Chien‑Yu Lin, Stephanie Wang, Arvind Krishnamurthy, Baris Kasikci
**Institutions:** University of Washington, Tsinghua University, University of California, Berkeley, University of Michigan

## 🎯 Pitch

NanoFlow revolutionizes large language model serving by converting computation bottlenecks into increased throughput through dynamic nano-batching and resource allocation within GPUs. This framework significantly enhances efficiency, delivering up to 1.91× throughput over existing systems, crucial for reducing costs and improving performance in high-demand environments utilizing thousands of GPUs.

---

## 1. Executive Summary
NanoFlow is a serving framework that boosts large language model (LLM) throughput by overlapping compute-, memory-, and network-bound operations inside each GPU. It does this by splitting a large request batch into multiple “nano-batches” and running duplicated “nano-operations” concurrently, combined with an auto-search procedure that builds an optimal intra-device pipeline and allocates GPU resources to each nano-operation.

The paper first shows—analytically and empirically—that modern LLM serving is often compute-bound at the end-to-end level, not memory- or network-bound as commonly assumed. NanoFlow exploits this by keeping compute units busy while memory and networking work proceed in parallel, delivering up to 1.91× throughput over state-of-the-art systems and reaching 50–72% of theoretically optimal throughput (Sections 3 and 6; Figures 2–3, 6–7, 11).

## 2. Context and Motivation
- Problem/gap:
  - LLM serving is widely believed to be memory-bound due to the key–value cache (KV-cache) and self-attention’s memory intensity. However, serving engines still underutilize GPU compute: heterogeneous operations (compute, memory, networking) run sequentially, leaving pipeline “bubbles” where compute sits idle (Section 3.6; Figure 4).
- Why important:
  - Planet-scale serving requires maximizing tokens per GPU per second to control cost and meet demand (Section 1). With tens of thousands of GPUs serving hundreds of millions of users, even small utilization gaps multiply into large resource and cost inefficiencies.
- Prior approaches and their limits:
  - System-level batching, paged attention for KV-cache, chunked prefill, and mixed prefill/decode batching improve utilization (Section 7; [17], [2], [3]), but they stop at the granularity of iterations or phases. Within the GPU, operations still execute serially; memory- and network-bound steps stall compute-bound ones (Figure 4).
- Positioning:
  - The paper offers (1) a cost model and validation showing most common LLM serving workloads are compute-bound end-to-end (Sections 3.2–3.4; Figures 2–3; Table 2), and (2) NanoFlow, which uses intra-device parallelism to overlap heterogeneous operations at fine granularity (Sections 3.7 and 4). It complements, rather than replaces, prior batching and memory optimizations by exploiting unused overlap opportunities inside each GPU.

Key terms (defined when first used):
- KV-cache: The per-request cache storing “Keys” and “Values” from attention for previously processed tokens, enabling fast decoding without recomputation (Section 2.1).
- GQA (Grouped Query Attention): Multiple attention heads share a single KV-cache head, reducing memory traffic per token (Section 3.3).
- GEMM vs. GEMV: General Matrix–Matrix Multiply (compute-heavy dense operations like linear projections) versus Matrix–Vector Multiply (used in decode attention where per-step work often reduces to vector-like operations) (Section 4.1.1).
- AllReduce (AR)/AllGather (AG): Collective communication primitives needed under tensor parallelism to combine or share intermediate results across GPUs (Section 2.3).

## 3. Technical Approach
The approach comprises two parts: an analytical model showing when and why to overlap work, and a serving system that operationalizes the overlap via nano-batching and auto-searched pipelines.

1) Cost model and classification of bottlenecks (Section 3)
- Idea in plain language:
  - For a batch to be processed, the GPU must (a) load data from memory, (b) perform compute, and, under tensor parallelism, (c) move intermediate data across GPUs. The slowest of these determines throughput.
- Key equations:
  - Memory time: T_mem = MemSize / MemBW (Equation 1).
  - Compute time (dominated by dense GEMMs): T_compute ≈ 2·B_dense·P_model / Compute (Equation 2), where `B_dense` is the token batch size used by dense operations and `P_model` is the number of model parameters.
  - Network time (tensor parallelism): T_net ≈ 4·(N_GPU·B_dense·D_model·S_type·L) / NetBW (Equation 3).
  - Memory/compute ratio TR ≈ (Compute/MemBW)·(MemSize/P_model)·(1/(2·B_dense)) (Equation 4).
  - Optimal throughput when compute-bound: Throughput_opt = Compute / (2·P_model) (Equation 5).
- Findings:
  - For modern models and interconnects, network rarely dominates: the ratio T_net/T_compute is typically < 1 for large D_model and fast fabrics like NVLink (Section 3.3; Figure 2).
  - With GQA and large batches, memory time is amortized and TR < 1, leaving compute dominant (Section 3.3; Figure 3).
  - Empirical validation on LLaMA‑2‑70B with 8×A100 shows per-operation times sum to a compute-bound profile (Table 2), matching the model’s prediction (Section 3.4).

2) Why existing engines underperform and the core NanoFlow idea (Sections 3.6–3.7)
- Problem:
  - Even if the overall workload is compute-bound, sequential execution of heterogeneous operations causes compute underutilization (pipeline bubbles) (Figure 4).
- Nano-batching and intra-device overlap:
  - Split one large batch into several “nano-batches” and duplicate each operation into “nano-operations,” each handling its nano-batch. Because different nano-batches have no data dependency, a memory-bound nano-operation can run concurrently with compute-bound nano-operations, and network collectives can also run concurrently (Section 3.7).
  - Although weights may be loaded more times, when workloads are compute-bound, the extra memory I/O can be hidden behind sustained compute (Section 3.7).

3) Auto-search for optimal intra-device pipelines (Section 4.1)
- Challenge:
  - How many nano-batches? What sizes? In what order should nano-operations run? How should GPU resources be apportioned among concurrent kernels? Concurrency can cause interference (mutual slowdowns) due to competition for GPU units and memory bandwidth.
- Step A — Kernel profiling and interference modeling (Section 4.1.1):
  - Profile best implementations of GEMM (compute), GEMV (memory), and network kernels for a grid of batch sizes up to the maximum dense batch (128…B_dense in steps of 128).
  - Interference is unpredictable and cannot be directly controlled, so NanoFlow introduces a GEMM-centric resource proxy `R`:
    - `R` = normalized GEMM performance when overlapped (e.g., `R=0.8` means the GEMM runs at 80% of its solo peak).
    - For the other kernel overlapped with GEMM, measure its normalized performance `P` at the same time (Section 4.1.1).
  - Build a mapping table “resource → performance” from measured overlaps (Table 3; Figure 5). For example:
    - “To get GEMV to 0.3 normalized performance, you must give up 0.2 GEMM performance (from 1.0 to 0.8)” (Figure 5; Table 3).
- Step B — Stage I MILP: pipeline structure search (Section 4.1.2)
  - Goal: Minimize total iteration time while eliminating compute bubbles, assuming no interference (uses the “best-time” profiles).
  - Output: number of nano-operations per stage, each nano-batch size, and the partial order (schedule) that respects true data dependencies (parent op dependencies + overlap only when nano-batch ranges do not intersect).
  - Constraints encoded:
    - Minimal splitting (start with two nano-ops per operation and increase only if needed).
    - Batch sizes ∈ {128, 256, …, B_dense}.
    - Overlap only heterogeneous-bound kernels (compute vs memory vs network).
    - Allow AR/AG transformations to explore collectives with different dependency shapes (Section 4.1.2).
- Step C — Stage II MILP: resource allocation with interference (Section 4.1.3)
  - Fix the Stage I structure, then assign a resource fraction `R` to each concurrent kernel over time; convert `R` to effective runtime using the profiled `P(R)` from Table 3.
  - Constrain that concurrent `R`s sum ≤ 1.0 at any time.
  - Output: a refined pipeline with concrete kernel choices and resource allocations.
- Example schedules:
  - For 70B models on 8×A100, Stage II yields a layered schedule with four overlapping nano-ops at the start of a decoding layer (KQV + decode attention + AG/AR) and two nano-ops elsewhere, with explicit resource shares (e.g., decode attention at resource utilization 0.4 achieves ~80% of its solo performance) (Section 4.1.4; Figure 6).

4) Runtime mechanisms (Section 4.2; Implementation in Section 5)
- Batch formation and scheduling:
  - Maintain a fixed, large dense token batch by mixing many decode requests with chunked prefill tokens (Sarathi-style prefill chunking at token granularity), which reduces variance and tail latency (Section 4.2.1).
  - Asynchronous scheduling: form the next batch while the GPU is still running the current iteration to hide CPU-side scheduling overhead; accept at most one extra decode token per finished request (negligible at average decode length >100; Table 4) (Section 4.2.1).
- Memory/KV management:
  - Predict peak memory to avoid OOM; offload KV to host/SSD if needed and reload later (Section 4.2.2).
  - Simultaneous offloading: copy KV vectors to host immediately after KQV generation using GPU-initiated device-to-host copies during compute-heavy FFN to hide overhead, with NUMA-aware thread binding (Section 4.2.2).
  - Efficient reload for PagedAttention: copy into a contiguous GPU buffer first, then scatter to paged destinations for 7–10× higher H2D bandwidth (Section 4.2.2).

## 4. Key Insights and Innovations
- End-to-end LLM serving is often compute-bound, not memory-bound:
  - Why it’s new/important:
    - The community often treats serving as memory-bound due to KV-cache and decode attention. The paper’s model and measurements show that, with batching and GQA, compute dominates for common models and datasets (Sections 3.3–3.4; Figures 2–3; Table 2).
  - Evidence:
    - Heatmaps in Figures 2–3 show compute dominates network and memory across popular models and hardware; Table 2’s per-operation accounting also sums to a compute-bound profile.
- Intra-device parallelism via nano-batching to overlap heterogeneous operations:
  - What’s different:
    - Prior serving engines build large batches but still run operations sequentially on each GPU (Figure 4). NanoFlow duplicates operations across nano-batches and overlaps compute-, memory-, and network-bound work, turning idle “bubbles” into useful work (Section 3.7; Figure 6).
  - Why it matters:
    - It systematically increases compute utilization—the true global bottleneck—without changing model architecture.
- Auto-search with a two-stage MILP plus interference profiling:
  - What’s different:
    - It decouples structural planning (Stage I) from interference-aware resource allocation (Stage II) to tame the search space (Section 4.1.2–4.1.3), using an empirically built `R → P` mapping (Table 3, Figure 5).
  - Why it matters:
    - It automates per-model, per-workload tuning of the overlap schedule and resource splits, yielding practical pipelines in ~10 minutes instead of hours/days (Section 4.1.2).
- Serving runtime that hides CPU overhead and supports multi-round conversations efficiently:
  - Asynchronous batch formation masks CPU scheduling overhead (Section 4.2.1).
  - KV offload/load pipeline aligns with compute phases and PagedAttention’s paging for high bandwidth reload (7–10× faster H2D for reloading; Section 4.2.2).

## 5. Experimental Analysis
- Setup (Section 6.1):
  - Hardware: 8×A100 80GB SXM, NVLink (intra-node) (Section 6.1).
  - Models: LLaMA‑2‑70B detailed; also LLaMA‑3‑70B, Qwen2‑72B, Deepseek‑67B, Mixtral 8×7B, and LLaMA‑3‑8B (Section 6.1).
  - Baselines: vLLM (paged attention, chunked prefill), DeepSpeed-FastGen, TensorRT‑LLM (Section 6.1).
  - Datasets: Splitwise, LMSYS‑Chat‑1M, ShareGPT; statistics in Table 4 (Section 6.1).
  - Metrics: throughput (tokens/s/GPU), latency normalized by output token count; SLO set to 200 ms/token (Section 6.3).

- Optimal bound and bottleneck validation (Sections 3.4–3.5):
  - Using Equation 5, peak Compute measured by CUTLASS, and `P_model`, the optimal per-GPU throughput for LLaMA‑2‑70B on 8×A100 is 1857 tokens/s (Section 3.5).
  - Table 2’s operation breakdown confirms compute-limited behavior for a large batch (Section 3.4).

- Throughput results (Section 6.2; Figure 7):
  - Constant-length inputs (Figure 7a): NanoFlow achieves on average 2.62× vLLM, 2.78× DeepSpeed‑FastGen, and 1.73× TensorRT‑LLM throughput. Best case reaches 68.5% of the optimal bound.
  - Dataset-length inputs (Figure 7b): NanoFlow achieves on average 4.18× vLLM, 3.45× DeepSpeed‑FastGen, and 1.91× TensorRT‑LLM throughput.
  - Quote:
    > “NanoFlow has the highest throughput in all settings and is able to achieve 68.5% of the theoretical optimal throughput in the best case.” (Section 6.2; Figure 7)

- Latency under load (Section 6.3; Figure 8):
  - At low request rates, NanoFlow’s normalized latency is slightly higher due to favoring large dense batches (throughput-oriented). As rate increases, NanoFlow sustains higher load within the 200 ms/token SLO.
  - For LMSYS‑Chat‑1M, NanoFlow supports up to 1.64× higher request rate than TensorRT‑LLM under the SLO (Figure 8b).
  - Tail behavior:
    > “99th-percentile latency is only 1.07× of the average latency at near-maximum throughput,” attributed to keeping dense batch size nearly constant (Section 6.3).

- Ablations (Section 6.4; Figure 9):
  - Splitting into nano-batches without overlap slows performance by 13.2% (overhead of duplication).
  - Overlapping network-bound with compute-bound kernels yields 1.07× speedup (prefill-heavy case); overlapping both memory- and network-bound with compute-bound yields 1.17× (decode-heavy case).
  - KV offloading adds ~3.0% slowdown from interference, but for multi-round LMSYS‑Chat workloads it reduces compute by 3.02× (Section 6.4).

- Resource usage (Section 6.5; Figure 10):
  - NanoFlow sustains high simultaneous utilization and reaches 68.5% average compute utilization across a layer, whereas the non-overlapped baseline shows serialized usage with large idle spans.

- Other models (Section 6.6; Figure 11):
  - Across LLaMA‑3‑70B, Qwen2‑72B, Deepseek‑67B, Mixtral 8×7B, and LLaMA‑3‑8B, NanoFlow attains 50–72% of optimal throughput and outperforms vLLM (Figure 11).

- Overall assessment:
  - The experimental suite is broad: multiple models (including MoE), three real-world traces plus controlled lengths, three strong baselines, and a theory-derived optimal bound for context. The ablations isolate the contributions of overlap and offload. Results consistently support the central claims: (i) compute-bound regimes are common, and (ii) intra-device overlap materially closes the gap to optimal.

## 6. Limitations and Trade-offs
- Assumption of compute-bound regime:
  - The design relies on workloads where end-to-end compute dominates. In extreme long-context decode or on hardware with unusually low compute/memory ratio, memory may dominate and overlapping may help less (Section 3.3; Figure 3).
- Dependence on large, steady batches:
  - NanoFlow targets high-throughput regimes with abundant requests. When traffic is sparse or highly bursty, forming large dense batches is harder, weakening benefits (Section 4.2.1).
- Interference modeling is approximate:
  - GPU runtime offers no direct knobs for partitioning compute/memory/network bandwidth; the `R → P` mapping is derived from pairwise profiling and assumed to generalize to triple overlaps (compute–memory–network) (Section 4.1.1). While sensitivity tests show low variance (≤5%), it remains a heuristic.
- Search optimality vs. time:
  - The two-stage MILP approximates the optimal solution to keep search time practical (~10 minutes), so it may leave some performance on the table (Section 4.1.2).
- Hardware and implementation scope:
  - Implementation targets NVIDIA GPUs and NVLink systems (Section 5; Section 6.1). Behavior on other architectures/interconnects may differ, although Table 1 argues ratios stay similar across vendors/generations (Section 3.3).
- Added memory I/O and complexity:
  - Nano-batching duplicates operations, increasing weight/KV movement. The approach bets on hiding these costs under compute; in regimes where this is not possible, overheads will surface (Section 3.7; Section 6.4).

## 7. Implications and Future Directions
- How it changes the landscape:
  - The work reframes LLM serving optimization priorities: once you maximize batch-level amortization, the true limiter is often GPU compute, and the biggest win is to keep compute units busy by overlapping the other subsystems. This suggests future serving engines should treat intra-device scheduling as a first-class optimization target, not just batch assembly and memory layout.
- Follow-up research:
  - Adaptive online scheduling: dynamically re-choose nano-batch counts and resource shares `R` as request mix and lengths drift, rather than offline search only.
  - Richer interference models: include triple-overlap profiling and predictive models that can extrapolate to unseen kernel combinations or new GPUs.
  - Integration with quantization/sparsity: combine NanoFlow with low-bit/structured-sparsity methods ([19], [55], [44]) to reduce compute while preserving overlap.
  - Cross-node/topology-aware extension: co-design overlap with pipeline parallelism and heterogeneous fabrics (e.g., ForestColl [54]) to sustain benefits beyond single-node tensor parallelism.
  - Vendor-agnostic implementations: port to AMD/Intel accelerators; validate the model’s claim about stable ratios across vendors (Table 1).
- Practical applications:
  - High-throughput services (chatbots, RAG backends, batch content generation) can reduce serving cost per token by 1.7–4.2× relative to leading open systems (Figure 7).
  - Latency-sensitive deployments can push higher request rates at the same SLO, with tight tails from consistent dense batch sizes (Section 6.3).

> Bottom line: By proving that compute is the end-to-end bottleneck for many real LLM serving workloads (Sections 3.3–3.5) and then explicitly filling compute bubbles with overlapped memory and network work (Figure 6), NanoFlow closes a large fraction of the gap to the theoretical throughput bound (up to 68.5% for LLaMA‑2‑70B and 50–72% across other models; Figures 7 and 11).
