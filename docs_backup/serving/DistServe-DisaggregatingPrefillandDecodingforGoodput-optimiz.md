# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

**ArXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

## 🎯 Pitch

DistServe pioneers a novel architecture for large language model (LLM) serving by disaggregating the prefill and decoding phases onto separate GPUs, eliminating interference and enabling tailored resource and parallelism strategies for each phase. This bandwidth-aware, goodput-optimized approach dramatically increases efficiency, delivering up to 7.4× higher request rates and 12.6× tighter latency SLOs than state-of-the-art systems, while maintaining quality of service for the vast majority of requests. By fundamentally rethinking LLM serving system design, DistServe provides a powerful path to cost-effective, scalable, and responsive deployment of advanced AI models.

---

## 1. Executive Summary
DistServe is a serving system for large language models (LLMs) that splits inference into two phases—`prefill` (processing the prompt to produce the first token) and `decoding` (streaming the remaining tokens)—and runs them on different GPUs. By co-optimizing resources and parallelism for each phase and placing them bandwidth-aware in a cluster, it maximizes per-GPU goodput (the maximum request rate that meets latency SLOs), achieving up to 7.4× higher request rates or 12.6× tighter latency SLOs than state-of-the-art systems while satisfying SLOs for over 90% of requests (Abstract; §6.2, Fig. 8–9).

## 2. Context and Motivation
- Problem addressed
  - LLM serving must meet two latency metrics: `TTFT` (time-to-first-token; the prefill duration) and `TPOT` (time-per-output-token; average time per generated token excluding the first) (§1, footnote 1). Many applications care about both; e.g., chatbots need low TTFT for responsiveness, while summarization needs low TPOT for fast generation (§1).
  - Existing systems colocate prefill and decoding on the same GPUs and batch them together for throughput (§2.2). This causes interference between the phases and couples their resource/parallelism choices, making it hard to meet both TTFT and TPOT without over-provisioning (§1, §2.3).

- Why it matters
  - GPUs are expensive; over-provisioning to satisfy both TTFT and TPOT inflates cost per query. The goal is to maximize per-GPU `goodput`—the highest request rate that meets latency SLOs for a target fraction of requests (SLO attainment)—to reduce cost (§1).

- Shortcomings of prior approaches
  - Continuous batching (e.g., vLLM) boosts throughput but blends long prefill steps with short decode steps, delaying decodes and hurting TPOT; decodes also slow prefill and hurt TTFT. Figure 2 shows both slowdowns rising with batch size; the effect is severe when adding even a single prefill to a decode-only batch (§2.3, Fig. 2).
  - Chunked prefill with piggybacking (e.g., DeepSpeed-MII; splits long prefills into chunks and “piggybacks” some decodes) mitigates but does not eliminate interference, and it incurs extra KV-cache memory traffic that grows quadratically with the number of chunks (§2.3).
  - Scheduling prefill and decoding sequentially (no batching) still causes queueing contention and underutilization (§2.3).
  - Parallelism coupling: colocating phases forces one parallelism plan to serve both, although prefill and decoding have different compute/bandwidth profiles and different latency goals (§2.3).

- Positioning
  - DistServe reframes the serving architecture: disaggregate prefill and decoding onto different GPUs, then independently optimize batching, parallelism, and replication choices for each phase (§1, §3–§4). It also adds bandwidth-aware placement to control communication of intermediate states (KV caches) (§3.3, §4.2).

## 3. Technical Approach
DistServe’s design has three parts: (1) disaggregation with per-phase optimization, (2) placement search (two algorithms for different cluster bandwidth regimes), and (3) runtime scheduling tuned for real workloads.

A. What is being disaggregated and why
- `Prefill` versus `decoding`
  - Prefill processes all input tokens in parallel to produce the first output token; it is often compute-bound for non-trivial prompts (e.g., a 13B model prefill with 512 tokens saturates an A100; §3.1, Fig. 3a).
  - Decoding generates tokens one-by-one and is bandwidth-bound because it repeatedly accesses model weights and KV caches while performing small compute per step (§2.1, §3.2; Fig. 3b).
- Key consequence: running them separately eliminates direct interference and lets each phase choose its own batching and parallelism to meet its target metric—TTFT for prefill, TPOT for decoding (§1, §3).

B. Per-phase design choices and analysis
- Batching
  - Prefill: small batches or even single-request execution when prompt length exceeds a model- and hardware-specific threshold `L_m` that saturates the GPU; larger batches beyond `L_m` only increase latency (§3.1, Fig. 3a).
  - Decoding: aggressive batching to amortize bandwidth limits and raise utilization; disaggregation enables many prefills to feed one decoding instance to build larger decode batches without harming TTFT (§3.2).
- Parallelism (two kinds)
  - `Intra-op parallelism` (tensor parallelism): partitions heavy ops (e.g., GEMMs) across GPUs; reduces execution time (helps TTFT/TPOT) but adds communication overhead and prefers high-bandwidth links like NVLINK (§2.2).
  - `Inter-op parallelism` (pipeline parallelism): splits layers across GPUs; slightly raises execution time due to inter-stage comms but increases capacity and reduces queueing delays as the number of stages grows (§2.2).
- Analytical guidance with queueing theory (for prefill)
  - With uniform prompt lengths and FCFS scheduling, prefill behaves like an M/D/1 queue (§3.1). Average TTFT combines execution time and queueing delay:
    - Base (no parallelism): Eq. (1) `Avg_TTFT = D + (R D^2) / (2 (1 - R D))`, where `R` is arrival rate and `D` is per-request execution time.
    - Inter-op (2 stages): Eq. (2) shows queueing diminishes because the slowest stage time `D_m` reduces the effective utilization in each stage.
    - Intra-op (2-way with speedup `K`<2): Eq. (3) reduces execution time to `D/K` but increases susceptibility to queueing as `R` grows (because service time per stage does not split).
  - Insight: intra-op wins at low rates (execution-time dominated) or under tight TTFT SLOs; inter-op wins at higher rates (queueing-dominated). This matches real measurements in Fig. 4a and the sensitivity analysis in Fig. 4b.
- Parallelism for decoding
  - With large decoding batch sizes (possible only after disaggregation), intra-op lowers TPOT but with diminishing returns (comm/partition overhead), while inter-op scales throughput nearly linearly (Fig. 5). Strategy: use intra-op just enough to meet TPOT SLO, then add inter-op/replicas to scale rate (§3.2).

C. Communication management (KV caches)
- `KV cache` stores attention keys/values for past tokens; it is produced during prefill and consumed throughout decoding. Transfers can be large (e.g., ~1.13 GB for 512 tokens on OPT-66B; §3.3) and potentially frequent (e.g., 10 rps ⇒ ~90 Gbps).
- Bandwidth-aware placement
  - If cross-node bandwidth is plentiful, prefill and decoding can be on any nodes; otherwise, DistServe co-locates matching pipeline stages of prefill and decoding within the same node to use intra-node NVLINK (hundreds of GB/s) for KV transfers (§3.3, §4.2).
- Pull versus push
  - Decoding instances “pull” KV caches from prefill instances when ready, using prefill GPU memory as a buffer. This avoids overwhelming decoders under bursts and keeps phases decoupled (§4.3).

D. Placement algorithms and simulation-guided search
- Definitions
  - An `instance` is a complete copy of model weights (possibly spread across multiple GPUs via model parallelism). DistServe deploys separate prefill instances and decoding instances, possibly at different degrees of inter-op/intra-op parallelism; then it replicates instances to meet the target arrival rate (§2.3, §4).
  - A `placement` is the full specification of how many prefill/decoding instances, their parallelism configs, and their mapping to nodes/GPUs (§4).
- High node-affinity scenario (good cross-node bandwidth)
  - Algorithm 1 (§4.1) enumerates feasible inter-op and intra-op configurations for each phase, simulates SLO attainment for the workload, and chooses the configs that maximize per-GPU goodput; replication then scales to the target rate. Complexity O(N·M^2) for N nodes-per-instance and M GPUs-per-node.
- Low node-affinity scenario (limited cross-node bandwidth)
  - Algorithm 2 (§4.2) enforces that corresponding inter-op stages of prefill and decoding are co-located inside a node. It enumerates intra-node configs for both phases that fit in M GPUs/node, simulates SLO attainment end-to-end (including KV transfer), picks the best within-node segment configuration, and replicates it.
- Why simulation
  - SLO attainment depends on realistic arrival processes and variable prompt/output lengths, which are hard to model analytically; DistServe fits distributions from history and uses a discrete-event simulator with an execution-time model grounded in operator FLOPs and memory access characteristics (Appendix A; §4.1 “Simulator building”). Table 2 shows simulator error <2% versus real runs across rates.

E. Runtime architecture and scheduling
- Architecture (Fig. 6; §5)
  - A controller dispatches requests FCFS to the least-loaded prefill instance; after prefill, it dispatches to the least-loaded decoding instance. Communication uses NCCL across nodes and asynchronous cudaMemcpy intra-node.
- Scheduling refinements (§4.3)
  - Pipeline bubble mitigation: schedule prefill batches to approach the GPU-saturating token budget `L_m` (profiled per model/GPU), and schedule decoding by maximizing active decode batch size.
  - Burst handling: “pull” KV as above.
  - Replanning: periodically re-run placement search when the workload mix shifts; the solver runs in seconds and model reloads in minutes (§4.3, §6.5).
  - Note: preemption and fault tolerance are discussed as future work (§4.3 “Preemption and fault tolerance”).

## 4. Key Insights and Innovations
- Disaggregation of prefill and decoding for goodput optimization (fundamental)
  - Prior work colocated phases to maximize throughput; DistServe separates them, removes cross-phase interference (empirically visible in Fig. 2), and tailors each for its latency target (§1, §2.3, §3). This reframing directly increases per-GPU goodput and simplifies meeting both TTFT and TPOT simultaneously.
- Phase-specific parallelism and batching guided by queueing/latency models (substantive)
  - Prefill: analytical M/D/1 modeling (Eq. 1–3) plus profiling informs when to use intra-op versus inter-op for different rates and SLO tightness; batching uses the `L_m` threshold (§3.1, Fig. 3a, Fig. 4).
  - Decoding: after disaggregation enables large batches, use intra-op minimally to hit TPOT SLO, then inter-op/replication to scale (Fig. 5).
- Bandwidth-aware placement with KV transfer locality (practical innovation)
  - Under limited cross-node bandwidth, Algorithm 2 constrains stage pairs to the same node to exploit NVLINK, cutting transfer delays to near-negligible share of end-to-end latency (Fig. 10a–b) while still allowing independent scaling of phases (§4.2, §6.3).
- Simulation-driven search for SLO attainment and goodput (enabling)
  - Building on a phase-specific latency model (Appendix A) and realistic workload sampling, DistServe can robustly pick non-obvious parallelism/replication plans (e.g., for OPT-175B, prefill PP×TP = 3×3, decoding 3×4; Appendix B, Table 3), which would be hard to determine by intuition alone (§4.1, §6.2; Appendix B).

## 5. Experimental Analysis
- Setup (§6.1)
  - Hardware: 4 nodes, each with 8×A100 80GB (NVLINK intra-node), cross-node network 25 Gbps. Default placement uses the low node-affinity algorithm (Algorithm 2) due to limited cross-node bandwidth.
  - Models and datasets: OPT-13B/66B/175B; ShareGPT for chatbot, HumanEval for code completion, LongBench for summarization (Fig. 7 shows input/output length distributions).
  - SLOs (Table 1): task-specific TTFT and TPOT, e.g., chatbot (13B): TTFT 0.25 s, TPOT 0.1 s; summarization (66B): TTFT 15 s, TPOT 0.15 s.
  - Metric: SLO attainment (percent of requests meeting both TTFT and TPOT). Two analysis modes: (1) vary per-GPU rate and read SLO attainment; (2) fix rate and scale both SLOs tighter using “SLO Scale,” reading the tightest scale that maintains the target attainment (90% main; 99% in Appendix C).
  - Baselines: vLLM (continuous batching, paged-attention; intra-op=1/4/8 for 13B/66B/175B per its paper), DeepSpeed-MII (chunked prefill with piggyback; comparable intra-op, but fails to run 175B on this setup due to kernel constraints and OOM; §6.1).

- Main results (per-GPU rate at 90% SLO attainment; Fig. 8–9)
  - Chatbot, ShareGPT
    - Against vLLM: DistServe handles 2.0×–4.6× higher per-GPU rate across 13B/66B/175B (Fig. 8 top row). Quote: “DistServe can sustain 2.0×–4.6× higher request rate compared to vLLM” (§6.2).
    - Against DeepSpeed-MII: DistServe sustains 1.6×–7.4× higher per-GPU rate (Fig. 8 top row; §6.2).
    - SLO tightness (SLO Scale): DistServe supports 1.8×–3.2× tighter SLOs vs vLLM and 1.7×–1.8× vs MII (Fig. 8 bottom row).
  - Code completion (66B, HumanEval; Fig. 9a)
    - DistServe vs vLLM: 5.7× higher rate; 1.4× tighter SLO.
    - DistServe vs MII: 1.6× higher rate; 1.4× tighter SLO.
    - Interpretation: stricter TTFT dominates for code completion; eliminating decode interference and using more intra-op on prefill reduces TTFT (§6.2).
  - Summarization (66B, LongBench; Fig. 9b)
    - DistServe vs vLLM: 4.3× higher rate; 12.6× tighter SLO.
    - DistServe vs MII: 1.8× higher rate; 2.6× tighter SLO.
    - Interpretation: long inputs stress prefill; TPOT becomes the bottleneck for responsiveness, and de-interleaving phases prevents long-prefill-induced decode slowdowns (§6.2).
  - 99% SLO attainment (Appendix C, Fig. 13–14)
    - DistServe maintains strong gains: 3×–8× higher rate and 1.24×–6.67× tighter SLO vs vLLM; 1.32×–8× higher rate and 1.20×–1.58× tighter SLO vs MII.

- Latency breakdown and communication overhead (§6.3, Fig. 10)
  - KV transfer accounts for under 0.1% of total latency even for OPT-175B in the low-bandwidth cluster, with >95% of transfers completing in <30 ms (CDF in Fig. 10b).
  - Reason: placement enforces prefill/decoding stage pairs inside a node over NVLINK (§4.2).

- Ablations and robustness (§6.4)
  - Simulator accuracy: Table 2 shows <2% error versus the real system for both vLLM and DistServe-Low across a range of rates.
  - Role of disaggregation versus better parallelism tuning: “vLLM++” (searching intra-op settings) offers no gains over vLLM; improvements come primarily from disaggregation, not from parallelism retuning under colocation (Fig. 11).
  - DistServe-High (Algorithm 1; unconstrained, high bandwidth) outperforms DistServe-Low (Algorithm 2; bandwidth-constrained), demonstrating that the placement constraints—not the concept—bound performance on low-bandwidth clusters (Fig. 11).

- Solver performance (§6.5, Fig. 12)
  - Placement search runs in seconds to minutes and parallelizes well; time grows with the number of GPUs explored but is independent of model size because the simulator is event-based.

- Overall assessment
  - The experiments are consistent with the mechanism: interference elimination + phase-specific optimization + bandwidth-aware placement yield better TTFT/TPOT and higher SLO attainment. The latency breakdown (Fig. 10) strongly supports that communication does not dominate when placed correctly. Ablations isolate disaggregation as the key factor.

## 6. Limitations and Trade-offs
- Assumptions about workload predictability
  - Placement relies on profiling and an assumed distribution for arrival processes and prompt/output lengths (§4.1). DistServe mitigates drift via periodic replanning (§4.3) but highly non-stationary workloads could degrade SLOs between replans.
- Resource overhead of disaggregation
  - Each phase maintains its own copy of model weights; memory and GPU count increase relative to colocation. This is significant for very large models (e.g., 175B) and can constrain within-node co-location (e.g., two full copies may not fit in 8×80GB GPUs), forcing more complex segment placements (§4.2).
- Cluster bandwidth constraints
  - If cross-node bandwidth is low and intra-node NVLINK is unavailable or oversubscribed, KV transfer may become non-negligible (§3.3). DistServe’s benefits depend on enforcing stage co-location or having high-speed interconnects.
- Lack of preemption and fault tolerance in the current prototype
  - The system can suffer convoy effects when long prefills block shorter ones; preemptive scheduling and fault tolerance are discussed but not implemented (§4.3).
- Applicability limits discussed by the authors (§7)
  - Throughput-optimized, latency-tolerant settings may prefer colocation with chunked-prefill to fill batches for maximum utilization.
  - Resource-constrained deployments (few or single GPUs) leave little room to disaggregate or choose independent parallelism strategies.
- Modeling simplifications
  - The M/D/1-based insight for prefill assumes uniform prefill lengths; real workloads are variable. DistServe addresses this with scheduling heuristics and simulation, but analytical guarantees do not directly transfer (§3.3, §4.3).
- Kernel/implementation constraints in baselines
  - DeepSpeed-MII could not run OPT-175B in this setup due to kernel divisibility and memory limits (§6.1), which slightly complicates baseline comparisons at the very high end.

## 7. Implications and Future Directions
- How this changes the landscape
  - It reframes LLM serving away from “max throughput via colocation” to “max goodput via disaggregation and phase-specialized optimization.” This aligns the system with user-perceived experience (low TTFT and TPOT) rather than aggregate token throughput.
- Practical applications
  - Interactive chat, copilots, and search assistants benefit from fast first-token display (TTFT) without sacrificing stream speed (TPOT). Batchy back-end tasks (e.g., summarization pipelines) can maintain tight TPOT at high rates even with long prompts (§6.2).
- Follow-up research and system extensions
  - Integrate preemption and fault tolerance (e.g., iteration-level preemption like FastServe, §8; checkpointing/streaming KV caches as in DéjàVu).
  - Combine with KV-reducing attention variants (GQA/MQA) to further cut transfer costs and increase decode batch sizes (§6.1 notes; §7).
  - Explore dynamic phase-to-phase mapping (adaptive numbers of prefill instances per decoding instance) under bursty or non-stationary load, potentially with online learning controlling the placement solver.
  - Extend to very long context models (million-token), where prefill–decode disparity grows; disaggregation likely becomes more, not less, beneficial (§7 “Long-context scenarios”).
  - Broaden to heterogeneous accelerators and disaggregated memory fabrics (CXL), leveraging the placement framework to map stages to diverse resources (§8 on resource disaggregation).

> Bottom line: By separating what LLM inference does into two different performance problems, then solving each with the right tools and placement, DistServe delivers large, consistent gains in SLO-centric serving efficiency across models, tasks, and cluster conditions (Fig. 8–9, Fig. 10, Table 2).
