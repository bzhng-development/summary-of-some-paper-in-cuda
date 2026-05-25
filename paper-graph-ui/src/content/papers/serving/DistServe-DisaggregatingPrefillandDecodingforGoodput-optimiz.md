# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

**ArXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)

## 🎯 Pitch

DistServe pioneers a novel serving architecture for large language models by separating the 'prefill' and 'decoding' stages onto different GPUs and independently optimizing their resource allocation and parallelism. This disaggregation eliminates interference between the two crucial phases, enabling up to 7.4× higher request throughput or meeting 12.6× stricter latency service-level objectives compared to leading systems—dramatically lowering operational costs and ensuring top-tier response quality for latency-critical applications like chatbots and coding assistants.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces DistServe, a serving architecture that splits large language model (LLM) inference into two separate GPU pipelines—one for the prompt “prefill” and one for token-by-token “decoding”—and then co-optimizes resources for each to maximize per-GPU goodput under latency service-level objectives. By eliminating interference between the two phases and choosing tailored parallelism for each, DistServe serves up to 7.4× more requests or meets 12.6× tighter latency SLOs compared to state-of-the-art systems while keeping at least 90% of requests within latency constraints (Abstract; §6.2, Fig. 8–9).

## 2. Context and Motivation
- Problem/gap:
  - LLM inference has two distinct phases:
    - `Prefill`: processes the entire input prompt at once to produce the first output token.
    - `Decoding`: then generates subsequent tokens one by one, each depending on prior tokens.
  - User-perceived latency is therefore twofold:
    - `TTFT` (Time To First Token): duration of prefill.
    - `TPOT` (Time Per Output Token): average time per generated token after the first (Footnote 1; §1).
  - Most serving systems colocate prefill and decoding and use continuous batching to maximize aggregate throughput (tokens/sec) (§2.2). This creates interference between phases and couples resource/parallelism choices, making it hard to meet both TTFT and TPOT SLOs without over-provisioning.

- Why it matters:
  - Many applications (chatbots, programming assistants, summarization) have strict TTFT/TPOT requirements; failing either harms user experience (Abstract; §1).
  - GPUs are expensive; maximizing “goodput”—the maximum request rate per GPU that satisfies the latency SLO for a target fraction of requests—is key to lowering cost per query (§1).

- Where prior approaches fall short:
  - Colocated continuous batching slows decoding when a long prefill joins the batch and vice versa, raising both TTFT and TPOT (Fig. 2; §2.3 “Prefill-decoding interference”).
  - Chunked-prefill with piggybacked decodes (e.g., SARATHI) reduces but does not remove interference and adds overhead, including an O(N^2) increase in KV cache reads across chunks (§2.3).
  - Colocation also forces a single parallelism/resource plan for both phases, despite their different compute/memory characteristics and SLO targets (§2.3 “Resource and parallelism coupling”).

- Positioning:
  - DistServe disaggregates prefill and decoding onto different GPUs to remove interference and decouple resource/parallelism decisions. It then uses a placement algorithm and a simulator to pick phase-specific configurations that satisfy TTFT/TPOT while maximizing per-GPU goodput (§1, §3–§4).

## 3. Technical Approach
At a high level, DistServe (1) splits execution into separate prefill and decoding “instances,” (2) uses analysis/simulation to choose batching and parallelism for each phase, and (3) places instances to minimize cross-node KV cache transfer.

Step-by-step:

1) Disaggregate the two phases into separate instances (§2.3; Fig. 6)
- A `prefill instance` hosts model weights and performs only prefill to produce the first token. It outputs intermediate state—the `KV cache` (key-value tensors needed to compute attention in later steps)—and the first token.
- A `decoding instance` (also with model weights) receives the KV cache and continues token-by-token generation.
- Multiple prefill instances can feed one decoding instance to build larger decoding batches without delaying prefill (§2.3; §3.2).

2) Analyze phase-specific compute patterns and choose batching/parallelism accordingly (§3)

  Prefill analysis (§3.1)
  - Compute behavior:
    - Prefill often becomes compute-bound when prompts are moderately long (e.g., 512 tokens on a 13B model saturates an A100; Fig. 3a).
  - Batching guideline:
    - Identify a per-model/GPU length threshold `L_m` beyond which prefill is compute-bound; avoid batching beyond this because it just linearly increases latency for all requests (Fig. 3a; §3.1).
  - Parallelism choice:
    - Two model-parallel forms:
      - `Intra-operator (tensor) parallelism`: splits large matrix multiplications across GPUs; reduces execution time but needs fast GPU–GPU bandwidth.
      - `Inter-operator (pipeline) parallelism`: splits layers into stages; adds inter-stage communication but increases system capacity.
    - Queueing perspective and key equations (§3.1; Fig. 4):
      - With uniform input lengths, prefill behaves like an M/D/1 queue (Poisson arrivals, deterministic service). The average TTFT is:
        - `Avg_TTFT = D + (R·D^2) / (2·(1−R·D))` (Eq. 1), where `D` is per-request execution time and `R` is arrival rate.
      - With 2-way inter-operator (pipeline) parallelism:
        - `Avg_TTFT_inter = D + (R·D^2) / (4·(2 − R·D))` (Eq. 2).
      - With 2-way intra-operator (tensor) parallelism and imperfect speedup K (1<K<2):
        - `Avg_TTFT_intra = D/K + (R·D^2) / (2·K·(K − R·D))` (Eq. 3).
      - Insight (Fig. 4a–b):
        - At low rates, reducing execution time (via intra-op) dominates.
        - At high rates, reducing queueing (via inter-op) dominates.

  Decoding analysis (§3.2)
  - Compute behavior:
    - Each decoding step handles only one new token per request; the step is memory-bandwidth-bound. GPU utilization is low unless batching across many concurrent requests (Fig. 3b).
  - Batching guideline:
    - Use many prefill instances to supply a large decoding batch without delaying prefill (§3.2).
    - Batch size may become limited by KV cache memory; techniques like PagedAttention and GQA can help scale (§3.2).
  - Parallelism choice under large decoding batches (approaching compute-bound):
    - `Intra-op` reduces per-step latency but has diminishing returns due to communication and underutilization.
    - `Inter-op` nearly linearly increases throughput; preferred once TPOT meets the SLO (Fig. 5).

3) Manage KV-cache communication (§3.3; §4.2)
- KV caches can be large: for OPT-66B, a single 512-token prefill produces ≈1.13 GB of KV cache; at 10 rps this would be ~90 Gbps (§3.3).
- DistServe places instances so that each prefill stage and its matching decoding stage reside on the same node; KV transfer then uses intra-node NVLINK (hundreds of GB/s on A100) rather than limited cross-node bandwidth (25 Gbps in the testbed) (§4.2).

4) Placement algorithms guided by simulation (§4; Alg. 1–2)
- DistServe searches over inter-op/intra-op configurations for each phase and uses a simulator to estimate per-GPU goodput under given SLO and workload.
- High-affinity network setting (fast cross-node, Alg. 1; §4.1):
  - Independently optimize prefill and decoding configurations via simulation (`simu_prefill`, `simu_decode`), then replicate each to match target traffic. Complexity is O(N·M^2) where N is max nodes per instance and M is GPUs per node (§4.1).
- Low-affinity network setting (limited cross-node bandwidth, Alg. 2; §4.2):
  - Constrain placements so that corresponding pipeline stages of prefill/decoding are colocated on the same node to use NVLINK for KV transfer.
  - Enumerate feasible intra-node combinations for both phases and pick the best via simulation; then replicate (§4.2).

5) Runtime and scheduling (§4.3; Fig. 6)
- Central controller dispatches:
  - Requests to the prefill instance with the shortest queue (FCFS).
  - After prefill, KV caches are “pulled” by the decoding instance that is least loaded (decoding fetches KV when ready), preventing bursts from overwhelming decoder memory (§4.3 “Combat burstiness”).
- Reducing pipeline bubbles with uneven prompt lengths:
  - Prefill: target batch total length near `L_m`; if a request exceeds `L_m` run it alone, otherwise pack multiple (§4.3 “Reducing pipeline bubbles”).
  - Decoding: operate at the largest stable batch size (§4.3).
- Replanning:
  - Periodically refit workload statistics and rerun the placement search; search finishes in seconds and model reloading in minutes, suitable for hour-scale workload drifts (§4.3; §6.5 Fig. 12).

6) Simulator and latency model (Appendix A; §4.1; §6.4)
- Simulator estimates per-phase execution time from GEMM and attention costs, fitted via profiling constants:
  - Prefill time `T_pre`: compute-bound GEMMs + memory-bound attention with FlashAttention; depends on total tokens `t` and squared lengths `t2` (Appendix A.2).
  - Decode time `T_dec`: memory-bound GEMMs at batch size `B` plus memory-bound attention proportional to the sum of generated tokens (Appendix A.3).
- Accuracy: SLO attainment predictions match real measurements within 2% across rates (Table 2; §6.4).

Implementation highlights (§5)
- Orchestration in Python, parallel engine in C++/CUDA; Ray actors manage GPU workers; NCCL for cross-node transfers and async `cudaMemcpy` intra-node; supports FlashAttention, PagedAttention, continuous batching; evaluated on OPT and LLaMA families.

## 4. Key Insights and Innovations
- Disaggregate prefill and decoding to remove interference (fundamental):
  - When colocated, a single long prefill delays all decodes in the same batch, raising TPOT; adding decodes into a prefill batch also raises TTFT—both visible in Fig. 2 (left and right panels). Disaggregation eliminates this mutual slowdown (§2.3).
- Goodput-centric, phase-specific parallelism planning (fundamental):
  - Use queueing analysis plus simulation to pick intra-op vs inter-op differently for prefill and decoding, and to select batching strategies that match their distinct SLOs (Fig. 3–5; Eq. 1–3; §3).
- Bandwidth-aware placement with stage colocation (novel engineering insight):
  - In low cross-node bandwidth clusters, colocate corresponding pipeline stages of prefill/decoding to keep KV transfer on NVLINK (§4.2). This makes communication overhead “insubstantial”: <0.1% of end-to-end latency for OPT-175B (Fig. 10).
- Lightweight, accurate simulation for rapid planning (incremental but impactful):
  - The latency model (Appendix A) coupled with replayed workload distributions yields plan searches in seconds with ≤2% error (Table 2; §6.5 Fig. 12).

## 5. Experimental Analysis
Methodology (§6.1)
- Cluster: 4 nodes × 8×A100-80GB (NVLINK intra-node, 25 Gbps inter-node). Low-affinity algorithm used unless noted (§6.1).
- Models: OPT-13B, OPT-66B, OPT-175B (§6.1).
- Workloads and SLOs (Table 1; Fig. 7):
  - Chatbot (ShareGPT): TTFT/TPOT SLOs scale with model size (e.g., 13B: 0.25s/0.1s; 175B: 4.0s/0.2s).
  - Code completion (HumanEval, 66B): 0.125s TTFT, 0.2s TPOT.
  - Summarization (LongBench, 66B): 15s TTFT, 0.15s TPOT; very long inputs.
- Metric: `SLO attainment`—fraction of requests whose TTFT and TPOT both meet their SLOs. Reported vs per-GPU rate and vs “SLO Scale” (both SLOs multiplied by a factor) (§6.1).
- Baselines (§6.1):
  - vLLM: continuous batching + PagedAttention; intra-op fixed at 1/4/8 for 13B/66B/175B.
  - DeepSpeed-MII: chunked-prefill with piggybacked decodes; not runnable for 175B in their environment due to kernel constraints (§6.1).

Main results (all at 90% SLO attainment unless noted)

- Chatbot, ShareGPT (Fig. 8):
  - Per-GPU rate:
    - DistServe sustains 2.0×–4.6× higher rate than vLLM across 13B–175B.
    - DistServe sustains 1.6×–7.4× higher rate than DeepSpeed-MII (13B, 66B).
  - Tightest SLO Scale achievable at fixed rate:
    - DistServe handles 1.8×–3.2× tighter SLOs than vLLM and 1.7×–1.8× than DeepSpeed-MII (second row of Fig. 8).
  - Why: Phase decoupling preserves low TPOT despite long prefills; vLLM’s colocated batches inflate TPOT (Fig. 2; §6.2). The chosen placement for 175B (Appendix B) uses prefill TP=3, PP=3 and decoding TP=4, PP=3, balancing load (§6.2).

- Code completion, HumanEval (66B) (Fig. 9a):
  - Rate: DistServe 5.7× higher than vLLM; 1.6× higher than DeepSpeed-MII.
  - SLO Scale: DistServe 1.4× tighter than both.
  - Why: TTFT is very tight here; reducing prefill execution time with intra-op where it matters and avoiding decode interference lets DistServe meet TTFT more often (§6.2).

- Summarization, LongBench (66B) (Fig. 9b):
  - Rate: DistServe 4.3× higher than vLLM; 1.8× higher than DeepSpeed-MII.
  - SLO Scale: DistServe 12.6× tighter than vLLM; 2.6× tighter than DeepSpeed-MII.
  - Why: Long inputs make prefill heavy; colocated systems inflate TPOT, failing tight TPOT SLO even if TTFT is loose (§6.2).

Robustness and supporting studies
- Latency breakdown (Fig. 10):
  - On OPT-175B, ShareGPT: KV transfer is <0.1% of total latency; >95% of transfers complete in <30 ms thanks to NVLINK stage colocation (§6.3).
- Simulator accuracy (Table 2): ≤2% error for both vLLM and DistServe-Low across rates on real cluster (§6.4).
- Ablations (Fig. 11):
  - vLLM++ (best intra-op chosen) ≈ vLLM → tuning parallelism within colocation brings little benefit due to prefill/decoding interference.
  - DistServe-High (no bandwidth constraints) > DistServe-Low (NVLINK colocation constraint), showing that fewer placement constraints can yield even higher goodput (§6.4).
- Algorithm runtime (Fig. 12):
  - Runs in seconds to minutes depending on GPUs considered; highly parallelizable; independent of model size because the simulator is event-based (§6.5).

Do the experiments support the claims?
- Yes. Improvements are large and consistent across models and applications, and analyses connect back to mechanisms: interference removal (Fig. 2), phase-tailored parallelism (Fig. 3–5; Eq. 1–3), and bandwidth-aware placement (Fig. 10; §4.2). Additional 99% SLO attainment results in Appendix C preserve the advantage (e.g., rate 3×–8× vs vLLM; SLO 1.24×–6.67× tighter).

## 6. Limitations and Trade-offs
- Requires duplicated model weights:
  - Prefill and decoding instances each host weights, increasing GPU memory footprint versus a single colocated instance (§2.3). This may limit applicability on very small clusters.
- Depends on workload prediction:
  - Placement uses workload distributions learned from history; abrupt shifts could degrade SLO attainment until replanning (§4.1; §4.3 “Replanning”).
- Bandwidth constraints still matter:
  - While stage colocation keeps most traffic on NVLINK, extremely bandwidth-constrained or fragmented deployments might reduce the benefit; cross-node transfers can become a bottleneck without careful placement (§3.3; §4.2).
- Missing runtime features:
  - No preemption or fault tolerance yet; failures in a decoding instance can affect multiple prefill instances and vice versa (§4.3 “Preemption and fault tolerance”).
- Scenario fit:
  - DistServe targets goodput under latency SLOs. For pure throughput-optimization (offline batch generation), chunked-prefill with piggybacking may be preferable (§7 “Throughput-optimized scenarios”).
- Resource-constrained settings:
  - With few or single GPUs, disaggregation has little room to operate; simpler colocated systems may be easier to manage (§7 “Resource-constrained scenarios”).

## 7. Implications and Future Directions
- Landscape impact:
  - Shifts LLM serving design from maximizing aggregate throughput to meeting per-request TTFT/TPOT SLOs cost-effectively. Phase-specialized scheduling and resource allocation become first-class considerations.
- Practical applications:
  - Real-time chat, code assistants, and interactive tools benefit from lower TTFT and stable TPOT without GPU over-provisioning; providers can cut cost per query while meeting user-perceived latency targets (Abstract; §6.2).
- Follow-up research:
  - Integrate preemption and fault tolerance that respect the two-phase dependency structure (§4.3).
  - Reduce memory duplication via shared-weight schemes or memory disaggregation.
  - Extend placement to multi-tenant clusters with fairness/priority and dynamic SLOs.
  - Combine with speculative decoding, long-context models (GQA/MQA, 1M-token contexts), and KV-cache streaming systems; the paper argues disaggregation remains valuable as prefill-vs-decode disparities grow with longer contexts (§7 “Long-context scenarios”).
  - Explore online learning for workload prediction and adaptive batching/parallelism beyond periodic replanning.

> Key quantitative takeaway: “DistServe can serve 7.4× more requests or 12.6× tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for >90% of requests” (Abstract; detailed per-task improvements in Fig. 8–9).
