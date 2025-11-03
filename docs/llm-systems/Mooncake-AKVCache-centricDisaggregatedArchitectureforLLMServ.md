# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

**ArXiv:** [2407.00079](https://arxiv.org/abs/2407.00079)

## 🎯 Pitch

Mooncake introduces a novel KVCache-centric, disaggregated architecture for large language model (LLM) serving, separating the prefill and decoding stages across resource pools and elevating the KVCache to a distributed, schedulable system component. By combining cache-aware scheduling, chunked and layer-wise prefill, and predictive early rejection, Mooncake dramatically increases throughput—up to 525% in simulations and 75% more requests in production workloads—while reliably meeting strict latency requirements, representing a significant advancement in scalable, efficient, and robust LLM deployment for real-world, high-demand scenarios.

---

## 1. Executive Summary (2-3 sentences)
Mooncake is a KVCache‑centric, disaggregated architecture for serving large language models (LLMs) that splits the prefill and decoding stages across separate GPU node pools, and treats the key/value cache (KVCache) as a first-class, distributed resource to schedule and replicate. A global scheduler (“Conductor”) jointly optimizes throughput and latency SLOs—time to first token (TTFT) and time between tokens (TBT)—with cache-aware placement, chunked pipeline prefill, layer-wise cache streaming, and prediction-based early rejection under overload. Experiments show up to 525% higher throughput than a strong baseline in long-context simulations (Figure 12) and 75% more handled requests on real traces while meeting SLOs (Figure 13).

## 2. Context and Motivation
- Problem addressed
  - Serving LLMs must balance two conflicting goals:
    - Maximize effective throughput (completed tokens/requests that meet SLOs).
    - Satisfy latency-related SLOs: TTFT (time until the first output token) and TBT (interval between subsequent tokens).
  - Long-context workloads make prefill heavy and decoding memory-bound; batching to increase throughput can worsen TBT, and remote cache reuse can worsen TTFT (Abstract; §2; Figure 2).
  - GPU supply is constrained, so clusters often run overloaded. Traditional work typically assumes all requests will be processed and focuses on utilization rather than overload handling (§1.1, §7).

- Why it matters
  - Production LLM services mix short and long inputs, variable arrivals, and different SLO classes. Violating SLOs degrades user experience and revenue, while wasting compute on requests that will later be rejected is costly (§2).
  - KVCache reuse (reusing attention K/V tensors from repeated prefixes) can drastically cut prefill compute, but it introduces scheduling complexity around where caches reside and how to move them without violating TTFT (§3; Figure 3).

- Prior approaches and gaps
  - Strong open-source/industrial systems (e.g., vLLM, TensorRT-LLM, FasterTransformer) improve memory and batching efficiency but couple prefill and decoding, so long prefills can disrupt decoding TBT (§8 Baseline; Related Work §9).
  - Recent disaggregation proposals (Splitwise, DistServe, TetriInfer) separate prefill/decoding, but the field underexplores:
    - KVCache as a schedulable, distributed object with replication/migration impact on SLOs.
    - Overload-oriented scheduling with early rejection and load prediction (§1.1, §7).
    - How to scale prefill across nodes for very long contexts without heavy cross-node communication (§5.1).

- Positioning
  - Mooncake extends disaggregation with a “KVCache-centric” design: a near-GPU, disaggregated KV cache spanning CPU/DRAM/SSD and a dedicated RDMA transport (“Messenger”), a cache-aware global scheduler, and prefill innovations (chunked pipeline parallelism and layer-wise KV streaming) (§3, §5, §6).
  - It introduces overload-aware early rejection with system-level load prediction to stop oscillations (§7, Figures 9–10, Table 3).

## 3. Technical Approach
Key terms used throughout:
- `KVCache`: saved key/value tensors from attention to avoid recomputing past context. KVCache size grows with prompt length and number of layers.
- `Prefill`: the parallel computation over all input tokens to produce the first output token and build initial KVCache; compute-heavy for long inputs.
- `Decoding`: iterative generation (one token per step per sequence) using KVCache; memory-bound and sensitive to batching for MFU.
- `TTFT` and `TBT`: latency metrics; SLOs bound their P90 values (§2).
- `MFU` (Model FLOPs Utilization): fraction of theoretical compute used by the model per GPU; improves with larger batches but risks TBT SLO violations.
- `Disaggregated architecture`: prefill and decoding run on separate GPU node pools; KVCache is stored and moved via a distributed cache spanning CPU/DRAM/SSD and per-GPU VRAM.
- `Continuous batching`: dynamically add/remove requests to the decoding batch each iteration to keep GPUs busy.

A. System architecture and data path (Figures 1, 3, 4; §3)
- Disaggregated KV cache
  - KVCache is partitioned into “paged blocks” stored in CPU memory across nodes, spilling to SSD as needed. Each block has a “prefix hash” that chains the block’s hash with all preceding blocks, enabling deduplication and detecting reusable prefixes (Figure 3; §3–§4.1).
  - “Messenger” is an RDMA-based service in each node that transfers KVCache between nodes and between CPU and GPU memories efficiently (§3; step 3 in Figure 4).

- Global Conductor scheduler orchestrates four steps per request (Figure 4; §3):
  1) KVCache reuse: identify reusable prefix blocks via prefix hashes; prefill instance preloads reusable KVCache from the disaggregated pool to GPU (if beneficial for SLO); skip if no cache (§3 step 1; §6.1).
  2) Incremental prefill: compute remaining tokens; if uncached tokens exceed a `prefill_chunk` threshold (usually >1000 tokens), split into chunks that can be processed in a pipeline across multiple prefill nodes (details below in B); newly generated incremental KVCache is stored in the distributed cache (§3 step 2; §5).
  3) KVCache transfer: Messenger streams KVCache layer-by-layer from prefill to the selected decoding node’s CPU memory, overlapping transfer with ongoing prefill computation (§3 step 3; §5.2).
  4) Decoding: once full KVCache is in decoding CPU DRAM, the request joins the next decoding batch. A local scheduler double-checks the TBT SLO at admission time (§3 step 4).

B. Prefill at scale: Chunked Pipeline Parallelism (CPP) and layer-wise streaming (§5)
- Why not sequence parallelism (SP) for long contexts?
  - SP splits the sequence across nodes and reduces cross-node overhead vs. pure tensor parallelism, but still requires at least one cross-node communication per layer (e.g., Ring/Striped Attention). It competes with KVCache transfer bandwidth and lowers MFU for prefill (§5.1).
  - Elastic SP complicates deployment and global scheduling; scaling SP groups frequently is operationally heavy for diverse workloads (§5.1).

- CPP: how it works
  - Form prefill “pipeline groups” of X nodes. Split a long prompt into chunks of up to `prefill_chunk` tokens. Different chunks of the same request are processed in parallel by different nodes, moving through the pipeline, thereby reducing end-to-end TTFT without per-layer cross-node sync (§5.1).
  - Communication occurs only at pipeline stage boundaries and can be overlapped with compute (like training pipeline parallelism), improving MFU and freeing the network for KVCache transfers (§5.1).

- Layer-wise prefill streaming and VRAM frugality
  - Prefill computes attention layer by layer. Mooncake launches asynchronous KVCache load/store per layer: load cached KV before computing a layer; after computing, asynchronously store that layer’s KVCache. All overlaps with prefill compute (§5.2).
  - Benefit: the effective VRAM “occupation cost” (size × time) is minimized since transfer overlaps compute, and prefill scheduling can largely ignore VRAM capacity as long as a single request fits. Figure 7 shows reduced latency for storing KVCache across long sequences (8k–128k) compared to a serialized approach (§5.2, Figure 7).

C. KVCache-centric global scheduling (Algorithm 1; §6)
- Inputs: pools of prefill instances `P` and decoding instances `D`, request `R`, cache block size `B`.
- Step 1—Prefix matching: compute block-level prefix hashes for `R` and find per-instance prefix match lengths (`prefix_len`) using the per-instance cache index (Algorithm 1 lines 1, 4; §6.1; Figure 3).
- Step 2—TTFT estimate per prefill instance: for each candidate instance:
  - Estimate queue delay (`Tqueue`) from the sum of queued prefill times (§6.1).
  - Estimate prefill compute time (`Tprefill`) as a function of prompt length and `prefix_len` using an offline-fitted model (§6.1).
  - If the best-matched cache is remote: add KV transfer time (`Ttransfer`) based on block size/volume and current network condition; if transfer is slower than recomputing, prefer recompute (§6.2).
  - Choose the instance minimizing predicted TTFT = `Tqueue + Ttransfer (if any) + Tprefill` (Algorithm 1 lines 5–23; §6.1–6.2).
- Step 3—Decoding admission and SLO check: select a decoding instance with load balancing; estimate TBT for admission time; reject if predicted TTFT or TBT exceeds SLOs (Algorithm 1 lines 24–27; §6.1).
- Step 4—Cache hot-spot migration: if the cache hit elsewhere is sufficiently longer than local (`kvcache_balancing_threshold`), trigger replication from the “best matched” instance to the chosen one to avoid future network hotspots (Algorithm 1 lines 28–30; §6.2).

D. Overload-oriented scheduling and early rejection (§7, Figures 9–10, Table 3)
- Load definition in a disaggregated system:
  - Prefill load and decoding load are judged by whether predicted TTFT and TBT will meet their SLO thresholds (`l_ttft` and `l_tbt`) (§7.1).
- Problem: Naïve early rejection oscillations
  - If the system accepts/rejects solely based on “current” decoding load before prefill, the inevitable time lag between prefill completion and decoding admission creates anti-phase oscillations: prefill becomes full, then decoding becomes full and rejects new admissions, then prefill starves, and so on (observed in production, Figure 9; conceptualized in Figure 10a; §7.3).
- Solution: Prediction-based early rejection
  - Predict near-future decoding load at the expected time prefilled requests will arrive. Mooncake uses a system-level predictor assuming a uniform per-request decoding time `t_d`:
    - Add requests expected to finish prefill by future time `t` to decoding queues; remove requests whose decoding would complete before `t`; compute the projected average TBT/SLO ratio; admit or reject the new request accordingly (§7.4).
  - This reduces oscillations and lowers unnecessary rejections (Figure 10b; Table 3).

E. Real-world trace and cache policy analysis (§4; Table 1; Figures 5–6)
- Open trace: 23,608 entries from a 1-hour window (timestamps, input/output lengths, remapped prefix-hash IDs), preserving reuse relationships without raw text (§4.1; Listing 1).
- Length stats: average input 7,590 tokens; output 182 (§4.2; Figure 5).
- Cache skew: more than 50% of blocks unused; few blocks extremely hot (CDF in Figure 6); motivates replication to prevent transfer contention (§4.2).
- Cache policy comparison (single global pool, simulated): LRU achieves slightly higher hit rates than LFU/LengthAwareCache across capacities; going from 1k to 50k blocks increases hit rate from ~30% to ~50%; larger gains plateau (Table 1; §4.2).

## 4. Key Insights and Innovations
- KVCache as a schedulable, distributed resource (fundamental)
  - Treats KVCache like a first-class cache with prefix-hash indexing, RDMA transfer (“Messenger”), and scheduler-driven replication/migration (Figures 1, 3; §3, §6.2). This differs from local-only cache reuse in systems like vanilla vLLM and enables near-GPU prefix caching without dedicated new hardware.
- Chunked Pipeline Parallelism (CPP) for prefill (fundamental)
  - Scales a single long-context prefill across multiple nodes without per-layer all-reduce or ring attention. Communication happens only at chunk boundaries and overlaps with computation (§5.1). This reduces TTFT for very long contexts while consuming less network than SP.
- Layer-wise KVCache streaming in prefill (incremental but impactful)
  - Asynchronous per-layer load/store overlaps with compute to minimize VRAM residency and prefill latency (Figure 7; §5.2). This unlocks additional scheduling freedom: prefill placement can largely ignore VRAM as long as one request fits.
- KVCache-centric scheduling with heuristic hot-spot migration (incremental but practical)
  - Global scheduling (Algorithm 1) blends best-prefix matching, queueing, transfer-time estimation, and a thresholded policy for recompute vs. remote fetch; it also replicates hot prefixes opportunistically to prevent network congestion (§6.1–§6.2; Figure 8).
- Overload-oriented early rejection with system-level prediction (fundamental for production)
  - Identifies and fixes the oscillation failure mode of naïve early rejection in disaggregated pipelines by predicting decoding load at the time prefilled requests will arrive (§7.2–§7.4; Figures 9–10; Table 3).

## 5. Experimental Analysis
- Methodology and setup
  - Testbed: multi-node cluster; each node has 8× NVIDIA A800 80GB (NVLink), RDMA NICs up to 800 Gbps. Nodes run either prefill or decoding instances (§8 Testbed).
  - Datasets and workloads (Table 2; §8.1):
    - Public: ArXiv Summarization (avg input 8,088; output 229; ~0% cache), L-Eval (avg input 19,019; output 72; >80% cache).
    - Simulated: 16k/32k/64k/128k prompts, 512-token outputs, 50% cache.
    - Real: 23k-request trace with timestamps (from §4).
  - Metrics and SLOs:
    - Evaluate throughput as max request rate (RPS) while keeping P90 TTFT ≤ 10× and P90 TBT ≤ 5× their single-request baselines (§2; §8 Metric).
    - For real-trace replay: fixed SLO caps (e.g., TTFT ≤ 30s; TBT ≤ 0.1 s/token; Figure 13).
  - Baseline: vLLM with continuous batching and PagedAttention; coupled prefill/decoding (Baseline in §8).

- Main quantitative results
  - Public datasets (Figure 11):
    - Mooncake-[3P+1D] vs vLLM-[4M]: higher RPS before hitting SLO limits—about +20% on ArXiv Summarization and +40% on L‑Eval—while keeping normalized P90 TTFT and TBT ≤ 1.0.
    - Mooncake-[2P+2D] has lower TBT but can hit TTFT limits earlier due to prefill/decoding mix; this highlights the need to right-size the pools for each workload mix (§8.1.1).
  - Long-context simulated (Figure 12):
    - Mooncake maintains batching in decoding and isolates it from long prefills; vLLM must fall back to single-request processing to avoid TBT blow-ups.
    - Throughput improvement ranges from +50% up to +525% across 16k–128k prompts while meeting SLOs (§8.1.2).
  - Real-trace replay (Figure 13):
    - With 10 prefill + 10 decoding instances, Mooncake’s TTFT CDF overlaps with 20 vLLM instances (both near 100% under TTFT SLO).
    - TBT SLO attainment: ~100% for Mooncake vs ~57% for vLLM; net result is ~75% more handled requests for Mooncake while meeting SLOs (§8.1.3).
  - Scheduling and cache balancing (Figure 8):
    - Cache-aware and KVCache-centric scheduling significantly reduce average TTFT relative to random or pure load-balancing strategies; the KVCache-centric strategy attains the SLO more reliably (§6.2).
  - Early rejection under overload (Table 3; §8.2):
    - At 2× replay speed, number of rejected requests drops from 4,183 (baseline) to 3,771 (early rejection) and to 3,589 (prediction-based early rejection). This indicates less wasted prefill work and smoother utilization.
  - Cache policy analysis on the trace (Table 1; §4.2):
    - LRU achieves slightly higher hit rates than LFU/LengthAwareCache at moderate capacities; hit rate rises from ~0.30 at 1k blocks to ~0.50 at 50k, then plateaus.

- Do the experiments support the claims?
  - Isolation of prefill and decoding plus cache-centric scheduling convincingly improves TBT at high throughput (Figures 11–13).
  - Long-context scenarios are where Mooncake shines most (Figure 12), consistent with design choices (CPP, layer-wise streaming).
  - Overload handling is evaluated with a sharp, measurable outcome (number of rejected requests) and shows clear gains (Table 3).
  - The scheduling micro-result (Figure 8) connects architectural ideas with TTFT outcomes, supporting the KVCache-centric choice.

- Ablations, failure cases, robustness
  - The paper reports a scheduling comparison (random vs load-balancing vs cache-aware vs KVCache-centric; Figure 8).
  - It analyzes cache capacity/algorithm effects (Table 1) and hot-spot skew (Figure 6), informing replication decisions.
  - It does not report sensitivity studies for predictor parameters (e.g., non-uniform `t_d`) or detailed network congestion models, which could affect transfer-time estimation (§6.1 notes the difficulty of predicting `Ttransfer`).

- Conditions and trade-offs
  - The best prefill/decoding ratio depends on workload mix (Figure 11 shows Mooncake-[3P+1D] outperforming [2P+2D] on TTFT). Static ratios may underperform under rapid workload shifts (§8.1.1).
  - Accepting more KV reuse from remote nodes can increase TTFT if the network is congested; Algorithm 1 trades off recompute vs transfer (§6.1–§6.2).

## 6. Limitations and Trade-offs
- Assumptions and environment
  - Results use a “dummy” model architecturally equivalent to LLaMA2‑70B and replayed traces for privacy/reproducibility (§1.2; §8). Exact numbers may differ on other models/hardware.
  - The design assumes high-bandwidth RDMA networking and spare CPU/DRAM/SSD capacity to house the disaggregated cache (§3 Testbed context).
- Prediction and heuristics
  - Prefill time is estimated from offline profiling; decoding load prediction uses a system-level uniform `t_d` approximation (§6.1; §7.4). Workloads with highly variable per-token latencies or output lengths may reduce accuracy.
  - The threshold `kvcache_balancing_threshold` for when to replicate vs recompute is manually tuned (§6.2), leaving potential performance on the table.
- Scope and scenarios not fully addressed
  - Multi-tenancy with strict priority/fairness across classes is not fully developed (future directions mention priority-aware scheduling).
  - The approach assumes some degree of prefix reuse; when reuse is rare and inputs are short, benefits diminish (Table 1 indicates reuse saturates around 50% on the sampled trace).
- Resource trade-offs
  - CPP uses multiple nodes per long request; if workloads are dominated by many small prompts, pipeline groups may be underutilized (though CPP is designed to add minimal overhead for short contexts; §5.1).
  - Replication of hot KV blocks consumes DRAM/SSD and network bandwidth; aggressive replication could conflict with other transfers (§6.2).
- Operational complexity
  - Maintaining a distributed cache index, coordinating Messenger transfers, and handling cache expiration/eviction across nodes adds system complexity not present in single-node designs (§3–§6).

## 7. Implications and Future Directions
- How this work shifts the landscape
  - Elevates KVCache to a primary schedulable unit. This reframing encourages designs that colocate and pre-position context, rather than only optimizing GPU kernels or batch sizes.
  - Demonstrates that disaggregating prefill and decoding, plus cache-aware scheduling, is a practical way to sustain TBT SLOs at high throughput—especially for long-context services (Figures 11–13).

- Follow-up research enabled or suggested
  - Request-level output-length prediction to refine early rejection/admission and batch sizing (§7.4).
  - Automated, adaptive policies for `kvcache_balancing_threshold`, replication counts, and prefill/decoding pool sizes using online learning.
  - Richer network-aware `Ttransfer` models and proactive prefetch/placement of popular prefixes (Figure 6 hot-spot skew).
  - Integration with KVCache compression/selection methods (e.g., KIVI, ZipCache, PyramidKV) to increase batch sizes and cache hit rates (Related Work §10).
  - Heterogeneous accelerator disaggregation: offload memory-bound attention to bandwidth-optimized devices (Future Work §10).
  - Priority- and SLO-class-aware admission control; fairness across tenants (§10).

- Practical applications
  - High-throughput chat/search assistants with very long context windows.
  - Enterprise workloads with shared system prompts, tools, or documents benefiting from prefix caching APIs (§3 mentions exposing a context caching API).
  - Batch APIs or offline jobs: freed VRAM from layer-wise prefill could be used to inline decoding for non-interactive batches with relaxed TBT (§5.2).

> Headline results to keep in mind:
> - Up to 525% throughput gain vs. vLLM under long-context simulations while meeting TTFT/TBT SLOs (Figure 12).
> - ~20–40% throughput gains on public datasets at SLO compliance (Figure 11).
> - ~75% more requests processed on real traces with TBT SLO nearly 100% vs. ~57% for vLLM (Figure 13).
> - Early rejection with prediction reduces rejected requests from 4,183 (baseline) to 3,589 (Table 3), indicating less wasted prefill compute under overload.

Overall, Mooncake’s central idea—scheduling around KVCache in a disaggregated system—offers a concrete, system-level path to scale long-context LLM serving under tight SLOs and constrained hardware.
