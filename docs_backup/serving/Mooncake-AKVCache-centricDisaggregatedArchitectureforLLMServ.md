# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

**ArXiv:** [2407.00079](https://arxiv.org/abs/2407.00079)

## 🎯 Pitch

Mooncake introduces a pioneering disaggregated architecture for large language model (LLM) serving, centered on distributed KVCache management and a global scheduler that intelligently coordinates prefill and decoding phases. By maximizing KVCache reuse, implementing chunked pipeline prefill, and leveraging prediction-based early rejection policies, Mooncake significantly boosts throughput—up to 525% in simulations and 75% more real-world requests—while reliably meeting strict latency SLOs in highly overloaded conditions. This approach addresses the critical challenge of scalable, efficient LLM serving under resource constraints, making advanced models more accessible and responsive for real-world applications.

---

## 1. Executive Summary (2-3 sentences)
Mooncake introduces a KVCache‑centric, disaggregated serving architecture for large language models (LLMs) that separates the “prefill” phase (processing the input prompt) from the “decoding” phase (autoregressive generation), and treats cache scheduling as the core optimization problem. By combining a distributed KVCache, a global scheduler (“Conductor”), chunked pipeline prefill, and a prediction‑based early‑rejection policy, Mooncake improves throughput under strict latency SLOs and overloaded conditions, reporting up to 525% higher throughput in long‑context simulations and 75% more served requests on real workloads while meeting SLOs (Abstract; §8.1.2; §8.1.3).

## 2. Context and Motivation
- The specific gap addressed
  - LLM serving has two very different phases: a computation‑heavy prefill stage and a memory‑bound decoding stage. Service providers must maximize goodput (throughput that meets SLOs) while ensuring two latency SLOs: time to first token (`TTFT`) and time between tokens (`TBT`) (§2).
  - Most prior work assumes sufficient capacity and focuses on utilization; in reality, providers often run in overloaded conditions where GPUs are scarce. In overload, naïvely accepting all requests wastes resources (e.g., when a request completes prefill but is later rejected for decoding) (§1.1; §7.1–§7.2).
  - KVCache reuse (reusing attention keys/values computed for a shared prefix) can save compute but makes scheduling harder: remote cache reuse raises `TTFT`, larger decoding batches lower `TBT` performance (§1.1, Fig. 2).

- Why this matters
  - Revenue and user experience hinge on serving more requests within SLOs. Long‑context workloads (e.g., 16k–128k token prompts) make prefill extremely expensive while decoding is memory‑bounded, so poorly coordinated scheduling can severely violate either `TTFT` or `TBT` (§2; Fig. 2).

- Prior approaches and shortcomings
  - Coupled serving (e.g., vLLM) shares resources between prefill and decoding, so long prefill jobs can disrupt decoding latency (§8 Baseline; §8.1.2).
  - Disaggregation (e.g., Splitwise, DistServe, TetriInfer) recognizes prefill/decoding differences, but:
    - They do not center the global scheduler on KVCache placement/migration; cache hotspots and remote fetch congestion are underexplored (§6.2).
    - Overload‑specific policies (predictive early rejection to avoid wasted prefill) and the resulting load‑fluctuation problem are not addressed (§7).
    - Long‑context acceleration via sequence parallelism (SP) demands frequent cross‑node communication and complex elastic resizing; it competes with KVCache transfers (§5.1).

- Positioning
  - Mooncake disaggregates prefill and decoding, builds a distributed KVCache using underutilized CPU/DRAM/SSD with RDMA transfer (Fig. 1, Fig. 3), and makes KVCache placement the first‑class scheduling target (Algorithm 1; §6). It also introduces chunked pipeline prefill for long contexts (§5.1) and prediction‑based early rejection that stabilizes load in overload (§7.4; Fig. 10b).

Definitions used once then assumed:
- `KVCache`: the per‑layer attention keys/values saved during prefill and extended during decoding; reusing it avoids recomputing shared prefixes.
- `Prefill`: one‑shot parallel processing of the entire input prompt to produce the first token and populate KVCache.
- `Decoding`: autoregressive generation, one token at a time per sequence.
- `TTFT`: latency from request arrival to first token.
- `TBT`: latency between consecutive tokens for a request.
- `Disaggregated architecture`: separating prefill and decoding into distinct node pools.
- `MFU` (Model FLOPs Utilization): how much of theoretical compute is realized.

## 3. Technical Approach
Mooncake’s core is a KVCache‑centric, disaggregated system with a global scheduler (“Conductor”) that jointly chooses where to prefill, where to decode, and where to place/migrate cache (Fig. 1; §3; Algorithm 1).

Step‑by‑step architecture and workflow (Fig. 4; §3):
1) KVCache reuse on prefill
   - Requests are tokenized and split into fixed‑size blocks (512 tokens used in the trace; §4.1).
   - A prefix‑hash is computed per block that chains all prior blocks’ hashes, enabling deduplication of any prefix (Fig. 3; “A=Hash(a) … E=Hash(D+e)”).
   - Conductor selects a prefill node considering:
     - Prefix match length available locally vs. remotely (more reuse = less compute),
     - Current queueing load (shorter wait = lower `TTFT`),
     - DRAM availability (cache residency constraint; Fig. 1; §6.1).
   - If beneficial and under a threshold, it pre‑migrates hot blocks to reduce future remote fetches (§6.2).

2) Incremental prefill (possibly chunked)
   - If uncached prompt length exceeds `prefill_chunk` (typically >1k tokens), the prefill splits the prompt into chunks that run in a pipeline across multiple nodes (§5.1, “chunked pipeline parallelism/CPP”). This reduces `TTFT` for long contexts by parallelizing different prompt chunks across nodes, with minimal per‑layer communication (only at pipeline boundaries).

3) Layer‑wise KVCache transfer
   - A dedicated RDMA service (“Messenger”) asynchronously streams each layer’s KVCache to the chosen decoding node’s CPU DRAM as soon as it’s produced (§3; Fig. 4 “Layer‑wise Load and Store”; §5.2). This overlap hides transfer latency and reduces GPU VRAM residency during prefill (Fig. 7).

4) Decoding with continuous batching
   - After the full KVCache lands in the decoding node’s CPU DRAM, it is loaded to GPU memory and the request joins the next decoding iteration (§3; §2). A local scheduler double‑checks `TBT` SLO given the most recent load; if violated, it rejects late, wasting any prefill work—this motivates early rejection (§3 step 4; §7).

Key design choices and why:
- Disaggregate prefill vs. decoding (Fig. 1)
  - Different objectives and constraints: maximize cache reuse and meet `TTFT` in prefill (DRAM bound), maximize throughput and meet `TBT` in decoding (VRAM bound). Coupling them forces trade‑offs and interference (Fig. 2; §2).
- KVCache‑centric scheduling (Algorithm 1; §6.1–§6.2)
  - Scheduling minimizes `TTFT` considering prefix hit length and queue time, and triggers cache replication/hotspot migration to avoid remote fetch congestion and to balance cache locality across prefill nodes (§6.2, Fig. 8).
- Chunked pipeline prefill (CPP) instead of sequence parallelism (SP)
  - SP reduces compute per node but still requires per‑layer cross‑node communication (ring/striped attention), degrades MFU, and needs complex elastic resizing (§5.1). CPP pipelines chunks across nodes with only stage‑boundary transfers, overlapping comm/compute, and naturally fits both short and long prompts without dynamic reconfiguration (§5.1).
- Layer‑wise prefill with async load/store
  - Overlaps per‑layer KVCache writes/reads with compute so prefill latency is close to the max of compute or KV transfer time; reduces VRAM residency during prefill and lets prefill scheduling ignore VRAM as long as a single request fits (Fig. 4; Fig. 7; §5.2).
- Prediction‑based early rejection
  - Rejects requests at admission if future decoding load will breach `TBT`, preventing wasted prefill and stabilizing disaggregated load (Fig. 9, Fig. 10b; §7.2–§7.4).

Algorithmic details (Algorithm 1; §6.1):
- For each request, compute `block_keys = PrefixHash(tokens, B)` and find the best prefix match across prefill nodes.
- Estimate `TTFT = Ttransfer + Tqueue + Tprefill`, where:
  - `Ttransfer` depends on remote block length and instantaneous network congestion (§6.1),
  - `Tqueue` is sum of queued prefill times on the instance,
  - `Tprefill` is predicted from offline profiling by request length and matched prefix (§6.1).
- Choose the prefill node with minimal predicted `TTFT` that meets the `TTFT` SLO.
- Independently select a decoding node with a predicted `TBT` that meets the `TBT` SLO.
- If either SLO would be violated, reject early; otherwise, proceed and optionally replicate hotspot cache blocks from the best‑match holder if the “balancing threshold” indicates consolidation is needed (§6.2, footnote on the manually tuned threshold).

Overload scheduling and the fluctuation fix (§7):
- Problem: Early rejection based on current decoding load causes anti‑phase oscillations between prefill and decoding loads due to prefill→decode lag (Fig. 9; Fig. 10a).
- Fix: Predict decoding load at the time the request would arrive for decoding, using a system‑level model that assumes a uniform decoding time `t_d` per request and updates the predicted batch/TBT status accordingly. Admit only if predicted `TBT` meets SLO (Fig. 10b; §7.4).

KVCache system and APIs (§3–§4):
- KVCache is paged and stored in CPU DRAM/SSD pools with eviction (LRU/LFU/LengthAware; Table 1).
- Prefix‑hash chaining enables deduplication and safe sharing of any prefix across sessions (Fig. 3).
- Messenger uses GPUDirect RDMA to move cache blocks between nodes asynchronously (Fig. 4).
- The system exposes a prefix cache API to external users for higher reuse (§3).

## 4. Key Insights and Innovations
- KVCache‑centric global scheduling as the first‑class objective
  - What’s new: Request placement and cache placement are jointly optimized, including on‑the‑fly cache replication/migration to reduce remote fetches and network hotspots (§6.2).
  - Why it matters: Directly targets the core compute savings lever (prefix reuse) while respecting `TTFT` and `TBT` SLOs; reduces network congestion and rebalances cache locality (Fig. 8).

- Disaggregated prefill/decoding with a distributed KVCache
  - What’s new: Treats CPU/DRAM/SSD attached to GPU nodes as an RDMA‑connected, near‑GPU cache pool; streams KVCache layer‑wise to decoders (§3; Fig. 1, Fig. 4).
  - Why it matters: Enables large, low‑cost cache capacity and high‑bandwidth transfers without extra hardware; supports near‑GPU prefix caching while freeing VRAM pressure (§3; §5.2).

- Chunked pipeline prefill (CPP) for long contexts in inference
  - What’s new: Applies pipeline parallelism to the prefill phase of inference (distinct from training), splitting long prompts across nodes with low comms overhead (§5.1).
  - Why it matters: Cuts `TTFT` for very long prompts while avoiding SP’s frequent per‑layer communication and complex elasticity; reduces network contention with KV transfers (§5.1).

- Prediction‑based early rejection to stabilize disaggregated overload
  - What’s new: An admission policy that predicts future decoding load (system‑level) at the time a prefilling request would arrive to decode; it avoids anti‑phase load oscillations inherent to disaggregated early rejection (§7.3–§7.4; Fig. 10b).
  - Why it matters: Saves wasted prefill compute when decoding will be overloaded, and increases effective capacity under overload (Table 3).

- Layer‑wise KV load/store overlap during prefill
  - What’s new: Preload the next layer’s KV and asynchronously store the current layer’s KV, overlapping with compute (§5.2).
  - Why it matters: Reduces end‑to‑end prefill latency overhead from KV I/O and allows prefill scheduling to largely ignore VRAM constraints for single‑request capacity (Fig. 7).

## 5. Experimental Analysis
- Evaluation methodology
  - Testbed: Multi‑node cluster, each node with 8× NVIDIA A800 80GB (NVLINK), 800 Gbps RDMA; nodes run either prefill or decoding (§8 Testbed).
  - Datasets and workloads (Table 2; §8.1):
    - Public: ArXiv Summarization (avg input 8,088; output 229; ~0% cache reuse), L‑Eval (avg input 19,019; output 72; >80% cache reuse).
    - Simulated: prompts of 16k/32k/64k/128k with 512‑token outputs, ~50% cache reuse.
    - Real trace: 23k requests sampled from production with timestamps, input/output lengths, and prefix‑hash IDs (§4; Fig. 5 shows length distributions).
  - Metrics and SLOs: P90 `TTFT` and `TBT` normalized to SLO limits. In end‑to‑end tests, `TTFT_P90 = 10×` and `TBT_P90 = 5×` of the single‑request baseline; real‑trace replay uses absolute caps of 30 s TTFT and 0.1 s/token TBT (§2; §8 Metric).
  - Baseline: vLLM with continuous batching and PagedAttention; coupled prefill/decoding (§8 Baseline).

- Main results
  - Public datasets (Fig. 11):
    - With 4 nodes total, Mooncake [3 prefill + 1 decode] outperforms vLLM [4 monolithic] by ~20% (ArXiv) and ~40% (L‑Eval) in achievable request rate while meeting both SLOs; [2P+2D] yields better `TBT` but worse `TTFT` due to prefill/decoding imbalance.
  - Long‑context simulated data (Fig. 12):
    - Mooncake sustains batching and SLOs while vLLM must fall back to single‑request processing to protect `TBT`. Reported throughput gains range from 50% up to 525% as prompt length grows to 128k, reflecting the value of disaggregation and cache reuse.
  - Real production trace (Fig. 13):
    - With Mooncake [10P+10D] vs vLLM [20 mixed], both show nearly 100% SLO satisfaction for `TTFT`. For `TBT`, Mooncake satisfies ~100% of requests while vLLM satisfies ~57%. Under these SLOs, Mooncake processes ~75% more requests.
  - Overload policies (Table 3; §8.2):
    - Replaying the real trace at 2× speed on an 8P+8D cluster, the number of rejected requests drops from 4,183 (baseline late rejection) to 3,771 (early rejection) and further to 3,589 with prediction‑based early rejection—evidence that predictive admission increases effective capacity while stabilizing load (§7.4, Fig. 10b).
  - Cache analysis (Table 1; Fig. 6; §4.2):
    - In the sample trace, moving from 1k to 50k cache blocks raises hit ratio from ~30% to ~50% (LRU best). Popularity is highly skewed: >50% of blocks never hit while some hit tens of thousands of times (Fig. 6), justifying hotspot replication (§6.2).

- Support for claims
  - The results directly align with the design goals:
    - Disaggregation plus KVCache‑centric scheduling protects `TBT` under long contexts (Fig. 12) while maintaining `TTFT` via cache‑aware prefill placement (Fig. 11, Fig. 8).
    - Prediction‑based early rejection reduces wasted prefill and stabilizes disaggregated load (Table 3; Fig. 10).
  - Caveats:
    - The end‑to‑end experiments use a “dummy model” with LLaMA2‑70B‑like architecture and replayed traces, not proprietary models or content (Abstract; §1.2; §8). This aids reproducibility but may limit generality across architectures and workloads.

- Ablations and robustness
  - Scheduling ablation (Fig. 8): Compared random selection, basic load‑balancing, cache‑aware, and full KVCache‑centric strategies on 8P+8D and 23k real requests; the latter markedly lowers average `TTFT` and improves SLO attainment.
  - Cache policy ablation (Table 1): Compares LRU, LFU, and LengthAware; LRU performs best on this trace.
  - Load fluctuation analysis (Fig. 9, Fig. 10): Demonstrates the oscillation induced by naïve early rejection and how prediction mitigates it.

## 6. Limitations and Trade-offs
- Assumptions and model choices
  - Prediction‑based early rejection currently uses a system‑level model assuming uniform decoding time `t_d` per request (§7.4). This is coarse; request‑level output length prediction is left as future work.
  - The KVCache hotspot balancing uses a manually tuned threshold (`kvcache_balancing_threshold`) to decide between transfer vs. recompute; it is not yet fully adaptive (§6.2, footnote).
- Scenarios not fully addressed
  - Workloads with extremely low prefix reuse benefit less from KVCache‑centric scheduling; the design still helps via disaggregation but cache replication gains diminish (Table 1 “~0% cache ratio” for ArXiv).
  - Multi‑tenant priority policies and mixed SLO classes are future work (§10).
- System complexity and resource demands
  - Requires RDMA networking and tight orchestration across prefill/decoding pools plus a distributed cache; network contention between SP‑like schemes and cache transfers motivated CPP, but large‑scale deployments still need careful bandwidth management (§5.1; §6.1).
  - Layer‑wise overlap reduces but does not eliminate KV I/O costs; mis‑sized chunks or skewed workloads can still expose transfer bottlenecks (Fig. 7; §5.2).
- External validity
  - Results are on a high‑end cluster (A800s, 800 Gbps) and a dummy model; performance portability to other hardware stacks (e.g., PCIe‑only, lower bandwidth) and non‑LLaMA architectures may vary (§8 Testbed).
- Operational trade‑offs
  - Disaggregation mandates capacity planning for prefill vs decoding pools; misallocation can hurt `TTFT` or `TBT` (Fig. 11 shows [2P+2D] vs [3P+1D] trade‑off).
  - Aggressive early rejection improves goodput but can reduce perceived availability; admission control needs product‑level tuning.

## 7. Implications and Future Directions
- How this changes the landscape
  - Treating KVCache as the scheduling center of gravity—and building the serving system around cache placement, migration, and reuse—shifts LLM‑serving design from monolithic utilization tuning to data‑centric, disaggregated orchestration. This is especially impactful for long‑context workloads where prefill dominates compute and decoding is memory‑bound (Fig. 2; §2).
- Follow‑up research enabled
  - Smarter prediction: request‑level output‑length predictors (learned or retrieval‑assisted) to refine admission decisions (§7.4).
  - Adaptive cache governance: online learning to tune replication thresholds, eviction policies for partial hits vs expiration, and congestion‑aware placement (§6.2; §10).
  - Heterogeneous accelerators and operator disaggregation: offload attention (memory‑bound) to bandwidth‑optimized devices while keeping MLPs on compute‑optimized GPUs; early simulation suggests promising throughput gains (§10).
  - KVCache reduction methods: compression/quantization, salient‑token selection, layer sharing, or hybrid architectures (surveyed in §10) compound with Mooncake by increasing batch sizes and cache hit rates.
- Practical applications
  - Production LLM APIs with strict latency SLOs and bursty, long‑context traffic (chat with large documents, code assistants).
  - Cost‑effective batch APIs: Mooncake’s layer‑wise overlap and freed VRAM on prefill nodes can absorb low‑priority batch decoding where `TBT` is relaxed (§5.2).
  - Context‑caching as a product: the exposed prefix cache API allows enterprise workflows to seed reusable context across sessions (§3).

> Headline results to remember:
> - “Up to a 525% increase in throughput in simulated long‑context scenarios while meeting SLOs” (Fig. 12; §8.1.2).
> - “~75% more requests processed under real traces with `TTBT` SLO met for nearly 100% of requests vs ~57% on vLLM” (Fig. 13; §8.1.3).
> - “Predictive early rejection reduces overload rejections from 4,183 to 3,589 in 2× trace replay” (Table 3; §8.2).

Overall, Mooncake demonstrates that a KVCache‑first, disaggregated design—augmented with pipeline prefill, cache‑aware scheduling, and predictive admission—can deliver substantial, measurable gains for long‑context LLM serving under realistic SLOs and overloaded conditions.
