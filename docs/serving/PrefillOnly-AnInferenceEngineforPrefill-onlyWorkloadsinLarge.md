# PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications

**ArXiv:** [2505.07203](https://arxiv.org/abs/2505.07203)

## 🎯 Pitch

PrefillOnly introduces the first LLM inference engine tailored specifically for 'prefill-only' workloads—applications that require only a single output token per request, such as recommendations or credit checks. By innovatively reducing active GPU memory via hybrid prefilling and leveraging predictable job completion times for optimized scheduling, PrefillOnly achieves up to 4× higher throughput and handles up to 5× longer input sequences compared to existing systems, all without compromising latency. This unlocks the efficient use of LLMs in high-volume, discriminative enterprise applications, where resource efficiency and scalability are critical.

---

## 1. Executive Summary
PrefillOnly is a serving engine specialized for “prefill-only” workloads—LLM applications that need just one output token per request (e.g., Yes/No decisions). It introduces a memory-lean execution path (hybrid prefilling) and a job-completion-time (JCT)–aware scheduler that exploits the determinism of one-token outputs. Across 4 hardware setups and 3 models, it achieves up to 4× higher queries-per-second (QPS) without increasing mean or P99 latency, and extends maximum input length (MIL) by up to 5× without parallelizing inference (Abstract; §7; Table 2; Fig. 6–8).

## 2. Context and Motivation
- Problem addressed
  - Many emerging LLM uses in discriminative tasks—recommendation, credit verification, data labeling—require only a single token output per request (§1, §2.2–2.3). This “prefill-only” pattern does not benefit from storing all key–value (KV) caches for multi-token decoding, and its JCT is predictable (always one token), but mainstream engines still assume variable output lengths (§1, §2.5–2.6).
  - Two missed opportunities in existing systems:
    - Excess GPU memory: KV caches for all layers and tokens are retained even though prefill-only requests do not decode multiple tokens (§1; Fig. 1b).
    - JCT uncertainty: conventional schedulers avoid JCT-based policies because outputs are variable-length. For prefill-only, JCT is predictable; ignoring this wastes scheduling leverage (§1, §2.5–2.6).

- Why it matters
  - Real deployments (recommendation, risk scoring) have very long inputs (user histories, credit logs) and huge traffic (tens of thousands of QPS), making GPU memory and throughput decisive (§2.4).
  - Mixing generative and prefill-only jobs causes severe interference, so prefill-only workloads often get dedicated GPU pools (§2.4).

- Shortcomings of prior approaches (§2.5)
  - Chunked prefilling: chunks the input to fit memory but slows attention kernels and cuts throughput (measured 14% drop for a 20k-token input with 512 token chunks; §2.5).
  - Tensor parallelism: increases MIL but imposes costly all-reduce communication; can increase latency without high-speed interconnects and always reduces overall throughput (§2.5).
  - Pipeline parallelism: introduces “pipeline bubbles” when request lengths vary; even with chunking to align stages, chunking itself hurts kernel efficiency (§2.5).
  - JCT-agnostic scheduling (e.g., FCFS) leaves performance on the table when JCT is predictable (§2.5).

- Position relative to existing work
  - Builds on vLLM’s production-grade runtime but rethinks execution and scheduling to fully exploit the prefill-only regime (§3; §7.1).
  - Complements prefix-caching and KV-cache management research by changing when/what to keep, not the cache format; it is compatible with future offloading/compression (§5, §8, §9).

## 3. Technical Approach
PrefillOnly is an end-to-end serving stack that couples an execution path with lower active memory to a JCT-aware scheduler that favors cache hits. Figure references indicate where mechanisms are shown.

- System workflow (§3.1; Fig. 2)
  1. Profile run: Given a target maximum input length (MIL), PrefillOnly runs a synthetic forward pass to measure the peak GPU memory needed for inference at that MIL. Remaining memory is reserved for prefix KV caches, preventing runtime OOM (§3.1).
  2. Runtime:
     - Accepts requests over an OpenAI-compatible HTTP API; a request is tokenized and queued (§3.1).
     - The scheduler selects one request per “step” via JCT-aware scoring (below), sends it to executors over ZeroMQ RPC (§3.1).
     - Executors run a single forward pass that computes the one-token output probability (e.g., P(Yes), P(No)); output is returned to the client (§2.3, §3.1).

- Why not batching for prefill-only (§6.1)
  - Decoding workloads are memory-bandwidth–bound, so batching increases throughput with minor runtime growth. Prefill-only workloads are compute-bound; batching does not significantly improve throughput but increases mean latency (§6.1). Hence, PrefillOnly schedules requests one-by-one.

- Execution path: hybrid prefilling (§4)
  - Observation: Peak GPU memory during prefill is dominated not by KV caches but by large intermediate tensors in the MLP (non-attention) blocks (§4.1). Figure 3a shows periodic 17–20 GB spikes on Llama‑3.1‑8B for a 32,768-token input; these correspond to MLP intermediates. Figure 4 quantifies sizes: with bfloat16, an intermediate tensor of shape `32768×28672` holds 14× more values per token than a one-layer KV cache (§4.1; Fig. 4).
  - Mechanism: Process non-attention (linear) layers chunk-by-chunk while keeping attention layers unchunked (§4.2).
    - Linear layers are chunkable because each chunk is independent.
    - Attention remains unchunked to preserve high kernel efficiency (§4.2).
  - Implementation via `torch.compile` (§4.3):
    - Group consecutive linear ops into a “virtual layer,” iterate over input chunks for that group, and concatenate outputs.
    - Two memory optimizations (§4.3):
      - Output preallocation: pre-allocate the final output tensor and write each chunk in place to avoid temporary double-buffering.
      - In-place compute: reuse input buffer for output when shapes match (chunk i maps to the same slice in input/output).
  - Effect: Hybrid prefilling lowers peak memory without harming attention kernel performance; Figure 3b shows ≈2 GB lower peak in the same 32,768-token example.

- KV-cache policy: suffix discarding/offloading (§5.1)
  - Goal: Keep only prefix KV caches that future requests might reuse; discard or offload suffix caches to fit long inputs without cross-GPU cache parallelization.
  - Enabler: Hybrid prefilling completes each request in a single forward pass, so caches need not persist between multiple passes; this makes selective discarding practical (§5.1).
  - Implementation: Reuses vLLM’s sliding-window abstractions; does not modify hardware kernels (§5.1).

- Scheduler: continuous JCT calibration (§6.2–6.3; Algorithm 1; Fig. 5)
  - Challenge: With prefix caching, a request’s JCT depends on whether its prefix is currently cached. Cache residency changes over time as jobs run and evict each other. A one-time JCT estimate at arrival becomes stale (§6.2).
  - Policy:
    - Before each scheduling decision (“step”), recompute every waiting request’s score:
      - Let `ninput` be the request’s input tokens, `ncached` the number that hit the prefix cache now, and `Tqueue` its waiting time.
      - Compute `score = get_jct(ninput, ncached) − λ·Tqueue` (§6.3, Algorithm 1).
        - `get_jct` can be a profile-based regressor over a grid of `(ninput, ncached)` values (1k-token granularity; trained by linear regression; §6.3).
        - In practice, the proxy `ninput − ncached` strongly correlates with measured JCT (Pearson r = 0.987 on Qwen‑32B FP8 on 1×A100; §6.3).
      - Pick the request with the lowest score; `λ` is a fairness knob countering starvation (larger `λ` prioritizes long-waiting jobs; §6.3; Fig. 11).
  - Why it works: Requests that can reuse fresh prefix caches see big JCT drops and get scheduled immediately, maximizing cache hits and preventing accidental evictions by unrelated jobs (§6.2–6.3; Fig. 5).

- Compatibility with parallelization (§5.2)
  - When there is no cache reuse and high-speed interconnects (e.g., NVLink) are available, tensor parallelism can reduce per-request latency at low QPS—but it sacrifices throughput due to communication. PrefillOnly targets high-QPS regimes where throughput dominates and communication overheads of parallelism hurt (§5.2; Fig. 8).

## 4. Key Insights and Innovations
- Prefill-only reframing (conceptual; §1–§2)
  - Insight: Many production decision tasks only need one output token. This makes output length fixed and JCT predictable while rendering most KV caches unnecessary (Fig. 1b).
  - Significance: Enables memory and scheduling strategies that are impossible or risky for generative decoding workloads.

- Hybrid prefilling (mechanism; §4)
  - What’s new: Chunk only the non-attention (linear/MLP) layers, which are responsible for peak memory, and keep attention layers intact.
  - Why it matters:
    - Cuts peak memory (Fig. 3b) by eliminating huge transient intermediates (Fig. 4) without harming attention efficiency (unlike full chunking).
    - Unlocks very long inputs without resorting to throughput-hurting parallelization or fully chunked prefill.
  - Distinct from prior chunked prefill: Previous approaches chunk the entire forward including attention, reducing kernel performance and total throughput (§2.5).

- Suffix KV cache discarding (policy; §5.1)
  - What’s new: Retain prefix caches likely to be reused, drop or offload suffix caches that won’t matter for single-token output.
  - Why it matters: Extends MIL substantially while preserving prefix reuse opportunities and avoiding cross-GPU cache sharding (§5.1, §5.2; Table 2).

- Continuous JCT calibration for scheduling (algorithm; §6.3)
  - What’s new: Recompute effective JCT for all waiting jobs before each step using current cache state; score includes a fairness term `−λ·Tqueue`.
  - Why it matters:
    - Proactively routes the schedule to harvest cache hits (illustrated in Fig. 5: achieves two cache hits vs. one under FIFO or naive SRJF).
    - Maintains low P99 while improving average latency in high-QPS settings (Fig. 6–7; Fig. 11).

- Generalizable implementation choices
  - Implemented atop vLLM with ~4.6k lines of Python; hybrid prefilling uses `torch.compile` graph rewriting, and KV policies reuse existing sliding-window abstractions—no custom kernels (§3.2, §7.1). This eases adoption across hardware and models.

## 5. Experimental Analysis
- Setup (§7.1)
  - Hardware and models (Table 3):
    - Low-end: 2× NVIDIA L4 (24 GB) with `Llama‑3.1‑8B`.
    - Mid-end: 2× A100 (40 GB) with `DeepSeek‑R1‑Distill‑Qwen‑32B‑FP8`.
    - High-end: 2× H100 (80 GB) with and without NVLink using `Llama‑3.3‑70B‑FP8`.
  - Workloads (Table 1):
    - Post recommendation (WL1): 20 users; profile length 11k–17k tokens; 50 posts/user, 150 tokens/post; high reuse potential; total ≈14M tokens.
    - Credit verification (WL2): 60 users; each history 40k–60k tokens; long-input stress test; total ≈3M tokens.
  - Arrival process: Poisson; QPS varied across {¼x, ½x, x, 2x, 3x, 4x}, where x is PrefillOnly’s saturated throughput with all requests arriving at once (§7.2).
  - Baselines (§7.1):
    - PagedAttention (vLLM’s FCFS with paging).
    - Chunked Prefill (Sarathi‑Serve style chunking; [3]; §2.5).
    - Pipeline Parallel (vLLM), degree 2.
    - Tensor Parallel (vLLM), degree 2.
    - All methods use prefix caching; non-parallel methods use user-id routing across GPUs.

- Main results
  - Throughput–latency tradeoff (Fig. 6; Fig. 7):
    - Quote: “PrefillOnly handles 1.4–4.0× larger query-per-second without inflating the average latency and P99 latency compared to baselines.” (Section 7 summary; also emphasized in Abstract).
    - Pattern: At high QPS, PrefillOnly consistently achieves the lowest mean and P99 latency across hardware and workloads (Fig. 6a–h; Fig. 7a–h).
    - At very low QPS, tensor/pipeline parallel can win on latency (more GPUs per request), but their peak throughput is lower and they scale worse as QPS rises (Fig. 6; discussion in §5.2, §7.2).
  - Maximum input length (MIL) (Table 2):
    - L4: PrefillOnly 130k vs PagedAttention 24k and Chunked 46k; pipeline/tensor reach 72k/195k but need parallelization. PrefillOnly supports both WL1 and WL2 at single-GPU inference.
    - A100: PrefillOnly 87k vs PagedAttention 11k, Chunked 17k; pipeline/tensor 38k/77k.
    - H100: PrefillOnly 97k vs PagedAttention 15k, Chunked 25k; pipeline/tensor 183k/238k.
    - Takeaway: PrefillOnly enables long sequences without parallelizing inference, covering both workloads across all GPU classes.
  - Source-of-improvement analyses (§7.2; Fig. 8–9):
    - WL1 (post recommendation): Chunked prefill’s throughput drops at high QPS due to “prefix cache throttling”; PrefillOnly avoids this by preferentially scheduling cache-hit jobs via continuous JCT calibration, while parallel baselines avoid throttling but pay communication overhead (Fig. 9; §7.2).
    - WL2 (credit verification): Tensor parallelism benefits from NVLink but still underperforms PrefillOnly in throughput; communication (all-reduce) and pipeline bubbles limit scaling (Fig. 8a–b; §7.2).
  - Hybrid prefilling ablation (Fig. 10):
    - Quote: “Hybrid prefilling improves the MIL by 7.9× without hurting the throughput, measured on a Qwen‑2.5‑32B model with FP8 on A100” (Fig. 10).
    - The figure also annotates contributions from chunking + preallocation + in-place optimizations.
  - Fairness knob λ (Fig. 11):
    - Increasing `λ` reduces P99 latency (curves shift left in the tail) at the cost of modestly higher mean latency; this demonstrates starvation control via the `−λ·Tqueue` term (§6.3; Fig. 11).

- Convincingness and caveats
  - Strengths:
    - Broad hardware coverage (L4/A100/H100; with/without NVLink).
    - Two workloads designed to stress cache reuse and long contexts (Table 1).
    - Clear head-to-head comparisons under the same serving stack (vLLM) and with prefix caching enabled for all methods.
    - Mechanism-level evidence (Fig. 3–4) that memory spikes come from MLP intermediates; hybrid prefilling directly targets that bottleneck.
  - Caveats:
    - Datasets are simulated (explicitly acknowledged; §7.1). External validity depends on how closely they mimic real traffic and prompts.
    - Many plots show qualitative trends; apart from Table 2 and Fig. 10, precise numeric deltas are not tabulated, which makes exact headroom less explicit.
    - A small discrepancy exists in the narrative around Fig. 10 (caption shows 7.9×; text in §7.2 mentions “more than 8.7×”). The figure should be taken as the source of truth.

## 6. Limitations and Trade-offs
- Scope limitations
  - Prefill-only assumption: The engine is specialized for single-token outputs. Multi-token generation is out of scope; mixing with generative decoding is discouraged due to interference (§2.4).
  - Output restriction: Prefill-only apps typically enforce an “allowed tokens” set (e.g., {Yes, No}) so that a single token is sufficient; this requires application-level discipline (§2.3, footnote 1).

- Cache policy trade-offs
  - Suffix KV discarding can reduce future reuse if a later request unexpectedly needs that suffix. The implementation currently discards rather than offloads; offloading to CPU is discussed as future work (§9).

- Scheduling assumptions
  - JCT proxy (`ninput − ncached`) is hardware- and model-calibrated; while correlation is high on the tested setup (r = 0.987, §6.3), different kernels or quantization could alter the mapping, requiring re-profiling.

- Generality and engineering constraints
  - Hybrid prefilling relies on clean graph segmentation of linear vs attention ops. Exotic architectures with nonstandard blocks or heavy non-linearities may need extra engineering.
  - `torch.compile` graph rewrites can be sensitive to framework versions and dynamic model code paths (§4.3, implementation detail).
  - Batching is intentionally avoided to lower latency in compute-bound prefill-only regimes; if a deployment is memory-bound or uses specialized attention kernels, the calculus could change (§6.1).

- Evaluation limits
  - Simulated data and Poisson arrivals may not capture all temporal correlations (bursts, heavy-tail lengths) seen in production (§7.1).
  - Results emphasize throughput and latency; energy efficiency and cost-per-decision are not reported.

## 7. Implications and Future Directions
- How this changes the landscape
  - Demonstrates that “prefill-only” is a distinct serving regime deserving its own engine. By embracing single-token determinism and discarding non-reusable caches, deployments can serve longer inputs on fewer GPUs and safely use JCT-aware scheduling.
  - Offers a practical path to run long-context decision pipelines (recommendation, credit scoring, labeling) without the complexity and overhead of model parallelism (Table 2; §5.2).

- Follow-up research enabled
  - Cache management:
    - CPU offloading of suffix caches (LMCache-style) to preserve reuse without inflating active GPU memory (§9).
    - Compatibility with cache compression (e.g., KVQuant, KIVI, H2O) and cache blending/fusion for RAG-style reuse (§8).
  - Scheduling:
    - Multi-tenant fairness, SLO-aware variants, and integration with cluster schedulers (building on the `−λ·Tqueue` formulation; §6.3).
    - Exploration of batched-vs-serial hybrid policies under specific hardware/kernel stacks.
  - System architecture:
    - Applying PrefillOnly to the “prefill node” in prefill–decode disaggregation systems (e.g., DistServe), potentially improving overall goodput (§9).
    - Latency-centric optimizations specific to prefill-only, such as contiguous GPU allocations instead of paging (§9).

- Practical applications
  - High-volume ranking and filtering: candidate scoring in recommenders (WL1), safety/credit checks (WL2), spam/abuse detection where outputs are categorical.
  - Data operations: large-scale labeling/triage pipelines that need confidence scores (single-token probabilities) rather than long generations (§2.3).
  - Any LLM service where input is long, output is constrained to a small set of tokens, and predictable latency is critical.

> Representative headline result: “PrefillOnly can process up to 4× larger queries per second without inflating average and P99 latency.” (Abstract; reinforced by Fig. 6–7)

> Representative mechanism evidence: Peak memory is dominated by MLP intermediates (Fig. 3–4), and hybrid prefilling reduces peak by ≈2 GB on a 32k-token example without chunking attention (Fig. 3b), which underpins larger MIL (Table 2; Fig. 10).

Overall, PrefillOnly is a well-motivated rethinking of LLM serving for single-token decisions. Its hybrid prefilling and continuously calibrated JCT scheduling are the main technical levers, and the evaluation—while simulated—maps cleanly onto the operational pain points of long-context, high-QPS enterprise workloads.
