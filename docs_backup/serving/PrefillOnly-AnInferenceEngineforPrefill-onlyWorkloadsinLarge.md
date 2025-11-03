# PrefillOnly: An Inference Engine for Prefill-only Workloads in Large Language Model Applications

**ArXiv:** [2505.07203](https://arxiv.org/abs/2505.07203)

## 🎯 Pitch

PrefillOnly introduces the first inference engine purpose-built for 'prefill-only' LLM workloads—where only a single output token is generated, as in recommendation or credit scoring. By combining a novel hybrid prefilling technique that slashes active GPU memory requirements with a job-completion-time-aware scheduler that exploits prefix cache reuse, PrefillOnly delivers up to 4× higher throughput and supports much longer inputs compared to existing systems. This unlocks efficient, scalable deployment of LLM-powered discriminative applications, transforming how resource-hungry AI workloads are served in production.

---

## 1. Executive Summary
PrefillOnly is a specialized inference engine for “prefill-only” large language model (LLM) workloads—requests that need the model to produce just one output token. It introduces hybrid prefilling (to drastically cut active GPU memory) and a continuously recalibrated, job-completion-time-aware scheduler (to exploit prefix cache hits), delivering up to 4× higher queries-per-second without increasing mean or P99 latency and extending maximum input length by up to 5× compared to non-parallel baselines (Abstract; §7.2; Table 2; Figures 6–7, 10).

## 2. Context and Motivation
- The specific problem
  - Many emerging applications (recommendation, credit verification, data labeling) only need a single token as the decision (e.g., “Yes/No”), not long generations. The paper calls this the prefill-only workload: requests run just the model’s prefilling stage and stop after the first token (§1, §2.3).
  - Existing LLM engines assume arbitrary output lengths, so they retain key–value (`KV`) caches for all layers to speed multi-token decoding. In prefill-only settings, those cached states are mostly unused yet occupy large GPU memory (§1, §2.5).
  - Scheduling is also suboptimal: with uncertain output lengths, engines rely on JCT-agnostic policies (e.g., FCFS). In prefill-only workloads, the output length is fixed (one token), so job completion time (JCT) is predictable, enabling better scheduling—if the engine can account for dynamic prefix-cache hits (§1, §2.5–§2.6).

- Why it matters
  - Real-world systems at scale: recommendation/credit workloads can require tens of thousands of queries per second and consume hundreds to thousands of high-end GPUs (§2.4). Long inputs (user histories, credit logs) push memory limits (§2.4).
  - Prefill-only requests are compute-bound rather than memory-bandwidth-bound (decoding is the opposite), so throughput techniques designed for decoding (e.g., large continuous batches) do not translate (§2.4, §6.1).

- Prior approaches and their gaps (§2.5)
  - Chunked prefilling: processes inputs in chunks to fit memory. It hurts attention kernel performance and reduces throughput; in measurements, end-to-end throughput drops by 14% when chunking a 20k input with 512 chunk size (§2.5).
  - Tensor parallelism: increases maximum input length (MIL) but introduces heavy inter-GPU all-reduce, reducing throughput and sometimes inflating latency if interconnects are slow (§2.5).
  - Pipeline parallelism: avoids all-reduce but suffers from pipeline bubbles with variable-length requests unless inputs are chunked (which again hurts throughput) (§2.5).
  - KV caching engines (e.g., vLLM’s PagedAttention) assume long decoding and keep KV states for all layers, capping MIL without parallelization (§2.5).

- Positioning
  - PrefillOnly targets prefill-only workloads explicitly. It removes the need to keep per-layer KV states during inference, maximizes single-GPU MIL, and schedules requests to maximize prefix-cache reuse via continuous JCT calibration (§3, §4, §5, §6).

Key terms used throughout:
- `KV cache`: key/value tensors produced by attention layers; stored to avoid recomputing past tokens during decoding (§2.1).
- `Prefix caching`: reusing the KV cache for a shared prefix between different requests (§2.1).
- `Prefill-only request`: a request that runs only the prefilling pass and emits exactly one output token (§1, §2.3).
- `JCT` (job completion time): total time a request will take to finish processing.
- `MIL` (maximum input length): the longest input sequence the engine can process on a given GPU under its memory constraints (Table 2).

## 3. Technical Approach
PrefillOnly comprises three core techniques (Figure 2; §3.2), plus a lightweight serving and scheduling architecture (§3.1).

A. System architecture and memory planning (§3.1)
- On startup (“profile run”), the system forwards a synthetic maximum-length request and measures peak GPU memory. The residual GPU memory is reserved for prefix KV caches (§3.1).
- Runtime components:
  - An OpenAI-compatible HTTP server ingests requests.
  - A scheduler process keeps a waiting queue, computes calibrated JCTs every step, and selects the next request (§3.1, §6.3).
  - Executor processes run the LLM forward pass with hybrid prefilling and return the one-token probability distribution (§3.1).

B. Hybrid prefilling: chunk non-attention layers, run attention layers normally (§4)
- Why it is needed
  - Simply dropping full-layer KV caches only modestly improves MIL (~1.6× on L4 with Llama‑3.1‑8B) because temporary “intermediate tensors” inside non-attention (linear/MLP) layers dominate peak memory (§2.6; §4.1).
  - Profiling shows periodic 2–3 GB memory spikes during prefilling of a 32,768-token request on Llama‑3.1‑8B that correspond to MLP intermediate tensors (§4.1; Figure 3a). These intermediates are large: for Llama‑3.1‑8B, the MLP’s intermediate activations are 32,768×28,672 and 32,768×14,336 in bf16—14× and 7× larger than a single-layer KV cache respectively (Figure 4; §4.1).

- How it works
  - PrefillOnly processes all non-attention layers chunk-by-chunk (e.g., over the token dimension), so at any moment it only materializes intermediate tensors for one chunk (§4.2).
  - Attention layers run “normally,” i.e., over the full sequence, preserving their efficiency (§4.2).

- Why this is safe and effective
  - Non-attention blocks are linear; their computation across chunks is independent and can be concatenated without changing results (§4.2).
  - Memory benefit: chunking the large MLP intermediates flattens the spikes, reducing peak memory by ~2 GB in the 32,768-token example (Figure 3b).

- Implementation details via `torch.compile` (§4.3)
  - Group consecutive linear ops into a single “virtual layer” and loop over chunks; concatenate outputs (§4.3).
  - Two memory optimizations:
    - Output preallocation: pre-allocate the final output tensor and write each chunk’s output directly into it to avoid a second full-size buffer (§4.3).
    - In-place reuse: when input/output shapes match, reuse the input buffer to store the output (§4.3).

C. Suffix KV cache discarding (or offloading) with prefix caching preserved (§5.1)
- Goal: keep KV states only for prefix tokens likely to be reused by future requests; discard or offload suffix tokens to stay within GPU memory while still allowing prefix cache hits (§5.1).
- Enabler: hybrid prefilling completes in a single forward pass, so no subsequent passes need the discarded KV states (§5.1).
- Implementation: leverages the sliding-window abstraction in vLLM; does not require custom kernels (§5.1).

D. Continuous JCT calibration and scheduling (§6)
- Why not batch prefill-only requests? Prefill-only inference is compute-bound; batching does not boost throughput much but increases latency, unlike decode-time batching (§6.1).
- Core idea: before scheduling each next request, recompute all waiting requests’ JCTs under the “current” prefix-cache state, then choose the shortest remaining job (§6.3, Algorithm 1).
  - JCT model: `get_jct(n_input, n_cached)` is profiled over a grid, or use a strong proxy: number of cache-miss tokens `n_input − n_cached`, which correlates with observed JCT at 0.987 Pearson on A100 with Qwen‑32B FP8 (§6.3).
  - Fairness: subtract `λ * T_queue` from the score to avoid starvation (λ trades average vs P99 latency; §6.3, Algorithm 1; Figure 11).

- Why continuous calibration matters (Figure 5; §6.2–§6.3)
  - Prefix caches are dynamic. A request that shares a prefix with one just executed can become much shorter (lower JCT) immediately after that execution. Calibrating JCTs “now” lets the scheduler prioritize those imminent cache hits, increasing cache-hit rate and lowering latency (§6.3, Figure 5).

E. When to parallelize vs not (§5.2)
- Without cache reuse, discarding KV caches with hybrid prefilling gives highest throughput on a single GPU because there is no inter-GPU overhead (§5.2).
- With high-speed interconnect (NVLink) and low QPS, tensor parallelism can minimize single-request latency, but it scales poorly at higher QPS due to communication (§5.2; Figures 6, 8).

## 4. Key Insights and Innovations
- Hybrid prefilling for memory-dominant intermediates (§4)
  - Novelty: prior chunking strategies typically chunk the whole forward (including attention), which slows attention kernels. PrefillOnly chunks only non-attention blocks and leaves attention unchunked, achieving large memory savings without harming attention performance (§4.2).
  - Significance: removes the main memory bottleneck (MLP intermediates; Figures 3–4), enabling much longer inputs on a single GPU (Table 2; Figure 10) and avoiding cross-GPU parallelization.

- Suffix KV cache discarding with preserved prefix reuse (§5.1)
  - Novelty: instead of fully dropping all KV states (which breaks prefix caching), PrefillOnly keeps prefix states and discards only suffix states, matching prefill-only’s need to reuse common prefixes across many requests.
  - Significance: higher MIL without parallel KV sharding and with maintained cache-hit benefits (§5.1, §5.2).

- Continuous JCT calibration for cache-aware SRJF scheduling (§6.3)
  - Novelty: JCT-aware scheduling is refreshed before every scheduling decision to reflect the current cache contents; uses a simple, accurate proxy for JCT (`n_input − n_cached`).
  - Significance: improves cache-hit rate and reduces latency, especially under high QPS where cache contention is severe (Figure 5; Figure 9).

- Practical, generalizable implementation
  - Uses `torch.compile` transformations and vLLM’s existing kernels; no hardware-specific kernels are modified (§3.2, §4.3, §5.1). This eases adoption across models and GPUs.

## 5. Experimental Analysis
- Evaluation methodology (§7.1)
  - Workloads (simulated to stress engines, summarized in Table 1):
    - Post recommendation: 20 users; each has a 11k–17k token profile (Normal distribution mean 14k, std 3k) and 50 candidate posts (150 tokens each); 14M total tokens.
    - Credit verification: 60 users; each has 40k–60k token credit history; 3M total tokens.
  - Hardware and models (Table 3):
    - 2× L4 (24 GB) with Llama‑3.1‑8B; 2× A100 (40 GB) with DeepSeek‑R1‑Distill‑Qwen‑32B FP8; 2× H100 (80 GB) with/without NVLink using Llama‑3.3‑70B FP8.
  - Baselines (all with prefix caching enabled): vLLM PagedAttention, chunked prefill (Sarathi‑Serve style), pipeline parallel, tensor parallel (§7.1).
  - Metrics: Query-per-second (QPS) vs mean latency and P99 latency (Figures 6–7); MIL (Table 2); throughput curves and bars (Figures 8–9).
  - Serving setup: non-parallel methods run one instance per GPU with user-id routing (§7.1). PrefillOnly fairness parameter default λ=500 (§7.1).

- Main results
  - Throughput vs latency
    - Across hardware and both workloads, PrefillOnly achieves the lowest latency at high QPS, indicating higher sustainable throughput than baselines (Figure 6).
    - Quote: “PrefillOnly handles 1.4–4.0× larger query-per-second without inflating the average latency and P99 latency compared to baselines.” (§7, Figure 6 and Figure 7). This echoes the abstract claim: “upto 4× larger queries per second without inflating average and P99 latency.”
    - At low QPS, tensor parallel can have lower single-request latency (especially with NVLink) because it uses multiple GPUs per request (§5.2; Figure 6d/h).

  - P99 latency
    - PrefillOnly’s SRJF with fairness retains competitive or better P99 latencies across scenarios (Figure 7), showing the fairness offset (λ) does not degrade tail latencies when tuned (see Figure 11 for λ trade-offs).

  - Maximum input length (MIL) without parallelization (Table 2)
    - On L4: PrefillOnly 130k tokens vs PagedAttention 24k and chunked prefill 46k; pipeline parallel 72k; tensor parallel 195k.
    - On A100: PrefillOnly 87k vs PagedAttention 11k and chunked 17k; pipeline 38k; tensor 77k.
    - On H100: PrefillOnly 97k vs PagedAttention 15k and chunked 25k; pipeline 183k; tensor 238k.
    - Interpretation: PrefillOnly extends MIL up to ~5× over non-parallel methods while staying single-GPU (Table 2). Parallel methods can push MIL further, but at the cost of inter-GPU overheads.

  - Why PrefillOnly wins (source analysis; §7.2)
    - Post recommendation: Under high QPS, chunked prefill suffers “prefix cache throttling”—insufficient cache space causes misses and performance collapse—whereas PrefillOnly’s continuously calibrated SRJF prioritizes imminent cache hits and maintains throughput (Figure 9). Pipeline/tensor parallel avoid cache throttling by sharding across GPUs but pay communication and synchronization overheads.
    - Credit verification: PrefillOnly avoids all-reduce and pipeline bubbles; even with NVLink accelerating tensor-parallel communication, PrefillOnly still has the highest throughput (Figure 8a–b).

  - Ablations and diagnostics
    - Hybrid prefilling effectiveness: On Qwen‑2.5‑32B FP8 (A100), hybrid prefilling increases MIL by 7.9× over vanilla vLLM; compared to chunked prefill, PrefillOnly reaches 87k tokens while “not hurting throughput” (Figure 10; §7.2).
    - Scheduling fairness: increasing λ improves P99 at the cost of mean latency (CDFs in Figure 11), allowing operators to tune SLA trade-offs.
    - JCT proxy quality: correlation 0.987 between JCT and cache-miss tokens on A100 + Qwen‑32B FP8 (§6.3).

- Do the experiments support the claims?
  - The multi-hardware, multi-model evaluation with two workload types shows consistent gains in the high-QPS regime and substantial MIL gains without parallelization (Figures 6–9; Table 2; Figure 10). The cache-aware scheduling example (Figure 5) and λ ablation (Figure 11) clarify mechanisms and trade-offs. A limitation is that datasets are simulated (Table 1), so real-world variability (e.g., prefix distribution, burstiness) remains to be validated.

## 6. Limitations and Trade-offs
- Scope: specialized for prefill-only
  - PrefillOnly is not designed for multi-token generation. Mixing with generative workloads on the same GPUs is discouraged due to interference (notably decoding throughput) (§2.4).
  - It assumes outputs can be restricted to a small set of tokens (e.g., “Yes/No”), which the serving layer enforces by constraining the candidate token set (§2.3).

- Cache policy trade-offs
  - Current implementation discards suffix KV caches, which cannot be reused later; offloading to CPU (e.g., LMCache) is proposed but not implemented (§9).
  - Benefits hinge on real prefix-sharing across requests; workloads without meaningful shared prefixes will see less scheduling gain (§6.2–§6.3).

- Throughput vs latency and hardware conditions
  - At low QPS and with fast interconnects (NVLink), tensor parallel can yield lower per-request latency (§5.2; Figure 6d/h).
  - Hybrid prefilling depends on accurate graph transformations and shape inference via `torch.compile`. Models with unusual non-linear blocks or dynamic control flow may complicate chunking (§4.3).

- Evaluation realism
  - Datasets are simulated; while parameters (token lengths, counts) are grounded in measured ranges, production traffic patterns, failure modes, and cache dynamics may differ (Table 1; §7.1).

## 7. Implications and Future Directions
- How it changes the landscape
  - Establishes prefill-only serving as a distinct operating point with different bottlenecks and opportunities than decoding-oriented serving. The key shift: treat MLP intermediates—not KV caches—as the main memory bottleneck and exploit deterministic single-token outputs for JCT-aware scheduling (Figures 3–4; §6).

- Follow-up research enabled or suggested
  - KV cache offloading and compression tailored to prefill-only: integrate CPU/NVMe KV tiers (e.g., LMCache) with hybrid prefilling (§9).
  - Better cache-aware scheduling: multi-queue SRJF with prefix-aware clustering; proactive “cache seeding” of hot prefixes; online learning of JCT beyond linear proxy (§6.3).
  - Disaggregated prefill/decode systems: deploy PrefillOnly on prefill nodes in split architectures to raise overall goodput (§9; see also DistServe in Related Work).
  - Extending hybrid prefilling to other architectures: e.g., MoE models, vision-language models where linear blocks dominate activation memory.

- Practical applications
  - High-QPS decision services: feed ranking/recommendation, fraud/credit checks, content moderation, large-scale labeling—any task where a calibrated single-token score suffices (§2.2–§2.4).
  - Long-context analytics: handle 80k–130k-token inputs on a single GPU (Table 2) without resorting to communication-heavy parallelism.

> Representative result: “PrefillOnly can process upto 4× larger queries per second without inflating average and P99 latency” (Abstract; supported by Figures 6–7), while expanding single-GPU MIL up to 5× relative to non-parallel baselines (Table 2) and showing a 7.9× MIL gain in an ablation (Figure 10).

Overall, PrefillOnly reframes LLM serving for discriminative, long-context, single-token workloads and pairs a memory-savvy execution plan (hybrid prefilling + suffix cache management) with an adaptive, cache-aware scheduler (continuous JCT calibration). This combination makes single-GPU inference viable for inputs previously thought to require model parallelism, and it sustains higher throughput under real cache dynamics.
