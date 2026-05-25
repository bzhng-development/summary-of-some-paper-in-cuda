# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

**ArXiv:** [2403.02310](https://arxiv.org/abs/2403.02310)

## 🎯 Pitch

Sarathi-Serve introduces a novel scheduling system for LLM inference that splits heavyweight prefill computations into manageable chunks and co-schedules them with ongoing decode steps in a stall-free manner. This innovation eliminates longstanding trade-offs between high throughput and low latency, enabling responsive, cost-efficient LLM serving—even under heavy loads—while unlocking the full potential of pipeline parallelism. As a result, applications like chatbots and code assistants can achieve both fast token streaming and dramatically improved serving capacity, solving a fundamental bottleneck in LLM deployment.

---

## 1. Executive Summary
This paper introduces Sarathi-Serve, a scheduling framework for large language model (LLM) inference that achieves high throughput without sacrificing interactive latency. It does so by splitting long “prefill” computations into bounded-size chunks and co-scheduling those chunks with ongoing “decode” iterations in a stall-free way, which also makes pipeline parallelism efficient by reducing pipeline bubbles.

## 2. Context and Motivation
- The problem addressed
  - An LLM request has two phases: `prefill` (process the full prompt to produce the first output token) and `decode` (generate subsequent tokens one-by-one). Prefill iterations are compute-heavy and parallel; decode iterations are memory-bound and underutilize GPU compute when batch sizes are small (§2.2).
  - Batching multiple requests boosts throughput, especially for decode, but current schedulers interleave prefills and decodes in ways that force a trade-off between throughput and latency (§1, §2.5, Figure 2).

- Why it matters
  - Real systems (chatbots, copilots) need both high throughput (to keep costs low) and low interactive latency. Two user-visible latency metrics matter (§2.4): `TTFT` (time to first token, reflecting responsiveness) and `TBT` (time between tokens, reflecting streaming smoothness). Under load, aggressive batching can balloon tail `TBT`, degrading perceived quality (Figure 1b).

- Where prior approaches fall short
  - Decode-prioritizing, request-level batching (e.g., FasterTransformer; Algorithm 1) keeps TBT low because decodes are never preempted, but throughput is poor: once some requests finish, the remainder run in shrinking, inefficient batches (§2.5).
  - Prefill-prioritizing, iteration-level batching (e.g., Orca, vLLM; Algorithm 2) increases throughput by eagerly inserting prefills, but this causes “generation stalls,” long pauses between output tokens when a full prefill sneaks between two decodes (Figure 1a, Figure 7). Tail TBT can spike for seconds.
  - With pipeline parallelism (PP)—needed for multi-node or very large models—iteration runtimes vary wildly across micro-batches because prefills are long and decodes are short. This produces “pipeline bubbles” (idle time) that waste GPU cycles (Figure 8; §3.3).

- Positioning
  - Sarathi-Serve sits between these extremes: it retains iteration-level batching for throughput, but it splits large prefills into bounded-size chunks and schedules those chunks only within a fixed per-iteration token budget, ensuring ongoing decodes are not stalled (§4). It also composes more uniform-compute micro-batches, mitigating PP bubbles (§3.3, §4.2).

## 3. Technical Approach
Sarathi-Serve’s design is guided by a careful cost analysis of prefill vs. decode and a key observation about arithmetic intensity.

- What makes co-scheduling feasible
  - Empirical profiling shows decode iterations are memory-bound at practical batch sizes: their execution time is dominated by fetching weights and KV-cache rather than math operations. Prefill iterations, by contrast, are compute-bound once sequence length is moderately large (∼512 tokens on A100 for Mistral-7B) (Figure 3, Figure 4; §3.1).
  - Arithmetic intensity analysis (FLOPs per byte fetched) on LLaMA2-70B confirms: small-token decode batches operate in the memory-bound regime; adding more tokens barely increases runtime until a threshold (Figure 5, Figure 6; §3.1). This “slack” allows opportunistically adding work to decode iterations with little latency impact.

- Chunked-prefills (§4.1)
  - Idea: split each prefill into near-equal compute-sized chunks rather than running it in one long iteration. Each chunk processes a bounded number of prompt tokens (e.g., 512–2048, tuned per hardware/SLO), which is enough to saturate GPU compute but short enough to not delay decodes significantly.
  - Mechanism: chunk i attends over the KV-cache from all prior chunks for the same request. This incurs extra memory reads for earlier chunks’ KV (N−1, N−2, …), but because prefill attention remains compute-bound at practical chunk sizes, overhead stays modest (§4.3; quantified later in Figure 14).

- Stall-free batching (§4.2, Algorithm 3)
  - Goal: admit new work without stalling ongoing decodes. Sarathi-Serve forms each iteration’s batch as follows:
    1) Pack all current decodes first (lines 6–8).
    2) Include at most one “continuation” prefill chunk for any in-progress prefill (lines 9–12).
    3) Admit new requests, but only add their prefill chunks up to a token budget τ (lines 13–20).
  - `Token budget τ` bounds the total number of tokens processed per iteration across decodes and prefills, thus bounding iteration latency and eliminating long prefill-induced stalls (Figure 7, “Stall-free Schedule”).
  - Empirical impact: when comparing hybrid batching “decode + full prefill” vs “decode + chunked prefill,” chunking limits the incremental iteration latency to small multipliers. With full prefills, the iteration time can blow up by up to 28.3×; with chunked prefills the increase is much smaller and predictable (Figure 9a,b).

- Determining the token budget τ (§4.3)
  - Trade-off: Smaller τ reduces TBT interference but introduces more chunks per prefill, which adds overhead from repeated KV-cache reads and fixed kernel costs. Larger τ is more efficient but risks violating TBT SLOs and worsening PP bubbles.
  - Practical considerations:
    - TBT SLO: choose the largest τ whose per-iteration latency fits the SLO envelope.
    - Tile quantization: GPU matmul kernels run most efficiently when dimensions align with tile sizes; using a chunk size of 257 vs 256 increased prefill time by 32% in one profile (§4.3).
    - PP bubbles: larger variations in per-iteration runtime produce more bubbles; τ must also be chosen to keep micro-batches uniform.
  - How they set τ: one-time profiling with Vidur (§4.3; [28]) to maximize capacity under given deployments. In experiments they typically use τ=512 for strict SLOs and τ=2048 for relaxed SLOs, with a special τ=1536 for LLaMA2-70B PP to tame bubbles (§5.1).

- Why this approach over alternatives
  - Decode-prioritizing avoids stalls but wastes throughput by letting batch sizes shrink (Algorithm 1; §3.2).
  - Prefill-prioritizing iteration-level batching (Algorithm 2) maximizes throughput but causes generation stalls that hurt tail TBT (Figure 1a, Figure 7; §3.2).
  - Disaggregating prefill and decode onto different replicas (e.g., SplitWise/DistServe; §6) can remove interference entirely, but requires moving KV-cache across replicas (bandwidth heavy) and can underutilize prefill replicas’ memory. Sarathi-Serve keeps a single-replica path and aims for robust tail latency without extra network complexity.

- Implementation (§4.4)
  - Built atop vLLM, augmented with: paged chunk-prefill kernels using FlashAttention v2 [38] and FlashInfer [74]; an extensible scheduler; PP and TP support via NCCL; telemetry.
  - Models/hardware evaluated include Mistral-7B (1×A100), Yi-34B (2×A100 TP2), LLaMA2-70B (8×A40 TP4–PP2), and Falcon-180B (8×A100 across two nodes TP4–PP2) (Table 1).

## 4. Key Insights and Innovations
- Stall-free batching via a bounded token budget
  - Novelty: Instead of “always admit prefills” (vLLM/Orca) or “never admit prefills until all decodes complete” (FasterTransformer), Sarathi-Serve caps the total tokens per iteration, first filling with decodes and then only small prefill chunks (Algorithm 3).
  - Significance: This removes generation stalls (long insertions of full-prefill iterations) while preserving the high decode batch sizes needed for good throughput (Figure 7 shows the new schedule has “No stalls”).

- Chunked-prefills to exploit arithmetic intensity slack
  - Novelty: Split prefills into near-equal compute-sized chunks that are piggybacked with decodes, ensuring both high compute utilization and bounded per-iteration latency (Figure 9).
  - Significance: Turns a fundamental bottleneck (decode memory-bound underutilization) into an opportunity to do extra work without harming interactive latency (Figure 5–6; §3.1).

- Uniform-compute micro-batches for pipeline parallelism
  - Novelty: By keeping each iteration’s compute within similar bounds, Sarathi-Serve minimizes PP bubbles caused by interleaving long prefills with short decodes (Figure 8).
  - Significance: Makes PP feasible on commodity networks by reducing idle periods and avoiding cross-node all-reduce overheads of high-degree TP (Figure 13a,b; §5.3).

- A simple, practical control knob (token budget) tied to SLOs
  - Novelty: Exposes a direct, measurable parameter (τ) to trade off tiny chunking overheads for strict tail-latency guarantees; chosen via one-time profiling (§4.3).
  - Significance: Cleanly navigates throughput/latency trade-offs, something prior systems handle only indirectly via batch size or admission heuristics (Figure 12).

## 5. Experimental Analysis
- Setup (§5)
  - Datasets (Table 2): `openchat_sharegpt4` (multi-turn chats, median 1730 prompt tokens) and `arxiv_summarization` (long scientific documents, median 7059 prompt tokens).
  - Metrics (§2.4): `TTFT` (median) and `TBT` (P99). Throughput is reported as `Capacity`—max queries per second that satisfy a P99 TBT SLO without queue blow-up.
  - SLOs (Table 3): “strict” and “relaxed” SLOs are set per model as multiples (5× and 25×) of a baseline decode-only iteration latency (e.g., 0.5s vs 0.1s P99 TBT targets for Mistral-7B).
  - Baselines: vLLM (iteration-level, prefill-prioritizing with PagedAttention), Orca (iteration-level with hybrid batches but still prefill-prioritizing), FasterTransformer (request-level, decode-prioritizing; discussed primarily conceptually).

- Main results on single-node/TP deployments (§5.1)
  - Capacity improvements:
    - Mistral-7B (1×A100): “strict” SLO, Sarathi-Serve achieves up to 2.78× capacity over Orca and 2.15× over vLLM on openchat_sharegpt4 (Figure 10a). On arxiv_summarization, gains are 1.82× and 1.97× respectively (Figure 10b).
    - Yi-34B (2×A100 TP2): up to 4.00× over Orca and 2.44× over vLLM (openchat_sharegpt4; Figure 10a). The text also reports up to 3.7× vs vLLM under strict SLOs (§1; §5.1).
  - Throughput–latency curves (Figure 12): with strict P99 TBT targets (e.g., 100ms for Mistral-7B), Sarathi-Serve with τ=512 sustains ∼3.5× higher capacity than vLLM; with relaxed targets (e.g., 1s for Yi-34B), τ=2048 yields ∼1.65× capacity over vLLM.

- PP and cross-node viability (§5.3)
  - Decode-only latency: For Falcon-180B, TP-8 across nodes roughly doubles median TBT compared to TP4–PP2 (Figure 13a).
  - Capacity: Under strict SLOs, Sarathi-Serve with TP4–PP2 improves capacity by 3.6× vs vLLM TP4–PP2 and 4.3× vs vLLM TP-8 (Figure 13b). With relaxed SLOs, capacity improves 1.48× over vLLM TP4–PP2 (Figure 13b).
  - Interpretation: The uniform-compute batches reduce PP bubbles that otherwise cripple vLLM under strict SLOs (Figure 8 and §5.3 discussion).

- Evidence that chunking preserves latency and limits overhead
  - Latency impact per iteration: Co-scheduling full prefills inflates iteration time (up to 28.3×) whereas chunked-prefills keep increases modest and bounded (Figure 9).
  - Chunking overheads: For Yi-34B, chunk sizes of 512/1024/2048 add at most ∼25% overhead on the prefill phase; for larger chunks the overhead is negligible (Figure 14; §5.4.1).
  - Ablation of the two mechanisms (Table 4): Using only hybrid-batching reduces TTFT but hurts TBT; using only chunked-prefills helps TBT but hurts TTFT slightly. Combining both achieves low TBT and reasonable TTFT simultaneously:
    > Table 4: with τ=1024 on Yi-34B (TP2), Sarathi-Serve yields P50 TTFT 0.76–3.90 s and P99 TBT 0.14–0.17 s across the two datasets; the “combined” configuration dominates single-mechanism variants on tail latency.

- Overall headline gains
  - Across models and hardware under tail-latency constraints, Sarathi-Serve increases serving capacity:
    > Up to 2.6× on Mistral-7B single A100; up to 3.7× on Yi-34B TP2; up to 6.3× on LLaMA2-70B TP4–PP2; and up to 5.6× on Falcon-180B TP4–PP2 (Figures 10–11; §1, §5.1).

- Do the experiments support the claims?
  - Yes. The paper ties mechanism to outcome with:
    - Micro-level cost analyses (Figure 3–6) that justify piggybacking prefills.
    - Scheduler timelines (Figure 7–8) showing how stalls/bubbles arise and are mitigated.
    - System-level capacity results under controlled SLOs across models/sizes (Figures 10–13, Table 3).
    - Ablations (Figure 14, Table 4) attributing gains to both chunking and stall-free admission.

## 6. Limitations and Trade-offs
- Token budget tuning is required
  - Choosing τ is deployment-specific: it depends on model, hardware, PP/TP configuration, and target SLO (§4.3). The paper uses profiling (Vidur) and a small set of fixed values (512 for strict, 2048 for relaxed, with one exception at 1536), but does not present an online auto-tuner. Mis-setting τ could either reintroduce stalls (too large) or add unnecessary overhead (too small).

- Chunking introduces overhead
  - Splitting prefills adds some inefficiency, primarily extra KV-cache reads and fixed kernel overhead. For small chunks (e.g., 512 tokens), overhead on prefill time can reach ∼25% (Figure 14). This is a deliberate trade to bound TBT; workloads that only care about total completion time (not interactive latency) might prefer full-prefill efficiency.

- Assumes enough decode work to piggyback on
  - The stall-free scheme relies on decoding being active to amortize chunked-prefill cost and keep arithmetic intensity balanced. In workloads dominated by long prefills and very short decodes (or few concurrent requests), the benefit diminishes, and TTFT might be affected by chunk splitting.

- Not a holistic solution to all scheduling goals
  - The paper focuses on throughput vs tail-TBT under SLOs. It does not address multi-tenant fairness or preemption (cf. FastServe, APIServe, fairness work in §6). It also does not include an end-to-end comparison against disaggregated prefill/decode systems (SplitWise/DistServe/TetriInfer), which could offer stronger TTFT but at higher system complexity and KV migration costs (§6).

- Hardware-specific nuances
  - Tile-quantization effects and PP bubble behavior are hardware/stack sensitive (§4.3). Gains on other GPU generations or interconnects may require re-profiling and may differ from reported numbers.

## 7. Implications and Future Directions
- What changes in practice
  - Sarathi-Serve offers a practical recipe to deliver both high throughput and stable interactive latency for LLM serving—especially important for chat and streaming applications. It also makes PP viable across commodity networks by reducing bubbles, enabling larger models to be served efficiently without high-degree TP or exotic interconnects (Figure 13).

- What it enables next
  - Dynamic token-budget control: online adaptation of τ based on observed latencies, load, and PP imbalance could further improve capacity and robustness beyond the fixed τ values used here (§5.1, §4.3).
  - Integration with fairness/preemption: combining stall-free batching with preemptive schedulers or fairness-aware admission could provide stronger QoS in multi-tenant clouds (§6).
  - Hybrid with disaggregated designs: selectively using disaggregated prefills for ultra-long prompts or TTFT-critical users, while keeping stall-free batching for the bulk of traffic, could yield the best of both worlds (noted in §6).
  - Compiler/runtime co-design: accounting for tile quantization and kernel launch overheads when choosing chunk sizes suggests opportunities for joint tuning in kernel libraries and schedulers (§4.3).

- Broader takeaway
  - The paper reframes the “throughput vs. latency” conflict in LLM serving as a batching and compute-shaping problem. By matching iteration compute to an SLO-driven envelope (token budget) and exploiting arithmetic-intensity slack in decode, it achieves a materially better Pareto frontier. The empirical results across four models and diverse deployments (Figures 10–13) suggest this approach is widely applicable.
