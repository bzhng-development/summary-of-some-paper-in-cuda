# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

**ArXiv:** [2403.02310](https://arxiv.org/abs/2403.02310)

## 🎯 Pitch

Sarathi-Serve introduces a novel LLM serving scheduler that breaks input prompt processing into 'chunked-prefills' and employs 'stall-free batching,' ensuring ongoing token generation never stalls—even as new requests arrive. This enables unprecedented GPU efficiency by eliminating the classical tradeoff between throughput and tail latency, delivering up to 5.6x gains in serving capacity without latency spikes. For real-world systems serving interactive LLMs, Sarathi-Serve means lower costs and consistently smooth user experiences, redefining how production-scale AI can be deployed.

---

## 1. Executive Summary
Sarathi-Serve is a scheduling and batching system for serving large language models (LLMs) that combines two ideas—chunked-prefills and stall-free batching—to deliver high throughput without sacrificing tail latency. It solves a core operational problem: when many users share a GPU, prefills (processing long prompts) and decodes (generating tokens) interfere, forcing a throughput–latency tradeoff; Sarathi-Serve schedules them so ongoing decodes never stall while still keeping batches large enough for efficiency.

## 2. Context and Motivation
- Problem addressed
  - Every LLM request has two phases:
    - `prefill`: process the entire input prompt to produce the first output token.
    - `decode`: generate the remaining output tokens one at a time.
  - Prefills are compute-bound (good GPU utilization), while decodes are memory-bound (poor utilization) because each decode uses only one token per request and must read a large key–value cache (`KV-cache`) of prior tokens (§2.2).
  - To obtain high throughput, serving systems batch multiple requests. But batching interleaves prefills and decodes across users, which can:
    - Increase throughput (by raising decode batch size).
    - Hurt tail latency when long prefills are scheduled between decode iterations, causing multi-second “generation stalls” (Figure 1a, §3.2).
- Why it matters
  - Throughput determines service cost (how many requests per GPU per second), while tail latency determines user experience (smoothness of token streaming). Many production systems have strict time-between-tokens (`TBT`) requirements (e.g., chat), so a method that reliably meets tail-latency SLOs while keeping GPUs busy has direct operational impact (§1, §2.4).
- Prior approaches and shortcomings
  - Decode-prioritizing, request-level batching (e.g., FasterTransformer): form a batch and run it to completion; no new prefills admitted until all decodes finish (Algorithm 1). This avoids decode interference and keeps TBT low but wastes compute when some requests finish early and batch size shrinks, reducing throughput (§2.5, Figure 7).
  - Prefill-prioritizing, iteration-level batching (e.g., Orca, vLLM): allow requests to enter/exit batches per iteration (Algorithm 2). New prefills are eagerly admitted whenever memory allows; this raises decode batch sizes (good for throughput) but can insert long prefill iterations between decode steps, causing “generation stalls” and high TBT (Figure 1a, §3.2, Figure 7).
  - Pipeline-parallel serving (PP) across nodes: even with iteration-level micro-batches, non-uniform iteration times (mix of long prefills and short decodes) create “pipeline bubbles” where stages idle, wasting GPU cycles (Figure 8, §3.3).
- Positioning relative to existing work
  - Sarathi-Serve sits between decode-prioritizing and prefill-prioritizing designs. It preserves the low-latency benefits of decode-prioritizing (no generation stalls) while achieving the high throughput of iteration-level batching by carefully throttling how many prefill tokens can co-run with decodes (§4). It also reduces PP bubbles by making micro-batches more uniform in compute (§4.2, §5.3).

## 3. Technical Approach
Sarathi-Serve has two core mechanisms that work together.

1) Chunked-prefills (compute-sized slicing of prompts)
- What: Split a prefill for a long prompt into multiple smaller chunks, each sized to be “large enough to be efficient but small enough not to stall decodes” (§4.1).
- Why this works technically
  - Decode iterations are memory-bound and underutilize compute. The ratio of FLOPs per byte (arithmetic intensity) for decode is low; adding some extra computation can increase utilization without proportionally increasing latency (Figure 5; also see the “flat” runtime region for linear layers at small token counts in Figure 6).
  - Prefills are compute-bound and highly efficient once each chunk contains a few hundred tokens. Figure 4 shows that linear layers dominate both prefill and decode time; even at moderately long sequences, attention is not the main cost. Figure 3 shows decode throughput scales strongly with batch size, while prefill saturates quickly.
- How: Execute a prompt across multiple iterations, adding one prefill chunk per iteration alongside ongoing decodes. Chunk sizes are chosen to saturate compute but keep iteration time bounded (§4.1, Figure 9).

2) Stall-free batching (build batches without stalling decodes)
- Goal: Admit new requests and compute their prefills without ever pausing the cadence of decode iterations for requests already generating tokens (§4.2).
- Core idea: For each iteration, form a hybrid batch consisting of:
  - All ongoing decodes (to maintain streaming cadence).
  - At most one chunk from each prefill candidate, subject to a pre-determined total token budget for the iteration, so the iteration’s latency remains bounded and predictable (§4.2, Algorithm 3).
- Algorithm flow (Algorithm 3)
  - Compute a per-iteration `token_budget` from the desired TBT SLO (how many total tokens can the GPU process in one iteration without violating tail latency) (§4.3).
  - Step 1: Insert all ongoing decode tokens into the batch (lines 6–8).
  - Step 2: If any prefills are already in progress, add their next chunk sized to fit within remaining budget (lines 9–12).
  - Step 3: Admit new requests; for each, add a prefill chunk sized to fit the remaining budget (lines 13–20).
  - Execute the hybrid batch and repeat (lines 22–24).
- Why it avoids generation stalls
  - Unlike vLLM and Orca, Sarathi-Serve never schedules a full, long prefill ahead of decodes. Instead, it throttles prefill work to fit the leftover “budget” each iteration, so decode timing is not disrupted (Figure 7, “Stall-free Schedule”).

3) Determining the token budget (how big can a batch be while meeting TBT?)
- Factors (§4.3):
  - TBT SLO: Smaller budgets reduce iteration latency but may over-chunk and add overhead.
  - Chunking overheads: Attention for each new chunk must read KV-cache from previous chunks; this repeats some memory traffic. The paper measures these overheads (Figure 14) and finds they are modest for chunk sizes ≥1024 and still moderate at 512.
  - Tile-quantization effects: GPU matmuls prefer matrix sizes divisible by tiling constants; chunk sizes like 257 can be noticeably slower than 256, causing up to ~32% extra prefill time in some cases (§4.3).
  - Pipeline bubbles: Larger chunks increase per-iteration variance, which can create PP bubbles. Budgets should balance SLO adherence and PP efficiency (§4.3).
- Practical selection: One-time profiling identifies the largest per-iteration token count that meets the TBT SLO; the authors use the Vidur profiler/simulator to search for good budgets (§4.3). In experiments, they commonly use budget 512 for strict SLOs and 2048 for relaxed SLOs, with a smaller budget (1536) for LLaMA2-70B relaxed to reduce bubbles (§5.1).

4) Implementation notes
- Built on vLLM with iteration-level batching and PagedAttention; extended to:
  - Support chunked-prefills using FlashAttention v2 and FlashInfer kernels (§4.4).
  - Add stall-free scheduling, multi-GPU pipeline parallelism, and telemetry (§4.4).
  - Use NCCL for both tensor and pipeline parallel communications (§4.4).

Illustrative example (Figure 7)
- When new requests C and D arrive while A and B are decoding:
  - vLLM/Orca: Schedule full prefills for C/D first, stalling A/B’s decodes for potentially seconds (“generation stall”).
  - FasterTransformer: Wait for all decodes to finish before admitting any prefills, which keeps TBT smooth but kills throughput.
  - Sarathi-Serve: Split C’s prefill into chunks Cp1, Cp2, etc., and co-schedule Cp1 with A/B decodes immediately, never stalling the decodes.

## 4. Key Insights and Innovations
- Using memory-bound slack in decode iterations to “carry” prefill work safely
  - Difference from prior work: Prior systems either forbade co-running prefills during decodes (decode-prioritizing) or admitted full prefills eagerly (prefill-prioritizing). Sarathi-Serve shows that the arithmetic-intensity gap makes it possible to co-run limited prefill work with minimal impact on TBT (Figure 5, Figure 6, §3.1).
  - Significance: This reframes the perceived binary tradeoff—either throughput or latency—into a tunable continuum controlled by token budget (§4.3, Figure 12).
- Chunked-prefills as a control knob, not just a data structure
  - Novelty: Chunking is not just to fit memory; it is used as a scheduling lever to bound per-iteration latency predictably while keeping prefills compute-efficient (Figure 9; §4.1–4.2).
  - Significance: Enables stall-free hybrid batches whose runtime is nearly independent of raw prompt length, directly tackling generation stalls (Figure 1a, Figure 7).
- Stall-free batching policy that prioritizes decode cadence without sacrificing throughput
  - Difference: Unlike Orca/vLLM’s eager-prefill admission (Algorithm 2), the algorithm always packs decodes first and only then fills the remaining budget with prefill chunks (Algorithm 3, §4.2).
  - Significance: Maintains low P99 TBT even under heavy load while achieving high batch sizes for decodes (Figure 1b, Figure 10–11).
- Uniform-compute micro-batches that reduce pipeline bubbles
  - Difference: Prior PP inference relies on micro-batching but still sees bubbles due to mixed, highly variable prefills and decodes (Figure 8). With chunking and budgeting, Sarathi-Serve makes batches more uniform in runtime.
  - Significance: Makes PP viable across commodity Ethernet by reducing inter-stage idling; this is critical for very large models where cross-node tensor parallelism is too communication-heavy (Figure 13a–b, §5.3).

## 5. Experimental Analysis
- Evaluation setup
  - Models and hardware (Table 1):
    - `Mistral-7B` on 1× A100 80GB.
    - `Yi-34B` on 2× A100 (TP-2).
    - `LLaMA2-70B` on 8× A40 (TP-4, PP-2).
    - `Falcon-180B` across 2 nodes, each 4× A100 (TP-4 per node, PP-2 across nodes).
  - Workloads (Table 2):
    - `openchat_sharegpt4`: conversations; median prompt 1,730 tokens; high variance.
    - `arxiv_summarization`: long documents; median prompt 7,059 tokens.
  - Metrics (§2.4):
    - `TTFT` (Time to First Token): median reported.
    - `TBT` (Time Between Tokens): P99 reported.
    - `Capacity`: maximum queries-per-second (QPS) served under a TBT SLO (queuing delay constrained to ≤2s median).
  - SLO definitions (Table 3): “strict” ≈ 5× baseline decode iteration time; “relaxed” ≈ 25×; e.g., strict P99 TBT is 0.1s (Mistral-7B), 0.2s (Yi-34B), 1s (LLaMA2-70B/Falcon-180B).
  - Baselines: `vLLM` (iteration-level, prefill-prioritizing, PagedAttention) and `Orca` (iteration-level, prefill-prioritizing). For PP experiments, also compare to `vLLM` using TP-only (TP-8) and hybrid TP-PP (§5.1, §5.3).

- Main results
  - Capacity improvements across models and SLOs (Figures 10–11)
    - Mistral-7B (Figure 10a–b):
      - Strict SLO: up to 2.78× higher capacity than vLLM/Orca on openchat; 1.82× on arxiv.
      - Relaxed SLO: 2.15× (openchat) and 1.97× (arxiv).
    - Yi-34B (Figure 10a–b):
      - Strict SLO: up to 4.00× (openchat) and 1.69× (arxiv).
      - Relaxed SLO: 2.44× (openchat) and 1.94× (arxiv).
    - LLaMA2-70B with PP (Figure 11a–b):
      - Strict SLO: 5.54× (openchat) and 4.60× (arxiv).
      - Relaxed SLO: 6.31× (openchat) and 3.00× (arxiv).
    - Falcon-180B with PP (Figure 11a–b):
      - Strict SLO: 4.69× (openchat) and 4.20× (arxiv).
      - Relaxed SLO: 5.62× (openchat) and 2.75× (arxiv).
    - These numbers align with the abstract’s headline improvements:
      > “2.6× higher serving capacity (Mistral-7B, 1×A100), up to 3.7× (Yi-34B, 2×A100), up to 5.6× (Falcon-180B with PP)” (Abstract).
  - Throughput–latency tradeoff is tunable (Figure 12)
    - Varying the P99 TBT SLO shows vLLM’s capacity saturates early due to generation stalls, largely independent of its configured max batch size (32/64/128).
    - Sarathi-Serve tracks the SLO by adjusting token budget: with strict SLO it uses 512, with relaxed 2048 (except 1536 for LLaMA2-70B to reduce bubbles). Example: for Mistral-7B at 0.1s P99 TBT, Sarathi-Serve achieves ≈3.5× capacity vs. vLLM (Figure 12, top).
  - Making PP viable over Ethernet (Figure 13)
    - Decode-only latency: TP-8 across nodes incurs >2× median TBT compared to TP-4 within node + PP-2 across nodes (Figure 13a), because TP requires cross-node all-reduces on the critical path.
    - Capacity: Sarathi-Serve with PP increases Falcon-180B capacity by 3.6× (strict SLO) and 1.48× (relaxed SLO) over vLLM hybrid TP-PP, primarily by reducing pipeline bubbles via more uniform iterations (Figure 13b).
  - Generation stalls eliminated while keeping iterations short (Figure 9)
    - If you co-run full prefills (Orca’s hybrid batching), batch time can jump by up to 28.3× vs. decode-only. With chunked-prefills, Sarathi-Serve increases iteration time only modestly across decode batch sizes and context lengths (Figure 9a–b).
  - Ablations and overheads
    - Chunking overhead: At chunk size 512, prefill overhead is ≤~25% relative to no-chunking; at 1024–2048, overhead is small (Figure 14).
    - Technique isolation (Table 4, Yi-34B, token budget 1024):
      - `hybrid-batching-only` (no chunking): P99 TBT is high (0.68s openchat; 1.38s arxiv) due to long-prefill stalls.
      - `chunked-prefills-only` (no hybrid decoding): P50 TTFT is high (1.04s openchat; 5.38s arxiv) since prefills are computed without benefiting from decode piggybacking.
      - Combined (Sarathi-Serve): low P99 TBT (0.14–0.17s) while keeping P50 TTFT relatively low (0.76–3.90s).

- Do the experiments support the claims?
  - Yes. The paper systematically connects the mechanism (memory-bound decodes can carry compute-bound prefill chunks) to measured outcomes:
    - Micro-level: arithmetic intensity and operator timing (Figures 3–6) explain why hybrid batches can be made stall-free.
    - Scheduler-level: timelines (Figure 7) show how generation stalls arise and are eliminated.
    - System-level: capacity vs SLO curves (Figure 12), PP viability (Figure 13), and large-model results (Figures 10–11) demonstrate generality across hardware and workloads.
  - Robustness checks include:
    - Multiple models and parallelization strategies (Table 1).
    - Two distinct workload profiles (Table 2).
    - Sensitivity to token budget selection and chunking overheads (Figure 14, §4.3).

## 6. Limitations and Trade-offs
- Token-budget selection requires profiling and is workload- and hardware-dependent
  - The optimal budget depends on TBT SLOs, PP/TP configuration, GPU tiling sensitivities, and workload prompt lengths (§4.3). The paper uses one-time profiling (Vidur) and fixed budgets per SLO; dynamic adaptation is not implemented (§5.1).
- Chunking adds overhead and extra KV reads
  - Splitting prefills means earlier chunks’ KV-cache is re-read by later chunks. Overheads are measured as moderate (≤~25% at chunk 512), but they are nonzero (Figure 14, §4.3).
- Sensitivity to GPU kernel tiling
  - Poorly chosen chunk sizes can trigger tile-quantization inefficiencies (e.g., 257 vs 256) with noticeable slowdown (§4.3).
- Not a disaggregated architecture
  - Systems like SplitWise/DistServe/TetriInfer split prefills and decodes across different replicas and migrate KV caches (§6). Sarathi-Serve keeps both on the same replica; it avoids migration costs but does not exploit full decoupling (and does not compare experimentally to those designs).
- Fairness and preemption are out of scope
  - The scheduler is throughput–latency optimized; multi-tenant fairness and preemption policies (e.g., in FastServe or fairness work in §6) are complementary but not integrated or evaluated here.
- Scope of hardware/software dependencies
  - Results are on A100/A40 GPUs and depend on optimized attention kernels (FlashAttention v2/FlashInfer) and vLLM’s PagedAttention (§4.4). Behavior on other accelerators or less optimized stacks may differ.

## 7. Implications and Future Directions
- What changes in the field
  - The long-accepted throughput–latency tradeoff for online LLM serving is not fundamental. By sizing and pacing prefill work, servers can deliver high-throughput iteration-level batching without harming token-streaming latency. This also makes pipeline-parallel serving practical over commodity networks by reducing bubbles (Figures 11–13).
- Follow-up research enabled
  - Adaptive token budgeting: learn or control-theoretically adjust chunk sizes and budgets online as load/latency fluctuates (§4.3 hints at this; authors fix budgets per SLO in §5.1).
  - Integration with fairness and preemption: combine stall-free batching with multi-tenant fairness or SRT-style preemption to optimize both responsiveness and equity (§6).
  - Comparative study with disaggregated serving: quantify the trade-offs between stall-free co-resident scheduling (no KV migration) and disaggregated pipelines (maximal isolation, potential network costs) (§6).
  - Extending to other model classes: how do MoE routing, retrieval-augmented attention, or very long-context models alter chunking/budgeting?
  - Compiler/runtime co-design: make chunk sizes tile-friendly automatically to avoid quantization slowdowns; fuse chunk-aware attention/KV-cache handling.
- Practical applications
  - Interactive assistants, code copilots, and search: stricter tail-latency SLOs benefit from stall-free batching (Figure 12 left), with improved capacity reducing serving cost.
  - Document-heavy copilots (e.g., summarization, long-context analytics): chunked-prefills prevent long prompts from causing streaming stalls (Table 2, Figure 9), stabilizing user experience.
  - Large-model deployments across commodity clusters: by minimizing PP bubbles (Figure 13), enterprises can serve 70B–180B-class models without needing expensive intra-node TP-only configurations.

> Bottom line: Sarathi-Serve operationalizes a simple but powerful principle—co-run only as much prefill as a decode iteration can absorb—turning a formerly hard tradeoff into a tunable knob. The result is consistently higher capacity at tight tail-latency SLOs across models, hardware, and workloads (Figures 10–13).
