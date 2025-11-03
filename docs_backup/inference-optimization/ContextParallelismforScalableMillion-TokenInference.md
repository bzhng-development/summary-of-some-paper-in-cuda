# Context Parallelism for Scalable Million-Token Inference

**ArXiv:** [2411.01783](https://arxiv.org/abs/2411.01783)

## 🎯 Pitch

This paper introduces context parallelism, a novel system-level method for scaling long-context large language model inference to millions of tokens by distributing input tokens—and their costly KV cache—across many GPUs, rather than just sharding model weights. The authors develop two lossless, latency-optimized ring attention algorithms (pass-KV and pass-Q) and adaptive heuristics that achieve near-linear prefill scaling, demonstrated by prefilling 1 million tokens on Llama3 405B in just 77 seconds with high parallel and hardware efficiency. This breakthrough directly tackles the critical latency bottleneck for state-of-the-art LLMs at 128K–1M context, making practical, cost-effective deployment of long-context applications possible in commercial data centers.

---

## 1. Executive Summary
This paper introduces context parallelism (CP) for long‑context large language model (LLM) inference and makes it practical at scale by building two latency‑optimized ring‑attention variants—`pass-KV` and `pass-Q`—plus load‑balancing and selection heuristics. On up to 128 H100 GPUs (16 nodes), the system achieves near‑linear prefill scaling, including a 1‑million‑token prefill of `Llama3 405B` in 77 seconds with 93% parallelization efficiency and 63% FLOPS utilization (Abstract; Appendix A; Figure 8).

## 2. Context and Motivation
- Problem addressed
  - Long‑context inference (hundreds of thousands to millions of tokens) is prohibitively slow and memory‑hungry because attention cost grows quadratically with context length and the key/value (KV) cache grows linearly (Section 2.2; Table 3).
  - On a single H100 host (8 GPUs), `Llama3 405B` needs ~60 s for 128K tokens or ~1200 s for 1M tokens in prefill (Introduction).
  - Existing multi‑GPU scaling (tensor parallelism, pipeline parallelism) either increases throughput rather than latency or incurs large cross‑host communication (Sections 3.2 and Table 2).

- Why it matters
  - Production LLMs already expose 128K–1M context (e.g., GPT‑4o 128K, Claude 200K, Gemini 1.5 Pro 1M; Section 1 and References). Without faster prefill, user latency and serving cost are impractical for many applications.

- Shortcomings of prior approaches
  - Tensor Parallelism (`TP`) shards weights and requires frequent cross‑GPU `AllReduce` across many linear layers. Cross‑host traffic is large and scales poorly (Table 2; Section 4.2.2, Figure 7).
  - Prior “ring attention” work focused on training with equal-length sequences and no persistent KV; it did not address multi‑turn inference constraints like partial prefill and decoding with cached KV (Section 1).

- Positioning
  - The paper targets system‑level optimizations (Section 2.3), preserving dense attention while distributing work across GPUs differently: shard the input sequence across GPUs by position (`context parallelism`) and exchange only attention tensors (`Q/K/V`) instead of sharded weights (Sections 3.2–3.3).
  - It contributes inference‑specific ring‑attention variants, load‑balanced sharding for variable lengths, and a principled, runtime selection heuristic (Sections 3.4–3.6; Algorithms 1–5).

## 3. Technical Approach
This section explains how the system achieves low‑latency long‑context inference. Core concepts:
- `prefill`: the forward pass over the prompt before generating the first token.
- `decode`: autoregressive generation, one token per step.
- `KV cache`: stored key/value projections per layer used so later tokens can attend to earlier tokens without recomputation.
- `CP` vs `TP`: CP shards tokens across nodes; TP shards weights within a node (Section 3.2; Figure 5). The system uses `TP8` within each node and `CPN` across nodes: `CPN+TP8`.

3.1 Why context parallelism helps
- Communication hotspot shifts: TP communicates during every linear layer; CP communicates only during attention, which occurs once per block (Table 2).
- Message size advantage with GQA: modern models often use `NH` query heads and far fewer `NKV` key/value heads. With `Llama3 405B`, `NH=128`, `NKV=8`. Passing KV instead of Q cuts message size by 16× (NH/NKV) (Table 2; Section 3.2).
- Memory trade‑off: CP doesn’t shard weights across nodes, so each node must still hold its full TP shard; authors use row‑wise FP8 quantization of feedforward layers so `Llama3 405B` fits in a single 8‑GPU node with `TP8` (Section 4.1).

3.2 Inference phases that must be supported (Section 3.3)
- Full prefill: no prior cache (`P=0`); tokens attend to earlier tokens in the same prompt.
- Partial prefill (persistent KV prefill): new prompt fragment of length `T` must attend over cached KV of length `P` plus itself.
- Decode: generate one token at a time; append its KV to the cache.

3.3 Communication/computation modeling and variant selection (Section 3.4; Table 3; Equations 1–3, 5)
- Shapes: `Q ∈ [T, NH, D/NH]`, `K,V ∈ [T+P, NKV, D/NH]`. With GQA, `NKV << NH`.
- Two ring styles:
  - `pass-KV`: circulate KV blocks; keep Q local.
  - `pass-Q`: circulate Q blocks; keep KV local. After the ring, an `All2All` is needed to return partial outputs to source ranks (Section 3.5.3).
- When to pass which?
  - Message size criterion (Eq. 1): choose `pass-Q` when the KV cache is large relative to new tokens:
    - `Q` is smaller than `KV` if `T/(T+P) ≤ 2*(NKV/NH)`. For `Llama3 405B`, threshold `≈ 12.5%`.
  - Roofline overlap conditions:
    - `pass-KV` communication hides under compute if `T ≥ N*C*NKV*e / (2*NH*BW)` (Eq. 2).
    - `pass-Q` comm hides if `(T+P) ≥ N*e*C/(4*BW)` (Eq. 3).
  - Combined heuristic (Algorithm 1): choose `pass-KV` when either T is large enough for overlap (Eq. 2) or cache miss rate `T/(T+P)` is above the 12.5% threshold; otherwise choose `pass-Q`.
  - Refinement including `All2All` cost for `pass-Q` (Eq. 5; Algorithm 5): the `pass-Q` region shrinks because its final `All2All` can dominate when `T` is larger.

3.4 Load‑balanced sharding for variable lengths (Section 3.5.1; Figures 1–2)
- Goal: balance compute and KV memory across CP ranks even with variable sequence lengths and multi‑turn conversation history.
- Strategy:
  - Full prefill: split each sequence into `2N` contiguous chunks `C0...C(2N-1)` and assign pair `(Ci, C(2N-i-1))` to rank `i`. This balances the O(T^2) causal‑attention work across ranks (Figure 1).
  - Partial prefill: shard only along the dimension of new tokens `T` (in `2N` chunks), independent of how cached tokens `P` are distributed (Figure 2). This keeps compute balanced even if history is skewed.

3.5 Ring `pass-KV` for prefill, including persistent KV (Section 3.5.2; Algorithm 2; Figure 3)
- Challenge: collective communication libraries prefer equal‑sized messages, but per‑rank KV lengths differ (due to past turns and padding).
- Solution: circulate uniformly sized KV blocks of length `max_i P_i + ceil(T/N)`. At each of the `N` ring steps, a rank:
  - Receives the next KV block (`SendRecv`).
  - Computes partial attention between its local `Qk` and the current KV block (`Attn(Qk, KV_s)`).
  - Overlaps communication with compute (Figure 3).
- After `N` steps, each rank holds N partial results `O_k^s`. A numerically stable “merge attention” combines them into the final output using the blockwise softmax trick (Appendix B, Eq. 4).

3.6 Ring `pass-Q` for partial prefill and decode (Sections 3.5.3 and 3.6; Algorithms 3–4; Figure 4)
- Keep KV stationary; circulate Q across the ring to reduce message size when `P` is large.
- Each rank computes `Attn(Q_s, KV_k)` locally as Q passes by; partial outputs are now “owned” by the KV ranks, so a final `All2All` returns each piece to the Q’s source (Figure 4; Algorithm 3).
- Decode‑specific load balancing: to avoid one rank hoarding all decode KV (which would hit memory limits), distribute one token per sequence per step in round‑robin across ranks (“offset by 1 index” per iteration) and route using `bids` (batch IDs) so each rank reads the right KV slice (Algorithm 4).

3.7 System setting and kernels (Section 4.1)
- Hardware: Grand Teton platforms with 8× H100 per node. Two fabrics:
  - GTT: 400 Gb/s per GPU RDMA back‑end.
  - GTI: 100 Gb/s per GPU TCP/IP front‑end.
- Model: `Llama3 405B` with FP8 row‑wise quantized feed‑forward layers. `TP8` within node; `CPN` across nodes (Figure 5).
- Kernels: FlashAttention‑3 for prefill; Flash‑Decoding with 256 KV splits for decode; CUDA Graphs for small decode steps.

## 4. Key Insights and Innovations
- Two complementary ring‑attention variants for inference (Sections 3.5.2–3.5.3)
  - `pass-KV` optimized for full prefill and higher miss rates; fully overlaps `SendRecv` with compute in the ring; no global collectives afterward. This is different from the all‑gather‑based KV scheme used during training and avoids a large sync on the critical path (Section 3.5.2).
  - `pass-Q` optimized for partial prefill and decode with large `P`; reduces message size but requires an `All2All` to reunite partial outputs (Section 3.5.3).
  - Significance: enables low latency across the full spectrum of KV hit rates rather than one mode of operation.

- Analytic and empirical selection heuristics (Section 3.4; Algorithms 1 and 5; Appendix D)
  - Roofline‑based thresholds give simple, static criteria tied to hardware (`C`, `BW`) and model (`NH`, `NKV`) that decide when communication hides under compute (Eqs. 2–3).
  - Incorporating `All2All` (Eq. 5) makes the selector realistic for `pass-Q`.
  - An additional data‑fit heuristic on `(log T, log(T/(T+P)))` provides a practical auto‑tuner when real‑world performance deviates from peaks (Appendix D; Figure 10).

- Load‑balanced sharding for both compute and KV memory with variable‑length, fused batches (Section 3.5.1; Figures 1–2)
  - Prior training‑oriented works assume uniform lengths; this paper handles fused, uneven prompts and multi‑turn history and keeps both memory and compute balanced.

- Decode KV distribution via round‑robin `pass-Q` (Section 3.6; Algorithm 4)
  - Ensures no single rank becomes a KV hotspot during long conversations—a practical requirement for persistent sessions.

These are fundamental system innovations rather than small parameter tweaks: the variants, the balancing scheme, and the selection logic collectively enable the reported scaling.

## 5. Experimental Analysis
Evaluation methodology and setup (Section 4.1)
- Model: `Llama3 405B` (126 layers, `D=16384`, `NH=128`, `NKV=8`; Table 9) with FP8 row‑wise quantized FFN.
- Hardware: up to 16 nodes (128 H100s). Two fabrics:
  - GTT (RDMA 400 Gb/s/GPU) and GTI (TCP/IP 100 Gb/s/GPU).
- Parallelism: `TP8` inside each node; `CPN` across nodes; one CP group per KV head (Figure 5).
- Metrics:
  - `TTFT` (time‑to‑first‑token): dominated by prefill.
  - `TTIT` (time‑to‑incremental‑token): per‑token decode latency.

5.1 Prefill latency vs number of CP nodes (Section 4.2.1; Figure 6)
- On GTT (RDMA), `pass-KV` full prefill scales near‑linearly from 1 to 8 nodes. For 128K tokens:
  - “With `CP8` on GTT… 128K token prefill in 5.85 s.” (Figure 6a).
- On GTI (TCP/IP), similar scaling up to 4 nodes despite much lower inter‑host bandwidth; achieved ~3 GB/s per rank and still overlapped comm with compute (Section 4.2.1; Figure 6b).
- Interpretation: the ring `pass-KV` successfully hides comm as anticipated by Eq. (2).

5.2 CP vs multi‑node TP scaling (Section 4.2.2; Figure 7)
- Scaling ratio is `τ1/τN` (higher is better; perfect scaling equals `N`).
- Result:
  - “Difference is ~15% between CP2 and TP16 on 2 nodes, but grows to ~100% at 8 nodes.” (Figure 7).
- Why: TP’s cross‑host `AllReduce` on many linear layers increasingly dominates; CP communicates less often with smaller messages (Table 2).

5.3 Scaling context capacity and latency to 1M tokens (Section 4.2.3; Figure 8; Appendix A)
- By sharding KV across CP ranks, capacity scales with nodes.
- Results with `pass-KV` full prefill:
  - “1M context in 77 s” and “128K in 3.8 s” on 16 nodes (Figure 8).
  - Appendix A computes per‑GPU achieved 502 TF/s on H100 and 93% parallelization efficiency; overall ~63% MFU given the power‑limited configuration.

5.4 Partial prefill: `pass-KV` vs `pass-Q` as KV miss rate varies (Section 4.2.4; Table 4–5; Figure 9)
- Setup: 128K total context (`P+T=128000`) on `CP4` (GTT). Vary `T` (miss rate `T/(T+P)`).
- Findings:
  - TTFT grows roughly linearly with miss rate for both variants (Table 4).
  - Crossover around 5% miss rate: below that, `pass-Q` is faster; above that, `pass-KV` wins (Figure 9).
  - Example numbers (Table 4):
    - 2.5% miss (`T=3200`): `pass-Q` 1046 ms vs `pass-KV` 1110 ms.
    - 5% miss (`T=6400`): essentially tied (1302 vs 1305 ms).
    - 10% miss (`T=12800`): `pass-KV` 2081 ms vs `pass-Q` 2205 ms.
- Time breakdown explains the crossover (Table 5):
  - At 2.5% miss, the exposed `SendRecv` for `pass-KV` across `N-1` ring steps exceeds the single `All2All` of `pass-Q`, so `pass-Q` wins.
  - At 10% miss, `pass-KV` communication is hidden under the much larger attention compute, so `pass-KV` wins.
- Alignment with theory:
  - The 12.5% message‑size threshold (Eq. 1) gives a hard upper bound: beyond it `pass-KV` is always favorable; the empirical crossover earlier (≈5%) is explained by `All2All` overhead (Eq. 5).

5.5 Decode performance (Section 4.3; Tables 6–8)
- Decode uses `pass-Q` with round‑robin sharding (Algorithm 4) and CUDA Graphs.
- Context length scalability at batch size 1 (Table 6):
  - `TP8` TTIT ≈ 44–46 ms across 8K–128K contexts.
  - `CP2+TP8` TTIT increases to ~60–66 ms; however, TTFT halves (e.g., 42 s → 21 s at 128K).
- Parallelism scalability at 128K, batch size 1 (Table 7):
  - `CP1+TP8` TTIT 46.26 ms; `CP2+TP8` 60.23 ms; `CP4+TP8` 71.31 ms.
  - Multi‑node TP (`TP16`, `TP32`) also degrades TTIT due to more cross‑host comm, but less than CP because TP avoids the `All2All` after attention.
- Microanalysis (Table 8):
  - As CP grows, per‑rank attention becomes faster (shorter effective sequence), but total `pass-Q` time rises due to more `SendRecv` iterations and the fixed‑cost `All2All`.
- Takeaway: CP is excellent for prefill latency; decode latency worsens as CP increases. The paper recommends decoupling prefill and decode onto different resources (Section 4.3, last paragraph), consistent with disaggregated serving designs (References to Mooncake and DistServe).

Assessment of evidence
- The prefill results convincingly show near‑linear scaling and robustness across different network fabrics (Figures 6–8).
- The `pass-KV` vs `pass-Q` analysis is strong: it provides theory (Eqs. 1–3, 5) and matching empirical break‑even points (Tables 4–5, Figure 9).
- Decode findings are honest about regressions and carefully analyzed (Tables 6–8).

## 6. Limitations and Trade-offs
- Memory/placement assumptions
  - CP does not shard weights across nodes; each node must fit its `TP8` shard. The results rely on row‑wise FP8 quantization to fit `Llama3 405B` in 8× H100 (Section 4.1). Models without such quantization or with larger layers may exceed per‑node memory.
- Dependence on GQA head asymmetry
  - The communication advantage of `pass-KV` vs `pass-Q` hinges on `NH >> NKV` (e.g., 128 vs 8). Models without this asymmetry reduce the benefit (Table 2; Eq. 1).
- Heuristic selection requires calibration
  - The thresholds use peak `C` (compute) and `BW` (bandwidth), but “achieved” values are lower; the paper notes it “fine‑tunes the thresholds based on empirical data” (footnote under Section 3.5). Appendix D introduces an empirical linear model; correctness depends on the deployment environment.
- Decode scaling
  - `pass-Q` needs `All2All` after the ring; padding to equalize per‑rank Q counts further hurts TTIT at small batch sizes (Section 4.3; Table 8). Without decoupled serving, prefill gains come with decode costs.
- System scope
  - The work is system‑level and does not evaluate accuracy tasks or end‑to‑end throughput under multi‑tenant loads. Straggler tolerance, failure handling, and dynamic load shedding are not covered.
- Communication regularity
  - Ring algorithms rely on equal‑sized messages; handling highly skewed histories requires padding to the max (`max_i P_i + ceil(T/N)`), adding overhead (Section 3.5.2).

## 7. Implications and Future Directions
- Impact on the field
  - Establishes context parallelism as a practical lever for long‑context inference latency, complementing (not replacing) tensor/pipeline parallelism. This is especially relevant as context windows reach millions of tokens.
  - Shows that with careful ring‑attention design and selection heuristics, cross‑host networks as modest as 100 Gb/s per GPU (TCP/IP) can still yield good scaling for large prefill workloads (Figure 6b).

- What this enables
  - Multi‑million‑token serving on commodity multi‑node clusters for applications like whole‑codebase analysis, multi‑hour meeting or video summarization, and large‑document synthesis—where TTFT previously dominated latency budgets.
  - Integration with disaggregated serving stacks that split prefill and decode onto different nodes (Section 4.3), aligning with recent KV‑centric architectures (References).

- Research directions
  - Reduce or eliminate `All2All` for `pass-Q` (e.g., hierarchical or fused collectives; smarter output accumulation that avoids a global exchange).
  - Better overlap and padding removal for decode to approach the per‑op gains shown in Table 8 while mitigating communication costs.
  - Online, model‑agnostic selector that learns thresholds from live telemetry (beyond Appendix D’s linear fit).
  - Joint use with approximate/retrieval methods for ultra‑long contexts so exact attention is applied only where helpful (Conclusion).
  - Extending to heterogeneous clusters and elastic scaling; straggler‑aware scheduling for fused var‑seq batches.

- Practical applications
  - Long‑form assistants ingesting books, code repositories, or logs (1M tokens ~ “~1 hour of video content”; Section 4.2.3).
  - Interactive agents with persistent memory where partial prefill performance matters; the `pass-Q`/`pass-KV` selector directly optimizes user‑perceived latency under varying KV hit rates (Table 4, Figure 9).

> Headline result: “1M context prefill with `Llama3 405B` in 77 s (93% parallelization efficiency, 63% FLOPS utilization) and 128K in 3.8 s” (Abstract; Figure 8; Appendix A).

> Break‑even guidance: For `Llama3 405B` (`NH=128`, `NKV=8`), `pass-Q` is preferable when the KV miss rate is under ≈5%, and `pass-KV` wins above ≈5%; `pass-KV` is always favorable above 12.5% by message‑size theory (Table 4–5; Figure 9; Eq. 1).
