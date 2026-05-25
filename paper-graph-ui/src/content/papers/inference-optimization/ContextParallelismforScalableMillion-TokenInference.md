# Context Parallelism for Scalable Million-Token Inference

**ArXiv:** [2411.01783](https://arxiv.org/abs/2411.01783)

## 🎯 Pitch

This paper introduces context parallelism for inference in large language models, enabling near-linear latency scaling for million-token contexts by distributing computation and memory across up to 128 GPUs—without altering model architecture or relying on approximations. The system’s innovations, including two lossless ring attention variants (pass-KV and pass-Q), make it possible to serve exact attention over million-token prompts in just seconds, unlocking practical real-world applications and dramatically reducing latency for LLM-powered systems with ultra-long contexts.

---

## 1. Executive Summary
This paper introduces context parallelism (CP) for large‑language‑model (LLM) inference and two exact “ring attention” variants—`pass‑KV` and `pass‑Q`—that collectively deliver near‑linear latency scaling for very long contexts across up to 128 H100 GPUs. The system achieves 1M‑token prefill in 77 seconds and 128K in 3.8 seconds on Llama‑3 405B while keeping FLOPS utilization high (Section 4.2.3; Appendix A), making million‑token exact attention practical on commodity multi‑node clusters.

## 2. Context and Motivation
- Problem addressed
  - Serving LLMs with very long prompts (128K–1M tokens) is slow and memory‑intensive. Prefill time grows super‑linearly because attention cost scales quadratically with context length; the KV cache (stored keys and values from prior tokens) grows linearly (Sections 2.2–2.3).
  - On a single 8×H100 node, a 128K prefill can take ~60 s and a 1M prefill ~1200 s for Llama‑3 405B (Section 1). This is unacceptable for interactive applications.

- Why it matters
  - Modern LLM products now expose 128K–1M context windows (Introduction; Background 2.1–2.3). Reducing prefill latency directly improves time‑to‑first‑token (TTFT) and user experience, and enabling million‑token exact attention widens application scope (e.g., long videos, large document sets).

- Prior approaches and gaps
  - Architectural and post‑training methods reduce attention cost by approximation or sparsity (Section 2.2–2.3), but do not deliver exact attention at million‑token scale.
  - System parallelisms:
    - Pipeline Parallelism (PP) boosts throughput but not latency (Section 3.2).
    - Tensor Parallelism (TP) shards weights but demands large inter‑node AllReduce bandwidth; scaling degrades across nodes (Table 2; Section 4.2.2).
    - Ring attention (previous work) focused on training with uniform sequence lengths and without persistent KV across multi‑turn inference (Introduction; Section 3.4).
  - Missing piece: a latency‑oriented, exact‑attention inference system that (1) scales across nodes, (2) handles multi‑turn sessions with persistent KV, and (3) balances compute, memory, and communication on heterogeneous networks.

- Positioning
  - This work stays within exact dense attention and introduces CP tailored for inference: shard along sequence length, communicate Q/K/V only in attention layers, and leave weights unsharded across nodes (weights are TP‑sharded only within a node). The paper contributes new algorithms, load‑balancing, and runtime heuristics to make this scalable and robust (Sections 3.2–3.6).

## 3. Technical Approach
Key terms (defined once, only where uncommon):
- `context parallelism (CP)`: distribute tokens of a sequence across devices (shard along sequence length), so each device computes attention for its token chunk and exchanges only Q/K/V as needed (Section 3.2).
- `prefill`: compute hidden states and KV cache for the input prompt before any generation; first token latency equals TTFT (Section 3.3).
- `decode`: autoregressive generation, one token at a time; per‑token latency is TTIT (Section 3.3).
- `persistent KV prefill` (partial prefill): a new user prompt attends to previously cached KV from earlier turns and to itself (Section 3.3).
- `GQA` (grouped‑query attention): models with many query heads (`NH`) but far fewer key/value heads (`NKV`), e.g., Llama‑3 405B has NH=128, NKV=8 (Table 9).

A. Why CP over TP across nodes
- Communication pattern and size
  - TP communicates on every linear layer via AllReduce; CP communicates only Q/K/V on attention layers via SendRecv (Table 2).
  - With GQA (NKV << NH), sending KV can be dramatically smaller than sending Q: in Llama‑3 405B, NH:NKV = 128:8, so KV messages are 16× smaller than Q messages (Section 3.2).
- Practical system design
  - Use TP within a node (TP8) to fit model weights in HBM; scale out across nodes with CP to minimize inter‑node traffic (Section 3.2; Figure 5).

B. Workload model: full prefill, partial prefill, decode (Section 3.3)
- Full prefill: first prompt, no existing KV (`P=0`).
- Partial prefill: new tokens of length `T` attend to cached tokens `P`.
- Decode: one new token per sequence (`T=1`) attends to cached tokens `P`.

C. Communication vs compute modeling (Section 3.4; Table 3)
- Shapes: `Q=[T, NH, D/NH]`, `K=V=[T+P, NKV, D/NH]`.
- Which to pass: KV or Q?
  - Prefer `pass‑KV` if `T/(T+P) ≤ 2*NKV/NH` (Equation (1)) because KV messages are smaller. For full prefill (`P=0`) this usually holds in GQA. For decode (`T=1`), Q is generally smaller.
- Can communication be hidden under compute?
  - For `pass‑KV` across `N` nodes, ring SendRecv is hidden if `T ≥ N * C * NKV * e / (2 * NH * BW)` (Equation (2)), independent of `P`.
  - For `pass‑Q`, ring SendRecv is hidden if `(T+P) ≥ N * e * C / (4 * BW)` (Equation (3)); larger total context helps.
- Heuristic (Algorithm 1): choose `pass‑KV` if either the compute‑overlap threshold (Eq. 2) or the size threshold (Eq. 1) is met; otherwise choose `pass‑Q`. Appendix C refines this by also accounting for an All2All at the end of `pass‑Q` (Equation (5), Algorithm 5).

D. Load‑balanced sharding of tokens and KV (Section 3.5.1)
- Challenge: causal attention makes later tokens attend to more history, so naive splits imbalance compute and KV memory across ranks.
- Method: evenly partition each sequence into `2N` chunks and assign pair `(Ci, C_{2N−i−1})` to CP rank `i`—this balances both compute and KV memory (Figures 1–2).
- For partial prefill, apply balancing on the new tokens `T` regardless of how cached tokens `P` are sharded (Figure 2).

E. Exact ring attention variants adapted for inference
1) `Ring pass‑KV` for prefill (Section 3.5.2; Figure 3; Algorithm 2)
   - K/V tensors move in a ring; each rank keeps local Q chunk.
   - In each of `N` ring steps, a rank:
     - sends its current K/V block and receives the next one (equal‑sized messages enforced via padding; lines 5–15 of Algorithm 2),
     - computes partial attention `GQA(Q_k, KV_s)` and accumulates results (line 12).
   - After the ring, combine partial results using “merge attention”—a numerically stable softmax merge based on per‑block log‑sum‑exp (`LSE`) (Appendix B, Equation (4)).
   - Why this helps: avoids a large all‑gather upfront (used in training), enables overlap of per‑step SendRecv with per‑step attention compute, and handles variable‑length fused batches.

2) `Ring pass‑Q` for prefill (Section 3.5.3; Figure 4; Algorithm 3)
   - Q tensors move in a ring; K/V remain stationary.
   - Each rank computes attention of received Q against its local KV at every step (line 8).
   - Because outputs are “owned” by the Q’s source rank, a final `All2All` plus permutation is needed to return partial outputs to owners (line 12).
   - Trade‑off: ring SendRecv can be hidden (Eq. 3), but the final `All2All` is on the critical path. Appendix C incorporates this into the selection rule (Eq. 5; Algorithm 5).

3) `Ring pass‑Q` for decode (Section 3.6; Algorithm 4)
   - One new token per sequence; to avoid one rank accumulating all decode KV (which would OOM), the system round‑robins which rank owns each sequence’s decode step across iterations.
   - This preserves balanced KV capacity across ranks. As with pass‑Q prefill, a final `All2All` is needed.

F. System and implementation (Section 4.1)
- Model: Llama‑3 405B with row‑wise FP8 weights; NH=128, NKV=8; 126 layers (Table 9).
- Hardware: Grand Teton nodes with 8×H100 (96GB) and NVLink; two interconnect variants:
  - GTT (training fabric): RDMA 400 Gb/s per GPU.
  - GTI (inference fabric): front‑end TCP/IP 100 Gb/s per GPU.
- Parallelism topology: TP8 within each node; CP across 1–16 nodes; one CP group per KV head; ring is 8‑way SendRecv (Figure 5).
- Kernels: FlashAttention‑3 for prefill, Flash Decoding with 256 K/V splits; CUDA Graphs for decode to remove kernel launch overhead (Section 4.3).

## 4. Key Insights and Innovations
- Two inference‑oriented, exact ring attention variants and a runtime switch
  - `pass‑KV` and `pass‑Q` are adapted for inference with variable sequence lengths and persistent KV (Sections 3.5–3.6). The switch uses analytically derived thresholds (Equations (1)–(3)) and a refined rule that includes `All2All` cost (Appendix C, Equation (5), Algorithm 5).
  - Significance: enables exact long‑context prefill and partial prefill to run near‑linearly across many nodes, while choosing the lower‑latency variant per request.

- Load‑balanced sharding of both compute and KV capacity
  - The `2N`‑chunk pairing (`Ci`, `C_{2N−i−1}`) ensures no rank becomes the compute or memory bottleneck, even across multi‑turn sessions where decode extends KV unevenly (Section 3.5.1; Figures 1–2).
  - Significance: avoids OOM and preserves scaling for real‑world variable‑length batches.

- Decode algorithm that preserves KV balance
  - Round‑robin ownership of decode steps plus `pass‑Q` avoids concentrating all new KV on a single rank (Section 3.6; Algorithm 4).
  - Significance: supports multi‑turn chat without sacrificing KV capacity per rank.

- Multi‑node CP with TP‑within‑node to minimize inter‑node traffic
  - CP communicates small Q/K/V messages at attention layers; TP would otherwise AllReduce activations on every linear layer (Table 2).
  - Significance: demonstrates robust scaling even on lower‑bandwidth TCP clusters (Figure 6b), broadening deployability beyond specialized RDMA fabrics.

- Empirical and analytical guidance
  - Analytical switch (Algorithms 1 and 5) and an optional compact empirical rule `h(T,P)=α log T + β log(T/(T+P)) + γ` to pick between `pass‑KV` and `pass‑Q` (Appendix D; Figure 10).
  - Significance: makes the system adaptable at runtime to request mix and network characteristics.

## 5. Experimental Analysis
- Setup (Section 4.1)
  - Model: Llama‑3 405B FP8; FlashAttention‑3 for prefill; Flash Decoding for generation.
  - Hardware: 1–16 nodes, each 8×H100; NVLink inside node; inter‑node: GTT RDMA 400 Gb/s/GPU or GTI TCP/IP 100 Gb/s/GPU.
  - Parallelism: CP across nodes, TP8 within node (Figure 5).
  - Metrics: TTFT (prefill latency), TTIT (per‑token decode latency).
  - Algorithms evaluated: `pass‑KV` and `pass‑Q` for full/partial prefill; `pass‑Q` for decode.

- Main results
  - Near‑linear prefill scaling with `pass‑KV` (Figures 6a–6b)
    - Quote:
      > On GTT, CP8 processes 128K tokens in 5.85 s (Figure 6a). On GTI (TCP), scaling holds up to 4 nodes with similar trends (Figure 6b), with achieved inter‑host bandwidth ~3 GB/s per rank.
    - Interpretation: ring SendRecv is successfully overlapped with attention compute for large `T` (Equation (2)).

  - CP vs multi‑node TP (Figure 7)
    - Quote:
      > At 2 nodes, CP vs TP latency differs by ~15%; at 8 nodes, TP is ~2× slower (scaling ratio diverges).
    - Reason: TP’s inter‑node AllReduce becomes the bottleneck as group size grows; CP’s attention‑only SendRecv scales better across nodes (Table 2).

  - Scaling context length with fixed capacity (Figure 8; Appendix A)
    - Quote:
      > CP16 achieves exact 1M‑token prefill in 77 s and 128K in 3.8 s (Figure 8), with 93% parallelization efficiency and ~63% FLOPS utilization (Appendix A).
    - Interpretation: despite quadratic attention cost (Table 3), sufficient CP ranks make million‑token exact attention feasible on 16 nodes.

  - Partial prefill: when to use `pass‑Q` vs `pass‑KV` (Table 4; Figure 9; Table 5)
    - Quote:
      > At 128K total, `pass‑Q` is faster when KV cache miss rate < 5% (e.g., `P=126,720`, `T=1,280`: 898.7 ms vs 1,023.4 ms). When miss rate > 5%, `pass‑KV` wins (e.g., `P=115,200`, `T=12,800`: 2,080.7 ms vs 2,205.3 ms) (Table 4; Figure 9).
      > Micro‑breakdown at 2.5%: exposed `pass‑KV` ring SendRecv exceeds `pass‑Q` All2All (Table 5), explaining `pass‑Q`’s advantage. At 10%, attention compute hides SendRecv, so `pass‑KV` is better.
    - Relation to theory: the 12.5% size threshold from Equation (1) (2·NKV/NH = 2·8/128) is an upper bound; after including All2All (Appendix C, Equation (5)) the empirical tipping point shifts to ~5%, matching Table 4 and Figure 9.

  - Decode performance and scaling (Section 4.3; Tables 6–8)
    - Single‑node vs CP2:
      > With 128K context, TP8 TTIT is 46.26 ms; CP2+TP8 increases TTIT to 60.23–66.63 ms across contexts (Table 6).
    - Scaling CP further:
      > At 128K, CP4+TP8 TTIT is 71.31 ms; TP32 is 47.3 ms (Table 7). Table 8 shows per‑op timings: individual attention gets faster with more CP ranks (effective context per rank shrinks), but ring SendRecv + final All2All dominate, increasing total pass‑Q time.
    - Conclusion: CP primarily benefits prefill. Decode scaling is limited by communication and padding; the paper recommends decoupling prefill and decode placements in a serving stack (Section 4.3).

- Do experiments support claims?
  - Yes for prefill: Figures 6–8 show consistent latency scaling across networks and up to 16 nodes, with concrete end‑to‑end numbers and utilization analysis (Appendix A).
  - Yes for the `pass‑KV`/`pass‑Q` switch: Table 4 and Table 5 validate the analytically derived thresholds and the influence of All2All (Appendix C).
  - Decode: authors explicitly acknowledge limited scalability and analyze the causes (Tables 6–8), aligning with their positioning of CP as a prefill accelerator.

- Ablations and robustness
  - Detailed micro‑timings for ring steps and All2All (Table 5, Table 8).
  - Two fabrics (RDMA vs TCP) to demonstrate robustness (Figure 6).
  - Analytical vs empirical heuristic alignment (Algorithm 1 vs Algorithm 5; Appendix D).

## 6. Limitations and Trade-offs
- Model sharding vs memory
  - CP does not shard weights across nodes; each node must host a TP8 shard of the full model (Section 3.2). Extremely large models exceeding TP8‑per‑node memory may still be constrained.
- Decode scalability
  - `pass‑Q` decode introduces per‑step ring SendRecv and a final All2All; TTIT worsens as CP increases (Tables 7–8). Padding to equalize message sizes also adds overhead for small batches (Section 4.3).
- Dependence on GQA head counts
  - CP’s communication advantage when passing KV relies on `NKV << NH`; models without this skew may see smaller gains (Table 2; Equation (1)).
- Quadratic attention remains
  - CP parallelizes exact attention but does not change its O(T^2) compute. At very long contexts, attention dominates TTFT (Section 4.2.3; Figure 8).
- Heuristic sensitivity
  - The runtime switch depends on estimated compute `C` and bandwidth `BW`. In practice, these are tuned empirically (note under Algorithm 1; Appendix D), and misclassification is possible when strategies are within ~1%.
- Networking assumptions
  - While CP works on TCP (Figure 6b), the best results rely on hiding communication under compute. Very low bandwidth or high jitter can expose SendRecv/All2All.
- Engineering complexity
  - Equal‑size message constraints require padding; fused variable‑length batching plus per‑step overlap and merge attention increase implementation complexity (Sections 3.5.2–3.5.3; Appendix B).

## 7. Implications and Future Directions
- How this changes the landscape
  - Makes million‑token exact attention practical on multi‑node clusters without exotic interconnects. For long‑context applications, system‑level CP becomes the default tool to reduce TTFT dramatically while keeping model architecture intact.

- Practical applications
  - Long‑document and video understanding (1M tokens ≈ ~1 hour of video; Section 4.2.3).
  - Enterprise assistants needing persistent, exact recall across multi‑turn sessions with large histories.
  - Batch processing of very large prompts (e.g., analytics over codebases or legal corpora) where TTFT dominates.

- Follow‑up research and engineering
  - Decouple prefill and decode placements in production serving (Section 4.3), e.g., disaggregated architectures that schedule prefill on CP‑rich pools and decode on comm‑lean or TP‑optimized pools.
  - Improve decode:
    - Remove padding by variable‑size collectives; better overlap of compute and All2All; compress partial outputs.
    - Explore hybrid `pass‑Q/KV` within a single batch to tailor per‑sequence choices.
  - Combine CP with algorithmic reductions:
    - Retrieval‑augmented or sparse attention for ultra‑long contexts to curb quadratic cost (Conclusion).
    - KV quantization and paged memory (Background 2.2) to raise capacity and throughput.
  - Adaptive runtime:
    - Online estimation of `C`, `BW`, and per‑batch `T`, `P` to choose between `pass‑KV` and `pass‑Q` using Algorithm 5; fall back to the empirical model in Appendix D for robustness.
  - Extend beyond text:
    - Apply CP ring attention to multimodal sequences (audio/video tokens) where long contexts are common.

Block‑quoted highlights (for quick reference):
> CP16 achieves 1M‑token exact prefill in 77 s and 128K in 3.8 s with Llama‑3 405B (Figure 8; Section 4.2.3).

> With 8 nodes on RDMA, 128K prefill completes in 5.85 s using pass‑KV (Figure 6a).

> CP scales better than multi‑node TP; at 8 nodes TP can be ~2× slower (Figure 7).

> For 128K partial prefill, pass‑Q is faster when KV miss rate < ~5%; pass‑KV dominates above that (Table 4; Figure 9; Table 5; Appendix C).

> Decode TTIT grows with CP due to ring SendRecv and All2All; CP is best leveraged for prefill, ideally on a serving stack that places prefill and decode differently (Section 4.3; Tables 6–8).
