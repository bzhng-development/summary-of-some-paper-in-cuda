# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

**ArXiv:** [2501.01005](https://arxiv.org/abs/2501.01005)

## 🎯 Pitch

FlashInfer introduces a breakthrough attention engine for large language model (LLM) inference, unifying diverse KV-cache layouts under a block-sparse abstraction and enabling the just-in-time (JIT) compilation of custom attention variants into highly optimized CUDA/CUTLASS kernels. Its dynamic, load-balanced scheduler and compatibility with major LLM serving frameworks translate directly into dramatic real-world speedups—reducing inter-token latency by up to 69%, accelerating long-context streaming by 30%, and enabling scalable, low-latency AI deployments across heterogeneous workloads. This general, high-performance attention solution addresses fundamental bottlenecks in LLM serving, paving the way for faster, more flexible, and more efficient generative AI systems.

---

## 1. Executive Summary
FlashInfer is a code-generation–based attention engine that speeds up and generalizes large‑language‑model (LLM) inference. It unifies diverse key–value (KV) cache layouts under a block‑sparse abstraction, compiles custom attention variants just‑in‑time (JIT) to highly tuned CUDA/CUTLASS kernels, and schedules work adaptively with a deterministic, CUDA‑Graph–friendly runtime. Integrated into SGLang, vLLM, and MLC‑Engine, it reduces inter‑token latency by 29–69% versus a strong Triton backend, accelerates long‑context streaming by 28–30%, and speeds parallel generation by 13–17% (Section 4; Figure 7, Figure 9, Figure 10).

## 2. Context and Motivation
- Problem addressed
  - LLM serving hinges on fast attention over a growing, irregular KV cache. Real deployments must handle diverse traffic patterns (prefill, batched decoding, prefix reuse, speculative/tree decoding), heterogeneous storage formats (paged, radix trees, masks), and model‑specific attention variants (grouped heads, soft‑cap biases, sliding windows). Existing kernels often assume uniform sequence lengths and a single layout, leading to load imbalance, under‑utilization of GPU features, and high maintenance across systems (Section 1).
- Why it matters
  - On modern GPUs, attention is the dominant latency contributor for inference. Operational intensity (work per byte) in serving simplifies to O(lqo) where `lqo` is the query length (Section 2.1). If attention kernels under‑utilize compute/bandwidth or stall on load imbalance, end‑to‑end latency and throughput degrade, directly affecting user responsiveness and serving cost.
- Prior approaches and gaps
  - FlashAttention/FA2/FA3 deliver high performance for dense attention (Sections 2.1, 2.1 references), but do not by themselves address:
    - Heterogeneous KV storage across serving engines (PagedAttention, radix trees, speculative trees).
    - Dynamic, highly skewed sequence length distributions that cause SM under‑utilization.
    - Rapid proliferation of attention variants without rewriting low‑level kernels.
  - Systems work (PagedAttention, radix trees, prefix‑reusing decoders) solves memory management, but requires specialized kernels or sacrifices performance otherwise (Section 1; Section 5.4).
- Positioning
  - FlashInfer bridges systems and kernels: it (1) abstracts all KV layouts as block‑sparse matrices, (2) compiles variant‑specific kernels on demand, and (3) schedules irregular work deterministically while remaining compatible with CUDA Graphs (Figure 1; Sections 3.1–3.3). It builds on FA2/FA3 algorithms but extends them to sparse layouts, variable lengths, and custom variants.

## 3. Technical Approach
FlashInfer is a full stack: data layout, kernel templates, a JIT variant system, and a dynamism‑aware scheduler with a CUDA‑Graphs–compatible runtime.

1) Unified data layout: Block‑Sparse KV cache
- What is a KV cache?
  - In decoding, attention reads past keys/values (K/V) stored across time; this KV cache grows per request and becomes the chief memory footprint.
- Abstraction
  - FlashInfer represents KV storage as a Block‑Sparse Row (BSR) matrix with block size `(Br, Bc)` (Section 3.1.1). `Br` aligns with the query tile size; `Bc` is chosen by the KV manager. This single format can encode:
    - PagedAttention page tables (Figure 2).
    - Radix‑tree layouts.
    - Tree attention (for speculative decoding) and importance masks (Section 3.1.1).
- Why BSR?
  - It allows skipping empty regions (sparsity), reusing on‑chip memory for blocks, and mapping well to tensor cores when blocks are staged through shared memory (Sections 2.3, 3.2.1).
- Composable formats
  - A single fixed block size is a compromise. FlashInfer introduces “composable formats”: keep multiple BSR views with different `Br`/`Bc` for different sub‑regions (Section 3.1.2; Figure 3). Example:
    - Shared prefixes across several requests are kept in a BSR with larger `Br`, so multiple queries can reuse the same KV in fast shared memory/registers.
    - Unique suffixes are kept in a BSR with small `Br` to avoid fragmentation and preserve flexibility.
  - Implementation does not move data; it derives index arrays for the sub‑matrices (Section 3.1.2).

2) Kernel templates and data movement
- Dense core with sparse staging
  - Regardless of storage, computation uses dense tensor cores after KV tiles are gathered into contiguous shared memory. For sparse storage, threads compute addresses from the BSR `indices` and copy pieces into shared memory; for dense storage, they use affine indexing (Figure 4; Section 3.2.1).
- Asynchronous global→shared copies
  - Uses 128‑byte `LDGSTS` async copy to maximize bandwidth. On Hopper, Tensor Memory Accelerator (TMA) is used only for contiguous KV because TMA needs fixed‑stride access; sparse gathers fall back to Ampere‑style async copy (Section 3.2.1).
  - Result: after staging, both dense and sparse paths share the same FlashAttention compute loops.
- Microkernels and tile sizes
  - FA2‑based kernels expose many tile sizes: `(Tq in {1,16,32,64,128}) × (K/V tile in {32,64,128})` (Section 3.2.2). Heuristics:
    - Choose the smallest `Tq` ≥ average query length (with head‑group fusion for GQA; Appendix A).
    - Maximize SM occupancy under register/shared‑memory constraints (Section 3.2.2).
  - FA3 variants align row tiles with WGMMA (Hopper) multiples of 64 (Section 3.2.3).
- GQA head‑group fusion
  - For Grouped‑Query Attention, fuse the query‑head dimension into the row (sequence) dimension so that one KV load serves multiple query heads (Appendix A; Figure 11). This is particularly helpful when query lengths are short.

3) JIT compiler for attention variants
- Motivation
  - New attention forms (e.g., logits soft‑cap, sliding windows, RoPE fusion, sigmoid self‑attention) proliferate. Hand‑maintaining specialized kernels is brittle.
- Interface
  - Users provide a “specification” that defines small functors:
    - `QueryTransform`, `KeyTransform`, `ValueTransform` (pre‑attention transforms).
    - `LogitsTransform`, `LogitsMask` (pre‑softmax score shaping/masking).
    - `OutputTransform` (post‑attention transform) (Section 3.2.3; Figure 5).
  - The JIT replaces template hooks with these functors and compiles a CUDA operator (via PyTorch JIT) registered as a custom op; also exposes a DLPack API for other runtimes (Figure 5).
- Capability
  - Supports disabling softmax (e.g., FlashSigmoid), fusing RoPE/normalization/projection into the kernel to avoid extra memory traffic (Section 3.2.3; Figure 5).

4) Dynamism‑aware, deterministic scheduler and runtime
- Core idea
  - Split irregular work into “chunks” and assign them to CTAs to minimize idle SMs—without atomics—and keep a deterministic reduction order compatible with CUDA Graphs (Section 3.3.1).
- Attention composition
  - FlashInfer treats each partial computation as an “Attention State”: pair of output vector and its `log-sum-exp` scale. States compose associatively:
    - Equation (1) defines `LSE(I) = log(sum_{i∈I} exp(q·k_i))`
    - Equation (2) defines `O(I) = Σ exp(q·k_i)/exp(LSE(I))·v_i` (Section 2.2).
    - The operator `⊕` merges two states by weighted averaging the outputs and summing scales in log space; it is associative/commutative (Section 2.2). This enables chunked, order‑agnostic reduction.
- Scheduling algorithm (Algorithm 1; Section 3.3.1)
  - Inputs: per‑request `lqo(i)`, `lkv(i)` and query tile size `Tq`.
  - Define a cost model `cost(lq, lkv) = α·lq + β·lkv` (α, β hyperparameters).
  - Compute a max KV chunk size `Lkv` based on total KV and available CTAs; split each tile’s KV into chunks of ≤ `Lkv`.
  - Greedily assign the longest remaining chunk to the CTA with minimum accumulated cost (priority queue). This balances load while keeping a deterministic order.
- CUDA‑Graph compatibility
  - Use persistent kernels with fixed grid size and pre‑allocated workspace regions so all device pointers remain constant across steps (Figure 6; Section 3.3.1; Appendix D.1).
  - Plan/Run API: `plan()` computes the step’s schedule on CPU and writes it to a device workspace; `run()` executes kernels that read the plan. CUDA Graph captures `run()`, not `plan()` (Listing 1).
- Memory and workspace
  - Metadata is built on a pinned host buffer then copied asynchronously to the device workspace (Appendix D).
  - “Writethrough” optimization: if a request is not split, its result bypasses the workspace and writes directly to the final output (Appendix D.2).
  - Upper bound for partial outputs: at most `2 × #CTA × Tq × Hqo × (D + 1)` elements because each split produces up to two tiles per CTA, and outputs include `D` features plus one `LSE` (Appendix D.3).

5) Engineering notes and measured overheads
- Sparse gathering overhead vs dense:
  - Decode kernels: performance difference is within ~1%.
  - Causal prefill: sparse is ~10% slower because TMA cannot be used with non‑affine indices; sparse falls back to Ampere‑style async copies (Appendix B; Figure 12).

## 4. Key Insights and Innovations
- Unified KV‑cache as composable Block‑Sparse Rows
  - Novelty: uses BSR as a lingua franca for paged/radix/tree/masked layouts and allows multiple BSR “views” with different block sizes to exploit shared prefixes without moving data (Section 3.1.1–3.1.2; Figure 2, Figure 3).
  - Significance: enables one set of kernels to serve diverse serving backends; improves memory locality and reduces redundant global memory reads for shared‑prefix scenarios.
- JIT‑compiled attention variants with minimal user code
  - Novelty: variant functors injected into high‑performance CUDA/CUTLASS templates, supporting features like fused RoPE/normalization and softmax‑free attention (Section 3.2.3; Figure 5).
  - Significance: rapidly adapts to new attention forms while keeping tensor‑core–level performance. This is a capability upgrade beyond fixed‑function kernels.
- Deterministic, load‑balanced scheduler compatible with CUDA Graphs
  - Novelty: Stream‑K–inspired tiling without nondeterministic atomics; splits KV across CTAs, composes results via the `⊕` operator, and keeps grid and pointers static for graph replay (Algorithm 1; Figure 6; Section 3.3.1).
  - Significance: addresses real serving dynamics (variable/severely skewed lengths) and keeps the benefits of CUDA Graphs—critical for minimizing CPU overhead.
- Hardware‑aware microkernels spanning tiny to large tiles + vector‑sparse staging
  - Novelty: expands FA2/FA3 tile space and gathers arbitrary block sizes into shared memory to exploit dense tensor cores even for small `Bc` (Sections 3.2.1–3.2.2).
  - Significance: improves utilization in decode (short queries), supports fine‑grained KV sparsity, and maintains near‑dense performance for sparse decoding.

## 5. Experimental Analysis
- Setup
  - Hardware/Software: NVIDIA A100‑40GB SXM and H100‑80GB SXM; CUDA 12.4; PyTorch 2.4.0; fp16 compute/storage (Section 4).
  - Integrations: SGLang v0.3.4; vLLM; MLC‑Engine (Section 4; Section 3.4).
  - Workloads:
    - End‑to‑end online serving: ShareGPT and synthetic “Variable” workloads with uniformly random sequence lengths in [512, 2048]. Metrics: median inter‑token latency (ITL) and time‑to‑first‑token (TTFT); request rate tuned so P99 TTFT < 200 ms (Section 4.1; Figure 7).
    - Kernel microbenchmarks: bandwidth or FLOPs utilization for decode and causal prefill on constant, uniform, and skewed distributions; MHA and GQA variants (Section 4.2; Figure 8).
    - Long‑context Streaming‑LLM with fused RoPE (Section 4.3; Figure 9).
    - Parallel generation with composable formats (Section 4.4; Figure 10).
    - Additional comparisons: FlexAttention on AttentionGym (Appendix G.1; Tables 1–4), shared‑prefix latency (Appendix G.2; Table 5), ablations of the scheduler (Appendix G.3; Tables 6–7), vLLM integration (Appendix G.4; Table 8), and fine‑grained sparsity (Quest) benchmarks (Appendix G.5; Tables 9–11).

- End‑to‑end results (SGLang integration; Figure 7)
  - Llama‑3.1‑8B on 1×H100:
    - ITL: ShareGPT 21.7 ms → 13.5 ms; Variable 29.6 ms → 9.1 ms.
    - TTFT: ShareGPT 49.2 ms → 38.8 ms; Variable 61.8 ms → 53.2 ms.
  - Llama‑3.1‑70B on 4×H100:
    - ITL: ShareGPT 48.3 ms → 24.0 ms; Variable 30.7 ms → 21.8 ms.
    - TTFT: ShareGPT 141.2 ms → 115.6 ms; Variable 165.2 ms → 157.8 ms.
  - Takeaway: substantial median latency reductions across model sizes and workloads.

- Kernel‑level behavior under dynamics (Figure 8)
  - Decode (H100/A100): FlashInfer achieves higher memory bandwidth utilization than FlashAttention, especially on uniform and skewed lengths and with GQA. Improvements arise from the load‑balanced scheduler and broader tile choices (Sections 3.2.2, 3.3.1).
  - Prefill: higher achieved FLOPs utilization for causal prefill; FlashInfer sustains performance as lengths vary (Figure 8, bottom).

- Long‑context Streaming‑LLM with fused RoPE (Figure 9)
  - End‑to‑end ITL (H100): for recent window sizes 1k/2k/4k tokens:
    - 13.2/13.3/13.4 ms with FlashInfer fused RoPE vs 18.2/19.1/20.0 ms with unfused FA kernels and 26.4/26.7/29.7 ms for the original implementation.
  - Kernel‑level bandwidth: fused RoPE achieves 1.6–3.7× higher bandwidth utilization than the unfused alternative (Figure 9 bottom).
  - Interpretation: fusing RoPE into attention (via JIT functors) avoids extra memory passes and unlocks substantial speedups.

- Parallel generation with composable formats (Figure 10)
  - With prefix caching enabled in MLC‑Engine and varying parallel degree `n`:
    - Peak gains around `n = 4`: ITL reduced by 13.73% (8B) and 17.42% (70B); TTFT reduced by 16.41% (8B) and 22.86% (70B).
  - Gains persist for `4 ≤ n ≤ 32`; for very small `n`, blocks don’t get large enough to benefit; for very large `n`, attention ceases to dominate (Section 4.4).

- Additional evaluations
  - FlexAttention comparison on AttentionGym (Appendix G.1):
    - Causal attention TFLOPs/s at 16k tokens: 612 vs 454 (+35%); with soft‑cap or ALiBi or sliding windows, FlashInfer is consistently faster across lengths (Tables 1–4).
    - Reason: CUTLASS templates exploit Hopper features (warp specialization, TMA) and finer register‑level control than current Triton paths (Appendix C).
  - Ablation of load‑balancing scheduler (Appendix G.3):
    - For long inputs U(4096, 16384): ITL 8.63 ms (with LB) vs 13.89 ms (without); TTFT 411 ms vs 422 ms; both better than a Triton baseline (Tables 6–7).
  - vLLM integration (Appendix G.4):
    - With fp8 KV cache (e4m3), median ITL improves 12.56 → 10.92 ms; bf16 shows parity to slight regression due to host‑side overheads outside FlashInfer’s kernels (Table 8).
  - Fine‑grained sparsity (Quest) (Appendix G.5):
    - FlashInfer’s batch‑decode attention shows up to ~20× lower latency than PyTorch SDPA/FlexAttention at long lengths by using vector‑sparse staging with dense tensor cores (Tables 9–11).
  - Sparse vs dense overhead (Appendix B):
    - Decode: within ~1%; Prefill: sparse ~10% slower since Hopper TMA cannot be used for non‑affine loads (Figure 12).

- Overall assessment
  - The experiments are diverse (system‑level, kernel‑level, ablations) and tie gains to specific mechanisms:
    - Scheduler explains robustness on variable/skewed lengths.
    - JIT fusion explains Streaming‑LLM gains.
    - Composable formats explain parallel generation speedups.
    - CUTLASS/TMA usage explains FlexAttention deltas.
  - Results are convincing for NVIDIA GPUs in serving settings.

## 6. Limitations and Trade-offs
- Scope limitations
  - Only forward attention is supported; training/backward kernels are future work (Section 6).
  - Targeted at NVIDIA GPUs via CUDA/CUTLASS; Triton or non‑NVIDIA backends are not primary targets yet (Appendix C).
- Hardware constraints
  - Hopper’s TMA cannot load non‑affine sparse rows, so sparse causal prefill cannot benefit from TMA and is ~10% slower than dense (Appendix B). Very large `Bc` would enable TMA but reduces layout flexibility (Appendix B).
- Scheduling assumptions
  - The cost model `α, β` and chunk sizing are heuristic (Algorithm 1). While deterministic and effective in reported settings, other hardware/batch mixes might benefit from auto‑tuning.
- Runtime trade‑offs
  - Plan/Run split puts planning on the CPU each step; while amortized across layers and captured via CUDA Graphs for run, extreme step rates or host contention could surface planning overhead (Section 3.3.1; Listing 1).
  - CUDA Graph compatibility requires fixed persistent grids and pre‑allocated, over‑provisioned workspace segments; this increases memory reservation (Appendix D.1, D.3).
- Data‑layout complexity
  - Composable formats require maintaining multiple sparse views (index arrays). Benefits depend on prefix‑sharing prevalence; for small shared prefixes or tiny batches, gains can be neutral or negative (Figure 10 scatter points below the diagonal).
- JIT compilation
  - First‑use JIT adds cold‑start latency; mitigating with caching and graph capture is necessary (Section 3.4).

## 7. Implications and Future Directions
- Impact on the field
  - Establishes a practical bridge from heterogeneous serving memory managers to high‑performance attention via a single block‑sparse abstraction and composable formats. This lowers the cost of adding new attention variants and of supporting new serving patterns without sacrificing kernel efficiency.
- What this enables
  - Rapid prototyping of attention ideas (e.g., soft‑cap, sliding windows, sigmoid attention, fused RoPE/normalization) in production contexts, because a few lines of variant code yield optimized kernels (Figure 5; Section 4.3).
  - Robust latency under real traffic: the deterministic load‑balanced scheduler can be a template for other irregular GPU operators that must remain CUDA‑Graph compatible (Algorithm 1; Figure 6).
  - Efficient long‑context and prefix‑heavy workloads: composable formats and attention state composition (`⊕`) cleanly leverage shared prefixes in tree/speculative decoding and parallel generation (Sections 2.2, 3.1.2, 4.4).
- Promising research directions
  - Backward pass templates and training support (Section 6).
  - Extending JIT to higher‑level DSLs (e.g., FlexAttention‑style frontends) and additional backends (AMD/MLIR/NVDSL/ThunderKittens) as noted in Section 7.
  - On‑device scheduling (reducing CPU planning), auto‑tuning of chunking and tile sizes, and integration with asynchronous store‑reduce techniques (e.g., FlashDecoding++) for even less orchestration overhead (Section 5.1 discussion).
  - Exploring TMA‑friendly sparse layouts (larger `Bc`) or hybrid loaders that choose between TMA and gathers adaptively (Appendix B).
- Practical applications
  - Production LLM serving platforms (chat, code assistants, agents) that must keep latency low under bursty, multi‑tenant traffic.
  - Long‑context applications (retrieval‑augmented generation with large histories, streaming dialogue) where fused kernels and KV sparsity matter.
  - Parallel sampling or multi‑branch decoding in agents and tool‑use systems, where shared prefixes are common and composable formats shine (Figure 10).

> Headline results: “FlashInfer achieve 29–69% inter‑token‑latency reduction compared to compiler backends for LLM serving benchmark, 28–30% latency reduction for long‑context inference, and 13–17% speedup for LLM serving with parallel generation” (Abstract; Section 4; Figures 7, 9, 10).

> Mechanistic evidence: sparse‑staging with dense tensor cores keeps decode overhead within ~1% vs dense (Appendix B; Figure 12), while the scheduler ablation shows large gains on skewed/long lengths (Appendix G.3; Tables 6–7), validating the core design choices.
