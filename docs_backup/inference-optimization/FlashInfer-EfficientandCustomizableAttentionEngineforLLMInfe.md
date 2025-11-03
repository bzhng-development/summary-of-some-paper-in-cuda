# FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

**ArXiv:** [2501.01005](https://arxiv.org/abs/2501.01005)

## 🎯 Pitch

FlashInfer introduces a high-performance attention engine for large language model (LLM) inference, leveraging a unified block-sparse KV-cache format, just-in-time compilation for custom attention variants, and a dynamic, load-balanced scheduler. By addressing workload heterogeneity and memory layout complexity, FlashInfer significantly reduces inference latency and boosts throughput in real-world LLM serving scenarios, enabling more scalable, flexible, and efficient deployment of advanced language models across diverse applications.

---

## 1. Executive Summary (2-3 sentences)
FlashInfer is a code-generated attention engine that makes large-language-model (LLM) inference faster and more adaptable by unifying memory layouts for the key–value cache (`KV‑cache`), compiling variant attention behaviors on the fly, and dynamically load-balancing variable-length requests. It reduces end-to-end latency in real serving systems and improves kernel-level efficiency across diverse scenarios like long contexts and parallel generation (Section 4; Figures 7–10).

## 2. Context and Motivation
- Problem addressed:
  - LLM serving spends much of its time in attention, which must read a `KV‑cache` populated during earlier tokens. Serving loads are heterogeneous: requests arrive with different lengths, share prefixes, or branch into trees (speculative decoding). Modern systems also store the `KV‑cache` in non-contiguous formats (page tables, radix trees) and increasingly apply custom attention variants (e.g., grouped query attention, masking, soft-cap logits) (Section 1).
  - Two main challenges:
    1) Workload dynamism and imbalance within a batch (prefill vs. decode; mixed sequence lengths), which wastes GPU compute if not scheduled well.
    2) Hardware-aware customization: efficient kernels need to match GPU architecture details and the memory layout of the `KV‑cache` (Section 1).

- Why it matters:
  - Latency and throughput are the dominant costs in LLM serving. Achieving high performance across diverse request patterns and model features (e.g., long context, speculative decoding, parallel outputs) has direct operational and user experience impact (TTFT and inter-token latency, Section 4.1, Figure 7).

- Prior approaches and their limitations:
  - FlashAttention/2/3 improve dense attention algorithmics and pipelines but assume contiguous `KV‑cache` and fixed tiling (Section 2.1).
  - PagedAttention and radix-tree managers optimize memory but leave kernels to adapt to non-contiguity (Section 3.1.1).
  - System frameworks (e.g., SGLang, vLLM) integrate specialized kernels per scenario, leading to maintenance cost and gaps when combining features (Section 1).
  - When sequences vary in length, naïve kernel launches create load imbalance; Stream-K-like schemes can help but may lose determinism if they rely on atomics (Section 3.3.1).

- How this work positions itself:
  - FlashInfer is a generalized attention engine that:
    - Represents diverse `KV‑cache` layouts by a block-sparse matrix abstraction and optional composition of multiple formats (Sections 3.1.1–3.1.2; Figures 2–3).
    - Generates high-performance CUDA/CUTLASS kernels tailored to attention variants via a JIT specification (Section 3.2.3; Figure 5).
    - Schedules variable-length work dynamically while staying compatible with `CUDAGraph` static-capture requirements (Section 3.3; Algorithm 1; Figure 6).

## 3. Technical Approach
FlashInfer spans memory layout, compute templates, and runtime scheduling. The key pieces work together as follows.

- Unified `KV‑cache` representation with block-sparse matrices (Section 3.1.1; Figure 2)
  - Idea: Treat the `KV‑cache` as a Block Sparse Row (`BSR`) matrix. A BSR groups non-zeros into rectangular blocks of size `(Br, Bc)` instead of single scattered entries.
    - `Br` (block rows) aligns with the query tile size executed by a thread block.
    - `Bc` (block columns) aligns with `KV‑cache` page allocation granularity.
  - Why: Page tables and radix trees store non-contiguous `(H, D)` chunks (heads × hidden-dim) per token; in BSR, these become non-zero blocks. This unifies indexing and enables skipping empty regions efficiently (Section 3.1.1).
  - Inputs/outputs are stored as ragged (jagged) tensors—contiguous per sequence but without padding across sequences—reducing memory and making packing efficient (Section 3.1.1).

- Composable formats for shared prefixes and memory efficiency (Section 3.1.2; Figure 3)
  - Problem: One fixed block size is a trade-off—large `Br` improves reuse in shared memory but increases fragmentation; small `Br` avoids fragmentation but loses reuse.
  - Mechanism: Decompose the global sparse matrix into multiple submatrices (“composable formats”), each with its own block size.
    - Example (Figure 3): Store the shared prefix (dense across a group of requests) with a larger `Br` so multiple queries can reuse the same `K/V` in shared memory; store the unique suffix with a small `Br` to avoid waste.
  - Benefit: No data movement; only index arrays are computed for each submatrix. Attention kernels for larger blocks reuse `K/V` in fast on-chip memory, increasing throughput.

- Compute templates with dense tensor cores over sparse inputs (Sections 3.2, 3.2.1, 3.2.2)
  - Loading from global to shared memory:
    - Challenge: Block sizes may not match tensor-core tile shapes, especially when `Bc` is small (vector-sparse).
    - Mechanism (Figure 4): Gather scattered global memory rows/columns into contiguous shared memory tiles, then apply dense tensor-core MMA to those tiles.
    - Implementation details:
      - Keep the last dimension (`head_dim`) contiguous for coalesced loads; use 128B `LDGSTS` asynchronous copies (Section 3.2.1).
      - On Hopper, use `TMA` (Tensor Memory Accelerator) only for contiguous `K/V` because it requires affine access; sparse gathers fall back to Ampere-style async copies (Section 3.2.1).
  - Tile-size variants and heuristics (Section 3.2.2):
    - Provide row tile sizes `{1,16,32,64,128}` × K/V tile sizes `{32,64,128}` for FA2; for FA3 (Hopper), row tiles are multiples of 64 (WGMMA).
    - Heuristic selection:
      1) Estimate average query length in the batch, pick the smallest row tile ≥ that length.
      2) Respect register/shared-memory constraints to keep SM occupancy high.
    - Special case: Row tile size `1` uses CUDA cores; larger tiles use tensor cores (Section 3.2.3).
  - Head-group fusion for GQA (Appendix A; Figure 11):
    - For short queries, fuse the “query-head” dimension into the row dimension so one `K/V` load in shared memory serves all heads in the group, improving reuse.

- JIT compiler for attention variants (Section 3.2.3; Figure 5)
  - Need: Many models customize attention (e.g., soft-cap logits, sliding windows, sigmoid attention, fused RoPE).
  - Mechanism: Users provide a small CUDA functor class with optional methods:
    - `QueryTransform`, `KeyTransform`, `ValueTransform` (e.g., fuse RoPE/projection/normalization).
    - `LogitsTransform`, `LogitsMask` (e.g., apply masks, soft-cap).
    - `OutputTransform`.
    - A flag `use_softmax` enables variants that replace softmax (e.g., FlashSigmoid).
  - The template inserts these functors into a pre-optimized FlashAttention skeleton (FA2/FA3), compiles them with PyTorch JIT, and registers a custom op (Figure 5). A framework-agnostic DLPack interface is also available.

- Dynamism-aware scheduling with deterministic reduction (Section 3.3; Algorithm 1; Figure 6)
  - Background: In serving, per-request `KV` lengths vary each step; naïve tiling creates SM underutilization. Also, `CUDAGraph` requires fixed kernel launch parameters and pointer addresses.
  - Key design:
    - Split long `KV` segments into “chunks,” compute a per-chunk cost `α*lq + β*lkv` (user-tunable), and greedily assign the longest chunks to the least-loaded CTAs using a priority queue (Algorithm 1).
    - The attention kernel writes “partial” outputs for each chunk; a separate contraction step merges them using the mathematically exact attention composition operator:
      - Define the Attention State as `(O(I), LSE(I))` where `LSE` is log-sum-exp over logits and `O` is the normalized weighted sum of values (Equations (1)–(2), Section 2.2).
      - Reduce partial states with an associative, commutative operator `⊕` that composes `(O, LSE)` pairs deterministically (Section 2.2).
  - `CUDAGraph` compatibility (Section 3.3): Use persistent kernels with fixed grid sizes and a pre-allocated workspace whose sections have fixed addresses; pass plan metadata (work queues and reduction maps) via a device buffer whose pointer does not change across replays (Figure 6; Appendix D.1). Planning happens on CPU each step (inspector–executor pattern; Listing 1), but the same plan can be reused across layers.

- User-facing API (Section 3.4; Listing 1)
  - Two-phase usage:
    1) `plan(seqlen_info)` on CPU builds the balanced schedule and writes plan metadata into the workspace buffer.
    2) `run(...)` executes the persistent kernel (captured inside a `CUDAGraph`).
  - Multiple `CUDAGraph`s can be pre-captured for different average query lengths or composable-format settings; the runtime selects the best graph per step.

## 4. Key Insights and Innovations
- Block-sparse as a unifying lens for `KV‑cache` heterogeneity (Section 3.1.1; Figure 2)
  - Innovation: Represent page tables, radix trees, masked KV reuse, and speculative decoding trees as a single `BSR` abstraction with arbitrary `(Br, Bc)`.
  - Why it matters: One kernel family can operate across many storage backends, reducing engineering duplication and enabling predictable performance.

- Composable sparse formats to exploit shared prefixes (Section 3.1.2; Figure 3)
  - Innovation: Decompose the `KV‑cache` into multiple sparse submatrices, each with a block size chosen for its structure (e.g., large `Br` for dense shared prefixes, small `Br` for unique suffixes).
  - Payoff: Real speedups in parallel generation and prefix-caching scenarios without moving data, only retargeting indices (Section 4.4; Figure 10; Appendix G.2, Table 5).

- Variant-agnostic attention via JIT functors (Section 3.2.3; Figure 5)
  - Innovation: A small set of plug-in functors covers many attention variants, including those that change logits or remove softmax. RoPE and other transforms can be fused into the kernel.
  - Payoff: Enables model-specific kernels (e.g., Streaming-LLM fused RoPE) with minimal code, yielding 28–30% lower end-to-end latency versus unfused baselines (Section 4.3; Figure 9 top).

- Dynamism-aware, deterministic scheduler compatible with `CUDAGraph` (Section 3.3; Algorithm 1; Figure 6)
  - Innovation: A priority-queue assignment of variable-length chunks to CTAs plus deterministic on-device reduction over Attention States `(O, LSE)`.
  - Payoff: High utilization on variable-length batches while preserving reproducibility and enabling graph capture (Section 4.2; Figure 8; Appendix G.3 Tables 6–7).

- Practical sparse-to-dense tensor-core execution (Section 3.2.1)
  - Innovation: Gather sparse tiles into shared memory and drive dense tensor-core MMAs, even for vector-sparse cases (`Bc=1`).
  - Trade-off awareness: On Hopper, TMA accelerates dense loads but not non-affine sparse gathers; decode overhead is negligible, prefill overhead is ~10% for sparse vs. dense (Appendix B; Figure 12).

## 5. Experimental Analysis
- Setup and metrics (Section 4):
  - Hardware: NVIDIA A100 40GB SXM and H100 80GB SXM, CUDA 12.4, PyTorch 2.4.0; FP16 compute (Section 4).
  - Serving integration: SGLang v0.3.4; comparisons to SGLang with Triton v3.0 attention backend (Section 4.1).
  - Workloads: ShareGPT and a synthetic “Variable” set with input lengths uniformly in [512, 2048]; control P99 TTFT < 200 ms (Section 4.1).
  - Metrics: Time-to-first-token (TTFT) and inter-token latency (ITL) end-to-end; kernel-level bandwidth and FLOPs utilization; specialized scenarios for Streaming-LLM and parallel generation (Sections 4.2–4.4).

- End-to-end serving gains (Figure 7):
  - Llama 3.1 8B on 1×H100:
    - ITL:
      > ShareGPT: 21.7 ms (Triton) → 13.5 ms (FlashInfer)  
      > Variable: 29.6 ms (Triton) → 9.1 ms (FlashInfer)
    - TTFT:
      > ShareGPT: 49.2 ms (Triton) → 38.8 ms (FlashInfer)  
      > Variable: 61.8 ms (Triton) → 53.2 ms (FlashInfer)
  - Llama 3.1 70B on 4×H100:
    - ITL:
      > ShareGPT: 48.3 ms (Triton) → 24.0 ms (FlashInfer)  
      > Variable: 30.7 ms (Triton) → 21.8 ms (FlashInfer)
    - TTFT:
      > ShareGPT: 141.2 ms (Triton) → 115.6 ms (FlashInfer)  
      > Variable: 165.2 ms (Triton) → 157.8 ms (FlashInfer)
  - Takeaway: Consistent reductions in both TTFT and ITL.

- Kernel-level efficiency under variable lengths (Figure 8):
  - Decode (bandwidth utilization, higher is better, H100):
    - Constant-1024: `MHA` 73% (FlashInfer) vs 65% (FlashAttention); `GQA-4` 73% vs 70%; `GQA-8` 58% vs 53%.
    - Uniform [512–1024]: `MHA` 43% vs 43%; `GQA-4` 52% vs 43%; `GQA-8` 36% vs 35%.
    - Skewed (Zipf, avg 1024): `MHA` 32% vs 29%; `GQA-4` 39% vs 32%; `GQA-8` 28% vs 29%.
  - Prefill (FLOPs utilization, H100):
    > Example: Uniform distribution shows a marked advantage for FlashInfer (Figure 8 bottom-left; e.g., 48% vs 37% for MHA).  
  - Interpretation: Gains on non-uniform/imbalanced batches match the scheduler’s goal (Section 3.3.1). Some configurations are comparable; overall trend favors FlashInfer especially with group heads.

- Customization: Streaming-LLM with fused RoPE (Figure 9)
  - End-to-end ITL (Vicuna-13B on MT-Bench):
    - H100:  
      > 13.2–13.4 ms (FlashInfer fused RoPE) vs 18.2–20.0 ms (unfused) vs 26.4–29.7 ms (original)
    - A100:  
      > 24.2–24.5 ms (fused) vs 33.5–34.7 ms (unfused) vs 42.1–43.5 ms (original)
  - Kernel bandwidth:
    > Fused RoPE achieves 1.6–3.7× higher bandwidth than unfused (Figure 9 bottom).
  - Conclusion: Fusing query/key transforms via JIT materially lowers latency (Section 3.2.3).

- Parallel generation with composable formats (Figure 10)
  - With prefix-caching in MLC-Engine:
    > Peak speedups at n=4: ITL −13.73% (8B), −17.42% (70B); TTFT −16.41% (8B), −22.86% (70B).  
    > Benefits persist for 4 ≤ n ≤ 32; diminish for n=1–2 (limited shared work) and plateau for very large n (attention no longer dominates).

- Additional evaluations and ablations (Appendix G)
  - vs. FlexAttention on AttentionGym (Tables 1–4):
    > Causal L=4096: 548 TFLOPs/s (FlashInfer) vs 421 (Flex, Table 1).  
    > ALiBi L=4096: 561 vs 426 (Table 3).  
    > Sliding window L=1024: 374 vs 292 (Table 4).  
    > Consistent advantages, especially at longer lengths; attributed to Hopper features (warp specialization, TMA) and fine-grained resource control (Appendix G.1).
  - Shared-prefix kernel latency (Table 5):
    > With 32k shared prefix and batch size 64: 254.54 µs (composable) vs 4090 µs (single format) — strong gains at long prefixes and larger batches (Appendix G.2).
  - Load-balancing ablation within SGLang (Tables 6–7):
    > Variable U(4096, 16384), RR=1: ITL 8.63 ms with scheduling vs 13.89 ms without; TTFT 411.02 ms vs 421.60 ms (and both beat Triton on ITL/TTFT) — showing the scheduler’s importance (Appendix G.3).
  - vLLM integration (Table 8):
    > With FP8 `KV‑cache` (e4m3): ITL 12.56 ms → 10.92 ms (−13%); BF16 shows parity to slight regression due to host-side overheads (Appendix G.4).
  - Fine-grained sparsity (Quest) (Tables 9–11):
    > For long sequences, FlashInfer is up to ~20× faster than PyTorch SDPA and >10× vs FlexAttention at small block sizes (vector-sparse), due to the sparse-gather + dense-tensor-core strategy (Appendix G.5).

- Do the results support the claims?
  - Yes. End-to-end wins in multiple serving stacks (SGLang, MLC-Engine), strong kernel metrics under variable lengths, substantial gains from fused variants and composable formats, and targeted ablations tying improvements to key design elements (Figures 7–10; Tables 1–11; Algorithm 1; Figure 6).

## 6. Limitations and Trade-offs
- Hardware specificity and features (Sections 3.2.1, Appendix C):
  - Uses CUDA/CUTLASS templates tuned for NVIDIA GPUs (sm75–sm90a). Some Hopper accelerators (`TMA`) cannot handle non-affine sparse gathers; decode overhead is negligible but prefill on sparse layouts sees ~10% penalty vs dense (Appendix B, Figure 12).
  - Triton backend is not used; portability to non-NVIDIA platforms is future work (Appendix C).

- Static capture requirements (Section 3.3; Appendix D):
  - `CUDAGraph` compatibility requires persistent kernels, fixed grid sizes, and preallocated workspace with fixed addresses. Users must provision a workspace sized to upper bounds (Appendix D.1, D.3).

- Planning overhead on CPU (Section 3.3; Listing 1):
  - The per-step `plan` runs on CPU. While amortized over many layers and small in practice, it is a dependency between steps; moving it on-device is listed as a future optimization (Appendix G.4 commentary).

- Forward-only support (Section 6):
  - Training/backward-pass kernels are not included; extending the JIT scheme to backward variants is future work.

- Heuristic tiling and scheduling hyperparameters (Section 3.2.2; Algorithm 1):
  - Tile-size selection and the cost function `α, β` are heuristic. While effective in reported settings, optimality is not guaranteed across all models/GPUs.

- Benefits depend on structure:
  - Composable formats deliver their best speedups when shared prefixes are sufficiently long and batches large (Appendix G.2, Table 5; Figure 10). For small `n` in parallel generation or very short shared regions, gains are modest.

## 7. Implications and Future Directions
- How this work changes the landscape:
  - Establishes a single attention engine that spans diverse `KV‑cache` layouts, attention variants, and variable-length serving, with competitive or superior speed to specialized kernels. This reduces integration friction across serving frameworks (Figure 1; Section 3.4) and encourages experimentation with new attention behaviors without sacrificing performance (Section 3.2.3).

- Follow-up research enabled or suggested:
  - Backward-pass codegen for training; automatic differentiation over the attention-spec functors.
  - Richer scheduling: extend Algorithm 1 for multi-GPU overlap with GEMM/communication (Appendix E), exploit TMA for larger sparse blocks, or adopt store-reduce strategies (e.g., FlashDecoding++/TMA Reduce) while retaining determinism (Related Work, Section 5.1; Section 6).
  - Higher-level DSL frontends that compile to FlashInfer specs (Section 7; references to Mirage, FlexAttention, ThunderKittens), and backends beyond CUDA/CUTLASS (Section 7).

- Practical applications and downstream use:
  - Production LLM serving across:
    - Latency-sensitive online serving (lower ITL/TTFT, Figure 7).
    - Long-context inference with fused RoPE or other transforms (Figure 9).
    - Parallel generation and agents with prefix-caching (Figure 10).
    - Sparse or pruned `KV‑cache` regimes (Appendix G.5).
  - System integration already demonstrated in SGLang, vLLM, and MLC-Engine with measured gains (Sections 4.1, 4.4; Appendix G.4).

Overall, FlashInfer’s combination of a unified `KV‑cache` abstraction (block-sparse + composition), JIT-compiled attention variants, and a deterministic, graph-friendly scheduler delivers both generality and performance. The empirical evidence (Figures 7–10; Tables 1–11) shows that these design choices translate into lower latency and better utilization across the main real-world bottlenecks in LLM serving.
