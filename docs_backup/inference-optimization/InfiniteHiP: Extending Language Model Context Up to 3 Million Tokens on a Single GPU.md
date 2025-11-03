# InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

**ArXiv:** [2502.08910](https://arxiv.org/abs/2502.08910)
**Authors:** Heejun Lee, Geon Park, Jaduk Suh, Sung Ju Hwang
**Institutions:** KAIST, DeepAuto.ai

## 🎯 Pitch

InfiniteHiP revolutionizes Transformer LLM inference by enabling million-token processing on a single 48-GB GPU without retraining, using hierarchical pruning, efficient memory offloading, and adaptive RoPE embeddings. This breakthrough drastically lowers the computational barrier, facilitating real-world applications in fields requiring long-context processing, such as legal and scientific domains, and sets a new benchmark for speed and scalability in natural language processing.

---

## 1. Executive Summary
InfiniteHiP is a training‑free inference framework that lets off‑the‑shelf Transformer LLMs process extremely long inputs—up to 3 million tokens—on a single 48‑GB GPU, while keeping quality and speed practical. It does this by (a) hierarchically pruning irrelevant tokens before attention, (b) offloading most of the key‑value (KV) cache to CPU memory with an aggressive GPU cache, and (c) carefully adjusting positional embeddings to generalize beyond the model’s original context window.

## 2. Context and Motivation
- The problem addressed
  - Long inputs are increasingly common (RAG, multi‑document reasoning, code bases, logs). Standard attention has quadratic time and linear memory growth w.r.t. sequence length, and pre‑trained LLMs often fail beyond their training context limit because their positional encoding does not extrapolate reliably.
  - This paper tackles three linked challenges (Sec. 1):
    - Computation: attention over millions of tokens is too slow.
    - Memory: KV caches grow linearly and quickly exceed GPU VRAM.
    - Out‑of‑length (OOL) generalization: pre‑trained models typically break when context > pre‑training window due to positional embedding limits (e.g., RoPE).

- Why this matters
  - Practical: Enables million‑token applications on a single commodity GPU (e.g., L40s 48 GB) without cutting content or retraining (Fig. 1, Secs. 1, 5.2).
  - Scientific: Shows that training‑free, structure‑aware pruning plus tailored RoPE handling can preserve quality far beyond pre‑trained limits (Secs. 4, 5.2).

- Prior approaches and gaps
  - FlashAttention v2 (FA2) reduces memory traffic but not computation; still quadratic (Sec. 1).
  - KV eviction schemes (e.g., H2O) save VRAM by “forgetting” tokens permanently—unsafe for tasks that need distant tokens later (Sec. 1).
  - HiP Attention (2024) offloads “cold” KV to CPU and fetches on demand, but its pruning relies on iterative heuristics with global synchronizations, limiting parallelism and speed (Sec. 1, 2).
  - InfLLM selects fixed representative tokens per block, not query‑adaptive; sacrifices precision to avoid CPU accesses within its kernel when offloading (Sec. 2, 5.3).

- Positioning
  - InfiniteHiP merges three threads—faster sparse attention, robust offloading, and OOL RoPE tuning—into one inference framework (Fig. 1). It improves on HiP with more accurate/faster modular pruning (Sec. 1, Fig. 2b), stronger cache policy (LRU), and layer‑aware RoPE strategies (Secs. 4, B, D; Table 5; Table 6).

## 3. Technical Approach
At a high level (Fig. 1), InfiniteHiP:
1) builds a sparse attention mask that keeps only the few key/value blocks likely to matter for each group of queries;
2) performs “paged block sparse attention” over those blocks;
3) stores most KV in CPU memory and uses a GPU cache for hot tokens; and
4) modifies RoPE indexing on the fly so the model remains stable far beyond its trained context.

Key components and how they work:

- Modular hierarchical context pruning (Secs. 3–4; Fig. 2b; Appendix A, Algs. 1–3)
  - Intuition: “Chunk sparsity.” In long sequences, the keys that matter concentrate in a few local chunks. Evidence in Fig. 2a:
    - Left: in a 128K context, <2% of chunks contain >12.5% of top‑2K keys.
    - Right: with 64‑token chunks, ~75% of chunks contain none of the top‑2K keys.
  - Setup:
    - The sequence of keys is partitioned into fixed‑size chunks of `l_c` tokens.
    - Queries are processed in blocks of size `b_q` for better GPU parallelism.
    - Some tokens are always kept without pruning: `sink` tokens (first `n_sink`, to preserve global anchors) and `streaming` tokens (most recent `n_stream`, to preserve recency).
  - Each pruning stage `S^(i) = (b_q^(i), l_c^(i), k^(i))` does:
    1) Divide the candidate key indices into chunks of size `l_c^(i)` (Appendix A, Alg. 2, lines 4–5).
    2) For each chunk, pick a representative token using a hierarchical top‑1 search that repeatedly splits the chunk and compares two candidates (Appendix A, Alg. 3). This takes O(log l_c) comparisons and accesses at most two tokens per chunk.
    3) Score each chunk by the maximum dot‑product between any query in the block and the representative key, max‑pooled across heads (Alg. 2, line 11).
    4) Keep the top `K^(i) = k^(i)/l_c^(i)` chunks and discard the rest (Eq. 2), propagating only the surviving tokens to the next stage (Eq. 1).
  - After N stages, the kept key indices `I_m^(N)` form a block‑sparse attention mask for the m‑th query block (Eqs. 1–3). The mask always includes the `sink` and `streaming` regions.

- Representative token selection (Appendix A, Alg. 3)
  - Purpose: approximate the index of the maximum attention score within a chunk without evaluating all keys.
  - Mechanism: a binary search over the chunk; at each step, select two candidates (left/right), apply RoPE to each, score them against all queries in the block, choose the better side, and recurse (Alg. 3, lines 5–16).
  - Benefit: few memory reads (at most two keys per chunk), high parallelism, implemented as a single kernel without global synchronization.

- Complexity and execution
  - The first pruning stage touches the full candidate range and is O(T_q·T_kv); subsequent stages are O(T_q) because they operate on much smaller surviving sets (Appendix A). In practice this is fast because the kernels are highly parallel, and stage‑1 can be cached for many decode steps (mask refresh; next bullet).
  - Sparse mask caching during decoding (Sec. 4; Appendix A, Alg. 4; Fig. 7): recompute stage i only every `n_refresh^(i)` steps. Later stages update more frequently (e.g., defaults 16/8/4 steps for stages 1/2/3; Appendix F), amortizing the expensive first stage.

- RoPE strategies for OOL generalization (Sec. 4; Appendix B–D; Table 5)
  - OOL problem: pre‑trained RoPE indices break beyond the original context. InfiniteHiP uses different RoPE “indexing styles” in different places:
    - During pruning:
      - “Chunk‑indexed RoPE”: give every key within a chunk the same position ID (the chunk ID, offset from the current query by `n_stream`). Used in layers 1–3 to emulate sliding‑window locality that these shallow layers tend to rely on (Appendix D; Fig. 10; Table 6).
      - “Relative‑style RoPE”: in the hierarchical selection, apply two nearby position offsets to left vs right candidates (Appendix A, Eqs. 4–5; Appendix B Fig. 8). Used from layer 4 onward to emphasize content over absolute distance.
    - During the final sparse attention:
      - “StreamingLLM‑style RoPE”: assign positions so the most recent selected key shares the position ID with the current query, producing a stable sliding effect (Sec. 4; Xiao et al., 2024b).
  - Empirical choice: Table 5 shows the best En.MC accuracy on ∞Bench when pruning uses Relative or InfLLM‑style and the sparse attention uses StreamingLLM‑style. The paper adopts Relative (pruning) + Streaming (attention).

- KV cache offloading with GPU cache (Sec. 4; Table 4; Fig. 1a)
  - Memory model: a unified address space (NVIDIA UVM) where most K/V live in CPU DRAM (e.g., 2 TB). The GPU holds a smaller “key bank” cache (and a separate bank specifically for sparse attention) (Sec. 4 under KV Cache Offloading).
  - On a cache miss during pruning or sparse attention, the kernel triggers UVM to fetch the needed K/V from CPU, and an LRU policy evicts cold entries from the GPU cache to make room (Sec. 4; Appendix A, Alg. 4, lines 12–13).
  - Pagination and graph capture: combines PagedAttention to organize KV pages (Sec. 4), and designs the kernel so the entire offloaded attention remains graph‑capturable—reducing CPU overhead compared to InfLLM (Sec. 5.3).

- Implementation notes (Sec. 4)
  - Triton kernels for all pruning stages (single parametric kernel).
  - Flash‑style kernels for block‑sparse attention (FlashAttention for prefill; FlashDecoding for decode), plus PagedAttention for KV management.
  - Integrated into SGLang runtime; experiments use AWQ Llama‑3.1 8B with FP8 KV cache for throughput studies (Tables 4, 11, 12).

## 4. Key Insights and Innovations
- Modular, query‑adaptive hierarchical pruning (Fig. 2b; Appendix A)
  - Novelty: replaces HiP’s iterative, synchronization‑heavy top‑k selection with per‑stage, highly parallel top‑1‑per‑chunk selection plus chunk scoring. This preserves query adaptivity (unlike InfLLM’s fixed representatives) and runs as a single GPU kernel per stage.
  - Why it matters: Higher recall of the true high‑attention tokens with lower compute (Fig. 6a shows +1.57 percentage points recall over InfLLM and +4.72 over HiP), enabling accurate yet fast sparse attention.

- Stage‑wise mask caching with configurable refresh (Appendix A, Alg. 4; Fig. 7)
  - Novelty: caches the sparse mask output at each stage and refreshes at different frequencies (e.g., 16/8/4). The first stage (linear in sequence length) can be run infrequently because attention patterns have temporal locality during decoding.
  - Impact: Large decoding speedups at long contexts (Table 3), with small quality loss (Table 2 shows similar or better scores even with “fast/flash” refresh schedules).

- Layer‑aware, mixed RoPE strategies for stable OOL generalization (Appendix B–D; Tables 5–6)
  - Novelty: use chunk‑indexed RoPE in the first 3 layers (to preserve sliding‑window behavior those layers naturally learn; Fig. 10), switch to relative‑style RoPE later, and use StreamingLLM‑style during sparse attention.
  - Evidence: Table 6 shows En.MC accuracy improves from 68.55% (Relative in all layers) to 74.23% when mixing Chunk‑indexed in layers 1–3 with Relative afterwards at 300K tokens—despite no training.

- Practical KV offloading that still preserves precision (Sec. 4; Table 4)
  - Novelty: allow kernels to access CPU memory mid‑computation under UVM and pair it with LRU caching. InfLLM avoids CPU memory within kernels and therefore downgrades precision (needs bigger windows to compensate).
  - Impact: With offloading, InfiniteHiP attains faster decode at large contexts than InfLLM (e.g., at 256K, 325 μs vs 1186 μs per token in Table 4 “Runtime (Flash)” vs InfLLM), while maintaining strong downstream quality (Tables 1–2).

## 5. Experimental Analysis
- Setup (Sec. 5.1; Appendix F)
  - Models: Llama‑3/3.1 8B Instruct, Mistral 0.2 7B Instruct; additional OOL tests on EXAONE‑3/3.5 7.8B and Gemma‑2 9B (Fig. 4; Table 10).
  - Benchmarks:
    - LongBench (avg ~32K context; Table 1).
    - ∞Bench (100K+ contexts; Table 2).
    - RULER (appendix; Tables 8–9).
    - Passkey retrieval stress test (Appendix E.1, Table 7).
  - Baselines: FA2 (dense, truncate), Dynamic‑NTK, Self‑Extend (RoPE scaling), LM‑Infinite, StreamingLLM (sink+streaming with RoPE adjustments), H2O (KV eviction), InfLLM (training‑free long context with representatives), HiP (hierarchical pruning + offload).
  - Metrics: task‑specific scores; “Avg. Rel.” is the mean of each subset’s score normalized by that column’s maximum (Tables 1–2). Latency measured in ms (prefill) and μs (decode) at various lengths (Table 3). With offloading: per‑token decode latency and VRAM usage (Table 4). Throughput in tokens/s on RTX4090 and L40s (Fig. 5; Tables 11–12).

- Main results
  - Quality at long contexts without training
    - LongBench (Table 1):
      - Llama‑3 8B: Avg. Rel. 100.00 with InfiniteHiP vs 92.83 with InfLLM; raw examples include strong gains in Few‑shot Learning “PC” (93.5 vs 84.0) and “RBP” (64.8 vs 46.5).
      - Mistral‑0.2 7B: Avg. Rel. 99.85 (InfiniteHiP) vs 96.99 (InfLLM 12K window).
    - ∞Bench (Table 2, longer contexts):
      - Llama‑3 8B: Avg. Rel. 47.08–47.21 (depending on window) with InfiniteHiP vs 43.05 (InfLLM).
      - Mistral‑0.2 7B: Avg. Rel. up to 54.96 (InfiniteHiP 5K) vs 54.02 (InfLLM 16K), despite InfiniteHiP attending 4× fewer key tokens than InfLLM (Sec. 5.2).
    - OOL generalization curves (Fig. 3): Llama‑3.1 8B En.MC accuracy increases as context grows past 128K with InfiniteHiP, whereas methods without OOL adaptation degrade.
    - Short‑context models extended (Fig. 4; Table 10): On Gemma‑2 9B, InfiniteHiP gains +24.45 points in En.MC accuracy and +22.03 points in En.QA Recall vs FA2 when extending beyond its 8K context.
    - RULER (Appendix Tables 8–9): InfiniteHiP variants substantially outperform FA2 and HiP at 128K–512K, e.g., average 16K‑shallow setting reaches 78.89 vs FA2’s 66.89.
    - Passkey (Appendix Table 7): On DeepSeek‑R1 distilled Qwen2‑14B with pre‑training window 128K, InfiniteHiP retrieves the passkey at up to 1M tokens with 98–100% accuracy across insertion positions.

  - Speed and memory
    - Attention latency (no offload; Table 3):
      - Decoding at 1M tokens (T=1024k): 234 μs/token (InfiniteHiP) vs 4,645 (FA2: 19.85× slower), 1,222 (InfLLM: 5.2× slower), and 450 (HiP: ~1.9× slower).
      - Prefill at 1M: 172 ms (InfiniteHiP) vs 3,490 ms (FA2: ~20.3× slower).
    - With KV offloading (Table 4, RTX4090 PCIe 4.0 x8; AWQ Llama‑3.1 FP8 KV):
      - T=256K: Runtime “Flash” 325 μs/token for InfiniteHiP vs 1,186 for InfLLM (~3.65× faster) at 6.1 GB VRAM.
      - T=1,024K: 844 μs/token (InfiniteHiP Flash) vs 1,864 (InfLLM), both using ~6.1 GB VRAM; mask and sparse‑attention hit ratios rise as more stages are cached.
      - Important note: CPU memory access latency is ~31.5× higher than VRAM (Sec. 5.3), so offloading makes attention extremely memory‑bound; the pruning stages plus stage caching are critical to reducing those accesses.
    - End‑to‑end throughput (Fig. 5; Tables 11–12):
      - RTX4090 24GB, 1M tokens: 40.1 tok/s (InfiniteHiP Offload‑Flash) vs 12.5 tok/s (SRT‑estimated) → 3.2×.
      - L40s 48GB, 3M tokens: 23.8 tok/s (InfiniteHiP Offload‑Flash) vs 3.3 tok/s (SRT‑estimated) → 7.25×.

- Ablations and diagnostics
  - Top‑k coverage (Fig. 6a): Higher recall of true attention mass than HiP and InfLLM across top‑k thresholds.
  - Number of pruning stages (Fig. 6b): ∞Bench En.MC rises from 70.31 (N=2) to 74.24 (N=3), exceeding FA2’s 67.25 at 128K.
  - RoPE combinations (Table 5): Best En.MC when pruning uses Relative (RT) or InfLLM (IL) style and sparse attention uses Streaming (ST); the paper selects RT (pruning) + ST (attention).
  - Layer‑wise RoPE mix (Table 6): Using Chunk‑indexed in the first 3 layers plus Relative afterwards outperforms Relative‑only by ~5.7 points on En.MC at 300K.

- Do the experiments support the claims?
  - Yes, across quality benchmarks (LongBench, ∞Bench, RULER), InfiniteHiP either matches or exceeds the strongest training‑free baselines while attending to fewer tokens; latency and throughput measurements demonstrate large gains at 1M+ contexts; offloading results show practicality on single‑GPU systems with limited VRAM (Tables 3–4; Fig. 5).

## 6. Limitations and Trade-offs
- Computational structure
  - The first pruning stage is O(T_q·T_kv) (Appendix A); although amortized via mask caching (Appendix A, Fig. 7), the prefill stage for million‑token prompts is still long on consumer hardware (Appendix G).
  - Overall performance depends on “chunk sparsity”—the assumption that high‑attention keys cluster in a few chunks (Sec. 3, Fig. 2a). Sequences with uniform relevance could reduce pruning effectiveness.

- Memory and hardware constraints
  - Offloading shifts the bottleneck to PCIe bandwidth/latency; on slower interconnects, the advantage shrinks (Sec. 5.3; Table 4). The method relies on NVIDIA UVM and well‑tuned kernel implementations (Sec. 4).
  - Although “infinite” in principle, you are practically limited by CPU RAM (often ≤2 TB in commodity servers) and by the overhead of frequent CPU‑GPU transfers (Appendix G).

- Algorithmic choices and sensitivity
  - Quality/speed depends on hyperparameters: chunk sizes `l_c`, tokens kept `k^(i)`, query block size `b_q`, and mask refresh intervals `n_refresh` (Appendix F). Different tasks may prefer different presets (Appendix G).
  - RoPE strategy is carefully engineered per layer and per stage. Suboptimal combinations degrade quality (Table 5 shows many weaker pairings). This increases configuration complexity.

- Scope
  - The work targets inference; it does not address training stability for long contexts. It also does not propose new attention learning mechanisms—rather, it engineers a fast, training‑free runtime.

## 7. Implications and Future Directions
- How it changes the landscape
  - Demonstrates that million‑token contexts on a single 48‑GB GPU are feasible without retraining, by combining query‑adaptive pruning, offloading, and layer‑aware RoPE control. This lowers the barrier for long‑context applications (legal, biomedical, code analysis, logs, multi‑doc QA) and for serving LLMs with very long user histories.

- Follow‑up research it enables
  - Smarter module configurations: task‑adaptive stage depths, chunk sizes, and refresh schedules (Appendix G).
  - Better offloading: KV quantization and compression integrated with the same pruning/offload pipeline (Appendix G cites KVQuant, KV compression).
  - Faster time‑to‑first‑token: integrate speculative attention/memory “lazy” initialization (Appendix G); combine with retrieval to further reduce active working sets.
  - Extending to multi‑GPU and faster interconnects (NVLink/PCIe Gen5), and to MoE models where expert routing interacts with pruning.

- Practical applications
  - Single‑GPU serving of very long documents for enterprise search and compliance, codebase understanding, session‑level assistants that keep entire multi‑day histories, and scientific workflows over huge logs or corpora.
  - As a drop‑in runtime (SGLang implementation) that improves both speed and maximum usable context on existing models (Sec. 1, Sec. 4, Fig. 5).

> Key takeaway: InfiniteHiP’s combination of modular hierarchical pruning (Fig. 2b; Algs. 1–3), per‑stage mask caching (Appendix A, Fig. 7), layer‑aware RoPE strategies (Appendix B–D; Tables 5–6), and practical KV offloading (Table 4) yields the first training‑free, single‑GPU pipeline that processes up to 3M tokens end‑to‑end with strong quality and large speedups (Tables 1–3; Fig. 5).
