# Efficient Memory Management for Large Language Model Serving with PagedAttention

**ArXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

## 🎯 Pitch

This paper introduces PagedAttention, a new attention algorithm and memory management strategy for LLM serving, inspired by operating system virtual memory and paging. By breaking up large, contiguous KV cache allocations into small, flexible 'pages' that can be dynamically allocated, shared, and copy-on-written, the vLLM system nearly eliminates memory waste and dramatically increases the number of requests that can be batched—enabling 2–4× higher throughput compared to prior systems at the same latency. This breakthrough directly tackles the growing memory bottleneck in LLM serving, making deploying and scaling large models vastly more cost-effective and capable.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces PagedAttention and the vLLM serving engine, a rethinking of how large language models (LLMs) store and access their attention “key-value” (KV) cache during generation. By replacing large, contiguous KV tensors with small, fixed-size “pages” that can be allocated, shared, and copied on demand—much like an operating system’s virtual memory—vLLM reduces memory waste to near zero and enables 2–4× higher serving throughput at similar latency compared to state-of-the-art systems (Figures 1, 2, 12; §6).

## 2. Context and Motivation
- Problem addressed
  - Serving LLMs efficiently is limited by GPU memory devoted to the `KV cache`—the stored “key” and “value” vectors that let the model attend to previously seen tokens during generation (§2.2). This cache is large, grows token-by-token, and its final size per request is unknown beforehand.
  - Existing systems allocate the KV cache as big, contiguous tensors per request and “reserve” space up to a maximum length, which causes fragmentation and prevents sharing (§3.1, Figure 3).

- Why this matters
  - Real systems are memory-bound: for a 13B-parameter model on an NVIDIA A100 40GB, ~65% of memory holds weights while nearly 30% is for KV cache (Figure 1, left). Inefficient KV management limits how many requests can be batched together, directly reducing throughput and raising cost per request (§1).
  - GPU compute capacity is growing faster than memory capacity (e.g., A100→H100 FLOPS >2×, but max memory still 80GB), so memory bottlenecks will worsen (§3).

- Where prior approaches fall short
  - FasterTransformer provides high-performance kernels but no fine-grained batching or KV memory virtualization; batch size is capped by static pre-reservation (§6.1).
  - Orca enables iteration-level scheduling (add/remove requests each decoding step) but still stores each request’s KV cache in contiguous chunks and must over-reserve space, causing internal and external fragmentation (§3.1). In experiments, effective KV usage for such systems can be as low as 20–38% (Figure 2).

- Positioning
  - The paper reframes KV memory as a virtual, paged address space: map many small logical “KV blocks” to physical blocks that need not be contiguous or pre-reserved (§4.1–4.3). This eliminates most fragmentation, enables sharing across sequences (parallel sampling, beam search), and supports preemption with swap/recompute strategies (§4.4–4.5). It complements prior scheduling work by making more requests fit in memory per step (§9).

## 3. Technical Approach
The system has two parts: the PagedAttention kernel (how attention reads/writes non-contiguous KV) and the vLLM memory/scheduling runtime (how KV blocks are allocated, shared, and moved).

- Background needed (brief)
  - KV cache: during the prompt (“prefill”) phase, the model computes key/value vectors for all prompt tokens; during the generation phase it appends one token at a time, reusing the cached KV for attention (§2.2). The cost/latency of generation is dominated by reading these stored KV vectors.

- Core idea: KV as paged memory
  - Replace one large, contiguous KV tensor per sequence with many fixed-size “KV blocks” of size `B` tokens (default B=16; §7.2). Each block stores keys/values for a small run of positions (across all attention heads/layers; a practical design stores per-head/layer blocks; §4.1 footnote).
  - Maintain a `block table` per sequence that maps `logical` block indices (0,1,2,…) to `physical` block IDs and tracks how many token slots in the last block are filled (§4.2, Figure 6).
  - Allocate a fresh physical block only when the last logical block becomes full; only the last block is partially filled. This limits per-sequence unused space to at most one block (§4.3).

- PagedAttention kernel (how queries read non-contiguous KV)
  - Instead of multiplying a query with one large, contiguous matrix of keys/values, the kernel iterates over logical blocks and fetches their physical locations via the block table (Figure 5).
  - Formulation (Eq. 4 in §4.1): attention on token i is computed in block chunks: compute scores against keys in each block, normalize over all blocks up to ⌈i/B⌉, then form the output by weighting and summing the value blocks. In essence, matrix-vector attention is decomposed into several block multiplications that are stitched together.
  - GPU optimizations (§5.1): fused reshape+write for KV block output, fused block-read with attention, and a fused “block copy” kernel (for copy-on-write) to minimize kernel launch and small-copy overheads. Coalesced access is maintained by assigning one GPU warp per block.

- KV cache manager and scheduler (§4.2–§4.3; Figure 4)
  - A centralized scheduler coordinates distributed GPU workers (Megatron-LM style tensor-parallel execution; §4.6).
  - A `block engine` on each worker holds a large pre-allocated slab of GPU memory sliced into fixed-size physical blocks; the scheduler manages logical–physical mappings via block tables that are broadcast each iteration (§4.6).
  - Iteration flow (Figure 6): during prefill, allocate only the blocks needed to hold the prompt tokens; during each generation step, either append into the last partially filled block or allocate exactly one new physical block. New blocks are assigned only when needed and only for the sequences actually active in that step.

- Memory sharing—two major patterns (§4.4)
  - Parallel sampling (“N best-of samples from the same prompt”):
    - Sequences share the prompt’s blocks; a per-block `reference count` tracks how many sequences point to the same physical block (Figure 8).
    - If a shared block must be modified (e.g., writing the next token into the last partially filled prompt block), vLLM performs `copy-on-write` (COW) at block granularity: allocate a new block, copy old contents, decrement the old block’s refcount, then write. This is identical to OS page COW but at token-block granularity.
  - Beam search:
    - Candidates form a tree that frequently share long prefixes. vLLM maintains sharing at block granularity across the beam; when a candidate drops from the top-k, its logical blocks are freed and any physical blocks whose refcount hits zero are released (Figure 9).
    - This avoids the “copy the whole prefix” behavior of conventional systems; only when writing inside a shared (non-final) block does vLLM need to copy a single block (COW).

- Shared prefix caching: popular “system prompts” (e.g., instructions and examples) can be materialized once and reused across requests, much like shared libraries across processes; only user-specific tail prompts are computed and stored anew (§4.4, Figure 10).

- Scheduling and preemption (§4.5)
  - Policy: first-come-first-serve (FCFS) to avoid starvation.
  - Eviction granularity: `all-or-nothing` per “sequence group” (e.g., the N beams in one beam search request are gang-scheduled). Since all blocks for a sequence are needed together, partial eviction offers no benefit (§4.5).
  - Recovery options when GPU blocks are exhausted:
    - `Swapping`: evicted KV blocks are copied to CPU RAM; vLLM includes a CPU block allocator. Swap space is bounded by KV memory size on GPU (§4.5).
    - `Recomputation`: when resumed, regenerate the KV cache by treating generated tokens so far as an extended prompt; this is faster than the original because it uses the batched, parallel “prefill” computation (§4.5).
  - Which to use? Microbenchmarks show recomputation cost is stable across block sizes, while swapping is inefficient for small blocks due to many tiny PCIe transfers; for medium block sizes (16–64) both are comparable (Figure 19; §7.3).

- Distributed execution (§4.6)
  - Tensor-parallel SPMD execution with all-reduce; the same block tables are broadcast to all workers each step. Each worker stores KV only for its subset of heads. No per-step synchronization is needed for memory management beyond the initial broadcast (Figure 4; §4.6).

- Implementation notes (§5)
  - 8.5K lines Python + 2K lines C++/CUDA; supports GPT/OPT/LLaMA; uses NCCL for communication and a FastAPI frontend with an OpenAI-compatible API.
  - Despite added indirection, the attention kernel overhead is modest: 20–26% slower per-kernel than FasterTransformer, yet overall throughput is much higher due to larger batch sizes enabled by paging (Figure 18a; §7.1).

## 4. Key Insights and Innovations
- KV cache virtualization for LLM serving (fundamental)
  - What’s new: Treat KV memory like OS virtual memory—paged, non-contiguous, mapped by a block table; attention runs on “pages” (KV blocks) instead of monolithic tensors (§4.1–4.3).
  - Why it matters: It eliminates both internal and external fragmentation (Figure 3), reducing waste to roughly one block per sequence and raising effective KV usage from ~20–38% in prior designs to ~96% (Figure 2). This directly increases feasible batch size and throughput (Figures 12–13).

- Block-granular sharing with copy-on-write (fundamental)
  - What’s new: Seamless sharing of long common prefixes across samples/beams and across requests (for shared system prompts), with automatic COW only when a shared block must be modified (§4.4; Figures 8–10).
  - Why it matters: Sharing reduces KV memory footprints by 6–10% for parallel sampling and 38–55% for beam search on Alpaca; even larger on ShareGPT (16–30% and 44–66%) (Figure 15; §6.3). It also removes expensive bulk prefix copies that older systems perform in beam search (§4.4).

- Preemption with swapping or recomputation (practical innovation)
  - What’s new: When memory is tight, evict whole sequence groups and either swap blocks to CPU RAM or recompute later (§4.5).
  - Why it matters: This prevents deadlock under unknown and varying output lengths. The paper shows regimes where recomputation is preferable (small blocks) and where swapping is competitive (larger blocks) (Figure 19; §7.3).

- Kernel fusion for paged attention (incremental but necessary)
  - What’s new: Fused kernels for block reshaping/writing, block-reading-with-attention, and bulk block copying (§5.1).
  - Why it matters: It offsets the cost of indirection and small-copy overhead, making the paged design viable on GPUs.

## 5. Experimental Analysis
- Setup (§6.1; Table 1; Figure 11)
  - Models: OPT-13B (1×A100-40GB), OPT-66B (4×A100), OPT-175B (8×A100-80GB); LLaMA-13B for a translation test. Table 1 details parameter sizes and available KV memory (e.g., 13B: 26GB params, 12GB KV).
  - Workloads: synthetic traces built from real conversations/instructions:
    - ShareGPT (long prompts, long outputs; mean input 161 tokens, output 338; Figure 11a).
    - Alpaca (shorter; mean input 19 tokens, output 58; Figure 11b).
  - Arrivals: Poisson with varying request rates. Metric: `normalized latency` (mean latency per request divided by its output length, s/token). This captures throughput saturation (curves “spike” when the system’s capacity is exceeded; Figures 12, 14, 16, 17).
  - Baselines:
    - FasterTransformer with a dynamic batching scheduler (max batch constrained by memory).
    - Orca variants: `Oracle` (knows the true output length—upper bound), `Pow2` (reserves up to next power-of-2), `Max` (reserves up to model max length). All use contiguous KV and buddy allocation (§6.1).

- Main results
  - Effective KV memory utilization
    - Quote (Figure 2): vLLM reaches “96.3%” KV usage vs “38.2%” (Orca-Oracle), “26.8%” (Orca-Pow2), and “20.4%” (Orca-Max). The remaining portions are reservation/internal/external fragmentation.
  - Basic sampling throughput (Figure 12; §6.2; Figure 13)
    - On ShareGPT, vLLM sustains “1.7×–2.7×” higher request rates than Orca-Oracle and “2.7×–8×” than Orca-Max at similar latency. It sustains up to “22×” higher rates than FasterTransformer, whose batch size is severely limited by KV reservation.
    - Example at OPT-13B: vLLM batches “30.42” concurrent requests vs “13.62” (Orca-Oracle) and “7.00” (Orca-Max) (Figure 13a).
    - On Alpaca (short sequences), the advantage vs Orca is smaller for OPT-175B because the setup has abundant KV memory, making the workload compute-bound rather than memory-bound (§6.2; Figure 12f; Figure 13b).
  - Parallel sampling and beam search (Figure 14; §6.3)
    - As parallel size or beam width grows, vLLM’s advantage increases because more sharing becomes possible. On OPT-13B/Alpaca, improvement over Orca-Oracle grows from “1.3×” (basic sampling) to “2.3×” (beam width 6) (§6.3).
    - Memory savings from sharing: “6.1–9.8%” for parallel sampling and “37.6–55.2%” for beam search on Alpaca; on ShareGPT, “16.2–30.5%” and “44.3–66.3%” (Figure 15).
  - Shared prefix caching (Figure 16; §6.4)
    - English→German translation with LLaMA-13B:
      - With a 1-shot prefix (80 tokens), vLLM delivers “1.67×” higher throughput vs Orca-Oracle (Figure 16a).
      - With a 5-shot prefix (341 tokens), the gain increases to “3.58×” (Figure 16b).
  - Chatbot setting (Figure 17; §6.5)
    - With long conversation history (truncated to last 1024 tokens), all Orca variants must reserve 1024 tokens for outputs due to buddy allocation; vLLM still pages. vLLM sustains roughly “2×” higher request rates than all Orca variants at similar latency.
  - Ablations and microbenchmarks (§7)
    - Kernel overhead: PagedAttention’s attention kernel is “20–26%” slower than FasterTransformer’s per call (Figure 18a), yet end-to-end is faster due to larger batches.
    - Block size trade-off: best performance at block sizes “16–128” on ShareGPT and “16–32” on Alpaca; too large increases internal fragmentation, too small underutilizes GPU (Figure 18b). Default is 16 (§7.2).
    - Swap vs recompute: swapping is inefficient at small blocks due to many small PCIe copies; recomputation cost is flat across block sizes and is never more than “20%” slower than swapping when swapping is favorable (Figure 19; §7.3).

- Overall assessment
  - The evaluation mixes challenging (ShareGPT) and easier (Alpaca) traces, three model sizes including multi-GPU settings, and diverse decoding methods. The consistent throughput gains, the sharp increase in batched requests (Figure 13), and the measured KV usage (Figure 2) collectively support the central claim: paging KV enables substantially higher throughput by fitting more concurrent sequences in memory.
  - The ablations convincingly address concerns about kernel overhead and design choices (block size, recovery method).

## 6. Limitations and Trade-offs
- Indirection overhead and kernel complexity
  - Attention kernels must follow block tables and handle non-contiguous memory, adding branching and address computations. This costs “20–26%” extra kernel time (Figure 18a). vLLM compensates with larger batches, but per-op latency is higher.
- Block size tuning
  - Too small: poorer GPU utilization and more metadata; too large: more internal fragmentation and fewer sharing opportunities (Figure 18b). A fixed block size (default 16) may be suboptimal for very short or very long prompts.
- Preemption granularity
  - All-or-nothing eviction of sequence groups simplifies design but may be coarse; evicting an entire group to free memory can cause head-of-line blocking for those requests (§4.5).
- Swapping constraints
  - Swap performance depends on CPU–GPU bandwidth and suffers at small block sizes (Figure 19a). Systems without ample CPU RAM or with limited interconnects may prefer recomputation, which increases compute load (§7.3).
- When the workload is compute-bound
  - If KV memory is abundant (e.g., OPT-175B on many GPUs + short Alpaca sequences), benefits shrink since throughput is limited by arithmetic rather than memory (Figure 12f; §6.2).
- Scope
  - The work targets inference-time serving for autoregressive Transformers. It does not cover training, non-autoregressive architectures, or model-specific tricks like KV quantization/compression; nor does it introduce QoS-aware scheduling beyond FCFS.

## 7. Implications and Future Directions
- How this changes the landscape
  - Treating the KV cache as a virtual, shareable, copy-on-write address space reframes LLM serving as a memory-systems problem. This unlocks larger effective batch sizes, enables mixing decoding strategies in one batch (§4.4), and reduces the per-request cost of serving long-context applications (Figure 17).

- Practical applications
  - Hosted LLM APIs and chat systems can cut cost and increase capacity without changing models; multi-sample code assistants and beam-search-heavy tasks (e.g., translation) benefit disproportionately due to sharing (Figures 14–16).
  - Shared-prefix caching can accelerate enterprise deployments where prompts share instruction templates (§6.4).

- Follow-up research directions
  - Adaptive paging policies: dynamic block sizes, smarter eviction than all-or-nothing, or cost-aware swap vs recompute decisions based on live bandwidth/compute measurements (§7.3).
  - KV compression and quantization within pages: combine paging with lossy or lossless KV compression to further increase capacity.
  - Hardware support: NIC/GPU/DRAM features for page-table-assisted gathers, block-level DMA, or KV-aware caches.
  - Multi-tenant schedulers: integrate paging with SLA-aware or priority scheduling instead of FCFS, exploring fairness/latency trade-offs.
  - Extending to other sequence models and long-context techniques (e.g., retrieval-augmented generation): page KV alongside external memory indices.
  - Distributed paging across nodes: remote KV “paging” over high-speed interconnects to pool memory across machines.

In short, this work contributes a robust system design—paging the KV cache with OS-style mechanisms—that materially improves the throughput of LLM serving, especially for long contexts and complex decoding, while introducing a clean abstraction layer that future techniques (compression, smarter scheduling, hardware help) can build on.
