# Efficient Memory Management for Large Language Model Serving with PagedAttention

**ArXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)

## 🎯 Pitch

This paper introduces PagedAttention, a novel attention algorithm inspired by operating systems, which enables key-value (KV) caches for large language models to be managed as non-contiguous fixed-size blocks rather than contiguous memory chunks. By dramatically reducing memory fragmentation and allowing KV cache sharing within and across requests, the authors’ vLLM system achieves up to 2–4× higher serving throughput at the same latency as leading solutions—a breakthrough that removes a major bottleneck in high-throughput, cost-effective deployment of large language models.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces PagedAttention and vLLM, an operating-system-inspired memory subsystem and serving engine that store attention key/value (“KV”) states in fixed-size pages (blocks) rather than contiguous tensors. By eliminating fragmentation and enabling sharing of KV pages across sequences, vLLM sustains 2–4× higher throughput at similar latency versus state-of-the-art systems across diverse models and workloads (Figures 12, 14, 16; Abstract).

## 2. Context and Motivation
- Problem addressed
  - Serving large language models (LLMs) efficiently requires batching many concurrent requests. The bottleneck is GPU memory, especially for the `KV cache`—the per-token key and value vectors used by attention during autoregressive generation (§2.2).
  - KV memory is large, grows/shrinks per request, and lengths are unknown in advance. Existing systems store each request’s KV cache in a contiguous chunk, causing (i) pre-reservation waste; (ii) internal fragmentation (unused space in over-provisioned chunks); and (iii) external fragmentation (allocator gaps) (§3.1, Figure 3).

- Why it matters
  - KV memory can consume >30% of GPU memory during serving (Figure 1 left) and scales linearly with tokens. For OPT-13B, a single token’s KV cache is ~800 KB and a 2048-token sequence can reach ~1.6 GB (§3), severely limiting batch size and throughput.
  - GPU compute has grown faster than memory capacity; memory will remain a scaling bottleneck (§3).

- Shortcomings of prior approaches
  - FasterTransformer: highly optimized kernels but no fine-grained iteration-level scheduling; uses contiguous allocations that over-reserve memory (Baseline 1 in §6.1).
  - Orca: iteration-level scheduler improves compute utilization but still allocates per-request contiguous chunks, incurring significant fragmentation and preventing KV sharing (§3.1, Baseline 2 in §6.1).
  - Empirical evidence: in existing systems, only 20.4–38.2% of KV memory holds useful token states; the rest is waste (Figure 2 and §3.1).

- Positioning
  - The paper reframes KV memory management through the lens of virtual memory and paging: divide KV into fixed-size blocks and map “logical” blocks of a sequence to “physical” blocks in GPU memory dynamically. This enables near-zero fragmentation and block-level sharing across sequences (§4.1–§4.3).

## 3. Technical Approach
The system has two core parts: the `PagedAttention` algorithm (how attention reads from paged KV memory) and the `vLLM` engine (block allocators, scheduler, sharing mechanisms, and distributed execution).

1) KV cache and generation basics (§2.2)
- Serving consists of:
  - Prompt phase: process the whole user prompt in parallel; compute K and V for prompt tokens and logits for the first generated token.
  - Autoregressive phase: at each step, generate one new token using the query of the latest token and all past K/V. KV states grow token-by-token.
- The KV cache stores past K/V so each new token can attend to all previous tokens without recomputing them.

2) PagedAttention: attention over paged KV memory (§4.1; Figure 5)
- Key idea: Store KV states in fixed-size `KV blocks` of `B` tokens, and allow these blocks to be non-contiguous in physical GPU memory.
- Attention is computed block-by-block. For a query vector q_i at position i:
  - Instead of multiplying q_i by all previous keys as one long vector, compute attention scores and weighted sums per block (Eq. 4). In notation:
    - Compute A_ij (the slice of attention weights for tokens inside block j up to i),
    - Then combine V_j A_ij^T across j to form the output o_i.
- Why it works: the attention math is unchanged; only the memory access pattern is restructured to page in blocks. This lets the kernel fetch discontiguous blocks and still compute correct attention.

3) KV Cache Manager: logical–physical block mapping (§4.2; Figure 6)
- GPU block engine: pre-partitions a chunk of GPU DRAM into equal-sized `physical KV blocks`.
- Block tables: one per sequence; map `logical` block indices (0,1,2,…) to physical block IDs and store a `#filled` count (how many token slots in the last logical block are occupied).
- Allocation strategy:
  - Allocate only the blocks needed so far. During prompt prefill, allocate enough blocks to hold the prompt’s tokens; in generation, allocate a new block only when the last one becomes full (§4.3).
  - This caps per-sequence waste to at most one partially filled block.

4) Decoding workflow with paged KV (§4.3; Figure 6–7)
- Prefill: compute prompt K/V; write them into logical blocks filled left-to-right. Map each logical block to any free physical block.
- Generation step t:
  - Read previous K/V via block table (PagedAttention kernel).
  - If the last block has space, append the new token’s K/V there; otherwise allocate a new physical block, update block table.
- Batching: each iteration, the scheduler batches currently active sequences, allocates any needed blocks, concatenates that iteration’s input tokens, runs the model, and writes new K/V into their mapped blocks (§4.3, last two paragraphs).

5) Memory sharing for advanced decoding (§4.4)
- Parallel sampling (multiple independent samples for one prompt):
  - Share prompt KV across samples by mapping their prompt logical blocks to the same physical blocks (Figure 8).
  - Use `copy-on-write` at block granularity: if a shared block needs modification (e.g., last prompt block when appending), allocate a new physical block, copy data, decrement reference count of the original, and write to the new block (Figure 8, “Ref count: 2 → 1”).
- Beam search (keep top-k evolving candidates):
  - KV sharing changes dynamically as beams split and merge. vLLM represents beams as sequences whose logical blocks often point to shared physical blocks.
  - When some beams are discarded, their references are dropped and physical blocks with zero refcount are freed; new beams get new blocks (Figure 9).
  - This avoids large block copies across beams; copy-on-write is needed only when generating inside a shared block.
- Shared prefix (system prompt or few-shot exemplars reused across requests):
  - Pre-reserve physical blocks for common prefixes (like shared libraries). A new request maps its initial logical blocks to these, then continues from there (Figure 10; §4.4 “Shared prefix”).

6) Scheduling, preemption, and recovery (§4.5)
- Scheduling policy: FCFS (first-come-first-served) for fairness; preempt the latest arrivals first under memory pressure.
- Eviction granularity: `all-or-nothing per sequence group` (e.g., all beams of a request together), because attention needs every block of a sequence. This fits paging semantics but exploits domain-specific “all blocks are used together.”
- Recovery choices when GPU blocks are insufficient:
  - `Swapping`: move entire sequence groups’ KV blocks to CPU RAM using a CPU block allocator; later swap back. Swap space is bounded by the GPU KV allocation (§4.5 “Swapping”).
  - `Recomputation`: when resuming, re-run the prompt phase on the concatenated prompt + already-generated tokens to rebuild KV, which is faster than originally computing token-by-token during generation (§4.5 “Recomputation”).
- System behavior: after preempting and evicting some sequences, the system stops accepting new requests until all preempted sequences finish (ensures space to bring them back) (§4.5).

7) Distributed execution (§4.6)
- Works with Megatron-LM style tensor parallelism (SPMD across attention heads) so every shard needs KV for the same token positions.
- The centralized scheduler maintains one block table per sequence that all GPU workers use. At each iteration, the scheduler broadcasts input tokens and the block tables; workers run attention using those mappings; all-reduce combines intermediate results; sampled tokens return to the scheduler (§4.6).

8) Implementation and kernel optimizations (§5)
- vLLM is implemented with PyTorch and custom CUDA kernels; NCCL for communication (§5).
- Optimizations (§5.1):
  - Fused reshape + block write: split new KV into blocks and write them to physical locations in one kernel.
  - Fused block read + attention: extend FasterTransformer’s attention kernel to read non-contiguous blocks based on block table; a warp reads one block; supports variable sequence lengths within the batch.
  - Fused block copy: batch many copy-on-write block copies into one kernel launch.

## 4. Key Insights and Innovations
- KV as virtual memory
  - Novelty: Treat a request’s KV states as a `paged` address space with fixed-size `blocks`; separate logical sequence order from physical placement (§4.1–§4.3).
  - Why significant: Eliminates external fragmentation entirely (all blocks same size), minimizes internal fragmentation (≤ one block per sequence), and avoids long-lived over-reservation (§3.1, Figures 2–3). This frees memory to batch more requests, directly boosting throughput (Figures 12–13).

- Block-level sharing with copy-on-write
  - Novelty: Combine block-level reference counting and copy-on-write to share large KV regions across sequences from the same request (parallel sampling, beam search) and across requests (shared prefixes) (§4.4, Figures 8–10).
  - Why significant: Reduces duplication of prompt KV, and in beam search reduces repeated memory copies when beams branch/prune. Savings reach 37.6–55.2% in beam search on Alpaca and 44.3–66.3% on ShareGPT (Figure 15; §6.3).

- Domain-aware preemption and recovery
  - Novelty: All-or-nothing eviction at the sequence-group level and two recovery strategies (`swap` or `recompute`) tailored to LLM generation semantics (§4.5).
  - Why significant: Keeps the system responsive under memory pressure while bounding swap space and enabling recomputation that can be faster than swapping, especially with small blocks (Figure 19).

- Practical distributed serving with a centralized KV manager
  - Novelty: One scheduler maintains the global logical→physical mapping; all tensor-parallel workers execute using the same block tables (§4.6).
  - Why significant: Makes the paging abstraction practical for multi-GPU models (e.g., OPT-175B on 8×A100-80GB in Table 1) without per-iteration synchronization beyond the broadcast of block tables.

## 5. Experimental Analysis
- Evaluation setup (§6.1)
  - Models: OPT-13B (1×A100-40GB), OPT-66B (4×A100-40GB), OPT-175B (8×A100-80GB). Table 1 lists parameter sizes and KV memory budgets; e.g., OPT-13B has 12 GB for KV cache and max ~15.7K KV slots.
  - Workloads: Synthetic traces from ShareGPT (long prompts and outputs) and Alpaca (shorter), with Poisson arrivals. Length distributions shown in Figure 11 (ShareGPT: input mean 161, output mean 338; Alpaca: input 19, output 58).
  - Baselines:
    - FasterTransformer (FT) with a dynamic batching scheduler (§6.1).
    - Orca in three provisioning modes: `Max` (reserve to 2048 tokens), `Pow2` (next power-of-two), and `Oracle` (true lengths, unrealizable in practice) (§6.1).
  - Metric: `normalized latency` = end-to-end latency divided by output length (s/token). Systems are compared by how high a request rate they sustain before latency explodes (1-hour traces; 15 minutes for OPT-175B).

- Main results
  - Throughput gains with basic sampling (Figure 12; §6.2)
    - On ShareGPT, vLLM sustains 1.7–2.7× higher request rates than Orca (Oracle) and 2.7–8× than Orca (Max), at similar normalized latency across OPT-13B/66B/175B (Figure 12a–c).
    - vLLM outperforms FT even more sharply (up to 22× higher request rates) because FT lacks iteration-level scheduling and suffers from memory over-reservation (§6.2).
    - On Alpaca (short sequences), vLLM still wins (Figure 12d–f), though with OPT-175B the advantage narrows because the setting becomes compute-bound (ample KV memory, short sequences, §6.2).
  - Batch size realized (Figure 13)
    - With OPT-13B on ShareGPT at 2 req/s, average concurrent batched requests: vLLM 30.42 vs Orca-Oracle 13.62, Orca-Pow2 9.81, Orca-Max 7.00 (Figure 13a).
    - On Alpaca at 30 req/s: vLLM 132.44 vs Orca-Oracle 72.75, Orca-Pow2 43.24, Orca-Max 7.00 (Figure 13b).
  - Memory utilization (Figure 2; §3.1)
    - Fragmentation analysis shows existing systems use only 20.4–38.2% of KV memory for actual token states, whereas vLLM achieves near-zero waste with 96.3% effective KV use (Figure 2).
    - Quote:
      > Only 20.4% – 38.2% of the KV cache memory is used to store actual token states in existing systems (§3.1, Figure 2), while vLLM reduces waste to near zero (96.3% usage).
  - Parallel sampling and beam search (Figure 14; §6.3)
    - As parallelism grows, vLLM’s advantage increases. With OPT-13B on Alpaca, normalized latency remains low at higher request rates than Orca across parallel sizes 2–6 and beam widths 2–6 (Figure 14a–f).
    - Measured KV sharing savings: 6.1–9.8% for parallel sampling and 37.6–55.2% for beam search on Alpaca; 16.2–30.5% and 44.3–66.3% respectively on ShareGPT (Figure 15; §6.3).
  - Shared prefix reuse (Figure 16; §6.4)
    - For WMT16 En–De with LLaMA-13B and shared translation exemplars: vLLM attains 1.67× (1-shot) and 3.58× (5-shot) higher throughput than Orca (Oracle).
  - Chatbot scenario (Figure 17; §6.5)
    - With long histories truncated to 1024 tokens in OPT-13B, vLLM sustains 2× higher request rates than all Orca variants; Orca variants behave similarly because buddy allocation reserves large output chunks regardless of predicted output length (§6.5).

- Ablations and diagnostics
  - Kernel microbenchmark (§7.1; Figure 18a):
    - PagedAttention kernels have 20–26% higher per-kernel latency than FT’s attention kernel due to block-table indirection and variable-length handling, but end-to-end wins arise from much larger batch sizes.
  - Block size trade-off (§7.2; Figure 18b):
    - Best performance typically at block sizes 16–128 (ShareGPT) and 16–32 (Alpaca). Default is 16 to balance GPU utilization and low fragmentation.
  - Swap vs recompute (§7.3; Figure 19):
    - Swapping suffers with small blocks due to many tiny PCIe transfers; recomputation cost is roughly constant across block sizes.
    - Quote:
      > Recomputation’s overhead is never higher than 20% of swapping’s latency; for block sizes 16–64, both are comparable (Figure 19).

- Do results support the claims?
  - Yes. Gains are shown across multiple model sizes, hardware scales, datasets, and decoding regimes. The mechanism–result link is clear: better KV memory utilization (Figure 2) translates into larger batches (Figure 13) and higher sustainable request rates (Figures 12, 14), especially in memory-bound regimes.

## 6. Limitations and Trade-offs
- Assumptions and scope
  - Benefits are strongest when serving is memory-bound. When compute dominates (e.g., large GPU memory and short sequences: OPT-175B on Alpaca), throughput gains narrow (§6.2, Figure 12f).
  - The design assumes all KV blocks of a sequence must be concurrently accessible; hence eviction is “all-or-nothing” at the sequence-group level (§4.5). This simplifies correctness but limits finer-grained memory reclamation.

- Overheads and engineering complexity
  - Attention kernel overhead: 20–26% slower microbenchmarks than FT’s kernel (Figure 18a). End-to-end gains offset this but the overhead is real (§7.1).
  - Block size tuning: Too small harms GPU utilization; too large increases fragmentation and reduces sharing opportunities (§7.2). vLLM defaults to 16, but workloads may benefit from tuning.
  - Copy-on-write operates at block granularity. If many writes happen inside shared last blocks, extra block copies occur; sub-block COW is not supported (§4.4).

- Scheduling and preemption behavior
  - FCFS prioritization may not meet latency SLOs for heterogeneous request profiles. Moreover, once sequences are preempted and swapped out, the system pauses accepting new requests until they finish (§4.5), which is a conservative choice that may reduce admission under extreme load.

- Generality
  - The OS-like paging analogy is tailored to LLM KV access patterns. The paper cautions that similar techniques may not help compute-bound workloads or those with static tensor shapes (e.g., conventional DNN serving/training), and indirection overhead could hurt (§8).

- Not addressed
  - Cross-model multi-tenant scheduling, SLA-aware batching, and heterogeneous accelerators are outside scope.
  - Energy or cost analysis is not included; CPU memory capacity limits for swapping are bounded by GPU KV allocation but still consume host resources (§4.5).

## 7. Implications and Future Directions
- How this changes the landscape
  - Treating KV cache as paged virtual memory is a conceptual shift: it disentangles sequence order from physical placement. This unlocks near-zero fragmentation, sharing across diverse decoding methods, and scalable batching on fixed-memory GPUs (Figures 2, 12–16).
  - Quote:
    > vLLM improves throughput by 2–4× with the same latency level compared to FasterTransformer and Orca (Abstract); improvements are more pronounced with longer sequences, larger models, and more complex decoding.

- Follow-up research
  - Adaptive block sizing: dynamically adjust block size per workload to optimize the utilization/fragmentation trade-off (§7.2).
  - SLA-aware scheduling: integrate deadlines or tail-latency constraints with the block allocator and preemption policies.
  - Predictive prefetching and admission control: use learned predictors of output lengths to pre-allocate blocks conservatively without hard over-reservation.
  - Deeper integration with optimized attention kernels (e.g., FlashAttention) that natively support paged access patterns.

- Practical applications
  - High-throughput chatbots, coding assistants, and translation services that need parallel sampling or beam search benefit from KV sharing (Figures 14–16).
  - Multi-tenant serving with shared system prompts or few-shot exemplars can cache and reuse large prefixes effectively (Figure 10, 16).
  - Cost-efficient deployment of very large models (e.g., 175B parameters) by increasing the number of concurrent users per GPU memory footprint (Table 1, Figures 12–13).

Overall, vLLM’s PagedAttention rethinks memory as the primary scaling lever for LLM serving. By borrowing mature ideas from OS (paging, copy-on-write) and adapting them to KV access patterns, it turns fragmented, over-reserved memory into a shared, elastic pool that directly converts into throughput.
