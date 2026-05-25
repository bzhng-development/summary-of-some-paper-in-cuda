# ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition

**ArXiv:** [2402.15220](https://arxiv.org/abs/2402.15220)

## 🎯 Pitch

ChunkAttention introduces a novel prefix-aware self-attention mechanism that drastically improves inference efficiency for large language models by detecting and sharing redundant key/value tensors among requests that have overlapping prompt prefixes. This approach not only minimizes memory usage and eliminates redundant computation in multi-tenant LLM serving, but also transforms the self-attention kernel with a two-phase partition algorithm that exploits shared structure for optimal data locality. The result is a 3.2–4.8× speedup over state-of-the-art solutions for long shared prompts—delivering significant throughput gains and enabling more scalable, cost-effective LLM deployments as context lengths grow.

---

## 1. Executive Summary
ChunkAttention introduces a prefix-aware way to store and compute self-attention for large language model (LLM) inference by sharing key/value (KV) tensors among requests that have the same prompt prefix and by restructuring the attention kernel to exploit this sharing. It matters because multi-tenant LLM deployments frequently reuse long system prompts; by removing redundancy and improving data locality, ChunkAttention reduces latency and memory use, achieving 3.2–4.8× faster self-attention kernels for shared prompts of 1K–4K tokens (Table 3, Figure 3).

## 2. Context and Motivation
- The specific gap
  - During decoding, self-attention repeatedly reads the cached keys and values for all prior tokens (the `KV cache`). This step is memory-bound—limited by memory bandwidth rather than computation—and therefore slow, especially for long contexts (Table 1 reports arithmetic intensity around 1.0 and the highest latency among layer components).
  - In multi-tenant serving, many requests share the same long “system prompt,” yet conventional KV caches store duplicate key/value tensors per request, wasting memory and bandwidth (§2.1).
- Why it’s important
  - Longer contexts are now common (e.g., 32K tokens), so KV memory and access grow linearly with sequence length (§1).
  - KV cache size constrains batch size and throughput. Example given: with FP16 on GPT-3 175B, each token requires ~4.5 MB; an 8×A100 (80G) server can hold only ~70K tokens or 35 sequences of 2K tokens (§1).
  - Real-world prompts can be very long and shared. Table 2 shows shared system prompts across research workloads; Appendix A illustrates a chatbot with 6 plugins producing a 1,766-token shared prompt.
- Prior approaches and shortcomings
  - vLLM’s PagedAttention improves memory management through paging but does not automatically discover or exploit cross-request prefix sharing. A proposal to pre-reserve memory for predefined prompts requires manual configuration and risks memory waste when hit rates are low (§1, §5).
  - FlashAttention greatly speeds up attention via I/O-aware tiling, but it assumes monolithic, contiguous KV layout and fixed shapes that suit training; it provides little gain when decoding only one token per step and cannot leverage cross-request sharing (§5).
- Positioning
  - ChunkAttention targets dynamic, runtime discovery of shared prefixes across requests and builds a `prefix-aware KV cache (PAKV)` to share the actual KV tensors in memory. On top of that structure, it redesigns the attention kernel via `two-phase partition (TPP)` to improve data locality and arithmetic intensity during decoding (§3).

Definitions (selective):
- `KV cache`: the stored key and value tensors for all previously processed tokens, reused at each decoding step to avoid recomputation.
- `Arithmetic intensity`: ratio of compute to memory access. Low values (≈1) indicate memory-bound behavior; Table 1 shows self-attention has the lowest arithmetic intensity and highest latency per token across modules.

## 3. Technical Approach
The approach has two tightly coupled parts: a new data structure for KV storage that can safely share memory across requests with common prefixes, and a new decomposition of the attention computation that leverages that structure.

1) Prefix-Aware KV Cache (PAKV) (§3.1, Figure 1)
- Core idea
  - Replace each request’s monolithic KV tensors with a shared `prefix tree` (a trie-like structure). Each node holds a `chunk`—a small, fixed-length segment of tokens (`c` tokens)—plus the corresponding slices of key and value tensors for those tokens.
- What is a `chunk` and why chunk?
  - A `chunk` contains:
    - the token IDs of a contiguous segment shared by some set of sequences,
    - the key slice of shape `b × h × c × d`,
    - the corresponding value slice.
  - Chunking along the sequence dimension provides natural “paging” units that can be shared and scheduled. It also aligns well with GPU parallelism later used in TPP.
- Tree maintenance during serving
  - Insert: when a new request arrives, traverse the tree to find the longest matching prefix; reuse existing nodes; only allocate new chunks for the non-shared suffix (Figure 1, step (1)).
  - Append: during decoding, when a leaf chunk fills, start a new chunk for that sequence (Figure 1, step (2)).
  - Delete: when a sequence completes, remove its path; shared nodes remain if used by others (Figure 1, step (3)).
  - Memory allocator: a pool allocator manages fixed-size chunk buffers to avoid frequent OS allocations; alignment waste is bounded by `(c − 1)/n` for sequence length `n` (§3.1).
- Impact on batch capacity
  - If `ns` of the `np + nc` total tokens are shared, the `sharing ratio` is `r = ns / (np + nc)`. Sharing increases the number of sequences that fit in memory by about `1 / (1 − r)` (§3.1).
- Why a prefix tree?
  - The tree ensures each node covers a contiguous block of sequence indices in the active batch. This makes slicing the batched query matrix efficient for GPU kernels (§3.1).

2) Two-Phase Partition (TPP) attention kernel (§3.2, Figure 2)
Goal: During decoding, compute attention with minimal redundant memory traffic and better GPU utilization by batching the work shared across sequences first, then finishing per-sequence work.

Background: iteration-based batching (§2.2)
- Each decoding step, the system concatenates the last token from each of `b` active sequences into a batched query matrix `Q ∈ R^{b×d}`. Keys/values come from the cache for the entire history. This creates room to exploit sharing at the chunk level.

Phase A — Chunk-first (shared chunks, batched over sequences)
- Process only chunks that multiple sequences share (the shared prefixes).
- For each shared chunk `C`, take its covered sequence-index range `[i, j)` and compute a `partial attention` result for the corresponding slice `Q_{i:j,:}` against `K^{(C)}` and `V^{(C)}` in parallel across chunks (Algorithm 1, Eq. (1)).
- Online softmax is used to keep numerical stability without synchronizing across partitions.
- What Eq. (1) computes (intuitively):
  - Compute attention scores of the selected queries against keys in chunk `C`.
  - Track, per row (i.e., per sequence), the maximum logit `m^(C)` and the sum of exponentiated, shifted scores `n^(C)`.
  - Multiply the normalized weights by values to produce a partial output `O^(C)`.
- Save the triplet `(O^(C), m^(C), n^(C))` for later reduction.
- Why it helps
  - Because many sequences share the same chunk, computing their dot products in a batch improves data locality and uses tensor cores efficiently (query becomes a matrix, not a vector), raising arithmetic intensity (§3.2, Eq. (1), Figure 2).

Phase B — Sequence-first (private chunks, per sequence)
- For each sequence i (i.e., each row `q_i` of `Q`), load and merge the partial results from all shared chunks using `attn_reduce` (Algorithm 2, Eq. (2)).
- The reduction uses two scaling factors that align the different softmax normalizations into one consistent result, then adds contributions to the running output and normalization term. Finally, it divides by the accumulated normalization to obtain the final attention output for that sequence.
- After merging shared chunks, continue with any remaining private chunks for that sequence by calling the same `partial_attn` and merging with `attn_reduce` (Algorithm 2).
- Why two phases?
  - If only sequence-first is used, shared chunks would be reloaded b times—heavy on memory bandwidth. If only chunk-first is used, you cannot complete attention for suffix chunks unique to each sequence. The split balances locality and completeness (§3.2).

3) Prefilling and GPU/CPU orchestration (§3.2–§3.3)
- Prefill (processing the initial prompt tokens):
  - Skip recomputing QKV and positional embeddings for shared prefixes (prefix lookup).
  - Use a strong existing kernel such as FlashAttention on the resulting contiguous KV to compute the full attention for the prompt (§3.2).
- Runtime context for kernels:
  - The CPU maintains the prefix tree and creates a compact “kernel context” list of `(chunk, start_index, end_index)` for shared and private chunks (examples shown atop Figure 2).
  - Optimization: generate this context concurrently with other GPU work (latency hiding), and copy lazily to the GPU only when the tree structure changes (e.g., every `c` steps when a chunk fills, on joins/leaves) (§3.3).

4) Implementation note: temporary memory (§3.3)
- Merging partial results immediately would avoid temporary buffers, but on GPUs that would require atomic operations due to overlapping writes for parent/child chunks, which are costly. The design therefore stores partials and merges later on GPU; on CPU, a spin lock approach can serialize merges with low overhead.

Selective definitions:
- `Online softmax`: computes softmax over partitioned inputs by maintaining running maxima and normalization terms so that partitions can be processed independently and later combined stably.
- `Iteration-based batching`: batching decoding across many sequences by feeding each step’s last token from each active sequence together (§2.2).

Design choices rationale:
- Prefix tree over OS-style paging: paging alone cannot discover runtime prefix matches, and predefined prompt caching is brittle and inflexible (§1, §5).
- Two-phase compute: exploits shared KV locality first for high-utilization matrix–matrix operations, then completes per-sequence work without reloading shared memory repeatedly (Figure 2, Algorithms 1–2).

## 4. Key Insights and Innovations
- Prefix-aware KV cache via a prefix tree (PAKV) is a fundamental innovation (§3.1, Figure 1).
  - What’s new: storing KV slices in a trie of token chunks enables exact cross-request sharing for any matching prompt prefix discovered at runtime—no offline configuration.
  - Why it matters: eliminates redundant KV copies, increases effective batch capacity by ≈ `1/(1−r)` given sharing ratio `r` (§3.1), and sets up compute kernels to slice queries efficiently.
- Two-phase partition (TPP) attention kernel tailored to PAKV (§3.2, Figure 2, Eq. (1)–(2)).
  - What’s new: a computation schedule that first batches across sequences on shared chunks, then finishes per-sequence suffixes with a numerically sound reduction combining partial softmaxes.
  - Why it matters: raises arithmetic intensity and memory locality where it counts—during decoding—yielding 2.8–3.2× kernel speedups even compared to a baseline that artificially shares physical memory (PagedAttn*, Table 3).
- Out-of-the-box, dynamic reuse in multi-tenant serving (§2.1, §5).
  - What’s new: automatic detection of shared prefixes across arbitrary incoming requests; no need for developers or operators to pre-register prompts.
  - Why it matters: aligns with real workloads that use long system prompts (Table 2; Appendix A example with 1,766 tokens).
- No-regret optimization
  - Observation: when no prefix is shared, TPP does not regress performance relative to competitive baselines (ns=0 rows in Table 3). This makes it safe to enable by default.

Distinguishing levels of novelty:
- Fundamental: the PAKV data structure and its coupling with a bespoke decoding-time kernel (TPP).
- Incremental: engineering around CPU–GPU context construction and pool allocation, which are important for practicality but not conceptually new (§3.3).

## 5. Experimental Analysis
Setup overview (§4)
- Hardware and precision: NVIDIA A100 (80G), CUDA 11.8, FP16.
- Model settings for microkernel: head dimension `d=128`, heads `h=32`, chunk size `c=64`.
- Workloads:
  - Microkernel: synthetic batches where all sequences start/finish together; prompts of length `np`, with `ns` shared prefix tokens and `nc` decoded tokens. Metrics: decoding latency and throughput (tokens/s).
  - End-to-end: Llama2 7B FP16 serving stack “ChunkLlama” that replaces attention with ChunkAttention but otherwise uses vLLM/HF kernels (§4.2). Requests arrive via a Poisson process; max batch size 32. Metrics: normalized latency (ms/token) and peak KV memory; baselines are vLLM 0.2.7 and TGI 1.3.4.

Baselines
- Microkernel: Naive (PyTorch), xFormers’ memory-efficient attention, FlashAttention (PyTorch integration), vLLM PagedAttention, and PagedAttention* (a diagnostic variant with forced physical sharing to isolate compute effects) (§4.1).
- End-to-end: vLLM and TGI (§4.2).

Key quantitative findings
- Kernel latency vs. shared prefix length (Table 3, batch size 32)
  - Quote:
    > For `np=4096, ns=4096`, latency (µs): Naive 1370.41, xformers 1713.13, FlashAttn 6300.65, PagedAttn 1399.51, PagedAttn* 663.84, ChunkAttn 206.22.
  - Takeaway: even when PagedAttention is given perfect physical sharing (PagedAttn*), ChunkAttention’s TPP is 3.2× faster (663.84→206.22 µs).
  - No-regret case: with no sharing (`ns=0, np=1024`), ChunkAttn (332.50 µs) is comparable to PagedAttn* (355.82 µs), so enabling TPP doesn’t hurt (§4.1).
- Throughput vs. number of decoded tokens `nc` (Figure 3)
  - Quote (selected rows):
    > For `ns=2048`, tokens/s: at `nc=512`, PagedAttn 39.85K vs. ChunkAttn 145.41K (3.6×); at `nc=2048`, 30.17K vs. 70.33K (2.3×).
  - Trend: speedup declines as sequences diverge during longer decoding (fewer shared chunks remain), but a strong advantage persists even at `nc=2048`.
- Throughput vs. batch size (Figure 4)
  - With `ns=2048, nc=64`, ChunkAttn scales beyond the throughput peak of other baselines, from 155K toks/s at batch 16 to 224K at batch 96, indicating better memory locality and utilization when sharing exists (§4.1).
- End-to-end serving performance (Figure 5, Table 4)
  - Throughput vs. latency curves (Figure 5) show higher throughput at similar or lower normalized latency when shared prefixes exist.
  - Memory and latency (Table 4):
    - Quote:
      > For `np=4096, ns=4096, nc=512, RPS=0.4`, normalized latency: vLLM 27.62 ms/tok vs. ChunkLlama 17.16; peak KV memory: 35.42 GB vs. 4.00 GB; peak batch size: 16 vs. 11.
    - Quote:
      > For `np=1024, ns=1024, nc=512, RPS=1.0`, normalized latency: vLLM 20.80 vs. ChunkLlama 14.07; KV memory: 14.79 GB vs. 3.28 GB.
  - Reported throughput gain: 
    > “1.6× (2.9 vs. 1.8) and 2.3× (2.3 vs. 1.0) higher throughput at <40 ms/token for 1K and 2K shared tokens” (§4.2 text, Figure 5).

Assessment of support
- The microkernel study isolates the kernel and demonstrates gains scale with shared prefix length and persist over a range of batch sizes (Table 3, Figures 3–4).
- PagedAttn* is a useful control that holds physical memory sharing constant; the gap to ChunkAttention quantifies the value of TPP’s computation schedule, not just memory aliasing.
- End-to-end experiments incorporate queueing, dynamic batching, and CPU/GPU overheads. Memory reduction is substantial (70–90% KV memory with long shared prefixes, Table 4), consistent with the structural sharing premise.

Ablations and robustness
- The study varies `ns`, `np`, `nc`, and batch size. It also examines the “no-sharing” case (ns=0), showing no regression.
- Explicit ablation of chunk size `c` or alternative tree granularities is not reported; performance has been tuned for common LLM settings (Limitations §7).

## 6. Limitations and Trade-offs
- Requires the shared prompt to be at the sequence start (§7: “Position of System Prompt”).
  - If an application puts shared content in the middle or end, the KV prefixes do not match, so memory sharing does not apply. This reflects a known sensitivity of LLMs to information location in long contexts.
- Benefit depends on the degree and persistence of sharing
  - As decoding progresses, sequences diverge; the chunk-first gains diminish (Figure 3). Workloads with short or no shared prefixes see smaller advantages.
- Engineering complexity and portability (§7: “Model and Hardware Compatibility”)
  - The kernel is hand-written in CUDA and tuned for specific head sizes, GPUs (A100, RTX 4090), and CPUs. Porting may require re-tuning and incurs development cost.
- Runtime overheads and parameters (§3.3)
  - CPU–GPU “kernel context” generation and copying introduce overhead, mitigated by lazy copying and overlap but still present when the prefix tree changes (e.g., chunk rollover every `c` steps).
  - Temporary buffers store partial results for the reduction; immediate reduction on GPU would require heavy atomics, so the current design trades memory for speed.
- Assumptions about serving stack
  - The approach assumes iteration-based batching is in place (§2.2). Systems without it would not realize the same benefits.
- Alternative evolutions of practice
  - If organizations move from long shared prompts to fine-tuned models for each application, sharing opportunities diminish (§7: “Fine Tuning”).

## 7. Implications and Future Directions
- Broader impact on LLM serving
  - Establishes that cross-request structural sharing is a powerful optimization axis in multi-tenant settings. PAKV reframes the KV cache as a deduplicated, shareable data structure, not per-request state.
  - The two-phase kernel shows how to reshape decoding-time computation to exploit that structure, not just store it efficiently.
- Practical applications
  - Hosted chatbots and tool-augmented agents with long plugin/tool specifications (Appendix A’s 1,766-token example) can cut latency and KV memory dramatically.
  - Batch capacity improves when memory is the bottleneck; operators can serve more concurrent sessions or reduce hardware footprint.
- Research avenues
  - Beyond exact-prefix sharing:
    - Approximate or fuzzy sharing for near-identical prompts (e.g., different metadata headers) using hashing or small edit-distance matching over tokens.
  - Scheduling and systems:
    - Joint scheduling that groups requests by shared-prefix extent to maximize chunk-first utilization; integration with speculative decoding or continuous batching schedulers.
  - Algorithmic extensions:
    - Adaptive chunk sizes to balance locality and fragmentation; analytical models to pick `c` per hardware.
    - Combining PAKV with KV compression or quantization for even lower memory footprints.
  - Hardware/portability:
    - Generalizing TPP to other accelerators and to high-level libraries (e.g., cuDNN/oneDNN primitives) for wider adoption.
  - Training-time synergies:
    - Prompt canonicalization or tool-spec normalization to maximize prefix matches at inference.
    - Exploring whether training can encourage representational stability that tolerates minor token differences while enabling safe KV reuse.

In sum, ChunkAttention introduces both a new way to store KV cache across requests (PAKV) and a decoding kernel that capitalizes on that structure (TPP). Results across microbenchmarks (Table 3, Figures 3–4) and end-to-end serving (Figure 5, Table 4) indicate sizable, condition-dependent speedups and memory reductions, with no downside when no prefixes are shared.
