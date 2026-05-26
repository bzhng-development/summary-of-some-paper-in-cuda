# InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

**ArXiv:** [2502.08910](https://arxiv.org/abs/2502.08910)

## 🎯 Pitch

InfiniteHiP introduces a novel, training-free framework that enables large language models to handle up to 3 million tokens on a single commodity GPU—achieving this by uniting efficient hierarchical block-sparse attention, dynamic position encoding for out-of-length generalization, and smart KV-cache offloading that avoids forgetting any context. This combination delivers unprecedented scalability and speed, empowering production LLM deployments for massive-context applications like retrieval-augmented generation and long-form analysis, without expensive retraining or hardware upgrades.

---

## 1. Executive Summary

This paper introduces **InfiniteHiP**, a training-free LLM inference framework that enables efficient processing of contexts up to 3 million tokens on a single 48GB GPU by combining three mechanisms: a modular hierarchical token pruning algorithm that dynamically eliminates irrelevant context tokens through multi-stage chunk-based filtering (iteratively narrowing from all tokens to ~3K–5K selected keys per query block), dynamic Rotary Position Embedding (RoPE) adjustment strategies that selectively apply different positional encoding methods to early versus later layers for out-of-length generalization beyond pretrained context windows, and an LRU-based key-value cache offloading mechanism that stores most context in host memory while keeping frequently accessed tokens on GPU. On the ∞Bench benchmark with instruction-tuned Llama 3 8B, InfiniteHiP achieves a 9.99 percentage-point improvement in relative score over the strongest baseline InfLLM while processing 4× fewer key tokens, and delivers an 18.95× speedup in attention decoding for a 1M-token context compared to FlashAttention2—establishing that long-context inference can simultaneously improve quality, speed, and memory efficiency, though the framework's gains are most pronounced when the pretrained model already possesses relevant knowledge within its capability range and the challenge lies in efficiently retrieving it from extended contexts rather than acquiring fundamentally new reasoning abilities.

## 2. Context and Motivation

### The Core Problem: Long-Context Inference Is Simultaneously Expensive, Memory-Hungry, and Brittle

The paper addresses a **three-part problem** that arises whenever a Transformer-based LLM is asked to process sequences longer than a few thousand tokens. These three challenges interact and compound one another, but prior work tends to address them in isolation. InfiniteHiP's stated goal is to solve all three simultaneously in a training-free framework.

**Challenge 1: Quadratic attention cost makes long contexts prohibitively slow.** In a standard Transformer, the attention mechanism computes pairwise dot products between every query token and every key token in the context. For a sequence of length $T$, the number of operations scales as $O(T^2)$ — double the context length, quadruple the computation. The paper notes that this dominates inference time, particularly during the decoding stage where the attention calculation is memory-bandwidth-bound rather than compute-bound. FlashAttention (Dao et al., 2022) reduces memory pressure by avoiding writing the full attention matrix to GPU global memory, but it explicitly does not reduce the *arithmetic* computation count — it still performs every pairwise dot product. For contexts of 1M tokens, this arithmetic cost is simply too high for practical throughput.

**Challenge 2: The key-value cache consumes GPU memory linearly with context length.** At each decoding step, the model reuses previously computed keys and values from all preceding tokens. The size of this KV cache grows as $T$, and for large models with long contexts, it quickly overwhelms GPU VRAM. A 3M-token context with the Llama 3.1 8B model requires roughly 192GB of KV cache storage — far exceeding the 24–48GB available on typical GPUs. The paper highlights that KV cache eviction methods (discussed below) address this by permanently deleting tokens, but this creates a risk of losing information that may be needed later. Alternative solutions like naive host-memory offloading (e.g., placing the entire KV cache in CPU DRAM and fetching as needed) exist but introduce latency penalties: accessing CPU memory over PCIe is approximately 31.5× more expensive in latency than accessing VRAM, as the paper quantifies in Section 5.3.

**Challenge 3: Pretrained LLMs fail to generalize beyond their training context length.** Standard RoPE assigns a unique position ID to each token, and models are typically trained with sequences truncated to a fixed maximum length (e.g., 8K for Llama 3, 32K for Mistral 0.2). When prompted with sequences longer than this, the model encounters position IDs it has never seen during training, and performance degrades catastrophically. This is an **out-of-length (OOL) generalization** problem. The paper emphasizes that while long-context fine-tuning (Rozière et al., 2024) can extend the effective context window, it requires "exorbitant training costs and high-quality training data" — making it infeasible for many practitioners, particularly those who want to deploy existing pretrained models without modification.

These three challenges are **coupled** in practice. You cannot simply make attention faster if the KV cache does not fit in GPU memory. You cannot simply offload the KV cache if the attention mechanism still performs $O(T^2)$ computation on the GPU-resident subset. And you cannot extend the effective context length if the model's positional encodings break down beyond the pretrained window. A practical solution must address all three simultaneously.

---

### Why This Problem Matters

The paper's motivation extends beyond academic interest in efficient attention. There are concrete deployment scenarios where long-context capability is becoming essential:

**Retrieval-augmented generation (RAG) and multi-document QA.** When an LLM must reason over many retrieved documents, legal contracts, or code repositories, the total context can easily reach hundreds of thousands of tokens. Without efficient long-context inference, practitioners must choose between truncating context (losing potentially critical information) or waiting for impractically long inference times.

**Multi-modal and in-context learning applications.** The introduction mentions that extending context length is "essential for improving comprehension and coherence in long-context, multi-modal, and retrieval-augmented language generation." In few-shot learning scenarios, performance often improves with more examples in the prompt — but fitting dozens of examples requires long contexts. The paper's benchmarks specifically include few-shot learning tasks (TREC, TQA in LongBench) and multi-document QA (MFQA, HQA).

**On-device and consumer-grade deployment.** The paper explicitly targets single-GPU scenarios (RTX 4090 with 24GB, L40S with 48GB). For local LLM users who cannot access datacenter-scale multi-GPU setups, the ability to process long documents without hitting memory limits or waiting minutes per token is a practical necessity. The throughput benchmarks in Figure 5 demonstrate that InfiniteHiP achieves 3.20× higher tokens-per-second on a 4090 at 1M context compared to the standard SGLang runtime.

**Energy efficiency and cost.** The impact statement notes that the method "can significantly enhance energy efficiency and reduce inference latency." For production deployments serving millions of queries, reducing per-token computation for long-context inputs translates directly to lower operational costs and carbon footprint.

---

### Where Prior Approaches Fall Short

The paper identifies several families of prior work and explains what each one misses:

#### Dense Attention Optimizations (FlashAttention and Its Limits)

FlashAttention (Dao et al., 2022) and Flash Decoding (Dao et al., 2023) are the gold standard for efficient dense attention. They restructure the attention computation to minimize HBM reads/writes by fusing operations in SRAM, which substantially reduces memory bandwidth usage and allows exact attention with lower latency. The paper acknowledges this explicitly: "FA2 significantly reduces memory consumption and bandwidth utilization."

However, the paper identifies two critical limitations:
1. **FA2 does not reduce arithmetic operations.** It computes every pairwise dot product in the attention matrix. For a 1M-token context, this is $10^{12}$ operations for a single attention layer — and models have 32 such layers. The paper's Table 3 quantifies this: FA2 takes 4,645 µs per decoding step for a 1M-token context, making it practically unusable at scale.
2. **FA2 does not address KV cache memory pressure.** The entire KV cache must live on GPU VRAM. For 3M tokens, this requires ~192GB, which exceeds any single consumer or datacenter GPU. The paper's Table 4 explicitly notes: "FA2 does not support KV cache offloading and thus cannot run decoding with a context window exceeding 128K tokens using a single RTX 4090."

#### Static and Dynamic Token Selection (Sink + Streaming Tokens)

StreamingLLM (Xiao et al., 2024b) observed that a small number of initial "sink" tokens and the most recent "streaming" tokens capture most of the attention mass in many LLMs. Its approach is to attend only to these tokens — a fixed window of the most recent positions plus a few initial tokens — discarding all middle context. LM-Infinite (Han et al., 2024) applies a similar strategy with RoPE adjustments for OOL generalization.

The paper critiques this as too coarse. Fixed-window methods permanently discard the vast middle of the context, which contains information that may be needed. The LongBench and ∞Bench results in Tables 1 and 2 show that StreamingLLM and LM-Infinite perform significantly worse than methods that dynamically select context tokens based on query relevance. For instance, on Llama 3 8B at LongBench, LM-Infinite achieves 83.23% relative score and StreamingLLM achieves 83.21%, compared to InfiniteHiP's 100% — a roughly 17 percentage-point gap.

#### KV Cache Eviction (Permanent Forgetting)

H2O (Zhang et al., 2023) is representative of a class of methods that monitor attention scores across decoding steps and evict "cold" KV cache entries that have not been recently accessed. The insight is that not all past tokens are equally important for future predictions, and GPU memory is better spent on the "heavy hitter" tokens with high cumulative attention.

The paper's explicit objection is: "these methods permanently erase past contexts, which may be needed again later." If a document-question pair requires attending to a specific paragraph early in the context, and that paragraph was evicted because it wasn't relevant to intermediate tokens, the information is lost forever. The ∞Bench results in Table 2 confirm this: H2O achieves only 3.95% relative score on Llama 3 8B, near the bottom of all methods tested. The paper frames this as a fundamental limitation of eviction-based approaches: they trade recall for memory, and for tasks requiring retrieval from arbitrary positions in long documents, this trade-off is disastrous.

#### Query-Aware Dynamic Selection (The Direct Predecessors)

The methods most directly comparable to InfiniteHiP are those that dynamically select which context tokens to attend to based on the current query:

**InfLLM (Xiao et al., 2024a)** divides the context into fixed-size blocks, pre-selects representative tokens for each block (using a heuristic that does not change with the query), and at each decoding step computes attention scores between the query and these representatives to select the top-k blocks. The paper identifies two weaknesses: (1) the representative tokens are static — they don't adapt to different queries — which limits precision; (2) InfLLM "chooses not to access the CPU memory while executing its attention kernel, so it has to sacrifice the precision of its top-k estimation algorithm," forcing it to use larger context windows (12K tokens for comparable performance) which increases memory-bandwidth pressure during decoding.

**HiP Attention (Lee et al., 2024b)** is the direct predecessor to this work and addresses several of InfLLM's limitations. HiP uses a hierarchical, iterative algorithm to estimate the top-k context blocks with highest attention scores, dynamically adapting to each query. It also pioneered the KV cache offloading strategy that InfiniteHiP extends: keeping a smaller "key bank" on GPU as a cache, with misses fetched from host memory via unified virtual memory (UVM).

The paper identifies specific weaknesses in HiP: (1) The hierarchical selection algorithm "involves many global thread synchronizations, which hinders parallelism" — making it slower in practice despite good asymptotic complexity; (2) The heuristic-based pruning is less accurate than InfiniteHiP's modular approach, with Figure 6a showing InfiniteHiP achieves 4.72 percentage points higher recall of attention probabilities than HiP; (3) HiP's offloading strategy lacks sophisticated cache management — InfiniteHiP adds an LRU eviction policy for the GPU-resident token cache.

**Quest (Tang et al., 2024)** uses element-wise min/max vectors cached per page to estimate attention scores, but the paper positions it as another point in the design space rather than as a direct competitor that is extensively benchmarked.

#### RoPE Adjustment for OOL Generalization (Piecemeal Solutions)

The paper discusses NTK-aware scaling (bloc97, 2023), which adjusts the RoPE base frequency to extend the effective context window — but notes that it fails on longer sequences (0% accuracy on ∞Bench En.MC in Table 2). Self-Extend (Jin et al., 2024) proposes mapping extended positions back to the pretrained range using floor division, which works better but the paper argues is suboptimal when applied uniformly across all layers: different attention heads exhibit different positional sensitivity patterns, and a one-size-fits-all RoPE adjustment leaves performance gains on the table.

---

### How This Paper Positions Itself

InfiniteHiP positions itself not as a single-algorithm contribution but as a **unified framework** that integrates solutions to all three challenges. The introduction states: "What sets InfiniteHiP apart is its innovative use of pruning modules... By providing a unified solution to all the aforementioned problems as a whole, InfiniteHiP demonstrates strong practicality and is well suited for real-world deployment."

The positioning relative to prior work is nuanced:

**Relative to HiP Attention:** InfiniteHiP is an explicit improvement of HiP (Lee et al., 2024b), which was developed by overlapping authors. The paper states three key improvements: "First, our hierarchical pruning modules achieve higher accuracy compared to HiP's heuristic-based hierarchical pruning. Second, the pruning algorithm within each module is significantly faster due to its enhanced parallelizability. Lastly, its modular design enables fine-grained control over pruning-stage caches, leading to much faster decoding than HiP." The subtext is that HiP provided the conceptual foundation (query-aware hierarchical selection, KV offloading via UVM) but left significant performance on the table due to implementation-level choices about parallelism and cache management.

**Relative to InfLLM:** InfiniteHiP competes directly with InfLLM on the "dynamic query-aware selection + offloading" axis. The paper argues its advantages are: more precise top-k estimation (because representative tokens adapt to the query rather than being static), smaller effective context windows (3K–5K vs. 12K tokens selected) leading to faster decoding, and a cache management strategy that allows accessing CPU memory during kernel execution rather than restricting all data to GPU before the kernel starts. The benchmark results support this: InfiniteHiP outperforms InfLLM on both LongBench (7.17pp relative improvement on Llama 3) and ∞Bench (9.99pp), while achieving 4.98× faster decoding attention latency (Table 3).

**Relative to RoPE extension methods:** InfiniteHiP does not propose a single new RoPE adjustment strategy but rather a **selective combination** of existing strategies applied differently to early versus later layers. Section 4 explains that the first three layers use Chunk-indexed RoPE (each context chunk gets a single position ID based on chunk index) while deeper layers use Relative-style RoPE (position IDs offset from the query during the hierarchical selection process). The motivation, explained in Appendix D, is that early layers exhibit "dynamic sliding window-like attention" that relies heavily on relative positional information, while later layers can process long-range dependencies using learned semantic content. This layer-specific strategy is not found in any of the baselines, which apply uniform RoPE adjustments.

**As a training-free drop-in replacement:** The paper repeatedly emphasizes that InfiniteHiP is training-free — it can be applied to any pretrained Transformer-based LLM "as a drop-in replacement" without fine-tuning. This positions it against long-context fine-tuning approaches (Rozière et al., 2024) which achieve strong performance but require "exorbitant training costs." The benchmarking on multiple model families (Llama 3 8B, Mistral 0.2 7B, Llama 3.1 8B, Gemma2 9B, EXAONE 3, EXAONE 3.5, DeepSeek R1 Distilled Qwen2 14B) is designed to demonstrate this universality.

**Integration into production serving systems:** The implementation in the SGLang framework signals that InfiniteHiP is intended for real-world deployment, not just academic benchmarking. The throughput measurements in Figure 5 and the detailed latency breakdowns in Tables 3 and 4 are designed to convince practitioners that the method is not just accurate but *fast enough to deploy*.

### The Intellectual Gap: Reconciling Speed, Memory, and Accuracy

A subtle but important point about the paper's positioning: prior work tends to optimize one dimension at the expense of others. Static window methods (StreamingLLM) are fast and memory-efficient but inaccurate on retrieval tasks. Dense attention with full context (FA2) is accurate but slow and memory-prohibitive. InfLLM is accurate and memory-efficient but sacrifices some decoding speed because its static representatives force larger selection windows. HiP is accurate and faster but its hierarchical algorithm has parallelism bottlenecks.

InfiniteHiP's stated contribution is achieving **all three simultaneously**: accuracy (outperforming all baselines on LongBench and ∞Bench), speed (18.95× faster decoding than FA2, 4.98× faster than InfLLM), and memory efficiency (processing 3M tokens on 48GB via offloading, using only 3.34% of FA2's VRAM requirement). The paper's framing suggests that this simultaneous optimization is not merely an engineering convenience but reflects a better algorithmic understanding of attention sparsity patterns — specifically, the observation that "the top-k tokens are concentrated in a small number of context chunks" (Section 3), which justifies aggressive but structured pruning.

## 3. Technical Approach

### 3.1 Reader Orientation

**What is being built:** InfiniteHiP is an inference-time system that replaces the standard dense attention mechanism in Transformer-based LLMs with a modular, hierarchical token pruning pipeline, a dynamic positional encoding adjustment strategy, and a host-memory key-value cache offloading mechanism — all working together without requiring any model retraining or fine-tuning.

**What problem it solves and the shape of the solution:** Given a long context of up to millions of tokens, the system must produce exactly the same output quality as dense attention while using dramatically less GPU memory and computation. The solution is shaped as a **three-stage funnel**: a cascade of pruning modules that rapidly eliminate irrelevant context chunks (reducing the effective attention window from millions of tokens to ~3K–5K), combined with selective RoPE adjustments that prevent the model from breaking on sequences longer than its training length, plus an LRU-based caching layer that keeps only the most frequently accessed tokens on GPU while the rest live in cheap host memory.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components arranged in a sequential pipeline that executes once per attention layer per decoding step:

1. **Context Pruning Pipeline (3 stages):** Takes the full context keys and the current query block as input. Stage 1 narrows from all tokens to ~32K candidates by selecting the top chunks based on representative token scores. Stage 2 narrows from ~32K to ~8K. Stage 3 narrows from ~8K to the final 2K–4K tokens. Each stage discards chunks estimated to have low attention scores for this specific query.

2. **Representative Token Selector (called inside each pruning stage):** A hierarchical, log-time algorithm that picks one token from each context chunk to serve as a proxy for the chunk's maximum attention score, using a binary-search-like procedure that avoids computing attention against all tokens in the chunk.

3. **Dynamic RoPE Adjustment Layer:** Intercepts every key and query vector before attention computation and applies position-dependent frequency scaling. Early layers (1–3) get Chunk-indexed RoPE where each context chunk receives a single position ID; later layers get Relative-style RoPE where position IDs are computed as offsets from the current query.

4. **Block Sparse Attention Kernel:** Performs the actual attention computation using only the surviving key indices from the pruning pipeline, plus the always-included sink tokens and streaming tokens. Implemented using Triton with PagedAttention for memory management.

5. **KV Cache Offloading Manager:** Maintains two GPU-resident key banks (one for mask selection, one for block sparse attention), a page table mapping global key indices to GPU bank slots, and an LRU eviction policy. On cache miss, missing keys are fetched from CPU memory via Nvidia Unified Virtual Memory (UVM) and placed in the GPU banks, evicting the least recently used entries.

Information flows as follows at each decoding step: the new query token enters → early layers apply Chunk-indexed RoPE, later layers apply Relative RoPE → the pruning pipeline accepts the query block and progressively selects ~3K–5K key indices from the full context → during pruning, any cache-missing keys trigger UVM page faults that fetch data from host memory and update the GPU cache with LRU eviction → the selected key indices plus sink/streaming tokens form a sparse attention mask → block sparse attention computes final attention output using only these selected keys and values → the output is fed to the next transformer layer, and the process repeats for all 32 layers.

### 3.3 Roadmap for the Deep Dive

- **First, the empirical motivation (Section 3 observations):** The chunk sparsity measurements that justify why hierarchical chunk-based pruning works — establishing that most context chunks contribute nothing to attention, which is the empirical foundation for the entire approach.

- **Second, the formal background:** How standard multi-head attention works and where the computational bottlenecks arise — providing the mathematical context needed to understand what the pruning pipeline replaces.

- **Third, the pruning pipeline in full detail:** How each of the three pruning stages works, including the chunk partitioning logic, the representative token selection algorithm (SelectRep), the chunk score estimation formula, and the top-K selection procedure — this is the core algorithmic contribution.

- **Fourth, the sparse attention mask caching mechanism:** How temporal locality in attention patterns is exploited to avoid recomputing the full pruning pipeline at every decoding step, and how the refresh interval hyperparameters allow trading accuracy for speed.

- **Fifth, the dynamic RoPE adjustment strategy:** The four RoPE interpolation styles (Dynamic Extend, InfLLM-style, Chunk-indexed, Relative), the layer-specific assignment (Chunk-indexed for layers 1–3, Relative for layers 4+), and the empirical motivation from attention pattern analysis in Appendix D.

- **Sixth, the KV cache offloading architecture:** The dual GPU key bank design, the page table structure, the LRU eviction policy, the UVM integration, and how this improves on HiP Attention's offloading strategy.

- **Seventh, the implementation details:** The Triton kernel design, the PagedAttention integration, and how the pruning stages are mapped to GPU parallelism.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems and algorithms paper** whose core insight is that attention in long-context LLMs exhibits extreme chunk-level sparsity — most context chunks contribute negligible attention scores for any given query — and that this sparsity can be exploited through a modular, parallelizable hierarchical pruning pipeline that simultaneously reduces computation, enables KV cache offloading, and supports out-of-length generalization through selective positional encoding adjustments.

---

#### The Empirical Foundation: Chunk Sparsity of Attention (Section 3)

Before designing any algorithm, the paper establishes *why* chunk-based pruning should work. The key observation, visualized in Figure 2a, comes from analyzing the attention score distributions in a Llama 3.1 8B model processing a 128K-token context.

**Measurement procedure.** The authors extract data from one attention layer and examine how the top-2048 key tokens (those with the highest attention scores for a given query) are distributed across the context. They partition the context into fixed-size chunks of varying sizes (8, 16, 32, 64, 128 tokens) and ask two questions: (1) what fraction of chunks contain what fraction of the top-2048 keys? and (2) what fraction of chunks contain *none* of the top-2048 keys?

**Finding 1: Extreme concentration.** The left histogram in Figure 2a shows that "fewer than 2% of the chunks contain more than 12.5% of the top-2K tokens in a 128K-token context." In other words, the most important keys are highly clustered — they do not spread uniformly across the sequence but concentrate in a small number of regions.

**Finding 2: Most chunks are empty of top tokens.** The right chart in Figure 2a shows that with 64-token chunks, "around 75% of the 64-token context chunks do not contain any top-2K tokens at all." As chunk size decreases, the fraction of empty chunks increases. This means that even at moderate chunk granularity, three-quarters of the context can be discarded without losing any of the top-2048 tokens.

**Operational implication.** These two findings together imply that selecting the few context chunks that contain top-k tokens can serve as a good approximation for selecting individual top-k tokens. If 75% of chunks are irrelevant, a pruning algorithm that filters out irrelevant chunks before performing fine-grained token selection can dramatically reduce computation without sacrificing attention quality. The challenge is to *identify* which chunks to keep — and to do so efficiently enough that the pruning itself does not become a bottleneck.

**Design consequence: hierarchical, multi-stage pruning.** Because the fraction of relevant chunks is small but the total number of chunks is still large (2,000 chunks of 64 tokens for 128K context), a single-stage filter would need to evaluate every chunk — which is still expensive. The paper's solution is a multi-stage cascade: the first stage does coarse filtering with large chunk sizes (256 tokens), the second stage refines with medium chunks (32 tokens), and the third with small chunks (8 tokens). Each stage operates on only the surviving tokens from the previous stage, so the total work across all three stages is far less than a single fine-grained pass over the full context.

---

#### Formal Background: Standard Multi-Head Attention

To understand what InfiniteHiP replaces, the paper briefly formalizes the standard attention computation. Given query, key, and value sequences `$Q, K, V \in \mathbb{R}^{H \times T \times d}$` where `$H$` is the number of attention heads, `$T$` is the sequence length, and `$d$` is the per-head embedding dimension:

$$O = \text{Concat}[O_1, \ldots, O_H]$$

where each head's output `$O_h$` is computed as:

$$S_h = Q_h K_h^{\top} \in \mathbb{R}^{T \times T}$$

$$P_h = \text{softmax}(S_h) \in \mathbb{R}^{T \times T}$$

$$O_h = P_h V_h \in \mathbb{R}^{T \times d}$$

The softmax is applied row-wise, causal masking and constant scaling are omitted for brevity, and `$S_h$` and `$P_h$` are called the attention scores and attention probabilities respectively.

**Where the cost comes from.** The matrix multiplication `$Q_h K_h^{\top}$` computes `$T \times T$` dot products — one for every query-key pair. For a 1M-token context with `$H = 32$` heads and `$d = 128$`, this is approximately `$32 \times 10^6 \times 10^6 \times 128 \times 2 \approx 8.2 \times 10^{15}$` floating-point operations per attention call. During decoding, this must be done for every new token generated.

**What the sparse approximation does.** InfiniteHiP replaces the full quadratic attention with block sparse attention where only a subset of key indices `$I \subset \{1, \ldots, T\}$` is selected for each query block. If `$|I| = k \ll T$`, the cost drops from `$O(T^2)$` to `$O(T \cdot k)$` per query block. For the default 3K window (`$k \approx 3,000$`) on a 1M-token context, this is a ~333× reduction in key tokens, translating to a proportional reduction in dot-product computation.

---

#### The Three-Stage Context Pruning Pipeline

The core algorithmic contribution is the modular pruning pipeline, described in Algorithms 1 and 2 of the paper (reproduced in full in Appendix A). This pipeline transforms the full set of key indices into a sparse subset for each query block through three consecutive pruning stages.

##### Stage Configuration

Each pruning stage `$S^{(i)}$` is parameterized by a tuple `$(b_q^{(i)}, l_c^{(i)}, k^{(i)})$` where:
- `$b_q^{(i)}$` is the query block size (how many consecutive query tokens are processed together)
- `$l_c^{(i)}$` is the chunk size (how many key tokens are grouped into one chunk for coarser or finer filtering)
- `$k^{(i)}$` is the number of key tokens to retain after this stage

The default hyperparameters for the three stages are:

| Stage | `$b_q$` | `$l_c$` | `$k$` |
|-------|---------|---------|-------|
| 1 | 64 | 256 | 32K |
| 2 | 64 | 32 | 8K |
| 3 | 64 | 8 | 2048 (4096 for layers ≤ 3) |

For the "5K window" preset, the third stage keeps `$k^{(3)} = 4096$` tokens across all layers. Additionally, `$n_{\text{sink}} = 256$` sink tokens (initial context tokens that act as attention anchors) and `$n_{\text{stream}} = 1024$` streaming tokens (most recent tokens) are always included and never pruned — they bypass the pruning pipeline entirely.

**Why three stages and why these numbers.** The progression from large chunks (256) to small chunks (8) creates a coarse-to-fine filtering strategy. Stage 1's large chunks mean fewer total chunks to evaluate (e.g., 1M tokens ÷ 256 = ~3,900 chunks), making the initial sweep cheap despite covering all tokens. Stage 1 reduces the candidate set from 1M to 32K. Stage 2 processes 32K tokens with 32-token chunks (1,000 chunks), reducing to 8K. Stage 3 processes 8K tokens with 8-token chunks (1,000 chunks), reducing to 2–4K. The total number of chunk evaluations across all three stages is approximately 3,900 + 1,000 + 1,000 = 5,900 — dramatically less than evaluating all 125,000 chunks of size 8 that would be needed for a single-stage filter, and far less than evaluating all 1M individual tokens.

##### Formal Mechanism of a Single Pruning Stage

Algorithm 2 (PruningStage) operates as follows, given a pruning stage specification, the previous stage's retained key indices `$I_m$` for the `$m$`-th query block, the full query tensor `$Q$`, and the full key tensor `$K$`:

**Step 1: Query blocking.** The query tensor `$Q \in \mathbb{R}^{H \times T_q \times d}$` is divided into `$n_{\text{block}} = T_q / b_q$` blocks, where the `$m$`-th query block for attention head `$h$` is:

$$q_{h,m} = Q_{h, m \cdot b_q : (m+1)b_q - 1} \in \mathbb{R}^{b_q \times d}$$

In plain language: for each attention head, extract consecutive chunks of `$b_q$` query tokens. With `$b_q = 64$`, each block contains 64 query vectors that will share the same sparse attention mask.

**Step 2: RoPE application to queries.** Each query block has RoPE applied according to the layer-dependent ApplyRopeQ function (described in the Dynamic RoPE section below), producing `$\tilde{q}_{h,m}$`.

**Step 3: Chunk partitioning of retained keys.** The retained key indices `$I_m$` from the previous stage are divided into `$n_{\text{chunk}} = |I_m| / l_c$` equally sized contiguous chunks. The `$j$`-th chunk is:

$$C_{m,j} = [I_m[j \cdot l_c], \ldots, I_m[(j+1) \cdot l_c - 1]]$$

This groups adjacent key indices — exploiting the locality observation that nearby tokens tend to have similar attention scores.

**Step 4: Representative token selection per chunk.** For each chunk `$C_{m,j}$` and each attention head `$h$`, the SelectRep algorithm (Algorithm 3) picks a single representative token index `$r_{h,m,j}$` from that chunk. This token serves as a proxy for the chunk's "attention-worthiness." The SelectRep algorithm is described in the next subsection.

**Step 5: Chunk score estimation.** For each chunk `$j$`, compute a single scalar score:

$$s_{m,j} = \max_{h=1..H,\; t=1..b_q} [\tilde{q}_{h,m}]_{t}^{\top} \tilde{k}_{h, r_{h,m,j}}$$

where `$\tilde{k}_{h, r_{h,m,j}}$` is the RoPE-applied key vector for the representative token of chunk `$j$` in head `$h$`.

**What this equation computes:** For the representative token in chunk `$j$`, compute the dot product between its key vector and each of the `$b_q$` query vectors in the query block, across all `$H$` attention heads. Take the maximum across all queries and all heads. This produces a single scalar `$s_{m,j}$` that estimates the *maximum* attention score that any query token in this block would assign to any key token in chunk `$j$`. The maximum captures the worst-case importance: if even a single query-head pair has high attention to the representative, the whole chunk might be relevant.

**Why the max:** alternatives like mean or sum would dilute the signal from highly relevant tokens within mostly-irrelevant chunks. If one token in a chunk is crucial, the entire chunk should be retained because block sparsity operates at chunk granularity. The max is the conservative choice that minimizes false rejections (dropping chunks that contain important tokens) at the cost of potentially retaining some false positives (keeping chunks where only the representative appears relevant).

**Step 6: Top-K chunk selection.** From the `$n_{\text{chunk}}$` scores, select the top `$K^{(i)} = k^{(i)} / l_c^{(i)}$` chunks:

$$T_m = \arg\text{top}_j K^{(i)}(s_{m,j})$$

This selects the `$K^{(i)}$` chunk indices with the highest estimated attention scores, discarding all others.

**Step 7: Index assembly.** The surviving key indices for this stage are the union of all tokens from the selected chunks:

$$I'_m = \bigcup_{\hat{j} \in T_m} C_{m,\hat{j}}$$

**Step 8: Query block propagation.** For the next pruning stage, the query block size may change, requiring re-indexing. If the next stage uses query block size `$b_q^{(i+1)}$`, then:

$$I^{(i)}_{m'} = I'^{(i)}_m \quad \text{for all } m' \text{ such that } m' = \lceil m \cdot b_q^{(i)} / b_q^{(i+1)} \rceil$$

In the default configuration where all three stages use `$b_q = 64$`, this simplifies to `$m' = m$`.

**Output after all stages.** After all `$N = 3$` stages complete, the paper produces sparse key indices `$I^{(N)}_m \in \{1, \ldots, T\}^{k^{(N)}}$` for each query block `$m = 1, \ldots, T_q / b_q^{(N)}$`. These indices, combined with the always-included sink and streaming tokens, define the block sparse attention mask.

##### The Initial Stage: Why It Is Asymptotically Different

A crucial detail appears in Appendices A and G: the first pruning stage is the only one whose cost grows linearly with the total context length. The paper states: "the initial pruning stage of InfiniteHiP's context pruning algorithm runs in `$O(T_q T_{kv})$` time, and all subsequent pruning stages run in `$O(T_q)$` time."

**Why this happens:** In the initial stage, the key indices `$I^{(0)}_m$` are `$[1, \ldots, T]$` — the entire context. The number of chunks is `$T / l_c^{(1)}$`, which grows with context length. The chunk score computation must evaluate all of these chunks. In subsequent stages, the number of input tokens is fixed at `$k^{(1)} = 32K$`, so the number of chunks is `$32K / l_c^{(2)}$`, which is independent of total context length.

**Practical consequence:** As context length grows to millions of tokens, the first pruning stage dominates the pruning cost. The paper mitigates this through mask caching: the first stage's output mask is not recomputed at every decoding step but is refreshed only periodically (every 16 decoding steps by default), exploiting the observation that attention patterns exhibit temporal locality.

##### Important: Sink and Streaming Tokens Are Never Pruned

The paper explicitly states that the initial `$n_{\text{sink}}$` tokens (positioned at the start of the context) and the most recent `$n_{\text{stream}}$` tokens (the latest generated tokens) are always included in the sparse attention mask and bypass the pruning pipeline entirely. This is reflected in Algorithm 1, line 1:

$$I^{(0)}_m = [n_{\text{sink}}, \ldots, b_q^{(1)} \cdot m - n_{\text{stream}}]$$

The middle tokens — everything between the sink region and the streaming region — are the only ones subject to pruning. This design reflects the central insight from StreamingLLM (Xiao et al., 2024b) that initial sink tokens and recent streaming tokens are disproportionately important for maintaining model coherence, and it ensures that the pruning pipeline can be aggressive with middle-context tokens without destabilizing the model's basic language modeling capability.

---

#### The Representative Token Selection Algorithm (SelectRep)

Algorithm 3 implements a hierarchical, binary-search-like procedure to pick one token from a chunk of `$l_c$` keys that approximately maximizes the attention score with the given query block `$q \in \mathbb{R}^{b_q \times d}$`. This is the same algorithm used in HiP Attention (Lee et al., 2024b) but adapted to a top-1 variant.

##### Algorithm Mechanics

Given a query block `$q$`, a chunk of key indices `$C \in \mathbb{N}^{l_c}$`, and all keys `$K \in \mathbb{R}^{T_{kv} \times d}$`:

**Initialization.** Load the key vectors corresponding to the chunk indices: `$k = [K_{C_1}, \ldots, K_{C_{l_c}}]^{\top} \in \mathbb{R}^{l_c \times d}$`. Set the initial search range to the entire chunk: `$(n_{\text{first}}^{(1)}, n_{\text{last}}^{(1)}) = (1, l_c)$`. Set `$n_{\text{iter}} = \lceil \log_2(l_c) \rceil$` — the number of binary search iterations needed.

**Iterative refinement.** For each iteration `$i = 1, \ldots, n_{\text{iter}}$`:

1. **Find the midpoint** of the current range: `$m^{(i)} = \lfloor (n_{\text{first}}^{(i)} + n_{\text{last}}^{(i)}) / 2 \rceil$`.

2. **Define two sub-ranges:** the left branch `$B_1^{(i)} = (n_{\text{first}}^{(i)} : m^{(i)} - 1)$` and the right branch `$B_2^{(i)} = (m^{(i)} : n_{\text{last}}^{(i)})$`.

3. **Pick the first token from each branch** as a representative for that branch: `$r_j^{(i)}$` is the first index in `$B_j^{(i)}$` for `$j = 1, 2$`.

4. **Score each branch's representative** by computing the max dot product between the query block and that representative's key:
   $$\sigma_j^{(i)} = \max_{t=1..b_q} [\tilde{q}]_t^{\top} \tilde{k}_{r_j^{(i)}}$$
   where `$\tilde{q}$` has RoPE applied and `$\tilde{k}$` has RoPE applied according to the branch (`$j = 1$` gets one position offset, `$j = 2$` gets another — see Dynamic RoPE section).

5. **Select the winning branch:** `$t^{(i)} = \arg\max_j \sigma_j^{(i)}$`.

6. **Narrow the search range:** `$(n_{\text{first}}^{(i+1)}, n_{\text{last}}^{(i+1)}) = B_{t^{(i)}}^{(i)}$`.

**Final output.** After `$n_{\text{iter}}$` iterations, the search range has been narrowed to a single index: `$r = n_{\text{first}}^{(n_{\text{iter}})}$`.

**What the algorithm computes:** Starting from the full chunk of `$l_c$` tokens, it recursively divides the chunk in half, evaluates one representative from each half, keeps the half whose representative has the higher attention score, and repeats until one token remains. It is analogous to a single-elimination tournament where each "match" compares two halves of a chunk by their first elements, and the winner advances.

**Why this works (the locality assumption):** The algorithm relies on the observation from Lee et al. (2024b) that "nearby tokens tend to display similar attention scores." If this holds, then the first token of a sub-range is a reasonable proxy for the maximum attention score within that sub-range — the actual maximum is likely nearby and similarly scored. The hierarchical tournament structure then approximately finds the token with the global maximum attention score without evaluating all `$l_c$` tokens.

**Time complexity:** The algorithm evaluates exactly `$2 \cdot \lceil \log_2(l_c) \rceil$` tokens (two representatives per iteration) rather than all `$l_c$` tokens. For `$l_c = 256$`, this is ~16 evaluations instead of 256 (a 16× reduction). For `$l_c = 8$`, it is 6 evaluations instead of 8.

**Why this matters for parallelism:** The paper states that unlike HiP Attention's version, this SelectRep algorithm "can be implemented with a single GPU kernel, without any global synchronizations between each iteration, while providing key sequence dimension parallelism like FlashDecode." The key insight is that because only two representatives are evaluated per iteration regardless of chunk size, and because all chunks across the key dimension can be processed in parallel (each chunk's tournament is independent), the algorithm maps efficiently to GPU thread blocks without requiring global barriers.

---

#### The Chunk Score Estimation Formula in Detail

Returning to the chunk score estimation from the pruning stage:

$$s_{m,j} = \max_{h=1..H,\; t=1..b_q} [\tilde{q}_{h,m}]_{t}^{\top} \tilde{k}_{h, r_{h,m,j}}$$

**What each symbol means:**
- `$m$` indexes the query block (there are `$T_q / b_q$` blocks total)
- `$j$` indexes the chunk within the retained key indices
- `$h$` indexes the attention head (1 through `$H$`, typically 32)
- `$t$` indexes the query position within the query block (1 through `$b_q$`, typically 64)
- `$r_{h,m,j}$` is the representative token index for chunk `$j$` selected by SelectRep for head `$h$` and query block `$m$`
- `$\tilde{q}_{h,m}$` is the RoPE-applied query block for head `$h$`, block `$m$`
- `$\tilde{k}_{h, r_{h,m,j}}$` is the RoPE-applied key vector for representative token `$r_{h,m,j}$` in head `$h$`

**Operational description:** For a single chunk of context keys, the system first selects one representative token per attention head using SelectRep. Then, for each head, it computes the dot product between all 64 query vectors in the current query block and that head's representative key vector. It takes the maximum across all queries (`$t = 1..b_q$`) and across all heads (`$h = 1..H$`). The result is one scalar per chunk representing the highest attention score any part of the query block would assign to any key token in that chunk (as estimated through the representative).

**Why max-pool across heads:** Different attention heads attend to different aspects of the context. A chunk that is irrelevant for 31 heads but critically important for 1 head (e.g., a head specializing in numerical reasoning attending to a number in the chunk) should be retained. Taking the max ensures that if *any* head finds the chunk important, the chunk survives. Taking the mean would dilute the signal and potentially discard chunks that matter for specialized heads.

**Why max-pool across query positions:** With `$b_q = 64$`, the query block contains 64 consecutive query tokens that will share the same sparse attention mask. The mask must include all chunks that are relevant to *any* of these 64 queries — if a chunk is important for query position 37 but not for query position 1, it must still be retained because the block sparse attention mask is shared across the entire block. The max across query positions ensures completeness.

**Why per-head representative selection:** The representative token is selected independently for each attention head because different heads have different positional sensitivities and content preferences. The token that best represents the chunk for one head may differ from the token that best represents it for another. This is particularly relevant when applying the layer-specific RoPE adjustments, which cause early-layer heads to rely more heavily on relative position information.

---

#### Sparse Attention Mask Caching (Temporal Locality Exploitation)

Computing the full three-stage pruning pipeline at every decoding step would add latency that partially offsets the savings from reduced attention computation. The paper introduces mask caching to amortize this cost: the sparse attention mask for each pruning stage is recomputed only periodically, not at every step.

**The temporal locality observation.** The paper states: "We observe that the sparse attention mask exhibits temporal locality." This means that during autoregressive decoding, the set of important context chunks changes slowly — the chunks that were important for decoding step `$t$` are likely to still be important for step `$t+1$`, with gradual drift as the model's focus shifts through the document.

**Caching mechanism per stage.** Each pruning stage `$i$` has a mask refresh interval `$n^{(i)}_{\text{refresh}}$` that controls how often its output mask is recomputed. When a mask is "cached," the stage is simply skipped and the previously computed mask is reused. The stage's output indices `$I^{(i)}_m$` are stored in GPU memory and only updated when `$(\text{step\_counter} \mod n^{(i)}_{\text{refresh}}) = 0$`.

**Default refresh intervals:**

| Stage | `$n_{\text{refresh}}$` (default) | `$n_{\text{refresh}}$` (fast) | `$n_{\text{refresh}}$` (flash) |
|-------|----------------------------------|-------------------------------|---------------------------------|
| 1 | 16 | 32 | 96 |
| 2 | 8 | 16 | 24 |
| 3 | 4 | 8 | 8 |

**Why different refresh rates per stage:** Stage 1 has the most expensive computation (it processes the full context, `$O(T)$` cost) and is refreshed least frequently. Stage 3 has the cheapest computation (processes only `$k^{(2)} = 8K$` tokens) and is refreshed most frequently to maintain precision on the final token selection. The paper has deliberately staggered the refresh rates so that all three stages are not refreshed simultaneously, distributing the recomputation cost across decoding steps.

**The "fast" and "flash" presets.** Table 2 introduces two speed-optimized configurations: "3K-fast" uses longer refresh intervals `$(32, 16, 8)$` and "3K-flash" uses even longer intervals `$(96, 24, 8)$`. The ∞Bench results show that 3K-fast achieves 98.35% relative score (vs. 98.17% for default 3K), and 3K-flash achieves 97.78% — minimal quality degradation for substantial speed improvements. The throughput benchmarks in Table 4 and Figure 5 show that the Flash configuration improves decoding throughput by approximately 3.14× in a 3M context compared to the Fast configuration.

**Visualization of stage caching.** Appendix A, Figure 7 provides a visual timeline: over the course of 17 decoding steps, Stage 1 refreshes at steps 0 and 16 (interval 16), Stage 2 refreshes at steps 0, 8, and 16 (interval 8), and Stage 3 refreshes at steps 0, 4, 8, 12, and 16 (interval 4). At step 5, for example, only Stage 3 is recomputed — Stages 1 and 2 reuse cached masks from earlier steps.

**Quantifying cache effectiveness.** Table 4 reports "Mask Hit Ratio" — the fraction of decoding steps where a given stage's cached mask was reused rather than recomputed. For stages 1, 2, and 3 at 256K context with all caching enabled, the hit ratios are not reported separately, but the overall pattern shows that at 1M context length with all stages cached, the mask hit ratio reaches 98.38% for Stage 2 and 88.97% for the sparse attention (SA) stage. This confirms that the vast majority of steps benefit from caching.

**Refresh strategy at the initial stage.** During decoding, the first stage is refreshed only when `$(c^{(1)} \mod n^{(1)}_{\text{refresh}}) = 0$`. The paper specifies that the sparse attention mask for the first stage is updated "periodically every `$n^{(1)}_{\text{refresh}}$` steps using the latest query block." This means the mask is computed using the most recent query token (or query block) at the time of refresh, not using historical queries. Subsequent steps reuse this mask until the next refresh.

---

#### Dynamic RoPE for Out-of-Length Generalization

The paper's approach to RoPE adjustment is unique in two ways: it applies different strategies within different *components* of the inference pipeline (pruning vs. block sparse attention), and it applies different strategies to different *layers* of the model (early vs. later).

##### The Problem: RoPE Breaks Beyond Training Context Length

Standard RoPE (Rotary Position Embedding) encodes position `$p$` by rotating the query and key vectors by an angle proportional to `$p$`:

$$f(q, p) = q \odot \cos(p \cdot \theta) + \text{rotate}(q) \odot \sin(p \cdot \theta)$$

where `$\theta$` are frequency parameters. For a model pretrained on sequences of maximum length `$L_{\text{train}}$` (e.g., 8K for Llama 3), positions `$p > L_{\text{train}}$` produce rotations with angles the model has never seen, leading to degraded attention patterns and catastrophic performance loss. Table 2 confirms this: FA2 with truncated context achieves perfectly normal accuracy on Llama 3 at 8K, but Dynamic-NTK applied to a 128K context yields 0.00% on three of four synthetic tasks — the RoPE extension fails completely.

##### Four RoPE Styles for Context Pruning

Table 5 explores four RoPE interpolation styles applied during the context pruning (mask generation) phase:

1. **Dynamic Extend (DE, SelfExtend-style):** Maps extended positions back to the pretrained range using floor division. If the trained range is `$[0, L_{\text{train}}]$` and a token is at position `$p > L_{\text{train}}$`, it is assigned a RoPE position of `$\lfloor p / s \rfloor$` where `$s = \lceil p / L_{\text{train}} \rceil$` is a scaling factor. This "stretches" the positional encoding space. Figure 9a visualization shows wiggly artifacts beyond the pretrained context length limit as the scaled positions create irregular patterns in the attention mask.

2. **InfLLM-style (IL):** Each context chunk receives a single position ID, but the assignment is based purely on content-based chunk selection rather than original position. The visualization in Figure 9d shows "strong vertical patterns because they rely only on the content information in the key vectors rather than the positional information."

3. **Chunk-indexed (CI):** Each key chunk is assigned a single position ID computed as the chunk's index offset from the current query. Specifically, the last chunk's position ID is offset by `$n_{\text{stream}}$` from the current query. All keys within the same chunk share the same position ID. This creates sliding-window-like artifacts in the attention mask, visible in Figure 9b as "curved wiggly artifacts in the second and third stages."

4. **Relative (RT):** During the hierarchical top-1 representative token selection, the left branch representative gets a position ID offset by `$n_{\text{stream}} + 1$` from the current query, and the right branch gets an offset by `$n_{\text{stream}}$`. These two possible offsets (`$n_{\text{stream}}$` and `$n_{\text{stream}} + 1$`) are the only values used for positional encoding during the pruning process, making it fully relative to the query position.

##### Three RoPE Styles for Block Sparse Attention

After the pruning mask is generated, the block sparse attention (BSA) kernel applies actual attention computation on the selected tokens. The paper explores three RoPE styles for this phase:

1. **Dynamic Extend (DE):** Same SelfExtend-style stretching as used in pruning.
2. **InfLLM-style (IL):** Chunk-based position IDs.
3. **StreamingLLM-style (ST):** The selected keys, including sink and streaming tokens, are given position IDs "sequentially in their original order, where the most recent token is given the same position ID as the current query." This preserves the relative ordering of tokens while compressing the position IDs of non-contiguous selected tokens to a dense sequential range.

##### Layer-Specific Strategy Assignment

The paper's most distinctive RoPE design choice is the layer-specific assignment:

**For layers 1–3:** Chunk-indexed RoPE is used during context pruning. These layers receive `ApplyRopeQ` and `ApplyRopeK` defined as:

$$\text{ApplyRopeQ}_l(q) = \text{ApplyRope}(q, p[\min\{i_{\text{orig}}, l_c + n_{\text{stream}}\}]) \quad \text{if } l \leq 3$$

$$\text{ApplyRopeK}_{l,j}(k) = \text{ApplyRope}(k, p[c_{\text{orig}}]) \quad \text{if } l \leq 3$$

where `$i_{\text{orig}}$` is the original position of the query, `$c_{\text{orig}}$` is the index of the chunk containing the key, and `$p_i \in \mathbb{R}^d$` is the rotary positional embedding vector for position `$i$`.

**For layers 4–32:** Relative-style RoPE is used:

$$\text{ApplyRopeQ}_l(q) = \text{ApplyRope}(q, p[n_{\text{stream}} + 1]) \quad \text{if } l > 3$$

$$\text{ApplyRopeK}_{l,j}(k) = \text{ApplyRope}(k, p[j - 1]) \quad \text{if } l > 3$$

where `$j$` is the branch index (1 or 2) in the SelectRep algorithm, making the position purely relative.

**Why this split (Appendix D provides the full justification):** The paper analyzes attention patterns across layers in a Llama 3.1 8B model (Figure 10) and finds that "the earlier layers (e.g., layers up to 5) strongly exhibit dynamic sliding window-like attention, which signifies that these layers focus on relative positional key tokens." In other words, early layers rely heavily on position information to build local representations — they need to know that token A is 3 positions from token B, not merely that both are in the same chunk. Chunk-indexed RoPE preserves this relative structure within chunks while making inter-chunk distances coarse.

Later layers, once positional representations are established, "can efficiently process long-range information in subsequent layers by leveraging learned semantics instead of positional cues." These layers benefit more from Relative RoPE, which eliminates absolute position entirely and focuses on content-based matching — the SelectRep tournament uses relative offsets that isolate which of two tokens is more relevant to the current query without being influenced by how far away they are.

**Ablation validation.** Table 5 in the main paper measures ∞Bench En.MC accuracy with various RoPE combinations. The chosen combination — Relative-style RoPE for context pruning and StreamingLLM-style RoPE for block sparse attention ("RT/ST") — achieves 70.31% accuracy on Llama 3.1 8B with 128K context. Table 6 in Appendix D shows that using Chunk-indexed RoPE in layers 1–3 plus Relative RoPE in layers 4–32 achieves 74.23%, compared to 68.55% for Relative RoPE in all layers — a 5.68 percentage point improvement from the layer-specific assignment.

**The RoPE overhead tradeoff.** The paper explicitly notes: "Since this dynamic RoPE trick incurs some computational overhead, it can be disabled when the OOL generalization capability is not needed." Table 3 quantifies this: with context extension (dynamic RoPE) enabled, the method "slows down about 1.6× in prefill and 5% in decoding due to overheads incurred by additional memory reads of precomputed cos and sin vectors." The overhead comes from needing to look up and apply different position-dependent cos/sin vectors for different tokens, rather than applying a uniform RoPE transformation.

---

#### KV Cache Offloading Architecture

The memory management system extends HiP Attention's offloading approach with three key improvements: dual GPU key banks, an LRU eviction policy, and integration with PagedAttention.

##### The Unified Memory Space

The full KV cache for all context tokens resides in a unified virtual memory space that spans both GPU VRAM and CPU DRAM. The paper uses Nvidia UVM (Unified Virtual Memory) to implement this, which means the operating system and GPU driver handle page migration transparently: when the GPU accesses a memory address corresponding to a page in host memory, a page fault occurs and the page is migrated to GPU memory.

The paper describes the setup: "We manage the KV cache on the unified memory space while keeping a smaller key bank on the GPU memory, which acts as a cache." The total KV cache for 3M tokens with Llama 3.1 8B (FP8 KV cache format) is approximately 192GB, while the GPU bank holds only a small working set — typically the 3K–5K actively selected tokens plus additional caching overhead.

##### Dual GPU Key Banks

InfiniteHiP maintains two separate key caches on GPU memory, unlike HiP Attention which used a single key bank:

1. **Mask-selection key bank:** Stores keys needed during the pruning pipeline execution. When a pruning stage accesses a key that is not in this bank, a UVM page fault triggers a fetch from host memory. The key is placed in the GPU bank, and the page table is updated to map the global key index to the GPU bank slot.

2. **Block sparse attention key bank:** Stores keys needed during the final attention computation. This separation allows different cache policies for the two phases — the keys important for pruning (coarse relevance estimation) may differ from those important for the actual attention dot products (fine-grained token interactions).

**The page table.** A page table mapping global key indices to GPU bank slot indices is maintained in GPU memory. This allows `$O(1)$` lookup of whether a given key token resides on GPU or needs to be fetched. On a cache hit, the GPU bank slot is accessed directly. On a miss, the slot is populated from unified memory and the page table entry is updated.

##### LRU Eviction Policy

The paper states: "Unlike HiP Attention, we employ the Least Recently Used (LRU) policy as the eviction mechanism." When a cache miss occurs and the GPU bank is full, the least recently accessed key token (across all slots) is evicted — its data is written back to unified memory if modified, and its slot is reused for the new key.

**Why LRU over HiP's approach:** HiP Attention tracked "cold" and "hot" tokens based on access frequency during pruning, but the paper found this heuristic-based policy suboptimal. LRU is a simple, well-understood policy that naturally adapts to changing attention patterns: if a chunk of context was important for recent tokens, its keys remain in cache; if attention shifts to a different part of the document, the old keys age out and get replaced.

**Quantifying offloading overhead.** Table 4 shows the "Offload" latency component at various context lengths and cache states. At 256K context with no stage caching ("Cached Stages: None"), offloading takes 2,039 µs out of 9,803 µs total — about 21% of total attention latency. At 1M context with all stages cached, offloading takes 786 µs out of 1,104 µs total — about 71%. This suggests that as context grows, offloading overhead becomes the dominant cost, even with aggressive caching, because the probability of needing tokens from host memory increases.

**Cache hit ratios.** Table 4 reports two hit ratio metrics: "Mask Hit Ratio" (fraction of decoding steps where a pruning stage's cached mask was reused) and "SA Hit Ratio" (fraction of sparse attention key accesses that hit the GPU cache). With all stages cached at 1M context, the SA hit ratio reaches 88.97% for all stages and 99.8% for the final attention computation. This means the LRU policy successfully keeps the actively needed keys on GPU, with only ~11% of pruning key accesses and ~0.2% of attention key accesses requiring host memory fetches.

---

#### Implementation Details: Triton Kernels and GPU Parallelism

The paper provides specific implementation details that bridge the gap between algorithmic description and practical performance.

##### Pruning Stage as a Single Reusable GPU Kernel

The paper states: "We implement a single GPU kernel for the pruning stage, which can be reused for all stages just with different parameters." This means stages 1, 2, and 3 all use the same compiled Triton kernel, parameterized by `$(b_q, l_c, k)$`. The kernel is launched with different grid dimensions and block sizes depending on the stage's input size.

**Why a single kernel:** GPU kernel compilation has non-trivial overhead. By compiling once and using parameterization, the framework avoids kernel launch overhead for stage transitions and allows the Triton compiler to optimize the kernel generically across all use cases.

##### Parallelism Strategy

The key insight enabling fast pruning is that the chunk score estimation and SelectRep operations are **embarrassingly parallel across chunks**. Each chunk's representative token selection and score computation are independent of all other chunks — there is no data dependency between chunk `$j$` and chunk `$j+1$`.

The paper states the algorithm "can be implemented with a single GPU kernel, without any global synchronizations between each iteration, while providing key sequence dimension parallelism like FlashDecode." This directly contrasts with HiP Attention, where "the iterative algorithm involves many global thread synchronizations, which hinders parallelism."

**How FlashDecode-style parallelism works:** The key sequence dimension (all context tokens) is split across GPU thread blocks. Each thread block processes a subset of chunks independently, computing SelectRep tournaments and chunk scores without coordination with other thread blocks. The SelectRep algorithm's internal iterations are handled within a single thread block using shared memory — no global barriers are needed. After all thread blocks complete, a global top-K operation selects the surviving chunks. This final top-K is the only global synchronization point in the entire pruning stage.

##### Block Sparse Attention Implementation

For the final attention computation on selected tokens, the paper implements two kernels:

**Prefill attention:** "We implement a method similar to FlashAttention for prefill." During prefill (processing the input context), the query sequence is long (the full input) and the key sequence is sparse (only selected tokens). The kernel fuses the attention computation in SRAM similar to FlashAttention, but with an irregular memory access pattern due to the sparse key indices.

**Decoding attention:** "We implement Flash Decoding for decoding." During decoding (generating one token at a time), the query is a single token and the key-value cache is large. The kernel splits the key sequence across thread blocks, each computing partial softmax results, then combines them with a final reduction — the standard Flash Decoding approach but applied to a sparse key index set.

##### PagedAttention Integration

The paper integrates PagedAttention (Kwon et al., 2023) "to alleviate the overhead from KV cache memory management." PagedAttention manages the KV cache in fixed-size pages (blocks of tokens) rather than as a contiguous tensor. This enables:
- **No fragmentation:** Pages can be allocated and freed without memory compaction.
- **Efficient sharing:** If multiple sequences share a common prefix, pages can be shared across requests.
- **Compatibility with offloading:** Pages can be moved between GPU and CPU memory individually, enabling fine-grained migration.

##### UVM and Graph Capture

The paper notes a specific optimization: "our UVM implementation makes the KV cache offloaded attention mechanism a graph-capturable operation, which allows us to avoid CPU overheads." CUDA graph capture records a sequence of kernel launches and memory operations into a single replayable graph, eliminating per-step kernel launch overhead — which is particularly important during decoding where the same sequence of operations repeats hundreds or thousands of times.

In contrast, InfLLM's offloading approach cannot be graph-captured because it involves CPU-side decisions about which pages to migrate, introducing per-step CPU-GPU synchronization that prevents graph reuse.

##### Throughput Benchmarks

Figure 5 and Tables 11–12 present end-to-end decoding throughput in the SGLang serving framework. Key numbers:

- **RTX 4090 (24GB) at 1M context:** InfiniteHiP 3K-Fast achieves 97.0 tok/s vs. estimated 12.5 tok/s for SRT (SGLang Runtime with FlashInfer) — a 7.76× improvement. With offloading enabled, this drops to 17.3 tok/s (3K-Fast Offload) and 40.1 tok/s (3K-Flash Offload).

- **L40S (48GB) at 3M context:** InfiniteHiP 3K-Fast achieves estimated 66.4 tok/s vs. estimated 3.3 tok/s for SRT — a 20.1× improvement. With offloading, 3K-Flash achieves 23.8 tok/s.

The Flash configuration achieves its speedup by reducing the mask refresh frequency: Stage 1 is refreshed only every 96 steps instead of every 16, meaning the expensive `$O(T)$` initial pruning stage runs 6× less frequently.

##### Prefill Latency Details

Table 3 reports prefill latency using "chunked prefill style attention, with a chunk size of 32K." At 1M context:
- FA2: 3,490 ms
- InfLLM (12K window): 183 ms
- HiP (1K window): 147 ms
- InfiniteHiP (3K, no Extend): 172 ms
- InfiniteHiP (3K, with Extend): 276 ms

The InfiniteHiP prefill is 20.29× faster than FA2. The breakdown shows that Stage 0 (first pruning stage) consumes 53.7% of total prefill latency at 1M tokens without extension, confirming that the initial coarse filtering dominates prefill cost. Stages 1 and 2 each consume ~9–13%, and the block sparse attention consumes 13.9%. The "Extra" component (10.1%) accounts for overhead from kernel launches, memory management, and other housekeeping.

##### Decoding Latency Details

At 1M context decoding (Table 3):
- FA2: 4,645 µs per step
- InfLLM: 1,222 µs (3.80× faster)
- HiP: 450 µs (10.3× faster than FA2)
- InfiniteHiP (3K): 234 µs (19.85× faster than FA2, 4.98× faster than InfLLM, 1.92× faster than HiP)
- InfiniteHiP (3K, with Extend): 245 µs (still 18.96× faster than FA2)

The mask caching mechanism is responsible for the large improvement over HiP. Table 3 shows that without any mask caching ("Total AR" — autoregressive, always refreshing), InfiniteHiP's decoding latency is 936 µs at 1M context. With caching enabled (the default "Total" row), this drops to 234 µs — a 4× reduction from caching alone. This is the key architectural difference from HiP, which did not have per-stage cache control.

##### The Sink/Streaming Token Special Handling

The paper treats sink and streaming tokens as a special bypass: "the initial `$n_{\text{sink}}$` tokens (sink tokens) and `$n_{\text{stream}}$` most recent tokens (streaming tokens) are always included." In Algorithm 1, line 1, the initial key indices `$I^{(0)}_m$` for the first pruning stage are constructed as `$[n_{\text{sink}}, \ldots, b_q^{(1)} \cdot m - n_{\text{stream}}]$` — the middle tokens between the sink region and the streaming region (minus the causal window). The sink tokens (indices 0 through `$n_{\text{sink}} - 1$`) and the streaming tokens (indices near the current query position) are excluded from the pruning input because they will be unconditionally added to the final sparse attention mask.

This means the pruning pipeline only operates on the "middle" context — everything between the fixed attention anchors at the beginning and the temporally recent tokens at the end. For a 1M-token context, this means ~998,720 tokens go through the pruning pipeline while 1,280 tokens (256 sink + 1024 streaming) are always retained. The final sparse attention mask includes the `$k^{(3)} = 2,048$` or 4,096 tokens from pruning plus the 1,280 always-included tokens, for a total window of ~3.3K–5.4K tokens.

---

#### Summary of Design Choices and Their Justifications

- **Three-stage hierarchical pruning** over single-stage: amortizes the cost of fine-grained selection; the first stage's `$O(T)$` cost runs only periodically due to caching; stages 2 and 3 have `$O(1)$` cost independent of context length.

- **Max-pooling across heads and query positions** for chunk scores: ensures conservative retention (minimizing false rejections) because block sparsity operates at chunk granularity.

- **Per-head independent representative token selection** over shared representatives: different heads have different positional and content preferences; independent selection preserves head specialization.

- **SelectRep's log-time tournament** over exhaustive per-chunk evaluation: exploits the locality observation that nearby tokens have similar attention scores, reducing `$l_c$` evaluations to `$2 \log_2(l_c)$`.

- **Single reusable Triton kernel for all pruning stages** over separate kernels: reduces compilation overhead and enables parameterized tuning.

- **Sink tokens (256) and streaming tokens (1024) never pruned:** follows StreamingLLM's insight that these tokens are disproportionately important for model coherence and stability.

- **Per-stage mask caching with staggered refresh intervals** over uniform caching: expensive Stage 1 refreshed least often (every 16/32/96 steps), cheap Stage 3 refreshed most often (every 4/8/8 steps), maximizing speed with minimal accuracy loss.

- **Chunk-indexed RoPE in layers 1–3, Relative RoPE in layers 4–32:** early layers exhibit sliding-window attention patterns that require positional information for local representation building; later layers process long-range dependencies using learned semantic content.

- **StreamingLLM-style RoPE for block sparse attention:** preserves relative ordering of selected tokens while compressing position IDs to a dense sequential range, avoiding out-of-distribution positions.

- **Dual GPU key banks (mask-selection + block sparse attention):** separates caching policies for the two phases; the reconstruction of mask-selection keys differs from the frequency of attention computation accesses.

- **LRU eviction** over HiP's frequency-based heuristic: simple, well-understood, naturally adapts to shifting attention patterns.

- **UVM with CUDA graph capture** over manual page migration: transparent page fault handling from the GPU driver, plus elimination of per-step kernel launch overhead through graph replay.

- **PagedAttention integration:** eliminates memory fragmentation from variable-length sparse key selections and enables page-level offloading granularity.

## 4. Key Insights and Innovations

### Innovation 1: Attention Sparsity Is Best Exploited at the Chunk Level, Not the Token Level

The paper's foundational conceptual move is the recognition that attention sparsity in long-context LLMs operates at a **chunk granularity** that makes hierarchical, block-based pruning both more efficient and more accurate than token-level selection. This is not merely an engineering convenience — it reflects a structural property of how attention distributes mass across long sequences.

**What the field assumed before this work:** Prior dynamic token selection methods, including the strongest baselines InfLLM and HiP Attention, implicitly treated the problem as one of *individual token ranking* — which tokens have the highest attention scores? — and designed algorithms to approximate this ranking. InfLLM pre-selects representative tokens per block but uses them only as proxies for block-level importance; the final selection still produces a set of individual tokens. HiP performs hierarchical token-level top-k estimation, narrowing from all tokens to the final selection through multiple rounds of individual token comparisons. The implicit assumption was that token-level granularity was necessary for preserving attention fidelity — that dropping entire chunks would inevitably discard crucial tokens that happen to sit in otherwise-irrelevant blocks.

**What InfiniteHiP recognizes differently:** The empirical observation in Section 3 (Figure 2a) shows that the top-k attention tokens are *not* uniformly distributed across the context. Instead, they concentrate heavily: fewer than 2% of 64-token chunks contain more than 12.5% of the top-2K tokens, while 75% of chunks contain *zero* top-2K tokens. This distribution implies that chunk-level filtering is not a crude approximation — it is an *information-theoretically appropriate* response to the actual structure of attention. If three-quarters of chunks are completely irrelevant, there is no penalty for discarding them wholesale; the only challenge is identifying which quarter to keep. The paper reframes the problem from "find the top-k individual tokens" to "find the top-K chunks, then refine within them" — a shift from a ranking problem to a *set retrieval problem* at the coarse stage, with ranking deferred to the fine stage.

**Why this is distinguished from prior chunk-based methods:** InfLLM also operates on chunks, but its representative tokens are static — precomputed once and reused for all queries. This means InfLLM's chunk-level decisions are approximate in a way that compounds errors: a chunk might be discarded because its static representative happens to be a low-attention token, even though a different token in that chunk would score highly for the current query. InfiniteHiP's SelectRep algorithm dynamically selects a query-dependent representative for each chunk *and* for each attention head independently, making the chunk retention decision substantially more accurate. The evidence in Figure 6a confirms this: InfiniteHiP achieves 1.57 percentage points higher recall of attention probabilities than InfLLM, meaning it discards fewer chunks that contain genuinely important tokens.

**Significance beyond performance:** This insight reframes the design space for sparse attention. If chunk-level sparsity is a fundamental property rather than an artifact of a particular model or context length, then future work on efficient attention should focus on *chunk selection algorithms* (how to rapidly identify relevant chunks) rather than *token ranking algorithms* (how to sort individual tokens by score). The paper's modular pruning pipeline — where Stage 1 identifies coarse regions of interest, Stage 2 narrows within those regions, and Stage 3 pinpoints specific tokens — is a direct architectural expression of this insight. It suggests a general template for attention approximation: match the algorithm's granularity to the natural clustering scale of the attention distribution.

There is also a subtle connection to the **locality assumption** that the paper inherits from HiP. The SelectRep algorithm works because nearby tokens tend to have similar attention scores — but this is *the same phenomenon* that creates chunk-level sparsity in the first place. If attention scores varied randomly from token to token, chunks would not have correlated importance and chunk-level pruning would fail. The fact that it works confirms that attention possesses a characteristic spatial correlation length, and that this length is substantially larger than individual tokens — making chunks of 8–256 tokens a natural unit of analysis.

---

### Innovation 2: The Pruning Pipeline Is Modular and Parallelizable by Design, Not by Optimization

Where HiP Attention achieved hierarchical pruning through a single iterative algorithm with internal top-k operations that required global thread synchronizations, InfiniteHiP decomposes the pruning process into **discrete, reusable stages** whose internal parallelism is explicitly designed for GPU execution. This changes the nature of the contribution from "a better pruning heuristic" to "a pruning architecture that can be efficiently implemented."

**What the field assumed before:** The dominant approach to hierarchical attention pruning — exemplified by HiP — treated the algorithm as a monolithic procedure. The hierarchical refinement was encoded in the control flow of a single algorithm (iterative narrowing with feedback between rounds), making it difficult to parallelize because each iteration depended on the results of the previous one, and because internal top-k operations required sorting or selection across the entire key dimension, creating global synchronization points. The paper explicitly names this weakness: HiP's "iterative algorithm involves many global thread synchronizations, which hinders parallelism." The field implicitly accepted that hierarchical pruning would have limited parallelism as an inherent cost of its multi-round structure.

**What InfiniteHiP does differently:** The paper reimagines the pruning pipeline as a **stack of independent, stateless modules**, each implementing the same algorithmic template (Algorithm 2) but with different parameters. The stages are connected only by their input/output key index sets — there is no control flow coupling between them. Critically, the SelectRep algorithm within each stage runs entirely within a single GPU thread block for each chunk, with no cross-chunk communication needed until the final top-K selection across chunks. This means the vast majority of the computation is embarrassingly parallel across chunks and across query blocks.

**The parallelism insight is threefold:**

1. **Intra-stage parallelism:** All chunks within a pruning stage are independent. The SelectRep tournament for chunk *j* does not depend on chunk *j+1*. This allows the GPU to process thousands of chunks simultaneously, with each thread block handling one or a few chunks entirely in shared memory. The paper explicitly contrasts this with FlashDecode-style parallelism, where the key sequence dimension is split across thread blocks — InfiniteHiP achieves the same scaling pattern.

2. **Inter-stage independence combined with caching:** Because each stage's output is a self-contained sparse index set, the stages can be cached independently with different refresh rates. This is not a minor implementation detail — it is an architectural property enabled by the modular design. Stage 1's output mask, once computed, is a valid input to Stage 2 for many subsequent decoding steps, even as Stage 3 is refreshed more frequently. This decoupling is what enables the 4× speedup from caching (Table 3: 936 µs without caching vs. 234 µs with caching at 1M context).

3. **Single-kernel reusability:** Because all stages implement the same algorithmic template, they can share a single compiled Triton kernel parameterized by chunk size, query block size, and retention count. This eliminates kernel compilation overhead and allows the Triton compiler to optimize generically across all pruning stages, which is an engineering insight with practical performance implications.

**Why this distinguishes InfiniteHiP from HiP, its direct predecessor:** HiP's hierarchical pruning was already conceptually multi-stage, but it was implemented as a single iterative procedure. InfiniteHiP's contribution is recognizing that making the stage boundaries *explicit and independent* — turning them from iterations of a loop into discrete modules with their own caching policies — unlocks parallelism and performance that the monolithic design could not achieve. This is a **systems architecture insight** masquerading as an algorithmic refinement. The 92% decoding speed improvement over HiP (234 µs vs. 450 µs per step in Table 3) is substantially attributable to this architectural change, not just to better token selection accuracy.

**Significance for future work:** The modular pipeline design establishes a template that can be extended or reconfigured. The paper hints at this in Appendix G: "we discovered numerous module design choices during our research. For example, increasing block sizes can reduce latency in masking... However, this comes at a cost of performance loss in NLU tasks." The implication is that the pipeline is *composable* — stages can be added, removed, or reparameterized for different task profiles without redesigning the entire system. The "3K," "5K," "fast," and "flash" presets are not just hyperparameter choices; they are evidence that the modular architecture supports meaningful reconfiguration without code changes.

---

### Innovation 3: Layer-Specific RoPE Adjustment Is Necessary Because Early and Late Layers Use Position Information Differently

The paper makes a diagnostic observation that leads to a non-obvious design choice: **the first few layers of a Transformer exhibit qualitatively different attention patterns from deeper layers**, and this difference demands different positional encoding strategies for out-of-length generalization. This moves beyond the dominant paradigm of applying a uniform RoPE adjustment across all layers.

**What the field assumed before:** Prior work on RoPE extension for OOL generalization — whether NTK-aware scaling, Self-Extend, or LM-Infinite — applied a single adjustment strategy uniformly to all layers and all attention heads. The implicit assumption was that position encoding is a monolithic property of the model: either the model can handle extended positions or it cannot, and the same correction factor should apply everywhere. Even methods like StreamingLLM that combine sink/streaming tokens with RoPE adjustments do not vary the adjustment by layer.

**What InfiniteHiP observes:** Appendix D and Figure 10 present a layer-by-layer analysis of attention patterns in Llama 3.1 8B. The finding is stark: early layers (roughly 1–5, with layers 1–3 selected as the cutoff) exhibit "dynamic sliding window-like attention" — they attend primarily to tokens at specific relative distances from the current position, creating diagonal banded patterns in the attention matrix. This signifies that these layers are **building local positional representations** — they need to know that token A is approximately N positions from token B to establish syntactic and local semantic structure.

Later layers, in contrast, show much more diffuse attention patterns. Once the early layers have constructed position-aware representations, deeper layers can "efficiently process long-range information... by leveraging learned semantics instead of positional cues." These layers benefit more from content-based matching than from precise positional information.

**The design consequence:** This observation directly motivated the layer-specific RoPE assignment. Layers 1–3 receive Chunk-indexed RoPE, which preserves coarse relative position information by assigning each context chunk a position ID based on its chunk index offset from the query. This maintains the sliding-window structure that early layers rely on. Layers 4–32 receive Relative-style RoPE, where position IDs during the SelectRep algorithm are simply binary offsets (left branch = offset + 1, right branch = offset) — eliminating absolute position entirely and focusing the pruning decision on content relevance.

**Evidence that this matters:** Table 6 in Appendix D provides the crucial ablation: using Relative RoPE in all layers yields 68.55% on ∞Bench En.MC; using Chunk-indexed RoPE in layers 1–3 and Relative RoPE in layers 4–32 yields 74.23%. This is a 5.68 percentage point improvement — substantial for a single design choice. Table 5's broader ablation shows that the choice of RoPE style in both pruning and block sparse attention matters significantly, with the RT/ST combination achieving 70.31% while the worst combination (DE/DE) achieves only 52.40%. These gaps are not marginal — they represent the difference between usable and unusable long-context performance.

**Why this is more than an engineering trick:** The layer-specific RoPE strategy reflects a **diagnostic understanding of Transformer internals** that goes beyond the standard "attention heads specialize" observation. It suggests that positional information is not uniformly important throughout the model, and that the *type* of positional information needed changes with depth. This has implications beyond the specific RoPE strategies used here: it implies that future OOL generalization methods should consider layer-specific interventions as a first-class design dimension, and that analyzing per-layer attention patterns (as in Figure 10) should be a standard diagnostic before choosing a position encoding extension strategy.

The connection to the pruning pipeline is also significant: the Chunk-indexed RoPE in early layers creates sliding-window-like artifacts in the pruning masks (visible in Figure 9b), which helps the block sparse attention approximate the diagonal attention patterns that early layers naturally exhibit (as discussed in Appendix D and Figure 11). This means the RoPE strategy and the pruning architecture are **co-designed**: the positional encoding choice influences which chunks get selected during pruning, and the block sparse attention must then faithfully reproduce the attended tokens. The paper's careful ablation of RoPE styles in *both* pruning and attention (Table 5) acknowledges that these are coupled design decisions, not independent knobs.

**A limitation the paper acknowledges:** The 1–3 layer cutoff is empirically determined for Llama 3.1 8B. The paper does not claim universality — different model architectures or sizes may exhibit different layer-wise attention patterns. Appendix G notes that "the combination of pruning modules should be studied more in future research," implicitly acknowledging that the layer-specific RoPE strategy may need recalibration for other models. However, the *diagnostic methodology* — analyze per-layer attention patterns, identify which layers need positional information, assign RoPE strategies accordingly — is the transferable contribution, not the specific cutoff.

---

### Innovation 4: KV Cache Offloading Can Be Practical If the Pruning Algorithm Informs the Caching Policy

The paper integrates KV cache offloading with the sparse attention mechanism in a way that makes long-context inference on consumer GPUs genuinely practical, transforming offloading from a theoretical possibility into a deployed system. The key insight is that **the pruning pipeline's access patterns can inform an intelligent caching policy**, making the latency cost of host-memory access tolerable.

**What the field assumed before:** KV cache offloading — storing most of the context in CPU DRAM and fetching to GPU on demand — was widely recognized as necessary for long contexts that exceed GPU memory, but was considered impractical for two reasons. First, the bandwidth gap: PCIe latency for CPU memory access is approximately 31.5× higher than VRAM access (quantified in the paper's analysis). Second, naive offloading would require fetching tokens unpredictably, making it impossible to hide latency through prefetching or to benefit from GPU graph capture. InfLLM avoided this by restricting all data to GPU before kernel execution — accepting larger context windows (12K tokens) and coarser selection to avoid runtime memory access. HiP Attention pioneered UVM-based offloading but lacked a sophisticated caching policy, resulting in suboptimal cache hit rates.

**What InfiniteHiP recognizes:** The pruning pipeline provides a **predictable, structured access pattern** that can be exploited by a caching layer. The pipeline processes the full context only in Stage 1 (which runs infrequently due to mask caching), and subsequent stages access only the surviving 32K → 8K → 3K tokens. The LRU eviction policy naturally retains these surviving tokens in GPU memory because they are repeatedly accessed across decoding steps — a chunk that survives Stage 1 for step *t* is likely to survive Stage 1 for step *t+1* (due to temporal locality), so its keys stay in the GPU cache.

More subtly, the **dual GPU key bank** design separates the caching policies for the mask-selection phase and the block sparse attention phase. The keys needed for pruning (coarse relevance estimation, which uses representative tokens) may differ from the keys needed for final attention computation (all tokens in selected chunks). By maintaining separate caches with independent LRU tracking, each phase can optimize its resident set without interference.

**Evidence of practicality:** Table 4 demonstrates that offloading is not just possible but efficient. At 1M context with all pruning stages cached, the total attention latency is 1,104 µs — of which only 786 µs is offloading overhead, and the sparse attention hit ratio reaches 98.38% for Stage 2 and 88.97% for the final attention stage. This means the GPU cache is successfully retaining the actively needed tokens, with relatively few expensive host-memory round-trips. The mask hit ratio of 98.38% means that nearly all decoding steps reuse cached pruning masks, avoiding the need to access host memory for coarse filtering entirely.

The end-to-end throughput numbers in Figure 5 cement the practicality claim: on an L40S 48GB GPU at 3M context, InfiniteHiP with offloading achieves 23.8 tok/s in the Flash configuration. While this is slower than the 66.4 tok/s achievable without offloading (which would require ~192GB of VRAM — impossible on the hardware), it represents a 7.25× improvement over the estimated SGLang runtime baseline, which cannot handle 3M tokens at all without offloading. In other words, offloading is not just a fallback — it is what makes 3M-token inference *possible at all* on this hardware.

**The UVM + graph capture insight:** The paper's implementation note that "our UVM implementation makes the KV cache offloaded attention mechanism a graph-capturable operation" is technically specific but conceptually important. CUDA graph capture eliminates per-step kernel launch overhead, which is particularly valuable during decoding where the same sequence of kernels executes hundreds or thousands of times. InfLLM's offloading approach cannot be graph-captured because it involves CPU-side page migration decisions, creating a CPU-GPU synchronization point on every step. InfiniteHiP's design keeps all cache management decisions on the GPU (the page table and LRU tracking are GPU-resident), enabling the entire attention mechanism — pruning, offloading, sparse attention — to be captured as a single replayable graph.

**Significance beyond this paper:** The integration of pruning-informed caching with UVM offloading establishes a design pattern for memory-constrained LLM inference: use the attention selection mechanism itself to predict which tokens will be needed, maintain a GPU cache of those tokens, and fall back to host memory only for the long tail of rarely-accessed context. This pattern extends beyond the specific pruning algorithm used here. Any sparse attention method that produces predictable access patterns (e.g., retrieval-augmented attention, chunk-based selection) could adopt the dual-bank LRU caching layer. The paper's contribution is demonstrating that this pattern works at scale on real hardware with real models, not just in simulation.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** Three benchmarks are used: (1) **LongBench** (Bai et al., 2023), with sequences averaging ~32K tokens, covering single-document QA, multi-document QA, summarization, few-shot learning, synthetic, and code tasks; (2) **∞Bench** (Zhang et al., 2024), with sequences over 100K tokens, covering synthetic retrieval tasks (passkey, number string, KV retrieval, math find) and natural language understanding (multiple-choice QA, summarization); (3) **RULER** (Tables 8–9, Appendix E.2), a benchmark for evaluating long-context retrieval capabilities across varying context lengths (4K–512K tokens) and task subtypes. The Passkey retrieval task is also used for latency measurements in Table 4 and throughput benchmarks.

- **Base model(s).** Primary experiments use instruction-tuned **Llama 3 8B** (Llama Team, 2024, 8K pretrained context length) and instruction-tuned **Mistral 0.2 7B** (Jiang et al., 2023, 32K pretrained context length). Additional evaluations extend to **Llama 3.1 8B** (128K pretrained context length) for latency benchmarks and ∞Bench, **Gemma2 9B** (8K pretrained context length), **EXAONE 3 7.8B** (4K pretrained context length), **EXAONE 3.5 7.8B** (32K pretrained context length), and **DeepSeek R1 Distilled Qwen2 14B** (128K pretrained context length) for out-of-length generalization testing. The paper argues these models are "representative of the capabilities of many contemporary LLMs" and cover a range of pretrained context lengths relevant for probing OOL generalization.

- **Metrics.** The primary metrics are **task-specific accuracy scores** as defined by each benchmark (e.g., F1 for QA tasks, ROUGE-L for summarization, exact match for multiple-choice). For aggregate comparison across heterogeneous tasks within a benchmark, the paper computes an **"Average Relative" score**: each subset's raw score is divided by the highest score achieved by any method on that subset (normalizing to a 0–100 scale), and these normalized scores are averaged across all subsets. The paper states this "better represents the differences in performance because the variance is normalized per subset." For latency benchmarks, **microseconds (µs) per attention operation** and **tokens per second (tok/s) throughput** are reported, with detailed per-stage breakdowns. For memory, **GPU VRAM usage in GB** is measured.

- **Baselines.** Eight baselines are compared, falling into several categories: **(1) Truncated FA2:** FlashAttention2 with context truncated in the middle to fit within the pretrained context length (standard dense attention baseline). **(2) RoPE extension methods:** Dynamic-NTK (bloc97, 2023) and Self-Extend (Jin et al., 2024) — both apply RoPE adjustments for OOL generalization while using dense FlashAttention2 on the full untruncated context. **(3) Sink + streaming approaches:** LM-Infinite (Han et al., 2024, denoted "Infinite" in tables) and StreamingLLM (Xiao et al., 2024b, denoted "Streaming") — both use fixed attention to initial sink tokens plus recent streaming tokens, combined with RoPE adjustments for OOL generalization. **(4) KV cache eviction:** H2O (Zhang et al., 2023) — retains top-k heavy-hitter KV tokens adaptively during decoding, discarding the rest permanently. **(5) Query-aware dynamic selection:** InfLLM (Xiao et al., 2024a) — divides context into blocks with static representative tokens, selects top-k blocks per query, evaluated at window sizes of 8K (Llama 3) or 6K/12K/16K (Mistral 0.2). **(6) Hierarchical pruning:** HiP Attention (Lee et al., 2024b) — the direct predecessor, evaluated at a 1K window size on Llama 3.

- **Generation budget / compute accounting.** The paper does not measure "generations" as in a sampling budget. Instead, the relevant compute metric is the **number of key tokens retained in the sparse attention window** — this determines both arithmetic computation (dot products scale linearly with retained tokens per query) and memory bandwidth consumption. InfiniteHiP is evaluated at two window presets: **3K** (retaining ~3,000 tokens: 2,048 from pruning plus 256 sink + 1,024 streaming) and **5K** (retaining ~5,000 tokens: 4,096 from pruning plus sink/streaming). These are compared against baselines at their respective window sizes (InfLLM at 8K/12K/16K, HiP at 1K, FA2 at the full truncated context). Latency experiments use a **1M-token context** for controlled comparison. The **mask refresh interval** is an additional compute parameter: "fast" uses (32, 16, 8) refresh intervals for stages 1–3, "flash" uses (96, 24, 8), trading accuracy for speed.

- **Cross-validation / statistical protocol.** No cross-validation or statistical significance testing is reported. Results are single-run evaluations on the standard benchmark test sets. The paper does not report confidence intervals, standard deviations, or multiple random seeds. For the ∞Bench En.MC scores used in the RoPE ablation study (Table 5), the context is truncated at 128K for controlled comparison.

---

### Main Quantitative Results

#### LongBench: Overall Quality at Moderate Context Lengths (~32K)

The headline result in Table 1 is that InfiniteHiP achieves the highest **Average Relative** score on LongBench for both model families, outperforming all baselines while processing substantially fewer key tokens.

**Llama 3 8B results.** InfiniteHiP with the 3K window achieves 47.72% absolute average (100.00% relative score), compared to the strongest baseline InfLLM at 44.47% absolute (92.83% relative) — a 7.17 percentage-point relative improvement. The closest baseline overall is the dense FA2 truncated baseline at 42.47% absolute (87.69% relative). What makes this significant is not just the score but the efficiency: InfiniteHiP selects only 3K tokens for attention while InfLLM selects 8K — a 2.67× reduction in key tokens. The gain over HiP Attention (1K window) is even more dramatic: HiP achieves 40.62% absolute (83.23% relative), meaning InfiniteHiP achieves roughly 7 percentage points higher absolute accuracy than its direct predecessor while processing 3× more retained tokens (suggesting HiP's 1K window was too aggressive).

Breaking down by task category: InfiniteHiP substantially outperforms all baselines on Summarization (31.7 vs. 30.8 for InfLLM), Multi-Document QA (MFQA: 50.3 vs. 49.0, HQA: 51.9 vs. 49.0), and Synthetic tasks (7.5 vs. 7.2 for the top baseline on PCount, but dramatically better on PR: 93.5 vs. 84.0 for InfLLM and RBP: 64.8 vs. 46.5 — these are passkey/variable-tracking-style retrieval tasks where query-aware dynamic selection excels). On Few-shot Learning (TREC: 75.5 vs. 73.5, TQA: 90.3 vs. 90.9 — essentially tied) and Code (LCC: 63.1 vs. 59.9), InfiniteHiP holds or slightly leads. The one notable exception is Single Document QA on Qasper, where InfLLM scores 43.7 vs. InfiniteHiP's 43.2 — a negligible difference.

**Mistral 0.2 7B results.** The pattern is similar but with somewhat smaller margins. InfiniteHiP with 3K window achieves 42.71% absolute (99.85% relative), compared to InfLLM at 39.46% absolute (91.23% relative) at 6K window and 41.46% absolute (96.99% relative) at 12K window. This means InfiniteHiP outperforms even InfLLM's 12K configuration by roughly 1.25 absolute percentage points while using 4× fewer key tokens. Against the dense FA2 baseline (41.29% absolute, 96.44% relative), InfiniteHiP achieves a small but consistent improvement. Against LM-Infinite (36.76%) and StreamingLLM (36.41%), the margin is substantial — approximately 6 percentage points absolute — confirming the value of dynamic over static token selection.

The per-task gains for Mistral are particularly notable on Multi-Document QA (HQA: 40.4 vs. InfLLM-12K at 39.5, MSQ: 23.2 vs. 18.9 — a 4.3 percentage-point improvement) and Summarization (QMSum: 23.8 vs. 23.8 — tied, but with the note that InfiniteHiP matches InfLLM-12K at ~4× fewer tokens). On the synthetic retrieval tasks (RBP: 62.1 vs. 52.1 for InfLLM-12K), InfiniteHiP shows consistently stronger retrieval from the full uncompressed context.

**Interpretation.** The LongBench results establish that InfiniteHiP's dynamic, query-aware pruning is not just possible but *better* than the best prior method (InfLLM) while being substantially more efficient. The improvement is not uniform — it is largest on tasks requiring precise retrieval from specific positions in long contexts (synthetic tasks like passkey retrieval, multi-document QA), while on tasks like few-shot learning where the model may rely more on local patterns near the query, the margin is smaller. This aligns with the paper's motivation: the pruning pipeline is designed to excel at identifying and retaining *relevant* context chunks, which matters most when the answer depends on specific facts buried in a long document.

#### ∞Bench: Performance at Extended Context Lengths (>100K)

The headline result in Table 2 is more dramatic than LongBench, because ∞Bench's longer contexts (over 100K tokens) create both a harder retrieval challenge and a harder OOL generalization challenge — and InfiniteHiP excels at both.

**Llama 3 8B results.** InfiniteHiP with the 3K window achieves 46.25% absolute average (98.17% relative score), compared to InfLLM at 43.05% absolute (89.07% relative) — a 9.99 percentage-point relative improvement. The 5K window preset pushes this further to 99.69% relative score (47.08% absolute), demonstrating that increasing the retained token budget yields additional gains. The "3K-fast" and "3K-flash" variants achieve near-identical quality (98.35% and 97.78% relative respectively), validating that mask caching with longer refresh intervals negligibly impacts accuracy while significantly improving throughput.

The most striking comparisons are against methods that lack OOL generalization. FA2 truncated to 8K scores 47.83% relative — meaning it performs less than half as well as InfiniteHiP when forced to discard all but the last 8K tokens of a 100K+ context. NTK-aware RoPE extension on the full 128K context collapses to 3.65% relative, confirming that naive frequency scaling fails catastrophically at these lengths. Self-Extend achieves 67.81% relative — significantly better than NTK but well below InfiniteHiP. LM-Infinite (42.52%) and StreamingLLM (42.53%) show that static sink + streaming approaches cannot effectively retrieve information from arbitrary positions in very long contexts. H2O's KV cache eviction scores only 3.95% relative — essentially complete failure — confirming the paper's claim that permanently discarding tokens is fatal for long-range retrieval.

**Mistral 0.2 7B results.** The pattern holds. InfiniteHiP with 5K window achieves 99.09% relative score (54.96% absolute), compared to InfLLM-16K at 94.77% relative. With the 3K window, InfiniteHiP achieves 94.44% relative — slightly below InfLLM-16K but with roughly 5.3× fewer retained tokens. The degradation from 5K to 3K on Mistral (94.44 vs. 99.09 relative) is more significant than on Llama 3, suggesting that Mistral may benefit more from a larger retained context or that the optimal pruning hyperparameters differ between model families.

**Synthetic vs. NLU breakdown.** Table 2 provides the crucial decomposition into synthetic tasks (RPK: passkey retrieval, RN: number string, RKV: KV retrieval, MF: math find) and NLU tasks (MC: multiple-choice QA, QA: question answering, SUM: summarization). The synthetic tasks reveal where InfiniteHiP's advantage comes from:

- **RPK (Retrieve PassKey):** InfiniteHiP achieves 99.83% (3K), essentially perfect, compared to InfLLM at 100% — effectively tied. Even LM-Infinite achieves only 6.80%. This task is the classic "needle in a haystack" test, and InfiniteHiP passes it near-perfectly while methods relying on static windows or eviction fail.
- **RKV (Retrieve KV):** InfiniteHiP achieves 9.60% (3K) and 10.80% (5K), compared to InfLLM at 5.00%. The absolute numbers are low (this is the hardest synthetic task), but InfiniteHiP roughly doubles InfLLM's performance — a meaningful relative gain.
- **MF (Math Find):** InfiniteHiP scores 17.71% — notably *lower* than InfLLM's 23.70%. This is the one synthetic task where InfiniteHiP underperforms, and it suggests that the aggressive pruning may be discarding tokens relevant for multi-step mathematical reasoning embedded in long contexts. The paper does not analyze this failure case explicitly.

For NLU tasks, the advantage is more mixed. On En.MC (multiple-choice QA with English), InfiniteHiP-3K achieves 57.21% vs. InfLLM's 43.70% — a 13.5 percentage-point absolute improvement, the largest single-task gain. On QA, InfiniteHiP scores 26.94% vs. 19.50% for InfLLM. On summarization, InfLLM holds a slight edge (29.17% vs. InfiniteHiP-3K's 36.35% — wait, rechecking: the values are 29.17 for InfLLM and 36.35 for InfiniteHiP-3K, meaning InfiniteHiP actually leads here as well on Llama 3). The pattern suggests InfiniteHiP's dynamic selection benefits both retrieval-heavy and reasoning-heavy tasks compared to baselines that lack OOL generalization capability.

**Out-of-length generalization with short-context models (Figure 4).** The experiments on EXAONE 3 (4K pretrained), EXAONE 3.5 (32K pretrained), and Gemma2 (8K pretrained) are perhaps the most compelling evidence for InfiniteHiP's OOL generalization capability. These models were never trained on sequences longer than their pretrained windows, yet InfiniteHiP enables them to process contexts up to 192K–256K tokens with *increasing* accuracy as context grows. For Gemma2 on En.MC: accuracy rises from 37.55% at 4K to 72.49% at 256K — a 34.94 percentage-point gain. For comparison, FA2 with Gemma2 achieves 42.36% at 4K but cannot process sequences longer than 8K (OOL). The fact that performance *improves* with longer contexts (rather than plateauing or degrading) provides strong evidence that the model can genuinely leverage the additional context when InfiniteHiP handles the positional encoding and retrieval.

#### RULER: Retrieval and Length Generalization (Appendix E.2)

Tables 8 and 9 present RULER results on Llama 3.1 8B across context lengths from 4K to 512K. The takeaway is nuanced:

**Average performance across lengths (Table 8).** InfiniteHiP with the 16K-shallow configuration achieves the highest overall average at 78.89%, followed by the 5K config at 78.37%. FA2 achieves 66.89% — but critically, FA2 scores 0.00% at 512K and 256K because it cannot process contexts beyond the pretrained 128K window. HiP Attention similarly scores 0.00% beyond the pretrained window, reaching only 57.34% overall. Both FA2 and HiP achieve strong scores within the pretrained range (e.g., FA2 at 8K: 96.05%; HiP at 8K: 95.90%) but collapse at longer lengths.

This reveals a fundamental tradeoff: methods that work within the pretrained context window (FA2, HiP) achieve higher peak quality than InfiniteHiP when evaluated within their operational range (FA2 at 8K: 96.05% vs. InfiniteHiP-3K at 8K: 96.70% — actually InfiniteHiP is slightly better here, but at 4K: FA2 is 96.05% vs. InfiniteHiP at 96.26% — effectively tied). The value proposition of InfiniteHiP is not better quality at short lengths but rather the ability to *extend* reasonable quality to lengths 4–64× beyond the pretrained window, where other methods fail entirely.

**Per-subset breakdown (Table 9).** The NIAH (Needle In A Haystack) variants reveal where InfiniteHiP excels: NIAH Single Key variants (NIAH1/2/3 SK) score 90.8–99.5%, near-perfect retrieval across all configurations. The Multi-Key and Multi-Value variants (NIAH MK, MV, MQ) show more degradation, particularly NIAH3 MK (32.5%) and NIAH2 MK (42.3%) — these tasks require tracking multiple pieces of information across the context, and aggressive pruning likely discards some of the relevant tokens. The Variable Retrieval (VR) and Common/Frequent Word Extraction (CWE, FWE) tasks show intermediate performance (82–98%), suggesting that pruning maintains semantic access patterns for natural language tasks better than for synthetic multi-hop tracking.

#### Attention Latency: Speed Comparisons (Tables 3 and 4)

The latency experiments in Section 5.3 quantify the computational efficiency that the pruning pipeline achieves.

**Prefill latency at 1M context (Table 3).** With "chunked prefill style attention" (32K chunk size):
- FA2 (1M window): 3,490 ms
- InfLLM (12K window): 183 ms (19.1× faster than FA2)
- HiP (1K window): 147 ms (23.7× faster than FA2)
- InfiniteHiP (3K window, no RoPE extend): 172 ms (20.3× faster than FA2)
- InfiniteHiP (3K window, with RoPE extend): 276 ms (12.6× faster than FA2)

The RoPE extension overhead is quantified: it adds roughly 60% to prefill latency (172 → 276 ms), attributed to "additional memory reads of precomputed cos and sin vectors." The per-stage breakdown at 1M tokens shows that Stage 0 (the first, `$O(T)$` pruning stage) consumes 53.7% of total prefill time — more than all other stages combined. Stage 1 consumes 9.2%, Stage 2 consumes 13.1%, and block sparse attention (BSA) consumes 13.9%. This breakdown confirms the paper's asymptotic analysis: the first stage dominates, and its cost grows with context length while subsequent stages have fixed cost.

**Decoding latency at 1M context (Table 3).** This is where InfiniteHiP's architectural advantages over HiP and InfLLM become most apparent:
- FA2: 4,645 µs per step
- InfLLM (12K): 1,222 µs (3.80× faster)
- HiP (1K): 450 µs (10.3× faster than FA2)
- InfiniteHiP (3K, no Extend): 234 µs (19.85× faster than FA2, 5.22× faster than InfLLM, 1.92× faster than HiP)
- InfiniteHiP (3K, with Extend): 245 µs (18.96× faster than FA2)

The critical comparison is against HiP: InfiniteHiP achieves 1.92× faster decoding despite processing 3× more retained tokens (3K vs. 1K). This cannot be explained by token count alone — it reflects the improved parallelism from the modular kernel design and the mask caching mechanism. The "Total (AR)" row — measuring decoding latency without any mask caching (all stages recomputed every step) — shows 936 µs, meaning caching alone provides a 4.0× speedup (936 → 234 µs). The per-stage breakdown in the cached case shows: Stage 0: 28.2%, Stage 1: 4.0%, Stage 2: 5.3%, BSA: 2.2%, Extra: 60.3%. The "Extra" component (60.3%) — which includes kernel launch overhead, memory management, and other framework-level costs — dominates the cached decoding latency, suggesting there is room for further engineering optimization.

**Latency with KV cache offloading (Table 4).** Table 4 is the most operationally realistic benchmark, measuring decoding latency with UVM-based KV cache offloading enabled on an RTX 4090 with PCIe 4.0 x8. Key findings:

- At 256K context with no stage caching ("Cached Stages: None"): total attention latency is 9,803 µs. The offloading component alone is 2,039 µs (21%). The SA (sparse attention) hit ratio is 58.92% — meaning over 40% of attention key accesses miss the GPU cache and require host memory fetches.
- At 256K with all stages cached ("Cached Stages: All"): total latency drops to 110 µs — an 89× reduction from the no-caching case at the same context length. The offloading component drops to 503 µs (but this appears in an intermediate column — rechecking: the "All" column at 256K shows BSA 31 µs, Offload not listed separately? Actually looking more carefully at the table structure: the columns are "None | S1 | S1&2 | All" where each represents which stages are cached. The "All" column at T=256k shows: Stage 0: -, Stage 1: -, Stage 2: -, BSA: 31, Offload: -, Extra: 79, Total: 110. At T=1024k "All" column: Stage 0: -, Stage 1: -, Stage 2: -, BSA: 30, Offload: -, Extra: 89, Total: 119. This is remarkable — with all stages cached at 1M context, total attention latency is only 119 µs, comparable to a 256K context with all stages cached (110 µs). The offloading latency is essentially eliminated because all needed tokens are in the GPU cache, and the SA hit ratio is 99.8%).

This demonstrates the compounding benefit of the modular caching design: when the pruning masks are cached (not recomputed), the GPU key cache naturally retains the actively accessed tokens because the same chunks are repeatedly selected. At 1M context with all stages cached, the GPU key cache achieves 99.8% hit rate, making offloading overhead negligible. This is a stronger result than any prior work on KV cache offloading has demonstrated at this context scale.

**Comparison against InfLLM with offloading.** InfLLM at 256K achieves 1,186 µs — InfiniteHiP at 256K with all stages cached achieves 110 µs, a 10.8× speedup. At 1M context, InfLLM achieves 1,234 µs vs. InfiniteHiP's 119 µs (all cached) — a 10.4× speedup. Even when stages are not cached (worst case), InfiniteHiP at 256K (9,803 µs) is substantially slower than InfLLM (1,186 µs), but with the aggressive caching that the paper demonstrates maintains accuracy (Table 2 shows 3K-fast and 3K-flash retain 98.35% and 97.78% relative performance), the amortized cost is dramatically lower.

#### Throughput in Production Serving Framework (Figure 5, Tables 11–12)

The SGLang end-to-end throughput benchmarks translate per-layer latency into a practical metric: tokens per second for continuous generation.

**RTX 4090 (24GB) at 1M context:** InfiniteHiP 3K-Fast (estimated) achieves 97.0 tok/s vs. SRT estimated at 12.5 tok/s — a 7.76× improvement. With offloading enabled: 3K-Fast offload achieves 17.3 tok/s, 3K-Flash offload achieves 40.1 tok/s. The Flash configuration's 2.3× speedup over Fast (40.1 vs. 17.3) comes from refreshing the expensive Stage 1 mask only every 96 steps instead of every 32, demonstrating that aggressive caching of coarse-stage masks is the primary lever for throughput.

**L40S (48GB) at 3M context:** InfiniteHiP 3K-Fast (estimated) achieves 66.4 tok/s vs. SRT estimated at 3.3 tok/s — a 20.1× improvement. With offloading: 3K-Fast achieves 7.6 tok/s, 3K-Flash achieves 23.8 tok/s — a 3.1× improvement from Flash over Fast. The fact that throughput remains usable (23.8 tok/s ≈ ~1.4 seconds per 30-token response) at 3M tokens on a single 48GB GPU, when the KV cache alone would require ~192GB without offloading, is the headline practical result — this would simply be impossible on this hardware with any prior method.

---

### Ablation Studies and Robustness Checks

**Number of pruning stages (Figure 6b):** On ∞Bench En.MC with context truncated to 128K, using N=2 pruning stages achieves 70.31% accuracy while N=3 achieves 74.24% — a 3.93 percentage-point improvement from the additional refinement stage. The paper notes that "the latency-performance optimal pruning module combination for each setting is found empirically," implying that stage count, chunk sizes, and retention counts are tuned per configuration rather than derived from a universal formula. No results are shown for N=1 or N>3, leaving the limits of stage stacking unexplored.

**Top-k token recall (Figure 6a):** Comparing the recall of attention probabilities (fraction of true top-k attention mass captured by the selected sparse tokens), InfiniteHiP achieves approximately 95–96% recall at 4,000 top-k tokens, compared to InfLLM at ~93–94% and HiP at ~90–91%. InfiniteHiP's advantage over HiP is 4.72 percentage points, and over InfLLM is 1.57 percentage points. The recall curves suggest diminishing returns: all methods achieve >88% recall at 1,000 tokens, and the gains from 1,000 to 4,000 tokens are modest (~6–7 percentage points). This implies that capturing the next few percent of attention mass requires disproportionately more retained tokens, and the sweet spot for efficiency is around 2,000–4,000 tokens.

**RoPE style combinations (Table 5):** This is the most systematic ablation of a design space in the paper. Testing four RoPE styles for context pruning (Dynamic Extend, InfLLM-style, Chunk-indexed, Relative) against three RoPE styles for block sparse attention (Dynamic Extend, InfLLM-style, StreamingLLM) yields a 4×3 grid of combinations evaluated on ∞Bench En.MC truncated to 128K. Key findings:
- The best combination (RT/ST: Relative in pruning, StreamingLLM in sparse attention) achieves 70.31%.
- The worst combination (DE/DE: Dynamic Extend in both) achieves only 52.40% — a 17.91 percentage-point gap, showing that RoPE strategy choice is not a minor hyperparameter.
- Across all pruning styles, StreamingLLM-style RoPE in sparse attention consistently achieves the highest average (64.85%), followed by InfLLM-style (64.19%), and Dynamic Extend (63.76%). This validates the paper's choice of StreamingLLM RoPE for the final attention computation.
- Across all attention styles, Relative RoPE in pruning achieves the highest average (68.56%), followed by InfLLM-style (68.41%), Chunk-indexed (67.39%), and Dynamic Extend (52.69%). The large gap between DE and the others confirms that SelfExtend-style stretching is poorly suited for the pruning algorithm's access patterns.

**Layer-specific RoPE assignment (Table 6, Appendix D):** Comparing Relative RoPE in all layers (68.55% on ∞Bench En.MC at 300K context) against Chunk-indexed RoPE in layers 1–3 with Relative RoPE in layers 4–32 (74.23%) shows a 5.68 percentage-point gain. This ablation directly validates the diagnostic methodology: analyzing per-layer attention patterns and assigning RoPE strategies accordingly. However, the cutoff at layer 3 is specific to Llama 3.1 8B — other models may require different boundaries, and the paper does not test sensitivity to this choice.

**Mask refresh interval (Tables 2 and 4):** The "fast" and "flash" presets vary the refresh intervals for pruning stages while keeping all other parameters identical. On ∞Bench, 3K-default achieves 98.17% relative score, 3K-fast achieves 98.35%, and 3K-flash achieves 97.78%. The fact that 3K-fast actually scores *higher* than the default — despite refreshing masks less frequently — suggests either noise in the evaluation or that the default refresh rates are unnecessarily conservative. The 3K-flash degradation (97.78% vs. 98.17%) is only 0.39 percentage points relative, while providing substantial throughput gains (3.14× over Fast at 3M context per Figure 5). This robustness to refresh interval is evidence that attention patterns genuinely change slowly during decoding — temporal locality is not just an enabling assumption but a strong empirical regularity.

**Context window size (3K vs. 5K, Tables 1 and 2):** The 5K window preset consistently outperforms the 3K window: on Llama 3 ∞Bench, 47.08% vs. 46.25% absolute average; on Mistral 0.2 ∞Bench, 54.96% vs. 51.52%. The gains are modest (~1–3.5 percentage points absolute) given the 67% increase in retained tokens (5K vs. 3K), suggesting diminishing returns — the 3K window already captures most of the achievable performance, and additional tokens provide marginal improvements. The paper does not systematically sweep window sizes from 1K to 10K to characterize this diminishing returns curve, which would be valuable for understanding the optimal operating point for different tasks and context lengths.

**Passkey retrieval on DeepSeek R1 (Table 7, Appendix E.1):** As a model-agnosticism check, the passkey retrieval test is run on DeepSeek R1 Distilled Qwen2 14B (128K pretrained context) with context extended to 1M tokens. InfiniteHiP achieves 100% retrieval accuracy at depths from 128K to 872K, and 96% at the shallowest depth (20-0% range, where the passkey is placed early in the context and aggressive middle-context pruning is most likely to discard it). This demonstrates that the framework transfers to a completely different model family without hyperparameter re-tuning for the passkey task — though the paper does not report full benchmark results for this model.

**Prefill chunk size (Table 3, implicit):** The prefill latency measurements use a "chunked prefill style attention, with a chunk size of 32K." This means during prefill, rather than processing all query tokens against all keys simultaneously, the queries are processed in 32K-token chunks — a standard optimization to bound peak memory usage. The paper does not ablate this chunk size, and it's unclear whether the pruning pipeline's effectiveness depends on query chunk size during prefill.

**UVM vs. no-offloading latency decomposition (Table 4, "Offload" row):** The offloading overhead is explicitly decomposed: at 256K with no stage caching, offloading accounts for 2,039 µs of 9,803 µs total (20.8%). As stages are cached, this overhead decreases because the UVM page faults for pruning stage accesses are eliminated — the GPU cache already contains the needed keys. At 1M context with all stages cached, offloading overhead is not separately listed (marked as "-"), implying it is negligible or zero because the 99.8% SA hit ratio means essentially all attention computation accesses GPU-resident keys.

**Negative result — H2O collapse on long contexts (Table 2):** H2O's KV cache eviction achieves only 3.95% relative on Llama 3 and 52.98% relative on Mistral 0.2 on ∞Bench. This is a strong negative result confirming that permanent token eviction is fundamentally incompatible with tasks requiring arbitrary-context retrieval — the paper's stated motivation for offloading over eviction. However, H2O performs better on Mistral 0.2 (52.98%) than on Llama 3 (3.95%), which the paper does not explain — it may be due to Mistral's larger pretrained context window (32K vs. 8K) making the model less dependent on long-range retrieval for ∞Bench's tasks, or differences in attention pattern dispersion between the model families.

---

### Critical Assessment

#### Claim 1: InfiniteHiP achieves higher quality than the best prior methods while being faster and more memory-efficient.

**What the experiments demonstrate:** The LongBench and ∞Bench results (Tables 1 and 2) consistently place InfiniteHiP at or near the top of the relative score rankings, with margins of 7–10 percentage points relative over InfLLM (the strongest baseline) on Llama 3. The latency results (Tables 3 and 4) show 5.22× faster decoding than InfLLM and 10.4× faster with offloading at 1M context. The memory results show processing 3M tokens on a 48GB GPU — which no baseline can do at all.

**What qualifies these claims:** Several important nuances are not fully surfaced in the paper's presentation:

1. **InfLLM's hyperparameters may not be optimally tuned.** InfLLM is evaluated at window sizes of 6K, 8K, and 12K on Llama 3, and 6K, 12K, and 16K on Mistral. The paper does not report how these window sizes were chosen, whether they were tuned per-model per-benchmark, or whether InfLLM could achieve better accuracy-latency tradeoffs at different settings. The comparison is between InfiniteHiP's carefully tuned presets and InfLLM's presumably-tuned-but-not-optimized configurations. A fairer comparison would sweep both methods across a range of parameter budgets and plot Pareto frontiers.

2. **The "3K" and "5K" presets are the result of hyperparameter optimization on the same test benchmarks.** The paper does not describe a held-out validation set or any hyperparameter search protocol that would prevent overfitting the window sizes, chunk sizes, and retention counts to the test sets. If the presets were tuned on LongBench and ∞Bench test data (or on data with similar distribution), the reported scores may be optimistically biased. The cross-validation protocol used in the reference example paper is absent here.

3. **HiP Attention's 1K window is not a fair comparison point.** The paper compares InfiniteHiP-3K against HiP-1K — but HiP was designed with different algorithmic constraints and its hyperparameters were presumably tuned for different tradeoffs. The paper does not report HiP performance at larger window sizes (e.g., HiP-3K or HiP-5K), which would isolate the architectural improvements (modular parallelism, LRU caching) from the simple effect of retaining more tokens.

4. **The "relative score" metric can obscure absolute differences.** The Average Relative score normalizes each subset's score to the maximum achieved by any method — so if one method achieves much higher scores on subsets where other methods also do well, the relative aggregation can amplify or diminish apparent gaps. For instance, on LongBench with Llama 3, InfiniteHiP achieves 47.72% absolute while InfLLM achieves 44.47% — a 3.25 percentage-point absolute gap that maps to a 7.17 percentage-point relative gap. Whether this is "large" depends on the application. The absolute gains on individual tasks are often modest (2–4 points), concentrated on a few retrieval-heavy tasks, and not tested for statistical significance.

#### Claim 2: InfiniteHiP enables out-of-length generalization that matches or exceeds dedicated RoPE extension methods while being integrated with sparse attention.

**What the experiments demonstrate:** Figure 4 and Table 10 show that InfiniteHiP enables short-context models (EXAONE 3 at 4K, Gemma2 at 8K) to process contexts up to 128K–256K with *improving* accuracy as context grows. Self-Extend (the strongest dedicated RoPE extension baseline) achieves 67.81% relative on Llama 3 ∞Bench, while InfiniteHiP-3K achieves 98.17% — a substantial gap. The layer-specific RoPE ablation (Table 6) shows 5.68 percentage points gain from the selective strategy over uniform Relative RoPE.

**What qualifies these claims:**

1. **Self-Extend is evaluated with dense attention on the full context, not with sparse attention.** Self-Extend + FA2 on 128K context is a different computational budget than InfiniteHiP's sparse attention on 3K tokens. The comparison conflates two differences: (a) RoPE adjustment strategy, and (b) dense vs. sparse attention. To isolate RoPE contribution, one would need to test Self-Extend-style RoPE within the InfiniteHiP pruning pipeline — which Table 5 does to some extent, showing DE/ST combination achieves only 51.09%, confirming that SelfExtend's RoPE is poorly matched to sparse attention's access patterns. But this comparison is buried in an ablation table rather than presented as a head-to-head with the dedicated RoPE extension baselines.

2. **The per-model layer cutoff (1–3) is empirically determined for one model family.** The paper shows that Chunk-indexed RoPE in early layers matters for Llama 3.1 8B (Table 6), but does not report per-layer attention pattern analysis for Mistral, Gemma2, or EXAONE. The OOL generalization results on these other models (Table 10, Figure 4) use the same framework, but it's unclear whether the layer-specific RoPE strategy was re-tuned or simply ported from Llama 3.1. If the latter, it's possible that different models would benefit from different cutoffs.

3. **The OOL generalization claim depends on having a RoPE mechanism at all.** Models using other positional encodings (ALiBi, learned absolute positions, no position encoding) are untested. The paper's claim of universal applicability to "any pretrained Transformer-based LLM" overreaches relative to the experimental evidence, which covers only RoPE-based models.

#### Claim 3: KV cache offloading with LRU-based GPU caching enables practical long-context inference on consumer hardware.

**What the experiments demonstrate:** Table 4 and Figure 5 show that InfiniteHiP processes 3M tokens on a single 48GB GPU with usable throughput (23.8 tok/s in the Flash configuration on L40S). The SA hit ratio reaches 99.8% with stage caching enabled, and the offloading overhead is negligible in the cached regime. This is a genuine engineering achievement.

**What qualifies these claims:**

1. **The throughput numbers rely on single-batch, single-sequence decoding.** The paper states: "We test only single-batch scenarios because we expect a single sequence to be larger than GPU VRAM." In production serving systems, batching multiple requests is essential for cost efficiency. The throughput advantages may differ substantially under multi-request batching, where GPU memory is shared across sequences and cache contention becomes a factor.

2. **The latency measurements are on a Passkey retrieval sample, not on representative benchmark tasks.** Table 4 notes the latency experiment uses "a Passkey retrieval task sample." The access patterns for passkey retrieval (finding a single buried sentence) may differ from those for summarization or multi-hop QA — tasks that require attending to many distributed chunks. The SA hit ratio of 99.8% on a passkey task may not generalize to tasks with more distributed attention patterns.

3. **The offloading comparison against InfLLM is somewhat artificial.** InfLLM "chooses not to access the CPU memory while executing its attention kernel" and therefore uses larger context windows (12K–16K) to compensate for less precise top-k estimation. InfiniteHiP accesses CPU memory during kernel execution via UVM. These are fundamentally different architectural choices about the GPU-CPU boundary, and the latency comparison reflects this design choice rather than a pure algorithmic superiority. A version of InfLLM that used UVM-based offloading (if feasible) might close some of the latency gap.

4. **The 3M-token claim depends on AWQ quantization and FP8 KV cache.** The throughput benchmarks specify "AWQ Llama 3.1 with FP8 KV cache" — using weight compression and reduced-precision KV cache. The paper does not report results at FP16 precision, which would require roughly 4× the KV cache memory and may not fit in the same GPU budget. The 3M-token figure is therefore contingent on the specific compression and quantization choices.

#### Missing experiments that would strengthen the paper:

1. **Pareto frontier analysis across accuracy and latency.** The paper reports accuracy and latency separately but does not plot them against each other for different hyperparameter configurations (window sizes, refresh rates, stage counts). Such a plot would reveal the optimal operating points and whether InfiniteHiP genuinely dominates the baselines or merely occupies a different point on a shared tradeoff curve.

2. **Evaluation on a broader range of context lengths for the main benchmarks.** LongBench averages ~32K tokens and ∞Bench >100K. The paper does not systematically sweep context lengths (e.g., 8K, 16K, 32K, 64K, 128K, 256K, 512K) for the main quality benchmarks, making it difficult to characterize where InfiniteHiP's advantages emerge relative to baselines. The RULER results (Tables 8–9) provide this sweep but for a more restricted task set.

3. **Multiple random seeds or statistical significance testing.** None of the benchmark results include error bars, standard deviations, or p-values. Without these, it's impossible to determine whether a 1–2 percentage-point difference (common for many task-level comparisons in Table 1) is statistically reliable or within noise.

4. **Ablation on the sink and streaming token counts.** The paper inherits `$n_{\text{sink}} = 256$` and `$n_{\text{stream}} = 1024$` from prior work but does not ablate these values. Given that they collectively account for ~1,280 of the ~3,000–5,000 retained tokens, their contribution to overall performance is substantial and should be characterized.

5. **Direct comparison between InfiniteHiP and a hypothetical "InfLLM with UVM offloading."** The paper argues that InfLLM's design prevents it from using UVM, but a modified version that could would provide a more controlled comparison isolating the pruning algorithm from the offloading mechanism.

6. **Effect of the pruning pipeline on tasks requiring multi-step reasoning from context.** The MF (Math Find) task on ∞Bench shows InfiniteHiP underperforming InfLLM (17.71% vs. 23.70% on Llama 3). The paper does not analyze this failure case or investigate whether larger window sizes or different pruning hyperparameters could recover the lost performance. This is particularly important because math reasoning from long contexts is a motivation stated in the introduction ("extending context length is essential for improving comprehension and coherence").

## 6. Limitations and Trade-offs

### 6.1 The First Pruning Stage Remains Asymptotically Linear in Context Length

**The assumption or constraint.** The modular pruning pipeline achieves near-constant cost for Stages 2 and 3, which operate on fixed-size input sets (32K → 8K → ~3K tokens) regardless of total context length. However, Stage 1 processes the *entire* context and the paper explicitly acknowledges in Appendix A: "the initial pruning stage of InfiniteHiP's context pruning algorithm runs in O(T_q T_{kv}) time, and all subsequent pruning stages run in O(T_q) time. This makes the initial pruning stage the most expensive one as the number of tokens increases." The amortization strategy — mask caching with refresh intervals of 16, 32, or 96 steps — reduces the *frequency* of this cost but does not reduce the cost *per occurrence*. As context grows from 1M to 3M to 10M tokens, the latency of a single Stage 1 refresh grows linearly, and at some context length, even occasional Stage 1 refreshes will dominate the amortized per-step latency.

**The consequence.** The paper's headline 18.95× decoding speedup over FA2 and 10.4× over InfLLM (Tables 3 and 4) is measured when Stage 1 is cached — meaning the amortized cost assumes mask reuse. But when the mask *must* be refreshed (every 16–96 steps, depending on configuration), the decoder experiences a latency spike proportional to context length. Table 3 quantifies this: at 1M context without any mask caching ("Total AR"), decoding latency is 936 µs — 4× slower than the cached 234 µs. This creates a **latency jitter problem**: most decoding steps run fast (119 µs at 1M with full caching per Table 4), but periodic Stage 1 refreshes introduce outlier steps that are orders of magnitude slower. For interactive applications with latency targets (e.g., sub-100ms time-to-first-token or per-token latency), these periodic spikes may violate service-level objectives even if *average* throughput is acceptable. Furthermore, the Stage 1 prefill cost at 3M tokens is not directly measured — extrapolating from Table 3's trend (Stage 0 at 1M consumes 53.7% of 172 ms ≈ 92 ms prefill cost), a 3M prefill Stage 1 alone would require roughly 3× that time (~276 ms for Stage 1 plus fixed costs for Stages 2–3 and BSA), which may be acceptable for batch processing but problematic for online serving.

**What evidence exists in the paper.** Table 3 directly measures decoding latency with and without mask caching: 234 µs (cached) vs. 936 µs (uncached) at 1M — a 4× difference. Table 4 shows that with all stages cached, the amortized per-step latency at 1M is only 119 µs, but this requires the SA hit ratio to be 99.8%, which is measured on a Passkey retrieval task that may have more concentrated attention patterns than general NLU tasks. Figure 5's throughput benchmarks report tokens-per-second averages but do not report latency distributions, tail latencies, or the frequency/cost of mask refresh pauses. The paper does not measure maximum or P99 decoding latency at any context length.

**Mitigation status.** The paper partially addresses this through mask caching with configurable refresh rates, but acknowledges the limitation in Appendix G: "Significant bottlenecks in the prefill stage. Even after replacing the quadratic attention mechanism with a near-linear alternative like InfiniteHiP, serving over 1M tokens still takes more than 10 minutes in many consumer grade hardwares." The authors propose speculative attention and lazy initialization as future work directions, but these are not implemented or evaluated. The fundamental tension — Stage 1 must occasionally scan the full context, and context length determines the cost of that scan — is inherent to any method that dynamically selects context tokens and cannot be fully eliminated by caching alone.

---

### 6.2 Difficulty Estimation Is Not Addressed — The Pipeline Has No Mechanism to Know When Pruning Is Failing

**The assumption or constraint.** InfiniteHiP's pruning pipeline operates as an open-loop system: it selects ~3K–5K tokens using the chunk score estimation and SelectRep heuristics from Section 4, but it has **no feedback mechanism** to detect when it has discarded information critical to answering the query. The pipeline does not monitor attention score distributions, does not estimate confidence in its selected mask, and does not adaptively expand the retention window when the current selection appears insufficient. The paper provides no difficulty estimation, no uncertainty quantification, and no adaptive budget allocation — the 3K or 5K window is applied uniformly to all prompts regardless of their complexity or the distribution of relevant information in the context.

**The consequence.** For tasks where the relevant information is tightly clustered in a few chunks, aggressive pruning works well — the topscoring chunks naturally contain the answer. But for **tasks requiring information distributed across many chunks**, the fixed retention budget may discard essential context. The paper provides one concrete signal of this failure mode: on the M ath Find (MF) synthetic task in ∞Bench (Table 2), InfiniteHiP scores 17.71% while InfLLM scores 23.70% — a 6 percentage-point deficit despite InfiniteHiP's ~3× fewer retained tokens. Math reasoning from long contexts may require attending to numerical values scattered across multiple paragraphs, and the pipeline's chunk-based selection may discard some of these values because no single chunk appears individually important enough to survive the top-K selection. More broadly, any task whose answer depends on integrating information from a large fraction of the context — long-form summarization, multi-hop reasoning over many documents, complex code understanding — may suffer from the uniform pruning budget.

This limitation is compounded by the absence of difficulty estimation: the system cannot distinguish between a simple passkey retrieval task (where 3K tokens is more than sufficient) and a complex multi-hop QA task (where 3K tokens may be insufficient) and allocate the retention budget accordingly. A 5K window provides marginally better performance (Table 2: 47.08% vs. 46.25% absolute average on Llama 3 ∞Bench), but this fixed expansion does not adapt to per-prompt needs.

**What evidence exists in the paper.** The MF deficit in Table 2 is the clearest signal. The RULER results (Table 9) show that Multi-Key NIAH variants degrade substantially: NIAH3 MK scores only 32.5% (3K config) while Single Key variants score 96%. The paper does not provide per-task retention analysis showing what fraction of ground-truth-relevant tokens are retained by the pruning pipeline for different task types, which would directly quantify the pruning error rate. No ablation varies the retention budget per task difficulty level to characterize the budget-accuracy tradeoff curve. The paper does not report attention recall for tasks where InfiniteHiP underperforms baselines.

**Mitigation status.** Not addressed. The paper does not mention adaptive retention budgets, confidence estimation, or any mechanism for the model to signal that it needs more context. Appendix G gestures at "task-dependent module configurations" as future work but provides no concrete direction. A natural extension — using the PRM-style scoring (from the reference example paper) or the chunk scores themselves — to estimate whether the current window is sufficient is not discussed.

---

### 6.3 The Method Is Validated on a Narrow Set of Benchmarks and Model Families, Making Generality Claims Unsupportable

**The assumption or constraint.** The paper evaluates InfiniteHiP on three long-context benchmarks (LongBench, ∞Bench, RULER) and primarily on two model families (Llama 3 8B, Mistral 0.2 7B), with additional models tested only on subsets (Gemma2, EXAONE 3/3.5 on selected ∞Bench tasks; DeepSeek R1 on passkey only). All benchmarks are English-language text understanding tasks. The OOL generalization claims rest on models that all use Rotary Position Embeddings — no results are provided for models with ALiBi, learned absolute positions, or other positional encoding schemes. The paper states in the introduction that InfiniteHiP "can be used as a drop-in replacement for any pretrained Transformer-based LLM," but this universality claim far exceeds the experimental coverage.

**The consequence.** Several important generalization questions remain unanswered:

1. **Model scale:** All experiments are on 7B–8B parameter models. The attention sparsity patterns observed in Section 3 (Figure 2a) — "fewer than 2% of the chunks contain more than 12.5% of the top-2K tokens" — were measured on Llama 3.1 8B. Larger models may exhibit different sparsity characteristics (more diffuse attention, more heads attending to different regions), which would change the optimal chunk sizes, retention budgets, and stage counts. The paper provides no evidence that the 3K–5K window generalizes to 70B+ models.

2. **Domain shift:** All benchmarks are English-language, text-only tasks. The introduction motivates long-context inference for "multi-modal, and retrieval-augmented language generation," but these domains are entirely untested. Visual-language models with interleaved image and text tokens may have fundamentally different attention patterns that the chunk-based pruning would not capture.

3. **Positional encoding schemes:** The dynamic RoPE adjustment strategy is specific to Rotary Position Embeddings. The paper provides no discussion of how (or whether) the framework would work with ALiBi, learned absolute positions, T5-style relative position biases, or no position encoding at all.

4. **Task diversity:** The benchmarks cover QA, summarization, few-shot learning, and synthetic retrieval — but not code generation (beyond LCC in LongBench), mathematical theorem proving, multi-turn dialogue with long conversation history, or agent-style planning over long observation sequences. The paper does not establish whether the pruning pipeline preserves the fine-grained token-level information needed for code syntax or mathematical notation.

**What evidence exists in the paper.** The paper's experiments demonstrate strong performance on the tested benchmarks, but the experimental breadth is narrow. The DeepSeek R1 pas skey test (Table 7) is the only result on a model outside the 7B–8B class, and it is a single synthetic task. The Gemma2 and EXAONE results (Table 10) cover only En.MC, En.QA, and En.Sum on ∞Bench — not LongBench or RULER. The paper does not report performance on any non-English dataset, any multi-modal dataset, or any code generation benchmark at long context lengths. The claim that InfiniteHiP works for "any pretrained Transformer-based LLM" (Section 1) is aspirational rather than empirically supported.

**Mitigation status.** The paper does not explicitly acknowledge this as a limitation. The scope of the experimental validation is not discussed in the limitations section (Appendix G focuses on latency and memory challenges rather than generalization). The claim of universal applicability remains unqualified in the main text.

---

### 6.4 Hyperparameter Sensitivity and the Cost of Preset Tuning Are Not Characterized

**The assumption or constraint.** InfiniteHiP introduces a substantial number of hyperparameters: three pruning stages each with query block size (b_q), chunk size (l_c), and retention count (k); the number of sink tokens (n_sink = 256); the number of streaming tokens (n_stream = 1024); mask refresh intervals for each stage; and a layer-specific RoPE strategy with a hard cutoff between layers 1–3 and 4+. The paper reports results for a small set of presets ("3K," "5K," "fast," "flash") but provides almost no sensitivity analysis for individual parameters. The paper states in Appendix F that "we discovered numerous module design choices during our research" and in Appendix G that "the combination of pruning modules should be studied more in future research," indicating that the hyperparameter space is large and the optimal configuration is not obvious.

**The consequence.** A practitioner seeking to deploy InfiniteHiP on a new model or task faces an unclear tuning burden. The presets were developed for and validated on specific model-benchmark combinations, and it is unknown whether they transfer. Key unanswered questions include:

- **Chunk sizes (256, 32, 8):** Were these derived from the chunk sparsity analysis in Figure 2a, or found by trial and error? Figure 2a shows that 75% of 64-token chunks are empty, but the paper uses 256-token chunks for Stage 1 and 8-token chunks for Stage 3. The relationship between chunk size, recall, and latency is not systematically characterized.

- **Retention counts (32K, 8K, 2K/4K):** How do these trade off accuracy and latency? The paper shows that 5K outperforms 3K modestly (Table 2), but does not sweep from 1K to 16K to characterize the diminishing returns curve. What is the minimum retention budget below which performance collapses?

- **Number of stages (N=3):** Figure 6b shows that N=3 outperforms N=2 by 3.93 percentage points on one task, but no results are shown for N=1 (single-stage pruning) or N>3. Is three stages optimal, or merely sufficient for the tested models?

- **Sink and streaming token counts (256, 1024):** These values are inherited from StreamingLLM and not ablated. Together they account for 1,280 of the ~3,280–5,280 retained tokens — roughly 25–40% of the total window. If these defaults are suboptimal, a substantial fraction of the attention budget is being wasted.

- **Layer-specific RoPE cutoff (layers 1–3 vs. 4+):** Table 6 validates this cutoff for Llama 3.1 8B, but it is unclear whether the cutoff generalizes to Mistral, Gemma2, or EXAONE. The paper does not report per-layer attention pattern analysis for any model other than Llama 3.1 8B.

**What evidence exists in the paper.** The paper provides some ablations (Figure 6b for stage count, Table 5 for RoPE style combinations, Table 6 for layer-specific RoPE), but these test one parameter at a time rather than exploring interactions. No hyperparameter sensitivity curves are shown — for any parameter, the paper reports at most 2–3 discrete values. The "fast" and "flash" presets vary only mask refresh intervals while holding all other parameters constant, leaving interaction effects unexplored (e.g., perhaps with longer refresh intervals, larger retention counts are needed to compensate for stale masks). The paper does not report the computational cost of finding the default presets — how many GPU-hours of search were required, whether Bayesian optimization or grid search was used, or whether the presets were tuned on the test benchmarks.

**Mitigation status.** The paper acknowledges this implicitly in Appendix G ("the combination of pruning modules should be studied more in future research... we discovered numerous module design choices") but does not frame it as a practical limitation for adopters. No guidance is provided for hyperparameter selection on new models or tasks. The presets are offered as-is, and practitioners are left to either trust them or invest in their own tuning experiments.

---

### 6.5 The Offloading Throughput Advantage Depends on Heavy Quantization and Single-Sequence Workloads

**The assumption or constraint.** The paper's headline result — processing 3 million tokens on a single L40s 48GB GPU — is achieved using AWQ weight quantization and FP8 KV cache compression. The throughput benchmarks in Figure 5 and Tables 11–12 specify "AWQ Llama3.1 with FP8 KV cache data type" in their captions. Without these compression techniques, the KV cache for 3M tokens with Llama 3.1 8B at FP16 would require approximately 4× the memory — roughly 768GB — which cannot fit in 2TB of CPU DRAM after accounting for model weights and overhead. Even at FP8, the total KV cache is ~192GB, and the offloading system must manage this within the available PCIe bandwidth. The throughput measurements are for single-batch, single-sequence decoding ("We test only single-batch scenarios because we expect a single sequence to be larger than GPU VRAM"), which is a workload pattern that minimizes cache contention but does not represent typical production serving where multiple requests are batched to maximize GPU utilization.

**The consequence.** The practical deployability of InfiniteHiP at extreme context lengths depends on quantization choices that themselves introduce quality degradation. The paper does not evaluate the interaction between KV cache quantization and the pruning pipeline — does FP8 KV cache compression affect the attention score estimates used by SelectRep? Does it change the chunk sparsity statistics that motivate the method? These questions are unanswered.

More critically, the **single-batch constraint** means InfiniteHiP cannot benefit from the batching efficiency that makes production LLM serving cost-effective. In a multi-request serving scenario, each sequence would compete for the limited GPU key cache slots, and the LRU eviction policy would thrash between sequences' active token sets. The SA hit ratio of 99.8% reported in Table 4 is measured on a single sequence; with 8 or 16 concurrent sequences, the GPU key cache would need to accommodate 8–16× as many active tokens, dramatically increasing the miss rate and offloading overhead. The paper does not evaluate InfiniteHiP under concurrent request loads.

Additionally, the **CPU DRAM ceiling** eventually bounds the maximum context length even with offloading. The paper notes in Appendix G: "in practice we are still limited to CPU memory, which is around 2TB (512GB per GPU; AWS provides around 2TB CPU memory for 8 GPU machines)." At FP8 precision with Llama 3.1 8B, 2TB of CPU DRAM corresponds to roughly 31M tokens — a hard ceiling that linear memory growth will eventually hit. KV cache compression or eviction would be necessary to exceed it, which reintroduces the "permanent forgetting" problem the paper criticizes in H2O.

**What evidence exists in the paper.** The throughput benchmarks in Figure 5 and Tables 11–12 demonstrate the quantization-dependent nature of the results — all offloading throughput numbers assume AWQ + FP8 KV cache. The paper does not report FP16 throughput, memory usage, or quality comparisons. The single-batch limitation is explicitly stated in the experimental setup but its implications for production serving are not discussed. The CPU memory ceiling is acknowledged in Appendix G as a "remaining challenge" but is not characterized quantitatively for the paper's headline 3M-token claim.

**Mitigation status.** The paper acknowledges these challenges in Appendix G: "we have several options: KV quantization, KV eviction, KV compression... it is crucial to further improve KV cache memory efficiency with quantization and compression." The authors frame their offloading framework as "a practical foundation for efficiently managing large working sets" and suggest that combining it with future compression techniques could extend the achievable context length further. However, the current paper does not implement or evaluate any such combination, and the 3M-token claim relies on quantization that is not ablated for quality impact.

---

### 6.6 No Statistical Rigor — Benchmark Results Lack Confidence Intervals, Multiple Seeds, or Validation Protocols

**The assumption or constraint.** The paper reports single-run evaluation results on the standard test sets of LongBench, ∞Bench, and RULER. No confidence intervals, standard deviations, statistical significance tests, or multiple random seeds are reported for any result in Tables 1, 2, 5, 6, 8, 9, or 10. The hyperparameter presets ("3K," "5K," "fast," "flash") are not described as being selected through a held-out validation procedure — the paper provides no evidence that the presets were not tuned directly on the test benchmarks. In contrast to the reference example paper, which uses two-fold cross-validation within difficulty bins on a fixed test set to select compute-optimal strategies without contamination, InfiniteHiP's strategy selection protocol is entirely opaque.

**The consequence.** The reported performance differences between InfiniteHiP and baselines may be statistically indistinguishable from noise for many task-level comparisons. For instance, on LongBench with Llama 3 (Table 1), the absolute accuracy difference between InfiniteHiP (47.72%) and InfLLM (44.47%) is 3.25 percentage points averaged over 16 heterogeneous tasks. Individual task differences are often smaller: Qasper (43.2 vs. 43.7 — InfiniteHiP trails by 0.5 points), TREC (75.5 vs. 73.5 — 2.0 points), MSQ (30.9 vs. 26.1 — 4.8 points, but this is a single task with unknown variance). Without variance estimates, a practitioner cannot determine whether the claimed 7.17 percentage-point relative improvement is robust or driven by a few outlier tasks where InfiniteHiP excels by chance.

More concerning is the possibility of **test-set overfitting through hyperparameter tuning**. With a rich hyperparameter space (chunk sizes, retention counts, stage counts, refresh intervals, RoPE styles, sink/streaming counts, layer cutoffs) and only 500–5000 test questions per benchmark, it is possible to achieve inflated test scores by selecting hyperparameters that happen to perform well on the specific test instances. The paper provides no information about how the presets were developed — whether through a held-out validation split, cross-validation, or direct optimization on test-set performance. The absence of any validation protocol is a material weakness given the number of tunable parameters.

The **relative score metric** used for aggregation further complicates interpretation. By normalizing each subset's score to the maximum achieved by any method, the metric amplifies differences on subsets with low absolute scores and compresses differences on subsets with high absolute scores. If one method achieves anomalously high performance on a single subset (e.g., due to hyperparameter overfitting on that subset), the relative scores of all other methods on that subset are deflated, skewing the aggregate comparison. The paper argues that relative scores "better represents the differences in performance because the variance is normalized per subset," but this normalization is performed without estimating or reporting what that variance actually is.

**What evidence exists in the paper.** The absence of statistical rigor is evident from the tables themselves: no ± values, no error bars, no significance annotations. The paper does not describe a hyperparameter tuning protocol, validation split, or cross-validation procedure anywhere in the main text or appendices. The experimental setup (Section 5.1) describes the benchmarks, models, baselines, and hyperparameters but does not address how hyperparameters were selected or how results should be interpreted statistically. The paper does not report the number of experimental trials or whether results were averaged over multiple runs.

**Mitigation status.** Not addressed. The paper does not acknowledge the lack of statistical rigor as a limitation. This is a significant departure from the standards demonstrated in the reference example paper, which used two-fold cross-validation, oracle vs. predicted difficulty comparisons, and transparent reporting of strategy selection methodology. For a systems paper that makes comparative performance claims as its primary contribution, the absence of basic statistical validation weakens the reliability of all reported performance rankings.

## 7. Implications and Future Directions
- How it shifts the landscape
  - Demonstrates that million‑token prompts are feasible on a single commodity‑class GPU without discarding context and without fine‑tuning, by co‑designing sparse attention, positional indexing, and memory management. This lowers the barrier to real‑world long‑context applications.
- Practical applications
  - Retrieval‑augmented generation over massive corpora, contract or codebase analysis, long‑horizon agents and logs, multi‑document summarization and QA, and interactive assistants that must retain full conversation history.
- Follow‑up research
  - Reducing TTFT: integrate speculative decoding/prefill or lazy initialization specifically tailored to modular pruning (Appendix G).
  - Memory efficiency: combine with KV quantization/compression (e.g., KVQuant, DeepSeek‑style compression) inside the same offloading framework (Appendix G).
  - Adaptive pruning policies: learn per‑layer or per‑task stage parameters online; develop reliability checks that detect when chunk sparsity fails and fall back gracefully.
  - Broader hardware support: extend UVM‑style paging and graph‑capturable kernels to non‑Nvidia accelerators; multi‑GPU paging across NVLink.
  - Training‑time synergy: pretraining with objectives that sharpen chunk sparsity or encourage stable attention locality could further improve pruning accuracy and OOL robustness.

Overall, InfiniteHiP is a well‑engineered, training‑free path to practical long‑context LLM inference. Its key strength is the coherent combination of algorithmic, architectural, and systems ideas that together deliver both speed and accuracy at unprecedented context scales, as evidenced across benchmark quality, latency, and throughput (Tables 1–5; Figures 3–5).
