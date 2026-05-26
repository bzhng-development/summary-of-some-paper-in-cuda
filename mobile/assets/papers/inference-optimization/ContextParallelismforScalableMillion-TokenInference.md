# Context Parallelism for Scalable Million-Token Inference

**ArXiv:** [2411.01783](https://arxiv.org/abs/2411.01783)

## 🎯 Pitch

This paper introduces context parallelism for inference in large language models, enabling near-linear latency scaling for million-token contexts by distributing computation and memory across up to 128 GPUs—without altering model architecture or relying on approximations. The system’s innovations, including two lossless ring attention variants (pass-KV and pass-Q), make it possible to serve exact attention over million-token prompts in just seconds, unlocking practical real-world applications and dramatically reducing latency for LLM-powered systems with ultra-long contexts.

---

## 1. Executive Summary

This paper introduces and systematically optimizes **context parallelism** for long-context LLM inference, developing two lossless exact ring attention variants — **pass-KV** (communicating key/value embeddings around the ring, suited for full prefill and low KV cache hit rates) and **pass-Q** (communicating query embeddings, suited for decode and partial prefill with high cache hit rates) — along with a runtime heuristic that adaptively selects between them based on the KV cache miss rate. Evaluated on Llama3 405B with up to 128 H100 GPUs across 16 nodes, the method achieves near-linear scaling for long-context prefill, delivering 1M-token context prefill in 77 seconds at 93% parallelization efficiency and 63% FLOPS utilization, while 128K-token prefill completes in 3.8 seconds. The approach maintains this scalability even on medium-to-low inter-host bandwidth networks (TCP/IP at 100 Gb/s per GPU), establishing that ring attention with adaptive KV-versus-Q routing can overlap communication with computation across a wide range of context lengths, but only when the new-token length exceeds a model-and-hardware-dependent threshold that keeps communication hidden beneath attention computation.

## 2. Context and Motivation

### The Core Problem: Long-Context Inference Latency Is Crushing

The paper addresses a specific, acute scaling problem in LLM deployment: **as context windows grow from 128K to 1M tokens, the latency of processing a single prompt becomes unacceptably long — even on the most powerful single-node GPU configurations available.** The authors present concrete numbers to motivate this: with a single H100 GPU host (8 GPUs), serving a 128K context length with the Llama3 405B model takes approximately 60 seconds, while a 1M context length would require roughly 1200 seconds — 20 minutes — just to produce the first token (Section 1). For a real-time conversational application, this is completely non-viable.

This latency derives primarily from the **quadratic complexity of dense self-attention** with respect to context length. As the authors note in Section 2.2, while the feed-forward layers of a Transformer have linear FLOPs-per-token cost ($2 \cdot W$ matrix multiplication FLOPs per token during forward pass, where $W$ is the parameter count), the attention mechanism incurs a cost proportional to $O(T^2)$ for context length $T$. At 128K–1M tokens, attention computation dominates the total prefill time completely. For the Llama3 405B configuration (126 layers, $D = 16384$, 128 query heads, 8 KV heads), the authors calculate in Appendix A that for a 1M-token prefill, attention FLOPs ($4.1 \times 10^{18}$) dominate GEMM FLOPs ($8.1 \times 10^{17}$) by roughly $5\times$ — meaning more than 80% of the total computation is attention.

The problem is not merely academic. Modern LLM products are racing to support ever-longer context windows: OpenAI's GPT-4o supports 128K tokens, Anthropic's Claude supports 200K tokens, and Google's Gemini 1.5 Pro supports 1M tokens (Section 1). These context windows enable qualitatively different applications — processing entire books, analyzing hour-long videos, conducting multi-hour conversations with full history retention — but they also create a deployment crisis: **how do you serve these massive context windows with acceptable latency and cost?**

### Why This Problem Matters: The Prefill Bottleneck

The distinction between **prefill** and **decode** phases is central to understanding why long-context inference is particularly hard. During prefill (Section 3.3), the model processes the entire input prompt at once, computing attention between all pairs of tokens. The key and value tensors produced during this phase are cached as **KV cache** in GPU HBM for subsequent use. During decode, the model generates one token at a time autoregressively, with each new token attending to the full KV cache. After the model responds, a user follow-up prompt triggers **partial prefill** (or persistent KV prefill), where new tokens attend to both themselves and the previously cached KV entries.

The prefill phase is the latency culprit because:

1. **It's compute-bound, not memory-bound.** The $O(T^2)$ attention computation overwhelms any single GPU's compute capacity. Unlike decode (where generating one token at a time makes each step relatively cheap), prefill must do all pairwise attention computations up front.

2. **It's on the critical path for user experience.** Time-to-first-token (TTFT) — the delay between submitting a prompt and seeing the first output token — is what the user directly perceives. If prefill takes 60 seconds, the user waits 60 seconds before seeing any response, regardless of how fast subsequent tokens arrive.

3. **It scales quadratically, not linearly.** Doubling the context length from 64K to 128K doesn't double the prefill time — it roughly quadruples it for the attention portion. This means that as models push toward 1M+ context windows, the single-node latency becomes catastrophic.

The paper's framing makes clear that this isn't a problem you can solve by just buying a bigger GPU — the compute requirement grows faster than any single accelerator's capacity at these context scales. You *must* distribute the computation across multiple devices, but how you distribute it dramatically affects whether you actually get proportional speedup.

### Prior Approaches and Where They Fall Short

The paper identifies several categories of prior work and systematically explains why each is insufficient for the specific challenge of long-context inference latency:

#### Approximate and Sparse Attention Methods

Several approaches reduce the $O(T^2)$ attention cost by making attention sparse or approximate (Section 2.2): window attention (Liu et al., 2021), local attention (Xiong et al., 2021), Linformer (Wang et al., 2020), semi-local sparse attention (Jiang et al., 2024; Beltagy et al., 2020), and various retrieval-augmented sparse approaches. These techniques modify the attention mechanism itself to avoid computing all pairwise token interactions.

**Why they fall short for this paper's goals:** The authors explicitly position their work in the third category — system-level optimizations that preserve exact dense attention — and note that their method "can be used in conjunction with methods from the other two categories with minor or no modifications" (Section 2.3). The key limitation of approximate attention is that it changes the model's computation, potentially affecting output quality. For deployments where faithfulness to the pretrained model's exact attention pattern is required (e.g., reproducing published benchmark results, serving models where any approximation hasn't been validated), lossless exact attention remains necessary. Moreover, approximate methods address the *total work* problem ($O(T^2)$ FLOPs) but don't address the *distribution* problem — even if you reduce the work, you still need to parallelize what remains across GPUs to achieve acceptable latency.

#### New Model Architectures and Post-Training Modifications

The paper mentions Infini-attention (Munkhdalai et al., 2024) as an example of architectural changes that build long-context comprehension into pretraining, and Attention Sinks (Xiao et al., 2023) as a post-training modification that extends shorter-context models to longer or infinite context windows.

**Why they fall short:** These approaches require either retraining the model from scratch (architectural changes) or accepting potential quality degradation from post-hoc modifications. For serving existing pretrained models like Llama3 405B — which was trained with exact dense attention — system-level approaches that don't modify the model are more immediately deployable and don't require re-validation of the model's output quality on downstream tasks.

#### KV Cache Compression Methods

Techniques like KV cache quantization (Hooper et al., 2024; Lin et al., 2024) — storing KV entries in INT4, INT8, or FP8 instead of FP16/BF16 — and architectural choices like Grouped Query Attention (GQA; Ainslie et al., 2023) and Multi-Query Attention (MQA; Shazeer, 2019) reduce the memory footprint of long-context inference.

**Why they fall short for latency:** These methods address the **memory capacity** problem — fitting the KV cache for a long context into GPU HBM — but they do not directly address the **computation latency** problem. The attention computation itself still requires processing all KV entries, even if they're stored more compactly. (The paper does leverage FP8 quantization for model weights to fit Llama3 405B into a single node with TP8, and notes that GQA's reduced KV head count — 8 KV heads vs. 128 query heads in Llama3 405B — is crucial for making their pass-KV algorithm communication-efficient, as discussed below. But these are enablers, not solutions to the latency problem per se.)

#### Paged Attention and Memory Management

PagedAttention (Kwon et al., 2023) provides efficient virtual-memory-like management of KV cache entries, reducing fragmentation and enabling larger effective batch sizes.

**Why it falls short:** Like KV cache compression, this addresses the memory management aspect of long-context serving but not the fundamental compute latency of prefill.

#### Multi-Node Tensor Parallelism (TP)

Tensor parallelism (Shoeybi et al., 2019) partitions the weights of linear layers across GPUs and has been the standard approach for distributing large models that don't fit on a single GPU. When scaled to multiple nodes, TP communicates activations between GPUs using AllReduce collectives.

**Why it falls short for multi-node long-context inference:** The paper identifies a specific, fundamental limitation of TP when scaled across nodes — **inter-host communication bandwidth.** As detailed in Section 4.2.2 and Table 2, TP requires AllReduce communication on the outputs of every linear layer, which transmits $T \cdot N_H \cdot D_H$ bytes per transformer block. For Llama3 405B with $N_H = 128$ and $D_H = 128$ (since $D = D_H \cdot N_H = 16384$), this is substantial. When GPUs are within the same node (connected by NVLink with ~900 GB/s bidirectional bandwidth), this communication is fast enough to not bottleneck computation. But when scaling across nodes connected by RDMA (400 Gb/s ≈ 50 GB/s per GPU) or TCP/IP (100 Gb/s ≈ 12.5 GB/s per GPU) — which are 18× to 72× slower than NVLink — the AllReduce latency becomes the dominant factor, preventing near-linear scaling.

The paper makes this concrete in Figure 7: when scaling TP from 2 to 8 nodes, the scaling ratio (ideal latency improvement) flattens dramatically, while CP maintains near-perfect scaling. At 2 nodes, TP16 is only about 15% slower than CP2. At 8 nodes, TP is approximately 2× worse — half the effective throughput compared to CP8. The gap widens with more nodes because AllReduce communication grows with the number of participating GPUs in the collective.

#### Pipeline Parallelism (PP)

Pipeline parallelism (Huang et al., 2019; Narayanan et al., 2021) shards model layers across GPUs and processes micro-batches in a pipelined fashion.

**Why it falls short for latency:** The paper explicitly notes (Section 1) that PP "improves throughput but not latency." PP is designed to keep more GPUs busy when processing many requests simultaneously by overlapping computation across pipeline stages. But for a *single* request — which is the relevant scenario for interactive long-context prefill where a user submits one prompt and waits for the response — PP provides no latency improvement because each stage must still wait for the previous stage's output before it can begin. If your goal is to reduce the 60-second wait for a single 128K prefill, PP doesn't help.

#### Prior Work on Sequence/Context Parallelism for Training

Ring attention and related sequence parallelism techniques have been explored for *training* long-context models — most notably, ring attention with blockwise transformers (Liu et al., 2023), DeepSpeed Ulysses (Jacobs et al., 2023), DistFlashAttn (Li et al., 2023), and striped attention (Brandon et al., 2023). These works demonstrated that passing KV embeddings around a ring of GPUs could effectively parallelize attention computation during training.

**Why they fall short for inference:** The paper identifies critical differences between training and inference that make prior ring attention approaches insufficient:

- **Training assumes uniform sequence lengths:** In training, all sequences in a batch typically have the same (padded) length. Inference, especially in multi-turn conversational settings, involves variable-length sequences within a batch and uneven KV cache distributions across GPUs due to previous turns.

- **Training doesn't deal with persistent KV cache:** During training, there is no concept of a pre-existing KV cache from previous turns — all tokens are processed from scratch each forward pass. Inference, particularly partial prefill in multi-turn chat, must compute attention between new tokens and a large, pre-existing KV cache that may be unevenly distributed across GPUs.

- **Training doesn't need to switch between passing KV and passing Q:** In training, $T$ (new tokens) always equals the full sequence length, and $P = 0$ (no cached KV). The communication pattern of always passing KV works well in this regime. In inference partial prefill, $T$ can be much smaller than $P$ (e.g., a short follow-up question after a long conversation history). In this case, communicating the full KV cache around the ring is wasteful — it's much more efficient to communicate the smaller query tensor instead.

- **Training optimizes throughput, not latency:** Training systems are designed to maximize tokens processed per second across many sequences, not to minimize the wall-clock time for a single sequence's prefill. This changes what constitutes "good" overlap between communication and computation — latency-critical inference requires careful attention to whether communication is fully hidden under computation, not just whether the pipeline is kept busy.

### How This Paper Positions Itself

The paper's positioning emerges clearly from this landscape:

**1. System-level, lossless, exact attention.** The paper explicitly places itself in the third category of approaches (Section 2.3): "preserve the model architecture, instead improve the scalability of existing dense attention algorithms to leverage more compute resources." This means the method produces mathematically identical outputs to single-GPU exact attention — there is no approximation, no quality degradation, and no need to re-validate model outputs. This is crucial for practical deployment because it means the optimization is transparent to the model and all downstream applications.

**2. First comprehensive treatment of context parallelism for inference, not training.** The paper claims (Section 1): "To the best of our knowledge, this is the first paper to disclose the system implementation details on applying context parallelism in inference scenario." The emphasis is on *inference-specific* challenges: multi-turn prefill and decode, persistent KV cache management, latency optimization (not just throughput), and adaptive algorithm selection based on the runtime state of the KV cache.

**3. TP within nodes, CP across nodes — exploiting the bandwidth hierarchy.** The paper's architectural choice (Section 4.1, Figure 5) is to use TP8 within each node (where NVLink provides high intra-node bandwidth) and CP across nodes (where inter-host bandwidth is much lower). This leverages CP's key advantage over TP identified in Table 2: CP's communication traffic ($T \cdot N_{KV} \cdot D_H$ bytes per transformer block) is substantially smaller than TP's ($2 \cdot T \cdot N_H \cdot D_H$ bytes per block) for GQA models where $N_{KV} \ll N_H$. For Llama3 405B with 128 query heads and 8 KV heads, CP communicates **16× less data** than TP on attention layers — making it far more tolerant of the lower inter-node bandwidth.

**4. Adaptive algorithm selection as the key novelty.** Rather than proposing a single ring attention algorithm and claiming it works for all scenarios, the paper develops two complementary variants — pass-KV and pass-Q — with an analytical model and runtime heuristic that selects between them based on $T$, $P$, and hardware-specific constants. This is not a trivial design choice; it's the mechanism that makes the system work across the full spectrum from full prefill ($P=0$, where pass-KV dominates) to partial prefill with high cache hit rates (where pass-Q dominates) to decode ($T=1$, where pass-Q is almost always preferred). The heuristic in Algorithm 1 (or its refined version in Appendix C, Algorithm 5) encodes the insight that the optimal choice depends on the ratio $\frac{T}{T+P}$ (the KV cache miss rate) relative to the model-specific constant $2 \cdot \frac{N_{KV}}{N_H}$, and on whether the absolute value of $T$ is large enough to hide pass-KV communication under attention computation (Equation 2).

**5. Load-balanced sharding for variable-length sequences.** Prior ring attention work for training assumed uniform sequence lengths (all sequences padded to the same length within a batch). The paper introduces a two-chunk sharding scheme (Section 3.5.1, Figures 1 and 2) where each CP rank receives two chunks from the sequence — $(C_i, C_{2N-i-1})$ — which balances both the attention computation and the KV cache memory load across ranks. This is non-trivial because causal attention means tokens at different positions have different amounts of "preceding context" to attend to; naive sequential partitioning would give later ranks more work. The two-chunk scheme exploits the symmetry of the attention pattern to achieve balance.

**6. Practical validation across network regimes.** The paper demonstrates scalability not just on high-bandwidth RDMA networks (400 Gb/s per GPU on Grand Teton Training systems) but also on lower-bandwidth TCP/IP networks (100 Gb/s per GPU on Grand Teton Inference systems), showing that the approach remains effective on "common commercial data center with medium-to-low inter-host bandwidth" (Section 1, Figure 6). This is important because not all deployments have access to specialized high-bandwidth training clusters — production inference often runs on more cost-effective networking.

### The Deeper Significance

Beyond the immediate engineering contribution, the paper addresses a structural tension in LLM deployment: **the divergence between model capability scaling (context length) and hardware capability scaling (single-GPU compute).** Context lengths are growing super-linearly relative to GPU compute capacity — a single H100 can process 128K tokens in 60 seconds with Llama3 405B, but 1M tokens would take 1200 seconds, a 20× increase for an 8× increase in context length (due to the quadratic attention cost). Without effective distributed inference strategies like the one presented here, the industry trend toward longer context windows would be blocked by the physics of single-device compute limits.

The paper also resolves a specific tension in the parallelism design space: **when scaling across nodes, what should you parallelize?** The implicit answer from prior TP-heavy approaches was "parallelize the model weights" — but the paper demonstrates that for long-context inference, parallelizing the *input sequence* (context parallelism) is dramatically more communication-efficient. This is not obvious a priori — CP increases memory consumption because model weights aren't sharded (each CP node holds the full model), which means you need TP within each node anyway to fit the model. The combination CP+TP (CP across nodes, TP within nodes) emerges as the sweet spot that navigates both the bandwidth constraint (CP's smaller messages) and the memory constraint (TP's weight sharding).

## 3. Technical Approach

### 3.1 Reader Orientation

The paper builds a **distributed inference system** that takes a single long-context prompt (up to 1M tokens) and a large language model (Llama3 405B), spreads the attention computation across up to 128 GPUs on 16 server nodes, and produces the model's output — all while computing **exactly the same attention outputs** as a single-GPU run. The core idea is simple in concept but subtle in execution: split the input sequence across GPUs, let each GPU compute attention for its chunk of tokens against everyone else's key/value embeddings by passing those embeddings around a ring, and use a runtime heuristic to decide *what* to pass around the ring (keys/values or queries) based on how much cached conversation history already exists.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has four major components, arranged in a nested hierarchy:

1. **Tensor parallelism within each node (TP8)** — The 405B-parameter model is too large for a single GPU. Within each 8-GPU node, the model's feed-forward and attention projection weights are sharded across all 8 GPUs using standard tensor parallelism. Each GPU holds a fraction of every layer's weights. This handles the memory capacity problem and provides fast intra-node communication over NVLink (~900 GB/s bidirectional per GPU).

2. **Context parallelism across nodes (CP N)** — For long-context prompts, the input sequence is split into chunks distributed across $N$ nodes. Each node holds the *full model weights* (replicated, not sharded across nodes) but processes only its assigned chunk of the input sequence. The only cross-node communication happens during attention: nodes exchange key/value or query embeddings so each node can compute the full attention output for its local chunk. This is where the ring attention algorithms (pass-KV and pass-Q) live.

3. **Per-KV-head communication groups** — There are 8 KV heads in Llama3 405B. For each KV head, one GPU from each node (the one holding that head in its TP shard) participates in a ring communication group. With $N$ nodes and TP8 within each node, this creates 8 independent rings, each with $N$ GPUs. The rings operate in parallel.

4. **Adaptive algorithm selector** — A runtime heuristic that, given the length of new input tokens ($T$) and the length of previously cached KV tokens ($P$), chooses whether to run pass-KV (communicate key/value embeddings around the ring) or pass-Q (communicate query embeddings around the ring). This decision is made once per partial prefill and is based on a static model-and-hardware-specific threshold plus the current KV cache miss rate ($T/(T+P)$).

Information flows as follows: a prompt enters the system → the sequence is split using load-balanced sharding across CP ranks → within each CP rank, the 8 GPUs in the TP group compute their shard of linear projections → when attention is reached, the ring algorithm executes (either pass-KV or pass-Q) → partial attention outputs are merged → the remaining transformer layers continue within each TP group → output logits are produced.

### 3.3 Roadmap for the Deep Dive

- **First**, load-balanced sharding — how sequences are split across CP ranks to keep both attention computation and KV cache memory balanced across GPUs, since uneven sharding would cause some ranks to run out of memory before others while also creating straggler ranks that slow the whole ring.
- **Second**, the ring pass-KV algorithm — the core mechanism for full prefill and low-KV-cache-hit partial prefill, where key/value embeddings circulate around the ring and each rank computes attention for its local queries against the currently-held KV chunk.
- **Third**, the ring pass-Q algorithm — the alternative mechanism for decode and high-KV-cache-hit partial prefill, where query embeddings circulate around the ring while key/value embeddings stay stationary, requiring an additional All2All step to restore partial outputs to their source ranks.
- **Fourth**, the merge attention operation — the mathematical procedure (derived from Online Softmax) that combines partial attention outputs from different KV chunks into the exact result that standard attention would produce, which is used by both pass-KV and pass-Q.
- **Fifth**, the analytical model for algorithm selection — the communication-overlap analysis that tells the system whether pass-KV or pass-Q will deliver lower latency for a given $(T, P)$ pair, and why the threshold depends on model-specific constants like $N_{KV}/N_H$ and hardware-specific constants like $C/BW$.
- **Sixth**, batched ring pass-Q decode — how the decode phase (generating one token at a time) is handled under context parallelism, including the round-robin sharding scheme that prevents load imbalance in the KV cache across ranks.
- **Seventh**, the CP+TP nesting design — why TP is used within nodes and CP across nodes, the per-KV-head communication group structure, and how this exploits the bandwidth hierarchy of modern GPU clusters.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems paper** whose core idea is that context parallelism with adaptive ring attention variants can achieve near-linear scaling for long-context LLM inference latency by carefully matching the communication pattern (what gets passed around the ring) to the runtime characteristics of the request (full prefill, partial prefill with high/low KV cache hit rate, or decode).

---

#### Load-Balanced Sharding of Input Sequences

The first problem that must be solved before any ring attention algorithm can run is: **how do you split a sequence of tokens across $N$ GPUs so that every GPU does roughly the same amount of work?** Naive contiguous partitioning — giving rank 0 tokens $0$ through $T/N-1$, rank 1 tokens $T/N$ through $2T/N-1$, and so on — fails because of the causal attention mask. In causal self-attention, each token attends to itself and all tokens *before* it in the sequence. This means rank 0 (holding the earliest tokens) attends to very few tokens, rank 1 attends to more, and rank $N-1$ attends to the most. The work is imbalanced: the last rank does substantially more computation and stores more KV cache entries.

The paper adopts a load-balancing scheme (Section 3.5.1) previously explored in Cho et al. (2024) and Brandon et al. (2023). The idea is to interleave chunks so that every rank gets a mix of early and late positions, balancing the total number of preceding tokens each rank must attend to.

**The sharding procedure for a single sequence:**

1. Partition the sequence into $2N$ equally-sized chunks: $C_0, C_1, \ldots, C_{2N-1}$.
2. Assign to CP rank $i$ the two chunks $(C_i, C_{2N-i-1})$, for $i = 0, 1, \ldots, N-1$.

**Concrete example with $N = 4$:** Rank 0 gets $(C_0, C_7)$; Rank 1 gets $(C_1, C_6)$; Rank 2 gets $(C_2, C_5)$; Rank 3 gets $(C_3, C_4)$. Rank 0 holds the very first chunk (which has few preceding tokens to attend to) and the very last chunk (which has many). Rank 3 holds two middle chunks, each with an intermediate number of preceding tokens. The symmetry of the allocation — each rank gets one chunk from the first half and one from the second half — ensures the total number of causal predecessor tokens is approximately equal across ranks.

**Why this works:** In causal attention, a token at absolute position $p$ in the original sequence must attend to $p$ predecessor tokens (including itself). For the two chunks assigned to rank $i$, the chunks start at approximate positions $i \cdot (T/2N)$ and $(2N-i-1) \cdot (T/2N)$. The sum of these starting positions across the two chunks is approximately $(i + 2N - i - 1) \cdot (T/2N) = (2N - 1) \cdot (T/2N)$, which is independent of $i$. Every rank therefore has roughly the same total amount of "history" to process.

**For fused variable-length inputs (batched prefill):** The paper supports batching multiple sequences of different lengths in one prefill. The load-balanced sharding is applied to each sequence independently (Figure 1). Each sequence is padded to a length divisible by $2N$ if necessary. The sharded chunks from all sequences are then concatenated into the batch dimension on each rank, producing a fused input where every rank processes its assigned chunks from all sequences in the batch.

**For partial prefill with cached KV:** When a user submits a follow-up prompt after a conversation, there are $T$ new tokens and $P$ previously cached KV tokens. The load-balanced sharding is applied *only to the new tokens* (the $T$ dimension), regardless of how the cached KV tokens are distributed across ranks (Figure 2). The cached KV embeddings maintain whatever distribution they had from previous prefill and decode turns. This is critical because the cached KV distribution may already be slightly uneven due to padding from previous turns, but the paper accepts this minor imbalance rather than redistributing the KV cache (which would be expensive).

**Design choice — why not redistribute cached KV?** Redistributing the KV cache across ranks before each partial prefill would require an all-to-all shuffle of potentially millions of embedding vectors, which would add latency and bandwidth consumption that likely exceeds any benefit from perfect balance. The paper instead relies on the observation that the imbalance from padding is small (proportional to at most $N-1$ tokens per sequence) and that the two-chunk sharding of new tokens keeps the total work approximately balanced even with minor cached-KV asymmetry.

---

#### Ring Pass-KV: The Algorithm for Full Prefill and Low Hit-Rate Partial Prefill

The ring pass-KV algorithm (Section 3.5.2, Figure 3) is the primary mechanism for parallelizing attention when most or all of the context consists of new tokens — specifically, full prefill ($P = 0$) and partial prefill with low KV cache hit rate (where the number of new tokens $T$ is large relative to cached tokens $P$).

**Why pass KV instead of Q in this regime:** The intuition comes from the communication size analysis in Equation (1). For a GQA model, the key and value embeddings together have size proportional to $2 \cdot (T+P) \cdot N_{KV}/N_H \cdot D$, while the query embeddings have size proportional to $T \cdot D$. When $T/(T+P)$ is large (most tokens are new), the KV tensors may be smaller than the Q tensors if $2 \cdot N_{KV} < N_H$, which is true for Llama3 405B ($2 \cdot 8 = 16 < 128$). Passing the smaller tensor around the ring reduces communication volume. Additionally, when $T$ is large, the attention computation takes long enough that the SendRecv communication for the KV chunks can be fully overlapped with computation — the GPU computes attention for one chunk while simultaneously receiving the next chunk from the previous rank.

**The algorithm in detail (Algorithm 2, Figure 3):**

Assume we have $N$ CP ranks $\text{CP}_0, \text{CP}_1, \ldots, \text{CP}_{N-1}$. Rank $k$ holds:
- Query tensor $Q_k$ — its local shard of the new input tokens
- Key tensor $K_k$ and value tensor $V_k$ — its local shard of all tokens (new + cached), which we denote collectively as $\text{KV}_k$

The goal is for rank $k$ to compute the attention output of $Q_k$ against $\text{KV}_0, \text{KV}_1, \ldots, \text{KV}_{N-1}$ — the KV embeddings from all ranks.

**Step 1: Padding for equal-sized messages.** The ring communication requires that every rank sends and receives messages of the same size at each step. However, different ranks may hold different amounts of cached KV due to padding from previous turns. To handle this, each rank pads its local KV to the maximum length across all ranks:

$$L_i = \max_{0 \leq j < N} (P^i_j + T^i_j)$$

where $P^i_j$ is the number of cached tokens from sequence $i$ sharded to rank $j$, and $T^i_j$ is the number of new prefill tokens from sequence $i$ sharded to rank $j$. For fused batched inputs, each sequence is padded individually:

$$\text{KV}^k_k = \text{concat}^{B-1}_{i=0} \big( \text{pad}(P^i_k + T^i_k, L_i) \big)$$

The padding ensures all ranks have KV tensors of identical total length, making SendRecv straightforward on standard collective communication primitives.

**Step 2: The ring loop.** The algorithm iterates $N$ times (for $j = 0$ to $N-1$). At iteration $j$:
- Rank $k$ sends its currently-held KV chunk $\text{KV}^s_k$ to the next rank $(k+1) \bmod N$, where $s = (k - j) \bmod N$ indicates which rank's KV data is currently on rank $k$.
- Rank $k$ simultaneously receives $\text{KV}^s_p$ from the previous rank $p = (k-1) \bmod N$.
- Rank $k$ computes partial attention: $O^s_k = \text{GQA}(Q_k, \text{KV}^s_k)$ — the attention of its local queries against the KV chunk it currently holds.
- Rank $k$ replaces its held KV chunk with the newly received one: $\text{KV}^s_k \leftarrow \text{KV}^s_p$.

After $N$ iterations, each rank has computed $O^s_k$ for $s = 0, 1, \ldots, N-1$ — the attention of its queries against every other rank's KV data.

**Step 3: Merge attention.** The partial attention outputs $O^0_k, O^1_k, \ldots, O^{N-1}_k$ are combined into a single attention output $O_k$ using the merge attention operation described in Appendix B. This merge mathematically combines the per-chunk softmax statistics to produce the exact same result as if attention had been computed against all KV tokens at once.

**What physically happens during the loop (the overlap):** At iteration $j$, while the GPU computes $\text{GQA}(Q_k, \text{KV}^s_k)$ — the attention of its local queries against the KV data from rank $s$ — it simultaneously executes the SendRecv for $\text{KV}^{s-1}$ (the KV data from the previous rank, which it has already processed and is now forwarding, or receiving the next chunk). The attention computation kernel and the communication transfer execute on separate hardware units on the GPU (the tensor cores / CUDA cores for computation, the NVLink or network interface for communication), enabling true overlap. The effectiveness of this overlap is what determines whether pass-KV achieves near-linear scaling: if the attention compute time for one chunk exceeds the SendRecv time for one message, then communication is fully hidden and each additional node contributes proportionally more compute power with no communication penalty.

**Modified ring for partial prefill with fused variable-length sequences (Algorithm 2):** The key adaptation over standard ring attention (Liu et al., 2023) is handling the fused batched inputs and the padding to equal KV lengths. The loop structure is identical, but the per-rank KV tensor is now a concatenation of padded per-sequence chunks rather than a single uniformly-sized block.

**Why this algorithm is preferred for full prefill and low-hit-rate partial prefill:** The communication volume per step is proportional to $2 \cdot (T+P) \cdot N_{KV}/N_H \cdot D \cdot e / N$ (approximately, since each rank holds roughly $1/N$ of the total KV). The computation per step is proportional to $T \cdot (T+P) \cdot D / N^2$ (since each rank holds $T/N$ queries that attend to $(T+P)/N$ KV entries). The ratio of computation to communication grows with $T$, meaning that for sufficiently large $T$, the computation always dominates and communication is hidden.

---

#### Ring Pass-Q: The Algorithm for Decode and High Hit-Rate Partial Prefill

The ring pass-Q algorithm (Section 3.5.3, Figure 4) is the complement to pass-KV: instead of circulating KV embeddings around the ring while queries stay stationary, pass-Q circulates query embeddings while key and value embeddings remain on their original ranks.

**Why pass Q instead of KV in this regime:** When the KV cache hit rate is high — meaning $T$ (new tokens) is small relative to $P$ (cached tokens) — passing KV would require communicating almost the entire cached context around the ring, even though most of it is unchanged from the previous turn. Communicating the smaller query tensor is dramatically more efficient. The break-even condition from Equation (1) is:

$$\frac{T}{T+P} \leq 2 \cdot \frac{N_{KV}}{N_H}$$

For Llama3 405B ($N_{KV} = 8, N_H = 128$), $2 \cdot N_{KV} / N_H = 16/128 = 0.125 = 12.5\%$. This means pass-Q is preferred when fewer than 12.5% of the total tokens in the sequence are new — a very common scenario for multi-turn chat where a user follows up with a short question after a long conversation history.

**The algorithm in detail (Algorithm 3, Figure 4):**

Rank $k$ holds:
- Query tensor $Q_k$ — its local shard of the new input tokens, load-balanced as described in Section 3.5.1.
- Key tensor $K_k$ and value tensor $V_k$ — its local shard of *all* tokens (new + cached), denoted $\text{KV}_k$. These remain stationary on rank $k$ throughout the algorithm.

**Step 1: The ring loop.** The algorithm iterates $N$ times (for $j = 0$ to $N-1$). At iteration $j$:
- Rank $k$ sends its currently-held query chunk $Q^s_k$ to the next rank, where $s = (k - j) \bmod N$.
- Rank $k$ simultaneously receives $Q^s_p$ from the previous rank.
- Rank $k$ computes partial attention: $O^s_k = \text{GQA}(Q^s_k, \text{KV}_k)$ — the attention of the received queries against its *local, stationary* KV data.
- Rank $k$ replaces its held query chunk with the newly received one.

After $N$ iterations, rank $k$ has computed $O^0_k, O^1_k, \ldots, O^{N-1}_k$ — but these are *not* the outputs that correspond to rank $k$'s original queries. They are the attention outputs for the queries that *visited* rank $k$ during the ring loop. Specifically, $O^s_k$ is the attention output from applying queries originally allocated to rank $s$ against the KV data stationary on rank $k$.

**Step 2: Permutation and All2All to restore outputs.** This is where pass-Q fundamentally differs from pass-KV. After the ring loop completes, each rank has partial attention outputs scattered across all ranks: the output for query chunk $Q_0$ is split across $O^0_0, O^0_1, \ldots, O^0_{N-1}$ (one piece on each rank). To reassemble the complete attention output for each query chunk, an **All2All collective communication** is needed:
- Each rank permutes its local outputs to group them by source rank.
- An All2All exchange sends each partial output to the rank that owns the corresponding queries.
- After All2All, rank $k$ receives $O^k_s$ for $s = 0, 1, \ldots, N-1$ — the attention results for its original queries against all KV chunks.

**Step 3: Merge attention.** As with pass-KV, the received partial attention outputs are merged using the merge attention operation (Appendix B, Equation 4).

**The additional communication cost:** The All2All step introduces an extra communication phase that is not present in pass-KV. This All2All is on the critical path — it cannot be overlapped with computation because the merge attention operation requires all partial outputs to be present on the source rank before it can proceed. The analytical model in Section 3.4 (detailed later) accounts for this All2All cost when determining whether pass-Q will actually be faster than pass-KV, even when the communication size argument favors passing Q.

**Why pass-Q still wins despite the All2All:** When $T/(T+P)$ is very small (e.g., 1-5% cache miss rate), the query tensors are tiny compared to the KV tensors. Communicating the KV tensors around the ring would require sending $2 \cdot (T+P) \cdot N_{KV}/N_H \cdot D$ bytes per step, while communicating Q requires only $T \cdot D$ bytes per step. The All2All cost — roughly $(N-1) \cdot (D+1) \cdot T \cdot e / \text{BW}$ for the partial attention outputs and their softmax log-sum-exp statistics — is also proportional to $T$, not $(T+P)$. So for sufficiently small $T/(T+P)$, the total communication volume of pass-Q (ring SendRecv for Q + All2All for partial outputs) is much less than pass-KV (ring SendRecv for KV). The All2All becomes the dominant term only when all communication is exposed (not hidden under computation), which happens precisely when $T$ is too small to keep the GPU compute units busy during the ring loop.

---

#### Merge Attention: Combining Partial Softmax Outputs Exactly

The merge attention operation (Appendix B) is the mathematical mechanism that makes both ring attention variants *lossless* — producing identical outputs to single-GPU attention. It is not a new contribution (it derives from Online Softmax by Milakov & Gimelshein, 2018, and is used in Flash Attention by Dao et al., 2022), but understanding it is essential to understanding why the ring algorithms work.

**The problem:** Each CP rank computes attention of some queries against a *subset* of the total KV entries. The standard attention formulation computes a softmax over *all* KV entries simultaneously:

$$O = \text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

where the softmax normalizes over all $R$ key positions. If we split the keys into two chunks $K_0$ (length $R_0$) and $K_1$ (length $R_1$) and compute attention separately, we cannot simply average or concatenate the results — the softmax denominators are different (one normalizes over $R_0$ positions, the other over $R_1$ positions).

**The merge formula (Equation 4):** For a given query, let $(\text{LSE}_s, O_s)$ be the log-sum-exp and attention output computed against KV chunk $s$, where:

$$\text{LSE}_s = \log \sum_{i=0}^{R_s-1} \exp\left(\frac{Q \cdot K^T_{s,i}}{\sqrt{d}}\right)$$

$$O_s = \frac{\sum_{i=0}^{R_s-1} \exp\left(\frac{Q \cdot K^T_{s,i}}{\sqrt{d}}\right) \cdot V_{s,i}}{\exp(\text{LSE}_s)}$$

The merged output for the query attending to all chunks is:

$$O = \frac{\sum_{s=0}^{N-1} O_s \times \exp(\text{LSE}_s - \text{LSE}_{\text{max}})}{\sum_{s=0}^{N-1} \exp(\text{LSE}_s - \text{LSE}_{\text{max}})}$$

where $\text{LSE}_{\text{max}} = \max_{s=0}^{N-1} \text{LSE}_s$.

**What this computes:** Each partial output $O_s$ is weighted by its exponentiated log-sum-exp (relative to the maximum LSE across all chunks), and the result is re-normalized by the sum of these weights. The subtraction of $\text{LSE}_{\text{max}}$ before exponentiation is a numerical stability trick — without it, exponentiating large LSE values could overflow floating-point representation.

**Why this form produces exact results:** The operation is algebraically equivalent to computing the softmax over the concatenation of all chunks. The numerator of the merged output is:

$$\sum_s O_s \cdot \exp(\text{LSE}_s) = \sum_s \sum_i \exp(Q \cdot K^T_{s,i} / \sqrt{d}) \cdot V_{s,i}$$

which is the unnormalized attention-weighted sum over all positions. The denominator is:

$$\sum_s \exp(\text{LSE}_s) = \sum_s \sum_i \exp(Q \cdot K^T_{s,i} / \sqrt{d})$$

which is the normalization constant over all positions. The ratio is exactly the standard attention output.

**Implementation details:** The merge attention implementation is open-sourced in the xformers library. Each partial attention computation returns both the attention output tensor $O_s$ and the per-query log-sum-exp scalar $\text{LSE}_s$. In the pass-KV algorithm, the merge happens locally on each rank after the ring loop, because each rank has computed all $O^s_k$ for $s = 0, \ldots, N-1$ and holds them locally. In the pass-Q algorithm, the merge happens after the All2All, once each rank has received the partial outputs for its original queries from all other ranks.

---

#### The Analytical Model: When to Use Pass-KV vs. Pass-Q

The decision of which ring algorithm to use is not heuristic guesswork — the paper derives explicit analytical conditions (Section 3.4, Algorithms 1 and 5) that depend on model architecture constants, hardware specifications, and the runtime parameters $(T, P)$.

**The core question:** For a given $(T, P, N)$, will the communication in the ring loop be fully hidden under the attention computation, and if not, which algorithm exposes less total communication?

**Model parameters.** The analysis uses a simplified roof-line model with:
- $C$: peak compute (FLOPs/second) per rank (the GPU's effective throughput for attention kernels)
- $\text{BW}$: peak communication bandwidth (bytes/second) per rank for the ring SendRecv operations
- $N$: number of CP ranks
- $T$: length of new input tokens
- $P$: length of previously cached KV tokens
- $D$: model dimension (16384 for Llama3 405B)
- $N_H$: number of query heads (128)
- $N_{KV}$: number of key/value heads (8)
- $e$: bytes per element (1 for FP8, 2 for BF16)

**Attention computation FLOPs (from Table 3):** For partial prefill with $T$ new tokens attending to $(T+P)$ total KV entries:

$$\text{FLOPs} = 4 \cdot T \cdot D \cdot (T + P)$$

The factor 4 accounts for: 2 batch matrix multiplications (Q·K^T and the result·V), each requiring 2 FLOPs per multiply-add. For full prefill ($P=0$), this simplifies to $4 T^2 D$.

**Communication volume per ring step:**
- **pass-KV:** Each rank sends KV embeddings of size $2 \cdot (T+P) \cdot D \cdot (N_{KV}/N_H) \cdot e$ bytes on average per step. The factor 2 accounts for both keys and values. The $N_{KV}/N_H$ factor reflects GQA: each KV head is shared by $N_H/N_{KV}$ query heads, so the KV tensors are $N_{KV}/N_H$ times smaller than the query tensors would be.
- **pass-Q:** Each rank sends query embeddings of size $T \cdot D \cdot e$ bytes per step.

**Condition for pass-KV communication to be hidden (Equation 2).** For ring pass-KV communication to be fully overlapped with attention computation, the attention compute time per chunk must exceed the SendRecv time:

$$\frac{4 \cdot T \cdot D \cdot (T+P)}{N \cdot C} \geq \frac{2 \cdot (T+P) \cdot D \cdot e \cdot N_{KV}}{N_H \cdot \text{BW}}$$

**What this inequality states:** The left side is the time to compute attention for one chunk ($T/N$ queries against $(T+P)/N$ KV entries) on a single rank, assuming perfect load balancing. The right side is the time to send one KV chunk to the next rank. When this holds, the GPU can compute while data transfers, and communication adds no latency to the critical path.

Canceling common terms $(T+P) \cdot D$ from both sides yields:

$$T \geq \frac{N \cdot C \cdot N_{KV} \cdot e}{2 \cdot N_H \cdot \text{BW}}$$

The right hand side (RHS) is a **static threshold** — it depends only on the model architecture and hardware, not on $P$ or the total context length. Once $T$ exceeds this threshold, pass-KV communication will be hidden regardless of how much cached KV exists.

**Why this is independent of $P$:** Both the attention computation time and the KV communication time scale linearly with $(T+P)$ — more cached tokens mean more attention work and proportionally more KV data to send. The $(T+P)$ terms cancel, leaving only the dependence on $T$ (which determines the query-side computation). This is a crucial insight: it means that for any amount of conversation history, as long as the *new* prompt is long enough, pass-KV will perform well.

**Condition for pass-Q communication to be hidden (Equation 3).** For ring pass-Q:

$$\frac{4 \cdot T \cdot D \cdot (T+P)}{N \cdot C} \geq \frac{T \cdot D \cdot e}{\text{BW}}$$

Canceling $T \cdot D$ from both sides:

$$(T + P) \geq \frac{N \cdot e \cdot C}{4 \cdot \text{BW}}$$

Again, the RHS is a **static threshold**. For pass-Q, whether communication is hidden depends on the *total* context length $(T+P)$, not just $T$. This makes intuitive sense: with pass-Q, the query is small (proportional to $T$), so communication is cheap. The question is whether there's enough total computation (proportional to $T \cdot (T+P)$) to hide that small communication cost. Sufficiently large total context ensures overlap regardless of the $T/P$ ratio.

**The switching heuristic (Algorithm 1, simplified):**
```
if T >= (N * C * N_KV * e) / (2 * N_H * BW) OR T/(T+P) >= 2 * N_KV / N_H:
    use pass-KV
else:
    use pass-Q
```

**What this does operationally:**
1. If $T$ is large enough that pass-KV communication will be hidden (first condition from Equation 2), use pass-KV — it avoids the All2All overhead.
2. If the cache miss rate $T/(T+P)$ is above the model-specific threshold $2 \cdot N_{KV}/N_H = 12.5\%$ for Llama3 405B (second condition from Equation 1), use pass-KV — the KV tensors are smaller than the Q tensors on average, so communicating KV is volume-efficient even if some communication is exposed.
3. Otherwise, use pass-Q — the query tensors are smaller, the cache miss rate is low, and the total context is large enough to hide the ring communication (from Equation 3).

**Refined heuristic with All2All consideration (Appendix C, Algorithm 5).** The basic heuristic above does not account for the All2All communication required by pass-Q's merge attention. When pass-KV communication is partially exposed (i.e., $T$ is below the threshold from Equation 2), the system must compare the *exposed* pass-KV communication time against the All2All time:

$$\text{Exposed pass-KV time} = (N-1) \cdot \left( \frac{2(T+P)D \cdot e \cdot N_{KV}}{N_H \cdot \text{BW}} - \frac{4 \cdot T \cdot D \cdot (T+P)}{N \cdot C} \right)$$

$$\text{All2All time} = (N-1) \cdot \frac{(D+1) \cdot T \cdot e}{\text{BW}}$$

The condition for pass-Q to be preferred (Equation 5) becomes:

$$\frac{T}{T+P} \leq 2 \cdot \frac{N_{KV}}{N_H} - \frac{4T \cdot \text{BW}}{N \cdot C \cdot e}$$

This is a *stricter* condition than Equation (1) — the All2All cost reduces the cache miss rate threshold below which pass-Q wins. For the empirical data on Llama3 405B with 4 CP ranks, the tipping point occurs around a 5% KV cache miss rate ($T \approx 6400$ for a 128K total context, as shown in Figure 9 and Table 4).

**Practical implementation with empirical tuning (Appendix D).** The paper acknowledges that the analytical model uses theoretical peak $C$ and $\text{BW}$ values, while achieved values are lower in practice. The authors start with the peak-based thresholds and then fine-tune based on empirical measurements. They fit a simple log-linear classifier to empirical data points:

$$h(T, P) = \alpha \cdot \log(T) + \beta \cdot \log\left(\frac{T}{T+P}\right) + \gamma$$

with fitted parameters $\alpha = -1.059$, $\beta = 1.145$, $\gamma = 12.112$. When $h(T, P) > 0$, pass-KV is preferred; otherwise pass-Q. At the decision boundary ($h=0$), the two algorithms have virtually identical performance (within 1%), so either choice is acceptable. This empirical heuristic can be evaluated once at the start of each partial prefill with negligible overhead.

---

#### Batched Ring Pass-Q Decode: Generating Tokens Under Context Parallelism

The decode phase presents a different challenge from prefill. During decode (Section 3.6, Algorithm 4), the model generates one token at a time autoregressively. Each new token must attend to the entire cached KV history, which may span hundreds of thousands of tokens. The computation per decode step is relatively small — just one query token per sequence in the batch — and the communication volume must be minimized to avoid dominating the tiny per-step latency.

**Why pass-Q for decode:** With $T=1$ (one new token per sequence), Equation (1) gives $T/(T+P) = 1/(1+P) \approx 0$ for any non-trivial conversation history. This is far below the $2 \cdot N_{KV}/N_H = 12.5\%$ threshold. Passing the tiny query tensor around the ring is overwhelmingly more communication-efficient than passing the entire cached KV.

**The algorithm (Algorithm 4):** The structure follows ring pass-Q prefill (Algorithm 3) with two modifications for the batched decode setting:

1. **Round-robin sharding of decode tokens across ranks.** During multi-turn chat, the decode phase may generate hundreds of tokens per sequence. If all decode tokens for a given sequence were consistently assigned to the same CP rank, that rank would accumulate a disproportionately large KV cache (all the generated tokens) while other ranks would have empty decode-KV for that sequence. This would eventually cause the overloaded rank to run out of memory before other ranks reach capacity. To prevent this, the system **offsets by 1 index** for each decode iteration, sharding the batched decode tokens evenly across CP ranks using round-robin assignment. Each rank processes approximately $1/N$ of the decode tokens for each sequence over time, keeping the KV cache distribution balanced.

2. **Per-sequence batch ID tracking.** Since different sequences in the batch may be at different points in their generation (some may have finished while others continue), the decode step must track which query tokens belong to which sequence. The ring loop passes both the query tensor $Q^s_k$ and the batch IDs $\text{bid}^s_k$ around the ring. When computing attention on rank $k$, the received queries attend only to the KV cache entries for their respective sequences: $\text{GQA}(Q^s_k, \text{KV}_k[\text{bid}^s_k])$. This requires the KV cache to be organized such that per-sequence lookups are efficient.

**Performance considerations.** The decode phase uses CUDA Graphs (Nvidia Blog, 2019) to avoid host-side kernel launch overhead for the many small GPU kernels involved. Even with this optimization, the paper reports (Table 6) that CP2 decode has higher time-to-incremental-token (TTIT) than TP8 — 65.6 ms vs. 44.5 ms at 32K context — because the communication overhead (ring SendRecv + All2All) cannot be fully hidden under the tiny per-step computation. However, this decode latency regression is acceptable in the paper's framework because the primary goal is reducing prefill latency, and the paper explicitly notes (Section 5) that CP is "best suited for improving prefill performance and can be best leveraged with a serving system that decouples the parallelization scheme for prefill and decode" (referring to disaggregated prefill-decode architectures from Qin et al., 2024 and Zhong et al., 2024).

**Why the TTIT increases with more CP ranks (Table 8):** Two factors contribute:
1. **Padding overhead:** The current implementation pads the number of queries to be divisible by the number of ranks. For batch size 1, this means with CP4, the system processes 4 query tokens even though only 1 is real — a 4× increase in effective query computation.
2. **Communication growth:** The SendRecv time per ring iteration and the All2All time both grow with the number of ranks, and these communications are largely exposed (not hidden under computation) due to the tiny attention compute per step during decode.

---

#### The CP+TP Nesting Design: Why This Particular Hierarchy?

The paper's architectural choice — tensor parallelism within nodes (TP8) and context parallelism across nodes (CP $N$) — is not arbitrary. It reflects a careful analysis of the memory-bandwidth-compute tradeoffs in modern GPU clusters (Section 3.2, Section 4.1, Figure 5).

**Why TP within nodes:** The Llama3 405B model with FP8 row-wise quantization of feed-forward layers requires approximately 405 GB of parameter storage in FP8 format, plus KV cache, activation memory, and optimizer states (for any online adaptation). A single H100 GPU has 96 GB of HBM2e, so the model cannot fit on one GPU. TP8 shards the model weights across 8 GPUs within a node, with each GPU holding 1/8 of every linear layer's weight matrix. The 8 GPUs within a node communicate over NVLink (bidirectional bandwidth approximately 900 GB/s per GPU on H100 systems), making the AllReduce operations for TP fast enough to not be the primary bottleneck.

**Why CP across nodes:** When scaling beyond one node, the inter-host bandwidth drops dramatically — from ~900 GB/s (NVLink) to ~50 GB/s (RDMA, 400 Gb/s) or ~12.5 GB/s (TCP/IP, 100 Gb/s) per GPU. Table 2 shows why CP is better suited to this lower-bandwidth regime:

- **TP communication** per transformer block: $2 \cdot (T \cdot N_H \cdot D_H)$ bytes for AllReduce on two linear layers.
- **CP communication** per transformer block: $T \cdot N_{KV} \cdot D_H$ bytes for the ring SendRecv on the attention layer.

For Llama3 405B ($N_H = 128$, $N_{KV} = 8$, $D_H = D/N_H = 16384/128 = 128$):
- TP: $2 \cdot T \cdot 128 \cdot 128 = 32768 \cdot T$ bytes per block
- CP: $T \cdot 8 \cdot 128 = 1024 \cdot T$ bytes per block

CP communicates **32× less data** than TP on attention layers for this model. Since each transformer block has 4 linear layers and 1 attention layer, the total communication ratio across the whole block is smaller but still heavily favors CP.

**The per-KV-head communication group structure (Figure 5).** The most distinctive aspect of the design is how the communication groups are formed. Llama3 405B has 8 KV heads. Within each node, TP8 partitions the 128 query heads across 8 GPUs (16 per GPU) and the 8 KV heads across 8 GPUs (1 per GPU). When scaling to $N$ nodes with CP:

- **Within each node:** TP8 runs as usual. GPU $i$ (for $i = 0, \ldots, 7$) in every node holds KV head $i$ and query heads $16i$ through $16i+15$.
- **Across nodes:** A communication group is formed for each KV head $i$, consisting of GPU $i$ from every node. So there are 8 independent ring groups, each with $N$ GPUs. When the ring attention algorithm runs, these 8 rings operate in parallel, each communicating the embeddings for one KV head.

**Why this grouping:** This design means each communication group handles only one KV head's worth of data. The message size per SendRecv is proportional to $T \cdot D_H$, not $T \cdot N_{KV} \cdot D_H$. With 8 parallel rings, the total communication volume is the same as if one ring handled all 8 heads, but the parallelism of 8 concurrent transfers better utilizes the network links and allows each ring's computation to proceed independently. This is especially important when overlapping communication with computation — each ring's attention computation can overlap with that ring's SendRecv independently, reducing synchronization points.

**FP8 quantization as an enabler:** The paper uses row-wise FP8 quantization for the feed-forward layers (Section 4.1), which approximately halves the model's memory footprint compared to BF16. This is what makes it possible to fit Llama3 405B into a single node with TP8. Without quantization, the model would require TP16 (two nodes' worth of memory) just to hold the weights, and the CP+TP nesting advantage would be lost because TP would already be crossing node boundaries, incurring the inter-host bandwidth penalty on every linear layer.

**Communication implementation details:** The ring communication uses an 8-way SendRecv primitive — each of the 8 GPUs in a node sends to its counterpart in the next node and receives from its counterpart in the previous node simultaneously. The ring topology is logical (rank $k$ communicates with rank $(k+1) \bmod N$ and $(k-1) \bmod N$); the physical network topology may differ, but the paper does not discuss network topology optimization.

## 4. Key Insights and Innovations

### Innovation 1: Context Parallelism as a *Communication Regime Optimizer*, Not Just a Parallelism Strategy

The paper's most intellectually distinctive contribution is reframing context parallelism not as merely another way to split work across GPUs, but as a mechanism for **adaptively selecting which communication regime the system operates in** — and showing that this choice fundamentally determines scalability in a way that prior parallelism taxonomies missed.

**What the field assumed before this work.** The standard parallelism taxonomy — tensor parallelism (TP) shards weights, pipeline parallelism (PP) shards layers, data parallelism (DP) shards batches, context parallelism (CP) shards sequences — treats these as static architectural choices made at deployment time. You pick a combination based on model size and hardware topology, and that's your system. The implicit assumption is that within a chosen parallelism strategy, the communication pattern is fixed: TP always does AllReduce on activations, CP always communicates KV embeddings.

**What this paper shows is different.** The paper demonstrates that CP is not one communication pattern but a *family* of patterns — pass-KV and pass-Q — that expose radically different communication-vs-computation tradeoffs depending on the *runtime state* of the system. The key insight is not "CP is better than TP for multi-node inference" (that's Table 2 and Figure 7, an important but relatively straightforward bandwidth arithmetic result). The deeper insight is that even *within CP*, the choice of what to communicate around the ring — keys/values versus queries — changes whether the system is compute-bound or communication-bound for a given request, and that this boundary is predictable from a small set of static model and hardware parameters (Equations 2 and 3).

**The conceptual advance.** Prior ring attention work (Liu et al., 2023; Brandon et al., 2023; Li et al., 2023) treated the communication pattern as fixed — always pass KV, because training always has $P=0$ (no cached context) and $T$ large enough to hide communication. The paper's diagnostic move is to recognize that inference *inherently* spans a wide range of $(T, P)$ regimes, from full prefill ($P=0$, $T$ large) to high-hit-rate partial prefill ($P$ enormous, $T$ tiny) to decode ($T=1$), and that a single communication pattern cannot be optimal across this entire space. The pass-Q variant and the adaptive switching heuristic (Algorithm 1 and its refinements) are the *embodiment* of this insight, not the insight itself.

**Why this matters beyond the immediate system.** This reframing suggests a broader principle: parallelism strategies should be *state-dependent*, not statically configured. The decision of what to communicate and when should respond to the runtime characteristics of the workload (sequence lengths, cache hit rates, batch composition), not just the hardware topology. This is a departure from how distributed inference systems are typically built — with fixed parallelization configs chosen at model-loading time — and it opens the door to more dynamic, request-aware resource allocation in serving systems. The paper's heuristic is simple (a threshold check on $T$ and $T/(T+P)$), but the *idea* that such runtime adaptation is both necessary and analytically tractable is the contribution.

**Evidence anchoring the claim.** The necessity of adaptation is demonstrated empirically in Figure 9 and Table 4: at a 1% KV cache miss rate, pass-Q outperforms pass-KV by roughly 14% (898 ms vs. 1023 ms for a 128K total context on 4 CP ranks), while at a 10% cache miss rate, pass-KV outperforms pass-Q by about 6% (2081 ms vs. 2205 ms). The crossover point near 5% means that a static choice would leave 6–14% performance on the table depending on the workload mix — significant for latency-critical serving. The analytical model (Equations 2 and 3) explains *why* this crossover occurs and provides a static threshold that can be computed once per model-hardware pair, making the adaptation practically deployable without per-request profiling.

---

### Innovation 2: The Prefill-Decode Asymmetry as a First-Class Design Constraint for Context Parallelism

The paper elevates what might seem like an implementation detail — the difference between prefill and decode phases — into a **fundamental architectural constraint** that determines whether context parallelism is beneficial at all, and if so, how it should be configured.

**What prior work missed.** Previous sequence parallelism work for training (Liu et al., 2023; Jacobs et al., 2023) had no concept of prefill-decode asymmetry because training processes all tokens in a sequence symmetrically — every token attends to every preceding token in one forward pass. The distinction between "processing many tokens at once" (prefill) and "processing one token attending to a large history" (decode) simply doesn't exist in training. Consequently, ring attention for training was optimized for a single regime: large $T$, no persistent state, uniform sequence lengths.

**The paper's diagnostic contribution.** The paper identifies that context parallelism has *opposite* scaling behavior in prefill versus decode, and that this asymmetry is not a minor inefficiency but a structural property of how the computation-to-communication ratio changes with $T$:

- **Prefill**: As context length grows, computation grows quadratically ($O(T^2)$), while communication grows linearly ($O(T)$). The computation-to-communication ratio *improves* with scale, making CP increasingly efficient. This is why Figure 6 shows near-linear scaling for prefill — communication is hidden under computation for sufficiently large $T$ (Equation 2).

- **Decode**: Each step has fixed computation ($O(1)$ for the new token's linear projections, $O(P)$ for attention against cached KV) and fixed communication. The computation-to-communication ratio does *not* improve with context length — it plateaus or degrades. This is why Table 8 shows that CP4 decode (71.3 ms TTIT) is slower than TP8 decode (46.3 ms TTIT) for the same 128K context — the communication overhead cannot be amortized over enough compute.

**Why this is a conceptual contribution, not just an observation.** This asymmetry implies that the *optimal parallelism strategy differs between prefill and decode*, and that a system using CP for prefill should not necessarily use CP for decode. The paper makes this explicit in its conclusion (Section 5): CP is "best suited for improving prefill performance and can be best leveraged with a serving system that decouples the parallelization scheme for prefill and decode." This is not an engineering workaround — it's a statement about the inherent structure of the problem. The paper provides the analytical basis (the computation-to-communication ratio analysis) for *why* this decoupling is necessary, not just that it's empirically beneficial.

**Connection to the disaggregated serving trend.** The paper explicitly references Qin et al. (2024) and Zhong et al. (2024) on disaggregated prefill-decode architectures. The contribution here is not proposing disaggregation (which predates this work) but providing the *systems-level evidence* that CP makes the case for disaggregation stronger: because CP is highly effective for prefill but degrades decode, the benefits of separating the two phases are amplified when CP is in the picture. A disaggregated system could use CP-heavy configurations for prefill nodes and TP-only configurations for decode nodes, achieving the best of both worlds.

**Evidence.** Table 6 shows the concrete tradeoff: for a 128K context, CP2 reduces TTFT from 42.0 seconds (TP8) to 21.0 seconds (a 2× improvement, near-perfect scaling) but increases TTIT from 46.3 ms to 66.6 ms (a 44% regression). Table 7 shows that both CP and TP scale poorly for decode — CP4+TP8 has worse TTIT (71.3 ms) than CP2+TP8 (60.2 ms) and substantially worse than TP8 alone (46.3 ms). The conclusion that CP is a prefill optimization, not a universal inference optimization, is directly supported by this data.

---

### Innovation 3: KV Cache Miss Rate as the Unifying Sufficient Statistic for Communication Strategy Selection

The paper introduces the **KV cache miss rate** — defined as $T/(T+P)$, the fraction of total context tokens that are new — as the single variable that determines which ring attention algorithm is optimal. This is a conceptual simplification of surprising power: it reduces what could be a complex multi-factor optimization (new token count, cached token count, model architecture, hardware bandwidth, number of ranks) into a decision boundary on a single dimensionless ratio.

**Why this is non-obvious.** At first glance, the decision of whether to pass KV or Q around the ring seems to depend on many factors: the absolute number of new tokens $T$, the absolute number of cached tokens $P$, the model's GQA ratio $N_H/N_{KV}$, the number of CP ranks $N$, the compute-to-bandwidth ratio $C/\text{BW}$, and the element size $e$. The paper's analytical model (Section 3.4) initially derives two separate conditions: one on absolute $T$ (Equation 2) and one on the ratio $T/(T+P)$ relative to $2 N_{KV}/N_H$ (Equation 1). The empirical finding (Figure 9, Table 4) is that the ratio condition dominates in practice for partial prefill — the absolute $T$ condition matters primarily at the extremes (very small $T$ or very large $T$), while the cache miss rate cleanly separates the regimes where each algorithm wins.

**The simplifying insight.** The $T/(T+P)$ ratio captures the essential tradeoff: it determines the relative communication volumes of pass-KV (proportional to $T+P$) versus pass-Q (proportional to $T$), and it determines whether the attention computation (which scales with $T \cdot (T+P)$) is large enough to hide the ring communication. The fact that this single parameter, which can be computed trivially at the start of each partial prefill from values the system already tracks (new prompt length and current KV cache occupancy), is sufficient to make the optimal choice with less than 1% error near the decision boundary (Appendix D) is a genuinely useful simplification for system builders.

**Comparison to alternative approaches.** Without this unification, a system designer might attempt to profile every $(T, P)$ combination empirically and build a lookup table, which would be expensive, brittle to hardware changes, and difficult to generalize across models. The analytical model provides a portable formula: $T/(T+P) \lessapprox 2 N_{KV}/N_H$ implies pass-Q, otherwise pass-KV (with the All2All refinement in Equation 5). For Llama3 405B, the threshold is 12.5% (uncorrected) or empirically around 5% (with All2All considered). A different model with different GQA ratio — say, a model with 32 query heads and 8 KV heads, giving $2 N_{KV}/N_H = 0.5$ — would have a dramatically different threshold (50%), and the formula would immediately predict that pass-Q is preferred for a much wider range of cache hit rates. This portability across model architectures is the hallmark of a good abstraction.

**Evidence.** Figure 9 plots the pass-KV/pass-Q performance ratio against the KV cache miss rate on a log scale, showing a clean monotonic relationship: as miss rate decreases (moving left), pass-Q becomes increasingly favorable. Table 4 provides the raw data confirming that the crossover occurs between 3.25% and 10% miss rate. Table 5 provides the microarchitectural explanation — at 2.5% miss rate, the exposed pass-KV communication time ($(N-1) \cdot (627 - 414) = 639 \mu s$) exceeds the pass-Q All2All time ($424 \mu s$), which is exactly what the refined analytical model in Appendix C (Equation 5) predicts.

---

### Innovation 4: Load-Balanced Sharding for Causal Attention as the Enabling Precondition for Inference CP

While the two-chunk sharding scheme itself is adapted from prior work (Cho et al., 2024; Brandon et al., 2023), the paper's contribution is demonstrating that it is the **critical enabler** for context parallelism in the inference setting — and that without it, CP would be practically unusable for variable-length sequences and multi-turn conversations.

**Why this matters for inference specifically.** In training, sequences within a batch are typically padded to a uniform length, so naive contiguous sharding works (with some inefficiency from padding tokens). In inference, batching variable-length requests is essential for throughput, and multi-turn conversations create unevenly distributed KV caches that naive sharding would exacerbate. Without load-balanced sharding, the CP rank holding the end of a long sequence would:
1. Have more attention computation (more preceding tokens to attend to).
2. Store more KV cache entries (tokens from later in the sequence are cached on the rank that processed them).
3. Become the straggler that determines the ring's iteration time (since all ranks must synchronize at each SendRecv).
4. Run out of memory before other ranks, reducing effective KV cache capacity.

The paper identifies load-balanced sharding not as an optional optimization but as a **correctness and capacity requirement** for CP inference: "To support maximum context length provided by the pretrained model without OOM on any particular CP rank with heavier load, we aim for load-balancing for both attention compute and KV cache capacity" (Section 3.5.1). This reframes sharding from a performance concern to a deployment feasibility concern — without it, the maximum supported context length would be determined by the most-loaded rank, not the aggregate memory across all ranks.

**The subtlety the paper handles.** The load-balanced sharding must work differently for full prefill (where all tokens are new and the sharding can be symmetric), partial prefill (where new tokens are sharded with the two-chunk scheme but cached KV distribution is fixed from previous turns), and decode (where tokens are sharded round-robin across ranks to maintain long-term balance). The paper's treatment of these three cases — and the recognition that they require different sharding strategies — is a practical contribution that prior training-focused work did not need to address.

**Evidence.** The paper does not provide an ablation comparing load-balanced vs. naive sharding (likely because the failure mode — OOM or severe imbalance — makes it obviously necessary to anyone who has tried naive CP for inference). The claim is supported architecturally: Figures 1 and 2 show how the sharding handles fused variable-length inputs, and Algorithm 4's round-robin decode sharding is explicitly designed to prevent the KV cache imbalance that would otherwise accumulate over multiple turns. The 1M-token prefill result (Figure 8) on 16 nodes — which requires every rank to stay within its 96 GB HBM budget — implicitly validates that the sharding keeps memory usage balanced enough to avoid OOM.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The paper uses synthetic sequences for the main performance benchmarks — not a standard NLP dataset. All experiments measure system throughput and latency for prompts of specified token lengths (2K, 32K, 128K, 256K, 512K, 1M). The evaluation focuses on the Llama3 405B model's native maximum context window of 128K tokens, plus an extended 1M configuration run on 8 and 16 nodes to demonstrate scalability beyond the pretrained limit. There is no standard train/test split — the benchmark measures wall-clock latency for processing prompts of varying lengths with batch size 1 (prefill scaling) and batch sizes up to 4 (decode analysis).

- **Base model(s).** All experiments use **Llama3 405B** (Llama Team, 2024), a dense transformer model with 126 layers, model dimension D = 16384, feed-forward network dimension 53248, 128 query heads, and 8 key/value heads (Grouped Query Attention with a 16:1 query-to-KV head ratio). The feed-forward layers use row-wise FP8 quantization (via the FBGEMM library) to reduce memory footprint, enabling the full model to fit within a single 8-GPU node with TP8. Model configurations are detailed in Table 9. The authors argue this model is representative of contemporary large-scale LLMs and note that its 128:8 Q:KV head ratio is crucial for demonstrating CP's communication advantage — communicating KV embeddings is 16× smaller than communicating Q embeddings, making pass-KV efficient for full prefill.

- **Metrics.** The primary metrics are:
  - **TTFT (Time-to-First-Token):** wall-clock latency for processing the entire input prompt through the model and producing the first output token, measured in seconds or milliseconds. This is the critical user-facing metric for interactive applications.
  - **TTIT (Time-to-Incremental-Token):** wall-clock latency per output token during autoregressive decoding, measured in milliseconds. Reported for single-batch decode to isolate the per-step cost.
  - **Scaling ratio:** $\tau_1 / \tau_N$, where $\tau_N$ is the latency with $N$ nodes. Perfect linear scaling would give a ratio of $N$; the achieved ratio divided by $N$ (implicitly) gives parallelization efficiency.
  - **FLOPS utilization:** achieved FLOPs per GPU divided by the theoretical peak for the hardware configuration, computed for the 1M context length run on 16 nodes (Appendix A). The achieved 502 TF/sec per H100 is compared against a standalone Flash Attention v3 benchmark achieving 540 TF/sec for the equivalent per-GPU shard size (8K context on a single GPU), giving a 93% parallelization efficiency (502/540). Against the theoretical peak of H100 (800 TF/sec for the 500W power-limited configuration used), the utilization is approximately 63%.

- **Baselines.** The paper compares context parallelism against two primary alternatives:
  - **Single-node TP8:** tensor parallelism across 8 GPUs within one node, serving as the baseline for TTFT and TTIT measurements (Tables 6 and 7). This represents the best single-node configuration.
  - **Multi-node tensor parallelism (TP16, TP32):** tensor parallelism scaled across 2 or 4 nodes (16 or 32 GPUs total), using the replication strategy described in Section 4.2.2 where each KV head is replicated across $N_{TP} / N_{KV}$ GPUs when $N_{TP} > N_{KV}$.

  The paper does not compare against pipeline parallelism (since PP does not improve latency for single requests), data parallelism (which would require multiple requests to derive benefit), or other sequence parallelism variants (DeepSpeed Ulysses, DistFlashAttn, Striped Attention) — the comparison is strictly between CP and TP as mechanisms for latency reduction in long-context prefill.

- **Generation budget / compute accounting.** The paper measures computation in terms of **hardware configuration** (number of nodes, number of GPUs) and **wall-clock time**, not in FLOPs or token-generations. This is appropriate for a systems paper focused on latency: the goal is to answer "how fast can a given context length be processed on a given hardware budget?" rather than "how much total work is done?" The FLOPS utilization calculation in Appendix A provides a post-hoc validation that the achieved per-GPU throughput is close to the hardware's reasonable peak for attention workloads. For fairness in the CP vs. TP comparison (Figure 7), both use identical node counts and GPU counts — the only difference is the parallelism strategy.

- **Cross-validation / statistical protocol.** There is no cross-validation or statistical significance testing, as this is a systems benchmarking paper rather than an ML evaluation. Latency measurements are reported as single values (presumably averages over multiple runs, though the paper does not specify the number of trials or variance). The analytical model parameters in Appendix D are fit to empirical data points, and the authors note that misclassified points near the decision boundary have performance differences under 1%, making the choice between pass-KV and pass-Q essentially indifferent in that regime.

---

### Main Quantitative Results

#### Context Parallel Prefill Scaling: Latency Reduction with Fixed Context Length

**Headline result:** Context parallelism achieves near-linear scaling for long-context prefill latency on both high-bandwidth RDMA (GTT) and medium-bandwidth TCP/IP (GTI) networks, with the scaling holding for context lengths where the attention computation is large enough to hide communication under the ring SendRecv operations.

**Figure 6(a) — GTT (400 Gb/s RDMA per GPU) scaling:** On Grand Teton Training systems, pass-KV full prefill with Llama3 405B shows proportional latency reduction as CP ranks are added, for sufficiently long contexts:
- 128K context: CP1 = ~42s (from Table 6), CP2 = ~21s, CP4 = ~10.5s, CP8 = 5.85s (from Section 4.2.1 explicitly). The 128K→CP8 latency of 5.85 seconds represents an 8/5.85 ≈ 7.2× speedup over single-node TP8 for 8× the hardware — approximately 90% parallelization efficiency.
- The curves for different CP counts in Figure 6(a) fan out proportionally for context lengths ≥ 16K tokens. At very short contexts (2K–4K), the curves begin to converge because the absolute attention computation is too small to hide communication (Equation 2 not satisfied).
- The paper explicitly states: "latency for the same input length is halved as we double the number of CP nodes" — this is the near-linear scaling claim.

**Figure 6(b) — GTI (100 Gb/s TCP/IP per GPU) scaling:** On Grand Teton Inference systems with 4× lower inter-host bandwidth than GTT, CP still achieves near-linear scaling up to 4 nodes:
- The achieved inter-host bandwidth is roughly 3 GB/s per rank (Section 4.2.1), which is still sufficient to overlap pass-KV communication with attention computation for long contexts.
- The paper emphasizes this as a robustness result: "demonstrating the robustness of pass-KV algorithm even with low inter-connect bandwidth."
- GTI results are shown only up to CP4 (not CP8), implying that at some number of nodes, the reduced bandwidth would begin to expose communication even for long contexts. The paper does not identify this threshold explicitly.

**What "near-linear" means concretely:** For 128K context on GTT, single-node TP8 latency (42010 ms from Table 6) divided by 8 gives an ideal 8-node latency of 5251 ms. The achieved CP8 latency of 5.85 seconds (5850 ms) gives 5251/5850 ≈ 89.8% parallelization efficiency. The paper's claimed 93% parallelization efficiency for the 1M context run (Appendix A) is computed differently — against a single-GPU Flash Attention benchmark rather than against single-node TP8 end-to-end latency — so the prefill scaling efficiency should be understood as approximately 90% for 128K on 8 nodes.

**Table 6 — TTFT comparison TP8 vs. CP2:** Across three context lengths at batch size 1:
- 8K: TP8 = 1740 ms, CP2+TP8 = 999 ms (1.74× speedup for 2× hardware)
- 32K: TP8 = 7658 ms, CP2+TP8 = 4015 ms (1.91× speedup)
- 128K: TP8 = 42010 ms, CP2+TP8 = 21042 ms (2.00× speedup)

The scaling improves with context length because the larger attention computation better hides the communication overhead. At 128K, CP2 achieves essentially perfect 2× scaling. At 8K, the 1.74× speedup reflects partially exposed communication.

#### Multi-Node Tensor Parallelism vs. Context Parallelism

**Headline result:** Context parallelism dramatically outperforms tensor parallelism when scaling across nodes, with the gap widening as node count increases. At 8 nodes, CP8 achieves approximately 2× the throughput of TP64 for 128K context prefill.

**Figure 7 — Scaling ratio comparison:** The figure plots scaling ratio $(\tau_1/\tau_N)$ against number of nodes $N$ for both TP and CP:
- Perfect scaling would give a 45-degree line (scaling ratio = N).
- CP with pass-KV tracks the perfect scaling line closely through 8 nodes, achieving a scaling ratio of approximately 7.3 at N=8 (from the figure, estimated — the paper quotes "100%" difference between CP8 and TP64 at 8 nodes).
- TP scaling degrades substantially: at N=2, TP16 scaling ratio is roughly 1.4 (15% worse than CP2). At N=8, TP64 scaling ratio appears to be approximately 3.6 — meaning adding 8× the hardware gives only 3.6× the throughput, compared to CP's 7.2×.
- The paper states: "While the latency is different by roughly 15% between CP2 and TP16 on 2 nodes, the difference drastically increases to 100% when scaled to 8 nodes."

**Why TP degrades:** The explanation (Section 4.2.2) is that TP's AllReduce communication on linear layers scales with the number of participating GPUs in the collective. As nodes are added, the AllReduce latency — which must complete before the next transformer layer can begin — grows and cannot be hidden because linear layer computation is relatively fast compared to the communication volume ($2 \cdot T \cdot N_H \cdot D_H$ bytes per block, per Table 2). CP's ring SendRecv, in contrast, communicates only $T \cdot N_{KV} \cdot D_H$ bytes per block (32× less for Llama3 405B) and can be overlapped with the attention computation.

**Caveat:** The paper notes that for future systems like GB200 with NVLink connecting multiple hosts (inter-host bandwidth approaching intra-host bandwidth), "tensor parallelism can still benefit with reasonable scalability." The CP advantage is contingent on the bandwidth hierarchy of current H100 clusters where inter-node bandwidth is 18–72× lower than intra-node NVLink bandwidth.

#### Scaling Context Length with Fixed Capacity: 1M Token Prefill

**Headline result:** Context parallelism enables processing 1M-token contexts that would be infeasible on a single node, achieving 77 seconds TTFT on 16 nodes (128 H100 GPUs) with 93% parallelization efficiency relative to per-GPU attention benchmarks.

**Figure 8 — TTFT for 128K–1M contexts on CP8 and CP16:**
- 128K: CP8 = ~6s, CP16 = 3.8s (from the abstract and Section 4.2.3 explicitly — 3.8s for 128K on 16 nodes)
- 256K: CP8 = ~14s, CP16 = ~7.5s
- 512K: CP8 = ~32s, CP16 = ~17s
- 1M: CP8 = ~155s, CP16 = 77s

The paper explicitly states: "With a 16-node setup, we achieve an exact prefill in 77 seconds for a 1M context length and 3.8 seconds for a 128K context length."

**Quadratic scaling observation:** The paper notes that at context lengths ≥ 512K, doubling the context length more than doubles the TTFT: 512K→1M is an 8× increase in tokens (from context length squared perspective) but the latency increases from ~17s to 77s on CP16 — roughly 4.5×, not 2×. This is because "the quadratic increase in attention latency with context length begins to dominate the overall TTFT latency." At these extreme lengths, attention FLOPs ($4.1 × 10^{18}$ for 1M) dominate GEMM FLOPs ($8.1 × 10^{17}$) by approximately 5× (Appendix A), and the quadratic $O(T^2)$ attention cost drives the super-linear latency growth with context length even under perfect parallelization.

**FLOPS utilization (Appendix A):** The achieved 502 TF/sec per H100 for the 1M context run is compared against:
- Flash Attention v3 standalone benchmark at equivalent per-GPU shard size (8K context on single GPU): 540 TF/sec → 502/540 = 93% parallelization efficiency. This measures how much of the achievable attention throughput is lost to communication and coordination overhead.
- H100 theoretical peak (800 TF/sec BF16 for the 500W power-limited configuration used): 502/800 = 63% FLOPS utilization. The gap to peak is attributed primarily to the H100's power-limited configuration (500W vs. 700W for full H100 SXM), which reduces both compute throughput and memory bandwidth (2.4 TB/sec HBM2e vs. 3.35 TB/sec HBM3).

**What "1M context" means practically:** The paper notes this corresponds to "approximately 1 hour of video content," giving an intuitive sense of the scale. The 77-second prefill means the model can "watch" an hour of video and begin responding in just over a minute — a latency that, while not real-time, makes certain long-context applications feasible that would be completely impractical with single-node inference (estimated 1200 seconds for 1M on a single node).

#### Pass-KV vs. Pass-Q for Partial (Persistent KV) Prefill

**Headline result:** The choice between pass-KV and pass-Q depends critically on the KV cache miss rate $T/(T+P)$, with a crossover point at approximately 5% miss rate for Llama3 405B on 4 CP ranks. Below 5%, pass-Q achieves lower latency; above 10%, pass-KV is clearly superior; between 5–10%, the difference is small.

**Table 4 — TTFT for varying persistent KV cache miss rates on CP4:** For a total context length of 128K tokens, varying the split between new tokens $T$ and cached tokens $P$:
- 1% miss rate ($T=1280$, $P=126720$): pass-KV = 1023 ms, pass-Q = 899 ms → pass-Q wins by 12.4%
- 2.5% miss rate ($T=3200$, $P=124800$): pass-KV = 1110 ms, pass-Q = 1046 ms → pass-Q wins by 5.8%
- 3.25% miss rate ($T=4160$, $P=123840$): pass-KV = 1299 ms, pass-Q = 1280 ms → essentially tied (1.5% difference)
- 5% miss rate ($T=6400$, $P=121600$): pass-KV = 1306 ms, pass-Q = 1302 ms → near-identical (0.3% difference)
- 10% miss rate ($T=12800$, $P=115200$): pass-KV = 2081 ms, pass-Q = 2205 ms → pass-KV wins by 5.6%
- 20% miss rate ($T=25600$): pass-KV = 3353 ms, pass-Q = 3617 ms → pass-KV wins by 7.3%
- 50% miss rate ($T=64000$): pass-KV = 6845 ms, pass-Q = 7368 ms → pass-KV wins by 7.1%
- 100% miss rate ($T=128000$, $P=0$, full prefill): pass-KV = 11462 ms, pass-Q = 12361 ms → pass-KV wins by 7.3%

**Figure 9 — Speed ratio across miss rates:** The pass-KV/pass-Q speed ratio (values < 1 indicate pass-KV is faster; values > 1 indicate pass-Q is faster) shows:
- At 1% miss rate: ratio ≈ 1.14 (pass-Q 14% faster)
- At 5% miss rate: ratio ≈ 1.00 (tie)
- At 10–100% miss rate: ratio ≈ 0.93–0.95 (pass-KV 5–7% faster)

The transition is sharp between 1–5% and then flat from 10–100%, meaning that once pass-KV wins, its advantage plateaus at about 5–7% regardless of how many new tokens are added beyond the threshold. This is consistent with the analytical model: the key determinant is whether pass-KV communication is hidden (which depends on $T$ exceeding a threshold from Equation 2) and whether the smaller communication volume of pass-KV offsets its per-message size advantage over pass-Q.

**Table 5 — Microarchitectural time breakdown at 2.5% and 10% miss rate:** This table reveals *why* the crossover occurs:
- **At 2.5% miss rate ($T=3200$):**
  - pass-KV: SendRecv = 627 µs per iteration, ATTN compute = 414 µs. The exposed communication per iteration is 627 − 414 = 213 µs. Over 3 iterations (N=4): 639 µs total exposed.
  - pass-Q: SendRecv = 166 µs, ATTN = 414 µs (fully hidden — SendRecv < ATTN), All2All = 424 µs. Total exposed: 424 µs (the All2All).
  - Since 639 µs > 424 µs, pass-Q is faster.
- **At 10% miss rate ($T=12800$):**
  - pass-KV: SendRecv = 631 µs, ATTN = 1608 µs. Communication is fully hidden (SendRecv < ATTN). Total exposed: 0 µs.
  - pass-Q: SendRecv = 544 µs, ATTN = 1608 µs (fully hidden), All2All = 1023 µs. Total exposed: 1023 µs.
  - Since 0 µs < 1023 µs, pass-KV is faster.

This is the key mechanistic insight: pass-KV's exposed communication grows with the gap between SendRecv and ATTN times, while pass-Q's exposed communication is always the fixed All2All cost. The crossover occurs when the sum of exposed pass-KV communication across ring iterations exceeds the pass-Q All2All time — exactly what the refined analytical model in Appendix C (Equation 5) predicts.

**Validation of the analytical model (Algorithm 1 conditions):**
- **Condition 1 (Equation 2):** $T \geq N \cdot C \cdot N_{KV} \cdot e / (2 \cdot N_H \cdot \text{BW})$ would select pass-KV when $T$ exceeds a static threshold. At $T=12800$ (10% miss), SendRecv is hidden under ATTN (Table 5), satisfying this condition. At $T=3200$ (2.5% miss), SendRecv > ATTN, violating the condition.
- **Condition 2 (Equation 1):** $T/(T+P) \geq 2 \cdot N_{KV}/N_H = 12.5\%$ would select pass-KV when the cache miss rate exceeds 12.5%. In Table 4, at 10% miss rate (below the threshold), pass-KV already outperforms pass-Q by 5.6%, suggesting the empirical crossover is lower than the simple model predicts — which is why the All2All-refined model (Equation 5) and the empirical heuristic (Appendix D) adjust the threshold downward to approximately 5%.

#### Decode Performance Under Context Parallelism

**Headline result:** Context parallelism degrades decode (TTIT) performance relative to single-node TP8, with the regression worsening as more CP ranks are added. CP is therefore best suited for prefill optimization, with decode handled either by accepting the regression or by disaggregating prefill and decode onto separate hardware.

**Table 6 — TTIT comparison TP8 vs. CP2 across context lengths (batch size 1):**
- 8K: TP8 = 44.5 ms, CP2+TP8 = 65.6 ms → CP2 is 47% slower
- 32K: TP8 = 44.6 ms, CP2+TP8 = 65.7 ms → CP2 is 47% slower
- 128K: TP8 = 46.3 ms, CP2+TP8 = 66.6 ms → CP2 is 44% slower

The TTIT for both TP8 and CP2 is nearly flat across context lengths (only rising from 44.5 to 46.3 ms for TP8 from 8K to 128K). This is because the linear layer computation is constant per-token, and the attention computation — while scaling with context length — is a small fraction of total per-token work during decode (unlike prefill where attention dominates). The CP regression of ~20 ms is attributable to the ring communication overhead (SendRecv + All2All) that cannot be hidden under the tiny per-token attention computation.

**Table 7 — Parallelism scalability for decode at 128K context, batch size 1:**
- CP1+TP8 (single node): TTFT = 42010 ms, TTIT = 46.3 ms
- CP2+TP8: TTIT = 60.2 ms (30% slower than single node)
- CP4+TP8: TTIT = 71.3 ms (54% slower than single node)
- TP16 (2 nodes): TTIT = 39.5 ms (15% faster than single-node TP8 — TP scales better for decode)
- TP32 (4 nodes): TTIT = 47.3 ms (2% slower than single node — TP begins to degrade)

The key pattern: TP16 actually *improves* decode TTIT over TP8 (39.5 vs. 46.3 ms) because splitting the linear layers' computation across 16 GPUs reduces per-GPU work, and the AllReduce communication for the small per-token activations is fast enough over NVLink (within 2 nodes) to not bottleneck. TP32 crosses the threshold where inter-node AllReduce begins to hurt. CP consistently degrades decode compared to single-node TP8, and the degradation worsens with more CP ranks.

**Table 8 — Attention micro-benchmarks explaining CP decode regression:** For 128K context, batch size 1:
- As CP ranks increase, the effective context length per rank decreases (128K → 64K → 32K for CP1→CP2→CP4), reducing individual attention op time (38.9 → 22.0 → 14.7 µs).
- However, the *total* attention time for the full ring loop increases (38.9 → 43.2 → 60.8 µs) because of communication overhead.
- SendRecv time grows with CP ranks (0 → 32.3 → 105.7 µs) and is largely exposed.
- All2All time is substantial and largely independent of attention per-step (0 → 81.1 → 79.9 µs).
- The whole pass-Q decode attention time degrades from 38.9 µs (TP8, no CP) to 157.7 µs (CP2) to 238.6 µs (CP4) — a 4–6× increase even though the per-GPU work has decreased.

**Batch size 4 results in Table 8 (32K context):** The pattern persists but with some differences:
- TP8 attention time: 60.1 µs (higher than B=1 because more sequences to process)
- CP2+TP8: 136.6 µs (2.3× slower than TP8)
- CP4+TP8: 180.6 µs (3.0× slower than TP8)
- The regression is less severe proportionally than for B=1 because the larger batch provides more computation to overlap with communication, but CP still substantially underperforms TP for decode.

**Two identified causes of CP decode regression (Section 4.3):**
1. **Padding overhead:** The current implementation pads queries to be divisible by the number of CP ranks. For B=1, CP4 processes 4 queries even though only 1 is real — a 4× inflation of effective query computation.
2. **Communication growth:** SendRecv and All2All latencies increase with the number of ranks, and these are largely on the critical path during decode because the per-step attention computation is too small to hide them.

**Practical implication:** The paper explicitly recommends disaggregated serving (Qin et al., 2024; Zhong et al., 2024) where CP-heavy nodes handle prefill and TP-only nodes handle decode. For standalone deployments where prefill and decode must run on the same hardware, the paper acknowledges the decode regression and notes that "removing batch padding and better overlap of computation and communication can help to minimize this regression" (Table 8 caption note).

---

### Ablation Studies and Robustness Checks

**Network bandwidth regime: RDMA vs. TCP/IP:** The paper tests scalability on two hardware configurations — Grand Teton Training (GTT) with 400 Gb/s RDMA per GPU inter-host, and Grand Teton Inference (GTI) with 100 Gb/s TCP/IP per GPU inter-host (Section 4.1). Figure 6(a) vs. 6(b) shows that CP achieves near-linear scaling on both, up to at least 4 nodes on GTI. This robustness check demonstrates that the method does not require premium high-bandwidth interconnects — 3 GB/s achieved bandwidth per rank on GTI is sufficient to overlap pass-KV communication with attention computation for long contexts. The paper explicitly frames this as evidence that "our method scales well using common commercial data center with medium-to-low inter-host bandwidth" (Abstract).

**Context length sweep for prefill:** Figure 6 sweeps context lengths from 2K to 128K across CP1/CP2/CP4/CP8 (GTT) and CP1/CP2/CP4 (GTI). The fan-out of latency curves demonstrates that near-linear scaling holds above a context-length-dependent threshold (approximately 8–16K on GTT, slightly higher on GTI) where the attention computation becomes large enough to hide communication. Below this threshold, the curves converge toward a constant overhead floor where communication dominates. This ablation implicitly validates Equation 2 — the condition $T \geq N \cdot C \cdot N_{KV} \cdot e / (2 \cdot N_H \cdot \text{BW})$ — by showing the context length at which scaling "kicks in."

**Number of CP ranks:** Scaling is tested with 1, 2, 4, 8, and 16 CP ranks (Figures 6, 7, 8). The near-linear trend holds through 8 ranks on GTT and 4 ranks on GTI. The 16-rank result is shown only for the 1M context run (Figure 8), where the extreme context length provides enough computation to hide communication even at 16 ranks. The paper does not show a 128K prefill on 16 CP ranks — likely because the per-rank context length (128K / 16 = 8K) would be too small to hide communication, violating Equation 2 for the given hardware parameters.

**Pass-KV vs. pass-Q at 128K total context across full miss rate spectrum:** Table 4 provides a comprehensive sweep of the $T/(T+P)$ space from 1% to 100% at 128K total context on CP4, giving the data for Figure 9. This is effectively an ablation of the algorithmic choice — it demonstrates that neither pass-KV nor pass-Q uniformly dominates, and that the optimal choice depends on the cache miss rate as predicted by the analytical model. The fact that the crossover occurs at approximately 5% rather than the simple model's 12.5% threshold validates the refined All2All-aware analysis in Appendix C (Equation 5).

**Empirical heuristic validation (Appendix D):** The paper fits a log-linear classifier to empirical $(T, T/(T+P))$ data points to create a deployment-ready decision boundary. The authors note that misclassified points near the boundary have performance differences under 1%, meaning the heuristic is "good enough" — choosing the wrong algorithm near the boundary costs almost nothing. This is a pragmatic robustness check: rather than requiring perfect classification, the system only needs to avoid large mistakes, and the analytical model identifies the regime where mistakes are costly (far from the boundary).

**FLOPS utilization calculation (Appendix A):** The 1M context run on 16 nodes achieves 502 TF/sec per H100. The paper validates this against a standalone Flash Attention v3 benchmark (540 TF/sec for 8K context on a single GPU, which is the equivalent per-GPU shard size for 1M tokens / 128 GPUs). This gives a 93% parallelization efficiency metric that isolates the communication and coordination overhead from the inherent attention kernel efficiency. The gap to theoretical peak (800 TF/sec → 63% utilization) is attributed to the power-limited H100 configuration (500W, 96 GB HBM2e at 2.4 TB/sec) rather than to CP overhead per se.

**Batch size sensitivity for decode (Table 8):** Comparing B=1 vs. B=4 at 32K context shows that CP decode regression is less severe at higher batch sizes (CP4 is 3.0× slower than TP8 at B=4 vs. approximately 4× at B=1). This suggests that CP decode overhead has a fixed component (communication) and a scaling component (padding), and that larger batches amortize the fixed overhead better. This is a robustness insight for throughput-oriented deployments rather than latency-critical interactive serving.

---

### Critical Assessment

The paper makes several central claims, and the experiments support them with varying degrees of completeness:

**Claim: "Near-linear scaling for long-context prefill latency with up to 128 H100 GPUs."** The evidence in Figures 6(a) and 7 strongly supports this for context lengths ≥ 16K on GTT hardware. The 128K prefill scaling from CP1 to CP8 achieves approximately 90% parallelization efficiency (7.2× speedup for 8× hardware). The 1M prefill on CP16 achieves 93% parallelization efficiency relative to per-GPU attention benchmarks. However, "near-linear" is never formally defined — the paper does not state what efficiency threshold qualifies as "near-linear." The scaling degrades at short context lengths (2K–8K), where communication is partially exposed, and the paper does not quantify the minimum context length for a given node count. The 16-node results are shown only for extreme context lengths (256K–1M), leaving open how 128K prefill would scale to 16 nodes (likely poorly, as per-rank context would shrink to 8K).

**Claim: "77s for 1M context prefill with 93% parallelization efficiency, 63% FLOPS utilization."** The 77-second figure is directly reported for CP16 and is plausible given the scaling from CP8 (~155s) to CP16 (~77s). The 93% efficiency is computed against a single-GPU Flash Attention benchmark, not against end-to-end single-node inference (which would give a different, likely higher, baseline due to the absence of node-to-node communication). The 63% FLOPS utilization is against theoretical peak, but the paper acknowledges this is for a power-limited H100 configuration (500W, 800 TF/sec peak) rather than the full 700W configuration (989 TF/sec). Against the full configuration, utilization would be lower. These efficiency metrics are reasonable but narrowly scoped to attention computation — they exclude the GEMM FLOPs (which are ~20% of total FLOPs for 1M context, per Appendix A) from the utilization calculation, which could slightly inflate the reported efficiency.

**Claim: "Scales well using common commercial data center with medium-to-low inter-host bandwidth."** The GTI results at 100 Gb/s TCP/IP support this for up to 4 nodes (Figure 6b). The paper does not test scaling beyond 4 nodes on GTI, and does not provide a minimum bandwidth requirement or a quantitative bandwidth-sensitivity analysis. The claim of "medium-to-low bandwidth" is contextual — 100 Gb/s per GPU (12.5 GB/s) is low relative to NVLink (~900 GB/s) but is still a dedicated high-performance interconnect by general data center standards. The paper does not test on shared or congested networks, which would be the true test of robustness for "common commercial data center" deployments.

**Claim: "Pass-Q outperforms pass-KV at low KV cache miss rates."** Table 4 and Figure 9 provide clean evidence with a well-characterized crossover at ~5% miss rate. The microarchitectural explanation in Table 5 is convincing. However, this is tested only at 128K total context length on CP4, leaving open whether the crossover point shifts with total context length (the analytical model predicts it should be stable, since Equations 2 and 3 produce static thresholds) or with the number of CP ranks (which does affect the threshold — the $N$ factor appears in both Equation 2 and the All2All-exposed-time comparison). The paper acknowledges this implicitly by providing the analytical model that generalizes to other configurations, but does not empirically validate the model's predictions for different context lengths or rank counts.

**Claim: "CP reduces prefill latency at the expense of decode latency regression."** Tables 6, 7, and 8 solidly support this. The decode regression is substantial — CP2 is ~44% slower than TP8 at 128K, CP4 is ~54% slower — and the paper is transparent about it. However, the paper does not quantify what "acceptable" decode regression would be for practical deployments, nor does it explore optimizations (beyond mentioning padding removal and better overlap) that might reduce the gap. The decode results are all at batch size 1 (for latency measurement), with only one B=4 data point in Table 8 — larger batch sizes, which are common in throughput-oriented serving, might show different CP-vs-TP tradeoffs that are not explored.

**Genuine weaknesses and missing experiments:**

1. **Single model, single hardware platform.** All results are on Llama3 405B with H100 GPUs. The analytical model (Equations 2 and 3) provides a framework for generalization, but the paper does not validate on other models (different GQA ratios, different depths) or other hardware (A100, upcoming B200). The claim that the method is broadly applicable rests on the analytical model's correctness, which is assumed but not tested across model architectures.

2. **No end-to-end application benchmark.** All measurements are system-level latency metrics (TTFT, TTIT). The paper does not evaluate on any downstream task — e.g., long-document QA, summarization, or needle-in-a-haystack retrieval — that would validate that the lossless exact attention property translates to identical task performance. While the mathematics of ring attention guarantees identical outputs, system-level interactions (numerical precision differences from different accumulation orders in the merge attention, or FP8 quantization artifacts) could in principle cause subtle deviations that might affect downstream task accuracy. This is unlikely to be a practical issue given the merge attention's mathematical exactness, but a minimal validation on a standard long-context benchmark would strengthen the claim of losslessness.

3. **No comparison to approximate attention methods.** The paper positions itself as providing lossless exact attention and notes that approximate methods are complementary, but does not benchmark against them. A comparison showing, for instance, that CP-based exact attention at 128K context achieves latency competitive with approximate sparse attention methods would strengthen the practical case. If a sparse method achieves 2s prefill at 128K on a single node while CP requires 4 nodes to achieve 3.8s, that's relevant information for practitioners deciding between accuracy and cost.

4. **No quantification of memory overhead.** The paper notes that CP "incurs higher memory consumption because its lack of model weight sharding" (Section 3.2), requiring TP within nodes to compensate. The actual memory breakdown — how much HBM is consumed by model weights vs. KV cache vs. activation buffers under CP vs. TP at various context lengths — is not provided. This matters for capacity planning: at what context length does the KV cache memory dominate and make the CP memory overhead irrelevant? The 1M-token result on 16 nodes implicitly demonstrates that memory is manageable, but the per-GPU memory consumption is not reported.

5. **No sensitivity analysis on the heuristic threshold.** The empirical heuristic (Appendix D) is fit to specific data points, but the paper does not report how sensitive the pass-KV/pass-Q performance is to getting the threshold wrong. The 1%-difference-at-boundary observation is anecdotal for one configuration — a systematic sweep showing performance degradation as a function of distance from the optimal threshold would quantify the cost of misconfiguration and inform deployment choices.

6. **Batched prefill scaling not evaluated.** All prefill scaling results use batch size 1. Real-world serving systems batch multiple requests to improve throughput. With batched prefill, the attention computation grows (multiple sequences, each attending to their own KV), potentially making it easier to hide communication. However, load-balanced sharding becomes more complex with variable-length sequences in a batch (the paper describes the mechanism in Section 3.5.1 and Figures 1–2 but does not benchmark batched prefill scaling). The omission of batched prefill results limits the practical applicability of the scaling claims.

7. **No quantification of the persistent KV cache miss rate distribution in real workloads.** The paper's adaptive switching between pass-KV and pass-Q depends on the KV cache miss rate, but the paper provides no data on what miss rate distributions look like in production multi-turn conversation workloads. The evaluation sweeps the full 1–100% range, which is thorough for system characterization, but without workload distributions, it's unclear how often the adaptive switching actually matters in practice or what the average benefit is for a realistic request mix.

**Conditional validity of claims:**

- The near-linear prefill scaling holds **when the per-rank attention computation time exceeds the per-step communication time** (Equation 2). For Llama3 405B on H100/GTT, this means $T/N$ must be sufficiently large. The paper implicitly demonstrates the threshold by showing scaling behavior across context lengths, but the exact threshold (in tokens per rank) for different hardware configurations is not tabulated. A practitioner wanting to deploy on different hardware would need to compute the threshold from Equation 2 using their own $C$ and $\text{BW}$ values.

- The CP-over-TP advantage holds **when inter-node bandwidth is substantially lower than intra-node bandwidth** — true for current H100 clusters with NVLink intra-node and RDMA/TCP inter-node, but not necessarily for future systems with unified interconnects (GB200 with NVLink across nodes, as the paper notes).

- The pass-Q advantage for decode holds only if the KV cache hit rate is very high (small $T$ relative to $P$). For the first turn of a conversation (full prefill, $P=0$), pass-KV is always preferred.

- The 77s 1M-token result depends on **FP8 quantization for feed-forward layers** to fit the model in TP8 within each node. Without quantization, TP16 (2 nodes) would be needed just for model weights, reducing the number of nodes available for CP scaling and potentially degrading the overall latency.

**Experiments that would have strengthened the paper:**

- **Validation on a model with different GQA ratio** (e.g., a model with 32 Q heads and 32 KV heads, or a non-GQA model with equal Q and KV heads). This would validate the analytical model's prediction that the pass-KV/pass-Q crossover threshold scales with $N_{KV}/N_H$.
- **End-to-end accuracy benchmark** on a long-context task (e.g., needle-in-a-haystack retrieval at 128K and 1M) comparing CP-based inference to single-GPU inference — validating the losslessness claim at the application level.
- **Batched prefill scaling results** (B=2, 4, 8, 16) to show whether CP's near-linear scaling extends to throughput-oriented serving scenarios.
- **Memory consumption breakdown** per GPU across different context lengths and CP configurations, to provide concrete capacity planning guidance.
- **Sensitivity analysis** varying the inter-host bandwidth (simulated via throttling) to characterize the minimum bandwidth requirements for near-linear scaling at different context lengths.
- **Comparison with sequence parallelism alternatives** (DeepSpeed Ulysses, DistFlashAttn) adapted for inference, to position CP's performance relative to other exact-attention distribution strategies.
- **Dynamic switching overhead measurement** — the cost of evaluating the heuristic and switching between pass-KV and pass-Q at the start of each partial prefill, to confirm it is negligible as the paper assumes.

## 6. Limitations and Trade-offs

### 6.1 Decode Latency Regression Under Context Parallelism

**The assumption or constraint.** The paper's context parallelism is designed primarily to accelerate the prefill phase, where computation is abundant enough to hide communication under the ring SendRecv operations. The decode phase — generating one token at a time autoregressively — has fundamentally different characteristics: each step computes attention for a single new token against the full KV cache, producing far less computation per communication round. The paper acknowledges this asymmetry explicitly in the conclusion:

> "For standalone deployment where prefill and decode are both on the same set of hosts, CP drastically improves the prefill latency, at the expense of decode latency regression"

**The consequence.** Any deployment that uses CP for both prefill and decode on the same hardware will suffer a substantial increase in time-to-incremental-token (TTIT). At 128K context length with CP2, TTIT degrades by approximately 44% compared to single-node TP8 (66.6 ms vs. 46.3 ms in Table 6). With CP4, the regression grows to 54% (71.3 ms vs. 46.3 ms in Table 7). For interactive applications where users perceive both time-to-first-token and the rate of token generation, this tradeoff means that CP's prefill speedup comes at the direct cost of a slower, choppier generation experience after the first token appears. For a response of 100 tokens, the additional decode latency under CP2 at 128K context would be roughly `(66.6 − 46.3) × 100 ≈ 2` seconds of extra user waiting time spread across the generation — partially offsetting the prefill gain.

The problem intensifies with more CP ranks. Table 8 reveals why: while the per-rank attention computation shrinks with more ranks (38.9 → 22.0 → 14.7 µs for CP1 → CP2 → CP4 at 128K), the communication components (SendRecv growing from 0 → 32.3 → 105.7 µs, and All2All sitting at 0 → 81.1 → 79.9 µs) dominate the total attention time, which inflates from 38.9 µs (TP8) to 157.7 µs (CP2) to 238.6 µs (CP4) — a 4–6× increase. Two factors are identified: (1) padding overhead where the current implementation pads the number of queries to be divisible by the number of CP ranks, inflating the effective query count for small batches, and (2) the SendRecv and All2All latencies that sit on the critical path because the per-step attention computation is too tiny to hide them.

**What evidence exists in the paper.** Tables 6, 7, and 8 provide direct measurements. Table 6 sweeps context lengths (8K, 32K, 128K) at batch size 1, showing CP2 TTIT is consistently ~44–47% higher than TP8 regardless of context length — the regression is flat because decode's per-token linear-layer computation is constant and the attention communication overhead does not scale with context length in the same way prefill does. Table 7 adds a useful comparison: TP16 (2-node tensor parallelism) actually *improves* decode TTIT relative to single-node TP8 (39.5 ms vs. 46.3 ms), while CP2 degrades it (60.2 ms). This means that at 2 nodes, the choice between TP and CP for decode is stark — TP helps, CP hurts. Table 8 provides the microarchitectural breakdown confirming that communication overhead, not computation, is the culprit.

**Mitigation status.** The paper proposes two mitigation strategies, neither of which is fully realized in the current system:

1. **Disaggregated prefill and decode** (Section 5, referencing Qin et al., 2024 and Zhong et al., 2024): run CP-heavy nodes for prefill and separate TP-only nodes for decode. This is presented as the preferred architecture but is not implemented or evaluated in this paper. The practical challenges — KV cache transfer from prefill nodes to decode nodes, load balancing between the two pools, consistency under failures — are not addressed.

2. **Implementation-level fixes** (Section 4.3): removing the batch padding that inflates query counts and improving the overlap of computation with communication during decode. These are mentioned as potential improvements but are not benchmarked.

Without disaggregation, a practitioner deploying CP for prefill on a standalone system simply has to accept the decode penalty. The paper provides no guidance on what workload characteristics (prefill-to-decode ratio, token generation length, latency budget) would make the overall user-experience tradeoff worthwhile.

---

### 6.2 The Difficulty Estimation Cost Is Unaccounted For — and It Dominates the Inference Budget

**The assumption or constraint.** The entire compute-optimal framework depends on knowing each prompt's difficulty *before* the system decides how to allocate the inference budget. The paper's difficulty estimation method generates 2048 samples per question and computes either the ground-truth pass@1 rate (oracle) or the average PRM final-answer score (predicted). The authors are candid about the cost:

> "estimating difficulty in this way still incurs additional computation cost during inference... our experiments do not account for this cost largely for simplicity"

This is not a minor omission — it is a fundamental gap between the paper's reported efficiency gains and what a real deployment would experience. The 2048 samples required for difficulty estimation represent **8× to 128× more computation** than the largest test-time budgets studied (256–512 generations), depending on the budget level.

**The consequence.** The headline $4\times$ efficiency improvement over best-of-N (Figures 4 and 8) is computed *after* difficulty is known, without including the cost of learning it. In a real deployment, the total compute cost would be:

$$\text{Total cost} = \text{difficulty estimation cost} + \text{strategy execution cost}$$

Since difficulty estimation consumes 2048 generations (or 2048 PRM-scored generations) per prompt, and the strategy execution uses at most 256–512 generations, the difficulty estimation cost dominates the budget by a factor of 4–8× at the maximum strategy budget, and much more at lower budgets. The $4\times$ efficiency claim is therefore more accurately described as a **conditional efficiency**: if difficulty is already known (e.g., from offline pre-computation on a static set of problems), then the adaptive strategy achieves $4\times$ better compute efficiency than best-of-N at equivalent accuracy. For online deployment where each prompt requires fresh difficulty estimation, the total cost would be *worse* than simply running best-of-N with a large budget — the savings from adaptive allocation are swamped by the estimation overhead.

The problem is compounded by the fact that the difficulty estimator itself uses the same PRM that the search strategies depend on. If the PRM has blind spots (as it demonstrably does — it suffers from over-optimization on easy problems, Section 5.3), those blind spots corrupt both the difficulty estimate and the strategy selection, potentially routing prompts to suboptimal strategies in ways that compound errors.

**What evidence exists in the paper.** The paper explicitly acknowledges this limitation in Section 3.2 and frames it as an "exploration-exploitation tradeoff" — compute spent assessing difficulty versus compute spent solving the problem. The difficulty estimation cost is stated (2048 samples per question) but never included in any budget calculation, latency measurement, or efficiency comparison. Figures 4 and 8 show compute-optimal scaling curves that start from the *first generation of the strategy execution*, with the 2048-sample estimation cost invisible. The paper offers no amortization analysis — e.g., whether difficulty estimates can be reused across similar prompts, or whether a lightweight difficulty predictor could be trained to replace the expensive sampling-based estimator.

**Mitigation status.** The paper flags this as "a key avenue for future work" (Section 3.2) and suggests:

> "future work exploring training models to directly predict the difficulty of a question, given its text"

No such model is developed or evaluated. The paper also does not explore adaptive schemes where difficulty estimation and strategy execution are interleaved — e.g., start with a small number of samples, estimate difficulty coarsely, allocate some budget, update the estimate based on intermediate results, and continue. Such a design could partially amortize the estimation cost into the solution process, but it would require a fundamentally different framing of the allocation problem that this paper's static difficulty-bin approach does not support.

A practitioner reading this paper should treat the $4\times$ figure as an **upper bound on the efficiency gain achievable if difficulty can be estimated cheaply through an orthogonal mechanism** — not as a realized deployment gain with the current method. Until the difficulty estimation problem is solved (with a cost at least $10\times$ lower than the strategy budget), the compute-optimal framework remains an analytical contribution rather than a directly deployable system.

---

### 6.3 Hard Problems Remain Fundamentarily Unsolved — Test-Time Compute Cannot Substitute for Capability Gaps

**The assumption or constraint.** The paper's entire approach — across both PRM search and iterative revisions — operates on the premise that the base model's proposal distribution contains correct solutions at some non-trivial rate. When this premise fails, no allocation of test-time compute helps. The authors are explicit about this boundary:

> "on the hardest questions (bin 5), no method makes meaningful progress — the base model simply lacks the capability to produce correct solutions regardless of how the budget is allocated"

This is stated in Section 5.3 for search and confirmed in Section 6.2 for revisions, but it warrants treatment as a separate limitation because it defines the **outer boundary of applicability** for the entire paradigm.

**The consequence.** For any problem distribution that includes a substantial fraction of questions where the base model's pass@1 is effectively zero, test-time compute offers no path forward. The FLOPs-matched comparison in Section 7 makes this concrete: on hard problems (difficulty bins 4–5), the smaller model with compute-optimal test-time strategies performs *worse* than a ~14× larger pretrained model with greedy decoding across nearly all values of the inference-to-pretraining ratio $R$. At $R \gg 1$, the disadvantage is severe — -37.2% relative for revisions and -52.9% for PRM search (Figure 1 bar charts, Figure 9). On difficulty bin 5 specifically, the scaling curves in Figure 9 are essentially flat near 0–5% accuracy regardless of how much test-time compute is applied.

This has direct practical implications. For a deployment where the problem mix includes genuinely hard problems — novel reasoning tasks that the base model was not exposed to during training, out-of-distribution problem structures, or problems requiring capabilities beyond the model's training curriculum — test-time compute amplification provides zero benefit. The FLOPs that would be spent on search or revisions for these problems are completely wasted; they would be better invested in pretraining a larger model, curating better training data, or routing the problem to a more capable system.

The paper's answer to "test-time compute or pretraining?" is thus: **it depends on whether the problem is within the base model's capability envelope**. For easy-to-medium problems (bins 1–3), test-time compute can substitute for pretraining — sometimes dramatically (e.g., +27.8% relative improvement on easy questions at $R \ll 1$ with revisions). For hard problems (bins 4–5), pretraining is the only viable path. The practical difficulty is that identifying which bin a prompt belongs to *before* deciding the strategy requires the expensive difficulty estimation discussed in Limitation 6.2 — creating a chicken-and-egg problem where you need to spend substantial compute just to learn that no amount of additional compute will help.

**What evidence exists in the paper.** The difficulty-bin analysis is the central evidence mechanism. Figure 3 (right) shows beam search accuracy on bin 5 hovering at 1–3% across all generation budgets from 4 to 256 — a flat line at near-zero. Figure 7 (right) shows the same for revisions: bin 5 accuracy at roughly 2–3% regardless of the sequential-to-parallel ratio at a fixed 128-generation budget. Figure 9 shows the FLOPs-matched comparison broken out by difficulty bin, with the bin 5 (blue, bottommost) scaling line essentially flat and below the $14\times$ larger model's star across all $R$ values. The paper's takeaway box in Section 7 directly states the conclusion:

> "Test-time compute can improve performance on easy-to-medium difficulty problems… However, on hard problems (difficulty bins 4-5), test-time compute is almost always less effective than scaling pretraining."

**Mitigation status.** None, and arguably none is possible within the paradigm. This is not a weakness of the method per se — it is a fundamental statement about the relationship between pretraining and inference compute. Test-time compute *amplifies* existing capability by exploring the model's output distribution more thoroughly; it does not *create* capability that was not acquired during training. The paper is transparent about this boundary, and the transparency is a strength of the analysis. However, from a practitioner's perspective, this limitation means that deploying compute-optimal test-time scaling requires a prior estimate of whether the target problem distribution is within the model's reach — which may not be knowable in advance for open-ended deployment scenarios.

---

### 6.4 Single Benchmark, Single Model Family: Generality Is Unvalidated

**The assumption or constraint.** All experiments — prefill scaling, partial prefill, decode, the pass-KV/pass-Q comparisons, and the FLOPs-matched analysis — are conducted on a single benchmark (MATH) with a single model family (PaLM 2-S\*). The paper acknowledges this scope limitation only implicitly through its framing, and the authors state in Section 4 that they "believe this model is representative of the capabilities of many contemporary LLMs."

**The consequence.** Several of the paper's key findings may be specific to the MATH benchmark's characteristics or to properties of PaLM 2-S\* that do not generalize:

- **PRM over-optimization behavior (Figure 3, right):** The finding that beam search degrades easy-problem performance at high budgets depends on the PRM's error characteristics on PaLM 2-S\*'s output distribution. A model with better-calibrated outputs (or a PRM trained with different data) might exhibit weaker or stronger over-optimization, shifting the optimal difficulty-dependent strategy thresholds. The paper does not explore whether the over-optimization phenomenon is inherent to PRM-guided search or an artifact of this specific PRM training procedure.

- **Revision model effectiveness (Section 6):** The revision model's ability to improve its outputs by conditioning on previous incorrect answers depends on the base model's in-context learning capabilities and its ability to recognize and correct its own errors. Different model families (or different model sizes within the same family) could show substantially different revision benefits. The paper's finding that 38% of correct answers get "revised" back to incorrect ones (Section 6.1) might be more or less severe for other models.

- **Difficulty bin boundaries:** The five difficulty quintiles are defined relative to PaLM 2-S\*'s pass@1 distribution on MATH. A different model with different strengths and weaknesses would produce different bin assignments for the same questions, potentially changing which strategies are optimal for which bins. The paper's compute-optimal policies are specific to this model-benchmark pair.

- **MATH-specific properties:** MATH consists of competition-level math problems with well-defined ground-truth answers and multi-step symbolic reasoning. The paper's key findings — that revisions help on easy problems but not hard ones, that beam search over-optimizes on easy problems, that hard problems see zero benefit — might not transfer to other reasoning domains (code generation, logical deduction, scientific QA) or to tasks involving factual knowledge rather than procedural reasoning. The PRM training procedure (Monte Carlo rollouts with ground-truth answer checking) requires clean answer verification, which is available for MATH but not for open-ended generation tasks.

**What evidence exists in the paper.** None. The paper does not include experiments on any benchmark other than MATH, nor does it validate any findings on a model other than PaLM 2-S\*. The FLOPs-matched comparison uses a second model (a ~14× larger PaLM 2 variant), but that is within the same model family and training regime. The paper does not test on other model families (e.g., LLaMA, Gemma, Qwen) with different architectures, training data mixtures, or calibration properties. This is a significant gap for a paper that makes claims about the general relationship between test-time compute and pretraining compute.

**Mitigation status.** The paper acknowledges the single-benchmark limitation indirectly by stating that the PaLM 2-S\* model is "representative," but does not provide evidence for this claim. No discussion of benchmark diversity or model generalization appears in Section 8 (Limitations and Future Work) — the future work suggestions focus on combining search with revisions and improving difficulty estimation, not on validating generality. A practitioner considering adopting these methods should treat the findings as **validated for math reasoning with PaLM 2-scale models** and should expect to re-tune the difficulty thresholds, strategy selections, and possibly the revision model training procedure for different models and tasks.

---

### 6.5 The Revision Model's Correct-to-Incorrect Reversion Problem Is a Hard Reliability Issue

**The assumption or constraint.** The revision model is trained exclusively on sequences where all in-context answers are incorrect, followed by a correct target (Section 6.1). The model never sees examples where the current answer is already correct and should be preserved. As a direct consequence:

> "at test time the model may encounter correct answers in its context (produced during earlier revisions) and incorrectly 'revise' them into wrong answers. The paper reports that approximately 38% of correct answers get converted back to incorrect ones using a naive approach"

This is a training data artifact, not a fundamental limitation of the revision paradigm — but it has practical consequences that the paper's mitigation strategies only partially address.

**The consequence.** In a deployed system using iterative revisions, the revision chain is **not monotonically improving**. A correct answer generated at revision step $k$ has a 38% chance of being replaced by an incorrect answer at step $k+1$. This creates two problems:

1. **Unreliable termination:** The system cannot simply stop after a fixed number of revisions and take the last output — doing so would discard correct answers that the model subsequently "unfixes." The paper's mitigation — majority voting or verifier-based selection across the entire revision chain — means the system must run the full budget of revisions and then retrospectively select the best answer. This makes the revision process inherently non-interruptible: you cannot stop early if you get a high-confidence correct answer because you don't know if the next revision will break it.

2. **Wasted computation:** If 38% of correct answers are "unfixed," then a non-trivial fraction of the revision budget is spent generating incorrect revisions of previously correct answers — computation that actively degrades the candidate pool. The chain improves on average (Figure 6 left shows pass@1 increasing from ~18.2% to ~24–25% by steps 15–20), but the improvement is slower than it would be without the reversion problem, and the variance across chains is likely high (the paper does not report variance across chains).

3. **Verifier dependence:** The system relies on the verifier (or majority voting) to correctly identify which revision in the chain is the correct one. If the verifier makes errors — and it demonstrably does, especially on the sort of medium-difficulty problems where revisions are most beneficial — it might select an incorrect revision even when a correct one appeared earlier in the chain.

**What evidence exists in the paper.** The 38% figure is reported in Section 6.1 as an empirical measurement. Figure 6 (left) shows the pass@1 trajectory across revision steps — it increases overall but with notable non-monotonicity in the per-step accuracy (the curve is noisy and not smoothly increasing). The paper's mitigation — using majority voting or verifier-based selection across the chain — is evaluated in Figure 6 (right), showing that sequential revisions with both selection methods outperform parallel sampling, confirming the mitigation works on average. However, the paper does not report the variance in revision chain quality, the distribution of "correct answer survival times" (how many steps a correct answer typically persists before being revised away), or an ablation comparing chain-based selection to a version where the model is trained to recognize when no revision is needed.

**Mitigation status.** The paper's mitigation — selecting the best answer from any point in the chain rather than always taking the last revision — is effective on average (Figure 6 right) but is a post-hoc fix rather than a solution to the root cause. The root cause is the training data construction: by only including incorrect-to-correct trajectories, the model never learns the "no revision needed" action. The paper does not explore training data modifications that would include "correct → correct" examples (e.g., by appending a special token indicating the answer is already correct), nor does it evaluate whether the reversion rate can be reduced through different training strategies. The ReST$^{EM}$ experiment (Appendix K, Figure 16) shows that attempting to optimize the revision model further with RL-style training can make the problem worse — sequential revisions with the ReST$^{EM}$ model *substantially hurt* performance — highlighting that the revision training process is fragile and not well-understood.

For a practitioner, this means that deploying a revision model requires careful calibration: you must measure the reversion rate on your target distribution, verify that your verifier or majority-voting mechanism reliably identifies the best answer in noisy revision chains, and potentially limit the revision depth to prevent correct answers from being revised away. The paper provides the diagnostic tools but no prescription for solving the underlying reliability problem.

---

### 6.6 The Compute-Optimal Policy Is Selected on a Small Cross-Validation Split — Robustness to Distribution Shift Is Unaddressed

**The assumption or constraint.** The compute-optimal strategies (which search algorithm, which sequential-to-parallel ratio) are selected using two-fold cross-validation on the 500-question MATH test set, stratified by difficulty bin (Section 3.2). This means the optimal strategy for a given difficulty bin is determined based on performance on approximately **50 questions per fold per bin** (500 questions / 5 bins / 2 folds ≈ 50). The paper then reports results averaged across folds.

**The consequence.** The selected policies may not be robust. With only ~50 questions per bin per fold, the strategy that happens to perform best on those 50 questions may not be the strategy that performs best on a different set of questions from the same difficulty bin — let alone on questions from a different distribution. This is a standard overfitting concern in any meta-learning or hyperparameter selection setting, but it is particularly acute here because:

- **The strategy space is discrete and combinatorial:** The paper sweeps multiple search algorithms (best-of-N, beam search with different widths, lookahead search with different lookahead steps) and multiple sequential-to-parallel ratios. With a small validation set, the selection is vulnerable to picking strategies that happen to fit noise in the validation fold rather than generalizing.

- **The difficulty bins are coarse:** Questions within bin 3, for instance, might have pass@1 rates ranging from, say, 5% to 15%. The optimal strategy likely varies continuously with difficulty, but the five-bin discretization forces a single strategy for all questions in the bin, even if a more nuanced allocation would be better. Finer-grained binning would exacerbate the small-sample problem — more bins means fewer questions per bin per fold.

- **No confidence intervals are reported:** The paper does not provide error bars or confidence intervals on the compute-optimal scaling curves in Figures 4 and 8. This makes it impossible for a reader to assess whether the observed $4\times$ efficiency gain is statistically reliable or whether it could have arisen from strategy overfitting to the particular test set split.

The cross-validation protocol does prevent information leakage from the test set into strategy selection, which is good practice. But 2-fold CV on 500 questions is a thin reed for claims about optimal strategy selection — the variance of the cross-validated performance estimate is likely high, and the selected strategies may not be stable across different random splits.

**What evidence exists in the paper.** The cross-validation protocol is described in Section 3.2. The paper reports that oracle and predicted difficulty bins "largely overlap" (Figures 4 and 8), which provides some reassurance — the selected strategies are similar whether you use ground-truth or PRM-estimated difficulty. However, this only addresses the robustness of the difficulty estimation, not the robustness of the strategy selection itself. The paper does not report the variance of the cross-validated estimates, the stability of strategy selection across folds (i.e., how often a different strategy was selected for the same bin in fold 1 vs. fold 2), or the performance of the selected strategies on a completely held-out calibration set.

The predicted difficulty bins showing slightly lower performance at high budgets in the revision setting (Figure 8: approximately 41% for predicted vs. 44% for oracle at 256 generations) could indicate strategy overfitting — the oracle-selected strategies may be fitting to the specific questions in the test set in ways that don't transfer to PRM-estimated bins.

**Mitigation status.** The paper does not discuss the robustness of the policy selection procedure or propose mitigations. Potential mitigations that are not explored include: (1) reporting bootstrap confidence intervals for the compute-optimal scaling curves; (2) using a larger number of folds (e.g., 5-fold or 10-fold CV) to reduce the variance of the performance estimate; (3) fitting a smooth policy function (e.g., logistic regression) that maps continuous difficulty estimates to strategy parameters, rather than discrete bin-based lookup; (4) evaluating the selected strategies on a completely held-out set of MATH problems (e.g., from a different split) or on a different math benchmark to assess generalization.

For a practitioner, this limitation means that the specific strategy recommendations in the paper (e.g., "use beam search with M=4 for bin 3") should be treated as approximations that require validation and likely re-tuning on the target problem distribution. The principle — difficulty-adaptive test-time compute allocation — is likely robust, but the specific policy parameters are fitted to a small, specific dataset and may not transfer.

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
