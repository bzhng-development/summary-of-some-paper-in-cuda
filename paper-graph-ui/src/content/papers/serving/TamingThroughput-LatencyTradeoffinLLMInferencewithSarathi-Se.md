# Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

**ArXiv:** [2403.02310](https://arxiv.org/abs/2403.02310)

## 🎯 Pitch

Sarathi-Serve introduces a novel scheduling system for LLM inference that splits heavyweight prefill computations into manageable chunks and co-schedules them with ongoing decode steps in a stall-free manner. This innovation eliminates longstanding trade-offs between high throughput and low latency, enabling responsive, cost-efficient LLM serving—even under heavy loads—while unlocking the full potential of pipeline parallelism. As a result, applications like chatbots and code assistants can achieve both fast token streaming and dramatically improved serving capacity, solving a fundamental bottleneck in LLM deployment.

---

## 1. Executive Summary

This paper introduces **Sarathi-Serve**, an LLM inference scheduler that resolves the fundamental throughput-latency tradeoff in serving systems through two key techniques—**chunked-prefills** (splitting long prefill requests into equal compute-sized chunks processed across multiple iterations) and **stall-free batching** (coalescing ongoing decodes with prefill chunks without pausing decode execution). Evaluated across Mistral-7B, Yi-34B, LLaMA2-70B, and Falcon-180B on the openchat_sharegpt4 and arxiv_summarization datasets, Sarathi-Serve achieves up to 2.6× higher serving capacity for Mistral-7B on a single A100 GPU and up to 5.6× gains for Falcon-180B with pipeline parallelism on 8 A100 GPUs compared to vLLM, while eliminating generation stalls that cause tail latency spikes in prefill-prioritizing schedulers. The system's uniform-compute hybrid batches also significantly reduce pipeline bubbles in cross-node deployments, establishing that high throughput and low tail latency can be jointly achieved without disaggregating prefill and decode phases—but only when token budgets are tuned to bound per-iteration latency based on the target SLO.

## 2. Context and Motivation

### The Core Problem: Throughput and Latency Are Fundamentally at Odds in LLM Inference

Every LLM serving request goes through two distinct phases with markedly different computational properties. The **prefill phase** processes the entire input prompt in parallel and produces the first output token. Because it handles hundreds to thousands of tokens simultaneously, prefill is **compute-bound** — it saturates GPU compute and achieves high arithmetic intensity. The **decode phase** generates subsequent output tokens one at a time, autoregressively, where each new token requires a full forward pass through the model. Since a decode iteration processes only a single token per request, it is **memory-bound** — it has very low arithmetic intensity and significantly underutilizes GPU compute capacity.

This asymmetry creates the central tension the paper addresses. Batching multiple requests together dramatically improves decode throughput because the cost of fetching model weights from GPU memory can be amortized across many requests. However, batching necessarily interleaves prefill and decode iterations from different requests. This interleaving is where the trouble starts: a prefill iteration can take orders of magnitude longer than a decode iteration depending on the prompt's length, and when a long prefill is scheduled between consecutive decode iterations of an ongoing request, that request experiences a **generation stall** — a sudden spike in time-between-tokens (TBT) latency that can last multiple seconds (Figure 1a shows stalls exceeding several seconds in vLLM). These stalls violate the latency service level objectives (SLOs) that interactive applications like chatbots require for fluid, responsive output.

The problem is not merely academic. LLM inference has become a dominant GPU workload across the industry — powering chatbots (ChatGPT, Claude), code assistants (GitHub Copilot, CodeWhisperer), search (Bing AI, Perplexity), and productivity tools (Microsoft Copilot, Google Duet AI). Each of these applications demands both **high throughput** (to keep serving costs tractable under heavy user load) and **low tail latency** (to maintain response fluidity for individual users). The paper's Figure 1b illustrates the practical consequence of failing to balance these objectives: as load increases on a state-of-the-art system like vLLM, the 99th percentile TBT latency degrades sharply, forcing operators into a difficult choice between over-provisioning expensive GPU capacity or delivering degraded user experience.

### Why This Problem Is Important

The throughput-latency tradeoff has real economic and architectural implications for LLM deployment:

**Cost sensitivity at scale.** LLM inference is extraordinarily expensive due to the GPU compute and memory requirements of large models. For organizations serving millions of requests daily, throughput directly determines the number of GPU replicas needed and therefore the total infrastructure cost. Each additional replica can cost thousands of dollars per month in cloud GPU rentals. Techniques that improve throughput without degrading latency thus translate directly to operational cost reduction.

**Latency requirements are non-negotiable for interactivity.** For chatbots and real-time assistants, users expect fluid token-by-token output generation. A TBT SLO of 100–200ms is typical — any spike that delays token delivery by seconds makes the interaction feel broken, even if the final response is correct. This is why the paper evaluates under both "strict" and "relaxed" SLOs (Section 5.2): the strict regime represents the latency targets required for interactive deployments, and the paper shows that existing systems fail to reach their maximum throughput under these constraints because generation stalls dominate.

**Parallelism choices compound the problem.** As models grow beyond what fits on a single GPU, operators must choose between tensor parallelism (TP) and pipeline parallelism (PP) for distributed inference. TP shards each layer across GPUs but requires expensive all-reduce communication operations in the critical path, making it viable only within a single node with high-bandwidth NVLink interconnects. PP splits the model by layers across GPUs and uses point-to-point communication, making it more suitable for cross-node deployments over commodity networks. However, PP introduces **pipeline bubbles** — periods where some pipeline stages sit idle waiting for the previous stage to complete its micro-batch. The paper identifies (Section 3.3) that these bubbles are not eliminated by existing iteration-level scheduling as prior work had assumed; they arise because the varying mix of prefill and decode tokens in each micro-batch creates heterogeneous execution times, leaving some pipeline stages waiting. On a model like Falcon-180B, a single 4K-token prefill micro-batch takes approximately 1150ms while a decode-only micro-batch (batch size 32) takes only 200ms — a difference that wastes 950ms of GPU time per bubble. These bubbles compound with batch size and prompt length, making PP across commodity networks nearly unusable for latency-sensitive applications.

**The industry lacks a principled resolution.** Prior to this work, no system had demonstrated that high throughput and low tail latency could be jointly achieved without physically disaggregating the prefill and decode phases across separate GPU pools. The tradeoff was considered inherent to the interleaved nature of batched LLM inference.

### Where Prior Approaches Fall Short

The paper systematically characterizes existing LLM inference schedulers into two broad categories and shows that both have fundamental limitations:

#### Decode-Prioritizing Schedulers (e.g., FasterTransformer, Triton Inference Server)

These systems use **request-level batching** (Algorithm 1 in the paper): they form a batch of requests, compute prefills for all of them, then run decodes for all of them until the last request in the batch completes. New requests are not admitted until the current batch is fully finished. This approach ensures that ongoing decodes are never interrupted by new prefills, yielding low and predictable TBT latency. However, it **severely compromises throughput** in three ways:

1. **Premature request completion wastes batch capacity.** Different requests in a batch have widely varying output lengths. When some requests finish early, the batch continues with a shrinking number of active requests, progressively reducing GPU utilization. The GPU is partially idle for the remaining decodes.
2. **Padding introduces computational waste.** To simplify batch processing, request-level systems pad shorter requests to match the longest sequence in the batch, performing useless computation on zero tokens.
3. **Head-of-line blocking.** New requests queue up waiting for the current batch to fully complete, increasing time-to-first-token (TTFT) for waiting users and limiting overall system throughput.

As Kwon et al. [53] demonstrated, iteration-level batching can achieve an order of magnitude higher throughput than request-level systems like FasterTransformer. This makes decode-prioritizing schedulers economically uncompetitive for high-volume serving, even though they provide good latency characteristics.

#### Prefill-Prioritizing Schedulers (e.g., Orca, vLLM, TensorRT-LLM)

These systems use **iteration-level batching** (Algorithm 2): requests can dynamically enter and exit the batch at each model iteration boundary. When GPU memory becomes available (e.g., a request finishes and its KV-cache can be freed), new requests are eagerly admitted by computing their prefill phase immediately. This approach improves throughput because it maximizes the batch size in subsequent decode iterations — the more requests are in the decode phase simultaneously, the better the GPU utilization.

However, the paper identifies a critical problem that these systems share despite their throughput advantages: **generation stalls**. A generation stall occurs when one or more prefills are scheduled between consecutive decode iterations of an ongoing request (Section 3.2, Figure 7). The stall's duration depends on the length of the incoming prompt's prefill, which can be thousands of tokens. For example, in the arxiv_summarization dataset, the median prompt length is 7,059 tokens (Table 2). A prefill of this length can take hundreds of milliseconds or even seconds, during which the ongoing decodes sit idle waiting for their next iteration. The result is a latency spike in TBT that can violate SLOs and degrade user experience.

The paper distinguishes between two variants of prefill-prioritizing schedulers:

- **Orca** supports **hybrid batches** that contain both prefill and decode tokens in the same iteration. This allows some overlap but does not solve the generation stall problem because the runtime of a hybrid batch is dominated by the prefill component when the prompt is long. As Figure 9 shows, combining a full prefill with decode batches can increase TBT latency by up to 28.3× compared to a decode-only batch.

- **vLLM** (paired with PagedAttention for efficient KV-cache memory management) takes a different approach: it never mixes prefills and decodes in the same batch. Instead, batches are either all-prefill or all-decode. When new requests arrive, vLLM schedules as many prefills as possible in one or more all-prefill iterations before resuming decodes. This avoids the hybrid batch runtime but makes generation stalls even more pronounced — the ongoing decodes are completely paused while the system processes potentially many full-length prefills sequentially. As Figure 1a demonstrates, these stalls can last multiple seconds.

A natural attempt to mitigate generation stalls is to reduce the maximum batch size, as suggested in Orca's original paper. However, the paper's experiments in Section 5.2 (Figure 12) show that this approach fails: running vLLM with batch sizes of 32, 64, or 128 produces nearly identical capacity under strict SLOs because the generation stalls from even a single long prefill are sufficient to violate the latency target. The larger batch sizes that PagedAttention enables cannot be leveraged in practice because the scheduler's prefill-prioritizing policy introduces latency spikes before throughput can scale.

#### Pipeline Parallelism: Bubbles Are Not Eliminated by Iteration-Level Scheduling

A key claim examined in the paper is that pipeline bubbles — a well-known problem in training — also plague inference when using pipeline parallelism, contrary to assumptions in prior work. The paper identifies (Section 3.3) three distinct types of bubbles that arise in LLM inference:

- **PB1: Varying prefill tokens across consecutive micro-batches.** Two consecutive micro-batches may contain different numbers of prefill tokens, leading to imbalanced execution times and idle time on the second pipeline stage.
- **PB2: Prefill-decode interference.** When a prefill-heavy micro-batch is followed by a decode-only micro-batch (or vice versa), the nearly order-of-magnitude difference in execution time creates a large bubble as the second stage waits for the slower first stage.
- **PB3: Variable decode attention cost.** Even within decode-only micro-batches, the attention computation cost varies by request because it depends on the accumulated KV-cache size, which grows linearly with output length. Two requests at different points in their generation have different per-token attention costs.

The paper quantifies the magnitude: for Falcon-180B with a 4K-token prompt, the prefill execution time is approximately 1150ms versus 200ms for a decode-only batch of size 32 — a potential bubble of 950ms per such transition. These bubbles are wasted GPU cycles that directly reduce throughput and increase end-to-end latency, making PP uncompetitive with in-node TP for latency-sensitive deployments on commodity networks.

#### Disaggregated Approaches: Orthogonal but with Their Own Costs

A third category of solutions has recently emerged: **disaggregating** the prefill and decode phases onto separate GPU replicas (Splitwise [58], DistServe [77], TetriInfer [47]). These systems eliminate prefill-decode interference entirely by running prefills on dedicated "prefill replicas" and decodes on dedicated "decode replicas," migrating the KV-cache between them upon prefill completion.

The paper acknowledges disaggregation as a complementary approach but identifies two practical limitations (Section 6):

1. **KV-cache migration requires high-bandwidth interconnects.** Transferring the full KV-cache state (which can be gigabytes for long contexts) from the prefill replica to the decode replica adds latency and network load. In the absence of high-bandwidth interconnects between replicas, this migration can become a bottleneck.
2. **GPU memory underutilization on prefill replicas.** Since only the decode replicas need to persist the KV-cache for the duration of output generation, the memory capacity of the prefill replicas is underutilized. This means the total GPU memory across the system is not used as efficiently as in a colocated design.

The paper positions Sarathi-Serve as a complementary approach that achieves the benefits of eliminating prefill-decode interference without the costs of disaggregation — through scheduling rather than physical separation.

### How This Paper Positions Itself

Sarathi-Serve positions itself as a **scheduling-based resolution** to the throughput-latency tradeoff that is fundamentally different from both prefill-prioritizing and decode-prioritizing approaches. Rather than choosing which phase to prioritize, it **eliminates the need to prioritize** through two mechanisms:

**Chunked-prefills** break long prefills into multiple small chunks of near-equal compute size, each processed in a separate iteration. This is based on two empirical observations: (1) a prefill with even a modest number of tokens (e.g., 512) effectively saturates GPU compute (Figure 4), so chunking does not meaningfully reduce prefill efficiency; (2) decode batches have significant slack in arithmetic intensity (Figure 5), meaning additional tokens can be processed alongside decodes without proportionally increasing iteration latency (Figure 6 shows execution time is largely stagnant in the 128–512 token range because the operation is memory-bound).

**Stall-free batching** leverages this chunking to construct hybrid batches that never delay ongoing decodes. The scheduler respects a **token budget** — a maximum total number of tokens per iteration — and fills each batch first with all ongoing decodes, then with prefill chunks from new requests until the budget is reached (Algorithm 3). By bounding the total tokens per iteration, the scheduler bounds the per-iteration latency and ensures that the TBT of ongoing decodes is unaffected by the presence of new prefill chunks. This is the critical distinction from Orca's hybrid batching: Orca combines decodes with *full* prefills (unbounded length), while Sarathi-Serve combines decodes with *chunks* of prefills (bounded length).

The paper draws a direct contrast with existing systems in Figure 2: vLLM and Orca fall on different points of a throughput-vs-TBT-latency tradeoff curve — prioritizing prefills gives throughput at the cost of latency. FasterTransformer sits at the opposite end — prioritizing decodes gives low latency at the cost of throughput. Sarathi-Serve is positioned **off this curve entirely**: it provides high throughput (through large effective batch sizes from hybrid batching) *and* low TBT latency (through stall-free scheduling that bounds per-iteration latency), resolving the tradeoff rather than navigating it.

For pipeline parallelism specifically, Sarathi-Serve's **uniform-compute batches** — where each micro-batch has approximately the same number of tokens due to the token budget constraint — eliminate the inter-batch runtime variance that causes pipeline bubbles. This makes PP viable for cross-node deployments over commodity networks, which the paper argues is essential for scaling inference on the largest models where TP within a single node is insufficient.

The paper does not claim to invent entirely new primitives — chunking and hybrid batching both have precedents — but rather shows that their **specific combination** (chunked-prefills + stall-free scheduling + token-budget-driven admission) creates a qualitatively different scheduling regime where the throughput-latency tradeoff is resolved within a single scheduler, without physical disaggregation, across a range of model sizes and parallelism strategies.

## 3. Technical Approach

### 3.1 Reader Orientation

Sarathi-Serve is an LLM inference scheduling system—a piece of software that decides which requests the GPU processes at each moment when multiple users are simultaneously asking the model to generate text. The core problem it solves is that existing schedulers force a tradeoff between serving many users efficiently (high throughput) and keeping response generation fluid for each user (low tail latency); Sarathi-Serve resolves this tradeoff by bounding how much computation happens per GPU iteration through **chunked-prefills** (splitting long input processing into small pieces) and **stall-free batching** (always including ongoing text generation in every iteration so it never gets paused by new arrivals).

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components arranged in a loop that executes once per model forward pass:

1. **Request Queue** — holds incoming requests (each with a prompt to process and a desired number of output tokens) until the scheduler admits them.
2. **Token Budget Calculator** — determines the maximum number of tokens (`$\tau$`) that can be processed in a single iteration without exceeding the target time-between-tokens (TBT) latency SLO. This is a one-time offline profiling step whose output is a scalar configuration value.
3. **Stall-Free Scheduler** (the core) — at each iteration, constructs a batch by (a) packing all currently active decode tokens, (b) adding one prefill chunk from any partially completed prompt already in the batch, and (c) admitting new requests as prefill chunks until the token budget `$\tau$` is reached. This is Algorithm 3 in the paper.
4. **Chunked-Prefill Executor** — given a prompt of potentially thousands of tokens, splits it into near-equal chunks (each a subset of the prompt tokens) and processes one chunk per iteration. Each chunk's attention operation accesses the KV-cache of all prior chunks from the same prompt.
5. **Hybrid Batch Processor** — executes a single forward pass of the model on a batch that may contain a mix of decode tokens (one per active request) and prefill tokens (from one or more chunks), producing the next output token for each decode request and updating KV-cache for prefill chunks.

Information flows cyclically: the scheduler reads the request queue → constructs a hybrid batch respecting the token budget → the GPU executes one forward pass on that batch → the scheduler removes finished requests, updates which prefills are partially complete, and repeats. New requests enter only when token budget slack exists after accommodating all ongoing work.

### 3.3 Roadmap for the Deep Dive

- **First**, the token budget and its determination (Section 4.3), since this single parameter governs the entire throughput-latency tradeoff and all other mechanisms depend on it.
- **Second**, chunked-prefills (Section 4.1)—what they are, why they're necessary, and the KV-cache access pattern that creates their overhead.
- **Third**, stall-free batching (Section 4.2, Algorithm 3)—the scheduling policy that uses chunked-prefills and the token budget to construct bounded-latency hybrid batches.
- **Fourth**, how these mechanisms eliminate pipeline bubbles (Section 3.3, 5.3) through uniform-compute micro-batches.
- **Fifth**, key implementation choices (Section 4.4) and the profiling methodology for determining the token budget in practice.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems design paper** whose core idea is that bounding the number of tokens processed per iteration—through chunked-prefills combined with a stall-free scheduling policy that prioritizes ongoing decodes—eliminates the throughput-latency tradeoff inherent in existing LLM serving schedulers.

---

#### The Token Budget: The Single Knob Controlling the Throughput-Latency Tradeoff

The token budget `$\tau$` is the maximum total number of tokens (across all requests, both prefill and decode) that Sarathi-Serve allows in a single iteration batch. This parameter is the linchpin of the entire system because **it directly bounds the per-iteration latency**: in the memory-bound regime where decode batches operate, execution time grows only marginally with additional tokens up to a device-specific threshold, after which it grows linearly (Figure 6). By keeping the total tokens per iteration at or below the point where the batch transitions from memory-bound to compute-bound, Sarathi-Serve ensures that adding prefill tokens to a decode batch incurs minimal additional latency.

**How the token budget is determined.** The paper identifies three competing factors that influence the choice of `$\tau$` (Section 4.3):

1. **TBT SLO requirement.** Smaller token budgets produce lower per-iteration latency, which translates to lower TBT. Under a strict SLO (e.g., P99 TBT of 100ms for Mistral-7B on A100), a small `$\tau$` is necessary. Under a relaxed SLO (e.g., P99 TBT of 500ms), a larger `$\tau$` allows more efficient prefill processing.

2. **Chunked-prefill overhead.** Smaller token budgets force prompts to be split into more chunks, increasing two sources of overhead: (a) lower GPU utilization because very small prefill chunks may not saturate compute, and (b) repeated KV-cache reads during attention—if a prompt is split into `$N$` chunks, the KV-cache of the first chunk is loaded from GPU HBM `$N-1$` times, the second chunk `$N-2$` times, and so on (each subsequent chunk's attention must attend to all prior chunks' keys and values). The paper quantifies this overhead in Figure 14: for Yi-34B with chunk size 512, the overhead is at most approximately 25%; with chunk size 2048, it is nearly negligible.

3. **Tile quantization effects.** GPUs compute matrix multiplications by partitioning matrices into fixed-size tiles assigned to thread blocks. When the token dimension is not divisible by the tile size, some thread blocks perform extraneous computation on padding elements—a phenomenon called tile quantization. The paper notes (Section 4.3) that "using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256" because 257 is not tile-aligned. This means `$\tau$` should be chosen to produce chunk sizes that are multiples of the GPU's tile size whenever possible.

4. **Pipeline bubble minimization** (for PP deployments). Larger chunks create higher inter-batch runtime variance, which leads to larger pipeline bubbles. Conversely, very small chunks increase fixed overheads and reduce arithmetic intensity. The token budget must balance these effects.

**Profiling methodology in practice.** Rather than analytically deriving `$\tau$`, the paper uses **Vidur** [28], an LLM inference profiler and simulator, to determine the token budget that maximizes system capacity under a specific deployment scenario (specific model, hardware, parallelism strategy, and TBT SLO). This is a one-time offline step: profile batches with varying numbers of tokens on the target hardware, measure the per-iteration latency, and select `$\tau$` as the largest number of tokens whose batch latency stays within the TBT SLO.

**Operational meaning of `$\tau$`.** The token budget is not a batch size limit per se—it is a token count limit. A batch containing 32 decode requests (32 tokens) plus a prefill chunk of 480 tokens would consume 512 tokens of the budget. The scheduler uses `$\tau$` in Algorithm 3 as a ceiling on `nt`, the accumulator variable tracking total tokens in the current batch.

**Values used in experiments.** The paper states (Section 5.1): "We use token budget of 2048 and 512 for all models under the relaxed and strict settings, respectively, except for the LLaMA2-70B relaxed configuration where we use token budget of 1536 to reduce the impact of pipeline bubbles." These values were chosen to keep per-iteration latency within the SLO thresholds in Table 3 (e.g., P99 TBT of 0.1s for Mistral-7B strict, 0.5s for relaxed).

---

#### Chunked-Prefills: Splitting Long Prompts into Bounded-Compute Units

Chunked-prefills is the mechanism that makes the token budget constraint viable even when prompts contain thousands of tokens. Without chunking, a single long prompt would necessarily exceed any reasonable token budget, forcing the scheduler to either violate the SLO (by including the full prefill and accepting high latency) or delay admitting the request until decodes finish (sacrificing throughput, as decode-prioritizing schedulers do). Chunked-prefills resolves this by decomposing a long prefill into smaller, independently schedulable units.

**What a chunk is.** A chunk is a contiguous subsequence of a prompt's input tokens. For a prompt of length `$L$` tokens split into chunks of maximum size `$C$` (where `$C \leq \tau$` is the configured chunk size), the chunks are tokens `$[0, C-1]$`, `$[C, 2C-1]$`, `$[2C, 3C-1]$`, and so on. Each chunk is processed in a separate model iteration as part of a hybrid batch.

**Why chunking is viable: prefill compute saturation at modest lengths.** The first key insight enabling chunked-prefills is that prefill throughput saturates at relatively modest token counts. Figure 3 (left panel) shows that "prefill throughput almost saturates even with a single request" — a single prompt of 1024 tokens already achieves near-peak tokens-per-second because the parallel processing of all prompt tokens efficiently utilizes GPU compute. Figure 4 breaks this down further: for Mistral-7B on A100, the prefill execution time for 512 tokens is roughly 50ms, for 1024 tokens roughly 100ms, and grows approximately linearly thereafter. This means that a chunk of 512–2048 tokens is "large enough" to achieve high GPU utilization—the marginal efficiency gain from processing all `$L$` tokens in one shot versus in chunks is small because the GPU is already compute-saturated at the chunk size.

**The KV-cache access pattern and its overhead.** The second key insight is that chunked-prefills do require additional memory reads for the attention operation, but this overhead is manageable. During the attention computation for chunk `$i$`, the model must attend to all tokens from chunks `$0$` through `$i-1$` (the causal attention mask prevents attending to future chunks). This means the KV-cache entries for chunk 0 are loaded from GPU HBM `$N-1$` times (once for each subsequent chunk's attention), chunk 1's KV-cache is loaded `$N-2$` times, and so on. The total number of KV-cache element reads across all chunks is:

$$\text{reads} = \sum_{i=0}^{N-1} \sum_{j=0}^{i} C = \sum_{i=0}^{N-1} (i+1)C = C \cdot \frac{N(N+1)}{2}$$

where `$C$` is the chunk size and `$N = \lceil L/C \rceil$` is the number of chunks.

Compared to a single full prefill, which reads each KV element exactly once during the initial computation (and then reads it again during subsequent decode iterations), chunked-prefills add extra reads during the prefill phase itself. However, the paper argues (Section 4.3) that "even at small chunk sizes attention prefill operation is compute bound" — meaning the extra memory reads do not become the bottleneck because the attention computation itself is dominated by the matrix multiplications (QK^T and softmax-weighted V), not by the KV-cache loads. The practical overhead measurement in Figure 14 confirms this: for Yi-34B with chunk size 512, the total prefill time increases by at most 25% compared to a single monolithic prefill of the same total length. For chunk size 2048, the overhead is negligible.

**How chunk size relates to token budget.** In the stall-free batching algorithm (Algorithm 3), the function `get_next_chunk_size(R, τ, nt)` computes the size of the next chunk for request R given the remaining token budget `$\tau - nt$`. If the prompt has `$L$` total tokens and `$p$` tokens have already been processed in prior chunks, the function returns `$\min(C, L - p, \tau - nt)$` — the minimum of the configured chunk size, the remaining tokens in the prompt, and the remaining token budget in the current batch. This ensures that (a) chunks never exceed the configured maximum size, (b) the last chunk of a prompt is appropriately sized (not padded), and (c) the chunk fits within the current batch's remaining budget.

**Design choice: why equal-sized chunks rather than adaptive sizing?** The paper uses near-equal-sized chunks (bounded by a fixed maximum `$C$`) rather than varying chunk sizes per batch. This is motivated by two considerations: (1) predictability—the scheduler knows each prefill chunk will consume at most `$C$` tokens, simplifying admission decisions; (2) uniform compute across iterations—equal-sized chunks contribute to the uniform micro-batch property that reduces pipeline bubbles. The paper does not explore adaptive chunk sizing based on the current batch's decode load, which could further optimize utilization but adds complexity.

---

#### Stall-Free Batching: The Scheduling Policy

Stall-free batching (Algorithm 3) is the scheduling policy that uses chunked-prefills and the token budget to construct hybrid batches that never delay ongoing decodes. It is an iteration-level scheduler—it runs at the start of each model forward pass to determine which requests participate in the next batch.

**Algorithm structure.** The scheduler maintains `B`, the set of currently active requests (those that have been admitted and have not yet finished generating). At each iteration, it initializes an empty batch and a token counter `nt = 0`. It then proceeds through three phases in strict priority order:

**Phase 1: Pack all running decodes (lines 6–8).**

```
for R in B do
    if is_prefill_complete(R) then
        nt ← nt + 1
```

Each active request whose prefill phase is complete contributes exactly 1 token to the batch (the next autoregressive decode token). The condition `is_prefill_complete(R)` is true when all chunks of R's prompt have been processed. This phase ensures that **every request currently in its decode phase is included in every iteration** — the defining property of stall-free scheduling. No decode request is ever left out of a batch while its prefill is still running; it may be joined by prefill chunks, but it is never paused.

**Phase 2: Include one partially completed prefill (lines 9–12).**

```
for R in B do
    if not is_prefill_complete(R) then
        c ← get_next_chunk_size(R, τ, nt)
        nt ← nt + c
```

If there are any requests in the batch whose prefill is still in progress (started in a previous iteration but not yet finished), the scheduler adds the next chunk of one such request. The paper's algorithm processes these sequentially through the loop, but in practice at most one partially completed prefill is advanced per iteration because the token budget is typically tight.

**Phase 3: Admit new requests as prefill chunks (lines 13–20).**

```
Rnew ← get_next_request()
while can_allocate_request(Rnew) ∧ nt < τ do
    c ← get_next_chunk_size(Rnew, τ, nt)
    if c > 0 then
        nt ← nt + c
        B ← B ∪ {Rnew}
    else
        break
```

New requests from the queue are admitted one at a time. For each candidate request, the scheduler computes how many tokens of its prefill can fit in the remaining budget `$\tau - nt$`. If at least one token fits (`c > 0`), the request is admitted with its first chunk of size `c`, and the token counter is updated. If the remaining budget is insufficient for even one prefill token, admission stops. The `can_allocate_request` check verifies that sufficient GPU memory exists for the request's KV-cache allocation.

**After batch construction (lines 22–24):** The hybrid batch is submitted to the GPU for one forward pass. After execution, `filter_finished_requests(B)` removes any requests that generated an end-of-sequence token or reached their maximum output length. The token counter `nt` is reset to 0 for the next iteration.

**Why this priority order matters.** The ordering is critical: decodes first, then in-progress prefills, then new prefills. This guarantees that:

1. **Decodes are never starved.** All active decode requests are always in the batch, so their per-iteration latency is the fixed cost of one decode forward pass plus the marginal cost of any added prefill chunks (which is bounded by `$\tau$`).
2. **Partially completed prefills make progress.** A prompt that was admitted in a previous iteration continues to be chunk-processed in subsequent iterations even if many new requests arrive, preventing prefill starvation.
3. **New requests fill residual budget.** Only after all ongoing work is accommodated does the scheduler use remaining budget for new admissions. This maximizes throughput (by starting new requests' prefills as early as possible) without compromising the latency of existing requests.

**Comparison with existing schedulers (Figure 7).** The paper provides a concrete timeline example to illustrate the difference. Requests A and B are in decode phase; requests C and D arrive. In vLLM (prefill-prioritizing, all-prefill batches), C and D's full prefills are scheduled before A and B's decodes resume, causing a multi-second generation stall for A and B. In Orca (prefill-prioritizing, hybrid batches), C and D's full prefills are combined with A and B's decodes in a single hybrid batch, but the batch runtime is dominated by the long prefills so A and B still experience a stall. In FasterTransformer (decode-prioritizing), A and B's decodes complete before C and D's prefills even begin, causing C and D to wait—low throughput. In Sarathi-Serve, C's prefill is split into two chunks; the first chunk runs alongside A and B's decodes in one iteration, the second chunk runs alongside A and B's decodes in the next iteration—A and B experience no stall because the per-iteration latency is bounded by `$\tau$`.

**The token budget as a latency bound.** The stall-free property relies on the token budget being small enough that a hybrid batch with `$\tau$` total tokens completes within the TBT SLO. Figure 9 empirically validates this: for Mistral-7B on A100 with a token budget of 256, the incremental cost of adding chunked prefill tokens to a decode batch is modest (the bars increase slowly), whereas adding a full prefill (Orca-style hybrid batching) increases latency by up to 28.3×. The paper's key engineering insight is that decode batches have significant "slack" in their execution time—because they are memory-bound, adding more compute (in the form of prefill tokens) does not proportionally increase latency until the batch transitions to compute-bound.

---

#### Eliminating Pipeline Bubbles Through Uniform-Compute Batches

The stall-free batching policy has a side effect that proves critical for pipeline-parallel deployments: it produces **near-uniform compute requirements across micro-batches**. Because every batch contains (a) all active decode tokens and (b) prefill chunks bounded by the token budget `$\tau$`, the total number of tokens per batch is roughly constant across iterations.

**Why this eliminates the three types of bubbles identified in Section 3.3:**

- **PB1 (varying prefill tokens):** Eliminated because each batch contains at most one prefill chunk per request, and the size of each chunk is bounded by the configured chunk size `$C$`. Two consecutive micro-batches have roughly the same total token count.
- **PB2 (prefill-decode interference):** Eliminated because there is no sharp prefill-only / decode-only boundary. Every batch is a hybrid batch with a controlled token count. The extreme runtime difference (1150ms for a full 4K prefill vs. 200ms for a decode batch of size 32, as measured for Falcon-180B) never occurs because full prefills are never scheduled—only bounded chunks.
- **PB3 (variable decode attention cost):** Mitigated but not fully eliminated. Even within decode requests, the attention cost grows with the accumulated KV-cache size, which increases over the course of generation. Two requests at different generation steps have different per-token attention costs. However, this variance is much smaller than the prefill-decode variance and is partially absorbed by the uniform total token count.

**Empirical validation (Section 5.3).** The paper demonstrates this on Falcon-180B deployed across two nodes (4 A100 GPUs each) with 4-way tensor parallelism within node and 2-way pipeline parallelism across nodes, connected over 100 Gbps Ethernet. Under strict SLOs, Sarathi-Serve achieves 3.6× higher capacity than vLLM with the same hybrid-parallel configuration, and 4.3× higher than vLLM with 8-way TP (Figure 13b). Under relaxed SLOs, the gain is 1.48× over vLLM's hybrid-parallel. This shows that Sarathi-Serve's uniform batches make PP viable on commodity networks—a configuration where vLLM's PP performance "drops sharply under the strict regime due to pipeline bubbles" (Section 5.3).

The paper notes that vLLM with 8-way TP actually has **lower capacity** than vLLM with hybrid parallelism under relaxed SLOs, even though TP avoids pipeline bubbles entirely. This is because cross-node TP incurs high communication overhead: Figure 13a shows that "cross node TP increases median TBT by more than 2× compared to a 4-way TP within node and PP across nodes." The all-reduce operations in TP's critical path dominate latency when GPUs are on different nodes connected by Ethernet, making TP-only deployment non-viable for latency-sensitive serving regardless of scheduling policy.

---

#### Key Implementation Details

Sarathi-Serve is implemented "on top of the open-source implementation of vLLM" (Section 4.4), inheriting its PagedAttention-based KV-cache memory management. The authors added several components:

**Attention kernel support.** The chunked-prefill mechanism requires attention kernels that can handle partial prefill sequences—computing attention for a chunk of a prompt while accessing the KV-cache of previously processed chunks from the same prompt. The paper implements this using FlashAttention v2 [38] and FlashInfer [74] kernels. "We use FlashAttention backend for all the evaluations in this paper due to its support for wider set of models" (Section 4.4). FlashAttention's tiling and recomputation strategy (which avoids materializing the full `$N \times N$` attention matrix) naturally extends to chunked computation, as each chunk's attention is computed independently with access to the accumulated KV-cache.

**Pipeline parallelism support.** The base vLLM codebase supports tensor parallelism but not pipeline parallelism. The authors extended it with "a hybrid parallel configuration with four tensor parallel workers and two pipeline stages (TP4-PP2)" for the LLaMA2-70B and Falcon-180B experiments (Section 5). Pipeline communication uses NCCL point-to-point operations. Micro-batches are constructed by the stall-free scheduler and dispatched to the first pipeline stage; they flow through subsequent stages in order.

**Telemetry system.** The implementation includes "an extensive telemetry system" for collecting latency and throughput metrics during experiments. This is used to measure the P50 and P99 values for TTFT and TBT reported in the evaluation.

**Configuration parameters and their values in experiments:**

| Parameter | Description | Strict SLO value | Relaxed SLO value |
|---|---|---|---|
| `$\tau$` (token budget) | Max tokens per iteration | 512 (all models) | 2048 (most models), 1536 (LLaMA2-70B) |
| Max batch size | Upper bound on number of requests | (implied by token budget + memory) | (implied by token budget + memory) |
| Chunk size `$C$` | Max tokens per prefill chunk | `$\leq \tau$` (determined by `get_next_chunk_size`) | `$\leq \tau$` |

The paper notes that "the system performance can be further enhanced by dynamically varying the token budget based on workload characteristics" (Section 5.1) but leaves this to future work. In the current implementation, `$\tau$` is a static configuration parameter.

**Tile quantization awareness.** The paper explicitly accounts for tile quantization effects when choosing chunk sizes (Section 4.3). For NVIDIA A100 GPUs, the tensor core tile size for FP16 matrix multiplication is 256×128. Choosing chunk sizes that are multiples of 256 (e.g., 512, 1024, 2048) avoids the extraneous computation from partially filled tiles. The 32% penalty for chunk size 257 versus 256 illustrates the magnitude of this effect. The `get_next_chunk_size` function in a production implementation might round down to the nearest tile-aligned size, though the paper does not specify whether this optimization is implemented.

**Relationship to vLLM's memory management.** Sarathi-Serve inherits vLLM's PagedAttention, which manages KV-cache in fixed-size blocks (pages) to eliminate fragmentation. When a prefill chunk is processed, its KV-cache entries are stored in newly allocated pages. Subsequent chunks of the same request read these pages during their attention computation. This is cleanly compatible with chunked-prefills because PagedAttention already supports random access to KV-cache blocks via page tables—each chunk simply needs the page table entries for all prior chunks.

---

#### Design Choices and Their Justifications

**Why chunked-prefills rather than simply limiting prefill admission?** A simpler approach would be to admit only prompts shorter than the token budget. This would bound latency but would fail on the arxiv_summarization dataset where the median prompt length is 7,059 tokens—nearly 14× the strict token budget of 512. Such prompts would be rejected or indefinitely delayed, making the system unusable for long-context applications. Chunked-prefills enables the system to serve prompts of arbitrary length while maintaining the latency bound.

**Why stall-free scheduling rather than prefill-decode disaggregation?** Disaggregated approaches (Splitwise, DistServe) eliminate prefill-decode interference by running them on separate GPU pools. The paper argues (Section 6) that this introduces two costs Sarathi-Serve avoids: (1) KV-cache migration latency and bandwidth requirements between prefill and decode replicas, which can be prohibitive on commodity networks; (2) memory underutilization on prefill replicas that don't need to persist KV-cache. Sarathi-Serve achieves the same interference elimination purely through scheduling, without additional data movement or memory imbalance. However, the paper acknowledges that disaggregated approaches can "execute prefills with maximum efficiency (and therefore yield better TTFT) unlike chunked prefills that are somewhat slower than full prefills" (Section 6), so there is a TTFT-throughput tradeoff between the two approaches.

**Why include only one in-progress prefill chunk per iteration?** The algorithm (lines 9–12) adds at most one partially completed prefill chunk before admitting new requests. This is a design choice that prevents a single long prompt from dominating the token budget across multiple iterations: if there are multiple partially completed prefills, the scheduler advances only one per iteration, ensuring that new requests can also be admitted in subsequent iterations. Without this restriction, a burst of long prompts could consume the entire token budget for many consecutive iterations, starving new admissions. The paper does not explicitly discuss this design choice, but it is implicit in the algorithm structure.

**Why the token budget is measured in tokens rather than FLOPs or time?** Tokens are a simple, model-agnostic proxy for compute. The paper's profiling methodology (using Vidur) maps token counts to wall-clock time for a specific model-hardware configuration, so the token budget `$\tau$` is set based on actual latency measurements. Using tokens rather than FLOPs avoids needing to model the complex interaction of attention (quadratic in sequence length) and linear layers (linear in batch tokens), while still capturing the dominant cost. The paper acknowledges this is approximate—the per-token cost of a decode iteration grows slightly with KV-cache size—but finds it sufficient for practical SLO adherence.

## 4. Key Insights and Innovations

### Innovation 1: The Throughput-Latency Tradeoff in LLM Serving Is a Scheduling Artifact, Not a Fundamental Constraint

The paper's most intellectually significant contribution is the reframing of the throughput-latency relationship in LLM inference. Prior to this work, the field implicitly treated the tradeoff as inherent to batched autoregressive generation: you could either prioritize prefill scheduling to maximize decode batch sizes (getting throughput at the cost of TBT latency, as in Orca and vLLM) or you could prioritize decode completion to maintain low TBT (getting predictable latency at the cost of throughput, as in FasterTransformer). The tradeoff was considered a design choice — pick your position on the curve based on application requirements.

Sarathi-Serve demonstrates that this framing is incorrect. The tradeoff is not a fundamental property of LLM inference but a **scheduling artifact** caused by the unbounded execution time of full prefill iterations. By decomposing prefills into bounded-compute chunks and guaranteeing that every iteration includes all active decodes, the system eliminates the mechanism that creates the tradeoff in the first place. The result is not a better point on the existing curve but a position **off the curve entirely**: Figure 2 places Sarathi-Serve in its own region where both high throughput and low TBT are simultaneously achievable without disaggregation.

What makes this contribution fundamental rather than incremental is that it identifies the **root cause** of the tradeoff rather than proposing a compromise. Orca and vLLM both accept the tradeoff as given and optimize within it — Orca by trying to overlap prefills and decodes in hybrid batches, vLLM by aggressively front-loading prefills to maximize subsequent decode batch size. Sarathi-Serve instead asks: why must a prefill be a monolithic operation at all? Once that question is asked, chunking emerges as the natural resolution, and the stall-free scheduling policy follows directly from the constraint that decodes must be in every batch.

The evidence for this reframing is in Figure 12: vLLM's capacity remains nearly identical whether its max batch size is set to 32, 64, or 128 under strict TBT SLOs. This demonstrates that the bottleneck is not batch size (which PagedAttention elegantly enables) but the scheduling policy that introduces generation stalls regardless of how many requests are in flight. Sarathi-Serve's capacity under the same strict SLO is 3.5× higher for Mistral-7B — not because it batches better, but because it eliminates the stalls that prevent vLLM from using the batch sizes it can already construct.

### Innovation 2: Generation Stall as a Diagnostic Concept That Unifies Prior Negative Results

The paper introduces **generation stall** as a precise diagnostic concept: a measurable cause of tail latency where consecutive decode iterations of an ongoing request are separated by one or more prefill iterations whose execution time dominates the TBT. This concept provides explanatory power beyond what existed before.

Prior work documented the symptoms — high tail latency under load (Figure 1b), poor performance under strict SLOs — but did not isolate generation stalls as the specific mechanism. By naming and measuring this phenomenon, the paper does three things that are individually significant:

**First, it explains why reducing batch size does not help under strict SLOs.** A natural remediation for tail latency in prefill-prioritizing systems would be to limit the batch size, as suggested in Orca [75]. Figure 12 shows this fails because "even a single long prefill" can create a generation stall that violates a strict SLO. The stall duration is dominated by the prefill length, not the number of concurrent requests. Reducing batch size from 128 to 32 does not meaningfully reduce the worst-case stall because the worst case is still a single 7K-token prompt arriving during decodes. This is a genuinely non-obvious finding: the intuition that "smaller batches = lower latency" holds for compute-bound operations but fails when latency spikes are caused by scheduling policy rather than batch compute time.

**Second, it reconciles the conflicting design choices of Orca and vLLM.** Both are prefill-prioritizing iteration-level schedulers, but Orca permits hybrid batches while vLLM does not. A superficial reading might conclude that Orca should have lower generation stall severity since it overlaps prefills and decodes. The paper shows (Figure 9) that hybrid batching without chunking actually exacerbates the problem: including a full 4K prefill in a decode batch increases TBT by up to 28.3× compared to a decode-only batch, creating a generation stall within a single iteration. The generation stall concept thus explains why vLLM's "all-prefill then all-decode" approach and Orca's hybrid approach both fail under latency constraints: they differ in mechanics but share the root cause of unbounded per-iteration prefill compute.

**Third, it establishes generation stalls as the mechanism behind pipeline bubbles in inference.** Prior work (e.g., FasterTransformer, FastServe) used micro-batches for pipeline parallelism without reporting pipeline bubbles, and Orca [75] claimed iteration-level scheduling eliminates them. The paper's bubble taxonomy (PB1, PB2, PB3 in Section 3.3) connects generation stalls to pipeline inefficiency: the varying execution times that cause stalls within a single GPU also cause bubbles across pipeline stages. This unifies two previously separate problems under a single diagnostic framework, enabling a single solution (uniform-compute batches via token budgeting) to address both.

### Innovation 3: Memory-Bound Slack as an Exploitable Resource for Zero-Cost Prefill Overlap

A central technical insight underpinning Sarathi-Serve is that decode batches operate in a **memory-bound regime with significant execution time slack** that can be exploited to process additional tokens with minimal latency impact. This is not merely an empirical observation but a deliberate exploitation of the GPU's compute-memory architecture for scheduling advantage.

Prior work recognized that decodes have low arithmetic intensity — the literature is clear that a single-token forward pass is bottlenecked on weight fetching rather than compute. But this property was treated as a *problem* to be mitigated (through batching to amortize weight loads) rather than an *opportunity* to be exploited. The paper's insight is that memory-boundedness means the execution time curve is flat over a significant range of token counts: Figure 6 shows that for LLaMA2-70B on A100, the linear layer execution time remains "largely stagnant in the 128–512 tokens range" because the dominant cost is loading weights from HBM, which happens once regardless of how many tokens are processed. Only when token count passes a critical threshold (around 500–600 tokens for this configuration) does the operation transition to compute-bound and the runtime begins scaling linearly.

This flat region is effectively **free compute**: additional prefill tokens can be processed alongside decodes for marginal additional latency. Sarathi-Serve's token budget is set to keep batches in or near this flat region, meaning the prefill computation is piggybacked onto decode iterations at near-zero incremental cost to TBT. Figure 9 validates this empirically: the "Decode + Chunked Prefill" bars grow much more slowly than the "Decode + Full Prefill" bars as sequence length increases.

What distinguishes this from conventional batching is the **asymmetry of exploitation**. Conventional batching amortizes weight loads across multiple requests doing the same operation (all decoding). Sarathi-Serve amortizes across **different operations** — the expensive weight load for decodes also serves the chunked prefills in the same batch, even though those prefills would individually be compute-bound. The decode tokens provide the memory-bound "base load" that makes the incremental prefill compute nearly free. This is a qualitatively different use of batching than what prior systems employed.

This insight also explains why Sarathi-Serve can achieve high throughput without disaggregation: it recovers the prefill throughput that disaggregated systems would achieve on dedicated prefill replicas, but does so by utilizing the otherwise-wasted memory bandwidth of decode iterations rather than requiring separate GPU resources.

### Innovation 4: The Token Budget as a Single Unified Control Knob Spanning Latency, Throughput, and Pipeline Efficiency 

Rather than providing separate mechanisms for controlling throughput (e.g., max batch size), latency (e.g., prefill admission throttling), and pipeline efficiency (e.g., micro-batch sizing), Sarathi-Serve collapses all three concerns into a single parameter: the token budget `τ`. This is a genuinely novel abstraction — not because such parameters are unprecedented individually, but because a single scalar simultaneously governs:

- **TBT tail latency**: by bounding per-iteration token count and therefore per-iteration execution time (Section 4.3).
- **System throughput**: by controlling how aggressively new requests are admitted (larger budgets admit more prefill tokens per iteration, increasing decode batch sizes sooner).
- **Pipeline bubble magnitude**: by creating uniform-compute micro-batches that eliminate inter-stage waiting (Section 3.3).
- **Chunking overhead**: by setting the maximum chunk size, which trades off GPU utilization against KV-cache re-read cost (Section 5.4.1).

The paper demonstrates that this single parameter, when set appropriately via one-time profiling, is sufficient to navigate the entire throughput-latency design space. Figure 12 shows that varying `τ` (between 512 for strict SLO and 2048 for relaxed SLO) produces the full range of system behaviors — low latency with moderate throughput at small `τ`, high throughput with relaxed latency at large `τ` — without requiring changes to the scheduling algorithm, admission policy, or memory management. This is a significant simplification over prior systems that required operators to tune multiple interacting parameters (batch size limits, prefill admission rates, memory allocation thresholds) with unclear joint effects.

The conceptual contribution is that `τ` is not an arbitrary knob but the **natural unit of scheduling** for hybrid-batch LLM inference. Because both prefill and decode compute scale primarily with token count (attention is quadratic but linear operations dominate, as Figure 4 shows), tokens serve as a workable common currency for bounding latency. This may seem obvious in hindsight, but prior systems measured work in sequences (FasterTransformer), requests (Orca), or memory pages (vLLM's PagedAttention) — none of which directly correspond to execution time. The paper's shift to token-based scheduling is a conceptual advance that enables the unified treatment of prefill and decode within a single budget constraint.

## 5. Experimental Analysis

### Evaluation Methodology

**Dataset.** The paper uses two datasets with distinct characteristics to emulate real-world serving scenarios (Table 2). **openchat_sharegpt4** [68] contains user-shared conversations with ChatGPT-4, with median prompt length of 1,730 tokens, P90 of 5,696 tokens, and median output length of 415 tokens. Its multi-round conversation structure produces high variance in prompt lengths. **arxiv_summarization** [36] is a collection of scientific publications and their abstracts from arXiv.org, with median prompt length of 7,059 tokens, P90 of 12,985 tokens, and median output length of 208 tokens. This dataset represents long-document summarization workloads analogous to Microsoft M365 Copilot and Google Duet AI. Request arrival times are generated using a Poisson distribution. Outliers are filtered by removing requests with total length exceeding 8,192 tokens for openchat_sharegpt4 and 16,384 tokens for arxiv_summarization.

**Base model(s).** The paper evaluates across four models spanning a wide range of scales (Table 1): **Mistral-7B** [51] (7B parameters, GQA with sliding window attention), **Yi-34B** [24] (34B parameters, GQA), **LLaMA2-70B** [66] (70B parameters, GQA), and **Falcon-180B** [31] (180B parameters, GQA). These represent the "best in their model size categories" at the time of publication. The models use Grouped Query Attention (GQA), which reduces KV-cache size by 8× compared to multi-head attention, making larger batch sizes feasible.

**Metrics.** Three primary metrics are used. **TBT (Time-Between-Tokens)** measures the interval between consecutive output token generations for a single request; the paper reports the 99th percentile (P99) to capture tail latency behavior. **TTFT (Time-To-First-Token)** measures the latency from request arrival to generation of the first output token; the paper reports the median (P50). **Capacity** (queries-per-second) is the maximum sustainable request load a system can serve while meeting specified latency targets, with the constraint that median scheduling delay does not exceed 2 seconds (ensuring queue stability). The paper defines two SLO configurations per model (Table 3): "strict" (P99 TBT set to 5× the execution time of a decode iteration with 4K prefill length and batch size 32, running without prefill interference) and "relaxed" (25× that same baseline). For Mistral-7B, this translates to P99 TBT of 0.1s (strict) and 0.5s (relaxed); for Yi-34B, 0.2s and 1.0s; for the 70B and 180B models, 1.0s and 5.0s.

**Baselines.** The paper compares against two state-of-the-art systems. **vLLM** [53] represents prefill-prioritizing iteration-level batching with PagedAttention-based memory management. It uses FCFS admission and schedules all-prefill or all-decode batches without mixing. vLLM is evaluated with max batch sizes of 32, 64, and 128 to test whether batch size tuning can mitigate the throughput-latency tradeoff. **Orca** [75] represents the first iteration-level batching system with support for hybrid prefill-decode batches; it also uses prefill-prioritizing FCFS scheduling but mixes prefill and decode tokens within a single batch. Note that Orca lacks PagedAttention and therefore supports smaller maximum batch sizes than vLLM due to memory fragmentation and large activation footprints for token-heavy batches.

**Generation budget / compute accounting.** The primary unit of scheduling is the **token budget** `τ` — the maximum total number of tokens (prefill + decode) allowed in a single iteration batch. This is not a direct compute FLOPs metric but a proxy validated through profiling: the paper uses Vidur [28], an LLM inference profiler, to determine the largest `τ` whose batch execution time stays within the target SLO for a given model-hardware configuration. This ensures that comparisons across systems are normalized by a shared latency target rather than by raw FLOPs. Sarathi-Serve uses `τ` = 512 (strict SLO) and `τ` = 2048 (relaxed SLO) for most configurations, with LLaMA2-70B relaxed using `τ` = 1536 to reduce pipeline bubbles. The max batch size for Sarathi-Serve is set to 128. For vLLM and Orca, the number of requests per batch is the budget; Sarathi-Serve operates in token space.

**Cross-validation / statistical protocol.** No formal cross-validation or statistical testing is reported. All experiments are conducted on fixed hardware configurations with the full datasets. The paper controls for queue stability by enforcing a 2-second limit on median scheduling delay to ensure that reported capacity values represent sustainable rather than transient throughput. The capacity evaluation methodology uses an increasing load sweep to find the maximum queries-per-second at which latency SLOs are met without queue blowup. The throughput-latency tradeoff analysis (Section 5.2, Figure 12) evaluates five SLO values to map the full tradeoff curve rather than a single operating point.

### Main Quantitative Results

#### Capacity Under Strict and Relaxed Latency SLOs (Section 5.1)

The central finding is that Sarathi-Serve consistently and substantially outperforms both vLLM and Orca across all four models, both datasets, and both SLO configurations. Under **strict SLOs** with the openchat_sharegpt4 dataset (Figure 10a), Sarathi-Serve achieves:

- **Mistral-7B (single A100):** 2.78× higher capacity than Orca, 2.15× higher than vLLM
- **Yi-34B (2 A100s, TP2):** 4.00× higher capacity than Orca, 3.7× higher than vLLM (the paper states "up to 3.7×" for vLLM in the abstract and introduction)

Under **relaxed SLOs** with the same dataset, gains narrow but remain significant:
- **Mistral-7B:** 2.44× higher than Orca (vLLM comparison not explicitly stated in Figure 10a text but the bar chart shows a visible gap)
- **Yi-34B:** substantial gains visible in the figure, though exact multiples are not separately reported in the text for the relaxed case

For the arxiv_summarization dataset (Figure 10b), with its much longer prompts (median 7,059 tokens), the gains are smaller but still consistent:
- **Mistral-7B:** 1.82× (strict), 1.97× (relaxed) over Orca; 1.69× (strict), 1.94× (relaxed) over vLLM
- **Yi-34B:** comparable relative improvements visible in the bar chart

The paper notes that "Orca and vLLM violate the P99 TBT latency SLO before they can reach their maximum serviceable throughput" (Section 5.1). In other words, their maximum sustainable capacity is **latency-limited**, not throughput-limited — they could process more requests if latency constraints were ignored, but generation stalls push TBT above the SLO threshold before the system saturates. Sarathi-Serve's capacity is also latency-limited, but the latency ceiling is higher because the token budget mechanism prevents generation stalls from occurring.

For larger models with pipeline parallelism (Figure 11), the gains are amplified due to Sarathi-Serve's elimination of pipeline bubbles:

- **LLaMA2-70B (8 A40s, TP4-PP2), openchat_sharegpt4, strict SLO:** 6.31× over Orca, 5.54× over vLLM
- **LLaMA2-70B, arxiv_summarization, strict SLO:** 4.60× over Orca, 4.20× over vLLM (from the text in Figure 11 caption)
- **Falcon-180B (8 A100s across 2 nodes, TP4-PP2), openchat_sharegpt4, strict SLO:** 5.62× over Orca, 4.69× over vLLM. Under relaxed SLO: 5.54× and 6.31× respectively. The arxiv_summarization dataset shows 3.00× (relaxed) and 2.75× (relaxed) gains.

A critical detail: vLLM **significantly outperforms Orca under relaxed SLOs** for most configurations (visible in Figures 10 and 11). The paper explains this with two factors: (1) "Orca batches prompts for multiple requests together (max sequence length * batch size compared to max sequence length in vLLM), which can lead to even higher tail latency in some cases" — Orca's hybrid batching without chunking creates very large token counts per iteration; (2) vLLM's PagedAttention enables much larger maximum batch sizes than Orca, whose unfragmented KV-cache management limits concurrency. This means that among the baselines, vLLM is the stronger comparison point for throughput.

#### Throughput-Latency Tradeoff Analysis (Section 5.2, Figure 12)

This experiment systematically varies the P99 TBT SLO and measures the resulting capacity for vLLM and Sarathi-Serve, producing the tradeoff curves that the paper's Figure 2 schematically depicts.

**vLLM's capacity is bottlenecked by generation stalls, not batch size.** Figure 12 shows vLLM evaluated with max batch sizes of 32, 64, and 128. Under strict SLOs (P99 TBT of 100ms for Mistral-7B), all three batch size configurations yield **nearly identical capacity** — approximately 0.15–0.2 queries per second. This is the paper's key evidence that reducing batch size does not mitigate the throughput-latency tradeoff: even with only 32 concurrent requests, the generation stall from a single long prefill (e.g., a 7K-token prompt from arxiv_summarization) is sufficient to violate a 100ms SLO. The paper states: "The maximum capacity of vLLM gets capped due to generation stalls under stringent TBT SLOs... the capacity of vLLM remains largely identical for all the three batch size settings." At relaxed SLOs (500ms for Mistral-7B), capacity does scale with batch size — vLLM-128 achieves roughly 2.5 queries per second, vLLM-32 about 1.5 — because the SLO is loose enough to absorb the generation stalls from moderate-length prompts.

**Sarathi-Serve's tradeoff is controllable via the token budget.** Two Sarathi-Serve configurations are shown: SS-512 (token budget 512) and SS-2048 (token budget 2048), both with max batch size 128. For Mistral-7B at 100ms SLO (the strictest point), SS-512 achieves approximately 0.7 queries per second — roughly 3.5× higher than any vLLM configuration. For Yi-34B at 200ms SLO, SS-512 achieves approximately 0.55 queries per second versus roughly 0.15 for vLLM — a 3.5× gain. At relaxed SLOs (500ms for Mistral-7B, 1s for Yi-34B), SS-2048 achieves approximately 1.65× higher capacity than vLLM.

The paper draws attention to an important asymmetry: "Sarathi-Serve achieves 3.5× higher capacity compared to vLLM under strict SLO (100ms, Mistral-7B) using a small token budget of 512. For scenarios with more relaxed SLO constraints, picking a larger token budget of 2048 allows Sarathi-Serve to operate more efficiently resulting in 1.65× higher capacity compared to vLLM (1s, Yi-34B)." The relative gain is highest under strict SLOs because vLLM's generation stalls become proportionally more damaging — a 2-second stall consumes 20× the 100ms SLO budget but only 2× the 1s SLO budget.

#### Pipeline Parallel Viability for Cross-Node Deployment (Section 5.3, Figure 13)

This experiment evaluates whether Sarathi-Serve's uniform-compute batches make pipeline parallelism viable for latency-sensitive serving over commodity networks, compared to the standard approach of tensor parallelism within a node.

**Tensor parallelism across nodes incurs prohibitive latency overhead.** Figure 13a shows decode-only batch latency for Falcon-180B on two configurations: 8-way TP (spanning two nodes, communicating over 100 Gbps Ethernet) and TP4-PP2 (4-way TP within each node, 2-way PP across nodes). The median TBT for cross-node TP is "more than 2×" that of the hybrid configuration. For example, at batch size 32, TP8 takes approximately 280ms versus roughly 140ms for TP4-PP2. This is because TP's all-reduce operations are on the critical path and must synchronize across the 100 Gbps Ethernet link, while PP's point-to-point communication transfers only activations between stages.

**vLLM's pipeline-parallel performance collapses under strict SLOs.** Figure 13b shows capacity for three configurations: vLLM with TP8, vLLM with TP4-PP2, and Sarathi-Serve with TP4-PP2, on the openchat_sharegpt4 dataset. Under relaxed SLO, vLLM PP achieves the highest capacity of the three — roughly 0.8–0.9 queries per second — while vLLM TP8 achieves only about 0.4 due to communication overhead. Sarathi-Serve PP achieves roughly 1.2 queries per second under relaxed SLO, a 1.48× gain over vLLM PP. Under **strict SLO**, the picture changes dramatically: vLLM TP8 fails to reach even 0.1 queries per second, vLLM PP reaches approximately 0.15, and Sarathi-Serve PP reaches approximately 0.55 — a 3.6× gain over vLLM PP. The paper attributes vLLM PP's sharp drop to pipeline bubbles: "vLLM can support a fairly high load with hybrid parallelism under relaxed SLO, [but] its performance drops sharply under the strict regime due to pipeline bubbles." Sarathi-Serve's uniform batches avoid these bubbles, making PP viable even under tight latency constraints.

### Ablation Studies and Robustness Checks

**Chunked-prefills overhead as a function of chunk size and prompt length (Section 5.4.1, Figure 14).** The paper measures the total prefill execution time for Yi-34B (TP2) when a prompt of a given total length is split into chunks of sizes 512, 1024, and 2048, normalized against the monolithic (no-chunk) baseline. For chunk size 2048, the overhead is nearly negligible across all prompt lengths (2K, 4K, 8K). For chunk size 1024, overhead ranges from approximately 5–15%. For chunk size 512, overhead reaches approximately 25% in the worst case. The paper attributes this overhead primarily to repeated KV-cache reads during attention ("the first chunk's KV-cache is loaded N−1 times, the second chunk's KV-cache is loaded N−2 times, and so on"). A second-order observation: the overhead is highest for long prompts at small chunk sizes, consistent with the O(N²) growth in KV-cache re-reads. The paper concludes that this overhead is "moderate" and acceptable given the latency benefits.

**Isolated effect of hybrid-batching vs. chunked-prefills vs. combined (Section 5.4.2, Table 4).** This ablation evaluates three configurations on Yi-34B (2 A100s, token budget 1024, 128 requests): hybrid-batching-only (mixing full prefills with decodes, analogous to Orca's policy but with Sarathi-Serve's implementation), chunked-prefills-only (splitting prefills but never mixing prefill and decode tokens in the same batch), and Sarathi-Serve (both techniques combined). The results in Table 4 show:

| Configuration | openchat P50 TTFT (s) | openchat P99 TBT (s) | arxiv P50 TTFT (s) | arxiv P99 TBT (s) |
|---|---|---|---|---|
| hybrid-batching-only | 0.53 | 0.68 | 3.78 | 1.38 |
| chunked-prefills-only | 1.04 | 0.17 | 5.38 | 0.20 |
| Sarathi-Serve (combined) | 0.76 | 0.14 | 3.90 | 0.17 |

The key insight: **hybrid-batching-only achieves good TTFT but terrible TBT** (0.68s for openchat, 1.38s for arxiv) because full prefills in hybrid batches create generation stalls. **Chunked-prefills-only achieves good TBT but terrible TTFT** (1.04s for openchat, 5.38s for arxiv) because prefill chunks are processed in isolation without decode overlap, slightly inefficiently due to chunking overhead, and new requests queue behind existing work. **Sarathi-Serve's combination achieves the best of both**: P99 TBT of 0.14–0.17s (comparable to chunked-prefills-only) and P50 TTFT of 0.76–3.90s (better than chunked-prefills-only, though worse than hybrid-batching-only). The paper states: "the two techniques work best together: chunked-prefills-only increases TTFT as prefill chunks are slightly inefficient whereas hybrid-batching-only increases TBT because long prefills can still create generation stalls."

**Effect of token budget on Sarathi-Serve's tradeoff position (Figure 12, implicitly).** While not framed as an ablation, the comparison of SS-512 versus SS-2048 in Figure 12 demonstrates that the token budget is the effective control knob for the throughput-latency tradeoff. SS-512 (smaller budget, more chunking, lower per-iteration latency) achieves 3.5× higher capacity than vLLM under strict SLOs but saturates at lower absolute throughput. SS-2048 (larger budget, less chunking, higher per-iteration latency) achieves higher absolute throughput at relaxed SLOs but cannot meet strict SLOs. This confirms that a single parameter governs the tradeoff and that no single "best" value exists independent of the target SLO.

**PP vs. TP tradeoff with and without Sarathi-Serve (Figure 13).** The comparison of vLLM TP8, vLLM TP4-PP2, and Sarathi-Serve TP4-PP2 demonstrates that pipeline parallelism without Sarathi-Serve is not competitive under strict SLOs (vLLM PP drops to ~0.15 qps, comparable to vLLM TP8 at ~0.08 qps), while Sarathi-Serve PP maintains competitive capacity (~0.55 qps). This validates the claim that pipeline bubbles, not pipeline communication overhead, are the primary barrier to PP adoption for latency-sensitive inference.

### Critical Assessment

#### Claim 1: Sarathi-Serve improves serving capacity by up to 3.7× for Yi-34B and 2.6× for Mistral-7B compared to vLLM.

The experiments substantially support these headline numbers, but the gains are **conditional on the SLO regime and dataset**. The 3.7× figure for Yi-34B is achieved under strict SLO with the openchat_sharegpt4 dataset (Figure 10a). Under relaxed SLO, this number narrows — the bar chart in Figure 10a shows the relative gap shrinking, and the paper's text does not claim a specific multiple for the relaxed case. For the arxiv_summarization dataset with its much longer prompts, the gains drop to 1.69–1.94× for Mistral-7B and comparable levels for Yi-34B. This is not a weakness of the system — longer prompts inherently benefit less from chunking because the overhead of repeated KV-cache reads grows with the number of chunks — but it means the 3.7× figure should be understood as a **best-case** under conditions that maximize the damage of generation stalls (many moderate-length prompts from a chat dataset) and minimize Sarathi-Serve's overheads.

A more subtle limitation: the capacity metric depends on the specific SLO thresholds chosen. Table 3 defines strict SLO as 5× the decode iteration time at batch size 32 with 4K prefill. These values are somewhat arbitrary — a different multiplier would produce different capacity ratios. The paper's Figure 12 partially addresses this by sweeping SLO values, but the 3.7× and 2.6× figures are point estimates on that sweep.

#### Claim 2: Sarathi-Serve eliminates generation stalls through stall-free scheduling.

This claim is **qualitatively demonstrated** by Figure 1a (generation stalls in vLLM exceeding several seconds, absent in Sarathi-Serve) and **quantitatively supported** by the TBT latency measurements. Sarathi-Serve's P99 TBT under load (Table 4, Figure 12) remains well-controlled at 0.14–0.17s for Yi-34B and corresponding values for other models, while vLLM's P99 TBT degrades sharply with load (Figure 1b). The mechanism — stall-free batching — is clearly articulated in Algorithm 3 and the paper's Figure 7 timeline comparison. However, the paper does not provide a direct head-to-head measurement of TBT distribution under identical load for vLLM versus Sarathi-Serve at the per-request level. Figure 1a shows a single representative timeline; a CDF of per-request P99 TBT across both systems under identical load would more rigorously demonstrate stall elimination.

#### Claim 3: Sarathi-Serve reduces pipeline bubbles, enabling up to 5.6× gains for Falcon-180B with pipeline parallelism.

This claim has the **strongest quantitative support** among the paper's headline results (Figure 13b shows 5.6× over Orca and 4.3× over vLLM for Falcon-180B under strict SLO), but the experimental design has several limitations:

**The comparison is against vLLM's PP implementation, which the authors themselves built.** The paper states (Section 4.4): "We also extend the base vLLM codebase to support... pipeline parallelism." While vLLM's open-source release has since added PP support, at the time of this paper's experiments, the authors implemented PP on top of vLLM. This means the baseline PP performance reflects the authors' own implementation quality. An independent PP implementation (e.g., from FasterTransformer or DeepSpeed-Inference) might perform differently, for better or worse.

**The bubble quantification is indirect.** The paper identifies three bubble types (PB1, PB2, PB3) and provides an example magnitude (950ms for a 4K prefill vs. 200ms decode transition on Falcon-180B), but does not directly measure bubble time as a fraction of total GPU cycles. The capacity improvement is attributed to bubble reduction, but this is an inference from the mechanism rather than a measurement. A direct measurement would profile GPU utilization across pipeline stages and attribute idle cycles to specific bubble types, which the paper does not do.

**The TP8 baseline has a fundamental disadvantage that is not Sarathi-Serve's contribution.** Figure 13a shows that TP8 across nodes has 2× higher decode latency than TP4-PP2 due to cross-node all-reduce overhead. Sarathi-Serve's advantage over TP8 is therefore partly due to choosing a better parallelism strategy (PP instead of TP) independently of the scheduling improvements. The paper's 4.3× figure (Sarathi-Serve PP vs. vLLM TP8) conflates parallelism choice with scheduling quality. The fair comparison for isolating Sarathi-Serve's pipeline bubble reduction is vLLM PP vs. Sarathi-Serve PP, which shows a 3.6× gain under strict SLO (Figure 13b). This is still substantial, but narrower.

#### Claim 4: The throughput-latency tradeoff is a scheduling artifact, not a fundamental constraint.

This is the paper's most ambitious claim and the evidence is strong but **incomplete**. Figure 12 convincingly demonstrates that Sarathi-Serve occupies a region of the throughput-latency space that vLLM cannot reach regardless of batch size tuning. However, the claim that the tradeoff is "not fundamental" requires showing that Sarathi-Serve's approach does not introduce new tradeoffs of its own:

**Chunked-prefills create a TTFT vs. TBT tradeoff that the paper acknowledges but does not fully characterize.** Table 4 shows that hybrid-batching-only achieves P50 TTFT of 0.53s (openchat) while Sarathi-Serve achieves 0.76s — a 43% increase. This is because chunked prefills take multiple iterations to complete the prefill phase, delaying the first output token. For applications where TTFT matters (e.g., showing the first word of a response quickly), this is a real cost. The paper frames this as acceptable because TTFT is "obtained only once per user request" while TBT affects every token, but the tradeoff exists and is not fully explored across different prompt lengths or SLO compositions.

**The token budget must be configured per model, hardware, and SLO.** This is not a "knob-free" system — operators must profile and set `τ` appropriately. The paper provides guidance (use Vidur for profiling) and shows that two values (512 and 2048) cover the strict/relaxed SLO range, but the sensitivity to `τ` is not ablated. What happens if `τ` is set 10% too high? Does TBT SLO violation occur gradually or catastrophically? The paper does not explore robustness to token budget misconfiguration.

#### Genuine Weaknesses and Missing Experiments

**Single hardware generation and small GPU count per experiment.** All experiments use NVIDIA A100 (80GB) or A40 (48GB) GPUs. There is no evaluation on H100 GPUs (which have different memory bandwidth and tensor core characteristics), on AMD GPUs, or on inference-specific accelerators. For models like Mistral-7B, only a single A100 is used; for Yi-34B, only 2 A100s. The paper does not explore how Sarathi-Serve's benefits scale with the degree of parallelism (e.g., TP8 within a single DGX node versus TP2).

**No comparison with disaggregated serving systems.** Section 6 explicitly acknowledges disaggregated approaches (Splitwise, DistServe, TetriInfer) as an alternative solution to the prefill-decode interference problem, but no quantitative comparison is provided. The paper argues that disaggregation has different costs (KV-cache migration, memory imbalance) but does not measure whether those costs are larger or smaller than Sarathi-Serve's chunking overhead. This is a significant omission because disaggregated systems can "execute prefills with maximum efficiency (and therefore yield better TTFT)" — whether the TTFT advantage outweighs the migration cost is an empirical question the paper does not answer.

**No ablation on the chunk size selection strategy.** The paper uses fixed chunk sizes (implied by the token budget `τ`) and notes tile quantization effects, but does not test whether adaptive chunk sizing (e.g., larger chunks when decode load is low, smaller when it is high) would improve throughput-latency Pareto efficiency. The `get_next_chunk_size` function in Algorithm 3 uses `min(chunk_size, remaining_prompt, remaining_budget)`, but the interaction of these three constraints is not explored.

**No multi-tenant fairness or priority scheduling evaluation.** The paper assumes FCFS admission at the request level. Real deployments often have priority classes (premium vs. free users), request age considerations, or fairness requirements. The interaction of stall-free batching with priority-based admission is not explored, and the paper acknowledges this by citing complementary work on fairness [62].

**No evaluation of TTFT under SLO constraints.** The capacity metric (Section 5.1) is defined by P99 TBT SLOs only. Table 4 shows that Sarathi-Serve increases P50 TTFT by 43% compared to hybrid-batching-only for the openchat dataset, but the capacity experiments do not measure whether TTFT itself violates any latency target as load increases. An application with a TTFT SLO (e.g., "first token within 500ms") might find Sarathi-Serve's chunking-induced prefill delay problematic even if TBT remains excellent.

**No end-to-end response latency measurement.** The paper focuses on TBT (per-token pacing) and TTFT (first token delay) separately, but does not report the total time from request arrival to final token generation. An application that cares about complete response latency (e.g., "answer within 5 seconds") might find that Sarathi-Serve's chunking delays the prefill completion in a way that shifts the entire response later, even if TBT is smooth. This metric is not reported.

## 6. Limitations and Trade-offs

### 6.1 Token Budget Selection Is Deployment-Specific and Requires One-Time Profiling

**The assumption or constraint.** The entire stall-free scheduling mechanism depends on a single scalar: the token budget `τ`, which governs per-iteration latency and therefore the tradeoff between throughput and TBT. The paper states that selecting `τ` is "a complex decision which depends on the desired TBT SLO, parallelism configuration, and specific hardware properties" (Section 4.3). The recommended approach is one-time profiling using Vidur [28], an LLM inference profiler, to determine the largest `τ` whose batch execution time stays within the target SLO.

**The consequence.** This means Sarathi-Serve is not a "knob-free" system — an operator must profile their specific model, hardware configuration, parallelism strategy, and latency target before deployment to select `τ`. The paper only evaluates two `τ` values (512 for strict SLO, 2048 for relaxed SLO, with a special case of 1536 for LLaMA2-70B relaxed to reduce pipeline bubbles — Section 5.1). The sensitivity of system performance to `τ` misconfiguration is **not characterized**: if an operator sets `τ` 10% or 20% too high (optimistic about available latency budget), does TBT SLO violation occur gradually or catastrophically? If `τ` is set conservatively too low, how much throughput is unnecessarily sacrificed? Without a robustness analysis, the operator faces an unpleasant choice between expensive per-model profiling and risky guesswork.

Additionally, the paper notes tile quantization effects (Section 4.3) — "using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256" — meaning `τ` must be chosen not just from a latency perspective but also to respect GPU tensor core tile boundaries (multiples of 256 for A100 FP16). This adds another dimension to the profiling burden that is specific to each GPU architecture.

**What evidence exists in the paper.** Figure 12 directly demonstrates that `τ` controls the throughput-latency tradeoff (SS-512 vs. SS-2048 produce different capacity-vs-SLO curves), but there is **no experiment** that varies `τ` in finer increments (e.g., 256, 384, 512, 768, 1024) to measure how sharply capacity degrades when `τ` is suboptimally chosen. The paper acknowledges this gap implicitly: "the system performance can be further enhanced by dynamically varying the token budget based on workload characteristics. We leave this exploration for future work" (Section 5.1).

**Mitigation status.** The paper provides profiling guidance (use Vidur) but no automated method for selecting `τ` at runtime, no adaptive scheme that adjusts `τ` based on observed latency, and no sensitivity analysis. The "future work" statement indicates the authors are aware of this limitation. A dynamic approach that starts with a conservative `τ` and gradually increases it while monitoring P99 TBT would partially address this, but is not implemented or evaluated.

---

### 6.2 Chunked-Prefills Increase Time-to-First-Token for Long Prompts

**The assumption or constraint.** Sarathi-Serve's chunked-prefills mechanism trades off TTFT against TBT by design: rather than processing a prompt's full prefill in a single iteration (as Orca and vLLM do), it spreads the prefill across multiple iterations. For a prompt of length `L` tokens and chunk size `C`, the first output token is produced only after `⌈L/C⌉` prefill iterations, each of which also includes decode work from other requests. The paper states this explicitly: "chunked-prefills-only increases TTFT as prefill chunks are slightly inefficient" (Section 5.4.2).

**The consequence.** For applications where TTFT matters — e.g., showing the first word of a chatbot response quickly to convey responsiveness — Sarathi-Serve introduces a systematic delay. Table 4 quantifies this: on the openchat_sharegpt4 dataset (median prompt 1,730 tokens), Sarathi-Serve achieves P50 TTFT of 0.76s, compared to 0.53s for hybrid-batching-only (full prefills with decode overlap, analogous to Orca) — a 43% increase. On the arxiv_summarization dataset (median prompt 7,059 tokens), the TTFT comparison is 3.78s (hybrid-batching-only) vs. 3.90s (Sarathi-Serve), a smaller relative gap but still a regression. The paper frames this as acceptable because "TTFT is obtained only once per user request" while "TBT affects every token" (Section 2.4), but this tradeoff may be unacceptable for applications with strict first-token SLOs (e.g., voice assistants where the user expects an immediate response, or streaming applications where initial latency shapes user perception of overall quality).

The disaggregated serving systems that Sarathi-Serve is positioned against (Splitwise [58], DistServe [77]) explicitly do not face this tradeoff: they execute prefills with full efficiency on dedicated prefill replicas. The paper acknowledges this in Section 6: "disaggregated approaches can execute prefills with maximum efficiency (and therefore yield better TTFT) unlike chunked prefills that are somewhat slower than full prefills." The implication is that Sarathi-Serve sacrifices TTFT relative to disaggregation-based approaches, trading it for the elimination of KV-cache migration costs and memory imbalance.

**What evidence exists in the paper.** Table 4 directly measures the TTFT impact. However, the **capacity experiments (Section 5.1) do not enforce any TTFT SLO** — they define capacity solely by P99 TBT constraints with a 2-second median scheduling delay limit. It is possible that under high load, Sarathi-Serve's TTFT degrades further as chunked prefills from multiple long prompts compete for token budget slots, but this is not measured. The paper does not report whether TTFT itself is SLO-compliant at the reported capacity values.

**Mitigation status.** Not addressed. The paper presents TTFT as a secondary metric and does not propose mechanisms to bound or optimize it (e.g., prioritizing the first chunk of a prompt, using larger chunks for the initial prefill iteration, or allowing TTFT-aware token budget selection). The "future work" on dynamic token budgets (Section 5.1) could potentially mitigate this by using larger `τ` when decode load is low, but this is not explored.

---

### 6.3 The Approach Has Not Been Tested on Modern Hardware Generations or Non-NVIDIA GPUs

**The assumption or constraint.** All experiments in the paper use NVIDIA A100 (80GB) or A40 (48GB) GPUs (Table 1). The paper was published in mid-2024, at which point NVIDIA H100 GPUs were already widely deployed in production serving infrastructure. H100 GPUs have substantially different memory bandwidth (3.35 TB/s vs. 2.0 TB/s for A100) and tensor core characteristics, which directly affect the memory-bound vs. compute-bound transition that Sarathi-Serve's token budget exploits.

**The consequence.** The core insight enabling Sarathi-Serve is that decode batches operate in the memory-bound regime with significant "slack" — execution time is "largely stagnant in the 128–512 tokens range" (Section 3.1, Figure 6) because the dominant cost is loading model weights from HBM, which happens once per batch regardless of how many tokens are processed. On H100 GPUs with 1.67× higher memory bandwidth, the memory-bound region is likely narrower — the transition to compute-bound occurs at a lower token count, meaning the "free compute" slack that Sarathi-Serve exploits is reduced. This could directly shrink the effectiveness of chunked-prefill piggybacking, requiring smaller token budgets and more aggressive chunking with correspondingly higher overhead.

Similarly, inference-specific accelerators (Google TPUs, AWS Inferentia, custom ASICs) have different memory hierarchies and compute characteristics that may not exhibit the same memory-bound flat region. The tile quantization effects that the paper identifies as important (32% penalty for chunk size 257 vs. 256 — Section 4.3) are architecture-specific; different GPU generations have different tile sizes, and non-NVIDIA hardware may have entirely different quantization behavior.

No evaluation is performed on AMD GPUs, which are increasingly used for inference in cloud deployments (e.g., Microsoft's use of AMD MI300X for Azure OpenAI service workloads).

**What evidence exists in the paper.** None. The paper evaluates only A100 and A40 GPUs. There is no discussion of how the token budget, chunk size selection, or pipeline bubble reduction would change on H100, A100-80GB-SXM vs. PCIe (which have different bandwidth), or non-NVIDIA hardware.

**Mitigation status.** Not addressed. The paper treats the profiling-based `τ` selection as covering hardware differences ("the token budget... depends on... specific hardware properties" — Section 4.3), but this assumes the same architectural patterns (memory-bound decode slack, tile quantization) hold across GPU generations, which is not verified.

---

### 6.4 Single Workload Domain: All Evaluations Use Summarization and Conversation Datasets with Exact Token Matching

**The assumption or constraint.** The paper evaluates on two datasets: openchat_sharegpt4 (multi-turn conversations) and arxiv_summarization (scientific paper summarization). Both are **text generation tasks** where the correctness metric is implicit (the system generates tokens and latency is measured; output quality is not evaluated). The paper does not test workloads with fundamentally different decode characteristics: speculative decoding (where multiple tokens are generated per iteration), code generation (which often has very long output sequences with structured syntax), embedding extraction (no decode phase at all), or retrieval-augmented generation where prompts are dynamically assembled from retrieved documents.

The paper's entire throughput-latency analysis depends on the assumption that decode phases are memory-bound with a single token per request per iteration, that prefill phases are compute-bound with large token counts, and that the per-token attention cost grows with KV-cache size. These assumptions do not hold for all LLM serving workloads:

- **Speculative decoding** generates multiple candidate tokens per iteration, changing the arithmetic intensity of decode batches — they may become compute-bound at lower batch sizes, reducing the slack Sarathi-Serve exploits.
- **Code generation** often produces hundreds or thousands of output tokens per request, meaning KV-cache sizes grow substantially during generation, changing the decode attention cost profile (PB3 bubble type becomes more significant).
- **Embedding models** have a prefill phase but no decode phase at all — the stall-free scheduling benefit is irrelevant because there are no decodes to stall.
- **RAG workloads** have highly variable prefill lengths depending on retrieved context, potentially creating worst-case prompt lengths far exceeding the arxiv_summarization P90 of 12,985 tokens.

**What evidence exists in the paper.** The paper evaluates only two text datasets with the characteristics in Table 2. The models tested (Mistral-7B, Yi-34B, LLaMA2-70B, Falcon-180B) are all standard autoregressive decoder-only transformers using GQA. There is **no evaluation** of encoder-decoder models (T5, BART), mixture-of-experts models (Mixtral), or models with non-standard attention patterns (e.g., sliding window only, linear attention). The paper does not discuss how Sarathi-Serve would interact with continuous batching of heterogeneous request types (e.g., mixing embedding extraction with text generation in the same deployment).

**Mitigation status.** Not addressed. The paper's claims are implicitly scoped to the evaluated models and datasets, but the title and abstract present Sarathi-Serve as a general LLM inference scheduler. The absence of workload diversity testing means a practitioner cannot confidently predict performance on their specific use case without replicating the profiling and capacity experiments themselves.

---

### 6.5 No Comparison Against Disaggregated Serving Systems That Address the Same Tradeoff

**The assumption or constraint.** The paper explicitly acknowledges disaggregated serving as an alternative approach to resolving the prefill-decode interference problem: "These solutions can entirely eliminate the interference between prefills and decodes" (Section 6, discussing Splitwise [58], DistServe [77], and TetriInfer [47]). Sarathi-Serve is positioned as an alternative that achieves similar benefits without KV-cache migration costs or memory imbalance. However, the paper provides **no quantitative comparison** against any disaggregated system.

**The consequence.** Without a head-to-head comparison, the paper cannot substantiate its claim that Sarathi-Serve's scheduling-based approach is preferable to disaggregation. The arguments made are qualitative:
- Disaggregation "requires migrating the KV cache of each request upon the completion of its prefill phase which could be challenging in the absence of high-bandwidth interconnects" (Section 6)
- Disaggregation "under-utilizes the GPU memory capacity of the prefill replicas i.e., only the decode replicas are responsible for storing the KV cache" (Section 6)

But these costs are not quantified against Sarathi-Serve's own costs: chunked-prefill overhead (up to ~25% prefill time increase at chunk size 512 — Figure 14), TTFT degradation (Section 5.4.2, Table 4), and the need for per-deployment token budget profiling (Section 4.3). Whether Sarathi-Serve's chunking overhead is smaller or larger than disaggregation's KV-cache migration overhead depends on prompt length distributions, available network bandwidth, and GPU memory capacity — all of which are deployment-specific and none of which are measured.

Furthermore, the disaggregated systems cited (Splitwise, DistServe) were published contemporaneously with or slightly after Sarathi-Serve (arXiv preprints in late 2023 / early 2024), so a direct comparison at publication time may not have been feasible. However, the paper's Section 6 frames disaggregation as a known alternative and provides qualitative arguments against it, which creates an expectation of quantitative support that is not met.

**What evidence exists in the paper.** None. There is no implementation of or comparison against Splitwise, DistServe, TetriInfer, or any other disaggregated serving system. The paper states: "We leave a quantitative comparison between Sarathi-Serve and disaggregation-based solutions for future work" (Section 6).

**Mitigation status.** Explicitly deferred to future work. This is the most significant "missing comparison" in the paper because disaggregated serving systems target exactly the same problem (prefill-decode interference causing the throughput-latency tradeoff) and represent the primary alternative architectural approach. A practitioner choosing between deploying Sarathi-Serve and deploying a disaggregated system receives no guidance from this paper on which performs better under what conditions.

---

### 6.6 The Scheduler Operates Under FCFS Admission Without Priority or Fairness Mechanisms

**The assumption or constraint.** Sarathi-Serve's stall-free batching algorithm (Algorithm 3) admits new requests in FCFS (first-come-first-served) order: `get_next_request()` returns the next request from the head of the queue, and requests are admitted as long as the token budget and GPU memory permit. There is no mechanism for request prioritization (premium vs. free users), deadline-aware scheduling (requests that must complete within a time bound), or fairness across clients (preventing a single user's many requests from starving others). The paper acknowledges this implicitly by citing complementary work on fairness [62] in the related work section.

**The consequence.** In multi-tenant deployments, FCFS admission with token-budget-constrained batching can create pathological behaviors:

- **Long-prompt starvation of short requests.** A burst of long prompts (e.g., 8K tokens, as in the arxiv P90) will consume the token budget for many consecutive iterations as their chunks are processed, delaying the admission of any new requests — including short, latency-sensitive ones — until the long prompts' prefills are complete or the queue drains. Unlike vLLM, which eagerly admits all available requests and processes their full prefills (creating generation stalls but admitting everyone quickly), Sarathi-Serve's chunk-based admission serializes long-prompt processing at the token budget granularity, potentially increasing scheduling delays for requests behind long prompts in the queue.
- **No latency differentiation.** A request from a paying customer with a 100ms TBT SLO and a request from a free-tier user with a 1s TBT SLO would be treated identically under FCFS with a single `τ`, even though the latter could tolerate larger token budgets and therefore more efficient prefill processing.
- **No preemption or requeuing.** If a request's generation is taking too long (exceeding some deadline), there is no mechanism to deprioritize or terminate it to free resources. The only exit condition is natural completion (end-of-sequence token or max output length reached).

**What evidence exists in the paper.** The paper does not evaluate any priority, fairness, or deadline-aware scheduling policies. Figure 12 shows capacity as a function of P99 TBT SLO, but this is a system-wide metric — it does not measure whether individual requests (or classes of requests) receive differentiated service. The paper notes that "Sarathi-Serve first calculates the budget of maximum number of tokens that can be executed in a batch based on user specified SLO" (Section 4.2), implying a single SLO for all requests.

**Mitigation status.** Not addressed. The paper cites Sheng et al. [62] as complementary work on fairness in LLM serving and notes that "such algorithmic optimizations are complementary to our approach and can benefit from lower prefill-decode interference enabled by Sarathi-Serve" (Section 6). This correctly identifies that fairness and stall-free batching are orthogonal concerns, but does not implement or evaluate any integration. A priority-aware variant of Algorithm 3 could use per-request token budgets (admit high-priority requests with smaller chunks for lower latency, low-priority with larger chunks for efficiency) or weighted fair queueing, but this is left entirely to future work.

## 7. Implications and Future Directions
- What changes in practice
  - Sarathi-Serve offers a practical recipe to deliver both high throughput and stable interactive latency for LLM serving—especially important for chat and streaming applications. It also makes PP viable across commodity networks by reducing bubbles, enabling larger models to be served efficiently without high-degree TP or exotic interconnects (Figure 13).

- What it enables next
  - Dynamic token-budget control: online adaptation of τ based on observed latencies, load, and PP imbalance could further improve capacity and robustness beyond the fixed τ values used here (§5.1, §4.3).
  - Integration with fairness/preemption: combining stall-free batching with preemptive schedulers or fairness-aware admission could provide stronger QoS in multi-tenant clouds (§6).
  - Hybrid with disaggregated designs: selectively using disaggregated prefills for ultra-long prompts or TTFT-critical users, while keeping stall-free batching for the bulk of traffic, could yield the best of both worlds (noted in §6).
  - Compiler/runtime co-design: accounting for tile quantization and kernel launch overheads when choosing chunk sizes suggests opportunities for joint tuning in kernel libraries and schedulers (§4.3).

- Broader takeaway
  - The paper reframes the “throughput vs. latency” conflict in LLM serving as a batching and compute-shaping problem. By matching iteration compute to an SLO-driven envelope (token budget) and exploiting arithmetic-intensity slack in decode, it achieves a materially better Pareto frontier. The empirical results across four models and diverse deployments (Figures 10–13) suggest this approach is widely applicable.
