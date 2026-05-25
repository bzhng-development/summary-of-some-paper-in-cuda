# MARCONI: PREFIX CACHING FOR THE ERA OF HYBRID LLMS

**ArXiv:** [2411.19379](https://arxiv.org/abs/2411.19379)

## 🎯 Pitch

Marconi introduces the first practical prefix caching system tailored for Hybrid LLMs, which combine Attention and recurrent SSM layers. By devising novel admission and FLOP-aware eviction policies that accommodate the in-place, 'all-or-nothing' state management of SSMs, Marconi dramatically increases cache utility—achieving up to 34.4× higher token hit rates and up to 71.1% lower time-to-first-token compared to state-of-the-art systems. This innovation removes a key bottleneck for efficiently serving long-context Hybrid LLM workloads, unlocking scalable, fast, and memory-efficient inference for next-generation language models.

---

## 1. Executive Summary

This paper introduces **Marconi**, the first prefix caching system designed for Hybrid LLMs—models that interleave efficient recurrent layers (State Space Models) with full Attention layers—analyzing its performance across LMSys, ShareGPT, and SWE-Bench workloads on a 7B Hybrid model and Jamba-1.5-Mini. Marconi replaces traditional recency-based caching with two named mechanisms: a **judicious admission policy** that caches SSM states only for high-reuse-likelihood prefixes (identifying common input prefixes via radix tree speculative insertion and conversation-continuation points at the last decoded token) and a **FLOP-aware eviction policy** that scores cache entries by compute savings normalized against memory footprint (measured as FLOP efficiency, favoring longer sequences whose SSM states deliver more compute savings per byte). Across diverse workloads, Marconi achieves up to 34.4× higher token hit rates and 71.1% lower P95 TTFT (617 ms reduction) compared to state-of-the-art prefix caching systems extended to Hybrid models, establishing that prefix caching remains viable for Hybrid LLMs only when admission and eviction account for the "all or nothing" reusability of in-place-updated SSM states.

## 2. Context and Motivation

### The Core Problem: Hybrid LLMs Break Prefix Caching

The fundamental problem this paper tackles is that **Hybrid LLMs—models combining Attention layers with recurrent State Space Model (SSM) layers—are incompatible with existing prefix caching systems**. Prefix caching is a critical efficiency optimization in LLM serving: it stores the internal model states (KVs for Attention layers, recurrent states for SSM layers) computed during the prefill phase, so that when a subsequent request shares a common prefix (e.g., the same system prompt or conversation history), those prefixes don't need to be recomputed. This directly reduces Time To First Token (TTFT) latency and increases throughput, with no accuracy penalty since the reuse is exact.

The incompatibility arises from a specific technical property of SSM layers that the paper identifies: **SSM states are updated in-place and cannot be rolled back** to represent intermediate prefixes of a sequence. This creates a dilemma that the paper frames as the central tension in Sections 1 and 3:

> "On one hand, maximizing reuse opportunities for future (arbitrary) workloads mandates caching fine-grained state checkpoints at regular intervals, e.g., every 256 tokens. On the other hand, increased checkpointing frequency inflates the number of cache entries generated per sequence, each of which is large (due to the sheer size of SSM states) but most of which present limited reuse opportunities, i.e., sparsely-hit. The net effect is cache thrashing with low-utility entries"

To understand why this is specifically a Hybrid LLM problem, we need to contrast how Transformers and SSMs manage their internal states during inference.

**In Transformers,** the Attention mechanism produces Key-Value (KV) caches for each token position. These KVs have a sequence dimension—the KV cache for the entire sequence of length $q$ is a concatenation of per-token KV pairs. Critically, if we want to reuse only a prefix of length $p$ (where $p < q$), we can simply **slice the KV tensors** along the sequence dimension to keep only the first $p$ positions. The tensor has the form `[batch, seq_len, num_heads, head_dim]`, and slicing `[:p]` along the `seq_len` dimension gives exactly what's needed. This property makes prefix caching in Transformers straightforward: you can cache the full sequence's KVs and later extract the prefix for partial matches.

**In SSMs,** the situation is fundamentally different. SSMs (exemplified by Mamba) maintain a **recurrent hidden state** that encodes all preceding tokens into a fixed-size representation. At each token position, the SSM reads the current token and the previous hidden state, then overwrites the hidden state in place to produce the next representation. The key phrase in the paper is "updated in place" (Section 3, SSM State Property 2):

> "SSM states are updated in place, so a sequence's states cannot be rolled back to represent its prefixes."

The hidden state after processing tokens $1...q$ is a fixed-size matrix with no sequence dimension—it's just $[D \times N]$ where $D$ is the model dimension and $N$ is the SSM state dimension. If a future request only needs to reuse tokens $1...p$, the state representing $1...q$ is **useless** because it has absorbed information from tokens $p+1...q$. You cannot "subtract" the contribution of those later tokens. To support arbitrary prefix matching, you would need to have **separately checkpointed** the SSM state at token $p$ before overwriting it with later computations.

**The "all or nothing" reusability.** This leads to what the paper calls the "all or nothing" nature of SSM cache entries (Section 3). A cached SSM state for a sequence of length $q$ is only reusable when the next request **exactly matches all $q$ tokens**. A partial match (tokens $1...p$ only) provides zero reuse benefit from that SSM state. To enable partial matches, you must cache checkpointed SSM states at regular intervals—but each checkpointed state is a full-sized SSM hidden state (unlike KVs where each token's contribution is additive and proportional to sequence length).

The paper quantifies this size disparity in SSM State Property 3 (Section 3):

> "SSM states are orders of magnitude larger than the KVs of a single token."

From Appendix A, for an SSM layer: state size = $2DN$ bytes (in FP16), where $N$ is the state dimension. For the 7B Hybrid model used in the paper, $D=4096$ and $N=128$, giving **~1 MB per SSM layer state**. In contrast, the KVs for a single token in an Attention layer are $4LD$ bytes where $L$ is sequence length—for one token, this is $4 \times 1 \times 4096$ = **~32 KB per layer** (key + value, FP16). The SSM state is roughly 32× larger than a single token's KVs. But this SSM state represents the **entire** sequence, not incremental per-token information. When checkpointed at fine granularity, many of these large SSM states are generated but few are ever reused.

---

### Why This Problem Is Important

The paper's motivation rests on converging trends that make this incompatibility increasingly urgent:

**1. Hybrid models are being widely adopted.** The paper cites numerous productionized Hybrid models (Section 2.1): Jamba 1.5 at 398B parameters (Team et al., 2024), Zamba (Glorioso et al., 2024), Samba (Ren et al., 2024), Cartesia's Rene model, Zyphra's Zamba2 series, and others. These models are gaining traction because they offer the efficiency benefits of SSMs (linear compute scaling with sequence length, constant memory footprint) while preserving the in-context learning and recall capabilities of Attention layers. As the paper notes:

> "These models blend quadratic Attention and subquadratic SSM layers, typically interleaved in a specific ratio (commonly 1 Attention layer for every 6-10 SSM layers)"

The trend in model development is clearly toward more SSM layers relative to Attention, as seen in the paper's microbenchmarks (Figure 12a). This means the caching problem described here will only become more acute.

**2. Context lengths are growing dramatically.** The paper documents the push toward longer contexts driven by multi-step reasoning (Khattab et al., 2023; Yao et al., 2022), detailed prompt templates (Liu et al., 2023), and increased few-shot examples. Commercial models like Gemini 1.5 and Claude 3 have reached 1M token context windows (Section 1). Longer sequences magnify both problems described above: they increase the total KVs that need caching for Attention layers (making prefix reuse savings larger), and they increase the number of SSM checkpoints that would be needed for fine-grained coverage (exacerbating memory pressure).

**3. SSM state dimensions are increasing.** The paper observes a clear trend in SSM architecture development: state dimensions are growing for better language modeling capability. The microbenchmark in Figure 12b shows Marconi's advantage over baselines growing from 5.7× to 35.4× as the SSM state dimension increases from 16 (Mamba1) to 128 (Mamba2). Larger state dimensions mean larger cached entries, which means the "sparsely-hit" problem becomes more severe—more memory wasted on states that are never reused.

**4. The efficiency stakes are high in production serving.** The paper is explicit that prefix caching is not an academic optimization—it's a critical production technique. TTFT is a user-facing latency metric for interactive applications (chatbots, coding assistants, AI-powered search), and prefix caching directly reduces it. Similarly, FLOP savings from caching translate to higher throughput and lower cost per query. Systems like vLLM (Kwon et al., 2023), SGLang (Zheng et al., 2023b), and production deployments at Character.AI and Anthropic all rely on prefix caching. If Hybrid models break this optimization, they could lose the efficiency advantages that justify their architectural complexity in the first place.

---

### Where Existing Approaches Fall Short

The paper systematically identifies limitations in how prior prefix caching systems handle Hybrid models. Importantly, there are **no existing systems that natively support prefix caching for Hybrid/SSM models**, so the paper's baselines are constructive extensions of state-of-the-art Transformer caching systems.

**Fine-grained checkpointing in vLLM+ produces cache thrashing.** The natural adaptation of vLLM's block-based prefix caching to Hybrid models is to checkpoint SSM states at the same block granularity used for KVs (e.g., every 32 tokens). The paper calls this approach "fine-grained checkpointing." The problem is visible in Figure 3:

- **Figure 3a** shows that with a block size of 32, 25.0% of KVs are reused by future requests, but only 0.4% of SSM states are reused—a 65.3× difference. Even at block size 128 (beyond what vLLM natively supports), SSM state reuse reaches only 3.3%.
- **Figure 3b** shows the memory consequence: a single 7B Hybrid model sequence of 10K tokens consumes 17.4 GB of cached states, 3.3× larger than a Transformer of equivalent size. The paper explicitly quantifies this: "for a 7B model, a single sequence of 10K tokens consumes 17.4 GB, 3.3× bigger than a Transformer of the same size."

The net effect is that fine-grained checkpointing **floods the cache with large, low-utility SSM states**, evicting entries that would have been more useful. The paper describes this as cache thrashing: the cache is constantly being overwritten with new states that provide minimal reuse while displacing potentially valuable ones.

**SGLang's radix tree approach doesn't address SSM state sparsity.** SGLang (Zheng et al., 2023b) uses a radix tree data structure to efficiently represent overlapping prefixes across sequences, reducing memory fragmentation for KVs. However, the paper argues this doesn't solve the fundamental problem:

> "Different from SGLang, Marconi uses the past requests in the radix tree to determine which SSM states to cache by performing speculative insertions before prefilling an upcoming sequence. Further, Marconi's philosophy of judicious state admission fundamentally differs from SGLang, which admits the states of all tokens during admission and is no different from other prefix caching systems like vLLM"

In other words, SGLang's radix tree improves **how** states are organized and looked up, but not **which** states are cached. It still admits all tokens' states, which for Hybrid models means admitting massive numbers of low-utility SSM states.

**Recency-based eviction (LRU) ignores compute-vs-memory tradeoffs.** All prior prefix caching systems use LRU (Least Recently Used) or variants as their eviction policy. The paper argues this is adequate for Transformers because the KV cache's memory footprint grows proportionally with the compute savings from reuse—longer sequences take more memory but also save more compute, so LRU naturally balances the tradeoff. But for Hybrid models, this relationship breaks:

> "whereas the size of KVs for a sequence is linearly proportional to the sequence length and (approximately) the compute savings from reusing that sequence, SSM state sizes are fixed regardless of sequence length and compute savings."

A short sequence with minimal compute savings takes the same SSM state memory as a long sequence with massive compute savings. Under LRU, these two entries compete purely on recency—there's no mechanism to favor the more "FLOP-efficient" entry. This is quantified in Figure 5, which shows FLOP efficiency (compute saved per byte of cache) for different model architectures: SSM-heavy models show much steeper FLOP efficiency curves as sequence length increases, meaning the penalty for evicting a long-sequence entry is much larger than for evicting a short-sequence one.

**No existing work accounts for the interaction between SSM states and KVs in eviction.** The paper's key insight is that in Hybrid models, cache entries contain both Attention KVs and SSM states, each with fundamentally different memory-vs-compute tradeoffs. Prior systems treat all cached states uniformly for eviction purposes. The paper argues this is insufficient:

> "Traditional prefix caching systems designed for Transformers don't need to consider FLOP efficiency because KVs' FLOP efficiency is near-constant, and most systems only use recency for eviction. However, ignoring it in Hybrid LLM inference risks evicting states with higher FLOP efficiency but do not have the best recency."

---

### How This Paper Positions Itself

Marconi's positioning is clear and specific: it is **the first prefix caching system for Hybrid LLMs**, and its novelty lies not in inventing new data structures but in **redesigning the caching policies (admission and eviction) to account for SSM states' "all or nothing" reusability and fixed-size, high-FLOP-efficiency properties**.

The paper explicitly frames this as a policy-level contribution:

> "Our system doesn't invent new data structures for KV cache management, and instead builds on the rich prior work in this space that also aims to manage prefixes' model states efficiently. Our main contribution lies in being the first system to redesign caching policies and their interactions with these data structures to practically address the unique properties of emerging Hybrid models."

This is an important distinction. The radix tree structure Marconi uses is adapted from prior work (SGLang), but the **decisions about what to cache and what to evict** are entirely novel. The admission policy introduces a taxonomy-based framework for predicting which SSM states merit caching—it's not just "cache everything" or "cache at fixed intervals." The eviction policy introduces a two-factor scoring function (recency + FLOP efficiency) that has no precedent in LLM serving systems.

The paper also positions itself within a broader trend toward Hybrid models that it believes is inexorable. The microbenchmarks in Section 5.4 are explicitly designed to show that Marconi's advantages grow with model characteristics that align with "recent model developments"—higher SSM layer ratios, larger state dimensions, longer sequences. This forward-looking positioning argues that Marconi isn't just solving today's problem but is **architecturally aligned with where the field is heading**.

Finally, the paper positions its contribution as making prefix caching **viable for Hybrid models at all**, not just making it marginally better. The baseline comparison in Figure 7 shows vLLM+ achieving token hit rates of roughly 6-12% (depending on workload), while Marconi achieves 25-42%. The paper frames this not as an incremental improvement but as the difference between prefix caching being functionally useless (single-digit hit rates) and being genuinely beneficial. The abstract's claim of "up to 34.4× higher token hit rates" is specifically measured against these extended baselines, establishing that without Marconi-style policies, prefix caching for Hybrid models largely fails.

The two mechanisms—judicious admission and FLOP-aware eviction—are presented as complementary solutions to two halves of the same problem. Admission addresses **which entries enter the cache** (preventing the deluge of low-utility SSM states), while eviction addresses **which entries stay in the cache** (ensuring that entries with high compute-saving potential aren't evicted in favor of short, recency-hot but FLOP-inefficient sequences). Together, they address the full lifecycle of a cache entry in a way that is specifically tailored to Hybrid model properties.

## 3. Technical Approach

### 3.1 Reader Orientation

This paper presents **Marconi**, a prefix caching system that decides which internal model states to keep and which to discard so that serving Hybrid LLMs—models that mix efficient recurrent (SSM) layers with Transformer (Attention) layers—can skip redundant computation across requests. The system solves the problem that SSM states are large, updated in-place, and cannot be partially reused, by carefully choosing *what* to cache (admission) and *what to evict* (eviction) based on predictions of which prefixes will actually get reused and how much compute those prefixes save relative to their memory cost.

### 3.2 Big-Picture Architecture (Diagram in Words)

The system has five major components:

1. **Radix Tree Index** — a space-optimized prefix tree that records which token sequences have been seen and maps each sequence (or prefix) to its cached model states. Each edge stores the KVs for its tokens and the SSM state representing all tokens prior to the edge’s last token. Nodes represent branch points where different request paths diverge.

2. **Judicious Admission Policy** — a decision procedure run before prefill that determines *which* SSM states to checkpoint. It does this by speculatively inserting the upcoming request’s input tokens into the radix tree to detect whether a new branch point (common prefix) is being created, and always checkpoints the SSM state after the last decoded token for conversation-continuation scenarios.

3. **SSM State Checkpointing Mechanism** — the low-level capability that actually captures an SSM state at a specific token position during prefill. For models with chunked state passing (e.g., Mamba-2), it snapshots the state at the second-to-last chunk boundary. For models without this (e.g., Mamba-1), it performs a two-pass prefill to materialize the exact state.

4. **FLOP-Aware Eviction Policy** — a scoring function that assigns a utility value to each radix tree node based on both how recently it was accessed (recency) and how much compute-per-byte it saves (FLOP efficiency). When the cache is full, nodes with the lowest utility scores are evicted.

5. **Bootstrap Parameter Tuner** — a retrospective grid search over possible values of the balancing coefficient α that trades off recency against FLOP efficiency. It replays a window of recent requests and selects the α that maximizes hit rate for the current workload.

Information flows as follows: a request arrives → the admission policy speculatively inserts its input tokens into the radix tree to identify whether new branch-point SSM states are needed → prefill proceeds, checkpointing SSM states at identified positions using the appropriate mechanism (chunked or two-pass) → KVs for all tokens are stored along tree edges → after decoding, the SSM state at the last decoded token is admitted → if the cache is full, the FLOP-aware eviction policy computes utility scores for all nodes and removes the lowest-utility ones → the bootstrap tuner periodically adjusts α based on observed workload patterns.

### 3.3 Roadmap for the Deep Dive

- **First**, the radix tree structure and the relationship between edges, nodes, and stored states—since every admission and eviction decision operates on this structure.
- **Second**, the judicious admission policy and its underlying taxonomy of prefix reuse scenarios—since this determines *which* SSM states enter the cache in the first place.
- **Third**, the specific checkpointing mechanisms for SSM states during prefill—since the admission policy's decisions must be physically realized via one of two hardware-dependent approaches.
- **Fourth**, the FLOP efficiency metric and how it is computed across heterogeneous layers—since this is the foundation for the eviction policy's scoring.
- **Fifth**, the FLOP-aware eviction policy and its utility score function—since this determines *which* states get removed under cache pressure.
- **Sixth**, the bootstrap α-tuning procedure—since the balance between recency and FLOP efficiency is workload-dependent and must be configured online.

### 3.4 Detailed, Sentence-Based Technical Breakdown

This is primarily a **systems paper** whose core idea is that prefix caching for Hybrid LLMs requires fundamentally different admission and eviction policies than those used for Transformers, because SSM states are large, fixed-size, in-place-updated, and cannot be partially reused.

---

#### The Radix Tree Structure

Marconi uses a radix tree (also called a compressed prefix tree) to organize all previously seen token sequences and map them to their cached model states. Understanding this structure is essential because every Marconi mechanism—admission, eviction, and cache lookup—operates on this tree.

A radix tree is a variant of a trie where edges are labeled by sequences of varying lengths rather than single characters. In Marconi's usage, each edge represents a span of consecutive tokens, and each node represents a point in the sequence space where requests diverge (branch) or converge (merge, though merges are not explicitly stored since the tree is rooted). Figure 4 illustrates this with a concrete example: the edge from the root to the first node might be labeled with the tokens "NYC is", and a branch might split into "a busy city" (one child edge) or "very huge" (another child edge) from the same parent node.

**What is stored per edge.** Each edge in the radix tree holds two types of model state:

1. **KV caches for the edge's tokens.** For every Attention layer in the Hybrid model, the edge stores the key and value tensors corresponding to the tokens that the edge represents. These are standard KV caches with a sequence dimension proportional to the number of tokens in the edge.

2. **SSM states for all tokens preceding the edge.** For every SSM layer, the edge stores the recurrent hidden state that represents *all tokens from the beginning of the sequence up to and including the last token in the edge*. In the "NYC is a busy city" example, the edge for "city" would store the SSM state after processing "NYC is a busy city"—that is, the cumulative state representing all five tokens.

This design choice—associating SSM states with edges rather than nodes—is deliberate. By storing the SSM state at the end of each edge (representing all prior context), Marconi can retrieve the correct prefix state without walking the tree: the state at the edge's endpoint is exactly what the next request needs if it continues from that point.

**Nodes represent branch points.** A node with multiple children represents a prefix that has been followed by different continuations across different requests—this is the "purely input" reuse scenario (e.g., a shared system prompt that different users append different questions to). A node with zero or one children represents a point that hasn't (yet) demonstrated branching reuse. The paper notes that nodes with ≤1 child may represent "input and output" prefixes of future requests (Section 4.1)—points where conversations will be appended to rather than branched from.

**Cache lookups are path traversals.** When a new request arrives, Marconi walks the radix tree following the longest matching path of input tokens. The deepest node reached on this traversal identifies the longest cached prefix. The SSM states and KVs for this prefix are retrieved from the edges along the path and used as the starting point for prefill computation, skipping the tokens already covered.

**Implementation detail: edges subsume parent states.** A critical structural property: when Marconi stores KVs for an edge, those KVs are a suffix of the full sequence. But to reuse the prefix, we need *all* preceding KVs as well. In a standard radix tree, you would need to walk from the root to collect the KVs of all ancestor edges. However, Marconi's design notes that during eviction (Section 4.3, implementation detail 1), when an intermediate node with a single child is evicted, its SSM states are released but its KVs are **absorbed by the child node** (concatenated). This means leaf-accessible KVs are always complete without requiring ancestor traversal for Attention layers. For SSM layers, only the last edge's state is needed (since it already represents all preceding tokens), so no ancestor traversal is needed either.

---

#### Judicious Admission Policy: The Taxonomy of Reuse Scenarios

Marconi's admission policy is the mechanism that decides *which* SSM states to checkpoint and insert into the radix tree. The driving insight is that, despite not knowing future requests, the reuse potential of each SSM state can be estimated by classifying potential reuse into two categories and handling each differently.

**The taxonomy.** Through analysis of real-world traces (Section 4.1), the authors identify two fundamentally different ways that prefixes get reused:

1. **Purely input prefixes.** The shared prefix is entirely part of a previous request's input tokens and does not include any model-generated output. Examples include system prompts (the same instructions prepended to many user queries), few-shot examples, and self-consistency prompting where the same context is used to generate multiple completions. These prefixes are typically shared across *many* distinct requests, often from different users or sessions.

2. **Input-and-output prefixes.** The shared prefix includes both previous input tokens and the model's generated output tokens. The canonical example is multi-turn conversation history: each new request in a conversation appends the user's latest message to the full history of previous exchanges (including the model's previous responses). LLM agent trajectories follow the same pattern: each step appends a new observation and action to the history of past interactions.

The key difference between these two categories is how they manifest in the radix tree. Purely input prefixes create **branch points**—the same prefix is followed by different continuations from different requests, resulting in nodes with multiple children. Input-and-output prefixes create **linear chains**—each new turn appends tokens to the end of the previous sequence, resulting in nodes with at most one child.

**Admission for purely input prefixes: speculative insertion.** Marconi identifies common input prefixes by detecting when a new request will create a branch point in the radix tree. The procedure (Section 4.1, Figure 4):

1. Before prefilling a new request, Marconi performs a **speculative insertion** of the request's input tokens into the radix tree. This is a hypothetical walk through the tree that identifies where the new sequence overlaps with previously seen sequences and where it diverges.

2. If the speculative insertion would create a **new intermediate node** (a branch point that didn't previously exist—meaning a prefix that was previously only seen as a linear path is now being branched from), then this prefix qualifies as a "purely input" pattern. Multiple requests are now sharing this prefix with different continuations.

3. When such a branch point is detected, Marconi checkpoints the SSM state at the branching position during the subsequent prefill. The SSM state at this position represents the common prefix and can be reused by all future requests that share it.

4. If no new intermediate node would be created (the new request either follows an existing path or extends a linear chain without branching), no extra SSM state is checkpointed for the prefill segment.

The crucial subtlety here: **a purely input prefix is only identified on its second occurrence**. The first time a sequence is seen, it creates a linear path in the radix tree with no branching. The second time a *different* request shares the same prefix but diverges, the speculative insertion detects the branch point and checkpoints the SSM state. This means purely input prefixes cannot be reused on their second occurrence—they are checkpointed during the second request's prefill, and only the third and subsequent requests benefit from the cached state. The paper argues this is an acceptable tradeoff:

> "While this approach sacrifices the benefit of reusing a 'purely input' prefix on its second occurrence, these prefixes are typically shared across many requests. As a result, missing savings on a single request has a negligible impact on overall savings."

**Admission for input-and-output prefixes: last decoded token.** For conversation-continuation and agent-trajectory patterns, the reuse always happens at the boundary between turns. A new message or observation is appended to the *end* of the previous sequence, which includes all prior input and output tokens. There is no branching—the conversation is a linear progression.

Marconi exploits this by **always checkpointing the SSM state after the last decoded token** of every sequence. This state represents the full conversation history up to that point and is exactly what the next turn in the conversation will need. Since conversations typically resume from exactly this point (appending the next user message to the full history), this checkpoint covers the most common reuse scenario with minimal overhead.

**Why not checkpoint everywhere?** The alternative—fine-grained checkpointing at regular intervals (e.g., every 256 tokens as done in vLLM+)—generates a checkpointed SSM state for every block boundary in the sequence. For a 10K-token sequence with block size 32, this produces ~312 SSM states per layer. Marconi, by contrast, admits at most two SSM states per sequence: one at the branching point (if identified) and one at the last decoded token. The paper explicitly quantifies the tradeoff:

> "judicious admission reduces coverage and slightly limits the potential reusability of arbitrary prefixes, as only up to two SSM states are admitted per sequence. However, due to the huge number of low-utility SSM states rejected from admission by Marconi, this altruistic approach significantly reduces the size and improves the utility of cached Hybrid model states, enhancing overall cache utilization."

The phrase "slightly limits the potential reusability of arbitrary prefixes" is important—Marconi cannot handle the case where a future request wants to reuse an arbitrary prefix of a sequence that wasn't a branch point and isn't the last decoded token. However, the empirical evidence (Figure 3a showing 0.4% SSM state reuse for fine-grained checkpointing) suggests such cases are extremely rare in practice, and the savings from not caching thousands of useless states far outweigh the lost opportunities.

---

#### SSM State Checkpointing During Prefill

Once the admission policy has identified a position where an SSM state needs to be checkpointed, Marconi must physically capture that state during the prefill computation. This is a new capability not needed for Transformers (where KVs can be trivially sliced). Marconi supports two methods, selected based on the model architecture (Section 4.1).

**Method 1: Chunked state passing (for models that support it).** Some SSM implementations, including Mamba-2 (Dao & Gu, 2024), perform prefill by splitting the input sequence into fixed-size chunks and computing intra-chunk states in parallel, followed by a sequential pass that propagates states between chunks. The procedure:

- When a checkpoint is needed at token position $p$, Marconi snapshots the SSM state at the chunk boundary immediately preceding $p$. For example, with chunk size 32 and a checkpoint requested at token 80, Marconi captures the state after token 64 (the second-to-last chunk boundary before $p$).

- This state represents tokens 1-64 rather than the exact prefix 1-80, meaning the checkpointed state is slightly shorter than the prefix. The paper acknowledges this:

> "This approach may miss some prefix caching opportunities within a chunk but introduces minimal runtime overhead."

- Optionally, custom kernels can "roll the state forward by a few tokens" to reach the exact position, but this is presented as an optional optimization, not the default path.

The advantage of this method is that it adds zero forward passes: the state is materialized during the chunk boundary computation that is already part of the normal prefill flow. The disadvantage is coarser granularity (chunk-level rather than token-level).

**Method 2: Two-pass prefill (for models without chunked state passing).** For SSMs like Mamba-1 (Gu & Dao, 2023) that don't support chunked state passing during prefill, Marconi performs a two-pass prefill:

1. The first pass prefills tokens 1 through $p$ (the prefix) and captures the SSM state at position $p$ (specifically, at the end of processing token $p$).
2. The second pass starts from this captured SSM state and prefills the remaining tokens $p+1$ through the end of the sequence.

For example, if the full sequence is 100 tokens and the checkpoint is needed at token 80, pass 1 processes tokens 1-80 and saves the SSM state; pass 2 resumes from that saved state and processes tokens 81-100.

This approach provides exact token-level checkpointing but incurs additional computation: the prefill computation for tokens 1-80 is performed twice (once in each pass). However, since the paper's admission policy limits checkpointing to at most one branching position per sequence, this overhead applies to at most one prefix per request.

**Design choice: exact vs. approximate checkpointing.** The paper provides both methods not as competing options but as compatibility pathways for different model implementations. Models with native chunked state passing get zero-overhead approximate checkpointing; models without get exact checkpointing at the cost of a two-pass prefill. The choice is made statically based on the model architecture, not dynamically per-request.

---

#### FLOP Efficiency Metric

The FLOP efficiency metric is the foundation of Marconi's eviction policy. It quantifies the compute savings per unit of memory for a cached entry, enabling Marconi to compare the "investment value" of different cache entries even when their memory footprints differ dramatically.

**The metric.** For a given cache entry (corresponding to a radix tree edge and its associated states), FLOP efficiency is:

$$\text{flop\_efficiency} = \frac{\text{Total FLOPs across layers}}{\text{Memory consumption of all states}}$$

where "Total FLOPs across layers" is the sum of floating-point operations that would be required to recompute all layers' contributions for the tokens covered by this cache entry, and "Memory consumption of all states" is the total bytes occupied by the Attention KVs and SSM states stored for this entry.

**Layer-by-layer computation.** The paper provides detailed FLOP and state size formulas in Appendix A (Table 1). For a given model with known dimensions:

- **Attention layer FLOPs:** $8LD^2 + 4L^2D$, where $L$ is sequence length and $D$ is the model dimension. The term $4L^2D$ is the quadratic self-attention cost; $8LD^2$ accounts for the linear projections (Q, K, V, and output).

- **MLP layer FLOPs:** $16LD^2$, representing two linear transformations (expand and contract).

- **SSM layer FLOPs:** $12LD^2 + 16LDN + 10L$, where $N$ is the SSM state dimension. This includes the input projection, the state-space computation (with the $16LDN$ term proportional to both sequence length and state dimension), and the output projection.

- **Attention layer state size (KVs):** $4LD$ bytes (in FP16: 2 bytes per parameter × key tensor of size $LD$ + value tensor of size $LD$, totaling $4LD$ bytes).

- **SSM layer state size:** $2DN$ bytes. This is constant regardless of sequence length—a single recurrent state matrix of dimension $D \times N$, stored in FP16 (2 bytes per parameter). Additionally, each SSM layer includes a small conv1d state ($\text{in\_channels} \times \text{conv\_kernel} \times 2$ bytes), which the paper notes accounts for only 6.1% of total SSM state size in the 7B model and is omitted from Table 1 for simplicity but included in all experiments.

**Why this metric matters (the key asymmetry).** For Attention layers, state size ($4LD$) is proportional to sequence length $L$, and FLOPs ($8LD^2 + 4L^2D$) are also proportional to $L$ (roughly). The FLOP-per-byte ratio for KVs is therefore approximately constant as sequence length varies:

$$\text{Attention FLOP efficiency} \approx L + 2D$$

which is dominated by the $2D$ term for typical dimensions ($D=4096$), making it near-constant across sequence lengths.

For SSM layers, state size ($2DN$) is **independent of sequence length** ($N$ is a fixed model architecture parameter), while FLOPs ($12LD^2 + 16LDN + 10L$) **scale linearly with $L$**. The FLOP-per-byte ratio is:

$$\text{SSM FLOP efficiency} \approx L \cdot \left(\frac{6D}{N} + 8 + \frac{5}{DN}\right)$$

which **grows linearly with $L$**. For the 7B model ($D=4096$, $N=128$), this simplifies to approximately $200L$—each additional token of sequence length increases the compute savings per byte by roughly 200 floating-point operations.

**The practical consequence.** A long sequence (say, 10K tokens) with mostly SSM layers saves vastly more compute per byte of cache than a short sequence (say, 100 tokens). Under a pure recency (LRU) eviction policy, these two entries would compete only on how recently they were accessed, ignoring that evicting the long sequence "wastes" much more potential compute savings per byte. Figure 5 visualizes this: FLOP efficiency for the Hybrid model grows steeply with sequence length, while Transformer FLOP efficiency is nearly flat.

**Per-layer FLOP decomposition.** Figure 14 (Appendix A) breaks down total FLOPs by layer type in the 7B Hybrid model. For short sequences (<5K tokens), SSM and MLP layers dominate total FLOPs (Attention layers are only 7.1% of layers). As sequences grow beyond 10K tokens, Attention FLOPs grow quadratically and begin consuming a significant fraction of total compute. This means the FLOP efficiency advantage of caching long sequences is partly due to the avoided quadratic Attention cost, but primarily due to the fact that SSM states save large amounts of linear compute at fixed memory cost.

---

#### FLOP-Aware Eviction Policy

The eviction policy determines which cache entries to remove when the cache is full and new entries need space. Marconi's policy assigns a utility score to each radix tree node and iteratively evicts the lowest-scoring nodes until sufficient space is freed.

**The utility score function.** For each node $n$ in the radix tree:

$$S(n) = \text{recency}(n) + \alpha \cdot \text{flop\_efficiency}(n)$$

where `$\text{recency}(n)$` is the timestamp-normalized recency of the node's last access, `$\text{flop\_efficiency}(n)$` is the FLOP-per-byte metric defined above, and `$\alpha$` is a balancing coefficient.

Both `\text{recency}(n)` and `\text{flop\_efficiency}(n)` are normalized to the range (0, 1) by comparing against all other nodes currently in the radix tree. The recency score is the normalized last-access timestamp (most recent = highest), and the FLOP efficiency score is the normalized compute-savings-per-byte (highest efficiency = highest).

**What it computes:** a scalar that combines two competing objectives—favor recently accessed entries (LRU behavior, good for temporal locality) and favor entries that save more compute per byte (FLOP-aware behavior, good for throughput). The coefficient α controls the relative weight: α = 0 gives pure LRU, while higher α values increasingly prioritize compute savings over recency.

**Why this form:** the additive combination of normalized recency and FLOP efficiency is deliberately simple, enabling fast score computation during eviction (no complex multi-objective optimization). The normalization to (0, 1) ensures neither term dominates purely due to different numerical scales. The authors chose this over more sophisticated cost-aware caching algorithms (e.g., GDSF, Cherkasova, 1998) because those algorithms use object size as a proxy for value—which works when size correlates with compute savings (as with KVs) but **fails for SSM states** where size is constant regardless of compute savings:

> "The KV size in Attention layers scales with sequence length, serving as a proxy for compute savings from cache hits, whereas SSM states are fixed-sized irrespective of sequence length or compute savings. Thus, size fails as a proxy in Hybrid LLM inference."

**Eviction granularity: nodes, not edges.** Marconi evicts at the granularity of radix tree nodes, not individual edges. When a node is evicted, all edges incident on that node are affected:

- The node's SSM states are freed. Since SSM states are stored per-edge (representing all prior context), the specific SSM states freed depend on which edges the node anchors.
- The node's KV caches are **absorbed by the child node** (Section 4.3, implementation detail 1). The child node concatenates its parent's KVs with its own, maintaining the property that each node can provide complete KVs for the path from root to that node without ancestor traversal.

**Which nodes are evictable.** Marconi has a specific rule about which nodes are candidates for eviction (Section 4.3, implementation detail 1):

> "All nodes with ≤1 children are considered for eviction, not just leaf nodes."

This is a deliberate departure from standard tree-based caching. Nodes with multiple children represent common prefixes shared by multiple distinct request paths—these are high-utility entries that should be preserved. Nodes with zero or one children represent linear chains (conversation continuations) or dead ends that are unlikely to be useful for future branching reuse. By making all low-fanout nodes evictable, Marconi can reclaim SSM state memory from intermediate points in long linear sequences, not just from leaf nodes.

**Recency tracking is localized.** When a cache hit occurs (a request reuses a cached prefix), Marconi updates the timestamp of only the **accessed node**—the node representing the end of the matched prefix (Section 4.3, implementation detail 2). In standard tree-based caching (e.g., SGLang), all ancestor nodes' timestamps are also updated because their states were "used" during the prefix lookup. Marconi's design is different because:

> "In Marconi, previous SSM states are not reused (Fig. 4c), and although ancestors' KVs are accessed, their KVs will be subsumed by child nodes if evicted."

Since the SSM state at the matched node already represents all preceding tokens, ancestor SSM states are never accessed (only the final cached state is used). And since ancestor KVs will be absorbed by their children upon eviction, updating ancestor timestamps would artificially inflate their recency without reflecting actual reuse probability. This localized timestamping ensures that intermediate nodes in long linear chains can age naturally and become eviction candidates even if their child nodes are frequently accessed.

**Working through an example.** Consider a long conversation represented as a linear chain in the radix tree: root → node1 → node2 → node3 → ... → node_k. Each node stores the SSM state for the conversation up to that point and the KVs for the tokens in its incoming edge.

- A new request in this conversation hits node_k (the latest state), reusing all prior tokens.
- Under Marconi, only node_k's timestamp is updated. Nodes 1 through k-1 do not get timestamp updates.
- Over time, nodes 1 through k-1 grow stale and their utility scores decrease (low recency).
- If another, unrelated conversation branch creates cache pressure, nodes 1 through k-1 are strong eviction candidates—their SSM states are large but haven't been directly accessed in a while, and their KVs would be absorbed by their children, so no information necessary for future reuse of node_k is lost.
- This frees large amounts of SSM state memory while preserving the ability to reuse the latest conversation state (node_k), which is exactly what future requests in this conversation will need.

**Why this form over alternatives:** The paper explicitly compares against size-based cost-aware caching (e.g., GDSF) and argues it is insufficient because SSM state size is not a proxy for utility. The additive scoring function with explicit FLOP efficiency is a lightweight augmentation of recency that captures what size-based approaches miss: the asymmetric relationship between memory cost and compute benefit in Hybrid models.

---

#### Bootstrap α-Tuning

The coefficient α in the utility score function controls the tradeoff between recency and FLOP efficiency, and the optimal value depends on workload characteristics. Manually setting α is impractical because workloads vary in sequence length distribution, arrival patterns, and cache size. Marconi includes an automated tuning mechanism.

**The bootstrap procedure (Section 4.2, "Managing the balance"):**

1. On startup, Marconi initializes α = 0 (pure LRU) and operates normally until the first cache eviction occurs. This gives the system an initial window of request history to analyze.

2. After the first eviction, Marconi takes a **snapshot** of the current radix tree state. The system enters a **bootstrap period** during which it continues using α = 0 (LRU) while simultaneously recording token-level information about all incoming requests. The token-level information includes which prefixes are requested and whether they hit or miss in the cache.

3. The bootstrap period lasts for **5–15× the number of requests seen before the first eviction**. The paper notes that this captures "a representative workload sample"—it ensures the bootstrap data is large enough to be statistically meaningful but bounded so tuning completes quickly.

4. Once sufficient bootstrap data is collected, Marconi **asynchronously launches a grid search** over possible α values. The search evaluates each candidate α by replaying the bootstrap requests through a simulation of the radix tree. For each candidate α, the simulation replays the same request sequence and computes the resulting token hit rate (the fraction of input tokens that would have been cached and reused under that α).

5. The grid search selects the α that maximizes the simulated hit rate. The paper notes:

> "This grid search is parallelized across CPU cores, significantly speeding up the tuning process, typically taking just a few seconds, often shorter than the time required to prefill and decode a single request."

6. After the grid search, Marconi adopts the winning α value for all subsequent eviction decisions.

**What is being optimized:** The objective is token hit rate, which directly corresponds to compute savings and TTFT reduction. The tuning is retrospective (uses past data) and produces a static α that persists until workload characteristics change significantly.

**Why this form:** The grid search is simple and requires no assumptions about workload distributions. It is computationally cheap because it replays token-level traces (not actual model computations) through a lightweight tree simulation. The asynchronous execution (launched in background threads) means it doesn't block the inference serving loop. The paper argues this is sufficient because α doesn't need to change rapidly—workload characteristics shift gradually, and re-tuning can be triggered periodically or when the hit rate degrades.

**Relation to online learning alternatives:** An alternative would be to continuously adapt α using reinforcement learning or online optimization. The paper chooses offline grid search for its simplicity and robustness—the bootstrap replay avoids the exploration risk of trying suboptimal α values on live traffic.

---

#### Design Choices Summary

**Why judicious admission over fine-grained checkpointing:** Fine-grained checkpointing produces thousands of cached SSM states per sequence, 99.6% of which are never reused (Figure 3a). Judicious admission produces at most two (one at the branch point, one at the last decoded token), covering the two dominant reuse patterns (shared input prefixes and conversation continuations) while rejecting the vast majority of low-utility states. The cost is sacrificing reuse of arbitrary intermediate prefixes, but the empirical evidence shows this is an extremely rare case.

**Why FLOP-aware eviction over pure LRU:** LRU assumes recency is a sufficient proxy for future value. This holds for Transformers where compute savings and memory cost both scale linearly with sequence length, so LRU naturally balances the tradeoff. In Hybrid models, SSM state memory cost is sequence-length-independent while compute savings scale linearly, breaking the correlation. FLOP-aware eviction explicitly accounts for this asymmetry by incorporating a compute-per-byte term.

**Why α-tuning via bootstrap replay over static configuration:** The optimal α depends on workload (sequence length distribution, arrival patterns, cache contention level). A static α (e.g., always favor recency or always favor FLOPs) would perform poorly on some workloads. The bootstrap replay provides workload-specific tuning with minimal engineering complexity and zero impact on live serving quality (since exploration happens in simulation).

**Why radix tree over flat block tables:** The radix tree naturally captures sequence overlap structure, making it trivial to detect branch points (multiple children) and linear chains (single child). A flat block table (as in vLLM) doesn't represent the relationships between sequences, making it difficult to implement the admission taxonomy (which requires knowing whether a prefix is shared or linear). The paper explicitly builds on SGLang's radix tree design but repurposes it for admission decisions, not just for space-efficient storage.

**Why horizontal FLOP accounting (summing across all layers) rather than per-layer tracking:** The FLOP efficiency metric aggregates across all layers (Attention, SSM, MLP) for a given sequence position. This is simpler than tracking per-layer FLOPs separately and matches the granularity of cache entries in the radix tree (each edge stores states for all layers). Per-layer tracking would add complexity without clear benefit since eviction decisions operate on whole nodes (which represent all layers' states at a sequence position).

## 4. Key Insights and Innovations

### Innovation 1: The Diagnosis that SSM States Produce "All-or-Nothing" Reuse, Not a Continuum

The paper's most foundational contribution is not a mechanism but a **diagnostic reframing**: it identifies that the core challenge of prefix caching for Hybrid LLMs is not simply that SSM states are large, but that their in-place update semantics create an "all or nothing" reuse pattern that is fundamentally different from the continuous, sliceable reuse of Transformer KV caches.

Prior to this work, the field treated prefix caching as a solved problem for Transformers, where the dominant assumption—implicit in systems like vLLM (Kwon et al., 2023), SGLang (Zheng et al., 2023b), and CachedAttention (Gao et al., 2024)—was that caching more states at finer granularity always increases reuse opportunities, with the only constraint being memory capacity. The KV cache's sequence dimension meant that a full sequence's KVs could be sliced to any prefix length, making partial reuse trivially efficient.

Marconi's diagnosis in Section 3 shows that this assumption **does not transfer** to SSM layers. Because SSM states are updated recurrently and overwritten in-place (Property 2), a state representing tokens 1…q is *categorically unusable* for a request that wants tokens 1…p where p < q—not inefficient, unusable. This transforms the caching problem from a continuous optimization (how fine-grained should checkpoints be?) to a discrete one (which exact sequence positions warrant separate state storage?).

The significance of this diagnosis is that it **explains the failure of naive extension** (vLLM+) without requiring any new mechanism. Figure 3a quantifies this: with block size 32, KVs achieve 25.0% reuse while SSM states manage only 0.4%—a 65.3× gap. This is not because the cache is too small or because the eviction policy is suboptimal; it is because **most checkpointed SSM states are storing positions that will never be requested as prefix boundaries**. The paper essentially proves that fine-grained checkpointing for SSM states is solving the wrong problem: it maximizes the *availability* of states for arbitrary prefix reuse while ignoring that the *demand* for arbitrary prefix reuse at SSM layers is negligible.

This insight is a **conceptual advance** rather than an incremental improvement. It reframes the caching problem from "how do we fit more states in memory?" to "which positions in a sequence are natural prefix boundaries for SSM reuse?" The answer—branch points (shared input prefixes) and sequence endpoints (conversation continuation points)—emerges from this reframing and drives the entire admission policy design. Without this diagnosis, the natural engineering instinct would be to try larger block sizes or more efficient state compression, neither of which addresses the fundamental misallocation.

---

### Innovation 2: A Taxonomy-Based Admission Policy that Replaces "Cache Everything" with "Cache Only Natural Reuse Points"

The second conceptual move is replacing the universal admission policy inherited from Transformer caching (admit all states, let eviction sort out which to keep) with a **selective admission policy guided by a workload-agnostic taxonomy of prefix reuse**. This is a fundamental departure from how all prior LLM prefix caching systems operate.

Every prior system—vLLM, SGLang, CachedAttention, Preble (Srivatsa et al., 2024), PrompCache (Gim et al., 2024)—admits the KV cache states for every token of every sequence. Admission is automatic and universal; sophistication is reserved for eviction (LRU, frequency-based, or cost-aware). This approach works for Transformers because per-token KVs are small and the admission cost is proportional to sequence length, so admitting everything is cheap. It fails catastrophically for Hybrid models because SSM states are large and constant-sized regardless of how many tokens they represent, making universal admission memory-prohibitive.

Marconi's taxonomy (Section 4.1) identifies exactly two scenarios where SSM states are re-used—purely input prefixes (system prompts, few-shot examples) creating branch points, and input-and-output prefixes (conversation history) creating linear continuations—and **refuses to cache SSM states for anything else**. This is not an optimization; it is a categorical rejection of the universal admission principle. The paper is explicit about this philosophy (Section 4.1, "Tradeoffs"):

> "judicious admission reduces coverage and slightly limits the potential reusability of arbitrary prefixes, as only up to two SSM states are admitted per sequence. However, due to the huge number of low-utility SSM states rejected from admission by Marconi, this altruistic approach significantly reduces the size and improves the utility of cached Hybrid model states"

What makes this distinctively innovative is that it **inverts the burden of proof**: instead of the system having to prove a state is *not* worth caching (through eviction), the admission policy requires each candidate state to prove it *is* worth caching by matching one of the two reuse patterns. The speculative insertion mechanism is the concrete realization: a state is only admitted if its insertion into the radix tree would create a branch point (indicating multi-request input sharing), or if it represents the last decoded token (the universal conversation continuation point).

The comparison with SGLang+ in the evaluation (Figure 8) demonstrates why this matters beyond the obvious memory savings. SGLang+ uses the same radix tree structure but admits all tokens' states—it organizes states more efficiently but doesn't reduce their number. Marconi's selective admission produces token hit rate improvements of 4.5–34.4× over vLLM+ (which also admits universally), and the comparison with SGLang+ isolates the eviction-policy contribution separately (showing FLOP-aware eviction adds 19.0–219.7% further improvement). The two mechanisms are cleanly separable in the evaluation design.

This is a **fundamental shift**, not a refinement. It changes the caching paradigm from "admit all, evict selectively" to "admit selectively, evict selectively," acknowledging that for SSM states, the admission decision is where most of the efficiency is won or lost. The taxonomy itself is not theoretically complex (two categories), but the decision to build admission policy around it—and the demonstration that these two categories empirically cover the vast majority of reuse—is what makes it significant.

---

### Innovation 3: FLOP Efficiency as a Unified Metric for Comparing Heterogeneous State Types in Eviction Decisions

The third conceptual contribution is the **FLOP efficiency metric** and its integration into cache eviction scoring, which addresses a structural asymmetry that prior caching systems—both in LLM serving and in general—could safely ignore because they never managed heterogeneous state types with fundamentally different memory-vs-compute relationships.

In traditional Transformer prefix caching, all cached states are KVs. The memory cost of a KV entry scales linearly with the sequence length it represents, and the compute savings from reusing that entry also scale (roughly) linearly with sequence length. This means the ratio of compute-saved to memory-cost is approximately constant across entries, and recency-based eviction (LRU) implicitly balances the tradeoff: longer sequences take more memory but also save more compute, so LRU naturally favors entries in proportion to their "value density."

In Hybrid models, cached entries contain **both** KVs (memory scales with L, compute savings scale with L) **and** SSM states (memory is constant regardless of L, compute savings scale with L). This breaks the proportionality. A short sequence of 100 tokens and a long sequence of 10K tokens require the **same SSM state memory** per layer—the recurrent state is fixed-size—but the long sequence saves ~100× more FLOPs when reused. Under pure LRU, these two entries compete only on recency; the long sequence receives no preference despite delivering dramatically more compute savings per byte.

The FLOP efficiency metric (Equation 1, formalized in Appendix A's Table 1) is not mathematically complex, but its **conceptual role** is what makes it an innovation: it is the first metric in LLM caching that explicitly accounts for the fact that different entries have different "investment returns" in terms of compute savings per unit of cache space. Prior cost-aware caching algorithms (e.g., GDSF, Cherkasova, 1998) use object size as a proxy for value, which fails here because SSM state size does not vary with compute savings. FLOP efficiency replaces size-based proxies with a direct, layer-type-aware accounting of saved operations.

The significance is demonstrated empirically in Figure 10. On a SWEBench trace, FLOP-aware eviction achieves a 99.4% higher token hit rate than LRU (32.7% vs. 16.4%). The fine-grained breakdown (Figure 10a) shows the mechanism at work: Marconi has a lower hit rate (-3.0%) for short sequences (<7K tokens) but a dramatically higher hit rate (+25.5%) for long sequences (>7K tokens). This is the **intentional tradeoff** that LRU cannot make: sacrificing recency-hot but FLOP-inefficient short sequences to preserve FLOP-efficient long ones. The TTFT distribution (Figure 10b) confirms this is a favorable trade: P5 TTFT increases by only 2.1 ms (short sequences are fast to prefill even on cache miss), while P50 and P95 TTFT decrease by 13.4% and 22.0% (74.2 ms and 274.9 ms).

This is a **conceptual advance with practical implications** rather than a theoretical breakthrough. The metric itself is straightforward arithmetic, but its integration into a caching policy—and the demonstration that the recency-vs-efficiency tradeoff is workload-dependent and can be auto-tuned (via bootstrap α search)—establishes a template for how future hybrid-architecture serving systems should handle eviction. It also implies that as SSM state dimensions continue to grow (Figure 12b shows Marconi's advantage growing from 5.7× to 35.4× as state dimension increases from 16 to 128), FLOP-aware eviction will become increasingly critical.

---

### Innovation 4: Empirical Quantification of the Difficulty-Dependent Nature of Caching Policy Optimality

This innovation is less about a specific mechanism and more about an **empirical finding that changes how system designers should think about caching policy configuration**. The paper demonstrates—through controlled microbenchmarks in Section 5.4—that the optimal caching policy is not a universal property of Hybrid models but depends sensitively on workload characteristics, model architecture parameters, and cache capacity.

Several findings support this:

**Cache contention modulates policy impact (Figure 11).** Marconi's improvement over SGLang+ is not monotonic in cache size. It is modest at extreme contention (24.3% improvement at 60 GB, where the cache is so constrained that even optimal eviction can only do so much) and at low contention (10.0% at 140 GB, where the cache is spacious enough that suboptimal eviction doesn't hurt). The peak improvement (68.3%) occurs at **moderate contention**, where eviction decisions are frequent enough to matter but the cache is not yet starved. This is a non-obvious interaction that system operators would not intuit from first principles.

**SSM layer ratio is a first-order determinant of policy benefit (Figure 12a).** As the Attention:SSM ratio shifts from 1:2 to 1:4 to 1:8, Marconi's improvement over baselines grows from 13.5%/5.8% to 66.6%/26.0% to 2.6×/59.7%. At the limit (pure Transformer, 0:36), all three systems perform identically. This means that **the very architectural trend that makes Hybrid models attractive (more SSM layers, fewer Attention layers) also makes intelligent caching proportionally more valuable**. A system designer choosing between a 1:2 and 1:8 Hybrid model for quality reasons is also implicitly choosing how much caching overhead they'll face; Marconi's curves in Figure 12a provide the empirical basis for factoring this into architectural decisions.

**SSM state dimension is a multiplier on the problem (Figure 12b).** As state dimension increases from 16 (Mamba-1) to 128 (Mamba-2), Marconi's token hit rate advantage over vLLM+ grows from 5.7× to 35.4×. This aligns with the trend toward larger state dimensions for better modeling capability (Gu & Dao, 2023; Dao & Gu, 2024) and means that caching policy will become more—not less—important as future SSMs evolve.

**Arrival patterns change the absolute hit rate but not the relative advantage (Figure 13).** Higher session arrival rates and longer inter-request response times both reduce absolute token hit rates (more sessions sharing fixed cache, longer delays between prefix reuses). However, Marconi's relative improvement over SGLang+ actually **increases** (from 1.4× to 1.6×) under these conditions, because contention between requests intensifies and eviction decisions become more consequential.

Taken together, these findings constitute a **diagnostic contribution**: they identify *which factors matter* for caching policy effectiveness and *how much* they matter. This is distinct from the mechanism contributions (Innovations 1-3) because it provides the empirical scaffolding for generalization beyond the specific workloads tested. The paper's claim that "Marconi performs better in scenarios with longer contexts, higher ratios of SSM layers, and larger SSM state dimensions—trends that align with recent model developments" (Section 5, key finding 3) is backed by systematic sweeps across all three factors, not just anecdotally.

This is an **empirical contribution with prescriptive force**. It tells future system builders: if you are deploying a Hybrid model with a high SSM ratio and large state dimension at moderate cache contention, Marconi-style policies are essential; if you are deploying a near-Transformer architecture with small SSM states at low contention, vanilla LRU may suffice. The quantification of these thresholds is what makes the contribution actionable rather than merely observational.

## 5. Experimental Analysis

### Evaluation Methodology

- **Dataset.** The evaluation uses three real-world request traces: **LMSys** (Zheng et al., 2023a), a large-scale conversation dataset with relatively long output sequences often reaching thousands of tokens; **ShareGPT** (sha, 2024), a multi-turn conversational dataset with succinct outputs typically tens to hundreds of tokens; and **SWE-Agent** (Yang et al., 2024a) acting on **SWE-Bench** (Jimenez et al., 2023), an agentic workload for software engineering tasks with widely varying input sequence lengths from hundreds to tens of thousands of tokens (Figure 6). Each dataset contains multiple chat sessions, each with multiple rounds of requests, and all traces are tokenized using the `meta-llama/Llama-2-7b-hf` tokenizer for consistency.

- **Base model(s).** The primary model is a **7B Hybrid model** with 4 Attention layers, 24 SSM layers, and 28 MLP layers (an Attention:SSM ratio of 1:6, following the common design pattern cited in Section 2.1). For TTFT latency measurements, the paper uses **Jamba-1.5-Mini**, a production 12B-active/52B-total-parameter Hybrid model served with state dimension 128 on four A100-40GB GPUs. All experiments use FP16 precision.

- **Metrics.** The primary metric is **token hit rate**, defined as the ratio of the number of input tokens that skipped prefill (due to cache hits) over the total number of input tokens across all requests. This approximates total FLOP savings—the paper argues it is a reasonable proxy because prefill is compute-bottlenecked. The latency metric is **Time to First Token (TTFT)**, reported at different percentiles (P5, P50, P95) in milliseconds. The paper does not evaluate downstream accuracy metrics because prefix reuse is exact and produces identical outputs.

- **Baselines.** Three baselines are compared against Marconi: **(1) Vanilla inference** performs no prefix caching, prefilling every request from scratch; **(2) vLLM+** extends vLLM (Kwon et al., 2023) to support Hybrid models by performing fine-grained checkpointing and caching a state for every token block, using a block size of 32 (the largest vLLM supports; this favors vLLM+ by minimizing both the number of low-utility SSM states admitted and KV internal fragmentation); **(3) SGLang+** extends SGLang (Zheng et al., 2023b) by applying the same judicious admission policy as Marconi to the radix tree, but uses LRU for eviction rather than FLOP-aware eviction. The "+" notation indicates these are constructive extensions by the authors since neither vLLM nor SGLang natively supports prefix caching for Hybrid or SSM models.

- **Generation budget / compute accounting.** The paper does not use a per-request compute budget in the traditional sense—prefix caching is a zero-accuracy-loss optimization that skips computation for shared prefixes. The effective "budget" is the **cache capacity** (GPU HBM), expressed in GB, which determines how many model states can be retained. The evaluation varies cache sizes from 60 GB (high contention) to 140 GB (low contention). The number of tokens per sequence, sequence length distributions, and arrival patterns (session arrival rate in sessions/second, inter-request response time in seconds) are the workload parameters that interact with cache capacity. Since caching is exact reuse with no approximations, the tradeoff is purely between memory consumption and compute savings—there is no generation budget or sampling involved.

- **Cross-validation / statistical protocol.** The paper does not use traditional cross-validation (this is a systems evaluation, not a machine learning training pipeline). Instead, statistical robustness is handled via **distributional analysis across requests**. Results in Figures 7, 8, 9, and 10 report quartiles (box plots) and percentiles (P5, P95 via whiskers, CDFs) across all requests in a trace, allowing assessment of both typical (median) and tail behavior. The bootstrap α-tuning procedure (Section 4.2) uses a retrospective replay of 5–15× the number of requests seen before the first eviction to select α, but this is part of the system's online operation rather than an evaluation methodology per se. The experiments are run on an AWS p4d.24xlarge instance with eight A100-40GB GPUs, 96 Intel Xeon Platinum 8275CL CPUs, and 1152GB DDR4 RAM.

---

### Main Quantitative Results

#### End-to-End Token Hit Rate: Marconi vs. vLLM+

The paper's headline comparison is between Marconi's judicious admission + FLOP-aware eviction against vLLM+'s fine-grained checkpointing + LRU eviction. **Figure 7** reports the token hit rate distribution (box plots with P5/P95 whiskers) across the three workloads:

- **LMSys:** Marconi achieves a token hit rate distribution with a median visibly higher than vLLM+. The paper reports an **average improvement of 4.5×** over vLLM+. The vLLM+ box plot shows a median token hit rate clustered in the low teens (roughly 10-15% from visual inspection of Figure 7a), while Marconi's median falls in the 25-35% range.

- **ShareGPT:** Marconi achieves an **average improvement of 7.3×** (Figure 7b). vLLM+'s token hit rate distribution is compressed closer to the origin relative to LMSys (consistent with ShareGPT's shorter sequences producing fewer sparsely-hit states, but still substantial waste), while Marconi's distribution spans a much wider and higher range.

- **SWEBench:** The most dramatic improvement: **34.4×** average token hit rate gain (Figure 7c). vLLM+'s box plot shows near-zero token hit rate for the vast majority of requests (the interquartile range sits barely above 0%), consistent with the extreme sequence length diversity in SWEBench (Figure 6c) causing massive cache thrashing. Marconi's distribution, while more variable (wider interquartile range), achieves token hit rates in the 15-40% range.

The paper attributes this improvement to admission selectivity: by rejecting the vast majority of SSM states (which vLLM+ admits indiscriminately), Marconi preserves cache space for the small number of states that will actually be reused.

---

#### FLOP-Aware Eviction vs. LRU Eviction (Marconi vs. SGLang+)

To isolate the contribution of FLOP-aware eviction from admission policy, the paper compares Marconi against SGLang+, which uses the same judicious admission but LRU for eviction. **Figure 8** reports the **token hit rate improvement of Marconi over SGLang+** (as distributions across different random seeds or arrival patterns):

- **SWEBench:** The largest improvement, with a **P95 win of 219.7%** (the best-case improvement in the long tail of runs). The distribution shows most runs clustering between 25% and 100% improvement, with the whisker extending out to the 219.7% mark.

- **LMSys:** A **P95 win of 45.6%**. The distribution is tighter, with most runs showing improvements in the 10-40% range.

- **ShareGPT:** A **P95 win of 19.0%**—the smallest improvement among the three workloads.

The paper explains this ordering by workload sequence-length characteristics (Figure 6): SWEBench has the widest input sequence length distribution (hundreds to tens of thousands of tokens), creating the largest asymmetry between short and long sequence FLOP efficiency. LMSys has a narrower distribution with most sequences under 10K tokens, and ShareGPT predominantly features sequences under 2K tokens. When most sequences are short, FLOP efficiency differences between entries are small, so LRU doesn't make many "mistakes" that FLOP-aware eviction would correct. When sequences span wide length ranges, LRU's blindness to compute savings becomes costly.

---

#### P95 TTFT Latency Reductions

**Figure 9** reports the CDFs of P95 TTFT relative to vanilla inference (no prefix caching) across the three workloads, comparing Marconi, SGLang+, and vLLM+:

- **LMSys (Figure 9a):** All three caching systems achieve P95 TTFT reductions relative to vanilla inference, but Marconi provides the largest reduction. The paper reports Marconi reduces P95 TTFT by up to **36.9%** (281.4 ms) compared to vanilla inference. Compared to vLLM+, Marconi delivers up to **36.1%** (275.4 ms) larger P95 TTFT reduction. Compared to SGLang+, the additional reduction is **17.2%** (131.1 ms).

- **ShareGPT (Figure 9b):** Marconi reduces P95 TTFT by up to **73.2%** (106.3 ms) vs. vanilla inference—a smaller absolute savings than LMSys because ShareGPT sequences are shorter and thus faster to prefill even without caching. Compared to vLLM+, Marconi delivers **71.1%** (103.3 ms) additional reduction; vs. SGLang+, **12.8%** (18.5 ms).

- **SWEBench (Figure 9c):** Marconi reduces P95 TTFT by up to **46.8%** (617.0 ms) vs. vanilla inference—the largest absolute savings due to SWEBench's long sequences. Compared to vLLM+, Marconi delivers **46.8%** (617.0 ms) additional reduction; vs. SGLang+, **24.7%** (325.7 ms).

The CDF shapes show Marconi's curve shifted left (lower TTFT) relative to both SGLang+ and vLLM+ across all three workloads, with vLLM+ consistently performing worst (its CDF is closest to 1.0, indicating highest latency). The gap between Marconi and SGLang+ narrows at lower percentiles (where cache pressure is lower and eviction policy matters less) and widens at higher percentiles (where contention drives eviction decisions).

---

#### Fine-Grained Analysis of FLOP-Aware Eviction on SWEBench

**Figure 10** provides per-request breakdowns on a specific SWEBench trace to explain *how* FLOP-aware eviction achieves its gains:

**Figure 10a** bins requests by input sequence length and plots the difference in average token hit rate between Marconi and SGLang+ (positive = Marconi wins). The key pattern:
- Sequences with **<7K tokens** show a negative difference, with hit rate reductions of up to **-3.0%** for Marconi. These are the sequences that suffer from FLOP-aware eviction's bias toward longer sequences.
- Sequences with **>7K tokens** show substantial positive differences, with hit rate improvements of up to **+25.5%** for Marconi.
- The crossover point is approximately 7K tokens.

This confirms the intentional tradeoff: Marconi sacrifices hit rate for shorter sequences (whose SSM states are FLOP-inefficient—the same memory cost but less compute savings) to preserve cache space for longer sequences (whose SSM states deliver more compute savings per byte). The paper reports an overall **90.3% improvement in FLOP saved** compared to SGLang+, despite the token hit rate improvement being only 99.4% in relative terms, because the sequences that gain hit rate are much longer and thus each hit saves far more compute.

**Figure 10b** shows the TTFT CDF for all requests in the trace, comparing Marconi, SGLang+, and vanilla inference. The key finding:
- At the **P5** (lowest latencies), Marconi is **6.3% worse** than SGLang+, with an absolute increase of only **2.1 ms** (visible in the magnified inset area). Short sequences cache-miss more often under Marconi, but prefill is so fast for these sequences that the absolute penalty is negligible.
- At the **P50**, Marconi is **13.4% better** (74.2 ms reduction).
- At the **P95**, Marconi is **22.0% better** (274.9 ms reduction).

The CDF visually confirms the tradeoff: Marconi's curve is slightly to the right of SGLang+ at the very bottom (higher latency for the fastest requests) but clearly to the left from the median upward (lower latency for the majority of requests and especially the tail).

---

### Ablation Studies and Robustness Checks

**Impact of cache contention on FLOP-aware eviction benefits (Figure 11):** The paper varies cache size from 60 GB (high contention, limited space forces frequent eviction) to 140 GB (low contention, most entries fit comfortably) and compares Marconi and SGLang+ token hit rates. At 60 GB, the improvement is **24.3%** (both systems struggle with very limited capacity). The improvement peaks at **68.3%** when the cache is moderately constrained (around 80–100 GB, estimated from the x-axis range), where eviction decisions are frequent and consequential. At 140 GB, the improvement drops to **10.0%**—when the cache is large enough, suboptimal eviction rarely occurs because most useful entries fit. This non-monotonic relationship demonstrates that FLOP-aware eviction's value is highest under **moderate** memory pressure, not extreme pressure (where cache capacity is the bottleneck regardless of policy) or abundance (where policy doesn't matter).

**Varying SSM:Attention layer ratios (Figure 12a):** The paper sweeps layer compositions from (32 SSM, 4 Attention, ratio 8:1) through (28 SSM, 7 Attention, ratio 4:1) and (24 SSM, 12 Attention, ratio 2:1) to (0 SSM, 36 Attention, pure Transformer, ratio 0). All results are normalized to the same vLLM+ hit rate for comparability. Marconi's improvement over vLLM+ grows from 13.5% (ratio 2:1) to 66.6% (ratio 4:1) to 2.6× (ratio 8:1). Over SGLang+, the improvement grows from 5.8% to 26.0% to 59.7%. At the pure Transformer extreme, all three systems achieve identical performance—there are no SSM states to mismanage, so admission and eviction policies reduce to standard Transformer caching behavior. This ablation confirms that Marconi's mechanisms become proportionally more valuable as models shift toward more SSM layers, which the paper argues is the dominant architectural trend.

**Varying SSM state dimensions (Figure 12b):** The paper sweeps state dimensions of 16, 32, 64, and 128 (corresponding approximately to Mamba-1 at the low end and Mamba-2 at the high end; Gu & Dao, 2023; Dao & Gu, 2024). Results are normalized hit rates. Marconi's token hit rate improvement over vLLM+ grows from **5.7×** (dim=16) to **9.6×** (dim=32) to **19.9×** (dim=64) to **35.4×** (dim=128). Over SGLang+, the normalized hit rate advantage grows from **1.6×** (dim=16) to **1.7×** (dim=32 and 64) to **1.9×** (dim=128). Larger state dimensions increase each SSM state's memory footprint (from $2D \cdot 16$ to $2D \cdot 128$ bytes per layer), magnifying the cost of caching low-utility states and thus amplifying the benefit of selective admission. The trend aligns with the paper's observation that state dimensions are increasing for better modeling capability.

**Varying request arrival patterns (Figure 13):** Two workload parameters are varied:
- **Session arrival rate (Figure 13a):** As the average number of sessions per second increases from 0.5 to 2, the absolute token hit rate decreases from 48.7% to 43.0% (Marconi) because more sessions compete for the fixed cache capacity. However, Marconi's **relative improvement over SGLang+ grows from 1.4× to 1.6×**, because increased contention makes eviction decisions more consequential—FLOP-aware eviction correctly prioritizes across a larger pool of competing entries.
- **Average inter-request response time within sessions (Figure 13b):** As response time increases from 5s to 10s, the token hit rate decreases from 25.9% to 24.1% (Marconi) because prefixes stay in cache longer between reuses, increasing the chance that an intervening request from another session evicts them. The relative improvement over SGLang+ again grows (1.4× to 1.6×), for the same contention-related reason.

These ablations demonstrate that while absolute hit rates degrade under adverse arrival patterns, Marconi's **relative advantage** over simpler policies actually increases, suggesting robustness to workload variability.

**Interaction between admission and eviction (implicit in Figures 7 and 8):** The paper does not provide a direct ablation where admission is varied while holding eviction fixed (or vice versa) with all combinations reported. However, the comparison between Figure 7 (vLLM+ with fine-grained admission + LRU) and Figure 8 (SGLang+ with judicious admission + LRU) implicitly ablates the admission policy: the jump from vLLM+ to SGLang+ (judged by the residual gap in Figure 9 or the token hit rate differences in Figures 7 vs. 8) isolates the admission contribution. Similarly, the gap between SGLang+ and Marconi in Figure 8 isolates the FLOP-aware eviction contribution. The paper reports the combined effect (34.4× over vLLM+) while separately quantifying the eviction contribution (19.0–219.7% over SGLang+), allowing approximate decomposition.

**Jamba-1.5-Mini for latency validation:** All token hit rate experiments in Section 5.2 and 5.3 use the 7B Hybrid model. The TTFT latency results in Figure 9 use Jamba-1.5-Mini, a different (production-scale) Hybrid model with 52B total parameters. The paper does not provide token hit rate results for Jamba-1.5-Mini or TTFT results for the 7B model, so the latency and hit rate measurements are not directly interconvertible. This is a minor evaluation limitation: the latency savings reported in milliseconds correspond to Jamba-1.5-Mini's compute characteristics, while the hit rate improvements correspond to the 7B model's caching behavior. The paper implicitly assumes that higher token hit rates on the 7B model would translate to proportional latency improvements on Jamba-1.5-Mini, which is reasonable given that prefill is compute-bottlenecked, but no direct validation is provided.

**No ablation of the bootstrap α-tuning procedure:** The paper does not report results comparing the bootstrap-tuned α against (a) a fixed default α, (b) an oracle optimal α chosen with full workload knowledge, or (c) a continuously-adapting α. The bootstrap procedure is described as the mechanism for setting α but its contribution to overall performance—as distinct from the FLOP-aware eviction policy itself—is not quantified through any comparative experiment. This is a notable missing ablation: we cannot determine how much of the FLOP-aware eviction gains depend on workload-specific tuning vs. a reasonable static default.

**No ablation of speculative insertion overhead:** The speculative insertion step (walking the radix tree to detect branch points before prefill) adds computation that is not present in vLLM+. The paper does not measure the latency overhead of this step, either in absolute terms or as a fraction of total request processing time. For short sequences where prefill is fast, this overhead could potentially offset some of the caching benefit, but no data is provided to assess this.

---

### Critical Assessment

The paper makes three central claims that the experimental results must substantiate:

**Claim 1: Marconi achieves up to 34.4× higher token hit rates compared to state-of-the-art prefix caching systems extended to Hybrid models.**

The experiments in Figure 7 directly support this: Marconi achieves 34.4× higher token hit rate than vLLM+ on SWEBench, 7.3× on ShareGPT, and 4.5× on LMSys. These numbers are exact empirical measurements on the reported traces. However, several qualifications are important:

- The 34.4× figure is the **upper bound across workloads**, obtained on SWEBench, which has the widest sequence length distribution and thus the worst baseline performance. The improvement on other workloads (4.5×, 7.3×) is still substantial but an order of magnitude smaller. The paper's abstract highlights 34.4× as the headline number, which is technically accurate but overstates the typical improvement an operator would see unless their workload resembles SWEBench.

- The comparison is specifically against **vLLM+**, which the paper itself configured to be the best-case baseline for existing systems (largest supported block size of 32, minimizing the number of low-utility SSM states). A smaller block size (the vLLM default of 16) would have produced even worse baseline performance and thus an even larger Marconi improvement. The paper's choice to favor the baseline is methodologically sound but means the 34.4× figure is measured against an already-optimized competitor, not a naïve configuration.

- The token hit rate improvements are measured on the **7B Hybrid model**. The TTFT results (Figure 9) use Jamba-1.5-Mini but the paper does not report token hit rates for that model. We must therefore take on faith that the 7B model's hit rate improvements translate to the Jamba model's latency improvements, without direct empirical coupling. This is plausible (hit rate is a property of the cache management policy, not the model), but unverified.

- The evaluation uses **three specific traces** (LMSys, ShareGPT, SWEBench) representing chatbot and agent workloads. The paper does not evaluate on other important workload patterns—batch inference with homogeneous prompts, retrieval-augmented generation where many requests share the same retrieved context, or mixture-of-experts routing that might create different multiplexing patterns. The 34.4× claim may not generalize to workloads not represented in these three traces.

**Claim 2: The improvements translate to 71.1% (617 ms) lower P95 TTFT—latency savings that are practically meaningful.**

Figure 9 supports this claim with one important nuance: the **71.1% figure** (on ShareGPT) and the **617 ms figure** (on SWEBench) come from **different workloads** and are **different metrics**. The 71.1% is the improvement over vLLM+ on ShareGPT (a relative measure), while the 617 ms is the absolute TTFT reduction on SWEBench. Combining them in the abstract as "71.1% or 617 ms lower TTFT" is rhetorically effective but masks that these represent different operating points.

More substantively: the TTFT improvements are measured relative to **vanilla inference without prefix caching**, not relative to the best possible latency the hardware could achieve. The 617 ms reduction on SWEBench represents a 46.8% reduction from the no-caching baseline. The absolute savings are large enough to be user-visible (hundreds of milliseconds), supporting the paper's claim of practical significance. However, the paper does not report **throughput** (tokens/s at the server level), which is the other half of the serving efficiency equation. Higher cache hit rates should improve throughput by reducing redundant prefill computation, but without throughput numbers, we cannot assess whether Marconi's cache management overhead (radix tree walks, speculative insertion, bootstrap grid search) offsets some of the compute savings at the system level.

Additionally, the TTFT measurements use Jamba-1.5-Mini on four A100-40GB GPUs. The paper does not report multi-GPU scaling behavior or whether the cache management becomes a bottleneck when the model is sharded across more GPUs (larger models) or when serving many concurrent requests. These are practical concerns for production deployment that the evaluation does not address.

**Claim 3: Marconi's FLOP-aware eviction improves token hit rate by 19.0–219.7% over LRU (SGLang+).**

Figure 8 and the Figure 10 breakdown provide strong evidence for this claim, with the mechanism clearly explained: FLOP-aware eviction sacrifices hit rate for short sequences to boost hit rate for long ones, a tradeoff that benefits overall FLOP savings because long sequence hits save disproportionately more compute. The claim is well-supported within the evaluated workload conditions.

However, several genuine limitations exist:

- **The bootstrap α-tuning procedure is a black box in the evaluation.** The paper does not report the α values selected by the bootstrap procedure for different workloads, nor does it compare bootstrap-tuned α against (a) a fixed α = 1 (always balance equally), (b) α = ∞ (pure FLOP-aware, no recency), or (c) an oracle α that maximizes hit rate on the full trace. Without these comparisons, we cannot distinguish how much of the FLOP-aware eviction benefit comes from having a FLOP term in the utility function at all versus having the *correct* α weight. If a fixed α = 1 performs nearly as well as the bootstrap-tuned value, then the bootstrap procedure is an unnecessary complexity; if tuning is critical, then the evaluation should quantify the gap.

- **The 219.7% P95 improvement appears to be an outlier.** The distribution in Figure 8 for SWEBench shows most runs clustering between 25% and 100% improvement, with the whisker extending to 219.7%. The median improvement appears closer to 50-75% (from visual inspection). Highlighting the P95 extreme overstates the typical gain, though the typical gains are still substantial.

- **The evaluation does not vary the radix tree structure itself.** Marconi inherits SGLang's radix tree design and adds admission/eviction policies on top. An alternative approach—using a flat block table (vLLM-style) with Marconi's admission and eviction policies—is not evaluated. This makes it impossible to determine whether the radix tree is essential or whether the policies alone drive the improvements. The paper's statement that "our system doesn't invent new data structures for KV cache management" (Section 4.1) is candid, but it also means the evaluation cannot separate the data structure contribution from the policy contribution.

**Claim 4: Marconi performs better with longer contexts, higher SSM ratios, and larger SSM state dimensions—trends aligning with recent model developments.**

Figures 12a, 12b, and 13 provide systematic evidence for all three trends. The sweeps are clean and the results are monotonic in the expected direction. This is the most methodologically robust part of the evaluation: each factor is varied in isolation while holding others constant, and the relative improvements are consistently reported.

However, these are **synthetic sweeps**: the paper varies one architectural parameter at a time on the 7B model while using real workload traces. The "higher SSM ratio" sweep changes the layer composition of what is presented as the same base model, which may not correspond to how real Hybrid models at different SSM ratios are actually designed and trained. A model with a 1:8 Attention:SSM ratio is not simply a 1:2 model with more SSM layers swapped in; it would typically be trained differently and have different quality characteristics. The paper's sweeps usefully isolate the caching-relevant parameters but should not be interpreted as evaluating real model variants.

**Missing experiments that would strengthen the paper:**

1. **Throughput benchmarks.** Hit rate and TTFT improvements should translate to higher throughput (requests/second), but no throughput numbers are reported. For production deployments, throughput per dollar is often the primary metric.

2. **Multi-GPU scaling.** The experiments use a single node with 8 GPUs. For the Jamba-1.5-Mini model, only 4 GPUs are used. Larger models sharded across more GPUs might expose communication bottlenecks in cache state retrieval that aren't visible at this scale.

3. **Cache miss penalty quantification.** Marconi's admission policy *intentionally* misses some reuse opportunities (arbitrary intermediate prefixes, second-occurrence purely-input prefixes). The paper argues these are rare, but doesn't quantify the **miss rate attributable to selective admission** (as opposed to capacity-driven eviction). A breakdown of why tokens miss—admission policy didn't cache the state vs. state was evicted vs. state was never seen—would clarify whether the admission policy's selectivity is leaving meaningful reuse on the table.

4. **Comparison with approximate prefix caching methods.** CacheBlend (Yao et al., 2024) and PromptCache (Gim et al., 2024) reuse KVs with approximations that trade accuracy for hit rate. Comparing Marconi's exact-but-selective approach against approximate-but-broad approaches for Hybrid models would contextualize the design choice to prioritize exact reuse.

5. **Breakdown of SSM state memory vs. KV memory in the cache.** The paper argues SSM states dominate cache capacity, but never shows the actual memory breakdown between KVs and SSM states in Marconi's cache for different workloads. This would validate the claim that SSM state management is the primary bottleneck and would show how much KV caching (the "easy" part) contributes to hit rates independently.

6. **Sensitivity to the bootstrap period length.** The bootstrap procedure uses 5–15× the pre-first-eviction request count. How sensitive is the tuned α (and resulting hit rate) to this multiplier? Too short a bootstrap and α might be poorly tuned; too long and the system operates suboptimally (on LRU) for many requests.

**Overall assessment:** The experiments convincingly demonstrate that Marconi's admission and eviction policies provide substantial improvements over vLLM+-style fine-grained checkpointing on three realistic workload traces, and that the improvements are mechanistically attributable to (a) avoiding caching of low-utility SSM states and (b) favoring long-sequence entries during eviction. The paper is transparent about the experimental configuration and about which claims are workload-dependent. The primary limitation is the narrow evaluation scope—single node, no throughput numbers, no comparison with approximate methods, and the coupling between hit rate measurements (on the 7B model) and latency measurements (on Jamba-1.5-Mini) that isn't directly validated. The synthetic microbenchmarks (layer ratios, state dimensions, arrival patterns) provide strong evidence that the benefits will generalize to future Hybrid models with more SSM-heavy architectures, but this extrapolation rests on the assumption that real future models will behave like parameter-swept versions of the current 7B model, which may or may not hold.

## 6. Limitations and Trade-offs

### 6.1 The Difficulty Estimation Cost Is Not Accounted for in Headline Efficiency Gains

Marconi's Judicious Admission policy relies on a speculative insertion operation that walks the radix tree before every prefill to detect branch points and decide which SSM states to checkpoint (Section 4.1). Additionally, the bootstrap α-tuning procedure replays 5–15× the pre-first-eviction request count through a grid search to select the balancing coefficient (Section 4.2). Neither of these costs is included in the reported token hit rate or TTFT latency measurements.

**Consequence.** The speculative insertion is a radix tree traversal that operates on token-level metadata—not a model forward pass—so its per-request overhead is likely modest in absolute terms. However, for short sequences where prefill itself is fast (e.g., ShareGPT, where P5 TTFT improvements are only ~2 ms, per Figure 10b), even a small constant overhead per request could erode or negate the latency savings at the low end. The bootstrap grid search is asynchronous and the paper claims it "typically tak[es] just a few seconds, often shorter than the time required to prefill and decode a single request," but this is an unverified claim—no measurement of bootstrap latency, CPU utilization during grid search, or interference with concurrent request serving is provided. For high-throughput deployments, even a few seconds of CPU-intensive replay every few hundred requests could create scheduling artifacts.

**What evidence exists.** The paper acknowledges the speculative insertion conceptually (Section 4.1: "prior to prefilling each sequence, Marconi employs a speculative insertion of the input tokens to see if new intermediate nodes will be created") and the bootstrap procedure (Section 4.2: "Marconi asynchronously launches a grid search over possible α values"), but **provides zero measurements** of either overhead. Figure 9 reports TTFT improvements but these are measured after any admission or tuning overhead has been absorbed; the latency numbers represent pure prefill savings, not end-to-end request latency including cache management overhead. The fine-grained analysis in Figure 10 shows Marconi's P5 TTFT is 6.3% worse than SGLang+—this is attributed to FLOP-aware eviction deprioritizing short sequences, but could be partially explained by admission overhead as well if SGLang+ uses identical admission but avoids the bootstrap grid search. The paper does not disentangle these effects.

**Mitigation status.** The paper does not address this limitation. The speculative insertion is described as part of the admission policy with no discussion of its time complexity or measured latency. The bootstrap procedure is described with an unverified timing claim and no sensitivity analysis of the bootstrap multiplier (5–15×). An operator deploying Marconi would need to measure these overheads on their own hardware and workload, but the paper provides no data to guide expectations.

---

### 6.2 The Hardest Problems in Prefix Caching Receive No Benefit—No Mechanism Handles Arbitrary Prefix Reuse

Marconi's Judicious Admission policy explicitly refuses to cache SSM states for any sequence position that is neither a detected branch point (shared input prefix) nor the last decoded token (conversation continuation point). The paper states this directly in Section 4.1 ("Tradeoffs"):

> "judicious admission reduces coverage and slightly limits the potential reusability of arbitrary prefixes, as only up to two SSM states are admitted per sequence."

The "reduces coverage" phrasing understates the categorical nature of the decision: if a workload contains patterns where requests reuse arbitrary intermediate prefixes of prior sequences—neither at conversation boundaries nor at branch points—Marconi provides **zero** cache hit benefit regardless of cache capacity, because the required SSM states were never admitted. The paper implicitly argues this pattern is rare (supported by Figure 3a showing 0.4% SSM state reuse under fine-grained checkpointing with block size 32), but this is an empirical observation on three specific traces, not a workload-agnostic guarantee.

**Consequence.** For workloads with prefix reuse patterns outside Marconi's taxonomy, the system's token hit rate collapses to vLLM+-like levels (single-digit percentages in Figure 7c) because the states needed for reuse were categorically excluded from the cache. The paper does not characterize what fraction of potential reuse opportunities are intentionally sacrificed by selective admission versus lost due to capacity-driven eviction. This makes it impossible for a practitioner to assess *a priori* whether their workload's reuse patterns match Marconi's taxonomy without running a trace analysis themselves.

**What evidence exists.** The paper's strongest evidence that arbitrary prefix reuse is rare comes from Figure 3a (vLLM+ baseline showing 0.4% SSM state reuse with block size 32) and the overall high hit rates Marconi achieves (25–42% in Figure 7). However, Figure 3a is measured on the vLLM+ baseline, which suffers from cache thrashing—the 0.4% reuse rate could be partially *caused by* fine-grained checkpointing overwhelming the cache, not an inherent property of the workload. If vLLM+ had infinite cache capacity, would SSM state reuse rates approach KVs' 25%? The paper provides no data to separate the effect of capacity pressure from the effect of genuinely rare reuse. On SWEBench (Figure 7c), vLLM+ achieves near-zero hit rate—but this is the workload with the widest sequence length distribution, where capacity pressure is extreme. It's possible that many intermediate SSM states *would* be reusable if they could be retained, but they can't because the cache is flooded. Marconi solves the flooding problem by not admitting them, but this doesn't prove they aren't reusable; it only proves that not admitting them is better than admitting them under extreme capacity pressure.

**Mitigation status.** The paper does not address this limitation directly. The taxonomy-based admission is presented as a strength, and the empirical evidence supports it for the tested workloads, but there is no discussion of workloads where the taxonomy might break down nor any mechanism for falling back to fine-grained checkpointing when detected reuse patterns warrant it. The admission policy is static and categorical—it doesn't adapt its selectivity based on cache contention or observed workload patterns beyond the α-tuning (which affects eviction, not admission).

---

### 6.3 The Generalization Evidence Rests Entirely on Synthetic Parameter Sweeps, Not Real Model Variants

The paper's central forward-looking argument is that Marconi's benefits will grow as models evolve toward higher SSM ratios, larger state dimensions, and longer sequences—"trends that align with recent model developments" (Section 5, key finding 3). The evidence for this claim comes from Figures 12a, 12b, and 13, which are synthetic sweeps on the same 7B base model architecture: changing the layer composition by swapping layer types in the model configuration file, changing the SSM state dimension parameter, and simulating different arrival patterns from the same trace distributions.

**Consequence.** A real Hybrid model with a 1:8 Attention:SSM ratio (like Jamba, Lieber et al., 2024) is not simply a 1:2 model with 6 extra SSM layers grafted in. It has different training dynamics, different layer interactions, different optimal state dimensions, and potentially different quality characteristics. The paper's sweeps isolate the *caching-relevant parameters* (number of SSM states generated, size of each state), but do not capture how those parameters interact with model quality, token distributions, or real hardware performance. An operator evaluating whether Marconi is needed for a specific production model (say, Jamba-1.5-Mini) cannot straightforwardly map the synthetic sweep results to their model because the sweeps vary one parameter at a time on a fixed base architecture, while real models vary all parameters simultaneously in interdependent ways.

Additionally, the TTFT latency results use Jamba-1.5-Mini (a real production model), but the token hit rate results—the primary metric—use the synthetic 7B model. The paper provides no token hit rate measurements for Jamba-1.5-Mini, so the coupling between "higher hit rate on the 7B model" and "lower TTFT on Jamba" is an assumption, not a demonstrated fact.

**What evidence exists.** The paper explicitly acknowledges the synthetic nature of the sweeps only indirectly—the 7B model is described as "a Hybrid model with {4, 24, 28} {Attention, SSM, MLP} layers" (Section 5.1), not as a published pretrained model. Figure 12a varies layer compositions, and Figure 12b varies state dimensions, but neither figure reports model quality (perplexity, downstream accuracy). The sweeps demonstrate that *if all else is equal*, increasing SSM ratio amplifies Marconi's benefit—but "all else equal" is artificial. The paper does not discuss this artificiality.

**Mitigation status.** The paper includes Jamba-1.5-Mini as a validation point for latency (Figure 9), which partially addresses the concern for one real model. However, without token hit rate numbers for Jamba, the latency results don't verify that the caching mechanisms that produced high hit rates on the 7B model also produce high hit rates on Jamba. An operator might reasonably worry that a different model's token distribution, conversation patterns, or serving infrastructure could change the effectiveness of the radix tree branch-point detection. The paper does not propose or conduct experiments on additional real model families.

---

### 6.4 Throughput—the Other Half of Serving Efficiency—Is Not Measured

The paper evaluates Marconi exclusively on token hit rate and TTFT latency. There are **zero throughput measurements** (requests per second, tokens per second at the server level) anywhere in the evaluation. This is a significant omission because prefix caching impacts both latency and throughput: higher hit rates reduce redundant prefill computation, which should increase throughput, but Marconi's cache management overhead (speculative insertion per request, radix tree maintenance, bootstrap grid search, chunked/two-pass prefill checkpointing) consumes CPU and GPU cycles that could offset the savings.

**Consequence.** In production LLM serving, throughput and latency are coupled. Reducing P95 TTFT by 617 ms (as on SWEBench) is valuable, but if the per-request cache management overhead reduces the server's maximum throughput by some fraction, the system-level cost-per-query might not improve in proportion to the latency win. For batch-oriented deployments (periodic evaluation, data generation pipelines), throughput is often the primary metric, and the paper provides no data to guide expectations. Additionally, the paper's argument that FLOP-aware eviction achieves a "90.3% improvement in FLOP saved" (Section 5.3) is a FLOP accounting claim, but without throughput numbers, we cannot verify that saved FLOPs translate to increased throughput—they might translate only to reduced GPU utilization if the system is not compute-bound, or they might be partially consumed by the cache management overhead.

**What evidence exists.** The paper reports token hit rate as the primary metric and argues it "approximates the total compute saved (FLOP) well" (Section 5.1). TTFT is reported for latency. Throughput is **never mentioned** in the evaluation methodology, results, or discussion. The hardware configuration (eight A100-40GB GPUs, Jamba-1.5-Mini on four GPUs) is described but no concurrent request load is specified—we don't know how many requests are in-flight simultaneously or whether the system is running at saturation. The arrival patterns in Figure 13 vary session and request rates, but these are trace-level parameters affecting which prefixes compete for cache space, not server throughput measurements.

**Mitigation status.** The paper does not address this limitation at all. The abstract, introduction, and conclusion all frame the contribution in terms of token hit rate and TTFT latency, with no mention of throughput. A practitioner deploying Marconi in a throughput-sensitive setting would need to measure throughput independently—the paper provides no guidance.

---

### 6.5 The Evaluation Scope Covers Only Three Specific Workload Traces on a Single Task Family (Conversation and Agent Interaction)

All experiments use three request traces: LMSys (chatbot conversations), ShareGPT (chatbot conversations with shorter outputs), and SWE-Bench (LLM agent software engineering interactions). These are all interactive, multi-turn workloads where the dominant reuse patterns are conversation continuations and shared system prompts—the two patterns Marconi's taxonomy is explicitly designed for (Section 4.1). This creates a potential circularity: the paper designs its admission policy to handle patterns observed in these traces, then demonstrates that the policy works well on these traces.

**Consequence.** The paper provides no evidence that Marconi generalizes to other important LLM serving workloads: batch inference with homogeneous prompts (many requests sharing the exact same input, common in evaluation and data synthesis), retrieval-augmented generation (RAG) where many requests share retrieved context blocks but differ in how they're combined, code completion where prefixes are accumulated incrementally through editing, or speculative decoding where multiple model instances process the same prefixes. Without evaluation on a broader workload set, the claim that Marconi's taxonomy-based admission "sufficiently estimates" reuse potential (Section 4.1) is only validated for the specific trace distributions tested.

The SWE-Bench result (34.4× improvement, Figure 7c) is the headline number but also the most extreme—it's the workload where fine-grained checkpointing fails most catastrophically due to wide sequence length variation. An operator whose workload doesn't have this extreme variance (e.g., homogeneous batch inference) might see much smaller improvements, but the paper provides no data to characterize when Marconi's benefits diminish to the point of being not worth the implementation complexity.

**What evidence exists.** The paper explicitly describes the three traces (Section 5.1, Figure 6) and acknowledges that the taxonomy was derived from "extensive analysis of prefix reusing patterns in various real-world datasets and request traces" (Section 4.1), citing Qin et al., 2024; Jimenez et al., 2023; cha, 2024; Zheng et al., 2023b. However, the taxonomy derivation analysis is not presented—no numbers, tables, or figures show *how* the authors classified reuse patterns into the two categories or what fraction of observed reuse fell into each. The reader must take on faith that the two-category taxonomy is exhaustive for the cited traces. The microbenchmarks (Figures 11–13) vary parameters within the same trace distributions, testing robustness to *parametric variation* (contention level, arrival rate) but not to *structural variation* (different workload types, different reuse patterns).

**Mitigation status.** The paper does not discuss scope limitations regarding workload type. The taxonomy is presented as a general finding derived from analysis of "various real-world... request traces," but the analysis itself is not shown and the evaluation is limited to three traces that align with the taxonomy. The paper does not propose future work on workload characterization or adaptive admission that could detect non-taxonomy reuse patterns at runtime.

---

### 6.6 The Bootstrap α-Tuning Procedure Is a Black Box with No Characterization of Its Effectiveness or Robustness

The FLOP-aware eviction policy's balancing coefficient α is set by a bootstrap procedure that replays recent request history through a grid search to find the α maximizing simulated hit rate (Section 4.2, "Managing the balance"). The paper provides **zero evaluation** of this procedure: no measurements of how much the tuned α varies across workloads, no comparison of bootstrap-tuned α against a fixed default (e.g., α = 1), no oracle comparison against the α that maximizes hit rate on the full trace, no sensitivity analysis of the bootstrap multiplier (5–15× pre-first-eviction requests), and no measurement of how quickly α should be retuned as workload characteristics shift.

**Consequence.** A practitioner cannot determine whether the bootstrap procedure is essential or incidental. If a fixed α = 1 works nearly as well as the bootstrap-tuned value across all workloads, then the bootstrap adds engineering complexity for minimal gain. If tuning is critical but the bootstrap multiplier must be carefully chosen (too short → noisy α; too long → many requests served with suboptimal LRU during bootstrap), then the paper provides no guidance. If α needs to be retuned when workload distributions shift (e.g., diurnal patterns in chatbot usage, deployment of a new model version with different output-length characteristics), the paper provides no mechanism or schedule for triggering re-tuning—and the static α would become stale, degrading eviction quality over time.

**What evidence exists.** The paper describes the bootstrap procedure algorithmically (Section 4.2) but **provides no experiments that evaluate it**. The FLOP-aware eviction results (Figures 8, 10, 11) are attributed to the eviction policy itself, but it's impossible to determine from the reported data whether the gains come from (a) having any FLOP efficiency term in the utility function, (b) having the *correct* α weight for the workload, or (c) having the bootstrap procedure specifically (as opposed to some other tuning method). Figure 11 shows FLOP-aware eviction gains across cache sizes, but doesn't report what α values were selected or how they varied. The latency results (Figure 9) likewise provide no visibility into α.

**Mitigation status.** The paper does not acknowledge this as a limitation. The bootstrap procedure is presented as a feature ("Marconi manages the balance by observing the workload and retrospectively setting the best configuration"), but its contribution to the overall system performance is unevaluated. The artifact appendix (Appendix B) mentions an "offline-optimal, static-α oracle policy" as an additional eviction variant implemented in the codebase but not included in the paper's results, suggesting the authors considered but did not report an oracle comparison. This is a missed opportunity to characterize the bootstrap's efficiency relative to the optimal α.

## 7. Implications and Future Directions
- Field impact
  - Makes prefix caching viable for Hybrid LLMs, removing a major deployment barrier for SSM-heavy models that are otherwise more efficient at long context (Figures 1c and 5). This can accelerate adoption of Hybrid architectures in production by improving both average and tail latencies (Figures 9–10).

- Practical applications
  - Conversational assistants and helpdesk agents with long system prompts and multi-turn histories.
  - Coding/agent systems (e.g., SWE-Bench scenarios) that repeatedly touch long context and benefit from reusing long prefixes.
  - Any service with template-heavy or prompt-engineered workloads where large parts of the input are shared.

- Suggested follow-ups
  - Adaptive α beyond grid search: reinforcement learning or bandit tuning that responds to non-stationary traffic.
  - Hierarchical or multi-tier caching (GPU/CPU/NVMe) informed by FLOP efficiency, combining Marconi’s policies with systems like CachedAttention and Pensieve.
  - Cluster-level routing that steers requests to GPUs holding high-FLOP-efficiency prefixes, integrating with Preble-style schedulers.
  - Broader support for SSM variants and generalized recurrent layers (Section 6) with kernels that expose chunked state boundaries, further reducing prefill overhead.
  - Proactive pre-seeding of hot purely-input prefixes at service startup using the radix tree abstraction.

Overall, Marconi reframes prefix caching for Hybrid LLMs as a joint problem of selective SSM admission and FLOP-aware eviction, validated across workloads with large, realistic variance in context lengths. By recognizing the unique “all-or-nothing” reuse of SSM states and quantifying compute-per-byte value, it delivers substantial and explainable gains in hit rate and TTFT.
