# MARCONI: PREFIX CACHING FOR THE ERA OF HYBRID LLMS

**ArXiv:** [2411.19379](https://arxiv.org/abs/2411.19379)

## 🎯 Pitch

Marconi introduces the first practical prefix caching system tailored for Hybrid LLMs, which combine Attention and recurrent SSM layers. By devising novel admission and FLOP-aware eviction policies that accommodate the in-place, 'all-or-nothing' state management of SSMs, Marconi dramatically increases cache utility—achieving up to 34.4× higher token hit rates and up to 71.1% lower time-to-first-token compared to state-of-the-art systems. This innovation removes a key bottleneck for efficiently serving long-context Hybrid LLM workloads, unlocking scalable, fast, and memory-efficient inference for next-generation language models.

---

## 1. Executive Summary
Marconi is a prefix-caching system designed specifically for Hybrid LLMs that mix Attention layers with recurrent State Space Model (SSM) layers. It introduces new admission and eviction policies that make caching practical and effective despite SSMs’ in-place state updates, yielding up to 34.4× higher token hit rates and up to 71.1% lower P95 time-to-first-token (TTFT) compared with extended versions of state-of-the-art systems (Figures 7–9).

## 2. Context and Motivation
- Problem addressed
  - Reusing computation across requests via prefix caching is essential for serving long-context LLM workloads (Section 2.2). Prior systems work well for Transformer-only models by reusing `KV` (key-value) caches from Attention layers.
  - Hybrid models combine Attention and SSM layers to improve efficiency for long contexts (Figure 1a; Section 2.1). SSMs maintain a compact recurrent state updated in place (Figure 1b), so the state after processing a sequence cannot be rolled back to represent a prefix.
  - This “all-or-nothing” property means reuse only happens on an exact match of the SSM state for the entire prefix. To support arbitrary prefix reuse, existing cache designs must checkpoint SSM states at many positions (fine-grained), which explodes memory and yields sparse reuse (Section 3).

- Why it matters
  - Long contexts are increasingly common in real deployments—few-shot prompting, chain-of-thought, multi-step agents, and large system prompts (Introduction; Section 2.2).
  - Hybrid models substantially reduce per-request compute and memory for long contexts versus pure Transformers (Figure 1c), but without a compatible prefix caching method, cross-request efficiency is lost.

- Where prior approaches fall short
  - Transformer-focused systems (e.g., vLLM, SGLang) depend on slicing KV caches by token position. With SSMs, the last state cannot be sliced to a shorter prefix, so they rely on frequent checkpointing (Section 3), creating:
    - Cache underutilization: many SSM states never hit (Figure 3a: with token block size 32, 25.0% of blocks’ KVs are reused vs only 0.4% of SSM states).
    - High memory pressure: even a single 10K-token sequence of a 7B Hybrid model can occupy 17.4 GB with moderate block sizes (Figure 3b), 3.3× a Transformer of equal size.

- Positioning
  - Marconi is the first system to make prefix caching practical for Hybrid models by:
    - Admitting only high-utility SSM states guided by a simple reuse taxonomy and a radix-tree index (Section 4.1; Figure 4).
    - Evicting with a FLOP-aware metric that considers compute saved per byte alongside recency (Section 4.2; Equation 2; Appendix A and Table 1).

## 3. Technical Approach
Marconi manages both KVs (Attention) and SSM states together, ensuring any cached prefix has all layer states needed for reuse (Section 4; Figure 4). The core design has two parts: judicious admission and FLOP-aware eviction.

- Unifying cache representation with a radix tree (Section 4; Figure 4)
  - Data structure: a radix tree maps sequences to cached states while compactly encoding shared prefixes. Edges carry KVs for the tokens in that edge and SSM states representing all tokens up to the edge’s last token (Figure 4c).
  - Rationale: For a cache hit to skip prefill, both Attention and SSM layers must be synchronized at the same prefix. Managing them together avoids mismatches (Section 4).

- Admission policy: “judicious” checkpointing (Section 4.1)
  - Reuse taxonomy (Section 4.1)
    - Purely-input reuse: shared system prompts and instruction preambles across many requests.
    - Input-and-output reuse: conversations or agent trajectories that resume from the last generated token of the previous turn.
  - How Marconi estimates reuse likelihood
    - Always checkpoint at the last decoded token of a sequence (covers input-and-output reuse between rounds).
    - For purely-input reuse, perform a speculative insertion of the next request’s input tokens into the radix tree before prefill (Figure 4a→4b). If this insertion creates an intermediate branch point, checkpoint the SSM state at that branch during prefill (Figure 4c).
    - Effect: Only up to two SSM states per sequence are admitted (branch point and last decoded token), rather than per-block or per-token states, massively reducing SSM-state clutter (Section 4.1).
  - How SSM states are actually materialized during prefill (Section 4.1, “Obtaining states during prefill”)
    - If the model supports chunked state passing (e.g., Mamba2-style kernels), Marconi saves the state at the boundary of the second-to-last chunk in the prefix. Optionally, a small forward roll can reach an exact token.
    - If not supported, Marconi does a lightweight two-pass prefill: pass 1 to reach the prefix state, pass 2 to continue from that checkpoint to the sequence end.
  - Trade-off (Section 4.1)
    - Purely-input prefixes start yielding reuse from their third occurrence (the second occurrence is used to discover and checkpoint the branch). This sacrifices a small one-time saving for dramatically higher overall cache utility and lower memory pressure.

- Eviction policy: FLOP-aware utility (Section 4.2; Equation 2; Appendix A and Table 1)
  - Problem: In Hybrid models, KV size grows with prefix length and is a reasonable proxy for compute saved; SSM state size is constant per layer regardless of prefix length. Size-only or recency-only policies mis-rank entries (Section 4.2).
  - FLOP efficiency metric (Equation 1; Appendix A): compute saved if a prefix is reused divided by the memory consumed for its states, aggregated across layers. Figure 5 shows FLOP efficiency grows more steeply for models with more SSM layers as sequence length increases.
  - Utility score per node n (Equation 2):
    - S(n) = recency(n) + α · flop_efficiency(n)
    - Both components are normalized to [0,1] over current nodes.
    - α balances recency vs. compute-per-byte. α=0 reduces to LRU.
  - Tuning α online (Section 4.2 “Managing the balance”)
    - Start with α=0 until first eviction.
    - Bootstrap by logging ~5–15× more requests than seen before that eviction; asynchronously grid-search α by replaying these requests on CPU cores; adopt α with highest hit rate.
  - Implementation detail: when evicting, nodes with ≤1 child are eligible (Section 4.3). Evicting an intermediate node drops its SSM states and absorbs its KVs into the child, preserving reuse continuity. On cache hits, only the accessed node’s timestamp is updated (ancestors’ timestamps are not) because their SSM states are not reused (Section 4.3).

- Why this design over alternatives
  - Fine-grained checkpointing (baseline) creates many large SSM entries with very low reuse (Figure 3a) and rapidly exceeds memory even for single long sequences (Figure 3b).
  - Size-aware evictions (e.g., GDSF-style) fail in Hybrid settings because SSM states’ size does not reflect the compute they can save; FLOP efficiency corrects this (Section 4.2, “Comparisons with existing size-based eviction algorithms”).

- Example walk-through (Figure 4)
  - Suppose incoming sequence “NYC is very huge.”
  - Speculative insertion shows a branch point after “NYC is” because another sequence “NYC is a busy city” exists. Marconi checkpoints SSM state at “NYC is” during prefill and always checkpoints the last decoded token state after finishing decode. This creates high-utility nodes with minimal SSM admissions.

## 4. Key Insights and Innovations
- Judicious SSM admission guided by a reuse taxonomy (Section 4.1; Figure 4)
  - Novelty: The system distinguishes reuse patterns—purely-input vs input-and-output—and only inserts SSM checkpoints where future reuse is likely.
  - Why it matters: Prevents floods of large, low-hit SSM states that thrash the cache (Figure 3a), while still enabling immediate reuse for conversational turns.

- FLOP-aware eviction for Hybrid models (Section 4.2; Equation 2; Appendix A; Figure 5)
  - Novelty: A utility metric that combines normalized recency with normalized compute-per-byte savings estimated across layer types.
  - Why it matters: Preferentially keeps long, expensive prefixes that yield outsized compute savings, improving tail latency and overall throughput (Figures 8–10).

- Unified cache for KVs and SSM states in a radix tree (Section 4; Figure 4)
  - Novelty: Co-manages different state types so cached prefixes are usable by all layers simultaneously; evicts intermediate nodes intelligently by absorbing KVs downward (Section 4.3).
  - Why it matters: Ensures correctness of reuse and avoids stale or unusable prefixes.

- Practical state acquisition during prefill (Section 4.1)
  - Novelty: Two concrete methods—chunked checkpointing or two-pass prefill—to obtain exact SSM states at needed boundaries without intrusive model changes.
  - Why it matters: Makes the approach implementable across diverse SSM variants and serving stacks.

These are fundamental system-level innovations rather than incremental tweaks: they redefine how prefix caching is done for models whose recurrent states cannot be rolled back.

## 5. Experimental Analysis
- Evaluation setup (Section 5.1)
  - Datasets/workloads (Figure 6):
    - `LMSys-Chat` conversations: longer generated outputs, many up to thousands of tokens.
    - `ShareGPT` conversations: shorter output sequences (tens to hundreds).
    - `SWE-Bench` with SWE-Agent: agentic, multi-turn software issue resolution; very wide input lengths (hundreds to tens of thousands of tokens).
  - Metrics:
    - Token hit rate: fraction of input tokens that skip prefill (proxy for compute saved).
    - TTFT (ms), percentiles P5, P50, P95.
  - Baselines:
    - `vLLM+`: extended with Hybrid support; fine-grained token-block checkpointing with block size 32 (largest their system supports; favors vLLM+; Section 5.1).
    - `SGLang+`: radix-tree mapping but LRU eviction; enhanced with Marconi’s judicious admission to isolate eviction effects (Section 5.1).
    - `Vanilla inference`: no prefix caching.
  - Models and hardware:
    - Main results: 7B Hybrid with {4 Attention, 24 SSM, 28 MLP} layers (Section 5.1).
    - TTFT insights include Jamba-1.5-Mini (Hybrid, 12B active/52B total), 4×A100-40GB (Section 5.1).
    - All FP16; experiments on 8×A100-40GB (Section 5.1).

- Main quantitative results
  - Token hit rate vs vLLM+ (Figure 7):
    - LMSys: 4.5× average improvement.
    - ShareGPT: 7.3× average improvement.
    - SWE-Bench: 34.4× average improvement.
    - These gains derive from admitting far fewer but higher-utility SSM states (Section 5.2).
  - Token hit rate vs SGLang+ (eviction-only effect; Figure 8):
    - P95 wins: 45.6% (LMSys), 19.0% (ShareGPT), 219.7% (SWE-Bench).
    - FLOP-aware eviction focuses cache on long, expensive prefixes; effect is largest when input lengths vary widely (SWE-Bench; Figure 6c).
  - TTFT reductions (Figure 9):
    - Relative to no caching: up to 36.9%, 73.2%, and 46.8% P95 TTFT reduction for LMSys, ShareGPT, SWE-Bench, corresponding to 281.4 ms, 106.3 ms, and 617.0 ms.
    - Over vLLM+: up to 36.1%, 71.1%, and 46.8% larger P95 TTFT reductions (275.4 ms, 103.3 ms, 617.0 ms).
    - Over SGLang+: up to 17.2%, 12.8%, and 24.7% (131.1 ms, 18.5 ms, 325.7 ms).

- Fine-grained analysis of FLOP-aware eviction (Section 5.3; Figure 10)
  - Trade-off across sequence lengths (Figure 10a):
    - For short sequences (<7K tokens), Marconi’s hit rate can be slightly lower (up to −3.0%) than LRU, because it prioritizes keeping long, FLOP-heavy prefixes.
    - For long sequences (>7K tokens), Marconi improves hit rate by up to +25.5%.
    - Net effect: overall 90.3% more FLOP saved than LRU on the SWE-Bench trace.
  - TTFT distribution (Figure 10b):
    - Slight P5 degradation vs LRU (6.3%), but absolute loss is ~2.1 ms because short sequences prefill quickly.
    - Larger P50 and P95 improvements: −13.4% and −22.0% (−74.2 ms and −274.9 ms).

- Microbenchmarks and ablations (Section 5.4)
  - Cache contention (Figure 11):
    - Marconi outperforms LRU by 24.3%–68.3% in hit rate, with the biggest gains under moderate contention where eviction choices matter most.
  - Layer composition (Figure 12a):
    - The more SSM-heavy the model (e.g., Attention:SSM = 1:8), the larger the benefit; serving a pure Transformer yields parity among systems.
  - SSM state dimension (Figure 12b):
    - As SSM state dimension N grows (e.g., 16→128), Marconi’s advantage over fine-grained checkpointing grows sharply (e.g., 5.7×→35.4× vs vLLM+), because larger SSM states amplify memory pressure for naive policies.
  - Arrival patterns (Figure 13):
    - More sessions per second or longer intra-session delays reduce absolute hit rates (reuses are sparser), but Marconi’s relative advantage grows due to higher inter-session contention.

- Convincingness
  - The evaluation spans three workload types, realistic request timing, and multiple cache sizes; it isolates admission (vs vLLM+) and eviction (vs SGLang+) contributions.
  - FLOP-efficiency rationale is grounded in a per-layer analysis (Appendix A, Table 1) and reflected empirically (Figure 5 and Figures 8–10).

- Representative quotes
  - > “Marconi improves token hit rates by an average of 4.5–34.4× … [and reduces] P95 TTFT by up to 71.1% (617.0 ms) compared to baseline prefix caching systems.” (Section 5.2; Figures 7–9)
  - > “Marconi achieves a higher hit rate for longer sequences while sacrificing the hit rate for some shorter sequences… [P50, P95] TTFT reduced by 13.4% and 22.0%.” (Section 5.3; Figure 10)

## 6. Limitations and Trade-offs
- Assumptions about reuse patterns (Section 4.1)
  - The admission policy presumes that conversational/agent traffic typically resumes from the last decoded token and that purely-input prefixes recur across many requests. Workloads with highly branching histories at arbitrary midpoints may benefit less.

- Missed savings on second occurrence of purely-input prefixes (Section 4.1, “Tradeoffs”)
  - Purely-input reuse begins from the third occurrence because the second is used to identify and checkpoint the branch. This is deliberate to avoid flooding the cache with low-utility SSM states.

- Overheads to obtain exact SSM states (Section 4.1)
  - Models without chunked state passing require a two-pass prefill to materialize a precise prefix state. While lightweight, it adds some overhead and requires engineering in serving frameworks.

- Tuning α (Section 4.2)
  - Marconi uses an online bootstrap and grid search to set α. While fast (seconds) and parallelized on CPUs, it is heuristic and depends on stationarity of recent traffic.

- Scope of evaluation
  - The system targets exact reuse. It does not attempt approximate reuse or cross-model reuse. It also does not cover distributed cluster-level routing to co-locate requests with caches (though complementary to systems like Preble).

- Interactions with other optimizations
  - Chunked prefill kernels and paged KV memory management are still needed (Section 6). Marconi focuses on caching policy; it relies on underlying engines to provide efficient per-layer kernels.

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
