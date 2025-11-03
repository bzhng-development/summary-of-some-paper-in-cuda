# MARCONI: PREFIX CACHING FOR THE ERA OF HYBRID LLMS

**ArXiv:** [2411.19379](https://arxiv.org/abs/2411.19379)

## 🎯 Pitch

MARCONI pioneers the first practical prefix caching system tailored for Hybrid LLMs, which combine Attention and recurrent State Space Model (SSM) layers. By introducing novel cache admission (judiciously storing only likely-to-be-reused states) and FLOP-aware eviction (prioritizing cache entries by compute savings, not just recency), MARCONI unlocks large efficiency gains in long-context applications—achieving up to 34.4× higher token hit rates and up to 71% lower time-to-first-token compared to state-of-the-art systems. This breakthrough makes scalable, low-latency deployment of Hybrid LLMs feasible, addressing a core bottleneck for next-generation AI services.

---

## 1. Executive Summary (2–3 sentences)
Marconi is a caching system that makes “prefix caching” practical for Hybrid large language models (LLMs) that combine Attention layers with recurrent State Space Model (SSM) layers. It introduces two core ideas—judicious cache admission and FLOP-aware eviction—that together turn the unique, difficult-to-cache state structure of SSMs into large, reliable latency savings, yielding up to 34.4× higher token hit rate and up to 71.1% lower P95 time-to-first-token (TTFT) compared to extended state-of-the-art systems (Fig. 7, Fig. 9).

## 2. Context and Motivation
- Problem addressed
  - Many LLM applications (chat, search, coding assistants) reuse long, shared prefixes across requests (e.g., system prompts, conversation history). Recomputing these prefixes wastes time and compute. Prefix caching avoids this by storing and reusing model states computed for common prefixes (Fig. 2).
  - Hybrid LLMs mix Attention and SSM layers to make long-context inference efficient (Fig. 1a, 1c), but their SSM layers update state in place, making standard prefix caching ineffective because you cannot “roll back” an SSM state to represent a shorter prefix (Fig. 1b; §3, “SSM State Properties 1–3”).

- Why it matters
  - Long contexts are now common (few-shot prompting, chain-of-thought, multi-turn chats), but serving costs and latency grow dramatically if prefixes are recomputed (§1, §2.2).
  - Hybrid models are gaining adoption for speed and memory efficiency at long context, yet cross-request efficiency (cache reuse) remains underexploited in them (§1).

- Where prior approaches fall short
  - Existing caching systems (e.g., vLLM, SGLang) were built for Transformers’ KV caches. They typically checkpoint every token block and rely on LRU-like recency for eviction (§2.2, §5.1).
  - With SSMs, fine-grained checkpointing floods the cache with many large state entries that are rarely reused:
    - With block size 32, 25.0% of KV blocks are reused, but only 0.4% of SSM states are reused (65.3× gap; Fig. 3a).
    - A single 10k-token sequence can consume 17.4 GB in a 7B Hybrid model with block size 16, 3.3× more than a comparable Transformer (Fig. 3b).
  - Result: sparse reuse, cache thrashing, poor trade-off between memory and compute savings (§3).

- Positioning
  - Marconi is the first system designed specifically to support prefix caching for Hybrid LLMs (§1). It rethinks both admission (what to cache) and eviction (what to keep) by modeling reuse likelihood and compute-per-byte savings across Attention and SSM layers (§4).

## 3. Technical Approach
Marconi’s design centers on two components: how to decide which states to admit into the cache (admission) and how to decide which ones to evict (eviction). It uses a radix tree to track sequence overlap and manage heterogeneous state types (KVs and SSM states) coherently.

A. Core data structure: `radix tree` (§4, Fig. 4)
- What it is: A compact prefix tree whose edges are labeled by token subsequences of variable length.
- How it’s used:
  - Each edge stores:
    - Attention KVs for the tokens on that edge.
    - The SSM state representing all tokens prior to the last token on that edge (Fig. 4c).
  - Each node represents a branch point in the sequence history:
    - Nodes with multiple children imply “purely input” prefixes that many requests share.
    - Nodes with ≤1 child are either linear continuations or potential boundaries of future continuation (§4.1, §4.3).

B. Admission: judiciously checkpoint only high-value SSM states (§4.1; Fig. 4)
- Problem to solve: SSM states are large, fixed-size, and cannot be rolled back; naive per-block checkpointing wastes memory and yields low reuse (Fig. 3).
- Key insight: Prefixes that get reused fall into two categories:
  1) “Purely input” prefixes: shared prompts/instructions across requests (e.g., a system prompt).
  2) “Input + output” prefixes: conversation histories where new messages append to the end (§4.1).
- Policy:
  - Always checkpoint the SSM state at the last decoded token (supports “input + output” reuse; typical for continuing a conversation).
  - For “purely input” prefixes, checkpoint only when there is clear evidence they are common:
    - Before prefilling a new request, perform a `speculative insertion` of the input tokens into the radix tree. If this would create a new intermediate node (a branch point), then checkpoint the SSM state at that branch point during prefill (Fig. 4a–c).
- How states are obtained during prefill (§4.1):
  - For SSMs that support `chunked state passing`, materialize the state at chunk boundaries (e.g., checkpoint the state of the second-to-last chunk; optionally roll forward a few tokens with a lightweight kernel to target an exact position).
  - For SSMs without chunking, run a `two-pass prefill`: first pass to the prefix to capture the exact SSM state, second pass to complete the sequence.

- Trade-off acknowledged (§4.1 “Tradeoffs”):
  - “Purely input” prefixes only start benefiting from reuse on their third occurrence (the second occurrence is used to identify and checkpoint the node).
  - In exchange, Marconi avoids flooding the cache with low-utility SSM states, improving overall cache utility.

C. Eviction: `FLOP-aware` policy that balances recency with compute saved per byte (§4.2; Eq. (1), Eq. (2), Fig. 5; Table 1)
- Motivation: KVs’ size scales with sequence length, but SSM states are fixed-size regardless of prefix length; longer prefixes save significantly more compute but don’t “look bigger” in SSM memory. A size- or recency-only policy can evict high-value, long-prefix entries (§4.2).
- Define `FLOP efficiency` (Eq. (1)): the total floating-point operations (FLOPs) saved by reusing a prefix entry divided by the memory (bytes) consumed by its states across all stateful layers (Attention KVs + SSM states).
  - Table 1 provides per-layer FLOP and state-size formulas; SSM layers’ “FLOPs saved per byte” grows faster with sequence length than Attention.
  - Fig. 5 shows FLOP efficiency increases more steeply for models with more SSM layers (Mamba > Hybrid > Transformer), strengthening the case for FLOP-aware eviction.
- Eviction score (Eq. (2)): `S(n) = recency(n) + α · flop_efficiency(n)`, with both terms normalized to (0, 1).
  - α tunes the balance; α=0 reduces to pure LRU.
  - Parent/child savings are computed relative to each other to avoid double-counting.
- How α is set (§4.2 “Managing the balance”):
  - Start with α=0. After the first eviction, collect a bootstrap trace (5–15× the number of requests before that eviction).
  - Run a parallel grid search over α on CPUs by replaying the bootstrap trace to pick the α that maximizes token hit rate; switch to that α online.

D. Additional implementation details (§4.3)
- Eviction candidates include all nodes with ≤1 child, not just leaves. Multi-child nodes represent widely shared “purely input” prefixes and are only evicted after their children are gone (then they become evictable single-child/leafless nodes).
- On cache hit, only the accessed node’s timestamp is refreshed; ancestors’ timestamps are not updated since their KVs, if evicted, are subsumed by child nodes (Fig. 4c).

Putting it together (simplified flow):
1) Lookup: find the longest matching prefix in the radix tree that has both KVs (for all Attention layers) and exactly matching SSM states for that prefix.
2) Before prefill: run speculative insertion of the new input to decide if an intermediate node will emerge; if yes, plan to checkpoint that SSM state during prefill; always plan to checkpoint the final decoded token’s SSM state.
3) Prefill and decode: compute states; materialize checkpoints as decided (chunked or two-pass); store edge/node entries.
4) If space needed: evict nodes with the lowest `S(n)` until enough memory is available.

## 4. Key Insights and Innovations
- Judicious SSM admission via reuse taxonomy and speculative insertion (fundamental)
  - Different reuse patterns call for different cache decisions: “purely input” vs “input + output” (§4.1). Only two SSM states per request are admitted at most: at the last decoded token (always) and at a speculative branch point (if identified).
  - This directly tackles the core SSM challenge—large, fixed-size, non-rollbackable states—preventing cache flooding by low-utility checkpoints (contrast Fig. 3a–b).

- FLOP-aware eviction that values compute saved per byte (fundamental)
  - Equation (1) defines FLOP efficiency; Equation (2) merges it with recency into a single utility.
  - This reframes eviction from “what was used recently” to “what yields the most saved compute for the space it occupies,” which is essential when SSM state size does not reflect saved compute (Fig. 5, Table 1).

- Unified, state-type-aware cache over a radix tree (incremental but enabling)
  - Managing KVs and SSM states together ensures a cache hit is “all or nothing”—all layers have consistent states for the same prefix (Fig. 4; §4).
  - Evicting an intermediate node drops its SSM state but merges its KV span into the child, minimizing fragmentation (§4.3).

- Practical state capture for SSMs during prefill (incremental)
  - Supports both chunked state passing and a two-pass prefill to materialize precise SSM states at needed boundaries (§4.1).

## 5. Experimental Analysis
- Evaluation setup (§5.1)
  - Datasets/workloads:
    - LMSys-Chat1M (multi-turn chat, long outputs) and ShareGPT (shorter outputs) with different input/output length distributions (Fig. 6a–b).
    - SWE-Bench via SWE-Agent (software-issue solving; wide input-length range; Fig. 6c).
  - Models:
    - Main: 7B Hybrid with {4 Attention, 24 SSM, 28 MLP} layers (§5.1).
    - TTFT analysis: Jamba‑1.5‑Mini (12B active/52B total), served on 4×A100‑40GB (§5.1).
  - Hardware: 8×A100‑40GB, 96 CPU cores, 1.1 TB RAM (§5.1).
  - Baselines (§5.1):
    - Vanilla (no caching).
    - vLLM+ (fine-grained token blocks; block size 32).
    - SGLang+ (uses the same admission as Marconi but eviction is LRU).
  - Metrics:
    - `Token hit rate` = fraction of input tokens whose prefill is skipped (proxy for total saved FLOPs).
    - TTFT percentiles (P5/P50/P95).

- End-to-end results (Fig. 7, Fig. 8, Fig. 9)
  - Against vLLM+ (fine-grained checkpointing):
    > “Marconi improves the token hit rate by an average of 4.5× (LMSys), 7.3× (ShareGPT), and 34.4× (SWE‑Bench)” (Fig. 7).
  - TTFT improvements:
    > Relative to no caching, P95 TTFT is reduced by up to 36.9% (281.4 ms) on LMSys, 73.2% (106.3 ms) on ShareGPT, and 46.8% (617.0 ms) on SWE‑Bench (Fig. 9).
    > Relative to vLLM+, Marconi reduces P95 TTFT by up to 36.1% (275.4 ms), 71.1% (103.3 ms), and 46.8% (617.0 ms) on the three datasets respectively (Fig. 9).
  - Against SGLang+ (to isolate the eviction effect):
    > Marconi’s FLOP-aware eviction yields up to 219.7% higher token hit rate on SWE‑Bench (P95 improvement; Fig. 8), and noticeable wins on LMSys (P95 45.6%) and ShareGPT (P95 19.0%).

- Why FLOP-aware eviction matters (fine-grained view; §5.3; Fig. 10)
  - On a SWE‑Bench trace:
    > SGLang+ reaches 16.4% overall token hit rate; Marconi reaches 32.7% (+99.4%).
  - Trade-off across sequence length (Fig. 10a):
    - For short inputs (<7k tokens), Marconi has up to −3.0% lower hit rate than LRU.
    - For long inputs (>7k tokens), Marconi is up to +25.5% better.
    - This deliberate trade favors long sequences where savings are largest; overall saved FLOPs are +90.3% compared to LRU (§5.3).
  - TTFT distribution (Fig. 10b):
    - Slight P5 degradation: +6.3% vs LRU, but only ~2.1 ms absolute.
    - Stronger tail wins: P50 −13.4% (−74.2 ms) and P95 −22.0% (−274.9 ms).

- Microbenchmarks and ablations (§5.4)
  - Cache contention (Fig. 11):
    - Gains are largest under moderate contention (e.g., +68.3% hit rate over LRU at 100 GB). When cache is too small, few prefixes fit; when too large, both policies can keep more entries.
  - Layer composition (Fig. 12a):
    - As Attention:SSM ratio shifts from 1:2 to 1:8, Marconi’s advantage grows—from +13.5% / +5.8% (over vLLM+/SGLang+) to 2.6× / +59.7%. With pure Transformers, all systems are similar.
  - SSM state dimension (Fig. 12b):
    - With larger SSM states (N from 16 to 128), Marconi’s improvement over vLLM+ grows from 5.7× to 35.4× and over LRU from 1.6–1.9× to ~9.6–19.9×.
  - Arrival patterns (Fig. 13):
    - More concurrent sessions per second and longer inter-request gaps both reduce absolute reuse (lower hit rates for everyone) but increase Marconi’s relative advantage due to higher contention.

- Do the experiments support the claims?
  - The results isolate admission (vs vLLM+) and eviction (vs SGLang+) effects, and they probe many axes: cache size, sequence-length mix, layer ratios, SSM state size, and traffic timing (§5.2–§5.4).
  - The analysis is consistent with the mechanism:
    - Admission reduces low-utility SSM entries (addressing Fig. 3).
    - FLOP-aware eviction prioritizes long, compute-heavy prefixes (consistent with Fig. 5, Table 1, and Fig. 10).

- Additional evidence of the underlying pain point (§3)
  - Fine-grained checkpointing creates many SSM checkpoints that are rarely reused (Fig. 3a) and can overwhelm memory even per single sequence (Fig. 3b). This directly motivates Marconi’s selective admission.

## 6. Limitations and Trade-offs
- Reuse detection latency for “purely input” prefixes (§4.1 “Tradeoffs”)
  - Benefits begin on the third occurrence (first occurrence creates the path; second identifies it as a branch to checkpoint). This delays savings for one request but avoids admitting many low-value SSM states.
- State capture overhead for SSMs without chunking (§4.1)
  - Two-pass prefill adds overhead to obtain exact SSM state at a branch point. The paper notes optional kernels can accelerate small roll-forwards but does not quantify this overhead end-to-end.
- Tailoring to Hybrid/SSM properties (§4.2, Table 1)
  - FLOP efficiency formulas and savings depend on model architecture details (e.g., layer counts, dimensions). The approach assumes these are known and stable enough to guide eviction.
- α tuning sensitivity (§4.2)
  - α is chosen by grid search over a bootstrap window. If workload characteristics shift quickly, α may lag until the next retuning. The paper does not describe continuous or online adaptation beyond the initial bootstrap.
- Exact-match reuse only
  - As with standard prefix caching, reuse requires exact prefix matches. Paraphrased prompts or semantically similar contexts are out-of-scope (the system avoids approximate reuse for accuracy reasons).
- Scope of evaluation
  - Results emphasize token hit rate and TTFT; throughput improvements (prefill tokens/s under high concurrency) are argued but not deeply quantified beyond the connection that prefill savings lower tail TPT (§2.2, note 2).
- Integration with distributed scheduling
  - Cluster-level routing (e.g., sending requests to the server holding the longest prefix) is not addressed; complementary systems like Preble target that dimension.

## 7. Implications and Future Directions
- How this changes the landscape
  - Hybrid LLMs rely on recurrent states to scale context efficiently, but those same states thwart traditional caching. Marconi demonstrates a principled path—admission guided by reuse structure plus eviction guided by FLOP-per-byte—to unlock cross-request efficiency in these models.
  - The FLOP-aware lens (Eq. (1), Eq. (2); Fig. 5) reorients cache policy design from size and recency to compute-per-memory payoff, which is broadly applicable to any system that mixes state types with different memory/compute trade-offs.

- Practical applications
  - Multi-turn chat and tool-using agents (e.g., coding assistants) where interactions append at the end particularly benefit from always checkpointing the last decoded token (§4.1). Shared “system prompts” benefit after being observed as branch points via speculative insertion (Fig. 4).
  - Workloads with long or highly skewed input lengths (e.g., SWE‑Bench; Fig. 6c) see large gains from FLOP-aware eviction (Fig. 8, Fig. 10).

- Follow-up research and engineering
  - Online/adaptive α tuning that continuously tracks workload drift without bootstrap windows.
  - Extending the FLOP-aware principle to distributed caching and cluster routing, combining Marconi with systems that place requests where the longest prefix resides.
  - Broader coverage of recurrent architectures beyond Mamba-style SSMs (§6 Related Work), especially those with different state update rules; the core “fixed-size, in-place state” constraints will persist, but FLOP/state-size models may need to be recalibrated.
  - Combining with hierarchical caches (GPU/CPU/SSD) where eviction decisions should account for medium costs and transfer time.
  - Kernel support to roll SSM states forward a small number of tokens cheaply, reducing the need for two-pass prefill.

Overall, Marconi reframes prefix caching for Hybrid LLMs as a selective, value-aware problem. The combination of speculative, evidence-based admission and FLOP-normalized eviction delivers consistent end-to-end benefits (Fig. 7–9), particularly in long-context and SSM-heavy regimes (Fig. 12), and provides a template for future caching systems targeting heterogeneous model states.
