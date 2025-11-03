# SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification

**ArXiv:** [2305.09781](https://arxiv.org/abs/2305.09781)

## 🎯 Pitch

SpecInfer introduces a fundamentally new approach to speeding up large language model (LLM) serving by using tree-based speculative inference, where multiple continuations are proposed by small, efficient models and then verified all at once via a novel tree-parallel decoding strategy. This innovation breaks through sequential and memory bottlenecks of conventional and prior speculative decoding methods, enabling much higher throughput and lower latency—achieving up to 2.8× faster distributed inference—while provably maintaining LLM generation quality. By making LLMs dramatically more efficient to serve at scale or on resource-limited hardware, SpecInfer has the potential to lower costs and unlock new real-time, high-volume LLM applications.

---

## 1. Executive Summary (2-3 sentences)
SpecInfer accelerates large language model (LLM) serving by replacing single-path speculative decoding with a tree of candidate continuations produced by small auxiliary models and verified in one shot by the LLM using a new tree-parallel decoding mechanism. This reduces sequential bottlenecks and memory traffic without changing generation quality (for greedy and stochastic decoding), yielding 1.5–2.8× speedups for distributed inference and 2.6–3.5× for offloading-based inference (Abstract; Fig. 7–8).

## 2. Context and Motivation
- Problem addressed
  - Autoregressive LLMs generate tokens sequentially; each new token depends on all previous tokens. Existing serving stacks therefore “incrementally decode” one token at a time (Fig. 1a; Alg. 1).
  - This creates two major bottlenecks:
    - Limited parallelism within a single request → low GPU utilization in the decoding phase.
    - High memory traffic to fetch model parameters and maintain the key-value cache (`KV-cache`) for attention over long contexts. Long contexts further limit how many requests can run concurrently due to `KV-cache` memory (Intro; “Reduced memory accesses to LLM parameters” in §2).

- Why it matters
  - Latency and cost: e.g., 175B-parameter models need many GPUs and seconds per request (Intro).
  - Offloading scenarios (serving on a single commodity GPU) are dominated by CPU↔GPU weight transfers; cutting decoding steps directly saves time and energy (§2 “Reduced memory accesses…”, §6.3).

- Prior approaches and their limits
  - Incremental decoding respects dependencies but exposes little intra-request parallelism (§1; Fig. 1a).
  - Sequence-based speculative inference: a small speculative model (`SSM`) predicts a single future sequence; the LLM verifies it in bulk [5, 22, 25, 44, 51]. Limitation: a single path infrequently matches the LLM because `SSM`s are much smaller and less accurate (§1).
  
- Positioning
  - SpecInfer expands speculation from “one sequence” to “a token tree” (many candidate continuations in parallel), then verifies all candidates in one LLM pass (§2; Fig. 1b, Fig. 2). It also introduces mechanisms so the verified tokens exactly match what the LLM would output under both greedy and stochastic sampling (Alg. 2; §4.3; Theorem 4.2).

## 3. Technical Approach
SpecInfer is a two-part system (Fig. 2):
1) a learning-based `speculator` that proposes a `token tree` of candidate continuations, and
2) a `token tree verifier` that uses the LLM to verify all candidates in parallel and append verified tokens to the output.

Key terms (paper-specific):
- `SSM` (small speculative model): a distilled/quantized/pruned model much smaller than the target LLM, used only to predict likely next tokens (§2).
- `Token tree`: a tree where each node is a candidate token and each root→node path is a speculative token sequence (Def. 3.1).
- `Tree attention`: computing attention outputs for every path in the token tree as if each were a separate sequence, but done in a fused, parallel way (Def. 4.1).
- `Topology-aware causal mask`: a mask that enforces causality across many alternative branches when computing attention in a single fused kernel (§4.2).
- `KV-cache`: cached attention keys/values for past tokens used to avoid recomputation during decoding.

Step-by-step pipeline
- Step A: Build a token tree (speculation)
  - Expansion-based tree construction: from one `SSM`, expand multiple top-k options at early steps using a preset expansion vector `⟨k1, k2, …, km⟩` (e.g., ⟨2,2,1⟩ makes four sequences; Fig. 3; §3). This caps the tree’s width and depth to control cost.
    - Motivation: with `k=5`, the chance that the LLM’s next token lies in the `SSM`’s top-k rises dramatically—from 52–57% to 96–97% for stochastic decoding and from 62–70% to 85–89% for greedy decoding (Table 1). Wider candidate pools make matches far likelier.
  - Merge-based tree construction: run multiple `SSM`s (e.g., different sizes or boost-tuned variants) in parallel and merge their outputs into one token tree (Def. 3.2; §3). They boost-tune `SSM`s with an unsupervised adaptive boosting procedure so their aggregate coverage better matches the LLM (§3 “Merge-based…”).

- Step B: Verify the tree in one LLM pass (tree-based parallel decoding)
  - Challenge: different candidate sequences have different `KV-cache` states; naively running each sequence separately repeats compute for shared prefixes and launches many small kernels (left of Fig. 4).
  - Technique 1—Shared `KV-cache` via depth-first traversal: SpecInfer traverses the tree in depth-first order, updating a single shared `KV-cache` so that, at each token, the cache holds exactly the keys/values for that token’s ancestors (§4.2, “Depth-first search to update KV-cache”; Fig. 4 center).
  - Technique 2—Topology-aware causal mask: SpecInfer fuses all candidate tokens into one batched attention computation, but uses a mask derived from the tree topology to forbid attention to tokens that are not on the same root→node path (§4.2, “Topology-aware causal mask”; right of Fig. 4). This yields the exact same attention outputs as if each path were decoded independently, but with far fewer kernel launches and no duplicate compute on shared prefixes.
  - Output: a vector `𝒪` with the LLM’s predicted next token distribution at every node in the tree (§4.3).

- Step C: Convert `𝒪` into verified tokens (quality-preserving verification)
  - Greedy decoding (VerifyGreedy in Alg. 2; §4.3): follow the tree from the root, moving to the unique child whose token matches the LLM’s top-1 at the parent. Stop at the first mismatch; then append the LLM’s chosen token at that node. This yields exactly the same next-token decision as incremental greedy decoding.
  - Stochastic decoding (VerifyStochastic in Alg. 2; §4.3): introduce multi-step speculative sampling (`MSS`).
    - Intuition: Treat each child candidate `x_s` at node `u` like a proposal from an `SSM`, and accept it with probability `min(1, P_LLM(x_s|u)/P_SSM(x_s|u))`. If rejected, subtract that candidate’s `P_SSM` mass from the LLM distribution to get a “residual” distribution and try another child. If all children are rejected, sample the next token from the residual LLM distribution directly (Fig. 5; Alg. 2 lines 29–43).
    - Guarantee: Theorem 4.2 proves that, for any history `U`, the distribution over the next token under `MSS` exactly equals the original LLM’s distribution (Eq. 6; §4.3). Theorem 4.3 shows `MSS`’s rejection probability is never worse and is often better than “naive sampling” (sample from LLM once and check membership in the tree) (§4.3).

- System design & implementation (for practical serving)
  - Runtime architecture: A request manager batches requests, runs `SSM`s with data parallelism, merges their outputs into token trees, and dispatches one LLM pass per iteration using tensor model parallelism (within a node) and pipeline parallelism (across nodes) as in Megatron-LM (§5.1; Fig. 6). Continuous batching is used between iterations (§5.1).
  - GPU kernels: A custom FasterTransformer-based attention kernel computes tree-parallel attention with the topology-aware causal mask; per-block shared-memory optimization reduces launch overhead (§5.2).
  - Overhead analysis: `SSM`s are 100–1000× smaller than the LLM, contributing <1% memory overhead each; extra memory for tree verification is negligible vs. long-sequence `KV-cache`. The added compute for verification exploits otherwise idle GPU capacity during incremental decoding (§5.3).

## 4. Key Insights and Innovations
- Token-tree speculative inference (fundamental)
  - What’s new: Move from single-path speculation to many-path token trees, constructed by either expanding an `SSM`’s top-k at early steps or merging multiple `SSM`s (§3; Fig. 2–3).
  - Why it matters: Greatly increases the chance the LLM’s next token lies among candidates (Table 1), enabling verification of multiple tokens per iteration (Table 2–3), thus reducing number of LLM decoding iterations.

- Tree-based parallel decoding with topology-aware masks (fundamental)
  - What’s new: A fused attention computation over the entire tree that reuses shared prefixes and enforces causality across branches in one kernel (§4.2; Fig. 4).
  - Why it matters: Eliminates redundant compute and many kernel launches, turning intra-request dependency into parallel work. Delivers up to 1.8× lower latency than sequence-by-sequence verification (Fig. 11) and underpins the end-to-end speedups (Fig. 7–8).

- Multi-step speculative sampling (quality-preserving advance)
  - What’s new: A branch-wise acceptance-rejection scheme that provably preserves the LLM’s original sampling distribution (Theorem 4.2) and improves acceptance vs. naive checks (Theorem 4.3; Table 3).
  - Why it matters: Enables tree-based speculation for stochastic decoding with no loss in output quality and higher verified-tokens-per-iteration.

- Practical serving architecture (incremental but important)
  - What’s new: Data-parallel `SSM`s plus tensor/pipeline-parallel LLM verification with continuous batching (§5.1), and a custom attention kernel (§5.2).
  - Why it matters: Shows the method runs at scale on multi-GPU/multi-node and offloading setups, integrating into real serving systems.

## 5. Experimental Analysis
- Setup (Sec. 6.1)
  - Models: LLaMA-7B and -65B, OPT-13B and -30B as LLMs; LLaMA-68M and OPT-125M as `SSM`s.
  - Datasets: Five prompt sets to emulate diverse requests—`CIP`, `CP`, `WebQA`, `Alpaca`, `PIQA`.
  - Hardware: Two AWS g5.12xlarge nodes (4× A10 24GB each), 100 Gbps Ethernet. Offloading uses a single A10.
  - Baselines: vLLM, HuggingFace TGI, FasterTransformer. Also two internal ablations: “SpecInfer with incremental decoding” (no speculation) and “SpecInfer with sequence-based speculation” (§6.2).

- Main end-to-end results (distributed serving; Fig. 7)
  - Quote:
    > SpecInfer outperforms incremental-decoding systems by 1.5–2.5× on single-node, multi-GPU inference and by 2.4–2.8× on multi-node, multi-GPU inference, while generating the exact same tokens as incremental decoding (§6.2; Fig. 7).
  - Observations:
    - SpecInfer with incremental decoding matches other frameworks, indicating speedups stem from the speculative/verification mechanisms rather than unrelated engineering.
    - Gains shrink at larger batch sizes because standard incremental decoding already fills the GPU; fewer idle cycles remain to verify wider trees (§6.2). Latency per request nevertheless increases with batch size across all systems.

- Offloading results (single GPU; Fig. 8)
  - Quote:
    > SpecInfer reduces per-token latency by 2.6–3.5× over FlexGen when serving OPT-13B/30B with CPU DRAM offloading (§6.3; Fig. 8).
  - Interpretation: By verifying multiple tokens per pass, SpecInfer cuts the number of decoding steps—and thus CPU↔GPU weight transfers—dominant in offloading.

- Speculation quality and tree width (Tables 1–2; Fig. 9–10)
  - Token coverage: Table 1 shows the probability that the LLM’s next token lies in the `SSM`’s top-k rises steeply with k. For stochastic decoding, top-1 is 52–57% while top-5 reaches 96–97% across datasets; greedy climbs from 62–70% to 85–89%.
  - Verified tokens per iteration: With LLaMA-7B/68M and speculation depth 8, average verified tokens increase as tree width grows (Table 2). Example (greedy, `CIP`): 2.73 (width 1) → 3.91 (width 5). Stochastic shows similar, smaller gains (e.g., `CIP`: 1.72 → 2.29).
  - Latency vs. width: On LLaMA-7B, wider trees reduce latency at small batch sizes (BS 1–2) but can hurt at large BS because verification work competes with batch compute. Width 2–3 balances best for BS≥4 (Fig. 10).

- Tree-parallel vs sequence-parallel verification (Fig. 11)
  - Quote:
    > Tree-based parallel decoding achieves on-par latency at small batch sizes and up to 1.8× lower latency at larger batch sizes compared to sequence-based decoding (§6.5; Fig. 11).
  - Reason: Fusing all branches removes duplicate compute on shared prefixes and reduces kernel overhead (§4.2).

- Stochastic decoding: MSS vs naive (Table 3)
  - Quote:
    > MSS improves average verified tokens per step by 1.26–1.28× across datasets while preserving the exact sampling distribution (Theorem 4.2; §6.6; Table 3).
  - Example: `Alpaca` increases from 1.87 (naive) to 2.38 (MSS).

- Overall assessment
  - The experiments consistently support the central claims: tree-based speculation plus tree-parallel verification reduces decoding iterations and end-to-end latency without quality loss. Ablations clarify where gains come from (tree width, verification kernel, MSS).
  - Caveats: vLLM and TGI do not support pipeline model parallelism across nodes, so the multi-node comparison uses FasterTransformer and internal baselines only (§6.2). The merge-based multi-SSM strategy is described but not the main focus of the reported numbers (§6.1, §3; extended comparisons are deferred to [28]).

## 6. Limitations and Trade-offs
- Dependence on `SSM`–LLM alignment
  - The approach gains most when `SSM` top-k predictions cover the LLM’s next tokens (Table 1). For domains where `SSM`s are poorly aligned, fewer tokens will verify, reducing speedup. The merge-based boosting procedure (§3) can help but requires preparation and is evaluated primarily in the extended version [28].

- Tree width vs. compute budget
  - Wider trees increase verification work; if the GPU is already saturated (large batch sizes), the extra work can offset benefits (Fig. 10). Choosing width 2–3 is a pragmatic trade-off at scale.

- Static expansion policy
  - The tree expansion uses a preset width vector `⟨k1,…,km⟩` (§3). Adaptive per-request/per-step expansion could yield better cost–benefit but is “an open research problem” (§3).

- Memory and implementation complexity
  - Although memory overhead is modest (<1% per `SSM`; §5.3), long contexts still dominate `KV-cache` memory; tree verification adds temporary attention buffers (§5.3).
  - The approach relies on custom kernels (FasterTransformer-based) and topology-aware masks (§5.2); integrating into all runtimes may require engineering effort.

- Baseline comparability
  - Multi-node comparisons omit vLLM/TGI because they lack pipeline model parallelism (§6.2). FasterTransformer is a strong baseline, but ecosystem differences should be considered when generalizing results.

- Proof reliance for stochastic equivalence
  - The equivalence of MSS (Theorem 4.2) and rejection-rate dominance (Theorem 4.3) are formal guarantees (proofs referenced in [28]). Correct implementation must match the theorem’s acceptance/reweighting logic.

## 7. Implications and Future Directions
- Field impact
  - Demonstrates that intra-request parallelism can be unlocked without changing the LLM or its outputs by restructuring decoding as “speculate many, verify once.” This reframes LLM serving from strictly sequential to partially parallel, complementing batching and model/tensor parallelism.

- Practical applications
  - Low-latency chat and agent systems where responsiveness matters (greatest gains at small batch sizes; Fig. 7, Fig. 10).
  - Single-GPU deployment with offloading (2.6–3.5× faster; Fig. 8), enabling larger models on commodity hardware.
  - Can be combined with orthogonal methods such as quantization/pruning, paged attention, and cache management to stack speedups (§7 “Related Work”).

- Research avenues
  - Adaptive, learned tree expansion: dynamic width/depth based on online confidence or budget (§3, §6.4).
  - Better `SSM` training/ensembling: alternative ensemble methods (voting, bagging, stacking) and task-specific alignment (§3).
  - Scheduling across requests and trees: joint optimization of batch size, tree width, and GPU occupancy under SLAs.
  - Extending tree-parallel verification to other modalities (speech, code, multimodal) and to decoding strategies like beam search (orthogonal but combinable; §7).

> Bottom line: By moving from single-path to tree-structured speculation and by verifying that tree in a single LLM pass with topology-aware attention, SpecInfer reduces both the number of decoding iterations and the per-iteration overhead—without sacrificing output quality. The design is principled (Theorem 4.2–4.3) and practical (Fig. 7–8), and it opens a path to further parallelism inside autoregressive generation.
