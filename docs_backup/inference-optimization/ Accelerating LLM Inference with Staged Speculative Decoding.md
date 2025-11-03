# Accelerating LLM Inference with Staged Speculative Decoding

**ArXiv:** [2308.04623](https://arxiv.org/abs/2308.04623)
**Authors:** Benjamin Spector, Chris Re
**Institutions:** 

## 🎯 Pitch

The paper introduces staged speculative decoding, an innovative approach that accelerates on-device language model inference by transforming speculative batches into a tree structure and adding a second speculation stage. This method significantly reduces decoding latency on a single NVIDIA RTX 4090 by up to 3.16× without sacrificing output quality, enhancing the feasibility of low-latency, personalized applications by optimizing GPU efficiency and lowering hardware demands.

---

## 1. Executive Summary (2-3 sentences)
This paper introduces staged speculative decoding, a two-part redesign of speculative decoding that accelerates small-batch, on-device inference for large language models by restructuring the speculative batch as a tree and adding an additional speculation stage. On a single NVIDIA RTX 4090 with a 762M-parameter `GPT-2-L` “oracle” model, it reports up to 3.16× lower single-batch decoding latency without degrading output quality (Section 4; Table 2).

## 2. Context and Motivation
- Problem addressed
  - Small-batch autoregressive decoding on GPUs is memory-bandwidth bound and thus slow, especially on-device. The arithmetic intensity (FLOPs per byte moved) during decoding is extremely low, so GPU utilization is poor (Section 2.2).
  - Example given: a reference PyTorch `GPT-2-L` inference requires ~1.4 GFLOPs per token, yet an RTX 4090 achieves only ~150 tokens/s, corresponding to ~0.13% compute utilization; the roofline model shows that at such low arithmetic intensity, performance is dominated by memory bandwidth (Figure 1; Section 2.2).

- Why this matters
  - Low latency is crucial for interactive applications; local inference enables personalization and privacy by keeping data on-device (Abstract; Introduction).
  - Improving small-batch performance democratizes access by lowering hardware requirements.

- Prior approaches and their gaps
  - General acceleration: quantization, FlashAttention, and batching (Section 2.3).
  - Speculative decoding (use a fast “draft” model to propose multiple next tokens, which a larger “oracle” confirms in one batched step) already boosts arithmetic intensity by converting sequential steps into a parallel batch (Section 2.3).
  - But its gains saturate quickly: the chance a draft model correctly predicts many consecutive tokens is exponentially small, so large speculative batches are often wasted when the oracle rejects early tokens (Abstract; Section 3.1).

- Positioning of this work
  - Builds directly on speculative decoding but removes its main bottlenecks by:
    - Turning the single speculative sequence into a tree of likely continuations to maximize useful verified tokens per batch (Section 3.1).
    - Applying speculation to the draft model itself (“staged” speculation) so the cost of building large speculative batches doesn’t become dominated by the draft (Section 3.2).

## 3. Technical Approach
The method comprises two orthogonal changes to how speculative decoding is constructed and executed.

- Baseline: speculative decoding (background)
  - A small `draft` model proposes k next tokens; the large `oracle` checks them in one pass.
  - If the oracle’s argmax (or sampled choice under a rejection-sampling scheme) matches the drafted token at position t, the token is accepted and the process advances; otherwise, the draft’s remaining proposed tokens are discarded and decoding “falls back” to the oracle one-step decoding (Section 2.3).
  - This trades extra draft computation for reduced oracle memory traffic by batching steps.

- Change 1 — Tree-structured speculative batches (Section 3.1)
  - Idea
    - Instead of proposing one linear sequence of k tokens, build a tree of possible sequences (branching early on most plausible tokens), then verify the entire tree with the oracle in a single batched step.
    - Rationale: moving compute from the tail of a single long path to multiple high-probability prefixes increases the expected number of correctly verified tokens, because agreement between models is more likely on early, higher-probability choices than deep in a single path.
  - Three benefits described:
    1. Higher expected accepted tokens per batch
       - By considering alternatives at earlier positions (e.g., top-2/3 options at the first uncertain step) rather than extending a single guess deeply, the batch contains more sequences that are likely to match the oracle (Section 3.1).
    2. More leaf nodes “for free”
       - The draft runs only at internal nodes; more branching raises the leaf-to-internal ratio, so a greater fraction of the batch comprises tokens not requiring additional draft computation (Section 3.1).
    3. Better parallelism for the draft
       - Work for the small draft model can be parallelized across tree levels; in the limit, the draft runs only a number of forward passes equal to tree depth, reducing its memory-bound cost (Section 3.1).
  - How it’s implemented
    - Partition attention during decoding into:
      - Cross-attention to the existing `KV cache` (the saved attention keys and values from the prompt and previously accepted tokens).
      - Self-attention within the speculative batch.
    - Construct the tree by:
      - Controlling positional embeddings for nodes so that each path in the tree corresponds to a consistent sequence position.
      - Applying a causal mask that encodes the tree’s parent-child relations, ensuring nodes only attend to valid ancestors within their path (Section 3.1).
    - Maintain a separate KV cache for the full tree-batch forward pass; after the oracle accepts tokens, append the appropriate cache slices to the main cache (Section 3.1).

- Change 2 — Staged speculation: speculating the draft itself (Section 3.2)
  - Problem
    - As one scales up speculative batch size, draft computation becomes a major cost; even a small transformer draft is still memory-bandwidth limited in small-batch settings (Section 3.2).
  - Solution
    - Add another speculation stage beneath the draft: `draft2`, which is even cheaper (e.g., an n-gram model). This `draft2` helps the draft construct its own tree-batches faster, exactly mirroring the speedup rationale applied to the oracle stage (Section 3.2).
  - Model roles in the paper’s system (Section 4)
    - `oracle`: `GPT-2-Large` (762M parameters), fine-tuned on The Stack’s Python subset.
    - `draft`: a 40M-parameter GPT-2, trained on the same data.
    - `draft2`: a Katz backoff trigram language model built from 120M tokens generated by the draft at T=1.5 (two hours of generation).
  - Why this design
    - Draft models ~15–20× smaller than the oracle often strike a good alignment-cost balance, but still cost nontrivial time for big batches (Section 3.2).
    - A tiny `draft2` (n-gram) can rapidly propose easy tokens (e.g., whitespace, common punctuation), enabling the `draft` to spend more capacity on harder parts of the tree (Figure 3 and its discussion in Section 4).

- Putting it together: staged, tree-structured speculative decoding
  - The system performs a tree-batched forward pass at the `draft2 → draft → oracle` stages, each time verifying or rejecting proposals. Accepted tokens are appended to the main KV cache; rejected branches are pruned, and decoding continues from the last accepted position (Sections 3.1–3.2).
  - For stochastic decoding (Top-k, temperature), it uses rejection sampling to preserve the original distribution: if a sampled token is not present among the speculative proposals, it falls back to oracle decoding at that step (Section 2.3 and Section 4).

- Why this approach over alternatives
  - It directly targets the true bottleneck—low arithmetic intensity—by converting serial steps into parallel verification steps (Figure 1).
  - It is orthogonal to optimizations like quantization and FlashAttention; those reduce per-step cost, while speculative batching reduces the number of sequential steps (Section 2.3).

## 4. Key Insights and Innovations
- Tree-structured speculative batches (Section 3.1)
  - What’s new: prior speculative decoding typically builds a single linear proposal; this work constructs a branching tree with controlled attention masks and positional embeddings.
  - Why it’s significant: it increases the expected number of accepted tokens per batch and improves draft efficiency by shifting compute to earlier, more probable branches.

- Staged speculation (Section 3.2)
  - What’s new: speculative decoding is applied recursively—first to accelerate the draft, then the oracle—using a very fast `draft2` like an n-gram model.
  - Why it’s significant: avoids inverting the cost structure where the draft becomes the bottleneck for large batches; unlocks larger, higher-quality speculative batches for the oracle.

- Attention partitioning and KV-cache slicing for trees (Section 3.1)
  - What’s new: a practical recipe for encoding a tree into a single batched forward pass via causal masking and careful positional embedding control, plus cache management to append only accepted paths.
  - Why it’s significant: turns a conceptual tree into a hardware-efficient operation compatible with GPU memory hierarchies.

- Empirical evidence that easy, low-entropy text is disproportionately accelerated (Figure 3; Section 4)
  - Observation: tokens such as whitespace are often accepted from `draft2` or `draft`, while semantically pivotal tokens (e.g., after “if”) tend to come from the `oracle`.
  - Significance: explains why speedups vary widely across prompts and suggests a “fixed cost per entropy” view of staged speculation’s benefits (Section 4).

## 5. Experimental Analysis
- Setup (Section 4)
  - Hardware: single quiesced NVIDIA RTX 4090.
  - Models:
    - `oracle`: 762M `GPT-2-L`, fine-tuned on The Stack (Python subset).
    - `draft`: 40M GPT-2 trained on the same data.
    - `draft2`: Katz backoff trigram from 120M tokens generated by the `draft`.
  - Dataset: HumanEval (164 Python coding prompts) (Section 4).
  - Methods compared:
    - Baseline: standard sequential decoding with the `oracle`.
    - Speculative: standard single-stage draft→oracle approach (as in prior work).
    - Staged spec.: this paper’s tree-structured, two-stage speculation.
  - Decoding regimes: deterministic (greedy) and Top-k (`k=50`, `T=1`).

- Metrics and measurements
  - Throughput: tokens per second.
  - Memory bandwidth: reported as relative consumption vs. baseline (Table 1).
  - Distribution preservation: maintained via verification and rejection sampling, so output quality is reported as preserved (Abstract; Section 2.3). No separate functional-accuracy metric (e.g., HumanEval pass@k) is reported.

- Quantitative results
  - Memory bandwidth (Table 1):
    - Deterministic: relative bandwidth 1.00 (baseline) → 0.31 (speculative) → 0.23 (staged).
    - Top-k: 1.00 → 0.48 → 0.35.
    - Interpretation: staged speculation reduces bandwidth demand substantially more than standard speculation, aligning with the roofline intuition (Figure 1).
  - Throughput (Table 2; Figure 2):
    - Deterministic: 150 (baseline) → 350 (speculative) → 475 (staged).
      - Reported average speedup ≈ 3.16× over the baseline reference implementation and ≈ 1.36× over standard speculative decoding (Section 4).
    - Top-k: 150 → 219 → 298.
      - Even with stochastic rejections, staged speculation remains ≈ 1.98× faster than baseline and ≈ 1.36× over standard speculation (Section 4).
    - Distribution across prompts (Figure 2): speedups vary greatly; some prompts approach an order-of-magnitude gain, others only ~2×.
  - Token-origin visualization (Figure 3):
    - Shows which stage generated each token for a T=1 HumanEval completion; easy tokens (e.g., indentation, whitespace) tend to come from `draft2`, more difficult but still predictable tokens from the `draft`, and crucial high-entropy tokens from the `oracle`.
    - The illustrated prompt gets ~2.5× speedup.

- Implementation overhead
  - Profiling indicates ~35% overhead attributed to Python infrastructure, suggesting a more optimized implementation or larger models could yield even better realized throughput (Section 4).

- Do the experiments support the claims?
  - For the paper’s targeted setting—single-batch, on-device decoding—the combination of bandwidth measurements (Table 1), throughput gains (Table 2), and hardware roofline analysis (Figure 1) coherently supports the claim that staged, tree-structured speculation increases arithmetic intensity and throughput without changing the model’s distribution.
  - Limits:
    - Evaluation is on one model family (GPT-2 variants) and one task suite (HumanEval prompts), with no quality metrics beyond distribution preservation.
    - No ablations for tree width/depth, or for how much each component (tree vs. staging) independently contributes, beyond the comparison to standard speculation.

## 6. Limitations and Trade-offs
- Applicability
  - Designed for small-batch, latency-sensitive decoding where GPU memory bandwidth is the bottleneck. In large-batch or multi-request server settings where compute dominates, gains may be smaller (Figure 1; Section 2.3).

- Variability in gains (Section 4; Figure 2)
  - Speedups are prompt-dependent and range widely (~2× to ~10×). Text with many low-entropy tokens (e.g., whitespace-heavy code) benefits more than dense, high-entropy text.

- Complexity and engineering cost
  - Implementing tree-structured batches requires careful attention masking, positional embedding control, and KV cache management (Section 3.1).
  - Adds system complexity (three models; cache slicing; staging logic).

- Limited evaluation breadth
  - Single GPU type; single oracle size (762M); no tests combining with quantization, FlashAttention, or other accelerations.
  - No explicit quality metrics (e.g., code correctness on HumanEval) beyond claiming distribution preservation; stochastic decoding correctness under rejection sampling is not quantified.

- Dependency on model alignment
  - Tree-batch effectiveness depends on alignment between `draft2 → draft → oracle`. If alignment is poor (e.g., domain shift), tree leaves may yield little acceptance, reducing benefits.

## 7. Implications and Future Directions
- How it changes the landscape
  - Shows that speculative decoding can scale beyond linear sequences by restructuring the speculative workload to better match GPU execution characteristics, pushing small-batch decoding closer to the memory-roofline’s efficient region (Figure 1).
  - Provides a concrete, cache-aware method to encode trees in a single batched forward pass.

- Follow-up research enabled or suggested (Section 4, “We see several paths…”)
  - Faster stochastic speculation:
    - Pre-sample multinomial CDFs to guide which draft tokens enter the tree, e.g., when a high CDF (0.99) is sampled, prioritize lower-ranked draft tokens that better match the sampled quantile.
  - Scaling up:
    - With 8-bit quantization, fit ~20B-parameter oracles on consumer GPUs in small-batch mode, enabling additional stages (e.g., `20B → 1B → 50M → n-gram`) and potentially larger speedups.
  - Better ultralight `draft2` models:
    - Replace n-grams with learned micro-models that still run in <10 μs but have stronger alignment than trigrams.

- Practical applications
  - On-device assistants, privacy-sensitive enterprise tools, offline coding aids, or embedded systems where low-latency single-stream generation is paramount.
  - Orthogonal to other inference optimizations: can be combined with quantization, efficient attention, and kernel fusion for cumulative gains.

- Open questions
  - What is the optimal tree shape (width/depth) as a function of `draft`–`oracle` divergence and entropy of the target distribution?
  - How do acceptance rates and speedups vary across domains (natural language vs. code vs. dialogues) and across model scales?
  - What is the best criterion to select which branches to include for stochastic decoding without hurting distributional fidelity?

> Overall, staged speculative decoding reframes speculative generation as a branching verification problem with multi-stage accelerators. The reported reductions in bandwidth (Table 1), higher tokens/s (Table 2), and roofline grounding (Figure 1) make a strong case for its effectiveness in the intended, latency-critical small-batch regime, while leaving ample room for broader empirical validation and system-level integration.
