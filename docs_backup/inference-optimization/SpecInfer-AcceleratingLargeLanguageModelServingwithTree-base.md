# SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification

**ArXiv:** [2305.09781](https://arxiv.org/abs/2305.09781)

## 🎯 Pitch

SpecInfer introduces a novel tree-based speculative inference technique for large language model serving: instead of speculating one token sequence at a time, it uses small helper models to generate a diverse token tree of possible futures, then verifies all candidates concurrently with the full LLM in a single pass. This dramatically reduces end-to-end inference latency and memory access overhead—especially for multi-GPU and offloading scenarios—while provably preserving output quality, making real-time, cost-efficient LLM serving much more practical at scale.

---

## 1. Executive Summary
SpecInfer introduces a way to speed up large language model (LLM) serving by predicting multiple possible future token sequences with small helper models and verifying all of them against the full LLM in a single pass. Instead of speculating one sequence at a time, SpecInfer organizes many candidate sequences as a “token tree” and verifies the entire tree in parallel, cutting end-to-end latency without changing model outputs (for greedy decoding) or output distribution (for stochastic decoding).

## 2. Context and Motivation
- Problem addressed
  - Generative LLMs produce text autoregressively—one token at a time—so each token depends on all prior tokens. This “incremental decoding” limits parallelism and forces repeated memory access to the model’s parameters and cached attention states (keys and values). See Introduction and Figure 1a.
  - Memory and data movement dominate runtime and energy (especially in offloading settings where model weights are moved between CPU DRAM and GPU memory). Section 2 (Reduced memory accesses) explains that each new token normally triggers another full pass over model parameters.

- Why it matters
  - Latency: Serving a single request from models like GPT-3–scale requires seconds (Introduction). Many real-time applications need lower latency and higher throughput.
  - Cost and efficiency: Memory bandwidth and data transfer dominate both performance and energy; cutting repeated passes over parameters yields practical cost/energy savings.

- Prior approaches and shortcomings
  - Incremental decoding: Correct but inherently sequential and memory-bound (Figure 1a).
  - Sequence-based speculative inference: Use a small model to propose one candidate sequence, then have the LLM verify it in parallel. Prior work accelerates decoding but is limited because a single speculative sequence often diverges from the LLM due to a large capacity gap (Section 1; references [5, 22, 25, 44, 51]).
  - Key shortcoming: One sequence yields low “hit rate”; verification success drops exponentially with the length of the predicted sequence.

- How SpecInfer positions itself
  - It broadens speculation from one sequence to a “token tree”—many candidate sequences—and verifies them all in one LLM pass (Figure 1b, Figure 2). This raises the chance that at least one branch aligns with the LLM while keeping a single verification pass per iteration.

## 3. Technical Approach
At a high level, a small speculative model (SSM) or a set of SSMs proposes many possible next-token continuations arranged as a token tree. The LLM verifies the entire tree in parallel and accepts the longest prefix that matches the LLM’s own decoding. Algorithm 2 summarizes the full loop.

Key concepts (defined where first used here if uncommon):

- `SSM (Small Speculative Model)`: a much smaller variant of the LLM (e.g., distilled/quantized/pruned) used to cheaply predict likely next tokens (Section 2; Section 3).
- `Token tree`: a tree where each node holds one token; the path from root to a node is a candidate sequence (Definition 3.1).
- `Tree attention`: the attention computation applied to every sequence represented by nodes in the tree, implemented so all nodes are computed efficiently together (Definition 4.1).
- `Key-value (KV) cache`: cached attention keys and values for all prior tokens, reused at each decoding step to avoid recomputing attention over the entire history (Section 4.2).
- `Topology-aware causal mask`: a customized attention mask that enforces causality across many branching sequences packed together (Section 4.2).
- `Multi-step speculative sampling (MSS)`: a verification process for stochastic decoding that preserves the exact output distribution of the original LLM while maximizing acceptance (Algorithm 2, lines 24–44; Figure 5).

Step-by-step:

1) Build a token tree (the “speculator” in Figure 2; Section 3)
   - Expansion-based construction from one SSM:
     - Observation: even when an SSM’s top-1 prediction differs from the LLM’s, the LLM’s choice often appears among the SSM’s top-k (Table 1). For LLaMA-7B verified against LLaMA-68M, the chance that the LLM’s next token is in the SSM’s top-5 is 85–89% for greedy and 96–97% for stochastic decoding across five datasets (Table 1).
     - Strategy: at selected steps, branch top-k candidates to widen the tree. A simple static rule uses an “expansion configuration” vector ⟨k1, k2, …, km⟩ that sets how many children to create at each speculative step (Section 3; Figure 3 uses ⟨2,2,1⟩). The paper uses ⟨1,1,3,1,1,1,1,1⟩ in experiments (Section 6.1).
   - Merge-based construction from multiple SSMs:
     - Run several diverse SSMs (fine-tuned with an unsupervised, boosting-like process on general text so their outputs collectively cover LLM outputs) and merge their predicted sequences into one tree (Definition 3.2; Section 3, “Merge-based token tree construction”).
     - Unsupervised “boost-tuning” procedure: repeatedly fine-tune one SSM to match the LLM where it currently fails, then move to the next SSM on the remaining “hard” prompts (Section 3). Outputs of all SSMs are merged so the tree covers the union of candidate sequences.

2) Verify the token tree in one LLM pass (the “verifier”; Section 4)
   - Tree attention (Definition 4.1):
     - Conceptual definition: for each node u (sequence Su), compute standard sequence attention on Su (Equations 1–4 generalize to trees in Equation 5).
   - Efficient implementation (“tree-based parallel decoding”; Section 4.2; Figure 4):
     - Problem: naive per-sequence decoding duplicates work and creates conflicting KV caches.
     - Solution 1: Depth-first traversal to update a shared KV cache. As the tree is traversed depth-first, shared prefixes are computed once and reused (left panel vs. DFS arrows in Figure 4).
     - Solution 2: Topology-aware causal mask to batch all nodes together:
       - Pack all tokens from verified prefixes and speculative branches into one big attention computation.
       - Use a custom mask so each node only attends to its ancestors on its unique path (and not to tokens from other branches) while still doing one fused kernel launch (Figure 4, right; “Topology-aware causal mask”).
       - This preserves identical outputs to running attention separately per sequence but cuts kernel launches and redundant work on shared prefixes.

3) Decide which tokens are accepted (Algorithm 2; Section 4.3)
   - Greedy decoding (`VerifyGreedy`, lines 16–22):
     - Start at the root; while the LLM’s next-token choice 𝒪(u) matches the token on one of u’s children, walk down that child and append it to the verified output. When a mismatch occurs, append 𝒪(u) (the LLM’s token) and stop this iteration (Algorithm 2).
     - This ensures the verified output equals exactly what incremental decoding would produce, just potentially many tokens per iteration instead of one (Figure 1b bottom).
   - Stochastic decoding with MSS (`VerifyStochastic`, lines 24–44; Figure 5):
     - Goal: sample from the exact same distribution as the LLM’s ordinary stochastic decoding, while maximizing accepted speculative tokens.
     - Mechanism:
       - Consider the children of the current node u (the speculative candidates from one or more SSMs).
       - Randomly pick a child s, draw r ∼ Uniform(0,1), and accept that child with probability min(1, P_LLM(xs|u)/P_SSM(xs|u)) (Algorithm 2, line 30). This is an acceptance-rejection step that corrects any bias from the SSM toward the true LLM distribution.
       - On rejection, subtract the SSM’s mass from the LLM’s probability and renormalize the “residual” LLM distribution (line 37), then try another candidate. If all candidates are rejected, sample the next token directly from the remaining LLM distribution (lines 41–43).
       - With multiple SSMs (merge-based), repeat acceptance-rejection across SSMs in sequence (Figure 5 pipeline).
     - Guarantees:
       - Theorem 4.2: MSS samples from exactly the same distribution as the LLM’s standard stochastic decoding (Equation 6).
       - Theorem 4.3: MSS has a uniformly lower probability of “rejecting all branches” than a naive scheme that first samples from the LLM and then checks whether the choice was in the speculative tree (Section 4.3).

4) System design and runtime (Section 5; Figure 6)
   - Request manager with continuous batching (Orca-style): schedules requests at iteration granularity so new requests can start before old ones finish (Section 5.1).
   - Parallelism:
     - SSMs: data-parallel across available GPUs.
     - LLM: hybrid Megatron-LM style—tensor model parallelism within a node and pipeline model parallelism across nodes (Section 5.1).
   - GPU kernel optimization: a custom attention kernel (built on FasterTransformer) fuses tree attention for all nodes using shared memory and the topology-aware mask (Section 5.2).
   - Overheads:
     - Memory: SSMs are 100–1000× smaller; each adds <1% memory (Section 5.3). The extra token-tree state is negligible compared to standard KV caching for long contexts.
     - Compute: extra SSM runs and extra verification over speculative branches are amortized by using spare GPU compute; bottlenecks are often memory/communication, not arithmetic (Section 5.3).

## 4. Key Insights and Innovations
- Token-tree speculation instead of single-sequence speculation (Section 3; Figure 3)
  - Novelty: represent many candidate continuations as a tree and verify all at once; prior approaches examine only one speculative path.
  - Significance: dramatically raises the chance of matching the LLM while keeping only one LLM pass per iteration. Empirically, this shifts single-token verification success from ≈52–57% to ≈96–97% for stochastic decoding when using top-5 candidates (Table 1).

- Tree-based parallel decoding with a topology-aware causal mask (Section 4.2; Figure 4)
  - Novelty: batch attention across all nodes in the token tree with one fused kernel, while enforcing per-branch causality through a custom mask and a shared KV cache updated via depth-first traversal.
  - Significance: eliminates redundant compute on shared prefixes and reduces kernel launches; up to 1.8× speedup versus sequence-based decoding of multiple candidates (Figure 11).

- Multi-step speculative sampling (MSS) that is distribution-preserving and higher-acceptance (Section 4.3; Figure 5)
  - Novelty: a multi-SSM acceptance-rejection scheme over a token tree that provably preserves the LLM’s original sampling distribution (Theorem 4.2) and lowers rejection probability compared to naive schemes (Theorem 4.3).
  - Significance: unlocks tree-based speculation for stochastic decoding without any loss in generative quality. Empirically increases verified tokens per step by ≈1.26–1.28× over naive sampling (Table 3).

- Merge-based ensemble of SSMs with unsupervised boost-tuning (Section 3)
  - Novelty: adaptively fine-tune multiple small models so their union covers more of the LLM’s outputs; merge their predictions into a single token tree (Definition 3.2).
  - Significance: addresses the capacity gap of any single small model; enables broader and complementary speculation without increasing per-SSM latency (SSMs run in parallel across GPUs).

Together, these are more than incremental engineering tweaks—they change how speculation is structured and verified, creating new parallelism within a single request while preserving output fidelity.

## 5. Experimental Analysis
- Evaluation setup (Section 6.1)
  - Models:
    - LLMs: `LLaMA-7B`, `OPT-30B`, `LLaMA-65B`.
    - SSMs: `LLaMA-68M`, `OPT-125M`.
  - Datasets for prompts: CIP, CP, WebQA, Alpaca, PIQA.
  - Hardware: two AWS g5.12xlarge instances, each with 4×NVIDIA A10 24GB GPUs; 100 Gbps Ethernet.
  - Baselines:
    - Distributed inference: vLLM, HuggingFace TGI, FasterTransformer, plus SpecInfer run in two reference modes—“Incremental Decoding” and “Sequence-based Speculative Inference” (Section 6.2).
    - Offloading inference: FlexGen (Section 6.3).
  - Default tree expansion config: ⟨1,1,3,1,1,1,1,1⟩ (Section 6.1).

- Main results
  - Distributed inference speedups (Figure 7):
    - Quote
      > “SpecInfer outperforms incremental decoding systems by 1.5–2.5× for single-node, multi-GPU inference and by 2.4–2.8× for multi-node, multi-GPU inference” (Section 6.2).
    - Tree vs sequence speculation: additional 1.2–1.5× reduction in latency (Figure 7; Section 6.2).
    - Caveat:
      > “Performance improvement reduces as the batch size increases” because incremental decoding gets more parallel work, leaving less spare compute for tree verification (Section 6.2).
  - Offloading inference speedups (Figure 8):
    - Quote
      > “Compared to FlexGen, SpecInfer reduces the per-token latency by 2.6–3.5× on a single 24GB A10 GPU for OPT-13B and OPT-30B” (Section 6.3).
    - Reason: fewer LLM decoding steps → fewer CPU↔GPU weight transfers.
  - Speculation quality and width (Figures 9–10; Table 2):
    - With tree width 5 (vs width 1), average verified tokens per step increase:
      - Greedy: e.g., CIP 2.73 → 3.91; CP 2.58 → 3.69 (Table 2).
      - Stochastic: e.g., Alpaca 1.79 → 2.38 (Table 2).
    - Latency vs width:
      - For small batches (BS=1–2), larger width lowers latency (more tokens verified per iteration).
      - For BS≥4, verification cost rises due to less spare compute; width 2–3 is best (Figure 10; Section 6.4).
  - Tree-based vs sequence-based parallel decoding (Figure 11):
    - Quote
      > “On-par performance for small batch sizes and up to 1.8× faster for large batch sizes” (Section 6.5).
  - MSS vs naive sampling for stochastic decoding (Table 3):
    - Verified tokens per step improve by 1.26–1.28× across datasets (e.g., Alpaca: 1.87 → 2.38; Section 6.6).
  - Why trees help in the first place (Table 1):
    - Single-step token inclusion probability rises steeply with k: for stochastic decoding, top-5 success 96–97% vs 52–57% for top-1 across datasets (Table 1).

- Do the experiments support the claims?
  - Latency improvements are quantified across models, hardware scales, and both standard and offloading settings (Figures 7–8).
  - Verified-token gains and sensitivity to tree width are analyzed (Figures 9–10; Table 2).
  - Tree decoding efficiency and MSS benefits are ablated (Figure 11; Table 3).
  - Output quality preservation:
    - Greedy: identical sequences by construction (Algorithm 2’s `VerifyGreedy`).
    - Stochastic: theoretical equivalence (Theorem 4.2), with empirical acceptance benefits (Table 3).
  - Overall, the evaluation is consistent and multi-faceted. One practical limitation is that some baselines (e.g., vLLM, TGI) cannot run multi-node, so LLaMA-65B comparisons use FasterTransformer and SpecInfer’s incremental mode (Section 6.2).

## 6. Limitations and Trade-offs
- Diminishing returns with high batch sizes
  - As batch size grows, incremental decoding has more parallel work, leaving less spare compute for tree verification; thus speedups shrink (Section 6.2; Figure 7).

- Static tree-expansion policy
  - The expansion configuration is fixed (e.g., ⟨1,1,3,1,1,1,1,1⟩). The paper notes dynamic expansion is an open problem (Section 3). A static policy may underperform on heterogeneous prompts.

- Reliance on SSM alignment
  - Speculation quality depends on SSM(s) being reasonably aligned with the LLM; while top-k helps (Table 1) and merge-based boosting increases coverage (Section 3), extreme domain shifts or poorly tuned SSMs could reduce benefits.

- Extra components and complexity
  - Engineering complexity: custom fused kernels, topology-aware masks, DFS KV-cache management (Section 5.2; Figure 4).
  - Multi-SSM training requires unsupervised boost-tuning with LLM-generated reference outputs on a corpus (OpenWebText in the paper), which adds pre-deployment overhead (Section 3).

- Not reducing inter-GPU communication volume directly
  - In distributed serving, the amount of activation communication per decoding step remains similar; the benefit comes from reducing the number of steps (Section 5.4).

- Resource trade-offs
  - Memory: even if modest (<1% per SSM; Section 5.3), multiple SSMs and larger trees still add overhead, especially at long sequence lengths due to KV caching.
  - Verification cost rises with tree width; the best width depends on batch size and hardware (Figure 10).

- Evaluation scope
  - Quality claims for stochastic decoding rest on the theoretical guarantee (Theorem 4.2). The paper does not report downstream task metrics (e.g., BLEU, ROUGE) under stochastic decoding; it argues they must be unchanged given distributional equivalence.

## 7. Implications and Future Directions
- How this shifts the field
  - It reframes an LLM as a high-throughput verifier rather than a strict one-token-at-a-time decoder, exposing new intra-request parallelism. This is complementary to batching across requests and to model compression techniques.
  - The approach targets the true bottleneck—repeated, memory-bound parameter and KV-cache accesses—by amortizing them across multiple verified tokens per pass.

- What it enables next
  - Dynamic tree policies: Learn to expand or prune branches on the fly based on observed LLM scores, SSM confidence, or hardware load (Section 3 acknowledges this as open).
  - Better SSM ensembles: Explore other ensemble methods (voting, bagging, stacking; Section 3) and automatic diversity promotion to further increase coverage with minimal cost.
  - Integration with other accelerations: Combine with quantization/pruning of both LLM and SSMs, KV-cache compression, or efficient attention variants to compound gains.
  - Scheduling and resource allocation: Jointly optimize batch size, tree width, and pipeline/tensor parallelism for latency targets and SLA constraints.
  - Broader modalities: Extend tree-based verification to code generation with grammar constraints, structured output decoding, or multimodal models.

- Practical applications
  - Low-latency serving of chatbots, assistants, and program synthesis where responsiveness matters.
  - Cost-effective deployment on limited GPU budgets via offloading, as shown by 2.6–3.5× gains over FlexGen (Figure 8).
  - Multi-node serving of very large models where reducing the number of decoding steps translates to lower pipeline bubbles and higher overall efficiency.

> Overall takeaway: By turning single-path speculation into tree-based speculation and verifying the entire tree in one pass via topology-aware attention, SpecInfer achieves 1.5–2.8× speedups for distributed inference and 2.6–3.5× for offloading without altering outputs (Figures 7–8), with strong ablations supporting both the systems techniques (Figure 11) and the speculation algorithms (Tables 1–3).
